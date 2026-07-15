#!/usr/bin/env python3
"""Direct linear-vacuum FT90 focus scan using the production transverse optics.

This is intentionally a separate 2D driver.  It reuses the project's sampled
FT90 intensity profile, discrete P0 normalization, thin-lens phase convention,
FFT k-space grid, and paraxial angular-spectrum propagator.  It does not call
the nonlinear propagation engine, so no nonlinear effect can be enabled by a
small input-power approximation or an implicit default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


def _get(mapping: dict[str, Any], dotted: str) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        value = value[part]
    return value


def _require_equal(config: dict[str, Any], spec: dict[str, Any]) -> None:
    for dotted, expected in spec["required_invariants"].items():
        actual = _get(config, dotted)
        if actual != expected:
            raise ValueError(f"vacuum-focus invariant failed: {dotted}={actual!r}, expected {expected!r}")
    disabled = spec["nonlinear_terms"]
    if any(bool(value) for value in disabled.values()):
        raise ValueError("all nonlinear/vacuum-excluded terms must be explicitly false")


def _parabolic_vertex(z: "np.ndarray", values: "np.ndarray") -> dict[str, float]:
    import numpy as np

    index = int(np.argmax(values))
    if index == 0 or index == len(values) - 1:
        raise ValueError("intensity maximum is at the sampled z-boundary; extend the scan interval")
    coeff = np.polyfit(z[index - 1:index + 2], values[index - 1:index + 2], deg=2)
    a, b, c = (float(item) for item in coeff)
    if not a < 0.0:
        raise ValueError("three-point focus interpolation is not concave; refine the z sampling")
    vertex = -b / (2.0 * a)
    if not float(z[index - 1]) <= vertex <= float(z[index + 1]):
        raise ValueError("parabolic focus vertex lies outside adjacent sampled points")
    return {
        "discrete_index": index,
        "z_discrete_m": float(z[index]),
        "z_parabolic_m": vertex,
        "I_parabolic_W_m2": float(a * vertex * vertex + b * vertex + c),
        "curvature_W_m2_per_m2": a,
    }


def _boundary_power_fraction(intensity_xy: "Any", dx: float, dy: float) -> float:
    import numpy as np

    array = np.asarray(intensity_xy, dtype=np.float64)
    total = float(array.sum()) * dx * dy
    if total <= 0.0:
        return float("nan")
    boundary = float(array[0, :].sum() + array[-1, :].sum() + array[1:-1, 0].sum() + array[1:-1, -1].sum())
    return boundary * dx * dy / total


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a direct 2D FT90 linear-vacuum focus scan")
    parser.add_argument("--config", required=True, help="vacuum FT90 JSON configuration")
    parser.add_argument("--stage-spec", required=True, help="vacuum-focus stage JSON")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gpu", action="store_true", help="select the project CuPy backend before importing KHz_filament")
    parser.add_argument("--z-min-m", type=float)
    parser.add_argument("--z-max-m", type=float)
    parser.add_argument("--dz-output-m", type=float)
    args = parser.parse_args()

    if args.gpu:
        os.environ["UPPE_USE_GPU"] = "1"
    else:
        os.environ.pop("UPPE_USE_GPU", None)

    # ``python tools/run_...py`` makes tools/ the initial import location;
    # explicitly retain the repository's normal KHz_filament import root.
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import numpy as np

    from KHz_filament.config import BeamConfig
    from KHz_filament.constants import c0, eps0
    from KHz_filament.device import to_cpu, xp
    from KHz_filament.grids import make_axes
    from KHz_filament.linear import lin_propagator
    from KHz_filament.runner import build_transverse_input_field

    config_path = Path(args.config).resolve()
    spec_path = Path(args.stage_spec).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    _require_equal(config, spec)

    grid = config["grid"]
    beam_data = config["beam"]
    propagation = config["propagation"]
    z_min = float(args.z_min_m if args.z_min_m is not None else propagation["z_min_m"])
    z_max = float(args.z_max_m if args.z_max_m is not None else propagation["z_max_m"])
    dz = float(args.dz_output_m if args.dz_output_m is not None else propagation["dz_output_m"])
    focus_m = float(propagation["geometric_focus_m"])
    if not (0.0 < z_min < z_max and dz > 0.0):
        raise ValueError("invalid axial scan limits")
    z = np.arange(z_min, z_max + 0.5 * dz, dz, dtype=np.float64)
    if z[-1] > z_max + 1e-12:
        z = z[:-1]

    axes = make_axes(int(grid["Nx"]), int(grid["Ny"]), int(grid["Nt"]), float(grid["Lx"]), float(grid["Ly"]), float(grid["Twin"]))
    ctype = xp.complex64
    beam = BeamConfig(**beam_data)
    field_txy, input_diag = build_transverse_input_field(axes, beam, ctype)
    center_t = int(xp.argmin(xp.abs(axes.t)))
    field_xy = field_txy[center_t].copy()

    # Identical thin-lens sign/convention to KHz_filament.runner.run_demo.
    k0 = float(beam.n0) * (2.0 * np.pi * c0 / float(beam.lam0)) / c0
    X, Y = xp.meshgrid(axes.x, axes.y, indexing="xy")
    phase_lens = -(k0 / (2.0 * float(beam.focal_length))) * (X ** 2 + Y ** 2)
    field_xy *= xp.exp(xp.array(1j, dtype=ctype) * phase_lens.astype(xp.float32))
    field_k = xp.fft.fft2(field_xy)

    prefactor = 0.5 * eps0 * c0 * float(beam.n0)
    x_cpu = np.asarray(to_cpu(axes.x), dtype=np.float64)
    y_cpu = np.asarray(to_cpu(axes.y), dtype=np.float64)
    X_cpu, Y_cpu = np.meshgrid(x_cpu, y_cpu, indexing="xy")
    r2_cpu = X_cpu ** 2 + Y_cpu ** 2
    ix0 = int(np.argmin(np.abs(x_cpu)))
    iy0 = int(np.argmin(np.abs(y_cpu)))

    rows: list[dict[str, float]] = []
    focus_field = None
    best_index = -1
    best_intensity = -np.inf
    for index, z_value in enumerate(z):
        # Direct-from-lens propagation: no accumulated axial stepping error.
        propagator = lin_propagator(axes.kperp2, k0, float(z_value), ctype=ctype)
        propagated = xp.fft.ifft2(field_k * propagator)
        intensity_xy = prefactor * xp.abs(propagated) ** 2
        intensity_cpu = np.asarray(to_cpu(intensity_xy), dtype=np.float64)
        total_power = float(intensity_cpu.sum()) * axes.dx * axes.dy
        max_flat = int(np.argmax(intensity_cpu))
        iy_max, ix_max = np.unravel_index(max_flat, intensity_cpu.shape)
        second_moment = float(np.sqrt(2.0 * float((r2_cpu * intensity_cpu).sum()) / max(float(intensity_cpu.sum()), 1e-300)))
        i_max = float(intensity_cpu[iy_max, ix_max])
        rows.append({
            "z_m": float(z_value),
            "x_focus_cm": 100.0 * (float(z_value) - focus_m),
            "I_onaxis_W_m2": float(intensity_cpu[iy0, ix0]),
            "I_max_W_m2": i_max,
            "I_max_x_m": float(x_cpu[ix_max]),
            "I_max_y_m": float(y_cpu[iy_max]),
            "w_second_moment_m": second_moment,
            "boundary_power_fraction": _boundary_power_fraction(intensity_cpu, axes.dx, axes.dy),
            "total_power_W": total_power,
        })
        if i_max > best_intensity:
            best_intensity = i_max
            best_index = index
            focus_field = intensity_cpu

    if focus_field is None:
        raise RuntimeError("no axial samples were evaluated")
    powers = np.asarray([row["total_power_W"] for row in rows], dtype=float)
    p0 = float(input_diag["input_peak_power_W"])
    for row in rows:
        row["power_relative_drift"] = (row["total_power_W"] - p0) / p0

    peak = _parabolic_vertex(z, np.asarray([row["I_max_W_m2"] for row in rows], dtype=float))
    peak["x_focus_cm"] = 100.0 * (peak["z_parabolic_m"] - focus_m)
    peak["sampling_half_step_uncertainty_cm"] = 50.0 * dz

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "vacuum_focus_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out_dir / "vacuum_focus_plane.npz", x_m=x_cpu, y_m=y_cpu, intensity_W_m2=focus_field, z_discrete_m=z[best_index])

    maximum_drift = float(np.max(np.abs((powers - p0) / p0)))
    quality = spec["quality_gates"]
    summary = {
        "stage_id": spec["stage_id"],
        "coordinate_definition": spec["coordinate_definition"],
        "config_path": str(config_path),
        "stage_spec_path": str(spec_path),
        "backend": "cupy" if getattr(xp, "__name__", "numpy") == "cupy" else "numpy",
        "geometry": {"geometric_focus_m": focus_m, "z_min_m": z_min, "z_max_m": z_max, "dz_output_m": dz, "samples": int(len(z))},
        "input": {
            "discrete_peak_power_W": p0,
            "target_peak_power_W": float(beam.P0_peak),
            "relative_power_error": abs(p0 - float(beam.P0_peak)) / float(beam.P0_peak),
            "input_boundary_I_fraction": float(input_diag["input_boundary_I_fraction"]),
            "profile_type": str(input_diag["input_profile_type"]),
            "profile_radius_m": float(input_diag["input_profile_radius_m"]),
            "edge_start_fraction": float(input_diag["input_profile_edge_start_fraction"]),
        },
        "focus_peak": peak,
        "power_conservation": {
            "minimum_total_power_W": float(powers.min()),
            "maximum_total_power_W": float(powers.max()),
            "maximum_relative_drift": maximum_drift,
        },
        "quality_checks": {
            "input_power_ok": abs(p0 - float(beam.P0_peak)) / float(beam.P0_peak) <= float(quality["maximum_input_power_relative_error"]),
            "power_drift_ok": maximum_drift <= float(quality["maximum_power_drift_relative"]),
            "focus_uncertainty_ok": peak["sampling_half_step_uncertainty_cm"] <= float(quality["maximum_peak_location_uncertainty_cm"]),
        },
        "artifacts": {"metrics_csv": str(metrics_path), "focus_plane_npz": str(out_dir / "vacuum_focus_plane.npz")},
    }
    (out_dir / "vacuum_focus_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not all(summary["quality_checks"].values()):
        raise RuntimeError(f"vacuum-focus quality gate failed: {summary['quality_checks']}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
