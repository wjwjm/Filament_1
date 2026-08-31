#!/usr/bin/env python3
"""Generate immutable HR-4E-1 representative screens from one HR-3B run.

The optical/deposition chain remains owned by :func:`runner.run_demo`.  This
tool only validates a caller-provided, already prepared one-pulse full-Isaacs
HR-3B configuration, reads its completed HR-3B state read-only, and selects
screen copies using an objective 20%-of-peak support rule.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.confio import load_all  # noqa: E402
from KHz_filament.device import debug_backend  # noqa: E402
from KHz_filament.hr4e_timestep import (  # noqa: E402
    e1a_geometry,
    e1b_geometry_translation,
    e1b_source_grid,
    json_safe,
    repository_git_sha,
    sha256_array,
    sha256_file,
)
from KHz_filament.hr4 import HR4_Y_MAX, HR4_Y_MIN  # noqa: E402


SELECTION_FRACTION = 0.20
POST_REFERENCE_SCHEMA = "khz_filament.hr4e1.real_post_reference.v1"


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def select_hr3b_screens(
    min_delta_n: Sequence[float],
    z_positions_m: Sequence[float] | None = None,
    *,
    support_fraction: float = SELECTION_FRACTION,
) -> dict[str, Any]:
    """Select peak and strict front/rear screens from a 20% support envelope.

    ``amplitude[k] = max(-min(delta_n[k]), 0)``.  The peak is the first
    maximum.  The support is the contiguous threshold component containing the
    peak; ``front`` is the last threshold screen strictly before the peak and
    ``rear`` the first threshold screen strictly after it.  A strict side is
    recorded as absent when no such screen exists.
    """
    fraction = _finite(support_fraction, "support_fraction")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("support_fraction must be in (0, 1]")
    values = np.asarray(min_delta_n, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("min_delta_n must be a nonempty finite one-dimensional array")
    if z_positions_m is None:
        z = np.arange(values.size, dtype=np.float64)
    else:
        z = np.asarray(z_positions_m, dtype=np.float64)
        if z.shape != values.shape or not np.all(np.isfinite(z)):
            raise ValueError("z_positions_m must match min_delta_n and be finite")
    amplitude = np.maximum(-values, 0.0)
    peak_index = int(np.argmax(amplitude))
    peak_amplitude = float(amplitude[peak_index])
    base = {
        "support_fraction": fraction,
        "peak_amplitude": peak_amplitude,
        "peak_threshold": fraction * peak_amplitude,
        "support_indices": [],
        "support_z_positions_m": [],
        "peak": None,
        "front": None,
        "rear": None,
        "front_absent_reason": None,
        "rear_absent_reason": None,
    }
    if peak_amplitude <= 0.0:
        base["front_absent_reason"] = "no_negative_hr3b_support"
        base["rear_absent_reason"] = "no_negative_hr3b_support"
        return base
    threshold = fraction * peak_amplitude
    above = amplitude >= threshold
    left = peak_index
    right = peak_index
    while left > 0 and bool(above[left - 1]):
        left -= 1
    while right + 1 < values.size and bool(above[right + 1]):
        right += 1
    support_indices = list(range(left, right + 1))
    base["support_indices"] = support_indices
    base["support_z_positions_m"] = [float(z[index]) for index in support_indices]
    base["peak"] = {
        "index": peak_index,
        "z_m": float(z[peak_index]),
        "min_delta_n": float(values[peak_index]),
        "amplitude": peak_amplitude,
    }
    if left < peak_index:
        base["front"] = {
            "index": left,
            "z_m": float(z[left]),
            "min_delta_n": float(values[left]),
            "amplitude": float(amplitude[left]),
        }
    else:
        base["front_absent_reason"] = "support_has_no_strict_pre_peak_screen"
    if right > peak_index:
        base["rear"] = {
            "index": right,
            "z_m": float(z[right]),
            "min_delta_n": float(values[right]),
            "amplitude": float(amplitude[right]),
        }
    else:
        base["rear_absent_reason"] = "support_has_no_strict_post_peak_screen"
    return base


select_representative_screens = select_hr3b_screens


def _validate_authoritative_config(config_path: Path) -> dict[str, Any]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("HR-4E-1 source config must contain a mapping")
    grid, beam, prop, ion, heat, run, raman = load_all(str(config_path))
    if int(run.Npulses) != 1:
        raise ValueError("POST reference generation requires run.Npulses=1")
    if not bool(getattr(heat, "hr3b_enabled", False)):
        raise ValueError("POST reference generation requires heat.hr3b_enabled=true")
    if bool(getattr(heat, "hr3c_enabled", False)):
        raise ValueError("POST reference generation requires HR-3C disabled")
    raman_source = raw.get("raman", {})
    propagation_source = raw.get("propagation", {})
    required_raman = {
        "operator_mode": "full_isaacs_eq27",
        "operator_convention": "isaacs_eq27",
        "operator_integrator": "heun",
        "nonlinear_split_order": "strang",
    }
    for key, expected in required_raman.items():
        if str(raman_source.get(key, getattr(raman, key, ""))).lower() != expected:
            raise ValueError(f"source config is not authoritative full-Isaacs: raman.{key}")
    if not bool(propagation_source.get("use_raman_full_operator", getattr(prop, "use_raman_full_operator", False))):
        raise ValueError("source config must enable propagation.use_raman_full_operator")
    expected_grid = e1b_source_grid()
    source_grid = raw.get("grid", {})
    if not isinstance(source_grid, Mapping):
        raise ValueError("source config lacks grid mapping")
    for key in ("Nx", "Ny"):
        if int(source_grid.get(key, -1)) != int(expected_grid[key]):
            raise ValueError(f"source config grid.{key} must equal E1-B source grid")
    for config_key, proof_key in (("Lx", "Lx_m"), ("Ly", "Ly_m")):
        if not math.isclose(float(source_grid.get(config_key, float('nan'))), float(expected_grid[proof_key]), rel_tol=0.0, abs_tol=1.0e-15):
            raise ValueError(f"source config grid.{config_key} must equal E1-B source grid")
    return {
        "grid": grid,
        "beam": beam,
        "prop": prop,
        "ion": ion,
        "heat": heat,
        "run": run,
        "raman": raman,
        "raw": dict(raw),
    }


def prepare_e1b_source_config(base_config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Derive a one-pulse HR-3B E1-B source config without altering its base.

    The derived file freezes only the E1 transverse sampling and HR-3B state
    capture switches; all optical/deposition physics remains the base config.
    """
    base = Path(base_config_path)
    output = Path(output_path)
    if not base.is_file() or output.exists():
        raise FileExistsError("base config must exist and derived config target must be new")
    raw = json.loads(base.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("base config must contain a mapping")
    derived = json.loads(json.dumps(raw))
    grid = derived.setdefault("grid", {})
    heat = derived.setdefault("heat", {})
    run = derived.setdefault("run", {})
    source_grid = e1b_source_grid()
    grid.update({
        "Nx": source_grid["Nx"], "Ny": source_grid["Ny"],
        "Lx": source_grid["Lx_m"], "Ly": source_grid["Ly_m"],
    })
    heat.update({"hr3b_enabled": True, "hr3c_enabled": False})
    run["Npulses"] = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(derived, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return {
        "base_config_path": str(base.resolve()),
        "base_config_sha256": sha256_file(base),
        "derived_config_path": str(output.resolve()),
        "derived_config_sha256": sha256_file(output),
        "patch": {
            "grid": {
                "Nx": source_grid["Nx"], "Ny": source_grid["Ny"],
                "Lx": source_grid["Lx_m"], "Ly": source_grid["Ly_m"],
            },
            "heat": {"hr3b_enabled": True, "hr3c_enabled": False},
            "run": {"Npulses": 1},
        },
    }


def _source_z_positions(result: Mapping[str, Any], n_intervals: int) -> np.ndarray:
    diagnostics = result.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    edges = diagnostics.get("z_edges")
    if edges is not None:
        edge_array = np.asarray(edges, dtype=np.float64)
        if edge_array.ndim == 1 and edge_array.size == n_intervals + 1 and np.all(np.isfinite(edge_array)):
            mids = 0.5 * (edge_array[:-1] + edge_array[1:])
            if np.all(np.diff(mids) > 0.0):
                return mids
    for key in ("z_axis", "thermal_map_z_mid_m"):
        values = diagnostics.get(key)
        if values is not None:
            array = np.asarray(values, dtype=np.float64)
            if array.ndim == 1 and array.size == n_intervals and np.all(np.isfinite(array)):
                return array
    raise ValueError("source diagnostics must provide z_edges or z positions aligned to HR-3B state")


def _state_path_for_output(path: Path) -> Path:
    return path.with_suffix(".hr3b_delta_n_th.npy")


def _write_npy(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)


def generate_post_reference(
    config_path: str | Path,
    out_dir: str | Path,
    *,
    runner_output_path: str | Path | None = None,
    dtype: str = "fp64",
    runner: Callable[..., Mapping[str, Any]] | None = None,
    preparation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one prepared source chain and write only selected immutable copies."""
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(config_file)
    prepared = _validate_authoritative_config(config_file)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    source_output = Path(runner_output_path) if runner_output_path is not None else (
        destination.parent / f"{destination.name}.hr3b_source.npz"
    )
    source_output = source_output.resolve()
    destination_resolved = destination.resolve()
    try:
        source_output.relative_to(destination_resolved)
    except ValueError:
        pass
    else:
        raise ValueError("runner output must be outside selected-screen output directory")
    source_state_path = _state_path_for_output(source_output)
    if source_output.exists() or source_state_path.exists():
        raise FileExistsError("refusing to overwrite an existing source HR-3B artifact")

    if runner is None:
        from KHz_filament.runner import run_demo

        runner = run_demo
    result = runner(
        grid=prepared["grid"],
        beam=prepared["beam"],
        prop=prepared["prop"],
        ion=prepared["ion"],
        heat=prepared["heat"],
        run=prepared["run"],
        raman=prepared["raman"],
        out_path=str(source_output),
        dtype=dtype,
        return_results=True,
    )
    diagnostics = result.get("diagnostics", {}) if isinstance(result, Mapping) else {}
    if not isinstance(diagnostics, Mapping):
        raise ValueError("runner result has no diagnostics mapping")
    if not bool(diagnostics.get("hr3b_authoritative", False)):
        raise ValueError("runner did not produce an authoritative HR-3B state")
    if not bool(diagnostics.get("authoritative_hr3a_thermal_source_available", False)):
        raise ValueError("runner did not report an authoritative HR-3A source")
    if not source_output.is_file():
        raise FileNotFoundError(f"runner did not produce source output: {source_output}")
    if not source_state_path.is_file():
        raise FileNotFoundError(f"runner did not produce HR-3B state: {source_state_path}")

    source_state = np.load(source_state_path, mmap_mode="r", allow_pickle=False)
    try:
        geometry = e1a_geometry()
        expected_shape = (int(geometry["Ny"]), int(geometry["Nx"]))
        if tuple(source_state.shape[1:]) != expected_shape:
            raise ValueError(
                "HR-3B source transverse grid must match E1 inclusive nodal "
                f"grid {expected_shape}, got {tuple(source_state.shape[1:])}"
            )
        if source_state.ndim != 3 or source_state.dtype.kind != "f" or not np.all(np.isfinite(source_state)):
            raise ValueError("HR-3B source state must be finite floating [K, Ny, Nx]")
        source_shape = tuple(int(value) for value in source_state.shape)
        source_dtype = source_state.dtype.name
        source_hash_before = sha256_array(source_state)
        z_positions = _source_z_positions(result, int(source_state.shape[0]))
        min_delta_n = np.min(source_state, axis=(1, 2))
        selection = select_hr3b_screens(min_delta_n, z_positions)
        screens: dict[str, Any] = {}
        for label in ("peak", "front", "rear"):
            selected = selection[label]
            if selected is None:
                screens[label] = None
                continue
            index = int(selected["index"])
            array = np.array(source_state[index], copy=True)
            array_path = destination_resolved / f"screen_{label}_delta_n.npy"
            if array_path.exists():
                raise FileExistsError(f"refusing to overwrite selected screen: {array_path}")
            _write_npy(array_path, array)
            screens[label] = {
                **selected,
                "array_path": array_path.name,
                "array_sha256": sha256_array(array),
                "file_sha256": sha256_file(array_path),
                "shape": list(array.shape),
                "dtype": array.dtype.name,
                "velocity_initialization": "zero",
                "coordinate_relabeling": {
                    "convention": "E1 inclusive collocated/nodal",
                    "x_m": "x_min_m + i*dx_m",
                    "y_m": "y_min_m + j*dy_m",
                    "y_range_m": [float(HR4_Y_MIN), float(HR4_Y_MAX)],
                },
            }
        source_hash_after = sha256_array(source_state)
    finally:
        close = getattr(source_state, "_mmap", None)
        if close is not None:
            close.close()
    if source_hash_before != source_hash_after:
        raise RuntimeError("source HR-3B state changed while generating reference")

    backend = diagnostics.get("backend")
    if not backend:
        backend = debug_backend().get("backend", "unknown")
    manifest_path = destination_resolved / "post_reference_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite reference manifest: {manifest_path}")
    manifest = {
        "schema": POST_REFERENCE_SCHEMA,
        "config_path": str(config_file.resolve()),
        "config_sha256": sha256_file(config_file),
        "config_preparation": dict(preparation) if preparation is not None else None,
        "runner_output_path": str(source_output),
        "runner_output_sha256": sha256_file(source_output) if source_output.is_file() else None,
        "hr3b_state_path": str(source_state_path),
        "hr3b_state_file_sha256": sha256_file(source_state_path),
        "hr3b_state_sha256": source_hash_before,
        "hr3b_state_shape": list(source_shape),
        "hr3b_state_dtype": source_dtype,
        "source_z_positions_m": [float(value) for value in z_positions],
        "source_min_delta_n": [float(value) for value in min_delta_n],
        "selection_rule": "peak=max(-min(delta_n)); contiguous >=20% peak support; front/rear are strict pre/post-peak endpoints",
        "support_fraction": SELECTION_FRACTION,
        "selection": selection,
        "screens": screens,
        "source_grid": e1b_source_grid(),
        "target_grid": e1a_geometry(),
        "geometry_translation": e1b_geometry_translation(),
        "n0": float(prepared["beam"].n0),
        "source_dtype": source_dtype,
        "source_backend": backend,
        "source_git_sha": repository_git_sha(),
        "velocity_initialization": "zero",
        "coordinate_convention": "E1 inclusive collocated/nodal endpoints sampled",
        "source_untouched": True,
        "source_hr3b_authoritative": True,
        "source_hr3a_authoritative": True,
        "backend": backend,
        "dtype_requested": dtype,
    }
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(json_safe(manifest), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


generate_hr4e1_post_reference = generate_post_reference


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="already prepared exact E1-B source config")
    source.add_argument("--base-config", type=Path, help="derive exact E1-B HR-3B source config here")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--runner-output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("fp32", "fp64"), default="fp64")
    args = parser.parse_args(argv)
    preparation = None
    config_path = args.config
    if args.base_config is not None:
        config_path = args.out_dir / "real_post_input_config.json"
        preparation = prepare_e1b_source_config(args.base_config, config_path)
    manifest = generate_post_reference(
        config_path, args.out_dir, runner_output_path=args.runner_output, dtype=args.dtype,
        preparation=preparation,
    )
    print(json.dumps({"manifest": str(args.out_dir / "post_reference_manifest.json"), "screens": manifest["screens"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
