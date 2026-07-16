#!/usr/bin/env python3
"""One manifest-defined FT90-profile linear-vacuum focus case.

This intentionally bypasses the nonlinear propagation engine: it samples the
project's transverse grid and cosine FT90 function, applies the identical
thin-lens phase convention, and evaluates every axial plane directly from the
lens-plane FFT.  Consequently no axial stepping accumulation or implicit
nonlinear term can affect the measured focus location.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _parabolic_vertex(z: np.ndarray, value: np.ndarray) -> dict[str, float]:
    index = int(np.argmax(value))
    if index == 0 or index == value.size - 1:
        raise RuntimeError("focus maximum reached a scan boundary")
    a, b, c = (float(item) for item in np.polyfit(z[index - 1:index + 2], value[index - 1:index + 2], 2))
    if not a < 0.0:
        raise RuntimeError("three-point focus fit is not concave")
    z_peak = -b / (2.0 * a)
    if not z[index - 1] <= z_peak <= z[index + 1]:
        raise RuntimeError("parabolic focus vertex is outside its adjacent samples")
    return {"z_discrete_m": float(z[index]), "z_parabolic_m": z_peak, "I_parabolic_W_m2": float(a * z_peak**2 + b * z_peak + c), "curvature_W_m2_per_m2": a}


def _encircled_radius(radius: np.ndarray, intensity: np.ndarray, fraction: float) -> float:
    order = np.argsort(radius.ravel())
    cumulative = np.cumsum(intensity.ravel()[order])
    if cumulative.size == 0 or cumulative[-1] <= 0:
        return float("nan")
    return float(radius.ravel()[order][min(int(np.searchsorted(cumulative, fraction * cumulative[-1])), cumulative.size - 1)])


def _boundary_power_fraction(intensity: np.ndarray, dx: float, dy: float) -> float:
    total = float(intensity.sum()) * dx * dy
    edge = float(intensity[0].sum() + intensity[-1].sum() + intensity[1:-1, 0].sum() + intensity[1:-1, -1].sum()) * dx * dy
    return edge / total if total > 0 else float("nan")


def _half_crossing(x: np.ndarray, y: np.ndarray, index_left: int, index_right: int, level: float) -> float:
    """Linearly interpolate one bracketed level crossing."""
    x0, x1 = float(x[index_left]), float(x[index_right])
    y0, y1 = float(y[index_left]), float(y[index_right])
    if y1 == y0:
        return 0.5 * (x0 + x1)
    return x0 + (level - y0) * (x1 - x0) / (y1 - y0)


def _axial_shape_metrics(x: np.ndarray, signal: np.ndarray) -> dict[str, float]:
    """FWHM of the contiguous main lobe plus the requested prefocus metric."""
    index = int(np.argmax(signal)); peak = float(signal[index]); half = 0.5 * peak
    left = index
    while left > 0 and signal[left - 1] >= half:
        left -= 1
    right = index
    while right < signal.size - 1 and signal[right + 1] >= half:
        right += 1
    if left == 0:
        x_left = float("nan")
    else:
        x_left = _half_crossing(x, signal, left - 1, left, half)
    if right == signal.size - 1:
        x_right = float("nan")
    else:
        x_right = _half_crossing(x, signal, right, right + 1, half)
    prefocus = signal[x < x_left] if np.isfinite(x_left) else np.asarray([], dtype=float)
    return {
        "peak_intensity_W_m2": peak,
        "x_half_left_cm": x_left,
        "x_half_right_cm": x_right,
        "axial_fwhm_cm": float(x_right - x_left) if np.isfinite(x_left) and np.isfinite(x_right) else float("nan"),
        "prefocus_sidelobe_max_ratio": float(prefocus.max() / peak) if prefocus.size else 0.0,
    }


def _profile_intensity(xp: Any, x: Any, y: Any, profile: dict[str, Any]) -> Any:
    """Return sampled intensity gI, delegating cosine profiles to production."""
    from KHz_filament.utils import transverse_intensity_profile

    kind = profile["kind"]
    zero_radius = float(profile["zero_radius_m"])
    flat_radius = float(profile["flat_radius_m"])
    if kind == "cosine":
        # This calls exactly the production FT90 intensity profile function.
        return transverse_intensity_profile(
            x, y,
            {"type": "flat_top_cosine", "radius_m": zero_radius, "edge_start_fraction": flat_radius / zero_radius},
            fallback_w0=zero_radius,
        )
    if kind == "hard":
        X, Y = xp.meshgrid(x, y, indexing="xy")
        return xp.where(X**2 + Y**2 <= zero_radius**2, 1.0, 0.0)
    raise ValueError(f"unknown profile kind: {kind}")


def _input_metrics(gI: np.ndarray, x: np.ndarray, y: np.ndarray, dx: float, dy: float, prefactor: float, peak_power: float) -> tuple[float, dict[str, Any]]:
    X, Y = np.meshgrid(x, y, indexing="xy")
    radius = np.sqrt(X**2 + Y**2)
    area = float(gI.sum()) * dx * dy
    amplitude = math.sqrt(peak_power / (prefactor * area))
    intensity = prefactor * amplitude**2 * gI
    r2_mean = float((radius**2 * intensity).sum() / intensity.sum())
    return amplitude, {
        "discrete_peak_power_W": float(intensity.sum()) * dx * dy,
        "peak_intensity_W_m2": float(intensity.max()),
        "effective_area_m2": area,
        "r50_m": _encircled_radius(radius, intensity, 0.5),
        "r90_m": _encircled_radius(radius, intensity, 0.9),
        "second_moment_radius_m": math.sqrt(2.0 * r2_mean),
        "boundary_intensity_fraction": float(np.max(np.r_[intensity[0], intensity[-1], intensity[1:-1, 0], intensity[1:-1, -1]]) / max(float(intensity.max()), 1e-300)),
        "radial_x_m": x.tolist(),
        "radial_I_W_m2": intensity[int(np.argmin(np.abs(y)))].tolist(),
    }


def _fresnel_onaxis_crosscheck(profile: dict[str, Any], common: dict[str, Any], z: np.ndarray, amplitude: float, n_r: int) -> dict[str, float]:
    """Continuous axisymmetric Fresnel check for any P1--P6 input profile."""
    from scipy.integrate import trapezoid

    if n_r < 16385:
        raise ValueError("continuous Fresnel crosscheck requires n_r >= 16385")
    r = np.linspace(0.0, float(profile["zero_radius_m"]), n_r)
    if profile["kind"] == "hard":
        g = np.ones_like(r)
    else:
        flat, zero = float(profile["flat_radius_m"]), float(profile["zero_radius_m"])
        g = np.where(r <= flat, 1.0, 0.5 * (1.0 + np.cos(np.pi * (r - flat) / (zero - flat))))
    amp = float(amplitude) * np.sqrt(np.clip(g, 0.0, None))
    k0 = 2.0 * np.pi / float(common["wavelength_m"])
    f = float(common["focal_length_m"])
    signal = np.empty_like(z)
    for i, zz in enumerate(z):
        phase = k0 * r**2 * (1.0 / zz - 1.0 / f) / 2.0
        signal[i] = abs(trapezoid(amp * np.exp(1j * phase) * r, r)) ** 2 / (zz**2)
    peak = _parabolic_vertex(z, signal)
    peak["x_focus_cm"] = 100.0 * (peak["z_parabolic_m"] - f)
    peak["radial_samples"] = int(n_r)
    return peak


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="generated individual profile case JSON")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--save-focus-plane", action="store_true")
    parser.add_argument("--fresnel-crosscheck", action="store_true")
    args = parser.parse_args()
    if args.gpu:
        os.environ["UPPE_USE_GPU"] = "1"
    else:
        os.environ.pop("UPPE_USE_GPU", None)
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from KHz_filament.constants import c0, eps0
    from KHz_filament.device import to_cpu, xp
    from KHz_filament.grids import make_axes
    from KHz_filament.linear import lin_propagator

    case = json.loads(Path(args.case).read_text(encoding="utf-8"))
    common, profile, grid = case["common"], case["profile"], case["grid"]
    axes = make_axes(int(grid["Nx"]), int(grid["Ny"]), 2, float(grid["Lx_m"]), float(grid["Ly_m"]), 1.0)
    gI = _profile_intensity(xp, axes.x, axes.y, profile)
    gI_cpu = np.asarray(to_cpu(gI), dtype=np.float64)
    x = np.asarray(to_cpu(axes.x), dtype=np.float64); y = np.asarray(to_cpu(axes.y), dtype=np.float64)
    prefactor = 0.5 * eps0 * c0 * float(common["refractive_index"])
    amplitude, input_metrics = _input_metrics(gI_cpu, x, y, axes.dx, axes.dy, prefactor, float(common["peak_power_W"]))
    field = (amplitude * xp.sqrt(gI)).astype(xp.complex64)
    k0 = 2.0 * np.pi / float(common["wavelength_m"])
    X, Y = xp.meshgrid(axes.x, axes.y, indexing="xy")
    field *= xp.exp(xp.array(-1j, dtype=field.dtype) * (k0 / (2.0 * float(common["focal_length_m"]))) * (X**2 + Y**2))
    field_k = xp.fft.fft2(field)
    z = np.arange(float(common["z_min_m"]), float(common["z_max_m"]) + 0.5 * float(common["dz_output_m"]), float(common["dz_output_m"]), dtype=float)
    z = z[z <= float(common["z_max_m"]) + 1e-12]
    ix0, iy0 = int(np.argmin(abs(x))), int(np.argmin(abs(y)))
    X_cpu, Y_cpu = np.meshgrid(x, y, indexing="xy"); r2 = X_cpu**2 + Y_cpu**2
    rows: list[dict[str, float]] = []; best = (-np.inf, None, -1)
    for index, zz in enumerate(z):
        propagated = xp.fft.ifft2(field_k * lin_propagator(axes.kperp2, k0, float(zz), ctype=field.dtype))
        intensity = np.asarray(to_cpu(prefactor * xp.abs(propagated)**2), dtype=np.float64)
        maximum = int(np.argmax(intensity)); iy, ix = np.unravel_index(maximum, intensity.shape)
        imax = float(intensity[iy, ix]); total = float(intensity.sum()) * axes.dx * axes.dy
        rows.append({"z_m": float(zz), "x_focus_cm": 100.0 * (float(zz) - float(common["focal_length_m"])), "I_onaxis_W_m2": float(intensity[iy0, ix0]), "I_max_W_m2": imax, "I_max_x_m": float(x[ix]), "I_max_y_m": float(y[iy]), "w_second_moment_m": math.sqrt(2.0 * float((r2 * intensity).sum()) / float(intensity.sum())), "boundary_power_fraction": _boundary_power_fraction(intensity, axes.dx, axes.dy), "total_power_W": total})
        if imax > best[0]: best = (imax, intensity, index)
    x_focus = np.asarray([row["x_focus_cm"] for row in rows], dtype=float)
    imax_signal = np.asarray([row["I_max_W_m2"] for row in rows], dtype=float)
    onaxis_signal = np.asarray([row["I_onaxis_W_m2"] for row in rows], dtype=float)
    peak = _parabolic_vertex(z, imax_signal)
    peak["x_focus_cm"] = 100.0 * (peak["z_parabolic_m"] - float(common["focal_length_m"]))
    peak["sampling_half_step_uncertainty_cm"] = 50.0 * float(common["dz_output_m"])
    peak_onaxis = _parabolic_vertex(z, onaxis_signal)
    peak_onaxis["x_focus_cm"] = 100.0 * (peak_onaxis["z_parabolic_m"] - float(common["focal_length_m"]))
    peak_onaxis["sampling_half_step_uncertainty_cm"] = 50.0 * float(common["dz_output_m"])
    shape_imax = _axial_shape_metrics(x_focus, imax_signal)
    shape_onaxis = _axial_shape_metrics(x_focus, onaxis_signal)
    power = np.asarray([row["total_power_W"] for row in rows]); drift = np.abs((power - input_metrics["discrete_peak_power_W"]) / input_metrics["discrete_peak_power_W"])
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "vacuum_focus_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if args.save_focus_plane:
        np.savez_compressed(out / "focus_plane.npz", x_m=x, y_m=y, intensity_W_m2=best[1], z_m=z[best[2]])
    summary: dict[str, Any] = {
        "case_id": case["case_id"], "label": case["label"], "coordinate_definition": case["coordinate_definition"],
        "backend": getattr(xp, "__name__", "numpy"), "common": common, "grid": grid, "profile": profile,
        "input": input_metrics, "focus_peak": peak, "focus_peak_onaxis": peak_onaxis,
        "axial_shape": {"imax": shape_imax, "onaxis": shape_onaxis},
        "focus_metrics": {
            "x_vac_fft_imax_cm": peak["x_focus_cm"],
            "x_vac_fft_onaxis_cm": peak_onaxis["x_focus_cm"],
            "fft_imax_minus_onaxis_cm": peak["x_focus_cm"] - peak_onaxis["x_focus_cm"],
            "axial_fwhm_imax_cm": shape_imax["axial_fwhm_cm"],
            "axial_fwhm_onaxis_cm": shape_onaxis["axial_fwhm_cm"],
            "prefocus_sidelobe_max_ratio_imax": shape_imax["prefocus_sidelobe_max_ratio"],
            "prefocus_sidelobe_max_ratio_onaxis": shape_onaxis["prefocus_sidelobe_max_ratio"],
        },
        "power_conservation": {"maximum_relative_drift": float(drift.max()), "maximum_boundary_power_fraction": float(max(row["boundary_power_fraction"] for row in rows))},
    }
    if args.fresnel_crosscheck:
        fresnel = _fresnel_onaxis_crosscheck(profile, common, z, amplitude, int(case.get("fresnel_radial_samples", 16385)))
        summary["fresnel_onaxis_crosscheck"] = fresnel
        summary["focus_metrics"].update({
            "x_vac_fresnel_onaxis_cm": fresnel["x_focus_cm"],
            "fft_imax_minus_fresnel_cm": peak["x_focus_cm"] - fresnel["x_focus_cm"],
            "fft_onaxis_minus_fresnel_cm": peak_onaxis["x_focus_cm"] - fresnel["x_focus_cm"],
        })
    (out / "vacuum_focus_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_id": case["case_id"], "x_focus_cm": peak["x_focus_cm"], "power_drift": float(drift.max())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
