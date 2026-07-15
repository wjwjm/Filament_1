#!/usr/bin/env python3
"""Create headless PNG diagnostics directly from a KHz-filament ``.npz`` output.

This is deliberately independent of MATLAB so it can run as the post-processing
step of a Slurm job.  The plotting conventions mirror
``matlab/diagnose_khzfil_out.m``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


FIGURE_SPECS = {
    "intensity": "01_intensity_vs_z.png",
    "plasma": "02_plasma_density_vs_z.png",
    "beam": "03_beam_radius_vs_z.png",
    "energy": "04_energy_vs_z.png",
    "fwhm": "05_fwhm_vs_z.png",
    "rho_tz": "06_rho_onaxis_t_z.png",
}

_ALIASES = {
    "intensity": "intensity", "i": "intensity", "fig1": "intensity", "figure1": "intensity",
    "plasma": "plasma", "rho": "plasma", "density": "plasma", "fig2": "plasma", "figure2": "plasma",
    "beam": "beam", "w_mom": "beam", "radius": "beam", "fig3": "beam", "figure3": "beam",
    "energy": "energy", "u": "energy", "fig4": "energy", "figure4": "energy",
    "fwhm": "fwhm", "width": "fwhm", "fig5": "fwhm", "figure5": "fwhm",
    "rho_tz": "rho_tz", "rho-onaxis-t": "rho_tz", "fig6": "rho_tz", "figure6": "rho_tz",
}

_Z_SERIES_FIELDS = (
    "U_z", "I_max_z", "I_onaxis_max_z", "I_center_t0_z", "I_peak_q99_z",
    "rho_onaxis_max_z", "rho_max_z", "rho_peak_q99_z", "w_mom_z",
    "fwhm_plasma_z", "fwhm_fluence_z",
)


def _json_value(value: Any) -> Any:
    """Convert NumPy scalars and non-finite values into JSON-safe data."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _as_vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim > 2 or (array.ndim == 2 and 1 not in array.shape):
        raise ValueError(f"{name} must be a one-dimensional array, got shape {array.shape}")
    try:
        return np.asarray(array, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _positive_for_log(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    result[~np.isfinite(result) | (result <= 0)] = np.nan
    return result


def _finite_peak(values: np.ndarray) -> tuple[float | None, int | None]:
    valid = np.isfinite(values)
    if not np.any(valid):
        return None, None
    index = int(np.nanargmax(np.where(valid, values, np.nan)))
    return float(values[index]), index


def _finite_min(values: np.ndarray) -> tuple[float | None, int | None]:
    valid = np.isfinite(values)
    if not np.any(valid):
        return None, None
    index = int(np.nanargmin(np.where(valid, values, np.nan)))
    return float(values[index]), index


def _normalise(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.full_like(values, np.nan, dtype=float)
    scale = np.nanmax(np.where(valid, values, np.nan))
    if not np.isfinite(scale) or scale == 0:
        return np.full_like(values, np.nan, dtype=float)
    return values / scale


def _parse_selection(selected_figures: str | Iterable[str]) -> set[str]:
    if isinstance(selected_figures, str):
        raw_items = selected_figures.replace(";", ",").split(",")
    else:
        raw_items = list(selected_figures)
    keys = [str(item).strip().lower() for item in raw_items if str(item).strip()]
    if not keys or "all" in keys:
        return set(FIGURE_SPECS)
    selected: set[str] = set()
    unknown: list[str] = []
    for key in keys:
        mapped = _ALIASES.get(key)
        if mapped is None:
            unknown.append(key)
        else:
            selected.add(mapped)
    if unknown:
        raise ValueError(f"unknown figure selection: {', '.join(unknown)}")
    return selected


def _sanity_warnings(series: dict[str, np.ndarray], z: np.ndarray) -> list[str]:
    warnings: list[str] = []
    if np.any(~np.isfinite(z)) or np.any(np.diff(z) <= 0):
        warnings.append("z_axis is not strictly increasing and finite.")

    energy = series.get("U_z")
    if energy is not None:
        finite = energy[np.isfinite(energy)]
        if finite.size >= 2 and finite[0] != 0:
            drift = (finite[-1] - finite[0]) / finite[0]
            if drift > 0.10:
                warnings.append("U_z grows by more than 10% without a gain mechanism.")

    intensity = series.get("I_max_z")
    if intensity is not None:
        positive = _positive_for_log(intensity)
        pairs = positive[1:] / positive[:-1]
        if np.any(pairs > 10):
            warnings.append("I_max_z has an adjacent >10x increase.")

    density = series.get("rho_onaxis_max_z")
    if density is not None and np.any(density[np.isfinite(density)] > 1e25):
        warnings.append("rho_onaxis_max_z exceeds the ~1e25 m^-3 air neutral-density scale.")

    for name in ("fwhm_plasma_z", "fwhm_fluence_z"):
        values = series.get(name)
        if values is not None and np.any(~np.isfinite(values) | (values <= 0)):
            warnings.append(f"{name} contains non-finite or non-positive values.")
    return warnings


def _write_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _quality_observations(series: dict[str, np.ndarray], z: np.ndarray) -> dict[str, Any]:
    """Return numerical facts used by stage quality gates."""
    observations: dict[str, Any] = {
        "z_strictly_increasing": bool(np.all(np.isfinite(z)) and np.all(np.diff(z) > 0)),
        "max_energy_growth_fraction": None,
        "max_adjacent_intensity_growth": None,
        "max_electron_density_m3": None,
        "fwhm_all_positive_finite": True,
    }
    if "U_z" in series:
        valid = series["U_z"][np.isfinite(series["U_z"])]
        if valid.size and valid[0] != 0:
            observations["max_energy_growth_fraction"] = float(np.max(valid / valid[0] - 1.0))
    if "I_max_z" in series:
        values = _positive_for_log(series["I_max_z"])
        ratios = values[1:] / values[:-1]
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size:
            observations["max_adjacent_intensity_growth"] = float(np.max(ratios))
    if "rho_onaxis_max_z" in series:
        values = series["rho_onaxis_max_z"]
        values = values[np.isfinite(values)]
        if values.size:
            observations["max_electron_density_m3"] = float(np.max(values))
    for name in ("fwhm_plasma_z", "fwhm_fluence_z"):
        if name in series and np.any(~np.isfinite(series[name]) | (series[name] <= 0)):
            observations["fwhm_all_positive_finite"] = False
    return observations


def generate_figures(
    npz_path: str | Path,
    figure_dir: str | Path,
    selected_figures: str | Iterable[str] = "all",
    z_shift_cm: float = 0.0,
    dpi: int = 200,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate selected diagnostics and return/write a JSON-serialisable summary.

    Missing optional fields skip only the affected figure.  A malformed
    z-dependent array is treated as a hard error so no misleading plot is made.
    """
    npz_path = Path(npz_path)
    figure_dir = Path(figure_dir)
    if not npz_path.is_file():
        raise FileNotFoundError(f"npz file not found: {npz_path}")
    if not np.isfinite(z_shift_cm):
        raise ValueError("z_shift_cm must be finite")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    selected = _parse_selection(selected_figures)
    with np.load(npz_path, allow_pickle=False) as raw:
        if "z_axis" not in raw.files:
            raise ValueError("z_axis is required for z-direction diagnostics")
        z = _as_vector(raw["z_axis"], "z_axis")
        if z.size == 0:
            raise ValueError("z_axis must not be empty")
        series: dict[str, np.ndarray] = {}
        z_series_names = set(_Z_SERIES_FIELDS)
        z_series_names.update(name for name in raw.files if name.endswith("_z") and name != "rho_onaxis_t_z")
        z_series_names.add("I_onaxis_max_interp_list")
        for name in sorted(z_series_names):
            if name not in raw.files:
                continue
            values = _as_vector(raw[name], name)
            if values.size != z.size:
                raise ValueError(f"{name} length ({values.size}) does not match z_axis length ({z.size})")
            series[name] = values
        rho_tz = np.asarray(raw["rho_onaxis_t_z"], dtype=float) if "rho_onaxis_t_z" in raw.files else None
        t_axis = _as_vector(raw["t_axis"], "t_axis") if "t_axis" in raw.files else None

    if rho_tz is not None:
        if rho_tz.ndim != 2:
            raise ValueError(f"rho_onaxis_t_z must be two-dimensional, got shape {rho_tz.shape}")
        if rho_tz.shape[0] == z.size:
            rho_tz_zt = rho_tz
        elif rho_tz.shape[1] == z.size:
            rho_tz_zt = rho_tz.T
        else:
            raise ValueError(f"rho_onaxis_t_z shape {rho_tz.shape} has no axis matching z_axis length ({z.size})")
        if t_axis is not None and t_axis.size != rho_tz_zt.shape[1]:
            raise ValueError(
                f"t_axis length ({t_axis.size}) does not match rho_onaxis_t_z time length ({rho_tz_zt.shape[1]})"
            )
    else:
        rho_tz_zt = None

    figure_dir.mkdir(parents=True, exist_ok=True)
    z_cm = z * 100.0 + float(z_shift_cm)
    z_label = "z (cm)" if z_shift_cm == 0 else f"z (cm), shifted {z_shift_cm:+g} cm"
    generated: list[str] = []
    skipped: dict[str, str] = {}
    warnings = _sanity_warnings(series, z)
    metrics: dict[str, Any] = {}
    quality_observations = _quality_observations(series, z)

    peak, index = _finite_peak(_positive_for_log(series["I_max_z"])) if "I_max_z" in series else (None, None)
    if peak is not None and index is not None:
        metrics["I_max_peak_W_m2"] = peak
        metrics["z_I_max_peak_m"] = float(z[index])
        metrics["z_I_max_peak_plot_cm"] = float(z_cm[index])
    peak, index = _finite_peak(_positive_for_log(series["rho_onaxis_max_z"])) if "rho_onaxis_max_z" in series else (None, None)
    if peak is not None and index is not None:
        metrics["rho_onaxis_peak_m_3"] = peak
        metrics["z_rho_onaxis_peak_m"] = float(z[index])
        metrics["z_rho_onaxis_peak_plot_cm"] = float(z_cm[index])
    minimum, index = _finite_min(series["w_mom_z"]) if "w_mom_z" in series else (None, None)
    if minimum is not None and index is not None:
        metrics["w_mom_min_m"] = minimum
        metrics["z_focus_est_m"] = float(z[index])
        metrics["z_focus_est_plot_cm"] = float(z_cm[index])
    if "U_z" in series:
        finite_indices = np.flatnonzero(np.isfinite(series["U_z"]))
        if finite_indices.size:
            first, last = int(finite_indices[0]), int(finite_indices[-1])
            u0, u_end = float(series["U_z"][first]), float(series["U_z"][last])
            metrics["U0_J"] = u0
            metrics["U_end_J"] = u_end
            if u0 != 0:
                metrics["U_drift_pct"] = (u_end / u0 - 1.0) * 100.0
            else:
                warnings.append("U_z starts at zero; relative energy drift is undefined.")

    def skip(key: str, reason: str) -> None:
        skipped[FIGURE_SPECS[key]] = reason
        warnings.append(f"Skipped {FIGURE_SPECS[key]}: {reason}")

    if "intensity" in selected:
        names = ("I_max_z", "I_onaxis_max_z", "I_center_t0_z")
        available = [name for name in names if name in series]
        if not available:
            skip("intensity", "none of I_max_z, I_onaxis_max_z, or I_center_t0_z is available")
        else:
            fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True)
            labels = {"I_max_z": "I_max_z", "I_onaxis_max_z": "I_onaxis_max_z", "I_center_t0_z": "I_center_t0_z"}
            for name in available:
                axes[0].semilogy(z_cm, _positive_for_log(series[name]), linewidth=1.5, label=labels[name])
            axes[0].set_ylabel("Intensity (W/m²)")
            axes[0].set_title("Intensity diagnostics (log scale)")
            axes[0].grid(True, which="both", alpha=0.3)
            axes[0].legend(loc="best")
            for name in ("I_max_z", "I_peak_q99_z"):
                if name in series:
                    axes[1].plot(z_cm, _normalise(series[name]), linewidth=1.5, label=f"{name} / max")
            axes[1].set_xlabel(z_label)
            axes[1].set_ylabel("Normalised")
            axes[1].set_title("Peak intensity normalisation")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc="best")
            path = figure_dir / FIGURE_SPECS["intensity"]
            _write_figure(fig, path, dpi)
            generated.append(path.name)

    if "plasma" in selected:
        names = ("rho_onaxis_max_z", "rho_max_z")
        available = [name for name in names if name in series]
        if not available:
            skip("plasma", "neither rho_onaxis_max_z nor rho_max_z is available")
        else:
            fig, ax = plt.subplots(figsize=(8.2, 4.8))
            for name in available:
                ax.semilogy(z_cm, _positive_for_log(series[name]), linewidth=1.6, label=name)
            if "rho_peak_q99_z" in series:
                ax.semilogy(z_cm, _positive_for_log(series["rho_peak_q99_z"]), "--", linewidth=1.2, label="rho_peak_q99_z")
            ax.set(xlabel=z_label, ylabel="Electron density (m⁻³)", title="Plasma density diagnostics (log scale)")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="best")
            path = figure_dir / FIGURE_SPECS["plasma"]
            _write_figure(fig, path, dpi)
            generated.append(path.name)

    if "beam" in selected:
        if "w_mom_z" not in series:
            skip("beam", "w_mom_z is unavailable")
        else:
            fig, ax = plt.subplots(figsize=(8.2, 4.8))
            values = series["w_mom_z"]
            ax.plot(z_cm, values * 1e3, linewidth=1.8, label="w_mom")
            if minimum is not None and index is not None:
                ax.plot(z_cm[index], minimum * 1e3, "ro", label="w_mom minimum")
                ax.axvline(z_cm[index], color="r", linestyle="--", linewidth=1.0)
            ax.set(xlabel=z_label, ylabel="w_mom (mm)", title="Second-moment beam radius")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            path = figure_dir / FIGURE_SPECS["beam"]
            _write_figure(fig, path, dpi)
            generated.append(path.name)

    if "energy" in selected:
        if "U_z" not in series:
            skip("energy", "U_z is unavailable")
        elif "U0_J" not in metrics or metrics["U0_J"] == 0:
            skip("energy", "U_z has no finite non-zero initial value")
        else:
            fig, ax_left = plt.subplots(figsize=(8.2, 4.8))
            values = series["U_z"]
            u0 = float(metrics["U0_J"])
            ax_left.plot(z_cm, values, linewidth=1.8, color="C0")
            ax_left.set_ylabel("U(z) (J)", color="C0")
            ax_left.tick_params(axis="y", labelcolor="C0")
            ax_right = ax_left.twinx()
            ax_right.plot(z_cm, (values / u0 - 1.0) * 100.0, "--", linewidth=1.4, color="C1")
            ax_right.set_ylabel("ΔU / U₀ (%)", color="C1")
            ax_right.tick_params(axis="y", labelcolor="C1")
            ax_left.set(xlabel=z_label, title="Pulse energy and relative drift")
            ax_left.grid(True, alpha=0.3)
            path = figure_dir / FIGURE_SPECS["energy"]
            _write_figure(fig, path, dpi)
            generated.append(path.name)

    if "fwhm" in selected:
        names = ("fwhm_plasma_z", "fwhm_fluence_z")
        available = [name for name in names if name in series]
        if not available:
            skip("fwhm", "neither fwhm_plasma_z nor fwhm_fluence_z is available")
        else:
            fig, ax = plt.subplots(figsize=(8.2, 4.8))
            for name in available:
                ax.plot(z_cm, series[name] * 1e6, linewidth=1.6, label=name.replace("_z", ""))
            ax.set(xlabel=z_label, ylabel="FWHM diameter (µm)", title="Transverse channel scale")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            path = figure_dir / FIGURE_SPECS["fwhm"]
            _write_figure(fig, path, dpi)
            generated.append(path.name)

    if "rho_tz" in selected:
        if rho_tz_zt is None:
            skip("rho_tz", "rho_onaxis_t_z is unavailable")
        elif t_axis is None:
            skip("rho_tz", "t_axis is unavailable")
        else:
            fig, ax = plt.subplots(figsize=(8.2, 5.2))
            plot_values = _positive_for_log(rho_tz_zt).T
            image = ax.pcolormesh(z_cm, t_axis * 1e15, np.log10(plot_values), shading="auto", cmap="turbo")
            ax.set(xlabel=z_label, ylabel="t (fs)", title="log₁₀(on-axis electron density)")
            colorbar = fig.colorbar(image, ax=ax)
            colorbar.set_label("log₁₀(m⁻³)")
            path = figure_dir / FIGURE_SPECS["rho_tz"]
            _write_figure(fig, path, dpi)
            generated.append(path.name)

    summary: dict[str, Any] = {
        "npz_path": str(npz_path),
        "figure_dir": str(figure_dir),
        "Nz": int(z.size),
        "z_shift_cm": float(z_shift_cm),
        "generated_figures": generated,
        "skipped_figures": skipped,
        "sanity_warnings": warnings,
        "quality_observations": {key: _json_value(value) for key, value in quality_observations.items()},
        "metrics": {key: _json_value(value) for key, value in metrics.items()},
    }
    if metadata:
        for key in ("stage_id", "stage_name", "run_id", "case_id", "case_label", "pulse_width_fs", "comparison_mode"):
            if key in metadata:
                summary[key] = _json_value(metadata[key])
    summary_path = figure_dir / "diagnostic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate headless KHz-filament diagnostic PNGs from an NPZ file")
    parser.add_argument("--npz", required=True, help="Input simulation NPZ file")
    parser.add_argument("--fig-dir", required=True, help="Directory for PNGs and diagnostic_summary.json")
    parser.add_argument("--fig-select", default="all", help="Comma-separated: all, intensity, plasma, beam, energy, fwhm, rho_tz")
    parser.add_argument("--z-shift-cm", type=float, default=0.0, help="Manual plotted z-axis shift in cm")
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution")
    parser.add_argument("--prefix", default="", help="Optional prefix added to generated PNG file names")
    parser.add_argument("--metadata-json", default=None, help="Optional JSON file with stage/case metadata")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    metadata = None
    if args.metadata_json:
        metadata = json.loads(Path(args.metadata_json).read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("--metadata-json must contain a JSON object")
    summary = generate_figures(args.npz, args.fig_dir, args.fig_select, args.z_shift_cm, args.dpi, metadata)
    prefix = args.prefix.strip()
    if prefix:
        figure_dir = Path(args.fig_dir)
        renamed: list[str] = []
        for name in summary["generated_figures"]:
            source = figure_dir / name
            target = figure_dir / f"{prefix}{name}"
            source.replace(target)
            renamed.append(target.name)
        summary["generated_figures"] = renamed
        summary["prefix"] = prefix
        (figure_dir / "diagnostic_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    for name in summary["generated_figures"]:
        print(f"[figures] wrote: {Path(args.fig_dir) / name}")
    print(f"[figures] summary: {summary['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
