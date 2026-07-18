#!/usr/bin/env python3
"""Fixed-coordinate 120 fs Talebpour Raman-phase causal comparison."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RHO_THRESHOLDS = (1e19, 1e20, 1e21, 1e22)
INTENSITY_THRESHOLDS = (1e16, 3e16, 1e17, 3e17)


def read_csv(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return {key: np.asarray([float(row[key]) for row in rows], dtype=float) for key in rows[0]}


def merge(base: dict[str, np.ndarray], extra: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if len(base["z_m"]) != len(extra["z_m"]) or not np.allclose(base["z_m"], extra["z_m"], rtol=0.0, atol=2e-8):
        raise ValueError("base and Raman-extra CSV axes are not identical")
    return {**base, **{key: value for key, value in extra.items() if key not in ("z_m", "x_focus_cm")}}


def crossing(x: np.ndarray, y: np.ndarray, threshold: float, *, descending: bool = False, start: int = 0) -> float | None:
    a, b = (y[:-1], y[1:])
    mask = (a >= threshold) & (b < threshold) if descending else (a < threshold) & (b >= threshold)
    hits = np.flatnonzero(mask & (np.arange(a.size) >= start))
    if not hits.size:
        return None
    i = int(hits[0]); denom = b[i] - a[i]
    return float(x[i] if denom == 0.0 else x[i] + (threshold - a[i]) * (x[i + 1] - x[i]) / denom)


def density_metrics(x: np.ndarray, rho: np.ndarray) -> dict[str, Any]:
    peak_i = int(np.argmax(rho)); peak = float(rho[peak_i]); top = rho >= 0.99 * peak
    left = peak_i
    while left > 0 and top[left - 1]: left -= 1
    right = peak_i
    while right + 1 < rho.size and top[right + 1]: right += 1
    half = 0.5 * peak
    half_left = crossing(x[:peak_i + 1], rho[:peak_i + 1], half)
    half_right = crossing(x, rho, half, descending=True, start=peak_i)
    durations = {}
    for threshold in RHO_THRESHOLDS:
        onset = crossing(x, rho, threshold)
        end = crossing(x, rho, threshold, descending=True, start=0)
        durations[str(int(threshold))] = None if onset is None or end is None else float(end - onset)
    return {
        "rho_peak_m3": peak, "peak_x_cm": float(x[peak_i]), "peak_top_center_cm": float((x[left] + x[right]) / 2.0),
        "fwhm_cm": None if half_left is None or half_right is None else float(half_right - half_left),
        "post_peak_half_distance_cm": None if half_right is None else float(half_right - x[peak_i]),
        "tail_area_above_half_m3_cm": float(np.trapezoid(np.maximum(rho[peak_i:] - half, 0.0), x[peak_i:])),
        "threshold_crossings_cm": {str(int(t)): crossing(x, rho, t) for t in RHO_THRESHOLDS},
        "threshold_duration_cm": durations,
    }


def rmse(x: np.ndarray, y: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray) -> float:
    lo, hi = max(float(x.min()), float(ref_x.min())), min(float(x.max()), float(ref_x.max()))
    grid = np.arange(lo, hi + 1e-12, 0.025)
    return float(math.sqrt(np.mean((np.interp(grid, x, y) - np.interp(grid, ref_x, ref_y)) ** 2)))


def _position_improvement(full: float | None, off: float | None, paper: float | None, epsilon: float) -> bool:
    return all(value is not None for value in (full, off, paper)) and abs(float(full) - float(paper)) + epsilon <= abs(float(off) - float(paper))


def classify(full: dict[str, Any], off: dict[str, Any], paper: dict[str, Any], *, epsilon_x_cm: float, numerical_ok: bool) -> tuple[str, dict[str, Any]]:
    full_21, off_21, paper_21 = (full["threshold_crossings_cm"]["1000000000000000000000"], off["threshold_crossings_cm"]["1000000000000000000000"], paper["threshold_crossings_cm"]["1000000000000000000000"])
    onset_effect = None if full_21 is None or off_21 is None else float(full_21 - off_21)
    global_effects = [
        abs(onset_effect) if onset_effect is not None else 0.0,
        abs(float(full["peak_top_center_cm"]) - float(off["peak_top_center_cm"])),
        abs(float(full["fwhm_cm"] or 0.0) - float(off["fwhm_cm"] or 0.0)),
    ]
    effect_resolved = max(global_effects) > epsilon_x_cm
    onset_improves = _position_improvement(full_21, off_21, paper_21, epsilon_x_cm)
    center_improves = _position_improvement(full["peak_top_center_cm"], off["peak_top_center_cm"], paper["peak_top_center_cm"], epsilon_x_cm)
    peak_collapse = full["rho_peak_m3"] < 0.5 * off["rho_peak_m3"]
    tail_bad = full["tail_area_above_half_m3_cm"] > 1.5 * off["tail_area_above_half_m3_cm"]
    if not numerical_ok:
        label = "raman_phase_inconclusive"
    elif effect_resolved and onset_improves and center_improves and not peak_collapse and not tail_bad:
        label = "raman_phase_supported"
    elif effect_resolved and not peak_collapse:
        label = "raman_phase_partially_supported"
    else:
        label = "raman_phase_not_supported"
    return label, {"onset_1e21_full_minus_off_cm": onset_effect, "effect_resolved": effect_resolved, "onset_improves_vs_pycap": onset_improves, "peak_center_improves_vs_pycap": center_improves, "peak_collapse": peak_collapse, "tail_unacceptably_worse": tail_bad, "numerical_path_ok": numerical_ok}


def _plot(path: Path, x: np.ndarray, series: list[tuple[np.ndarray, str]], title: str, *, log: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    for y, label in series: ax.plot(x, y, label=label, lw=1.4)
    if log: ax.set_yscale("log")
    ax.set(xlabel="x_focus (cm)", title=title); ax.grid(alpha=0.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def compare(full: dict[str, np.ndarray], off: dict[str, np.ndarray], pycap: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    x = full["x_focus_cm"]
    if not np.allclose(x, off["x_focus_cm"], rtol=0.0, atol=2e-6): raise ValueError("full and Raman-off axes differ")
    eps = max(0.10, 3.0 * float(np.median(np.diff(x))))
    fm, om = density_metrics(x, full["rho_max_z"]), density_metrics(x, off["rho_max_z"])
    px, py = pycap; pm = density_metrics(px, py)
    numerical_ok = (float(np.max(full["adaptive_rejection_count_z"])) == float(np.max(off["adaptive_rejection_count_z"])) and float(np.max(full["safety_mode_trigger_count_z"])) == float(np.max(off["safety_mode_trigger_count_z"])))
    label, decision = classify(fm, om, pm, epsilon_x_cm=eps, numerical_ok=numerical_ok)
    return {
        "schema": "khz_filament.phase6.raman_phase_causality.v1", "coordinate_definition": "x_focus_cm = 100 * (z_m - 0.95)",
        "formal_curve_shift_cm": 0.0, "epsilon_x_cm": eps, "classification": label, "decision": decision,
        "metrics": {"talebpour_full": fm, "raman_phase_off": om, "pycap_120fs": pm},
        "unshifted_rmse_m3": {"full_vs_pycap": rmse(x, full["rho_max_z"], px, py), "raman_phase_off_vs_pycap": rmse(x, off["rho_max_z"], px, py)},
        "raman_peaks": {key: {"full": float(np.max(np.abs(full[key]))), "raman_phase_off": float(np.max(np.abs(off[key])))} for key in ("dphi_rot_max_abs_z", "dphi_rot_applied_max_abs_z", "alpha_R_applied_max_z")},
        "numerical_path": {"full": {key: float(np.max(full[key])) for key in ("dz_used_z", "adaptive_rejection_count_z", "safety_mode_trigger_count_z")}, "raman_phase_off": {key: float(np.max(off[key])) for key in ("dz_used_z", "adaptive_rejection_count_z", "safety_mode_trigger_count_z")}},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True); parser.add_argument("--full-raman", type=Path, required=True)
    parser.add_argument("--raman-off", type=Path, required=True); parser.add_argument("--raman-off-raman", type=Path, required=True)
    parser.add_argument("--pycap", type=Path, required=True); parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    full, off = merge(read_csv(args.full), read_csv(args.full_raman)), merge(read_csv(args.raman_off), read_csv(args.raman_off_raman))
    paper = read_csv(args.pycap); result = compare(full, off, (paper["x_focus_cm"], paper["rho_1e16_cm3"] * 1e22)); x = full["x_focus_cm"]
    (args.out_dir / "raman_phase_causality_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    for case, metrics in result["metrics"].items():
        rows.append({"case": case, "rho_peak_m3": metrics["rho_peak_m3"], "peak_top_center_cm": metrics["peak_top_center_cm"], "fwhm_cm": metrics["fwhm_cm"], **{f"rho_{key}_x_cm": value for key, value in metrics["threshold_crossings_cm"].items()}})
    with (args.out_dir / "raman_phase_causality_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    decision = {"classification": result["classification"], "epsilon_x_cm": result["epsilon_x_cm"], **result["decision"]}
    with (args.out_dir / "phase6_decision.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decision)); writer.writeheader(); writer.writerow(decision)
    report = ["# 120 fs Raman phase causality", "", f"Classification: **{result['classification']}**.", "", f"- Fixed coordinate: `{result['coordinate_definition']}`", f"- epsilon_x: `{result['epsilon_x_cm']:.3f} cm`", f"- Full minus Raman-off 1e21 onset: `{result['decision']['onset_1e21_full_minus_off_cm']}` cm", f"- Formal curve shift: `0.0 cm`", "", "This result applies only to the controlled 120 fs Raman-phase single-factor ablation; it does not establish a 40 fs conclusion."]
    (args.out_dir / "raman_phase_causality_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    pyinterp = np.interp(x, paper["x_focus_cm"], paper["rho_1e16_cm3"] * 1e22, left=np.nan, right=np.nan)
    plots = [
        ("01_density_absolute.png", [(full["rho_max_z"], "Tal full"), (off["rho_max_z"], "Raman phase off"), (pyinterp, "PyCAP")], "Density, absolute coordinate", True),
        ("02_density_onset_zoom.png", [(full["rho_max_z"], "Tal full"), (off["rho_max_z"], "Raman phase off")], "Density onset", True),
        ("03_intensity.png", [(full["I_max_z"], "Tal full"), (off["I_max_z"], "Raman phase off")], "Peak intensity", True),
        ("04_species.png", [(full["rho_N2_max_z"], "Full N2"), (off["rho_N2_max_z"], "Off N2"), (full["rho_O2_max_z"], "Full O2"), (off["rho_O2_max_z"], "Off O2")], "Species density", True),
        ("05_raman_raw_phase.png", [(full["dphi_rot_max_abs_z"], "Full raw"), (off["dphi_rot_max_abs_z"], "Off raw")], "Raw Raman phase diagnostic", True),
        ("06_raman_applied_phase.png", [(full["dphi_rot_applied_max_abs_z"], "Full applied"), (off["dphi_rot_applied_max_abs_z"], "Off applied")], "Applied Raman phase diagnostic", True),
        ("07_raman_absorption.png", [(full["alpha_R_applied_max_z"], "Full"), (off["alpha_R_applied_max_z"], "Off")], "Applied Raman absorption", True),
        ("08_plasma_phase.png", [(full["dphi_plasma_applied_max_abs_z"], "Full"), (off["dphi_plasma_applied_max_abs_z"], "Off")], "Applied plasma phase", True),
        ("09_ionization_loss.png", [(full["alpha_ion_applied_max_z"], "Full"), (off["alpha_ion_applied_max_z"], "Off")], "Ionization loss", True),
        ("10_energy.png", [(full["E_dep_cumulative_z"], "Full"), (off["E_dep_cumulative_z"], "Off")], "Cumulative deposited energy", False),
        ("11_step_size.png", [(full["dz_used_z"], "Full"), (off["dz_used_z"], "Off")], "Actual propagation step", False),
        ("12_safety_counters.png", [(full["adaptive_rejection_count_z"], "Full rejections"), (off["adaptive_rejection_count_z"], "Off rejections"), (full["safety_mode_trigger_count_z"], "Full safety"), (off["safety_mode_trigger_count_z"], "Off safety")], "Numerical-path counters", False),
    ]
    for name, series, title, log in plots: _plot(args.out_dir / name, x, series, title, log=log)
    print(result["classification"])


if __name__ == "__main__": main()
