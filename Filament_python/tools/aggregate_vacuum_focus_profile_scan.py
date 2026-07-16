#!/usr/bin/env python3
"""Aggregate FT90 window convergence and independent Fresnel evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


PROFILE_IDS = ["P1_current_ft90", "P2_zero_at_90_narrow", "P3_zero_at_90_wide", "P4_hard_top_R", "P5_hard_top_0p9R", "P6_P2_second_moment_matched"]


def _load_summary(root: Path, case_id: str) -> dict[str, Any]:
    return json.loads((root / case_id / "vacuum_focus_summary.json").read_text(encoding="utf-8"))


def _density_shifts(path: Path) -> dict[str, float]:
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    return {label: float(value["classification"]["mean_absolute_rising_shift_cm"]) for label, value in cases.items()}


def _case_row(item: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    metric = summary["focus_metrics"]
    return {
        "case_id": item["case_id"], "profile_id": item["profile_id"], "window_id": item["window_id"], "kind": item["kind"],
        "Nx": item["grid"]["Nx"], "Ny": item["grid"]["Ny"], "Lx_m": item["grid"]["Lx_m"], "Ly_m": item["grid"]["Ly_m"],
        "x_vac_fft_imax_cm": metric["x_vac_fft_imax_cm"], "x_vac_fft_onaxis_cm": metric["x_vac_fft_onaxis_cm"],
        "x_vac_fresnel_onaxis_cm": metric.get("x_vac_fresnel_onaxis_cm", float("nan")),
        "fft_imax_minus_onaxis_cm": metric["fft_imax_minus_onaxis_cm"],
        "fft_imax_minus_fresnel_cm": metric.get("fft_imax_minus_fresnel_cm", float("nan")),
        "fft_onaxis_minus_fresnel_cm": metric.get("fft_onaxis_minus_fresnel_cm", float("nan")),
        "axial_fwhm_imax_cm": metric["axial_fwhm_imax_cm"], "axial_fwhm_onaxis_cm": metric["axial_fwhm_onaxis_cm"],
        "prefocus_sidelobe_max_ratio_imax": metric["prefocus_sidelobe_max_ratio_imax"],
        "prefocus_sidelobe_max_ratio_onaxis": metric["prefocus_sidelobe_max_ratio_onaxis"],
        "power_drift": summary["power_conservation"]["maximum_relative_drift"],
        "boundary_power_fraction": summary["power_conservation"]["maximum_boundary_power_fraction"],
        "input_power_W": summary["input"]["discrete_peak_power_W"],
        "input_second_moment_radius_m": summary["input"]["second_moment_radius_m"],
        "p6_nominal_radius_scale": summary["profile"].get("nominal_radius_scale", 1.0),
    }


def _window_check(rows: dict[tuple[str, str], dict[str, Any]], low: str, high: str, gate: float) -> tuple[list[dict[str, Any]], bool]:
    p1_low, p1_high = rows[(PROFILE_IDS[0], low)], rows[(PROFILE_IDS[0], high)]
    result: list[dict[str, Any]] = []
    for profile in PROFILE_IDS:
        left, right = rows[(profile, low)], rows[(profile, high)]
        delta_low = left["x_vac_fft_onaxis_cm"] - p1_low["x_vac_fft_onaxis_cm"]
        delta_high = right["x_vac_fft_onaxis_cm"] - p1_high["x_vac_fft_onaxis_cm"]
        result.append({
            "profile_id": profile, "low_window": low, "high_window": high,
            "x_fft_onaxis_low_cm": left["x_vac_fft_onaxis_cm"], "x_fft_onaxis_high_cm": right["x_vac_fft_onaxis_cm"],
            "absolute_focus_change_cm": right["x_vac_fft_onaxis_cm"] - left["x_vac_fft_onaxis_cm"],
            "delta_vs_P1_low_cm": delta_low, "delta_vs_P1_high_cm": delta_high,
            "differential_focus_change_cm": delta_high - delta_low,
            "absolute_ok": abs(right["x_vac_fft_onaxis_cm"] - left["x_vac_fft_onaxis_cm"]) <= gate,
            "differential_ok": abs(delta_high - delta_low) <= gate,
        })
    return result, all(bool(row["absolute_ok"] and row["differential_ok"]) for row in result)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--case-results", required=True)
    parser.add_argument("--density-fit", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    root, out = Path(args.case_results), Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    density = _density_shifts(Path(args.density_fit)); gates = manifest["quality_gates"]
    case_rows = [_case_row(item, _load_summary(root, item["case_id"])) for item in manifest["cases"]]
    by_key = {(row["profile_id"], row["window_id"]): row for row in case_rows if row["kind"] == "window"}
    for profile in PROFILE_IDS:
        for window in ("8mm_512", "10mm_640", "12mm_768", "14mm_896"):
            if (profile, window) not in by_key:
                raise RuntimeError(f"missing required window case {profile} @ {window}")
    p1_8 = by_key[(PROFILE_IDS[0], "8mm_512")]
    resolution_row = next(row for row in case_rows if row["kind"] == "resolution")
    resolution_delta = resolution_row["x_vac_fft_onaxis_cm"] - p1_8["x_vac_fft_onaxis_cm"]
    resolution_ok = abs(resolution_delta) <= gates["maximum_resolution_delta_cm"]
    check_10_12, ok_10_12 = _window_check(by_key, "10mm_640", "12mm_768", gates["maximum_window_delta_cm"])
    check_12_14, ok_12_14 = _window_check(by_key, "12mm_768", "14mm_896", gates["maximum_window_delta_cm"])
    final_window = "12mm_768" if ok_10_12 else "14mm_896"
    window_ok = ok_10_12 or ok_12_14
    active_window_rows = [row for row in case_rows if row["kind"] == "window" and row["window_id"] == final_window]
    p1_final = next(row for row in active_window_rows if row["profile_id"] == PROFILE_IDS[0])
    fresnel_rows: list[dict[str, Any]] = []
    axisymmetry_ok = True
    fresnel_ok = True
    final_metrics: list[dict[str, Any]] = []
    for row in active_window_rows:
        delta_fft = row["x_vac_fft_onaxis_cm"] - p1_final["x_vac_fft_onaxis_cm"]
        delta_fresnel = row["x_vac_fresnel_onaxis_cm"] - p1_final["x_vac_fresnel_onaxis_cm"]
        fft_onaxis_ok = abs(row["fft_onaxis_minus_fresnel_cm"]) <= gates["maximum_crosscheck_delta_cm"]
        differential_ok = abs(delta_fft - delta_fresnel) <= gates["maximum_crosscheck_delta_cm"]
        imax_onaxis_ok = abs(row["fft_imax_minus_onaxis_cm"]) <= gates["maximum_fft_imax_onaxis_delta_cm"]
        fresnel_rows.append({
            "profile_id": row["profile_id"], "window_id": final_window,
            "x_vac_fft_imax_cm": row["x_vac_fft_imax_cm"], "x_vac_fft_onaxis_cm": row["x_vac_fft_onaxis_cm"],
            "x_vac_fresnel_onaxis_cm": row["x_vac_fresnel_onaxis_cm"], "fft_imax_minus_fresnel_cm": row["fft_imax_minus_fresnel_cm"],
            "fft_onaxis_minus_fresnel_cm": row["fft_onaxis_minus_fresnel_cm"], "fft_imax_minus_onaxis_cm": row["fft_imax_minus_onaxis_cm"],
            "delta_fft_vs_P1_cm": delta_fft, "delta_fresnel_vs_P1_cm": delta_fresnel,
            "delta_fft_minus_fresnel_cm": delta_fft - delta_fresnel, "fft_onaxis_fresnel_ok": fft_onaxis_ok,
            "differential_fresnel_ok": differential_ok, "fft_imax_onaxis_ok": imax_onaxis_ok,
        })
        axisymmetry_ok &= imax_onaxis_ok; fresnel_ok &= fft_onaxis_ok and differential_ok
        final_metrics.append({**row, "delta_x_vac_vs_P1_cm": delta_fft, "closure_epsilon_120_cm": delta_fft + density["120 fs"], "closure_epsilon_40_cm": delta_fft + density["40 fs"]})
    candidates = [row for row in final_metrics if row["profile_id"] != PROFILE_IDS[0] and row["delta_x_vac_vs_P1_cm"] > 0 and abs(row["closure_epsilon_120_cm"]) <= 1.0 and abs(row["closure_epsilon_40_cm"]) <= 1.0]
    positive = [row["delta_x_vac_vs_P1_cm"] for row in final_metrics if row["profile_id"] != PROFILE_IDS[0] and row["delta_x_vac_vs_P1_cm"] > 0]
    numerical_status = {"resolution_convergence_ok": resolution_ok, "window_convergence_ok": window_ok, "independent_fresnel_crosscheck_ok": fresnel_ok, "fft_imax_onaxis_consistent": axisymmetry_ok}
    if not all(numerical_status.values()):
        classification = "inconclusive"
    elif candidates:
        classification = "supported"
    elif any(1.0 <= value <= 2.0 for value in positive):
        classification = "partially_supported"
    else:
        classification = "not_supported"

    _write_csv(out / "vacuum_focus_all_cases_v2.csv", case_rows)
    _write_csv(out / "vacuum_focus_profile_metrics_v2.csv", final_metrics)
    _write_csv(out / "vacuum_focus_window_convergence.csv", check_10_12 + check_12_14)
    _write_csv(out / "vacuum_focus_fresnel_crosscheck.csv", fresnel_rows)
    _write_csv(out / "vacuum_focus_axial_shape_metrics.csv", [{key: row[key] for key in ("profile_id", "window_id", "axial_fwhm_imax_cm", "axial_fwhm_onaxis_cm", "prefocus_sidelobe_max_ratio_imax", "prefocus_sidelobe_max_ratio_onaxis")} for row in final_metrics])
    summary = {
        "coordinate_definition": manifest["coordinate_definition"], "final_window": final_window, "density_rising_shifts_cm": density,
        "numerical_status": numerical_status, "resolution_delta_cm": resolution_delta,
        "window_pairs": {"10_to_12_ok": ok_10_12, "12_to_14_ok": ok_12_14}, "physical_evidence": {
            "all_candidate_differential_shifts_cm": {row["profile_id"]: row["delta_x_vac_vs_P1_cm"] for row in final_metrics if row["profile_id"] != PROFILE_IDS[0]},
            "candidates_closing_both_pulse_widths": [row["profile_id"] for row in candidates],
            "largest_positive_delta_x_vac_cm": max(positive, default=0.0),
        }, "final_classification": classification, "quality_gates": gates,
    }
    (out / "vacuum_focus_profile_summary_v2.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    labels = [row["profile_id"].split("_")[0] for row in final_metrics]; values = [row["delta_x_vac_vs_P1_cm"] for row in final_metrics]
    ax.bar(labels, values, color=["0.35" if row["profile_id"] == PROFILE_IDS[0] else "#c62828" if row["delta_x_vac_vs_P1_cm"] > 0 else "#1f4e79" for row in final_metrics])
    ax.axhline(0, color="0.25", lw=0.9); ax.axhspan(2.5, 3.3, color="#c62828", alpha=0.12, label="needed downstream compensation")
    ax.set(xlabel="profile definition", ylabel=r"$\Delta x_{\rm vac}$ vs P1 (cm)", title=f"Converged-window differential vacuum focus ({final_window})")
    ax.legend(frameon=False); fig.savefig(out / "vacuum_focus_profile_comparison_v2.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    for profile in PROFILE_IDS:
        subset = [row for row in case_rows if row["profile_id"] == profile and row["kind"] == "window"]
        subset.sort(key=lambda row: row["Lx_m"])
        ax.plot([row["Lx_m"] * 1e3 for row in subset], [row["x_vac_fft_onaxis_cm"] for row in subset], "o-", label=profile.split("_")[0])
    ax.set(xlabel="transverse window (mm)", ylabel=r"$x_{\rm vac,FFT,onaxis}$ (cm)", title="Window convergence at fixed Δx≈15.625 μm")
    ax.legend(frameon=False, ncol=2); fig.savefig(out / "vacuum_focus_window_convergence_v2.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    fft = [row["x_vac_fft_onaxis_cm"] for row in fresnel_rows]; fresnel = [row["x_vac_fresnel_onaxis_cm"] for row in fresnel_rows]
    ax.scatter(fresnel, fft, color="#1f4e79")
    lo, hi = min(fft + fresnel) - 0.15, max(fft + fresnel) + 0.15; ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    for row in fresnel_rows: ax.annotate(row["profile_id"].split("_")[0], (row["x_vac_fresnel_onaxis_cm"], row["x_vac_fft_onaxis_cm"]), xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel=r"continuous Fresnel $x_{\rm vac,onaxis}$ (cm)", ylabel=r"FFT $x_{\rm vac,onaxis}$ (cm)", title=f"Independent Fresnel crosscheck ({final_window})")
    fig.savefig(out / "vacuum_focus_fresnel_crosscheck.png", dpi=220); plt.close(fig)
    report = f'''# FT90 vacuum-focus window closure and independent Fresnel verification

All coordinates use `x_focus = 100*(z-0.95) cm`; zero is permanently the 0.95 m geometric focus.

## Numerical status

- Resolution convergence (P1 512² vs 1024² at 8 mm): `{resolution_ok}`; difference `{resolution_delta:.4f}` cm.
- Window convergence (all P1--P6): `{window_ok}`.  10→12 mm: `{ok_10_12}`; 12→14 mm: `{ok_12_14}`.
- Independent continuous Fresnel crosscheck at `{final_window}`: `{fresnel_ok}`.
- FFT `I_max` versus on-axis focus consistency: `{axisymmetry_ok}`.

## Physical evidence

- Fixed-density rising shifts: 120 fs `{density["120 fs"]:.3f}` cm; 40 fs `{density["40 fs"]:.3f}` cm.
- Final-window positive differential shifts: `{positive}` cm.
- Candidates closing both pulse widths within 1 cm: `{[row["profile_id"] for row in candidates]}`.

## Final classification

**{classification}**.

{"In the converged and independently verified window, the P1--P6 mathematical definitions do not supply a downstream shift capable of compensating the 2.6--3.3 cm density-rise advance." if classification == "not_supported" else "The classification follows the numerical and closure criteria above; it does not claim to reproduce PyCAP."}
'''
    (out / "vacuum_focus_profile_scan_report_v2.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
