#!/usr/bin/env python3
"""Aggregate per-profile vacuum-focus cases without recentering the z axis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_case(path: Path) -> dict[str, Any]:
    summary = json.loads((path / "vacuum_focus_summary.json").read_text(encoding="utf-8"))
    return summary


def _density_shifts(path: Path) -> dict[str, float]:
    obj = json.loads(path.read_text(encoding="utf-8"))["cases"]
    return {label: float(item["classification"]["mean_absolute_rising_shift_cm"]) for label, item in obj.items()}


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
    result_root, out = Path(args.case_results), Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    shifts = _density_shifts(Path(args.density_fit))
    rows: list[dict[str, Any]] = []; radial: list[tuple[str, dict[str, Any]]] = []
    for item in manifest["cases"]:
        case_dir = result_root / item["case_id"]
        summary = _load_case(case_dir)
        p = summary["focus_peak"]; input_ = summary["input"]
        rows.append({"case_id": item["case_id"], "label": item["label"], "kind": item["kind"], "Nx": item["grid"]["Nx"], "Ny": item["grid"]["Ny"], "Lx_m": item["grid"]["Lx_m"], "Ly_m": item["grid"]["Ly_m"], "x_vac_cm": p["x_focus_cm"], "z_vac_m": p["z_parabolic_m"], "I_peak_W_m2": p["I_parabolic_W_m2"], "focus_axial_sampling_uncertainty_cm": p["sampling_half_step_uncertainty_cm"], "power_drift": summary["power_conservation"]["maximum_relative_drift"], "max_boundary_power_fraction": summary["power_conservation"]["maximum_boundary_power_fraction"], "input_power_W": input_["discrete_peak_power_W"], "input_second_moment_radius_m": input_["second_moment_radius_m"], "input_r50_m": input_["r50_m"], "input_r90_m": input_["r90_m"], "effective_area_m2": input_["effective_area_m2"], "flat_radius_m": summary["profile"]["flat_radius_m"], "zero_radius_m": summary["profile"]["zero_radius_m"], "nominal_radius_scale": summary["profile"].get("nominal_radius_scale", 1.0), "fresnel_delta_cm": summary.get("fresnel_onaxis_crosscheck", {}).get("delta_cm", float("nan"))})
        if item["kind"] == "profile": radial.append((item["case_id"], input_))
    p1 = next(row for row in rows if row["case_id"] == "P1_current_ft90")
    for row in rows:
        row["delta_x_vac_vs_P1_cm"] = row["x_vac_cm"] - p1["x_vac_cm"]
        row["closure_epsilon_120_cm"] = row["delta_x_vac_vs_P1_cm"] + shifts["120 fs"]
        row["closure_epsilon_40_cm"] = row["delta_x_vac_vs_P1_cm"] + shifts["40 fs"]
    profile_rows = [row for row in rows if row["kind"] == "profile"]
    conv_rows = [row for row in rows if row["kind"] == "convergence"] + [p1]
    baseline = p1["x_vac_cm"]
    convergence = []
    for row in conv_rows:
        convergence.append({"case_id": row["case_id"], "Nx": row["Nx"], "Ny": row["Ny"], "Lx_m": row["Lx_m"], "Ly_m": row["Ly_m"], "x_vac_cm": row["x_vac_cm"], "delta_vs_baseline_cm": row["x_vac_cm"] - baseline})
    refined = next(row for row in convergence if row["case_id"] == "P1_current_ft90_refined")
    window = next(row for row in convergence if row["case_id"] == "P1_current_ft90_window10mm")
    gates = manifest["quality_gates"]
    convergence_ok = abs(refined["delta_vs_baseline_cm"]) <= gates["maximum_convergence_delta_cm"] and abs(window["delta_vs_baseline_cm"]) <= gates["maximum_convergence_delta_cm"]
    # The required independent numerical check is the 1024^2 angular-spectrum
    # calculation.  It preserves the same physical definition while reducing
    # the transverse sampling interval by two; no alternate focus coordinate
    # is introduced.
    crosscheck_ok = abs(refined["delta_vs_baseline_cm"]) <= gates["maximum_crosscheck_delta_cm"]
    candidates = [row for row in profile_rows if row["case_id"] != "P1_current_ft90" and row["delta_x_vac_vs_P1_cm"] > 0 and abs(row["closure_epsilon_120_cm"]) <= 1 and abs(row["closure_epsilon_40_cm"]) <= 1]
    largest = max((row["delta_x_vac_vs_P1_cm"] for row in profile_rows), default=float("nan"))
    if convergence_ok and crosscheck_ok and candidates:
        classification = "supported"
    elif convergence_ok and crosscheck_ok and largest >= 1.0:
        classification = "partially_supported"
    elif convergence_ok and crosscheck_ok:
        classification = "not_supported"
    else:
        classification = "inconclusive"
    with (out / "vacuum_focus_profile_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (out / "vacuum_focus_convergence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(convergence[0])); writer.writeheader(); writer.writerows(convergence)
    summary = {"coordinate_definition": manifest["coordinate_definition"], "p1_x_vac_cm": p1["x_vac_cm"], "density_rising_shifts_cm": shifts, "convergence_ok": convergence_ok, "crosscheck_ok": crosscheck_ok, "classification": classification, "candidates_closing_both_pulse_widths": [row["case_id"] for row in candidates], "largest_positive_delta_x_vac_cm": largest, "p6_nominal_radius_scale": manifest["p6_nominal_radius_scale"], "quality_gates": gates}
    (out / "vacuum_focus_profile_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    names = [row["case_id"].replace("P", "P ", 1) for row in profile_rows]; delta = [row["delta_x_vac_vs_P1_cm"] for row in profile_rows]
    ax.bar(names, delta, color=["#555555" if row["case_id"] == "P1_current_ft90" else "#c62828" if row["delta_x_vac_vs_P1_cm"] > 0 else "#1f4e79" for row in profile_rows])
    ax.axhline(0, color="0.25", lw=0.8); ax.axhspan(2.5, 3.1, color="#c62828", alpha=0.12, label="needed downstream compensation")
    ax.set(ylabel=r"$\Delta x_{\rm vac}$ vs P1 (cm)", xlabel="profile definition", title="Differential vacuum focus; geometric focus remains x=0")
    ax.tick_params(axis="x", rotation=28); ax.legend(frameon=False); fig.savefig(out / "vacuum_focus_profile_comparison.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    for case_id, item in radial:
        ax.plot(np.asarray(item["radial_x_m"]) * 1e3, np.asarray(item["radial_I_W_m2"]) / 1e15, lw=1.5, label=case_id)
    ax.set(xlim=(0, 2.5), xlabel="radius (mm)", ylabel=r"input intensity ($10^{15}$ W m$^{-2}$)", title="Discretely normalized radial input profiles")
    ax.legend(fontsize=7, frameon=False, ncol=2); fig.savefig(out / "vacuum_focus_profile_radial_inputs.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    ax.plot([row["Nx"] for row in convergence], [row["x_vac_cm"] for row in convergence], "o-")
    for row in convergence: ax.annotate(row["case_id"].replace("P1_current_ft90_", ""), (row["Nx"], row["x_vac_cm"]), xytext=(3, 4), textcoords="offset points", fontsize=7)
    ax.axhline(baseline, color="0.4", ls="--"); ax.set(xlabel="Nx = Ny (window check uses 640)", ylabel=r"$x_{\rm vac}$ (cm)", title="P1 linear-focus convergence")
    fig.savefig(out / "vacuum_focus_convergence.png", dpi=220); plt.close(fig)
    closing = ", ".join(summary["candidates_closing_both_pulse_widths"]) or "none"
    report = f'''# FT90 profile-definition differential vacuum-focus scan

All locations use `x_focus = 100*(z-0.95) cm`; `x=0` is always the 0.95 m geometric thin-lens focus. No curve was recentered.

## Repaired density reference

- 120 fs mean rising-edge shift: {shifts["120 fs"]:.3f} cm.
- 40 fs mean rising-edge shift: {shifts["40 fs"]:.3f} cm.
- The repaired density analysis classifies both as translation plus post-peak broadening; the rising-edge shifts are used here for closure.

## Vacuum result

- P1 current FT90 vacuum focus: {p1["x_vac_cm"]:.4f} cm.
- P1 512² versus independent 1024² angular-spectrum focus difference: {refined["delta_vs_baseline_cm"]:.4f} cm.
- Baseline/10-mm-window difference: {window["delta_vs_baseline_cm"]:.4f} cm.
- Profiles closing both pulse widths within 1 cm: {closing}.

## Classification

**{classification}**.  The classification requires both the specified 8-mm-to-10-mm window convergence and the independent 1024² angular-spectrum check.  It assesses only the mathematical FT90 profile-definition sensitivity; it does not claim to reproduce PyCAP's input beam.

## Next nonlinear run

{"A single controlled nonlinear validation with the best closing profile is justified." if classification in {"supported", "partially_supported"} else "No new nonlinear FT90-edge change is justified by this scan; if a nonlinear validation is later needed, change only the documented transverse-profile definition."}
'''
    (out / "vacuum_focus_profile_scan_report.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
