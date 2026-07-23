#!/usr/bin/env python3
"""Analyze strict full-Eq.27 Raman-feedback ON/OFF Test A outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THRESHOLDS = (1e19, 1e20, 1e21, 1e22)
FOCUS_M = 0.95
REQUIRED = (
    "z_axis", "rho_max_z", "rho_onaxis_max_z", "I_max_z", "w_mom_z", "fwhm_time_z", "U_z",
    "alpha_ion_applied_max_z", "dphi_plasma_applied_max_abs_z", "dphi_elec_applied_max_abs_z",
    "raman_IR_max_raw", "raman_rhs_l2_norm", "raman_target_loss_step_J", "raman_actual_loss_step_J",
    "raman_closure_residual_step", "raman_cumulative_closure_residual",
)


def crossing(x: np.ndarray, y: np.ndarray, level: float, *, falling=False, begin=0):
    a, b = y[:-1], y[1:]
    mask = ((a >= level) & (b < level)) if falling else ((a < level) & (b >= level))
    idx = np.flatnonzero(mask & (np.arange(mask.size) >= begin))
    if not idx.size:
        return None
    i = int(idx[0])
    return float(x[i] + (level - a[i]) * (x[i + 1] - x[i]) / (b[i + 1] - a[i]))


def interpolate(x: np.ndarray, y: np.ndarray, point: float):
    if point < x[0] or point > x[-1]:
        return None
    return float(np.interp(point, x, y))


def shape(x: np.ndarray, rho: np.ndarray) -> dict:
    i = int(np.argmax(rho)); peak = float(rho[i]); level = .99 * peak
    left = i
    while left > 0 and rho[left - 1] >= level: left -= 1
    right = i
    while right + 1 < rho.size and rho[right + 1] >= level: right += 1
    rise50, fall50 = crossing(x[:i + 1], rho[:i + 1], .5 * peak), crossing(x, rho, .5 * peak, falling=True, begin=i)
    rise10, rise90 = crossing(x[:i + 1], rho[:i + 1], .1 * peak), crossing(x[:i + 1], rho[:i + 1], .9 * peak)
    fall90, fall10 = crossing(x, rho, .9 * peak, falling=True, begin=i), crossing(x, rho, .1 * peak, falling=True, begin=i)
    def tail(offset):
        start = float(x[i] + offset)
        if start >= x[-1]: return None
        j = int(np.searchsorted(x, start, side="left"))
        xx = np.concatenate(([start], x[j:]))
        yy = np.concatenate(([np.interp(start, x, rho)], rho[j:]))
        return float(np.trapezoid(yy, xx))
    return {
        "peak_rho_m3": peak, "peak_x_focus_cm": float(x[i]),
        "peak_plateau_center_cm": float((x[left] + x[right]) / 2),
        "fwhm_cm": None if rise50 is None or fall50 is None else float(fall50 - rise50),
        "rise_10_90_cm": None if rise10 is None or rise90 is None else float(rise90 - rise10),
        "fall_90_10_cm": None if fall90 is None or fall10 is None else float(fall10 - fall90),
        "rho_peak_plus_5cm_m3": interpolate(x, rho, x[i] + 5),
        "rho_peak_plus_10cm_m3": interpolate(x, rho, x[i] + 10),
        "post_peak_tail_integral_m3_cm": float(np.trapezoid(rho[i:], x[i:])),
        "post_peak_plus_5cm_tail_integral_m3_cm": tail(5),
        "post_peak_plus_10cm_tail_integral_m3_cm": tail(10),
    }


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(set(REQUIRED).difference(data.files))
        if missing:
            raise ValueError(f"{path} missing required diagnostics: {missing}")
        result = {key: np.asarray(data[key]) for key in REQUIRED}
        for key in ("raman_operator_feedback_enabled", "raman_operator_applied", "alpha_R_applied_max_z", "raman_convolution_count_step", "raman_operator_substep_count"):
            if key in data: result[key] = np.asarray(data[key])
    z = np.asarray(result["z_axis"], float)
    if z.size < 2 or not np.all(np.diff(z) > 0): raise ValueError(f"{path} has invalid z axis")
    if any(not np.all(np.isfinite(np.asarray(value, float))) for value in result.values() if np.issubdtype(np.asarray(value).dtype, np.number)):
        raise ValueError(f"{path} contains NaN/Inf")
    result["x_focus_cm"] = 100.0 * (z - FOCUS_M)
    return result


def audit(case: str, data: dict) -> dict:
    feedback = bool(np.asarray(data.get("raman_operator_feedback_enabled", False)).item())
    applied = np.asarray(data.get("raman_operator_applied", []), bool)
    rhs = np.asarray(data["raman_rhs_l2_norm"], float)
    raw = np.asarray(data["raman_IR_max_raw"], float)
    checks = {
        "core_diagnostics_present_and_finite": True,
        "legacy_alpha_zero": bool(np.max(np.abs(np.asarray(data.get("alpha_R_applied_max_z", [0]), float))) == 0),
        "raw_raman_response_nonzero": bool(np.max(np.abs(raw)) > 0),
        "feedback_state": feedback if case == "on" else not feedback,
        "operator_application_state": bool(np.all(applied)) if case == "on" else bool(not np.any(applied)),
        "rhs_state": bool(np.max(np.abs(rhs)) > 0) if case == "on" else bool(np.max(np.abs(rhs)) == 0),
    }
    return {"schema": "phase8c.full_eq27_raman.test_a.diagnostic_audit.v1", "case": case, "status": "passed" if all(checks.values()) else "failed", "checks": checks}


def pycap(path: Path):
    raw = np.genfromtxt(path, delimiter=",", names=True)
    return np.asarray(raw["x_focus_cm"], float), np.asarray(raw["rho_1e16_cm3"], float) * 1e22


def write_csv(path: Path, rows: list[dict]):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--on", type=Path, required=True); parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pycap", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "density_translation_width" / "density_translation_width_20260715_002" / "paper_pycap_120fs.csv")
    args = parser.parse_args(argv); args.out_dir.mkdir(parents=True, exist_ok=True)
    on, off = load(args.on), load(args.off)
    x, rho_on, rho_off = on["x_focus_cm"], np.asarray(on["rho_max_z"], float), np.asarray(off["rho_max_z"], float)
    if not np.array_equal(x, off["x_focus_cm"]): raise ValueError("ON/OFF z axes differ")
    on_audit, off_audit = audit("on", on), audit("off", off)
    (args.out_dir / "test_a_on_diagnostic_audit.json").write_text(json.dumps(on_audit, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "test_a_off_diagnostic_audit.json").write_text(json.dumps(off_audit, indent=2) + "\n", encoding="utf-8")
    if on_audit["status"] != "passed" or off_audit["status"] != "passed": raise SystemExit("diagnostic audit failed")
    threshold_rows = []
    for level in THRESHOLDS:
        x_on, x_off = crossing(x, rho_on, level), crossing(x, rho_off, level)
        threshold_rows.append({"threshold_m3": level, "x_crossing_on_cm": x_on, "x_crossing_off_cm": x_off, "delta_x_on_minus_off_cm": None if x_on is None or x_off is None else x_on - x_off})
    on_shape, off_shape = shape(x, rho_on), shape(x, rho_off)
    rows = []
    for case, payload in (("full_eq27_raman_on", on_shape), ("full_eq27_raman_off", off_shape)):
        rows.extend({"case": case, "metric": key, "value": value} for key, value in payload.items())
    write_csv(args.out_dir / "test_a_metrics.csv", rows)
    write_csv(args.out_dir / "test_a_crossing_shifts.csv", threshold_rows)
    px, py = pycap(args.pycap)
    write_csv(args.out_dir / "test_a_pycap_comparison.csv", [{"x_focus_cm": xx, "rho_pycap_m3": yy, "rho_on_m3": float(np.interp(xx, x, rho_on)), "rho_off_m3": float(np.interp(xx, x, rho_off))} for xx, yy in zip(px, py)])
    delta22 = next(row["delta_x_on_minus_off_cm"] for row in threshold_rows if row["threshold_m3"] == 1e22)
    classification = "inconclusive_missing_1e22_crossing" if delta22 is None else ("full_eq27_raman_major_contributor" if abs(delta22) >= 1.0 else "full_eq27_raman_not_material_for_onset" if abs(delta22) <= .10 else "full_eq27_raman_partial_contributor")
    summary = {"schema": "phase8c.full_eq27_raman.test_a.effect_summary.v1", "coordinate": "x_focus_cm = 100 * (z_m - 0.95), no shifting/smoothing/renormalization", "classification": classification, "crossings": threshold_rows, "shape": {"on": on_shape, "off": off_shape}, "effect_chain": {"raman_feedback": "on: applied RHS nonzero; off: raw response retained with RHS zero", "intensity_field": "I_max_z", "electronic_kerr": "dphi_elec_applied_max_abs_z", "ionization_proxy": "alpha_ion_applied_max_z", "density": "rho_max_z", "plasma": "dphi_plasma_applied_max_abs_z"}}
    (args.out_dir / "test_a_effect_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(); ax.semilogy(x, rho_on, label="full Eq.27 ON"); ax.semilogy(x, rho_off, label="full Eq.27 OFF"); ax.semilogy(px, py, "k--", label="PyCAP digitized"); ax.set(xlabel="x_focus (cm)", ylabel="rho_max (m^-3)"); ax.legend(); fig.tight_layout(); fig.savefig(args.out_dir / "rho_max_on_off_pycap.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(); ax.plot(x, on["I_max_z"], label="ON"); ax.plot(x, off["I_max_z"], label="OFF"); ax.set(xlabel="x_focus (cm)", ylabel="I_max (W m^-2)"); ax.legend(); fig.tight_layout(); fig.savefig(args.out_dir / "i_max_on_off.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(); ax.plot([r["threshold_m3"] for r in threshold_rows], [r["delta_x_on_minus_off_cm"] for r in threshold_rows], "o-"); ax.set_xscale("log"); ax.axhline(0, color="k", lw=.8); ax.set(xlabel="density threshold (m^-3)", ylabel="ON - OFF crossing shift (cm)"); fig.tight_layout(); fig.savefig(args.out_dir / "crossing_shift_vs_threshold.png", dpi=180); plt.close(fig)
    effect = ["raman_rhs_l2_norm", "I_max_z", "dphi_elec_applied_max_abs_z", "alpha_ion_applied_max_z", "rho_max_z", "dphi_plasma_applied_max_abs_z"]
    fig, axes = plt.subplots(len(effect), 1, sharex=True, figsize=(7, 10))
    for ax, field in zip(axes, effect): ax.plot(x, on[field], label="ON"); ax.plot(x, off[field], label="OFF"); ax.set_ylabel(field)
    axes[0].legend(); axes[-1].set_xlabel("x_focus (cm)"); fig.tight_layout(); fig.savefig(args.out_dir / "raman_intensity_density_effect_chain.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
