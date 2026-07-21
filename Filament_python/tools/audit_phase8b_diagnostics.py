#!/usr/bin/env python3
"""Audit a completed Phase 8B production diagnostic archive against its contract."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _scalar(data, key):
    return np.asarray(data[key]).item()


def _check(name, passed, actual, expected) -> dict:
    return {"name": name, "passed": bool(passed), "actual": actual, "expected": expected}


def audit(data, contract: dict, job: str) -> dict:
    checks = []
    required = contract["required_fields"]
    required_keys = {"z_axis", *required["aligned_z_histories"], *required["z_leading_arrays"], *required["scalar_or_text"]}
    missing = sorted(required_keys.difference(data.files))
    checks.append(_check("required_fields", not missing, missing, "empty missing-field list"))
    if missing:
        return {"job": job, "status": "failed", "checks": checks}

    z = np.asarray(data["z_axis"], dtype=float)
    dz = np.asarray(data["dz_used_z"], dtype=float)
    n = z.size
    checks.append(_check("nonempty", n > 0, n, ">0"))
    checks.append(_check("z_strictly_increasing", n == 1 or np.all(np.diff(z) > 0.0), bool(n == 1 or np.all(np.diff(z) > 0.0)), True))
    checks.append(_check("positive_dz", np.all(dz > 0.0), float(np.min(dz)) if dz.size else math.nan, ">0"))
    checks.append(_check("z_reconstructed_by_dz", np.allclose(z, np.cumsum(dz), rtol=0.0, atol=2e-12), float(np.max(np.abs(z-np.cumsum(dz)))) if n else math.nan, "<=2e-12 m"))
    z_final = float(contract["fixed_coordinates"]["z_final_m"])
    z_tol = float(contract["fixed_coordinates"]["z_final_absolute_tolerance_m"])
    checks.append(_check("z_final", abs(float(z[-1])-z_final) <= z_tol, float(z[-1]), z_final))
    aligned_bad = [key for key in required["aligned_z_histories"] if np.asarray(data[key]).shape[:1] != (n,)]
    checks.append(_check("aligned_z_histories", not aligned_bad, aligned_bad, "all first dimensions equal len(z_axis)"))
    rho_tz = np.asarray(data["rho_onaxis_t_z"])
    checks.append(_check("rho_onaxis_t_z_aligned", rho_tz.ndim == 2 and rho_tz.shape[0] == n, list(rho_tz.shape), [n, "Nt"]))
    nonfinite = []
    for key in required["aligned_z_histories"] + list(required["z_leading_arrays"]):
        values = np.asarray(data[key])
        if np.issubdtype(values.dtype, np.number) and not np.all(np.isfinite(values)):
            nonfinite.append(key)
    checks.append(_check("finite_numeric_arrays", not nonfinite, nonfinite, "empty non-finite list"))

    checks.append(_check("operator_mode", str(_scalar(data, "raman_operator_mode")) == "full_isaacs_eq27", str(_scalar(data, "raman_operator_mode")), "full_isaacs_eq27"))
    checks.append(_check("legacy_alpha_zero", np.max(np.abs(np.asarray(data["alpha_R_applied_max_z"], dtype=float))) == 0.0, float(np.max(np.abs(data["alpha_R_applied_max_z"]))), 0.0))
    checks.append(_check("delta_n_semantics", str(_scalar(data, "delta_n_rot_applied_semantics")) == "not_applicable_full_complex_operator", str(_scalar(data, "delta_n_rot_applied_semantics")), "not_applicable_full_complex_operator"))

    feedback = bool(_scalar(data, "raman_operator_feedback_enabled"))
    applied = np.asarray(data["raman_operator_applied"], dtype=bool)
    rhs = np.asarray(data["raman_rhs_l2_norm"], dtype=float)
    raw_ir = np.asarray(data["raman_IR_max_raw"], dtype=float)
    target = np.asarray(data["raman_target_loss_cumulative_J"], dtype=float)
    actual = np.asarray(data["raman_actual_loss_cumulative_J"], dtype=float)
    conv = np.asarray(data["raman_convolution_count_step"], dtype=int)
    substeps = np.asarray(data["raman_operator_substep_count"], dtype=int)
    if job == "on":
        checks.extend([
            _check("feedback_enabled", feedback, feedback, True),
            _check("operator_applied", np.all(applied), bool(np.all(applied)), True),
            _check("raw_IR_nonzero", np.max(raw_ir) > 0.0, float(np.max(raw_ir)), ">0"),
            _check("rhs_nonzero", np.max(rhs) > 0.0, float(np.max(rhs)), ">0"),
            _check("target_loss_nonzero", target[-1] > 0.0, float(target[-1]), ">0 J"),
            _check("actual_loss_nonzero", actual[-1] > 0.0, float(actual[-1]), ">0 J"),
            _check("strang_substeps", np.all(substeps == 2), sorted(set(substeps.tolist())), [2]),
            _check("strang_convolutions", np.all(conv == 4), sorted(set(conv.tolist())), [4]),
        ])
        closure = np.asarray(data["raman_closure_residual_step"], dtype=float)
        p99 = float(np.percentile(closure, 99))
        cumulative = float(np.asarray(data["raman_cumulative_closure_residual"], dtype=float)[-1])
        checks.append(_check("raman_step_closure_p99", p99 < contract["raman_energy_contract"]["per_step_p99_lt"], p99, contract["raman_energy_contract"]["per_step_p99_lt"]))
        checks.append(_check("raman_cumulative_closure", cumulative < contract["raman_energy_contract"]["cumulative_final_lt"], cumulative, contract["raman_energy_contract"]["cumulative_final_lt"]))
    elif job == "off":
        checks.extend([
            _check("feedback_disabled", not feedback, feedback, False),
            _check("operator_not_applied", not np.any(applied), bool(np.any(applied)), False),
            _check("raw_IR_nonzero", np.max(raw_ir) > 0.0, float(np.max(raw_ir)), ">0"),
            _check("target_loss_nonzero", target[-1] > 0.0, float(target[-1]), ">0 J"),
            _check("rhs_zero", np.max(np.abs(rhs)) == 0.0, float(np.max(np.abs(rhs))), 0.0),
            _check("actual_loss_zero", abs(float(actual[-1])) <= 1e-15, float(actual[-1]), "<=1e-15 J"),
            _check("operator_substeps_zero", np.all(substeps == 0), sorted(set(substeps.tolist())), [0]),
            _check("raw_diagnostic_convolution", np.all(conv == 1), sorted(set(conv.tolist())), [1]),
        ])
    else:
        raise ValueError(f"unknown job kind: {job}")

    U = np.asarray(data["U_z"], dtype=float)
    U_step = np.asarray(data["U_step_change_z"], dtype=float)
    U0 = float(U[0] - U_step[0])
    total_dep = np.asarray(data["E_dep_cumulative_z"], dtype=float)
    total_residual = np.abs((U0-U)-total_dep) / max(abs(U0), 1e-30)
    final_total = float(total_residual[-1])
    focus_lo, focus_hi = contract["total_energy_contract"]["near_focus_window_m"]
    focus_mask = (z >= focus_lo) & (z <= focus_hi)
    near_focus = float(np.max(total_residual[focus_mask])) if np.any(focus_mask) else math.inf
    checks.append(_check("total_energy_final", final_total < contract["total_energy_contract"]["final_lt"], final_total, contract["total_energy_contract"]["final_lt"]))
    checks.append(_check("total_energy_near_focus", near_focus < contract["total_energy_contract"]["near_focus_max_lt"], near_focus, contract["total_energy_contract"]["near_focus_max_lt"]))
    passed = all(item["passed"] for item in checks)
    return {
        "schema": "khz_filament.phase8b.diagnostic_audit.v1",
        "job": job,
        "status": "passed" if passed else "failed",
        "record_count": int(n),
        "nominal_record_count": int(contract["record_axis"]["nominal_record_count"]),
        "checks": checks,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--job", choices=("on", "off"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    with np.load(args.npz, allow_pickle=False) as data:
        result = audit(data, contract, args.job)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
