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


def _archive_ulp(value, dtype) -> np.ndarray:
    """Return positive local ULP sizes for a floating-point archive dtype."""
    archive_dtype = np.dtype(dtype)
    if not np.issubdtype(archive_dtype, np.floating):
        archive_dtype = np.dtype(np.float64)
    values = np.asarray(value, dtype=archive_dtype)
    return np.abs(np.spacing(values)).astype(np.float64)


def coordinate_audit(z, dz, contract: dict) -> list[dict]:
    """Separate accepted-step integrity from archive representation precision.

    The propagation archive may be float32 even though all audit reductions use
    float64.  The accepted-step sum therefore has a provable quantization bound
    based on the stored increments, while the archived coordinate axis is tested
    in its own dtype using an explicit ULP budget.
    """
    z_raw = np.asarray(z)
    dz_raw = np.asarray(dz)
    z64 = np.asarray(z_raw, dtype=np.float64)
    dz64 = np.asarray(dz_raw, dtype=np.float64)
    n = int(z64.size)
    record_axis = contract["record_axis"]
    target = float(contract["fixed_coordinates"]["z_final_m"])
    nominal = int(record_axis["nominal_record_count"])
    adaptive = bool(record_axis["nominal_derivation"]["adaptive_substep_enabled"])

    checks = []
    expected_count = f"=={nominal}" if not adaptive else f">={nominal} (adaptive accepted-step history)"
    count_ok = n == nominal if not adaptive else n >= nominal
    checks.append(_check("execution_record_count", count_ok, n, expected_count))
    checks.append(_check("z_strictly_increasing", n == 1 or np.all(np.diff(z64) > 0.0), bool(n == 1 or np.all(np.diff(z64) > 0.0)), True))
    checks.append(_check("positive_dz", np.all(dz64 > 0.0), float(np.min(dz64)) if dz64.size else math.nan, ">0"))

    execution_distance = math.fsum(dz64.tolist())
    # Each stored increment has at most half a ULP rounding error relative to
    # its accepted high-precision value.  This bound remains strict enough to
    # reject a missing 50/100 um step while accepting the only precision that
    # is available for legacy float32 archives.
    execution_bound = max(2e-12, float(np.sum(0.5 * _archive_ulp(dz_raw, dz_raw.dtype))))
    execution_error = abs(execution_distance - target)
    checks.append(_check(
        "execution_distance_reaches_target",
        execution_error <= execution_bound,
        {
            "sum_dz_float64_m": execution_distance,
            "target_m": target,
            "absolute_error_m": execution_error,
            "quantization_bound_m": execution_bound,
        },
        "absolute_error_m <= accumulated half-ULP dz quantization bound",
    ))

    cumulative = np.cumsum(dz64, dtype=np.float64)
    axis_ulp = _archive_ulp(z_raw, z_raw.dtype)
    axis_error = np.abs(z64 - cumulative)
    axis_ulp_error = axis_error / np.maximum(axis_ulp, np.finfo(np.float64).tiny)
    axis_budget = int(contract["fixed_coordinates"]["archive_z_axis_reconstruction_ulp_budget"])
    checks.append(_check(
        "archive_axis_reconstruction_ulp",
        bool(np.all(axis_ulp_error <= axis_budget)),
        {
            "max_absolute_error_m": float(np.max(axis_error)) if n else math.nan,
            "max_error_ulp": float(np.max(axis_ulp_error)) if n else math.nan,
            "archive_dtype": str(z_raw.dtype),
        },
        f"<= {axis_budget} ULP at every archived z_axis sample",
    ))

    final_ulp = float(_archive_ulp(np.asarray(target), z_raw.dtype))
    final_error = abs(float(z64[-1]) - target) if n else math.inf
    final_error_ulp = final_error / max(final_ulp, np.finfo(np.float64).tiny)
    final_budget = int(contract["fixed_coordinates"]["archive_z_final_ulp_budget"])
    checks.append(_check(
        "archive_z_final_ulp",
        final_error_ulp <= final_budget,
        {
            "z_final_archived_m": float(z64[-1]) if n else math.nan,
            "target_m": target,
            "absolute_error_m": final_error,
            "ulp_m": final_ulp,
            "error_ulp": final_error_ulp,
            "archive_dtype": str(z_raw.dtype),
        },
        f"<= {final_budget} ULP at z_final in archive dtype",
    ))
    return checks


def audit(data, contract: dict, job: str) -> dict:
    checks = []
    required = contract["required_fields"]
    required_keys = {"z_axis", *required["aligned_z_histories"], *required["z_leading_arrays"], *required["scalar_or_text"]}
    missing = sorted(required_keys.difference(data.files))
    checks.append(_check("required_fields", not missing, missing, "empty missing-field list"))
    if missing:
        return {"job": job, "status": "failed", "checks": checks}

    z = np.asarray(data["z_axis"])
    dz = np.asarray(data["dz_used_z"])
    n = z.size
    checks.append(_check("nonempty", n > 0, n, ">0"))
    checks.extend(coordinate_audit(z, dz, contract))
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
    z64 = np.asarray(z, dtype=float)
    focus_mask = (z64 >= focus_lo) & (z64 <= focus_hi)
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
