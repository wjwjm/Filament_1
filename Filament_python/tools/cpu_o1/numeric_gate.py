"""Fail-closed numerical admission for local CPU-Q1 window comparisons."""
from __future__ import annotations

import json
import os
from pathlib import Path


NAN_PLACEHOLDER_FIELDS = frozenset({
    "energy_step_start_J",
    "energy_after_linear_half1_J",
    "energy_after_raman_pre_J",
    "energy_after_nonraman_J",
    "energy_after_raman_post_J",
    "energy_after_linear_half2_J",
})
TIMING_FIELDS = frozenset({
    "raman_operator_walltime_step_s",
    "linear_walltime_step_s",
    "ionization_walltime_step_s",
    "total_walltime_step_s",
})


def classify_comparison(row: dict) -> dict:
    """Classify one numeric field without inferring scientific tolerance."""
    if not isinstance(row, dict) or not isinstance(row.get("metrics"), dict):
        return {"classification": "TOOL_FAILURE", "proceed": False,
                "reason": "comparison_record_or_metrics_missing"}
    name = str(row.get("field_name", ""))
    scope = str(row.get("scope", "science"))
    metrics = row["metrics"]
    if scope == "timing" and name not in TIMING_FIELDS:
        return {"field_name": name, "classification": "TOOL_FAILURE",
                "proceed": False, "reason": "unapproved_timing_exclusion_name"}
    if name in TIMING_FIELDS:
        return {"field_name": name, "classification": "TIMING_EXCLUDED",
                "proceed": True, "reason": "frozen_timing_exclusion"}

    required = ("reference_shape", "candidate_shape", "reference_dtype",
                "candidate_dtype", "dtype_equal", "canonical_exact",
                "bitwise_equal", "all_finite")
    if any(key not in metrics for key in required):
        return {"field_name": name, "classification": "TOOL_FAILURE",
                "proceed": False, "reason": "comparison_metadata_missing"}
    reference_shape, candidate_shape = metrics["reference_shape"], metrics["candidate_shape"]
    if (not isinstance(reference_shape, list) or not isinstance(candidate_shape, list)
            or any(isinstance(size, bool) or not isinstance(size, int) or size < 0
                   for size in reference_shape + candidate_shape)
            or not isinstance(metrics["reference_dtype"], str)
            or not metrics["reference_dtype"]
            or not isinstance(metrics["candidate_dtype"], str)
            or not metrics["candidate_dtype"]
            or not isinstance(metrics["dtype_equal"], bool)
            or not isinstance(metrics["canonical_exact"], bool)
            or not isinstance(metrics["bitwise_equal"], bool)
            or not isinstance(metrics["all_finite"], bool)):
        return {"field_name": name, "classification": "TOOL_FAILURE",
                "proceed": False, "reason": "comparison_metadata_invalid"}
    try:
        import numpy as np
        reference_dtype = np.dtype(metrics["reference_dtype"])
        candidate_dtype = np.dtype(metrics["candidate_dtype"])
        if reference_dtype.kind not in "biufc" or candidate_dtype.kind not in "biufc":
            raise TypeError("unsupported dtype kind")
    except (TypeError, ValueError):
        return {"field_name": name, "classification": "TOOL_FAILURE",
                "proceed": False, "reason": "comparison_dtype_metadata_invalid"}
    dtype_strings_equal = metrics["reference_dtype"] == metrics["candidate_dtype"]
    if metrics["dtype_equal"] != dtype_strings_equal:
        return {"field_name": name, "classification": "TOOL_FAILURE",
                "proceed": False, "reason": "comparison_dtype_metadata_inconsistent"}
    if reference_shape != candidate_shape or not metrics["dtype_equal"]:
        return {"field_name": name,
                "classification": "NONFINITE_OR_TOPOLOGY_FAILURE",
                "proceed": False, "reason": "shape_or_dtype_contract_mismatch"}

    if metrics.get("shape_mismatch") or metrics.get("topology_equal") is False:
        return {"field_name": name,
                "classification": "NONFINITE_OR_TOPOLOGY_FAILURE",
                "proceed": False, "reason": "topology_mismatch"}

    placeholder_allowed = (
        scope == "diagnostics"
        and name in NAN_PLACEHOLDER_FIELDS
        and metrics.get("reference_dtype") == "float64"
        and metrics.get("candidate_dtype") == "float64"
        and metrics.get("reference_all_nan") is True
        and metrics.get("candidate_all_nan") is True
        and metrics.get("reference_operand_nonfinite_count") == metrics.get("element_count")
        and metrics.get("candidate_operand_nonfinite_count") == metrics.get("element_count")
        and metrics.get("reference_operand_infinity_count") == 0
        and metrics.get("candidate_operand_infinity_count") == 0
        and metrics.get("same_nonfinite_value_count") == metrics.get("element_count")
        and metrics.get("nonfinite_mismatch_count") == 0
        and metrics.get("canonical_exact") is True
        and metrics.get("bitwise_equal") is True
    )
    overflow_counts = (
        "finite_subtraction_overflow_count",
        "reference_magnitude_overflow_count",
        "candidate_magnitude_overflow_count",
        "difference_magnitude_overflow_count",
        "normalization_overflow_count",
    )
    has_overflow = any(int(metrics.get(name, 0) or 0) > 0 for name in overflow_counts)
    if has_overflow:
        return {"field_name": name,
                "classification": "NONFINITE_OR_TOPOLOGY_FAILURE",
                "proceed": False, "reason": "numeric_overflow"}
    if metrics.get("all_finite") is False and not placeholder_allowed:
        return {"field_name": name,
                "classification": "NONFINITE_OR_TOPOLOGY_FAILURE",
                "proceed": False, "reason": "nonfinite_values_outside_exact_whitelist"}

    exact = bool(metrics["canonical_exact"] and metrics["bitwise_equal"])
    if exact:
        return {"field_name": name, "classification": "CURRENT_LINUX_LOCAL_EXACT",
                "proceed": True,
                "reason": "exact_whitelisted_nan_placeholder" if placeholder_allowed
                else "canonical_and_bitwise_exact"}

    if metrics.get("all_finite") is not True:
        return {"field_name": name,
                "classification": "NONFINITE_OR_TOPOLOGY_FAILURE",
                "proceed": False, "reason": "nonfinite_comparison_not_exact"}
    if metrics.get("value_equal") is True:
        reason = "finite_nonexact_signed_zero_or_representation_difference"
        classification = "FINITE_NONEXACT_DIFFERENCE"
    else:
        reason = "finite_nonzero_difference"
        classification = "FINITE_NONZERO_DIFFERENCE"
    return {"field_name": name, "classification": classification,
            "proceed": False, "reason": reason}


def build_numeric_gate(*, identity_checks: dict[str, bool],
                       linux_environment_eligible: bool,
                       comparisons: list[dict] | None = None,
                       tool_error: str | None = None) -> dict:
    """Build a strict-JSON gate receipt and a matrix continuation decision."""
    if tool_error is not None:
        classification = "TOOL_FAILURE"
        rows = []
        reasons = [str(tool_error)]
    else:
        rows = [classify_comparison(row) for row in (comparisons or [])]
        reasons = [row["reason"] for row in rows if not row.get("proceed")]
        if any(row["classification"] == "TOOL_FAILURE" for row in rows):
            classification = "TOOL_FAILURE"
        elif any(not row.get("proceed") for row in rows):
            priority = (
                "NONFINITE_OR_TOPOLOGY_FAILURE",
                "FINITE_NONZERO_DIFFERENCE",
                "FINITE_NONEXACT_DIFFERENCE",
            )
            classification = next(
                (kind for kind in priority
                 if any(row["classification"] == kind for row in rows)),
                "NONFINITE_OR_TOPOLOGY_FAILURE",
            )
        elif any(not value for value in identity_checks.values()):
            classification = "IDENTITY_FAIL"
            reasons.append("identity_check_failed")
        elif not linux_environment_eligible:
            classification = "IDENTITY_FAIL"
            reasons.append("linux_environment_not_qualified")
        elif any(row.get("classification") != "TIMING_EXCLUDED" for row in rows):
            classification = "CURRENT_LINUX_LOCAL_EXACT"
        else:
            classification = "IDENTITY_PASS"
    identity_pass = bool(identity_checks) and all(bool(value) for value in identity_checks.values())
    eligible = bool(identity_pass and linux_environment_eligible)
    proceed = bool(eligible and classification in {
        "IDENTITY_PASS", "CURRENT_LINUX_LOCAL_EXACT",
    })
    return {
        "schema": "filament.cpu_q1.numeric_gate.v1",
        "classification": classification,
        "identity_status": "IDENTITY_PASS" if identity_pass else "IDENTITY_FAIL",
        "linux_environment_eligible": bool(linux_environment_eligible),
        "eligible": eligible,
        "proceed_to_next_identity": proceed,
        "stop_reasons": reasons,
        "comparisons": rows,
        "production_qualification": "NOT_PRODUCTION_QUALIFIED",
    }


def write_numeric_gate(path: str | os.PathLike, receipt: dict) -> None:
    """Write a new gate receipt without replacing prior evidence."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, ensure_ascii=False, sort_keys=True,
                         allow_nan=False, indent=2) + "\n"
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
