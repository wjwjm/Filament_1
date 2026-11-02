"""Formal scalar analysis rebound from the reviewed 6F implementation."""

from __future__ import annotations

from copy import deepcopy
from types import FunctionType
from typing import Any, Mapping

from tools.cpu_s4_6f_fullsize_parallel import analysis as fullsize_analysis

from .formal_protocol import CANDIDATE, PHASE, matrix, pairings, plan


SCHEMA = "filament.cpu.s4_6i.formal_analysis.v1"


def _bound_analysis() -> FunctionType:
    namespace = dict(fullsize_analysis.analyze.__globals__)
    namespace.update(matrix=matrix, pairings=pairings)
    function = FunctionType(
        fullsize_analysis.analyze.__code__,
        namespace,
        name="formal_fullsize_analyze",
        argdefs=fullsize_analysis.analyze.__defaults__,
        closure=fullsize_analysis.analyze.__closure__,
    )
    function.__kwdefaults__ = fullsize_analysis.analyze.__kwdefaults__
    return function


def _require_service(row: Mapping[str, Any], key: str, case: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise RuntimeError(f"S46I_FORMAL_CLOCK_MISSING:{case}:{key}")
    return float(value)


def analyze(cases: Mapping[str, Mapping[str, Any]], candidate: str = CANDIDATE) -> dict[str, Any]:
    """Apply unchanged 6F gates and add non-gating service-time diagnostics."""

    base = _bound_analysis()(cases, candidate)
    result = deepcopy(base)
    pair_rows = result.get("pairs", [])
    for row in pair_rows:
        reference = cases[row["reference"]]
        selected = cases[row["candidate"]]
        reference_service = _require_service(reference, "propagation_service_s", row["reference"])
        selected_service = _require_service(selected, "propagation_service_s", row["candidate"])
        reference_application = _require_service(reference, "application_service_s", row["reference"])
        selected_application = _require_service(selected, "application_service_s", row["candidate"])
        row.update(
            {
                "propagation_reference_s": reference_service,
                "propagation_selected_s": selected_service,
                "propagation_saved_s": reference_service - selected_service,
                # Keep the selected case's service clock under the concise
                # name used by entry receipts; reference/saved values remain
                # explicit beside it for the pairwise report.
                "propagation_service_s": selected_service,
                "application_reference_s": reference_application,
                "application_selected_s": selected_application,
                "application_saved_s": reference_application - selected_application,
            }
        )

    block_application: list[dict[str, Any]] = []
    for block in ("K1", "K2"):
        members = [row for row in pair_rows if row.get("block") == block]
        arithmetic_saved = sum(float(row["application_saved_s"]) for row in members)
        original_block = next(row for row in result["blocks"] if row.get("block") == block)
        block_application.append(
            {
                "block": block,
                "arithmetic_saved_s": arithmetic_saved,
                "arithmetic_saved_positive": arithmetic_saved > 0.0,
                "original_application_log_gain": float(original_block["application_gain"]),
                "original_application_log_gain_positive": float(original_block["application_gain"]) > 0.0,
            }
        )
    full_application_saved = sum(float(row["application_saved_s"]) for row in pair_rows)
    application_supported = bool(
        full_application_saved > 0.0
        and all(row["arithmetic_saved_positive"] for row in block_application)
        and all(row["original_application_log_gain_positive"] for row in block_application)
    )
    result.update(
        {
            "schema": SCHEMA,
            "formal_phase": PHASE,
            "candidate": CANDIDATE,
            "original_gates_unchanged": True,
            "formal_plan_schema": plan()["schema"],
            "full_application_net_saved_s": full_application_saved,
            "complete_application_saved_s": full_application_saved,
            "block_application": block_application,
            "application_speedup_supported": application_supported,
            "application_speedup_support_basis": {
                "requires_full_application_net_saved_positive": True,
                "requires_both_block_arithmetic_savings_positive": True,
                "requires_both_original_block_log_gains_positive": True,
                "does_not_replace_original_application_guard": True,
            },
            "full_propagation_service_saved_s": sum(float(row["propagation_saved_s"]) for row in pair_rows),
        }
    )
    return result


__all__ = ["SCHEMA", "analyze"]
