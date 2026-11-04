"""Formal scalar analysis with the S4-6J positive-net-savings gates."""

from __future__ import annotations

import math
import statistics
from copy import deepcopy
from types import FunctionType
from typing import Any, Mapping

from tools.cpu_s4_6f_fullsize_parallel import analysis as fullsize_analysis

from .alpha_formal_protocol import CANDIDATE, PHASE, matrix, pairings, plan


SCHEMA = "filament.cpu_s4_6j.alpha_formal_analysis.v1"


def _bound_analysis() -> FunctionType:
    namespace = dict(fullsize_analysis.analyze.__globals__)
    namespace.update(matrix=matrix, pairings=pairings)
    function = FunctionType(
        fullsize_analysis.analyze.__code__,
        namespace,
        name="alpha_formal_fullsize_analyze",
        argdefs=fullsize_analysis.analyze.__defaults__,
        closure=fullsize_analysis.analyze.__closure__,
    )
    function.__kwdefaults__ = fullsize_analysis.analyze.__kwdefaults__
    return function


def _positive_clock(row: Mapping[str, Any], key: str, case: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"S46J_FORMAL_CLOCK_MISSING:{case}:{key}")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise RuntimeError(f"S46J_FORMAL_CLOCK_NONPOSITIVE:{case}:{key}")
    return float(value)


def analyze(cases: Mapping[str, Mapping[str, Any]], candidate: str = CANDIDATE) -> dict[str, Any]:
    base = _bound_analysis()(cases, candidate)
    result = deepcopy(base)
    pairs_out = result.get("pairs", [])
    for row in pairs_out:
        reference = cases[row["reference"]]
        selected = cases[row["candidate"]]
        row.update(
            {
                "propagation_reference_s": _positive_clock(reference, "propagation_service_s", row["reference"]),
                "propagation_selected_s": _positive_clock(selected, "propagation_service_s", row["candidate"]),
                "application_reference_s": _positive_clock(reference, "application_service_s", row["reference"]),
                "application_selected_s": _positive_clock(selected, "application_service_s", row["candidate"]),
            }
        )
        row["propagation_saved_s"] = row["propagation_reference_s"] - row["propagation_selected_s"]
        row["application_saved_s"] = row["application_reference_s"] - row["application_selected_s"]
        row["management_reference_s"] = float(reference.get("source_factory_preparation_s", 0.0))
        row["management_selected_s"] = float(selected.get("source_factory_preparation_s", 0.0))

    block_application: list[dict[str, Any]] = []
    for block in ("K1", "K2"):
        members = [row for row in pairs_out if row.get("block") == block]
        saved = sum(float(row["application_saved_s"]) for row in members)
        body = next(row for row in result["blocks"] if row.get("block") == block)
        block_application.append(
            {
                "block": block,
                "net_saved_s": saved,
                "net_saved_positive": saved > 0.0,
                "body_gain": float(body["body_gain"]),
                "POST_gain": float(body["POST_gain"]),
            }
        )
    application_saved_values = [float(row["application_saved_s"]) for row in pairs_out]
    application_saved_median = statistics.median(application_saved_values)
    full_application_saved = sum(application_saved_values)
    old_gates = dict(result.get("gates", {}))
    gates = {
        "science_exact": old_gates.get("science_exact") is True,
        "body_paired_median_strictly_positive": float(result["body_paired_median_gain"]) > 0.0,
        "both_blocks_positive": all(row["body_gain"] > 0.0 for row in result["blocks"]),
        "both_orders_positive": all(row["body_gain"] > 0.0 for row in result["orders"]),
        "body_exceeds_observed_RR": old_gates.get("body_exceeds_observed_RR") is True,
        "POST_both_blocks_positive": all(row["POST_gain"] > 0.0 for row in result["blocks"]),
        "application_guard": old_gates.get("application_guard") is True,
        "application_saved_median_strictly_positive": application_saved_median > 0.0,
        "application_block_net_saved_positive": all(row["net_saved_positive"] for row in block_application),
        "first_POST_domains": old_gates.get("first_POST_domains") is True,
    }
    passed = all(gates.values())
    if passed:
        status = "FULLSIZE_ALPHA_PERFORMANCE_GATES_PASS_MANUAL_REVIEW_PENDING"
        classification = "PASS_MANUAL_REVIEW_PENDING"
    elif not gates["body_paired_median_strictly_positive"]:
        status = "NO_POSITIVE_FULL_BODY_GAIN"
        classification = "NO_POSITIVE_FULL_BODY_GAIN"
    elif not (gates["application_guard"] and gates["application_saved_median_strictly_positive"] and gates["application_block_net_saved_positive"]):
        status = "FULLSIZE_ALPHA_APPLICATION_GUARD_FAILURE"
        classification = "FULLSIZE_ALPHA_APPLICATION_GUARD_FAILURE"
    else:
        status = "FULLSIZE_ALPHA_INCONCLUSIVE_VARIABILITY_OR_ORDER"
        classification = "FULLSIZE_ALPHA_INCONCLUSIVE_VARIABILITY_OR_ORDER"
    result.update(
        {
            "schema": SCHEMA,
            "formal_phase": PHASE,
            "candidate": CANDIDATE,
            "gates": gates,
            "status": status,
            "classification": classification,
            "body_paired_median_gain": float(result["body_paired_median_gain"]),
            "application_saved_s_by_pair": application_saved_values,
            "application_saved_s_median": application_saved_median,
            "complete_application_saved_s": full_application_saved,
            "block_application": block_application,
            "full_propagation_service_saved_s": sum(float(row["propagation_saved_s"]) for row in pairs_out),
            "formal_plan_schema": plan()["schema"],
            "service_breakdown": {
                "fields": [
                    "propagation_service_s",
                    "later_body_s",
                    "application_service_s",
                    "source_factory_preparation_s",
                    "initialization_total_before_propagation_s",
                    "first_POST_latency_s",
                    "pool_finalization_and_management",
                ],
                "pair_count": len(pairs_out),
            },
            "original_gates_recomputed": True,
            "historical_cases_pooled": 0,
            "production_qualified": False,
            "default": "I0",
            "review_gate": "MANUAL_REVIEW_PENDING",
        }
    )
    return result


__all__ = ["SCHEMA", "analyze"]
