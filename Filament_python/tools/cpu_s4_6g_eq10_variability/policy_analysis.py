"""Pure-stdlib frozen memory-policy plan and scalar crossover analysis."""

from __future__ import annotations

import math
from typing import Any, Mapping


_ORDER = (
    ("QUAL_R2", "R", 2, "I", "QUALIFICATION"),
    ("QUAL_X2", "X", 2, "I", "QUALIFICATION"),
    ("CTRL_EARLY_R1", "R", 8, "D", "RR_CONTROL"),
    ("CTRL_EARLY_R2", "R", 8, "D", "RR_CONTROL"),
    ("K1_R1", "R", 8, "D", "FORMAL"),
    ("K1_X1", "X", 8, "D", "FORMAL"),
    ("K1_X2", "X", 8, "I", "FORMAL"),
    ("K1_R2", "R", 8, "I", "FORMAL"),
    ("K2_R1", "R", 8, "I", "FORMAL"),
    ("K2_X1", "X", 8, "I", "FORMAL"),
    ("K2_X2", "X", 8, "D", "FORMAL"),
    ("K2_R2", "R", 8, "D", "FORMAL"),
    ("CTRL_LATE_R1", "R", 8, "D", "RR_CONTROL"),
    ("CTRL_LATE_R2", "R", 8, "D", "RR_CONTROL"),
)

_METRICS = (
    "later_body",
    "propagation",
    "application",
    "launch_scope_s",
    "process_service_s",
    "initialization_total_s",
)


def _matrix() -> list[dict[str, Any]]:
    return [
        {"case": case, "role": role, "intervals": intervals, "policy": policy, "kind": kind}
        for case, role, intervals, policy, kind in _ORDER
    ]


def policy_plan() -> dict[str, Any]:
    """Return the only permitted 14-window memory-policy crossover plan."""

    return {
        "schema": "filament.cpu_s46g.policy_plan.v1",
        "task": "CPU-S4-6G-MEMORY-POLICY-CROSSOVER-20261101",
        "candidate": "EQ10_P48",
        "shape": [384, 351, 301],
        "dtype": "complex128",
        "expected_mems": [0, 1, 2, 3],
        "policies": {
            "D": {"label": "default", "argv_prefix": []},
            "I": {"label": "interleave", "argv_prefix": ["/usr/bin/numactl", "--interleave=all"]},
        },
        "matrix": _matrix(),
        "order": [row[0] for row in _ORDER],
        "serial": True,
        "new_process_each_window": True,
        "propagation_warmup_intervals": 0,
        "observer_period_s": 5.0,
        "policy_snapshot_period_s": 15.0,
        "policy_plan_source": "PRIVATE_POLICY_MATRIX_NOT_S46F_MATRIX",
        "diagnostic_only": True,
        "default": "I0",
        "production_qualified": False,
    }


def _finite(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError("POLICY_NONFINITE_CLOCK")
    return float(value)


def _metric_values(case: Mapping[str, Any], launch: Mapping[str, Any]) -> dict[str, float]:
    clocks = launch.get("clock_metrics")
    if not isinstance(clocks, Mapping):
        raise ValueError("POLICY_PROPAGATION_CLOCK_MISSING")
    return {
        "later_body": _finite(case.get("later_body_s")),
        "propagation": _finite(clocks.get("propagation_s")),
        "application": _finite(case.get("application_service_s")),
        "launch_scope_s": _finite(launch.get("launch_scope_s")),
        "process_service_s": _finite(launch.get("process_service_s")),
        "initialization_total_s": _finite(clocks.get("initialization_total_s")),
    }


def _log_ratio(selected: float, reference: float) -> float:
    return math.log(selected / reference)


def _effect(selected: Mapping[str, float], reference: Mapping[str, float]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for metric in _METRICS:
        log_ratio = _log_ratio(selected[metric], reference[metric])
        output[metric] = {
            "I_s": selected[metric],
            "D_s": reference[metric],
            "ratio_I_over_D": selected[metric] / reference[metric],
            "log_ratio_I_over_D": log_ratio,
            "gain_I_over_D": 1.0 - math.exp(log_ratio),
            "saved_s_I_over_D": reference[metric] - selected[metric],
        }
    return output


def _classify_log_effect(value: float, drift: float) -> str:
    if value < 0 and abs(value) > drift:
        return "OBSERVED_POLICY_FASTER"
    if value > 0 and abs(value) > drift:
        return "OBSERVED_POLICY_SLOWER"
    return "NO_DETERMINATE_POLICY_SIGNAL"


def analyze_policy(
    accepted: Mapping[str, Mapping[str, Any]],
    launch_receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Analyze D/I policy effects without importing NumPy or old 6F analysis."""

    plan = policy_plan()
    expected = plan["matrix"]
    if list(accepted) != plan["order"] or list(launch_receipts) != plan["order"]:
        raise ValueError("POLICY_COMPLETE_ORDERED_FOURTEEN_REQUIRED")
    for row in expected:
        case = row["case"]
        result = accepted[case]
        launch = launch_receipts[case]
        if result.get("role") != row["role"] or int(result.get("intervals")) != row["intervals"]:
            raise ValueError("POLICY_CASE_IDENTITY:" + case)
        if result.get("exact_gate_pass") is not True:
            raise ValueError("POLICY_EXACT_REQUIRED:" + case)
        if launch.get("policy") != row["policy"]:
            raise ValueError("POLICY_ORDER_BINDING:" + case)
        if launch.get("policy_evidence", {}).get("valid_body_snapshot") is not True:
            raise ValueError("POLICY_BODY_EVIDENCE_REQUIRED:" + case)

    values = {case: _metric_values(accepted[case], launch_receipts[case]) for case in plan["order"]}
    rr_drift: dict[str, float] = {}
    rr_drift_table: dict[str, list[dict[str, Any]]] = {}
    for metric in _METRICS:
        rows = []
        for first, second in (("CTRL_EARLY_R1", "CTRL_EARLY_R2"), ("CTRL_LATE_R1", "CTRL_LATE_R2")):
            log_ratio = _log_ratio(values[second][metric], values[first][metric])
            rows.append({
                "pair": f"{first}/{second}",
                "first_case": first,
                "second_case": second,
                "first_s": values[first][metric],
                "second_s": values[second][metric],
                "ratio_second_over_first": values[second][metric] / values[first][metric],
                "log_ratio_second_over_first": log_ratio,
                "abs_log_drift": abs(log_ratio),
            })
        rr_drift_table[metric] = rows
        rr_drift[metric] = max(row["abs_log_drift"] for row in rows)

    role_pairs = {
        "K1": {"D": {"R": "K1_R1", "X": "K1_X1"}, "I": {"X": "K1_X2", "R": "K1_R2"}},
        "K2": {"I": {"R": "K2_R1", "X": "K2_X1"}, "D": {"X": "K2_X2", "R": "K2_R2"}},
    }
    policy_pairs: list[dict[str, Any]] = []
    block_effects: dict[str, dict[str, float]] = {}
    role_effects: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    policy_pair_table: list[dict[str, Any]] = []
    for block, policies in role_pairs.items():
        role_effects[block] = {}
        formal_order = [row["case"] for row in expected if row["kind"] == "FORMAL" and row["case"].startswith(block + "_")]
        pair_rows: list[dict[str, Any]] = []
        for role in ("R", "X"):
            default_case = policies["D"][role]
            interleave_case = policies["I"][role]
            effect = _effect(values[interleave_case], values[default_case])
            role_effects[block][role] = effect
            pair_rows.append({
                "role": role,
                "default": {"case": default_case, "order_index": plan["order"].index(default_case) + 1, "metrics": values[default_case]},
                "interleave": {"case": interleave_case, "order_index": plan["order"].index(interleave_case) + 1, "metrics": values[interleave_case]},
                "effect_I_over_D": effect,
                "actual_order": formal_order,
            })
            for policy, case in (("D", default_case), ("I", interleave_case)):
                policy_pairs.append({
                    "block": block,
                    "policy": policy,
                    "role": role,
                    "case": case,
                    "counterpart_case": interleave_case if policy == "D" else default_case,
                    "order_index": plan["order"].index(case) + 1,
                    "metrics": values[case],
                    "effect_I_over_D": effect,
                    "actual_order": formal_order,
                })
        policy_pair_table.append({"block": block, "actual_order": formal_order, "roles": pair_rows})
        # This is retained solely as a descriptive balanced summary. It is not
        # used by any classification or acceptance decision.
        block_effects[block] = {
            metric: sum(role_effects[block][role][metric]["log_ratio_I_over_D"] for role in ("R", "X")) / 2.0
            for metric in _METRICS
        }

    role_metric_classification: dict[str, dict[str, dict[str, str]]] = {}
    for role in ("R", "X"):
        role_metric_classification[role] = {"K1": {}, "K2": {}, "overall": {}}
        for block in ("K1", "K2"):
            for metric in _METRICS:
                role_metric_classification[role][block][metric] = _classify_log_effect(
                    role_effects[block][role][metric]["log_ratio_I_over_D"], rr_drift[metric]
                )
        for metric in _METRICS:
            block_labels = [role_metric_classification[role][block][metric] for block in ("K1", "K2")]
            role_metric_classification[role]["overall"][metric] = (
                block_labels[0] if block_labels[0] == block_labels[1] and block_labels[0] != "NO_DETERMINATE_POLICY_SIGNAL"
                else "NO_DETERMINATE_POLICY_SIGNAL"
            )

    metric_classification = {}
    for metric in _METRICS:
        role_labels = [role_metric_classification[role]["overall"][metric] for role in ("R", "X")]
        metric_classification[metric] = (
            role_labels[0] if role_labels[0] == role_labels[1] and role_labels[0] != "NO_DETERMINATE_POLICY_SIGNAL"
            else "NO_DETERMINATE_POLICY_SIGNAL"
        )
    decisive = {value for value in metric_classification.values() if value != "NO_DETERMINATE_POLICY_SIGNAL"}
    classification = decisive.pop() if len(decisive) == 1 and all(value in decisive for value in metric_classification.values()) else "NO_DETERMINATE_POLICY_SIGNAL"
    role_average_effects = {
        role: {
            metric: sum(role_effects[block][role][metric]["log_ratio_I_over_D"] for block in ("K1", "K2")) / 2.0
            for metric in _METRICS
        }
        for role in ("R", "X")
    }
    return {
        "schema": "filament.cpu_s46g.policy_result.v1",
        "task": plan["task"],
        "candidate": "EQ10_P48",
        "diagnostic_kind": "MEMORY_POLICY_CROSSOVER",
        "classification": classification,
        "diagnostic_only": True,
        "default": "I0",
        "production_qualified": False,
        "policy_plan": plan,
        "rr_drift_max_abs_log": rr_drift,
        "rr_drift_table": rr_drift_table,
        "policy_effects_by_block": block_effects,
        "policy_effects_by_role": role_effects,
        "policy_effects_by_role_average": role_average_effects,
        "policy_pairs": policy_pairs,
        "policy_pair_table": policy_pair_table,
        "role_metric_classification": role_metric_classification,
        "metric_classification": metric_classification,
        "accepted_case_count": len(accepted),
        "launch_receipt_count": len(launch_receipts),
        "interpretation": "Allocation-local D/I policy effect only; not an old anomaly root-cause claim or production acceptance.",
    }


__all__ = ["analyze_policy", "policy_plan"]
