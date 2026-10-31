"""Fixed scalar C14 analysis; no numerical operators or old samples."""

from __future__ import annotations

import math
import statistics

from .c_formal14_protocol import MATRIX, PAIRS, RESULT_SCHEMA


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def _log_ratio(reference: dict, candidate: dict, key: str) -> float:
    left = reference.get(key)
    right = candidate.get(key)
    _require(type(left) in (int, float) and type(right) in (int, float), "C14_FINITE_CLOCK_TYPE:" + key)
    _require(math.isfinite(left) and math.isfinite(right) and min(left, right) > 0.0, "C14_FINITE_POSITIVE_CLOCK:" + key)
    return math.log(right / left)


def analyze(cases: dict[str, dict]) -> dict:
    expected_names = [row[0] for row in MATRIX]
    _require(list(cases) == expected_names, "C14_COMPLETE_ORDERED_FOURTEEN_CASES")
    for name, role, intervals, _kind in MATRIX:
        row = cases[name]
        _require(row.get("role") == role and int(row.get("intervals", -1)) == intervals, "C14_FIXED_CASE_IDENTITY")
        _require(row.get("exact_gate_pass") is True, "C14_ALL_CASES_EXACT_BEFORE_PERFORMANCE")
    metrics = {"body": "later_body_s", "POST": "post_gap_median_s", "application": "application_service_s"}
    pairs = []
    for name, reference_name, candidate_name, order in PAIRS:
        reference = cases[reference_name]
        candidate = cases[candidate_name]
        row = dict(pair=name, block=name[:2], reference=reference_name, candidate=candidate_name, order=order)
        for label, key in metrics.items():
            value = _log_ratio(reference, candidate, key)
            row.update(
                {
                    label + "_log_ratio": value,
                    label + "_gain": 1.0 - math.exp(value),
                    label + "_R_s": reference[key],
                    label + "_C_s": candidate[key],
                    label + "_saved_s": reference[key] - candidate[key],
                }
            )
        row["application_ratio"] = candidate["application_service_s"] / reference["application_service_s"]
        pairs.append(row)
    blocks = []
    for block in ("K1", "K2"):
        members = [row for row in pairs if row["block"] == block]
        _require(len(members) == 2, "C14_BLOCK_PAIR_COUNT")
        record = dict(block=block)
        for label in metrics:
            value = statistics.mean(row[label + "_log_ratio"] for row in members)
            record.update({label + "_log_ratio": value, label + "_gain": 1.0 - math.exp(value)})
        blocks.append(record)
    controls = []
    for label in ("EARLY", "LATE"):
        first = cases[f"CTRL_{label}_R1"]
        second = cases[f"CTRL_{label}_R2"]
        controls.append(
            dict(
                control=label,
                **{
                    key + "_absolute_log_variation": abs(_log_ratio(first, second, metric))
                    for key, metric in metrics.items()
                },
            )
        )
    observed_noise = max(row["body_absolute_log_variation"] for row in controls)
    orders = []
    for order in ("RC", "CR"):
        value = statistics.mean(row["body_log_ratio"] for row in pairs if row["order"] == order)
        orders.append(dict(order=order, body_log_ratio=value, body_gain=1.0 - math.exp(value)))
    body_log = statistics.mean(row["body_log_ratio"] for row in blocks)
    median_gain = statistics.median(row["body_gain"] for row in pairs)
    gates = dict(
        science_exact=True,
        body_paired_median_at_least_3percent=median_gain >= 0.03,
        both_blocks_positive=all(row["body_gain"] > 0.0 for row in blocks),
        both_orders_positive=all(row["body_gain"] > 0.0 for row in orders),
        body_exceeds_observed_RR=-body_log > observed_noise,
        POST_both_blocks_positive=all(row["POST_gain"] > 0.0 for row in blocks),
        application_guard=statistics.median(row["application_ratio"] for row in pairs) <= 1.02 and all(row["application_ratio"] <= 1.05 for row in pairs),
        first_POST_domains=all(row.get("clock_domain_pass") is True for row in cases.values()),
    )
    if all(gates.values()):
        status = "C14_SMALLGRID_PERFORMANCE_GATES_PASS_MANUAL_REVIEW_PENDING"
        classification = "MANUAL_REVIEW_PENDING"
    elif not gates["body_paired_median_at_least_3percent"]:
        status = "C14_NO_QUALIFYING_FULL_BODY_GAIN"
        classification = "C14_NO_QUALIFYING_FULL_BODY_GAIN"
    elif not gates["application_guard"] or not gates["POST_both_blocks_positive"] or not gates["first_POST_domains"]:
        status = "C14_APPLICATION_OR_POST_GUARD_FAILURE"
        classification = "C14_APPLICATION_OR_POST_GUARD_FAILURE"
    else:
        status = "C14_INCONCLUSIVE_VARIABILITY_OR_ORDER"
        classification = "C14_INCONCLUSIVE_VARIABILITY_OR_ORDER"
    return dict(
        schema=RESULT_SCHEMA,
        status=status,
        classification=classification,
        gates=gates,
        pairs=pairs,
        blocks=blocks,
        orders=orders,
        RR_controls=controls,
        body_paired_median_gain=median_gain,
        balanced_body_gain=1.0 - math.exp(body_log),
        observed_RR_log_variation=observed_noise,
        balanced_application_gain=1.0 - math.exp(statistics.mean(row["application_log_ratio"] for row in blocks)),
        first_POST_review=[
            {
                "case": name,
                "first_POST_latency_s": row["first_POST_latency_s"],
                "initialization_s": row["initialization_within_application_s"],
                "first_body_s": row["first_body_s"],
                "c_pool": row["c_pool"],
            }
            for name, row in cases.items()
        ],
        unexplained_first_POST_delay_review="PENDING_BLOCKS_MANUAL_ACCEPTANCE",
        formal_windows=8,
        reference_control_windows=4,
        qualification_windows=2,
        historical_cases_pooled=0,
        automatic_fullsize=False,
        production_qualified=False,
        inference="Two time blocks within one allocation; observed RR variation is not a confidence bound",
        default="I0",
        review_gate="MANUAL_REVIEW_PENDING",
    )


__all__ = ["analyze"]
