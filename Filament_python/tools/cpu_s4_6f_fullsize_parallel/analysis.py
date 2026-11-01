"""Scalar-only full-size analysis for one semantic candidate."""

from __future__ import annotations

import math
import statistics

from .protocol import matrix, pairings


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def analyze(cases, candidate: str):
    expected = matrix(candidate)
    require(list(cases) == [row["case"] for row in expected], "S46F_COMPLETE_ORDERED_FOURTEEN_CASES")
    for row in expected:
        value = cases[row["case"]]
        require(value.get("role") == row["role"] and int(value.get("intervals")) == row["intervals"], "S46F_FIXED_CASE_IDENTITY")
        require(value.get("exact_gate_pass") is True, "S46F_ALL_CASES_EXACT_BEFORE_PERFORMANCE")

    metrics = {"body": "later_body_s", "POST": "post_gap_median_s", "application": "application_service_s"}

    def log_ratio(reference, selected, key):
        a, b = reference[key], selected[key]
        require(type(a) in (int, float) and type(b) in (int, float) and math.isfinite(a) and math.isfinite(b) and min(a, b) > 0, "S46F_FINITE_POSITIVE_CLOCK:" + key)
        return math.log(b / a)

    pairs = []
    for pairing in pairings(candidate):
        name = pairing["pair"]
        reference_name = pairing["reference"]
        selected_name = pairing["candidate"]
        order = pairing["order"]
        row = dict(pair=name, block=name[:2], reference=reference_name, candidate=selected_name, order=order)
        for label, key in metrics.items():
            value = log_ratio(cases[reference_name], cases[selected_name], key)
            row.update({label + "_log_ratio": value, label + "_gain": 1 - math.exp(value), label + "_R_s": cases[reference_name][key], label + "_X_s": cases[selected_name][key], label + "_saved_s": cases[reference_name][key] - cases[selected_name][key]})
        row["application_ratio"] = cases[selected_name]["application_service_s"] / cases[reference_name]["application_service_s"]
        pairs.append(row)

    blocks = []
    for block in ("K1", "K2"):
        members = [row for row in pairs if row["block"] == block]
        record = dict(block=block)
        for label in metrics:
            value = statistics.mean(row[label + "_log_ratio"] for row in members)
            record.update({label + "_log_ratio": value, label + "_gain": 1 - math.exp(value)})
        blocks.append(record)

    controls = []
    for label in ("EARLY", "LATE"):
        r1, r2 = (cases[f"CTRL_{label}_R{i}"] for i in (1, 2))
        controls.append(dict(control=label, **{key + "_absolute_log_variation": abs(log_ratio(r1, r2, metric)) for key, metric in metrics.items()}))
    noise = max(row["body_absolute_log_variation"] for row in controls)
    orders = []
    for order in ("RX", "XR"):
        value = statistics.mean(row["body_log_ratio"] for row in pairs if row["order"] == order)
        orders.append(dict(order=order, body_log_ratio=value, body_gain=1 - math.exp(value)))
    body_log = statistics.mean(row["body_log_ratio"] for row in blocks)
    median_gain = statistics.median(row["body_gain"] for row in pairs)
    gates = dict(
        science_exact=True,
        body_paired_median_at_least_3percent=median_gain >= 0.03,
        both_blocks_positive=all(row["body_gain"] > 0 for row in blocks),
        both_orders_positive=all(row["body_gain"] > 0 for row in orders),
        body_exceeds_observed_RR=-body_log > noise,
        POST_both_blocks_positive=all(row["POST_gain"] > 0 for row in blocks),
        application_guard=statistics.median(row["application_ratio"] for row in pairs) <= 1.02 and all(row["application_ratio"] <= 1.05 for row in pairs),
        first_POST_domains=all(row.get("clock_domain_pass") is True for row in cases.values()),
    )
    if all(gates.values()):
        status = "FULLSIZE_PERFORMANCE_GATES_PASS_MANUAL_REVIEW_PENDING"
        classification = "PASS_MANUAL_REVIEW_PENDING"
    elif not gates["body_paired_median_at_least_3percent"]:
        status = "NO_QUALIFYING_FULL_BODY_GAIN"
        classification = "NO_QUALIFYING_FULL_BODY_GAIN"
    elif not (gates["application_guard"] and gates["POST_both_blocks_positive"]):
        status = "FULLSIZE_APPLICATION_OR_POST_GUARD_FAILURE"
        classification = "FULLSIZE_APPLICATION_OR_POST_GUARD_FAILURE"
    else:
        status = "FULLSIZE_INCONCLUSIVE_VARIABILITY_OR_ORDER"
        classification = "FULLSIZE_INCONCLUSIVE_VARIABILITY_OR_ORDER"
    return dict(
        schema="filament.cpu_s4_6f.result.v1",
        candidate=str(candidate).upper(), status=status, classification=classification,
        gates=gates, pairs=pairs, blocks=blocks, orders=orders, RR_controls=controls,
        body_paired_median_gain=median_gain, balanced_body_gain=1 - math.exp(body_log),
        observed_RR_log_variation=noise,
        balanced_application_gain=1 - math.exp(statistics.mean(row["application_log_ratio"] for row in blocks)),
        first_POST_review=[dict(case=name, first_POST_latency_s=row["first_POST_latency_s"], initialization_s=row["initialization_within_application_s"], first_body_s=row["first_body_s"], pool=row.get("pointwise_pool")) for name, row in cases.items()],
        unexplained_first_POST_delay_review="PENDING_BLOCKS_MANUAL_ACCEPTANCE",
        formal_windows=8, reference_control_windows=4, qualification_windows=2,
        historical_cases_pooled=0, automatic_fullsize=False, production_qualified=False,
        inference="Two time blocks within one allocation; observed RR variation is not a confidence bound",
        default="I0", review_gate="MANUAL_REVIEW_PENDING",
    )


__all__ = ["analyze", "require"]
