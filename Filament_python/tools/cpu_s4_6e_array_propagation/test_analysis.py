"""Deterministic tests of the frozen denominator, order and guard decisions."""
import copy
import pytest
from .protocol import MATRIX
from .analysis import analyze


def cases():
    return {name: dict(role=role, intervals=n, exact_gate_pass=True, later_body_s=10.0 if role == "R" else 9.0,
                       post_gap_median_s=2.0 if role == "R" else 1.8, application_service_s=20.0 if role == "R" else 19.0,
                       first_POST_latency_s=3.0, initialization_within_application_s=1.0, first_body_s=2.0,
                       intensity_pool={}, clock_domain_pass=True) for name, role, n, kind in MATRIX}


def test_s46e_scalar_performance_excludes_qualification_and_rr():
    rows = cases()
    rows["QUAL_A2"]["later_body_s"] = 999
    result = analyze(rows)
    assert all(result["gates"].values())
    assert result["body_paired_median_gain"] == pytest.approx(.1)
    assert result["automatic_fullsize"] is False


def test_s46e_scalar_negative_no_added_batch_or_new_denominator():
    rows = cases()
    for row in rows.values():
        if row["role"] == "A":
            row["later_body_s"] = 10.1
    assert analyze(rows)["status"] == "NO_QUALIFYING_FULL_BODY_GAIN"
    rows["EXTRA"] = copy.deepcopy(rows["K1_A1"])
    with pytest.raises(RuntimeError, match="FOURTEEN"):
        analyze(rows)


def test_s46e_scalar_rr_and_application_block_false_gain():
    rows = cases()
    rows["CTRL_LATE_R2"]["later_body_s"] = 12
    rows["K1_A1"]["application_service_s"] = 22
    result = analyze(rows)
    assert not result["gates"]["body_exceeds_observed_RR"]
    assert not result["gates"]["application_guard"]
    assert result["status"] == "INCONCLUSIVE_VARIABILITY_ORDER_OR_GUARD"
