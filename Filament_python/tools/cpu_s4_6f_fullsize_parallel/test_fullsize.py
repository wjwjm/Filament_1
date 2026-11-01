from __future__ import annotations

import ast
import gc
import inspect
import weakref

import pytest

from tools.cpu_s4_6e_array_propagation import instrument as intensity_instrument

from .analysis import analyze
from .protocol import MICRO_SHAPE, RESOURCE, SHAPE, candidate_spec, expected_counts, matrix, plan
from .role_spec import _pool_names, pool_contract


def _cases(candidate="EQ10_P48", *, body=0.9, application=0.95, post=0.9):
    values = {}
    for row in matrix(candidate):
        case, role, intervals = row["case"], row["role"], row["intervals"]
        value = 100.0 if role == "R" else 100.0 * body
        if row["kind"] == "RR_CONTROL":
            value = 100.0 if case.endswith("R1") else 101.0
        app = 200.0 if role == "R" else 200.0 * application
        gap = 10.0 if role == "R" else 10.0 * post
        values[case] = dict(role=role, intervals=intervals, exact_gate_pass=True, later_body_s=value, application_service_s=app, post_gap_median_s=gap, first_POST_latency_s=3.0, initialization_within_application_s=1.0, first_body_s=20.0, clock_domain_pass=True, pointwise_pool={})
    return values


def test_fixed_fullsize_matrix_and_counts():
    assert [row["case"] for row in matrix("EQ10_P48")] == [
        "QUAL_R2", "QUAL_X2", "CTRL_EARLY_R1", "CTRL_EARLY_R2",
        "K1_R1", "K1_X1", "K1_X2", "K1_R2", "K2_X1", "K2_R1",
        "K2_R2", "K2_X2", "CTRL_LATE_R1", "CTRL_LATE_R2",
    ]
    assert expected_counts(2) == {"half_steps": 4, "fresh_responses": 8, "original_math_dispatches": 32}
    assert expected_counts(8) == {"half_steps": 16, "fresh_responses": 32, "original_math_dispatches": 128}
    assert plan("EQ10_P48")["shape"] == list(SHAPE)
    assert plan("EQ10_P48")["local_cache"]["max_bytes"] == 256 * 1024**2
    assert candidate_spec("PHASE_ABSORB_P24")["deferred"] is True


def test_role_contracts_are_separate():
    assert pool_contract("EQ10_P48") == {"workers": 48, "labels": 4, "groups_per_interval": 4}
    assert pool_contract("INTENSITY_P24") == {"workers": 24, "labels": 3, "groups_per_interval": 4}
    assert pool_contract("PHASE_ABSORB_P24") == {"workers": 24, "labels": 2, "groups_per_interval": 2}
    assert _pool_names("EQ10") == ("close_eq10_pool", "eq10_pool_snapshot", "take_eq10_pool_records")
    assert _pool_names("INTENSITY") == ("close_intensity_pool", "intensity_pool_snapshot", "take_intensity_pool_records")
    assert _pool_names("PHASE_ABSORB") == ("close_c_pool", "c_pool_snapshot", "take_c_pool_records")
    assert MICRO_SHAPE != SHAPE


def test_resource_contract_separates_attempt_and_study_budget():
    assert RESOURCE["per_job_allocated_core_hours"] == 192
    assert RESOURCE["max_jobids"] == 4
    assert RESOURCE["study_allocated_core_hours_max"] == 672


def test_analysis_pass_and_negative_classifications():
    passed = analyze(_cases(), "EQ10_P48")
    assert passed["classification"] == "PASS_MANUAL_REVIEW_PENDING"
    assert passed["gates"]["body_paired_median_at_least_3percent"]
    negative = analyze(_cases(body=1.02), "EQ10_P48")
    assert negative["classification"] == "NO_QUALIFYING_FULL_BODY_GAIN"
    guard = analyze(_cases(body=0.9, application=1.08), "EQ10_P48")
    assert guard["classification"] == "FULLSIZE_APPLICATION_OR_POST_GUARD_FAILURE"


def test_intensity_releases_magnitude_after_square_barrier():
    tree = ast.parse(textwrap_dedent(inspect.getsource(intensity_instrument._helpers)))
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_helpers")
    intensity = next(node for node in ast.walk(function) if isinstance(node, ast.FunctionDef) and node.name == "intensity")
    statements = intensity.body
    square = next(i for i, node in enumerate(statements) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "ufunc" and node.value.args and isinstance(node.value.args[0], ast.Constant) and node.value.args[0].value == "A_SQUARE")
    release = next(i for i, node in enumerate(statements) if isinstance(node, ast.Delete) and any(isinstance(target, ast.Name) and target.id == "magnitude" for target in node.targets))
    result_alloc = next(i for i, node in enumerate(statements) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "result" for target in node.targets))
    assert square < release < result_alloc


def test_intensity_runtime_releases_magnitude_before_final_allocate(monkeypatch):
    import numpy as np

    pool = intensity_instrument.IntensityPool(1)
    magnitude_ref = {"value": None}

    def synchronous(label, function, operands, output):
        if label == "A_SQUARE":
            magnitude_ref["value"] = weakref.ref(operands[0])
        function(*operands, out=output)
        return output

    pool.ufunc = synchronous
    original_empty = intensity_instrument.np.empty
    allocation_checks = []
    calls = {"count": 0}

    def checked_empty(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 3:
            gc.collect()
            allocation_checks.append(magnitude_ref["value"]() is None)
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(intensity_instrument.np, "empty", checked_empty)
    try:
        E = np.ones((2, 2, 2), dtype=np.complex128) * (1.0 + 2.0j)
        value = intensity_instrument._helpers(pool)[intensity_instrument.HELPER_NAMES["intensity"]](E, 3.0, np.float64)
        assert np.allclose(value, 15.0, rtol=0.0, atol=1e-14)
        assert allocation_checks == [True]
    finally:
        pool.close()


def textwrap_dedent(value):
    import textwrap

    return textwrap.dedent(value)
