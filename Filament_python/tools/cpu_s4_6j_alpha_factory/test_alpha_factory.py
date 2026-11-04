"""Opt-in lifecycle tests for the S4-6J alpha factory adapter.

Factory construction starts persistent worker pools and therefore remains
disabled in the ordinary lightweight test collection.  Set
``S46J_ALPHA_FACTORY_TESTS=1`` for the focused local engineering checks.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from tools.cpu_s4_5b_steady_cost.adapter import make_recorder_pair
from tools.cpu_s4_6j_alpha_factory import alpha_factory as factory
from tools.cpu_s4_6j_alpha_factory.alpha_hook import AlphaParallelHook
from tools.cpu_s4_6j_alpha_factory.alpha_pool import AlphaPool
from tools.cpu_s4_6j_alpha_factory.protocol import MICRO_SHAPE

CANDIDATE = factory.CANDIDATE


def test_factory_candidate_identity_is_alpha_not_cost_six():
    assert CANDIDATE == "ION_ALPHA_CLEANUP_P24"
    assert factory.candidate_spec(CANDIDATE)["semantic_id"] == CANDIDATE
    with pytest.raises(ValueError, match="CANDIDATE_IDENTITY"):
        factory.candidate_spec("FOUR_COMPONENT_FACTORY_COST")


pytestmark = pytest.mark.skipif(
    os.environ.get("S46J_ALPHA_FACTORY_TESTS") != "1",
    reason="S4-6J factory construction is opt-in",
)


def _build(tmp_path, role: str):
    pair = make_recorder_pair(False)
    built = factory.build_role(
        "PURE_OFF_SMALLGRID",
        capture_root=tmp_path / ("capture_" + role),
        role=role,
        local_engineering=True,
        pool_override=1,
        recorder_pair=pair,
        shape=MICRO_SHAPE,
    )
    return pair, built


def test_reference_has_four_components_and_no_alpha_pool(tmp_path):
    _pair, built = _build(tmp_path, "R")
    dispatch = built[1]
    try:
        assert type(dispatch.cpu_s4_6j_alpha_hook) is AlphaParallelHook
        assert dispatch.cpu_s4_6j_alpha_pool is None
        assert dispatch.cpu_s4_6j_external_role == "R"
        binding = factory.verify_bindings(dispatch)
        assert binding["external_role"] == "R"
        assert factory.pool_snapshot(dispatch)["alpha"]["pool"] is None
    finally:
        final = factory.close_role(dispatch, CANDIDATE)
        assert final["all_pools_retired_before_output"]


def test_candidate_has_fifth_pool_and_repeat_close_is_idempotent(tmp_path):
    _pair, built = _build(tmp_path, "X")
    dispatch = built[1]
    assert type(dispatch.cpu_s4_6j_alpha_pool) is AlphaPool
    assert dispatch.cpu_s4_6j_alpha_pool.count == 1
    first = factory.close_role(dispatch, CANDIDATE)
    second = factory.close_role(dispatch, CANDIDATE)
    assert first["all_pools_retired_before_output"]
    assert second == first
    assert not dispatch.cpu_s4_6j_alpha_pool.snapshot()["active_thread_ids"]


def test_alpha_pool_identity_tamper_is_rejected(tmp_path):
    _pair, built = _build(tmp_path, "X")
    dispatch = built[1]
    original = dispatch.cpu_s4_6j_alpha_pool
    try:
        dispatch.cpu_s4_6j_alpha_pool = object()
        with pytest.raises(factory.AlphaFactoryError, match="ALPHA_POOL"):
            factory.verify_bindings(dispatch)
    finally:
        dispatch.cpu_s4_6j_alpha_pool = original
        factory.close_role(dispatch, CANDIDATE)


def test_constructor_failure_retires_alpha_pool(tmp_path, monkeypatch):
    captured = {}

    def fail_hook(pool, *, alpha_pool=None, capture_store=None):
        captured["pool"] = alpha_pool
        raise RuntimeError("injected alpha hook construction failure")

    monkeypatch.setattr(factory, "AlphaParallelHook", fail_hook)
    with pytest.raises(RuntimeError, match="injected alpha hook"):
        factory.build_role(
            "PURE_OFF_SMALLGRID",
            capture_root=tmp_path / "failed",
            role="X",
            local_engineering=True,
            pool_override=1,
            recorder_pair=make_recorder_pair(False),
            shape=MICRO_SHAPE,
        )
    assert type(captured["pool"]) is AlphaPool
    assert captured["pool"].snapshot()["closed"]
    assert not captured["pool"].snapshot()["active_thread_ids"]


def test_failure_callback_routes_through_all_pool_finalizer(tmp_path):
    _pair, built = _build(tmp_path, "X")
    dispatch = built[1]
    hook = dispatch.cpu_s4_6j_alpha_hook
    try:
        hook._failure_cleanup()
        final = dispatch._cpu_s4_6j_finalized
        assert final["all_pools_retired_before_output"]
        assert final["alpha"]["pool"]["closed"]
    finally:
        # The callback is intentionally idempotent; this also exercises the
        # normal close anchor after a task-failure callback.
        factory.close_role(dispatch, CANDIDATE)


def test_alpha_propagator_wrapper_preserves_g2_attrs_and_closes_on_error(monkeypatch):
    calls = []

    def wrapped(*args, **kwargs):
        raise RuntimeError("propagation failure")

    wrapped.__g2_inner__ = object()
    wrapped.__g2_actual_code__ = object()
    wrapped.__g2_expected_code__ = object()
    dispatch = SimpleNamespace()
    monkeypatch.setattr(
        factory,
        "finalize_factory",
        lambda current, *, reason="NORMAL": calls.append((current, reason)) or {"ok": True},
    )
    guarded = factory._wrap_alpha_propagator(wrapped, dispatch)
    with pytest.raises(RuntimeError, match="propagation failure"):
        guarded()
    assert calls == [(dispatch, "TASK_FAILURE")]
    assert guarded.__g2_inner__ is wrapped.__g2_inner__
    assert guarded.__g2_actual_code__ is wrapped.__g2_actual_code__
    assert guarded.__g2_expected_code__ is wrapped.__g2_expected_code__
    assert guarded.__s46j_alpha_failure_wrapper__ is True


def test_first_finalize_identity_mismatch_closes_all_known_alpha_pools(tmp_path):
    _pair, built = _build(tmp_path, "X")
    dispatch = built[1]
    original = dispatch.cpu_s4_6j_alpha_pool
    replacement = AlphaPool(1)
    dispatch.cpu_s4_6j_alpha_pool = replacement
    try:
        with pytest.raises(factory.AlphaFactoryError, match="POOL_FINALIZATION_FAILED"):
            factory.close_role(dispatch, CANDIDATE)
        receipt = dispatch._cpu_s4_6j_finalization_failure
        assert receipt["alpha"]["identity_initial"]["valid"] is False
        assert replacement.snapshot()["closed"]
        assert original.snapshot()["closed"]
        assert not replacement.snapshot()["active_thread_ids"]
        assert not original.snapshot()["active_thread_ids"]
    finally:
        dispatch.cpu_s4_6j_alpha_pool = original
        try:
            factory.close_role(dispatch, CANDIDATE)
        except BaseException:
            pass


def test_alpha_snapshot_failure_is_retained_in_bounded_failure_receipt(tmp_path):
    _pair, built = _build(tmp_path, "X")
    dispatch = built[1]
    pool = dispatch.cpu_s4_6j_alpha_pool
    original_snapshot = pool.snapshot

    def fail_snapshot():
        raise RuntimeError("snapshot failure")

    pool.snapshot = fail_snapshot
    try:
        with pytest.raises(factory.AlphaFactoryError, match="POOL_FINALIZATION_FAILED"):
            factory.close_role(dispatch, CANDIDATE)
        receipt = dispatch._cpu_s4_6j_finalization_failure
        assert receipt["alpha"]["pool"]["close_error"] in {
            "RuntimeError",
            "S46J_ALPHA_POOL_SNAPSHOT_NOT_MAPPING",
        }
    finally:
        pool.snapshot = original_snapshot
        try:
            factory.close_role(dispatch, CANDIDATE)
        except BaseException:
            pass
