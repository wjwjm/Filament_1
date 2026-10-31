"""Focused private G2 callsite and helper-binding checks."""

from __future__ import annotations

import ast
from types import SimpleNamespace

import numpy as np
import pytest

from KHz_filament.nonlinear import apply_nonlinear

from .c_candidate import (
    C_APPLY_HELPER,
    ROLE_CANDIDATE,
    SourceBindingError,
    _verify_active_compiled_call,
    _instrument_c_callsite,
    _make_c_call,
)
from .c_capture import CaptureStore
from .c_instrument import CInstrumentationError, compile_private_apply
from .c_pool import CPool


def test_s46e_c_callsite_replaces_only_the_single_g2_apply_and_strips():
    module = ast.parse(
        """
def propagate_one_pulse(x):
    interval_index = 0
    y = _g_apply_nonlinear(x, x, x, 0.1, dn_gas=None, k0=None)
    return y
"""
    )
    receipt = _instrument_c_callsite(module)
    assert receipt["replacement_count"] == 1
    calls = [node for node in ast.walk(module) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert any(node.func.id == C_APPLY_HELPER for node in calls)


def test_s46e_c_helper_uses_original_state_guard_and_rejects_tamper():
    E = np.ones((2, 2, 2), dtype=np.complex128)
    phase = np.zeros_like(E.real)
    alpha = np.zeros_like(E.real)
    pool = CPool(1)
    math_function, _ = compile_private_apply(apply_nonlinear, pool)
    class State:
        def __init__(self):
            self.calls = []
        def _g_nonlinear_call(self, original, args, kwargs):
            self.calls.append((original, args, kwargs))
            return original(*args, **kwargs)
    state = State()
    dispatch = SimpleNamespace(cpu_s4_6e_c_state=state, cpu_s4_6e_c_pool=pool)
    helper = _make_c_call(state, dispatch, math_function, pool, role=ROLE_CANDIDATE, backend=math_function.__cpu_s46e_c_backend__)
    result = helper(E, phase, alpha, 0.1, dn_gas=None, k0=None)
    assert result is E
    assert len(state.calls) == 1
    seal = helper.__cpu_s46e_c_function_seal__
    original_code = seal["code"]
    seal["code"] = object()
    with pytest.raises(CInstrumentationError, match="CODE_CHANGED"):
        helper(E, phase, alpha, 0.1, dn_gas=None, k0=None)
    seal["code"] = original_code
    dispatch.cpu_s4_6e_c_state = object()
    with pytest.raises(SourceBindingError, match="ACTIVE_STATE"):
        helper(E, phase, alpha, 0.1, dn_gas=None, k0=None)
    pool.close()


def test_s46e_c_final_compiled_propagator_helper_identity_is_sealed():
    def active():
        return None

    namespace = active.__globals__
    previous = namespace.get(C_APPLY_HELPER)
    sentinel = object()
    namespace[C_APPLY_HELPER] = sentinel
    try:
        wrapped = SimpleNamespace(__g2_inner__=active)
        assert _verify_active_compiled_call(wrapped, sentinel)[0] is active
        namespace[C_APPLY_HELPER] = object()
        with pytest.raises(SourceBindingError, match="HELPER_BINDING_CHANGED"):
            _verify_active_compiled_call(wrapped, sentinel)
    finally:
        if previous is None:
            namespace.pop(C_APPLY_HELPER, None)
        else:
            namespace[C_APPLY_HELPER] = previous


def test_s46e_c_local_factory_seals_actual_compiled_helper(tmp_path):
    from tools.cpu_s4_5b_steady_cost.adapter import make_recorder_pair

    from .c_candidate import build_role

    for role in ("R", "C"):
        pair = make_recorder_pair(False)
        built = build_role(
            "PURE_OFF_SMALLGRID",
            capture_root=tmp_path / role,
            role=role,
            local_engineering=True,
            pool_override=1,
            recorder_pair=pair,
        )
        try:
            dispatch = built[1]
            helper = dispatch.cpu_s4_6e_c_apply
            active = dispatch.cpu_s4_6e_c_active_propagator
            assert active.__globals__[C_APPLY_HELPER] is helper
        finally:
            dispatch.close_c_pool()


def test_s46e_c_helper_passes_source_interval_to_bounded_capture():
    E = np.ones((2, 2, 2), dtype=np.complex128)
    phase = np.zeros_like(E.real)
    alpha = np.zeros_like(E.real)
    pool = CPool(1)
    store = CaptureStore((0,), expected_total_calls=1, expected_interval_sequence=(0,), max_records=1)
    store.begin("G1")
    class State:
        def _g_nonlinear_call(self, original, args, kwargs):
            return original(*args, **kwargs)
    state = State()
    dispatch = SimpleNamespace(cpu_s4_6e_c_state=state, cpu_s4_6e_c_pool=pool)
    helper = _make_c_call(
        state,
        dispatch,
        apply_nonlinear,
        pool,
        role="R",
        backend={"xp_object": np, "underlying": np, "fft_dispatch_object": None},
        capture_store=store,
    )
    helper(E, phase, alpha, 0.1, 0, dn_gas=None, k0=None)
    rows = store.require_complete()
    assert rows[0]["ordinal_1based"] == 1
    assert rows[0]["output"] is not None
    pool.close()


def test_s46e_c_capture_after_call_failure_aborts_state_before_propagation_continues():
    E = np.ones((2, 2, 2), dtype=np.complex128)
    phase = np.zeros_like(E.real)
    alpha = np.zeros_like(E.real)
    pool = CPool(1)

    class FailingCapture:
        def before_call(self, interval_index, args, kwargs):
            return None

        def after_call(self, interval_index, result, args, kwargs):
            raise RuntimeError("capture-after-failure")

    class State:
        def __init__(self):
            self.aborts = []

        def _g_nonlinear_call(self, original, args, kwargs):
            return original(*args, **kwargs)

        def _g_abort_propagation(self, reason):
            self.aborts.append(reason)

    state = State()
    dispatch = SimpleNamespace(cpu_s4_6e_c_state=state, cpu_s4_6e_c_pool=pool)
    helper = _make_c_call(
        state,
        dispatch,
        apply_nonlinear,
        pool,
        role="R",
        backend={"xp_object": np, "underlying": np, "fft_dispatch_object": None},
        capture_store=FailingCapture(),
    )
    with pytest.raises(RuntimeError, match="capture-after-failure"):
        helper(E, phase, alpha, 0.1, dn_gas=None, k0=None)
    assert state.aborts == ["c_capture_exception"]
    pool.close()
