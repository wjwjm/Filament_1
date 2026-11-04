"""Focused inverse, binding, and small-array tests for the 6K overlay."""

from __future__ import annotations

import ast
import copy
import importlib
from pathlib import Path

import numpy as np
import pytest

deposition = importlib.import_module('KHz_filament.deposition')
heat = importlib.import_module('KHz_filament.heat')
propagate = importlib.import_module('KHz_filament.propagate')
from tools.cpu_s4_5b_steady_cost.adapter import make_recorder_pair
from tools.cpu_s4_5b_steady_cost.timing import context
from tools.cpu_s4_6i_residual_parallel.power_hook import INTERVAL_KW as POWER_INTERVAL_KW
from tools.cpu_s4_6i_residual_parallel.power_volume import PowerVolumePool
from tools.cpu_s4_6j_alpha_factory.alpha_hook import AlphaParallelHook
from tools.cpu_s4_6j_alpha_factory.alpha_pool import AlphaPool

from .cost_hook import (
    HEAT_HELPER,
    Q_IB_HELPER,
    FiveChainCostError,
    FiveChainCostHook,
)
from .registry import registry_snapshot


def _propagator_ast() -> ast.FunctionDef:
    path = Path(propagate.__file__).resolve()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "propagate_one_pulse"
    )


def _make_hooks(*, enabled: bool = True):
    pair = make_recorder_pair(False)
    power_pool = PowerVolumePool(1)
    alpha_pool = AlphaPool(1)
    cost = FiveChainCostHook(pair.b, enabled=enabled)
    hook = AlphaParallelHook(
        power_pool,
        alpha_pool=alpha_pool,
        residual_cost_hook=cost,
    )
    return pair, cost, hook


def _assert_ieee_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert np.array_equal(np.isnan(actual), np.isnan(expected))
    assert np.array_equal(np.isinf(actual), np.isinf(expected))
    finite = ~(np.isnan(actual) | np.isinf(actual))
    np.testing.assert_array_equal(actual[finite], expected[finite])
    both_inf = np.isinf(actual) & np.isinf(expected)
    if np.any(both_inf):
        assert np.array_equal(np.signbit(actual[both_inf]), np.signbit(expected[both_inf]))
    both_zero = finite & (actual == 0.0) & (expected == 0.0)
    if np.any(both_zero):
        np.testing.assert_array_equal(
            actual[both_zero].view(np.uint64), expected[both_zero].view(np.uint64)
        )


def test_registry_exposes_disjoint_five_chain_sites() -> None:
    snapshot = registry_snapshot()
    assert snapshot["scope_label"] == "NONLINEAR_ARRAYS"
    rows = snapshot["sites"]
    assert len({row["source_site"] for row in rows}) == len(rows)
    phase = [row for row in rows if row["unit"] == "alpha_phase_assembly"]
    assert [row["name"] for row in phase if row["eligible"]] == [
        "alpha_total",
        "kerr",
        "rawplasma",
        "phase",
    ]
    assert not any(row["eligible"] and row["parent"] for row in rows)


def test_five_chain_transform_inverse_restores_alpha_power_ast() -> None:
    _pair, cost, hook = _make_hooks()
    try:
        original = _propagator_ast()
        changed = copy.deepcopy(original)
        receipt = hook.instrument(changed)
        assert receipt["residual_cost"]["enabled"] is True
        assert receipt["residual_cost"]["mathematical_changes"] == 0
        restored = hook.strip_generated_nodes(changed)
        assert ast.dump(restored, include_attributes=False) == ast.dump(
            original, include_attributes=False
        )
        compile(ast.Module(body=[changed], type_ignores=[]), "<s46k-cost>", "exec")
    finally:
        hook.close()


def test_cost_off_skips_generated_nodes_but_keeps_source_binding() -> None:
    _pair, cost, hook = _make_hooks(enabled=False)
    try:
        original = _propagator_ast()
        changed = copy.deepcopy(original)
        receipt = hook.instrument(changed)
        assert receipt["residual_cost"]["enabled"] is False
        assert ast.dump(hook.strip_generated_nodes(changed), include_attributes=False) == ast.dump(
            original, include_attributes=False
        )
        active = {"xp": np, **hook.runtime_helpers()}
        hook.bind_active(active)
        assert cost.verify_binding(active)["enabled"] is False
    finally:
        hook.close()


def test_private_q_ib_and_heat_helpers_preserve_nonfinite_math() -> None:
    _pair, cost, hook = _make_hooks()
    try:
        helpers = cost.runtime_helpers()
        alpha = np.array(
            [[[np.nan, np.inf], [-np.inf, 1.0]], [[2.0, -3.0], [4.0, 5.0]]],
            dtype=np.float64,
        )
        intensity = np.array(
            [[[1.0, 2.0], [3.0, 4.0]], [[np.nan, 6.0], [7.0, 8.0]]],
            dtype=np.float64,
        )
        with np.errstate(all="ignore"):
            expected_q = deposition.q_ib_from_power(alpha, intensity, 0.25)
            actual_q = helpers[Q_IB_HELPER](
                alpha,
                intensity,
                0.25,
                **{POWER_INTERVAL_KW: 1},
            )
        _assert_ieee_equal(actual_q, expected_q)

        expected_h = heat.heat_Q_per_z(
            intensity,
            alpha,
            0.25,
            ion_source=np.zeros_like(alpha),
            Wt=np.zeros_like(alpha),
            rho=np.zeros_like(alpha),
            Ui=1.0,
            N0=1.0,
        )
        actual_h = helpers[HEAT_HELPER](
            intensity,
            alpha,
            0.25,
            ion_source=np.zeros_like(alpha),
            Wt=np.zeros_like(alpha),
            rho=np.zeros_like(alpha),
            Ui=1.0,
            N0=1.0,
        )
        _assert_ieee_equal(actual_h, expected_h)
    finally:
        hook.close()


def test_context_wrapper_records_q_ib_and_restores_owner_context() -> None:
    pair, cost, hook = _make_hooks()
    try:
        source = np.ones((2, 2, 2), dtype=np.float64)
        metadata = context(2, ordinal=2, occurrence_class="PER_INTERVAL", source_site="test.interval")
        with pair.b.scope("INTERVAL_BODY", context=metadata, common=True):
            cost.runtime_helpers()[Q_IB_HELPER](
                source,
                source,
                0.25,
                **{POWER_INTERVAL_KW: 2},
            )
        rows = [
            row
            for row in pair.b.snapshot()["records"]
            if row["label"] == "NONLINEAR_ARRAYS"
        ]
        assert [row["context"]["source_site"] for row in rows] == [
            "deposition.py:50-57:q_ib_multiplication"
        ]
        assert rows[0]["context"]["interval_index"] == 2
        assert getattr(hook, "_context_local").__dict__ == {}
    finally:
        hook.close()


def test_active_helper_and_owner_backend_tampering_fails_closed() -> None:
    _pair, cost, hook = _make_hooks()
    try:
        active = {"xp": np, **hook.runtime_helpers()}
        hook.bind_active(active)
        broken = dict(active)
        broken[Q_IB_HELPER] = lambda *args, **kwargs: None
        with pytest.raises(FiveChainCostError, match="ACTIVE_HELPER_CHANGED"):
            cost.verify_binding(broken)

        namespace = cost._q_ib_namespace
        assert namespace is not None
        original_xp = namespace["xp"]
        namespace["xp"] = object()
        with pytest.raises(FiveChainCostError, match="Q_IB_XP_SLOT_CHANGED"):
            cost.verify_binding(active)
        namespace["xp"] = original_xp
        cost.verify_binding(active)
    finally:
        hook.close()


def test_alias_tampering_cannot_hide_behind_inverse() -> None:
    _pair, _cost, hook = _make_hooks()
    try:
        changed = copy.deepcopy(_propagator_ast())
        hook.instrument(changed)
        alias = next(
            node
            for node in ast.walk(changed)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == Q_IB_HELPER
        )
        alias.func.id = "wrong_q_ib_helper"
        with pytest.raises(FiveChainCostError, match="ALIAS_REPLACEMENT_CHANGED"):
            hook.strip_generated_nodes(changed)
    finally:
        hook.close()


@pytest.mark.parametrize('target', ['wrapper_closure', 'heat_cap', 'context_owner'])
def test_new_private_binding_cannot_change_without_detection(target):
    import threading
    _pair, cost, hook = _make_hooks()
    restore = None
    try:
        active = {'xp': np, **hook.runtime_helpers()}
        hook.bind_active(active)
        if target == 'wrapper_closure':
            wrapper = cost._q_ib_wrapper
            cells = dict(zip(wrapper.__code__.co_freevars, wrapper.__closure__))
            cell = cells['target']
            previous = cell.cell_contents
            cell.cell_contents = lambda *args, **kwargs: None
            restore = lambda: setattr(cell, 'cell_contents', previous)
        elif target == 'heat_cap':
            namespace = cost._heat_namespace
            previous = namespace['Q_CAP']
            namespace['Q_CAP'] = 3.0
            restore = lambda: namespace.__setitem__('Q_CAP', previous)
        else:
            previous = hook._context_local
            hook._context_local = threading.local()
            restore = lambda: setattr(hook, '_context_local', previous)
        with pytest.raises(RuntimeError):
            cost.verify_binding(active)
        restore()
        restore = None
        cost.verify_binding(active)
    finally:
        if restore is not None:
            restore()
        hook.close()
