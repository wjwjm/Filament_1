from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from tools.cpu_s4_5b_steady_cost import adapter
from tools.cpu_s4_6i_residual_parallel import cost_factory
from tools.cpu_s4_6i_residual_parallel.cost_factory import make_cost_hook
from tools.cpu_s4_6i_residual_parallel.instrument import (
    instrument_heat_ast,
    instrument_residual_cost_ast,
)
from tools.cpu_s4_6i_residual_parallel.registry import AUXILIARY_UNITS, COST_UNITS, registry_snapshot


def _frozen_function(path: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1
    return matches[0]


def test_registry_is_disjoint_and_source_ordered() -> None:
    snapshot = registry_snapshot()
    names = [row["name"] for row in snapshot["units"]]
    orders = [row["source_order"] for row in snapshot["units"]]
    assert len(names) == len(set(names))
    assert orders == sorted(orders)
    assert tuple(snapshot["power_volume"]["contexts"]) == (
        "q_ion",
        "q_ib",
        "direct_ion",
        "direct_ib",
    )


def test_propagation_cost_transform_strips_to_post_b_ast() -> None:
    path = "Filament_python/KHz_filament/propagate.py"
    function = _frozen_function(path, "propagate_one_pulse")
    b_adapter = __import__("tools.cpu_s4_5b_steady_cost.adapter", fromlist=["adapter"])
    b_adapter.instrument_propagation_ast(function)
    receipt = instrument_residual_cost_ast(function)
    assert receipt["array_values_retained"] is False
    assert receipt["scientific_expression_changes"] == 0
    assert receipt["inverse_proof"] == "strip_residual_cost_restores_post_B_C_ast"
    assert len(receipt["scope_sites"]) == len(COST_UNITS) + len(AUXILIARY_UNITS)


def test_heat_cost_transform_inverse_proof() -> None:
    function = _frozen_function("Filament_python/KHz_filament/heat.py", "heat_Q_per_z")
    receipt = instrument_heat_ast(function)
    assert receipt["array_values_retained"] is False
    assert receipt["scientific_expression_changes"] == 0


def test_original_deposition_and_heat_outputs_are_unchanged() -> None:
    pair = adapter.make_recorder_pair(False)
    hook = make_cost_hook(pair.b)
    helpers = hook.runtime_helpers()
    from KHz_filament import deposition, heat

    power = np.array([[[1.0, np.nan], [np.inf, -2.0]]], dtype=np.float64)
    alpha = np.array([[[0.5, 0.25], [0.0, 2.0]]], dtype=np.float64)
    power_before = power.copy()
    alpha_before = alpha.copy()
    assert np.array_equal(
        helpers[cost_factory.Q_ION_HELPER](power, 0.25),
        deposition.q_ion_from_power(power, 0.25),
        equal_nan=True,
    )
    assert np.array_equal(
        helpers[cost_factory.Q_IB_HELPER](alpha, power, 0.25),
        deposition.q_ib_from_power(alpha, power, 0.25),
        equal_nan=True,
    )
    assert np.array_equal(
        helpers[cost_factory.DIRECT_IB_HELPER](alpha * power, 0.25, 1.0, 1.0, 1.0),
        deposition.direct_interval_energy(alpha * power, 0.25, 1.0, 1.0, 1.0),
        equal_nan=True,
    )
    ion_source = np.array([[1.0, np.nan], [np.inf, -2.0]], dtype=np.float64)
    assert np.array_equal(
        helpers[cost_factory.HEAT_HELPER](
            power[0], alpha[0], 0.25, ion_source=ion_source, Wt=None, rho=None, Ui=None, N0=None
        ),
        heat.heat_Q_per_z(
            power[0], alpha[0], 0.25, ion_source=ion_source, Wt=None, rho=None, Ui=None, N0=None
        ),
        equal_nan=True,
    )
    assert np.array_equal(power, power_before, equal_nan=True)
    assert np.array_equal(alpha, alpha_before, equal_nan=True)
    assert hook.snapshot()["events_dropped"] == 0
    sites = {row["context"]["source_site"] for row in hook.snapshot()["records"]}
    assert "deposition.py:23:_power_volume:q_ion" in sites
    assert "deposition.py:23:_power_volume:q_ib" in sites
    assert "deposition.py:23:_power_volume:direct_ib" in sites
    assert "deposition.py:42:time_sum:q_ion" in sites
    assert "deposition.py:131:independent_3d_sum:direct_ib" in sites
    assert "heat.py:36:time_sum" in sites


def test_cost_scope_is_common_and_bounded() -> None:
    pair = adapter.make_recorder_pair(False)
    hook = make_cost_hook(pair.b)
    scope = hook.runtime_helpers()[cost_factory.SCOPE_HELPER]
    with scope("NONLINEAR_ARRAYS", 0, "PER_INTERVAL", "test_cost:scope"):
        pass
    snapshot = hook.snapshot()
    assert snapshot["event_count"] == 1
    assert snapshot["events_dropped"] == 0
    assert not snapshot["incomplete_scopes"]
    hook.verify_binding()


def test_helper_and_receipt_tamper_rejected() -> None:
    pair = adapter.make_recorder_pair(False)
    hook = make_cost_hook(pair.b)
    helper = hook.runtime_helpers()[cost_factory.SCOPE_HELPER]
    original_code = helper.__code__
    try:
        helper.__code__ = original_code.replace(
            co_firstlineno=int(original_code.co_firstlineno) + 1
        )
        with pytest.raises(Exception):
            hook.verify_binding()
    finally:
        helper.__code__ = original_code
    hook.verify_binding()
    original = hook._receipt["enabled"]
    try:
        hook._receipt["enabled"] = False
        with pytest.raises(Exception):
            hook.verify_binding()
    finally:
        hook._receipt["enabled"] = original


def test_private_clone_code_and_global_tampering_rejected():
    hook = make_cost_hook(adapter.make_recorder_pair(False).b)
    clone = hook._source_clones["deposition.integrate_power_to_q"]
    saved = clone.__code__
    try:
        clone.__code__ = saved.replace(co_firstlineno=saved.co_firstlineno+1)
        with pytest.raises(Exception, match="SEAL_CODE_CHANGED"):
            hook.verify_binding()
    finally:
        clone.__code__ = saved
    helper = clone.__globals__["_power_volume"]
    try:
        clone.__globals__["_power_volume"] = lambda x: x
        with pytest.raises(Exception, match="SOURCE_CLONE_GLOBAL_CHANGED"):
            hook.verify_binding()
    finally:
        clone.__globals__["_power_volume"] = helper
    hook.verify_binding()


@pytest.mark.parametrize("cost_enabled", [False, True])
def test_h_factory_role_x_and_always_finalizes(tmp_path: Path, cost_enabled: bool) -> None:
    pair = adapter.make_recorder_pair(False)
    dispatch = None
    try:
        result = cost_factory.build_role(
            "PURE_OFF_SMALLGRID",
            capture_root=tmp_path,
            cost_enabled=cost_enabled,
            local_engineering=True,
            pool_override=1,
            recorder_pair=pair,
        )
        dispatch = result[1]
        assert dispatch.cpu_s4_6h_role == "X"
        check = cost_factory.verify_bindings(dispatch)
        assert check["cost_enabled"] is cost_enabled
        if cost_enabled:
            assert dispatch.cpu_s4_6i_cost_hook is not None
            assert cost_factory.take_cost_snapshot(dispatch)["events_dropped"] == 0
        else:
            assert dispatch.cpu_s4_6i_cost_hook is None
    finally:
        if dispatch is not None:
            final = cost_factory.finalize_factory(dispatch)
            assert final["all_pools_retired_before_output"] is True
