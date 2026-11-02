"""Focused S4-6H contracts; no propagation, pytest execution, or HPC access."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.cpu_s4_6d_eq10_propagation.instrument import HELPER_NAMES as EQ10_HELPERS, _helpers as eq10_helpers
from tools.cpu_s4_6e_array_propagation.instrument import HELPER_NAMES as INTENSITY_HELPERS, _helpers as intensity_helpers
from tools.cpu_s4_6e_array_propagation.c_pool import CPool

from .combined_candidate import (
    COMBINED_SEMANTIC_ID,
    HPC_CONTRACT,
    LOCAL_CONTRACT,
    _PoolSet,
    _make_pools,
    _owned_source_identity,
    _wrap_propagator_for_failure,
    build_candidate,
    build_reference,
    finalize_factory,
    source_identity,
    verify_bindings,
)
combined_module = importlib.import_module(__package__ + ".combined_candidate")
from .instrument import (
    instrument_combined_stage,
    strip_combined_generated_nodes,
)


def _stage_node() -> ast.FunctionDef:
    source = Path(__file__).resolve().parents[2] / "KHz_filament" / "raman.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "isaacs_raman_stage"
    )


def test_s46h_joint_ast_has_one_inverse_and_both_component_receipts() -> None:
    stage = _stage_node()
    baseline = ast.dump(stage, include_attributes=False)
    receipt = instrument_combined_stage(stage)
    assert receipt["component_order"] == ("intensity", "eq10")
    assert receipt["replacement_counts"] == {"intensity": 1, "eq10": 4}
    assert receipt["scientific_expression_changes"] == 0
    assert receipt["stripped_ast_sha256"] == receipt["source_ast_sha256"]
    assert ast.dump(strip_combined_generated_nodes(stage), include_attributes=False) == baseline


def test_s46h_joint_ast_rejects_component_source_tamper() -> None:
    stage = _stage_node()
    assignment = next(
        node
        for node in ast.walk(stage)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "intensity"
    )
    assignment.value.func.attr = "real"
    with pytest.raises(Exception, match="(ABS_SHAPE|A_INTENSITY_ASTYPE_SHAPE)"):
        instrument_combined_stage(stage)


def test_s46h_three_pool_exact_alias_lifetime_and_normal_finalization() -> None:
    poolset = _make_pools("X", LOCAL_CONTRACT)
    dispatch = SimpleNamespace(cpu_s4_6h_poolset=poolset)
    try:
        field = np.ascontiguousarray(
            np.arange(2 * 2 * 2, dtype=np.float64).reshape(2, 2, 2) + 1j,
            dtype=np.complex128,
        )
        frozen = field.copy()
        intensity = intensity_helpers(poolset.intensity)[INTENSITY_HELPERS["intensity"]]
        result = intensity(field, 2.0, np.float64)
        np.testing.assert_array_equal(result, (2.0 * np.abs(field) ** 2).astype(np.float64))
        assert not np.shares_memory(result, field)
        np.testing.assert_array_equal(field, frozen)

        values = np.ascontiguousarray(np.arange(8, dtype=np.float64).reshape(2, 2, 2))
        eq10 = eq10_helpers(poolset.eq10)[EQ10_HELPERS["response"]]
        output = eq10(values, 2.0)
        np.testing.assert_array_equal(output, values / 2.0)
        backing = np.empty((2, 2, 2), dtype=np.float64)
        backing.fill(1.0)
        input_view = backing.view()
        with pytest.raises(Exception, match="ALIAS"):
            poolset.eq10.ufunc("B_RESPONSE_DIVIDE", np.divide, (input_view, 2.0), backing)

        finalized = finalize_factory(dispatch)
        assert finalized["all_pools_retired_before_output"] is True
        assert all(row["pool"]["closed"] for key, row in finalized.items() if key in {"intensity", "eq10", "c"})
        assert all(row["pool"]["active_thread_ids"] == [] for key, row in finalized.items() if key in {"intensity", "eq10", "c"})
        assert finalize_factory(dispatch) == finalized
    finally:
        if not hasattr(dispatch, "_cpu_s4_6h_finalized"):
            poolset.close_and_drain()


def test_s46h_c_pool_rejects_field_alias_before_write() -> None:
    poolset = _make_pools("H", LOCAL_CONTRACT)
    try:
        field = np.ones((2, 2, 2), dtype=np.complex128)
        alias = field.real
        alpha = np.zeros_like(field.real)
        with pytest.raises(Exception, match="E_ALIAS"):
            CPool.validate_domain(field, alias, alpha)
    finally:
        poolset.close_and_drain()


def test_s46h_task_failure_retires_all_three_pools() -> None:
    poolset = _make_pools("X", LOCAL_CONTRACT)
    dispatch = SimpleNamespace(cpu_s4_6h_poolset=poolset)

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("joint task failure")

    try:
        wrapped = _wrap_propagator_for_failure(fail, dispatch)
        with pytest.raises(RuntimeError, match="joint task failure"):
            wrapped()
        assert all(pool.snapshot()["closed"] for _, pool in poolset.values())
        assert all(pool.snapshot()["active_thread_ids"] == [] for _, pool in poolset.values())
    finally:
        if not hasattr(dispatch, "_cpu_s4_6h_finalized"):
            poolset.close_and_drain()


def test_s46h_constructor_failure_closes_already_created_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePool:
        def __init__(self, count: int) -> None:
            self.count = count
            self.closed = False

        def close(self) -> None:
            self.closed = True

    created: list[FakePool] = []

    def make_intensity(count: int) -> FakePool:
        pool = FakePool(count)
        created.append(pool)
        return pool

    def fail_eq10(count: int) -> FakePool:
        raise RuntimeError("eq10 construction failure")

    monkeypatch.setattr(combined_module, "IntensityPool", make_intensity)
    monkeypatch.setattr(combined_module, "Eq10Pool", fail_eq10)
    with pytest.raises(RuntimeError, match="eq10 construction failure"):
        _make_pools("X", LOCAL_CONTRACT)
    assert created and all(pool.closed for pool in created)


def test_s46h_long_operation_records_are_bounded_and_explicitly_counted() -> None:
    poolset = _make_pools("X", LOCAL_CONTRACT)
    try:
        source = np.ones((2, 2, 2), dtype=np.complex128)
        output = np.empty(source.shape, dtype=np.float64)
        for _ in range(300):
            poolset.intensity.ufunc("A_ABS", np.absolute, (source,), output)
        snapshot = poolset.intensity.snapshot()
        assert snapshot["total_records"] == 300
        assert snapshot["retained_records"] == 256
        assert snapshot["dropped_records"] == 44
        assert snapshot["pending_scalar_records"] == 256
        assert snapshot["records_scalar_only"] is True
        assert snapshot["array_values_retained"] is False
    finally:
        poolset.close_and_drain()


def test_s46h_close_failure_does_not_cache_success(monkeypatch: pytest.MonkeyPatch) -> None:
    poolset = _make_pools("X", LOCAL_CONTRACT)
    dispatch = SimpleNamespace(cpu_s4_6h_poolset=poolset)
    original_close = poolset.eq10.close

    def fail_close() -> None:
        raise RuntimeError("injected close failure")

    monkeypatch.setattr(poolset.eq10, "close", fail_close)
    try:
        with pytest.raises(Exception, match="S46H_POOL_FINALIZATION_FAILED"):
            finalize_factory(dispatch)
        assert not hasattr(dispatch, "_cpu_s4_6h_finalized")
        assert dispatch._cpu_s4_6h_finalization_failure["close_errors"]
    finally:
        monkeypatch.setattr(poolset.eq10, "close", original_close)
        if not hasattr(dispatch, "_cpu_s4_6h_finalized"):
            finalize_factory(dispatch)


def test_s46h_recorded_close_failure_stays_failed_on_repeated_finalize(monkeypatch) -> None:
    poolset = _make_pools("X", LOCAL_CONTRACT)
    dispatch = SimpleNamespace(cpu_s4_6h_poolset=poolset)
    original = poolset.eq10._workers.close

    def fail_after_join():
        original()
        raise RuntimeError("injected failure after join")

    monkeypatch.setattr(poolset.eq10._workers, "close", fail_after_join)
    for _ in range(2):
        with pytest.raises(Exception, match="S46H_POOL_FINALIZATION_FAILED"):
            finalize_factory(dispatch)
        assert not hasattr(dispatch, "_cpu_s4_6h_finalized")
        evidence = dispatch._cpu_s4_6h_finalization_failure
        assert evidence["eq10"]["pool"]["close_error"] == "RuntimeError"
        assert evidence["all_pools_retired_before_output"] is False
    assert all(not pool.snapshot()["active_thread_ids"] for _, pool in poolset.values())


def _set_local_factory_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("UPPE_USE_GPU", "0")
    monkeypatch.setenv("NUMBA_NUM_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("BLIS_NUM_THREADS", "1")
    monkeypatch.setenv("NUMEXPR_NUM_THREADS", "1")
    monkeypatch.setenv("NUMBA_CPU_NAME", "generic")
    monkeypatch.setenv("NUMBA_THREADING_LAYER", "workqueue")


@pytest.mark.parametrize("builder,role", ((build_reference, "R"), (build_candidate, "X")))
def test_s46h_actual_w1_factories_bind_verify_and_close(
    builder, role: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_local_factory_environment(monkeypatch)
    from tools.cpu_s4_5b_steady_cost.adapter import make_recorder_pair

    built = builder(
        "PURE_OFF_SMALLGRID",
        capture_root=tmp_path / role,
        local_engineering=True,
        pool_override=1,
        recorder_pair=make_recorder_pair(False),
    )
    dispatch = built[1]
    try:
        checked = verify_bindings(dispatch)
        assert checked["role"] == role
        assert dispatch.cpu_s4_6h_source_identity["combined_semantic_id"] == COMBINED_SEMANTIC_ID
    finally:
        finalized = finalize_factory(dispatch)
        assert finalized["all_pools_retired_before_output"] is True


def test_s46h_original_helper_code_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_local_factory_environment(monkeypatch)
    from tools.cpu_s4_5b_steady_cost.adapter import make_recorder_pair

    built = build_candidate(
        "PURE_OFF_SMALLGRID",
        capture_root=tmp_path / "X",
        local_engineering=True,
        pool_override=1,
        recorder_pair=make_recorder_pair(False),
    )
    dispatch = built[1]
    state = dispatch.cpu_s4_6h_state
    name = next(iter(state._cpu_s46h_helper_originals))
    original = state._cpu_s46h_helper_originals[name]
    original_code = original.__code__
    try:
        semantic_id = state._cpu_s46h_source_identity["combined_semantic_id"]
        state._cpu_s46h_source_identity["combined_semantic_id"] = "tampered"
        with pytest.raises(Exception, match="S46H_SOURCE_IDENTITY_CHANGED"):
            verify_bindings(dispatch)
        state._cpu_s46h_source_identity["combined_semantic_id"] = semantic_id
        original.__code__ = original_code.replace(co_firstlineno=original_code.co_firstlineno + 1)
        with pytest.raises(Exception, match="S46H_SEAL_CODE"):
            verify_bindings(dispatch)
    finally:
        original.__code__ = original_code
        finalize_factory(dispatch)


def test_s46h_source_identity_contains_all_three_closures_and_semantic_id() -> None:
    identity = source_identity()
    assert identity["combined_semantic_id"] == COMBINED_SEMANTIC_ID
    assert set(identity["source_closure"]) == {"intensity_p24", "eq10_p48", "phase_absorb_c_p24"}
    assert identity["pool_workers_hpc"] == {"intensity": 24, "eq10": 48, "c": 24}
    owned = _owned_source_identity()
    assert owned["files"]["combined_candidate.py"]["raw_sha256"]
    assert HPC_CONTRACT["eq10_workers"] == 48 and HPC_CONTRACT["intensity_workers"] == HPC_CONTRACT["c_workers"] == 24


@pytest.mark.parametrize("suffix", ("guard", "close_failure"))
@pytest.mark.parametrize("tamper", ("code", "closure"))
def test_s46h_runtime_guard_and_cleanup_seals_fail_before_stage(
    suffix, tamper, monkeypatch, tmp_path
):
    _set_local_factory_environment(monkeypatch)
    from tools.cpu_s4_5b_steady_cost.adapter import make_recorder_pair

    built = build_candidate(
        "PURE_OFF_SMALLGRID", capture_root=tmp_path / "X",
        local_engineering=True, pool_override=1,
        recorder_pair=make_recorder_pair(False),
    )
    dispatch = built[1]
    state = dispatch.cpu_s4_6h_state
    function = getattr(state, "_cpu_s46h_" + suffix)
    code = function.__code__
    cell = function.__closure__[0]
    contents = cell.cell_contents
    try:
        if tamper == "code":
            function.__code__ = code.replace(co_firstlineno=code.co_firstlineno + 1)
        else:
            cell.cell_contents = object()
        with pytest.raises(Exception, match="S46H_SEAL_(CODE|CLOSURE)"):
            verify_bindings(dispatch)
        with pytest.raises(Exception, match="S46H_SEAL_(CODE|CLOSURE)"):
            state._cpu_s46h_stage_entry_guard()
        assert all(pool.snapshot()["closed"] for _, pool in dispatch.cpu_s4_6h_poolset.values())
    finally:
        function.__code__ = code
        cell.cell_contents = contents
        finalize_factory(dispatch)
