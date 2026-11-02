"""Focused tests for the selected P24 power-volume factory layer."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from KHz_filament import deposition
from tools.cpu_s4_6i_residual_parallel.power_hook import (
    DIRECT_IB_HELPER,
    DIRECT_ION_HELPER,
    INTERVAL_KW,
    Q_IB_HELPER,
    Q_ION_HELPER,
    PowerHookError,
    PowerVolumeHook,
)
from tools.cpu_s4_6i_residual_parallel.power_volume import (
    POWER_VOLUME_MAX_RECORDS,
    PowerVolumeDomainError,
    make_power_volume_pool,
    power_volume_cleanup_p24,
)
from tools.cpu_s4_6i_residual_parallel.selected_factory import (
    ROLE_CANDIDATE,
    ROLE_REFERENCE,
    SelectedFactoryError,
    build_role,
    finalize_factory,
    verify_bindings,
)


def _edge_source() -> np.ndarray:
    return np.array(
        [
            [[np.nan, -1.0, np.inf], [-np.inf, 0.0, -0.0]],
            [[2.0, -3.0, 4.0], [5.0, 6.0, -7.0]],
        ],
        dtype=np.float64,
    )


@pytest.mark.parametrize('count', (1,24))
def test_power_volume_exact_readonly_alias_and_old_lifetime(count) -> None:
    source = _edge_source()
    before = source.copy()
    source.setflags(write=False)
    pool = make_power_volume_pool(count)
    try:
        first = power_volume_cleanup_p24(source, pool=pool, context="q_ion", interval_index=1)
        first_copy = first.copy()
        second = power_volume_cleanup_p24(source, pool=pool, context="q_ib", interval_index=2)
        reference = deposition._power_volume(source)
        assert first.dtype == np.dtype("float64")
        assert first.flags.c_contiguous and first.flags.owndata
        assert not np.shares_memory(source, first)
        assert not np.shares_memory(source, second)
        assert np.array_equal(first, reference, equal_nan=True)
        assert np.array_equal(second, reference, equal_nan=True)
        assert np.array_equal(first, first_copy, equal_nan=True)
        assert np.array_equal(source, before, equal_nan=True)
        assert pool.snapshot()["dropped_records"] == 0
    finally:
        first_close = pool.close()
        second_close = pool.close()
    assert first_close["closed"] and second_close["closed"]
    assert second_close["active_thread_ids"] == []


@pytest.mark.parametrize(
    "source",
    [
        np.ones((2, 2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float64),
        np.ones((0, 2, 2), dtype=np.float64),
        np.asfortranarray(np.ones((2, 2, 2), dtype=np.float64)),
    ],
)
def test_power_volume_rejects_unsupported_domain(source: np.ndarray) -> None:
    pool = make_power_volume_pool()
    try:
        with pytest.raises(PowerVolumeDomainError):
            power_volume_cleanup_p24(source, pool=pool)
    finally:
        pool.close()


def test_power_volume_mechanism_drain_keeps_all_scalar_rows() -> None:
    source = np.ones((1, 1, 1), dtype=np.float64)
    pool = make_power_volume_pool(1)
    try:
        for index in range((POWER_VOLUME_MAX_RECORDS // 2) + 6):
            power_volume_cleanup_p24(source, pool=pool, context="q_ion", interval_index=index)
            assert len(pool.take_records()) == 2
        snapshot = pool.snapshot()
        assert snapshot["total_records"] == POWER_VOLUME_MAX_RECORDS + 12
        assert snapshot["dropped_records"] == 0
        assert snapshot["invalid_records_dropped"] == 0
        assert snapshot["pending_scalar_records"] == 0
        assert snapshot["array_values_retained"] is False
    finally:
        pool.close()


def test_power_volume_task_failure_closes_pool_and_notifies_owner() -> None:
    class BrokenBackend:
        asarray = staticmethod(np.asarray)

        @staticmethod
        def nan_to_num(*args, **kwargs):
            raise RuntimeError("injected-round-failure")

        maximum = staticmethod(np.maximum)

    notified: list[bool] = []
    pool = make_power_volume_pool()
    with pytest.raises(RuntimeError, match="injected-round-failure"):
        power_volume_cleanup_p24(
            np.ones((2, 2, 2), dtype=np.float64),
            pool=pool,
            backend=BrokenBackend,
            on_failure=lambda: notified.append(True),
        )
    snapshot = pool.snapshot()
    assert notified == [True]
    assert snapshot["closed"] is True
    assert snapshot["active_thread_ids"] == []


def test_power_hook_ast_inverse_source_binding_and_capture() -> None:
    seen: list[tuple[str, int | None, str, tuple[int, ...]]] = []
    class Store:
        def before_call(self, *, context, interval_index, source, source_site):
            return context,interval_index,source_site,source.copy()
        def after_call(self, token, *, source, output):
            context,interval,site,before=token
            assert np.array_equal(source,before,equal_nan=True)
            assert not np.shares_memory(source,output)
            seen.append((context,interval,site,tuple(output.shape)))
    hook = PowerVolumeHook(None,capture_store=Store())
    try:
        path = Path(deposition.__file__).resolve().parent / "propagate.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "propagate_one_pulse")
        baseline = copy.deepcopy(function)
        receipt = hook.instrument(function)
        restored = hook.strip_generated_nodes(function)
        assert receipt["replacement_count"] == 4
        assert ast.dump(restored, include_attributes=False) == ast.dump(baseline, include_attributes=False)
        source = np.ones((2, 2, 2), dtype=np.float64)
        helpers = hook.runtime_helpers()
        result = helpers[Q_ION_HELPER](source, 1.0, **{INTERVAL_KW: 7})
        q_ib = helpers[Q_IB_HELPER](source, source, 1.0, **{INTERVAL_KW: 8})
        direct_ion = helpers[DIRECT_ION_HELPER](source, 1.0, 1.0, 1.0, 1.0, **{INTERVAL_KW: 9})
        direct_ib = helpers[DIRECT_IB_HELPER](source, 1.0, 1.0, 1.0, 1.0, **{INTERVAL_KW: 10})
        assert result.shape == (2, 2) and q_ib.shape == (2, 2)
        assert isinstance(direct_ion, float) and isinstance(direct_ib, float)
        assert [item[0] for item in seen] == ["q_ion", "q_ib", "direct_ion", "direct_ib"]
        assert hook.verify_binding()["formal_timing_enabled"] is False
        original = deposition._power_volume
        deposition._power_volume = lambda value: value  # type: ignore[assignment]
        try:
            with pytest.raises(PowerHookError):
                hook.verify_binding()
        finally:
            deposition._power_volume = original
    finally:
        hook.close()


def test_selected_factory_reference_and_candidate_close(tmp_path: Path) -> None:
    reference = build_role(
        "PURE_OFF_SMALLGRID",
        capture_root=tmp_path / "r",
        role=ROLE_REFERENCE,
        local_engineering=True,
        pool_override=1,
    )
    assert verify_bindings(reference[1])["role"] == ROLE_REFERENCE
    reference_final = finalize_factory(reference[1])
    assert "power_volume" not in reference_final
    candidate = build_role(
        "PURE_OFF_SMALLGRID",
        capture_root=tmp_path / "x",
        role=ROLE_CANDIDATE,
        local_engineering=True,
        pool_override=1,
    )
    assert verify_bindings(candidate[1])["role"] == ROLE_CANDIDATE
    assert candidate[1].cpu_s4_6i_power_pool.count == 1
    candidate_final = finalize_factory(candidate[1])
    assert candidate_final["power_volume"]["pool"]["closed"] is True
    assert candidate_final["all_pools_retired_before_output"] is True
    assert finalize_factory(candidate[1]) == candidate_final


def test_selected_finalization_close_error_does_not_cache_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = build_role(
        "PURE_OFF_SMALLGRID",
        capture_root=tmp_path / "close-error",
        role=ROLE_CANDIDATE,
        local_engineering=True,
        pool_override=1,
    )
    dispatch = candidate[1]
    hook = dispatch.cpu_s4_6i_power_hook
    pool = dispatch.cpu_s4_6i_power_pool

    def fail_close() -> None:
        raise RuntimeError("injected-close-failure")

    monkeypatch.setattr(hook, "close", fail_close)
    with pytest.raises(SelectedFactoryError):
        finalize_factory(dispatch)
    assert not hasattr(dispatch, "_cpu_s4_6i_finalized")
    # The injected method bypasses the pool's real shutdown; retire it in the
    # test cleanup without changing production lifecycle behavior.
    monkeypatch.undo()
    pool.close()
