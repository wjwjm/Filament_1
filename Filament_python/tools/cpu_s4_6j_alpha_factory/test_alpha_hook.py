"""Focused tests for the source-bound S4-6J alpha hook.

These tests deliberately stop at AST/helper and small-array engineering
boundaries.  They do not run a propagation or a production-size case.
"""

from __future__ import annotations

import ast
import copy
import gc
import weakref
from pathlib import Path

import numpy as np
import pytest

from KHz_filament import propagate
from tools.cpu_s4_6i_residual_parallel.power_volume import PowerVolumePool
from tools.cpu_s4_6j_alpha_factory import alpha_hook as ah
from tools.cpu_s4_6j_alpha_factory.alpha_hook import (
    ALPHA_CLEAN_RESULT_HELPER,
    ALPHA_CLEAN_SOURCE_HELPER,
    ALPHA_CLIP_HELPER,
    ALPHA_INTERVAL_KW,
    ALPHA_RATIO_HELPER,
    ALPHA_SOURCE_SITE,
    AlphaHookError,
    AlphaParallelHook,
    instrument_alpha_ast,
    strip_alpha_ast,
)
from tools.cpu_s4_6j_alpha_factory.alpha_pool import AlphaPool


def _propagate_function() -> ast.FunctionDef:
    path = Path(propagate.__file__).resolve()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "propagate_one_pulse"
    )


def _assert_ieee_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    actual_nan = np.isnan(actual)
    expected_nan = np.isnan(expected)
    assert np.array_equal(actual_nan, expected_nan)
    actual_inf = np.isinf(actual)
    expected_inf = np.isinf(expected)
    assert np.array_equal(actual_inf, expected_inf)
    finite = ~(actual_nan | actual_inf)
    np.testing.assert_array_equal(actual[finite], expected[finite])
    both_inf = actual_inf & expected_inf
    assert np.array_equal(np.signbit(actual[both_inf]), np.signbit(expected[both_inf]))
    both_zero = finite & (actual == 0.0) & (expected == 0.0)
    if np.any(both_zero):
        np.testing.assert_array_equal(
            actual[both_zero].view(np.uint64), expected[both_zero].view(np.uint64)
        )


def _source() -> np.ndarray:
    return np.array(
        [
            [[np.nan, np.inf, -np.inf], [-0.0, 0.0, 1.0]],
            [[-1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
            [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]],
        ],
        dtype=np.float64,
    )


def _make_hook(*, candidate: bool = True, capture_store=None) -> AlphaParallelHook:
    power_pool = PowerVolumePool(1)
    alpha_pool = AlphaPool(1) if candidate else None
    try:
        return AlphaParallelHook(
            power_pool,
            alpha_pool=alpha_pool,
            capture_store=capture_store,
        )
    except BaseException:
        power_pool.close()
        if alpha_pool is not None:
            alpha_pool.close()
        raise


def test_alpha_ast_replaces_four_statements_and_restores_joint_power_ast() -> None:
    original = _propagate_function()
    changed = copy.deepcopy(original)
    receipt = instrument_alpha_ast(changed)
    restored = strip_alpha_ast(changed)
    assert receipt["replacement_count"] == 4
    assert receipt["mathematical_changes"] == 0
    assert ast.dump(restored, include_attributes=False) == ast.dump(
        original, include_attributes=False
    )
    assert sum(
        isinstance(node, ast.Name) and node.id in ah._ALPHA_HELPER_SET
        for node in ast.walk(changed)
    ) == 4
    compile(ast.Module(body=[changed], type_ignores=[]), "<s46j-alpha-hook>", "exec")

    tampered = copy.deepcopy(changed)
    target = next(
        node
        for node in ast.walk(tampered)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == ALPHA_RATIO_HELPER
    )
    target.args[1] = ast.Name(id="wrong_intensity", ctx=ast.Load())
    with pytest.raises(AlphaHookError, match="ARGUMENTS_CHANGED"):
        strip_alpha_ast(tampered)

    duplicate = copy.deepcopy(changed)
    for body in ah._iter_statement_lists(duplicate):
        for index, statement in enumerate(body):
            if (
                isinstance(statement, ast.Assign)
                and isinstance(statement.value, ast.Call)
                and getattr(statement.value.func, "id", None) == ALPHA_CLEAN_SOURCE_HELPER
            ):
                body.insert(index + 1, copy.deepcopy(statement))
                break
        else:
            continue
        break
    with pytest.raises(AlphaHookError, match="HELPER_COUNT_CHANGED"):
        strip_alpha_ast(duplicate)

    moved = copy.deepcopy(changed)
    for body in ah._iter_statement_lists(moved):
        for index in range(len(body) - 3):
            if (
                isinstance(body[index], ast.Assign)
                and isinstance(body[index].value, ast.Call)
                and getattr(body[index].value.func, "id", None) == ALPHA_CLEAN_SOURCE_HELPER
            ):
                body[index], body[index + 1] = body[index + 1], body[index]
                break
        else:
            continue
        break
    with pytest.raises(AlphaHookError, match="PARENT_POSITION_CHANGED"):
        strip_alpha_ast(moved)


def test_hook_inverse_strips_alpha_before_existing_power_aliases() -> None:
    hook = _make_hook(candidate=True)
    try:
        original = _propagate_function()
        changed = copy.deepcopy(original)
        receipt = hook.instrument(changed)
        restored = hook.strip_generated_nodes(changed)
        assert receipt["alpha"]["replacement_count"] == 4
        assert ast.dump(restored, include_attributes=False) == ast.dump(
            original, include_attributes=False
        )
    finally:
        hook.close()


def test_reference_and_p24_helpers_are_exact_and_record_each_round() -> None:
    source = _source()
    source_before = source.copy()
    intensity = np.abs(source.copy()) + 0.25
    intensity[0, 0, 0] = 0.0
    intensity_before = intensity.copy()
    reference = _make_hook(candidate=False)
    candidate = _make_hook(candidate=True)
    try:
        ref = reference.runtime_helpers()
        cand = candidate.runtime_helpers()
        with np.errstate(all="ignore"):
            ref_source = ref[ALPHA_CLEAN_SOURCE_HELPER](
                source, **{ALPHA_INTERVAL_KW: 1}
            )
            cand_source = cand[ALPHA_CLEAN_SOURCE_HELPER](
                source, **{ALPHA_INTERVAL_KW: 1}
            )
            ref_ratio = ref[ALPHA_RATIO_HELPER](
                ref_source, intensity, 0.5, **{ALPHA_INTERVAL_KW: 1}
            )
            cand_ratio = cand[ALPHA_RATIO_HELPER](
                cand_source, intensity, 0.5, **{ALPHA_INTERVAL_KW: 1}
            )
            ref_clean = ref[ALPHA_CLEAN_RESULT_HELPER](
                ref_ratio, **{ALPHA_INTERVAL_KW: 1}
            )
            cand_clean = cand[ALPHA_CLEAN_RESULT_HELPER](
                cand_ratio, **{ALPHA_INTERVAL_KW: 1}
            )
        ref_clip = ref_clean.copy()
        cand_clip = cand_clean.copy()
        ref_clip[0, 0, 0] = -3.0
        cand_clip[0, 0, 0] = -3.0
        ref_clip_result = ref[ALPHA_CLIP_HELPER](ref_clip, **{ALPHA_INTERVAL_KW: 1})
        cand_clip_result = cand[ALPHA_CLIP_HELPER](cand_clip, **{ALPHA_INTERVAL_KW: 1})

        _assert_ieee_equal(cand_source, ref_source)
        _assert_ieee_equal(cand_ratio, ref_ratio)
        _assert_ieee_equal(cand_clean, ref_clean)
        _assert_ieee_equal(cand_clip_result, ref_clip_result)
        _assert_ieee_equal(source, source_before)
        _assert_ieee_equal(intensity, intensity_before)
        assert cand_clip_result is cand_clip
        assert candidate.take_alpha_records()
        rows = candidate.take_alpha_records()
        assert rows == []
    finally:
        reference.close()
        candidate.close()


def test_alpha_records_include_both_ratio_rounds_and_old_input_is_not_retained() -> None:
    source = _source()
    source_ref = weakref.ref(source)
    intensity = np.ones_like(source)
    hook = _make_hook(candidate=True)
    try:
        helper = hook.runtime_helpers()[ALPHA_CLEAN_SOURCE_HELPER]
        cleaned = helper(source, **{ALPHA_INTERVAL_KW: 2})
        hook.runtime_helpers()[ALPHA_RATIO_HELPER](
            cleaned, intensity, 0.5, **{ALPHA_INTERVAL_KW: 2}
        )
        rows = hook.take_alpha_records()
        assert [row["label"] for row in rows] == [
            "ALPHA_CLEAN_SOURCE",
            "ALPHA_DENOM",
            "ALPHA_RATIO",
        ]
        assert [row["helper"] for row in rows] == [
            ALPHA_CLEAN_SOURCE_HELPER,
            ALPHA_RATIO_HELPER,
            ALPHA_RATIO_HELPER,
        ]
        assert all(row["interval_index"] == 2 for row in rows)
        assert all(row["source"] == ALPHA_SOURCE_SITE for row in rows)
        del source
        gc.collect()
        assert source_ref() is None
    finally:
        hook.close()


def test_active_backend_and_kernel_binding_tampering_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    hook = _make_hook(candidate=True)
    try:
        helpers = hook.runtime_helpers()
        active = {"xp": np, **helpers}
        hook.bind_active(active)
        broken_active = dict(active)
        broken_active[ALPHA_CLIP_HELPER] = lambda *args, **kwargs: None
        with pytest.raises(AlphaHookError, match="ACTIVE_HELPER_CHANGED"):
            hook.verify_binding(broken_active)

        expected_divide = hook._alpha_backend["divide"]
        hook._alpha_backend["divide"] = object()
        with pytest.raises(AlphaHookError, match="BACKEND_CHANGED"):
            hook.verify_binding()
        hook._alpha_backend["divide"] = expected_divide

        original_clean = ah.alpha_kernel.clean_source
        monkeypatch.setattr(ah.alpha_kernel, "clean_source", lambda *args, **kwargs: None)
        with pytest.raises(AlphaHookError, match="KERNEL_BINDING_CHANGED"):
            hook.verify_binding()
        monkeypatch.setattr(ah.alpha_kernel, "clean_source", original_clean)
    finally:
        hook.close()


def test_saved_reference_namespace_and_instance_pool_methods_are_bound() -> None:
    hook = _make_hook(candidate=True)
    try:
        original_xp = hook._alpha_reference_namespace["xp"]
        hook._alpha_reference_namespace["xp"] = object()
        with pytest.raises(AlphaHookError, match="REFERENCE_BACKEND_BINDING_CHANGED"):
            hook.verify_binding()
        hook._alpha_reference_namespace["xp"] = original_xp

        original_run_round = hook._alpha_pool.run_round
        hook._alpha_pool.run_round = lambda *args, **kwargs: None
        with pytest.raises(AlphaHookError, match="INSTANCE_METHOD_CHANGED"):
            hook.verify_binding()
        hook._alpha_pool.run_round = original_run_round
        hook.verify_binding()
    finally:
        hook.close()


def test_alpha_drain_error_keeps_scalar_receipt_and_failure_count() -> None:
    hook = _make_hook(candidate=True)
    pool = hook._alpha_pool
    original_take = pool.take_records
    try:
        pool.take_records = lambda: [
            {"label": "WRONG", "workers": [], "array": np.ones((2, 2, 2))}
        ]
        source = np.ones((5, 2, 3), dtype=np.float64)
        with pytest.raises(AlphaHookError, match="ACTIVITY_ORDER_CHANGED"):
            hook.runtime_helpers()[ALPHA_CLEAN_SOURCE_HELPER](
                source, **{ALPHA_INTERVAL_KW: 0}
            )
        rows = hook.take_alpha_records()
        assert len(rows) == 1
        assert rows[0]["invalid_record"] is not False
        assert rows[0]["drain_error"] == "S46J_ALPHA_ACTIVITY_ORDER_CHANGED"
        assert hook.alpha_snapshot()["hook_record_failures"] == 1
    finally:
        pool.take_records = original_take
        hook.close()


def test_capture_store_is_explicit_and_failure_cleanup_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int | None, str]] = []

    class Store:
        def before_call(self, *, context, interval_index, source, source_site):
            calls.append((context, interval_index, source_site))
            return object()

        def after_call(self, token, *, source, output):
            assert token is not None
            assert isinstance(output, np.ndarray)

    store = Store()
    hook = _make_hook(candidate=True, capture_store=store)
    notified: list[str] = []
    try:
        hook.set_failure_cleanup(lambda: notified.append("failure"))
        source = np.ones((5, 2, 3), dtype=np.float64)
        hook.runtime_helpers()[ALPHA_CLEAN_SOURCE_HELPER](
            source, **{ALPHA_INTERVAL_KW: 3}
        )
        assert calls == [(ALPHA_CLEAN_SOURCE_HELPER, 3, ALPHA_SOURCE_SITE)]
        assert hook._alpha_pool._failure_cleanup is not None

        monkeypatch.setattr(ah.alpha_kernel, "clean_source", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("hook failure")))
        with pytest.raises(RuntimeError, match="hook failure"):
            hook.runtime_helpers()[ALPHA_CLEAN_SOURCE_HELPER](
                source, **{ALPHA_INTERVAL_KW: 3}
            )
        assert notified == ["failure"]
    finally:
        hook.close()


def test_capture_contract_receives_only_immediate_named_inputs_and_outputs() -> None:
    events = []

    class Store:
        def before_call(
            self,
            *,
            context,
            interval_index,
            source,
            source_site,
            capture_inputs=None,
        ):
            events.append(("before", context, capture_inputs))
            return object()

        def after_call(
            self,
            token,
            *,
            source,
            output,
            capture_inputs=None,
            capture_outputs=None,
        ):
            events.append(("after", capture_inputs, capture_outputs, source, output))

    source = _source()
    intensity = np.ones_like(source)
    hook = _make_hook(candidate=True, capture_store=Store())
    try:
        helpers = hook.runtime_helpers()
        cleaned = helpers[ALPHA_CLEAN_SOURCE_HELPER](
            source, **{ALPHA_INTERVAL_KW: 0}
        )
        ratio = helpers[ALPHA_RATIO_HELPER](
            cleaned, intensity, 0.5, **{ALPHA_INTERVAL_KW: 0}
        )
        clean = helpers[ALPHA_CLEAN_RESULT_HELPER](
            ratio, **{ALPHA_INTERVAL_KW: 0}
        )
        clean[0, 0, 0] = -1.0
        final = helpers[ALPHA_CLIP_HELPER](clean, **{ALPHA_INTERVAL_KW: 0})
        assert events[0][0] == "before"
        assert events[0][2]["source_input"] is source
        assert events[1][0] == "after"
        assert events[1][2]["cleaned_source_output"] is cleaned
        ratio_before = next(event for event in events if event[0] == "before" and event[1] == ALPHA_RATIO_HELPER)
        assert ratio_before[2]["intensity_input"] is intensity
        assert ratio_before[2]["I_floor"] == 0.5
        clip_after = [event for event in events if event[0] == "after"][-1]
        assert clip_after[2]["final_alpha"] is final
        assert clip_after[2]["inplace"] is True
    finally:
        hook.close()


def test_capture_before_failure_runs_cleanup_once() -> None:
    notified: list[str] = []

    class Store:
        def before_call(self, **kwargs):
            raise RuntimeError("capture before failure")

        def after_call(self, *args, **kwargs):
            raise AssertionError("after_call must not run")

    hook = _make_hook(candidate=True, capture_store=Store())
    hook.set_failure_cleanup(lambda: notified.append("failure"))
    try:
        with pytest.raises(RuntimeError, match="capture before failure"):
            hook.runtime_helpers()[ALPHA_CLEAN_SOURCE_HELPER](
                np.ones((5, 2, 3), dtype=np.float64), **{ALPHA_INTERVAL_KW: 0}
            )
        assert notified == ["failure"]
    finally:
        hook.close()
