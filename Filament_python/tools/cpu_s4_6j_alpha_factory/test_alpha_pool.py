from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from tools.cpu_s4_6j_alpha_factory import alpha_pool as ap
from tools.cpu_s4_6j_alpha_factory.alpha_pool import (
    AlphaPool,
    AlphaPoolDomainError,
    AlphaPoolError,
    clean_alpha,
    clean_source,
    clip_alpha,
    ratio,
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


def _assert_ieee_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    """Compare values and classifications, then compare zero sign bits."""

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
        actual_bits = actual[both_zero].view(np.uint64)
        expected_bits = expected[both_zero].view(np.uint64)
        np.testing.assert_array_equal(actual_bits, expected_bits)


def _assert_scalar_only(value: object) -> None:
    if isinstance(value, np.ndarray):
        raise AssertionError("scientific array retained in activity record")
    if isinstance(value, dict):
        for item in value.values():
            _assert_scalar_only(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_scalar_only(item)


def test_five_rounds_are_exact_and_preserve_lifetimes() -> None:
    source = _source()
    original = source.copy()
    intensity = np.abs(source.copy()) + 0.25
    intensity[0, 0, 0] = 0.0
    intensity_before = intensity.copy()
    pool = AlphaPool(3)
    try:
        cleaned = clean_source(source, pool=pool)
        expected_cleaned = np.nan_to_num(
            original, nan=0.0, posinf=0.0, neginf=0.0, copy=True
        )
        _assert_ieee_equal(cleaned, expected_cleaned)
        assert cleaned is not source
        assert not np.shares_memory(cleaned, source)
        _assert_ieee_equal(source, original)
        cleaned_before_ratio = cleaned.copy()

        with np.errstate(all="ignore"):
            quotient = ratio(cleaned, intensity, 0.5, pool=pool)
        expected_ratio = cleaned / (intensity + 0.5)
        _assert_ieee_equal(quotient, expected_ratio)
        assert not np.shares_memory(quotient, cleaned)
        _assert_ieee_equal(intensity, intensity_before)
        _assert_ieee_equal(cleaned, cleaned_before_ratio)
        quotient_before_cleanup = quotient.copy()

        cleaned_alpha = clean_alpha(quotient, pool=pool)
        expected_alpha = np.nan_to_num(
            expected_ratio, nan=0.0, posinf=0.0, neginf=0.0, copy=True
        )
        _assert_ieee_equal(cleaned_alpha, expected_alpha)
        assert not np.shares_memory(cleaned_alpha, quotient)
        _assert_ieee_equal(quotient, quotient_before_cleanup)

        clip_input = cleaned_alpha.copy()
        clip_input[0, 0, 0] = -3.0
        clip_expected = clip_input.copy()
        np.maximum(clip_expected, 0.0, out=clip_expected)
        external_view = clip_input.view()
        identity = id(clip_input)
        clipped = clip_alpha(clip_input, pool=pool)
        _assert_ieee_equal(clipped, clip_expected)
        _assert_ieee_equal(external_view, clip_expected)
        assert id(clipped) == identity

        snapshot = pool.snapshot()
        assert snapshot["operation_counts"] == {
            "ALPHA_CLEAN_SOURCE": 1,
            "ALPHA_DENOM": 1,
            "ALPHA_RATIO": 1,
            "ALPHA_CLEAN_RESULT": 1,
            "ALPHA_CLIP": 1,
        }
        assert snapshot["total_rounds"] == 5
        assert snapshot["total_records"] == 5
        assert snapshot["dropped_records"] == 0
        assert snapshot["invalid_records"] == 0
        assert snapshot["scalar_sums"]["items"] == 5 * source.shape[0]
        for row in pool.take_records():
            _assert_scalar_only(row)
    finally:
        pool.close()


def test_zero_inf_nan_and_signed_zero_match_numpy() -> None:
    source = np.array([[[np.nan, np.inf, -np.inf, -0.0, 0.0]]], dtype=np.float64)
    intensity = np.zeros_like(source)
    pool = AlphaPool(1)
    try:
        cleaned = clean_source(source, pool=pool)
        with np.errstate(all="ignore"):
            quotient = ratio(cleaned, intensity, 0.0, pool=pool)
            expected = np.divide(cleaned, intensity + 0.0)
        _assert_ieee_equal(quotient, expected)
    finally:
        pool.close()


def test_extreme_float64_values_and_old_outputs_remain_stable() -> None:
    limits = np.finfo(np.float64)
    source = np.array(
        [
            [[limits.max, -limits.max, limits.tiny]],
            [[-limits.tiny, -0.0, 0.0]],
            [[np.nan, np.inf, -np.inf]],
            [[-1.0, 1.0, -3.0]],
            [[limits.eps, -limits.eps, 2.0]],
        ],
        dtype=np.float64,
    )
    intensity = np.full(source.shape, 2.0, dtype=np.float64)
    pool = AlphaPool(3)
    try:
        with np.errstate(all="ignore"):
            cleaned = clean_source(source, pool=pool)
            quotient = ratio(cleaned, intensity, 0.5, pool=pool)
            cleaned_alpha = clean_alpha(quotient, pool=pool)
        expected_cleaned = np.nan_to_num(
            source, nan=0.0, posinf=0.0, neginf=0.0, copy=True
        )
        expected_ratio = expected_cleaned / (intensity + 0.5)
        expected_alpha = np.nan_to_num(
            expected_ratio, nan=0.0, posinf=0.0, neginf=0.0, copy=True
        )
        _assert_ieee_equal(cleaned, expected_cleaned)
        _assert_ieee_equal(quotient, expected_ratio)
        _assert_ieee_equal(cleaned_alpha, expected_alpha)

        old_output = cleaned_alpha.copy()
        clip_target = cleaned_alpha.copy()
        clip_target[3, 0, 2] = -limits.max
        expected_clip = clip_target.copy()
        np.maximum(expected_clip, 0.0, out=expected_clip)
        external_view = clip_target.view()
        clipped = clip_alpha(clip_target, pool=pool)
        _assert_ieee_equal(clipped, expected_clip)
        _assert_ieee_equal(external_view, expected_clip)
        _assert_ieee_equal(cleaned_alpha, old_output)
    finally:
        pool.close()


def test_clip_rejects_readonly_and_bad_layout_without_writing() -> None:
    pool = AlphaPool(1)
    try:
        readonly = np.ones((2, 2, 2), dtype=np.float64)
        readonly.setflags(write=False)
        with pytest.raises(AlphaPoolDomainError, match="WRITABLE"):
            clip_alpha(readonly, pool=pool)

        noncontiguous = np.ones((2, 2, 2), dtype=np.float64)[:, :, ::-1]
        with pytest.raises(AlphaPoolDomainError, match="C_CONTIGUOUS"):
            clean_source(noncontiguous, pool=pool)
    finally:
        pool.close()


def test_numpy_error_strategy_is_preserved_and_retires_on_task_failure() -> None:
    callback_calls: list[str] = []
    pool = AlphaPool(3, on_failure=lambda: callback_calls.append("failed"))
    source = np.zeros((3, 2, 2), dtype=np.float64)
    try:
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            with pytest.raises(FloatingPointError):
                ratio(source, source, 0.0, pool=pool)
        assert pool.snapshot()["closed"] is True
        assert callback_calls == ["failed"]
        with pytest.raises(AlphaPoolError, match="RETIRED"):
            clean_source(source, pool=pool)
    finally:
        pool.close()

    task_pool = AlphaPool(3, on_failure=lambda: callback_calls.append("task"))
    try:
        with pytest.raises(RuntimeError, match="task failure"):
            task_pool.run_round(
                "ALPHA_CLEAN_SOURCE",
                source,
                lambda value: (_ for _ in ()).throw(RuntimeError("task failure")),
            )
        assert task_pool.snapshot()["closed"] is True
        assert callback_calls[-1] == "task"
    finally:
        task_pool.close()


def test_invalid_activity_record_keeps_scalar_evidence_before_retirement() -> None:
    pool = AlphaPool(1)
    source = np.ones((1, 1, 1), dtype=np.float64)
    invalid_row = {
        "label": "ALPHA_CLEAN_SOURCE",
        "requested_workers": np.int64(1),
        "workers": [
            {
                "thread_id": np.int64(11),
                "items": np.int64(1),
                "wall_ns": np.int64(7),
                "thread_cpu_ns": np.int64(5),
                "debug_array": np.ones((1, 1, 1), dtype=np.float64),
            }
        ],
    }
    pool._workers.take_records = lambda: [invalid_row]  # type: ignore[method-assign]
    try:
        with pytest.raises(AlphaPoolError, match="INVALID_ACTIVITY_RECORD"):
            pool.run_round("ALPHA_CLEAN_SOURCE", source, lambda value: value.copy())
        snapshot = pool.snapshot()
        assert snapshot["closed"] is True
        assert snapshot["total_records"] == 1
        assert snapshot["invalid_records"] == 1
        assert snapshot["dropped_records"] == 0
        assert snapshot["attempted_operation_counts"]["ALPHA_CLEAN_SOURCE"] == 1
        assert snapshot["completed_operation_counts"]["ALPHA_CLEAN_SOURCE"] == 1
        assert snapshot["record_failure_counts"]["ALPHA_CLEAN_SOURCE"] == 1
        rows = pool.take_records()
        assert rows and rows[0]["invalid_record"] is True
        _assert_scalar_only(rows[0])
    finally:
        pool.close()


def test_close_is_idempotent_and_post_close_calls_are_rejected() -> None:
    pool = AlphaPool(1)
    source = np.ones((1, 1, 1), dtype=np.float64)
    first = pool.close()
    second = pool.close()
    assert first["closed"] is True
    assert second["closed"] is True
    with pytest.raises(AlphaPoolError, match="RETIRED"):
        clean_source(source, pool=pool)


def test_constructor_failure_closes_partial_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    real_executor = ap.ThreadPoolExecutor
    holder: dict[str, object] = {}

    class FailingExecutor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.inner = real_executor(*args, **kwargs)
            self.calls = 0
            holder["executor"] = self.inner

        def submit(self, *args: object, **kwargs: object):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("second submit failure")
            return self.inner.submit(*args, **kwargs)

        def shutdown(self, *args: object, **kwargs: object) -> None:
            self.inner.shutdown(*args, **kwargs)

    monkeypatch.setattr(ap, "ThreadPoolExecutor", FailingExecutor)
    with pytest.raises(RuntimeError, match="second submit failure"):
        AlphaPool(3)
    assert holder["executor"] is not None
    started_threads = tuple(getattr(holder["executor"], "_threads", ()))
    assert started_threads
    assert all(not thread.is_alive() for thread in started_threads)


def test_constructor_cleanup_keeps_primary_submit_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    real_executor = ap.ThreadPoolExecutor
    holder: dict[str, object] = {}

    class FailingCleanupExecutor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.inner = real_executor(*args, **kwargs)
            self.calls = 0
            holder["executor"] = self.inner

        def submit(self, *args: object, **kwargs: object):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("primary submit failure")
            return self.inner.submit(*args, **kwargs)

        def shutdown(self, *args: object, **kwargs: object) -> None:
            self.inner.shutdown(*args, **kwargs)
            raise RuntimeError("cleanup shutdown failure")

    monkeypatch.setattr(ap, "ThreadPoolExecutor", FailingCleanupExecutor)
    with pytest.raises(RuntimeError, match="primary submit failure") as caught:
        AlphaPool(3)
    assert any(
        "CONSTRUCTOR_CLEANUP_FAILED" in note
        for note in getattr(caught.value, "__notes__", [])
    )
    assert all(not thread.is_alive() for thread in getattr(holder["executor"], "_threads", ()))


def test_runtime_partial_submit_aborts_barrier_and_drains_tasks() -> None:
    pool = AlphaPool(3)
    inner = pool._workers.executor
    calls = 0

    class PartialSubmitExecutor:
        def __init__(self, wrapped):
            self.inner = wrapped

        @property
        def _threads(self):
            return self.inner._threads

        def submit(self, *args: object, **kwargs: object):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("runtime second submit failure")
            return self.inner.submit(*args, **kwargs)

        def shutdown(self, *args: object, **kwargs: object) -> None:
            return self.inner.shutdown(*args, **kwargs)

    pool._workers.executor = PartialSubmitExecutor(inner)
    source = np.ones((5, 2, 3), dtype=np.float64)
    try:
        with pytest.raises(RuntimeError, match="runtime second submit failure"):
            pool.run_round("ALPHA_CLEAN_SOURCE", source, lambda value: value.copy())
        snapshot = pool.snapshot()
        assert snapshot["closed"] is True
        assert snapshot["attempted_operation_counts"]["ALPHA_CLEAN_SOURCE"] == 1
        assert snapshot["completed_operation_counts"]["ALPHA_CLEAN_SOURCE"] == 0
        assert snapshot["active_thread_ids"] == []
    finally:
        pool.close()


def test_close_failure_can_retry_join_and_preserves_first_error() -> None:
    pool = AlphaPool(1)
    inner = pool._workers.executor
    calls = 0

    class RetryCloseExecutor:
        def __init__(self, wrapped):
            self.inner = wrapped

        @property
        def _threads(self):
            return self.inner._threads

        def submit(self, *args: object, **kwargs: object):
            return self.inner.submit(*args, **kwargs)

        def shutdown(self, *args: object, **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("first close failure")
            return self.inner.shutdown(*args, **kwargs)

    pool._workers.executor = RetryCloseExecutor(inner)
    try:
        with pytest.raises(RuntimeError, match="first close failure"):
            pool.close()
        failed = pool.snapshot()
        assert failed["closed"] is True
        assert failed["close_failed"] is True
        assert failed["close_retryable"] is True
        assert failed["close_error"] == "RuntimeError"
        assert failed["active_thread_ids"]
        closed = pool.close()
        assert closed["closed"] is True
        assert closed["close_failed"] is False
        assert closed["close_retryable"] is False
        assert closed["close_error"] == "RuntimeError"
        assert closed["active_thread_ids"] == []
    finally:
        pool.close()


def test_round_rejects_wrong_output_dtype_and_marks_incomplete_round() -> None:
    pool = AlphaPool(1)
    source = np.ones((1, 1, 1), dtype=np.float64)
    try:
        with pytest.raises(AlphaPoolError, match="OPERATION_OUTPUT"):
            pool.run_round(
                "ALPHA_CLEAN_SOURCE",
                source,
                lambda value: value.astype(np.float32, copy=True),
            )
        snapshot = pool.snapshot()
        assert snapshot["attempted_operation_counts"]["ALPHA_CLEAN_SOURCE"] == 1
        assert snapshot["completed_operation_counts"]["ALPHA_CLEAN_SOURCE"] == 0
    finally:
        pool.close()


def test_ratio_does_not_retain_input_arrays_after_return() -> None:
    source = np.ones((5, 2, 2), dtype=np.float64)
    intensity = np.ones_like(source)
    source_ref = weakref.ref(source)
    intensity_ref = weakref.ref(intensity)
    pool = AlphaPool(3)
    denominator_refs: list[weakref.ReferenceType[np.ndarray]] = []
    original_run_round = pool.run_round

    def capture_denominator(label: str, source_value: np.ndarray, operation, **kwargs):
        result = original_run_round(label, source_value, operation, **kwargs)
        if label == "ALPHA_DENOM":
            denominator_refs.append(weakref.ref(result))
        return result

    pool.run_round = capture_denominator  # type: ignore[method-assign]
    try:
        cleaned = clean_source(source, pool=pool)
        result = ratio(cleaned, intensity, 1.0, pool=pool)
        assert result is not cleaned
        assert result is not intensity
        gc.collect()
        assert denominator_refs and denominator_refs[0]() is None
        assert pool.snapshot()["closed"] is False
    finally:
        pool.close()
    del result, cleaned, source, intensity
    gc.collect()
    assert source_ref() is None
    assert intensity_ref() is None
