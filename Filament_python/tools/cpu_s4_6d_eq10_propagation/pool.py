"""Bounded, source-owned workers for the four Eq10 pointwise operations."""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Callable

import numpy as np

from tools.cpu_s4_6c_array_coverage.chains import WORKERS, Workers


MAX_RECORDS = 128
POINTWISE_LABELS = (
    "B_SPECTRAL_MULTIPLY",
    "B_DERIVATIVE_DIVIDE",
    "B_RESPONSE_DIVIDE",
    "B_PRODUCT",
)


class PoolError(RuntimeError):
    """A private Eq10 pool failed closed."""


class SafeWorkers(Workers):
    """S4-6C workers with constructor-failure cleanup."""

    def __init__(self, count: int) -> None:
        self.executor = None
        try:
            super().__init__(count)
        except BaseException:
            executor = getattr(self, "executor", None)
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            raise


class Eq10Pool:
    """Own one persistent S4-6C worker pool and only scalar activity."""

    def __init__(self, count: int, *, max_records: int = MAX_RECORDS) -> None:
        if type(count) is not int or count not in WORKERS:
            raise PoolError("EQ10_WORKER_CONFIGURATION")
        if type(max_records) is not int or max_records <= 0 or max_records > MAX_RECORDS:
            raise PoolError("EQ10_RECORD_CAPACITY")
        self.count = int(count)
        self.max_records = int(max_records)
        self.closed = False
        self._lock = threading.RLock()
        self._records: list[dict[str, Any]] = []
        self._labels: dict[str, int] = {label: 0 for label in POINTWISE_LABELS}
        self._close_error: str | None = None
        self._created_ns = time.perf_counter_ns()
        self._close_ns: int | None = None
        self._lifetime_ns: int | None = None
        worker = SafeWorkers(self.count)
        self._workers = worker
        self.initialization_ns = int(worker.initialization_ns)

    def _require_open(self) -> None:
        if self.closed:
            raise PoolError("EQ10_RETIRED_WORKER_POOL")

    def _record(self, label: str) -> None:
        self._labels[label] = self._labels.get(label, 0) + 1
        rows = self._workers.take_records()
        if rows:
            remaining = self.max_records - len(self._records)
            if len(rows) > remaining:
                raise PoolError("EQ10_RECORD_CAPACITY_EXHAUSTED")
            self._records.extend(copy.deepcopy(rows))

    def ufunc(
        self,
        label: str,
        function: Callable[..., Any],
        operands: tuple[Any, ...],
        output: np.ndarray,
    ) -> np.ndarray:
        """Run one allowlisted S4-6C ufunc and retain scalar records only."""

        if label not in POINTWISE_LABELS:
            raise PoolError("EQ10_POINTWISE_LABEL")
        with self._lock:
            self._require_open()
            self._validate_call(label, function, operands, output)
            try:
                self._workers.ufunc(label, function, operands, output)
                self._record(label)
            except BaseException:
                # Workers waits for submitted tasks before propagating an
                # exception.  Retire this pool so no later stage can reuse a
                # failed owner or retain a partially completed operation.
                self.close()
                raise
        return output

    @staticmethod
    def _validate_call(
        label: str,
        function: Callable[..., Any],
        operands: tuple[Any, ...],
        output: np.ndarray,
    ) -> None:
        expected = {
            "B_SPECTRAL_MULTIPLY": np.multiply,
            "B_DERIVATIVE_DIVIDE": np.divide,
            "B_RESPONSE_DIVIDE": np.divide,
            "B_PRODUCT": np.multiply,
        }[label]
        if function is not expected:
            raise PoolError("EQ10_BACKEND_CALLABLE_CHANGED")
        if type(output) is not np.ndarray or not output.flags.c_contiguous or not output.flags.owndata:
            raise PoolError("EQ10_OUTPUT_LAYOUT_OR_OWNERSHIP")
        if output.dtype not in (
            np.dtype("complex64"),
            np.dtype("complex128"),
            np.dtype("float32"),
            np.dtype("float64"),
        ) or output.ndim != 3:
            raise PoolError("EQ10_OUTPUT_DTYPE_OR_RANK")
        for operand in operands:
            if isinstance(operand, np.ndarray):
                if type(operand) is not np.ndarray or operand.ndim != 3:
                    raise PoolError("EQ10_INPUT_BACKEND_OR_RANK")
                if np.shares_memory(operand, output):
                    raise PoolError("EQ10_INPUT_OUTPUT_ALIAS")
        if label == "B_SPECTRAL_MULTIPLY":
            if len(operands) != 2 or not all(isinstance(item, np.ndarray) for item in operands):
                raise PoolError("EQ10_SPECTRAL_OPERANDS")
            if operands[1].shape != output.shape or operands[0].shape[0] != output.shape[0]:
                raise PoolError("EQ10_SPECTRAL_SHAPE")
        elif label in ("B_DERIVATIVE_DIVIDE", "B_RESPONSE_DIVIDE"):
            if len(operands) != 2 or not isinstance(operands[0], np.ndarray):
                raise PoolError("EQ10_DIVIDE_OPERANDS")
            if operands[0].shape != output.shape:
                raise PoolError("EQ10_DIVIDE_SHAPE")
        elif label == "B_PRODUCT":
            if len(operands) != 2 or not all(isinstance(item, np.ndarray) for item in operands):
                raise PoolError("EQ10_PRODUCT_OPERANDS")
            if operands[0].shape != output.shape or operands[1].shape != output.shape:
                raise PoolError("EQ10_PRODUCT_SHAPE")

    def take_records(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._records
            self._records = []
            return copy.deepcopy(rows)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": "filament.cpu_s4_6d.eq10_pool.v1",
                "workers": self.count,
                "closed": bool(self.closed),
                "initialization_ns": int(self.initialization_ns),
                "close_ns": None if self._close_ns is None else int(self._close_ns),
                "lifetime_ns": None if self._lifetime_ns is None else int(self._lifetime_ns),
                "close_error": self._close_error,
                "pending_scalar_records": len(self._records),
                "operation_counts": dict(self._labels),
                "active_thread_ids": list(getattr(self._workers, "initial_thread_ids", ())),
                "records_capacity": self.max_records,
                "array_values_retained": False,
            }

    def close(self) -> dict[str, Any]:
        """Idempotently join and retire the pool."""

        with self._lock:
            if self.closed:
                return self.snapshot()
            shutdown_started = time.perf_counter_ns()
            try:
                self._workers.close()
            except BaseException as exc:
                self._close_error = type(exc).__name__
                self.closed = True
                finished = time.perf_counter_ns()
                self._close_ns = finished - shutdown_started
                self._lifetime_ns = finished - self._created_ns
                raise
            self.closed = True
            finished = time.perf_counter_ns()
            self._close_ns = finished - shutdown_started
            self._lifetime_ns = finished - self._created_ns
            return self.snapshot()


class NullEq10Pool:
    """No-op role companion used by the L1 reference role."""

    count = 0
    closed = False

    def take_records(self) -> list[dict[str, Any]]:
        return []

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "filament.cpu_s4_6d.eq10_pool.v1",
            "workers": 0,
            "closed": bool(self.closed),
            "initialization_ns": 0,
            "close_ns": 0,
            "lifetime_ns": 0,
            "close_error": None,
            "pending_scalar_records": 0,
            "operation_counts": {label: 0 for label in POINTWISE_LABELS},
            "active_thread_ids": [],
            "records_capacity": 0,
            "array_values_retained": False,
        }

    def close(self) -> dict[str, Any]:
        self.closed = True
        return self.snapshot()


__all__ = [
    "Eq10Pool",
    "NullEq10Pool",
    "PoolError",
    "SafeWorkers",
    "MAX_RECORDS",
    "POINTWISE_LABELS",
]
