"""Private persistent workers for the three intensity pointwise segments."""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Callable

import numpy as np

from tools.cpu_s4_6c_array_coverage.chains import WORKERS, Workers


MAX_RECORDS = 128
POINTWISE_LABELS = ("A_ABS", "A_SQUARE", "A_SCALE")


class PoolError(RuntimeError):
    """A private intensity pool failed closed."""


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


class IntensityPool:
    """Own one persistent worker pool and retain scalar activity only."""

    def __init__(self, count: int, *, max_records: int = MAX_RECORDS) -> None:
        if type(count) is not int or count not in WORKERS:
            raise PoolError("A_WORKER_CONFIGURATION")
        if type(max_records) is not int or max_records <= 0 or max_records > MAX_RECORDS:
            raise PoolError("A_RECORD_CAPACITY")
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
            raise PoolError("A_RETIRED_WORKER_POOL")

    def _record(self, label: str) -> None:
        self._labels[label] = self._labels.get(label, 0) + 1
        rows = self._workers.take_records()
        if rows:
            remaining = self.max_records - len(self._records)
            if len(rows) > remaining:
                raise PoolError("A_RECORD_CAPACITY_EXHAUSTED")
            self._records.extend(copy.deepcopy(rows))

    def ufunc(
        self,
        label: str,
        function: Callable[..., Any],
        operands: tuple[Any, ...],
        output: np.ndarray,
    ) -> np.ndarray:
        """Run one allowlisted NumPy ufunc over fixed axis-zero slices."""

        if label not in POINTWISE_LABELS:
            raise PoolError("A_POINTWISE_LABEL")
        with self._lock:
            self._require_open()
            self._validate_call(label, function, operands, output)
            try:
                self._workers.ufunc(label, function, operands, output)
                self._record(label)
            except BaseException:
                # Workers joins all submitted tasks before raising.  Retire
                # the owner so a failed operation cannot be reused.
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
        expected = {"A_ABS": np.absolute, "A_SQUARE": np.square, "A_SCALE": np.multiply}[label]
        if function is not expected:
            raise PoolError("A_BACKEND_CALLABLE_CHANGED")
        if (
            type(output) is not np.ndarray
            or output.ndim != 3
            or output.dtype != np.dtype("float64")
            or not output.flags.c_contiguous
            or not output.flags.owndata
        ):
            raise PoolError("A_OUTPUT_LAYOUT_DTYPE_OR_OWNERSHIP")
        if label == "A_ABS":
            if len(operands) != 1 or type(operands[0]) is not np.ndarray:
                raise PoolError("A_ABS_OPERANDS")
            source = operands[0]
            if source.dtype != np.dtype("complex128") or source.shape != output.shape or not source.flags.c_contiguous:
                raise PoolError("A_ABS_INPUT_DOMAIN")
        elif label == "A_SQUARE":
            if len(operands) != 1 or type(operands[0]) is not np.ndarray:
                raise PoolError("A_SQUARE_OPERANDS")
            source = operands[0]
            if source.dtype != np.dtype("float64") or source.shape != output.shape or not source.flags.c_contiguous:
                raise PoolError("A_SQUARE_INPUT_DOMAIN")
        else:
            if len(operands) != 2 or not isinstance(operands[0], (float, np.floating)):
                raise PoolError("A_SCALE_OPERANDS")
            source = operands[1]
            if type(source) is not np.ndarray or source.dtype != np.dtype("float64") or source.shape != output.shape or not source.flags.c_contiguous:
                raise PoolError("A_SCALE_INPUT_DOMAIN")
        for operand in operands:
            if isinstance(operand, np.ndarray) and np.shares_memory(operand, output):
                raise PoolError("A_INPUT_OUTPUT_ALIAS")

    def take_records(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._records
            self._records = []
            return copy.deepcopy(rows)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            executor = getattr(self._workers, "executor", None)
            live_threads = []
            for thread in getattr(executor, "_threads", ()):
                if thread.is_alive() and getattr(thread, "native_id", None) is not None:
                    live_threads.append(int(thread.native_id))
            return {
                "schema": "filament.cpu_s4_6e.intensity_pool.v1",
                "workers": self.count,
                "closed": bool(self.closed),
                "initialization_ns": int(self.initialization_ns),
                "close_ns": None if self._close_ns is None else int(self._close_ns),
                "lifetime_ns": None if self._lifetime_ns is None else int(self._lifetime_ns),
                "close_error": self._close_error,
                "pending_scalar_records": len(self._records),
                "operation_counts": dict(self._labels),
                "initial_thread_ids": list(getattr(self._workers, "initial_thread_ids", ())),
                "active_thread_ids": sorted(live_threads),
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


class NullIntensityPool:
    """Reference-role companion; it never submits candidate work."""

    count = 0
    closed = False

    def take_records(self) -> list[dict[str, Any]]:
        return []

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "filament.cpu_s4_6e.intensity_pool.v1",
            "workers": 0,
            "closed": bool(self.closed),
            "initialization_ns": 0,
            "close_ns": 0,
            "lifetime_ns": 0,
            "close_error": None,
            "pending_scalar_records": 0,
            "operation_counts": {label: 0 for label in POINTWISE_LABELS},
            "initial_thread_ids": [],
            "active_thread_ids": [],
            "records_capacity": 0,
            "array_values_retained": False,
        }

    def close(self) -> dict[str, Any]:
        self.closed = True
        return self.snapshot()


__all__ = [
    "IntensityPool",
    "NullIntensityPool",
    "PoolError",
    "SafeWorkers",
    "MAX_RECORDS",
    "POINTWISE_LABELS",
]
