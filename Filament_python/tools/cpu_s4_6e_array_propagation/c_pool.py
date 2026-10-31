"""Private persistent workers for the C/P24 nonlinear field updates.

The pool owns only bounded scalar activity records.  Scientific arrays are
passed to a task for the duration of one operation and are never retained by
the pool after the operation returns.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Callable

import numpy as np

from tools.cpu_s4_6c_array_coverage.chains import WORKERS, Workers


MAX_RECORDS = 128
POINTWISE_LABELS = ("C_PHASE", "C_ABSORPTION")


class CPoolError(RuntimeError):
    """The private C pool rejected an operation or failed closed."""


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


def _is_ndarray(value: Any) -> bool:
    return type(value) is np.ndarray


class CPool:
    """One persistent P24 pool for the two original nonlinear expressions."""

    def __init__(self, count: int, *, max_records: int = MAX_RECORDS) -> None:
        if type(count) is not int or count not in WORKERS:
            raise CPoolError("C_WORKER_CONFIGURATION")
        if type(max_records) is not int or max_records <= 0 or max_records > MAX_RECORDS:
            raise CPoolError("C_RECORD_CAPACITY")
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
            raise CPoolError("C_RETIRED_WORKER_POOL")

    @staticmethod
    def _require_cpu_array(value: Any, *, dtype: str, shape: tuple[int, ...], name: str) -> None:
        if (
            not _is_ndarray(value)
            or value.dtype != np.dtype(dtype)
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise CPoolError(f"C_{name}_DOMAIN")

    @classmethod
    def validate_domain(
        cls,
        E: Any,
        phase: Any,
        alpha: Any,
        *,
        dn_gas: Any = None,
    ) -> None:
        """Validate all arrays before the first in-place field write."""

        if (
            not _is_ndarray(E)
            or E.ndim != 3
            or E.dtype != np.dtype("complex128")
            or not E.flags.c_contiguous
            or not E.flags.writeable
        ):
            raise CPoolError("C_E_LAYOUT_DTYPE_OR_WRITABLE")
        shape = tuple(int(item) for item in E.shape)
        # Report aliasing before layout so a shared, non-contiguous view cannot
        # be mistaken for an ordinary unsupported dtype/layout input.
        for name, value in (("PHASE", phase), ("ALPHA", alpha)):
            if _is_ndarray(value) and np.shares_memory(value, E):
                raise CPoolError(f"C_{name}_E_ALIAS")
        if dn_gas is not None and _is_ndarray(dn_gas) and np.shares_memory(dn_gas, E):
            raise CPoolError("C_DN_GAS_E_ALIAS")
        cls._require_cpu_array(phase, dtype="float64", shape=shape, name="PHASE")
        cls._require_cpu_array(alpha, dtype="float64", shape=shape, name="ALPHA")
        if dn_gas is not None:
            expected = shape[1:]
            cls._require_cpu_array(dn_gas, dtype="float64", shape=expected, name="DN_GAS")

    def _record(self, label: str) -> None:
        self._labels[label] = self._labels.get(label, 0) + 1
        rows = self._workers.take_records()
        if rows:
            remaining = self.max_records - len(self._records)
            if len(rows) > remaining:
                raise CPoolError("C_RECORD_CAPACITY_EXHAUSTED")
            self._records.extend(copy.deepcopy(rows))

    def _run(self, label: str, E: np.ndarray, operation: Callable[[int, int], None]) -> None:
        if label not in POINTWISE_LABELS:
            raise CPoolError("C_POINTWISE_LABEL")
        with self._lock:
            self._require_open()
            if not _is_ndarray(E) or E.ndim != 3 or not E.flags.c_contiguous:
                raise CPoolError("C_E_OPERATION_LAYOUT")
            try:
                self._workers.run(label, int(E.shape[0]), operation)
                self._record(label)
            except BaseException:
                # Workers.run waits for every submitted task before exposing
                # the first task exception.  A failed operation retires the
                # pool so it cannot be reused by a failed propagation.
                try:
                    self.close()
                except BaseException:
                    pass
                raise

    def phase_update(
        self,
        E: np.ndarray,
        worker: Callable[[int, int], None],
        *,
        phase: np.ndarray | None = None,
        ctype: Any = np.dtype("complex128"),
    ) -> np.ndarray:
        """Run the first source-bound AST worker over axis-0 slices."""

        self._require_open()
        if not callable(worker):
            raise CPoolError("C_PHASE_WORKER")
        try:
            if np.dtype(ctype) != np.dtype("complex128"):
                raise CPoolError("C_COMPLEX_DTYPE")
        except (TypeError, ValueError) as exc:
            raise CPoolError("C_COMPLEX_DTYPE") from exc
        if phase is not None:
            self._require_cpu_array(phase, dtype="float64", shape=tuple(E.shape), name="PHASE")
            if np.shares_memory(phase, E):
                raise CPoolError("C_PHASE_E_ALIAS")
        self._run("C_PHASE", E, worker)
        return E

    def absorption_update(
        self,
        E: np.ndarray,
        worker: Callable[[int, int], None],
        *,
        alpha: np.ndarray | None = None,
        ctype: Any = np.dtype("complex128"),
    ) -> np.ndarray:
        """Run the second source-bound AST worker over axis-0 slices."""

        self._require_open()
        if not callable(worker):
            raise CPoolError("C_ABSORPTION_WORKER")
        try:
            if np.dtype(ctype) != np.dtype("complex128"):
                raise CPoolError("C_COMPLEX_DTYPE")
        except (TypeError, ValueError) as exc:
            raise CPoolError("C_COMPLEX_DTYPE") from exc
        if alpha is not None:
            self._require_cpu_array(alpha, dtype="float64", shape=tuple(E.shape), name="ALPHA")
            if np.shares_memory(alpha, E):
                raise CPoolError("C_ALPHA_E_ALIAS")
        self._run("C_ABSORPTION", E, worker)
        return E

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
                "schema": "filament.cpu_s4_6e.c_pool.v1",
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


class NullCPool:
    """Reference-role companion; it never submits candidate work."""

    count = 0
    closed = False

    def take_records(self) -> list[dict[str, Any]]:
        return []

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "filament.cpu_s4_6e.c_pool.v1",
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
    "CPool",
    "NullCPool",
    "CPoolError",
    "SafeWorkers",
    "MAX_RECORDS",
    "POINTWISE_LABELS",
]
