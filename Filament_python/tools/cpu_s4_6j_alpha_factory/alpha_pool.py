"""Independent five-round P24 helpers for the ion-alpha cleanup chain.

This module intentionally stops at the numerical kernel boundary.  The caller
keeps the four original assignments and array-lifetime boundaries.  The pool
stores only bounded scalar worker records; scientific arrays are never kept by
the pool after a round completes.
"""

from __future__ import annotations

import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Callable

import numpy as np

from tools.cpu_s4_6c_array_coverage.chains import Workers


ALPHA_SCHEMA = "filament.cpu.s4_6j.alpha_pool.v1"
ALPHA_SEMANTIC_ID = "ION_ALPHA_CLEANUP_P24"
ALPHA_WORKERS = 24
ALPHA_MAX_RECORDS = 128
ALPHA_LABELS = (
    "ALPHA_CLEAN_SOURCE",
    "ALPHA_DENOM",
    "ALPHA_RATIO",
    "ALPHA_CLEAN_RESULT",
    "ALPHA_CLIP",
)
_ALPHA_LABEL_SET = frozenset(ALPHA_LABELS)
_ALPHA_WORKER_COUNTS = (1, 3, ALPHA_WORKERS)


class AlphaPoolError(RuntimeError):
    """The alpha pool rejected an operation or failed closed."""


class AlphaPoolDomainError(AlphaPoolError):
    """An input was outside the fixed native float64 CPU domain."""


class AlphaSafeWorkers(Workers):
    """Workers with the P1/P3/P24 engineering count contract."""

    def __init__(self, count: int) -> None:
        if type(count) is not int or count not in _ALPHA_WORKER_COUNTS:
            raise AlphaPoolError("S46J_ALPHA_WORKER_CONFIGURATION")
        self.count = int(count)
        self.closed = False
        self.records = []
        self._lock = threading.Lock()
        started = time.perf_counter_ns()
        self.executor = None
        barrier: threading.Barrier | None = None
        try:
            self.executor = ThreadPoolExecutor(
                max_workers=count,
                thread_name_prefix="s46j-alpha",
            )
            barrier = threading.Barrier(count)

            def prime() -> int:
                barrier.wait(timeout=60)
                return threading.get_native_id()

            jobs = [self.executor.submit(prime) for _ in range(count)]
            self.initial_thread_ids = [job.result() for job in jobs]
            if len(set(self.initial_thread_ids)) != count:
                raise AlphaPoolError("S46J_ALPHA_POOL_NOT_FULLY_STARTED")
            self.initialization_ns = time.perf_counter_ns() - started
        except BaseException as primary:
            if barrier is not None:
                try:
                    barrier.abort()
                except BaseException:
                    pass
            executor = self.executor
            if executor is not None:
                try:
                    executor.shutdown(wait=True, cancel_futures=True)
                except BaseException as cleanup:
                    try:
                        primary.add_note(
                            "S46J_ALPHA_CONSTRUCTOR_CLEANUP_FAILED:"
                            + type(cleanup).__name__
                        )
                    except BaseException:
                        pass
            raise

    def run(self, label: str, length: int, operation: Callable[[int, int], Any]) -> None:
        """Run one round with submit-failure barrier abort and full draining."""

        if self.closed:
            raise AlphaPoolError("S46J_ALPHA_RETIRED_WORKER_POOL")
        if not self._lock.acquire(blocking=False):
            raise AlphaPoolError("S46J_ALPHA_CONCURRENT_OR_REENTRANT_CHAIN")
        barrier = threading.Barrier(self.count)
        errors = np.geterr()
        jobs = []

        def worker(index: int) -> dict[str, int]:
            lo, hi = length * index // self.count, length * (index + 1) // self.count
            barrier.wait(timeout=60)
            wall_started = time.perf_counter_ns()
            cpu_started = time.thread_time_ns()
            with np.errstate(**errors):
                operation(lo, hi)
            return {
                "thread_id": threading.get_native_id(),
                "items": hi - lo,
                "wall_ns": time.perf_counter_ns() - wall_started,
                "thread_cpu_ns": time.thread_time_ns() - cpu_started,
            }

        try:
            try:
                for index in range(self.count):
                    jobs.append(self.executor.submit(worker, index))
            except BaseException as primary:
                try:
                    barrier.abort()
                except BaseException:
                    pass
                for job in jobs:
                    job.cancel()
                wait(jobs)
                raise primary

            # All submitted tasks are joined before any result is observed;
            # this preserves the original task exception and prevents a worker
            # from outliving its owning round.
            wait(jobs)
            rows = [job.result() for job in jobs]
            self.records.append(
                {
                    "label": label,
                    "requested_workers": self.count,
                    "actual_workers": len({row["thread_id"] for row in rows if row["items"]}),
                    "workers": rows,
                }
            )
        finally:
            self._lock.release()


def _require_float64_3d(value: Any, name: str, *, writable: bool = False) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise AlphaPoolDomainError(f"S46J_ALPHA_{name}_REQUIRES_NDARRAY")
    if value.ndim != 3:
        raise AlphaPoolDomainError(f"S46J_ALPHA_{name}_REQUIRES_3D")
    if value.dtype != np.dtype("float64") or not value.dtype.isnative:
        raise AlphaPoolDomainError(f"S46J_ALPHA_{name}_REQUIRES_NATIVE_FLOAT64")
    if not value.flags.c_contiguous:
        raise AlphaPoolDomainError(f"S46J_ALPHA_{name}_REQUIRES_C_CONTIGUOUS")
    if any(int(size) <= 0 for size in value.shape):
        raise AlphaPoolDomainError(f"S46J_ALPHA_{name}_REQUIRES_NONEMPTY")
    if writable and not value.flags.writeable:
        raise AlphaPoolDomainError(f"S46J_ALPHA_{name}_REQUIRES_WRITABLE")
    return value


def _require_backend(backend: Any) -> Any:
    if backend is not np:
        raise AlphaPoolError("S46J_ALPHA_NUMPY_BACKEND_REQUIRED")
    for name in ("add", "divide", "nan_to_num", "maximum"):
        if not callable(getattr(backend, name, None)):
            raise AlphaPoolError(f"S46J_ALPHA_BACKEND_{name.upper()}_MISSING")
    return backend


class AlphaPool:
    """One persistent P24 pool for exactly five sequential alpha rounds."""

    def __init__(
        self,
        count: int = ALPHA_WORKERS,
        *,
        max_records: int = ALPHA_MAX_RECORDS,
        on_failure: Callable[[], Any] | None = None,
    ) -> None:
        if type(count) is not int or count not in _ALPHA_WORKER_COUNTS:
            raise AlphaPoolError("S46J_ALPHA_WORKER_CONFIGURATION")
        if type(max_records) is not int or max_records <= 0 or max_records > ALPHA_MAX_RECORDS:
            raise AlphaPoolError("S46J_ALPHA_RECORD_CAPACITY")
        if on_failure is not None and not callable(on_failure):
            raise AlphaPoolError("S46J_ALPHA_FAILURE_CALLBACK")
        self.count = int(count)
        self.max_records = int(max_records)
        self.closed = False
        self._lock = threading.RLock()
        self._records: list[dict[str, Any]] = []
        self._labels = {label: 0 for label in ALPHA_LABELS}
        self._attempted_labels = {label: 0 for label in ALPHA_LABELS}
        self._completed_labels = {label: 0 for label in ALPHA_LABELS}
        self._record_failures = {label: 0 for label in ALPHA_LABELS}
        self._close_error: str | None = None
        self._close_exception: BaseException | None = None
        self._close_retryable = False
        self._created_ns = time.perf_counter_ns()
        self._close_ns: int | None = None
        self._lifetime_ns: int | None = None
        self._failure_cleanup = on_failure
        self._failure_notified = False
        self._total_records = 0
        self._dropped_records = 0
        self._invalid_records = 0
        self._scalar_sums = {
            "requested_workers": 0,
            "actual_workers": 0,
            "worker_count": 0,
            "items": 0,
            "wall_ns": 0,
            "thread_cpu_ns": 0,
        }
        self._workers = AlphaSafeWorkers(self.count)
        self.initialization_ns = int(self._workers.initialization_ns)

    def _require_open(self) -> None:
        if self.closed:
            raise AlphaPoolError("S46J_ALPHA_RETIRED_WORKER_POOL")

    def set_failure_cleanup(self, callback: Callable[[], Any] | None) -> None:
        if callback is not None and not callable(callback):
            raise AlphaPoolError("S46J_ALPHA_FAILURE_CALLBACK")
        with self._lock:
            self._failure_cleanup = callback
            self._failure_notified = False

    @staticmethod
    def _scalar_tree(value: Any) -> bool:
        """Return whether a record contains only bounded scalar containers."""

        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return True
        if isinstance(value, np.generic):
            return True
        if isinstance(value, dict):
            return all(
                AlphaPool._scalar_tree(key) and AlphaPool._scalar_tree(item)
                for key, item in value.items()
            )
        if isinstance(value, (tuple, list)):
            return all(AlphaPool._scalar_tree(item) for item in value)
        return False

    @staticmethod
    def _scalar_record(row: Any) -> bool:
        if not isinstance(row, dict) or not isinstance(row.get("label"), str):
            return False
        workers = row.get("workers")
        if not isinstance(workers, list):
            return False
        return AlphaPool._scalar_tree(row) and all(
            isinstance(worker, dict)
            and all(
                isinstance(worker.get(name), int)
                for name in ("thread_id", "items", "wall_ns", "thread_cpu_ns")
            )
            for worker in workers
        )

    @staticmethod
    def _add_scalar(sums: dict[str, int], name: str, value: Any) -> None:
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            sums[name] += int(value)

    @staticmethod
    def _scalar_projection(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, np.generic):
            return AlphaPool._scalar_projection(value.item())
        if isinstance(value, np.ndarray):
            return "<non_scalar_value_omitted>"
        if isinstance(value, dict):
            return {str(key): AlphaPool._scalar_projection(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [AlphaPool._scalar_projection(item) for item in value]
        return "<non_scalar_value_omitted>"

    def _record(self, label: str) -> None:
        rows = self._workers.take_records()
        failure: str | None = None
        for row in rows:
            self._total_records += 1
            if isinstance(row, dict) and isinstance(row.get("workers"), list):
                workers = row["workers"]
                self._add_scalar(self._scalar_sums, "requested_workers", row.get("requested_workers"))
                self._add_scalar(self._scalar_sums, "actual_workers", row.get("actual_workers"))
                self._add_scalar(self._scalar_sums, "worker_count", len(workers))
                for worker in workers:
                    if isinstance(worker, dict):
                        for name in ("items", "wall_ns", "thread_cpu_ns"):
                            value = worker.get(name)
                            self._add_scalar(self._scalar_sums, name, value)
            invalid = not self._scalar_record(row)
            if invalid:
                self._invalid_records += 1
            if len(self._records) >= self.max_records:
                self._dropped_records += 1
                failure = failure or "S46J_ALPHA_RECORD_CAPACITY_EXHAUSTED"
                continue
            if invalid:
                projected = self._scalar_projection(row)
                if isinstance(projected, dict):
                    projected["invalid_record"] = True
                else:
                    projected = {"invalid_record": True, "record": projected}
                self._records.append(copy.deepcopy(projected))
                failure = failure or "S46J_ALPHA_INVALID_ACTIVITY_RECORD"
            else:
                self._records.append(copy.deepcopy(self._scalar_projection(row)))
        if failure is not None:
            self._record_failures[label] += 1
            raise AlphaPoolError(failure)
        self._labels[label] += 1

    def _notify_failure(self) -> None:
        try:
            self.close()
        except BaseException:
            pass
        callback = None
        with self._lock:
            if not self._failure_notified:
                self._failure_notified = True
                callback = self._failure_cleanup
        if callback is not None:
            try:
                callback()
            except BaseException:
                pass

    def run_round(
        self,
        label: str,
        source: np.ndarray,
        operation: Callable[..., Any],
        *,
        inplace: bool = False,
        extra_sources: tuple[np.ndarray, ...] = (),
    ) -> np.ndarray:
        """Run one complete axis-0 round and join all submitted tasks."""

        try:
            with self._lock:
                self._require_open()
                if label not in _ALPHA_LABEL_SET:
                    raise AlphaPoolError("S46J_ALPHA_OPERATION_LABEL")
                _require_float64_3d(source, "SOURCE", writable=inplace)
                for index, extra in enumerate(extra_sources):
                    _require_float64_3d(extra, f"EXTRA_{index}")
                    if extra.shape != source.shape:
                        raise AlphaPoolDomainError("S46J_ALPHA_EXTRA_SOURCE_SHAPE")
                if not callable(operation):
                    raise AlphaPoolError("S46J_ALPHA_OPERATION_CALLABLE")
                output = source if inplace else np.empty_like(source)

                def worker(lo: int, hi: int) -> None:
                    value = operation(
                        source[lo:hi],
                        *(extra[lo:hi] for extra in extra_sources),
                    )
                    if not inplace:
                        if (
                            type(value) is not np.ndarray
                            or value.shape != source[lo:hi].shape
                            or value.dtype != source.dtype
                            or not value.dtype.isnative
                            or not value.flags.c_contiguous
                        ):
                            raise AlphaPoolError("S46J_ALPHA_OPERATION_OUTPUT")
                        output[lo:hi] = value

                self._attempted_labels[label] += 1
                self._workers.run(label, int(source.shape[0]), worker)
                self._completed_labels[label] += 1
                self._record(label)
                return output
        except BaseException:
            self._notify_failure()
            raise

    def take_records(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._records
            self._records = []
            return copy.deepcopy(rows)

    def annotate_last_operation(self, *, context: str, interval_index: int | None) -> None:
        with self._lock:
            for row in self._records[-1:]:
                row["context"] = str(context)
                row["interval_index"] = None if interval_index is None else int(interval_index)
                row["source"] = "propagate.py:830-833:alpha_ion"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            executor = getattr(self._workers, "executor", None)
            live_threads = []
            for thread in getattr(executor, "_threads", ()):
                if thread.is_alive() and getattr(thread, "native_id", None) is not None:
                    live_threads.append(int(thread.native_id))
            return {
                "schema": ALPHA_SCHEMA + ".pool.v1",
                "semantic_id": ALPHA_SEMANTIC_ID,
                "workers": self.count,
                "closed": bool(self.closed),
                "initialization_ns": int(self.initialization_ns),
                "close_ns": None if self._close_ns is None else int(self._close_ns),
                "lifetime_ns": None if self._lifetime_ns is None else int(self._lifetime_ns),
                "close_error": self._close_error,
                "close_failed": bool(self._close_retryable),
                "close_retryable": bool(self._close_retryable),
                "pending_scalar_records": len(self._records),
                "operation_counts": dict(self._labels),
                "attempted_operation_counts": dict(self._attempted_labels),
                "completed_operation_counts": dict(self._completed_labels),
                "record_failure_counts": dict(self._record_failures),
                "total_rounds": int(sum(self._labels.values())),
                "total_records": int(self._total_records),
                "dropped_records": int(self._dropped_records),
                "invalid_records": int(self._invalid_records),
                # Kept as an explicit compatibility alias for the first
                # evidence schema; invalid records are retained as projected
                # scalar evidence before failure retirement.
                "invalid_records_dropped": int(self._invalid_records),
                "scalar_sums": copy.deepcopy(self._scalar_sums),
                "records_scalar_only": True,
                "initial_thread_ids": list(getattr(self._workers, "initial_thread_ids", ())),
                "active_thread_ids": sorted(live_threads),
                "records_capacity": self.max_records,
                "array_values_retained": False,
            }

    def close(self) -> dict[str, Any]:
        """Join and retire the pool; repeated close is safe."""

        with self._lock:
            if self.closed and not self._close_retryable:
                return self.snapshot()
            started = time.perf_counter_ns()
            try:
                self._workers.close()
            except BaseException as exc:
                if self._close_exception is None:
                    self._close_exception = exc
                    self._close_error = type(exc).__name__
                else:
                    try:
                        self._close_exception.add_note(
                            "S46J_ALPHA_CLOSE_RETRY_FAILED:" + type(exc).__name__
                        )
                    except BaseException:
                        pass
                self.closed = True
                self._close_retryable = True
                finished = time.perf_counter_ns()
                self._close_ns = finished - started
                self._lifetime_ns = finished - self._created_ns
                if self._close_exception is not exc:
                    raise self._close_exception
                raise
            self.closed = True
            self._close_retryable = False
            finished = time.perf_counter_ns()
            self._close_ns = finished - started
            self._lifetime_ns = finished - self._created_ns
            return self.snapshot()


def _require_pool(pool: AlphaPool) -> AlphaPool:
    if not isinstance(pool, AlphaPool):
        raise AlphaPoolError("S46J_ALPHA_POOL_TYPE")
    pool._require_open()
    return pool


def clean_source(source: Any, *, pool: AlphaPool, backend: Any = np) -> np.ndarray:
    """First round: independent ``nan_to_num`` source cleanup."""

    _require_backend(backend)
    _require_pool(pool)
    source = _require_float64_3d(source, "SOURCE")
    return pool.run_round(
        "ALPHA_CLEAN_SOURCE",
        source,
        lambda value: backend.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0, copy=True),
    )


def ratio(
    cleaned_source: Any,
    intensity: Any,
    I_floor: Any,
    *,
    pool: AlphaPool,
    backend: Any = np,
) -> np.ndarray:
    """Second/fourth rounds: denominator add, then independent division."""

    _require_backend(backend)
    _require_pool(pool)
    cleaned_source = _require_float64_3d(cleaned_source, "CLEAN_SOURCE")
    intensity = _require_float64_3d(intensity, "INTENSITY")
    if cleaned_source.shape != intensity.shape:
        raise AlphaPoolDomainError("S46J_ALPHA_RATIO_SHAPE")
    if type(I_floor) is not float:
        raise AlphaPoolDomainError("S46J_ALPHA_IFLOOR_PYTHON_FLOAT")
    denominator: np.ndarray | None = None
    try:
        denominator = pool.run_round(
            "ALPHA_DENOM",
            intensity,
            lambda value: backend.add(value, I_floor),
        )
        result = pool.run_round(
            "ALPHA_RATIO",
            cleaned_source,
            lambda value, denominator_slice: backend.divide(value, denominator_slice),
            extra_sources=(denominator,),
        )
        return result
    except BaseException:
        raise
    finally:
        # The explicit deletion is part of the lifetime contract.  No worker
        # record or pool field retains the denominator array.
        denominator = None


def clean_alpha(alpha: Any, *, pool: AlphaPool, backend: Any = np) -> np.ndarray:
    """Third round: independent cleanup of the quotient."""

    _require_backend(backend)
    _require_pool(pool)
    alpha = _require_float64_3d(alpha, "ALPHA")
    return pool.run_round(
        "ALPHA_CLEAN_RESULT",
        alpha,
        lambda value: backend.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0, copy=True),
    )


def clip_alpha(alpha: Any, *, pool: AlphaPool, backend: Any = np) -> np.ndarray:
    """Fifth round: in-place nonnegative clip, returning the same object."""

    _require_backend(backend)
    _require_pool(pool)
    alpha = _require_float64_3d(alpha, "ALPHA", writable=True)
    return pool.run_round(
        "ALPHA_CLIP",
        alpha,
        lambda value: backend.maximum(value, 0.0, out=value),
        inplace=True,
    )


def make_alpha_pool(
    count: int = ALPHA_WORKERS,
    *,
    on_failure: Callable[[], Any] | None = None,
) -> AlphaPool:
    return AlphaPool(count, on_failure=on_failure)


__all__ = [
    "ALPHA_LABELS",
    "ALPHA_MAX_RECORDS",
    "ALPHA_SCHEMA",
    "ALPHA_SEMANTIC_ID",
    "ALPHA_WORKERS",
    "AlphaPool",
    "AlphaPoolDomainError",
    "AlphaPoolError",
    "clean_alpha",
    "clean_source",
    "clip_alpha",
    "make_alpha_pool",
    "ratio",
]
