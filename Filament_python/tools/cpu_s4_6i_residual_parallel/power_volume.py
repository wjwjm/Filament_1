"""Bounded P24 implementation of the frozen deposition power cleanup.

This module owns only the selected ``_power_volume`` replacement.  It keeps
the scientific expression split into the same two NumPy operations as the
source function and delegates worker lifecycle to the existing C pool
implementation.  Only scalar worker activity survives an operation.
"""

from __future__ import annotations

import copy
import threading
from typing import Any, Callable

import numpy as np

from tools.cpu_s4_6e_array_propagation.c_pool import CPool, CPoolError, MAX_RECORDS


POWER_VOLUME_SCHEMA = "filament.cpu.s4_6i.residual_parallel.power_volume.v1"
POWER_VOLUME_SEMANTIC_ID = "POWER_VOLUME_CLEANUP_P24"
POWER_VOLUME_WORKERS = 24
POWER_VOLUME_MAX_RECORDS = MAX_RECORDS
NAN_TO_NUM_LABEL = "P24_POWER_NAN_TO_NUM"
MAXIMUM_LABEL = "P24_POWER_MAXIMUM"
SOURCE_SITE = "deposition.py:23:_power_volume"


class PowerVolumeError(RuntimeError):
    """The selected power-volume candidate failed closed."""


class PowerVolumeDomainError(PowerVolumeError):
    """The selected candidate received an unsupported CPU source array."""


def validate_power_volume_input(power: Any) -> np.ndarray:
    """Validate the exact CPU input domain without changing the input."""

    if type(power) is not np.ndarray:
        raise PowerVolumeDomainError("S46I_POWER_VOLUME_REQUIRES_NUMPY_NDARRAY")
    if power.ndim != 3:
        raise PowerVolumeDomainError("S46I_POWER_VOLUME_REQUIRES_3D")
    if power.dtype != np.dtype("float64") or not power.dtype.isnative:
        raise PowerVolumeDomainError("S46I_POWER_VOLUME_REQUIRES_NATIVE_FLOAT64")
    if not power.flags.c_contiguous:
        raise PowerVolumeDomainError("S46I_POWER_VOLUME_REQUIRES_C_CONTIGUOUS")
    if any(int(size) <= 0 for size in power.shape):
        raise PowerVolumeDomainError("S46I_POWER_VOLUME_REQUIRES_NONEMPTY")
    return power


class PowerVolumePool(CPool):
    """One bounded pool: P24 formally, P1 only for mechanism/local engineering."""

    def __init__(self, count: int = POWER_VOLUME_WORKERS) -> None:
        if type(count) is not int or count not in (1, POWER_VOLUME_WORKERS):
            raise PowerVolumeError("S46I_POWER_VOLUME_REQUIRES_P1_OR_P24_POOL")
        super().__init__(count, max_records=POWER_VOLUME_MAX_RECORDS)
        self._labels = {NAN_TO_NUM_LABEL: 0, MAXIMUM_LABEL: 0}
        self._total_records = 0
        self._dropped_records = 0
        self._invalid_records_dropped = 0
        self._scalar_sums = {
            "items": 0,
            "wall_ns": 0,
            "thread_cpu_ns": 0,
        }

    @staticmethod
    def _scalar_record(row: Any) -> bool:
        if not isinstance(row, dict):
            return False
        if not isinstance(row.get("label"), str):
            return False
        workers = row.get("workers")
        if not isinstance(workers, list):
            return False
        for worker in workers:
            if not isinstance(worker, dict):
                return False
            if not all(isinstance(worker.get(name), int) for name in ("thread_id", "items", "wall_ns", "thread_cpu_ns")):
                return False
        return True

    def _record(self, label: str) -> None:
        # The mechanism drains each completed call. Formal windows contain
        # only 64 rows. Capacity/invalid-record failures never discard evidence.
        self._labels[label] = self._labels.get(label, 0) + 1
        rows = self._workers.take_records()
        for row in rows:
            self._total_records += 1
            if not self._scalar_record(row):
                self._invalid_records_dropped += 1
                raise CPoolError("S46I_POWER_INVALID_ACTIVITY_RECORD")
            workers = row["workers"]
            self._scalar_sums["items"] += sum(int(item["items"]) for item in workers)
            self._scalar_sums["wall_ns"] += sum(int(item["wall_ns"]) for item in workers)
            self._scalar_sums["thread_cpu_ns"] += sum(int(item["thread_cpu_ns"]) for item in workers)
            if len(self._records) < self.max_records:
                self._records.append(copy.deepcopy(row))
            else:
                self._dropped_records += 1
                raise CPoolError("S46I_POWER_RECORD_CAPACITY_EXHAUSTED")

    def run_round(
        self,
        label: str,
        source: np.ndarray,
        operation: Callable[[int, int], None],
    ) -> None:
        """Run one scalar-record round through the inherited Workers."""

        with self._lock:
            self._require_open()
            if label not in (NAN_TO_NUM_LABEL, MAXIMUM_LABEL):
                raise CPoolError("S46I_POWER_OPERATION_LABEL")
            if type(source) is not np.ndarray or source.ndim != 3 or not source.flags.c_contiguous:
                raise CPoolError("S46I_POWER_VOLUME_OPERATION_LAYOUT")
            try:
                self._workers.run(label, int(source.shape[0]), operation)
                self._record(label)
            except BaseException:
                try:
                    self.close()
                except BaseException:
                    pass
                raise

    def annotate_last_operation(self, *, context: str, interval_index: int | None) -> None:
        """Attach only scalar context metadata to the two latest round rows."""

        with self._lock:
            rows = self._records[-2:]
            for row in rows:
                row["context"] = str(context)
                row["interval_index"] = None if interval_index is None else int(interval_index)
                row["source"] = SOURCE_SITE

    def snapshot(self) -> dict[str, Any]:
        result = super().snapshot()
        result.update(
            {
                "schema": POWER_VOLUME_SCHEMA + ".pool.v1",
                "semantic_id": POWER_VOLUME_SEMANTIC_ID,
                "total_records": int(self._total_records),
                "dropped_records": int(self._dropped_records),
                "invalid_records_dropped": int(self._invalid_records_dropped),
                "scalar_sums": copy.deepcopy(self._scalar_sums),
                "records_capacity": int(self.max_records),
                "array_values_retained": False,
            }
        )
        return result


def make_power_volume_pool(count: int = POWER_VOLUME_WORKERS) -> PowerVolumePool:
    """Construct one independent fixed P24 pool."""

    return PowerVolumePool(count)


def _run_round(
    pool: PowerVolumePool,
    source: np.ndarray,
    *,
    label: str,
    operation: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Run one complete axis-0 round and wait for every slice."""

    output = np.empty_like(source)

    def worker(lo: int, hi: int) -> None:
        # One vectorized operation per assigned slice.  No source write is
        # possible, including when the caller supplied a readonly ndarray.
        value = operation(source[lo:hi])
        output[lo:hi] = value

    pool.run_round(label, source, worker)
    return output


def power_volume_cleanup_p24(
    power: Any,
    *,
    pool: PowerVolumePool,
    backend: Any = np,
    context: str = "unknown",
    interval_index: int | None = None,
    on_failure: Callable[[], Any] | None = None,
) -> np.ndarray:
    """Evaluate the frozen cleanup in two complete P24 rounds.

    Reference input capture belongs to the source-bound reference hook, not
    this candidate kernel. No scientific arrays survive in activity records.
    """

    try:
        validate_power_volume_input(power)
        try:
            source = backend.asarray(power)
        except BaseException as exc:
            raise PowerVolumeDomainError("S46I_POWER_VOLUME_ASARRAY_FAILED") from exc
        validate_power_volume_input(source)
        if not isinstance(pool, PowerVolumePool):
            raise PowerVolumeError("S46I_POWER_VOLUME_POOL_TYPE")
        pool._require_open()
    except BaseException:
        if callable(on_failure):
            try:
                on_failure()
            except BaseException:
                pass
        raise
    try:
        # Keep this order explicit.  The first round is fully joined before
        # the independent final output is allocated and the second round
        # begins; the intermediate remains alive until round two completes.
        cleaned = _run_round(
            pool,
            source,
            label=NAN_TO_NUM_LABEL,
            operation=lambda value: backend.nan_to_num(
                value,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
        )
        result = _run_round(
            pool,
            cleaned,
            label=MAXIMUM_LABEL,
            operation=lambda value: backend.maximum(value, 0.0),
        )
        pool.annotate_last_operation(
            context=str(context),
            interval_index=None if interval_index is None else int(interval_index),
        )
        del cleaned
        return result
    except BaseException:
        if callable(on_failure):
            try:
                on_failure()
            except BaseException:
                pass
        raise


__all__ = [
    "MAXIMUM_LABEL",
    "NAN_TO_NUM_LABEL",
    "POWER_VOLUME_MAX_RECORDS",
    "POWER_VOLUME_SCHEMA",
    "POWER_VOLUME_SEMANTIC_ID",
    "POWER_VOLUME_WORKERS",
    "PowerVolumeDomainError",
    "PowerVolumeError",
    "PowerVolumePool",
    "SOURCE_SITE",
    "make_power_volume_pool",
    "power_volume_cleanup_p24",
    "validate_power_volume_input",
]
