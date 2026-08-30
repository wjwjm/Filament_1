"""HR-3C-B disk-backed current/next storage and streaming diffusion pass."""

from __future__ import annotations

import math
import shutil
import time
from pathlib import Path

import numpy as np

from .device import to_cpu, xp
from . import slow_diffusion


_HEADER_RESERVE_BYTES = 4096


def _validate_state_layout(*, n_intervals: int, shape, dtype) -> tuple[int, tuple[int, int], np.dtype]:
    count = int(n_intervals)
    slice_shape = tuple(int(value) for value in shape)
    real_dtype = np.dtype(dtype)
    if count <= 0:
        raise ValueError("HR-3C ping-pong state requires at least one interval")
    if len(slice_shape) != 2 or min(slice_shape) <= 0:
        raise ValueError("HR-3C ping-pong state slice shape must be positive [Ny, Nx]")
    if real_dtype.kind != "f":
        raise ValueError("HR-3C ping-pong state dtype must be real floating point")
    return count, slice_shape, real_dtype


def estimate_state_bytes(*, n_intervals: int, shape, dtype) -> int:
    """Return raw payload bytes for one ``[K, Ny, Nx]`` state file."""
    count, slice_shape, real_dtype = _validate_state_layout(
        n_intervals=n_intervals, shape=shape, dtype=dtype,
    )
    return int(count * slice_shape[0] * slice_shape[1] * real_dtype.itemsize)


def estimate_pingpong_bytes(*, n_intervals: int, shape, dtype) -> int:
    """Return raw payload bytes for separate current and next state files."""
    return 2 * estimate_state_bytes(n_intervals=n_intervals, shape=shape, dtype=dtype)


class PingPongSlowStateStore:
    """Two disk-backed interval-centered state files without role promotion.

    ``current`` is the authoritative input for one HR-3C diffusion pass.
    ``next`` is scratch output only; HR-3C-B deliberately has no role-swap,
    generation, restart, or checkpoint semantics.
    """

    def __init__(self, *, output_path: str, n_intervals: int, shape, dtype, check_disk_space: bool = True):
        self.n_intervals, self.shape, self.dtype = _validate_state_layout(
            n_intervals=n_intervals, shape=shape, dtype=dtype,
        )
        root = Path(output_path).with_suffix("")
        self.current_path = root.with_name(root.name + ".hr3c_delta_n_th_current.npy")
        self.next_path = root.with_name(root.name + ".hr3c_delta_n_th_next.npy")
        self.state_shape = (self.n_intervals, *self.shape)
        self.next_complete = False
        self.next_valid = False
        self._current = self._next = None

        if check_disk_space:
            required = estimate_pingpong_bytes(
                n_intervals=self.n_intervals, shape=self.shape, dtype=self.dtype,
            ) + 2 * _HEADER_RESERVE_BYTES
            free = shutil.disk_usage(self.current_path.parent).free
            if free < required:
                raise OSError(
                    f"HR-3C ping-pong preflight requires {required} bytes but only {free} are free"
                )

        self._current = np.lib.format.open_memmap(
            self.current_path, mode="w+", dtype=self.dtype, shape=self.state_shape,
        )
        self._next = np.lib.format.open_memmap(
            self.next_path, mode="w+", dtype=self.dtype, shape=self.state_shape,
        )
        self._current.fill(0.0)
        self._next.fill(0.0)
        self._current.flush()
        self._next.flush()

    def _require_open(self) -> None:
        if self._current is None or self._next is None:
            raise RuntimeError("HR-3C ping-pong state store is closed")

    def _bounds(self, start: int, stop: int) -> tuple[int, int]:
        first, last = int(start), int(stop)
        if first < 0 or last <= first or last > self.n_intervals:
            raise IndexError("HR-3C batch range is outside the persistent state")
        return first, last

    def _index(self, interval_index: int) -> int:
        index = int(interval_index)
        if index < 0 or index >= self.n_intervals:
            raise IndexError("HR-3C interval index is outside the persistent state")
        return index

    def read_current_interval(self, interval_index: int):
        self._require_open()
        return self._current[self._index(interval_index)]

    def read_current_batch(self, start: int, stop: int):
        self._require_open()
        first, last = self._bounds(start, stop)
        return self._current[first:last]

    def update_current_interval(self, interval_index: int, increment):
        """Prepare/update current for a future HR-3C-C caller; not used by B pass."""
        self._require_open()
        index = self._index(interval_index)
        value = np.asarray(to_cpu(increment), dtype=self.dtype)
        if value.shape != self.shape or not np.all(np.isfinite(value)):
            raise ValueError("HR-3C current increment has invalid shape or values")
        np.add(self._current[index], value, out=self._current[index])
        if not np.all(np.isfinite(self._current[index])):
            raise ValueError("HR-3C current update produced non-finite values")
        return self._current[index]

    def begin_next_pass(self) -> None:
        self._require_open()
        self.next_complete = False
        self.next_valid = False

    def write_next_batch(self, start: int, values) -> None:
        self._require_open()
        batch = np.asarray(to_cpu(values), dtype=self.dtype)
        if batch.ndim != 3 or batch.shape[1:] != self.shape or batch.shape[0] <= 0:
            raise ValueError("HR-3C next batch must have shape [B, Ny, Nx]")
        first, last = self._bounds(start, int(start) + int(batch.shape[0]))
        if not np.all(np.isfinite(batch)):
            raise ValueError("HR-3C next batch must be finite")
        self._next[first:last] = batch

    def flush_next(self) -> None:
        self._require_open()
        self._next.flush()

    def mark_next_complete(self) -> None:
        self._require_open()
        self.next_complete = True
        self.next_valid = True

    def mark_next_invalid(self) -> None:
        self.next_complete = False
        self.next_valid = False

    def close(self) -> None:
        if self._current is not None:
            self._current.flush()
        if self._next is not None:
            self._next.flush()
        self._current = None
        self._next = None

    def metadata(self) -> dict[str, object]:
        return {
            "hr3c_pingpong_schema": "khz_filament.hr3c.pingpong.v1",
            "hr3c_current_filename": self.current_path.name,
            "hr3c_next_filename": self.next_path.name,
            "hr3c_state_shape": self.state_shape,
            "hr3c_state_dtype": self.dtype.name,
            "hr3c_state_interval_centered": True,
            "hr3c_current_authoritative": True,
            "hr3c_next_complete": bool(self.next_complete),
            "hr3c_next_valid_scratch": bool(self.next_valid),
            "hr3c_role_swap_deferred_to": "HR-3C-C",
        }


def diffuse_current_to_next(
    store: PingPongSlowStateStore,
    *,
    kperp2,
    D_th: float,
    f_rep: float,
    edge_threshold: float | None = slow_diffusion.DEFAULT_EDGE_CONTAMINATION_THRESHOLD,
    batch_intervals: int,
) -> dict[str, object]:
    """Stream current to next in independent z batches without role promotion."""
    batch_size = int(batch_intervals)
    if batch_size <= 0:
        raise ValueError("batch_intervals must be a positive integer")
    dt_interpulse = slow_diffusion.validate_hr3c_parameters(D_th=D_th, f_rep=f_rep)
    kernel = slow_diffusion.build_diffusion_kernel(kperp2, D_th=D_th, f_rep=f_rep)
    if tuple(kernel.shape) != tuple(store.shape):
        raise ValueError("kperp2 shape must match the ping-pong state slice shape")

    started = time.perf_counter()
    n_batches = 0
    max_R_edge = 0.0
    store.begin_next_pass()
    try:
        for start in range(0, store.n_intervals, batch_size):
            stop = min(start + batch_size, store.n_intervals)
            host_current = store.read_current_batch(start, stop)
            device_current = xp.asarray(host_current)
            device_next, batch_summary = slow_diffusion.diffuse_batch_2d(
                device_current,
                kperp2=kperp2,
                D_th=D_th,
                f_rep=f_rep,
                edge_threshold=edge_threshold,
                kernel=kernel,
                batch_offset=start,
                return_summary=True,
            )
            store.write_next_batch(start, to_cpu(device_next))
            max_R_edge = max(max_R_edge, float(batch_summary["max_R_edge"]))
            n_batches += 1
            del host_current, device_current, device_next
        store.flush_next()
        store.mark_next_complete()
    except Exception:
        store.mark_next_invalid()
        raise
    finally:
        del kernel

    bytes_per_state = estimate_state_bytes(
        n_intervals=store.n_intervals, shape=store.shape, dtype=store.dtype,
    )
    return {
        "n_intervals": store.n_intervals,
        "n_batches": n_batches,
        "batch_intervals": batch_size,
        "dtype": store.dtype.name,
        "state_shape": store.state_shape,
        "bytes_read": bytes_per_state,
        "bytes_written": bytes_per_state,
        "max_R_edge": max_R_edge,
        "first_failed_interval": -1,
        "complete": True,
        "next_authoritative": False,
        "dt_interpulse_s": dt_interpulse,
        "walltime_s": time.perf_counter() - started,
    }


__all__ = [
    "PingPongSlowStateStore",
    "diffuse_current_to_next",
    "estimate_pingpong_bytes",
    "estimate_state_bytes",
]
