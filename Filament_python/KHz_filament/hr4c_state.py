"""HR-4C three-field disk-backed transactional full-z state lifecycle."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .device import to_cpu
from .hr3c_state_machine import _atomic_json, _fsync_memmap, state_fingerprint
from .slow_state_pingpong import _HEADER_RESERVE_BYTES, _validate_state_layout, estimate_state_bytes
from .hr4 import advance_hr4_single_screen


HR4C_SCHEMA = "khz_filament.hr4c.three_field_state.v1"
HR4C_FIELDS = ("delta_n", "vx", "vy")


def estimate_hr4c_generation_bytes(*, n_intervals: int, shape, dtype) -> int:
    """Return the physical payload for one three-field generation."""
    return 3 * estimate_state_bytes(n_intervals=n_intervals, shape=shape, dtype=dtype)


def estimate_hr4c_slot_bytes(*, n_intervals: int, shape, dtype) -> int:
    """Return the physical payload for authoritative plus staging slots."""
    return 2 * estimate_hr4c_generation_bytes(n_intervals=n_intervals, shape=shape, dtype=dtype)


def estimate_hr4c_working_set_bytes(*, batch_intervals: int, shape, dtype) -> int:
    """Conservative host/operator accounting independent of full-z K."""
    batch = int(batch_intervals)
    _, slice_shape, real_dtype = _validate_state_layout(n_intervals=1, shape=shape, dtype=dtype)
    if batch <= 0:
        raise ValueError("HR-4C batch_intervals must be positive")
    pixels = int(slice_shape[0] * slice_shape[1])
    return int((6 * batch + 12) * pixels * real_dtype.itemsize)


def _validate_z_edges(z_edges, n_intervals: int) -> np.ndarray:
    edges = np.asarray(z_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size != int(n_intervals) + 1 or not np.all(np.isfinite(edges)):
        raise ValueError("HR-4C z_edges must be finite with length K + 1")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("HR-4C z_edges must be strictly increasing")
    return edges


class HR4CThreeFieldStore:
    """Six HR-3C-style memmap slots with one atomic three-field authority."""

    def __init__(
        self, *, output_path: str, n_intervals: int, shape, dtype, z_edges,
        dx: float, dy: float, check_disk_space: bool = True,
    ):
        self.n_intervals, self.shape, self.dtype = _validate_state_layout(
            n_intervals=n_intervals, shape=shape, dtype=dtype,
        )
        self.state_shape = (self.n_intervals, *self.shape)
        self.output_path = str(output_path)
        self.z_edges = _validate_z_edges(z_edges, self.n_intervals)
        self.dx, self.dy = float(dx), float(dy)
        self.grid_fingerprint = state_fingerprint(
            z_edges=self.z_edges, shape=self.shape, dtype=self.dtype, dx=self.dx, dy=self.dy,
        )
        root = Path(output_path).with_suffix("")
        self.manifest_path = root.with_name(root.name + ".hr4c_state_manifest.json")
        self.slot_paths = {
            role: {
                field: root.with_name(root.name + ".hr4c_" + field + "_" + role + ".npy")
                for field in HR4C_FIELDS
            }
            for role in ("current", "next")
        }
        if self.manifest_path.exists() or any(path.exists() for roles in self.slot_paths.values() for path in roles.values()):
            raise FileExistsError("HR-4C new state refuses to overwrite existing manifest or slots")
        if check_disk_space:
            required = estimate_hr4c_slot_bytes(
                n_intervals=self.n_intervals, shape=self.shape, dtype=self.dtype,
            ) + 6 * _HEADER_RESERVE_BYTES
            free = shutil.disk_usage(self.manifest_path.parent).free
            if free < required:
                raise OSError(
                    f"HR-4C three-field storage preflight requires {required} bytes but only {free} are free"
                )
        self._current = {
            field: np.lib.format.open_memmap(
                self.slot_paths["current"][field], mode="w+", dtype=self.dtype, shape=self.state_shape,
            )
            for field in HR4C_FIELDS
        }
        self._next = {
            field: np.lib.format.open_memmap(
                self.slot_paths["next"][field], mode="w+", dtype=self.dtype, shape=self.state_shape,
            )
            for field in HR4C_FIELDS
        }
        for array in (*self._current.values(), *self._next.values()):
            array.fill(0.0)
            array.flush()
        self._written = {field: set() for field in HR4C_FIELDS}
        self.manifest = {
            "schema_version": HR4C_SCHEMA,
            "transaction_status": "committed",
            "generation": 0,
            "staging_generation": None,
            "fields": list(HR4C_FIELDS),
            "state_shape": list(self.state_shape),
            "state_dtype": self.dtype.name,
            "n_intervals": self.n_intervals,
            "Ny": self.shape[0],
            "Nx": self.shape[1],
            "grid_fingerprint": self.grid_fingerprint,
            "z_ordering": "input_z_edges_increasing",
            "authoritative_filenames": {field: self.slot_paths["current"][field].name for field in HR4C_FIELDS},
            "scratch_filenames": {field: self.slot_paths["next"][field].name for field in HR4C_FIELDS},
            "initialization": {"mode": "zeros"},
            "last_evolution": None,
            "last_abort": None,
        }
        self._write_manifest()

    @classmethod
    def open_existing(
        cls, *, output_path: str, n_intervals: int, shape, dtype, z_edges, dx: float, dy: float,
    ):
        self = cls.__new__(cls)
        self.n_intervals, self.shape, self.dtype = _validate_state_layout(
            n_intervals=n_intervals, shape=shape, dtype=dtype,
        )
        self.state_shape = (self.n_intervals, *self.shape)
        self.output_path = str(output_path)
        self.z_edges = _validate_z_edges(z_edges, self.n_intervals)
        self.dx, self.dy = float(dx), float(dy)
        self.grid_fingerprint = state_fingerprint(
            z_edges=self.z_edges, shape=self.shape, dtype=self.dtype, dx=self.dx, dy=self.dy,
        )
        root = Path(output_path).with_suffix("")
        self.manifest_path = root.with_name(root.name + ".hr4c_state_manifest.json")
        if not self.manifest_path.is_file():
            raise FileNotFoundError("HR-4C reopen requires a state manifest")
        self.slot_paths = {
            role: {
                field: root.with_name(root.name + ".hr4c_" + field + "_" + role + ".npy")
                for field in HR4C_FIELDS
            }
            for role in ("current", "next")
        }
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest(self.manifest)
        self._open_slots_from_manifest()
        self._written = {field: set() for field in HR4C_FIELDS}
        if self.manifest["transaction_status"] == "staging":
            self.abort_staging(reason="reopen_discarded_incomplete_staging")
        return self

    @classmethod
    def initialize_from_legacy_delta_n(
        cls, *, output_path: str, legacy_delta_n_path: str, n_intervals: int, shape, dtype,
        z_edges, dx: float, dy: float, batch_intervals: int,
    ):
        """Create a new three-field generation without modifying legacy HR-3C data."""
        store = cls(
            output_path=output_path, n_intervals=n_intervals, shape=shape, dtype=dtype,
            z_edges=z_edges, dx=dx, dy=dy,
        )
        legacy = np.load(legacy_delta_n_path, mmap_mode="r")
        if legacy.shape != store.state_shape or legacy.dtype != store.dtype:
            store.close()
            raise ValueError("HR-4C legacy delta_n layout does not match requested three-field state")
        batch = int(batch_intervals)
        if batch <= 0:
            store.close()
            raise ValueError("HR-4C batch_intervals must be positive")
        store.begin_staging()
        try:
            for start in range(0, store.n_intervals, batch):
                stop = min(start + batch, store.n_intervals)
                values = {
                    "delta_n": np.asarray(legacy[start:stop], dtype=store.dtype),
                    "vx": np.zeros((stop - start, *store.shape), dtype=store.dtype),
                    "vy": np.zeros((stop - start, *store.shape), dtype=store.dtype),
                }
                store.write_staging_batch(start, values)
            store.commit_staging({
                "operation": "legacy_delta_n_initialization",
                "legacy_delta_n_filename": Path(legacy_delta_n_path).name,
                "batch_intervals": batch,
            })
        except Exception:
            store.abort_staging(reason="legacy_initialization_failure")
            raise
        return store

    def _expected_manifest(self) -> dict[str, object]:
        return {
            "schema_version": HR4C_SCHEMA,
            "fields": list(HR4C_FIELDS),
            "state_shape": list(self.state_shape),
            "state_dtype": self.dtype.name,
            "n_intervals": self.n_intervals,
            "Ny": self.shape[0],
            "Nx": self.shape[1],
            "grid_fingerprint": self.grid_fingerprint,
            "z_ordering": "input_z_edges_increasing",
        }

    def _validate_manifest(self, manifest: Mapping[str, Any]) -> None:
        for key, expected in self._expected_manifest().items():
            if manifest.get(key) != expected:
                raise ValueError(f"HR-4C manifest mismatch: {key}")
        if manifest.get("transaction_status") not in ("committed", "staging"):
            raise ValueError("HR-4C manifest transaction status is invalid")
        generation = manifest.get("generation")
        if not isinstance(generation, int) or generation < 0:
            raise ValueError("HR-4C manifest generation is invalid")
        staging = manifest.get("staging_generation")
        if manifest["transaction_status"] == "committed":
            if staging is not None:
                raise ValueError("HR-4C committed manifest may not retain a staging generation")
        elif staging != generation + 1:
            raise ValueError("HR-4C staging generation invariant failed")
        allowed = {field: {path.name for path in (self.slot_paths["current"][field], self.slot_paths["next"][field])} for field in HR4C_FIELDS}
        for key in ("authoritative_filenames", "scratch_filenames"):
            files = manifest.get(key)
            if not isinstance(files, dict) or set(files) != set(HR4C_FIELDS):
                raise ValueError("HR-4C manifest field-file mapping is invalid")
            for field in HR4C_FIELDS:
                if files[field] not in allowed[field]:
                    raise ValueError("HR-4C manifest slot filename is invalid")
        if any(
            manifest["authoritative_filenames"][field] == manifest["scratch_filenames"][field]
            for field in HR4C_FIELDS
        ):
            raise ValueError("HR-4C manifest field slot invariant failed")

    def _open_slots_from_manifest(self) -> None:
        self._current = {}
        self._next = {}
        all_slot_paths = [
            *self.slot_paths["current"].values(),
            *self.slot_paths["next"].values(),
        ]
        for field in HR4C_FIELDS:
            current = self.manifest["authoritative_filenames"][field]
            scratch = self.manifest["scratch_filenames"][field]
            current_path = next(path for path in all_slot_paths if path.name == current)
            scratch_path = next(path for path in all_slot_paths if path.name == scratch)
            if not current_path.is_file() or not scratch_path.is_file():
                raise FileNotFoundError("HR-4C persistent field slot is missing")
            current_map = np.lib.format.open_memmap(current_path, mode="r+")
            scratch_map = np.lib.format.open_memmap(scratch_path, mode="r+")
            if current_map.shape != self.state_shape or scratch_map.shape != self.state_shape:
                raise ValueError("HR-4C persistent field shape is invalid")
            if current_map.dtype != self.dtype or scratch_map.dtype != self.dtype:
                raise ValueError("HR-4C persistent field dtype is invalid")
            self._current[field] = current_map
            self._next[field] = scratch_map

    def _write_manifest(self) -> None:
        _atomic_json(self.manifest_path, self.manifest)

    def _replace_manifest(self, manifest: Mapping[str, Any]) -> None:
        """Durably replace the authority selector before changing local roles."""
        replacement = dict(manifest)
        _atomic_json(self.manifest_path, replacement)
        self.manifest = replacement

    def _bounds(self, start: int, stop: int) -> tuple[int, int]:
        first, last = int(start), int(stop)
        if first < 0 or last <= first or last > self.n_intervals:
            raise IndexError("HR-4C z batch is outside the persistent state")
        return first, last

    def read_authoritative_batch(self, start: int, stop: int) -> dict[str, np.ndarray]:
        first, last = self._bounds(start, stop)
        return {field: self._current[field][first:last] for field in HR4C_FIELDS}

    def begin_staging(self) -> None:
        if self.manifest["transaction_status"] != "committed":
            raise ValueError("HR-4C cannot begin a second staging transaction")
        self._written = {field: set() for field in HR4C_FIELDS}
        staging_manifest = dict(self.manifest)
        staging_manifest.update({
            "transaction_status": "staging",
            "staging_generation": int(self.manifest["generation"]) + 1,
            "last_abort": None,
        })
        self._replace_manifest(staging_manifest)

    def write_staging_field_batch(self, field: str, start: int, values) -> None:
        if self.manifest["transaction_status"] != "staging":
            raise ValueError("HR-4C staging write requires an active transaction")
        if field not in HR4C_FIELDS:
            raise ValueError("HR-4C staging field is invalid")
        batch = np.asarray(to_cpu(values), dtype=self.dtype)
        if batch.ndim != 3 or batch.shape[1:] != self.shape or batch.shape[0] <= 0:
            raise ValueError("HR-4C staging field batch must have shape [B, Ny, Nx]")
        first, last = self._bounds(start, int(start) + int(batch.shape[0]))
        if not np.all(np.isfinite(batch)):
            raise ValueError("HR-4C staging field batch must be finite")
        duplicate = set(range(first, last)) & self._written[field]
        if duplicate:
            raise ValueError("HR-4C staging screen may be written only once per field")
        self._next[field][first:last] = batch
        self._written[field].update(range(first, last))

    def write_staging_batch(self, start: int, values: Mapping[str, Any]) -> None:
        if set(values) != set(HR4C_FIELDS):
            raise ValueError("HR-4C staging batch requires delta_n, vx, and vy together")
        for field in HR4C_FIELDS:
            self.write_staging_field_batch(field, start, values[field])

    def validate_staging(self, *, batch_intervals: int) -> None:
        if self.manifest["transaction_status"] != "staging":
            raise ValueError("HR-4C staging validation requires an active transaction")
        batch = int(batch_intervals)
        if batch <= 0:
            raise ValueError("HR-4C batch_intervals must be positive")
        expected = set(range(self.n_intervals))
        for field in HR4C_FIELDS:
            if self._written[field] != expected:
                raise ValueError("HR-4C staging field completeness validation failed")
            array = self._next[field]
            if array.shape != self.state_shape or array.dtype != self.dtype:
                raise ValueError("HR-4C staging field layout validation failed")
            for start in range(0, self.n_intervals, batch):
                stop = min(start + batch, self.n_intervals)
                if not np.all(np.isfinite(array[start:stop])):
                    raise ValueError("HR-4C staging finite validation failed")

    def commit_staging(self, evolution: Mapping[str, Any]) -> None:
        try:
            batch = int(evolution.get("batch_intervals", 0))
            self.validate_staging(batch_intervals=batch)
            for field in HR4C_FIELDS:
                self._next[field].flush()
                _fsync_memmap(self._next[field].filename)
            committed_manifest = dict(self.manifest)
            committed_manifest.update({
                "transaction_status": "committed",
                "generation": int(self.manifest["staging_generation"]),
                "staging_generation": None,
                "authoritative_filenames": {field: Path(self._next[field].filename).name for field in HR4C_FIELDS},
                "scratch_filenames": {field: Path(self._current[field].filename).name for field in HR4C_FIELDS},
                "last_evolution": dict(evolution),
                "last_abort": None,
            })
            _atomic_json(self.manifest_path, committed_manifest)
        except Exception as error:
            self.abort_staging(reason=type(error).__name__)
            raise
        self._current, self._next = self._next, self._current
        self.manifest = committed_manifest
        self._written = {field: set() for field in HR4C_FIELDS}

    def abort_staging(self, *, reason: str) -> None:
        if self.manifest["transaction_status"] != "staging":
            return
        aborted_manifest = dict(self.manifest)
        aborted_manifest.update({
            "transaction_status": "committed",
            "staging_generation": None,
            "last_abort": str(reason),
        })
        self._replace_manifest(aborted_manifest)
        self._written = {field: set() for field in HR4C_FIELDS}

    def authoritative_metadata(self) -> dict[str, object]:
        return {
            "schema_version": self.manifest["schema_version"],
            "generation": self.manifest["generation"],
            "fields": tuple(HR4C_FIELDS),
            "state_shape": tuple(self.state_shape),
            "dtype": self.dtype.name,
            "grid_fingerprint": self.grid_fingerprint,
            "z_ordering": self.manifest["z_ordering"],
            "authoritative_filenames": dict(self.manifest["authoritative_filenames"]),
        }

    def close(self) -> None:
        for array in (*self._current.values(), *self._next.values()):
            array.flush()
        self._current = {}
        self._next = {}


def evolve_hr4_full_z(
    store: HR4CThreeFieldStore, *, dt_hydro: float, n_hydro_steps: int,
    batch_intervals: int, chi: float, nu: float, n0: float,
    gravity_x: float = 0.0, gravity_y: float = -9.81,
    cfl_limit: float = 1.0,
    failure_injector: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Create and atomically promote one full-z HR-4C evolution generation."""
    batch = int(batch_intervals)
    steps = int(n_hydro_steps)
    if batch <= 0 or steps <= 0:
        raise ValueError("HR-4C batch_intervals and n_hydro_steps must be positive")
    started = time.perf_counter()
    n_batches = 0
    max_operator_temp = 0
    total_screen_seconds = 0.0
    source_generation = int(store.manifest["generation"])
    store.begin_staging()
    try:
        for start in range(0, store.n_intervals, batch):
            stop = min(start + batch, store.n_intervals)
            if failure_injector is not None:
                failure_injector(start, stop)
            incoming = store.read_authoritative_batch(start, stop)
            outgoing = {field: np.empty_like(incoming[field]) for field in HR4C_FIELDS}
            for local in range(stop - start):
                result = advance_hr4_single_screen(
                    incoming["delta_n"][local], incoming["vx"][local], incoming["vy"][local],
                    dx=store.dx, dy=store.dy, dt_hydro=dt_hydro, chi=chi, nu=nu, n0=n0,
                    gravity_x=gravity_x, gravity_y=gravity_y, cfl_limit=cfl_limit,
                    n_steps=steps,
                )
                for field in HR4C_FIELDS:
                    outgoing[field][local] = np.asarray(to_cpu(result[field]), dtype=store.dtype)
                performance = result["performance"]
                total_screen_seconds += float(performance["wall_time_s_total"])
                max_operator_temp = max(max_operator_temp, int(performance["temporary_working_set_estimate_bytes"]))
            store.write_staging_batch(start, outgoing)
            n_batches += 1
            del incoming, outgoing
        evolution = {
            "operation": "full_z_interpulse_evolution",
            "source_generation": source_generation,
            "dt_hydro": float(dt_hydro),
            "n_hydro_steps": steps,
            "batch_intervals": batch,
            "chi": float(chi),
            "nu": float(nu),
            "gravity_x": float(gravity_x),
            "gravity_y": float(gravity_y),
            "z_scan_order": "z_batch_outer_then_screen_then_all_hydro_steps",
        }
        store.commit_staging(evolution)
    except Exception as error:
        store.abort_staging(reason=type(error).__name__)
        raise
    bytes_one_field = estimate_state_bytes(
        n_intervals=store.n_intervals, shape=store.shape, dtype=store.dtype,
    )
    return {
        "generation": int(store.manifest["generation"]),
        "source_generation": source_generation,
        "n_intervals": store.n_intervals,
        "n_batches": n_batches,
        "batch_intervals": batch,
        "n_hydro_steps": steps,
        "bytes_read": 3 * bytes_one_field,
        "bytes_written": 3 * bytes_one_field,
        "working_set_estimate_bytes": estimate_hr4c_working_set_bytes(
            batch_intervals=batch, shape=store.shape, dtype=store.dtype,
        ),
        "max_single_screen_operator_temp_bytes": max_operator_temp,
        "slow_time_history_stored": False,
        "full_z_materialized": False,
        "screen_operator_walltime_s_sum": total_screen_seconds,
        "walltime_s": time.perf_counter() - started,
    }


__all__ = [
    "HR4CThreeFieldStore",
    "HR4C_FIELDS",
    "HR4C_SCHEMA",
    "estimate_hr4c_generation_bytes",
    "estimate_hr4c_slot_bytes",
    "estimate_hr4c_working_set_bytes",
    "evolve_hr4_full_z",
]
