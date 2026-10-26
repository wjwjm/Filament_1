"""Bounded, per-owner BK-NEE factor tables for CPU-S4-4A.

The cache is deliberately a lifecycle helper, not a scientific operator.  A
caller opens one half-step with :meth:`begin_half`, asks for each plane through
:meth:`factor` (the caller supplies the original lambda body), and publishes a
new table only after the complete linear function has returned successfully
via :meth:`finish_half`.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import copy
import json
import time
from typing import Any, Callable, Mapping

import numpy as np

from tools.hpc_ops import provenance_v2 as pv


CACHE_SCHEMA = "filament.cpu_s4_4a.linear_factor_cache.v2"
ALGORITHM_ID = "bk_nee_transfer_kernel_exp_astype.v2"
DEFAULT_MAX_TABLES = 2
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
MAX_EVENTS = 64
KEY_METADATA_RESERVE = 16 * 1024


class CacheContractError(RuntimeError):
    """The candidate cache contract cannot be established."""


@dataclass(frozen=True)
class _Entry:
    key: tuple[Any, ...]
    table: tuple[np.ndarray, ...]
    factor_bytes: int
    key_reserve_bytes: int
    key_id: int
    geometry: tuple[int, int, int]
    public_key: dict[str, Any]


@dataclass
class _Pending:
    key: tuple[Any, ...]
    key_id: int
    geometry: tuple[int, int, int]
    work_dtype: np.dtype
    public_key: dict[str, Any]
    key_miss: bool
    entry: _Entry | None
    factors: list[np.ndarray]
    next_plane: int = 0
    requests: int = 0
    builds: int = 0
    hits: int = 0
    misses: int = 0
    build_attempts: int = 0
    failed_requests: int = 0
    guard_wall_s: float = 0.0
    build_wall_s: float = 0.0


def _dtype_token(dtype: Any) -> str:
    try:
        return np.dtype(dtype).str
    except (TypeError, ValueError) as exc:
        raise CacheContractError(f"unsupported dtype in cache key: {dtype!r}") from exc


def _scalar_bits(value: Any, *, name: str) -> bytes:
    try:
        return np.asarray(float(value), dtype=np.float64).tobytes()
    except (TypeError, ValueError, OverflowError) as exc:
        raise CacheContractError(f"{name} is not convertible to float64") from exc


def _validate_native_float_array(value: Any, *, name: str, ndim: int) -> tuple[np.ndarray, tuple[Any, ...]]:
    """Validate already-converted native CPU float64 arrays without copying."""

    if type(value) is not np.ndarray:
        raise CacheContractError(f"{name} must be a native NumPy ndarray")
    array = value
    if array.ndim != int(ndim):
        raise CacheContractError(f"{name} must be {ndim}D, got {array.ndim}D")
    if array.dtype != np.dtype(np.float64):
        raise CacheContractError(f"{name} must be native float64, got {array.dtype}")
    if not bool(array.flags.c_contiguous) or any(int(stride) <= 0 for stride in array.strides):
        raise CacheContractError(f"{name} must be positive-stride native C layout")
    if any(int(size) <= 0 for size in array.shape):
        raise CacheContractError(f"{name} must be non-empty")
    record = (
        tuple(int(size) for size in array.shape),
        str(array.dtype.str),
        array.tobytes(order="C"),
    )
    return array, record


def _backend_token(xp: Any, *, allow_toy_backend: bool) -> tuple[str, ...]:
    if allow_toy_backend and xp is np:
        return ("TOY_CPU_NUMPY", "numpy", "native", "unit-test-only")
    if type(xp).__name__ != "_XPNamespace":
        raise CacheContractError("unknown CPU backend namespace")
    underlying = getattr(xp, "xp", None)
    if underlying is not np:
        raise CacheContractError("CPU-S4-4A admits only native NumPy backend")
    fft = getattr(xp, "fft", None)
    dispatch = getattr(fft, "dispatch", None)
    if type(dispatch).__name__ != "_ScipyDispatch":
        raise CacheContractError("unknown or non-Scipy FFT dispatch")
    dispatch_module = str(type(dispatch).__module__).replace("\\", ".")
    if not dispatch_module.endswith("cpu_p1r2_j2"):
        raise CacheContractError("FFT dispatch is not the private J2 dispatch")
    return (
        "J2_CPU_NUMPY",
        str(type(xp).__module__).replace("\\", "."),
        type(xp).__qualname__,
        dispatch_module,
        type(dispatch).__qualname__,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    raise CacheContractError("cache metadata must contain only scalar values and containers")


class LinearFactorCache:
    """A cooperative single-controller cache of complete readonly factor tables."""

    def __init__(
        self,
        *,
        owner_identity: str,
        algorithm_source_identity: str,
        max_tables: int = DEFAULT_MAX_TABLES,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        if int(max_tables) < 1:
            raise ValueError("max_tables must be positive")
        if int(max_bytes) <= 0:
            raise ValueError("max_bytes must be positive")
        self.owner_identity = str(owner_identity)
        self.algorithm_source_identity = str(algorithm_source_identity)
        self.max_tables = int(max_tables)
        self.max_bytes = int(max_bytes)
        self._entries: OrderedDict[tuple[Any, ...], _Entry] = OrderedDict()
        self._pending: _Pending | None = None
        self._next_key_id = 1
        self._events: list[dict[str, Any]] = []
        self._events_dropped = 0
        self._persistent_bytes = 0
        self._stats: dict[str, int | float] = {
            "factor_requests": 0,
            "plane_builds": 0,
            "hits": 0,
            "misses": 0,
            "key_misses": 0,
            "build_attempts": 0,
            "failed_requests": 0,
            "entries": 0,
            "evictions": 0,
            "invalidations": 0,
            "cache_bytes": 0,
            "persistent_bytes_reserved": 0,
            "key_reserve_bytes": 0,
            "build_wall": 0.0,
            "lookup_guard_wall": 0.0,
        }

    @staticmethod
    def _key_reserve(omega_record: tuple[Any, ...], kperp_record: tuple[Any, ...]) -> int:
        return int(len(omega_record[2]) + len(kperp_record[2]) + KEY_METADATA_RESERVE)

    def _make_key(
        self,
        *,
        omega: Any,
        kperp2: Any,
        geometry: tuple[int, int, int] | None,
        work_dtype: Any,
        output_dtype: Any,
        k0: Any,
        omega0: Any,
        beta2: Any,
        denom_floor: Any,
        dz: Any,
        precision_strategy: Any,
        norm_strategy: Any,
        backend: Any,
        source_version: Any,
        allow_toy_backend: bool,
    ) -> tuple[tuple[Any, ...], tuple[int, int, int], int, dict[str, Any]]:
        omega_array, omega_record = _validate_native_float_array(omega, name="Omega", ndim=1)
        kperp_array, kperp_record = _validate_native_float_array(kperp2, name="kperp2", ndim=2)
        actual_geometry = (
            int(omega_array.shape[0]),
            int(kperp_array.shape[0]),
            int(kperp_array.shape[1]),
        )
        if geometry is not None and tuple(int(item) for item in geometry) != actual_geometry:
            raise CacheContractError("specified geometry does not match converted Omega/kperp2")
        if _dtype_token(work_dtype) != np.dtype(np.complex128).str:
            raise CacheContractError("CPU-S4-4A work dtype must be native complex128")
        if _dtype_token(output_dtype) != np.dtype(np.complex128).str:
            raise CacheContractError("CPU-S4-4A output dtype must be native complex128")
        source = str(source_version)
        if not source:
            raise CacheContractError("algorithm/source version is required")
        backend_token = _backend_token(backend, allow_toy_backend=allow_toy_backend)
        precision = str(precision_strategy or "baseline_complex64").lower()
        norm = str(norm_strategy or precision).lower()
        scalar_records = {
            "k0": _scalar_bits(k0, name="k0"),
            "omega0": _scalar_bits(omega0, name="omega0"),
            "beta2": _scalar_bits(beta2, name="beta2"),
            "denom_floor": _scalar_bits(denom_floor, name="denom_floor"),
            "dz": _scalar_bits(dz, name="dz"),
        }
        key = (
            CACHE_SCHEMA,
            ALGORITHM_ID,
            source,
            actual_geometry,
            omega_record,
            kperp_record,
            _dtype_token(work_dtype),
            _dtype_token(output_dtype),
            tuple((name, scalar_records[name]) for name in sorted(scalar_records)),
            precision,
            norm,
            backend_token,
        )
        public = {
            "geometry": list(actual_geometry),
            "Omega": {
                "shape": list(omega_record[0]),
                "dtype": omega_record[1],
                "raw_sha256": pv.raw_sha256(omega_record[2]),
            },
            "kperp2": {
                "shape": list(kperp_record[0]),
                "dtype": kperp_record[1],
                "raw_sha256": pv.raw_sha256(kperp_record[2]),
            },
            "work_dtype": _dtype_token(work_dtype),
            "output_dtype": _dtype_token(output_dtype),
            "precision_strategy": precision,
            "norm_strategy": norm,
            "backend": list(backend_token),
            "source_version": source,
            "scalar_bits": {name: scalar_records[name].hex() for name in sorted(scalar_records)},
        }
        return key, actual_geometry, self._key_reserve(omega_record, kperp_record), public

    def _sync_counts(self) -> None:
        self._stats["entries"] = int(len(self._entries))
        self._stats["cache_bytes"] = int(sum(entry.factor_bytes for entry in self._entries.values()))
        self._stats["key_reserve_bytes"] = int(sum(entry.key_reserve_bytes for entry in self._entries.values()))
        self._persistent_bytes = int(
            sum(entry.factor_bytes + entry.key_reserve_bytes for entry in self._entries.values())
        )
        self._stats["persistent_bytes_reserved"] = int(self._persistent_bytes)

    def _record_event(self, event: dict[str, Any]) -> None:
        self._events.append(_jsonable(event))
        if len(self._events) > MAX_EVENTS:
            self._events.pop(0)
            self._events_dropped += 1

    def _evict_for(self, factor_bytes: int, key_reserve_bytes: int) -> None:
        required = int(factor_bytes) + int(key_reserve_bytes)
        if required > self.max_bytes:
            raise CacheContractError(
                f"factor payload plus key reserve exceeds cache capacity: {required}>{self.max_bytes}"
            )
        while self._entries and (
            len(self._entries) >= self.max_tables
            or self._persistent_bytes + required > self.max_bytes
        ):
            key, entry = self._entries.popitem(last=False)
            self._stats["evictions"] = int(self._stats["evictions"]) + 1
            self._record_event(
                {
                    "event": "eviction",
                    "key_id": int(entry.key_id),
                    "geometry": list(entry.geometry),
                    "factor_bytes": int(entry.factor_bytes),
                    "key_reserve_bytes": int(entry.key_reserve_bytes),
                }
            )
        self._sync_counts()

    def begin_half(
        self,
        *,
        omega: Any,
        kperp2: Any,
        geometry: tuple[int, int, int] | None,
        work_dtype: Any,
        output_dtype: Any,
        k0: Any,
        omega0: Any,
        beta2: Any,
        denom_floor: Any,
        dz: Any,
        precision_strategy: Any,
        norm_strategy: Any,
        backend: Any,
        source_version: Any,
        allow_toy_backend: bool = False,
    ) -> None:
        if self._pending is not None:
            raise CacheContractError("a prior half-step has not been finished")
        guard_started = time.perf_counter()
        try:
            key, actual_geometry, key_reserve, public_key = self._make_key(
                omega=omega,
                kperp2=kperp2,
                geometry=geometry,
                work_dtype=work_dtype,
                output_dtype=output_dtype,
                k0=k0,
                omega0=omega0,
                beta2=beta2,
                denom_floor=denom_floor,
                dz=dz,
                precision_strategy=precision_strategy,
                norm_strategy=norm_strategy,
                backend=backend,
                source_version=source_version,
                allow_toy_backend=allow_toy_backend,
            )
        finally:
            guard_wall = time.perf_counter() - guard_started
            self._stats["lookup_guard_wall"] = float(self._stats["lookup_guard_wall"]) + guard_wall
        nt, ny, nx = actual_geometry
        factor_bytes = int(nt * ny * nx * np.dtype(work_dtype).itemsize)
        if key in self._entries:
            entry = self._entries.pop(key)
            self._entries[key] = entry
            key_miss = False
            entry_ref: _Entry | None = entry
            key_id = int(entry.key_id)
        else:
            self._stats["key_misses"] = int(self._stats["key_misses"]) + 1
            # Only complete tables are evicted, and only at this safe boundary.
            self._evict_for(factor_bytes, key_reserve)
            key_miss = True
            entry_ref = None
            key_id = int(self._next_key_id)
            self._next_key_id += 1
        self._pending = _Pending(
            key=key,
            key_id=key_id,
            geometry=actual_geometry,
            work_dtype=np.dtype(work_dtype),
            public_key=public_key,
            key_miss=key_miss,
            entry=entry_ref,
            factors=[],
            guard_wall_s=float(guard_wall),
        )

    @staticmethod
    def _readonly_factor(value: Any, *, geometry: tuple[int, int, int], work_dtype: np.dtype) -> np.ndarray:
        if type(value) is not np.ndarray:
            raise CacheContractError("factor builder returned a non-native NumPy array")
        expected_shape = (int(geometry[1]), int(geometry[2]))
        if tuple(int(item) for item in value.shape) != expected_shape:
            raise CacheContractError(f"factor plane shape mismatch: {value.shape}!={expected_shape}")
        if value.dtype != work_dtype:
            raise CacheContractError(f"factor plane dtype mismatch: {value.dtype}!={work_dtype}")
        if not bool(value.flags.c_contiguous) or any(int(stride) <= 0 for stride in value.strides):
            raise CacheContractError("factor plane must be positive-stride native C layout")
        setter = getattr(value, "setflags", None)
        if not callable(setter):
            raise CacheContractError("factor plane has no readonly setflags contract")
        try:
            setter(write=False)
        except (TypeError, ValueError) as exc:
            raise CacheContractError("factor plane could not be made readonly") from exc
        if bool(value.flags.writeable):
            raise CacheContractError("factor plane remained writable")
        return value

    def _event_payload(self, pending: _Pending, *, outcome: str, reason: str | None = None) -> dict[str, Any]:
        event = {
            "event": "half",
            "outcome": str(outcome),
            "key_id": int(pending.key_id),
            "key_miss_delta": int(bool(pending.key_miss)),
            "factor_requests_delta": int(pending.requests),
            "plane_builds_delta": int(pending.builds),
            "hits_delta": int(pending.hits),
            "misses_delta": int(pending.misses),
            "build_attempts_delta": int(pending.build_attempts),
            "failed_requests_delta": int(pending.failed_requests),
            "guard_wall_s": float(pending.guard_wall_s),
            "build_wall_s": float(pending.build_wall_s),
            "geometry": list(pending.geometry),
            "key": copy.deepcopy(pending.public_key),
        }
        if reason is not None:
            event["reason"] = str(reason)
        return event

    def _invalidate_pending(self, reason: str) -> None:
        pending = self._pending
        if pending is None:
            return
        self._pending = None
        if not pending.entry:
            self._stats["invalidations"] = int(self._stats["invalidations"]) + 1
        pending.factors.clear()
        self._record_event(self._event_payload(pending, outcome="failure", reason=reason))

    def factor(self, plane_index: int, builder: Callable[[], Any]) -> np.ndarray:
        pending = self._pending
        if pending is None:
            raise CacheContractError("factor requested outside begin_half/finish_half")
        index = int(plane_index)
        if index != int(pending.next_plane) or index < 0 or index >= int(pending.geometry[0]):
            self._invalidate_pending("plane_sequence_or_geometry_mismatch")
            raise CacheContractError("factor plane sequence does not match active half")
        pending.requests += 1
        self._stats["factor_requests"] = int(self._stats["factor_requests"]) + 1
        if pending.entry is not None:
            factor = pending.entry.table[index]
            # Recheck readonly state on every approved use; reflection that
            # changes it is outside the cooperative contract and fails closed.
            try:
                if bool(factor.flags.writeable):
                    raise CacheContractError("cached factor became writable")
            except AttributeError as exc:
                self._invalidate_pending("cached_factor_contract_changed")
                raise CacheContractError("cached factor lost native array flags") from exc
            pending.hits += 1
            self._stats["hits"] = int(self._stats["hits"]) + 1
            pending.next_plane = index + 1
            return factor
        pending.build_attempts += 1
        self._stats["build_attempts"] = int(self._stats["build_attempts"]) + 1
        started = time.perf_counter()
        try:
            if not callable(builder):
                raise CacheContractError("factor builder is not callable")
            factor = self._readonly_factor(
                builder(),
                geometry=pending.geometry,
                work_dtype=pending.work_dtype,
            )
        except BaseException:
            elapsed = time.perf_counter() - started
            pending.build_wall_s += elapsed
            self._stats["build_wall"] = float(self._stats["build_wall"]) + elapsed
            pending.failed_requests += 1
            self._stats["failed_requests"] = int(self._stats["failed_requests"]) + 1
            self._invalidate_pending("factor_builder_or_layout_failure")
            raise
        elapsed = time.perf_counter() - started
        pending.build_wall_s += elapsed
        self._stats["build_wall"] = float(self._stats["build_wall"]) + elapsed
        pending.factors.append(factor)
        pending.builds += 1
        pending.misses += 1
        self._stats["plane_builds"] = int(self._stats["plane_builds"]) + 1
        self._stats["misses"] = int(self._stats["misses"]) + 1
        pending.next_plane = index + 1
        return factor

    def finish_half(self, *, success: bool) -> None:
        pending = self._pending
        if pending is None:
            return
        if not bool(success):
            self._invalidate_pending("caller_or_linear_function_failure")
            return
        expected = int(pending.geometry[0])
        if pending.next_plane != expected:
            self._invalidate_pending("linear_function_returned_before_all_planes")
            raise CacheContractError("cannot publish an incomplete factor table")
        if pending.entry is None:
            if len(pending.factors) != expected:
                self._invalidate_pending("factor_count_incomplete")
                raise CacheContractError("cannot publish an incomplete factor table")
            table = tuple(pending.factors)
            factor_bytes = int(sum(int(factor.nbytes) for factor in table))
            # Recover the exact reserved byte count from the pending key's
            # private array records, without exposing those bytes in metadata.
            omega_record = pending.key[4]
            kperp_record = pending.key[5]
            key_reserve = int(self._key_reserve(omega_record, kperp_record))
            entry = _Entry(
                key=pending.key,
                table=table,
                factor_bytes=factor_bytes,
                key_reserve_bytes=key_reserve,
                key_id=int(pending.key_id),
                geometry=pending.geometry,
                public_key=copy.deepcopy(pending.public_key),
            )
            self._entries[pending.key] = entry
            pending.factors.clear()
            self._sync_counts()
        self._pending = None
        self._record_event(self._event_payload(pending, outcome="success"))

    def snapshot(self) -> dict[str, Any]:
        self._sync_counts()
        result = {
            "schema": CACHE_SCHEMA,
            "algorithm_id": ALGORITHM_ID,
            "algorithm_source_identity": self.algorithm_source_identity,
            "owner_identity": self.owner_identity,
            "max_tables": int(self.max_tables),
            "max_bytes": int(self.max_bytes),
            "persistent_capacity_scope": "factor_payload_plus_conservative_key_container_reserve",
            "cache_bytes_scope": "complete_factor_payload_only_not_process_peak",
            "events_dropped": int(self._events_dropped),
            "stats": {str(key): _jsonable(value) for key, value in self._stats.items()},
            "events": copy.deepcopy(self._events),
        }
        json.dumps(result, allow_nan=False)
        return copy.deepcopy(result)


__all__ = [
    "ALGORITHM_ID",
    "CACHE_SCHEMA",
    "CacheContractError",
    "LinearFactorCache",
]
