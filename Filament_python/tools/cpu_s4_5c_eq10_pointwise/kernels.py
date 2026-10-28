"""Two fixed-shape pointwise kernels for the Eq10 derivative candidate.

The implementation is intentionally narrow.  It accepts the real operands
produced by the source NumPy expression, writes a fresh native-layout output,
and does not perform a contiguous conversion or a scientific reduction.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from types import FunctionType
from typing import Any

import numpy as np
from tools.hpc_ops import provenance_v2 as _provenance_v2


CANDIDATE_IDENTITY = "EQ10_DERIVATIVE_POINTWISE_SAME_ORDER"
_ROLE = "C_EQ10_V1"
_BLOCK_SIZE = 32
_SUPPORTED_WORKERS = frozenset({1, 48})
_LOCAL_DIAGNOSTIC_WORKERS = 2
_MAX_DISPATCHES = 256


class KernelUnsupported(RuntimeError):
    """Raised when an input is outside the frozen candidate support domain."""


def _reject_optimized_python() -> None:
    if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE"):
        raise KernelUnsupported("PYTHONOPTIMIZE_FORBIDDEN")


_reject_optimized_python()


def source_identity() -> dict[str, str]:
    path = Path(__file__).resolve()
    repo = path.parents[3]
    relative = path.relative_to(repo).as_posix()
    oid = _provenance_v2._git_blob_oid(repo, relative)
    blob = _provenance_v2._read_git_blob(repo, oid)
    canonical = _provenance_v2.canonical_lf_sha256(path.read_bytes())
    if canonical != _provenance_v2.canonical_lf_sha256(blob):
        raise KernelUnsupported("KERNEL_SOURCE_DIFFERS_FROM_COMMITTED_CANONICAL_BLOB")
    return {
        "path": str(path),
        "relative_path": relative,
        "classification": "tracked_text",
        "hash_scope": _provenance_v2.TRACKED_HASH_SCOPE,
        "git_blob_oid": oid,
        "canonical_lf_sha256": canonical,
        "raw_sha256": _provenance_v2.raw_sha256_file(path),
        "raw_byte_observation": "Current checkout/cache observation only; not a cross-platform validation value",
    }


_NUMBA_BUNDLE: dict[str, Any] | None = None
_NUMBA_PRANGE: Any = range
_NUMBA_GET_THREAD_ID: Any = lambda: 0


def _k1_serial(coeff, spectrum, output, order_code, pixels, block_size, owners):
    blocks = (output.size + block_size * spectrum.shape[0] - 1) // (block_size * spectrum.shape[0])
    for block in _NUMBA_PRANGE(blocks):
        owner = _NUMBA_GET_THREAD_ID()
        if owner < owners.shape[0]:
            owners[owner] = 1
        start = block * block_size * spectrum.shape[0]
        stop = min(start + block_size * spectrum.shape[0], output.size)
        nt, ny, nx = output.shape
        for flat in range(start, stop):
            if order_code == 0:
                t = flat // (ny * nx)
                rem = flat - t * ny * nx
                y = rem // nx
                x = rem - y * nx
            else:
                x = flat // (nt * ny)
                rem = flat - x * nt * ny
                y = rem // nt
                t = rem - y * nt
            # Keep the source operation as one complex multiplication.
            output[t, y, x] = coeff[t, 0, 0] * spectrum[t, y, x]


def _k2_serial(real_view, scale, output, order_code, pixels, block_size, owners):
    blocks = (output.size + block_size * real_view.shape[0] - 1) // (block_size * real_view.shape[0])
    for block in _NUMBA_PRANGE(blocks):
        owner = _NUMBA_GET_THREAD_ID()
        if owner < owners.shape[0]:
            owners[owner] = 1
        start = block * block_size * real_view.shape[0]
        stop = min(start + block_size * real_view.shape[0], output.size)
        nt, ny, nx = output.shape
        for flat in range(start, stop):
            if order_code == 0:
                t = flat // (ny * nx)
                rem = flat - t * ny * nx
                y = rem // nx
                x = rem - y * nx
            else:
                x = flat // (nt * ny)
                rem = flat - x * nt * ny
                y = rem // nt
                t = rem - y * nt
            output[t, y, x] = real_view[t, y, x] / scale


_NUMBA_BUNDLE = None


def _load_numba_bundle() -> dict[str, Any]:
    global _NUMBA_BUNDLE, _NUMBA_PRANGE, _NUMBA_GET_THREAD_ID
    if _NUMBA_BUNDLE is not None:
        return _NUMBA_BUNDLE
    try:
        import numba
    except Exception as exc:  # pragma: no cover - environment dependent
        raise KernelUnsupported("NUMBA_UNAVAILABLE") from exc
    _NUMBA_PRANGE = numba.prange
    _NUMBA_GET_THREAD_ID = numba.get_thread_id
    bodies: dict[str, Any] = {"k1": _k1_serial, "k2": _k2_serial}
    dispatchers: dict[str, tuple[Any, Any]] = {}
    for name, serial_body in bodies.items():
        parallel_name = serial_body.__name__ + "_parallel"
        parallel_code = serial_body.__code__.replace(
            co_name=parallel_name, co_qualname=parallel_name
        )
        parallel_body = FunctionType(
            parallel_code,
            serial_body.__globals__,
            parallel_name,
            serial_body.__defaults__,
            serial_body.__closure__,
        )
        parallel_body.__module__ = serial_body.__module__
        parallel_body.__qualname__ = parallel_name
        globals()[parallel_name] = parallel_body
        dispatchers[name] = (
            numba.njit(serial_body, cache=True, fastmath=False, error_model="numpy"),
            numba.njit(parallel_body, cache=True, fastmath=False, parallel=True, error_model="numpy"),
        )
    cache_identity = {
        "source": source_identity(),
        "candidate": CANDIDATE_IDENTITY,
        "role": _ROLE,
        "block_size": _BLOCK_SIZE,
        "signatures": {
            "k1": "complex128[3d],complex128[3d]->complex128[3d]",
            "k2": "float64[3d],float64->float64[3d]",
        },
        "parallel": {"serial": False, "parallel": True},
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "numba": getattr(numba, "__version__", "unknown"),
            "NUMBA_THREADING_LAYER": os.environ.get("NUMBA_THREADING_LAYER", ""),
            "NUMBA_CPU_NAME": os.environ.get("NUMBA_CPU_NAME", ""),
            "NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS", ""),
        },
        "function_names": {
            "serial": {name: dispatchers[name][0].py_func.__name__ for name in bodies},
            "parallel": {name: dispatchers[name][1].py_func.__name__ for name in bodies},
        },
        "compiled_signatures": {},
    }
    cache_identity["cache_key"] = hashlib.sha256(
        json.dumps(cache_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _NUMBA_BUNDLE = {
        "numba": numba,
        "dispatchers": dispatchers,
        "executions": [],
        "cache_identity": cache_identity,
    }
    return _NUMBA_BUNDLE


def _new_runtime_view() -> dict[str, Any]:
    bundle = _load_numba_bundle()
    return {
        "numba": bundle["numba"],
        "dispatchers": bundle["dispatchers"],
        "executions": [],
        "cache_identity": bundle["cache_identity"],
    }


def _validate_array(value: Any, *, name: str, dtype: np.dtype[Any]) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise KernelUnsupported(f"{name}_MUST_BE_BASE_NUMPY_ARRAY")
    if value.ndim != 3:
        raise KernelUnsupported(f"{name}_MUST_BE_3D")
    if value.dtype != dtype:
        raise KernelUnsupported(f"{name}_DTYPE_MUST_BE_{dtype}")
    if any(int(stride) < 0 or (int(stride) == 0 and int(dim) != 1)
           for stride, dim in zip(value.strides, value.shape)):
        raise KernelUnsupported(f"{name}_REQUIRES_POSITIVE_STRIDES")
    if any(int(dim) <= 0 for dim in value.shape):
        raise KernelUnsupported(f"{name}_EMPTY_SHAPE_UNSUPPORTED")
    return value


def _validate_scale(scale: Any) -> float:
    if isinstance(scale, (bool, np.bool_)):
        raise KernelUnsupported("INTENSITY_SCALE_MUST_BE_FLOAT")
    if isinstance(scale, np.ndarray):
        if scale.ndim != 0:
            raise KernelUnsupported("INTENSITY_SCALE_MUST_BE_SCALAR")
        scale = scale.item()
    if not isinstance(scale, (float, np.floating)):
        raise KernelUnsupported("INTENSITY_SCALE_MUST_BE_FLOAT")
    value = float(scale)
    return value


def _layout(value: np.ndarray) -> dict[str, Any]:
    flags = value.flags
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "strides": list(value.strides),
        "c_contiguous": bool(flags.c_contiguous),
        "f_contiguous": bool(flags.f_contiguous),
        "owndata": bool(flags.owndata),
        "writeable": bool(flags.writeable),
        "base_type": None if value.base is None else type(value.base).__name__,
        "data_pointer": int(value.__array_interface__["data"][0]),
        "object_id": int(id(value)),
        "base_object_id": None if value.base is None else int(id(value.base)),
        "base_data_pointer": int(value.base.__array_interface__["data"][0]) if isinstance(value.base, np.ndarray) else None,
        "pointer_scope": "Current-process native relation only; never a cross-process identity",
    }


def _output_order_code(value: np.ndarray) -> int:
    if value.flags.c_contiguous:
        return 0
    if value.flags.f_contiguous:
        return 1
    raise KernelUnsupported("OUTPUT_MUST_BE_C_OR_F_CONTIGUOUS")


def _thread_mask(numba: Any, workers: int):
    maximum = int(numba.config.NUMBA_NUM_THREADS)
    if workers > maximum:
        raise KernelUnsupported(f"NUMBA_THREAD_LIMIT:{workers}>{maximum}")
    old = int(numba.get_num_threads())
    changed = old != workers

    @contextlib.contextmanager
    def manager():
        try:
            if changed:
                numba.set_num_threads(workers)
            yield
        finally:
            if changed:
                numba.set_num_threads(old)

    return manager()


class KernelExecutor:
    """Single-controller executor for the two Eq10 materialization kernels."""

    def __init__(
        self,
        *,
        workers: int = 1,
        role: str = _ROLE,
        block_size: int = _BLOCK_SIZE,
        diagnostic_local: bool = False,
    ):
        _reject_optimized_python()
        if role != _ROLE:
            raise KernelUnsupported("ROLE_MUST_BE_C_EQ10_V1")
        if isinstance(workers, bool):
            raise KernelUnsupported("WORKERS_MUST_BE_1_OR_48_OR_LOCAL_DIAGNOSTIC_2")
        workers = int(workers)
        if diagnostic_local:
            if workers != _LOCAL_DIAGNOSTIC_WORKERS:
                raise KernelUnsupported("LOCAL_DIAGNOSTIC_WORKERS_FIXED_2")
        elif workers not in _SUPPORTED_WORKERS:
            raise KernelUnsupported("WORKERS_MUST_BE_1_OR_48")
        if int(block_size) != _BLOCK_SIZE:
            raise KernelUnsupported("BLOCK_SIZE_FIXED_32")
        self.workers = workers
        self.role = role
        self.block_size = _BLOCK_SIZE
        self.diagnostic_local = bool(diagnostic_local)
        self._control_thread_id = threading.get_ident()
        self._lock = threading.Lock()
        self._runtime = _new_runtime_view()
        self._dispatches: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def _entry(self, operation: str):
        if threading.get_ident() != self._control_thread_id:
            raise KernelUnsupported("SINGLE_CONTROL_THREAD_REQUIRED")
        if not self._lock.acquire(blocking=False):
            raise KernelUnsupported("CONCURRENT_OR_REENTRANT_CALL_REJECTED")
        try:
            yield
        finally:
            self._lock.release()

    def _run(self, operation: str, args: tuple[Any, ...], inputs: tuple[np.ndarray, ...], output: np.ndarray) -> np.ndarray:
        _reject_optimized_python()
        if len(self._dispatches) >= _MAX_DISPATCHES:
            raise KernelUnsupported("DISPATCH_RECORD_LIMIT_EXCEEDED")
        numba = self._runtime["numba"]
        serial, parallel = self._runtime["dispatchers"][operation]
        dispatcher = serial if self.workers == 1 else parallel
        owners = np.zeros(self.workers, dtype=np.uint8)
        before = int(numba.get_num_threads())
        record: dict[str, Any] = {
            "operation": operation,
            "status": "STARTED",
            "requested_workers": self.workers,
            "block_size": self.block_size,
            "owner_flag_capacity": int(owners.size),
            "owner_flag_bytes": int(owners.nbytes),
            "thread_mask_before": before,
            "thread_mask_inside": None,
            "thread_mask_after": None,
            "active_thread_ids": [],
            "input_layouts": [_layout(value) for value in inputs],
            "output_layout": _layout(output),
            "native_alias_relations": {
                "output_shares_input": [bool(np.shares_memory(output, value)) for value in inputs],
                "input_pair_shares_memory": [bool(np.shares_memory(inputs[i], inputs[j]))
                    for i in range(len(inputs)) for j in range(i + 1, len(inputs))],
            },
            "function": {
                "serial_name": getattr(serial.py_func, "__name__", ""),
                "parallel_name": getattr(parallel.py_func, "__name__", ""),
                "parallel": self.workers != 1,
            },
        }
        try:
            with _thread_mask(numba, self.workers):
                record["thread_mask_inside"] = int(numba.get_num_threads())
                dispatcher(*(args + (owners,)))
            record["status"] = "PASS"
        except BaseException as exc:
            record["status"] = "ERROR"
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)
            raise
        finally:
            record["compiled_signatures"] = [str(signature) for signature in dispatcher.signatures]
            self._runtime["cache_identity"]["compiled_signatures"][operation] = list(record["compiled_signatures"])
            runtime_key = {
                "source": self._runtime["cache_identity"]["source"], "role": self.role,
                "operation": operation, "parallel": self.workers != 1,
                "actual_signatures": record["compiled_signatures"],
                "effective_environment": {
                    **self._runtime["cache_identity"]["environment"],
                    "CPU_NAME": str(numba.config.CPU_NAME),
                    "CPU_FEATURES": str(numba.config.CPU_FEATURES),
                    "THREADING_LAYER_CONFIG": str(numba.config.THREADING_LAYER),
                    "threading_layer_actual": numba.threading_layer() if self.workers != 1 else "serial",
                },
                "block_elements": int(output.shape[0]) * self.block_size,
                "fastmath": False, "error_model": "numpy",
            }
            record["compiled_cache_binding"] = runtime_key
            record["compiled_cache_key"] = hashlib.sha256(json.dumps(runtime_key, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            record["cache_key_semantics"] = "Post-compilation internal key includes actual signatures; NumPy source/canonical provenance is recorded separately"
            record["thread_mask_after"] = int(numba.get_num_threads())
            record["active_thread_ids"] = [int(item) for item in np.flatnonzero(owners)]
            record["present_thread_ids"] = list(record["active_thread_ids"])
            self._dispatches.append(record)
        return output

    def k1(self, coeff: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
        """Materialize ``coeff[:,None,None] * spectrum`` in source order."""
        with self._entry("k1"):
            coeff = _validate_array(coeff, name="COEFF", dtype=np.dtype("complex128"))
            spectrum = _validate_array(spectrum, name="SPECTRUM", dtype=np.dtype("complex128"))
            if coeff.shape[1:] != (1, 1) or coeff.shape[0] != spectrum.shape[0]:
                raise KernelUnsupported("COEFF_SHAPE_MUST_BE_NT_1_1")
            output = np.empty_like(spectrum, order="K", subok=False)
            order_code = _output_order_code(output)
            return self._run(
                "k1",
                (coeff, spectrum, output, order_code, int(spectrum.shape[1] * spectrum.shape[2]), self.block_size),
                (coeff, spectrum), output,
            )

    def k2(self, real_view: np.ndarray, scale: Any) -> np.ndarray:
        """Materialize ``real_view / scale`` while accepting native real views."""
        with self._entry("k2"):
            real_view = _validate_array(real_view, name="REAL_VIEW", dtype=np.dtype("float64"))
            scale_value = _validate_scale(scale)
            output = np.empty_like(real_view, order="K", subok=False)
            order_code = _output_order_code(output)
            return self._run(
                "k2",
                (real_view, scale_value, output, order_code, int(real_view.shape[1] * real_view.shape[2]), self.block_size),
                (real_view,), output,
            )

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "filament.cpu_s4_5c.pointwise_executor.v1",
            "candidate": CANDIDATE_IDENTITY,
            "role": self.role,
            "workers": self.workers,
            "diagnostic_local": self.diagnostic_local,
            "block_size": self.block_size,
            "control_thread_id": int(self._control_thread_id),
            "dispatch_count": len(self._dispatches),
            "dispatches": json.loads(json.dumps(self._dispatches)),
            "cache_identity": self._runtime["cache_identity"],
            "cache_policy": "serial and parallel functions have distinct Python code names and cache identities; no worker/block scan",
            "scientific_operations": {"fft": "external_unchanged", "ifft": "external_unchanged", "reductions": "external_unchanged"},
        }

