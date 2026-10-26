"""CPU-S4-3C Raman spatial-block candidate.

This module is an isolated adapter around the private propagator returned by
``tools.cpu_o1.cpu_o1.make_joint_propagator_factory``.  The production Raman
module is never patched.  The source-bound AST clone keeps the reference
intensity and IIR expressions intact while replacing only the five R1
operations assigned to S4-3C.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager, redirect_stdout
import copy
from dataclasses import dataclass
import functools
import importlib
import io
import inspect
from pathlib import Path
import subprocess
import threading
import time
from types import FunctionType
from typing import Any, Callable, Mapping

import numpy as np

from tools.hpc_ops import provenance_v2 as _provenance_v2


RAMAN_RELATIVE_PATH = "Filament_python/KHz_filament/raman.py"
RAMAN_SOURCE_BLOB = "5bb2702c109103ff882ee3642fedc748040c41b8"
RAMAN_CANONICAL_LF_SHA256 = (
    "bc211fcde7ac1ed1dd3891ce20abd46201903ebed4eb9859d9ff55bdcc5a6887"
)
CANDIDATE_SCHEMA = "filament.cpu_s4_3c.candidate.v1"
TRANSFORMATION_IDENTITY = "cpu_s4_3c_r8_raman_spatial_block_ast_clone_v1"
APPLY_NAME = "apply_isaacs_raman_operator_step"
STAGE_NAME = "isaacs_raman_stage"
SUPPORTED_WORKERS = frozenset({1, 24, 48, 96})
SUPPORTED_BLOCKS = frozenset({32, 128, 256})
MAX_SCRATCH_BYTES = 640 * 1024**2
OPERATION_NAMES = (
    "R_PRODUCT",
    "R_DERIVATIVE",
    "R_RHS",
    "H_PREDICTOR",
    "H_CORRECTOR",
)


class SourceBindingError(RuntimeError):
    """The frozen Raman source or one of its AST anchors drifted."""


class CandidateUnsupported(RuntimeError):
    """A call is outside the candidate support contract."""


class StageBindingError(ValueError):
    """A supplied stage payload is not safely bound to the current field."""


class WorkspaceAliasError(ValueError):
    """An input or externally supplied stage payload aliases another input."""


class StagePayload(dict):
    """A normal stage mapping with an optional non-invasive lifecycle token."""

    __slots__ = ("binding",)

    def __init__(self, payload: Mapping[str, Any], binding: Mapping[str, Any] | None = None):
        super().__init__(payload)
        self.binding = dict(binding) if binding is not None else None


def _source_function() -> Callable[..., Any]:
    """Return the imported Raman stage used to locate the source checkout."""
    try:
        module = importlib.import_module("KHz_filament.raman")
        function = getattr(module, STAGE_NAME)
    except Exception as exc:  # pragma: no cover - import environment dependent
        raise SourceBindingError("unable to import the Raman source function") from exc
    if not callable(function):
        raise SourceBindingError(f"imported Raman module has no callable {STAGE_NAME}")
    return function


def _source_path(source_function: Callable[..., Any]) -> Path:
    try:
        value = inspect.getsourcefile(source_function) or source_function.__code__.co_filename
    except Exception as exc:  # pragma: no cover - malformed callable
        raise SourceBindingError("unable to inspect the Raman source function") from exc
    path = Path(value).resolve()
    if not path.is_file():
        raise SourceBindingError(f"Raman source file is missing: {path}")
    return path


def _source_repo_root(source_path: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(source_path.parent), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise SourceBindingError(
            "unable to locate the Raman source Git checkout: "
            f"git rev-parse exited {result.returncode}: {result.stderr.strip()}"
        )
    return Path(result.stdout.strip()).resolve()


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise SourceBindingError(f"expected one {name} function, found {len(matches)}")
    return matches[0]


def _assignment_nodes(function_node: ast.FunctionDef, target: str) -> list[ast.Assign]:
    return [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == target
    ]


def _expr_dump(expression: str) -> str:
    return ast.dump(ast.parse(expression, mode="eval").body, include_attributes=False)


_PREDICTOR_DUMP = _expr_dump("(E + float(dz) * k1).astype(E.dtype, copy=False)")
_CORRECTOR_DUMP = _expr_dump(
    "(E + 0.5 * float(dz) * (k1 + stage2['rhs'])).astype(E.dtype, copy=False)"
)


def _source_identity(source_function: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Verify the frozen source and return the independent candidate binding."""
    source_function = _source_function() if source_function is None else source_function
    path = _source_path(source_function)
    repo_root = _source_repo_root(path)
    try:
        relative_path = path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise SourceBindingError(f"Raman source is outside its Git checkout: {path}") from exc
    if relative_path != RAMAN_RELATIVE_PATH:
        raise SourceBindingError(
            f"Raman source path drifted: {relative_path} != {RAMAN_RELATIVE_PATH}"
        )
    raw = path.read_bytes()
    record = {
        "path": relative_path,
        "classification": "tracked_text",
        "hash_scope": _provenance_v2.TRACKED_HASH_SCOPE,
        "git_blob_oid": RAMAN_SOURCE_BLOB,
        "canonical_lf_sha256": RAMAN_CANONICAL_LF_SHA256,
    }
    try:
        _provenance_v2.validate_record(repo_root, record, require_hash_scope=True)
    except Exception as exc:
        raise SourceBindingError(f"Raman provenance validation failed: {exc}") from exc
    canonical = _provenance_v2.canonical_lf_bytes(raw)
    blob = str(RAMAN_SOURCE_BLOB)
    canonical_sha = _provenance_v2.canonical_lf_sha256(raw)
    tree = ast.parse(canonical.decode("utf-8"), filename=str(path))
    stage_node = _find_function(tree, STAGE_NAME)
    apply_node = _find_function(tree, APPLY_NAME)
    response_nodes = _assignment_nodes(stage_node, "response")
    product_nodes = _assignment_nodes(stage_node, "product")
    derivative_nodes = _assignment_nodes(stage_node, "normalized_derivative")
    rhs_nodes = _assignment_nodes(stage_node, "rhs")
    predictor_nodes = _assignment_nodes(apply_node, "predictor")
    result_nodes = _assignment_nodes(apply_node, "result")
    if len(response_nodes) != 2 or not any(
        isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "raman_convolve_intensity"
        for node in response_nodes
    ):
        raise SourceBindingError("Raman response AST anchors are not unique or have changed")
    if len(product_nodes) != 1 or len(derivative_nodes) != 1 or len(rhs_nodes) != 1:
        raise SourceBindingError("Raman stage AST anchors are not unique or have changed")
    if len(predictor_nodes) != 1 or len(result_nodes) != 2:
        raise SourceBindingError("Raman Heun AST anchors are not unique or have changed")
    predictor_matches = [
        node
        for node in predictor_nodes
        if ast.dump(node.value, include_attributes=False) == _PREDICTOR_DUMP
    ]
    corrector_matches = [
        node
        for node in result_nodes
        if ast.dump(node.value, include_attributes=False) == _CORRECTOR_DUMP
    ]
    if len(predictor_matches) != 1 or len(corrector_matches) != 1:
        raise SourceBindingError("Raman predictor/corrector AST anchors changed")
    return {
        "schema": CANDIDATE_SCHEMA,
        "transformation": TRANSFORMATION_IDENTITY,
        "relative_path": RAMAN_RELATIVE_PATH,
        "source_path": str(path),
        "git_blob": blob,
        "canonical_lf_sha256": canonical_sha,
        "raw_sha256": _provenance_v2.raw_sha256(raw),
        "stage_function": STAGE_NAME,
        "apply_function": APPLY_NAME,
        "allowed_replacements": {
            "R_INTENSITY": "reference_numpy_expression",
            "R_IIR": "reference_raman_convolve_intensity exact_piecewise_linear",
            "R_PRODUCT": (
                "_s43c_product block kernel; source float64-array times "
                "complex128-array component multiplication with directed IEEE zero semantics"
            ),
            "R_DERIVATIVE": "FFT adapter + _s43c_normalized_derivative pointwise kernel",
            "R_RHS": "_s43c_rhs pointwise kernel",
            "H_PREDICTOR": "_s43c_predictor block kernel",
            "H_CORRECTOR": "_s43c_corrector block kernel",
        },
        "ast_anchors": {
            "stage_function_firstlineno": int(stage_node.lineno),
            "apply_function_firstlineno": int(apply_node.lineno),
            "response_assignments": [int(node.lineno) for node in response_nodes],
            "product": int(product_nodes[0].lineno),
            "normalized_derivative": int(derivative_nodes[0].lineno),
            "rhs": int(rhs_nodes[0].lineno),
            "predictor": int(predictor_matches[0].lineno),
            "corrector": int(corrector_matches[0].lineno),
        },
    }


def source_identity(source_function: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Return a fresh, verified source binding for the frozen Raman source."""
    return _source_identity(source_function)


def _candidate_source_identity() -> dict[str, Any]:
    path = Path(__file__).resolve()
    raw = path.read_bytes()
    return {
        "path": str(path),
        "raw_sha256": _provenance_v2.raw_sha256(raw),
        "canonical_lf_sha256": _provenance_v2.canonical_lf_sha256(raw),
    }


# Numba is intentionally imported lazily.  The parent process must set the
# compiler/threading environment before importing this module's kernels.
_NUMBA_BUNDLE: dict[str, Any] | None = None
_NUMBA_PRANGE = range
_NUMBA_GET_THREAD_ID: Callable[[], int] = lambda: 0


def _product_body(response, field, output, pixels, block_size, owners):
    size = field.shape[0]
    block_count = (pixels + block_size - 1) // block_size
    for block_id in _NUMBA_PRANGE(block_count):
        owner_id = _NUMBA_GET_THREAD_ID()
        if owner_id < owners.shape[0]:
            owners[owner_id] = 1
        start = block_id * block_size
        stop = min(start + block_size, pixels)
        for time_index in range(size // pixels):
            for pixel in range(start, stop):
                flat = time_index * pixels + pixel
                scale = response[flat]
                value = field[flat]
                # Keep a rounded nonzero-operand product, including underflow.
                # Exact zero operands retain the source array expression's complex
                # zero-term addition/subtraction. The directed vector oracle binds
                # this behavior; no NumPy internal algorithm is assumed.
                real = scale * value.real
                imag = scale * value.imag
                if scale == 0.0 or value.real == 0.0:
                    real = real - 0.0 * value.imag
                if scale == 0.0 or value.imag == 0.0:
                    imag = imag + 0.0 * value.real
                output[flat] = complex(real, imag)


def _derivative_body(spectrum, multiplier, output, pixels, block_size, owners):
    size = spectrum.shape[0]
    block_count = (pixels + block_size - 1) // block_size
    for block_id in _NUMBA_PRANGE(block_count):
        owner_id = _NUMBA_GET_THREAD_ID()
        if owner_id < owners.shape[0]:
            owners[owner_id] = 1
        start = block_id * block_size
        stop = min(start + block_size, pixels)
        for time_index in range(size // pixels):
            for pixel in range(start, stop):
                flat = time_index * pixels + pixel
                output[flat] = multiplier[flat // pixels] * spectrum[flat]


def _rhs_body(product, derivative, output, prefactor, pixels, block_size, owners):
    size = product.shape[0]
    block_count = (pixels + block_size - 1) // block_size
    for block_id in _NUMBA_PRANGE(block_count):
        owner_id = _NUMBA_GET_THREAD_ID()
        if owner_id < owners.shape[0]:
            owners[owner_id] = 1
        start = block_id * block_size
        stop = min(start + block_size, pixels)
        for time_index in range(size // pixels):
            for pixel in range(start, stop):
                flat = time_index * pixels + pixel
                first = 1j * prefactor * product[flat]
                second = prefactor * derivative[flat]
                output[flat] = first - second


def _predictor_body(field, k1, output, dz, pixels, block_size, owners):
    size = field.shape[0]
    block_count = (pixels + block_size - 1) // block_size
    for block_id in _NUMBA_PRANGE(block_count):
        owner_id = _NUMBA_GET_THREAD_ID()
        if owner_id < owners.shape[0]:
            owners[owner_id] = 1
        start = block_id * block_size
        stop = min(start + block_size, pixels)
        for time_index in range(size // pixels):
            for pixel in range(start, stop):
                flat = time_index * pixels + pixel
                increment = dz * k1[flat]
                output[flat] = field[flat] + increment


def _corrector_body(field, k1, k2, output, dz, pixels, block_size, owners):
    size = field.shape[0]
    block_count = (pixels + block_size - 1) // block_size
    for block_id in _NUMBA_PRANGE(block_count):
        owner_id = _NUMBA_GET_THREAD_ID()
        if owner_id < owners.shape[0]:
            owners[owner_id] = 1
        start = block_id * block_size
        stop = min(start + block_size, pixels)
        for time_index in range(size // pixels):
            for pixel in range(start, stop):
                flat = time_index * pixels + pixel
                slope_sum = k1[flat] + k2[flat]
                increment = 0.5 * dz * slope_sum
                output[flat] = field[flat] + increment


def _load_numba_bundle() -> dict[str, Any]:
    global _NUMBA_BUNDLE, _NUMBA_PRANGE, _NUMBA_GET_THREAD_ID
    if _NUMBA_BUNDLE is not None:
        return _NUMBA_BUNDLE
    try:
        import numba
    except Exception as exc:  # pragma: no cover - depends on the local environment
        raise CandidateUnsupported("numba_unavailable") from exc
    _NUMBA_PRANGE = numba.prange
    _NUMBA_GET_THREAD_ID = numba.get_thread_id
    bodies = {
        "product": _product_body,
        "derivative": _derivative_body,
        "rhs": _rhs_body,
        "predictor": _predictor_body,
        "corrector": _corrector_body,
    }
    kernels: dict[str, tuple[Any, Any]] = {}
    for name, body in bodies.items():
        # Numba 0.61.2 disk keys do not distinguish these two compiler modes
        # for one Python body. Give the parallel clone a distinct function
        # identity while preserving exactly the same instruction sequence.
        parallel_name = body.__name__ + "_parallel"
        parallel_code = body.__code__.replace(
            co_name=parallel_name, co_qualname=parallel_name
        )
        parallel_body = FunctionType(
            parallel_code, body.__globals__, parallel_name,
            body.__defaults__, body.__closure__,
        )
        parallel_body.__module__ = body.__module__
        parallel_body.__qualname__ = parallel_name
        globals()[parallel_name] = parallel_body
        kernels[name] = (
            numba.njit(body, cache=True, fastmath=False),
            numba.njit(parallel_body, cache=True, fastmath=False, parallel=True),
        )
    _NUMBA_BUNDLE = {
        "numba": numba,
        "kernels": kernels,
        "executions": [],
        "execution_limit": 256,
        "kernel_metadata": {},
    }
    return _NUMBA_BUNDLE


def _new_runtime_view() -> dict[str, Any]:
    """Share compiled dispatchers while isolating execution evidence per state."""
    bundle = _load_numba_bundle()
    return {
        "numba": bundle["numba"],
        "kernels": bundle["kernels"],
        "executions": [],
        "execution_limit": int(bundle.get("execution_limit", 256)),
        "kernel_metadata": {},
    }


@contextmanager
def _thread_mask(numba: Any, workers: int):
    """Set only the current Numba mask and restore it on every exit path."""
    maximum = int(numba.config.NUMBA_NUM_THREADS)
    if workers > maximum:
        raise CandidateUnsupported(f"numba_thread_limit:{workers}>{maximum}")
    old = int(numba.get_num_threads())
    changed = False
    try:
        if old != workers:
            numba.set_num_threads(workers)
            changed = True
        yield
    finally:
        if changed:
            numba.set_num_threads(old)


def _run_kernel(
    name: str,
    args: tuple[Any, ...],
    *,
    workers: int,
    runtime: dict[str, Any] | None = None,
) -> None:
    bundle = runtime if runtime is not None else _load_numba_bundle()
    numba = bundle["numba"]
    serial, parallel = bundle["kernels"][name]
    kernel = serial if workers == 1 else parallel
    owner_count = max(1, min(int(workers), 96))
    owners = np.zeros(owner_count, dtype=np.uint8, order="C")
    record = {
        "kernel": str(name),
        "requested_workers": int(workers),
        "owner_flag_capacity": int(owner_count),
        "owner_flag_bytes": int(owners.nbytes),
        "thread_mask_before": int(numba.get_num_threads()),
        "thread_mask_inside": None,
        "thread_mask_after": None,
        "active_thread_ids": [],
    }
    try:
        with _thread_mask(numba, workers):
            record["thread_mask_inside"] = int(numba.get_num_threads())
            kernel(*(args + (owners,)))
    finally:
        record["thread_mask_after"] = int(numba.get_num_threads())
        record["active_thread_ids"] = [
            int(index) for index in np.flatnonzero(owners)
        ]
        record["active_thread_count"] = int(len(record["active_thread_ids"]))
        executions = bundle.setdefault("executions", [])
        executions.append(record)
        limit = max(1, int(bundle.get("execution_limit", 256)))
        if len(executions) > limit:
            del executions[:-limit]
        metadata = bundle.setdefault("kernel_metadata", {})
        entry = metadata.setdefault(str(name), {})
        entry["serial_signatures"] = [str(item) for item in serial.signatures]
        entry["parallel_signatures"] = [str(item) for item in parallel.signatures]
        entry["parallel_requested"] = bool(workers != 1)
        if workers != 1 and "parallel_diagnostics" not in entry:
            try:
                diagnostic_stream = io.StringIO()
                with redirect_stdout(diagnostic_stream):
                    diagnostic_result = parallel.parallel_diagnostics(level=4)
                diagnostic_text = diagnostic_stream.getvalue()
                if diagnostic_result is not None:
                    diagnostic_text += str(diagnostic_result)
                entry["parallel_diagnostics"] = diagnostic_text
            except Exception as exc:  # pragma: no cover - Numba-version dependent
                entry["parallel_diagnostics_error"] = type(exc).__name__


def _require_1d_c(value: Any, *, name: str, dtype: np.dtype) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy ndarray")
    if value.ndim != 1 or value.dtype != dtype or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be dense C {dtype} 1-D")
    return value


def _require_3d_c(value: Any, *, name: str, dtype: np.dtype) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy ndarray")
    if value.ndim != 3 or value.dtype != dtype or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be dense C {dtype} 3-D")
    if any(int(size) < 1 for size in value.shape):
        raise ValueError(f"{name} must have positive dimensions")
    if any(int(stride) <= 0 for stride in value.strides):
        raise ValueError(f"{name} has non-positive strides")
    return value


def product_kernel(
    response: np.ndarray,
    field: np.ndarray,
    *,
    block_size: int = 32,
    workers: int = 1,
    runtime: dict[str, Any] | None = None,
) -> np.ndarray:
    """Evaluate response times field in dense C order."""
    response = _require_3d_c(response, name="response", dtype=np.dtype("float64"))
    field = _require_3d_c(field, name="field", dtype=np.dtype("complex128"))
    if response.shape != field.shape:
        raise ValueError("response and field shapes differ")
    output = np.empty(field.shape, dtype=np.complex128, order="C")
    _run_kernel(
        "product",
        (response.reshape(-1), field.reshape(-1), output.reshape(-1), int(field.shape[1] * field.shape[2]), int(block_size)),
        workers=int(workers),
        runtime=runtime,
    )
    return output


def derivative_kernel(
    spectrum: np.ndarray,
    multiplier: np.ndarray,
    *,
    block_size: int = 32,
    workers: int = 1,
    runtime: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply only the pointwise spectral multiplier; FFTs remain external."""
    spectrum = _require_3d_c(spectrum, name="spectrum", dtype=np.dtype("complex128"))
    multiplier = _require_1d_c(multiplier, name="multiplier", dtype=np.dtype("complex128"))
    if spectrum.shape[0] != multiplier.shape[0]:
        raise ValueError("multiplier length differs from spectrum time axis")
    output = np.empty(spectrum.shape, dtype=np.complex128, order="C")
    pixels = int(spectrum.shape[1] * spectrum.shape[2])
    _run_kernel(
        "derivative",
        (spectrum.reshape(-1), multiplier, output.reshape(-1), pixels, int(block_size)),
        workers=int(workers),
        runtime=runtime,
    )
    return output


def rhs_kernel(
    product: np.ndarray,
    normalized_derivative: np.ndarray,
    prefactor: float,
    *,
    block_size: int = 32,
    workers: int = 1,
    runtime: dict[str, Any] | None = None,
) -> np.ndarray:
    """Evaluate the Eq. (27) pointwise RHS expression in source order."""
    product = _require_3d_c(product, name="product", dtype=np.dtype("complex128"))
    normalized_derivative = _require_3d_c(
        normalized_derivative, name="normalized_derivative", dtype=np.dtype("complex128")
    )
    if product.shape != normalized_derivative.shape:
        raise ValueError("product and normalized_derivative shapes differ")
    output = np.empty(product.shape, dtype=np.complex128, order="C")
    _run_kernel(
        "rhs",
        (product.reshape(-1), normalized_derivative.reshape(-1), output.reshape(-1), float(prefactor), int(product.shape[1] * product.shape[2]), int(block_size)),
        workers=int(workers),
        runtime=runtime,
    )
    return output


def predictor_kernel(
    field: np.ndarray,
    k1: np.ndarray,
    dz: float,
    *,
    block_size: int = 32,
    workers: int = 1,
    runtime: dict[str, Any] | None = None,
) -> np.ndarray:
    """Evaluate the Heun predictor in dense C order."""
    field = _require_3d_c(field, name="field", dtype=np.dtype("complex128"))
    k1 = _require_3d_c(k1, name="k1", dtype=np.dtype("complex128"))
    if field.shape != k1.shape:
        raise ValueError("field and k1 shapes differ")
    output = np.empty(field.shape, dtype=np.complex128, order="C")
    _run_kernel(
        "predictor",
        (field.reshape(-1), k1.reshape(-1), output.reshape(-1), float(dz), int(field.shape[1] * field.shape[2]), int(block_size)),
        workers=int(workers),
        runtime=runtime,
    )
    return output


def corrector_kernel(
    field: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
    dz: float,
    *,
    block_size: int = 32,
    workers: int = 1,
    runtime: dict[str, Any] | None = None,
) -> np.ndarray:
    """Evaluate the Heun corrector in dense C order."""
    field = _require_3d_c(field, name="field", dtype=np.dtype("complex128"))
    k1 = _require_3d_c(k1, name="k1", dtype=np.dtype("complex128"))
    k2 = _require_3d_c(k2, name="k2", dtype=np.dtype("complex128"))
    if field.shape != k1.shape or field.shape != k2.shape:
        raise ValueError("field and Heun RHS shapes differ")
    output = np.empty(field.shape, dtype=np.complex128, order="C")
    _run_kernel(
        "corrector",
        (
            field.reshape(-1),
            k1.reshape(-1),
            k2.reshape(-1),
            output.reshape(-1),
            float(dz),
            int(field.shape[1] * field.shape[2]),
            int(block_size),
        ),
        workers=int(workers),
        runtime=runtime,
    )
    return output


def _c_array(value: np.ndarray) -> np.ndarray:
    """Return *value* in C layout for a deterministic raw-byte seal."""
    return value if value.flags.c_contiguous else np.ascontiguousarray(value)


def _array_raw_sha256(value: np.ndarray) -> str:
    c_array = _c_array(value)
    return _provenance_v2.raw_sha256(memoryview(c_array).cast("B"))


def _array_content_token(value: np.ndarray) -> dict[str, Any]:
    return {
        "shape": [int(s) for s in value.shape],
        "dtype": value.dtype.str,
        "strides_bytes": [int(s) for s in value.strides],
        "raw_sha256": _array_raw_sha256(value),
    }


def _scalar_token(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return (
            "ndarray",
            value.dtype.str,
            tuple(int(s) for s in value.shape),
            _array_raw_sha256(value),
        )
    return (type(value).__name__, repr(value))


def _parameter_token(kwargs: Mapping[str, Any]) -> tuple[Any, ...]:
    names = ("Omega", "dt", "omega0", "n0", "n_R", "omega_R", "Gamma_R")
    tokens = [(name, _scalar_token(kwargs.get(name))) for name in names]
    method = str(kwargs.get("method", "iir")).lower()
    sampling = str(
        kwargs.get("iir_sampling", "exact_piecewise_linear")
    ).lower()
    tokens.extend((("method", method), ("iir_sampling", sampling)))
    # The supported exact-IIR source path does not read chunk_pixels.  Keep
    # that fact explicit in the identity while retaining the value for paths
    # where the source FFT implementation gives it execution meaning.
    chunk_token: Any
    if method == "iir":
        chunk_token = ("ignored_for_iir",)
    else:
        chunk_token = _scalar_token(kwargs.get("chunk_pixels", 65536))
    tokens.append(("chunk_pixels", chunk_token))
    return tuple(tokens)


def _field_token(field: np.ndarray, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "shape": [int(s) for s in field.shape],
        "dtype": field.dtype.str,
        "strides_bytes": [int(s) for s in field.strides],
        "content_raw_sha256": _array_raw_sha256(field),
        "data_pointer": int(field.__array_interface__["data"][0]),
        "object_id": int(id(field)),
        "parameter_token": _parameter_token(kwargs),
    }


def _stage_payload_token(payload: Mapping[str, Any]) -> dict[str, Any]:
    rhs = payload.get("rhs")
    q_positive = payload.get("q_R_positive")
    if not isinstance(rhs, np.ndarray) or not isinstance(q_positive, np.ndarray):
        raise StageBindingError("stage payload content seal requires rhs and q_R_positive")
    return {
        "rhs": _array_content_token(rhs),
        "q_R_positive": _array_content_token(q_positive),
    }


def _stage_binding(
    field: np.ndarray,
    kwargs: Mapping[str, Any],
    origin: str,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "filament.cpu_s4_3c.stage_binding.v1",
        "origin": str(origin),
        "field": _field_token(field, kwargs),
        "payload": _stage_payload_token(payload) if payload is not None else None,
        "created_monotonic_ns": time.monotonic_ns(),
    }


def _validate_stage_payload(
    field: np.ndarray,
    payload: Any,
    kwargs: Mapping[str, Any],
    *,
    require_c: bool,
    require_binding: bool = True,
) -> str | None:
    if payload is None or (isinstance(payload, Mapping) and len(payload) == 0):
        return None
    if not isinstance(payload, Mapping):
        return "stage1_not_mapping"
    rhs = payload.get("rhs")
    q_positive = payload.get("q_R_positive")
    if not isinstance(rhs, np.ndarray) or not isinstance(q_positive, np.ndarray):
        return "stage1_required_arrays_missing"
    if rhs.dtype != np.dtype("complex128") or rhs.shape != field.shape:
        return "stage1_rhs_shape_or_dtype"
    if q_positive.dtype != np.dtype("float64") or q_positive.shape != field.shape[1:]:
        return "stage1_q_shape_or_dtype"
    if np.shares_memory(field, rhs) or np.shares_memory(field, q_positive):
        raise WorkspaceAliasError("field aliases external stage1 payload")
    if np.shares_memory(rhs, q_positive):
        raise WorkspaceAliasError("external stage1 rhs aliases q_R_positive")
    if any(int(s) <= 0 for s in rhs.strides + q_positive.strides):
        raise WorkspaceAliasError("external stage1 payload has non-positive strides")
    if require_c and (not rhs.flags.c_contiguous or not q_positive.flags.c_contiguous):
        return "stage1_non_native_C_layout"
    binding = getattr(payload, "binding", None)
    if binding is None:
        if require_binding:
            raise StageBindingError("STAGE_IDENTITY_UNVERIFIED")
        return None
    if binding is not None:
        if not isinstance(binding, Mapping) or binding.get("schema") != "filament.cpu_s4_3c.stage_binding.v1":
            raise StageBindingError("external stage1 binding token is invalid")
        if binding.get("origin") not in {
            "reference",
            "candidate",
            "source_bound_internal",
        }:
            raise StageBindingError("external stage1 binding origin is invalid")
        lifecycle = binding.get("created_monotonic_ns")
        if isinstance(lifecycle, bool) or not isinstance(lifecycle, int) or lifecycle <= 0:
            raise StageBindingError("external stage1 lifecycle token is invalid")
        expected = _field_token(field, kwargs)
        supplied = binding.get("field")
        if supplied != expected:
            raise StageBindingError("external stage1 field or Raman parameter identity differs")
        supplied_payload = binding.get("payload")
        if not isinstance(supplied_payload, Mapping):
            raise StageBindingError("STAGE_PAYLOAD_UNVERIFIED")
        expected_payload = _stage_payload_token(payload)
        if supplied_payload != expected_payload:
            raise StageBindingError("external stage1 payload content identity differs")
    return None


def bind_reference_stage1(
    field: np.ndarray,
    stage1: Mapping[str, Any],
    *,
    kwargs: Mapping[str, Any] | None = None,
    **parameters: Any,
) -> StagePayload:
    """Attach an optional lifecycle token to a reference-produced stage.

    Reference mappings must pass through this helper before candidate apply so
    field, parameter, and lifecycle identity are explicit without recomputing
    any scientific quantity.
    """
    field = _require_3d_c(field, name="field", dtype=np.dtype("complex128"))
    supplied_kwargs = dict(kwargs or {})
    supplied_kwargs.update(parameters)
    reason = _validate_stage_payload(
        field, stage1, supplied_kwargs, require_c=False, require_binding=False
    )
    if reason is not None:
        raise StageBindingError(reason)
    return StagePayload(
        stage1,
        _stage_binding(field, supplied_kwargs, "reference", stage1),
    )


def _support_reason(
    field: Any,
    kwargs: Mapping[str, Any],
    *,
    raman_namespace: Mapping[str, Any],
    apply: bool,
) -> str | None:
    if not isinstance(field, np.ndarray):
        return "non_numpy_field_or_gpu_backend"
    if field.ndim != 3 or field.dtype != np.dtype("complex128"):
        return "field_dtype_or_ndim_not_complex128_3d"
    if any(int(size) < 1 for size in field.shape):
        return "field_has_non_positive_dimensions"
    if not field.flags.c_contiguous:
        return "field_layout_not_native_C"
    if any(int(s) <= 0 for s in field.strides):
        return "field_has_non_positive_strides"
    proxy = raman_namespace.get("xp")
    backend = getattr(proxy, "xp", proxy)
    if backend is not np:
        return "gpu_or_non_numpy_backend"
    method = str(kwargs.get("method", "iir")).lower()
    if method != "iir":
        return "method_not_iir"
    sampling = str(kwargs.get("iir_sampling", "exact_piecewise_linear")).lower()
    if sampling != "exact_piecewise_linear":
        return "iir_sampling_not_exact_piecewise_linear"
    if apply and str(kwargs.get("integrator", "heun")).lower() != "heun":
        return "integrator_not_heun"
    for name in ("Omega", "dt", "omega0", "n0", "n_R", "omega_R", "Gamma_R"):
        if name not in kwargs:
            return f"missing_parameter:{name}"
    if kwargs.get("omega_R") is None or kwargs.get("Gamma_R") is None:
        return "rotational_parameters_missing"
    return None


def _ast_call(expression: str) -> ast.expr:
    return ast.parse(expression, mode="eval").body


def _clone_transformed_function(
    source_path: Path,
    namespace: dict[str, Any],
    function_name: str,
    transform: Callable[[ast.FunctionDef], None],
) -> Callable[..., Any]:
    tree = ast.parse(
        _provenance_v2.canonical_lf_bytes(source_path.read_bytes()).decode("utf-8"),
        filename=str(source_path),
    )
    function_node = _find_function(tree, function_name)
    function_node = copy.deepcopy(function_node)
    transform(function_node)
    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)
    local_namespace: dict[str, Any] = {}
    exec(compile(module, str(source_path), "exec"), namespace, local_namespace)
    transformed = local_namespace.get(function_name)
    if not callable(transformed):
        raise SourceBindingError(f"AST clone did not produce {function_name}")
    source_function = namespace.get(function_name)
    if callable(source_function):
        transformed.__defaults__ = source_function.__defaults__
        transformed.__kwdefaults__ = source_function.__kwdefaults__
        transformed.__annotations__ = getattr(source_function, "__annotations__", {})
        transformed.__module__ = source_function.__module__
        transformed.__qualname__ = source_function.__qualname__
    return transformed


def _transform_stage(function_node: ast.FunctionDef) -> None:
    # R_INTENSITY and R_IIR remain the source expressions.  Only the five R1
    # spatial-block operations below are replaced in the private AST clone.
    replacements = {
        "product": "_s43c_product(response, E, rdtype)",
        "normalized_derivative": "_s43c_normalized_derivative(multiplier, product, xp)",
        "rhs": "_s43c_rhs(product, normalized_derivative, prefactor, ctype)",
    }
    for target, expression in replacements.items():
        nodes = _assignment_nodes(function_node, target)
        if len(nodes) != 1:
            raise SourceBindingError(f"candidate {target} splice anchor changed")
        nodes[0].value = ast.copy_location(_ast_call(expression), nodes[0].value)


def _transform_apply(function_node: ast.FunctionDef) -> None:
    predictor_nodes = [
        node
        for node in _assignment_nodes(function_node, "predictor")
        if ast.dump(node.value, include_attributes=False) == _PREDICTOR_DUMP
    ]
    corrector_nodes = [
        node
        for node in _assignment_nodes(function_node, "result")
        if ast.dump(node.value, include_attributes=False) == _CORRECTOR_DUMP
    ]
    if len(predictor_nodes) != 1 or len(corrector_nodes) != 1:
        raise SourceBindingError("candidate Heun splice anchors changed")
    predictor_nodes[0].value = ast.copy_location(
        _ast_call("_s43c_predictor(E, k1, float(dz))"), predictor_nodes[0].value
    )
    corrector_nodes[0].value = ast.copy_location(
        _ast_call("_s43c_corrector(E, k1, stage2['rhs'], float(dz))"),
        corrector_nodes[0].value,
    )


def _clone_reference_pair(
    raman_namespace: dict[str, Any],
    reference_stage: Callable[..., Any],
    reference_apply: Callable[..., Any],
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    reference_globals = dict(raman_namespace)
    stage_clone = FunctionType(
        reference_stage.__code__, reference_globals, reference_stage.__name__,
        reference_stage.__defaults__, reference_stage.__closure__,
    )
    stage_clone.__kwdefaults__ = reference_stage.__kwdefaults__
    stage_clone.__annotations__ = getattr(reference_stage, "__annotations__", {})
    stage_clone.__module__ = reference_stage.__module__
    stage_clone.__qualname__ = reference_stage.__qualname__
    reference_globals[STAGE_NAME] = stage_clone
    apply_clone = FunctionType(
        reference_apply.__code__, reference_globals, reference_apply.__name__,
        reference_apply.__defaults__, reference_apply.__closure__,
    )
    apply_clone.__kwdefaults__ = reference_apply.__kwdefaults__
    apply_clone.__annotations__ = getattr(reference_apply, "__annotations__", {})
    apply_clone.__module__ = reference_apply.__module__
    apply_clone.__qualname__ = reference_apply.__qualname__
    return stage_clone, apply_clone


def _make_product(
    response: np.ndarray,
    field: np.ndarray,
    rdtype: Any,
    *,
    block_size: int,
    workers: int,
    runtime: dict[str, Any],
    observe: Callable[[str, Any], None],
) -> np.ndarray:
    response_cast = response.astype(rdtype, copy=False)
    output = product_kernel(
        response_cast,
        field,
        block_size=block_size,
        workers=workers,
        runtime=runtime,
    )
    observe("R_PRODUCT", output)
    return output


def _make_normalized_derivative(
    multiplier: np.ndarray,
    product: np.ndarray,
    xp: Any,
    *,
    block_size: int,
    workers: int,
    runtime: dict[str, Any],
    observe: Callable[[str, Any], None],
) -> np.ndarray:
    fft = xp.fft
    spectrum = fft.fft(product, axis=0)
    if not isinstance(spectrum, np.ndarray) or spectrum.dtype != np.dtype("complex128"):
        raise CandidateUnsupported("FFT_spectrum_not_native_complex128")
    if not spectrum.flags.c_contiguous:
        raise CandidateUnsupported("FFT_spectrum_not_native_C")
    multiplier_array = np.asarray(multiplier, dtype=np.complex128)
    if multiplier_array.ndim != 1 or not multiplier_array.flags.c_contiguous:
        multiplier_array = np.ascontiguousarray(multiplier_array)
    scaled = derivative_kernel(
        spectrum,
        multiplier_array,
        block_size=block_size,
        workers=workers,
        runtime=runtime,
    )
    normalized = fft.ifft(scaled, axis=0)
    observe("R_DERIVATIVE", normalized)
    return normalized


def _make_rhs(
    product: np.ndarray,
    normalized_derivative: np.ndarray,
    prefactor: float,
    ctype: Any,
    *,
    block_size: int,
    workers: int,
    runtime: dict[str, Any],
    observe: Callable[[str, Any], None],
) -> np.ndarray:
    output = rhs_kernel(
        product,
        normalized_derivative,
        prefactor,
        block_size=block_size,
        workers=workers,
        runtime=runtime,
    )
    observe("R_RHS", output)
    return output.astype(ctype, copy=False)


def _make_predictor(
    field: np.ndarray,
    k1: np.ndarray,
    dz: float,
    *,
    block_size: int,
    workers: int,
    runtime: dict[str, Any],
    observe: Callable[[str, Any], None],
) -> np.ndarray:
    output = predictor_kernel(
        field, k1, dz, block_size=block_size, workers=workers, runtime=runtime
    )
    observe("H_PREDICTOR", output)
    return output


def _make_corrector(
    field: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
    dz: float,
    *,
    block_size: int,
    workers: int,
    runtime: dict[str, Any],
    observe: Callable[[str, Any], None],
) -> np.ndarray:
    output = corrector_kernel(
        field, k1, k2, dz, block_size=block_size, workers=workers, runtime=runtime
    )
    observe("H_CORRECTOR", output)
    return output


@dataclass
class CandidateState:
    """Per-factory candidate state and auditable dispatch surface."""

    raman_namespace: dict[str, Any]
    reference_stage: Callable[..., Any]
    reference_apply: Callable[..., Any]
    reference_stage_clone: Callable[..., Any]
    reference_apply_clone: Callable[..., Any]
    candidate_stage_inner: Callable[..., Any]
    candidate_apply_inner: Callable[..., Any]
    source_identity: dict[str, Any]
    block_size: int
    raman_workers: int
    observe_callback: Callable[[str, Any], None] | None = None
    runtime: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._local = threading.local()
        self._control_thread_id = int(threading.main_thread().ident or -1)
        self._runtime = self.runtime if self.runtime is not None else _new_runtime_view()
        self._stats: dict[str, Any] = {
            "schema": CANDIDATE_SCHEMA,
            "candidate_family": "RAMAN_SPATIAL_BLOCK_PARALLEL_V1",
            "block_size": int(self.block_size),
            "raman_workers": int(self.raman_workers),
            "candidate_calls": 0,
            "reference_fallback_calls": 0,
            "operation_counts": {name: 0 for name in OPERATION_NAMES},
            "fallback_reasons": {},
            "workspace": {
                "per_worker_bytes": 0,
                "max_bytes": int(MAX_SCRATCH_BYTES),
                "formula": "no private per-worker scratch; each block writes disjoint full-field output elements",
                "full_field_outputs_are_candidate_roles": True,
                "owner_flag_bytes": int(min(self.raman_workers, 96)),
                "owner_flag_dtype": "uint8",
                "owner_flag_limit_bytes": 96,
                "owner_flag_bound_reference_bytes": int(16 * self.block_size),
            },
        }
        # Expose callable kernels for directed bitwise tests without exposing
        # internal Numba dispatch objects.
        self.kernels = {
            "product": functools.partial(
                product_kernel,
                block_size=self.block_size,
                workers=self.raman_workers,
                runtime=self._runtime,
            ),
            "derivative": functools.partial(
                derivative_kernel,
                block_size=self.block_size,
                workers=self.raman_workers,
                runtime=self._runtime,
            ),
            "rhs": functools.partial(
                rhs_kernel,
                block_size=self.block_size,
                workers=self.raman_workers,
                runtime=self._runtime,
            ),
            "predictor": functools.partial(
                predictor_kernel,
                block_size=self.block_size,
                workers=self.raman_workers,
                runtime=self._runtime,
            ),
            "corrector": functools.partial(
                corrector_kernel,
                block_size=self.block_size,
                workers=self.raman_workers,
                runtime=self._runtime,
            ),
        }

    @property
    def stats(self) -> dict[str, Any]:
        return self.metrics()

    def metrics(self) -> dict[str, Any]:
        result = dict(self._stats)
        result["operation_counts"] = dict(self._stats["operation_counts"])
        result["fallback_reasons"] = dict(self._stats["fallback_reasons"])
        result["events"] = list(self.events)
        result["source_identity"] = dict(self.source_identity)
        result["candidate_source_identity"] = _candidate_source_identity()
        result["execution"] = dict(getattr(self, "execution", {}))
        result["numba_runtime_executions"] = [
            dict(item) for item in self._runtime.get("executions", [])
        ]
        result["numba_kernel_metadata"] = copy.deepcopy(
            self._runtime.get("kernel_metadata", {})
        )
        return result

    def _observe(self, name: str, value: Any) -> None:
        if self.observe_callback is not None:
            self.observe_callback(str(name), value)

    @contextmanager
    def _entry(self, kind: str):
        current_thread_id = int(threading.get_ident())
        if current_thread_id != self._control_thread_id:
            raise RuntimeError(
                "CPU-S4-3C candidate requires the main control thread "
                f"({self._control_thread_id}), got {current_thread_id}"
            )
        depth = int(getattr(self._local, "depth", 0))
        if depth == 0:
            if not self._lock.acquire(blocking=False):
                raise RuntimeError("CPU-S4-3C candidate rejects concurrent/reentrant calls")
        self._local.depth = depth + 1
        try:
            yield
        finally:
            new_depth = int(getattr(self._local, "depth", 1)) - 1
            self._local.depth = new_depth
            if depth == 0:
                self._lock.release()

    def _record_candidate(self, path: str, counts: Mapping[str, int], field: Any) -> None:
        self._stats["candidate_calls"] += 1
        for name, count in counts.items():
            self._stats["operation_counts"][name] += int(count)
        self.events.append(
            {
                "kind": "CANDIDATE_EXECUTED",
                "path": str(path),
                "field_shape": list(getattr(field, "shape", ())),
                "field_dtype": str(getattr(field, "dtype", "")),
                "block_size": int(self.block_size),
                "raman_workers": int(self.raman_workers),
                "actual_op_counts": {name: int(counts.get(name, 0)) for name in OPERATION_NAMES},
            }
        )

    def _record_fallback(self, path: str, reason: str, field: Any) -> None:
        reason = str(reason)
        self._stats["reference_fallback_calls"] += 1
        reasons = self._stats["fallback_reasons"]
        reasons[reason] = int(reasons.get(reason, 0)) + 1
        self.events.append(
            {
                "kind": "REFERENCE_FALLBACK",
                "path": str(path),
                "reason": reason,
                "field_shape": list(getattr(field, "shape", ())),
                "field_dtype": str(getattr(field, "dtype", "")),
                "actual_op_counts": {name: 0 for name in OPERATION_NAMES},
            }
        )

    def _stage_dispatch(
        self,
        field: Any,
        *,
        _source_bound_internal: bool = False,
        **kwargs: Any,
    ) -> Any:
        reason = _support_reason(
            field, kwargs, raman_namespace=self.raman_namespace, apply=False
        )
        if reason is not None:
            self._record_fallback("stage", reason, field)
            return self.reference_stage_clone(field, **kwargs)
        try:
            result = self.candidate_stage_inner(field, **kwargs)
        except CandidateUnsupported as exc:
            self._record_fallback("stage", str(exc), field)
            return self.reference_stage_clone(field, **kwargs)
        if isinstance(result, Mapping):
            binding_origin = (
                "source_bound_internal" if _source_bound_internal else "candidate"
            )
            result = StagePayload(
                result,
                _stage_binding(
                    field,
                    kwargs,
                    binding_origin,
                    None if _source_bound_internal else result,
                ),
            )
        self._record_candidate(
            "stage",
            {
                "R_PRODUCT": 1,
                "R_DERIVATIVE": 1,
                "R_RHS": 1,
                "H_PREDICTOR": 0,
                "H_CORRECTOR": 0,
            },
            field,
        )
        return result

    def stage(self, field: Any, **kwargs: Any) -> Any:
        source_bound_internal = int(getattr(self._local, "depth", 0)) > 0
        with self._entry("stage"):
            return self._stage_dispatch(
                field,
                _source_bound_internal=source_bound_internal,
                **kwargs,
            )

    def _apply_dispatch(self, field: Any, dz: Any, **kwargs: Any) -> Any:
        reason = _support_reason(
            field, kwargs, raman_namespace=self.raman_namespace, apply=True
        )
        stage1 = kwargs.get("stage1")
        if reason is None:
            if isinstance(field, np.ndarray) and field.dtype == np.dtype("complex128"):
                stage_reason = _validate_stage_payload(
                    field, stage1, kwargs, require_c=True
                )
                if stage_reason is not None:
                    reason = stage_reason
        if reason is not None:
            self._record_fallback("apply", reason, field)
            return self.reference_apply_clone(field, dz, **kwargs)
        try:
            result = self.candidate_apply_inner(field, dz, **kwargs)
        except CandidateUnsupported as exc:
            self._record_fallback("apply", str(exc), field)
            return self.reference_apply_clone(field, dz, **kwargs)
        self._record_candidate(
            "apply",
            {
                "R_PRODUCT": 0,
                "R_DERIVATIVE": 0,
                "R_RHS": 0,
                "H_PREDICTOR": 1,
                "H_CORRECTOR": 1,
            },
            field,
        )
        return result

    def apply(self, field: Any, dz: Any, **kwargs: Any) -> Any:
        with self._entry("apply"):
            return self._apply_dispatch(field, dz, **kwargs)


def apply_candidate(state: CandidateState, field: Any, dz: Any, **kwargs: Any) -> Any:
    """Explicit helper for callers holding a ``CandidateState``."""
    if not isinstance(state, CandidateState):
        raise TypeError("apply_candidate requires a CandidateState as its first argument")
    return state.apply(field, dz, **kwargs)


def _validate_factory_value(name: str, value: Any, allowed: frozenset[int]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in allowed:
        choices = "/".join(str(item) for item in sorted(allowed))
        raise ValueError(f"{name} must be one of {choices}")
    return int(value)


def make_candidate_factory(
    kernel_threads: int,
    temporal_workers: int,
    spatial_workers: int,
    raman_workers: int = 1,
    block_size: int = 32,
    *,
    observe: Callable[[str, Any], None] | None = None,
):
    """Return a fresh private propagator factory for RAMAN_SPATIAL_BLOCK_PARALLEL_V1."""
    if isinstance(kernel_threads, bool) or not isinstance(kernel_threads, int) or not 1 <= kernel_threads <= 96:
        raise ValueError("kernel_threads must be an integer in [1, 96]")
    temporal_workers = _validate_factory_value("temporal_workers", temporal_workers, frozenset({1, 8, 24, 48, 96}))
    spatial_workers = _validate_factory_value("spatial_workers", spatial_workers, frozenset({1, 8, 24, 48, 96}))
    raman_workers = _validate_factory_value("raman_workers", raman_workers, SUPPORTED_WORKERS)
    block_size = _validate_factory_value("block_size", block_size, SUPPORTED_BLOCKS)
    # Validate source identity before the factory can create a propagator.
    source_binding = _source_identity()
    from tools.cpu_o1.cpu_o1 import make_joint_propagator_factory

    base_factory = make_joint_propagator_factory(
        "I",
        kernel_threads=int(kernel_threads),
        # Keep the existing I ionization plan at its reference block size;
        # C's Raman block size is private to this module and its kernels.
        block_pixels=32,
        layout="NtB",
        temporal_workers=int(temporal_workers),
        spatial_workers=int(spatial_workers),
    )

    def factory(frozen_propagator):
        propagator, dispatch = base_factory(frozen_propagator)
        raman_namespace = getattr(dispatch, "cpu_o1_raman_namespace", None)
        if raman_namespace is None:
            private_stage = propagator.__globals__.get(STAGE_NAME)
            raman_namespace = getattr(private_stage, "__globals__", None)
        if not isinstance(raman_namespace, dict):
            raise SourceBindingError("private I propagator has no mutable Raman namespace")
        import KHz_filament.raman as production_raman

        if raman_namespace is vars(production_raman):
            raise SourceBindingError("candidate refuses the production Raman namespace")
        reference_stage = raman_namespace.get(STAGE_NAME)
        reference_apply = raman_namespace.get(APPLY_NAME)
        if not callable(reference_stage) or not callable(reference_apply):
            raise SourceBindingError("private Raman namespace lacks stage/apply functions")
        source = _source_identity(reference_stage)
        reference_stage_clone, reference_apply_clone = _clone_reference_pair(
            raman_namespace, reference_stage, reference_apply
        )
        state_holder: dict[str, CandidateState] = {}

        # Candidate helper functions are bound into the private namespace only.
        # Each factory receives a fresh namespace from the existing I factory.
        requested_observer = observe

        def notify(name: str, value: Any) -> None:
            state = state_holder.get("state")
            if state is not None:
                state._observe(name, value)

        runtime = _new_runtime_view()
        raman_namespace["_s43c_product"] = functools.partial(
            _make_product,
            block_size=block_size,
            workers=raman_workers,
            runtime=runtime,
            observe=notify,
        )
        raman_namespace["_s43c_normalized_derivative"] = functools.partial(
            _make_normalized_derivative,
            block_size=block_size,
            workers=raman_workers,
            runtime=runtime,
            observe=notify,
        )
        raman_namespace["_s43c_rhs"] = functools.partial(
            _make_rhs,
            block_size=block_size,
            workers=raman_workers,
            runtime=runtime,
            observe=notify,
        )
        raman_namespace["_s43c_predictor"] = functools.partial(
            _make_predictor,
            block_size=block_size,
            workers=raman_workers,
            runtime=runtime,
            observe=notify,
        )
        raman_namespace["_s43c_corrector"] = functools.partial(
            _make_corrector,
            block_size=block_size,
            workers=raman_workers,
            runtime=runtime,
            observe=notify,
        )
        source_path = _source_path(reference_stage)
        candidate_stage_inner = _clone_transformed_function(
            source_path, raman_namespace, STAGE_NAME, _transform_stage
        )
        candidate_apply_inner = _clone_transformed_function(
            source_path, raman_namespace, APPLY_NAME, _transform_apply
        )
        state = CandidateState(
            raman_namespace=raman_namespace,
            reference_stage=reference_stage,
            reference_apply=reference_apply,
            reference_stage_clone=reference_stage_clone,
            reference_apply_clone=reference_apply_clone,
            candidate_stage_inner=candidate_stage_inner,
            candidate_apply_inner=candidate_apply_inner,
            source_identity=source,
            block_size=block_size,
            raman_workers=raman_workers,
            observe_callback=requested_observer,
            runtime=runtime,
        )
        state_holder["state"] = state

        # The wrapper is installed in both private lookup sites.  The cloned
        # apply function resolves this exact stage wrapper for stage two; its
        # thread-local depth allows the nested stage call without opening a
        # second control-thread entry.
        raman_namespace[STAGE_NAME] = state.stage
        raman_namespace[APPLY_NAME] = state.apply
        propagator.__globals__[STAGE_NAME] = state.stage
        propagator.__globals__[APPLY_NAME] = state.apply

        execution = {
            "schema": CANDIDATE_SCHEMA,
            "candidate_family": "RAMAN_SPATIAL_BLOCK_PARALLEL_V1",
            "transformation": TRANSFORMATION_IDENTITY,
            "kernel_threads": int(kernel_threads),
            "temporal_workers": int(temporal_workers),
            "spatial_workers": int(spatial_workers),
            "raman_workers": int(raman_workers),
            "block_size": int(block_size),
            "layout": "native_C",
            "control_thread_policy": "main_thread_only",
            "control_thread_id": int(state._control_thread_id),
            "supported": {
                "backend": "numpy_cpu",
                "dtype": "complex128/float64",
                "method": "iir",
                "iir_sampling": "exact_piecewise_linear",
                "integrator": "heun",
                "rotational_model": "full_isaacs_raman_operator",
            },
            "fallback": "F/transpose/positive_slice/c64/GPU/other configs",
            "source_identity": dict(source),
            "candidate_source_identity": _candidate_source_identity(),
            "scratch": {
                "formula": "no private per-worker scratch; each block writes disjoint full-field output elements",
                "per_worker_bytes": 0,
                "owner_flag_bytes": int(min(raman_workers, 96)),
                "owner_flag_dtype": "uint8",
                "owner_flag_limit_bytes": 96,
                "owner_flag_bound_reference_bytes": int(16 * block_size),
                "max_bytes": int(MAX_SCRATCH_BYTES),
            },
        }
        state.execution = execution
        state.dispatch = dispatch
        state.propagator = propagator
        state.candidate_stage = state.stage
        state.candidate_apply = state.apply
        propagator._cpu_s4_3c_state = state
        propagator._cpu_s4_3c_candidate = state
        propagator._cpu_s4_3c_execution = dict(execution)
        propagator._cpu_s4_3c_source_identity = dict(source)
        dispatch.cpu_s4_3c_state = state
        dispatch.cpu_s4_3c_candidate = state
        dispatch.cpu_s4_3c_stage = state.stage
        dispatch.cpu_s4_3c_apply = state.apply
        dispatch.cpu_s4_3c_kernels = state.kernels
        dispatch.cpu_s4_3c_candidate_events = state.events
        dispatch.cpu_s4_3c_stats = state._stats
        dispatch.cpu_s4_3c_source_identity = dict(source)
        dispatch.cpu_s4_3c_execution = dict(execution)
        dispatch.cpu_s4_3c_dispatch = {
            "stage": state.stage,
            "apply": state.apply,
            "kernels": state.kernels,
            "events": state.events,
        }
        factory.last_propagator = propagator
        factory.last_dispatch = dispatch
        factory.last_state = state
        factory.last_candidate = state
        factory.last_source_identity = dict(source)
        return propagator, dispatch

    factory.last_propagator = None
    factory.last_dispatch = None
    factory.last_state = None
    factory.last_candidate = None
    factory.last_source_identity = dict(source_binding)
    factory.candidate_family = "RAMAN_SPATIAL_BLOCK_PARALLEL_V1"
    factory.candidate_schema = CANDIDATE_SCHEMA
    return factory


__all__ = [
    "CandidateState",
    "CandidateUnsupported",
    "SourceBindingError",
    "StageBindingError",
    "StagePayload",
    "WorkspaceAliasError",
    "apply_candidate",
    "bind_reference_stage1",
    "corrector_kernel",
    "derivative_kernel",
    "make_candidate_factory",
    "predictor_kernel",
    "product_kernel",
    "rhs_kernel",
    "source_identity",
]
