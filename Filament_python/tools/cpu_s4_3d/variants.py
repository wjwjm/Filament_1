"""CPU-S4-3D private variants built from the verified S4-3C R8 module.

This module is an adapter, not a second Raman implementation.  Every variant
loads the frozen R8 candidate into a fresh, uniquely named module namespace.
The R8 stage/apply glue, binding checks, state bookkeeping, FFT calls, and
diagnostic path therefore remain the authority.  ``C_FLAT_EQWORK`` changes
only the five scalar Numba loop bodies; ``I_BIND`` changes only those five
operations to the original NumPy expressions while retaining the same R8
entry and binding path.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import FunctionType, ModuleType
from typing import Any, Callable

import numpy as np

from tools.hpc_ops import provenance_v2 as _provenance_v2


R8_RAW_SHA256 = "252fc1e6521812c29872d5b848e4082841f0b8ae0ad15661394f6b31347e0a32"
R8_SIZE_BYTES = 59808
VARIANT_SCHEMA = "filament.cpu_s4_3d.variant.v1"
_VARIANTS = frozenset({"C_SPATIAL32", "C_FLAT_EQWORK", "I_BIND"})
_MODULE_CACHE: dict[tuple[str, str, str, str], ModuleType] = {}


class OutputSlot:
    """Small result holder used by the NumPy binding adapters.

    It intentionally allocates no output array.  The expression assigned to
    ``value`` owns the normal NumPy result, which is returned directly.
    """

    __slots__ = ("value",)

    def __init__(self, value: Any = None) -> None:
        self.value = value


def _default_r8_path() -> Path:
    return (Path(__file__).resolve().parent.parent / "cpu_s4_3c" / "candidate.py").resolve()


def _variant_source_raw_sha256() -> str:
    """Use the repository's sole raw-byte provenance implementation."""
    return _provenance_v2.raw_sha256_file(Path(__file__).resolve())


def _check_r8(path: Path) -> tuple[Path, str]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen R8 candidate is missing: {path}")
    size = int(path.stat().st_size)
    if size != R8_SIZE_BYTES:
        raise RuntimeError(f"frozen R8 size drift: {size} != {R8_SIZE_BYTES}")
    digest = _provenance_v2.raw_sha256_file(path)
    if digest != R8_RAW_SHA256:
        raise RuntimeError(f"frozen R8 raw SHA256 drift: {digest} != {R8_RAW_SHA256}")
    return path, digest


def _clone_into(module: ModuleType, function: Callable[..., Any], public_name: str) -> Callable[..., Any]:
    """Clone a helper into private R8 globals with a stable code identity."""
    token = str(module.__dict__["__cpu_s4d_identity_token__"])
    stable_name = f"{public_name}_{token}"
    code = function.__code__.replace(
        co_name=stable_name,
        co_qualname=stable_name,
    )
    clone = FunctionType(
        code,
        module.__dict__,
        stable_name,
        function.__defaults__,
        function.__closure__,
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = getattr(function, "__annotations__", {})
    clone.__module__ = module.__name__
    clone.__qualname__ = f"{module.__name__}.{stable_name}"
    module.__dict__[public_name] = clone
    return clone


def _load_r8(path: Path, variant: str, source_hash: str, r8_hash: str) -> ModuleType:
    """Import the complete R8 file under an isolated module name."""
    token = f"{variant.lower()}_{source_hash[:16]}_{r8_hash[:16]}"
    name = f"_filament_cpu_s4d_r8_{token}"
    existing = sys.modules.get(name)
    if isinstance(existing, ModuleType):
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to create frozen R8 import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__cpu_s4_3d_private__"] = True
    module.__dict__["__cpu_s4d_identity_token__"] = token
    module.__dict__["__cpu_s4d_variant_source_raw_sha256__"] = source_hash
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    module.__dict__["__cpu_s4_3d_r8_path__"] = str(path)
    module.__dict__["__cpu_s4_3d_r8_raw_sha256__"] = R8_RAW_SHA256
    return module


def _flat_product_body(response, field, output, pixels, block_size, owners):
    N = field.shape[0]
    chunk = (N // pixels) * block_size
    task_count = (N + chunk - 1) // chunk
    for task_id in _NUMBA_PRANGE(task_count):
        owner_id = _NUMBA_GET_THREAD_ID()
        if owner_id < owners.shape[0]:
            owners[owner_id] = 1
        start = task_id * chunk
        stop = min(start + chunk, N)
        for flat in range(start, stop):
            scale = response[flat]
            value = field[flat]
            # Keep the R8 signed-zero and nonzero-underflow handling byte-for-
            # byte at the scalar-expression level.
            real = scale * value.real
            imag = scale * value.imag
            if scale == 0.0 or value.real == 0.0:
                real = real - 0.0 * value.imag
            if scale == 0.0 or value.imag == 0.0:
                imag = imag + 0.0 * value.real
            output[flat] = complex(real, imag)


def _flat_derivative_body(spectrum, multiplier, output, pixels, block_size, owners):
    N = spectrum.shape[0]
    chunk = (N // pixels) * block_size
    task_count = (N + chunk - 1) // chunk
    for task_id in _NUMBA_PRANGE(task_count):
        owner_id = _NUMBA_GET_THREAD_ID()
        if owner_id < owners.shape[0]:
            owners[owner_id] = 1
        start = task_id * chunk
        stop = min(start + chunk, N)
        for flat in range(start, stop):
            output[flat] = multiplier[flat // pixels] * spectrum[flat]


def _flat_rhs_body(product, derivative, output, prefactor, pixels, block_size, owners):
    N = product.shape[0]
    chunk = (N // pixels) * block_size
    task_count = (N + chunk - 1) // chunk
    for task_id in _NUMBA_PRANGE(task_count):
        owner_id = _NUMBA_GET_THREAD_ID()
        if owner_id < owners.shape[0]:
            owners[owner_id] = 1
        start = task_id * chunk
        stop = min(start + chunk, N)
        for flat in range(start, stop):
            first = 1j * prefactor * product[flat]
            second = prefactor * derivative[flat]
            output[flat] = first - second


def _flat_predictor_body(field, k1, output, dz, pixels, block_size, owners):
    N = field.shape[0]
    chunk = (N // pixels) * block_size
    task_count = (N + chunk - 1) // chunk
    for task_id in _NUMBA_PRANGE(task_count):
        owner_id = _NUMBA_GET_THREAD_ID()
        if owner_id < owners.shape[0]:
            owners[owner_id] = 1
        start = task_id * chunk
        stop = min(start + chunk, N)
        for flat in range(start, stop):
            increment = dz * k1[flat]
            output[flat] = field[flat] + increment


def _flat_corrector_body(field, k1, k2, output, dz, pixels, block_size, owners):
    N = field.shape[0]
    chunk = (N // pixels) * block_size
    task_count = (N + chunk - 1) // chunk
    for task_id in _NUMBA_PRANGE(task_count):
        owner_id = _NUMBA_GET_THREAD_ID()
        if owner_id < owners.shape[0]:
            owners[owner_id] = 1
        start = task_id * chunk
        stop = min(start + chunk, N)
        for flat in range(start, stop):
            slope_sum = k1[flat] + k2[flat]
            increment = 0.5 * dz * slope_sum
            output[flat] = field[flat] + increment


def _install_flat_bodies(module: ModuleType) -> None:
    for name, body in (
        ("_product_body", _flat_product_body),
        ("_derivative_body", _flat_derivative_body),
        ("_rhs_body", _flat_rhs_body),
        ("_predictor_body", _flat_predictor_body),
        ("_corrector_body", _flat_corrector_body),
    ):
        _clone_into(module, body, name)
    # A fresh module normally has no compiled bundle.  Reset explicitly so a
    # caller cannot accidentally reuse a dispatcher after reconfiguration.
    module.__dict__["_NUMBA_BUNDLE"] = None


def _bind_product_kernel(response, field, *, block_size=32, workers=1, runtime=None):
    response = _require_3d_c(response, name="response", dtype=np.dtype("float64"))
    field = _require_3d_c(field, name="field", dtype=np.dtype("complex128"))
    if response.shape != field.shape:
        raise ValueError("response and field shapes differ")
    slot = _S4D_OutputSlot()
    slot.value = response.astype(field.real.dtype, copy=False) * field
    return slot.value


def _bind_derivative_kernel(spectrum, multiplier, *, block_size=32, workers=1, runtime=None):
    spectrum = _require_3d_c(spectrum, name="spectrum", dtype=np.dtype("complex128"))
    multiplier = _require_1d_c(multiplier, name="multiplier", dtype=np.dtype("complex128"))
    if spectrum.shape[0] != multiplier.shape[0]:
        raise ValueError("multiplier length differs from spectrum time axis")
    slot = _S4D_OutputSlot()
    slot.value = multiplier[:, None, None] * spectrum
    return slot.value


def _bind_rhs_kernel(product, normalized_derivative, prefactor, *, block_size=32, workers=1, runtime=None):
    product = _require_3d_c(product, name="product", dtype=np.dtype("complex128"))
    normalized_derivative = _require_3d_c(
        normalized_derivative, name="normalized_derivative", dtype=np.dtype("complex128")
    )
    if product.shape != normalized_derivative.shape:
        raise ValueError("product and normalized_derivative shapes differ")
    slot = _S4D_OutputSlot()
    slot.value = (1j * float(prefactor) * product - float(prefactor) * normalized_derivative)
    return slot.value


def _bind_predictor_kernel(field, k1, dz, *, block_size=32, workers=1, runtime=None):
    field = _require_3d_c(field, name="field", dtype=np.dtype("complex128"))
    k1 = _require_3d_c(k1, name="k1", dtype=np.dtype("complex128"))
    if field.shape != k1.shape:
        raise ValueError("field and k1 shapes differ")
    slot = _S4D_OutputSlot()
    slot.value = (field + float(dz) * k1).astype(field.dtype, copy=False)
    return slot.value


def _bind_corrector_kernel(field, k1, k2, dz, *, block_size=32, workers=1, runtime=None):
    field = _require_3d_c(field, name="field", dtype=np.dtype("complex128"))
    k1 = _require_3d_c(k1, name="k1", dtype=np.dtype("complex128"))
    k2 = _require_3d_c(k2, name="k2", dtype=np.dtype("complex128"))
    if field.shape != k1.shape or field.shape != k2.shape:
        raise ValueError("field and Heun RHS shapes differ")
    slot = _S4D_OutputSlot()
    slot.value = (field + 0.5 * float(dz) * (k1 + k2)).astype(field.dtype, copy=False)
    return slot.value


def _bind_make_product(response, field, rdtype, *, block_size=32, workers=1, runtime=None, observe):
    response_cast = response.astype(rdtype, copy=False)
    output = product_kernel(response_cast, field, block_size=block_size, workers=workers, runtime=runtime)
    observe("R_PRODUCT", output)
    return output


def _bind_make_normalized_derivative(multiplier, product, xp, *, block_size=32, workers=1, runtime=None, observe):
    fft = xp.fft
    spectrum = fft.fft(product, axis=0)
    scaled = derivative_kernel(spectrum, multiplier, block_size=block_size, workers=workers, runtime=runtime)
    normalized = fft.ifft(scaled, axis=0)
    observe("R_DERIVATIVE", normalized)
    return normalized


def _bind_make_rhs(product, normalized_derivative, prefactor, ctype, *, block_size=32, workers=1, runtime=None, observe):
    output = rhs_kernel(product, normalized_derivative, prefactor, block_size=block_size, workers=workers, runtime=runtime)
    observe("R_RHS", output)
    return output.astype(ctype, copy=False)


def _bind_make_predictor(field, k1, dz, *, block_size=32, workers=1, runtime=None, observe):
    output = predictor_kernel(field, k1, dz, block_size=block_size, workers=workers, runtime=runtime)
    observe("H_PREDICTOR", output)
    return output


def _bind_make_corrector(field, k1, k2, dz, *, block_size=32, workers=1, runtime=None, observe):
    output = corrector_kernel(field, k1, k2, dz, block_size=block_size, workers=workers, runtime=runtime)
    observe("H_CORRECTOR", output)
    return output


def _bind_runtime_view():
    names = ("product", "derivative", "rhs", "predictor", "corrector")
    metadata = {
        name: {
            "serial_signatures": [],
            "parallel_signatures": [],
            "parallel_diagnostics": "NUMPY_NOT_COMPILED",
        }
        for name in names
    }
    return {
        "numba": None,
        "kernels": {},
        "executions": [],
        "execution_limit": 256,
        "kernel_metadata": metadata,
        "compiled": False,
        "binding_mode": "I_BIND_NUMPY_EXPRESSION",
    }


def _install_i_bind(module: ModuleType) -> None:
    module.__dict__["_S4D_OutputSlot"] = OutputSlot
    for name, helper in (
        ("product_kernel", _bind_product_kernel),
        ("derivative_kernel", _bind_derivative_kernel),
        ("rhs_kernel", _bind_rhs_kernel),
        ("predictor_kernel", _bind_predictor_kernel),
        ("corrector_kernel", _bind_corrector_kernel),
        ("_make_product", _bind_make_product),
        ("_make_normalized_derivative", _bind_make_normalized_derivative),
        ("_make_rhs", _bind_make_rhs),
        ("_make_predictor", _bind_make_predictor),
        ("_make_corrector", _bind_make_corrector),
        ("_new_runtime_view", _bind_runtime_view),
    ):
        _clone_into(module, helper, name)


def _variant_identity(
    name: str,
    path: Path,
    module: ModuleType,
    source_hash: str,
) -> dict[str, Any]:
    return {
        "schema": VARIANT_SCHEMA,
        "variant": name,
        "r8_path": str(path),
        "r8_raw_sha256": R8_RAW_SHA256,
        "r8_size_bytes": R8_SIZE_BYTES,
        "variant_source_raw_sha256": source_hash,
        "private_module": module.__name__,
        "source_identity_role": "frozen_R8_preserved_separately",
    }


def load_variant(name: str, frozen_r8_path: str | Path | None = None) -> ModuleType:
    """Load one S4-3D variant from the verified R8 source.

    The private module name is stable for a fixed variant/R8/source revision,
    so a valid disk-cache identity can be reused across processes.  No
    dispatcher or mutable R8 globals are shared with ``cpu_s4_3c``.
    """
    normalized = str(name).upper()
    if normalized not in _VARIANTS:
        raise ValueError(f"unknown S4-3D variant {name!r}; choose {sorted(_VARIANTS)}")
    path, digest = _check_r8(Path(frozen_r8_path) if frozen_r8_path is not None else _default_r8_path())
    source_hash = _variant_source_raw_sha256()
    key = (normalized, str(path), source_hash, digest)
    cached = _MODULE_CACHE.get(key)
    if cached is not None:
        return cached
    module = _load_r8(path, normalized, source_hash, digest)
    if normalized == "C_FLAT_EQWORK":
        _install_flat_bodies(module)
    elif normalized == "I_BIND":
        _install_i_bind(module)
    identity = _variant_identity(normalized, path, module, source_hash)
    identity["r8_raw_sha256"] = digest
    module.__dict__["__cpu_s4_3d_variant__"] = normalized
    module.__dict__["__cpu_s4_3d_variant_identity__"] = identity
    module.__dict__["__cpu_s4_3d_management_mode__"] = (
        "NUMPY_NOT_COMPILED" if normalized == "I_BIND" else "R8_NUMBA_PRIVATE_BUNDLE"
    )
    if normalized == "I_BIND":
        module.__dict__["__cpu_s4_3d_management_matched__"] = False
        module.__dict__["__cpu_s4_3d_binding_path_reused__"] = True
    _MODULE_CACHE[key] = module
    return module


def _decorate_factory(
    base_factory: Callable[..., Any],
    module: ModuleType,
    name: str,
    path: Path,
    source_hash: str,
):
    identity = _variant_identity(name, path, module, source_hash)

    def factory(frozen_propagator):
        propagator, dispatch = base_factory(frozen_propagator)
        state = getattr(base_factory, "last_state", None)
        if state is not None:
            execution = dict(getattr(state, "execution", {}))
            execution.update({
                "variant": name,
                "variant_identity": dict(identity),
                "source_identity": dict(getattr(state, "source_identity", {})),
                "variant_module": module.__name__,
            })
            if name == "I_BIND":
                execution.update({
                    "management_matched": False,
                    "management_mode": "MATCHED_BINDING_CONTROL_UNAVAILABLE",
                    "binding_path_reused": True,
                    "per_op_numba_dispatch": False,
                    "numba_compiled": False,
                    "parallel_diagnostics": "NUMPY_NOT_COMPILED",
                    "actual_worker_ids": "UNOBSERVED_NOT_NUMBA",
                })
            state.execution = execution
            state.variant_identity = dict(identity)
            state.variant_name = name
        for target in (propagator, dispatch):
            setattr(target, "cpu_s4_3d_variant", name)
            setattr(target, "cpu_s4_3d_variant_identity", dict(identity))
            setattr(target, "cpu_s4_3d_variant_module", module)
        if state is not None:
            setattr(dispatch, "cpu_s4_3d_state", state)
        factory.last_propagator = propagator
        factory.last_dispatch = dispatch
        factory.last_state = state
        factory.last_variant_identity = dict(identity)
        return propagator, dispatch

    for attr in (
        "candidate_family", "candidate_schema", "last_propagator",
        "last_dispatch", "last_state", "last_candidate", "last_source_identity",
    ):
        if hasattr(base_factory, attr):
            setattr(factory, attr, getattr(base_factory, attr))
    factory.variant = name
    factory.variant_module = module
    factory.variant_identity = dict(identity)
    factory.frozen_r8_path = str(path)
    factory.frozen_r8_raw_sha256 = R8_RAW_SHA256
    return factory


def make_factory(
    variant: str = "C_FLAT_EQWORK",
    *,
    frozen_r8_path: str | Path | None = None,
    **factory_args: Any,
):
    """Build a factory for one private S4-3D variant.

    The arguments after ``variant`` are passed unchanged to the frozen R8
    ``make_candidate_factory``.  Defaults are only convenience values for
    lightweight construction; callers performing a measurement should provide
    the pre-registered worker and block settings explicitly.
    """
    module = load_variant(variant, frozen_r8_path=frozen_r8_path)
    path, _ = _check_r8(Path(frozen_r8_path) if frozen_r8_path is not None else _default_r8_path())
    source_hash = _variant_source_raw_sha256()
    defaults = {
        "kernel_threads": 24,
        "temporal_workers": 24,
        "spatial_workers": 24,
        "raman_workers": 1,
        "block_size": 32,
    }
    defaults.update(factory_args)
    base_factory = module.make_candidate_factory(**defaults)
    return _decorate_factory(base_factory, module, str(variant).upper(), path, source_hash)


__all__ = ["OutputSlot", "R8_RAW_SHA256", "load_variant", "make_factory"]
