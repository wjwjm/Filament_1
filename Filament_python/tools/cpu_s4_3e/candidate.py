"""CPU-S4-3E private Raman Heun handoff.

The implementation is intentionally an adapter around the verified S4-3C
candidate.  The frozen Raman expressions, FFT dispatch, result tree, and
public stage/apply checks remain in the R8 module.  This layer adds a private
complete-call facade with two policies:

``C_STRICT``
    Uses the public R8 stage and apply entry points, including their complete
    binding seals.

``C_INTERNAL``
    Creates stage one and consumes it in one synchronous lexical call.  E and
    Raman parameter identity tokens are retained at the three original
    boundaries; only the rhs/q payload and predictor content seals are
    replaced by the private producer/consumer lifecycle.  No external stage,
    trust flag, object id, or read-only flag grants this path access.

The adapter never modifies ``raman.py`` or ``propagate.py``.  The returned
propagator is a source-bound clone whose Raman-only branch calls the complete
facade, while the complete Eq.27 branch and all other propagation code remain
source code clones of the frozen entry.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
import copy
import functools
import importlib
import importlib.util
import inspect
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import FunctionType, MethodType, ModuleType
from typing import Any, Callable, Mapping

import numpy as np

from tools.hpc_ops import provenance_v2 as _provenance_v2


R8_RAW_SHA256 = "252fc1e6521812c29872d5b848e4082841f0b8ae0ad15661394f6b31347e0a32"
R8_SIZE_BYTES = 59808
RAMAN_RELATIVE_PATH = "Filament_python/KHz_filament/raman.py"
RAMAN_SOURCE_BLOB = "5bb2702c109103ff882ee3642fedc748040c41b8"
RAMAN_CANONICAL_LF_SHA256 = (
    "bc211fcde7ac1ed1dd3891ce20abd46201903ebed4eb9859d9ff55bdcc5a6887"
)
PROPAGATE_RELATIVE_PATH = "Filament_python/KHz_filament/propagate.py"
PROPAGATE_SOURCE_BLOB = "f156111398415c2b92e4609ccc1da075fcf5d1e3"
PROPAGATE_CANONICAL_LF_SHA256 = (
    "909fbcf5ef05d2c50b8ed27506b450d40127b3973709d9b4b6a961799dc8d1f6"
)
CANDIDATE_SCHEMA = "filament.cpu_s4_3e.trusted_internal_handoff.v1"
TRANSFORMATION_IDENTITY = "cpu_s4_3e_p0_private_raman_heun_handoff_v1"
STAGE_NAME = "isaacs_raman_stage"
APPLY_NAME = "apply_isaacs_raman_operator_step"
_VARIANTS = frozenset({"C_STRICT", "C_INTERNAL"})
_HASH_ROLE_ORDER = (
    "E_stage1",
    "Omega_stage1",
    "rhs1_stage1_producer",
    "q1_stage1_producer",
    "E_apply",
    "Omega_apply",
    "rhs1_apply_consumer",
    "q1_apply_consumer",
    "predictor_stage2",
    "Omega_stage2",
)
_INTERNAL_HASH_ROLE_ORDER = (
    "E_stage1",
    "Omega_stage1",
    "E_apply",
    "Omega_apply",
    "Omega_stage2",
)
_STAGE_KWARG_NAMES = (
    "Omega",
    "dt",
    "omega0",
    "n0",
    "n_R",
    "omega_R",
    "Gamma_R",
    "method",
    "chunk_pixels",
    "iir_sampling",
)
_R8_NUMBA_BODY_NAMES = (
    "_product_body",
    "_derivative_body",
    "_rhs_body",
    "_predictor_body",
    "_corrector_body",
)
_R8_MODULE_CACHE: dict[tuple[str, str], ModuleType] = {}


class SourceBindingError(RuntimeError):
    """A frozen source or AST handoff anchor has drifted."""


class CandidateUnsupported(RuntimeError):
    """A call is outside the private handoff support contract."""


class HandoffRejected(RuntimeError):
    """The complete-call guard rejected a lifecycle or routing condition."""


class StageBindingError(ValueError):
    """An input identity or externally supplied stage is not valid."""


class _RouteToStrict(RuntimeError):
    """Internal pre-science route request."""


class _CallContext:
    __slots__ = (
        "owner",
        "generation",
        "status",
        "field",
        "kwargs",
        "field_token",
        "parameter_token",
        "stage1",
        "started_science",
        "failure",
        "route",
        "predictor",
        "stage2_consumed",
    )

    def __init__(
        self,
        owner: Any,
        generation: int,
        field: np.ndarray,
        kwargs: Mapping[str, Any],
    ):
        self.owner = owner
        self.generation = int(generation)
        self.status = "CREATED"
        self.field = field
        self.kwargs = dict(kwargs)
        self.field_token = None
        self.parameter_token = None
        self.stage1 = None
        self.started_science = False
        self.failure = None
        self.route = None
        self.predictor = None
        self.stage2_consumed = False


def _default_r8_path() -> Path:
    return (Path(__file__).resolve().parent.parent / "cpu_s4_3c" / "candidate.py").resolve()


def _check_r8(path: Path) -> tuple[Path, str]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen R8 candidate is missing: {path}")
    size = int(path.stat().st_size)
    if size != R8_SIZE_BYTES:
        raise SourceBindingError(f"frozen R8 size drift: {size} != {R8_SIZE_BYTES}")
    digest = _provenance_v2.raw_sha256_file(path)
    if digest != R8_RAW_SHA256:
        raise SourceBindingError(f"frozen R8 raw SHA256 drift: {digest} != {R8_RAW_SHA256}")
    return path, digest


def _clone_into(module: ModuleType, function: Callable[..., Any], public_name: str) -> Callable[..., Any]:
    """Clone an R8 compiled helper with a stable Numba cache identity."""
    token = str(module.__dict__["__cpu_s4e_identity_token__"])
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


def _load_r8(path: Path) -> ModuleType:
    """Load R8 under an identity stable across adapter revisions/processes."""
    path, digest = _check_r8(path)
    key = (str(path), digest)
    cached = _R8_MODULE_CACHE.get(key)
    if cached is not None:
        return cached
    token = f"r8_{digest[:16]}"
    name = f"_filament_cpu_s4e_r8_{digest[:16]}"
    existing = sys.modules.get(name)
    if isinstance(existing, ModuleType):
        _R8_MODULE_CACHE[key] = existing
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SourceBindingError(f"unable to create frozen R8 import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__cpu_s4_3e_private__"] = True
    module.__dict__["__cpu_s4e_identity_token__"] = token
    module.__dict__["__cpu_s4_3e_r8_path__"] = str(path)
    module.__dict__["__cpu_s4_3e_r8_raw_sha256__"] = digest
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    for body_name in _R8_NUMBA_BODY_NAMES:
        body = module.__dict__.get(body_name)
        if not callable(body):
            sys.modules.pop(name, None)
            raise SourceBindingError(f"R8 compiled helper is unavailable: {body_name}")
        _clone_into(module, body, body_name)
    _R8_MODULE_CACHE[key] = module
    return module


def _git_root(path: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise SourceBindingError(
            f"source Git root lookup failed ({result.returncode}): {result.stderr.strip()}"
        )
    return Path(result.stdout.strip()).resolve()


def _tracked_source_record(path: Path, expected_blob: str, expected_lf: str) -> dict[str, Any]:
    root = _git_root(path)
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise SourceBindingError(f"source is outside its Git checkout: {path}") from exc
    raw = path.read_bytes()
    record = {
        "path": relative,
        "classification": "tracked_text",
        "hash_scope": _provenance_v2.TRACKED_HASH_SCOPE,
        "git_blob_oid": expected_blob,
        "canonical_lf_sha256": expected_lf,
    }
    try:
        _provenance_v2.validate_record(root, record, require_hash_scope=True)
    except Exception as exc:
        raise SourceBindingError(f"source provenance validation failed: {exc}") from exc
    return {
        "path": relative,
        "classification": "tracked_text",
        "hash_scope": _provenance_v2.TRACKED_HASH_SCOPE,
        "git_blob_oid": expected_blob,
        "canonical_lf_sha256": _provenance_v2.canonical_lf_sha256(raw),
        "raw_sha256": _provenance_v2.raw_sha256(raw),
        "size_bytes": int(len(raw)),
    }


def source_identity(frozen_r8_path: str | Path | None = None) -> dict[str, Any]:
    """Return verified identities for the frozen Raman and propagation sources."""
    r8_path = Path(frozen_r8_path) if frozen_r8_path is not None else _default_r8_path()
    r8_path, r8_digest = _check_r8(r8_path)
    r8 = _load_r8(r8_path)
    raman_function = r8._source_function()
    raman_path = r8._source_path(raman_function)
    raman = r8.source_identity(raman_function)
    # S4-3C historically exposed the tracked Git blob as ``git_blob`` while
    # the S4-3E provenance schema uses the explicit ``*_oid`` spelling. Keep
    # both names in this adapter record without changing the frozen source
    # identity implementation.
    if "git_blob_oid" not in raman:
        raman["git_blob_oid"] = raman.get("git_blob")
    propagate = importlib.import_module("KHz_filament.propagate").propagate_one_pulse
    propagate_path = Path(inspect.getsourcefile(propagate) or propagate.__code__.co_filename).resolve()
    propagate_record = _tracked_source_record(
        propagate_path, PROPAGATE_SOURCE_BLOB, PROPAGATE_CANONICAL_LF_SHA256
    )
    if raman.get("relative_path") != RAMAN_RELATIVE_PATH:
        raise SourceBindingError("Raman source relative path drifted")
    if raman.get("git_blob") != RAMAN_SOURCE_BLOB:
        raise SourceBindingError("Raman source Git blob drifted")
    return {
        "schema": CANDIDATE_SCHEMA,
        "transformation": TRANSFORMATION_IDENTITY,
        "raman": dict(raman),
        "propagate": propagate_record,
        "r8": {
            "path": str(r8_path),
            "hash_scope": "raw_bytes",
            "raw_sha256": r8_digest,
            "size_bytes": R8_SIZE_BYTES,
        },
    }


def _clone_function_with_globals(
    function: Callable[..., Any], globals_dict: dict[str, Any], name: str | None = None
) -> Callable[..., Any]:
    public_name = str(name or function.__name__)
    clone = FunctionType(
        function.__code__,
        globals_dict,
        public_name,
        function.__defaults__,
        function.__closure__,
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = getattr(function, "__annotations__", {})
    clone.__module__ = function.__module__
    clone.__qualname__ = function.__qualname__
    return clone


def _local_scalar_token(
    value: Any, *, hash_array: Callable[[np.ndarray, str], str], role: str
) -> Any:
    if isinstance(value, np.ndarray):
        return (
            "ndarray",
            value.dtype.str,
            tuple(int(s) for s in value.shape),
            hash_array(value, role),
        )
    return (type(value).__name__, repr(value))


def _stage_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Select exactly the parameters accepted by the Raman stage function."""
    return {name: kwargs[name] for name in _STAGE_KWARG_NAMES if name in kwargs}


def _validate_factory_value(name: str, value: Any, allowed: frozenset[int]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in allowed:
        choices = "/".join(str(item) for item in sorted(allowed))
        raise ValueError(f"{name} must be one of {choices}")
    return int(value)


class _HandoffState:
    """Per-factory complete-call owner with scalar-only public metrics."""

    def __init__(
        self,
        *,
        variant: str,
        r8_state: Any,
        source: dict[str, Any],
        block_size: int,
        raman_workers: int,
        observe: Callable[[str, Any], None] | None,
    ) -> None:
        self.variant = str(variant)
        self.r8_state = r8_state
        self.source = copy.deepcopy(source)
        self.block_size = int(block_size)
        self.raman_workers = int(raman_workers)
        self.observe_callback = observe
        # R8 keeps the transformed scientific functions in the private Raman
        # namespace, while CandidateState validation/hash helpers live in the
        # private candidate module namespace.  They must remain separate.
        self._raman_namespace = r8_state.candidate_stage_inner.__globals__
        # CandidateState methods normally share the imported R8 module globals.
        # Capture that dictionary for the internal helpers, then install
        # per-state strict method clones below so strict hash accounting never
        # patches a module-global function shared by another factory.
        self._candidate_namespace = type(r8_state).stage.__globals__
        self._control_thread_id = int(threading.main_thread().ident or -1)
        self._lock = threading.Lock()
        self._local = threading.local()
        self._generation = 0
        self._hash_records: list[dict[str, Any]] = []
        # Keep the original runtime alive and bind every internal call to the
        # same object.  Replacing the R8 runtime after factory construction
        # must invalidate this handoff instead of silently reusing stale
        # helper closures.
        self._bound_runtime = r8_state._runtime
        self._runtime_generation = id(self._bound_runtime)
        self._bound_runtime_configuration = self._runtime_configuration()
        self._bound_active_layer = self._active_threading_layer()
        execution = getattr(r8_state, "execution", {})
        self._control_mask_threads = int(execution.get("kernel_threads", 24))
        self._stage2_capability = object()
        self._strict_namespace = self._install_strict_wrappers()
        self._stats: dict[str, Any] = {
            "schema": CANDIDATE_SCHEMA,
            "variant": self.variant,
            "candidate_family": "RAMAN_TRUSTED_INTERNAL_HANDOFF_V1",
            "state_machine": "CREATED->READY->CONSUMING->CONSUMED / FAILED / INVALIDATED",
            "candidate_full_calls": 0,
            "strict_full_calls": 0,
            "internal_full_calls": 0,
            "strict_route_calls": 0,
            "failed_full_calls": 0,
            "hash_leaf_calls": 0,
            "hash_leaf_logical_bytes": 0,
            "last_hash_leaf_calls": 0,
            "last_hash_leaf_logical_bytes": 0,
            "last_hash_leaf_role_counts": {},
            "hash_leaf_role_counts": {},
            "remaining_hash_calls_target": 10 if self.variant == "C_STRICT" else 5,
            "remaining_hash_logical_bytes_target": (
                3_247_298_352 if self.variant == "C_STRICT" else 1_298_248_704
            ),
            "operation_counts": {
                "R_PRODUCT": 0,
                "R_DERIVATIVE": 0,
                "R_RHS": 0,
                "H_PREDICTOR": 0,
                "H_CORRECTOR": 0,
            },
            "fallback_reasons": {},
            "last_call": {},
        }
        self._internal_stage_inner, self._internal_apply_inner = self._build_internal_clones()

    @property
    def r8_module(self) -> Any:
        return self.r8_state.candidate_stage_inner.__globals__.get("__name__")

    def _runtime_configuration(self) -> tuple[Any, ...]:
        """Bind pool, target, threading and cache settings without array scans."""
        numba = importlib.import_module('numba')
        return (
            numba.config.NUMBA_NUM_THREADS,
            numba.config.CPU_NAME,
            numba.config.THREADING_LAYER,
            numba.config.CACHE_DIR,
            numba.config.CPU_FEATURES,
            str(numba.config.OPT),
            bool(numba.config.DISABLE_JIT),
            tuple(numba.config.THREADING_LAYER_PRIORITY),
            *(os.environ.get(name) for name in (
                'NUMBA_NUM_THREADS', 'NUMBA_CPU_NAME', 'NUMBA_THREADING_LAYER',
                'NUMBA_CACHE_DIR',
            )),
        )

    @staticmethod
    def _active_threading_layer() -> str | None:
        try:
            return str(importlib.import_module('numba').threading_layer())
        except ValueError:
            return None

    def _assert_runtime_binding(self) -> None:
        if self.r8_state._runtime is not self._bound_runtime:
            raise HandoffRejected('RUNTIME_BINDING_CHANGED')
        if self._runtime_configuration() != self._bound_runtime_configuration:
            raise HandoffRejected('RUNTIME_CONFIGURATION_CHANGED')
        active = self._active_threading_layer()
        if self._bound_active_layer is None and active is not None:
            requested = self._bound_runtime_configuration[2]
            if requested != 'default' and active != requested:
                raise HandoffRejected('RUNTIME_CONFIGURATION_CHANGED')
            self._bound_active_layer = active
        elif active != self._bound_active_layer:
            raise HandoffRejected('RUNTIME_CONFIGURATION_CHANGED')

    def _install_strict_wrappers(self) -> dict[str, Any]:
        """Clone R8 binding methods with a factory-local hash recorder."""
        namespace = dict(self._candidate_namespace)

        def counted(value: np.ndarray) -> str:
            original = self._candidate_namespace.get("_array_raw_sha256")
            if not callable(original):
                raise SourceBindingError("R8 array hash helper is unavailable")
            digest = str(original(value))
            records = getattr(self._local, "strict_hash_records", None)
            if records is not None:
                records.append(
                    {
                        "logical_bytes": int(value.nbytes),
                        "shape": [int(s) for s in value.shape],
                        "dtype": value.dtype.str,
                    }
                )
            return digest

        namespace["_array_raw_sha256"] = counted
        clone_names = (
            "_array_content_token",
            "_scalar_token",
            "_parameter_token",
            "_field_token",
            "_stage_payload_token",
            "_stage_binding",
            "_validate_stage_payload",
            "_support_reason",
        )
        for name in clone_names:
            original = self._candidate_namespace.get(name)
            if not callable(original):
                raise SourceBindingError(f"R8 strict helper missing: {name}")
            namespace[name] = _clone_function_with_globals(original, namespace, name)
        # CandidateState methods live on the frozen class, not in its module
        # globals.  Keep the contextmanager-decorated _entry bound to the
        # frozen implementation; clone the ordinary methods with this
        # factory-local globals dictionary.
        for name in ("_stage_dispatch", "_apply_dispatch", "stage", "apply"):
            original = getattr(type(self.r8_state), name, None)
            if not callable(original):
                raise SourceBindingError(f"R8 strict method missing: {name}")
            namespace[name] = _clone_function_with_globals(original, namespace, name)
            setattr(self.r8_state, name, MethodType(namespace[name], self.r8_state))
        self._rebind_public_surfaces()
        return namespace

    def _rebind_public_surfaces(self) -> None:
        """Keep every private lookup site on this state's strict wrappers."""
        stage = self.r8_state.stage
        apply = self.r8_state.apply
        self._raman_namespace[STAGE_NAME] = stage
        self._raman_namespace[APPLY_NAME] = apply
        dispatch = getattr(self.r8_state, "dispatch", None)
        propagator = getattr(self.r8_state, "propagator", None)
        if dispatch is not None:
            dispatch.cpu_s4_3c_stage = stage
            dispatch.cpu_s4_3c_apply = apply
            dispatch.cpu_s4_3c_dispatch = {
                "stage": stage,
                "apply": apply,
                "kernels": getattr(self.r8_state, "kernels", {}),
                "events": getattr(self.r8_state, "events", []),
            }
        if propagator is not None:
            propagator.__globals__[STAGE_NAME] = stage
            propagator.__globals__[APPLY_NAME] = apply

    def _build_internal_clones(self) -> tuple[Callable[..., Any], Callable[..., Any]]:
        """Clone transformed R8 stage/apply with a private no-observer namespace."""
        internal_globals = dict(self._raman_namespace)
        no_observer = lambda _name, _value: None
        helper_specs = {
            "_s43c_product": ("_make_product", "product"),
            "_s43c_normalized_derivative": ("_make_normalized_derivative", "derivative"),
            "_s43c_rhs": ("_make_rhs", "rhs"),
            "_s43c_predictor": ("_make_predictor", "predictor"),
            "_s43c_corrector": ("_make_corrector", "corrector"),
        }
        for public_name, (helper_name, _kernel_name) in helper_specs.items():
            helper = self._candidate_namespace.get(helper_name)
            if not callable(helper):
                raise SourceBindingError(f"R8 private helper missing: {helper_name}")
            fixed = {
                "block_size": self.block_size,
                "workers": self.raman_workers,
                "runtime": self.r8_state._runtime,
                "observe": no_observer,
            }
            producer = functools.partial(helper, **fixed)
            if public_name == "_s43c_predictor":
                # The predictor is a private producer-consumer handoff.  Its
                # identity is registered only after the frozen predictor
                # helper returns while the active call owns the stage2
                # capability; a caller supplying an arbitrary array cannot
                # satisfy this binding.
                def produce_predictor(
                    field: Any,
                    k1: Any,
                    dz: Any,
                    *,
                    _producer: Callable[..., Any] = producer,
                ) -> Any:
                    context = self._assert_active_context(
                        getattr(self._local, "context", None),
                        allowed_states=("CONSUMING",),
                        operation="predictor",
                        require_stage2_window=True,
                    )
                    if context.predictor is not None:
                        raise HandoffRejected("PREDICTOR_ALREADY_REGISTERED")
                    output = _producer(field, k1, dz)
                    self._assert_active_context(
                        context,
                        allowed_states=("CONSUMING",),
                        operation="predictor",
                        require_stage2_window=True,
                    )
                    context.predictor = output
                    return output

                internal_globals[public_name] = produce_predictor
            else:
                internal_globals[public_name] = producer
        stage_inner = _clone_function_with_globals(
            self.r8_state.candidate_stage_inner, internal_globals, "_s43e_stage_inner"
        )
        apply_inner = _clone_function_with_globals(
            self.r8_state.candidate_apply_inner, internal_globals, "_s43e_apply_inner"
        )
        internal_globals[STAGE_NAME] = self._stage2_dispatch
        return stage_inner, apply_inner

    def _record_hash(self, value: np.ndarray, role: str) -> str:
        if not isinstance(value, np.ndarray):
            raise TypeError("hash boundary requires a NumPy ndarray")
        hasher = self._candidate_namespace.get("_array_raw_sha256")
        if not callable(hasher):
            raise SourceBindingError("R8 array hash helper is unavailable")
        digest = str(hasher(value))
        self._hash_records.append(
            {
                "role": str(role),
                "logical_bytes": int(value.nbytes),
                "shape": [int(s) for s in value.shape],
                "dtype": value.dtype.str,
            }
        )
        return digest

    def _parameter_token(self, kwargs: Mapping[str, Any]) -> tuple[Any, ...]:
        names = ("Omega", "dt", "omega0", "n0", "n_R", "omega_R", "Gamma_R")
        tokens = [
            (
                name,
                _local_scalar_token(
                    kwargs.get(name), hash_array=self._record_hash, role=f"Omega_{name}"
                ),
            )
            for name in names
        ]
        method = str(kwargs.get("method", "iir")).lower()
        sampling = str(kwargs.get("iir_sampling", "exact_piecewise_linear")).lower()
        tokens.extend((("method", method), ("iir_sampling", sampling)))
        if method == "iir":
            chunk_token: Any = ("ignored_for_iir",)
        else:
            chunk_token = _local_scalar_token(
                kwargs.get("chunk_pixels", 65536),
                hash_array=self._record_hash,
                role="chunk_pixels",
            )
        tokens.append(("chunk_pixels", chunk_token))
        return tuple(tokens)

    def _field_token(self, field: np.ndarray, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "shape": [int(s) for s in field.shape],
            "dtype": field.dtype.str,
            "strides_bytes": [int(s) for s in field.strides],
            "content_raw_sha256": self._record_hash(field, "E"),
            "data_pointer": int(field.__array_interface__["data"][0]),
            "object_id": int(id(field)),
            "parameter_token": self._parameter_token(kwargs),
        }

    def _support_reason(self, field: Any, kwargs: Mapping[str, Any], *, apply: bool) -> str | None:
        support = self._candidate_namespace.get("_support_reason")
        if not callable(support):
            raise SourceBindingError("R8 support gate is unavailable")
        return support(field, kwargs, raman_namespace=self._raman_namespace, apply=apply)

    def _validate_private_stage(self, field: np.ndarray, payload: Any, kwargs: Mapping[str, Any]) -> None:
        validator = self._candidate_namespace.get("_validate_stage_payload")
        if not callable(validator):
            raise SourceBindingError("R8 stage validator is unavailable")
        reason = validator(
            field,
            payload,
            kwargs,
            require_c=True,
            require_binding=False,
        )
        if reason is not None:
            raise CandidateUnsupported(str(reason))

    def _record_operations(self, path: str, field: Any, counts: Mapping[str, int]) -> None:
        recorder = getattr(self.r8_state, "_record_candidate", None)
        if callable(recorder):
            recorder(str(path), counts, field)
        for name, count in counts.items():
            self._stats["operation_counts"][name] += int(count)

    def _restore_control_thread_mask(self) -> None:
        """Restore the frozen control-thread mask after every complete call."""
        runtime = self._bound_runtime
        numba = runtime.get("numba") if isinstance(runtime, Mapping) else None
        if numba is None:
            return
        try:
            target = int(self._control_mask_threads)
            maximum = int(numba.config.NUMBA_NUM_THREADS)
            if target > maximum:
                self._stats.setdefault("mask_restore_failures", 0)
                self._stats["mask_restore_failures"] += 1
                return
            if int(numba.get_num_threads()) != target:
                numba.set_num_threads(target)
        except Exception:
            # Preserve the original science/validation exception.  The
            # failure remains visible in scalar metrics for the run receipt.
            self._stats.setdefault("mask_restore_failures", 0)
            self._stats["mask_restore_failures"] += 1

    def _assert_active_context(
        self,
        context: Any,
        *,
        allowed_states: tuple[str, ...],
        operation: str,
        require_stage2_window: bool = False,
    ) -> _CallContext:
        """Validate the private lifecycle owner before any internal work."""
        current_thread = int(threading.get_ident())
        if current_thread != self._control_thread_id:
            raise HandoffRejected("CONTROL_THREAD_MISMATCH")
        active = getattr(self._local, "context", None)
        if (
            not isinstance(context, _CallContext)
            or context.owner is not self
            or active is not context
            or context.generation != self._generation
        ):
            raise HandoffRejected("OWNER_OR_GENERATION_MISMATCH")
        self._assert_runtime_binding()
        if context.status not in allowed_states:
            raise HandoffRejected(
                f"INVALID_CONTEXT_STATE:{operation}:{context.status}"
            )
        if require_stage2_window and getattr(self._local, "stage2_capability", None) is not self._stage2_capability:
            raise HandoffRejected("OWNER_OR_GENERATION_MISMATCH")
        return context

    @contextmanager
    def _stage2_window(self, context: _CallContext):
        """Open the lexical stage2 capability only around the private apply."""
        self._assert_active_context(
            context,
            allowed_states=("CONSUMING",),
            operation="stage2_window",
        )
        previous = getattr(self._local, "stage2_capability", None)
        if previous is not None:
            raise HandoffRejected("REENTRANT_OR_CONCURRENT_ENTRY")
        self._local.stage2_capability = self._stage2_capability
        try:
            yield
        finally:
            self._local.stage2_capability = previous

    def _stage2_dispatch(self, field: Any, **kwargs: Any) -> Any:
        context = self._assert_active_context(
            getattr(self._local, "context", None),
            allowed_states=("CONSUMING",),
            operation="stage2",
            require_stage2_window=True,
        )
        if context.predictor is None or field is not context.predictor:
            raise HandoffRejected("PREDICTOR_BINDING_MISMATCH")
        if context.stage2_consumed:
            raise HandoffRejected("STAGE2_ALREADY_CONSUMED")
        context.stage2_consumed = True
        return self._internal_stage(context, field, kwargs, stage2=True)

    def _internal_stage(
        self,
        context: _CallContext,
        field: Any,
        kwargs: Mapping[str, Any],
        *,
        stage2: bool,
    ) -> Any:
        context = self._assert_active_context(
            context,
            allowed_states=("CONSUMING",) if stage2 else ("READY",),
            operation="stage2" if stage2 else "stage1",
            require_stage2_window=stage2,
        )
        reason = self._support_reason(field, kwargs, apply=False)
        if reason is not None:
            if stage2:
                raise HandoffRejected(f"OWNER_OR_GENERATION_MISMATCH:{reason}")
            raise _RouteToStrict(reason)
        if stage2:
            current_parameter = self._parameter_token(kwargs)
            if current_parameter != context.parameter_token:
                raise StageBindingError("external stage1 field or Raman parameter identity differs")
        else:
            context.field_token = self._field_token(field, kwargs)
            context.parameter_token = context.field_token["parameter_token"]
        context.started_science = True
        result = self._internal_stage_inner(field, **_stage_kwargs(kwargs))
        self._validate_private_stage(field, result, kwargs)
        self._record_operations(
            "stage2" if stage2 else "stage1",
            field,
            {
                "R_PRODUCT": 1,
                "R_DERIVATIVE": 1,
                "R_RHS": 1,
                "H_PREDICTOR": 0,
                "H_CORRECTOR": 0,
            },
        )
        if not stage2:
            context.stage1 = result
        return result

    def _internal_apply(self, context: _CallContext, field: Any, dz: Any, kwargs: Mapping[str, Any]) -> Any:
        context = self._assert_active_context(
            context,
            allowed_states=("READY",),
            operation="apply",
        )
        reason = self._support_reason(field, kwargs, apply=True)
        if reason is not None:
            raise HandoffRejected(f"OWNER_OR_GENERATION_MISMATCH:{reason}")
        current_field = self._field_token(field, kwargs)
        if current_field != context.field_token:
            raise StageBindingError("external stage1 field or Raman parameter identity differs")
        self._validate_private_stage(field, context.stage1, kwargs)
        context.status = "CONSUMING"
        context.started_science = True
        context.predictor = None
        context.stage2_consumed = False
        with self._stage2_window(context):
            result = self._internal_apply_inner(field, dz, stage1=context.stage1, **kwargs)
        self._assert_active_context(
            context,
            allowed_states=("CONSUMING",),
            operation="apply_complete",
        )
        if context.predictor is None or not context.stage2_consumed:
            raise HandoffRejected("STAGE2_PREDICTOR_NOT_CONSUMED")
        self._record_operations(
            "apply",
            field,
            {
                "R_PRODUCT": 0,
                "R_DERIVATIVE": 0,
                "R_RHS": 0,
                "H_PREDICTOR": 1,
                "H_CORRECTOR": 1,
            },
        )
        return result

    @contextmanager
    def _strict_hash_counter(self):
        """Expose a factory-local list to the cloned strict hash helper."""
        records: list[dict[str, Any]] = []
        self._local.strict_hash_records = records
        try:
            yield records
        finally:
            self._local.strict_hash_records = None

    def _set_hash_stats(self, records: list[dict[str, Any]], *, internal: bool) -> None:
        if internal:
            effective = list(self._hash_records)
            roles = list(_INTERNAL_HASH_ROLE_ORDER[: len(effective)])
            if len(effective) > len(_INTERNAL_HASH_ROLE_ORDER):
                roles.extend(
                    str(item.get("role", "unknown"))
                    for item in effective[len(_INTERNAL_HASH_ROLE_ORDER) :]
                )
        else:
            effective = list(records)
            roles = list(_HASH_ROLE_ORDER[: len(effective)])
        logical_bytes = int(sum(int(item["logical_bytes"]) for item in effective))
        self._stats["hash_leaf_calls"] += int(len(effective))
        self._stats["hash_leaf_logical_bytes"] += logical_bytes
        role_counts: dict[str, int] = {}
        for role in roles:
            role_counts[role] = int(role_counts.get(role, 0)) + 1
        self._stats["last_hash_leaf_calls"] = int(len(effective))
        self._stats["last_hash_leaf_logical_bytes"] = logical_bytes
        self._stats["last_hash_leaf_role_counts"] = role_counts
        for role, count in role_counts.items():
            self._stats["hash_leaf_role_counts"][role] = int(
                self._stats["hash_leaf_role_counts"].get(role, 0)
            ) + int(count)

    def _strict_full_heun(self, field: Any, dz: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        with self._strict_hash_counter() as records:
            stage1 = self.r8_state.stage(field, **_stage_kwargs(kwargs))
            result = self.r8_state.apply(field, dz, stage1=stage1, **kwargs)
        self._set_hash_stats(records, internal=False)
        return {"stage1": stage1, "result": result}

    def _validate_full_kwargs(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        forbidden = sorted(
            key
            for key in kwargs
            if key in {"stage1", "trusted", "skip_hash", "internal", "generation", "owner"}
        )
        if forbidden:
            raise TypeError(
                "CPU-S4-3E full Raman entry does not accept external trust controls: "
                + ", ".join(forbidden)
            )
        result = dict(kwargs)
        if result.get("return_response", True) is False or result.get("return_energy", True) is False:
            raise CandidateUnsupported("full_heun_requires_stage_response_and_energy")
        return result

    def full_heun(self, field: Any, dz: Any, **kwargs: Any) -> dict[str, Any]:
        """Run one complete Raman Heun call and return the original result tree."""
        full_kwargs = self._validate_full_kwargs(kwargs)
        current_thread = int(threading.get_ident())
        if current_thread != self._control_thread_id:
            raise HandoffRejected("CONTROL_THREAD_MISMATCH")
        if not self._lock.acquire(blocking=False):
            raise HandoffRejected("REENTRANT_OR_CONCURRENT_ENTRY")
        self._generation += 1
        self._hash_records = []
        context = _CallContext(self, self._generation, field, full_kwargs)
        self._local.context = context
        self._stats["candidate_full_calls"] += 1
        started = time.perf_counter()
        try:
            self._assert_runtime_binding()
            route_reason = None
            if self.variant == "C_INTERNAL":
                omega = full_kwargs.get("Omega")
                if self.observe_callback is not None or self.r8_state.observe_callback is not None:
                    route_reason = "observer_enabled"
                elif type(field) is not np.ndarray:
                    route_reason = "field_array_subclass_or_unknown_array"
                elif isinstance(omega, np.ndarray) and type(omega) is not np.ndarray:
                    route_reason = "omega_array_subclass_or_unknown_array"
                else:
                    route_reason = self._support_reason(field, full_kwargs, apply=True)
            if route_reason is not None:
                context.route = "strict"
                context.status = "READY"
                self._stats["strict_route_calls"] += 1
                self._stats["fallback_reasons"][route_reason] = int(
                    self._stats["fallback_reasons"].get(route_reason, 0)
                ) + 1
                bundle = self._strict_full_heun(field, dz, full_kwargs)
                self._stats["strict_full_calls"] += 1
                context.status = "CONSUMED"
                return bundle
            if self.variant == "C_STRICT":
                context.route = "strict"
                context.status = "READY"
                bundle = self._strict_full_heun(field, dz, full_kwargs)
                self._stats["strict_full_calls"] += 1
                context.status = "CONSUMED"
                return bundle
            context.status = "READY"
            context.route = "internal"
            self._internal_stage(context, field, full_kwargs, stage2=False)
            result = self._internal_apply(context, field, dz, full_kwargs)
            context.status = "CONSUMED"
            self._stats["internal_full_calls"] += 1
            self._set_hash_stats([], internal=True)
            return {"stage1": context.stage1, "result": result}
        except _RouteToStrict as exc:
            if context.started_science:
                context.status = "FAILED"
                context.failure = str(exc)
                raise HandoffRejected(str(exc)) from exc
            context.status = "READY"
            context.route = "strict"
            self._stats["strict_route_calls"] += 1
            reason = str(exc) or "private_boundary_unavailable"
            self._stats["fallback_reasons"][reason] = int(
                self._stats["fallback_reasons"].get(reason, 0)
            ) + 1
            bundle = self._strict_full_heun(field, dz, full_kwargs)
            self._stats["strict_full_calls"] += 1
            context.status = "CONSUMED"
            return bundle
        except BaseException as exc:
            context.status = "FAILED" if context.started_science else "INVALIDATED"
            context.failure = f"{type(exc).__name__}:{exc}"
            self._stats["failed_full_calls"] += 1
            raise
        finally:
            self._restore_control_thread_mask()
            self._stats["last_call"] = {
                "generation": int(context.generation),
                "status": str(context.status),
                "elapsed_s": float(time.perf_counter() - started),
                "started_science": bool(context.started_science),
                "failure": context.failure,
                "route": context.route,
            }
            self._local.context = None
            self._lock.release()

    def _propagate_substep(
        self,
        field: Any,
        step_length: Any,
        stage_kwargs: Mapping[str, Any],
        *,
        integrator: str,
        transverse_cell_area: float,
    ) -> dict[str, Any]:
        return self.full_heun(
            field,
            step_length,
            integrator=integrator,
            return_diagnostics=True,
            transverse_cell_area=transverse_cell_area,
            **dict(stage_kwargs),
        )

    def metrics(self) -> dict[str, Any]:
        result = copy.deepcopy(self._stats)
        result["source_identity"] = copy.deepcopy(self.source)
        result["r8_candidate_metrics"] = self.r8_state.metrics()
        result["control_thread_id"] = int(self._control_thread_id)
        result["runtime_generation"] = int(self._runtime_generation)
        result["observer_configured"] = bool(self.observe_callback is not None)
        result["internal_arrays_exposed"] = False
        result["last_full_heun_is_callable"] = True
        return result

    @property
    def stats(self) -> dict[str, Any]:
        return self.metrics()


def _transform_propagator_function(source_path: Path) -> Callable[..., Any]:
    """Clone only the Raman else branch of the anchored propagation source."""
    source = _provenance_v2.canonical_lf_bytes(source_path.read_bytes()).decode("utf-8")
    tree = ast.parse(source, filename=str(source_path))
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "propagate_one_pulse"
    ]
    if len(matches) != 1:
        raise SourceBindingError("propagate_one_pulse source anchor is not unique")
    function_node = copy.deepcopy(matches[0])
    nested = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.FunctionDef) and node.name == "_full_raman_substep"
    ]
    if len(nested) != 1:
        raise SourceBindingError("_full_raman_substep source anchor is not unique")
    branch = [
        node
        for node in nested[0].body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "full_isaacs_complete_mode"
    ]
    if len(branch) != 1 or not branch[0].orelse:
        raise SourceBindingError("Raman complete/rotational branch anchor changed")
    replacement = ast.parse(
        """
bundle = _cpu_s4_3e_full_raman_substep(
    field,
    step_length,
    stage_kwargs,
    integrator=getattr(raman_conf, 'operator_integrator', 'heun'),
    transverse_cell_area=axes.dx * axes.dy,
)
first_stage = bundle['stage1']
updated, diagnostics = bundle['result']
""",
        mode="exec",
    ).body
    branch[0].orelse = [ast.copy_location(node, branch[0].orelse[0]) for node in replacement]
    ast.fix_missing_locations(function_node)
    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)
    local_namespace: dict[str, Any] = {}
    exec(compile(module, str(source_path), "exec"), {}, local_namespace)
    result = local_namespace.get("propagate_one_pulse")
    if not callable(result):
        raise SourceBindingError("propagation AST clone did not produce propagate_one_pulse")
    return result


def _clone_bound_propagator(
    base: Callable[..., Any], handoff: _HandoffState
) -> tuple[Callable[..., Any], dict[str, Any]]:
    production = importlib.import_module("KHz_filament.propagate").propagate_one_pulse
    path = Path(inspect.getsourcefile(production) or production.__code__.co_filename).resolve()
    identity = _tracked_source_record(
        path, PROPAGATE_SOURCE_BLOB, PROPAGATE_CANONICAL_LF_SHA256
    )
    transformed = _transform_propagator_function(path)
    private = dict(base.__globals__)
    private["_cpu_s4_3e_full_raman_substep"] = handoff._propagate_substep
    clone = _clone_function_with_globals(transformed, private, "propagate_one_pulse")
    clone.__j1_inner_code__ = getattr(base, "__j1_inner_code__", base.__code__)
    for name in (
        "_cpu_o1_execution",
        "_cpu_o1_candidate_events",
        "_cpu_o1_rate_workspace",
        "_cpu_s4_3c_state",
        "_cpu_s4_3c_candidate",
        "_cpu_s4_3c_execution",
        "_cpu_s4_3c_source_identity",
    ):
        if hasattr(base, name):
            setattr(clone, name, getattr(base, name))
    clone.__cpu_s4_3e_source_binding__ = copy.deepcopy(identity)
    clone.__cpu_s4_3e_variant__ = handoff.variant
    return clone, identity


def make_factory(
    variant: str = "C_INTERNAL",
    *,
    frozen_r8_path: str | Path | None = None,
    kernel_threads: int = 24,
    temporal_workers: int = 48,
    spatial_workers: int = 24,
    raman_workers: int = 48,
    block_size: int = 32,
    observe: Callable[[str, Any], None] | None = None,
):
    """Return a factory producing a source-bound S4-3E propagator/dispatch pair."""
    normalized = str(variant).upper()
    if normalized not in _VARIANTS:
        raise ValueError(f"unknown S4-3E variant {variant!r}; choose {sorted(_VARIANTS)}")
    r8_path, r8_digest = _check_r8(
        Path(frozen_r8_path) if frozen_r8_path is not None else _default_r8_path()
    )
    r8 = _load_r8(r8_path)
    source = source_identity(r8_path)
    base_factory = r8.make_candidate_factory(
        kernel_threads=int(kernel_threads),
        temporal_workers=int(temporal_workers),
        spatial_workers=int(spatial_workers),
        raman_workers=_validate_factory_value("raman_workers", raman_workers, frozenset({1, 24, 48, 96})),
        block_size=_validate_factory_value("block_size", block_size, frozenset({32, 128, 256})),
        observe=observe,
    )

    def factory(frozen_propagator: Callable[..., Any]):
        base_propagator, dispatch = base_factory(frozen_propagator)
        r8_state = getattr(dispatch, "cpu_s4_3c_state", None)
        if r8_state is None:
            raise SourceBindingError("R8 factory did not expose its private state")
        handoff = _HandoffState(
            variant=normalized,
            r8_state=r8_state,
            source=source,
            block_size=block_size,
            raman_workers=raman_workers,
            observe=observe,
        )
        propagator, propagate_identity = _clone_bound_propagator(base_propagator, handoff)
        dispatch.cpu_s4_3e_state = handoff
        dispatch.cpu_s4_3e_full_heun = handoff.full_heun
        dispatch.cpu_s4_3e_metrics = handoff.metrics
        dispatch.cpu_s4_3e_source_identity = copy.deepcopy(source)
        dispatch.cpu_s4_3e_propagate_source_identity = copy.deepcopy(propagate_identity)
        dispatch.cpu_s4_3e_variant = normalized
        dispatch.cpu_s4_3e_full_heun_contract = {
            "input": "field, dz, original Raman/operator kwargs only",
            "output": "{'stage1': original_stage_dict, 'result': original_apply_result}",
            "external_stage1": "rejected",
            "internal_hash_leaf_target": 5 if normalized == "C_INTERNAL" else 10,
            "private_arrays_exposed_to_observer": False,
        }
        propagator._cpu_s4_3e_state = handoff
        propagator._cpu_s4_3e_full_heun = handoff.full_heun
        propagator._cpu_s4_3e_execution = handoff.metrics()
        factory.last_propagator = propagator
        factory.last_dispatch = dispatch
        factory.last_state = handoff
        factory.last_full_heun = handoff.full_heun
        factory.last_source_identity = copy.deepcopy(source)
        factory.frozen_r8_path = str(r8_path)
        factory.frozen_r8_raw_sha256 = r8_digest
        return propagator, dispatch

    factory.variant = normalized
    factory.candidate_schema = CANDIDATE_SCHEMA
    factory.candidate_family = "RAMAN_TRUSTED_INTERNAL_HANDOFF_V1"
    factory.last_propagator = None
    factory.last_dispatch = None
    factory.last_state = None
    factory.last_full_heun = None
    factory.last_source_identity = copy.deepcopy(source)
    factory.source_identity = copy.deepcopy(source)
    factory.frozen_r8_path = str(r8_path)
    factory.frozen_r8_raw_sha256 = r8_digest
    return factory


__all__ = [
    "CandidateUnsupported",
    "HandoffRejected",
    "SourceBindingError",
    "make_factory",
    "source_identity",
]
