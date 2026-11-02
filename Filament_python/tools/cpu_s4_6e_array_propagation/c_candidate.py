"""Private G2 adapter for the C/P24 nonlinear field candidate.

This module deliberately leaves the frozen G2 adapter and the A/P24 entry
points untouched.  It clones only the existing G2 namespace, adds one
source-bound nonlinear call helper before final compilation, and keeps the
public six-tuple shape used by the existing CPU candidates.
"""

from __future__ import annotations

import ast
import builtins
import copy
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np

from tools.hpc_ops import provenance_v2 as pv

from .c_instrument import (
    CInstrumentationError,
    _seal,
    compile_private_apply,
    source_binding,
    validate_cpu_backend,
    verify_seal,
    verify_private_apply,
)
from .c_pool import CPool, NullCPool


CANDIDATE_SCHEMA = "filament.cpu_s4_6e.c_propagation.candidate.v1"
DEFAULT_VARIANT = "NONLINEAR_POINTWISE_P24_V1"
ROLE_REFERENCE = "R"
ROLE_CANDIDATE = "C"
C_APPLY_HELPER = "_c_apply_nonlinear"
MODES = ("QUALIFY_BOUNDARY", "PURE_OFF_SMALLGRID")
LOCAL_CONTRACT = {
    "numba_pool": 1,
    "outer_mask": 1,
    "temporal_workers": 1,
    "spatial_workers": 1,
    "raman_workers": 1,
    "c_workers": 1,
    "scope": "LOCAL_ENGINEERING_ONLY",
    "hpc_allowed": False,
}
HPC_CONTRACT = {
    "numba_pool": 48,
    "outer_mask": 24,
    "temporal_workers": 48,
    "spatial_workers": 24,
    "raman_workers": 48,
    "c_workers": 24,
    "scope": "HPC_FIXED_W48_W24_C24",
    "hpc_allowed": True,
}


class CandidateUnsupported(RuntimeError):
    """The fixed C role cannot satisfy the selected route."""


class SourceBindingError(RuntimeError):
    """A frozen G2 or private C identity failed closed."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _code_identity(function: Callable[..., Any]) -> str:
    import hashlib
    import marshal

    code = getattr(function, "__code__", None)
    if code is None:
        raise SourceBindingError("C_CODE_IDENTITY_MISSING")
    return hashlib.sha256(marshal.dumps(code)).hexdigest()


def _owned_source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    repo = _repo_root()
    files: dict[str, dict[str, Any]] = {}
    for name in ("__init__.py", "c_pool.py", "c_instrument.py", "c_candidate.py"):
        path = (root / name).resolve(strict=True)
        raw = path.read_bytes()
        record: dict[str, Any] = {
            "path": str(path),
            "size_bytes": int(len(raw)),
            "raw_sha256": pv.raw_sha256(raw),
            "classification": "external",
            "hash_scope": "raw_bytes",
        }
        relative = path.relative_to(repo).as_posix()
        try:
            oid = pv._git_blob_oid(repo, relative)
        except pv.ProvenanceError:
            if os.environ.get("SLURM_JOB_ID"):
                raise SourceBindingError("C_COMMITTED_OWNED_SOURCE_REQUIRED")
        else:
            blob = pv._read_git_blob(repo, oid)
            if pv.canonical_lf_bytes(raw) != pv.canonical_lf_bytes(blob):
                raise SourceBindingError("C_OWNED_SOURCE_CANONICAL_BLOB_MISMATCH")
            record.update(
                {
                    "classification": "tracked_text",
                    "hash_scope": pv.TRACKED_HASH_SCOPE,
                    "git_blob_oid": oid,
                    "canonical_lf_sha256": pv.canonical_lf_sha256(blob),
                    "relative_path": relative,
                    "raw_byte_observation": "CURRENT_CHECKOUT_ONLY_NOT_CROSS_PLATFORM_IDENTITY",
                }
            )
        files[name] = record
    composite = (
        "producer_module:tools.cpu_s4_6e_array_propagation.c_candidate\n"
        + "".join(f"{name}:{files[name]['raw_sha256']}\n" for name in sorted(files))
    ).encode("utf-8")
    return {
        "schema": CANDIDATE_SCHEMA + ".owned_sources.v1",
        "producer_module_identity": "tools.cpu_s4_6e_array_propagation.c_candidate",
        "files": files,
        "composite_raw_sha256": pv.raw_sha256(composite),
    }


def source_identity() -> dict[str, Any]:
    parent = importlib.import_module("tools.cpu_s4_5a_l1_residual.candidate")
    return {
        "schema": CANDIDATE_SCHEMA,
        "candidate_name": DEFAULT_VARIANT,
        "roles": [ROLE_REFERENCE, ROLE_CANDIDATE],
        "parent_candidate": "tools.cpu_s4_5a_l1_residual",
        "parent_source_identity": copy.deepcopy(parent.source_identity()),
        "owned_source_identity": _owned_source_identity(),
        "nonlinear_source_binding": "Filament_python/KHz_filament/nonlinear.py:apply_nonlinear",
        "new_math_algorithm": False,
        "pointwise_rounds": ["complex_phase_multiply", "absorption_multiply"],
        "worker_count_hpc": 24,
        "production_qualified": False,
    }


def _normalise_mode(mode: str) -> str:
    value = str(mode).upper()
    if value == "PURE_OFF":
        value = "PURE_OFF_SMALLGRID"
    if value not in MODES:
        raise CandidateUnsupported(f"C_MODE_UNSUPPORTED:{mode}")
    return value


def _resolve_resources(*, local_engineering: bool, pool_override: int | None) -> dict[str, Any]:
    if local_engineering:
        if os.environ.get("SLURM_JOB_ID"):
            raise CandidateUnsupported("C_LOCAL_ENGINEERING_ON_SLURM")
        if pool_override != 1:
            raise CandidateUnsupported("C_LOCAL_REQUIRES_POOL_OVERRIDE_1")
        return copy.deepcopy(LOCAL_CONTRACT)
    if pool_override is not None:
        raise CandidateUnsupported("C_HPC_POOL_OVERRIDE_FORBIDDEN")
    return copy.deepcopy(HPC_CONTRACT)


def _build_parent(*, local_engineering: bool, resources: Mapping[str, Any], recorder_pair: Any) -> tuple[Any, Any, Any]:
    parent = importlib.import_module("tools.cpu_s4_5a_l1_residual.candidate")
    import numba

    if os.environ.get("UPPE_USE_GPU", "0") != "0":
        raise CandidateUnsupported("C_CPU_ROUTE_REQUIRES_UPPE_USE_GPU_0")
    actual = int(numba.config.NUMBA_NUM_THREADS)
    if actual != int(resources["numba_pool"]):
        raise CandidateUnsupported(f"C_NUMBA_POOL_MISMATCH:{actual}:{resources['numba_pool']}")
    numba.set_num_threads(int(resources["outer_mask"]))
    factory = parent.make_factory(
        local_engineering=bool(local_engineering),
        kernel_threads=1 if local_engineering else 24,
        temporal_workers=int(resources["temporal_workers"]),
        spatial_workers=int(resources["spatial_workers"]),
        raman_workers=int(resources["raman_workers"]),
        profile=False,
        recorder=recorder_pair.s45a,
    )
    propagator, dispatch = parent._bind_factory(factory)
    state = getattr(dispatch, "cpu_s45a_state", None) or getattr(dispatch, "cpu_s4_4a_state", None)
    if state is None:
        raise SourceBindingError("C_PARENT_STATE_MISSING")
    return propagator, dispatch, state


def _find_nonlinear_call(module: ast.Module) -> ast.Call:
    calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_g_apply_nonlinear"
    ]
    if len(calls) != 1:
        raise SourceBindingError(f"C_G_APPLY_CALL_COUNT:{len(calls)}")
    return calls[0]


def _strip_c_callsite(module: ast.Module) -> ast.Module:
    restored = copy.deepcopy(module)
    calls = [
        node
        for node in ast.walk(restored)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == C_APPLY_HELPER
    ]
    if len(calls) != 1:
        raise SourceBindingError(f"C_PRIVATE_APPLY_CALL_COUNT:{len(calls)}")
    if not calls[0].args or not (
        isinstance(calls[0].args[-1], ast.Name)
        and calls[0].args[-1].id == "interval_index"
    ):
        raise SourceBindingError("C_INTERVAL_INDEX_STRIP_SHAPE")
    calls[0].args.pop()
    calls[0].func = ast.Name(id="_g_apply_nonlinear", ctx=ast.Load())
    ast.fix_missing_locations(restored)
    return restored


def _ast_hash(node: ast.AST) -> str:
    import hashlib

    return hashlib.sha256(ast.dump(node, include_attributes=False).encode("utf-8")).hexdigest()


def _instrument_c_callsite(source: ast.Module) -> dict[str, Any]:
    baseline = copy.deepcopy(source)
    call = _find_nonlinear_call(source)
    interval_names = [
        node
        for node in ast.walk(source)
        if isinstance(node, ast.Name)
        and node.id == "interval_index"
    ]
    if not interval_names:
        raise SourceBindingError("C_INTERVAL_INDEX_SOURCE_BINDING_MISSING")
    call.func = ast.Name(id=C_APPLY_HELPER, ctx=ast.Load())
    # The source loop already owns the exact interval_index.  Pass it through
    # the private helper only; _make_c_call removes it before invoking the
    # original state math function.
    call.args.append(ast.Name(id="interval_index", ctx=ast.Load()))
    ast.fix_missing_locations(source)
    restored = _strip_c_callsite(source)
    if ast.dump(restored, include_attributes=False) != ast.dump(baseline, include_attributes=False):
        raise SourceBindingError("C_PRIVATE_CALLSITE_STRIP_MISMATCH")
    return {
        "schema": CANDIDATE_SCHEMA + ".callsite.v1",
        "original_helper": "_g_apply_nonlinear",
        "replacement_helper": C_APPLY_HELPER,
        "replacement_count": 1,
        "interval_argument": "interval_index",
        "baseline_ast_sha256": _ast_hash(baseline),
        "stripped_ast_sha256": _ast_hash(restored),
        "inverse_proof": True,
    }


def _make_c_call(
    state: Any,
    dispatch: Any,
    math_function: Callable[..., Any],
    pool: CPool | NullCPool,
    *,
    role: str,
    backend: Mapping[str, Any] | None,
    capture_store: Any | None = None,
) -> Callable[..., Any]:
    if not callable(math_function):
        raise SourceBindingError("C_MATH_FUNCTION_MISSING")

    def c_apply(*args: Any, **kwargs: Any) -> Any:
        _verify_c_call_seal(
            c_apply,
            state=state,
            dispatch=dispatch,
            math_function=math_function,
            pool=pool,
            backend=backend,
            role=role,
            capture_store=capture_store,
        )
        if getattr(dispatch, "cpu_s4_6e_c_state", None) is not state:
            raise SourceBindingError("C_ACTIVE_STATE_CHANGED")
        if getattr(dispatch, "cpu_s4_6e_c_pool", None) is not pool:
            raise SourceBindingError("C_ACTIVE_POOL_CHANGED")
        if pool.closed:
            raise SourceBindingError("C_RETIRED_WORKER_POOL")
        interval_index = None
        call_args = tuple(args)
        if len(call_args) == 5 and isinstance(call_args[-1], (int, np.integer)):
            interval_index = int(call_args[-1])
            call_args = call_args[:-1]
        if backend is None:
            raise SourceBindingError("C_BACKEND_GUARD_MISSING")
        validate_cpu_backend(backend["xp_object"])
        if role == ROLE_CANDIDATE:
            verify_private_apply(math_function)
        # The original G state owns the pre/post capability and exception
        # abort path.  The candidate math callable is the only substituted
        # value; args/kwargs and source ordering remain unchanged.
        if capture_store is not None:
            capture_store.before_call(interval_index, call_args, kwargs)
        result = state._g_nonlinear_call(math_function, call_args, dict(kwargs))
        if capture_store is not None:
            try:
                capture_store.after_call(interval_index, result, call_args, kwargs)
            except BaseException:
                abort = getattr(state, "_g_abort_propagation", None)
                if callable(abort):
                    abort("c_capture_exception")
                raise
        return result

    c_apply.__name__ = C_APPLY_HELPER
    c_apply.__cpu_s46e_c_kind__ = "C_SOURCE_BOUND_NONLINEAR_CALL"
    c_apply.__cpu_s46e_c_state__ = state
    c_apply.__cpu_s46e_c_dispatch__ = dispatch
    c_apply.__cpu_s46e_c_pool__ = pool
    c_apply.__cpu_s46e_c_math__ = math_function
    c_apply.__cpu_s46e_c_role__ = role
    c_apply.__cpu_s46e_c_backend__ = backend
    c_apply.__cpu_s46e_c_capture_store__ = capture_store
    c_apply.__cpu_s46e_c_function_seal__ = _seal(
        c_apply,
        bindings={
            "state": state,
            "dispatch": dispatch,
            "math_function": math_function,
            "pool": pool,
            "backend": backend,
            "role": role,
            "capture_store": capture_store,
        },
    )
    return c_apply


def _verify_c_call_seal(
    helper: Callable[..., Any],
    *,
    state: Any,
    dispatch: Any,
    math_function: Callable[..., Any],
    pool: CPool | NullCPool,
    backend: Mapping[str, Any] | None,
    role: str,
    capture_store: Any | None,
) -> None:
    seal = getattr(helper, "__cpu_s46e_c_function_seal__", None)
    if not isinstance(seal, Mapping):
        raise SourceBindingError("C_CALL_FUNCTION_SEAL_MISSING")
    verify_seal(seal)
    expected = {
        "state": state,
        "dispatch": dispatch,
        "math_function": math_function,
        "pool": pool,
        "backend": backend,
        "role": role,
        "capture_store": capture_store,
    }
    bindings = seal.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(expected):
        raise SourceBindingError("C_CALL_BINDINGS_INVALID")
    for name, value in expected.items():
        same = bindings.get(name) == value if name == "role" else bindings.get(name) is value
        if not same:
            raise SourceBindingError(f"C_CALL_BINDING_CHANGED:{name}")


def _verify_active_compiled_call(wrapped: Any, helper: Callable[..., Any]) -> tuple[Any, dict[str, Any]]:
    active = getattr(wrapped, "__g2_inner__", None)
    if not callable(active):
        raise SourceBindingError("C_ACTIVE_COMPILED_PROPAGATOR_MISSING")
    globals_dict = getattr(active, "__globals__", None)
    if not isinstance(globals_dict, dict):
        raise SourceBindingError("C_ACTIVE_COMPILED_PROPAGATOR_GLOBALS_MISSING")
    if globals_dict.get(C_APPLY_HELPER) is not helper:
        raise SourceBindingError("C_ACTIVE_COMPILED_HELPER_BINDING_CHANGED")
    return active, globals_dict


def make_c_compile_interceptor(
    *,
    g2_namespace: Mapping[str, Any],
    role: str,
    mode: str,
    bindings: tuple[tuple[str, str], ...],
    residual_cost_hook: Any | None = None,
) -> Callable[..., Any]:
    """Apply old timing/boundary, then C, then B before final compilation.

    The frozen timing adapter intentionally validates the original nonlinear
    call anchor before any private callsite substitution.  Reimplementing its
    short, already-tested sequence here keeps that order explicit while the
    old G2/B tree is still stripped back to the untouched baseline.
    """

    b_adapter = importlib.import_module("tools.cpu_s4_5b_steady_cost.adapter")
    timing_module = g2_namespace.get("_timing")
    inject_timing = getattr(timing_module, "_inject_timing", None)
    insert_boundary = g2_namespace.get("_insert_boundary")
    strip_boundary = g2_namespace.get("_strip_boundary")
    structure_without_timing = g2_namespace.get("_structure_without_timing_ast")
    if not callable(inject_timing) or not callable(structure_without_timing):
        raise SourceBindingError("C_G2_TIMING_SEAM_UNAVAILABLE")
    mode_name = "PURE_OFF_SMALLGRID" if str(mode).upper() == "PURE_OFF_REAL8" else str(mode).upper()
    if mode_name == "QUALIFY_BOUNDARY" and not callable(insert_boundary):
        raise SourceBindingError("C_G2_BOUNDARY_INSERT_SEAM_UNAVAILABLE")
    if mode_name == "QUALIFY_BOUNDARY" and not callable(strip_boundary):
        raise SourceBindingError("C_G2_BOUNDARY_STRIP_SEAM_UNAVAILABLE")
    if residual_cost_hook is not None:
        for name in ("instrument", "strip_generated_nodes", "runtime_helpers", "verify_binding"):
            if not callable(getattr(residual_cost_hook, name, None)):
                raise SourceBindingError(f"C_RESIDUAL_COST_SEAM_UNAVAILABLE:{name}")
    injected = False

    def interceptor(source: Any, filename: Any, compile_mode: str, *args: Any, **kwargs: Any) -> Any:
        nonlocal injected
        if not injected and isinstance(source, ast.Module):
            baseline = copy.deepcopy(source)
            timing_receipt = inject_timing(source, role=role)
            boundary_receipt = None
            if mode_name == "QUALIFY_BOUNDARY":
                boundary_receipt = insert_boundary(source, bindings)
            c_receipt = _instrument_c_callsite(source)
            function_node = b_adapter._find_function(source, "propagate_one_pulse")
            b_receipt = b_adapter.instrument_propagation_ast(function_node)
            residual_receipt = None
            post_b_before_residual = None
            if residual_cost_hook is not None:
                post_b_before_residual = copy.deepcopy(function_node)
                residual_receipt = residual_cost_hook.instrument(function_node)
                stripped_residual = residual_cost_hook.strip_generated_nodes(function_node)
                if ast.dump(stripped_residual, include_attributes=False) != ast.dump(
                    post_b_before_residual, include_attributes=False
                ):
                    raise SourceBindingError("C_RESIDUAL_COST_STRIP_MISMATCH")
            restored_joint = source if residual_cost_hook is None else residual_cost_hook.strip_generated_nodes(source)
            without_b = b_adapter.strip_generated_nodes(restored_joint)
            # Restore the C callsite before invoking the old G2 timing and
            # boundary strippers; their expected baseline is the frozen
            # ``_g_apply_nonlinear`` call, not the private C helper name.
            without_b_for_old = _strip_c_callsite(without_b)
            if mode_name == "QUALIFY_BOUNDARY":
                without_old_timing, count = strip_boundary(
                    structure_without_timing(without_b_for_old)
                )
                if count != 1:
                    raise SourceBindingError(f"C_BOUNDARY_STRIP_COUNT:{count}")
                post_strip = without_old_timing
            else:
                post_strip = structure_without_timing(without_b_for_old)
            if ast.dump(post_strip, include_attributes=False) != ast.dump(baseline, include_attributes=False):
                raise SourceBindingError("C_G2_AST_CHANGED_BEYOND_C_AND_B")
            interceptor.static_receipt = {
                "schema": CANDIDATE_SCHEMA + ".g2_static",
                "role": role,
                "mode": mode_name,
                "timing": copy.deepcopy(timing_receipt),
                "boundary": copy.deepcopy(boundary_receipt),
                "b_propagation": copy.deepcopy(b_receipt),
                "residual_cost": copy.deepcopy(residual_receipt),
                "c_callsite": copy.deepcopy(c_receipt),
                "baseline_ast_sha256": _ast_hash(baseline),
                "post_strip_ast_sha256": _ast_hash(post_strip),
                "c_callsite_before_final_compile": True,
                "old_g2_tree_preserved": True,
                "actual_clone_guard_seam": "BEFORE_FINAL_COMPILE",
            }
            injected = True
        return builtins.compile(source, filename, compile_mode, *args, **kwargs)

    interceptor.static_receipt = None
    return interceptor


def attach_with_c(
    raw_propagator: Callable[..., Any],
    dispatch: Any,
    *,
    role: str,
    mode: str,
    capture_root: str | Path,
    recorder: Any,
    pool: CPool | NullCPool,
    math_function: Callable[..., Any] | None = None,
    capture_store: Any | None = None,
    residual_cost_hook: Any | None = None,
) -> tuple[Any, Any, Any, dict[str, Any], Any, Any, dict[str, Any]]:
    """Attach a C-aware G2 clone while preserving the existing six-tuple."""

    if not callable(raw_propagator):
        raise CandidateUnsupported("C_RAW_PROPAGATOR_REQUIRED")
    role_name = str(role).upper()
    if role_name not in (ROLE_REFERENCE, ROLE_CANDIDATE):
        raise CandidateUnsupported(f"C_ROLE_INVALID:{role_name}")
    if residual_cost_hook is not None and role_name != ROLE_CANDIDATE:
        raise CandidateUnsupported("C_RESIDUAL_COST_REFERENCE_FORBIDDEN")
    mode_name = _normalise_mode(mode)
    adapter = importlib.import_module("tools.cpu_s4_5b_steady_cost.adapter")
    if type(recorder) is not getattr(adapter, "TimingRecorder", object):
        raise CandidateUnsupported("C_RECORDER_EXACT_TYPE_REQUIRED")
    from tools.cpu_s4_3g3_smallgrid.scientific_context import frozen_tools
    from tools.cpu_s4_checkpoint_support import g2_adapter_path, load_g2_adapter

    repo = _repo_root()
    frozen_tools(repo)
    original = load_g2_adapter()
    if Path(getattr(original, "__file__", "")).resolve() != g2_adapter_path().resolve(strict=True):
        raise SourceBindingError("C_FROZEN_G2_MODULE_MISMATCH")
    namespace = adapter.clone_g2_namespace(original)
    common = SimpleNamespace(**vars(original._common))
    common.factory = lambda unused_role: (lambda unused_base: (raw_propagator, dispatch))
    namespace["_common"] = common

    base_globals = getattr(raw_propagator, "__globals__", {})
    original_apply = getattr(getattr(dispatch, "cpu_s4_3g_state", None), "_g_nonlinear_function", None)
    if not callable(original_apply):
        original_apply = base_globals.get("apply_nonlinear")
    if not callable(original_apply):
        raise SourceBindingError("C_ORIGINAL_APPLY_MISSING")
    if role_name == ROLE_CANDIDATE and math_function is None:
        math_function, math_receipt = compile_private_apply(original_apply, pool)  # type: ignore[arg-type]
    elif role_name == ROLE_REFERENCE:
        math_function = original_apply
        math_receipt = {"reference_original": True, "source_binding": source_binding(original_apply)}
    else:
        math_receipt = getattr(math_function, "__cpu_s46e_c_instrumentation__", {})

    state = getattr(dispatch, "cpu_s4_3g_state", None)
    if state is None:
        raise SourceBindingError("C_G_STATE_MISSING")
    backend = validate_cpu_backend(getattr(original_apply, "__globals__", {}).get("xp"))
    if role_name == ROLE_CANDIDATE:
        backend = getattr(math_function, "__cpu_s46e_c_backend__", None)
        if not isinstance(backend, Mapping):
            raise SourceBindingError("C_PRIVATE_MATH_BACKEND_MISSING")
    c_apply = _make_c_call(
        state,
        dispatch,
        math_function,
        pool,
        role=role_name,
        backend=backend,
        capture_store=capture_store,
    )
    dispatch.cpu_s4_6e_c_state = state
    dispatch.cpu_s4_6e_c_pool = pool
    dispatch.cpu_s4_6e_c_apply = c_apply
    dispatch.cpu_s4_6e_c_math = math_function
    dispatch.cpu_s4_6e_c_backend = backend

    b_helpers = adapter.bind_runtime_helpers(recorder)
    residual_helpers = {}
    if residual_cost_hook is not None:
        if getattr(residual_cost_hook, "recorder", recorder) is not recorder:
            raise SourceBindingError("C_RESIDUAL_COST_RECORDER_BINDING_CHANGED")
        try:
            residual_helpers = dict(residual_cost_hook.runtime_helpers())
        except BaseException as exc:
            raise SourceBindingError("C_RESIDUAL_COST_HELPER_BINDING_FAILED") from exc

    def compile_interceptor(*, role: str, mode: str, bindings: tuple[tuple[str, str], ...]):
        return make_c_compile_interceptor(
            role=role,
            mode=mode,
            bindings=bindings,
            g2_namespace=namespace,
            residual_cost_hook=residual_cost_hook,
        )

    namespace["_make_compile_interceptor"] = compile_interceptor
    original_clone_g = namespace.get("_clone_g")
    original_clone_i0 = namespace.get("_clone_i0")
    if not callable(original_clone_g) or not callable(original_clone_i0):
        raise SourceBindingError("C_G2_CLONE_ENTRY_MISSING")

    def clone_g(*args: Any, helpers: Mapping[str, Any], **kwargs: Any):
        extended = dict(helpers)
        extended.update(b_helpers)
        extended.update(residual_helpers)
        extended[C_APPLY_HELPER] = c_apply
        return original_clone_g(*args, helpers=extended, **kwargs)

    def clone_i0(*args: Any, helpers: Mapping[str, Any], **kwargs: Any):
        extended = dict(helpers)
        extended.update(b_helpers)
        extended.update(residual_helpers)
        extended[C_APPLY_HELPER] = c_apply
        return original_clone_i0(*args, helpers=extended, **kwargs)

    namespace["_clone_g"] = clone_g
    namespace["_clone_i0"] = clone_i0
    frozen_mode = "QUALIFY_BOUNDARY" if mode_name == "QUALIFY_BOUNDARY" else "PURE_OFF_REAL8"
    wrapped, actual_dispatch, clock, binding, audit, lease = namespace["build_role"](
        "G_CONTROLLED_INPUT",
        frozen_mode,
        capture_root=Path(capture_root).resolve(),
    )
    if actual_dispatch is not dispatch:
        raise SourceBindingError("C_ACTUAL_DISPATCH_BINDING_CHANGED")
    active_propagator, active_globals = _verify_active_compiled_call(wrapped, c_apply)
    _verify_c_call_seal(
        c_apply,
        state=state,
        dispatch=actual_dispatch,
        math_function=math_function,
        pool=pool,
        backend=backend,
        role=role_name,
        capture_store=capture_store,
    )
    actual_dispatch.cpu_s4_6e_c_active_propagator = active_propagator
    actual_dispatch.cpu_s4_6e_c_active_globals = active_globals
    binding = copy.deepcopy(binding)
    binding.update(
        {
            "c_candidate_schema": CANDIDATE_SCHEMA,
            "c_role": role_name,
            "c_mode": mode_name,
            "c_apply_helper": C_APPLY_HELPER,
            "c_pool_workers": int(getattr(pool, "count", 0)),
            "c_pool_scope": "ONE_PERSISTENT_POOL_PER_FACTORY",
            "c_math_receipt": copy.deepcopy(math_receipt),
            "c_common_timing_profile": False,
            "c_g2_compile_seam": "PRIVATE_NAMESPACE_BEFORE_FINAL_COMPILE",
            "c_old_g2_wrapper_untouched": True,
            "c_source_binding": source_binding(original_apply),
            "c_capture_enabled": capture_store is not None,
            "c_residual_cost_enabled": residual_cost_hook is not None,
            "c_residual_cost_binding": None
            if residual_cost_hook is None
            else copy.deepcopy(residual_cost_hook.binding()),
        }
    )
    return wrapped, actual_dispatch, clock, binding, audit, lease, {
        "math_function": math_function,
        "math_receipt": copy.deepcopy(math_receipt),
        "c_apply": c_apply,
        "residual_cost_hook": residual_cost_hook,
        "residual_cost_binding": None
        if residual_cost_hook is None
        else copy.deepcopy(residual_cost_hook.binding()),
    }


def _install_pool_api(dispatch: Any, pool: CPool | NullCPool) -> None:
    dispatch.close_c_pool = pool.close
    dispatch.c_pool_snapshot = pool.snapshot
    dispatch.take_c_pool_records = pool.take_records
    dispatch.cpu_s4_6e_c_pool = pool


def build_role(
    mode: str,
    *,
    capture_root: str | Path,
    role: str = ROLE_CANDIDATE,
    local_engineering: bool = False,
    pool_override: int | None = None,
    recorder_pair: Any | None = None,
    capture_store: Any | None = None,
) -> tuple[Any, Any, Any, dict[str, Any], Any, Any]:
    """Build the C/R six-tuple for local engineering or approved HPC use."""

    mode_name = _normalise_mode(mode)
    role_name = str(role).upper()
    if role_name not in (ROLE_REFERENCE, ROLE_CANDIDATE):
        raise CandidateUnsupported(f"C_ROLE_UNSUPPORTED:{role}")
    resources = _resolve_resources(local_engineering=bool(local_engineering), pool_override=pool_override)
    adapter = importlib.import_module("tools.cpu_s4_5b_steady_cost.adapter")
    if recorder_pair is None:
        recorder_pair = adapter.make_recorder_pair(False)
    if type(recorder_pair) is not getattr(adapter, "RecorderPair", object):
        raise CandidateUnsupported("C_RECORDER_PAIR_EXACT_TYPE_REQUIRED")
    if bool(getattr(recorder_pair, "profile", True)):
        raise CandidateUnsupported("C_RECORDER_PAIR_PROFILE_MUST_BE_OFF")
    pool: CPool | NullCPool = NullCPool() if role_name == ROLE_REFERENCE else CPool(int(resources["c_workers"]))
    try:
        propagator, dispatch, state = _build_parent(
            local_engineering=bool(local_engineering),
            resources=resources,
            recorder_pair=recorder_pair,
        )
        s45a = importlib.import_module("tools.cpu_s4_5a_l1_residual.instrument")
        s45a.install_propagation_wrappers(propagator, state)
        pre_check = s45a.verify_bindings(state)
        owned = _owned_source_identity()
        wrapped, actual_dispatch, clock, runtime_binding, audit, lease, private = attach_with_c(
            propagator,
            dispatch,
            role=role_name,
            mode=mode_name,
            capture_root=Path(capture_root).resolve(),
            recorder=recorder_pair.b,
            pool=pool,
            capture_store=capture_store,
        )
        s45a.bind_active_propagation(wrapped, state)
        post_check = s45a.verify_bindings(state)
        _install_pool_api(actual_dispatch, pool)
        actual_dispatch.cpu_s4_6e_c_role = role_name
        actual_dispatch.cpu_s4_6e_c_binding = {
            "schema": CANDIDATE_SCHEMA + ".binding.v1",
            "role": role_name,
            "mode": mode_name,
            "candidate": DEFAULT_VARIANT,
            "resource_contract": copy.deepcopy(resources),
            "runtime_wrapper": copy.deepcopy(runtime_binding),
            "s45a_pre_binding_check": copy.deepcopy(pre_check),
            "s45a_post_binding_check": copy.deepcopy(post_check),
            "owned_source_identity": owned,
            "production_qualified": False,
            "scientific_qualification": False,
            "pool": pool.snapshot(),
            "common_timing_profile": False,
        }
        actual_dispatch.cpu_s4_6e_c_binding["binding_check"] = copy.deepcopy(verify_bindings(actual_dispatch))
        return wrapped, actual_dispatch, clock, copy.deepcopy(actual_dispatch.cpu_s4_6e_c_binding), audit, lease
    except BaseException:
        try:
            pool.close()
        except BaseException:
            pass
        raise


def verify_bindings(dispatch: Any) -> dict[str, Any]:
    role = getattr(dispatch, "cpu_s4_6e_c_role", None)
    if role not in (ROLE_REFERENCE, ROLE_CANDIDATE):
        raise SourceBindingError("C_ROLE_BINDING_INVALID")
    pool = getattr(dispatch, "cpu_s4_6e_c_pool", None)
    state = getattr(dispatch, "cpu_s4_6e_c_state", None)
    if pool is None or state is None:
        raise SourceBindingError("C_STATE_OR_POOL_MISSING")
    if getattr(dispatch, "cpu_s4_6e_c_pool", None) is not pool:
        raise SourceBindingError("C_POOL_BINDING_CHANGED")
    helper = getattr(dispatch, "cpu_s4_6e_c_apply", None)
    math_function = getattr(dispatch, "cpu_s4_6e_c_math", None)
    backend = getattr(dispatch, "cpu_s4_6e_c_backend", None)
    if not callable(helper) or not callable(math_function) or not isinstance(backend, Mapping):
        raise SourceBindingError("C_HELPER_OR_MATH_BINDING_MISSING")
    if getattr(helper, "__cpu_s46e_c_state__", None) is not state or getattr(helper, "__cpu_s46e_c_pool__", None) is not pool:
        raise SourceBindingError("C_HELPER_BINDING_CHANGED")
    _verify_c_call_seal(
        helper,
        state=state,
        dispatch=dispatch,
        math_function=math_function,
        pool=pool,
        backend=backend,
        role=role,
        capture_store=getattr(helper, "__cpu_s46e_c_capture_store__", None),
    )
    active_propagator = getattr(dispatch, "cpu_s4_6e_c_active_propagator", None)
    active_globals = getattr(dispatch, "cpu_s4_6e_c_active_globals", None)
    if not callable(active_propagator) or not isinstance(active_globals, dict):
        raise SourceBindingError("C_ACTIVE_COMPILED_PROPAGATOR_BINDING_MISSING")
    if getattr(active_propagator, "__globals__", None) is not active_globals:
        raise SourceBindingError("C_ACTIVE_COMPILED_PROPAGATOR_GLOBALS_CHANGED")
    if active_globals.get(C_APPLY_HELPER) is not helper:
        raise SourceBindingError("C_ACTIVE_COMPILED_HELPER_BINDING_CHANGED")
    residual_cost_hook = getattr(dispatch, "cpu_s4_6i_cost_hook", None)
    if residual_cost_hook is not None:
        verify_cost = getattr(residual_cost_hook, "verify_binding", None)
        if not callable(verify_cost):
            raise SourceBindingError("C_RESIDUAL_COST_HOOK_MISSING")
        try:
            cost_check = verify_cost(active_globals)
        except BaseException as exc:
            raise SourceBindingError("C_RESIDUAL_COST_BINDING_FAILED") from exc
    else:
        cost_check = None
    validate_cpu_backend(backend["xp_object"])
    if role == ROLE_CANDIDATE:
        if not isinstance(pool, CPool):
            raise SourceBindingError("C_CANDIDATE_POOL_TYPE_CHANGED")
        verify_private_apply(math_function)
    else:
        if not isinstance(pool, NullCPool):
            raise SourceBindingError("C_REFERENCE_POOL_TYPE_CHANGED")
    return {
        "schema": CANDIDATE_SCHEMA + ".binding.v1",
        "role": role,
        "state_bound": getattr(dispatch, "cpu_s4_6e_c_state", None) is state,
        "helper_bound": True,
        "math_source_bound": role == ROLE_REFERENCE or bool(getattr(math_function, "__cpu_s46e_c_source_binding__", None)),
        "pool": pool.snapshot(),
        "production_qualified": False,
        "residual_cost": cost_check,
    }


__all__ = [
    "CANDIDATE_SCHEMA",
    "C_APPLY_HELPER",
    "CandidateUnsupported",
    "DEFAULT_VARIANT",
    "ROLE_CANDIDATE",
    "ROLE_REFERENCE",
    "SourceBindingError",
    "attach_with_c",
    "build_role",
    "make_c_compile_interceptor",
    "source_identity",
    "verify_bindings",
]
