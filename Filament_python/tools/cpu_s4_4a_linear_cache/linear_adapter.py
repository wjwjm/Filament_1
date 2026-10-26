"""Private source-bound AST seam for the CPU-S4-4A linear-factor cache."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import FunctionType
from typing import Any, Callable, Mapping

import numpy as np

from tools.hpc_ops import provenance_v2 as pv

from .cache import ALGORITHM_ID, CACHE_SCHEMA, CacheContractError, LinearFactorCache


LINEAR_FUNCTION_NAME = "step_linear_bk_nee_factorized"
CACHED_FUNCTION_NAME = "_cpu_s4_4a_cached_step_linear_bk_nee_factorized"
LINEAR_RELATIVE_SUFFIX = "KHz_filament/linear.py"
HELPER_NAME = "__cpu_s4_4a_factor_for_plane"
ADAPTER_SCHEMA = "filament.cpu_s4_4a.linear_adapter.v2"


class LinearBindingError(RuntimeError):
    """The source-bound original producer or private clone is invalid."""


def _code_identity(code: Any) -> str:
    payload = bytes(getattr(code, "co_code", b"")) + repr(
        getattr(code, "co_consts", ())
    ).encode("utf-8", "backslashreplace")
    return pv.raw_sha256(payload)


def _validate_original_namespace(original: Callable[..., Any]) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    if not callable(original):
        raise LinearBindingError("ORIGINAL_LINEAR_NOT_CALLABLE")
    code = getattr(original, "__code__", None)
    globals_dict = getattr(original, "__globals__", None)
    if code is None or not isinstance(globals_dict, dict):
        raise LinearBindingError("ORIGINAL_LINEAR_CODE_OR_NAMESPACE_MISSING")
    if str(getattr(original, "__name__", "")) != LINEAR_FUNCTION_NAME:
        raise LinearBindingError("ORIGINAL_LINEAR_FUNCTION_NAME_MISMATCH")
    if str(getattr(original, "__module__", "")) != "KHz_filament.linear":
        raise LinearBindingError("ORIGINAL_LINEAR_MODULE_NAME_MISMATCH")
    source_path = Path(str(getattr(code, "co_filename", ""))).resolve(strict=True)
    if not str(source_path).replace("\\", "/").endswith(LINEAR_RELATIVE_SUFFIX):
        raise LinearBindingError(f"ORIGINAL_LINEAR_SOURCE_PATH_UNKNOWN:{source_path}")
    raw = source_path.read_bytes()
    from KHz_filament import linear as frozen_linear
    repo = Path(__file__).resolve().parents[3]
    relative = "Filament_python/KHz_filament/linear.py"
    expected_blob = "72878dd20daae656a5f9c8c7e39fa955ae8b30c5"
    if source_path != (repo / relative).resolve() or code is not frozen_linear.step_linear_bk_nee_factorized.__code__:
        raise LinearBindingError("ORIGINAL_LINEAR_FROZEN_CODE_OR_PATH_MISMATCH")
    actual_blob = pv._git(repo, "rev-parse", "HEAD:" + relative)
    if actual_blob != expected_blob:
        raise LinearBindingError("ORIGINAL_LINEAR_FROZEN_GIT_BLOB_MISMATCH")
    xp = globals_dict.get("xp")
    if type(xp).__name__ != "_XPNamespace":
        raise LinearBindingError("ORIGINAL_LINEAR_XP_NAMESPACE_NOT_J2")
    xp_module = str(type(xp).__module__).replace("\\", ".")
    if not xp_module.endswith("cpu_p1r2_j2") or getattr(xp, "xp", None) is not np:
        raise LinearBindingError("ORIGINAL_LINEAR_XP_NAMESPACE_NOT_NATIVE_J2_NUMPY")
    fft = getattr(xp, "fft", None)
    dispatch = getattr(fft, "dispatch", None)
    if type(dispatch).__name__ != "_ScipyDispatch":
        raise LinearBindingError("ORIGINAL_LINEAR_FFT_DISPATCH_NOT_SCIPY")
    dispatch_module = str(type(dispatch).__module__).replace("\\", ".")
    if not dispatch_module.endswith("cpu_p1r2_j2"):
        raise LinearBindingError("ORIGINAL_LINEAR_FFT_DISPATCH_MODULE_NOT_J2")
    for name in (
        "observe",
        "observer",
        "trace",
        "trace_callback",
        "callback",
        "internal_callback",
        "_internal_callback",
        "_observer_callback",
    ):
        if globals_dict.get(name) is not None:
            raise LinearBindingError(f"ORIGINAL_LINEAR_UNKNOWN_OBSERVER_OR_TRACE:{name}")
    source_record = {
        "path": str(source_path),
        "classification": "tracked_text",
        "hash_scope": pv.TRACKED_HASH_SCOPE,
        "git_blob_oid": actual_blob,
        "checkout_raw_sha256": pv.raw_sha256(raw),
        "canonical_lf_sha256": pv.canonical_lf_sha256(raw),
        "size_bytes": int(len(raw)),
        "function_name": LINEAR_FUNCTION_NAME,
        "function_first_line": int(code.co_firstlineno),
        "xp_namespace_type": type(xp).__name__,
        "xp_namespace_module": xp_module,
        "dispatch_type": type(dispatch).__name__,
        "dispatch_module": dispatch_module,
    }
    return globals_dict, source_path, source_record


def _expected_transfer_expression() -> ast.AST:
    return ast.parse(
        "xp.exp(onej * phase_xy * float(dz)).astype(work_ctype, copy=False)",
        mode="eval",
    ).body


def _expected_phase_expression() -> ast.AST:
    return ast.parse(
        "coeff_diff[i] * kperp2 + coeff_gvd[i]",
        mode="eval",
    ).body


def _find_linear_node(tree: ast.Module, first_line: int) -> ast.FunctionDef:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == LINEAR_FUNCTION_NAME]
    if len(matches) != 1:
        raise LinearBindingError(f"LINEAR_AST_FUNCTION_ANCHOR:{len(matches)}")
    node = matches[0]
    if int(node.lineno) != int(first_line):
        raise LinearBindingError(f"LINEAR_AST_FIRST_LINE_DRIFT:{node.lineno}:{first_line}")
    return node


def _raw_transfer_lambda(node: ast.FunctionDef) -> ast.Lambda:
    matches: list[ast.Call] = []
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Assign):
            continue
        if len(candidate.targets) != 1 or not isinstance(candidate.targets[0], ast.Name):
            continue
        if candidate.targets[0].id != "prop2d":
            continue
        if isinstance(candidate.value, ast.Call) and isinstance(candidate.value.func, ast.Attribute):
            if candidate.value.func.attr == "run":
                matches.append(candidate.value)
    if len(matches) != 1:
        raise LinearBindingError(f"LINEAR_TRANSFER_PROFILER_ANCHOR:{len(matches)}")
    profiler_run = matches[0]
    if len(profiler_run.args) != 2 or not isinstance(profiler_run.args[1], ast.Lambda):
        raise LinearBindingError("LINEAR_TRANSFER_PROFILER_LAMBDA_ANCHOR_MISSING")
    if not (
        isinstance(profiler_run.args[0], ast.Constant)
        and profiler_run.args[0].value == "transfer_kernel_preparation"
    ):
        raise LinearBindingError("LINEAR_TRANSFER_PROFILER_STAGE_DRIFT")
    return profiler_run.args[1]


def _find_transfer_lambda(node: ast.FunctionDef) -> ast.Lambda:
    lambda_node = _raw_transfer_lambda(node)
    if ast.dump(lambda_node.body, include_attributes=False) != ast.dump(
        _expected_transfer_expression(), include_attributes=False
    ):
        raise LinearBindingError("LINEAR_TRANSFER_EXPRESSION_DRIFT")
    phase_matches = [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Assign)
        and len(candidate.targets) == 1
        and isinstance(candidate.targets[0], ast.Name)
        and candidate.targets[0].id == "phase_xy"
    ]
    if len(phase_matches) != 1 or ast.dump(phase_matches[0].value, include_attributes=False) != ast.dump(
        _expected_phase_expression(), include_attributes=False
    ):
        raise LinearBindingError("LINEAR_PHASE_EXPRESSION_DRIFT")
    return lambda_node


def _replace_transfer_expression(node: ast.FunctionDef, *, helper_name: str) -> dict[str, Any]:
    original_lambda = _find_transfer_lambda(node)
    original_body = copy.deepcopy(original_lambda.body)
    original_dump = ast.dump(node, include_attributes=False)
    builder_lambda = copy.deepcopy(original_lambda)
    args = [
        ast.Name(id="i", ctx=ast.Load()),
        ast.Name(id="Omega", ctx=ast.Load()),
        ast.Name(id="kperp2", ctx=ast.Load()),
        ast.Name(id="Ew", ctx=ast.Load()),
        ast.Name(id="dz", ctx=ast.Load()),
        ast.Name(id="k0", ctx=ast.Load()),
        ast.Name(id="omega0", ctx=ast.Load()),
        ast.Name(id="beta2", ctx=ast.Load()),
        ast.Name(id="denom_floor", ctx=ast.Load()),
        ast.Name(id="strategy", ctx=ast.Load()),
        ast.Name(id="work_ctype", ctx=ast.Load()),
        ast.Name(id="output_ctype", ctx=ast.Load()),
        ast.Name(id="xp", ctx=ast.Load()),
        builder_lambda,
    ]
    original_lambda.body = ast.copy_location(
        ast.Call(func=ast.Name(id=helper_name, ctx=ast.Load()), args=args, keywords=[]),
        original_lambda.body,
    )
    ast.fix_missing_locations(node)
    restored = copy.deepcopy(node)
    restored.name = LINEAR_FUNCTION_NAME
    restored_lambda = _raw_transfer_lambda(restored)
    restored_lambda.body = original_body
    ast.fix_missing_locations(restored)
    restored_dump = ast.dump(restored, include_attributes=False)
    if restored_dump != original_dump:
        raise LinearBindingError("LINEAR_AST_RESTORE_MISMATCH")
    transformed_dump = ast.dump(node, include_attributes=False)
    return {
        "original_ast_sha256": pv.raw_sha256(original_dump.encode("utf-8")),
        "transformed_ast_sha256": pv.raw_sha256(transformed_dump.encode("utf-8")),
        "restored_ast_sha256": pv.raw_sha256(restored_dump.encode("utf-8")),
        "ast_restore_matches": True,
        "original_transfer_body_dump": ast.dump(original_body, include_attributes=False),
    }


def _factor_for_plane(
    owner: Any,
    plane_index: int,
    omega: Any,
    kperp2: Any,
    field_frequency: Any,
    dz: Any,
    k0: Any,
    omega0: Any,
    beta2: Any,
    denom_floor: Any,
    strategy: Any,
    work_ctype: Any,
    output_ctype: Any,
    xp: Any,
    builder: Callable[[], Any],
) -> Any:
    return owner._cpu_s4_4a_prepare_factor(
        plane_index=int(plane_index),
        builder=builder,
        omega=omega,
        kperp2=kperp2,
        geometry=tuple(int(item) for item in field_frequency.shape),
        dz=dz,
        k0=k0,
        omega0=omega0,
        beta2=beta2,
        denom_floor=denom_floor,
        precision_strategy=strategy,
        norm_strategy=strategy,
        work_dtype=work_ctype,
        output_dtype=output_ctype,
        backend=xp,
        source_version=owner._cpu_s4_4a_source_version,
    )


def _validate_cached_runtime(owner: Any, original: Callable[..., Any], cached: Callable[..., Any]) -> None:
    private = getattr(owner, "_cpu_s4_4a_private_binding", None)
    if not isinstance(private, Mapping):
        raise LinearBindingError("CACHED_PRIVATE_BINDING_MISSING")
    if original is not private.get("original") or getattr(original, "__code__", None) is not private.get("original_code"):
        raise LinearBindingError("ORIGINAL_GUARD_PRODUCER_CHANGED")
    if cached is not private.get("cached") or getattr(cached, "__code__", None) is not private.get("cached_code"):
        raise LinearBindingError("CACHED_PRODUCER_CODE_CHANGED")
    if cached.__globals__ is not private.get("namespace"):
        raise LinearBindingError("CACHED_GLOBAL_NAMESPACE_CHANGED")
    if cached.__name__ != CACHED_FUNCTION_NAME or cached.__qualname__ != CACHED_FUNCTION_NAME or cached.__module__ != __name__:
        raise LinearBindingError("CACHED_PRODUCER_METADATA_CHANGED")
    if getattr(cached, "__cpu_s4_4a_owner__", None) is not owner or getattr(cached, "__cpu_s4_4a_original_function__", None) is not original or getattr(cached, "__cpu_s4_4a_original_code__", None) is not private.get("original_code"):
        raise LinearBindingError("CACHED_PRODUCER_OWNER_OR_GUARD_METADATA_CHANGED")
    if getattr(cached.__code__, "co_name", "") != CACHED_FUNCTION_NAME:
        raise LinearBindingError("CACHED_CODE_NAME_UNKNOWN")
    if str(getattr(cached.__code__, "co_filename", "")) != str(Path(__file__).resolve()):
        raise LinearBindingError("CACHED_CODE_FILENAME_UNKNOWN")
    helper_name = str(private.get("helper_name", ""))
    helper = cached.__globals__.get(helper_name)
    if helper is not private.get("helper") or getattr(helper, "__code__", None) is not private.get("helper_code"):
        raise LinearBindingError("CACHED_HELPER_CHANGED_OR_UNKNOWN")
    if getattr(helper, "__cpu_s4_4a_owner__", None) is not owner:
        raise LinearBindingError("CACHED_HELPER_OWNER_CHANGED")
    original_xp = original.__globals__.get("xp")
    if cached.__globals__.get("xp") is not original_xp:
        raise LinearBindingError("CACHED_XP_NAMESPACE_CHANGED")
    if getattr(cached.__globals__.get("xp"), "fft", None) is not private.get("fft"):
        raise LinearBindingError("CACHED_FFT_NAMESPACE_CHANGED")
    if getattr(private.get("fft"), "dispatch", None) is not private.get("dispatch"):
        raise LinearBindingError("CACHED_FFT_DISPATCH_CHANGED")


def make_cached_linear(original: Callable[..., Any], owner: Any) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Compile a new-code producer using a cloned J2 global namespace."""

    original_globals, source_path, original_source = _validate_original_namespace(original)
    new_globals = dict(original_globals)
    original_xp = original_globals["xp"]
    new_globals["xp"] = original_xp
    if new_globals.get("xp") is not original_xp or getattr(new_globals["xp"], "fft", None) is not getattr(original_xp, "fft", None):
        raise LinearBindingError("CACHED_XP_FFT_OBJECT_CLONE_MISMATCH")
    helper_name = HELPER_NAME

    def bound_helper(*args: Any) -> Any:
        return _factor_for_plane(owner, *args)

    bound_helper.__cpu_s4_4a_owner__ = owner
    new_globals[helper_name] = bound_helper
    source = source_path.read_text(encoding="utf-8")
    source_tree = ast.parse(source, filename=str(source_path))
    original_node = copy.deepcopy(_find_linear_node(source_tree, int(original.__code__.co_firstlineno)))
    transformed_node = copy.deepcopy(original_node)
    transform = _replace_transfer_expression(transformed_node, helper_name=helper_name)
    transformed_node.name = CACHED_FUNCTION_NAME
    ast.fix_missing_locations(transformed_node)
    module = ast.Module(body=[transformed_node], type_ignores=[])
    ast.fix_missing_locations(module)
    local_namespace: dict[str, Any] = {}
    compile_path = Path(__file__).resolve()
    exec(compile(module, str(compile_path), "exec"), new_globals, local_namespace)
    cached = local_namespace.get(CACHED_FUNCTION_NAME)
    if not isinstance(cached, FunctionType):
        raise LinearBindingError("LINEAR_AST_CLONE_DID_NOT_PRODUCE_FUNCTION")
    cached.__kwdefaults__ = copy.deepcopy(getattr(original, "__kwdefaults__", None))
    cached.__defaults__ = getattr(original, "__defaults__", None)
    cached.__annotations__ = copy.deepcopy(getattr(original, "__annotations__", {}))
    cached.__name__ = CACHED_FUNCTION_NAME
    cached.__qualname__ = CACHED_FUNCTION_NAME
    cached.__module__ = __name__
    cached.__cpu_s4_4a_original_function__ = original
    cached.__cpu_s4_4a_original_code__ = original.__code__
    cached.__cpu_s4_4a_owner__ = owner
    cached.__cpu_s4_4a_helper_name__ = helper_name
    cached.__cpu_s4_4a_source_binding__ = copy.deepcopy(original_source)
    cached.__cpu_s4_4a_code_identity__ = _code_identity(cached.__code__)
    private = {
        "original": original,
        "original_code": original.__code__,
        "cached": cached,
        "cached_code": cached.__code__,
        "namespace": new_globals,
        "helper_name": helper_name,
        "helper": bound_helper,
        "helper_code": bound_helper.__code__,
        "xp": original_xp,
        "fft": original_xp.fft,
        "dispatch": original_xp.fft.dispatch,
    }
    cached.__cpu_s4_4a_private_binding__ = private
    public_binding = {
        "schema": ADAPTER_SCHEMA,
        "kind": "NEW_AST_CACHED_EXECUTION_PRODUCER",
        "algorithm_id": ALGORITHM_ID,
        "cache_schema": CACHE_SCHEMA,
        "ast_replacement_count": 1,
        "ast_restore_matches": bool(transform["ast_restore_matches"]),
        "original_ast_sha256": transform["original_ast_sha256"],
        "transformed_ast_sha256": transform["transformed_ast_sha256"],
        "restored_ast_sha256": transform["restored_ast_sha256"],
        "helper_name": helper_name,
        "original_guard_producer": {
            "function_id": int(id(original)),
            "code_id": int(id(original.__code__)),
            "code_identity": _code_identity(original.__code__),
            "global_namespace_id": int(id(original_globals)),
            "source": copy.deepcopy(original_source),
        },
        "cached_execution_producer": {
            "function_id": int(id(cached)),
            "code_id": int(id(cached.__code__)),
            "code_name": str(cached.__code__.co_name),
            "code_filename": str(cached.__code__.co_filename),
            "code_identity": str(cached.__cpu_s4_4a_code_identity__),
            "global_namespace_id": int(id(new_globals)),
            "global_namespace_is_original": False,
            "xp_same_object": bool(new_globals["xp"] is original_xp),
            "fft_same_object": bool(new_globals["xp"].fft is original_xp.fft),
            "cache_owner_id": int(id(owner)),
            "source_path": str(compile_path),
            "source_kind": "AST_DERIVED_NEW_CODE",
            "restored_original_ast_sha256": transform["restored_ast_sha256"],
        },
    }
    return cached, public_binding


def make_controlled_state_class(
    e_module: Any,
    *,
    candidate_schema: str,
    candidate_unsupported: type[BaseException],
    handoff_rejected: type[BaseException],
) -> type:
    """Extend the accepted SG controlled state with only the cache seam."""

    from tools.cpu_s4_3g3_smallgrid import controlled_callsite as sg_controlled

    base = sg_controlled.make_controlled_state_class(
        e_module,
        candidate_schema=candidate_schema,
        candidate_unsupported=candidate_unsupported,
        handoff_rejected=handoff_rejected,
    )

    class _ControlledHandoffState(base):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._cpu_s4_4a_cache = LinearFactorCache(
                owner_identity=f"state:{id(self)}",
                algorithm_source_identity=ALGORITHM_ID,
            )
            self._cpu_s4_4a_source_version = ""
            self._cpu_s4_4a_cached_linear: Callable[..., Any] | None = None
            self._cpu_s4_4a_original_linear: Callable[..., Any] | None = None
            self._cpu_s4_4a_linear_binding: dict[str, Any] = {}
            self._cpu_s4_4a_private_binding: dict[str, Any] = {}

        def _cpu_s4_4a_prepare_factor(self, **kwargs: Any) -> Any:
            plane_index = int(kwargs.pop("plane_index"))
            builder = kwargs.pop("builder")
            if plane_index == 0:
                self._cpu_s4_4a_cache.begin_half(**kwargs)
            return self._cpu_s4_4a_cache.factor(plane_index, builder)

        def _cpu_s4_4a_ensure_cached(self, original: Callable[..., Any]) -> Callable[..., Any]:
            if self._cpu_s4_4a_cached_linear is not None:
                try:
                    _validate_cached_runtime(self, original, self._cpu_s4_4a_cached_linear)
                except LinearBindingError as exc:
                    self._reject(f"CPU_S4_4A_CACHED_BINDING:{exc}")
                return self._cpu_s4_4a_cached_linear
            if original is not getattr(self, "_g_linear_function", None):
                self._reject("CPU_S4_4A_GUARD_PRODUCER_NOT_BOUND")
            cached, binding = make_cached_linear(original, self)
            self._cpu_s4_4a_source_version = ":".join((
                str(binding["original_guard_producer"]["source"]["canonical_lf_sha256"]),
                str(binding["transformed_ast_sha256"]),
                pv.canonical_lf_sha256(Path(__file__).read_bytes()),
            ))
            self._cpu_s4_4a_cache.algorithm_source_identity = f"{ALGORITHM_ID}:{self._cpu_s4_4a_source_version}"
            self._cpu_s4_4a_original_linear = original
            self._cpu_s4_4a_cached_linear = cached
            self._cpu_s4_4a_linear_binding = binding
            self._cpu_s4_4a_private_binding = getattr(cached, "__cpu_s4_4a_private_binding__", {})
            try:
                _validate_cached_runtime(self, original, cached)
            except LinearBindingError as exc:
                self._reject(f"CPU_S4_4A_CACHED_BINDING:{exc}")
            return cached

        def _g_linear_call(
            self,
            site: int,
            original: Callable[..., Any],
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> Any:
            try:
                cached = self._cpu_s4_4a_ensure_cached(original)
                result = super()._g_linear_call(site, cached, args, kwargs)
            except BaseException:
                self._cpu_s4_4a_cache.finish_half(success=False)
                raise
            try:
                self._cpu_s4_4a_cache.finish_half(success=True)
            except BaseException:
                if bool(getattr(self, "_g_propagation_active", False)):
                    self._g_abort_propagation("cpu_s4_4a_finish_half_failure")
                raise
            return result

        def cache_snapshot(self) -> dict[str, Any]:
            return self._cpu_s4_4a_cache.snapshot()

        def linear_binding(self) -> dict[str, Any]:
            return copy.deepcopy(self._cpu_s4_4a_linear_binding)

        def metrics(self) -> dict[str, Any]:
            result = super().metrics()
            result["cpu_s4_4a_cache"] = self.cache_snapshot()
            result["cpu_s4_4a_linear_binding"] = self.linear_binding()
            return result

    _ControlledHandoffState.__name__ = "_ControlledHandoffState"
    _ControlledHandoffState.__qualname__ = "_ControlledHandoffState"
    _ControlledHandoffState.__module__ = __name__
    return _ControlledHandoffState


__all__ = [
    "ADAPTER_SCHEMA",
    "ALGORITHM_ID",
    "CacheContractError",
    "LinearBindingError",
    "LINEAR_FUNCTION_NAME",
    "make_cached_linear",
    "make_controlled_state_class",
]
