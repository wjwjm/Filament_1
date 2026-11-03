"""Source-bound four-context hook for ``POWER_VOLUME_CLEANUP_P24``."""

from __future__ import annotations

import ast
import copy
import importlib
import os
import threading
from pathlib import Path
from types import FunctionType
from typing import Any, Callable, Mapping

import numpy as np

from tools.cpu_s4_6h_combined_parallel.instrument import _code_identity, _seal, verify_seal
from tools.hpc_ops import provenance_v2 as pv
from tools.cpu_s4_6j_alpha_factory.factory_timing import scope as factory_scope
from . import power_volume as implementation

from .power_volume import (
    POWER_VOLUME_SCHEMA,
    POWER_VOLUME_SEMANTIC_ID,
    PowerVolumePool,
    SOURCE_SITE,
    power_volume_cleanup_p24,
)


POWER_HOOK_SCHEMA = "filament.cpu.s4_6i.residual_parallel.power_hook.v1"
Q_ION_HELPER = "__cpu_s46i_power_q_ion_from_power__"
Q_IB_HELPER = "__cpu_s46i_power_q_ib_from_power__"
DIRECT_ION_HELPER = "__cpu_s46i_power_direct_ion_interval_energy__"
DIRECT_IB_HELPER = "__cpu_s46i_power_direct_ib_interval_energy__"
INTERVAL_KW = "__cpu_s46i_interval_index__"
_ALIAS_MARK = "_cpu_s46i_power_alias_call"
_ALIAS_ORIGINAL = "_cpu_s46i_power_alias_original"
_ALIAS_INTERVAL = "_cpu_s46i_power_alias_interval"


class PowerHookError(RuntimeError):
    """A source, AST, closure, or runtime helper failed closed."""


def _ast_hash(node: ast.AST) -> str:
    return pv.raw_sha256(ast.dump(node, include_attributes=False).encode("utf-8"))


def owned_source_identity() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[3]
    records = {}
    for name in ('power_volume.py', 'power_hook.py', 'selected_factory.py'):
        path = Path(__file__).with_name(name)
        relative = path.relative_to(repo).as_posix()
        with factory_scope(
            "source_read_hash",
            site="source_read_hash.power_hook.owned_source_identity.read_bytes",
        ):
            payload = path.read_bytes()
        record = dict(path=relative, size_bytes=len(payload), classification='local_worktree_text',
                      hash_scope='raw_bytes', raw_sha256=pv.raw_sha256(payload))
        try:
            with factory_scope(
                "git_blob_read",
                site="git_blob_read.power_hook.owned_source_identity.git_objects",
            ):
                oid = pv._git_blob_oid(repo, relative)
                blob = pv._read_git_blob(repo, oid)
        except pv.ProvenanceError:
            if os.environ.get('SLURM_JOB_ID'):
                raise PowerHookError('S46I_POWER_COMMITTED_SOURCE_REQUIRED')
        else:
            if pv.canonical_lf_bytes(payload) != pv.canonical_lf_bytes(blob):
                if os.environ.get('SLURM_JOB_ID'):
                    raise PowerHookError('S46I_POWER_OWNED_SOURCE_BLOB_MISMATCH')
            else:
                record.update(classification='tracked_text', hash_scope=pv.TRACKED_HASH_SCOPE,
                              git_blob_oid=oid, canonical_lf_sha256=pv.canonical_lf_sha256(blob))
        records[name] = record
    return dict(schema=POWER_HOOK_SCHEMA+'.owned_sources.v1', records=records)


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise PowerHookError(f"S46I_POWER_SOURCE_FUNCTION_COUNT:{name}:{len(matches)}")
    return matches[0]


def _source_function(path: Path, name: str) -> tuple[ast.FunctionDef, str]:
    try:
        with factory_scope(
            "source_read_hash",
            site="source_read_hash.power_hook.source_function.read_text",
        ):
            source = path.read_text(encoding="utf-8")
        with factory_scope(
            "ast_parse_lookup",
            site="ast_parse_lookup.power_hook.source_function.parse_lookup",
        ):
            tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise PowerHookError(f"S46I_POWER_SOURCE_READ_FAILED:{name}") from exc
    with factory_scope(
        "ast_parse_lookup",
        site="ast_parse_lookup.power_hook.source_function.parse_lookup",
    ):
        node = _find_function(tree, name)
    return copy.deepcopy(node), _ast_hash(node)


def _clone_function(function: Callable[..., Any], namespace: dict[str, Any], *, name: str) -> FunctionType:
    code = getattr(function, "__code__", None)
    if code is None:
        raise PowerHookError(f"S46I_POWER_SOURCE_CODE_MISSING:{name}")
    with factory_scope(
        "ast_transform_compile",
        site="ast_transform_compile.power_hook.clone_function",
    ):
        clone = FunctionType(
            code,
            namespace,
            name=name,
            argdefs=getattr(function, "__defaults__", None),
            closure=getattr(function, "__closure__", None),
        )
    clone.__kwdefaults__ = getattr(function, "__kwdefaults__", None)
    clone.__annotations__ = getattr(function, "__annotations__", {})
    clone.__module__ = getattr(function, "__module__", None)
    clone.__qualname__ = getattr(function, "__qualname__", name)
    clone.__doc__ = getattr(function, "__doc__", None)
    return clone


def _contains_array(value: Any, *, depth: int = 0) -> bool:
    if depth > 5:
        return False
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, Mapping):
        return any(_contains_array(item, depth=depth + 1) for item in value.values())
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_array(item, depth=depth + 1) for item in value)
    return False


def _find_call(function_node: ast.FunctionDef, original: str, line: int) -> ast.Call:
    matches = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == original
        and int(getattr(node, "lineno", -1)) == int(line)
    ]
    if len(matches) != 1:
        raise PowerHookError(f"S46I_POWER_ALIAS_CALL_COUNT:{original}:{line}:{len(matches)}")
    return matches[0]


_ALIAS_SITES = (
    ("q_ion_from_power", Q_ION_HELPER, 1117, "q_ion"),
    ("q_ib_from_power", Q_IB_HELPER, 1119, "q_ib"),
    ("direct_interval_energy", DIRECT_ION_HELPER, 1129, "direct_ion"),
    ("direct_interval_energy", DIRECT_IB_HELPER, 1132, "direct_ib"),
)


def instrument_power_ast(function_node: ast.FunctionDef) -> dict[str, Any]:
    """Replace exactly the four selected deposition callsites."""

    if not isinstance(function_node, ast.FunctionDef):
        raise PowerHookError("S46I_POWER_PROPAGATOR_FUNCTION_REQUIRED")
    baseline = copy.deepcopy(function_node)
    aliases: list[dict[str, Any]] = []
    for original, replacement, line, context in _ALIAS_SITES:
        call = _find_call(function_node, original, line)
        call.func = ast.Name(id=replacement, ctx=ast.Load())
        call.keywords.append(
            ast.keyword(
                arg=INTERVAL_KW,
                value=ast.Name(id="interval_index", ctx=ast.Load()),
            )
        )
        setattr(call, _ALIAS_MARK, True)
        setattr(call, _ALIAS_ORIGINAL, original)
        setattr(call, _ALIAS_INTERVAL, True)
        aliases.append(
            {
                "original": original,
                "replacement": replacement,
                "source_line": int(line),
                "context": context,
                "interval_keyword": INTERVAL_KW,
            }
        )
    ast.fix_missing_locations(function_node)
    stripped = strip_power_ast(function_node)
    if ast.dump(stripped, include_attributes=False) != ast.dump(baseline, include_attributes=False):
        raise PowerHookError("S46I_POWER_STRIPPED_POST_B_AST_MISMATCH")
    return {
        "schema": POWER_HOOK_SCHEMA + ".propagation.v1",
        "source_ast_sha256": _ast_hash(baseline),
        "instrumented_ast_sha256": _ast_hash(function_node),
        "stripped_ast_sha256": _ast_hash(stripped),
        "alias_sites": tuple(aliases),
        "replacement_count": len(aliases),
        "scientific_expression_changes": 0,
        "formal_timing_enabled": False,
        "inverse_proof": "strip_power_ast_restores_post_B_C_ast",
        "array_values_retained": False,
    }


def strip_power_ast(node: ast.AST) -> ast.AST:
    """Restore only this hook's aliases and interval keyword arguments."""

    result = copy.deepcopy(node)
    replacements = {replacement: original for original, replacement, _, _ in _ALIAS_SITES}

    class Stripper(ast.NodeTransformer):
        def visit_Call(self, current: ast.Call) -> Any:
            result_call = self.generic_visit(current)
            marked = bool(getattr(result_call, _ALIAS_MARK, False))
            function_name = getattr(result_call.func, "id", None)
            if not marked and function_name in replacements:
                raise PowerHookError("S46I_POWER_ALIAS_MARK_MISSING")
            if not marked:
                return result_call
            original = getattr(result_call, _ALIAS_ORIGINAL, None)
            if not isinstance(original, str) or original not in {item[0] for item in _ALIAS_SITES}:
                raise PowerHookError("S46I_POWER_ALIAS_ORIGINAL_INVALID")
            if not any(item[0] == original and item[1] == function_name for item in _ALIAS_SITES):
                raise PowerHookError("S46I_POWER_ALIAS_REPLACEMENT_INVALID")
            interval_keywords = [item for item in result_call.keywords if item.arg == INTERVAL_KW]
            if len(interval_keywords) != 1 or not (
                isinstance(interval_keywords[0].value, ast.Name)
                and interval_keywords[0].value.id == "interval_index"
            ):
                raise PowerHookError("S46I_POWER_INTERVAL_KEYWORD_INVALID")
            result_call.keywords = [item for item in result_call.keywords if item.arg != INTERVAL_KW]
            result_call.func = ast.Name(id=original, ctx=ast.Load())
            return result_call

    stripped = Stripper().visit(result)
    if stripped is None:
        raise PowerHookError("S46I_POWER_STRIP_RESULT_MISSING")
    ast.fix_missing_locations(stripped)
    return stripped


class PowerVolumeHook:
    """Duck-typed hook consumed by the frozen C/G2 compile interceptor."""

    def __init__(
        self,
        pool: PowerVolumePool | None,
        *,
        capture_store: Any | None = None,
    ) -> None:
        if pool is not None and type(pool) is not PowerVolumePool:
            raise PowerHookError("S46I_POWER_HOOK_POOL_EXACT_TYPE_REQUIRED")
        if capture_store is not None and not all(callable(getattr(capture_store, name, None)) for name in ('before_call','after_call')):
            raise PowerHookError("S46I_POWER_HOOK_CAPTURE_STORE_REQUIRED")
        if (pool is None) != (capture_store is not None):
            raise PowerHookError('S46I_POWER_CAPTURE_REFERENCE_ONLY')
        self.pool = pool
        self._bound_pool = pool
        self._bound_count = None if pool is None else pool.count
        self.capture_store = capture_store
        self._bound_capture_store = capture_store
        self._capture_methods = {} if capture_store is None else {
            name: getattr(capture_store, name).__func__ for name in ('before_call','after_call')}
        self._capture_method_seals = {name:_seal(value) for name,value in self._capture_methods.items()}
        self._failure_cleanup: Callable[[], Any] | None = None
        self._context_local = threading.local()
        deposition = importlib.import_module("KHz_filament.deposition")
        self._deposition_module = deposition
        dep_path = Path(str(deposition.__file__)).resolve(strict=True)
        self._dep_path = dep_path
        self._original_functions = {
            "deposition._require_finite_nonnegative_dt": deposition._require_finite_nonnegative_dt,
            "deposition._power_volume": deposition._power_volume,
            "deposition.integrate_power_to_q": deposition.integrate_power_to_q,
            "deposition.q_ion_from_power": deposition.q_ion_from_power,
            "deposition.q_ib_from_power": deposition.q_ib_from_power,
            "deposition.direct_interval_energy": deposition.direct_interval_energy,
        }
        self._source_xp = self._original_functions["deposition._power_volume"].__globals__.get("xp")
        if self._source_xp is not np:
            raise PowerHookError("S46I_POWER_SOURCE_NUMPY_BACKEND_REQUIRED")
        self._backend_functions = {name:getattr(np,name) for name in ('asarray','nan_to_num','maximum','empty_like','sum')}
        self._implementation_functions = {name:getattr(implementation,name) for name in
            ('validate_power_volume_input','_run_round','power_volume_cleanup_p24','make_power_volume_pool')}
        self._implementation_seals = {name:_seal(fn) for name,fn in self._implementation_functions.items()}
        self._pool_methods = {name:getattr(PowerVolumePool,name) for name in
            ('__init__','_record','run_round','annotate_last_operation','snapshot','take_records','close')}
        self._pool_method_seals = {name:_seal(fn) for name,fn in self._pool_methods.items()}
        self._owned_sources = owned_source_identity()
        self._source_ast_hashes = {
            name: _source_function(dep_path, name.rsplit(".", 1)[1])[1]
            for name in self._original_functions
        }
        self._bound_power = self._make_bound_power()
        private_dep = dict(vars(deposition))
        private_dep["_power_volume"] = self._bound_power
        self._private_dep_namespace = private_dep
        integrate = _clone_function(
            deposition.integrate_power_to_q,
            private_dep,
            name="__cpu_s46i_power_integrate_power_to_q__",
        )
        private_dep["integrate_power_to_q"] = integrate
        q_ion = _clone_function(
            deposition.q_ion_from_power,
            private_dep,
            name="__cpu_s46i_power_q_ion_source_clone__",
        )
        q_ib = _clone_function(
            deposition.q_ib_from_power,
            private_dep,
            name="__cpu_s46i_power_q_ib_source_clone__",
        )
        direct = _clone_function(
            deposition.direct_interval_energy,
            private_dep,
            name="__cpu_s46i_power_direct_interval_energy_clone__",
        )
        self._source_clones = {
            "deposition.integrate_power_to_q": integrate,
            "deposition.q_ion_from_power": q_ion,
            "deposition.q_ib_from_power": q_ib,
            "deposition.direct_interval_energy": direct,
        }
        self._context_wrappers = {
            Q_ION_HELPER: self._make_context_wrapper(q_ion, "q_ion"),
            Q_IB_HELPER: self._make_context_wrapper(q_ib, "q_ib"),
            DIRECT_ION_HELPER: self._make_context_wrapper(direct, "direct_ion"),
            DIRECT_IB_HELPER: self._make_context_wrapper(direct, "direct_ib"),
        }
        self._source_seals = {name: _seal(function) for name, function in self._original_functions.items()}
        self._clone_seals = {name: _seal(function) for name, function in self._source_clones.items()}
        self._bound_power_seal = _seal(self._bound_power)
        self._helper_seals = {name: _seal(function) for name, function in self._context_wrappers.items()}
        self._transform_seals = {
            "instrument": _seal(instrument_power_ast),
            "strip": _seal(strip_power_ast),
        }
        self._backend_slots = {
            name: {
                key: private_dep.get(key)
                for key in set(function.__code__.co_names) | {"xp", "np", "_np"}
                if key in private_dep
            }
            for name, function in self._source_clones.items()
        }
        with factory_scope(
            "metadata",
            site="metadata.power_hook.constructor.receipt",
        ):
            self._receipt = self._make_receipt()
        self._receipt_fingerprint = copy.deepcopy(self._receipt)
        self._active_globals: dict[str, Any] | None = None

    def _make_bound_power(self) -> Callable[..., Any]:
        owner = self

        def bound_power(power: Any) -> Any:
            context = str(getattr(owner._context_local, "context", "unknown"))
            interval = getattr(owner._context_local, "interval", None)
            if owner.pool is None:
                token = owner.capture_store.before_call(context=context, interval_index=interval,
                    source=power, source_site=SOURCE_SITE)
                try:
                    result = owner._original_functions['deposition._power_volume'](power)
                    owner.capture_store.after_call(token, source=power, output=result)
                    return result
                except BaseException:
                    if owner._failure_cleanup is not None:
                        owner._failure_cleanup()
                    raise
            return power_volume_cleanup_p24(
                power,
                pool=owner.pool,
                backend=owner._source_xp,
                context=context,
                interval_index=interval,
                on_failure=owner._failure_cleanup,
            )

        bound_power.__name__ = "__cpu_s46i_power_volume_cleanup_p24__"
        bound_power.__qualname__ = bound_power.__name__
        bound_power.__cpu_s46i_power_volume_kind__ = "FIXED_POWER_VOLUME_CLEANUP_P24"
        bound_power.__cpu_s46i_power_volume_pool__ = self.pool
        return bound_power

    def _make_context_wrapper(self, target: Callable[..., Any], context: str) -> Callable[..., Any]:
        owner = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            interval = kwargs.pop(INTERVAL_KW, None)
            previous_context = getattr(owner._context_local, "context", None)
            previous_interval = getattr(owner._context_local, "interval", None)
            owner._context_local.context = context
            owner._context_local.interval = interval
            try:
                return target(*args, **kwargs)
            finally:
                if previous_context is None:
                    try:
                        del owner._context_local.context
                    except AttributeError:
                        pass
                else:
                    owner._context_local.context = previous_context
                if previous_interval is None:
                    try:
                        del owner._context_local.interval
                    except AttributeError:
                        pass
                else:
                    owner._context_local.interval = previous_interval

        wrapped.__name__ = context
        wrapped.__qualname__ = context
        wrapped.__cpu_s46i_power_volume_kind__ = "FIXED_POWER_CONTEXT_WRAPPER"
        wrapped.__cpu_s46i_power_volume_target__ = target
        wrapped.__cpu_s46i_power_volume_context__ = context
        wrapped.__cpu_s46i_power_volume_pool__ = self.pool
        return wrapped

    def _make_receipt(self) -> dict[str, Any]:
        return {
            "schema": POWER_HOOK_SCHEMA,
            "semantic_id": POWER_VOLUME_SEMANTIC_ID,
            "source_site": SOURCE_SITE,
            "runtime_helpers": tuple(sorted(self._context_wrappers)),
            "context_order": ("q_ion", "q_ib", "direct_ion", "direct_ib"),
            "source_ast_hashes": copy.deepcopy(self._source_ast_hashes),
            "source_code_identities": {
                name: _code_identity(function) for name, function in self._original_functions.items()
            },
            "clone_code_identities": {
                name: _code_identity(function) for name, function in self._source_clones.items()
            },
            "pool_workers": self._bound_count,
            "evaluation": 'ORIGINAL_REFERENCE_CAPTURE' if self.pool is None else 'PARALLEL_TWO_ROUND',
            "owned_sources": copy.deepcopy(self._owned_sources),
            "implementation_code_identities": {name:seal['code_identity'] for name,seal in self._implementation_seals.items()},
            "source_backend": "KHz_filament.deposition.xp",
            "source_nan_to_num_identity": int(id(self._source_xp.nan_to_num)),
            "source_maximum_identity": int(id(self._source_xp.maximum)),
            "formal_timing_enabled": False,
            "capture_enabled": self.capture_store is not None,
            "scientific_expression_changes": 0,
            "array_values_retained": False,
            "inverse_ast_layer": "POWER_ALIASES_ONLY",
        }

    def set_failure_cleanup(self, callback: Callable[[], Any] | None) -> None:
        if callback is not None and not callable(callback):
            raise PowerHookError("S46I_POWER_FAILURE_CLEANUP_REQUIRED")
        self._failure_cleanup = callback

    def runtime_helpers(self) -> dict[str, Callable[..., Any]]:
        return dict(self._context_wrappers)

    def instrument(self, function_node: ast.FunctionDef) -> dict[str, Any]:
        with factory_scope(
            "ast_transform_compile",
            site="ast_transform_compile.power_hook.instrument",
        ):
            return instrument_power_ast(function_node)

    def strip_generated_nodes(self, node: ast.AST) -> ast.AST:
        with factory_scope(
            "ast_transform_compile",
            site="ast_transform_compile.power_hook.strip",
        ):
            return strip_power_ast(node)

    def binding(self) -> dict[str, Any]:
        return copy.deepcopy(self._receipt)

    def bind_active(self, active_globals: Mapping[str, Any]) -> None:
        if not isinstance(active_globals, dict):
            raise PowerHookError("S46I_POWER_ACTIVE_GLOBALS_REQUIRED")
        self.verify_binding(active_globals)
        self._active_globals = active_globals

    def verify_binding(self, active_globals: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self._receipt != self._receipt_fingerprint:
            raise PowerHookError("S46I_POWER_RECEIPT_CHANGED")
        if owned_source_identity() != self._owned_sources:
            raise PowerHookError('S46I_POWER_OWNED_SOURCE_CHANGED')
        if self.capture_store is not self._bound_capture_store:
            raise PowerHookError('S46I_POWER_CAPTURE_STORE_CHANGED')
        for name, method in self._capture_methods.items():
            if getattr(self.capture_store,name).__func__ is not method:
                raise PowerHookError('S46I_POWER_CAPTURE_METHOD_CHANGED')
            verify_seal(self._capture_method_seals[name])
        for name, function in self._implementation_functions.items():
            if getattr(implementation,name) is not function:
                raise PowerHookError('S46I_POWER_KERNEL_BINDING_CHANGED:'+name)
            verify_seal(self._implementation_seals[name])
        if power_volume_cleanup_p24 is not self._implementation_functions['power_volume_cleanup_p24']:
            raise PowerHookError('S46I_POWER_ACTIVE_KERNEL_CHANGED')
        for name, function in self._pool_methods.items():
            if getattr(PowerVolumePool,name) is not function:
                raise PowerHookError('S46I_POWER_POOL_METHOD_CHANGED:'+name)
            verify_seal(self._pool_method_seals[name])
        expected_helpers = {
            Q_ION_HELPER,
            Q_IB_HELPER,
            DIRECT_ION_HELPER,
            DIRECT_IB_HELPER,
        }
        if set(self._context_wrappers) != expected_helpers:
            raise PowerHookError("S46I_POWER_HELPER_REGISTRY_CHANGED")
        if self.pool is not self._bound_pool or (self.pool is not None and
                (type(self.pool) is not PowerVolumePool or self.pool.count != self._bound_count)):
            raise PowerHookError("S46I_POWER_POOL_BINDING_CHANGED")
        if self._source_xp is not np:
            raise PowerHookError("S46I_POWER_SOURCE_NUMPY_BACKEND_CHANGED")
        for name, function in self._backend_functions.items():
            if getattr(self._source_xp,name) is not function:
                raise PowerHookError("S46I_POWER_SOURCE_MATH_BACKEND_CHANGED:"+name)
        if implementation.np is not self._source_xp:
            raise PowerHookError('S46I_POWER_KERNEL_BACKEND_CHANGED')
        verify_seal(self._bound_power_seal)
        if getattr(self._bound_power, "__cpu_s46i_power_volume_pool__", None) is not self.pool:
            raise PowerHookError("S46I_POWER_BOUND_POOL_CHANGED")
        for name, seal in self._transform_seals.items():
            verify_seal(seal)
        for name, function in self._original_functions.items():
            verify_seal(self._source_seals[name])
            attribute = name.rsplit(".", 1)[1]
            if getattr(self._deposition_module, attribute, None) is not function:
                raise PowerHookError(f"S46I_POWER_SOURCE_FUNCTION_CHANGED:{name}")
            current_node, current_hash = _source_function(self._dep_path, name.rsplit(".", 1)[1])
            if current_hash != self._source_ast_hashes[name]:
                raise PowerHookError(f"S46I_POWER_SOURCE_AST_CHANGED:{name}")
            globals_dict = getattr(function, "__globals__", None)
            if not isinstance(globals_dict, dict) or globals_dict.get("xp") is not np:
                raise PowerHookError(f"S46I_POWER_SOURCE_BACKEND_CHANGED:{name}")
        for name, function in self._source_clones.items():
            seal = self._clone_seals[name]
            verify_seal(seal)
            if function.__globals__ is not self._private_dep_namespace:
                raise PowerHookError(f"S46I_POWER_CLONE_NAMESPACE_CHANGED:{name}")
            for key, expected in self._backend_slots[name].items():
                if function.__globals__.get(key) is not expected:
                    raise PowerHookError(f"S46I_POWER_CLONE_BACKEND_CHANGED:{name}:{key}")
        if self._private_dep_namespace.get("_power_volume") is not self._bound_power:
            raise PowerHookError("S46I_POWER_CLONE_POWER_SLOT_CHANGED")
        if self._private_dep_namespace.get("integrate_power_to_q") is not self._source_clones["deposition.integrate_power_to_q"]:
            raise PowerHookError("S46I_POWER_CLONE_INTEGRATE_SLOT_CHANGED")
        if self._private_dep_namespace.get("xp") is not self._source_xp:
            raise PowerHookError("S46I_POWER_CLONE_XP_SLOT_CHANGED")
        for name, helper in self._context_wrappers.items():
            verify_seal(self._helper_seals[name])
            if getattr(helper, "__cpu_s46i_power_volume_pool__", None) is not self.pool:
                raise PowerHookError(f"S46I_POWER_HELPER_POOL_CHANGED:{name}")
            expected_target = {
                Q_ION_HELPER: self._source_clones["deposition.q_ion_from_power"],
                Q_IB_HELPER: self._source_clones["deposition.q_ib_from_power"],
                DIRECT_ION_HELPER: self._source_clones["deposition.direct_interval_energy"],
                DIRECT_IB_HELPER: self._source_clones["deposition.direct_interval_energy"],
            }[name]
            if getattr(helper, "__cpu_s46i_power_volume_target__", None) is not expected_target:
                raise PowerHookError(f"S46I_POWER_HELPER_TARGET_CHANGED:{name}")
            if any(
                _contains_array(cell.cell_contents)
                for cell in (getattr(helper, "__closure__", None) or ())
            ):
                raise PowerHookError(f"S46I_POWER_HELPER_CLOSURE_ARRAY:{name}")
        if any(
            _contains_array(cell.cell_contents)
            for cell in (getattr(self._bound_power, "__closure__", None) or ())
        ):
            raise PowerHookError("S46I_POWER_BOUND_CLOSURE_ARRAY")
        if active_globals is not None:
            if not isinstance(active_globals, dict):
                raise PowerHookError("S46I_POWER_ACTIVE_GLOBALS_INVALID")
            for name, helper in self._context_wrappers.items():
                if active_globals.get(name) is not helper:
                    raise PowerHookError(f"S46I_POWER_ACTIVE_HELPER_CHANGED:{name}")
            self._active_globals = active_globals
        return {
            "schema": POWER_HOOK_SCHEMA + ".binding.v1",
            "semantic_id": POWER_VOLUME_SEMANTIC_ID,
            "active_globals_bound": active_globals is not None or self._active_globals is not None,
            "context_order": ("q_ion", "q_ib", "direct_ion", "direct_ib"),
            "pool_workers": self._bound_count,
            "pool_identity": None if self.pool is None else int(id(self.pool)),
            "source_backend": "KHz_filament.deposition.xp",
            "formal_timing_enabled": False,
            "array_values_retained": False,
        }

    def take_records(self) -> list[dict[str, Any]]:
        return [] if self.pool is None else self.pool.take_records()

    def snapshot(self) -> dict[str, Any]:
        return None if self.pool is None else copy.deepcopy(self.pool.snapshot())

    def close(self) -> dict[str, Any]:
        return None if self.pool is None else self.pool.close()


__all__ = [
    "DIRECT_IB_HELPER",
    "DIRECT_ION_HELPER",
    "INTERVAL_KW",
    "POWER_HOOK_SCHEMA",
    "PowerHookError",
    "PowerVolumeHook",
    "Q_IB_HELPER",
    "Q_ION_HELPER",
    "instrument_power_ast",
    "strip_power_ast",
]
