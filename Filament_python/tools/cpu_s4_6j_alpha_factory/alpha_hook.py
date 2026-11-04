"""Source-bound alpha cleanup hook layered on the existing power hook.

The hook owns only the four ``propagate.py`` alpha assignments.  The power
volume aliases remain owned by :class:`PowerVolumeHook`; stripping this layer
first therefore returns the joint power-instrumented AST before the parent
stripper restores the remaining aliases.
"""

from __future__ import annotations

import ast
import copy
import importlib
import inspect
import threading
from pathlib import Path
from types import FunctionType
from typing import Any, Callable, Mapping

import numpy as np

from tools.cpu_s4_6i_residual_parallel.power_hook import (
    PowerHookError,
    PowerVolumeHook,
    _ast_hash,
    _contains_array,
    _find_function,
    _seal,
    verify_seal,
)
from tools.cpu_s4_6i_residual_parallel.power_volume import PowerVolumePool
from tools.cpu_s4_6i_residual_parallel.instrument import _iter_statement_lists

from . import alpha_pool as alpha_kernel
from .alpha_pool import (
    ALPHA_MAX_RECORDS,
    AlphaPool,
)
from .factory_timing import source_identity as factory_source_identity


ALPHA_HOOK_SCHEMA = "filament.cpu.s4_6j.alpha_factory.alpha_hook.v1"
ALPHA_HOOK_SEMANTIC_ID = "ION_ALPHA_CLEANUP_P24"
ALPHA_SOURCE_SITE = "propagate.py:830-833:alpha_ion"
ALPHA_INTERVAL_KW = "__s46j_interval_index__"

ALPHA_CLEAN_SOURCE_HELPER = "__cpu_s46j_alpha_clean_source__"
ALPHA_RATIO_HELPER = "__cpu_s46j_alpha_ratio__"
ALPHA_CLEAN_RESULT_HELPER = "__cpu_s46j_alpha_clean_result__"
ALPHA_CLIP_HELPER = "__cpu_s46j_alpha_clip__"
ALPHA_HELPER_NAMES = {
    "clean_source": ALPHA_CLEAN_SOURCE_HELPER,
    "ratio": ALPHA_RATIO_HELPER,
    "clean_result": ALPHA_CLEAN_RESULT_HELPER,
    "clip": ALPHA_CLIP_HELPER,
}
_ALPHA_HELPER_SET = frozenset(ALPHA_HELPER_NAMES.values())
_ALPHA_MARK = "_cpu_s46j_alpha_generated"
_ALPHA_INDEX = "_cpu_s46j_alpha_index"

_ALPHA_SOURCE = """ion_source_prop = xp.nan_to_num(ion_source_prop, nan=0.0, posinf=0.0, neginf=0.0)
alpha_ion = ion_source_prop / (I + I_floor)
alpha_ion = xp.nan_to_num(alpha_ion, nan=0.0, posinf=0.0, neginf=0.0)
xp.maximum(alpha_ion, 0.0, out=alpha_ion)
"""
_ALPHA_EXPECTED = tuple(ast.parse(_ALPHA_SOURCE).body)
_ALPHA_ARGUMENTS = (
    ("ion_source_prop",),
    ("ion_source_prop", "I", "I_floor"),
    ("alpha_ion",),
    ("alpha_ion",),
)
_ALPHA_HELPER_BY_INDEX = (
    ALPHA_CLEAN_SOURCE_HELPER,
    ALPHA_RATIO_HELPER,
    ALPHA_CLEAN_RESULT_HELPER,
    ALPHA_CLIP_HELPER,
)


class AlphaHookError(PowerHookError):
    """The alpha source, helper, pool, or AST binding failed closed."""


def _dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _alpha_block(function_node: ast.FunctionDef) -> tuple[list[ast.stmt], int]:
    matches: list[tuple[list[ast.stmt], int]] = []
    expected = _dump(ast.Module(body=list(_ALPHA_EXPECTED), type_ignores=[]))
    for body in _iter_statement_lists(function_node):
        for start in range(0, max(0, len(body) - len(_ALPHA_EXPECTED) + 1)):
            candidate = body[start : start + len(_ALPHA_EXPECTED)]
            if int(getattr(candidate[0], "lineno", -1)) != 830:
                continue
            if _dump(ast.Module(body=candidate, type_ignores=[])) == expected:
                matches.append((body, start))
    if len(matches) != 1:
        raise AlphaHookError("S46J_ALPHA_FROZEN_FOUR_STATEMENTS_REQUIRED")
    return matches[0]


def _generated_call(index: int) -> ast.Call:
    helper = _ALPHA_HELPER_BY_INDEX[index]
    call = ast.Call(
        func=ast.Name(id=helper, ctx=ast.Load()),
        args=[ast.Name(id=name, ctx=ast.Load()) for name in _ALPHA_ARGUMENTS[index]],
        keywords=[
            ast.keyword(
                arg=ALPHA_INTERVAL_KW,
                value=ast.Name(id="interval_index", ctx=ast.Load()),
            )
        ],
    )
    setattr(call, _ALPHA_MARK, True)
    setattr(call, _ALPHA_INDEX, int(index))
    return call


def instrument_alpha_ast(function_node: ast.FunctionDef) -> dict[str, Any]:
    """Replace exactly the four alpha statements with thin helper calls."""

    if not isinstance(function_node, ast.FunctionDef):
        raise AlphaHookError("S46J_ALPHA_PROPAGATOR_FUNCTION_REQUIRED")
    baseline = copy.deepcopy(function_node)
    body, start = _alpha_block(function_node)
    replacements: list[ast.stmt] = []
    for index, original in enumerate(body[start : start + len(_ALPHA_EXPECTED)]):
        call = _generated_call(index)
        if index < 3:
            if not isinstance(original, ast.Assign) or len(original.targets) != 1:
                raise AlphaHookError("S46J_ALPHA_ASSIGNMENT_SHAPE_CHANGED")
            replacement: ast.stmt = ast.Assign(
                targets=copy.deepcopy(original.targets),
                value=call,
                type_comment=getattr(original, "type_comment", None),
            )
        else:
            replacement = ast.Expr(value=call)
        ast.copy_location(replacement, original)
        if hasattr(original, "end_lineno"):
            replacement.end_lineno = getattr(original, "end_lineno", None)
            replacement.end_col_offset = getattr(original, "end_col_offset", None)
        replacements.append(replacement)
    body[start : start + len(_ALPHA_EXPECTED)] = replacements
    ast.fix_missing_locations(function_node)
    stripped = strip_alpha_ast(function_node)
    if _dump(stripped) != _dump(baseline):
        raise AlphaHookError("S46J_ALPHA_INVERSE_AST_MISMATCH")
    return {
        "schema": ALPHA_HOOK_SCHEMA + ".instrument.v1",
        "source_site": ALPHA_SOURCE_SITE,
        "source_ast_sha256": _ast_hash(baseline),
        "instrumented_ast_sha256": _ast_hash(function_node),
        "stripped_ast_sha256": _ast_hash(stripped),
        "helper_names": dict(ALPHA_HELPER_NAMES),
        "replacement_count": len(replacements),
        "mathematical_changes": 0,
        "assignments_and_lifetimes_preserved": True,
        "array_values_retained": False,
        "inverse_proof": "strip_alpha_ast_restores_joint_power_ast",
    }


def _validate_generated_call(call: ast.Call, index: int) -> None:
    helper = _ALPHA_HELPER_BY_INDEX[index]
    if not isinstance(call.func, ast.Name) or call.func.id != helper:
        raise AlphaHookError("S46J_ALPHA_HELPER_ID_CHANGED")
    if len(call.args) != len(_ALPHA_ARGUMENTS[index]) or any(
        not isinstance(node, ast.Name) or node.id != expected
        for node, expected in zip(call.args, _ALPHA_ARGUMENTS[index])
    ):
        raise AlphaHookError("S46J_ALPHA_HELPER_ARGUMENTS_CHANGED")
    if len(call.keywords) != 1 or call.keywords[0].arg != ALPHA_INTERVAL_KW:
        raise AlphaHookError("S46J_ALPHA_INTERVAL_KEYWORD_CHANGED")
    value = call.keywords[0].value
    if not isinstance(value, ast.Name) or value.id != "interval_index":
        raise AlphaHookError("S46J_ALPHA_INTERVAL_ARGUMENT_CHANGED")
    if not bool(getattr(call, _ALPHA_MARK, False)) or int(getattr(call, _ALPHA_INDEX, -1)) != index:
        raise AlphaHookError("S46J_ALPHA_HELPER_MARK_CHANGED")


def strip_alpha_ast(node: ast.AST) -> ast.AST:
    """Restore the four original expressions and reject helper tampering."""

    result = copy.deepcopy(node)
    helper_calls: list[ast.Call] = []
    for current in ast.walk(result):
        if not isinstance(current, ast.Call):
            continue
        helper_name = getattr(current.func, "id", None)
        if helper_name in _ALPHA_HELPER_SET:
            index = _ALPHA_HELPER_BY_INDEX.index(helper_name)
            _validate_generated_call(current, index)
            helper_calls.append(current)
    if len(helper_calls) != len(_ALPHA_EXPECTED):
        raise AlphaHookError("S46J_ALPHA_HELPER_COUNT_CHANGED")
    indices = [
        _ALPHA_HELPER_BY_INDEX.index(getattr(call.func, "id"))
        for call in helper_calls
    ]
    if sorted(indices) != list(range(len(_ALPHA_EXPECTED))):
        raise AlphaHookError("S46J_ALPHA_HELPER_COUNT_CHANGED")

    matches: list[tuple[list[ast.stmt], int]] = []
    for body in _iter_statement_lists(result):
        for start in range(0, max(0, len(body) - len(_ALPHA_EXPECTED) + 1)):
            block = body[start : start + len(_ALPHA_EXPECTED)]
            if int(getattr(block[0], "lineno", -1)) != 830:
                continue
            valid = True
            for index, statement in enumerate(block):
                expected = _ALPHA_EXPECTED[index]
                if int(getattr(statement, "lineno", -1)) != 830 + index:
                    valid = False
                    break
                call = None
                if index < 3:
                    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                        valid = False
                        break
                    expected_assign = expected
                    if not isinstance(expected_assign, ast.Assign) or _dump(
                        statement.targets[0]
                    ) != _dump(expected_assign.targets[0]):
                        valid = False
                        break
                    call = statement.value
                else:
                    if not isinstance(statement, ast.Expr):
                        valid = False
                        break
                    call = statement.value
                if not isinstance(call, ast.Call):
                    valid = False
                    break
                if getattr(call.func, "id", None) != _ALPHA_HELPER_BY_INDEX[index]:
                    valid = False
                    break
                _validate_generated_call(call, index)
            if valid:
                matches.append((body, start))
    if len(matches) != 1:
        raise AlphaHookError("S46J_ALPHA_HELPER_PARENT_POSITION_CHANGED")

    body, start = matches[0]
    for index, statement in enumerate(_ALPHA_EXPECTED):
        replacement = copy.deepcopy(statement)
        ast.copy_location(replacement, body[start + index])
        body[start + index] = replacement
    ast.fix_missing_locations(result)
    return result


def _compile_function(node: ast.FunctionDef, *, namespace: dict[str, Any], path: Path) -> FunctionType:
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    local: dict[str, Any] = {}
    exec(compile(module, str(path), "exec", dont_inherit=True, optimize=0), namespace, local)
    function = local.get(node.name)
    if not isinstance(function, FunctionType) or function.__globals__ is not namespace:
        raise AlphaHookError("S46J_ALPHA_REFERENCE_COMPILE_FAILED")
    return function


def _reference_helpers(module: Any) -> tuple[dict[str, FunctionType], str, Path, dict[str, Any]]:
    path = Path(str(module.__file__)).resolve(strict=True)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    function = _find_function(tree, "propagate_one_pulse")
    body, start = _alpha_block(function)
    selected = body[start : start + len(_ALPHA_EXPECTED)]
    namespace = dict(vars(module))
    helpers: dict[str, FunctionType] = {}
    templates = (
        (ALPHA_CLEAN_SOURCE_HELPER, ("ion_source_prop",)),
        (ALPHA_RATIO_HELPER, ("ion_source_prop", "I", "I_floor")),
        (ALPHA_CLEAN_RESULT_HELPER, ("alpha_ion",)),
        (ALPHA_CLIP_HELPER, ("alpha_ion",)),
    )
    for index, ((name, arguments), statement) in enumerate(zip(templates, selected)):
        if index < 3:
            expression = copy.deepcopy(statement.value)
        else:
            expression = copy.deepcopy(statement.value)
        function_node = ast.FunctionDef(
            name=name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=argument) for argument in arguments],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=[
                ast.Return(value=expression)
                if index < 3
                else ast.Expr(value=expression),
            ],
            decorator_list=[],
            returns=None,
        )
        if index == 3:
            function_node.body.append(ast.Return(value=ast.Name(id="alpha_ion", ctx=ast.Load())))
        ast.copy_location(function_node, statement)
        helpers[name] = _compile_function(function_node, namespace=namespace, path=path)
    block_hash = _ast_hash(ast.Module(body=copy.deepcopy(selected), type_ignores=[]))
    return helpers, block_hash, path, namespace


def _scalar_only(value: Any, *, depth: int = 0) -> bool:
    if depth > 8:
        return False
    if isinstance(value, np.ndarray):
        return False
    if isinstance(value, Mapping):
        return all(_scalar_only(item, depth=depth + 1) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_scalar_only(item, depth=depth + 1) for item in value)
    return isinstance(value, (str, bytes, int, float, bool, type(None), np.generic))


def _scalar_projection(value: Any, *, depth: int = 0) -> Any:
    """Project drain evidence without retaining scientific arrays."""

    if depth > 8:
        return "<nested_value_omitted>"
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return _scalar_projection(value.item(), depth=depth + 1)
    if isinstance(value, np.ndarray):
        return "<non_scalar_value_omitted>"
    if isinstance(value, Mapping):
        return {
            str(key): _scalar_projection(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_scalar_projection(item, depth=depth + 1) for item in value]
    return "<non_scalar_value_omitted>"


class AlphaParallelHook(PowerVolumeHook):
    """Power hook plus four source-bound alpha helper replacements."""

    def __init__(
        self,
        pool: PowerVolumePool,
        *,
        alpha_pool: AlphaPool | None = None,
        capture_store: Any | None = None,
    ) -> None:
        if type(pool) is not PowerVolumePool:
            raise AlphaHookError("S46J_ALPHA_POWER_POOL_EXACT_TYPE_REQUIRED")
        if alpha_pool is not None and type(alpha_pool) is not AlphaPool:
            raise AlphaHookError("S46J_ALPHA_POOL_EXACT_TYPE_REQUIRED")
        if capture_store is not None and not all(
            callable(getattr(capture_store, name, None)) for name in ("before_call", "after_call")
        ):
            raise AlphaHookError("S46J_ALPHA_CAPTURE_STORE_REQUIRED")

        # The existing power hook remains the owner of the power component.
        # Alpha capture is intentionally private because the parent hook forbids
        # a capture store alongside a live power pool.
        super().__init__(pool, capture_store=None)
        self._alpha_pool = alpha_pool
        self._bound_alpha_pool = alpha_pool
        self._alpha_capture_store = capture_store
        self._bound_alpha_capture_store = capture_store
        self._alpha_capture_methods = {} if capture_store is None else {
            name: getattr(capture_store, name).__func__ for name in ("before_call", "after_call")
        }
        self._alpha_capture_accepts_inputs = False
        self._alpha_capture_accepts_outputs = False
        if capture_store is not None:
            before_signature = inspect.signature(capture_store.before_call)
            after_signature = inspect.signature(capture_store.after_call)
            self._alpha_capture_accepts_inputs = (
                "capture_inputs" in before_signature.parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in before_signature.parameters.values()
                )
            )
            self._alpha_capture_accepts_outputs = (
                "capture_inputs" in after_signature.parameters
                or "capture_outputs" in after_signature.parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in after_signature.parameters.values()
                )
            )
        self._alpha_propagate = importlib.import_module("KHz_filament.propagate")
        self._alpha_xp = getattr(self._alpha_propagate, "xp", None)
        if self._alpha_xp is not np:
            raise AlphaHookError("S46J_ALPHA_SOURCE_NUMPY_BACKEND_REQUIRED")
        self._alpha_original_propagate = getattr(self._alpha_propagate, "propagate_one_pulse", None)
        if not callable(self._alpha_original_propagate):
            raise AlphaHookError("S46J_ALPHA_PROPAGATE_SOURCE_MISSING")
        self._alpha_reference_helpers, self._alpha_block_hash, self._alpha_propagate_path, self._alpha_reference_namespace = _reference_helpers(
            self._alpha_propagate
        )
        self._alpha_source_identity = factory_source_identity(
            extra_paths=(
                __file__,
                Path(alpha_kernel.__file__).resolve(),
                self._alpha_propagate_path,
            )
        )
        self._alpha_source_fingerprint = copy.deepcopy(self._alpha_source_identity)
        self._alpha_source_seals = {
            "propagate_one_pulse": _seal(self._alpha_original_propagate),
        }
        self._alpha_reference_seals = {
            name: _seal(function) for name, function in self._alpha_reference_helpers.items()
        }
        self._alpha_backend = {
            name: getattr(np, name) for name in ("add", "divide", "nan_to_num", "maximum")
        }
        self._alpha_kernel_functions = {
            "clean_source": alpha_kernel.clean_source,
            "ratio": alpha_kernel.ratio,
            "clean_alpha": alpha_kernel.clean_alpha,
            "clip_alpha": alpha_kernel.clip_alpha,
            "require_float64_3d": alpha_kernel._require_float64_3d,
        }
        self._alpha_kernel_seals = {
            name: _seal(function) for name, function in self._alpha_kernel_functions.items()
        }
        self._alpha_pool_methods = {} if alpha_pool is None else {
            name: getattr(AlphaPool, name)
            for name in ("run_round", "take_records", "snapshot", "close", "set_failure_cleanup")
        }
        self._alpha_pool_method_seals = {
            name: _seal(function) for name, function in self._alpha_pool_methods.items()
        }
        self._alpha_runtime_helpers = self._make_alpha_helpers()
        self._alpha_helper_seals = {
            name: _seal(function) for name, function in self._alpha_runtime_helpers.items()
        }
        self._alpha_transform_seals = {
            "instrument": _seal(instrument_alpha_ast),
            "strip": _seal(strip_alpha_ast),
        }
        self._alpha_records: list[dict[str, Any]] = []
        self._alpha_record_capacity = ALPHA_MAX_RECORDS
        self._alpha_total_records = 0
        self._alpha_dropped_records = 0
        self._alpha_invalid_records = 0
        self._alpha_record_failures = 0
        self._alpha_close_result: dict[str, Any] | None = None
        self._alpha_receipt = self._make_alpha_receipt()
        self._alpha_receipt_fingerprint = copy.deepcopy(self._alpha_receipt)
        self._alpha_active_globals: Mapping[str, Any] | None = None
        self._alpha_failure_lock = threading.Lock()
        self._alpha_failure_notified = False
        self._alpha_failure_callback: Callable[[], Any] | None = None

    def _make_alpha_receipt(self) -> dict[str, Any]:
        return {
            "schema": ALPHA_HOOK_SCHEMA,
            "semantic_id": ALPHA_HOOK_SEMANTIC_ID,
            "source_site": ALPHA_SOURCE_SITE,
            "helper_names": dict(ALPHA_HELPER_NAMES),
            "source_ast_hash": self._alpha_block_hash,
            "source_identity": copy.deepcopy(self._alpha_source_identity),
            "evaluation": "PARALLEL_P24" if self._alpha_pool is not None else "ORIGINAL_REFERENCE",
            "alpha_pool_workers": None if self._alpha_pool is None else int(self._alpha_pool.count),
            "capture_enabled": self._alpha_capture_store is not None,
            "capture_contract": {
                "before": {
                    "clean_source": ("source_input",),
                    "ratio": ("intensity_input", "I_floor"),
                },
                "after": {
                    "clean_source": ("cleaned_source_output",),
                    "clip": ("final_alpha", "inplace"),
                },
                "arrays_are_transient_only": True,
            },
            "mathematical_changes": 0,
            "array_values_retained": False,
            "inverse_ast_layer": "ALPHA_THEN_POWER_ALIASES",
        }

    @staticmethod
    def _validate_interval(interval_index: Any) -> int | None:
        if interval_index is None:
            return None
        if type(interval_index) is not int or not 0 <= interval_index < 8:
            raise AlphaHookError("S46J_ALPHA_INTERVAL_DOMAIN")
        return int(interval_index)

    @staticmethod
    def _validate_source(value: Any, name: str, *, writable: bool = False) -> np.ndarray:
        try:
            return alpha_kernel._require_float64_3d(value, name, writable=writable)
        except BaseException as exc:
            if isinstance(exc, AlphaHookError):
                raise
            raise AlphaHookError(str(exc)) from exc

    def _verify_alpha_runtime(self, helper: str, interval_index: Any) -> int | None:
        interval = self._validate_interval(interval_index)
        if self._alpha_xp is not np or getattr(self._alpha_propagate, "xp", None) is not np:
            raise AlphaHookError("S46J_ALPHA_SOURCE_BACKEND_CHANGED")
        if self._alpha_pool is not self._bound_alpha_pool:
            raise AlphaHookError("S46J_ALPHA_POOL_BINDING_CHANGED")
        if self._alpha_pool is not None and self._alpha_pool.closed:
            raise AlphaHookError("S46J_ALPHA_RETIRED_POOL")
        if helper not in self._alpha_runtime_helpers:
            raise AlphaHookError("S46J_ALPHA_HELPER_REGISTRY_CHANGED")
        return interval

    def _capture_before(
        self,
        helper: str,
        interval: int | None,
        source: np.ndarray,
        *,
        capture_inputs: Mapping[str, Any] | None = None,
    ) -> Any:
        if self._alpha_capture_store is None:
            return None
        kwargs = dict(
            context=helper,
            interval_index=interval,
            source=source,
            source_site=ALPHA_SOURCE_SITE,
        )
        if capture_inputs is not None and self._alpha_capture_accepts_inputs:
            kwargs["capture_inputs"] = capture_inputs
        return self._alpha_capture_store.before_call(
            **kwargs,
        )

    def _capture_after(
        self,
        token: Any,
        source: np.ndarray,
        output: np.ndarray,
        *,
        capture_inputs: Mapping[str, Any] | None = None,
        capture_outputs: Mapping[str, Any] | None = None,
    ) -> None:
        if self._alpha_capture_store is not None:
            kwargs = dict(source=source, output=output)
            if self._alpha_capture_accepts_outputs:
                signature = inspect.signature(self._alpha_capture_store.after_call)
                accepts_kwargs = any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
                if "capture_inputs" in signature.parameters or accepts_kwargs:
                    kwargs["capture_inputs"] = capture_inputs
                if "capture_outputs" in signature.parameters or accepts_kwargs:
                    kwargs["capture_outputs"] = capture_outputs
            self._alpha_capture_store.after_call(token, **kwargs)

    def _drain_alpha_records(self, helper: str, interval: int | None) -> None:
        if self._alpha_pool is None:
            return
        rows = self._alpha_pool.take_records()
        self._alpha_total_records += len(rows)
        expected = {
            ALPHA_CLEAN_SOURCE_HELPER: ("ALPHA_CLEAN_SOURCE",),
            ALPHA_RATIO_HELPER: ("ALPHA_DENOM", "ALPHA_RATIO"),
            ALPHA_CLEAN_RESULT_HELPER: ("ALPHA_CLEAN_RESULT",),
            ALPHA_CLIP_HELPER: ("ALPHA_CLIP",),
        }[helper]
        error: str | None = None
        if len(self._alpha_records) + len(rows) > self._alpha_record_capacity:
            error = "S46J_ALPHA_ACTIVITY_CAPACITY_EXHAUSTED"
        elif tuple(
            row.get("label") if isinstance(row, Mapping) else None for row in rows
        ) != expected:
            error = "S46J_ALPHA_ACTIVITY_ORDER_CHANGED"
        for row in rows:
            scalar = _scalar_only(row)
            if not scalar and error is None:
                error = "S46J_ALPHA_ACTIVITY_ARRAY_RETAINED"
            if len(self._alpha_records) >= self._alpha_record_capacity:
                self._alpha_dropped_records += 1
                continue
            item = copy.deepcopy(row) if scalar else _scalar_projection(row)
            if not isinstance(item, dict):
                item = {"record": item}
            if not scalar:
                self._alpha_invalid_records += 1
                item["invalid_record"] = True
            item.update(
                component="alpha",
                helper=helper,
                context=helper,
                interval_index=interval,
                source=ALPHA_SOURCE_SITE,
            )
            if error is not None and error != "S46J_ALPHA_ACTIVITY_CAPACITY_EXHAUSTED":
                item["drain_error"] = error
            self._alpha_records.append(item)
        if error is not None:
            self._alpha_record_failures += 1
            raise AlphaHookError(error)

    def _invoke(
        self,
        helper: str,
        interval_index: Any,
        source: np.ndarray,
        operation: Callable[[], np.ndarray],
        *,
        capture_inputs: Mapping[str, Any] | None = None,
        capture_outputs: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        interval = self._verify_alpha_runtime(helper, interval_index)
        token = None
        try:
            token = self._capture_before(
                helper,
                interval,
                source,
                capture_inputs=capture_inputs,
            )
            result = operation()
            self._drain_alpha_records(helper, interval)
            resolved_outputs = None
            if capture_outputs is not None:
                resolved_outputs = {
                    name: (result if isinstance(value, str) and value == "__result__" else value)
                    for name, value in capture_outputs.items()
                }
            self._capture_after(
                token,
                source,
                result,
                capture_inputs=capture_inputs,
                capture_outputs=resolved_outputs,
            )
            return result
        except BaseException:
            try:
                self._notify_alpha_failure_once()
            except BaseException:
                pass
            raise

    def _make_alpha_helpers(self) -> dict[str, Callable[..., Any]]:
        owner = self

        def alpha_clean_source(source: Any, *, __s46j_interval_index__: Any = None) -> np.ndarray:
            value = owner._validate_source(source, "SOURCE")
            return owner._invoke(
                ALPHA_CLEAN_SOURCE_HELPER,
                __s46j_interval_index__,
                value,
                lambda: (
                    alpha_kernel.clean_source(value, pool=owner._alpha_pool, backend=owner._alpha_xp)
                    if owner._alpha_pool is not None
                    else owner._alpha_reference_helpers[ALPHA_CLEAN_SOURCE_HELPER](value)
                ),
                capture_inputs={"source_input": value},
                capture_outputs={"cleaned_source_output": "__result__"},
            )

        def alpha_ratio(
            source: Any,
            intensity: Any,
            I_floor: Any,
            *,
            __s46j_interval_index__: Any = None,
        ) -> np.ndarray:
            source_value = owner._validate_source(source, "CLEAN_SOURCE")
            intensity_value = owner._validate_source(intensity, "INTENSITY")
            if source_value.shape != intensity_value.shape:
                raise AlphaHookError("S46J_ALPHA_RATIO_SHAPE")
            if type(I_floor) is not float:
                raise AlphaHookError("S46J_ALPHA_IFLOOR_PYTHON_FLOAT")
            return owner._invoke(
                ALPHA_RATIO_HELPER,
                __s46j_interval_index__,
                source_value,
                lambda: (
                    alpha_kernel.ratio(
                        source_value,
                        intensity_value,
                        I_floor,
                        pool=owner._alpha_pool,
                        backend=owner._alpha_xp,
                    )
                    if owner._alpha_pool is not None
                    else owner._alpha_reference_helpers[ALPHA_RATIO_HELPER](
                        source_value, intensity_value, I_floor
                    )
                ),
                capture_inputs={
                    "intensity_input": intensity_value,
                    "I_floor": I_floor,
                },
            )

        def alpha_clean_result(alpha: Any, *, __s46j_interval_index__: Any = None) -> np.ndarray:
            value = owner._validate_source(alpha, "ALPHA")
            return owner._invoke(
                ALPHA_CLEAN_RESULT_HELPER,
                __s46j_interval_index__,
                value,
                lambda: (
                    alpha_kernel.clean_alpha(value, pool=owner._alpha_pool, backend=owner._alpha_xp)
                    if owner._alpha_pool is not None
                    else owner._alpha_reference_helpers[ALPHA_CLEAN_RESULT_HELPER](value)
                ),
            )

        def alpha_clip(alpha: Any, *, __s46j_interval_index__: Any = None) -> np.ndarray:
            value = owner._validate_source(alpha, "ALPHA", writable=True)
            return owner._invoke(
                ALPHA_CLIP_HELPER,
                __s46j_interval_index__,
                value,
                lambda: (
                    alpha_kernel.clip_alpha(value, pool=owner._alpha_pool, backend=owner._alpha_xp)
                    if owner._alpha_pool is not None
                    else owner._alpha_reference_helpers[ALPHA_CLIP_HELPER](value)
                ),
                capture_outputs={"final_alpha": "__result__", "inplace": True},
            )

        helpers = {
            ALPHA_CLEAN_SOURCE_HELPER: alpha_clean_source,
            ALPHA_RATIO_HELPER: alpha_ratio,
            ALPHA_CLEAN_RESULT_HELPER: alpha_clean_result,
            ALPHA_CLIP_HELPER: alpha_clip,
        }
        for name, helper in helpers.items():
            helper.__name__ = name
            helper.__qualname__ = name
            helper.__cpu_s46j_alpha_kind__ = "FIXED_ALPHA_P24_OR_REFERENCE"
            helper.__cpu_s46j_alpha_pool__ = self._alpha_pool
        return helpers

    def _notify_alpha_failure_once(self) -> None:
        with self._alpha_failure_lock:
            if self._alpha_failure_notified:
                return
            self._alpha_failure_notified = True
            callback = self._alpha_failure_callback
        if callback is not None:
            callback()

    def set_failure_cleanup(self, callback: Callable[[], Any] | None) -> None:
        if callback is not None and not callable(callback):
            raise AlphaHookError("S46J_ALPHA_FAILURE_CLEANUP_REQUIRED")
        self._alpha_failure_callback = callback
        with self._alpha_failure_lock:
            self._alpha_failure_notified = False
        shared = self._notify_alpha_failure_once if callback is not None else None
        # Route both the inherited power path and AlphaPool's notify path
        # through one hook gate; _invoke uses the same gate.
        super().set_failure_cleanup(shared)
        if self._alpha_pool is not None:
            self._alpha_pool.set_failure_cleanup(shared)

    def instrument(self, function_node: ast.FunctionDef) -> dict[str, Any]:
        result = super().instrument(function_node)
        result = dict(result, alpha=instrument_alpha_ast(function_node))
        return result

    def strip_generated_nodes(self, node: ast.AST) -> ast.AST:
        # The alpha layer is removed first, leaving the joint power AST for
        # PowerVolumeHook's existing inverse proof.
        return super().strip_generated_nodes(strip_alpha_ast(node))

    def runtime_helpers(self) -> dict[str, Callable[..., Any]]:
        return dict(super().runtime_helpers(), **self._alpha_runtime_helpers)

    def binding(self) -> dict[str, Any]:
        result = super().binding()
        result["alpha"] = copy.deepcopy(self._alpha_receipt)
        return result

    def bind_active(self, active_globals: Mapping[str, Any]) -> None:
        if not isinstance(active_globals, dict):
            raise AlphaHookError("S46J_ALPHA_ACTIVE_GLOBALS_REQUIRED")
        super().bind_active(active_globals)
        self._alpha_active_globals = active_globals

    def verify_binding(self, active_globals: Mapping[str, Any] | None = None) -> dict[str, Any]:
        result = super().verify_binding(active_globals)
        if self._alpha_receipt != self._alpha_receipt_fingerprint:
            raise AlphaHookError("S46J_ALPHA_RECEIPT_CHANGED")
        if factory_source_identity(
            extra_paths=(__file__, Path(alpha_kernel.__file__).resolve(), self._alpha_propagate_path)
        ) != self._alpha_source_fingerprint:
            raise AlphaHookError("S46J_ALPHA_SOURCE_IDENTITY_CHANGED")
        for name, seal in self._alpha_source_seals.items():
            verify_seal(seal)
            if getattr(self._alpha_propagate, name.rsplit(".", 1)[-1], None) is not seal["function"]:
                raise AlphaHookError("S46J_ALPHA_SOURCE_FUNCTION_CHANGED")
        current_helpers, current_hash, current_path, current_namespace = _reference_helpers(self._alpha_propagate)
        if current_path != self._alpha_propagate_path or current_hash != self._alpha_block_hash:
            raise AlphaHookError("S46J_ALPHA_SOURCE_AST_CHANGED")
        if (
            not isinstance(self._alpha_reference_namespace, dict)
            or self._alpha_reference_namespace.get("xp") is not self._alpha_xp
        ):
            raise AlphaHookError("S46J_ALPHA_REFERENCE_BACKEND_BINDING_CHANGED")
        if current_namespace.get("xp") is not self._alpha_xp:
            raise AlphaHookError("S46J_ALPHA_REFERENCE_BACKEND_CHANGED")
        for name, function in self._alpha_reference_helpers.items():
            verify_seal(self._alpha_reference_seals[name])
            if current_helpers[name].__code__.co_code != function.__code__.co_code:
                raise AlphaHookError("S46J_ALPHA_REFERENCE_CODE_CHANGED")
            if function.__globals__ is not self._alpha_reference_namespace:
                raise AlphaHookError("S46J_ALPHA_REFERENCE_NAMESPACE_CHANGED")
        for name, function in self._alpha_kernel_functions.items():
            actual_name = name if name != "require_float64_3d" else "_require_float64_3d"
            if getattr(alpha_kernel, actual_name, None) is not function:
                raise AlphaHookError("S46J_ALPHA_KERNEL_BINDING_CHANGED:" + name)
            verify_seal(self._alpha_kernel_seals[name])
        for name, function in self._alpha_backend.items():
            if getattr(self._alpha_xp, name, None) is not function:
                raise AlphaHookError("S46J_ALPHA_BACKEND_CHANGED:" + name)
        for name, seal in self._alpha_transform_seals.items():
            verify_seal(seal)
        for name, helper in self._alpha_runtime_helpers.items():
            verify_seal(self._alpha_helper_seals[name])
            if helper.__closure__ and any(_contains_array(cell.cell_contents) for cell in helper.__closure__):
                raise AlphaHookError("S46J_ALPHA_HELPER_CLOSURE_ARRAY:" + name)
        if self._alpha_pool is not self._bound_alpha_pool:
            raise AlphaHookError("S46J_ALPHA_POOL_BINDING_CHANGED")
        if self._alpha_pool is not None:
            if type(self._alpha_pool) is not AlphaPool:
                raise AlphaHookError("S46J_ALPHA_POOL_TYPE_CHANGED")
            for name, function in self._alpha_pool_methods.items():
                if getattr(AlphaPool, name) is not function:
                    raise AlphaHookError("S46J_ALPHA_POOL_METHOD_CHANGED:" + name)
                bound = getattr(self._alpha_pool, name, None)
                if (
                    getattr(bound, "__self__", None) is not self._alpha_pool
                    or getattr(bound, "__func__", None) is not function
                ):
                    raise AlphaHookError("S46J_ALPHA_POOL_INSTANCE_METHOD_CHANGED:" + name)
                verify_seal(self._alpha_pool_method_seals[name])
        if self._alpha_capture_store is not self._bound_alpha_capture_store:
            raise AlphaHookError("S46J_ALPHA_CAPTURE_STORE_CHANGED")
        for name, function in self._alpha_capture_methods.items():
            if getattr(self._alpha_capture_store, name).__func__ is not function:
                raise AlphaHookError("S46J_ALPHA_CAPTURE_METHOD_CHANGED:" + name)
        if active_globals is not None:
            if not isinstance(active_globals, dict):
                raise AlphaHookError("S46J_ALPHA_ACTIVE_GLOBALS_INVALID")
            for name, helper in self.runtime_helpers().items():
                if active_globals.get(name) is not helper:
                    raise AlphaHookError("S46J_ALPHA_ACTIVE_HELPER_CHANGED:" + name)
            if active_globals.get("xp") is not self._alpha_xp:
                raise AlphaHookError("S46J_ALPHA_ACTIVE_BACKEND_CHANGED")
            self._alpha_active_globals = active_globals
        result["alpha"] = {
            "schema": ALPHA_HOOK_SCHEMA + ".binding.v1",
            "active_globals_bound": active_globals is not None or self._alpha_active_globals is not None,
            "alpha_pool_workers": None if self._alpha_pool is None else int(self._alpha_pool.count),
            "runtime_helpers": tuple(sorted(self._alpha_runtime_helpers)),
            "source_site": ALPHA_SOURCE_SITE,
            "mathematical_changes": 0,
            "array_values_retained": False,
        }
        return result

    def take_alpha_records(self) -> list[dict[str, Any]]:
        rows = self._alpha_records
        self._alpha_records = []
        return copy.deepcopy(rows)

    def alpha_snapshot(self) -> dict[str, Any] | None:
        if self._alpha_pool is None:
            return None
        result = copy.deepcopy(self._alpha_pool.snapshot())
        result.update(
            {
                "hook_total_records": int(self._alpha_total_records),
                "hook_dropped_records": int(self._alpha_dropped_records),
                "hook_invalid_records": int(self._alpha_invalid_records),
                "hook_record_failures": int(self._alpha_record_failures),
                "hook_pending_records": len(self._alpha_records),
                "records_capacity": self._alpha_record_capacity,
                "array_values_retained": False,
            }
        )
        return result

    def close(self) -> dict[str, Any] | None:
        primary: BaseException | None = None
        result: dict[str, Any] | None = None
        try:
            result = super().close()
        except BaseException as exc:
            primary = exc
        if self._alpha_pool is not None:
            try:
                self._alpha_close_result = copy.deepcopy(self._alpha_pool.close())
            except BaseException as exc:
                if primary is None:
                    primary = exc
                else:
                    try:
                        primary.add_note("S46J_ALPHA_CLOSE_SECONDARY:" + type(exc).__name__)
                    except BaseException:
                        pass
        if primary is not None:
            raise primary
        return result


__all__ = [
    "ALPHA_CLEAN_RESULT_HELPER",
    "ALPHA_CLEAN_SOURCE_HELPER",
    "ALPHA_CLIP_HELPER",
    "ALPHA_HELPER_NAMES",
    "ALPHA_HOOK_SCHEMA",
    "ALPHA_HOOK_SEMANTIC_ID",
    "ALPHA_INTERVAL_KW",
    "ALPHA_RATIO_HELPER",
    "ALPHA_SOURCE_SITE",
    "AlphaHookError",
    "AlphaParallelHook",
    "instrument_alpha_ast",
    "strip_alpha_ast",
]
