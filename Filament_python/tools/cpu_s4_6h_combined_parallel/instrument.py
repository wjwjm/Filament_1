"""Joint source-bound Raman-stage compiler for the S4-6H candidate.

The existing S4-6E and S4-6D installers each require the untouched Raman
stage.  This module therefore uses their already-validated AST transforms as
components of one compilation, and proves that removing both component
transformations restores the original stage exactly.  It deliberately does
not own the C/G2 nonlinear callsite; that source-bound seal remains in
``cpu_s4_6e_array_propagation.c_candidate`` and is attached once by the
factory.
"""

from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from types import FunctionType
from typing import Any, Callable, Mapping

from tools.cpu_s4_6d_eq10_propagation.instrument import (
    HELPER_NAMES as EQ10_HELPER_NAMES,
    instrument_eq10_stage,
    strip_generated_nodes as strip_eq10_nodes,
)
from tools.cpu_s4_6e_array_propagation.instrument import (
    HELPER_NAMES as INTENSITY_HELPER_NAMES,
    instrument_intensity_stage,
    strip_generated_nodes as strip_intensity_nodes,
)


INSTRUMENT_SCHEMA = "filament.cpu_s4_6h.combined_parallel.instrument.v1"
JOINT_GUARD_NAME = "__cpu_s46h_joint_stage_guard"
COMPONENT_HELPER_NAMES = {
    "intensity": dict(INTENSITY_HELPER_NAMES),
    "eq10": dict(EQ10_HELPER_NAMES),
}


class InstrumentationBindingError(RuntimeError):
    """The frozen Raman stage or joint binding failed closed."""


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise InstrumentationBindingError(reason)


def _ast_hash(node: ast.AST) -> str:
    return hashlib.sha256(
        ast.dump(node, include_attributes=False).encode("utf-8")
    ).hexdigest()


def _code_identity(function: Callable[..., Any]) -> str:
    code = getattr(function, "__code__", None)
    _require(code is not None, "S46H_CALLABLE_CODE_MISSING")

    # marshal's representation of code objects can change when string
    # constants become interned between the compile and verification passes.
    # Hash the immutable code fields through an explicit tagged serializer so
    # a valid function cannot fail its seal merely because another compiler
    # pass ran in the meantime.
    def stable(value: Any) -> Any:
        if isinstance(value, type(code)):
            return (
                "code",
                value.co_argcount,
                value.co_posonlyargcount,
                value.co_kwonlyargcount,
                value.co_nlocals,
                value.co_stacksize,
                value.co_flags,
                stable(value.co_code),
                stable(value.co_consts),
                stable(value.co_names),
                stable(value.co_varnames),
                stable(value.co_freevars),
                stable(value.co_cellvars),
                stable(value.co_filename),
                stable(value.co_name),
                stable(getattr(value, "co_qualname", "")),
                value.co_firstlineno,
                stable(value.co_linetable),
                stable(value.co_exceptiontable),
            )
        if value is None or isinstance(value, (bool, int, float, complex)):
            return (type(value).__name__, repr(value))
        if isinstance(value, str):
            return ("str", value)
        if isinstance(value, bytes):
            return ("bytes", value.hex())
        if isinstance(value, tuple):
            return ("tuple", tuple(stable(item) for item in value))
        return (type(value).__name__, repr(value))

    return hashlib.sha256(repr(stable(code)).encode("utf-8")).hexdigest()


def strip_combined_generated_nodes(stage: ast.FunctionDef) -> ast.FunctionDef:
    """Strip the one joint guard and both component helper families."""

    result = copy.deepcopy(stage)

    class StripJointGuard(ast.NodeTransformer):
        def visit_Expr(self, node: ast.Expr) -> ast.AST | None:
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == JOINT_GUARD_NAME
            ):
                _require(not node.value.args and not node.value.keywords, "S46H_JOINT_GUARD_SHAPE")
                return None
            return self.generic_visit(node)

    result = StripJointGuard().visit(result)
    _require(isinstance(result, ast.FunctionDef), "S46H_STRIP_RESULT_INVALID")
    # B's stripper restores the four Eq10 expressions and leaves A's helper
    # call intact; A's stripper then restores the original intensity AST.
    result = strip_eq10_nodes(result)
    result = strip_intensity_nodes(result)
    _require(isinstance(result, ast.FunctionDef), "S46H_STRIP_FUNCTION_INVALID")
    ast.fix_missing_locations(result)
    return result


def instrument_combined_stage(stage: ast.FunctionDef) -> dict[str, Any]:
    """Apply the validated A and B transforms once to one stage AST.

    The operation order is fixed (intensity, then Eq10) and is part of the
    receipt.  The returned receipt proves that stripping both component
    transforms returns the original AST, while preserving every unrelated
    node in the Raman stage.
    """

    _require(isinstance(stage, ast.FunctionDef), "S46H_STAGE_FUNCTION_REQUIRED")
    baseline = copy.deepcopy(stage)
    intensity_receipt = instrument_intensity_stage(stage)
    eq10_receipt = instrument_eq10_stage(stage)
    stripped = strip_combined_generated_nodes(stage)
    _require(
        ast.dump(stripped, include_attributes=False)
        == ast.dump(baseline, include_attributes=False),
        "S46H_STRIPPED_AST_MISMATCH",
    )
    return {
        "schema": INSTRUMENT_SCHEMA,
        "source_ast_sha256": _ast_hash(baseline),
        "instrumented_ast_sha256": _ast_hash(stage),
        "stripped_ast_sha256": _ast_hash(stripped),
        "component_order": ("intensity", "eq10"),
        "intensity": copy.deepcopy(intensity_receipt),
        "eq10": copy.deepcopy(eq10_receipt),
        "helper_names": copy.deepcopy(COMPONENT_HELPER_NAMES),
        "replacement_counts": {
            "intensity": 1,
            "eq10": 4,
        },
        "pointwise_segments": [
            "complex_absolute",
            "square",
            "coefficient_scale",
            "spectral_multiply",
            "ifft_real_divide",
            "response_divide",
            "product",
        ],
        "fft_calls": int(eq10_receipt["fft_calls"]),
        "ifft_calls": int(eq10_receipt["ifft_calls"]),
        "sum_calls": int(eq10_receipt["sum_calls"]),
        "array_values_retained": False,
        "private_guard_before_helper_calls": True,
        "inverse_proof": "strip_joint_guard_eq10_then_intensity_restores_original_stage_ast",
        "scientific_expression_changes": 0,
    }


def insert_joint_guard(stage: ast.FunctionDef) -> dict[str, Any]:
    """Insert exactly one entry guard after an optional function docstring."""

    _require(isinstance(stage, ast.FunctionDef), "S46H_STAGE_FUNCTION_REQUIRED")
    calls = [
        node
        for node in ast.walk(stage)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == JOINT_GUARD_NAME
    ]
    _require(not calls, "S46H_JOINT_GUARD_ALREADY_PRESENT")
    guard = ast.Expr(
        value=ast.Call(
            func=ast.Name(id=JOINT_GUARD_NAME, ctx=ast.Load()),
            args=[],
            keywords=[],
        )
    )
    index = 1 if ast.get_docstring(stage) is not None else 0
    stage.body.insert(index, ast.copy_location(guard, stage.body[index] if stage.body else stage))
    ast.fix_missing_locations(stage)
    return {
        "name": JOINT_GUARD_NAME,
        "insert_index": int(index),
        "argument_count": 0,
    }


def _closure_cells(function: Callable[..., Any]) -> tuple[dict[str, Any], ...]:
    cells: list[dict[str, Any]] = []
    for cell in getattr(function, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError as exc:  # pragma: no cover - defensive closure case
            raise InstrumentationBindingError("S46H_SEAL_EMPTY_CLOSURE_CELL") from exc
        cells.append({"cell": cell, "value": value, "value_id": int(id(value))})
    return tuple(cells)


def _seal(
    function: Callable[..., Any],
    *,
    bindings: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require(callable(function), "S46H_SEAL_FUNCTION_REQUIRED")
    return {
        "function": function,
        "code": function.__code__,
        "code_identity": _code_identity(function),
        "globals": function.__globals__,
        "closure": function.__closure__,
        "closure_cells": _closure_cells(function),
        "defaults": function.__defaults__,
        "kwdefaults_object": function.__kwdefaults__,
        "kwdefaults": dict(function.__kwdefaults__ or {}),
        "bindings": None if bindings is None else dict(bindings),
        "metadata": None if metadata is None else copy.deepcopy(dict(metadata)),
    }


def verify_seal(seal: Mapping[str, Any]) -> None:
    """Verify code, namespace, closure, defaults and metadata identity."""

    function = seal.get("function")
    _require(callable(function), "S46H_SEAL_FUNCTION_MISSING")
    _require(function.__code__ is seal.get("code"), "S46H_SEAL_CODE_CHANGED")
    _require(_code_identity(function) == seal.get("code_identity"), "S46H_SEAL_CODE_IDENTITY_CHANGED")
    _require(function.__globals__ is seal.get("globals"), "S46H_SEAL_GLOBALS_CHANGED")
    _require(function.__closure__ is seal.get("closure"), "S46H_SEAL_CLOSURE_CHANGED")
    actual_cells = _closure_cells(function)
    expected_cells = seal.get("closure_cells")
    _require(isinstance(expected_cells, tuple), "S46H_SEAL_CLOSURE_CELLS_MISSING")
    _require(len(actual_cells) == len(expected_cells), "S46H_SEAL_CLOSURE_SIZE_CHANGED")
    for actual, expected in zip(actual_cells, expected_cells):
        _require(actual["cell"] is expected.get("cell"), "S46H_SEAL_CLOSURE_CELL_CHANGED")
        _require(actual["value"] is expected.get("value"), "S46H_SEAL_CLOSURE_VALUE_CHANGED")
        _require(actual["value_id"] == expected.get("value_id"), "S46H_SEAL_CLOSURE_VALUE_ID_CHANGED")
    _require(function.__defaults__ is seal.get("defaults"), "S46H_SEAL_DEFAULTS_CHANGED")
    _require(function.__kwdefaults__ is seal.get("kwdefaults_object"), "S46H_SEAL_KWDEFAULTS_OBJECT_CHANGED")
    current_kw = function.__kwdefaults__ or {}
    expected_kw = seal.get("kwdefaults", {})
    _require(
        current_kw.keys() == expected_kw.keys()
        and all(current_kw[name] is value for name, value in expected_kw.items()),
        "S46H_SEAL_KWDEFAULTS_CHANGED",
    )
    bindings = seal.get("bindings")
    if bindings is not None:
        _require(isinstance(bindings, Mapping), "S46H_SEAL_BINDINGS_INVALID")
        for expected in bindings.values():
            _require(
                any(cell["value"] is expected for cell in actual_cells),
                "S46H_SEAL_BINDING_NOT_IN_CLOSURE",
            )
    metadata = seal.get("metadata")
    if metadata is not None:
        _require(isinstance(metadata, Mapping), "S46H_SEAL_METADATA_INVALID")
        for name, expected in metadata.items():
            _require(getattr(function, name, None) == expected, "S46H_SEAL_METADATA_CHANGED")


def _compile_function(
    stage: ast.FunctionDef,
    *,
    namespace: dict[str, Any],
    source_path: Path,
) -> FunctionType:
    module = ast.fix_missing_locations(ast.Module(body=[stage], type_ignores=[]))
    local: dict[str, Any] = {}
    exec(compile(module, str(source_path), "exec"), namespace, local)
    function = local.get(stage.name) or namespace.get(stage.name)
    _require(isinstance(function, FunctionType), "S46H_STAGE_COMPILE_FAILED")
    _require(function.__globals__ is namespace, "S46H_STAGE_GLOBAL_NAMESPACE_CHANGED")
    return function


def compile_combined_stage(
    stage: ast.FunctionDef,
    *,
    namespace: dict[str, Any],
    source_path: Path,
    entry_guard: Callable[[], None],
) -> tuple[FunctionType, dict[str, Any]]:
    """Transform, guard, compile and seal one joint Raman stage."""

    baseline = copy.deepcopy(stage)
    receipt = instrument_combined_stage(stage)
    namespace[JOINT_GUARD_NAME] = entry_guard
    guard_receipt = insert_joint_guard(stage)
    stripped = strip_combined_generated_nodes(stage)
    _require(
        ast.dump(stripped, include_attributes=False)
        == ast.dump(baseline, include_attributes=False),
        "S46H_ENTRY_GUARD_STRIP_PROOF",
    )
    compiled = _compile_function(stage, namespace=namespace, source_path=source_path)
    compiled.__joint_source_ast_sha256__ = receipt["source_ast_sha256"]
    compiled.__joint_instrumented_ast_sha256__ = _ast_hash(stage)
    compiled.__joint_stripped_ast_sha256__ = _ast_hash(stripped)
    compiled.__joint_guard_name__ = JOINT_GUARD_NAME
    compiled.__joint_instrumentation__ = copy.deepcopy(
        {
            **receipt,
            "instrumented_ast_sha256": _ast_hash(stage),
            "stripped_ast_sha256": _ast_hash(stripped),
            "entry_guard": copy.deepcopy(guard_receipt),
            "compiled_code_identity": _code_identity(compiled),
        }
    )
    compiled.__joint_function_seal__ = _seal(
        compiled,
        metadata={
            "__joint_source_ast_sha256__": receipt["source_ast_sha256"],
            "__joint_instrumented_ast_sha256__": _ast_hash(stage),
            "__joint_stripped_ast_sha256__": _ast_hash(stripped),
            "__joint_guard_name__": JOINT_GUARD_NAME,
        },
    )
    return compiled, copy.deepcopy(compiled.__joint_instrumentation__)


def verify_compiled_stage(
    stage: Callable[..., Any],
    *,
    seal: Mapping[str, Any],
    namespace: Mapping[str, Any],
    entry_guard: Callable[..., Any],
) -> dict[str, Any]:
    """Verify the compiled joint function and its private namespace slots."""

    _require(callable(stage), "S46H_COMPILED_STAGE_MISSING")
    verify_seal(seal)
    _require(stage.__globals__ is namespace, "S46H_COMPILED_STAGE_GLOBALS_CHANGED")
    _require(namespace.get(JOINT_GUARD_NAME) is entry_guard, "S46H_STAGE_GUARD_SLOT_CHANGED")
    _require(getattr(stage, "__joint_guard_name__", None) == JOINT_GUARD_NAME, "S46H_STAGE_GUARD_METADATA_CHANGED")
    return {
        "schema": INSTRUMENT_SCHEMA + ".compiled_binding",
        "code_identity": _code_identity(stage),
        "globals_bound": True,
        "entry_guard_bound": True,
        "seal_verified": True,
    }


# Explicit aliases keep the private API readable to callers that refer to the
# operation as a joint stage rather than a combined stage.
instrument_joint_stage = instrument_combined_stage
compile_joint_stage = compile_combined_stage
strip_joint_generated_nodes = strip_combined_generated_nodes
instrument_stage = instrument_combined_stage
compile_stage = compile_combined_stage


__all__ = [
    "COMPONENT_HELPER_NAMES",
    "INSTRUMENT_SCHEMA",
    "InstrumentationBindingError",
    "JOINT_GUARD_NAME",
    "_ast_hash",
    "_code_identity",
    "_compile_function",
    "_seal",
    "compile_combined_stage",
    "compile_joint_stage",
    "compile_stage",
    "insert_joint_guard",
    "instrument_combined_stage",
    "instrument_joint_stage",
    "instrument_stage",
    "strip_joint_generated_nodes",
    "strip_combined_generated_nodes",
    "verify_compiled_stage",
    "verify_seal",
]
