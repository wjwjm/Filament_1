"""Source-bound replacement of the frozen Raman intensity expression."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import FunctionType
from typing import Any, Callable, Mapping

import numpy as np
import scipy.fft as scipy_fft

from .pool import IntensityPool


INSTRUMENT_SCHEMA = "filament.cpu_s4_6e.intensity_instrument.v1"
GUARD_NAME = "__cpu_s46e_stage_guard"
HELPER_NAMES = {"intensity": "__cpu_s46e_intensity"}
CPU_XP_NAMESPACE_TYPE = "_XPNamespace"
CPU_XP_MODULE_SUFFIX = "cpu_p1r2_j2"
CPU_FFT_DISPATCH_TYPE = "_ScipyDispatch"


class InstrumentationBindingError(RuntimeError):
    """The frozen stage could not be transformed or verified."""


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise InstrumentationBindingError(reason)


def validate_cpu_backend(xp: Any, *, expected_workers: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Validate the accepted NumPy proxy and its fixed FFT worker pair."""

    _require(type(xp).__name__ == CPU_XP_NAMESPACE_TYPE, "A_CPU_XP_NAMESPACE_TYPE")
    xp_module = str(type(xp).__module__).replace("\\", ".")
    _require(xp_module.endswith(CPU_XP_MODULE_SUFFIX), "A_CPU_XP_NAMESPACE_MODULE")
    underlying = getattr(xp, "xp", None)
    _require(underlying is np, "A_CPU_BACKEND_REQUIRED")
    _require(getattr(xp, "abs", None) is np.absolute, "A_CPU_ABS_BINDING")
    fft_namespace = getattr(xp, "fft", None)
    _require(type(fft_namespace).__name__ == "_FFTNamespace", "A_CPU_FFT_NAMESPACE_TYPE")
    fft_module = str(type(fft_namespace).__module__).replace("\\", ".")
    _require(fft_module.endswith(CPU_XP_MODULE_SUFFIX), "A_CPU_FFT_NAMESPACE_MODULE")
    dispatch = getattr(fft_namespace, "dispatch", None)
    _require(type(dispatch).__name__ == CPU_FFT_DISPATCH_TYPE, "A_CPU_FFT_DISPATCH_TYPE")
    dispatch_module = str(type(dispatch).__module__).replace("\\", ".")
    _require(dispatch_module.endswith(CPU_XP_MODULE_SUFFIX), "A_CPU_FFT_DISPATCH_MODULE")
    _require(getattr(dispatch, "numpy_fft", None) is np.fft, "A_CPU_NUMPY_FFT_BINDING")
    _require(getattr(dispatch, "scipy_fft", None) is scipy_fft, "A_CPU_SCIPY_FFT_BINDING")
    descriptor = getattr(dispatch, "descriptor", None)
    _require(isinstance(descriptor, dict) and set(descriptor) == {"temporal_workers", "spatial_workers"}, "A_CPU_FFT_DESCRIPTOR")
    descriptor = {name: int(value) for name, value in descriptor.items()}
    if expected_workers is not None:
        expected = {"temporal_workers": int(expected_workers["temporal_workers"]), "spatial_workers": int(expected_workers["spatial_workers"])}
        _require(descriptor == expected, "A_CPU_FFT_WORKERS_MISMATCH")
    return {
        "xp_namespace_type": CPU_XP_NAMESPACE_TYPE,
        "xp_namespace_module": xp_module,
        "fft_namespace_type": type(fft_namespace).__name__,
        "fft_namespace_module": fft_module,
        "dispatch_type": CPU_FFT_DISPATCH_TYPE,
        "dispatch_module": dispatch_module,
        "fft_workers": dict(descriptor),
        "underlying_backend": "numpy",
        "xp_object": xp,
        "dispatch_object": dispatch,
    }


def _code_identity(function: Callable[..., Any]) -> str:
    import hashlib
    import marshal

    code = getattr(function, "__code__", None)
    _require(code is not None, "A_CALLABLE_CODE_MISSING")
    return hashlib.sha256(marshal.dumps(code)).hexdigest()


def _function_node(stage: ast.FunctionDef) -> ast.Assign:
    matches = [
        node
        for node in ast.walk(stage)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "intensity"
    ]
    _require(len(matches) == 1, "A_INTENSITY_ASSIGNMENT_COUNT")
    return matches[0]


def _is_call_path(node: ast.AST, path: tuple[str, ...]) -> bool:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    names = [node.func.attr]
    value: ast.AST = node.func.value
    while isinstance(value, ast.Attribute):
        names.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        names.append(value.id)
    return tuple(reversed(names)) == path


def _call(name: str, args: list[ast.AST], template: ast.AST) -> ast.Call:
    node = ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=args, keywords=[])
    return ast.copy_location(node, template)


def instrument_intensity_stage(stage: ast.FunctionDef) -> dict[str, Any]:
    """Replace exactly the frozen ``abs -> square -> scale`` expression."""

    _require(isinstance(stage, ast.FunctionDef), "A_STAGE_FUNCTION_REQUIRED")
    baseline = copy.deepcopy(stage)
    assignment = _function_node(stage)
    value = assignment.value
    _require(isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "astype", "A_INTENSITY_ASTYPE_SHAPE")
    _require(len(value.args) == 1 and len(value.keywords) == 1 and value.keywords[0].arg == "copy" and isinstance(value.keywords[0].value, ast.Constant) and value.keywords[0].value.value is False, "A_INTENSITY_ASTYPE_ARGUMENTS")
    core = value.func.value
    _require(isinstance(core, ast.BinOp) and isinstance(core.op, ast.Mult), "A_INTENSITY_SCALE_SHAPE")
    coefficient, power = core.left, core.right
    _require(isinstance(power, ast.BinOp) and isinstance(power.op, ast.Pow) and isinstance(power.right, ast.Constant) and power.right.value == 2, "A_INTENSITY_SQUARE_SHAPE")
    abs_call = power.left
    _require(_is_call_path(abs_call, ("xp", "abs")) and len(abs_call.args) == 1, "A_INTENSITY_ABS_SHAPE")
    expression = _call(HELPER_NAMES["intensity"], [abs_call.args[0], coefficient, value.args[0]], value)
    assignment.value = expression
    ast.fix_missing_locations(stage)
    stripped = strip_generated_nodes(stage)
    _require(ast.dump(stripped, include_attributes=False) == ast.dump(baseline, include_attributes=False), "A_STRIPPED_AST_MISMATCH")
    return {
        "schema": INSTRUMENT_SCHEMA,
        "source_ast_sha256": _ast_hash(baseline),
        "instrumented_ast_sha256": _ast_hash(stage),
        "stripped_ast_sha256": _ast_hash(stripped),
        "helper_names": dict(HELPER_NAMES),
        "replacement_counts": {"intensity": 1, "abs": 1, "square": 1, "scale": 1},
        "pointwise_segments": ["complex_absolute", "square", "coefficient_scale"],
        "fft_calls": 0,
        "ifft_calls": 0,
        "sum_calls": 0,
        "array_values_retained": False,
        "private_guard_before_helper_calls": True,
        "strip_proof": "generated_helper_removed_to_original_intensity_ast",
    }


def strip_generated_nodes(stage: ast.FunctionDef) -> ast.FunctionDef:
    result = copy.deepcopy(stage)

    class Stripper(ast.NodeTransformer):
        def visit_Expr(self, node: ast.Expr):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == GUARD_NAME:
                _require(not node.value.args and not node.value.keywords, "A_STAGE_GUARD_SHAPE")
                return None
            return self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> ast.AST:
            if isinstance(node.func, ast.Name) and node.func.id == HELPER_NAMES["intensity"]:
                _require(len(node.args) == 3 and not node.keywords, "A_STRIP_HELPER_ARGS")
                abs_call = ast.Call(
                    func=ast.Attribute(value=ast.Name(id="xp", ctx=ast.Load()), attr="abs", ctx=ast.Load()),
                    args=[self.visit(node.args[0])],
                    keywords=[],
                )
                power = ast.BinOp(left=abs_call, op=ast.Pow(), right=ast.Constant(value=2))
                core = ast.BinOp(left=self.visit(node.args[1]), op=ast.Mult(), right=power)
                return ast.Call(
                    func=ast.Attribute(value=core, attr="astype", ctx=ast.Load()),
                    args=[self.visit(node.args[2])],
                    keywords=[ast.keyword(arg="copy", value=ast.Constant(value=False))],
                )
            return self.generic_visit(node)

    restored = Stripper().visit(result)
    ast.fix_missing_locations(restored)
    return restored


def _ast_hash(node: ast.AST) -> str:
    import hashlib

    return hashlib.sha256(ast.dump(node, include_attributes=False).encode("utf-8")).hexdigest()


def _compile_function(stage: ast.FunctionDef, *, namespace: dict[str, Any], source_path: Path) -> FunctionType:
    module = ast.fix_missing_locations(ast.Module(body=[stage], type_ignores=[]))
    local: dict[str, Any] = {}
    exec(compile(module, str(source_path), "exec"), namespace, local)
    function = local.get(stage.name) or namespace.get(stage.name)
    _require(isinstance(function, FunctionType), "A_STAGE_COMPILE_FAILED")
    return function


def _seal(function: Callable[..., Any]) -> dict[str, Any]:
    _require(callable(function), "A_SEAL_CALLABLE_REQUIRED")
    closure = getattr(function, "__closure__", None) or ()
    return {
        "function": function,
        "code": function.__code__,
        "code_identity": _code_identity(function),
        "globals": function.__globals__,
        "closure": function.__closure__,
        "closure_cells": tuple(cell.cell_contents for cell in closure),
        "defaults": function.__defaults__,
        "kwdefaults_object": function.__kwdefaults__,
        "kwdefaults": dict(function.__kwdefaults__ or {}),
    }


def verify_seal(seal: Mapping[str, Any]) -> None:
    function = seal.get("function")
    _require(callable(function), "A_SEAL_FUNCTION_MISSING")
    _require(function.__code__ is seal.get("code"), "A_SEAL_CODE_CHANGED")
    _require(function.__globals__ is seal.get("globals"), "A_SEAL_GLOBALS_CHANGED")
    closure = getattr(function, "__closure__", None) or ()
    expected = seal.get("closure_cells", ())
    _require(len(closure) == len(expected), "A_SEAL_CLOSURE_SIZE")
    _require(all(cell.cell_contents is value for cell, value in zip(closure, expected)), "A_SEAL_CLOSURE_CHANGED")
    _require(function.__defaults__ is seal.get("defaults"), "A_SEAL_DEFAULTS_CHANGED")
    _require(function.__kwdefaults__ is seal.get("kwdefaults_object"), "A_SEAL_KWDEFAULTS_OBJECT_CHANGED")
    current, expected_kw = function.__kwdefaults__ or {}, seal.get("kwdefaults", {})
    _require(current.keys() == expected_kw.keys() and all(current[key] is value for key, value in expected_kw.items()), "A_SEAL_KWDEFAULTS_CHANGED")


def _helpers(pool: IntensityPool, *, guard: Callable[[], None] | None = None) -> dict[str, Callable[..., Any]]:
    if not isinstance(pool, IntensityPool):
        raise InstrumentationBindingError("A_POOL_TYPE")
    if guard is None:
        guard = lambda: None

    def intensity(E: Any, coefficient: Any, rdtype: Any) -> Any:
        guard()
        magnitude = np.empty(E.shape, dtype=np.float64)
        pool.ufunc("A_ABS", np.absolute, (E,), magnitude)
        squared = np.empty(E.shape, dtype=np.float64)
        pool.ufunc("A_SQUARE", np.square, (magnitude,), squared)
        del magnitude
        result = np.empty(E.shape, dtype=np.float64)
        pool.ufunc("A_SCALE", np.multiply, (coefficient, squared), result)
        return result.astype(rdtype, copy=False)

    intensity.__name__ = HELPER_NAMES["intensity"]
    intensity.__cpu_s46e_kind__ = "FIXED_INTENSITY_POINTWISE_HELPER"
    intensity.__cpu_s46e_pool__ = pool
    return {HELPER_NAMES["intensity"]: intensity}


def install_stage(state: Any, pool: IntensityPool) -> dict[str, Any]:
    """Install the three-operation clone in the existing private R8 state."""

    r8_state = getattr(state, "r8_state", None)
    r8_stage = getattr(r8_state, "candidate_stage_inner", None)
    r8_stage_before = r8_stage
    original = getattr(state, "_internal_stage_inner", None)
    stage_owner = state
    stage_attribute = "_internal_stage_inner"
    if not callable(original) and callable(r8_stage):
        original = r8_stage
        stage_owner = r8_state
        stage_attribute = "candidate_stage_inner"
    _require(callable(original), "A_ORIGINAL_STAGE_MISSING")
    candidate_namespace = getattr(state, "_candidate_namespace", None)
    _require(isinstance(candidate_namespace, dict), "A_CANDIDATE_NAMESPACE_MISSING")
    s45a = __import__("tools.cpu_s4_5a_l1_residual.instrument", fromlist=["unused"])
    builder = getattr(s45a, "_expected_transformed_node", None)
    identity = getattr(s45a, "_code_identity", None)
    _require(callable(builder) and callable(identity), "A_SOURCE_AST_SEAM_MISSING")
    source_path, stage_node, expected = builder(name="isaacs_raman_stage", transform_name="_transform_stage", candidate_namespace=candidate_namespace)
    _require(identity(expected) == identity(original), "A_R8_STAGE_SOURCE_MISMATCH")
    baseline_globals = getattr(original, "__globals__", None)
    _require(isinstance(baseline_globals, dict), "A_STAGE_GLOBALS_MISSING")
    xp = baseline_globals.get("xp")
    backend = validate_cpu_backend(xp, expected_workers=getattr(state, "_cpu_s46e_expected_backend_workers", None))
    _require(baseline_globals.get("_np") is np, "A_CPU_NUMPY_ALIAS_CHANGED")
    backend_metadata = {key: value for key, value in backend.items() if key not in {"xp_object", "dispatch_object"}}
    active_stage_ref: dict[str, Any] = {"stage": None}

    def guard_before_helper() -> None:
        active = active_stage_ref.get("stage")
        if active is None or getattr(stage_owner, stage_attribute, None) is not active:
            raise InstrumentationBindingError("A_ACTIVE_STAGE_CHANGED")
        if getattr(state, "_cpu_s46e_stage", None) is not active:
            raise InstrumentationBindingError("A_ACTIVE_STAGE_STATE_CHANGED")
        if getattr(state, "_cpu_s46e_pool", None) is not pool:
            raise InstrumentationBindingError("A_ACTIVE_POOL_CHANGED")
        _require(not pool.closed, "A_RETIRED_WORKER_POOL")

    helpers = _helpers(pool, guard=guard_before_helper)
    private_globals = dict(baseline_globals)
    private_globals.update(helpers)
    scipy_slots = {name: getattr(scipy_fft, name) for name in ("fft", "ifft")}

    def guard_stage() -> None:
        guard_before_helper()
        active = active_stage_ref["stage"]
        _require(active.__globals__ is private_globals, "A_STAGE_GLOBAL_NAMESPACE_CHANGED")
        _require(private_globals.get(GUARD_NAME) is guard_stage, "A_STAGE_GUARD_SLOT_CHANGED")
        _require(private_globals.get("xp") is xp and private_globals.get("_np") is np, "A_STAGE_BACKEND_SLOT_CHANGED")
        validate_cpu_backend(xp, expected_workers=backend["fft_workers"])
        _require(xp.fft.dispatch is backend["dispatch_object"], "A_STAGE_FFT_DISPATCH_CHANGED")
        _require(all(getattr(scipy_fft, name) is value for name, value in scipy_slots.items()), "A_STAGE_SCIPY_TRANSFORM_CHANGED")
        slots = state._cpu_s46e_backend_slots
        _require(np.absolute is slots["absolute"] and np.square is slots["square"] and np.multiply is slots["multiply"], "A_STAGE_UFUNC_CHANGED")
        for name, helper in helpers.items():
            _require(private_globals.get(name) is helper and helper.__globals__.get("np") is np, "A_STAGE_HELPER_SLOT_CHANGED:" + name)
            verify_seal(state._cpu_s46e_helper_seals[name])
        verify_seal(state._cpu_s46e_stage_seal)
        verify_seal(state._cpu_s46e_stage_entry_guard_seal)

    private_globals[GUARD_NAME] = guard_stage
    receipt = instrument_intensity_stage(stage_node)
    stage_node.body.insert(1 if ast.get_docstring(stage_node) is not None else 0, ast.Expr(value=ast.Call(func=ast.Name(id=GUARD_NAME, ctx=ast.Load()), args=[], keywords=[])))
    ast.fix_missing_locations(stage_node)
    _require(_ast_hash(strip_generated_nodes(stage_node)) == receipt["source_ast_sha256"], "A_STAGE_GUARD_STRIP_PROOF")
    receipt.update(instrumented_ast_sha256=_ast_hash(stage_node), stage_entry_guard=True)
    instrumented = _compile_function(stage_node, namespace=private_globals, source_path=source_path)
    instrumented.__name__ = getattr(original, "__name__", instrumented.__name__)
    instrumented.__qualname__ = getattr(original, "__qualname__", instrumented.__qualname__)
    instrumented.__module__ = getattr(original, "__module__", instrumented.__module__)
    instrumented.__doc__ = getattr(original, "__doc__", instrumented.__doc__)
    instrumented.__cpu_s46e_original__ = original
    instrumented.__cpu_s46e_original_code_identity__ = identity(original)
    instrumented.__cpu_s46e_source_path__ = str(source_path)
    setattr(stage_owner, stage_attribute, instrumented)
    active_stage_ref["stage"] = instrumented
    state._cpu_s46e_original_stage = original
    state._cpu_s46e_stage = instrumented
    state._cpu_s46e_stage_owner = stage_owner
    state._cpu_s46e_stage_attribute = stage_attribute
    state._cpu_s46e_active_stage_ref = active_stage_ref
    state._cpu_s46e_guard = guard_before_helper
    state._cpu_s46e_guard_seal = _seal(guard_before_helper)
    state._cpu_s46e_stage_entry_guard = guard_stage
    state._cpu_s46e_stage_entry_guard_seal = _seal(guard_stage)
    state._cpu_s46e_scipy_slots = scipy_slots
    state._cpu_s46e_r8_stage_original = r8_stage_before
    state._cpu_s46e_backend_slots = {"numpy": np, "absolute": np.absolute, "square": np.square, "multiply": np.multiply, "xp": backend["xp_object"], "dispatch": backend["dispatch_object"], "underlying": backend["xp_object"].xp}
    state._cpu_s46e_backend_binding = backend_metadata
    state._cpu_s46e_pool = pool
    state._cpu_s46e_private_namespace = private_globals
    state._cpu_s46e_helper_seals = {name: _seal(function) for name, function in helpers.items()}
    state._cpu_s46e_stage_seal = _seal(instrumented)
    state._cpu_s46e_lineage = {
        "original_stage_code_identity": identity(original),
        "r8_expected_stage_code_identity": identity(expected),
        "r8_stage_code_identity": None if not callable(r8_stage) else identity(r8_stage),
        "instrumented_stage_code_identity": identity(instrumented),
        "stage_owner": stage_owner.__class__.__name__,
        "stage_attribute": stage_attribute,
        "source_path": str(source_path),
        "source_ast_sha256": receipt["source_ast_sha256"],
        "stripped_ast_sha256": receipt["stripped_ast_sha256"],
        "helper_names": dict(HELPER_NAMES),
        "scientific_expression_changes": 0,
        "pointwise_segments": ["complex_absolute", "square", "coefficient_scale"],
        "array_values_retained": False,
        "private_guard_before_helper_calls": True,
        "guard_code_identity": _code_identity(guard_before_helper),
        "backend_binding": copy.deepcopy(backend_metadata),
    }
    return copy.deepcopy({**receipt, "lineage": state._cpu_s46e_lineage, "helper_registry_fixed": True})


__all__ = [
    "HELPER_NAMES",
    "GUARD_NAME",
    "INSTRUMENT_SCHEMA",
    "InstrumentationBindingError",
    "validate_cpu_backend",
    "_code_identity",
    "_seal",
    "install_stage",
    "instrument_intensity_stage",
    "strip_generated_nodes",
    "verify_seal",
]
