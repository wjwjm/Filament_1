"""Source-bound Eq10 pointwise replacement for the private L1 clone."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import FunctionType
from typing import Any, Callable, Mapping

import numpy as np
import scipy.fft as scipy_fft

from .pool import Eq10Pool


INSTRUMENT_SCHEMA = "filament.cpu_s4_6d.eq10_instrument.v1"
GUARD_NAME = "__cpu_s46d_stage_guard"
HELPER_NAMES = {
    "spectral": "__cpu_s46d_spectral_multiply",
    "derivative": "__cpu_s46d_derivative_divide",
    "response": "__cpu_s46d_response_divide",
    "product": "__cpu_s46d_product",
}
CPU_XP_NAMESPACE_TYPE = "_XPNamespace"
CPU_XP_MODULE_SUFFIX = "cpu_p1r2_j2"
CPU_FFT_DISPATCH_TYPE = "_ScipyDispatch"


def validate_cpu_backend(
    xp: Any,
    *,
    expected_workers: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Validate the accepted J2 NumPy proxy and its fixed FFT worker pair.

    The stage is compiled in a private namespace whose ``xp`` is the J2
    ``_XPNamespace`` proxy, not the bare NumPy module.  Checking its concrete
    type/module, underlying NumPy object, SciPy dispatch type/module, and
    descriptor prevents a GPU or arbitrary object with an ``xp`` attribute
    from entering the candidate while preserving the 1/1 or 48/24 FFT route.
    """

    _require(type(xp).__name__ == CPU_XP_NAMESPACE_TYPE, "EQ10_CPU_XP_NAMESPACE_TYPE")
    xp_module = str(type(xp).__module__).replace("\\", ".")
    _require(xp_module.endswith(CPU_XP_MODULE_SUFFIX), "EQ10_CPU_XP_NAMESPACE_MODULE")
    underlying = getattr(xp, "xp", None)
    _require(underlying is np, "EQ10_CPU_BACKEND_REQUIRED")
    fft_namespace = getattr(xp, "fft", None)
    _require(type(fft_namespace).__name__ == "_FFTNamespace", "EQ10_CPU_FFT_NAMESPACE_TYPE")
    fft_module = str(type(fft_namespace).__module__).replace("\\", ".")
    _require(fft_module.endswith(CPU_XP_MODULE_SUFFIX), "EQ10_CPU_FFT_NAMESPACE_MODULE")
    dispatch = getattr(fft_namespace, "dispatch", None)
    _require(type(dispatch).__name__ == CPU_FFT_DISPATCH_TYPE, "EQ10_CPU_FFT_DISPATCH_TYPE")
    dispatch_module = str(type(dispatch).__module__).replace("\\", ".")
    _require(dispatch_module.endswith(CPU_XP_MODULE_SUFFIX), "EQ10_CPU_FFT_DISPATCH_MODULE")
    _require(getattr(dispatch, "numpy_fft", None) is np.fft, "EQ10_CPU_NUMPY_FFT_BINDING")
    _require(getattr(dispatch, "scipy_fft", None) is scipy_fft, "EQ10_CPU_SCIPY_FFT_BINDING")
    descriptor = getattr(dispatch, "descriptor", None)
    _require(
        isinstance(descriptor, dict)
        and set(descriptor) == {"temporal_workers", "spatial_workers"},
        "EQ10_CPU_FFT_DESCRIPTOR",
    )
    descriptor = {name: int(value) for name, value in descriptor.items()}
    if expected_workers is not None:
        expected = {
            "temporal_workers": int(expected_workers["temporal_workers"]),
            "spatial_workers": int(expected_workers["spatial_workers"]),
        }
        _require(descriptor == expected, "EQ10_CPU_FFT_WORKERS_MISMATCH")
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


class InstrumentationBindingError(RuntimeError):
    """The frozen stage could not be transformed or verified."""


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise InstrumentationBindingError(reason)


def _code_identity(function: Callable[..., Any]) -> str:
    import hashlib
    import marshal

    code = getattr(function, "__code__", None)
    _require(code is not None, "EQ10_CALLABLE_CODE_MISSING")
    return hashlib.sha256(marshal.dumps(code)).hexdigest()


def _function_node(stage: ast.FunctionDef) -> tuple[ast.If, ast.If, ast.Assign, ast.Assign, ast.Assign]:
    return_energy = [
        node
        for node in stage.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "return_energy"
    ]
    _require(len(return_energy) == 1, "EQ10_RETURN_ENERGY_BRANCH_COUNT")
    branch = return_energy[0]
    positive = [
        node
        for node in branch.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "intensity_scale"
    ]
    _require(len(positive) == 1, "EQ10_INTENSITY_SCALE_BRANCH_COUNT")
    pos = positive[0]

    def assignment(name: str, body: list[ast.stmt], *, zero_ok: bool = False) -> ast.Assign:
        matches = [
            node
            for node in body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        ]
        if zero_ok:
            _require(len(matches) == 1, f"EQ10_ASSIGNMENT_COUNT:{name}")
        else:
            _require(len(matches) == 1, f"EQ10_ASSIGNMENT_COUNT:{name}")
        return matches[0]

    derivative = assignment("derivative_increment_norm", pos.body)
    response = assignment("response_norm", pos.body)
    product = assignment("u_signed", pos.body)
    return branch, pos, derivative, response, product


def _is_call_path(node: ast.AST, path: tuple[str, ...]) -> bool:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    current: ast.AST = node.func
    names = [node.func.attr]
    value = node.func.value
    while isinstance(value, ast.Attribute):
        names.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        names.append(value.id)
    return tuple(reversed(names)) == path


def _name(node: ast.AST, value: str) -> bool:
    return isinstance(node, ast.Name) and node.id == value


def _call(name: str, args: list[ast.AST], template: ast.AST) -> ast.Call:
    node = ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=args, keywords=[])
    return ast.copy_location(node, template)


def instrument_eq10_stage(stage: ast.FunctionDef) -> dict[str, Any]:
    """Replace exactly four pointwise BinOps while retaining all other nodes."""

    _require(isinstance(stage, ast.FunctionDef), "EQ10_STAGE_FUNCTION_REQUIRED")
    baseline = copy.deepcopy(stage)
    _, positive, derivative, response, product = _function_node(stage)
    counts = {key: 0 for key in HELPER_NAMES}

    value = derivative.value
    _require(isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div), "EQ10_DERIVATIVE_DIVIDE_SHAPE")
    real = value.left
    _require(isinstance(real, ast.Attribute) and real.attr == "real", "EQ10_IFFT_REAL_VIEW_SHAPE")
    ifft_call = real.value
    _require(_is_call_path(ifft_call, ("xp", "fft", "ifft")), "EQ10_IFFT_CALL_SHAPE")
    _require(len(ifft_call.args) >= 1, "EQ10_IFFT_ARGUMENT_MISSING")
    spectral = ifft_call.args[0]
    _require(isinstance(spectral, ast.BinOp) and isinstance(spectral.op, ast.Mult), "EQ10_SPECTRAL_MULTIPLY_SHAPE")
    _require(_is_call_path(spectral.right, ("xp", "fft", "fft")), "EQ10_FFT_CALL_SHAPE")
    ifft_call.args[0] = _call(HELPER_NAMES["spectral"], [spectral.left, spectral.right], spectral)
    # Pass the complex IFFT result; the helper takes the strided ``.real``
    # view itself so the generated call preserves the original view boundary.
    derivative.value = _call(HELPER_NAMES["derivative"], [ifft_call, value.right], value)
    counts["spectral"] += 1
    counts["derivative"] += 1

    value = response.value
    _require(isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div), "EQ10_RESPONSE_DIVIDE_SHAPE")
    _require(_name(value.right, "intensity_scale"), "EQ10_RESPONSE_SCALE_NAME")
    response.value = _call(HELPER_NAMES["response"], [value.left, value.right], value)
    counts["response"] += 1

    value = product.value
    _require(isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult), "EQ10_PRODUCT_SCALE_SHAPE")
    _require(_name(value.left, "energy_scale"), "EQ10_PRODUCT_ENERGY_SCALE_ORDER")
    _require(isinstance(value.right, ast.Call) and _is_call_path(value.right, ("xp", "sum")), "EQ10_SUM_CALL_SHAPE")
    _require(len(value.right.args) == 1, "EQ10_SUM_ARGUMENT_COUNT")
    inner = value.right.args[0]
    _require(isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Mult), "EQ10_PRODUCT_POINTWISE_SHAPE")
    value.right.args[0] = _call(HELPER_NAMES["product"], [inner.left, inner.right], inner)
    counts["product"] += 1
    ast.fix_missing_locations(stage)
    _require(all(value == 1 for value in counts.values()), "EQ10_FOUR_POINTWISE_ANCHORS")
    stripped = strip_generated_nodes(stage)
    _require(
        ast.dump(stripped, include_attributes=False) == ast.dump(baseline, include_attributes=False),
        "EQ10_STRIPPED_AST_MISMATCH",
    )
    return {
        "schema": INSTRUMENT_SCHEMA,
        "source_ast_sha256": _ast_hash(baseline),
        "instrumented_ast_sha256": _ast_hash(stage),
        "stripped_ast_sha256": _ast_hash(stripped),
        "helper_names": dict(HELPER_NAMES),
        "replacement_counts": dict(counts),
        "fft_calls": 1,
        "ifft_calls": 1,
        "sum_calls": 1,
        "array_values_retained": False,
        "private_guard_before_helper_calls": True,
        "strip_proof": "generated_helpers_removed_to_original_stage_ast",
    }


def strip_generated_nodes(stage: ast.FunctionDef) -> ast.FunctionDef:
    result = copy.deepcopy(stage)

    class Stripper(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == GUARD_NAME:
                _require(not node.value.args and not node.value.keywords, "EQ10_STAGE_GUARD_SHAPE")
                return None
            return self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> ast.AST:
            for key, helper in HELPER_NAMES.items():
                if isinstance(node.func, ast.Name) and node.func.id == helper:
                    if key == "spectral":
                        _require(len(node.args) == 2, "EQ10_STRIP_SPECTRAL_ARGS")
                        return ast.BinOp(left=self.visit(node.args[0]), op=ast.Mult(), right=self.visit(node.args[1]))
                    if key in ("derivative", "response"):
                        _require(len(node.args) == 2, "EQ10_STRIP_DIVIDE_ARGS")
                        left = self.visit(node.args[0])
                        if key == "derivative":
                            left = ast.Attribute(value=left, attr="real", ctx=ast.Load())
                        return ast.BinOp(left=left, op=ast.Div(), right=self.visit(node.args[1]))
                    if key == "product":
                        _require(len(node.args) == 2, "EQ10_STRIP_PRODUCT_ARGS")
                        return ast.BinOp(left=self.visit(node.args[0]), op=ast.Mult(), right=self.visit(node.args[1]))
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
    _require(isinstance(function, FunctionType), "EQ10_STAGE_COMPILE_FAILED")
    return function


def _seal(function: Callable[..., Any]) -> dict[str, Any]:
    _require(callable(function), "EQ10_SEAL_CALLABLE_REQUIRED")
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
    _require(callable(function), "EQ10_SEAL_FUNCTION_MISSING")
    _require(function.__code__ is seal.get("code"), "EQ10_SEAL_CODE_CHANGED")
    _require(function.__globals__ is seal.get("globals"), "EQ10_SEAL_GLOBALS_CHANGED")
    closure = getattr(function, "__closure__", None) or ()
    expected = seal.get("closure_cells", ())
    _require(len(closure) == len(expected), "EQ10_SEAL_CLOSURE_SIZE")
    _require(all(cell.cell_contents is value for cell, value in zip(closure, expected)), "EQ10_SEAL_CLOSURE_CHANGED")
    _require(function.__defaults__ is seal.get("defaults"), "EQ10_SEAL_DEFAULTS_CHANGED")
    _require(function.__kwdefaults__ is seal.get("kwdefaults_object"), "EQ10_SEAL_KWDEFAULTS_OBJECT_CHANGED")
    current, expected_kw = function.__kwdefaults__ or {}, seal.get("kwdefaults", {})
    _require(current.keys() == expected_kw.keys() and all(current[key] is value for key, value in expected_kw.items()), "EQ10_SEAL_KWDEFAULTS_CHANGED")


def _helpers(
    pool: Eq10Pool,
    *,
    guard: Callable[[], None] | None = None,
) -> dict[str, Callable[..., Any]]:
    if not isinstance(pool, Eq10Pool):
        raise InstrumentationBindingError("EQ10_POOL_TYPE")
    if guard is None:
        guard = lambda: None

    def spectral(multiplier: Any, spectrum: Any) -> Any:
        output = np.empty_like(spectrum)
        guard()
        return pool.ufunc("B_SPECTRAL_MULTIPLY", np.multiply, (multiplier, spectrum), output)

    def derivative(inverse_real: Any, scale: Any) -> Any:
        real = inverse_real.real
        output = np.empty_like(real)
        guard()
        return pool.ufunc("B_DERIVATIVE_DIVIDE", np.divide, (real, scale), output)

    def response(value: Any, scale: Any) -> Any:
        output = np.empty_like(value)
        guard()
        return pool.ufunc("B_RESPONSE_DIVIDE", np.divide, (value, scale), output)

    def product(left: Any, right: Any) -> Any:
        output = np.empty_like(left)
        guard()
        return pool.ufunc("B_PRODUCT", np.multiply, (left, right), output)

    for function, name in (
        (spectral, HELPER_NAMES["spectral"]),
        (derivative, HELPER_NAMES["derivative"]),
        (response, HELPER_NAMES["response"]),
        (product, HELPER_NAMES["product"]),
    ):
        function.__name__ = name
        function.__cpu_s46d_kind__ = "FIXED_EQ10_POINTWISE_HELPER"
        function.__cpu_s46d_pool__ = pool
    return {
        HELPER_NAMES["spectral"]: spectral,
        HELPER_NAMES["derivative"]: derivative,
        HELPER_NAMES["response"]: response,
        HELPER_NAMES["product"]: product,
    }


def install_stage(state: Any, pool: Eq10Pool) -> dict[str, Any]:
    """Install the four-operation clone in the existing private R8 state."""

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
    _require(callable(original), "EQ10_ORIGINAL_STAGE_MISSING")
    candidate_namespace = getattr(state, "_candidate_namespace", None)
    _require(isinstance(candidate_namespace, dict), "EQ10_CANDIDATE_NAMESPACE_MISSING")
    s45a = __import__("tools.cpu_s4_5a_l1_residual.instrument", fromlist=["unused"])
    builder = getattr(s45a, "_expected_transformed_node", None)
    identity = getattr(s45a, "_code_identity", None)
    _require(callable(builder) and callable(identity), "EQ10_SOURCE_AST_SEAM_MISSING")
    source_path, stage_node, expected = builder(
        name="isaacs_raman_stage",
        transform_name="_transform_stage",
        candidate_namespace=candidate_namespace,
    )
    _require(identity(expected) == identity(original), "EQ10_R8_STAGE_SOURCE_MISMATCH")
    baseline_globals = getattr(original, "__globals__", None)
    _require(isinstance(baseline_globals, dict), "EQ10_STAGE_GLOBALS_MISSING")
    xp = baseline_globals.get("xp")
    backend = validate_cpu_backend(
        xp,
        expected_workers=getattr(state, "_cpu_s46d_expected_backend_workers", None),
    )
    _require(baseline_globals.get("_np") is np, "EQ10_CPU_NUMPY_ALIAS_CHANGED")
    backend_metadata = {
        key: value
        for key, value in backend.items()
        if key not in {"xp_object", "dispatch_object"}
    }
    active_stage_ref: dict[str, Any] = {"stage": None}

    def guard_before_helper() -> None:
        """Reject a replaced active stage before submitting any ufunc."""

        active = active_stage_ref.get("stage")
        if active is None or getattr(stage_owner, stage_attribute, None) is not active:
            raise InstrumentationBindingError("EQ10_ACTIVE_STAGE_CHANGED")
        if getattr(state, "_cpu_s46d_stage", None) is not active:
            raise InstrumentationBindingError("EQ10_ACTIVE_STAGE_STATE_CHANGED")
        if getattr(state, "_cpu_s46d_pool", None) is not pool:
            raise InstrumentationBindingError("EQ10_ACTIVE_POOL_CHANGED")
        _require(not pool.closed, "EQ10_RETIRED_WORKER_POOL")

    helpers = _helpers(pool, guard=guard_before_helper)
    private_globals = dict(baseline_globals)
    private_globals.update(helpers)
    scipy_slots = {name: getattr(scipy_fft, name) for name in ('fft', 'ifft')}

    def guard_stage() -> None:
        guard_before_helper()
        active = active_stage_ref['stage']
        _require(active.__globals__ is private_globals, "EQ10_STAGE_GLOBAL_NAMESPACE_CHANGED")
        _require(private_globals.get(GUARD_NAME) is guard_stage, "EQ10_STAGE_GUARD_SLOT_CHANGED")
        _require(private_globals.get('xp') is xp and private_globals.get('_np') is np, "EQ10_STAGE_BACKEND_SLOT_CHANGED")
        validate_cpu_backend(xp, expected_workers=backend['fft_workers'])
        _require(xp.fft.dispatch is backend['dispatch_object'], "EQ10_STAGE_FFT_DISPATCH_CHANGED")
        _require(all(getattr(scipy_fft, name) is value for name, value in scipy_slots.items()), "EQ10_STAGE_SCIPY_TRANSFORM_CHANGED")
        _require(np.multiply is state._cpu_s46d_backend_slots['multiply'] and np.divide is state._cpu_s46d_backend_slots['divide'], "EQ10_STAGE_UFUNC_CHANGED")
        for name, helper in helpers.items():
            _require(private_globals.get(name) is helper and helper.__globals__.get('np') is np, "EQ10_STAGE_HELPER_SLOT_CHANGED:"+name)
            verify_seal(state._cpu_s46d_helper_seals[name])
        verify_seal(state._cpu_s46d_stage_seal)
        verify_seal(state._cpu_s46d_stage_entry_guard_seal)

    private_globals[GUARD_NAME] = guard_stage
    receipt = instrument_eq10_stage(stage_node)
    stage_node.body.insert(1 if ast.get_docstring(stage_node) is not None else 0, ast.Expr(value=ast.Call(func=ast.Name(id=GUARD_NAME, ctx=ast.Load()), args=[], keywords=[])))
    ast.fix_missing_locations(stage_node)
    _require(_ast_hash(strip_generated_nodes(stage_node)) == receipt['source_ast_sha256'], "EQ10_STAGE_GUARD_STRIP_PROOF")
    receipt.update(instrumented_ast_sha256=_ast_hash(stage_node), stage_entry_guard=True)
    instrumented = _compile_function(stage_node, namespace=private_globals, source_path=source_path)
    instrumented.__name__ = getattr(original, "__name__", instrumented.__name__)
    instrumented.__qualname__ = getattr(original, "__qualname__", instrumented.__qualname__)
    instrumented.__module__ = getattr(original, "__module__", instrumented.__module__)
    instrumented.__doc__ = getattr(original, "__doc__", instrumented.__doc__)
    instrumented.__cpu_s46d_original__ = original
    instrumented.__cpu_s46d_original_code_identity__ = identity(original)
    instrumented.__cpu_s46d_source_path__ = str(source_path)
    setattr(stage_owner, stage_attribute, instrumented)
    active_stage_ref["stage"] = instrumented
    state._cpu_s46d_original_stage = original
    state._cpu_s46d_stage = instrumented
    state._cpu_s46d_stage_owner = stage_owner
    state._cpu_s46d_stage_attribute = stage_attribute
    state._cpu_s46d_active_stage_ref = active_stage_ref
    state._cpu_s46d_guard = guard_before_helper
    state._cpu_s46d_guard_seal = _seal(guard_before_helper)
    state._cpu_s46d_stage_entry_guard = guard_stage
    state._cpu_s46d_stage_entry_guard_seal = _seal(guard_stage)
    state._cpu_s46d_scipy_slots = scipy_slots
    state._cpu_s46d_r8_stage_original = r8_stage_before
    state._cpu_s46d_backend_slots = {
        "numpy": np,
        "multiply": np.multiply,
        "divide": np.divide,
        "xp": backend["xp_object"],
        "dispatch": backend["dispatch_object"],
        "underlying": backend["xp_object"].xp,
    }
    state._cpu_s46d_backend_binding = backend_metadata
    state._cpu_s46d_pool = pool
    state._cpu_s46d_private_namespace = private_globals
    state._cpu_s46d_helper_seals = {name: _seal(function) for name, function in helpers.items()}
    state._cpu_s46d_stage_seal = _seal(instrumented)
    state._cpu_s46d_lineage = {
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
        "array_values_retained": False,
        "private_guard_before_helper_calls": True,
        "guard_code_identity": _code_identity(guard_before_helper),
        "backend_binding": copy.deepcopy(backend_metadata),
    }
    return copy.deepcopy({**receipt, "lineage": state._cpu_s46d_lineage, "helper_registry_fixed": True})


__all__ = [
    "HELPER_NAMES",
    "GUARD_NAME",
    "INSTRUMENT_SCHEMA",
    "InstrumentationBindingError",
    "validate_cpu_backend",
    "_code_identity",
    "_seal",
    "install_stage",
    "instrument_eq10_stage",
    "strip_generated_nodes",
    "verify_seal",
]
