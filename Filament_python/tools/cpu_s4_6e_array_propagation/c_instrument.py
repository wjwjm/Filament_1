"""Source-bound C/P24 replacement for the frozen nonlinear field update."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
from pathlib import Path
from types import FunctionType
from typing import Any, Callable, Mapping

import numpy as np

from tools.hpc_ops import provenance_v2 as pv

from .c_pool import CPool, CPoolError


INSTRUMENT_SCHEMA = "filament.cpu_s4_6e.c_instrument.v1"
GUARD_NAME = "__cpu_s46e_c_apply_guard"
PHASE_HELPER_NAME = "__cpu_s46e_c_phase_update"
ABSORPTION_HELPER_NAME = "__cpu_s46e_c_absorption_update"
PHASE_WORKER_NAME = "__cpu_s46e_c_phase_worker"
ABSORPTION_WORKER_NAME = "__cpu_s46e_c_absorption_worker"
HELPER_NAMES = {
    "guard": GUARD_NAME,
    "phase": PHASE_HELPER_NAME,
    "absorption": ABSORPTION_HELPER_NAME,
}
WORKER_NAMES = {
    "phase": PHASE_WORKER_NAME,
    "absorption": ABSORPTION_WORKER_NAME,
}


class CInstrumentationError(RuntimeError):
    """The frozen nonlinear source or private C binding failed closed."""


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise CInstrumentationError(reason)


def _ast_hash(node: ast.AST) -> str:
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode("utf-8")).hexdigest()


def _function_node(tree: ast.Module, name: str = "apply_nonlinear") -> ast.FunctionDef:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    _require(len(matches) == 1, f"C_FUNCTION_ANCHOR:{name}:{len(matches)}")
    return matches[0]


def _is_xp_exp_astype(value: ast.AST) -> bool:
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "astype"
        and len(value.args) == 1
        and len(value.keywords) == 1
        and value.keywords[0].arg == "copy"
        and isinstance(value.keywords[0].value, ast.Constant)
        and value.keywords[0].value.value is False
        and isinstance(value.args[0], ast.Name)
        and value.args[0].id == "ctype"
        and isinstance(value.func.value, ast.Call)
        and isinstance(value.func.value.func, ast.Attribute)
        and isinstance(value.func.value.func.value, ast.Name)
        and value.func.value.func.value.id == "xp"
        and value.func.value.func.attr == "exp"
        and len(value.func.value.args) == 1
        and not value.func.value.keywords
    )


def _validated_pointwise_updates(function: ast.FunctionDef) -> tuple[ast.AugAssign, ast.AugAssign]:
    """Return the two frozen pointwise updates after shape validation."""

    _require(isinstance(function, ast.FunctionDef), "C_FUNCTION_REQUIRED")
    updates = [
        node
        for node in function.body
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "E"
        and isinstance(node.op, ast.Mult)
    ]
    _require(len(updates) == 2, f"C_AUGASSIGN_COUNT:{len(updates)}")
    _require(all(_is_xp_exp_astype(node.value) for node in updates), "C_AUGASSIGN_EXPRESSION_SHAPE")
    first, second = updates
    first_arg = first.value.func.value.args[0]
    second_arg = second.value.func.value.args[0]
    _require(
        isinstance(first_arg, ast.BinOp)
        and isinstance(first_arg.op, ast.Mult)
        and isinstance(first_arg.left, ast.Name)
        and first_arg.left.id == "onej"
        and isinstance(first_arg.right, ast.Name)
        and first_arg.right.id == "phase",
        "C_PHASE_EXPRESSION_SHAPE",
    )
    _require(
        isinstance(second_arg, ast.BinOp)
        and isinstance(second_arg.op, ast.Mult)
        and isinstance(second_arg.left, ast.BinOp)
        and isinstance(second_arg.left.left, ast.UnaryOp)
        and isinstance(second_arg.left.left.op, ast.USub)
        and isinstance(second_arg.left.left.operand, ast.Constant)
        and second_arg.left.left.operand.value == 0.5
        and isinstance(second_arg.left.right, ast.Name)
        and second_arg.left.right.id == "alpha"
        and isinstance(second_arg.right, ast.Call)
        and isinstance(second_arg.right.func, ast.Name)
        and second_arg.right.func.id == "float"
        and len(second_arg.right.args) == 1
        and isinstance(second_arg.right.args[0], ast.Name)
        and second_arg.right.args[0].id == "dz",
        "C_ABSORPTION_EXPRESSION_SHAPE",
    )
    return first, second


def _helper_call(name: str, args: list[ast.AST], template: ast.AST) -> ast.Expr:
    call = ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=args, keywords=[])
    return ast.copy_location(ast.Expr(value=call), template)


def instrument_apply_nonlinear(function: ast.FunctionDef) -> dict[str, Any]:
    """Replace exactly the two frozen in-place expressions and prove inverse."""

    baseline = copy.deepcopy(function)
    first, second = _validated_pointwise_updates(function)

    first_index = function.body.index(first)
    second_index = function.body.index(second)
    function.body[first_index] = _helper_call(
        PHASE_HELPER_NAME,
        [ast.Name(id="E", ctx=ast.Load()), ast.Name(id="onej", ctx=ast.Load()), ast.Name(id="phase", ctx=ast.Load()), ast.Name(id="ctype", ctx=ast.Load())],
        first,
    )
    # The first replacement does not change the index of the second one.
    function.body[second_index] = _helper_call(
        ABSORPTION_HELPER_NAME,
        [ast.Name(id="E", ctx=ast.Load()), ast.Name(id="alpha", ctx=ast.Load()), ast.Name(id="dz", ctx=ast.Load()), ast.Name(id="ctype", ctx=ast.Load())],
        second,
    )
    guard = ast.Expr(
        value=ast.Call(
            func=ast.Name(id=GUARD_NAME, ctx=ast.Load()),
            args=[
                ast.Name(id="E", ctx=ast.Load()),
                ast.Name(id="phase", ctx=ast.Load()),
                ast.Name(id="alpha", ctx=ast.Load()),
                ast.Name(id="dz", ctx=ast.Load()),
                ast.Name(id="dn_gas", ctx=ast.Load()),
                ast.Name(id="k0", ctx=ast.Load()),
            ],
            keywords=[],
        )
    )
    function.body.insert(0, ast.copy_location(guard, function.body[0]))
    ast.fix_missing_locations(function)
    stripped = strip_generated_nodes(function)
    _require(ast.dump(stripped, include_attributes=False) == ast.dump(baseline, include_attributes=False), "C_STRIPPED_AST_MISMATCH")
    return {
        "schema": INSTRUMENT_SCHEMA,
        "source_ast_sha256": _ast_hash(baseline),
        "instrumented_ast_sha256": _ast_hash(function),
        "stripped_ast_sha256": _ast_hash(stripped),
        "helper_names": dict(HELPER_NAMES),
        "replacement_counts": {"phase": 1, "absorption": 1},
        "prelude_preserved": True,
        "math_reordering": False,
        "array_values_retained": False,
        "private_guard_before_helper_calls": True,
        "strip_proof": "generated_helpers_and_guard_removed_to_original_apply_nonlinear_ast",
    }


def strip_generated_nodes(function: ast.FunctionDef) -> ast.FunctionDef:
    """Remove C nodes and restore the exact two original AugAssign nodes."""

    result = copy.deepcopy(function)

    class Stripper(ast.NodeTransformer):
        def visit_Expr(self, node: ast.Expr) -> ast.AST | None:
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                if node.value.func.id == GUARD_NAME:
                    _require(len(node.value.args) == 6 and not node.value.keywords, "C_GUARD_STRIP_SHAPE")
                    return None
                if node.value.func.id in (PHASE_HELPER_NAME, ABSORPTION_HELPER_NAME):
                    _require(not node.value.keywords, "C_HELPER_STRIP_KEYWORDS")
                    args = [self.visit(item) for item in node.value.args]
                    if node.value.func.id == PHASE_HELPER_NAME:
                        _require(len(args) == 4, "C_PHASE_HELPER_STRIP_ARGS")
                        rhs_arg: ast.AST = ast.BinOp(left=ast.Name(id="onej", ctx=ast.Load()), op=ast.Mult(), right=ast.Name(id="phase", ctx=ast.Load()))
                    else:
                        _require(len(args) == 4, "C_ABSORPTION_HELPER_STRIP_ARGS")
                        rhs_arg = ast.BinOp(
                            left=ast.BinOp(left=ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=0.5)), op=ast.Mult(), right=ast.Name(id="alpha", ctx=ast.Load())),
                            op=ast.Mult(),
                            right=ast.Call(func=ast.Name(id="float", ctx=ast.Load()), args=[ast.Name(id="dz", ctx=ast.Load())], keywords=[]),
                        )
                    exp_call = ast.Call(func=ast.Attribute(value=ast.Name(id="xp", ctx=ast.Load()), attr="exp", ctx=ast.Load()), args=[rhs_arg], keywords=[])
                    cast = ast.Call(
                        func=ast.Attribute(value=exp_call, attr="astype", ctx=ast.Load()),
                        args=[ast.Name(id="ctype", ctx=ast.Load())],
                        keywords=[ast.keyword(arg="copy", value=ast.Constant(value=False))],
                    )
                    return ast.copy_location(ast.AugAssign(target=ast.Name(id="E", ctx=ast.Store()), op=ast.Mult(), value=cast), node)
            return self.generic_visit(node)

    restored = Stripper().visit(result)
    _require(isinstance(restored, ast.FunctionDef), "C_STRIP_RESULT_INVALID")
    ast.fix_missing_locations(restored)
    return restored


def validate_cpu_backend(xp: Any) -> dict[str, Any]:
    """Validate the source-bound CPU backend, including the ``xp.abs`` slot."""

    underlying = np if xp is np else getattr(xp, "xp", None)
    _require(underlying is np, "C_CPU_BACKEND_REQUIRED")
    _require(getattr(xp, "abs", None) is np.absolute, "C_CPU_ABS_BINDING")
    fft = getattr(xp, "fft", None)
    dispatch = getattr(fft, "dispatch", None)
    return {
        "xp_object": xp,
        "underlying": underlying,
        "xp_type": type(xp).__name__,
        "xp_module": str(type(xp).__module__).replace("\\", "."),
        "fft_dispatch_object": dispatch,
        "fft_dispatch_type": type(dispatch).__name__ if dispatch is not None else None,
    }


def validate_runtime_backend(xp: Any, expected: Mapping[str, Any]) -> None:
    current = validate_cpu_backend(xp)
    _require(current["xp_object"] is expected["xp_object"], "C_CPU_BACKEND_OBJECT_CHANGED")
    _require(current["underlying"] is expected["underlying"], "C_CPU_BACKEND_UNDERLYING_CHANGED")
    _require(current["fft_dispatch_object"] is expected["fft_dispatch_object"], "C_CPU_FFT_DISPATCH_CHANGED")


def _source_path(original: Callable[..., Any]) -> Path:
    code = getattr(original, "__code__", None)
    _require(code is not None, "C_ORIGINAL_APPLY_CODE_MISSING")
    path = Path(str(code.co_filename)).resolve(strict=True)
    repo = Path(__file__).resolve().parents[3]
    _require(path == (repo / "Filament_python/KHz_filament/nonlinear.py").resolve(), "C_NONLINEAR_SOURCE_PATH")
    return path


def source_binding(original: Callable[..., Any]) -> dict[str, Any]:
    path = _source_path(original)
    raw = path.read_bytes()
    expected = importlib.import_module("tools.cpu_s4_3g.controlled_callsite")
    canonical = pv.canonical_lf_sha256(raw)
    _require(canonical == expected._NONLINEAR_CANONICAL_LF_SHA256, "C_NONLINEAR_SOURCE_HASH")
    _require(Path(path).read_text(encoding="utf-8").splitlines()[33].strip().startswith("def apply_nonlinear"), "C_NONLINEAR_SOURCE_LINE")
    return {
        "path": str(path),
        "relative_path": "Filament_python/KHz_filament/nonlinear.py",
        "classification": "tracked_text",
        "hash_scope": pv.TRACKED_HASH_SCOPE,
        "canonical_lf_sha256": canonical,
        "source_blob": expected._NONLINEAR_SOURCE_BLOB,
        "function": "apply_nonlinear",
        "function_first_line": 34,
        "raw_sha256": pv.raw_sha256(raw),
    }


def _compile_function(function: ast.FunctionDef, namespace: dict[str, Any], source_path: Path) -> FunctionType:
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    local: dict[str, Any] = {}
    exec(compile(module, str(source_path), "exec"), namespace, local)
    compiled = local.get(function.name)
    _require(isinstance(compiled, FunctionType), "C_APPLY_COMPILE_FAILED")
    return compiled


def _slice_update(update: ast.AugAssign, array_name: str) -> ast.AugAssign:
    """Copy one frozen update and add only the axis-0 task slice."""

    node = copy.deepcopy(update)

    class SliceArrays(ast.NodeTransformer):
        def visit_Name(self, item: ast.Name) -> ast.AST:
            if item.id not in {"E", array_name}:
                return item
            # The Name nested inside a Subscript is always a Load.  The
            # Subscript itself inherits the original context, so the target
            # E remains Store while phase/alpha remain Load.
            value = ast.copy_location(ast.Name(id=item.id, ctx=ast.Load()), item)
            sliced = ast.Subscript(
                value=value,
                slice=ast.Slice(
                    lower=ast.Name(id="lo", ctx=ast.Load()),
                    upper=ast.Name(id="hi", ctx=ast.Load()),
                ),
                ctx=copy.deepcopy(item.ctx),
            )
            return ast.copy_location(sliced, item)

    result = SliceArrays().visit(node)
    _require(isinstance(result, ast.AugAssign), "C_WORKER_SLICE_RESULT_INVALID")
    ast.fix_missing_locations(result)
    return result


def _worker_definition(
    name: str,
    update: ast.AugAssign,
    array_name: str,
    source_path: Path,
) -> tuple[ast.FunctionDef, dict[str, Any]]:
    original_dump = ast.dump(update, include_attributes=False)
    sliced = _slice_update(update, array_name)
    receipt = {
        "schema": INSTRUMENT_SCHEMA + ".worker",
        "name": name,
        "array_name": array_name,
        "axis": 0,
        "source": str(source_path),
        "original_ast_sha256": _ast_hash(update),
        "sliced_ast_sha256": _ast_hash(sliced),
        "original_ast": original_dump,
        "sliced_ast": ast.dump(sliced, include_attributes=False),
        "astype_copy_false": _is_xp_exp_astype(sliced.value),
    }
    _require(receipt["astype_copy_false"], "C_WORKER_ASTOYPE_COPY_FALSE_MISSING")
    args = ["E", array_name, "onej", "ctype", "lo", "hi"] if array_name == "phase" else [
        "E", array_name, "dz", "ctype", "lo", "hi"
    ]
    definition = ast.FunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=item) for item in args],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[sliced],
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    ast.copy_location(definition, update)
    ast.fix_missing_locations(definition)
    return definition, receipt


def _compile_workers(
    updates: tuple[ast.AugAssign, ast.AugAssign],
    namespace: dict[str, Any],
    source_path: Path,
) -> tuple[dict[str, FunctionType], dict[str, dict[str, Any]]]:
    definitions = []
    receipts: dict[str, dict[str, Any]] = {}
    for kind, name, update, array_name in (
        ("phase", PHASE_WORKER_NAME, updates[0], "phase"),
        ("absorption", ABSORPTION_WORKER_NAME, updates[1], "alpha"),
    ):
        definition, receipt = _worker_definition(name, update, array_name, source_path)
        receipt["kind"] = kind
        definitions.append(definition)
        receipts[kind] = receipt
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    local: dict[str, Any] = {}
    exec(compile(module, str(source_path), "exec"), namespace, local)
    workers = {
        "phase": local.get(PHASE_WORKER_NAME),
        "absorption": local.get(ABSORPTION_WORKER_NAME),
    }
    for kind, worker in workers.items():
        _require(isinstance(worker, FunctionType), f"C_{kind.upper()}_WORKER_COMPILE_FAILED")
        _require(worker.__globals__ is namespace, "C_WORKER_GLOBALS_OBJECT_CHANGED")
        namespace[WORKER_NAMES[kind]] = worker
        worker.__cpu_s46e_c_worker_kind__ = kind
        worker.__cpu_s46e_c_worker_original_ast_sha256__ = receipts[kind]["original_ast_sha256"]
        worker.__cpu_s46e_c_worker_sliced_ast_sha256__ = receipts[kind]["sliced_ast_sha256"]
        worker.__cpu_s46e_c_worker_receipt__ = copy.deepcopy(receipts[kind])
    return workers, receipts


def _closure_cells(function: Callable[..., Any]) -> tuple[dict[str, Any], ...]:
    closure = function.__closure__ or ()
    cells: list[dict[str, Any]] = []
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError as exc:
            raise CInstrumentationError("C_SEAL_EMPTY_CLOSURE_CELL") from exc
        cells.append(
            {
                "cell": cell,
                "value": value,
                "value_id": int(id(value)),
            }
        )
    return tuple(cells)


def _seal(
    function: Callable[..., Any],
    *,
    bindings: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "function": function,
        "code": function.__code__,
        "globals": function.__globals__,
        "closure": function.__closure__,
        "closure_cells": _closure_cells(function),
        "bindings": None if bindings is None else dict(bindings),
        "metadata": None if metadata is None else copy.deepcopy(dict(metadata)),
    }


def verify_seal(seal: Mapping[str, Any]) -> None:
    function = seal.get("function")
    _require(callable(function), "C_SEAL_FUNCTION_MISSING")
    _require(function.__code__ is seal.get("code"), "C_SEAL_CODE_CHANGED")
    _require(function.__globals__ is seal.get("globals"), "C_SEAL_GLOBALS_CHANGED")
    _require(function.__closure__ is seal.get("closure"), "C_SEAL_CLOSURE_CHANGED")
    expected_cells = seal.get("closure_cells")
    actual_cells = _closure_cells(function)
    _require(isinstance(expected_cells, tuple), "C_SEAL_CLOSURE_CELLS_MISSING")
    _require(len(actual_cells) == len(expected_cells), "C_SEAL_CLOSURE_CELL_COUNT_CHANGED")
    for actual, expected in zip(actual_cells, expected_cells):
        _require(actual["cell"] is expected.get("cell"), "C_SEAL_CLOSURE_CELL_CHANGED")
        _require(actual["value"] is expected.get("value"), "C_SEAL_CLOSURE_VALUE_CHANGED")
        _require(actual["value_id"] == expected.get("value_id"), "C_SEAL_CLOSURE_VALUE_ID_CHANGED")
    bindings = seal.get("bindings")
    if bindings is not None:
        _require(isinstance(bindings, Mapping), "C_SEAL_BINDINGS_INVALID")
        values = tuple(item["value"] for item in actual_cells)
        for expected in bindings.values():
            _require(any(value is expected for value in values), "C_SEAL_BINDING_NOT_IN_CLOSURE")
    metadata = seal.get("metadata")
    if metadata is not None:
        _require(isinstance(metadata, Mapping), "C_SEAL_METADATA_INVALID")
        for name, expected in metadata.items():
            _require(getattr(function, name, None) == expected, "C_SEAL_METADATA_CHANGED")


def _make_helpers(
    pool: CPool,
    xp: Any,
    backend: Mapping[str, Any],
    *,
    private_globals: dict[str, Any],
    phase_worker: Callable[..., Any],
    absorption_worker: Callable[..., Any],
) -> dict[str, Callable[..., Any]]:
    if not isinstance(pool, CPool):
        raise CInstrumentationError("C_POOL_TYPE")

    def guard(E: Any, phase: Any, alpha: Any, dz: Any, dn_gas: Any, k0: Any) -> None:
        validate_runtime_backend(xp, backend)
        if private_globals.get("xp") is not xp:
            raise CInstrumentationError("C_PRIVATE_XP_SLOT_CHANGED")
        if pool.closed:
            raise CPoolError("C_RETIRED_WORKER_POOL")
        # This check runs before the source prelude and before either E write.
        CPool.validate_domain(E, phase, alpha, dn_gas=dn_gas)
        try:
            float(dz)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CPoolError("C_DZ_SCALAR") from exc

    def phase(E: Any, onej: Any, phase_array: Any, ctype: Any) -> Any:
        validate_runtime_backend(xp, backend)
        operation = lambda lo, hi: phase_worker(E, phase_array, onej, ctype, lo, hi)
        return pool.phase_update(E, operation, phase=phase_array, ctype=ctype)

    def absorption(E: Any, alpha: Any, dz: Any, ctype: Any) -> Any:
        validate_runtime_backend(xp, backend)
        operation = lambda lo, hi: absorption_worker(E, alpha, dz, ctype, lo, hi)
        return pool.absorption_update(E, operation, alpha=alpha, ctype=ctype)

    guard.__name__ = GUARD_NAME
    phase.__name__ = PHASE_HELPER_NAME
    absorption.__name__ = ABSORPTION_HELPER_NAME
    for function, kind in ((guard, "C_APPLY_GUARD"), (phase, "C_PHASE_UPDATE"), (absorption, "C_ABSORPTION_UPDATE")):
        function.__cpu_s46e_c_kind__ = kind
        function.__cpu_s46e_c_pool__ = pool
        function.__cpu_s46e_c_backend__ = backend["xp_object"]
    phase.__cpu_s46e_c_worker__ = phase_worker
    absorption.__cpu_s46e_c_worker__ = absorption_worker
    return {
        GUARD_NAME: guard,
        PHASE_HELPER_NAME: phase,
        ABSORPTION_HELPER_NAME: absorption,
    }


def compile_private_apply(original: Callable[..., Any], pool: CPool) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Compile a private C math function from the frozen original source."""

    path = _source_path(original)
    binding = source_binding(original)
    raw = pv.canonical_lf_bytes(path.read_bytes()).decode("utf-8")
    tree = ast.parse(raw, filename=str(path))
    original_function = copy.deepcopy(_function_node(tree))
    worker_updates = _validated_pointwise_updates(original_function)
    function = copy.deepcopy(original_function)
    backend = validate_cpu_backend(getattr(original, "__globals__", {}).get("xp"))
    namespace = dict(getattr(original, "__globals__", {}))
    workers, worker_receipts = _compile_workers(worker_updates, namespace, path)
    helpers = _make_helpers(
        pool,
        backend["xp_object"],
        backend,
        private_globals=namespace,
        phase_worker=workers["phase"],
        absorption_worker=workers["absorption"],
    )
    namespace.update(helpers)
    receipt = instrument_apply_nonlinear(function)
    compiled = _compile_function(function, namespace, path)
    compiled.__name__ = getattr(original, "__name__", compiled.__name__)
    compiled.__qualname__ = getattr(original, "__qualname__", compiled.__qualname__)
    compiled.__module__ = getattr(original, "__module__", compiled.__module__)
    compiled.__cpu_s46e_c_original__ = original
    compiled.__cpu_s46e_c_pool__ = pool
    compiled.__cpu_s46e_c_backend__ = backend
    compiled.__cpu_s46e_c_source_binding__ = binding
    compiled.__cpu_s46e_c_instrumentation__ = receipt
    helper_bindings = {
        GUARD_NAME: {
            "pool": pool,
            "backend": backend,
            "private_globals": namespace,
            "xp": backend["xp_object"],
        },
        PHASE_HELPER_NAME: {
            "pool": pool,
            "backend": backend,
            "xp": backend["xp_object"],
            "worker": workers["phase"],
        },
        ABSORPTION_HELPER_NAME: {
            "pool": pool,
            "backend": backend,
            "xp": backend["xp_object"],
            "worker": workers["absorption"],
        },
    }
    compiled.__cpu_s46e_c_helper_seals__ = {
        name: _seal(function, bindings=helper_bindings[name]) for name, function in helpers.items()
    }
    compiled.__cpu_s46e_c_workers__ = dict(workers)
    compiled.__cpu_s46e_c_worker_receipts__ = copy.deepcopy(worker_receipts)
    compiled.__cpu_s46e_c_worker_seals__ = {
        kind: _seal(
            worker,
            metadata={
                "__cpu_s46e_c_worker_kind__": worker_receipts[kind]["kind"],
                "__cpu_s46e_c_worker_original_ast_sha256__": worker_receipts[kind]["original_ast_sha256"],
                "__cpu_s46e_c_worker_sliced_ast_sha256__": worker_receipts[kind]["sliced_ast_sha256"],
                "__cpu_s46e_c_worker_receipt__": worker_receipts[kind],
            },
        )
        for kind, worker in workers.items()
    }
    compiled.__cpu_s46e_c_private_globals__ = namespace
    compiled.__cpu_s46e_c_function_seal__ = _seal(compiled)
    receipt = {
        **receipt,
        "source_binding": copy.deepcopy(binding),
        "backend": {
            key: value
            for key, value in backend.items()
            if key not in {"xp_object", "fft_dispatch_object", "underlying"}
        },
        "helper_names": dict(HELPER_NAMES),
        "workers": copy.deepcopy(worker_receipts),
    }
    compiled.__cpu_s46e_c_instrumentation__ = receipt
    return compiled, receipt


def verify_private_apply(function: Callable[..., Any]) -> dict[str, Any]:
    if not callable(function):
        raise CInstrumentationError("C_PRIVATE_APPLY_NOT_CALLABLE")
    original = getattr(function, "__cpu_s46e_c_original__", None)
    pool = getattr(function, "__cpu_s46e_c_pool__", None)
    if not callable(original) or not isinstance(pool, CPool):
        raise CInstrumentationError("C_PRIVATE_APPLY_METADATA_MISSING")
    backend = getattr(function, "__cpu_s46e_c_backend__", None)
    if not isinstance(backend, Mapping):
        raise CInstrumentationError("C_PRIVATE_APPLY_BACKEND_MISSING")
    validate_runtime_backend(getattr(original, "__globals__", {}).get("xp"), backend)
    private_globals = getattr(function, "__cpu_s46e_c_private_globals__", None)
    _require(isinstance(private_globals, dict), "C_PRIVATE_APPLY_GLOBALS_MISSING")
    _require(private_globals is function.__globals__, "C_PRIVATE_APPLY_GLOBALS_OBJECT_CHANGED")
    _require(private_globals.get("xp") is backend["xp_object"], "C_PRIVATE_APPLY_XP_SLOT_CHANGED")
    function_seal = getattr(function, "__cpu_s46e_c_function_seal__", None)
    _require(isinstance(function_seal, Mapping), "C_PRIVATE_APPLY_FUNCTION_SEAL_MISSING")
    verify_seal(function_seal)
    for seal in getattr(function, "__cpu_s46e_c_helper_seals__", {}).values():
        verify_seal(seal)
    workers = getattr(function, "__cpu_s46e_c_workers__", None)
    worker_receipts = getattr(function, "__cpu_s46e_c_worker_receipts__", None)
    worker_seals = getattr(function, "__cpu_s46e_c_worker_seals__", None)
    _require(isinstance(workers, Mapping), "C_PRIVATE_APPLY_WORKERS_MISSING")
    _require(isinstance(worker_receipts, Mapping), "C_PRIVATE_APPLY_WORKER_RECEIPTS_MISSING")
    _require(isinstance(worker_seals, Mapping), "C_PRIVATE_APPLY_WORKER_SEALS_MISSING")
    for kind, name in WORKER_NAMES.items():
        worker = workers.get(kind)
        _require(isinstance(worker, FunctionType), f"C_{kind.upper()}_WORKER_MISSING")
        _require(worker.__globals__ is private_globals, "C_WORKER_GLOBALS_OBJECT_CHANGED")
        _require(private_globals.get(name) is worker, "C_WORKER_GLOBAL_BINDING_CHANGED")
        receipt = worker_receipts.get(kind)
        _require(isinstance(receipt, Mapping), f"C_{kind.upper()}_WORKER_RECEIPT_MISSING")
        _require(receipt.get("name") == name, "C_WORKER_NAME_CHANGED")
        _require(receipt == getattr(worker, "__cpu_s46e_c_worker_receipt__", None), "C_WORKER_RECEIPT_CHANGED")
        _require(
            receipt.get("original_ast_sha256") == getattr(worker, "__cpu_s46e_c_worker_original_ast_sha256__", None)
            and receipt.get("sliced_ast_sha256") == getattr(worker, "__cpu_s46e_c_worker_sliced_ast_sha256__", None),
            "C_WORKER_AST_HASH_CHANGED",
        )
        seal = worker_seals.get(kind)
        _require(isinstance(seal, Mapping), f"C_{kind.upper()}_WORKER_SEAL_MISSING")
        _require(seal.get("function") is worker, "C_WORKER_BINDING_CHANGED")
        verify_seal(seal)
        helper = getattr(function, "__cpu_s46e_c_helper_seals__", {}).get(HELPER_NAMES[kind])
        _require(isinstance(helper, Mapping), "C_WORKER_HELPER_SEAL_MISSING")
        helper_function = helper.get("function")
        _require(callable(helper_function), "C_WORKER_HELPER_MISSING")
        helper_values = tuple(item["value"] for item in _closure_cells(helper_function))
        _require(any(value is worker for value in helper_values), "C_WORKER_HELPER_BINDING_CHANGED")
    return {
        "schema": INSTRUMENT_SCHEMA + ".binding",
        "original_code_id": int(id(original.__code__)),
        "private_code_id": int(id(function.__code__)),
        "pool_workers": int(pool.count),
        "backend_xp_type": str(backend["xp_type"]),
        "prelude_preserved": True,
        "two_rounds": True,
        "workers": {
            kind: {
                "original_ast_sha256": worker_receipts[kind]["original_ast_sha256"],
                "sliced_ast_sha256": worker_receipts[kind]["sliced_ast_sha256"],
            }
            for kind in WORKER_NAMES
        },
    }


__all__ = [
    "ABSORPTION_HELPER_NAME",
    "CInstrumentationError",
    "GUARD_NAME",
    "HELPER_NAMES",
    "INSTRUMENT_SCHEMA",
    "PHASE_HELPER_NAME",
    "PHASE_WORKER_NAME",
    "ABSORPTION_WORKER_NAME",
    "compile_private_apply",
    "instrument_apply_nonlinear",
    "source_binding",
    "strip_generated_nodes",
    "validate_cpu_backend",
    "validate_runtime_backend",
    "verify_private_apply",
]
