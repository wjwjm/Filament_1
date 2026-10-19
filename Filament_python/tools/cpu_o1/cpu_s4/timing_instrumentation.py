"""Source-bound AST timing for the active S4 Raman PROFILE clone only.

The helper wraps existing statements in timer contexts without changing their
expressions or order.  It is used only with the private Raman namespace made by
the P1-R2 factory; callers must never install the returned functions globally.
"""
from __future__ import annotations

import ast
import copy
import importlib
import inspect
from pathlib import Path
import subprocess
from types import FunctionType


class InstrumentationError(RuntimeError):
    pass


_SITE_SPECS = {
    "isaacs_raman_stage": (
        ("intensity", "raman.stage.intensity_generation", ("E",), "derived_array"),
        ("product", "raman.stage.response_field_product", ("response", "E"), "derived_array"),
        ("normalized_derivative", "raman.stage.spectral_derivative_output",
         ("multiplier", "product"), "spectral_output"),
        ("rhs", "raman.stage.rhs_output", ("product", "normalized_derivative"),
         "derived_array"),
    ),
    "apply_isaacs_raman_operator_step": (
        ("predictor", "raman.heun.predictor_update", ("E", "k1"), "heun_predictor"),
        ("stage2", "raman.heun.stage2_recomputed", (), "stage_call"),
        ("result", "raman.heun.corrector_update", ("E", "k1", "stage2['rhs']"),
         "heun_corrector"),
    ),
    "stable_field_fluence_and_loss": (
        ("before_real", "raman.energy.before_real.float64_cast",
         ("before",), "component_float64_cast"),
        ("after_real", "raman.energy.after_real.float64_cast",
         ("after",), "component_float64_cast"),
        ("before_imag", "raman.energy.before_imag.float64_cast",
         ("before",), "component_float64_cast"),
        ("after_imag", "raman.energy.after_imag.float64_cast",
         ("after",), "component_float64_cast"),
    ),
}


def _is_name(node, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _assigned_names(node: ast.stmt) -> set[str]:
    names = set()
    targets = node.targets if isinstance(node, ast.Assign) else (
        [node.target] if isinstance(node, (ast.AnnAssign, ast.NamedExpr)) else [])
    for target in targets:
        for child in ast.walk(target):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                names.add(child.id)
    return names


def _single_name_assignment(statement: ast.stmt, name: str) -> bool:
    return (isinstance(statement, ast.Assign) and len(statement.targets) == 1
            and _is_name(statement.targets[0], name))


def _astype_call(statement: ast.stmt) -> ast.Call | None:
    value = getattr(statement, "value", None)
    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
            and value.func.attr == "astype"):
        return value
    return None


def _stage2_rhs_reference(node: ast.AST) -> bool:
    return any(isinstance(value, ast.Subscript)
               and _is_name(value.value, "stage2")
               and isinstance(value.slice, ast.Constant)
               and value.slice.value == "rhs" for value in ast.walk(node))


def _energy_component_cast_matches(target: str, statement: ast.stmt) -> bool:
    expected = {
        "before_real": ("before", "real"),
        "after_real": ("after", "real"),
        "before_imag": ("before", "imag"),
        "after_imag": ("after", "imag"),
    }.get(target)
    call = _astype_call(statement)
    if (expected is None or not _single_name_assignment(statement, target)
            or call is None or not isinstance(call.func.value, ast.Attribute)):
        return False
    source_name, component_name = expected
    component = call.func.value
    if component.attr != component_name or not isinstance(component.value, ast.Subscript):
        return False
    source = component.value
    time_slice = source.slice
    if not ( _is_name(source.value, source_name)
            and isinstance(time_slice, ast.Slice)
            and _is_name(time_slice.lower, "start")
            and _is_name(time_slice.upper, "stop")
            and time_slice.step is None):
        return False
    dtype_ok = (bool(call.args) and isinstance(call.args[0], ast.Attribute)
                and _is_name(call.args[0].value, "xp")
                and call.args[0].attr == "float64")
    copy_false = any(keyword.arg == "copy"
                     and isinstance(keyword.value, ast.Constant)
                     and keyword.value.value is False for keyword in call.keywords)
    return dtype_ok and copy_false


def _target_matches(function_name: str, target: str, statement: ast.stmt) -> bool:
    if target not in _assigned_names(statement):
        return False
    value = getattr(statement, "value", None)
    if function_name == "apply_isaacs_raman_operator_step":
        if target == "stage2":
            return (_single_name_assignment(statement, "stage2")
                    and isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id == "isaacs_raman_stage")
        if target == "result":
            return (_single_name_assignment(statement, "result")
                    and _astype_call(statement) is not None
                    and _stage2_rhs_reference(value))
        if target == "predictor":
            return (_single_name_assignment(statement, "predictor")
                    and _astype_call(statement) is not None
                    and any(_is_name(node, "k1") for node in ast.walk(value)))
    if function_name == "stable_field_fluence_and_loss":
        return _energy_component_cast_matches(target, statement)
    return True


def _function_node(tree: ast.Module, function: FunctionType) -> ast.FunctionDef:
    expected_line = int(function.__code__.co_firstlineno)
    matches = [node for node in tree.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == function.__name__ and node.lineno == expected_line]
    if len(matches) != 1 or not isinstance(matches[0], ast.FunctionDef):
        raise InstrumentationError(
            f"source function identity is ambiguous: {function.__name__}:{expected_line}")
    if function.__closure__:
        raise InstrumentationError(f"cannot source-clone closure function {function.__name__}")
    return matches[0]


def _source_binding(function: FunctionType, source_path: Path, source_bytes: bytes,
                    locations: list[dict]) -> dict:
    # The repository provenance module owns canonical text hash semantics.
    from tools.hpc_ops.provenance_v2 import canonical_lf_sha256
    from tools.cpu_o1.cpu_o1 import P1R2_SOURCE_BLOBS

    repo_root = Path(__file__).resolve().parents[4]
    relative_path = "Filament_python/KHz_filament/raman.py"
    expected_path = (repo_root / relative_path).resolve(strict=True)
    if source_path.resolve(strict=True) != expected_path:
        raise InstrumentationError("active Raman function source path differs from the frozen module path")
    expected_blob = P1R2_SOURCE_BLOBS.get(relative_path)
    if not expected_blob:
        raise InstrumentationError("frozen Raman Git blob identity is unavailable")
    expected_bytes = subprocess.run(["git", "cat-file", "blob", expected_blob],
        cwd=repo_root, capture_output=True, check=False).stdout
    expected_hash = canonical_lf_sha256(expected_bytes)
    source_hash = canonical_lf_sha256(source_bytes)
    if not expected_bytes or source_hash != expected_hash:
        raise InstrumentationError("Raman source differs from the frozen P1-R2 blob identity")

    return {
        "module": str(function.__module__),
        "source_path": source_path.as_posix(),
        "source_canonical_lf_sha256": source_hash,
        "frozen_git_blob_oid": expected_blob,
        "function": function.__name__,
        "function_first_line": int(function.__code__.co_firstlineno),
        "statement_locations": locations,
        "transformation": "private_ast_timer_context_v1",
    }


def _location_text(source_path: Path, line: int, column: int) -> str:
    return f"{source_path.as_posix()}:{int(line)}:{int(column)}"


def _input_expression(text: str) -> ast.expr:
    if text == "stage2['rhs']":
        return ast.Subscript(
            value=ast.Name(id="stage2", ctx=ast.Load()),
            slice=ast.Constant(value="rhs"), ctx=ast.Load())
    return ast.Name(id=text, ctx=ast.Load())


def _call_expr(scope_name: str, operation: str, inputs: tuple[str, ...],
               output: str, kind: str) -> ast.Expr:
    call = ast.Call(
        func=ast.Attribute(value=ast.Name(id=scope_name, ctx=ast.Load()),
                           attr="observe_arrays", ctx=ast.Load()),
        args=[],
        keywords=[
            ast.keyword(arg="operation", value=ast.Constant(value=operation)),
            ast.keyword(arg="inputs", value=ast.Tuple(
                elts=[_input_expression(name) for name in inputs], ctx=ast.Load())),
            ast.keyword(arg="output", value=ast.Name(id=output, ctx=ast.Load())),
            ast.keyword(arg="kind", value=ast.Constant(value=kind)),
        ])
    node = ast.Expr(value=call)
    node._s4_inserted_array_observation = True
    return node


def instrument_active_raman_function(function: FunctionType, timer) -> tuple[FunctionType, dict]:
    """Clone and instrument one frozen active Raman function in a private namespace.

    Supported functions and their exact statement targets are intentionally
    fixed above.  A changed or ambiguous source fails closed before execution.
    The returned source binding records the repository-authoritative canonical
    LF digest and exact statement line/column locations.
    """
    if not inspect.isfunction(function) or function.__name__ not in _SITE_SPECS:
        raise InstrumentationError(
            "only source-audited full_isaacs_eq27 functions and its stable-energy helper may be instrumented")
    module = importlib.import_module(str(function.__module__))
    if function.__globals__ is module.__dict__:
        raise InstrumentationError("refusing to instrument a global science-module function")
    source_name = inspect.getsourcefile(function)
    if not source_name:
        raise InstrumentationError(f"source file unavailable for {function.__name__}")
    source_path = Path(source_name).resolve(strict=True)
    source_bytes = source_path.read_bytes()
    source_text = source_bytes.decode("utf-8", errors="strict")
    tree = ast.parse(source_text, filename=str(source_path))
    original_node = _function_node(tree, function)
    transformed = copy.deepcopy(original_node)
    specs = _SITE_SPECS[function.__name__]
    found: dict[str, list[ast.stmt]] = {target: [] for target, *_ in specs}
    for statement in ast.walk(transformed):
        if not isinstance(statement, ast.stmt):
            continue
        for target, *_ in specs:
            if _target_matches(function.__name__, target, statement):
                found[target].append(statement)
    bad = {name: len(nodes) for name, nodes in found.items() if len(nodes) != 1}
    if bad:
        raise InstrumentationError(f"statement locations are not unique: {bad}")

    scope_name = "__cpu_s4_profile_scope__"
    all_names = {node.id for node in ast.walk(transformed) if isinstance(node, ast.Name)}
    while scope_name in all_names:
        scope_name += "_"
    by_identity = {id(nodes[0]): (target, *next(spec[1:] for spec in specs if spec[0] == target))
                   for target, nodes in found.items()}
    locations = []
    def rewrite(node):
        for field, value in list(ast.iter_fields(node)):
            if not isinstance(value, list):
                if isinstance(value, ast.AST):
                    rewrite(value)
                continue
            rewritten = []
            for statement in value:
                if not isinstance(statement, ast.stmt):
                    rewritten.append(statement)
                    continue
                spec = by_identity.get(id(statement))
                if spec is None:
                    rewrite(statement)
                    rewritten.append(statement)
                    continue
                target, operation, inputs, kind = spec
                site = _location_text(source_path, statement.lineno, statement.col_offset)
                role = {
                    "predictor": "predictor_update",
                    "stage2": "stage2_recomputed",
                    "result": "corrector_update",
                    "intensity": "stage_expression",
                    "product": "stage_expression",
                    "normalized_derivative": "stage_expression",
                    "rhs": "stage_expression",
                    "before_real": "energy_component_float64_cast",
                    "after_real": "energy_component_float64_cast",
                    "before_imag": "energy_component_float64_cast",
                    "after_imag": "energy_component_float64_cast",
                }[target]
                keywords = [
                    ast.keyword(arg="callsite", value=ast.Constant(value=site)),
                    ast.keyword(arg="stage_role", value=ast.Constant(value=role)),
                    ast.keyword(arg="parent_stage", value=ast.Constant(
                        f"raman.{function.__name__}")),
                ]
                context = ast.Call(
                    func=ast.Attribute(value=ast.Name(id="__s4_timer__", ctx=ast.Load()),
                                       attr="statement", ctx=ast.Load()),
                    args=[ast.Constant(value=operation)], keywords=keywords)
                with_node = ast.With(
                    items=[ast.withitem(context_expr=context,
                        optional_vars=ast.Name(id=scope_name, ctx=ast.Store()))],
                    body=[statement], type_comment=None)
                with_node._s4_timer_instrumentation = True
                ast.copy_location(with_node, statement)
                if kind != "stage_call":
                    observation = _call_expr(scope_name, operation, inputs, target, kind)
                    ast.copy_location(observation, statement)
                    with_node.body.append(observation)
                rewritten.append(with_node)
                locations.append({"operation": operation, "line": int(statement.lineno),
                    "column": int(statement.col_offset), "stage_role": role})
            setattr(node, field, rewritten)
    rewrite(transformed)

    # Remove only the nodes generated above and prove the frozen statements are
    # byte-order/AST-equivalent after that removal.
    restored = strip_s4_timing_nodes(transformed)
    if ast.dump(restored, include_attributes=False) != ast.dump(
            original_node, include_attributes=False):
        raise InstrumentationError("stripped PROFILE AST differs from the frozen function AST")

    module = ast.Module(body=[transformed], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = function.__globals__
    if namespace.get("__name__") != "KHz_filament.raman":
        raise InstrumentationError("active Raman function is not in the private Raman namespace")
    namespace["__s4_timer__"] = timer
    compiled = compile(module, str(source_path), "exec")
    exec(compiled, namespace)
    instrumented = namespace[function.__name__]
    instrumented.__defaults__ = function.__defaults__
    instrumented.__kwdefaults__ = function.__kwdefaults__
    instrumented.__annotations__ = dict(function.__annotations__)
    instrumented.__doc__ = function.__doc__
    instrumented.__qualname__ = function.__qualname__
    binding = _source_binding(function, source_path, source_bytes, locations)
    return instrumented, binding


def strip_s4_timing_nodes(function_node: ast.FunctionDef) -> ast.FunctionDef:
    """Remove generated timer scopes and metadata calls for AST equality checks."""
    result = copy.deepcopy(function_node)

    class Stripper(ast.NodeTransformer):
        def generic_visit(self, node):
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    replaced = []
                    for item in value:
                        if getattr(item, "_s4_timer_instrumentation", False):
                            restored_body = [child for child in item.body
                                if not getattr(child, "_s4_inserted_array_observation", False)]
                            replaced.extend(restored_body)
                        elif getattr(item, "_s4_inserted_array_observation", False):
                            continue
                        else:
                            replaced.append(self.visit(item) if isinstance(item, ast.AST) else item)
                    setattr(node, field, replaced)
                elif isinstance(value, ast.AST):
                    setattr(node, field, self.visit(value))
            return node

    return Stripper().visit(result)

