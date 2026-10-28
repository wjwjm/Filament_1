"""Private S4-5D body-clock and saved-fluence AST attachment.

This module owns only the narrow clone seam.  The frozen G2 adapter is cloned
through the accepted S4-5B helper, then receives the original timing and
boundary nodes plus a reversible body clock and the source-bound saved-fluence
call seam.  No production module globals are mutated.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import copy
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from tools.cpu_s4_5b_steady_cost.adapter import clone_g2_namespace
from tools.cpu_s4_5b_steady_cost.instrument import seal_callable, verify_callable_seal
from tools.hpc_ops import provenance_v2 as pv

from .fluence import postprocess_source_proof, second_moment_radius_from_f2d


ADAPTER_SCHEMA = "filament.cpu_s4_5d_fluence_reuse.adapter.v1"
MAX_BODY_ROWS = 8
S45D_HELPER_NAMES = (
    "__cpu_s45d_body__",
    "__cpu_s45d_saved_event__",
    "__cpu_s45d_radius_event__",
    "__cpu_s45d_fwhm_event__",
    "__cpu_s45d_second_moment_radius__",
)


class AdapterBindingError(RuntimeError):
    """A private source or G2 clone contract failed closed."""


def require(ok: bool, why: str) -> None:
    if not ok:
        raise AdapterBindingError(why)


def dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


class BodyClock:
    """The shared original body clock, limited to exactly eight intervals."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.active: int | None = None

    @contextlib.contextmanager
    def body(self, index: int):
        require(
            self.active is None and len(self.rows) < MAX_BODY_ROWS,
            "S45D_BODY_CLOCK_CAPACITY_OR_REENTRY",
        )
        index = int(index)
        require(index == len(self.rows), "S45D_BODY_SOURCE_ORDINAL")
        self.active = index
        wall = time.perf_counter_ns()
        cpu = time.process_time_ns()
        failed = True
        try:
            yield
            failed = False
        finally:
            end_cpu = time.process_time_ns()
            end_wall = time.perf_counter_ns()
            self.rows.append(
                {
                    "interval_index": index,
                    "ordinal": index + 1,
                    "attempt": 1,
                    "start_wall_ns": wall,
                    "end_wall_ns": end_wall,
                    "start_cpu_ns": cpu,
                    "end_cpu_ns": end_cpu,
                    "wall_ns": end_wall - wall,
                    "cpu_ns": end_cpu - cpu,
                    "failed": failed,
                }
            )
            self.active = None

    def snapshot(self) -> dict[str, Any]:
        rows = copy.deepcopy(self.rows)
        return {
            "schema": ADAPTER_SCHEMA + ".body_clock.v1",
            "rows": rows,
            "first_body_s": sum(row["wall_ns"] for row in rows if row["ordinal"] == 1)
            * 1e-9,
            "later_body_s": sum(row["wall_ns"] for row in rows if row["ordinal"] >= 2)
            * 1e-9,
            "body_contract": (
                "Identical original for-body first statement through original bottom, "
                "including save/progress/control"
            ),
            "fine_profile_enabled": False,
            "empty_scope_subtracted": False,
            "max_rows": MAX_BODY_ROWS,
        }


class FluenceCounter:
    """Scalar-only qualification counters; no field or array is retained."""

    def __init__(self, *, role: str, candidate_applied: bool) -> None:
        self.role = str(role)
        self.candidate_applied = bool(candidate_applied)
        self.saved_calls = 0
        self.outer_integrals = 0
        self.inner_integrals = 0
        self.radius_calls = 0
        self.fwhm_calls = 0
        self.first_saved_interval: int | None = None
        self._saved_intervals: list[int] = []

    def saved_event(self, interval_index: int, dt: float) -> None:
        interval = int(interval_index)
        require(type(dt) is float, "S45D_DT_MUST_BE_PYTHON_FLOAT")
        self.saved_calls += 1
        self.outer_integrals += 1
        self._saved_intervals.append(interval)
        if self.first_saved_interval is None:
            self.first_saved_interval = interval

    def radius_event(self, value: Any) -> Any:
        # The wrapper is called after the radius calculation has returned.  In
        # the reference role that calculation contains the second trapezoid;
        # D's private post-processing contains no inner time integration.
        self.radius_calls += 1
        if not self.candidate_applied:
            self.inner_integrals += 1
        return value

    def fwhm_event(self, value: Any) -> Any:
        self.fwhm_calls += 1
        return value

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": ADAPTER_SCHEMA + ".fluence_counter.v1",
            "role": self.role,
            "saved_calls": int(self.saved_calls),
            "outer_integrals": int(self.outer_integrals),
            "inner_integrals": int(self.inner_integrals),
            "radius_calls": int(self.radius_calls),
            "fwhm_calls": int(self.fwhm_calls),
            "fwhm_calls_total_known": int(self.fwhm_calls * 2),
            "total_fwhm_known2S": int(self.fwhm_calls * 2),
            "first_saved": (
                None
                if self.first_saved_interval is None
                else {"interval_index": int(self.first_saved_interval)}
            ),
            "saved_intervals": [int(value) for value in self._saved_intervals],
            "candidate_applied": bool(self.candidate_applied),
            "fallback": False,
            "rejected": False,
            "count_method": {
                "saved_calls": "source-bound scalar marker immediately after original F2D integral",
                "outer_integrals": "same marker after successful original F2D integral",
                "dt_contract": "marker receives the actual Python float dt from the saved branch",
                "inner_integrals": (
                    "source-bound inference from the original radius call: A retains its "
                    "bound original integral and D private post-processing has none; "
                    "the backend trapezoid itself is not separately hooked"
                ),
                "radius_calls": "source-bound scalar wrapper around w_mom result",
                "fwhm_calls": (
                    "source-bound scalar wrapper around the original F2D fluence FWHM call; "
                    "the rho_maxt FWHM call remains untouched"
                ),
            },
            "unobserved_counts": ["inner_integrals_backend_trapezoid_not_directly_hooked"],
            "arrays_retained": False,
        }


QUAL_HELPER_NAMES=('__cpu_s45d_qual_begin__','__cpu_s45d_qual_radius__','__cpu_s45d_qual_fwhm__')

def _fixed_helper_names(mapping):
    names=set(mapping)
    require(names in (set(S45D_HELPER_NAMES),set(S45D_HELPER_NAMES)|set(QUAL_HELPER_NAMES)), 'S45D_EXACT_OFF_OR_QUAL_HELPER_SET')
    return tuple(sorted(names))

def bind_helper_slots(
    active_clone: Any,
    helpers: Mapping[str, Callable[..., Any]],
    *,
    expected_owners: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Seal private helper slots after the actual clone is captured."""

    namespace = getattr(active_clone, "__globals__", None)
    require(isinstance(namespace, dict), "S45D_ACTIVE_CLONE_GLOBALS_MISSING")
    names=_fixed_helper_names(helpers)
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        function = helpers[name]
        require(namespace.get(name) is function, f"S45D_HELPER_SLOT_NOT_BOUND:{name}")
        implementation = getattr(function, "__func__", function)
        require(callable(implementation), f"S45D_HELPER_IMPLEMENTATION_INVALID:{name}")
        owner = getattr(function, "__self__", None)
        expected_owner = expected_owners.get(name)
        if expected_owner is None:
            require(owner is None, f"S45D_HELPER_UNEXPECTED_BOUND_OWNER:{name}")
        else:
            require(owner is expected_owner, f"S45D_HELPER_BOUND_OWNER_CHANGED:{name}")
        result[name] = {
            "slot": function,
            "implementation": implementation,
            "implementation_seal": seal_callable(implementation),
            "owner": owner,
            "owner_id": None if owner is None else int(id(owner)),
        }
    return result


def verify_helper_slots(active_clone: Any, bindings: Mapping[str, Mapping[str, Any]]) -> None:
    """Fail closed if a private helper slot or implementation moved."""

    namespace = getattr(active_clone, "__globals__", None)
    require(isinstance(namespace, dict), "S45D_ACTIVE_CLONE_GLOBALS_MISSING")
    names=_fixed_helper_names(bindings)
    for name in names:
        binding = bindings[name]
        slot = binding.get("slot")
        implementation = binding.get("implementation")
        require(namespace.get(name) is slot, f"S45D_HELPER_SLOT_CHANGED:{name}")
        require(getattr(slot, "__func__", slot) is implementation, f"S45D_HELPER_FUNCTION_CHANGED:{name}")
        require(getattr(slot, "__self__", None) is binding.get("owner"), f"S45D_HELPER_OWNER_CHANGED:{name}")
        seal = binding.get("implementation_seal")
        require(isinstance(seal, Mapping), f"S45D_HELPER_SEAL_MISSING:{name}")
        try:
            verify_callable_seal(seal)
        except Exception as exc:  # normalize the existing 5B seal failure
            raise AdapterBindingError(f"S45D_HELPER_SEAL_CHANGED:{name}") from exc


def _name(node: ast.AST, expected: str) -> bool:
    return isinstance(node, ast.Name) and node.id == expected


def _attr(node: ast.AST, base: str, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id == base
    )


def _keyword_map(node: ast.Call) -> dict[str, ast.keyword]:
    result: dict[str, ast.keyword] = {}
    for keyword in node.keywords:
        require(keyword.arg is not None, "S45D_CALL_STAR_KWARGS_UNSUPPORTED")
        require(keyword.arg not in result, "S45D_CALL_DUPLICATE_KEYWORD")
        result[keyword.arg] = keyword
    return result


def _find_function(module: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(matches) == 1, f"S45D_FUNCTION_COUNT:{name}:{len(matches)}")
    return matches[0]


def _find_single_assignment(function: ast.FunctionDef, target_name: str) -> ast.Assign:
    matches: list[ast.Assign] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) == 1 and _name(node.targets[0], target_name):
            matches.append(node)
    require(len(matches) == 1, f"S45D_ASSIGNMENT_COUNT:{target_name}:{len(matches)}")
    return matches[0]


def _assert_original_f2d_assignment(node: ast.Assign) -> ast.Call:
    require(isinstance(node.value, ast.Call), "S45D_F2D_INTEGRAL_NOT_CALL")
    call = node.value
    require(len(call.args) == 1 and _name(call.args[0], "I_now"), "S45D_F2D_INPUT_NOT_I_NOW")
    keywords = _keyword_map(call)
    require(set(keywords) == {"dx", "axis"}, "S45D_F2D_KEYWORDS_CHANGED")
    require(_name(keywords["dx"].value, "dt"), "S45D_F2D_DT_CHANGED")
    require(isinstance(keywords["axis"].value, ast.Constant), "S45D_F2D_AXIS_CHANGED")
    require(keywords["axis"].value.value == 0, "S45D_F2D_AXIS_CHANGED")
    require(isinstance(call.func, ast.IfExp), "S45D_F2D_TRAPEZOID_SELECTION_CHANGED")
    return call


def _assert_original_radius_call(node: ast.Assign) -> ast.Call:
    require(isinstance(node.value, ast.Call), "S45D_RADIUS_ASSIGNMENT_NOT_CALL")
    call = node.value
    require(isinstance(call.func, ast.Name) and call.func.id == "second_moment_radius", "S45D_RADIUS_FUNCTION_CHANGED")
    require(
        len(call.args) == 3
        and _name(call.args[0], "I_now")
        and _attr(call.args[1], "axes", "x")
        and _attr(call.args[2], "axes", "y"),
        "S45D_RADIUS_ARGUMENT_ORDER_CHANGED",
    )
    keywords = _keyword_map(call)
    require(set(keywords) == {"dt", "frac_keep", "rel_floor"}, "S45D_RADIUS_KEYWORDS_CHANGED")
    require(_name(keywords["dt"].value, "dt"), "S45D_RADIUS_DT_CHANGED")
    return call


def _find_fwhm_calls(function: ast.FunctionDef) -> list[ast.Call]:
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_fwhm_diameter_xy_center"
    ]
    require(len(calls) == 2, f"S45D_FWHM_CALL_COUNT:{len(calls)}")
    for call in calls:
        require(len(call.args) == 4 and not call.keywords, "S45D_FWHM_CALL_SHAPE_CHANGED")
        require(isinstance(call.args[1], ast.Name), "S45D_FWHM_AXES_ARGUMENT_UNEXPECTED")
    fluence_calls = [call for call in calls if _name(call.args[0], "F2D")]
    require(len(fluence_calls) == 1, f"S45D_FLUENCE_FWHM_CALL_COUNT:{len(fluence_calls)}")
    return fluence_calls


def _insert_after(root: ast.AST, target: ast.AST, inserted: ast.stmt) -> None:
    """Insert one statement after ``target`` in its owning statement list."""

    for _field, value in ast.iter_fields(root):
        if isinstance(value, list):
            for index, item in enumerate(value):
                if item is target:
                    value.insert(index + 1, inserted)
                    return
                if isinstance(item, ast.AST):
                    try:
                        _insert_after(item, target, inserted)
                    except LookupError:
                        pass
                    else:
                        return
        elif isinstance(value, ast.AST):
            try:
                _insert_after(value, target, inserted)
            except LookupError:
                pass
            else:
                return
    raise LookupError("S45D_TARGET_STATEMENT_NOT_FOUND")


def _wrapper(name: str, value: ast.AST) -> ast.Call:
    return ast.copy_location(
        ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=[value], keywords=[]),
        value,
    )


def _apply_fluence_nodes(function: ast.FunctionDef, *, candidate_applied: bool, qualification: bool=False) -> dict[str, Any]:
    f2d_assignment = _find_single_assignment(function, "F2D")
    _assert_original_f2d_assignment(f2d_assignment)
    radius_assignment = _find_single_assignment(function, "w_mom")
    original_radius = _assert_original_radius_call(radius_assignment)
    fwhm_calls = _find_fwhm_calls(function)

    saved_marker = ast.Expr(
        value=ast.Call(
            func=ast.Name(id="__cpu_s45d_saved_event__", ctx=ast.Load()),
            args=[
                ast.Name(id="interval_index", ctx=ast.Load()),
                ast.Name(id="dt", ctx=ast.Load()),
            ],
            keywords=[],
        )
    )
    ast.copy_location(saved_marker, f2d_assignment)
    try:
        _insert_after(function, f2d_assignment, saved_marker)
    except LookupError as exc:
        raise AdapterBindingError("S45D_F2D_SAVED_MARKER_INSERT_FAILED") from exc
    if qualification:
        keywords=_keyword_map(original_radius)
        begin=ast.Expr(value=ast.Call(func=ast.Name(id='__cpu_s45d_qual_begin__',ctx=ast.Load()),args=[
            ast.Name(id=name,ctx=ast.Load()) for name in ('I_now','F2D','dt','axes','y0','x0')]+[
            copy.deepcopy(keywords['frac_keep'].value),copy.deepcopy(keywords['rel_floor'].value),ast.Name(id='interval_index',ctx=ast.Load())],keywords=[]))
        _insert_after(function,saved_marker,ast.copy_location(begin,f2d_assignment))

    radius_call = copy.deepcopy(original_radius)
    if candidate_applied:
        keywords = _keyword_map(radius_call)
        radius_call.func = ast.copy_location(
            ast.Name(id="__cpu_s45d_second_moment_radius__", ctx=ast.Load()),
            radius_call.func,
        )
        radius_call.args = [
            ast.Name(id="F2D", ctx=ast.Load()),
            copy.deepcopy(original_radius.args[1]),
            copy.deepcopy(original_radius.args[2]),
        ]
        radius_call.keywords = [
            copy.deepcopy(keywords["frac_keep"]),
            copy.deepcopy(keywords["rel_floor"]),
        ]
    radius_assignment.value = _wrapper("__cpu_s45d_radius_event__", radius_call)
    if qualification:radius_assignment.value=_wrapper('__cpu_s45d_qual_radius__',radius_assignment.value)

    for call in fwhm_calls:
        call_parent = _wrapper("__cpu_s45d_fwhm_event__", copy.deepcopy(call))
        if qualification:call_parent=_wrapper('__cpu_s45d_qual_fwhm__',call_parent)
        # Replace this call in its owning expression without touching its
        # arguments or the original FWHM function itself.
        replaced = False
        for parent in ast.walk(function):
            for field, value in ast.iter_fields(parent):
                if isinstance(value, list):
                    for index, item in enumerate(value):
                        if item is call:
                            value[index] = call_parent
                            replaced = True
                elif value is call:
                    setattr(parent, field, call_parent)
                    replaced = True
        require(replaced, "S45D_FWHM_WRAPPER_INSERT_FAILED")

    return {
        "saved_marker_count": 1,
        "f2d_integral_count": 1,
        "radius_call_count": 1,
        "fwhm_call_count": len(fwhm_calls),
        "total_original_fwhm_call_count": 2,
        "wrapped_fwhm_input": "F2D",
        "rho_maxt_fwhm_untouched": True,
        "candidate_applied": bool(candidate_applied),
        "first_save_qualification_enabled": bool(qualification),
        "inner_integral_removed": bool(candidate_applied),
        "f2d_input": "I_now",
        "f2d_axis": 0,
        "f2d_dt_expression": "dt",
        "radius_helper": (
            "__cpu_s45d_second_moment_radius__" if candidate_applied else "second_moment_radius"
        ),
        "fwhm_helper": "_fwhm_diameter_xy_center",
    }


class _StripS45D(ast.NodeTransformer):
    """Remove only S4-5D generated statements for inverse AST proof."""

    def visit_With(self, node: ast.With):  # noqa: N802 - ast API spelling
        if len(node.items) == 1 and isinstance(node.items[0].context_expr, ast.Call):
            call = node.items[0].context_expr
            if isinstance(call.func, ast.Name) and call.func.id == "__cpu_s45d_body__":
                return [self.visit(item) for item in node.body]
        return self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr):  # noqa: N802 - ast API spelling
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            if value.func.id == "__cpu_s45d_saved_event__":
                require(len(value.args) == 2 and not value.keywords, "S45D_SAVED_MARKER_SHAPE_CHANGED")
                return None
            if value.func.id=='__cpu_s45d_qual_begin__':
                require(len(value.args)==9 and not value.keywords,'S45D_QUAL_BEGIN_SHAPE')
                return None
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):  # noqa: N802 - ast API spelling
        node = self.generic_visit(node)
        if not isinstance(node.func, ast.Name):
            return node
        if node.func.id in ('__cpu_s45d_qual_radius__','__cpu_s45d_qual_fwhm__'):
            require(len(node.args)==1 and not node.keywords,'S45D_QUAL_RESULT_SHAPE')
            return node.args[0]
        if node.func.id == "__cpu_s45d_radius_event__":
            require(len(node.args) == 1 and not node.keywords, "S45D_RADIUS_WRAPPER_SHAPE_CHANGED")
            return node.args[0]
        if node.func.id == "__cpu_s45d_fwhm_event__":
            require(len(node.args) == 1 and not node.keywords, "S45D_FWHM_WRAPPER_SHAPE_CHANGED")
            return node.args[0]
        if node.func.id == "__cpu_s45d_second_moment_radius__":
            require(
                len(node.args) == 3
                and _name(node.args[0], "F2D")
                and _attr(node.args[1], "axes", "x")
                and _attr(node.args[2], "axes", "y"),
                "S45D_PRIVATE_RADIUS_CALL_SHAPE_CHANGED",
            )
            keywords = _keyword_map(node)
            require(set(keywords) == {"frac_keep", "rel_floor"}, "S45D_PRIVATE_RADIUS_KEYWORDS_CHANGED")
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="second_moment_radius", ctx=ast.Load()),
                    args=[
                        ast.Name(id="I_now", ctx=ast.Load()),
                        ast.Attribute(
                            value=ast.Name(id="axes", ctx=ast.Load()),
                            attr="x",
                            ctx=ast.Load(),
                        ),
                        ast.Attribute(
                            value=ast.Name(id="axes", ctx=ast.Load()),
                            attr="y",
                            ctx=ast.Load(),
                        ),
                    ],
                    keywords=[
                        ast.keyword(arg="dt", value=ast.Name(id="dt", ctx=ast.Load())),
                        copy.deepcopy(keywords["frac_keep"]),
                        copy.deepcopy(keywords["rel_floor"]),
                    ],
                ),
                node,
            )
        return node


def _ast_hash(node: ast.AST) -> str:
    return pv.canonical_lf_sha256(dump(node).encode("utf-8"))


def make_compile_interceptor(
    *,
    role: str,
    mode: str,
    bindings: tuple[tuple[str, str], ...],
    g2_namespace: Mapping[str, Any],
    body_clock: BodyClock,
    candidate_applied: bool,
    source_dependencies: Mapping[str, Any],
) -> Callable[..., Any]:
    """Create the private interceptor installed before the frozen G2 guard."""

    mode = str(mode).upper()
    if mode == "PURE_OFF_REAL8":
        mode = "PURE_OFF_SMALLGRID"
    require(mode in ("QUALIFY_BOUNDARY", "PURE_OFF_SMALLGRID"), "S45D_G2_MODE_INVALID")
    timing_module = g2_namespace.get("_timing")
    inject_timing = getattr(timing_module, "_inject_timing", None)
    insert_boundary = g2_namespace.get("_insert_boundary")
    strip_boundary = g2_namespace.get("_strip_boundary")
    structure_without_timing = g2_namespace.get("_structure_without_timing_ast")
    require(callable(inject_timing), "S45D_G2_TIMING_SEAM_UNAVAILABLE")
    require(callable(structure_without_timing), "S45D_G2_STRUCTURE_SEAM_UNAVAILABLE")
    if mode == "QUALIFY_BOUNDARY":
        require(callable(insert_boundary), "S45D_G2_BOUNDARY_INSERT_SEAM_UNAVAILABLE")
        require(callable(strip_boundary), "S45D_G2_BOUNDARY_STRIP_SEAM_UNAVAILABLE")
    dependency_files = source_dependencies.get("files", {})
    diagnostics_record = dependency_files.get("Filament_python/KHz_filament/diagnostics.py")
    require(isinstance(diagnostics_record, Mapping), "S45D_DIAGNOSTICS_SOURCE_RECORD_MISSING")
    diagnostics_path = diagnostics_record.get("path")
    require(isinstance(diagnostics_path, str), "S45D_DIAGNOSTICS_SOURCE_PATH_MISSING")
    post_ast_proof = postprocess_source_proof(diagnostics_path)
    require(post_ast_proof.get("post_ast_equal") is True, "S45D_PRIVATE_POST_AST_PROOF_FAILED")
    require(post_ast_proof.get("public_2d_helper_references") == 0, "S45D_PUBLIC_2D_HELPER_REFERENCE")

    injected = False

    def interceptor(source: Any, filename: Any, compile_mode: str, *args: Any, **kwargs: Any) -> Any:
        nonlocal injected
        if not injected and isinstance(source, ast.Module):
            baseline = copy.deepcopy(source)
            timing_receipt = inject_timing(source, role=role)
            boundary_receipt = None
            if mode == "QUALIFY_BOUNDARY":
                boundary_receipt = insert_boundary(source, bindings)
            function_node = _find_function(source, "propagate_one_pulse")
            loops = [
                node
                for node in ast.walk(function_node)
                if isinstance(node, ast.For)
                and isinstance(node.target, ast.Tuple)
                and [getattr(item, "id", None) for item in node.target.elts]
                == ["interval_index", "interval"]
            ]
            require(len(loops) == 1 and len(loops[0].body) > 10, "S45D_COMPLETE_ORIGINAL_BODY_ANCHOR")
            loop = loops[0]
            body_with = ast.With(
                items=[
                    ast.withitem(
                        context_expr=ast.Call(
                            func=ast.Name(id="__cpu_s45d_body__", ctx=ast.Load()),
                            args=[ast.Name(id="interval_index", ctx=ast.Load())],
                            keywords=[],
                        )
                    )
                ],
                body=loop.body,
            )
            ast.copy_location(body_with, loop.body[0])
            loop.body = [body_with]
            fluence_receipt = _apply_fluence_nodes(
                function_node, candidate_applied=bool(candidate_applied), qualification=mode=='QUALIFY_BOUNDARY'
            )

            stripped = _StripS45D().visit(copy.deepcopy(source))
            require(isinstance(stripped, ast.Module), "S45D_STRIPPED_MODULE_INVALID")
            stripped = structure_without_timing(stripped)
            if mode == "QUALIFY_BOUNDARY":
                stripped, count = strip_boundary(stripped)
                require(count == 1, "S45D_ORIGINAL_BOUNDARY_STRIP_COUNT")
            require(dump(stripped) == dump(baseline), "S45D_G2_AST_CHANGED_BEYOND_PRIVATE_SEAM")
            ast.fix_missing_locations(source)
            interceptor.static_receipt = {
                "schema": ADAPTER_SCHEMA + ".g2_static",
                "role": str(role),
                "mode": mode,
                "timing": copy.deepcopy(timing_receipt),
                "boundary": copy.deepcopy(boundary_receipt),
                "baseline_ast_sha256": _ast_hash(baseline),
                "stripped_ast_sha256": _ast_hash(stripped),
                "old_guard_capture": "AFTER_COMPILE_INSERTION",
                "inverse_ast_exact": True,
                "fluence_reuse": copy.deepcopy(fluence_receipt),
                "postprocess_ast_proof": copy.deepcopy(post_ast_proof),
                "body_clock": {
                    "helper": "__cpu_s45d_body__",
                    "max_rows": MAX_BODY_ROWS,
                    "body_contract": "ORIGINAL_FOR_BODY_FIRST_STATEMENT_THROUGH_ORIGINAL_BOTTOM",
                },
                "source_dependencies": copy.deepcopy(dict(source_dependencies)),
                "scientific_algorithm_changed": False,
                "unknown_observers_added": False,
            }
            injected = True
        return builtins.compile(source, filename, compile_mode, *args, **kwargs)

    interceptor.static_receipt = None
    return interceptor


def _owned_records(owned_identity: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in owned_identity.get("files", {}).values():
        if not isinstance(record, Mapping):
            continue
        relative = record.get("relative_path") or record.get("path")
        if isinstance(relative, str):
            result[relative.replace("\\", "/")] = copy.deepcopy(dict(record))
    return result


def attach(
    raw_propagator: Callable[..., Any],
    dispatch: Any,
    *,
    role: str,
    mode: str,
    capture_root: str | Path,
    owned_identity: Mapping[str, Any],
    body_clock: BodyClock,
    counter: FluenceCounter,
    candidate_applied: bool,
    source_dependencies: Mapping[str, Any],
) -> tuple[Any, Any, Any, dict[str, Any], Any, Any]:
    """Attach a source-bound G2 clone and return the original six-tuple."""

    require(callable(raw_propagator), "S45D_RAW_PROPAGATOR_REQUIRED")
    require(isinstance(body_clock, BodyClock), "S45D_BODY_CLOCK_REQUIRED")
    require(isinstance(counter, FluenceCounter), "S45D_COUNTER_REQUIRED")
    from tools.cpu_s4_checkpoint_support import g2_adapter_path, load_g2_adapter
    from tools.cpu_s4_3g3_smallgrid.scientific_context import frozen_tools

    repo = Path(__file__).resolve().parents[3]
    frozen_tools(repo)
    original = load_g2_adapter()
    expected_adapter = g2_adapter_path()
    require(
        Path(getattr(original, "__file__", "")).resolve() == expected_adapter.resolve(strict=True),
        "S45D_ACTUAL_FROZEN_G2_MODULE",
    )
    namespace = clone_g2_namespace(original)
    common = SimpleNamespace(**vars(original._common))
    common.factory = lambda unused_role: (lambda unused_base: (raw_propagator, dispatch))
    namespace["_common"] = common

    old_record = namespace["_frozen_tracked_record"]
    own = _owned_records(owned_identity)

    def tracked_record(relative: str) -> dict[str, Any] | None:
        key = str(relative).replace("\\", "/")
        record = own.get(key)
        if record is None:
            return old_record(relative)
        required = ("classification", "hash_scope", "git_blob_oid", "canonical_lf_sha256")
        require(all(field in record for field in required), "S45D_OWNED_TRACKED_RECORD_INCOMPLETE")
        return {
            "path": key,
            **{field: record[field] for field in required},
        }

    namespace["_frozen_tracked_record"] = tracked_record

    helpers = {
        "__cpu_s45d_body__": body_clock.body,
        "__cpu_s45d_saved_event__": counter.saved_event,
        "__cpu_s45d_radius_event__": counter.radius_event,
        "__cpu_s45d_fwhm_event__": counter.fwhm_event,
        "__cpu_s45d_second_moment_radius__": second_moment_radius_from_f2d,
    }
    witness=None
    if str(mode).upper()=='QUALIFY_BOUNDARY':
        from .qualification import FirstSaveWitness
        witness=FirstSaveWitness(Path(capture_root).parent/'FLUENCE_QUALIFICATION')
        helpers.update(__cpu_s45d_qual_begin__=witness.begin,__cpu_s45d_qual_radius__=witness.radius,__cpu_s45d_qual_fwhm__=witness.fwhm)
    dispatch.cpu_s45d_witness=witness

    def make_interceptor(*, role: str, mode: str, bindings: tuple[tuple[str, str], ...]):
        return make_compile_interceptor(
            role=role,
            mode=mode,
            bindings=bindings,
            g2_namespace=namespace,
            body_clock=body_clock,
            candidate_applied=bool(candidate_applied),
            source_dependencies=source_dependencies,
        )

    namespace["_make_compile_interceptor"] = make_interceptor
    original_clone_g = namespace.get("_clone_g")
    original_clone_i0 = namespace.get("_clone_i0")
    require(callable(original_clone_g), "S45D_G2_CLONE_G_MISSING")
    require(callable(original_clone_i0), "S45D_G2_CLONE_I0_MISSING")

    def clone_g(*args: Any, helpers: Mapping[str, Any], **kwargs: Any):
        extended = dict(helpers)
        extended.update(helpers_local)
        return original_clone_g(*args, helpers=extended, **kwargs)

    def clone_i0(*args: Any, helpers: Mapping[str, Any], **kwargs: Any):
        extended = dict(helpers)
        extended.update(helpers_local)
        return original_clone_i0(*args, helpers=extended, **kwargs)

    helpers_local = helpers
    namespace["_clone_g"] = clone_g
    namespace["_clone_i0"] = clone_i0

    mode_name = str(mode).upper()
    frozen_mode = "QUALIFY_BOUNDARY" if mode_name == "QUALIFY_BOUNDARY" else "PURE_OFF_REAL8"
    result = namespace["build_role"](
        "G_CONTROLLED_INPUT",
        frozen_mode,
        capture_root=Path(capture_root).resolve(),
    )
    require(isinstance(result, tuple) and len(result) == 6, "S45D_G2_BUILD_ROLE_SIX_TUPLE")
    wrapped, actual_dispatch, clock, binding, audit, lease = result
    require(actual_dispatch is dispatch, "S45D_ACTUAL_DISPATCH_BINDING_CHANGED")
    active_clone = getattr(wrapped, "__g2_inner__", None)
    require(callable(active_clone), "S45D_ACTIVE_CLONE_CAPTURE_MISSING")
    expected_owners = {
        "__cpu_s45d_body__": body_clock,
        "__cpu_s45d_saved_event__": counter,
        "__cpu_s45d_radius_event__": counter,
        "__cpu_s45d_fwhm_event__": counter,
        "__cpu_s45d_second_moment_radius__": None,
    }
    if witness is not None:expected_owners.update({name:witness for name in QUAL_HELPER_NAMES})
    helper_bindings = bind_helper_slots(
        active_clone,
        helpers,
        expected_owners=expected_owners,
    )
    verify_helper_slots(active_clone, helper_bindings)
    actual_dispatch.cpu_s45d_active_clone = active_clone
    actual_dispatch.cpu_s45d_helper_bindings = helper_bindings
    # The interceptor itself is created inside the cloned G2 call.  Its static
    # receipt is exposed through the G2 binding; retain that exact receipt and
    # fail closed if the adapter did not return it.
    static_receipt = copy.deepcopy(binding.get("static_receipt"))
    require(isinstance(static_receipt, dict), "S45D_COMPILE_INTERCEPTOR_STATIC_RECEIPT_MISSING")
    require(static_receipt.get("inverse_ast_exact") is True, "S45D_INVERSE_AST_PROOF_MISSING")
    post_ast = static_receipt.get("postprocess_ast_proof")
    require(isinstance(post_ast, Mapping), "S45D_POST_AST_PROOF_MISSING")
    require(post_ast.get("post_ast_equal") is True, "S45D_POST_AST_PROOF_FALSE")
    require(
        post_ast.get("original_post_ast_sha256") == post_ast.get("private_post_ast_sha256"),
        "S45D_POST_AST_HASH_MISMATCH",
    )

    binding = copy.deepcopy(binding)
    binding.update(
        {
            "s45d_adapter_schema": ADAPTER_SCHEMA,
            "s45d_role": str(role),
            "s45d_mode": mode_name,
            "s45d_candidate_applied": bool(candidate_applied),
            "s45d_body_clock": body_clock.snapshot(),
            "s45d_static_receipt": copy.deepcopy(static_receipt),
            "s45d_owned_source_identity": copy.deepcopy(dict(owned_identity)),
            "s45d_source_dependencies": copy.deepcopy(dict(source_dependencies)),
            "s45d_helper_slots": {
                name: {
                    "code_identity": seal["implementation_seal"]["code_identity"],
                    "owner_id": seal["owner_id"],
                }
                for name, seal in helper_bindings.items()
            },
            "s45d_unknown_observers_added": False,
            "s45d_arrays_retained": False,
        }
    )
    original_wrapped=wrapped
    def guarded(*args,**kwargs):
        verify_helper_slots(active_clone,helper_bindings)
        try:return original_wrapped(*args,**kwargs)
        finally:
            if witness is not None:witness.close()
    guarded.__dict__.update(original_wrapped.__dict__)
    guarded.__name__=original_wrapped.__name__
    guarded.__module__=original_wrapped.__module__
    return guarded, actual_dispatch, clock, binding, audit, lease


__all__ = [
    "ADAPTER_SCHEMA",
    "AdapterBindingError",
    "BodyClock",
    "FluenceCounter",
    "MAX_BODY_ROWS",
    "S45D_HELPER_NAMES",
    "attach",
    "bind_helper_slots",
    "dump",
    "make_compile_interceptor",
    "require",
    "verify_helper_slots",
]
