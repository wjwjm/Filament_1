"""Private AST and callable helpers for the CPU-S4-5B targeted scopes.

The routines here are intentionally small and source-bound.  They insert only
scalar ``with`` scopes, three fixed Eq10 call shims, and a scalar zero marker.
The generated nodes carry private attributes so :func:`strip_generated_nodes`
can restore the exact input AST before a clone is accepted.  No array, field,
exception, frame, or result object is captured by a recorder callback.
"""

from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import marshal
from pathlib import Path
from types import CodeType, FunctionType, MethodType
from typing import Any, Callable, Mapping

from .timing import (
    NEW_A8_LABELS,
    NEW_B8_LABELS,
    TimingContext,
    TimingRecorder,
    context,
)


INSTRUMENT_SCHEMA = "filament.cpu_s4_5b.private_instrument.v1"
PROPAGATION_SCOPE_HELPER = "__cpu_s45b_scope__"
PROPAGATION_MARKER_HELPER = "__cpu_s45b_marker__"
EQ10_HELPER_NAMES = {
    "EQ10_FFT": "__cpu_s45b_eq10_fft__",
    "EQ10_IFFT": "__cpu_s45b_eq10_ifft__",
    "EQ10_SUM": "__cpu_s45b_eq10_sum__",
}
_GENERATED_CALL_NAMES = frozenset(
    {PROPAGATION_MARKER_HELPER, *EQ10_HELPER_NAMES.values()}
)
_OCCURRENCE_CLASSES = frozenset(
    ("WINDOW_ONCE", "FIRST_INTERVAL_SETUP", "PER_INTERVAL", "SAVE_BRANCH", "WINDOW_FINALIZE")
)


class InstrumentationBindingError(RuntimeError):
    """A source, AST, helper, or closure identity failed closed."""


def ast_identity(node: ast.AST) -> str:
    """Return a deterministic structure digest without source line metadata."""

    return hashlib.sha256(
        ast.dump(node, annotate_fields=True, include_attributes=False).encode("utf-8")
    ).hexdigest()


def code_identity(function: Callable[..., Any]) -> str:
    """Return a structural code-object identity suitable for a local receipt."""

    code = getattr(function, "__code__", None)
    if not isinstance(code, CodeType):
        return ""
    return hashlib.sha256(marshal.dumps(code)).hexdigest()


def _function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1 or not isinstance(matches[0], ast.FunctionDef):
        raise InstrumentationBindingError(f"FUNCTION_ANCHOR_COUNT:{name}:{len(matches)}")
    return matches[0]


def _copy_location(target: ast.AST, source: ast.AST) -> ast.AST:
    ast.copy_location(target, source)
    if hasattr(source, "end_lineno"):
        target.end_lineno = getattr(source, "end_lineno", None)
        target.end_col_offset = getattr(source, "end_col_offset", None)
    return target


def _call_path(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        prefix = _call_path(node.value)
        return None if prefix is None else (*prefix, node.attr)
    return None


def _scope_call(
    label: str,
    interval_expr: ast.expr,
    occurrence_class: str,
    *,
    source_site: str = "",
    helper_name: str = PROPAGATION_SCOPE_HELPER,
) -> ast.Call:
    if label not in NEW_A8_LABELS and label not in NEW_B8_LABELS:
        raise InstrumentationBindingError(f"B_SCOPE_LABEL_INVALID:{label}")
    if occurrence_class not in _OCCURRENCE_CLASSES:
        raise InstrumentationBindingError(f"B_OCCURRENCE_CLASS_INVALID:{occurrence_class}")
    return ast.Call(
        func=ast.Name(id=helper_name, ctx=ast.Load()),
        args=[
            ast.Constant(value=str(label)),
            interval_expr,
            ast.Constant(value=str(occurrence_class)),
            ast.Constant(value=str(source_site)),
        ],
        keywords=[],
    )


def _wrap_statements(
    statements: list[ast.stmt],
    label: str,
    interval_expr: ast.expr,
    occurrence_class: str,
) -> ast.With:
    if not statements:
        raise InstrumentationBindingError(f"B_EMPTY_SCOPE_SPAN:{label}")
    first = statements[0]
    last = statements[-1]
    start_line = int(getattr(first, "lineno", 0) or 0)
    end_line = int(getattr(last, "end_lineno", getattr(last, "lineno", 0)) or 0)
    source_site = f"{label}@{start_line}-{end_line}"
    wrapper = ast.With(
        items=[
            ast.withitem(
                context_expr=_scope_call(
                    label,
                    interval_expr,
                    occurrence_class,
                    source_site=source_site,
                ),
                optional_vars=None,
            )
        ],
        body=list(statements),
        type_comment=None,
    )
    wrapper._cpu_s45b_scope_wrapper = True
    wrapper._cpu_s45b_label = str(label)
    wrapper._cpu_s45b_occurrence_class = str(occurrence_class)
    wrapper._cpu_s45b_interval_expr = copy.deepcopy(interval_expr)
    return _copy_location(wrapper, statements[0])


def _assignment_names(statement: ast.stmt) -> tuple[str, ...]:
    if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return ()
    targets: list[ast.AST] = []
    if isinstance(statement, ast.Assign):
        targets.extend(statement.targets)
    elif isinstance(statement, ast.AnnAssign):
        targets.append(statement.target)
    else:
        targets.append(statement.target)
    return tuple(
        sorted(
            {
                node.id
                for target in targets
                for node in ast.walk(target)
                if isinstance(node, ast.Name)
            }
        )
    )


def _find_unique_top_level(
    statements: list[ast.stmt], predicate: Callable[[ast.stmt], bool], label: str
) -> int:
    matches = [index for index, statement in enumerate(statements) if predicate(statement)]
    if len(matches) != 1:
        raise InstrumentationBindingError(f"B_ANCHOR_COUNT:{label}:{len(matches)}")
    return matches[0]


def _contains_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(item, ast.Name) and item.id == name for item in ast.walk(node))


def _find_save_branch(statements: list[ast.stmt]) -> int:
    return _find_unique_top_level(
        statements,
        lambda statement: isinstance(statement, ast.If)
        and _contains_name(statement.test, "save_count")
        and _contains_name(statement.test, "interval_index"),
        "PROP_SAVED_DIAGNOSTICS_BRANCH",
    )


def _find_for_loop(function_node: ast.FunctionDef) -> tuple[list[ast.stmt], int, ast.For]:
    matches: list[tuple[list[ast.stmt], int, ast.For]] = []

    def visit(node: ast.AST) -> None:
        # A nested helper function is not the propagation loop even if it
        # happens to reuse the same local names.
        if node is not function_node and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return
        for _field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for index, statement in enumerate(value):
                    if isinstance(statement, ast.For):
                        target = statement.target
                        iterator = statement.iter
                        if (
                            isinstance(target, ast.Tuple)
                            and len(target.elts) == 2
                            and isinstance(target.elts[0], ast.Name)
                            and target.elts[0].id == "interval_index"
                            and isinstance(target.elts[1], ast.Name)
                            and target.elts[1].id == "interval"
                            and isinstance(iterator, ast.Call)
                            and isinstance(iterator.func, ast.Name)
                            and iterator.func.id == "enumerate"
                            and len(iterator.args) == 1
                            and isinstance(iterator.args[0], ast.Attribute)
                            and iterator.args[0].attr == "intervals"
                        ):
                            matches.append((value, index, statement))
                    if isinstance(statement, ast.AST):
                        visit(statement)
            elif isinstance(value, ast.AST):
                visit(value)

    visit(function_node)
    if len(matches) != 1:
        raise InstrumentationBindingError(f"PROP_INTERVAL_LOOP_COUNT:{len(matches)}")
    return matches[0]


def _replace_post_commit_scope(statements: list[ast.stmt]) -> dict[str, Any]:
    """Wrap the one real POST call without changing its argument evaluation."""

    matches: list[tuple[list[ast.stmt], int, ast.Expr]] = []

    def visit_list(values: list[ast.stmt]) -> None:
        for index, statement in enumerate(values):
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id == "post_commit_hook"
            ):
                matches.append((values, index, statement))
            for _field, value in ast.iter_fields(statement):
                if isinstance(value, list) and value and all(isinstance(item, ast.stmt) for item in value):
                    visit_list(value)
                elif isinstance(value, ast.AST):
                    for child_field, child_value in ast.iter_fields(value):
                        if isinstance(child_value, list) and child_value and all(
                            isinstance(item, ast.stmt) for item in child_value
                        ):
                            visit_list(child_value)

    visit_list(statements)
    if len(matches) != 1:
        raise InstrumentationBindingError(f"PROP_POST_COMMIT_CALL_COUNT:{len(matches)}")
    values, index, statement = matches[0]
    values[index] = _wrap_statements(
        [statement],
        "INTERVAL_POST_COMMIT",
        ast.Name(id="interval_index", ctx=ast.Load()),
        "PER_INTERVAL",
    )
    return {
        "label": "INTERVAL_POST_COMMIT",
        "line": int(getattr(statement, "lineno", 0)),
        "column": int(getattr(statement, "col_offset", 0)),
        "anchor": "post_commit_hook",
    }


def _wrap_save_branch(branch: ast.If) -> dict[str, Any]:
    if not branch.body:
        raise InstrumentationBindingError("PROP_SAVED_DIAGNOSTICS_BRANCH_EMPTY")
    branch.body = [
        _wrap_statements(
            branch.body,
            "INTERVAL_SAVED_DIAGNOSTICS",
            ast.Name(id="interval_index", ctx=ast.Load()),
            "SAVE_BRANCH",
        )
    ]
    return {
        "label": "INTERVAL_SAVED_DIAGNOSTICS",
        "line": int(getattr(branch, "lineno", 0)),
        "column": int(getattr(branch, "col_offset", 0)),
        "occurrence_class": "SAVE_BRANCH",
    }


def instrument_propagation_ast(function_node: ast.FunctionDef) -> dict[str, Any]:
    """Insert the fixed A8 scopes into one actual propagate clone.

    Anchors are selected by statement structure and names, not by blindly
    slicing source line numbers.  The returned receipt records all selected
    ranges.  Removing generated nodes must reproduce the input AST exactly.
    """

    if not isinstance(function_node, ast.FunctionDef):
        raise InstrumentationBindingError("PROP_FUNCTION_NODE_REQUIRED")
    baseline = copy.deepcopy(function_node)
    loop_statements, loop_index, loop = _find_for_loop(function_node)
    loop_body = list(loop.body)
    if any(isinstance(node, (ast.While, ast.Continue, ast.Break)) for node in ast.walk(loop)):
        raise InstrumentationBindingError("PROP_UNSUPPORTED_CONTROL_TRANSFER")

    linear_candidates = [
        index
        for index, statement in enumerate(loop_body)
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "use_uppe"
    ]
    # The production split has two fixed linear half-steps.  A8 owns the
    # first source anchor; the full INTERVAL_BODY shell already contains both
    # halves, so no second B label or double-counted child is introduced.
    if len(linear_candidates) not in (1, 2):
        raise InstrumentationBindingError(
            f"B_ANCHOR_COUNT:PROP_LINEAR_BRANCH:{len(linear_candidates)}"
        )
    linear_index = linear_candidates[0]
    energy_start = _find_unique_top_level(
        loop_body,
        lambda statement: "operator_energy_after_nonraman" in _assignment_names(statement),
        "PROP_ENERGY_START",
    )
    energy_post = _find_unique_top_level(
        loop_body,
        lambda statement: "operator_energy_after_raman_post" in _assignment_names(statement),
        "PROP_ENERGY_POST",
    )
    energy_final = _find_unique_top_level(
        loop_body,
        lambda statement: "operator_energy_after_linear_half2" in _assignment_names(statement),
        "PROP_ENERGY_FINAL",
    )
    save_index = _find_save_branch(loop_body)
    if not (0 < linear_index < energy_start <= energy_post < energy_final < save_index):
        raise InstrumentationBindingError(
            "PROP_ANCHOR_ORDER_INVALID"
        )

    # Insert the innermost child scopes first.  Their enclosing range indices
    # are then still the original statement boundaries.
    post_receipt = _replace_post_commit_scope(loop_body)
    save_receipt = _wrap_save_branch(loop_body[save_index])

    # The POST child was inserted inside its original enclosing statement.  A
    # broad deposition range includes that child and remains an honest parent.
    deposition = _wrap_statements(
        loop_body[energy_post + 1 : energy_final],
        "INTERVAL_DEPOSITION_AND_QACC",
        ast.Name(id="interval_index", ctx=ast.Load()),
        "PER_INTERVAL",
    )
    loop_body[energy_post + 1 : energy_final] = [deposition]
    # The save branch was after energy_final and therefore its index moved by
    # one only if the deposition slice was nonempty; locate it structurally.
    save_index_after = _find_save_branch(loop_body)

    energy = _wrap_statements(
        loop_body[energy_start : energy_post + 1],
        "INTERVAL_ENERGY_BOOKKEEPING",
        ast.Name(id="interval_index", ctx=ast.Load()),
        "PER_INTERVAL",
    )
    loop_body[energy_start : energy_post + 1] = [energy]
    # The two other-array ranges are deliberately separate around the existing
    # linear branch.  Existing helper timers remain nested inside them.
    energy_start_after = _find_unique_top_level(
        loop_body,
        lambda statement: isinstance(statement, ast.With)
        and getattr(statement, "_cpu_s45b_label", "") == "INTERVAL_ENERGY_BOOKKEEPING",
        "PROP_ENERGY_WRAPPED",
    )
    linear_after_candidates = [
        index
        for index, statement in enumerate(loop_body)
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "use_uppe"
    ]
    if len(linear_after_candidates) not in (1, 2):
        raise InstrumentationBindingError(
            f"B_ANCHOR_COUNT:PROP_LINEAR_BRANCH_AFTER:{len(linear_after_candidates)}"
        )
    linear_index_after = linear_after_candidates[0]
    if linear_index_after + 1 < energy_start_after:
        loop_body[linear_index_after + 1 : energy_start_after] = [
            _wrap_statements(
                loop_body[linear_index_after + 1 : energy_start_after],
                "INTERVAL_PREP_AND_OTHER_ARRAYS",
                ast.Name(id="interval_index", ctx=ast.Load()),
                "PER_INTERVAL",
            )
        ]
    if linear_index_after > 0:
        loop_body[0:linear_index_after] = [
            _wrap_statements(
                loop_body[0:linear_index_after],
                "INTERVAL_PREP_AND_OTHER_ARRAYS",
                ast.Name(id="interval_index", ctx=ast.Load()),
                "PER_INTERVAL",
            )
        ]

    # Full body shell is the outermost scope and encloses the original loop
    # bottom, including progress code and exception cleanup.
    loop.body = [
        _wrap_statements(
            loop_body,
            "INTERVAL_BODY",
            ast.Name(id="interval_index", ctx=ast.Load()),
            "PER_INTERVAL",
        )
    ]

    # Entry and finalization are fixed windows around the actual For node.
    # Preserve a leading docstring outside the scope for normal function
    # metadata semantics.
    entry_start = 1 if (
        loop_statements is function_node.body
        and function_node.body
        and isinstance(function_node.body[0], ast.Expr)
        and isinstance(function_node.body[0].value, ast.Constant)
        and isinstance(function_node.body[0].value.value, str)
    ) else 0
    if loop_index - entry_start <= 0:
        raise InstrumentationBindingError("PROP_ENTRY_INIT_SPAN_EMPTY")
    entry_span = loop_statements[entry_start:loop_index]
    loop_statements[entry_start:loop_index] = [
        _wrap_statements(
            entry_span,
            "PROP_ENTRY_INIT",
            ast.Constant(value=0),
            "WINDOW_ONCE",
        )
    ]
    loop_statements_after, loop_index_after, _ = _find_for_loop(function_node)
    tail = loop_statements_after[loop_index_after + 1 :]
    if not tail:
        raise InstrumentationBindingError("PROP_FINALIZE_EXIT_SPAN_EMPTY")
    loop_statements_after[loop_index_after + 1 :] = [
        _wrap_statements(
            tail,
            "PROP_FINALIZE_EXIT",
            ast.Constant(value=0),
            "WINDOW_FINALIZE",
        )
    ]
    ast.fix_missing_locations(function_node)

    stripped = strip_generated_nodes(function_node)
    if ast.dump(stripped, include_attributes=False) != ast.dump(
        baseline, include_attributes=False
    ):
        raise InstrumentationBindingError("PROP_B_STRIPPED_AST_MISMATCH")
    scope_sites = [
        {
            "label": getattr(node, "_cpu_s45b_label", ""),
            "occurrence_class": getattr(node, "_cpu_s45b_occurrence_class", ""),
            "line": int(getattr(node, "lineno", 0)),
            "end_line": int(getattr(node, "end_lineno", getattr(node, "lineno", 0))),
        }
        for node in ast.walk(function_node)
        if isinstance(node, ast.With) and getattr(node, "_cpu_s45b_scope_wrapper", False)
    ]
    return {
        "schema": INSTRUMENT_SCHEMA + ".propagation",
        "source_ast_sha256": ast_identity(baseline),
        "stripped_ast_sha256": ast_identity(stripped),
        "scope_sites": sorted(scope_sites, key=lambda row: (row["line"], row["label"])),
        "post_commit": post_receipt,
        "saved_diagnostics": save_receipt,
        "anchor_counts": {
            "interval_for": 1,
            "linear_branch": 1,
            "linear_branch_candidates": int(len(linear_candidates)),
            "energy_start": 1,
            "energy_post": 1,
            "energy_final": 1,
            "save_branch": 1,
            "post_commit": 1,
        },
        "control_transfers_added": 0,
        "array_values_retained": False,
    }


def _is_assign_name(statement: ast.stmt, name: str) -> bool:
    return name in _assignment_names(statement)


def _find_return_energy(function_node: ast.FunctionDef) -> ast.If:
    matches = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "return_energy"
    ]
    if len(matches) != 1:
        raise InstrumentationBindingError(f"EQ10_RETURN_ENERGY_BRANCH_COUNT:{len(matches)}")
    return matches[0]


def _replace_fixed_call(
    node: ast.AST,
    *,
    path: tuple[str, ...],
    helper_name: str,
    label: str,
) -> tuple[ast.AST, int]:
    count = 0

    class Rewriter(ast.NodeTransformer):
        def visit_Call(self, call: ast.Call) -> ast.AST:
            nonlocal count
            if _call_path(call.func) == path:
                if any(keyword.arg is None for keyword in call.keywords):
                    raise InstrumentationBindingError(f"EQ10_CALL_STAR_KWARGS:{label}")
                args_tuple = ast.Tuple(elts=[copy.deepcopy(arg) for arg in call.args], ctx=ast.Load())
                keys = [ast.Constant(value=keyword.arg) for keyword in call.keywords]
                values = [copy.deepcopy(keyword.value) for keyword in call.keywords]
                replacement = ast.Call(
                    func=ast.Name(id=helper_name, ctx=ast.Load()),
                    args=[
                        copy.deepcopy(call.func),
                        args_tuple,
                        ast.Dict(keys=keys, values=values),
                        ast.Constant(
                            value=f"{label}@{int(getattr(call, 'lineno', 0) or 0)}"
                        ),
                    ],
                    keywords=[],
                )
                replacement._cpu_s45b_call_wrapper = True
                replacement._cpu_s45b_call_label = str(label)
                replacement._cpu_s45b_original_call = copy.deepcopy(call)
                count += 1
                return _copy_location(replacement, call)
            return self.generic_visit(call)

    result = Rewriter().visit(node)
    return result, count


def instrument_eq10_stage(function_node: ast.FunctionDef) -> dict[str, Any]:
    """Insert the eight B8 scopes in the unique ``return_energy`` branch."""

    if not isinstance(function_node, ast.FunctionDef):
        raise InstrumentationBindingError("EQ10_FUNCTION_NODE_REQUIRED")
    baseline = copy.deepcopy(function_node)
    branch = _find_return_energy(function_node)
    scale_body = branch.body
    scale_index = _find_unique_top_level(
        scale_body,
        lambda statement: _is_assign_name(statement, "intensity_scale"),
        "EQ10_SCALE_MAX",
    )
    positive_branch_matches = [
        statement
        for statement in scale_body
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Compare)
        and isinstance(statement.test.left, ast.Name)
        and statement.test.left.id == "intensity_scale"
    ]
    if len(positive_branch_matches) != 1:
        raise InstrumentationBindingError(
            f"EQ10_INTENSITY_POSITIVE_BRANCH_COUNT:{len(positive_branch_matches)}"
        )
    positive_branch = positive_branch_matches[0]
    positive = positive_branch.body
    derivative_index = _find_unique_top_level(
        positive,
        lambda statement: _is_assign_name(statement, "derivative_increment_norm"),
        "EQ10_DERIVATIVE_EXPR",
    )
    response_index = _find_unique_top_level(
        positive,
        lambda statement: _is_assign_name(statement, "response_norm"),
        "EQ10_RESPONSE_NORM",
    )
    scale_energy_index = _find_unique_top_level(
        positive,
        lambda statement: _is_assign_name(statement, "energy_scale"),
        "EQ10_ENERGY_SCALE",
    )
    product_index = _find_unique_top_level(
        positive,
        lambda statement: _is_assign_name(statement, "u_signed"),
        "EQ10_PRODUCT_REDUCE_SCALE",
    )
    materialize_indices = [
        index
        for index, statement in enumerate(scale_body)
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.slice, ast.Constant)
            and target.slice.value in ("u_R_signed", "q_R_positive")
            for target in statement.targets
        )
    ]
    if len(materialize_indices) != 2 or materialize_indices[1] != materialize_indices[0] + 1:
        raise InstrumentationBindingError("EQ10_MATERIALIZE_ANCHOR_COUNT_OR_ORDER")
    if not (derivative_index < response_index == scale_energy_index - 1 < product_index):
        raise InstrumentationBindingError("EQ10_ANCHOR_ORDER_INVALID")

    # Fixed call sites are rewritten before enclosing statement scopes.
    derivative_node, fft_count = _replace_fixed_call(
        positive[derivative_index],
        path=("xp", "fft", "fft"),
        helper_name=EQ10_HELPER_NAMES["EQ10_FFT"],
        label="EQ10_FFT",
    )
    derivative_node, ifft_count = _replace_fixed_call(
        derivative_node,
        path=("xp", "fft", "ifft"),
        helper_name=EQ10_HELPER_NAMES["EQ10_IFFT"],
        label="EQ10_IFFT",
    )
    if fft_count != 1 or ifft_count != 1:
        raise InstrumentationBindingError("EQ10_FFT_IFFT_ANCHOR_COUNT")
    positive[derivative_index] = derivative_node
    product_node, sum_count = _replace_fixed_call(
        positive[product_index],
        path=("xp", "sum"),
        helper_name=EQ10_HELPER_NAMES["EQ10_SUM"],
        label="EQ10_SUM",
    )
    if sum_count != 1:
        raise InstrumentationBindingError("EQ10_SUM_ANCHOR_COUNT")
    positive[product_index] = product_node

    # The response/scale pair must stay contiguous and in the original order.
    scale_body[scale_index] = _wrap_statements(
        [scale_body[scale_index]],
        "EQ10_SCALE_MAX",
        ast.Constant(value=None),
        "PER_INTERVAL",
    )
    positive[derivative_index] = _wrap_statements(
        [positive[derivative_index]],
        "EQ10_DERIVATIVE_EXPR",
        ast.Constant(value=None),
        "PER_INTERVAL",
    )
    response_index = _find_unique_top_level(
        positive,
        lambda statement: _is_assign_name(statement, "response_norm"),
        "EQ10_RESPONSE_NORM_AFTER",
    )
    positive[response_index : response_index + 2] = [
        _wrap_statements(
            positive[response_index : response_index + 2],
            "EQ10_NORMALIZE_SCALE",
            ast.Constant(value=None),
            "PER_INTERVAL",
        )
    ]
    product_index = _find_unique_top_level(
        positive,
        lambda statement: _is_assign_name(statement, "u_signed"),
        "EQ10_PRODUCT_REDUCE_SCALE_AFTER",
    )
    positive[product_index] = _wrap_statements(
        [positive[product_index]],
        "EQ10_PRODUCT_REDUCE_SCALE",
        ast.Constant(value=None),
        "PER_INTERVAL",
    )
    materialize_indices = [
        index
        for index, statement in enumerate(scale_body)
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.slice, ast.Constant)
            and target.slice.value in ("u_R_signed", "q_R_positive")
            for target in statement.targets
        )
    ]
    scale_body[materialize_indices[0] : materialize_indices[1] + 1] = [
        _wrap_statements(
            scale_body[materialize_indices[0] : materialize_indices[1] + 1],
            "EQ10_MATERIALIZE",
            ast.Constant(value=None),
            "PER_INTERVAL",
        )
    ]

    # The zero branch gets only a scalar marker.  It is not a new scope label.
    zero_branch = positive_branch.orelse
    zero_assignment = [
        statement
        for statement in zero_branch
        if _is_assign_name(statement, "u_signed")
    ]
    if len(zero_assignment) != 1:
        raise InstrumentationBindingError("EQ10_ZERO_BRANCH_ANCHOR_COUNT")
    marker = ast.Expr(
        value=ast.Call(
            func=ast.Name(id=PROPAGATION_MARKER_HELPER, ctx=ast.Load()),
            args=[
                ast.Constant(value="ZERO_INTENSITY"),
                ast.Constant(
                    value=(
                        "ZERO_INTENSITY@"
                        f"{int(getattr(zero_assignment[0], 'lineno', 0) or 0)}"
                    )
                ),
            ],
            keywords=[],
        )
    )
    marker._cpu_s45b_marker = True
    _copy_location(marker, zero_assignment[0])
    zero_branch.insert(zero_branch.index(zero_assignment[0]), marker)
    ast.fix_missing_locations(function_node)

    stripped = strip_generated_nodes(function_node)
    if ast.dump(stripped, include_attributes=False) != ast.dump(
        baseline, include_attributes=False
    ):
        raise InstrumentationBindingError("EQ10_B_STRIPPED_AST_MISMATCH")
    return {
        "schema": INSTRUMENT_SCHEMA + ".eq10",
        "source_ast_sha256": ast_identity(baseline),
        "stripped_ast_sha256": ast_identity(stripped),
        "branch_count": 1,
        "fft_calls": 1,
        "ifft_calls": 1,
        "sum_calls": 1,
        "zero_markers": 1,
        "scope_labels": list(NEW_B8_LABELS),
        "array_values_retained": False,
    }


def strip_generated_nodes(node: ast.AST) -> ast.AST:
    """Remove only generated B wrappers and reconstruct their original calls."""

    result = copy.deepcopy(node)

    class Stripper(ast.NodeTransformer):
        def visit_With(self, current: ast.With) -> Any:
            if getattr(current, "_cpu_s45b_scope_wrapper", False):
                body: list[ast.stmt] = []
                for statement in current.body:
                    visited = self.visit(statement)
                    if isinstance(visited, list):
                        body.extend(visited)
                    elif visited is not None:
                        body.append(visited)
                return body
            return self.generic_visit(current)

        def visit_Expr(self, current: ast.Expr) -> Any:
            if getattr(current, "_cpu_s45b_marker", False):
                return None
            if (
                isinstance(current.value, ast.Call)
                and isinstance(current.value.func, ast.Name)
                and current.value.func.id == PROPAGATION_MARKER_HELPER
            ):
                return None
            return self.generic_visit(current)

        def visit_Call(self, current: ast.Call) -> Any:
            if getattr(current, "_cpu_s45b_call_wrapper", False):
                original = getattr(current, "_cpu_s45b_original_call", None)
                if not isinstance(original, ast.Call):
                    raise InstrumentationBindingError("B_CALL_ORIGINAL_AST_MISSING")
                # An IFFT wrapper can contain the already generated FFT
                # wrapper.  Revisit the restored call so nested B nodes are
                # removed in the same pass and the original expression is
                # recovered exactly.
                return self.visit(copy.deepcopy(original))
            if (
                isinstance(current.func, ast.Name)
                and current.func.id in _GENERATED_CALL_NAMES
            ):
                original = getattr(current, "_cpu_s45b_original_call", None)
                if isinstance(original, ast.Call):
                    return self.visit(copy.deepcopy(original))
                raise InstrumentationBindingError("B_GENERATED_CALL_UNMARKED")
            return self.generic_visit(current)

        def generic_visit(self, current: ast.AST) -> Any:
            for field, value in list(ast.iter_fields(current)):
                if isinstance(value, list):
                    restored: list[Any] = []
                    for item in value:
                        visited = self.visit(item) if isinstance(item, ast.AST) else item
                        if isinstance(visited, list):
                            restored.extend(visited)
                        elif visited is not None or item is None:
                            restored.append(visited)
                    setattr(current, field, restored)
                elif isinstance(value, ast.AST):
                    visited = self.visit(value)
                    if visited is None:
                        raise InstrumentationBindingError("B_GENERATED_NODE_IN_EXPRESSION")
                    setattr(current, field, visited)
            return current

    return Stripper().visit(result)


def make_scope_helper(recorder: TimingRecorder) -> Callable[..., Any]:
    """Bind the fixed AST scope helper to one B-owned recorder."""

    if type(recorder) is not TimingRecorder:
        raise InstrumentationBindingError("B_RECORDER_EXACT_TYPE_REQUIRED")

    def scope(
        label: str,
        interval_index: int | None = None,
        occurrence_class: str = "PER_INTERVAL",
        source_site: str = "",
    ):
        current = recorder.current_context()
        if interval_index is None:
            metadata = recorder.bridged_context(
                require_companion=True,
                source_site=str(source_site),
            )
        else:
            metadata = context(
                int(interval_index),
                current.phase,
                current.stage,
                ordinal=int(interval_index),
                attempt=1,
                occurrence_class=str(occurrence_class),
                source_site=str(source_site),
            )
        return recorder.scope(label, context=metadata)

    scope.__name__ = PROPAGATION_SCOPE_HELPER
    scope.__cpu_s45b_kind__ = "FIXED_SCOPE_HELPER"
    scope.__cpu_s45b_recorder__ = recorder
    return scope


def make_marker_helper(recorder: TimingRecorder) -> Callable[..., Any]:
    """Bind the one scalar zero-intensity marker helper."""

    if type(recorder) is not TimingRecorder:
        raise InstrumentationBindingError("B_RECORDER_EXACT_TYPE_REQUIRED")

    def marker(kind: str, source_site: str = "") -> None:
        current = recorder.bridged_context(
            require_companion=True,
            source_site=str(source_site),
        )
        recorder.marker(
            str(kind),
            interval_index=int(current.interval),
            phase=str(current.phase),
            stage=str(current.stage),
            source_site=str(current.source_site),
        )

    marker.__name__ = PROPAGATION_MARKER_HELPER
    marker.__cpu_s45b_kind__ = "FIXED_SCALAR_MARKER"
    marker.__cpu_s45b_recorder__ = recorder
    return marker


def _same_fixed_backend_callable(actual: Any, expected: Any) -> bool:
    """Match only a fixed callable or its freshly materialized proxy closure.

    The accepted SciPy dispatch creates a new call closure per attribute access.
    Code/globals and every closed capture must still be the registered objects;
    this does not accept another implementation by name or shape.
    """
    if actual is expected:
        return True
    if type(actual) is MethodType and type(expected) is MethodType:
        return actual.__func__ is expected.__func__ and actual.__self__ is expected.__self__
    if type(actual) is not FunctionType or type(expected) is not FunctionType:
        return False
    if actual.__code__ is not expected.__code__ or actual.__globals__ is not expected.__globals__:
        return False
    if actual.__defaults__ is not expected.__defaults__ or actual.__kwdefaults__ is not expected.__kwdefaults__:
        return False
    left,right=actual.__closure__ or (),expected.__closure__ or ()
    return len(left)==len(right) and all(a.cell_contents is b.cell_contents for a,b in zip(left,right))


def make_eq10_call_helpers(
    recorder: TimingRecorder,
    registry: Mapping[str, Callable[..., Any]],
) -> dict[str, Callable[..., Any]]:
    """Create exactly three transparent Eq10 call shims.

    ``registry`` is private and fixed to the actual loaded ``xp.fft.fft``,
    ``xp.fft.ifft`` and ``xp.sum`` callables.  The helpers reject any other
    callable identity; they call the original once and retain no argument or
    result after returning.
    """

    if type(recorder) is not TimingRecorder:
        raise InstrumentationBindingError("B_RECORDER_EXACT_TYPE_REQUIRED")
    required = {"EQ10_FFT", "EQ10_IFFT", "EQ10_SUM"}
    if set(registry) != required or any(not callable(value) for value in registry.values()):
        raise InstrumentationBindingError("EQ10_FIXED_CALL_REGISTRY_INVALID")

    def make(label: str, helper_name: str) -> Callable[..., Any]:
        expected = registry[label]
        fixed_matcher = _same_fixed_backend_callable

        def call(
            function: Callable[..., Any],
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            source_site: str = "",
        ) -> Any:
            if not fixed_matcher(function, expected):
                raise InstrumentationBindingError(f"EQ10_CALLABLE_BINDING_CHANGED:{label}")
            if not isinstance(args, tuple) or not isinstance(kwargs, dict):
                raise InstrumentationBindingError(f"EQ10_CALL_ARGUMENT_CONTAINER_INVALID:{label}")
            with recorder.scope(
                label,
                context=recorder.bridged_context(
                    require_companion=True,
                    source_site=str(source_site),
                ),
            ):
                return function(*args, **kwargs)

        call.__name__ = helper_name
        call.__cpu_s45b_kind__ = "FIXED_EQ10_CALL_HELPER"
        call.__cpu_s45b_label__ = label
        call.__cpu_s45b_original__ = expected
        call.__cpu_s45b_recorder__ = recorder
        return call

    return {
        EQ10_HELPER_NAMES["EQ10_FFT"]: make("EQ10_FFT", EQ10_HELPER_NAMES["EQ10_FFT"]),
        EQ10_HELPER_NAMES["EQ10_IFFT"]: make("EQ10_IFFT", EQ10_HELPER_NAMES["EQ10_IFFT"]),
        EQ10_HELPER_NAMES["EQ10_SUM"]: make("EQ10_SUM", EQ10_HELPER_NAMES["EQ10_SUM"]),
    }


def seal_callable(function: Callable[..., Any]) -> dict[str, Any]:
    """Record fixed callable identity without retaining argument values."""

    code = getattr(function, "__code__", None)
    globals_dict = getattr(function, "__globals__", None)
    closure = getattr(function, "__closure__", None)
    if not callable(function) or not isinstance(code, CodeType) or not isinstance(globals_dict, dict):
        raise InstrumentationBindingError("FIXED_CALLABLE_INVALID")
    return {
        "function": function,
        "code": code,
        "code_identity": code_identity(function),
        "globals": globals_dict,
        "closure": closure,
        "closure_cells": tuple(cell.cell_contents for cell in (closure or ())),
    }


def verify_callable_seal(seal: Mapping[str, Any]) -> None:
    function = seal.get("function")
    if not callable(function):
        raise InstrumentationBindingError("FIXED_CALLABLE_SEAL_FUNCTION_MISSING")
    if getattr(function, "__code__", None) is not seal.get("code"):
        raise InstrumentationBindingError("FIXED_CALLABLE_CODE_CHANGED")
    if getattr(function, "__globals__", None) is not seal.get("globals"):
        raise InstrumentationBindingError("FIXED_CALLABLE_GLOBALS_CHANGED")
    if getattr(function, "__closure__", None) is not seal.get("closure"):
        raise InstrumentationBindingError("FIXED_CALLABLE_CLOSURE_CHANGED")
    closure = getattr(function, "__closure__", None) or ()
    expected = seal.get("closure_cells", ())
    if len(closure) != len(expected) or any(cell.cell_contents is not value for cell, value in zip(closure, expected)):
        raise InstrumentationBindingError("FIXED_CALLABLE_CLOSURE_CELL_CHANGED")


def _python_callable_component(function: Any) -> Any:
    """Return the fixed Python implementation behind a backend callable."""

    bound_func = getattr(function, "__func__", None)
    if bound_func is not None:
        bound_code = getattr(bound_func, "__code__", None)
        bound_globals = getattr(bound_func, "__globals__", None)
        if isinstance(bound_code, CodeType) and isinstance(bound_globals, dict):
            return bound_func
    code = getattr(function, "__code__", None)
    globals_dict = getattr(function, "__globals__", None)
    if isinstance(code, CodeType) and isinstance(globals_dict, dict):
        return function
    wrapped = getattr(function, "__wrapped__", None)
    if wrapped is not None:
        wrapped_code = getattr(wrapped, "__code__", None)
        wrapped_globals = getattr(wrapped, "__globals__", None)
        if isinstance(wrapped_code, CodeType) and isinstance(wrapped_globals, dict):
            return wrapped
    return None


def seal_fixed_backend_callable(function: Callable[..., Any], *, label: str = "") -> dict[str, Any]:
    """Seal one actual FFT/IFFT/sum backend callable.

    The registry caller supplies the exact loaded backend objects.  This
    routine never accepts a user callback as a substitute: it records object
    identity, bound-method components, and the Python implementation (or
    ``__wrapped__`` implementation) used by a C dispatcher.
    """

    if not callable(function):
        raise InstrumentationBindingError("FIXED_BACKEND_CALLABLE_INVALID")
    implementation = _python_callable_component(function)
    if implementation is None:
        raise InstrumentationBindingError("FIXED_BACKEND_PYTHON_IMPLEMENTATION_MISSING")
    implementation_seal = seal_callable(implementation)
    bound_func = getattr(function, "__func__", None)
    bound_self = getattr(function, "__self__", None)
    if bound_func is not None and bound_self is not None:
        if not callable(bound_func):
            raise InstrumentationBindingError("FIXED_BACKEND_BOUND_METHOD_INVALID")
    return {
        "schema": INSTRUMENT_SCHEMA + ".fixed_backend",
        "label": str(label),
        "function": function,
        "function_id": int(id(function)),
        "implementation": implementation,
        "implementation_seal": implementation_seal,
        "wrapped": getattr(function, "__wrapped__", None),
        "bound_func": bound_func,
        "bound_func_id": None if bound_func is None else int(id(bound_func)),
        "bound_self": bound_self,
        "bound_self_id": None if bound_self is None else int(id(bound_self)),
    }


def verify_fixed_backend_callable_seal(seal: Mapping[str, Any]) -> None:
    function = seal.get("function")
    if not callable(function) or int(id(function)) != int(seal.get("function_id", -1)):
        raise InstrumentationBindingError("FIXED_BACKEND_FUNCTION_CHANGED")
    implementation = seal.get("implementation")
    if _python_callable_component(function) is not implementation:
        raise InstrumentationBindingError("FIXED_BACKEND_IMPLEMENTATION_CHANGED")
    if seal.get("wrapped") is not getattr(function, "__wrapped__", None):
        raise InstrumentationBindingError("FIXED_BACKEND_WRAPPED_CHANGED")
    bound_func = getattr(function, "__func__", None)
    bound_self = getattr(function, "__self__", None)
    if bound_func is not seal.get("bound_func") or bound_self is not seal.get("bound_self"):
        raise InstrumentationBindingError("FIXED_BACKEND_BOUND_COMPONENT_CHANGED")
    verify_callable_seal(seal["implementation_seal"])


def seal_fixed_backend_registry(
    registry: Mapping[str, Callable[..., Any]],
) -> dict[str, dict[str, Any]]:
    required = {"EQ10_FFT", "EQ10_IFFT", "EQ10_SUM"}
    if set(registry) != required:
        raise InstrumentationBindingError("EQ10_FIXED_CALL_REGISTRY_INVALID")
    return {
        str(label): seal_fixed_backend_callable(function, label=str(label))
        for label, function in registry.items()
    }


def verify_fixed_backend_registry(
    registry: Mapping[str, Callable[..., Any]],
    seals: Mapping[str, Mapping[str, Any]],
) -> None:
    required = {"EQ10_FFT", "EQ10_IFFT", "EQ10_SUM"}
    if set(registry) != required or set(seals) != required:
        raise InstrumentationBindingError("EQ10_FIXED_CALL_REGISTRY_KEYS_CHANGED")
    for label in sorted(required):
        seal = seals[label]
        if registry[label] is not seal.get("function"):
            raise InstrumentationBindingError(f"EQ10_FIXED_CALLABLE_CHANGED:{label}")
        verify_fixed_backend_callable_seal(seal)


def copy_function(function: Callable[..., Any], globals_dict: dict[str, Any], name: str | None = None) -> Callable[..., Any]:
    """Clone one function with a private globals dictionary."""

    code = getattr(function, "__code__", None)
    if not isinstance(code, CodeType):
        raise InstrumentationBindingError("FUNCTION_CODE_MISSING")
    clone = FunctionType(
        code,
        globals_dict,
        str(name or getattr(function, "__name__", "function")),
        getattr(function, "__defaults__", None),
        getattr(function, "__closure__", None),
    )
    clone.__kwdefaults__ = getattr(function, "__kwdefaults__", None)
    clone.__annotations__ = dict(getattr(function, "__annotations__", {}) or {})
    clone.__module__ = getattr(function, "__module__", None)
    clone.__qualname__ = getattr(function, "__qualname__", clone.__name__)
    clone.__doc__ = getattr(function, "__doc__", None)
    return clone


def compile_function_node(
    function_node: ast.FunctionDef,
    *,
    namespace: dict[str, Any],
    source_path: str | Path,
) -> Callable[..., Any]:
    """Compile one transformed function in a caller-owned private namespace."""

    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)
    code = builtins.compile(module, str(source_path), "exec", dont_inherit=True, optimize=0)
    local_namespace: dict[str, Any] = {}
    exec(code, namespace, local_namespace)
    function = local_namespace.get(function_node.name)
    if not callable(function):
        raise InstrumentationBindingError("COMPILED_FUNCTION_MISSING")
    return function


def binding_receipt(
    *,
    original: Callable[..., Any],
    instrumented: Callable[..., Any],
    stripped_ast_sha256: str,
    scope_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a scalar source/actual binding receipt for parent integration."""

    return {
        "schema": INSTRUMENT_SCHEMA + ".binding",
        "original_code_identity": code_identity(original),
        "instrumented_code_identity": code_identity(instrumented),
        "original_code_object_id": int(id(getattr(original, "__code__", None))),
        "instrumented_code_object_id": int(id(getattr(instrumented, "__code__", None))),
        "original_globals_id": int(id(getattr(original, "__globals__", None))),
        "instrumented_globals_id": int(id(getattr(instrumented, "__globals__", None))),
        "stripped_ast_sha256": str(stripped_ast_sha256),
        "scope_receipt": copy.deepcopy(dict(scope_receipt)),
        "array_values_retained": False,
    }


__all__ = [
    "EQ10_HELPER_NAMES",
    "INSTRUMENT_SCHEMA",
    "InstrumentationBindingError",
    "PROPAGATION_MARKER_HELPER",
    "PROPAGATION_SCOPE_HELPER",
    "ast_identity",
    "binding_receipt",
    "code_identity",
    "compile_function_node",
    "copy_function",
    "instrument_eq10_stage",
    "instrument_propagation_ast",
    "make_eq10_call_helpers",
    "make_marker_helper",
    "make_scope_helper",
    "seal_callable",
    "seal_fixed_backend_callable",
    "seal_fixed_backend_registry",
    "strip_generated_nodes",
    "verify_callable_seal",
    "verify_fixed_backend_callable_seal",
    "verify_fixed_backend_registry",
]
