"""Source-bound residual-cost AST instrumentation for S4-6I.

The transform is deliberately a final, reversible layer.  It receives the
tree after the existing G2/C and S4-5B transforms, adds only scalar ``with``
scopes and private helper-name substitutions, and strips back to that exact
post-B tree.  The existing C adapter then performs its normal inverse proof
back to the frozen source tree.
"""

from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from typing import Any, Callable

from .registry import AUXILIARY_UNITS, COST_UNITS, PARENT_LABELS, SCOPE_HELPER, SCOPE_LABEL


INSTRUMENT_SCHEMA = "filament.cpu.s4_6i.residual_parallel.instrument.v1"
_WRAPPER_MARK = "_cpu_s46i_cost_scope_wrapper"
_ALIAS_MARK = "_cpu_s46i_cost_alias_call"
_ALIAS_ORIGINAL = "_cpu_s46i_cost_original_func"


class ResidualInstrumentationError(RuntimeError):
    """A source anchor or inverse AST proof failed closed."""


def ast_identity(node: ast.AST) -> str:
    return hashlib.sha256(
        ast.dump(node, annotate_fields=True, include_attributes=False).encode("utf-8")
    ).hexdigest()


def _copy_location(target: ast.AST, source: ast.AST) -> ast.AST:
    ast.copy_location(target, source)
    if hasattr(source, "end_lineno"):
        target.end_lineno = getattr(source, "end_lineno", None)
        target.end_col_offset = getattr(source, "end_col_offset", None)
    return target


def _scope_call(source_site: str, interval_name: str | None) -> ast.Call:
    interval: ast.expr = (
        ast.Name(id=interval_name, ctx=ast.Load())
        if interval_name is not None
        else ast.Constant(value=None)
    )
    return ast.Call(
        func=ast.Name(id=SCOPE_HELPER, ctx=ast.Load()),
        args=[
            ast.Constant(value=SCOPE_LABEL),
            interval,
            ast.Constant(value="PER_INTERVAL"),
            ast.Constant(value=str(source_site)),
        ],
        keywords=[],
    )


def _wrap(statements: list[ast.stmt], *, source_site: str, interval_name: str | None) -> ast.With:
    if not statements:
        raise ResidualInstrumentationError("S46I_EMPTY_SCOPE_SPAN")
    wrapper = ast.With(
        items=[ast.withitem(context_expr=_scope_call(source_site, interval_name), optional_vars=None)],
        body=list(statements),
        type_comment=None,
    )
    setattr(wrapper, _WRAPPER_MARK, True)
    setattr(wrapper, "_cpu_s46i_source_site", str(source_site))
    return _copy_location(wrapper, statements[0])  # type: ignore[return-value]


def _iter_statement_lists(node: ast.AST):
    for _field, value in ast.iter_fields(node):
        if isinstance(value, list):
            if value and all(isinstance(item, ast.stmt) for item in value):
                yield value
            for item in value:
                if isinstance(item, ast.AST):
                    yield from _iter_statement_lists(item)
        elif isinstance(value, ast.AST):
            yield from _iter_statement_lists(value)


def _span_match(values: list[ast.stmt], unit: Any) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    for start, statement in enumerate(values):
        line = int(getattr(statement, "lineno", -1))
        selector = str(getattr(unit, "selector", ""))
        if selector in {"alpha_phase_assembly", "q_ib_multiplication"}:
            if line < int(unit.line_start) or line > int(unit.line_end):
                continue
        elif line != int(unit.line_start):
            continue
        for end in range(start, len(values)):
            candidate = values[end]
            if int(getattr(candidate, "end_lineno", getattr(candidate, "lineno", -1))) != int(unit.line_end):
                continue
            matches.append((start, end + 1))
    return matches


def _assignment_names(statement: ast.stmt) -> tuple[str, ...]:
    targets: list[ast.AST] = []
    if isinstance(statement, ast.Assign):
        targets.extend(statement.targets)
    elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
        targets.append(statement.target)
    return tuple(
        sorted(
            {
                item.id
                for target in targets
                for item in ast.walk(target)
                if isinstance(item, ast.Name)
            }
        )
    )


def _call_names(statement: ast.AST) -> set[str]:
    return {
        node.func.id
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def _has_call_attribute(statement: ast.AST, attribute: str) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
        for node in ast.walk(statement)
    )


def _unit_matches(unit: Any, statements: list[ast.stmt]) -> bool:
    names = set(_assignment_names(statements[0])) if statements else set()
    all_names = set().union(*(_call_names(statement) for statement in statements)) if statements else set()
    selector = str(unit.selector)
    if selector.startswith("aux:"):
        expected = selector[4:]
        if expected == "Qacc":
            return len(statements) == 1 and isinstance(statements[0], ast.AugAssign) and names == {"Qacc"}
        if expected == "ledger":
            return names == {"E_dep_plasma_direct"} and "E_dep_plasma_pulse" in _assignment_names(statements[-1])
        if expected == "slow_state":
            return len(statements) == 1 and isinstance(statements[0], ast.If) and any(
                isinstance(node, ast.Name) and node.id == "thermal_slow_state" for node in ast.walk(statements[0].test))
        return expected in all_names
    if selector == "inten_ion":
        return names == {"I"} and "inten_ion" in all_names
    if selector == "photo_source_fallback":
        if len(statements) != 1 or not isinstance(statements[0], ast.If):
            return False
        test = statements[0].test
        return isinstance(test, ast.Compare) and any(
            isinstance(node, ast.Name) and node.id == "photoionization_energy_rate"
            for node in ast.walk(test)
        )
    if selector == "photo_source_cleanup":
        return (
            len(statements) == 2
            and _assignment_names(statements[0]) == ("photoionization_energy_rate",)
            and isinstance(statements[1], ast.Expr)
            and _has_call_attribute(statements[1], "maximum")
        )
    if selector == "ib_alpha":
        return (
            len(statements) == 2
            and _assignment_names(statements[0]) == ("alpha_ib",)
            and "ib_alpha" in _call_names(statements[0])
            and _has_call_attribute(statements[1], "maximum")
        )
    if selector == "alpha_ion_raw":
        return len(statements) == 1 and _assignment_names(statements[0]) == ("alpha_ion_raw",)
    if selector == "alpha_ion_applied_cleanup":
        return (
            len(statements) == 4
            and _assignment_names(statements[0]) == ("ion_source_prop",)
            and _assignment_names(statements[1]) == ("alpha_ion",)
            and _assignment_names(statements[2]) == ("alpha_ion",)
            and isinstance(statements[3], ast.Expr)
            and _has_call_attribute(statements[0], "nan_to_num")
            and _has_call_attribute(statements[2], "nan_to_num")
            and _has_call_attribute(statements[3], "maximum")
        )
    if selector == "alpha_phase_assembly":
        expected = (
            "alpha_R_applied",
            "alpha_ion_applied",
            "alpha_total",
            "dphi_k",
            "dphi_p",
            "dphi_p_raw",
            "phase",
        )
        got = tuple(
            name
            for statement in statements
            for name in _assignment_names(statement)
        )
        return set(got) == set(expected) and len(got) == len(expected)
    if selector == "direct_ib_power":
        return len(statements) == 1 and _assignment_names(statements[0]) == ("ib_power",)
    if selector == "q_ib":
        return len(statements) == 1 and _assignment_names(statements[0]) == ("q_ib",) and "q_ib_from_power" in all_names
    if selector == "heat_q_per_z":
        return len(statements) == 1 and _assignment_names(statements[0]) == ("Qslice",) and "heat_Q_per_z" in all_names
    return False


def _find_unit_span(function_node: ast.FunctionDef, unit: Any) -> tuple[list[ast.stmt], int, int]:
    matches: list[tuple[list[ast.stmt], int, int]] = []
    for values in _iter_statement_lists(function_node):
        for start, end in _span_match(values, unit):
            if _unit_matches(unit, values[start:end]):
                matches.append((values, start, end))
    if len(matches) != 1:
        raise ResidualInstrumentationError(
            f"S46I_ANCHOR_COUNT:{unit.name}:{len(matches)}"
        )
    return matches[0]


def _find_call(function_node: ast.FunctionDef, name: str, line: int) -> ast.Call:
    matches = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        and int(getattr(node, "lineno", -1)) == int(line)
    ]
    if len(matches) != 1:
        raise ResidualInstrumentationError(f"S46I_ALIAS_CALL_COUNT:{name}:{line}:{len(matches)}")
    return matches[0]


def _alias_call(function_node: ast.FunctionDef, *, original: str, replacement: str, line: int) -> None:
    call = _find_call(function_node, original, line)
    call.func = ast.Name(id=replacement, ctx=ast.Load())
    setattr(call, _ALIAS_MARK, True)
    setattr(call, _ALIAS_ORIGINAL, str(original))


def instrument_residual_cost_ast(function_node: ast.FunctionDef) -> dict[str, Any]:
    """Apply all registered residual scopes to one B/C-transformed function."""

    if not isinstance(function_node, ast.FunctionDef):
        raise ResidualInstrumentationError("S46I_PROPAGATOR_FUNCTION_REQUIRED")
    baseline = copy.deepcopy(function_node)
    sites: list[dict[str, Any]] = []
    for unit in COST_UNITS + AUXILIARY_UNITS:
        values, start, end = _find_unit_span(function_node, unit)
        original = values[start:end]
        if str(unit.selector) == "photo_source_fallback":
            branch = original[0]
            if not isinstance(branch, ast.If) or not branch.body:
                raise ResidualInstrumentationError("S46I_FALLBACK_BRANCH_BODY_MISSING")
            branch.body = [
                _wrap(
                    list(branch.body),
                    source_site=unit.source_site,
                    interval_name="interval_index",
                )
            ]
        else:
            values[start:end] = [
                _wrap(
                    original,
                    source_site=unit.source_site,
                    interval_name=None if unit.scope_kind == "outside_body" else "interval_index",
                )
            ]
        sites.append(
            {
                "name": unit.name,
                "source_site": unit.source_site,
                "line_start": int(unit.line_start),
                "line_end": int(unit.line_end),
                "context": unit.context,
                "eligible": bool(getattr(unit, "eligible", True)),
                "scope_kind": str(getattr(unit, "scope_kind", "eligible_scope")),
                "branch_sensitive": bool(getattr(unit, "branch_sensitive", False)),
            }
        )

    # Enable the existing broad parents in COST without nesting a duplicate
    # parent. Stripping this alias first restores the original B wrapper.
    parent_sites = []
    for node in ast.walk(function_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "__cpu_s45b_scope__":
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value in PARENT_LABELS:
                parent_sites.append(node.args[0].value)
                setattr(node, _ALIAS_MARK, True)
                setattr(node, _ALIAS_ORIGINAL, node.func.id)
                node.func = ast.Name(id=SCOPE_HELPER, ctx=ast.Load())
    if set(parent_sites) != set(PARENT_LABELS) or parent_sites.count("INTERVAL_PREP_AND_OTHER_ARRAYS") != 2:
        raise ResidualInstrumentationError("S46I_EXISTING_PARENT_SCOPE_COVERAGE")

    # Keep call-level costs disjoint from the source scopes.  The wrappers
    # below only substitute a private global helper; they do not alter any
    # arguments or expression order.
    _alias_call(
        function_node,
        original="q_ion_from_power",
        replacement="__cpu_s46i_q_ion_from_power__",
        line=1117,
    )
    _alias_call(
        function_node,
        original="q_ib_from_power",
        replacement="__cpu_s46i_q_ib_from_power__",
        line=1119,
    )
    _alias_call(
        function_node,
        original="direct_interval_energy",
        replacement="__cpu_s46i_direct_ion_interval_energy__",
        line=1129,
    )
    _alias_call(
        function_node,
        original="direct_interval_energy",
        replacement="__cpu_s46i_direct_ib_interval_energy__",
        line=1132,
    )
    _alias_call(
        function_node,
        original="heat_Q_per_z",
        replacement="__cpu_s46i_heat_q_per_z__",
        line=1222,
    )
    ast.fix_missing_locations(function_node)
    stripped = strip_residual_cost_ast(function_node)
    if ast.dump(stripped, include_attributes=False) != ast.dump(
        baseline, include_attributes=False
    ):
        raise ResidualInstrumentationError("S46I_STRIPPED_POST_B_AST_MISMATCH")
    return {
        "schema": INSTRUMENT_SCHEMA + ".propagation",
        "source_ast_sha256": ast_identity(baseline),
        "instrumented_ast_sha256": ast_identity(function_node),
        "stripped_ast_sha256": ast_identity(stripped),
        "scope_sites": tuple(sites),
        "enabled_existing_parents": tuple(parent_sites),
        "alias_helpers": {
            "q_ion_from_power": "__cpu_s46i_q_ion_from_power__",
            "q_ib_from_power": "__cpu_s46i_q_ib_from_power__",
            "direct_ion": "__cpu_s46i_direct_ion_interval_energy__",
            "direct_ib": "__cpu_s46i_direct_ib_interval_energy__",
            "heat_Q_per_z": "__cpu_s46i_heat_q_per_z__",
        },
        "array_values_retained": False,
        "scientific_expression_changes": 0,
        "inverse_proof": "strip_residual_cost_restores_post_B_C_ast",
    }


def instrument_heat_ast(function_node: ast.FunctionDef) -> dict[str, Any]:
    """Wrap heat's pointwise source/clip block, excluding its final sum."""

    if not isinstance(function_node, ast.FunctionDef):
        raise ResidualInstrumentationError("S46I_HEAT_FUNCTION_REQUIRED")
    baseline = copy.deepcopy(function_node)
    candidates: list[tuple[list[ast.stmt], int, int]] = []
    for values in _iter_statement_lists(function_node):
        for start, end in _span_match(
            values,
            type("HeatUnit", (), {"line_start": 25, "line_end": 35})(),
        ):
            if start < end and any(
                isinstance(statement, ast.If) for statement in values[start:end]
            ) and any(
                _assignment_names(statement) == ("Q",)
                for statement in values[start:end]
            ):
                candidates.append((values, start, end))
    if len(candidates) != 1:
        raise ResidualInstrumentationError(f"S46I_HEAT_ANCHOR_COUNT:{len(candidates)}")
    values, start, end = candidates[0]
    values[start:end] = [
        _wrap(
            values[start:end],
            source_site="heat.py:25-35:heat_Q_per_z",
            interval_name=None,
        )
    ]
    ast.fix_missing_locations(function_node)
    stripped = strip_residual_cost_ast(function_node)
    if ast.dump(stripped, include_attributes=False) != ast.dump(
        baseline, include_attributes=False
    ):
        raise ResidualInstrumentationError("S46I_HEAT_STRIPPED_AST_MISMATCH")
    return {
        "schema": INSTRUMENT_SCHEMA + ".heat",
        "source_ast_sha256": ast_identity(baseline),
        "instrumented_ast_sha256": ast_identity(function_node),
        "stripped_ast_sha256": ast_identity(stripped),
        "source_site": "heat.py:25-35:heat_Q_per_z",
        "array_values_retained": False,
        "scientific_expression_changes": 0,
        "inverse_proof": "strip_residual_cost_restores_heat_ast",
    }


def instrument_q_ib_ast(function_node: ast.FunctionDef) -> dict[str, Any]:
    """Scope q_ib's pointwise source construction, excluding its reduction."""

    if not isinstance(function_node, ast.FunctionDef):
        raise ResidualInstrumentationError("S46I_Q_IB_FUNCTION_REQUIRED")
    baseline = copy.deepcopy(function_node)
    candidates: list[tuple[list[ast.stmt], int, int]] = []
    for values in _iter_statement_lists(function_node):
        for start, end in _span_match(
            values,
            type("QIbUnit", (), {"line_start": 50, "line_end": 57, "selector": "q_ib_multiplication"})(),
        ):
            if len(values[start:end]) == 3 and isinstance(values[start + 2], ast.Try):
                candidates.append((values, start, end))
    if len(candidates) != 1:
        raise ResidualInstrumentationError(f"S46I_Q_IB_ANCHOR_COUNT:{len(candidates)}")
    values, start, end = candidates[0]
    values[start:end] = [
        _wrap(
            values[start:end],
            source_site="deposition.py:50-57:q_ib_multiplication",
            interval_name=None,
        )
    ]
    ast.fix_missing_locations(function_node)
    stripped = strip_residual_cost_ast(function_node)
    if ast.dump(stripped, include_attributes=False) != ast.dump(
        baseline, include_attributes=False
    ):
        raise ResidualInstrumentationError("S46I_Q_IB_STRIPPED_AST_MISMATCH")
    return {
        "schema": INSTRUMENT_SCHEMA + ".q_ib",
        "source_ast_sha256": ast_identity(baseline),
        "instrumented_ast_sha256": ast_identity(function_node),
        "stripped_ast_sha256": ast_identity(stripped),
        "source_site": "deposition.py:50-57:q_ib_multiplication",
        "array_values_retained": False,
        "scientific_expression_changes": 0,
        "inverse_proof": "strip_residual_cost_restores_q_ib_ast",
    }


def strip_residual_cost_ast(node: ast.AST) -> ast.AST:
    """Remove only S4-6I wrappers and restore original helper names."""

    result = copy.deepcopy(node)

    class Stripper(ast.NodeTransformer):
        def visit_With(self, current: ast.With) -> Any:
            if getattr(current, _WRAPPER_MARK, False):
                restored: list[ast.stmt] = []
                for statement in current.body:
                    visited = self.visit(statement)
                    if isinstance(visited, list):
                        restored.extend(visited)
                    elif visited is not None:
                        restored.append(visited)
                return restored
            return self.generic_visit(current)

        def visit_Call(self, current: ast.Call) -> Any:
            result = self.generic_visit(current)
            if getattr(result, _ALIAS_MARK, False):
                original = getattr(result, _ALIAS_ORIGINAL, None)
                if not isinstance(original, str) or not isinstance(result.func, ast.Name):
                    raise ResidualInstrumentationError("S46I_ALIAS_ORIGINAL_MISSING")
                result.func = ast.Name(id=original, ctx=ast.Load())
            return result

    stripped = Stripper().visit(result)
    if stripped is None:
        raise ResidualInstrumentationError("S46I_STRIP_RESULT_MISSING")
    ast.fix_missing_locations(stripped)
    return stripped


__all__ = [
    "INSTRUMENT_SCHEMA",
    "ResidualInstrumentationError",
    "ast_identity",
    "instrument_heat_ast",
    "instrument_q_ib_ast",
    "instrument_residual_cost_ast",
    "strip_residual_cost_ast",
]
