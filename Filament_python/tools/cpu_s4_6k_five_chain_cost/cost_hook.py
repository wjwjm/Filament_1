"""Source-bound scalar timing overlay for the accepted five-chain factory.

This module deliberately owns no numerical kernel.  It adds reversible
``NONLINEAR_ARRAYS`` scopes around the pure-pointwise work that is not already
parallelized by :class:`AlphaParallelHook`/``PowerVolumeHook``.  The private
``q_ib`` and heat clones retain the original owner's backend, bound power
helper, integration clone, and context wrapper; the inherited helpers and
their global slots are never replaced.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import importlib
from pathlib import Path
from types import FunctionType
from typing import Any, Callable, Mapping

import numpy as np

from tools.cpu_s4_5b_steady_cost.timing import TimingRecorder, context
from tools.cpu_s4_6h_combined_parallel.instrument import _seal, verify_seal
from tools.cpu_s4_6i_residual_parallel.power_hook import (
    INTERVAL_KW as POWER_INTERVAL_KW,
    Q_IB_HELPER as POWER_Q_IB_HELPER,
    PowerHookError,
    _contains_array,
)
from tools.hpc_ops import provenance_v2 as pv

from tools.cpu_s4_6j_alpha_factory.factory_timing import source_identity as factory_source_identity

from .registry import SCOPE_HELPER, SCOPE_LABEL, SITES, registry_snapshot


COST_SCHEMA = "filament.cpu.s4_6k.five_chain_cost.cost_hook.v1"
Q_IB_HELPER = "__cpu_s46k_q_ib_from_power__"
HEAT_HELPER = "__cpu_s46k_heat_q_per_z__"

_WRAPPER_MARK = "_cpu_s46k_cost_scope_wrapper"
_WRAPPER_SITE = "_cpu_s46k_cost_scope_site"
_ALIAS_MARK = "_cpu_s46k_cost_alias_call"
_ALIAS_ORIGINAL = "_cpu_s46k_cost_alias_original"
_ALIAS_REPLACEMENT = "_cpu_s46k_cost_alias_replacement"


class FiveChainCostError(PowerHookError):
    """A source, owner, backend, recorder, or inverse proof failed closed."""


def _ast_hash(node: ast.AST) -> str:
    return pv.raw_sha256(ast.dump(node, include_attributes=False).encode("utf-8"))


def _dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


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
        args=[ast.Constant(value=str(source_site)), interval],
        keywords=[],
    )


def _wrap(
    statements: list[ast.stmt], *, source_site: str, interval_name: str | None,
) -> ast.With:
    if not statements:
        raise FiveChainCostError("S46K_EMPTY_COST_SCOPE")
    wrapper = ast.With(
        items=[ast.withitem(context_expr=_scope_call(source_site, interval_name), optional_vars=None)],
        body=list(statements),
        type_comment=None,
    )
    setattr(wrapper, _WRAPPER_MARK, True)
    setattr(wrapper, _WRAPPER_SITE, str(source_site))
    return _copy_location(wrapper, statements[0])  # type: ignore[return-value]


def _assignment_names(statement: ast.AST) -> tuple[str, ...]:
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


def _has_attribute_call(statement: ast.AST, attribute: str) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
        for node in ast.walk(statement)
    )


def _find_span(
    function_node: ast.FunctionDef,
    line_start: int,
    line_end: int,
    predicate: Callable[[list[ast.stmt]], bool],
) -> tuple[list[ast.stmt], int, int]:
    matches: list[tuple[list[ast.stmt], int, int]] = []
    for values in _iter_statement_lists(function_node):
        for start, statement in enumerate(values):
            if int(getattr(statement, "lineno", -1)) != int(line_start):
                continue
            for end in range(start + 1, len(values) + 1):
                last = values[end - 1]
                if int(getattr(last, "end_lineno", getattr(last, "lineno", -1))) != int(line_end):
                    continue
                candidate = values[start:end]
                if predicate(candidate):
                    matches.append((values, start, end))
    if len(matches) != 1:
        raise FiveChainCostError(f"S46K_COST_ANCHOR_COUNT:{line_start}-{line_end}:{len(matches)}")
    return matches[0]


def _find_statement(
    function_node: ast.FunctionDef,
    line: int,
    predicate: Callable[[ast.stmt], bool],
) -> tuple[list[ast.stmt], int]:
    matches: list[tuple[list[ast.stmt], int]] = []
    for values in _iter_statement_lists(function_node):
        for index, statement in enumerate(values):
            if int(getattr(statement, "lineno", -1)) == int(line) and predicate(statement):
                matches.append((values, index))
    if len(matches) != 1:
        raise FiveChainCostError(f"S46K_COST_STATEMENT_COUNT:{line}:{len(matches)}")
    return matches[0]


def _find_if(function_node: ast.FunctionDef, line: int) -> ast.If:
    matches = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.If)
        and int(getattr(node, "lineno", -1)) == int(line)
        and any(
            isinstance(item, ast.Name) and item.id == "photoionization_energy_rate"
            for item in ast.walk(node.test)
        )
    ]
    if len(matches) != 1:
        raise FiveChainCostError(f"S46K_FALLBACK_ANCHOR_COUNT:{line}:{len(matches)}")
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
        raise FiveChainCostError(f"S46K_COST_ALIAS_COUNT:{name}:{line}:{len(matches)}")
    return matches[0]


def _replace_call(
    function_node: ast.FunctionDef,
    *,
    original: str,
    replacement: str,
    line: int,
) -> None:
    call = _find_call(function_node, original, line)
    call.func = ast.Name(id=replacement, ctx=ast.Load())
    setattr(call, _ALIAS_MARK, True)
    setattr(call, _ALIAS_ORIGINAL, str(original))
    setattr(call, _ALIAS_REPLACEMENT, str(replacement))


def _site_by_name(name: str):
    for site in SITES:
        if site.name == name:
            return site
    raise FiveChainCostError("S46K_COST_SITE_NOT_REGISTERED:" + str(name))


def _site_map() -> dict[str, Any]:
    return {site.source_site: site for site in SITES}


def instrument_cost_ast(function_node: ast.FunctionDef, *, enabled: bool = True) -> dict[str, Any]:
    """Add only the registered five-chain timing scopes/aliases."""

    if not isinstance(function_node, ast.FunctionDef):
        raise FiveChainCostError("S46K_COST_PROPAGATOR_FUNCTION_REQUIRED")
    baseline = copy.deepcopy(function_node)
    if not enabled:
        return {
            "schema": COST_SCHEMA + ".propagation.v1",
            "enabled": False,
            "source_ast_sha256": _ast_hash(baseline),
            "instrumented_ast_sha256": _ast_hash(function_node),
            "stripped_ast_sha256": _ast_hash(function_node),
            "sites": (),
            "aliases": {},
            "mathematical_changes": 0,
            "array_values_retained": False,
            "inverse_proof": "disabled_no_generated_nodes",
        }

    site_rows: list[dict[str, Any]] = []

    def record(name: str) -> None:
        site = _site_by_name(name)
        site_rows.append(
            {
                "name": site.name,
                "unit": site.unit,
                "source_site": site.source_site,
                "source_order": int(site.source_order),
                "eligible": bool(site.eligible),
                "scope_kind": site.scope_kind,
                "parent": bool(site.parent),
            }
        )

    # Pre-ionization pure pointwise spans.  The exact assignment/call shapes
    # make a changed source fail closed rather than moving a clock silently.
    values, start, end = _find_span(
        function_node,
        690,
        690,
        lambda block: len(block) == 1
        and _assignment_names(block[0]) == ("I",)
        and "inten_ion" in _call_names(block[0]),
    )
    site = _site_by_name("prep_intensity")
    values[start:end] = [_wrap(values[start:end], source_site=site.source_site, interval_name="interval_index")]
    record(site.name)

    fallback = _find_if(function_node, 790)
    if not fallback.body:
        raise FiveChainCostError("S46K_FALLBACK_BRANCH_BODY_MISSING")
    site = _site_by_name("photo_source_fallback")
    fallback.body = [_wrap(list(fallback.body), source_site=site.source_site, interval_name="interval_index")]
    record(site.name)

    values, start, end = _find_span(
        function_node,
        797,
        803,
        lambda block: len(block) == 2
        and _assignment_names(block[0]) == ("photoionization_energy_rate",)
        and isinstance(block[1], ast.Expr)
        and _has_attribute_call(block[1], "maximum"),
    )
    site = _site_by_name("photo_source_cleanup")
    values[start:end] = [_wrap(values[start:end], source_site=site.source_site, interval_name="interval_index")]
    record(site.name)

    values, start, end = _find_span(
        function_node,
        814,
        815,
        lambda block: len(block) == 2
        and _assignment_names(block[0]) == ("alpha_ib",)
        and "ib_alpha" in _call_names(block[0])
        and _has_attribute_call(block[1], "maximum"),
    )
    site = _site_by_name("ib_alpha")
    values[start:end] = [_wrap(values[start:end], source_site=site.source_site, interval_name="interval_index")]
    record(site.name)

    values, start, end = _find_span(
        function_node,
        819,
        819,
        lambda block: len(block) == 1 and _assignment_names(block[0]) == ("alpha_ion_raw",),
    )
    site = _site_by_name("alpha_ion_raw")
    values[start:end] = [_wrap(values[start:end], source_site=site.source_site, interval_name="interval_index")]
    record(site.name)

    # The parent is retained only for clock ancestry/closure.  The four
    # arithmetic children are the ranking units; selectors remain visible but
    # explicitly non-eligible.
    values, start, end = _find_span(
        function_node,
        947,
        955,
        lambda block: set(
            name for statement in block for name in _assignment_names(statement)
        )
        == {"alpha_ion_applied", "alpha_R_applied", "alpha_total", "dphi_k", "dphi_p_raw", "dphi_p", "phase"},
    )
    parent_site = _site_by_name("alpha_phase_assembly")
    parent = _wrap(values[start:end], source_site=parent_site.source_site, interval_name="interval_index")
    values[start:end] = [parent]
    record(parent_site.name)

    phase_lines = {
        "alpha_ion_applied": 947,
        "alpha_R": 948,
        "alpha_total": 949,
        "kerr": 952,
        "rawplasma": 953,
        "plasma": 954,
        "phase": 955,
    }
    for name, line in phase_lines.items():
        values_child, index = _find_statement(
            function_node,
            line,
            lambda statement, line=line: isinstance(statement, (ast.Assign, ast.AnnAssign))
            and bool(_assignment_names(statement)),
        )
        site = _site_by_name(name)
        values_child[index : index + 1] = [
            _wrap([values_child[index]], source_site=site.source_site, interval_name="interval_index")
        ]
        record(site.name)

    values, start, end = _find_span(
        function_node,
        1118,
        1118,
        lambda block: len(block) == 1 and _assignment_names(block[0]) == ("ib_power",),
    )
    site = _site_by_name("direct_ib_power")
    values[start:end] = [_wrap(values[start:end], source_site=site.source_site, interval_name="interval_index")]
    record(site.name)

    # The power hook has already replaced q_ib_from_power with its private
    # helper.  Only that existing call is redirected to our private, timed
    # source clone; the owner's four helper slots remain untouched.
    _replace_call(
        function_node,
        original=POWER_Q_IB_HELPER,
        replacement=Q_IB_HELPER,
        line=1119,
    )
    _replace_call(
        function_node,
        original="heat_Q_per_z",
        replacement=HEAT_HELPER,
        line=1222,
    )
    record("q_ib_multiplication")
    record("heat_q_pointwise")

    ast.fix_missing_locations(function_node)
    stripped = strip_cost_ast(function_node)
    if _dump(stripped) != _dump(baseline):
        raise FiveChainCostError("S46K_COST_STRIPPED_AST_MISMATCH")
    return {
        "schema": COST_SCHEMA + ".propagation.v1",
        "enabled": True,
        "source_ast_sha256": _ast_hash(baseline),
        "instrumented_ast_sha256": _ast_hash(function_node),
        "stripped_ast_sha256": _ast_hash(stripped),
        "sites": tuple(site_rows),
        "aliases": {"q_ib_from_power": Q_IB_HELPER, "heat_Q_per_z": HEAT_HELPER},
        "mathematical_changes": 0,
        "array_values_retained": False,
        "inverse_proof": "strip_cost_ast_restores_post_alpha_power_ast",
    }


def strip_cost_ast(node: ast.AST) -> ast.AST:
    """Remove only the S4-6K wrappers/aliases and reject tampering."""

    result = copy.deepcopy(node)
    wrapper_sites: list[str] = []
    alias_pairs: list[tuple[str, str]] = []

    class Stripper(ast.NodeTransformer):
        def visit_With(self, current: ast.With) -> Any:
            if not bool(getattr(current, _WRAPPER_MARK, False)):
                return self.generic_visit(current)
            if len(current.items) != 1:
                raise FiveChainCostError("S46K_COST_WRAPPER_ITEMS_CHANGED")
            expression = current.items[0].context_expr
            if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
                raise FiveChainCostError("S46K_COST_WRAPPER_CALL_CHANGED")
            if expression.func.id != SCOPE_HELPER or len(expression.args) != 2:
                raise FiveChainCostError("S46K_COST_WRAPPER_HELPER_CHANGED")
            if not isinstance(expression.args[0], ast.Constant):
                raise FiveChainCostError("S46K_COST_WRAPPER_SITE_CHANGED")
            site = str(expression.args[0].value)
            if site not in _site_map():
                raise FiveChainCostError("S46K_COST_WRAPPER_SITE_UNKNOWN:" + site)
            wrapper_sites.append(site)
            body: list[ast.stmt] = []
            for statement in current.body:
                visited = self.visit(statement)
                if isinstance(visited, list):
                    body.extend(item for item in visited if item is not None)
                elif visited is not None:
                    body.append(visited)
            return body

        def visit_Call(self, current: ast.Call) -> Any:
            value = self.generic_visit(current)
            if not bool(getattr(value, _ALIAS_MARK, False)):
                return value
            if not isinstance(value.func, ast.Name):
                raise FiveChainCostError("S46K_COST_ALIAS_FUNCTION_CHANGED")
            original = getattr(value, _ALIAS_ORIGINAL, None)
            replacement = getattr(value, _ALIAS_REPLACEMENT, None)
            if not isinstance(original, str) or not isinstance(replacement, str):
                raise FiveChainCostError("S46K_COST_ALIAS_METADATA_MISSING")
            if value.func.id != replacement or replacement not in {Q_IB_HELPER, HEAT_HELPER}:
                raise FiveChainCostError("S46K_COST_ALIAS_REPLACEMENT_CHANGED")
            alias_pairs.append((original, replacement))
            value.func = ast.Name(id=original, ctx=ast.Load())
            return value

    stripped = Stripper().visit(result)
    if stripped is None:
        raise FiveChainCostError("S46K_COST_STRIP_RESULT_MISSING")
    ast.fix_missing_locations(stripped)
    return stripped


def _compile_function(
    node: ast.FunctionDef,
    *,
    namespace: dict[str, Any],
    source_path: Path,
    name: str,
) -> FunctionType:
    node = copy.deepcopy(node)
    node.name = str(name)
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    local: dict[str, Any] = {}
    exec(compile(module, str(source_path), "exec", dont_inherit=True, optimize=0), namespace, local)
    function = local.get(name)
    if not isinstance(function, FunctionType) or function.__globals__ is not namespace:
        raise FiveChainCostError("S46K_PRIVATE_CLONE_COMPILE_FAILED:" + str(name))
    return function


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise FiveChainCostError(f"S46K_SOURCE_FUNCTION_COUNT:{name}:{len(matches)}")
    return matches[0]


def _source_ast(path: Path, name: str) -> tuple[ast.FunctionDef, str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    node = _find_function(tree, name)
    return copy.deepcopy(node), _ast_hash(node)


def _instrument_q_ib_clone(node: ast.FunctionDef) -> ast.FunctionDef:
    baseline = copy.deepcopy(node)
    matches = []
    for values in _iter_statement_lists(node):
        for start, statement in enumerate(values):
            if int(getattr(statement, "lineno", -1)) != 52:
                continue
            if start + 2 >= len(values):
                continue
            if int(getattr(values[start + 2], "lineno", -1)) != 54:
                continue
            if _assignment_names(values[start]) != ("alpha",) or _assignment_names(values[start + 1]) != (
                "optical_intensity",
            ):
                continue
            if not isinstance(values[start + 2], ast.Try):
                continue
            matches.append((values, start))
    if len(matches) != 1:
        raise FiveChainCostError("S46K_Q_IB_CLONE_ANCHOR_COUNT:" + str(len(matches)))
    values, start = matches[0]
    site = _site_by_name("q_ib_multiplication")
    values[start : start + 3] = [_wrap(values[start : start + 3], source_site=site.source_site, interval_name=None)]
    ast.fix_missing_locations(node)
    if _dump(strip_cost_ast(node)) != _dump(baseline):
        raise FiveChainCostError("S46K_Q_IB_CLONE_INVERSE_AST_MISMATCH")
    return node


def _instrument_heat_clone(node: ast.FunctionDef) -> ast.FunctionDef:
    baseline = copy.deepcopy(node)
    candidates: list[tuple[list[ast.stmt], int, int]] = []
    for values in _iter_statement_lists(node):
        for start, statement in enumerate(values):
            if int(getattr(statement, "lineno", -1)) != 25:
                continue
            for end in range(start + 1, len(values) + 1):
                if int(getattr(values[end - 1], "end_lineno", getattr(values[end - 1], "lineno", -1))) != 35:
                    continue
                block = values[start:end]
                if any(isinstance(item, ast.If) for item in block) and any(
                    _assignment_names(item) == ("Q",) for item in block
                ):
                    candidates.append((values, start, end))
    if len(candidates) != 1:
        raise FiveChainCostError("S46K_HEAT_CLONE_ANCHOR_COUNT:" + str(len(candidates)))
    values, start, end = candidates[0]
    site = _site_by_name("heat_q_pointwise")
    values[start:end] = [_wrap(values[start:end], source_site=site.source_site, interval_name=None)]
    ast.fix_missing_locations(node)
    if _dump(strip_cost_ast(node)) != _dump(baseline):
        raise FiveChainCostError("S46K_HEAT_CLONE_INVERSE_AST_MISMATCH")
    return node


class FiveChainCostHook:
    """Optional timing-only overlay attached to one ``AlphaParallelHook``."""

    def __init__(self, recorder: TimingRecorder, *, enabled: bool = False) -> None:
        if type(recorder) is not TimingRecorder:
            raise FiveChainCostError("S46K_RECORDER_EXACT_TYPE_REQUIRED")
        if bool(getattr(recorder, "enabled", True)):
            raise FiveChainCostError("S46K_RECORDER_PROFILE_MUST_BE_OFF")
        if not isinstance(enabled, bool):
            raise FiveChainCostError("S46K_COST_ENABLED_MUST_BE_BOOL")
        self.recorder = recorder
        self._bound_recorder = recorder
        self.enabled = bool(enabled)
        self._bound_enabled = self.enabled
        self._owner: Any | None = None
        self._bound_owner: Any | None = None
        self._scope_helper = self._make_scope_helper()
        self._deposition = importlib.import_module("KHz_filament.deposition")
        self._heat = importlib.import_module("KHz_filament.heat")
        self._deposition_path = Path(str(self._deposition.__file__)).resolve(strict=True)
        self._heat_path = Path(str(self._heat.__file__)).resolve(strict=True)
        self._original_q_ib = getattr(self._deposition, "q_ib_from_power", None)
        self._original_heat = getattr(self._heat, "heat_Q_per_z", None)
        if not callable(self._original_q_ib) or not callable(self._original_heat):
            raise FiveChainCostError("S46K_SOURCE_HELPER_MISSING")
        self._q_ib_source_node, self._q_ib_source_hash = _source_ast(self._deposition_path, "q_ib_from_power")
        self._heat_source_node, self._heat_source_hash = _source_ast(self._heat_path, "heat_Q_per_z")
        self._source_identity = factory_source_identity(
            extra_paths=(
                __file__,
                Path(__file__).with_name("registry.py"),
                self._deposition_path,
                self._heat_path,
            )
        )
        self._source_fingerprint = copy.deepcopy(self._source_identity)
        self._backend = {
            name: getattr(np, name)
            for name in ("asarray", "nan_to_num", "maximum", "clip", "sum")
        }
        self._recorder_methods = {
            name: getattr(type(recorder), name)
            for name in ("scope", "context", "current_context", "bridged_context")
        }
        self._recorder_method_seals = {name: _seal(function) for name, function in self._recorder_methods.items()}
        self._scope_seal = _seal(self._scope_helper)
        self._method_seals = {
            name: _seal(function) for name, function in vars(type(self)).items()
            if isinstance(function, FunctionType)
        }
        self._transform_seals = {
            name: _seal(function) for name, function in globals().items()
            if isinstance(function, FunctionType) and function.__module__ == __name__
        }
        self._original_seals = {
            'q_ib': _seal(self._original_q_ib), 'heat': _seal(self._original_heat)
        }
        self._original_slots = {
            name: {key: function.__globals__[key] for key in function.__code__.co_names
                   if key in function.__globals__}
            for name, function in (('q_ib', self._original_q_ib), ('heat', self._original_heat))
        }
        self._runtime_helpers: dict[str, Callable[..., Any]] = {}
        self._q_ib_clone: FunctionType | None = None
        self._heat_clone: FunctionType | None = None
        self._q_ib_wrapper: Callable[..., Any] | None = None
        self._q_ib_namespace: dict[str, Any] | None = None
        self._heat_namespace: dict[str, Any] | None = None
        self._clone_seals: dict[str, Mapping[str, Any]] = {}
        self._runtime_seals: dict[str, Mapping[str, Any]] = {}
        self._owner_slots: dict[str, Any] = {}
        self._active_globals: Mapping[str, Any] | None = None
        self._receipt = self._make_receipt()
        self._receipt_fingerprint = copy.deepcopy(self._receipt)

    def _make_scope_helper(self) -> Callable[..., Any]:
        owner = self
        allowed_sites = frozenset(site.source_site for site in SITES)

        def scope(source_site: str, interval_index: int | None = None):
            if not isinstance(source_site, str) or source_site not in allowed_sites:
                raise FiveChainCostError("S46K_SCOPE_SOURCE_SITE_INVALID")
            if interval_index is not None and (
                type(interval_index) is not int or not 0 <= interval_index < 8
            ):
                raise FiveChainCostError("S46K_SCOPE_INTERVAL_DOMAIN")
            if not owner.enabled:
                return contextlib.nullcontext()
            recorder = owner.recorder
            current = recorder.current_context()
            if interval_index is None:
                metadata = recorder.bridged_context(
                    require_companion=False,
                    source_site=source_site,
                )
            else:
                metadata = context(
                    int(interval_index),
                    current.phase,
                    current.stage,
                    ordinal=int(interval_index),
                    attempt=1,
                    occurrence_class="PER_INTERVAL",
                    source_site=source_site,
                )
            # This is the explicit common=True path; the recorder remains
            # profile=False and no fine-grained PROFILE scopes are enabled.
            return recorder.scope(SCOPE_LABEL, context=metadata, common=True)

        scope.__name__ = SCOPE_HELPER
        scope.__qualname__ = SCOPE_HELPER
        scope.__cpu_s46k_kind__ = "FIXED_FIVE_CHAIN_COST_SCOPE"
        scope.__cpu_s46k_recorder__ = self.recorder
        return scope

    def _make_receipt(self) -> dict[str, Any]:
        return {
            "schema": COST_SCHEMA,
            "enabled": bool(self.enabled),
            "scope_label": SCOPE_LABEL,
            "scope_helper": SCOPE_HELPER,
            "runtime_helpers": tuple(sorted(self._runtime_helpers)),
            "registry": registry_snapshot(),
            "source_identity": copy.deepcopy(self._source_identity),
            "source_ast_hashes": {
                "deposition.q_ib_from_power": self._q_ib_source_hash,
                "heat.heat_Q_per_z": self._heat_source_hash,
            },
            "owner_bound": self._owner is not None,
            "parallel_math_untouched": True,
            "common_scope": True,
            "profile": False,
            "array_values_retained": False,
            "scientific_expression_changes": 0,
        }

    def bind_owner(self, owner: Any) -> None:
        """Bind to the already-constructed ``AlphaParallelHook`` once."""

        if self._owner is not None:
            if owner is not self._owner:
                raise FiveChainCostError("S46K_OWNER_IDENTITY_CHANGED")
            return
        required = ("_private_dep_namespace", "_bound_power", "_source_xp", "_make_context_wrapper")
        if not all(hasattr(owner, name) for name in required):
            raise FiveChainCostError("S46K_ALPHA_OWNER_INTERFACE_MISSING")
        private = getattr(owner, "_private_dep_namespace")
        bound_power = getattr(owner, "_bound_power")
        source_xp = getattr(owner, "_source_xp")
        if not isinstance(private, dict) or private.get("_power_volume") is not bound_power:
            raise FiveChainCostError("S46K_OWNER_BOUND_POWER_CHANGED")
        if source_xp is not np or private.get("xp") is not source_xp:
            raise FiveChainCostError("S46K_OWNER_BACKEND_CHANGED")
        integrate = private.get("integrate_power_to_q")
        if not callable(integrate):
            raise FiveChainCostError("S46K_OWNER_INTEGRATE_CLONE_MISSING")
        wrapper_method = getattr(owner, "_make_context_wrapper")
        if not callable(wrapper_method):
            raise FiveChainCostError("S46K_OWNER_CONTEXT_WRAPPER_MISSING")

        qib_node = _instrument_q_ib_clone(copy.deepcopy(self._q_ib_source_node))
        qib_namespace = dict(private)
        qib_namespace[SCOPE_HELPER] = self._scope_helper
        qib_clone = _compile_function(
            qib_node,
            namespace=qib_namespace,
            source_path=self._deposition_path,
            name="__cpu_s46k_q_ib_source_clone__",
        )
        heat_node = _instrument_heat_clone(copy.deepcopy(self._heat_source_node))
        heat_namespace = dict(vars(self._heat))
        heat_namespace[SCOPE_HELPER] = self._scope_helper
        heat_clone = _compile_function(
            heat_node,
            namespace=heat_namespace,
            source_path=self._heat_path,
            name="__cpu_s46k_heat_q_per_z_source_clone__",
        )
        qib_wrapper = wrapper_method(qib_clone, "q_ib")
        if getattr(qib_wrapper, "__cpu_s46i_power_volume_target__", None) is not qib_clone:
            raise FiveChainCostError("S46K_OWNER_CONTEXT_TARGET_CHANGED")

        self._owner = owner
        self._bound_owner = owner
        self._owner_slots = {
            "private_namespace": private,
            "bound_power": bound_power,
            "source_xp": source_xp,
            "integrate_power_to_q": integrate,
            "context_wrapper": getattr(wrapper_method, "__func__", wrapper_method),
            "context_local": owner._context_local,
        }
        self._q_ib_namespace = qib_namespace
        self._heat_namespace = heat_namespace
        self._q_ib_clone = qib_clone
        self._heat_clone = heat_clone
        self._q_ib_wrapper = qib_wrapper
        self._runtime_helpers = {
            SCOPE_HELPER: self._scope_helper,
            Q_IB_HELPER: qib_wrapper,
            HEAT_HELPER: heat_clone,
        }
        self._clone_seals = {
            "q_ib": _seal(qib_clone),
            "heat": _seal(heat_clone),
        }
        self._runtime_seals = {name: _seal(helper) for name, helper in self._runtime_helpers.items()}
        self._clone_slots = {
            name: {key: function.__globals__[key] for key in function.__code__.co_names
                   if key in function.__globals__}
            for name, function in (('q_ib', qib_clone), ('heat', heat_clone))
        }
        self._receipt = self._make_receipt()
        self._receipt_fingerprint = copy.deepcopy(self._receipt)

    def runtime_helpers(self) -> dict[str, Callable[..., Any]]:
        return dict(self._runtime_helpers)

    def instrument(self, function_node: ast.FunctionDef) -> dict[str, Any]:
        receipt = instrument_cost_ast(function_node, enabled=self.enabled)
        self._last_propagation_receipt = copy.deepcopy(receipt)
        return receipt

    def strip_generated_nodes(self, node: ast.AST) -> ast.AST:
        return strip_cost_ast(node)

    def binding(self) -> dict[str, Any]:
        return copy.deepcopy(self._receipt)

    def bind_active(self, active_globals: Mapping[str, Any]) -> None:
        if not isinstance(active_globals, dict):
            raise FiveChainCostError("S46K_ACTIVE_GLOBALS_REQUIRED")
        self.verify_binding(active_globals)
        self._active_globals = active_globals

    def verify_binding(self, active_globals: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if type(self.recorder) is not TimingRecorder or self.recorder is not self._bound_recorder:
            raise FiveChainCostError("S46K_RECORDER_IDENTITY_CHANGED")
        if bool(self.recorder.enabled) or self.enabled is not self._bound_enabled:
            raise FiveChainCostError("S46K_RECORDER_PROFILE_OR_ENABLE_CHANGED")
        if self._receipt != self._receipt_fingerprint:
            raise FiveChainCostError("S46K_RECEIPT_CHANGED")
        if factory_source_identity(
            extra_paths=(
                __file__,
                Path(__file__).with_name("registry.py"),
                self._deposition_path,
                self._heat_path,
            )
        ) != self._source_fingerprint:
            raise FiveChainCostError("S46K_SOURCE_IDENTITY_CHANGED")
        for name, function in self._recorder_methods.items():
            actual = getattr(type(self.recorder), name, None)
            if actual is not function:
                raise FiveChainCostError("S46K_RECORDER_METHOD_CHANGED:" + name)
            verify_seal(self._recorder_method_seals[name])
        if getattr(self._deposition, "q_ib_from_power", None) is not self._original_q_ib:
            raise FiveChainCostError("S46K_Q_IB_SOURCE_FUNCTION_CHANGED")
        if getattr(self._heat, "heat_Q_per_z", None) is not self._original_heat:
            raise FiveChainCostError("S46K_HEAT_SOURCE_FUNCTION_CHANGED")
        current_qib, current_qib_hash = _source_ast(self._deposition_path, "q_ib_from_power")
        current_heat, current_heat_hash = _source_ast(self._heat_path, "heat_Q_per_z")
        if current_qib_hash != self._q_ib_source_hash or current_heat_hash != self._heat_source_hash:
            raise FiveChainCostError("S46K_SOURCE_AST_CHANGED")
        for name, function in self._backend.items():
            if getattr(np, name, None) is not function:
                raise FiveChainCostError("S46K_BACKEND_FUNCTION_CHANGED:" + name)
        verify_seal(self._scope_seal)
        for name, seal in self._method_seals.items():
            if getattr(type(self), name) is not seal['function']:
                raise FiveChainCostError('S46K_HOOK_METHOD_CHANGED:' + name)
            verify_seal(seal)
        for name, seal in self._transform_seals.items():
            if globals().get(name) is not seal['function']:
                raise FiveChainCostError('S46K_TRANSFORM_ALIAS_CHANGED:' + name)
            verify_seal(seal)
        for name, seal in self._original_seals.items():
            verify_seal(seal)
            for key, expected in self._original_slots[name].items():
                if seal['function'].__globals__.get(key) is not expected:
                    raise FiveChainCostError('S46K_ORIGINAL_GLOBAL_CHANGED:' + name + ':' + key)
        if self._owner is not self._bound_owner:
            raise FiveChainCostError("S46K_OWNER_IDENTITY_CHANGED")
        if self._owner is not None:
            owner = self._owner
            if owner._context_local is not self._owner_slots['context_local']:
                raise FiveChainCostError('S46K_OWNER_CONTEXT_LOCAL_CHANGED')
            if getattr(owner, "_bound_power", None) is not self._owner_slots["bound_power"]:
                raise FiveChainCostError("S46K_OWNER_BOUND_POWER_CHANGED")
            if getattr(owner, "_source_xp", None) is not self._owner_slots["source_xp"]:
                raise FiveChainCostError("S46K_OWNER_BACKEND_CHANGED")
            if getattr(owner, "_private_dep_namespace", None) is not self._owner_slots["private_namespace"]:
                raise FiveChainCostError("S46K_OWNER_NAMESPACE_CHANGED")
            private = self._owner_slots["private_namespace"]
            if private.get("_power_volume") is not self._owner_slots["bound_power"]:
                raise FiveChainCostError("S46K_OWNER_POWER_SLOT_CHANGED")
            if private.get("integrate_power_to_q") is not self._owner_slots["integrate_power_to_q"]:
                raise FiveChainCostError("S46K_OWNER_INTEGRATE_SLOT_CHANGED")
            if private.get("xp") is not self._owner_slots["source_xp"]:
                raise FiveChainCostError("S46K_OWNER_XP_SLOT_CHANGED")
            wrapper_method = getattr(owner, "_make_context_wrapper", None)
            if getattr(wrapper_method, "__func__", wrapper_method) is not self._owner_slots["context_wrapper"]:
                raise FiveChainCostError("S46K_OWNER_CONTEXT_WRAPPER_CHANGED")
            if self._q_ib_clone is None or self._heat_clone is None or self._q_ib_wrapper is None:
                raise FiveChainCostError("S46K_PRIVATE_CLONES_MISSING")
            for name, function in self._clone_seals.items():
                verify_seal(function)
            if self._q_ib_clone.__globals__ is not self._q_ib_namespace:
                raise FiveChainCostError("S46K_Q_IB_NAMESPACE_CHANGED")
            if self._heat_clone.__globals__ is not self._heat_namespace:
                raise FiveChainCostError("S46K_HEAT_NAMESPACE_CHANGED")
            if self._q_ib_namespace.get("_power_volume") is not self._owner_slots["bound_power"]:
                raise FiveChainCostError("S46K_Q_IB_POWER_SLOT_CHANGED")
            if self._q_ib_namespace.get("integrate_power_to_q") is not self._owner_slots["integrate_power_to_q"]:
                raise FiveChainCostError("S46K_Q_IB_INTEGRATE_SLOT_CHANGED")
            if self._q_ib_namespace.get("xp") is not self._owner_slots["source_xp"]:
                raise FiveChainCostError("S46K_Q_IB_XP_SLOT_CHANGED")
            if self._heat_namespace.get("xp") is not getattr(self._heat, "xp", None):
                raise FiveChainCostError("S46K_HEAT_XP_SLOT_CHANGED")
            if getattr(self._q_ib_wrapper, "__cpu_s46i_power_volume_target__", None) is not self._q_ib_clone:
                raise FiveChainCostError("S46K_Q_IB_WRAPPER_TARGET_CHANGED")
            if getattr(self._q_ib_wrapper, "__cpu_s46i_power_volume_context__", None) != "q_ib":
                raise FiveChainCostError("S46K_Q_IB_WRAPPER_CONTEXT_CHANGED")
            for name, seal in self._runtime_seals.items():
                if self._runtime_helpers.get(name) is not seal['function']:
                    raise FiveChainCostError('S46K_RUNTIME_HELPER_IDENTITY_CHANGED:' + name)
                verify_seal(seal)
            for name, function in (('q_ib', self._q_ib_clone), ('heat', self._heat_clone)):
                for key, expected in self._clone_slots[name].items():
                    if function.__globals__.get(key) is not expected:
                        raise FiveChainCostError('S46K_CLONE_GLOBAL_CHANGED:' + name + ':' + key)
        if active_globals is not None:
            if not isinstance(active_globals, dict):
                raise FiveChainCostError("S46K_ACTIVE_GLOBALS_INVALID")
            for name, helper in self._runtime_helpers.items():
                if active_globals.get(name) is not helper:
                    raise FiveChainCostError("S46K_ACTIVE_HELPER_CHANGED:" + name)
            if active_globals.get("xp") is not np:
                raise FiveChainCostError("S46K_ACTIVE_BACKEND_CHANGED")
            self._active_globals = active_globals
        return {
            "schema": COST_SCHEMA + ".binding.v1",
            "enabled": bool(self.enabled),
            "profile": False,
            "common_scope": True,
            "owner_bound": self._owner is not None,
            "runtime_helpers": tuple(sorted(self._runtime_helpers)),
            "source_bound": True,
            "backend_bound": True,
            "closure_arrays": False,
            "parallel_math_untouched": True,
        }

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.recorder.snapshot())


__all__ = [
    "COST_SCHEMA",
    "FiveChainCostError",
    "FiveChainCostHook",
    "HEAT_HELPER",
    "Q_IB_HELPER",
    "SCOPE_HELPER",
    "SCOPE_LABEL",
    "instrument_cost_ast",
    "strip_cost_ast",
]
