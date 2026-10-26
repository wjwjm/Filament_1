"""Source-bound scalar timing adapter for the CPU-S4-3G1 comparison.

The adapter clones a propagation function in a private globals dictionary and
inserts only scalar clock calls at the frozen propagation anchors.  It never
patches a source module, a factory cache, a dispatch object, or a field.  The
same AST clock placement is used for the original I0 function, the inherited
S4-3E/F transformer, and the private S4-3G call-site builder.
"""

from __future__ import annotations

import ast
import builtins
import copy
import importlib
import subprocess
import sys
import time
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any, Callable, Mapping

from tools.hpc_ops import provenance_v2 as _provenance_v2


SCHEMA = "filament.cpu_s4_3g1.timing_adapter.v1"
PROPAGATE_RELATIVE_PATH = "Filament_python/KHz_filament/propagate.py"
PROPAGATE_SOURCE_BLOB = "f156111398415c2b92e4609ccc1da075fcf5d1e3"
PROPAGATE_CANONICAL_LF_SHA256 = (
    "909fbcf5ef05d2c50b8ed27506b450d40127b3973709d9b4b6a961799dc8d1f6"
)
BK_LINES = (649, 1248)
APPLY_LINE = 963
RAMAN_STAGE_LINE = 617
RAMAN_APPLY_LINE = 618
POST_RAMAN_LINE = 969
FINAL_RETURN_LINE = 2133
_MONITORED_BK_NAMES = frozenset({"step_linear_bk_nee_factorized", "_g_step_linear_bk_nee"})
_MONITORED_APPLY_NAMES = frozenset({"apply_nonlinear", "_g_apply_nonlinear"})
_MONITORED_STAGE_NAMES = frozenset({"isaacs_raman_stage", "isaacs_complete_eq27_stage"})
_MONITORED_APPLY_RAMAN_NAMES = frozenset(
    {"apply_isaacs_raman_operator_step", "apply_isaacs_complete_eq27_operator_step"}
)
_TRANSFORMED_RAMAN_SUBSTEP_NAMES = frozenset(
    {"_cpu_s4_3e_full_raman_substep", "_g_controlled_raman_substep"}
)


class TimingBindingError(RuntimeError):
    """The requested source role or frozen timing anchor is unavailable."""


class WindowClock:
    """Scalar-only interval and nested-Heun clock.

    ``rows`` and ``half_rows`` contain scalar metadata and durations only.  No
    field, stage, diagnostic, or other array reference is retained.  A call
    that raises or leaves an interval incomplete is aborted and contributes no
    formal row.
    """

    __slots__ = (
        "role",
        "rows",
        "half_rows",
        "call_count",
        "completed_calls",
        "aborted_calls",
        "_call_open",
        "_segment_open",
        "_segment_index",
        "_segment_start_wall",
        "_segment_start_cpu",
        "_heun_open",
        "_heun_index",
        "_heun_start_wall",
        "_heun_start_cpu",
        "_next_interval",
        "_next_half",
    )

    def __init__(self, role: str) -> None:
        self.role = str(role).upper()
        self.rows: list[dict[str, Any]] = []
        self.half_rows: list[dict[str, Any]] = []
        self.call_count = 0
        self.completed_calls = 0
        self.aborted_calls = 0
        self._call_open = False
        self._segment_open = False
        self._segment_index = -1
        self._segment_start_wall = 0.0
        self._segment_start_cpu = 0.0
        self._heun_open = False
        self._heun_index = -1
        self._heun_start_wall = 0.0
        self._heun_start_cpu = 0.0
        self._next_interval = 0
        self._next_half = 0

    def reset(self) -> None:
        """Start a fresh call and discard prior per-call rows."""
        self.rows.clear()
        self.half_rows.clear()
        self._call_open = True
        self._segment_open = False
        self._segment_index = -1
        self._heun_open = False
        self._heun_index = -1
        self._next_interval = 0
        self._next_half = 0
        self.call_count += 1

    def begin_segment(self) -> None:
        if not self._call_open or self._segment_open:
            raise TimingBindingError("TIMING_SEGMENT_BEGIN_STATE")
        self._segment_open = True
        self._segment_index = self._next_interval
        self._next_interval += 1
        self._segment_start_wall = time.perf_counter()
        self._segment_start_cpu = time.process_time()

    def end_segment(self) -> None:
        if not self._segment_open or self._heun_open:
            raise TimingBindingError("TIMING_SEGMENT_END_STATE")
        wall = time.perf_counter() - self._segment_start_wall
        cpu = time.process_time() - self._segment_start_cpu
        self.rows.append(
            {
                "role": self.role,
                "scope": "controlled_segment",
                "interval_index": int(self._segment_index),
                "wall_s": float(wall),
                "process_cpu_s": float(cpu),
            }
        )
        self._segment_open = False
        self._segment_index = -1

    def begin_heun(self) -> None:
        if not self._call_open or not self._segment_open or self._heun_open:
            raise TimingBindingError("TIMING_HEUN_BEGIN_STATE")
        self._heun_open = True
        self._heun_index = self._next_half
        self._next_half += 1
        self._heun_start_wall = time.perf_counter()
        self._heun_start_cpu = time.process_time()

    def end_heun(self) -> None:
        if not self._heun_open:
            raise TimingBindingError("TIMING_HEUN_END_STATE")
        wall = time.perf_counter() - self._heun_start_wall
        cpu = time.process_time() - self._heun_start_cpu
        self.half_rows.append(
            {
                "role": self.role,
                "scope": "heun_core",
                "interval_index": int(self._segment_index),
                "half_index": int(self._heun_index),
                "wall_s": float(wall),
                "process_cpu_s": float(cpu),
            }
        )
        self._heun_open = False
        self._heun_index = -1

    def abort(self) -> None:
        """Retire an incomplete call without publishing formal rows."""
        if self._call_open or self._segment_open or self._heun_open:
            self.aborted_calls += 1
        self.rows.clear()
        self.half_rows.clear()
        self._call_open = False
        self._segment_open = False
        self._heun_open = False
        self._segment_index = -1
        self._heun_index = -1

    def finalize_call(self) -> None:
        if self._segment_open or self._heun_open:
            self.abort()
            return
        if self._call_open:
            self._call_open = False
            self.completed_calls += 1


def _adapter_identity() -> dict[str, Any]:
    path = Path(__file__).resolve(strict=True)
    raw = path.read_bytes()
    return {
        "path": str(path),
        "classification": "external",
        "hash_scope": "raw_bytes",
        "raw_sha256": _provenance_v2.raw_sha256(raw),
        "size_bytes": int(len(raw)),
    }


def _propagate_identity(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    canonical = _provenance_v2.canonical_lf_sha256(raw)
    if canonical != PROPAGATE_CANONICAL_LF_SHA256:
        raise TimingBindingError("PROPAGATE_CANONICAL_SOURCE_DRIFT")
    result = subprocess.run(
        ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise TimingBindingError("PROPAGATE_GIT_ROOT_UNAVAILABLE")
    root = Path(result.stdout.strip()).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise TimingBindingError("PROPAGATE_SOURCE_OUTSIDE_GIT_ROOT") from exc
    if relative != PROPAGATE_RELATIVE_PATH:
        raise TimingBindingError(f"PROPAGATE_RELATIVE_PATH_DRIFT:{relative}")
    record = {
        "path": relative,
        "classification": "tracked_text",
        "hash_scope": _provenance_v2.TRACKED_HASH_SCOPE,
        "git_blob_oid": PROPAGATE_SOURCE_BLOB,
        "canonical_lf_sha256": canonical,
    }
    try:
        _provenance_v2.validate_record(root, record, require_hash_scope=True)
    except Exception as exc:
        raise TimingBindingError("PROPAGATE_GIT_BLOB_DRIFT") from exc
    return {
        **record,
        "raw_sha256": _provenance_v2.raw_sha256(raw),
        "size_bytes": int(len(raw)),
    }


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise TimingBindingError(f"PROPAGATE_FUNCTION_ANCHOR:{name}:{len(matches)}")
    return matches[0]


def _call_name(node: ast.Call) -> str | None:
    return node.func.id if isinstance(node.func, ast.Name) else None


def _statement_lists(
    node: ast.AST,
    path: tuple[tuple[ast.AST, str], ...] = (),
):
    """Yield every real statement list and its ancestry.

    The caller must insert into one of these lists.  Walking ``ast.walk`` on a
    statement is deliberately avoided: an enclosing ``If``/``Try`` must not
    be mistaken for the exact assignment containing an anchor call.
    """
    for field_name, value in ast.iter_fields(node):
        edge = path + ((node, field_name),)
        if isinstance(value, list):
            if value and all(isinstance(item, ast.stmt) for item in value):
                yield value, node, field_name, edge
                for item in value:
                    yield from _statement_lists(item, edge)
            else:
                for item in value:
                    if isinstance(item, ast.AST):
                        yield from _statement_lists(item, edge)
        elif isinstance(value, ast.AST):
            yield from _statement_lists(value, edge)


def _assignment_call(
    statement: ast.stmt,
    line: int,
    names: frozenset[str],
) -> bool:
    """Match only the exact Assign statement carrying an anchored call."""
    if not isinstance(statement, ast.Assign):
        return False
    return any(
        isinstance(node, ast.Call)
        and int(getattr(node, "lineno", -1)) == int(line)
        and _call_name(node) in names
        for node in ast.walk(statement.value)
    )


def _matching_statements(
    root: ast.AST,
    predicate: Callable[[ast.stmt], bool],
):
    matches = []
    for statements, owner, field_name, path in _statement_lists(root):
        for index, statement in enumerate(statements):
            if predicate(statement):
                matches.append((statements, index, statement, owner, field_name, path))
    return matches


def _insert_relative(
    root: ast.AST,
    predicate: Callable[[ast.stmt], bool],
    new_node: ast.stmt,
    *,
    after: bool,
) -> bool:
    """Insert beside the exact matching statement in its owning body list."""
    matches = _matching_statements(root, predicate)
    if len(matches) != 1:
        return False
    statements, index, statement, _owner, _field_name, _path = matches[0]
    location = ast.copy_location(new_node, statement)
    statements.insert(index + (1 if after else 0), location)
    return True


def _is_timer_call(statement: ast.stmt, name: str | None = None) -> bool:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    if not isinstance(statement.value.func, ast.Name):
        return False
    return statement.value.func.id.startswith("_timing_") and (
        name is None or statement.value.func.id == name
    )


def _structure_without_timing(module: ast.Module) -> str:
    """Return a structural AST dump after removing only adapter nodes."""
    restored = copy.deepcopy(module)
    function_node = _function(restored, "propagate_one_pulse")
    body = list(function_node.body)
    for index in range(len(body) - 1):
        if not _is_timer_call(body[index], "_timing_call_begin"):
            continue
        wrapper = body[index + 1]
        if not isinstance(wrapper, ast.Try) or len(wrapper.finalbody) != 1:
            continue
        if not _is_timer_call(wrapper.finalbody[0], "_timing_call_finalize"):
            continue
        function_node.body = body[:index] + list(wrapper.body) + body[index + 2 :]
        break

    def strip(node: ast.AST) -> None:
        for field_name, value in ast.iter_fields(node):
            if isinstance(value, list):
                if value and all(isinstance(item, ast.stmt) for item in value):
                    kept = [item for item in value if not _is_timer_call(item)]
                    setattr(node, field_name, kept)
                    for item in kept:
                        strip(item)
                else:
                    for item in value:
                        if isinstance(item, ast.AST):
                            strip(item)
            elif isinstance(value, ast.AST):
                strip(value)

    strip(restored)
    return ast.dump(restored, annotate_fields=True, include_attributes=False)


def _validate_transformed_raman(function_node: ast.FunctionDef, nested_fn: ast.FunctionDef, role: str) -> None:
    role_name = str(role).upper()
    if role_name == "I0":
        stage_calls = _matching_statements(
            nested_fn,
            lambda stmt: _assignment_call(stmt, RAMAN_STAGE_LINE, _MONITORED_STAGE_NAMES),
        )
        apply_calls = _matching_statements(
            nested_fn,
            lambda stmt: _assignment_call(stmt, RAMAN_APPLY_LINE, _MONITORED_APPLY_RAMAN_NAMES),
        )
        if len(stage_calls) != 1 or len(apply_calls) != 1:
            raise TimingBindingError("RAMAN_CALLSITE_ANCHOR_DRIFT")
        return

    if role_name not in {"F", "G"}:
        raise TimingBindingError(f"UNKNOWN_TRANSFORMED_ROLE:{role_name}")
    branches = [
        node
        for node in nested_fn.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "full_isaacs_complete_mode"
    ]
    if len(branches) != 1 or not branches[0].orelse:
        raise TimingBindingError("RAMAN_ROTATIONAL_BRANCH_ANCHOR_DRIFT")
    branch = branches[0]
    all_replacements = [
        node
        for node in ast.walk(nested_fn)
        if isinstance(node, ast.Call) and _call_name(node) in _TRANSFORMED_RAMAN_SUBSTEP_NAMES
    ]
    rotational_replacements = [
        node
        for statement in branch.orelse
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and _call_name(node) in _TRANSFORMED_RAMAN_SUBSTEP_NAMES
    ]
    if len(all_replacements) != 1 or len(rotational_replacements) != 1:
        raise TimingBindingError("RAMAN_TRANSFORMED_SUBSTEP_ANCHOR_DRIFT")
    if any(
        isinstance(node, ast.Call)
        and (
            _call_name(node) in _MONITORED_STAGE_NAMES
            or _call_name(node) in _MONITORED_APPLY_RAMAN_NAMES
        )
        for statement in branch.orelse
        for node in ast.walk(statement)
    ):
        raise TimingBindingError("RAMAN_ROTATIONAL_ORIGINAL_CALL_RETAINED")


def _validate_timer_layout(module: ast.Module, role: str) -> dict[str, Any]:
    function_node = _function(module, "propagate_one_pulse")
    nested_fn = next(
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.FunctionDef) and node.name == "_full_raman_substep"
    )
    bk_names = _MONITORED_BK_NAMES
    bk = _matching_statements(
        function_node,
        lambda stmt: _assignment_call(stmt, BK_LINES[0], bk_names),
    )
    post = _matching_statements(
        function_node,
        lambda stmt: _assignment_call(stmt, POST_RAMAN_LINE, frozenset({"_full_raman_substep"})),
    )
    segment_begin = _matching_statements(
        function_node,
        lambda stmt: _is_timer_call(stmt, "_timing_segment_begin"),
    )
    segment_end = _matching_statements(
        function_node,
        lambda stmt: _is_timer_call(stmt, "_timing_segment_end"),
    )
    if len(bk) != 1 or len(post) != 1 or len(segment_begin) != 1 or len(segment_end) != 1:
        raise TimingBindingError("TIMING_NEIGHBOR_COUNT_DRIFT")
    bk_list, bk_index, bk_statement, _bk_owner, _bk_field, _bk_path = bk[0]
    post_list, post_index, post_statement, _post_owner, _post_field, _post_path = post[0]
    begin_list, begin_index, _begin_statement, _begin_owner, _begin_field, begin_path = segment_begin[0]
    end_list, end_index, _end_statement, _end_owner, _end_field, end_path = segment_end[0]
    if begin_list is not bk_list or begin_index != bk_index - 1:
        raise TimingBindingError("TIMING_SEGMENT_BEGIN_NOT_BEFORE_BK_ASSIGNMENT")
    if end_list is not post_list or end_index != post_index + 1:
        raise TimingBindingError("TIMING_SEGMENT_END_NOT_AFTER_POST_ASSIGNMENT")
    if int(getattr(bk_statement, "lineno", -1)) != BK_LINES[0] or int(
        getattr(post_statement, "lineno", -1)
    ) != POST_RAMAN_LINE:
        raise TimingBindingError("TIMING_NEIGHBOR_LINE_DRIFT")
    if not any(isinstance(owner, ast.Try) and field == "body" for owner, field in end_path):
        raise TimingBindingError("TIMING_SEGMENT_END_NOT_REACHABLE_IN_OUTER_TRY")

    if not nested_fn.body or not _is_timer_call(nested_fn.body[0], "_timing_heun_begin"):
        raise TimingBindingError("TIMING_HEUN_BEGIN_NOT_FIRST")
    nested_returns = [
        index for index, node in enumerate(nested_fn.body) if isinstance(node, ast.Return)
    ]
    heun_ends = [
        index
        for index, node in enumerate(nested_fn.body)
        if _is_timer_call(node, "_timing_heun_end")
    ]
    if len(nested_returns) != 1 or heun_ends != [nested_returns[0] - 1]:
        raise TimingBindingError("TIMING_HEUN_END_NOT_BEFORE_RETURN")

    final_returns = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.Return) and int(getattr(node, "lineno", -1)) == FINAL_RETURN_LINE
    ]
    if len(final_returns) != 1:
        raise TimingBindingError("PROPAGATE_FINAL_RETURN_ANCHOR_DRIFT")
    return {
        "segment_begin_neighbor": "Assign@649",
        "segment_end_neighbor": "Assign@969",
        "segment_end_outer_try_body_reachable": True,
        "heun_begin": "nested_body_first_statement",
        "heun_end": "nested_return_predecessor",
        "role": str(role).upper(),
    }


def _source_raman_anchors(tree: ast.Module) -> tuple[ast.FunctionDef, ast.FunctionDef]:
    function_node = _function(tree, "propagate_one_pulse")
    nested = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.FunctionDef) and node.name == "_full_raman_substep"
    ]
    if len(nested) != 1:
        raise TimingBindingError("RAMAN_SUBSTEP_ANCHOR_NOT_UNIQUE")
    nested_fn = nested[0]
    stage = _matching_statements(
        nested_fn,
        lambda stmt: _assignment_call(stmt, RAMAN_STAGE_LINE, _MONITORED_STAGE_NAMES),
    )
    apply = _matching_statements(
        nested_fn,
        lambda stmt: _assignment_call(stmt, RAMAN_APPLY_LINE, _MONITORED_APPLY_RAMAN_NAMES),
    )
    if len(stage) != 1 or len(apply) != 1:
        raise TimingBindingError("RAMAN_CALLSITE_ANCHOR_DRIFT")
    return function_node, nested_fn


def _validate_original_source(source_path: Path) -> None:
    source_text = _provenance_v2.canonical_lf_bytes(source_path.read_bytes()).decode("utf-8")
    tree = ast.parse(source_text, filename=str(source_path))
    function_node, nested_fn = _source_raman_anchors(tree)
    bk = [
        _matching_statements(
            function_node,
            lambda stmt, line=line: _assignment_call(stmt, line, frozenset({"step_linear_bk_nee_factorized"})),
        )
        for line in BK_LINES
    ]
    nonlinear = _matching_statements(
        function_node,
        lambda stmt: _assignment_call(stmt, APPLY_LINE, frozenset({"apply_nonlinear"})),
    )
    post = _matching_statements(
        function_node,
        lambda stmt: _assignment_call(stmt, POST_RAMAN_LINE, frozenset({"_full_raman_substep"})),
    )
    final = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.Return) and int(getattr(node, "lineno", -1)) == FINAL_RETURN_LINE
    ]
    if any(len(items) != 1 for items in bk) or len(nonlinear) != 1 or len(post) != 1:
        raise TimingBindingError("ORIGINAL_PROPAGATE_ANCHOR_DRIFT")
    if len(final) != 1 or len([node for node in ast.walk(nested_fn) if isinstance(node, ast.Return)]) != 1:
        raise TimingBindingError("ORIGINAL_RETURN_ANCHOR_DRIFT")


def _call_statement(name: str) -> ast.Expr:
    return ast.Expr(
        value=ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=[], keywords=[])
    )


def _inject_timing(module: ast.Module, *, role: str) -> dict[str, Any]:
    before_structure = _structure_without_timing(module)
    function_node = _function(module, "propagate_one_pulse")
    nested = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.FunctionDef) and node.name == "_full_raman_substep"
    ]
    if len(nested) != 1:
        raise TimingBindingError("RAMAN_SUBSTEP_ANCHOR_NOT_UNIQUE")
    nested_fn = nested[0]

    bk_assignments = [
        _matching_statements(
            function_node,
            lambda stmt, line=line: _assignment_call(stmt, line, _MONITORED_BK_NAMES),
        )
        for line in BK_LINES
    ]
    if any(len(matches) != 1 for matches in bk_assignments):
        raise TimingBindingError("BK_CALLSITE_ANCHOR_DRIFT")
    nonlinear_assignments = _matching_statements(
        function_node,
        lambda stmt: _assignment_call(stmt, APPLY_LINE, _MONITORED_APPLY_NAMES),
    )
    if len(nonlinear_assignments) != 1:
        raise TimingBindingError("NONLINEAR_CALLSITE_ANCHOR_DRIFT")
    post_assignments = _matching_statements(
        function_node,
        lambda stmt: _assignment_call(stmt, POST_RAMAN_LINE, frozenset({"_full_raman_substep"})),
    )
    if len(post_assignments) != 1:
        raise TimingBindingError("POST_RAMAN_ASSIGNMENT_ANCHOR_DRIFT")
    final_returns = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.Return) and int(getattr(node, "lineno", -1)) == FINAL_RETURN_LINE
    ]
    if len(final_returns) != 1:
        raise TimingBindingError("PROPAGATE_FINAL_RETURN_ANCHOR_DRIFT")
    nested_returns = [node for node in ast.walk(nested_fn) if isinstance(node, ast.Return)]
    if len(nested_returns) != 1:
        raise TimingBindingError("RAMAN_SUBSTEP_RETURN_ANCHOR_DRIFT")

    _validate_transformed_raman(function_node, nested_fn, role)

    if not _insert_relative(
        function_node,
        lambda stmt: _assignment_call(stmt, BK_LINES[0], _MONITORED_BK_NAMES),
        _call_statement("_timing_segment_begin"),
        after=False,
    ):
        raise TimingBindingError("TIMING_SEGMENT_BEGIN_INSERT_FAILED")
    if not _insert_relative(
        function_node,
        lambda stmt: _assignment_call(stmt, POST_RAMAN_LINE, frozenset({"_full_raman_substep"})),
        _call_statement("_timing_segment_end"),
        after=True,
    ):
        raise TimingBindingError("TIMING_SEGMENT_END_INSERT_FAILED")

    # The nested timer begins before stage_kwargs construction and ends after
    # the original updated/diagnostics math, immediately before its return.
    nested_fn.body.insert(0, _call_statement("_timing_heun_begin"))
    return_index = next(index for index, node in enumerate(nested_fn.body) if isinstance(node, ast.Return))
    nested_fn.body.insert(return_index, _call_statement("_timing_heun_end"))

    # Reset/finalize are scalar lifecycle hooks around the source body.  The
    # source's mathematical order and all anchored statements remain intact.
    entry_index = 1 if (
        function_node.body
        and isinstance(function_node.body[0], ast.Expr)
        and isinstance(function_node.body[0].value, ast.Constant)
        and isinstance(function_node.body[0].value.value, str)
    ) else 0
    body = list(function_node.body[entry_index:])
    call_try = ast.Try(
        body=body,
        handlers=[],
        orelse=[],
        finalbody=[_call_statement("_timing_call_finalize")],
    )
    function_node.body = (
        list(function_node.body[:entry_index])
        + [_call_statement("_timing_call_begin"), call_try]
    )
    ast.fix_missing_locations(module)
    layout = _validate_timer_layout(module, role)
    after_structure = _structure_without_timing(module)
    if before_structure != after_structure:
        raise TimingBindingError("TIMING_AST_STRUCTURE_CHANGED_AFTER_STRIP")
    return {
        "schema": SCHEMA + ".static",
        "source_structure_equal_after_timer_strip": True,
        "timer_layout": layout,
        "transformed_raman_role": str(role).upper(),
    }


def _compile_interceptor(role: str) -> Callable[..., Any]:
    injected = False

    def compile_interceptor(source: Any, filename: Any, mode: str, *args: Any, **kwargs: Any) -> Any:
        nonlocal injected
        if not injected and isinstance(source, ast.Module):
            compile_interceptor.timing_receipt = _inject_timing(source, role=role)
            injected = True
        return builtins.compile(source, filename, mode, *args, **kwargs)

    compile_interceptor.timing_receipt = None
    return compile_interceptor


def _clock_helpers(clock: WindowClock) -> dict[str, Callable[[], None]]:
    def call_begin() -> None:
        clock.reset()

    def segment_begin() -> None:
        clock.begin_segment()

    def segment_end() -> None:
        clock.end_segment()

    def heun_begin() -> None:
        clock.begin_heun()

    def heun_end() -> None:
        clock.end_heun()

    def call_finalize() -> None:
        clock.finalize_call()

    return {
        "_timing_call_begin": call_begin,
        "_timing_segment_begin": segment_begin,
        "_timing_segment_end": segment_end,
        "_timing_heun_begin": heun_begin,
        "_timing_heun_end": heun_end,
        "_timing_call_finalize": call_finalize,
    }


def _copy_function(function: Callable[..., Any], globals_dict: dict[str, Any]) -> Callable[..., Any]:
    clone = FunctionType(
        function.__code__,
        globals_dict,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    clone.__kwdefaults__ = getattr(function, "__kwdefaults__", None)
    clone.__annotations__ = getattr(function, "__annotations__", {})
    clone.__module__ = getattr(function, "__module__", None)
    clone.__qualname__ = getattr(function, "__qualname__", function.__name__)
    clone.__doc__ = getattr(function, "__doc__", None)
    return clone


def _copy_metadata(source: Callable[..., Any], clone: Callable[..., Any], role: str) -> None:
    clone.__dict__.update(dict(getattr(source, "__dict__", {})))
    clone.__j1_inner_code__ = clone.__code__
    clone.__timing_original_code__ = getattr(source, "__code__", None)
    clone.__timing_original_j1_inner_code__ = getattr(
        source, "__j1_inner_code__", getattr(source, "__code__", None)
    )
    clone.__timing_role__ = str(role).upper()
    clone.__timing_actual_code__ = clone.__code__


def _clone_original(
    source_path: Path,
    source: Callable[..., Any],
    namespace: dict[str, Any],
    helpers: Mapping[str, Callable[[], None]],
) -> tuple[Callable[..., Any], dict[str, Any]]:
    source_text = _provenance_v2.canonical_lf_bytes(source_path.read_bytes()).decode("utf-8")
    tree = ast.parse(source_text, filename=str(source_path))
    module = ast.Module(body=[copy.deepcopy(_function(tree, "propagate_one_pulse"))], type_ignores=[])
    ast.fix_missing_locations(module)
    private = dict(namespace)
    private.update(helpers)
    interceptor = _compile_interceptor("I0")
    exec(interceptor(module, str(source_path), "exec"), private, local_namespace := {})
    clone = local_namespace.get("propagate_one_pulse")
    if not callable(clone):
        raise TimingBindingError("I0_TIMED_CLONE_MISSING")
    receipt = getattr(interceptor, "timing_receipt", None)
    if not isinstance(receipt, dict):
        raise TimingBindingError("I0_TIMING_STATIC_RECEIPT_MISSING")
    return clone, receipt


def _handoff_module(dispatch: Any, factory: Any | None) -> ModuleType:
    state = getattr(dispatch, "cpu_s4_3e_state", None)
    if state is None:
        state = getattr(dispatch, "cpu_s4_3f_state", None)
    module_name = getattr(getattr(state, "__class__", None), "__module__", None)
    module = sys.modules.get(str(module_name)) if module_name else None
    if not isinstance(module, ModuleType) or not callable(getattr(module, "_transform_propagator_function", None)):
        if factory is not None:
            inherited = getattr(factory, "inherited_s43e_factory", None)
            candidate = getattr(inherited, "__globals__", {}).get("_transform_propagator_function")
            if callable(candidate):
                module = sys.modules.get(getattr(candidate, "__module__", ""))
    if not isinstance(module, ModuleType):
        raise TimingBindingError("F_HANDOFF_TRANSFORMER_MODULE_UNAVAILABLE")
    return module


def _clone_f(
    source_path: Path,
    source: Callable[..., Any],
    dispatch: Any,
    factory: Any | None,
    helpers: Mapping[str, Callable[[], None]],
) -> tuple[Callable[..., Any], dict[str, Any]]:
    module = _handoff_module(dispatch, factory)
    transformer = getattr(module, "_transform_propagator_function", None)
    if not callable(transformer):
        raise TimingBindingError("F_HANDOFF_TRANSFORMER_UNAVAILABLE")
    transformer_globals = dict(getattr(transformer, "__globals__", {}))
    interceptor = _compile_interceptor("F")
    transformer_globals["compile"] = interceptor
    transformed_builder = _copy_function(transformer, transformer_globals)
    transformed = transformed_builder(source_path)
    if not callable(transformed):
        raise TimingBindingError("F_TRANSFORMER_DID_NOT_RETURN_FUNCTION")
    private = dict(getattr(source, "__globals__", {}))
    private.update(helpers)
    clone = _copy_function(transformed, private)
    receipt = getattr(interceptor, "timing_receipt", None)
    if not isinstance(receipt, dict):
        raise TimingBindingError("F_TIMING_STATIC_RECEIPT_MISSING")
    return clone, receipt


def _clone_g(
    source_path: Path,
    source: Callable[..., Any],
    helpers: Mapping[str, Callable[[], None]],
) -> tuple[Callable[..., Any], dict[str, Any]]:
    module = importlib.import_module("tools.cpu_s4_3g.controlled_callsite")
    builder = getattr(module, "_clone_function", None)
    if not callable(builder):
        raise TimingBindingError("G_CLONE_BUILDER_UNAVAILABLE")
    builder_globals = dict(getattr(builder, "__globals__", {}))
    interceptor = _compile_interceptor("G")
    builder_globals["compile"] = interceptor
    builder_copy = _copy_function(builder, builder_globals)
    private = dict(getattr(source, "__globals__", {}))
    private.update(helpers)
    clone, _identity = builder_copy(source_path, private)
    if not callable(clone):
        raise TimingBindingError("G_TIMED_CLONE_MISSING")
    receipt = getattr(interceptor, "timing_receipt", None)
    if not isinstance(receipt, dict):
        raise TimingBindingError("G_TIMING_STATIC_RECEIPT_MISSING")
    return clone, receipt


def instrument(
    prop: Callable[..., Any],
    dispatch: Any,
    role: str,
    *,
    factory: Any | None = None,
) -> tuple[Callable[..., Any], WindowClock, dict[str, Any]]:
    """Return ``(timed_propagator, clock, binding)`` for role I0/F/G.

    ``factory`` is optional and only helps discover the inherited S4-3E
    transformer when an F dispatch does not retain its private handoff module.
    It is never mutated.
    """
    role_name = str(role).upper()
    if role_name not in {"I0", "F", "G"}:
        raise TimingBindingError(f"UNKNOWN_TIMING_ROLE:{role}")
    if not callable(prop):
        raise TimingBindingError("PROPAGATOR_NOT_CALLABLE")
    code = getattr(prop, "__code__", None)
    if code is None:
        raise TimingBindingError("PROPAGATOR_CODE_MISSING")
    source_path = Path(code.co_filename).resolve(strict=True)
    propagate_identity = _propagate_identity(source_path)
    _validate_original_source(source_path)
    clock = WindowClock(role_name)
    helpers = _clock_helpers(clock)

    if role_name == "G":
        state = getattr(dispatch, "cpu_s4_3g_state", None)
        if state is None or getattr(prop, "_cpu_s4_3g_state", state) is not state:
            raise TimingBindingError("G_DISPATCH_STATE_MISMATCH")
        timed, static_receipt = _clone_g(source_path, prop, helpers)
    elif role_name == "F":
        state = getattr(dispatch, "cpu_s4_3e_state", None) or getattr(dispatch, "cpu_s4_3f_state", None)
        if state is None:
            raise TimingBindingError("F_DISPATCH_STATE_MISSING")
        timed, static_receipt = _clone_f(source_path, prop, dispatch, factory, helpers)
    else:
        namespace = getattr(prop, "__globals__", None)
        if not isinstance(namespace, dict):
            raise TimingBindingError("I0_PROPAGATOR_GLOBALS_MISSING")
        timed, static_receipt = _clone_original(source_path, prop, namespace, helpers)

    _copy_metadata(prop, timed, role_name)
    binding = {
        "schema": SCHEMA,
        "role": role_name,
        "adapter": _adapter_identity(),
        "propagate": propagate_identity,
        "anchors": {
            "bk": list(BK_LINES),
            "apply_nonlinear": APPLY_LINE,
            "raman_stage": RAMAN_STAGE_LINE,
            "raman_apply": RAMAN_APPLY_LINE,
            "post_raman_assignment": POST_RAMAN_LINE,
            "final_return": FINAL_RETURN_LINE,
        },
        "clock": {
            "segment": "first_BK_assignment_before_call_to_post_full_raman_assignment_before_append",
            "heun_core": "nested_full_raman_substep_first_statement_to_return_before_math_reordering",
            "failure": "abort_without_formal_row",
        },
        "private_globals_copy": True,
        "module_or_cache_mutated": False,
        "actual_code_filename": str(timed.__code__.co_filename),
        "actual_code_firstlineno": int(timed.__code__.co_firstlineno),
        "j1_inner_code_is_actual": getattr(timed, "__j1_inner_code__", None) is timed.__code__,
        "original_code_firstlineno": int(getattr(code, "co_firstlineno", -1)),
        "static_receipt": static_receipt,
    }
    return timed, clock, binding


__all__ = ["SCHEMA", "TimingBindingError", "WindowClock", "instrument"]
