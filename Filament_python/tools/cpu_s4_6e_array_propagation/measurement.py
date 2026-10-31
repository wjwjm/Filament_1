"""Original SG application clock with a source-bound pre-output pool join."""

from __future__ import annotations

import ast
import copy
import inspect
import json
import marshal
import textwrap
import time
from types import FunctionType

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_5a_l1_residual import measurement as parent
from tools.hpc_ops import provenance_v2 as pv


class _Add(parent._Add):
    def __init__(self):
        super().__init__()
        self.close_anchors = 0

    def visit_Assign(self, node):
        result = super().visit_Assign(node)
        if ast.unparse(node.targets[0]) == "ordinary":
            self.close_anchors += 1
            close = ast.Expr(value=ast.Call(func=ast.Name(id="_s46e_close_before_output", ctx=ast.Load()), args=[], keywords=[]))
            return [ast.copy_location(close, node), result]
        return result


class _Strip(parent._Strip):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "_s46e_close_before_output":
            return None
        return super().visit_Expr(node)


def run_window(args, built, *, timer, application_token, **kwargs):
    original = sg._run_window
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    before = parent._dump(tree)
    adder = _Add()
    transformed = ast.fix_missing_locations(adder.visit(copy.deepcopy(tree)))
    parent.require(adder.close_anchors == 1 and adder.counts == dict(INPUT_PREPARATION=1, PROPAGATION=1, DIAGNOSTICS=1, OUTPUT=1), "S46E_EXACT_APPLICATION_ANCHORS")
    parent.require(parent._dump(_Strip().visit(copy.deepcopy(transformed))) == before, "S46E_REMOVING_TIMER_AND_CLOSE_RESTORES_HARNESS_AST")
    c_pool = getattr(built[1], "cpu_s4_6e_c_pool", None)
    is_c_candidate = c_pool is not None
    close = built[1].close_c_pool if is_c_candidate else built[1].close_intensity_pool
    lifecycle = {}
    finalized = {}

    def close_before_output():
        lifecycle["close_started_s"] = time.perf_counter()
        close()
        if is_c_candidate:
            activity = built[1].take_c_pool_records()
            snapshot = built[1].c_pool_snapshot()
            from .c_pool import POINTWISE_LABELS

            count = int(args.intervals) if args.reported_role == "C" else 0
            expected_activity = 2 * count
            take_again = built[1].take_c_pool_records
        else:
            activity = built[1].take_intensity_pool_records()
            snapshot = built[1].intensity_pool_snapshot()
            from .pool import POINTWISE_LABELS

            count = 4 * int(args.intervals) if args.reported_role == "A" else 0
            expected_activity = 3 * count
            take_again = built[1].take_intensity_pool_records
        workers = 1 if args.local_engineering else 24
        parent.require(snapshot["closed"] and snapshot["close_error"] is None, "S46E_POOL_JOIN_BEFORE_NORMAL_OUTPUT")
        parent.require(snapshot["pending_scalar_records"] == 0 and snapshot["active_thread_ids"] == [], "S46E_POOL_SNAPSHOT_AFTER_DRAIN_AND_JOIN")
        parent.require(snapshot["operation_counts"] == {label: count for label in POINTWISE_LABELS}, "S46E_POOL_COUNTS_BEFORE_NORMAL_OUTPUT")
        parent.require(len(activity) == expected_activity, "S46E_POOL_ACTIVITY_COUNT_BEFORE_NORMAL_OUTPUT")
        parent.require(all(row["requested_workers"] == row["actual_workers"] == workers and len(row["workers"]) == workers and sum(worker["items"] for worker in row["workers"]) == 384 for row in activity), "S46E_POOL_WORKERS_BEFORE_NORMAL_OUTPUT")
        parent.require(len(json.dumps(activity, allow_nan=False).encode()) <= 8 * 1024**2, "S46E_POOL_ACTIVITY_BOUNDED_BEFORE_NORMAL_OUTPUT")
        parent.require(not take_again(), "S46E_POOL_ACTIVITY_DRAINED_BEFORE_NORMAL_OUTPUT")
        finalized.update(pool=snapshot, activity=activity)
        built[1]._cpu_s46e_finalized_pool = dict(finalized)
        if is_c_candidate:
            built[1]._cpu_s46e_finalized_c_pool = dict(finalized)
        lifecycle["join_completed_s"] = time.perf_counter()

    namespace = dict(
        original.__globals__,
        _s45a_timer=timer,
        _s45a_application_token=application_token,
        _s46e_close_before_output=close_before_output,
    )
    for name in ("context", "fingerprints"):
        frozen = namespace[name]

        def forward(*values, _frozen=frozen, **named):
            with timer.scope("INPUT_PREPARATION"):
                return _frozen(*values, **named)

        namespace[name] = forward
    local = {}
    exec(compile(transformed, original.__code__.co_filename, "exec"), namespace, local)
    inner = local["_run_window"]
    namespace["_run_window"] = inner
    wrapped = FunctionType(sg.run_window.__code__, namespace, sg.run_window.__name__, sg.run_window.__defaults__, sg.run_window.__closure__)
    wrapped.__kwdefaults__ = sg.run_window.__kwdefaults__
    try:
        result = wrapped(args, built, **kwargs)
    except BaseException as error:
        try:
            close()
        except BaseException as cleanup:
            error.add_note("S46E_CLOSE_FAILURE:" + type(cleanup).__name__)
        raise
    end = kwargs["application_started"] + result["optical_application_window_s"]
    parent.require(lifecycle["join_completed_s"] < end, "S46E_CLOSE_INSIDE_ORIGINAL_APPLICATION_CLOCK")
    binding = dict(
        schema="filament.cpu_s4_6e.harness.v1",
        source_file=original.__code__.co_filename,
        source_canonical_lf_sha256=pv.canonical_lf_sha256(source.encode()),
        original_ast_sha256=pv.raw_sha256(before.encode()),
        actual_code_sha256=pv.raw_sha256(marshal.dumps(inner.__code__)),
        stripped_AST_exact=True,
        close_anchor="before original ordinary=writer(...)",
        lifecycle=lifecycle,
        pool_finalized_before_output=True,
        original_application_finished_s=end,
        scientific_expression_changes=0,
        propagate_AST_rebuilt=False,
    )
    sg.write(args.work / "MEASUREMENT_HARNESS_BINDING.json", binding)
    return result
