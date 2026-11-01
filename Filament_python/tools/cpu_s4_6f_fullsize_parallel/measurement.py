"""SG application clock with semantic pool finalization before normal output."""

from __future__ import annotations

import ast
import copy
import inspect
import json
import marshal
import textwrap
import time
from types import FunctionType
from typing import Any

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_5a_l1_residual import measurement as parent
from tools.hpc_ops import provenance_v2 as pv

from .protocol import candidate_spec
from .role_spec import close_role, pool_contract, snapshot_role, take_role_records

POINTWISE_LABELS = {
    "EQ10_P48": ("B_SPECTRAL_MULTIPLY", "B_DERIVATIVE_DIVIDE", "B_RESPONSE_DIVIDE", "B_PRODUCT"),
    "INTENSITY_P24": ("A_ABS", "A_SQUARE", "A_SCALE"),
    "PHASE_ABSORB_P24": ("C_PHASE", "C_ABSORPTION"),
}


class _CompletedTimer:
    """Keep the original propagation shell in the common OFF clock."""

    def __init__(self, timer):
        self._timer = timer

    def scope(self, label, *values, **named):
        if label == "PROPAGATION":
            named["common"] = True
        return self._timer.scope(label, *values, **named)

    def end(self, token):
        return self._timer.end(token)


class _Add(parent._Add):
    def __init__(self):
        super().__init__()
        self.close_anchors = 0

    def visit_Assign(self, node):
        result = super().visit_Assign(node)
        if ast.unparse(node.targets[0]) == "ordinary":
            self.close_anchors += 1
            close = ast.Expr(value=ast.Call(func=ast.Name(id="_s46f_close_before_output", ctx=ast.Load()), args=[], keywords=[]))
            return [ast.copy_location(close, node), result]
        return result


class _Strip(parent._Strip):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "_s46f_close_before_output":
            return None
        return super().visit_Expr(node)


def _validate_pool(semantic: str, role: str, intervals: int, snapshot: dict[str, Any], activity: list[dict[str, Any]], *, local_engineering: bool) -> None:
    spec = candidate_spec(semantic)
    expected_count = 0 if role == "R" else (int(intervals) if spec["pointwise_operation"] == "PHASE_ABSORB" else 4 * int(intervals))
    workers = 1 if local_engineering else int(spec["workers"])
    expected_ops = {label: expected_count for label in POINTWISE_LABELS[str(semantic).upper()]}
    if not snapshot.get("closed") or snapshot.get("close_error") is not None:
        raise RuntimeError("S46F_POOL_JOIN_BEFORE_NORMAL_OUTPUT")
    if snapshot.get("pending_scalar_records") != 0 or snapshot.get("active_thread_ids") != []:
        raise RuntimeError("S46F_POOL_SNAPSHOT_AFTER_DRAIN_AND_JOIN")
    if snapshot.get("operation_counts") != expected_ops:
        raise RuntimeError("S46F_POOL_COUNTS_BEFORE_NORMAL_OUTPUT")
    expected_activity = int(spec["pointwise_labels"]) * expected_count
    if len(activity) != expected_activity:
        raise RuntimeError("S46F_POOL_ACTIVITY_COUNT_BEFORE_NORMAL_OUTPUT")
    if len(json.dumps(activity, allow_nan=False).encode("utf-8")) > 8 * 1024**2:
        raise RuntimeError("S46F_POOL_ACTIVITY_BOUNDED_BEFORE_NORMAL_OUTPUT")
    for row in activity:
        if row.get("requested_workers") != workers or row.get("actual_workers") != workers or len(row.get("workers", ())) != workers:
            raise RuntimeError("S46F_POOL_WORKER_COUNT_BEFORE_NORMAL_OUTPUT")
        if sum(int(worker.get("items", 0)) for worker in row.get("workers", ())) != 384:
            raise RuntimeError("S46F_POOL_PARTITION_BEFORE_NORMAL_OUTPUT")


def run_window(args, built, *, timer, application_token, window=None, **kwargs):
    """Execute the frozen SG window with one private close-before-output hook."""

    target = sg.run_window if window is None else window
    target_globals = target.__globals__
    original = target_globals.get("_run_window", sg._run_window)
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    before = parent._dump(tree)
    adder = _Add()
    transformed = ast.fix_missing_locations(adder.visit(copy.deepcopy(tree)))
    parent.require(adder.close_anchors == 1 and adder.counts == dict(INPUT_PREPARATION=1, PROPAGATION=1, DIAGNOSTICS=1, OUTPUT=1), "S46F_EXACT_APPLICATION_ANCHORS")
    parent.require(parent._dump(_Strip().visit(copy.deepcopy(transformed))) == before, "S46F_REMOVING_CLOSE_RESTORES_HARNESS_AST")

    semantic = str(args.semantic_candidate).upper()
    role = str(args.reported_role).upper()
    lifecycle: dict[str, float] = {}
    finalized: dict[str, Any] = {}

    def close_before_output():
        lifecycle["close_started_s"] = time.perf_counter()
        close_role(built[1], semantic)
        activity = take_role_records(built[1], semantic)
        snapshot = snapshot_role(built[1], semantic)
        _validate_pool(semantic, role, int(args.intervals), snapshot, activity, local_engineering=bool(args.local_engineering))
        # A second drain is a scalar-only contract check and proves no worker
        # activity remains queued before the original writer starts.
        if take_role_records(built[1], semantic):
            raise RuntimeError("S46F_POOL_ACTIVITY_DRAINED_ONCE")
        finalized.update(pool=snapshot, activity=activity)
        built[1]._cpu_s4_6f_finalized_pool = dict(finalized)
        lifecycle["join_completed_s"] = time.perf_counter()

    clock_timer = _CompletedTimer(timer)
    namespace = dict(target_globals)
    namespace.update(_s45a_timer=clock_timer, _s45a_application_token=application_token, _s46f_close_before_output=close_before_output)
    for name in ("context", "fingerprints"):
        if name in namespace:
            frozen = namespace[name]

            def forward(*values, _frozen=frozen, **named):
                with clock_timer.scope("INPUT_PREPARATION"):
                    return _frozen(*values, **named)

            namespace[name] = forward
    local: dict[str, Any] = {}
    exec(compile(transformed, original.__code__.co_filename, "exec"), namespace, local)
    inner = local["_run_window"]
    namespace["_run_window"] = inner
    wrapped = FunctionType(target.__code__, namespace, target.__name__, target.__defaults__, target.__closure__)
    wrapped.__kwdefaults__ = target.__kwdefaults__
    try:
        result = wrapped(args, built, **kwargs)
    except BaseException as error:
        try:
            close_role(built[1], semantic)
        except BaseException as cleanup:
            error.add_note("S46F_CLOSE_FAILURE:" + type(cleanup).__name__)
        raise
    end = kwargs["application_started"] + result["optical_application_window_s"]
    if lifecycle.get("join_completed_s", float("inf")) >= end:
        raise RuntimeError("S46F_CLOSE_OUTSIDE_ORIGINAL_APPLICATION_CLOCK")
    binding = dict(
        schema="filament.cpu_s4_6f.harness.v1",
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
        pool_contract=pool_contract(semantic),
        completed_propagation_common=True,
    )
    sg.write(args.work / "MEASUREMENT_HARNESS_BINDING.json", binding)
    return result


__all__ = ["run_window"]
