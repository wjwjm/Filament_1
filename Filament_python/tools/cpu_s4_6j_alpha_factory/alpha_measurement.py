"""Original application clock with separate four-pool and alpha validation."""

from __future__ import annotations

import json
from types import FunctionType
from typing import Any

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_6f_fullsize_parallel import measurement as parent
from tools.cpu_s4_6i_residual_parallel import selected_measurement as selected
from tools.cpu_s4_6i_residual_parallel.selected_protocol import CANDIDATE as POWER

from .alpha_pool import ALPHA_LABELS
from .alpha_role_spec import (
    close_role,
    pool_contract,
    snapshot_role,
    take_role_records,
)
from .alpha_protocol import CANDIDATE, COMPONENTS, candidate_spec

KEYS = selected.KEYS


def _scalar_tree(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return True
    if isinstance(value, dict):
        return all(_scalar_tree(key) and _scalar_tree(item) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return all(_scalar_tree(item) for item in value)
    return False


def _validate_alpha(
    role: str,
    intervals: int,
    snapshot: dict[str, Any],
    activity: list[dict[str, Any]],
    *,
    local_engineering: bool,
) -> dict[str, Any]:
    alpha_rows = [row for row in activity if row.get("label") in ALPHA_LABELS]
    expected_workers = 1 if local_engineering else 24
    alpha_snapshot = snapshot.get("alpha")
    if role == "R":
        if alpha_rows or alpha_snapshot is not None:
            raise RuntimeError("S46J_REFERENCE_ALPHA_ACTIVITY_PRESENT")
        return {
            "enabled": False,
            "calls": 0,
            "rounds": 0,
            "records": 0,
            "fallback": 0,
        }
    if not isinstance(alpha_snapshot, dict):
        raise RuntimeError("S46J_ALPHA_POOL_SNAPSHOT_MISSING")
    if (
        not alpha_snapshot.get("closed")
        or alpha_snapshot.get("close_error") is not None
        or alpha_snapshot.get("active_thread_ids")
        or alpha_snapshot.get("pending_scalar_records") != 0
        or alpha_snapshot.get("dropped_records", 0)
        or alpha_snapshot.get("invalid_records", 0)
        or alpha_snapshot.get("array_values_retained")
    ):
        raise RuntimeError("S46J_ALPHA_POOL_LIFECYCLE_OR_RECORD_LOSS")
    counts = alpha_snapshot.get("operation_counts")
    expected_counts = {label: int(intervals) for label in ALPHA_LABELS}
    if counts != expected_counts:
        raise RuntimeError("S46J_ALPHA_OPERATION_COUNTS")
    if alpha_snapshot.get("completed_operation_counts") != expected_counts or alpha_snapshot.get("attempted_operation_counts") != expected_counts or any(alpha_snapshot.get("record_failure_counts", {}).values()):
        raise RuntimeError("S46J_ALPHA_OPERATION_COMPLETION")
    expected_order = [
        (interval, label)
        for interval in range(int(intervals))
        for label in ALPHA_LABELS
    ]
    actual_order = [
        (row.get("interval_index"), row.get("label")) for row in alpha_rows
    ]
    if actual_order != expected_order:
        raise RuntimeError("S46J_ALPHA_ACTIVITY_ORDER")
    if len(alpha_rows) != 5 * int(intervals):
        raise RuntimeError("S46J_ALPHA_ACTIVITY_COUNT")
    for row in alpha_rows:
        if not _scalar_tree(row):
            raise RuntimeError("S46J_ALPHA_SCIENCE_ARRAY_ACTIVITY")
        if (
            row.get("requested_workers") != expected_workers
            or row.get("actual_workers") != expected_workers
            or len(row.get("workers", ())) != expected_workers
        ):
            raise RuntimeError("S46J_ALPHA_WORKER_COUNT")
        if sum(int(worker.get("items", 0)) for worker in row.get("workers", ())) != 384:
            raise RuntimeError("S46J_ALPHA_PARTITION")
    return {
        "enabled": True,
        "calls": int(intervals),
        "rounds": 5 * int(intervals),
        "records": len(alpha_rows),
        "workers": expected_workers,
        "fallback": 0,
        "labels": list(ALPHA_LABELS),
    }


def _validate_pools(
    semantic: str,
    role: str,
    intervals: int,
    snapshot: dict[str, Any],
    activity: list[dict[str, Any]],
    *,
    local_engineering: bool,
) -> None:
    candidate_spec(semantic)
    base_activity = [row for row in activity if row.get("label") not in ALPHA_LABELS]
    alpha_receipt = _validate_alpha(
        str(role).upper(), intervals, snapshot, activity,
        local_engineering=local_engineering,
    )
    base_snapshot = dict(snapshot)
    base_snapshot.pop("alpha", None)
    selected.validate_pools(
        POWER,
        "X",  # alpha_factory always keeps the four-pool internal role X
        intervals,
        base_snapshot,
        base_activity,
        local_engineering=local_engineering,
    )
    expected_base = 38 * int(intervals)
    expected_alpha = 0 if str(role).upper() == "R" else 5 * int(intervals)
    if len(base_activity) != expected_base or len(activity) != expected_base + expected_alpha:
        raise RuntimeError("S46J_ALPHA_TOTAL_ACTIVITY_COUNT")
    if snapshot.get("all_pools_retired_before_output") is not True:
        raise RuntimeError("S46J_ALL_POOLS_NOT_RETIRED_BEFORE_OUTPUT")
    if not _scalar_tree(alpha_receipt):
        raise RuntimeError("S46J_ALPHA_RECEIPT_NONSCALAR")


def _closed(dispatch: Any, semantic: str) -> dict[str, Any]:
    return close_role(dispatch, semantic)


def _snapshot(dispatch: Any, semantic: str) -> dict[str, Any]:
    return snapshot_role(dispatch, semantic)


def _take(dispatch: Any, semantic: str) -> list[dict[str, Any]]:
    if getattr(dispatch, "_s46j_measurement_records_taken", False):
        return []
    result = take_role_records(dispatch, semantic)
    dispatch._s46j_measurement_records_taken = True
    return result


def run_window(args: Any, built: Any, **kwargs: Any) -> dict[str, Any]:
    """Reuse 6F's AST harness and replace only role/lifecycle validation."""

    candidate_spec(str(args.semantic_candidate).upper())
    namespace = dict(parent.run_window.__globals__)
    namespace.update(
        close_role=_closed,
        snapshot_role=_snapshot,
        take_role_records=_take,
        _validate_pool=_validate_pools,
        pool_contract=pool_contract,
    )
    result = FunctionType(parent.run_window.__code__, namespace)(args, built, **kwargs)
    finalized = getattr(built[1], "_cpu_s4_6f_finalized_pool", {})
    activity = finalized.get("activity", []) if isinstance(finalized, dict) else []
    alpha_rows = [row for row in activity if row.get("label") in ALPHA_LABELS]
    alpha_receipt = {
        "schema": "filament.cpu_s4_6j.alpha_measurement.v1",
        "semantic_candidate": CANDIDATE,
        "role": str(args.reported_role).upper(),
        "intervals": int(args.intervals),
        "alpha_activity_count": len(alpha_rows),
        "alpha_activity": alpha_rows,
        "records_scalar_only": all(_scalar_tree(row) for row in alpha_rows),
        "fallback": 0,
        "counts_separate_from_original_math_dispatches": True,
    }
    if hasattr(args, "work"):
        sg.write(args.work / "ALPHA_ACTIVITY.json", alpha_receipt)
    result["alpha_measurement"] = alpha_receipt
    result["alpha_pointwise_calls"] = int(args.intervals) if args.reported_role == "X" else 0
    result["alpha_rounds"] = 5 * result["alpha_pointwise_calls"]
    result["pointwise_components"] = list(COMPONENTS)
    return result


validate_pools = _validate_pools
__all__ = ["run_window", "validate_pools", "KEYS", "_snapshot"]
