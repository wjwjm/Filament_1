"""Reuse the original application clock and close anchor for all three pools."""
from types import FunctionType

from tools.cpu_s4_6f_fullsize_parallel import measurement as parent
from .combined_candidate import finalize_factory
from .protocol import CANDIDATE, COMPONENTS, candidate_spec

KEYS = dict(INTENSITY_P24="intensity", EQ10_P48="eq10", PHASE_ABSORB_P24="c")


def _closed(dispatch, semantic):
    candidate_spec(semantic)
    return finalize_factory(dispatch)


def _snapshot(dispatch, semantic):
    value = _closed(dispatch, semantic)
    components = {key: value[key]["pool"] for key in KEYS.values()}
    return dict(
        closed=all(row["closed"] for row in components.values()),
        close_error=None if not value.get("close_errors") else value["close_errors"],
        active_thread_ids=sorted({tid for row in components.values() for tid in row["active_thread_ids"]}),
        initial_thread_ids=sorted({tid for row in components.values() for tid in row.get("initial_thread_ids", [])}),
        pending_scalar_records=sum(row["pending_scalar_records"] for row in components.values()),
        initialization_ns=sum(row.get("initialization_ns", 0) for row in components.values()),
        close_ns=sum(row.get("close_ns") or 0 for row in components.values()),
        operation_counts={label: count for row in components.values() for label, count in row["operation_counts"].items()},
        components=components, array_values_retained=False)


def _take(dispatch, semantic):
    if getattr(dispatch, "_s46h_measurement_records_taken", False):
        return []
    value = _closed(dispatch, semantic)
    dispatch._s46h_measurement_records_taken = True
    return [row for key in KEYS.values() for row in value[key]["activity"]]


def validate_pools(semantic, role, intervals, snapshot, activity, *, local_engineering):
    candidate_spec(semantic)
    for name, key in KEYS.items():
        labels = parent.POINTWISE_LABELS[name]
        rows = [row for row in activity if row.get("label") in labels]
        pool = snapshot["components"][key]
        if pool.get("dropped_records", 0) or pool.get("invalid_records_dropped", 0):
            raise RuntimeError("S46H_QUALIFICATION_ACTIVITY_DROPPED")
        parent._validate_pool(name, role, intervals, pool, rows,
                              local_engineering=local_engineering)
    if len(activity) != (0 if role == "R" else 30 * int(intervals)):
        raise RuntimeError("S46H_COMBINED_ACTIVITY_TOTAL")


def run_window(args, built, **kwargs):
    namespace = dict(parent.run_window.__globals__)
    namespace.update(close_role=_closed, snapshot_role=_snapshot, take_role_records=_take,
                     _validate_pool=validate_pools,
                     pool_contract=lambda semantic: dict(candidate=CANDIDATE, components=list(COMPONENTS), workers=[24,48,24]))
    # Same source AST, stripped-AST proof, normal-output anchor and original
    # application denominator as 6F; only the three-pool hooks differ.
    return FunctionType(parent.run_window.__code__, namespace)(args, built, **kwargs)
