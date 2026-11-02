"""Selected S4-6I factory: H baseline plus the P24 power-volume chain."""

from __future__ import annotations

import copy
import functools
import importlib
import os
from pathlib import Path
from typing import Any, Mapping

from .power_hook import POWER_HOOK_SCHEMA, PowerHookError, PowerVolumeHook, owned_source_identity
from .power_volume import (
    POWER_VOLUME_SCHEMA,
    POWER_VOLUME_SEMANTIC_ID,
    PowerVolumePool,
    make_power_volume_pool,
)


SELECTED_FACTORY_SCHEMA = "filament.cpu.s4_6i.residual_parallel.selected_factory.v1"
SELECTED_SEMANTIC_ID = POWER_VOLUME_SEMANTIC_ID
ROLE_REFERENCE = "R"
ROLE_CANDIDATE = "X"
POOL_ROLE = "X"
ACTIVITY_BUFFER_CAPACITY = 128


class SelectedFactoryError(RuntimeError):
    """The selected H/P24 factory failed closed."""


def _normalise_role(role: Any) -> str:
    value = str(role).upper()
    value = {
        "REFERENCE": ROLE_REFERENCE,
        "BASELINE": ROLE_REFERENCE,
        "CANDIDATE": ROLE_CANDIDATE,
        "COMBINED": ROLE_CANDIDATE,
        "H": ROLE_CANDIDATE,
        "J": ROLE_CANDIDATE,
    }.get(value, value)
    if value not in (ROLE_REFERENCE, ROLE_CANDIDATE):
        raise SelectedFactoryError(f"S46I_ROLE_UNSUPPORTED:{role}")
    return value


def _combined_module() -> Any:
    return importlib.import_module("tools.cpu_s4_6h_combined_parallel.combined_candidate")


def _empty_activity() -> dict[str, list[dict[str, Any]]]:
    return {"intensity": [], "eq10": [], "c": [], "power_volume": []}


def _append_rows(dispatch: Any, name: str, target: list[dict[str, Any]], rows: Any) -> None:
    if isinstance(rows, list):
        stats = getattr(dispatch, "_cpu_s4_6i_activity_stats", {}).get(name)
        if not isinstance(stats, dict):
            stats = {"total": 0, "dropped": 0}
            getattr(dispatch, "_cpu_s4_6i_activity_stats", {})[name] = stats
        stats["total"] += len(rows)
        remaining = max(0, ACTIVITY_BUFFER_CAPACITY - len(target))
        target.extend(copy.deepcopy(rows[:remaining]))
        stats["dropped"] += max(0, len(rows) - remaining)


def _collect_call_activity(dispatch: Any) -> None:
    """Move one completed wrapper call into the bounded selected accumulator."""

    accum = getattr(dispatch, "_cpu_s4_6i_activity_accumulator", None)
    if not isinstance(accum, dict):
        return
    h_last = getattr(dispatch, "_cpu_s4_6h_last_call_records", None)
    if isinstance(h_last, Mapping):
        for name in ("intensity", "eq10", "c"):
            _append_rows(dispatch, name, accum[name], h_last.get(name))
        dispatch._cpu_s4_6h_last_call_records = {}
    hook = getattr(dispatch, "cpu_s4_6i_power_hook", None)
    if isinstance(hook, PowerVolumeHook):
        _append_rows(dispatch, "power_volume", accum["power_volume"], hook.take_records())


def _failure_cleanup(dispatch: Any) -> None:
    try:
        finalize_factory(dispatch, reason="TASK_FAILURE")
    except BaseException:
        # The original task exception remains authoritative.  The finalizer
        # stores its bounded failure receipt on the dispatch for inspection.
        pass


def _selected_propagator(wrapped: Any, dispatch: Any) -> Any:
    @functools.wraps(wrapped)
    def selected_propagator(*args: Any, **kwargs: Any) -> Any:
        try:
            result = wrapped(*args, **kwargs)
            _collect_call_activity(dispatch)
            return result
        except BaseException as error:
            try:
                finalize_factory(dispatch, reason="TASK_FAILURE")
            except BaseException as cleanup:
                try:
                    error.add_note("S46I_SELECTED_CLOSE_FAILURE:" + type(cleanup).__name__)
                except BaseException:
                    pass
            raise

    for name, value in getattr(wrapped, "__dict__", {}).items():
        if name != "__wrapped__":
            setattr(selected_propagator, name, value)
    for name in (
        "__g2_inner__",
        "__g2_actual_code__",
        "__g2_expected_code__",
        "__s46h_failure_wrapper__",
    ):
        setattr(selected_propagator, name, getattr(wrapped, name, None))
    selected_propagator.__s46i_selected_failure_wrapper__ = True
    return selected_propagator


def _power_result(dispatch: Any) -> tuple[dict[str, Any], list[dict[str, Any]], tuple[str, ...]]:
    hook = getattr(dispatch, "cpu_s4_6i_power_hook", None)
    pool = getattr(dispatch, "cpu_s4_6i_power_pool", None)
    owned = getattr(dispatch, '_cpu_s4_6i_constructed_power_pool', None)
    hook_pool = getattr(hook, 'pool', None)
    bound_pool = getattr(hook, '_bound_pool', None)
    known = []
    for candidate in (owned, pool, hook_pool, bound_pool):
        if type(candidate) is PowerVolumePool and all(candidate is not previous for previous in known):
            known.append(candidate)
    if not known:
        return (
            {"closed": True, "active_thread_ids": [], "close_error": None, "pending_scalar_records": 0},
            [],
            (),
        )
    errors: list[str] = []
    if not (owned is pool is hook_pool is bound_pool):
        errors.append('power_volume:POOL_IDENTITY_MISMATCH')
    try:
        if isinstance(hook,PowerVolumeHook):
            hook.close()
    except BaseException as exc:
        errors.append("power_volume:" + type(exc).__name__)
    records = []
    # Retire every known constructed/bound pool even after an identity or
    # close-hook failure; never report success from only one substituted object.
    for current in known:
        try:
            current.close()
        except BaseException as exc:
            errors.append('power_volume:direct_close:'+type(exc).__name__)
        try:
            records.extend(current.take_records())
        except BaseException as exc:
            errors.append('power_volume:drain:'+type(exc).__name__)
    primary = owned if type(owned) is PowerVolumePool else known[0]
    try:
        snapshot = primary.snapshot()
    except BaseException as exc:
        errors.append("power_volume:snapshot:" + type(exc).__name__)
        snapshot = {"closed": bool(getattr(primary, "closed", False)), "active_thread_ids": []}
    if snapshot.get("close_error") is not None:
        errors.append("power_volume:recorded_close_error:" + str(snapshot["close_error"]))
    return copy.deepcopy(snapshot), copy.deepcopy(records), tuple(errors)


def finalize_factory(dispatch: Any, *, reason: str = "NORMAL") -> dict[str, Any]:
    """Close/join/drain H's three pools and the selected P24 pool."""

    existing = getattr(dispatch, "_cpu_s4_6i_finalized", None)
    if isinstance(existing, dict):
        return copy.deepcopy(existing)
    combined = _combined_module()
    _collect_call_activity(dispatch)
    accum = getattr(dispatch, "_cpu_s4_6i_activity_accumulator", _empty_activity())
    if not isinstance(accum, dict):
        accum = _empty_activity()
    # H's finalizer merges its last-call map.  Replace it with the complete
    # selected accumulator so the public activity tree remains one tree.
    dispatch._cpu_s4_6h_last_call_records = {
        name: copy.deepcopy(accum.get(name, [])) for name in ("intensity", "eq10", "c")
    }
    power_enabled = any(type(value) is PowerVolumePool for value in (
        getattr(dispatch, "cpu_s4_6i_power_pool", None),
        getattr(dispatch, '_cpu_s4_6i_constructed_power_pool', None),
        getattr(getattr(dispatch, 'cpu_s4_6i_power_hook', None), '_bound_pool', None)))
    power_snapshot, power_records, power_errors = _power_result(dispatch)
    h_result: dict[str, Any] | None = None
    h_error: BaseException | None = None
    try:
        h_result = combined.finalize_factory(dispatch, reason=reason)
    except BaseException as exc:
        h_error = exc
        failure = getattr(dispatch, "_cpu_s4_6h_finalization_failure", None)
        if isinstance(failure, dict):
            h_result = copy.deepcopy(failure)
        else:
            h_result = {}
    final = copy.deepcopy(h_result or {})
    if power_enabled:
        final["power_volume"] = {
            "pool": power_snapshot,
            "activity": copy.deepcopy(accum.get("power_volume", [])) + power_records,
        }
    final.update(
        {
            "schema": SELECTED_FACTORY_SCHEMA + ".finalized.v1",
            "reason": str(reason),
            "selected_semantic_id": SELECTED_SEMANTIC_ID,
            "pool_role": POOL_ROLE,
            "records_scalar_only": True,
        }
    )
    close_errors = list(final.get("close_errors", ()))
    if power_enabled:
        close_errors.extend(power_errors)
        if power_snapshot.get('dropped_records',0) or power_snapshot.get('invalid_records_dropped',0):
            close_errors.append('power_volume:ACTIVITY_RECORD_LOSS')
        if power_snapshot.get("close_error") is not None and not any(
            item.startswith("power_volume:recorded_close_error") for item in close_errors
        ):
            close_errors.append("power_volume:recorded_close_error:" + str(power_snapshot["close_error"]))
    final["close_errors"] = tuple(close_errors)
    final["activity_records_total"] = copy.deepcopy(
        getattr(dispatch, "_cpu_s4_6i_activity_stats", {})
    )
    final["activity_records_dropped"] = {
        name: int(stats.get("dropped", 0))
        for name, stats in getattr(dispatch, "_cpu_s4_6i_activity_stats", {}).items()
        if isinstance(stats, dict)
    }
    close_errors.extend('activity_records_dropped:'+name for name,count in final['activity_records_dropped'].items() if count)
    final['close_errors'] = tuple(close_errors)
    components = [
        row
        for name, row in final.items()
        if name in (("intensity", "eq10", "c", "power_volume") if power_enabled else ("intensity", "eq10", "c"))
        and isinstance(row, dict)
    ]
    all_closed = bool(components) and all(bool(row.get("pool", {}).get("closed", False)) for row in components)
    live = tuple(
        name
        for name in (("intensity", "eq10", "c", "power_volume") if power_enabled else ("intensity", "eq10", "c"))
        if isinstance(final.get(name), dict)
        and bool(final[name].get("pool", {}).get("active_thread_ids"))
    )
    final["all_pools_retired_before_output"] = bool(all_closed and not live and not close_errors)
    final["all_closed"] = bool(all_closed)
    final["live_pools"] = live
    if not final["all_pools_retired_before_output"] or h_error is not None:
        final["finalization_failed"] = True
        dispatch._cpu_s4_6i_finalization_failure = copy.deepcopy(final)
        raise SelectedFactoryError("S46I_POOL_FINALIZATION_FAILED") from h_error
    dispatch._cpu_s4_6i_activity_accumulator = _empty_activity()
    dispatch._cpu_s4_6i_activity_stats = {
        name: {"total": 0, "dropped": 0} for name in _empty_activity()
    }
    dispatch._cpu_s4_6i_finalized = copy.deepcopy(final)
    state = getattr(dispatch, "cpu_s4_6h_state", None)
    if state is not None:
        state._cpu_s4_6i_finalized = copy.deepcopy(final)
    return copy.deepcopy(final)


def take_records(dispatch: Any) -> dict[str, list[dict[str, Any]]]:
    """Take all currently drained scalar rows in the selected component shape."""

    _collect_call_activity(dispatch)
    accum = getattr(dispatch, "_cpu_s4_6i_activity_accumulator", _empty_activity())
    if not isinstance(accum, dict):
        return _empty_activity()
    names = ("intensity", "eq10", "c", "power_volume") if type(
        getattr(dispatch, "cpu_s4_6i_power_pool", None)) is PowerVolumePool else ("intensity", "eq10", "c")
    result = {name: copy.deepcopy(accum.get(name, [])) for name in names}
    dispatch._cpu_s4_6i_activity_accumulator = _empty_activity()
    dispatch._cpu_s4_6i_activity_stats = {
        name: {"total": 0, "dropped": 0} for name in _empty_activity()
    }
    return result


def metadata(dispatch: Any) -> dict[str, Any]:
    combined = _combined_module()
    result = combined.metadata(dispatch)
    result.update(
        {
            "schema": SELECTED_FACTORY_SCHEMA + ".binding.v1",
            "selected_semantic_id": SELECTED_SEMANTIC_ID,
            "role": getattr(dispatch, "cpu_s4_6i_selected_role", None),
            "pool_role": POOL_ROLE,
            "formal_timing_enabled": False,
            "capture_enabled": bool(getattr(dispatch, "cpu_s4_6i_power_capture_enabled", False)),
            "power_volume": copy.deepcopy(
                getattr(getattr(dispatch, "cpu_s4_6i_power_hook", None), "binding", lambda: None)()
            ),
        }
    )
    return result


def source_identity() -> dict[str, Any]:
    """Return H's source identity with the selected semantic binding."""

    combined = _combined_module()
    return {
        "schema": SELECTED_FACTORY_SCHEMA + ".source_identity.v1",
        "selected_semantic_id": SELECTED_SEMANTIC_ID,
        "pool_role": POOL_ROLE,
        "h_source_identity": copy.deepcopy(combined.source_identity()),
        "owned_sources": owned_source_identity(),
        "external_roles": (ROLE_REFERENCE, ROLE_CANDIDATE),
        "scientific_expression_changes": 0,
    }


def selected_pool_snapshot(dispatch: Any) -> dict[str, Any]:
    poolset = getattr(dispatch, "cpu_s4_6h_poolset", None)
    if poolset is None or not callable(getattr(poolset, "snapshots", None)):
        raise SelectedFactoryError("S46I_H_POOLSET_MISSING")
    result = poolset.snapshots()
    hook = getattr(dispatch, "cpu_s4_6i_power_hook", None)
    if isinstance(hook, PowerVolumeHook) and hook.pool is not None:
        result["power_volume"] = hook.snapshot()
    return copy.deepcopy(result)


def verify_bindings(dispatch: Any) -> dict[str, Any]:
    combined = _combined_module()
    if getattr(dispatch, 'cpu_s4_6i_owned_sources', None) != owned_source_identity():
        raise SelectedFactoryError('S46I_SELECTED_OWNED_SOURCE_CHANGED')
    h_check = combined.verify_bindings(dispatch)
    selected_role = _normalise_role(getattr(dispatch, "cpu_s4_6i_selected_role", None))
    if getattr(dispatch, "cpu_s4_6i_pool_role", None) != POOL_ROLE:
        raise SelectedFactoryError("S46I_POOL_ROLE_BINDING_CHANGED")
    if h_check.get("role") != POOL_ROLE:
        raise SelectedFactoryError("S46I_H_POOL_ROLE_CHANGED")
    hook = getattr(dispatch, "cpu_s4_6i_power_hook", None)
    pool = getattr(dispatch, "cpu_s4_6i_power_pool", None)
    if selected_role == ROLE_CANDIDATE:
        if not isinstance(hook, PowerVolumeHook) or type(pool) is not PowerVolumePool:
            raise SelectedFactoryError("S46I_POWER_BINDING_MISSING")
        if hook.pool is not pool or pool is not getattr(dispatch,'_cpu_s4_6i_constructed_power_pool',None):
            raise SelectedFactoryError('S46I_POWER_POOL_IDENTITY_CHANGED')
        power_check = hook.verify_binding(getattr(dispatch, "cpu_s4_6e_c_active_globals", None))
    else:
        capturing = bool(getattr(dispatch,'cpu_s4_6i_power_capture_enabled',False))
        if pool is not None or (hook is not None) != capturing:
            raise SelectedFactoryError("S46I_REFERENCE_POWER_PRESENT")
        if capturing:
            if not isinstance(hook,PowerVolumeHook) or hook.pool is not None:
                raise SelectedFactoryError('S46I_REFERENCE_CAPTURE_HOOK')
            power_check = hook.verify_binding(getattr(dispatch, 'cpu_s4_6e_c_active_globals', None))
        else:
            power_check = None
    return {
        "schema": SELECTED_FACTORY_SCHEMA + ".binding.v1",
        "role": selected_role,
        "pool_role": POOL_ROLE,
        "selected_semantic_id": SELECTED_SEMANTIC_ID,
        "h_binding": copy.deepcopy(h_check),
        "power_binding": copy.deepcopy(power_check),
        "formal_timing_enabled": False,
    }


def build_role(
    mode: str,
    *,
    capture_root: str | Path,
    role: str = ROLE_CANDIDATE,
    local_engineering: bool = False,
    pool_override: int | None = None,
    recorder_pair: Any | None = None,
    capture_store: Any | None = None,
) -> tuple[Any, Any, Any, dict[str, Any], Any, Any]:
    """Build external R/X while retaining H's six-tuple runtime shape.

    Both external roles use H's active role X.  R is the three-pool baseline;
    X adds one independent P24 power-volume pool and its four aliases.
    """

    selected_role = _normalise_role(role)
    use_power = selected_role == ROLE_CANDIDATE
    if use_power and capture_store is not None:
        raise SelectedFactoryError('S46I_CAPTURE_REFERENCE_ONLY')
    if local_engineering:
        if pool_override not in (None, 1):
            raise SelectedFactoryError('S46I_LOCAL_W1_ONLY')
        pool_override = 1
    elif pool_override is not None:
        raise SelectedFactoryError('S46I_FORMAL_P24_FIXED')
    pool: PowerVolumePool | None = None
    hook: PowerVolumeHook | None = None
    try:
        if use_power:
            pool = make_power_volume_pool(1 if local_engineering else 24)
        if use_power or capture_store is not None:
            hook = PowerVolumeHook(pool, capture_store=capture_store)
        combined = _combined_module()
        result = combined.build_role(
            mode,
            capture_root=Path(capture_root).resolve(),
            role=combined.ROLE_CANDIDATE,
            local_engineering=bool(local_engineering),
            pool_override=pool_override,
            recorder_pair=recorder_pair,
            cost_enabled=hook is not None,
            residual_cost_hook=hook,
        )
    except BaseException:
        if hook is not None:
            try:
                hook.close()
            except BaseException:
                pass
            try:
                hook.take_records()
            except BaseException:
                pass
        elif pool is not None:
            try:
                pool.close()
            except BaseException:
                pass
        raise
    dispatch = result[1]
    dispatch.cpu_s4_6i_selected_role = selected_role
    dispatch.cpu_s4_6i_pool_role = POOL_ROLE
    dispatch.cpu_s4_6i_selected_semantic_id = SELECTED_SEMANTIC_ID
    dispatch.cpu_s4_6i_power_pool = pool
    dispatch._cpu_s4_6i_constructed_power_pool = pool
    dispatch.cpu_s4_6i_power_hook = hook
    dispatch.cpu_s4_6i_power_capture_enabled = capture_store is not None
    dispatch.cpu_s4_6i_owned_sources = owned_source_identity()
    dispatch.cpu_s4_6i_formal_timing_enabled = False
    dispatch._cpu_s4_6i_activity_accumulator = _empty_activity()
    dispatch._cpu_s4_6i_activity_stats = {
        name: {"total": 0, "dropped": 0} for name in _empty_activity()
    }
    if hook is not None:
        active_globals = getattr(dispatch, "cpu_s4_6e_c_active_globals", None)
        try:
            hook.bind_active(active_globals)
        except BaseException:
            try:
                finalize_factory(dispatch, reason="POWER_BINDING_FAILURE")
            except BaseException:
                pass
            raise
        hook.set_failure_cleanup(lambda: _failure_cleanup(dispatch))
    dispatch.close_selected_pools = lambda: finalize_factory(dispatch)
    dispatch.cpu_s4_6i_finalizer = finalize_factory
    # Preserve the original H measurement close anchor while making it close
    # all four pools for the selected X role.
    dispatch.cpu_s4_6h_finalizer = lambda *args, **kwargs: finalize_factory(dispatch, **kwargs)
    dispatch.combined_pool_snapshot = lambda: selected_pool_snapshot(dispatch)
    dispatch.take_combined_pool_records = lambda: take_records(dispatch)
    dispatch.close_combined_pools = lambda: finalize_factory(dispatch)
    try:
        selected_propagator = _selected_propagator(result[0], dispatch)
        selected_binding = metadata(dispatch)
        selected_binding["h_role"] = combined.ROLE_CANDIDATE
        selected_binding["power_hook_schema"] = None if hook is None else POWER_HOOK_SCHEMA
        selected_binding["power_volume_schema"] = None if pool is None else POWER_VOLUME_SCHEMA
        dispatch.cpu_s4_6i_selected_binding = copy.deepcopy(selected_binding)
        # Verify after the wrapper and private aliases are fully installed; H's
        # verifier remains authoritative for the three original pools/stage.
        dispatch.cpu_s4_6i_selected_binding_check = verify_bindings(dispatch)
        return (
            selected_propagator,
            dispatch,
            result[2],
            copy.deepcopy(selected_binding),
            result[4],
            result[5],
        )
    except BaseException:
        try:
            finalize_factory(dispatch, reason="CONSTRUCTION_FAILURE")
        except BaseException:
            pass
        raise


def build_candidate(*args: Any, **kwargs: Any):
    kwargs["role"] = ROLE_CANDIDATE
    return build_role(*args, **kwargs)


def build_reference(*args: Any, **kwargs: Any):
    kwargs["role"] = ROLE_REFERENCE
    return build_role(*args, **kwargs)


__all__ = [
    "POOL_ROLE",
    "ROLE_CANDIDATE",
    "ROLE_REFERENCE",
    "SELECTED_FACTORY_SCHEMA",
    "SELECTED_SEMANTIC_ID",
    "SelectedFactoryError",
    "build_candidate",
    "build_reference",
    "build_role",
    "finalize_factory",
    "metadata",
    "selected_pool_snapshot",
    "source_identity",
    "take_records",
    "verify_bindings",
]
