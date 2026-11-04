"""Thin S4-6J factory adapter for the alpha-cleanup candidate.

The four-component S4-6I factory remains the owner of the common stage,
source guards and its original pools.  This module only supplies an
``AlphaParallelHook`` through a private ``FunctionType`` namespace and adds
the fifth pool to the lifecycle/evidence envelope.  No factory timing scope
is activated here.
"""

from __future__ import annotations

import copy
import functools
from pathlib import Path
from types import FunctionType
from typing import Any, Mapping

from tools.cpu_s4_4b_fullsize_linear_cache import candidate as cache_candidate
from tools.cpu_s4_4b_fullsize_linear_cache import contract as cache_contract
from tools.cpu_s4_5b_steady_cost.adapter import make_recorder_pair
from tools.cpu_s4_6i_residual_parallel import selected_factory as selected
from tools.cpu_s4_6i_residual_parallel.power_hook import _seal, verify_seal
from tools.cpu_s4_6i_residual_parallel.power_volume import PowerVolumePool

from . import alpha_hook as alpha_hook_module
from . import alpha_pool as alpha_pool_module
from .alpha_hook import AlphaHookError, AlphaParallelHook
from .alpha_pool import ALPHA_SEMANTIC_ID, AlphaPool
from .factory_timing import source_identity as factory_source_identity
from .protocol import MICRO_SHAPE, SHAPE


ALPHA_FACTORY_SCHEMA = "filament.cpu.s4_6j.alpha_factory.v1"
CANDIDATE = ALPHA_SEMANTIC_ID
ROLE_REFERENCE = "R"
ROLE_CANDIDATE = "X"
POOL_ROLE = "X"
ACTIVITY_BUFFER_CAPACITY = 128


class AlphaFactoryError(RuntimeError):
    """The alpha factory failed closed while preserving its receipt."""


def candidate_spec(name: Any = CANDIDATE) -> dict[str, Any]:
    """Narrow alpha candidate identity; do not inherit the COST6 label."""

    if str(name).upper() != CANDIDATE:
        raise ValueError("S46J_ALPHA_CANDIDATE_IDENTITY")
    return {
        "semantic_id": CANDIDATE,
        "module": "tools.cpu_s4_6j_alpha_factory.alpha_factory",
        "pointwise_operation": "ION_ALPHA_CLEANUP_P24",
        "workers": 24,
        "production_qualified": False,
    }


def _normalise_role(role: Any) -> str:
    value = str(role).upper()
    aliases = {
        "REFERENCE": ROLE_REFERENCE,
        "BASELINE": ROLE_REFERENCE,
        "CANDIDATE": ROLE_CANDIDATE,
    }
    value = aliases.get(value, value)
    if value not in (ROLE_REFERENCE, ROLE_CANDIDATE):
        raise AlphaFactoryError(f"S46J_ALPHA_ROLE_UNSUPPORTED:{role}")
    return value


def _normalise_mode(mode: Any) -> str:
    value = str(mode).upper()
    if value == "PURE_OFF_FULLSIZE":
        return "PURE_OFF_SMALLGRID"
    if value == "PURE_OFF":
        return "PURE_OFF_SMALLGRID"
    if value not in ("QUALIFY_BOUNDARY", "PURE_OFF_SMALLGRID"):
        raise AlphaFactoryError(f"S46J_ALPHA_MODE_UNSUPPORTED:{mode}")
    return value


def _empty_alpha_activity() -> list[dict[str, Any]]:
    return []


def _empty_alpha_stats() -> dict[str, int]:
    return {"total": 0, "dropped": 0}


def _append_alpha_rows(dispatch: Any, rows: Any) -> None:
    if not isinstance(rows, list) or not rows:
        return
    target = getattr(dispatch, "_cpu_s4_6j_alpha_activity", None)
    if not isinstance(target, list):
        target = []
        dispatch._cpu_s4_6j_alpha_activity = target
    stats = getattr(dispatch, "_cpu_s4_6j_alpha_activity_stats", None)
    if not isinstance(stats, dict):
        stats = _empty_alpha_stats()
        dispatch._cpu_s4_6j_alpha_activity_stats = stats
    stats["total"] += len(rows)
    remaining = max(0, ACTIVITY_BUFFER_CAPACITY - len(target))
    target.extend(copy.deepcopy(rows[:remaining]))
    stats["dropped"] += max(0, len(rows) - remaining)


def _collect_alpha_activity(dispatch: Any) -> None:
    hook = getattr(dispatch, "cpu_s4_6j_alpha_hook", None)
    if not isinstance(hook, AlphaParallelHook):
        hook = getattr(dispatch, "cpu_s4_6i_power_hook", None)
    if isinstance(hook, AlphaParallelHook):
        _append_alpha_rows(dispatch, hook.take_alpha_records())
    pool = getattr(dispatch, "cpu_s4_6j_alpha_pool", None)
    if type(pool) is AlphaPool:
        _append_alpha_rows(dispatch, pool.take_records())


def _close_unattached(
    hooks: list[Any], power_pools: list[Any], alpha_pool: Any | None,
) -> None:
    """Best-effort constructor cleanup without replacing the primary error."""

    for hook in hooks:
        try:
            hook.close()
        except BaseException:
            pass
    seen: set[int] = set()
    for pool in [*power_pools, alpha_pool]:
        if pool is None or id(pool) in seen:
            continue
        seen.add(id(pool))
        try:
            pool.close()
        except BaseException:
            pass
        try:
            pool.take_records()
        except BaseException:
            pass


def _alpha_pool_snapshot(dispatch: Any) -> dict[str, Any] | None:
    pool = getattr(dispatch, "cpu_s4_6j_alpha_pool", None)
    if pool is None:
        return None
    if type(pool) is not AlphaPool:
        return {
            "schema": "filament.cpu.s4_6j.alpha_factory.pool_snapshot_failure.v1",
            "closed": False,
            "active_thread_ids": [],
            "close_error": "S46J_ALPHA_POOL_IDENTITY_TYPE_CHANGED",
            "pool_type": f"{type(pool).__module__}.{type(pool).__qualname__}",
        }
    try:
        hook = getattr(dispatch, "cpu_s4_6j_alpha_hook", None)
        if isinstance(hook, AlphaParallelHook):
            snapshot = hook.alpha_snapshot()
            if isinstance(snapshot, dict):
                return copy.deepcopy(snapshot)
        snapshot = pool.snapshot()
        if not isinstance(snapshot, dict):
            raise TypeError("ALPHA_POOL_SNAPSHOT_NOT_MAPPING")
        return copy.deepcopy(snapshot)
    except BaseException as exc:
        return {
            "schema": "filament.cpu.s4_6j.alpha_factory.pool_snapshot_failure.v1",
            "closed": bool(getattr(pool, "closed", False)),
            "active_thread_ids": [],
            "close_error": type(exc).__name__,
        }


def _alpha_identity(dispatch: Any) -> dict[str, Any]:
    """Return scalar evidence for every alpha pool alias before retirement."""

    hook = getattr(dispatch, "cpu_s4_6j_alpha_hook", None)
    current = getattr(dispatch, "cpu_s4_6j_alpha_pool", None)
    constructed = getattr(dispatch, "_cpu_s4_6j_constructed_alpha_pool", None)
    hook_current = getattr(hook, "_alpha_pool", None)
    hook_bound = getattr(hook, "_bound_alpha_pool", None)

    def describe(value: Any) -> dict[str, Any]:
        return {
            "is_none": value is None,
            "type": None
            if value is None
            else f"{type(value).__module__}.{type(value).__qualname__}",
            "object_id": None if value is None else int(id(value)),
            "is_alpha_pool": type(value) is AlphaPool,
        }

    aliases = {
        "current": current,
        "constructed": constructed,
        "hook_current": hook_current,
        "hook_bound": hook_bound,
    }
    values = list(aliases.values())
    non_none = [value for value in values if value is not None]
    all_same = not non_none or all(value is non_none[0] for value in non_none)
    expected_role = getattr(dispatch, "cpu_s4_6j_external_role", None)
    valid = all_same and all(type(value) is AlphaPool for value in non_none)
    if expected_role == ROLE_REFERENCE:
        valid = all(value is None for value in values)
    elif expected_role != ROLE_CANDIDATE:
        valid = False
    return {
        "schema": ALPHA_FACTORY_SCHEMA + ".pool_identity.v1",
        "external_role": expected_role,
        "aliases": {name: describe(value) for name, value in aliases.items()},
        "all_aliases_same": bool(all_same),
        "valid": bool(valid),
    }


def _known_alpha_pools(dispatch: Any) -> list[AlphaPool]:
    hook = getattr(dispatch, "cpu_s4_6j_alpha_hook", None)
    values = (
        getattr(dispatch, "cpu_s4_6j_alpha_pool", None),
        getattr(dispatch, "_cpu_s4_6j_constructed_alpha_pool", None),
        getattr(hook, "_alpha_pool", None),
        getattr(hook, "_bound_alpha_pool", None),
    )
    known: list[AlphaPool] = []
    for value in values:
        if type(value) is AlphaPool and not any(value is previous for previous in known):
            known.append(value)
    return known


def _check_pool_identity(dispatch: Any) -> None:
    """Check the constructed/bound/current alpha and power pool identities."""

    hook = getattr(dispatch, "cpu_s4_6j_alpha_hook", None)
    if type(hook) is not AlphaParallelHook:
        raise AlphaFactoryError("S46J_ALPHA_HOOK_IDENTITY_CHANGED")
    power = getattr(dispatch, "cpu_s4_6i_power_pool", None)
    if hook.pool is not power or power is not getattr(dispatch, "_cpu_s4_6i_constructed_power_pool", None):
        raise AlphaFactoryError("S46J_ALPHA_POWER_POOL_IDENTITY_CHANGED")
    identity = _alpha_identity(dispatch)
    if not identity["valid"]:
        raise AlphaFactoryError("S46J_ALPHA_POOL_IDENTITY_CHANGED")


def _alpha_retire(
    dispatch: Any,
) -> tuple[dict[str, Any] | None, tuple[str, ...], dict[str, Any]]:
    """Retry hook/all-known-pool close and return scalar evidence."""

    hook = getattr(dispatch, "cpu_s4_6j_alpha_hook", None)
    if not isinstance(hook, AlphaParallelHook):
        hook = getattr(dispatch, "cpu_s4_6i_power_hook", None)
    identity = _alpha_identity(dispatch)
    errors: list[str] = []
    if not identity.get("valid", False):
        errors.append("alpha:POOL_IDENTITY_MISMATCH")
    if isinstance(hook, AlphaParallelHook):
        try:
            hook.close()
        except BaseException as exc:
            errors.append("alpha:hook_close:" + type(exc).__name__)
    known = _known_alpha_pools(dispatch)
    identity["known_pool_count"] = len(known)
    identity["known_pool_ids"] = [int(id(pool)) for pool in known]
    snapshots: list[dict[str, Any]] = []
    for pool in known:
        # AlphaParallelHook already attempts this close.  Calling every known
        # identity directly is intentional: a substituted current pool must
        # not leave the constructed/bound pool live.
        try:
            pool.close()
        except BaseException as exc:
            errors.append("alpha:pool_close:" + type(exc).__name__)
        try:
            _append_alpha_rows(dispatch, pool.take_records())
        except BaseException as exc:
            errors.append("alpha:drain:" + type(exc).__name__)
        try:
            current_snapshot = pool.snapshot()
            if not isinstance(current_snapshot, dict):
                raise TypeError("ALPHA_POOL_SNAPSHOT_NOT_MAPPING")
            snapshots.append(copy.deepcopy(current_snapshot))
        except BaseException as exc:
            errors.append("alpha:snapshot:" + type(exc).__name__)
            snapshots.append(
                {
                    "schema": "filament.cpu.s4_6j.alpha_factory.pool_snapshot_failure.v1",
                    "closed": bool(getattr(pool, "closed", False)),
                    "active_thread_ids": [],
                    "close_error": type(exc).__name__,
                }
            )
    snapshot = snapshots[0] if snapshots else _alpha_pool_snapshot(dispatch)
    for current_snapshot in snapshots or ([snapshot] if isinstance(snapshot, dict) else []):
        if current_snapshot.get("close_error") is not None:
            errors.append("alpha:recorded_close_error:" + str(current_snapshot["close_error"]))
        if current_snapshot.get("active_thread_ids"):
            errors.append("alpha:LIVE_THREADS")
        if int(current_snapshot.get("dropped_records", 0)) or int(
            current_snapshot.get("invalid_records", 0)
        ):
            errors.append("alpha:ACTIVITY_RECORD_LOSS")
        if int(current_snapshot.get("hook_dropped_records", 0)):
            errors.append("alpha:HOOK_ACTIVITY_RECORD_LOSS")
    if len(snapshots) > 1:
        if any(
            bool(row.get("active_thread_ids")) or not bool(row.get("closed", False))
            for row in snapshots
        ):
            errors.append("alpha:KNOWN_POOL_NOT_RETIRED")
    return snapshot, tuple(errors), identity


def _selected_components(dispatch: Any) -> tuple[str, ...]:
    names = ("intensity", "eq10", "c", "power_volume")
    return tuple(name for name in names if isinstance(getattr(dispatch, "_cpu_s4_6i_finalized", None), dict))


def take_records(dispatch: Any) -> dict[str, list[dict[str, Any]]]:
    """Drain the four selected records plus a separate alpha stream."""

    result = selected.take_records(dispatch)
    _collect_alpha_activity(dispatch)
    result["alpha"] = copy.deepcopy(getattr(dispatch, "_cpu_s4_6j_alpha_activity", []))
    dispatch._cpu_s4_6j_alpha_activity = _empty_alpha_activity()
    dispatch._cpu_s4_6j_alpha_activity_stats = _empty_alpha_stats()
    return result


def pool_snapshot(dispatch: Any) -> dict[str, Any]:
    result = selected.selected_pool_snapshot(dispatch)
    alpha_snapshot = _alpha_pool_snapshot(dispatch)
    result["alpha"] = {
        "enabled": alpha_snapshot is not None,
        "pool": alpha_snapshot,
    }
    return copy.deepcopy(result)


def _merge_alpha_final(
    dispatch: Any,
    base: Mapping[str, Any],
    *,
    reason: str,
    selected_error: BaseException | None,
    initial_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _collect_alpha_activity(dispatch)
    alpha_snapshot, alpha_errors, final_identity = _alpha_retire(dispatch)
    _collect_alpha_activity(dispatch)
    alpha_rows = copy.deepcopy(getattr(dispatch, "_cpu_s4_6j_alpha_activity", []))
    alpha_stats = copy.deepcopy(getattr(dispatch, "_cpu_s4_6j_alpha_activity_stats", _empty_alpha_stats()))
    final = copy.deepcopy(dict(base))
    final["alpha"] = {
        "enabled": alpha_snapshot is not None,
        "pool": alpha_snapshot,
        "activity": alpha_rows,
        "records_scalar_only": True,
        "activity_records_total": int(alpha_stats.get("total", 0)),
        "activity_records_dropped": int(alpha_stats.get("dropped", 0)),
        "identity_initial": copy.deepcopy(dict(initial_identity or final_identity)),
        "identity_after": copy.deepcopy(final_identity),
    }
    errors = list(final.get("close_errors", ()))
    errors.extend(alpha_errors)
    if not bool(final_identity.get("valid", False)):
        errors.append("alpha:POOL_IDENTITY_MISMATCH")
    if int(alpha_stats.get("dropped", 0)):
        errors.append("alpha:FACTORY_ACTIVITY_RECORD_LOSS")
    final.update(
        {
            "schema": ALPHA_FACTORY_SCHEMA + ".finalized.v1",
            "reason": str(reason),
            "semantic_id": CANDIDATE,
            "external_role": getattr(dispatch, "cpu_s4_6j_external_role", None),
            "pool_role": POOL_ROLE,
            "close_errors": tuple(errors),
            "records_scalar_only": True,
        }
    )
    selected_live = tuple(final.get("live_pools", ()))
    alpha_live = bool(isinstance(alpha_snapshot, dict) and alpha_snapshot.get("active_thread_ids"))
    live = tuple(dict.fromkeys([*selected_live, *(["alpha"] if alpha_live else [])]))
    selected_closed = bool(final.get("all_pools_retired_before_output", False))
    alpha_closed = alpha_snapshot is None or bool(alpha_snapshot.get("closed", False))
    final["live_pools"] = live
    final["all_closed"] = bool(final.get("all_closed", False) and alpha_closed)
    final["all_pools_retired_before_output"] = bool(
        selected_closed and alpha_closed and not live and not errors
    )
    selected_totals = final.get("activity_records_total")
    if not isinstance(selected_totals, dict):
        selected_totals = {}
    selected_totals = copy.deepcopy(selected_totals)
    selected_totals["alpha"] = int(alpha_stats.get("total", 0))
    final["activity_records_total"] = selected_totals
    selected_dropped = final.get("activity_records_dropped")
    if not isinstance(selected_dropped, dict):
        selected_dropped = {}
    selected_dropped = copy.deepcopy(selected_dropped)
    selected_dropped["alpha"] = int(alpha_stats.get("dropped", 0))
    final["activity_records_dropped"] = selected_dropped
    if selected_error is not None:
        final["finalization_failed"] = True
        final["primary_finalization_error"] = {
            "type": type(selected_error).__name__,
            "message": str(selected_error),
        }
    elif not final["all_pools_retired_before_output"]:
        final["finalization_failed"] = True
    return final


def finalize_factory(dispatch: Any, *, reason: str = "NORMAL") -> dict[str, Any]:
    """Close all five pools, retry failed joins, and return scalar evidence."""

    existing = getattr(dispatch, "_cpu_s4_6j_finalized", None)
    if isinstance(existing, dict):
        _check_pool_identity(dispatch)
        current = pool_snapshot(dispatch)
        for name, row in current.items():
            if isinstance(row, dict) and isinstance(row.get("pool"), dict):
                if row["pool"].get("active_thread_ids"):
                    raise AlphaFactoryError("S46J_FINALIZED_POOL_REACTIVATED:" + str(name))
            elif name == "alpha" and isinstance(row, dict):
                pool = row.get("pool")
                if isinstance(pool, dict) and pool.get("active_thread_ids"):
                    raise AlphaFactoryError("S46J_FINALIZED_POOL_REACTIVATED:alpha")
        return copy.deepcopy(existing)

    initial_identity = _alpha_identity(dispatch)
    selected_error: BaseException | None = None
    first_failure: dict[str, Any] | None = None
    try:
        base = selected.finalize_factory(dispatch, reason=reason)
    except BaseException as exc:
        selected_error = exc
        failure = getattr(dispatch, "_cpu_s4_6i_finalization_failure", None)
        base = copy.deepcopy(failure) if isinstance(failure, dict) else {}
        first_failure = copy.deepcopy(base)
        # The selected finalizer can observe an alpha close failure through
        # the joint hook and stop before the independent alpha receipt is
        # complete.  Retry the real joins once before constructing the final
        # receipt; this is deliberately not a status-only retry.
        _collect_alpha_activity(dispatch)
        _alpha_retire(dispatch)
        try:
            base = selected.finalize_factory(dispatch, reason=reason)
            selected_error = None
        except BaseException as retry_error:
            selected_error = retry_error
            retry_failure = getattr(dispatch, "_cpu_s4_6i_finalization_failure", None)
            if isinstance(retry_failure, dict):
                base = copy.deepcopy(retry_failure)

    final = _merge_alpha_final(
        dispatch,
        base,
        reason=reason,
        selected_error=selected_error,
        initial_identity=initial_identity,
    )
    prior = getattr(dispatch, "_cpu_s4_6j_finalization_failure", None)
    if first_failure is not None:
        prior = first_failure
    if isinstance(prior, dict):
        final["prior_finalization_failure"] = copy.deepcopy(prior)
    if not final.get("all_pools_retired_before_output", False):
        final["finalization_failed"] = True
        dispatch._cpu_s4_6j_finalization_failure = copy.deepcopy(final)
        error = AlphaFactoryError("S46J_ALPHA_POOL_FINALIZATION_FAILED")
        if selected_error is not None:
            raise error from selected_error
        raise error

    dispatch._cpu_s4_6j_alpha_activity = _empty_alpha_activity()
    dispatch._cpu_s4_6j_alpha_activity_stats = _empty_alpha_stats()
    dispatch._cpu_s4_6j_finalized = copy.deepcopy(final)
    state = getattr(dispatch, "cpu_s4_6h_state", None)
    if state is not None:
        state._cpu_s4_6j_finalized = copy.deepcopy(final)
    return copy.deepcopy(final)


def metadata(dispatch: Any) -> dict[str, Any]:
    result = selected.metadata(dispatch)
    result.update(
        {
            "schema": ALPHA_FACTORY_SCHEMA + ".binding.v1",
            "semantic_id": CANDIDATE,
            "external_role": getattr(dispatch, "cpu_s4_6j_external_role", None),
            "internal_role": "X",
            "pool_role": POOL_ROLE,
            "alpha_pool": _alpha_pool_snapshot(dispatch),
            "alpha_enabled": getattr(dispatch, "cpu_s4_6j_alpha_pool", None) is not None,
            "source_identity": copy.deepcopy(getattr(dispatch, "cpu_s4_6j_source_identity", None)),
            "fullsize_cache_binding": copy.deepcopy(getattr(dispatch, "cpu_s4_6j_cache_binding", None)),
        }
    )
    return result


def source_identity() -> dict[str, Any]:
    """Bind the adapter and alpha kernel files without changing PV2 semantics."""

    return {
        "schema": ALPHA_FACTORY_SCHEMA + ".source_identity.v1",
        "semantic_id": CANDIDATE,
        "selected_source_identity": copy.deepcopy(selected.source_identity()),
        "alpha_source_identity": factory_source_identity(
            extra_paths=(
                __file__,
                Path(alpha_hook_module.__file__).resolve(),
                Path(alpha_pool_module.__file__).resolve(),
            )
        ),
        "scientific_expression_changes": 0,
        "internal_role": "X",
    }


def _wrap_alpha_propagator(wrapped: Any, dispatch: Any) -> Any:
    """Add the alpha-owned failure receipt around selected's wrapper."""

    @functools.wraps(wrapped)
    def alpha_propagator(*args: Any, **kwargs: Any) -> Any:
        try:
            result = wrapped(*args, **kwargs)
            _collect_alpha_activity(dispatch)
            return result
        except BaseException as error:
            try:
                finalize_factory(dispatch, reason="TASK_FAILURE")
            except BaseException as cleanup:
                try:
                    error.add_note("S46J_ALPHA_CLOSE_FAILURE:" + type(cleanup).__name__)
                except BaseException:
                    pass
            raise

    for name, value in getattr(wrapped, "__dict__", {}).items():
        if name != "__wrapped__":
            setattr(alpha_propagator, name, value)
    for name in (
        "__g2_inner__",
        "__g2_actual_code__",
        "__g2_expected_code__",
        "__s46h_failure_wrapper__",
        "__s46i_selected_failure_wrapper__",
    ):
        setattr(alpha_propagator, name, getattr(wrapped, name, None))
    alpha_propagator.__s46j_alpha_failure_wrapper__ = True
    return alpha_propagator


def verify_bindings(dispatch: Any, semantic: str = CANDIDATE) -> dict[str, Any]:
    candidate_spec(semantic)
    selected_check = selected.verify_bindings(dispatch)
    if getattr(dispatch, "cpu_s4_6j_external_role", None) not in (ROLE_REFERENCE, ROLE_CANDIDATE):
        raise AlphaFactoryError("S46J_EXTERNAL_ROLE_BINDING_CHANGED")
    _check_pool_identity(dispatch)
    construct = getattr(dispatch, "_cpu_s4_6j_construct_hook", None)
    construct_seal = getattr(dispatch, "_cpu_s4_6j_construct_hook_seal", None)
    if not callable(construct) or not isinstance(construct_seal, Mapping):
        raise AlphaFactoryError("S46J_ALPHA_CONSTRUCTOR_BINDING_MISSING")
    if construct_seal.get("function") is not construct:
        raise AlphaFactoryError("S46J_ALPHA_CONSTRUCTOR_BINDING_CHANGED")
    verify_seal(construct_seal)
    hook = getattr(dispatch, "cpu_s4_6j_alpha_hook", None)
    if getattr(dispatch, "cpu_s4_6i_power_hook", None) is not hook:
        raise AlphaFactoryError("S46J_ALPHA_POWER_HOOK_ALIAS_CHANGED")
    alpha = getattr(dispatch, "cpu_s4_6j_alpha_pool", None)
    if getattr(dispatch, "cpu_s4_6j_external_role", None) == ROLE_REFERENCE:
        if alpha is not None or hook._alpha_pool is not None:
            raise AlphaFactoryError("S46J_REFERENCE_ALPHA_POOL_PRESENT")
    else:
        if type(alpha) is not AlphaPool or hook._alpha_pool is not alpha:
            raise AlphaFactoryError("S46J_CANDIDATE_ALPHA_POOL_IDENTITY_CHANGED")
        expected_workers = 1 if getattr(dispatch, "cpu_s4_6j_local_engineering", False) else 24
        if int(alpha.count) != expected_workers:
            raise AlphaFactoryError("S46J_ALPHA_WORKER_CONTRACT_CHANGED")
    alpha_check = hook.verify_binding(getattr(dispatch, "cpu_s4_6e_c_active_globals", None))
    current_source = source_identity()
    if current_source != getattr(dispatch, "cpu_s4_6j_source_identity", None):
        raise AlphaFactoryError("S46J_ALPHA_FACTORY_SOURCE_IDENTITY_CHANGED")
    cache_snapshot = dispatch.cpu_s4_6j_cache_snapshot()
    expected_shape = tuple(getattr(dispatch, "cpu_s4_6j_shape", ()))
    if tuple(cache_snapshot.get("configured_shape", ())) != expected_shape:
        raise AlphaFactoryError("S46J_ALPHA_CACHE_SHAPE_BINDING_CHANGED")
    return {
        "schema": ALPHA_FACTORY_SCHEMA + ".binding.v1",
        "semantic_id": CANDIDATE,
        "external_role": getattr(dispatch, "cpu_s4_6j_external_role", None),
        "internal_role": "X",
        "selected": copy.deepcopy(selected_check),
        "alpha": copy.deepcopy(alpha_check),
        "cache": copy.deepcopy(cache_snapshot),
        "source_identity_bound": True,
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
    residual_cost_hook: Any | None = None,
    shape: tuple[int, int, int] | None = None,
    semantic: str = CANDIDATE,
) -> tuple[Any, Any, Any, dict[str, Any], Any, Any]:
    """Build external R/X while retaining selected H+power ownership."""

    candidate_spec(semantic)
    external_role = _normalise_role(role)
    mode_name = _normalise_mode(mode)
    expected_shape = tuple(MICRO_SHAPE if local_engineering else SHAPE)
    if shape is not None and tuple(shape) != expected_shape:
        raise AlphaFactoryError("S46J_ALPHA_GRID_DOMAIN")
    cache_contract.validate_shape(
        expected_shape,
        scope=(
            cache_contract.LOCAL_ENGINEERING_SCOPE
            if local_engineering
            else cache_contract.FULLSIZE_SCOPE
        ),
    )
    if recorder_pair is None:
        recorder_pair = make_recorder_pair(False)
    if local_engineering:
        if pool_override not in (None, 1):
            raise AlphaFactoryError("S46J_ALPHA_LOCAL_W1_ONLY")
        pool_override = 1
    elif pool_override is not None:
        raise AlphaFactoryError("S46J_ALPHA_FORMAL_P24_FIXED")

    alpha_holder: dict[str, Any] = {"pool": None}
    alpha_pool_obj: AlphaPool | None = None
    tracked_hooks: list[Any] = []
    tracked_power_pools: list[Any] = []
    if external_role == ROLE_CANDIDATE:
        alpha_pool_obj = AlphaPool(1 if local_engineering else 24)
        alpha_holder["pool"] = alpha_pool_obj

    def construct(pool: Any, *, capture_store: Any | None = None) -> AlphaParallelHook:
        tracked_power_pools.append(pool)
        hook = AlphaParallelHook(
            pool,
            alpha_pool=alpha_holder["pool"],
            capture_store=capture_store_external,
            residual_cost_hook=residual_cost_hook,
        )
        tracked_hooks.append(hook)
        return hook

    capture_store_external = capture_store
    dispatch = None
    try:
        namespace = dict(selected.build_role.__globals__)
        namespace["PowerVolumeHook"] = construct
        factory = FunctionType(
            selected.build_role.__code__,
            namespace,
            name=selected.build_role.__name__,
            argdefs=selected.build_role.__defaults__,
            closure=selected.build_role.__closure__,
        )
        factory.__kwdefaults__ = selected.build_role.__kwdefaults__
        built = factory(
            mode_name,
            capture_root=Path(capture_root).resolve(),
            role="X",
            local_engineering=bool(local_engineering),
            pool_override=pool_override,
            recorder_pair=recorder_pair,
            capture_store=None,
        )
        dispatch = built[1]
        hook = getattr(dispatch, "cpu_s4_6i_power_hook", None)
        if type(hook) is not AlphaParallelHook:
            raise AlphaFactoryError("S46J_ALPHA_PRIVATE_HOOK_NOT_INSTALLED")
        dispatch.cpu_s4_6j_external_role = external_role
        dispatch.cpu_s4_6j_local_engineering = bool(local_engineering)
        dispatch.cpu_s4_6j_shape = expected_shape
        dispatch.cpu_s4_6j_alpha_pool = alpha_pool_obj
        dispatch._cpu_s4_6j_constructed_alpha_pool = alpha_pool_obj
        dispatch.cpu_s4_6j_alpha_hook = hook
        dispatch._cpu_s4_6j_construct_hook = construct
        dispatch._cpu_s4_6j_construct_hook_seal = _seal(construct)
        dispatch._cpu_s4_6j_alpha_activity = _empty_alpha_activity()
        dispatch._cpu_s4_6j_alpha_activity_stats = _empty_alpha_stats()
        dispatch._cpu_s4_6j_alpha_factory_schema = ALPHA_FACTORY_SCHEMA
        cache_binding = cache_candidate._replace_empty_cache(
            dispatch,
            local_engineering=bool(local_engineering),
            configured_shape=expected_shape,
        )
        dispatch.cpu_s4_6j_cache_binding = copy.deepcopy(cache_binding)
        dispatch.cpu_s4_6j_cache_snapshot = dispatch.cpu_s4_4a_cache_snapshot
        dispatch._cpu_s4_6j_source_identity = source_identity()
        dispatch.cpu_s4_6j_source_identity = copy.deepcopy(dispatch._cpu_s4_6j_source_identity)
        hook.set_failure_cleanup(lambda: finalize_factory(dispatch, reason="TASK_FAILURE"))
        dispatch.close_selected_pools = lambda: finalize_factory(dispatch)
        dispatch.cpu_s4_6i_finalizer = lambda *args, **kwargs: finalize_factory(dispatch, **kwargs)
        dispatch.cpu_s4_6h_finalizer = lambda *args, **kwargs: finalize_factory(dispatch, **kwargs)
        dispatch.combined_pool_snapshot = lambda: pool_snapshot(dispatch)
        dispatch.take_combined_pool_records = lambda: take_records(dispatch)
        dispatch.close_combined_pools = lambda: finalize_factory(dispatch)
        binding = metadata(dispatch)
        binding.update(
            {
                "shape": list(expected_shape),
                "external_role": external_role,
                "internal_role": "X",
                "alpha_enabled": alpha_pool_obj is not None,
                "fullsize_cache_binding": copy.deepcopy(cache_binding),
                "source_identity": copy.deepcopy(dispatch.cpu_s4_6j_source_identity),
            }
        )
        dispatch.cpu_s4_6j_binding = copy.deepcopy(binding)
        dispatch.cpu_s4_6j_binding_check = verify_bindings(dispatch)
        alpha_propagator = _wrap_alpha_propagator(built[0], dispatch)
        return (alpha_propagator, built[1], built[2], copy.deepcopy(binding), built[4], built[5])
    except BaseException as error:
        if dispatch is not None:
            try:
                finalize_factory(dispatch, reason="CONSTRUCTION_FAILURE")
            except BaseException as cleanup:
                try:
                    error.add_note("S46J_ALPHA_CLOSE_FAILURE:" + type(cleanup).__name__)
                except BaseException:
                    pass
        _close_unattached(tracked_hooks, tracked_power_pools, alpha_pool_obj)
        raise


def build_candidate(*args: Any, **kwargs: Any):
    kwargs["role"] = ROLE_CANDIDATE
    return build_role(*args, **kwargs)


def build_reference(*args: Any, **kwargs: Any):
    kwargs["role"] = ROLE_REFERENCE
    return build_role(*args, **kwargs)


def close_role(dispatch: Any, semantic: str = CANDIDATE) -> dict[str, Any]:
    candidate_spec(semantic)
    return finalize_factory(dispatch)


__all__ = [
    "ALPHA_FACTORY_SCHEMA",
    "AlphaFactoryError",
    "candidate_spec",
    "POOL_ROLE",
    "ROLE_CANDIDATE",
    "ROLE_REFERENCE",
    "build_candidate",
    "build_reference",
    "build_role",
    "close_role",
    "finalize_factory",
    "metadata",
    "pool_snapshot",
    "source_identity",
    "take_records",
    "verify_bindings",
]
