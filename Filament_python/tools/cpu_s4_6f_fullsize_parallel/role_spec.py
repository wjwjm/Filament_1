"""Bounded adapter from semantic 6F roles to the frozen 6D/6E builders."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import Any

from tools.cpu_s4_4b_fullsize_linear_cache import candidate as cache_candidate
from tools.cpu_s4_4b_fullsize_linear_cache import contract as cache_contract

from .protocol import MICRO_SHAPE, SHAPE, candidate_spec


class RoleUnsupported(RuntimeError):
    """The selected semantic role is outside this bounded adapter."""


def _legacy_module(spec: dict[str, Any]):
    return importlib.import_module(spec["module"])


def _legacy_mode(mode: str) -> str:
    value = str(mode).upper()
    if value == "PURE_OFF_FULLSIZE":
        return "PURE_OFF_SMALLGRID"
    return value


def _pool_names(operation: str) -> tuple[str, str, str]:
    if operation == "EQ10":
        return "close_eq10_pool", "eq10_pool_snapshot", "take_eq10_pool_records"
    if operation == "INTENSITY":
        return "close_intensity_pool", "intensity_pool_snapshot", "take_intensity_pool_records"
    if operation == "PHASE_ABSORB":
        return "close_c_pool", "c_pool_snapshot", "take_c_pool_records"
    raise RoleUnsupported(f"unknown pointwise operation: {operation}")


def build_role(
    semantic: str,
    mode: str,
    *,
    capture_root: str | Path,
    role: str,
    local_engineering: bool,
    recorder_pair: Any,
    shape: tuple[int, int, int] | None = None,
    pool_override: int | None = None,
) -> tuple[Any, Any, Any, dict[str, Any], Any, Any]:
    """Build one frozen role, then replace only its empty 4A cache."""

    spec = candidate_spec(semantic)
    configured_shape = tuple(MICRO_SHAPE if local_engineering else SHAPE if shape is None else shape)
    cache_contract.validate_shape(
        configured_shape,
        scope=cache_contract.LOCAL_ENGINEERING_SCOPE if local_engineering else cache_contract.FULLSIZE_SCOPE,
    )
    module = _legacy_module(spec)
    role_name = str(role).upper()
    if role_name not in ("R", "X"):
        raise RoleUnsupported(f"invalid role mapping for {semantic!r}")
    legacy_role = "R" if role_name == "R" else spec["old_role"]
    kwargs: dict[str, Any] = dict(
        capture_root=Path(capture_root).resolve(),
        role=legacy_role,
        local_engineering=bool(local_engineering),
        pool_override=pool_override,
        recorder_pair=recorder_pair,
    )
    if spec["old_role"] == "C":
        # Full capture/mechanism is deliberately a later phase.  Passing no
        # store preserves the original C math and still exercises its pool.
        kwargs["capture_store"] = None
    built = module.build_role(_legacy_mode(mode), **kwargs)
    if not isinstance(built, tuple) or len(built) != 6:
        raise RoleUnsupported("LEGACY_ROLE_DID_NOT_RETURN_SIX_TUPLE")
    wrapped, dispatch, clock, binding, audit, lease = built
    cache_binding = cache_candidate._replace_empty_cache(
        dispatch,
        local_engineering=bool(local_engineering),
        configured_shape=configured_shape,
    )
    dispatch.cpu_s4_6f_semantic_id = str(semantic).upper()
    dispatch.cpu_s4_6f_old_role = legacy_role
    dispatch.cpu_s4_6f_role_spec = copy.deepcopy(spec)
    dispatch.cpu_s4_6f_shape = configured_shape
    dispatch.cpu_s4_6f_cache_binding = copy.deepcopy(cache_binding)
    dispatch.cpu_s4_6f_cache_snapshot = dispatch.cpu_s4_4b_cache_snapshot
    dispatch.cpu_s4_6f_pool_api = _pool_names(spec["pointwise_operation"])
    binding = copy.deepcopy(binding)
    binding.update(
        {
            "schema": "filament.cpu_s4_6f.role_binding.v1",
            "semantic_id": str(semantic).upper(),
            "old_role": legacy_role,
            "candidate_module": spec["module"],
            "shape": list(configured_shape),
            "fullsize_cache_binding": copy.deepcopy(cache_binding),
            "fullsize_cache_snapshot": copy.deepcopy(dispatch.cpu_s4_4b_cache_snapshot()),
            "scientific_qualification": False,
            "production_qualified": False,
        }
    )
    dispatch.cpu_s4_6f_binding = copy.deepcopy(binding)
    return wrapped, dispatch, clock, binding, audit, lease


def verify_role(dispatch: Any, semantic: str) -> dict[str, Any]:
    spec = candidate_spec(semantic)
    module = _legacy_module(spec)
    check = module.verify_bindings(dispatch)
    if getattr(dispatch, "cpu_s4_6f_semantic_id", None) != str(semantic).upper():
        raise RoleUnsupported("S46F_SEMANTIC_ID_BINDING_CHANGED")
    cache = dispatch.cpu_s4_6f_cache_snapshot()
    if tuple(cache.get("configured_shape", ())) != tuple(getattr(dispatch, "cpu_s4_6f_shape", ())):
        raise RoleUnsupported("S46F_CACHE_SHAPE_BINDING_CHANGED")
    return {"legacy": check, "semantic_id": str(semantic).upper(), "cache": cache}


def close_role(dispatch: Any, semantic: str) -> dict[str, Any]:
    close_name, _, _ = _pool_names(candidate_spec(semantic)["pointwise_operation"])
    return getattr(dispatch, close_name)()


def snapshot_role(dispatch: Any, semantic: str) -> dict[str, Any]:
    _, snapshot_name, _ = _pool_names(candidate_spec(semantic)["pointwise_operation"])
    value = getattr(dispatch, snapshot_name)()
    # 6D's legacy snapshot predates the live-thread fix and reports the
    # initial IDs.  Re-read the executor here without retaining arrays so the
    # 6F close-before-output receipt records actual post-join liveness.
    pool = getattr(dispatch, "cpu_s4_6d_eq10_pool", None)
    workers = getattr(pool, "_workers", None)
    executor = getattr(workers, "executor", None)
    if executor is not None:
        value["active_thread_ids"] = sorted(
            int(thread.native_id)
            for thread in getattr(executor, "_threads", ())
            if thread.is_alive() and getattr(thread, "native_id", None) is not None
        )
    return value


def take_role_records(dispatch: Any, semantic: str) -> list[dict[str, Any]]:
    _, _, take_name = _pool_names(candidate_spec(semantic)["pointwise_operation"])
    return getattr(dispatch, take_name)()


def pool_contract(semantic: str) -> dict[str, Any]:
    spec = candidate_spec(semantic)
    return {"workers": int(spec["workers"]), "labels": int(spec["pointwise_labels"]), "groups_per_interval": 2 if spec["pointwise_operation"] == "PHASE_ABSORB" else 4}


__all__ = ["RoleUnsupported", "build_role", "close_role", "pool_contract", "snapshot_role", "take_role_records", "verify_role"]
