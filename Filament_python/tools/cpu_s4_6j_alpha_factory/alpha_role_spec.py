"""R/X role adapter for the four-pool R0 plus optional alpha pool."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from tools.cpu_s4_4b_fullsize_linear_cache import contract as cache_contract

from . import alpha_factory
from .alpha_factory import ROLE_CANDIDATE, ROLE_REFERENCE
from .alpha_protocol import CANDIDATE, COMPONENTS, MICRO_SHAPE, SHAPE, candidate_spec


def _expected_shape(local_engineering: bool) -> tuple[int, int, int]:
    return tuple(MICRO_SHAPE if local_engineering else SHAPE)


def build_role(
    semantic: str,
    mode: str,
    *,
    capture_root: str | Path,
    role: str,
    local_engineering: bool,
    recorder_pair: Any,
    cost_mode: str = "OFF",
    shape: tuple[int, int, int] | None = None,
    pool_override: int | None = None,
    capture_store: Any | None = None,
) -> tuple[Any, Any, Any, dict[str, Any], Any, Any]:
    candidate_spec(semantic)
    if str(cost_mode).upper() != "OFF":
        raise ValueError("S46J_ALPHA_OFF_ONLY")
    external_role = str(role).upper()
    if external_role not in (ROLE_REFERENCE, ROLE_CANDIDATE):
        raise ValueError("S46J_ALPHA_ROLE_R_OR_X")
    expected = _expected_shape(bool(local_engineering))
    if shape is not None and tuple(shape) != expected:
        raise ValueError("S46J_ALPHA_GRID_DOMAIN")
    cache_contract.validate_shape(
        expected,
        scope=(
            cache_contract.LOCAL_ENGINEERING_SCOPE
            if local_engineering
            else cache_contract.FULLSIZE_SCOPE
        ),
    )
    if local_engineering and pool_override not in (None, 1):
        raise ValueError("S46J_ALPHA_LOCAL_W1_ONLY")
    if not local_engineering and pool_override is not None:
        raise ValueError("S46J_ALPHA_FORMAL_FIXED_WORKERS")
    built = alpha_factory.build_role(
        "PURE_OFF_SMALLGRID" if mode == "PURE_OFF_FULLSIZE" else mode,
        capture_root=Path(capture_root).resolve(),
        role=external_role,
        local_engineering=bool(local_engineering),
        pool_override=pool_override,
        recorder_pair=recorder_pair,
        capture_store=capture_store,
        shape=expected,
        semantic=CANDIDATE,
    )
    binding = copy.deepcopy(built[3])
    binding.update(
        {
            "schema": "filament.cpu.s4_6j.alpha_role_spec.binding.v1",
            "semantic_id": CANDIDATE,
            "external_role": external_role,
            "internal_role": "X",
            "shape": list(expected),
            "components": list(COMPONENTS),
            "pool_role": "X",
            "alpha_enabled": external_role == ROLE_CANDIDATE,
            "cost_mode": "OFF",
            "production_qualified": False,
        }
    )
    return built[:3] + (binding,) + built[4:]


def verify_role(dispatch: Any, semantic: str = CANDIDATE) -> dict[str, Any]:
    candidate_spec(semantic)
    result = alpha_factory.verify_bindings(dispatch, semantic=CANDIDATE)
    result["role_spec"] = {
        "semantic_id": CANDIDATE,
        "external_role": getattr(dispatch, "cpu_s4_6j_external_role", None),
        "components": list(COMPONENTS),
        "production_qualified": False,
    }
    return result


def close_role(dispatch: Any, semantic: str = CANDIDATE) -> dict[str, Any]:
    candidate_spec(semantic)
    return alpha_factory.close_role(dispatch, semantic=CANDIDATE)


def pool_contract(semantic: str = CANDIDATE) -> dict[str, Any]:
    candidate_spec(semantic)
    return {
        "candidate": CANDIDATE,
        "components": list(COMPONENTS),
        "baseline_workers": [24, 48, 24, 24],
        "alpha_workers": 24,
        "alpha_labels": 5,
        "internal_role": "X",
        "external_roles": [ROLE_REFERENCE, ROLE_CANDIDATE],
    }


def _final(dispatch: Any, semantic: str = CANDIDATE) -> dict[str, Any]:
    candidate_spec(semantic)
    existing = getattr(dispatch, "_cpu_s4_6j_finalized", None)
    if isinstance(existing, dict):
        return copy.deepcopy(existing)
    return close_role(dispatch, semantic)


def snapshot_role(dispatch: Any, semantic: str = CANDIDATE) -> dict[str, Any]:
    final = _final(dispatch, semantic)
    components = {
        name: copy.deepcopy(final[name]["pool"])
        for name in ("intensity", "eq10", "c", "power_volume")
        if isinstance(final.get(name), Mapping) and isinstance(final[name].get("pool"), Mapping)
    }
    alpha_part = final.get("alpha") if isinstance(final.get("alpha"), Mapping) else {}
    alpha_pool = alpha_part.get("pool") if isinstance(alpha_part, Mapping) else None
    all_components = list(components.values())
    if isinstance(alpha_pool, Mapping):
        all_components.append(alpha_pool)
    return {
        "schema": "filament.cpu_s4_6j.alpha_role_spec.pool_snapshot.v1",
        "closed": bool(all_components) and all(bool(row.get("closed", False)) for row in all_components),
        "close_error": None if not final.get("close_errors") else list(final["close_errors"]),
        "active_thread_ids": sorted(
            {int(tid) for row in all_components for tid in row.get("active_thread_ids", ())}
        ),
        "initial_thread_ids": sorted(
            {int(tid) for row in all_components for tid in row.get("initial_thread_ids", ())}
        ),
        "pending_scalar_records": sum(int(row.get("pending_scalar_records", 0)) for row in all_components),
        "initialization_ns": sum(int(row.get("initialization_ns", 0)) for row in all_components),
        "close_ns": sum(int(row.get("close_ns") or 0) for row in all_components),
        "operation_counts": {
            label: int(count)
            for row in all_components
            for label, count in row.get("operation_counts", {}).items()
        },
        "components": components,
        "alpha": copy.deepcopy(alpha_pool),
        "all_pools_retired_before_output": bool(final.get("all_pools_retired_before_output", False)),
        "array_values_retained": False,
    }


def take_role_records(dispatch: Any, semantic: str = CANDIDATE) -> list[dict[str, Any]]:
    final = _final(dispatch, semantic)
    rows: list[dict[str, Any]] = []
    for name in ("intensity", "eq10", "c", "power_volume"):
        part = final.get(name)
        if isinstance(part, Mapping) and isinstance(part.get("activity"), list):
            rows.extend(copy.deepcopy(part["activity"]))
    alpha_part = final.get("alpha")
    if isinstance(alpha_part, Mapping) and isinstance(alpha_part.get("activity"), list):
        rows.extend(copy.deepcopy(alpha_part["activity"]))
    return rows


__all__ = [
    "build_role",
    "close_role",
    "pool_contract",
    "snapshot_role",
    "take_role_records",
    "verify_role",
]
