"""Role adapter for the fixed optical small-grid benchmark."""

from __future__ import annotations

import copy
import importlib
import os
from pathlib import Path
from typing import Any, Callable

from .candidate import (
    CANDIDATE_SCHEMA,
    DEFAULT_VARIANT,
    LOCAL_ENGINEERING_CONTRACT,
    POOL_CONTRACT,
    SMALLGRID_MODES,
    SMALLGRID_ROLES,
    CandidateUnsupported,
    _owned_source_identity,
    make_factory,
    unoptimized,
)
from .factory_binding import private_factory
from tools.hpc_ops import provenance_v2 as pv


ADAPTER_SCHEMA = "filament.cpu_s4_3g3_smallgrid.adapter.v1"
LINEAR_RETURN_FORMS = (
    "field",
    "field_plus_energy",
    "field_plus_profile",
    "field_plus_energy_plus_profile",
)
BOUNDARY_CONTRACT = {
    "qualify": {
        "collector": "PARENT_RUNNER_REQUIRED",
        "owner_bound": True,
        "generation_bound": True,
        "capture_once": True,
        "raw_reference_escape": False,
    },
    "pure_off": {
        "collector_enabled": False,
        "transparent": True,
    },
    "timing": {
        "clock": "PARENT_RUNNER_REQUIRED",
        "raw_array_retention": False,
    },
}


def _normalize_mode(mode: str) -> str:
    value = str(mode).upper()
    aliases = {"PURE_OFF_REAL8": "PURE_OFF_SMALLGRID", "PURE_OFF": "PURE_OFF_SMALLGRID"}
    value = aliases.get(value, value)
    if value not in SMALLGRID_MODES:
        raise ValueError(f"unsupported small-grid mode: {mode!r}")
    return value


def _normalize_role(role: str) -> str:
    value = str(role).upper()
    if value not in SMALLGRID_ROLES:
        raise ValueError(f"unsupported small-grid role: {role!r}")
    return value


def _resolve_resource_contract(
    role_name: str,
    *,
    local_engineering: bool,
    pool_override: int | None,
) -> dict[str, Any]:
    """Resolve resources without relabelling a frozen W48 factory.

    The frozen G2/G3 R8 worker contract does not accept the optional local-4
    values.  Keep the local contract explicit for the parent runner, but fail
    before creating a scientific factory until that runner supplies a genuine
    resource adapter.  In particular, never return a W48 factory with
    ``actual_pool=4`` metadata.
    """
    if local_engineering:
        if os.environ.get("SLURM_JOB_ID"):
            raise CandidateUnsupported("LOCAL_ENGINEERING_ONLY_FORBIDDEN_ON_SLURM")
        if pool_override != int(LOCAL_ENGINEERING_CONTRACT["numba_pool"]):
            raise ValueError(
                "local_engineering requires explicit pool_override=1; "
                "the local contract is not inferred"
            )
        return copy.deepcopy(LOCAL_ENGINEERING_CONTRACT)
    if pool_override is not None:
        raise ValueError("pool_override requires local_engineering=True")
    return copy.deepcopy(POOL_CONTRACT[role_name])


def _bind_factory(factory: Callable[..., Any]) -> tuple[Callable[..., Any], Any]:
    """Bind a selected factory to the real frozen propagation entry."""
    propagate_module = importlib.import_module("KHz_filament.propagate")
    frozen_propagator = getattr(propagate_module, "propagate_one_pulse", None)
    if not callable(frozen_propagator):
        raise CandidateUnsupported("FROZEN_PROPAGATOR_ENTRY_MISSING")
    propagator, dispatch = factory(frozen_propagator)
    if not callable(propagator) or dispatch is None:
        raise CandidateUnsupported("ROLE_FACTORY_BINDING_INCOMPLETE")
    return propagator, dispatch


def build_role(
    role: str,
    mode: str,
    *,
    capture_root: str | Path,
    local_engineering: bool = False,
    pool_override: int | None = None,
) -> tuple[Any, Any, Any, dict[str, Any], Any, Any]:
    """Build a named small-grid role with explicit pool and capture contract.

    ``G2_SMALL`` is the frozen G2 path and ``G3_SMALL`` is the lifecycle
    candidate.  The returned tuple follows the existing G2 adapter shape:
    ``(propagator, dispatch, clock, binding, audit, lease)``.  ``clock``,
    ``audit`` and ``lease`` are intentionally left for the parent boundary
    runner, which owns capture and timing policy.
    The declared W24/W48 route is never silently relabelled.  The optional
    local-4 contract is exposed as a static contract but remains fail-closed
    until an upper-layer resource adapter is supplied.
    """
    role_name = _normalize_role(role)
    unoptimized()
    mode_name = _normalize_mode(mode)
    capture = Path(capture_root).resolve()
    expected = _resolve_resource_contract(
        role_name,
        local_engineering=bool(local_engineering),
        pool_override=pool_override,
    )
    import numba
    actual_pool = int(numba.config.NUMBA_NUM_THREADS)
    if actual_pool != int(expected["numba_pool"]):
        raise CandidateUnsupported(f"ACTUAL_NUMBA_POOL_MISMATCH:{actual_pool}:{expected['numba_pool']}")
    numba.set_num_threads(int(expected["outer_mask"]))
    settings = dict(kernel_threads=1, temporal_workers=1, spatial_workers=1) if local_engineering else dict(kernel_threads=24, temporal_workers=48, spatial_workers=24)
    if role_name == "I0_SMALL":
        from tools.cpu_o1.cpu_o1 import make_joint_propagator_factory

        factory = make_joint_propagator_factory(
            "I", block_pixels=32, **settings,
        )
        propagator, dispatch = _bind_factory(factory)
        binding = {
            "adapter_schema": ADAPTER_SCHEMA,
            "schema": ADAPTER_SCHEMA,
            "source_schema": "FROZEN_I0_JOINT_FACTORY_I",
            "role": role_name,
            "mode": mode_name,
            "capture_root": str(capture),
            "candidate": "I0_CURRENT_STREAMING_CONTROL",
            "pool_contract": copy.deepcopy(expected),
            "actual_pool": int(expected["numba_pool"]),
            "resource_scope": "LOCAL_ENGINEERING_ONLY" if local_engineering else "HPC_FIXED_CONTRACT",
            "scientific_qualification": False,
            "boundary_capture": "PARENT_RUNNER_REQUIRED_FOR_QUALIFY_BOUNDARY",
            "timing": "PARENT_RUNNER_LOW_INTERFERENCE_CLOCK",
            "boundary_contract": copy.deepcopy(BOUNDARY_CONTRACT),
            "linear_return_forms": list(LINEAR_RETURN_FORMS),
        }
        return _with_runtime(propagator, dispatch, binding, role_name, mode_name, capture)
    if role_name == "G2_SMALL":
        # The frozen adapter is intentionally used only for the G2 reference;
        # it retains E_stage1 and the original four-role accounting.
        from tools.cpu_s4_3g import candidate as frozen_g

        own = _owned_source_identity()
        own["composite_raw_sha256"] = pv.raw_sha256((own["composite_raw_sha256"] + ":G2_SMALL").encode("ascii"))
        frozen_factory = private_factory(frozen_g, owned_identity=own, local_engineering=local_engineering)
        propagator, dispatch = _bind_factory(frozen_factory)
        binding = {
            "adapter_schema": ADAPTER_SCHEMA,
            "schema": ADAPTER_SCHEMA,
            "source_schema": frozen_g.CANDIDATE_SCHEMA,
            "role": role_name,
            "mode": mode_name,
            "capture_root": str(capture),
            "candidate": "G2_REAL8_BASELINE_V1",
            "pool_contract": copy.deepcopy(expected),
            "actual_pool": int(expected["numba_pool"]),
            "resource_scope": "LOCAL_ENGINEERING_ONLY" if local_engineering else "HPC_FIXED_CONTRACT",
            "scientific_qualification": False,
            "e_stage1": 1,
            "e_apply": 0,
            "omega_hash_roles": ["Omega_stage1", "Omega_apply", "Omega_stage2"],
            "boundary_capture": "PARENT_RUNNER_REQUIRED_FOR_QUALIFY_BOUNDARY",
            "timing": "PARENT_RUNNER_LOW_INTERFERENCE_CLOCK",
            "boundary_contract": copy.deepcopy(BOUNDARY_CONTRACT),
            "linear_return_forms": list(LINEAR_RETURN_FORMS),
        }
        return _with_runtime(propagator, dispatch, binding, role_name, mode_name, capture)
    factory = make_factory(local_engineering=local_engineering)
    propagator, dispatch = _bind_factory(factory)
    binding = {
        "adapter_schema": ADAPTER_SCHEMA,
        "schema": CANDIDATE_SCHEMA,
        "role": "G3_SMALL",
        "mode": mode_name,
        "capture_root": str(capture),
        "candidate": DEFAULT_VARIANT,
        "pool_contract": copy.deepcopy(expected),
        "actual_pool": int(expected["numba_pool"]),
        "resource_scope": "LOCAL_ENGINEERING_ONLY" if local_engineering else "HPC_FIXED_CONTRACT",
        "outer_mask": int(expected["outer_mask"]),
        "scientific_qualification": False,
        "lifecycle_identity": "controlled_lifetime_v1",
        "e_stage1": 0,
        "e_apply": 0,
        "omega_hash_roles": ["Omega_stage1", "Omega_apply", "Omega_stage2"],
        "boundary_capture": "PARENT_RUNNER_REQUIRED_FOR_QUALIFY_BOUNDARY",
        "timing": "PARENT_RUNNER_LOW_INTERFERENCE_CLOCK",
        "boundary_contract": copy.deepcopy(BOUNDARY_CONTRACT),
        "linear_return_forms": list(LINEAR_RETURN_FORMS),
    }
    return _with_runtime(propagator, dispatch, binding, role_name, mode_name, capture)


def _with_runtime(propagator, dispatch, binding, role, mode, capture):
    from .runtime_adapter import attach
    wrapped, dispatch, clock, wrapper_binding, audit, lease = attach(
        propagator, dispatch, role=role, mode=mode, capture_root=capture,
        owned_identity=_owned_source_identity(),
    )
    binding = copy.deepcopy(binding)
    binding["runtime_wrapper"] = wrapper_binding
    binding["boundary_capture"] = "SOURCE_CODE_BOUND_G2_COLLECTOR" if mode == "QUALIFY_BOUNDARY" else "DISABLED"
    binding["timing"] = "FROZEN_LOW_INTERFERENCE_WINDOW_CLOCK"
    binding["boundary_contract"]["qualify"]["collector"] = "SOURCE_CODE_BOUND_G2_COLLECTOR"
    binding["boundary_contract"]["timing"]["clock"] = "FROZEN_LOW_INTERFERENCE_WINDOW_CLOCK"
    return wrapped, dispatch, clock, binding, audit, lease


__all__ = [
    "ADAPTER_SCHEMA",
    "BOUNDARY_CONTRACT",
    "LINEAR_RETURN_FORMS",
    "build_role",
]
