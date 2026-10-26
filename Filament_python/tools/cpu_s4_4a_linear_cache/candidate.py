"""CPU-S4-4A L1 linear-factor-cache candidate factory."""

from __future__ import annotations

import copy
import importlib
import os
from pathlib import Path
from typing import Any, Callable

from tools.hpc_ops import provenance_v2 as pv

from .linear_adapter import ADAPTER_SCHEMA, ALGORITHM_ID, make_controlled_state_class
from tools.cpu_s4_3g3_smallgrid.factory_binding import private_factory


CANDIDATE_SCHEMA = "filament.cpu_s4_4a_linear_cache.candidate.v1"
DEFAULT_VARIANT = "L1_LINEAR_FACTOR_CACHE_V1"
LINEAR_CACHE_ROLE = "L1_LINEAR_FACTOR_CACHE"
LINEAR_CACHE_MODES = ("QUALIFY_BOUNDARY", "PURE_OFF_SMALLGRID")
POOL_CONTRACT = {
    "numba_pool": 48,
    "outer_mask": 24,
    "temporal_workers": 48,
    "spatial_workers": 24,
}
LOCAL_ENGINEERING_CONTRACT = {
    "numba_pool": 1,
    "outer_mask": 1,
    "temporal_workers": 1,
    "spatial_workers": 1,
    "scope": "LOCAL_ENGINEERING_ONLY",
    "hpc_allowed": False,
    "execution_status": "EXPLICIT_FIXED_1_RESOURCE_ADAPTER",
}


class CandidateUnsupported(RuntimeError):
    """The bounded L1 candidate cannot serve the requested route."""


class SourceBindingError(RuntimeError):
    """A frozen dependency or newly owned source identity is invalid."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _owned_source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    files: dict[str, dict[str, Any]] = {}
    for name in ("__init__.py", "cache.py", "linear_adapter.py", "candidate.py"):
        path = root / name
        raw = path.read_bytes()
        files[name] = {
            "path": str(path.resolve()),
            "size_bytes": int(len(raw)),
            "raw_sha256": pv.raw_sha256(raw),
            "hash_scope": "raw_bytes",
            "classification": "external",
        }
    composite = ("producer_module:tools.cpu_s4_4a_linear_cache.linear_adapter\n" + "".join(
        f"{name}:{files[name]['raw_sha256']}\n" for name in sorted(files)
    )).encode("utf-8")
    return {
        "schema": "filament.cpu_s4_4a_linear_cache.owned_sources.v1",
        "files": files,
        "producer_module_identity": "tools.cpu_s4_4a_linear_cache.linear_adapter",
        "composite_raw_sha256": pv.raw_sha256(composite),
    }


def _legacy_candidate():
    return importlib.import_module("tools.cpu_s4_3g.candidate")


def _normalise_mode(mode: str) -> str:
    value = str(mode).upper()
    aliases = {"PURE_OFF": "PURE_OFF_SMALLGRID", "PURE_OFF_REAL8": "PURE_OFF_SMALLGRID"}
    value = aliases.get(value, value)
    if value not in LINEAR_CACHE_MODES:
        raise ValueError(f"unsupported CPU-S4-4A mode: {mode!r}")
    return value


def _resolve_resources(*, local_engineering: bool, pool_override: int | None) -> dict[str, Any]:
    if local_engineering:
        if os.environ.get("SLURM_JOB_ID"):
            raise CandidateUnsupported("LOCAL_ENGINEERING_ONLY_FORBIDDEN_ON_SLURM")
        if pool_override != 1:
            raise ValueError("local_engineering requires explicit pool_override=1")
        return copy.deepcopy(LOCAL_ENGINEERING_CONTRACT)
    if pool_override is not None:
        raise ValueError("pool_override requires local_engineering=True")
    return copy.deepcopy(POOL_CONTRACT)


def make_factory(
    variant: str = DEFAULT_VARIANT,
    *,
    kernel_threads: int = 24,
    temporal_workers: int = 48,
    spatial_workers: int = 24,
    raman_workers: int = 48,
    block_size: int = 32,
    local_engineering: bool = False,
):
    """Build a fresh L1 factory with an empty per-state factor cache."""

    normalized = str(variant).upper()
    if normalized not in {DEFAULT_VARIANT, LINEAR_CACHE_ROLE, "G_CONTROLLED_INPUT"}:
        raise ValueError(f"unknown CPU-S4-4A candidate variant: {variant!r}")
    if int(block_size) != 32:
        raise ValueError("CPU-S4-4A freezes block_size=32")
    actual = (int(kernel_threads), int(temporal_workers), int(spatial_workers), int(raman_workers))
    if local_engineering:
        expected = (1, 1, 1, 1)
        if actual != expected:
            raise ValueError(f"CPU-S4-4A local resource contract mismatch: {actual}")
    else:
        # The frozen G factory's kernel-thread contract is W24 while its
        # outer Numba pool remains W48.
        expected = (24, 48, 24, 48)
    if not local_engineering and actual != expected:
        raise ValueError(f"CPU-S4-4A resource contract mismatch: {actual}!={expected}")
    legacy = _legacy_candidate()
    owned = _owned_source_identity()
    legacy_factory = private_factory(
        legacy,
        owned_identity=owned,
        lifecycle_builder=make_controlled_state_class,
        local_engineering=bool(local_engineering),
        candidate_schema=CANDIDATE_SCHEMA,
    )

    def factory(frozen_propagator: Callable[..., Any]):
        propagator, dispatch = legacy_factory(frozen_propagator)
        state = getattr(dispatch, "cpu_s4_3g_state", None)
        if state is None or not hasattr(state, "cache_snapshot"):
            raise SourceBindingError("L1_LINEAR_CACHE_STATE_MISSING")
        # Build and verify the separate execution producer while the factory
        # is inside the caller's main application clock.  The inherited G/SG
        # guard still retains the original producer identity.
        original_linear = getattr(state, "_g_linear_function", None)
        if not callable(original_linear):
            raise SourceBindingError("L1_LINEAR_CACHE_ORIGINAL_PRODUCER_MISSING")
        state._cpu_s4_4a_ensure_cached(original_linear)
        source = copy.deepcopy(getattr(dispatch, "cpu_s4_3g_source_identity", {}))
        source.update(
            {
                "schema": CANDIDATE_SCHEMA,
                "candidate_name": DEFAULT_VARIANT,
                "algorithm_id": ALGORITHM_ID,
                "owned_namespace": str(Path(__file__).resolve().parent),
                "owned_source_identity": copy.deepcopy(owned),
                "cached_execution_producer": copy.deepcopy(state.linear_binding()),
            }
        )
        dispatch.cpu_s4_3g_source_identity = source
        dispatch.cpu_s4_3g_variant = DEFAULT_VARIANT
        dispatch.cpu_s4_3g_candidate_schema = CANDIDATE_SCHEMA
        dispatch.cpu_s4_4a_linear_binding = copy.deepcopy(state.linear_binding())
        dispatch.cpu_s4_4a_cache_snapshot = state.cache_snapshot
        dispatch.cpu_s4_4a_algorithm_id = ALGORITHM_ID
        dispatch.cpu_s4_4a_candidate_schema = CANDIDATE_SCHEMA
        dispatch.cpu_s4_4a_state = state
        factory.last_propagator = propagator
        factory.last_dispatch = dispatch
        factory.last_state = state
        factory.last_source_identity = copy.deepcopy(source)
        return propagator, dispatch

    factory.variant = DEFAULT_VARIANT
    factory.candidate_schema = CANDIDATE_SCHEMA
    factory.candidate_family = "LINEAR_FACTOR_CACHE_PRIVATE_STATE_V1"
    factory.owned_source_identity = copy.deepcopy(owned)
    factory.source_identity = {
        "schema": CANDIDATE_SCHEMA,
        "candidate_name": DEFAULT_VARIANT,
        "algorithm_id": ALGORITHM_ID,
        "owned_namespace": str(Path(__file__).resolve().parent),
        "owned_source_identity": copy.deepcopy(owned),
        "frozen_dependency": copy.deepcopy(getattr(legacy_factory, "source_identity", {})),
    }
    factory.local_engineering = bool(local_engineering)
    factory.smallgrid_construction = copy.deepcopy(getattr(legacy_factory, "smallgrid_construction", {}))
    factory.last_propagator = None
    factory.last_dispatch = None
    factory.last_state = None
    factory.last_source_identity = copy.deepcopy(factory.source_identity)
    return factory


def source_identity() -> dict[str, Any]:
    legacy = _legacy_candidate()
    return {
        "schema": CANDIDATE_SCHEMA,
        "candidate_name": DEFAULT_VARIANT,
        "algorithm_id": ALGORITHM_ID,
        "owned_namespace": str(Path(__file__).resolve().parent),
        "owned_source_identity": _owned_source_identity(),
        "frozen_dependency": copy.deepcopy(legacy.source_identity()),
        "reference_cache_module": "NOT_IMPORTED_OR_LOADED_BY_G3_REF",
    }


def _bind_factory(factory: Callable[..., Any]) -> tuple[Any, Any]:
    propagate_module = importlib.import_module("KHz_filament.propagate")
    frozen_propagator = getattr(propagate_module, "propagate_one_pulse", None)
    if not callable(frozen_propagator):
        raise CandidateUnsupported("FROZEN_PROPAGATOR_ENTRY_MISSING")
    propagator, dispatch = factory(frozen_propagator)
    if not callable(propagator) or dispatch is None:
        raise CandidateUnsupported("ROLE_FACTORY_BINDING_INCOMPLETE")
    return propagator, dispatch


def build_role(
    mode: str,
    *,
    capture_root: str | Path,
    local_engineering: bool = False,
    pool_override: int | None = None,
) -> tuple[Any, Any, Any, dict[str, Any], Any, Any]:
    """Build only the L1 role using the SG six-tuple adapter contract."""

    mode_name = _normalise_mode(mode)
    expected = _resolve_resources(local_engineering=bool(local_engineering), pool_override=pool_override)
    import numba

    actual_pool = int(numba.config.NUMBA_NUM_THREADS)
    if actual_pool != int(expected["numba_pool"]):
        raise CandidateUnsupported(f"ACTUAL_NUMBA_POOL_MISMATCH:{actual_pool}:{expected['numba_pool']}")
    numba.set_num_threads(int(expected["outer_mask"]))
    settings = dict(
        kernel_threads=int(expected["numba_pool"] if local_engineering else 24),
        temporal_workers=int(expected["temporal_workers"]),
        spatial_workers=int(expected["spatial_workers"]),
        raman_workers=int(expected.get("raman_workers", 1 if local_engineering else 48)),
    )
    factory = make_factory(local_engineering=bool(local_engineering), **settings)
    propagator, dispatch = _bind_factory(factory)
    capture = Path(capture_root).resolve()
    binding = {
        "adapter_schema": ADAPTER_SCHEMA,
        "schema": CANDIDATE_SCHEMA,
        "role": LINEAR_CACHE_ROLE,
        "mode": mode_name,
        "capture_root": str(capture),
        "candidate": DEFAULT_VARIANT,
        "algorithm_id": ALGORITHM_ID,
        "pool_contract": copy.deepcopy(expected),
        "actual_pool": int(expected["numba_pool"]),
        "resource_scope": "LOCAL_ENGINEERING_ONLY" if local_engineering else "HPC_FIXED_CONTRACT",
        "scientific_qualification": False,
        "e_stage1": 0,
        "e_apply": 0,
        "omega_hash_roles": ["Omega_stage1", "Omega_apply", "Omega_stage2"],
        "cache_snapshot": dispatch.cpu_s4_4a_cache_snapshot(),
        "linear_binding": copy.deepcopy(dispatch.cpu_s4_4a_linear_binding),
    }
    # The SG runtime adapter supplies the existing six-tuple collector/clock
    # boundary.  It receives the expanded owned identity, so its strict
    # external binding remains separate from the old G2 identity.
    from tools.cpu_s4_3g3_smallgrid.runtime_adapter import attach

    wrapped, actual_dispatch, clock, wrapper_binding, audit, lease = attach(
        propagator,
        dispatch,
        role=LINEAR_CACHE_ROLE,
        mode=mode_name,
        capture_root=capture,
        owned_identity=_owned_source_identity(),
    )
    if actual_dispatch is not dispatch:
        raise SourceBindingError("L1_RUNTIME_DISPATCH_BINDING_CHANGED")
    binding["runtime_wrapper"] = copy.deepcopy(wrapper_binding)
    binding["boundary_capture"] = (
        "SOURCE_CODE_BOUND_G2_COLLECTOR" if mode_name == "QUALIFY_BOUNDARY" else "DISABLED"
    )
    dispatch.cpu_s4_4a_cache_snapshot = dispatch.cpu_s4_4a_cache_snapshot
    return wrapped, dispatch, clock, binding, audit, lease


__all__ = [
    "ADAPTER_SCHEMA",
    "CANDIDATE_SCHEMA",
    "CandidateUnsupported",
    "DEFAULT_VARIANT",
    "LINEAR_CACHE_MODES",
    "LINEAR_CACHE_ROLE",
    "LOCAL_ENGINEERING_CONTRACT",
    "POOL_CONTRACT",
    "SourceBindingError",
    "build_role",
    "make_factory",
    "source_identity",
]
