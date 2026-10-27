"""CPU-S4-4B full-size candidate seam over the accepted 4A factory."""

from __future__ import annotations

import copy
import importlib
import os
from pathlib import Path
from typing import Any, Callable

from tools.hpc_ops import provenance_v2 as pv

from . import contract
from .cache import (
    CACHE_SCHEMA,
    FULLSIZE_POLICY_ID,
    FULLSIZE_POLICY_SOURCE_VERSION,
    FullsizeLinearFactorCache,
    policy_source_version,
    source_identity as cache_source_identity,
)


CANDIDATE_SCHEMA = "filament.cpu_s4_4b_fullsize_linear_cache.candidate.v1"
ALGORITHM_ID = "bk_nee_transfer_kernel_exp_astype.v2"
DEFAULT_VARIANT = "L1_CACHE_FULLSIZE"
FULLSIZE_ROLE = "L1_CACHE_FULLSIZE"
LINEAR_CACHE_ROLE = FULLSIZE_ROLE
LINEAR_CACHE_MODES = ("QUALIFY_BOUNDARY", "PURE_OFF_FULLSIZE")
POOL_CONTRACT = {
    "numba_pool": 48,
    "outer_mask": 24,
    "temporal_workers": 48,
    "spatial_workers": 24,
    "raman_workers": 48,
}
LOCAL_ENGINEERING_CONTRACT = {
    "numba_pool": 1,
    "outer_mask": 1,
    "temporal_workers": 1,
    "spatial_workers": 1,
    "raman_workers": 1,
    "scope": contract.LOCAL_ENGINEERING_SCOPE,
    "hpc_allowed": False,
    "execution_status": "EXPLICIT_FIXED_1_RESOURCE_ADAPTER",
}


class CandidateUnsupported(RuntimeError):
    """The bounded full-size candidate cannot serve the requested route."""


class SourceBindingError(RuntimeError):
    """A frozen 4A dependency or new full-size seam is invalid."""


def _legacy_candidate():
    return importlib.import_module("tools.cpu_s4_4a_linear_cache.candidate")


def _normalise_mode(mode: str) -> str:
    value = str(mode).upper()
    aliases = {
        "PURE_OFF": "PURE_OFF_FULLSIZE",
        "PURE_OFF_SMALLGRID": "PURE_OFF_FULLSIZE",
        "PURE_OFF_REAL8": "PURE_OFF_FULLSIZE",
    }
    value = aliases.get(value, value)
    if value not in LINEAR_CACHE_MODES:
        raise ValueError(f"unsupported CPU-S4-4B mode: {mode!r}")
    return value


def _validate_shape(shape: Any, *, local_engineering: bool) -> tuple[int, int, int]:
    try:
        actual = contract.validate_shape(shape)
    except ValueError as exc:
        raise CandidateUnsupported(str(exc)) from exc
    if local_engineering:
        if actual == contract.FULL_SHAPE:
            raise CandidateUnsupported("FULLSIZE_SHAPE_FORBIDDEN_IN_LOCAL_ENGINEERING")
        if actual not in contract.LOCAL_ENGINEERING_SHAPES:
            raise CandidateUnsupported("LOCAL_ENGINEERING_SHAPE_NOT_ADMITTED")
    elif actual != contract.FULL_SHAPE:
        raise CandidateUnsupported("HPC_FULLSIZE_ROLE_REQUIRES_FULL_SHAPE")
    return actual


def _new_owned_source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    repo = Path(__file__).resolve().parents[3]
    records: dict[str, dict[str, Any]] = {}
    for name in ("__init__.py", "contract.py", "cache.py", "candidate.py"):
        path = (root / name).resolve(strict=True)
        raw = path.read_bytes()
        relative = path.relative_to(repo).as_posix()
        git_blob = pv._git(repo, "rev-parse", "HEAD:" + relative)
        canonical = pv.canonical_lf_sha256(raw)
        if canonical != pv.canonical_lf_sha256(pv._read_git_blob(repo, git_blob)):
            raise SourceBindingError("NEW_CANDIDATE_SOURCE_DIFFERS_FROM_COMMITTED_BLOB")
        records[name] = {
            "path": relative,
            "canonical_lf_sha256": canonical,
            "hash_scope": pv.TRACKED_HASH_SCOPE,
            "classification": "tracked_text",
            "git_blob_oid": git_blob,
        }
    composite = (
        f"{CANDIDATE_SCHEMA}\n"
        + "".join(
            f"{name}:{records[name]['canonical_lf_sha256']}\n"
            for name in sorted(records)
        )
    ).encode("ascii")
    return {
        "schema": "filament.cpu_s4_4b.fullsize_linear_cache.owned_sources.v1",
        "provenance_implementation": "tools.hpc_ops.provenance_v2",
        "files": records,
        "actual_loaded_paths": {name: str(root / name) for name in records},
        "composite_canonical_lf_sha256": pv.raw_sha256(composite),
        "frozen_dependency_is_separate": True,
    }


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


def _replace_empty_cache(
    dispatch: Any,
    *,
    local_engineering: bool,
    configured_shape: tuple[int, int, int],
) -> dict[str, Any]:
    """Replace only the empty 4A cache, retaining the existing state/owner."""

    state = getattr(dispatch, "cpu_s4_4a_state", None)
    if state is None:
        raise SourceBindingError("CPU_S4_4B_STATE_MISSING")
    old_cache = getattr(state, "_cpu_s4_4a_cache", None)
    from tools.cpu_s4_4a_linear_cache.cache import LinearFactorCache
    from tools.cpu_s4_4a_linear_cache import linear_adapter as frozen_adapter
    if type(old_cache) is not LinearFactorCache:
        raise SourceBindingError("CPU_S4_4A_CACHE_MISSING")
    if old_cache._pending is not None:
        raise SourceBindingError("CPU_S4_4A_PENDING_HALF_BEFORE_REPLACEMENT")
    old_snapshot = old_cache.snapshot()
    old_stats = old_snapshot.get("stats", {})
    if any(
        int(old_stats.get(field, 0)) != 0
        for field in ("factor_requests", "entries", "cache_bytes", "persistent_bytes_reserved")
    ):
        raise SourceBindingError("CPU_S4_4A_CACHE_NOT_EMPTY_BEFORE_SCIENCE")
    if getattr(state, "_cpu_s4_4a_cache", None) is not old_cache:
        raise SourceBindingError("CPU_S4_4A_CACHE_OWNER_BINDING_CHANGED")
    owner_identity = str(getattr(old_cache, "owner_identity", ""))
    if owner_identity != f"state:{id(state)}":
        raise SourceBindingError("CPU_S4_4A_CACHE_OWNER_IDENTITY_MISSING")
    old_source_version = str(getattr(state, "_cpu_s4_4a_source_version", ""))
    state._cpu_s4_4a_ensure_cached(state._g_linear_function)
    linear = state.linear_binding()
    expected_source = ":".join((linear["original_guard_producer"]["source"]["canonical_lf_sha256"], linear["transformed_ast_sha256"], pv.canonical_lf_sha256(Path(frozen_adapter.__file__).read_bytes())))
    if old_source_version != expected_source or old_cache.algorithm_source_identity != f"{ALGORITHM_ID}:{expected_source}":
        raise SourceBindingError("CPU_S4_4A_SOURCE_VERSION_OR_ALGORITHM_MISMATCH")
    new_source_version = policy_source_version(old_source_version)
    scope = contract.LOCAL_ENGINEERING_SCOPE if local_engineering else contract.FULLSIZE_SCOPE
    try:
        contract.validate_shape(configured_shape, scope=scope)
    except ValueError as exc:
        raise SourceBindingError(f"CPU_S4_4B_CONFIGURED_SHAPE_SCOPE:{exc}") from exc
    new_cache = FullsizeLinearFactorCache(
        owner_identity=owner_identity,
        algorithm_source_identity=f"CPU_S4_4B:{new_source_version}",
        resource_scope=scope,
        configured_shape=configured_shape,
    )
    state._cpu_s4_4a_source_version = new_source_version
    state._cpu_s4_4a_cache = new_cache
    if getattr(state, "_cpu_s4_4a_cache", None) is old_cache:
        raise SourceBindingError("CPU_S4_4B_CACHE_REPLACEMENT_FAILED")
    snapshot = new_cache.snapshot()
    if any(
        int(snapshot["stats"].get(field, 0)) != 0
        for field in ("factor_requests", "entries", "cache_bytes")
    ):
        raise SourceBindingError("CPU_S4_4B_CACHE_NOT_EMPTY_AFTER_CONSTRUCTION")
    dispatch.cpu_s4_4b_cache_snapshot = state.cache_snapshot
    return {
        "state_id": int(id(state)),
        "owner_identity": owner_identity,
        "old_cache_type": f"{type(old_cache).__module__}.{type(old_cache).__qualname__}",
        "new_cache_type": f"{type(new_cache).__module__}.{type(new_cache).__qualname__}",
        "resource_scope": scope,
        "configured_shape": list(configured_shape),
        "source_version": new_source_version,
        "policy_source_version": FULLSIZE_POLICY_SOURCE_VERSION,
        "snapshot": snapshot,
        "old_cache_snapshot": old_snapshot,
    }


def make_factory(
    variant: str = DEFAULT_VARIANT,
    *,
    kernel_threads: int = 24,
    temporal_workers: int = 48,
    spatial_workers: int = 24,
    raman_workers: int = 48,
    block_size: int = 32,
    local_engineering: bool = False,
    shape: tuple[int, int, int] = contract.FULL_SHAPE,
):
    """Wrap the accepted 4A factory and replace its empty cache before use."""

    normalized = str(variant).upper()
    if normalized not in {DEFAULT_VARIANT, FULLSIZE_ROLE, "L1_LINEAR_FACTOR_CACHE", "G_CONTROLLED_INPUT"}:
        raise ValueError(f"unknown CPU-S4-4B candidate variant: {variant!r}")
    configured_shape = _validate_shape(shape, local_engineering=bool(local_engineering))
    if int(block_size) != 32:
        raise ValueError("CPU-S4-4B freezes block_size=32")
    expected = (1, 1, 1, 1) if local_engineering else (24, 48, 24, 48)
    actual = (int(kernel_threads), int(temporal_workers), int(spatial_workers), int(raman_workers))
    if actual != expected:
        raise ValueError(f"CPU-S4-4B resource contract mismatch: {actual}!={expected}")
    legacy = _legacy_candidate()
    legacy_factory = legacy.make_factory(
        variant="G_CONTROLLED_INPUT",
        kernel_threads=int(kernel_threads),
        temporal_workers=int(temporal_workers),
        spatial_workers=int(spatial_workers),
        raman_workers=int(raman_workers),
        block_size=32,
        local_engineering=bool(local_engineering),
    )

    def factory(frozen_propagator: Callable[..., Any]):
        propagator, dispatch = legacy_factory(frozen_propagator)
        _replace_empty_cache(
            dispatch,
            local_engineering=bool(local_engineering),
            configured_shape=configured_shape,
        )
        factory.last_propagator = propagator
        factory.last_dispatch = dispatch
        factory.last_state = getattr(dispatch, "cpu_s4_4a_state", None)
        return propagator, dispatch

    factory.variant = DEFAULT_VARIANT
    factory.candidate_schema = CANDIDATE_SCHEMA
    factory.candidate_family = "FULLSIZE_LINEAR_FACTOR_CACHE_PRIVATE_STATE_V1"
    factory.local_engineering = bool(local_engineering)
    factory.configured_shape = configured_shape
    factory.policy_source_identity = copy.deepcopy(_new_owned_source_identity())
    factory.frozen_dependency = copy.deepcopy(getattr(legacy_factory, "source_identity", {}))
    factory.last_propagator = None
    factory.last_dispatch = None
    factory.last_state = None
    return factory


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
    shape: tuple[int, int, int] = contract.FULL_SHAPE,
) -> tuple[Any, Any, Any, dict[str, Any], Any, Any]:
    """Build the six-tuple role and replace the empty 4A cache pre-science."""

    mode_name = _normalise_mode(mode)
    configured_shape = _validate_shape(shape, local_engineering=bool(local_engineering))
    expected = _resolve_resources(local_engineering=bool(local_engineering), pool_override=pool_override)
    import numba

    actual_pool = int(numba.config.NUMBA_NUM_THREADS)
    if actual_pool != int(expected["numba_pool"]):
        raise CandidateUnsupported(f"ACTUAL_NUMBA_POOL_MISMATCH:{actual_pool}:{expected['numba_pool']}")
    numba.set_num_threads(int(expected["outer_mask"]))
    legacy = _legacy_candidate()
    legacy_mode = "QUALIFY_BOUNDARY" if mode_name == "QUALIFY_BOUNDARY" else "PURE_OFF_SMALLGRID"
    built = legacy.build_role(
        legacy_mode,
        capture_root=Path(capture_root).resolve(),
        local_engineering=bool(local_engineering),
        pool_override=pool_override,
    )
    if not isinstance(built, tuple) or len(built) != 6:
        raise CandidateUnsupported("CPU_S4_4A_BUILD_ROLE_DID_NOT_RETURN_SIX_TUPLE")
    wrapped, dispatch, clock, old_binding, audit, lease = built
    cache_binding = _replace_empty_cache(
        dispatch,
        local_engineering=bool(local_engineering),
        configured_shape=configured_shape,
    )
    state = getattr(dispatch, "cpu_s4_4a_state", None)
    if state is None:
        raise SourceBindingError("CPU_S4_4B_STATE_MISSING_AFTER_BUILD")
    new_snapshot = dispatch.cpu_s4_4b_cache_snapshot()
    new_owned = _new_owned_source_identity()
    binding = copy.deepcopy(old_binding)
    binding.update(
        {
            "schema": CANDIDATE_SCHEMA,
            "role": FULLSIZE_ROLE,
            "rolescope": "FULLSIZE_PRIVATE_L1",
            "mode": mode_name,
            "candidate": DEFAULT_VARIANT,
            "shape": list(configured_shape),
            "resource_scope": contract.LOCAL_ENGINEERING_SCOPE
            if local_engineering
            else contract.FULLSIZE_SCOPE,
            "pool_contract": copy.deepcopy(expected),
            "actual_pool": int(expected["numba_pool"]),
            "scientific_qualification": False,
            "e_stage1": 0,
            "e_apply": 0,
            "algorithm_id": FULLSIZE_POLICY_ID,
            "frozen_algorithm_id": ALGORITHM_ID,
            "cache_policy_id": FULLSIZE_POLICY_ID,
            "cache_schema": CACHE_SCHEMA,
            "cache_snapshot": copy.deepcopy(new_snapshot),
            "fullsize_cache_snapshot": copy.deepcopy(new_snapshot),
            "fullsize_cache_binding": copy.deepcopy(cache_binding),
            "new_owned_source_identity": copy.deepcopy(new_owned),
            "frozen_dependency": copy.deepcopy(
                getattr(dispatch, "cpu_s4_3g_source_identity", {})
            ),
            "source_version": str(state._cpu_s4_4a_source_version),
            "policy_source_version": FULLSIZE_POLICY_SOURCE_VERSION,
        }
    )
    dispatch.cpu_s4_4b_binding = copy.deepcopy(binding)
    dispatch.cpu_s4_4b_source_identity = {
        "schema": CANDIDATE_SCHEMA,
        "candidate_name": DEFAULT_VARIANT,
        "new_owned_source_identity": copy.deepcopy(new_owned),
        "frozen_dependency": copy.deepcopy(getattr(dispatch, "cpu_s4_3g_source_identity", {})),
        "policy_source_version": FULLSIZE_POLICY_SOURCE_VERSION,
    }
    dispatch.cpu_s4_4b_cache_snapshot = state.cache_snapshot
    dispatch.cpu_s4_4b_candidate_schema = CANDIDATE_SCHEMA
    dispatch.cpu_s4_4b_shape = tuple(configured_shape)
    return wrapped, dispatch, clock, binding, audit, lease


def source_identity() -> dict[str, Any]:
    legacy = _legacy_candidate()
    return {
        "schema": CANDIDATE_SCHEMA,
        "candidate_name": DEFAULT_VARIANT,
        "algorithm_id": FULLSIZE_POLICY_ID,
        "owned_namespace": str(Path(__file__).resolve().parent),
        "owned_source_identity": _new_owned_source_identity(),
        "policy_source_identity": copy.deepcopy(cache_source_identity()),
        "frozen_dependency": copy.deepcopy(legacy.source_identity()),
        "reference_cache_module": "NOT_IMPORTED_OR_LOADED_BY_G3_REF",
    }


__all__ = [
    "CANDIDATE_SCHEMA",
    "ALGORITHM_ID",
    "CandidateUnsupported",
    "DEFAULT_VARIANT",
    "FULLSIZE_ROLE",
    "LINEAR_CACHE_MODES",
    "LINEAR_CACHE_ROLE",
    "LOCAL_ENGINEERING_CONTRACT",
    "POOL_CONTRACT",
    "SourceBindingError",
    "build_role",
    "make_factory",
    "source_identity",
]
