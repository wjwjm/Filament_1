"""G3 small-grid candidate with a typed private lifecycle identity.

The frozen G3 factory is loaded as a private implementation dependency.  A
temporary builder hook installs the new state subclass into that private E
module; no frozen source file or production module is edited.
"""

from __future__ import annotations

import copy
import importlib
import os
import sys
from pathlib import Path
from typing import Any, Callable

from . import controlled_callsite
from .factory_binding import private_factory
from tools.hpc_ops import provenance_v2 as pv


CANDIDATE_SCHEMA = "filament.cpu_s4_3g3_smallgrid.controlled_lifetime.v1"
DEFAULT_VARIANT = "G3_SG_CONTROLLED_LIFECYCLE_V1"
SMALLGRID_ROLES = ("I0_SMALL", "G2_SMALL", "G3_SMALL")
SMALLGRID_MODES = ("QUALIFY_BOUNDARY", "PURE_OFF_SMALLGRID")
POOL_CONTRACT = {
    "I0_SMALL": {"numba_pool": 24, "outer_mask": 24, "temporal_workers": 48, "spatial_workers": 24},
    "G2_SMALL": {"numba_pool": 48, "outer_mask": 24, "temporal_workers": 48, "spatial_workers": 24},
    "G3_SMALL": {"numba_pool": 48, "outer_mask": 24, "temporal_workers": 48, "spatial_workers": 24},
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


class SourceBindingError(RuntimeError):
    """A frozen dependency identity or private module binding is invalid."""


class CandidateUnsupported(RuntimeError):
    """The requested role/mode is outside this bounded candidate."""


class HandoffRejected(RuntimeError):
    """A controlled lifecycle handoff was rejected."""


class StageBindingError(ValueError):
    """A strict stage or payload identity failed."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def unoptimized() -> None:
    if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE"):
        raise RuntimeError("SMALLGRID_OPTIMIZATION_FORBIDDEN")


def _frozen_paths() -> dict[str, Path]:
    root = _repo_root() / "Filament_python" / "tools"
    return {
        "f": root / "cpu_s4_3f" / "candidate.py",
        "e": root / "cpu_s4_3e" / "candidate.py",
        "d": root / "cpu_s4_3d" / "variants.py",
        "r8": root / "cpu_s4_3c" / "candidate.py",
    }


def _legacy_candidate():
    return importlib.import_module("tools.cpu_s4_3g.candidate")


def _owned_source_identity() -> dict[str, Any]:
    """Return the independent raw identity of this new adapter namespace."""
    root = Path(__file__).resolve().parent
    records: dict[str, dict[str, Any]] = {}
    for name in ("__init__.py", "candidate.py", "controlled_callsite.py", "adapter.py", "factory_binding.py", "runtime_adapter.py"):
        path = root / name
        raw = path.read_bytes()
        records[name] = {
            "path": str(path),
            "size_bytes": int(len(raw)),
            "raw_sha256": pv.raw_sha256(raw),
            "hash_scope": "raw_bytes",
            "classification": "external",
        }
    producer_module_identity = controlled_callsite.__name__
    composite_input = ("producer_module:" + producer_module_identity + "\n" + "".join(
        f"{name}:{records[name]['raw_sha256']}\n" for name in sorted(records)
    )).encode("utf-8")
    return {
        "schema": "filament.cpu_s4_3g3_smallgrid.owned_sources.v1",
        "files": records,
        "producer_module_identity": producer_module_identity,
        "composite_raw_sha256": pv.raw_sha256(composite_input),
    }


def _build_g3_factory(
    *,
    frozen_f_path: str | Path | None,
    frozen_e_path: str | Path | None,
    frozen_d_path: str | Path | None,
    frozen_r8_path: str | Path | None,
    kernel_threads: int,
    temporal_workers: int,
    spatial_workers: int,
    raman_workers: int,
    block_size: int,
    observe: Callable[[str, Any], None] | None,
    local_engineering: bool,
):
    if observe is not None:
        raise CandidateUnsupported("observer_or_trace_not_supported_for_G3_SMALL")
    if (int(kernel_threads), int(temporal_workers), int(spatial_workers), int(raman_workers), int(block_size)) != (24, 48, 24, 48, 32):
        raise ValueError("G3_SMALL freezes kernel_threads=24, temporal_workers=48, spatial_workers=24, raman_workers=48, block_size=32")
    legacy = _legacy_candidate()
    default_paths = _frozen_paths()
    paths = {
        "f": Path(frozen_f_path) if frozen_f_path is not None else default_paths["f"],
        "e": Path(frozen_e_path) if frozen_e_path is not None else default_paths["e"],
        "d": Path(frozen_d_path) if frozen_d_path is not None else default_paths["d"],
        "r8": Path(frozen_r8_path) if frozen_r8_path is not None else default_paths["r8"],
    }
    try:
        for name, path in paths.items():
            if path.resolve(strict=True) != default_paths[name].resolve(strict=True):
                raise SourceBindingError("EXPLICIT_FROZEN_PATH_DIFFERS_FROM_BOUND_CLOSURE")
        legacy_factory = private_factory(
            legacy, owned_identity=_owned_source_identity(),
            lifecycle_builder=controlled_callsite.make_controlled_state_class,
            local_engineering=local_engineering, candidate_schema=CANDIDATE_SCHEMA,
        )
    except Exception as exc:
        raise SourceBindingError(f"G3_SMALL_PRIVATE_FACTORY:{type(exc).__name__}:{exc}") from exc

    def factory(frozen_propagator: Callable[..., Any]):
        propagator, dispatch = legacy_factory(frozen_propagator)
        state = getattr(dispatch, "cpu_s4_3g_state", None)
        if state is None or state.__class__.__module__ != controlled_callsite.__name__:
            raise SourceBindingError("G3_SMALL_FACTORY_LIFECYCLE_STATE_MISSING")
        route = copy.deepcopy(getattr(dispatch, "cpu_s4_3g_route_contract", {}))
        route.update(
            {
                "candidate": DEFAULT_VARIANT,
                "candidate_schema": CANDIDATE_SCHEMA,
                "lifecycle_identity": "controlled_lifetime_v1",
                "controlled_hash_roles": [
                    "Omega_stage1",
                    "Omega_apply",
                    "Omega_stage2",
                ],
                "expected_controlled_E_stage1": 0,
                "expected_controlled_E_apply": 0,
                "public_trust_kwargs": "forbidden",
            }
        )
        source = copy.deepcopy(getattr(dispatch, "cpu_s4_3g_source_identity", {}))
        source.update(
            {
                "schema": CANDIDATE_SCHEMA,
                "candidate_name": DEFAULT_VARIANT,
                "lifecycle_identity": "controlled_lifetime_v1",
                "owned_namespace": str(Path(__file__).resolve().parent),
                "owned_source_identity": copy.deepcopy(_owned_source_identity()),
            }
        )
        dispatch.cpu_s4_3g_variant = DEFAULT_VARIANT
        dispatch.cpu_s4_3g_route_contract = route
        dispatch.cpu_s4_3g_source_identity = source
        dispatch.cpu_s4_3g_candidate_schema = CANDIDATE_SCHEMA
        dispatch.cpu_s4_3g_lifecycle_identity = "controlled_lifetime_v1"
        dispatch.cpu_s4_3g_smallgrid_pool_contract = copy.deepcopy(LOCAL_ENGINEERING_CONTRACT if local_engineering else POOL_CONTRACT["G3_SMALL"])
        execution = state.metrics()
        execution.update(
            {
                "schema": CANDIDATE_SCHEMA,
                "candidate_name": DEFAULT_VARIANT,
                "lifecycle_identity": "controlled_lifetime_v1",
            }
        )
        dispatch.cpu_s4_3g_execution = copy.deepcopy(execution)
        propagator._cpu_s4_3g_execution = copy.deepcopy(execution)
        factory.last_propagator = propagator
        factory.last_dispatch = dispatch
        factory.last_state = state
        factory.last_full_heun = state.full_heun
        factory.last_source_identity = copy.deepcopy(source)
        return propagator, dispatch

    factory.variant = DEFAULT_VARIANT
    factory.candidate_schema = CANDIDATE_SCHEMA
    factory.candidate_family = "CONTROLLED_LIFETIME_IDENTITY_V1"
    factory.smallgrid_pool_contract = copy.deepcopy(LOCAL_ENGINEERING_CONTRACT if local_engineering else POOL_CONTRACT["G3_SMALL"])
    factory.smallgrid_construction = copy.deepcopy(legacy_factory.smallgrid_construction)
    factory.lifecycle_identity_schema = "controlled_lifetime_v1"
    factory.owned_source_identity = _owned_source_identity()
    factory.source_identity = {
        "schema": CANDIDATE_SCHEMA,
        "candidate_name": DEFAULT_VARIANT,
        "owned_namespace": str(Path(__file__).resolve().parent),
        "owned_source_identity": copy.deepcopy(factory.owned_source_identity),
        "frozen_identity": copy.deepcopy(getattr(legacy_factory, "source_identity", {})),
        "lifecycle_identity": "controlled_lifetime_v1",
    }
    factory.frozen_math_contract = {
        "linear_model": "bk_nee",
        "operator_mode": "full_isaacs_eq27",
        "raman_method": "iir",
        "iir_sampling": "exact_piecewise_linear",
        "integrator": "heun",
        "split_order": "strang",
        "dtype": "complex128",
        "layout": "native_C",
        "omega_hash_roles": ["Omega_stage1", "Omega_apply", "Omega_stage2"],
        "e_stage1": 0,
        "e_apply": 0,
    }
    return factory


def make_factory(
    variant: str = DEFAULT_VARIANT,
    *,
    frozen_f_path: str | Path | None = None,
    frozen_e_path: str | Path | None = None,
    frozen_d_path: str | Path | None = None,
    frozen_r8_path: str | Path | None = None,
    kernel_threads: int = 24,
    temporal_workers: int = 48,
    spatial_workers: int = 24,
    raman_workers: int = 48,
    block_size: int = 32,
    observe: Callable[[str, Any], None] | None = None,
    local_engineering: bool = False,
):
    """Build the only new G3 candidate; source paths remain explicit."""
    unoptimized()
    normalized = str(variant).upper()
    if normalized not in {DEFAULT_VARIANT, "G3_SMALL", "G_CONTROLLED_INPUT"}:
        raise ValueError(f"unknown small-grid candidate variant: {variant!r}")
    return _build_g3_factory(
        frozen_f_path=frozen_f_path,
        frozen_e_path=frozen_e_path,
        frozen_d_path=frozen_d_path,
        frozen_r8_path=frozen_r8_path,
        kernel_threads=kernel_threads,
        temporal_workers=temporal_workers,
        spatial_workers=spatial_workers,
        raman_workers=raman_workers,
        block_size=block_size,
        observe=observe,
        local_engineering=bool(local_engineering),
    )


def source_identity() -> dict[str, Any]:
    """Return candidate and frozen source identities without scientific work."""
    unoptimized()
    paths = _frozen_paths()
    legacy = _legacy_candidate()
    return {
        "schema": CANDIDATE_SCHEMA,
        "candidate": DEFAULT_VARIANT,
        "owned_namespace": str(Path(__file__).resolve().parent),
        "owned_source_identity": _owned_source_identity(),
        "frozen_paths": {name: str(path.resolve(strict=True)) for name, path in paths.items()},
        "frozen_identity": legacy.source_identity(
            frozen_f_path=paths["f"],
            frozen_e_path=paths["e"],
            frozen_d_path=paths["d"],
            frozen_r8_path=paths["r8"],
        ),
        "lifecycle_identity": "controlled_lifetime_v1",
        "mathematics_unchanged": True,
    }


__all__ = [
    "CANDIDATE_SCHEMA",
    "DEFAULT_VARIANT",
    "SMALLGRID_MODES",
    "SMALLGRID_ROLES",
    "POOL_CONTRACT",
    "LOCAL_ENGINEERING_CONTRACT",
    "CandidateUnsupported",
    "HandoffRejected",
    "SourceBindingError",
    "StageBindingError",
    "make_factory",
    "source_identity",
]
