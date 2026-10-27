"""CPU-S4-5A L1 profile factory over the accepted CPU-S4-4A candidate."""

from __future__ import annotations

import copy
import importlib
import os
from pathlib import Path
from types import FunctionType
from typing import Any, Callable

from tools.hpc_ops import provenance_v2 as pv

from .instrument import (
    INSTRUMENT_SCHEMA,
    InstrumentationBindingError,
    make_instrumented_state_class,
    verify_bindings,
    install_propagation_wrappers,
    bind_active_propagation,
)
from .timing import TimingRecorder


CANDIDATE_SCHEMA = "filament.cpu_s4_5a_l1_residual.candidate.v1"
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
    """The bounded S45A role cannot serve the requested route."""


class SourceBindingError(RuntimeError):
    """A frozen dependency or private instrumentation identity is invalid."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _owned_source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    records: dict[str, dict[str, Any]] = {}
    for name in ("__init__.py", "candidate.py", "instrument.py", "timing.py", "measurement.py", "entry.py"):
        path = root / name
        raw = path.read_bytes()
        records[name] = {
            "path": str(path.resolve()),
            "size_bytes": int(len(raw)),
            "raw_sha256": pv.raw_sha256(raw),
            "hash_scope": "raw_bytes",
            "classification": "external",
        }
    composite = (
        "producer_module:tools.cpu_s4_5a_l1_residual\n"
        + "".join(f"{name}:{records[name]['raw_sha256']}\n" for name in sorted(records))
    ).encode("utf-8")
    return {
        "schema": "filament.cpu_s4_5a.l1_owned_sources.v1",
        "files": records,
        "producer_module_identity": "tools.cpu_s4_5a_l1_residual",
        "composite_raw_sha256": pv.raw_sha256(composite),
        "instrumentation_schema": INSTRUMENT_SCHEMA,
    }


def _normalise_mode(mode: str) -> str:
    value = str(mode).upper()
    value = {"PURE_OFF": "PURE_OFF_SMALLGRID", "PURE_OFF_REAL8": "PURE_OFF_SMALLGRID"}.get(value, value)
    if value not in LINEAR_CACHE_MODES:
        raise ValueError(f"unsupported CPU-S4-5A mode: {mode!r}")
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


def _clone_make_factory(*, profile: bool, recorder: TimingRecorder | None) -> Callable[..., Any]:
    """Clone only the accepted 4A factory, replacing its state builder seam."""

    source_module = importlib.import_module("tools.cpu_s4_4a_linear_cache.candidate")
    source = getattr(source_module, "make_factory", None)
    if not callable(source):
        raise SourceBindingError("S44A_MAKE_FACTORY_MISSING")
    path=(_repo_root()/'Filament_python/tools/cpu_s4_4a_linear_cache/candidate.py').resolve(strict=True)
    if Path(str(source_module.__file__)).resolve()!=path or source.__globals__ is not vars(source_module):
        raise SourceBindingError("S44A_PARENT_FACTORY_PATH_OR_GLOBALS_MISMATCH")
    from tools.cpu_s4_checkpoint_support import load_g2_adapter
    parent_binding=load_g2_adapter()._method_binding(source)
    relative=path.relative_to(_repo_root()).as_posix()
    if pv._git(_repo_root(),'rev-parse','HEAD:'+relative)!=pv._git(_repo_root(),'rev-parse','c2644d21f7f8fd201b5264d0245c7b4a37643899:'+relative):
        raise SourceBindingError("S44A_PARENT_FACTORY_FROZEN_BLOB_MISMATCH")
    namespace = dict(getattr(source, "__globals__", {}))
    if not namespace:
        raise SourceBindingError("S44A_MAKE_FACTORY_GLOBALS_MISSING")

    def builder(
        e_module: Any,
        *,
        candidate_schema: str,
        candidate_unsupported: type[BaseException],
        handoff_rejected: type[BaseException],
    ) -> type:
        return make_instrumented_state_class(
            e_module,
            candidate_schema=candidate_schema,
            candidate_unsupported=candidate_unsupported,
            handoff_rejected=handoff_rejected,
            profile=bool(profile),
            recorder=recorder,
        )

    namespace.update(
        {
            "_owned_source_identity": _owned_source_identity,
            "make_controlled_state_class": builder,
        }
    )
    clone = FunctionType(source.__code__, namespace, "make_factory", source.__defaults__, source.__closure__)
    clone.__kwdefaults__ = source.__kwdefaults__
    clone.__annotations__ = source.__annotations__
    clone.__module__ = __name__
    clone.__qualname__ = "make_factory"
    clone.cpu_s45a_parent_factory_binding=parent_binding
    return clone


def make_factory(
    variant: str = DEFAULT_VARIANT,
    *,
    kernel_threads: int = 24,
    temporal_workers: int = 48,
    spatial_workers: int = 24,
    raman_workers: int = 48,
    block_size: int = 32,
    local_engineering: bool = False,
    profile: bool = False,
    recorder: TimingRecorder | None = None,
):
    """Build the accepted 4A L1 cache factory with optional scalar timing."""

    normalized = str(variant).upper()
    if normalized not in {DEFAULT_VARIANT, LINEAR_CACHE_ROLE, "G_CONTROLLED_INPUT"}:
        raise ValueError(f"unknown CPU-S4-5A candidate variant: {variant!r}")
    if int(block_size) != 32:
        raise ValueError("CPU-S4-5A freezes block_size=32")
    expected = (1, 1, 1, 1) if local_engineering else (24, 48, 24, 48)
    actual = (int(kernel_threads), int(temporal_workers), int(spatial_workers), int(raman_workers))
    if actual != expected:
        raise ValueError(f"CPU-S4-5A resource contract mismatch: {actual}!={expected}")
    cloned = _clone_make_factory(profile=bool(profile), recorder=recorder)
    factory = cloned(
        normalized,
        kernel_threads=int(kernel_threads),
        temporal_workers=int(temporal_workers),
        spatial_workers=int(spatial_workers),
        raman_workers=int(raman_workers),
        block_size=int(block_size),
        local_engineering=bool(local_engineering),
    )
    # The cloned factory already exposes the accepted cache identity.  Add a
    # private S45A overlay without changing its numerical algorithm identity.
    factory.cpu_s4_5a_profile = bool(profile)
    factory.cpu_s4_5a_recorder = recorder
    factory.cpu_s4_5a_instrumentation_schema = INSTRUMENT_SCHEMA
    factory.cpu_s4_5a_owned_source_identity = copy.deepcopy(_owned_source_identity())
    factory.cpu_s45a_parent_factory_binding=copy.deepcopy(cloned.cpu_s45a_parent_factory_binding)
    factory.candidate_schema = CANDIDATE_SCHEMA
    factory.source_identity = {
        "schema": CANDIDATE_SCHEMA,
        "candidate_name": DEFAULT_VARIANT,
        "parent_candidate": "tools.cpu_s4_4a_linear_cache.candidate",
        "parent_algorithm": "L1_LINEAR_FACTOR_CACHE_V1",
        "profile": bool(profile),
        "instrumentation": INSTRUMENT_SCHEMA,
        "owned_source_identity": copy.deepcopy(_owned_source_identity()),
        "frozen_dependency": copy.deepcopy(getattr(factory, "source_identity", {})),
    }
    return factory


def source_identity() -> dict[str, Any]:
    parent = importlib.import_module("tools.cpu_s4_4a_linear_cache.candidate")
    return {
        "schema": CANDIDATE_SCHEMA,
        "candidate_name": DEFAULT_VARIANT,
        "parent_candidate": "tools.cpu_s4_4a_linear_cache.candidate",
        "parent_source_identity": copy.deepcopy(parent.source_identity()),
        "owned_source_identity": _owned_source_identity(),
        "instrumentation": INSTRUMENT_SCHEMA,
        "new_math_algorithm": False,
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
    profile: bool = False,
    recorder: TimingRecorder | None = None,
) -> tuple[Any, Any, Any, dict[str, Any], Any, Any]:
    """Build one fixed L1 role using the existing SG six-tuple adapter."""

    mode_name = _normalise_mode(mode)
    expected = _resolve_resources(local_engineering=bool(local_engineering), pool_override=pool_override)
    import numba

    actual_pool = int(numba.config.NUMBA_NUM_THREADS)
    if actual_pool != int(expected["numba_pool"]):
        raise CandidateUnsupported(f"ACTUAL_NUMBA_POOL_MISMATCH:{actual_pool}:{expected['numba_pool']}")
    numba.set_num_threads(int(expected["outer_mask"]))
    if recorder is None:
        recorder = TimingRecorder(enabled=bool(profile))
    if type(recorder) is not TimingRecorder or recorder.enabled is not bool(profile):
        raise CandidateUnsupported("FIXED_SCALAR_RECORDER_TYPE_AND_SWITCH_REQUIRED")
    settings = dict(
        kernel_threads=int(expected["numba_pool"] if local_engineering else 24),
        temporal_workers=int(expected["temporal_workers"]),
        spatial_workers=int(expected["spatial_workers"]),
        raman_workers=int(expected.get("raman_workers", 1 if local_engineering else 48)),
    )
    factory = make_factory(
        local_engineering=bool(local_engineering),
        profile=bool(profile),
        recorder=recorder,
        **settings,
    )
    propagator, dispatch = _bind_factory(factory)
    state = getattr(dispatch, "cpu_s4_3g_state", None)
    if state is None:
        raise SourceBindingError("S45A_STATE_MISSING")
    install_propagation_wrappers(propagator, state)
    binding_check = verify_bindings(state)
    capture = Path(capture_root).resolve()
    binding = {
        "schema": CANDIDATE_SCHEMA,
        "instrumentation_schema": INSTRUMENT_SCHEMA,
        "role": "L1_PROFILE" if profile else "L1_OFF",
        "mode": mode_name,
        "capture_root": str(capture),
        "candidate": DEFAULT_VARIANT,
        "pool_contract": copy.deepcopy(expected),
        "actual_pool": int(expected["numba_pool"]),
        "resource_scope": "LOCAL_ENGINEERING_ONLY" if local_engineering else "HPC_FIXED_CONTRACT",
        "profile": bool(profile),
        "scientific_qualification": False,
        "cache_algorithm": "UNCHANGED_S44A_LINEAR_FACTOR_CACHE",
        "binding_check": copy.deepcopy(binding_check),
        "actual_parent_factory_binding":copy.deepcopy(factory.cpu_s45a_parent_factory_binding),
        "instrumentation_bindings": copy.deepcopy(getattr(state, "_cpu_s45a_bindings", {})),
    }
    from tools.cpu_s4_3g3_smallgrid.runtime_adapter import attach

    wrapped, actual_dispatch, clock, wrapper_binding, audit, lease = attach(
        propagator,
        dispatch,
        role="L1_PROFILE" if profile else "L1_OFF",
        mode=mode_name,
        capture_root=capture,
        owned_identity=_owned_source_identity(),
    )
    if actual_dispatch is not dispatch:
        raise SourceBindingError("S45A_RUNTIME_DISPATCH_BINDING_CHANGED")
    bind_active_propagation(wrapped, state)
    post_check = verify_bindings(state)
    binding["runtime_wrapper"] = copy.deepcopy(wrapper_binding)
    binding["post_binding_check"] = copy.deepcopy(post_check)
    dispatch.cpu_s45a_profile = bool(profile)
    dispatch.cpu_s45a_recorder = recorder
    dispatch.cpu_s45a_state = state
    dispatch.cpu_s45a_binding = copy.deepcopy(binding)
    dispatch.cpu_s45a_instrumentation_bindings = copy.deepcopy(getattr(state, "_cpu_s45a_bindings", {}))
    return wrapped, dispatch, clock, binding, audit, lease


__all__ = [
    "CANDIDATE_SCHEMA",
    "CandidateUnsupported",
    "DEFAULT_VARIANT",
    "InstrumentationBindingError",
    "LINEAR_CACHE_MODES",
    "LINEAR_CACHE_ROLE",
    "LOCAL_ENGINEERING_CONTRACT",
    "POOL_CONTRACT",
    "SourceBindingError",
    "build_role",
    "make_factory",
    "source_identity",
    "verify_bindings",
]
