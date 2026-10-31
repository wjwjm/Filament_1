"""Private L1 role adapter for the Eq10/P48 propagation candidate."""

from __future__ import annotations

import copy
import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from tools.hpc_ops import provenance_v2 as pv

from .instrument import (
    HELPER_NAMES,
    INSTRUMENT_SCHEMA,
    InstrumentationBindingError,
    _code_identity,
    install_stage,
    validate_cpu_backend,
    verify_seal,
)
from .pool import Eq10Pool, NullEq10Pool


CANDIDATE_SCHEMA = "filament.cpu_s4_6d.eq10_propagation.candidate.v1"
DEFAULT_VARIANT = "EQ10_POINTWISE_P48_V1"
ROLE_REFERENCE = "R"
ROLE_CANDIDATE = "B"
MODES = ("QUALIFY_BOUNDARY", "PURE_OFF_SMALLGRID")
LOCAL_CONTRACT = {
    "numba_pool": 1,
    "outer_mask": 1,
    "temporal_workers": 1,
    "spatial_workers": 1,
    "raman_workers": 1,
    "eq10_workers": 1,
    "scope": "LOCAL_ENGINEERING_ONLY",
    "hpc_allowed": False,
}
HPC_CONTRACT = {
    "numba_pool": 48,
    "outer_mask": 24,
    "temporal_workers": 48,
    "spatial_workers": 24,
    "raman_workers": 48,
    "eq10_workers": 48,
    "scope": "HPC_FIXED_W48_W24",
    "hpc_allowed": True,
}


class CandidateUnsupported(RuntimeError):
    """The fixed Eq10 role cannot satisfy the selected route."""


class SourceBindingError(RuntimeError):
    """A frozen L1 or private Eq10 identity failed closed."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _owned_source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    files: dict[str, dict[str, Any]] = {}
    repo = _repo_root()
    sources = {
        "__init__.py": root / "__init__.py",
        "pool.py": root / "pool.py",
        "instrument.py": root / "instrument.py",
        "candidate.py": root / "candidate.py",
        # The worker implementation is intentionally reused from S4-6C.  Its
        # identity must travel with this candidate rather than being implicit.
        "chains.py": repo / "Filament_python/tools/cpu_s4_6c_array_coverage/chains.py",
    }
    for name, source in sources.items():
        path = source.resolve(strict=True)
        raw = path.read_bytes()
        record: dict[str, Any] = {
            "path": str(path),
            "size_bytes": int(len(raw)),
            "raw_sha256": pv.raw_sha256(raw),
            "classification": "external",
            "hash_scope": "raw_bytes",
        }
        relative = path.relative_to(repo).as_posix()
        try:
            oid = pv._git_blob_oid(repo, relative)
        except pv.ProvenanceError:
            # Local development may inspect uncommitted private files.  Once
            # a scheduler environment is active, raw fallback is forbidden.
            if os.environ.get("SLURM_JOB_ID"):
                raise SourceBindingError("EQ10_COMMITTED_OWNED_SOURCE_REQUIRED")
        else:
            blob = pv._read_git_blob(repo, oid)
            if pv.canonical_lf_bytes(raw) != pv.canonical_lf_bytes(blob):
                raise SourceBindingError("EQ10_OWNED_SOURCE_CANONICAL_BLOB_MISMATCH")
            record.update(
                {
                    "classification": "tracked_text",
                    "hash_scope": pv.TRACKED_HASH_SCOPE,
                    "git_blob_oid": oid,
                    "canonical_lf_sha256": pv.canonical_lf_sha256(blob),
                    "relative_path": relative,
                    "raw_byte_observation": "CURRENT_CHECKOUT_ONLY_NOT_CROSS_PLATFORM_IDENTITY",
                }
            )
        files[name] = record
    composite = (
        "producer_module:tools.cpu_s4_6d_eq10_propagation\n"
        + "".join(f"{name}:{files[name]['raw_sha256']}\n" for name in sorted(files))
    ).encode("utf-8")
    return {
        "schema": CANDIDATE_SCHEMA + ".owned_sources.v1",
        "producer_module_identity": "tools.cpu_s4_6d_eq10_propagation",
        "files": files,
        "composite_raw_sha256": pv.raw_sha256(composite),
        "instrumentation_schema": INSTRUMENT_SCHEMA,
    }


def source_identity() -> dict[str, Any]:
    parent = importlib.import_module("tools.cpu_s4_5a_l1_residual.candidate")
    return {
        "schema": CANDIDATE_SCHEMA,
        "candidate_name": DEFAULT_VARIANT,
        "roles": [ROLE_REFERENCE, ROLE_CANDIDATE],
        "parent_candidate": "tools.cpu_s4_5a_l1_residual",
        "parent_source_identity": copy.deepcopy(parent.source_identity()),
        "owned_source_identity": _owned_source_identity(),
        "instrumentation_schema": INSTRUMENT_SCHEMA,
        "new_math_algorithm": False,
        "pointwise_segments": [
            "spectral_multiply",
            "ifft_real_divide",
            "response_divide",
            "product",
        ],
    }


def _normalise_mode(mode: str) -> str:
    value = str(mode).upper()
    if value == "PURE_OFF":
        value = "PURE_OFF_SMALLGRID"
    if value not in MODES:
        raise CandidateUnsupported(f"EQ10_MODE_UNSUPPORTED:{mode}")
    return value


def _resolve_resources(*, local_engineering: bool, pool_override: int | None) -> dict[str, Any]:
    if local_engineering:
        if os.environ.get("SLURM_JOB_ID"):
            raise CandidateUnsupported("EQ10_LOCAL_ENGINEERING_ON_SLURM")
        if pool_override != 1:
            raise CandidateUnsupported("EQ10_LOCAL_REQUIRES_POOL_OVERRIDE_1")
        return copy.deepcopy(LOCAL_CONTRACT)
    if pool_override is not None:
        raise CandidateUnsupported("EQ10_HPC_POOL_OVERRIDE_FORBIDDEN")
    return copy.deepcopy(HPC_CONTRACT)


def _build_parent(
    *, local_engineering: bool, resources: dict[str, Any], recorder_pair: Any
) -> tuple[Any, Any, Any]:
    parent = importlib.import_module("tools.cpu_s4_5a_l1_residual.candidate")
    import numba
    import numpy as np

    if os.environ.get("UPPE_USE_GPU", "0") != "0":
        raise CandidateUnsupported("EQ10_CPU_ROUTE_REQUIRES_UPPE_USE_GPU_0")
    actual = int(numba.config.NUMBA_NUM_THREADS)
    if actual != int(resources["numba_pool"]):
        raise CandidateUnsupported(f"EQ10_NUMBA_POOL_MISMATCH:{actual}:{resources['numba_pool']}")
    numba.set_num_threads(int(resources["outer_mask"]))
    factory = parent.make_factory(
        local_engineering=bool(local_engineering),
        kernel_threads=1 if local_engineering else 24,
        temporal_workers=int(resources["temporal_workers"]),
        spatial_workers=int(resources["spatial_workers"]),
        raman_workers=int(resources["raman_workers"]),
        profile=False,
        recorder=recorder_pair.s45a,
    )
    propagator, dispatch = parent._bind_factory(factory)
    state = getattr(dispatch, "cpu_s45a_state", None)
    if state is None:
        state = getattr(dispatch, "cpu_s4_4a_state", None)
    if state is None:
        raise SourceBindingError("EQ10_PARENT_STATE_MISSING")
    r8_state = getattr(state, "r8_state", None)
    stage = getattr(state, "_internal_stage_inner", None)
    r8_stage = getattr(r8_state, "candidate_stage_inner", None)
    if not callable(stage) or not callable(r8_stage):
        raise SourceBindingError("EQ10_PARENT_R8_STAGE_BINDING_MISSING")
    expected_workers = {
        "temporal_workers": int(resources["temporal_workers"]),
        "spatial_workers": int(resources["spatial_workers"]),
    }
    state._cpu_s46d_expected_backend_workers = dict(expected_workers)
    try:
        backend = validate_cpu_backend(
            getattr(stage, "__globals__", {}).get("xp"),
            expected_workers=expected_workers,
        )
    except InstrumentationBindingError as exc:
        raise CandidateUnsupported(str(exc)) from exc
    if getattr(stage, "__globals__", {}).get("_np") is not np:
        raise CandidateUnsupported("EQ10_CPU_NUMPY_ALIAS_CHANGED")
    state._cpu_s46d_parent_backend_slots = {
        "xp": backend["xp_object"],
        "dispatch": backend["dispatch_object"],
        "underlying": backend["xp_object"].xp,
    }
    state._cpu_s46d_parent_backend_binding = {
        key: value
        for key, value in backend.items()
        if key not in {"xp_object", "dispatch_object"}
    }
    state._cpu_s46d_parent_r8_stage_identity = _code_identity(r8_stage)
    state._cpu_s46d_parent_stage_identity = _code_identity(stage)
    return propagator, dispatch, state


def _attach_runtime(
    propagator: Any,
    dispatch: Any,
    *,
    role: str,
    mode: str,
    capture_root: Path,
    recorder_pair: Any,
    owned_identity: dict[str, Any],
) -> tuple[Any, Any, Any, dict[str, Any], Any, Any]:
    adapter = importlib.import_module("tools.cpu_s4_5b_steady_cost.adapter")

    wrapped, actual_dispatch, clock, binding, audit, lease = adapter.attach_with_b(
        propagator,
        dispatch,
        role="G_CONTROLLED_INPUT",
        mode=mode,
        capture_root=capture_root,
        owned_identity=owned_identity,
        recorder=recorder_pair.b,
    )
    if actual_dispatch is not dispatch:
        raise SourceBindingError("EQ10_RUNTIME_DISPATCH_CHANGED")
    return wrapped, actual_dispatch, clock, binding, audit, lease


def _install_pool_api(dispatch: Any, state: Any, pool: Eq10Pool | NullEq10Pool) -> None:
    def close_eq10_pool() -> dict[str, Any]:
        return pool.close()

    def eq10_pool_snapshot() -> dict[str, Any]:
        return pool.snapshot()

    def take_eq10_pool_records() -> list[dict[str, Any]]:
        return pool.take_records()

    dispatch.close_eq10_pool = close_eq10_pool
    dispatch.eq10_pool_snapshot = eq10_pool_snapshot
    dispatch.take_eq10_pool_records = take_eq10_pool_records
    dispatch.cpu_s4_6d_eq10_pool = pool
    dispatch.cpu_s4_6d_state = state


def verify_bindings(dispatch: Any) -> dict[str, Any]:
    """Recheck the private stage/helper seals without touching array values."""

    role = getattr(dispatch, "cpu_s4_6d_role", None)
    if role not in (ROLE_REFERENCE, ROLE_CANDIDATE):
        raise SourceBindingError("EQ10_ROLE_BINDING_INVALID")
    state = getattr(dispatch, "cpu_s4_6d_state", None)
    pool = getattr(dispatch, "cpu_s4_6d_eq10_pool", None)
    if state is None or pool is None:
        raise SourceBindingError("EQ10_STATE_OR_POOL_MISSING")

    adapter = importlib.import_module("tools.cpu_s4_5b_steady_cost.adapter")
    pair = getattr(dispatch, "cpu_s4_5b_recorder_pair", None)
    if type(pair) is not getattr(adapter, "RecorderPair", object):
        raise SourceBindingError("EQ10_RECORDER_PAIR_BINDING_CHANGED")
    if pair.profile is not False or getattr(dispatch, "cpu_s4_5b_profile", None) is not False:
        raise SourceBindingError("EQ10_COMMON_PROFILE_NOT_OFF")
    if getattr(dispatch, "cpu_s4_5b_state", None) is not state:
        raise SourceBindingError("EQ10_COMMON_STATE_BINDING_CHANGED")
    common_binding = getattr(dispatch, "cpu_s4_5b_binding", None)
    if not isinstance(common_binding, dict):
        raise SourceBindingError("EQ10_COMMON_BODY_BINDING_MISSING")
    if common_binding.get("s45b_g2_compile_seam") != "PRIVATE_NAMESPACE_BEFORE_FINAL_COMPILE":
        raise SourceBindingError("EQ10_COMMON_BODY_COMPILE_SEAM_CHANGED")
    if set(common_binding.get("s45b_runtime_helpers", ())) != {
        "__cpu_s45b_scope__",
        "__cpu_s45b_marker__",
    }:
        raise SourceBindingError("EQ10_COMMON_BODY_HELPERS_MISSING")

    stage = getattr(state, "_internal_stage_inner", None)
    r8_state = getattr(state, "r8_state", None)
    r8_stage = getattr(r8_state, "candidate_stage_inner", None)
    if not callable(stage) or not callable(r8_stage):
        raise SourceBindingError("EQ10_PARENT_STAGE_BINDING_CHANGED")
    if role == ROLE_CANDIDATE:
        stage_owner = getattr(state, "_cpu_s46d_stage_owner", None)
        stage_attribute = getattr(state, "_cpu_s46d_stage_attribute", None)
        active = getattr(state, "_cpu_s46d_stage", None)
        if (
            stage_owner is None
            or not isinstance(stage_attribute, str)
            or getattr(stage_owner, stage_attribute, None) is not active
            or active is not stage
        ):
            raise SourceBindingError("EQ10_ACTIVE_STAGE_OBJECT_CHANGED")
    stage_globals = getattr(stage, "__globals__", {})
    try:
        backend = validate_cpu_backend(
            stage_globals.get("xp"),
            expected_workers=getattr(state, "_cpu_s46d_expected_backend_workers", None),
        )
    except InstrumentationBindingError as exc:
        raise SourceBindingError(str(exc)) from exc
    if stage_globals.get("_np") is not np:
        raise SourceBindingError("EQ10_CPU_NUMPY_ALIAS_CHANGED")
    backend_metadata = {
        key: value
        for key, value in backend.items()
        if key not in {"xp_object", "dispatch_object"}
    }
    parent_backend_binding = getattr(state, "_cpu_s46d_parent_backend_binding", None)
    if parent_backend_binding != backend_metadata:
        raise SourceBindingError("EQ10_PARENT_BACKEND_BINDING_CHANGED")
    parent_backend_slots = getattr(state, "_cpu_s46d_parent_backend_slots", {})
    if (
        parent_backend_slots.get("xp") is not backend["xp_object"]
        or parent_backend_slots.get("dispatch") is not backend["dispatch_object"]
        or parent_backend_slots.get("underlying") is not np
    ):
        raise SourceBindingError("EQ10_PARENT_BACKEND_SLOT_CHANGED")
    parent_r8_identity = getattr(state, "_cpu_s46d_parent_r8_stage_identity", None)
    if parent_r8_identity != _code_identity(r8_stage):
        raise SourceBindingError("EQ10_R8_STAGE_CODE_CHANGED")
    expected_parent_stage = getattr(state, "_cpu_s46d_parent_stage_identity", None)
    if role == ROLE_REFERENCE:
        if expected_parent_stage != _code_identity(stage):
            raise SourceBindingError("EQ10_PARENT_STAGE_CODE_CHANGED")
        if type(pool) is not NullEq10Pool:
            raise SourceBindingError("EQ10_REFERENCE_POOL_BINDING_CHANGED")
        return {
            "schema": CANDIDATE_SCHEMA + ".binding.v1",
            "role": role,
            "stage_replaced": False,
            "common_body_instrumentation_bound": True,
            "pool": pool.snapshot(),
        }

    stage_seal = getattr(state, "_cpu_s46d_stage_seal", None)
    helper_seals = getattr(state, "_cpu_s46d_helper_seals", {})
    if stage_seal is None or len(helper_seals) != 4:
        raise SourceBindingError("EQ10_STAGE_OR_HELPER_SEALS_MISSING")
    original = getattr(state, "_cpu_s46d_original_stage", None)
    if not callable(original) or expected_parent_stage != _code_identity(original):
        raise SourceBindingError("EQ10_ORIGINAL_STAGE_CODE_CHANGED")
    active_ref = getattr(state, "_cpu_s46d_active_stage_ref", None)
    if not isinstance(active_ref, dict) or active_ref.get("stage") is not active:
        raise SourceBindingError("EQ10_ACTIVE_STAGE_GUARD_REFERENCE_CHANGED")
    r8_original = getattr(state, "_cpu_s46d_r8_stage_original", None)
    if stage_owner is not r8_state and getattr(r8_state, "candidate_stage_inner", None) is not r8_original:
        raise SourceBindingError("EQ10_R8_ACTIVE_STAGE_OBJECT_CHANGED")
    if getattr(state, "_cpu_s46d_pool", None) is not pool or type(pool) is not Eq10Pool:
        raise SourceBindingError("EQ10_POOL_BINDING_CHANGED")
    verify_seal(stage_seal)
    guard_seal = getattr(state, "_cpu_s46d_guard_seal", None)
    if guard_seal is None or getattr(state, "_cpu_s46d_guard", None) is not guard_seal.get("function"):
        raise SourceBindingError("EQ10_PRE_CALL_GUARD_BINDING_CHANGED")
    verify_seal(guard_seal)
    backend_slots = getattr(state, "_cpu_s46d_backend_slots", {})
    if (
        backend_slots.get("numpy") is not np
        or backend_slots.get("multiply") is not np.multiply
        or backend_slots.get("divide") is not np.divide
        or backend_slots.get("xp") is not backend["xp_object"]
        or backend_slots.get("dispatch") is not backend["dispatch_object"]
        or backend_slots.get("underlying") is not np
    ):
        raise SourceBindingError("EQ10_FIXED_BACKEND_SLOT_CHANGED")
    private_namespace = getattr(state, "_cpu_s46d_private_namespace", {})
    if (
        not isinstance(private_namespace, dict)
        or private_namespace.get("xp") is not backend["xp_object"]
        or private_namespace.get("_np") is not np
    ):
        raise SourceBindingError("EQ10_PRIVATE_BACKEND_SLOT_CHANGED")
    if set(helper_seals) != set(HELPER_NAMES.values()):
        raise SourceBindingError("EQ10_HELPER_REGISTRY_CHANGED")
    for name, seal in helper_seals.items():
        function = seal.get("function")
        if stage_globals.get(name) is not function or private_namespace.get(name) is not function:
            raise SourceBindingError(f"EQ10_STAGE_GLOBAL_SLOT_CHANGED:{name}")
        if getattr(function, "__cpu_s46d_kind__", None) != "FIXED_EQ10_POINTWISE_HELPER":
            raise SourceBindingError(f"EQ10_HELPER_KIND_CHANGED:{name}")
        if getattr(function, "__cpu_s46d_pool__", None) is not pool:
            raise SourceBindingError(f"EQ10_HELPER_POOL_CHANGED:{name}")
        if getattr(function, "__globals__", {}).get("np") is not np:
            raise SourceBindingError(f"EQ10_HELPER_BACKEND_CHANGED:{name}")
        verify_seal(seal)
    return {
        "schema": CANDIDATE_SCHEMA + ".binding.v1",
        "role": role,
        "stage_replaced": True,
        "stage_code_identity": _code_identity(state._cpu_s46d_stage),
        "helper_count": len(helper_seals),
        "pool": pool.snapshot(),
        "lineage": copy.deepcopy(getattr(state, "_cpu_s46d_lineage", {})),
        "common_body_instrumentation_bound": True,
    }


def build_role(
    mode: str,
    *,
    capture_root: str | Path,
    role: str = ROLE_CANDIDATE,
    local_engineering: bool = False,
    pool_override: int | None = None,
    recorder_pair: Any | None = None,
) -> tuple[Any, Any, Any, dict[str, Any], Any, Any]:
    """Build R or B over the accepted L1 six-tuple runtime contract."""

    mode_name = _normalise_mode(mode)
    role_name = str(role).upper()
    if role_name not in (ROLE_REFERENCE, ROLE_CANDIDATE):
        raise CandidateUnsupported(f"EQ10_ROLE_UNSUPPORTED:{role}")
    resources = _resolve_resources(local_engineering=bool(local_engineering), pool_override=pool_override)
    adapter = importlib.import_module("tools.cpu_s4_5b_steady_cost.adapter")
    if recorder_pair is None:
        recorder_pair = adapter.make_recorder_pair(False)
    expected_pair = getattr(adapter, "RecorderPair", None)
    if expected_pair is None or type(recorder_pair) is not expected_pair:
        raise CandidateUnsupported("EQ10_RECORDER_PAIR_EXACT_TYPE_REQUIRED")
    if bool(getattr(recorder_pair, "profile", True)):
        raise CandidateUnsupported("EQ10_RECORDER_PAIR_PROFILE_MUST_BE_OFF")
    pool: Eq10Pool | NullEq10Pool = NullEq10Pool()
    try:
        propagator, dispatch, state = _build_parent(
            local_engineering=bool(local_engineering),
            resources=resources,
            recorder_pair=recorder_pair,
        )
        stage_receipt: dict[str, Any] = {
            "stage_replaced": False,
            "schema": INSTRUMENT_SCHEMA,
        }
        if role_name == ROLE_CANDIDATE:
            pool = Eq10Pool(int(resources["eq10_workers"]))
            stage_receipt = install_stage(state, pool)
        s45a = importlib.import_module("tools.cpu_s4_5a_l1_residual.instrument")
        s45a.install_propagation_wrappers(propagator, state)
        pre_check = s45a.verify_bindings(state)
        capture = Path(capture_root).resolve()
        owned = _owned_source_identity()
        wrapped, actual_dispatch, clock, wrapper_binding, audit, lease = _attach_runtime(
            propagator,
            dispatch,
            role=role_name,
            mode=mode_name,
            capture_root=capture,
            recorder_pair=recorder_pair,
            owned_identity=owned,
        )
        s45a.bind_active_propagation(wrapped, state)
        post_check = s45a.verify_bindings(state)
        _install_pool_api(actual_dispatch, state, pool)
        actual_dispatch.cpu_s4_5b_profile = False
        actual_dispatch.cpu_s4_5b_recorder_pair = recorder_pair
        actual_dispatch.cpu_s4_5b_state = state
        actual_dispatch.cpu_s4_6d_role = role_name
        actual_dispatch.cpu_s4_6d_candidate_schema = CANDIDATE_SCHEMA
        actual_dispatch.cpu_s4_6d_binding = {
            "schema": CANDIDATE_SCHEMA + ".binding.v1",
            "role": role_name,
            "mode": mode_name,
            "candidate": DEFAULT_VARIANT,
            "resource_contract": copy.deepcopy(resources),
            "stage": copy.deepcopy(stage_receipt),
            "s45a_pre_binding_check": copy.deepcopy(pre_check),
            "s45a_post_binding_check": copy.deepcopy(post_check),
            "parent_linear_binding": copy.deepcopy(getattr(actual_dispatch, "cpu_s4_4a_linear_binding", {})),
            "runtime_wrapper": copy.deepcopy(wrapper_binding),
            "owned_source_identity": owned,
            "scientific_qualification": False,
            "production_qualified": False,
            "recorder_pair_bound": True,
            "common_body_instrumentation_bound": True,
        }
        actual_dispatch.cpu_s4_5b_binding = copy.deepcopy(wrapper_binding)
        actual_dispatch.cpu_s4_5b_binding.update(
            {
                "schema": CANDIDATE_SCHEMA,
                "eq10_role": role_name,
                "eq10_mode": mode_name,
                "eq10_stage": copy.deepcopy(stage_receipt),
                "eq10_common_body_instrumentation_bound": True,
                "eq10_s45a_pre_binding_check": copy.deepcopy(pre_check),
                "eq10_s45a_post_binding_check": copy.deepcopy(post_check),
            }
        )
        actual_dispatch.cpu_s4_5b_instrumentation_bindings = copy.deepcopy(
            getattr(state, "_cpu_s45a_bindings", {})
        )
        actual_dispatch.cpu_s4_6d_binding["common_body_binding"] = copy.deepcopy(
            actual_dispatch.cpu_s4_5b_binding
        )
        binding_check = verify_bindings(actual_dispatch)
        actual_dispatch.cpu_s4_6d_binding["binding_check"] = copy.deepcopy(binding_check)
        actual_dispatch.cpu_s4_5b_binding["b_post_binding_check"] = copy.deepcopy(binding_check)
        return wrapped, actual_dispatch, clock, copy.deepcopy(actual_dispatch.cpu_s4_6d_binding), audit, lease
    except BaseException:
        try:
            pool.close()
        except BaseException:
            pass
        raise


__all__ = [
    "CANDIDATE_SCHEMA",
    "CandidateUnsupported",
    "DEFAULT_VARIANT",
    "HPC_CONTRACT",
    "LOCAL_CONTRACT",
    "ROLE_CANDIDATE",
    "ROLE_REFERENCE",
    "SourceBindingError",
    "build_role",
    "source_identity",
    "verify_bindings",
]
