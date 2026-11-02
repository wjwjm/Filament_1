"""Private common-L1 reference and joint S4-6H candidate factories.

The factory keeps the accepted six-tuple runtime shape while composing three
independent semantic pools in one role instance: intensity A/P24, Eq10 B/P48,
and nonlinear C/P24.  Only the Raman stage is compiled jointly.  C is
attached once at the G2 callsite and its source/closure guards remain owned
by the C adapter.
"""

from __future__ import annotations

import copy
import functools
import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Mapping

import numpy as np

from tools.hpc_ops import provenance_v2 as pv

from tools.cpu_s4_6d_eq10_propagation.instrument import (
    HELPER_NAMES as EQ10_HELPER_NAMES,
    InstrumentationBindingError as Eq10InstrumentationBindingError,
    _helpers as make_eq10_helpers,
    validate_cpu_backend as validate_eq10_backend,
)
from tools.cpu_s4_6d_eq10_propagation.pool import Eq10Pool, NullEq10Pool
from tools.cpu_s4_6e_array_propagation.c_candidate import (
    ROLE_CANDIDATE as C_ROLE_CANDIDATE,
    ROLE_REFERENCE as C_ROLE_REFERENCE,
    SourceBindingError as CSourceBindingError,
    _build_parent as build_c_parent,
    _install_pool_api as install_c_pool_api,
    attach_with_c,
    source_identity as c_source_identity,
    verify_bindings as verify_c_bindings,
)
from tools.cpu_s4_6e_array_propagation.instrument import (
    HELPER_NAMES as INTENSITY_HELPER_NAMES,
    InstrumentationBindingError as IntensityInstrumentationBindingError,
    _helpers as make_intensity_helpers,
    validate_cpu_backend as validate_intensity_backend,
)
from tools.cpu_s4_6e_array_propagation.pool import IntensityPool, NullIntensityPool
from tools.cpu_s4_6e_array_propagation.c_pool import CPool, NullCPool

from .instrument import (
    INSTRUMENT_SCHEMA,
    InstrumentationBindingError,
    JOINT_GUARD_NAME,
    _code_identity,
    _seal,
    compile_combined_stage,
    verify_compiled_stage,
    verify_seal,
)


CANDIDATE_SCHEMA = "filament.cpu_s4_6h.combined_parallel.candidate.v1"
DEFAULT_VARIANT = "INTENSITY_P24_EQ10_P48_PHASE_ABSORB_P24_V1"
COMBINED_SEMANTIC_ID = "S46H_INTENSITY_P24_EQ10_P48_PHASE_ABSORB_P24"
ROLE_REFERENCE = "R"
ROLE_CANDIDATE = "X"
MODES = ("QUALIFY_BOUNDARY", "PURE_OFF_SMALLGRID")

LOCAL_CONTRACT = {
    "numba_pool": 1,
    "outer_mask": 1,
    "temporal_workers": 1,
    "spatial_workers": 1,
    "raman_workers": 1,
    "intensity_workers": 1,
    "eq10_workers": 1,
    "c_workers": 1,
    "scope": "LOCAL_ENGINEERING_ONLY",
    "hpc_allowed": False,
}
HPC_CONTRACT = {
    "numba_pool": 48,
    "outer_mask": 24,
    "temporal_workers": 48,
    "spatial_workers": 24,
    "raman_workers": 48,
    "intensity_workers": 24,
    "eq10_workers": 48,
    "c_workers": 24,
    "scope": "HPC_FIXED_W48_A24_B48_C24",
    "hpc_allowed": True,
}


class CandidateUnsupported(RuntimeError):
    """The fixed combined role cannot satisfy the selected route."""


class SourceBindingError(RuntimeError):
    """A frozen parent, stage, helper or source closure failed closed."""


@dataclass
class _PoolSet:
    intensity: IntensityPool | NullIntensityPool
    eq10: Eq10Pool | NullEq10Pool
    c: CPool | NullCPool

    def values(self) -> tuple[tuple[str, Any], ...]:
        return (("intensity", self.intensity), ("eq10", self.eq10), ("c", self.c))

    def close_and_drain(self) -> dict[str, Any]:
        """Close/join every pool and drain only scalar records."""

        result: dict[str, Any] = {}
        errors: list[str] = []
        for name, pool in self.values():
            try:
                pool.close()
            except BaseException as exc:
                errors.append(f"{name}:{type(exc).__name__}")
            try:
                records = pool.take_records()
            except BaseException as exc:
                errors.append(f"{name}:drain:{type(exc).__name__}")
                records = []
            try:
                snapshot = pool.snapshot()
            except BaseException as exc:
                errors.append(f"{name}:snapshot:{type(exc).__name__}")
                snapshot = {"closed": bool(getattr(pool, "closed", False))}
            if snapshot.get("close_error") is not None:
                errors.append(f"{name}:recorded_close_error:{snapshot['close_error']}")
            result[name] = {"pool": copy.deepcopy(snapshot), "activity": copy.deepcopy(records)}
        result["schema"] = CANDIDATE_SCHEMA + ".pools.v1"
        result["all_closed"] = all(
            bool(row["pool"].get("closed", False))
            for row in result.values()
            if isinstance(row, dict) and "pool" in row
        )
        result["close_errors"] = tuple(errors)
        result["records_scalar_only"] = True
        return result

    def snapshots(self) -> dict[str, Any]:
        return {name: copy.deepcopy(pool.snapshot()) for name, pool in self.values()}

    def records(self) -> dict[str, list[dict[str, Any]]]:
        return {name: copy.deepcopy(pool.take_records()) for name, pool in self.values()}


PoolSet = _PoolSet


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalise_mode(mode: str) -> str:
    value = str(mode).upper()
    if value == "PURE_OFF":
        value = "PURE_OFF_SMALLGRID"
    if value not in MODES:
        raise CandidateUnsupported(f"S46H_MODE_UNSUPPORTED:{mode}")
    return value


def _normalise_role(role: str) -> str:
    value = str(role).upper()
    value = {
        "CANDIDATE": ROLE_CANDIDATE,
        "COMBINED": ROLE_CANDIDATE,
        "J": ROLE_CANDIDATE,
        # H was used by the first private draft; retain it only as an
        # unadvertised compatibility alias while X remains canonical.
        "H": ROLE_CANDIDATE,
    }.get(value, value)
    if value not in (ROLE_REFERENCE, ROLE_CANDIDATE):
        raise CandidateUnsupported(f"S46H_ROLE_UNSUPPORTED:{role}")
    return value


def _resolve_resources(*, local_engineering: bool, pool_override: int | None) -> dict[str, Any]:
    if local_engineering:
        if os.environ.get("SLURM_JOB_ID"):
            raise CandidateUnsupported("S46H_LOCAL_ENGINEERING_ON_SLURM")
        if pool_override != 1:
            raise CandidateUnsupported("S46H_LOCAL_REQUIRES_POOL_OVERRIDE_1")
        return copy.deepcopy(LOCAL_CONTRACT)
    if pool_override is not None:
        raise CandidateUnsupported("S46H_HPC_POOL_OVERRIDE_FORBIDDEN")
    return copy.deepcopy(HPC_CONTRACT)


def _owned_source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    repo = _repo_root()
    files: dict[str, dict[str, Any]] = {}
    for name in ("__init__.py", "instrument.py", "combined_candidate.py"):
        path = (root / name).resolve(strict=True)
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
            if os.environ.get("SLURM_JOB_ID"):
                raise SourceBindingError("S46H_COMMITTED_OWNED_SOURCE_REQUIRED")
        else:
            blob = pv._read_git_blob(repo, oid)
            if pv.canonical_lf_bytes(raw) != pv.canonical_lf_bytes(blob):
                raise SourceBindingError("S46H_OWNED_SOURCE_CANONICAL_BLOB_MISMATCH")
            record.update({
                "classification": "tracked_text",
                "hash_scope": pv.TRACKED_HASH_SCOPE,
                "git_blob_oid": oid,
                "canonical_lf_sha256": pv.canonical_lf_sha256(blob),
                "relative_path": relative,
                "raw_byte_observation": "CURRENT_CHECKOUT_ONLY_NOT_CROSS_PLATFORM_IDENTITY",
            })
        files[name] = record
    composite = (
        "producer_module:tools.cpu_s4_6h_combined_parallel\n"
        + "".join(f"{name}:{files[name]['raw_sha256']}\n" for name in sorted(files))
    ).encode("utf-8")
    return {
        "schema": CANDIDATE_SCHEMA + ".owned_sources.v1",
        "producer_module_identity": "tools.cpu_s4_6h_combined_parallel",
        "files": files,
        "composite_raw_sha256": pv.raw_sha256(composite),
    }


def source_identity() -> dict[str, Any]:
    """Return the combined identity, retaining all three source closures."""

    intensity = importlib.import_module("tools.cpu_s4_6e_array_propagation.candidate")
    eq10 = importlib.import_module("tools.cpu_s4_6d_eq10_propagation.candidate")
    parent = importlib.import_module("tools.cpu_s4_5a_l1_residual.candidate")
    components = {
        "intensity_p24": copy.deepcopy(intensity.source_identity()),
        "eq10_p48": copy.deepcopy(eq10.source_identity()),
        "phase_absorb_c_p24": copy.deepcopy(c_source_identity()),
    }
    return {
        "schema": CANDIDATE_SCHEMA,
        "candidate_name": DEFAULT_VARIANT,
        "combined_semantic_id": COMBINED_SEMANTIC_ID,
        "roles": [ROLE_REFERENCE, ROLE_CANDIDATE],
        "parent_candidate": "tools.cpu_s4_5a_l1_residual",
        "parent_source_identity": copy.deepcopy(parent.source_identity()),
        "owned_source_identity": _owned_source_identity(),
        "source_closure": components,
        "component_order": ["intensity_p24", "eq10_p48", "phase_absorb_c_p24"],
        "instrumentation_schema": INSTRUMENT_SCHEMA,
        "new_math_algorithm": False,
        "stage_concurrency": False,
        "pool_workers_hpc": {"intensity": 24, "eq10": 48, "c": 24},
        "production_qualified": False,
    }


COMBINED_RECORD_CAPACITY = 256


def _live_thread_ids(pool: Any) -> list[int]:
    executor = getattr(getattr(pool, "_workers", None), "executor", None)
    live: list[int] = []
    for thread in getattr(executor, "_threads", ()):
        if thread.is_alive() and getattr(thread, "native_id", None) is not None:
            live.append(int(thread.native_id))
    return sorted(live)


def _scalar_only(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return False
    if isinstance(value, Mapping):
        return all(_scalar_only(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_scalar_only(item) for item in value)
    return isinstance(value, (str, bytes, int, float, bool, type(None), np.generic))


def _add_scalar(sums: dict[str, int], name: str, value: Any) -> None:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        sums[name] = int(sums.get(name, 0)) + int(value)


def _combined_record(pool: Any, label: str) -> None:
    """Drain worker records after every operation before legacy caps apply."""

    pool._labels[label] = pool._labels.get(label, 0) + 1
    rows = pool._workers.take_records()
    for row in rows:
        pool._s46h_total_records += 1
        if not _scalar_only(row):
            pool._s46h_dropped_records += 1
            pool._s46h_invalid_records += 1
            continue
        _add_scalar(pool._s46h_scalar_sums, "requested_workers", row.get("requested_workers"))
        _add_scalar(pool._s46h_scalar_sums, "actual_workers", row.get("actual_workers"))
        workers = row.get("workers", ())
        if isinstance(workers, (list, tuple)):
            _add_scalar(pool._s46h_scalar_sums, "worker_count", len(workers))
            for worker in workers:
                if isinstance(worker, Mapping):
                    _add_scalar(pool._s46h_scalar_sums, "items", worker.get("items"))
                    _add_scalar(pool._s46h_scalar_sums, "wall_ns", worker.get("wall_ns"))
                    _add_scalar(pool._s46h_scalar_sums, "thread_cpu_ns", worker.get("thread_cpu_ns"))
        if len(pool._s46h_pending_records) < COMBINED_RECORD_CAPACITY:
            pool._s46h_pending_records.append(copy.deepcopy(row))
        else:
            pool._s46h_dropped_records += 1
    # Never let the inherited 128-row list retain an operation row.
    pool._records.clear()


def _combined_take_records(pool: Any) -> list[dict[str, Any]]:
    rows = pool._s46h_pending_records
    pool._s46h_pending_records = []
    # This is only a defensive drain for rows created before adapter install.
    legacy = pool._s46h_original_take_records()
    if legacy:
        for row in legacy:
            if _scalar_only(row) and len(rows) < COMBINED_RECORD_CAPACITY:
                rows.append(copy.deepcopy(row))
            else:
                pool._s46h_dropped_records += 1
    return copy.deepcopy(rows)


def _combined_snapshot(pool: Any) -> dict[str, Any]:
    snapshot = pool._s46h_original_snapshot()
    snapshot.update(
        {
            "active_thread_ids": _live_thread_ids(pool),
            "pending_scalar_records": len(pool._s46h_pending_records),
            "records_capacity": COMBINED_RECORD_CAPACITY,
            "total_records": int(pool._s46h_total_records),
            "retained_records": len(pool._s46h_pending_records),
            "dropped_records": int(pool._s46h_dropped_records),
            "invalid_records_dropped": int(pool._s46h_invalid_records),
            "scalar_sums": copy.deepcopy(pool._s46h_scalar_sums),
            "records_scalar_only": True,
        }
    )
    return snapshot


def _install_combined_pool_adapter(pool: Any) -> Any:
    """Attach a combined-owned per-operation scalar-record adapter."""

    pool._s46h_original_snapshot = pool.snapshot
    pool._s46h_original_take_records = pool.take_records
    pool._s46h_pending_records: list[dict[str, Any]] = []
    pool._s46h_total_records = 0
    pool._s46h_dropped_records = 0
    pool._s46h_invalid_records = 0
    pool._s46h_scalar_sums = {
        "requested_workers": 0,
        "actual_workers": 0,
        "worker_count": 0,
        "items": 0,
        "wall_ns": 0,
        "thread_cpu_ns": 0,
    }
    pool._record = MethodType(_combined_record, pool)
    pool.take_records = MethodType(_combined_take_records, pool)
    pool.snapshot = MethodType(_combined_snapshot, pool)
    return pool


def _make_pools(role: str, resources: Mapping[str, Any]) -> _PoolSet:
    role_name = _normalise_role(role)
    if role_name == ROLE_REFERENCE:
        return _PoolSet(NullIntensityPool(), NullEq10Pool(), NullCPool())
    created: list[Any] = []
    try:
        intensity = IntensityPool(int(resources["intensity_workers"]))
        created.append(intensity)
        eq10 = Eq10Pool(int(resources["eq10_workers"]))
        created.append(eq10)
        c_pool = CPool(int(resources["c_workers"]))
        created.append(c_pool)
        poolset = _PoolSet(intensity, eq10, c_pool)
        for _name, pool in poolset.values():
            _install_combined_pool_adapter(pool)
        return poolset
    except BaseException:
        for pool in reversed(created):
            try:
                pool.close()
            except BaseException:
                pass
            try:
                pool.take_records()
            except BaseException:
                pass
        raise


def _build_parent(*, local_engineering: bool, resources: Mapping[str, Any], recorder_pair: Any) -> tuple[Any, Any, Any]:
    """Build exactly one common L1 parent state for either role."""

    try:
        return build_c_parent(
            local_engineering=bool(local_engineering),
            resources=dict(resources),
            recorder_pair=recorder_pair,
        )
    except (CandidateUnsupported, CSourceBindingError):
        raise
    except BaseException as exc:
        raise SourceBindingError("S46H_PARENT_CONSTRUCTION_FAILED") from exc


def _stage_owner(state: Any) -> tuple[Any, str, Callable[..., Any], Any]:
    r8_state = getattr(state, "r8_state", None)
    r8_stage = getattr(r8_state, "candidate_stage_inner", None)
    original = getattr(state, "_internal_stage_inner", None)
    owner = state
    attribute = "_internal_stage_inner"
    if not callable(original) and callable(r8_stage):
        original = r8_stage
        owner = r8_state
        attribute = "candidate_stage_inner"
    if not callable(original):
        raise SourceBindingError("S46H_ORIGINAL_STAGE_MISSING")
    return owner, attribute, original, r8_stage


def _backend_metadata(backend: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in backend.items() if key not in {"xp_object", "dispatch_object"}}


def _make_joint_helpers(
    *,
    state: Any,
    poolset: _PoolSet,
    stage_owner: Any,
    stage_attribute: str,
    baseline_globals: Mapping[str, Any],
    expected_workers: Mapping[str, int],
) -> tuple[dict[str, Callable[..., Any]], dict[str, Any], dict[str, Any]]:
    """Build both component helper families over one shared stage guard."""

    active_ref: dict[str, Any] = {"stage": None}
    baseline_namespace = dict(baseline_globals)
    xp = baseline_namespace.get("xp")
    try:
        intensity_backend = validate_intensity_backend(xp, expected_workers=expected_workers)
        eq10_backend = validate_eq10_backend(xp, expected_workers=expected_workers)
    except (IntensityInstrumentationBindingError, Eq10InstrumentationBindingError) as exc:
        raise CandidateUnsupported(str(exc)) from exc
    if baseline_namespace.get("_np") is not np:
        raise CandidateUnsupported("S46H_CPU_NUMPY_ALIAS_CHANGED")
    if intensity_backend["xp_object"] is not eq10_backend["xp_object"]:
        raise SourceBindingError("S46H_COMPONENT_BACKEND_OBJECT_MISMATCH")
    if intensity_backend["dispatch_object"] is not eq10_backend["dispatch_object"]:
        raise SourceBindingError("S46H_COMPONENT_BACKEND_DISPATCH_MISMATCH")

    def close_failure() -> None:
        try:
            poolset.close_and_drain()
        except BaseException:
            pass

    state._cpu_s46h_close_failure = close_failure
    state._cpu_s46h_close_failure_seal = _seal(close_failure)

    def guard_before_helper() -> None:
        try:
            active = active_ref.get("stage")
            if active is None or getattr(stage_owner, stage_attribute, None) is not active:
                raise SourceBindingError("S46H_ACTIVE_STAGE_CHANGED")
            if getattr(state, "_cpu_s46h_stage", None) is not active:
                raise SourceBindingError("S46H_ACTIVE_STAGE_STATE_CHANGED")
            for name, pool in poolset.values():
                if bool(getattr(pool, "closed", False)):
                    raise SourceBindingError(f"S46H_RETIRED_{name.upper()}_POOL")
            validate_intensity_backend(xp, expected_workers=expected_workers)
            validate_eq10_backend(xp, expected_workers=expected_workers)
        except BaseException:
            close_failure()
            raise

    intensity_helpers = make_intensity_helpers(poolset.intensity, guard=guard_before_helper)
    eq10_helpers = make_eq10_helpers(poolset.eq10, guard=guard_before_helper)
    raw_helpers: dict[str, Callable[..., Any]] = {}
    raw_helpers.update(intensity_helpers)
    raw_helpers.update(eq10_helpers)

    helpers: dict[str, Callable[..., Any]] = {}
    helper_components: dict[str, str] = {}
    helper_pools: dict[str, Any] = {}
    helper_originals: dict[str, Callable[..., Any]] = {}
    helper_original_seals: dict[str, dict[str, Any]] = {}
    for name, original in raw_helpers.items():
        component = "intensity" if name in INTENSITY_HELPER_NAMES.values() else "eq10"
        pool = poolset.intensity if component == "intensity" else poolset.eq10

        @functools.wraps(original)
        def safe_helper(*args: Any, _original=original, **kwargs: Any) -> Any:
            try:
                return _original(*args, **kwargs)
            except BaseException:
                close_failure()
                raise

        safe_helper.__name__ = name
        safe_helper.__cpu_s46h_kind__ = (
            "FIXED_INTENSITY_POINTWISE_HELPER"
            if component == "intensity"
            else "FIXED_EQ10_POINTWISE_HELPER"
        )
        safe_helper.__cpu_s46h_component__ = component
        safe_helper.__cpu_s46h_pool__ = pool
        safe_helper.__cpu_s46h_original__ = original
        # Retain the component-owned semantic tags while adding the joint
        # owner tags; no old installer is invoked or stacked.
        for attribute in (
            "__cpu_s46e_kind__",
            "__cpu_s46e_pool__",
            "__cpu_s46d_kind__",
            "__cpu_s46d_pool__",
        ):
            if hasattr(original, attribute):
                setattr(safe_helper, attribute, getattr(original, attribute))
        helpers[name] = safe_helper
        helper_components[name] = component
        helper_pools[name] = pool
        helper_originals[name] = original
        helper_original_seals[name] = _seal(original)

    private_namespace = dict(baseline_namespace)
    private_namespace.update(helpers)
    scipy_fft = importlib.import_module("scipy.fft")
    scipy_slots = {name: getattr(scipy_fft, name) for name in ("fft", "ifft")}
    backend_slots = {
        "numpy": np,
        "absolute": np.absolute,
        "square": np.square,
        "multiply": np.multiply,
        "divide": np.divide,
        "xp": xp,
        "dispatch": intensity_backend["dispatch_object"],
        "underlying": np,
    }
    backend_binding = _backend_metadata(intensity_backend)
    return (
        helpers,
        {
            "active_ref": active_ref,
            "guard_before_helper": guard_before_helper,
            "private_namespace": private_namespace,
            "helper_components": helper_components,
            "helper_pools": helper_pools,
            "helper_originals": helper_originals,
            "helper_original_seals": helper_original_seals,
            "scipy_slots": scipy_slots,
            "backend_slots": backend_slots,
            "backend_binding": backend_binding,
            "xp": xp,
            "expected_workers": dict(expected_workers),
        },
        {"intensity": intensity_backend, "eq10": eq10_backend},
    )


def _install_joint_stage(state: Any, poolset: _PoolSet) -> dict[str, Any]:
    owner, attribute, original, r8_stage = _stage_owner(state)
    if getattr(state, "_cpu_s46h_stage", None) is not None:
        raise SourceBindingError("S46H_STAGE_ALREADY_INSTALLED")
    candidate_namespace = getattr(state, "_candidate_namespace", None)
    if not isinstance(candidate_namespace, dict):
        raise SourceBindingError("S46H_CANDIDATE_NAMESPACE_MISSING")
    s45a = importlib.import_module("tools.cpu_s4_5a_l1_residual.instrument")
    builder = getattr(s45a, "_expected_transformed_node", None)
    identity = getattr(s45a, "_code_identity", None)
    if not callable(builder) or not callable(identity):
        raise SourceBindingError("S46H_SOURCE_AST_SEAM_MISSING")
    source_path, stage_node, expected = builder(
        name="isaacs_raman_stage",
        transform_name="_transform_stage",
        candidate_namespace=candidate_namespace,
    )
    if identity(expected) != identity(original):
        raise SourceBindingError("S46H_R8_STAGE_SOURCE_MISMATCH")
    baseline_globals = getattr(original, "__globals__", None)
    if not isinstance(baseline_globals, dict):
        raise SourceBindingError("S46H_STAGE_GLOBALS_MISSING")
    expected_workers = getattr(state, "_cpu_s46h_expected_backend_workers", None)
    if not isinstance(expected_workers, Mapping):
        raise SourceBindingError("S46H_EXPECTED_BACKEND_WORKERS_MISSING")
    helpers, helper_state, _backends = _make_joint_helpers(
        state=state,
        poolset=poolset,
        stage_owner=owner,
        stage_attribute=attribute,
        baseline_globals=baseline_globals,
        expected_workers=expected_workers,
    )
    private_namespace = helper_state["private_namespace"]
    active_ref = helper_state["active_ref"]

    def guard_stage() -> None:
        try:
            for suffix in ("guard", "close_failure"):
                function = getattr(state, "_cpu_s46h_" + suffix, None)
                seal = getattr(state, "_cpu_s46h_" + suffix + "_seal", None)
                if not isinstance(seal, Mapping) or seal.get("function") is not function:
                    raise SourceBindingError(f"S46H_RUNTIME_GUARD_CHANGED:{suffix}")
                verify_seal(seal)
            if helper_state["guard_before_helper"] is not state._cpu_s46h_guard:
                raise SourceBindingError("S46H_RUNTIME_GUARD_SLOT_CHANGED")
            helper_state["guard_before_helper"]()
            active = active_ref.get("stage")
            if active is None or active.__globals__ is not private_namespace:
                raise SourceBindingError("S46H_STAGE_GLOBAL_NAMESPACE_CHANGED")
            if private_namespace.get(JOINT_GUARD_NAME) is not guard_stage:
                raise SourceBindingError("S46H_STAGE_GUARD_SLOT_CHANGED")
            if private_namespace.get("xp") is not helper_state["xp"] or private_namespace.get("_np") is not np:
                raise SourceBindingError("S46H_STAGE_BACKEND_SLOT_CHANGED")
            validate_intensity_backend(helper_state["xp"], expected_workers=expected_workers)
            validate_eq10_backend(helper_state["xp"], expected_workers=expected_workers)
            if helper_state["xp"].fft.dispatch is not helper_state["backend_slots"]["dispatch"]:
                raise SourceBindingError("S46H_STAGE_FFT_DISPATCH_CHANGED")
            scipy_fft = importlib.import_module("scipy.fft")
            if any(getattr(scipy_fft, name) is not value for name, value in helper_state["scipy_slots"].items()):
                raise SourceBindingError("S46H_STAGE_SCIPY_TRANSFORM_CHANGED")
            slots = helper_state["backend_slots"]
            if (
                np.absolute is not slots["absolute"]
                or np.square is not slots["square"]
                or np.multiply is not slots["multiply"]
                or np.divide is not slots["divide"]
            ):
                raise SourceBindingError("S46H_STAGE_UFUNC_CHANGED")
            for name, helper in helpers.items():
                if private_namespace.get(name) is not helper:
                    raise SourceBindingError(f"S46H_STAGE_HELPER_SLOT_CHANGED:{name}")
                seal = getattr(state, "_cpu_s46h_helper_seals", {}).get(name)
                if not isinstance(seal, Mapping):
                    raise SourceBindingError(f"S46H_HELPER_SEAL_MISSING:{name}")
                verify_seal(seal)
            stage_seal = getattr(state, "_cpu_s46h_stage_seal", None)
            if isinstance(stage_seal, Mapping):
                verify_seal(stage_seal)
        except BaseException:
            try:
                # A seal failure may identify the cleanup closure itself.
                poolset.close_and_drain()
            except BaseException:
                pass
            raise

    private_namespace[JOINT_GUARD_NAME] = guard_stage
    compiled, receipt = compile_combined_stage(
        stage_node,
        namespace=private_namespace,
        source_path=Path(source_path),
        entry_guard=guard_stage,
    )
    compiled.__name__ = getattr(original, "__name__", compiled.__name__)
    compiled.__qualname__ = getattr(original, "__qualname__", compiled.__qualname__)
    compiled.__module__ = getattr(original, "__module__", compiled.__module__)
    compiled.__doc__ = getattr(original, "__doc__", compiled.__doc__)
    compiled.__s46h_original_stage__ = original
    compiled.__s46h_original_code_identity__ = identity(original)
    compiled.__s46h_source_path__ = str(source_path)
    setattr(owner, attribute, compiled)
    active_ref["stage"] = compiled
    state._cpu_s46h_original_stage = original
    state._cpu_s46h_original_stage_seal = _seal(original)
    state._cpu_s46h_stage = compiled
    state._cpu_s46h_stage_owner = owner
    state._cpu_s46h_stage_attribute = attribute
    state._cpu_s46h_active_stage_ref = active_ref
    state._cpu_s46h_guard = helper_state["guard_before_helper"]
    state._cpu_s46h_guard_seal = _seal(helper_state["guard_before_helper"])
    state._cpu_s46h_stage_entry_guard = guard_stage
    state._cpu_s46h_stage_entry_guard_seal = _seal(guard_stage)
    state._cpu_s46h_scipy_slots = helper_state["scipy_slots"]
    state._cpu_s46h_r8_stage_original = r8_stage
    state._cpu_s46h_backend_slots = helper_state["backend_slots"]
    state._cpu_s46h_backend_binding = copy.deepcopy(helper_state["backend_binding"])
    state._cpu_s46h_original_backend_slots = dict(helper_state["backend_slots"])
    state._cpu_s46h_private_namespace = private_namespace
    state._cpu_s46h_helpers = helpers
    state._cpu_s46h_helper_components = helper_state["helper_components"]
    state._cpu_s46h_helper_pools = helper_state["helper_pools"]
    state._cpu_s46h_helper_originals = helper_state["helper_originals"]
    state._cpu_s46h_helper_original_seals = helper_state["helper_original_seals"]
    state._cpu_s46h_helper_seals = {name: _seal(helper) for name, helper in helpers.items()}
    state._cpu_s46h_stage_seal = getattr(compiled, "__joint_function_seal__", None)
    if not isinstance(state._cpu_s46h_stage_seal, Mapping):
        raise SourceBindingError("S46H_STAGE_SEAL_MISSING_AFTER_COMPILE")
    state._cpu_s46h_stage_receipt = copy.deepcopy(receipt)
    state._cpu_s46h_lineage = {
        "original_stage_code_identity": identity(original),
        "r8_expected_stage_code_identity": identity(expected),
        "r8_stage_code_identity": None if not callable(r8_stage) else identity(r8_stage),
        "instrumented_stage_code_identity": identity(compiled),
        "stage_owner": owner.__class__.__name__,
        "stage_attribute": attribute,
        "source_path": str(source_path),
        "source_ast_sha256": receipt["source_ast_sha256"],
        "stripped_ast_sha256": receipt["stripped_ast_sha256"],
        "helper_names": {"intensity": dict(INTENSITY_HELPER_NAMES), "eq10": dict(EQ10_HELPER_NAMES)},
        "scientific_expression_changes": 0,
        "stage_concurrency": False,
        "array_values_retained": False,
        "private_guard_before_helper_calls": True,
        "guard_code_identity": _code_identity(helper_state["guard_before_helper"]),
        "backend_binding": copy.deepcopy(helper_state["backend_binding"]),
    }
    return copy.deepcopy({**receipt, "lineage": state._cpu_s46h_lineage, "helper_registry_fixed": True})


def _install_pool_api(dispatch: Any, state: Any, poolset: _PoolSet) -> None:
    dispatch.cpu_s4_6h_state = state
    dispatch.cpu_s4_6h_poolset = poolset
    dispatch.cpu_s4_6h_intensity_pool = poolset.intensity
    dispatch.cpu_s4_6h_eq10_pool = poolset.eq10
    dispatch.cpu_s4_6h_c_pool = poolset.c
    dispatch.close_combined_pools = lambda: finalize_factory(dispatch)
    dispatch.combined_pool_snapshot = lambda: poolset.snapshots()
    def take_combined_pool_records() -> dict[str, list[dict[str, Any]]]:
        last = getattr(dispatch, "_cpu_s4_6h_last_call_records", None)
        direct = poolset.records()
        dispatch._cpu_s4_6h_last_call_records = {}
        if isinstance(last, Mapping):
            merged = copy.deepcopy(dict(last))
            for name, rows in direct.items():
                merged.setdefault(name, []).extend(rows)
            return merged
        return direct

    dispatch.take_combined_pool_records = take_combined_pool_records
    dispatch.close_intensity_pool = lambda: poolset.intensity.close()
    dispatch.close_eq10_pool = lambda: poolset.eq10.close()
    dispatch.close_c_pool = lambda: poolset.c.close()
    dispatch.intensity_pool_snapshot = lambda: poolset.intensity.snapshot()
    dispatch.eq10_pool_snapshot = lambda: poolset.eq10.snapshot()
    dispatch.c_pool_snapshot = lambda: poolset.c.snapshot()
    dispatch.take_intensity_pool_records = lambda: poolset.intensity.take_records()
    dispatch.take_eq10_pool_records = lambda: poolset.eq10.take_records()
    dispatch.take_c_pool_records = lambda: poolset.c.take_records()


def _wrap_propagator_for_failure(wrapped: Callable[..., Any], dispatch: Any) -> Callable[..., Any]:
    """Retire every semantic pool if any stage/callsite task fails."""

    @functools.wraps(wrapped)
    def combined_propagator(*args: Any, **kwargs: Any) -> Any:
        try:
            result = wrapped(*args, **kwargs)
            poolset = getattr(dispatch, "cpu_s4_6h_poolset", None)
            if isinstance(poolset, _PoolSet):
                dispatch._cpu_s4_6h_last_call_records = poolset.records()
            return result
        except BaseException as error:
            try:
                finalize_factory(dispatch, reason="TASK_FAILURE")
            except BaseException as cleanup:
                error.add_note("S46H_CLOSE_FAILURE:" + type(cleanup).__name__)
            raise

    for name, value in getattr(wrapped, "__dict__", {}).items():
        if name not in {"__wrapped__"}:
            setattr(combined_propagator, name, value)
    combined_propagator.__g2_inner__ = getattr(wrapped, "__g2_inner__", None)
    combined_propagator.__g2_actual_code__ = getattr(wrapped, "__g2_actual_code__", None)
    combined_propagator.__g2_expected_code__ = getattr(wrapped, "__g2_expected_code__", None)
    combined_propagator.__s46h_failure_wrapper__ = True
    return combined_propagator


def _component_pool_types(role: str) -> tuple[type[Any], type[Any], type[Any]]:
    if _normalise_role(role) == ROLE_REFERENCE:
        return NullIntensityPool, NullEq10Pool, NullCPool
    return IntensityPool, Eq10Pool, CPool


def finalize_factory(dispatch: Any, *, reason: str = "NORMAL") -> dict[str, Any]:
    """Close/join/drain all three pools before normal output is published."""

    existing = getattr(dispatch, "_cpu_s4_6h_finalized", None)
    if isinstance(existing, dict):
        return copy.deepcopy(existing)
    poolset = getattr(dispatch, "cpu_s4_6h_poolset", None)
    if not isinstance(poolset, _PoolSet):
        raise SourceBindingError("S46H_POOLSET_MISSING")
    finalized = poolset.close_and_drain()
    last_call = getattr(dispatch, "_cpu_s4_6h_last_call_records", None)
    if isinstance(last_call, Mapping):
        # The wrapper drains after each propagation call.  Keep the public
        # finalizer shape stable by merging those scalar rows into the owning
        # component rather than exposing a second parallel activity tree.
        for name, rows in last_call.items():
            if name not in finalized or not isinstance(finalized[name], dict):
                continue
            activity = finalized[name].setdefault("activity", [])
            if isinstance(activity, list) and isinstance(rows, list):
                activity.extend(copy.deepcopy(rows))
        dispatch._cpu_s4_6h_last_call_records = {}
    finalized.update({
        "schema": CANDIDATE_SCHEMA + ".finalized.v1",
        "reason": str(reason),
        "combined_semantic_id": COMBINED_SEMANTIC_ID,
        "all_pools_retired_before_output": bool(finalized.get("all_closed")),
    })
    live_pools = [
        name
        for name, row in finalized.items()
        if isinstance(row, dict)
        and "pool" in row
        and bool(row["pool"].get("active_thread_ids"))
    ]
    close_errors = tuple(finalized.get("close_errors", ()))
    if close_errors or not bool(finalized.get("all_closed")) or live_pools:
        finalized["all_pools_retired_before_output"] = False
        finalized["finalization_failed"] = True
        finalized["live_pools"] = tuple(live_pools)
        dispatch._cpu_s4_6h_finalization_failure = copy.deepcopy(finalized)
        raise SourceBindingError("S46H_POOL_FINALIZATION_FAILED")
    dispatch._cpu_s4_6h_finalized = copy.deepcopy(finalized)
    state = getattr(dispatch, "cpu_s4_6h_state", None)
    if state is not None:
        state._cpu_s4_6h_finalized = copy.deepcopy(finalized)
    return copy.deepcopy(finalized)


def metadata(dispatch: Any) -> dict[str, Any]:
    """Return a bounded copy of combined binding/lifecycle metadata."""

    binding = getattr(dispatch, "cpu_s4_6h_binding", None)
    if not isinstance(binding, Mapping):
        raise SourceBindingError("S46H_BINDING_METADATA_MISSING")
    return copy.deepcopy(dict(binding))


def verify_bindings(dispatch: Any) -> dict[str, Any]:
    """Verify joint stage, all component pools and all source closures."""

    role = _normalise_role(getattr(dispatch, "cpu_s4_6h_role", None))
    state = getattr(dispatch, "cpu_s4_6h_state", None)
    poolset = getattr(dispatch, "cpu_s4_6h_poolset", None)
    if state is None or not isinstance(poolset, _PoolSet):
        raise SourceBindingError("S46H_STATE_OR_POOLSET_MISSING")
    expected_types = _component_pool_types(role)
    for (name, pool), expected in zip(poolset.values(), expected_types):
        if type(pool) is not expected:
            raise SourceBindingError(f"S46H_{name.upper()}_POOL_TYPE_CHANGED")
    if getattr(dispatch, "cpu_s4_6h_role", None) != role:
        raise SourceBindingError("S46H_ROLE_BINDING_CHANGED")
    stored_identity = getattr(state, "_cpu_s46h_source_identity", None)
    if not isinstance(stored_identity, Mapping):
        raise SourceBindingError("S46H_SOURCE_IDENTITY_MISSING")
    current_identity = source_identity()
    if current_identity != stored_identity:
        raise SourceBindingError("S46H_SOURCE_IDENTITY_CHANGED")
    dispatch_identity = getattr(dispatch, "cpu_s4_6h_source_identity", None)
    if not isinstance(dispatch_identity, Mapping) or dict(dispatch_identity) != dict(stored_identity):
        raise SourceBindingError("S46H_DISPATCH_SOURCE_IDENTITY_CHANGED")
    stored_binding = getattr(state, "_cpu_s46h_binding", None)
    dispatch_binding = getattr(dispatch, "cpu_s4_6h_binding", None)
    if not isinstance(stored_binding, Mapping) or not isinstance(dispatch_binding, Mapping):
        raise SourceBindingError("S46H_BINDING_METADATA_MISSING")
    comparable_dispatch_binding = dict(dispatch_binding)
    comparable_dispatch_binding.pop("binding_check", None)
    if dict(stored_binding) != comparable_dispatch_binding:
        raise SourceBindingError("S46H_BINDING_METADATA_CHANGED")

    try:
        c_check = verify_c_bindings(dispatch)
    except BaseException as exc:
        raise SourceBindingError("S46H_C_BINDING_FAILED") from exc
    s45a = importlib.import_module("tools.cpu_s4_5a_l1_residual.instrument")
    try:
        s45a_check = s45a.verify_bindings(state)
    except BaseException as exc:
        raise SourceBindingError("S46H_S45A_BINDING_FAILED") from exc

    # Resolve the active owner independently from the saved original.  The
    # candidate replaces the owner's stage slot; asking _stage_owner() after
    # that replacement would mistake the compiled joint stage for the frozen
    # original and would make the identity check self-referential.
    if role == ROLE_REFERENCE:
        owner, attribute, original, _r8_stage = _stage_owner(state)
        if getattr(state, "_cpu_s46h_stage", None) is not None:
            raise SourceBindingError("S46H_REFERENCE_STAGE_REPLACED")
        if getattr(state, "_cpu_s46h_parent_stage_identity", None) != _code_identity(original):
            raise SourceBindingError("S46H_REFERENCE_STAGE_CHANGED")
        return {
            "schema": CANDIDATE_SCHEMA + ".binding.v1",
            "role": role,
            "combined_semantic_id": COMBINED_SEMANTIC_ID,
            "stage_replaced": False,
            "component_pools": poolset.snapshots(),
            "source_closure_bound": True,
            "c_binding": c_check,
            "s45a_binding": s45a_check,
        }

    owner = getattr(state, "_cpu_s46h_stage_owner", None)
    attribute = getattr(state, "_cpu_s46h_stage_attribute", None)
    stage = getattr(state, "_cpu_s46h_stage", None)
    original = getattr(state, "_cpu_s46h_original_stage", None)
    if owner is None or not isinstance(attribute, str) or not callable(original):
        raise SourceBindingError("S46H_ORIGINAL_STAGE_BINDING_MISSING")
    active_owner_stage = getattr(owner, attribute, None)
    if active_owner_stage is not stage:
        raise SourceBindingError("S46H_ACTIVE_STAGE_OBJECT_CHANGED")
    if not callable(stage) or getattr(owner, attribute, None) is not stage:
        raise SourceBindingError("S46H_ACTIVE_STAGE_OBJECT_CHANGED")
    if getattr(state, "_cpu_s46h_stage_owner", None) is not owner or getattr(state, "_cpu_s46h_stage_attribute", None) != attribute:
        raise SourceBindingError("S46H_STAGE_OWNER_BINDING_CHANGED")
    if getattr(state, "_cpu_s46h_original_stage", None) is not original:
        raise SourceBindingError("S46H_ORIGINAL_STAGE_BINDING_CHANGED")
    if getattr(state, "_cpu_s46h_parent_stage_identity", None) != _code_identity(original):
        raise SourceBindingError("S46H_ORIGINAL_STAGE_CODE_CHANGED")
    original_seal = getattr(state, "_cpu_s46h_original_stage_seal", None)
    if not isinstance(original_seal, Mapping):
        raise SourceBindingError("S46H_ORIGINAL_STAGE_SEAL_MISSING")
    if original_seal.get("function") is not original:
        raise SourceBindingError("S46H_ORIGINAL_STAGE_BINDING_CHANGED")
    verify_seal(original_seal)
    original_globals = getattr(original, "__globals__", None)
    original_slots = getattr(state, "_cpu_s46h_original_backend_slots", None)
    if not isinstance(original_globals, dict) or not isinstance(original_slots, Mapping):
        raise SourceBindingError("S46H_ORIGINAL_BACKEND_SLOTS_MISSING")
    if (
        original_globals.get("xp") is not original_slots.get("xp")
        or original_globals.get("_np") is not np
        or ("np" in original_globals and original_globals.get("np") is not np)
    ):
        raise SourceBindingError("S46H_ORIGINAL_BACKEND_SLOT_CHANGED")
    if (
        original_slots.get("numpy") is not np
        or original_slots.get("absolute") is not np.absolute
        or original_slots.get("square") is not np.square
        or original_slots.get("multiply") is not np.multiply
        or original_slots.get("divide") is not np.divide
        or original_slots.get("underlying") is not np
    ):
        raise SourceBindingError("S46H_ORIGINAL_NUMPY_SLOT_CHANGED")
    active_ref = getattr(state, "_cpu_s46h_active_stage_ref", None)
    if not isinstance(active_ref, dict) or active_ref.get("stage") is not stage:
        raise SourceBindingError("S46H_ACTIVE_STAGE_GUARD_REFERENCE_CHANGED")
    private_namespace = getattr(state, "_cpu_s46h_private_namespace", None)
    helpers = getattr(state, "_cpu_s46h_helpers", None)
    helper_components = getattr(state, "_cpu_s46h_helper_components", None)
    helper_pools = getattr(state, "_cpu_s46h_helper_pools", None)
    helper_seals = getattr(state, "_cpu_s46h_helper_seals", None)
    helper_originals = getattr(state, "_cpu_s46h_helper_originals", None)
    helper_original_seals = getattr(state, "_cpu_s46h_helper_original_seals", None)
    if not isinstance(private_namespace, dict) or not isinstance(helpers, Mapping) or not isinstance(helper_seals, Mapping):
        raise SourceBindingError("S46H_HELPER_BINDING_MISSING")
    if not isinstance(helper_originals, Mapping) or not isinstance(helper_original_seals, Mapping):
        raise SourceBindingError("S46H_ORIGINAL_HELPER_SEALS_MISSING")
    expected_names = set(INTENSITY_HELPER_NAMES.values()) | set(EQ10_HELPER_NAMES.values())
    if (
        set(helpers) != expected_names
        or set(helper_seals) != expected_names
        or set(helper_originals) != expected_names
        or set(helper_original_seals) != expected_names
    ):
        raise SourceBindingError("S46H_HELPER_REGISTRY_CHANGED")
    stage_globals = getattr(stage, "__globals__", None)
    if stage_globals is not private_namespace:
        raise SourceBindingError("S46H_STAGE_GLOBAL_NAMESPACE_CHANGED")
    expected_workers = getattr(state, "_cpu_s46h_expected_backend_workers", None)
    try:
        intensity_backend = validate_intensity_backend(stage_globals.get("xp"), expected_workers=expected_workers)
        eq10_backend = validate_eq10_backend(stage_globals.get("xp"), expected_workers=expected_workers)
    except (IntensityInstrumentationBindingError, Eq10InstrumentationBindingError) as exc:
        raise SourceBindingError(str(exc)) from exc
    if stage_globals.get("_np") is not np or intensity_backend["dispatch_object"] is not eq10_backend["dispatch_object"]:
        raise SourceBindingError("S46H_STAGE_BACKEND_BINDING_CHANGED")
    if getattr(state, "_cpu_s46h_backend_binding", None) != _backend_metadata(intensity_backend):
        raise SourceBindingError("S46H_PARENT_BACKEND_BINDING_CHANGED")
    for name, helper in helpers.items():
        if private_namespace.get(name) is not helper:
            raise SourceBindingError(f"S46H_STAGE_HELPER_SLOT_CHANGED:{name}")
        component = helper_components.get(name)
        pool = helper_pools.get(name)
        expected_pool = poolset.intensity if component == "intensity" else poolset.eq10
        if pool is not expected_pool or getattr(helper, "__cpu_s46h_pool__", None) is not expected_pool:
            raise SourceBindingError(f"S46H_HELPER_POOL_CHANGED:{name}")
        if getattr(helper, "__cpu_s46h_component__", None) != component:
            raise SourceBindingError(f"S46H_HELPER_COMPONENT_CHANGED:{name}")
        original_helper = helper_originals.get(name)
        if getattr(helper, "__cpu_s46h_original__", None) is not original_helper:
            raise SourceBindingError(f"S46H_HELPER_ORIGINAL_CHANGED:{name}")
        original_helper_seal = helper_original_seals.get(name)
        if not callable(original_helper) or not isinstance(original_helper_seal, Mapping):
            raise SourceBindingError(f"S46H_ORIGINAL_HELPER_SEAL_MISSING:{name}")
        if original_helper_seal.get("function") is not original_helper:
            raise SourceBindingError(f"S46H_ORIGINAL_HELPER_CHANGED:{name}")
        verify_seal(original_helper_seal)
        verify_seal(helper_seals[name])
    entry_guard = getattr(state, "_cpu_s46h_stage_entry_guard", None)
    entry_guard_seal = getattr(state, "_cpu_s46h_stage_entry_guard_seal", None)
    stage_seal = getattr(state, "_cpu_s46h_stage_seal", None)
    for suffix in ("guard", "close_failure"):
        function = getattr(state, "_cpu_s46h_" + suffix, None)
        seal = getattr(state, "_cpu_s46h_" + suffix + "_seal", None)
        if not isinstance(seal, Mapping) or seal.get("function") is not function:
            raise SourceBindingError(f"S46H_RUNTIME_GUARD_CHANGED:{suffix}")
        verify_seal(seal)
    if not callable(entry_guard) or not isinstance(entry_guard_seal, Mapping) or not isinstance(stage_seal, Mapping):
        raise SourceBindingError("S46H_STAGE_SEALS_MISSING")
    verify_seal(entry_guard_seal)
    verify_seal(stage_seal)
    verify_compiled_stage(stage, seal=stage_seal, namespace=private_namespace, entry_guard=entry_guard)
    return {
        "schema": CANDIDATE_SCHEMA + ".binding.v1",
        "role": role,
        "combined_semantic_id": COMBINED_SEMANTIC_ID,
        "stage_replaced": True,
        "stage_code_identity": _code_identity(stage),
        "helper_count": len(helpers),
        "component_pools": poolset.snapshots(),
        "source_closure_bound": True,
        "stage_concurrency": False,
        "c_binding": c_check,
        "s45a_binding": s45a_check,
        "lineage": copy.deepcopy(getattr(state, "_cpu_s46h_lineage", {})),
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
    """Build the common L1 reference or joint candidate six-tuple."""

    mode_name = _normalise_mode(mode)
    role_name = _normalise_role(role)
    resources = _resolve_resources(local_engineering=bool(local_engineering), pool_override=pool_override)
    adapter = importlib.import_module("tools.cpu_s4_5b_steady_cost.adapter")
    if recorder_pair is None:
        recorder_pair = adapter.make_recorder_pair(False)
    if type(recorder_pair) is not getattr(adapter, "RecorderPair", object):
        raise CandidateUnsupported("S46H_RECORDER_PAIR_EXACT_TYPE_REQUIRED")
    if bool(getattr(recorder_pair, "profile", True)):
        raise CandidateUnsupported("S46H_RECORDER_PAIR_PROFILE_MUST_BE_OFF")

    poolset = _make_pools(role_name, resources)
    try:
        propagator, dispatch, state = _build_parent(
            local_engineering=bool(local_engineering),
            resources=resources,
            recorder_pair=recorder_pair,
        )
        state._cpu_s46h_expected_backend_workers = {
            "temporal_workers": int(resources["temporal_workers"]),
            "spatial_workers": int(resources["spatial_workers"]),
        }
        stage_owner, _stage_attribute, parent_stage, parent_r8_stage = _stage_owner(state)
        state._cpu_s46h_parent_stage_identity = _code_identity(parent_stage)
        state._cpu_s46h_parent_r8_stage_identity = (
            None if not callable(parent_r8_stage) else _code_identity(parent_r8_stage)
        )
        s45a = importlib.import_module("tools.cpu_s4_5a_l1_residual.instrument")
        s45a.install_propagation_wrappers(propagator, state)
        pre_check = s45a.verify_bindings(state)
        source = source_identity()
        c_role = C_ROLE_REFERENCE if role_name == ROLE_REFERENCE else C_ROLE_CANDIDATE
        # Exactly one C/G2 attach for this role.  The Raman stage is compiled
        # jointly only after the C clone is bound.
        wrapped, actual_dispatch, clock, runtime_binding, audit, lease, c_private = attach_with_c(
            propagator,
            dispatch,
            role=c_role,
            mode=mode_name,
            capture_root=Path(capture_root).resolve(),
            recorder=recorder_pair.b,
            pool=poolset.c,
        )
        if actual_dispatch is not dispatch:
            raise SourceBindingError("S46H_ACTUAL_DISPATCH_BINDING_CHANGED")
        # Preserve the existing C verifier's public/private pool contract in
        # addition to the combined-owned aliases installed below.
        install_c_pool_api(actual_dispatch, poolset.c)
        actual_dispatch.cpu_s4_6e_c_role = c_role
        s45a.bind_active_propagation(wrapped, state)
        post_check = s45a.verify_bindings(state)

        stage_receipt: dict[str, Any] = {"stage_replaced": False, "schema": INSTRUMENT_SCHEMA}
        if role_name == ROLE_CANDIDATE:
            stage_receipt = _install_joint_stage(state, poolset)
        _install_pool_api(actual_dispatch, state, poolset)
        actual_dispatch.cpu_s4_6h_role = role_name
        actual_dispatch.cpu_s4_6h_candidate_schema = CANDIDATE_SCHEMA
        actual_dispatch.cpu_s4_6h_source_identity = copy.deepcopy(source)
        actual_dispatch.cpu_s4_6h_c_private = copy.deepcopy({
            "math_receipt": c_private.get("math_receipt"),
            "c_source_binding": runtime_binding.get("c_source_binding"),
            "c_role": runtime_binding.get("c_role"),
        })
        state._cpu_s46h_source_identity = copy.deepcopy(source)
        state._cpu_s46h_c_private = copy.deepcopy(actual_dispatch.cpu_s4_6h_c_private)
        binding = {
            "schema": CANDIDATE_SCHEMA + ".binding.v1",
            "role": role_name,
            "mode": mode_name,
            "candidate": DEFAULT_VARIANT,
            "combined_semantic_id": COMBINED_SEMANTIC_ID,
            "resource_contract": copy.deepcopy(resources),
            "stage": copy.deepcopy(stage_receipt),
            "component_pools": {
                "intensity": poolset.intensity.snapshot(),
                "eq10": poolset.eq10.snapshot(),
                "c": poolset.c.snapshot(),
            },
            "source_closure": copy.deepcopy(source["source_closure"]),
            "intensity_source": copy.deepcopy(source["source_closure"]["intensity_p24"]),
            "eq10_source": copy.deepcopy(source["source_closure"]["eq10_p48"]),
            "c_source": copy.deepcopy(source["source_closure"]["phase_absorb_c_p24"]),
            "runtime_wrapper": copy.deepcopy(runtime_binding),
            "c_private": copy.deepcopy(actual_dispatch.cpu_s4_6h_c_private),
            "s45a_pre_binding_check": copy.deepcopy(pre_check),
            "s45a_post_binding_check": copy.deepcopy(post_check),
            "production_qualified": False,
            "scientific_qualification": False,
            "stage_concurrency": False,
            "recorder_pair_bound": True,
            "all3_source_closures_bound": True,
        }
        state._cpu_s46h_binding = copy.deepcopy(binding)
        actual_dispatch.cpu_s4_6h_binding = copy.deepcopy(binding)
        _install_pool_api(actual_dispatch, state, poolset)
        actual_dispatch.cpu_s4_6h_binding["binding_check"] = copy.deepcopy(verify_bindings(actual_dispatch))
        actual_dispatch.cpu_s4_6h_finalizer = finalize_factory
        final_propagator = _wrap_propagator_for_failure(wrapped, actual_dispatch)
        return (
            final_propagator,
            actual_dispatch,
            clock,
            copy.deepcopy(actual_dispatch.cpu_s4_6h_binding),
            audit,
            lease,
        )
    except BaseException:
        try:
            poolset.close_and_drain()
        except BaseException:
            pass
        raise


def build_reference(
    mode: str,
    *,
    capture_root: str | Path,
    local_engineering: bool = False,
    pool_override: int | None = None,
    recorder_pair: Any | None = None,
) -> tuple[Any, Any, Any, dict[str, Any], Any, Any]:
    return build_role(
        mode,
        capture_root=capture_root,
        role=ROLE_REFERENCE,
        local_engineering=local_engineering,
        pool_override=pool_override,
        recorder_pair=recorder_pair,
    )


def build_candidate(
    mode: str,
    *,
    capture_root: str | Path,
    local_engineering: bool = False,
    pool_override: int | None = None,
    recorder_pair: Any | None = None,
) -> tuple[Any, Any, Any, dict[str, Any], Any, Any]:
    return build_role(
        mode,
        capture_root=capture_root,
        role=ROLE_CANDIDATE,
        local_engineering=local_engineering,
        pool_override=pool_override,
        recorder_pair=recorder_pair,
    )


# Private spellings used by review/qualification callers.
_verify_joint_bindings = verify_bindings
_finalize_combined = finalize_factory
_combined_metadata = metadata


__all__ = [
    "CANDIDATE_SCHEMA",
    "COMBINED_SEMANTIC_ID",
    "CandidateUnsupported",
    "DEFAULT_VARIANT",
    "HPC_CONTRACT",
    "LOCAL_CONTRACT",
    "ROLE_CANDIDATE",
    "ROLE_REFERENCE",
    "PoolSet",
    "SourceBindingError",
    "build_role",
    "build_reference",
    "build_candidate",
    "finalize_factory",
    "metadata",
    "source_identity",
    "verify_bindings",
    "_combined_metadata",
    "_finalize_combined",
    "_verify_joint_bindings",
]
