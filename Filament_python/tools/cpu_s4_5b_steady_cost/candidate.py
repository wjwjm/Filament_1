"""CPU-S4-5B roles over the accepted S4-4A/S4-5A L1 candidate.

The module owns only the private B recorder, AST seam, and role metadata.  The
scientific factory, linear factor cache, G3 lifecycle, strict input guards,
and six-tuple G2 runtime contract are borrowed from S4-5A without changing
their source.  B's propagation clone is built through :func:`attach_with_b`
before the final G2 wrapper captures its code.
"""

from __future__ import annotations

import ast
import copy
import importlib
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from .adapter import (
    ADAPTER_SCHEMA,
    DOMAIN,
    MODES,
    PURE_OFF_SMALLGRID,
    QUALIFY_BOUNDARY,
    ROLE_OFF,
    ROLE_PROFILE,
    ROLES,
    AdapterBindingError,
    RecorderPair,
    accept_s45a_recorder,
    attach_with_b,
    make_recorder_pair,
)
from .instrument import (
    _same_fixed_backend_callable,
    EQ10_HELPER_NAMES,
    INSTRUMENT_SCHEMA,
    InstrumentationBindingError,
    code_identity,
    compile_function_node,
    instrument_eq10_stage,
    make_eq10_call_helpers,
    seal_callable,
    seal_fixed_backend_registry,
    verify_callable_seal,
    verify_fixed_backend_registry,
)
from .timing import (
    DEFAULT_MAX_RECORDS,
    NEW_A8_LABELS,
    NEW_B8_LABELS,
    SCHEMA,
    TimingRecorder,
)
from tools.hpc_ops import provenance_v2 as _provenance


CANDIDATE_SCHEMA = "filament.cpu_s4_5b_steady_cost.candidate.v1"
DEFAULT_VARIANT = "L1_LINEAR_FACTOR_CACHE_V1"
LOCAL_POOL_CONTRACT = {
    "numba_pool": 1,
    "outer_mask": 1,
    "temporal_workers": 1,
    "spatial_workers": 1,
    "raman_workers": 1,
    "scope": "LOCAL_ENGINEERING_ONLY",
    "hpc_allowed": False,
}
HPC_POOL_CONTRACT = {
    "numba_pool": 48,
    "outer_mask": 24,
    "temporal_workers": 48,
    "spatial_workers": 24,
    "raman_workers": 48,
    "scope": "HPC_FIXED_W48_W24",
    "hpc_allowed": True,
}
CACHE_CONTRACT = {
    "capacity_mib": 256,
    "max_tables": 2,
    "policy": "S45A_ACCEPTED_LINEAR_FACTOR_CACHE_UNCHANGED",
}


class CandidateUnsupported(RuntimeError):
    """The fixed S45B role cannot satisfy the requested route."""


class SourceBindingError(RuntimeError):
    """A frozen parent or private B source identity failed closed."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _owned_source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    names = (
        "__init__.py",
        "timing.py",
        "instrument.py",
        "adapter.py",
        "candidate.py",
        "test_timing.py",
        "test_bindings.py",
        "README.md",
    )
    files: dict[str, dict[str, Any]] = {}
    repo = _repo_root()
    for name in names:
        path = (root / name).resolve(strict=True)
        raw = path.read_bytes()
        files[name] = {
            "path": str(path),
            "size_bytes": int(len(raw)),
            "raw_sha256": _provenance.raw_sha256(raw),
            "classification": "external",
            "hash_scope": "raw_bytes",
        }
        relative = path.relative_to(repo).as_posix()
        try:
            oid = _provenance._git_blob_oid(repo, relative)
        except _provenance.ProvenanceError:
            # Before the first checkpoint these are local engineering artifacts.
            # Formal entry admission separately requires clean committed sources.
            if os.environ.get("SLURM_JOB_ID"):
                raise SourceBindingError("S45B_COMMITTED_OWNED_SOURCE_REQUIRED")
        else:
            blob = _provenance._read_git_blob(repo, oid)
            if _provenance.canonical_lf_bytes(raw) != _provenance.canonical_lf_bytes(blob):
                raise SourceBindingError("S45B_OWNED_SOURCE_CANONICAL_BLOB_MISMATCH")
            files[name].update(
                classification="tracked_text",
                hash_scope=_provenance.TRACKED_HASH_SCOPE,
                git_blob_oid=oid,
                canonical_lf_sha256=_provenance.canonical_lf_sha256(blob),
                relative_path=relative,
                raw_byte_observation="CURRENT_CHECKOUT_ONLY_NOT_CROSS_PLATFORM_IDENTITY",
            )
    composite = (
        "producer_module:tools.cpu_s4_5b_steady_cost\n"
        + "".join(f"{name}:{files[name]['raw_sha256']}\n" for name in sorted(files))
    ).encode("utf-8")
    return {
        "schema": CANDIDATE_SCHEMA + ".owned_sources",
        "producer_module_identity": "tools.cpu_s4_5b_steady_cost",
        "files": files,
        "composite_raw_sha256": _provenance.raw_sha256(composite),
        "instrumentation_schema": INSTRUMENT_SCHEMA,
    }


def source_identity() -> dict[str, Any]:
    parent = importlib.import_module("tools.cpu_s4_5a_l1_residual.candidate")
    return {
        "schema": CANDIDATE_SCHEMA,
        "candidate_name": DEFAULT_VARIANT,
        "domain": DOMAIN,
        "roles": list(ROLES),
        "parent_candidate": "tools.cpu_s4_5a_l1_residual",
        "parent_source_identity": copy.deepcopy(parent.source_identity()),
        "owned_source_identity": _owned_source_identity(),
        "instrumentation_schema": INSTRUMENT_SCHEMA,
        "new_math_algorithm": False,
        "cache_contract": copy.deepcopy(CACHE_CONTRACT),
    }


def make_recorder(
    profile: bool,
    *,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> RecorderPair:
    """Return the B recorder plus the exact S4-5A companion."""

    return make_recorder_pair(profile, max_records=max_records)


def _normalise_mode(mode: str) -> str:
    value = str(mode).upper()
    if value == "PURE_OFF":
        value = PURE_OFF_SMALLGRID
    if value not in MODES:
        raise CandidateUnsupported(f"S45B_MODE_UNSUPPORTED:{mode}")
    return value


def _resolve_resources(*, local_engineering: bool, pool_override: int | None) -> dict[str, Any]:
    if local_engineering:
        if os.environ.get("SLURM_JOB_ID"):
            raise CandidateUnsupported("LOCAL_ENGINEERING_ONLY_FORBIDDEN_ON_SLURM")
        if pool_override != 1:
            raise CandidateUnsupported("LOCAL_ENGINEERING_REQUIRES_EXPLICIT_POOL_OVERRIDE_1")
        return copy.deepcopy(LOCAL_POOL_CONTRACT)
    if pool_override is not None:
        raise CandidateUnsupported("HPC_ROLE_DOES_NOT_ACCEPT_POOL_OVERRIDE")
    return copy.deepcopy(HPC_POOL_CONTRACT)


def _install_eq10_stage(state: Any, recorder: TimingRecorder) -> dict[str, Any]:
    """Rebuild the existing R8/S45A stage AST and add only B8 nodes."""

    if type(recorder) is not TimingRecorder:
        raise InstrumentationBindingError("B_RECORDER_EXACT_TYPE_REQUIRED")
    original_stage = getattr(state, "_internal_stage_inner", None)
    r8_original_stage = getattr(state, "_cpu_s45a_original_stage", None)
    if r8_original_stage is None:
        r8_original_stage = original_stage
    source_globals = getattr(original_stage, "__globals__", None)
    candidate_namespace = getattr(state, "_candidate_namespace", None)
    if (
        not callable(original_stage)
        or not callable(r8_original_stage)
        or not isinstance(source_globals, dict)
    ):
        raise InstrumentationBindingError("B8_STAGE_PRIVATE_BINDING_MISSING")
    if not isinstance(candidate_namespace, dict):
        raise InstrumentationBindingError("B8_CANDIDATE_NAMESPACE_MISSING")
    transform = candidate_namespace.get("_transform_stage")
    if not callable(transform):
        raise InstrumentationBindingError("B8_R8_STAGE_TRANSFORM_MISSING")

    s45a_instrument = importlib.import_module("tools.cpu_s4_5a_l1_residual.instrument")
    s45a_code_identity = getattr(s45a_instrument, "_code_identity", None)
    expected_builder = getattr(s45a_instrument, "_expected_transformed_node", None)
    wrap_stage = getattr(s45a_instrument, "_wrap_stage_sites", None)
    canonical_tree = getattr(s45a_instrument, "_canonical_tree", None)
    if (
        not callable(expected_builder)
        or not callable(wrap_stage)
        or not callable(canonical_tree)
        or not callable(s45a_code_identity)
    ):
        raise InstrumentationBindingError("B8_S45A_AST_SEAM_UNAVAILABLE")
    source_path, stage_node, _ = expected_builder(
        name="isaacs_raman_stage",
        transform_name="_transform_stage",
        candidate_namespace=candidate_namespace,
    )
    # Recreate the accepted S45A outer stage sites first; B8 is nested within
    # the existing RAMAN_ENERGY tree and therefore does not remove its clock.
    s45a_profile = bool(getattr(state, "_cpu_s45a_profile", False))
    if s45a_profile:
        if getattr(original_stage, "__cpu_s45a_original__", None) is not r8_original_stage:
            raise InstrumentationBindingError("B8_S45A_STAGE_ORIGINAL_NOT_R8")
        a_binding = getattr(state, "_cpu_s45a_bindings", {}).get("internal_stage", {})
        if a_binding.get("original_code_identity") != s45a_code_identity(r8_original_stage):
            raise InstrumentationBindingError("B8_S45A_R8_STAGE_CODE_BINDING_MISMATCH")
        a_probe_node = copy.deepcopy(stage_node)
        wrap_stage(a_probe_node)
        a_probe = s45a_instrument._compile_function_node(
            a_probe_node,
            namespace=dict(source_globals),
            source_path=source_path,
        )
        if s45a_code_identity(a_probe) != s45a_code_identity(original_stage):
            raise InstrumentationBindingError("B8_S45A_AST_CODE_PROOF_FAILED")
        # PROFILE already owns a private S45A namespace; preserve its outer
        # RAMAN_INTENSITY/RAMAN_ENERGY scopes and extend that namespace only.
        s45a_sites = wrap_stage(stage_node)
        # S4A's guard contract requires the active stage to remain in its
        # existing private namespace.  This is not a module-global mutation;
        # B helper slots are sealed individually below.
        private_globals = source_globals
    else:
        # OFF still follows the same source path, but must not patch the
        # production Raman module globals.  Build a private namespace with no
        # S45A fine scopes and let B scopes become no-ops.
        s45a_sites = {"sites": []}
        private_globals = dict(source_globals)
    xp = private_globals.get("xp")
    fft_namespace = getattr(xp, "fft", None)
    registry = {
        "EQ10_FFT": getattr(fft_namespace, "fft", None),
        "EQ10_IFFT": getattr(fft_namespace, "ifft", None),
        "EQ10_SUM": getattr(xp, "sum", None),
    }
    backend_seals = seal_fixed_backend_registry(registry)
    b_helpers = make_eq10_call_helpers(recorder, registry)
    private_globals.update(b_helpers)
    # Scope and marker helpers are supplied by candidate.bindings after the
    # state is built; the private function names are fixed in the AST.
    from .instrument import make_marker_helper, make_scope_helper

    private_globals["__cpu_s45b_scope__"] = make_scope_helper(recorder)
    private_globals["__cpu_s45b_marker__"] = make_marker_helper(recorder)
    b_receipt = instrument_eq10_stage(stage_node)
    instrumented = compile_function_node(
        stage_node,
        namespace=private_globals,
        source_path=source_path,
    )
    instrumented.__dict__.update(dict(getattr(original_stage, "__dict__", {})))
    instrumented.__name__ = getattr(original_stage, "__name__", instrumented.__name__)
    instrumented.__qualname__ = getattr(original_stage, "__qualname__", instrumented.__qualname__)
    instrumented.__module__ = getattr(original_stage, "__module__", instrumented.__module__)
    instrumented.__doc__ = getattr(original_stage, "__doc__", instrumented.__doc__)
    instrumented.__cpu_s45a_instrumented__ = True
    instrumented.__cpu_s45a_original__ = r8_original_stage
    instrumented.__cpu_s45a_original_code_identity__ = code_identity(r8_original_stage)
    instrumented.__cpu_s45a_label__ = "RAMAN_STAGE_BODY"
    instrumented.__cpu_s45a_stage_sites__ = copy.deepcopy(s45a_sites)
    state._internal_stage_inner = instrumented
    if getattr(state, "_cpu_s45a_original_stage", None) is None:
        state._cpu_s45a_original_stage = r8_original_stage
    if s45a_profile and "internal_stage" in state._cpu_s45a_bindings:
        state._cpu_s45a_bindings["internal_stage"]["instrumented_code_identity"] = s45a_code_identity(
            instrumented
        )
    state._cpu_s45b_eq10_helpers = b_helpers
    state._cpu_s45b_eq10_backend_registry = dict(registry)
    state._cpu_s45b_eq10_backend_seals = backend_seals
    state._cpu_s45b_backend_matcher_seal = seal_callable(_same_fixed_backend_callable)
    state._cpu_s45b_eq10_callable_seals = {
        name: seal_callable(function) for name, function in b_helpers.items()
    }
    state._cpu_s45b_stage_seal = seal_callable(instrumented)
    state._cpu_s45b_scope_helper = private_globals["__cpu_s45b_scope__"]
    state._cpu_s45b_marker_helper = private_globals["__cpu_s45b_marker__"]
    state._cpu_s45b_stage_helper_seals = {
        **{name: seal_callable(function) for name, function in b_helpers.items()},
        "__cpu_s45b_scope__": seal_callable(state._cpu_s45b_scope_helper),
        "__cpu_s45b_marker__": seal_callable(state._cpu_s45b_marker_helper),
    }
    state._cpu_s45b_lineage = {
        "r8_original_stage_code_identity": code_identity(r8_original_stage),
        "r8_original_stage_structural_code_identity": s45a_code_identity(r8_original_stage),
        "s45a_stage_code_identity": code_identity(original_stage),
        "s45a_stage_structural_code_identity": s45a_code_identity(original_stage),
        "s45a_ast_code_proof": bool(s45a_profile),
        "s45a_stage_sites": copy.deepcopy(s45a_sites),
        "b_stage_code_identity": code_identity(instrumented),
        "b_stage_structural_code_identity": s45a_code_identity(instrumented),
        "b_original_field_points_to_r8": True,
    }
    return {
        **copy.deepcopy(b_receipt),
        "s45a_stage_timing_sites": copy.deepcopy(s45a_sites),
        "actual_original_stage_code_identity": code_identity(r8_original_stage),
        "s45a_stage_code_identity": code_identity(original_stage),
        "s45a_ast_code_proof": bool(s45a_profile),
        "actual_instrumented_stage_code_identity": code_identity(instrumented),
        "helper_names": sorted(b_helpers),
        "backend_registry_labels": sorted(registry),
        "backend_registry_sealed": True,
        "lineage": copy.deepcopy(state._cpu_s45b_lineage),
        "helper_registry_fixed": True,
        "array_values_retained": False,
    }


def verify_bindings(dispatch: Any) -> dict[str, Any]:
    """Verify B-only callable, stage, and G2 helper identities."""

    pair = getattr(dispatch, "cpu_s4_5b_recorder_pair", None)
    if not isinstance(pair, RecorderPair):
        raise SourceBindingError("S45B_RECORDER_PAIR_MISSING")
    state = getattr(dispatch, "cpu_s4_5b_state", None)
    if state is None:
        raise SourceBindingError("S45B_STATE_MISSING")
    try:
        verify_callable_seal(state._cpu_s45b_backend_matcher_seal)
    except InstrumentationBindingError as exc:
        raise SourceBindingError("S45B_EQ10_FIXED_MATCHER_CHANGED") from exc
    if (
        getattr(pair.b, "_companion_recorder", None) is not pair.s45a
        or getattr(pair.b, "_companion_identity", None) != pair.identity_token
        or getattr(pair.b, "_companion_profile", None) is not pair.profile
    ):
        raise SourceBindingError("S45B_SCALAR_COMPANION_BINDING_CHANGED")
    importlib.import_module("tools.cpu_s4_5a_l1_residual.candidate").verify_bindings(state)
    for name, seal in getattr(state, "_cpu_s45b_eq10_callable_seals", {}).items():
        try:
            verify_callable_seal(seal)
        except InstrumentationBindingError as exc:
            raise SourceBindingError(f"S45B_EQ10_HELPER_CHANGED:{name}") from exc
    try:
        verify_fixed_backend_registry(
            getattr(state, "_cpu_s45b_eq10_backend_registry", {}),
            getattr(state, "_cpu_s45b_eq10_backend_seals", {}),
        )
    except InstrumentationBindingError as exc:
        raise SourceBindingError("S45B_EQ10_BACKEND_REGISTRY_CHANGED") from exc
    for name, seal in getattr(state, "_cpu_s45b_stage_helper_seals", {}).items():
        if getattr(state._internal_stage_inner, "__globals__", {}).get(name) is not seal.get("function"):
            raise SourceBindingError(f"S45B_STAGE_GLOBAL_SLOT_CHANGED:{name}")
        try:
            verify_callable_seal(seal)
        except InstrumentationBindingError as exc:
            raise SourceBindingError(f"S45B_STAGE_HELPER_CHANGED:{name}") from exc
    stage_seal = getattr(state, "_cpu_s45b_stage_seal", None)
    if stage_seal is not None:
        try:
            verify_callable_seal(stage_seal)
        except InstrumentationBindingError as exc:
            raise SourceBindingError("S45B_STAGE_CODE_OR_CLOSURE_CHANGED") from exc
    stage = getattr(state, "_internal_stage_inner", None)
    r8_original = getattr(state, "_cpu_s45a_original_stage", None)
    if not callable(stage) or not callable(r8_original) or getattr(
        stage, "__cpu_s45a_original__", None
    ) is not r8_original:
        raise SourceBindingError("S45B_STAGE_ORIGINAL_BINDING_CHANGED")
    lineage = getattr(state, "_cpu_s45b_lineage", {})
    if lineage.get("r8_original_stage_code_identity") != code_identity(r8_original):
        raise SourceBindingError("S45B_R8_LINEAGE_CHANGED")
    active_seal = getattr(state, "_cpu_s45b_active_propagation_seal", None)
    active_namespace = None if active_seal is None else getattr(active_seal.get("function"), "__globals__", None)
    if isinstance(active_namespace, dict):
        for name in ("__cpu_s45b_scope__", "__cpu_s45b_marker__"):
            helper = active_namespace.get(name)
            if not callable(helper) or getattr(helper, "__cpu_s45b_recorder__", None) is not pair.b:
                raise SourceBindingError(f"S45B_PROPAGATION_HELPER_CHANGED:{name}")
    else:
        raise SourceBindingError("S45B_ACTIVE_PROPAGATION_NAMESPACE_MISSING")
    for name, seal in getattr(state, "_cpu_s45b_propagation_helper_seals", {}).items():
        if active_namespace.get(name) is not seal.get("function"):
            raise SourceBindingError(f"S45B_PROPAGATION_GLOBAL_SLOT_CHANGED:{name}")
        try:
            verify_callable_seal(seal)
        except InstrumentationBindingError as exc:
            raise SourceBindingError(f"S45B_PROPAGATION_HELPER_SEAL_CHANGED:{name}") from exc
    active_seal = getattr(state, "_cpu_s45b_active_propagation_seal", None)
    if active_seal is not None:
        try:
            verify_callable_seal(active_seal)
        except InstrumentationBindingError as exc:
            raise SourceBindingError("S45B_ACTIVE_PROPAGATION_CODE_OR_CLOSURE_CHANGED") from exc
    binding = copy.deepcopy(getattr(dispatch, "cpu_s4_5b_binding", {}))
    return {
        "status": "PASS_S45B_PRIVATE_BINDINGS",
        "profile": bool(pair.profile),
        "domain": DOMAIN,
        "b_helper_count": len(getattr(state, "_cpu_s45b_eq10_helpers", {})),
        "propagation_helpers_bound": bool(
            set(getattr(state, "_cpu_s45b_propagation_helper_seals", {}))
            == {"__cpu_s45b_scope__", "__cpu_s45b_marker__"}
        ),
        "old_s45a_guards_untouched": True,
        "binding": binding,
    }


def build_role(
    mode: str,
    *,
    capture_root: str | Path,
    local_engineering: bool = False,
    pool_override: int | None = None,
    profile: bool,
    recorder_pair: RecorderPair | None = None,
    max_records: int = DEFAULT_MAX_RECORDS,
    role: str | None = None,
) -> tuple[Any, Any, Any, dict[str, Any], Any, Any]:
    """Build one S45B role with the unchanged G2 six-tuple contract."""

    mode_name = _normalise_mode(mode)
    if not isinstance(profile, bool):
        raise CandidateUnsupported("S45B_PROFILE_MUST_BE_BOOL")
    expected_role = ROLE_PROFILE if profile else ROLE_OFF
    if role is not None and str(role).upper() != expected_role:
        raise CandidateUnsupported("S45B_ROLE_PROFILE_SWITCH_MISMATCH")
    resources = _resolve_resources(
        local_engineering=bool(local_engineering), pool_override=pool_override
    )
    if recorder_pair is None:
        recorder_pair = make_recorder_pair(profile, max_records=max_records)
    if not isinstance(recorder_pair, RecorderPair) or recorder_pair.profile is not profile:
        raise CandidateUnsupported("S45B_RECORDER_PAIR_PROFILE_MISMATCH")
    if type(recorder_pair.b) is not TimingRecorder:
        raise CandidateUnsupported("S45B_RECORDER_EXACT_TYPE_REQUIRED")
    accept_s45a_recorder(recorder_pair.s45a, profile=profile)

    s45a = importlib.import_module("tools.cpu_s4_5a_l1_residual.candidate")
    import numba

    if int(numba.config.NUMBA_NUM_THREADS) != int(resources["numba_pool"]):
        raise CandidateUnsupported(
            f"S45B_ACTUAL_NUMBA_POOL_MISMATCH:{numba.config.NUMBA_NUM_THREADS}:{resources['numba_pool']}"
        )
    numba.set_num_threads(int(resources["outer_mask"]))
    settings = {
        "kernel_threads": 24 if not local_engineering else 1,
        "temporal_workers": int(resources["temporal_workers"]),
        "spatial_workers": int(resources["spatial_workers"]),
        "raman_workers": int(resources["raman_workers"]),
    }
    factory = s45a.make_factory(
        DEFAULT_VARIANT,
        local_engineering=bool(local_engineering),
        profile=bool(profile),
        recorder=recorder_pair.s45a,
        **settings,
    )
    propagate, dispatch = s45a._bind_factory(factory)
    state = getattr(dispatch, "cpu_s4_3g_state", None)
    if state is None:
        raise SourceBindingError("S45B_STATE_MISSING")

    eq10_binding = _install_eq10_stage(state, recorder_pair.b)
    s45a.install_propagation_wrappers(propagate, state)
    pre_check = s45a.verify_bindings(state)
    owned = _owned_source_identity()
    wrapped, actual_dispatch, clock, binding, audit, lease = attach_with_b(
        propagate,
        dispatch,
        role="G_CONTROLLED_INPUT",
        mode=mode_name,
        capture_root=Path(capture_root).resolve(),
        owned_identity=owned,
        recorder=recorder_pair.b,
    )
    if actual_dispatch is not dispatch:
        raise SourceBindingError("S45B_RUNTIME_DISPATCH_BINDING_CHANGED")
    active_clone = getattr(wrapped, "__g2_inner__", None)
    active_namespace = getattr(active_clone, "__globals__", None)
    if not isinstance(active_namespace, dict):
        raise SourceBindingError("S45B_ACTIVE_CLONE_NAMESPACE_MISSING")
    state._cpu_s45b_propagation_helper_seals = {
        name: seal_callable(active_namespace[name])
        for name in ("__cpu_s45b_scope__", "__cpu_s45b_marker__")
        if name in active_namespace
    }
    if set(state._cpu_s45b_propagation_helper_seals) != {
        "__cpu_s45b_scope__",
        "__cpu_s45b_marker__",
    }:
        raise SourceBindingError("S45B_ACTIVE_PROPAGATION_HELPERS_MISSING")
    state._cpu_s45b_active_propagation_seal = seal_callable(active_clone)
    binding["s45b_propagation_helper_bindings"] = {
        name: {
            "code_identity": seal["code_identity"],
            "code_object_id": int(id(seal["code"])),
            "globals_id": int(id(seal["globals"])),
            "closure_cell_count": len(seal.get("closure_cells", ())),
            "recorder_identity": int(id(recorder_pair.b)),
        }
        for name, seal in state._cpu_s45b_propagation_helper_seals.items()
    }
    # Existing S45A seals are installed after the actual clone is bound.  B's
    # extra helpers are checked separately by verify_bindings.
    s45a.bind_active_propagation(wrapped, state)
    post_check = s45a.verify_bindings(state)
    dispatch.cpu_s4_5b_profile = bool(profile)
    dispatch.cpu_s4_5b_recorder_pair = recorder_pair
    dispatch.cpu_s4_5b_state = state
    binding.update(
        {
            "schema": CANDIDATE_SCHEMA,
            "domain": DOMAIN,
            "role": expected_role,
            "mode": mode_name,
            "profile": bool(profile),
            "resource_contract": copy.deepcopy(resources),
            "cache_contract": copy.deepcopy(CACHE_CONTRACT),
            "candidate": DEFAULT_VARIANT,
            "s45a_pre_binding_check": copy.deepcopy(pre_check),
            "s45a_post_binding_check": copy.deepcopy(post_check),
            "b_eq10_binding": copy.deepcopy(eq10_binding),
            "b_recorder_schema": SCHEMA,
            "b_max_records": int(recorder_pair.b.max_records),
            "b_metadata_max_bytes": 64 * 1024 * 1024,
            "scientific_qualification": False,
            "new_math_algorithm": False,
        }
    )
    dispatch.cpu_s4_5b_binding = copy.deepcopy(binding)
    dispatch.cpu_s4_5b_instrumentation_bindings = copy.deepcopy(
        getattr(state, "_cpu_s45a_bindings", {})
    )
    b_check = verify_bindings(dispatch)
    dispatch.cpu_s4_5b_binding["b_post_binding_check"] = copy.deepcopy(b_check)
    return wrapped, dispatch, clock, dispatch.cpu_s4_5b_binding, audit, lease


__all__ = [
    "CACHE_CONTRACT",
    "CANDIDATE_SCHEMA",
    "CandidateUnsupported",
    "DEFAULT_VARIANT",
    "DOMAIN",
    "HPC_POOL_CONTRACT",
    "LOCAL_POOL_CONTRACT",
    "MODES",
    "ROLE_OFF",
    "ROLE_PROFILE",
    "ROLES",
    "SourceBindingError",
    "build_role",
    "make_recorder",
    "source_identity",
    "verify_bindings",
]
