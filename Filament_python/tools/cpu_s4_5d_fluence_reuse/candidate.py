"""CPU-S4-5D source-bound saved-fluence reuse candidate.

``L1_REF_5D`` keeps the accepted 4A/4B L1 factory and original diagnostics.
``D_FLUENCE_REUSE_V1`` changes only the private saved-branch seam: the
already-computed F2D is passed to a private copy of the original 3-D radius
post-processing.  The public diagnostics module and all production kernels
remain unchanged.
"""

from __future__ import annotations

import copy
import importlib
import os
import sys
from pathlib import Path
from typing import Any

from tools.hpc_ops import provenance_v2 as pv

from .adapter import (
    ADAPTER_SCHEMA,
    AdapterBindingError,
    BodyClock,
    FluenceCounter,
    attach,
    require,
    verify_helper_slots,
)
from .fluence import postprocess_source_proof


CANDIDATE_SCHEMA = "filament.cpu_s4_5d_fluence_reuse.candidate.v1"
DEFAULT_VARIANT = "SAVED_FLUENCE_TRAPEZOID_REUSE_SAME_ORDER"
DOMAIN = "S45D_SAVED_FLUENCE_REUSE"
ROLE_REFERENCE = "L1_REF_5D"
ROLE_REUSE = "D_FLUENCE_REUSE_V1"
ROLES = (ROLE_REFERENCE, ROLE_REUSE)
MODES = ("QUALIFY_BOUNDARY", "PURE_OFF_SMALLGRID")
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
EXPECTED_FROZEN_DEPENDENCIES = {
    "Filament_python/KHz_filament/diagnostics.py": {
        "git_blob_oid": "bc6165ec951092ff09f02c6d01bbf7d79fc2c11a",
        "canonical_lf_sha256": "8714cdecbe481cb8fb06430b0634150a3548bbc6c1f8ff4b5c92d9c282c66970",
    },
    "Filament_python/KHz_filament/propagate.py": {
        "git_blob_oid": "f156111398415c2b92e4609ccc1da075fcf5d1e3",
        "canonical_lf_sha256": "909fbcf5ef05d2c50b8ed27506b450d40127b3973709d9b4b6a961799dc8d1f6",
    },
}


class CandidateUnsupported(RuntimeError):
    """The bounded S4-5D role cannot serve the requested route."""


class SourceBindingError(RuntimeError):
    """A frozen source, factory, or private clone identity is invalid."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalise_mode(mode: str) -> str:
    value = str(mode).upper()
    if value in {"OFF", "PURE_OFF", "PURE_OFF_REAL8", "PURE_OFF_FULLSIZE"}:
        return "PURE_OFF_SMALLGRID"
    if value not in MODES:
        raise CandidateUnsupported(f"S45D_MODE_UNSUPPORTED:{mode}")
    return value


def _normalise_role(role: str) -> str:
    value = str(role).upper()
    if value not in ROLES:
        raise CandidateUnsupported(f"S45D_ROLE_UNSUPPORTED:{role}")
    return value


def _owned_source_identity() -> dict[str, Any]:
    """Hash every committed ``.py``/``.md`` file in this namespace.

    Root-owned files are discovered dynamically, so a later committed entry
    or local-chain file is included without changing this allowlist logic.
    """

    root = Path(__file__).resolve().parent
    repo = _repo_root()
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".py", ".md"}:
            continue
        path = path.resolve(strict=True)
        relative = path.relative_to(repo).as_posix()
        raw = path.read_bytes()
        try:
            oid = pv._git_blob_oid(repo, relative)
            blob = pv._read_git_blob(repo, oid)
        except Exception as exc:  # provenance helper exposes implementation-specific errors
            raise SourceBindingError(f"S45D_OWNED_SOURCE_NOT_COMMITTED:{relative}") from exc
        canonical = pv.canonical_lf_sha256(blob)
        require(
            pv.canonical_lf_bytes(raw) == pv.canonical_lf_bytes(blob),
            f"S45D_OWNED_SOURCE_CANONICAL_MISMATCH:{relative}",
        )
        files[path.name] = {
            "path": str(path),
            "relative_path": relative,
            "size_bytes": int(len(raw)),
            "raw_sha256": pv.raw_sha256(raw),
            "classification": "tracked_text",
            "hash_scope": pv.TRACKED_HASH_SCOPE,
            "git_blob_oid": oid,
            "canonical_lf_sha256": canonical,
            "raw_byte_observation": "Checkout only; canonical identity is authoritative across platforms",
        }
    required = {"__init__.py", "adapter.py", "candidate.py", "fluence.py", "test_fluence.py", "README.md"}
    require(required.issubset(files), "S45D_OWNED_NAMESPACE_INCOMPLETE")
    composite = (
        f"{CANDIDATE_SCHEMA}\n"
        + "".join(
            f"{name}:{files[name]['canonical_lf_sha256']}\n" for name in sorted(files)
        )
    ).encode("ascii")
    return {
        "schema": CANDIDATE_SCHEMA + ".owned_sources.v1",
        "provenance_implementation": "tools.hpc_ops.provenance_v2",
        "producer_module_identity": "tools.cpu_s4_5d_fluence_reuse",
        "files": files,
        "composite_canonical_lf_sha256": pv.raw_sha256(composite),
        "allowlist": "ALL_NAMESPACE_PY_MD",
    }


def _frozen_source_identity() -> dict[str, Any]:
    repo = _repo_root()
    records: dict[str, dict[str, Any]] = {}
    for relative, expected in EXPECTED_FROZEN_DEPENDENCIES.items():
        path = (repo / relative).resolve(strict=True)
        oid = pv._git_blob_oid(repo, relative)
        canonical = pv.canonical_lf_sha256(path.read_bytes())
        require(oid == expected["git_blob_oid"], f"S45D_FROZEN_BLOB_MISMATCH:{relative}")
        require(
            canonical == expected["canonical_lf_sha256"],
            f"S45D_FROZEN_CANONICAL_MISMATCH:{relative}",
        )
        records[relative] = {
            "path": str(path),
            "relative_path": relative,
            "classification": "tracked_text",
            "hash_scope": pv.TRACKED_HASH_SCOPE,
            "git_blob_oid": oid,
            "canonical_lf_sha256": canonical,
        }
    return {
        "schema": CANDIDATE_SCHEMA + ".frozen_dependencies.v1",
        "files": records,
        "source_contract": "BOUND_PROPAGATE_AND_DIAGNOSTICS_BLOBS",
    }


def _resolve_resources(*, local_engineering: bool, pool_override: int | None) -> dict[str, Any]:
    if local_engineering:
        if os.environ.get("SLURM_JOB_ID"):
            raise CandidateUnsupported("LOCAL_ENGINEERING_ONLY_FORBIDDEN_ON_SLURM")
        if pool_override != 1:
            raise CandidateUnsupported("S45D_LOCAL_ENGINEERING_REQUIRES_POOL_OVERRIDE_1")
        return copy.deepcopy(LOCAL_POOL_CONTRACT)
    if pool_override is not None:
        raise CandidateUnsupported("S45D_POOL_OVERRIDE_REQUIRES_LOCAL_ENGINEERING")
    return copy.deepcopy(HPC_POOL_CONTRACT)


def _bind_l1_factory(*, local_engineering: bool, fullsize: bool):
    """Bind one accepted 4A/4B factory without touching production modules."""

    l1 = importlib.import_module("tools.cpu_s4_4a_linear_cache.candidate")
    if fullsize:
        if local_engineering:
            raise CandidateUnsupported("S45D_FULLSIZE_LOCAL_ENGINEERING_FORBIDDEN")
        full = importlib.import_module("tools.cpu_s4_4b_fullsize_linear_cache.candidate")
        factory = full.make_factory(local_engineering=False)
    else:
        resources = LOCAL_POOL_CONTRACT if local_engineering else HPC_POOL_CONTRACT
        factory = l1.make_factory(
            kernel_threads=1 if local_engineering else 24,
            temporal_workers=int(resources["temporal_workers"]),
            spatial_workers=1 if local_engineering else 24,
            raman_workers=int(resources["raman_workers"]),
            local_engineering=bool(local_engineering),
        )
    propagator, dispatch = l1._bind_factory(factory)
    require(callable(propagator) and dispatch is not None, "S45D_L1_FACTORY_BINDING_INCOMPLETE")
    if fullsize:
        # 4B retains the same accepted cache contract under the 4A dispatch
        # namespace; exposing the 4A view keeps the existing SG runner API.
        if not hasattr(dispatch, "cpu_s4_4b_cache_snapshot"):
            raise SourceBindingError("S45D_FULLSIZE_CACHE_SNAPSHOT_MISSING")
        dispatch.cpu_s4_4a_cache_snapshot = dispatch.cpu_s4_4b_cache_snapshot
    return propagator, dispatch


def build_role(
    mode: str,
    *,
    role: str,
    capture_root: str | Path,
    local_engineering: bool = False,
    pool_override: int | None = None,
    fullsize: bool = False,
) -> tuple[Any, Any, Any, dict[str, Any], Any, Any]:
    """Build the source-bound L1 reference or D reuse six-tuple."""
    require(not any(name.startswith('tools.cpu_s4_5c_eq10_pointwise') for name in sys.modules),'S45D_C_PACKAGE_FORBIDDEN')

    mode_name = _normalise_mode(mode)
    role_name = _normalise_role(role)
    resources = _resolve_resources(
        local_engineering=bool(local_engineering), pool_override=pool_override
    )
    source_dependencies = _frozen_source_identity()
    import numba

    actual_pool = int(numba.config.NUMBA_NUM_THREADS)
    require(
        actual_pool == int(resources["numba_pool"]),
        f"S45D_ACTUAL_NUMBA_POOL_MISMATCH:{actual_pool}:{resources['numba_pool']}",
    )
    numba.set_num_threads(int(resources["outer_mask"]))
    propagator, dispatch = _bind_l1_factory(
        local_engineering=bool(local_engineering), fullsize=bool(fullsize)
    )
    body_clock = BodyClock()
    candidate_applied = role_name == ROLE_REUSE
    counter = FluenceCounter(role=role_name, candidate_applied=candidate_applied)
    owned = _owned_source_identity()
    wrapped, actual_dispatch, clock, binding, audit, lease = attach(
        propagator,
        dispatch,
        role=role_name,
        mode=mode_name,
        capture_root=Path(capture_root).resolve(),
        owned_identity=owned,
        body_clock=body_clock,
        counter=counter,
        candidate_applied=candidate_applied,
        source_dependencies=source_dependencies,
    )
    require(actual_dispatch is dispatch, "S45D_RUNTIME_DISPATCH_BINDING_CHANGED")
    dispatch.cpu_s45d_role = role_name
    dispatch.cpu_s45d_candidate = DEFAULT_VARIANT
    dispatch.cpu_s45d_body_clock = body_clock
    dispatch.cpu_s45d_counter = counter
    dispatch.cpu_s45d_fluence_snapshot = counter.snapshot
    dispatch.cpu_s45d_owned_source_identity = copy.deepcopy(owned)
    dispatch.cpu_s45d_source_dependencies = copy.deepcopy(source_dependencies)
    dispatch.cpu_s45d_binding = copy.deepcopy(
        {
            **binding,
            "schema": CANDIDATE_SCHEMA,
            "adapter_schema": ADAPTER_SCHEMA,
            "domain": DOMAIN,
            "role": role_name,
            "mode": mode_name,
            "candidate": DEFAULT_VARIANT,
            "fullsize": bool(fullsize),
            "resource_contract": copy.deepcopy(resources),
            "actual_pool": int(actual_pool),
            "resource_scope": resources["scope"],
            "scientific_qualification": False,
            "new_math_algorithm": False,
            "original_public_diagnostics_unchanged": True,
            "source_dependencies": copy.deepcopy(source_dependencies),
            "owned_source_identity": copy.deepcopy(owned),
            "postprocess_ast_proof": copy.deepcopy(
                binding.get("s45d_static_receipt", {}).get("postprocess_ast_proof", {})
            ),
            "counter_contract": counter.snapshot(),
            "body_clock_contract": body_clock.snapshot(),
            "old_g2_guards_untouched": True,
            "unknown_observers_added": False,
        }
    )
    verify_bindings(dispatch)
    return wrapped, actual_dispatch, clock, dispatch.cpu_s45d_binding, audit, lease


def verify_bindings(dispatch: Any) -> dict[str, Any]:
    """Verify source identity, helper slots, static receipt, and scalar API."""
    require(not any(name.startswith('tools.cpu_s4_5c_eq10_pointwise') for name in sys.modules),'S45D_C_PACKAGE_FORBIDDEN')

    role = getattr(dispatch, "cpu_s45d_role", None)
    require(role in ROLES, "S45D_ROLE_BINDING_MISSING")
    counter = getattr(dispatch, "cpu_s45d_counter", None)
    require(isinstance(counter, FluenceCounter), "S45D_COUNTER_BINDING_MISSING")
    snapshot = getattr(dispatch, "cpu_s45d_fluence_snapshot", None)
    require(callable(snapshot), "S45D_COUNTER_SNAPSHOT_MISSING")
    require(getattr(snapshot, "__self__", None) is counter, "S45D_COUNTER_SNAPSHOT_OWNER_CHANGED")
    body_clock = getattr(dispatch, "cpu_s45d_body_clock", None)
    require(isinstance(body_clock, BodyClock), "S45D_BODY_CLOCK_BINDING_MISSING")
    binding = getattr(dispatch, "cpu_s45d_binding", None)
    require(isinstance(binding, dict), "S45D_BINDING_MISSING")
    static = binding.get("s45d_static_receipt")
    require(isinstance(static, dict), "S45D_STATIC_RECEIPT_MISSING")
    require(static.get("inverse_ast_exact") is True, "S45D_STATIC_INVERSE_PROOF_MISSING")
    require(static.get("old_guard_capture") == "AFTER_COMPILE_INSERTION", "S45D_G2_GUARD_SEAM_CHANGED")
    require(binding.get("old_g2_guards_untouched") is True, "S45D_OLD_G2_GUARD_FLAG_MISSING")
    require(binding.get("unknown_observers_added") is False, "S45D_UNKNOWN_OBSERVER_FLAG")
    require(binding.get("original_public_diagnostics_unchanged") is True, "S45D_PUBLIC_DIAGNOSTICS_FLAG")
    active_clone = getattr(dispatch, "cpu_s45d_active_clone", None)
    helper_bindings = getattr(dispatch, "cpu_s45d_helper_bindings", None)
    require(callable(active_clone), "S45D_ACTIVE_CLONE_BINDING_MISSING")
    require(isinstance(helper_bindings, dict), "S45D_HELPER_BINDINGS_MISSING")
    verify_helper_slots(active_clone, helper_bindings)
    post_ast = static.get("postprocess_ast_proof")
    require(isinstance(post_ast, dict), "S45D_POST_AST_PROOF_MISSING")
    require(post_ast.get("post_ast_equal") is True, "S45D_POST_AST_PROOF_FALSE")
    require(
        post_ast.get("original_post_ast_sha256") == post_ast.get("private_post_ast_sha256"),
        "S45D_POST_AST_HASH_MISMATCH",
    )
    expected_applied = role == ROLE_REUSE
    require(
        bool(static.get("fluence_reuse", {}).get("candidate_applied")) is expected_applied,
        "S45D_STATIC_ROLE_REUSE_MISMATCH",
    )
    return {
        "status": "PASS_S45D_PRIVATE_BINDINGS",
        "role": role,
        "domain": DOMAIN,
        "candidate": DEFAULT_VARIANT,
        "candidate_applied": expected_applied,
        "body_clock_bound": True,
        "fluence_counter_bound": True,
        "static_receipt": copy.deepcopy(static),
        "old_g2_guards_untouched": True,
        "public_diagnostics_unchanged": True,
        "arrays_retained": False,
    }


def source_identity() -> dict[str, Any]:
    post_ast = postprocess_source_proof()
    return {
        "schema": CANDIDATE_SCHEMA,
        "candidate_name": DEFAULT_VARIANT,
        "domain": DOMAIN,
        "roles": list(ROLES),
        "owned_source_identity": _owned_source_identity(),
        "frozen_source_identity": _frozen_source_identity(),
        "postprocess_ast_proof": post_ast,
        "new_math_algorithm": False,
        "public_diagnostics_unchanged": True,
        "private_helper": "fluence.second_moment_radius_from_f2d",
    }


__all__ = [
    "CANDIDATE_SCHEMA",
    "CandidateUnsupported",
    "DEFAULT_VARIANT",
    "DOMAIN",
    "EXPECTED_FROZEN_DEPENDENCIES",
    "HPC_POOL_CONTRACT",
    "LOCAL_POOL_CONTRACT",
    "MODES",
    "ROLE_REFERENCE",
    "ROLE_REUSE",
    "ROLES",
    "SourceBindingError",
    "build_role",
    "source_identity",
    "verify_bindings",
]
