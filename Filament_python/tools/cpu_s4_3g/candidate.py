"""CPU-S4-3G controlled call-site Raman candidate.

G is a private composition of the frozen F/E/D/R8 assets.  It changes one
handoff boundary only: a field returned by the real BK-NEE producer may be
borrowed synchronously by both rotational Raman half-steps.  The public/raw
Raman entry remains the strict E path and therefore never accepts an external
trust or ownership control.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from tools.hpc_ops import provenance_v2 as _provenance_v2

from .controlled_callsite import (
    CallsiteBindingError,
    PROPAGATE_CANONICAL_LF_SHA256,
    PROPAGATE_RELATIVE_PATH,
    PROPAGATE_SOURCE_BLOB,
    bind_propagator,
    make_controlled_state_class,
    source_binding,
)


F_RAW_SHA256 = "2100fda7adfb2581151eaad46eb3e992133090c54876718759f1ea99df7de6eb"
F_SIZE_BYTES = 8265
E_RAW_SHA256 = "a268bcfb74fbde603b0b3dbae26028e73a59a4022e3140bba9bae39a20e01e34"
E_SIZE_BYTES = 49397
D_RAW_SHA256 = "a42c02ec76e0d3f5e0b6d809d2696dd58d27ea6ce3fb95e784080b2b985f58ee"
D_SIZE_BYTES = 19012
R8_RAW_SHA256 = "252fc1e6521812c29872d5b848e4082841f0b8ae0ad15661394f6b31347e0a32"
R8_SIZE_BYTES = 59808

CANDIDATE_SCHEMA = "filament.cpu_s4_3g.controlled_callsite_borrow.v1"
TRANSFORMATION_IDENTITY = "cpu_s4_3g_private_f_sha_composed_controlled_callsite_v1"
DEFAULT_VARIANT = "G_CONTROLLED_INPUT"
_COMPATIBILITY_ALIASES = frozenset({"C_INTERNAL", "C_STRICT"})
_BODY_NAMES = ("_product_body", "_derivative_body", "_rhs_body", "_predictor_body", "_corrector_body")
_OWNED_SOURCE_FILES = ("__init__.py", "candidate.py", "controlled_callsite.py")

_MODULE_CACHE: dict[tuple[str, str, str, str], ModuleType] = {}


class SourceBindingError(RuntimeError):
    """A frozen G source or loader identity is not the registered asset."""


class CandidateUnsupported(RuntimeError):
    """The requested propagation is outside G's narrow controlled scope."""


class HandoffRejected(RuntimeError):
    """A private capability/lifetime/routing contract was rejected."""


class StageBindingError(ValueError):
    """Compatibility name for E/R8 stage binding failures."""


def _reject_optimized() -> None:
    if int(getattr(sys.flags, "optimize", 0)) != 0 or os.environ.get("PYTHONOPTIMIZE"):
        raise HandoffRejected("PYTHONOPTIMIZE_OR_O_FLAG_FORBIDDEN")


def _owned_source_identity() -> tuple[dict[str, dict[str, Any]], str]:
    """Return exact raw identities and a stable composite for the G package.

    The entry file's raw hash remains a separate field in the public source
    identity.  Private module/cache names use the composite so a repair to
    ``controlled_callsite.py`` or ``__init__.py`` cannot reuse an older G
    namespace compiled from a different owned source set.
    """
    package = Path(__file__).resolve().parent
    records: dict[str, dict[str, Any]] = {}
    composite_parts: list[str] = ["filament.cpu_s4_3g.owned_sources.v1"]
    for relative_name in _OWNED_SOURCE_FILES:
        path = (package / relative_name).resolve(strict=True)
        raw_sha256 = _provenance_v2.raw_sha256_file(path)
        size_bytes = int(path.stat().st_size)
        records[relative_name] = {
            "path": str(path),
            "classification": "external",
            "hash_scope": "raw_bytes",
            "raw_sha256": raw_sha256,
            "size_bytes": size_bytes,
        }
        composite_parts.append(f"{relative_name}={raw_sha256}:{size_bytes}")
    composite_sha256 = _provenance_v2.raw_sha256(
        "\n".join(composite_parts).encode("ascii")
    )
    return records, composite_sha256


def _implementation_sha256() -> str:
    """Return the owned-source composite used for private G namespaces."""
    return _owned_source_identity()[1]


def _default_paths() -> tuple[Path, Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return (
        root / "cpu_s4_3f" / "candidate.py",
        root / "cpu_s4_3e" / "candidate.py",
        root / "cpu_s4_3d" / "variants.py",
        root / "cpu_s4_3c" / "candidate.py",
    )


def _record(path: Path, expected: str, size: int, label: str) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    actual = _provenance_v2.raw_sha256_file(path)
    actual_size = int(path.stat().st_size)
    if actual != expected or actual_size != int(size):
        raise SourceBindingError(
            f"{label} source identity conflict: {path}; {actual}; {actual_size}"
        )
    return {
        "path": str(path),
        "classification": "external",
        "hash_scope": "raw_bytes",
        "raw_sha256": actual,
        "size_bytes": actual_size,
    }


def _own_record() -> dict[str, Any]:
    path = Path(__file__).resolve()
    entry_raw_sha256 = _provenance_v2.raw_sha256_file(path)
    record = _record(path, entry_raw_sha256, int(path.stat().st_size), "G entry")
    owned_sources, composite_sha256 = _owned_source_identity()
    record.update(
        {
            "entry_raw_sha256": entry_raw_sha256,
            "composite_sha256": composite_sha256,
            "implementation_sha256": composite_sha256,
            "owned_source_order": list(_OWNED_SOURCE_FILES),
            "owned_sources": owned_sources,
        }
    )
    return record


def _load_module(path: Path, role: str, implementation_sha: str) -> ModuleType:
    path = Path(path).resolve(strict=True)
    digest = _provenance_v2.raw_sha256_file(path)
    key = (str(path), str(role), digest, str(implementation_sha))
    cached = _MODULE_CACHE.get(key)
    if cached is not None:
        return cached
    token = _provenance_v2.raw_sha256(
        f"G|{implementation_sha}|{role}|{digest}".encode("ascii")
    )[:20]
    name = f"_filament_cpu_s4g_{str(role).lower()}_{token}"
    existing = sys.modules.get(name)
    if isinstance(existing, ModuleType):
        _MODULE_CACHE[key] = existing
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SourceBindingError(f"private import spec unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(
        {
            "__cpu_s4_3g_private__": True,
            "__cpu_s4_3g_role__": str(role),
            "__cpu_s4_3g_implementation_sha256__": implementation_sha,
            "__cpu_s4_3g_owned_source_composite__": implementation_sha,
            "__cpu_s4_3g_source_sha256__": digest,
        }
    )
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    _MODULE_CACHE[key] = module
    return module


def _verify_installed_g_r8(r8_module: ModuleType, implementation_sha: str) -> None:
    """Verify the complete G D-body installation without mutating R8.

    D's installer resets the R8 bundle, so an already compiled module must be
    reused only when all G provenance and frozen configuration markers are
    present and the five body functions still resolve in the R8 namespace.
    """
    if str(getattr(r8_module, "__cpu_s4d_variant_source_raw_sha256__", "")) != implementation_sha:
        raise SourceBindingError("G private R8 namespace is not G-SHA-derived")
    if str(getattr(r8_module, "__cpu_s4_3g_implementation_sha256__", "")) != implementation_sha:
        raise SourceBindingError("G R8 installation implementation identity mismatch")
    if str(getattr(r8_module, "__cpu_s4_3g_variant__", "")) != DEFAULT_VARIANT:
        raise SourceBindingError("G R8 installation variant marker missing")
    if getattr(r8_module, "__cpu_s4_3g_flat_bodies_installed__", None) is not True:
        raise SourceBindingError("G R8 D-body installation marker missing")
    if tuple(getattr(r8_module, "__cpu_s4_3g_body_names__", ())) != tuple(_BODY_NAMES):
        raise SourceBindingError("G R8 D-body name set mismatch")
    if int(getattr(r8_module, "__cpu_s4_3g_block_size__", -1)) != 32:
        raise SourceBindingError("G R8 D-body block-size marker mismatch")
    if int(getattr(r8_module, "__cpu_s4_3g_raman_workers__", -1)) != 48:
        raise SourceBindingError("G R8 D-body worker marker mismatch")
    if getattr(r8_module, "__cpu_s4_3g_fastmath__", None) is not False:
        raise SourceBindingError("G R8 D-body fastmath marker mismatch")
    for body_name in _BODY_NAMES:
        body = getattr(r8_module, body_name, None)
        if not callable(body) or getattr(body, "__globals__", None) is not r8_module.__dict__:
            raise SourceBindingError(f"G D body binding mismatch: {body_name}")


def _load_private_r8(
    r8_path: Path,
    d_path: Path,
    implementation_sha: str,
) -> tuple[ModuleType, ModuleType]:
    d_module = _load_module(d_path, "D", implementation_sha)
    if str(getattr(d_module, "__cpu_s4_3g_implementation_sha256__", "")) != implementation_sha:
        raise SourceBindingError("G private D namespace is not G-SHA-derived")
    if not callable(getattr(d_module, "_load_r8", None)) or not callable(
        getattr(d_module, "_install_flat_bodies", None)
    ):
        raise SourceBindingError("D private loader/install symbols unavailable")
    r8_module = d_module._load_r8(
        Path(r8_path).resolve(),
        "G_CONTROLLED_INPUT",
        implementation_sha,
        R8_RAW_SHA256,
    )
    if not bool(getattr(r8_module, "__cpu_s4d_identity_token__", "")):
        raise SourceBindingError("G private R8 namespace token missing")
    if str(getattr(r8_module, "__cpu_s4d_variant_source_raw_sha256__", "")) != implementation_sha:
        raise SourceBindingError("G private R8 namespace is not G-SHA-derived")
    if str(getattr(r8_module, "__cpu_s4_3d_r8_raw_sha256__", "")) != R8_RAW_SHA256:
        raise SourceBindingError("G private R8 raw identity marker mismatch")
    installed = getattr(r8_module, "__cpu_s4_3g_flat_bodies_installed__", None) is True
    if installed:
        # The first G factory may already have compiled the bundle.  Reusing
        # it is safe only after a non-mutating verification of every marker.
        _verify_installed_g_r8(r8_module, implementation_sha)
        return r8_module, d_module
    if getattr(r8_module, "_NUMBA_BUNDLE", None) is not None:
        raise SourceBindingError("R8 compiled bundle lacks G D-body installation/provenance")
    # D's exact five bodies are installed only in this G-derived R8 module.
    d_module._install_flat_bodies(r8_module)
    r8_module.__dict__.update(
        {
            "__cpu_s4_3g_variant__": DEFAULT_VARIANT,
            "__cpu_s4_3g_flat_bodies_installed__": True,
            "__cpu_s4_3g_body_names__": tuple(_BODY_NAMES),
            "__cpu_s4_3g_block_size__": 32,
            "__cpu_s4_3g_raman_workers__": 48,
            "__cpu_s4_3g_fastmath__": False,
            "__cpu_s4_3g_implementation_sha256__": implementation_sha,
        }
    )
    _verify_installed_g_r8(r8_module, implementation_sha)
    return r8_module, d_module


def _load_private_e(
    e_path: Path,
    r8_path: Path,
    r8_module: ModuleType,
    implementation_sha: str,
) -> ModuleType:
    e_module = _load_module(e_path, "E", implementation_sha)

    if str(getattr(e_module, "__cpu_s4_3g_implementation_sha256__", "")) != implementation_sha:
        raise SourceBindingError("G private E namespace is not G-SHA-derived")
    expected_r8_path = Path(r8_path).resolve()
    existing_state_class = e_module.__dict__.get("__cpu_s4_3g_state_class__")
    if existing_state_class is None:
        # A same-process caller may have created the G state before this
        # idempotence marker was introduced.  Recognize that exact private
        # class and stamp the marker instead of subclassing it again.
        prior_state_class = e_module.__dict__.get("_HandoffState")
        if (
            isinstance(prior_state_class, type)
            and prior_state_class.__name__ == "_ControlledHandoffState"
            and prior_state_class.__module__ == make_controlled_state_class.__module__
        ):
            existing_state_class = prior_state_class
            e_module.__dict__["__cpu_s4_3g_state_class__"] = existing_state_class
    if existing_state_class is not None:
        if not isinstance(existing_state_class, type) or e_module.__dict__.get("_HandoffState") is not existing_state_class:
            raise SourceBindingError("G private E state-class provenance mismatch")
        if e_module.__dict__.get("__cpu_s4_3g_r8_module_object__") is not r8_module:
            raise SourceBindingError("G private E namespace already bound to a different R8")
        if Path(str(e_module.__dict__.get("__cpu_s4_3g_r8_path__", ""))).resolve() != expected_r8_path:
            raise SourceBindingError("G private E namespace already bound to a different R8 path")
        return e_module

    def private_r8_loader(requested: Path) -> ModuleType:
        checked = _record(Path(requested), R8_RAW_SHA256, R8_SIZE_BYTES, "R8")
        if Path(checked["path"]).resolve() != expected_r8_path:
            raise SourceBindingError("G R8 request differs from factory binding")
        return r8_module

    # These assignments alter only the E module loaded under the G-derived
    # private name.  The frozen cpu_s4_3e and cpu_s4_3f modules are untouched.
    e_module.__dict__["_load_r8"] = private_r8_loader
    e_module.__dict__["_default_r8_path"] = lambda: expected_r8_path
    state_class = make_controlled_state_class(
        e_module,
        candidate_schema=CANDIDATE_SCHEMA,
        candidate_unsupported=CandidateUnsupported,
        handoff_rejected=HandoffRejected,
    )
    e_module.__dict__["_HandoffState"] = state_class
    e_module.__dict__.update(
        {
            "__cpu_s4_3g_state_class__": state_class,
            "__cpu_s4_3g_r8_module_object__": r8_module,
            "__cpu_s4_3g_r8_path__": str(expected_r8_path),
        }
    )
    return e_module


def _make_source_identity(
    *,
    own: dict[str, Any],
    f: dict[str, Any],
    e: dict[str, Any],
    d: dict[str, Any],
    r8: dict[str, Any],
    e_module: ModuleType,
    propagate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": CANDIDATE_SCHEMA,
        "candidate_name": DEFAULT_VARIANT,
        "transformation": TRANSFORMATION_IDENTITY,
        "implementation": own,
        "frozen_F": f,
        "frozen_E": e,
        "frozen_D": d,
        "frozen_R8": r8,
        "private_E_module": e_module.__name__,
        "private_R8_module": str(getattr(e_module, "__cpu_s4_3g_r8_module__", "")),
        "propagate": propagate,
        "controlled_scope": {
            "linear_model": "bk_nee",
            "operator_mode": "full_isaacs_eq27",
            "method": "iir",
            "iir_sampling": "exact_piecewise_linear",
            "integrator": "heun",
            "split_order": "strang",
            "dtype": "complex128",
            "layout": "native_C",
            "block_formula": "Nt*32",
            "raman_workers": 48,
            "fastmath": False,
            "monitoring_policy": "OFF_ONLY_FAIL_CLOSED",
            "j1_q1_monitored_routes": "UNSUPPORTED_UNQUALIFIED",
        },
    }


def source_identity(
    frozen_r8_path: str | Path | None = None,
    *,
    frozen_f_path: str | Path | None = None,
    frozen_e_path: str | Path | None = None,
    frozen_d_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and return all frozen source identities without building state."""
    _reject_optimized()
    default_f, default_e, default_d, default_r8 = _default_paths()
    f_path = Path(frozen_f_path) if frozen_f_path is not None else default_f
    e_path = Path(frozen_e_path) if frozen_e_path is not None else default_e
    d_path = Path(frozen_d_path) if frozen_d_path is not None else default_d
    r8_path = Path(frozen_r8_path) if frozen_r8_path is not None else default_r8
    own = _own_record()
    f = _record(f_path, F_RAW_SHA256, F_SIZE_BYTES, "F")
    e = _record(e_path, E_RAW_SHA256, E_SIZE_BYTES, "E")
    d = _record(d_path, D_RAW_SHA256, D_SIZE_BYTES, "D")
    r8 = _record(r8_path, R8_RAW_SHA256, R8_SIZE_BYTES, "R8")
    # Static source binding is useful even before the private modules are
    # loaded.  It proves that the future AST hook points at the frozen file.
    propagate_path = Path(__file__).resolve().parents[2] / "KHz_filament" / "propagate.py"
    propagate = source_binding(propagate_path)
    return _make_source_identity(
        own=own,
        f=f,
        e=e,
        d=d,
        r8=r8,
        e_module=ModuleType("<not-loaded>"),
        propagate=propagate,
    )


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
):
    """Return the single G factory and its source-bound propagation clone.

    ``C_INTERNAL``/``C_STRICT`` are accepted only as frozen-test aliases.  A
    raw/public call through either alias still uses the strict F fallback;
    only the producer-bound propagation clone can execute ``G_ACTUAL``.
    """
    _reject_optimized()
    normalized = str(variant).upper()
    if normalized != DEFAULT_VARIANT and normalized not in _COMPATIBILITY_ALIASES:
        raise ValueError(f"unknown G variant {variant!r}; choose {DEFAULT_VARIANT}")
    if observe is not None:
        raise CandidateUnsupported("observer_or_trace_not_supported_for_G")
    if int(block_size) != 32:
        raise ValueError("G freezes D contiguous task size at Nt*32")
    if int(raman_workers) != 48:
        raise ValueError("G freezes Raman worker count at W48")
    if int(kernel_threads) != 24 or int(temporal_workers) != 48 or int(spatial_workers) != 24:
        raise ValueError("G freezes kernel_threads=24, temporal_workers=48, spatial_workers=24")

    default_f, default_e, default_d, default_r8 = _default_paths()
    f_path = Path(frozen_f_path) if frozen_f_path is not None else default_f
    e_path = Path(frozen_e_path) if frozen_e_path is not None else default_e
    d_path = Path(frozen_d_path) if frozen_d_path is not None else default_d
    r8_path = Path(frozen_r8_path) if frozen_r8_path is not None else default_r8
    own = _own_record()
    f_identity = _record(f_path, F_RAW_SHA256, F_SIZE_BYTES, "F")
    e_identity = _record(e_path, E_RAW_SHA256, E_SIZE_BYTES, "E")
    d_identity = _record(d_path, D_RAW_SHA256, D_SIZE_BYTES, "D")
    r8_identity = _record(r8_path, R8_RAW_SHA256, R8_SIZE_BYTES, "R8")
    implementation_sha = str(own["implementation_sha256"])
    r8_module, d_module = _load_private_r8(r8_path, d_path, implementation_sha)
    e_module = _load_private_e(e_path, r8_path, r8_module, implementation_sha)

    # E's source_identity also proves the Raman source binding and the actual
    # production propagate source.  It executes only inside the G E module.
    e_source = e_module.source_identity(r8_path)
    propagate_path = Path(__file__).resolve().parents[2] / "KHz_filament" / "propagate.py"
    propagate_identity = source_binding(propagate_path)
    source = _make_source_identity(
        own=own,
        f=f_identity,
        e=e_identity,
        d=d_identity,
        r8=r8_identity,
        e_module=e_module,
        propagate=propagate_identity,
    )
    source["e_source_identity"] = copy.deepcopy(e_source)
    source["private_R8_module"] = r8_module.__name__
    source["private_D_module"] = d_module.__name__

    def factory(frozen_propagator: Callable[..., Any]):
        _reject_optimized()
        if not callable(frozen_propagator):
            raise SourceBindingError("frozen propagator is not callable")
        propagator, dispatch = e_factory(frozen_propagator)
        handoff = getattr(dispatch, "cpu_s4_3e_state", None)
        if handoff is None or handoff.__class__.__module__ != __name__.replace(
            ".candidate", ".controlled_callsite"
        ):
            # The class is intentionally defined by the G callsite builder;
            # do not accept a state supplied by F/E caches.
            if handoff is None or not handoff.__class__.__name__.startswith("_ControlledHandoffState"):
                raise SourceBindingError("G factory did not create its private controlled state")
        # E's patched _clone_bound_propagator is installed below, once the
        # state exists.  The existing E clone is therefore replaced by the G
        # source-bound clone before it can be returned to callers.
        if getattr(propagator, "__cpu_s4_3g_source_binding__", None) is None:
            # The G E namespace intentionally calls this hook instead of E's
            # original transformer.  It is a hard identity check, not a
            # fallback to the old E propagation clone.
            raise SourceBindingError("G source-bound propagate hook was not used")
        dispatch.cpu_s4_3g_state = handoff
        dispatch.cpu_s4_3g_full_heun = handoff.full_heun
        dispatch.cpu_s4_3g_metrics = handoff.metrics
        dispatch.cpu_s4_3g_backend_binding = copy.deepcopy(
            getattr(propagator, "__cpu_s4_3g_backend_binding__", {})
        )
        dispatch.cpu_s4_3g_source_identity = copy.deepcopy(source)
        dispatch.cpu_s4_3g_variant = DEFAULT_VARIANT
        dispatch.cpu_s4_3g_route_contract = {
            "actual": "G_ACTUAL",
            "fallback": "F_STRICT_FALLBACK",
            "rejected": "REJECTED",
            "raw_full_heun": "F_STRICT_FALLBACK",
            "public_trust_kwargs": "forbidden",
            "controlled_hash_roles": [
                "E_stage1",
                "Omega_stage1",
                "Omega_apply",
                "Omega_stage2",
            ],
            "d_body_names": list(_BODY_NAMES),
            "d_block_size": 32,
            "d_raman_workers": 48,
            "d_fastmath": False,
        }
        propagator._cpu_s4_3g_state = handoff
        propagator._cpu_s4_3g_full_heun = handoff.full_heun
        propagator._cpu_s4_3g_execution = handoff.metrics()
        dispatch.cpu_s4_3g_execution = handoff.metrics()
        factory.last_propagator = propagator
        factory.last_dispatch = dispatch
        factory.last_state = handoff
        factory.last_full_heun = handoff.full_heun
        factory.last_source_identity = copy.deepcopy(source)
        return propagator, dispatch

    # Patch E's private clone hook in the G-private E module before any caller
    # invokes the returned factory.  The hook captures each actual state.
    def g_clone_bound(base: Callable[..., Any], handoff: Any):
        clone, binding = bind_propagator(base, handoff, source_path=propagate_path)
        handoff._g_propagate_binding = copy.deepcopy(binding)
        return clone, binding["propagate"]

    e_module.__dict__["_clone_bound_propagator"] = g_clone_bound
    # Build E's factory only after the private clone hook is installed; no
    # frozen module/cache is involved.
    e_factory = e_module.make_factory(
        "C_STRICT" if normalized == "C_STRICT" else "C_INTERNAL",
        frozen_r8_path=r8_path,
        kernel_threads=24,
        temporal_workers=48,
        spatial_workers=24,
        raman_workers=48,
        block_size=32,
        observe=None,
    )

    factory.variant = DEFAULT_VARIANT
    factory.alias_variant = normalized
    factory.candidate_schema = CANDIDATE_SCHEMA
    factory.candidate_family = "RAMAN_CONTROLLED_CALLSITE_BORROW_V1"
    factory.source_identity = copy.deepcopy(source)
    factory.last_source_identity = copy.deepcopy(source)
    factory.frozen_f_raw_sha256 = F_RAW_SHA256
    factory.frozen_e_raw_sha256 = E_RAW_SHA256
    factory.frozen_d_raw_sha256 = D_RAW_SHA256
    factory.frozen_r8_raw_sha256 = R8_RAW_SHA256
    factory.private_e_module = e_module.__name__
    factory.private_r8_module = r8_module.__name__
    factory.private_d_module = d_module.__name__
    factory.last_propagator = None
    factory.last_dispatch = None
    factory.last_state = None
    factory.last_full_heun = None
    return factory


__all__ = [
    "CandidateUnsupported",
    "HandoffRejected",
    "SourceBindingError",
    "StageBindingError",
    "CANDIDATE_SCHEMA",
    "DEFAULT_VARIANT",
    "make_factory",
    "source_identity",
]
