"""Bind frozen timing/boundary machinery to one explicitly selected SG role."""
from __future__ import annotations

import copy
from pathlib import Path
from types import FunctionType, SimpleNamespace

from tools.hpc_ops import provenance_v2 as pv
from tools.cpu_s4_checkpoint_support import load_g2_adapter, g2_adapter_path


def attach(raw_propagator, dispatch, *, role, mode, capture_root, owned_identity):
    """Reuse the frozen G2 clone/collector without changing its shared globals."""
    from .scientific_context import frozen_tools
    repo = Path(__file__).resolve().parents[3]
    frozen_tools(repo)
    original = load_g2_adapter()
    expected_adapter = g2_adapter_path()
    if Path(getattr(original, "__file__", "")).resolve() != expected_adapter.resolve(strict=True):
        raise RuntimeError("SMALLGRID_FROZEN_ADAPTER_ACTUAL_MODULE_MISMATCH")
    namespace = dict(vars(original))
    for name, value in tuple(namespace.items()):
        if isinstance(value, FunctionType) and value.__globals__ is vars(original):
            clone = FunctionType(value.__code__, namespace, value.__name__, value.__defaults__, value.__closure__)
            clone.__kwdefaults__ = value.__kwdefaults__
            clone.__annotations__ = value.__annotations__
            clone.__qualname__ = value.__qualname__
            clone.__module__ = value.__module__
            namespace[name] = clone
        elif name.endswith("CACHE") and isinstance(value, dict):
            namespace[name] = {}
    common = SimpleNamespace(**vars(original._common))
    common.factory = lambda unused_role: (lambda unused_base: (raw_propagator, dispatch))
    namespace["_common"] = common
    old_external = namespace["_external_frozen_source_binding"]
    own = {str(Path(record["path"]).resolve()): copy.deepcopy(record) for record in owned_identity["files"].values()}

    def external_binding(path, raw):
        actual_path = Path(path).resolve(strict=True)
        record = own.get(str(actual_path))
        if record is None:
            return old_external(path, raw)
        digest = pv.raw_sha256(raw)
        if digest != record["raw_sha256"] or len(raw) != record["size_bytes"]:
            raise RuntimeError("SMALLGRID_OWNED_SOURCE_CHANGED_BEFORE_CALL")
        return dict(relative_path=actual_path.relative_to(repo).as_posix(), filename=str(actual_path),
                    classification="external", hash_scope="raw_bytes", git_blob_oid=None,
                    external_frozen_source=True, external_binding_report="SG_OWNED_PRE_CALL_SOURCE_RECORDS",
                    expected_raw_sha256=record["raw_sha256"], source_raw_sha256=digest,
                    source_size_bytes=len(raw), external_source_path_match="exact_owned_namespace")

    namespace["_external_frozen_source_binding"] = external_binding
    frozen_role = "I0" if role == "I0_SMALL" else "G_CONTROLLED_INPUT"
    frozen_mode = "QUALIFY_BOUNDARY" if mode == "QUALIFY_BOUNDARY" else "PURE_OFF_REAL8"
    wrapped, actual_dispatch, clock, binding, audit, lease = namespace["build_role"](frozen_role, frozen_mode, capture_root=capture_root)
    if actual_dispatch is not dispatch:
        raise RuntimeError("SMALLGRID_ACTUAL_DISPATCH_BINDING_CHANGED")
    binding = copy.deepcopy(binding)
    binding.update(smallgrid_role=role, smallgrid_mode=mode, owned_candidate=copy.deepcopy(owned_identity),
                   wrapper_scope="UNCHANGED_G2_COLLECTOR_TIMING_PREPOST_GUARDS_WITH_NEW_OWNED_SOURCE_BINDING")
    return wrapped, actual_dispatch, clock, binding, audit, lease
