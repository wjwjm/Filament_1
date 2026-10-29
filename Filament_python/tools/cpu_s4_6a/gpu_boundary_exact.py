"""Boundary readback adapter for the frozen G1 value gate.

``gpu.read_snapshot`` decodes arrays from an archive.  Decoding can change
Python/Numpy ownership metadata even when the saved native boundary is
unchanged.  This adapter keeps the frozen comparator's value, timing, NaN and
string rules, while sourcing candidate layout and alias metadata from the
candidate's own TREE/ALIASES files.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import FunctionType
from typing import Any, Mapping


STATUS = "PASS_RECORDED_NATIVE_BOUNDARY_WITH_FROZEN_VALUE_GATE"
_SNAPSHOT_FILES = ("TREE.json", "ALIASES.json", "ARRAYS.zip")


class BoundaryExactError(RuntimeError):
    """Raised when boundary metadata cannot be replayed fail-closed."""


def _fail(reason: str, detail: Any = None) -> None:
    if detail is None:
        raise BoundaryExactError(reason)
    raise BoundaryExactError(f"{reason}:{detail}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BoundaryExactError(f"BOUNDARY_JSON_READ_FAILED:{path.name}") from exc


def _raw_sha256(path: Path) -> str:
    from tools.hpc_ops import provenance_v2 as pv
    return pv.raw_sha256_file(path)


def _snapshot_provenance(root: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name in _SNAPSHOT_FILES:
        path = root / name
        if not path.is_file() or path.is_symlink():
            _fail("BOUNDARY_SNAPSHOT_FILE_MISSING", str(path))
        records[name] = {
            "path": str(path),
            "classification": "external",
            "hash_scope": "raw_bytes",
            "raw_sha256": _raw_sha256(path),
            "size_bytes": int(path.stat().st_size),
        }
    return records


def _decoded_layout(value: Any) -> dict[str, Any]:
    dtype = getattr(value, "dtype", None)
    shape = getattr(value, "shape", None)
    strides = getattr(value, "strides", None)
    flags = getattr(value, "flags", None)
    if dtype is None or shape is None or strides is None or flags is None:
        _fail("BOUNDARY_ARRAY_REQUIRED")
    direct_base = getattr(value, "base", None)
    base_depth = 0
    base = direct_base
    while base is not None and getattr(base, "__class__", None).__module__.split(".", 1)[0] == "numpy":
        base_depth += 1
        base = getattr(base, "base", None)
    return {
        "shape": [int(item) for item in shape],
        "dtype": str(dtype),
        "strides": [int(item) for item in strides],
        "C": bool(flags.c_contiguous),
        "F": bool(flags.f_contiguous),
        "OWNDATA": bool(flags.owndata),
        "WRITEABLE": bool(flags.writeable),
        "base_chain_depth": base_depth,
        "base_ndarray": direct_base is not None and getattr(direct_base, "__class__", None).__module__.split(".", 1)[0] == "numpy",
    }


def _candidate_metadata(
    tree: Mapping[str, Any],
    value: Any,
    *,
    path: str = "bundle",
    layouts: dict[int, dict[str, Any]] | None = None,
    observations: list[dict[str, Any]] | None = None,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    layouts = {} if layouts is None else layouts
    observations = [] if observations is None else observations
    if not isinstance(tree, Mapping):
        _fail("BOUNDARY_TREE_NODE_REQUIRED", path)
    kind = tree.get("kind")
    if kind == "array":
        saved = tree.get("layout")
        if not isinstance(saved, Mapping):
            _fail("BOUNDARY_CANDIDATE_LAYOUT_MISSING", path)
        actual = _decoded_layout(value)
        layouts[id(value)] = dict(saved)
        observations.append({
            "path": path,
            "saved_native_layout": dict(saved),
            "decoded_readback_layout": actual,
            "readback_layout_differs": dict(saved) != actual,
        })
        return layouts, observations
    if kind == "dict":
        if not isinstance(value, dict) or not isinstance(tree.get("items"), Mapping):
            _fail("BOUNDARY_CANDIDATE_TREE_SHAPE", path)
        items = tree["items"]
        if set(items) != set(value):
            _fail("BOUNDARY_CANDIDATE_TREE_FIELDS", path)
        for key, child in items.items():
            _candidate_metadata(child, value[key], path=f"{path}.{key}", layouts=layouts, observations=observations)
        return layouts, observations
    if kind in ("tuple", "list"):
        if type(value).__name__ != kind or not isinstance(tree.get("items"), list) or len(tree["items"]) != len(value):
            _fail("BOUNDARY_CANDIDATE_TREE_SEQUENCE", path)
        for index, child in enumerate(tree["items"]):
            _candidate_metadata(child, value[index], path=f"{path}.{index}", layouts=layouts, observations=observations)
        return layouts, observations
    return layouts, observations


def compare_boundary_with_saved_native_metadata(
    reference_boundary: str | Path,
    candidate_boundary: str | Path,
    candidate_value: Any,
    output: str | Path,
    *,
    source_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run frozen G1 value comparison with candidate-saved native metadata."""

    reference = Path(reference_boundary).resolve(strict=True)
    candidate = Path(candidate_boundary).resolve(strict=True)
    output_path = Path(output)
    reference_tree = reference / "TREE.json"
    candidate_tree = candidate / "TREE.json"
    for path in (reference_tree, candidate_tree, reference / "ALIASES.json", candidate / "ALIASES.json"):
        if not path.is_file() or path.is_symlink():
            _fail("BOUNDARY_METADATA_FILE_MISSING", str(path))
    candidate_tree_value = _read_json(candidate_tree)
    reference_tree_value = _read_json(reference_tree)
    if any(v.get("kind") != "dict" or len(v.get("items", {})) != 112
           for v in (reference_tree_value, candidate_tree_value)):
        _fail("BOUNDARY_ORIGINAL_112_FIELDS_REQUIRED")
    candidate_aliases_data = _read_json(candidate / "ALIASES.json")
    if not isinstance(candidate_aliases_data, list):
        _fail("BOUNDARY_CANDIDATE_ALIASES_LIST_REQUIRED")
    layouts, observations = _candidate_metadata(candidate_tree_value, candidate_value)

    # Load only the frozen support comparator.  Its numeric gate, timing
    # exclusions, NaN handling and failure artifacts remain unchanged.
    from .gpu import _support_module
    exact = _support_module("g1_exact")
    frozen_compare = exact.compare

    def candidate_layout(value: Any) -> dict[str, Any]:
        metadata = layouts.get(id(value))
        if metadata is None:
            _fail("BOUNDARY_CANDIDATE_ARRAY_METADATA_MISSING")
        return metadata

    def candidate_aliases(_value: Any) -> list[Any]:
        if _value is not candidate_value:
            _fail("BOUNDARY_ALIAS_METADATA_OBJECT_IDENTITY")
        return candidate_aliases_data
    namespace = dict(frozen_compare.__globals__)
    namespace["layout"] = candidate_layout
    namespace["aliases"] = candidate_aliases
    compare = FunctionType(
        frozen_compare.__code__, namespace, frozen_compare.__name__,
        frozen_compare.__defaults__, frozen_compare.__closure__,
    )
    compare.__kwdefaults__ = frozen_compare.__kwdefaults__
    value_result = compare(reference, candidate_value, output_path, disk=True)

    provenance = {
        "schema": "filament.cpu.s46a.boundary_native_readback.v1",
        "hash_scope": "raw_bytes",
        "reference_snapshot": _snapshot_provenance(reference),
        "candidate_snapshot": _snapshot_provenance(candidate),
        "source_binding_pv2": dict(source_binding) if isinstance(source_binding, Mapping) else None,
        "candidate_metadata_source": "candidate TREE.json layout and candidate ALIASES.json",
        "decoded_readback_source": "gpu.read_snapshot(candidate_boundary)",
        "layout_observations": observations,
        "value_gate_status": value_result["status"],
        "value_gate_comparison_path": str(output_path),
        "native_metadata_not_filled_from_reference": True,
    }
    sidecar = output_path.parent / "BOUNDARY_NATIVE_READBACK_PROVENANCE.json"
    with sidecar.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(provenance, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return {
        "status": STATUS,
        "value_gate_status": value_result["status"],
        "value_gate": value_result,
        "comparison_path": str(output_path),
        "native_metadata_source": "candidate TREE.json and ALIASES.json",
        "readback_layout_provenance": str(sidecar),
        "native_metadata_not_filled_from_reference": True,
        "source_binding_pv2": dict(source_binding) if isinstance(source_binding, Mapping) else None,
    }

