"""Exact, narrowly scoped handling for the two GPU resource telemetry leaves.

The scientific output comparator remains the frozen G1/G2 comparator.  This
module only builds a temporary disk view which omits the two explicitly
approved resource vectors, validates those vectors separately, and records the
omission in a provenance sidecar.  It deliberately has no NumPy import at
module import time so importing the GPU manager does not pull in CPU optional
dependencies.
"""
from __future__ import annotations

import errno
import io
import json
import os
from pathlib import Path
import shutil
import zipfile
from typing import Any, Mapping


RESOURCE_KEYS = (
    "gpu_allocated_step_bytes",
    "gpu_reserved_step_bytes",
)
RESOURCE_PATHS = tuple("bundle.returned.2." + key for key in RESOURCE_KEYS)
RESOURCE_POLICY = "USER_APPROVED_TWO_GPU_RESOURCE_TELEMETRY_KEYS_ONLY"
_VECTOR_LENGTH = 8


class ResourceTelemetryError(RuntimeError):
    """Raised when the approved telemetry cannot be validated fail-closed."""


def _fail(reason: str, detail: Any = None) -> None:
    if detail is None:
        raise ResourceTelemetryError(reason)
    raise ResourceTelemetryError(f"{reason}:{detail}")


def _strict_int(value: Any, name: str) -> int:
    # bool is an int subclass but is never a valid byte count.
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("RESOURCE_INTEGER_REQUIRED", name)
    result = int(value)
    if result < 0:
        _fail("RESOURCE_INTEGER_NEGATIVE", name)
    return result


def _layout(value: Any) -> dict[str, Any]:
    """Return the same small layout vocabulary used by the frozen comparator."""
    dtype = getattr(value, "dtype", None)
    shape = getattr(value, "shape", None)
    strides = getattr(value, "strides", None)
    flags = getattr(value, "flags", None)
    if dtype is None or shape is None or strides is None or flags is None:
        _fail("RESOURCE_ARRAY_REQUIRED")
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


def _vector(value: Any, key: str, device_total_bytes: int) -> dict[str, Any]:
    if key not in RESOURCE_KEYS:
        _fail("RESOURCE_KEY_NOT_APPROVED", key)
    dtype = getattr(value, "dtype", None)
    if dtype is None or str(dtype) != "int64" or getattr(dtype, "kind", None) != "i" or getattr(dtype, "itemsize", None) != 8:
        _fail("RESOURCE_DTYPE_NOT_INT64", key)
    shape = tuple(getattr(value, "shape", ()))
    if shape != (_VECTOR_LENGTH,):
        _fail("RESOURCE_SHAPE_NOT_EIGHT", key,)
    total = _strict_int(device_total_bytes, "device_total_bytes")
    if total <= 0:
        _fail("RESOURCE_DEVICE_TOTAL_NOT_POSITIVE")
    if total > (1 << 63) - 1:
        _fail("RESOURCE_DEVICE_TOTAL_EXCEEDS_INT64")
    try:
        values = [int(item) for item in value.tolist()]
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        _fail("RESOURCE_VALUES_UNREADABLE", key)
        raise AssertionError from exc
    if len(values) != _VECTOR_LENGTH or any(item < 0 for item in values):
        _fail("RESOURCE_VALUES_NEGATIVE", key)
    if any(item > total for item in values):
        _fail("RESOURCE_VALUES_EXCEED_DEVICE_TOTAL", key)
    return {
        "path": "bundle.returned.2." + key,
        "values": values,
        "dtype": str(dtype),
        "shape": list(shape),
        "layout": _layout(value),
        "device_total_bytes": total,
    }


def _candidate_vectors(bundle: Any) -> Mapping[str, Any]:
    if not isinstance(bundle, Mapping):
        _fail("RESOURCE_BUNDLE_MAPPING_REQUIRED")
    returned = bundle.get("returned")
    if not isinstance(returned, (tuple, list)) or len(returned) <= 2:
        _fail("RESOURCE_RETURNED_THIRD_ITEM_MISSING")
    diagnostics = returned[2]
    if not isinstance(diagnostics, Mapping):
        _fail("RESOURCE_DIAGNOSTICS_MAPPING_REQUIRED")
    missing = [key for key in RESOURCE_KEYS if key not in diagnostics]
    if missing:
        _fail("RESOURCE_KEYS_MISSING", ",".join(missing))
    return diagnostics


def inspect_candidate_resources(
    bundle: Any,
    device_total_bytes: int,
    *,
    fingerprint_source: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate and describe candidate vectors without changing ``bundle``."""
    diagnostics = _candidate_vectors(bundle)
    vectors = {
        key: _vector(diagnostics[key], key, device_total_bytes) for key in RESOURCE_KEYS
    }
    allocated = vectors[RESOURCE_KEYS[0]]["values"]
    reserved = vectors[RESOURCE_KEYS[1]]["values"]
    if any(left > right for left, right in zip(allocated, reserved)):
        _fail("RESOURCE_ALLOCATED_EXCEEDS_RESERVED")
    result = {
        "schema": "filament.cpu_s46a.gpu_resource_telemetry.v1",
        "policy": RESOURCE_POLICY,
        "allowed_keys": list(RESOURCE_KEYS),
        "source": "bundle.returned[2]",
        "vectors": vectors,
        "pairwise_validation": {
            "allocated_le_reserved": True,
            "vector_length": _VECTOR_LENGTH,
        },
    }
    if fingerprint_source is not None:
        result["fingerprint_source"] = _source_record(fingerprint_source)
    return result


def _source_record(path: str | os.PathLike[str]) -> dict[str, Any]:
    from tools.hpc_ops import provenance_v2 as pv
    target = Path(path)
    if not target.is_file():
        _fail("RESOURCE_FINGERPRINT_SOURCE_MISSING", str(target))
    return {
        "path": str(target),
        "classification": "external",
        "hash_scope": "raw_bytes",
        "raw_sha256": pv.raw_sha256_file(target),
        "size_bytes": int(target.stat().st_size),
    }


def _tree_at_path(tree: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    parts = path.split(".")
    if parts[:1] != ["bundle"]:
        _fail("RESOURCE_TREE_PATH_ROOT", path)
    node: Any = tree
    for part in parts[1:]:
        if not isinstance(node, Mapping):
            _fail("RESOURCE_TREE_PATH_NOT_CONTAINER", path)
        kind = node.get("kind")
        if kind == "dict":
            items = node.get("items")
            if not isinstance(items, Mapping) or part not in items:
                _fail("RESOURCE_TREE_PATH_MISSING", path)
            node = items[part]
        elif kind in ("tuple", "list"):
            items = node.get("items")
            try:
                node = items[int(part)]
            except (TypeError, ValueError, IndexError):
                _fail("RESOURCE_TREE_PATH_MISSING", path)
        else:
            _fail("RESOURCE_TREE_PATH_NOT_CONTAINER", path)
    if not isinstance(node, Mapping):
        _fail("RESOURCE_TREE_NODE_INVALID", path)
    return node


def _reference_vectors(reference_root: Path, device_total_bytes: int) -> dict[str, dict[str, Any]]:
    tree_path = reference_root / "TREE.json"
    archive_path = reference_root / "ARRAYS.zip"
    if not tree_path.is_file() or not archive_path.is_file():
        _fail("RESOURCE_REFERENCE_ARRAYS_OR_TREE_MISSING", str(reference_root))
    tree = json.loads(tree_path.read_text(encoding="utf-8-sig"))
    result: dict[str, dict[str, Any]] = {}
    import numpy as np

    with zipfile.ZipFile(archive_path, "r") as archive:
        for key, path in zip(RESOURCE_KEYS, RESOURCE_PATHS):
            node = _tree_at_path(tree, path)
            if node.get("kind") != "array" or not node.get("file"):
                _fail("RESOURCE_REFERENCE_NODE_NOT_ARRAY", path)
            try:
                member = archive.getinfo(str(node["file"]))
                if member.file_size > 4096:
                    _fail("RESOURCE_REFERENCE_MEMBER_NOT_BOUNDED", path)
                raw = archive.read(member)
            except KeyError:
                _fail("RESOURCE_REFERENCE_ARRAY_MEMBER_MISSING", path)
            value = np.load(io.BytesIO(raw), allow_pickle=False)
            checked = _vector(value, key, device_total_bytes)
            # Preserve the frozen tree layout, not the layout reconstructed by
            # np.load (which intentionally owns a fresh decoded buffer).
            frozen_layout = node.get("layout")
            if not isinstance(frozen_layout, Mapping):
                _fail("RESOURCE_REFERENCE_LAYOUT_MISSING", path)
            if frozen_layout.get("dtype") != checked["dtype"] or frozen_layout.get("shape") != checked["shape"]:
                _fail("RESOURCE_REFERENCE_LAYOUT_MISMATCH", path)
            checked["layout"] = dict(frozen_layout)
            result[key] = checked
    return result


def _strip_tree(node: Any, path: str = "bundle") -> Any:
    if isinstance(node, dict):
        if node.get("kind") == "dict" and isinstance(node.get("items"), dict):
            node["items"] = {
                key: _strip_tree(value, path + "." + str(key))
                for key, value in node["items"].items()
                if path + "." + str(key) not in RESOURCE_PATHS
            }
        elif node.get("kind") in ("tuple", "list") and isinstance(node.get("items"), list):
            node["items"] = [
                _strip_tree(value, path + "." + str(index))
                for index, value in enumerate(node["items"])
            ]
        else:
            for key, value in list(node.items()):
                node[key] = _strip_tree(value, path + "." + str(key))
    elif isinstance(node, list):
        return [_strip_tree(value, path + "." + str(index)) for index, value in enumerate(node)]
    return node


def _strip_candidate(bundle: Any) -> Any:
    if not isinstance(bundle, Mapping):
        _fail("RESOURCE_BUNDLE_MAPPING_REQUIRED")
    returned = bundle.get("returned")
    if not isinstance(returned, (tuple, list)) or len(returned) <= 2:
        _fail("RESOURCE_RETURNED_THIRD_ITEM_MISSING")
    candidate = dict(bundle)
    stripped = dict(returned[2])
    for key in RESOURCE_KEYS:
        if key not in stripped:
            _fail("RESOURCE_KEYS_MISSING", key)
        stripped.pop(key)
    copied_returned = list(returned)
    copied_returned[2] = stripped
    candidate["returned"] = tuple(copied_returned) if isinstance(returned, tuple) else copied_returned
    return candidate


def _link_or_copy(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        return "hardlink"
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copyfile(source, target)
        return "copy_fallback_exdev"


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _make_reference_view(reference_root: Path, view_root: Path) -> tuple[Path, dict[str, Any]]:
    reference_resolved = reference_root.resolve(strict=True)
    view_resolved = view_root.resolve(strict=False)
    for child, parent in ((view_resolved, reference_resolved), (reference_resolved, view_resolved)):
        try:
            child.relative_to(parent)
        except ValueError:
            continue
        _fail("RESOURCE_VIEW_SOURCE_OVERLAP", str(view_root))
    view_root.mkdir(parents=True, exist_ok=False)
    tree = json.loads((reference_root / "TREE.json").read_text(encoding="utf-8-sig"))
    aliases = json.loads((reference_root / "ALIASES.json").read_text(encoding="utf-8-sig"))
    if not isinstance(aliases, list):
        _fail("RESOURCE_REFERENCE_ALIASES_NOT_LIST")
    filtered_aliases = [
        row for row in aliases
        if isinstance(row, Mapping)
        and row.get("left") not in RESOURCE_PATHS
        and row.get("right") not in RESOURCE_PATHS
    ]
    _write_json(view_root / "TREE.json", _strip_tree(tree))
    _write_json(view_root / "ALIASES.json", filtered_aliases)
    copied: dict[str, str] = {}
    for source in reference_root.iterdir():
        if not source.is_file() or source.name in {"TREE.json", "ALIASES.json"}:
            continue
        copied[source.name] = _link_or_copy(source, view_root / source.name)
    if "ARRAYS.zip" not in copied:
        _fail("RESOURCE_REFERENCE_ARRAYS_ZIP_MISSING")
    return view_root, {"files": copied, "tree": "sanitized_copy", "aliases": "sanitized_copy"}


def _delta(candidate: list[int], reference: list[int]) -> list[int]:
    return [int(right) - int(left) for left, right in zip(reference, candidate)]


def compare_terminal_with_resource_policy(
    extended: Any,
    reference_output: str | os.PathLike[str],
    candidate_bundle: Any,
    output: str | os.PathLike[str],
    *,
    device_total_bytes: int,
    view_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Compare all non-resource leaves with the frozen comparator.

    The returned status explicitly describes the approved resource exception;
    it is never the old full-output exact status.
    """
    reference_root = Path(reference_output).resolve(strict=True)
    output_path = Path(output)
    if view_root is None:
        view_path = output_path.parent / "TERMINAL_RESOURCE_VIEW"
    else:
        view_path = Path(view_root)
    candidate_observation = inspect_candidate_resources(
        candidate_bundle, device_total_bytes,
        fingerprint_source=output_path.parent / "OUTPUT" / "FINGERPRINTS.json",
    )
    reference_vectors = _reference_vectors(reference_root, device_total_bytes)
    candidate_vectors = candidate_observation["vectors"]
    reference_source = _source_record(reference_root / "FINGERPRINTS.json")
    view_path, view_provenance = _make_reference_view(reference_root, view_path)
    science_result = extended.compare_full_output(
        view_path, _strip_candidate(candidate_bundle), output_path
    )
    if science_result.get("status") != "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS":
        _fail("RESOURCE_SCIENCE_COMPARISON_NOT_EXACT", science_result.get("status"))
    sidecar = {
        "schema": "filament.cpu.s46a.gpu_resource_telemetry.v1",
        "policy": RESOURCE_POLICY,
        "allowed_keys": list(RESOURCE_KEYS),
        "excluded_paths": list(RESOURCE_PATHS),
        "full_output_exact_old_scope": False,
        "scientific_comparison_status": science_result["status"],
        "scientific_comparison_path": str(output_path),
        "reference_view": {
            "path": str(view_path),
            "source": str(reference_root),
            "provenance": view_provenance,
        },
        "original_full_fingerprint_source": {
            "reference": reference_source,
            "candidate": candidate_observation["fingerprint_source"],
        },
        "device_total_bytes": _strict_int(device_total_bytes, "device_total_bytes"),
        "vectors": {},
    }
    for key in RESOURCE_KEYS:
        sidecar["vectors"][key] = {
            "path": RESOURCE_PATHS[RESOURCE_KEYS.index(key)],
            "reference": reference_vectors[key],
            "candidate": candidate_vectors[key],
            "per_step_delta_candidate_minus_reference": _delta(
                candidate_vectors[key]["values"], reference_vectors[key]["values"]
            ),
        }
    sidecar_path = output_path.parent / "RESOURCE_TELEMETRY_EXACT.json"
    _write_json(sidecar_path, sidecar)
    return {
        "status": "PASS_SCIENCE_VIEW_WITH_APPROVED_RESOURCE_TELEMETRY",
        "scientific_status": science_result["status"],
        "scientific_comparison": science_result,
        "resource_telemetry_policy": RESOURCE_POLICY,
        "resource_telemetry_sidecar": str(sidecar_path),
        "reference_view": str(view_path),
        "full_output_exact_old_scope": False,
        "excluded_resource_paths": list(RESOURCE_PATHS),
    }

