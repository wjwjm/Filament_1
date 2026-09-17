"""Small, parameterized evidence helpers for E5-1A.

The scientific comparison contract is deliberately stricter than a file hash:
both sides must be finite, have identical shape and dtype, have equal
``sha256_array`` values, and pass ``numpy.array_equal``.  The helpers here
only inspect supplied arrays/files and write durable evidence; they do not
choose a scientific baseline or alter any lifecycle state.
"""

from __future__ import annotations

import json
import os
import hashlib
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .hr4e_timestep import sha256_array, sha256_file


EVIDENCE_SCHEMA = "khz_filament.hr4e5.e5_1a.evidence.v1"
READY_SCHEMA = "khz_filament.hr4e5.e5_1a.ready.v1"


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path: str | Path, value: Mapping[str, Any], *, overwrite: bool = True) -> None:
    """Write a JSON receipt using a same-directory fsynced replacement."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(_json_safe(dict(value)), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if not overwrite and destination.exists():
            raise FileExistsError(destination)
        os.replace(temporary, destination)
        if os.name != "nt":
            try:
                descriptor = os.open(str(destination.parent), os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError:
                pass
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _as_array(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as error:
        raise ValueError(f"{name} is not array-like") from error
    if array.dtype == np.dtype("O"):
        raise ValueError(f"{name} has object dtype")
    return array


def _first_difference(left: np.ndarray, right: np.ndarray) -> list[int] | None:
    if left.shape != right.shape:
        return None
    equal = np.equal(left, right)
    if equal.ndim == 0:
        return None if bool(equal) else []
    where = np.argwhere(~equal)
    return where[0].astype(int).tolist() if where.size else None


def compare_arrays_exact(
    reference: Any, candidate: Any, *, name: str = "array",
    reference_path: str | Path | None = None,
    candidate_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare two arrays under the E5 exact contract."""
    left, right = _as_array(reference, name=f"{name} reference"), _as_array(candidate, name=f"{name} candidate")
    shape_equal = bool(left.shape == right.shape)
    dtype_equal = bool(left.dtype == right.dtype)
    left_finite = bool(np.all(np.isfinite(left)))
    right_finite = bool(np.all(np.isfinite(right)))
    left_hash = sha256_array(left) if left_finite else None
    right_hash = sha256_array(right) if right_finite else None
    hash_equal = bool(left_hash is not None and left_hash == right_hash)
    array_equal = bool(shape_equal and dtype_equal and left_finite and right_finite and np.array_equal(left, right))
    status = "PASS" if shape_equal and dtype_equal and left_finite and right_finite and hash_equal and array_equal else "FAIL"
    return {
        "schema": EVIDENCE_SCHEMA,
        "name": str(name),
        "status": status,
        "shape_equal": shape_equal,
        "dtype_equal": dtype_equal,
        "reference_shape": list(left.shape),
        "candidate_shape": list(right.shape),
        "reference_dtype": left.dtype.name,
        "candidate_dtype": right.dtype.name,
        "reference_finite": left_finite,
        "candidate_finite": right_finite,
        "reference_sha256_array": left_hash,
        "candidate_sha256_array": right_hash,
        "hash_equal": hash_equal,
        "array_equal": array_equal,
        "first_difference_index": _first_difference(left, right) if not array_equal and shape_equal and dtype_equal else None,
        "reference_path": None if reference_path is None else str(reference_path),
        "candidate_path": None if candidate_path is None else str(candidate_path),
    }


def compare_array_exact(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Singular spelling retained for callers that use the contract wording."""
    return compare_arrays_exact(*args, **kwargs)


def _load_array(value: Any) -> tuple[np.ndarray, str | None]:
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix == ".npy":
            return np.load(path, mmap_mode="r", allow_pickle=False), str(path)
        if path.suffix == ".npz":
            loaded = np.load(path, allow_pickle=False)
            names = list(loaded.files)
            if len(names) != 1:
                loaded.close()
                raise ValueError(f"NPZ object must contain exactly one array: {path}")
            array = np.asarray(loaded[names[0]])
            loaded.close()
            return array, str(path)
    return _as_array(value, name="object"), None


def compare_object_sets(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], *,
    layer: str = "scientific", require_finite: bool = True,
) -> dict[str, Any]:
    """Compare named arrays and fail if an object is missing on either side."""
    left_keys, right_keys = set(reference), set(candidate)
    rows: list[dict[str, Any]] = []
    missing_reference = sorted(right_keys - left_keys)
    missing_candidate = sorted(left_keys - right_keys)
    for key in sorted(left_keys & right_keys):
        left, left_path = _load_array(reference[key])
        right, right_path = _load_array(candidate[key])
        row = compare_arrays_exact(left, right, name=f"{layer}:{key}", reference_path=left_path, candidate_path=right_path)
        if not require_finite:
            # The E5 default requires finite values.  This opt-out exists only
            # for provenance-only metadata probes and is explicit in evidence.
            row["finite_requirement"] = "not_required"
            if row["shape_equal"] and row["dtype_equal"] and row["hash_equal"] and row["array_equal"]:
                row["status"] = "PASS"
        rows.append(row)
    failures = [row for row in rows if row.get("status") != "PASS"]
    status = "PASS" if left_keys and not missing_reference and not missing_candidate and not failures else "FAIL"
    return {
        "schema": EVIDENCE_SCHEMA,
        "layer": str(layer),
        "status": status,
        "expected_object_count": len(left_keys),
        "candidate_object_count": len(right_keys),
        "compared_object_count": len(rows),
        "missing_reference": missing_reference,
        "missing_candidate": missing_candidate,
        "mismatch_count": len(failures) + len(missing_reference) + len(missing_candidate),
        "rows": rows,
        "created_utc": _utc(),
    }


compare_exact_objects = compare_object_sets


def write_exact_report(path: str | Path, report: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(report)
    payload.setdefault("schema", EVIDENCE_SCHEMA)
    payload.setdefault("created_utc", _utc())
    atomic_json(path, payload)
    payload["report_path"] = str(Path(path))
    payload["report_sha256"] = sha256_file(path)
    return payload


def object_manifest(
    objects: Mapping[str, Any], *, layer: str, root: str | Path | None = None,
    role: str = "scientific_array",
) -> list[dict[str, Any]]:
    """Describe objects without copying them, suitable for a receipt index."""
    base = None if root is None else Path(root).resolve()
    rows = []
    for name in sorted(objects):
        value = objects[name]
        array, source_path = _load_array(value)
        path_text = source_path
        if path_text is not None and base is not None:
            try:
                path_text = str(Path(path_text).resolve().relative_to(base)).replace("\\", "/")
            except ValueError:
                path_text = str(Path(path_text).resolve())
        finite = bool(np.all(np.isfinite(array)))
        rows.append({
            "layer": str(layer), "name": str(name), "role": str(role),
            "path": path_text, "shape": list(array.shape), "dtype": array.dtype.name,
            "finite": finite, "sha256_array": sha256_array(array) if finite else None,
            "bytes": int(array.nbytes),
        })
    return rows


def lineage_binding(
    *, parent_root: str | Path, child_root: str | Path,
    parent_generation: str, parent_content_sha256: str,
    child_generation: str, child_content_sha256: str,
    fields: Mapping[str, Mapping[str, Any]],
    exact_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the durable parent NEXT to child CURRENT binding receipt."""
    return {
        "schema": READY_SCHEMA,
        "status": "READY",
        "parent_root": str(Path(parent_root).resolve()),
        "child_root": str(Path(child_root).resolve()),
        "parent_generation": str(parent_generation),
        "parent_content_sha256": str(parent_content_sha256),
        "child_generation": str(child_generation),
        "child_content_sha256": str(child_content_sha256),
        "fields": {str(name): dict(value) for name, value in fields.items()},
        "exact_report": dict(exact_report),
        "created_utc": _utc(),
    }


def _safe_child_path(base: Path, value: str | Path, *, label: str) -> Path:
    """Resolve a receipt path without crossing the child root."""
    raw = Path(str(value))
    if not raw.is_absolute():
        raw = base / raw
    if ".." in raw.parts:
        raise ValueError(f"{label} contains parent traversal")
    try:
        resolved = raw.resolve(strict=False)
        resolved.relative_to(base)
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} escapes child root") from error
    current = base
    try:
        relative = raw.relative_to(base)
    except ValueError as error:  # pragma: no cover - guarded above
        raise ValueError(f"{label} escapes child root") from error
    for part in relative.parts:
        current = current / part
        try:
            attributes = int(getattr(current.lstat(), "st_file_attributes", 0))
            if current.is_symlink() or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                raise ValueError(f"{label} contains a symlink or reparse point")
        except FileNotFoundError:
            # The caller will produce the more useful missing-artifact error.
            break
        except OSError as error:
            raise ValueError(f"{label} cannot be inspected") from error
    return resolved


def _content_hash(records: Sequence[Mapping[str, Any]]) -> str:
    source = json.dumps(list(records), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def _validate_child_current_payload(
    base: Path, manifest: Mapping[str, Any], receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-read every child CURRENT screen and all three slow fields.

    READY is a restart gate, so hashes stored in a receipt are only a claim.
    This function verifies the actual NPZ payload, metadata, shape, dtype and
    finite values before allowing the receipt to be used.
    """
    expected_shape = tuple(int(value) for value in manifest.get("shape", ()))
    expected_dtype = np.dtype(str(manifest.get("dtype", "")))
    records = manifest.get("records")
    count = int(manifest.get("expected_screen_count", -1))
    fields = receipt.get("fields")
    if len(expected_shape) != 2 or expected_dtype != np.dtype(np.float64):
        raise ValueError("successor CURRENT layout is invalid")
    if not isinstance(records, list) or len(records) != count or count <= 0:
        raise ValueError("successor CURRENT record count is invalid")
    if not isinstance(fields, Mapping) or len(fields) != count * len(("delta_n", "vx", "vy")):
        raise ValueError("successor READY field inventory is incomplete")
    expected_field_keys = {f"{index}:{name}" for index in range(count) for name in ("delta_n", "vx", "vy")}
    if set(str(key) for key in fields) != expected_field_keys:
        raise ValueError("successor READY field inventory does not cover all screens")
    if _content_hash([record.get("current") for record in records]) != str(manifest.get("current_content_sha256", "")):
        raise ValueError("successor CURRENT content provenance is invalid")
    rows = []
    for ordinal, record in enumerate(records):
        if int(record.get("ordinal", -1)) != ordinal:
            raise ValueError("successor CURRENT record order is invalid")
        current_entry = record.get("current")
        if not isinstance(current_entry, Mapping):
            raise ValueError(f"successor CURRENT entry is missing: {ordinal}")
        path = _safe_child_path(base, current_entry.get("artifact", ""), label=f"screen {ordinal} artifact")
        if not path.is_file():
            raise ValueError(f"successor CURRENT artifact is missing: {ordinal}")
        if sha256_file(path) != str(current_entry.get("file_sha256", "")):
            raise ValueError(f"successor CURRENT artifact file hash mismatch: {ordinal}")
        try:
            with np.load(path, allow_pickle=False) as loaded:
                if set(loaded.files) != {"delta_n", "vx", "vy", "metadata_json"}:
                    raise ValueError(f"successor CURRENT artifact object inventory is invalid: {ordinal}")
                metadata = json.loads(str(loaded["metadata_json"].item()))
                if (
                    metadata.get("schema") != "khz_filament.hr4e5s.streaming.v1"
                    or metadata.get("namespace") != "CURRENT"
                    or int(metadata.get("ordinal", -1)) != ordinal
                    or str(metadata.get("screen_id")) != str(record.get("screen_id"))
                    or metadata.get("generation") != manifest.get("current_generation")
                    or metadata.get("z_m") != record.get("z_m")
                    or metadata.get("shape") != list(expected_shape)
                    or metadata.get("dtype") != "float64"
                ):
                    raise ValueError(f"successor CURRENT artifact metadata is invalid: {ordinal}")
                manifest_hashes = current_entry.get("field_sha256")
                if not isinstance(manifest_hashes, Mapping):
                    raise ValueError(f"successor CURRENT field hashes are missing: {ordinal}")
                metadata_hashes = metadata.get("field_sha256")
                if not isinstance(metadata_hashes, Mapping):
                    raise ValueError(f"successor CURRENT metadata hashes are missing: {ordinal}")
                for name in ("delta_n", "vx", "vy"):
                    array = np.asarray(loaded[name])
                    if array.shape != expected_shape or array.dtype != expected_dtype:
                        raise ValueError(f"successor CURRENT field layout is invalid: {ordinal}:{name}")
                    if not bool(np.all(np.isfinite(array))):
                        raise ValueError(f"successor CURRENT field is non-finite: {ordinal}:{name}")
                    digest = sha256_array(array)
                    identity = fields.get(f"{ordinal}:{name}")
                    if not isinstance(identity, Mapping):
                        raise ValueError(f"successor READY field identity is missing: {ordinal}:{name}")
                    if (
                        digest != str(manifest_hashes.get(name, ""))
                        or digest != str(metadata_hashes.get(name, ""))
                        or digest != str(identity.get("sha256_array", ""))
                        or identity.get("shape") != list(expected_shape)
                        or str(identity.get("dtype")) != expected_dtype.name
                    ):
                        raise ValueError(f"successor CURRENT field identity mismatch: {ordinal}:{name}")
                    rows.append({"ordinal": ordinal, "field": name, "sha256_array": digest, "shape": list(array.shape), "dtype": array.dtype.name, "finite": True})
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            if isinstance(error, ValueError) and str(error).startswith("successor"):
                raise
            raise ValueError(f"successor CURRENT artifact cannot be read: {ordinal}") from error
    return {"status": "PASS", "screen_count": count, "field_count": len(rows), "fields": rows}


def validate_ready_receipt(
    root: str | Path, *, expected_parent_root: str | Path | None = None,
    expected_parent_generation: str | None = None,
) -> dict[str, Any]:
    """Validate a self-contained successor READY receipt and its report."""
    base = Path(root).resolve()
    receipt_path = base / "E5_1A_READY.json"
    if not receipt_path.is_file():
        raise ValueError("successor READY receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != READY_SCHEMA or receipt.get("status") != "READY":
        raise ValueError("successor READY receipt is invalid")
    if Path(str(receipt.get("child_root", ""))).resolve() != base:
        raise ValueError("successor READY child root mismatch")
    if expected_parent_root is not None and Path(str(receipt.get("parent_root", ""))).resolve() != Path(expected_parent_root).resolve():
        raise ValueError("successor READY parent root mismatch")
    if expected_parent_generation is not None and receipt.get("parent_generation") != str(expected_parent_generation):
        raise ValueError("successor READY parent generation mismatch")
    lineage_path = _safe_child_path(base, 'E5_1A_LINEAGE.json', label='lineage')
    lineage = json.loads(lineage_path.read_text(encoding='utf-8'))
    if lineage.get('schema') != 'khz_filament.hr4e5.e5_1a.formal_entry.v1' or lineage.get('status') != 'READY':
        raise ValueError('successor lineage schema or status is invalid')
    if lineage.get('parent_root') != receipt.get('parent_root') or lineage.get('child_root') != str(base):
        raise ValueError('successor lineage conflicts with READY')
    current_manifest = _safe_child_path(base, "streaming_manifest.json", label="successor streaming manifest")
    if not current_manifest.is_file():
        raise ValueError("successor streaming manifest is missing")
    manifest = json.loads(current_manifest.read_text(encoding="utf-8"))
    if manifest.get('block_size') != 8 or manifest.get('queue_depth') != 16 or manifest.get('recovery_active', False):
        raise ValueError('successor READY requires fixed block8/queue16 and no active recovery')
    if manifest.get("current_generation") != receipt.get("child_generation") or manifest.get("current_content_sha256") != receipt.get("child_content_sha256"):
        raise ValueError("successor READY current identity mismatch")
    if manifest.get("next_generation") == manifest.get("current_generation"):
        raise ValueError("successor generation identity is invalid")
    if manifest.get("queue") or manifest.get("recovery_backlog") or manifest.get("barrier") is not None or manifest.get("promotion") is not None:
        raise ValueError("successor READY root is not a fresh CURRENT root")
    child_payload = _validate_child_current_payload(base, manifest, receipt)
    report_info = receipt.get("exact_report")
    if not isinstance(report_info, Mapping):
        raise ValueError("successor READY exact report metadata is missing")
    report_path = _safe_child_path(base, str(report_info.get("path", "")), label="successor READY exact report")
    if not report_path.is_file() or sha256_file(report_path) != str(report_info.get("sha256", "")):
        raise ValueError("successor READY exact report is missing or changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or int(report.get("compared_object_count", -1)) != child_payload["field_count"] or int(report.get("mismatch_count", -1)) != 0:
        raise ValueError("successor READY exact report is not PASS")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != child_payload["field_count"] or any(row.get("status") != "PASS" for row in rows):
        raise ValueError("successor READY exact report rows are incomplete")
    return {**dict(receipt), "validated_child_payload": child_payload}


__all__ = [
    "EVIDENCE_SCHEMA", "READY_SCHEMA", "atomic_json", "compare_array_exact",
    "compare_arrays_exact", "compare_exact_objects", "compare_object_sets",
    "lineage_binding", "object_manifest", "sha256_array", "sha256_file",
    "validate_ready_receipt", "write_exact_report",
]
