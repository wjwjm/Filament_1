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
PAIRED_EXACT_SCHEMA = "khz_filament.hr4e5.e5_1a.paired_exact.v1"
_PAIR_FIELDS = ("delta_n", "vx", "vy")
_PAIR_SINKS = ("ion", "ib", "raman", "qthermal", "increment", "state_after")
_PAIR_LEDGERS = (
    "E_dep_ion_interval_J", "E_dep_ib_interval_J", "E_dep_raman_interval_J",
    "E_dep_plasma_interval_J", "E_thermal_interval_J", "delta_n_increment_min",
    "delta_n_increment_onaxis", "delta_n_state_min_after_update",
    "delta_n_state_onaxis_after_update",
)


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


def _file_identity(path: Path) -> dict[str, Any]:
    stat_value = path.stat()
    return {
        "st_dev": int(getattr(stat_value, "st_dev", -1)),
        "st_ino": int(getattr(stat_value, "st_ino", -1)),
        "st_nlink": int(getattr(stat_value, "st_nlink", 1)),
        "size": int(stat_value.st_size),
        "mtime_ns": int(getattr(stat_value, "st_mtime_ns", 0)),
    }


def _load_locator(value: Any) -> tuple[np.ndarray, Path | None, dict[str, Any]]:
    """Load an object plus a precise NPZ member/slice locator."""
    locator: dict[str, Any] = {}
    source: Path | None = None
    target = value
    if isinstance(value, Mapping):
        target = value.get("path", value.get("file", value.get("value")))
        if target is None and "array" in value:
            target = value["array"]
        if value.get("member") is not None or value.get("npz_member") is not None:
            locator["member"] = str(value.get("member", value.get("npz_member")))
        if value.get("slice_locator") is not None:
            locator["slice"] = _json_safe(value["slice_locator"])
        elif value.get("slice") is not None:
            locator["slice"] = _json_safe(value["slice"])
    if isinstance(target, (str, Path)):
        source = Path(target).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() == ".npz":
            with np.load(source, allow_pickle=False) as loaded:
                member = locator.get("member")
                if member is None:
                    if len(loaded.files) != 1:
                        raise ValueError(f"NPZ object requires an explicit member locator: {source}")
                    member = loaded.files[0]
                    locator["member"] = str(member)
                if str(member) not in loaded.files:
                    raise ValueError(f"NPZ member locator is missing: {member}")
                array = np.asarray(loaded[str(member)])
        else:
            array = np.asarray(np.load(source, mmap_mode="r", allow_pickle=False))
    else:
        array = _as_array(target, name="object")
    if "slice" in locator:
        selector = locator["slice"]
        if not isinstance(selector, (list, tuple)):
            raise ValueError("slice locator must be a list")
        try:
            array = np.asarray(array[tuple(selector)])
        except Exception as error:
            raise ValueError("slice locator is invalid") from error
    return array, source, locator


def build_expected_object_set(
    objects: Mapping[str, Any], *, campaign_id: str, trajectory: str, pulse: int,
    attempt: int, namespace: str, generation: str | None = None,
    root: str | Path | None = None, source_indices: Mapping[str, int] | None = None,
    role: str = "scientific_array",
) -> list[dict[str, Any]]:
    """Derive a durable object set from actual files and precise locators."""
    base = None if root is None else Path(root).resolve()
    rows: list[dict[str, Any]] = []
    for name in sorted(objects):
        array, source, locator = _load_locator(objects[name])
        if source is None:
            raise ValueError("formal object sets require file-backed arrays")
        if base is None:
            relative = str(source)
        else:
            try:
                relative = source.relative_to(base).as_posix()
            except ValueError as error:
                raise ValueError(f"object file escapes declared root: {source}") from error
        finite = bool(np.all(np.isfinite(array)))
        if not finite:
            raise ValueError(f"object is non-finite: {name}")
        source_index = int(source_indices[name]) if source_indices and name in source_indices else None
        row = {
            "schema": EVIDENCE_SCHEMA, "status": "PASS", "campaign_id": str(campaign_id),
            "trajectory": str(trajectory), "pulse": int(pulse), "attempt": int(attempt),
            "generation": None if generation is None else str(generation),
            "namespace": str(namespace), "source_index": source_index, "name": str(name),
            "role": str(role), "relative_path": relative, "locator": locator,
            "shape": list(array.shape), "dtype": array.dtype.name, "finite": True,
            "canonical_array_hash": sha256_array(array), "sha256_array": sha256_array(array),
            "file_hash": sha256_file(source), "file_sha256": sha256_file(source),
            "file_identity": _file_identity(source),
            "creation_record": {"scope": "derived_object_set", "intent_id": None},
            "comparison_row_id": f"{namespace}:{source_index}:{name}",
        }
        rows.append(row)
    if not rows:
        raise ValueError("expected object set is empty")
    return rows


def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("campaign_id"), row.get("trajectory"), row.get("pulse"), row.get("attempt"),
        row.get("generation"), row.get("namespace"), row.get("source_index"), row.get("name"),
        row.get("relative_path"), json.dumps(row.get("locator", {}), sort_keys=True, separators=(",", ":")),
    )


def validate_object_set(
    rows: Sequence[Mapping[str, Any]], *, expected_rows: Sequence[Mapping[str, Any]] | None = None,
    campaign_id: str | None = None, trajectory: str | None = None, pulse: int | None = None,
    attempt: int | None = None, root: str | Path | None = None,
) -> dict[str, Any]:
    """Re-read object files and reject missing, duplicate, replaced or mislocated rows."""
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("object set rows are missing")
    base = None if root is None else Path(root).resolve()
    seen: set[tuple[Any, ...]] = set()
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("object row is invalid")
        row = dict(raw)
        key = _row_key(row)
        if key in seen:
            raise ValueError("duplicate object-set row")
        seen.add(key)
        if row.get("status") not in (None, "PASS"):
            raise ValueError("object row is not PASS")
        for field in ("campaign_id", "trajectory", "pulse", "attempt", "namespace", "source_index", "name", "relative_path", "locator", "shape", "dtype", "finite", "canonical_array_hash", "file_identity", "creation_record", "comparison_row_id"):
            if field not in row:
                raise ValueError(f"object row is missing {field}")
        if campaign_id is not None and str(row["campaign_id"]) != str(campaign_id):
            raise ValueError("object-set campaign mismatch")
        if trajectory is not None and str(row["trajectory"]) != str(trajectory):
            raise ValueError("object-set trajectory mismatch")
        if pulse is not None and int(row["pulse"]) != int(pulse):
            raise ValueError("object-set pulse mismatch")
        if attempt is not None and int(row["attempt"]) != int(attempt):
            raise ValueError("object-set attempt mismatch")
        if row.get("finite") is not True:
            raise ValueError("object-set row is not finite")
        if not isinstance(row.get("creation_record"), Mapping):
            raise ValueError("object row creation record is missing")
        path_value = Path(str(row["relative_path"]))
        if path_value.is_absolute() or ".." in path_value.parts:
            raise ValueError("object row locator escapes root")
        path = path_value if base is None else base / path_value
        if not path.is_file() or path.is_symlink():
            raise ValueError("object file is missing or linked")
        expected_identity = row.get("file_identity")
        actual_identity = _file_identity(path)
        if dict(expected_identity) != actual_identity:
            raise ValueError("object file identity changed")
        digest = sha256_file(path)
        if digest != str(row.get("file_hash", row.get("file_sha256", ""))):
            raise ValueError("object file hash changed")
        loaded, loaded_path, locator = _load_locator({"path": path, **dict(row.get("locator", {}))})
        if loaded_path is None or dict(locator) != dict(row.get("locator", {})):
            raise ValueError("object locator is incomplete")
        if list(loaded.shape) != list(row["shape"]) or loaded.dtype.name != str(row["dtype"]):
            raise ValueError("object shape or dtype changed")
        if not bool(np.all(np.isfinite(loaded))):
            raise ValueError("object became non-finite")
        canonical = sha256_array(loaded)
        if canonical != str(row["canonical_array_hash"]):
            raise ValueError("object canonical array hash changed")
        normalized.append(row)
    if expected_rows is not None:
        expected_keys = {_row_key(dict(item)) for item in expected_rows}
        if seen != expected_keys:
            raise ValueError("object set does not match expected object collection")
    binding_source = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    binding_hash = hashlib.sha256(binding_source).hexdigest()
    return {"schema": "khz_filament.hr4e5.e5_1a.object_set_binding.v1", "status": "PASS",
            "campaign_id": None if campaign_id is None else str(campaign_id),
            "object_count": len(normalized), "binding_sha256": binding_hash, "rows": normalized}


def bind_exact_report(
    path: str | Path, report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *,
    campaign_id: str, root: str | Path | None = None,
) -> dict[str, Any]:
    """Attach a validated object-set binding to an exact report."""
    binding = validate_object_set(rows, campaign_id=campaign_id, root=root)
    payload = {**dict(report), "object_set_binding": {key: value for key, value in binding.items() if key != "rows"}, "rows": [dict(row) for row in rows]}
    payload["object_set_binding"]["binding_sha256"] = binding["binding_sha256"]
    return write_exact_report(path, payload)


def _paired_exact_keys(*, k: int, n_pulses: int, pulse: int) -> set[str]:
    if k <= 0 or n_pulses <= 0 or pulse < 0 or pulse >= n_pulses:
        raise ValueError("formal exact contract dimensions are invalid")
    namespaces = ("pre", "post") if pulse == n_pulses - 1 else ("pre", "post", "next")
    keys = {
        f"screen:{namespace}:{index}:{field}"
        for namespace in namespaces for index in range(k) for field in _PAIR_FIELDS
    }
    keys.update(f"sink:{name}:{index}" for name in _PAIR_SINKS for index in range(k))
    keys.update(f"ledger:{name}" for name in _PAIR_LEDGERS)
    keys.add("optical:final:final_optical_field")
    return keys


def _paired_descriptor_path(
    root: Path, descriptor: Mapping[str, Any], *, label: str,
    require_file: bool = True,
) -> Path:
    raw = descriptor.get("path", descriptor.get("relative_path"))
    if raw is None:
        raise ValueError(f"{label} path is missing")
    candidate = Path(str(raw))
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root) or resolved.is_symlink() or (require_file and not resolved.is_file()):
        raise ValueError(f"{label} path is missing or outside exact root")
    return resolved


def _validate_paired_descriptor(
    descriptor: Mapping[str, Any], *, root: Path, side: str, campaign_id: str,
    pulse: int, attempt: int, storage_budget: Any | None,
    metadata_only: bool = False,
) -> tuple[np.ndarray | None, Path, dict[str, Any]]:
    if str(descriptor.get("trajectory", "")) != side:
        raise ValueError(f"formal exact {side} trajectory binding is invalid")
    if str(descriptor.get("campaign_id", "")) != str(campaign_id):
        raise ValueError(f"formal exact {side} campaign binding is invalid")
    if int(descriptor.get("pulse", -1)) != int(pulse) or int(descriptor.get("attempt", -1)) != int(attempt):
        raise ValueError(f"formal exact {side} pulse/attempt binding is invalid")
    creation = descriptor.get("creation_record")
    if not isinstance(creation, Mapping) or not str(creation.get("intent_id", "")) or not str(creation.get("generation", "")):
        raise ValueError(f"formal exact {side} creation record is incomplete")
    path = _paired_descriptor_path(
        root, descriptor, label=f"formal exact {side}", require_file=not metadata_only,
    )
    if str(descriptor.get("relative_path", "")).replace("\\", "/") != path.relative_to(root).as_posix():
        raise ValueError(f"formal exact {side} relative locator changed")
    locator = descriptor.get("locator", {})
    if not isinstance(locator, Mapping):
        raise ValueError(f"formal exact {side} locator is invalid")
    if "member" in locator and not str(locator.get("member", "")):
        raise ValueError(f"formal exact {side} locator member is empty")
    if "slice" in locator and not isinstance(locator.get("slice"), (list, tuple)):
        raise ValueError(f"formal exact {side} slice locator is invalid")
    present = path.is_file() and not path.is_symlink()
    array: np.ndarray | None = None
    if present:
        array, loaded_path, actual_locator = _load_locator({"path": path, **dict(locator)})
        if loaded_path is None or dict(actual_locator) != dict(locator):
            raise ValueError(f"formal exact {side} locator is incomplete")
        if descriptor.get("shape") is None or list(descriptor.get("shape")) != list(array.shape):
            raise ValueError(f"formal exact {side} shape changed")
        if str(descriptor.get("dtype", "")) != array.dtype.name or descriptor.get("finite") is not True:
            raise ValueError(f"formal exact {side} dtype/finite contract changed")
        if not bool(np.all(np.isfinite(array))):
            raise ValueError(f"formal exact {side} object became non-finite")
        file_digest = sha256_file(path)
        if str(descriptor.get("file_sha256", descriptor.get("file_hash", ""))) != file_digest:
            raise ValueError(f"formal exact {side} file hash changed")
        if dict(descriptor.get("file_identity", {})) != _file_identity(path):
            raise ValueError(f"formal exact {side} file identity changed")
        array_digest = sha256_array(array)
        if str(descriptor.get("canonical_array_hash", descriptor.get("sha256_array", ""))) != array_digest:
            raise ValueError(f"formal exact {side} canonical array hash changed")
    elif not metadata_only:
        raise ValueError(f"formal exact {side} path is missing or outside exact root")
    else:
        # After an authorized GC, the immutable descriptor remains the only
        # available object evidence.  Validate its shape/dtype/hash claims,
        # but do not attempt to reopen the deleted array.
        shape = descriptor.get("shape")
        if (not isinstance(shape, (list, tuple)) or not shape
                or any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) or int(item) < 0 for item in shape)):
            raise ValueError(f"formal exact {side} metadata shape is missing")
        try:
            dtype = np.dtype(str(descriptor.get("dtype", "")))
        except (TypeError, ValueError) as error:
            raise ValueError(f"formal exact {side} metadata dtype is invalid") from error
        if descriptor.get("finite") is not True:
            raise ValueError(f"formal exact {side} metadata finite contract changed")
        file_digest = str(descriptor.get("file_sha256", descriptor.get("file_hash", "")))
        array_digest = str(descriptor.get("canonical_array_hash", descriptor.get("sha256_array", "")))
        identity = descriptor.get("file_identity")
        if not file_digest or not array_digest or not isinstance(identity, Mapping):
            raise ValueError(f"formal exact {side} metadata hashes or identity are missing")
        if str(dtype.name) != str(descriptor.get("dtype")):
            raise ValueError(f"formal exact {side} metadata dtype is invalid")
    if storage_budget is not None:
        try:
            relative = path.relative_to(Path(storage_budget.root).resolve()).as_posix()
        except ValueError:
            raise ValueError(f"formal exact {side} path escapes storage root")
        artifact = storage_budget.artifacts().get(relative)
        if not isinstance(artifact, Mapping):
            raise ValueError(f"formal exact {side} object has no storage ownership record")
        if str(artifact.get("campaign_id", "")) != str(campaign_id):
            raise ValueError(f"formal exact {side} storage ownership campaign changed")
        intent_id = str(creation.get("intent_id"))
        if str(artifact.get("intent_id", "")) != intent_id:
            raise ValueError(f"formal exact {side} storage ownership intent changed")
        if str(artifact.get("sha256", artifact.get("expected_sha256", ""))) != file_digest:
            raise ValueError(f"formal exact {side} storage ownership file hash changed")
        if dict(artifact.get("identity", {})) != dict(descriptor.get("file_identity", {})):
            raise ValueError(f"formal exact {side} storage ownership file identity changed")
        try:
            completed_intent = storage_budget.validate_intent(
                intent_id, path=path, require_completed=True,
            )
        except Exception as error:
            raise ValueError(f"formal exact {side} creation intent is not durably complete") from error
        artifact_creation = artifact.get("creation_record")
        if not isinstance(artifact_creation, Mapping):
            raise ValueError(f"formal exact {side} artifact creation record is incomplete")
        if str(artifact_creation.get("generation", "")) != str(creation.get("generation")):
            raise ValueError(f"formal exact {side} storage ownership generation changed")
        if str(completed_intent.get("generation", "")) != str(creation.get("generation")):
            raise ValueError(f"formal exact {side} completed intent generation changed")
        if str(completed_intent.get("admission_hash", "")) != str(storage_budget.admission_hash or ""):
            raise ValueError(f"formal exact {side} completed intent admission changed")
        for field in ("trajectory", "pulse", "attempt"):
            if str(artifact.get(field)) != str(descriptor.get(field)):
                raise ValueError(f"formal exact {side} storage ownership {field} changed")
            if str(completed_intent.get(field)) != str(descriptor.get(field)):
                raise ValueError(f"formal exact {side} completed intent {field} changed")
    return array, path, {"array_sha256": array_digest, "file_sha256": file_digest,
                          "objects_present": bool(present)}


def validate_paired_exact_report(
    report_path: str | Path, *, admission_identity: Mapping[str, Any], campaign_id: str,
    pulse: int, attempt: int = 0, root: str | Path, storage_budget: Any | None = None,
    expected_sha256: str | None = None, metadata_only: bool = False,
) -> dict[str, Any]:
    """Reload and compare the complete formal R/C object contract.

    A formal exact receipt is a paired object inventory, not a callback status.
    Every expected screen, sink, ledger and final-optical key must occur once;
    each side is reloaded from its locator and compared elementwise.  In
    ``metadata_only`` mode an authorized GC may have removed both arrays: the
    immutable descriptors, hashes, and creation ownership are still checked;
    any objects that remain on disk continue through the elementwise path.
    """
    if storage_budget is None:
        raise ValueError("formal paired exact validation requires StorageBudget ownership")
    path = Path(report_path).resolve()
    if not path.is_file() or (expected_sha256 is not None and sha256_file(path) != str(expected_sha256)):
        raise ValueError("formal exact report is missing or changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != PAIRED_EXACT_SCHEMA or payload.get("status") != "PASS":
        raise ValueError("formal exact report schema or status is invalid")
    identity_hash = str(admission_identity.get("identity_sha256", ""))
    if str(payload.get("admission_identity_sha256", "")) != identity_hash:
        raise ValueError("formal exact admission identity binding is invalid")
    if str(payload.get("campaign_id", "")) != str(campaign_id) or int(payload.get("pulse", -1)) != int(pulse):
        raise ValueError("formal exact campaign or pulse binding is invalid")
    if int(payload.get("attempt", -1)) != int(attempt):
        raise ValueError("formal exact attempt binding is invalid")
    k = int(admission_identity.get("k", -1))
    n_pulses = int(admission_identity.get("n_pulses", -1))
    expected_keys = _paired_exact_keys(k=k, n_pulses=n_pulses, pulse=int(pulse))
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected_keys):
        raise ValueError("formal exact object row count is incomplete")
    seen: set[str] = set()
    compared: list[dict[str, Any]] = []
    base = Path(root).resolve()
    for raw in rows:
        if not isinstance(raw, Mapping) or raw.get("status") != "PASS":
            raise ValueError("formal exact object row is invalid")
        key = str(raw.get("comparison_key", ""))
        if not key or key in seen:
            raise ValueError("formal exact object key is missing or duplicated")
        seen.add(key)
        if key not in expected_keys:
            raise ValueError(f"formal exact object key is not in the contract: {key}")
        reference = raw.get("reference")
        candidate = raw.get("candidate")
        if not isinstance(reference, Mapping) or not isinstance(candidate, Mapping):
            raise ValueError(f"formal exact object pair is incomplete: {key}")
        key_parts = key.split(":")
        if key_parts[0] == "screen" and len(key_parts) == 4:
            expected = {"role": "screen", "namespace": key_parts[1], "source_index": int(key_parts[2]), "name": key_parts[3]}
        elif key_parts[0] == "sink" and len(key_parts) == 3:
            expected = {"role": "sink", "namespace": "sink", "source_index": int(key_parts[2]), "name": key_parts[1]}
        elif key_parts[0] == "ledger" and len(key_parts) == 2:
            expected = {"role": "ledger", "namespace": "ledger", "source_index": None, "name": key_parts[1]}
        elif key_parts == ["optical", "final", "final_optical_field"]:
            expected = {"role": "final_optical", "namespace": "final", "source_index": None, "name": "final_optical_field"}
        else:  # pragma: no cover - expected keys are generated above
            raise ValueError(f"formal exact object key has invalid locator contract: {key}")
        for side_descriptor in (reference, candidate):
            if any(side_descriptor.get(field) != value for field, value in expected.items()):
                raise ValueError(f"formal exact object locator does not match contract: {key}")
        left, left_path, left_info = _validate_paired_descriptor(
            reference, root=base, side="R", campaign_id=campaign_id, pulse=int(pulse),
            attempt=int(attempt), storage_budget=storage_budget, metadata_only=metadata_only,
        )
        right, right_path, right_info = _validate_paired_descriptor(
            candidate, root=base, side="C", campaign_id=campaign_id, pulse=int(pulse),
            attempt=int(attempt), storage_budget=storage_budget, metadata_only=metadata_only,
        )
        if left_info["array_sha256"] != right_info["array_sha256"]:
            raise ValueError(f"formal exact paired array hash mismatch: {key}")
        if left is None or right is None:
            if left is not None or right is not None:
                raise ValueError(f"formal exact paired object retention differs: {key}")
            result = {
                "schema": EVIDENCE_SCHEMA, "name": key, "status": "PASS",
                "metadata_only": True, "objects_present": False,
                "reference_path": str(left_path), "candidate_path": str(right_path),
            }
        else:
            result = compare_arrays_exact(left, right, name=key, reference_path=left_path, candidate_path=right_path)
            if result.get("status") != "PASS":
                raise ValueError(f"formal exact paired array mismatch: {key}")
        if raw.get("reference_sha256_array") != left_info["array_sha256"]:
            raise ValueError(f"formal exact reference comparison hash changed: {key}")
        if raw.get("candidate_sha256_array") != right_info["array_sha256"]:
            raise ValueError(f"formal exact candidate comparison hash changed: {key}")
        compared.append({**dict(raw), **result})
    if seen != expected_keys:
        raise ValueError("formal exact object keys are missing or unexpected")
    expected_screen_count = (12 if int(pulse) == n_pulses - 1 else 15) * k
    if int(payload.get("screen_count", -1)) != expected_screen_count:
        raise ValueError("formal exact screen count does not match pulse contract")
    if int(payload.get("ledger_count", -1)) != len(_PAIR_LEDGERS) or int(payload.get("optical_count", -1)) != 1:
        raise ValueError("formal exact ledger or optical count does not match contract")
    if int(payload.get("expected_object_count", -1)) != len(expected_keys) or int(payload.get("compared_object_count", -1)) != len(expected_keys) or int(payload.get("mismatch_count", -1)) != 0:
        raise ValueError("formal exact summary counts are incomplete")
    return {**payload, "rows": compared, "report_path": str(path), "report_sha256": sha256_file(path),
            "validated_object_count": len(compared), "expected_object_count": len(expected_keys),
            "metadata_only": bool(metadata_only),
            "objects_validated": not bool(metadata_only) or all(not row.get("metadata_only", False) for row in compared)}


def validate_paired_exact_report_metadata(
    report_path: str | Path, *, admission_identity: Mapping[str, Any], campaign_id: str,
    pulse: int, attempt: int = 0, root: str | Path, storage_budget: Any | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a durable paired exact report after an authorized object GC."""
    return validate_paired_exact_report(
        report_path, admission_identity=admission_identity, campaign_id=campaign_id,
        pulse=pulse, attempt=attempt, root=root, storage_budget=storage_budget,
        expected_sha256=expected_sha256, metadata_only=True,
    )


def validate_durable_report(
    report_path: str | Path, *, expected_sha256: str | None = None,
    campaign_id: str | None = None, trajectory: str | None = None,
    pulse: int | None = None, attempt: int | None = None, root: str | Path | None = None,
    expected_rows: Sequence[Mapping[str, Any]] | None = None,
    validate_objects: bool = True,
) -> dict[str, Any]:
    path = Path(report_path).resolve()
    if not path.is_file():
        raise ValueError("durable report is missing")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != str(expected_sha256):
        raise ValueError("durable report hash changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise ValueError("durable report is not PASS")
    rows = payload.get("rows")
    if isinstance(payload.get("object_set_binding"), Mapping):
        if validate_objects:
            binding = validate_object_set(rows, expected_rows=expected_rows, campaign_id=campaign_id,
                                         trajectory=trajectory, pulse=pulse, attempt=attempt, root=root)
        else:
            if not isinstance(rows, Sequence) or not rows:
                raise ValueError("durable report object rows are missing")
            if any(not isinstance(row, Mapping) for row in rows):
                raise ValueError("durable report object rows are invalid")
            if campaign_id is not None and any(str(row.get("campaign_id")) != str(campaign_id) for row in rows):
                raise ValueError("durable report object-set campaign mismatch")
            if trajectory is not None and any(str(row.get("trajectory")) != str(trajectory) for row in rows):
                raise ValueError("durable report object-set trajectory mismatch")
            if pulse is not None and any(int(row.get("pulse", -1)) != int(pulse) for row in rows):
                raise ValueError("durable report object-set pulse mismatch")
            normalized = [dict(row) for row in rows]
            source = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            binding = {"status": "PASS", "campaign_id": campaign_id, "object_count": len(normalized),
                       "binding_sha256": hashlib.sha256(source).hexdigest(), "rows": normalized,
                       "objects_validated": False}
        saved_binding = payload["object_set_binding"].get("binding_sha256")
        if str(saved_binding) != binding["binding_sha256"]:
            raise ValueError("durable report object-set binding hash changed")
        payload["validated_object_set"] = binding
    elif expected_rows is not None or campaign_id is not None:
        raise ValueError("durable report lacks object-set binding")
    return {**payload, "report_path": str(path), "report_sha256": digest}


derive_expected_object_set = build_expected_object_set
validate_evidence_report = validate_durable_report


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
    "EVIDENCE_SCHEMA", "READY_SCHEMA", "PAIRED_EXACT_SCHEMA", "atomic_json", "compare_array_exact",
    "compare_arrays_exact", "compare_exact_objects", "compare_object_sets",
    "bind_exact_report", "build_expected_object_set", "derive_expected_object_set", "lineage_binding", "object_manifest",
    "sha256_array", "sha256_file", "validate_durable_report", "validate_object_set", "validate_paired_exact_report",
    "validate_paired_exact_report_metadata",
    "validate_evidence_report",
    "validate_ready_receipt", "write_exact_report",
]
