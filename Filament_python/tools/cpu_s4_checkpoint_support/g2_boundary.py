"""Private source-bound first-boundary capture for CPU-S4-3G2.

The module is report-local and has no dependency on the production
propagation modules. A generated G2 propagator calls
:func:`explicit_boundary_capture` with explicit names at the frozen
first-boundary statement. The collector copies the six science/post arrays
and the 99 list diagnostics into an isolated ``.npy`` snapshot, records the
seven live scalar values, and drops all live references before returning.
The snapshot is evidence for a later full-element comparison; it never
participates in the numerical operation or qualification decision.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "filament.cpu_s4_3g2.boundary_lease.v2"
BOUNDARY_ASSIGNMENT = "pe = int(getattr(p, 'progress_every_z', 0) or 0)"
POST_FIELDS = ("delta_n", "vx", "vy")
DYNAMIC_SCALARS = (
    "z",
    "interval_index",
    "dz_try",
    "U0_baseline",
    "U_now",
    "adaptive_rejection_count",
    "safety_mode_trigger_count",
)
SCIENCE_ARRAY_FIELDS = ("E", "Qacc", "Qacc_raman")
EXPECTED_DIAGNOSTIC_COUNT = 99
EXPECTED_TOTAL_FIELD_COUNT = 112
MAPPING_FILENAME = "HISTORICAL_BOUNDARY_FIELD_MAP_112.csv"
BOUNDARY_MAPPING_IDENTITY = {
    "hash_scope": "raw_bytes",
    "sha256": "26d7de0ab58861d083ffa0e1ab0e06edc29c00db8dd184b102651c2a246f5c90",
    "size_bytes": 21160,
}


class BoundaryBindingError(RuntimeError):
    """The source-bound boundary contract is unavailable or violated."""


class BoundaryLeaseError(RuntimeError):
    """A lease was used outside its bound source/thread/state."""


class BoundaryCaptureError(RuntimeError):
    """The live boundary payload does not satisfy the frozen schema."""


def _require(condition: bool, reason: str, detail: Any = None) -> None:
    if not condition:
        suffix = "" if detail is None else f":{detail!r}"
        raise BoundaryBindingError(f"{reason}{suffix}")


def _safe_scalar(value: Any) -> Any:
    """Convert one scalar to JSON-safe data without accepting containers."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - target environments include NumPy
        np = None
    if np is not None and isinstance(value, np.generic):
        return _safe_scalar(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, complex):
        return {
            "real": _safe_scalar(float(value.real)),
            "imag": _safe_scalar(float(value.imag)),
        }
    raise BoundaryCaptureError(f"BOUNDARY_NONSCALAR_VALUE:{type(value).__name__}")


def _array_layout(value: Any) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise BoundaryCaptureError("NUMPY_REQUIRED_FOR_BOUNDARY_ARRAY") from exc
    if type(value) is not np.ndarray:
        raise BoundaryCaptureError("BOUNDARY_ARRAY_NOT_NATIVE_NDARRAY")
    base = value
    depth = 0
    while isinstance(getattr(base, "base", None), np.ndarray):
        depth += 1
        base = base.base
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": value.dtype.str,
        "strides": [int(item) for item in value.strides],
        "C": bool(value.flags.c_contiguous),
        "F": bool(value.flags.f_contiguous),
        "OWNDATA": bool(value.flags.owndata),
        "WRITEABLE": bool(value.flags.writeable),
        "base_chain_depth": int(depth),
        "base_ndarray": bool(isinstance(getattr(value, "base", None), np.ndarray)),
        "pointer": int(value.__array_interface__["data"][0]),
    }


def _raw_digest(value: Any, *, label: str) -> str:
    layout = _array_layout(value)
    if not layout["C"] or any(int(stride) <= 0 for stride in value.strides):
        raise BoundaryCaptureError(f"BOUNDARY_ARRAY_NOT_POSITIVE_C:{label}")
    try:
        return hashlib.sha256(memoryview(value).cast("B")).hexdigest()
    except (TypeError, ValueError) as exc:
        raise BoundaryCaptureError(f"BOUNDARY_ARRAY_BYTES_UNAVAILABLE:{label}") from exc


def _file_sha256(path: Path) -> str:
    """Hash one preserved evidence member without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_filename(label: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("._")
    if not value:
        raise BoundaryCaptureError("BOUNDARY_EMPTY_SNAPSHOT_LABEL")
    return value


def _member_records(root: Path) -> list[dict[str, Any]]:
    """Return size/hash evidence for every preserved file below ``root``."""
    if not root.exists():
        return []
    result = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            result.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                    "sha256_raw": _file_sha256(path),
                }
            )
        except OSError as exc:
            result.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": None,
                    "sha256_raw": None,
                    "read_error": f"{type(exc).__name__}:{exc}",
                }
            )
    return result


def _write_snapshot_array(value: Any, *, label: str, directory: Path) -> dict[str, Any]:
    """Copy one native array and save it as a non-pickle NPY member."""
    import numpy as np

    source_layout = _array_layout(value)
    source_digest = _raw_digest(value, label=label)
    snapshot = np.array(value, copy=True, order="C")
    if type(snapshot) is not np.ndarray:
        raise BoundaryCaptureError(f"BOUNDARY_SNAPSHOT_NOT_NDARRAY:{label}")
    try:
        shares_memory = bool(np.shares_memory(value, snapshot))
    except (TypeError, ValueError) as exc:
        raise BoundaryCaptureError(f"BOUNDARY_SNAPSHOT_ALIAS_CHECK_FAILED:{label}") from exc
    if shares_memory:
        raise BoundaryCaptureError(f"BOUNDARY_SNAPSHOT_ALIASES_SOURCE:{label}")
    filename = _safe_filename(label) + ".npy"
    path = directory / filename
    with path.open("xb") as stream:
        np.save(stream, snapshot, allow_pickle=False)
    snapshot_digest = _raw_digest(snapshot, label=f"snapshot:{label}")
    if source_digest != snapshot_digest:
        raise BoundaryCaptureError(f"BOUNDARY_SNAPSHOT_DIGEST_DRIFT:{label}")
    result = {
        "field_name": str(label),
        "snapshot_file": filename,
        "source_layout": source_layout,
        "snapshot_layout": _array_layout(snapshot),
        "nbytes": int(value.nbytes),
        "source_sha256_raw": source_digest,
        "snapshot_sha256_raw": snapshot_digest,
        "snapshot_npy_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "snapshot_isolated": True,
        "source_snapshot_shares_memory": False,
    }
    del snapshot
    return result


def _sequence_snapshot(value: Any, *, label: str, directory: Path) -> dict[str, Any]:
    """Snapshot one frozen one-element diagnostic list without object dtype."""
    import numpy as np

    if not isinstance(value, (list, tuple)) or len(value) != 1:
        raise BoundaryCaptureError(f"BOUNDARY_DIAGNOSTIC_SEQUENCE:{label}")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise BoundaryCaptureError(f"BOUNDARY_DIAGNOSTIC_ARRAY:{label}") from exc
    if array.dtype.kind not in "biufcSU":
        raise BoundaryCaptureError(f"BOUNDARY_DIAGNOSTIC_DTYPE:{label}")
    result = _write_snapshot_array(array, label=label, directory=directory)
    del array
    return result


def _tree_array_node(record: Mapping[str, Any]) -> dict[str, Any]:
    """Describe an existing snapshot member without copying its bytes."""
    layout = dict(record["snapshot_layout"])
    # g1_exact TREE layouts intentionally exclude process-local data pointers;
    # the receipt retains the fuller native/snapshot layout including pointer.
    layout.pop("pointer", None)
    try:
        import numpy as np

        layout["dtype"] = str(np.dtype(layout["dtype"]))
    except (ImportError, TypeError, ValueError):
        raise BoundaryCaptureError("BOUNDARY_TREE_DTYPE_INVALID")
    return {
        "kind": "array",
        "file": str(record["snapshot_file"]),
        "layout": layout,
    }


def _tree_scalar_node(value: Any, safe_value: Any) -> dict[str, Any]:
    """Encode a scalar as a g1_exact-compatible numeric tree leaf when possible."""
    if isinstance(safe_value, (bool, int, float)):
        try:
            import numpy as np

            dtype = str(np.asarray(value).dtype)
        except (ImportError, TypeError, ValueError):
            dtype = "float64" if isinstance(safe_value, float) else "int64"
        return {"kind": "scalar", "dtype": dtype, "value": safe_value}
    return {"kind": "other", "value": safe_value}


def _snapshot_aliases(array_records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Record alias metadata for independent snapshot copies.

    The live/native alias contract is recorded separately in the receipt.  The
    isolated files are intentionally independent values, so this manifest
    describes the snapshot topology rather than pretending it is native.
    """
    ordered_records = sorted(array_records, key=lambda record: str(record["field_name"]))
    names = ["bundle." + str(record["field_name"]) for record in ordered_records]
    result = []
    for index, left in enumerate(names):
        for right in names[index:]:
            result.append(
                {
                    "left": left,
                    "right": right,
                    "same_object": bool(left == right),
                    "shares_memory": bool(left == right),
                }
            )
    return result


def _write_snapshot_tree(
    directory: Path,
    *,
    array_records: Iterable[Mapping[str, Any]],
    diagnostic_records: Iterable[Mapping[str, Any]],
    scalar_values: Mapping[str, Any],
    scalar_safe_values: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a numeric 112-field tree over the already-written snapshot files.

    This deliberately references the per-field NPY files instead of creating a
    second copy of the large arrays.  The tree is exact-value evidence only;
    its C/OWNDATA layout belongs to the snapshot copies and is kept separate
    from the native live layouts in the receipt.
    """
    arrays = list(array_records)
    diagnostics = list(diagnostic_records)
    if len(arrays) != len(SCIENCE_ARRAY_FIELDS) + len(POST_FIELDS):
        raise BoundaryCaptureError("BOUNDARY_TREE_ARRAY_COUNT")
    if len(diagnostics) != EXPECTED_DIAGNOSTIC_COUNT:
        raise BoundaryCaptureError("BOUNDARY_TREE_DIAGNOSTIC_COUNT")
    if len(scalar_values) != len(DYNAMIC_SCALARS):
        raise BoundaryCaptureError("BOUNDARY_TREE_SCALAR_COUNT")
    array_by_name = {str(record["field_name"]): record for record in arrays + diagnostics}
    tree = {
        "kind": "dict",
        "items": {
            "E": _tree_array_node(array_by_name["E"]),
            "Qacc": _tree_array_node(array_by_name["Qacc"]),
            "Qacc_raman": _tree_array_node(array_by_name["Qacc_raman"]),
            "POST": {
                "kind": "dict",
                "items": {
                    name: _tree_array_node(array_by_name[f"POST.{name}"])
                    for name in POST_FIELDS
                },
            },
            "diagnostics": {
                "kind": "dict",
                "items": {
                    str(record["field_name"]).split(".", 1)[1]: _tree_array_node(record)
                    for record in diagnostics
                },
            },
            "scalars": {
                "kind": "dict",
                "items": {
                    name: _tree_scalar_node(scalar_values[name], scalar_safe_values[name])
                    for name in DYNAMIC_SCALARS
                },
            },
        },
    }
    array_count = len(arrays) + len(diagnostics)
    field_count = array_count + len(scalar_values)
    if field_count != EXPECTED_TOTAL_FIELD_COUNT:
        raise BoundaryCaptureError("BOUNDARY_TREE_FIELD_COUNT")
    aliases = _snapshot_aliases(arrays + diagnostics)
    fingerprints = {
        "bundle." + str(record["field_name"]): {
            "snapshot_sha256_raw": str(record["snapshot_sha256_raw"]),
            "snapshot_npy_sha256": str(record["snapshot_npy_sha256"]),
            "layout": _tree_array_node(record)["layout"],
            "nbytes": int(record["nbytes"]),
            "hash_scope": "SNAPSHOT_ARRAY_RAW_BYTES",
        }
        for record in arrays + diagnostics
    }
    (directory / "TREE.json").write_text(
        _json_dumps(tree), encoding="utf-8", newline="\n"
    )
    (directory / "ALIASES.json").write_text(
        _json_dumps(aliases), encoding="utf-8", newline="\n"
    )
    (directory / "FINGERPRINTS.json").write_text(
        _json_dumps(fingerprints), encoding="utf-8", newline="\n"
    )
    return {
        "root": str(directory),
        "tree_file": "TREE.json",
        "aliases_file": "ALIASES.json",
        "fingerprints_file": "FINGERPRINTS.json",
        "field_count": int(field_count),
        "array_field_count": int(array_count),
        "scalar_field_count": int(len(scalar_values)),
        "layout_scope": "SNAPSHOT_COPIES_ONLY",
        "native_layouts_in_receipt": True,
        "native_aliases_in_receipt": True,
        "qualification_eligible": False,
    }


def _identity_code(code: Any) -> dict[str, Any]:
    if code is None:
        raise BoundaryBindingError("BOUNDARY_CALLER_CODE_MISSING")
    raw = bytes(getattr(code, "co_code", b""))
    return {
        "filename": str(getattr(code, "co_filename", "")),
        "name": str(getattr(code, "co_name", "")),
        "firstlineno": int(getattr(code, "co_firstlineno", -1)),
        "bytecode_sha256": hashlib.sha256(raw).hexdigest(),
        "code_object_id": int(id(code)),
    }


def load_boundary_mapping(path: str | Path | None = None) -> dict[str, Any]:
    """Read and validate the frozen 112-field mapping used by the AST hook."""
    mapping_path = (
        Path(path).resolve(strict=True)
        if path is not None
        else Path(__file__).resolve().parents[1] / "inputs" / MAPPING_FILENAME
    )
    try:
        raw = mapping_path.read_bytes()
    except OSError as exc:
        raise BoundaryBindingError("BOUNDARY_MAPPING_READ_FAILED") from exc
    actual_size = len(raw)
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if (
        actual_size != int(BOUNDARY_MAPPING_IDENTITY["size_bytes"])
        or actual_sha256 != str(BOUNDARY_MAPPING_IDENTITY["sha256"])
    ):
        raise BoundaryBindingError("BOUNDARY_MAPPING_IDENTITY_MISMATCH")
    try:
        with mapping_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise BoundaryBindingError("BOUNDARY_MAPPING_READ_FAILED") from exc
    required = {
        "field_name",
        "category",
        "live_source",
        "historical_record",
        "historical_evidence",
        "G2_runtime",
    }
    if not rows or not required.issubset(rows[0]):
        raise BoundaryBindingError("BOUNDARY_MAPPING_HEADER")
    names = [str(row["field_name"]) for row in rows]
    if len(rows) != EXPECTED_TOTAL_FIELD_COUNT or len(set(names)) != len(names):
        raise BoundaryBindingError("BOUNDARY_MAPPING_COUNT_OR_DUPLICATE")
    expected_prefix = list(SCIENCE_ARRAY_FIELDS) + [f"POST.{name}" for name in POST_FIELDS]
    if names[: len(expected_prefix)] != expected_prefix:
        raise BoundaryBindingError("BOUNDARY_MAPPING_PREFIX_ORDER")
    diagnostics = []
    for row in rows:
        field = str(row["field_name"])
        if field.startswith("diagnostics."):
            label = field.split(".", 1)[1]
            live = str(row["live_source"])
            if not label.isidentifier() or not live.isidentifier():
                raise BoundaryBindingError(f"BOUNDARY_DIAGNOSTIC_NAME:{field}:{live}")
            diagnostics.append((label, live))
    if len(diagnostics) != EXPECTED_DIAGNOSTIC_COUNT:
        raise BoundaryBindingError("BOUNDARY_DIAGNOSTIC_COUNT")
    scalars = []
    for row in rows:
        field = str(row["field_name"])
        if field.startswith("scalars."):
            label = field.split(".", 1)[1]
            live = str(row["live_source"])
            if label not in DYNAMIC_SCALARS or label != live:
                raise BoundaryBindingError(f"BOUNDARY_SCALAR_MAPPING:{field}:{live}")
            scalars.append(label)
    if tuple(scalars) != DYNAMIC_SCALARS:
        raise BoundaryBindingError("BOUNDARY_SCALAR_ORDER")
    return {
        "path": str(mapping_path),
        "rows": rows,
        "diagnostic_bindings": tuple(diagnostics),
        "dynamic_scalars": DYNAMIC_SCALARS,
        "field_count": len(rows),
        "identity": {
            **BOUNDARY_MAPPING_IDENTITY,
            "path": str(mapping_path),
        },
    }


def _post_fields(
    thermal_slow_state: Any,
    interval_index: int,
    state_after: Any,
    *,
    directory: Path,
    science_arrays: Mapping[str, Any] | None = None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    method = getattr(thermal_slow_state, "post_fields", None)
    if not callable(method):
        raise BoundaryCaptureError("BOUNDARY_POST_FIELDS_METHOD_MISSING")
    try:
        payload = method(int(interval_index), state_after)
    except BaseException as exc:
        raise BoundaryCaptureError("BOUNDARY_POST_FIELDS_READ_FAILED") from exc
    if type(payload) is not dict or tuple(sorted(payload)) != tuple(sorted(POST_FIELDS)):
        raise BoundaryCaptureError("BOUNDARY_POST_FIELDS_SCHEMA")
    formal_metadata: dict[str, Any] = {
        "view_class": f"{type(thermal_slow_state).__module__}.{type(thermal_slow_state).__qualname__}",
        "view_root": None,
        "view_current_generation": getattr(thermal_slow_state, "current_generation", None),
        "view_current_content_sha256": getattr(thermal_slow_state, "current_content_sha256", None),
        "durable_post_checked": False,
        "durable_pre_identity_checked": False,
    }
    formal_metadata["view_aliases"] = _alias_rows(
        {f"POST.{name}": payload[name] for name in POST_FIELDS}
    )
    view_read = getattr(thermal_slow_state, "_read", None)
    try:
        read_fields = view_read[int(interval_index)] if isinstance(view_read, Mapping) else None
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise BoundaryCaptureError("BOUNDARY_FORMAL_VIEW_READ_FIELDS_MISSING") from exc
    if not isinstance(read_fields, Mapping) or tuple(sorted(read_fields)) != tuple(sorted(POST_FIELDS)):
        raise BoundaryCaptureError("BOUNDARY_FORMAL_VIEW_READ_FIELDS_MISSING")
    post_source_values = {
        **{f"POST.{name}": payload[name] for name in POST_FIELDS},
        "state_after": state_after,
        **{f"VIEW_READ.{name}": read_fields[name] for name in POST_FIELDS},
    }
    post_source_aliases = _alias_rows(post_source_values)
    if any(
        row["left"] != row["right"] and bool(row["shares_memory"])
        for row in post_source_aliases
    ):
        raise BoundaryCaptureError("BOUNDARY_POST_COPY_ALIAS")
    formal_metadata["post_source_aliases"] = post_source_aliases
    formal_metadata["post_copy_aliases_verified"] = True
    del post_source_values, read_fields, view_read
    if science_arrays is not None:
        live_science = dict(science_arrays)
        live_science.update({f"POST.{name}": payload[name] for name in POST_FIELDS})
        if tuple(live_science) != (
            "E",
            "Qacc",
            "Qacc_raman",
            "POST.delta_n",
            "POST.vx",
            "POST.vy",
        ):
            raise BoundaryCaptureError("BOUNDARY_SCIENCE_ALIAS_FIELD_ORDER")
        formal_metadata["science_aliases"] = _alias_rows(live_science)
        formal_metadata["science_native_layouts"] = {
            name: _array_layout(value) for name, value in live_science.items()
        }
    root = getattr(thermal_slow_state, "root", None)
    if root is not None:
        root_path = Path(root).resolve(strict=True)
        formal_metadata["view_root"] = str(root_path)
        try:
            from KHz_filament.hr4e5s_streaming import StreamingLifecycle
        except Exception as exc:
            raise BoundaryCaptureError("BOUNDARY_FORMAL_STREAMING_IMPORT_FAILED") from exc
        try:
            lifecycle = StreamingLifecycle.open(root_path)
            durable_generation = lifecycle.manifest.get("current_generation")
            durable_content = lifecycle.manifest.get("current_content_sha256")
            view_generation = formal_metadata.get("view_current_generation")
            view_content = formal_metadata.get("view_current_content_sha256")
            if (
                not durable_generation
                or not durable_content
                or not view_generation
                or not view_content
                or str(durable_generation) != str(view_generation)
                or str(durable_content) != str(view_content)
            ):
                raise BoundaryCaptureError("BOUNDARY_DURABLE_PRE_IDENTITY_CHANGED")
            formal_metadata.update(
                {
                    "durable_pre_identity_checked": True,
                    "durable_manifest_generation": str(durable_generation),
                    "durable_manifest_content_sha256": str(durable_content),
                }
            )
            record = lifecycle.manifest["records"][int(interval_index)]
            post_record = record.get("post")
            if not isinstance(post_record, dict):
                raise BoundaryCaptureError("BOUNDARY_DURABLE_POST_MISSING")
            durable = lifecycle._artifact_fields(post_record, namespace="POST")
            for name in POST_FIELDS:
                left = payload[name]
                right = durable[name]
                if type(left) is not type(right) or left.dtype != right.dtype or left.shape != right.shape:
                    raise BoundaryCaptureError(f"BOUNDARY_DURABLE_POST_LAYOUT:{name}")
                if _raw_digest(left, label=f"formal_view:{name}") != _raw_digest(right, label=f"durable:{name}"):
                    raise BoundaryCaptureError(f"BOUNDARY_DURABLE_POST_VALUE:{name}")
            formal_metadata.update(
                {
                    "durable_post_checked": True,
                    "durable_record_ordinal": int(record["ordinal"]),
                    "durable_post_artifact": str(post_record["artifact"]),
                    "durable_post_file_sha256": str(post_record["file_sha256"]),
                    "durable_post_field_sha256": dict(post_record.get("field_sha256", {})),
                    "durable_post_namespace": "POST",
                }
            )
        except BoundaryCaptureError:
            raise
        except BaseException as exc:
            raise BoundaryCaptureError("BOUNDARY_DURABLE_POST_READ_FAILED") from exc
    try:
        result = tuple(
            _write_snapshot_array(payload[name], label=f"POST.{name}", directory=directory)
            for name in POST_FIELDS
        )
    finally:
        del payload
    if root is not None and not bool(formal_metadata["durable_post_checked"]):
        raise BoundaryCaptureError("BOUNDARY_DURABLE_POST_NOT_CHECKED")
    if root is not None and not bool(formal_metadata["durable_pre_identity_checked"]):
        raise BoundaryCaptureError("BOUNDARY_DURABLE_PRE_IDENTITY_CHANGED")
    return result, formal_metadata


def _validate_g_state_guard(state: Any, role: str) -> dict[str, Any] | None:
    if str(role).upper() != "G_CONTROLLED_INPUT":
        return None
    required = (
        "_g_pending_cap",
        "_g_active_cap_scope",
        "_g_last_scope",
        "_g_propagation_active",
    )
    if any(not hasattr(state, name) for name in required):
        raise BoundaryCaptureError("BOUNDARY_G_STATE_GUARD_FIELDS_MISSING")
    pending = getattr(state, "_g_pending_cap")
    active_scope = getattr(state, "_g_active_cap_scope")
    last_scope = getattr(state, "_g_last_scope")
    propagation_active = getattr(state, "_g_propagation_active")
    if pending is not None:
        raise BoundaryCaptureError("BOUNDARY_G_PENDING_CAPABILITY_NOT_CLOSED")
    if active_scope is not None:
        raise BoundaryCaptureError("BOUNDARY_G_ACTIVE_CAPABILITY_NOT_CLOSED")
    if str(last_scope) != "interval_closed":
        raise BoundaryCaptureError("BOUNDARY_G_SCOPE_NOT_INTERVAL_CLOSED")
    if propagation_active is not True:
        raise BoundaryCaptureError("BOUNDARY_G_PROPAGATION_NOT_ACTIVE_AT_CAPTURE")
    return {
        "pending_capability_none": True,
        "active_capability_scope": None,
        "last_scope": str(last_scope),
        "propagation_active_until_finalize": True,
    }


def _alias_rows(values: Mapping[str, Any]) -> list[dict[str, Any]]:
    names = list(values)
    result = []
    for index, left_name in enumerate(names):
        left = values[left_name]
        for right_name in names[index:]:
            right = values[right_name]
            try:
                import numpy as np

                shares = bool(np.shares_memory(left, right))
            except (TypeError, ValueError):
                shares = left is right
            result.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "same_object": bool(left is right),
                    "shares_memory": shares,
                }
            )
    return result


def _linear_result_metadata(
    linear_result: Any,
    E: Any,
    *,
    need_energy: Any,
    need_profile: Any,
) -> dict[str, Any]:
    """Record stage-2 ``linear_result`` under the four frozen return forms."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise BoundaryCaptureError("NUMPY_REQUIRED_FOR_LINEAR_RESULT_ALIAS") from exc

    if type(need_energy) is not bool or type(need_profile) is not bool:
        raise BoundaryCaptureError("BOUNDARY_LINEAR_FLAGS_NOT_BOOL")
    need_energy = bool(need_energy)
    need_profile = bool(need_profile)
    if not need_energy and not need_profile:
        expected_kind = "ndarray"
        expected_length = None
        if type(linear_result) is not np.ndarray:
            raise BoundaryCaptureError("BOUNDARY_LINEAR_RESULT_EXPECTED_NDARRAY")
        container_type = "ndarray"
        container_length: int | None = None
        element0 = linear_result
        element0_index: int | None = None
    else:
        expected_kind = "tuple"
        expected_length = 3 if need_energy and need_profile else 2
        if type(linear_result) is not tuple or len(linear_result) != expected_length:
            raise BoundaryCaptureError(
                f"BOUNDARY_LINEAR_RESULT_EXPECTED_TUPLE_LEN:{expected_length}"
            )
        container_type = "tuple"
        container_length = len(linear_result)
        element0 = linear_result[0]
        element0_index = 0
    if type(element0) is not np.ndarray or type(E) is not np.ndarray:
        raise BoundaryCaptureError("BOUNDARY_LINEAR_RESULT_ELEMENT_NOT_NATIVE_ARRAY")
    if element0 is not E:
        raise BoundaryCaptureError("BOUNDARY_LINEAR_RESULT_FIELD0_NOT_E")
    try:
        shares = bool(np.shares_memory(element0, E))
    except (TypeError, ValueError) as exc:
        raise BoundaryCaptureError("BOUNDARY_LINEAR_RESULT_ALIAS_CHECK_FAILED") from exc
    if not shares:
        raise BoundaryCaptureError("BOUNDARY_LINEAR_RESULT_FIELD0_NO_ALIAS")
    return {
        "source_variable": "linear_result",
        "source_scope": "propagate_one_pulse.stage2_boundary",
        "source_candidate_lines": [649, 1248],
        "need_energy": need_energy,
        "need_profile": need_profile,
        "expected_container_kind": expected_kind,
        "expected_container_length": expected_length,
        "container_type": container_type,
        "container_length": container_length,
        "element0_index": element0_index,
        "element0_is_E": bool(element0 is E),
        "element0_shares_memory_E": shares,
        "element0_layout": _array_layout(element0),
        "E_layout": _array_layout(E),
        "retained_live_refs": 0,
        "retained_live_bytes": 0,
    }


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


class BoundaryCollector:
    """One-shot scalar plus isolated-snapshot collector for interval zero."""

    __slots__ = (
        "role",
        "diagnostic_bindings",
        "dynamic_names",
        "source_identity",
        "mapping_identity",
        "snapshot_root",
        "_captured",
        "_thread_id",
        "_receipt",
        "_capture_index",
        "_failure_index",
    )

    def __init__(
        self,
        role: str,
        *,
        diagnostic_bindings: Iterable[tuple[str, str]],
        dynamic_names: Iterable[str] = DYNAMIC_SCALARS,
        source_identity: Mapping[str, Any] | None = None,
        mapping_identity: Mapping[str, Any] | None = None,
        snapshot_root: str | Path,
    ) -> None:
        role_name = str(role).upper()
        if role_name == "G":
            role_name = "G_CONTROLLED_INPUT"
        if role_name not in {"I0", "G_CONTROLLED_INPUT"}:
            raise BoundaryBindingError("BOUNDARY_ROLE_UNSUPPORTED")
        bindings = tuple((str(label), str(name)) for label, name in diagnostic_bindings)
        if len(bindings) != EXPECTED_DIAGNOSTIC_COUNT:
            raise BoundaryBindingError("BOUNDARY_DIAGNOSTIC_COUNT")
        if len({label for label, _ in bindings}) != len(bindings):
            raise BoundaryBindingError("BOUNDARY_DIAGNOSTIC_LABEL_DUPLICATE")
        if any(not label.isidentifier() or not name.isidentifier() for label, name in bindings):
            raise BoundaryBindingError("BOUNDARY_DIAGNOSTIC_LOCAL_NAME_INVALID")
        dynamic = tuple(str(name) for name in dynamic_names)
        if dynamic != DYNAMIC_SCALARS:
            raise BoundaryBindingError("BOUNDARY_DYNAMIC_SCALAR_CONTRACT_DRIFT")
        self.role = role_name
        self.diagnostic_bindings = bindings
        self.dynamic_names = dynamic
        self.source_identity = dict(source_identity or {})
        self.mapping_identity = dict(mapping_identity or {})
        self.snapshot_root = Path(snapshot_root).resolve()
        self._captured = False
        self._thread_id: int | None = None
        self._receipt: dict[str, Any] | None = None
        self._capture_index = 0
        self._failure_index = 0

    @property
    def captured(self) -> bool:
        return bool(self._captured)

    def _reserve_capture_dir(self) -> tuple[Path, Path]:
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        while True:
            name = f"boundary_{self._capture_index:03d}"
            self._capture_index += 1
            final = self.snapshot_root / name
            temporary = self.snapshot_root / (name + ".tmp")
            if final.exists() or temporary.exists():
                continue
            temporary.mkdir()
            return temporary, final

    def _reserve_failure_dir(self) -> tuple[Path, Path]:
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        while True:
            name = f"failure_{self._failure_index:03d}"
            self._failure_index += 1
            final = self.snapshot_root / name
            temporary = self.snapshot_root / (name + ".tmp")
            if final.exists() or temporary.exists():
                continue
            temporary.mkdir()
            return temporary, final

    def _preserve_failure(self, temporary: Path, final: Path, exc: BaseException) -> None:
        """Move partial evidence into a unique failure root without deletion."""
        failure_temporary: Path | None = None
        failure_final: Path | None = None
        try:
            failure_temporary, failure_final = self._reserve_failure_dir()
            preserved_parts = []
            for label, source in (("temporary", temporary), ("finalized_partial", final)):
                if source.exists():
                    target = failure_temporary / label
                    source.replace(target)
                    preserved_parts.append(
                        {"label": label, "relative_path": label, "original_path": str(source)}
                    )
            receipt = {
                "schema": SCHEMA + ".failure",
                "status": "FAIL_BOUNDARY_CAPTURE_PRESERVED",
                "qualification_eligible": False,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "exception_repr": repr(exc),
                "preserved_root": str(failure_final),
                "preserved_parts": preserved_parts,
                "members": _member_records(failure_temporary),
                "unique_evidence_preserved": True,
                "deletion_performed": False,
            }
            with (failure_temporary / "FAILURE_RECEIPT.json").open(
                "x", encoding="utf-8", newline="\n"
            ) as stream:
                stream.write(_json_dumps(receipt))
            failure_temporary.replace(failure_final)
        except BaseException as preserve_exc:
            # Never remove a partial capture merely because preservation failed.
            # Leave the original temporary/final directory in place and make a
            # best-effort non-overwriting receipt inside the first available root.
            fallback_roots = [path for path in (temporary, final, failure_temporary) if path is not None and path.exists()]
            for root in fallback_roots:
                target = root / "FAILURE_RECEIPT.json"
                if target.exists():
                    continue
                try:
                    fallback = {
                        "schema": SCHEMA + ".failure",
                        "status": "FAIL_BOUNDARY_CAPTURE_PRESERVED_WITH_RECEIPT_ERROR",
                        "qualification_eligible": False,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "preservation_error": f"{type(preserve_exc).__name__}:{preserve_exc}",
                        "preserved_root": str(root),
                        "members": _member_records(root),
                        "unique_evidence_preserved": True,
                        "deletion_performed": False,
                    }
                    with target.open("x", encoding="utf-8", newline="\n") as stream:
                        stream.write(_json_dumps(fallback))
                    break
                except BaseException:
                    continue

    def capture(
        self,
        *,
        interval_index: Any,
        E: Any,
        Qacc: Any,
        Qacc_raman: Any,
        linear_result: Any,
        need_energy: Any,
        need_profile: Any,
        thermal_slow_state: Any,
        state_after: Any,
        z: Any,
        dz_try: Any,
        U0_baseline: Any,
        U_now: Any,
        adaptive_rejection_count: Any,
        safety_mode_trigger_count: Any,
        diagnostic_values: tuple[Any, ...],
        caller_code: Any,
        lease: "BoundaryLease",
    ) -> dict[str, Any] | None:
        """Capture interval zero only; all payload names are explicit."""
        try:
            index = int(interval_index)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BoundaryCaptureError("BOUNDARY_INTERVAL_INDEX_INVALID") from exc
        if index != 0:
            return None
        if self._captured:
            raise BoundaryCaptureError("BOUNDARY_CAPTURE_REENTRY")
        if not isinstance(diagnostic_values, tuple) or len(diagnostic_values) != len(self.diagnostic_bindings):
            raise BoundaryCaptureError("BOUNDARY_DIAGNOSTIC_VALUE_TUPLE_SCHEMA")
        current_thread = int(threading.get_ident())
        if self._thread_id is None:
            self._thread_id = current_thread
        elif self._thread_id != current_thread:
            raise BoundaryCaptureError("BOUNDARY_CAPTURE_THREAD_MISMATCH")
        lease._validate(caller_code=caller_code, thread_id=current_thread)
        g_state_guard = _validate_g_state_guard(lease.state, self.role)
        started = time.perf_counter()
        temporary, final = self._reserve_capture_dir()
        try:
            source_arrays = {"E": E, "Qacc": Qacc, "Qacc_raman": Qacc_raman}
            linear_result_metadata = _linear_result_metadata(
                linear_result,
                E,
                need_energy=need_energy,
                need_profile=need_profile,
            )
            arrays = [
                _write_snapshot_array(value, label=label, directory=temporary)
                for label, value in source_arrays.items()
            ]
            post_arrays, post_verification = _post_fields(
                    thermal_slow_state,
                    index,
                    state_after,
                    directory=temporary,
                science_arrays={"E": E, "Qacc": Qacc, "Qacc_raman": Qacc_raman},
                )
            arrays.extend(post_arrays)
            diagnostics = [
                _sequence_snapshot(value, label=f"diagnostics.{label}", directory=temporary)
                for (label, _), value in zip(self.diagnostic_bindings, diagnostic_values)
            ]
            scalar_inputs = dict(
                zip(
                    self.dynamic_names,
                    (
                        z,
                        interval_index,
                        dz_try,
                        U0_baseline,
                        U_now,
                        adaptive_rejection_count,
                        safety_mode_trigger_count,
                    ),
                )
            )
            scalar_values = {
                name: _safe_scalar(value) for name, value in scalar_inputs.items()
            }
            snapshot_tree = _write_snapshot_tree(
                temporary,
                array_records=arrays,
                diagnostic_records=diagnostics,
                scalar_values=scalar_inputs,
                scalar_safe_values=scalar_values,
            )
            alias_values = {label: value for label, value in source_arrays.items()}
            receipt = {
                "schema": SCHEMA,
                "status": "PASS_BOUNDARY_CAPTURE_WITH_ISOLATED_SNAPSHOTS",
                "qualification_eligible": False,
                "role": self.role,
                "interval_index": 0,
                "caller_code": _identity_code(caller_code),
                "source_identity": dict(self.source_identity),
                "mapping_identity": dict(self.mapping_identity),
                "snapshot_root": str(self.snapshot_root),
                "snapshot_directory": str(final),
                "snapshot_tree": snapshot_tree,
                "arrays": arrays,
                "diagnostics": diagnostics,
                "linear_result_binding": linear_result_metadata,
                "dynamic_scalars": scalar_values,
                "post_fields": list(POST_FIELDS),
                "post_verification": post_verification,
                "g_state_guard": g_state_guard,
                "field_count": 6 + len(diagnostics) + len(scalar_values),
                "array_aliases_input": _alias_rows(alias_values),
                "array_aliases_post": list(post_verification.get("view_aliases", [])),
                "array_aliases_science": list(post_verification.get("science_aliases", [])),
                "retained_live_array_refs": 0,
                "retained_live_array_bytes": 0,
                "exclusive_capture_s": 0.0,
                "snapshot_isolation": "COPY_FOR_EXACT_ONLY_NOT_USED_FOR_QUALIFICATION",
            }
            if int(receipt["field_count"]) != EXPECTED_TOTAL_FIELD_COUNT:
                raise BoundaryCaptureError("BOUNDARY_FIELD_COUNT")
            receipt["native_layouts"] = {
                str(row["field_name"]): dict(row["source_layout"])
                for row in arrays + diagnostics
            }
            receipt["aliases"] = {
                "input": list(receipt["array_aliases_input"]),
                "post": list(receipt["array_aliases_post"]),
                "science": list(receipt["array_aliases_science"]),
                "linear_result": dict(linear_result_metadata),
            }
            receipt["exclusive_capture_s"] = float(time.perf_counter() - started)
            (temporary / "BOUNDARY_RECEIPT.json").write_text(
                _json_dumps(receipt), encoding="utf-8", newline="\n"
            )
            temporary.replace(final)
            receipt["snapshot_directory"] = str(final)
            receipt["snapshot_tree"]["root"] = str(final)
            (final / "BOUNDARY_RECEIPT.json").write_text(
                _json_dumps(receipt), encoding="utf-8", newline="\n"
            )
            self._receipt = receipt
            self._captured = True
            lease._mark_captured()
            del arrays, diagnostics, scalar_values, scalar_inputs, snapshot_tree, source_arrays, alias_values, linear_result_metadata
            return dict(receipt)
        except BaseException as exc:
            self._preserve_failure(temporary, final, exc)
            raise

    def receipt(self) -> dict[str, Any] | None:
        return None if self._receipt is None else dict(self._receipt)

    def binding(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "role": self.role,
            "diagnostic_count": len(self.diagnostic_bindings),
            "diagnostic_bindings": [
                {"field_name": label, "live_source": source}
                for label, source in self.diagnostic_bindings
            ],
            "dynamic_scalars": list(self.dynamic_names),
            "post_fields": list(POST_FIELDS),
            "snapshot_root": str(self.snapshot_root),
            "retained_live_array_refs": 0,
            "retained_live_array_bytes": 0,
            "source_identity": dict(self.source_identity),
            "mapping_identity": dict(self.mapping_identity),
            "capture_once_interval_zero": True,
            "snapshot_not_qualification": True,
        }


class BoundaryLease:
    """Source-code/thread/state-bound one-shot qualification capability."""

    __slots__ = (
        "collector",
        "state",
        "mode",
        "generation",
        "generation_getter",
        "expected_capture_generation",
        "caller_code",
        "thread_id",
        "_active",
        "_captured",
        "_retired",
        "_validated_generation",
    )

    def __init__(
        self,
        collector: BoundaryCollector,
        *,
        state: Any = None,
        mode: str = "QUALIFY_BOUNDARY",
        generation: int | None = None,
        generation_getter: Any = None,
        expected_capture_generation: int | None = None,
        caller_code: Any = None,
    ) -> None:
        if str(mode) != "QUALIFY_BOUNDARY":
            raise BoundaryBindingError("BOUNDARY_LEASE_MODE_UNSUPPORTED")
        if not isinstance(collector, BoundaryCollector):
            raise BoundaryBindingError("BOUNDARY_COLLECTOR_TYPE")
        if generation_getter is not None and not callable(generation_getter):
            raise BoundaryBindingError("BOUNDARY_GENERATION_GETTER_NOT_CALLABLE")
        self.collector = collector
        self.state = state
        self.mode = str(mode)
        self.generation = None if generation is None else int(generation)
        self.generation_getter = generation_getter
        self.expected_capture_generation = (
            None
            if expected_capture_generation is None
            else int(expected_capture_generation)
        )
        self.caller_code = caller_code
        self.thread_id: int | None = None
        self._active = False
        self._captured = False
        self._retired = False
        self._validated_generation: int | None = None

    def activate(self) -> "BoundaryLease":
        if self._retired:
            raise BoundaryLeaseError("BOUNDARY_LEASE_RETIRED")
        if self._active:
            raise BoundaryLeaseError("BOUNDARY_LEASE_REENTRANT")
        if self.generation_getter is not None:
            try:
                current = int(self.generation_getter())
            except BaseException as exc:
                raise BoundaryLeaseError("BOUNDARY_ACTIVATION_GENERATION_READ_FAILED") from exc
            if current < 0:
                raise BoundaryLeaseError("BOUNDARY_ACTIVATION_GENERATION_INVALID")
            if self.generation is not None and current != int(self.generation):
                raise BoundaryLeaseError("BOUNDARY_ACTIVATION_GENERATION_STALE")
            self._validated_generation = current
        self.thread_id = int(threading.get_ident())
        self._active = True
        return self

    def bind_code(self, caller_code: Any) -> None:
        if self.caller_code is not None and self.caller_code is not caller_code:
            raise BoundaryBindingError("BOUNDARY_CALLER_CODE_REBIND")
        self.caller_code = caller_code

    def _validate(self, *, caller_code: Any, thread_id: int) -> None:
        if self._retired or not self._active:
            raise BoundaryLeaseError("BOUNDARY_LEASE_INACTIVE")
        if self._captured:
            raise BoundaryLeaseError("BOUNDARY_LEASE_ALREADY_CAPTURED")
        if self.thread_id is None or int(thread_id) != int(self.thread_id):
            raise BoundaryLeaseError("BOUNDARY_LEASE_THREAD_MISMATCH")
        if self.caller_code is None or caller_code is not self.caller_code:
            raise BoundaryLeaseError("BOUNDARY_LEASE_CALLER_CODE_MISMATCH")
        if self.state is None:
            raise BoundaryLeaseError("BOUNDARY_LEASE_STATE_RETIRED")
        if self.generation_getter is not None:
            try:
                current = int(self.generation_getter())
            except BaseException as exc:
                raise BoundaryLeaseError("BOUNDARY_GENERATION_READ_FAILED") from exc
            if current < 0:
                raise BoundaryLeaseError("BOUNDARY_GENERATION_INVALID")
            self._validated_generation = current
            if self.generation is not None and current < int(self.generation):
                raise BoundaryLeaseError("BOUNDARY_GENERATION_REGRESSION")
            if (
                self.expected_capture_generation is not None
                and current != int(self.expected_capture_generation)
            ):
                raise BoundaryLeaseError("BOUNDARY_CAPTURE_GENERATION_MISMATCH")

    def _mark_captured(self) -> None:
        self._captured = True

    def close(self) -> None:
        self._active = False
        self._retired = True
        self.state = None
        self.generation_getter = None
        self.caller_code = None

    def result(self) -> dict[str, Any] | None:
        return self.collector.receipt()

    def transparency(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "mode": self.mode,
            "active": bool(self._active),
            "captured": bool(self._captured),
            "retired": bool(self._retired),
            "thread_id": self.thread_id,
            "caller_code_bound": self.caller_code is not None,
            "validated_generation": self._validated_generation,
            "expected_capture_generation": self.expected_capture_generation,
            "state_bound": self.state is not None,
            "collector": self.collector.binding(),
        }


def make_boundary_lease(
    role: str,
    *,
    diagnostic_bindings: Iterable[tuple[str, str]],
    source_identity: Mapping[str, Any] | None = None,
    mapping_identity: Mapping[str, Any] | None = None,
    snapshot_root: str | Path,
    state: Any = None,
    generation: int | None = None,
    generation_getter: Any = None,
    expected_capture_generation: int | None = None,
    caller_code: Any = None,
) -> BoundaryLease:
    collector = BoundaryCollector(
        role,
        diagnostic_bindings=diagnostic_bindings,
        source_identity=source_identity,
        mapping_identity=mapping_identity,
        snapshot_root=snapshot_root,
    )
    return BoundaryLease(
        collector,
        state=state,
        generation=generation,
        generation_getter=generation_getter,
        expected_capture_generation=expected_capture_generation,
        caller_code=caller_code,
    )


def explicit_boundary_capture(
    lease: BoundaryLease,
    *,
    interval_index: Any,
    E: Any,
    Qacc: Any,
    Qacc_raman: Any,
    linear_result: Any,
    need_energy: Any,
    need_profile: Any,
    thermal_slow_state: Any,
    state_after: Any,
    z: Any,
    dz_try: Any,
    U0_baseline: Any,
    U_now: Any,
    adaptive_rejection_count: Any,
    safety_mode_trigger_count: Any,
    diagnostic_values: tuple[Any, ...],
) -> dict[str, Any] | None:
    """Entry used by the clone; caller code is read and immediately dropped."""
    if not isinstance(lease, BoundaryLease):
        raise BoundaryLeaseError("BOUNDARY_LEASE_TYPE")
    frame = sys._getframe(1)
    caller_code = frame.f_code
    del frame
    return lease.collector.capture(
        interval_index=interval_index,
        E=E,
        Qacc=Qacc,
        Qacc_raman=Qacc_raman,
        linear_result=linear_result,
        need_energy=need_energy,
        need_profile=need_profile,
        thermal_slow_state=thermal_slow_state,
        state_after=state_after,
        z=z,
        dz_try=dz_try,
        U0_baseline=U0_baseline,
        U_now=U_now,
        adaptive_rejection_count=adaptive_rejection_count,
        safety_mode_trigger_count=safety_mode_trigger_count,
        diagnostic_values=diagnostic_values,
        caller_code=caller_code,
        lease=lease,
    )


__all__ = [
    "SCHEMA",
    "BOUNDARY_ASSIGNMENT",
    "POST_FIELDS",
    "DYNAMIC_SCALARS",
    "SCIENCE_ARRAY_FIELDS",
    "EXPECTED_DIAGNOSTIC_COUNT",
    "EXPECTED_TOTAL_FIELD_COUNT",
    "BoundaryBindingError",
    "BoundaryLeaseError",
    "BoundaryCaptureError",
    "BoundaryCollector",
    "BoundaryLease",
    "load_boundary_mapping",
    "make_boundary_lease",
    "explicit_boundary_capture",
]
