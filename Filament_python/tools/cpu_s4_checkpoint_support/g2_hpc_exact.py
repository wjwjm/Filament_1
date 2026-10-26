"""Exact, provenance-preserving helpers for the private G2 HPC harness.

The numerical implementation remains in the frozen A07/G1 modules.  This
module only loads isolated evidence, records current native metadata, and
delegates current-output comparisons to the unchanged ``g1_exact`` helper.
Historical boundary records have no native alias/layout evidence; those
fields are therefore compared by value and dtype/shape only.
"""
from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path
from typing import Any, Iterable, Mapping


SCIENCE_ARRAY_FIELDS = ("E", "Qacc", "Qacc_raman")
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
EXPECTED_BOUNDARY_FIELDS = 112
EXPECTED_BOUNDARY_ARRAY_FIELDS = 105
EXPECTED_BOUNDARY_DIAGNOSTICS = 99
BOUNDARY_RECEIPT_SCHEMA = "filament.cpu_s4_3g2.boundary_lease.v2"
EXPECTED_TERMINAL_DIAGNOSTIC_ARRAYS = 128
EXPECTED_TERMINAL_SCALARS = 102
EXPECTED_TERMINAL_COMPARISON_ROWS = 241
PROFILE_DIAGNOSTIC_FIELDS = (
    "bk_nee_profile_halfstep_1_json",
    "bk_nee_profile_halfstep_2_json",
)


def require(condition: bool, reason: str, context: Any = None) -> None:
    if not condition:
        raise RuntimeError({"gate": str(reason), "context": context})


def _json_safe(value: Any) -> Any:
    """Make evidence JSON-safe without silently changing scientific values."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - target environments provide NumPy
        np = None
    if np is not None and isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, complex):
        return {"real": _json_safe(float(value.real)), "imag": _json_safe(float(value.imag))}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def write_json(path: str | os.PathLike[str], value: Any) -> Path:
    """Write one new receipt without replacing an earlier evidence payload."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(_json_safe(value), stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return target


def raw_sha256_file(path: str | os.PathLike[str]) -> str:
    from tools.hpc_ops import provenance_v2 as pv

    return str(pv.raw_sha256_file(Path(path)))


def _array_layout(value: Any, *, pointer: bool = True) -> dict[str, Any]:
    import numpy as np

    require(isinstance(value, np.ndarray), "NATIVE_METADATA_ARRAY_REQUIRED", type(value).__name__)
    base = value
    depth = 0
    while isinstance(getattr(base, "base", None), np.ndarray):
        depth += 1
        base = base.base
    layout: dict[str, Any] = {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "strides": [int(item) for item in value.strides],
        "C": bool(value.flags.c_contiguous),
        "F": bool(value.flags.f_contiguous),
        "OWNDATA": bool(value.flags.owndata),
        "WRITEABLE": bool(value.flags.writeable),
        "base_chain_depth": int(depth),
        "base_ndarray": bool(isinstance(getattr(value, "base", None), np.ndarray)),
    }
    if pointer:
        layout["pointer"] = int(value.__array_interface__["data"][0])
    return layout


def _arrays(value: Any, path: str = "bundle") -> Iterable[tuple[str, Any]]:
    import numpy as np

    if isinstance(value, np.ndarray):
        yield path, value
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _arrays(child, f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from _arrays(child, f"{path}.{index}")


def native_aliases(value: Any) -> list[dict[str, Any]]:
    """Return all current pairwise ndarray aliases, including self rows."""
    import numpy as np

    named = list(_arrays(value))
    rows: list[dict[str, Any]] = []
    for index, (left_name, left) in enumerate(named):
        for right_name, right in named[index:]:
            rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "same_object": bool(left is right),
                    "shares_memory": bool(np.shares_memory(left, right)),
                }
            )
    return rows


def native_metadata(value: Any, *, pointer: bool = True) -> dict[str, Any]:
    """Capture current output layouts and aliases before any archive copies."""
    return {
        "schema": "filament.cpu_s4_3g2.current_native_output.v1",
        "pointer_included": bool(pointer),
        "arrays": {
            name: _array_layout(array, pointer=pointer)
            for name, array in _arrays(value)
        },
        "aliases": native_aliases(value),
        "historical_native_layout": "UNOBSERVED_IN_ORIGINAL_HISTORY",
    }


def compare_native_metadata(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Compare current native metadata while deliberately ignoring pointers."""
    def without_pointers(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): without_pointers(item) for key, item in value.items() if str(key) != "pointer"}
        if isinstance(value, list):
            return [without_pointers(item) for item in value]
        return value

    reference = without_pointers(reference)
    candidate = without_pointers(candidate)
    left_arrays = reference.get("arrays", {})
    right_arrays = candidate.get("arrays", {})
    failures: list[dict[str, Any]] = []
    if set(left_arrays) != set(right_arrays):
        failures.append({"reason": "array_path_set", "reference": sorted(left_arrays), "candidate": sorted(right_arrays)})
    for name in sorted(set(left_arrays) & set(right_arrays)):
        left = {key: value for key, value in left_arrays[name].items() if key != "pointer"}
        right = {key: value for key, value in right_arrays[name].items() if key != "pointer"}
        if left != right:
            failures.append({"reason": "native_layout", "field": name, "reference": left, "candidate": right})
    if reference.get("aliases") != candidate.get("aliases"):
        failures.append({"reason": "native_aliases", "reference": reference.get("aliases"), "candidate": candidate.get("aliases")})
    return {
        "status": "PASS_CURRENT_NATIVE_LAYOUT_ALIAS_EXACT" if not failures else "FAIL_CURRENT_NATIVE_LAYOUT_ALIAS",
        "pointer_excluded_from_comparison": True,
        "historical_native_layout": "UNOBSERVED_IN_ORIGINAL_HISTORY",
        "failures": failures,
    }


def file_records(root: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Index raw file size/hash without following symlinks."""
    root_path = Path(root).resolve(strict=True)
    paths = [root_path] if root_path.is_file() else sorted(root_path.rglob("*"))
    rows = []
    for path in paths:
        if not path.is_file() or path.is_symlink():
            continue
        rows.append(
            {
                "path": path.name if root_path.is_file() else path.relative_to(root_path).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256_raw": raw_sha256_file(path),
                "hash_scope": "raw_bytes",
            }
        )
    return rows


def save_file_index(path: str | os.PathLike[str], roots: Mapping[str, str | os.PathLike[str]]) -> dict[str, Any]:
    records = {str(name): file_records(value) for name, value in roots.items()}
    payload = {"schema": "filament.cpu_s4_3g2.raw_file_index.v1", "roots": records}
    write_json(path, payload)
    return payload


def _unique_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Decode one JSON object without allowing silent duplicate-key overwrite."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError({"gate": "BOUNDARY_JSON_DUPLICATE_KEY", "context": str(key)})
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream, object_pairs_hook=_unique_object_pairs)


def _boundary_receipt(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        if isinstance(value.get("boundary"), Mapping):
            return dict(value["boundary"])
        if "snapshot_directory" in value and "field_count" in value:
            return dict(value)
        if "boundary_receipt" in value and isinstance(value["boundary_receipt"], Mapping):
            return dict(value["boundary_receipt"])
        mode_capture = value.get("mode_capture")
        if isinstance(mode_capture, Mapping) and isinstance(mode_capture.get("receipt"), Mapping):
            return _boundary_receipt(mode_capture["receipt"])
    require(isinstance(value, (str, os.PathLike, Path)), "BOUNDARY_RECEIPT_INPUT")
    path = Path(value).resolve(strict=True)
    if path.is_dir():
        receipt = path / "BOUNDARY_RECEIPT.json"
        if not receipt.is_file() and (path / "CASE_RECEIPT.json").is_file():
            return _boundary_receipt(_load_json(path / "CASE_RECEIPT.json"))
        path = receipt
    data = _load_json(path)
    return _boundary_receipt(data)


def _safe_to_value(value: Any) -> Any:
    if value == "NaN":
        return float("nan")
    if value == "Infinity":
        return float("inf")
    if value == "-Infinity":
        return float("-inf")
    if isinstance(value, Mapping) and set(value) == {"real", "imag"}:
        return complex(_safe_to_value(value["real"]), _safe_to_value(value["imag"]))
    return value


def _receipt_source_path(value: Any) -> Path | None:
    """Return the supplied receipt file when ``value`` names one."""
    if isinstance(value, Mapping):
        return None
    if not isinstance(value, (str, os.PathLike, Path)):
        require(False, "BOUNDARY_RECEIPT_INPUT")
    path = Path(value).resolve(strict=True)
    if path.is_dir():
        receipt = path / "BOUNDARY_RECEIPT.json"
        if not receipt.is_file() and (path / "CASE_RECEIPT.json").is_file():
            return path / "CASE_RECEIPT.json"
        path = receipt
    require(path.is_file(), "BOUNDARY_RECEIPT_FILE_MISSING", str(path))
    return path


def _npy_header(path: Path) -> dict[str, Any]:
    """Read only an NPY header; no array payload is mapped or loaded."""
    import numpy as np

    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        readers = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
            (3, 0): np.lib.format.read_array_header_2_0,
        }
        require(version in readers, "BOUNDARY_NPY_VERSION", {"path": str(path), "version": version})
        shape, fortran_order, dtype = readers[version](stream)
    return {
        "shape": [int(item) for item in shape],
        "fortran_order": bool(fortran_order),
        "dtype": np.dtype(dtype).str,
        "version": [int(item) for item in version],
    }


def _safe_relative_snapshot(root: Path, raw_name: Any) -> Path:
    name = Path(str(raw_name))
    require(not name.is_absolute() and ".." not in name.parts, "BOUNDARY_SNAPSHOT_PATH")
    path = (root / name).resolve(strict=False)
    require(path.is_relative_to(root), "BOUNDARY_SNAPSHOT_ROOT_ESCAPE", str(path))
    return path


def _scalar_kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str) and value in {"NaN", "Infinity", "-Infinity"}:
        return "nonfinite_float"
    return type(value).__name__


def _scalar_fingerprint(value: Any) -> dict[str, Any]:
    """Record source/canonical scalar identity without changing its value."""
    kind = _scalar_kind(value)
    if isinstance(value, float):
        return {
            "type": kind,
            "value": value if math.isfinite(value) else ("NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")),
            "ieee754_bits": struct.pack("!d", value).hex(),
        }
    if kind == "nonfinite_float":
        canonical = _safe_to_value(value)
        return {
            "type": kind,
            "value": value,
            "canonical_type": _scalar_kind(canonical),
            "canonical_value": "NaN" if math.isnan(canonical) else ("Infinity" if canonical > 0 else "-Infinity"),
            "ieee754_bits": struct.pack("!d", canonical).hex(),
        }
    return {"type": kind, "value": value}


def _scalar_value_preserved(source: Any, canonical: Any) -> bool:
    """Check the existing safe-value conversion by type/value or float bits."""
    if isinstance(source, float):
        return isinstance(canonical, float) and struct.pack("!d", source) == struct.pack("!d", canonical)
    if isinstance(source, str) and source in {"NaN", "Infinity", "-Infinity"}:
        if not isinstance(canonical, float):
            return False
        marker = "NaN" if math.isnan(canonical) else ("Infinity" if canonical > 0 else "-Infinity")
        return marker == source
    return type(source) is type(canonical) and source == canonical


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def _load_frozen_mapping() -> dict[str, Any]:
    """Use the source-bound CSV validator; never duplicate its raw-byte gate."""
    try:
        from .g2_boundary import load_boundary_mapping
    except ImportError:  # support_module() imports this file as a top-level module
        from g2_boundary import load_boundary_mapping
    return load_boundary_mapping()


def _inspect_boundary_metadata(
    receipt: Mapping[str, Any],
    *,
    require_members: bool,
    header_only: bool = True,
) -> dict[str, Any]:
    """Validate boundary metadata before any array payload is loaded."""
    require(isinstance(receipt, Mapping), "BOUNDARY_RECEIPT_OBJECT")
    receipt_schema = receipt.get("schema")
    require(isinstance(receipt_schema, str) and receipt_schema == BOUNDARY_RECEIPT_SCHEMA,
            "BOUNDARY_RECEIPT_SCHEMA")
    field_count = receipt.get("field_count")
    require(isinstance(field_count, int) and not isinstance(field_count, bool) and field_count == EXPECTED_BOUNDARY_FIELDS,
            "BOUNDARY_FIELD_COUNT", field_count)
    snapshot_directory = receipt.get("snapshot_directory")
    require(isinstance(snapshot_directory, str) and snapshot_directory != "", "BOUNDARY_SNAPSHOT_DIRECTORY")
    root = Path(snapshot_directory).resolve(strict=False)
    arrays = receipt.get("arrays")
    diagnostics = receipt.get("diagnostics")
    require(isinstance(arrays, list) and isinstance(diagnostics, list), "BOUNDARY_ARRAY_RECORDS_MAPPING")
    require(len(arrays) == len(SCIENCE_ARRAY_FIELDS) + len(POST_FIELDS), "BOUNDARY_ARRAY_FIELD_COUNT", len(arrays))
    require(len(diagnostics) == EXPECTED_BOUNDARY_DIAGNOSTICS, "BOUNDARY_DIAGNOSTIC_COUNT", len(diagnostics))
    expected_array_order = list(SCIENCE_ARRAY_FIELDS) + [f"POST.{name}" for name in POST_FIELDS]
    native_layout_keys = {"shape", "dtype", "strides", "C", "F", "OWNDATA", "WRITEABLE", "base_chain_depth", "base_ndarray", "pointer"}
    record_keys = {
        "field_name", "snapshot_file", "source_layout", "snapshot_layout", "nbytes",
        "source_sha256_raw", "snapshot_sha256_raw", "snapshot_npy_sha256",
        "snapshot_isolated", "source_snapshot_shares_memory",
    }
    names: list[str] = []
    records = arrays + diagnostics
    for record in records:
        require(isinstance(record, Mapping), "BOUNDARY_ARRAY_RECORD_OBJECT")
        raw_name = record.get("field_name")
        require(isinstance(raw_name, str), "BOUNDARY_ARRAY_FIELD_NAME_TYPE", raw_name)
        name = raw_name
        require(name != "" and name not in names, "BOUNDARY_ARRAY_FIELD_NAME", name)
        names.append(name)
        require(set(record) == record_keys, "BOUNDARY_ARRAY_RECORD_KEYS", name)
        snapshot_file = record.get("snapshot_file")
        require(isinstance(snapshot_file, str) and snapshot_file != "", "BOUNDARY_SNAPSHOT_FILE_NAME", name)
        snapshot_path = Path(snapshot_file)
        require(not snapshot_path.is_absolute() and ".." not in snapshot_path.parts,
                "BOUNDARY_SNAPSHOT_PATH", name)
        require(isinstance(record["nbytes"], int) and not isinstance(record["nbytes"], bool)
                and record["nbytes"] >= 0, "BOUNDARY_ARRAY_NBYTES", name)
        for key in ("source_sha256_raw", "snapshot_sha256_raw", "snapshot_npy_sha256"):
            require(_is_sha256(record[key]), "BOUNDARY_ARRAY_SHA256", {"field": name, "key": key})
        require(record["snapshot_isolated"] is True and record["source_snapshot_shares_memory"] is False,
                "BOUNDARY_ARRAY_ISOLATION", name)
        for layout_key in ("source_layout", "snapshot_layout"):
            layout = record[layout_key]
            require(isinstance(layout, Mapping) and set(layout) == native_layout_keys,
                    "BOUNDARY_ARRAY_LAYOUT", {"field": name, "key": layout_key})
            shape = layout["shape"]
            require(isinstance(shape, list)
                    and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in shape),
                    "BOUNDARY_ARRAY_LAYOUT_SHAPE", {"field": name, "key": layout_key})
            require(isinstance(layout["dtype"], str) and layout["dtype"] != "",
                    "BOUNDARY_ARRAY_LAYOUT_DTYPE", {"field": name, "key": layout_key})
            require(isinstance(layout["strides"], list)
                    and len(shape) == len(layout["strides"])
                    and all(isinstance(item, int) and not isinstance(item, bool) for item in layout["strides"]),
                    "BOUNDARY_ARRAY_LAYOUT_STRIDES", {"field": name, "key": layout_key})
            for flag in ("C", "F", "OWNDATA", "WRITEABLE", "base_ndarray"):
                require(isinstance(layout[flag], bool), "BOUNDARY_ARRAY_LAYOUT_BOOL",
                        {"field": name, "key": layout_key, "flag": flag})
            require(isinstance(layout["base_chain_depth"], int)
                    and not isinstance(layout["base_chain_depth"], bool)
                    and layout["base_chain_depth"] >= 0,
                    "BOUNDARY_ARRAY_LAYOUT_DEPTH", {"field": name, "key": layout_key})
            require(isinstance(layout["pointer"], int)
                    and not isinstance(layout["pointer"], bool)
                    and layout["pointer"] >= 0,
                    "BOUNDARY_ARRAY_LAYOUT_POINTER", {"field": name, "key": layout_key})
    array_count = len(expected_array_order)
    require(names[:array_count] == expected_array_order, "BOUNDARY_ARRAY_FIELD_ORDER", names[:array_count])
    diagnostic_names = names[array_count:]
    require(len(diagnostic_names) == EXPECTED_BOUNDARY_DIAGNOSTICS
            and all(name.startswith("diagnostics.") and name.split(".", 1)[1].isidentifier()
                    for name in diagnostic_names),
            "BOUNDARY_DIAGNOSTIC_FIELD_SET", diagnostic_names[:3])
    for name, record in zip(names, records):
        require(record["snapshot_file"].replace("\\", "/") == f"{name}.npy",
                "BOUNDARY_SNAPSHOT_FILE_MAPPING", name)
    mapping = _load_frozen_mapping()
    mapping_names = [str(row["field_name"]) for row in mapping["rows"]]
    require(names + [f"scalars.{name}" for name in DYNAMIC_SCALARS] == mapping_names,
            "BOUNDARY_MAPPING_NAME_ORDER", {"receipt": names, "mapping": mapping_names})
    receipt_mapping_identity = receipt.get("mapping_identity")
    mapping_identity = mapping["identity"]
    require(isinstance(receipt_mapping_identity, Mapping), "BOUNDARY_MAPPING_IDENTITY_MAPPING")
    require(isinstance(receipt_mapping_identity.get("path"), str)
            and receipt_mapping_identity.get("hash_scope") == "raw_bytes"
            and receipt_mapping_identity.get("sha256") == mapping_identity["sha256"]
            and isinstance(receipt_mapping_identity.get("size_bytes"), int)
            and not isinstance(receipt_mapping_identity.get("size_bytes"), bool)
            and receipt_mapping_identity.get("size_bytes") == mapping_identity["size_bytes"],
            "BOUNDARY_MAPPING_IDENTITY")
    post_fields = receipt.get("post_fields")
    require(isinstance(post_fields, list) and all(isinstance(item, str) for item in post_fields),
            "BOUNDARY_POST_FIELD_TYPE")
    require(post_fields == list(POST_FIELDS), "BOUNDARY_POST_FIELD_SET")
    retained_refs = receipt.get("retained_live_array_refs")
    require(isinstance(retained_refs, int) and not isinstance(retained_refs, bool) and retained_refs == 0,
            "BOUNDARY_RETAINED_LIVE_ARRAY_REFS")
    native_layouts = receipt.get("native_layouts")
    require(isinstance(native_layouts, Mapping), "BOUNDARY_NATIVE_LAYOUTS_MAPPING")
    require(set(native_layouts) == set(names), "BOUNDARY_NATIVE_LAYOUTS_KEYS")
    for name in names:
        layout = native_layouts[name]
        require(isinstance(layout, Mapping) and set(layout) == native_layout_keys, "BOUNDARY_NATIVE_LAYOUT", name)
        require(isinstance(layout["shape"], list)
                and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in layout["shape"]),
                "BOUNDARY_NATIVE_LAYOUT_SHAPE", name)
        require(isinstance(layout["dtype"], str) and layout["dtype"] != "", "BOUNDARY_NATIVE_LAYOUT_DTYPE", name)
        require(isinstance(layout["strides"], list)
                and len(layout["shape"]) == len(layout["strides"])
                and all(isinstance(item, int) and not isinstance(item, bool) for item in layout["strides"]),
                "BOUNDARY_NATIVE_LAYOUT_STRIDES", name)
        for key in ("C", "F", "OWNDATA", "WRITEABLE", "base_ndarray"):
            require(isinstance(layout[key], bool), "BOUNDARY_NATIVE_LAYOUT_BOOL", {"field": name, "key": key})
        require(isinstance(layout["base_chain_depth"], int)
                and not isinstance(layout["base_chain_depth"], bool)
                and layout["base_chain_depth"] >= 0,
                "BOUNDARY_NATIVE_LAYOUT_DEPTH", name)
        require(isinstance(layout["pointer"], int)
                and not isinstance(layout["pointer"], bool)
                and layout["pointer"] >= 0,
                "BOUNDARY_NATIVE_LAYOUT_POINTER", name)
    aliases = receipt.get("aliases")
    require(isinstance(aliases, Mapping), "BOUNDARY_ALIASES_MAPPING")
    require(set(aliases) == {"input", "post", "science", "linear_result"}, "BOUNDARY_ALIASES_KEYS")
    alias_row_keys = {"left", "right", "same_object", "shares_memory"}
    for key in ("input", "post", "science"):
        require(isinstance(aliases[key], list), "BOUNDARY_ALIASES_LIST", key)
        for row in aliases[key]:
            require(isinstance(row, Mapping) and set(row) == alias_row_keys, "BOUNDARY_ALIAS_ROW", key)
            require(isinstance(row["left"], str) and isinstance(row["right"], str), "BOUNDARY_ALIAS_NAMES", key)
            require(isinstance(row["same_object"], bool) and isinstance(row["shares_memory"], bool),
                    "BOUNDARY_ALIAS_FLAGS", key)
    require(isinstance(aliases["linear_result"], Mapping), "BOUNDARY_ALIASES_LINEAR_RESULT")
    linear_result = aliases["linear_result"]
    linear_keys = {
        "E_layout", "container_length", "container_type", "element0_index", "element0_is_E",
        "element0_layout", "element0_shares_memory_E", "expected_container_kind",
        "expected_container_length", "need_energy", "need_profile", "retained_live_bytes",
        "retained_live_refs", "source_candidate_lines", "source_scope", "source_variable",
    }
    require(set(linear_result) == linear_keys, "BOUNDARY_LINEAR_RESULT_KEYS")
    for key in ("E_layout", "element0_layout"):
        layout = linear_result[key]
        require(isinstance(layout, Mapping) and set(layout) == native_layout_keys,
                "BOUNDARY_LINEAR_RESULT_LAYOUT", key)
        require(isinstance(layout["shape"], list)
                and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in layout["shape"]),
                "BOUNDARY_LINEAR_RESULT_SHAPE", key)
        require(isinstance(layout["dtype"], str) and layout["dtype"] != "", "BOUNDARY_LINEAR_RESULT_DTYPE", key)
        require(isinstance(layout["strides"], list)
                and len(layout["shape"]) == len(layout["strides"])
                and all(isinstance(item, int) and not isinstance(item, bool) for item in layout["strides"]),
                "BOUNDARY_LINEAR_RESULT_STRIDES", key)
        for flag in ("C", "F", "OWNDATA", "WRITEABLE", "base_ndarray"):
            require(isinstance(layout[flag], bool), "BOUNDARY_LINEAR_RESULT_LAYOUT_BOOL", {"key": key, "field": flag})
        require(isinstance(layout["base_chain_depth"], int)
                and not isinstance(layout["base_chain_depth"], bool)
                and layout["base_chain_depth"] >= 0,
                "BOUNDARY_LINEAR_RESULT_DEPTH", key)
        require(isinstance(layout["pointer"], int)
                and not isinstance(layout["pointer"], bool)
                and layout["pointer"] >= 0,
                "BOUNDARY_LINEAR_RESULT_POINTER", key)
    require(isinstance(linear_result["container_length"], int)
            and not isinstance(linear_result["container_length"], bool), "BOUNDARY_LINEAR_RESULT_LENGTH")
    require(isinstance(linear_result["container_type"], str)
            and isinstance(linear_result["expected_container_kind"], str), "BOUNDARY_LINEAR_RESULT_CONTAINER")
    require(isinstance(linear_result["element0_index"], int)
            and not isinstance(linear_result["element0_index"], bool), "BOUNDARY_LINEAR_RESULT_INDEX")
    for key in ("element0_is_E", "element0_shares_memory_E", "need_energy", "need_profile"):
        require(isinstance(linear_result[key], bool), "BOUNDARY_LINEAR_RESULT_BOOL", key)
    require(isinstance(linear_result["expected_container_length"], int)
            and not isinstance(linear_result["expected_container_length"], bool), "BOUNDARY_LINEAR_RESULT_EXPECTED_LENGTH")
    for key in ("retained_live_bytes", "retained_live_refs"):
        require(isinstance(linear_result[key], int)
                and not isinstance(linear_result[key], bool)
                and linear_result[key] >= 0, "BOUNDARY_LINEAR_RESULT_RETAINED", key)
    require(isinstance(linear_result["source_candidate_lines"], list)
            and all(isinstance(item, int) and not isinstance(item, bool)
                    for item in linear_result["source_candidate_lines"]), "BOUNDARY_LINEAR_RESULT_LINES")
    for key in ("source_scope", "source_variable"):
        require(isinstance(linear_result[key], str), "BOUNDARY_LINEAR_RESULT_SOURCE", key)
    scalar_values = receipt.get("dynamic_scalars")
    require(isinstance(scalar_values, Mapping), "BOUNDARY_DYNAMIC_SCALARS_MAPPING")
    require(all(isinstance(name, str) for name in scalar_values), "BOUNDARY_DYNAMIC_SCALAR_KEY_TYPE")
    source_scalar_order = tuple(scalar_values)
    require(len(source_scalar_order) == len(DYNAMIC_SCALARS) and set(source_scalar_order) == set(DYNAMIC_SCALARS),
            "BOUNDARY_DYNAMIC_SCALAR_SET", source_scalar_order)
    scalar_types = {name: _scalar_kind(scalar_values[name]) for name in source_scalar_order}
    float_names = {"z", "dz_try", "U0_baseline", "U_now"}
    int_names = {"interval_index", "adaptive_rejection_count", "safety_mode_trigger_count"}
    for name in float_names:
        require(scalar_types[name] in {"float", "nonfinite_float"}, "BOUNDARY_DYNAMIC_SCALAR_TYPE", name)
    for name in int_names:
        require(scalar_types[name] == "int", "BOUNDARY_DYNAMIC_SCALAR_TYPE", name)
    require(len(names) + len(source_scalar_order) == EXPECTED_BOUNDARY_FIELDS, "BOUNDARY_TOTAL_FIELD_COUNT")
    scalar_identity = {}
    for name in DYNAMIC_SCALARS:
        source_value = scalar_values[name]
        canonical_value = _safe_to_value(source_value)
        scalar_identity[name] = {
            "source": _scalar_fingerprint(source_value),
            "canonical": _scalar_fingerprint(canonical_value),
            "value_preserved_by_name": _scalar_value_preserved(source_value, canonical_value),
        }
    scalar_values_unchanged = all(row["value_preserved_by_name"] for row in scalar_identity.values())
    require(scalar_values_unchanged, "BOUNDARY_SCALAR_VALUE_CHANGED_BY_STANDARD_VIEW")
    missing_snapshot_members: list[str] = []
    for record in records:
        path = _safe_relative_snapshot(root, record["snapshot_file"])
        if not path.is_file() or path.is_symlink():
            missing_snapshot_members.append(str(path))
            if require_members:
                require(False, "BOUNDARY_SNAPSHOT_MEMBER_MISSING", str(path))
            continue
        if require_members and header_only:
            header = _npy_header(path)
            layout = record["snapshot_layout"]
            require(header["shape"] == [int(item) for item in layout["shape"]], "BOUNDARY_SNAPSHOT_HEADER_SHAPE", record["field_name"])
            require(header["dtype"] == str(layout["dtype"]), "BOUNDARY_SNAPSHOT_HEADER_DTYPE", record["field_name"])
    return {
        "schema": "filament.cpu_s4_3g2.boundary_json_compatibility.v1",
        "status": "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY",
        "receipt_schema": receipt_schema,
        "source_receipt_key_order": [str(name) for name in receipt],
        "source_field_order": names + [f"scalars.{name}" for name in source_scalar_order],
        "canonical_field_order": mapping_names,
        "source_scalar_order": list(source_scalar_order),
        "canonical_scalar_order": list(DYNAMIC_SCALARS),
        "scalar_types": scalar_types,
        "scalar_identity": scalar_identity,
        "scalar_mapping": [
            {
                "name": name,
                "source_position": source_scalar_order.index(name),
                "canonical_position": DYNAMIC_SCALARS.index(name),
                "type": scalar_types[name],
            }
            for name in DYNAMIC_SCALARS
        ],
        "scalar_values_unchanged": scalar_values_unchanged,
        "array_record_count": len(records),
        "diagnostic_record_count": len(diagnostics),
        "snapshot_members_required": bool(require_members),
        "header_only": bool(header_only),
        "missing_snapshot_members": missing_snapshot_members,
        "snapshot_directory": str(root),
        "mapping_identity": dict(mapping_identity),
    }


def _inspect_boundary_input(
    value: Any,
    *,
    require_members: bool,
    header_only: bool,
    expected_receipt_path: str | os.PathLike[str] | None = None,
    expected_raw_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash/parse/validate one input once, returning its receipt and report."""
    source_path = _receipt_source_path(value)
    identity: dict[str, Any] = {}
    if source_path is not None:
        # Bind the receipt bytes before parsing its schema or walking any NPY
        # references. This keeps a stale/incorrect file from reaching headers.
        identity = {
            "receipt_path": str(source_path),
            "receipt_size_bytes": int(source_path.stat().st_size),
            "receipt_raw_sha256": raw_sha256_file(source_path),
            "receipt_hash_scope": "raw_bytes",
        }
    if expected_receipt_path is not None:
        expected = Path(expected_receipt_path).resolve()
        require(source_path is not None and source_path == expected, "BOUNDARY_RECEIPT_PATH_IDENTITY", str(source_path))
    if expected_size_bytes is not None:
        require(isinstance(expected_size_bytes, int) and not isinstance(expected_size_bytes, bool),
                "BOUNDARY_EXPECTED_SIZE_TYPE")
        require(source_path is not None and identity.get("receipt_size_bytes") == expected_size_bytes,
                "BOUNDARY_RECEIPT_SIZE_IDENTITY", identity.get("receipt_size_bytes"))
    if expected_raw_sha256 is not None:
        require(isinstance(expected_raw_sha256, str), "BOUNDARY_EXPECTED_RAW_SHA256_TYPE")
        require(source_path is not None and identity.get("receipt_raw_sha256") == expected_raw_sha256,
                "BOUNDARY_RECEIPT_RAW_IDENTITY", identity.get("receipt_raw_sha256"))
    receipt = _boundary_receipt(value)
    compatibility = _inspect_boundary_metadata(receipt, require_members=require_members, header_only=header_only)
    compatibility.update(identity)
    compatibility["scientific_comparison_status"] = "NOT_RUN_METADATA_ONLY"
    return receipt, compatibility


def inspect_boundary_metadata(
    value: Any,
    *,
    require_members: bool = True,
    header_only: bool = True,
    expected_receipt_path: str | os.PathLike[str] | None = None,
    expected_raw_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> dict[str, Any]:
    """Perform the strict pre-array JSON/NPY metadata gate.

    ``require_members=False`` is for historical receipt-only review when the
    archived snapshot files are not local. It never authorizes a scientific
    comparison; ``load_boundary_capture`` always requires the members.
    """
    _, compatibility = _inspect_boundary_input(
        value,
        require_members=require_members,
        header_only=header_only,
        expected_receipt_path=expected_receipt_path,
        expected_raw_sha256=expected_raw_sha256,
        expected_size_bytes=expected_size_bytes,
    )
    return compatibility


def _profile_without_clock(value: Any) -> Any:
    """Apply the frozen g1_exact profile tree rule, including CLOCK_LEAVES."""
    import numpy as np
    from g1_exact import CLOCK_LEAVES

    if isinstance(value, Mapping):
        return {
            str(key): _profile_without_clock(child)
            for key, child in value.items()
            if str(key) not in CLOCK_LEAVES
        }
    if isinstance(value, (list, tuple)):
        return [_profile_without_clock(child) for child in value]
    if isinstance(value, np.generic):
        return _profile_without_clock(value.item())
    return value


def _decode_profile_array(value: Any) -> list[Any]:
    import numpy as np

    values = np.asarray(value).tolist()
    if isinstance(values, str):
        values = [values]
    require(isinstance(values, list), "HISTORICAL_PROFILE_ARRAY_SCHEMA")
    decoded = []
    try:
        for item in values:
            decoded.append(_profile_without_clock(json.loads(str(item))))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError({"gate": "HISTORICAL_PROFILE_JSON_PARSE"}) from exc
    return decoded


def _load_scalar_diagnostics(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Decode the frozen propagator scalar sidecar without new semantics."""
    from tools.cpu_o1.q1_t0_real import _read_scalar_sidecar

    values = _read_scalar_sidecar(Path(path).resolve(strict=True))
    require(isinstance(values, dict), "HISTORICAL_SCALAR_DIAGNOSTIC_SCHEMA", str(path))
    return {str(name): value for name, value in values.items()}


def load_boundary_capture(value: Any) -> dict[str, Any]:
    """Load an A07 boundary snapshot without copying its arrays into memory."""
    import numpy as np

    # Run the same file-identity/metadata gate used by the public preflight
    # before resolving the receipt again or invoking np.load on any member.
    receipt, compatibility = _inspect_boundary_input(value, require_members=True, header_only=True)
    root = Path(str(receipt["snapshot_directory"])).resolve(strict=True)
    fields: dict[str, Any] = {}
    records = list(receipt.get("arrays", [])) + list(receipt.get("diagnostics", []))
    for record in records:
        name = str(record["field_name"])
        file = root / str(record["snapshot_file"])
        fields[name] = np.load(file, mmap_mode="r", allow_pickle=False)
    # The writer historically emitted sort_keys=True. Read by semantic name,
    # while preserving the original source order in the compatibility receipt.
    scalar_payload = {
        name: _safe_to_value(receipt["dynamic_scalars"][name])
        for name in DYNAMIC_SCALARS
    }
    fields.update({f"scalars.{name}": value for name, value in scalar_payload.items()})
    require(len(fields) == EXPECTED_BOUNDARY_FIELDS, "BOUNDARY_TOTAL_FIELD_COUNT", len(fields))
    return {
        "receipt": receipt,
        "root": root,
        "fields": fields,
        "json_compatibility": compatibility,
        "native_metadata": {
            "layouts": dict(receipt.get("native_layouts", {})),
            "aliases": dict(receipt.get("aliases", {})),
            "pointer_excluded_from_comparison": True,
        },
    }


def _resolve_scalar_path(plan: Mapping[str, Any], task_root: Path | None = None) -> Path:
    candidate = plan.get("recovered_scalars")
    if isinstance(candidate, Mapping):
        candidate = candidate.get("path") or candidate.get("file")
    if candidate:
        path = Path(str(candidate))
        if not path.is_absolute() and task_root is not None:
            path = task_root / path
        if path.is_file():
            return path.resolve()
    roots = [task_root] if task_root is not None else []
    roots.extend([Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]])
    for root in roots:
        path = root / "inputs" / "RECOVERED_FIRST_BOUNDARY_SCALARS.json"
        if path.is_file():
            return path.resolve()
    raise RuntimeError({"gate": "RECOVERED_SCALARS_MISSING"})


def load_recovered_scalars(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    data = _load_json(Path(path).resolve(strict=True))
    if isinstance(data, Mapping):
        data = data.get("recovered") or data.get("recovered_scalars") or data.get("scalars")
    require(isinstance(data, list), "RECOVERED_SCALAR_SCHEMA")
    rows = []
    for row in data:
        require(isinstance(row, Mapping), "RECOVERED_SCALAR_RECORD")
        require("name" in row and "value" in row and "comparison_dtype" in row, "RECOVERED_SCALAR_FIELDS", row)
        rows.append({"name": str(row["name"]), "value": row["value"], "comparison_dtype": str(row["comparison_dtype"])})
    require(tuple(row["name"] for row in rows) == DYNAMIC_SCALARS, "RECOVERED_SCALAR_ORDER")
    return rows


def load_historical_boundary(
    oracle_root: str | os.PathLike[str],
    scalar_path: str | os.PathLike[str],
    *,
    mapping_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Load the old 105-field boundary plus the separately accepted 7 scalars."""
    import numpy as np

    root = Path(oracle_root).resolve(strict=True)
    first = root / "first_boundary"
    require(first.is_dir(), "HISTORICAL_FIRST_BOUNDARY_MISSING", str(first))
    require(mapping_path is not None, "HISTORICAL_BOUNDARY_MAPPING_REQUIRED")
    mapping = Path(mapping_path).resolve(strict=True)
    require(mapping.is_file(), "HISTORICAL_BOUNDARY_MAPPING_MISSING", str(mapping))
    try:
        import csv

        with mapping.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames is not None and "field_name" in reader.fieldnames,
                    "HISTORICAL_BOUNDARY_MAPPING_HEADER", reader.fieldnames)
            mapping_rows = list(reader)
    except (OSError, UnicodeError, csv.Error, RuntimeError) as exc:
        raise RuntimeError({
            "gate": "HISTORICAL_BOUNDARY_MAPPING_PARSE",
            "context": str(mapping),
        }) from exc
    mapped_fields = [str(row.get("field_name", "")) for row in mapping_rows]
    require(
        len(mapped_fields) == EXPECTED_BOUNDARY_FIELDS
        and all(mapped_fields)
        and len(set(mapped_fields)) == EXPECTED_BOUNDARY_FIELDS,
        "HISTORICAL_BOUNDARY_MAPPING_FIELD_SET",
        {"count": len(mapped_fields), "duplicates": sorted({name for name in mapped_fields if mapped_fields.count(name) > 1})},
    )
    expected_non_diagnostics = {
        *SCIENCE_ARRAY_FIELDS,
        *(f"POST.{name}" for name in POST_FIELDS),
        *(f"scalars.{name}" for name in DYNAMIC_SCALARS),
    }
    mapped_diagnostics = [
        name.split(".", 1)[1]
        for name in mapped_fields
        if name.startswith("diagnostics.")
    ]
    require(
        len(mapped_diagnostics) == EXPECTED_BOUNDARY_DIAGNOSTICS
        and len(set(mapped_diagnostics)) == EXPECTED_BOUNDARY_DIAGNOSTICS
        and set(mapped_fields) == expected_non_diagnostics | {f"diagnostics.{name}" for name in mapped_diagnostics},
        "HISTORICAL_BOUNDARY_MAPPING_COMPLETE",
        {"diagnostic_count": len(mapped_diagnostics)},
    )
    fields: dict[str, Any] = {}
    for name in SCIENCE_ARRAY_FIELDS:
        path = first / f"{name}.npy"
        require(path.is_file(), "HISTORICAL_FIRST_BOUNDARY_MEMBER_MISSING", str(path))
        fields[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    with np.load(first / "POST.npz", allow_pickle=False) as archive:
        require(set(POST_FIELDS).issubset(archive.files), "HISTORICAL_POST_FIELDS", archive.files)
        for name in POST_FIELDS:
            fields[f"POST.{name}"] = np.asarray(archive[name])
    with np.load(first / "diagnostics.npz", allow_pickle=False) as archive:
        archive_names = set(archive.files)
        missing = sorted(set(mapped_diagnostics) - archive_names)
        require(not missing, "HISTORICAL_DIAGNOSTIC_MAPPING_MEMBER_MISSING", missing)
        require(len(mapped_diagnostics) == EXPECTED_BOUNDARY_DIAGNOSTICS,
                "HISTORICAL_DIAGNOSTIC_COUNT", len(mapped_diagnostics))
        for name in mapped_diagnostics:
            value = np.asarray(archive[name])
            require(value.dtype.kind in "biufc", "HISTORICAL_DIAGNOSTIC_MAPPING_NONNUMERIC", name)
            fields[f"diagnostics.{name}"] = value
    scalars = load_recovered_scalars(scalar_path)
    for row in scalars:
        fields[f"scalars.{row['name']}"] = row["value"]
    require(len(fields) == EXPECTED_BOUNDARY_FIELDS, "HISTORICAL_BOUNDARY_FIELD_COUNT", len(fields))
    return {
        "root": root,
        "first_boundary": first,
        "fields": fields,
        "scalars": scalars,
        "native_metadata": "UNOBSERVED_IN_ORIGINAL_HISTORY",
        "scalar_source": str(Path(scalar_path).resolve()),
        "mapping_source": str(mapping),
        "mapped_diagnostic_fields": [f"diagnostics.{name}" for name in mapped_diagnostics],
    }


def _comparison_modules():
    from tools.cpu_o1 import q1_t0
    from tools.cpu_o1.numeric_gate import classify_comparison

    return q1_t0, classify_comparison


def _first_failure_path(output: Path) -> Path:
    stem = output.with_suffix("")
    target = stem.with_name(stem.name + ".FIRST_FAILURE.json")
    index = 1
    while target.exists():
        target = stem.with_name(stem.name + f".FIRST_FAILURE_{index:03d}.json")
        index += 1
    return target


def _failure_payload(
    field: str,
    row: Mapping[str, Any],
    gate: Mapping[str, Any],
    candidate: Any,
    identity: Any,
    *,
    failure_index: int | None = None,
) -> dict[str, Any]:
    metrics = row.get("metrics", {}) if isinstance(row, Mapping) else {}
    examples = metrics.get("worst_differences", []) if isinstance(metrics, Mapping) else []
    first_example = examples[0] if isinstance(examples, list) and examples else None
    if isinstance(first_example, Mapping):
        candidate_value = first_example.get("candidate")
        value_index = first_example.get("index")
        reference_value = first_example.get("reference")
    elif hasattr(candidate, "ndim") and int(getattr(candidate, "ndim", 1)) == 0:
        candidate_value = candidate.item()
        value_index = []
        reference_value = None
    else:
        candidate_value = None
        value_index = None
        reference_value = None
    payload = {
        "schema": "filament.cpu_s4_3g2.first_failure.v1",
        "field_name": field,
        "first_failure_index": failure_index,
        "value_index": value_index,
        "reference_value": _json_safe(reference_value),
        "candidate_value": _json_safe(candidate_value),
        "comparison_row": dict(row),
        "classification": dict(gate),
        "candidate_identity": identity,
        "candidate_layout": _array_layout(candidate, pointer=False) if hasattr(candidate, "dtype") and hasattr(candidate, "shape") else None,
    }
    return payload


def compare_historical_boundary(
    historical: Mapping[str, Any],
    current: Mapping[str, Any],
    output: str | os.PathLike[str],
    *,
    candidate_identity: Any = None,
) -> dict[str, Any]:
    """Compare all 105 arrays and 7 recovered scalars with the frozen gate."""
    import numpy as np

    q1_t0, classify_comparison = _comparison_modules()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = dict(historical["fields"])
    candidate = dict(current["fields"])
    rows: list[dict[str, Any]] = []
    first_failure: Path | None = None
    missing = sorted(set(expected) - set(candidate))
    extra = sorted(set(candidate) - set(expected))
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        def emit(record: dict[str, Any]) -> None:
            stream.write(json.dumps(_json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

        if missing or extra:
            record = {
                "field_name": "__field_set__",
                "scope": "science",
                "classification": "NONFINITE_OR_TOPOLOGY_FAILURE",
                "proceed": False,
                "reason": "field_set_mismatch",
                "missing": missing,
                "extra": extra,
            }
            emit(record)
            first_failure = _first_failure_path(target)
            write_json(
                first_failure,
                {
                    "schema": "filament.cpu_s4_3g2.first_failure.v1",
                    **record,
                    "first_failure_index": None,
                    "value_index": None,
                    "reference_value": None,
                    "candidate_value": None,
                    "candidate_layout": None,
                    "candidate_identity": candidate_identity,
                },
            )
        for name in sorted(set(expected) & set(candidate)):
            left = expected[name]
            right = candidate[name]
            if name.startswith("scalars."):
                scalar_name = name.split(".", 1)[1]
                scalar_dtype = next(
                    row["comparison_dtype"]
                    for row in historical["scalars"]
                    if row["name"] == scalar_name
                )
                left = np.asarray(left, dtype=np.dtype(scalar_dtype))
                right = np.asarray(right, dtype=np.dtype(scalar_dtype))
            scope = "diagnostics" if name.startswith("diagnostics.") else "science"
            leaf = name.rsplit(".", 1)[-1]
            row = q1_t0._array_comparison_row(
                leaf,
                left,
                right,
                scope=scope,
                zero_mask_path=target.parent / (target.stem + "_zero_masks") / f"{len(rows):05d}.bin",
            )
            gate = classify_comparison(row)
            record = {
                **row,
                "tree_field": name,
                "classification": gate,
                "reference_layout": _array_layout(left, pointer=False) if isinstance(left, np.ndarray) else None,
                "candidate_layout": _array_layout(right, pointer=False) if isinstance(right, np.ndarray) else None,
                "historical_native_layout": "UNOBSERVED_IN_ORIGINAL_HISTORY",
                "candidate_identity": candidate_identity,
            }
            emit(record)
            rows.append(record)
            if not gate.get("proceed") and first_failure is None:
                first_failure = _first_failure_path(target)
                write_json(
                    first_failure,
                    _failure_payload(
                        name,
                        row,
                        gate,
                        right,
                        candidate_identity,
                        failure_index=len(rows) - 1,
                    ),
                )
                if isinstance(right, np.ndarray):
                    failure_array = first_failure.with_suffix(".npy")
                    with failure_array.open("xb") as stream_array:
                        np.save(stream_array, right, allow_pickle=False)
    failed = bool(first_failure is not None)
    return {
        "status": "PASS_HISTORICAL_BOUNDARY_112_EXACT" if not failed else "FAIL_HISTORICAL_BOUNDARY_NONEXACT",
        "field_count": len(rows) + (1 if missing or extra else 0),
        "array_field_count": EXPECTED_BOUNDARY_ARRAY_FIELDS,
        "scalar_field_count": len(DYNAMIC_SCALARS),
        "rows": rows,
        "first_failure": None if first_failure is None else str(first_failure),
        "historical_native_layout": "UNOBSERVED_IN_ORIGINAL_HISTORY",
        "candidate_identity": candidate_identity,
        "tolerance": "none",
        "nan_policy": "unchanged_numeric_gate",
        "timing_policy": "no_timing_exclusion",
    }


def compare_current_boundary_pair(
    reference: Any,
    candidate: Any,
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    """Use unchanged g1_exact for snapshot values, then compare current metadata."""
    from g1_exact import compare

    left = load_boundary_capture(reference)
    right = load_boundary_capture(candidate)
    output_path = Path(output)
    def boundary_value(capture: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np

        fields = capture["fields"]
        arrays = {
            "E": fields["E"],
            "Qacc": fields["Qacc"],
            "Qacc_raman": fields["Qacc_raman"],
            "POST": {
                name: fields[f"POST.{name}"]
                for name in POST_FIELDS
            },
            "diagnostics": {
                name.split(".", 1)[1]: fields[name]
                for name in fields
                if name.startswith("diagnostics.")
            },
            "scalars": {
                name.split(".", 1)[1]: value
                for name, value in fields.items()
                if name.startswith("scalars.")
            },
        }
        # The frozen snapshot alias list visits array names in sorted order.
        # Preserve scalar leaves as scalars: converting them to 0-D arrays
        # would invent seven array owners and additional alias relations.
        # Only this current-pair candidate is materialized as C/OWNDATA;
        # historical values remain mmap-backed and native checks stay separate.
        def snapshot_value(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    key: snapshot_value(value[key])
                    for key in sorted(value)
                }
            if isinstance(value, np.ndarray):
                return np.array(value, copy=True, order="C")
            return value

        return snapshot_value(arrays)

    value_result = compare(left["root"], boundary_value(right), output_path, disk=True)
    left_native = {
        "arrays": left["native_metadata"].get("layouts", {}),
        "aliases": left["native_metadata"].get("aliases", {}),
    }
    right_native = {
        "arrays": right["native_metadata"].get("layouts", {}),
        "aliases": right["native_metadata"].get("aliases", {}),
    }
    native_result = compare_native_metadata(left_native, right_native)
    require(native_result["status"] == "PASS_CURRENT_NATIVE_LAYOUT_ALIAS_EXACT", "CURRENT_BOUNDARY_NATIVE_LAYOUT_ALIAS", native_result)
    return {
        "status": "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT",
        "value": value_result,
        "native": native_result,
        "historical_native_layout": "UNOBSERVED_IN_ORIGINAL_HISTORY",
    }


def compare_full_output(reference_output: Any, bundle: Any, output: str | os.PathLike[str]) -> dict[str, Any]:
    """Delegate terminal/current pair comparison to the frozen G1 comparator."""
    from g1_exact import compare

    return compare(Path(reference_output).resolve(strict=True), bundle, Path(output), disk=True)


def historical_terminal_records(oracle_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Read historical terminal files for a separate, optional audit path."""
    import numpy as np

    root = Path(oracle_root).resolve(strict=True)
    propagation = root / "propagation"
    require(propagation.is_dir(), "HISTORICAL_PROPAGATION_MISSING", str(propagation))
    result: dict[str, Any] = {
        "final_E": np.load(propagation / "final_E.npy", mmap_mode="r", allow_pickle=False),
        "diagnostics": {},
        "scalar_diagnostics": {},
        "POST": {},
        "sinks": {},
    }
    with np.load(propagation / "final_POST.npz", allow_pickle=False) as archive:
        result["POST"] = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(propagation / "diagnostics.npz", allow_pickle=False) as archive:
        # Match the original terminal writer's normalized_return contract:
        # numeric diagnostics use the frozen numeric gate.  The two profile
        # JSON sidecars are retained separately below and are not part of the
        # original 235-row endpoint comparison contract.
        for name in archive.files:
            value = np.asarray(archive[name])
            if value.dtype.kind in "biufc":
                result["diagnostics"][name] = value
    sidecar = propagation / "RAW_LINEAR_PROFILE_SIDECARS.json"
    if sidecar.is_file():
        payload = _load_json(sidecar)
        require(isinstance(payload, Mapping) and isinstance(payload.get("raw_profile_sidecars"), Mapping),
                "HISTORICAL_PROFILE_SIDECAR_SCHEMA", str(sidecar))
        for name, row in payload["raw_profile_sidecars"].items():
            if name in PROFILE_DIAGNOSTIC_FIELDS and name not in result["diagnostics"]:
                require(isinstance(row, Mapping) and "original_values" in row,
                        "HISTORICAL_PROFILE_SIDECAR_SCHEMA", name)
                result.setdefault("profile_sidecars", {})[name] = _decode_profile_array(row["original_values"])
    # The accepted historical writer used a separate profile JSON file rather
    # than the newer RAW_LINEAR_PROFILE_SIDECARS envelope.  Read that exact
    # legacy file when the diagnostic archive does not carry the profile rows;
    # do not synthesize or infer missing terminal fields.
    legacy_profile = propagation / "linear_halfstep_profiles.json"
    if legacy_profile.is_file():
        payload = _load_json(legacy_profile)
        require(isinstance(payload, Mapping), "HISTORICAL_PROFILE_SIDECAR_SCHEMA", str(legacy_profile))
        for source_name, target_name in (
            ("halfstep1_json", PROFILE_DIAGNOSTIC_FIELDS[0]),
            ("halfstep2_json", PROFILE_DIAGNOSTIC_FIELDS[1]),
        ):
            if target_name in result["diagnostics"]:
                continue
            require(source_name in payload, "HISTORICAL_PROFILE_SIDECAR_MEMBER_MISSING", source_name)
            result.setdefault("profile_sidecars", {})[target_name] = _decode_profile_array(payload[source_name])
    scalar_path = root / "scientific_scalar_diagnostics.json"
    require(scalar_path.is_file(), "HISTORICAL_SCALAR_DIAGNOSTICS_MISSING", str(scalar_path))
    result["scalar_diagnostics"] = _load_scalar_diagnostics(scalar_path)
    result["scalar_source"] = str(scalar_path)
    for path in sorted(propagation.glob("sinks.*.npy")):
        result["sinks"][path.name] = np.load(path, mmap_mode="r", allow_pickle=False)
    return result


def current_terminal_fields(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten current returned/POST/sink values for optional historical audit."""
    import numpy as np

    returned = bundle["returned"]
    result: dict[str, Any] = {
        "final_E": returned[0],
        "scientific_Qacc": returned[1],
        "scalar_diagnostics": {},
    }
    diagnostics = returned[2]
    for name, value in diagnostics.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
            result[f"diagnostics.{name}"] = value
    from tools.cpu_o1.q1_t0 import _numeric_scalar_diagnostics

    _, scalar_values = _numeric_scalar_diagnostics(dict(diagnostics))
    result["scalar_diagnostics"] = {str(name): value for name, value in scalar_values.items()}
    post = bundle.get("POST", {})
    if post:
        first = next(iter(post.values()))
        for name in first:
            result[f"POST.{name}"] = np.stack([post[key][name] for key in sorted(post, key=lambda item: int(item))], axis=0)
    for name, value in bundle.get("sinks", {}).items():
        result[f"sinks.{name}"] = value
    return result


def compare_terminal_records(
    historical: Mapping[str, Any],
    bundle: Mapping[str, Any],
    output: str | os.PathLike[str],
    *,
    candidate_identity: Any = None,
) -> dict[str, Any]:
    """Optional exact field comparison against the old terminal raw files."""
    import numpy as np

    q1_t0, classify_comparison = _comparison_modules()
    expected: dict[str, Any] = {
        "final_E": historical["final_E"],
        "scientific_Qacc": historical.get("scientific_Qacc"),
    }
    expected.update({f"POST.{name}": value for name, value in historical.get("POST", {}).items()})
    expected.update({f"diagnostics.{name}": value for name, value in historical.get("diagnostics", {}).items()})
    expected.update({f"sinks.{name}": value for name, value in historical.get("sinks", {}).items()})
    actual = current_terminal_fields(bundle)
    expected_scalars = {str(name): value for name, value in historical.get("scalar_diagnostics", {}).items()}
    actual_scalars = {str(name): value for name, value in actual.pop("scalar_diagnostics", {}).items()}
    scalar_names = set(expected_scalars) | set(actual_scalars)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    first_failure: Path | None = None
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        for index, name in enumerate(sorted(set(expected) | set(actual) | scalar_names)):
            if name in scalar_names:
                if name not in expected_scalars or name not in actual_scalars:
                    failures.append(name)
                    record = {
                        "field_name": name,
                        "classification": "NONFINITE_OR_TOPOLOGY_FAILURE",
                        "scope": "diagnostics",
                        "diagnostic_source": "private_propagator_return_diag_scalar",
                        "proceed": False,
                    }
                    stream.write(json.dumps(_json_safe({**record, "candidate_identity": candidate_identity}), allow_nan=False) + "\n")
                    if first_failure is None:
                        first_failure = _first_failure_path(target)
                        write_json(first_failure, {
                            "schema": "filament.cpu_s4_3g2.first_failure.v1",
                            **record,
                            "first_failure_index": index,
                            "value_index": None,
                            "reference_value": None,
                            "candidate_value": None,
                            "candidate_layout": None,
                            "candidate_identity": candidate_identity,
                        })
                    continue
                left, right = expected_scalars[name], actual_scalars[name]
                from tools.cpu_p1r2 import cpu_p1r2_j1 as j1

                scope = "timing" if name in set(j1.TIMING_DIAGNOSTIC_FIELDS) else "diagnostics"
                row = q1_t0._array_comparison_row(
                    name,
                    np.asarray(left),
                    np.asarray(right),
                    scope=scope,
                    zero_mask_path=target.parent / (target.stem + "_zero_masks") / f"{index:05d}.bin",
                )
                gate = classify_comparison(row)
                record = {
                    **row,
                    "tree_field": name,
                    "classification": gate,
                    "diagnostic_source": "private_propagator_return_diag_scalar",
                    "candidate_identity": candidate_identity,
                }
                stream.write(json.dumps(_json_safe(record), allow_nan=False) + "\n")
                if not gate.get("proceed"):
                    failures.append(name)
                    if first_failure is None:
                        first_failure = _first_failure_path(target)
                        write_json(first_failure, _failure_payload(
                            name, row, gate, np.asarray(right), candidate_identity, failure_index=index
                        ))
                continue
            if name not in expected or name not in actual:
                failures.append(name)
                record = {
                    "field_name": name,
                    "classification": "NONFINITE_OR_TOPOLOGY_FAILURE",
                    "proceed": False,
                }
                stream.write(json.dumps(_json_safe({**record, "candidate_identity": candidate_identity}), allow_nan=False) + "\n")
                if first_failure is None:
                    first_failure = _first_failure_path(target)
                    write_json(
                        first_failure,
                        {
                            "schema": "filament.cpu_s4_3g2.first_failure.v1",
                            **record,
                            "first_failure_index": index,
                            "value_index": None,
                            "reference_value": None,
                            "candidate_value": None,
                            "candidate_layout": None,
                            "candidate_identity": candidate_identity,
                        },
                    )
                continue
            left, right = expected[name], actual[name]
            from tools.cpu_p1r2 import cpu_p1r2_j1 as j1

            leaf = name.rsplit(".", 1)[-1]
            scope = (
                "timing" if name.startswith("diagnostics.") and leaf in set(j1.TIMING_DIAGNOSTIC_FIELDS)
                else ("diagnostics" if name.startswith("diagnostics.") else "science")
            )
            row = q1_t0._array_comparison_row(name.rsplit(".", 1)[-1], left, right, scope=scope,
                zero_mask_path=target.parent / (target.stem + "_zero_masks") / f"{index:05d}.bin")
            gate = classify_comparison(row)
            stream.write(json.dumps(_json_safe({**row, "tree_field": name, "classification": gate, "candidate_identity": candidate_identity}), allow_nan=False) + "\n")
            if not gate.get("proceed"):
                failures.append(name)
                if first_failure is None:
                    first_failure = _first_failure_path(target)
                    write_json(
                        first_failure,
                        _failure_payload(
                            name,
                            row,
                            gate,
                            right,
                            candidate_identity,
                            failure_index=index,
                        ),
                    )
                    if isinstance(right, np.ndarray):
                        with first_failure.with_suffix(".npy").open("xb") as stream_array:
                            np.save(stream_array, right, allow_pickle=False)
    return {
        "status": "PASS_HISTORICAL_TERMINAL_EXACT" if not failures else "FAIL_HISTORICAL_TERMINAL_NONEXACT",
        "failed_fields": failures,
        "comparison_row_count": len(expected) + len(scalar_names),
        "scientific_field_count": 5,
        "diagnostic_array_field_count": len(historical.get("diagnostics", {})),
        "scalar_diagnostic_field_count": len(expected_scalars),
        "sink_field_count": len(historical.get("sinks", {})),
        "profile_sidecars": sorted(historical.get("profile_sidecars", {})),
        "first_failure": None if first_failure is None else str(first_failure),
        "candidate_identity": candidate_identity,
        "historical_native_layout": "UNOBSERVED_IN_ORIGINAL_HISTORY",
    }


__all__ = [
    "SCIENCE_ARRAY_FIELDS",
    "POST_FIELDS",
    "DYNAMIC_SCALARS",
    "EXPECTED_BOUNDARY_FIELDS",
    "EXPECTED_BOUNDARY_ARRAY_FIELDS",
    "EXPECTED_BOUNDARY_DIAGNOSTICS",
    "BOUNDARY_RECEIPT_SCHEMA",
    "EXPECTED_TERMINAL_DIAGNOSTIC_ARRAYS",
    "EXPECTED_TERMINAL_SCALARS",
    "EXPECTED_TERMINAL_COMPARISON_ROWS",
    "PROFILE_DIAGNOSTIC_FIELDS",
    "native_metadata",
    "native_aliases",
    "compare_native_metadata",
    "file_records",
    "save_file_index",
    "inspect_boundary_metadata",
    "load_boundary_capture",
    "load_recovered_scalars",
    "load_historical_boundary",
    "compare_historical_boundary",
    "compare_current_boundary_pair",
    "compare_full_output",
    "historical_terminal_records",
    "compare_terminal_records",
    "write_json",
    "require",
]
