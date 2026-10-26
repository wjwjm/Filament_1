"""Narrow bridge for the frozen historical PRE storage mismatch.

The historical science comparator remains in :mod:`g2_hpc_exact`.  This
module only verifies the explicitly authorised storage-protocol difference
after that comparator has emitted its original raw rows and first-failure
payload.  No tolerance, NaN, timing, or scientific comparison rule is
implemented here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


TASK = "CPU-S4-3G2-HPC-REAL8-APPLICATION-20261023"
CONTRACT_SCHEMA = "filament.g2.memory_streaming_storage_bridge.v1"
BRIDGE_STATUS = "PASS_HISTORICAL_SCIENCE_EXACT_STORAGE_BRIDGE"
RAW_STATUS = "FAIL_HISTORICAL_TERMINAL_NONEXACT"
FIELD = "hr3b_state_disk_backed"
HISTORICAL_SCHEMA = "filament.cpu_p1r2.j1_diagnostic_pre_memory.v1"
HISTORICAL_STORAGE = "in_memory_copy"
CURRENT_SCHEMA = "khz_filament.hr4e5.e5_1a.streaming_pre.v1"
CURRENT_FILENAME = "streaming_manifest.json"
OLD_SCALAR_SHA256 = "7276759b49a8ec136e17c6375ee1454f5d3d4bfff9937ab33e2e0b5ba9df3e2a"
OLD_PRE_RECEIPT_SHA256 = "dc80c5570bdeb8b5093d8af96d99a0da6230ddb947f5b3e9cd8f7b6a21acf64a"
AUTHORIZATION_SHA256 = "c9113505dab04b849ec87a01539942e5b3012121363c764ec65d241ea3fbaf69"
AUTHORIZATION_SIZE_BYTES = 62
EXPECTED_ROWS = 241
PRE_FIELDS = ("delta_n", "vx", "vy")
PRE_INTERVALS = 8


def _require(condition: bool, reason: str, context: Any = None) -> None:
    if not condition:
        raise RuntimeError({"gate": str(reason), "context": context})


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def _raw_sha256(path: str | Path, pv: Any = None) -> str:
    if pv is None:
        from tools.hpc_ops import provenance_v2 as pv_module

        pv = pv_module
    return str(pv.raw_sha256_file(Path(path)))


def _json_safe(value: Any) -> Any:
    """Convert metadata containers without changing scalar meaning."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _json_safe(value.item())
    except ImportError:  # pragma: no cover - runtime environments provide NumPy
        pass
    return value


def _contract_value(contract: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = contract.get(key, default)
    return value


def validate_contract(contract: Mapping[str, Any], *, expected_task: str = TASK) -> dict[str, Any]:
    """Validate the immutable contract's science and protocol declarations."""
    _require(isinstance(contract, Mapping), "STORAGE_BRIDGE_CONTRACT_MAPPING")
    _require(contract.get("schema") == CONTRACT_SCHEMA, "STORAGE_BRIDGE_CONTRACT_SCHEMA", contract.get("schema"))
    _require(contract.get("task") == expected_task, "STORAGE_BRIDGE_CONTRACT_TASK", contract.get("task"))
    _require(contract.get("field") == FIELD, "STORAGE_BRIDGE_FIELD", contract.get("field"))
    _require(contract.get("dtype") == "bool", "STORAGE_BRIDGE_DTYPE", contract.get("dtype"))
    _require(contract.get("shape") == [], "STORAGE_BRIDGE_SHAPE", contract.get("shape"))
    _require(contract.get("historical_value") is False and contract.get("current_value") is True,
             "STORAGE_BRIDGE_VALUES", {"historical": contract.get("historical_value"), "current": contract.get("current_value")})
    _require(contract.get("historical_schema") == HISTORICAL_SCHEMA, "STORAGE_BRIDGE_HISTORICAL_SCHEMA")
    _require(contract.get("historical_storage") == HISTORICAL_STORAGE, "STORAGE_BRIDGE_HISTORICAL_STORAGE")
    _require(contract.get("current_schema") == CURRENT_SCHEMA, "STORAGE_BRIDGE_CURRENT_SCHEMA")
    _require(contract.get("current_filename") == CURRENT_FILENAME, "STORAGE_BRIDGE_CURRENT_FILENAME")
    authorization = contract.get("authorization")
    _require(isinstance(authorization, Mapping)
             and authorization.get("classification") == "external"
             and authorization.get("hash_scope") == "raw_bytes"
             and authorization.get("raw_sha256") == AUTHORIZATION_SHA256
             and authorization.get("size_bytes") == AUTHORIZATION_SIZE_BYTES,
             "STORAGE_BRIDGE_AUTHORIZATION_BINDING", authorization)
    _require(contract.get("science_comparator_modified") is False, "STORAGE_BRIDGE_COMPARATOR_MODIFIED")
    _require(contract.get("old_oracle_modified") is False, "STORAGE_BRIDGE_ORACLE_MODIFIED")
    _require(contract.get("new_timing_or_nan_exclusions") is False, "STORAGE_BRIDGE_GATE_RULE_MODIFIED")
    _require(contract.get("raw_comparator", "").startswith("Unchanged compare_terminal_records"),
             "STORAGE_BRIDGE_RAW_COMPARATOR_DECLARATION")
    return dict(contract)


def load_contract(path: str | Path, *, expected_task: str = TASK) -> dict[str, Any]:
    target = Path(path).resolve(strict=True)
    return validate_contract(_read_json(target), expected_task=expected_task)


def metadata_snapshot(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return an immutable, JSON-safe copy for a receipt."""
    _require(isinstance(metadata, Mapping), "STORAGE_PROTOCOL_METADATA_MAPPING", type(metadata).__name__)
    return _json_safe(dict(metadata))


def _shape(value: Any, reason: str) -> list[int]:
    _require(isinstance(value, (list, tuple)), reason, value)
    result = []
    for item in value:
        _require(isinstance(item, int) and not isinstance(item, bool), reason, value)
        result.append(int(item))
    _require(bool(result) and all(item > 0 for item in result), reason, value)
    return result


def _require_bool(value: Any, reason: str) -> bool:
    _require(type(value) is bool, reason, {"type": type(value).__name__, "value": value})
    return bool(value)


def current_pre_field_identity(
    fields: Mapping[str, Any],
    *,
    intervals: int = PRE_INTERVALS,
) -> dict[str, Any]:
    """Capture canonical identities for the actual current PRE fields.

    The caller supplies the fields read from the fresh streaming CURRENT
    generation.  This records the source-side PRE identity during the
    application span; the historical binding is checked later, after the
    application clock has stopped.
    """
    import numpy as np
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1

    _require(isinstance(fields, Mapping), "CURRENT_PRE_FIELDS_MAPPING", type(fields).__name__)
    _require(set(fields) == set(PRE_FIELDS), "CURRENT_PRE_FIELDS_SET", sorted(fields))
    count = int(intervals)
    _require(count == PRE_INTERVALS, "CURRENT_PRE_INTERVAL_COUNT", count)
    arrays: dict[str, Any] = {}
    expected_shape: tuple[int, ...] | None = None
    for name in PRE_FIELDS:
        value = np.asarray(fields[name])
        _require(value.dtype == np.dtype("float64"), "CURRENT_PRE_FIELD_DTYPE", {"field": name, "dtype": str(value.dtype)})
        _require(value.ndim == 3 and int(value.shape[0]) == count,
                 "CURRENT_PRE_FIELD_SHAPE", {"field": name, "shape": list(value.shape)})
        _require(value.flags.c_contiguous, "CURRENT_PRE_FIELD_CONTIGUOUS", name)
        _require(bool(np.isfinite(value).all()), "CURRENT_PRE_FIELD_FINITE", name)
        if expected_shape is None:
            expected_shape = tuple(int(item) for item in value.shape)
        _require(tuple(int(item) for item in value.shape) == expected_shape,
                 "CURRENT_PRE_FIELD_SHAPE_MATCH", {"field": name, "shape": list(value.shape), "expected": list(expected_shape)})
        arrays[name] = value
    _require(expected_shape is not None, "CURRENT_PRE_FIELD_EMPTY")
    return {
        "status": "PASS_CURRENT_PRE_FIELD_IDENTITY",
        "field_names": list(PRE_FIELDS),
        "field_canonical_sha256": {name: str(j1._canonical_hash(arrays[name])) for name in PRE_FIELDS},
        "dtype": "float64",
        "shape": list(expected_shape),
        "intervals": count,
        "source": "fresh StreamingLifecycle CURRENT artifacts",
    }


def validate_pre_field_binding(
    historical_metadata: Mapping[str, Any],
    current_pre_identity: Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind every historical PRE canonical hash to the current actual PRE."""
    if contract is not None:
        validate_contract(contract)
    historical = validate_historical_metadata(historical_metadata, contract)
    _require(isinstance(current_pre_identity, Mapping), "CURRENT_PRE_IDENTITY_MAPPING")
    _require(current_pre_identity.get("dtype") == "float64", "CURRENT_PRE_IDENTITY_DTYPE", current_pre_identity.get("dtype"))
    _require(current_pre_identity.get("intervals") == PRE_INTERVALS,
             "CURRENT_PRE_IDENTITY_INTERVALS", current_pre_identity.get("intervals"))
    current_shape = _shape(current_pre_identity.get("shape"), "CURRENT_PRE_IDENTITY_SHAPE")
    historical_shape = _shape(historical.get("hr3b_state_shape"), "HISTORICAL_PRE_IDENTITY_SHAPE")
    _require(current_shape == historical_shape and current_shape[0] == PRE_INTERVALS,
             "PRE_FIELD_SHAPE_BINDING", {"historical": historical_shape, "current": current_shape})
    expected = historical.get("field_canonical_sha256")
    actual = current_pre_identity.get("field_canonical_sha256")
    _require(isinstance(expected, Mapping) and set(expected) == set(PRE_FIELDS),
             "HISTORICAL_PRE_FIELD_HASHES", expected)
    _require(isinstance(actual, Mapping) and set(actual) == set(PRE_FIELDS),
             "CURRENT_PRE_FIELD_HASHES", actual)
    _require(all(isinstance(actual.get(name), str) and len(actual[name]) == 64
                  and all(char in "0123456789abcdef" for char in actual[name])
                  for name in PRE_FIELDS),
             "CURRENT_PRE_FIELD_HASH_FORMAT", actual)
    mismatches = {
        name: {"historical": str(expected.get(name)), "current": str(actual.get(name))}
        for name in PRE_FIELDS
        if str(expected.get(name)) != str(actual.get(name))
    }
    _require(not mismatches, "PRE_FIELD_CANONICAL_HASH_BINDING", mismatches)
    return {
        "status": "PASS_PRE_FIELD_CANONICAL_BINDING",
        "field_names": list(PRE_FIELDS),
        "historical": {name: str(expected[name]) for name in PRE_FIELDS},
        "current": {name: str(actual[name]) for name in PRE_FIELDS},
        "dtype": "float64",
        "shape": current_shape,
        "intervals": PRE_INTERVALS,
    }


def validate_current_metadata(
    metadata: Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate metadata returned directly by the live Streaming PRE view."""
    if contract is not None:
        validate_contract(contract)
    snapshot = metadata_snapshot(metadata)
    _require(snapshot.get("hr3b_state_schema") == (contract or {}).get("current_schema", CURRENT_SCHEMA),
             "CURRENT_STORAGE_PROTOCOL_SCHEMA", snapshot)
    _require(snapshot.get("hr3b_state_filename") == (contract or {}).get("current_filename", CURRENT_FILENAME),
             "CURRENT_STORAGE_PROTOCOL_FILENAME", snapshot)
    _require(_require_bool(snapshot.get("hr3b_state_disk_backed"), "CURRENT_STORAGE_PROTOCOL_BOOL") is True,
             "CURRENT_STORAGE_PROTOCOL_NOT_DISK_BACKED", snapshot)
    _require(snapshot.get("hr3b_state_dtype") == "float64", "CURRENT_STORAGE_PROTOCOL_DTYPE", snapshot)
    shape = _shape(snapshot.get("hr3b_state_shape"), "CURRENT_STORAGE_PROTOCOL_SHAPE")
    _require(len(shape) == 3 and shape[0] == PRE_INTERVALS,
             "CURRENT_STORAGE_PROTOCOL_INTERVALS", shape)
    return snapshot


def validate_historical_metadata(
    metadata: Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the old PRE metadata carried by the sealed child receipt."""
    if contract is not None:
        validate_contract(contract)
    snapshot = metadata_snapshot(metadata)
    _require(snapshot.get("hr3b_state_schema") == (contract or {}).get("historical_schema", HISTORICAL_SCHEMA),
             "HISTORICAL_STORAGE_PROTOCOL_SCHEMA", snapshot)
    _require(snapshot.get("storage") == (contract or {}).get("historical_storage", HISTORICAL_STORAGE),
             "HISTORICAL_STORAGE_PROTOCOL_STORAGE", snapshot)
    _require(_require_bool(snapshot.get("hr3b_state_disk_backed"), "HISTORICAL_STORAGE_PROTOCOL_BOOL") is False,
             "HISTORICAL_STORAGE_PROTOCOL_NOT_IN_MEMORY", snapshot)
    _require(snapshot.get("hr3b_state_dtype") == "float64", "HISTORICAL_STORAGE_PROTOCOL_DTYPE", snapshot)
    shape = _shape(snapshot.get("hr3b_state_shape"), "HISTORICAL_STORAGE_PROTOCOL_SHAPE")
    _require(len(shape) == 3 and shape[0] == PRE_INTERVALS,
             "HISTORICAL_STORAGE_PROTOCOL_INTERVALS", shape)
    hashes = snapshot.get("field_canonical_sha256")
    _require(isinstance(hashes, Mapping) and set(hashes) == set(PRE_FIELDS),
             "HISTORICAL_PRE_FIELD_HASHES", hashes)
    _require(all(isinstance(hashes.get(name), str) and len(hashes[name]) == 64
                  and all(char in "0123456789abcdef" for char in hashes[name])
                  for name in PRE_FIELDS),
             "HISTORICAL_PRE_FIELD_HASH_FORMAT", hashes)
    return snapshot


def validate_protocol_pair(
    historical_metadata: Mapping[str, Any],
    current_metadata: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate both protocol identities independently before bridging a row."""
    validate_contract(contract)
    historical = validate_historical_metadata(historical_metadata, contract)
    current = validate_current_metadata(current_metadata, contract)
    return {
        "status": "PASS_STORAGE_PROTOCOL_PAIR",
        "field": FIELD,
        "dtype": "bool",
        "shape": [],
        "historical_value": False,
        "current_value": True,
        "historical": historical,
        "current": current,
        "historical_protocol_source": "sealed historical propagation/child_receipt.json PRE_metadata",
        "current_protocol_source": "live StreamingPulseReadView.metadata()",
    }


def verify_historical_artifacts(
    oracle_root: str | Path,
    contract: Mapping[str, Any],
    *,
    pv: Any = None,
) -> dict[str, Any]:
    """Bind the historical scalar sidecar, PRE receipt, and OUTPUT_SEAL."""
    validate_contract(contract)
    root = Path(oracle_root).resolve(strict=True)
    scalar = root / "scientific_scalar_diagnostics.json"
    receipt = root / "propagation" / "child_receipt.json"
    seal_path = root / "OUTPUT_SEAL.json"
    _require(scalar.is_file(), "HISTORICAL_SCALAR_SIDECAR_MISSING", str(scalar))
    _require(receipt.is_file(), "HISTORICAL_PRE_RECEIPT_MISSING", str(receipt))
    _require(seal_path.is_file(), "HISTORICAL_OUTPUT_SEAL_MISSING", str(seal_path))
    scalar_sha = _raw_sha256(scalar, pv)
    receipt_sha = _raw_sha256(receipt, pv)
    _require(scalar_sha == contract.get("old_oracle_scalar_sha256") == OLD_SCALAR_SHA256,
             "HISTORICAL_SCALAR_SIDECAR_RAW_BINDING", scalar_sha)
    _require(receipt_sha == contract.get("original_PRE_receipt_sha256") == OLD_PRE_RECEIPT_SHA256,
             "HISTORICAL_PRE_RECEIPT_RAW_BINDING", receipt_sha)
    seal = _read_json(seal_path)
    _require(isinstance(seal, Mapping) and seal.get("schema") == "filament.cpu_s4.output_seal.v1",
             "HISTORICAL_OUTPUT_SEAL_SCHEMA", seal.get("schema") if isinstance(seal, Mapping) else type(seal).__name__)
    files = seal.get("files")
    _require(isinstance(files, Mapping), "HISTORICAL_OUTPUT_SEAL_FILES")
    _require(files.get("scientific_scalar_diagnostics.json") == scalar_sha,
             "HISTORICAL_OUTPUT_SEAL_SCALAR_BINDING", files.get("scientific_scalar_diagnostics.json"))
    _require(files.get("propagation/child_receipt.json") == receipt_sha,
             "HISTORICAL_OUTPUT_SEAL_PRE_RECEIPT_BINDING", files.get("propagation/child_receipt.json"))
    receipt_payload = _read_json(receipt)
    _require(isinstance(receipt_payload, Mapping), "HISTORICAL_PRE_RECEIPT_SCHEMA")
    metadata = receipt_payload.get("PRE_metadata")
    _require(isinstance(metadata, Mapping), "HISTORICAL_PRE_METADATA_MISSING")
    historical_protocol = validate_historical_metadata(metadata, contract)
    return {
        "oracle_root": str(root),
        "scalar_sidecar": {"path": str(scalar), "raw_sha256": scalar_sha, "size_bytes": scalar.stat().st_size},
        "pre_receipt": {"path": str(receipt), "raw_sha256": receipt_sha, "size_bytes": receipt.stat().st_size},
        "output_seal": {
            "path": str(seal_path),
            "raw_sha256": _raw_sha256(seal_path, pv),
            "corresponding_hashes": {
                "scientific_scalar_diagnostics.json": files.get("scientific_scalar_diagnostics.json"),
                "propagation/child_receipt.json": files.get("propagation/child_receipt.json"),
            },
        },
        "historical_metadata": historical_protocol,
        "receipt_payload_schema": receipt_payload.get("schema"),
    }


def _raw_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError({"gate": "STORAGE_BRIDGE_RAW_ROW_PARSE", "context": {"line": line_number, "path": str(path)}}) from exc
        _require(isinstance(row, Mapping), "STORAGE_BRIDGE_RAW_ROW_MAPPING", {"line": line_number, "path": str(path)})
        rows.append(dict(row))
    return rows


def _classification_proceeds(row: Mapping[str, Any]) -> bool:
    classification = row.get("classification")
    if isinstance(classification, Mapping):
        return classification.get("proceed") is True
    return classification in ("METADATA_EXACT", "TIMING_EXCLUDED")


def validate_raw_terminal_result(
    raw_result: Mapping[str, Any],
    raw_path: str | Path,
    historical_metadata: Mapping[str, Any],
    current_metadata: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    pv: Any = None,
) -> dict[str, Any]:
    """Validate the unchanged comparator result and its one expected gap."""
    validate_contract(contract)
    protocol = validate_protocol_pair(historical_metadata, current_metadata, contract)
    target = Path(raw_path).resolve(strict=True)
    rows = _raw_rows(target)
    _require(raw_result.get("status") == RAW_STATUS, "STORAGE_BRIDGE_RAW_STATUS", raw_result.get("status"))
    _require(raw_result.get("comparison_row_count") == EXPECTED_ROWS, "STORAGE_BRIDGE_RAW_ROW_COUNT", raw_result.get("comparison_row_count"))
    _require(len(rows) == EXPECTED_ROWS, "STORAGE_BRIDGE_RAW_FILE_ROW_COUNT", len(rows))
    failed = raw_result.get("failed_fields")
    _require(failed == [FIELD], "STORAGE_BRIDGE_UNIQUE_FAILED_FIELD", failed)
    named = [str(row.get("field_name")) for row in rows]
    _require(named.count(FIELD) == 1, "STORAGE_BRIDGE_STORAGE_ROW_COUNT", named.count(FIELD))
    storage_row = next(row for row in rows if row.get("field_name") == FIELD)
    metrics = storage_row.get("metrics")
    _require(isinstance(metrics, Mapping), "STORAGE_BRIDGE_STORAGE_ROW_METRICS")
    _require(metrics.get("reference_dtype") == "bool" and metrics.get("candidate_dtype") == "bool",
             "STORAGE_BRIDGE_STORAGE_ROW_DTYPE", metrics)
    _require(metrics.get("reference_shape") == [] and metrics.get("candidate_shape") == [],
             "STORAGE_BRIDGE_STORAGE_ROW_SHAPE", metrics)
    _require(metrics.get("all_finite") is True and metrics.get("value_equal") is False
             and metrics.get("canonical_exact") is False and metrics.get("bitwise_equal") is False,
             "STORAGE_BRIDGE_STORAGE_ROW_NONEXACT_KIND", metrics)
    _require(not _classification_proceeds(storage_row), "STORAGE_BRIDGE_STORAGE_ROW_MUST_NONEXACT", storage_row.get("classification"))
    for row in rows:
        if row is not storage_row:
            _require(_classification_proceeds(row), "STORAGE_BRIDGE_OTHER_RAW_ROW_NONEXACT", row.get("field_name"))
    first_failure_value = raw_result.get("first_failure")
    _require(first_failure_value not in (None, ""), "STORAGE_BRIDGE_FIRST_FAILURE_MISSING")
    first_failure = Path(str(first_failure_value)).resolve(strict=True)
    payload = _read_json(first_failure)
    _require(isinstance(payload, Mapping), "STORAGE_BRIDGE_FIRST_FAILURE_SCHEMA")
    _require(payload.get("field_name") == FIELD, "STORAGE_BRIDGE_FIRST_FAILURE_FIELD", payload.get("field_name"))
    for key in ("first_failure_index", "value_index", "reference_value", "candidate_value", "candidate_layout", "candidate_identity"):
        _require(key in payload, "STORAGE_BRIDGE_FIRST_FAILURE_PAYLOAD", key)
    return {
        "protocol": protocol,
        "storage_row": dict(storage_row),
        "raw_rows": rows,
        "raw_first_failure": {
            "path": str(first_failure),
            "raw_sha256": _raw_sha256(first_failure, pv),
            "payload": dict(payload),
        },
    }


def bridge_terminal_comparison(
    raw_result: Mapping[str, Any],
    raw_path: str | Path,
    historical_metadata: Mapping[str, Any],
    current_metadata: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    pv: Any = None,
    historical_artifacts: Mapping[str, Any] | None = None,
    current_pre_field_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a bridge receipt while retaining the raw nonexact comparison."""
    _require(isinstance(historical_artifacts, Mapping), "STORAGE_BRIDGE_HISTORICAL_ARTIFACTS_REQUIRED")
    _require(
        historical_artifacts.get("scalar_sidecar", {}).get("raw_sha256") == OLD_SCALAR_SHA256
        and historical_artifacts.get("pre_receipt", {}).get("raw_sha256") == OLD_PRE_RECEIPT_SHA256,
        "STORAGE_BRIDGE_HISTORICAL_ARTIFACT_BINDING",
        historical_artifacts,
    )
    validation = validate_raw_terminal_result(
        raw_result,
        raw_path,
        historical_metadata,
        current_metadata,
        contract,
        pv=pv,
    )
    pre_binding = validate_pre_field_binding(
        historical_metadata,
        current_pre_field_identity,
        contract,
    )
    target = Path(raw_path).resolve(strict=True)
    return {
        "status": BRIDGE_STATUS,
        "science_status": "PASS_HISTORICAL_SCIENCE_EXACT",
        "raw_status": raw_result.get("status"),
        "failed_fields": list(raw_result.get("failed_fields", [])),
        "raw_failed_fields": list(raw_result.get("failed_fields", [])),
        "comparison_row_count": int(raw_result.get("comparison_row_count", -1)),
        "raw_comparison_row_count": int(raw_result.get("comparison_row_count", -1)),
        "diagnostic_array_field_count": raw_result.get("diagnostic_array_field_count"),
        "scalar_diagnostic_field_count": raw_result.get("scalar_diagnostic_field_count"),
        "sink_field_count": raw_result.get("sink_field_count"),
        "profile_sidecars": raw_result.get("profile_sidecars", []),
        "historical_native_layout": raw_result.get("historical_native_layout", "UNOBSERVED_IN_ORIGINAL_HISTORY"),
        "raw_comparison": {
            "path": str(target),
            "raw_sha256": _raw_sha256(target, pv),
            "status": raw_result.get("status"),
            "failed_fields": list(raw_result.get("failed_fields", [])),
            "first_failure": raw_result.get("first_failure"),
            "row_count": len(validation["raw_rows"]),
            "records_preserved": True,
        },
        "first_failure": validation["raw_first_failure"],
        "storage_bridge": {
            "field": FIELD,
            "dtype": "bool",
            "shape": [],
            "historical_value": False,
            "current_value": True,
            "unique_expected_gap": True,
            "protocol": validation["protocol"],
            "pre_field_binding": pre_binding,
            "storage_row": validation["storage_row"],
        },
        "historical_artifacts": None if historical_artifacts is None else dict(historical_artifacts),
        "raw_nonexact_preserved": True,
        "science_comparator_modified": False,
        "old_oracle_modified": False,
        "new_timing_or_nan_exclusions": False,
    }


__all__ = [
    "BRIDGE_STATUS",
    "RAW_STATUS",
    "FIELD",
    "PRE_FIELDS",
    "PRE_INTERVALS",
    "EXPECTED_ROWS",
    "load_contract",
    "validate_contract",
    "metadata_snapshot",
    "validate_current_metadata",
    "validate_historical_metadata",
    "current_pre_field_identity",
    "validate_pre_field_binding",
    "validate_protocol_pair",
    "verify_historical_artifacts",
    "validate_raw_terminal_result",
    "bridge_terminal_comparison",
]
