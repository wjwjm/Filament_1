"""Strict, array-free admission checks for the C full-size mechanism result."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping

from tools.hpc_ops import provenance_v2 as pv

from .c_fullsize_protocol import CAPTURE_INTERVALS, CAPTURE_TOTAL_CALLS, EVIDENCE_SCHEMA, FORMAL_CALLS, RESOURCES, RESULT_SCHEMA, SHAPE, WARMUP_CALLS


class MechanismEvidenceError(RuntimeError):
    """The C full-size mechanism evidence failed closed."""


REQUIRED_SOURCE_PATHS = (
    "Filament_python/tools/cpu_s4_6f_fullsize_parallel/c_fullsize_entry.py",
    "Filament_python/tools/cpu_s4_6f_fullsize_parallel/c_fullsize_evidence.py",
    "Filament_python/tools/cpu_s4_6f_fullsize_parallel/c_fullsize_hpc.py",
    "Filament_python/tools/cpu_s4_6f_fullsize_parallel/c_fullsize_protocol.py",
    "Filament_python/tools/cpu_s4_6f_fullsize_parallel/c_fullsize_cpu.sbatch",
    "Filament_python/tools/cpu_s4_6f_fullsize_parallel/entry.py",
    "Filament_python/tools/cpu_s4_6f_fullsize_parallel/hpc.py",
    "Filament_python/tools/cpu_s4_6f_fullsize_parallel/protocol.py",
    "Filament_python/tools/cpu_s4_6f_fullsize_parallel/measurement.py",
    "Filament_python/tools/cpu_s4_6f_fullsize_parallel/role_spec.py",
    "Filament_python/tools/cpu_s4_6e_array_propagation/c_candidate.py",
    "Filament_python/tools/cpu_s4_6e_array_propagation/c_capture.py",
    "Filament_python/tools/cpu_s4_6e_array_propagation/c_instrument.py",
    "Filament_python/tools/cpu_s4_6e_array_propagation/c_mechanism.py",
    "Filament_python/tools/cpu_s4_6e_array_propagation/c_pool.py",
    "Filament_python/tools/cpu_s4_6e_array_propagation/measurement.py",
)


def _require(ok: bool, reason: str, detail: Any = None) -> None:
    if not ok:
        suffix = "" if detail is None else f":{detail}"
        raise MechanismEvidenceError(reason + suffix)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MechanismEvidenceError(f"S46F_C_MECHANISM_JSON:{path.name}") from exc
    _require(isinstance(value, dict), "S46F_C_MECHANISM_JSON_OBJECT", path.name)
    return value


def _raw_record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {
        "path": str(path),
        "classification": "external",
        "hash_scope": "raw_bytes",
        "size_bytes": int(path.stat().st_size),
        "raw_sha256": pv.raw_sha256_file(path),
    }


def _safe_child(root: Path, relative: str) -> Path:
    path = (root / str(relative)).resolve()
    _require(path.is_relative_to(root.resolve()), "S46F_C_MECHANISM_PATH_ESCAPE", relative)
    return path


def _source_identities(source_manifest: Mapping[str, Any], repo: Path | None) -> dict[str, str]:
    rows = source_manifest.get("records")
    _require(isinstance(rows, list), "S46F_C_MECHANISM_SOURCE_RECORDS")
    by_path = {str(row.get("path")): row for row in rows if isinstance(row, Mapping)}
    result: dict[str, str] = {}
    for relative in REQUIRED_SOURCE_PATHS:
        row = by_path.get(relative)
        _require(isinstance(row, Mapping), "S46F_C_MECHANISM_SOURCE_MISSING", relative)
        _require(row.get("classification") == "tracked_text" and row.get("hash_scope") == pv.TRACKED_HASH_SCOPE, "S46F_C_MECHANISM_SOURCE_SCOPE", relative)
        digest = str(row.get("canonical_lf_sha256", ""))
        _require(bool(digest), "S46F_C_MECHANISM_SOURCE_DIGEST", relative)
        result[relative] = digest
        if repo is not None:
            path = (repo / relative).resolve(strict=True)
            _require(path.is_relative_to(repo.resolve()), "S46F_C_MECHANISM_SOURCE_CONFINEMENT", relative)
            _require(pv.canonical_lf_sha256(path.read_bytes()) == digest, "S46F_C_MECHANISM_SOURCE_CHANGED", relative)
    return result


def _gain_map(result: Mapping[str, Any]) -> dict[str, float]:
    mechanism = result.get("mechanism", result)
    _require(isinstance(mechanism, Mapping), "S46F_C_MECHANISM_RESULT_SECTION")
    _require(int(mechanism.get("formal_calls", -1)) == FORMAL_CALLS and int(mechanism.get("warmup_calls", -1)) == WARMUP_CALLS, "S46F_C_MECHANISM_CALL_COUNTS")
    _require(mechanism.get("exact_by_config") == {"REF": True, "P1": True, "P24": True}, "S46F_C_MECHANISM_EXACT_CONFIGS")
    rows = mechanism.get("formal_rows")
    _require(isinstance(rows, list) and len(rows) == FORMAL_CALLS, "S46F_C_MECHANISM_FORMAL_ROWS")
    expected_order = {"G1": ["REF", "P1", "P24"], "G2": ["P24", "P1", "REF"]}
    _require(mechanism.get("config_order") == expected_order, "S46F_C_MECHANISM_CONFIG_ORDER")
    values: dict[tuple[str, str, int], list[float]] = {}
    repeats: dict[tuple[str, str, int], set[int]] = {}
    for row in rows:
        _require(isinstance(row, Mapping) and row.get("exact") is True, "S46F_C_MECHANISM_FORMAL_NONEXACT")
        group = str(row.get("group"))
        config = str(row.get("config"))
        interval = int(row.get("interval_index", -1))
        repeat = int(row.get("repeat", -1))
        seconds = float(row.get("seconds", float("nan")))
        _require(group in expected_order and config in ("REF", "P1", "P24") and interval in CAPTURE_INTERVALS and repeat in range(1, 6), "S46F_C_MECHANISM_FORMAL_IDENTITY")
        _require(math.isfinite(seconds) and seconds > 0.0, "S46F_C_MECHANISM_FORMAL_CLOCK")
        key = (group, config, interval)
        values.setdefault(key, []).append(seconds)
        repeats.setdefault(key, set()).add(repeat)
    gains: dict[str, float] = {}
    reported = mechanism.get("p24_block_gains")
    expected_keys = {f"{group}:{interval}" for group in ("G1", "G2") for interval in CAPTURE_INTERVALS}
    _require(isinstance(reported, Mapping) and set(reported) == expected_keys, "S46F_C_MECHANISM_SIX_GAIN_BLOCKS")
    for group in ("G1", "G2"):
        for interval in CAPTURE_INTERVALS:
            for config in ("REF", "P1", "P24"):
                key_tuple = (group, config, interval)
                _require(len(values.get(key_tuple, ())) == 5 and repeats.get(key_tuple) == {1, 2, 3, 4, 5}, "S46F_C_MECHANISM_BLOCK_REPEAT_SET", f"{group}:{config}:{interval}")
            ref = statistics.median(values[(group, "REF", interval)])
            candidate = statistics.median(values[(group, "P24", interval)])
            gain = 1.0 - candidate / ref
            key = f"{group}:{interval}"
            _require(math.isfinite(gain) and gain >= 0.05, "S46F_C_MECHANISM_P24_GAIN_GATE", key)
            _require(math.isclose(float(reported[key]), gain, rel_tol=0.0, abs_tol=1e-12), "S46F_C_MECHANISM_REPORTED_GAIN_MISMATCH", key)
            gains[key] = gain
    return gains


def _capture_receipt(root: Path) -> dict[str, Any]:
    capture = _read(_safe_child(root, "CAPTURE_RECEIPT.json"))
    _require(capture.get("shape") == list(SHAPE), "S46F_C_MECHANISM_CAPTURE_SHAPE")
    _require(capture.get("selected_interval_indices") == list(CAPTURE_INTERVALS) and int(capture.get("expected_total_calls", -1)) == CAPTURE_TOTAL_CALLS, "S46F_C_MECHANISM_CAPTURE_INTERVALS")
    metadata = capture.get("capture_metadata")
    _require(isinstance(metadata, Mapping) and metadata.get("selected_interval_indices") == list(CAPTURE_INTERVALS) and int(metadata.get("expected_total_calls", -1)) == CAPTURE_TOTAL_CALLS and int(metadata.get("record_count", -1)) == len(CAPTURE_INTERVALS), "S46F_C_MECHANISM_CAPTURE_METADATA")
    records = metadata.get("records")
    _require(isinstance(records, Mapping) and set(records) == {str(item) for item in CAPTURE_INTERVALS}, "S46F_C_MECHANISM_CAPTURE_FINGERPRINT_INDEX")
    for row in records.values():
        _require(isinstance(row, Mapping) and isinstance(row.get("input_fingerprints"), Mapping) and set(row["input_fingerprints"]) == {"E", "phase", "alpha", "dn_gas"} and isinstance(row.get("output_fingerprint"), Mapping), "S46F_C_MECHANISM_CAPTURE_FINGERPRINTS")
    _require(capture.get("uncaptured_terminal_status") == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS", "S46F_C_MECHANISM_REFERENCE_TERMINAL")
    _require(capture.get("captured_terminal_status") == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS", "S46F_C_MECHANISM_CAPTURED_TERMINAL")
    _require(bool(capture.get("historical_reference_output")) and bool(capture.get("captured_reference_output")) and capture.get("captured_reference_is_uncaptured_output") is True, "S46F_C_MECHANISM_REFERENCE_RELATION")
    _require(capture.get("cpu_diagnostics_profile") is True and capture.get("diagnostic_outputs") is True and capture.get("fine_performance_profile") is False, "S46F_C_MECHANISM_DIAGNOSTIC_CLOCK_CONTRACT")
    _require(capture.get("historical_terminal_status") == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS", "S46F_C_MECHANISM_HISTORICAL_TERMINAL")
    files = capture.get("raw_files")
    _require(isinstance(files, list) and files, "S46F_C_MECHANISM_CAPTURE_RAW_FILES")
    for record in files:
        _require(isinstance(record, Mapping) and record.get("classification") == "external" and record.get("hash_scope") == "raw_bytes" and record.get("raw_sha256") and int(record.get("size_bytes", -1)) > 0, "S46F_C_MECHANISM_CAPTURE_RAW_RECORD")
    package_manifest = capture.get("capture_package_manifest")
    _require(isinstance(package_manifest, Mapping) and package_manifest.get("classification") == "external" and package_manifest.get("hash_scope") == "raw_bytes" and package_manifest.get("raw_sha256") and int(package_manifest.get("size_bytes", -1)) > 0, "S46F_C_MECHANISM_CAPTURE_PACKAGE_RAW_RECORD")
    package_path = Path(str(package_manifest.get("path"))).resolve(strict=True)
    _require(pv.raw_sha256_file(package_path) == package_manifest.get("raw_sha256") and int(package_path.stat().st_size) == int(package_manifest.get("size_bytes")), "S46F_C_MECHANISM_CAPTURE_PACKAGE_RAW_IDENTITY")
    package = _read(package_path)
    _require(package.get("schema") == "filament.cpu_s4_6e.c_capture.package.v1" and package.get("metadata", {}).get("selected_interval_indices") == list(CAPTURE_INTERVALS), "S46F_C_MECHANISM_CAPTURE_PACKAGE_SCHEMA")
    source = capture.get("source_binding")
    _require(isinstance(source, Mapping) and source.get("role") == "R" and source.get("source_identity_digest"), "S46F_C_MECHANISM_CAPTURE_SOURCE_BINDING")
    return capture


def validate_mechanism_evidence(evidence_root: str | Path, *, repo: str | Path | None = None, expected_execution_commit: str | None = None) -> dict[str, Any]:
    """Validate the lightweight full-size evidence without loading scientific arrays."""

    root = Path(evidence_root).resolve(strict=True)
    def receipt_path(name: str) -> Path:
        terminal = root / (Path(name).stem + "_TERMINAL.json")
        return terminal if terminal.is_file() else _safe_child(root, name)

    admission_path = receipt_path("ADMISSION.json")
    scheduler_path = receipt_path("SCHEDULER_ACCOUNTING.json")
    resource_path = receipt_path("RESOURCE_RECEIPT.json")
    admission = _read(admission_path)
    _require(admission.get("schema") == EVIDENCE_SCHEMA and admission.get("status") == "PASS_C_FULLSIZE_MECHANISM_EXACT_GAIN_GATE", "S46F_C_MECHANISM_ADMISSION_STATUS")
    _require(admission.get("shape") == list(SHAPE) and admission.get("selected_interval_indices") == list(CAPTURE_INTERVALS) and int(admission.get("expected_total_calls", -1)) == CAPTURE_TOTAL_CALLS, "S46F_C_MECHANISM_ADMISSION_DOMAIN")
    execution = _read(_safe_child(root, "EXECUTION.json"))
    source_manifest = _read(_safe_child(root, "EXECUTION_SOURCE_MANIFEST.json"))
    input_contract = _read(_safe_child(root, "INPUT_CONTRACT.json"))
    full_manifest = _read(_safe_child(root, "FULL_INPUT_MANIFEST.json"))
    reference_contract = _read(_safe_child(root, "REFERENCE_INPUT_RAW_RECORDS.json"))
    _require(input_contract.get("schema") == "filament.cpu_s4_6f.input_contract.v1" and input_contract.get("classification") == "external" and input_contract.get("hash_scope") == "raw_bytes", "S46F_C_MECHANISM_INPUT_CONTRACT_SCHEMA")
    _require(input_contract.get("shape") == list(SHAPE) and input_contract.get("dtype") == "complex128" and input_contract.get("native_layout") == "C", "S46F_C_MECHANISM_INPUT_DOMAIN")
    _require(full_manifest.get("schema") == "filament.cpu_s4_6f.full_input_manifest.v1" and full_manifest.get("classification") == "external" and full_manifest.get("hash_scope") == "raw_bytes", "S46F_C_MECHANISM_FULL_MANIFEST_SCHEMA")
    _require(full_manifest.get("shape") == list(SHAPE) and full_manifest.get("dtype") == "complex128" and full_manifest.get("native_layout") == "C", "S46F_C_MECHANISM_FULL_MANIFEST_DOMAIN")
    fingerprints = input_contract.get("expected_input_fingerprints")
    _require(isinstance(fingerprints, Mapping) and full_manifest.get("fingerprints") == fingerprints, "S46F_C_MECHANISM_INPUT_FINGERPRINT_BINDING")
    canonical_hash = input_contract.get("canonical_manifest_raw_sha256")
    _require(isinstance(canonical_hash, str) and bool(canonical_hash) and full_manifest.get("canonical_manifest_raw_sha256") == canonical_hash == execution.get("canonical_manifest_raw_sha256"), "S46F_C_MECHANISM_CANONICAL_MANIFEST_BINDING")
    _require(reference_contract.get("schema") == "filament.cpu_s4_5a.reference_files.v1" and reference_contract.get("hash_scope") == "raw_bytes", "S46F_C_MECHANISM_REFERENCE_SCHEMA")
    reference = reference_contract.get("reference", {})
    _require(reference.get("shape") == list(SHAPE) and int(reference.get("intervals", -1)) == 8 and reference.get("role") == "L1_LINEAR_FACTOR_CACHE", "S46F_C_MECHANISM_REFERENCE_DOMAIN")
    for key in ("output", "boundary_path", "stages"):
        text = str(reference.get(key, ""))
        _require(text.startswith("/") or Path(text).is_absolute(), "S46F_C_MECHANISM_REFERENCE_PATH", key)
    reference_records = reference_contract.get("records")
    if reference_records is not None:
        _require(isinstance(reference_records, list) and reference_records, "S46F_C_MECHANISM_REFERENCE_RECORDS")
        for row in reference_records:
            _require(isinstance(row, Mapping) and row.get("classification") == "external" and row.get("hash_scope") == "raw_bytes" and row.get("path") and row.get("raw_sha256") and int(row.get("size_bytes", -1)) >= 0, "S46F_C_MECHANISM_REFERENCE_RECORD_IDENTITY")
    execution_commit = str(execution.get("execution_commit", ""))
    _require(execution_commit and execution_commit == source_manifest.get("head"), "S46F_C_MECHANISM_EXECUTION_SOURCE_BINDING")
    if expected_execution_commit is not None:
        _require(execution_commit == str(expected_execution_commit), "S46F_C_MECHANISM_EXECUTION_COMMIT")
    _require(admission.get("execution_commit") == execution_commit, "S46F_C_MECHANISM_ADMISSION_EXECUTION")
    source_raw = _raw_record(root / "EXECUTION_SOURCE_MANIFEST.json")
    _require(admission.get("source_manifest_raw_sha256") == source_raw["raw_sha256"], "S46F_C_MECHANISM_SOURCE_MANIFEST_RAW")
    for name, field in (("INPUT_CONTRACT.json", "input_contract_raw_sha256"), ("FULL_INPUT_MANIFEST.json", "full_input_manifest_raw_sha256"), ("REFERENCE_INPUT_RAW_RECORDS.json", "reference_contract_raw_sha256")):
        _require(admission.get(field) == pv.raw_sha256_file(root / name), "S46F_C_MECHANISM_INPUT_RAW_IDENTITY", name)
    identities = _source_identities(source_manifest, None if repo is None else Path(repo).resolve(strict=True))
    result = _read(_safe_child(root, "RESULT.json"))
    _require(result.get("schema") == RESULT_SCHEMA and result.get("status") == "PASS_C_FULLSIZE_MECHANISM_EXACT_GAIN_GATE", "S46F_C_MECHANISM_RESULT_STATUS")
    gains = _gain_map(result)
    _require(admission.get("p24_block_gains") == gains, "S46F_C_MECHANISM_ADMISSION_GAIN_REPLAY")
    capture = _capture_receipt(root)
    scheduler = _read(scheduler_path)
    _require(scheduler.get("scheduler_terminal_state") == "COMPLETED" and scheduler.get("exit_code") == "0:0", "S46F_C_MECHANISM_SCHEDULER_TERMINAL")
    _require(int(scheduler.get("allocated_cpus", -1)) == int(RESOURCES["physical_cpus"]) and str(scheduler.get("requested_memory")) in ("128G", "128GiB") and int(scheduler.get("gpus", 0)) == 0, "S46F_C_MECHANISM_SCHEDULER_RESOURCES")
    resource = _read(resource_path)
    _require(resource.get("physical_cpus") == RESOURCES["physical_cpus"] and resource.get("memory_GiB") == RESOURCES["memory_GiB"] and resource.get("gpus") == 0 and resource.get("terminal") is True, "S46F_C_MECHANISM_RESOURCE_RECEIPT")
    return {
        "schema": EVIDENCE_SCHEMA + ".admission.v1",
        "status": "PASS_C_FULLSIZE_MECHANISM_ADMISSION",
        "execution_commit": execution_commit,
        "source_manifest_raw_sha256": source_raw["raw_sha256"],
        "source_identities": identities,
        "formal_calls": FORMAL_CALLS,
        "warmup_calls": WARMUP_CALLS,
        "all_configs_exact": True,
        "p24_block_gains": gains,
        "selected_interval_indices": list(CAPTURE_INTERVALS),
        "capture": capture,
        "scheduler": {key: scheduler.get(key) for key in ("job_id", "scheduler_terminal_state", "exit_code", "actual_runtime_s", "actual_allocated_core_hours", "allocated_cpus", "requested_memory", "gpus")},
        "verified_files": [_raw_record(path) for path in (admission_path, root / "EXECUTION.json", root / "EXECUTION_SOURCE_MANIFEST.json", root / "INPUT_CONTRACT.json", root / "FULL_INPUT_MANIFEST.json", root / "REFERENCE_INPUT_RAW_RECORDS.json", root / "RESULT.json", scheduler_path, resource_path, root / "CAPTURE_RECEIPT.json")],
        "scientific_arrays_loaded": 0,
    }


def _configured_root(report: Path) -> Path:
    execution = _read(report / "EXECUTION.json")
    configured = execution.get("c_fullsize_mechanism_evidence_root") or os.environ.get("S46F_C_FULLSIZE_MECHANISM_EVIDENCE")
    _require(configured, "S46F_C_FULLSIZE_MECHANISM_EVIDENCE_ROOT_REQUIRED")
    path = Path(str(configured))
    return (path if path.is_absolute() else report / path).resolve(strict=True)


def require_fullsize_mechanism_admission(report: str | Path, *, repo: str | Path | None = None) -> dict[str, Any]:
    report_path = Path(report).resolve(strict=True)
    root = _configured_root(report_path)
    execution = _read(report_path / "EXECUTION.json")
    expected = execution.get("c_fullsize_mechanism_execution_commit")
    result = validate_mechanism_evidence(root, repo=repo, expected_execution_commit=expected)
    expected_admission_sha = execution.get("c_fullsize_mechanism_admission_raw_sha256")
    if expected_admission_sha:
        admission_candidates = [root / "ADMISSION.json"]
        if (root / "ADMISSION_TERMINAL.json").is_file():
            admission_candidates.append(root / "ADMISSION_TERMINAL.json")
        _require(any(pv.raw_sha256_file(path) == expected_admission_sha for path in admission_candidates), "S46F_C_MECHANISM_ADMISSION_RAW_IDENTITY")
    return result


__all__ = ["MechanismEvidenceError", "REQUIRED_SOURCE_PATHS", "require_fullsize_mechanism_admission", "validate_mechanism_evidence"]
