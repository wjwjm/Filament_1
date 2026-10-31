"""Read-only admission checks for the frozen C mechanism result."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import shutil
import statistics
from typing import Any, Mapping

from tools.hpc_ops import provenance_v2 as pv

from .c_formal14_protocol import BASE, MECHANISM_EXECUTION, SHAPE


REQUIRED_EXPORT_FILES = (
    "PLAN.json",
    "HPC_EXECUTION_PLAN.json",
    "EXECUTION.json",
    "EXECUTION_SOURCE_MANIFEST.json",
    "INPUT_CONTRACT.json",
    "REFERENCE_INPUT_RAW_RECORDS.json",
    "RESOURCE_AUTHORIZATION.json",
    "SUBMISSION_PREFLIGHT.json",
    "SUBMISSION_LOCK.json",
    "SUBMISSION_RECEIPT.json",
    "NODE_GATE.json",
    "BENCHMARK_INPUT_MANIFEST.json",
    "RESULT.json",
    "SCHEDULER_ACCOUNTING.json",
    "SCALAR_REPLAY.json",
)
SOURCE_IDENTITY_PATHS = (
    "Filament_python/tools/cpu_s4_6e_array_propagation/c_candidate.py",
    "Filament_python/tools/cpu_s4_6e_array_propagation/c_instrument.py",
    "Filament_python/tools/cpu_s4_6e_array_propagation/c_pool.py",
    "Filament_python/tools/cpu_s4_6e_array_propagation/c_mechanism.py",
    "Filament_python/tools/cpu_s4_6e_array_propagation/c_capture.py",
    "Filament_python/tools/cpu_s4_6e_array_propagation/measurement.py",
)


class MechanismAdmissionError(RuntimeError):
    """The frozen mechanism export failed the C14 admission contract."""


def _require(ok: bool, reason: str, detail: Any = None) -> None:
    if not ok:
        suffix = "" if detail is None else f":{detail}"
        raise MechanismAdmissionError(reason + suffix)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MechanismAdmissionError(f"C14_EVIDENCE_JSON:{path.name}") from exc
    _require(isinstance(value, dict), "C14_EVIDENCE_JSON_OBJECT", path.name)
    return value


def _within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    _require(path.is_relative_to(root), "C14_EVIDENCE_PATH_ESCAPE", relative)
    return path


def _records_by_path(export: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records = export.get("records")
    _require(isinstance(records, list), "C14_EVIDENCE_EXPORT_RECORDS")
    result: dict[str, dict[str, Any]] = {}
    for row in records:
        _require(isinstance(row, dict), "C14_EVIDENCE_EXPORT_RECORD")
        relative = str(row.get("path", ""))
        _require(relative and not Path(relative).is_absolute(), "C14_EVIDENCE_EXPORT_RELATIVE_PATH", relative)
        _require(relative not in result, "C14_EVIDENCE_EXPORT_DUPLICATE_PATH", relative)
        result[relative] = row
    return result


def _verify_export_files(root: Path, records: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    for relative, record in records.items():
        path = _within(root, relative)
        _require(path.is_file(), "C14_EVIDENCE_EXPORT_FILE_MISSING", relative)
        _require(record.get("classification") == "external" and record.get("hash_scope") == "raw_bytes", "C14_EVIDENCE_EXPORT_HASH_SCOPE", relative)
        size = int(path.stat().st_size)
        raw_sha = pv.raw_sha256_file(path)
        _require(size == int(record.get("size_bytes", -1)) and raw_sha == record.get("raw_sha256"), "C14_EVIDENCE_EXPORT_FILE_HASH", relative)
        verified[relative] = {
            "path": relative,
            "classification": "external",
            "hash_scope": "raw_bytes",
            "size_bytes": size,
            "raw_sha256": raw_sha,
        }
    return verified


def _gain_map(result: Mapping[str, Any]) -> dict[str, float]:
    mechanism = result.get("mechanism")
    _require(isinstance(mechanism, Mapping), "C14_MECHANISM_RESULT_SECTION")
    rows = mechanism.get("formal_rows")
    _require(isinstance(rows, list) and len(rows) == 90, "C14_MECHANISM_FORMAL_ROWS")
    config_order = mechanism.get("config_order")
    _require(config_order == {"G1": ["REF", "P1", "P24"], "G2": ["P24", "P1", "REF"]}, "C14_MECHANISM_CONFIG_ORDER")
    selected = (0, 3, 7)
    values: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        _require(isinstance(row, Mapping) and row.get("exact") is True, "C14_MECHANISM_FORMAL_NONEXACT")
        group = str(row.get("group"))
        config = str(row.get("config"))
        interval = int(row.get("interval_index", -1))
        repeat = int(row.get("repeat", -1))
        _require(group in ("G1", "G2") and config in ("REF", "P1", "P24") and interval in selected and repeat in range(1, 6), "C14_MECHANISM_FORMAL_ROW_IDENTITY")
        seconds = float(row.get("seconds"))
        _require(math.isfinite(seconds) and seconds > 0.0, "C14_MECHANISM_FORMAL_CLOCK")
        values.setdefault((group, config, interval), []).append(seconds)
    for group in ("G1", "G2"):
        for config in ("REF", "P1", "P24"):
            for interval in selected:
                _require(len(values.get((group, config, interval), ())) == 5, "C14_MECHANISM_FORMAL_ROW_COUNT", f"{group}:{config}:{interval}")
    reported = mechanism.get("p24_block_gains")
    _require(isinstance(reported, Mapping) and set(reported) == {"G1:0", "G1:3", "G1:7", "G2:0", "G2:3", "G2:7"}, "C14_MECHANISM_SIX_P24_BLOCKS")
    output: dict[str, float] = {}
    for group in ("G1", "G2"):
        for interval in selected:
            reference_median = statistics.median(values[(group, "REF", interval)])
            candidate_median = statistics.median(values[(group, "P24", interval)])
            gain = 1.0 - candidate_median / reference_median
            key = f"{group}:{interval}"
            _require(math.isfinite(gain) and gain >= 0.05, "C14_MECHANISM_P24_GAIN_GATE", key)
            _require(math.isclose(float(reported[key]), gain, rel_tol=0.0, abs_tol=1e-12), "C14_MECHANISM_REPORTED_GAIN_MISMATCH", key)
            output[key] = gain
    return output


def _verify_source_identity(source_manifest: Mapping[str, Any], repo: Path | None) -> dict[str, str]:
    records = source_manifest.get("records")
    _require(isinstance(records, list), "C14_MECHANISM_SOURCE_RECORDS")
    by_path = {str(row.get("path")): row for row in records if isinstance(row, dict)}
    identities: dict[str, str] = {}
    for relative in SOURCE_IDENTITY_PATHS:
        row = by_path.get(relative)
        _require(isinstance(row, dict), "C14_MECHANISM_SOURCE_IDENTITY_MISSING", relative)
        _require(row.get("classification") == "tracked_text" and row.get("hash_scope") == pv.TRACKED_HASH_SCOPE, "C14_MECHANISM_SOURCE_IDENTITY_SCOPE", relative)
        digest = str(row.get("canonical_lf_sha256", ""))
        _require(bool(digest), "C14_MECHANISM_SOURCE_IDENTITY_DIGEST", relative)
        identities[relative] = digest
        if repo is not None:
            path = (repo / relative).resolve(strict=True)
            _require(path.is_relative_to(repo), "C14_MECHANISM_SOURCE_IDENTITY_CONFINEMENT", relative)
            _require(pv.canonical_lf_sha256(path.read_bytes()) == digest, "C14_MECHANISM_SOURCE_IDENTITY_CHANGED", relative)
    if repo is not None:
        for relative, row in by_path.items():
            if relative.startswith("Filament_python/KHz_filament/"):
                path = (repo / relative).resolve(strict=True)
                _require(pv.canonical_lf_sha256(path.read_bytes()) == row.get("canonical_lf_sha256"), "C14_MECHANISM_SCIENCE_SOURCE_CHANGED", relative)
    return identities


def validate_mechanism_evidence(evidence_root: str | Path, *, repo: str | Path | None = None) -> dict[str, Any]:
    """Validate the exported mechanism without opening scientific arrays."""

    root = Path(evidence_root).resolve(strict=True)
    export_path = root / "EXPORT_MANIFEST.json"
    export = _read(export_path)
    _require(export.get("schema") == "filament.cpu_s46e.c_scalar_export.v1", "C14_MECHANISM_EXPORT_SCHEMA")
    records = _records_by_path(export)
    for relative in REQUIRED_EXPORT_FILES:
        _require(relative in records, "C14_MECHANISM_REQUIRED_EXPORT_FILE", relative)
    verified_files = _verify_export_files(root, records)
    execution = _read(root / "EXECUTION.json")
    _require(execution.get("schema") == "filament.cpu_s4_6e_c.execution.v1", "C14_MECHANISM_EXECUTION_SCHEMA")
    _require(execution.get("execution_commit") == MECHANISM_EXECUTION and execution.get("baseline_commit") == "1c9f599d4f7fb0426374997981d0c52ffeaab1aa", "C14_MECHANISM_EXECUTION_IDENTITY")
    _require(export.get("execution_commit") == MECHANISM_EXECUTION, "C14_MECHANISM_EXPORT_EXECUTION_IDENTITY")
    _require(records["EXECUTION.json"].get("raw_sha256") == pv.raw_sha256_file(root / "EXECUTION.json"), "C14_MECHANISM_EXECUTION_HASH")
    source_manifest = _read(root / "EXECUTION_SOURCE_MANIFEST.json")
    _require(source_manifest.get("head") == MECHANISM_EXECUTION, "C14_MECHANISM_SOURCE_HEAD")
    _require(records["EXECUTION_SOURCE_MANIFEST.json"].get("raw_sha256") == pv.raw_sha256_file(root / "EXECUTION_SOURCE_MANIFEST.json"), "C14_MECHANISM_SOURCE_MANIFEST_HASH")
    source_identities = _verify_source_identity(source_manifest, None if repo is None else Path(repo).resolve(strict=True))
    result = _read(root / "RESULT.json")
    _require(result.get("schema") == "filament.cpu_s4_6e_c.result.v1", "C14_MECHANISM_RESULT_SCHEMA")
    _require(result.get("status") == "C_CAPTURE_AND_MECHANISM_EXACT_GATE_PASS_MANUAL_REVIEW_PENDING" and result.get("production_qualified") is False, "C14_MECHANISM_RESULT_STATUS")
    mechanism = result.get("mechanism")
    _require(isinstance(mechanism, Mapping), "C14_MECHANISM_RESULT_SECTION")
    _require(int(mechanism.get("formal_calls", -1)) == 90 and int(mechanism.get("warmup_calls", -1)) == 36, "C14_MECHANISM_CALL_COUNTS")
    _require(mechanism.get("mechanism_gate") is True and mechanism.get("exact_by_config") == {"REF": True, "P1": True, "P24": True}, "C14_MECHANISM_EXACT_GATE")
    gains = _gain_map(result)
    scalar = _read(root / "SCALAR_REPLAY.json")
    _require(scalar.get("status") == "PASS_EXACT_SCALAR_RECOMPUTATION" and int(scalar.get("formal_calls", -1)) == 90 and int(scalar.get("warmup_calls", -1)) == 36 and scalar.get("mechanism_gate") is True, "C14_MECHANISM_SCALAR_REPLAY")
    replay_gains = scalar.get("p24_block_gains")
    _require(isinstance(replay_gains, Mapping) and set(replay_gains) == set(gains), "C14_MECHANISM_SCALAR_GAIN_KEYS")
    for key, value in gains.items():
        _require(math.isclose(float(replay_gains[key]), value, rel_tol=0.0, abs_tol=1e-15), "C14_MECHANISM_SCALAR_GAIN_MISMATCH", key)
    scheduler = _read(root / "SCHEDULER_ACCOUNTING.json")
    _require(scheduler.get("job_id") == "8859283" and scheduler.get("scheduler_terminal_state") == "COMPLETED" and scheduler.get("exit_code") == "0:0", "C14_MECHANISM_SCHEDULER_TERMINAL")
    rows = scheduler.get("rows")
    _require(isinstance(rows, list) and rows and all(row.get("State") == "COMPLETED" and row.get("ExitCode") == "0:0" for row in rows), "C14_MECHANISM_SCHEDULER_ROWS")
    _require(scheduler.get("allocated_cpus") == 96 and scheduler.get("requested_memory") == "32G", "C14_MECHANISM_SCHEDULER_RESOURCES")
    return {
        "schema": "filament.cpu_s4_6e.c14.mechanism_admission.v1",
        "status": "PASS_C14_MECHANISM_ADMISSION",
        "mechanism_execution_commit": MECHANISM_EXECUTION,
        "mechanism_result_raw_sha256": pv.raw_sha256_file(root / "RESULT.json"),
        "mechanism_result_size_bytes": int((root / "RESULT.json").stat().st_size),
        "mechanism_export_manifest_raw_sha256": pv.raw_sha256_file(export_path),
        "mechanism_source_manifest_raw_sha256": pv.raw_sha256_file(root / "EXECUTION_SOURCE_MANIFEST.json"),
        "formal_calls": 90,
        "warmup_calls": 36,
        "all_configs_exact": True,
        "p24_block_gains": gains,
        "scheduler": {key: scheduler[key] for key in ("job_id", "scheduler_terminal_state", "exit_code", "actual_runtime_s", "actual_allocated_core_hours", "allocated_cpus", "requested_memory", "node") if key in scheduler},
        "source_identities": source_identities,
        "verified_export_records": list(verified_files.values()),
        "scientific_arrays_loaded": 0,
    }


def copy_mechanism_export(evidence_root: str | Path, destination: str | Path) -> dict[str, Any]:
    """Copy the verified lightweight export, preserving relative paths and raw bytes."""

    root = Path(evidence_root).resolve(strict=True)
    target = Path(destination).resolve()
    _require(not target.exists(), "C14_MECHANISM_COPY_NON_OVERWRITING")
    admission = validate_mechanism_evidence(root)
    export = _read(root / "EXPORT_MANIFEST.json")
    target.mkdir(parents=True)
    for row in export["records"]:
        relative = str(row["path"])
        source = _within(root, relative)
        destination_path = _within(target, relative)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination_path)
    shutil.copyfile(root / "EXPORT_MANIFEST.json", target / "EXPORT_MANIFEST.json")
    return admission


__all__ = ["MechanismAdmissionError", "copy_mechanism_export", "validate_mechanism_evidence"]
