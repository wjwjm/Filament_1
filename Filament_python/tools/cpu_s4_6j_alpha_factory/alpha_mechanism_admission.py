"""Strict admission checks for the completed original-size alpha mechanism."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from tools.hpc_ops import provenance_v2 as pv

from .alpha_formal_protocol import CANDIDATE, SHAPE


SCHEMA = "filament.cpu.s4_6j.alpha_mechanism_admission.v1"
STATUS = "PASS_S46J_ALPHA_ORIGINAL_SIZE_MECHANISM"
REMOTE_OWNER = Path("/publicfs01/fs1-t/home/t0s000727/user_wjm")
RUNTIME_PATHS = (
    "Filament_python/tools/cpu_s4_6j_alpha_factory/alpha_pool.py",
    "Filament_python/tools/cpu_s4_6j_alpha_factory/alpha_hook.py",
    "Filament_python/tools/cpu_s4_6j_alpha_factory/alpha_factory.py",
    "Filament_python/tools/cpu_s4_6j_alpha_factory/alpha_role_spec.py",
    "Filament_python/tools/cpu_s4_6j_alpha_factory/alpha_measurement.py",
    "Filament_python/tools/cpu_s4_6j_alpha_factory/alpha_entry.py",
    "Filament_python/tools/cpu_s4_6j_alpha_factory/alpha_capture.py",
    "Filament_python/tools/cpu_s4_6j_alpha_factory/alpha_capture_hpc.py",
    "Filament_python/tools/cpu_s4_6j_alpha_factory/alpha_mechanism.py",
    "Filament_python/tools/cpu_s4_6j_alpha_factory/alpha_protocol.py",
    "Filament_python/tools/cpu_s4_6f_fullsize_parallel/measurement.py",
    "Filament_python/tools/cpu_s4_6f_fullsize_parallel/analysis.py",
    "Filament_python/tools/cpu_s4_6i_residual_parallel/selected_measurement.py",
)


class MechanismAdmissionError(RuntimeError):
    """The capture/mechanism receipt failed a fixed formal gate."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise MechanismAdmissionError(reason)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MechanismAdmissionError(f"S46J_ALPHA_JSON_READ:{path.name}") from exc
    _require(isinstance(value, dict), f"S46J_ALPHA_JSON_OBJECT:{path.name}")
    return value


def _mapping_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        path = record.get("path")
        if isinstance(path, str):
            result[path] = record
    return result


def _current_blob(repo: Path, path: str) -> str:
    try:
        return pv._git(repo, "rev-parse", "HEAD:" + path)
    except Exception as exc:
        raise MechanismAdmissionError("S46J_ALPHA_RUNTIME_BLOB_UNAVAILABLE:" + path) from exc


def _runtime_identity(report: Path, repo: Path, admission: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = report / "EXECUTION_SOURCE_MANIFEST.json"
    _require(manifest_path.is_file(), "S46J_ALPHA_EXECUTION_SOURCE_MANIFEST_MISSING")
    current_manifest = _read(manifest_path)
    current = _mapping_records(pv.iter_records(current_manifest, require_hash_scope=True))
    artifact_records = admission.get("unchanged_runtime_records")
    _require(isinstance(artifact_records, list), "S46J_ALPHA_RUNTIME_RECORDS_MISSING")
    artifact = _mapping_records(artifact_records)
    _require(len(artifact) == len(RUNTIME_PATHS), "S46J_ALPHA_RUNTIME_RECORD_COUNT")
    checked: list[dict[str, Any]] = []
    for path in RUNTIME_PATHS:
        old = artifact.get(path)
        now = current.get(path)
        _require(old is not None, "S46J_ALPHA_RUNTIME_ARTIFACT_MISSING:" + path)
        _require(now is not None, "S46J_ALPHA_RUNTIME_CURRENT_MISSING:" + path)
        for record in (old, now):
            _require(
                record.get("classification") == "tracked_text",
                "S46J_ALPHA_RUNTIME_CLASS:" + path,
            )
            _require(
                record.get("hash_scope") == pv.TRACKED_HASH_SCOPE,
                "S46J_ALPHA_RUNTIME_SCOPE:" + path,
            )
        _require(
            old.get("git_blob_oid") == now.get("git_blob_oid")
            and old.get("canonical_lf_sha256") == now.get("canonical_lf_sha256"),
            "S46J_ALPHA_RUNTIME_IDENTITY_CHANGED:" + path,
        )
        file_path = repo / path
        _require(file_path.is_file() and not file_path.is_symlink(), "S46J_ALPHA_RUNTIME_FILE_MISSING:" + path)
        _require(_current_blob(repo, path) == now.get("git_blob_oid"), "S46J_ALPHA_RUNTIME_BLOB_CHANGED:" + path)
        _require(
            pv.canonical_lf_sha256(file_path.read_bytes()) == now.get("canonical_lf_sha256"),
            "S46J_ALPHA_RUNTIME_CANONICAL_CHANGED:" + path,
        )
        checked.append(
            {
                "path": path,
                "git_blob_oid": now.get("git_blob_oid"),
                "canonical_lf_sha256": now.get("canonical_lf_sha256"),
            }
        )
    return {
        "manifest_raw_sha256": pv.raw_sha256_file(manifest_path),
        "records": checked,
        "ancestor_relaxation": False,
    }


def _scheduler_ok(admission: Mapping[str, Any]) -> dict[str, Any]:
    scheduler = admission.get("scheduler")
    _require(isinstance(scheduler, Mapping), "S46J_ALPHA_SCHEDULER_MISSING")
    job_id = scheduler.get("job_id")
    _require(str(job_id).isdigit() and str(job_id) == str(admission.get("capture_job_id")), "S46J_ALPHA_SCHEDULER_JOB_ID")
    _require(scheduler.get("all_steps_completed") is True, "S46J_ALPHA_SCHEDULER_INCOMPLETE")
    steps = scheduler.get("steps")
    _require(isinstance(steps, list) and len(steps) >= 3, "S46J_ALPHA_SCHEDULER_STEPS_MISSING")
    for step in steps:
        _require(isinstance(step, Mapping), "S46J_ALPHA_SCHEDULER_STEP_SHAPE")
        step_id = str(step.get("job_id", ""))
        _require(step_id == str(job_id) or step_id.startswith(str(job_id) + "."), "S46J_ALPHA_SCHEDULER_STEP_JOB_ID")
        _require(str(step.get("state", "")).upper() == "COMPLETED", "S46J_ALPHA_SCHEDULER_STATE")
        _require(step.get("exit_code") == "0:0", "S46J_ALPHA_SCHEDULER_EXIT")
    _require(any(str(step.get("job_id")) == str(job_id) for step in steps), "S46J_ALPHA_SCHEDULER_TOP_ROW")
    return {"job_id": job_id, "steps": len(steps), "all_steps_completed": True}


def _sha_field(admission: Mapping[str, Any], name: str) -> str:
    value = admission.get(name)
    _require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower()), "S46J_ALPHA_FIELD:" + name)
    return value


def _admission_scalars(admission: Mapping[str, Any], *, local_engineering: bool) -> dict[str, Any]:
    job_id = admission.get("capture_job_id")
    commit = admission.get("capture_execution_commit")
    _require(str(job_id).isdigit(), "S46J_ALPHA_CAPTURE_JOB_ID")
    _require(isinstance(commit, str) and len(commit) == 40, "S46J_ALPHA_CAPTURE_COMMIT")
    for name in (
        "capture_result_raw_sha256",
        "capture_source_manifest_raw_sha256",
        "input_contract_raw_sha256",
        "reference_selection_raw_sha256",
        "capture_manifest_raw_sha256",
    ):
        _sha_field(admission, name)
    if admission.get("mechanism_result_raw_sha256") is not None:
        _sha_field(admission, "mechanism_result_raw_sha256")
    root = admission.get("capture_remote_root")
    _require(isinstance(root, str) and root.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].startswith("filament_cpu_s46j_alpha_"), "S46J_ALPHA_REMOTE_ROOT")
    normalized = root.replace("\\", "/").rstrip("/")
    _require(local_engineering or Path(normalized).parent == REMOTE_OWNER, "S46J_ALPHA_REMOTE_OWNER")
    _require(isinstance(admission.get("source_binding"), Mapping), "S46J_ALPHA_SOURCE_BINDING")
    result = {
        "capture_job_id": job_id,
        "capture_execution_commit": commit,
        "capture_remote_root": root,
        **{name: admission[name] for name in (
            "capture_result_raw_sha256",
            "capture_source_manifest_raw_sha256",
            "input_contract_raw_sha256",
            "reference_selection_raw_sha256",
            "capture_manifest_raw_sha256",
        )},
    }
    if "mechanism_result_raw_sha256" in admission:
        result["mechanism_result_raw_sha256"] = admission["mechanism_result_raw_sha256"]
    return result


def _remote_capture_identity(admission: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(admission["capture_remote_root"]))
    _require(root.parent == REMOTE_OWNER and root.resolve(strict=True) == root, "S46J_ALPHA_CANONICAL_REMOTE_ROOT")
    report = root / "report"
    files = {
        "RESULT.json": "capture_result_raw_sha256",
        "EXECUTION_SOURCE_MANIFEST.json": "capture_source_manifest_raw_sha256",
        "INPUT_CONTRACT.json": "input_contract_raw_sha256",
        "REFERENCE_SELECTION.json": "reference_selection_raw_sha256",
    }
    checked: dict[str, str] = {}
    for name, field in files.items():
        path = report / name
        _require(path.is_file() and not path.is_symlink(), "S46J_ALPHA_REMOTE_FILE:" + name)
        checked[name] = pv.raw_sha256_file(path)
        _require(checked[name] == admission[field], "S46J_ALPHA_REMOTE_IDENTITY:" + name)
    result = _read(report / "RESULT.json")
    execution = _read(report / "EXECUTION.json")
    submission = _read(report / "SUBMISSION_RECEIPT.json")
    _require(result.get("candidate") == CANDIDATE and result.get("shape") == list(SHAPE), "S46J_ALPHA_REMOTE_RESULT_DOMAIN")
    _require(result.get("science_exact") is True and result.get("formal14_allowed") is True, "S46J_ALPHA_REMOTE_RESULT_GATE")
    cases = result.get("cases")
    captured_case = cases.get("R8_CAPTURED") if isinstance(cases, Mapping) else None
    _require(isinstance(captured_case, Mapping), "S46J_ALPHA_REMOTE_CAPTURE_CASE")
    mechanism = result.get("mechanism")
    _require(isinstance(mechanism, Mapping) and mechanism.get("all_exact") is True, "S46J_ALPHA_REMOTE_MECHANISM")
    _require(
        result.get("execution_commit") == captured_case.get("execution_commit") == mechanism.get("execution_commit") == execution.get("execution_commit") == admission["capture_execution_commit"],
        "S46J_ALPHA_REMOTE_EXECUTION_COMMIT",
    )
    _require(
        mechanism.get("inputs") == 3
        and mechanism.get("warmup_calls") == 36
        and mechanism.get("formal_calls") == 90
        and mechanism.get("formal14_allowed") is True
        and mechanism.get("blocks") == admission.get("blocks"),
        "S46J_ALPHA_REMOTE_MECHANISM_COUNTS",
    )
    case_manifest_sha = captured_case.get("capture_manifest_raw_sha256")
    mechanism_manifest_sha = mechanism.get("capture_manifest_raw_sha256")
    _require(
        case_manifest_sha == mechanism_manifest_sha == admission["capture_manifest_raw_sha256"],
        "S46J_ALPHA_REMOTE_CAPTURE_MANIFEST_BINDING",
    )
    case_binding = captured_case.get("capture_source_binding")
    mechanism_binding = mechanism.get("source_binding")
    _require(
        case_binding == mechanism_binding == admission.get("source_binding"),
        "S46J_ALPHA_REMOTE_SOURCE_BINDING",
    )
    manifest_path = Path(str(admission.get("capture_manifest_path", "")))
    if not manifest_path.is_absolute():
        manifest_path = report / manifest_path
    _require(manifest_path.is_file() and manifest_path.resolve().is_relative_to(root.resolve()), "S46J_ALPHA_CAPTURE_MANIFEST_PATH")
    manifest_sha = pv.raw_sha256_file(manifest_path)
    _require(manifest_sha == admission["capture_manifest_raw_sha256"], "S46J_ALPHA_CAPTURE_MANIFEST_SHA")
    capture = _read(manifest_path)
    _require(capture.get("total_calls") == 32 and capture.get("captured_states") == 3 and capture.get("array_file_count") == 12, "S46J_ALPHA_CAPTURE_COUNTS")
    _require([row.get("key") for row in capture.get("records", [])] == ["i1", "i4", "i8"], "S46J_ALPHA_CAPTURE_KEYS")
    _require(capture.get("source_binding") == admission.get("source_binding"), "S46J_ALPHA_CAPTURE_BINDING")
    _require(str(submission.get("job_id")) == str(admission["capture_job_id"]), "S46J_ALPHA_REMOTE_SUBMISSION")
    return {
        "root": str(root),
        "report": str(report),
        "raw_sha256": checked,
        "capture_manifest_path": str(manifest_path),
        "capture_manifest_raw_sha256": manifest_sha,
        "status": "PASS_RETAINED_ALPHA_CAPTURE_IDENTITY",
    }


def verify(report: str | Path, *, repo: str | Path | None = None, local_engineering: bool = False) -> dict[str, Any]:
    report_path = Path(report).resolve(strict=True)
    if report_path.is_file():
        report_path = report_path.parent
    admission_path = report_path / "MECHANISM_ADMISSION.json"
    execution_path = report_path / "EXECUTION.json"
    _require(admission_path.is_file(), "S46J_ALPHA_ADMISSION_MISSING")
    _require(execution_path.is_file(), "S46J_ALPHA_EXECUTION_MISSING")
    admission = _read(admission_path)
    execution = _read(execution_path)
    _require(admission.get("schema") == SCHEMA, "S46J_ALPHA_ADMISSION_SCHEMA")
    _require(admission.get("status") == STATUS, "S46J_ALPHA_ADMISSION_STATUS")
    _require(admission.get("candidate") == CANDIDATE and admission.get("shape") == list(SHAPE), "S46J_ALPHA_ADMISSION_DOMAIN")
    _require(admission.get("captured_states") == 3 and admission.get("array_file_count") == 12 and admission.get("total_calls") == 32, "S46J_ALPHA_CAPTURE_COMPLETENESS")
    _require(admission.get("input_count") == 3 and admission.get("warmup_calls") == 36 and admission.get("formal_calls") == 90, "S46J_ALPHA_MECHANISM_COUNTS")
    blocks = admission.get("blocks")
    _require(isinstance(blocks, list) and len(blocks) == 6, "S46J_ALPHA_MECHANISM_BLOCK_COUNT")
    _require(all(isinstance(row, Mapping) and row.get("gain_pass") is True and isinstance(row.get("p24_gain"), (int, float)) and not isinstance(row.get("p24_gain"), bool) and math.isfinite(float(row["p24_gain"])) and float(row["p24_gain"]) >= 0.05 for row in blocks), "S46J_ALPHA_MECHANISM_GAIN_GATE")
    expected_keys = {(group, key) for group in (1, 2) for key in ("i1", "i4", "i8")}
    actual_keys = [(row.get("group"), row.get("key")) for row in blocks]
    _require(set(actual_keys) == expected_keys and len(actual_keys) == len(set(actual_keys)), "S46J_ALPHA_MECHANISM_BLOCK_KEYS")
    _require(admission.get("all_exact") is True and admission.get("formal14_allowed") is True, "S46J_ALPHA_MECHANISM_EXACT_GATE")
    scalar = _admission_scalars(admission, local_engineering=local_engineering)
    for field in ("capture_execution_commit", "capture_result_raw_sha256", "capture_source_manifest_raw_sha256", "input_contract_raw_sha256", "reference_selection_raw_sha256", "capture_manifest_raw_sha256"):
        _require(admission[field] == execution.get(field), "S46J_ALPHA_EXECUTION_BINDING:" + field)
    input_contract_path = report_path / "INPUT_CONTRACT.json"
    reference_selection_path = report_path / "REFERENCE_SELECTION.json"
    _require(input_contract_path.is_file(), "S46J_ALPHA_INPUT_CONTRACT_MISSING")
    _require(reference_selection_path.is_file(), "S46J_ALPHA_REFERENCE_SELECTION_MISSING")
    _require(
        pv.raw_sha256_file(input_contract_path) == admission["input_contract_raw_sha256"],
        "S46J_ALPHA_INPUT_CONTRACT_CHANGED",
    )
    _require(
        pv.raw_sha256_file(reference_selection_path) == admission["reference_selection_raw_sha256"],
        "S46J_ALPHA_REFERENCE_SELECTION_CHANGED",
    )
    _require(execution.get("mechanism_admission_raw_sha256") == pv.raw_sha256_file(admission_path), "S46J_ALPHA_ADMISSION_EXECUTION_BINDING")
    repo_path = Path(repo).resolve() if repo is not None else Path(__file__).resolve().parents[3]
    head = pv._git(repo_path, "rev-parse", "HEAD")
    if local_engineering:
        ancestor: bool | str = "SKIPPED_LOCAL_ENGINEERING"
    else:
        ancestor = pv._git(repo_path, "merge-base", scalar["capture_execution_commit"], head) == scalar["capture_execution_commit"]
        _require(ancestor is True, "S46J_ALPHA_CAPTURE_COMMIT_NOT_ANCESTOR")
    scheduler = _scheduler_ok(admission)
    runtime = _runtime_identity(report_path, repo_path, admission)
    remote = None if local_engineering else _remote_capture_identity(admission)
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "candidate": CANDIDATE,
        "shape": list(SHAPE),
        "captured_states": 3,
        "array_file_count": 12,
        "total_calls": 32,
        "input_count": 3,
        "warmup_calls": 36,
        "formal_calls": 90,
        "blocks": blocks,
        "all_exact": True,
        "formal14_allowed": True,
        "admission_raw_sha256": pv.raw_sha256_file(admission_path),
        "execution_raw_sha256": pv.raw_sha256_file(execution_path),
        "capture_commit_ancestor": ancestor,
        "scheduler": scheduler,
        "runtime_identity": runtime,
        "retained_remote_capture": remote,
        "local_engineering": bool(local_engineering),
        **scalar,
    }


require_capture_admission = verify


__all__ = [
    "RUNTIME_PATHS",
    "SCHEMA",
    "STATUS",
    "MechanismAdmissionError",
    "require_capture_admission",
    "verify",
]
