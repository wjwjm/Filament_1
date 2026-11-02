"""Scalar/PV2 admission checks for the selected original-grid mechanism."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from tools.hpc_ops import provenance_v2 as pv

from .formal_protocol import CANDIDATE, CONTEXTS, SHAPE


SCHEMA = "filament.cpu_s4_6i.mechanism_admission.v1"
STATUS = "PASS_SELECTED_ORIGINAL_GRID_MECHANISM"
REMOTE_OWNER = Path('/publicfs01/fs1-t/home/t0s000727/user_wjm')
RUNTIME_PATHS = (
    "Filament_python/tools/cpu_s4_6i_residual_parallel/power_volume.py",
    "Filament_python/tools/cpu_s4_6i_residual_parallel/power_hook.py",
    "Filament_python/tools/cpu_s4_6i_residual_parallel/selected_factory.py",
    "Filament_python/tools/cpu_s4_6i_residual_parallel/selected_role_spec.py",
    "Filament_python/tools/cpu_s4_6i_residual_parallel/selected_measurement.py",
    "Filament_python/tools/cpu_s4_6i_residual_parallel/selected_entry.py",
)


class MechanismAdmissionError(RuntimeError):
    """The mechanism receipt is absent, stale, or fails a fixed gate."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise MechanismAdmissionError(reason)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MechanismAdmissionError(f"S46I_MECHANISM_JSON_READ:{path.name}") from exc
    _require(isinstance(value, dict), f"S46I_MECHANISM_JSON_OBJECT:{path.name}")
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
        raise MechanismAdmissionError("S46I_MECHANISM_RUNTIME_BLOB_UNAVAILABLE:" + path) from exc


def _runtime_identity(report: Path, repo: Path, admission: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = report / "EXECUTION_SOURCE_MANIFEST.json"
    _require(manifest_path.is_file(), "S46I_MECHANISM_EXECUTION_SOURCE_MANIFEST_MISSING")
    current_manifest = _read(manifest_path)
    current = _mapping_records(pv.iter_records(current_manifest, require_hash_scope=True))
    artifact_records = admission.get("unchanged_runtime_records")
    _require(isinstance(artifact_records, list), "S46I_MECHANISM_RUNTIME_RECORDS_MISSING")
    artifact = _mapping_records(artifact_records)
    checked: list[dict[str, Any]] = []
    _require(len(artifact) == 6, "S46I_MECHANISM_RUNTIME_RECORD_COUNT")
    for path in RUNTIME_PATHS:
        old = artifact.get(path)
        now = current.get(path)
        _require(old is not None, "S46I_MECHANISM_RUNTIME_ARTIFACT_MISSING:" + path)
        _require(now is not None, "S46I_MECHANISM_RUNTIME_CURRENT_MISSING:" + path)
        for record in (old, now):
            _require(record.get("classification") == "tracked_text", "S46I_MECHANISM_RUNTIME_CLASS:" + path)
            _require(record.get("hash_scope") == pv.TRACKED_HASH_SCOPE, "S46I_MECHANISM_RUNTIME_SCOPE:" + path)
        _require(
            old.get("git_blob_oid") == now.get("git_blob_oid")
            and old.get("canonical_lf_sha256") == now.get("canonical_lf_sha256"),
            "S46I_MECHANISM_RUNTIME_IDENTITY_CHANGED:" + path,
        )
        file_path = repo / path
        _require(file_path.is_file() and not file_path.is_symlink(), "S46I_MECHANISM_RUNTIME_FILE_MISSING:" + path)
        _require(_current_blob(repo, path) == now.get("git_blob_oid"), "S46I_MECHANISM_RUNTIME_BLOB_CHANGED:" + path)
        try:
            current_canonical = pv.canonical_lf_sha256(file_path.read_bytes())
        except OSError as exc:
            raise MechanismAdmissionError("S46I_MECHANISM_RUNTIME_CANONICAL_UNAVAILABLE:" + path) from exc
        _require(current_canonical == now.get("canonical_lf_sha256"), "S46I_MECHANISM_RUNTIME_CANONICAL_CHANGED:" + path)
        checked.append({"path": path, "git_blob_oid": now.get("git_blob_oid"), "canonical_lf_sha256": now.get("canonical_lf_sha256")})
    return {"manifest_raw_sha256": pv.raw_sha256_file(manifest_path), "records": checked}


def _scheduler_ok(admission: Mapping[str, Any]) -> dict[str, Any]:
    scheduler = admission.get("scheduler")
    _require(isinstance(scheduler, Mapping), "S46I_MECHANISM_SCHEDULER_MISSING")
    job_id = scheduler.get("job_id")
    _require(str(job_id).isdigit() and str(job_id) == str(admission['capture_job_id']), "S46I_MECHANISM_SCHEDULER_JOB_ID")
    _require(scheduler.get("all_steps_completed") is True, "S46I_MECHANISM_SCHEDULER_INCOMPLETE")
    steps = scheduler.get("steps")
    _require(isinstance(steps, list) and len(steps) >= 3, "S46I_MECHANISM_SCHEDULER_STEPS_MISSING")
    for step in steps:
        _require(isinstance(step, Mapping), "S46I_MECHANISM_SCHEDULER_STEP_SHAPE")
        step_id = str(step.get('job_id', ''))
        _require(step_id == str(job_id) or step_id.startswith(str(job_id) + '.'), "S46I_MECHANISM_SCHEDULER_STEP_JOB_ID")
        _require(str(step.get("state", "")).upper() == "COMPLETED", "S46I_MECHANISM_SCHEDULER_STATE")
        _require(step.get("exit_code") == "0:0", "S46I_MECHANISM_SCHEDULER_EXIT")
    _require(any(str(step['job_id']) == str(job_id) for step in steps), 'S46I_MECHANISM_SCHEDULER_TOP_ROW')
    return {"job_id": job_id, "steps": len(steps), "all_steps_completed": True}


def _admission_scalars(admission: Mapping[str, Any], *, local_engineering: bool) -> dict[str, Any]:
    """Validate scalar provenance fields without opening capture payloads."""

    job_id = admission.get("capture_job_id")
    result_sha = admission.get("capture_result_raw_sha256")
    source_sha = admission.get("capture_source_manifest_raw_sha256")
    input_sha = admission.get("input_contract_raw_sha256")
    remote_root = admission.get("capture_remote_root")
    for name, value in (
        ("capture_job_id", job_id),
        ("capture_result_raw_sha256", result_sha),
        ("capture_source_manifest_raw_sha256", source_sha),
        ("input_contract_raw_sha256", input_sha),
    ):
        _require(isinstance(value, str) and value, "S46I_MECHANISM_FIELD:" + name)
    _require(
        isinstance(result_sha, str) and len(result_sha) == 64 and all(char in "0123456789abcdef" for char in result_sha.lower()),
        "S46I_MECHANISM_RESULT_SHA256",
    )
    _require(
        isinstance(source_sha, str) and len(source_sha) == 64 and all(char in "0123456789abcdef" for char in source_sha.lower()),
        "S46I_MECHANISM_SOURCE_MANIFEST_SHA256",
    )
    _require(
        isinstance(input_sha, str) and len(input_sha) == 64 and all(char in "0123456789abcdef" for char in input_sha.lower()),
        "S46I_MECHANISM_INPUT_CONTRACT_SHA256",
    )
    normalized_root = remote_root.replace("\\", "/").rstrip("/") if isinstance(remote_root, str) else ""
    _require(
        bool(normalized_root)
        and normalized_root.rsplit("/", 1)[-1].startswith("filament_cpu_s46i_power_capture_"),
        "S46I_MECHANISM_REMOTE_ROOT",
    )
    _require(local_engineering or Path(normalized_root).parent == REMOTE_OWNER, "S46I_MECHANISM_REMOTE_OWNER")
    return {
        "capture_job_id": job_id,
        "capture_result_raw_sha256": result_sha,
        "capture_source_manifest_raw_sha256": source_sha,
        "input_contract_raw_sha256": input_sha,
        "capture_remote_root": remote_root,
    }


def _remote_capture_identity(admission: Mapping[str, Any]) -> dict[str, Any]:
    """Bind admission to the actual retained capture result on the t0 host."""
    root = Path(admission['capture_remote_root'])
    _require(root.parent == REMOTE_OWNER and root.resolve(strict=True) == root,
             'S46I_MECHANISM_CANONICAL_REMOTE_ROOT')
    report = root / 'report'
    checked = {}
    for filename, field in (
        ('RESULT.json', 'capture_result_raw_sha256'),
        ('EXECUTION_SOURCE_MANIFEST.json', 'capture_source_manifest_raw_sha256'),
        ('INPUT_CONTRACT.json', 'input_contract_raw_sha256'),
    ):
        path = report / filename
        _require(path.is_file() and not path.is_symlink(), 'S46I_MECHANISM_REMOTE_FILE:' + filename)
        checked[filename] = pv.raw_sha256_file(path)
        _require(checked[filename] == admission[field], 'S46I_MECHANISM_REMOTE_IDENTITY:' + filename)
    result = _read(report / 'RESULT.json')
    execution = _read(report / 'EXECUTION.json')
    submission = _read(report / 'SUBMISSION_RECEIPT.json')
    _require(result['execution_commit'] == execution['execution_commit'] == admission['capture_execution_commit']
             and str(submission['job_id']) == admission['capture_job_id'], 'S46I_MECHANISM_REMOTE_EXECUTION')
    _require(result['candidate'] == CANDIDATE and result['shape'] == list(SHAPE)
             and result['science_exact'] is True and result['formal14_allowed'] is True
             and result['mechanism']['blocks'] == admission['blocks'], 'S46I_MECHANISM_REMOTE_RESULT')
    return dict(root=str(root), raw_sha256=checked, status='PASS_RETAINED_CAPTURE_IDENTITY')


def verify(
    report: str | Path,
    *,
    repo: str | Path | None = None,
    local_engineering: bool = False,
) -> dict[str, Any]:
    """Validate the external admission receipt and formal execution binding."""

    report_path = Path(report).resolve(strict=True)
    if report_path.is_file():
        report_path = report_path.parent
    admission_path = report_path / "MECHANISM_ADMISSION.json"
    execution_path = report_path / "EXECUTION.json"
    _require(admission_path.is_file(), "S46I_MECHANISM_ADMISSION_MISSING")
    _require(execution_path.is_file(), "S46I_MECHANISM_EXECUTION_MISSING")
    admission = _read(admission_path)
    execution = _read(execution_path)
    _require(admission.get("schema") == SCHEMA, "S46I_MECHANISM_SCHEMA")
    _require(admission.get("status") == STATUS, "S46I_MECHANISM_STATUS")
    _require(admission.get("candidate") == CANDIDATE, "S46I_MECHANISM_CANDIDATE")
    _require(admission.get("shape") == list(SHAPE), "S46I_MECHANISM_ORIGINAL_GRID_SHAPE")
    _require(admission.get("input_count") == 12, "S46I_MECHANISM_INPUT_COUNT")
    _require(admission.get("warmup_calls") == 144, "S46I_MECHANISM_WARMUP_COUNT")
    _require(admission.get("formal_calls") == 360, "S46I_MECHANISM_FORMAL_COUNT")
    scalar_fields = _admission_scalars(admission, local_engineering=local_engineering)
    blocks = admission.get("blocks")
    _require(isinstance(blocks, list) and len(blocks) == 24, "S46I_MECHANISM_GROUP_COUNT")
    _require(admission.get("all_exact") is True, "S46I_MECHANISM_EXACT")
    _require(admission.get("formal14_allowed") is True, "S46I_MECHANISM_FORMAL14")
    _require(
        all(
            isinstance(row, Mapping)
            and row.get("group") in (1, 2)
            and isinstance(row.get("key"), str)
            and row.get("gain_pass") is True
            and isinstance(row.get("p24_gain"), (int, float))
            and not isinstance(row.get("p24_gain"), bool)
            and math.isfinite(row["p24_gain"])
            and row["p24_gain"] >= 0.05
            for row in blocks
        ),
        "S46I_MECHANISM_GAIN_GATE",
    )
    expected_keys = {
        (group, f"i{interval}_{context}")
        for group in (1, 2)
        for interval in (1, 4, 8)
        for context in CONTEXTS
    }
    actual_keys = [(row["group"], row["key"]) for row in blocks]
    _require(len(actual_keys) == len(set(actual_keys)) and set(actual_keys) == expected_keys, "S46I_MECHANISM_BLOCK_KEYS")
    _require(
        {group: sum(row["group"] == group for row in blocks) for group in (1, 2)} == {1: 12, 2: 12},
        "S46I_MECHANISM_BLOCK_GROUPS",
    )

    source_manifest_sha = admission.get("capture_source_manifest_raw_sha256")
    input_contract_sha = admission.get("input_contract_raw_sha256")
    _require(source_manifest_sha == execution.get("capture_source_manifest_raw_sha256"), "S46I_MECHANISM_CAPTURE_MANIFEST_BINDING")
    _require(input_contract_sha == execution.get("input_contract_raw_sha256"), "S46I_MECHANISM_INPUT_CONTRACT_BINDING")
    input_contract_path = report_path / "INPUT_CONTRACT.json"
    _require(input_contract_path.is_file(), "S46I_MECHANISM_INPUT_CONTRACT_MISSING")
    _require(
        pv.raw_sha256_file(input_contract_path) == input_contract_sha,
        "S46I_MECHANISM_INPUT_CONTRACT_CHANGED",
    )
    admission_raw_sha256 = pv.raw_sha256_file(admission_path)
    _require(execution.get("mechanism_admission_raw_sha256") == admission_raw_sha256, "S46I_MECHANISM_EXECUTION_ADMISSION_BINDING")

    source_binding = admission.get("source_binding", {})
    capture_commit = admission.get("capture_execution_commit") or source_binding.get("execution_commit")
    _require(isinstance(capture_commit, str) and len(capture_commit) == 40, "S46I_MECHANISM_CAPTURE_COMMIT")
    repo_path = Path(repo).resolve() if repo is not None else Path(__file__).resolve().parents[3]
    head = pv._git(repo_path, "rev-parse", "HEAD")
    if local_engineering:
        ancestor: bool | str = "SKIPPED_LOCAL_ENGINEERING"
    else:
        ancestor = pv._git(repo_path, "merge-base", capture_commit, head) == capture_commit
        _require(ancestor is True, "S46I_MECHANISM_CAPTURE_COMMIT_NOT_ANCESTOR")
    # Local engineering may omit remote file access, but it must still bind
    # the real completed scheduler evidence from the selected mechanism run.
    scheduler = _scheduler_ok(admission)
    runtime = _runtime_identity(report_path, repo_path, admission)
    remote = None if local_engineering else _remote_capture_identity(admission)
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "candidate": CANDIDATE,
        "shape": list(SHAPE),
        "input_count": 12,
        "warmup_calls": 144,
        "formal_calls": 360,
        "groups": 24,
        "all_exact": True,
        "formal14_allowed": True,
        "admission_raw_sha256": admission_raw_sha256,
        "execution_raw_sha256": pv.raw_sha256_file(execution_path),
        "capture_execution_commit": capture_commit,
        **scalar_fields,
        "current_head": head,
        "capture_commit_ancestor": ancestor,
        "scheduler": scheduler,
        "runtime_identity": runtime,
        "retained_remote_capture": remote,
        "local_engineering": bool(local_engineering),
    }


def require_capture_admission(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Require the selected original-grid mechanism admission."""
    return verify(*args, **kwargs)


__all__ = [
    "SCHEMA",
    "STATUS",
    "RUNTIME_PATHS",
    "MechanismAdmissionError",
    "require_capture_admission",
    "verify",
]
