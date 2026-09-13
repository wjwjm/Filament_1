#!/usr/bin/env python3
"""Resumable, fail-closed monitor for the S5-1 F01--F06 fault matrix.

The monitor owns only submission receipts and evidence reports.  It never
opens, reconstructs, or modifies a Streaming lifecycle root.  The fault and
recovery batch jobs remain the only processes that execute lifecycle work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA = "khz_filament.hr4e5s.s5.fault_matrix_monitor.v1"
FAULT_CASES = ("F01", "F02", "F03", "F04", "F05", "F06")
REPAIR_CASES = ("F03", "F04", "F05")
INITIAL = "PENDING_FAULT_SUBMISSION"
TERMINAL_SACCT = {
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
    "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "SPECIAL_EXIT", "STOPPED", "TIMEOUT",
}
CASE_TERMINAL = {"FAULT_CONTRACT_FAIL", "PASS", "FAIL"}
STATE_RANK = {
    INITIAL: 0,
    "SUBMITTED_FAULT": 1,
    "FAULT_RUNNING": 2,
    "FAULT_TERMINAL": 3,
    "FAULT_AUDIT_PENDING": 4,
    "FAULT_CONTRACT_PASS": 5,
    "RECOVERY_SUBMITTED": 6,
    "RECOVERY_RUNNING": 7,
    "RECOVERY_TERMINAL": 8,
    "EXACT_COMPARE_PENDING": 9,
    "PASS": 10,
}


def _utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(event), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _require(value: object, message: str) -> None:
    if not value:
        raise ValueError(message)


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    value = _read_json(manifest_path)
    _require(value.get("schema") == SCHEMA, "invalid fault-matrix manifest schema")
    for key in ("run_root", "repo", "expected_sha", "preflight", "reference_case_root", "submit_script", "clean_reference_job", "historical_clean_reference_job", "cases"):
        _require(value.get(key), f"matrix manifest is missing {key}")
    cases = value["cases"]
    _require(isinstance(cases, list), "matrix manifest cases must be a list")
    identifiers = [str(item.get("case_id", "")) for item in cases if isinstance(item, Mapping)]
    _require(tuple(identifiers) in (FAULT_CASES, REPAIR_CASES), "matrix manifest must contain F01--F06 or the ordered F03--F05 repair subset")
    for item in cases:
        _require(isinstance(item, Mapping), "matrix case must be an object")
        case_id, fault_id, fault_screen = (str(item.get(key, "")) for key in ("case_id", "fault_id", "fault_screen"))
        _require(fault_id.startswith(case_id + "_"), f"{case_id} fault id is not case-bound")
        _require(fault_screen, f"{case_id} is missing its target screen")
    _require(str(value["clean_reference_job"]).isdigit(), "matrix manifest clean reference job must be numeric")
    _require(str(value["historical_clean_reference_job"]) == "238465", "matrix manifest must retain historical clean 238465")
    _require(manifest_path.parent.resolve() == Path(str(value["run_root"])).resolve(), "matrix manifest must be stored at its declared run root")
    return value


def _state_paths(manifest_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    root = manifest_path.parent
    return (
        root / "monitor_state.json",
        root / "monitor_events.jsonl",
        root / "s5_1_fault_matrix.csv",
        root / "s5_1_exact_summary.json",
        root / "s5_1_rerun_report.md",
    )


def _fresh_state(manifest_path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "run_root": str(manifest["run_root"]),
        "expected_sha": str(manifest["expected_sha"]),
        "created_utc": _utc(),
        "updated_utc": _utc(),
        "monitor_status": "OPEN",
        "cases": {str(item["case_id"]): {"state": INITIAL, "history": []} for item in manifest["cases"]},
    }


def load_or_create_state(manifest_path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    state_path, _, _, _, _ = _state_paths(manifest_path)
    if not state_path.exists():
        return _fresh_state(manifest_path, manifest)
    state = _read_json(state_path)
    _require(state.get("schema") == SCHEMA, "invalid monitor state schema")
    _require(state.get("manifest_sha256") == _sha256_file(manifest_path), "monitor manifest changed after state creation")
    _require(state.get("expected_sha") == str(manifest["expected_sha"]), "monitor state execution SHA mismatch")
    case_ids = {str(item["case_id"]) for item in manifest["cases"]}
    _require(set(state.get("cases", {})) == case_ids, "monitor state case set mismatch")
    return state


def _record_event(state: dict[str, Any], case_id: str, event: str, **detail: Any) -> dict[str, Any]:
    entry = {"timestamp_utc": _utc(), "case_id": case_id, "event": event, **detail}
    state["cases"][case_id].setdefault("history", []).append(entry)
    return entry


def _transition(state: dict[str, Any], case_id: str, target: str, reason: str, **detail: Any) -> dict[str, Any]:
    case = state["cases"][case_id]
    source = str(case["state"])
    if source == target:
        return _record_event(state, case_id, "state_reaffirmed", state=target, reason=reason, **detail)
    if source in CASE_TERMINAL:
        raise RuntimeError(f"{case_id} is terminal at {source}; refusing transition to {target}")
    if target not in CASE_TERMINAL:
        _require(target in STATE_RANK and STATE_RANK[target] > STATE_RANK.get(source, -1), f"non-forward state transition {source}->{target}")
    case["state"] = target
    return _record_event(state, case_id, "state_transition", from_state=source, to_state=target, reason=reason, **detail)


def _mark_fail(state: dict[str, Any], case_id: str, reason: str, **detail: Any) -> dict[str, Any]:
    state["cases"][case_id]["failure"] = {"reason": reason, **detail}
    return _transition(state, case_id, "FAIL", reason, **detail)


def _run(argv: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), cwd=cwd, check=False, capture_output=True, text=True, encoding="utf-8")


def _scheduler_snapshot(job_id: str) -> dict[str, Any]:
    squeue = _run(("squeue", "--noheader", "--jobs", job_id, "--format=%T"), Path.cwd())
    sacct = _run(("sacct", "-X", "--jobs", job_id, "--format=JobIDRaw,State,ExitCode,Elapsed,NodeList,Start,End,Reason", "--parsable2", "--noheader"), Path.cwd())
    snapshot: dict[str, Any] = {
        "queried_utc": _utc(),
        "job_id": job_id,
        "squeue": {"returncode": squeue.returncode, "stdout": squeue.stdout.strip(), "stderr": squeue.stderr.strip()},
        "sacct": {"returncode": sacct.returncode, "stdout": sacct.stdout.strip(), "stderr": sacct.stderr.strip()},
        "state": "UNKNOWN",
        "terminal": False,
    }
    lines = [line for line in sacct.stdout.splitlines() if line.strip()]
    if sacct.returncode == 0 and lines:
        values = lines[0].split("|")
        if len(values) >= 8:
            snapshot["accounting"] = dict(zip(("job_id", "state", "exit_code", "elapsed", "node_list", "start", "end", "reason"), values, strict=True))
            state = values[1].split()[0]
            snapshot["state"] = state
            snapshot["terminal"] = state in TERMINAL_SACCT
            return snapshot
    if squeue.returncode == 0 and squeue.stdout.strip():
        snapshot["state"] = squeue.stdout.splitlines()[0].strip().split()[0]
    return snapshot


def _receipt_path(run_root: Path, case_id: str, stage: str) -> Path:
    suffix = "fault" if stage == "fault" else "recovery"
    return run_root / f"{case_id}_{suffix}_submission_receipt.tsv"


def _read_receipt(path: Path, *, case_id: str, stage: str, expected_sha: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1:
        raise ValueError(f"invalid receipt row count: {path}")
    row = dict(rows[0])
    if row.get("case_id") != case_id or row.get("case_mode") != stage or row.get("execution_sha") != expected_sha:
        raise ValueError(f"receipt identity mismatch: {path}")
    if not str(row.get("job_id", "")).isdigit():
        raise ValueError(f"non-numeric receipt job id: {path}")
    return row


def _persist(state_path: Path, event_path: Path, state: dict[str, Any], event: Mapping[str, Any] | None = None) -> None:
    state["updated_utc"] = _utc()
    _atomic_json(state_path, state)
    if event is not None:
        _append_event(event_path, event)


def _submission_args(manifest: Mapping[str, Any], case: Mapping[str, Any], stage: str) -> list[str]:
    root = str(manifest["run_root"])
    arguments = [
        str(manifest["submit_script"]), str(manifest["repo"]), root, str(manifest["expected_sha"]),
        str(manifest["preflight"]), str(case["case_id"]), stage, str(manifest["reference_case_root"]),
    ]
    if stage == "fault":
        arguments.extend((str(case["fault_id"]), str(case["fault_screen"])))
    return arguments


def _submit_once(
    manifest: Mapping[str, Any], state: dict[str, Any], state_path: Path, event_path: Path,
    case: Mapping[str, Any], stage: str, submitter: Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]],
) -> None:
    case_id = str(case["case_id"])
    run_root = Path(str(manifest["run_root"]))
    entry = state["cases"][case_id]
    key = f"{stage}_submission"
    receipt_path = _receipt_path(run_root, case_id, stage)
    submission = entry.setdefault(key, {})
    existing = _read_receipt(receipt_path, case_id=case_id, stage=stage, expected_sha=str(manifest["expected_sha"]))
    if existing is not None:
        submission["receipt"] = str(receipt_path)
        submission["job_id"] = str(existing["job_id"])
        target = "SUBMITTED_FAULT" if stage == "fault" else "RECOVERY_SUBMITTED"
        event = (
            _transition(state, case_id, target, "existing_numeric_receipt_recovered", stage=stage, job_id=submission["job_id"])
            if state["cases"][case_id]["state"] != target
            else _record_event(state, case_id, "submission_receipt_reused", stage=stage, job_id=submission["job_id"])
        )
        _persist(state_path, event_path, state, event)
        return
    if submission.get("intent"):
        event = _mark_fail(state, case_id, "SUBMISSION_UNCERTAIN", stage=stage, receipt=str(receipt_path))
        _persist(state_path, event_path, state, event)
        return
    argv = _submission_args(manifest, case, stage)
    submission["intent"] = {"timestamp_utc": _utc(), "argv": argv}
    event = _record_event(state, case_id, "submission_intent_persisted", stage=stage)
    _persist(state_path, event_path, state, event)
    completed = submitter(argv, run_root)
    existing = _read_receipt(receipt_path, case_id=case_id, stage=stage, expected_sha=str(manifest["expected_sha"]))
    if existing is None:
        event = _mark_fail(
            state, case_id, "SUBMISSION_UNCERTAIN", stage=stage, returncode=completed.returncode,
            stdout=completed.stdout.strip(), stderr=completed.stderr.strip(), receipt=str(receipt_path),
        )
        _persist(state_path, event_path, state, event)
        return
    submission.update({"receipt": str(receipt_path), "job_id": str(existing["job_id"]), "completed_utc": _utc()})
    target = "SUBMITTED_FAULT" if stage == "fault" else "RECOVERY_SUBMITTED"
    event = _transition(state, case_id, target, "numeric_receipt_persisted", stage=stage, job_id=submission["job_id"])
    _persist(state_path, event_path, state, event)


def _contract_pass(path: Path, case: Mapping[str, Any]) -> bool:
    try:
        check = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        check.get("status") == "PASS" and check.get("contract_match") is True
        and check.get("fault_id") == case["fault_id"] and check.get("target_screen") == case["fault_screen"]
    )


def _recovery_bootstrap_order(case_root: Path, *, expected_sha: str, case_id: str) -> tuple[bool, dict[str, Any]]:
    """Read-only proof that the serial bootstrap precedes every hydro claim."""
    receipt_path = case_root / "recovery" / "restart_reconstructed.json"
    manifest_path = case_root / "injected" / "lifecycle" / "streaming_manifest.json"
    try:
        receipt, manifest = _read_json(receipt_path), _read_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        return False, {"reason": "missing_or_invalid_recovery_bootstrap_receipt", "error_type": type(exc).__name__}
    if (
        receipt.get("schema") != "khz_filament.hr4e5s.s5.recovery_bootstrap.v1"
        or receipt.get("status") != "PASS"
        or receipt.get("bootstrap_event") != "RESTART_RECONSTRUCTED"
        or receipt.get("runtime_sha") != expected_sha
        or receipt.get("case_id") != case_id
    ):
        return False, {"reason": "recovery_bootstrap_receipt_identity_invalid"}
    events = manifest.get("telemetry_events")
    index = receipt.get("telemetry_event_index")
    if isinstance(index, bool) or not isinstance(index, int) or not isinstance(events, list) or not 0 <= index < len(events):
        return False, {"reason": "recovery_bootstrap_receipt_telemetry_invalid"}
    event = events[index]
    if not isinstance(event, Mapping) or event.get("event") != "RESTART_RECONSTRUCTED" or event.get("actor") != "s5_restart" or event.get("bootstrap") is not True:
        return False, {"reason": "recovery_bootstrap_event_missing"}
    bootstrap_indexes = [
        event_index for event_index, item in enumerate(events)
        if isinstance(item, Mapping) and item.get("event") == "RESTART_RECONSTRUCTED"
    ]
    claim_indexes = [
        event_index for event_index, item in enumerate(events)
        if isinstance(item, Mapping) and item.get("event") == "HYDRO_CLAIM" and item.get("actor") == "hydro_consumer"
    ]
    if bootstrap_indexes != [index] or not claim_indexes or index >= min(claim_indexes):
        return False, {
            "reason": "RECOVERY_BOOTSTRAP_ORDER_VIOLATION",
            "bootstrap_indexes": bootstrap_indexes,
            "hydro_claim_indexes": claim_indexes,
        }
    return True, {
        "receipt": str(receipt_path), "bootstrap_event_index": index,
        "first_hydro_claim_index": min(claim_indexes), "bootstrap_before_first_claim": True,
        "queue_size": receipt.get("queue_size"), "backlog_size": receipt.get("backlog_size"),
        "reconstructed_pending_post_count": receipt.get("reconstructed_pending_post_count"),
        "reconstructed_stale_hydro_running_count": receipt.get("reconstructed_stale_hydro_running_count"),
    }


def _exact_pass(case_root: Path, *, expected_sha: str | None = None, case_id: str | None = None) -> tuple[bool, str, dict[str, Any] | None]:
    audit_path = case_root / "recovery" / "final_lifecycle_audit.json"
    comparison_path = case_root / "comparison" / "s5_1_exact_comparison.json"
    try:
        audit, comparison = _read_json(audit_path), _read_json(comparison_path)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"missing_or_invalid_final_evidence:{type(exc).__name__}", None
    audit_ok = (
        audit.get("pointer_present") is True
        and isinstance(audit.get("barrier"), Mapping) and audit["barrier"].get("status") == "PASS"
        and isinstance(audit.get("promotion"), Mapping) and audit["promotion"].get("authoritative_namespace") == "NEXT"
    )
    comparison_ok = (
        comparison.get("status") == "PASS" and comparison.get("expected_field_comparisons") == 432
        and comparison.get("completed_field_comparisons") == 432 and comparison.get("mismatch_count") == 0
        and all(comparison.get(key) is True for key in (
            "completion_map_exact", "ownership_exact", "barrier_exact", "promotion_generation_exact",
            "authoritative_manifest_exact", "final_artifact_inventory_clean",
        ))
        and isinstance(comparison.get("recovery_provenance"), Mapping)
        and comparison["recovery_provenance"].get("status") == "PASS"
        and comparison["recovery_provenance"].get("recovery_provenance_exact") is True
    )
    if expected_sha is not None and case_id is not None:
        bootstrap_ok, bootstrap_detail = _recovery_bootstrap_order(case_root, expected_sha=expected_sha, case_id=case_id)
        if not bootstrap_ok:
            return False, "RECOVERY_BOOTSTRAP_ORDER_VIOLATION", bootstrap_detail
    else:
        bootstrap_detail = None
    return (audit_ok and comparison_ok, "exact_evidence_pass" if audit_ok and comparison_ok else "exact_evidence_requirement_failed", bootstrap_detail)


def _scheduler_success(snapshot: Mapping[str, Any]) -> bool:
    accounting = snapshot.get("accounting")
    return isinstance(accounting, Mapping) and accounting.get("state") == "COMPLETED" and accounting.get("exit_code") == "0:0"


def _update_final_reports(manifest_path: Path, manifest: Mapping[str, Any], state: dict[str, Any]) -> None:
    _, _, csv_path, summary_path, report_path = _state_paths(manifest_path)
    rows = []
    for case_id in (str(item["case_id"]) for item in manifest["cases"]):
        case = state["cases"][case_id]
        rows.append({
            "case_id": case_id, "state": case["state"],
            "fault_job_id": case.get("fault_submission", {}).get("job_id", ""),
            "fault_terminal": case.get("fault_scheduler", {}).get("accounting", {}).get("state", ""),
            "recovery_job_id": case.get("recovery_submission", {}).get("job_id", ""),
            "recovery_terminal": case.get("recovery_scheduler", {}).get("accounting", {}).get("state", ""),
            "failure_reason": case.get("failure", {}).get("reason", ""),
        })
    all_terminal = all(row["state"] in CASE_TERMINAL for row in rows)
    all_pass = all(row["state"] == "PASS" for row in rows)
    state["monitor_status"] = "READY_FOR_S5_1_MANUAL_REVIEW" if all_pass else ("READY_FOR_S5_1_DEFECT_REVIEW" if all_terminal else "OPEN")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=csv_path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
        temporary_csv = Path(handle.name)
    os.replace(temporary_csv, csv_path)
    summary = {
        "schema": SCHEMA, "execution_sha": manifest["expected_sha"],
        "clean_reference_job": str(manifest["clean_reference_job"]),
        "historical_clean_reference_job": str(manifest["historical_clean_reference_job"]),
        "monitor_status": state["monitor_status"], "cases": rows,
        "duplicate_submission_count": 0,
        "lost_screen_count": 0 if all_pass else None,
        "temp_as_authoritative_count": 0 if all_pass else None,
        "premature_barrier_count": 0 if all_pass else None,
        "premature_promotion_count": 0 if all_pass else None,
        "mixed_generation_count": 0 if all_pass else None,
        "science_mismatch": False if all_pass else None,
        "recovery_bootstrap_order": {
            row["case_id"]: state["cases"][row["case_id"]].get("recovery_bootstrap_order") for row in rows
        },
    }
    _atomic_json(summary_path, summary)
    report = [
        f"# HR-4E-5S S5-1 recovery monitor ({','.join(row['case_id'] for row in rows)})", "",
        f"- execution SHA: `{manifest['expected_sha']}`",
        f"- new clean reference: `{manifest['clean_reference_job']}` (exact-equivalent to historical `238465`)",
        f"- monitor status: `{state['monitor_status']}`", "",
        "| Case | State | Fault job | Fault terminal | Recovery job | Recovery terminal | Failure |", "|---|---|---:|---|---:|---|---|",
    ]
    report.extend(f"| {row['case_id']} | {row['state']} | {row['fault_job_id']} | {row['fault_terminal']} | {row['recovery_job_id']} | {row['recovery_terminal']} | {row['failure_reason']} |" for row in rows)
    report_payload = "\n".join(report) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=report_path.name + ".", suffix=".tmp", dir=report_path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(report_payload); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, report_path)


def advance_matrix(
    manifest_path: str | Path,
    *,
    scheduler_query: Callable[[str], dict[str, Any]] = _scheduler_snapshot,
    submitter: Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]] = _run,
) -> dict[str, Any]:
    """Advance each case at most through the next durable lifecycle gate."""
    source = Path(manifest_path)
    manifest = load_manifest(source)
    state_path, event_path, _, _, _ = _state_paths(source)
    state = load_or_create_state(source, manifest)
    for case in manifest["cases"]:
        case_id = str(case["case_id"])
        entry = state["cases"][case_id]
        current = str(entry["state"])
        if current in CASE_TERMINAL:
            continue
        if current == INITIAL:
            _submit_once(manifest, state, state_path, event_path, case, "fault", submitter)
            continue
        if current in {"SUBMITTED_FAULT", "FAULT_RUNNING"}:
            job_id = entry.get("fault_submission", {}).get("job_id")
            if not isinstance(job_id, str) or not job_id.isdigit():
                event = _mark_fail(state, case_id, "MISSING_FAULT_JOB_ID")
            else:
                snapshot = scheduler_query(job_id)
                entry["fault_scheduler"] = snapshot
                if snapshot.get("terminal"):
                    event = _transition(state, case_id, "FAULT_TERMINAL", "sacct_terminal", scheduler=snapshot)
                elif snapshot.get("state") != "UNKNOWN":
                    event = _transition(state, case_id, "FAULT_RUNNING", "scheduler_active", scheduler=snapshot)
                else:
                    event = _record_event(state, case_id, "scheduler_state_unavailable", stage="fault", scheduler=snapshot)
            _persist(state_path, event_path, state, event)
            continue
        if current == "FAULT_TERMINAL":
            event = _transition(state, case_id, "FAULT_AUDIT_PENDING", "await_persisted_injected_audit")
            _persist(state_path, event_path, state, event)
            continue
        if current == "FAULT_AUDIT_PENDING":
            case_root = Path(str(manifest["run_root"])) / case_id
            audit_path, contract_path = case_root / "injected" / "disk_state_audit.json", case_root / "contract_check.json"
            if not audit_path.is_file() or not contract_path.is_file():
                event = _transition(state, case_id, "FAULT_CONTRACT_FAIL", "missing_fault_audit_or_contract", audit=str(audit_path), contract=str(contract_path))
            else:
                try:
                    injected_audit = _read_json(audit_path)
                except (OSError, json.JSONDecodeError) as exc:
                    event = _transition(state, case_id, "FAULT_CONTRACT_FAIL", "invalid_injected_audit", error_type=type(exc).__name__)
                else:
                    if _contract_pass(contract_path, case) and isinstance(injected_audit.get("fault_provenance"), list):
                        entry["injected_audit_sha256"] = _sha256_file(audit_path)
                        entry["contract_check_sha256"] = _sha256_file(contract_path)
                        event = _transition(state, case_id, "FAULT_CONTRACT_PASS", "persisted_contract_match")
                    else:
                        event = _transition(state, case_id, "FAULT_CONTRACT_FAIL", "contract_match_false", contract=str(contract_path))
            _persist(state_path, event_path, state, event)
            continue
        if current == "FAULT_CONTRACT_PASS":
            _submit_once(manifest, state, state_path, event_path, case, "recovery", submitter)
            continue
        if current in {"RECOVERY_SUBMITTED", "RECOVERY_RUNNING"}:
            job_id = entry.get("recovery_submission", {}).get("job_id")
            if not isinstance(job_id, str) or not job_id.isdigit():
                event = _mark_fail(state, case_id, "MISSING_RECOVERY_JOB_ID")
            else:
                snapshot = scheduler_query(job_id)
                entry["recovery_scheduler"] = snapshot
                if snapshot.get("terminal"):
                    event = _transition(state, case_id, "RECOVERY_TERMINAL", "sacct_terminal", scheduler=snapshot)
                elif snapshot.get("state") != "UNKNOWN":
                    event = _transition(state, case_id, "RECOVERY_RUNNING", "scheduler_active", scheduler=snapshot)
                else:
                    event = _record_event(state, case_id, "scheduler_state_unavailable", stage="recovery", scheduler=snapshot)
            _persist(state_path, event_path, state, event)
            continue
        if current == "RECOVERY_TERMINAL":
            snapshot = entry.get("recovery_scheduler", {})
            if not _scheduler_success(snapshot):
                event = _mark_fail(state, case_id, "RECOVERY_TERMINAL_FAILURE", scheduler=snapshot)
            else:
                event = _transition(state, case_id, "EXACT_COMPARE_PENDING", "recovery_completed_0_0")
            _persist(state_path, event_path, state, event)
            continue
        if current == "EXACT_COMPARE_PENDING":
            passed, reason, bootstrap_detail = _exact_pass(
                Path(str(manifest["run_root"])) / case_id,
                expected_sha=str(manifest["expected_sha"]), case_id=case_id,
            )
            if bootstrap_detail is not None:
                entry["recovery_bootstrap_order"] = bootstrap_detail
            event = _transition(state, case_id, "PASS", reason) if passed else _mark_fail(state, case_id, reason, recovery_bootstrap_order=bootstrap_detail)
            _persist(state_path, event_path, state, event)
            continue
        event = _mark_fail(state, case_id, "UNKNOWN_MONITOR_STATE", observed=current)
        _persist(state_path, event_path, state, event)
    _update_final_reports(source, manifest, state)
    _persist(state_path, event_path, state)
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=3600)
    parser.add_argument("--resume", action="store_true", help="resume or initialize persisted monitor state")
    parser.add_argument("--once", action="store_true", help="perform one state-machine pass and exit")
    args = parser.parse_args(argv)
    if not 60 <= args.poll_seconds <= 3600:
        parser.error("--poll-seconds must be between 60 and 3600")
    if not args.resume:
        parser.error("--resume is required so an invocation never ignores persisted state")
    while True:
        state = advance_matrix(args.manifest)
        print(json.dumps({"monitor_status": state["monitor_status"], "updated_utc": state["updated_utc"]}, sort_keys=True))
        if args.once or state["monitor_status"] != "OPEN":
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
