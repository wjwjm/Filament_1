#!/usr/bin/env python3
"""Persistent, receipt-safe controller for the single S5-FINAL worker loss."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "khz_filament.hr4e5s.s5_final.monitor.v1"
TERMINAL = {"PASS", "READY_FOR_S5_FINAL_DEFECT_REVIEW"}
SINGLE_ALLOCATION_MODE = "SINGLE_ALLOCATION_TWO_EXECUTION_EPOCHS"


def phase0_pre_signal_gate(*, target: Mapping[str, Any], survivor: Mapping[str, Any],
                           controller_hostname: str, target_listpids: Mapping[str, Any],
                           survivor_listpids: Mapping[str, Any], target_heartbeat_live: bool,
                           survivor_heartbeat_live: bool, signal_intent_exists: bool,
                           signal_command: Sequence[str]) -> dict[str, Any]:
    """Fail-closed contract for the allocation-internal observability gate.

    This function deliberately deals only with Slurm/process receipts.  It
    must remain usable before a CUDA context, lifecycle root, or scientific
    worker exists.
    """
    checks: list[dict[str, Any]] = []

    def check(name: str, value: bool) -> None:
        checks.append({"name": name, "pass": bool(value)})

    target_step = f"{target.get('job_id')}.{target.get('step_id')}"
    survivor_step = f"{survivor.get('job_id')}.{survivor.get('step_id')}"
    identity_keys = ("actor", "job_id", "step_id", "pid", "hostname")
    target_valid = all(target.get(key) not in {None, ""} for key in identity_keys)
    survivor_valid = all(survivor.get(key) not in {None, ""} for key in identity_keys)
    check("target_identity_complete", target_valid)
    check("survivor_identity_complete", survivor_valid)
    check("actors_are_distinct", target.get("actor") == "probe_target" and survivor.get("actor") == "probe_survivor")
    check("steps_are_distinct", target_step != survivor_step)
    check("pids_are_distinct", target.get("pid") != survivor.get("pid"))
    check("controller_on_target_compute_node", controller_hostname == target.get("hostname"))
    check("controller_on_survivor_compute_node", controller_hostname == survivor.get("hostname"))
    check("target_heartbeat_live", target_heartbeat_live)
    check("survivor_heartbeat_live", survivor_heartbeat_live)
    check("target_listpids_returned", target_listpids.get("returncode") == 0)
    check("survivor_listpids_returned", survivor_listpids.get("returncode") == 0)
    check("target_pid_currently_proven", str(target.get("pid")) in str(target_listpids.get("stdout", "")))
    check("survivor_pid_currently_proven", str(survivor.get("pid")) in str(survivor_listpids.get("stdout", "")))
    check("signal_intent_unused", not signal_intent_exists)
    check("signal_scope_is_exact_target_step", list(signal_command) == ["scancel", "--signal=TERM", target_step])
    return {"schema": SCHEMA, "kind": "phase0_pre_signal_gate", "status": "PASS" if all(item["pass"] for item in checks) else "FAIL", "checks": checks}


def phase0_post_signal_gate(*, target_pid: int, target_wait_nonzero: bool,
                            target_still_live: bool, survivor_still_live: bool,
                            target_listpids_after: Mapping[str, Any]) -> dict[str, Any]:
    """Prove that the selected target, not the allocation, was terminated."""
    checks = [
        {"name": "target_step_exited", "pass": bool(target_wait_nonzero)},
        {"name": "target_pid_not_live", "pass": not target_still_live},
        {"name": "survivor_still_live", "pass": bool(survivor_still_live)},
        {"name": "target_pid_absent_from_post_signal_listpids", "pass": str(target_pid) not in str(target_listpids_after.get("stdout", ""))},
    ]
    return {"schema": SCHEMA, "kind": "phase0_post_signal_gate", "status": "PASS" if all(item["pass"] for item in checks) else "FAIL", "checks": checks}


def _read(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def _run(args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), cwd=cwd, text=True, capture_output=True, check=False)


def _state_paths(manifest: Path) -> tuple[Path, Path]:
    return manifest.with_name("monitor_state.json"), manifest.with_name("monitor_events.jsonl")


def _initial_state(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": SCHEMA, "status": "WAIT_FOR_DUAL_CLAIMS", "run_root": manifest["run_root"], "initial_job_id": manifest["initial_job_id"], "recovery_job_id": None, "signal_sent": False, "updated_epoch": time.time()}


def _event(state: dict[str, Any], name: str, **detail: Any) -> dict[str, Any]:
    state["updated_epoch"] = time.time()
    return {"schema": SCHEMA, "event": name, "status": state["status"], "epoch": state["updated_epoch"], **detail}


def _receipt_job(path: Path, expected_mode: str, expected_sha: str) -> str | None:
    try:
        rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
        row = rows[0]
        return str(row["job_id"]) if len(rows) == 1 and row.get("case_mode") == expected_mode and row.get("execution_sha") == expected_sha and str(row.get("job_id", "")).isdigit() else None
    except (OSError, KeyError, IndexError, csv.Error):
        return None


def _scheduler(job_id: str, cwd: Path) -> dict[str, Any]:
    queue = _run(["squeue", "-h", "-j", job_id, "-o", "%T"], cwd)
    queued = queue.stdout.strip()
    accounting = _run(["sacct", "-X", "-j", job_id, "--format=State,ExitCode", "--parsable2", "--noheader"], cwd)
    lines = [line for line in accounting.stdout.splitlines() if line.strip()]
    state, exit_code = (lines[0].split("|", 2) + ["", ""])[:2] if lines else ("", "")
    terminal = state.split()[0] in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}
    return {"queue_state": queued, "sacct_state": state, "exit_code": exit_code, "terminal": terminal, "queue_returncode": queue.returncode, "sacct_returncode": accounting.returncode}


def _identities(case_root: Path, phase: str, job_id: str) -> tuple[list[dict[str, Any]], str | None]:
    directory = case_root / ("identities" if phase == "initial" else "recovery/identities")
    names = ("optical_producer", "hydro_consumer_0", "hydro_consumer_1")
    try:
        values = [_read(directory / f"{name}.json") for name in names]
    except (OSError, json.JSONDecodeError):
        return [], "identity_receipts_incomplete"
    if any(v.get("status") != "READY" or v.get("actor") not in names or str(v.get("job_id")) != job_id or not str(v.get("step_id", "")).isdigit() or not isinstance(v.get("worker_pid"), int) for v in values):
        return [], "identity_receipts_invalid"
    if len({str(v["step_id"]) for v in values}) != 3 or len({int(v["worker_pid"]) for v in values}) != 3:
        return [], "identity_steps_or_pids_not_distinct"
    if any(not str(v.get("gpu_visible_devices", "")) for v in values):
        return [], "gpu_identity_missing"
    return values, None


def _live_claims(case_root: Path) -> list[dict[str, Any]]:
    try:
        manifest = _read(case_root / "lifecycle/streaming_manifest.json")
    except (OSError, json.JSONDecodeError):
        return []
    claims = []
    for record in manifest.get("records", []):
        transitions = record.get("transitions", [])
        last = transitions[-1] if transitions else {}
        if record.get("state") == "HYDRO_RUNNING" and record.get("post") is not None and record.get("next") is None and str(last.get("actor", "")).startswith("hydro_consumer_"):
            claims.append({"actor": str(last["actor"]), "ordinal": int(record["ordinal"])})
    barrier = manifest.get("barrier")
    has_next = any(record.get("next") is not None for record in manifest.get("records", []))
    if isinstance(barrier, Mapping) and barrier.get("status") == "PASS":
        return []
    return claims if has_next else []


def _step_has_pid(identity: Mapping[str, Any], cwd: Path) -> tuple[bool, str]:
    step = f"{identity['job_id']}.{identity['step_id']}"
    result = _run(["scontrol", "listpids", step], cwd)
    return str(identity["worker_pid"]) in result.stdout, result.stdout


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _defect(state: dict[str, Any], reason: str, **detail: Any) -> dict[str, Any]:
    state["status"] = "READY_FOR_S5_FINAL_DEFECT_REVIEW"; state["defect"] = {"reason": reason, **detail}
    return _event(state, "defect", reason=reason, **detail)


def advance(manifest_path: Path) -> dict[str, Any]:
    manifest = _read(manifest_path)
    state_path, events_path = _state_paths(manifest_path)
    state = _read(state_path) if state_path.exists() else _initial_state(manifest)
    root, case_root, cwd = Path(manifest["run_root"]), Path(manifest["run_root"]) / "scenario", Path(manifest["repo"])
    event: dict[str, Any]
    if state["status"] in TERMINAL:
        return state
    if state["status"] == "WAIT_FOR_DUAL_CLAIMS":
        scheduler = _scheduler(str(manifest["initial_job_id"]), cwd)
        identities, problem = _identities(case_root, "initial", str(manifest["initial_job_id"]))
        claims = _live_claims(case_root)
        target = str(manifest["target_actor"])
        active = {item["actor"] for item in claims}
        if scheduler["terminal"]:
            event = _defect(
                state,
                "INITIAL_JOB_TERMINAL_BEFORE_SAFE_SIGNAL",
                scheduler=scheduler,
                identity_problem=problem,
                active_actors=sorted(active),
            )
        elif problem or target not in active or len(active) < 2:
            event = _event(state, "awaiting_safe_target", identity_problem=problem, active_actors=sorted(active))
        else:
            identity = next(v for v in identities if v["actor"] == target)
            live, raw = _step_has_pid(identity, cwd)
            if not live:
                event = _defect(state, "TARGET_STEP_PID_UNPROVEN", actor=target, step=f"{identity['job_id']}.{identity['step_id']}")
            else:
                receipt = case_root / "worker_loss_receipt.json"
                if receipt.exists():
                    event = _defect(state, "DUPLICATE_WORKER_LOSS_RECEIPT")
                else:
                    command = ["scancel", "--signal=KILL", f"{identity['job_id']}.{identity['step_id']}"]
                    signal = _run(command, cwd)
                    if signal.returncode != 0:
                        event = _defect(state, "TARGET_SIGNAL_FAILED", returncode=signal.returncode)
                    else:
                        _write_json_once(receipt, {"schema": SCHEMA, "status": "SENT", "target_actor": target, "target_identity": identity, "active_claims": claims, "scontrol_listpids_before": raw, "signal_command": command, "signal_returncode": signal.returncode})
                        state["signal_sent"] = True; state["status"] = "WAIT_FOR_OLD_JOB_QUIESCENCE"
                        event = _event(state, "worker_loss_signal_sent", actor=target)
    elif state["status"] == "WAIT_FOR_OLD_JOB_QUIESCENCE":
        scheduler = _scheduler(str(manifest["initial_job_id"]), cwd)
        identities, problem = _identities(case_root, "initial", str(manifest["initial_job_id"]))
        live = [] if problem else [identity["actor"] for identity in identities if _step_has_pid(identity, cwd)[0]]
        failure = case_root / "initial_failure_propagation.json"
        if not scheduler["terminal"] or live:
            event = _event(state, "awaiting_old_writer_quiescence", scheduler=scheduler, live_actors=live)
        elif scheduler["sacct_state"].split()[0] == "COMPLETED" and scheduler["exit_code"] == "0:0":
            event = _defect(state, "OLD_JOB_DID_NOT_FAIL_FAST", scheduler=scheduler)
        elif not failure.is_file():
            event = _defect(state, "FAILURE_PROPAGATION_RECEIPT_MISSING", scheduler=scheduler)
        else:
            quiescence = case_root / "old_job_quiescence.json"
            _write_json_once(quiescence, {"schema": SCHEMA, "status": "PASS", "scheduler": scheduler, "identity_receipts": identities, "live_writer_actors": live})
            runner = [sys.executable, str(Path(manifest["repo"]) / "Filament_python/tools/run_hr4e5s_s5_final.py")]
            snapshot = _run(runner + ["snapshot", "--stream-root", str(case_root / "lifecycle"), "--out", str(case_root / "interrupted_state_inventory.json")], cwd)
            effects = _run(runner + ["freeze-effects", "--stream-root", str(case_root / "lifecycle"), "--inventory", str(case_root / "interrupted_state_inventory.json"), "--out", str(case_root / "expected_recovery_effects.json")], cwd)
            if snapshot.returncode or effects.returncode:
                event = _defect(state, "RECOVERY_EFFECTS_FREEZE_FAILED", snapshot_returncode=snapshot.returncode, effects_returncode=effects.returncode)
            else:
                state["status"] = "RECOVERY_SUBMISSION_PENDING"; event = _event(state, "quiescence_and_effects_frozen")
    elif state["status"] == "RECOVERY_SUBMISSION_PENDING":
        receipt = root / "recovery_submission_receipt.tsv"
        existing = _receipt_job(receipt, "recovery", str(manifest["expected_sha"])) if receipt.is_file() else None
        if existing is None:
            intent = root / "recovery_submission_intent.json"
            if intent.exists():
                event = _defect(state, "RECOVERY_SUBMISSION_UNCERTAIN", intent=str(intent))
            else:
                _write_json_once(intent, {"schema": SCHEMA, "status": "INTENT", "expected_sha": manifest["expected_sha"]})
                submit = _run([str(manifest["submit_script"]), str(manifest["repo"]), str(root), str(manifest["expected_sha"]), str(manifest["preflight"]), "recovery", str(manifest["reference_case_root"])], cwd)
                existing = _receipt_job(receipt, "recovery", str(manifest["expected_sha"]))
                if submit.returncode or existing is None:
                    event = _defect(state, "RECOVERY_SUBMISSION_UNCERTAIN", submit_returncode=submit.returncode)
                else:
                    state["recovery_job_id"] = existing; state["status"] = "WAIT_FOR_RECOVERY_TERMINAL"; event = _event(state, "recovery_submitted", job_id=existing)
        else:
            state["recovery_job_id"] = existing; state["status"] = "WAIT_FOR_RECOVERY_TERMINAL"; event = _event(state, "existing_recovery_receipt_reused", job_id=existing)
    elif state["status"] == "WAIT_FOR_RECOVERY_TERMINAL":
        scheduler = _scheduler(str(state["recovery_job_id"]), cwd)
        if not scheduler["terminal"]:
            event = _event(state, "awaiting_recovery_terminal", scheduler=scheduler)
        elif scheduler["sacct_state"].split()[0] != "COMPLETED" or scheduler["exit_code"] != "0:0":
            event = _defect(state, "RECOVERY_TERMINAL_FAILURE", scheduler=scheduler)
        else:
            try:
                exact = _read(case_root / "comparison/exact_comparison.json")
                bootstrap = _read(case_root / "recovery/restart_reconstructed.json")
                checks_ok = exact.get("status") == "PASS" and exact.get("expected_field_comparisons") == 432 and exact.get("completed_field_comparisons") == 432 and exact.get("mismatch_count") == 0 and exact.get("recovery_provenance", {}).get("status") == "PASS" and bootstrap.get("bootstrap_event") == "S5_FINAL_RESTART_RECONSTRUCTED"
            except (OSError, json.JSONDecodeError):
                checks_ok = False
            if not checks_ok:
                event = _defect(state, "RECOVERY_AUDIT_FAILED_OR_MISSING")
            else:
                state["status"] = "PASS"; event = _event(state, "s5_final_execution_pass")
    else:
        event = _defect(state, "UNKNOWN_MONITOR_STATE", observed=state["status"])
    _atomic(state_path, state); _append(events_path, event)
    return state


def _single_paths(manifest: Path) -> tuple[Path, Path, Path]:
    return (manifest.with_name("single_allocation_state.json"),
            manifest.with_name("single_allocation_events.jsonl"),
            manifest.with_name("controller_defect.json"))


def _single_initial(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": SCHEMA, "execution_mode": SINGLE_ALLOCATION_MODE,
            "status": "WAIT_FOR_DUAL_ARMING", "run_root": manifest["run_root"],
            "allocation_id": str(manifest["allocation_id"]), "signal_sent": False,
            "recovery_is_new_allocation": False,
            "recovery_is_fresh_process_restart": True, "updated_epoch": time.time()}


def _single_defect(root: Path, state: dict[str, Any], reason: str, **detail: Any) -> dict[str, Any]:
    state["status"] = "READY_FOR_S5_FINAL_DEFECT_REVIEW"
    state["defect"] = {"reason": reason, **detail}
    _atomic(root / "controller_defect.json", {"schema": SCHEMA, "status": "FAIL", **state["defect"]})
    return _event(state, "defect", reason=reason, **detail)


def _single_identities(case_root: Path, allocation_id: str) -> tuple[list[dict[str, Any]], str | None]:
    directory = case_root / "initial/identities"
    names = ("optical_producer", "hydro_consumer_0", "hydro_consumer_1")
    try:
        values = [_read(directory / f"{name}.json") for name in names]
    except (OSError, json.JSONDecodeError):
        return [], "identity_receipts_incomplete"
    if any(v.get("status") != "READY" or v.get("actor") not in names or str(v.get("job_id")) != allocation_id or v.get("execution_epoch") != "initial" for v in values):
        return [], "identity_receipts_invalid"
    if len({str(v.get("step_id")) for v in values}) != 3 or len({int(v.get("worker_pid", -1)) for v in values}) != 3:
        return [], "identity_steps_or_pids_not_distinct"
    return values, None


def _single_armings(case_root: Path, identities: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    expected = {str(item["actor"]): item for item in identities if str(item["actor"]).startswith("hydro_consumer_")}
    try:
        values = [_read(case_root / "initial/arming" / f"{actor}.json") for actor in sorted(expected)]
    except (OSError, json.JSONDecodeError):
        return [], "arming_receipts_incomplete"
    for value in values:
        identity = expected.get(str(value.get("actor")))
        if identity is None or value.get("status") != "ARMED" or value.get("execution_epoch") != "initial" or int(value.get("worker_pid", -1)) != int(identity["worker_pid"]) or value.get("node") != identity.get("node") or not isinstance(value.get("block"), list) or not value["block"]:
            return [], "arming_receipts_invalid"
    return values, None


def _single_listpids(identity: Mapping[str, Any], cwd: Path) -> tuple[bool, dict[str, Any]]:
    step = f"{identity['job_id']}.{identity['step_id']}"
    started = time.time(); result = _run(["scontrol", "listpids", step], cwd); finished = time.time()
    evidence = {"command": ["scontrol", "listpids", step], "hostname": socket.gethostname(),
                "target_node": identity.get("node"), "target_pid": int(identity["worker_pid"]),
                "started_epoch": started, "finished_epoch": finished, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    return result.returncode == 0 and str(identity["worker_pid"]) in result.stdout, evidence


def advance_single_allocation(manifest_path: Path) -> dict[str, Any]:
    """Controller for one allocation with two fresh scientific-process epochs."""
    manifest = _read(manifest_path)
    if manifest.get("execution_mode") != SINGLE_ALLOCATION_MODE:
        raise ValueError("single-allocation controller manifest mode is invalid")
    state_path, events_path, _ = _single_paths(manifest_path)
    state = _read(state_path) if state_path.exists() else _single_initial(manifest)
    root, case_root, cwd = Path(manifest["run_root"]), Path(manifest["run_root"]) / "scenario", Path(manifest["repo"])
    ready = case_root / "controller_ready.json"
    if not ready.exists():
        _atomic(ready, {"schema": SCHEMA, "status": "READY", "execution_mode": SINGLE_ALLOCATION_MODE,
                        "controller_pid": os.getpid(), "controller_hostname": socket.gethostname(),
                        "allocation_id": str(manifest["allocation_id"])})
    if state["status"] in TERMINAL:
        return state
    if state["status"] == "WAIT_FOR_DUAL_ARMING":
        identities, problem = _single_identities(case_root, str(manifest["allocation_id"]))
        armings, arm_problem = _single_armings(case_root, identities) if not problem else ([], problem)
        if problem or arm_problem:
            event = _event(state, "awaiting_arming", identity_problem=problem, arming_problem=arm_problem)
        else:
            target = str(manifest["target_actor"])
            identity = next(item for item in identities if item["actor"] == target)
            hydro_identities = [item for item in identities if str(item["actor"]).startswith("hydro_consumer_")]
            observations = []
            for item in hydro_identities:
                live, evidence = _single_listpids(item, cwd)
                observations.append({"actor": item["actor"], "live": live, "evidence": evidence})
            live = all(bool(item["live"]) for item in observations)
            evidence = next(item["evidence"] for item in observations if item["actor"] == target)
            evidence_path = case_root / "initial/local_listpids_evidence.json"
            _atomic(evidence_path, {"schema": SCHEMA, "status": "PASS" if live else "FAIL", "arming": armings, "observations": observations})
            if not live:
                event = _single_defect(root, state, "TARGET_STEP_PID_UNPROVEN", actor=target, step=f"{identity['job_id']}.{identity['step_id']}")
            else:
                intent = case_root / "initial/worker_loss_intent.json"
                if intent.exists():
                    event = _single_defect(root, state, "DUPLICATE_WORKER_LOSS_INTENT")
                else:
                    _atomic(intent, {"schema": SCHEMA, "status": "INTENT", "target_identity": identity, "arming": armings, "listpids_evidence": str(evidence_path)})
                    command = ["scancel", "--signal=KILL", f"{identity['job_id']}.{identity['step_id']}"]
                    signal = _run(command, cwd)
                    if signal.returncode:
                        event = _single_defect(root, state, "TARGET_SIGNAL_FAILED", command=command, returncode=signal.returncode, stderr=signal.stderr)
                    else:
                        _atomic(case_root / "initial/worker_loss_receipt.json", {"schema": SCHEMA, "status": "SENT", "target_identity": identity, "signal_command": command, "signal_returncode": signal.returncode, "listpids_evidence": str(evidence_path)})
                        state["signal_sent"] = True; state["status"] = "WAIT_FOR_INITIAL_QUIESCENCE"
                        event = _event(state, "worker_loss_signal_sent", actor=target)
    elif state["status"] == "WAIT_FOR_INITIAL_QUIESCENCE":
        stopped = case_root / "initial/initial_workers_stopped.json"
        if not stopped.is_file():
            event = _event(state, "awaiting_initial_worker_cleanup")
        else:
            identities, problem = _single_identities(case_root, str(manifest["allocation_id"]))
            if problem:
                event = _single_defect(root, state, "INITIAL_IDENTITY_LOST_BEFORE_QUIESCENCE", identity_problem=problem)
            else:
                evidence = []; live = False
                for identity in identities:
                    present, item = _single_listpids(identity, cwd); evidence.append(item); live = live or present
                # A terminated Slurm step may no longer be queryable.  Its absence,
                # with all prior identity receipts retained, is the required
                # quiescence evidence; only a still-live old writer is a failure.
                if live:
                    event = _single_defect(root, state, "INITIAL_EXECUTION_NOT_QUIESCENT", listpids=evidence)
                else:
                    _atomic(case_root / "initial_execution_quiescence.json", {"schema": SCHEMA, "status": "PASS", "initial_workers_stopped": _read(stopped), "listpids": evidence})
                    runner = [sys.executable, str(cwd / "Filament_python/tools/run_hr4e5s_s5_final.py")]
                    snapshot = _run(runner + ["snapshot", "--stream-root", str(case_root / "lifecycle"), "--out", str(case_root / "interrupted_state_inventory.json")], cwd)
                    effects = _run(runner + ["freeze-effects", "--stream-root", str(case_root / "lifecycle"), "--inventory", str(case_root / "interrupted_state_inventory.json"), "--out", str(case_root / "expected_recovery_effects.json")], cwd)
                    if snapshot.returncode or effects.returncode:
                        event = _single_defect(root, state, "RECOVERY_EFFECTS_FREEZE_FAILED", snapshot_returncode=snapshot.returncode, effects_returncode=effects.returncode)
                    else:
                        _atomic(case_root / "recovery_ready.json", {"schema": SCHEMA, "status": "PASS", "execution_mode": SINGLE_ALLOCATION_MODE, "recovery_is_new_allocation": False, "recovery_is_fresh_process_restart": True})
                        state["status"] = "WAIT_FOR_FINAL_AUDIT"; event = _event(state, "recovery_ready")
    elif state["status"] == "WAIT_FOR_FINAL_AUDIT":
        exact_path, bootstrap_path = case_root / "comparison/exact_comparison.json", case_root / "recovery/restart_reconstructed.json"
        if not exact_path.is_file() or not bootstrap_path.is_file():
            event = _event(state, "awaiting_recovery_final_audit")
        else:
            exact, bootstrap = _read(exact_path), _read(bootstrap_path)
            passed = exact.get("status") == "PASS" and exact.get("expected_field_comparisons") == 432 and exact.get("completed_field_comparisons") == 432 and exact.get("mismatch_count") == 0 and exact.get("recovery_provenance", {}).get("status") == "PASS" and bootstrap.get("bootstrap_event") == "S5_FINAL_RESTART_RECONSTRUCTED"
            if not passed:
                event = _single_defect(root, state, "RECOVERY_AUDIT_FAILED_OR_MISSING")
            else:
                state["status"] = "PASS"; event = _event(state, "s5_final_single_allocation_pass")
    else:
        event = _single_defect(root, state, "UNKNOWN_MONITOR_STATE", observed=state["status"])
    _atomic(state_path, state); _append(events_path, event)
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--resume", action="store_true"); parser.add_argument("--single-allocation", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=15); parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if not args.resume or not 5 <= args.poll_seconds <= 3600:
        parser.error("--resume is required and --poll-seconds must be 5..3600")
    while True:
        state = advance_single_allocation(args.manifest) if args.single_allocation else advance(args.manifest)
        print(json.dumps({"status": state["status"], "updated_epoch": state["updated_epoch"]}, sort_keys=True))
        if args.once or state["status"] in TERMINAL:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
