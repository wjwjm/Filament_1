from __future__ import annotations

import json
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest

from KHz_filament.hr4e5s_s5_final import (
    S5_FINAL_SCHEMA,
    bootstrap_recovery,
    freeze_expected_recovery_effects,
    snapshot_interrupted_state,
    validate_bootstrap_receipt,
    validate_recovery_provenance,
)
from KHz_filament.hr4e5s_streaming import StreamingLifecycle


def _monitor_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "monitor_hr4e5s_s5_final.py"
    spec = importlib.util.spec_from_file_location("s5_final_monitor", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lifecycle(tmp_path: Path, name: str) -> StreamingLifecycle:
    count = 16
    current = {
        "delta_n": np.full((count, 8, 8), -1.0e-6, dtype=np.float64),
        "vx": np.zeros((count, 8, 8), dtype=np.float64),
        "vy": np.zeros((count, 8, 8), dtype=np.float64),
    }
    records = [{"ordinal": i, "screen_id": f"z{i:05d}", "z_m": i * 1.0e-4} for i in range(count)]
    return StreamingLifecycle.create(root=tmp_path / name, current=current, screen_records=records,
                                     current_generation="s5-final-current", dx_m=1.0e-4, dy_m=1.0e-4,
                                     queue_depth=16, actor="test")


def _prepare_posts(lifecycle: StreamingLifecycle) -> None:
    for ordinal in range(16):
        lifecycle.deposition_finalized(ordinal, actor="optical")
        lifecycle.commit_post_from_delta_n(ordinal, lifecycle.current_fields(ordinal)["delta_n"] - 1.0e-8, actor="optical")
        lifecycle.enqueue_post(ordinal, actor="optical")


def _finish(lifecycle: StreamingLifecycle) -> None:
    while True:
        block = lifecycle.claim_block(actor="hydro_consumer_0")
        if not block:
            break
        for ordinal in block:
            fields = lifecycle._artifact_fields(lifecycle.manifest["records"][ordinal]["post"], namespace="POST")
            lifecycle.commit_next(ordinal, fields, actor="hydro_consumer_0")
    assert lifecycle.validate_barrier(actor="barrier")["status"] == "PASS"
    assert lifecycle.promote_next_to_current(actor="barrier")["authoritative_namespace"] == "NEXT"


def test_s5_final_freezes_two_claims_before_single_bootstrap_and_checks_retry_history(tmp_path):
    reference = _lifecycle(tmp_path, "reference")
    _prepare_posts(reference)
    _finish(reference)
    candidate = _lifecycle(tmp_path, "candidate")
    _prepare_posts(candidate)
    assert candidate.claim_block(actor="hydro_consumer_0") == list(range(8))
    assert candidate.claim_block(actor="hydro_consumer_1") == list(range(8, 16))
    candidate.begin_hydro_screen(0, actor="hydro_consumer_0")
    candidate.begin_hydro_screen(8, actor="hydro_consumer_1")
    inventory = tmp_path / "inventory.json"
    effects = tmp_path / "effects.json"
    receipt = tmp_path / "bootstrap.json"
    snapshot = snapshot_interrupted_state(lifecycle_root=candidate.root, out_path=inventory)
    assert snapshot["active_hydro_actors"] == ["hydro_consumer_0", "hydro_consumer_1"]
    frozen = freeze_expected_recovery_effects(lifecycle_root=candidate.root, inventory_path=inventory, out_path=effects)
    assert frozen["status"] == "FROZEN"
    assert frozen["retry_deltas"]["0"] == frozen["retry_deltas"]["8"] == 1
    bootstrap_recovery(lifecycle_root=candidate.root, effects_path=effects, out_path=receipt, runtime_sha="a" * 40)
    validate_bootstrap_receipt(receipt_path=receipt, lifecycle_root=candidate.root, runtime_sha="a" * 40)
    with pytest.raises(ValueError, match="already recorded"):
        bootstrap_recovery(lifecycle_root=candidate.root, effects_path=effects, out_path=tmp_path / "again.json", runtime_sha="a" * 40)
    _finish(StreamingLifecycle.open(candidate.root))
    result = validate_recovery_provenance(reference_lifecycle_root=reference.root, candidate_lifecycle_root=candidate.root, effects_path=effects)
    assert result["status"] == "PASS"
    assert result["retry_deltas"]["0"] == result["retry_deltas"]["8"] == 1


def test_s5_final_rejects_effect_freeze_without_two_live_actors(tmp_path):
    lifecycle = _lifecycle(tmp_path, "single")
    _prepare_posts(lifecycle)
    assert lifecycle.claim_block(actor="hydro_consumer_0") == list(range(8))
    lifecycle.begin_hydro_screen(0, actor="hydro_consumer_0")
    inventory = tmp_path / "inventory.json"
    snapshot_interrupted_state(lifecycle_root=lifecycle.root, out_path=inventory)
    with pytest.raises(ValueError, match="two distinct active"):
        freeze_expected_recovery_effects(lifecycle_root=lifecycle.root, inventory_path=inventory, out_path=tmp_path / "effects.json")


def test_s5_final_batch_has_separate_worker_identity_and_faults_off_contract():
    root = Path(__file__).resolve().parents[1]
    batch = (root / "tools" / "hr4e5s_s5_final.sbatch").read_text(encoding="utf-8")
    assert 'unset HR4_S5_FAULT_ID HR4_S5_FAULT_SCREEN HR4_S5_FAULT_ONCE' in batch
    assert 'readonly FINAL_RUNNER="$REPO_DIR/Filament_python/tools/run_hr4e5s_s5_final.py"' in batch
    assert 'readonly S3_RUNNER="$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py"' in batch
    assert 'export FINAL_RUNNER S3_RUNNER' in batch
    assert batch.index('readonly FINAL_RUNNER') < batch.index('export FINAL_RUNNER S3_RUNNER') < batch.index('launch_worker()')
    assert 'write-identity' in batch
    assert 'hydro_consumer_${index}' in batch
    assert 'S5_FINAL_RESTART_RECONSTRUCTED' not in batch  # receipt validation stays in the dedicated Python adapter
    assert 'test ! -e "$CASE_ROOT/recovery"' in batch


def test_s5_final_monitor_is_step_scoped_and_never_uses_process_name_kills():
    root = Path(__file__).resolve().parents[1]
    monitor = (root / "tools" / "monitor_hr4e5s_s5_final.py").read_text(encoding="utf-8")
    assert 'scontrol", "listpids"' in monitor
    assert 'scancel", "--signal=KILL"' in monitor
    assert "pkill" not in monitor
    assert "WAIT_FOR_OLD_JOB_QUIESCENCE" in monitor
    assert "RECOVERY_SUBMISSION_UNCERTAIN" in monitor


def test_s5_final_async_preflight_keeps_start_receipt_outside_the_run_root():
    root = Path(__file__).resolve().parents[1]
    script = (root / "tools" / "hpc_ops" / "start_hr4e5s_s5_final_preflight_async.sh").read_text(encoding="utf-8")
    assert 'STATUS_DIR="${RUN_ROOT}.s5_final_preflight_async"' in script
    assert 'test ! -e "$RUN_ROOT" && test ! -e "$STATUS_DIR"' in script
    assert 'nohup bash "$PREFLIGHT"' in script


def test_s5_final_monitor_rejects_wrong_job_identity_and_duplicate_recovery_intent(tmp_path):
    monitor = _monitor_module()
    identities = tmp_path / "scenario" / "identities"
    identities.mkdir(parents=True)
    for actor, step, pid in (("optical_producer", "0", 101), ("hydro_consumer_0", "1", 102), ("hydro_consumer_1", "2", 103)):
        (identities / f"{actor}.json").write_text(json.dumps({"status": "READY", "actor": actor, "job_id": "wrong", "step_id": step, "worker_pid": pid, "gpu_visible_devices": "0"}), encoding="utf-8")
    values, reason = monitor._identities(tmp_path / "scenario", "initial", "999")
    assert values == [] and reason == "identity_receipts_invalid"
    root = tmp_path / "run"
    root.mkdir()
    manifest = root / "s5_final_monitor_manifest.json"
    manifest.write_text(json.dumps({"repo": str(tmp_path), "run_root": str(root), "expected_sha": "a" * 40, "preflight": str(root / "preflight.json"), "submit_script": str(root / "submit.sh"), "reference_case_root": str(root / "reference"), "initial_job_id": "700", "target_actor": "hydro_consumer_1"}), encoding="utf-8")
    state_path, _ = monitor._state_paths(manifest)
    state_path.write_text(json.dumps({"schema": monitor.SCHEMA, "status": "RECOVERY_SUBMISSION_PENDING", "run_root": str(root), "initial_job_id": "700", "recovery_job_id": None, "signal_sent": True, "updated_epoch": 0}), encoding="utf-8")
    (root / "recovery_submission_intent.json").write_text("{}", encoding="utf-8")
    result = monitor.advance(manifest)
    assert result["status"] == "READY_FOR_S5_FINAL_DEFECT_REVIEW"
    assert result["defect"]["reason"] == "RECOVERY_SUBMISSION_UNCERTAIN"


def test_s5_final_monitor_fails_closed_when_initial_job_terminates_before_identity_receipts(tmp_path, monkeypatch):
    monitor = _monitor_module()
    root = tmp_path / "run"
    root.mkdir()
    manifest = root / "s5_final_monitor_manifest.json"
    manifest.write_text(json.dumps({"repo": str(tmp_path), "run_root": str(root), "expected_sha": "a" * 40, "preflight": str(root / "preflight.json"), "submit_script": str(root / "submit.sh"), "reference_case_root": str(root / "reference"), "initial_job_id": "700", "target_actor": "hydro_consumer_1"}), encoding="utf-8")
    scheduler = {"queue_state": "", "sacct_state": "FAILED", "exit_code": "1:0", "terminal": True, "queue_returncode": 1, "sacct_returncode": 0}
    monkeypatch.setattr(monitor, "_scheduler", lambda job_id, cwd: scheduler)
    monkeypatch.setattr(monitor, "_step_has_pid", lambda *args: pytest.fail("terminal initial job must not be signalled"))
    result = monitor.advance(manifest)
    assert result["status"] == "READY_FOR_S5_FINAL_DEFECT_REVIEW"
    assert result["defect"]["reason"] == "INITIAL_JOB_TERMINAL_BEFORE_SAFE_SIGNAL"
    assert result["defect"]["identity_problem"] == "identity_receipts_incomplete"
    assert not (root / "recovery_submission_intent.json").exists()


def test_s5_final_provenance_rejects_retry_for_already_committed_next(tmp_path):
    reference = _lifecycle(tmp_path, "reference-next")
    _prepare_posts(reference)
    _finish(reference)
    effects = tmp_path / "effects-next.json"
    effects.write_text(json.dumps({"schema": S5_FINAL_SCHEMA, "status": "FROZEN", "retry_deltas": {str(i): (1 if i == 0 else 0) for i in range(16)}, "expected_recovery_attempts": []}), encoding="utf-8")
    result = validate_recovery_provenance(reference_lifecycle_root=reference.root, candidate_lifecycle_root=reference.root, effects_path=effects)
    assert result["status"] == "FAIL"
    assert any(item["name"] == "retry_count_delta_exact" and not item["pass"] for item in result["checks"])


def test_s5_final_single_allocation_controller_arms_both_workers_before_one_step_signal(tmp_path, monkeypatch):
    monitor = _monitor_module()
    root, case = tmp_path / "run", tmp_path / "run" / "scenario"
    (case / "initial" / "identities").mkdir(parents=True)
    (case / "initial" / "arming").mkdir(parents=True)
    manifest = root / "single.json"
    manifest.write_text(json.dumps({"execution_mode": monitor.SINGLE_ALLOCATION_MODE, "repo": str(tmp_path), "run_root": str(root), "expected_sha": "a" * 40, "reference_case_root": str(root / "reference"), "allocation_id": "700", "target_actor": "hydro_consumer_1"}), encoding="utf-8")
    identities = {}
    for actor, step, pid in (("optical_producer", "0", 101), ("hydro_consumer_0", "1", 102), ("hydro_consumer_1", "2", 103)):
        value = {"status": "READY", "actor": actor, "job_id": "700", "step_id": step, "worker_pid": pid, "node": "node-a", "execution_epoch": "initial"}
        identities[actor] = value
        (case / "initial" / "identities" / f"{actor}.json").write_text(json.dumps(value), encoding="utf-8")
    calls = []
    monkeypatch.setattr(monitor, "_single_listpids", lambda identity, cwd: (True, {"returncode": 0, "target_pid": identity["worker_pid"]}))
    monkeypatch.setattr(monitor, "_run", lambda args, cwd: (calls.append(list(args)) or subprocess.CompletedProcess(args, 0, "", "")))
    waiting = monitor.advance_single_allocation(manifest)
    assert waiting["status"] == "WAIT_FOR_DUAL_ARMING"
    assert not calls
    for actor in ("hydro_consumer_0", "hydro_consumer_1"):
        identity = identities[actor]
        (case / "initial" / "arming" / f"{actor}.json").write_text(json.dumps({"status": "ARMED", "actor": actor, "execution_epoch": "initial", "worker_pid": identity["worker_pid"], "node": "node-a", "block": [0]}), encoding="utf-8")
    advanced = monitor.advance_single_allocation(manifest)
    assert advanced["status"] == "WAIT_FOR_INITIAL_QUIESCENCE"
    assert advanced["signal_sent"] is True
    assert calls == [["scancel", "--signal=KILL", "700.2"]]
    assert (case / "initial" / "worker_loss_intent.json").is_file()
    assert (case / "initial" / "worker_loss_receipt.json").is_file()


def test_s5_final_single_allocation_batch_is_explicit_and_preserves_the_legacy_batch():
    root = Path(__file__).resolve().parents[1]
    single = (root / "tools" / "hr4e5s_s5_final_single_allocation.sbatch").read_text(encoding="utf-8")
    submit = (root / "tools" / "hpc_ops" / "submit_hr4e5s_s5_final.sh").read_text(encoding="utf-8")
    assert 'test "$CASE_MODE" = single_allocation' in single
    assert '"$PYTHON" "$MONITOR" --manifest "$MONITOR_MANIFEST" --single-allocation --resume --poll-seconds 5' in single
    assert '--arming-dir "$CASE_ROOT/initial/arming" --execution-epoch initial' in single
    assert 'recovery_is_new_allocation' in (root / "tools" / "monitor_hr4e5s_s5_final.py").read_text(encoding="utf-8")
    assert 'single_allocation) BATCH="$SINGLE_ALLOCATION_BATCH"' in submit
    assert (root / "tools" / "hr4e5s_s5_final.sbatch").is_file()


def test_s5_final_site_observability_probe_is_cpu_only_and_step_scoped():
    batch = (Path(__file__).resolve().parents[1] / "tools" / "hr4e5s_s5_final_observability_probe.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpu" in batch
    assert "--gpus-per-task" not in batch
    assert "--gres=gpu" not in batch
    assert 'scancel --signal=TERM "$target_step"' in batch
    assert 'scontrol listpids "$step"' in batch
    assert "SIGNAL_SCOPE_TOO_BROAD" in batch
    assert "TARGET_PID_NOT_PROVEN" in batch


def _phase0_identity(actor: str, step: str, pid: int, hostname: str = "node-a") -> dict:
    return {"actor": actor, "job_id": "700", "step_id": step, "pid": pid, "hostname": hostname}


def _phase0_listpids(pid: int, *, live: bool = True) -> dict:
    return {"returncode": 0, "stdout": f"{pid} probe" if live else ""}


def test_s5_final_phase0_rejects_unproven_or_ended_target_before_signal():
    monitor = _monitor_module()
    target, survivor = _phase0_identity("probe_target", "2", 103), _phase0_identity("probe_survivor", "1", 102)
    result = monitor.phase0_pre_signal_gate(
        target=target, survivor=survivor, controller_hostname="node-a",
        target_listpids=_phase0_listpids(103, live=False), survivor_listpids=_phase0_listpids(102),
        target_heartbeat_live=True, survivor_heartbeat_live=True, signal_intent_exists=False,
        signal_command=["scancel", "--signal=TERM", "700.2"],
    )
    assert result["status"] == "FAIL"
    assert not next(item for item in result["checks"] if item["name"] == "target_pid_currently_proven")["pass"]


def test_s5_final_phase0_rejects_duplicate_intent_identity_collision_wrong_host_and_broad_scope():
    monitor = _monitor_module()
    target, survivor = _phase0_identity("probe_target", "2", 103), _phase0_identity("probe_survivor", "2", 103, "node-b")
    result = monitor.phase0_pre_signal_gate(
        target=target, survivor=survivor, controller_hostname="node-a",
        target_listpids=_phase0_listpids(103), survivor_listpids=_phase0_listpids(103),
        target_heartbeat_live=True, survivor_heartbeat_live=True, signal_intent_exists=True,
        signal_command=["scancel", "--signal=TERM", "700"],
    )
    assert result["status"] == "FAIL"
    failed = {item["name"] for item in result["checks"] if not item["pass"]}
    assert {"steps_are_distinct", "pids_are_distinct", "controller_on_survivor_compute_node", "signal_intent_unused", "signal_scope_is_exact_target_step"} <= failed


def test_s5_final_phase0_post_signal_requires_target_quiescence_and_survivor_liveness():
    monitor = _monitor_module()
    failed = monitor.phase0_post_signal_gate(
        target_pid=103, target_wait_nonzero=False, target_still_live=True,
        survivor_still_live=True, target_listpids_after=_phase0_listpids(103),
    )
    assert failed["status"] == "FAIL"
    passed = monitor.phase0_post_signal_gate(
        target_pid=103, target_wait_nonzero=True, target_still_live=False,
        survivor_still_live=True, target_listpids_after=_phase0_listpids(103, live=False),
    )
    assert passed["status"] == "PASS"


def test_s5_final_single_allocation_phase0_precedes_cuda_and_science_start():
    batch = (Path(__file__).resolve().parents[1] / "tools" / "hr4e5s_s5_final_single_allocation.sbatch").read_text(encoding="utf-8")
    phase0_pass = 'phase0_observability_result.json" "status=PASS"'
    cuda_enable = "export UPPE_USE_GPU=1"
    initialize = '"$PYTHON" "$S3_RUNNER" initialize-stream'
    assert "--gpus-per-task=0" in batch
    assert "env -u CUDA_VISIBLE_DEVICES -u UPPE_USE_GPU" in batch
    assert 'capture_listpids listpids_before "$SLURM_JOB_ID"' in batch
    assert 'scancel --signal=TERM "$PHASE0_TARGET_STEP"' in batch
    assert phase0_pass in batch
    assert batch.index(phase0_pass) < batch.index(cuda_enable) < batch.index(initialize)
    assert "READY_FOR_S5_FINAL_DEFECT_REVIEW" in batch
