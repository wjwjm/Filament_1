"""S5-FINAL multi-worker loss orchestration evidence helpers.

This module deliberately contains no propagation, deposition, or hydro
operator.  It adds a scenario-specific recovery contract around the existing
durable ``StreamingLifecycle.reconstruct_queue`` transaction.  The established
F01--F06 fault contracts remain in :mod:`hr4e5s_s5` unchanged.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .hr4e5s_s3 import finalize_streaming, run_optical_path
from .hr4e5s_s5 import (
    _artifact_inventory,
    _authoritative_lifecycle,
    _atomic_json,
    _check,
    _field,
    _final_optical,
    inspect_lifecycle,
)
from .hr4e5s_streaming import FIELDS, RECOVERY_ATTEMPT_SCHEMA, StreamingLifecycle
from .hr4e_timestep import sha256_file


S5_FINAL_SCHEMA = "khz_filament.hr4e5s.s5_final.v1"
S5_FINAL_CASE_ID = "S5_FINAL_WORKER_LOSS"


def _read(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _require_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value.lower()):
        raise ValueError(f"{name} must be a 40-character commit SHA")
    return value


def _hydro_claims(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for record in manifest["records"]:
        if record.get("state") != "HYDRO_RUNNING" or record.get("post") is None or record.get("next") is not None:
            continue
        transitions = list(record.get("transitions", []))
        transition = transitions[-1] if transitions else {}
        actor = str(transition.get("actor", ""))
        if not actor.startswith("hydro_consumer_"):
            continue
        claims.append({
            "ordinal": int(record["ordinal"]), "screen_id": str(record["screen_id"]),
            "actor": actor, "retry_count": int(record["retry_count"]),
            "hydro_attempt_started": bool(record.get("hydro_attempt_started")),
            "post_file_sha256": str(record["post"]["file_sha256"]),
            "post_field_sha256": dict(record["post"]["field_sha256"]),
        })
    return sorted(claims, key=lambda item: (item["actor"], item["ordinal"]))


def write_worker_identity(*, out_path: str | Path, actor: str, worker_pid: int,
                          step_id: str, job_id: str, node: str, phase: str,
                          execution_epoch: str | None = None) -> dict[str, Any]:
    """Write a launcher-owned identity receipt before replacing its shell.

    ``worker_pid`` is intentionally supplied by the ``exec``-ing shell: after
    receipt persistence it becomes the Python worker PID, so a controller may
    compare the exact process with the Slurm step rather than guessing from a
    launcher shell or process name.
    """
    if not actor.startswith(("hydro_consumer_", "optical_producer")):
        raise ValueError("invalid S5-FINAL actor")
    if not isinstance(worker_pid, int) or worker_pid <= 1:
        raise ValueError("worker PID is invalid")
    if not str(job_id).isdigit() or not str(step_id).isdigit() or not str(node):
        raise ValueError("Slurm worker identity is incomplete")
    result = {
        "schema": S5_FINAL_SCHEMA, "kind": "worker_identity", "status": "READY",
        "actor": actor, "worker_pid": worker_pid, "job_id": str(job_id),
        "step_id": str(step_id), "node": str(node), "phase": str(phase),
        "gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "worker_id": os.environ.get("HR4E5S_WORKER_ID", actor),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if execution_epoch is not None:
        if execution_epoch not in {"initial", "recovery"}:
            raise ValueError("S5-FINAL execution epoch is invalid")
        result["execution_epoch"] = execution_epoch
    _atomic_json(Path(out_path), result)
    return result


def consume_final_streaming(*, lifecycle_root: str | Path, hydro: Mapping[str, Any],
                            producer_complete: str | Path, actor: str,
                            arming_dir: str | Path | None = None,
                            execution_epoch: str | None = None,
                            bootstrap_ready_path: str | Path | None = None,
                            arm_timeout_s: float = 120.0) -> dict[str, Any]:
    """S5-FINAL consumer with an optional, non-scientific arming rendezvous."""
    if execution_epoch == "recovery":
        if bootstrap_ready_path is None:
            raise ValueError("S5-FINAL recovery consumer requires bootstrap-ready receipt")
        ready = _read(bootstrap_ready_path)
        if (ready.get("schema") != S5_FINAL_SCHEMA or ready.get("kind") != "bootstrap_ready_receipt"
                or ready.get("status") != "PASS" or ready.get("actor") != "optical_producer"
                or ready.get("execution_epoch") != "recovery"):
            raise ValueError("S5-FINAL recovery bootstrap-ready receipt is invalid")
    if arming_dir is None:
        from .hr4e5s_s3 import consume_streaming
        return consume_streaming(lifecycle_root=lifecycle_root, hydro=hydro,
                                 producer_complete=producer_complete, actor=actor)
    if execution_epoch != "initial":
        raise ValueError("arming is only valid for the initial S5-FINAL epoch")
    root = Path(arming_dir)
    root.mkdir(parents=True, exist_ok=True)
    receipt_path, release_path = root / f"{actor}.json", root / f"{actor}.release.json"
    lifecycle, completed, marker, idle = StreamingLifecycle.open(lifecycle_root), [], Path(producer_complete), False

    def arm(block: list[int]) -> None:
        if receipt_path.exists():
            raise FileExistsError("S5-FINAL arming receipt already exists")
        _atomic_json(receipt_path, {
            "schema": S5_FINAL_SCHEMA, "kind": "arming_receipt", "status": "ARMED",
            "actor": actor, "execution_epoch": execution_epoch, "worker_pid": os.getpid(),
            "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()), "block": list(block),
            "armed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        deadline = time.monotonic() + float(arm_timeout_s)
        while not release_path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("S5-FINAL controller arming timeout")
            time.sleep(0.05)

    while True:
        block = lifecycle.run_one_hydro_block(
            dt_hydro=float(hydro["dt_hydro"]), n_hydro_steps=int(hydro["n_hydro_steps"]),
            chi=float(hydro["chi"]), nu=float(hydro["nu"]), n0=float(hydro["n0"]),
            gravity_x=float(hydro["gravity_x"]), gravity_y=float(hydro["gravity_y"]),
            cfl_limit=float(hydro["cfl_limit"]), actor=actor, before_hydro_block=arm,
        )
        if block:
            if idle:
                lifecycle.record_telemetry("CONSUMER_IDLE_END", actor=actor)
                idle = False
            completed.append(block)
            continue
        lifecycle = StreamingLifecycle.open(lifecycle_root)
        if marker.is_file() and not lifecycle.manifest["queue"]:
            if not [item for item in lifecycle.manifest["records"] if item["next"] is None]:
                lifecycle.record_telemetry("CONSUMER_COMPLETE", actor=actor)
                return {"completed_blocks": completed, "status": "PASS", "actor": actor}
        if not idle:
            lifecycle.record_telemetry("CONSUMER_IDLE_BEGIN", actor=actor)
            idle = True
        time.sleep(0.05)


def snapshot_interrupted_state(*, lifecycle_root: str | Path, out_path: str | Path) -> dict[str, Any]:
    """Persist an immutable pre-recovery inventory, without reconstruction."""
    base = inspect_lifecycle(lifecycle_root=lifecycle_root)
    lifecycle = StreamingLifecycle.open(lifecycle_root)
    base.update({
        "schema": S5_FINAL_SCHEMA, "kind": "interrupted_state_inventory",
        "manifest_sha256": sha256_file(lifecycle.manifest_path),
        "active_hydro_claims": _hydro_claims(lifecycle.manifest),
        "active_hydro_actors": sorted({item["actor"] for item in _hydro_claims(lifecycle.manifest)}),
    })
    _atomic_json(Path(out_path), base)
    return base


def freeze_expected_recovery_effects(*, lifecycle_root: str | Path, inventory_path: str | Path,
                                     out_path: str | Path) -> dict[str, Any]:
    """Freeze all permitted stale retries before recovery changes state."""
    inventory = _read(inventory_path)
    lifecycle = StreamingLifecycle.open(lifecycle_root)
    manifest_sha = sha256_file(lifecycle.manifest_path)
    if inventory.get("schema") != S5_FINAL_SCHEMA or inventory.get("kind") != "interrupted_state_inventory":
        raise ValueError("interrupted-state inventory schema is invalid")
    if inventory.get("manifest_sha256") != manifest_sha:
        raise ValueError("interrupted lifecycle changed before effects were frozen")
    claims = _hydro_claims(lifecycle.manifest)
    if len({claim["actor"] for claim in claims}) < 2:
        raise ValueError("two distinct active hydro workers were not proven")
    retry_deltas = {str(record["ordinal"]): 0 for record in lifecycle.manifest["records"]}
    attempts: list[dict[str, Any]] = []
    for claim in claims:
        if claim["hydro_attempt_started"]:
            retry_deltas[str(claim["ordinal"])] = 1
            attempts.append({
                "ordinal": claim["ordinal"], "screen_id": claim["screen_id"],
                "retry_count": claim["retry_count"] + 1,
                "post_file_sha256": claim["post_file_sha256"],
                "post_field_sha256": claim["post_field_sha256"],
                "actor_before_loss": claim["actor"],
            })
    result = {
        "schema": S5_FINAL_SCHEMA, "kind": "expected_recovery_effects", "status": "FROZEN",
        "lifecycle_root": str(Path(lifecycle_root).resolve()), "source_inventory": str(Path(inventory_path).resolve()),
        "interrupted_manifest_sha256": manifest_sha, "active_claims": claims,
        "retry_deltas": retry_deltas, "expected_recovery_attempts": attempts,
        "submitted_next_ordinals": [int(r["ordinal"]) for r in lifecycle.manifest["records"] if r["next"] is not None],
        "allowed_non_authoritative_temporary_artifacts": list(inventory.get("temporary_artifacts", [])),
        "allowed_unreferenced_final_artifacts": dict(inventory.get("unreferenced_final_artifacts", {})),
    }
    _atomic_json(Path(out_path), result)
    return result


def bootstrap_recovery(*, lifecycle_root: str | Path, effects_path: str | Path, out_path: str | Path,
                       runtime_sha: str, case_id: str = S5_FINAL_CASE_ID) -> dict[str, Any]:
    """Perform exactly one serial S5-FINAL reconstruction after frozen audit."""
    _require_sha(runtime_sha, "runtime SHA")
    effects, receipt_path = _read(effects_path), Path(out_path)
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    if effects.get("schema") != S5_FINAL_SCHEMA or effects.get("status") != "FROZEN":
        raise ValueError("expected recovery effects are not frozen")
    lifecycle = StreamingLifecycle.open(lifecycle_root)
    if any(item.get("event") == "S5_FINAL_RESTART_RECONSTRUCTED" for item in lifecycle.manifest.get("telemetry_events", [])):
        raise ValueError("S5-FINAL bootstrap was already recorded")
    if effects.get("interrupted_manifest_sha256") != sha256_file(lifecycle.manifest_path):
        raise ValueError("interrupted lifecycle changed after effects were frozen")
    queue = lifecycle.reconstruct_queue(actor="s5_final_restart")
    lifecycle = StreamingLifecycle.open(lifecycle_root)
    event = lifecycle.record_telemetry(
        "S5_FINAL_RESTART_RECONSTRUCTED", actor="s5_final_restart", bootstrap=True,
        expected_effects_sha256=sha256_file(effects_path), reconstructed_queue=list(queue),
    )
    lifecycle = StreamingLifecycle.open(lifecycle_root)
    events = list(lifecycle.manifest["telemetry_events"])
    index = len(events) - 1
    if index < 0 or events[index] != event:
        raise RuntimeError("S5-FINAL bootstrap event was not durably appended")
    result = {
        "schema": S5_FINAL_SCHEMA, "kind": "bootstrap_receipt", "status": "PASS",
        "case_id": case_id, "runtime_sha": runtime_sha, "lifecycle_root": str(Path(lifecycle_root).resolve()),
        "expected_effects_sha256": sha256_file(effects_path), "bootstrap_event": "S5_FINAL_RESTART_RECONSTRUCTED",
        "telemetry_event_index": index, "telemetry_event_count": len(events),
        "queue_ordinals": list(lifecycle.manifest["queue"]), "backlog_ordinals": list(lifecycle.manifest.get("recovery_backlog", [])),
        "queue_depth": int(lifecycle.manifest["queue_depth"]), "manifest_sha256": sha256_file(lifecycle.manifest_path),
    }
    _atomic_json(receipt_path, result)
    validate_bootstrap_receipt(receipt_path=receipt_path, lifecycle_root=lifecycle_root, runtime_sha=runtime_sha)
    return result


def validate_bootstrap_receipt(*, receipt_path: str | Path, lifecycle_root: str | Path,
                               runtime_sha: str | None, require_current_telemetry_count: bool = True) -> None:
    """Validate the dedicated receipt before any recovery worker starts."""
    receipt, lifecycle = _read(receipt_path), StreamingLifecycle.open(lifecycle_root)
    if receipt.get("schema") != S5_FINAL_SCHEMA or receipt.get("kind") != "bootstrap_receipt" or receipt.get("status") != "PASS":
        raise ValueError("S5-FINAL bootstrap receipt is invalid")
    if receipt.get("lifecycle_root") != str(Path(lifecycle_root).resolve()) or receipt.get("runtime_sha") != runtime_sha:
        raise ValueError("S5-FINAL bootstrap receipt identity mismatch")
    events, index = list(lifecycle.manifest.get("telemetry_events", [])), receipt.get("telemetry_event_index")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(events):
        raise ValueError("S5-FINAL bootstrap event index is invalid")
    event = events[index]
    if not isinstance(event, Mapping) or event.get("event") != "S5_FINAL_RESTART_RECONSTRUCTED" or event.get("actor") != "s5_final_restart" or event.get("bootstrap") is not True:
        raise ValueError("S5-FINAL bootstrap event is invalid")
    if sum(1 for item in events if isinstance(item, Mapping) and item.get("event") == "S5_FINAL_RESTART_RECONSTRUCTED") != 1:
        raise ValueError("S5-FINAL reconstruction is not unique")
    if require_current_telemetry_count and int(receipt.get("telemetry_event_count", -1)) != len(events):
        raise ValueError("S5-FINAL bootstrap receipt was superseded before worker launch")
    if list(receipt.get("queue_ordinals", [])) != list(lifecycle.manifest["queue"]) or list(receipt.get("backlog_ordinals", [])) != list(lifecycle.manifest.get("recovery_backlog", [])):
        raise ValueError("S5-FINAL bootstrap queue does not match lifecycle")
    if int(receipt.get("queue_depth", -1)) != int(lifecycle.manifest["queue_depth"]):
        raise ValueError("S5-FINAL bootstrap queue depth is invalid")


def _write_bootstrap_ready_receipt(*, ready_path: str | Path, receipt_path: str | Path,
                                   lifecycle_root: str | Path, runtime_sha: str) -> dict[str, Any]:
    """Atomically persist the optical-owned recovery launch boundary.

    This private helper deliberately does not revalidate the bootstrap.  Its
    sole caller in ``run_recovery_optical`` invokes it immediately *after* the
    internal validator returns; the public helper below validates first.
    """
    _require_sha(runtime_sha, "runtime SHA")
    destination, receipt, root = Path(ready_path), Path(receipt_path), Path(lifecycle_root)
    lifecycle = StreamingLifecycle.open(root)
    result = {
        "schema": S5_FINAL_SCHEMA, "kind": "bootstrap_ready_receipt", "status": "PASS",
        "actor": "optical_producer", "execution_epoch": "recovery",
        "lifecycle_root": str(root.resolve()), "runtime_sha": runtime_sha,
        "bootstrap_receipt": str(receipt.resolve()), "bootstrap_receipt_sha256": sha256_file(receipt),
        "lifecycle_manifest_sha256": sha256_file(lifecycle.manifest_path),
        "queue_ordinals": list(lifecycle.manifest["queue"]),
        "backlog_ordinals": list(lifecycle.manifest.get("recovery_backlog", [])),
        "ready_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_json(destination, result)
    return result


def write_bootstrap_ready_receipt(*, ready_path: str | Path, receipt_path: str | Path,
                                  lifecycle_root: str | Path, runtime_sha: str) -> dict[str, Any]:
    """Validate a bootstrap boundary, then create its one-time ready receipt."""
    if Path(ready_path).exists():
        raise FileExistsError(ready_path)
    validate_bootstrap_receipt(receipt_path=receipt_path, lifecycle_root=lifecycle_root, runtime_sha=runtime_sha,
                               require_current_telemetry_count=False)
    return _write_bootstrap_ready_receipt(ready_path=ready_path, receipt_path=receipt_path,
                                          lifecycle_root=lifecycle_root, runtime_sha=runtime_sha)


def validate_bootstrap_ready_receipt(*, ready_path: str | Path, receipt_path: str | Path,
                                     lifecycle_root: str | Path, runtime_sha: str) -> None:
    """Reject hydro launch unless the optical validator created this receipt."""
    ready, receipt, root = _read(ready_path), Path(receipt_path), Path(lifecycle_root)
    if (ready.get("schema") != S5_FINAL_SCHEMA or ready.get("kind") != "bootstrap_ready_receipt"
            or ready.get("status") != "PASS" or ready.get("actor") != "optical_producer"
            or ready.get("execution_epoch") != "recovery"):
        raise ValueError("S5-FINAL bootstrap-ready receipt is invalid")
    if (ready.get("lifecycle_root") != str(root.resolve()) or ready.get("runtime_sha") != runtime_sha
            or ready.get("bootstrap_receipt") != str(receipt.resolve())
            or ready.get("bootstrap_receipt_sha256") != sha256_file(receipt)):
        raise ValueError("S5-FINAL bootstrap-ready receipt identity mismatch")
    fingerprint = ready.get("lifecycle_manifest_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("S5-FINAL bootstrap-ready lifecycle fingerprint is invalid")
    # The ready receipt captures the immutable bootstrap manifest identity.  It
    # must match the receipt's frozen boundary, not the current live manifest:
    # hydro is allowed to mutate queue/backlog after ready has been persisted.
    if fingerprint != _read(receipt).get("manifest_sha256"):
        raise ValueError("S5-FINAL bootstrap-ready lifecycle fingerprint mismatch")
    validate_bootstrap_receipt(receipt_path=receipt, lifecycle_root=root, runtime_sha=runtime_sha,
                               require_current_telemetry_count=False)


def validate_reference_for_comparison(*, reference_root: str | Path,
                                      expected_lifecycle_root: str | Path | None,
                                      input_manifest_path: str | Path) -> dict[str, Any]:
    """Fail closed before S5-FINAL comparison accepts a clean-reference path.

    The comparison itself remains the frozen exact comparator.  This gate only
    proves that its reference inputs are complete, readable, and statically
    compatible with the candidate contract before a long allocation reaches
    final comparison startup.
    """
    reference_root, input_path = map(Path, (reference_root, input_manifest_path))
    expected_root = None if expected_lifecycle_root is None else Path(expected_lifecycle_root)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, **detail: Any) -> None:
        checks.append({"name": name, "pass": bool(passed), **detail})

    manifest_path = reference_root / "lifecycle" / "streaming_manifest.json"
    check("streaming_manifest_exists", manifest_path.is_file())
    try:
        _read(manifest_path)
        check("streaming_manifest_json_parseable", True)
    except Exception as error:
        check("streaming_manifest_json_parseable", False, error=f"{type(error).__name__}: {error}")
    try:
        reference = StreamingLifecycle.open(reference_root / "lifecycle")
        check("streaming_lifecycle_open", True)
    except Exception as error:
        check("streaming_lifecycle_open", False, error=f"{type(error).__name__}: {error}")
        return {"schema": S5_FINAL_SCHEMA, "status": "FAIL", "checks": checks}
    expected = None
    if expected_root is not None:
        try:
            expected = StreamingLifecycle.open(expected_root)
            check("expected_lifecycle_open", True)
        except Exception as error:
            check("expected_lifecycle_open", False, error=f"{type(error).__name__}: {error}")
            return {"schema": S5_FINAL_SCHEMA, "status": "FAIL", "checks": checks}
    try:
        input_manifest = _read(input_path)
        check("input_manifest_parseable", True)
    except Exception as error:
        check("input_manifest_parseable", False, error=f"{type(error).__name__}: {error}")
        return {"schema": S5_FINAL_SCHEMA, "status": "FAIL", "checks": checks}

    ref_manifest = reference.manifest
    if expected is not None:
        expected_manifest = expected.manifest
        for key in ("expected_screen_count", "queue_depth", "block_size", "dx_m", "dy_m", "shape", "dtype",
                    "current_generation", "next_generation"):
            check(f"identity_{key}", ref_manifest.get(key) == expected_manifest.get(key),
                  reference=ref_manifest.get(key), expected=expected_manifest.get(key))
    expected_count = (int(expected.manifest.get("expected_screen_count", -1)) if expected is not None else 48)
    check("frozen_screen_count", expected_count == 48 if expected is None else True, expected=expected_count)
    records = list(ref_manifest.get("records", []))
    input_records = list(input_manifest.get("screen_records", []))
    check("input_screen_count", len(input_records) == expected_count, observed=len(input_records), expected=expected_count)
    identity = [(record.get("ordinal"), record.get("screen_id"), record.get("z_m")) for record in records]
    expected_identity = [(record.get("ordinal"), record.get("screen_id"), record.get("z_m")) for record in input_records]
    check("screen_identity_matches_input", identity == expected_identity)
    hydro = dict(input_manifest.get("hydro", {}))
    check("input_queue_depth_matches", int(ref_manifest.get("queue_depth", -1)) == int(hydro.get("queue_depth", -2)))
    check("input_block_size_matches", int(ref_manifest.get("block_size", -1)) == int(hydro.get("block_size", -2)))
    check("input_dx_matches", float(ref_manifest.get("dx_m", 0.0)) == float(input_manifest.get("dx_m", -1.0)))
    check("input_dy_matches", float(ref_manifest.get("dy_m", 0.0)) == float(input_manifest.get("dy_m", -1.0)))
    readable = {"POST": 0, "NEXT": 0}
    for record in records:
        for namespace in ("POST", "NEXT"):
            try:
                entry = record.get(namespace.lower())
                if entry is None:
                    raise ValueError("entry missing")
                fields = reference._artifact_fields(entry, namespace=namespace)
                if set(fields) != {"delta_n", "vx", "vy"}:
                    raise ValueError("three authoritative fields are incomplete")
                readable[namespace] += 1
            except Exception as error:
                check(f"{namespace.lower()}_screen_{record.get('ordinal')}_readable", False,
                      error=f"{type(error).__name__}: {error}")
    check("post_arrays_complete", readable["POST"] == expected_count, readable=readable["POST"], expected=expected_count)
    check("next_arrays_complete", readable["NEXT"] == expected_count, readable=readable["NEXT"], expected=expected_count)
    optical = reference_root / "optical"
    required = ["optical_run.json", "final_optical_field.npy", "scientific_ledger.npz",
                "s3_optical.hr3a_qion_samples.npy", "s3_optical.hr3a_qib_samples.npy",
                "s3_optical.hr3a_qraman_samples.npy"]
    for name in required:
        check(f"optical_{name}_exists", (optical / name).is_file())
    try:
        reference_optical_run = _read(optical / "optical_run.json")
        expected_input_sha256 = sha256_file(input_path)
        observed_input_sha256 = reference_optical_run.get("input_manifest_sha256")
        check("reference_input_manifest_sha256_matches", observed_input_sha256 == expected_input_sha256,
              reference=observed_input_sha256, expected=expected_input_sha256)
    except Exception as error:
        check("reference_input_manifest_sha256_matches", False, error=f"{type(error).__name__}: {error}")
    for name in required[1:]:
        path = optical / name
        try:
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as values:
                    check(f"optical_{name}_readable", bool(values.files), fields=sorted(values.files))
            else:
                values = np.load(path, mmap_mode="r", allow_pickle=False)
                count_ok = name == "final_optical_field.npy" or values.shape[0] == expected_count
                check(f"optical_{name}_readable", bool(count_ok), shape=list(values.shape), dtype=str(values.dtype))
        except Exception as error:
            check(f"optical_{name}_readable", False, error=f"{type(error).__name__}: {error}")
    result = {"schema": S5_FINAL_SCHEMA, "reference_root": str(reference_root.resolve()),
              "expected_lifecycle_root": None if expected_root is None else str(expected_root.resolve()), "input_manifest": str(input_path.resolve()),
              "checks": checks}
    result["status"] = "PASS" if all(item["pass"] for item in checks) else "FAIL"
    return result


def validate_recovery_provenance(*, reference_lifecycle_root: str | Path, candidate_lifecycle_root: str | Path,
                                 effects_path: str | Path, out_path: str | Path | None = None) -> dict[str, Any]:
    """Strictly compare multi-worker retry history to its pre-frozen contract."""
    checks: list[dict[str, Any]] = []
    def check(name: str, passed: bool, **detail: Any) -> None:
        checks.append({"name": name, "pass": bool(passed), **detail})
    effects = _read(effects_path)
    reference, candidate = StreamingLifecycle.open(reference_lifecycle_root), StreamingLifecycle.open(candidate_lifecycle_root)
    ref_records, cand_records = list(reference.manifest["records"]), list(candidate.manifest["records"])
    identities = [(r["ordinal"], r["screen_id"], r["z_m"]) for r in ref_records] == [(r["ordinal"], r["screen_id"], r["z_m"]) for r in cand_records]
    check("reference_candidate_record_identity_exact", identities)
    deltas = {str(c["ordinal"]): int(c["retry_count"]) - int(r["retry_count"]) for r, c in zip(ref_records, cand_records)} if identities else {}
    expected = {str(key): int(value) for key, value in dict(effects.get("retry_deltas", {})).items()}
    check("retry_count_delta_exact", deltas == expected, observed=deltas, expected=expected)
    attempts = list(candidate.manifest.get("recovery_attempts", []))
    expected_attempts = list(effects.get("expected_recovery_attempts", []))
    compact = [{key: item.get(key) for key in ("ordinal", "screen_id", "retry_count", "post_file_sha256", "post_field_sha256")} for item in attempts if isinstance(item, Mapping)]
    compact_expected = [{key: item.get(key) for key in ("ordinal", "screen_id", "retry_count", "post_file_sha256", "post_field_sha256")} for item in expected_attempts]
    valid_attempts = len(compact) == len(attempts) and all(item.get("schema") == RECOVERY_ATTEMPT_SCHEMA and item.get("reason") == "STALE_HYDRO_RUNNING_RECONSTRUCTED" for item in attempts if isinstance(item, Mapping))
    check("recovery_attempts_exact", valid_attempts and compact == compact_expected, observed=compact, expected=compact_expected)
    check("terminal_queue_and_backlog_empty", not candidate.manifest["queue"] and not candidate.manifest.get("recovery_backlog", []))
    result = {"schema": S5_FINAL_SCHEMA, "kind": "recovery_provenance", "checks": checks, "retry_deltas": deltas, "recovery_attempts": attempts}
    result["recovery_provenance_exact"] = all(item["pass"] for item in checks)
    result["status"] = "PASS" if result["recovery_provenance_exact"] else "FAIL"
    if out_path is not None:
        _atomic_json(Path(out_path), result)
    return result


def compare_exact(*, reference_lifecycle_root: str | Path, reference_optical_dir: str | Path,
                  candidate_lifecycle_root: str | Path, candidate_optical_dir: str | Path,
                  effects_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Reuse S5's field/ownership core with an S5-FINAL provenance contract."""
    from .hr4e5s_s5 import compare_clean_reference
    destination = Path(out_dir)
    provenance = validate_recovery_provenance(reference_lifecycle_root=reference_lifecycle_root, candidate_lifecycle_root=candidate_lifecycle_root, effects_path=effects_path)
    result = compare_clean_reference(
        reference_lifecycle_root=reference_lifecycle_root, reference_optical_dir=reference_optical_dir,
        candidate_lifecycle_root=candidate_lifecycle_root, candidate_optical_dir=candidate_optical_dir,
        out_dir=destination, recovery_provenance_override=provenance,
    )
    _atomic_json(destination / "recovery_provenance.json", provenance)
    _atomic_json(destination / "exact_comparison.json", result)
    return result


def run_recovery_optical(*, input_manifest_path: str | Path, out_dir: str | Path,
                         lifecycle_root: str | Path, bootstrap_receipt_path: str | Path,
                         bootstrap_ready_path: str | Path) -> dict[str, Any]:
    runtime_sha = os.environ.get("EXPECTED_GIT_SHA")

    def validated_bootstrap_boundary(**kwargs: Any) -> None:
        validate_bootstrap_receipt(**kwargs)
        _write_bootstrap_ready_receipt(ready_path=bootstrap_ready_path,
                                       receipt_path=bootstrap_receipt_path,
                                       lifecycle_root=lifecycle_root,
                                       runtime_sha=_require_sha(runtime_sha, "runtime SHA"))

    return run_optical_path(input_manifest_path=input_manifest_path, out_dir=out_dir, streaming_root=lifecycle_root,
                            resume=True, bootstrap_receipt_path=bootstrap_receipt_path,
                            bootstrap_receipt_validator=validated_bootstrap_boundary)


__all__ = ["S5_FINAL_CASE_ID", "S5_FINAL_SCHEMA", "bootstrap_recovery", "compare_exact", "consume_final_streaming", "finalize_streaming", "freeze_expected_recovery_effects", "run_recovery_optical", "snapshot_interrupted_state", "validate_bootstrap_ready_receipt", "validate_bootstrap_receipt", "validate_reference_for_comparison", "validate_recovery_provenance", "write_bootstrap_ready_receipt", "write_worker_identity"]
