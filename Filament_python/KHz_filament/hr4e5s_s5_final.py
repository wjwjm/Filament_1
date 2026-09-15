"""S5-FINAL multi-worker loss orchestration evidence helpers.

This module deliberately contains no propagation, deposition, or hydro
operator.  It adds a scenario-specific recovery contract around the existing
durable ``StreamingLifecycle.reconstruct_queue`` transaction.  The established
F01--F06 fault contracts remain in :mod:`hr4e5s_s5` unchanged.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from .hr4e5s_s3 import consume_streaming, finalize_streaming, run_optical_path
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
                          step_id: str, job_id: str, node: str, phase: str) -> dict[str, Any]:
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
    _atomic_json(Path(out_path), result)
    return result


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
                         lifecycle_root: str | Path, bootstrap_receipt_path: str | Path) -> dict[str, Any]:
    return run_optical_path(input_manifest_path=input_manifest_path, out_dir=out_dir, streaming_root=lifecycle_root,
                            resume=True, bootstrap_receipt_path=bootstrap_receipt_path,
                            bootstrap_receipt_validator=validate_bootstrap_receipt)


__all__ = ["S5_FINAL_CASE_ID", "S5_FINAL_SCHEMA", "bootstrap_recovery", "compare_exact", "consume_streaming", "finalize_streaming", "freeze_expected_recovery_effects", "run_recovery_optical", "snapshot_interrupted_state", "validate_bootstrap_receipt", "validate_recovery_provenance", "write_worker_identity"]
