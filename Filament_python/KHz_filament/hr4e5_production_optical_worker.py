"""E5-1B WP2 real Candidate optical producer worker entry."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .confio import load_all
from .hr4e5_formal_entry import build_prefix_schedule, run_streaming_optical_pulse
from .hr4e5_evidence import atomic_json, sha256_file
from .hr4e5_storage import PROCESS_RECEIPT_SCHEMA, StorageBudget, process_identity
from .hr4e5_recovery_identity import capture_implementation_identity
from .hr4e5_recovery_output import qualify_output_attempt, activate_output_attempt


def _schedule(config_path: Path, prefix_intervals: int):
    from .hr4e5_production import _schedule_from_config
    components = load_all(str(config_path))
    return build_prefix_schedule(_schedule_from_config(components[2]), int(prefix_intervals))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lifecycle-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pulse", required=True, type=int)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--prefix-intervals", required=True, type=int)
    parser.add_argument("--admission-identity-sha256", required=True)
    parser.add_argument("--storage-root", required=True, type=Path)
    parser.add_argument("--intent-id", required=True)
    parser.add_argument("--reservation-id", required=True)
    parser.add_argument("--writer-id", required=True)
    parser.add_argument("--grant-id", required=True)
    parser.add_argument("--grant-nonce", required=True)
    parser.add_argument("--scientific-pair-id", required=True)
    parser.add_argument("--implementation-identity-sha256", required=True)
    parser.add_argument("--resume-output", action="store_true")
    args = parser.parse_args()

    root = Path(args.storage_root).resolve()
    admission = json.loads((root / "E5_1A_ADMISSION_IDENTITY.json").read_text(encoding="utf-8"))
    budget = StorageBudget.open_for_worker(
        root, campaign_id=str(admission["campaign_id"]), admission_identity=admission,
    )
    grant = budget.writer_grants().get(args.grant_id)
    if not isinstance(grant, dict):
        raise RuntimeError("optical child grant is missing")
    scope = grant["scope"]
    implementation = capture_implementation_identity()
    if (scope.get("role") != "OPTICAL" or scope.get("trajectory") != "C"
            or scope.get("pulse") != args.pulse
            or scope.get("scientific_pair_id") != args.scientific_pair_id
            or scope.get("admission_identity_sha256") != args.admission_identity_sha256
            or scope.get("reservation_id") != args.reservation_id
            or scope.get("intent_id") != args.intent_id
            or scope.get("implementation_identity") != implementation
            or implementation.get("identity_sha256") != args.implementation_identity_sha256):
        raise RuntimeError("optical child grant scope or implementation differs")
    budget.record_child_writer_launch_candidate(
        args.grant_id, nonce=args.grant_nonce, process=process_identity())
    identity = budget.bind_child_writer_grant(
        args.grant_id, nonce=args.grant_nonce, scope=scope)
    budget.authorize_child_writer_operation(
        args.grant_id, nonce=args.grant_nonce,
        coordinator_epoch=identity["coordinator_epoch"],
        operation="OPTICAL_OUTPUT", path=Path(args.output_dir),
    )
    budget.authorize_child_writer_operation(
        args.grant_id, nonce=args.grant_nonce,
        coordinator_epoch=identity["coordinator_epoch"],
        operation="POST_WRITE", path=Path(args.lifecycle_root),
    )
    if str(admission.get("identity_sha256", "")) != str(args.admission_identity_sha256):
        raise RuntimeError("optical worker admission identity differs")
    config_path = Path(args.config_path).resolve()
    schedule = _schedule(config_path, args.prefix_intervals)
    started = time.monotonic()
    error: BaseException | None = None
    try:
        result = run_streaming_optical_pulse(
            lifecycle_root=args.lifecycle_root, schedule=schedule, output_dir=args.output_dir,
            config_path=config_path, final=False, storage_budget=budget,
            admission_identity=admission, creation_intent=budget.intent(args.intent_id),
            trajectory="C", pulse=int(args.pulse), role="OPTICAL", defer_intent_completion=True,
            resume=args.resume_output,
        )
        qualified = qualify_output_attempt(
            campaign_root=root, state_root=args.lifecycle_root,
            output_dir=result["output_dir"],
            attempt_id=f"C_p{args.pulse}_{args.grant_id}",
            campaign_id=str(admission["campaign_id"]),
            scientific_pair_id=args.scientific_pair_id, trajectory="C",
            pulse=args.pulse, generation=str(scope["generation"]),
            coordinator_epoch=identity["coordinator_epoch"],
            writer_grant_id=args.grant_id,
            admission_identity_sha256=args.admission_identity_sha256,
            implementation_identity_sha256=args.implementation_identity_sha256,
        )
        activate_output_attempt(
            campaign_root=root, state_root=args.lifecycle_root,
            record_path=qualified["record_path"],
            expected_record_sha256=qualified["record_sha256"],
        )
        ended = time.monotonic()
        atomic_json(
            Path(args.lifecycle_root) / "PRODUCER_COMPLETE.json",
            {
                "schema": "khz_filament.hr4e5.e5_1b.producer_complete.v1",
                "status": "PASS", "pulse": int(args.pulse),
                "admission_identity_sha256": str(args.admission_identity_sha256),
                "scientific_pair_id": args.scientific_pair_id,
                "producer_grant_id": args.grant_id,
                "producer_coordinator_epoch": str(identity["coordinator_epoch"]),
                "completed_monotonic_s": float(ended),
            },
            overwrite=True,
        )
        process = process_identity()
        receipt = {
            "schema": PROCESS_RECEIPT_SCHEMA,
            "campaign_id": str(admission.get("campaign_id", "")), "pulse": int(args.pulse),
            "trajectory": "C", "role": "C_OPTICAL", "worker_index": None,
            "pid": process["pid"], "process_start_token": process["process_start_token"],
            "coordinator_epoch": str(identity.get("coordinator_epoch", "")),
            "admission_identity_sha256": str(args.admission_identity_sha256),
            "lifecycle_root": str(Path(args.lifecycle_root).resolve()),
            "started_monotonic_s": float(started), "ended_monotonic_s": float(ended),
            "exit_status": "PASS", "created_files": [], "writer_id": args.grant_id,
            "intent_id": str(args.intent_id), "reservation_id": str(args.reservation_id),
            "backend": "numpy/cpu", "physical_gpu_identity": "NOT_APPLICABLE_LOCAL_CPU",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        path = Path(args.lifecycle_root) / "process_receipts" / "C_OPTICAL.json"
        if path.exists():
            path = path.with_name(f"C_OPTICAL_{args.grant_id}.json")
        atomic_json(path, receipt, overwrite=False)
    except BaseException as exc:
        error = exc
        raise
    finally:
        ended = time.monotonic()
        try:
            budget.close_child_writer_grant(
                args.grant_id, nonce=args.grant_nonce,
                coordinator_epoch=identity["coordinator_epoch"],
                status="INTERRUPTED" if error is not None else "CLOSED",
                reason=None if error is None else f"{type(error).__name__}: {error}",
            )
        except Exception:
            if error is None:
                raise
    print(json.dumps({**receipt, "optical_result": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
