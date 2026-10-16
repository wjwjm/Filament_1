"""E5-1B WP2 real Candidate optical producer worker entry."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from .confio import load_all
from .hr4e5_formal_entry import (
    build_prefix_schedule, resolve_storage_lock_timeout_s, run_streaming_optical_pulse,
)
from .hr4e5_evidence import atomic_json, sha256_file
from .hr4e5_storage import PROCESS_RECEIPT_SCHEMA, StorageBudget, process_identity
from .hr4e5_recovery_identity import capture_implementation_identity
from .hr4e5_recovery_output import qualify_output_attempt, activate_output_attempt


def _gpu_process_receipt_fields(
    role: str,
    *,
    campaign_id: str,
    scientific_pair_id: str,
    coordinator_epoch: str,
    implementation_identity_sha256: str,
) -> dict[str, Any] | None:
    gpu_mode = os.environ.get("UPPE_USE_GPU") == "1"
    configured = any(os.environ.get(name) for name in (
        "HR4E5_GPU_ROLE", "HR4E5_GPU_AUDIT_ROOT", "HR4E5_GPU_STARTUP_RECEIPT",
    ))
    if not gpu_mode:
        if configured:
            raise RuntimeError("GPU worker receipt environment is configured while GPU mode is disabled")
        return None
    if os.environ.get("HR4E5_GPU_ROLE") != role:
        raise RuntimeError(f"GPU worker role does not match {role}")
    audit_root = os.environ.get("HR4E5_GPU_AUDIT_ROOT")
    if not audit_root:
        raise RuntimeError("GPU worker receipt audit root is missing")
    startup_value = os.environ.get("HR4E5_GPU_STARTUP_RECEIPT")
    startup_path = Path(startup_value) if startup_value else Path(audit_root) / f"{role}.startup.json"
    if not startup_path.is_absolute():
        startup_path = Path(audit_root) / startup_path
    startup_path = startup_path.resolve()
    try:
        startup = json.loads(startup_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"GPU startup receipt is unavailable for {role}") from exc
    identity = startup.get("identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError(f"GPU startup receipt identity is invalid for {role}")
    process = identity.get("process")
    gpu = identity.get("gpu")
    assignment = startup.get("assignment")
    if not isinstance(process, Mapping) or not isinstance(gpu, Mapping) or not isinstance(assignment, Mapping):
        raise RuntimeError(f"GPU startup receipt identity is incomplete for {role}")
    expected = {
        "campaign_id": campaign_id,
        "scientific_pair_id": scientific_pair_id,
        "coordinator_epoch": coordinator_epoch,
        "implementation_identity_sha256": implementation_identity_sha256,
    }
    if (startup.get("receipt_type") != "gpu_startup" or startup.get("status") != "PASS"
            or startup.get("role") != role or startup.get("backend") != "cupy/cuda"
            or any(startup.get(key) != value for key, value in expected.items())
            or process.get("pid") != process_identity()["pid"]
            or not isinstance(gpu.get("uuid"), str) or not gpu.get("uuid")
            or assignment.get("uuid") != gpu.get("uuid")
            or not isinstance(assignment.get("token"), str) or not assignment.get("token")
            or assignment.get("token") != os.environ.get("HR4E5_GPU_ASSIGNED_TOKEN")
            or startup.get("cuda_visible_devices") != os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise RuntimeError(f"GPU startup receipt identity is invalid for {role}")
    return {
        "backend": str(startup["backend"]),
        "physical_gpu_identity": str(gpu["uuid"]),
        "gpu_role": role,
        "gpu_assignment_token": str(assignment["token"]),
        "gpu_startup_receipt_path": str(startup_path),
        "gpu_startup_receipt_sha256": sha256_file(startup_path),
    }


def _mark_gpu_success(role: str) -> None:
    if os.environ.get("UPPE_USE_GPU") != "1":
        return
    if os.environ.get("HR4E5_GPU_ROLE") != role:
        raise RuntimeError(f"GPU worker role does not match {role}")
    from .hr4e5_gpu_bootstrap import update_final_metadata

    update_final_metadata(science_output_status="PASS", writer_grant_status="CLOSED")


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
    lock_timeout_s = resolve_storage_lock_timeout_s(
        admission, expected_hash=args.admission_identity_sha256,
    )
    config_path = Path(args.config_path).resolve()
    lut_identity = admission.get("lut_identity", {})
    lut_staging = lut_identity.get("staging") if isinstance(lut_identity, Mapping) else None
    if isinstance(lut_staging, Mapping):
        from .hr4e5_production import validate_staged_frozen_lut_cache_hit

        validate_staged_frozen_lut_cache_hit(
            config_path, root, lut_identity=lut_identity,
            staging_receipt_path=str(lut_staging.get("receipt_path", "")),
            admission_identity_sha256=str(args.admission_identity_sha256),
            require_campaign_cwd=True,
        )
    elif (admission.get("execution_mode") == "FORMAL_SITE_ADMISSION"
          and lut_identity.get("mode") == "RATE_TABLE"):
        raise RuntimeError("formal optical worker lacks admitted frozen LUT inputs")
    budget = StorageBudget.open_for_worker(
        root, campaign_id=str(admission["campaign_id"]), admission_identity=admission,
        lock_timeout_s=lock_timeout_s,
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
    schedule = _schedule(config_path, args.prefix_intervals)
    started = time.monotonic()
    error: BaseException | None = None
    receipt: dict[str, Any] | None = None
    path: Path | None = None
    grant_close: Mapping[str, Any] | None = None
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
        gpu_fields = _gpu_process_receipt_fields(
            "C_OPT",
            campaign_id=str(admission["campaign_id"]),
            scientific_pair_id=args.scientific_pair_id,
            coordinator_epoch=str(identity["coordinator_epoch"]),
            implementation_identity_sha256=args.implementation_identity_sha256,
        )
        if gpu_fields is not None:
            receipt.update(gpu_fields)
        path = Path(args.lifecycle_root) / "process_receipts" / "C_OPTICAL.json"
        if path.exists():
            path = path.with_name(f"C_OPTICAL_{args.grant_id}.json")
    except BaseException as exc:
        error = exc
        raise
    finally:
        ended = time.monotonic()
        try:
            grant_close = budget.close_child_writer_grant(
                args.grant_id, nonce=args.grant_nonce,
                coordinator_epoch=identity["coordinator_epoch"],
                status="INTERRUPTED" if error is not None else "CLOSED",
                reason=None if error is None else f"{type(error).__name__}: {error}",
            )
        except Exception:
            if error is None:
                raise
        if error is None:
            if not isinstance(grant_close, Mapping) or grant_close.get("state") != "CLOSED":
                raise RuntimeError("optical child writer grant did not close successfully")
            _mark_gpu_success("C_OPT")
    if receipt is None or path is None:
        raise RuntimeError("optical worker completed without a process receipt")
    atomic_json(path, receipt, overwrite=False)
    print(json.dumps({**receipt, "optical_result": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
