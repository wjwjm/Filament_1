"""E5-1B WP2 real Candidate hydro worker entry.

This module owns process orchestration only.  All scientific work remains the
existing ``StreamingLifecycle.run_one_hydro_block`` call and the frozen HR-4
single-screen solver behind it.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

from .confio import load_all
from .hr4d_pulse_lifecycle import build_interpulse_step_schedule
from .hr4e5_evidence import atomic_json, sha256_file
from .hr4e5_formal_entry import resolve_storage_lock_timeout_s
from .hr4e5s_streaming import StreamingLifecycle
from .hr4e5_storage import PROCESS_RECEIPT_SCHEMA, StorageBudget, process_identity
from .hr4e5_recovery_identity import capture_implementation_identity


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
    assignment = startup.get("assignment", {})
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


def _hydro_parameters(config_path: Path) -> dict[str, Any]:
    _grid, beam, _prop, _ion, heat, _run, _raman = load_all(str(config_path))
    schedule = build_interpulse_step_schedule(
        f_rep=float(heat.f_rep), dt_hydro=float(heat.dt_hydro)
    )
    if schedule.remainder_s != 0.0:
        raise ValueError("production hydro worker does not support a remainder step")
    return {
        "dt_hydro": float(heat.dt_hydro),
        "n_hydro_steps": int(schedule.full_step_count),
        "chi": float(heat.chi),
        "nu": float(heat.nu),
        "n0": float(beam.n0),
        "gravity_x": float(getattr(heat, "gravity_x", 0.0)),
        "gravity_y": float(getattr(heat, "gravity_y", -9.81)),
        "cfl_limit": 1.0,
    }


def _producer_complete(root: Path, *, scientific_pair_id: str,
                       producer_grant_id: str) -> bool:
    path = root / "PRODUCER_COMPLETE.json"
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("schema") == "khz_filament.hr4e5.e5_1b.producer_complete.v1"
        and value.get("status") == "PASS"
        and value.get("scientific_pair_id") == scientific_pair_id
        and value.get("producer_grant_id") == producer_grant_id
    )


def _receipt(root: Path, *, pulse: int, worker_index: int, lifecycle_root: Path,
             storage_root: Path, admission_hash: str, started: float, ended: float,
             claimed: list[int], created: list[Path], identity: Mapping[str, Any],
             exit_status: str, error: BaseException | None = None,
             gpu_fields: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if exit_status not in {"PASS", "FAIL"} or (exit_status == "FAIL" and error is None):
        raise ValueError("hydro process receipt requires a truthful terminal status")
    path = lifecycle_root / "process_receipts" / f"C_HYDRO_{int(worker_index):03d}.json"
    if path.exists():
        path = lifecycle_root / "process_receipts" / f"C_HYDRO_{int(worker_index):03d}_{identity['grant_id']}.json"
    payload = {
        "schema": PROCESS_RECEIPT_SCHEMA,
        "campaign_id": str(identity.get("campaign_id", "")),
        "pulse": int(pulse), "trajectory": "C", "role": "C_HYDRO",
        "worker_index": int(worker_index), "pid": process_identity()["pid"],
        "process_start_token": process_identity()["process_start_token"],
        "coordinator_epoch": str(identity.get("coordinator_epoch", "")),
        "admission_identity_sha256": str(admission_hash),
        "lifecycle_root": str(lifecycle_root.resolve()),
        "started_monotonic_s": float(started), "ended_monotonic_s": float(ended),
        "exit_status": str(exit_status), "claimed_ordinals": [int(value) for value in claimed],
        "created_files": [str(path.resolve()) for path in created],
        "writer_id": str(identity.get("grant_id", "")),
        "intent_id": str(identity.get("intent_id", "")),
        "reservation_id": str(identity.get("reservation_id", "")),
        "backend": "numpy/cpu", "physical_gpu_identity": "NOT_APPLICABLE_LOCAL_CPU",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if error is not None:
        payload["error"] = {
            "type": type(error).__name__, "message": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        }
    if gpu_fields is not None:
        payload.update(gpu_fields)
    payload["exit_status"] = str(exit_status)
    atomic_json(path, payload, overwrite=False)
    return {**payload, "report_path": str(path), "report_sha256": sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lifecycle-root", required=True)
    parser.add_argument("--pulse", required=True, type=int)
    parser.add_argument("--worker-index", required=True, type=int)
    parser.add_argument("--hydro-workers", required=True, type=int)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--admission-identity-sha256", required=True)
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--intent-id")
    parser.add_argument("--reservation-id")
    parser.add_argument("--writer-id")
    parser.add_argument("--grant-id", required=True)
    parser.add_argument("--grant-nonce", required=True)
    parser.add_argument("--scientific-pair-id", required=True)
    parser.add_argument("--producer-grant-id", required=True)
    parser.add_argument("--implementation-identity-sha256", required=True)
    args = parser.parse_args()

    lifecycle_root = Path(args.lifecycle_root).resolve()
    storage_root = Path(args.storage_root).resolve() if args.storage_root is not None else None
    config_path = Path(args.config_path).resolve()
    if int(args.hydro_workers) <= 0 or int(args.worker_index) < 0:
        raise ValueError("invalid Candidate hydro worker topology")
    if storage_root is None or not args.intent_id or not args.reservation_id or not args.writer_id:
        raise ValueError("real multiprocess hydro worker requires storage ownership identity")
    if args.hydro_workers > 1:
        # One worker alone must not mistake its last block for producer completion.
        pass

    admission = json.loads((storage_root / "E5_1A_ADMISSION_IDENTITY.json").read_text(encoding="utf-8"))
    lock_timeout_s = resolve_storage_lock_timeout_s(
        admission, expected_hash=args.admission_identity_sha256,
    )
    budget = StorageBudget.open_for_worker(
        storage_root, campaign_id=str(admission["campaign_id"]), admission_identity=admission,
        lock_timeout_s=lock_timeout_s,
    )
    grant = budget.writer_grants().get(args.grant_id)
    if not isinstance(grant, Mapping):
        raise RuntimeError("hydro child grant is missing")
    scope = grant["scope"]
    implementation = capture_implementation_identity()
    if (scope.get("role") != "HYDRO" or scope.get("role_index") != args.worker_index
            or scope.get("trajectory") != "C" or scope.get("pulse") != args.pulse
            or scope.get("scientific_pair_id") != args.scientific_pair_id
            or scope.get("admission_identity_sha256") != args.admission_identity_sha256
            or scope.get("reservation_id") != args.reservation_id
            or scope.get("intent_id") != args.intent_id
            or scope.get("implementation_identity") != implementation
            or implementation.get("identity_sha256") != args.implementation_identity_sha256):
        raise RuntimeError("hydro child grant scope or implementation differs")
    budget.record_child_writer_launch_candidate(
        args.grant_id, nonce=args.grant_nonce, process=process_identity())
    identity = budget.bind_child_writer_grant(
        args.grant_id, nonce=args.grant_nonce, scope=scope)
    budget.authorize_child_writer_operation(
        args.grant_id, nonce=args.grant_nonce,
        coordinator_epoch=identity["coordinator_epoch"],
        operation="NEXT_COMMIT", path=lifecycle_root / "next",
    )
    hydro = _hydro_parameters(config_path)
    lifecycle = StreamingLifecycle.open(lifecycle_root)
    actor = f"e5_1b_c_hydro_{args.pulse}_{args.worker_index}"
    claimed_ordinals: list[int] = []
    created_files: list[Path] = []
    started = time.monotonic()
    error: BaseException | None = None
    grant_close: Mapping[str, Any] | None = None
    try:
        while True:
            block = lifecycle.run_one_hydro_block(actor=actor, **hydro)
            if block:
                claimed_ordinals.extend(int(value) for value in block)
                created_files.extend(
                    lifecycle_root / "next" / f"screen_{int(value):06d}.npz" for value in block
                )
                continue
            if _producer_complete(
                lifecycle_root, scientific_pair_id=args.scientific_pair_id,
                producer_grant_id=args.producer_grant_id):
                manifest = lifecycle.manifest
                if not manifest.get("queue") and not manifest.get("recovery_backlog"):
                    break
            time.sleep(0.01)
    except BaseException as exc:
        error = exc
    finally:
        ended = time.monotonic()
        try:
            grant_close = budget.close_child_writer_grant(
                args.grant_id, nonce=args.grant_nonce,
                coordinator_epoch=identity["coordinator_epoch"],
                status="INTERRUPTED" if error is not None else "CLOSED",
                reason=None if error is None else f"{type(error).__name__}: {error}",
            )
        except BaseException as close_error:
            if error is None:
                error = close_error
            else:
                try:
                    error.add_note(
                        f"grant close failed after primary error: {type(close_error).__name__}: {close_error}"
                    )
                except Exception:
                    pass
        if error is None:
            if not isinstance(grant_close, Mapping) or grant_close.get("state") != "CLOSED":
                error = RuntimeError("hydro child writer grant did not close successfully")
        gpu_fields = None
        try:
            gpu_fields = _gpu_process_receipt_fields(
                f"H{args.worker_index}",
                campaign_id=str(admission["campaign_id"]),
                scientific_pair_id=args.scientific_pair_id,
                coordinator_epoch=str(identity["coordinator_epoch"]),
                implementation_identity_sha256=args.implementation_identity_sha256,
            )
        except BaseException as receipt_error:
            if error is None:
                error = receipt_error
            else:
                try:
                    error.add_note(
                        f"GPU receipt validation failed after primary error: "
                        f"{type(receipt_error).__name__}: {receipt_error}"
                    )
                except Exception:
                    pass
        if error is None:
            try:
                _mark_gpu_success(f"H{args.worker_index}")
            except BaseException as marker_error:
                error = marker_error
        receipt_error = None
        try:
            receipt = _receipt(
                lifecycle_root, pulse=args.pulse, worker_index=args.worker_index,
                lifecycle_root=lifecycle_root, storage_root=storage_root,
                admission_hash=args.admission_identity_sha256, started=started,
                ended=ended, claimed=claimed_ordinals, created=created_files, identity=identity,
                exit_status="PASS" if error is None else "FAIL", error=error,
                gpu_fields=gpu_fields,
            )
        except BaseException as write_error:
            receipt_error = write_error
            if error is None:
                error = write_error
            else:
                try:
                    error.add_note(
                        f"failure receipt write failed after primary error: "
                        f"{type(write_error).__name__}: {write_error}"
                    )
                except Exception:
                    pass
    if error is not None:
        raise error
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
