"""E5-1B WP2 real Candidate hydro worker entry.

This module owns process orchestration only.  All scientific work remains the
existing ``StreamingLifecycle.run_one_hydro_block`` call and the frozen HR-4
single-screen solver behind it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

from .confio import load_all
from .hr4d_pulse_lifecycle import build_interpulse_step_schedule
from .hr4e5_evidence import atomic_json, sha256_file
from .hr4e5s_streaming import StreamingLifecycle
from .hr4e5_storage import PROCESS_RECEIPT_SCHEMA, StorageBudget, process_identity


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


def _producer_complete(root: Path) -> bool:
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
    )


def _receipt(root: Path, *, pulse: int, worker_index: int, lifecycle_root: Path,
             storage_root: Path, admission_hash: str, started: float, ended: float,
             claimed: list[int], created: list[Path], identity: Mapping[str, Any]) -> dict[str, Any]:
    path = lifecycle_root / "process_receipts" / f"C_HYDRO_{int(worker_index):03d}.json"
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
        "exit_status": "PASS", "claimed_ordinals": [int(value) for value in claimed],
        "created_files": [str(path.resolve()) for path in created],
        "writer_id": str(identity.get("writer_id", "")),
        "intent_id": str(identity.get("intent_id", "")),
        "reservation_id": str(identity.get("reservation_id", "")),
        "backend": "numpy/cpu", "physical_gpu_identity": "NOT_APPLICABLE_LOCAL_CPU",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(path, payload, overwrite=True)
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
    budget = StorageBudget.open_for_worker(
        storage_root, campaign_id=str(admission["campaign_id"]), admission_identity=admission,
    )
    identity = budget.writer_receipt(args.writer_id)
    if (str(identity.get("intent_id", "")) != str(args.intent_id)
            or str(identity.get("reservation_id", "")) != str(args.reservation_id)):
        raise RuntimeError("hydro writer identity differs from requested ownership")
    hydro = _hydro_parameters(config_path)
    lifecycle = StreamingLifecycle.open(lifecycle_root)
    actor = f"e5_1b_c_hydro_{args.pulse}_{args.worker_index}"
    claimed_ordinals: list[int] = []
    created_files: list[Path] = []
    started = time.monotonic()
    error: BaseException | None = None
    try:
        while True:
            block = lifecycle.run_one_hydro_block(actor=actor, **hydro)
            if block:
                claimed_ordinals.extend(int(value) for value in block)
                created_files.extend(
                    lifecycle_root / "next" / f"screen_{int(value):06d}.npz" for value in block
                )
                continue
            if _producer_complete(lifecycle_root):
                manifest = lifecycle.manifest
                if not manifest.get("queue") and not manifest.get("recovery_backlog"):
                    break
            time.sleep(0.01)
    except BaseException as exc:
        error = exc
        raise
    finally:
        ended = time.monotonic()
        try:
            budget.close_writer(
                args.writer_id, coordinator_epoch=identity["coordinator_epoch"],
                status="INTERRUPTED" if error is not None else "CLOSED",
                reason=None if error is None else f"{type(error).__name__}: {error}",
            )
        except Exception:
            if error is None:
                raise
        receipt = _receipt(
            lifecycle_root, pulse=args.pulse, worker_index=args.worker_index,
            lifecycle_root=lifecycle_root, storage_root=storage_root,
            admission_hash=args.admission_identity_sha256, started=started,
            ended=ended, claimed=claimed_ordinals, created=created_files, identity=identity,
        )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
