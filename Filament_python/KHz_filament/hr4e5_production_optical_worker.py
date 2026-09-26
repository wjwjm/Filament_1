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
    args = parser.parse_args()

    root = Path(args.storage_root).resolve()
    admission = json.loads((root / "E5_1A_ADMISSION_IDENTITY.json").read_text(encoding="utf-8"))
    budget = StorageBudget.open_for_worker(
        root, campaign_id=str(admission["campaign_id"]), admission_identity=admission,
    )
    identity = budget.writer_receipt(args.writer_id)
    if (str(identity.get("intent_id", "")) != str(args.intent_id)
            or str(identity.get("reservation_id", "")) != str(args.reservation_id)):
        raise RuntimeError("optical writer identity differs from requested ownership")
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
        )
        ended = time.monotonic()
        atomic_json(
            Path(args.lifecycle_root) / "PRODUCER_COMPLETE.json",
            {
                "schema": "khz_filament.hr4e5.e5_1b.producer_complete.v1",
                "status": "PASS", "pulse": int(args.pulse),
                "admission_identity_sha256": str(args.admission_identity_sha256),
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
            "exit_status": "PASS", "created_files": [], "writer_id": str(args.writer_id),
            "intent_id": str(args.intent_id), "reservation_id": str(args.reservation_id),
            "backend": "numpy/cpu", "physical_gpu_identity": "NOT_APPLICABLE_LOCAL_CPU",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        path = Path(args.lifecycle_root) / "process_receipts" / "C_OPTICAL.json"
        atomic_json(path, receipt, overwrite=True)
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
    print(json.dumps({**receipt, "optical_result": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
