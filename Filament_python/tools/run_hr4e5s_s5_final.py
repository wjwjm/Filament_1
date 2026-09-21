#!/usr/bin/env python3
"""Narrow entry points for the one S5-FINAL 1+2 worker-loss scenario."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.hr4e5s_s3 import finalize_streaming  # noqa: E402
from KHz_filament.hr4e5s_s5_final import (  # noqa: E402
    S5_FINAL_CASE_ID,
    bootstrap_recovery,
    compare_exact,
    consume_final_streaming,
    freeze_expected_recovery_effects,
    run_recovery_optical,
    snapshot_interrupted_state,
    validate_bootstrap_ready_receipt,
    validate_bootstrap_receipt,
    write_worker_identity,
)


def _read(path: Path) -> dict:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    identity = sub.add_parser("write-identity")
    identity.add_argument("--out", type=Path, required=True); identity.add_argument("--actor", required=True)
    identity.add_argument("--pid", type=int, required=True); identity.add_argument("--step-id", required=True)
    identity.add_argument("--job-id", required=True); identity.add_argument("--node", required=True); identity.add_argument("--phase", required=True); identity.add_argument("--execution-epoch")
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--stream-root", type=Path, required=True); snapshot.add_argument("--out", type=Path, required=True)
    effects = sub.add_parser("freeze-effects")
    effects.add_argument("--stream-root", type=Path, required=True); effects.add_argument("--inventory", type=Path, required=True); effects.add_argument("--out", type=Path, required=True)
    bootstrap = sub.add_parser("bootstrap")
    bootstrap.add_argument("--stream-root", type=Path, required=True); bootstrap.add_argument("--effects", type=Path, required=True); bootstrap.add_argument("--out", type=Path, required=True); bootstrap.add_argument("--runtime-sha", required=True)
    validate = sub.add_parser("validate-bootstrap")
    validate.add_argument("--stream-root", type=Path, required=True); validate.add_argument("--receipt", type=Path, required=True); validate.add_argument("--runtime-sha", required=True)
    validate_ready = sub.add_parser("validate-bootstrap-ready")
    validate_ready.add_argument("--stream-root", type=Path, required=True); validate_ready.add_argument("--receipt", type=Path, required=True); validate_ready.add_argument("--ready", type=Path, required=True); validate_ready.add_argument("--runtime-sha", required=True)
    optical = sub.add_parser("recovery-optical")
    optical.add_argument("--input", type=Path, required=True); optical.add_argument("--stream-root", type=Path, required=True); optical.add_argument("--out-dir", type=Path, required=True); optical.add_argument("--bootstrap-receipt", type=Path, required=True); optical.add_argument("--bootstrap-ready", type=Path, required=True)
    consume = sub.add_parser("consume")
    consume.add_argument("--input", type=Path, required=True); consume.add_argument("--stream-root", type=Path, required=True); consume.add_argument("--producer-complete", type=Path, required=True); consume.add_argument("--actor", required=True); consume.add_argument("--out", type=Path, required=True); consume.add_argument("--arming-dir", type=Path); consume.add_argument("--execution-epoch"); consume.add_argument("--bootstrap-ready", type=Path)
    final = sub.add_parser("finalize")
    final.add_argument("--stream-root", type=Path, required=True); final.add_argument("--out", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--reference-lifecycle", type=Path, required=True); compare.add_argument("--reference-optical", type=Path, required=True); compare.add_argument("--candidate-lifecycle", type=Path, required=True); compare.add_argument("--candidate-optical", type=Path, required=True); compare.add_argument("--effects", type=Path, required=True); compare.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "write-identity":
        result = write_worker_identity(out_path=args.out, actor=args.actor, worker_pid=args.pid, step_id=args.step_id, job_id=args.job_id, node=args.node, phase=args.phase, execution_epoch=args.execution_epoch)
    elif args.command == "snapshot":
        result = snapshot_interrupted_state(lifecycle_root=args.stream_root, out_path=args.out)
    elif args.command == "freeze-effects":
        result = freeze_expected_recovery_effects(lifecycle_root=args.stream_root, inventory_path=args.inventory, out_path=args.out)
    elif args.command == "bootstrap":
        result = bootstrap_recovery(lifecycle_root=args.stream_root, effects_path=args.effects, out_path=args.out, runtime_sha=args.runtime_sha, case_id=S5_FINAL_CASE_ID)
    elif args.command == "validate-bootstrap":
        validate_bootstrap_receipt(receipt_path=args.receipt, lifecycle_root=args.stream_root, runtime_sha=args.runtime_sha)
        result = {"status": "PASS", "receipt": str(args.receipt)}
    elif args.command == "validate-bootstrap-ready":
        validate_bootstrap_ready_receipt(ready_path=args.ready, receipt_path=args.receipt, lifecycle_root=args.stream_root, runtime_sha=args.runtime_sha)
        result = {"status": "PASS", "receipt": str(args.receipt), "ready": str(args.ready)}
    elif args.command == "recovery-optical":
        result = run_recovery_optical(input_manifest_path=args.input, lifecycle_root=args.stream_root, out_dir=args.out_dir, bootstrap_receipt_path=args.bootstrap_receipt, bootstrap_ready_path=args.bootstrap_ready)
    elif args.command == "consume":
        result = consume_final_streaming(lifecycle_root=args.stream_root, hydro=_read(args.input)["hydro"], producer_complete=args.producer_complete, actor=args.actor, arming_dir=args.arming_dir, execution_epoch=args.execution_epoch, bootstrap_ready_path=args.bootstrap_ready)
        _write(args.out, result)
    elif args.command == "finalize":
        result = finalize_streaming(lifecycle_root=args.stream_root, out_path=args.out)
    else:
        result = compare_exact(reference_lifecycle_root=args.reference_lifecycle, reference_optical_dir=args.reference_optical, candidate_lifecycle_root=args.candidate_lifecycle, candidate_optical_dir=args.candidate_optical, effects_path=args.effects, out_dir=args.out_dir)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
