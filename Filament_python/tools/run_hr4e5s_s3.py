#!/usr/bin/env python3
"""Narrow command entry points for HR-4E-5S S3 qualification only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.hr4e5s_s3 import (  # noqa: E402
    compare_exact,
    consume_streaming,
    create_streaming_lifecycle,
    finalize_streaming,
    prepare_input_manifest,
    run_batch_hydro,
    run_optical_path,
)


def _read(path: Path) -> dict:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source-manifest", type=Path, required=True)
    prepare.add_argument("--source-state", type=Path, required=True)
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    initialize = sub.add_parser("initialize-stream")
    initialize.add_argument("--input", type=Path, required=True); initialize.add_argument("--root", type=Path, required=True)
    optical = sub.add_parser("optical")
    optical.add_argument("--input", type=Path, required=True); optical.add_argument("--out-dir", type=Path, required=True); optical.add_argument("--stream-root", type=Path); optical.add_argument("--resume", action="store_true")
    batch = sub.add_parser("batch-hydro")
    batch.add_argument("--input", type=Path, required=True); batch.add_argument("--optical-dir", type=Path, required=True); batch.add_argument("--out-dir", type=Path, required=True)
    consumer = sub.add_parser("consume")
    consumer.add_argument("--input", type=Path, required=True); consumer.add_argument("--stream-root", type=Path, required=True); consumer.add_argument("--producer-complete", type=Path, required=True); consumer.add_argument("--out", type=Path, required=True); consumer.add_argument("--actor", default="hydro_consumer")
    final = sub.add_parser("finalize-stream")
    final.add_argument("--stream-root", type=Path, required=True); final.add_argument("--out", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--input", type=Path, required=True); compare.add_argument("--batch-optical-dir", type=Path, required=True); compare.add_argument("--batch-hydro-dir", type=Path, required=True); compare.add_argument("--stream-optical-dir", type=Path, required=True); compare.add_argument("--stream-root", type=Path, required=True); compare.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_input_manifest(source_manifest_path=args.source_manifest, source_state_path=args.source_state, config_path=args.config, out_path=args.out)
    elif args.command == "initialize-stream":
        lifecycle = create_streaming_lifecycle(input_manifest=_read(args.input), root=args.root)
        result = {"root": str(lifecycle.root), "manifest": str(lifecycle.manifest_path)}
    elif args.command == "optical":
        result = run_optical_path(input_manifest_path=args.input, out_dir=args.out_dir, streaming_root=args.stream_root, resume=bool(args.resume))
    elif args.command == "batch-hydro":
        result = run_batch_hydro(input_manifest_path=args.input, optical_dir=args.optical_dir, out_dir=args.out_dir)
    elif args.command == "consume":
        result = consume_streaming(lifecycle_root=args.stream_root, hydro=_read(args.input)["hydro"], producer_complete=args.producer_complete, actor=args.actor)
        if args.out.exists():
            raise FileExistsError(args.out)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    elif args.command == "finalize-stream":
        result = finalize_streaming(lifecycle_root=args.stream_root, out_path=args.out)
    else:
        result = compare_exact(input_manifest_path=args.input, batch_optical_dir=args.batch_optical_dir, batch_hydro_dir=args.batch_hydro_dir, streaming_optical_dir=args.stream_optical_dir, streaming_root=args.stream_root, out_dir=args.out_dir)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
