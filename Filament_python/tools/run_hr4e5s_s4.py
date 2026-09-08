#!/usr/bin/env python3
"""Narrow command entry points for HR-4E-5S S4 scheduling qualification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.hr4e5s_s3 import consume_streaming, finalize_streaming  # noqa: E402
from KHz_filament.hr4e5s_s4 import compare_replay_next, enqueue_hydro_replay, prepare_hydro_replay, summarize_telemetry  # noqa: E402


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
    prepare = sub.add_parser("prepare-replay")
    prepare.add_argument("--input", type=Path, required=True); prepare.add_argument("--batch-optical-dir", type=Path, required=True); prepare.add_argument("--root", type=Path, required=True)
    enqueue = sub.add_parser("enqueue-replay")
    enqueue.add_argument("--stream-root", type=Path, required=True); enqueue.add_argument("--producer-complete", type=Path, required=True); enqueue.add_argument("--out", type=Path, required=True)
    consume = sub.add_parser("consume")
    consume.add_argument("--input", type=Path, required=True); consume.add_argument("--stream-root", type=Path, required=True); consume.add_argument("--producer-complete", type=Path, required=True); consume.add_argument("--actor", required=True); consume.add_argument("--out", type=Path, required=True)
    final = sub.add_parser("finalize")
    final.add_argument("--stream-root", type=Path, required=True); final.add_argument("--out", type=Path, required=True)
    compare = sub.add_parser("compare-replay")
    compare.add_argument("--batch-hydro-dir", type=Path, required=True); compare.add_argument("--stream-root", type=Path, required=True); compare.add_argument("--out", type=Path, required=True)
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--stream-root", type=Path, required=True); summarize.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-replay":
        result = prepare_hydro_replay(input_manifest_path=args.input, batch_optical_dir=args.batch_optical_dir, root=args.root)
    elif args.command == "enqueue-replay":
        result = enqueue_hydro_replay(lifecycle_root=args.stream_root, producer_complete=args.producer_complete); _write(args.out, result)
    elif args.command == "consume":
        result = consume_streaming(lifecycle_root=args.stream_root, hydro=_read(args.input)["hydro"], producer_complete=args.producer_complete, actor=args.actor); _write(args.out, result)
    elif args.command == "finalize":
        result = finalize_streaming(lifecycle_root=args.stream_root, out_path=args.out)
    elif args.command == "compare-replay":
        result = compare_replay_next(batch_hydro_dir=args.batch_hydro_dir, lifecycle_root=args.stream_root, out_path=args.out)
    else:
        result = summarize_telemetry(lifecycle_root=args.stream_root, out_dir=args.out_dir)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
