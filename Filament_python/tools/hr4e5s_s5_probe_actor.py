#!/usr/bin/env python3
"""Stdlib-only Phase-0 probe actor and receipt helper for S5-FINAL.

This entry point intentionally has no dependency on the scientific package,
CuPy, or CUDA.  The production batch invokes this file directly; keeping the
Python source out of a nested ``bash -c`` heredoc makes the launcher boundary
locally executable and testable.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "khz_filament.hr4e5s.s5_final.phase0.v1"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    if not path.parent.is_dir():
        raise FileNotFoundError(f"parent directory does not exist: {path.parent}")
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _slurm_identity(*, local_test: bool) -> dict[str, str]:
    required = ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURMD_NODENAME")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing Slurm identity: {', '.join(missing)}")
    if not local_test and os.environ.get("S5_LOCAL_TEST"):
        raise RuntimeError("production probe rejects S5_LOCAL_TEST environment")
    return {name: os.environ[name] for name in required}


def write_record(out: Path, pairs: Sequence[str]) -> int:
    value: dict[str, Any] = {"schema": SCHEMA}
    for pair in pairs:
        key, separator, raw = pair.partition("=")
        if not separator or not key:
            raise ValueError("record entries must be key=value")
        if raw in {"true", "false"}:
            value[key] = raw == "true"
        elif raw.lstrip("-").isdigit():
            value[key] = int(raw)
        else:
            value[key] = raw
    _atomic_json(out, value)
    return 0


def run_actor(args: argparse.Namespace) -> int:
    identity = _slurm_identity(local_test=args.local_test)
    out, heartbeat = Path(args.identity), Path(args.heartbeat)
    if not heartbeat.parent.is_dir():
        raise FileNotFoundError(f"heartbeat parent does not exist: {heartbeat.parent}")
    if args.actor not in {"probe_target", "probe_survivor"}:
        raise ValueError("invalid Phase-0 probe actor")
    value = {
        "schema": SCHEMA,
        "kind": "identity",
        "actor": args.actor,
        "job_id": identity["SLURM_JOB_ID"],
        "step_id": identity["SLURM_STEP_ID"],
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "start_time": time.time(),
        "heartbeat_path": str(heartbeat),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cupy_imported": False,
        "cuda_context_initialized": False,
        "local_test": bool(args.local_test),
    }
    _atomic_json(out, value)
    stopped = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopped:
        heartbeat.write_text(f"{time.time_ns()}\n", encoding="ascii", newline="\n")
        time.sleep(args.heartbeat_seconds)
    return 143


def capture_listpids(args: argparse.Namespace) -> int:
    started = time.time()
    completed = subprocess.run(["scontrol", "listpids", args.query], text=True, capture_output=True, check=False)
    _atomic_json(Path(args.out), {
        "schema": SCHEMA,
        "kind": "listpids_evidence",
        "command": ["scontrol", "listpids", args.query],
        "hostname": socket.gethostname(),
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "started_time": started,
        "finished_time": time.time(),
    })
    return completed.returncode


def field(path: Path, key: str) -> int:
    value = json.loads(path.read_text(encoding="utf-8"))
    print(value[key])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("write-record")
    record.add_argument("--out", type=Path, required=True)
    record.add_argument("entries", nargs="+")
    actor = commands.add_parser("actor")
    actor.add_argument("--actor", required=True)
    actor.add_argument("--identity", required=True)
    actor.add_argument("--heartbeat", required=True)
    actor.add_argument("--heartbeat-seconds", type=float, default=0.2)
    actor.add_argument("--local-test", action="store_true")
    capture = commands.add_parser("capture-listpids")
    capture.add_argument("--out", required=True)
    capture.add_argument("--query", required=True)
    get = commands.add_parser("field")
    get.add_argument("--path", type=Path, required=True)
    get.add_argument("--key", required=True)
    args = parser.parse_args(argv)
    if args.command == "write-record":
        return write_record(args.out, args.entries)
    if args.command == "actor":
        if args.heartbeat_seconds <= 0:
            parser.error("--heartbeat-seconds must be positive")
        return run_actor(args)
    if args.command == "capture-listpids":
        return capture_listpids(args)
    return field(args.path, args.key)


if __name__ == "__main__":
    raise SystemExit(main())
