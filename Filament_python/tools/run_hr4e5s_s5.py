#!/usr/bin/env python3
"""Narrow S5 lifecycle inspection and clean-reference comparison entry points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.hr4e5s_s5 import compare_clean_reference, inspect_lifecycle  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--lifecycle-root", type=Path, required=True)
    inspect.add_argument("--out", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--reference-lifecycle-root", type=Path, required=True)
    compare.add_argument("--reference-optical-dir", type=Path, required=True)
    compare.add_argument("--candidate-lifecycle-root", type=Path, required=True)
    compare.add_argument("--candidate-optical-dir", type=Path, required=True)
    compare.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        result = inspect_lifecycle(lifecycle_root=args.lifecycle_root, out_path=args.out)
    else:
        result = compare_clean_reference(
            reference_lifecycle_root=args.reference_lifecycle_root,
            reference_optical_dir=args.reference_optical_dir,
            candidate_lifecycle_root=args.candidate_lifecycle_root,
            candidate_optical_dir=args.candidate_optical_dir,
            out_dir=args.out_dir,
        )
    print(json.dumps(result, sort_keys=True, default=str))
    return 0 if result.get("status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
