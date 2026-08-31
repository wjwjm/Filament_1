#!/usr/bin/env python3
"""Run one bounded HR-4E-1 timestep case and write JSON/CSV observables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.hr4e_timestep import (  # noqa: E402
    E1A_SNAPSHOT_TIMES_S,
    E1B_SNAPSHOT_TIMES_S,
    run_e1a_case,
    run_e1b_case,
    write_case_manifest,
    write_observables_csv,
)


def _case_id(dt_us: float, benchmark: str) -> str:
    prefix = "E1A" if benchmark == "E1-A" else "E1B"
    text = f"{dt_us:.12g}".replace(".", "p")
    return f"{prefix}_dt{text}us"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("E1-A", "E1-B"), default="E1-A")
    parser.add_argument("--dt-us", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--e1b-screen", type=Path)
    parser.add_argument("--screen-id")
    parser.add_argument("--screen-index", type=int)
    parser.add_argument("--screen-z-m", type=float)
    parser.add_argument("--dtype", choices=("fp32", "fp64"), default="fp64")
    parser.add_argument(
        "--snapshot-us",
        nargs="*",
        type=float,
        default=None,
        help="optional increasing snapshot times; default is the formal E1 list",
    )
    args = parser.parse_args(argv)
    if args.dt_us <= 0.0 or not np.isfinite(args.dt_us):
        parser.error("--dt-us must be positive and finite")
    case_id = args.case_id or _case_id(args.dt_us, args.benchmark)
    snapshot_times_s = (
        (E1B_SNAPSHOT_TIMES_S if args.benchmark == "E1-B" else E1A_SNAPSHOT_TIMES_S)
        if args.snapshot_us is None
        else tuple(float(value) * 1.0e-6 for value in args.snapshot_us)
    )
    dtype = np.float32 if args.dtype == "fp32" else np.float64
    screen_identity = None
    if args.benchmark == "E1-B":
        if args.e1b_screen is None:
            parser.error("--e1b-screen is required for E1-B")
        screen_identity = {
            "screen_id": args.screen_id or args.e1b_screen.stem,
            "screen_index": args.screen_index,
            "screen_z_m": args.screen_z_m,
        }
        screen_identity = {key: value for key, value in screen_identity.items() if value is not None}
        result = run_e1b_case(
            args.e1b_screen,
            dt_hydro=args.dt_us * 1.0e-6,
            screen_identity=screen_identity,
            snapshot_times_s=snapshot_times_s,
        )
    else:
        result = run_e1a_case(
            dt_hydro=args.dt_us * 1.0e-6,
            snapshot_times_s=snapshot_times_s,
            dtype=dtype,
        )
    result = dict(result)
    result["case_id"] = case_id
    result["requested_dt_us"] = float(args.dt_us)
    result["output_contract"] = {
        "full_hydro_history_stored": False,
        "recorded_snapshots": [float(item) for item in snapshot_times_s],
        "json_observables": f"{case_id}.json",
        "csv_observables": f"{case_id}.csv",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{case_id}.json"
    csv_path = args.out_dir / f"{case_id}.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError(f"refusing to overwrite existing E1 case outputs for {case_id}")
    json_path = write_case_manifest(result, json_path)
    csv_path = write_observables_csv(result, csv_path)
    print(
        json.dumps(
            {
                "case_id": case_id,
                "status": result["status"],
                "json": str(json_path),
                "csv": str(csv_path),
                "hydro_step_count": result["hydro_step_count"],
                "wall_time_s": result["wall_time_s"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
