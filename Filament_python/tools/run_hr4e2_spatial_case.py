#!/usr/bin/env python3
"""Run one bounded HR-4E-2 E2-A or E2-B validation case."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.hr4e_spatial import E2_SNAPSHOT_TIMES_S, run_e2_advection_case, run_e2_case
from KHz_filament.hr4e_timestep import write_case_manifest, write_observables_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("E2-A", "E2-B"), required=True)
    parser.add_argument("--dx-um", type=float, required=True)
    parser.add_argument("--dt-us", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    if args.dx_um <= 0.0 or args.dt_us <= 0.0:
        parser.error("--dx-um and --dt-us must be positive")
    if args.family == "E2-B":
        result = run_e2_advection_case(spacing_m=args.dx_um * 1.0e-6, dt_hydro=args.dt_us * 1.0e-6)
    else:
        result = run_e2_case(family="E2-A", spacing_m=args.dx_um * 1.0e-6, dt_hydro=args.dt_us * 1.0e-6, snapshot_times_s=E2_SNAPSHOT_TIMES_S)
    result = dict(result)
    result["case_id"] = args.case_id
    result["requested_dx_um"] = args.dx_um
    result["requested_dt_us"] = args.dt_us
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{args.case_id}.json"
    csv_path = args.out_dir / f"{args.case_id}.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError("refusing to overwrite existing E2 case outputs")
    write_case_manifest(result, json_path)
    write_observables_csv(result, csv_path)
    print(json.dumps({"case_id": args.case_id, "status": result["status"], "json": str(json_path), "csv": str(csv_path), "steps": result["hydro_step_count"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
