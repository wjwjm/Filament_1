#!/usr/bin/env python3
"""Run one E2-C validation-only real-POST hydro spatial case."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.hr4e_real_spatial import run_e2c_case
from KHz_filament.hr4e_timestep import write_case_manifest, write_observables_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--screen-id", required=True)
    parser.add_argument("--screen-index", type=int, required=True)
    parser.add_argument("--screen-z-m", type=float, required=True)
    parser.add_argument("--dx-um", type=float, required=True)
    parser.add_argument("--dt-us", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    result = run_e2c_case(
        str(args.screen),
        source_manifest_path=str(args.source_manifest),
        screen_identity={"screen_id": args.screen_id, "screen_index": args.screen_index, "screen_z_m": args.screen_z_m},
        spacing_m=args.dx_um * 1.0e-6,
        dt_hydro=args.dt_us * 1.0e-6,
    )
    result = dict(result)
    result.update({"case_id": args.case_id, "requested_dx_um": args.dx_um, "requested_dt_us": args.dt_us})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = args.out_dir / f"{args.case_id}.json", args.out_dir / f"{args.case_id}.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError("refusing to overwrite existing E2-C outputs")
    write_case_manifest(result, json_path)
    write_observables_csv(result, csv_path)
    print(json.dumps({"case_id": args.case_id, "status": result["status"], "json": str(json_path)}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
