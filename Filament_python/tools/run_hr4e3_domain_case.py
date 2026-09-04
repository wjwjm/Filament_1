#!/usr/bin/env python3
"""Run one non-overwriting HR-4E-3 synthetic or real-POST domain case."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.hr4e_domain import (
    build_e3_real_post_state, build_e3_synthetic_state, e3_geometry, run_e3_case,
)
from KHz_filament.hr4e_timestep import write_case_manifest, write_observables_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("synthetic", "real_post"), required=True)
    parser.add_argument("--domain", choices=("D0", "D1", "D2", "D3"), required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--screen", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--screen-id")
    parser.add_argument("--screen-index", type=int)
    parser.add_argument("--screen-z-m", type=float)
    args = parser.parse_args()
    geometry = e3_geometry(args.domain)
    json_path, csv_path, checkpoints = args.out_dir / f"{args.case_id}.json", args.out_dir / f"{args.case_id}.csv", args.out_dir / "checkpoints"
    if json_path.exists() or csv_path.exists() or checkpoints.exists():
        raise FileExistsError("refusing to overwrite existing E3 outputs")
    if args.kind == "synthetic":
        state, metadata = build_e3_synthetic_state(geometry), {"kind": "analytic_gaussian", "domain_extension": "analytic_direct_evaluation"}
    else:
        required = (args.screen, args.source_manifest, args.screen_id, args.screen_index, args.screen_z_m)
        if any(value is None for value in required):
            parser.error("real_post requires --screen, --source-manifest, and complete screen identity")
        prepared = build_e3_real_post_state(str(args.screen), source_manifest_path=str(args.source_manifest), screen_identity={"screen_id": args.screen_id, "screen_index": args.screen_index, "screen_z_m": args.screen_z_m}, geometry=geometry)
        state, metadata = prepared["state"], {"kind": "real_hr3b_post_validation_extension", **prepared}
        metadata.pop("state")
    result = run_e3_case(case_id=args.case_id, family="E3-A" if args.kind == "synthetic" else "E3-B", geometry=geometry, state=state, checkpoint_dir=checkpoints, initial_metadata=metadata)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_case_manifest(result, json_path)
    write_observables_csv(result, csv_path)
    print(json.dumps({"case_id": args.case_id, "status": result["status"], "json": str(json_path)}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
