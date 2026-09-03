#!/usr/bin/env python3
"""Generate a non-overwriting E2-C validation-representation preflight report."""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.hr4e_real_spatial import build_e2c_validation_state
from KHz_filament.hr4e_spatial import E2_COMMON_DT_S
from KHz_filament.hr4e_timestep import json_safe

SPACINGS = (20.0e-6, 10.0e-6, 5.0e-6)
OBSERVABLES = ("xc_m", "yc_m", "sigma_x_m", "sigma_y_m", "min_delta_n", "M0_negative_index_m2")

def tolerance(name: str):
    if name in {"xc_m", "yc_m"}: return "absolute", 1.6e-6
    if name in {"sigma_x_m", "sigma_y_m", "M0_negative_index_m2"}: return "relative", 0.01
    return "relative", 0.02

def rel(a, b): return 0.0 if b == 0.0 and a == 0.0 else abs(a - b) / abs(b)

def rows(states):
    output = []
    for name in OBSERVABLES:
        values = [float(item["initial_metrics"][name]) for item in states]
        d20, d10 = abs(values[0] - values[1]), abs(values[1] - values[2])
        kind, limit = tolerance(name)
        near_zero = kind == "absolute" and max(abs(v) for v in values) <= limit
        value = d10 if kind == "absolute" else rel(values[1], values[2])
        trend = None if near_zero else d10 < d20
        output.append({"observable": name, "Q_20um": values[0], "Q_10um": values[1], "Q_5um": values[2], "D20_10": d20, "D10_5": d10, "p_obs": None if trend is None or d20 == 0.0 or d10 == 0.0 else math.log2(d20 / d10), "trend_applicable": not near_zero, "trend_status": "N/A_NEAR_ZERO" if near_zero else ("PASS" if trend else "FAIL"), "10_vs_5_value": value, "10_vs_5_tolerance": limit, "hard_gate_pass": value <= limit and (trend is None or trend)})
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    spec = json.loads(args.sources.read_text(encoding="utf-8"))
    reports = []
    for screen in spec["screens"]:
        identity = {key: screen[key] for key in ("screen_id", "screen_index", "screen_z_m")}
        states = [build_e2c_validation_state(screen["screen"], source_manifest_path=spec["source_manifest"], screen_identity=identity, spacing_m=spacing) for spacing in SPACINGS]
        screen_rows = rows(states)
        reports.append({"screen_identity": identity, "source_provenance": states[0]["source_provenance"], "validation_representation": states[0]["validation_representation"], "targets": [{"dx_m": spacing, "grid": state["geometry"], "target_state_sha256": state["target_state_sha256"], "target_velocity_sha256": state["target_velocity_sha256"]} for spacing, state in zip(SPACINGS, states)], "rows": screen_rows, "status": "PASS" if all(row["hard_gate_pass"] for row in screen_rows) else "INVALID_VALIDATION_REPRESENTATION"})
    report = {"schema": "khz_filament.hr4e2c.preflight.v1", "scope_is_hydro_only_validation": True, "production_multigrid_mapping_modified": False, "dt_hydro_s": E2_COMMON_DT_S, "screens": reports, "status": "PASS" if all(item["status"] == "PASS" for item in reports) else "INVALID_VALIDATION_REPRESENTATION"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2

if __name__ == "__main__": raise SystemExit(main())
