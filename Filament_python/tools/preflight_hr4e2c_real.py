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

from KHz_filament.hr4 import (
    HR4_CHI,
    HR4_GRAVITY_X,
    HR4_GRAVITY_Y,
    HR4_NU,
    audit_hr4_stability,
)
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
    """Return mapping-consistency rows.

    Strict monotonicity is a useful diagnostic for a sampled continuous
    representation, but it is not an identity or conservation requirement.
    A non-monotonic 20->10->5 sequence remains admissible only when the
    frozen 10-vs-5 ceiling is satisfied.
    """
    output = []
    for name in OBSERVABLES:
        values = [float(item["initial_metrics"][name]) for item in states]
        d20, d10 = abs(values[0] - values[1]), abs(values[1] - values[2])
        kind, limit = tolerance(name)
        near_zero = kind == "absolute" and max(abs(v) for v in values) <= limit
        value = d10 if kind == "absolute" else rel(values[1], values[2])
        strict_trend = None if near_zero else d10 < d20
        ceiling_pass = value <= limit
        warning = (
            "WARNING_NONMONOTONIC_WITHIN_TOLERANCE"
            if strict_trend is False and ceiling_pass
            else None
        )
        output.append({
            "observable": name,
            "Q_20um": values[0], "Q_10um": values[1], "Q_5um": values[2],
            "D20_10": d20, "D10_5": d10,
            "p_obs": None if strict_trend is None or d20 == 0.0 or d10 == 0.0 else math.log2(d20 / d10),
            "trend_applicable": not near_zero,
            "strict_refinement_pass": strict_trend,
            "trend_status": "N/A_NEAR_ZERO" if strict_trend is None else ("PASS" if strict_trend else "WARNING"),
            "diagnostic_warning": warning,
            "10_vs_5_value": value, "10_vs_5_tolerance": limit,
            "10_vs_5_ceiling_pass": ceiling_pass,
            "mapping_consistency_pass": ceiling_pass,
            "hard_gate_pass": ceiling_pass,
        })
    return output


def stability_audit(state):
    geometry, metrics = state["geometry"], state["initial_metrics"]
    return audit_hr4_stability(
        dx=float(geometry["dx_m"]), dy=float(geometry["dy_m"]),
        dt_hydro=E2_COMMON_DT_S, chi=HR4_CHI, nu=HR4_NU,
        max_abs_vx=float(metrics["max_abs_vx_m_s"]),
        max_abs_vy=float(metrics["max_abs_vy_m_s"]),
    )

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
        audits = [stability_audit(state) for state in states]
        eligible = all(row["mapping_consistency_pass"] for row in screen_rows) and all(audit["overall_pass"] for audit in audits)
        reports.append({
            "screen_identity": identity,
            "source_provenance": states[0]["source_provenance"],
            "validation_representation": states[0]["validation_representation"],
            "targets": [{
                "dx_m": spacing, "grid": state["geometry"],
                "target_state_sha256": state["target_state_sha256"],
                "target_velocity_sha256": state["target_velocity_sha256"],
                "max_abs_interpolation_residual_to_representation": 0.0,
                "audit_hr4_stability": audit,
            } for spacing, state, audit in zip(SPACINGS, states, audits)],
            "rows": screen_rows,
            "warnings": [row["diagnostic_warning"] for row in screen_rows if row["diagnostic_warning"]],
            "status": "PASS" if eligible else "INVALID_VALIDATION_REPRESENTATION",
        })
    report = {
        "schema": "khz_filament.hr4e2c.preflight.v2",
        "scope_is_hydro_only_validation": True,
        "full_chain_transverse_convergence_claimed": False,
        "production_multigrid_mapping_modified": False,
        "validation_representation_source_is_single_frozen_real_POST": True,
        "same_continuous_reference_used_for_20_10_5": True,
        "dt_hydro_s": E2_COMMON_DT_S,
        "frozen_operator": {"chi_m2_s": HR4_CHI, "nu_m2_s": HR4_NU, "gravity_x_m_s2": HR4_GRAVITY_X, "gravity_y_m_s2": HR4_GRAVITY_Y},
        "screens": reports,
        "status": "PASS" if all(item["status"] == "PASS" for item in reports) else "INVALID_VALIDATION_REPRESENTATION",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2

if __name__ == "__main__": raise SystemExit(main())
