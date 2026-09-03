#!/usr/bin/env python3
"""Classify validation-only E2-C real-POST spatial convergence by screen."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from KHz_filament.hr4e_spatial import (
    E2_CENTROID_TOLERANCE_M,
    E2_EXTREME_RELATIVE_TOLERANCE,
    E2_M0_RELATIVE_TOLERANCE,
    E2_WIDTH_RELATIVE_TOLERANCE,
)
from KHz_filament.hr4e_timestep import json_safe

SPACINGS = (20.0e-6, 10.0e-6, 5.0e-6)
OBSERVABLES = ("xc_m", "yc_m", "sigma_x_m", "sigma_y_m", "min_delta_n", "max_abs_vx_m_s", "max_abs_vy_m_s", "max_abs_v_m_s", "M0_negative_index_m2")
WIDTHS = {"sigma_x_m", "sigma_y_m"}
EXTREMES = {"min_delta_n", "max_abs_vx_m_s", "max_abs_vy_m_s", "max_abs_v_m_s"}


def _rel(a: float, b: float) -> float:
    return 0.0 if a == b == 0.0 else (float("inf") if b == 0.0 else abs(a - b) / abs(b))


def _tolerance(observable: str) -> tuple[str, float]:
    if observable in {"xc_m", "yc_m"}:
        return "absolute", E2_CENTROID_TOLERANCE_M
    if observable in WIDTHS:
        return "relative", E2_WIDTH_RELATIVE_TOLERANCE
    if observable == "M0_negative_index_m2":
        return "relative", E2_M0_RELATIVE_TOLERANCE
    return "relative", E2_EXTREME_RELATIVE_TOLERANCE


def _snapshot(case: Mapping[str, Any], horizon_us: float) -> Mapping[str, Any] | None:
    return next((item for item in case.get("snapshots", []) if math.isclose(float(item.get("time_us", float("nan"))), horizon_us, rel_tol=0.0, abs_tol=1.0e-8)), None)


def _screen_id(case: Mapping[str, Any]) -> str:
    return str(case["source_provenance"]["screen_identity"]["screen_id"])


def _screen_report(cases: Sequence[Mapping[str, Any]], horizon_us: float = 100.0) -> dict[str, Any]:
    selected = []
    for spacing in SPACINGS:
        matching = [case for case in cases if math.isclose(float(case["configuration"]["grid"]["dx_m"]), spacing, rel_tol=0.0, abs_tol=1.0e-15)]
        if len(matching) != 1:
            return {"status": "INVALID", "reason": "missing_or_ambiguous_grid"}
        selected.append(matching[0])
    base = selected[0]
    source = base.get("source_provenance", {})
    representation = base.get("validation_representation", {})
    common = all(
        item.get("source_provenance") == source
        and item.get("validation_representation") == representation
        and item.get("configuration", {}).get("dt_hydro_s") == base.get("configuration", {}).get("dt_hydro_s")
        and item.get("configuration", {}).get("operator") == base.get("configuration", {}).get("operator")
        for item in selected
    )
    snaps = [_snapshot(case, horizon_us) for case in selected]
    clean = all(snap and not snap.get("boundary_contaminated", True) for snap in snaps)
    stable = all(case.get("status") == "PASS" and case.get("stability", {}).get("overall_pass") for case in selected)
    rows = []
    for observable in OBSERVABLES:
        values = [float(snap[observable]) if snap else float("nan") for snap in snaps]
        q20, q10, q5 = values
        d20, d10 = abs(q20 - q10), abs(q10 - q5)
        kind, limit = _tolerance(observable)
        error = d10 if kind == "absolute" else _rel(q10, q5)
        ceiling_pass = math.isfinite(error) and error <= limit
        machine_zero = max(abs(value) for value in values) <= 1.0e-14
        trend = None if machine_zero else d10 < d20
        rows.append({
            "observable": observable, "Q_20um": q20, "Q_10um": q10, "Q_5um": q5,
            "D20_10": d20, "D10_5": d10,
            "p_obs": None if trend is None or d20 == 0.0 or d10 == 0.0 else math.log2(d20 / d10),
            "10_vs_5_value": error, "10_vs_5_tolerance": limit, "10_vs_5_pass": ceiling_pass,
            "trend_applicable": not machine_zero, "trend_status": "N/A_MACHINE_ZERO" if trend is None else ("PASS" if trend else "WARNING"),
            "diagnostic_warning": "WARNING_NONMONOTONIC_WITHIN_TOLERANCE" if trend is False and ceiling_pass else None,
        })
    if not common or not stable or not clean:
        status = "INVALID"
    elif not all(row["10_vs_5_pass"] for row in rows):
        status = "FAIL"
    elif any(row["diagnostic_warning"] for row in rows):
        status = "WARNING"
    else:
        status = "PASS"
    return {
        "status": status, "screen_identity": source.get("screen_identity"),
        "source_provenance": source, "validation_representation": representation,
        "configuration_guard_pass": common, "stable": stable, "boundary_contaminated": not clean,
        "horizon_us": horizon_us, "rows": rows, "case_ids": [case.get("case_id") for case in selected],
    }


def summarize(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_screen: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        by_screen.setdefault(_screen_id(case), []).append(case)
    reports = [_screen_report(group) for _, group in sorted(by_screen.items())]
    statuses = [report["status"] for report in reports]
    status = "PASS" if statuses and all(item == "PASS" for item in statuses) else ("WARNING" if statuses and all(item in {"PASS", "WARNING"} for item in statuses) else ("FAIL" if "FAIL" in statuses else "INVALID"))
    return {
        "schema": "khz_filament.hr4e2c.real_spatial_summary.v1", "status": status,
        "classification": "E2-C_PASS" if status == "PASS" else f"E2-C_{status}",
        "scope_is_hydro_only_validation": True,
        "full_chain_transverse_convergence_claimed": False,
        "production_multigrid_mapping_modified": False,
        "validation_only_statement": "The 5 um initial field adds no physical information beyond the frozen 10 um POST morphology; bilinear sampling is an E2-C validation adapter, not a production HR-3B-to-HR-4 mapper.",
        "screens": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = summarize([json.loads(path.read_text(encoding="utf-8")) for path in args.case])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if report["status"] in {"PASS", "WARNING"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
