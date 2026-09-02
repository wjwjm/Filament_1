#!/usr/bin/env python3
"""Deterministic HR-4E-2 spatial and temporal convergence summaries."""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.hr4e_spatial import (
    E2_CENTROID_TOLERANCE_M, E2_EXTREME_RELATIVE_TOLERANCE,
    E2_M0_RELATIVE_TOLERANCE, E2_WIDTH_RELATIVE_TOLERANCE,
)
from KHz_filament.hr4e_timestep import json_safe

OBSERVABLES = ("xc_m", "yc_m", "sigma_x_m", "sigma_y_m", "min_delta_n", "max_abs_vx_m_s", "max_abs_vy_m_s", "max_abs_v_m_s", "M0_negative_index_m2")
WIDTHS = {"sigma_x_m", "sigma_y_m"}
EXTREMES = {"min_delta_n", "max_abs_vx_m_s", "max_abs_vy_m_s", "max_abs_v_m_s"}
GRID_KEYS = {"Nx", "Ny", "dx_m", "dy_m"}


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def load_case(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot(case: Mapping[str, Any], time_us: float) -> Mapping[str, Any] | None:
    for item in case.get("snapshots", []):
        if math.isclose(float(item.get("time_us", float("nan"))), time_us, rel_tol=0.0, abs_tol=1.0e-8):
            return item
    return None


def _rel(value: float, reference: float) -> float:
    if reference == 0.0:
        return 0.0 if value == 0.0 else float("inf")
    return abs(value - reference) / abs(reference)


def _tolerance(observable: str) -> tuple[str, float]:
    if observable in {"xc_m", "yc_m"}:
        return "absolute", E2_CENTROID_TOLERANCE_M
    if observable in WIDTHS or observable == "M0_negative_index_m2":
        return "relative", E2_WIDTH_RELATIVE_TOLERANCE if observable in WIDTHS else E2_M0_RELATIVE_TOLERANCE
    return "relative", E2_EXTREME_RELATIVE_TOLERANCE


def _geometry_guard(cases: Sequence[Mapping[str, Any]], *, allow_dt: bool = False) -> dict[str, Any]:
    if len(cases) < 2:
        return {"pass": False, "reason": "need at least two cases"}
    ref = cases[0]["configuration"]
    problems: list[str] = []
    for case in cases[1:]:
        config = case.get("configuration", {})
        for key in ("family", "operator", "snapshot_times_s", "initial_state"):
            if config.get(key) != ref.get(key):
                problems.append(f"{case.get('case_id')}: {key} drift")
        if not allow_dt and config.get("dt_hydro_s") != ref.get("dt_hydro_s"):
            problems.append(f"{case.get('case_id')}: dt drift")
        left, right = ref.get("grid", {}), config.get("grid", {})
        for key in set(left) | set(right):
            if key in GRID_KEYS:
                continue
            if left.get(key) != right.get(key):
                problems.append(f"{case.get('case_id')}: grid.{key} drift")
    return {"pass": not problems, "problems": problems}


def spatial_report(cases: Sequence[Mapping[str, Any]], *, horizons_us: Sequence[float] = (100.0, 1000.0)) -> dict[str, Any]:
    by_spacing = {float(case["configuration"]["grid"]["dx_m"]): case for case in cases}
    required = (20e-6, 10e-6, 5e-6)
    if any(item not in by_spacing for item in required):
        return {"status": "NOT_RUN", "classification": "D", "reason": "missing 20/10/5 um case"}
    selected = [by_spacing[item] for item in required]
    guard = _geometry_guard(selected)
    rows: list[dict[str, Any]] = []
    for horizon in horizons_us:
        snaps = [_snapshot(case, horizon) for case in selected]
        contaminated = any(not snap or snap.get("boundary_contaminated", False) for snap in snaps)
        for obs in OBSERVABLES:
            values = [None if snap is None else snap.get(obs) for snap in snaps]
            try:
                q20, q10, q5 = [_finite(item, obs) for item in values]
            except (TypeError, ValueError):
                q20 = q10 = q5 = None
            d20_10 = None if q20 is None else abs(q20 - q10)
            d10_5 = None if q20 is None else abs(q10 - q5)
            kind, tolerance = _tolerance(obs)
            error10_5 = None if q20 is None else (d10_5 if kind == "absolute" else _rel(q10, q5))
            pass10_5 = error10_5 is not None and error10_5 <= tolerance and not contaminated
            trend = d20_10 is not None and d10_5 is not None and (d10_5 < d20_10 or (d20_10 == 0.0 and d10_5 == 0.0))
            p_obs = math.log2(d20_10 / d10_5) if d20_10 and d10_5 and d20_10 > 0.0 and d10_5 > 0.0 else None
            rows.append({"horizon_us": horizon, "observable": obs, "Q_20um": q20, "Q_10um": q10, "Q_5um": q5, "D20_10": d20_10, "D10_5": d10_5, "p_obs": p_obs, "10_vs_5_value": error10_5, "10_vs_5_tolerance": tolerance, "10_vs_5_pass": pass10_5, "trend_D10_5_lt_D20_10": trend, "boundary_contaminated": contaminated})
    relevant = [row for row in rows if not row["boundary_contaminated"]]
    all_cases_pass = all(case.get("status") == "PASS" and case.get("stability", {}).get("overall_pass") for case in selected)
    accepted = bool(guard.get("pass") and relevant and all_cases_pass and all(row["10_vs_5_pass"] for row in relevant) and all(row["trend_D10_5_lt_D20_10"] for row in relevant))
    return {"schema": "khz_filament.hr4e2.spatial_report.v1", "status": "PASS" if accepted else "FAIL", "classification": "A" if accepted else "D", "config_guard": guard, "rows": rows, "case_ids": [case.get("case_id") for case in selected]}


def temporal_guard(coarse: Mapping[str, Any], fine: Mapping[str, Any], spatial: Mapping[str, Any], *, horizon_us: float = 100.0) -> dict[str, Any]:
    guard = _geometry_guard([coarse, fine], allow_dt=True)
    c_snap, f_snap = _snapshot(coarse, horizon_us), _snapshot(fine, horizon_us)
    spatial_rows = {row["observable"]: row for row in spatial.get("rows", []) if row["horizon_us"] == horizon_us}
    rows = []
    for obs in OBSERVABLES:
        c = _finite(c_snap[obs], obs) if c_snap else None
        f = _finite(f_snap[obs], obs) if f_snap else None
        d_time = None if c is None or f is None else abs(c - f)
        d_space = spatial_rows.get(obs, {}).get("D10_5")
        kind, tolerance = _tolerance(obs)
        if d_time is None:
            passed, rule = False, "missing"
        elif d_space is not None and d_space > 0.0:
            passed, rule = d_time <= 0.25 * d_space, "ratio"
        else:
            passed, rule = (d_time <= tolerance if kind == "absolute" else _rel(c, f) <= tolerance), "absolute_or_relative"
        rows.append({"observable": obs, "D_time_5um": d_time, "D_space_10_5": d_space, "rule": rule, "pass": passed})
    return {"schema": "khz_filament.hr4e2.temporal_guard.v1", "status": "PASS" if guard.get("pass") and all(row["pass"] for row in rows) else "FAIL", "config_guard": guard, "rows": rows}


def advection_report(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(cases, key=lambda case: float(case["configuration"]["grid"]["dx_m"]), reverse=True)
    guard = _geometry_guard(ordered)
    metrics = ("centroid_error_x_m", "centroid_error_y_m", "sigma_x_growth_m", "sigma_y_growth_m", "peak_amplitude_loss", "L1_field_error_m2", "L2_field_error_m", "effective_artificial_diffusion_x_m2_s", "effective_artificial_diffusion_y_m2_s")
    rows = []
    for metric in metrics:
        values = [float(case.get("advection_exact", {}).get(metric, float("nan"))) for case in ordered]
        monotonic = all(math.isfinite(item) for item in values) and values[2] < values[1] < values[0]
        rows.append({"metric": metric, "20um": values[0], "10um": values[1], "5um": values[2], "monotonic_refinement": monotonic})
    return {"schema": "khz_filament.hr4e2.advection_report.v1", "status": "PASS" if guard.get("pass") and all(row["monotonic_refinement"] for row in rows) else "FAIL", "config_guard": guard, "rows": rows}


def write_report(report: Mapping[str, Any], out_path: Path) -> None:
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(json_safe(dict(report)), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("spatial", "temporal", "advection"), required=True)
    parser.add_argument("--case", action="append", type=Path, required=True)
    parser.add_argument("--spatial-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    cases = [load_case(path) for path in args.case]
    if args.mode == "spatial":
        report = spatial_report(cases)
    elif args.mode == "advection":
        report = advection_report(cases)
    else:
        if len(cases) != 2 or args.spatial_report is None:
            parser.error("temporal mode requires exactly two --case and --spatial-report")
        report = temporal_guard(cases[0], cases[1], load_case(args.spatial_report))
    write_report(report, args.out)
    print(json.dumps({"status": report["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
