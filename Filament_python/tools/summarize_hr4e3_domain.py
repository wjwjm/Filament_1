#!/usr/bin/env python3
"""Summarize non-overwriting HR-4E-3 domain/boundary evidence."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.hr4e_domain import common_d0_field_metrics
from KHz_filament.hr4e_spatial import E2_CENTROID_TOLERANCE_M, E2_EXTREME_RELATIVE_TOLERANCE, E2_M0_RELATIVE_TOLERANCE, E2_WIDTH_RELATIVE_TOLERANCE
from KHz_filament.hr4e_timestep import json_safe

OBSERVABLES = ("xc_m", "yc_m", "sigma_x_m", "sigma_y_m", "min_delta_n", "max_abs_vx_m_s", "max_abs_vy_m_s", "max_abs_v_m_s", "M0_negative_index_m2")


def _snapshot(case: Mapping[str, Any], time_us: float) -> Mapping[str, Any]:
    matches = [item for item in case.get("snapshots", []) if math.isclose(float(item["time_us"]), time_us, rel_tol=0.0, abs_tol=1e-8)]
    if len(matches) != 1:
        raise ValueError(f"case {case.get('case_id')} lacks unique {time_us} us snapshot")
    return matches[0]


def _limit(name: str) -> tuple[str, float]:
    if name in {"xc_m", "yc_m"}:
        return "absolute", E2_CENTROID_TOLERANCE_M
    if name in {"sigma_x_m", "sigma_y_m"}:
        return "relative", E2_WIDTH_RELATIVE_TOLERANCE
    if name == "M0_negative_index_m2":
        return "relative", E2_M0_RELATIVE_TOLERANCE
    return "relative", E2_EXTREME_RELATIVE_TOLERANCE


def _difference(reference: float, candidate: float, name: str) -> dict[str, Any]:
    kind, tolerance = _limit(name)
    absolute = abs(candidate - reference)
    relative = 0.0 if reference == candidate == 0.0 else float("inf") if reference == 0.0 else absolute / abs(reference)
    value = absolute if kind == "absolute" else relative
    return {"observable": name, "D0": reference, "candidate": candidate, "absolute_difference": absolute, "relative_difference": relative, "tolerance_kind": kind, "tolerance": tolerance, "pass": math.isfinite(value) and value <= tolerance}


def _case_key(case: Mapping[str, Any]) -> tuple[str, str]:
    family = str(case["family"])
    screen = "synthetic" if family == "E3-A" else str(case["initial_state"]["source_provenance"]["screen_identity"]["screen_id"])
    return family, screen


def _screen_report(cases: list[Mapping[str, Any]], horizon_us: float) -> dict[str, Any]:
    by_domain = {str(case["configuration"]["grid"]["domain_id"]): case for case in cases}
    if set(by_domain) != {"D0", "D1", "D2", "D3"}:
        return {"status": "E3-E_INVALID_OR_INCONCLUSIVE", "reason": "missing_or_ambiguous_domain_case"}
    stable = all(case["status"] == "PASS" and case["stability"]["overall_pass"] for case in by_domain.values())
    snapshots = {name: _snapshot(case, horizon_us) for name, case in by_domain.items()}
    clean = all(not snap["boundary_contaminated"] for snap in snapshots.values())
    comparisons = {}
    for domain in ("D1", "D2", "D3"):
        rows = [_difference(float(snapshots["D0"][name]), float(snapshots[domain][name]), name) for name in OBSERVABLES]
        fields = common_d0_field_metrics(snapshots["D0"]["checkpoint"]["path"], snapshots[domain]["checkpoint"]["path"])
        comparisons[domain] = {"scalar_rows": rows, "scalar_pass": all(row["pass"] for row in rows), "field_metrics": fields}
    return {"status": "PASS" if stable and clean and all(item["scalar_pass"] for item in comparisons.values()) else "FAIL", "stable": stable, "boundary_clean": clean, "horizon_us": horizon_us, "comparisons": comparisons, "d0_clearance": snapshots["D0"]["clearance_m"], "d0_clearance_in_sigma": snapshots["D0"]["clearance_in_sigma"], "d0_edge": snapshots["D0"]["field_edge_metrics"]}


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def classify_final(scalar_eligible: bool) -> dict[str, Any]:
    return {"schema": "khz_filament.hr4e3.final_decision.v1", "automatic_scalar_boundary_eligibility": scalar_eligible, "final_class": "PENDING_MANUAL_FIELD_REVIEW" if scalar_eligible else "E3-E_INVALID_OR_INCONCLUSIVE", "reason": "Field L1/L2/Linf diagnostics are reported without an invented automatic production threshold; final E3-A/B/C/D classification requires review of those persisted fields.", "deferred_e2_velocity_caveat": "OPEN_HISTORICAL_CAVEAT", "no_hr4f_or_hr5_started": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in args.case]
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for case in cases:
        groups.setdefault(_case_key(case), []).append(case)
    reports = {f"{family}_{screen}": {"100us": _screen_report(group, 100.0), "1ms": _screen_report(group, 1000.0)} for (family, screen), group in sorted(groups.items())}
    real = {key: value for key, value in reports.items() if key.startswith("E3-B_")}
    synthetic = reports.get("E3-A_synthetic")
    scalar_eligible = bool(synthetic) and all(value["100us"]["status"] == "PASS" and value["1ms"]["status"] == "PASS" for value in real.values()) and synthetic["100us"]["status"] == "PASS" and synthetic["1ms"]["status"] == "PASS"
    decision = classify_final(scalar_eligible)
    args.out_dir.mkdir(parents=True)
    _write_json(args.out_dir / "e3_scalar_100us.json", {key: value["100us"] for key, value in reports.items()})
    _write_json(args.out_dir / "e3_scalar_1ms.json", {key: value["1ms"] for key, value in reports.items()})
    _write_json(args.out_dir / "e3_common_interior_field_comparison.json", {key: {time: value[time]["comparisons"] if "comparisons" in value[time] else value[time] for time in ("100us", "1ms")} for key, value in reports.items()})
    _write_json(args.out_dir / "e3_boundary_edge_report.json", {key: {time: {"clearance_m": value[time].get("d0_clearance"), "clearance_in_sigma": value[time].get("d0_clearance_in_sigma"), "edge": value[time].get("d0_edge")} for time in ("100us", "1ms")} for key, value in reports.items()})
    _write_json(args.out_dir / "e3_lateral_vertical_sensitivity.json", reports)
    _write_json(args.out_dir / "hr4e3_final_decision.json", decision)
    markdown = "# HR-4E-3 Domain / Boundary Summary\n\nAutomatic scalar/boundary eligibility: `{} `.\n\nFinal class: `{}`.\n\nThe persisted common-D0 full-field metrics are in `e3_common_interior_field_comparison.json`; no automatic field threshold was introduced.\n".format(scalar_eligible, decision["final_class"])
    (args.out_dir / "HR4E3_DOMAIN_SUMMARY.md").write_text(markdown, encoding="utf-8", newline="\n")
    print(json.dumps({"status": decision["final_class"], "out_dir": str(args.out_dir)}, sort_keys=True))
    return 0 if scalar_eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
