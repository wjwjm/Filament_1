#!/usr/bin/env python3
"""Evaluate Stage 1 artifacts and write a non-ranking report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from plot_khzfil_out import FIGURE_SPECS


def evaluate_case_quality(case_summary: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    observations = case_summary.get("quality_observations", {})
    failures: list[str] = []
    warnings = list(case_summary.get("sanity_warnings", []))
    if gates.get("require_strictly_increasing_z") and not observations.get("z_strictly_increasing"):
        failures.append("z_axis is not strictly increasing")
    checks = (("max_energy_growth_fraction", "maximum_energy_growth_fraction"), ("max_adjacent_intensity_growth", "maximum_adjacent_intensity_growth"), ("max_electron_density_m3", "maximum_electron_density_m3"))
    for observed, gate in checks:
        value = observations.get(observed)
        if value is not None and value > gates[gate]: failures.append(f"{observed}={value} exceeds {gates[gate]}")
    if gates.get("require_positive_fwhm") and not observations.get("fwhm_all_positive_finite", False):
        failures.append("FWHM contains non-positive or non-finite values")
    missing = set(FIGURE_SPECS.values()) - set(case_summary.get("generated_figures", []))
    if missing: failures.append(f"required figures missing: {sorted(missing)}")
    return {"technical_pass": not failures, "quality_gate_pass": not failures, "failures": failures, "warnings": warnings}


def build_stage1_report(stage_dir: str | Path) -> dict[str, Any]:
    stage_dir = Path(stage_dir)
    spec = json.loads((stage_dir / "stage_spec_snapshot.json").read_text(encoding="utf-8"))
    manifest = json.loads((stage_dir / "submission_manifest.json").read_text(encoding="utf-8"))
    case_reports: dict[str, Any] = {}
    job_statuses: dict[str, str | None] = {}
    technical_complete = True
    for case in spec["cases"]:
        case_id = case["case_id"]
        summary_path = stage_dir / "cases" / case_id / "figures" / "diagnostic_summary.json"
        metadata_path = stage_dir / "cases" / case_id / "run_metadata.json"
        if not summary_path.is_file() or not metadata_path.is_file():
            technical_complete = False; case_reports[case_id] = {"technical_pass": False, "quality_gate_pass": False, "failures": ["missing case artifacts"], "warnings": []}; continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        job_statuses[case_id] = metadata.get("status")
        case_reports[case_id] = evaluate_case_quality(json.loads(summary_path.read_text(encoding="utf-8")), spec["quality_gates"])
        if metadata.get("status") != "completed":
            technical_complete = False
            case_reports[case_id]["technical_pass"] = False
            case_reports[case_id]["failures"].append(f"case status is {metadata.get('status')!r}, not completed")
    comparison_path = stage_dir / "comparison" / "comparison_summary.json"
    metrics_path = stage_dir / "comparison" / "comparison_metrics.csv"
    if not comparison_path.is_file() or not metrics_path.is_file(): technical_complete = False
    comparison = json.loads(comparison_path.read_text(encoding="utf-8")) if comparison_path.is_file() else {}
    comparison_metrics = list(csv.DictReader(metrics_path.open(encoding="utf-8"))) if metrics_path.is_file() else []
    quality_pass = technical_complete and all(value["quality_gate_pass"] for value in case_reports.values())
    report = {
        "stage_id": spec["stage_id"], "stage_name": spec["stage_name"], "objective": spec["objective"],
        "comparison_mode": spec["comparison_mode"], "fixed_peak_power_W": spec["required_invariants"]["beam.P0_peak"],
        "technical_status": "completed" if technical_complete else "incomplete",
        "quality_gate_status": "passed" if quality_pass else ("failed" if technical_complete else "not_evaluated"),
        "scientific_interpretation_status": "requires_review", "cases": case_reports,
        "case_job_statuses": job_statuses,
        "comparison_summary": "comparison/comparison_summary.json" if comparison_path.is_file() else None,
        "comparison_metrics": comparison_metrics,
        "comparison_figures": comparison.get("generated_figures", []),
        "submission_manifest": manifest,
        "conclusion_limit": "No optimization objective or ranking rule is defined; no case is declared optimal.",
    }
    reports = stage_dir / "reports"; reports.mkdir(parents=True, exist_ok=True)
    (reports / "stage1_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [f"# {spec['display_name']}", "", "## Status", "", f"- Technical: {report['technical_status']}", f"- Quality gates: {report['quality_gate_status']}", "- Scientific interpretation: requires_review", "", "## Fixed comparison", "", "- Same peak power: 17 GW", "- Cases: 40 fs and 120 fs", "- No automatic optimal-case conclusion is produced.", "", "## Case quality"]
    for case_id, evaluation in case_reports.items(): lines.append(f"- {case_id}: {'pass' if evaluation['quality_gate_pass'] else 'fail'}; {', '.join(evaluation['failures']) or 'no gate failure'}")
    if comparison_metrics:
        lines.extend(["", "## Core metrics", "", "| Case | Imax peak | rho on-axis peak | w min | Energy drift (%) |", "| --- | ---: | ---: | ---: | ---: |"])
        for row in comparison_metrics:
            lines.append(f"| {row.get('case_label', row.get('case_id'))} | {row.get('I_max_z_peak', '')} | {row.get('rho_onaxis_max_z_peak', '')} | {row.get('w_mom_min_m', '')} | {row.get('U_drift_pct', '')} |")
    lines.extend(["", "## Comparison outputs", "", *[f"- comparison/{name}" for name in report["comparison_figures"]], "", "## Next human decision", "", "Review filament length, electron-density level, energy loss, and intensity stability against the intended optimization objective. No ranking is performed in Stage 1."])
    (reports / "stage1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--stage-dir", required=True); args = parser.parse_args()
    report = build_stage1_report(args.stage_dir)
    return 0 if report["technical_status"] == "completed" else 1


if __name__ == "__main__": raise SystemExit(main())
