#!/usr/bin/env python3
"""Create non-overwriting E2-C mapping-versus-evolution adjudication reports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.hr4e2c_adjudication import adjudicate
from KHz_filament.hr4e_timestep import json_safe


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(json_safe(dict(value)), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _markdown(report: Mapping[str, Any]) -> str:
    lines = ["# HR-4E-2C Mapping-vs-Evolution Adjudication", ""]
    lines += [f"- Status: `{report['status']}`", f"- Decision: `{report['decision']}`", ""]
    for screen in report["screens"]:
        identity = screen["screen_identity"]
        lines += [f"## {identity['screen_id']}", "", "| Observable | Initial status | Evolution trend | Category | Fine tolerance fraction |", "|---|---|---|---|---:|"]
        for row in screen["rows"]:
            lines.append(f"| {row['observable']} | {row['initial_status']} | {row['evolution_trend_status']} | {row['adjudication_category']} | {row['fraction_of_tolerance_final']:.6g} |")
        lines.append("")
    lines += ["## Scope", "", "This is a report-only adjudication using stored t=0 and 100 us scalar snapshots. It does not alter HR-4 physics, raw E2-C artifacts, the validation adapter, HR-5, or HR-4F.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, action="append", required=True)
    parser.add_argument("--e2a", type=Path, required=True)
    parser.add_argument("--e2b", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    report = adjudicate([_load(path) for path in args.case], e2a=_load(args.e2a), e2b=_load(args.e2b))
    report["input_case_manifests"] = [str(path) for path in args.case]
    report["analysis_only"] = True
    report["raw_fields_downloaded"] = False
    report["gpu_jobs_submitted"] = False
    screens = report["screens"]
    _write(args.out_dir / "e2c_t0_mapping_observables.json", {"schema": "khz_filament.hr4e2c.t0_observables.v1", "screens": [{"screen_identity": screen["screen_identity"], "case_ids": screen["case_ids"], "rows": [{key: row[key] for key in ("observable", "Q_20um_0", "Q_10um_0", "Q_5um_0", "tolerance_kind", "frozen_10_vs_5_tolerance", "initial_status")} for row in screen["rows"]]} for screen in screens]})
    _write(args.out_dir / "e2c_mapping_difference_report.json", {"schema": "khz_filament.hr4e2c.mapping_difference.v1", "screens": [{"screen_identity": screen["screen_identity"], "rows": [{key: row[key] for key in ("observable", "D20_10_init", "D10_5_init", "D20_10_init_metric", "D10_5_init_metric", "p_init", "initial_status", "fraction_of_tolerance_init")} for row in screen["rows"]]} for screen in screens]})
    _write(args.out_dir / "e2c_evolution_increment_report.json", {"schema": "khz_filament.hr4e2c.evolution_increment.v1", "screens": [{"screen_identity": screen["screen_identity"], "rows": [{key: row[key] for key in ("observable", "Delta_Q20", "Delta_Q10", "Delta_Q5", "D20_10_evol", "D10_5_evol", "D20_10_evol_metric", "D10_5_evol_metric", "p_evol", "evolution_trend_status", "fraction_of_tolerance_evol")} for row in screen["rows"]]} for screen in screens]})
    _write(args.out_dir / "e2c_mapping_vs_evolution_adjudication.json", report)
    _write(args.out_dir / "hr4e2_final_decision_adjudicated.json", {key: report[key] for key in ("schema", "status", "decision", "e2a_status", "e2b_status", "hard_10_vs_5_tolerances_pass", "configuration_and_boundary_guards_pass", "has_material_hydro_or_mixed_warning", "near_limit", "production_candidate_statement", "scope_is_hydro_only_validation", "full_chain_transverse_convergence_claimed", "production_multigrid_mapping_modified", "validation_only_statement", "execution_sha_note", "input_case_manifests", "analysis_only", "raw_fields_downloaded", "gpu_jobs_submitted")})
    (args.out_dir / "e2c_mapping_vs_evolution_adjudication.md").write_text(_markdown(report), encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "decision": report["decision"], "out_dir": str(args.out_dir)}, sort_keys=True))
    return 0 if report["status"] in {"PASS", "WARNING"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
