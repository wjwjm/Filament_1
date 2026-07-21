#!/usr/bin/env python3
"""Generate Phase 8B-P gates, report, and changelog from recorded evidence."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from KHz_filament.config_normalize import normalize_config  # noqa: E402


GATE_FIELDS = (
    "status", "evidence", "numerical_result", "threshold", "comparison_operator",
    "physical_impact", "production_impact", "required_action",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _gate(status, evidence, numerical_result, threshold, comparison_operator, physical_impact, production_impact, required_action):
    if status not in {"passed", "failed", "inconclusive", "not_applicable"}:
        raise ValueError(f"invalid gate status: {status}")
    return dict(zip(GATE_FIELDS, (
        status, evidence, numerical_result, threshold, comparison_operator,
        physical_impact, production_impact, required_action,
    )))


def _boolean_gate(checks, *, evidence, threshold, physical_impact, production_impact, required_action):
    if not checks or any(value is None for value in checks.values()):
        status = "inconclusive"
        result = False
    else:
        result = all(bool(value) for value in checks.values())
        status = "passed" if result else "failed"
    return _gate(
        status, evidence, {"checks": checks, "comparison_result": result}, threshold, "all",
        physical_impact, production_impact, required_action,
    )


def _threshold_gate(value, threshold, *, evidence, op="lt", physical_impact, production_impact, required_action):
    if not _finite(value) or not _finite(threshold):
        return _gate("inconclusive", evidence, {"value": value, "comparison_result": False}, threshold, op, physical_impact, production_impact, required_action)
    if op == "lt":
        passed = float(value) < float(threshold)
    elif op == "le":
        passed = float(value) <= float(threshold)
    elif op == "ge":
        passed = float(value) >= float(threshold)
    else:
        raise ValueError(op)
    return _gate("passed" if passed else "failed", evidence, {"value": value, "comparison_result": passed}, threshold, op, physical_impact, production_impact, required_action)


def _raises_value_error(raw: dict) -> bool:
    try:
        normalize_config(raw)
    except ValueError:
        return True
    return False


def build_gates(results_dir: Path, *, full_pytest_passed: bool, full_pytest_summary: str) -> tuple[dict, dict]:
    baseline_diff = _load(results_dir / "baseline_to_full_operator_config_diff.json")
    on_off_diff = _load(results_dir / "full_operator_on_vs_off_config_diff.json")
    combined = _load(results_dir / "combined_operator_summary.json")
    smoke = _load(results_dir / "phase8b_full_size_smoke_metrics.json")
    runtime = _load(results_dir / "phase8b_runtime_estimate.json")
    contract = _load(results_dir / "phase8b_expected_diagnostic_contract.json")
    on_path = ROOT / "configs" / "isaacs_raman_closure" / "120fs_talebpour_isaacs_full_operator_on.json"
    off_path = ROOT / "configs" / "isaacs_raman_closure" / "120fs_talebpour_isaacs_full_operator_feedback_off.json"
    on = _load(on_path)
    off = _load(off_path)

    invalid_phase = deepcopy(on); invalid_phase["propagation"]["use_raman_phase"] = True
    invalid_prop_abs = deepcopy(on); invalid_prop_abs["propagation"]["use_raman_absorption"] = True
    invalid_raman_abs = deepcopy(on); invalid_raman_abs["raman"]["absorption"] = True

    gates = {}
    gates["baseline_config_lock_gate"] = _boolean_gate({
        "diff_status_passed": baseline_diff.get("status") == "passed",
        "unexpected_paths_empty": baseline_diff.get("unexpected_paths") == [],
        "authorized_paths_exact": sorted(item["path"] for item in baseline_diff.get("differences", [])) == sorted(baseline_diff.get("authorized_paths", [])),
    }, evidence="baseline_to_full_operator_config_diff.json", threshold="only authorized Raman paths differ from Phase 6 production baseline", physical_impact="locks all non-Raman production physics", production_impact="prevents accidental grid, ionization, beam, or safety changes", required_action="restore the Phase 6 baseline and regenerate the config diff")
    gates["on_off_single_factor_gate"] = _boolean_gate({
        "diff_status_passed": on_off_diff.get("status") == "passed",
        "only_full_operator_switch": [item.get("path") for item in on_off_diff.get("differences", [])] == ["propagation.use_raman_full_operator"],
        "on_true_off_false": on["propagation"]["use_raman_full_operator"] is True and off["propagation"]["use_raman_full_operator"] is False,
    }, evidence="full_operator_on_vs_off_config_diff.json", threshold="exactly one physical difference: propagation.use_raman_full_operator", physical_impact="isolates full-operator causality", production_impact="makes Job 2 a valid single-factor control", required_action="remove every additional ON/OFF config difference")
    gates["explicit_operator_switch_gate"] = _boolean_gate({
        "on_full_mode": on["raman"]["operator_mode"] == "full_isaacs_eq27",
        "off_full_mode": off["raman"]["operator_mode"] == "full_isaacs_eq27",
        "legacy_phase_false": not on["propagation"]["use_raman_phase"] and not off["propagation"]["use_raman_phase"],
        "on_switch_true": on["propagation"]["use_raman_full_operator"] is True,
        "off_switch_false": off["propagation"]["use_raman_full_operator"] is False,
    }, evidence="two formal Phase 8B configs; test_phase8b_full_operator_switch.py", threshold="full mode uses the explicit full-operator feedback switch", physical_impact="distinguishes complex Eq.27 feedback from legacy Raman phase", production_impact="prevents misleading phase-only causality labels", required_action="repair full-mode switch semantics")
    gates["legacy_absorption_rejection_gate"] = _boolean_gate({
        "formal_configs_absorption_off": not on["propagation"]["use_raman_absorption"] and not off["propagation"]["use_raman_absorption"],
        "full_plus_phase_rejected": _raises_value_error(invalid_phase),
        "full_plus_propagation_absorption_rejected": _raises_value_error(invalid_prop_abs),
        "full_plus_raman_absorption_rejected": _raises_value_error(invalid_raman_abs),
    }, evidence="config_normalize.py; test_phase8b_full_operator_switch.py", threshold="all legacy phase/absorption combinations rejected in full mode", physical_impact="prevents Raman energy double counting", production_impact="legacy alpha_R and conv_deriv remain disabled", required_action="restore strict full-mode validation")

    on_contract = smoke.get("on_contract", {})
    off_contract = smoke.get("off_contract", {})
    gates["full_operator_diagnostic_wiring_gate"] = _boolean_gate({
        "on_completed": on_contract.get("completed"),
        "on_raw_IR": on_contract.get("raw_IR_nonzero"),
        "on_rhs_applied": on_contract.get("applied_rhs_nonzero"),
        "on_actual_loss": on_contract.get("actual_loss_nonzero"),
        "off_completed": off_contract.get("completed"),
        "off_raw_IR": off_contract.get("raw_IR_nonzero"),
        "off_rhs_zero": off_contract.get("applied_rhs_zero"),
        "off_actual_loss_zero": off_contract.get("actual_loss_zero"),
    }, evidence="phase8b_full_size_smoke_metrics.json; test_phase8b_raman_diagnostics.py", threshold="ON applies the operator; OFF preserves raw diagnostics without field feedback", physical_impact="makes raw response and applied complex feedback separately observable", production_impact="supports post-run ON/OFF audit", required_action="repair production diagnostic wiring before a full job")
    on_metrics = smoke["on"]
    raman_relative_difference = abs(float(on_metrics["raman_actual_loss_total_J"])-float(on_metrics["raman_target_loss_total_J"])) / max(abs(float(on_metrics["raman_target_loss_total_J"])), 1e-300)
    gates["raman_energy_accounting_gate"] = _boolean_gate({
        "closure_finite": smoke.get("energy_closure_finite"),
        "target_positive": float(on_metrics["raman_target_loss_total_J"]) > 0.0,
        "actual_positive": float(on_metrics["raman_actual_loss_total_J"]) > 0.0,
        "cumulative_closure_below_stop_limit": float(on_metrics["raman_cumulative_closure_final"]) < 5e-3,
        "integrated_target_actual_difference_below_stop_limit": raman_relative_difference < 5e-3,
        "legacy_alpha_zero": float(on_metrics["legacy_alpha_R_max"]) == 0.0,
    }, evidence="phase8b_full_size_smoke_metrics.json", threshold="finite accounting and cumulative target/actual mismatch <0.5%; legacy alpha_R=0", physical_impact="tracks Eq.10 target against actual Raman field energy exchange", production_impact="blocks propagation on missing or duplicated Raman energy", required_action="repair Raman field-energy accounting")
    gates["convolution_reuse_gate"] = _boolean_gate({
        "on_two_per_operator_substep": on_contract.get("two_convolutions_per_operator_substep"),
        "on_two_strang_substeps": on_contract.get("two_strang_substeps_per_z_step"),
        "off_one_raw_diagnostic_convolution": off_contract.get("one_raw_diagnostic_convolution_per_z_step"),
    }, evidence="phase8b_full_size_smoke_metrics.json", threshold="2 convolutions per Heun operator application; no duplicate Eq.10 convolution", physical_impact="preserves stage-consistent I_R and Eq.10 diagnostics", production_impact="controls full-job runtime and duplicate work", required_action="reuse each stage response for RHS and energy diagnostics")

    refined_order = float(combined["refined_estimated_order"])
    gates["combined_split_convergence_gate"] = _threshold_gate(refined_order, 1.5, evidence="combined_operator_dz_convergence.csv; combined_operator_summary.json", op="ge", physical_impact="verifies convergence of Raman/non-Raman Strang composition", production_impact="selects the admissible combined nonlinear ordering", required_action="repair or refine the combined nonlinear split")
    production = combined["production_vs_dz2"]
    gates["combined_split_production_step_gate"] = _boolean_gate({
        "field_l2_lt_0.1pct": float(production["field_l2_difference"]) < 1e-3,
        "I_max_lt_0.2pct": float(production["I_max_relative_difference"]) < 2e-3,
        "rho_max_lt_0.5pct": float(production["rho_max_relative_difference"]) < 5e-3,
        "raman_loss_lt_0.5pct": float(production["raman_loss_relative_difference"]) < 5e-3,
        "formal_config_uses_strang": on["raman"]["nonlinear_split_order"] == "strang",
    }, evidence="combined_operator_observable_comparison.csv; combined_operator_summary.json", threshold="production dz vs two dz/2 references within field/I/rho/Raman limits", physical_impact="bounds operator-order bias at production dz", production_impact="locks nonlinear_split_order=strang", required_action="reduce dz or repair the combined split")

    gates["full_size_smoke_gate"] = _boolean_gate({
        "recorded_gate": smoke["gates"].get("full_size_smoke_gate"),
        "on_state_completed": smoke["slurm"].get("on_state") == "COMPLETED 0:0",
        "off_state_completed": smoke["slurm"].get("off_state") == "COMPLETED 0:0",
        "on_off_finite": bool(on_metrics["finite"]) and bool(smoke["off"]["finite"]),
        "no_full_production_jobs": smoke["execution_scope"].get("full_production_slurm_jobs_submitted") == 0,
    }, evidence="phase8b_full_size_smoke_metrics.json; smoke_evidence/", threshold="serial 20-step ON/OFF full-grid smokes complete without NaN/Inf", physical_impact="tests the real full grid and production call chain", production_impact="blocks full submission on OOM or numerical failure", required_action="resolve the full-grid smoke failure")
    gates["memory_gate"] = _threshold_gate(float(smoke["peak_reserved_fraction"]), 0.85, evidence="phase8b_full_size_smoke_metrics.json", physical_impact="keeps GPU allocation below the safety ceiling", production_impact="reduces full-job OOM risk", required_action="optimize memory reuse without reducing the production grid")
    runtime_ok = float(runtime["estimated_fraction_of_time_limit"]) < 0.8 and float(runtime["full_operator_slowdown_vs_legacy"]) <= 3.0
    gates["runtime_gate"] = _gate("passed" if runtime_ok else "failed", "phase8b_runtime_estimate.json", {"estimated_fraction_of_time_limit": runtime["estimated_fraction_of_time_limit"], "slowdown_vs_legacy": runtime["full_operator_slowdown_vs_legacy"], "comparison_result": runtime_ok}, "time-limit fraction<0.8 and slowdown<=3.0", "all", "ensures the full run fits its allocation", "prevents submitting a job likely to time out", "optimize convolution reuse or request a justified time limit")
    gates["expected_diagnostic_contract_gate"] = _boolean_gate({
        "schema": contract.get("schema") == "khz_filament.phase8b.expected_diagnostic_contract.v1",
        "nominal_records": contract["record_axis"].get("nominal_record_count") == 15000,
        "strict_z": contract["record_axis"].get("strictly_increasing") is True and contract["record_axis"].get("duplicates_allowed") is False,
        "z_final": contract["fixed_coordinates"].get("z_final_m") == 1.3,
        "focus_fixed": contract["fixed_coordinates"].get("vacuum_focus_m") == 0.95,
        "raman_p99": contract["raman_energy_contract"].get("per_step_p99_lt") == 1e-3,
        "raman_cumulative": contract["raman_energy_contract"].get("cumulative_final_lt") == 5e-3,
        "total_energy_final": contract["total_energy_contract"].get("final_lt") == 1e-2,
        "total_energy_focus": contract["total_energy_contract"].get("near_focus_max_lt") == 2e-2,
    }, evidence="phase8b_expected_diagnostic_contract.json; test_phase8b_diagnostic_contract.py", threshold="all required production diagnostics, units, coordinates, and energy thresholds recorded", physical_impact="defines acceptance of completed full propagation", production_impact="drives Job 1/Job 2 post-run audits", required_action="repair and regenerate the diagnostic contract")

    required_names = [name for name in gates]
    required_statuses = {name: gates[name]["status"] for name in required_names}
    admission = all(status == "passed" for status in required_statuses.values()) and bool(full_pytest_passed)
    gates["full_job_submission_gate"] = _gate(
        "passed" if admission else "failed",
        "aggregate Phase 8B-P gates and complete local pytest",
        {"required_gates": required_statuses, "local_full_pytest_passed": bool(full_pytest_passed), "local_full_pytest_summary": full_pytest_summary, "comparison_result": admission},
        "all preflight gates passed and complete local pytest has zero failures",
        "all",
        "controls whether full 1.3 m propagation may be separately authorized",
        "authorizes preparation of Phase 8B-R only after a new user approval",
        "resolve every failed or inconclusive gate; then request explicit Phase 8B-R approval",
    )
    meta = {
        "selected_nonlinear_split": on["raman"]["nonlinear_split_order"],
        "full_operator_mode": on["raman"]["operator_mode"],
        "short_smoke_job_ids": [smoke["slurm"]["on_job_id"], smoke["slurm"]["off_job_id"]],
        "full_production_jobs_submitted": 0,
        "production_propagation_executed": False,
        "github_actions_ci_evidence": "unavailable",
        "phase8b_r_executed": False,
    }
    return gates, meta


def _report(gates: dict, meta: dict, results_dir: Path) -> str:
    smoke = _load(results_dir / "phase8b_full_size_smoke_metrics.json")
    runtime = _load(results_dir / "phase8b_runtime_estimate.json")
    combined = _load(results_dir / "combined_operator_summary.json")
    admission = gates["full_job_submission_gate"]["status"]
    lines = [
        "# Phase 8B-P controlled-propagation preflight report", "",
        "## Decision", "",
        f"- `full_job_submission_gate`: **{admission}**",
        "- Phase 8B-R executed: **false**",
        "- Full 1.3 m Slurm jobs submitted: **0**",
        "- GitHub Actions CI evidence: **unavailable**",
        "- Required next action: merge this preflight, then obtain explicit user approval before preparing Job 1.", "",
        "## Full-grid smoke evidence", "",
        f"- Short smoke Job IDs: {', '.join(meta['short_smoke_job_ids'])} (both `COMPLETED 0:0`).",
        f"- GPU: {smoke['gpu_type']}; grid: 512x512x384; 20 z steps per case.",
        f"- Peak reserved GPU memory: {100*smoke['peak_reserved_fraction']:.3f}% (threshold <85%).",
        f"- Mean ON step time: {runtime['mean_step_walltime_s']:.6f} s; projected 15000-step runtime: {runtime['estimated_15000_step_walltime_h']:.3f} h.",
        f"- Requested 8 h fraction: {100*runtime['estimated_fraction_of_time_limit']:.3f}%; slowdown vs legacy: {runtime['full_operator_slowdown_vs_legacy']:.3f}x.",
        f"- ON convolution count: {smoke['on']['convolution_count_per_operator_substep']:.0f} per Heun application and {smoke['on']['convolution_count_per_z_step']:.0f} per Strang z step.",
        f"- OFF raw diagnostic convolution count: {smoke['off']['convolution_count_per_z_step']:.0f} per z step.",
        f"- ON Raman cumulative closure residual: {smoke['on']['raman_cumulative_closure_final']:.6g}.", "",
        "## Combined nonlinear split", "",
        f"- Selected order: `{meta['selected_nonlinear_split']}`.",
        f"- Refined estimated order: {combined['refined_estimated_order']:.6f} (threshold >=1.5).",
        f"- Production dz vs dz/2 field L2 difference: {combined['production_vs_dz2']['field_l2_difference']:.6g}.", "",
        "## Gate summary", "",
        "| Gate | Status |", "|---|---|",
    ]
    lines.extend(f"| `{name}` | {gate['status']} |" for name, gate in gates.items())
    lines.extend(["", "The completed full jobs, if later authorized, must still satisfy the stricter diagnostic contract, including Raman per-step closure p99 <1e-3, cumulative closure <5e-3, and total-energy closure limits.", ""])
    return "\n".join(lines)


def _changelog() -> str:
    return """# Phase 8B-P changelog

- Task P1: copied the Phase 6 production baseline, introduced the explicit full-operator switch, and generated single-factor config diffs.
- Task P2: wired full-operator Raman diagnostics and reused each Heun stage convolution for RHS and Eq.10 energy accounting.
- Task P3: implemented opt-in nonlinear split ordering, validated Strang composition, and selected `strang` for the formal configs.
- Task P4: added performance instrumentation and ran two strictly serial 20-step full-grid Slurm smokes; no full propagation was run.
- Task P5: defined the machine-readable production diagnostic and energy contract plus the completed-run auditor.
- Task P6: regenerated all preflight gates and reports and required the complete local pytest result for full-job submission admission.

No production Raman parameters, non-Raman physics, PyCAP data, or Phase 5-8A.1 historical results were changed. No raw NPZ/MAT/LUT file was committed.
"""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "isaacs_raman_closure" / "phase8b_preflight")
    parser.add_argument("--full-pytest-status", choices=("passed", "failed"), required=True)
    parser.add_argument("--full-pytest-summary", required=True)
    args = parser.parse_args(argv)
    gates, meta = build_gates(args.results_dir, full_pytest_passed=args.full_pytest_status == "passed", full_pytest_summary=args.full_pytest_summary)
    (args.results_dir / "phase8b_preflight_gate_summary.json").write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8")
    (args.results_dir / "phase8b_preflight_report.md").write_text(_report(gates, meta, args.results_dir), encoding="utf-8")
    (args.results_dir / "phase8b_preflight_changelog.md").write_text(_changelog(), encoding="utf-8")


if __name__ == "__main__":
    main()
