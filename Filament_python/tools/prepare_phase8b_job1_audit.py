#!/usr/bin/env python3
"""Prepare Phase 8B-R Task R1 inputs without submitting the full Job 1."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from KHz_filament.config_normalize import normalize_config  # noqa: E402
from tools.prepare_phase8b_preflight_configs import (  # noqa: E402
    AUTHORIZED_BASELINE_DIFFS,
    diff_records,
)


BASELINE = ROOT / "configs" / "ionization_model_propagation" / "120fs_talebpour_full_model.json"
JOB1_CONFIG = ROOT / "configs" / "isaacs_raman_closure" / "120fs_talebpour_isaacs_full_operator_on.json"
PREFLIGHT = ROOT / "results" / "isaacs_raman_closure" / "phase8b_preflight"
OUT_DIR = ROOT / "results" / "isaacs_raman_closure" / "phase8b_controlled_propagation"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT.parent, text=True, encoding="utf-8"
    ).strip()


def build_artifacts(
    *, baseline_path: Path, config_path: Path, preflight_dir: Path,
    out_dir: Path, generated_utc: str | None = None,
) -> dict[str, dict]:
    baseline = _load(baseline_path)
    job1 = _load(config_path)
    normalized = normalize_config(job1)
    smoke = _load(preflight_dir / "phase8b_full_size_smoke_metrics.json")
    gates = _load(preflight_dir / "phase8b_preflight_gate_summary.json")
    contract = _load(preflight_dir / "phase8b_expected_diagnostic_contract.json")
    differences = diff_records(baseline, job1)
    difference_paths = {row["path"] for row in differences}
    unexpected = sorted(difference_paths - AUTHORIZED_BASELINE_DIFFS)
    missing_authorized = sorted(AUTHORIZED_BASELINE_DIFFS - difference_paths)

    config_diff = {
        "schema": "khz_filament.phase8b_r.job1_config_diff.v1",
        "baseline": _repo_path(baseline_path),
        "job1_config": _repo_path(config_path),
        "differences": differences,
        "authorized_paths": sorted(AUTHORIZED_BASELINE_DIFFS),
        "unexpected_paths": unexpected,
        "missing_authorized_paths": missing_authorized,
        "status": "passed" if not unexpected and not missing_authorized else "failed",
    }

    on = smoke["on"]
    energy_contract = contract["raman_energy_contract"]
    numerical_checks = {
        "preflight_full_job_submission_gate_passed": (
            gates["full_job_submission_gate"]["status"] == "passed"
        ),
        "job1_config_diff_passed": config_diff["status"] == "passed",
        "strict_full_model": normalized["raman"]["operator_mode"] == "full_isaacs_eq27",
        "eq27_convention": normalized["raman"]["operator_convention"] == "isaacs_eq27",
        "heun_integrator": normalized["raman"]["operator_integrator"] == "heun",
        "strang_split": normalized["raman"]["nonlinear_split_order"] == "strang",
        "full_operator_enabled": normalized["propagation"]["use_raman_full_operator"] is True,
        "legacy_phase_disabled": normalized["propagation"]["use_raman_phase"] is False,
        "propagation_absorption_disabled": normalized["propagation"]["use_raman_absorption"] is False,
        "raman_absorption_disabled": normalized["raman"]["absorption"] is False,
        "smoke_completed": smoke["slurm"]["on_state"] == "COMPLETED 0:0",
        "smoke_p99_below_contract": (
            float(on["raman_step_closure_p99"])
            < float(energy_contract["per_step_p99_lt"])
        ),
        "smoke_cumulative_below_contract": (
            float(on["raman_cumulative_closure_final"])
            < float(energy_contract["cumulative_final_lt"])
        ),
        "legacy_alpha_exactly_zero": float(on["legacy_alpha_R_max"]) == 0.0,
        "fixed_raman_parameters": (
            float(normalized["beam"]["n2_air"]) == 7.8e-24
            and float(normalized["raman"]["n_R"]) == 2.3e-23
            and float(normalized["raman"]["omega_R"]) == 1.6e13
            and float(normalized["raman"]["Gamma_R"]) == 1.3e13
        ),
        "fixed_production_beam": (
            float(normalized["beam"]["lam0"]) == 800e-9
            and float(normalized["beam"]["P0_peak"]) == 17e9
            and float(normalized["beam"]["focal_length"]) == 0.95
            and float(normalized["beam"]["w0"]) == 1.979e-3
            and float(normalized["beam"]["tau_fwhm"]) == 120e-15
            and float(normalized["propagation"]["z_max"]) == 1.3
        ),
        "no_raw_large_result_in_r1_audit": not any(out_dir.rglob("*.npz")),
    }
    input_audit = {
        "schema": "khz_filament.phase8b_r.job1_input_audit.v1",
        "status": "passed" if all(numerical_checks.values()) else "failed",
        "checks": numerical_checks,
        "metrics": {
            "corrected_on_smoke_job_id": str(smoke["slurm"]["on_job_id"]),
            "raman_step_closure_p99": float(on["raman_step_closure_p99"]),
            "raman_step_closure_p99_threshold": float(energy_contract["per_step_p99_lt"]),
            "raman_cumulative_closure_final": float(on["raman_cumulative_closure_final"]),
            "raman_cumulative_closure_threshold": float(energy_contract["cumulative_final_lt"]),
            "legacy_alpha_R_max": float(on["legacy_alpha_R_max"]),
            "energy_projection_initial_residual_p99": float(
                on.get("energy_projection_initial_residual_p99", 0.0)
            ),
            "energy_projection_scale_deviation_max": float(
                on.get("energy_projection_scale_deviation_max", 0.0)
            ),
        },
        "inputs": {
            "baseline": {"path": _repo_path(baseline_path), "sha256": _sha256(baseline_path)},
            "job1_config": {"path": _repo_path(config_path), "sha256": _sha256(config_path)},
            "preflight_gate_summary": {
                "path": _repo_path(preflight_dir / "phase8b_preflight_gate_summary.json"),
                "sha256": _sha256(preflight_dir / "phase8b_preflight_gate_summary.json"),
            },
            "diagnostic_contract": {
                "path": _repo_path(preflight_dir / "phase8b_expected_diagnostic_contract.json"),
                "sha256": _sha256(preflight_dir / "phase8b_expected_diagnostic_contract.json"),
            },
        },
        "authorization": {
            "phase8b_r_task": "R1 only",
            "full_job_submission_authorized": False,
            "phase8b_r_task_r2_authorized": False,
        },
    }

    generated = generated_utc or datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema": "khz_filament.phase8b_r.job1_execution_manifest.v1",
        "generated_utc": generated,
        "case": "120fs_talebpour_isaacs_full_operator_on",
        "prepared_from_head_sha": _git("rev-parse", "HEAD"),
        "prepared_on_branch": _git("branch", "--show-current"),
        "future_execution_git_sha": None,
        "future_execution_git_sha_rule": (
            "must equal the merged Phase 8B-R Task R1 main SHA at a separately authorized R2"
        ),
        "config": {"path": _repo_path(config_path), "sha256": _sha256(config_path)},
        "input_audit_status": input_audit["status"],
        "full_job_submitted": False,
        "production_propagation_executed": False,
        "slurm_job_id": None,
        "submission_order": None,
        "authorization_scope": "configuration preparation, input audit, and admission correction only",
        "next_authority_required": "explicit user approval for Phase 8B-R Task R2",
    }

    job1_contract = {
        "schema": "khz_filament.phase8b_r.job1_expected_diagnostic_contract.v1",
        "source_contract": {
            "path": _repo_path(preflight_dir / "phase8b_expected_diagnostic_contract.json"),
            "sha256": _sha256(preflight_dir / "phase8b_expected_diagnostic_contract.json"),
        },
        "source_config": {"path": _repo_path(config_path), "sha256": _sha256(config_path)},
        "fixed_coordinates": contract["fixed_coordinates"],
        "record_axis": contract["record_axis"],
        "required_fields": contract["required_fields"],
        "units": contract["units"],
        "common_invariants": contract["common_invariants"],
        "job1_full_operator_on": contract["job1_full_operator_on"],
        "raman_energy_contract": contract["raman_energy_contract"],
        "total_energy_contract": contract["total_energy_contract"],
        "execution_policy": {
            **contract["submission_policy"],
            "full_job_submission_authorized_now": False,
            "phase8b_r_task_r2_user_approval_required": True,
        },
    }

    correction = {
        "schema": "khz_filament.phase8b_r.job1_preflight_correction_audit.v1",
        "status": "passed" if all(numerical_checks.values()) else "failed",
        "false_positive_identified": {
            "original_on_smoke_job_id": "179288",
            "original_raman_step_closure_p99": 0.01772617394104599,
            "original_gate_omitted_p99": True,
        },
        "investigation": {
            "stable_difference_smoke_job_id": "179619",
            "stable_difference_raman_step_closure_p99": 0.004451706493273376,
            "corrected_smoke_job_id": str(smoke["slurm"]["on_job_id"]),
            "corrected_raman_step_closure_p99": float(on["raman_step_closure_p99"]),
            "corrected_raman_cumulative_closure_final": float(
                on["raman_cumulative_closure_final"]
            ),
            "legacy_alpha_R_max": float(on["legacy_alpha_R_max"]),
            "projection_scale_deviation_max": float(
                on.get("energy_projection_scale_deviation_max", 0.0)
            ),
        },
        "gate_correction": {
            "raman_energy_accounting_gate_requires_p99": True,
            "full_job_submission_gate_transitively_requires_p99": True,
            "p99_threshold": float(energy_contract["per_step_p99_lt"]),
            "cumulative_threshold": float(energy_contract["cumulative_final_lt"]),
        },
        "configuration_summary_correction": {
            "raman.absorption": False,
            "propagation.use_raman_absorption": False,
            "effective_absorption": "off",
            "legacy_alpha_R_max": float(on["legacy_alpha_R_max"]),
        },
        "execution_scope": {
            "short_smoke_jobs_used_as_evidence": ["179288", "179619", str(smoke["slurm"]["on_job_id"])],
            "new_full_slurm_jobs_submitted": 0,
            "production_propagation_executed": False,
            "phase8b_r_task_r2_executed": False,
        },
    }
    return {
        "job1_config_diff.json": config_diff,
        "job1_execution_manifest.json": manifest,
        "job1_input_audit.json": input_audit,
        "job1_expected_diagnostic_contract.json": job1_contract,
        "job1_preflight_correction_audit.json": correction,
    }


def _correction_report(payload: dict) -> str:
    investigation = payload["investigation"]
    return "\n".join([
        "# Phase 8B-R Task R1 preflight correction audit",
        "",
        f"- Status: **{payload['status']}**",
        "- Full 1.3 m jobs submitted: **0**",
        "- Phase 8B-R Task R2 executed: **false**",
        "- The original preflight was a false positive because the measured per-step p99 closure was not part of admission.",
        f"- Original p99: `{payload['false_positive_identified']['original_raman_step_closure_p99']}`.",
        f"- Stable-difference-only p99: `{investigation['stable_difference_raman_step_closure_p99']}`.",
        f"- Corrected p99: `{investigation['corrected_raman_step_closure_p99']}` (contract `<1e-3`).",
        f"- Corrected cumulative closure: `{investigation['corrected_raman_cumulative_closure_final']}` (contract `<5e-3`).",
        f"- Legacy Raman alpha maximum: `{investigation['legacy_alpha_R_max']}`.",
        "- Strict full configuration and summary now agree that Raman absorption is OFF.",
        "- A separately authorized Task R2 is still required before any full Job 1 submission.",
        "",
    ])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--job1-config", type=Path, default=JOB1_CONFIG)
    parser.add_argument("--preflight-dir", type=Path, default=PREFLIGHT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--generated-utc")
    args = parser.parse_args(argv)
    artifacts = build_artifacts(
        baseline_path=args.baseline,
        config_path=args.job1_config,
        preflight_dir=args.preflight_dir,
        out_dir=args.out_dir,
        generated_utc=args.generated_utc,
    )
    for name, payload in artifacts.items():
        _write(args.out_dir / name, payload)
    (args.out_dir / "job1_preflight_correction_audit.md").write_text(
        _correction_report(artifacts["job1_preflight_correction_audit.json"]),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
