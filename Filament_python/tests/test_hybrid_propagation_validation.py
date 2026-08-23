"""Low-cost contract tests for the Hybrid Propagation 0.60 m campaign tools.

These tests use synthetic axial traces only.  They never invoke ``sbatch``,
CUDA, or a propagation and never copy a raw NPZ into the derived report.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    path = TOOLS / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_pair(tmp_path: Path, *, shifted: bool = False) -> tuple[Path, Path, Path]:
    post = _load("postprocess_hybrid_propagation_validation.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    z = np.arange(1, 102, dtype=float) * 0.01
    rho_ref = 2.0e22 * np.exp(-((z - 0.75) / 0.05) ** 2)
    rho_hyb = 2.0e22 * np.exp(-((z - (0.77 if shifted else 0.75)) / 0.05) ** 2)
    intensity = 1.0e20 * np.exp(-((z - 0.75) / 0.05) ** 2)
    starts = np.concatenate((np.asarray([0.0]), z[:-1]))
    ends = z.copy()
    trace_fields = (
        "rho_onaxis_max_z", "E_dep_z", "E_dep_rot_z", "E_dep_total_z",
        "alpha_R_max_z", "alpha_R_raw_max_z", "alpha_R_applied_max_z",
        "alpha_ion_raw_max_z", "alpha_ion_corr_max_z", "alpha_ion_applied_max_z",
        "alpha_ib_max_z", "alpha_total_max_z", "delta_n_elec_max_z",
        "delta_n_rot_max_z", "delta_n_plasma_min_z",
        "delta_n_elec_applied_max_z", "delta_n_rot_applied_max_z",
        "delta_n_plasma_applied_min_z", "dphi_kerr_max_abs_z",
        "dphi_elec_max_abs_z", "dphi_rot_max_abs_z", "dphi_plasma_max_abs_z",
        "dphi_elec_applied_max_abs_z", "dphi_rot_applied_max_abs_z",
        "dphi_plasma_raw_max_abs_z", "dphi_plasma_applied_max_abs_z",
        "raman_rhs_l2_norm", "raman_IR_max_raw", "raman_target_loss_step_J",
        "raman_actual_loss_step_J",
    )
    common = {
        "z_axis": z.astype(np.float32),
        "rho_max_z": rho_ref,
        "I_max_z": intensity,
        "U_z": np.full(z.size, 1.0),
        "step_start_z_m": starts,
        "step_end_z_m": ends,
        "nonlinear_operator_applied": np.ones(z.size, dtype=bool),
        "nonlinear_operator_call_count_step": np.ones(z.size, dtype=np.int64),
        "ionization_solver_call_count_step": np.ones(z.size, dtype=np.int64),
        "linear_walltime_step_s": np.full(z.size, 0.01),
        "ionization_walltime_step_s": np.full(z.size, 0.005),
        "raman_operator_walltime_step_s": np.full(z.size, 0.004),
        "nonlinear_walltime_step_s": np.full(z.size, 0.01),
        "total_walltime_step_s": np.full(z.size, 0.10),
        "raman_operator_substep_count": np.full(z.size, 2, dtype=np.int64),
        "raman_convolution_count_step": np.ones(z.size, dtype=np.int64),
        "raman_operator_applied": np.ones(z.size, dtype=bool),
        "gpu_allocated_step_bytes": np.full(z.size, 1000, dtype=np.int64),
        "gpu_reserved_step_bytes": np.full(z.size, 2000, dtype=np.int64),
        "propagation_mode": np.asarray("full_nonlinear_from_z0"),
        "z_nl_start_m": np.asarray(0.0),
        "diagnostic_validation_passed": np.asarray(True),
        "operator_energy_diagnostics_enabled": np.asarray(True),
        "energy_step_start_J": np.full(z.size, 1.0),
        "energy_after_linear_half1_J": np.full(z.size, 1.0),
        "energy_after_raman_pre_J": np.full(z.size, 1.0),
        "energy_after_nonraman_J": np.full(z.size, 1.0),
        "energy_after_raman_post_J": np.full(z.size, 1.0),
        "energy_after_linear_half2_J": np.full(z.size, 1.0),
        "adaptive_rejection_count_z": np.zeros(z.size),
        "safety_mode_trigger_count_z": np.zeros(z.size),
        **{key: np.full(z.size, 1.0) for key in trace_fields},
    }
    for case in ("reference", "hybrid"):
        case_dir = run_dir / case
        case_dir.mkdir()
        payload = dict(common)
        metadata_mode = "full_nonlinear_from_z0"
        metadata_start = 0.0
        if case == "hybrid":
            payload["nonlinear_operator_applied"] = starts >= 0.60
            payload["rho_max_z"] = np.where(payload["nonlinear_operator_applied"], rho_hyb, 0.0)
            payload["nonlinear_operator_call_count_step"] = payload["nonlinear_operator_applied"].astype(np.int64)
            payload["ionization_solver_call_count_step"] = payload["nonlinear_operator_applied"].astype(np.int64)
            payload["nonlinear_walltime_step_s"] = np.where(payload["nonlinear_operator_applied"], 0.01, 0.0)
            payload["ionization_walltime_step_s"] = np.where(payload["nonlinear_operator_applied"], 0.005, 0.0)
            payload["raman_operator_walltime_step_s"] = np.where(payload["nonlinear_operator_applied"], 0.004, 0.0)
            payload["total_walltime_step_s"] = np.where(payload["nonlinear_operator_applied"], 0.08, 0.05)
            payload["raman_operator_substep_count"] = 2 * payload["nonlinear_operator_applied"].astype(np.int64)
            payload["raman_convolution_count_step"] = payload["nonlinear_operator_applied"].astype(np.int64)
            payload["raman_operator_applied"] = payload["nonlinear_operator_applied"].copy()
            for key in trace_fields:
                payload[key] = payload["nonlinear_operator_applied"].astype(float)
            payload["propagation_mode"] = np.asarray("hybrid")
            payload["z_nl_start_m"] = np.asarray(0.6)
            metadata_mode = "hybrid"
            metadata_start = 0.6
        npz_path = case_dir / f"{case}.npz"
        np.savez(npz_path, **payload)
        metadata = {
            "schema": post.CASE_SCHEMA,
            "campaign_id": post.CAMPAIGN_ID,
            "case_id": case,
            "status": "completed",
            "exit_code": 0,
            "slurm_job_id": "123",
            "execution_git_sha": "a" * 40,
            "config_path": f"/data/configs/{case}.json",
            "config_sha256": f"{case:0<64}"[:64],
            "gpu_model": post.EXPECTED_GPU,
            "cpu_threads": 8,
            "backend": "cupy",
            "dtype": "fp32",
            "linear_model": "bk_nee",
            "linear_precision_strategy": "baseline_complex64",
            "thread_environment": {"OMP_NUM_THREADS": "8"},
            "started_at_utc": "2026-08-23T00:00:00+00:00",
            "ended_at_utc": "2026-08-23T00:10:00+00:00",
            "case_total_walltime_s": 10.0 if case == "reference" else 8.0,
            "npz_sha256": post.sha256(npz_path),
            "propagation_mode": metadata_mode,
            "z_nl_start_m": metadata_start,
            "diagnostic_validation_passed": True,
        }
        (case_dir / f"{case}_job_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    pair = {
        "schema": post.PAIR_SCHEMA,
        "campaign_id": post.CAMPAIGN_ID,
        "status": "completed",
        "exit_code": 0,
        "slurm_job_id": "123",
        "execution_git_sha": "a" * 40,
        "gpu_model": post.EXPECTED_GPU,
        "case_order": ["reference", "hybrid"],
        "allocation_count": 1,
        "started_at_utc": "2026-08-23T00:00:00+00:00",
        "ended_at_utc": "2026-08-23T00:20:00+00:00",
        "execution_lock_sha256": "b" * 64,
        "provenance_v2_path": "/data/provenance_v2.json",
        "provenance_v2_sha256": "c" * 64,
    }
    (run_dir / "paired_job_metadata.json").write_text(json.dumps(pair, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema": post.MANIFEST_SCHEMA,
        "campaign_id": post.CAMPAIGN_ID,
        "remote_campaign_root": post.REMOTE_ROOT,
        "strict_config_diff": [
            {"path": "propagation.propagation_mode", "reference": "full_nonlinear_from_z0", "hybrid": "hybrid"},
            {"path": "propagation.z_nl_start", "reference": 0.0, "hybrid": 0.6},
        ],
        "cases": {
            "reference": {"config_sha256": "reference".ljust(64, "0")},
            "hybrid": {"config_sha256": "hybrid".ljust(64, "0")},
        },
    }
    manifest_path = tmp_path / "submission_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    pair["manifest_sha256"] = post.sha256(manifest_path)
    (run_dir / "paired_job_metadata.json").write_text(json.dumps(pair, indent=2) + "\n", encoding="utf-8")
    scheduler = tmp_path / "scheduler_terminal_evidence.json"
    scheduler.write_text(json.dumps({"job_id": "123", "state": "COMPLETED", "exit_code": "0", "source": "test"}) + "\n", encoding="utf-8")
    return run_dir, manifest_path, scheduler


def _postprocess(tmp_path: Path, *, shifted: bool = False):
    post = _load("postprocess_hybrid_propagation_validation.py")
    run_dir, manifest, scheduler = _write_pair(tmp_path, shifted=shifted)
    out_dir = tmp_path / "derived"
    audit = post.process_pair(run_dir, out_dir, manifest_path=manifest, scheduler_terminal_evidence=scheduler)
    return post, run_dir, out_dir, audit


def test_postprocess_requires_scheduler_terminal_evidence(tmp_path):
    post = _load("postprocess_hybrid_propagation_validation.py")
    run_dir, manifest, _ = _write_pair(tmp_path)
    with pytest.raises(post.InsufficientEvidenceError, match="scheduler terminal evidence"):
        post.process_pair(run_dir, tmp_path / "derived", manifest_path=manifest)


def test_postprocess_derives_csv_audit_and_retains_raw_npz(tmp_path):
    post, run_dir, out_dir, audit = _postprocess(tmp_path)
    assert audit["status"] == "complete_evidence"
    assert (out_dir / "reference_axial.csv").is_file()
    assert (out_dir / "hybrid_axial.csv").is_file()
    assert (out_dir / "performance.csv").is_file()
    assert (out_dir / "hybrid_propagation_validation_audit.json").is_file()
    assert (run_dir / "reference" / "reference.npz").is_file()
    assert not (out_dir / "reference.npz").exists()


def test_compare_passes_or_rejects_only_after_complete_pair(tmp_path):
    post, run_dir, out_dir, audit = _postprocess(tmp_path)
    compare = _load("compare_hybrid_propagation_validation.py")
    with pytest.raises(compare.InsufficientEvidenceError, match="explicit visual-veto"):
        compare.compare_pair(
            out_dir / "hybrid_propagation_validation_audit.json",
            tmp_path / "missing_visual_review",
        )
    veto = tmp_path / "visual_veto.json"
    veto.write_text(json.dumps({"veto": False, "reviewer": "synthetic-test"}) + "\n", encoding="utf-8")
    result = compare.compare_pair(out_dir / "hybrid_propagation_validation_audit.json", tmp_path / "comparison", visual_veto=veto)
    assert result["classification"] == "hybrid_0p60_supported"
    assert not result["failed_gates"]
    assert (tmp_path / "comparison" / "hybrid_propagation_validation_comparison.png").is_file()

    # A complete but deliberately shifted hybrid trace is a mechanical gate
    # failure, not an evidence failure and not a retry request.
    _, _, shifted_out, _ = _postprocess(tmp_path / "shifted", shifted=True)
    shifted_result = compare.compare_pair(shifted_out / "hybrid_propagation_validation_audit.json", tmp_path / "shifted_comparison", visual_veto=False)
    assert shifted_result["classification"] == "hybrid_0p60_not_supported"
    assert "G1_onset_1e22" in shifted_result["failed_gates"]

    with pytest.raises(compare.InsufficientEvidenceError, match="curve and added/disappeared feature"):
        compare.compare_pair(
            out_dir / "hybrid_propagation_validation_audit.json",
            tmp_path / "invalid_visual_veto",
            visual_veto={"veto": True, "reason": "too vague"},
        )


def test_campaign_shell_contract_has_single_allocation_and_no_retry():
    submit = (TOOLS / "submit_hybrid_propagation_validation.sh").read_text(encoding="utf-8")
    batch = (TOOLS / "hybrid_propagation_validation.sbatch").read_text(encoding="utf-8")
    assert submit.count("sbatch --hold --parsable") == 1
    assert "case_order" in batch and "reference" in batch and "hybrid" in batch
    assert "release_failure_record" in submit or "sbatch_failure_record" in submit
    assert "retry" not in submit.lower() and "retry" not in batch.lower()
    assert "RAW NPZ" not in submit
    assert "PyCAP" not in submit and "PyCAP" not in batch
    assert "run_from_file" in batch
    assert batch.count("run_case(\"reference\"") == 1
    assert batch.count("run_case(\"hybrid\"") == 1
