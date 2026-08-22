from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


NUMERICAL_FIELDS = (
    "U_rel_change_z",
    "E_dep_cumulative_z",
    "E_loss_from_input_z",
    "dz_used_z",
    "adaptive_rejection_count_z",
    "safety_mode_trigger_count_z",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_case(tmp_path: Path, name: str, rho: np.ndarray, *, include_numerical: bool = True) -> tuple[Path, Path]:
    x = np.linspace(-10.0, 5.0, rho.size)
    intensity = 1.0e17 * np.exp(-((x + 2.0) / 2.5) ** 2) + 1.0e15
    columns = {
        "x_focus_cm": x,
        "rho_max_z": rho,
        "I_max_z": intensity,
    }
    if include_numerical:
        columns.update({
            "U_rel_change_z": np.zeros_like(x),
            "E_dep_cumulative_z": np.linspace(0.0, 1.0e-9, x.size),
            "E_loss_from_input_z": np.zeros_like(x),
            "dz_used_z": np.full_like(x, 1.0e-4),
            "adaptive_rejection_count_z": np.zeros_like(x),
            "safety_mode_trigger_count_z": np.zeros_like(x),
        })
    axial = tmp_path / f"{name}_axial.csv"
    headers = list(columns)
    axial.write_text(
        ",".join(headers) + "\n" + "\n".join(
            ",".join(str(columns[key][index]) for key in headers)
            for index in range(x.size)
        ) + "\n",
        encoding="utf-8",
    )
    extras = tmp_path / f"{name}_extras.csv"
    extras.write_text(
        "x_focus_cm,IR_max_z\n" + "\n".join(f"{xx},1.0" for xx in x) + "\n",
        encoding="utf-8",
    )
    return axial, extras


def _write_audit(
    tmp_path: Path,
    name: str,
    job_id: str,
    axial: Path,
    extras: Path,
    *,
    passed: bool = True,
    path_override: Path | None = None,
) -> Path:
    audit = tmp_path / f"{name}_audit.json"
    payload = {
        "schema": "test.c2.audit.v1",
        "gate": "passed" if passed else "failed",
        "run_metadata": {"slurm_job_id": job_id},
        "inputs": {
            "axial": {"path": str(path_override or axial), "sha256": _sha(axial)},
            "extras": {"path": str(extras), "sha256": _sha(extras)},
        },
        "evidence": {
            "finite": True,
            "crossings": True,
            "overlap": True,
            "numerical": True,
        },
    }
    audit.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return audit


def _write_candidate_chain(
    tmp_path: Path,
    rho: np.ndarray,
    post,
    *,
    include_numerical: bool = True,
    job_id: str = "180800",
) -> tuple[Path, Path, Path]:
    """Build a self-contained candidate raw/metadata/lock chain for tests."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_axial, source_extras = _write_case(
        tmp_path, "candidate_chain_source", rho, include_numerical=include_numerical,
    )
    run_dir = tmp_path / f"candidate_chain_run_{job_id}_{'full' if include_numerical else 'incomplete'}"
    run_dir.mkdir()
    npz_path = run_dir / "candidate.npz"
    x = np.linspace(-10.0, 5.0, rho.size)
    z = 0.95 + x / 100.0
    ones = np.ones_like(x)
    raw = {
        "z_axis": z, "rho_max_z": rho, "I_max_z": 1.0e17 * np.exp(-((x + 2.0) / 2.5) ** 2) + 1.0e15,
        "rho_onaxis_max_z": rho, "w_mom_z": ones, "fwhm_time_z": ones, "U_z": ones,
        "alpha_ion_applied_max_z": ones, "dphi_plasma_applied_max_abs_z": ones,
        "dphi_elec_applied_max_abs_z": ones, "raman_IR_max_raw": ones,
        "raman_rhs_l2_norm": ones, "raman_target_loss_step_J": ones,
        "raman_actual_loss_step_J": ones, "raman_closure_residual_step": ones,
        "raman_cumulative_closure_residual": ones, "U_rel_change_z": np.zeros_like(x),
        "E_dep_cumulative_z": np.linspace(0.0, 1.0e-9, x.size), "E_loss_from_input_z": np.zeros_like(x),
        "dz_used_z": np.full_like(x, 1.0e-4), "adaptive_rejection_count_z": np.zeros_like(x),
        "safety_mode_trigger_count_z": np.zeros_like(x),
    }
    np.savez(npz_path, **raw)
    config_path = post.FILAMENT_ROOT / "results" / "isaacs_complete_eq27" / "120fs_talebpour_isaacs_complete_eq27.json"
    manifest_path = post.FILAMENT_ROOT / "results" / "isaacs_complete_eq27" / "submission_manifest.json"
    metadata_path = run_dir / "metadata.json"
    execution_sha = subprocess.check_output(
        ["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"], text=True,
    ).strip()
    branch = subprocess.check_output(
        ["git", "-C", str(ROOT.parent), "branch", "--show-current"], text=True,
    ).strip()
    staging_path = tmp_path / f"staging_provenance_{job_id}.json"
    staging_path.write_text(json.dumps({
        "schema": "khz_filament.isaacs_complete_eq27.staging_provenance.v1",
        "method": "verified_git_bundle_after_remote_github_transport_failure",
        "source_class": "verified_bundle_non_strict",
        "expected_git_sha": execution_sha,
        "branch": branch,
        "github_push_verified": True,
        "bundle_path": str(tmp_path / f"bundle_{job_id}.bundle"),
        "bundle_sha256": "b" * 64,
        "remote_failure_logs": ["github_transport_failure.log"],
    }, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "schema": "khz_filament.isaacs_complete_eq27.job_metadata.v1", "status": "completed", "exit_code": 0,
        "slurm_job_id": job_id, "execution_git_sha": execution_sha, "propagation_invocations": 1,
        "profiling_enabled": False, "case_id": "complete_eq27", "gpu_model": "NVIDIA GeForce RTX 5090",
        "operator_mode": post.COMPLETE_MODE, "use_raman_full_operator": True,
        "config_path": str(config_path.resolve()), "config_sha256": _sha(config_path),
        "npz_sha256": _sha(npz_path), "manifest_path": str(manifest_path.resolve()), "manifest_sha256": _sha(manifest_path),
        "global_consumed_lock": str((tmp_path / f"global_{job_id}_{'full' if include_numerical else 'incomplete'}" / ".consumed.lock").resolve()),
        "staging_provenance_path": str(staging_path.resolve()),
        "staging_provenance_sha256": _sha(staging_path),
        "staging_provenance_method": "verified_git_bundle_after_remote_github_transport_failure",
        "staging_provenance_source_class": "verified_bundle_non_strict",
        "staging_provenance_branch": branch,
        "method": "verified_git_bundle_after_remote_github_transport_failure",
        "source_class": "verified_bundle_non_strict",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    lock_path = tmp_path / "execution_lock.json"
    lock = {
        "schema": "khz_filament.isaacs_complete_eq27.c2_execution_lock.v1", "campaign_id": "isaacs_complete_eq27_c2",
        "remote_campaign_root": "/data/run01/scvi806/user_Wangjimin/isaacs_complete_eq27_c2",
        "status": "authorized_not_consumed", "expected_gpu_model": "NVIDIA GeForce RTX 5090",
        "operator_mode": post.COMPLETE_MODE, "use_raman_full_operator": True, "expected_git_sha": execution_sha,
        "manifest_path": "Filament_python/results/isaacs_complete_eq27/submission_manifest.json",
        "config_path": "results/isaacs_complete_eq27/120fs_talebpour_isaacs_complete_eq27.json",
        "derived_config_path": "results/isaacs_complete_eq27/120fs_talebpour_isaacs_complete_eq27.json",
        "config_sha256": _sha(config_path), "manifest_sha256": _sha(manifest_path),
    }
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    metadata["execution_lock_path"] = str(lock_path.resolve())
    metadata["execution_lock_sha256"] = _sha(lock_path)
    submission_path = run_dir / "SUBMISSION_LOCK"
    submission_path.write_text(
        "\n".join([
            "case_id=complete_eq27", "campaign_id=isaacs_complete_eq27_c2",
            "manifest_path=" + str(manifest_path.resolve()), "manifest_sha256=" + _sha(manifest_path),
            "execution_lock_path=" + str(lock_path.resolve()), "execution_lock_sha256=" + _sha(lock_path),
            "config_path=" + str(config_path.resolve()), "expected_config_sha256=" + _sha(config_path),
            "expected_git_sha=" + execution_sha, "reservation_token=token-" + job_id,
        ]) + "\n", encoding="utf-8",
    )
    metadata["submission_lock_path"] = str(submission_path.resolve())
    metadata["submission_lock_sha256"] = _sha(submission_path)
    global_record = tmp_path / f"global_{job_id}_{'full' if include_numerical else 'incomplete'}" / ".consumed.lock" / "submission_record.txt"
    global_record.parent.mkdir(parents=True)
    global_record.write_text(
        "\n".join([
            "campaign_id=isaacs_complete_eq27_c2", "remote_campaign_root=/data/run01/scvi806/user_Wangjimin/isaacs_complete_eq27_c2",
            "manifest_path=" + str(manifest_path.resolve()), "manifest_sha256=" + _sha(manifest_path),
            "execution_lock_path=" + str(lock_path.resolve()), "execution_lock_sha256=" + _sha(lock_path),
            "expected_git_sha=" + execution_sha, "run_dir=" + str(run_dir.resolve()),
            "reservation_token=token-" + job_id,
        ]) + "\n", encoding="utf-8",
    )
    receipt_path = run_dir / "job_receipt.json"
    receipt = {
        "schema": "khz_filament.isaacs_complete_eq27.job_receipt.v1",
        "state": "held", "job_id": job_id, "reservation_token": "token-" + job_id,
        "campaign_id": "isaacs_complete_eq27_c2",
        "remote_campaign_root": "/data/run01/scvi806/user_Wangjimin/isaacs_complete_eq27_c2",
        "run_dir": str(run_dir.resolve()), "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha(manifest_path), "execution_lock_path": str(lock_path.resolve()),
        "execution_lock_sha256": _sha(lock_path), "config_path": str(config_path.resolve()),
        "config_sha256": _sha(config_path), "expected_git_sha": execution_sha,
        "staging_provenance_path": str(staging_path.resolve()),
        "staging_provenance_sha256": _sha(staging_path),
        "staging_provenance_method": "verified_git_bundle_after_remote_github_transport_failure",
        "staging_provenance_source_class": "verified_bundle_non_strict",
        "staging_provenance_branch": branch,
        "method": "verified_git_bundle_after_remote_github_transport_failure",
        "source_class": "verified_bundle_non_strict",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["job_receipt_path"] = str(receipt_path.resolve())
    metadata["job_receipt_sha256"] = _sha(receipt_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    result = {
        "passed": True, "failures": [], "metadata": metadata,
        "semantic_values": {"raman_operator_mode": post.COMPLETE_MODE}, "field_status": {}, "closure": {},
        "axial_rows": _csv_rows(source_axial), "raman_rows": _csv_rows(source_extras),
        "npz_sha256": _sha(npz_path), "config_sha256": _sha(config_path), "expected_execution_sha": execution_sha,
        "provenance_binding": {
            "npz": {"path": str(npz_path.resolve()), "sha256": _sha(npz_path)},
            "metadata": {"path": str(metadata_path.resolve()), "sha256": _sha(metadata_path)},
            "config": {"path": str(config_path.resolve()), "sha256": _sha(config_path)},
            "manifest": {"path": str(manifest_path.resolve()), "sha256": _sha(manifest_path)},
            "execution_lock": {"path": str(lock_path.resolve()), "sha256": _sha(lock_path)},
            "submission_lock": {"path": str(submission_path.resolve()), "sha256": _sha(submission_path)},
            "global_consumed_lock": {"path": str(global_record.resolve()), "sha256": _sha(global_record)},
            "job_receipt": {"path": str(receipt_path.resolve()), "sha256": _sha(receipt_path)},
            "staging_provenance": {"path": str(staging_path.resolve()), "sha256": _sha(staging_path)},
            "staging_provenance_method": "verified_git_bundle_after_remote_github_transport_failure",
            "staging_provenance_source_class": "verified_bundle_non_strict",
            "staging_provenance_branch": branch,
            "method": "verified_git_bundle_after_remote_github_transport_failure",
            "source_class": "verified_bundle_non_strict",
        },
    }
    post.validate = lambda *args, **kwargs: result
    loaded_compare = sys.modules.get("compare_isaacs_complete_eq27")
    if loaded_compare is not None:
        loaded_compare.validate_manifest_lock = lambda *args, **kwargs: {
            "config_path": config_path.resolve(), "head": execution_sha,
        }
    out_dir = tmp_path / "candidate_chain_audit"
    post.write_audit(result, out_dir, npz_path=npz_path, config_path=config_path)
    return out_dir / "isaacs_complete_eq27_reaudit.json", out_dir / "isaacs_complete_eq27_axial_diagnostics.csv", out_dir / "isaacs_complete_eq27_raman_extras.csv"


def _write_fixed_raw_case(tmp_path: Path, role: str, rho: np.ndarray, fallback, compare) -> dict[str, str]:
    """Create a small valid raw chain and patch fixed evidence for unit tests."""
    x = np.linspace(-10.0, 5.0, rho.size)
    z = 0.95 + x / 100.0
    ones = np.ones_like(x)
    raw = {
        "z_axis": z,
        "rho_max_z": rho,
        "rho_onaxis_max_z": rho,
        "I_max_z": 1.0e17 * np.exp(-((x + 2.0) / 2.5) ** 2) + 1.0e15,
        "w_mom_z": ones,
        "fwhm_time_z": ones,
        "U_z": ones,
        "alpha_ion_applied_max_z": ones,
        "dphi_plasma_applied_max_abs_z": ones,
        "dphi_elec_applied_max_abs_z": ones,
        "raman_IR_max_raw": ones,
        "raman_rhs_l2_norm": ones,
        "raman_target_loss_step_J": ones,
        "raman_actual_loss_step_J": ones,
        "raman_closure_residual_step": ones,
        "raman_cumulative_closure_residual": ones,
        "U_rel_change_z": np.zeros_like(x),
        "E_dep_cumulative_z": np.linspace(0.0, 1.0e-9, x.size),
        "E_loss_from_input_z": np.zeros_like(x),
        "dz_used_z": np.full_like(x, 1.0e-4),
        "adaptive_rejection_count_z": np.zeros_like(x),
        "safety_mode_trigger_count_z": np.zeros_like(x),
    }
    case = fallback.FIXED_RAW_EVIDENCE[role]["case"]
    stem = "on" if case == "on" else "off"
    raw_path = tmp_path / role / f"test_a_{stem}.npz"
    raw_path.parent.mkdir(parents=True)
    np.savez(raw_path, **raw)
    metadata_path = raw_path.with_name(f"test_a_{stem}_job_metadata.json")
    metadata = {
        "schema": "phase8c.full_eq27_raman.test_a.job_metadata.v1",
        "case_id": case,
        "slurm_job_id": fallback.FIXED_RAW_EVIDENCE[role]["job_id"],
        "expected_sha": fallback.EXPECTED_EXECUTION_SHA,
        "actual_sha": fallback.EXPECTED_EXECUTION_SHA,
        "sha_match": True,
        "git_status_clean": True,
        "config_path": f"/data/configs/{role}.json",
        "config_sha256": "a" * 64 if role == "current_full_eq27" else "b" * 64,
        "gpu_model": fallback.EXPECTED_GPU_MODEL,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    diagnostic_path = raw_path.with_name(f"test_a_{stem}.diagnostic_report.json")
    diagnostic_path.write_text(json.dumps({"status": "COMPLETED", "exit_code": "0:0"}) + "\n", encoding="utf-8")
    expected = dict(fallback.FIXED_RAW_EVIDENCE[role])
    expected.update({
        "npz_path": str(raw_path), "npz_sha256": _sha(raw_path),
        "metadata_path": str(metadata_path), "metadata_sha256": _sha(metadata_path),
        "diagnostic_report_path": str(diagnostic_path), "diagnostic_report_sha256": _sha(diagnostic_path),
        "config_sha256": metadata["config_sha256"],
    })
    fallback.FIXED_RAW_EVIDENCE[role] = expected
    fallback._scheduler_evidence = lambda job_id: {
        "job_id": job_id,
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed": "00:01:00",
        "node_list": "test-node",
        "submit_time": "2026-08-22T00:00:00",
        "start_time": "2026-08-22T00:00:01",
        "end_time": "2026-08-22T00:01:01",
        "source": "live_sacct",
    }
    compare._scheduler_evidence = fallback._scheduler_evidence
    compare.FIXED_FALLBACK_RAW[role] = expected
    return expected


def _write_fixed_fallback_audits(tmp_path: Path, current_rho: np.ndarray, off_rho: np.ndarray, fallback, compare):
    current = _write_fixed_raw_case(tmp_path, "current_full_eq27", current_rho, fallback, compare)
    off = _write_fixed_raw_case(tmp_path, "raman_off", off_rho, fallback, compare)
    pair = fallback.prepare_pair(out_dir=tmp_path / "fallback_audits")
    return (
        tmp_path / "fallback_audits" / "current_full_eq27_fallback_audit.json",
        tmp_path / "fallback_audits" / "raman_off_fallback_audit.json",
        current,
        off,
    )


def _csv_rows(path: Path) -> list[dict[str, float]]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def test_prepare_is_single_exact_operator_field_and_preserves_fixed_contract():
    module = _load("prepare_isaacs_complete_eq27_job.py")
    base = json.loads(module.BASE.read_text(encoding="utf-8"))
    derived, differences = module.build(base)
    assert derived["raman"]["operator_mode"] == module.COMPLETE_MODE
    assert differences == [{
        "path": "raman.operator_mode",
        "full_isaacs_eq27": "full_isaacs_eq27",
        "full_isaacs_eq27_complete": "full_isaacs_eq27_complete",
    }]
    assert derived["beam"]["P0_peak"] == base["beam"]["P0_peak"] == 17e9
    assert derived["propagation"]["dz"] == base["propagation"]["dz"] == 1e-4
    assert derived["raman"]["n_R"] == base["raman"]["n_R"] == 2.3e-23


def test_prepare_manifest_has_exactly_one_job_and_no_scan_or_profiling(tmp_path):
    module = _load("prepare_isaacs_complete_eq27_job.py")
    module.main(["--out-dir", str(tmp_path)])
    manifest = json.loads((tmp_path / "submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["campaign_id"] == "isaacs_complete_eq27_c2"
    assert manifest["remote_campaign_root"] == "/data/run01/scvi806/user_Wangjimin/isaacs_complete_eq27_c2"
    assert manifest["status"] == "prepared_not_submitted"
    assert manifest["expected_git_sha"] is None
    assert manifest["execution_lock_required"] is True
    assert manifest["expected_git_sha_resolution"] == "external execution_lock generated after final source commit"
    assert manifest["comparison_inputs"]["pycap_120fs_sha256"] == "9b43e75ebc08ccb0a7796829e45c6727b42ab12cd661b9a3d8d235ef89d31461"
    assert manifest["locked_base_config_sha256"] == module.LOCKED_BASE_SHA256
    assert manifest["jobs_authorized"] == manifest["full_jobs_authorized"] == 1
    assert manifest["jobs_submitted"] == manifest["full_production_jobs_submitted"] == 0
    assert manifest["scan_jobs_authorized"] == 0
    assert manifest["profiling_jobs_authorized"] == 0
    assert manifest["parameter_scan_authorized"] is False
    assert manifest["resources"] == {
        **manifest["resources"],
        "partition": "gpu", "gpu_count": 1, "cpu_threads": 8,
        "requested_time": "15:00:00", "expected_gpu_model": "NVIDIA GeForce RTX 5090",
    }
    assert json.loads((tmp_path / "c2_config_diff.json").read_text(encoding="utf-8"))["differences"] == manifest["strict_config_diff"]


def test_execution_lock_binds_clean_head_manifest_and_config_without_submission(tmp_path, monkeypatch):
    module = _load("create_isaacs_complete_eq27_execution_lock.py")
    repo = module.REPO
    manifest_path = module.DEFAULT_MANIFEST
    output = tmp_path / "external" / "execution-lock.json"
    monkeypatch.setattr(module, "_git", lambda *args: {
        ("status", "--porcelain=v1"): "",
        ("ls-files", "--error-unmatch", "--", "Filament_python/results/isaacs_complete_eq27/submission_manifest.json"): "Filament_python/results/isaacs_complete_eq27/submission_manifest.json",
        ("rev-parse", "HEAD"): "a" * 40,
        ("merge-base", "--is-ancestor", module.C1_COMMIT, "a" * 40): "",
    }[args])

    payload = module.create_lock(manifest_path, output)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert payload == written
    assert written["status"] == "authorized_not_consumed"
    assert written["expected_git_sha"] == "a" * 40
    assert written["manifest_path"] == "Filament_python/results/isaacs_complete_eq27/submission_manifest.json"
    assert written["manifest_sha256"] == _sha(manifest_path)
    assert written["config_path"] == "results/isaacs_complete_eq27/120fs_talebpour_isaacs_complete_eq27.json"
    assert written["config_sha256"] == "6c5967f41a0f5d110b3220d5637f764bccb4fe2bd4406647ef30172780954b73"
    source = (ROOT / "tools" / "create_isaacs_complete_eq27_execution_lock.py").read_text(encoding="utf-8")
    assert 'subprocess.run(["sbatch"' not in source


def test_staging_provenance_schema_hash_and_field_tamper_are_rejected(tmp_path):
    module = _load("create_isaacs_complete_eq27_execution_lock.py")
    head = subprocess.check_output(["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(["git", "-C", str(ROOT.parent), "branch", "--show-current"], text=True).strip()
    payload = {
        "schema": module.STAGING_PROVENANCE_SCHEMA,
        "method": module.STAGING_PROVENANCE_METHOD,
        "source_class": module.STAGING_PROVENANCE_SOURCE_CLASS,
        "expected_git_sha": head,
        "branch": branch,
        "github_push_verified": True,
        "bundle_path": str(tmp_path / "verified.bundle"),
        "bundle_sha256": "b" * 64,
        "remote_failure_logs": ["remote-github-timeout.log"],
    }
    path = tmp_path / "staging_provenance.json"

    def write(value):
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    write(payload)
    valid = module.validate_staging_provenance(
        path, expected_sha256=_sha(path), expected_git_sha=head, repo=ROOT.parent,
        expected_branch=branch,
    )
    assert valid["source_class"] == "verified_bundle_non_strict"

    with pytest.raises(module.ExecutionLockError, match="SHA mismatch"):
        module.validate_staging_provenance(path, expected_sha256="0" * 64, expected_git_sha=head, repo=ROOT.parent)

    for field, value in {
        "schema": "tampered.schema",
        "method": "direct_github_push",
        "source_class": "strict_remote_verified",
        "branch": "tampered-branch",
        "github_push_verified": False,
        "bundle_path": "",
        "bundle_sha256": "",
        "remote_failure_logs": [],
    }.items():
        tampered = dict(payload)
        tampered[field] = value
        write(tampered)
        with pytest.raises(module.ExecutionLockError):
            module.validate_staging_provenance(
                path, expected_sha256=_sha(path), expected_git_sha=head, repo=ROOT.parent,
            )


def test_submit_and_batch_scripts_are_single_case_and_single_invocation():
    submit = (ROOT / "tools" / "submit_isaacs_complete_eq27_job.sh").read_text(encoding="utf-8")
    batch = (ROOT / "tools" / "isaacs_complete_eq27_full.sbatch").read_text(encoding="utf-8")
    assert submit.count('if sbatch_output="$(sbatch --hold --parsable') == 1
    assert 'sbatch_output="$(sbatch --hold --parsable' in submit
    assert "mkdir -- \"${RUN_DIR}\"" in submit
    assert "SUBMISSION_LOCK" in submit
    assert "IFS=$'\\t' read -r CONFIG_PATH EXPECTED_CONFIG_SHA256" in submit
    assert "slurm_job_id.txt" in submit
    assert "CASE_ID=complete_eq27" in submit
    assert "\npython " not in submit
    assert batch.count("python - <<'PY'") == 1
    assert batch.count("run_from_file(") == 1
    assert 'payload["propagation_invocations"] += 1' in batch
    assert '"propagation_invocations": 0' in batch
    assert '"propagation_invocations": 1' not in batch
    assert 'candidate NPZ already exists' in batch
    assert 'job metadata already exists' in batch
    assert "CASE_ID=complete_eq27" in batch
    for token in ("EXPECTED_GIT_SHA", "EXPECTED_CONFIG_SHA256", "EXPECTED_GPU_MODEL", "git status --porcelain", "CUPY_CACHE_DIR", "gpu_allocated_step_bytes", "gpu_reserved_step_bytes"):
        assert token in batch


def test_submit_manifest_binding_and_campaign_lock_are_global_across_run_dirs():
    submit = (ROOT / "tools" / "submit_isaacs_complete_eq27_job.sh").read_text(encoding="utf-8")
    for token in (
        ': "${MANIFEST_PATH:?missing MANIFEST_PATH}"',
        ': "${EXPECTED_MANIFEST_SHA256:?missing EXPECTED_MANIFEST_SHA256}"',
        ': "${EXECUTION_LOCK_PATH:?missing EXECUTION_LOCK_PATH}"',
        ': "${EXPECTED_EXECUTION_LOCK_SHA256:?missing EXPECTED_EXECUTION_LOCK_SHA256}"',
        'sha256sum "${MANIFEST_PATH}"',
        'sha256sum "${EXECUTION_LOCK_PATH}"',
        "python3 - \"${MANIFEST_PATH}\"",
        "authorized_not_consumed",
        'expected_git_sha',
        'derived_config_sha256',
        'remote_campaign_root',
        'jobs_submitted',
        'GLOBAL_CONSUMED_LOCK="${FIXED_REMOTE_CAMPAIGN_ROOT}/.consumed.lock"',
        'mkdir -- "${GLOBAL_CONSUMED_LOCK}"',
        'manifest_sha256=%s',
        'job_id_source=job_receipt',
    ):
        assert token in submit
    assert submit.count('if sbatch_output="$(sbatch --hold --parsable') == 1
    assert submit.index('mkdir -- "${RUN_DIR}"') < submit.index('mkdir -- "${GLOBAL_CONSUMED_LOCK}"')
    assert 'REMOTE_CAMPAIGN_ROOT="${FIXED_REMOTE_CAMPAIGN_ROOT}"' in submit
    assert 'MANIFEST_CONFIG_PATH="${CONFIG_PATH:-}"' in submit
    assert 'CONFIG_PATH does not resolve to manifest derived_config' in submit
    assert 'execution lock manifest_sha256 does not match actual manifest hash' in submit
    assert 'execution lock config_path does not match manifest derived_config' in submit
    assert 'source worktree is not clean' in submit
    assert 'rm -rf -- "${RUN_DIR}" "${GLOBAL_CONSUMED_LOCK}"' not in submit
    assert 'ambiguous_sbatch_nonzero' in submit
    assert '>> "${GLOBAL_LOCK_RECORD}"' not in submit
    assert '>> "${SUBMISSION_LOCK}"' not in submit
    assert 'write_post_sbatch_failure' in submit
    assert 'campaign lock, RUN_DIR, and any held job are retained' in submit


def test_submit_and_batch_bind_gpu_and_operational_provenance_without_cleanup():
    submit = (ROOT / "tools" / "submit_isaacs_complete_eq27_job.sh").read_text(encoding="utf-8")
    batch = (ROOT / "tools" / "isaacs_complete_eq27_full.sbatch").read_text(encoding="utf-8")
    for token in (
        'EXPECTED_GPU_MODEL="${EXPECTED_GPU_MODEL}"',
        'lock.get("expected_gpu_model") != os.environ["EXPECTED_GPU_MODEL"]',
        'resources.get(key) != expected',
        'config_diff(source_payload, derived_payload)',
        'source_sha != manifest.get("source_config_sha256")',
        'pycap_sha = "9b43e75ebc08ccb0a7796829e45c6727b42ab12cd661b9a3d8d235ef89d31461"',
    ):
        assert token in submit
    for token in (
        ': "${CAMPAIGN_ID:?missing CAMPAIGN_ID}"',
        ': "${MANIFEST_PATH:?missing MANIFEST_PATH}"',
        ': "${EXECUTION_LOCK_PATH:?missing EXECUTION_LOCK_PATH}"',
        ': "${GLOBAL_CONSUMED_LOCK:?missing GLOBAL_CONSUMED_LOCK}"',
        '"campaign_id": os.environ["CAMPAIGN_ID"]',
        '"manifest_sha256": os.environ["EXPECTED_MANIFEST_SHA256"]',
        '"execution_lock_sha256": os.environ["EXPECTED_EXECUTION_LOCK_SHA256"]',
        '"global_consumed_lock": os.environ["GLOBAL_CONSUMED_LOCK"]',
    ):
        assert token in batch
    assert 'rm -rf -- "${RUN_DIR}"' not in submit


def test_submit_batch_and_receipt_require_external_staging_provenance_binding():
    submit = (ROOT / "tools" / "submit_isaacs_complete_eq27_job.sh").read_text(encoding="utf-8")
    batch = (ROOT / "tools" / "isaacs_complete_eq27_full.sbatch").read_text(encoding="utf-8")
    for token in (
        ': "${STAGING_PROVENANCE_PATH:?missing STAGING_PROVENANCE_PATH}"',
        ': "${EXPECTED_STAGING_PROVENANCE_SHA256:?missing EXPECTED_STAGING_PROVENANCE_SHA256}"',
        "validate_staging_provenance",
        "staging_provenance_path",
        "staging_provenance_sha256",
        "STAGING_PROVENANCE_METHOD",
        "STAGING_PROVENANCE_SOURCE_CLASS",
    ):
        assert token in submit
        assert token in batch
    assert 'staging_provenance_path' in submit[submit.index('sbatch_output='):]
    assert 'staging_provenance_sha256' in batch
    assert 'submission_record.txt' in submit


def test_candidate_raw_chain_rejects_staging_provenance_and_receipt_tamper(tmp_path):
    post = _load("postprocess_isaacs_complete_eq27.py")
    compare = _load("compare_isaacs_complete_eq27.py")
    audit_path, axial, extras = _write_candidate_chain(tmp_path, np.ones(8) * 1.2e22, post)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    staging_path = Path(payload["raw_source"]["staging_provenance"]["path"])
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    staging["method"] = "direct_github_push"
    staging_path.write_text(json.dumps(staging), encoding="utf-8")
    failures = compare._candidate_raw_chain(
        payload, audit_path=audit_path, expected_job_id="180800", axial=axial, extras=extras,
    )
    assert any("staging provenance" in item.lower() for item in failures)

    audit_path, axial, extras = _write_candidate_chain(tmp_path / "receipt", np.ones(8) * 1.2e22, post)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    receipt_path = Path(payload["raw_source"]["job_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["staging_provenance_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    failures = compare._candidate_raw_chain(
        payload, audit_path=audit_path, expected_job_id="180800", axial=axial, extras=extras,
    )
    assert any("receipt" in item.lower() and "staging" in item.lower() for item in failures)


def test_candidate_audit_and_report_preserve_verified_bundle_non_strict_limitation(tmp_path):
    post = _load("postprocess_isaacs_complete_eq27.py")
    audit_path, _, _ = _write_candidate_chain(tmp_path, np.ones(8) * 1.2e22, post)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["provenance_class"] == "verified_bundle_non_strict"
    assert payload["raw_source"]["provenance_class"] == "verified_bundle_non_strict"
    report = audit_path.with_name("isaacs_complete_eq27_reaudit_report.md").read_text(encoding="utf-8")
    assert "verified_bundle_non_strict" in report
    assert "direct GitHub remote push/fetch" in report


def test_compare_rejects_replaced_pycap_and_tampered_fixed_raw_chain(tmp_path, monkeypatch):
    module = _load("compare_isaacs_complete_eq27.py")
    fake_pycap = tmp_path / "replacement_pycap.csv"
    fake_pycap.write_text("x_focus_cm,rho_1e16_cm3\n0,1\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(module, "FIXED_PYCAP_PATH", tmp_path / "fixed_pycap.csv")
    monkeypatch.setattr(module, "FIXED_PYCAP_SHA256", _sha(fake_pycap))
    assert module._fixed_pycap_failures(fake_pycap)

    fallback = _load("prepare_isaacs_eq27_fallback_comparator_audit.py")
    raw = _write_fixed_raw_case(tmp_path, "current_full_eq27", np.ones(8) * 1.2e22, fallback, module)
    audit = fallback.audit_comparator(role="current_full_eq27", out_dir=tmp_path / "audit")
    audit_path = tmp_path / "audit" / "current_full_eq27_fallback_audit.json"
    Path(raw["npz_path"]).write_bytes(Path(raw["npz_path"]).read_bytes() + b"tamper")
    failures = module._fallback_raw_chain(
        audit, label="current_full_eq27", expected_job_id="180748",
    )
    assert any("wrong SHA256" in item for item in failures)


def test_classification_A_B_C_thresholds():
    module = _load("compare_isaacs_complete_eq27.py")
    supported = module.classify(
        shift_abs_cm=1.0, onset_improvement_cm=0.6,
        candidate_peak_density_rel_error=0.10,
        current_rmse=10.0, candidate_rmse=10.5,
        candidate_peak_position_error_cm=0.4, current_peak_position_error_cm=0.8,
    )
    not_supported = module.classify(
        shift_abs_cm=0.05, onset_improvement_cm=0.6,
        candidate_peak_density_rel_error=0.10,
        current_rmse=10.0, candidate_rmse=10.0,
        candidate_peak_position_error_cm=0.4, current_peak_position_error_cm=0.8,
    )
    partial = module.classify(
        shift_abs_cm=0.3, onset_improvement_cm=0.2,
        candidate_peak_density_rel_error=0.80,
        current_rmse=10.0, candidate_rmse=30.0,
        candidate_peak_position_error_cm=2.0, current_peak_position_error_cm=0.8,
    )
    assert supported == "electronic_eq27_operator_supported"
    assert not_supported == "electronic_eq27_operator_not_supported"
    assert partial == "electronic_eq27_operator_partial"


def test_postprocess_contract_rejects_missing_complete_operator_traces_without_copying_npz(tmp_path):
    prepare = _load("prepare_isaacs_complete_eq27_job.py")
    post = _load("postprocess_isaacs_complete_eq27.py")
    config_path = tmp_path / "complete.json"
    derived, _ = prepare.build(json.loads(prepare.BASE.read_text(encoding="utf-8")))
    config_path.write_text(json.dumps(derived), encoding="utf-8")
    npz_path = tmp_path / "candidate.npz"
    np.savez(npz_path)
    out_dir = tmp_path / "postprocess_123"
    result = post.validate(npz_path, config_path)
    assert result["passed"] is False
    assert any("raman_operator_mode" in item for item in result["failures"])
    post.write_audit(result, out_dir, npz_path=npz_path, config_path=config_path)
    assert (out_dir / "isaacs_complete_eq27_reaudit.json").is_file()
    assert not (out_dir / "candidate.npz").exists()


def test_postprocess_requires_completed_zero_exit_single_invocation_jobid_and_hashes(tmp_path):
    prepare = _load("prepare_isaacs_complete_eq27_job.py")
    post = _load("postprocess_isaacs_complete_eq27.py")
    config_path = tmp_path / "complete.json"
    derived, _ = prepare.build(json.loads(prepare.BASE.read_text(encoding="utf-8")))
    config_path.write_text(json.dumps(derived), encoding="utf-8")
    npz_path = tmp_path / "candidate.npz"
    np.savez(npz_path)
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "status": "failed",
        "exit_code": False,
        "propagation_invocations": 2,
        "slurm_job_id": " ",
        "execution_git_sha": "wrong",
        "config_sha256": "wrong",
        "npz_sha256": "wrong",
    }), encoding="utf-8")
    result = post.validate(
        npz_path, config_path, metadata_path, expected_execution_sha="expected",
    )
    assert result["passed"] is False
    failures = "\n".join(result["failures"])
    assert "status is not completed" in failures
    assert "exit_code is not zero" in failures
    assert "propagation_invocations is not exactly 1" in failures
    assert "lacks a non-empty slurm job id" in failures
    assert "execution_git_sha does not exactly match" in failures
    assert "config_sha256 does not match" in failures
    assert "npz_sha256 does not match" in failures


def test_compare_contract_writes_summary_and_keeps_invalid_jobs_out_of_classification(tmp_path, monkeypatch):
    module = _load("compare_isaacs_complete_eq27.py")
    post = _load("postprocess_isaacs_complete_eq27.py")
    fallback = _load("prepare_isaacs_eq27_fallback_comparator_audit.py")
    x = np.linspace(-10.0, 5.0, 121)
    current = 1.2e22 * np.exp(-((x + 3.0) / 2.0) ** 2) + 1.0e18
    off = 1.1e22 * np.exp(-((x + 2.5) / 2.0) ** 2) + 1.0e18
    candidate = 1.2e22 * np.exp(-((x + 1.5) / 2.0) ** 2) + 1.0e18
    current_audit, off_audit, _, _ = _write_fixed_fallback_audits(tmp_path, current, off, fallback, module)
    current_axial = tmp_path / "fallback_audits" / "current_full_eq27_fallback_axial_diagnostics.csv"
    current_extras = tmp_path / "fallback_audits" / "current_full_eq27_fallback_raman_extras.csv"
    off_axial = tmp_path / "fallback_audits" / "raman_off_fallback_axial_diagnostics.csv"
    off_extras = tmp_path / "fallback_audits" / "raman_off_fallback_raman_extras.csv"
    candidate_audit, candidate_axial, candidate_extras = _write_candidate_chain(tmp_path, candidate, post)

    pycap = tmp_path / "pycap.csv"
    pycap.write_text(
        "x_focus_cm,rho_1e16_cm3\n" + "\n".join(f"{xx},{rr / 1e22}" for xx, rr in zip(x, candidate)) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "FIXED_PYCAP_PATH", pycap)
    monkeypatch.setattr(module, "FIXED_PYCAP_SHA256", _sha(pycap))
    out_dir = tmp_path / "comparison_123"
    summary = module.compare(
        current_axial, current_extras,
        off_axial, off_extras,
        candidate_axial, candidate_extras,
        pycap, out_dir,
        current_audit, off_audit, candidate_audit,
    )
    assert summary["comparator_provenance"]["current_full_eq27"]["job_id"] == "180748"
    assert summary["comparator_provenance"]["raman_off"]["job_id"] == "180749"
    assert summary["comparator_provenance"]["class"] == "fallback_verified_non_strict"
    assert summary["comparator_provenance"]["excluded_invalid_jobs"] == ["179706", "179988"]
    assert (out_dir / "comparison_summary.json").is_file()
    assert (out_dir / "comparison_report.md").is_file()
    assert "fallback_verified_non_strict" in (out_dir / "comparison_report.md").read_text(encoding="utf-8")


def test_compare_rejects_arbitrary_invalid_csv_or_audit(tmp_path, monkeypatch):
    module = _load("compare_isaacs_complete_eq27.py")
    post = _load("postprocess_isaacs_complete_eq27.py")
    fallback = _load("prepare_isaacs_eq27_fallback_comparator_audit.py")
    x = np.linspace(-10.0, 5.0, 121)
    rho = 1.2e22 * np.exp(-((x + 2.0) / 2.0) ** 2) + 1.0e18
    current_audit, off_audit, _, _ = _write_fixed_fallback_audits(tmp_path, rho, rho, fallback, module)
    current_axial = tmp_path / "fallback_audits" / "current_full_eq27_fallback_axial_diagnostics.csv"
    current_extras = tmp_path / "fallback_audits" / "current_full_eq27_fallback_raman_extras.csv"
    off_axial = tmp_path / "fallback_audits" / "raman_off_fallback_axial_diagnostics.csv"
    off_extras = tmp_path / "fallback_audits" / "raman_off_fallback_raman_extras.csv"
    candidate_audit, candidate_axial, candidate_extras = _write_candidate_chain(tmp_path, rho, post)
    pycap = tmp_path / "pycap.csv"
    pycap.write_text(
        "x_focus_cm,rho_1e16_cm3\n" + "\n".join(f"{xx},{rr / 1e22}" for xx, rr in zip(x, rho)) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "FIXED_PYCAP_PATH", pycap)
    monkeypatch.setattr(module, "FIXED_PYCAP_SHA256", _sha(pycap))

    invalid_csv = tmp_path / "invalid_axial.csv"
    invalid_csv.write_text("x_focus_cm,rho_max_z,I_max_z\n0,n/a,1\n1,2,3\n", encoding="utf-8")
    invalid_audit_payload = json.loads(current_audit.read_text(encoding="utf-8"))
    invalid_audit_payload["artifacts"]["axial"] = {"path": str(invalid_csv), "sha256": _sha(invalid_csv)}
    invalid_audit_payload["axial"] = invalid_audit_payload["artifacts"]["axial"]
    invalid_audit = tmp_path / "invalid_audit.json"
    invalid_audit.write_text(json.dumps(invalid_audit_payload), encoding="utf-8")
    with pytest.raises(module.InsufficientEvidenceError, match="insufficient_evidence"):
        module.compare(
            invalid_csv, current_extras,
            off_axial, off_extras,
            candidate_axial, candidate_extras,
            pycap, tmp_path / "invalid_csv_out",
            invalid_audit, off_audit, candidate_audit,
        )

    failed_payload = json.loads(current_audit.read_text(encoding="utf-8"))
    failed_payload["gate"] = "failed"
    failed_audit = tmp_path / "failed_audit.json"
    failed_audit.write_text(json.dumps(failed_payload), encoding="utf-8")
    with pytest.raises(module.InsufficientEvidenceError, match="status/gate is not passed"):
        module.compare(
            current_axial, current_extras,
            off_axial, off_extras,
            candidate_axial, candidate_extras,
            pycap, tmp_path / "invalid_audit_out",
            failed_audit, off_audit, candidate_audit,
        )


def test_compare_stops_on_missing_crossing_or_numerical_evidence(tmp_path, monkeypatch):
    module = _load("compare_isaacs_complete_eq27.py")
    post = _load("postprocess_isaacs_complete_eq27.py")
    fallback = _load("prepare_isaacs_eq27_fallback_comparator_audit.py")
    x = np.linspace(-10.0, 5.0, 121)
    good_rho = 1.2e22 * np.exp(-((x + 2.0) / 2.0) ** 2) + 1.0e18
    low_rho = 1.0e18 * np.ones_like(x)
    current_audit, off_audit, _, _ = _write_fixed_fallback_audits(tmp_path, good_rho, good_rho, fallback, module)
    current_axial = tmp_path / "fallback_audits" / "current_full_eq27_fallback_axial_diagnostics.csv"
    current_extras = tmp_path / "fallback_audits" / "current_full_eq27_fallback_raman_extras.csv"
    off_axial = tmp_path / "fallback_audits" / "raman_off_fallback_axial_diagnostics.csv"
    off_extras = tmp_path / "fallback_audits" / "raman_off_fallback_raman_extras.csv"
    candidate_audit, candidate_axial, candidate_extras = _write_candidate_chain(tmp_path, low_rho, post)
    pycap = tmp_path / "pycap.csv"
    pycap.write_text(
        "x_focus_cm,rho_1e16_cm3\n" + "\n".join(f"{xx},{rr / 1e22}" for xx, rr in zip(x, good_rho)) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "FIXED_PYCAP_PATH", pycap)
    monkeypatch.setattr(module, "FIXED_PYCAP_SHA256", _sha(pycap))
    with pytest.raises(module.InsufficientEvidenceError, match="crossing"):
        module.compare(
            current_axial, current_extras,
            off_axial, off_extras,
            candidate_axial, candidate_extras,
            pycap, tmp_path / "missing_crossing_out",
            current_audit, off_audit, candidate_audit,
        )

    incomplete_audit, incomplete_axial, incomplete_extras = _write_candidate_chain(
        tmp_path, good_rho, post, include_numerical=False,
    )
    with pytest.raises(module.InsufficientEvidenceError, match="numerical evidence"):
        module.compare(
            current_axial, current_extras,
            off_axial, off_extras,
            incomplete_axial, incomplete_extras,
            pycap, tmp_path / "missing_numerical_out",
            current_audit, off_audit, incomplete_audit,
        )


def test_candidate_postprocess_and_fallback_audits_feed_compare(tmp_path, monkeypatch):
    post = _load("postprocess_isaacs_complete_eq27.py")
    fallback = _load("prepare_isaacs_eq27_fallback_comparator_audit.py")
    compare = _load("compare_isaacs_complete_eq27.py")
    x = np.linspace(-10.0, 5.0, 121)
    current = 1.2e22 * np.exp(-((x + 3.0) / 2.0) ** 2) + 1.0e18
    off = 1.1e22 * np.exp(-((x + 2.5) / 2.0) ** 2) + 1.0e18
    candidate = 1.2e22 * np.exp(-((x + 1.5) / 2.0) ** 2) + 1.0e18
    current_audit, off_audit, _, _ = _write_fixed_fallback_audits(tmp_path, current, off, fallback, compare)
    current_axial = tmp_path / "fallback_audits" / "current_full_eq27_fallback_axial_diagnostics.csv"
    current_extras = tmp_path / "fallback_audits" / "current_full_eq27_fallback_raman_extras.csv"
    off_axial = tmp_path / "fallback_audits" / "raman_off_fallback_axial_diagnostics.csv"
    off_extras = tmp_path / "fallback_audits" / "raman_off_fallback_raman_extras.csv"
    candidate_audit, candidate_axial, candidate_extras = _write_candidate_chain(tmp_path, candidate, post)
    candidate_payload = json.loads(candidate_audit.read_text(encoding="utf-8"))
    assert candidate_payload["artifacts"]["axial"]["sha256"] == _sha(candidate_axial)
    assert candidate_payload["artifacts"]["extras"]["sha256"] == _sha(candidate_extras)
    pycap = tmp_path / "pycap.csv"
    pycap.write_text(
        "x_focus_cm,rho_1e16_cm3\n" + "\n".join(f"{xx},{rr / 1e22}" for xx, rr in zip(x, candidate)) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(compare, "FIXED_PYCAP_PATH", pycap)
    monkeypatch.setattr(compare, "FIXED_PYCAP_SHA256", _sha(pycap))
    summary = compare.compare(
        current_axial, current_extras,
        off_axial, off_extras,
        candidate_axial, candidate_extras,
        pycap, tmp_path / "comparison",
        current_audit,
        off_audit,
        candidate_audit,
    )
    assert summary["evidence_gate"] == "passed"
    assert summary["comparator_provenance"]["current_full_eq27"]["job_id"] == "180748"


def test_fallback_comparator_audit_rejects_alternate_inputs_and_tampered_raw_chain(tmp_path, monkeypatch):
    fallback = _load("prepare_isaacs_eq27_fallback_comparator_audit.py")
    rho = np.ones(8) * 1.2e22
    expected = _write_fixed_raw_case(tmp_path, "current_full_eq27", rho, fallback, _load("compare_isaacs_complete_eq27.py"))
    with pytest.raises(fallback.FallbackAuditError, match="SHA256 mismatch"):
        fallback.FIXED_RAW_EVIDENCE["current_full_eq27"]["npz_sha256"] = "0" * 64
        fallback.audit_comparator(role="current_full_eq27", out_dir=tmp_path / "bad")
    fallback.FIXED_RAW_EVIDENCE["current_full_eq27"] = expected
    with pytest.raises(fallback.FallbackAuditError, match="unsupported fallback role"):
        fallback.audit_comparator(role="arbitrary", out_dir=tmp_path / "bad_role")
    source = tmp_path / "fake.csv"
    source.write_text("x_focus_cm,rho_max_z\n0,1\n1,2\n", encoding="utf-8")
    with pytest.raises(TypeError):
        fallback.audit_comparator(role="current_full_eq27", out_dir=tmp_path / "bad_csv", axial=source)


def test_fallback_scheduler_evidence_requires_completed_zero_exit(monkeypatch):
    fallback = _load("prepare_isaacs_eq27_fallback_comparator_audit.py")

    class Result:
        stdout = "180748|RUNNING|0:0|00:01:00|g0609|s|b|e\n"

    monkeypatch.setattr(fallback.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(fallback.FallbackAuditError, match="not an admitted completed run"):
        fallback._scheduler_evidence("180748")


def test_c1_artifact_and_ancestor_tamper_stop_before_lock(tmp_path, monkeypatch):
    module = _load("create_isaacs_complete_eq27_execution_lock.py")
    manifest = json.loads(module.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    monkeypatch.setattr(module, "_git", lambda *args: "")
    tampered = dict(manifest)
    tampered["c1_gate"] = dict(tampered["c1_gate"], summary_sha256="0" * 64)
    with pytest.raises(module.ExecutionLockError, match="c1_gate binding"):
        module._validate_c1_gate(tampered, module.C1_COMMIT)
    tampered = dict(manifest, parent_c1_commit="0" * 40)
    with pytest.raises(module.ExecutionLockError, match="parent_c1_commit"):
        module._validate_c1_gate(tampered, module.C1_COMMIT)
    monkeypatch.setattr(module, "_git", lambda *args: (_ for _ in ()).throw(module.ExecutionLockError("not ancestor")))
    with pytest.raises(module.ExecutionLockError, match="not an ancestor"):
        module._validate_c1_gate(manifest, module.C1_COMMIT)


def test_fallback_compare_rechecks_live_scheduler_and_rejects_tampered_audit(tmp_path):
    fallback = _load("prepare_isaacs_eq27_fallback_comparator_audit.py")
    compare = _load("compare_isaacs_complete_eq27.py")
    _write_fixed_raw_case(tmp_path, "current_full_eq27", np.ones(8) * 1.2e22, fallback, compare)
    audit = fallback.audit_comparator(role="current_full_eq27", out_dir=tmp_path / "audit")
    audit["raw_source"]["scheduler_evidence"]["state"] = "COMPLETED"
    audit["raw_source"]["scheduler_evidence"]["exit_code"] = "0:1"
    compare._scheduler_evidence = lambda job_id: {
        "job_id": job_id, "state": "COMPLETED", "exit_code": "0:0", "source": "live_sacct",
    }
    failures = compare._fallback_raw_chain(audit, label="current_full_eq27", expected_job_id="180748")
    assert any("scheduler evidence" in item for item in failures)


def test_candidate_raw_chain_rejects_tampered_gpu_lock_and_job(tmp_path):
    post = _load("postprocess_isaacs_complete_eq27.py")
    compare = _load("compare_isaacs_complete_eq27.py")
    audit_path, axial, extras = _write_candidate_chain(tmp_path, np.ones(8) * 1.2e22, post)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    metadata_path = Path(payload["raw_source"]["metadata"]["path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["gpu_model"] = "tampered GPU"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    failures = compare._candidate_raw_chain(
        payload, audit_path=audit_path, expected_job_id="180800", axial=axial, extras=extras,
    )
    assert any("metadata SHA256" in item or "GPU model" in item for item in failures)

    audit_path, axial, extras = _write_candidate_chain(tmp_path / "lock", np.ones(8) * 1.2e22, post)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    lock_path = Path(payload["raw_source"]["execution_lock"]["path"])
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["use_raman_full_operator"] = False
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    failures = compare._candidate_raw_chain(
        payload, audit_path=audit_path, expected_job_id="180800", axial=axial, extras=extras,
    )
    assert any("execution lock SHA256" in item or "use_raman_full_operator" in item for item in failures)

    audit_path, axial, extras = _write_candidate_chain(tmp_path / "job", np.ones(8) * 1.2e22, post)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["job_id"] = "999999"
    failures = compare._candidate_raw_chain(
        payload, audit_path=audit_path, expected_job_id="999999", axial=axial, extras=extras,
    )
    assert any("job id" in item for item in failures)


def test_malformed_npz_length_mismatch_produces_failed_audit_not_index_error(tmp_path):
    post = _load("postprocess_isaacs_complete_eq27.py")
    config = post.FILAMENT_ROOT / "results" / "isaacs_complete_eq27" / "120fs_talebpour_isaacs_complete_eq27.json"
    npz = tmp_path / "malformed.npz"
    z = np.linspace(0.0, 1.3, 4)
    np.savez(npz, z_axis=z, raman_operator_applied=np.ones(3))
    result = post.validate(npz, config, expected_execution_sha="expected")
    assert result["passed"] is False
    assert any("raman_operator_applied is not z-aligned" in item for item in result["failures"])
    out = tmp_path / "audit"
    payload = post.write_audit(result, out, npz_path=npz, config_path=config)
    assert payload["gate"] == "failed"
    assert (out / "isaacs_complete_eq27_reaudit.json").is_file()


def test_direct_batch_and_submit_liveness_bindings_are_explicit_and_safe():
    batch = (ROOT / "tools" / "isaacs_complete_eq27_full.sbatch").read_text(encoding="utf-8")
    submit = (ROOT / "tools" / "submit_isaacs_complete_eq27_job.sh").read_text(encoding="utf-8")
    for token in (
        "validate_manifest_lock", "CONFIG_PATH is not the fixed", "submission lock", "global consumed",
        "use_raman_full_operator", "execution_lock_sha256", "propagation_invocations",
    ):
        assert token in batch
    assert 'SBATCH_STARTED=1' in submit
    assert 'ambiguous_sbatch_malformed_job_id' in submit
    assert 'RUN_OWNER_MARKER' in submit
    assert 'rmdir -- "${RUN_DIR}"' in submit
    assert 'rmdir -- "${GLOBAL_CONSUMED_LOCK}"' in submit
    assert 'rm -rf --' not in submit
    assert submit.index('mkdir -- "${RUN_DIR}"') < submit.index('mkdir -- "${GLOBAL_CONSUMED_LOCK}"')


def test_shared_preflight_rejects_non_ancestor_before_lock_validation(tmp_path, monkeypatch):
    module = _load("create_isaacs_complete_eq27_execution_lock.py")

    def fake_git(*args):
        if args == ("rev-parse", "HEAD"):
            return module.C1_COMMIT
        if args == ("status", "--porcelain=v1"):
            return ""
        if args[:2] == ("merge-base", "--is-ancestor"):
            raise module.ExecutionLockError("not ancestor")
        return ""

    monkeypatch.setattr(module, "_git", fake_git)
    with pytest.raises(module.ExecutionLockError, match="not an ancestor"):
        module.validate_manifest_lock(
            module.DEFAULT_MANIFEST,
            tmp_path / "unreached-lock.json",
            expected_git_sha=module.C1_COMMIT,
            require_clean=True,
        )


def test_held_receipt_release_and_direct_batch_bindings_are_explicit():
    submit = (ROOT / "tools" / "submit_isaacs_complete_eq27_job.sh").read_text(encoding="utf-8")
    batch = (ROOT / "tools" / "isaacs_complete_eq27_full.sbatch").read_text(encoding="utf-8")
    for token in (
        "sbatch --hold --parsable", "scontrol release", "job_receipt.json",
        "reservation_token", "JOB_RECEIPT_SHA256", "release_failure",
        "ambiguous_job_receipt_failure", "any held job are retained",
    ):
        assert token in submit
    for token in (
        'JOB_RECEIPT_PATH:?missing JOB_RECEIPT_PATH',
        'receipt_job_id != str(os.environ.get("SLURM_JOB_ID") or "").strip()',
        'receipt_token', 'job_receipt_sha256',
        'submission/global records must not be edited after sbatch',
    ):
        assert token in batch


def test_candidate_fake_lock_is_rejected_by_shared_c1_validator(tmp_path, monkeypatch):
    post = _load("postprocess_isaacs_complete_eq27.py")
    compare = _load("compare_isaacs_complete_eq27.py")
    audit_path, axial, extras = _write_candidate_chain(tmp_path, np.ones(8) * 1.2e22, post)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    create = _load("create_isaacs_complete_eq27_execution_lock.py")
    real_validator = create.validate_manifest_lock
    monkeypatch.setattr(compare, "validate_manifest_lock", real_validator)

    def fake_git(*args):
        if args == ("rev-parse", "HEAD"):
            return "d" * 40
        if args == ("status", "--porcelain=v1"):
            return ""
        if args[:2] == ("merge-base", "--is-ancestor"):
            raise create.ExecutionLockError("fixed C1 commit is not an ancestor")
        return ""

    monkeypatch.setattr(create, "_git", fake_git)
    failures = compare._candidate_raw_chain(
        payload, audit_path=audit_path, expected_job_id="180800", axial=axial, extras=extras,
    )
    assert any("shared manifest/C1 execution-lock validation failed" in item for item in failures)


def test_malformed_2d_z_axis_and_string_ir_write_failed_audit(tmp_path):
    post = _load("postprocess_isaacs_complete_eq27.py")
    config = post.FILAMENT_ROOT / "results" / "isaacs_complete_eq27" / "120fs_talebpour_isaacs_complete_eq27.json"
    npz = tmp_path / "malformed_2d_string_ir.npz"
    np.savez(
        npz,
        z_axis=np.ones((2, 2)),
        IR_max_z=np.asarray(["not-a-number", "still-not-a-number"]),
    )
    result = post.validate(npz, config, expected_execution_sha="expected")
    assert result["passed"] is False
    failures = "\n".join(result["failures"])
    assert "z_axis is not one-dimensional" in failures
    payload = post.write_audit(result, tmp_path / "audit", npz_path=npz, config_path=config)
    assert payload["gate"] == "failed"
    assert (tmp_path / "audit" / "isaacs_complete_eq27_reaudit.json").is_file()
    string_npz = tmp_path / "malformed_string_ir.npz"
    np.savez(string_npz, z_axis=np.linspace(0.0, 1.0, 2), IR_max_z=np.asarray(["bad", "worse"]))
    string_result = post.validate(string_npz, config, expected_execution_sha="expected")
    assert any("IR_max_z is not numeric" in item for item in string_result["failures"])


def test_postprocess_cap_hit_stops_candidate_admission(tmp_path, monkeypatch):
    post = _load("postprocess_isaacs_complete_eq27.py")
    real_validate = post.validate
    audit_path, _, _ = _write_candidate_chain(tmp_path, np.ones(8) * 1.2e22, post)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    raw_source = payload["raw_source"]
    npz_path = Path(raw_source["npz"]["path"])
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    arrays["I_max_z"] = np.full(arrays["I_max_z"].shape, post.EXPECTED_I_CAP)
    np.savez(npz_path, **arrays)
    metadata_path = Path(raw_source["metadata"]["path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["npz_sha256"] = _sha(npz_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(post, "validate", real_validate)
    monkeypatch.setattr(post, "validate_manifest_lock", lambda *args, **kwargs: {
        "config_path": Path(raw_source["config"]["path"]).resolve(),
    })
    result = post.validate(
        npz_path, Path(raw_source["config"]["path"]), metadata_path,
        expected_execution_sha="d" * 40,
        execution_lock_path=Path(raw_source["execution_lock"]["path"]),
        submission_lock_path=Path(raw_source["submission_lock"]["path"]),
        manifest_path=Path(raw_source["manifest"]["path"]),
        expected_execution_lock_sha256=raw_source["execution_lock"]["sha256"],
        job_receipt_path=Path(raw_source["job_receipt"]["path"]),
    )
    assert result["passed"] is False
    assert any("I_max_z reaches ionization.I_cap margin" in item for item in result["failures"])


def test_compare_cap_gate_rejects_cap_hit_in_all_three_cases():
    compare = _load("compare_isaacs_complete_eq27.py")
    series = {
        name: {"I_max_z": np.asarray([compare.EXPECTED_I_CAP])}
        for name in ("current_full_eq27", "raman_off", "candidate_complete_eq27")
    }
    failures = compare._case_cap_failures(series)
    assert len(failures) == 3
