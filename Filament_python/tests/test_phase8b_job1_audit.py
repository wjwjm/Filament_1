from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "results" / "isaacs_raman_closure" / "phase8b_preflight"


def _module():
    path = ROOT / "tools" / "prepare_phase8b_job1_audit.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_job1_audit_passes_corrected_smoke_without_authorizing_submission(tmp_path):
    module = _module()
    artifacts = module.build_artifacts(
        baseline_path=module.BASELINE,
        config_path=module.JOB1_CONFIG,
        preflight_dir=PREFLIGHT,
        out_dir=tmp_path,
        generated_utc="2026-07-21T00:00:00+00:00",
    )
    assert artifacts["job1_config_diff.json"]["status"] == "passed"
    audit = artifacts["job1_input_audit.json"]
    assert audit["status"] == "passed"
    assert audit["metrics"]["raman_step_closure_p99"] < 1e-3
    assert audit["metrics"]["raman_cumulative_closure_final"] < 5e-3
    assert audit["metrics"]["legacy_alpha_R_max"] == 0.0
    manifest = artifacts["job1_execution_manifest.json"]
    assert manifest["full_job_submitted"] is False
    assert manifest["production_propagation_executed"] is False
    assert manifest["slurm_job_id"] is None
    assert manifest["future_execution_git_sha"] is None
    contract = artifacts["job1_expected_diagnostic_contract.json"]
    assert contract["execution_policy"]["full_job_submission_authorized_now"] is False
    assert contract["raman_energy_contract"]["per_step_p99_lt"] == 1e-3


def test_job1_audit_fails_if_recorded_p99_regresses(tmp_path):
    module = _module()
    copied = tmp_path / "preflight"
    copied.mkdir()
    for path in PREFLIGHT.iterdir():
        if path.is_file():
            (copied / path.name).write_bytes(path.read_bytes())
    smoke_path = copied / "phase8b_full_size_smoke_metrics.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["on"]["raman_step_closure_p99"] = 0.01772617394104599
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    artifacts = module.build_artifacts(
        baseline_path=module.BASELINE,
        config_path=module.JOB1_CONFIG,
        preflight_dir=copied,
        out_dir=tmp_path / "audit",
        generated_utc="2026-07-21T00:00:00+00:00",
    )
    audit = artifacts["job1_input_audit.json"]
    assert audit["status"] == "failed"
    assert audit["checks"]["smoke_p99_below_contract"] is False
