from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "isaacs_raman_closure" / "phase8b_preflight"
REQUIRED_GATES = {
    "baseline_config_lock_gate",
    "on_off_single_factor_gate",
    "explicit_operator_switch_gate",
    "legacy_absorption_rejection_gate",
    "full_operator_diagnostic_wiring_gate",
    "raman_energy_accounting_gate",
    "convolution_reuse_gate",
    "combined_split_convergence_gate",
    "combined_split_production_step_gate",
    "full_size_smoke_gate",
    "memory_gate",
    "runtime_gate",
    "expected_diagnostic_contract_gate",
    "full_job_submission_gate",
}


def _module():
    path = ROOT / "tools" / "finalize_phase8b_preflight.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_threshold_gate_enforces_finite_numeric_comparison():
    module = _module()
    kwargs = dict(evidence="x", physical_impact="x", production_impact="x", required_action="x")
    assert module._threshold_gate(0.049, 1e-10, **kwargs)["status"] == "failed"
    assert module._threshold_gate(0.005, 0.01, **kwargs)["status"] == "passed"
    assert module._threshold_gate(float("nan"), 0.01, **kwargs)["status"] == "inconclusive"


def test_recorded_preflight_evidence_passes_all_required_gates():
    module = _module()
    gates, meta = module.build_gates(RESULTS, full_pytest_passed=True, full_pytest_summary="synthetic test pass")
    assert set(gates) == REQUIRED_GATES
    assert all(gate["status"] == "passed" for gate in gates.values())
    assert meta["full_production_jobs_submitted"] == 0
    for gate in gates.values():
        assert set(gate) == set(module.GATE_FIELDS)


def test_full_pytest_failure_blocks_full_job_submission():
    module = _module()
    gates, _ = module.build_gates(RESULTS, full_pytest_passed=False, full_pytest_summary="one failure")
    assert gates["full_job_submission_gate"]["status"] == "failed"


def test_recorded_step_closure_p99_is_a_hard_submission_gate(tmp_path):
    import json
    import shutil

    module = _module()
    copied = tmp_path / "preflight"
    shutil.copytree(RESULTS, copied)
    smoke_path = copied / "phase8b_full_size_smoke_metrics.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["on"]["raman_step_closure_p99"] = 0.01772617394104599
    smoke_path.write_text(json.dumps(smoke, indent=2) + "\n", encoding="utf-8")
    gates, _ = module.build_gates(
        copied, full_pytest_passed=True, full_pytest_summary="synthetic test pass")
    assert gates["raman_energy_accounting_gate"]["status"] == "failed"
    assert gates["full_job_submission_gate"]["status"] == "failed"
    result = gates["full_job_submission_gate"]["numerical_result"]
    assert result["raman_step_closure_p99"] == 0.01772617394104599
    assert result["raman_step_closure_p99_threshold"] == 1e-3
    assert result["raman_step_closure_p99_below_contract"] is False
