from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "isaacs_raman_closure" / "phase8a1_production_closure"


def test_phase8a1_required_artifacts_are_present_and_nonempty():
    required = (
        "gate_computation_correction.md",
        "time_derivative_convention.json",
        "time_derivative_validation.csv",
        "time_derivative_validation.png",
        "raman_fft_direct_comparison.csv",
        "raman_iir_direct_convergence.csv",
        "eq10_eq11_validation_v2.csv",
        "eq10_eq11_convergence_v2.csv",
        "isaacs_operator_prefactor_derivation.md",
        "isaacs_operator_prefactor.json",
        "production_split_vs_full_operator.csv",
        "production_split_vs_full_operator.png",
        "production_operator_waveform_metrics.csv",
        "raman_local_energy_closure.csv",
        "raman_global_energy_closure.csv",
        "raman_dz_convergence.csv",
        "raman_energy_closure.png",
        "phase8a1_full_pytest_failures.txt",
        "phase8a1_gate_summary.json",
        "phase8a1_final_report.md",
        "phase8a1_changelog.md",
        "raman_architecture_decision_v2.md",
        "raman_architecture_decision_v2.json",
    )
    missing = [name for name in required if not (OUT / name).is_file()]
    empty = [name for name in required if (OUT / name).is_file() and (OUT / name).stat().st_size == 0]
    assert not missing, f"missing Phase 8A.1 artifacts: {missing}"
    assert not empty, f"empty Phase 8A.1 artifacts: {empty}"


def test_phase8a1_gate_bundle_has_complete_contract_and_admits_full_operator():
    required_gates = {
        "gate_generator_integrity_gate",
        "source_equation_mapping_gate",
        "parameter_boundary_gate",
        "configuration_ambiguity_gate",
        "time_derivative_sign_gate",
        "tdiff_fft_consistency_gate",
        "kernel_normalization_gate",
        "fft_linear_convolution_gate",
        "iir_convergence_gate",
        "eq10_signed_energy_gate",
        "eq11_analytic_recovery_gate",
        "operator_prefactor_gate",
        "production_split_comparison_gate",
        "full_operator_reference_gate",
        "no_double_counting_gate",
        "local_energy_closure_gate",
        "global_energy_closure_gate",
        "dz_convergence_gate",
        "full_pytest_gate",
        "propagation_admission_gate",
    }
    required_fields = {
        "status", "evidence", "numerical_result", "threshold",
        "comparison_operator", "physical_impact", "production_impact",
        "required_action",
    }
    gates = json.loads((OUT / "phase8a1_gate_summary.json").read_text(encoding="utf-8"))
    assert set(gates) == required_gates
    assert all(required_fields == set(item) for item in gates.values())
    assert all(item["status"] in {"passed", "failed", "inconclusive", "not_applicable"} for item in gates.values())
    assert gates["production_split_comparison_gate"]["status"] == "failed"
    assert gates["full_operator_reference_gate"]["status"] == "passed"
    assert gates["propagation_admission_gate"]["status"] == "passed"
    decision = json.loads((OUT / "raman_architecture_decision_v2.json").read_text(encoding="utf-8"))
    assert decision["selected_architecture"] == "ready_full_operator"
    assert decision["phase8b_executed"] is False
    assert decision["new_slurm_jobs_submitted"] == 0
