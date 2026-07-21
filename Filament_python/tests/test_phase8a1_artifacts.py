from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "isaacs_raman_closure" / "phase8a1_production_closure"


def test_phase8a1_task1_through_task6_artifacts_are_present_and_nonempty():
    required = (
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
    )
    missing = [name for name in required if not (OUT / name).is_file()]
    empty = [name for name in required if (OUT / name).is_file() and (OUT / name).stat().st_size == 0]
    assert not missing, f"missing Phase 8A.1 artifacts: {missing}"
    assert not empty, f"empty Phase 8A.1 artifacts: {empty}"
