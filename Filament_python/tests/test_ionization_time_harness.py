from __future__ import annotations

import json
import sys
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import validate_ionization_time_integrator as harness


def _config(path: Path, tau_fwhm_s: float) -> Path:
    value = {
        "grid": {"Nx": 8, "Ny": 8, "Nt": 32, "Lx": 1e-3, "Ly": 1e-3, "Twin": 320e-15},
        "beam": {"lam0": 800e-9, "n0": 1.00027, "tau_fwhm": tau_fwhm_s, "energy_J": 1e-9, "P0_peak": None},
        "ionization": {
            "time_mode": "full",
            "integrator": "rk4",
            "beta_rec": 0.0,
            "species": [
                {"name": "N2", "rate": "mpa_fact", "ell": 2, "I_mp": 1e18, "Ip_eV": 15.6, "fraction": 0.8},
                {"name": "O2", "rate": "mpa_fact", "ell": 2, "I_mp": 8e17, "Ip_eV": 12.1, "fraction": 0.2},
            ],
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_harness_writes_required_0d_series_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "_git_sha", lambda: "d" * 40)
    config_40 = _config(tmp_path / "40fs.json", 40e-15)
    config_120 = _config(tmp_path / "120fs.json", 120e-15)
    metadata = harness.run_0d_ionization_harness([config_40, config_120], [1e16, 1e17], tmp_path / "output")

    assert metadata["temporal_convention"]["tau_fwhm_interpretation"] == "intensity FWHM"
    assert len(metadata["cases"]) == 4
    assert (tmp_path / "output" / "ionization_integrator_cases.csv").is_file()
    with np.load(tmp_path / "output" / "ionization_integrator_timeseries.npz", allow_pickle=False) as data:
        case_id = str(data["case_ids"][0])
        for suffix in ("t_s", "I_W_m2", "W_N2_s-1", "W_O2_s-1", "rho_N2_m3", "rho_O2_m3", "rho_total_m3"):
            assert f"{case_id}__{suffix}" in data
        assert np.all(data[f"{case_id}__I_W_m2"] >= 0.0)
        assert np.all(data[f"{case_id}__rho_total_m3"] >= 0.0)


def test_comparison_recomputes_refined_grids_and_reports_preclip_stability(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "_git_sha", lambda: "e" * 40)
    config = _config(tmp_path / "40fs.json", 40e-15)
    metadata = harness.run_integrator_comparison([config], [1e16], tmp_path / "comparison")
    assert metadata["refinements"] == [1, 2, 4, 8]
    with (tmp_path / "comparison" / "ionization_integrator_error_summary.csv").open(encoding="utf-8", newline="") as handle:
        errors = list(csv.DictReader(handle))
    assert {int(row["refinement_factor"]) for row in errors} == {1, 2, 4, 8}
    assert {row["species"] for row in errors} == {"N2", "O2", "total"}
    assert all(np.isfinite(float(row["max_W_dt"])) for row in errors)
    assert "preclip_step_max" in errors[0]
    with np.load(tmp_path / "comparison" / "ionization_integrator_timeseries.npz", allow_pickle=False) as data:
        key = str(errors[0]["case_label"])
        assert f"{key}__rho_N2_rk4_f8_m3" in data
        assert f"{key}__rho_total_reference_f8_m3" in data


def test_stability_diagnostic_does_not_change_production_rk4_solution(tmp_path):
    config = _config(tmp_path / "40fs.json", 40e-15)
    plain = harness.run_production_0d_case(config, 1e17, diagnose_integrator_stability=False)
    diagnosed = harness.run_production_0d_case(config, 1e17, diagnose_integrator_stability=True)
    np.testing.assert_allclose(plain.rho_total_m3, diagnosed.rho_total_m3, rtol=0.0, atol=0.0)
    assert diagnosed.stability_by_species is not None
    assert set(diagnosed.stability_by_species) == {"N2", "O2"}


def test_classifier_uses_quantitative_error_and_preclip_gates():
    cases = [{"case_label": "case", "final_ionization_fraction": "1e-3"}]
    baseline = {
        "case_label": "case", "species": "total", "refinement_factor": "1", "I_peak_W_m2": "1e17",
        "rho_final_rel_error": "0.005", "rho_time_max_rel_error": "0.006", "rise_time_error_fs": "0.2",
        "step_clip_count": "0", "intermediate_violation_count": "0",
    }
    assert harness.classify_integrator_evidence(cases, [baseline])["classification"] == "not_supported"
    severe = dict(baseline, rho_time_max_rel_error="0.06")
    assert harness.classify_integrator_evidence(cases, [severe])["classification"] == "supported"
    clipped = dict(baseline, step_clip_count="1")
    assert harness.classify_integrator_evidence(cases, [clipped])["classification"] == "supported"
