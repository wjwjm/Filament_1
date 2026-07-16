from __future__ import annotations

import json
import sys
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
