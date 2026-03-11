import json
import pytest

pytest.importorskip("numpy")
from KHz_filament.confio import load_all


def _base_cfg():
    return {
        "grid": {"Nx": 16, "Ny": 16, "Nt": 16, "Lx": 1e-3, "Ly": 1e-3, "Twin": 1e-13},
        "beam": {
            "lam0": 8e-7,
            "n0": 1.00027,
            "w0": 2e-4,
            "tau_fwhm": 50e-15,
            "E0_peak": 0.0,
            "energy_J": None,
            "I0_peak": None,
            "focal_length": 0.3,
        },
        "propagation": {"z_max": 1e-3, "dz": 1e-4},
        "ionization": {},
        "heat": {},
        "run": {},
        "raman": {},
    }


def test_beam_derive_from_i0_peak(tmp_path):
    cfg = _base_cfg()
    cfg["beam"]["I0_peak"] = 1e16
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    _, beam, *_ = load_all(str(p))
    assert beam.E0_peak > 0.0


def test_beam_energy_i0_mutually_exclusive(tmp_path):
    cfg = _base_cfg()
    cfg["beam"]["energy_J"] = 1e-3
    cfg["beam"]["I0_peak"] = 1e16
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="mutually exclusive"):
        load_all(str(p))
