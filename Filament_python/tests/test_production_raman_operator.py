from __future__ import annotations

import csv
import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import compare_production_raman_operator as comparator


def test_production_comparator_invokes_real_split_and_full_product(tmp_path):
    out = tmp_path / "operator"
    subprocess.run([sys.executable, str(ROOT / "tools" / "compare_production_raman_operator.py"), "--out-dir", str(out)], check=True)
    rows = list(csv.DictReader((out / "production_split_vs_full_operator.csv").open()))
    assert len(rows) == 6
    assert all("shock_intensity" in row["production_split_update"] for row in rows)
    assert all(row["full_reference_update"].startswith("Eq27 Heun") for row in rows)
    assert all(float(row["front_back_asymmetry_difference"]) == float(row["front_back_asymmetry_difference"]) for row in rows)
    prefactor = json.loads((out / "isaacs_operator_prefactor.json").read_text())
    assert prefactor["relative_difference"] < 1e-3
    assert prefactor["selected_candidate_prefactor"] == prefactor["full_reference_prefactor"]


def test_old_surrogate_comparator_has_no_production_gate_output():
    source = (ROOT / "tools" / "compare_isaacs_raman_operator.py").read_text()
    assert "production_split_vs_full_operator.csv" not in source


def test_actual_production_split_functions_are_invoked(monkeypatch):
    calls = {name: 0 for name in (
        "raman_convolve_intensity", "shock_intensity",
        "kerr_phase_from_deltan", "apply_nonlinear",
    )}
    for name in tuple(calls):
        original = getattr(comparator, name)

        def wrapper(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(comparator, name, wrapper)

    dt = 0.625e-15
    t = (np.arange(2048) - 1024) * dt
    field = comparator.pulse(t, "120fs_tl")
    result = comparator.production_split_step(field, 1e-5, dt)
    assert np.isfinite(result).all()
    assert all(count > 0 for count in calls.values())


def test_pulse_centered_asymmetry_is_finite_and_symmetric_gaussian_is_unity():
    dt = 0.3125e-15
    t = (np.arange(4096) - 2048) * dt
    symmetric = comparator.pulse(t, "120fs_tl")
    value = comparator.asymmetry(t, symmetric)
    assert np.isfinite(value) and value > 0.0
    assert abs(value - 1.0) < 0.02
