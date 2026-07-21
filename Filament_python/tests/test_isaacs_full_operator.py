from __future__ import annotations

import pathlib
import sys
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from KHz_filament.config_normalize import normalize_config
from KHz_filament.constants import c0, eps0
from KHz_filament.raman import apply_isaacs_raman_operator_step, isaacs_raman_field_rhs


def _strict(mode="full_isaacs_eq27"):
    return {"propagation": {"use_raman_absorption": False}, "raman": {
        "model": "isaacs_rot_sinexp", "n_R": 2.3e-23, "omega_R": 1.6e13,
        "Gamma_R": 1.3e13, "operator_convention": "isaacs_eq27",
        "iir_sampling": "exact_piecewise_linear", "operator_mode": mode,
        "absorption": False,
    }}


def test_full_operator_configuration_is_opt_in_and_rejects_legacy_absorption():
    assert normalize_config(_strict())["raman"]["operator_mode"] == "full_isaacs_eq27"
    bad = _strict(); bad["propagation"]["use_raman_absorption"] = True
    with pytest.raises(ValueError, match="rejects legacy Raman absorption"):
        normalize_config(bad)


def test_split_energy_closed_rejects_legacy_clipping():
    bad = _strict("split_energy_closed"); bad["raman"]["absorption_model"] = "conv_deriv"
    with pytest.raises(ValueError, match="rejects legacy conv_deriv"):
        normalize_config(bad)


def test_full_operator_rhs_and_heun_are_finite_and_recompute_stage():
    nt, dt = 1024, .5e-15
    t = (np.arange(nt) - nt // 3) * dt
    omega = 2 * np.pi * np.fft.fftfreq(nt, dt)
    intensity = 5e17 * np.exp(-4 * np.log(2) * (t / 120e-15) ** 2)
    field = np.sqrt(2 * intensity / (eps0 * c0 * 1.00027)).astype(complex)[:, None, None]
    args = dict(Omega=omega, dt=dt, omega0=2*np.pi*c0/800e-9, n0=1.00027,
                n_R=2.3e-23, omega_R=1.6e13, Gamma_R=1.3e13)
    rhs = np.asarray(isaacs_raman_field_rhs(field, **args))
    euler = np.asarray(apply_isaacs_raman_operator_step(field, 2e-5, integrator="euler", **args))
    heun = np.asarray(apply_isaacs_raman_operator_step(field, 2e-5, integrator="heun", **args))
    assert np.isfinite(rhs).all() and np.isfinite(heun).all()
    assert not np.array_equal(euler, heun)
