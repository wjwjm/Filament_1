from __future__ import annotations

import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from KHz_filament.raman_isaacs_reference import C0, boxcar_edge_signed_energy, eq11_alpha, isaacs_kernel, causal_convolution_direct, signed_energy_from_response


def test_eq10_boxcar_sign_and_eq11_recovery():
    n_R, omega, gamma, I0, tau = 2.3e-23, 1.6e13, 1.3e13, 5e17, 120e-15
    u = boxcar_edge_signed_energy(I0, tau, n_R=n_R, omega_R=omega, Gamma_R=gamma)
    alpha = eq11_alpha(I0, tau, n_R=n_R, omega_R=omega, Gamma_R=gamma)
    assert u < 0.0
    assert np.isclose(-u / (I0 * tau), alpha, rtol=1e-14)


def test_no_per_time_positive_clipping_is_used_for_corrected_quantity():
    dt = .25e-15
    t = np.arange(4096) * dt
    i = 5e17 * np.exp(-4 * np.log(2) * ((t - 800 * dt) / 120e-15) ** 2)
    r = causal_convolution_direct(i, isaacs_kernel(t, 1.6e13, 1.3e13), dt)
    energy = signed_energy_from_response(i, r, dt, n_R=2.3e-23)
    assert energy.q_R_positive == max(-energy.u_R_signed, 0.0)
    assert energy.legacy_clipped_result != energy.q_R_positive
