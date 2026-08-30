from __future__ import annotations

import numpy as np


def test_ciddor_simple_refractivity_has_physical_density_scaling():
    from KHz_filament.air_dispersion import n_air_ciddor_simple, n_of_omega
    from KHz_filament.constants import c0

    wavelength_um = 0.8
    pressure_pa = 101325.0
    temperature_k = 293.15
    n = float(n_air_ciddor_simple(wavelength_um, P=pressure_pa, T=temperature_k))
    omega = 2.0 * np.pi * c0 / (wavelength_um * 1e-6)
    n_from_omega = float(n_of_omega(omega, P=pressure_pa, T=temperature_k))

    assert n > 1.0
    assert 1e-4 < n - 1.0 < 1e-3
    np.testing.assert_allclose(n_from_omega, n, rtol=0.0, atol=1e-15)

    n_low_pressure = float(n_air_ciddor_simple(wavelength_um, P=0.5 * pressure_pa, T=temperature_k))
    n_high_temperature = float(n_air_ciddor_simple(wavelength_um, P=pressure_pa, T=1.1 * temperature_k))
    assert n_low_pressure - 1.0 < n - 1.0
    assert n_high_temperature - 1.0 < n - 1.0


def test_uppe_linear_advance_smoke_preserves_field_norm():
    from KHz_filament.config import BeamConfig, PropagationConfig
    from KHz_filament.constants import c0
    from KHz_filament.grids import make_axes
    from KHz_filament import runner

    axes = make_axes(4, 4, 8, 4e-4, 4e-4, 80e-15)
    beam = BeamConfig(lam0=800e-9)
    prop = PropagationConfig(linear_model="uppe")
    field = np.ones((8, 4, 4), dtype=np.complex128)
    k0 = beam.n0 * 2.0 * np.pi * c0 / beam.lam0

    advanced = runner._linear_advance(
        field, 1e-4, axes=axes, kperp2=axes.kperp2, k0=k0, prop=prop, beam=beam,
    )

    assert np.isfinite(advanced).all()
    np.testing.assert_allclose(np.linalg.norm(advanced), np.linalg.norm(field), rtol=1e-12)
