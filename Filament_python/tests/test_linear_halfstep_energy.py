from __future__ import annotations

import numpy as np


def test_bk_halfstep_stage_energies_are_float64_and_telescope():
    from KHz_filament.linear import step_linear_bk_nee_factorized

    rng = np.random.default_rng(7)
    field = (rng.normal(size=(16, 6, 6)) + 1j * rng.normal(size=(16, 6, 6))).astype(np.complex64)
    omega = 2 * np.pi * np.fft.fftfreq(16, 2e-15)
    kx = 2 * np.pi * np.fft.fftfreq(6, 1e-5)
    kperp2 = kx[None, :] ** 2 + kx[:, None] ** 2
    out, audit = step_linear_bk_nee_factorized(
        field, Omega=omega, kperp2=kperp2, k0=7e6,
        omega0=2*np.pi*3e8/800e-9, dz=5e-5,
        return_energy_diagnostics=True, energy_scale=1.0,
    )
    assert out.dtype == field.dtype
    for value in audit.values():
        assert np.isfinite(value)
    np.testing.assert_allclose(
        audit["energy_after_forward_fft_J"], audit["energy_before_J"], rtol=1e-7, atol=0.0)
    np.testing.assert_allclose(
        audit["energy_after_inverse_fft_J"], audit["energy_after_transfer_J"], rtol=1e-7, atol=0.0)


def test_opt_in_linear_halfstep_diagnostics_do_not_modify_field_and_are_accounted(tmp_path):
    from KHz_filament.config import BeamConfig, GridConfig, HeatConfig, IonizationConfig, PropagationConfig, RamanConfig, RunConfig
    from KHz_filament.runner import run_demo

    common = dict(
        grid=GridConfig(Nx=8, Ny=8, Nt=32, Lx=8e-4, Ly=8e-4, Twin=320e-15),
        beam=BeamConfig(w0=1.5e-4, tau_fwhm=120e-15, energy_J=1e-8, P0_peak=None, focal_length=None),
        ion=IonizationConfig(species=[]), heat=HeatConfig(), run=RunConfig(Npulses=1),
        raman=RamanConfig(enabled=False), dtype="fp32",
    )
    base = dict(z_max=2e-5, dz=1e-5, linear_model="bk_nee", auto_substep=False,
                focus_window_step=False, limit_focus_window=False, progress_every_z=0,
                energy_probe_every=0, diag_extra=False, use_self_steepening=False,
                use_electronic_kerr=False, use_raman_phase=False, use_raman_full_operator=False,
                use_raman_absorption=False, use_plasma_phase=False, use_ionization_loss=False,
                use_ionization_solver=False)
    off = tmp_path / "off.npz"
    on = tmp_path / "on.npz"
    run_demo(prop=PropagationConfig(**base), out_path=str(off), **common)
    run_demo(prop=PropagationConfig(**base, diag_linear_halfstep_energy=True), out_path=str(on), **common)
    with np.load(off, allow_pickle=False) as d0, np.load(on, allow_pickle=False) as d1:
        np.testing.assert_array_equal(d0["U_z"], d1["U_z"])
        assert d1["linear_halfstep_energy_diagnostics_enabled"].item()
        for half in (1, 2):
            before = d1[f"linear_halfstep_{half}_energy_before_J"]
            after = d1[f"linear_halfstep_{half}_energy_after_J"]
            delta = d1[f"linear_halfstep_{half}_field_delta_J"]
            residual = d1[f"linear_halfstep_{half}_unaccounted_residual_J"]
            assert before.dtype == np.float64
            np.testing.assert_allclose(delta, after - before, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(residual, delta, rtol=0.0, atol=0.0)
            assert np.all(np.isfinite(d1[f"linear_halfstep_{half}_energy_after_transfer_J"]))
