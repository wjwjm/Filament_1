from __future__ import annotations

import numpy as np


def _run(tmp_path, enabled, split_order="after_other"):
    from KHz_filament.config import (
        BeamConfig,
        GridConfig,
        HeatConfig,
        IonizationConfig,
        PropagationConfig,
        RamanConfig,
        RunConfig,
    )
    from KHz_filament.runner import run_demo

    path = tmp_path / f"full_{'on' if enabled else 'off'}_{split_order}.npz"
    run_demo(
        grid=GridConfig(Nx=8, Ny=8, Nt=64, Lx=8e-4, Ly=8e-4, Twin=640e-15),
        beam=BeamConfig(
            w0=1.5e-4, tau_fwhm=120e-15, energy_J=1e-8,
            P0_peak=None, focal_length=None,
        ),
        prop=PropagationConfig(
            z_max=1e-5, dz=1e-5, linear_model="paraxial",
            auto_substep=False, focus_window_step=False,
            limit_focus_window=False, progress_every_z=0,
            energy_probe_every=0, diag_extra=False,
            use_self_steepening=False, use_electronic_kerr=False,
            use_raman_phase=False, use_raman_full_operator=enabled,
            use_raman_absorption=False, use_plasma_phase=False,
            use_ionization_loss=False, use_ionization_solver=False,
            measure_performance=True,
        ),
        ion=IonizationConfig(species=[]),
        heat=HeatConfig(),
        run=RunConfig(Npulses=1),
        raman=RamanConfig(
            enabled=True, model="isaacs_rot_sinexp", n_R=2.3e-23,
            omega_R=1.6e13, Gamma_R=1.3e13, T_R=None, T2=None,
            operator_mode="full_isaacs_eq27",
            operator_convention="isaacs_eq27",
            iir_sampling="exact_piecewise_linear",
            operator_integrator="heun", absorption=False,
            nonlinear_split_order=split_order,
        ),
        out_path=str(path), dtype="fp64",
    )
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def test_full_operator_on_wires_energy_and_reuses_two_convolutions(tmp_path):
    data = _run(tmp_path, True)
    assert data["raman_operator_mode"].item() == "full_isaacs_eq27"
    assert np.all(data["raman_operator_applied"])
    assert np.all(data["raman_convolution_count_step"] == 2)
    assert np.max(data["raman_rhs_l2_norm"]) > 0.0
    assert np.max(data["raman_IR_max_raw"]) > 0.0
    assert np.max(data["raman_target_loss_step_J"]) > 0.0
    assert np.max(data["raman_actual_loss_step_J"]) > 0.0
    assert np.max(data["raman_closure_residual_step"]) < 1e-3
    np.testing.assert_allclose(data["E_dep_rot_z"], data["raman_actual_loss_step_J"])
    assert np.all(data["alpha_R_applied_max_z"] == 0.0)
    assert data["delta_n_rot_applied_semantics"].item() == "not_applicable_full_complex_operator"
    assert np.all(data["total_walltime_step_s"] > 0.0)
    assert np.all(data["linear_walltime_step_s"] >= 0.0)
    assert np.all(data["ionization_walltime_step_s"] >= 0.0)


def test_full_operator_feedback_off_keeps_raw_diagnostics_without_field_loss(tmp_path):
    data = _run(tmp_path, False)
    assert not np.any(data["raman_operator_applied"])
    assert np.all(data["raman_convolution_count_step"] == 1)
    assert np.all(data["raman_rhs_l2_norm"] == 0.0)
    assert np.max(data["raman_IR_max_raw"]) > 0.0
    assert np.max(data["raman_target_loss_step_J"]) > 0.0
    assert np.all(data["raman_actual_loss_step_J"] == 0.0)
    assert np.all(data["E_dep_rot_z"] == 0.0)
    assert np.all(data["alpha_R_applied_max_z"] == 0.0)


def test_strang_uses_two_live_raman_substeps_and_four_total_convolutions(tmp_path):
    data = _run(tmp_path, True, "strang")
    assert np.all(data["raman_operator_substep_count"] == 2)
    assert np.all(data["raman_convolution_count_step"] == 4)
    assert np.max(data["raman_rhs_l2_norm"]) > 0.0
    assert np.max(data["raman_closure_residual_step"]) < 1e-3


def test_heun_stage_api_calls_convolution_exactly_twice(monkeypatch):
    import KHz_filament.raman as raman
    from KHz_filament.constants import c0, eps0

    calls = 0
    original = raman.raman_convolve_intensity

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(raman, "raman_convolve_intensity", counted)
    nt, dt = 256, 1e-15
    t = (np.arange(nt) - nt // 2) * dt
    intensity = 5e17 * np.exp(-4*np.log(2)*(t/120e-15)**2)
    field = np.sqrt(2*intensity/(eps0*c0*1.00027)).astype(complex)[:, None, None]
    omega = 2*np.pi*np.fft.fftfreq(nt, dt)
    kwargs = dict(
        Omega=omega, dt=dt, omega0=2*np.pi*c0/800e-9, n0=1.00027,
        n_R=2.3e-23, omega_R=1.6e13, Gamma_R=1.3e13,
        iir_sampling="exact_piecewise_linear",
    )
    stage1 = raman.isaacs_raman_stage(field, **kwargs)
    _, diagnostics = raman.apply_isaacs_raman_operator_step(
        field, 1e-5, stage1=stage1, return_diagnostics=True, **kwargs)
    assert calls == 2
    assert diagnostics["convolution_count"] == 2
    assert diagnostics["rhs_l2_norm_stage1"] > 0.0
    assert diagnostics["rhs_l2_norm_stage2"] > 0.0
