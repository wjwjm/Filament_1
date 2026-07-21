from __future__ import annotations

import numpy as np


def _run(
    tmp_path, enabled, split_order="after_other", dtype="fp64", *,
    energy_J=1e-8, dz=1e-5, diag_operator_energy=False,
):
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
            w0=1.5e-4, tau_fwhm=120e-15, energy_J=energy_J,
            P0_peak=None, focal_length=None,
        ),
        prop=PropagationConfig(
            z_max=dz, dz=dz, linear_model="paraxial",
            auto_substep=False, focus_window_step=False,
            limit_focus_window=False, progress_every_z=0,
            energy_probe_every=0, diag_extra=False,
            use_self_steepening=False, use_electronic_kerr=False,
            use_raman_phase=False, use_raman_full_operator=enabled,
            use_raman_absorption=False, use_plasma_phase=False,
            use_ionization_loss=False, use_ionization_solver=False,
            measure_performance=True, diag_operator_energy=diag_operator_energy,
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
        out_path=str(path), dtype=dtype,
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


def test_opt_in_operator_energy_diagnostics_resolve_every_strang_suboperator(tmp_path):
    data = _run(tmp_path, True, "strang", diag_operator_energy=True)
    keys = (
        "energy_step_start_J", "energy_after_linear_half1_J",
        "energy_after_raman_pre_J", "energy_after_nonraman_J",
        "energy_after_raman_post_J", "energy_after_linear_half2_J",
    )
    assert data["operator_energy_diagnostics_enabled"].item()
    for key in keys:
        assert data[key].dtype == np.float64
        assert data[key].shape == data["U_z"].shape
        assert np.all(np.isfinite(data[key]))
    np.testing.assert_allclose(data["energy_after_linear_half2_J"], data["U_z"])
    split_sum = (
        (data["energy_after_linear_half1_J"] - data["energy_step_start_J"])
        + (data["energy_after_raman_pre_J"] - data["energy_after_linear_half1_J"])
        + (data["energy_after_nonraman_J"] - data["energy_after_raman_pre_J"])
        + (data["energy_after_raman_post_J"] - data["energy_after_nonraman_J"])
        + (data["energy_after_linear_half2_J"] - data["energy_after_raman_post_J"])
    )
    np.testing.assert_allclose(split_sum, data["energy_after_linear_half2_J"] - data["energy_step_start_J"])
    u0 = float(data["U_z"][0] - data["U_step_change_z"][0])
    total_closure = abs((u0 - float(data["U_z"][-1])) - float(data["E_dep_cumulative_z"][-1])) / u0
    assert total_closure < 0.01


def test_float32_strang_reports_stable_energy_difference_boundary(tmp_path):
    data = _run(
        tmp_path, True, "strang", dtype="fp32",
        energy_J=1e-5, dz=1e-4)
    assert data["raman_actual_loss_evaluation"].item() == "stable_component_difference_float64"
    assert np.all(np.isfinite(data["raman_closure_residual_step"]))
    assert np.max(data["raman_actual_loss_step_J"]) > 0.0


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


def test_float32_field_loss_uses_stable_component_difference():
    from KHz_filament.raman import stable_field_fluence_and_loss
    from KHz_filament.constants import c0, eps0

    rng = np.random.default_rng(42)
    before = (
        rng.normal(size=(96, 12, 10))
        + 1j * rng.normal(size=(96, 12, 10))
    ).astype(np.complex64) * np.float32(2e7)
    phase = np.float32(2e-4)
    amplitude = np.float32(1.0 - 8e-7)
    after = (before * amplitude * np.exp(1j * phase)).astype(np.complex64)
    before_fluence, stable_loss = stable_field_fluence_and_loss(
        before, after, dt=1.25e-15, n0=1.00027, chunk_t=17)

    scale = 0.5 * eps0 * c0 * 1.00027 * 1.25e-15
    before128 = before.astype(np.complex128)
    after128 = after.astype(np.complex128)
    reference_before = scale * np.sum(np.abs(before128) ** 2, axis=0)
    reference_loss = scale * np.sum(
        np.abs(before128) ** 2 - np.abs(after128) ** 2, axis=0)
    np.testing.assert_allclose(before_fluence, reference_before, rtol=2e-14, atol=0.0)
    np.testing.assert_allclose(stable_loss, reference_loss, rtol=2e-9, atol=1e-30)


def test_float32_energy_projection_does_not_depend_on_diagnostic_request():
    from KHz_filament.constants import c0, eps0
    from KHz_filament.raman import apply_isaacs_raman_operator_step

    nt, dt = 192, 2.5e-15
    t = (np.arange(nt) - nt // 2) * dt
    intensity = 5e17 * np.exp(-4 * np.log(2) * (t / 120e-15) ** 2)
    field = np.sqrt(2 * intensity / (eps0 * c0 * 1.00027)).astype(
        np.complex64)[:, None, None]
    omega = 2 * np.pi * np.fft.fftfreq(nt, dt)
    kwargs = dict(
        Omega=omega, dt=dt, omega0=2 * np.pi * c0 / 800e-9,
        n0=1.00027, n_R=2.3e-23, omega_R=1.6e13,
        Gamma_R=1.3e13, iir_sampling="exact_piecewise_linear",
    )
    without_diagnostics = apply_isaacs_raman_operator_step(
        field, 1e-4, return_diagnostics=False, **kwargs)
    with_diagnostics, diagnostics = apply_isaacs_raman_operator_step(
        field, 1e-4, return_diagnostics=True, **kwargs)
    np.testing.assert_array_equal(without_diagnostics, with_diagnostics)
    assert diagnostics["global_closure_residual"] < 1e-3
    assert abs(diagnostics["energy_projection_scale"] - 1.0) < 1e-6


def test_strict_full_summary_reports_effective_absorption_off(capsys):
    from KHz_filament.config import (
        BeamConfig, GridConfig, HeatConfig, IonizationConfig,
        PropagationConfig, RamanConfig, RunConfig,
    )
    from KHz_filament.summary import print_sim_summary

    grid = GridConfig(Nx=4, Ny=4, Nt=8, Lx=4e-4, Ly=4e-4, Twin=80e-15)
    beam = BeamConfig(
        w0=1e-4, tau_fwhm=40e-15, energy_J=1e-9,
        P0_peak=None, focal_length=None)
    prop = PropagationConfig(
        z_max=1e-4, dz=1e-4, linear_model="paraxial",
        auto_substep=False, focus_window_step=False,
        use_raman_phase=False, use_raman_full_operator=True,
        use_raman_absorption=False)
    raman = RamanConfig(
        enabled=True, model="isaacs_rot_sinexp", n_R=2.3e-23,
        omega_R=1.6e13, Gamma_R=1.3e13, T_R=None, T2=None,
        operator_mode="full_isaacs_eq27", absorption=False)
    field = np.ones((grid.Nt, grid.Ny, grid.Nx), dtype=np.complex128)
    print_sim_summary(
        grid=grid, beam=beam, prop=prop, ion=IonizationConfig(species=[]),
        heat=HeatConfig(), run=RunConfig(), axes=None, E=field, raman=raman)
    output = capsys.readouterr().out
    absorption_line = next(
        line for line in output.splitlines() if "Absorption" in line)
    assert "OFF" in absorption_line
    assert "effective_scheme=off" in absorption_line
    assert "configured=OFF" in absorption_line
    assert "raman_absorption=OFF" in output
