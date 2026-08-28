from __future__ import annotations

import numpy as np
import pytest


def _run(
    tmp_path,
    *,
    enabled=True,
    split_order="after_other",
    operator_mode="full_isaacs_eq27",
    raman_enabled=True,
    absorption=False,
    use_raman_absorption=False,
    electronic_kerr=False,
    n_R=2.3e-23,
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

    path = tmp_path / (
        f"hr2c_{operator_mode}_{'on' if enabled else 'off'}"
        f"_{split_order}_{'enabled' if raman_enabled else 'disabled'}.npz"
    )
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
            use_self_steepening=False, use_electronic_kerr=electronic_kerr,
            use_raman_phase=False, use_raman_full_operator=enabled,
            use_raman_absorption=use_raman_absorption,
            use_plasma_phase=False, use_ionization_loss=False,
            use_ionization_solver=False, measure_performance=True,
        ),
        ion=IonizationConfig(species=[]),
        heat=HeatConfig(),
        run=RunConfig(Npulses=1),
        raman=RamanConfig(
            enabled=raman_enabled, model="isaacs_rot_sinexp", n_R=n_R,
            omega_R=1.6e13, Gamma_R=1.3e13, T_R=None, T2=None,
            operator_mode=operator_mode,
            operator_convention="isaacs_eq27",
            iir_sampling="exact_piecewise_linear",
            operator_integrator="heun", absorption=absorption,
            nonlinear_split_order=split_order,
            absorption_model="conv_deriv",
        ),
        out_path=str(path), dtype="fp64",
    )
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def test_q_raman_helper_units_geometry_and_input_validation():
    from KHz_filament.deposition import (
        interval_energy_from_fluence_gain,
        interval_energy_from_q,
        q_raman_from_actual_fluence_loss,
        q_raman_from_target_fluence_gain,
    )

    loss = np.array([[2.0, -1.0], [np.nan, 4.0]], dtype=np.float64)
    dz, dx, dy = 0.5, 0.2, 0.3
    q_raman = q_raman_from_actual_fluence_loss(loss, dz)
    np.testing.assert_allclose(q_raman, np.array([[4.0, 0.0], [0.0, 8.0]]))
    expected_energy = np.sum(np.maximum(np.nan_to_num(loss), 0.0)) * dx * dy
    assert interval_energy_from_q(q_raman, dx, dy, dz) == pytest.approx(expected_energy)
    signed_energy = np.sum(np.nan_to_num(loss)) * dx * dy
    assert expected_energy != pytest.approx(signed_energy)
    target = np.array([[2.0, -1.0], [np.nan, 4.0]], dtype=np.float64)
    q_target = q_raman_from_target_fluence_gain(target, dz)
    target_energy = interval_energy_from_fluence_gain(target, dx, dy)
    np.testing.assert_allclose(q_target, q_raman)
    assert interval_energy_from_q(q_target, dx, dy, dz) == pytest.approx(target_energy)
    with pytest.raises(ValueError, match="shape"):
        q_raman_from_actual_fluence_loss(np.ones((2, 2, 1)), dz)
    with pytest.raises(ValueError, match="positive"):
        q_raman_from_actual_fluence_loss(np.ones((2, 2)), 0.0)


def test_full_operator_on_has_authoritative_interval_and_pulse_ledgers(tmp_path):
    data = _run(tmp_path, enabled=True)
    assert data["raman_deposition_authoritative"].item()
    assert data["raman_deposition_source"].item() == "eq10_heun_positive_rotational_energy"
    n_intervals = int(data["n_intervals"])
    q_energy = np.asarray(data["E_dep_raman_interval_J"])
    operator_energy = np.asarray(data["E_dep_raman_interval_operator_J"])
    reduction_reference = np.asarray(data["E_dep_raman_interval_reduction_reference_J"])
    reduction_residual = np.asarray(data["E_dep_raman_interval_closure_residual_J"])
    operator_residual = np.asarray(data["E_dep_raman_operator_energy_residual_J"])
    assert len(q_energy) == len(operator_energy) == len(reduction_residual) == n_intervals
    assert np.max(q_energy) > 0.0
    np.testing.assert_allclose(q_energy, data["raman_target_loss_step_J"], rtol=1e-10, atol=1e-30)
    np.testing.assert_allclose(q_energy, reduction_reference, rtol=2e-5, atol=1e-30)
    np.testing.assert_allclose(reduction_residual, q_energy - reduction_reference)
    np.testing.assert_allclose(operator_residual, q_energy - operator_energy)
    assert data["deposition_raman_deposition_reduction_closure_status"].item() == "pass"
    assert data["deposition_raman_operator_energy_closure_status"].item() == "pass"
    np.testing.assert_allclose(data["E_dep_raman_pulse_J"], q_energy.sum())
    np.testing.assert_allclose(
        data["E_dep_raman_operator_pulse_J"], operator_energy.sum()
    )
    np.testing.assert_allclose(
        data["E_dep_raman_pulse_closure_residual_J"], reduction_residual.sum()
    )


def test_full_operator_feedback_off_uses_zero_actual_deposition(tmp_path):
    data = _run(tmp_path, enabled=False)
    assert data["raman_deposition_authoritative"].item()
    assert data["raman_deposition_source"].item() == "operator_not_applied"
    assert np.max(data["raman_target_loss_step_J"]) > 0.0
    assert np.all(data["raman_actual_loss_step_J"] == 0.0)
    assert np.all(data["E_dep_raman_interval_J"] == 0.0)
    assert data["E_dep_raman_pulse_J"].item() == 0.0


def test_strang_combines_two_operator_substeps_into_one_interval_entry(tmp_path):
    data = _run(tmp_path, enabled=True, split_order="strang")
    assert np.all(data["raman_operator_substep_count"] == 2)
    assert len(data["E_dep_raman_interval_J"]) == int(data["n_intervals"]) == 1
    np.testing.assert_allclose(
        data["E_dep_raman_pulse_J"], data["E_dep_raman_interval_J"].sum()
    )
    np.testing.assert_allclose(
        data["E_dep_raman_interval_J"], data["raman_target_loss_step_J"],
        rtol=1e-10, atol=1e-30,
    )


def test_legacy_raman_mode_is_non_authoritative_and_has_zero_new_ledger(tmp_path):
    data = _run(
        tmp_path, enabled=False, operator_mode="legacy_split",
        absorption=True, use_raman_absorption=True,
    )
    assert not data["raman_deposition_authoritative"].item()
    assert data["raman_deposition_source"].item() == "legacy_unavailable"
    assert np.all(data["E_dep_raman_interval_J"] == 0.0)
    assert np.all(data["E_dep_raman_interval_operator_J"] == 0.0)
    assert data["E_dep_raman_pulse_J"].item() == 0.0


def test_complete_electronic_only_path_creates_no_raman_medium_deposition(tmp_path):
    data = _run(
        tmp_path,
        enabled=True,
        operator_mode="full_isaacs_eq27_complete",
        electronic_kerr=True,
        n_R=0.0,
    )
    assert data["raman_deposition_source"].item() == "eq10_heun_positive_rotational_energy"
    assert np.all(data["E_dep_raman_interval_J"] == 0.0)
    assert data["E_dep_raman_pulse_J"].item() == 0.0
    assert data["deposition_raman_deposition_reduction_closure_status"].item() == "pass"


def test_raman_off_is_authoritative_zero_without_local_or_q_stack(tmp_path):
    data = _run(tmp_path, enabled=False, raman_enabled=False)
    assert data["raman_deposition_authoritative"].item()
    assert data["raman_deposition_source"].item() == "off"
    assert np.all(data["E_dep_raman_interval_J"] == 0.0)
    assert data["E_dep_raman_pulse_J"].item() == 0.0
    assert not any(
        key.startswith("q_raman") or key.startswith("actual_local_fluence_loss")
        for key in data
    )
