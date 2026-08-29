from __future__ import annotations

import json

import numpy as np


def _run(
    tmp_path,
    *,
    full_operator=True,
    raman_enabled=True,
    operator_mode="full_isaacs_eq27",
    absorption=False,
    use_raman_absorption=False,
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

    output = tmp_path / (
        f"hr2d_{operator_mode}_{'on' if full_operator else 'off'}"
        f"_{'raman' if raman_enabled else 'raman_off'}.npz"
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
            use_self_steepening=False, use_electronic_kerr=False,
            use_raman_phase=False, use_raman_full_operator=full_operator,
            use_raman_absorption=use_raman_absorption,
            use_plasma_phase=False, use_ionization_loss=False,
            use_ionization_solver=False, measure_performance=True,
        ),
        ion=IonizationConfig(species=[]),
        heat=HeatConfig(),
        run=RunConfig(Npulses=1),
        raman=RamanConfig(
            enabled=raman_enabled, model="isaacs_rot_sinexp", n_R=2.3e-23,
            omega_R=1.6e13, Gamma_R=1.3e13, T_R=None, T2=None,
            operator_mode=operator_mode,
            operator_convention="isaacs_eq27",
            iir_sampling="exact_piecewise_linear",
            operator_integrator="heun", absorption=absorption,
            nonlinear_split_order="after_other",
            absorption_model="conv_deriv",
        ),
        out_path=str(output), dtype="fp64",
    )
    with np.load(output, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def test_unified_helper_sums_all_authoritative_mechanisms_and_closes_level2():
    from KHz_filament.deposition import build_unified_deposition_ledger

    ledger = build_unified_deposition_ledger(
        ion_interval_J=[1.0, 2.0],
        ib_interval_J=[0.25, 0.5],
        raman_interval_J=[0.5, 1.0],
        ion_interval_reference_J=[1.0, 2.0],
        ib_interval_reference_J=[0.25, 0.5],
        raman_interval_reference_J=[0.5, 1.0],
        ion_pulse_J=3.0,
        ib_pulse_J=0.75,
        raman_pulse_J=1.5,
        ion_configured=True,
        ib_configured=True,
        raman_configured=True,
        raman_authoritative=True,
        raman_source="eq10_heun_positive_rotational_energy",
        ionization_feedback_enabled=True,
        raman_feedback_enabled=True,
        field_in_J=8.0,
        field_out_J=2.75,
        raman_operator_relative_residuals=[2e-4, 3e-4],
        raman_operator_cumulative_relative_residual=4e-4,
    )
    np.testing.assert_allclose(ledger["total_interval_J"], [1.75, 3.5])
    assert ledger["total_authoritative"]
    assert ledger["total_pulse_J"] == 5.25
    assert ledger["level1_all_available_pass"]
    assert ledger["level2_all_available_pass"]
    assert ledger["total_level2_status"] == "pass"
    assert ledger["field_loss_J"] == 5.25
    assert ledger["field_residual_J"] == 0.0
    assert ledger["raman_operator_energy_closure_status"] == "pass"


def test_full_operator_unifies_canonical_interval_and_pulse_ledgers(tmp_path):
    data = _run(tmp_path, full_operator=True)
    ion = np.asarray(data["E_dep_ion_interval_J"])
    ib = np.asarray(data["E_dep_ib_interval_J"])
    raman = np.asarray(data["E_dep_raman_interval_J"])
    total = np.asarray(data["E_dep_total_interval_J"])

    assert data["total_deposition_authoritative"].item()
    assert data["field_energy_bookkeeping_authoritative"].item()
    assert len(total) == int(data["n_intervals"])
    np.testing.assert_allclose(total, ion + ib + raman)
    np.testing.assert_allclose(data["E_dep_total_pulse_J"], total.sum())
    assert data["deposition_level1_all_available_mechanism_closure_pass"].item()
    assert data["deposition_level2_all_available_mechanism_closure_pass"].item()
    assert data["E_dep_total_level2_closure_status"].item() == "pass"
    for key in (
        "E_field_in_J",
        "E_field_out_J",
        "E_field_loss_J",
        "E_dep_accounted_authoritative_J",
        "E_field_energy_bookkeeping_residual_J",
        "E_field_energy_bookkeeping_relative_residual",
    ):
        assert np.isfinite(data[key]).item()
    status = json.loads(data["deposition_mechanism_status_json"].item())
    assert status["raman"]["authoritative"]
    assert status["raman"]["source"] == "eq10_heun_positive_rotational_energy"
    assert status["raman"]["deposition_reduction_closure_status"] == "pass"
    assert status["raman"]["operator_energy_closure_status"] == "pass"


def test_raman_off_does_not_block_authoritative_total(tmp_path):
    data = _run(tmp_path, full_operator=False, raman_enabled=False)
    total = np.asarray(data["E_dep_total_interval_J"])
    np.testing.assert_allclose(
        total,
        np.asarray(data["E_dep_ion_interval_J"])
        + np.asarray(data["E_dep_ib_interval_J"]),
    )
    assert data["total_deposition_authoritative"].item()
    status = json.loads(data["deposition_mechanism_status_json"].item())
    assert not status["raman"]["configured"]
    assert not status["raman"]["active"]
    assert status["raman"]["source"] == "off"


def test_legacy_raman_blocks_total_without_fallback_to_legacy_estimate(tmp_path):
    data = _run(
        tmp_path, full_operator=False, operator_mode="legacy_split",
        absorption=True, use_raman_absorption=True,
    )
    assert not data["total_deposition_authoritative"].item()
    assert not data["field_energy_bookkeeping_authoritative"].item()
    assert np.all(np.isnan(data["E_dep_total_interval_J"]))
    assert np.isnan(data["E_dep_total_pulse_J"]).item()
    assert "raman:legacy_unavailable" in data[
        "total_deposition_unavailable_reason"
    ].item()
    assert data["deposition_raman_level1_closure_status"].item() == "unavailable"
    assert data["deposition_raman_level2_closure_status"].item() == "unavailable"
    assert data["field_energy_bookkeeping_status"].item() == "unavailable"
    status = json.loads(data["deposition_mechanism_status_json"].item())
    assert status["raman"]["active"]
    assert not status["raman"]["authoritative"]
    assert status["raman"]["source"] == "legacy_unavailable"


def test_feedback_off_uses_actual_zero_raman_in_unified_total(tmp_path):
    data = _run(tmp_path, full_operator=False)
    assert data["raman_deposition_source"].item() == "operator_not_applied"
    assert np.max(data["raman_target_loss_step_J"]) > 0.0
    assert np.all(data["E_dep_raman_interval_J"] == 0.0)
    np.testing.assert_allclose(
        data["E_dep_total_interval_J"],
        np.asarray(data["E_dep_ion_interval_J"])
        + np.asarray(data["E_dep_ib_interval_J"]),
    )
    assert data["total_deposition_authoritative"].item()
    assert data["deposition_raman_level1_closure_status"].item() == "pass"
    assert data["deposition_raman_operator_energy_closure_status"].item() == "not_applicable"


def test_unified_output_keeps_only_scalar_longitudinal_ledgers(tmp_path):
    data = _run(tmp_path, full_operator=True)
    assert not any(
        key.startswith(("q_ion", "q_ib", "q_raman", "q_total"))
        or key.startswith("actual_local_fluence_loss")
        for key in data
    )
    assert np.asarray(data["E_dep_total_interval_J"]).ndim == 1


def test_hr3a_thermal_ledger_consumes_authoritative_interval_maps(tmp_path):
    data = _run(tmp_path, full_operator=True)

    assert data["thermalization_ledger_schema"].item() == (
        "khz_filament.thermalization_ledger.v1"
    )
    assert data["thermalization_source"].item() == "hr2_authoritative_deposition"
    assert data["thermalization_authoritative"].item()
    assert data["thermalization_t1_ion_status"].item() == "pass"
    assert data["thermalization_t1_raman_status"].item() == "pass"
    assert data["thermalization_t2_ion_status"].item() == "pass"
    assert data["thermalization_t3_channel_sum_status"].item() == "pass"
    assert data["thermalization_zero_channel_pass"].item()
    assert np.asarray(data["q_th_ion"]).shape == (
        int(data["n_intervals"]), 8, 8
    )
    assert np.array_equal(data["q_th_ib"], np.zeros_like(data["q_th_ib"]))
    np.testing.assert_allclose(
        data["q_thermal"], data["q_th_ion"] + data["q_th_ib"] + data["q_th_raman"]
    )
    np.testing.assert_allclose(
        data["E_thermal_interval_J"],
        data["E_th_ion_interval_J"]
        + data["E_th_ib_interval_J"]
        + data["E_th_raman_interval_J"],
    )
