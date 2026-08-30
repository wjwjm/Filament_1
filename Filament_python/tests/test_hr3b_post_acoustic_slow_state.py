from __future__ import annotations

import numpy as np
import pytest


def _parameters():
    return {"rho0": 1.23, "Cv": 1000.0 / 1.4, "T0": 293.15, "n0": 1.00027}


def test_post_acoustic_mapping_sign_zero_and_thermodynamic_closure():
    from KHz_filament.slow_state import map_post_acoustic_increment

    result = map_post_acoustic_increment(
        np.full((2, 3), 4.0), source_authoritative=True, **_parameters()
    )
    assert result["authoritative"]
    assert np.all(result["delta_n_increment"] < 0.0)
    assert np.all(result["delta_rho"] < 0.0)
    assert np.all(result["delta_t_impulse"] > 0.0)
    assert np.all(result["delta_t_post"] > 0.0)
    np.testing.assert_allclose(result["mapping_residual"], 0.0, atol=1e-30)
    np.testing.assert_allclose(result["isobaric_residual"], 0.0, atol=1e-15)

    zero = map_post_acoustic_increment(
        np.zeros((2, 3)), source_authoritative=True, **_parameters()
    )
    for name in ("delta_n_increment", "delta_rho", "delta_t_impulse", "delta_t_post"):
        np.testing.assert_array_equal(zero[name], np.zeros((2, 3)))


@pytest.mark.parametrize("bad", [np.array([[np.nan]]), np.array([[np.inf]]), np.array([[-1.0]])])
def test_post_acoustic_mapping_rejects_invalid_heat_source(bad):
    from KHz_filament.slow_state import map_post_acoustic_increment

    with pytest.raises(ValueError):
        map_post_acoustic_increment(bad, source_authoritative=True, **_parameters())
    with pytest.raises(ValueError, match="authoritative"):
        map_post_acoustic_increment(
            np.ones((1, 1)), source_authoritative=False, **_parameters()
        )


def test_disk_backed_state_is_zero_additive_and_persistent(tmp_path):
    from KHz_filament.slow_state import ThermalSlowStateStore

    store = ThermalSlowStateStore(
        output_path=str(tmp_path / "run.npz"), n_intervals=3, shape=(2, 2), dtype=np.float32
    )
    np.testing.assert_array_equal(store.read_interval(0), np.zeros((2, 2), dtype=np.float32))
    increment = np.full((2, 2), -0.25, dtype=np.float32)
    np.testing.assert_allclose(store.update_interval(1, increment), increment)
    np.testing.assert_allclose(store.update_interval(1, increment), 2.0 * increment)
    np.testing.assert_array_equal(store.read_interval(0), np.zeros((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(store.read_interval(2), np.zeros((2, 2), dtype=np.float32))
    metadata = store.finalize()
    reopened = np.lib.format.open_memmap(tmp_path / metadata["hr3b_state_filename"], mode="r")
    assert reopened.shape == (3, 2, 2)
    assert reopened.dtype == np.float32
    np.testing.assert_allclose(reopened[1], 2.0 * increment)


def test_hr3b_parameters_are_explicit_and_validated():
    from KHz_filament.config import HeatConfig, PropagationConfig
    from KHz_filament.slow_state import validate_hr3b_parameters

    heat = HeatConfig()
    assert heat.rho0 > 0.0 and heat.Cv > 0.0
    assert PropagationConfig().air_T > 0.0
    assert validate_hr3b_parameters(n0=1.00027, T0=293.15, rho0=heat.rho0, Cv=heat.Cv) > 0.0
    with pytest.raises(ValueError, match="positive"):
        validate_hr3b_parameters(n0=1.00027, T0=293.15, rho0=0.0, Cv=heat.Cv)


def test_hr3b_sparse_sink_streams_shared_sample_slots(tmp_path):
    from KHz_filament.longitudinal import build_longitudinal_schedule
    from KHz_filament.slow_state import HR3BDiagnosticSink
    from KHz_filament.thermalization import build_physical_sample_plan

    schedule = build_longitudinal_schedule(0.005, 0.010)
    plan = build_physical_sample_plan(
        schedule, focus_center_m=None, focus_halfwidth_m=0.0,
        focus_enabled=False, focal_plane_m=None,
    )
    sink = HR3BDiagnosticSink(
        plan=plan, output_path=str(tmp_path / "run.npz"), shape=(2, 2),
        dtype=np.float32, enabled=True,
    )
    for interval in plan.interval_index:
        sink.record_sample(int(interval), np.full((2, 2), -float(interval)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="only once"):
        sink.record_sample(int(plan.interval_index[0]), np.zeros((2, 2)), np.zeros((2, 2)))
    meta = sink.finalize()
    assert meta["hr3b_map_archive_complete"]
    increment = np.lib.format.open_memmap(tmp_path / meta["hr3b_increment_archive_filename"], mode="r")
    state = np.lib.format.open_memmap(tmp_path / meta["hr3b_state_after_archive_filename"], mode="r")
    assert increment.shape == state.shape == (plan.count, 2, 2)
    assert increment.dtype == state.dtype == np.float32


def test_low_level_slow_index_phase_law_is_reused():
    from KHz_filament.nonlinear import apply_nonlinear

    field = np.ones((2, 2, 2), dtype=np.complex128)
    delta_n = np.full((2, 2), -2.0e-5)
    k0, dz = 7.0e6, 1.0e-4
    updated = apply_nonlinear(
        field.copy(), np.zeros_like(field.real), np.zeros_like(field.real), dz,
        dn_gas=delta_n, k0=k0,
    )
    expected = np.broadcast_to(
        np.exp(1j * k0 * delta_n * dz)[None, :, :], field.shape
    )
    np.testing.assert_allclose(updated, expected)


def test_forward_store_update_is_invisible_to_same_pulse_and_visible_to_next(monkeypatch, tmp_path):
    from KHz_filament.config import IonizationConfig, PropagationConfig, RamanConfig
    from KHz_filament.constants import N0_air, Ui_N2, c0
    from KHz_filament.grids import make_axes
    from KHz_filament.longitudinal import build_deposition_contract, build_longitudinal_schedule
    from KHz_filament import propagate as propagation
    from KHz_filament.slow_state import ThermalSlowStateStore, validate_hr3b_parameters
    from KHz_filament.thermalization import thermalize_interval

    axes = make_axes(4, 4, 8, 4e-4, 4e-4, 80e-15)
    prop = PropagationConfig(
        z_max=2e-4, dz=1e-4, linear_model="paraxial", auto_substep=False,
        focus_window_step=False, limit_focus_window=False, progress_every_z=0,
        energy_probe_every=0, diag_extra=False, use_electronic_kerr=False,
        use_raman_phase=False, use_raman_absorption=False, use_plasma_phase=False,
        use_ionization_loss=False, use_ionization_solver=False,
    )
    raman = RamanConfig(enabled=False, absorption=False)
    schedule = build_longitudinal_schedule(1e-4, 2e-4)
    contract = build_deposition_contract(schedule, axes=axes)
    store = ThermalSlowStateStore(
        output_path=str(tmp_path / "order.npz"), n_intervals=schedule.n_intervals,
        shape=(4, 4), dtype=np.float64,
    )
    beta = validate_hr3b_parameters(**_parameters())

    def thermalize_constant(**kwargs):
        q = np.ones((4, 4), dtype=np.float64)
        reference = float(q.sum() * axes.dx * axes.dy * kwargs["dz"])
        return thermalize_interval(
            q_ion=q, q_ib=np.zeros_like(q), q_raman=np.zeros_like(q),
            dz=kwargs["dz"], dx=axes.dx, dy=axes.dy,
            mechanisms={
                "ion": {"active": True, "authoritative": True, "source": "test"},
                "ib": {"active": False, "authoritative": True, "source": "off"},
                "raman": {"active": False, "authoritative": True, "source": "off"},
            },
            reference_interval_J={"ion": reference, "ib": 0.0, "raman": 0.0},
        )

    original_apply = propagation.apply_nonlinear
    consumed = []

    def record_apply(*args, **kwargs):
        consumed.append(np.array(kwargs["dn_gas"], copy=True))
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(propagation, "thermalize_interval", thermalize_constant)
    monkeypatch.setattr(propagation, "apply_nonlinear", record_apply)

    def run_one():
        return propagation.propagate_one_pulse(
            np.zeros((8, 4, 4), dtype=np.complex128), kperp2=axes.kperp2,
            k0=2.0 * np.pi * c0 / 800e-9 * 1.00027,
            omega0=2.0 * np.pi * c0 / 800e-9, dz=prop.dz, z_max=prop.z_max,
            n0=1.00027, n2=0.0, Ui=Ui_N2, N0=N0_air,
            ion_conf=IonizationConfig(species=[]), dn_gas=None, dt=axes.dt,
            axes=axes, prop_conf=prop, raman_conf=raman, record_every_z=1,
            longitudinal_schedule=schedule, deposition_contract=contract,
            thermal_slow_state=store,
            hr3b_parameters={**_parameters(), "beta_th": beta},
        )

    _, _, first = run_one()
    assert len(consumed) == schedule.n_intervals
    assert all(np.all(values == 0.0) for values in consumed)
    first_increment = -beta
    np.testing.assert_allclose(store.read_interval(0), first_increment)
    np.testing.assert_allclose(first["delta_n_state_min_after_update"], [first_increment] * 2)

    consumed.clear()
    _, _, second = run_one()
    assert len(consumed) == schedule.n_intervals
    assert all(np.allclose(values, first_increment) for values in consumed)
    np.testing.assert_allclose(store.read_interval(0), 2.0 * first_increment)
    np.testing.assert_allclose(second["delta_n_state_min_after_update"], [2.0 * first_increment] * 2)


def test_authoritative_hr3b_runner_does_not_call_legacy_diffusion(monkeypatch, tmp_path):
    from KHz_filament import runner
    from KHz_filament.config import (
        BeamConfig, GridConfig, HeatConfig, IonizationConfig,
        PropagationConfig, RamanConfig, RunConfig,
    )

    def legacy_diffusion_must_not_run(*args, **kwargs):
        raise AssertionError("legacy Q2D/gamma_heat path must be isolated from HR-3B")

    monkeypatch.setattr(runner, "diffuse_dn_gas", legacy_diffusion_must_not_run)
    output = tmp_path / "authoritative_hr3b.npz"
    runner.run_demo(
        grid=GridConfig(Nx=4, Ny=4, Nt=8, Lx=4e-4, Ly=4e-4, Twin=80e-15),
        beam=BeamConfig(w0=1e-4, tau_fwhm=40e-15, energy_J=1e-10, focal_length=None),
        prop=PropagationConfig(
            z_max=1e-4, dz=1e-4, linear_model="paraxial", auto_substep=False,
            focus_window_step=False, limit_focus_window=False, progress_every_z=0,
            energy_probe_every=0, diag_extra=False, use_electronic_kerr=False,
            use_raman_phase=False, use_raman_absorption=False, use_plasma_phase=False,
            use_ionization_loss=False, use_ionization_solver=False,
        ),
        ion=IonizationConfig(species=[]), heat=HeatConfig(hr3b_enabled=True),
        run=RunConfig(Npulses=1), raman=RamanConfig(enabled=False, absorption=False),
        out_path=str(output), dtype="fp32",
    )
    with np.load(output, allow_pickle=False) as data:
        assert data["authoritative_hr_slow_state_update_active"].item()
        assert not data["legacy_slow_heat_compatibility_path_active"].item()
        assert data["hr3b_state_interval_centered"].item()
        assert data["hr3b_state_disk_backed"].item()
    assert output.with_suffix(".hr3b_delta_n_th.npy").is_file()
