from __future__ import annotations

import json

import numpy as np
import pytest


N0 = 1.00027


def _controller(tmp_path, *, name="run", n_pulses=2, f_rep=1.0e6, dt_hydro=1.0e-6,
                dx=1.0e-4, shape=(7, 7), chi=0.0, nu=0.0, gravity_y=0.0, resume=False):
    from KHz_filament.hr4d_pulse_lifecycle import HR4DPulseController

    return HR4DPulseController(
        output_path=str(tmp_path / f"{name}.npz"), n_intervals=2, shape=shape, dtype=np.float64,
        z_edges=np.array([0.0, 1.0e-4, 2.0e-4]), dx=dx, dy=dx, n_pulses=n_pulses,
        f_rep=f_rep, dt_hydro=dt_hydro, batch_intervals=1, chi=chi, nu=nu, n0=N0,
        gravity_y=gravity_y, resume=resume,
    )


def _pulse_runner(records, *, increment_scale=-1.0e-6, fail=None):
    def run(field, transaction):
        pulse = len(records["inputs"])
        records["inputs"].append(np.array(field, copy=True))
        records["ids"].append(id(field))
        for index in range(transaction.store.n_intervals):
            pre = transaction.read_interval(index)
            if fail == "propagation" and index == 0:
                raise RuntimeError("injected propagation failure")
            increment = np.full_like(pre, increment_scale * (pulse + 1) * (index + 1))
            transaction.update_interval(index, increment)
        if fail == "conversion":
            raise RuntimeError("injected conversion failure")
        field[...] = field + (pulse + 1)
        return {"pulse": pulse, "field_id": id(field)}
    return run


def _post_first_pulse(controller, source, records):
    from KHz_filament.hr4d_pulse_lifecycle import run_one_pulse_transition

    run_one_pulse_transition(controller, source, _pulse_runner(records))


def test_d1_d2_d3_fresh_source_post_delta_n_and_velocity_continuity(tmp_path):
    from KHz_filament.hr4d_pulse_lifecycle import run_hr4_pulse_train

    controller = _controller(tmp_path, n_pulses=2)
    source = np.arange(16, dtype=np.complex128).reshape(4, 4)
    records = {"inputs": [], "ids": []}
    result = run_hr4_pulse_train(controller, source, _pulse_runner(records))
    assert result["pulse_calls_this_invocation"] == 2
    assert len(records["inputs"]) == 2
    np.testing.assert_array_equal(records["inputs"][0], source)
    np.testing.assert_array_equal(records["inputs"][1], source)
    final = controller.store.read_authoritative_batch(0, 2)
    np.testing.assert_array_equal(final["vx"], np.zeros_like(final["vx"]))
    np.testing.assert_array_equal(final["vy"], np.zeros_like(final["vy"]))
    assert result["optical_working_field_history_stored"] is False
    controller.close()

    post_only = _controller(tmp_path, name="post_only", n_pulses=1)
    post_records = {"inputs": [], "ids": []}
    run_hr4_pulse_train(post_only, source, _pulse_runner(post_records))
    post = post_only.store.read_authoritative_batch(0, 2)
    np.testing.assert_array_equal(post["delta_n"][0], np.full((7, 7), -1e-6))
    np.testing.assert_array_equal(post["delta_n"][1], np.full((7, 7), -2e-6))
    np.testing.assert_array_equal(post["vx"], np.zeros_like(post["vx"]))
    np.testing.assert_array_equal(post["vy"], np.zeros_like(post["vy"]))
    post_only.close()


@pytest.mark.parametrize("n_pulses, expected_interpulse", [(1, 0), (2, 1), (5, 4)])
def test_d4_d5_d6_pulse_counts_and_final_post_has_no_extra_evolution(tmp_path, n_pulses, expected_interpulse):
    from KHz_filament.hr4d_pulse_lifecycle import run_hr4_pulse_train

    controller = _controller(tmp_path, name=f"n{n_pulses}", n_pulses=n_pulses)
    records = {"inputs": [], "ids": []}
    report = run_hr4_pulse_train(controller, np.ones((3, 3), dtype=np.complex128), _pulse_runner(records))
    metadata = report["final_metadata"]
    assert report["pulse_calls_this_invocation"] == n_pulses
    assert report["interpulse_calls_this_invocation"] == expected_interpulse
    assert metadata["phase"] == "POST" and metadata["pulse_index"] == n_pulses - 1
    assert metadata["run_complete"] is True
    assert controller.next_action == "complete"
    controller.close()


def test_d7_d8_exact_full_and_remainder_interpulse_schedule(tmp_path):
    from KHz_filament.hr4d_pulse_lifecycle import build_interpulse_step_schedule

    exact = build_interpulse_step_schedule(f_rep=1.0e5, dt_hydro=1.0e-6)
    assert exact.full_step_count == 10 and exact.remainder_s == 0.0
    remainder = build_interpulse_step_schedule(f_rep=4.0e5, dt_hydro=1.0e-6)
    assert remainder.entries[0] == (1.0e-6, 2)
    assert remainder.entries[1][0] == pytest.approx(0.5e-6, abs=1e-18)
    assert remainder.entries[1][1] == 1
    assert remainder.full_step_count * remainder.dt_hydro_s + remainder.remainder_s == pytest.approx(
        remainder.duration_s, abs=1e-18
    )

    controller = _controller(tmp_path, n_pulses=2, f_rep=4.0e5)
    records = {"inputs": [], "ids": []}
    _post_first_pulse(controller, np.ones((3, 3), dtype=np.complex128), records)
    summary = controller.run_interpulse_transition()
    assert summary["full_step_count"] == 2
    assert summary["remainder_s"] == pytest.approx(0.5e-6, abs=1e-18)
    assert summary["step_schedule"][0] == (1.0e-6, 2)
    assert summary["step_schedule"][1][0] == pytest.approx(0.5e-6, abs=1e-18)
    assert summary["step_schedule"][1][1] == 1
    assert controller.metadata["phase"] == "PRE" and controller.metadata["pulse_index"] == 1
    controller.close()


def test_d9_remainder_stability_failure_is_fail_closed(tmp_path):
    controller = _controller(
        tmp_path, n_pulses=2, f_rep=2.0e5, dt_hydro=10.0e-6, dx=10.0e-6, chi=21.7e-6,
    )
    records = {"inputs": [], "ids": []}
    _post_first_pulse(controller, np.ones((3, 3), dtype=np.complex128), records)
    before = np.array(controller.store.read_authoritative_batch(0, 2)["delta_n"], copy=True)
    with pytest.raises(ValueError, match="stability audit"):
        controller.run_interpulse_transition()
    assert controller.metadata["phase"] == "POST" and controller.metadata["pulse_index"] == 0
    np.testing.assert_array_equal(controller.store.read_authoritative_batch(0, 2)["delta_n"], before)
    controller.close()


def test_d10_d11_d12_restart_actions_are_unambiguous(tmp_path):
    from KHz_filament.hr4d_pulse_lifecycle import run_hr4_pulse_train

    source = np.ones((3, 3), dtype=np.complex128)
    records = {"inputs": [], "ids": []}
    pre = _controller(tmp_path, name="pre", n_pulses=2)
    with pytest.raises(RuntimeError, match="propagation"):
        run_hr4_pulse_train(pre, source, _pulse_runner(records, fail="propagation"))
    assert pre.next_action == "pulse" and pre.metadata["pulse_index"] == 0
    pre.close()
    resumed_pre = _controller(tmp_path, name="pre", n_pulses=2, resume=True)
    assert resumed_pre.next_action == "pulse"
    run_hr4_pulse_train(resumed_pre, source, _pulse_runner(records))
    assert resumed_pre.is_complete
    resumed_pre.close()

    post = _controller(tmp_path, name="post", n_pulses=2)
    post_records = {"inputs": [], "ids": []}
    _post_first_pulse(post, source, post_records)
    post.close()
    resumed_post = _controller(tmp_path, name="post", n_pulses=2, resume=True)
    assert resumed_post.next_action == "interpulse"
    report = run_hr4_pulse_train(resumed_post, source, _pulse_runner(post_records))
    assert report["pulse_calls_this_invocation"] == 1 and report["interpulse_calls_this_invocation"] == 1
    assert resumed_post.is_complete
    resumed_post.close()

    final = _controller(tmp_path, name="final", n_pulses=1)
    final_records = {"inputs": [], "ids": []}
    run_hr4_pulse_train(final, source, _pulse_runner(final_records))
    final.close()
    resumed_final = _controller(tmp_path, name="final", n_pulses=1, resume=True)
    assert resumed_final.next_action == "complete"
    result = run_hr4_pulse_train(resumed_final, source, _pulse_runner(final_records))
    assert result["pulse_calls_this_invocation"] == result["interpulse_calls_this_invocation"] == 0
    resumed_final.close()


@pytest.mark.parametrize("failure", ["propagation", "conversion"])
def test_d13_d14_optical_or_conversion_failure_leaves_pre_authoritative(tmp_path, failure):
    from KHz_filament.hr4d_pulse_lifecycle import run_one_pulse_transition

    controller = _controller(tmp_path, name=failure, n_pulses=1)
    records = {"inputs": [], "ids": []}
    with pytest.raises(RuntimeError, match=failure):
        run_one_pulse_transition(controller, np.ones((3, 3), dtype=np.complex128), _pulse_runner(records, fail=failure))
    assert controller.metadata["phase"] == "PRE" and controller.metadata["pulse_index"] == 0
    assert controller.store.manifest["transaction_status"] == "committed"
    controller.close()


def test_d15_d16_commit_and_interpulse_failures_preserve_nearest_authority(tmp_path, monkeypatch):
    from KHz_filament.hr4d_pulse_lifecycle import run_one_pulse_transition

    controller = _controller(tmp_path, name="commit", n_pulses=2)
    records = {"inputs": [], "ids": []}
    original_commit = controller.store.commit_staging
    monkeypatch.setattr(controller.store, "commit_staging", lambda *a, **k: (_ for _ in ()).throw(OSError("commit fail")))
    with pytest.raises(OSError, match="commit fail"):
        run_one_pulse_transition(controller, np.ones((3, 3), dtype=np.complex128), _pulse_runner(records))
    assert controller.metadata["phase"] == "PRE"
    monkeypatch.setattr(controller.store, "commit_staging", original_commit)
    _post_first_pulse(controller, np.ones((3, 3), dtype=np.complex128), records)
    before = np.array(controller.store.read_authoritative_batch(0, 2)["delta_n"], copy=True)
    with pytest.raises(RuntimeError, match="interpulse"):
        controller.run_interpulse_transition(failure_injector=lambda start, stop: (_ for _ in ()).throw(RuntimeError("interpulse")))
    assert controller.metadata["phase"] == "POST" and controller.metadata["pulse_index"] == 0
    np.testing.assert_array_equal(controller.store.read_authoritative_batch(0, 2)["delta_n"], before)
    controller.close()


def test_d17_metadata_tamper_fails_closed_and_d18_optical_memory_is_bounded(tmp_path):
    from KHz_filament.hr4d_pulse_lifecycle import HR4DPulseController, run_hr4_pulse_train

    controller = _controller(tmp_path, name="memory", n_pulses=5)
    source = np.ones((4, 4), dtype=np.complex128)
    records = {"inputs": [], "ids": []}
    report = run_hr4_pulse_train(controller, source, _pulse_runner(records))
    assert len(records["ids"]) == 5
    for input_field in records["inputs"]:
        np.testing.assert_array_equal(input_field, source)
    np.testing.assert_array_equal(source, np.ones_like(source))
    assert report["optical_working_field_history_stored"] is False
    controller.close()
    manifest_path = tmp_path / "memory.hr4c_state_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline = json.loads(json.dumps(manifest))
    cases = [
        ("phase", "unknown", "phase"),
        ("pulse_index", -1, "pulse_index"),
        ("pulse_index", 5, "pulse_index"),
        ("predecessor_generation", 0, "predecessor_generation"),
    ]
    for field, value, message in cases:
        tampered = json.loads(json.dumps(baseline))
        tampered["authoritative_metadata"][field] = value
        manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            HR4DPulseController(
                output_path=str(tmp_path / "memory.npz"), n_intervals=2, shape=(7, 7), dtype=np.float64,
                z_edges=np.array([0.0, 1.0e-4, 2.0e-4]), dx=1.0e-4, dy=1.0e-4, n_pulses=5,
                f_rep=1.0e6, dt_hydro=1.0e-6, batch_intervals=1, chi=0.0, nu=0.0, n0=N0,
                gravity_y=0.0, resume=True,
            )
    tampered = json.loads(json.dumps(baseline))
    tampered["authoritative_metadata"]["flow_parameters"]["f_rep_hz"] = 2.0e6
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="flow parameters"):
        HR4DPulseController(
            output_path=str(tmp_path / "memory.npz"), n_intervals=2, shape=(7, 7), dtype=np.float64,
            z_edges=np.array([0.0, 1.0e-4, 2.0e-4]), dx=1.0e-4, dy=1.0e-4, n_pulses=5,
            f_rep=1.0e6, dt_hydro=1.0e-6, batch_intervals=1, chi=0.0, nu=0.0, n0=N0,
            gravity_y=0.0, resume=True,
        )


def test_d19_repository_level_real_propagation_to_hr3ab_to_hr4c_path(tmp_path):
    from KHz_filament.config import BeamConfig, GridConfig, IonizationConfig, PropagationConfig, RamanConfig
    from KHz_filament.constants import N0_air, Ui_N2, c0, n2_air
    from KHz_filament.device import xp
    from KHz_filament.grids import make_axes
    from KHz_filament.hr4d_pulse_lifecycle import run_hr4_pulse_train
    from KHz_filament.longitudinal import build_deposition_contract, build_longitudinal_schedule
    from KHz_filament.propagate import propagate_one_pulse
    from KHz_filament.runner import build_transverse_input_field
    from KHz_filament.slow_state import validate_hr3b_parameters

    grid = GridConfig(Nx=8, Ny=8, Nt=8, Lx=8e-4, Ly=8e-4, Twin=80e-15)
    beam = BeamConfig(w0=1.5e-4, tau_fwhm=40e-15, energy_J=1e-10, focal_length=None)
    prop = PropagationConfig(
        z_max=2e-4, dz=1e-4, linear_model="paraxial", auto_substep=False,
        focus_window_step=False, limit_focus_window=False, progress_every_z=0,
        energy_probe_every=0, diag_extra=False, use_electronic_kerr=False,
        use_raman_phase=False, use_raman_absorption=False, use_plasma_phase=False,
        use_ionization_loss=False, use_ionization_solver=False,
    )
    axes = make_axes(grid.Nx, grid.Ny, grid.Nt, grid.Lx, grid.Ly, grid.Twin)
    source, _ = build_transverse_input_field(axes, beam, xp.complex64)
    schedule = build_longitudinal_schedule(dz=prop.dz, z_max=prop.z_max)
    contract = build_deposition_contract(schedule, axes=axes)
    controller = _controller(
        tmp_path, name="real", n_pulses=2, f_rep=1.0e6, dx=axes.dx, shape=(grid.Ny, grid.Nx),
        chi=21.7e-6, nu=1.5e-5,
    )
    beta = validate_hr3b_parameters(rho0=1.23, Cv=1000.0 / 1.4, T0=prop.air_T, n0=beam.n0)
    hr3b = {"rho0": 1.23, "Cv": 1000.0 / 1.4, "T0": prop.air_T, "n0": beam.n0, "beta_th": beta}
    omega0 = 2.0 * np.pi * c0 / beam.lam0
    k0 = beam.n0 * omega0 / c0
    ion = IonizationConfig(species=[])
    raman = RamanConfig(enabled=False, absorption=False)

    def real_pulse(field, transaction):
        return propagate_one_pulse(
            field, kperp2=axes.kperp2, k0=k0, omega0=omega0, dz=prop.dz, z_max=prop.z_max,
            n0=beam.n0, n2=n2_air, Ui=Ui_N2, N0=N0_air, ion_conf=ion,
            dn_gas=xp.zeros((grid.Ny, grid.Nx), dtype=xp.float32), dt=axes.dt, axes=axes,
            prop_conf=prop, raman_conf=raman, record_onaxis_rho_time=False, record_every_z=1,
            longitudinal_schedule=schedule, deposition_contract=contract,
            thermal_slow_state=transaction, hr3b_parameters=hr3b,
        )

    report = run_hr4_pulse_train(controller, source, real_pulse)
    assert report["pulse_calls_this_invocation"] == 2
    assert report["interpulse_calls_this_invocation"] == 1
    assert controller.is_complete and controller.metadata["phase"] == "POST"
    controller.close()
