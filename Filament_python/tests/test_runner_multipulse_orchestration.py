from __future__ import annotations

import numpy as np


def _tiny_components(*, npulses: int):
    from KHz_filament.config import (
        BeamConfig,
        GridConfig,
        HeatConfig,
        IonizationConfig,
        PropagationConfig,
        RamanConfig,
        RunConfig,
    )

    return {
        "grid": GridConfig(Nx=8, Ny=8, Nt=16, Lx=8e-4, Ly=8e-4, Twin=160e-15),
        "beam": BeamConfig(
            w0=1.5e-4,
            tau_fwhm=40e-15,
            energy_J=1e-9,
            focal_length=None,
        ),
        "prop": PropagationConfig(
            z_max=2e-4,
            dz=1e-4,
            linear_model="paraxial",
            auto_substep=False,
            focus_window_step=False,
            limit_focus_window=False,
            progress_every_z=0,
            diag_extra=False,
            energy_probe_every=0,
        ),
        "ion": IonizationConfig(
            species=[
                {
                    "name": "test",
                    "rate": "mpa_fact",
                    "ell": 2,
                    "I_mp": 1e18,
                    "Ip_eV": 15.0,
                    "fraction": 1.0,
                }
            ]
        ),
        "heat": HeatConfig(f_rep=1e3),
        "run": RunConfig(Npulses=npulses),
        "raman": RamanConfig(enabled=False, absorption=False),
        "dtype": "fp32",
    }


def test_runner_multipulse_uses_fresh_source_and_inherits_medium_state(monkeypatch, tmp_path):
    from KHz_filament import runner

    field_inputs = []
    field_objects = []
    field_input_ids = []
    medium_inputs = []
    schedule_inputs = []
    contract_inputs = []

    def fake_propagate_one_pulse(E, **kwargs):
        pulse_number = len(field_inputs) + 1
        field_inputs.append(np.array(E, copy=True))
        field_objects.append(E)
        field_input_ids.append(id(E))
        medium_inputs.append(np.array(kwargs["dn_gas"], copy=True))
        schedule_inputs.append(kwargs["longitudinal_schedule"])
        contract_inputs.append(kwargs["deposition_contract"])
        E[...] = E * 10.0 + pulse_number
        return E, np.ones(E.shape[-2:], dtype=np.float32), {}

    def fake_diffuse_dn_gas(dn_gas, Q2D, *args):
        return np.asarray(dn_gas) + np.asarray(Q2D)

    monkeypatch.setattr(runner, "propagate_one_pulse", fake_propagate_one_pulse)
    monkeypatch.setattr(runner, "diffuse_dn_gas", fake_diffuse_dn_gas)

    out_path = tmp_path / "multipulse.npz"
    result = runner.run_demo(
        **_tiny_components(npulses=3),
        out_path=str(out_path),
        return_results=True,
    )

    assert len(field_inputs) == 3
    assert len(set(field_input_ids)) == 3
    assert not np.shares_memory(field_objects[0], field_objects[1])
    assert not np.shares_memory(field_objects[1], field_objects[2])
    for field_input in field_inputs[1:]:
        np.testing.assert_array_equal(field_input, field_inputs[0])
    assert len({id(schedule) for schedule in schedule_inputs}) == 1
    assert len({id(contract) for contract in contract_inputs}) == 1
    assert contract_inputs[0].schedule is schedule_inputs[0]

    expected_medium = [0.0, 1.0, 2.0]
    for actual, expected in zip(medium_inputs, expected_medium, strict=True):
        np.testing.assert_array_equal(actual, np.full_like(actual, expected))

    np.testing.assert_array_equal(result["pulse_summary"]["pulse_index"], [1, 2, 3])
    np.testing.assert_array_equal(result["pulse_summary"]["dn_gas_min"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result["pulse_summary"]["dn_gas_max"], [1.0, 2.0, 3.0])
    with np.load(out_path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["pulse_index"], [1, 2, 3])
        np.testing.assert_array_equal(saved["pulse_dn_gas_min"], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(saved["pulse_dn_gas_max"], [1.0, 2.0, 3.0])


def test_runner_npulses_one_matches_direct_single_pulse_propagation(tmp_path):
    from KHz_filament.config import RunConfig
    from KHz_filament.constants import N0_air, Ui_N2, c0, n2_air
    from KHz_filament.device import xp
    from KHz_filament.diagnostics import intensity
    from KHz_filament.grids import make_axes
    from KHz_filament.propagate import propagate_one_pulse
    from KHz_filament.runner import build_transverse_input_field, run_demo

    components = _tiny_components(npulses=1)
    grid = components["grid"]
    beam = components["beam"]
    prop = components["prop"]
    ion = components["ion"]
    raman = components["raman"]

    result = run_demo(
        **components,
        out_path=str(tmp_path / "runner_single.npz"),
        return_results=True,
    )

    axes = make_axes(grid.Nx, grid.Ny, grid.Nt, grid.Lx, grid.Ly, grid.Twin)
    E_source, _ = build_transverse_input_field(axes, beam, xp.complex64)
    omega0 = 2.0 * np.pi * c0 / beam.lam0
    k0 = beam.n0 * omega0 / c0
    n2_used = float(getattr(prop, "n2", getattr(beam, "n2_air", n2_air)))
    E_direct, _, diag_direct = propagate_one_pulse(
        E_source,
        kperp2=axes.kperp2,
        k0=k0,
        omega0=omega0,
        dz=prop.dz,
        z_max=prop.z_max,
        n0=beam.n0,
        n2=n2_used,
        Ui=Ui_N2,
        N0=N0_air,
        ion_conf=ion,
        dn_gas=xp.zeros((grid.Ny, grid.Nx), dtype=xp.float32),
        dt=axes.dt,
        axes=axes,
        prop_conf=prop,
        raman_conf=raman,
        record_onaxis_rho_time=True,
        record_every_z=1,
    )

    np.testing.assert_array_equal(result["E_final"], np.asarray(E_direct))
    np.testing.assert_array_equal(result["I_final"], np.asarray(intensity(E_direct, beam.n0)))
    for key in (
        "z_axis",
        "U_z",
        "I_max_z",
        "rho_max_z",
        "rho_onaxis_max_z",
        "E_dep_z",
        "E_dep_rot_z",
        "E_dep_total_z",
        "z_edges",
        "dz_intervals",
        "deposition_channels",
        "deposition_q_shape",
    ):
        np.testing.assert_array_equal(result["diagnostics"][key], np.asarray(diag_direct[key]))
    assert int(result["diagnostics"]["n_intervals"]) == int(diag_direct["n_intervals"])

    assert RunConfig().Npulses == 1
