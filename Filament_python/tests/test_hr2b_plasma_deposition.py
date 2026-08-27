from __future__ import annotations

import numpy as np


def _constant_species_wfunc(rate: float, ui_j: float = 2.0, *, time_mode="full"):
    def wfunc(inp):
        return np.full_like(inp, rate, dtype=np.float64)

    wfunc._expects = "I"
    wfunc._time_mode = time_mode
    wfunc._integrator = "euler"
    wfunc._species_entries = ({
        "name": "synthetic",
        "fraction": 1.0,
        "W_s": wfunc,
        "W_runtime": wfunc,
        "Ui_J": ui_j,
    },)
    return wfunc


def test_photoionization_source_excludes_recombination_and_preserves_zero_beta_compatibility():
    from KHz_filament.ionization.runtime import evolve_rho_time

    inp = np.ones((5, 1, 1), dtype=np.float64)
    kwargs = dict(
        input_array=inp,
        dt=0.1,
        N0=10.0,
        Wfunc=_constant_species_wfunc(1.0),
        return_species_terms=True,
    )
    rho, _, beta_terms = evolve_rho_time(beta_rec=0.05, **kwargs)
    photo = beta_terms["photoionization_energy_rate"]
    net = beta_terms["drho_dt_u_sum"]
    assert np.all(photo >= 0.0)
    assert np.any(np.abs(photo - net) > 0.0)
    np.testing.assert_allclose(photo - net, 2.0 * 0.05 * rho * rho)

    _, _, zero_beta_terms = evolve_rho_time(
        beta_rec=0.0,
        **{**kwargs, "Wfunc": _constant_species_wfunc(1.0)},
    )
    np.testing.assert_allclose(
        zero_beta_terms["photoionization_energy_rate"],
        zero_beta_terms["drho_dt_u_sum"],
    )


def test_quasi_static_species_path_returns_explicit_photo_source():
    from KHz_filament.ionization.runtime import evolve_rho_time

    inp = np.ones((4, 2, 3), dtype=np.float64)
    _, _, terms = evolve_rho_time(
        inp,
        dt=0.2,
        N0=5.0,
        beta_rec=0.5,
        Wfunc=_constant_species_wfunc(0.25, time_mode="qs_peak"),
        quasi_static_time=True,
        time_stat="peak",
        return_species_terms=True,
    )
    assert terms["photoionization_energy_rate"].shape == inp.shape
    assert np.all(terms["photoionization_energy_rate"] >= 0.0)
    np.testing.assert_allclose(
        terms["photoionization_energy_rate"], terms["drho_dt_u_sum"]
    )


def test_q_maps_interval_energy_and_level1_direct_closure():
    from KHz_filament.deposition import (
        direct_interval_energy,
        interval_energy_from_q,
        q_ib_from_power,
        q_ion_from_power,
    )

    photo = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    alpha_ib = np.full_like(photo, 2.0)
    intensity = np.full_like(photo, 3.0)
    dt, dx, dy, dz = 0.5, 0.2, 0.3, 0.4

    q_ion = q_ion_from_power(photo, dt)
    q_ib = q_ib_from_power(alpha_ib, intensity, dt)
    np.testing.assert_allclose(q_ion, photo.sum(axis=0) * dt)
    np.testing.assert_allclose(q_ib, (alpha_ib * intensity).sum(axis=0) * dt)
    assert q_ion.shape == (2, 3)
    assert q_ib.shape == (2, 3)

    ion_from_q = interval_energy_from_q(q_ion, dx, dy, dz)
    ion_direct = direct_interval_energy(photo, dt, dx, dy, dz)
    ib_from_q = interval_energy_from_q(q_ib, dx, dy, dz)
    ib_direct = direct_interval_energy(alpha_ib * intensity, dt, dx, dy, dz)
    np.testing.assert_allclose(ion_from_q, ion_direct)
    np.testing.assert_allclose(ib_from_q, ib_direct)


def test_ib_disabled_is_exactly_zero():
    from KHz_filament.deposition import direct_interval_energy, q_ib_from_power

    alpha_ib = np.zeros((3, 2, 2), dtype=np.float64)
    intensity = np.ones_like(alpha_ib)
    assert np.array_equal(q_ib_from_power(alpha_ib, intensity, 0.25), np.zeros((2, 2)))
    assert direct_interval_energy(alpha_ib * intensity, 0.25, 1.0, 1.0, 1.0) == 0.0


def _tiny_components():
    from KHz_filament.config import (
        BeamConfig,
        GridConfig,
        HeatConfig,
        IonizationConfig,
        PropagationConfig,
        RamanConfig,
        RunConfig,
    )

    return dict(
        grid=GridConfig(Nx=8, Ny=8, Nt=16, Lx=8e-4, Ly=8e-4, Twin=160e-15),
        beam=BeamConfig(
            w0=1.5e-4,
            tau_fwhm=40e-15,
            energy_J=1e-9,
            focal_length=None,
        ),
        prop=PropagationConfig(
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
        ion=IonizationConfig(
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
        heat=HeatConfig(f_rep=1e3),
        run=RunConfig(Npulses=1),
        raman=RamanConfig(enabled=False, absorption=False),
        dtype="fp32",
    )


def test_tiny_cpu_runner_has_canonical_interval_ledger_and_level2_closure(tmp_path):
    from KHz_filament.runner import run_demo

    components = _tiny_components()
    components["ion"].sigma_ib = 0.0
    components["ion"].nu_ei_const = 0.0
    result = run_demo(
        **components,
        out_path=str(tmp_path / "hr2b_tiny.npz"),
        return_results=True,
    )
    diag = result["diagnostics"]
    n_intervals = int(diag["n_intervals"])
    assert n_intervals == len(diag["dz_intervals"])
    ion = np.asarray(diag["E_dep_ion_interval_J"])
    ib = np.asarray(diag["E_dep_ib_interval_J"])
    plasma = np.asarray(diag["E_dep_plasma_interval_J"])
    assert len(ion) == len(ib) == len(plasma) == n_intervals
    np.testing.assert_allclose(plasma, ion + ib)
    np.testing.assert_allclose(diag["E_dep_ion_pulse_J"], ion.sum())
    np.testing.assert_allclose(diag["E_dep_ib_pulse_J"], ib.sum())
    np.testing.assert_allclose(diag["E_dep_plasma_pulse_J"], plasma.sum())
    np.testing.assert_allclose(
        diag["E_dep_ion_interval_J"], diag["E_dep_ion_interval_direct_J"],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        diag["E_dep_ib_interval_J"], diag["E_dep_ib_interval_direct_J"],
        rtol=1e-6,
    )
    assert np.array_equal(ib, np.zeros_like(ib))
    assert np.array_equal(
        np.asarray(diag["alpha_ib_max_z"]),
        np.zeros_like(np.asarray(diag["alpha_ib_max_z"])),
    )
