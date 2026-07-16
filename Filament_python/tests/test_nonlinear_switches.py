from __future__ import annotations

import json

import numpy as np


def _small_components(*, self_steepening: bool = False, **switches):
    from KHz_filament.config import (
        BeamConfig,
        GridConfig,
        HeatConfig,
        IonizationConfig,
        PropagationConfig,
        RamanConfig,
        RunConfig,
    )

    prop = PropagationConfig(
        z_max=2e-4,
        dz=1e-4,
        linear_model="paraxial",
        auto_substep=False,
        focus_window_step=False,
        limit_focus_window=False,
        progress_every_z=0,
        diag_extra=False,
        energy_probe_every=0,
        use_self_steepening=self_steepening,
        **switches,
    )
    return {
        "grid": GridConfig(Nx=8, Ny=8, Nt=16, Lx=8e-4, Ly=8e-4, Twin=160e-15),
        "beam": BeamConfig(w0=1.5e-4, tau_fwhm=40e-15, energy_J=1e-9, focal_length=None),
        "prop": prop,
        "ion": IonizationConfig(
            species=[{
                "name": "test",
                "rate": "mpa_fact",
                "ell": 2,
                "I_mp": 1e18,
                "Ip_eV": 15.0,
                "fraction": 1.0,
            }]
        ),
        "heat": HeatConfig(f_rep=1e3),
        "run": RunConfig(Npulses=1),
        "raman": RamanConfig(enabled=True, absorption=True, absorption_model="closed_form"),
    }


def _run(tmp_path, name: str, *, self_steepening: bool = False, **switches):
    from KHz_filament.runner import run_demo

    out_path = tmp_path / f"{name}.npz"
    run_demo(**_small_components(self_steepening=self_steepening, **switches), out_path=str(out_path), dtype="fp32")
    return out_path


def _load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def test_default_switches_match_explicit_full_model(tmp_path):
    default_path = _run(tmp_path, "legacy_default", self_steepening=True)
    explicit_path = _run(
        tmp_path,
        "explicit_full",
        self_steepening=True,
        use_electronic_kerr=True,
        use_raman_phase=True,
        use_plasma_phase=True,
        use_ionization_loss=True,
        use_raman_absorption=True,
        use_ionization_solver=True,
    )
    default = _load(default_path)
    explicit = _load(explicit_path)
    for key in ("I_out_center_t", "U_z", "I_max_z", "rho_max_z", "dphi_kerr_max_abs_z", "alpha_total_max_z"):
        np.testing.assert_allclose(default[key], explicit[key], rtol=2e-6, atol=1e-12)


def test_electronic_kerr_off_keeps_raw_diagnostic_but_not_applied_phase(tmp_path):
    data = _load(_run(tmp_path, "electronic_off", use_electronic_kerr=False))
    assert np.max(data["delta_n_elec_max_z"]) > 0.0
    assert np.all(data["delta_n_elec_applied_max_z"] == 0.0)
    assert np.all(data["dphi_elec_applied_max_abs_z"] == 0.0)


def test_raman_phase_and_absorption_switches_are_independent(tmp_path):
    phase_off = _load(_run(tmp_path, "raman_phase_off", use_raman_phase=False, use_raman_absorption=True))
    assert np.max(phase_off["IR_abs_max_z"]) > 0.0
    assert np.max(phase_off["delta_n_rot_max_z"]) > 0.0
    assert np.all(phase_off["delta_n_rot_applied_max_z"] == 0.0)
    assert np.max(phase_off["alpha_R_applied_max_z"]) > 0.0

    absorption_off = _load(_run(tmp_path, "raman_absorption_off", use_raman_phase=True, use_raman_absorption=False))
    assert np.max(absorption_off["delta_n_rot_applied_max_z"]) > 0.0
    assert np.max(absorption_off["alpha_R_raw_max_z"]) > 0.0
    assert np.all(absorption_off["alpha_R_applied_max_z"] == 0.0)


def test_plasma_phase_and_ionization_loss_off_keep_ionization_diagnostics(tmp_path):
    plasma_off = _load(_run(tmp_path, "plasma_off", use_plasma_phase=False))
    assert np.max(plasma_off["rho_max_z"]) > 0.0
    assert np.max(plasma_off["dphi_plasma_raw_max_abs_z"]) > 0.0
    assert np.all(plasma_off["dphi_plasma_applied_max_abs_z"] == 0.0)

    loss_off = _load(_run(tmp_path, "ionization_loss_off", use_ionization_loss=False))
    assert np.max(loss_off["rho_max_z"]) > 0.0
    assert np.max(loss_off["alpha_ion_raw_max_z"]) > 0.0
    assert np.all(loss_off["alpha_ion_applied_max_z"] == 0.0)


def test_legacy_json_uses_legacy_raman_defaults(tmp_path):
    from KHz_filament.config import resolve_nonlinear_switches
    from KHz_filament.confio import load_all

    cfg = {
        "grid": {"Nx": 8, "Ny": 8, "Nt": 16, "Lx": 8e-4, "Ly": 8e-4, "Twin": 160e-15},
        "beam": {"energy_J": 1e-9, "P0_peak": None},
        "ionization": {"species": []},
        "raman": {"enabled": False, "absorption": True},
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    _, _, prop, ion, _, _, raman = load_all(str(path))
    resolved = resolve_nonlinear_switches(prop, raman, ion)
    assert prop.use_raman_phase is None
    assert prop.use_raman_absorption is None
    assert resolved.use_raman_phase is False
    assert resolved.use_raman_absorption is False
    assert resolved.use_ionization_solver is False

    cfg["propagation"] = {"use_raman_phase": True, "use_raman_absorption": True}
    explicit_path = tmp_path / "explicit_overrides_legacy.json"
    explicit_path.write_text(json.dumps(cfg), encoding="utf-8")
    _, _, explicit_prop, explicit_ion, _, _, explicit_raman = load_all(str(explicit_path))
    explicit = resolve_nonlinear_switches(explicit_prop, explicit_raman, explicit_ion)
    assert explicit.use_raman_phase is True
    assert explicit.use_raman_absorption is True
    assert explicit.compute_raman_convolution is True
