from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _components(
    *, mode: str, z_start: float, z_max: float, dz: float,
    linear_model: str = "paraxial", **switches,
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

    defaults = {
        "use_self_steepening": True,
        "use_electronic_kerr": True,
        "use_raman_phase": False,
        "use_raman_full_operator": True,
        "use_plasma_phase": True,
        "use_ionization_loss": True,
        "use_raman_absorption": False,
        "use_ionization_solver": True,
    }
    defaults.update(switches)
    return {
        "grid": GridConfig(Nx=8, Ny=8, Nt=32, Lx=8e-4, Ly=8e-4, Twin=320e-15),
        "beam": BeamConfig(
            w0=1.5e-4, tau_fwhm=120e-15, energy_J=1e-9, focal_length=None
        ),
        "prop": PropagationConfig(
            propagation_mode=mode,
            z_nl_start=z_start,
            z_max=z_max,
            dz=dz,
            linear_model=linear_model,
            auto_substep=False,
            focus_window_step=False,
            limit_focus_window=False,
            progress_every_z=0,
            energy_probe_every=0,
            diag_extra=True,
            measure_performance=True,
            **defaults,
        ),
        "ion": IonizationConfig(
            species=[{
                "name": "test",
                "rate": "mpa_fact",
                "ell": 2,
                "I_mp": 1e18,
                "Ip_eV": 15.0,
                "fraction": 1.0,
            }],
            I_cap=1e19,
            W_cap=1e19,
        ),
        "heat": HeatConfig(f_rep=1e3),
        "run": RunConfig(Npulses=1),
        "raman": RamanConfig(
            enabled=True,
            model="isaacs_rot_sinexp",
            diagnose=True,
            absorption=False,
            n_R=2.3e-23,
            omega_R=1.6e13,
            Gamma_R=1.3e13,
            operator_mode="full_isaacs_eq27",
            operator_convention="isaacs_eq27",
            iir_sampling="exact_piecewise_linear",
            operator_integrator="heun",
            nonlinear_split_order="strang",
        ),
    }


def _run(tmp_path, name: str, *, dtype: str = "fp64", **kwargs):
    from KHz_filament.runner import run_demo

    return run_demo(
        **_components(**kwargs),
        out_path=str(tmp_path / f"{name}.npz"),
        dtype=dtype,
        return_results=True,
    )


def _relative_l2(left, right) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    return float(np.linalg.norm((left - right).ravel()) / max(np.linalg.norm(right.ravel()), 1e-300))


def test_config_defaults_and_hybrid_validation(tmp_path):
    from KHz_filament.confio import load_all
    from KHz_filament.config_normalize import normalize_config

    legacy = normalize_config({"propagation": {"z_max": 1.3, "limit_focus_window": False}})
    assert legacy["propagation"]["propagation_mode"] == "full_nonlinear_from_z0"
    assert legacy["propagation"]["z_nl_start"] == 0.0

    valid = normalize_config({
        "propagation": {
            "propagation_mode": "hybrid", "z_nl_start": 0.6,
            "z_max": 1.3, "limit_focus_window": False,
        },
        "run": {"Npulses": 1},
    })
    assert valid["propagation"]["z_nl_start"] == 0.6
    assert valid["run"]["Npulses"] == 1
    assert isinstance(valid["run"]["Npulses"], int)

    path = tmp_path / "hybrid.json"
    path.write_text(json.dumps(valid), encoding="utf-8")
    _, _, prop, *_ = load_all(str(path))
    assert prop.propagation_mode == "hybrid" and prop.z_nl_start == 0.6

    invalid = [
        {"propagation_mode": "unknown", "z_nl_start": 0.0, "z_max": 1.3, "limit_focus_window": False},
        {"propagation_mode": "hybrid", "z_nl_start": 0.0, "z_max": 1.3, "limit_focus_window": False},
        {"propagation_mode": "hybrid", "z_nl_start": 1.3, "z_max": 1.3, "limit_focus_window": False},
        {"propagation_mode": "hybrid", "z_nl_start": 0.6, "z_max": 1.3, "limit_focus_window": True},
        {"propagation_mode": "full_nonlinear_from_z0", "z_nl_start": 0.6, "z_max": 1.3, "limit_focus_window": False},
    ]
    for propagation in invalid:
        with pytest.raises(ValueError):
            normalize_config({"propagation": propagation})
    with pytest.raises(ValueError, match="restricted to run.Npulses=1"):
        normalize_config({
            "propagation": {
                "propagation_mode": "hybrid", "z_nl_start": 0.6,
                "z_max": 1.3, "limit_focus_window": False,
            },
            "run": {"Npulses": 2},
        })
    with pytest.raises(ValueError, match="run.Npulses must be an integer"):
        normalize_config({
            "propagation": {
                "propagation_mode": "hybrid", "z_nl_start": 0.6,
                "z_max": 1.3, "limit_focus_window": False,
            },
            "run": {"Npulses": 1.5},
        })


def test_run_demo_rejects_direct_multi_pulse_hybrid_entry(tmp_path):
    from KHz_filament.config import RunConfig
    from KHz_filament.runner import run_demo

    for value in (2, 1.0, "1", True):
        components = _components(mode="hybrid", z_start=0.6, z_max=0.8, dz=0.1)
        components["run"] = RunConfig(Npulses=value)
        with pytest.raises(ValueError, match="hybrid propagation v1 requires run.Npulses=1"):
            run_demo(
                **components,
                out_path=str(tmp_path / f"must_not_run_{value!s}.npz"),
                dtype="fp32",
            )


def test_h1_unaligned_boundary_clips_and_starts_nonlinearity_at_boundary(tmp_path):
    result = _run(
        tmp_path, "unaligned", mode="hybrid", z_start=0.6, z_max=0.9, dz=0.35
    )
    diag = result["diagnostics"]
    np.testing.assert_allclose(diag["step_start_z_m"], [0.0, 0.35, 0.6], rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(diag["step_end_z_m"], [0.35, 0.6, 0.9], rtol=0.0, atol=2e-15)
    np.testing.assert_array_equal(diag["nonlinear_operator_applied"], [False, False, True])
    np.testing.assert_array_equal(diag["nonlinear_operator_call_count_step"], [0, 0, 1])
    np.testing.assert_array_equal(diag["ionization_solver_call_count_step"], [0, 0, 1])
    assert np.all(np.asarray(diag["rho_max_z"])[:2] == 0.0)
    assert np.all(np.asarray(diag["E_dep_total_z"])[:2] == 0.0)
    assert np.all(np.asarray(diag["raman_operator_substep_count"])[:2] == 0)
    assert np.all(np.asarray(diag["raman_convolution_count_step"])[:2] == 0)
    assert not np.any(np.asarray(diag["raman_operator_applied"])[:2])
    for key in (
        "alpha_R_max_z", "alpha_ion_raw_max_z", "alpha_ion_applied_max_z",
        "alpha_ib_max_z", "alpha_total_max_z", "delta_n_plasma_min_z",
        "dphi_kerr_max_abs_z", "dphi_elec_max_abs_z", "dphi_rot_max_abs_z",
        "dphi_plasma_max_abs_z", "nonlinear_walltime_step_s",
        "ionization_walltime_step_s",
    ):
        assert np.all(np.asarray(diag[key])[:2] == 0.0), key
    assert np.asarray(diag["raman_operator_substep_count"])[2] == 2


def test_h1_aligned_0p60_has_no_roundoff_microstep(tmp_path):
    result = _run(
        tmp_path, "aligned", mode="hybrid", z_start=0.6, z_max=0.8, dz=0.1
    )
    diag = result["diagnostics"]
    starts = np.asarray(diag["step_start_z_m"], dtype=float)
    ends = np.asarray(diag["step_end_z_m"], dtype=float)
    active = np.asarray(diag["nonlinear_operator_applied"], dtype=bool)
    assert starts.size == 8
    assert ends[5] == np.float64(0.6)
    assert starts[6] == np.float64(0.6)
    assert np.min(ends - starts) > 0.099999999999
    np.testing.assert_array_equal(active, [False] * 6 + [True] * 2)


def test_h1_production_0p60_is_aligned_before_focus_schedule():
    mother = Path(__file__).resolve().parents[1] / (
        "configs/isaacs_raman_closure/120fs_talebpour_isaacs_full_operator_on.json"
    )
    config = json.loads(mother.read_text(encoding="utf-8"))
    propagation = config["propagation"]
    z_start = 0.60
    dz = float(propagation["dz"])
    steps = z_start / dz
    assert steps == pytest.approx(round(steps), rel=0.0, abs=1e-12)
    assert propagation["limit_focus_window"] is False
    assert float(propagation["focus_center_m"]) - float(propagation["focus_halfwidth_m"]) > z_start


def test_h2_linear_segment_matches_existing_pure_linear_path(tmp_path):
    switches = dict(
        use_self_steepening=False,
        use_electronic_kerr=False,
        use_raman_phase=False,
        use_raman_full_operator=False,
        use_plasma_phase=False,
        use_ionization_loss=False,
        use_raman_absorption=False,
        use_ionization_solver=False,
    )
    hybrid = _run(
        tmp_path, "hybrid_linear", mode="hybrid", z_start=0.6,
        z_max=0.8, dz=0.1, linear_model="bk_nee", **switches,
    )
    reference = _run(
        tmp_path, "reference_linear", mode="full_nonlinear_from_z0", z_start=0.0,
        z_max=0.8, dz=0.1, linear_model="bk_nee", **switches,
    )

    field_h = np.asarray(hybrid["E_final"])
    field_r = np.asarray(reference["E_final"])
    intensity_h = np.abs(field_h) ** 2
    intensity_r = np.abs(field_r) ** 2
    spectrum_h = np.abs(np.fft.fft(field_h, axis=0))
    spectrum_r = np.abs(np.fft.fft(field_r, axis=0))
    phase_mask = (np.abs(field_r) >= 1e-8 * np.max(np.abs(field_r)))
    phase_rms = float(np.sqrt(np.mean(np.angle(field_h[phase_mask] * np.conj(field_r[phase_mask])) ** 2)))

    assert _relative_l2(field_h, field_r) <= 1e-7
    assert _relative_l2(intensity_h, intensity_r) <= 1e-7
    assert _relative_l2(spectrum_h, spectrum_r) <= 1e-7
    energy_h = float(np.sum(intensity_h))
    energy_r = float(np.sum(intensity_r))
    assert abs(energy_h - energy_r) / energy_r <= 1e-7
    assert phase_rms <= 1e-6

    hdiag = hybrid["diagnostics"]
    rdiag = reference["diagnostics"]
    boundary = int(np.flatnonzero(np.asarray(hdiag["step_end_z_m"]) == np.float64(0.6))[0])
    for key in ("U_z", "I_max_z", "I_onaxis_max_z", "I_center_t0_z", "w_mom_z"):
        np.testing.assert_allclose(hdiag[key][boundary], rdiag[key][boundary], rtol=1e-7, atol=1e-12)


def test_fp32_h1_and_h2_production_precision_contract(tmp_path):
    switches = dict(
        use_self_steepening=False,
        use_electronic_kerr=False,
        use_raman_phase=False,
        use_raman_full_operator=False,
        use_plasma_phase=False,
        use_ionization_loss=False,
        use_raman_absorption=False,
        use_ionization_solver=False,
    )
    unaligned = _run(
        tmp_path, "hybrid_unaligned_fp32", dtype="fp32", mode="hybrid",
        z_start=0.6, z_max=0.8, dz=0.35, linear_model="bk_nee", **switches,
    )
    udiag = unaligned["diagnostics"]
    np.testing.assert_allclose(udiag["step_start_z_m"], [0.0, 0.35, 0.6], rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(udiag["step_end_z_m"], [0.35, 0.6, 0.8], rtol=0.0, atol=2e-15)
    np.testing.assert_array_equal(udiag["nonlinear_operator_applied"], [False, False, True])

    # H2 must use the same two-half-step schedule in both cases.  An unaligned
    # boundary intentionally changes the z partition and is therefore only an
    # H1 boundary test, not a valid pure-linear equivalence comparison.
    hybrid = _run(
        tmp_path, "hybrid_linear_fp32", dtype="fp32", mode="hybrid",
        z_start=0.6, z_max=0.8, dz=0.1, linear_model="bk_nee", **switches,
    )
    reference = _run(
        tmp_path, "reference_linear_fp32", dtype="fp32",
        mode="full_nonlinear_from_z0", z_start=0.0, z_max=0.8, dz=0.1,
        linear_model="bk_nee", **switches,
    )
    hdiag = hybrid["diagnostics"]
    np.testing.assert_array_equal(hdiag["nonlinear_operator_applied"], [False] * 6 + [True] * 2)
    field_h = np.asarray(hybrid["E_final"])
    field_r = np.asarray(reference["E_final"])
    intensity_h = np.abs(field_h) ** 2
    intensity_r = np.abs(field_r) ** 2
    spectrum_h = np.abs(np.fft.fft(field_h, axis=0))
    spectrum_r = np.abs(np.fft.fft(field_r, axis=0))
    phase_mask = np.abs(field_r) >= 1e-8 * np.max(np.abs(field_r))
    phase_rms = float(np.sqrt(np.mean(np.angle(field_h[phase_mask] * np.conj(field_r[phase_mask])) ** 2)))
    assert field_h.dtype == np.complex64
    assert _relative_l2(field_h, field_r) <= 1e-7
    assert _relative_l2(intensity_h, intensity_r) <= 1e-7
    assert _relative_l2(spectrum_h, spectrum_r) <= 1e-7
    energy_h = float(np.sum(intensity_h))
    energy_r = float(np.sum(intensity_r))
    assert abs(energy_h - energy_r) / energy_r <= 1e-7
    assert phase_rms <= 1e-6


def test_default_and_explicit_reference_modes_match(tmp_path):
    from KHz_filament.config import PropagationConfig
    from KHz_filament.runner import run_demo

    explicit = _run(
        tmp_path, "explicit_full", mode="full_nonlinear_from_z0",
        z_start=0.0, z_max=3e-4, dz=1e-4,
    )
    components = _components(
        mode="full_nonlinear_from_z0", z_start=0.0, z_max=3e-4, dz=1e-4
    )
    components["prop"] = PropagationConfig(
        **{**components["prop"].__dict__, "propagation_mode": "full_nonlinear_from_z0", "z_nl_start": 0.0}
    )
    default = run_demo(
        **components, out_path=str(tmp_path / "default_full.npz"),
        dtype="fp64", return_results=True,
    )
    np.testing.assert_array_equal(default["E_final"], explicit["E_final"])
    for key in ("U_z", "I_max_z", "rho_max_z", "raman_operator_substep_count"):
        np.testing.assert_array_equal(default["diagnostics"][key], explicit["diagnostics"][key])
