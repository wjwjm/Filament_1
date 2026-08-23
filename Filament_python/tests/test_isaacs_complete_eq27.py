from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from KHz_filament.config import IonizationConfig, PropagationConfig, RamanConfig
from KHz_filament.config_normalize import normalize_config
from KHz_filament.constants import N0_air, Ui_N2, c0, eps0
from KHz_filament.grids import make_axes
from KHz_filament.propagate import propagate_one_pulse
import KHz_filament.raman as raman_module
from KHz_filament.raman import (
    apply_isaacs_complete_eq27_operator_step,
    isaacs_complete_eq27_stage,
    isaacs_raman_stage,
)


N0 = 1.00027
N2 = 7.8e-24
N_R = 2.3e-23
OMEGA_R = 1.6e13
GAMMA_R = 1.3e13
OMEGA0 = 2.0 * np.pi * c0 / 800e-9
DT = 2.5e-15
NT = 384


def _field(dtype=np.complex128):
    t = (np.arange(NT, dtype=np.float64) - NT // 2) * DT
    intensity = 5.0e17 * np.exp(-4.0 * np.log(2.0) * (t / 120e-15) ** 2)
    phase = 0.15 * np.sin(2.0 * np.pi * t / 85e-15) + 1.2e27 * t * t
    amplitude = np.sqrt(2.0 * intensity / (eps0 * c0 * N0))
    return (amplitude * np.exp(1j * phase)).astype(dtype)[:, None, None]


def _omega():
    return 2.0 * np.pi * np.fft.fftfreq(NT, d=DT)


def _reference_response(intensity):
    """Independent fp64 exact-PWL recurrence for the fixed audit envelope."""
    a = GAMMA_R - 1j * OMEGA_R
    r = np.exp(-a * DT)
    c = (1.0 - r) / a
    c1 = c - (1.0 - r * (1.0 + a * DT)) / (a * a * DT)
    c0_ = c - c1
    k = 1.0 / np.imag(1.0 / a)
    values = np.asarray(intensity, dtype=np.float64).reshape(NT, -1)
    response = np.zeros_like(values, dtype=np.float64)
    state = np.zeros(values.shape[1], dtype=np.complex128)
    for index in range(1, NT):
        state = r * state + c0_ * values[index - 1] + c1 * values[index]
        response[index] = np.imag(k * state)
    return response.reshape(np.asarray(intensity).shape)


def _reference_rhs(field, *, n2, n_R):
    intensity = 0.5 * eps0 * c0 * N0 * np.abs(field) ** 2
    response = _reference_response(intensity)
    source = (float(n2) * intensity + float(n_R) * response) * field
    derivative = np.fft.ifft(
        (1j * _omega())[:, None, None] * np.fft.fft(source, axis=0), axis=0
    )
    return 1j * (OMEGA0 / c0) * source - derivative / c0


def _rhs_kwargs():
    return dict(
        Omega=_omega(), dt=DT, omega0=OMEGA0, n0=N0,
        n2=N2, n_R=N_R, omega_R=OMEGA_R, Gamma_R=GAMMA_R,
        method="iir", iir_sampling="exact_piecewise_linear",
    )


def _rel_l2(actual, expected):
    return float(np.linalg.norm(np.asarray(actual) - np.asarray(expected)) / max(
        np.linalg.norm(np.asarray(expected)), 1e-300
    ))


def test_electronic_rotational_and_combined_closure_against_direct_fp64():
    field = _field()
    stage = isaacs_complete_eq27_stage(
        field, return_components=True, **_rhs_kwargs()
    )
    electronic_reference = _reference_rhs(field, n2=N2, n_R=0.0)
    rotational_reference = _reference_rhs(field, n2=0.0, n_R=N_R)
    combined_reference = _reference_rhs(field, n2=N2, n_R=N_R)
    assert _rel_l2(stage["rhs_electronic"], electronic_reference) < 1e-6
    assert _rel_l2(stage["rhs_rotational"], rotational_reference) < 1e-6
    assert _rel_l2(stage["rhs"], combined_reference) < 1e-6
    assert _rel_l2(stage["rhs"], stage["rhs_electronic"] + stage["rhs_rotational"]) < 1e-12


def test_combined_stage_uses_one_derivative_pair_until_components_are_requested(monkeypatch):
    field = _field()
    counts = {"fft": 0, "ifft": 0}
    original_fft = raman_module.xp.fft.fft
    original_ifft = raman_module.xp.fft.ifft

    def counted_fft(*args, **kwargs):
        counts["fft"] += 1
        return original_fft(*args, **kwargs)

    def counted_ifft(*args, **kwargs):
        counts["ifft"] += 1
        return original_ifft(*args, **kwargs)

    monkeypatch.setattr(raman_module.xp.fft, "fft", counted_fft)
    monkeypatch.setattr(raman_module.xp.fft, "ifft", counted_ifft)
    isaacs_complete_eq27_stage(
        field, return_response=False, return_energy=False, **_rhs_kwargs()
    )
    assert counts == {"fft": 1, "ifft": 1}
    counts.update(fft=0, ifft=0)
    isaacs_complete_eq27_stage(
        field, return_response=False, return_energy=False,
        return_components=True, **_rhs_kwargs()
    )
    assert counts == {"fft": 2, "ifft": 2}


def test_coefficients_are_single_counted_and_zeroable():
    field = _field()
    both = isaacs_complete_eq27_stage(field, return_components=True, **_rhs_kwargs())
    electronic_only = isaacs_complete_eq27_stage(
        field, return_components=True, **{**_rhs_kwargs(), "n_R": 0.0}
    )
    rotational_only = isaacs_complete_eq27_stage(
        field, return_components=True, **{**_rhs_kwargs(), "n2": 0.0}
    )
    assert np.allclose(electronic_only["rhs_rotational"], 0.0)
    assert np.allclose(rotational_only["rhs_electronic"], 0.0)
    assert _rel_l2(electronic_only["rhs"], both["rhs_electronic"]) < 1e-12
    assert _rel_l2(rotational_only["rhs"], both["rhs_rotational"]) < 1e-12

    doubled_electronic = isaacs_complete_eq27_stage(
        field, return_components=True, **{**_rhs_kwargs(), "n2": 2.0 * N2, "n_R": 0.0}
    )
    doubled_rotational = isaacs_complete_eq27_stage(
        field, return_components=True, **{**_rhs_kwargs(), "n2": 0.0, "n_R": 2.0 * N_R}
    )
    assert _rel_l2(doubled_electronic["rhs"], 2.0 * both["rhs_electronic"]) < 1e-12
    assert _rel_l2(doubled_rotational["rhs"], 2.0 * both["rhs_rotational"]) < 1e-12


def test_eq27_sign_and_vacuum_prefactor_are_exact():
    field = _field()
    stage = isaacs_complete_eq27_stage(field, return_components=True, **_rhs_kwargs())
    intensity = 0.5 * eps0 * c0 * N0 * np.abs(field) ** 2
    response = _reference_response(intensity)
    source = (N2 * intensity + N_R * response) * field
    derivative = np.fft.ifft(
        (1j * _omega())[:, None, None] * np.fft.fft(source, axis=0), axis=0
    )
    expected = 1j * (OMEGA0 / c0) * source - derivative / c0
    assert _rel_l2(stage["rhs"], expected) < 1e-6
    wrong_medium_prefactor = 1j * (N0 * OMEGA0 / c0) * source - derivative / c0
    assert _rel_l2(stage["rhs"], wrong_medium_prefactor) > 1e-5


def test_heun_error_decreases_when_dz_is_halved():
    field = _field()
    kwargs = _rhs_kwargs()
    dz = 1.0e-5
    whole = apply_isaacs_complete_eq27_operator_step(
        field, dz, integrator="heun", **kwargs
    )
    half = apply_isaacs_complete_eq27_operator_step(
        field, dz / 2.0, integrator="heun", **kwargs
    )
    half = apply_isaacs_complete_eq27_operator_step(
        half, dz / 2.0, integrator="heun", **kwargs
    )
    difference = float(np.linalg.norm(whole - half))
    finer = apply_isaacs_complete_eq27_operator_step(
        field, dz / 4.0, integrator="heun", **kwargs
    )
    finer = apply_isaacs_complete_eq27_operator_step(
        finer, dz / 4.0, integrator="heun", **kwargs
    )
    finer = apply_isaacs_complete_eq27_operator_step(
        finer, dz / 4.0, integrator="heun", **kwargs
    )
    finer = apply_isaacs_complete_eq27_operator_step(
        finer, dz / 4.0, integrator="heun", **kwargs
    )
    finer_difference = float(np.linalg.norm(half - finer))
    assert difference > 0.0 and finer_difference > 0.0
    assert difference / finer_difference > 2.5


def test_complex64_projection_is_reported_without_changing_the_strategy():
    field = _field(np.complex64)
    _, diagnostics = apply_isaacs_complete_eq27_operator_step(
        field, 1.0e-5, integrator="heun", return_diagnostics=True,
        diagnose_projection_difference=True, **_rhs_kwargs()
    )
    assert diagnostics["projection_difference_diagnostics_enabled"]
    assert np.isfinite(diagnostics["energy_projection_scale"])
    assert np.isfinite(diagnostics["projection_field_relative_l2"])
    assert np.isfinite(diagnostics["projection_energy_difference_relative"])
    if diagnostics["projection_field_relative_l2"] < 1e-7:
        assert diagnostics["projection_status"] == "not_primary_for_C2"


def test_pure_complex128_field_vs_eq10_energy_closure_uses_long_window():
    field = _field(np.complex128)
    _, diagnostics = apply_isaacs_complete_eq27_operator_step(
        field, 1.0e-5, integrator="heun", return_diagnostics=True,
        diagnose_projection_difference=True, **_rhs_kwargs()
    )
    edge_ratio = max(np.max(np.abs(field[0])), np.max(np.abs(field[-1]))) / np.max(np.abs(field))
    assert edge_ratio < 1e-6
    assert diagnostics["global_closure_residual"] < 1e-6
    assert diagnostics["global_closure_residual"] == pytest.approx(3.36e-8, rel=0.1)


def test_old_mode_and_default_configuration_remain_unchanged():
    normalized = normalize_config({})
    assert normalized["raman"]["operator_mode"] == "legacy_split"
    assert normalized["raman"]["operator_convention"] == "legacy"
    assert RamanConfig().operator_mode == "legacy_split"
    assert PropagationConfig().use_raman_full_operator is None

    field = _field()
    kwargs = dict(
        Omega=_omega(), dt=DT, omega0=OMEGA0, n0=N0,
        n_R=N_R, omega_R=OMEGA_R, Gamma_R=GAMMA_R,
        method="iir", iir_sampling="exact_piecewise_linear",
    )
    old_stage = isaacs_raman_stage(field, **kwargs)
    old_reference = _reference_rhs(field, n2=0.0, n_R=N_R)
    assert _rel_l2(old_stage["rhs"], old_reference) < 1e-6


def test_complete_mode_configuration_is_opt_in_and_keeps_full_operator_coupling():
    raw = {
        "propagation": {
            "use_raman_full_operator": True,
            "use_raman_phase": False,
            "use_raman_absorption": False,
        },
        "raman": {
            "model": "isaacs_rot_sinexp", "n_R": N_R,
            "omega_R": OMEGA_R, "Gamma_R": GAMMA_R,
            "operator_mode": "full_isaacs_eq27_complete", "absorption": False,
        },
    }
    assert normalize_config(raw)["raman"]["operator_mode"] == "full_isaacs_eq27_complete"


def test_complete_propagation_traces_operator_and_scalar_phase_semantics():
    nx = ny = 8
    nt = 16
    axes = make_axes(nx, ny, nt, Lx=8e-4, Ly=8e-4, Twin=160e-15)
    t = np.asarray(axes.t)
    x = np.asarray(axes.x)
    y = np.asarray(axes.y)
    temporal = np.exp(-2.0 * np.log(2.0) * (t / 40e-15) ** 2)
    transverse = np.exp(-((x[None, :] / 2.0e-4) ** 2 + (y[:, None] / 2.0e-4) ** 2))
    field = (1.0e7 * temporal[:, None, None] * transverse[None, ...]).astype(np.complex64)
    n0 = 1.00027
    omega0 = 2.0 * np.pi * c0 / 800e-9
    prop = PropagationConfig(
        linear_model="paraxial",
        use_self_steepening=True,
        use_electronic_kerr=True,
        use_raman_phase=False,
        use_raman_full_operator=True,
        use_plasma_phase=False,
        use_ionization_loss=False,
        use_ionization_solver=False,
        use_raman_absorption=False,
        focus_window_step=False,
        energy_probe_every=0,
        progress_every_z=0,
    )
    raman = RamanConfig(
        enabled=True,
        model="isaacs_rot_sinexp",
        T2=None,
        T_R=None,
        omega_R=OMEGA_R,
        Gamma_R=GAMMA_R,
        n_R=N_R,
        operator_mode="full_isaacs_eq27_complete",
        operator_convention="isaacs_eq27",
        iir_sampling="exact_piecewise_linear",
        operator_integrator="heun",
        nonlinear_split_order="after_other",
        absorption=False,
    )
    _, _, diagnostics = propagate_one_pulse(
        field,
        kperp2=axes.kperp2,
        k0=n0 * omega0 / c0,
        omega0=omega0,
        dz=1.0e-5,
        z_max=1.0e-5,
        n0=n0,
        n2=N2,
        Ui=Ui_N2,
        N0=N0_air,
        ion_conf=IonizationConfig(species=[]),
        dn_gas=np.zeros((ny, nx), dtype=np.float32),
        dt=float(axes.dt),
        axes=axes,
        prop_conf=prop,
        raman_conf=raman,
        record_every_z=1,
    )
    assert np.all(diagnostics["raman_operator_applied"])
    assert np.max(diagnostics["raman_rhs_l2_norm"]) > 0.0
    assert np.all(diagnostics["dphi_kerr_max_abs_z"] == 0.0)
    assert np.max(diagnostics["delta_n_elec_applied_max_z"]) > 0.0
    assert diagnostics["delta_n_elec_applied_semantics"].item() == (
        "equivalent_n2_I_trace_full_complex_operator"
    )
    assert diagnostics["dphi_kerr_semantics"].item() == (
        "not_applicable_scalar_phase_full_complex_operator"
    )
    assert diagnostics["self_steepening_semantics"].item() == (
        "full_product_derivative_D_S_in_complete_complex_operator"
    )
