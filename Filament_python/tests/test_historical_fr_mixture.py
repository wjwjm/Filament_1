from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from KHz_filament.config import IonizationConfig, PropagationConfig, RamanConfig
from KHz_filament.config_normalize import _normalize_raman
from KHz_filament.constants import N0_air, Ui_N2, c0
from KHz_filament.grids import make_axes
from KHz_filament.nonlinear import kerr_phase, kerr_phase_from_deltan, shock_intensity
from KHz_filament.propagate import propagate_one_pulse
from KHz_filament.raman import (
    historical_fr_mixture_index,
    historical_fr_mixture_response,
)


def _historical_iir_4c330ac(intensity: np.ndarray, dt: float, T2: float, T_R: float) -> np.ndarray:
    """Replicate the 2026-03-18 (4c330ac) IIR loop exactly."""
    dtype = intensity.dtype
    omega = 2.0 * np.pi / float(T_R)
    gamma = 1.0 / float(T2)
    ctype = np.complex64 if dtype == np.float32 else np.complex128
    a = np.array(gamma - 1j * omega, dtype=ctype)
    r = np.exp(-a * dt)
    c = (1.0 - r) / a
    inv_a = 1.0 / a
    denom = np.imag(inv_a) + np.array(1e-300, dtype=inv_a.real.dtype)
    k = 1.0 / denom
    S = np.array(0.0, dtype=ctype)
    out = np.empty_like(intensity, dtype=dtype)
    for n in range(intensity.shape[0]):
        S = r * S + c * intensity[n]
        out[n] = np.imag(k * S).astype(dtype, copy=False)
    return out


def _direct_unit_area_convolution(intensity: np.ndarray, dt: float, T2: float, T_R: float) -> np.ndarray:
    omega = 2.0 * np.pi / float(T_R)
    gamma = 1.0 / float(T2)
    t = np.arange(intensity.shape[0], dtype=np.float64) * dt
    pref = (omega * omega + gamma * gamma) / omega
    # Analytic prefactor already normalizes the infinite-domain kernel to
    # unit area; do not renormalize over the truncated simulation window.
    h = pref * np.exp(-gamma * t) * np.sin(omega * t)
    return np.convolve(intensity, h, mode="full")[: intensity.shape[0]] * dt


def _gaussian_120fs(Nt: int = 384, dt: float = 2.5e-15) -> np.ndarray:
    t = np.arange(Nt) * dt
    center = t[Nt // 2]
    return np.exp(-4.0 * np.log(2.0) * ((t - center) / 120e-15) ** 2)


@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-9), (np.float32, 1e-5)])
def test_iir_matches_4c330ac_historical_response(dtype, tol):
    dt = 2.5e-15
    I1d = _gaussian_120fs(384, dt).astype(dtype)
    I = I1d[:, None, None]
    actual = np.asarray(
        historical_fr_mixture_response(I, dt=dt, T2=80e-12, T_R=8.4e-12)
    )[:, 0, 0]
    reference = _historical_iir_4c330ac(I1d, dt, T2=80e-12, T_R=8.4e-12)
    assert np.max(np.abs(actual - reference)) / max(np.max(np.abs(reference)), 1e-30) < tol


def test_iir_matches_direct_unit_area_kernel():
    dt = 2.5e-15
    I1d = _gaussian_120fs(384, dt)
    I = I1d[:, None, None]
    iir = np.asarray(
        historical_fr_mixture_response(I, dt=dt, T2=80e-12, T_R=8.4e-12)
    )[:, 0, 0]
    direct = _direct_unit_area_convolution(I1d, dt, T2=80e-12, T_R=8.4e-12)
    assert np.max(np.abs(iir - direct)) / max(np.max(np.abs(direct)), 1e-30) < 5e-2


def test_mixture_index_and_phase_match_historical():
    dt = 2.5e-15
    Nt = 384
    I1d = _gaussian_120fs(Nt, dt)
    I = I1d[:, None, None]
    IR = historical_fr_mixture_response(I, dt=dt, T2=80e-12, T_R=8.4e-12)
    f_R = 0.15
    n2 = 7.8e-24
    dn = historical_fr_mixture_index(I, IR, f_R=f_R, n2=n2)
    assert np.allclose(dn, n2 * ((1.0 - f_R) * I + f_R * IR))

    omega0 = 2.0 * np.pi * c0 / 800e-9
    k0 = 1.00027 * omega0 / c0
    dz = 1e-4
    Omega = 2.0 * np.pi * np.fft.fftfreq(Nt, d=dt)
    I_nl = (1.0 - f_R) * I + f_R * IR
    ours = kerr_phase_from_deltan(
        shock_intensity(dn, Omega, omega0, dt=dt, method="tdiff"), k0, dz)
    historical = kerr_phase(
        shock_intensity(I_nl, Omega, omega0, dt=dt, method="tdiff"), k0, n2, dz)
    assert np.allclose(ours, historical, rtol=1e-6, atol=1e-20)


def test_fR_zero_reduces_to_electronic_kerr():
    dt = 2.5e-15
    I1d = _gaussian_120fs(384, dt)
    I = I1d[:, None, None]
    IR = historical_fr_mixture_response(I, dt=dt, T2=80e-12, T_R=8.4e-12)
    n2 = 7.8e-24
    dn = historical_fr_mixture_index(I, IR, f_R=0.0, n2=n2)
    assert np.allclose(dn, n2 * I)
    assert np.allclose(n2 * 0.0 * IR, 0.0)


def test_normalize_raman_accepts_and_rejects_historical_fr_mixture():
    good = {
        "model": "rot_sinexp",
        "method": "iir",
        "iir_sampling": "legacy_right_hold",
        "f_R": 0.15,
        "T2": 8e-11,
        "T_R": 8.4e-12,
    }
    _normalize_raman(dict(good))
    mixed = dict(good, operator_mode="historical_fr_mixture")
    _normalize_raman(mixed)
    assert mixed["operator_mode"] == "historical_fr_mixture"

    invalid = [
        dict(good, operator_mode="historical_fr_mixture", method="fft"),
        dict(good, operator_mode="historical_fr_mixture", model="exp"),
        dict(good, operator_mode="historical_fr_mixture", iir_sampling="trapezoidal"),
        {k: v for k, v in good.items() if k != "f_R"},
        dict(good, operator_mode="historical_fr_mixture", f_R=1.0),
    ]
    for raw in invalid:
        raw = dict(raw, operator_mode="historical_fr_mixture")
        with pytest.raises(ValueError):
            _normalize_raman(raw)


def _load_prepare_module():
    path = ROOT / "tools" / "prepare_historical_fr_mixture_job.py"
    spec = importlib.util.spec_from_file_location("prepare_historical_fr_mixture_job", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_postprocess_module():
    path = ROOT / "tools" / "postprocess_historical_fr_mixture.py"
    spec = importlib.util.spec_from_file_location("postprocess_historical_fr_mixture", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_postprocess_accepts_isolated_batch_actual_sha_metadata():
    module = _load_postprocess_module()
    assert module.execution_git_sha({"execution_git_sha": "legacy"}) == "legacy"
    assert module.execution_git_sha({"actual_sha": "fbbf189"}) == "fbbf189"
    assert module.execution_git_sha({}) == ""


def test_prepare_tool_single_field_config_diff():
    module = _load_prepare_module()
    base = module.json.loads(module.BASE.read_text(encoding="utf-8"))
    derived, differences = module.build(base)
    assert derived["raman"]["operator_mode"] == "historical_fr_mixture"
    assert differences == [{
        "path": "raman.operator_mode",
        "base": None,
        "historical_fr_mixture": "historical_fr_mixture",
    }]


def _run_small_grid(*, operator_mode, f_R=0.15, use_raman_phase=True):
    Nx = Ny = 24
    Nt = 64
    axes = make_axes(Nx, Ny, Nt, Lx=2.0e-3, Ly=2.0e-3, Twin=400e-15)
    lam0 = 800e-9
    n0 = 1.00027
    omega0 = 2.0 * np.pi * c0 / lam0
    k0 = n0 * omega0 / c0
    dt = float(axes.dt)

    t = np.asarray(axes.t)
    x = np.asarray(axes.x)
    y = np.asarray(axes.y)
    env_t = np.exp(-2.0 * np.log(2.0) * (t / (120e-15 / 2.0)) ** 2)
    w0 = 0.4e-3
    env_x = np.exp(-(x / w0) ** 2)
    env_y = np.exp(-(y / w0) ** 2)
    E = (env_t[:, None, None] * (env_y[None, :, None] * env_x[None, None, :])).astype(np.complex64)

    prop = PropagationConfig(
        linear_model="bk_nee",
        use_self_steepening=True,
        use_electronic_kerr=True,
        use_raman_phase=use_raman_phase,
        use_plasma_phase=True,
        use_ionization_loss=True,
        use_raman_absorption=False,
        use_ionization_solver=False,
        focus_window_step=False,
    )
    raman = RamanConfig(
        enabled=True,
        f_R=f_R,
        model="rot_sinexp",
        T2=80e-12,
        T_R=8.4e-12,
        absorption=False,
        operator_mode=operator_mode,
        omega_R=1.6e13,
        Gamma_R=1.3e13,
        n_R=2.3e-23,
    )
    ion = IonizationConfig(species=None)
    dn_gas = np.zeros((Ny, Nx), dtype=np.float32)
    _, _, diag = propagate_one_pulse(
        E,
        kperp2=axes.kperp2,
        k0=k0,
        omega0=omega0,
        dz=2e-4,
        z_max=6e-4,
        n0=n0,
        n2=7.8e-24,
        Ui=Ui_N2,
        N0=N0_air,
        ion_conf=ion,
        dn_gas=dn_gas,
        dt=dt,
        axes=axes,
        prop_conf=prop,
        raman_conf=raman,
        record_every_z=1,
    )
    return {key: np.asarray(value) for key, value in diag.items() if value is not None}


def test_legacy_split_path_unchanged():
    diag = _run_small_grid(operator_mode="legacy_split")
    n_R = 2.3e-23
    assert np.all(np.isfinite(diag["I_max_z"]))
    assert np.all(np.isfinite(diag["delta_n_rot_max_z"]))
    assert np.allclose(diag["delta_n_rot_max_z"], n_R * diag["IR_max_z"], rtol=1e-4, atol=0.0)
    assert diag["f_R_used_historical_fr_mixture"] == 0.0


def test_mixture_fR_zero_equals_electronic_kerr():
    mixture = _run_small_grid(operator_mode="historical_fr_mixture", f_R=0.0)
    electronic = _run_small_grid(operator_mode="legacy_split", use_raman_phase=False)
    assert np.all(np.isfinite(mixture["I_max_z"]))
    assert np.allclose(mixture["delta_n_rot_applied_max_z"], 0.0, atol=1e-30)
    assert np.allclose(mixture["I_max_z"], electronic["I_max_z"], rtol=1e-4, atol=0.0)
    assert np.allclose(
        mixture["dphi_kerr_max_abs_z"], electronic["dphi_kerr_max_abs_z"], rtol=1e-4, atol=0.0)
    assert mixture["f_R_used_historical_fr_mixture"] == 0.0


def test_mixture_uses_historical_kernel_params_in_diag():
    diag = _run_small_grid(operator_mode="historical_fr_mixture", f_R=0.15)
    assert abs(diag["f_R_used_historical_fr_mixture"] - 0.15) < 1e-6
    assert abs(diag["historical_raman_omega_R_rad_s"] - 2.0 * np.pi / 8.4e-12) / (2.0 * np.pi / 8.4e-12) < 1e-6
    assert abs(diag["historical_raman_Gamma_R_1_s"] - 1.0 / 80e-12) / (1.0 / 80e-12) < 1e-6
