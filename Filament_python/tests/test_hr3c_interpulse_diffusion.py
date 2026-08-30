from __future__ import annotations

import inspect

import numpy as np
import pytest


def _axes(nx: int = 64, length_m: float = 4.0e-3):
    from KHz_filament.grids import make_axes

    return make_axes(nx, nx, 8, length_m, length_m, 80e-15)


def test_c1_c2_zero_and_uniform_states_are_invariant():
    from KHz_filament.config import HeatConfig
    from KHz_filament.slow_diffusion import diffuse_interval_2d

    axes = _axes()
    heat = HeatConfig()
    zero = np.zeros((64, 64), dtype=np.float64)
    np.testing.assert_array_equal(
        diffuse_interval_2d(zero, kperp2=axes.kperp2, D_th=heat.D_th, f_rep=heat.f_rep), zero,
    )

    uniform = np.full((64, 64), -2.0e-4, dtype=np.float64)
    evolved = diffuse_interval_2d(
        uniform, kperp2=axes.kperp2, D_th=heat.D_th, f_rep=heat.f_rep,
        edge_threshold=None,
    )
    np.testing.assert_allclose(evolved, uniform, rtol=0.0, atol=1e-15)


def test_c3_gaussian_broadening_matches_analytical_solution():
    from KHz_filament.config import HeatConfig
    from KHz_filament.slow_diffusion import diffuse_interval_2d, validate_hr3c_parameters

    axes = _axes(nx=256, length_m=8.0e-3)
    heat = HeatConfig(f_rep=1.0e3)
    x, y = np.asarray(axes.x), np.asarray(axes.y)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    radius_squared = xx**2 + yy**2
    width0 = 4.0e-4
    amplitude0 = -2.0e-4
    initial = amplitude0 * np.exp(-radius_squared / width0**2)

    evolved = np.asarray(diffuse_interval_2d(
        initial, kperp2=axes.kperp2, D_th=heat.D_th, f_rep=heat.f_rep,
    ))
    dt = validate_hr3c_parameters(D_th=heat.D_th, f_rep=heat.f_rep)
    width_squared = width0**2 + 4.0 * heat.D_th * dt
    expected = amplitude0 * width0**2 / width_squared * np.exp(-radius_squared / width_squared)
    measured_width_squared = np.sum(radius_squared * (-evolved)) / np.sum(-evolved)

    np.testing.assert_allclose(measured_width_squared, width_squared, rtol=3e-6)
    np.testing.assert_allclose(evolved[128, 128], amplitude0 * width0**2 / width_squared, rtol=3e-6)
    np.testing.assert_allclose(evolved, expected, rtol=4e-6, atol=1e-16)


def test_c4_c5_c8_preserve_negative_channel_integral_and_real_dtype():
    from KHz_filament.config import HeatConfig
    from KHz_filament.slow_diffusion import diffuse_interval_2d

    axes = _axes()
    heat = HeatConfig()
    x, y = np.asarray(axes.x), np.asarray(axes.y)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    state64 = (-2.0e-4 * np.exp(-(xx**2 + yy**2) / (3.0e-4)**2)).astype(np.float64)
    evolved64 = np.asarray(diffuse_interval_2d(
        state64, kperp2=axes.kperp2, D_th=heat.D_th, f_rep=heat.f_rep,
    ))
    assert float(np.max(evolved64)) <= 1e-14 * float(np.max(np.abs(state64)))
    np.testing.assert_allclose(evolved64.sum(), state64.sum(), rtol=1e-12, atol=1e-30)

    state32 = state64.astype(np.float32)
    evolved32 = np.asarray(diffuse_interval_2d(
        state32, kperp2=axes.kperp2, D_th=heat.D_th, f_rep=heat.f_rep,
    ))
    assert evolved32.dtype == np.float32
    assert np.isfinite(evolved32).all()


def test_c6_edge_contamination_gate_fails_closed():
    from KHz_filament.config import HeatConfig
    from KHz_filament.slow_diffusion import (
        DEFAULT_EDGE_CONTAMINATION_THRESHOLD,
        diffuse_interval_2d,
        evaluate_edge_contamination,
    )

    axes = _axes(nx=16, length_m=1.0e-3)
    heat = HeatConfig()
    state = np.full((16, 16), -1.0e-4, dtype=np.float64)
    edge = evaluate_edge_contamination(state)
    assert edge["R_edge"] == pytest.approx(1.0)
    assert edge["R_edge"] > DEFAULT_EDGE_CONTAMINATION_THRESHOLD
    with pytest.raises(ValueError, match="edge-contamination gate failed"):
        diffuse_interval_2d(
            state, kperp2=axes.kperp2, D_th=heat.D_th, f_rep=heat.f_rep,
        )


def test_c7_authoritative_parameters_do_not_fallback_to_legacy_d_gas():
    from KHz_filament.config import HeatConfig
    from KHz_filament.config_normalize import normalize_config
    from KHz_filament.config_schema import HEAT_HR3C_FIELDS
    from KHz_filament.slow_diffusion import build_diffusion_kernel, validate_hr3c_parameters

    heat = HeatConfig()
    assert heat.D_th == pytest.approx(21.7e-6)
    assert HEAT_HR3C_FIELDS["D_th"] == "authoritative HR-3C transverse thermal diffusivity [m^2/s]"
    assert "D_gas" not in inspect.signature(validate_hr3c_parameters).parameters
    assert validate_hr3c_parameters(D_th=heat.D_th, f_rep=heat.f_rep) == pytest.approx(1.0 / heat.f_rep)
    with pytest.raises(ValueError, match="positive"):
        validate_hr3c_parameters(D_th=0.0, f_rep=heat.f_rep)
    with pytest.raises(ValueError, match="finite"):
        validate_hr3c_parameters(D_th=np.nan, f_rep=heat.f_rep)

    normalized = normalize_config({"heat": {"D_th": "2.17e-5", "f_rep": "1000"}})
    assert normalized["heat"]["D_th"] == pytest.approx(21.7e-6)
    with pytest.raises(ValueError, match="D_th"):
        normalize_config({"heat": {"D_th": 0.0}})
    with pytest.raises(ValueError, match="f_rep"):
        normalize_config({"heat": {"f_rep": 0.0}})

    axes = _axes()
    kernel = np.asarray(build_diffusion_kernel(axes.kperp2, D_th=heat.D_th, f_rep=heat.f_rep))
    assert np.isfinite(kernel).all()
    assert np.all(kernel > 0.0)
    assert np.all(kernel <= 1.0)
    assert kernel[0, 0] == pytest.approx(1.0)
