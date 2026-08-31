from __future__ import annotations

import numpy as np
import pytest


def _state():
    from KHz_filament.hr4 import create_hr4_slow_state
    return create_hr4_slow_state(n_intervals=2, shape=(4, 5), dtype=np.float32)


def _stable_kwargs():
    return {
        "dx": 10.0e-6, "dy": 10.0e-6, "dt_hydro": 1.0e-6,
        "chi": 21.7e-6, "nu": 1.5e-5, "max_abs_vx": 1.0e-3, "max_abs_vy": 2.0e-3,
    }


def test_state_contract_is_three_field_zero_initial_and_metadata_is_explicit():
    state = _state()
    assert state.delta_n.shape == state.vx.shape == state.vy.shape == (2, 4, 5)
    assert state.delta_n.dtype == state.vx.dtype == state.vy.dtype == np.float32
    np.testing.assert_array_equal(state.vx, 0.0)
    np.testing.assert_array_equal(state.vy, 0.0)
    metadata = state.metadata()
    assert metadata["hr4_state_authoritative_fields"] == ("delta_n", "vx", "vy")
    assert metadata["hr4_state_stage"] == "pre_pulse"
    assert metadata["hr4_state_geometry_m"]["y_min"] < metadata["hr4_state_geometry_m"]["y_max"]


@pytest.mark.parametrize("delta_n, vx, vy", [
    (np.zeros((2, 4, 5), np.float32), np.zeros((2, 4, 4), np.float32), np.zeros((2, 4, 5), np.float32)),
    (np.zeros((2, 4, 5), np.float32), np.zeros((2, 4, 5), np.float64), np.zeros((2, 4, 5), np.float32)),
    (np.zeros((2, 4, 5), np.int32), np.zeros((2, 4, 5), np.int32), np.zeros((2, 4, 5), np.int32)),
    (np.full((2, 4, 5), np.nan, np.float32), np.zeros((2, 4, 5), np.float32), np.zeros((2, 4, 5), np.float32)),
])
def test_state_contract_rejects_invalid_shape_or_dtype(delta_n, vx, vy):
    from KHz_filament.hr4 import HR4SlowState
    with pytest.raises(ValueError):
        HR4SlowState(delta_n=delta_n, vx=vx, vy=vy)


def test_derived_thermodynamic_diagnostics_have_frozen_signs():
    from KHz_filament.hr4 import delta_T_from_delta_n, delta_rho_from_delta_n
    delta_n = np.full((1, 2, 3), -2.7e-5, dtype=np.float64)
    density = delta_rho_from_delta_n(delta_n, n0=1.00027)
    temperature = delta_T_from_delta_n(delta_n, n0=1.00027)
    np.testing.assert_allclose(density, delta_n / 0.00027)
    np.testing.assert_allclose(temperature, -delta_n / 0.00027)
    assert np.all(density < 0.0) and np.all(temperature > 0.0)


def test_pulse_post_updates_only_delta_n_and_preserves_velocity_exactly():
    from KHz_filament.hr4 import apply_hr4_pulse_post
    pre = _state()
    pre.vx[0, 1, 2], pre.vy[1, 3, 4] = np.float32(0.125), np.float32(-0.375)
    increment = np.full(pre.delta_n.shape, -1.0e-6, dtype=np.float32)
    post = apply_hr4_pulse_post(pre, increment)
    np.testing.assert_array_equal(post.delta_n, increment)
    np.testing.assert_array_equal(post.vx, pre.vx)
    np.testing.assert_array_equal(post.vy, pre.vy)
    assert post.stage == "post_pulse"
    with pytest.raises(ValueError, match="pre_pulse"):
        apply_hr4_pulse_post(post, increment)


def test_geometry_gravity_and_parameter_authority_are_frozen_and_fail_closed():
    from KHz_filament.config import HeatConfig
    from KHz_filament.config_normalize import normalize_config
    from KHz_filament.config_schema import HEAT_HR4_FIELDS
    from KHz_filament.hr4 import HR4_CHI, HR4_GRAVITY_X, HR4_GRAVITY_Y, HR4_NU, validate_hr4_parameters
    heat = HeatConfig()
    assert heat.hr4_enabled is False and heat.chi == heat.D_th == HR4_CHI
    assert heat.nu == HR4_NU and heat.gravity_x == HR4_GRAVITY_X and heat.gravity_y == HR4_GRAVITY_Y
    assert heat.y_min < heat.y_max and heat.x_min < heat.x_max
    assert HEAT_HR4_FIELDS["chi"].startswith("must equal authoritative HR-3C")
    assert normalize_config({"heat": {"hr4_enabled": True, "hr3b_enabled": True}})["heat"]["hr4_enabled"]
    with pytest.raises(ValueError, match="requires"):
        normalize_config({"heat": {"hr4_enabled": True}})
    with pytest.raises(ValueError, match="chi"):
        normalize_config({"heat": {"chi": 2.0e-5}})
    with pytest.raises(ValueError, match="gravity"):
        normalize_config({"heat": {"gravity_y": -9.8}})
    with pytest.raises(ValueError, match="geometry"):
        normalize_config({"heat": {"x_min": -1.4e-3}})
    with pytest.raises(ValueError, match="advection_scheme"):
        validate_hr4_parameters(
            chi=HR4_CHI, D_th=HR4_CHI, nu=HR4_NU, gravity_x=0.0, gravity_y=-9.81,
            x_min=-1.5e-3, x_max=1.5e-3, y_min=-1.0e-3, y_max=2.5e-3,
            dx=1e-5, dy=1e-5, dt_hydro=1e-6, advection_scheme="weno",
            diffusion_scheme="explicit_central_fd", time_integrator="explicit_euler",
            grid_layout="collocated", boundary_delta_n="ambient_dirichlet_zero",
            boundary_velocity="open_zero_gradient_outflow_ambient_inflow",
        )


def test_boundary_helper_applies_ambient_outflow_inflow_and_deterministic_corners():
    from KHz_filament.hr4 import apply_hr4_open_boundaries
    delta_n = np.full((4, 5), -1.0, dtype=np.float64)
    vx, vy = np.zeros((4, 5), dtype=np.float64), np.zeros((4, 5), dtype=np.float64)
    vx[:, 0], vx[:, -1], vy[0, :], vy[-1, :] = -2.0, 2.0, -3.0, 3.0
    vx[1, 1], vy[1, 1] = 4.0, 5.0
    vx[1, 3], vy[1, 3] = 6.0, 7.0
    vx[2, 1], vy[2, 1] = 8.0, 9.0
    vx[2, 3], vy[2, 3] = 10.0, 11.0
    dn_out, vx_out, vy_out = apply_hr4_open_boundaries(delta_n, vx, vy)
    assert np.all(dn_out[0, :] == 0.0) and np.all(dn_out[:, -1] == 0.0)
    assert (vx_out[0, 0], vy_out[0, 0]) == (4.0, 5.0)
    assert (vx_out[0, -1], vy_out[0, -1]) == (6.0, 7.0)
    assert (vx_out[-1, 0], vy_out[-1, 0]) == (8.0, 9.0)
    assert (vx_out[-1, -1], vy_out[-1, -1]) == (10.0, 11.0)
    assert (vx_out[1, 0], vy_out[1, 0]) == (vx[1, 1], vy[1, 1])
    assert (vx_out[-1, 2], vy_out[-1, 2]) == (vx[-2, 2], vy[-2, 2])
    vx[:, 0] = 1.0
    _, vx_inflow, vy_inflow = apply_hr4_open_boundaries(delta_n, vx, vy)
    assert np.all(vx_inflow[1:-1, 0] == 0.0) and np.all(vy_inflow[1:-1, 0] == 0.0)
    assert (vx_inflow[0, 0], vy_inflow[0, 0]) == (0.0, 0.0)


def test_stability_audit_reports_independent_diffusion_and_cfl_failures():
    from KHz_filament.hr4 import audit_hr4_stability, require_hr4_stability
    stable = audit_hr4_stability(**_stable_kwargs())
    assert stable["overall_pass"] and stable["passed_diffusion"] and stable["passed_advection"]
    assert 0.0 < stable["diffusion_number_nu"] < stable["diffusion_number_chi"] <= 0.5
    diffusion_bad = audit_hr4_stability(**{**_stable_kwargs(), "dt_hydro": 2.0e-6})
    assert not diffusion_bad["passed_diffusion"] and diffusion_bad["passed_advection"]
    nu_bad = audit_hr4_stability(**{**_stable_kwargs(), "nu": 2.6e-5})
    assert nu_bad["diffusion_number_chi"] <= 0.5 < nu_bad["diffusion_number_nu"]
    assert not nu_bad["passed_diffusion"]
    cfl_bad = audit_hr4_stability(**{**_stable_kwargs(), "max_abs_vx": 20.0})
    assert cfl_bad["passed_diffusion"] and not cfl_bad["passed_advection"]
    with pytest.raises(ValueError, match="stability audit failed"):
        require_hr4_stability(**{**_stable_kwargs(), "max_abs_vx": 20.0})
    for invalid in ("dx", "dy", "dt_hydro"):
        with pytest.raises(ValueError, match="positive"):
            audit_hr4_stability(**{**_stable_kwargs(), invalid: 0.0})
