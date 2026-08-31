from __future__ import annotations

import numpy as np
import pytest


N0 = 1.00027
DX = DY = 10.0e-6
DT = 1.0e-6
CHI = 21.7e-6
NU = 1.5e-5


def _grid(n: int, *, x_min: float | None = None, y_min: float | None = None):
    x0 = -0.5 * (n - 1) * DX if x_min is None else x_min
    y0 = -0.5 * (n - 1) * DY if y_min is None else y_min
    x = x0 + np.arange(n) * DX
    y = y0 + np.arange(n) * DY
    return np.meshgrid(x, y, indexing="xy"), x0, y0


def _gaussian(n: int, sigma: float, amplitude: float, *, x_center: float = 0.0, y_center: float = 0.0):
    (xx, yy), x_min, y_min = _grid(n)
    return amplitude * np.exp(-((xx - x_center) ** 2 + (yy - y_center) ** 2) / (2.0 * sigma**2)), x_min, y_min


def _advance(delta_n, vx, vy, **overrides):
    from KHz_filament.hr4 import advance_hr4_single_screen

    kwargs = {
        "dx": DX, "dy": DY, "dt_hydro": DT, "chi": CHI, "nu": NU, "n0": N0,
        "gravity_x": 0.0, "gravity_y": -9.81,
    }
    kwargs.update(overrides)
    return advance_hr4_single_screen(delta_n, vx, vy, **kwargs)


def test_b1_zero_state_is_invariant_for_one_and_many_steps():
    zero = np.zeros((9, 9), dtype=np.float64)
    one = _advance(zero, zero, zero)
    many = _advance(zero, zero, zero, n_steps=11)
    for result in (one, many):
        np.testing.assert_array_equal(result["delta_n"], zero)
        np.testing.assert_array_equal(result["vx"], zero)
        np.testing.assert_array_equal(result["vy"], zero)
        assert result["observables"]["max_abs_v"] == 0.0
        assert result["performance"]["slow_time_history_stored"] is False


def test_b2_negative_index_channel_has_upward_buoyancy_and_no_x_kick():
    from KHz_filament.hr4 import compute_hr4_rhs

    delta_n = np.zeros((9, 9), dtype=np.float64)
    delta_n[4, 4] = -1.0e-4
    zero = np.zeros_like(delta_n)
    result = _advance(delta_n, zero, zero, chi=0.0, nu=0.0)
    assert result["vy"][4, 4] > 0.0
    np.testing.assert_array_equal(result["vx"], zero)
    with pytest.raises(ValueError, match="gravity_x"):
        compute_hr4_rhs(
            delta_n, zero, zero, dx=DX, dy=DY, chi=0.0, nu=0.0, n0=N0,
            gravity_x=1.0, gravity_y=-9.81,
        )


@pytest.mark.parametrize(
    "vx_value, vy_value, expected",
    [
        (2.0, 0.0, 2.0 * (10.0 - 4.0) / DX),
        (-2.0, 0.0, -2.0 * (20.0 - 10.0) / DX),
        (0.0, 3.0, 3.0 * (10.0 - 1.0) / DY),
        (0.0, -3.0, -3.0 * (30.0 - 10.0) / DY),
    ],
)
def test_b3_upwind_stencil_selects_the_correct_local_direction(vx_value, vy_value, expected):
    from KHz_filament.hr4 import upwind_advection

    q = np.zeros((5, 5), dtype=np.float64)
    q[2, 2], q[2, 1], q[2, 3], q[1, 2], q[3, 2] = 10.0, 4.0, 20.0, 1.0, 30.0
    vx = np.zeros_like(q)
    vy = np.zeros_like(q)
    vx[2, 2], vy[2, 2] = vx_value, vy_value
    advection = upwind_advection(q, vx, vy, dx=DX, dy=DY)
    assert advection[2, 2] == pytest.approx(expected)


def test_b4_laplacian_is_second_order_central_for_scalar_and_velocity_fields():
    from KHz_filament.hr4 import laplacian_fd

    jj, ii = np.indices((9, 9))
    scalar = (ii * DX) ** 2 + 3.0 * (jj * DY) ** 2
    velocity = 2.0 * (ii * DX) ** 2 - 4.0 * (jj * DY) ** 2
    scalar_lap = laplacian_fd(scalar, dx=DX, dy=DY)
    velocity_lap = laplacian_fd(velocity, dx=DX, dy=DY)
    np.testing.assert_allclose(scalar_lap[1:-1, 1:-1], 8.0, atol=1e-12)
    np.testing.assert_allclose(velocity_lap[1:-1, 1:-1], -4.0, atol=1e-12)


def test_b5_pure_gaussian_thermal_diffusion_broadens_against_analytic_solution():
    sigma0, amplitude, steps = 100.0e-6, -1.0e-4, 50
    delta_n, x_min, y_min = _gaussian(81, sigma0, amplitude)
    zero = np.zeros_like(delta_n)
    result = _advance(
        delta_n, zero, zero, n_steps=steps, nu=0.0, gravity_y=0.0, x_min=x_min, y_min=y_min,
    )
    elapsed = steps * DT
    sigma = np.sqrt(sigma0**2 + 2.0 * CHI * elapsed)
    analytic_center = amplitude * sigma0**2 / sigma**2
    assert result["observables"]["thermal_channel_width_m"] > sigma0
    assert result["observables"]["thermal_channel_width_m"] == pytest.approx(sigma, rel=0.025)
    assert result["delta_n"][40, 40] == pytest.approx(analytic_center, rel=0.025)


def test_b6_constant_velocity_advection_moves_a_blob_at_the_correct_centroid_speed():
    sigma, speed, steps = 60.0e-6, 5.0e-2, 200
    delta_n, x_min, y_min = _gaussian(81, sigma, -1.0e-4)
    vx = np.full_like(delta_n, speed)
    vy = np.zeros_like(delta_n)
    result = _advance(
        delta_n, vx, vy, chi=0.0, nu=0.0, gravity_y=0.0, n_steps=steps, x_min=x_min, y_min=y_min,
    )
    initial_x = 0.0
    assert result["observables"]["thermal_channel_centroid_x_m"] - initial_x == pytest.approx(
        speed * steps * DT, abs=2.5e-6
    )


def test_b7_viscous_velocity_diffusion_broadens_and_reduces_gaussian_peak():
    sigma0, amplitude, steps = 100.0e-6, 0.1, 50
    vx, _, _ = _gaussian(81, sigma0, amplitude)
    zero = np.zeros_like(vx)
    result = _advance(zero, vx, zero, chi=0.0, gravity_y=0.0, n_steps=steps)
    sigma = np.sqrt(sigma0**2 + 2.0 * NU * steps * DT)
    expected_peak = amplitude * sigma0**2 / sigma**2
    assert result["vx"][40, 40] < amplitude
    assert result["vx"][40, 40] == pytest.approx(expected_peak, rel=0.03)


def test_b8_unsplit_euler_does_not_advect_delta_n_with_new_buoyant_velocity():
    from KHz_filament.hr4 import apply_hr4_boundaries

    delta_n, x_min, y_min = _gaussian(41, 50.0e-6, -1.0e-3)
    zero = np.zeros_like(delta_n)
    bounded_delta_n, _, _ = apply_hr4_boundaries(delta_n, zero, zero)
    first = _advance(
        delta_n, zero, zero, chi=0.0, nu=0.0, dt_hydro=100.0e-6,
        x_min=x_min, y_min=y_min,
    )
    np.testing.assert_allclose(first["delta_n"], bounded_delta_n, atol=1e-18, rtol=0.0)
    assert first["observables"]["max_abs_vy"] > 0.0
    second = _advance(
        first["delta_n"], first["vx"], first["vy"], chi=0.0, nu=0.0, dt_hydro=100.0e-6,
        x_min=x_min, y_min=y_min,
    )
    assert second["observables"]["thermal_channel_centroid_y_m"] > first["observables"]["thermal_channel_centroid_y_m"]


@pytest.mark.parametrize(
    "face, interior_velocity, expected_inflow",
    [
        ("left", 1.0, True), ("left", -1.0, False),
        ("right", -1.0, True), ("right", 1.0, False),
        ("bottom", 1.0, True), ("bottom", -1.0, False),
        ("top", -1.0, True), ("top", 1.0, False),
    ],
)
def test_b9_face_boundary_contract_uses_adjacent_interior_normal_velocity(face, interior_velocity, expected_inflow):
    from KHz_filament.hr4 import apply_hr4_boundaries

    delta_n = np.full((7, 7), -1.0, dtype=np.float64)
    vx, vy = np.zeros_like(delta_n), np.zeros_like(delta_n)
    if face == "left":
        vx[1:-1, 1] = interior_velocity
        boundary = (slice(1, -1), 0)
        interior = (slice(1, -1), 1)
    elif face == "right":
        vx[1:-1, -2] = interior_velocity
        boundary = (slice(1, -1), -1)
        interior = (slice(1, -1), -2)
    elif face == "bottom":
        vy[1, 1:-1] = interior_velocity
        boundary = (0, slice(1, -1))
        interior = (1, slice(1, -1))
    else:
        vy[-2, 1:-1] = interior_velocity
        boundary = (-1, slice(1, -1))
        interior = (-2, slice(1, -1))
    _, bounded_vx, bounded_vy = apply_hr4_boundaries(delta_n, vx, vy)
    if expected_inflow:
        np.testing.assert_array_equal(bounded_vx[boundary], 0.0)
        np.testing.assert_array_equal(bounded_vy[boundary], 0.0)
    else:
        np.testing.assert_array_equal(bounded_vx[boundary], vx[interior])
        np.testing.assert_array_equal(bounded_vy[boundary], vy[interior])


@pytest.mark.parametrize(
    "corner, vx_value, vy_value, expected_zero",
    [
        ((0, 0), 1.0, 1.0, True), ((0, -1), -1.0, 1.0, True),
        ((-1, 0), 1.0, -1.0, True), ((-1, -1), -1.0, -1.0, True),
        ((0, 0), -1.0, -1.0, False), ((0, -1), 1.0, -1.0, False),
        ((-1, 0), -1.0, 1.0, False), ((-1, -1), 1.0, 1.0, False),
    ],
)
def test_b9_corner_boundary_contract_is_deterministic(corner, vx_value, vy_value, expected_zero):
    from KHz_filament.hr4 import apply_hr4_boundaries

    delta_n = np.full((7, 7), -1.0, dtype=np.float64)
    vx, vy = np.zeros_like(delta_n), np.zeros_like(delta_n)
    row = 1 if corner[0] == 0 else -2
    col = 1 if corner[1] == 0 else -2
    vx[row, col], vy[row, col] = vx_value, vy_value
    _, bounded_vx, bounded_vy = apply_hr4_boundaries(delta_n, vx, vy)
    if expected_zero:
        assert (bounded_vx[corner], bounded_vy[corner]) == (0.0, 0.0)
    else:
        assert (bounded_vx[corner], bounded_vy[corner]) == (vx_value, vy_value)


def test_b10_topward_blob_does_not_periodically_wrap_to_the_bottom():
    n, sigma, speed, steps = 61, 20.0e-6, 0.1, 150
    x_min = y_min = -0.5 * (n - 1) * DX
    delta_n, _, _ = _gaussian(n, sigma, -1.0e-4, y_center=100.0e-6)
    vx, vy = np.zeros_like(delta_n), np.full_like(delta_n, speed)
    result = _advance(
        delta_n, vx, vy, chi=0.0, nu=0.0, gravity_y=0.0,
        dt_hydro=5.0e-6, n_steps=steps, x_min=x_min, y_min=y_min,
    )
    assert result["observables"]["thermal_channel_centroid_y_m"] > 100.0e-6
    assert np.max(np.abs(result["delta_n"][1:3, :])) < 1.0e-15


def test_b11_coupled_negative_channel_rises_and_broadens_with_bounded_memory():
    sigma, amplitude, steps = 80.0e-6, -1.0e-3, 400
    delta_n, x_min, y_min = _gaussian(81, sigma, amplitude, y_center=-60.0e-6)
    zero = np.zeros_like(delta_n)
    initial = _advance(delta_n, zero, zero, n_steps=1, x_min=x_min, y_min=y_min)
    result = _advance(delta_n, zero, zero, n_steps=steps, x_min=x_min, y_min=y_min)
    assert result["observables"]["thermal_channel_centroid_y_m"] > initial["observables"]["thermal_channel_centroid_y_m"]
    assert result["observables"]["thermal_channel_width_m"] > initial["observables"]["thermal_channel_width_m"]
    assert result["observables"]["max_abs_vy"] > 0.0
    assert result["observables"]["max_abs_v"] >= result["observables"]["max_abs_vy"]
    assert result["performance"]["temporary_working_set_estimate_bytes"] > 0


def test_b12_development_timestep_short_comparison_is_stable_and_trend_consistent():
    from KHz_filament.hr4 import audit_hr4_stability

    sigma, amplitude, duration = 80.0e-6, -1.0e-3, 200.0e-6
    delta_n, x_min, y_min = _gaussian(65, sigma, amplitude, y_center=-40.0e-6)
    zero = np.zeros_like(delta_n)
    one_us = _advance(delta_n, zero, zero, n_steps=int(duration / DT), x_min=x_min, y_min=y_min)
    half_us = _advance(
        delta_n, zero, zero, dt_hydro=0.5e-6, n_steps=int(duration / 0.5e-6),
        x_min=x_min, y_min=y_min,
    )
    for key in ("thermal_channel_centroid_y_m", "thermal_channel_width_m", "max_abs_vy", "min_delta_n"):
        assert np.isfinite(one_us["observables"][key])
        assert np.isfinite(half_us["observables"][key])
    assert one_us["observables"]["thermal_channel_centroid_y_m"] > y_min
    assert one_us["observables"]["thermal_channel_width_m"] > sigma
    assert one_us["observables"]["thermal_channel_centroid_y_m"] == pytest.approx(
        half_us["observables"]["thermal_channel_centroid_y_m"], abs=4.0e-6
    )
    assert one_us["observables"]["thermal_channel_width_m"] == pytest.approx(
        half_us["observables"]["thermal_channel_width_m"], rel=0.08
    )
    assert one_us["observables"]["max_abs_vy"] == pytest.approx(
        half_us["observables"]["max_abs_vy"], rel=0.12
    )
    audit = audit_hr4_stability(
        dx=DX, dy=DY, dt_hydro=DT, chi=CHI, nu=NU,
        max_abs_vx=one_us["observables"]["max_abs_vx"],
        max_abs_vy=one_us["observables"]["max_abs_vy"],
    )
    assert audit["passed_diffusion_chi"] and audit["passed_diffusion_nu"]
    assert audit["passed_combined_chi"] and audit["passed_combined_nu"]
