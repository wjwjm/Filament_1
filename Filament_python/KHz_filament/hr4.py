"""HR-4A contracts and HR-4B single-screen isobaric flow operator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping

import numpy as np

from .device import debug_backend, to_cpu, xp

HR4_CHI = 21.7e-6
HR4_NU = 1.5e-5
HR4_GRAVITY_X = 0.0
HR4_GRAVITY_Y = -9.81
HR4_X_MIN, HR4_X_MAX = -1.5e-3, 1.5e-3
HR4_Y_MIN, HR4_Y_MAX = -1.0e-3, 2.5e-3
HR4_DX = HR4_DY = 10.0e-6
HR4_DT_HYDRO = 1.0e-6
HR4_CFL_LIMIT = 1.0  # provisional development threshold; not production-frozen

HR4_STATE_SCHEMA = "khz_filament.hr4a.slow_state.v1"
HR4_ALLOWED_ADVECTION = "first_order_upwind"
HR4_ALLOWED_DIFFUSION = "explicit_central_fd"
HR4_ALLOWED_INTEGRATOR = "explicit_euler"
HR4_ALLOWED_GRID_LAYOUT = "collocated"
HR4_ALLOWED_DELTA_N_BOUNDARY = "ambient_dirichlet_zero"
HR4_ALLOWED_VELOCITY_BOUNDARY = "open_zero_gradient_outflow_ambient_inflow"

HR4_DEFAULTS: Mapping[str, object] = {
    "hr4_enabled": False, "chi": HR4_CHI, "nu": HR4_NU,
    "gravity_x": HR4_GRAVITY_X, "gravity_y": HR4_GRAVITY_Y,
    "x_min": HR4_X_MIN, "x_max": HR4_X_MAX,
    "y_min": HR4_Y_MIN, "y_max": HR4_Y_MAX,
    "dx": HR4_DX, "dy": HR4_DY, "dt_hydro": HR4_DT_HYDRO,
    "advection_scheme": HR4_ALLOWED_ADVECTION,
    "diffusion_scheme": HR4_ALLOWED_DIFFUSION,
    "time_integrator": HR4_ALLOWED_INTEGRATOR,
    "grid_layout": HR4_ALLOWED_GRID_LAYOUT,
    "boundary_delta_n": HR4_ALLOWED_DELTA_N_BOUNDARY,
    "boundary_velocity": HR4_ALLOWED_VELOCITY_BOUNDARY,
}
HR4_CONFIG_FIELDS = frozenset(HR4_DEFAULTS)


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"HR-4 {name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"HR-4 {name} must be finite")
    return result


def validate_hr4_parameters(
    *, chi: float, D_th: float, nu: float, gravity_x: float, gravity_y: float,
    x_min: float, x_max: float, y_min: float, y_max: float, dx: float, dy: float,
    dt_hydro: float, advection_scheme: str, diffusion_scheme: str,
    time_integrator: str, grid_layout: str, boundary_delta_n: str,
    boundary_velocity: str,
) -> dict[str, object]:
    """Validate frozen HR-4A parameters; this function never evolves a state."""
    values = {
        name: _finite_real(value, name)
        for name, value in {
            "chi": chi, "D_th": D_th, "nu": nu,
            "gravity_x": gravity_x, "gravity_y": gravity_y,
            "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
            "dx": dx, "dy": dy, "dt_hydro": dt_hydro,
        }.items()
    }
    if values["chi"] != values["D_th"]:
        raise ValueError("HR-4 chi must exactly reuse authoritative HR-3C D_th")
    if values["chi"] != HR4_CHI:
        raise ValueError("HR-4 chi must equal frozen STP 21.7e-6 m^2/s")
    if values["nu"] != HR4_NU:
        raise ValueError("HR-4 nu must equal frozen STP 1.5e-5 m^2/s")
    if values["gravity_x"] != HR4_GRAVITY_X or values["gravity_y"] != HR4_GRAVITY_Y:
        raise ValueError("HR-4 gravity must be exactly (0, -9.81) m/s^2")
    if (values["x_min"], values["x_max"], values["y_min"], values["y_max"]) != (
        HR4_X_MIN, HR4_X_MAX, HR4_Y_MIN, HR4_Y_MAX,
    ):
        raise ValueError("HR-4 geometry must use the frozen transverse domain")
    if min(values["dx"], values["dy"], values["dt_hydro"]) <= 0.0:
        raise ValueError("HR-4 dx, dy, and dt_hydro must be positive")
    choices = {
        "advection_scheme": (str(advection_scheme), HR4_ALLOWED_ADVECTION),
        "diffusion_scheme": (str(diffusion_scheme), HR4_ALLOWED_DIFFUSION),
        "time_integrator": (str(time_integrator), HR4_ALLOWED_INTEGRATOR),
        "grid_layout": (str(grid_layout), HR4_ALLOWED_GRID_LAYOUT),
        "boundary_delta_n": (str(boundary_delta_n), HR4_ALLOWED_DELTA_N_BOUNDARY),
        "boundary_velocity": (str(boundary_velocity), HR4_ALLOWED_VELOCITY_BOUNDARY),
    }
    for name, (actual, expected) in choices.items():
        if actual != expected:
            raise ValueError(f"HR-4 {name} must be {expected!r}")
    return {**values, **{name: actual for name, (actual, _) in choices.items()}}


def normalize_hr4_config_values(heat: MutableMapping[str, Any]) -> None:
    """Fail closed on supplied HR-4A fields without adding runner integration."""
    if not any(name in heat for name in HR4_CONFIG_FIELDS):
        return
    if "hr4_enabled" in heat and not isinstance(heat["hr4_enabled"], bool):
        raise ValueError("heat.hr4_enabled must be true or false.")
    values = {name: heat.get(name, default) for name, default in HR4_DEFAULTS.items()}
    validation_values = {name: value for name, value in values.items() if name != "hr4_enabled"}
    validation_values["D_th"] = heat.get("D_th", HR4_CHI)
    normalized = validate_hr4_parameters(**validation_values)
    for name in HR4_CONFIG_FIELDS - {"hr4_enabled"}:
        if name in heat:
            heat[name] = normalized[name]
    if bool(values["hr4_enabled"]) and not bool(heat.get("hr3b_enabled", False)):
        raise ValueError("heat.hr4_enabled requires heat.hr3b_enabled=true.")


@dataclass(frozen=True)
class HR4Geometry:
    """Metadata for one collocated transverse screen."""

    x_min: float = HR4_X_MIN
    x_max: float = HR4_X_MAX
    y_min: float = HR4_Y_MIN
    y_max: float = HR4_Y_MAX
    dx: float = HR4_DX
    dy: float = HR4_DY

    def __post_init__(self) -> None:
        values = {name: _finite_real(value, name) for name, value in vars(self).items()}
        if values["x_min"] >= values["x_max"] or values["y_min"] >= values["y_max"]:
            raise ValueError("HR-4 geometry bounds must increase")
        if values["dx"] <= 0.0 or values["dy"] <= 0.0:
            raise ValueError("HR-4 geometry dx and dy must be positive")


def _finite_float_volume(value: Any, name: str) -> np.ndarray:
    array = xp.asarray(value)
    if array.ndim != 3 or np.dtype(array.dtype).kind != "f":
        raise ValueError(f"HR-4 {name} must be a real floating [K, Ny, Nx] array")
    if min(array.shape) <= 0 or not bool(xp.all(xp.isfinite(array))):
        raise ValueError(f"HR-4 {name} must have positive finite dimensions and values")
    return array


@dataclass
class HR4SlowState:
    """Authoritative three-field state interface; it owns no disk lifecycle."""

    delta_n: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    geometry: HR4Geometry = field(default_factory=HR4Geometry)
    stage: str = "pre_pulse"

    def __post_init__(self) -> None:
        self.delta_n = _finite_float_volume(self.delta_n, "delta_n")
        self.vx = _finite_float_volume(self.vx, "vx")
        self.vy = _finite_float_volume(self.vy, "vy")
        if self.delta_n.shape != self.vx.shape or self.delta_n.shape != self.vy.shape:
            raise ValueError("HR-4 delta_n, vx, and vy must have identical [K, Ny, Nx] shapes")
        if self.delta_n.dtype != self.vx.dtype or self.delta_n.dtype != self.vy.dtype:
            raise ValueError("HR-4 delta_n, vx, and vy must have identical dtypes")
        if self.stage not in ("pre_pulse", "post_pulse"):
            raise ValueError("HR-4 stage must be 'pre_pulse' or 'post_pulse'")

    def metadata(self) -> dict[str, object]:
        return {
            "hr4_state_schema": HR4_STATE_SCHEMA,
            "hr4_state_authoritative_fields": ("delta_n", "vx", "vy"),
            "hr4_state_shape": tuple(self.delta_n.shape),
            "hr4_state_dtype": self.delta_n.dtype.name,
            "hr4_state_stage": self.stage,
            "hr4_state_geometry_m": vars(self.geometry).copy(),
            "hr4_state_units": {"delta_n": "1", "vx": "m/s", "vy": "m/s"},
        }


def create_hr4_slow_state(
    *, n_intervals: int, shape: tuple[int, int], dtype: Any = np.float32,
    geometry: HR4Geometry | None = None,
) -> HR4SlowState:
    """Create an exact-zero initial state; callers explicitly control its size."""
    intervals, slice_shape, result_dtype = int(n_intervals), tuple(map(int, shape)), np.dtype(dtype)
    if intervals <= 0 or len(slice_shape) != 2 or min(slice_shape) <= 0:
        raise ValueError("HR-4 state requires positive K and [Ny, Nx]")
    if result_dtype.kind != "f":
        raise ValueError("HR-4 state dtype must be real floating point")
    volume_shape = (intervals, *slice_shape)
    return HR4SlowState(
        xp.zeros(volume_shape, result_dtype), xp.zeros(volume_shape, result_dtype),
        xp.zeros(volume_shape, result_dtype), geometry or HR4Geometry(),
    )


def _increment_for_state(value: Any, state: HR4SlowState) -> np.ndarray:
    increment = _finite_float_volume(value, "delta_n_hr3b")
    if increment.shape != state.delta_n.shape or increment.dtype != state.delta_n.dtype:
        raise ValueError("HR-4 delta_n_hr3b must match authoritative state shape and dtype")
    return increment


def apply_hr4_pulse_post(pre: HR4SlowState, delta_n_hr3b: Any) -> HR4SlowState:
    """Apply only the frozen POST map: index increment, no velocity kick."""
    if pre.stage != "pre_pulse":
        raise ValueError("HR-4 pulse POST helper requires pre_pulse state")
    increment = _increment_for_state(delta_n_hr3b, pre)
    return HR4SlowState(pre.delta_n + increment, pre.vx.copy(), pre.vy.copy(), pre.geometry, "post_pulse")


def delta_rho_from_delta_n(delta_n: Any, *, n0: float) -> np.ndarray:
    """Derived relative density perturbation; never authoritative storage."""
    field_ = _finite_float_volume(delta_n, "delta_n")
    n0_value = _finite_real(n0, "n0")
    if n0_value <= 1.0:
        raise ValueError("HR-4 n0 must be greater than one")
    return field_ / (n0_value - 1.0)


def delta_T_from_delta_n(delta_n: Any, *, n0: float) -> np.ndarray:
    """Derived relative temperature perturbation; never authoritative storage."""
    return -delta_rho_from_delta_n(delta_n, n0=n0)


def _finite_screen(value: Any, name: str) -> np.ndarray:
    array = xp.asarray(value)
    if array.ndim != 2 or np.dtype(array.dtype).kind != "f" or min(array.shape) < 3:
        raise ValueError(f"HR-4 {name} must be real floating [Ny, Nx] with Ny, Nx >= 3")
    if not bool(xp.all(xp.isfinite(array))):
        raise ValueError(f"HR-4 {name} must be finite")
    return array


def _validated_screen_triplet(delta_n: Any, vx: Any, vy: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index, x_velocity, y_velocity = (
        _finite_screen(delta_n, "delta_n"), _finite_screen(vx, "vx"), _finite_screen(vy, "vy")
    )
    if index.shape != x_velocity.shape or index.shape != y_velocity.shape:
        raise ValueError("HR-4 boundary fields must have identical shapes")
    if index.dtype != x_velocity.dtype or index.dtype != y_velocity.dtype:
        raise ValueError("HR-4 boundary fields must have identical dtypes")
    return index, x_velocity, y_velocity


def apply_hr4_boundaries(delta_n: Any, vx: Any, vy: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply ambient-index and local interior-normal inflow/outflow boundaries."""
    index, x_velocity, y_velocity = _validated_screen_triplet(delta_n, vx, vy)
    index_out, vx_out, vy_out = index.copy(), x_velocity.copy(), y_velocity.copy()
    index_out[0, :] = index_out[-1, :] = index_out[:, 0] = index_out[:, -1] = 0.0
    faces = (
        ((slice(1, -1), 0), x_velocity[1:-1, 1] > 0.0, (slice(1, -1), 1)),
        ((slice(1, -1), -1), x_velocity[1:-1, -2] < 0.0, (slice(1, -1), -2)),
        ((0, slice(1, -1)), y_velocity[1, 1:-1] > 0.0, (1, slice(1, -1))),
        ((-1, slice(1, -1)), y_velocity[-2, 1:-1] < 0.0, (-2, slice(1, -1))),
    )
    for edge, inflow, interior in faces:
        vx_out[edge] = xp.where(inflow, 0.0, x_velocity[interior])
        vy_out[edge] = xp.where(inflow, 0.0, y_velocity[interior])
    corners = (
        ((0, 0), x_velocity[1, 1] > 0.0, y_velocity[1, 1] > 0.0, (1, 1)),
        ((0, -1), x_velocity[1, -2] < 0.0, y_velocity[1, -2] > 0.0, (1, -2)),
        ((-1, 0), x_velocity[-2, 1] > 0.0, y_velocity[-2, 1] < 0.0, (-2, 1)),
        ((-1, -1), x_velocity[-2, -2] < 0.0, y_velocity[-2, -2] < 0.0, (-2, -2)),
    )
    for corner, x_inflow, y_inflow, diagonal in corners:
        if bool(x_inflow or y_inflow):
            vx_out[corner] = vy_out[corner] = 0.0
        else:
            vx_out[corner], vy_out[corner] = x_velocity[diagonal], y_velocity[diagonal]
    return index_out, vx_out, vy_out


def apply_hr4_open_boundaries(delta_n: Any, vx: Any, vy: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible HR-4A name for the frozen boundary policy."""
    return apply_hr4_boundaries(delta_n, vx, vy)


def _positive_spacing(value: float, name: str) -> float:
    result = _finite_real(value, name)
    if result <= 0.0:
        raise ValueError(f"HR-4 {name} must be positive")
    return result


def _nonnegative_coefficient(value: float, name: str) -> float:
    result = _finite_real(value, name)
    if result < 0.0:
        raise ValueError(f"HR-4 {name} must be non-negative")
    return result


def upwind_advection(q: Any, vx: Any, vy: Any, *, dx: float, dy: float):
    """Return vx*dq/dx + vy*dq/dy with local first-order upwinding."""
    scalar, x_velocity, y_velocity = _validated_screen_triplet(q, vx, vy)
    dx_value, dy_value = _positive_spacing(dx, "dx"), _positive_spacing(dy, "dy")
    result = xp.zeros_like(scalar)
    center = scalar[1:-1, 1:-1]
    backward_x = (center - scalar[1:-1, :-2]) / dx_value
    forward_x = (scalar[1:-1, 2:] - center) / dx_value
    backward_y = (center - scalar[:-2, 1:-1]) / dy_value
    forward_y = (scalar[2:, 1:-1] - center) / dy_value
    local_vx = x_velocity[1:-1, 1:-1]
    local_vy = y_velocity[1:-1, 1:-1]
    dqdx = xp.where(local_vx >= 0.0, backward_x, forward_x)
    dqdy = xp.where(local_vy >= 0.0, backward_y, forward_y)
    result[1:-1, 1:-1] = local_vx * dqdx + local_vy * dqdy
    return result


def laplacian_fd(q: Any, *, dx: float, dy: float):
    """Return the second-order central transverse finite-difference Laplacian."""
    scalar = _finite_screen(q, "q")
    dx_value, dy_value = _positive_spacing(dx, "dx"), _positive_spacing(dy, "dy")
    result = xp.zeros_like(scalar)
    center = scalar[1:-1, 1:-1]
    result[1:-1, 1:-1] = (
        (scalar[1:-1, 2:] - 2.0 * center + scalar[1:-1, :-2]) / dx_value**2
        + (scalar[2:, 1:-1] - 2.0 * center + scalar[:-2, 1:-1]) / dy_value**2
    )
    return result


def compute_hr4_rhs(
    delta_n: Any, vx: Any, vy: Any, *,
    dx: float, dy: float, chi: float, nu: float, n0: float,
    gravity_x: float = HR4_GRAVITY_X, gravity_y: float = HR4_GRAVITY_Y,
) -> dict[str, np.ndarray]:
    """Compute all HR-4B RHS terms from one bounded old-state tuple."""
    old_delta_n, old_vx, old_vy = apply_hr4_boundaries(delta_n, vx, vy)
    chi_value, nu_value = _nonnegative_coefficient(chi, "chi"), _nonnegative_coefficient(nu, "nu")
    n0_value = _finite_real(n0, "n0")
    if n0_value <= 1.0:
        raise ValueError("HR-4 n0 must be greater than one")
    gx, gy = _finite_real(gravity_x, "gravity_x"), _finite_real(gravity_y, "gravity_y")
    if gx != 0.0:
        raise ValueError("HR-4B buoyancy is frozen to the vy equation; gravity_x must be zero")
    rhs_delta_n = -upwind_advection(old_delta_n, old_vx, old_vy, dx=dx, dy=dy)
    rhs_delta_n += chi_value * laplacian_fd(old_delta_n, dx=dx, dy=dy)
    rhs_vx = -upwind_advection(old_vx, old_vx, old_vy, dx=dx, dy=dy)
    rhs_vx += nu_value * laplacian_fd(old_vx, dx=dx, dy=dy)
    rhs_vy = -upwind_advection(old_vy, old_vx, old_vy, dx=dx, dy=dy)
    rhs_vy += nu_value * laplacian_fd(old_vy, dx=dx, dy=dy)
    rhs_vy += old_delta_n / (n0_value - 1.0) * gy
    if not all(bool(xp.all(xp.isfinite(item))) for item in (rhs_delta_n, rhs_vx, rhs_vy)):
        raise ValueError("HR-4B RHS contains non-finite values")
    return {
        "old_delta_n": old_delta_n, "old_vx": old_vx, "old_vy": old_vy,
        "rhs_delta_n": rhs_delta_n, "rhs_vx": rhs_vx, "rhs_vy": rhs_vy,
    }


def thermal_channel_observables(
    delta_n: Any, vx: Any, vy: Any, *, dx: float, dy: float,
    x_min: float = HR4_X_MIN, y_min: float = HR4_Y_MIN,
) -> dict[str, object]:
    """Return scalar diagnostics for a negative-index thermal channel."""
    index, x_velocity, y_velocity = _validated_screen_triplet(delta_n, vx, vy)
    dx_value, dy_value = _positive_spacing(dx, "dx"), _positive_spacing(dy, "dy")
    x0, y0 = _finite_real(x_min, "x_min"), _finite_real(y_min, "y_min")
    weights = xp.maximum(-index, 0.0)
    weight_sum = float(to_cpu(xp.sum(weights)))
    speed = xp.sqrt(x_velocity**2 + y_velocity**2)
    result = {
        "min_delta_n": float(to_cpu(xp.min(index))),
        "max_delta_n": float(to_cpu(xp.max(index))),
        "max_abs_vx": float(to_cpu(xp.max(xp.abs(x_velocity)))),
        "max_abs_vy": float(to_cpu(xp.max(xp.abs(y_velocity)))),
        "max_abs_v": float(to_cpu(xp.max(speed))),
        "thermal_channel_centroid_x_m": float("nan"),
        "thermal_channel_centroid_y_m": float("nan"),
        "thermal_channel_width_m": float("nan"),
        "thermal_channel_defined": bool(weight_sum > 0.0),
    }
    if weight_sum == 0.0:
        return result
    x = x0 + xp.arange(index.shape[1], dtype=index.dtype) * dx_value
    y = y0 + xp.arange(index.shape[0], dtype=index.dtype) * dy_value
    x_grid, y_grid = xp.meshgrid(x, y, indexing="xy")
    centroid_x = xp.sum(weights * x_grid) / weight_sum
    centroid_y = xp.sum(weights * y_grid) / weight_sum
    radial_variance = xp.sum(weights * ((x_grid - centroid_x)**2 + (y_grid - centroid_y)**2)) / weight_sum
    result["thermal_channel_centroid_x_m"] = float(to_cpu(centroid_x))
    result["thermal_channel_centroid_y_m"] = float(to_cpu(centroid_y))
    result["thermal_channel_width_m"] = float(to_cpu(xp.sqrt(0.5 * radial_variance)))
    return result


def advance_hr4_single_screen(
    delta_n: Any, vx: Any, vy: Any, *,
    dx: float, dy: float, dt_hydro: float, chi: float, nu: float, n0: float,
    gravity_x: float = HR4_GRAVITY_X, gravity_y: float = HR4_GRAVITY_Y,
    cfl_limit: float = HR4_CFL_LIMIT, n_steps: int = 1,
    require_stable: bool = True, x_min: float = HR4_X_MIN, y_min: float = HR4_Y_MIN,
) -> dict[str, object]:
    """Advance an independent screen with fixed-step unsplit Forward Euler."""
    if isinstance(n_steps, bool) or int(n_steps) != n_steps or int(n_steps) <= 0:
        raise ValueError("HR-4 n_steps must be a positive integer")
    dt_value = _positive_spacing(dt_hydro, "dt_hydro")
    current_delta_n, current_vx, current_vy = _validated_screen_triplet(delta_n, vx, vy)
    started = __import__("time").perf_counter()
    last_audit: dict[str, object] | None = None
    for _ in range(int(n_steps)):
        old_delta_n, old_vx, old_vy = apply_hr4_boundaries(current_delta_n, current_vx, current_vy)
        audit = audit_hr4_stability(
            dx=dx, dy=dy, dt_hydro=dt_value, chi=chi, nu=nu,
            max_abs_vx=float(to_cpu(xp.max(xp.abs(old_vx)))),
            max_abs_vy=float(to_cpu(xp.max(xp.abs(old_vy)))),
            cfl_limit=cfl_limit,
        )
        if bool(require_stable) and not audit["overall_pass"]:
            raise ValueError(
                "HR-4 stability audit failed before single-screen advance: "
                f"combined_chi={audit['combined_number_chi']:.6g}, "
                f"combined_nu={audit['combined_number_nu']:.6g}"
            )
        rhs = compute_hr4_rhs(
            old_delta_n, old_vx, old_vy, dx=dx, dy=dy, chi=chi, nu=nu, n0=n0,
            gravity_x=gravity_x, gravity_y=gravity_y,
        )
        next_delta_n = old_delta_n + dt_value * rhs["rhs_delta_n"]
        next_vx = old_vx + dt_value * rhs["rhs_vx"]
        next_vy = old_vy + dt_value * rhs["rhs_vy"]
        if not all(bool(xp.all(xp.isfinite(item))) for item in (next_delta_n, next_vx, next_vy)):
            raise ValueError("HR-4B Euler update produced non-finite values")
        current_delta_n, current_vx, current_vy = apply_hr4_boundaries(next_delta_n, next_vx, next_vy)
        last_audit = audit
    elapsed = __import__("time").perf_counter() - started
    element_bytes = int(np.dtype(current_delta_n.dtype).itemsize)
    return {
        "delta_n": current_delta_n, "vx": current_vx, "vy": current_vy,
        "steps": int(n_steps), "stability": last_audit,
        "observables": thermal_channel_observables(
            current_delta_n, current_vx, current_vy, dx=dx, dy=dy, x_min=x_min, y_min=y_min,
        ),
        "performance": {
            "grid_shape": tuple(current_delta_n.shape),
            "dtype": np.dtype(current_delta_n.dtype).name,
            "backend": debug_backend()["backend"],
            "wall_time_s_total": elapsed,
            "wall_time_s_per_step": elapsed / int(n_steps),
            "temporary_working_set_estimate_bytes": int(12 * current_delta_n.size * element_bytes),
            "slow_time_history_stored": False,
        },
    }


def audit_hr4_stability(
    *, dx: float, dy: float, dt_hydro: float, chi: float, nu: float,
    max_abs_vx: float, max_abs_vy: float, cfl_limit: float = HR4_CFL_LIMIT,
) -> dict[str, object]:
    """Report independent explicit-diffusion and upwind-CFL constraints."""
    values = {
        name: _finite_real(value, name)
        for name, value in {
            "dx": dx, "dy": dy, "dt_hydro": dt_hydro, "chi": chi, "nu": nu,
            "max_abs_vx": max_abs_vx, "max_abs_vy": max_abs_vy, "cfl_limit": cfl_limit,
        }.items()
    }
    if min(values["dx"], values["dy"], values["dt_hydro"], values["cfl_limit"]) <= 0.0:
        raise ValueError("HR-4 dx, dy, dt_hydro, and cfl_limit must be positive")
    if values["chi"] < 0.0 or values["nu"] < 0.0:
        raise ValueError("HR-4 chi and nu must be non-negative")
    if values["max_abs_vx"] < 0.0 or values["max_abs_vy"] < 0.0:
        raise ValueError("HR-4 maximum absolute velocities must be non-negative")
    scale = values["dt_hydro"] * (1.0 / values["dx"] ** 2 + 1.0 / values["dy"] ** 2)
    chi_number, nu_number = values["chi"] * scale, values["nu"] * scale
    cfl = values["max_abs_vx"] * values["dt_hydro"] / values["dx"] + values["max_abs_vy"] * values["dt_hydro"] / values["dy"]
    if not all(math.isfinite(item) for item in (chi_number, nu_number, cfl)):
        raise ValueError("HR-4 stability audit produced non-finite derived values")
    passed_diffusion = chi_number <= 0.5 and nu_number <= 0.5
    passed_advection = cfl <= values["cfl_limit"]
    combined_chi = cfl + 2.0 * chi_number
    combined_nu = cfl + 2.0 * nu_number
    if not math.isfinite(combined_chi) or not math.isfinite(combined_nu):
        raise ValueError("HR-4 combined stability audit produced non-finite values")
    passed_combined = combined_chi <= 1.0 and combined_nu <= 1.0
    return {
        "diffusion_number_chi": chi_number, "diffusion_number_nu": nu_number,
        "advection_CFL": cfl,
        "passed_diffusion_chi": chi_number <= 0.5,
        "passed_diffusion_nu": nu_number <= 0.5,
        "passed_diffusion": passed_diffusion,
        "combined_number_chi": combined_chi, "combined_number_nu": combined_nu,
        "passed_combined_chi": combined_chi <= 1.0,
        "passed_combined_nu": combined_nu <= 1.0,
        "passed_advection": passed_advection, "passed_combined": passed_combined,
        "overall_pass": bool(passed_diffusion and passed_advection and passed_combined),
    }


def require_hr4_stability(**kwargs: Any) -> dict[str, object]:
    """Fail closed rather than silently proceeding after a failed audit."""
    audit = audit_hr4_stability(**kwargs)
    if not audit["overall_pass"]:
        raise ValueError(
            "HR-4 stability audit failed: "
            f"diffusion_chi={audit['diffusion_number_chi']:.6g}, "
            f"diffusion_nu={audit['diffusion_number_nu']:.6g}, CFL={audit['advection_CFL']:.6g}"
        )
    return audit
