"""HR-4A isobaric transverse slow-flow contracts; no PDE advance lives here."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping

import numpy as np

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
    array = np.asarray(value)
    if array.ndim != 3 or array.dtype.kind != "f":
        raise ValueError(f"HR-4 {name} must be a real floating [K, Ny, Nx] array")
    if min(array.shape) <= 0 or not np.all(np.isfinite(array)):
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
        np.zeros(volume_shape, result_dtype), np.zeros(volume_shape, result_dtype),
        np.zeros(volume_shape, result_dtype), geometry or HR4Geometry(),
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
    array = np.asarray(value)
    if array.ndim != 2 or array.dtype.kind != "f" or min(array.shape) < 3:
        raise ValueError(f"HR-4 {name} must be real floating [Ny, Nx] with Ny, Nx >= 3")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"HR-4 {name} must be finite")
    return array


def apply_hr4_open_boundaries(delta_n: Any, vx: Any, vy: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply ambient-index/open-velocity policy without a PDE step or wrap-around."""
    index, x_velocity, y_velocity = (
        _finite_screen(delta_n, "delta_n"), _finite_screen(vx, "vx"), _finite_screen(vy, "vy")
    )
    if index.shape != x_velocity.shape or index.shape != y_velocity.shape:
        raise ValueError("HR-4 boundary fields must have identical shapes")
    if index.dtype != x_velocity.dtype or index.dtype != y_velocity.dtype:
        raise ValueError("HR-4 boundary fields must have identical dtypes")
    index_out, vx_out, vy_out = index.copy(), x_velocity.copy(), y_velocity.copy()
    index_out[0, :] = index_out[-1, :] = index_out[:, 0] = index_out[:, -1] = 0.0
    faces = (
        ((slice(1, -1), 0), x_velocity[1:-1, 0] > 0.0, (slice(1, -1), 1)),
        ((slice(1, -1), -1), x_velocity[1:-1, -1] < 0.0, (slice(1, -1), -2)),
        ((0, slice(1, -1)), y_velocity[0, 1:-1] > 0.0, (1, slice(1, -1))),
        ((-1, slice(1, -1)), y_velocity[-1, 1:-1] < 0.0, (-2, slice(1, -1))),
    )
    for edge, inflow, interior in faces:
        vx_out[edge] = np.where(inflow, 0.0, x_velocity[interior])
        vy_out[edge] = np.where(inflow, 0.0, y_velocity[interior])
    corners = (
        ((0, 0), x_velocity[0, 0] > 0.0, y_velocity[0, 0] > 0.0, (1, 1)),
        ((0, -1), x_velocity[0, -1] < 0.0, y_velocity[0, -1] > 0.0, (1, -2)),
        ((-1, 0), x_velocity[-1, 0] > 0.0, y_velocity[-1, 0] < 0.0, (-2, 1)),
        ((-1, -1), x_velocity[-1, -1] < 0.0, y_velocity[-1, -1] < 0.0, (-2, -2)),
    )
    for corner, x_inflow, y_inflow, diagonal in corners:
        if bool(x_inflow or y_inflow):
            vx_out[corner] = vy_out[corner] = 0.0
        else:
            vx_out[corner], vy_out[corner] = x_velocity[diagonal], y_velocity[diagonal]
    return index_out, vx_out, vy_out


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
    if min(values["dx"], values["dy"], values["dt_hydro"], values["chi"], values["nu"], values["cfl_limit"]) <= 0.0:
        raise ValueError("HR-4 dx, dy, dt_hydro, chi, nu, and cfl_limit must be positive")
    if values["max_abs_vx"] < 0.0 or values["max_abs_vy"] < 0.0:
        raise ValueError("HR-4 maximum absolute velocities must be non-negative")
    scale = values["dt_hydro"] * (1.0 / values["dx"] ** 2 + 1.0 / values["dy"] ** 2)
    chi_number, nu_number = values["chi"] * scale, values["nu"] * scale
    cfl = values["max_abs_vx"] * values["dt_hydro"] / values["dx"] + values["max_abs_vy"] * values["dt_hydro"] / values["dy"]
    passed_diffusion = chi_number <= 0.5 and nu_number <= 0.5
    passed_advection = cfl <= values["cfl_limit"]
    return {
        "diffusion_number_chi": chi_number, "diffusion_number_nu": nu_number,
        "advection_CFL": cfl, "passed_diffusion": passed_diffusion,
        "passed_advection": passed_advection, "overall_pass": bool(passed_diffusion and passed_advection),
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
