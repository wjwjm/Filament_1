"""HR-4E-2 transverse-grid validation helpers.

This module is an evidence layer over the frozen HR-4B single-screen advance.
It does not change the PDE, state contract, boundary treatment, or integrator.
"""

from __future__ import annotations

import math
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .device import as_xp, debug_backend, to_cpu, xp
from .hr4 import (
    HR4_CFL_LIMIT,
    HR4_CHI,
    HR4_DX,
    HR4_DY,
    HR4_GRAVITY_X,
    HR4_GRAVITY_Y,
    HR4_NU,
    HR4_X_MAX,
    HR4_X_MIN,
    HR4_Y_MAX,
    HR4_Y_MIN,
    advance_hr4_single_screen,
    audit_hr4_stability,
)
from .hr4e_timestep import (
    E1A_AMPLITUDE,
    E1A_CENTER_X_M,
    E1A_CENTER_Y_M,
    E1A_SIGMA_M,
    HR4_N0,
    _stability_maximum,
    classify_boundary_contamination,
    json_safe,
    nodal_axes,
    repository_git_sha,
    sha256_array,
    thermal_channel_metrics,
)


E2_SCHEMA = "khz_filament.hr4e2.spatial_case.v1"
E2_SPACINGS_M: tuple[float, ...] = (20.0e-6, 10.0e-6, 5.0e-6)
E2_COMMON_DT_S = 0.125e-6
E2_FINE_GUARD_DT_S = 0.0625e-6
E2_SNAPSHOT_TIMES_S: tuple[float, ...] = (0.0, 25.0e-6, 50.0e-6, 100.0e-6, 250.0e-6, 500.0e-6, 1.0e-3)
E2_PRIMARY_HORIZONS_US: tuple[float, ...] = (100.0, 1000.0)
E2_ADVECTION_VX_M_S = 0.20
E2_ADVECTION_VY_M_S = 0.10
E2_ADVECTION_DURATION_S = 1.0e-3
E2_CENTROID_TOLERANCE_M = 0.02 * E1A_SIGMA_M
E2_WIDTH_RELATIVE_TOLERANCE = 0.01
E2_EXTREME_RELATIVE_TOLERANCE = 0.02
E2_M0_RELATIVE_TOLERANCE = 0.01


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def e2_geometry(spacing_m: float) -> dict[str, float | int | str]:
    """Return an inclusive nodal grid with the frozen physical E1 domain."""
    spacing = _finite(spacing_m, "spacing_m")
    if spacing <= 0.0:
        raise ValueError("spacing_m must be positive")
    x_cells = (HR4_X_MAX - HR4_X_MIN) / spacing
    y_cells = (HR4_Y_MAX - HR4_Y_MIN) / spacing
    nx, ny = int(round(x_cells)) + 1, int(round(y_cells)) + 1
    if not math.isclose(x_cells, nx - 1, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("frozen x domain is not an integer number of cells at this spacing")
    if not math.isclose(y_cells, ny - 1, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("frozen y domain is not an integer number of cells at this spacing")
    return {
        "x_min_m": float(HR4_X_MIN), "x_max_m": float(HR4_X_MAX),
        "y_min_m": float(HR4_Y_MIN), "y_max_m": float(HR4_Y_MAX),
        "dx_m": spacing, "dy_m": spacing, "Nx": nx, "Ny": ny,
        "grid_layout": "collocated_nodal_inclusive",
    }


def build_e2_synthetic_state(
    spacing_m: float,
    *,
    dtype: Any = np.float64,
    vx_m_s: float = 0.0,
    vy_m_s: float = 0.0,
) -> dict[str, Any]:
    """Evaluate the analytic E1 Gaussian directly on an E2 grid."""
    result_dtype = np.dtype(dtype)
    if result_dtype.kind != "f":
        raise ValueError("E2 dtype must be real floating point")
    geometry = e2_geometry(spacing_m)
    x, y = nodal_axes(
        x_min=float(geometry["x_min_m"]), y_min=float(geometry["y_min_m"]),
        dx=float(geometry["dx_m"]), dy=float(geometry["dy_m"]),
        shape=(int(geometry["Ny"]), int(geometry["Nx"])),
    )
    x_grid, y_grid = xp.meshgrid(as_xp(x), as_xp(y), indexing="xy")
    delta_n = -xp.asarray(
        E1A_AMPLITUDE * xp.exp(
            -((x_grid - E1A_CENTER_X_M) ** 2 + (y_grid - E1A_CENTER_Y_M) ** 2)
            / (2.0 * E1A_SIGMA_M ** 2)
        ), dtype=result_dtype,
    )
    vx = xp.full_like(delta_n, _finite(vx_m_s, "vx_m_s"))
    vy = xp.full_like(delta_n, _finite(vy_m_s, "vy_m_s"))
    return {"delta_n": delta_n, "vx": vx, "vy": vy}


def build_snapshot_schedule(dt_hydro: float, times_s: Sequence[float]) -> tuple[tuple[float, int], ...]:
    dt = _finite(dt_hydro, "dt_hydro")
    if dt <= 0.0:
        raise ValueError("dt_hydro must be positive")
    times = tuple(_finite(item, "snapshot time") for item in times_s)
    if not times or times[0] != 0.0 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("snapshot times must start at zero and be strictly increasing")
    result: list[tuple[float, int]] = []
    previous = 0
    for time_s in times:
        count = int(round(time_s / dt))
        if not math.isclose(time_s / dt, count, rel_tol=0.0, abs_tol=1.0e-9) or count < previous:
            raise ValueError("snapshot time must be an integral increasing step count")
        result.append((time_s, count))
        previous = count
    return tuple(result)


def e2_metrics(delta_n: Any, vx: Any, vy: Any, *, geometry: Mapping[str, Any]) -> dict[str, Any]:
    """Return E2 observables, including physical-coordinate M0."""
    result = thermal_channel_metrics(
        delta_n, vx, vy,
        dx=float(geometry["dx_m"]), dy=float(geometry["dy_m"]),
        x_min=float(geometry["x_min_m"]), y_min=float(geometry["y_min_m"]),
        x_max=float(geometry["x_max_m"]), y_max=float(geometry["y_max_m"]),
    )
    result["M0_negative_index_m2"] = float(result["weight_sum"]) * float(geometry["dx_m"]) * float(geometry["dy_m"])
    result.update(classify_boundary_contamination(result))
    return result


def e2_configuration(
    *, family: str, geometry: Mapping[str, Any], dt_hydro: float,
    chi: float, nu: float, gravity_x: float, gravity_y: float,
    snapshot_times_s: Sequence[float], initial_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "family": str(family), "grid": dict(geometry), "dt_hydro_s": _finite(dt_hydro, "dt_hydro"),
        "operator": {
            "chi_m2_s": _finite(chi, "chi"), "nu_m2_s": _finite(nu, "nu"),
            "gravity_x_m_s2": _finite(gravity_x, "gravity_x"), "gravity_y_m_s2": _finite(gravity_y, "gravity_y"),
            "n0": HR4_N0, "cfl_limit": HR4_CFL_LIMIT,
            "advection_scheme": "first_order_upwind", "diffusion_scheme": "explicit_central_fd",
            "time_integrator": "explicit_euler", "boundary_delta_n": "ambient_dirichlet_zero",
            "boundary_velocity": "open_zero_gradient_outflow_ambient_inflow",
        },
        "execution": {"backend": debug_backend()["backend"], "dtype": "float64", "git_sha": repository_git_sha()},
        "snapshot_times_s": [float(item) for item in snapshot_times_s],
        "initial_state": dict(initial_state),
    }


def run_e2_case(
    *, family: str, spacing_m: float, dt_hydro: float, snapshot_times_s: Sequence[float],
    state: Mapping[str, Any] | None = None, chi: float = HR4_CHI, nu: float = HR4_NU,
    gravity_x: float = HR4_GRAVITY_X, gravity_y: float = HR4_GRAVITY_Y,
    initial_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance one E2 screen with the unmodified frozen HR-4B operator."""
    geometry = e2_geometry(spacing_m)
    if state is None:
        state = build_e2_synthetic_state(spacing_m)
    current = {key: xp.array(state[key], copy=True) for key in ("delta_n", "vx", "vy")}
    expected_shape = (int(geometry["Ny"]), int(geometry["Nx"]))
    if any(tuple(current[key].shape) != expected_shape for key in current):
        raise ValueError("E2 state shape does not match its declared grid")
    initial_delta_n_sha256 = sha256_array(current["delta_n"])
    schedule = build_snapshot_schedule(dt_hydro, snapshot_times_s)
    audits: list[Mapping[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    completed = 0
    started = time.perf_counter()
    status, failure_reason = "PASS", None
    try:
        for time_s, target_steps in schedule:
            remaining = target_steps - completed
            if remaining:
                advanced = advance_hr4_single_screen(
                    current["delta_n"], current["vx"], current["vy"],
                    dx=float(geometry["dx_m"]), dy=float(geometry["dy_m"]), dt_hydro=dt_hydro,
                    chi=chi, nu=nu, n0=HR4_N0, gravity_x=gravity_x, gravity_y=gravity_y,
                    n_steps=remaining, require_stable=True,
                )
                current = {key: advanced[key] for key in ("delta_n", "vx", "vy")}
                audits.append(advanced["stability"])
                completed = target_steps
            metrics = e2_metrics(current["delta_n"], current["vx"], current["vy"], geometry=geometry)
            audit = audit_hr4_stability(
                dx=float(geometry["dx_m"]), dy=float(geometry["dy_m"]), dt_hydro=dt_hydro,
                chi=chi, nu=nu, max_abs_vx=metrics["max_abs_vx_m_s"], max_abs_vy=metrics["max_abs_vy_m_s"],
            )
            audits.append(audit)
            snapshots.append({"time_s": float(time_s), "time_us": float(time_s * 1.0e6), "hydro_step_count": completed, **metrics, "stability": audit})
    except (ValueError, FloatingPointError) as error:
        status, failure_reason = "FAIL_STABILITY", str(error)
    initial = {
        "kind": "analytic_gaussian" if initial_metadata is None else initial_metadata.get("kind", "external"),
        "delta_n_sha256": initial_delta_n_sha256,
        "shape": list(expected_shape), "dtype": np.dtype(current["delta_n"].dtype).name,
        "analytic_definition": {"amplitude": E1A_AMPLITUDE, "sigma_m": E1A_SIGMA_M, "center_x_m": E1A_CENTER_X_M, "center_y_m": E1A_CENTER_Y_M},
    }
    if initial_metadata:
        initial.update(dict(initial_metadata))
    configuration = e2_configuration(
        family=family, geometry=geometry, dt_hydro=dt_hydro, chi=chi, nu=nu,
        gravity_x=gravity_x, gravity_y=gravity_y, snapshot_times_s=snapshot_times_s, initial_state=initial,
    )
    return {
        "schema": E2_SCHEMA, "family": str(family), "status": status, "failure_reason": failure_reason,
        "configuration": configuration, "configuration_sha256": sha256_array(np.frombuffer(str(configuration).encode(), dtype=np.uint8)),
        "initial_state": initial, "initial_state_sha256": initial.get("delta_n_sha256"),
        "backend": debug_backend()["backend"], "dtype": np.dtype(current["delta_n"].dtype).name,
        "git_sha": repository_git_sha(), "snapshots": snapshots, "hydro_step_count": completed,
        "stability": _stability_maximum(audits) if audits else {"overall_pass": False},
        "wall_time_s": time.perf_counter() - started, "slow_time_history_stored": False,
    }


def analytic_translated_gaussian(spacing_m: float, time_s: float, *, vx_m_s: float = E2_ADVECTION_VX_M_S, vy_m_s: float = E2_ADVECTION_VY_M_S):
    geometry = e2_geometry(spacing_m)
    x, y = nodal_axes(x_min=float(geometry["x_min_m"]), y_min=float(geometry["y_min_m"]), dx=float(geometry["dx_m"]), dy=float(geometry["dy_m"]), shape=(int(geometry["Ny"]), int(geometry["Nx"])))
    x_grid, y_grid = xp.meshgrid(as_xp(x), as_xp(y), indexing="xy")
    return -xp.asarray(E1A_AMPLITUDE * xp.exp(-((x_grid - vx_m_s * time_s) ** 2 + (y_grid - vy_m_s * time_s) ** 2) / (2.0 * E1A_SIGMA_M ** 2)), dtype=xp.float64)


def run_e2_advection_case(*, spacing_m: float, dt_hydro: float = E2_COMMON_DT_S, duration_s: float = E2_ADVECTION_DURATION_S) -> dict[str, Any]:
    """Run E2-B's zero-physical-diffusion, prescribed-velocity diagnostic."""
    state = build_e2_synthetic_state(spacing_m, vx_m_s=E2_ADVECTION_VX_M_S, vy_m_s=E2_ADVECTION_VY_M_S)
    result = run_e2_case(family="E2-B", spacing_m=spacing_m, dt_hydro=dt_hydro, snapshot_times_s=(0.0, duration_s), state=state, chi=0.0, nu=0.0, gravity_x=0.0, gravity_y=0.0, initial_metadata={"kind": "analytic_gaussian_uniform_velocity", "vx_m_s": E2_ADVECTION_VX_M_S, "vy_m_s": E2_ADVECTION_VY_M_S})
    if result["status"] != "PASS":
        return result
    geometry = e2_geometry(spacing_m)
    # Re-run only the final state reconstruction is intentionally avoided; the
    # final snapshot contains all scalar diagnostics.  Compute analytic sampled
    # error from an identical fresh deterministic advance for E2-B only.
    current = state
    advanced = advance_hr4_single_screen(current["delta_n"], current["vx"], current["vy"], dx=float(geometry["dx_m"]), dy=float(geometry["dy_m"]), dt_hydro=dt_hydro, chi=0.0, nu=0.0, n0=HR4_N0, gravity_x=0.0, gravity_y=0.0, n_steps=int(round(duration_s / dt_hydro)))
    numerical = advanced["delta_n"]
    exact = analytic_translated_gaussian(spacing_m, duration_s)
    error = numerical - exact
    cell_area = float(geometry["dx_m"]) * float(geometry["dy_m"])
    final = result["snapshots"][-1]
    exact_metrics = e2_metrics(exact, xp.full_like(exact, E2_ADVECTION_VX_M_S), xp.full_like(exact, E2_ADVECTION_VY_M_S), geometry=geometry)
    sigma_x_growth = final["sigma_x_m"] - E1A_SIGMA_M
    sigma_y_growth = final["sigma_y_m"] - E1A_SIGMA_M
    result["advection_exact"] = {
        "vx_m_s": E2_ADVECTION_VX_M_S, "vy_m_s": E2_ADVECTION_VY_M_S, "duration_s": duration_s,
        "centroid_error_x_m": abs(final["xc_m"] - exact_metrics["xc_m"]), "centroid_error_y_m": abs(final["yc_m"] - exact_metrics["yc_m"]),
        "sigma_x_growth_m": sigma_x_growth, "sigma_y_growth_m": sigma_y_growth,
        "peak_amplitude_loss": abs(final["min_delta_n"]) / E1A_AMPLITUDE - 1.0,
        "L1_field_error_m2": float(to_cpu(xp.sum(xp.abs(error), dtype=xp.float64))) * cell_area,
        "L2_field_error_m": float(math.sqrt(float(to_cpu(xp.sum(error ** 2, dtype=xp.float64))) * cell_area)),
        "effective_artificial_diffusion_x_m2_s": max(sigma_x_growth * (final["sigma_x_m"] + E1A_SIGMA_M) / (2.0 * duration_s), 0.0),
        "effective_artificial_diffusion_y_m2_s": max(sigma_y_growth * (final["sigma_y_m"] + E1A_SIGMA_M) / (2.0 * duration_s), 0.0),
    }
    return result


__all__ = [
    "E2_SCHEMA", "E2_SPACINGS_M", "E2_COMMON_DT_S", "E2_FINE_GUARD_DT_S", "E2_SNAPSHOT_TIMES_S", "E2_PRIMARY_HORIZONS_US",
    "e2_geometry", "build_e2_synthetic_state", "build_snapshot_schedule", "e2_metrics", "run_e2_case", "run_e2_advection_case", "analytic_translated_gaussian",
]
