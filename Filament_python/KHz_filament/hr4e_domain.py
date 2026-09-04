"""Validation-only HR-4E-3 transverse domain/boundary convergence helpers.

This evidence layer drives the frozen single-screen HR-4 operator on a fixed
10 um / 1 us grid family.  It neither changes HR-4 physics nor supplies a
production mapping from HR-3B into the slow-flow state.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .device import debug_backend, to_cpu, xp
from .hr4 import HR4_CHI, HR4_GRAVITY_X, HR4_GRAVITY_Y, HR4_NU, advance_hr4_single_screen, audit_hr4_stability
from .hr4e_real_spatial import _assert_identity
from .hr4e_spatial import build_snapshot_schedule, e2_metrics
from .hr4e_timestep import (
    E1A_AMPLITUDE, E1A_CENTER_X_M, E1A_CENTER_Y_M, E1A_SIGMA_M,
    E1_BOUNDARY_FIRST_RING_RATIO_LIMIT, HR4_N0, classify_boundary_contamination,
    json_safe, load_e1b_screen, repository_git_sha, sha256_array, sha256_file,
)


E3_SCHEMA = "khz_filament.hr4e3.domain_case.v1"
E3_SPACING_M = 10.0e-6
E3_DT_S = 1.0e-6
E3_SNAPSHOT_TIMES_S = (0.0, 100.0e-6, 1.0e-3)
E3_DOMAINS = {
    "D0": (-1.5e-3, 1.5e-3, -1.0e-3, 2.5e-3),
    "D1": (-2.0e-3, 2.0e-3, -1.0e-3, 2.5e-3),
    "D2": (-1.5e-3, 1.5e-3, -1.5e-3, 3.5e-3),
    "D3": (-2.0e-3, 2.0e-3, -1.5e-3, 3.5e-3),
}


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def e3_geometry(domain_id: str, spacing_m: float = E3_SPACING_M) -> dict[str, Any]:
    """Return one inclusive nodal E3 domain without changing endpoints."""
    if domain_id not in E3_DOMAINS:
        raise ValueError(f"unknown E3 domain: {domain_id}")
    spacing = _finite(spacing_m, "spacing_m")
    if spacing <= 0.0:
        raise ValueError("spacing_m must be positive")
    x_min, x_max, y_min, y_max = E3_DOMAINS[domain_id]
    x_cells, y_cells = (x_max - x_min) / spacing, (y_max - y_min) / spacing
    nx, ny = int(round(x_cells)) + 1, int(round(y_cells)) + 1
    if not math.isclose(x_cells, nx - 1, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("E3 x domain is not an integer number of cells")
    if not math.isclose(y_cells, ny - 1, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("E3 y domain is not an integer number of cells")
    return {
        "domain_id": domain_id, "x_min_m": x_min, "x_max_m": x_max,
        "y_min_m": y_min, "y_max_m": y_max, "dx_m": spacing, "dy_m": spacing,
        "Nx": nx, "Ny": ny, "grid_layout": "collocated_nodal_inclusive",
    }


def e3_axes(geometry: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (
        float(geometry["x_min_m"]) + np.arange(int(geometry["Nx"]), dtype=np.float64) * float(geometry["dx_m"]),
        float(geometry["y_min_m"]) + np.arange(int(geometry["Ny"]), dtype=np.float64) * float(geometry["dy_m"]),
    )


def nested_d0_slices(geometry: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact D0 subset in an enlarged E3 geometry."""
    d0 = e3_geometry("D0", float(geometry["dx_m"]))
    if not math.isclose(float(geometry["dy_m"]), float(d0["dy_m"]), rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("E3 requires equal x/y D0 spacing")
    x_offset = (float(d0["x_min_m"]) - float(geometry["x_min_m"])) / float(geometry["dx_m"])
    y_offset = (float(d0["y_min_m"]) - float(geometry["y_min_m"])) / float(geometry["dy_m"])
    ix, iy = int(round(x_offset)), int(round(y_offset))
    if not math.isclose(x_offset, ix, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(y_offset, iy, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("D0 coordinates are not aligned with the candidate domain")
    if ix < 0 or iy < 0 or ix + int(d0["Nx"]) > int(geometry["Nx"]) or iy + int(d0["Ny"]) > int(geometry["Ny"]):
        raise ValueError("D0 is not nested in the candidate domain")
    x, y = e3_axes(geometry)
    d0x, d0y = e3_axes(d0)
    # The grids are exact integer-node nests in the declared decimal geometry.
    # Their separate binary ``origin + i*dx`` evaluations can differ by a few
    # ULP, so verify coordinate equality at a sub-attometre tolerance rather
    # than falsely rejecting identical physical nodes for representation noise.
    if not np.allclose(x[ix:ix + int(d0["Nx"])], d0x, rtol=0.0, atol=1.0e-15) or not np.allclose(y[iy:iy + int(d0["Ny"])], d0y, rtol=0.0, atol=1.0e-15):
        raise ValueError("D0 coordinate values are not aligned")
    return {"y_start": iy, "y_stop": iy + int(d0["Ny"]), "x_start": ix, "x_stop": ix + int(d0["Nx"]), "aligned": True, "coordinate_alignment_atol_m": 1.0e-15}


def _edge_metrics(field: Any) -> dict[str, Any]:
    array = xp.asarray(field, dtype=xp.float64)
    magnitude = xp.abs(array)
    peak = float(to_cpu(xp.max(magnitude)))
    edge = xp.concatenate((magnitude[0, :], magnitude[-1, :], magnitude[1:-1, 0], magnitude[1:-1, -1]))
    inner = xp.concatenate((magnitude[1, 1:-1], magnitude[-2, 1:-1], magnitude[2:-2, 1], magnitude[2:-2, -2]))
    edge_max, inner_max = float(to_cpu(xp.max(edge))), float(to_cpu(xp.max(inner)))
    return {
        "max_abs_boundary": edge_max, "max_abs_first_inner_ring": inner_max, "max_abs_domain": peak,
        "edge_to_peak_ratio": 0.0 if peak == 0.0 else edge_max / peak,
        "first_inner_ring_to_peak_ratio": 0.0 if peak == 0.0 else inner_max / peak,
        "exact_zero": bool(to_cpu(xp.all(array == 0.0))),
    }


def initial_edge_validity(state: Mapping[str, Any]) -> dict[str, Any]:
    """Apply only the existing E1 first-inner-ring threshold to a D0 POST state."""
    fields = {name: _edge_metrics(state[name]) for name in ("delta_n", "vx", "vy")}
    delta = fields["delta_n"]
    boundary = classify_boundary_contamination(e2_metrics(state["delta_n"], state["vx"], state["vy"], geometry=e3_geometry("D0")))
    valid = bool(delta["first_inner_ring_to_peak_ratio"] < E1_BOUNDARY_FIRST_RING_RATIO_LIMIT and not boundary["boundary_contaminated"])
    return {
        "schema": "khz_filament.hr4e3.initial_edge_validity.v1", "fields": fields,
        "existing_first_inner_ring_ratio_limit": E1_BOUNDARY_FIRST_RING_RATIO_LIMIT,
        "frozen_boundary_convention": boundary, "status": "PASS" if valid else "E3B_INVALID_INITIAL_DOMAIN_TRUNCATION",
    }


def build_e3_synthetic_state(geometry: Mapping[str, Any]) -> dict[str, Any]:
    x, y = e3_axes(geometry)
    x_grid, y_grid = xp.meshgrid(xp.asarray(x), xp.asarray(y), indexing="xy")
    delta = -xp.asarray(E1A_AMPLITUDE * xp.exp(-((x_grid - E1A_CENTER_X_M) ** 2 + (y_grid - E1A_CENTER_Y_M) ** 2) / (2.0 * E1A_SIGMA_M ** 2)), dtype=xp.float64)
    return {"delta_n": delta, "vx": xp.zeros_like(delta), "vy": xp.zeros_like(delta)}


def build_e3_real_post_state(screen_path: str, *, source_manifest_path: str, screen_identity: Mapping[str, Any], geometry: Mapping[str, Any]) -> dict[str, Any]:
    """Embed the immutable D0 POST state in an enlarged validation-only box."""
    loaded = load_e1b_screen(screen_path, source_manifest_path=source_manifest_path)
    identity = _assert_identity(loaded, screen_identity)
    d0 = e3_geometry("D0")
    if dict(loaded["target_grid"]) != {key: d0[key] for key in d0 if key != "domain_id"}:
        raise ValueError("real POST source does not prove the authoritative D0 geometry")
    source = {name: xp.asarray(loaded[name], dtype=xp.float64) for name in ("delta_n", "vx", "vy")}
    edge_audit = initial_edge_validity(source)
    if edge_audit["status"] != "PASS":
        raise ValueError("E3B_INVALID_INITIAL_DOMAIN_TRUNCATION")
    slices = nested_d0_slices(geometry)
    state = {name: xp.zeros((int(geometry["Ny"]), int(geometry["Nx"])), dtype=xp.float64) for name in source}
    for name, value in source.items():
        state[name][slices["y_start"]:slices["y_stop"], slices["x_start"]:slices["x_stop"]] = value
    return {
        "state": state, "source_provenance": {
            "screen_identity": identity, "source_post_file_sha256": loaded["source_file_sha256"],
            "source_post_array_sha256": loaded["source_array_sha256"], "source_full_state_file_sha256": loaded["source_state_file_sha256"],
            "source_full_state_array_sha256": loaded["source_state_array_sha256"], "source_manifest_sha256": loaded["source_manifest_sha256"],
            "source_git_sha": loaded["source_git_sha"],
        }, "initial_edge_validity": edge_audit,
        "extension": {"kind": "validation_only_zero_padding", "exterior_delta_n": 0.0, "exterior_vx": 0.0, "exterior_vy": 0.0, "d0_slices": slices},
    }


def e3_metrics(delta_n: Any, vx: Any, vy: Any, *, geometry: Mapping[str, Any]) -> dict[str, Any]:
    result = e2_metrics(delta_n, vx, vy, geometry=geometry)
    xc, yc, sx, sy = (float(result[key]) for key in ("xc_m", "yc_m", "sigma_x_m", "sigma_y_m"))
    clearances = {"left": xc - float(geometry["x_min_m"]), "right": float(geometry["x_max_m"]) - xc, "bottom": yc - float(geometry["y_min_m"]), "top": float(geometry["y_max_m"]) - yc}
    result["clearance_m"] = clearances
    result["clearance_in_sigma"] = {"left": clearances["left"] / sx, "right": clearances["right"] / sx, "bottom": clearances["bottom"] / sy, "top": clearances["top"] / sy}
    result["field_edge_metrics"] = {name: _edge_metrics(value) for name, value in {"delta_n": delta_n, "vx": vx, "vy": vy}.items()}
    return result


def write_e3_checkpoint(state: Mapping[str, Any], geometry: Mapping[str, Any], time_s: float, path: str | Path) -> dict[str, Any]:
    """Persist exactly one sparse, float64 E3 slow-state checkpoint."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(to_cpu(state[name]), dtype=np.float64) for name in ("delta_n", "vx", "vy")}
    if any(value.dtype != np.dtype("float64") for value in arrays.values()):
        raise ValueError("E3 checkpoints must preserve float64")
    x, y = e3_axes(geometry)
    metadata = {"schema": "khz_filament.hr4e3.checkpoint.v1", "time_s": float(time_s), "grid": dict(geometry), "dtype": "float64"}
    np.savez_compressed(destination, **arrays, x_m=x, y_m=y, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    return {"path": str(destination), "sha256": sha256_file(destination), "time_s": float(time_s), "time_us": float(time_s * 1.0e6), "arrays": {name: sha256_array(value) for name, value in arrays.items()}, "dtype": "float64", "grid": dict(geometry)}


def read_e3_checkpoint(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as data:
        result = {name: np.array(data[name], copy=True) for name in ("delta_n", "vx", "vy", "x_m", "y_m")}
        metadata = json.loads(str(data["metadata_json"].item()))
    if any(result[name].dtype != np.dtype("float64") for name in ("delta_n", "vx", "vy", "x_m", "y_m")):
        raise ValueError("E3 checkpoint dtype is not float64")
    result["metadata"] = metadata
    return result


def _field_norms(reference: Any, candidate: Any, geometry: Mapping[str, Any]) -> dict[str, float]:
    ref, other = np.asarray(to_cpu(reference), dtype=np.float64), np.asarray(to_cpu(candidate), dtype=np.float64)
    error, area = other - ref, float(geometry["dx_m"]) * float(geometry["dy_m"])
    l1, l2, linf = float(np.sum(np.abs(error)) * area), float(math.sqrt(np.sum(error ** 2) * area)), float(np.max(np.abs(error)))
    rl1, rl2, rlinf = float(np.sum(np.abs(ref)) * area), float(math.sqrt(np.sum(ref ** 2) * area)), float(np.max(np.abs(ref)))
    return {"absolute_L1": l1, "relative_L1": 0.0 if rl1 == 0.0 and l1 == 0.0 else float("inf") if rl1 == 0.0 else l1 / rl1, "absolute_L2": l2, "relative_L2": 0.0 if rl2 == 0.0 and l2 == 0.0 else float("inf") if rl2 == 0.0 else l2 / rl2, "absolute_Linf": linf, "relative_Linf": 0.0 if rlinf == 0.0 and linf == 0.0 else float("inf") if rlinf == 0.0 else linf / rlinf}


def common_d0_field_metrics(d0_checkpoint: str | Path, expanded_checkpoint: str | Path) -> dict[str, Any]:
    d0, large = read_e3_checkpoint(d0_checkpoint), read_e3_checkpoint(expanded_checkpoint)
    d0_grid, large_grid = d0["metadata"]["grid"], large["metadata"]["grid"]
    slices = nested_d0_slices(large_grid)
    if d0_grid != e3_geometry("D0"):
        raise ValueError("reference checkpoint is not E3 D0")
    return {"common_region": "D0_exact_coordinates", "alignment": slices, "fields": {name: _field_norms(d0[name], large[name][slices["y_start"]:slices["y_stop"], slices["x_start"]:slices["x_stop"]], d0_grid) for name in ("delta_n", "vx", "vy")}}


def run_e3_case(*, case_id: str, family: str, geometry: Mapping[str, Any], state: Mapping[str, Any], checkpoint_dir: str | Path, initial_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Advance a fixed E3 domain case and persist only 0/100 us/1 ms fields."""
    current = {name: xp.array(state[name], copy=True) for name in ("delta_n", "vx", "vy")}
    expected = (int(geometry["Ny"]), int(geometry["Nx"]))
    if any(tuple(current[name].shape) != expected or np.dtype(current[name].dtype) != np.dtype("float64") for name in current):
        raise ValueError("E3 state must be float64 and match geometry")
    snapshots, audits, completed = [], [], 0
    started, status, failure = time.perf_counter(), "PASS", None
    for time_s, target_steps in build_snapshot_schedule(E3_DT_S, E3_SNAPSHOT_TIMES_S):
        try:
            remaining = target_steps - completed
            if remaining:
                advanced = advance_hr4_single_screen(current["delta_n"], current["vx"], current["vy"], dx=E3_SPACING_M, dy=E3_SPACING_M, dt_hydro=E3_DT_S, chi=HR4_CHI, nu=HR4_NU, n0=HR4_N0, gravity_x=HR4_GRAVITY_X, gravity_y=HR4_GRAVITY_Y, n_steps=remaining, require_stable=True)
                current = {name: advanced[name] for name in current}
                audits.append(advanced["stability"]); completed = target_steps
            metrics = e3_metrics(current["delta_n"], current["vx"], current["vy"], geometry=geometry)
            audit = audit_hr4_stability(dx=E3_SPACING_M, dy=E3_SPACING_M, dt_hydro=E3_DT_S, chi=HR4_CHI, nu=HR4_NU, max_abs_vx=metrics["max_abs_vx_m_s"], max_abs_vy=metrics["max_abs_vy_m_s"])
            audits.append(audit)
            checkpoint = write_e3_checkpoint(current, geometry, time_s, Path(checkpoint_dir) / f"t{int(round(time_s * 1.0e6)):07d}us.npz")
            snapshots.append({"time_s": float(time_s), "time_us": float(time_s * 1.0e6), "hydro_step_count": completed, **metrics, "stability": audit, "checkpoint": checkpoint})
        except (ValueError, FloatingPointError, OSError) as error:
            status, failure = "FAIL_STABILITY_OR_PERSISTENCE", str(error)
            break
    return {"schema": E3_SCHEMA, "case_id": case_id, "family": family, "status": status, "failure_reason": failure, "configuration": {"grid": dict(geometry), "dt_hydro_s": E3_DT_S, "operator": {"chi_m2_s": HR4_CHI, "nu_m2_s": HR4_NU, "gravity_x_m_s2": HR4_GRAVITY_X, "gravity_y_m_s2": HR4_GRAVITY_Y, "advection_scheme": "first_order_upwind", "diffusion_scheme": "explicit_central_fd", "time_integrator": "explicit_euler", "boundary_delta_n": "ambient_dirichlet_zero", "boundary_velocity": "open_zero_gradient_outflow_ambient_inflow"}, "execution": {"backend": debug_backend()["backend"], "dtype": "float64", "git_sha": repository_git_sha()}, "snapshot_times_s": list(E3_SNAPSHOT_TIMES_S)}, "initial_state": {"delta_n_sha256": sha256_array(state["delta_n"]), **dict(initial_metadata)}, "snapshots": snapshots, "hydro_step_count": completed, "stability": {"overall_pass": bool(audits) and all(bool(item["overall_pass"]) for item in audits)}, "wall_time_s": time.perf_counter() - started, "slow_time_history_stored": False, "checkpoint_policy": "sparse_t0_t100us_t1ms_only"}


__all__ = ["E3_SCHEMA", "E3_SPACING_M", "E3_DT_S", "E3_SNAPSHOT_TIMES_S", "E3_DOMAINS", "e3_geometry", "e3_axes", "nested_d0_slices", "initial_edge_validity", "build_e3_synthetic_state", "build_e3_real_post_state", "e3_metrics", "write_e3_checkpoint", "read_e3_checkpoint", "common_d0_field_metrics", "run_e3_case"]
