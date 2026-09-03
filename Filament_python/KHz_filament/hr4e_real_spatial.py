"""Validation-only HR-4E-2C sampling of frozen real HR-3B POST screens.

This module deliberately does not provide a production HR-3B-to-HR-4 mapper.
It samples one immutable E1-B POST screen with deterministic bilinear
interpolation solely to expose identical real morphology to E2-C hydro grids.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .device import to_cpu, xp
from .hr4e_spatial import E2_COMMON_DT_S, e2_geometry, e2_metrics, run_e2_case
from .hr4e_timestep import load_e1b_screen, sha256_array


E2C_DURATION_S = 100.0e-6
E2C_SCHEMA = "khz_filament.hr4e2.real_spatial_case.v1"
E2C_REPRESENTATION = "bilinear_uniform_validation_only"


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _identity(loaded: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(loaded["screen_identity"])
    return {
        "screen_id": str(identity["screen_id"]),
        "screen_index": int(identity.get("screen_index", identity["index"])),
        "screen_z_m": _finite(identity.get("screen_z_m", identity["z_m"]), "screen_z_m"),
    }


def _assert_identity(loaded: Mapping[str, Any], requested: Mapping[str, Any]) -> dict[str, Any]:
    actual = _identity(loaded)
    required = ("screen_id", "screen_index", "screen_z_m")
    if any(key not in requested for key in required):
        raise ValueError("E2-C requires complete screen_id, screen_index, and screen_z_m")
    if str(requested["screen_id"]) != actual["screen_id"]:
        raise ValueError("E2-C screen_id does not match immutable manifest")
    if int(requested["screen_index"]) != actual["screen_index"]:
        raise ValueError("E2-C screen_index does not match immutable manifest")
    if not math.isclose(float(requested["screen_z_m"]), actual["screen_z_m"], rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("E2-C screen_z_m does not match immutable manifest")
    return actual


def _target_axes(geometry: Mapping[str, Any]):
    nx, ny = int(geometry["Nx"]), int(geometry["Ny"])
    x = float(geometry["x_min_m"]) + float(geometry["dx_m"]) * xp.arange(nx, dtype=xp.float64)
    y = float(geometry["y_min_m"]) + float(geometry["dy_m"]) * xp.arange(ny, dtype=xp.float64)
    return x, y


def bilinear_uniform_validation_sample(source: Any, source_grid: Mapping[str, Any], target_grid: Mapping[str, Any]):
    """Sample one uniform nodal field without filtering, clipping, or smoothing."""
    field = xp.asarray(source, dtype=xp.float64)
    source_ny, source_nx = field.shape
    if source_nx < 2 or source_ny < 2:
        raise ValueError("bilinear source needs at least two nodes on each axis")
    sx0, sy0 = float(source_grid["x_min_m"]), float(source_grid["y_min_m"])
    sdx, sdy = float(source_grid["dx_m"]), float(source_grid["dy_m"])
    tx, ty = _target_axes(target_grid)
    fx = (tx - sx0) / sdx
    fy = (ty - sy0) / sdy
    tolerance = 1.0e-12
    if bool(to_cpu(xp.any(fx < -tolerance))) or bool(to_cpu(xp.any(fx > (source_nx - 1) + tolerance))):
        raise ValueError("target x range lies outside immutable source range")
    if bool(to_cpu(xp.any(fy < -tolerance))) or bool(to_cpu(xp.any(fy > (source_ny - 1) + tolerance))):
        raise ValueError("target y range lies outside immutable source range")
    ix0 = xp.clip(xp.floor(fx).astype(xp.int64), 0, source_nx - 2)
    iy0 = xp.clip(xp.floor(fy).astype(xp.int64), 0, source_ny - 2)
    wx = xp.clip(fx - ix0, 0.0, 1.0)
    wy = xp.clip(fy - iy0, 0.0, 1.0)
    f00 = field[iy0[:, None], ix0[None, :]]
    f10 = field[iy0[:, None], ix0[None, :] + 1]
    f01 = field[iy0[:, None] + 1, ix0[None, :]]
    f11 = field[iy0[:, None] + 1, ix0[None, :] + 1]
    return ((1.0 - wy)[:, None] * ((1.0 - wx)[None, :] * f00 + wx[None, :] * f10)
            + wy[:, None] * ((1.0 - wx)[None, :] * f01 + wx[None, :] * f11))


def build_e2c_validation_state(
    screen_path: str,
    *,
    source_manifest_path: str,
    screen_identity: Mapping[str, Any],
    spacing_m: float,
) -> dict[str, Any]:
    """Build one target-grid state from a single immutable real POST screen."""
    loaded = load_e1b_screen(screen_path, source_manifest_path=source_manifest_path)
    identity = _assert_identity(loaded, screen_identity)
    geometry = e2_geometry(spacing_m)
    delta_n = bilinear_uniform_validation_sample(loaded["delta_n"], loaded["target_grid"], geometry)
    exact_zero_velocity = bool(to_cpu(xp.all(loaded["vx"] == 0.0))) and bool(to_cpu(xp.all(loaded["vy"] == 0.0)))
    if exact_zero_velocity:
        vx, vy = xp.zeros_like(delta_n), xp.zeros_like(delta_n)
        velocity_rule = "exact_zero_preserved"
    else:
        vx = bilinear_uniform_validation_sample(loaded["vx"], loaded["target_grid"], geometry)
        vy = bilinear_uniform_validation_sample(loaded["vy"], loaded["target_grid"], geometry)
        velocity_rule = "bilinear_uniform_validation_only"
    state = {"delta_n": delta_n, "vx": vx, "vy": vy}
    source = {
        "screen_identity": identity,
        "source_post_file_sha256": loaded["source_file_sha256"],
        "source_post_array_sha256": loaded["source_array_sha256"],
        "source_full_state_file_sha256": loaded["source_state_file_sha256"],
        "source_full_state_array_sha256": loaded["source_state_array_sha256"],
        "source_manifest_path": loaded["source_manifest_path"],
        "source_manifest_sha256": loaded["source_manifest_sha256"],
        "source_git_sha": loaded["source_git_sha"],
        "source_grid": dict(loaded["target_grid"]),
    }
    representation = {
        "kind": E2C_REPRESENTATION,
        "delta_n_rule": "bilinear_uniform_no_filter_no_smoothing_no_clipping",
        "velocity_rule": velocity_rule,
        "same_continuous_reference_used_for_20_10_5": True,
        "validation_representation_source_is_single_frozen_real_POST": True,
        "production_multigrid_mapping_modified": False,
        "scope_is_hydro_only_validation": True,
        "full_chain_transverse_convergence_claimed": False,
    }
    return {
        "state": state,
        "geometry": geometry,
        "source_provenance": source,
        "validation_representation": representation,
        "target_state_sha256": sha256_array(delta_n),
        "target_velocity_sha256": {"vx": sha256_array(vx), "vy": sha256_array(vy)},
        "initial_metrics": e2_metrics(delta_n, vx, vy, geometry=geometry),
    }


def run_e2c_case(
    screen_path: str,
    *,
    source_manifest_path: str,
    screen_identity: Mapping[str, Any],
    spacing_m: float,
    dt_hydro: float = E2_COMMON_DT_S,
) -> dict[str, Any]:
    """Advance full frozen HR-4 physics from one validation-only sampled POST."""
    prepared = build_e2c_validation_state(
        screen_path,
        source_manifest_path=source_manifest_path,
        screen_identity=screen_identity,
        spacing_m=spacing_m,
    )
    metadata = {
        "kind": "real_hr3b_post_validation_representation",
        "source_provenance": prepared["source_provenance"],
        "validation_representation": prepared["validation_representation"],
        "target_state_sha256": prepared["target_state_sha256"],
        "target_velocity_sha256": prepared["target_velocity_sha256"],
    }
    result = run_e2_case(
        family="E2-C",
        spacing_m=spacing_m,
        dt_hydro=dt_hydro,
        snapshot_times_s=(0.0, E2C_DURATION_S),
        state=prepared["state"],
        initial_metadata=metadata,
    )
    result.update({
        "schema": E2C_SCHEMA,
        "source_provenance": prepared["source_provenance"],
        "validation_representation": prepared["validation_representation"],
        "target_state_sha256": prepared["target_state_sha256"],
        "target_velocity_sha256": prepared["target_velocity_sha256"],
    })
    return result
