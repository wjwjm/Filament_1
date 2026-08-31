"""HR-4E-1 timestep-convergence benchmark helpers.

This module is deliberately a thin validation layer around the frozen HR-4B
single-screen operator.  It owns no new fluid model and does not change the
HR-4C/HR-4D lifecycle.  The default E1-A state is a deterministic, synthetic
    negative Gaussian on the provisional inclusive collocated/nodal grid.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .device import as_xp, debug_backend, to_cpu, xp
from .hr4 import (
    HR4_CHI,
    HR4_CFL_LIMIT,
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


E1A_SCHEMA = "khz_filament.hr4e1.timestep_case.v1"
E1B_SCHEMA = "khz_filament.hr4e1.real_post_case.v1"
E1A_AMPLITUDE = 1.0e-5
E1A_SIGMA_M = 80.0e-6
E1A_CENTER_X_M = 0.0
E1A_CENTER_Y_M = 0.0
E1A_SNAPSHOT_TIMES_US: tuple[float, ...] = (0.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0)
# Parse the decimal microsecond labels directly so the recorded physical times
# are the same canonical Python floats as literals such as ``100.0e-6``.  A
# chained multiplication (``value * 1.0e-6``) can land one ULP below that
# value, which is needlessly visible in exact schedule/manifest comparisons.
E1A_SNAPSHOT_TIMES_S: tuple[float, ...] = tuple(
    float(f"{value:g}e-6") for value in E1A_SNAPSHOT_TIMES_US
)
E1A_PRIMARY_HORIZONS_US: tuple[float, ...] = (100.0, 1000.0)
E1B_SNAPSHOT_TIMES_US: tuple[float, ...] = (0.0, 25.0, 50.0, 100.0)
E1B_SNAPSHOT_TIMES_S: tuple[float, ...] = tuple(
    float(f"{value:g}e-6") for value in E1B_SNAPSHOT_TIMES_US
)
E1B_PRIMARY_HORIZONS_US: tuple[float, ...] = (100.0,)
E1A_SUPPORTED_DT_US: tuple[float, ...] = (1.0, 0.5, 0.25)
E1A_DT_TOLERANCE_S = 1.0e-15
E1A_CENTROID_TOLERANCE_M = 0.02 * E1A_SIGMA_M
E1A_WIDTH_RELATIVE_TOLERANCE = 0.01
E1A_EXTREME_RELATIVE_TOLERANCE = 0.02
E1_BOUNDARY_FIRST_RING_RATIO_LIMIT = 1.0e-3
E1_BOUNDARY_SIGMA_Y_TOP_CLEARANCE_LIMIT = 0.25
HR4_N0 = 1.00027


def e1b_source_grid() -> dict[str, float | int | str]:
    """Return the only source grid accepted for E1-B pure translation.

    The optical runner's transverse axes use ``L/N`` spacing.  Consequently
    the exact source proof is Nx=301, Ny=351, Lx=3.01 mm, Ly=3.51 mm, which
    yields 10 um spacing and source ranges [-1.5, 1.5] mm and [-1.75, 1.75]
    mm.  E1-B only relabels the y origin by +0.75 mm; it never interpolates.
    """
    return {
        "Nx": 301,
        "Ny": 351,
        "Lx_m": 3.01e-3,
        "Ly_m": 3.51e-3,
        "dx_m": 10.0e-6,
        "dy_m": 10.0e-6,
        "x_min_m": -1.5e-3,
        "x_max_m": 1.5e-3,
        "y_min_m": -1.75e-3,
        "y_max_m": 1.75e-3,
        "grid_layout": "runner_centered_l_over_n",
    }


def e1b_geometry_translation() -> dict[str, Any]:
    """Return the documented pure coordinate translation from source to E1."""
    source = e1b_source_grid()
    target = e1a_geometry()
    return {
        "method": "pure_translation_no_interpolation",
        "x_translation_m": 0.0,
        "y_translation_m": 0.75e-3,
        "source_x_range_m": [float(source["x_min_m"]), float(source["x_max_m"])],
        "source_y_range_m": [float(source["y_min_m"]), float(source["y_max_m"])],
        "target_x_range_m": [float(target["x_min_m"]), float(target["x_max_m"])],
        "target_y_range_m": [float(target["y_min_m"]), float(target["y_max_m"])],
    }


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _as_real_screen(value: Any, name: str, *, copy: bool = False):
    array = xp.array(value, copy=True) if copy else as_xp(value)
    if array.ndim != 2 or array.dtype.kind != "f" or min(array.shape) < 3:
        raise ValueError(f"{name} must be a real floating [Ny, Nx] screen with Ny, Nx >= 3")
    if not bool(xp.all(xp.isfinite(array))):
        raise ValueError(f"{name} must be finite")
    return array


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash file bytes without opening an array writable."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    """Hash an array's shape, dtype, and C-order values deterministically."""
    # Hashing is a one-time provenance operation at an I/O boundary.  It must
    # not occur inside the hydro-step loop; transfer a CuPy value only here.
    array = np.asarray(to_cpu(value))
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def repository_git_sha(repo_root: str | Path | None = None) -> str | None:
    """Return the current Git SHA when available; otherwise return ``None``."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def e1a_geometry() -> dict[str, float | int | str]:
    """Return the frozen E1-A inclusive collocated/nodal grid metadata."""
    nx_float = (HR4_X_MAX - HR4_X_MIN) / HR4_DX
    ny_float = (HR4_Y_MAX - HR4_Y_MIN) / HR4_DY
    # Endpoints are included: x=-1.5..+1.5 mm and y=-1..+2.5 mm.  The
    # interval count is 300 by 350, while the collocated nodal array has one
    # sample at each endpoint (301 by 351).
    nx, ny = int(round(nx_float)) + 1, int(round(ny_float)) + 1
    if not math.isclose(nx_float, nx - 1, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("E1-A x extent is not an integer number of cells")
    if not math.isclose(ny_float, ny - 1, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("E1-A y extent is not an integer number of cells")
    return {
        "x_min_m": float(HR4_X_MIN),
        "x_max_m": float(HR4_X_MAX),
        "y_min_m": float(HR4_Y_MIN),
        "y_max_m": float(HR4_Y_MAX),
        "dx_m": float(HR4_DX),
        "dy_m": float(HR4_DY),
        "Nx": nx,
        "Ny": ny,
        "grid_layout": "collocated_nodal_inclusive",
    }


def nodal_axes(
    *,
    x_min: float = HR4_X_MIN,
    y_min: float = HR4_Y_MIN,
    dx: float = HR4_DX,
    dy: float = HR4_DY,
    shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build inclusive collocated/nodal coordinates for a screen."""
    dx_value, dy_value = _finite_float(dx, "dx"), _finite_float(dy, "dy")
    if dx_value <= 0.0 or dy_value <= 0.0:
        raise ValueError("dx and dy must be positive")
    if shape is None:
        geometry = e1a_geometry()
        shape = (int(geometry["Ny"]), int(geometry["Nx"]))
    if len(shape) != 2 or min(int(item) for item in shape) < 3:
        raise ValueError("shape must contain Ny and Nx, both at least three")
    y_size, x_size = (int(item) for item in shape)
    return (
        _finite_float(x_min, "x_min") + np.arange(x_size, dtype=np.float64) * dx_value,
        _finite_float(y_min, "y_min") + np.arange(y_size, dtype=np.float64) * dy_value,
    )


def build_e1a_initial_state(
    *, dtype: Any = np.float64, return_axes: bool = False
) -> dict[str, Any]:
    """Construct the deterministic E1-A Gaussian and exact-zero velocities."""
    result_dtype = np.dtype(dtype)
    if result_dtype.kind != "f":
        raise ValueError("E1-A dtype must be real floating point")
    geometry = e1a_geometry()
    x, y = nodal_axes(shape=(int(geometry["Ny"]), int(geometry["Nx"])))
    x_grid, y_grid = xp.meshgrid(as_xp(x), as_xp(y), indexing="xy")
    delta_n = -xp.asarray(
        E1A_AMPLITUDE
        * xp.exp(
            -((x_grid - E1A_CENTER_X_M) ** 2 + (y_grid - E1A_CENTER_Y_M) ** 2)
            / (2.0 * E1A_SIGMA_M**2)
        ),
        dtype=result_dtype,
    )
    vx = xp.zeros_like(delta_n)
    vy = xp.zeros_like(delta_n)
    state: dict[str, Any] = {"delta_n": delta_n, "vx": vx, "vy": vy}
    if return_axes:
        state.update({"x": x, "y": y})
    return state


# Readable aliases used by small external validation scripts.
synthetic_initial_state = build_e1a_initial_state
build_synthetic_initial_state = build_e1a_initial_state


def load_e1b_screen(
    path: str | Path,
    *,
    source_manifest_path: str | Path | None = None,
    require_e1a_grid: bool = True,
) -> dict[str, Any]:
    """Load one immutable HR-3B POST screen and initialise zero velocity.

    E1-B is valid only when the adjacent (or explicitly supplied) reference
    manifest proves the exact HR-3B source grid and its pure coordinate
    translation into E1-A.  The source is opened read-only
    (``mmap_mode='r'``), verified against both file/array hashes, and never
    modified.  A private copy is returned because the caller will evolve the
    state.  ``require_e1a_grid`` is retained for API compatibility; the source
    proof is always mandatory and always fixes the E1-A shape.
    """
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"E1-B screen does not exist: {source_path}")
    manifest_path = Path(source_manifest_path) if source_manifest_path is not None else (
        source_path.parent / "post_reference_manifest.json"
    )
    if not manifest_path.is_file():
        raise ValueError(
            "E1-B screen requires a post_reference_manifest.json proving the "
            "exact source grid and pure translation"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid E1-B source manifest: {manifest_path}") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("E1-B source manifest must contain a mapping")

    expected_source = e1b_source_grid()
    source_grid = manifest.get("source_grid")
    if not isinstance(source_grid, Mapping):
        raise ValueError("E1-B source manifest lacks exact source_grid proof")
    for key, expected in expected_source.items():
        actual = source_grid.get(key)
        if isinstance(expected, float):
            if actual is None or not math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1.0e-15
            ):
                raise ValueError(f"E1-B source grid mismatch at source_grid.{key}")
        elif actual != expected:
            raise ValueError(f"E1-B source grid mismatch at source_grid.{key}")

    target_grid = manifest.get("target_grid")
    expected_target = e1a_geometry()
    if not isinstance(target_grid, Mapping):
        raise ValueError("E1-B source manifest lacks target_grid proof")
    for key, expected in expected_target.items():
        actual = target_grid.get(key)
        if isinstance(expected, float):
            if actual is None or not math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1.0e-15
            ):
                raise ValueError(f"E1-B target grid mismatch at target_grid.{key}")
        elif actual != expected:
            raise ValueError(f"E1-B target grid mismatch at target_grid.{key}")

    expected_translation = e1b_geometry_translation()
    translation = manifest.get("geometry_translation")
    if not isinstance(translation, Mapping):
        raise ValueError("E1-B source manifest lacks geometry_translation proof")
    if translation.get("method") != expected_translation["method"]:
        raise ValueError("E1-B source must use pure translation without interpolation")
    for key in ("x_translation_m", "y_translation_m"):
        if not math.isclose(
            float(translation.get(key)),
            float(expected_translation[key]),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError(f"E1-B geometry translation mismatch at {key}")

    screens = manifest.get("screens")
    if not isinstance(screens, Mapping):
        raise ValueError("E1-B source manifest lacks selected screen hashes")
    screen_matches: list[tuple[str, Mapping[str, Any]]] = []
    source_resolved = source_path.resolve()
    for label, entry in screens.items():
        if not isinstance(entry, Mapping) or entry.get("array_path") is None:
            continue
        candidate = Path(str(entry["array_path"]))
        if not candidate.is_absolute():
            candidate = manifest_path.parent / candidate
        if candidate.resolve() == source_resolved:
            screen_matches.append((str(label), entry))
    if len(screen_matches) != 1:
        raise ValueError(
            "E1-B screen must match exactly one immutable selected-screen manifest entry"
        )
    screen_label, screen_entry = screen_matches[0]
    source_file_sha256 = sha256_file(source_path)
    expected_file_sha256 = screen_entry.get("file_sha256")
    if not isinstance(expected_file_sha256, str) or source_file_sha256 != expected_file_sha256:
        raise ValueError("E1-B screen file hash does not match the immutable reference")

    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    try:
        geometry = e1a_geometry()
        expected_shape = (int(geometry["Ny"]), int(geometry["Nx"]))
        if tuple(source.shape) != expected_shape:
            raise ValueError(
                "E1-B screen must use the exact E1 inclusive nodal grid "
                f"{expected_shape}, got {tuple(source.shape)}"
            )
        delta_n = _as_real_screen(source, "E1-B delta_n", copy=True)
    finally:
        close = getattr(source, "_mmap", None)
        if close is not None:
            close.close()
    source_array_sha256 = sha256_array(delta_n)
    expected_array_sha256 = screen_entry.get("array_sha256")
    if not isinstance(expected_array_sha256, str) or source_array_sha256 != expected_array_sha256:
        raise ValueError("E1-B screen array hash does not match the immutable reference")
    if screen_entry.get("shape") is not None and list(delta_n.shape) != list(screen_entry["shape"]):
        raise ValueError("E1-B screen shape does not match its immutable reference")
    if screen_entry.get("dtype") is not None and delta_n.dtype.name != str(screen_entry["dtype"]):
        raise ValueError("E1-B screen dtype does not match its immutable reference")

    n0 = _finite_float(manifest.get("n0"), "E1-B source n0")
    if n0 <= 1.0:
        raise ValueError("E1-B source n0 must be greater than one")
    source_dtype = str(manifest.get("source_dtype", delta_n.dtype.name))
    source_backend = str(manifest.get("source_backend", manifest.get("backend", "unknown")))
    source_git_sha = manifest.get("source_git_sha", manifest.get("git_sha"))
    source_state_file_sha256 = manifest.get("hr3b_state_file_sha256")
    source_state_array_sha256 = manifest.get("hr3b_state_sha256")
    vx = xp.zeros_like(delta_n)
    vy = xp.zeros_like(delta_n)
    return {
        "delta_n": delta_n,
        "vx": vx,
        "vy": vy,
        "source_path": str(source_path),
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_sha256": source_file_sha256,
        "source_file_sha256": source_file_sha256,
        "source_array_sha256": source_array_sha256,
        "source_state_file_sha256": source_state_file_sha256,
        "source_state_array_sha256": source_state_array_sha256,
        "source_dtype": source_dtype,
        "source_backend": source_backend,
        "source_git_sha": source_git_sha,
        "n0": n0,
        "screen_identity": {
            "screen_id": screen_label,
            **{
                key: screen_entry[key]
                for key in ("index", "screen_index", "z_m", "screen_z_m")
                if key in screen_entry
            },
        },
        "source_grid": dict(source_grid),
        "target_grid": dict(target_grid),
        "geometry_translation": dict(translation),
        "velocity_initialization": "zero",
        "coordinate_relabeling": {
            "convention": "pure_translation_no_interpolation",
            "x_m": "source_x_m + 0.0",
            "y_m": "source_y_m + 0.75e-3",
            "source_screen_shape": list(delta_n.shape),
            "source_grid": dict(source_grid),
            "target_grid": dict(target_grid),
            "translation_m": {
                "x": 0.0,
                "y": 0.75e-3,
            },
            "interpolation": False,
        },
    }


def _validate_state_triplet(state: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = ("delta_n", "vx", "vy")
    if any(name not in state for name in required):
        raise ValueError("state must contain delta_n, vx, and vy")
    delta_n, vx, vy = tuple(_as_real_screen(state[name], name) for name in required)
    if delta_n.shape != vx.shape or delta_n.shape != vy.shape:
        raise ValueError("delta_n, vx, and vy must have identical shapes")
    if delta_n.dtype != vx.dtype or delta_n.dtype != vy.dtype:
        raise ValueError("delta_n, vx, and vy must have identical dtypes")
    return delta_n, vx, vy


def thermal_channel_metrics(
    delta_n: Any,
    vx: Any,
    vy: Any,
    *,
    dx: float = HR4_DX,
    dy: float = HR4_DY,
    x_min: float = HR4_X_MIN,
    y_min: float = HR4_Y_MIN,
    x_max: float | None = HR4_X_MAX,
    y_max: float | None = HR4_Y_MAX,
) -> dict[str, Any]:
    """Calculate E1 observables using ``W=max(-delta_n, 0)``.

    ``sigma_x_m`` and ``sigma_y_m`` are separate second-moment RMS widths,
    not radial RMS width, FWHM, or a diameter.  Coordinates use the inclusive
    collocated/nodal convention of HR-4 (the endpoints are sampled).
    """
    state = {"delta_n": delta_n, "vx": vx, "vy": vy}
    index, x_velocity, y_velocity = _validate_state_triplet(state)
    dx_value, dy_value = _finite_float(dx, "dx"), _finite_float(dy, "dy")
    x0, y0 = _finite_float(x_min, "x_min"), _finite_float(y_min, "y_min")
    if dx_value <= 0.0 or dy_value <= 0.0:
        raise ValueError("dx and dy must be positive")
    if x_max is None:
        x_max_value = x0 + index.shape[1] * dx_value
    else:
        x_max_value = _finite_float(x_max, "x_max")
    if y_max is None:
        y_max_value = y0 + index.shape[0] * dy_value
    else:
        y_max_value = _finite_float(y_max, "y_max")

    weights = xp.maximum(-index, 0.0)
    weight_sum = float(to_cpu(xp.sum(weights, dtype=xp.float64)))
    x, y = nodal_axes(
        x_min=x0, y_min=y0, dx=dx_value, dy=dy_value, shape=index.shape
    )
    x_grid, y_grid = xp.meshgrid(as_xp(x), as_xp(y), indexing="xy")
    if weight_sum > 0.0:
        xc = float(to_cpu(xp.sum(weights * x_grid, dtype=xp.float64) / weight_sum))
        yc = float(to_cpu(xp.sum(weights * y_grid, dtype=xp.float64) / weight_sum))
        sigma_x = float(to_cpu(xp.sqrt(
            xp.sum(weights * (x_grid - xc) ** 2, dtype=xp.float64) / weight_sum
        )))
        sigma_y = float(to_cpu(xp.sqrt(
            xp.sum(weights * (y_grid - yc) ** 2, dtype=xp.float64) / weight_sum
        )))
        defined = True
    else:
        xc = yc = sigma_x = sigma_y = float("nan")
        defined = False

    maximum_abs_domain = float(to_cpu(xp.max(xp.abs(index))))
    boundary = xp.concatenate(
        (
            index[0, :].ravel(),
            index[-1, :].ravel(),
            index[1:-1, 0].ravel(),
            index[1:-1, -1].ravel(),
        )
    )
    maximum_abs_boundary = float(to_cpu(xp.max(xp.abs(boundary))))
    first_ring = xp.concatenate(
        (
            index[1, 1:-1].ravel(),
            index[-2, 1:-1].ravel(),
            index[2:-2, 1].ravel(),
            index[2:-2, -2].ravel(),
        )
    )
    maximum_abs_first_ring = float(to_cpu(xp.max(xp.abs(first_ring))))
    if maximum_abs_domain > 0.0:
        edge_metric = maximum_abs_boundary / maximum_abs_domain
        first_ring_metric = maximum_abs_first_ring / maximum_abs_domain
    else:
        edge_metric = first_ring_metric = 0.0
    top_clearance = y_max_value - yc if defined else float("nan")
    sigma_y_over_top = sigma_y / top_clearance if defined and top_clearance > 0.0 else float("nan")
    speed = xp.sqrt(x_velocity**2 + y_velocity**2)

    result: dict[str, Any] = {
        "weight_sum": weight_sum,
        "thermal_channel_defined": defined,
        "width_definition": "second_moment_sigma",
        "xc_m": xc,
        "yc_m": yc,
        "sigma_x_m": sigma_x,
        "sigma_y_m": sigma_y,
        "min_delta_n": float(to_cpu(xp.min(index))),
        "max_delta_n": float(to_cpu(xp.max(index))),
        "max_abs_vx_m_s": float(to_cpu(xp.max(xp.abs(x_velocity)))),
        "max_abs_vy_m_s": float(to_cpu(xp.max(xp.abs(y_velocity)))),
        "max_abs_v_m_s": float(to_cpu(xp.max(speed))),
        "max_boundary_abs_delta_n": maximum_abs_boundary,
        "max_first_interior_ring_abs_delta_n": maximum_abs_first_ring,
        "max_domain_abs_delta_n": maximum_abs_domain,
        "formal_edge_boundary_ratio": edge_metric,
        "first_interior_ring_ratio": first_ring_metric,
        "edge_contamination_ratio": edge_metric,
        "edge_metric": edge_metric,
        "y_max_m": y_max_value,
        "top_clearance_m": top_clearance,
        "sigma_y_over_top_clearance": sigma_y_over_top,
    }
    # Compact aliases make CSV/JSON outputs easy to consume while retaining
    # explicit SI-unit names as the canonical fields above.
    result.update(
        {
            "xc": xc,
            "yc": yc,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "max_abs_vx": result["max_abs_vx_m_s"],
            "max_abs_vy": result["max_abs_vy_m_s"],
            "max_abs_v": result["max_abs_v_m_s"],
        }
    )
    return result


def classify_boundary_contamination(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen E1 boundary-contamination criteria to one snapshot.

    The formal outer-edge ratio remains a diagnostic only.  Automatic
    contamination is triggered by either the first-interior-ring proxy or the
    top-clearance width ratio, with no inferred threshold for the outer edge.
    """
    first_ring = float(metrics.get("first_interior_ring_ratio", float("nan")))
    sigma_top = float(metrics.get("sigma_y_over_top_clearance", float("nan")))
    first_ring_hit = math.isfinite(first_ring) and (
        first_ring >= E1_BOUNDARY_FIRST_RING_RATIO_LIMIT
    )
    sigma_top_hit = math.isfinite(sigma_top) and (
        sigma_top >= E1_BOUNDARY_SIGMA_Y_TOP_CLEARANCE_LIMIT
    )
    reasons = []
    if first_ring_hit:
        reasons.append("first_interior_ring_ratio")
    if sigma_top_hit:
        reasons.append("sigma_y_over_top_clearance")
    return {
        "boundary_contaminated": bool(first_ring_hit or sigma_top_hit),
        "boundary_contamination_reasons": reasons,
        "boundary_contamination_criteria": {
            "first_interior_ring_ratio": first_ring,
            "first_interior_ring_ratio_limit": E1_BOUNDARY_FIRST_RING_RATIO_LIMIT,
            "sigma_y_over_top_clearance": sigma_top,
            "sigma_y_over_top_clearance_limit": E1_BOUNDARY_SIGMA_Y_TOP_CLEARANCE_LIMIT,
            "formal_edge_boundary_ratio": metrics.get("formal_edge_boundary_ratio"),
            "formal_edge_ratio_is_diagnostic_only": True,
        },
    }


def build_snapshot_step_schedule(
    dt_hydro: float, snapshot_times_s: Sequence[float] = E1A_SNAPSHOT_TIMES_S
) -> tuple[tuple[float, int], ...]:
    """Return exact cumulative step counts for requested snapshot times.

    Formal E1 cases use times divisible by all three candidate timesteps.  A
    non-integral request fails closed rather than silently changing the sampled
    physical time.
    """
    dt = _finite_float(dt_hydro, "dt_hydro")
    if dt <= 0.0:
        raise ValueError("dt_hydro must be positive")
    times = tuple(_finite_float(value, "snapshot time") for value in snapshot_times_s)
    if not times or abs(times[0]) > E1A_DT_TOLERANCE_S:
        raise ValueError("snapshot times must start at zero")
    if any(value < 0.0 for value in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("snapshot times must be non-negative and strictly increasing")
    previous = 0
    rows: list[tuple[float, int]] = []
    for time_s in times:
        ratio = time_s / dt
        count = int(round(ratio))
        if not math.isclose(ratio, count, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(f"snapshot time {time_s!r} is not an integer number of hydro steps")
        if count < previous:
            raise ValueError("snapshot step counts must increase")
        rows.append((time_s, count))
        previous = count
    return tuple(rows)


def _stability_maximum(audits: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(audits)
    if not items:
        raise ValueError("at least one stability audit is required")
    numeric_keys = (
        "diffusion_number_chi",
        "diffusion_number_nu",
        "advection_CFL",
        "combined_number_chi",
        "combined_number_nu",
    )
    result: dict[str, Any] = {
        key: max(float(item[key]) for item in items) for key in numeric_keys
    }
    result.update(
        {
            "audit_count": len(items),
            "overall_pass": all(bool(item.get("overall_pass", False)) for item in items),
            "passed_diffusion": all(bool(item.get("passed_diffusion", False)) for item in items),
            "passed_advection": all(bool(item.get("passed_advection", False)) for item in items),
            "passed_combined": all(bool(item.get("passed_combined", False)) for item in items),
            "last_audit": dict(items[-1]),
        }
    )
    return result


def benchmark_configuration(
    *,
    dt_hydro: float,
    benchmark: str = "E1-A",
    screen_identity: Mapping[str, Any] | None = None,
    snapshot_times_s: Sequence[float] | None = None,
    physical_horizons_us: Sequence[float] | None = None,
    n0: float = HR4_N0,
    dtype: str | None = None,
    backend: str | None = None,
    git_sha: str | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the configuration subset guarded by the E1 summarizer."""
    dt = _finite_float(dt_hydro, "dt_hydro")
    benchmark_name = str(benchmark)
    is_e1b = benchmark_name.upper().startswith("E1-B")
    if snapshot_times_s is None:
        snapshot_times_s = E1B_SNAPSHOT_TIMES_S if is_e1b else E1A_SNAPSHOT_TIMES_S
    n0_value = _finite_float(n0, "n0")
    if n0_value <= 1.0:
        raise ValueError("n0 must be greater than one")
    if physical_horizons_us is None:
        physical_horizons_us = E1B_PRIMARY_HORIZONS_US if is_e1b else E1A_PRIMARY_HORIZONS_US
    physical_horizons_us = tuple(
        _finite_float(value, "physical horizon") for value in physical_horizons_us
    )
    if (
        not physical_horizons_us
        or any(value < 0.0 for value in physical_horizons_us)
        or any(b <= a for a, b in zip(physical_horizons_us, physical_horizons_us[1:]))
    ):
        raise ValueError("physical horizons must be non-negative and strictly increasing")
    configuration: dict[str, Any] = {
        "benchmark": benchmark_name,
        "grid": e1a_geometry(),
        "operator": {
            "chi_m2_s": HR4_CHI,
            "nu_m2_s": HR4_NU,
            "gravity_x_m_s2": HR4_GRAVITY_X,
            "gravity_y_m_s2": HR4_GRAVITY_Y,
            "n0": n0_value,
            "cfl_limit": HR4_CFL_LIMIT,
            "advection_scheme": "first_order_upwind",
            "diffusion_scheme": "explicit_central_fd",
            "time_integrator": "explicit_euler",
            "boundary_delta_n": "ambient_dirichlet_zero",
            "boundary_velocity": "open_zero_gradient_outflow_ambient_inflow",
        },
        "dt_hydro_s": dt,
        "snapshot_times_s": [float(value) for value in snapshot_times_s],
        "physical_horizons_s": [float(value) * 1.0e-6 for value in physical_horizons_us],
    }
    if is_e1b:
        if not isinstance(source_metadata, Mapping):
            raise ValueError("E1-B configuration requires immutable source metadata")
        required_source_fields = (
            "source_path",
            "source_manifest_path",
            "source_file_sha256",
            "source_array_sha256",
            "source_state_file_sha256",
            "source_state_array_sha256",
            "source_dtype",
            "source_backend",
            "source_git_sha",
            "screen_identity",
            "source_grid",
            "target_grid",
            "geometry_translation",
        )
        missing_source_fields = [
            key for key in required_source_fields
            if source_metadata.get(key) in (None, "")
        ]
        if missing_source_fields:
            raise ValueError(
                "E1-B configuration source metadata missing: "
                + ", ".join(missing_source_fields)
            )
        configuration["initial_state"] = {
            "kind": "real_hr3b_post_state",
            "source_authority": "HR-3B_POST",
            "source_path": source_metadata.get("source_path"),
            "source_manifest_path": source_metadata.get("source_manifest_path"),
            "source_file_sha256": source_metadata.get("source_file_sha256"),
            "source_array_sha256": source_metadata.get("source_array_sha256"),
            "source_state_file_sha256": source_metadata.get("source_state_file_sha256"),
            "source_state_array_sha256": source_metadata.get("source_state_array_sha256"),
            "source_dtype": source_metadata.get("source_dtype"),
            "source_backend": source_metadata.get("source_backend"),
            "source_git_sha": source_metadata.get("source_git_sha"),
            "screen_identity": source_metadata.get("screen_identity"),
            "source_grid": source_metadata.get("source_grid"),
            "target_grid": source_metadata.get("target_grid"),
            "geometry_translation": source_metadata.get("geometry_translation"),
            "velocity_initialization": "zero",
            "interpolation": False,
        }
    else:
        configuration["initial_condition"] = {
            "kind": "negative_gaussian",
            "amplitude": E1A_AMPLITUDE,
            "sigma_m": E1A_SIGMA_M,
            "center_x_m": E1A_CENTER_X_M,
            "center_y_m": E1A_CENTER_Y_M,
            "velocity_initialization": "zero",
        }
    if screen_identity is not None:
        configuration["screen_identity"] = dict(screen_identity)
    if dtype is not None or backend is not None or git_sha is not None:
        configuration["execution"] = {
            "dtype": dtype,
            "backend": backend,
            "git_sha": git_sha,
        }
    return configuration


def run_timestep_case(
    *,
    dt_hydro: float,
    state: Mapping[str, Any] | None = None,
    benchmark: str = "E1-A",
    screen_identity: Mapping[str, Any] | None = None,
    snapshot_times_s: Sequence[float] | None = None,
    physical_horizons_us: Sequence[float] | None = None,
    chi: float = HR4_CHI,
    nu: float = HR4_NU,
    n0: float = HR4_N0,
    gravity_x: float = HR4_GRAVITY_X,
    gravity_y: float = HR4_GRAVITY_Y,
    cfl_limit: float = HR4_CFL_LIMIT,
    dx: float = HR4_DX,
    dy: float = HR4_DY,
    x_min: float = HR4_X_MIN,
    y_min: float = HR4_Y_MIN,
    x_max: float | None = HR4_X_MAX,
    y_max: float | None = HR4_Y_MAX,
    dtype: Any = np.float64,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance one E1 screen and retain only requested observables.

    The default path is the formal E1-A benchmark.  ``state`` is injectable
    solely for small unit tests and E1-B's immutable screen handoff; it does
    not change the default benchmark definition.
    """
    dt = _finite_float(dt_hydro, "dt_hydro")
    if dt <= 0.0:
        raise ValueError("dt_hydro must be positive")
    benchmark_name = str(benchmark)
    is_e1b = benchmark_name.upper().startswith("E1-B")
    if snapshot_times_s is None:
        snapshot_times_s = E1B_SNAPSHOT_TIMES_S if is_e1b else E1A_SNAPSHOT_TIMES_S
    n0_value = _finite_float(n0, "n0")
    if n0_value <= 1.0:
        raise ValueError("n0 must be greater than one")
    if state is None:
        state = build_e1a_initial_state(dtype=dtype)
    delta_n, vx, vy = _validate_state_triplet(state)
    # Work on private arrays so an E1-B source or a caller's fixture cannot be
    # changed by the operator's boundary copies.
    current = {
        "delta_n": xp.array(delta_n, copy=True),
        "vx": xp.array(vx, copy=True),
        "vy": xp.array(vy, copy=True),
    }
    schedule = build_snapshot_step_schedule(dt, snapshot_times_s)
    started = time.perf_counter()
    audits: list[dict[str, Any]] = []
    initial_audit = audit_hr4_stability(
        dx=dx,
        dy=dy,
        dt_hydro=dt,
        chi=chi,
        nu=nu,
        max_abs_vx=float(to_cpu(xp.max(xp.abs(current["vx"])))),
        max_abs_vy=float(to_cpu(xp.max(xp.abs(current["vy"])))),
        cfl_limit=cfl_limit,
    )
    audits.append(dict(initial_audit))
    snapshots: list[dict[str, Any]] = []
    failure_reason: str | None = None
    status = "PASS"
    completed_steps = 0
    if not bool(initial_audit.get("overall_pass", False)):
        status = "FAIL_STABILITY"
        failure_reason = "initial HR-4 stability audit failed"
    for snapshot_time_s, target_steps in schedule:
        if status != "PASS":
            break
        while completed_steps < target_steps:
            try:
                result = advance_hr4_single_screen(
                    current["delta_n"],
                    current["vx"],
                    current["vy"],
                    dx=dx,
                    dy=dy,
                    dt_hydro=dt,
                    chi=chi,
                    nu=nu,
                    n0=n0_value,
                    gravity_x=gravity_x,
                    gravity_y=gravity_y,
                    cfl_limit=cfl_limit,
                    n_steps=1,
                    require_stable=True,
                    x_min=x_min,
                    y_min=y_min,
                )
            except ValueError as error:
                status = "FAIL_STABILITY" if "stability" in str(error).lower() else "FAIL_PHYSICS"
                failure_reason = f"{type(error).__name__}: {error}"
                break
            # Keep the live state on the active backend.  No host conversion is
            # performed between HR-4B steps.
            current = {name: result[name] for name in ("delta_n", "vx", "vy")}
            audits.append(dict(result["stability"]))
            completed_steps += 1
        if status != "PASS":
            break
        metrics = thermal_channel_metrics(
            current["delta_n"],
            current["vx"],
            current["vy"],
            dx=dx,
            dy=dy,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )
        metrics["time_s"] = float(snapshot_time_s)
        metrics["time_us"] = float(snapshot_time_s * 1.0e6)
        metrics["hydro_step_count"] = int(completed_steps)
        metrics["stability"] = dict(audits[-1])
        metrics.update(classify_boundary_contamination(metrics))
        snapshots.append(metrics)

    # Explicitly audit the state produced by the final successful update.  This
    # catches a boundary/velocity stability violation that might occur after an
    # operator returned its per-step audit and fails closed before qualification.
    if status == "PASS" and completed_steps > 0:
        final_max_abs_vx = float(to_cpu(xp.max(xp.abs(current["vx"]))))
        final_max_abs_vy = float(to_cpu(xp.max(xp.abs(current["vy"]))))
        final_audit = audit_hr4_stability(
            dx=dx,
            dy=dy,
            dt_hydro=dt,
            chi=chi,
            nu=nu,
            max_abs_vx=final_max_abs_vx,
            max_abs_vy=final_max_abs_vy,
            cfl_limit=cfl_limit,
        )
        audits.append(dict(final_audit))
        if snapshots and snapshots[-1]["hydro_step_count"] == completed_steps:
            snapshots[-1]["stability"] = dict(final_audit)
        if not bool(final_audit.get("overall_pass", False)):
            status = "FAIL_STABILITY"
            failure_reason = "final HR-4 stability audit failed"
    elapsed = time.perf_counter() - started
    actual_backend = debug_backend().get("backend", "unknown")
    actual_git_sha = repository_git_sha()
    initial_hash = sha256_array(delta_n)
    config = benchmark_configuration(
        dt_hydro=dt,
        benchmark=benchmark_name,
        screen_identity=screen_identity,
        snapshot_times_s=snapshot_times_s,
        physical_horizons_us=physical_horizons_us,
        n0=n0_value,
        dtype=delta_n.dtype.name,
        backend=actual_backend,
        git_sha=actual_git_sha,
        source_metadata=source_metadata,
    )
    # The operator parameters used by an injected test state are still recorded
    # explicitly, while the default configuration remains frozen E1-A.
    config["operator"].update(
        {
            "chi_m2_s": float(chi),
            "nu_m2_s": float(nu),
            "gravity_x_m_s2": float(gravity_x),
            "gravity_y_m_s2": float(gravity_y),
            "cfl_limit": float(cfl_limit),
        }
    )
    initial_state = {
        "delta_n_sha256": initial_hash,
        "vx_sha256": sha256_array(vx),
        "vy_sha256": sha256_array(vy),
        "shape": list(delta_n.shape),
        "dtype": delta_n.dtype.name,
    }
    if is_e1b and isinstance(source_metadata, Mapping):
        for key in (
            "source_path",
            "source_manifest_path",
            "source_file_sha256",
            "source_array_sha256",
            "source_state_file_sha256",
            "source_state_array_sha256",
            "source_dtype",
            "source_backend",
            "source_git_sha",
            "n0",
            "screen_identity",
            "source_grid",
            "target_grid",
            "geometry_translation",
            "coordinate_relabeling",
        ):
            if key in source_metadata:
                initial_state[key] = source_metadata[key]
        initial_state["velocity_initialization"] = "zero"
    return {
        "schema": E1B_SCHEMA if is_e1b else E1A_SCHEMA,
        "benchmark": benchmark_name,
        "status": status,
        "failure_reason": failure_reason,
        "configuration": config,
        "configuration_sha256": sha256_bytes(_canonical_json(config).encode("ascii")),
        "initial_state_sha256": initial_hash,
        "initial_state": initial_state,
        "backend": actual_backend,
        "dtype": delta_n.dtype.name,
        "snapshot_times_s_requested": [float(item[0]) for item in schedule],
        "snapshots": snapshots,
        "hydro_step_count": int(completed_steps),
        "wall_time_s": float(elapsed),
        "wall_time_s_per_step": float(elapsed / completed_steps) if completed_steps else 0.0,
        "stability": _stability_maximum(audits),
        "slow_time_history_stored": False,
        "git_sha": actual_git_sha,
        "boundary_contamination_criteria": {
            "first_interior_ring_ratio_limit": E1_BOUNDARY_FIRST_RING_RATIO_LIMIT,
            "sigma_y_over_top_clearance_limit": E1_BOUNDARY_SIGMA_Y_TOP_CLEARANCE_LIMIT,
            "formal_edge_ratio_is_diagnostic_only": True,
        },
    }


def run_e1a_case(**kwargs: Any) -> dict[str, Any]:
    """Run the formal E1-A synthetic case."""
    kwargs.setdefault("benchmark", "E1-A")
    return run_timestep_case(**kwargs)


def run_e1b_case(
    screen_path: str | Path,
    *,
    screen_identity: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a timestep case from one immutable HR-3B POST screen."""
    loaded = load_e1b_screen(screen_path)
    kwargs.setdefault("benchmark", "E1-B")
    loaded_identity = loaded["screen_identity"]
    if screen_identity is not None and dict(screen_identity) != dict(loaded_identity):
        raise ValueError("E1-B screen identity does not match immutable source manifest")
    kwargs.setdefault("screen_identity", loaded_identity)
    kwargs.setdefault("source_metadata", loaded)
    kwargs.setdefault("n0", loaded["n0"])
    kwargs.setdefault("snapshot_times_s", E1B_SNAPSHOT_TIMES_S)
    kwargs.setdefault("physical_horizons_us", E1B_PRIMARY_HORIZONS_US)
    result = run_timestep_case(state=loaded, **kwargs)
    return result


def json_safe(value: Any) -> Any:
    """Convert NumPy scalars/arrays and NaN to strict JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value.__class__.__module__.startswith("cupy"):
        return json_safe(to_cpu(value))
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_case_manifest(result: Mapping[str, Any], out_path: str | Path) -> Path:
    """Write one deterministic case JSON manifest."""
    path = Path(out_path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing case manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = json_safe(dict(result))
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(safe, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path


def write_observables_csv(result: Mapping[str, Any], out_path: str | Path) -> Path:
    """Write snapshot observables in deterministic column order."""
    import csv

    path = Path(out_path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing case CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshots = list(result.get("snapshots", ()))
    fields = [
        "time_s",
        "time_us",
        "hydro_step_count",
        "xc_m",
        "yc_m",
        "sigma_x_m",
        "sigma_y_m",
        "min_delta_n",
        "max_delta_n",
        "max_abs_vx_m_s",
        "max_abs_vy_m_s",
        "max_abs_v_m_s",
        "formal_edge_boundary_ratio",
        "first_interior_ring_ratio",
        "top_clearance_m",
        "sigma_y_over_top_clearance",
        "boundary_contaminated",
        "boundary_contamination_reasons",
        "thermal_channel_defined",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for snapshot in snapshots:
            writer.writerow({key: json_safe(snapshot.get(key)) for key in fields})
    return path


__all__ = [
    "E1A_SCHEMA",
    "E1B_SCHEMA",
    "E1A_AMPLITUDE",
    "E1A_SIGMA_M",
    "E1A_SNAPSHOT_TIMES_US",
    "E1A_SNAPSHOT_TIMES_S",
    "E1A_PRIMARY_HORIZONS_US",
    "E1B_SNAPSHOT_TIMES_US",
    "E1B_SNAPSHOT_TIMES_S",
    "E1B_PRIMARY_HORIZONS_US",
    "E1A_SUPPORTED_DT_US",
    "E1A_CENTROID_TOLERANCE_M",
    "E1A_WIDTH_RELATIVE_TOLERANCE",
    "E1A_EXTREME_RELATIVE_TOLERANCE",
    "E1_BOUNDARY_FIRST_RING_RATIO_LIMIT",
    "E1_BOUNDARY_SIGMA_Y_TOP_CLEARANCE_LIMIT",
    "HR4_N0",
    "sha256_bytes",
    "sha256_file",
    "sha256_array",
    "repository_git_sha",
    "e1a_geometry",
    "e1b_source_grid",
    "e1b_geometry_translation",
    "nodal_axes",
    "build_e1a_initial_state",
    "synthetic_initial_state",
    "build_synthetic_initial_state",
    "load_e1b_screen",
    "thermal_channel_metrics",
    "classify_boundary_contamination",
    "build_snapshot_step_schedule",
    "benchmark_configuration",
    "run_timestep_case",
    "run_e1a_case",
    "run_e1b_case",
    "json_safe",
    "write_case_manifest",
    "write_observables_csv",
]
