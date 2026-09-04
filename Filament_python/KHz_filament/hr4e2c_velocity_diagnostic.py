"""Report-only helpers for the HR-4E-2C velocity-field diagnostic.

This module deliberately has no GPU, propagation, or source-state construction
dependency.  It only analyses already-persisted NumPy arrays.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def nested_node_indices(coarse: np.ndarray, fine: np.ndarray, *, atol: float = 1.0e-15) -> np.ndarray:
    """Return exact factor-of-two nested-node indices, or raise clearly."""
    coarse = np.asarray(coarse, dtype=np.float64)
    fine = np.asarray(fine, dtype=np.float64)
    if coarse.ndim != 1 or fine.ndim != 1 or coarse.size < 2 or fine.size < 2:
        raise ValueError("coordinates must be one-dimensional nodal arrays")
    if (fine.size - 1) != 2 * (coarse.size - 1):
        raise ValueError("grids are not factor-of-two inclusive nodal grids")
    indices = np.arange(coarse.size, dtype=np.int64) * 2
    if not np.allclose(fine[indices], coarse, rtol=0.0, atol=atol):
        raise ValueError("fine-grid nodes do not exactly align with coarse-grid nodes")
    return indices


def restrict_nested_field(fine_field: np.ndarray, y_indices: np.ndarray, x_indices: np.ndarray) -> np.ndarray:
    """Restrict a fine (y, x) field by index selection without interpolation."""
    fine_field = np.asarray(fine_field, dtype=np.float64)
    if fine_field.ndim != 2:
        raise ValueError("field must be two dimensional")
    return fine_field[np.ix_(y_indices, x_indices)]


def _relative(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else None
    return numerator / denominator


def field_norms(coarse: np.ndarray, fine_restricted: np.ndarray, *, dA: float) -> dict[str, float | None]:
    """Compute the requested area-weighted relative field norms."""
    coarse = np.asarray(coarse, dtype=np.float64)
    fine_restricted = np.asarray(fine_restricted, dtype=np.float64)
    if coarse.shape != fine_restricted.shape or coarse.ndim != 2:
        raise ValueError("coarse and restricted fine fields must be matching 2-D arrays")
    if not (math.isfinite(dA) and dA > 0.0):
        raise ValueError("dA must be finite and positive")
    difference = fine_restricted - coarse
    l1_abs = float(np.sum(np.abs(difference)) * dA)
    l2_abs = float(math.sqrt(float(np.sum(difference * difference) * dA)))
    linf_abs = float(np.max(np.abs(difference)))
    l1_den = float(np.sum(np.abs(fine_restricted)) * dA)
    l2_den = float(math.sqrt(float(np.sum(fine_restricted * fine_restricted) * dA)))
    linf_den = float(np.max(np.abs(fine_restricted)))
    return {
        "E_L1": _relative(l1_abs, l1_den), "E_L2": _relative(l2_abs, l2_den),
        "E_Linf": linf_abs, "E_Linf_rel": _relative(linf_abs, linf_den),
        "L1_abs": l1_abs, "L2_abs": l2_abs,
    }


def velocity_energy(field: np.ndarray, *, dA: float) -> float:
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 2 or not (math.isfinite(dA) and dA > 0.0):
        raise ValueError("invalid field or dA")
    return float(np.sum(field * field) * dA)


def plume_weighted_velocity(vy: np.ndarray, delta_n: np.ndarray, *, dA: float) -> float | None:
    vy = np.asarray(vy, dtype=np.float64)
    delta_n = np.asarray(delta_n, dtype=np.float64)
    if vy.shape != delta_n.shape:
        raise ValueError("vy and delta_n must have matching shapes")
    weight = np.maximum(-delta_n, 0.0)
    denominator = float(np.sum(weight) * dA)
    return None if denominator == 0.0 else float(np.sum(weight * vy) * dA / denominator)


def peak_audit(vy: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Describe the raw discrete maximum of |vy| and its 3 by 3 neighborhood."""
    vy = np.asarray(vy, dtype=np.float64)
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if vy.shape != (y.size, x.size):
        raise ValueError("field shape does not match y/x coordinates")
    iy, ix = (int(item) for item in np.unravel_index(int(np.argmax(np.abs(vy))), vy.shape))
    boundary = iy in {0, vy.shape[0] - 1} or ix in {0, vy.shape[1] - 1}
    neighborhood: list[list[float]] | None = None
    if not boundary:
        neighborhood = np.abs(vy[iy - 1:iy + 2, ix - 1:ix + 2]).tolist()
    return {
        "raw_max_abs_vy": float(abs(vy[iy, ix])), "raw_vy": float(vy[iy, ix]),
        "index_yx": [iy, ix], "x_m": float(x[ix]), "y_m": float(y[iy]),
        "interior": not boundary, "neighborhood_abs_vy_3x3": neighborhood,
    }


def quadratic_subgrid_peak(vy: np.ndarray, x: np.ndarray, y: np.ndarray, peak: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fit |vy| on the local 3 by 3 stencil; never extrapolate beyond one cell."""
    peak = dict(peak or peak_audit(vy, x, y))
    if not peak["interior"]:
        return {"status": "SUBGRID_PEAK_INVALID", "reason": "boundary_peak"}
    iy, ix = (int(value) for value in peak["index_yx"])
    vy, x, y = np.asarray(vy, dtype=np.float64), np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    xx, yy = np.meshgrid(x[ix - 1:ix + 2] - x[ix], y[iy - 1:iy + 2] - y[iy], indexing="xy")
    zz = np.abs(vy[iy - 1:iy + 2, ix - 1:ix + 2])
    design = np.column_stack((xx.ravel() ** 2, yy.ravel() ** 2, xx.ravel() * yy.ravel(), xx.ravel(), yy.ravel(), np.ones(9)))
    coeff, _, rank, _ = np.linalg.lstsq(design, zz.ravel(), rcond=None)
    if rank != 6 or not np.all(np.isfinite(coeff)):
        return {"status": "SUBGRID_PEAK_INVALID", "reason": "rank_or_nonfinite_fit"}
    a, b, c, d, e, f = (float(value) for value in coeff)
    hessian = np.array(((2.0 * a, c), (c, 2.0 * b)), dtype=np.float64)
    if not np.all(np.linalg.eigvalsh(hessian) < 0.0):
        return {"status": "SUBGRID_PEAK_INVALID", "reason": "not_local_maximum"}
    try:
        local = np.linalg.solve(hessian, -np.array((d, e), dtype=np.float64))
    except np.linalg.LinAlgError:
        return {"status": "SUBGRID_PEAK_INVALID", "reason": "singular_hessian"}
    if not np.all(np.isfinite(local)) or abs(local[0]) > abs(dx) or abs(local[1]) > abs(dy):
        return {"status": "SUBGRID_PEAK_INVALID", "reason": "stationary_point_outside_central_neighborhood"}
    x0, y0 = (float(value) for value in local)
    value = float(a * x0 * x0 + b * y0 * y0 + c * x0 * y0 + d * x0 + e * y0 + f)
    if not math.isfinite(value):
        return {"status": "SUBGRID_PEAK_INVALID", "reason": "nonfinite_reconstruction"}
    return {
        "status": "VALID", "reconstructed_max_abs_vy": value,
        "x_m": float(x[ix] + x0), "y_m": float(y[iy] + y0),
        "offset_x_m": x0, "offset_y_m": y0,
        "raw_minus_reconstructed": float(peak["raw_max_abs_vy"] - value),
    }


def convergence_order(coarse_error: float | None, fine_error: float | None) -> float | None:
    if coarse_error is None or fine_error is None or coarse_error <= 0.0 or fine_error <= 0.0:
        return None
    return math.log2(coarse_error / fine_error)


def classify_velocity_screen(*, norms_20_10: Mapping[str, Any], norms_10_5: Mapping[str, Any],
                              energy_20: float, energy_10: float, energy_5: float,
                              raw_peaks: Sequence[Mapping[str, Any]], subgrid_peaks: Sequence[Mapping[str, Any]]) -> str:
    """Classify without introducing a new production tolerance.

    V4 is reserved for exact-zero field differences.  All other decisions use
    refinement direction and whether the local reconstruction repairs the raw
    maximum trend; ambiguous inputs remain V3 rather than being promoted.
    """
    l1_20, l1_10 = norms_20_10.get("E_L1"), norms_10_5.get("E_L1")
    l2_20, l2_10 = norms_20_10.get("E_L2"), norms_10_5.get("E_L2")
    linf_20, linf_10 = norms_20_10.get("E_Linf_rel"), norms_10_5.get("E_Linf_rel")
    if all(value == 0.0 for value in (l1_20, l1_10, l2_20, l2_10, linf_20, linf_10)):
        return "V4_NUMERICALLY_NEGLIGIBLE_FIELD_DIFFERENCE"
    field_improves = all(isinstance(a, (float, int)) and isinstance(b, (float, int)) and b < a for a, b in ((l1_20, l1_10), (l2_20, l2_10)))
    ev_20_10, ev_10_5 = abs(energy_20 - energy_10), abs(energy_10 - energy_5)
    energy_improves = ev_10_5 < ev_20_10 or ev_20_10 == ev_10_5 == 0.0
    raw_values = [float(item["raw_max_abs_vy"]) for item in raw_peaks]
    valid_subgrid = [item for item in subgrid_peaks if item.get("status") == "VALID"]
    raw_nonmonotonic = not (abs(raw_values[1] - raw_values[2]) < abs(raw_values[0] - raw_values[1]))
    reconstructed_regular = False
    if len(valid_subgrid) == 3:
        values = [float(item["reconstructed_max_abs_vy"]) for item in valid_subgrid]
        reconstructed_regular = abs(values[1] - values[2]) < abs(values[0] - values[1])
    if field_improves and energy_improves and raw_nonmonotonic and reconstructed_regular:
        return "V1_PEAK_SAMPLING_DOMINATED"
    if not field_improves and not energy_improves and not reconstructed_regular:
        return "V2_FIELD_NONCONVERGENCE"
    return "V3_MIXED"
