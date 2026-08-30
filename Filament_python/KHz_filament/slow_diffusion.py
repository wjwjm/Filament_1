"""HR-3C-A transverse interpulse diffusion for one ``delta_n_th`` slice."""

from __future__ import annotations

import math

from .device import xp


DEFAULT_EDGE_CONTAMINATION_THRESHOLD = 1.0e-3


def validate_hr3c_parameters(*, D_th: float, f_rep: float) -> float:
    """Validate authoritative HR-3C inputs and return ``dt_interpulse`` in s."""
    values = {"D_th": D_th, "f_rep": f_rep}
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise ValueError("HR-3C D_th and f_rep must be finite")
    if float(D_th) <= 0.0 or float(f_rep) <= 0.0:
        raise ValueError("HR-3C D_th and f_rep must be positive")
    return 1.0 / float(f_rep)


def _finite_2d_real_map(value, name: str):
    result = xp.asarray(value)
    if result.ndim != 2 or result.dtype.kind != "f":
        raise ValueError(f"{name} must be a real floating-point [Ny, Nx] map")
    if not bool(xp.all(xp.isfinite(result))):
        raise ValueError(f"{name} must be finite")
    return result


def build_diffusion_kernel(kperp2, *, D_th: float, f_rep: float):
    """Build ``exp(-D_th*k_perp^2/f_rep)`` on the existing transverse grid."""
    dt_interpulse = validate_hr3c_parameters(D_th=D_th, f_rep=f_rep)
    k2 = _finite_2d_real_map(kperp2, "kperp2")
    if bool(xp.any(k2 < 0.0)):
        raise ValueError("kperp2 must be non-negative")

    kernel = xp.exp(-float(D_th) * k2 * dt_interpulse)
    # A positive lower clamp preserves the exact-damping intent while enforcing
    # the frozen finite, 0 < G <= 1 kernel contract after floating underflow.
    kernel = xp.maximum(kernel, xp.finfo(kernel.dtype).tiny)
    kernel = xp.where(k2 == 0.0, 1.0, kernel)
    if not bool(xp.all(xp.isfinite(kernel))) or not bool(xp.all(kernel > 0.0)) or not bool(xp.all(kernel <= 1.0)):
        raise ValueError("HR-3C diffusion kernel violates the finite 0 < G <= 1 contract")
    return kernel


def evaluate_edge_contamination(delta_n_th, *, edge_width: int = 1) -> dict[str, float]:
    """Return the periodic-boundary contamination metric for one 2-D state slice."""
    state = _finite_2d_real_map(delta_n_th, "delta_n_th")
    width = int(edge_width)
    if width <= 0 or 2 * width > min(state.shape):
        raise ValueError("edge_width must fit within the [Ny, Nx] state slice")
    amplitude = xp.abs(state)
    global_max = float(xp.max(amplitude))
    if global_max == 0.0:
        return {"R_edge": 0.0, "edge_max_abs": 0.0, "global_max_abs": 0.0}
    boundary_max = float(xp.max(xp.concatenate((
        amplitude[:width, :].ravel(), amplitude[-width:, :].ravel(),
        amplitude[width:-width, :width].ravel(), amplitude[width:-width, -width:].ravel(),
    ))))
    return {
        "R_edge": boundary_max / global_max,
        "edge_max_abs": boundary_max,
        "global_max_abs": global_max,
    }


def _roundoff_tolerance(state) -> float:
    scale = max(float(xp.max(xp.abs(state))), 1.0e-30)
    return 128.0 * float(xp.finfo(state.dtype).eps) * scale


def diffuse_interval_2d(
    delta_n_th,
    *,
    kperp2,
    D_th: float,
    f_rep: float,
    edge_threshold: float | None = DEFAULT_EDGE_CONTAMINATION_THRESHOLD,
):
    """Diffuse one interval slice over exactly ``dt_interpulse = 1/f_rep``.

    This operator owns no disk-backed lifecycle and never materializes a
    ``[K, Ny, Nx]`` volume.  The default edge gate is fail-closed for the
    periodic spectral boundary; analytical unit tests may explicitly disable
    it with ``edge_threshold=None``.
    """
    state = _finite_2d_real_map(delta_n_th, "delta_n_th")
    kernel = build_diffusion_kernel(kperp2, D_th=D_th, f_rep=f_rep)
    if kernel.shape != state.shape:
        raise ValueError("kperp2 shape must match the [Ny, Nx] state slice")

    evolved = xp.real(xp.fft.ifft2(xp.fft.fft2(state) * kernel)).astype(state.dtype, copy=False)
    if not bool(xp.all(xp.isfinite(evolved))):
        raise ValueError("HR-3C diffusion produced non-finite values")
    roundoff_tolerance = _roundoff_tolerance(state)
    if float(xp.max(state)) <= roundoff_tolerance and float(xp.max(evolved)) > roundoff_tolerance:
        raise ValueError("HR-3C diffusion produced a significant positive thermal-index channel")

    if edge_threshold is not None:
        threshold = float(edge_threshold)
        if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
            raise ValueError("edge_threshold must be finite and satisfy 0 <= threshold <= 1")
        edge = evaluate_edge_contamination(evolved)
        if edge["R_edge"] > threshold:
            raise ValueError(
                "HR-3C edge-contamination gate failed: "
                f"R_edge={edge['R_edge']:.6e} exceeds {threshold:.6e}"
            )
    return evolved


__all__ = [
    "DEFAULT_EDGE_CONTAMINATION_THRESHOLD",
    "build_diffusion_kernel",
    "diffuse_interval_2d",
    "evaluate_edge_contamination",
    "validate_hr3c_parameters",
]
