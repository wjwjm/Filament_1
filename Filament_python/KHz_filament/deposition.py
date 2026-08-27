"""Mechanism-resolved plasma deposition for one optical interval.

The helpers in this module intentionally operate on one ``[Nt, Ny, Nx]``
power source at a time.  They return only the current interval's ``[Ny, Nx]``
map or a scalar energy; no longitudinal stack is allocated here.
"""

from __future__ import annotations

import math

from .device import xp


def _require_finite_nonnegative_dt(dt: float) -> float:
    value = float(dt)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("dt must be a finite non-negative number")
    return value


def _power_volume(power):
    source = xp.asarray(power)
    if source.ndim != 3:
        raise ValueError("deposition power source must have shape [Nt, Ny, Nx]")
    # Deposition channels represent positive energy transfer.  Keep the
    # reduction finite and non-negative without retaining the 3-D source.
    return xp.maximum(
        xp.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
    )


def integrate_power_to_q(power, dt: float):
    """Time-integrate one power source into an interval-average ``q`` map.

    The returned map has shape ``[Ny, Nx]`` and units J/m^3.  The reduction
    deliberately follows the existing ``sum(axis=0) * dt`` convention.
    """
    dt_value = _require_finite_nonnegative_dt(dt)
    source = _power_volume(power)
    return xp.sum(source, axis=0) * dt_value


def q_ion_from_power(photoionization_energy_rate, dt: float):
    """Return ``q_ion`` from the authoritative photoionization power source."""
    return integrate_power_to_q(photoionization_energy_rate, dt)


def q_ib_from_power(alpha_ib, intensity, dt: float):
    """Return ``q_ib`` from ``alpha_ib * intensity`` for one interval."""
    alpha = xp.asarray(alpha_ib)
    optical_intensity = xp.asarray(intensity)
    try:
        source = alpha * optical_intensity
    except ValueError as exc:
        raise ValueError("alpha_ib and intensity must be broadcast-compatible") from exc
    return integrate_power_to_q(source, dt)


def q_raman_from_actual_fluence_loss(actual_local_fluence_loss, dz: float):
    """Convert one actual Raman fluence-loss map to ``q_raman``.

    ``actual_local_fluence_loss`` is the field-difference result for one
    longitudinal interval and has units J/m^2.  Raman deposition represents a
    positive medium energy gain, so non-finite values are cleared and only the
    positive part is retained before dividing by the interval length.
    """
    dz_value = float(dz)
    if not math.isfinite(dz_value) or dz_value <= 0.0:
        raise ValueError("dz must be a finite positive number")
    loss = xp.asarray(actual_local_fluence_loss)
    if loss.ndim != 2:
        raise ValueError("actual Raman fluence-loss map must have shape [Ny, Nx]")
    loss = xp.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    return xp.maximum(loss, 0.0) / dz_value


def interval_energy_from_q(q, dx: float, dy: float, dz: float) -> float:
    """Convert one ``q`` map to interval energy using ``sum(q)*dx*dy*dz``."""
    q_map = xp.asarray(q)
    if q_map.ndim != 2:
        raise ValueError("q deposition map must have shape [Ny, Nx]")
    cell_volume = float(dx) * float(dy) * float(dz)
    if not math.isfinite(cell_volume) or cell_volume < 0.0:
        raise ValueError("dx, dy, and dz must define a finite non-negative volume")
    return float(xp.sum(q_map) * cell_volume)


def direct_interval_energy(power, dt: float, dx: float, dy: float, dz: float) -> float:
    """Independently integrate a 3-D source directly to interval energy.

    This is the Level-1 closure reference for ``interval_energy_from_q``;
    importantly, it does not reuse the already reduced ``q`` map.
    """
    dt_value = _require_finite_nonnegative_dt(dt)
    source = _power_volume(power)
    cell_volume = float(dx) * float(dy) * float(dz)
    if not math.isfinite(cell_volume) or cell_volume < 0.0:
        raise ValueError("dx, dy, and dz must define a finite non-negative volume")
    return float(xp.sum(source) * dt_value * cell_volume)


__all__ = [
    "direct_interval_energy",
    "integrate_power_to_q",
    "interval_energy_from_q",
    "q_ib_from_power",
    "q_ion_from_power",
    "q_raman_from_actual_fluence_loss",
]
