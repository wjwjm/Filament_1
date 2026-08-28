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


def q_raman_from_target_fluence_gain(target_local_fluence_gain, dz: float):
    """Return local Raman medium deposition from Eq.10/Heun target gain.

    ``target_local_fluence_gain`` is the positive rotational medium-energy
    gain accumulated by the full Raman operator over one longitudinal
    interval.  It is deliberately distinct from a signed before/after field
    difference, which can include conservative local redistribution.
    """
    dz_value = float(dz)
    if not math.isfinite(dz_value) or dz_value <= 0.0:
        raise ValueError("dz must be a finite positive number")
    gain = xp.asarray(target_local_fluence_gain)
    if gain.ndim != 2:
        raise ValueError("target Raman fluence-gain map must have shape [Ny, Nx]")
    gain = xp.nan_to_num(gain, nan=0.0, posinf=0.0, neginf=0.0)
    return xp.maximum(gain, 0.0) / dz_value


def interval_energy_from_fluence_gain(fluence_gain, dx: float, dy: float) -> float:
    """Direct scalar reduction of one positive Raman fluence-gain map."""
    gain = xp.asarray(fluence_gain)
    if gain.ndim != 2:
        raise ValueError("fluence-gain map must have shape [Ny, Nx]")
    area = float(dx) * float(dy)
    if not math.isfinite(area) or area < 0.0:
        raise ValueError("dx and dy must define a finite non-negative area")
    gain = xp.nan_to_num(gain, nan=0.0, posinf=0.0, neginf=0.0)
    return float(xp.sum(xp.maximum(gain, 0.0)) * area)


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


def build_unified_deposition_ledger(
    *,
    ion_interval_J,
    ib_interval_J,
    raman_interval_J,
    ion_interval_reference_J,
    ib_interval_reference_J,
    raman_interval_reference_J,
    ion_pulse_J: float,
    ib_pulse_J: float,
    raman_pulse_J: float,
    ion_configured: bool,
    ib_configured: bool,
    raman_configured: bool,
    raman_authoritative: bool,
    raman_source: str,
    ionization_feedback_enabled: bool,
    raman_feedback_enabled: bool,
    field_in_J: float,
    field_out_J: float,
    raman_operator_relative_residuals=(),
    raman_operator_cumulative_relative_residual: float | None = None,
):
    """Build the HR-2D scalar unified deposition and field-bookkeeping ledger.

    Inputs are already-reduced canonical interval energies.  This helper is
    deliberately scalar-only: it never creates a longitudinal q/map payload
    and never derives deposition from the optical field-energy difference.
    """
    rtol = 2.0e-5
    atol = 1.0e-30
    raman_operator_step_p99_rtol = 1.0e-3
    raman_operator_cumulative_rtol = 5.0e-3

    def _ledger(values, name):
        result = tuple(float(value) for value in values)
        if not all(math.isfinite(value) for value in result):
            raise ValueError(f"{name} must contain only finite scalar energies")
        return result

    def _same_length(*ledgers):
        lengths = {len(ledger) for ledger in ledgers}
        if len(lengths) != 1:
            raise ValueError("canonical deposition interval ledgers must share one length")

    def _allclose(observed, reference):
        return all(
            abs(value - target) <= atol + rtol * max(abs(value), abs(target))
            for value, target in zip(observed, reference)
        )

    def _scalar_close(observed, reference):
        return abs(observed - reference) <= atol + rtol * max(
            abs(observed), abs(reference)
        )

    def _p99(values):
        ordered = sorted(float(value) for value in values)
        if not ordered:
            return math.nan
        index = 0.99 * (len(ordered) - 1)
        lower = int(math.floor(index))
        upper = int(math.ceil(index))
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    ion = _ledger(ion_interval_J, "ion interval ledger")
    ib = _ledger(ib_interval_J, "IB interval ledger")
    raman = _ledger(raman_interval_J, "Raman interval ledger")
    ion_reference = _ledger(ion_interval_reference_J, "ion interval reference")
    ib_reference = _ledger(ib_interval_reference_J, "IB interval reference")
    raman_reference = _ledger(
        raman_interval_reference_J, "Raman interval reference"
    )
    raman_operator_relative = _ledger(
        raman_operator_relative_residuals,
        "Raman operator relative residuals",
    )
    _same_length(ion, ib, raman, ion_reference, ib_reference, raman_reference)
    if raman_operator_relative and len(raman_operator_relative) != len(raman):
        raise ValueError("Raman operator residuals must match the interval ledger")

    mechanisms = {
        "ion": {
            "configured": bool(ion_configured),
            "active": bool(ion_configured),
            "authoritative": True,
            "source": (
                "positive_photoionization_energy_rate"
                if ion_configured else "off"
            ),
            "feedback_applied": bool(ionization_feedback_enabled),
        },
        "ib": {
            "configured": bool(ib_configured),
            "active": bool(ib_configured),
            "authoritative": True,
            "source": "alpha_ib_times_intensity" if ib_configured else "off",
            "feedback_applied": True,
        },
        "raman": {
            "configured": bool(raman_configured),
            "active": bool(raman_configured),
            "authoritative": bool(raman_authoritative),
            "source": str(raman_source),
            "feedback_applied": bool(raman_feedback_enabled),
        },
    }

    for status in mechanisms.values():
        status["interval_ledger_available"] = True
        status["pulse_scalar_available"] = True

    def _level1_status(name, observed, reference):
        status = mechanisms[name]
        if not status["active"]:
            return "not_applicable"
        if not status["authoritative"]:
            return "unavailable"
        return "pass" if _allclose(observed, reference) else "failed"

    def _level2_status(name, observed, interval):
        status = mechanisms[name]
        if not status["active"]:
            return "not_applicable"
        if not status["authoritative"]:
            return "unavailable"
        return "pass" if _scalar_close(observed, sum(interval)) else "failed"

    level1 = {
        "ion": _level1_status("ion", ion, ion_reference),
        "ib": _level1_status("ib", ib, ib_reference),
        "raman": _level1_status("raman", raman, raman_reference),
    }
    level2 = {
        "ion": _level2_status("ion", float(ion_pulse_J), ion),
        "ib": _level2_status("ib", float(ib_pulse_J), ib),
        "raman": _level2_status("raman", float(raman_pulse_J), raman),
    }
    raman_operator_step_p99 = _p99(raman_operator_relative)
    raman_operator_cumulative = (
        float(raman_operator_cumulative_relative_residual)
        if raman_operator_cumulative_relative_residual is not None
        else math.nan
    )
    raman_status = mechanisms["raman"]
    if not raman_status["active"]:
        raman_operator_status = "not_applicable"
    elif not raman_status["authoritative"]:
        raman_operator_status = "unavailable"
    elif not raman_status["feedback_applied"]:
        raman_operator_status = "not_applicable"
    elif not raman_operator_relative or not math.isfinite(raman_operator_cumulative):
        raman_operator_status = "unavailable"
    elif (
        raman_operator_step_p99 <= raman_operator_step_p99_rtol
        and raman_operator_cumulative <= raman_operator_cumulative_rtol
    ):
        raman_operator_status = "pass"
    else:
        raman_operator_status = "failed"
    for name in mechanisms:
        mechanisms[name]["level1_closure_status"] = level1[name]
        mechanisms[name]["level2_closure_status"] = level2[name]
    raman_status["deposition_reduction_closure_status"] = level1["raman"]
    raman_status["operator_energy_closure_status"] = raman_operator_status
    raman_status["operator_energy_step_p99"] = raman_operator_step_p99
    raman_status["operator_energy_cumulative_relative"] = raman_operator_cumulative

    required = [
        status for status in mechanisms.values() if status["active"]
    ]
    total_authoritative = all(status["authoritative"] for status in required)
    unavailable = [
        f"{name}:{status['source']}"
        for name, status in mechanisms.items()
        if status["active"] and not status["authoritative"]
    ]
    if total_authoritative:
        total_interval = tuple(
            ion_value + ib_value + raman_value
            for ion_value, ib_value, raman_value in zip(ion, ib, raman)
        )
        total_pulse = sum(total_interval)
        total_level2_status = "pass" if _scalar_close(
            total_pulse,
            float(ion_pulse_J) + float(ib_pulse_J) + float(raman_pulse_J),
        ) else "failed"
    else:
        total_interval = tuple(math.nan for _ in ion)
        total_pulse = math.nan
        total_level2_status = "unavailable"

    all_available_level1_pass = all(
        value in {"pass", "not_applicable", "unavailable"}
        for value in level1.values()
    ) and not any(value == "failed" for value in level1.values())
    all_available_level2_pass = all(
        value in {"pass", "not_applicable", "unavailable"}
        for value in level2.values()
    ) and not any(value == "failed" for value in level2.values())

    accounted_authoritative = (
        float(ion_pulse_J)
        + float(ib_pulse_J)
        + (float(raman_pulse_J) if raman_authoritative else 0.0)
    )
    field_in = float(field_in_J)
    field_out = float(field_out_J)
    field_values_finite = math.isfinite(field_in) and math.isfinite(field_out)
    field_loss = field_in - field_out if field_values_finite else math.nan
    residual = field_loss - accounted_authoritative if field_values_finite else math.nan
    denominator = max(
        abs(field_loss) if field_values_finite else 0.0,
        abs(accounted_authoritative),
        abs(field_in) * 1.0e-15 if field_values_finite else 0.0,
        1.0e-300,
    )
    relative_residual = residual / denominator if field_values_finite else math.nan

    return {
        "mechanisms": mechanisms,
        "level1": level1,
        "level2": level2,
        "level1_all_available_pass": bool(all_available_level1_pass),
        "level2_all_available_pass": bool(all_available_level2_pass),
        "total_authoritative": bool(total_authoritative),
        "total_unavailable_reason": ";".join(unavailable) if unavailable else "",
        "total_interval_J": total_interval,
        "total_pulse_J": float(total_pulse),
        "total_level2_status": total_level2_status,
        "accounted_authoritative_J": float(accounted_authoritative),
        "field_in_J": field_in,
        "field_out_J": field_out,
        "field_loss_J": float(field_loss),
        "field_residual_J": float(residual),
        "field_relative_residual": float(relative_residual),
        "field_bookkeeping_authoritative": bool(
            total_authoritative and field_values_finite
        ),
        "closure_relative_tolerance": rtol,
        "raman_operator_energy_closure_status": raman_operator_status,
        "raman_operator_energy_step_p99": raman_operator_step_p99,
        "raman_operator_energy_cumulative_relative": raman_operator_cumulative,
        "raman_operator_energy_step_p99_tolerance": raman_operator_step_p99_rtol,
        "raman_operator_energy_cumulative_tolerance": raman_operator_cumulative_rtol,
    }


__all__ = [
    "direct_interval_energy",
    "build_unified_deposition_ledger",
    "interval_energy_from_fluence_gain",
    "integrate_power_to_q",
    "interval_energy_from_q",
    "q_ib_from_power",
    "q_ion_from_power",
    "q_raman_from_actual_fluence_loss",
    "q_raman_from_target_fluence_gain",
]
