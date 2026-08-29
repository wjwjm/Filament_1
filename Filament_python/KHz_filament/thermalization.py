"""HR-3A microscopic-thermalization ledger.

The module is intentionally downstream of HR-2 deposition: it accepts only
authoritative interval-average deposition maps and never inspects optical field
loss, electron-density changes, or legacy slow-heat diagnostics.
"""

from __future__ import annotations

import math

import numpy as np


THERMALIZATION_CHANNELS = ("ion", "ib", "raman")
_RTOL = 2.0e-5
_ATOL = 1.0e-30


def _maps(value, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is required")
    result = np.asarray(value)
    if result.ndim != 3:
        raise ValueError(f"{name} must have shape [K, Ny, Nx]")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(result < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return result


def _geometry(z_edges, dz_intervals, dx: float, dy: float, count: int) -> np.ndarray:
    edges = np.asarray(z_edges, dtype=np.float64)
    dz = np.asarray(dz_intervals, dtype=np.float64)
    if edges.ndim != 1 or dz.ndim != 1 or len(edges) != count + 1 or len(dz) != count:
        raise ValueError("thermalization schedule does not match interval maps")
    if not np.all(np.isfinite(edges)) or not np.all(np.isfinite(dz)):
        raise ValueError("thermalization schedule must be finite")
    if np.any(dz <= 0.0) or not np.allclose(np.diff(edges), dz, rtol=1e-12, atol=1e-12):
        raise ValueError("thermalization schedule has invalid interval geometry")
    area = float(dx) * float(dy)
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("dx and dy must define a finite positive area")
    return dz * area


def _mechanisms(mechanisms: dict[str, dict[str, object]]) -> tuple[dict[str, dict[str, object]], list[str]]:
    if not isinstance(mechanisms, dict):
        raise TypeError("deposition mechanism metadata must be a dictionary")
    copied: dict[str, dict[str, object]] = {}
    unavailable: list[str] = []
    for channel in THERMALIZATION_CHANNELS:
        source = mechanisms.get(channel)
        if not isinstance(source, dict):
            raise ValueError(f"missing deposition mechanism metadata for {channel}")
        active = bool(source.get("active", False))
        authoritative = bool(source.get("authoritative", False))
        copied[channel] = {
            "active": active,
            "authoritative": authoritative,
            "source": str(source.get("source", "missing")),
        }
        if active and not authoritative:
            unavailable.append(f"{channel}:{copied[channel]['source']}")
    return copied, unavailable


def _unavailable(shape, mechanisms, reason: str) -> dict[str, object]:
    nan = np.full(shape, np.nan, dtype=np.float64)
    return {
        "authoritative": False,
        "unavailable_reason": reason,
        "mechanisms": mechanisms,
        "q_th_ion": nan.copy(),
        "q_th_ib": nan.copy(),
        "q_th_raman": nan.copy(),
        "q_thermal": nan.copy(),
        "E_th_ion_interval_J": np.full(shape[0], np.nan),
        "E_th_ib_interval_J": np.full(shape[0], np.nan),
        "E_th_raman_interval_J": np.full(shape[0], np.nan),
        "E_thermal_interval_J": np.full(shape[0], np.nan),
        "E_th_ion_pulse_J": math.nan,
        "E_th_ib_pulse_J": math.nan,
        "E_th_raman_pulse_J": math.nan,
        "E_thermal_pulse_J": math.nan,
        "level_t1": {channel: "unavailable" for channel in THERMALIZATION_CHANNELS},
        "level_t2": {channel: "unavailable" for channel in THERMALIZATION_CHANNELS},
        "level_t3": "unavailable",
        "zero_channel_pass": False,
    }


def build_complete_thermalization_ledger(
    *,
    q_ion,
    q_ib,
    q_raman,
    z_edges,
    dz_intervals,
    dx: float,
    dy: float,
    deposition_mechanisms: dict[str, dict[str, object]],
    deposition_interval_J: dict[str, object],
) -> dict[str, object]:
    """Build the HR-3A complete-microscopic-thermalization ledger.

    The identity conversion is deliberate: it preserves the physical distinction
    between deposition and thermalized heat while implementing the reference
    model's complete eventual-thermalization approximation.  Non-authoritative
    active input is represented as unavailable, never replaced by a legacy
    estimator.
    """
    ion = _maps(q_ion, "q_ion")
    ib = _maps(q_ib, "q_ib")
    raman = _maps(q_raman, "q_raman")
    if ion.shape != ib.shape or ion.shape != raman.shape:
        raise ValueError("thermalization channel maps must have identical shapes")
    cell_measure = _geometry(z_edges, dz_intervals, dx, dy, ion.shape[0])
    mechanisms, unavailable = _mechanisms(deposition_mechanisms)
    for channel, values in (("ion", ion), ("ib", ib), ("raman", raman)):
        if not mechanisms[channel]["active"] and not np.array_equal(values, np.zeros_like(values)):
            raise ValueError(f"inactive {channel} deposition channel must be exact zero")
    if unavailable:
        return _unavailable(ion.shape, mechanisms, ";".join(unavailable))

    source = {"ion": ion, "ib": ib, "raman": raman}
    thermal = {channel: values.copy() for channel, values in source.items()}
    q_total = thermal["ion"] + thermal["ib"] + thermal["raman"]
    reference = {}
    for channel in THERMALIZATION_CHANNELS:
        if channel not in deposition_interval_J:
            raise ValueError(f"missing deposition interval reduction for {channel}")
        value = np.asarray(deposition_interval_J[channel], dtype=np.float64)
        if value.shape != (ion.shape[0],) or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid deposition interval reduction for {channel}")
        reference[channel] = value
    energies = {
        channel: np.sum(values, axis=(1, 2), dtype=np.float64) * cell_measure
        for channel, values in thermal.items()
    }
    total_energy = energies["ion"] + energies["ib"] + energies["raman"]
    level_t1 = {
        channel: "pass" if np.array_equal(thermal[channel], source[channel]) else "failed"
        for channel in THERMALIZATION_CHANNELS
    }
    level_t2 = {
        channel: "pass" if np.allclose(energies[channel], reference[channel], rtol=_RTOL, atol=_ATOL) else "failed"
        for channel in THERMALIZATION_CHANNELS
    }
    t3_maps = np.array_equal(q_total, thermal["ion"] + thermal["ib"] + thermal["raman"])
    t3_energy = np.allclose(total_energy, energies["ion"] + energies["ib"] + energies["raman"], rtol=_RTOL, atol=_ATOL)
    zero_channel_pass = all(
        np.array_equal(thermal[channel], np.zeros_like(thermal[channel]))
        for channel in THERMALIZATION_CHANNELS
        if not mechanisms[channel]["active"]
    )
    closure_pass = (
        all(status == "pass" for status in level_t1.values())
        and all(status == "pass" for status in level_t2.values())
        and t3_maps
        and t3_energy
        and zero_channel_pass
    )
    return {
        "authoritative": bool(closure_pass),
        "unavailable_reason": "" if closure_pass else "thermalization_closure_failed",
        "mechanisms": mechanisms,
        "q_th_ion": thermal["ion"],
        "q_th_ib": thermal["ib"],
        "q_th_raman": thermal["raman"],
        "q_thermal": q_total,
        "E_th_ion_interval_J": energies["ion"],
        "E_th_ib_interval_J": energies["ib"],
        "E_th_raman_interval_J": energies["raman"],
        "E_thermal_interval_J": total_energy,
        "E_th_ion_pulse_J": float(np.sum(energies["ion"])),
        "E_th_ib_pulse_J": float(np.sum(energies["ib"])),
        "E_th_raman_pulse_J": float(np.sum(energies["raman"])),
        "E_thermal_pulse_J": float(np.sum(total_energy)),
        "level_t1": level_t1,
        "level_t2": level_t2,
        "level_t3": "pass" if t3_maps and t3_energy else "failed",
        "zero_channel_pass": bool(zero_channel_pass),
    }


__all__ = ["THERMALIZATION_CHANNELS", "build_complete_thermalization_ledger"]
