"""HR-3B post-acoustic thermal-index mapping and disk-backed slow state."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .device import to_cpu, xp


_RTOL = 2.0e-5
_ATOL = 1.0e-30


def validate_hr3b_parameters(*, rho0: float, Cv: float, T0: float, n0: float) -> float:
    """Validate the frozen post-acoustic mapping parameters and return beta."""
    values = {"rho0": rho0, "Cv": Cv, "T0": T0, "n0": n0}
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise ValueError("HR-3B parameters must be finite")
    if float(rho0) <= 0.0 or float(Cv) <= 0.0 or float(T0) <= 0.0:
        raise ValueError("HR-3B rho0, Cv, and T0 must be positive")
    if float(n0) <= 1.0:
        raise ValueError("HR-3B n0 must be greater than one")
    return (float(n0) - 1.0) / (float(rho0) * float(Cv) * float(T0))


def _finite_nonnegative_map(value, name: str):
    result = xp.asarray(value)
    if result.ndim != 2 or not bool(xp.all(xp.isfinite(result))):
        raise ValueError(f"{name} must be a finite [Ny, Nx] map")
    if bool(xp.any(result < 0.0)):
        raise ValueError(f"{name} must be non-negative")
    return result


def map_post_acoustic_increment(
    q_thermal,
    *,
    source_authoritative: bool,
    rho0: float,
    Cv: float,
    T0: float,
    n0: float,
) -> dict[str, object]:
    """Map one authoritative HR-3A interval source to a post-acoustic jump."""
    if not bool(source_authoritative):
        raise ValueError("HR-3B requires authoritative HR-3A q_thermal")
    beta = validate_hr3b_parameters(rho0=rho0, Cv=Cv, T0=T0, n0=n0)
    q = _finite_nonnegative_map(q_thermal, "q_thermal")
    increment = -float(beta) * q
    delta_t_impulse = q / (float(rho0) * float(Cv))
    delta_rho = float(rho0) * increment / (float(n0) - 1.0)
    delta_t_post = -float(T0) * delta_rho / float(rho0)
    mapping_residual = increment + float(beta) * q
    isobaric_residual = delta_t_post / float(T0) + delta_rho / float(rho0)
    tolerance = _ATOL + _RTOL * max(float(xp.max(xp.abs(increment))), 1.0e-300)
    mapping_ok = float(xp.max(xp.abs(mapping_residual))) <= tolerance
    isobaric_ok = float(xp.max(xp.abs(isobaric_residual))) <= _ATOL + _RTOL
    sign_ok = bool(
        xp.all(increment <= 0.0)
        and xp.all(delta_rho <= 0.0)
        and xp.all(delta_t_impulse >= 0.0)
        and xp.all(delta_t_post >= 0.0)
    )
    return {
        "authoritative": bool(mapping_ok and sign_ok and isobaric_ok),
        "beta_th": float(beta),
        "delta_n_increment": increment,
        "delta_t_impulse": delta_t_impulse,
        "delta_rho": delta_rho,
        "delta_t_post": delta_t_post,
        "mapping_residual": mapping_residual,
        "isobaric_residual": isobaric_residual,
        "mapping_ok": bool(mapping_ok),
        "sign_ok": bool(sign_ok),
        "thermodynamic_ok": bool(isobaric_ok),
    }


class ThermalSlowStateStore:
    """One interval-centered, disk-backed `delta_n_th[K, Ny, Nx]` state file."""

    def __init__(self, *, output_path: str, n_intervals: int, shape, dtype):
        if int(n_intervals) <= 0:
            raise ValueError("HR-3B state requires at least one interval")
        self.shape = tuple(int(value) for value in shape)
        if len(self.shape) != 2 or min(self.shape) <= 0:
            raise ValueError("HR-3B state slice shape must be positive [Ny, Nx]")
        self.n_intervals = int(n_intervals)
        self.dtype = np.dtype(dtype)
        if self.dtype.kind != "f":
            raise ValueError("HR-3B slow state dtype must be real floating point")
        self.path = Path(output_path).with_suffix(".hr3b_delta_n_th.npy")
        self._state = np.lib.format.open_memmap(
            self.path,
            mode="w+",
            dtype=self.dtype,
            shape=(self.n_intervals, *self.shape),
        )
        self._state.fill(0.0)
        self._state.flush()

    def _index(self, interval_index: int) -> int:
        index = int(interval_index)
        if index < 0 or index >= self.n_intervals:
            raise IndexError("HR-3B interval index is outside the persistent state")
        return index

    def read_interval(self, interval_index: int):
        """Return the current host-backed slice without materializing the volume."""
        return self._state[self._index(interval_index)]

    def update_interval(self, interval_index: int, delta_n_increment):
        """Add one transient increment in place and return the updated slice view."""
        index = self._index(interval_index)
        increment = np.asarray(to_cpu(delta_n_increment), dtype=self.dtype)
        if increment.shape != self.shape or not np.all(np.isfinite(increment)):
            raise ValueError("HR-3B increment has invalid shape or values")
        before = self._state[index]
        np.add(before, increment, out=before)
        if not np.all(np.isfinite(before)):
            raise ValueError("HR-3B state update produced non-finite values")
        return before

    def flush(self) -> None:
        if self._state is not None:
            self._state.flush()

    def finalize(self) -> dict[str, object]:
        self.flush()
        return self.metadata()

    def metadata(self) -> dict[str, object]:
        return {
            "hr3b_state_schema": "khz_filament.hr3b.delta_n_th.v1",
            "hr3b_state_filename": self.path.name,
            "hr3b_state_dtype": self.dtype.name,
            "hr3b_state_shape": (self.n_intervals, *self.shape),
            "hr3b_state_interval_centered": True,
            "hr3b_state_disk_backed": True,
        }


class HR3BDiagnosticSink:
    """Streaming sparse HR-3B diagnostics on the HR-3A sample-plan indices."""

    def __init__(self, *, plan, output_path: str, shape, dtype, enabled: bool):
        self.plan = plan
        self.shape = tuple(int(value) for value in shape)
        self.dtype = np.dtype(dtype)
        self.enabled = bool(enabled)
        self._slots = {int(interval): slot for slot, interval in enumerate(plan.interval_index)}
        self._written: set[int] = set()
        stem = Path(output_path)
        self.increment_path = stem.with_suffix(".hr3b_delta_n_increment_samples.npy")
        self.state_after_path = stem.with_suffix(".hr3b_delta_n_state_after_update_samples.npy")
        self._increments = self._states = None
        if self.enabled:
            archive_shape = (int(plan.count), *self.shape)
            self._increments = np.lib.format.open_memmap(
                self.increment_path, mode="w+", dtype=self.dtype, shape=archive_shape
            )
            self._states = np.lib.format.open_memmap(
                self.state_after_path, mode="w+", dtype=self.dtype, shape=archive_shape
            )

    def record_sample(self, interval_index: int, delta_n_increment, delta_n_state_after) -> None:
        slot = self._slots.get(int(interval_index))
        if slot is None or not self.enabled:
            return
        if slot in self._written:
            raise ValueError("HR-3B sample slot may be written only once")
        increment = np.asarray(to_cpu(delta_n_increment), dtype=self.dtype)
        state = np.asarray(delta_n_state_after, dtype=self.dtype)
        if (
            increment.shape != self.shape or state.shape != self.shape
            or not np.all(np.isfinite(increment)) or not np.all(np.isfinite(state))
        ):
            raise ValueError("HR-3B sample map has invalid shape or values")
        self._increments[slot] = increment
        self._states[slot] = state
        self._written.add(slot)

    def finalize(self) -> dict[str, object]:
        complete = bool(self.enabled and len(self._written) == self.plan.count)
        if self._increments is not None:
            self._increments.flush()
            self._increments = None
        if self._states is not None:
            self._states.flush()
            self._states = None
        return {
            "hr3b_map_archive_schema": "khz_filament.hr3b.sparse_maps.v1",
            "hr3b_increment_archive_filename": self.increment_path.name if self.enabled else "",
            "hr3b_state_after_archive_filename": self.state_after_path.name if self.enabled else "",
            "hr3b_map_archive_dtype": self.dtype.name if self.enabled else "",
            "hr3b_map_archive_shape": (self.plan.count, *self.shape) if self.enabled else (0, *self.shape),
            "hr3b_map_archive_complete": complete,
            "hr3b_map_archive_enabled": self.enabled,
            "hr3b_map_archive_disabled_reason": "" if self.enabled else "multi_pulse_archive_deferred_to_hr3c",
        }


class HR3BScalarLedger:
    """O(K) scalar closure record for one pulse's post-acoustic updates."""

    _VALUE_KEYS = (
        "increment_min", "increment_onaxis", "state_min_after", "state_onaxis_after",
        "impulse_temperature_max", "post_temperature_max", "density_min",
        "mapping_residual_max_abs", "isobaric_residual_max_abs",
    )

    def __init__(self):
        self.values = {name: [] for name in self._VALUE_KEYS}
        self.mapping_ok = True
        self.sign_ok = True
        self.thermodynamic_ok = True
        self.first_failed_interval = -1
        self.first_failed_level = ""

    def append(self, interval_index: int, result: dict[str, object], state_before, state_after) -> None:
        increment = result["delta_n_increment"]
        onaxis = (increment.shape[0] // 2, increment.shape[1] // 2)
        self.values["increment_min"].append(float(xp.min(increment)))
        self.values["increment_onaxis"].append(float(increment[onaxis]))
        self.values["state_min_after"].append(float(np.min(state_after)))
        self.values["state_onaxis_after"].append(float(state_after[onaxis]))
        self.values["impulse_temperature_max"].append(float(xp.max(result["delta_t_impulse"])))
        self.values["post_temperature_max"].append(float(xp.max(result["delta_t_post"])))
        self.values["density_min"].append(float(xp.min(result["delta_rho"])))
        self.values["mapping_residual_max_abs"].append(float(xp.max(xp.abs(result["mapping_residual"]))))
        self.values["isobaric_residual_max_abs"].append(float(xp.max(xp.abs(result["isobaric_residual"]))))
        for level, passed in (
            ("B2", bool(result["mapping_ok"])),
            ("B3", bool(result["sign_ok"])),
            ("B4", bool(result["thermodynamic_ok"])),
        ):
            if not passed and self.first_failed_interval < 0:
                self.first_failed_interval = int(interval_index)
                self.first_failed_level = level
        self.mapping_ok = self.mapping_ok and bool(result["mapping_ok"])
        self.sign_ok = self.sign_ok and bool(result["sign_ok"])
        self.thermodynamic_ok = self.thermodynamic_ok and bool(result["thermodynamic_ok"])

    def as_dict(self) -> dict[str, object]:
        arrays = {name: np.asarray(values, dtype=np.float64) for name, values in self.values.items()}
        authoritative = bool(self.mapping_ok and self.sign_ok and self.thermodynamic_ok)
        return {
            "hr3b_authoritative": authoritative,
            "hr3b_source_authority_status": "pass",
            "hr3b_mapping_status": "pass" if self.mapping_ok else "failed",
            "hr3b_sign_status": "pass" if self.sign_ok else "failed",
            "hr3b_thermodynamic_status": "pass" if self.thermodynamic_ok else "failed",
            "hr3b_first_failed_interval": self.first_failed_interval,
            "hr3b_first_failed_level": self.first_failed_level,
            "delta_n_increment_min": arrays["increment_min"],
            "delta_n_increment_onaxis": arrays["increment_onaxis"],
            "delta_n_state_min_after_update": arrays["state_min_after"],
            "delta_n_state_onaxis_after_update": arrays["state_onaxis_after"],
            "Delta_T_impulse_max": arrays["impulse_temperature_max"],
            "Delta_T_post_max": arrays["post_temperature_max"],
            "delta_rho_min": arrays["density_min"],
            "hr3b_mapping_max_abs_residual": arrays["mapping_residual_max_abs"],
            "hr3b_isobaric_max_abs_residual": arrays["isobaric_residual_max_abs"],
        }


__all__ = [
    "HR3BDiagnosticSink", "HR3BScalarLedger", "ThermalSlowStateStore",
    "map_post_acoustic_increment", "validate_hr3b_parameters",
]
