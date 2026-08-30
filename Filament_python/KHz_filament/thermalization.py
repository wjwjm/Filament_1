"""Streaming HR-3A thermalization and sparse diagnostic storage."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .device import xp, to_cpu


THERMALIZATION_CHANNELS = ("ion", "ib", "raman")
_RTOL = 2.0e-5
_ATOL = 1.0e-30


@dataclass(frozen=True)
class ThermalSamplePlan:
    target_z_m: np.ndarray
    interval_index: np.ndarray
    z_left_m: np.ndarray
    z_right_m: np.ndarray
    z_mid_m: np.ndarray
    snap_error_m: np.ndarray
    region: np.ndarray
    reason: np.ndarray

    @property
    def count(self) -> int:
        return int(self.interval_index.size)


def _physical_targets(start: float, end: float, spacing: float) -> list[float]:
    if end < start:
        return []
    count = int(math.floor((end - start) / spacing + 1.0e-12))
    values = [start + index * spacing for index in range(count + 1)]
    if not values or not math.isclose(values[-1], end, rel_tol=0.0, abs_tol=1e-12):
        values.append(end)
    return values


def build_physical_sample_plan(
    schedule,
    *,
    focus_center_m: float | None,
    focus_halfwidth_m: float,
    focus_enabled: bool,
    focal_plane_m: float | None,
    outer_spacing_m: float = 5.0e-3,
    focus_spacing_m: float = 1.0e-3,
) -> ThermalSamplePlan:
    """Map fixed physical-z targets to nearest interval midpoints."""
    if outer_spacing_m <= 0.0 or focus_spacing_m <= 0.0:
        raise ValueError("sample spacings must be positive")
    edges = np.asarray(schedule.z_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("sample plan requires a nonempty longitudinal schedule")
    mids = 0.5 * (edges[:-1] + edges[1:])
    start, end = float(edges[0]), float(edges[-1])
    candidates: list[tuple[float, str]] = [
        (target, "outer") for target in _physical_targets(start, end, outer_spacing_m)
    ]
    if focus_enabled and focus_center_m is not None and focus_halfwidth_m > 0.0:
        left = max(start, float(focus_center_m) - float(focus_halfwidth_m))
        right = min(end, float(focus_center_m) + float(focus_halfwidth_m))
        if left <= right:
            candidates.extend((target, "focus") for target in _physical_targets(left, right, focus_spacing_m))
            candidates.extend(((left, "landmark:focus_left"), (right, "landmark:focus_right")))
    candidates.extend(((float(mids[0]), "landmark:first_interval"), (float(mids[-1]), "landmark:last_interval")))
    if focal_plane_m is not None and math.isfinite(float(focal_plane_m)) and start <= float(focal_plane_m) <= end:
        candidates.append((float(focal_plane_m), "landmark:focal_plane"))

    grouped: dict[int, list[tuple[float, str]]] = {}
    for target, reason in candidates:
        # argmin selects the lower index on a midpoint-distance tie.
        index = int(np.argmin(np.abs(mids - target)))
        grouped.setdefault(index, []).append((target, reason))
    rows = []
    for index in sorted(grouped):
        entries = grouped[index]
        mid = float(mids[index])
        target, _ = min(entries, key=lambda item: (abs(mid - item[0]), item[0]))
        reasons = tuple(dict.fromkeys(reason for _, reason in entries))
        regions = []
        if any(reason == "outer" for reason in reasons):
            regions.append("outer")
        if any(reason == "focus" for reason in reasons):
            regions.append("focus")
        if any(reason.startswith("landmark") for reason in reasons):
            regions.append("landmark")
        rows.append((target, index, float(edges[index]), float(edges[index + 1]), mid, mid - target, "|".join(regions), "|".join(reasons)))
    return ThermalSamplePlan(
        target_z_m=np.asarray([row[0] for row in rows], dtype=np.float64),
        interval_index=np.asarray([row[1] for row in rows], dtype=np.int64),
        z_left_m=np.asarray([row[2] for row in rows], dtype=np.float64),
        z_right_m=np.asarray([row[3] for row in rows], dtype=np.float64),
        z_mid_m=np.asarray([row[4] for row in rows], dtype=np.float64),
        snap_error_m=np.asarray([row[5] for row in rows], dtype=np.float64),
        region=np.asarray([row[6] for row in rows], dtype="U32"),
        reason=np.asarray([row[7] for row in rows], dtype="U128"),
    )


class ThermalDiagnosticSink:
    """Disk-backed sparse map sink; it never participates in physics."""

    def __init__(self, *, plan: ThermalSamplePlan, output_path: str, shape, dtype, enabled: bool, mode: str = "production"):
        self.plan = plan
        self.output_path = Path(output_path)
        self.shape = tuple(int(value) for value in shape)
        self.dtype = np.dtype(dtype)
        self.enabled = bool(enabled)
        self.mode = str(mode)
        if self.mode not in {"production", "validation"}:
            raise ValueError("thermal diagnostic mode must be production or validation")
        self._slots = {int(interval): slot for slot, interval in enumerate(plan.interval_index)}
        self._written: set[int] = set()
        self._maps = None
        self._component_maps: dict[str, object] = {}
        self.archive_path = self.output_path.with_suffix(".hr3a_qthermal_samples.npy")
        if self.enabled:
            self._maps = np.lib.format.open_memmap(
                self.archive_path, mode="w+", dtype=self.dtype,
                shape=(plan.count, *self.shape),
            )

    def record_sample(self, interval_index: int, q_thermal, *, components=None) -> None:
        slot = self._slots.get(int(interval_index))
        if slot is None or not self.enabled:
            return
        if slot in self._written:
            raise ValueError("thermal sample slot may be written only once")
        value = np.asarray(to_cpu(q_thermal), dtype=self.dtype)
        if value.shape != self.shape or not np.all(np.isfinite(value)):
            raise ValueError("thermal sample map has invalid shape or values")
        self._maps[slot] = value
        if self.mode == "validation" and components:
            for name, component in components.items():
                if name not in THERMALIZATION_CHANNELS:
                    raise ValueError("unknown validation thermal component")
                archive = self._component_maps.get(name)
                if archive is None:
                    component_path = self.output_path.with_suffix(f".hr3a_q{name}_samples.npy")
                    archive = np.lib.format.open_memmap(
                        component_path, mode="w+", dtype=self.dtype,
                        shape=(self.plan.count, *self.shape),
                    )
                    self._component_maps[name] = archive
                archive[slot] = np.asarray(to_cpu(component), dtype=self.dtype)
        self._written.add(slot)

    def finalize(self) -> dict[str, object]:
        complete = bool(self.enabled and len(self._written) == self.plan.count)
        if self._maps is not None:
            self._maps.flush()
            self._maps = None
        for archive in self._component_maps.values():
            archive.flush()
        return {
            "thermal_map_archive_schema": "khz_filament.hr3a.qthermal_samples.v1",
            "thermal_map_archive_filename": self.archive_path.name if self.enabled else "",
            "thermal_map_archive_dtype": self.dtype.name if self.enabled else "",
            "thermal_map_archive_shape": (self.plan.count, *self.shape) if self.enabled else (0, *self.shape),
            "thermal_map_archive_complete": complete,
            "thermal_map_archive_enabled": self.enabled,
            "thermal_map_archive_mode": self.mode,
            "thermal_map_archive_disabled_reason": "" if self.enabled else "multi_pulse_archive_deferred_to_hr3c",
        }


def _map(value, name: str):
    result = xp.asarray(value)
    if result.ndim != 2 or not bool(xp.all(xp.isfinite(result))):
        raise ValueError(f"{name} must be a finite [Ny, Nx] map")
    if bool(xp.any(result < 0.0)):
        raise ValueError(f"{name} must be non-negative")
    return result


def _energy(q, dx: float, dy: float, dz: float) -> float:
    return float(xp.sum(q, dtype=xp.float64) * float(dx) * float(dy) * float(dz))


def thermalize_interval(*, q_ion, q_ib, q_raman, dz: float, dx: float, dy: float, mechanisms, reference_interval_J, x=None, y=None):
    """Finalize one interval after all HR-2 deposition components exist."""
    ion, ib, raman = _map(q_ion, "q_ion"), _map(q_ib, "q_ib"), _map(q_raman, "q_raman")
    if ion.shape != ib.shape or ion.shape != raman.shape:
        raise ValueError("thermalization maps must have identical shapes")
    unavailable = []
    t1_channel_status = {}
    for name, value in (("ion", ion), ("ib", ib), ("raman", raman)):
        active = bool(mechanisms[name]["active"])
        authoritative = bool(mechanisms[name]["authoritative"])
        if not active and bool(xp.any(value != 0.0)):
            raise ValueError(f"inactive {name} channel must be exact zero")
        if active and not authoritative:
            unavailable.append(
                f"{name}:{mechanisms[name].get('source', 'missing')}"
            )
            t1_channel_status[name] = "unavailable"
        else:
            t1_channel_status[name] = "pass"
    if unavailable:
        nan = math.nan
        return {"authoritative": False,
                "reason": "t1_non_authoritative_channel:" + ";".join(unavailable),
                "t1_ok": False, "t1_status": "unavailable",
                "t1_channel_status": t1_channel_status,
                "t2_ok": False, "t2_status": "unavailable",
                "t2_channel_status": {name: "unavailable" for name in THERMALIZATION_CHANNELS},
                "t3_ok": False, "t3_status": "unavailable", "q_thermal": None,
                "energies": {name: nan for name in (*THERMALIZATION_CHANNELS, "total")},
                "t2": {name: nan for name in THERMALIZATION_CHANNELS}, "t3": nan,
                "max": nan, "onaxis": nan, "radius": nan}
    q_total = ion + ib + raman
    channel_maps = {"ion": ion, "ib": ib, "raman": raman}
    energies = {name: _energy(value, dx, dy, dz) for name, value in channel_maps.items()}
    energies["total"] = _energy(q_total, dx, dy, dz)
    residuals = {name: energies[name] - float(reference_interval_J[name]) for name in THERMALIZATION_CHANNELS}
    t2_channel_ok = {
        name: abs(value) <= _ATOL + _RTOL * max(
            abs(energies[name]), abs(float(reference_interval_J[name]))
        )
        for name, value in residuals.items()
    }
    t2_ok = all(t2_channel_ok.values())
    t3 = energies["total"] - sum(energies[name] for name in THERMALIZATION_CHANNELS)
    t3_ok = abs(t3) <= _ATOL + _RTOL * max(abs(energies["total"]), 1.0e-300)
    total_sum = float(xp.sum(q_total, dtype=xp.float64))
    if total_sum > 0.0 and x is not None and y is not None:
        r2 = y[:, None] ** 2 + x[None, :] ** 2
        radius = math.sqrt(float(xp.sum(q_total * r2, dtype=xp.float64)) / total_sum)
    else:
        radius = math.nan
    reason = ""
    if not t2_ok:
        reason = "t2_reduction_closure_failed"
    elif not t3_ok:
        reason = "t3_channel_sum_closure_failed"
    return {"authoritative": bool(t2_ok and t3_ok), "reason": reason,
            "t1_ok": True, "t1_status": "pass",
            "t1_channel_status": t1_channel_status,
            "t2_ok": t2_ok, "t2_status": "pass" if t2_ok else "failed",
            "t2_channel_status": {
                name: "pass" if t2_channel_ok[name] else "failed"
                for name in THERMALIZATION_CHANNELS
            },
            "t3_ok": t3_ok, "t3_status": "pass" if t3_ok else "failed",
            "q_thermal": q_total, "energies": energies, "t2": residuals, "t3": t3,
            "max": float(xp.max(q_total)), "onaxis": float(q_total[q_total.shape[0] // 2, q_total.shape[1] // 2]), "radius": radius}


class ThermalScalarLedger:
    """O(K) online scalar diagnostic accumulator."""
    def __init__(self, mechanisms):
        self.mechanisms = mechanisms
        self.values = {name: [] for name in ("ion", "ib", "raman", "total", "max", "onaxis", "radius", "t2_ion", "t2_ib", "t2_raman", "t3")}
        self.first_failed_interval = -1
        self.first_failed_level = ""
        self.reason = ""
        self.t1_all_pass, self.t1_any_unavailable = True, False
        self.t2_all_pass, self.t2_any_unavailable = True, False
        self.t3_all_pass, self.t3_any_unavailable = True, False

    def append(self, index: int, result) -> None:
        for name in THERMALIZATION_CHANNELS:
            self.values[name].append(result["energies"][name])
            self.values[f"t2_{name}"].append(result["t2"][name])
        self.values["total"].append(result["energies"]["total"])
        self.values["t3"].append(result["t3"])
        self.values["max"].append(result["max"])
        self.values["onaxis"].append(result["onaxis"])
        self.values["radius"].append(result["radius"])
        for level in ("t1", "t2", "t3"):
            status = str(result[f"{level}_status"])
            if status != "pass":
                setattr(self, f"{level}_all_pass", False)
                if status == "unavailable":
                    setattr(self, f"{level}_any_unavailable", True)
                if self.first_failed_interval < 0:
                    self.first_failed_interval = int(index)
                    self.first_failed_level = level.upper()
                    self.reason = str(result["reason"])

    def as_dict(self) -> dict[str, object]:
        value = lambda name: np.asarray(self.values[name], dtype=np.float64)
        pulse_reductions_available = not self.t1_any_unavailable

        def aggregate_status(all_pass: bool, any_unavailable: bool) -> str:
            if all_pass:
                return "pass"
            return "unavailable" if any_unavailable else "failed"

        def max_abs(name):
            values = value(name)
            finite = values[np.isfinite(values)]
            return float(np.max(np.abs(finite))) if finite.size else math.nan
        return {
            "thermalization_authoritative": bool(
                self.t1_all_pass and self.t2_all_pass and self.t3_all_pass
            ),
            "thermalization_unavailable_reason": self.reason,
            "E_th_ion_interval_J": value("ion"), "E_th_ib_interval_J": value("ib"),
            "E_th_raman_interval_J": value("raman"), "E_thermal_interval_J": value("total"),
            "E_th_ion_pulse_J": float(np.nansum(value("ion"))) if pulse_reductions_available else math.nan,
            "E_th_ib_pulse_J": float(np.nansum(value("ib"))) if pulse_reductions_available else math.nan,
            "E_th_raman_pulse_J": float(np.nansum(value("raman"))) if pulse_reductions_available else math.nan,
            "E_thermal_pulse_J": float(np.nansum(value("total"))) if pulse_reductions_available else math.nan,
            "q_thermal_max_J_m3": value("max"), "q_thermal_onaxis_J_m3": value("onaxis"),
            "q_thermal_second_moment_radius_m": value("radius"),
            "thermalization_t2_ion_residual_J": value("t2_ion"), "thermalization_t2_ib_residual_J": value("t2_ib"),
            "thermalization_t2_raman_residual_J": value("t2_raman"), "thermalization_t3_energy_residual_J": value("t3"),
            "thermalization_max_abs_T2_residual_ion_J": max_abs("t2_ion"), "thermalization_max_abs_T2_residual_ib_J": max_abs("t2_ib"),
            "thermalization_max_abs_T2_residual_raman_J": max_abs("t2_raman"), "thermalization_max_abs_T3_residual_J": max_abs("t3"),
            "thermalization_first_failed_interval": self.first_failed_interval,
            "thermalization_first_failed_level": self.first_failed_level,
            "thermalization_t1_status": aggregate_status(self.t1_all_pass, self.t1_any_unavailable),
            "thermalization_t2_status": aggregate_status(self.t2_all_pass, self.t2_any_unavailable),
            "thermalization_t3_status": aggregate_status(self.t3_all_pass, self.t3_any_unavailable),
        }


def _status_from_intervals(statuses) -> str:
    """Reduce per-interval diagnostic states without conflating T1/T2/T3."""
    states = tuple(str(status) for status in statuses)
    if all(state == "pass" for state in states):
        return "pass"
    return "unavailable" if "unavailable" in states else "failed"


# Compatibility-only whole-stack helper for small unit tests. Production uses
# thermalize_interval plus ThermalScalarLedger.
def build_complete_thermalization_ledger(**kwargs):
    for name in ("q_ion", "q_ib", "q_raman"):
        if kwargs.get(name) is None:
            raise ValueError(f"{name} is required")
    q_ion = np.asarray(kwargs["q_ion"])
    q_ib = np.asarray(kwargs["q_ib"])
    q_raman = np.asarray(kwargs["q_raman"])
    if q_ion.ndim != 3 or q_ion.shape != q_ib.shape or q_ion.shape != q_raman.shape:
        raise ValueError("q maps must have matching [K, Ny, Nx] shapes")
    edges = np.asarray(kwargs["z_edges"], dtype=np.float64)
    dz_values = np.asarray(kwargs["dz_intervals"], dtype=np.float64)
    if (
        edges.shape != (q_ion.shape[0] + 1,)
        or dz_values.shape != (q_ion.shape[0],)
        or not np.allclose(np.diff(edges), dz_values, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError("thermalization schedule has invalid interval geometry")
    ledger = ThermalScalarLedger(kwargs["deposition_mechanisms"])
    results = []
    for index in range(q_ion.shape[0]):
        result = thermalize_interval(q_ion=q_ion[index], q_ib=q_ib[index], q_raman=q_raman[index],
            dz=float(dz_values[index]), dx=kwargs["dx"], dy=kwargs["dy"],
            mechanisms=kwargs["deposition_mechanisms"], reference_interval_J={name: kwargs["deposition_interval_J"][name][index] for name in THERMALIZATION_CHANNELS})
        results.append(result)
        ledger.append(index, result)
    summary = ledger.as_dict()
    maps = (q_ion, q_ib, q_raman, q_ion + q_ib + q_raman)
    if summary["thermalization_t1_status"] != "pass":
        maps = tuple(np.full_like(value, np.nan, dtype=np.float64) for value in maps)
    level_t1 = {
        name: _status_from_intervals(result["t1_channel_status"][name] for result in results)
        for name in THERMALIZATION_CHANNELS
    }
    level_t2 = {
        name: _status_from_intervals(result["t2_channel_status"][name] for result in results)
        for name in THERMALIZATION_CHANNELS
    }
    level_t3 = summary["thermalization_t3_status"]
    unavailable_reason = summary["thermalization_unavailable_reason"]
    summary.update({"authoritative": summary["thermalization_authoritative"], "unavailable_reason": unavailable_reason, "mechanisms": kwargs["deposition_mechanisms"],
                    "q_th_ion": maps[0], "q_th_ib": maps[1], "q_th_raman": maps[2], "q_thermal": maps[3],
                    "level_t1": level_t1, "level_t2": level_t2, "level_t3": level_t3, "zero_channel_pass": True})
    return summary


__all__ = ["ThermalDiagnosticSink", "ThermalSamplePlan", "ThermalScalarLedger", "build_complete_thermalization_ledger", "build_physical_sample_plan", "thermalize_interval"]
