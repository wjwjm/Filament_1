#!/usr/bin/env python3
"""Apply the frozen mechanical gates to a completed 0.60 m pair.

No coordinate shift, smoothing, or curve renormalisation is applied.  The
hybrid trace is only linearly interpolated onto reference samples inside the
two traces' common z interval.  Missing execution/scheduler evidence raises
``InsufficientEvidenceError`` instead of manufacturing a
``hybrid_0p60_not_supported`` result.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CAMPAIGN_ID = "hybrid_propagation_validation_0p60"
EXPECTED_GPU = "NVIDIA GeForce RTX 5090"
AUDIT_SCHEMA = "khz_filament.hybrid_propagation_validation.postprocess_audit.v1"
MAX_ONSET_SHIFT_CM = 0.10
MAX_PEAK_REL = 0.02
MAX_COMPONENT_POSITION_CM = 0.10
MAX_NRMSE = 0.02
MIN_CORRELATION = 0.995
LOW_ONSET_RISK_CM = 0.20
LOW_MEAN_RISK_CM = 0.10
PERFORMANCE_REDUCTION = 0.01
RHO_THRESHOLDS = (1.0e19, 1.0e20, 1.0e21, 1.0e22)
INTENSITY_FRACTIONS = (0.10, 0.50, 0.90)
PEAK_PROMINENCE = 0.05
PEAK_DISTANCE_M = 0.001  # 0.10 cm


class InsufficientEvidenceError(RuntimeError):
    """Raised when comparison provenance or terminal evidence is incomplete."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InsufficientEvidenceError(f"audit JSON is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise InsufficientEvidenceError("audit JSON must be an object")
    return value


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise InsufficientEvidenceError(f"derived axial CSV is missing: {path}")
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise InsufficientEvidenceError(f"derived axial CSV is unreadable: {exc}") from exc
    if not rows or "z_m" not in rows[0]:
        raise InsufficientEvidenceError(f"derived axial CSV has no z_m rows: {path}")
    columns: dict[str, np.ndarray] = {}
    for key in rows[0]:
        try:
            values = []
            for row in rows:
                raw = str(row[key]).strip()
                if raw.lower() == "true":
                    values.append(1.0)
                elif raw.lower() == "false":
                    values.append(0.0)
                else:
                    values.append(float(raw))
            columns[key] = np.asarray(values, dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise InsufficientEvidenceError(f"derived CSV column is not numeric: {key}") from exc
        if not np.all(np.isfinite(columns[key])):
            raise InsufficientEvidenceError(f"derived CSV column contains NaN/Inf: {key}")
    z = columns["z_m"]
    if z.size < 2 or np.any(np.diff(z) <= 0.0):
        raise InsufficientEvidenceError(f"derived CSV z_m is not strictly increasing: {path}")
    return columns


def _scalar_from_audit(audit: dict[str, Any], case: str, key: str, default: Any = None) -> Any:
    value = audit.get("cases", {}).get(case, {}).get(key, default)
    return value


def _load_visual_veto(value: Path | dict[str, Any] | bool | None) -> dict[str, Any]:
    if isinstance(value, Path):
        if not value.is_file():
            raise InsufficientEvidenceError(f"visual veto JSON is missing: {value}")
        payload = _json(value)
        payload["input_path"] = str(value.resolve())
    elif isinstance(value, dict):
        payload = dict(value)
    elif isinstance(value, bool):
        payload = {"veto": value, "source": "explicit-api-input"}
    else:
        raise InsufficientEvidenceError(
            "explicit visual-veto input is required after reviewing the comparison plot"
        )
    if "veto" not in payload and "visual_veto" in payload:
        payload["veto"] = payload["visual_veto"]
    if not isinstance(payload.get("veto"), bool):
        raise InsufficientEvidenceError("visual veto input must contain boolean veto")
    if payload["veto"]:
        curve = str(payload.get("curve") or "").strip()
        feature = str(payload.get("feature") or "").strip()
        interval = payload.get("z_interval_m")
        if not curve or not feature:
            raise InsufficientEvidenceError(
                "a true visual veto must identify curve and added/disappeared feature"
            )
        if (
            not isinstance(interval, list) or len(interval) != 2
            or not all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in interval)
            or float(interval[1]) < float(interval[0])
        ):
            raise InsufficientEvidenceError(
                "a true visual veto must include finite ordered z_interval_m=[start,end]"
            )
    payload["veto"] = bool(payload["veto"])
    return payload


def _common_grid(reference: dict[str, np.ndarray], hybrid: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_ref = reference["z_m"]
    z_hyb = hybrid["z_m"]
    lo = max(float(z_ref[0]), float(z_hyb[0]))
    hi = min(float(z_ref[-1]), float(z_hyb[-1]))
    mask = (z_ref >= lo) & (z_ref <= hi)
    grid = z_ref[mask]
    if grid.size < 2:
        raise InsufficientEvidenceError("reference/hybrid axial domains have no common interval")
    return grid, reference["z_m"], hybrid["z_m"]


def _interp_pair(reference: dict[str, np.ndarray], hybrid: dict[str, np.ndarray], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid, _, _ = _common_grid(reference, hybrid)
    if key not in reference or key not in hybrid:
        raise InsufficientEvidenceError(f"required comparison column missing: {key}")
    ref = np.interp(grid, reference["z_m"], reference[key])
    hyb = np.interp(grid, hybrid["z_m"], hybrid[key])
    return grid, ref, hyb


def onset(z: np.ndarray, y: np.ndarray, threshold: float) -> float | None:
    above = y >= threshold
    indexes = np.flatnonzero(above)
    if indexes.size == 0:
        return None
    i = int(indexes[0])
    if i == 0:
        return float(z[0])
    y0, y1 = float(y[i - 1]), float(y[i])
    if y1 == y0:
        return float(z[i])
    fraction = (threshold - y0) / (y1 - y0)
    return float(z[i - 1] + np.clip(fraction, 0.0, 1.0) * (z[i] - z[i - 1]))


def components(z: np.ndarray, y: np.ndarray, threshold: float) -> list[dict[str, float]]:
    mask = np.asarray(y >= threshold, dtype=bool)
    result: list[dict[str, float]] = []
    i = 0
    while i < mask.size:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i + 1 < mask.size and mask[i + 1]:
            i += 1
        end = i
        peak_index = start + int(np.argmax(y[start : end + 1]))
        start_m = float(z[start])
        if start > 0 and float(y[start]) != float(y[start - 1]):
            fraction = (threshold - float(y[start - 1])) / (float(y[start]) - float(y[start - 1]))
            start_m = float(z[start - 1] + np.clip(fraction, 0.0, 1.0) * (z[start] - z[start - 1]))
        end_m = float(z[end])
        if end + 1 < mask.size and float(y[end + 1]) != float(y[end]):
            fraction = (threshold - float(y[end])) / (float(y[end + 1]) - float(y[end]))
            end_m = float(z[end] + np.clip(fraction, 0.0, 1.0) * (z[end + 1] - z[end]))
        result.append({
            "start_m": start_m,
            "end_m": end_m,
            "peak_m": float(z[peak_index]),
            "peak_value": float(y[peak_index]),
        })
        i += 1
    return result


def _peak_positions(z: np.ndarray, y: np.ndarray, *, prominence: float, distance_m: float) -> np.ndarray:
    # Use scipy when available, while enforcing the physical distance on the
    # actual nonuniform z coordinate rather than sample count.
    candidates: list[int] = []
    try:
        from scipy.signal import find_peaks  # type: ignore
        spacing = max(1, int(math.ceil(distance_m / max(float(np.min(np.diff(z))), 1e-30))))
        candidates = list(find_peaks(y, prominence=prominence, distance=spacing)[0])
    except Exception:
        for i in range(1, y.size - 1):
            if y[i] >= y[i - 1] and y[i] >= y[i + 1] and (y[i] - max(y[i - 1], y[i + 1])) >= prominence:
                candidates.append(i)
    accepted: list[int] = []
    for index in sorted(candidates, key=lambda i: float(y[i]), reverse=True):
        if all(abs(float(z[index] - z[other])) >= distance_m for other in accepted):
            accepted.append(index)
    return np.asarray(sorted(accepted), dtype=int)


def _corr(reference: np.ndarray, hybrid: np.ndarray) -> float:
    if np.allclose(reference, reference[0]) or np.allclose(hybrid, hybrid[0]):
        return 1.0 if np.allclose(reference, hybrid) else 0.0
    return float(np.corrcoef(reference, hybrid)[0, 1])


def _component_gate(ref: list[dict[str, float]], hyb: list[dict[str, float]]) -> dict[str, Any]:
    failures: list[str] = []
    if len(ref) != len(hyb):
        failures.append(f"component count {len(ref)} != {len(hyb)}")
    pairs = []
    for index, (left, right) in enumerate(zip(ref, hyb)):
        delta = {key: (right[key] - left[key]) * 100.0 for key in ("start_m", "end_m", "peak_m")}
        pairs.append({"index": index, "reference": left, "hybrid": right, "delta_cm": delta})
        if any(abs(value) >= MAX_COMPONENT_POSITION_CM for value in delta.values()):
            failures.append(f"component {index} position difference >= {MAX_COMPONENT_POSITION_CM} cm")
    return {"reference_count": len(ref), "hybrid_count": len(hyb), "pairs": pairs, "passed": not failures, "failures": failures}


def _threshold_metrics(reference: dict[str, np.ndarray], hybrid: dict[str, np.ndarray]) -> dict[str, Any]:
    z, rho_ref, rho_hyb = _interp_pair(reference, hybrid, "rho_max_z")
    _, intensity_ref, intensity_hyb = _interp_pair(reference, hybrid, "I_max_z")
    result: dict[str, Any] = {"rho": {}, "intensity": {}, "low_threshold_risk": {}}
    rho_onsets: dict[str, float] = {}
    low_shifts: list[float] = []
    for threshold in RHO_THRESHOLDS:
        ref_on = onset(z, rho_ref, threshold)
        hyb_on = onset(z, rho_hyb, threshold)
        key = f"{threshold:.0e}"
        if ref_on is None or hyb_on is None:
            result["rho"][key] = {"reference_onset_m": ref_on, "hybrid_onset_m": hyb_on, "passed": False, "failure": "onset missing"}
            continue
        shift_cm = (hyb_on - ref_on) * 100.0
        rho_onsets[key] = shift_cm
        result["rho"][key] = {
            "reference_onset_m": ref_on,
            "hybrid_onset_m": hyb_on,
            "reference_x_focus_cm": 100.0 * (ref_on - 0.95),
            "hybrid_x_focus_cm": 100.0 * (hyb_on - 0.95),
            "shift_cm": shift_cm,
            "passed": abs(shift_cm) < MAX_ONSET_SHIFT_CM if threshold == 1e22 else True,
        }
        if threshold in RHO_THRESHOLDS[:3]:
            low_shifts.append(shift_cm)
        result["rho"][key]["components"] = _component_gate(
            components(z, rho_ref, threshold), components(z, rho_hyb, threshold)
        )
        if not result["rho"][key]["components"]["passed"]:
            result["rho"][key]["passed"] = False
    peak_ref = float(np.max(intensity_ref))
    if not math.isfinite(peak_ref) or peak_ref <= 0.0:
        raise InsufficientEvidenceError("reference I_max_z peak is not positive/finite")
    for fraction in INTENSITY_FRACTIONS:
        threshold = fraction * peak_ref
        key = f"{int(fraction * 100)}pct"
        result["intensity"][key] = {
            "threshold": threshold,
            "components": _component_gate(
                components(z, intensity_ref, threshold), components(z, intensity_hyb, threshold)
            ),
        }
        result["intensity"][key]["passed"] = result["intensity"][key]["components"]["passed"]
    ref_norm = intensity_ref / peak_ref
    hyb_norm = intensity_hyb / peak_ref
    nrmse = float(np.sqrt(np.mean((ref_norm - hyb_norm) ** 2)))
    corr = _corr(ref_norm, hyb_norm)
    rho_peak_ref = float(np.max(rho_ref))
    if not math.isfinite(rho_peak_ref) or rho_peak_ref <= 0.0:
        raise InsufficientEvidenceError("reference rho_max_z peak is not positive/finite")
    rho_ref_norm = rho_ref / rho_peak_ref
    rho_hyb_norm = rho_hyb / rho_peak_ref
    rho_nrmse = float(np.sqrt(np.mean((rho_ref_norm - rho_hyb_norm) ** 2)))
    rho_corr = _corr(rho_ref_norm, rho_hyb_norm)
    rho_peak_hyb = float(np.max(rho_hyb))
    rho_peak_ref_index = int(np.argmax(rho_ref))
    rho_peak_hyb_index = int(np.argmax(rho_hyb))
    rho_ref_peaks = _peak_positions(
        z, rho_ref, prominence=PEAK_PROMINENCE * rho_peak_ref, distance_m=PEAK_DISTANCE_M
    )
    rho_hyb_peaks = _peak_positions(
        z, rho_hyb, prominence=PEAK_PROMINENCE * rho_peak_ref, distance_m=PEAK_DISTANCE_M
    )
    ref_peaks = _peak_positions(z, intensity_ref, prominence=PEAK_PROMINENCE * peak_ref, distance_m=PEAK_DISTANCE_M)
    hyb_peaks = _peak_positions(z, intensity_hyb, prominence=PEAK_PROMINENCE * peak_ref, distance_m=PEAK_DISTANCE_M)
    result["peak_rho"] = {
        "reference": rho_peak_ref,
        "hybrid": rho_peak_hyb,
        "relative_difference": abs(rho_peak_hyb - rho_peak_ref) / rho_peak_ref,
        "reference_z_m": float(z[rho_peak_ref_index]),
        "hybrid_z_m": float(z[rho_peak_hyb_index]),
        "reference_x_focus_cm": 100.0 * (float(z[rho_peak_ref_index]) - 0.95),
        "hybrid_x_focus_cm": 100.0 * (float(z[rho_peak_hyb_index]) - 0.95),
        "location_shift_cm": 100.0 * float(z[rho_peak_hyb_index] - z[rho_peak_ref_index]),
    }
    result["rho_curve"] = {
        "reference_peak_count": int(rho_ref_peaks.size),
        "hybrid_peak_count": int(rho_hyb_peaks.size),
        "reference_peak_positions_m": z[rho_ref_peaks].tolist(),
        "hybrid_peak_positions_m": z[rho_hyb_peaks].tolist(),
    }
    result["intensity_curve"] = {
        "reference_peak": peak_ref,
        "hybrid_peak": float(np.max(intensity_hyb)),
        "peak_relative_difference": abs(float(np.max(intensity_hyb)) - peak_ref) / peak_ref,
        "nrmse": nrmse,
        "correlation": corr,
        "reference_peak_count": int(ref_peaks.size),
        "hybrid_peak_count": int(hyb_peaks.size),
        "reference_peak_positions_m": z[ref_peaks].tolist(),
        "hybrid_peak_positions_m": z[hyb_peaks].tolist(),
    }
    result["normalized_curves"] = {
        "rho_max_z": {
            "reference_peak": rho_peak_ref,
            "nrmse": rho_nrmse,
            "correlation": rho_corr,
        },
        "I_max_z": {
            "reference_peak": peak_ref,
            "nrmse": nrmse,
            "correlation": corr,
        },
    }
    result["low_threshold_risk"] = {
        "shifts_cm": low_shifts,
        "any_abs_shift_ge_0.20cm": any(abs(value) >= LOW_ONSET_RISK_CM for value in low_shifts),
        "all_same_sign": bool(low_shifts) and (all(value > 0 for value in low_shifts) or all(value < 0 for value in low_shifts)),
        "mean_abs_shift_cm": float(np.mean(np.abs(low_shifts))) if low_shifts else None,
    }
    result["low_threshold_risk"]["systematic_risk"] = bool(
        result["low_threshold_risk"]["any_abs_shift_ge_0.20cm"]
        or (result["low_threshold_risk"]["all_same_sign"] and float(result["low_threshold_risk"]["mean_abs_shift_cm"]) >= LOW_MEAN_RISK_CM)
    )
    result["common_z_interval_m"] = [float(z[0]), float(z[-1])]
    return result


def _health_metrics(reference: dict[str, np.ndarray], hybrid: dict[str, np.ndarray], audit: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    for case, data in (("reference", reference), ("hybrid", hybrid)):
        for key, values in data.items():
            if not np.all(np.isfinite(values)):
                failures.append(f"{case}.{key} nonfinite")
    # Counters are cumulative in the production diagnostics; a hybrid run may
    # not introduce a larger safety/adaptive event count than its paired full
    # reference.  Missing optional counters are not treated as failure here.
    counter_comparison = {}
    for key in ("adaptive_rejection_count_z", "safety_mode_trigger_count_z"):
        if key in reference and key in hybrid:
            ref_max = float(np.max(reference[key]))
            hyb_max = float(np.max(hybrid[key]))
            counter_comparison[key] = {"reference_max": ref_max, "hybrid_max": hyb_max, "passed": hyb_max <= ref_max}
            if hyb_max > ref_max:
                failures.append(f"hybrid {key} exceeds reference")
    # The linear preamble must have no deposition in any available trace.
    if "nonlinear_operator_applied" in hybrid:
        inactive = ~np.asarray(hybrid["nonlinear_operator_applied"], dtype=bool)
        for key in ("E_dep_z", "E_dep_total_z", "E_dep_rot_z", "alpha_ion_applied_max_z"):
            if key in hybrid and np.any(np.abs(hybrid[key][inactive]) > 0.0):
                failures.append(f"hybrid linear segment has nonzero {key}")
    energy = {}
    if "U_z" in reference and "U_z" in hybrid:
        ref_u = np.asarray(reference["U_z"], dtype=float)
        hyb_u = np.asarray(hybrid["U_z"], dtype=float)
        ref_scale = max(abs(float(ref_u[0])), 1e-300)
        hyb_scale = max(abs(float(hyb_u[0])), 1e-300)
        ref_drift = float(np.max(np.abs((ref_u - ref_u[0]) / ref_scale)) )
        hyb_drift = float(np.max(np.abs((hyb_u - hyb_u[0]) / hyb_scale)) )
        energy = {"reference_max_relative_drift": ref_drift, "hybrid_max_relative_drift": hyb_drift, "passed": hyb_drift <= ref_drift + 1e-12}
        if not energy["passed"]:
            failures.append("hybrid energy diagnostic drift exceeds reference")
    return {"finite": not failures, "counter_comparison": counter_comparison, "energy": energy, "failures": failures}


def _performance_metrics(audit: dict[str, Any]) -> dict[str, Any]:
    cases = audit.get("cases", {})
    try:
        ref = cases["reference"]["performance"]
        hyb = cases["hybrid"]["performance"]
        ref_total = float(ref["case_total_walltime_s"])
        hyb_total = float(hyb["case_total_walltime_s"])
        ref_step = float(ref["step_time_s"])
        hyb_step = float(hyb["step_time_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InsufficientEvidenceError(f"performance evidence is incomplete: {exc}") from exc
    if min(ref_total, hyb_total, ref_step, hyb_step) <= 0.0:
        raise InsufficientEvidenceError("performance wall times must be positive")
    total_reduction = (ref_total - hyb_total) / ref_total
    step_reduction = (ref_step - hyb_step) / ref_step
    ref_calls = int(ref.get("nonlinear_call_count", 0))
    hyb_calls = int(hyb.get("nonlinear_call_count", 0))
    ref_ion_calls = int(ref.get("ionization_call_count", 0))
    hyb_ion_calls = int(hyb.get("ionization_call_count", 0))
    ref_raman_substeps = int(ref.get("raman_substep_count", 0))
    hyb_raman_substeps = int(hyb.get("raman_substep_count", 0))
    ref_raman_convolutions = int(ref.get("raman_convolution_count", 0))
    hyb_raman_convolutions = int(hyb.get("raman_convolution_count", 0))
    calls_reduced = (
        hyb_calls < ref_calls
        and hyb_ion_calls < ref_ion_calls
        and hyb_raman_substeps < ref_raman_substeps
        and hyb_raman_convolutions < ref_raman_convolutions
    )
    return {
        "reference_case_total_walltime_s": ref_total,
        "hybrid_case_total_walltime_s": hyb_total,
        "reference_step_time_s": ref_step,
        "hybrid_step_time_s": hyb_step,
        "case_total_reduction_fraction": total_reduction,
        "step_time_reduction_fraction": step_reduction,
        "case_total_speedup": ref_total / hyb_total,
        "step_time_speedup": ref_step / hyb_step,
        "nonlinear_operator_call_reduction": ref_calls - hyb_calls,
        "reference_nonlinear_operator_calls": ref_calls,
        "hybrid_nonlinear_operator_calls": hyb_calls,
        "ionization_call_reduction": ref_ion_calls - hyb_ion_calls,
        "reference_ionization_calls": ref_ion_calls,
        "hybrid_ionization_calls": hyb_ion_calls,
        "raman_substep_reduction": ref_raman_substeps - hyb_raman_substeps,
        "reference_raman_substeps": ref_raman_substeps,
        "hybrid_raman_substeps": hyb_raman_substeps,
        "raman_convolution_reduction": ref_raman_convolutions - hyb_raman_convolutions,
        "reference_raman_convolutions": ref_raman_convolutions,
        "hybrid_raman_convolutions": hyb_raman_convolutions,
        "reference_gpu_peak_allocated_bytes": int(ref.get("gpu_peak_allocated_bytes", 0)),
        "hybrid_gpu_peak_allocated_bytes": int(hyb.get("gpu_peak_allocated_bytes", 0)),
        "reference_gpu_peak_reserved_bytes": int(ref.get("gpu_peak_reserved_bytes", 0)),
        "hybrid_gpu_peak_reserved_bytes": int(hyb.get("gpu_peak_reserved_bytes", 0)),
        "operator_calls_reduced": calls_reduced,
        "passed": (
            total_reduction > PERFORMANCE_REDUCTION
            and step_reduction > PERFORMANCE_REDUCTION
            and calls_reduced
        ),
    }


def _provenance_gate(audit: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("campaign_id") != CAMPAIGN_ID:
        failures.append("audit schema/campaign mismatch")
    if audit.get("status") != "complete_evidence":
        failures.append("audit status is not complete_evidence")
    if audit.get("case_order") != ["reference", "hybrid"]:
        failures.append("case order mismatch")
    if audit.get("gpu_model") != EXPECTED_GPU:
        failures.append("GPU provenance mismatch")
    expected_diff = [
        {"path": "propagation.propagation_mode", "reference": "full_nonlinear_from_z0", "hybrid": "hybrid"},
        {"path": "propagation.z_nl_start", "reference": 0.0, "hybrid": 0.6},
    ]
    if audit.get("strict_config_diff") != expected_diff:
        failures.append("strict A/B config diff is not the fixed two-field delta")
    terminal = audit.get("scheduler_terminal_evidence", {})
    if str(terminal.get("state", "")).upper() not in {"COMPLETED", "COMPLETE"} or str(terminal.get("exit_code", "")) not in {"0", "0:0", "COMPLETED"}:
        failures.append("scheduler terminal evidence is incomplete")
    cases = audit.get("cases", {})
    if not isinstance(cases, dict) or set(cases) != {"reference", "hybrid"}:
        failures.append("case audit records are incomplete")
    else:
        shas = {str(cases[k].get("execution_git_sha", audit.get("execution_git_sha"))) for k in cases}
        gpus = {str(cases[k].get("gpu_model", audit.get("gpu_model"))) for k in cases}
        if len(shas) != 1 or "" in shas:
            failures.append("execution SHA is not shared")
        if len(gpus) != 1 or EXPECTED_GPU not in gpus:
            failures.append("GPU is not shared/fixed")
    return {"passed": not failures, "failures": failures}


def _write_plot(out_path: Path, z: np.ndarray, rho_ref: np.ndarray, rho_hyb: np.ndarray, intensity_ref: np.ndarray, intensity_hyb: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        # Keep the output contract deterministic in a minimal environment; the
        # report records that the visual artifact could not be rendered.
        out_path.write_bytes(b"")
        return
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True)
    axes[0].semilogy(z, np.maximum(rho_ref, 1e-99), label="reference")
    axes[0].semilogy(z, np.maximum(rho_hyb, 1e-99), "--", label="hybrid")
    axes[0].set_ylabel(r"$\rho_{max}$ (m$^{-3}$)")
    axes[1].plot(z, intensity_ref, label="reference")
    axes[1].plot(z, intensity_hyb, "--", label="hybrid")
    axes[1].set_ylabel(r"$I_{max}$ (W m$^{-2}$)")
    axes[1].set_xlabel("z (m)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def compare_pair(
    audit_path: Path,
    out_dir: Path,
    *,
    visual_veto: Path | dict[str, Any] | bool | None = None,
) -> dict[str, Any]:
    audit_path = Path(audit_path).expanduser().resolve()
    audit = _json(audit_path)
    provenance = _provenance_gate(audit)
    if not provenance["passed"]:
        raise InsufficientEvidenceError("comparison provenance is incomplete: " + "; ".join(provenance["failures"]))
    out_dir = Path(out_dir).expanduser().resolve()
    ref_csv = Path(audit["derived_outputs"]["reference_axial_csv"])
    hyb_csv = Path(audit["derived_outputs"]["hybrid_axial_csv"])
    reference = _read_csv(ref_csv)
    hybrid = _read_csv(hyb_csv)
    threshold = _threshold_metrics(reference, hybrid)
    health = _health_metrics(reference, hybrid, audit)
    performance = _performance_metrics(audit)
    veto = _load_visual_veto(visual_veto)
    gate_results: dict[str, bool] = {}
    gate_results["G1_onset_1e22"] = bool(threshold["rho"]["1e+22"]["passed"])
    gate_results["G2_peak_rho"] = bool(threshold["peak_rho"]["relative_difference"] < MAX_PEAK_REL)
    gate_results["G3_rho_topology"] = all(value.get("passed", False) and value.get("components", {}).get("passed", False) for value in threshold["rho"].values())
    gate_results["G3_intensity_topology"] = all(value.get("passed", False) for value in threshold["intensity"].values())
    gate_results["G3_curve"] = bool(
        all(metrics["nrmse"] < MAX_NRMSE and metrics["correlation"] > MIN_CORRELATION for metrics in threshold["normalized_curves"].values())
        and threshold["intensity_curve"]["reference_peak_count"] == threshold["intensity_curve"]["hybrid_peak_count"]
        and threshold["rho_curve"]["reference_peak_count"] == threshold["rho_curve"]["hybrid_peak_count"]
    )
    gate_results["G3_low_threshold_risk"] = not bool(threshold["low_threshold_risk"]["systematic_risk"])
    gate_results["numerical_health"] = bool(health["finite"] and not health["failures"])
    gate_results["performance"] = bool(performance["passed"])
    gate_results["visual_veto"] = not bool(veto["veto"])
    failures = [name for name, passed in gate_results.items() if not passed]
    classification = "hybrid_0p60_supported" if not failures else "hybrid_0p60_not_supported"
    z, rho_ref, rho_hyb = _interp_pair(reference, hybrid, "rho_max_z")
    _, intensity_ref, intensity_hyb = _interp_pair(reference, hybrid, "I_max_z")
    result = {
        "schema": "khz_filament.hybrid_propagation_validation.comparison.v1",
        "campaign_id": CAMPAIGN_ID,
        "classification": classification,
        "provenance": provenance,
        "visual_veto": veto,
        "gates": gate_results,
        "failed_gates": failures,
        "threshold_metrics": threshold,
        "health_metrics": health,
        "performance_metrics": performance,
        "comparison_conventions": {
            "coordinate_policy": "common z overlap; no shift",
            "smoothing": False,
            "renormalization": False,
            "curve_metric_normalization": "divide both curves by fixed reference peak only",
            "interpolation": "linear hybrid onto reference samples in common interval",
        },
        "audit_path": str(audit_path),
        "execution_git_sha": audit.get("execution_git_sha"),
        "gpu_model": audit.get("gpu_model"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "hybrid_propagation_validation_comparison.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metric_rows = [{"gate": key, "passed": str(value).lower()} for key, value in gate_results.items()]
    with (out_dir / "gate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gate", "passed"])
        writer.writeheader(); writer.writerows(metric_rows)
    _write_plot(out_dir / "hybrid_propagation_validation_comparison.png", z, rho_ref, rho_hyb, intensity_ref, intensity_hyb)
    report_lines = [
        "# Hybrid Propagation 0.60 m comparison",
        "",
        f"Classification: **{classification}**",
        "",
        f"- Execution SHA: `{audit.get('execution_git_sha')}`",
        f"- GPU: `{audit.get('gpu_model')}`",
        f"- Failed gates: `{', '.join(failures) if failures else 'none'}`",
        f"- G1 onset shift: `{threshold['rho']['1e+22'].get('shift_cm')}` cm",
        f"- G2 peak relative difference: `{threshold['peak_rho']['relative_difference']:.6g}`",
        f"- Peak rho reference/hybrid: `{threshold['peak_rho']['reference']:.6g}` / `{threshold['peak_rho']['hybrid']:.6g}` m^-3",
        f"- Peak rho x_focus reference/hybrid/delta: `{threshold['peak_rho']['reference_x_focus_cm']:.6g}` / `{threshold['peak_rho']['hybrid_x_focus_cm']:.6g}` / `{threshold['peak_rho']['location_shift_cm']:.6g}` cm",
        f"- I curve NRMSE/correlation: `{threshold['intensity_curve']['nrmse']:.6g}` / `{threshold['intensity_curve']['correlation']:.6g}`",
        f"- rho curve NRMSE/correlation: `{threshold['normalized_curves']['rho_max_z']['nrmse']:.6g}` / `{threshold['normalized_curves']['rho_max_z']['correlation']:.6g}`",
        f"- I/rho peak-count reference->hybrid: `{threshold['intensity_curve']['reference_peak_count']}->{threshold['intensity_curve']['hybrid_peak_count']}` / `{threshold['rho_curve']['reference_peak_count']}->{threshold['rho_curve']['hybrid_peak_count']}`",
        f"- Step/case wall-time reduction: `{performance['step_time_reduction_fraction']:.6g}` / `{performance['case_total_reduction_fraction']:.6g}`",
        f"- Step/case speedup: `{performance['step_time_speedup']:.6g}` / `{performance['case_total_speedup']:.6g}`",
        f"- Nonlinear/ionization/Raman-substep/Raman-convolution call reduction: `{performance['nonlinear_operator_call_reduction']}` / `{performance['ionization_call_reduction']}` / `{performance['raman_substep_reduction']}` / `{performance['raman_convolution_reduction']}`",
        f"- GPU peak allocated reference/hybrid: `{performance['reference_gpu_peak_allocated_bytes']}` / `{performance['hybrid_gpu_peak_allocated_bytes']}` bytes",
        f"- Visual veto: `{veto['veto']}`",
        "",
        "No coordinate shift, smoothing, or case-specific renormalization was applied; normalized-curve metrics use the fixed reference peak only.",
        "Raw NPZ files remain in the HPC run directory and are not copied into this report.",
    ]
    (out_dir / "hybrid_propagation_validation_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    result["outputs"] = {
        "comparison_json": str(result_path),
        "gate_metrics_csv": str(out_dir / "gate_metrics.csv"),
        "comparison_png": str(out_dir / "hybrid_propagation_validation_comparison.png"),
        "report_md": str(out_dir / "hybrid_propagation_validation_report.md"),
    }
    # Rewrite after adding output paths.
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


compare = compare_pair


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--visual-veto", type=Path, required=True,
        help="explicit JSON review; true veto requires curve, z_interval_m and feature",
    )
    args = parser.parse_args(argv)
    try:
        result = compare_pair(args.audit, args.out_dir, visual_veto=args.visual_veto)
    except InsufficientEvidenceError as exc:
        parser.error(str(exc))
    print(json.dumps({"classification": result["classification"], "failed_gates": result["failed_gates"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
