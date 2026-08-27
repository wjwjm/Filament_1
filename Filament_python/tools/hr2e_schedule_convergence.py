"""HR-2E longitudinal schedule proposal and conservative convergence analysis.

Historical CSV inputs are proposal-only.  Final convergence decisions require
HR-2D canonical interval ledgers from new NPZ outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


CHANNEL_KEYS = {
    "ion": ("E_dep_ion_interval_J", "E_dep_ion_pulse_J"),
    "ib": ("E_dep_ib_interval_J", "E_dep_ib_pulse_J"),
    "raman": ("E_dep_raman_interval_J", "E_dep_raman_pulse_J"),
    "total": ("E_dep_total_interval_J", "E_dep_total_pulse_J"),
}
PRIMARY_CHANNELS = ("ion", "ib", "raman", "total")
ROUND_OFF_TAIL_FRACTION = 0.1


def _scalar(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"missing canonical field: {key}")
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar")
    return value.reshape(()).item()


def _bool(data: Mapping[str, Any], key: str) -> bool:
    return bool(_scalar(data, key))


def _text(data: Mapping[str, Any], key: str) -> str:
    value = _scalar(data, key)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def validate_schedule_arrays(
    z_edges: Iterable[float], dz_intervals: Iterable[float], *, atol: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(tuple(z_edges), dtype=np.float64)
    dz = np.asarray(tuple(dz_intervals), dtype=np.float64)
    if edges.ndim != 1 or dz.ndim != 1 or edges.size != dz.size + 1:
        raise ValueError("z_edges/dz_intervals length mismatch")
    if not np.all(np.isfinite(edges)) or not np.all(np.isfinite(dz)):
        raise ValueError("schedule arrays must be finite")
    differences = np.diff(edges)
    if np.any(differences <= 0.0) or np.any(dz <= 0.0):
        raise ValueError("schedule intervals must be strictly positive")
    if not np.allclose(differences, dz, rtol=1e-11, atol=atol):
        raise ValueError("dz_intervals do not match z_edges")
    return edges, dz


def schedule_summary(
    z_edges: Iterable[float],
    dz_intervals: Iterable[float],
    *,
    base_dz: float,
    focus_dz: float,
) -> dict[str, Any]:
    edges, dz = validate_schedule_arrays(z_edges, dz_intervals)
    representative_min = min(float(base_dz), float(focus_dz))
    tail_mask = dz < representative_min * ROUND_OFF_TAIL_FRACTION
    return {
        "z_start_m": float(edges[0]),
        "z_end_m": float(edges[-1]),
        "n_intervals": int(dz.size),
        "base_dz_m": float(base_dz),
        "focus_dz_m": float(focus_dz),
        "roundoff_tail_present": bool(np.any(tail_mask)),
        "roundoff_tail_count": int(np.count_nonzero(tail_mask)),
        "roundoff_tail_min_dz_m": (
            float(np.min(dz[tail_mask])) if np.any(tail_mask) else None
        ),
    }


def conservative_remap(
    source_edges: Iterable[float],
    source_interval_energy_J: Iterable[float],
    target_edges: Iterable[float],
    *,
    atol: float = 1e-12,
) -> np.ndarray:
    """Conservatively integrate piecewise-constant J/m onto target bins."""
    src_edges = np.asarray(tuple(source_edges), dtype=np.float64)
    src_energy = np.asarray(tuple(source_interval_energy_J), dtype=np.float64)
    tgt_edges = np.asarray(tuple(target_edges), dtype=np.float64)
    src_edges, src_dz = validate_schedule_arrays(src_edges, np.diff(src_edges), atol=atol)
    tgt_edges, _ = validate_schedule_arrays(tgt_edges, np.diff(tgt_edges), atol=atol)
    if src_energy.ndim != 1 or src_energy.size != src_dz.size:
        raise ValueError("source interval energy length mismatch")
    if not np.all(np.isfinite(src_energy)) or np.any(src_energy < 0.0):
        raise ValueError("source interval energy must be finite and non-negative")
    if not (
        math.isclose(src_edges[0], tgt_edges[0], rel_tol=0.0, abs_tol=atol)
        and math.isclose(src_edges[-1], tgt_edges[-1], rel_tol=0.0, abs_tol=atol)
    ):
        raise ValueError("source and target schedules must span the same z range")

    density = src_energy / src_dz
    result = np.zeros(tgt_edges.size - 1, dtype=np.float64)
    source_index = 0
    for target_index, (left, right) in enumerate(zip(tgt_edges[:-1], tgt_edges[1:])):
        while source_index < src_energy.size and src_edges[source_index + 1] <= left:
            source_index += 1
        probe = source_index
        while probe < src_energy.size and src_edges[probe] < right:
            overlap = min(right, src_edges[probe + 1]) - max(left, src_edges[probe])
            if overlap > 0.0:
                result[target_index] += density[probe] * overlap
            if src_edges[probe + 1] >= right:
                break
            probe += 1
    if not math.isclose(
        float(result.sum()), float(src_energy.sum()), rel_tol=2e-11, abs_tol=1e-18
    ):
        raise ValueError("conservative remap failed energy conservation")
    return result


def union_edges(first: Iterable[float], second: Iterable[float]) -> np.ndarray:
    a = np.asarray(tuple(first), dtype=np.float64)
    b = np.asarray(tuple(second), dtype=np.float64)
    validate_schedule_arrays(a, np.diff(a))
    validate_schedule_arrays(b, np.diff(b))
    if not (math.isclose(a[0], b[0], abs_tol=1e-12) and math.isclose(a[-1], b[-1], abs_tol=1e-12)):
        raise ValueError("schedules do not share endpoints")
    values = np.unique(np.concatenate((a, b)))
    values[0] = a[0]
    values[-1] = a[-1]
    return values


def cumulative_curve(interval_energy_J: Iterable[float]) -> np.ndarray:
    energy = np.asarray(tuple(interval_energy_J), dtype=np.float64)
    total = float(energy.sum())
    if total <= 0.0:
        return np.zeros(energy.size + 1, dtype=np.float64)
    return np.concatenate(([0.0], np.cumsum(energy) / total))


def _quantile_location(edges: np.ndarray, energy: np.ndarray, quantile: float) -> float | None:
    total = float(energy.sum())
    if total <= 0.0:
        return None
    target = quantile * total
    cumulative = np.cumsum(energy)
    index = int(np.searchsorted(cumulative, target, side="left"))
    before = float(cumulative[index - 1]) if index else 0.0
    bin_energy = float(energy[index])
    fraction = 0.0 if bin_energy <= 0.0 else (target - before) / bin_energy
    return float(edges[index] + fraction * (edges[index + 1] - edges[index]))


def profile_metrics(edges: Iterable[float], interval_energy_J: Iterable[float]) -> dict[str, Any]:
    edge_array = np.asarray(tuple(edges), dtype=np.float64)
    energy = np.asarray(tuple(interval_energy_J), dtype=np.float64)
    edge_array, dz = validate_schedule_arrays(edge_array, np.diff(edge_array))
    if energy.ndim != 1 or energy.size != dz.size:
        raise ValueError("profile energy length mismatch")
    if not np.all(np.isfinite(energy)) or np.any(energy < 0.0):
        raise ValueError("profile energy must be finite and non-negative")
    total = float(energy.sum())
    if total <= 0.0:
        return {
            "pulse_energy_J": 0.0,
            "z10_m": None,
            "z50_m": None,
            "z90_m": None,
            "centroid_m": None,
            "peak_dE_dz_J_per_m": 0.0,
            "peak_z_m": None,
            "peak_tie_count": 0,
        }
    density = energy / dz
    peak = float(np.max(density))
    peak_indices = np.flatnonzero(np.isclose(density, peak, rtol=1e-12, atol=0.0))
    peak_index = int(peak_indices[0])
    midpoint = 0.5 * (edge_array[:-1] + edge_array[1:])
    return {
        "pulse_energy_J": total,
        "z10_m": _quantile_location(edge_array, energy, 0.10),
        "z50_m": _quantile_location(edge_array, energy, 0.50),
        "z90_m": _quantile_location(edge_array, energy, 0.90),
        "centroid_m": float(np.sum(energy * midpoint) / total),
        "peak_dE_dz_J_per_m": peak,
        "peak_z_m": float(midpoint[peak_index]),
        "peak_tie_count": int(peak_indices.size),
    }


def _local_dz(edges: np.ndarray, location: float | None) -> float | None:
    if location is None:
        return None
    index = int(np.searchsorted(edges, location, side="right") - 1)
    index = min(max(index, 0), edges.size - 2)
    return float(edges[index + 1] - edges[index])


def validate_canonical_mapping(data: Mapping[str, Any], *, label: str = "run") -> dict[str, Any]:
    edges = np.asarray(data.get("z_edges"), dtype=np.float64)
    dz = np.asarray(data.get("dz_intervals"), dtype=np.float64)
    edges, dz = validate_schedule_arrays(edges, dz)
    n_intervals = int(_scalar(data, "n_intervals"))
    if n_intervals != dz.size:
        raise ValueError(f"{label}: n_intervals mismatch")
    if not _bool(data, "deposition_level1_all_available_mechanism_closure_pass"):
        raise ValueError(f"{label}: Level-1 deposition closure failed")
    if not _bool(data, "deposition_level2_all_available_mechanism_closure_pass"):
        raise ValueError(f"{label}: Level-2 deposition closure failed")
    if _text(data, "E_dep_total_level2_closure_status") != "pass":
        raise ValueError(f"{label}: total Level-2 deposition closure is not pass")
    field_in = float(_scalar(data, "E_field_in_J"))
    if not math.isfinite(field_in) or field_in <= 0.0:
        raise ValueError(f"{label}: invalid E_field_in_J")

    channels: dict[str, np.ndarray] = {}
    pulse: dict[str, float] = {}
    for channel, (array_key, pulse_key) in CHANNEL_KEYS.items():
        values = np.asarray(data[array_key], dtype=np.float64)
        if values.ndim != 1 or values.size != n_intervals:
            raise ValueError(f"{label}: {array_key} length mismatch")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label}: {array_key} is not finite")
        if np.any(values < -1e-18):
            raise ValueError(f"{label}: {array_key} contains negative deposition")
        values = np.maximum(values, 0.0)
        scalar = float(_scalar(data, pulse_key))
        if not math.isfinite(scalar) or not math.isclose(
            scalar, float(values.sum()), rel_tol=2e-10, abs_tol=1e-18
        ):
            raise ValueError(f"{label}: {pulse_key} does not match interval sum")
        channels[channel] = values
        pulse[channel] = scalar

    if not np.allclose(
        channels["total"],
        channels["ion"] + channels["ib"] + channels["raman"],
        rtol=2e-10,
        atol=1e-18,
    ):
        raise ValueError(f"{label}: total interval ledger does not equal channel sum")
    if not _bool(data, "total_deposition_authoritative"):
        raise ValueError(f"{label}: total deposition is not authoritative")
    if not _bool(data, "deposition_raman_authoritative"):
        raise ValueError(f"{label}: Raman deposition is not authoritative")
    if _text(data, "raman_deposition_source") != "actual_field_fluence_loss":
        raise ValueError(f"{label}: Raman source is not actual_field_fluence_loss")
    applied = np.asarray(data.get("raman_operator_applied"), dtype=bool)
    if applied.ndim != 1 or applied.size != n_intervals or not np.all(applied):
        raise ValueError(f"{label}: full Raman operator was not applied throughout")
    if _text(data, "deposition_raman_level1_closure_status") != "pass":
        raise ValueError(f"{label}: Raman Level-1 closure is not pass")
    if _text(data, "deposition_raman_level2_closure_status") != "pass":
        raise ValueError(f"{label}: Raman Level-2 closure is not pass")
    if not _bool(data, "field_energy_bookkeeping_authoritative"):
        raise ValueError(f"{label}: field-energy bookkeeping is not authoritative")
    if _text(data, "field_energy_bookkeeping_status") != "available":
        raise ValueError(f"{label}: field-energy bookkeeping is unavailable")
    field_guardrail = {}
    for key in (
        "E_field_out_J",
        "E_field_loss_J",
        "E_dep_accounted_authoritative_J",
        "E_field_energy_bookkeeping_residual_J",
        "E_field_energy_bookkeeping_relative_residual",
    ):
        value = float(_scalar(data, key))
        if not math.isfinite(value):
            raise ValueError(f"{label}: {key} is not finite")
        field_guardrail[key] = value
    return {
        "label": label,
        "z_edges": edges,
        "dz_intervals": dz,
        "n_intervals": n_intervals,
        "field_in_J": field_in,
        "channels": channels,
        "pulse": pulse,
        "field_energy_guardrail": field_guardrail,
    }


def load_canonical_npz(path: str | Path, *, label: str | None = None) -> dict[str, Any]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as loaded:
        copied = {key: loaded[key].copy() for key in loaded.files}
    result = validate_canonical_mapping(copied, label=label or source.stem)
    result["path"] = str(source)
    return result


def load_case_with_provenance(
    npz_path: str | Path,
    metadata_path: str | Path,
    manifest_path: str | Path,
    *,
    case_id: str,
) -> dict[str, Any]:
    run = load_canonical_npz(npz_path, label=case_id)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if metadata.get("schema") != "khz_filament.hr2e.job_metadata.v1":
        raise ValueError(f"{case_id}: invalid job metadata schema")
    if manifest.get("schema") != "khz_filament.hr2e.stage1_preflight.v1":
        raise ValueError(f"{case_id}: invalid preflight manifest schema")
    expected = next((case for case in manifest.get("cases", []) if case.get("case_id") == case_id), None)
    if expected is None:
        raise ValueError(f"{case_id}: case is absent from preflight manifest")
    for key in ("case_id", "config_sha256", "dtype"):
        if str(metadata.get(key)) != str(expected.get(key)):
            raise ValueError(f"{case_id}: metadata mismatch for {key}")
    git_sha = str(metadata.get("git_sha", ""))
    if len(git_sha) != 40 or any(character not in "0123456789abcdef" for character in git_sha.lower()):
        raise ValueError(f"{case_id}: invalid execution Git SHA")
    expected_suffix = "/Filament_python/" + str(expected["config_path"])
    actual_config = str(metadata.get("config_path", "")).replace("\\", "/")
    if not actual_config.endswith(expected_suffix):
        raise ValueError(f"{case_id}: config path does not match manifest")
    actual_config_path = Path(actual_config)
    if not actual_config_path.is_file():
        raise ValueError(f"{case_id}: bound config file is unavailable")
    actual_config_sha = hashlib.sha256(actual_config_path.read_bytes()).hexdigest()
    if actual_config_sha != str(expected["config_sha256"]):
        raise ValueError(f"{case_id}: bound config file hash mismatch")
    run.update({
        "case_id": case_id,
        "git_sha": git_sha,
        "config_sha256": str(metadata["config_sha256"]),
        "dtype": str(metadata["dtype"]),
        "pulse_width": str(expected["pulse_width"]),
        "schedule": str(expected["schedule"]),
        "raman_mode": str(expected["raman_mode"]),
        "metadata_path": str(metadata_path),
    })
    return run


def validate_comparison_provenance(runs: Mapping[str, Mapping[str, Any]]) -> None:
    if not runs:
        raise ValueError("no runs supplied for provenance validation")
    for label, run in runs.items():
        if run.get("schedule") != label:
            raise ValueError(f"{label}: schedule label mismatch")
    for key in ("git_sha", "dtype", "pulse_width", "raman_mode"):
        values = {str(run.get(key)) for run in runs.values()}
        if len(values) != 1:
            raise ValueError(f"comparison provenance mismatch: {key}")
    if next(iter(runs.values()))["raman_mode"] != "full_isaacs_eq27":
        raise ValueError("comparison does not use the validated full Isaacs Raman mode")


def validate_execution_manifest(
    runs: Mapping[str, Mapping[str, Any]],
    preflight_manifest_path: str | Path,
    execution_manifest_path: str | Path,
) -> None:
    preflight_path = Path(preflight_manifest_path)
    execution = json.loads(Path(execution_manifest_path).read_text(encoding="utf-8"))
    if execution.get("schema") != "khz_filament.hr2e.execution_manifest.v1":
        raise ValueError("invalid HR-2E execution manifest schema")
    expected_manifest_sha = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
    if execution.get("preflight_manifest_sha256") != expected_manifest_sha:
        raise ValueError("execution manifest does not bind the preflight manifest")
    git_values = {str(run["git_sha"]) for run in runs.values()}
    if git_values != {str(execution.get("expected_git_sha", ""))}:
        raise ValueError("execution manifest Git SHA does not match job metadata")
    expected_cases = set(execution.get("case_ids", []))
    actual_cases = {str(run["case_id"]) for run in runs.values()}
    if not actual_cases.issubset(expected_cases):
        raise ValueError("execution manifest does not include every compared case")


def compare_channel(
    candidate: Mapping[str, Any], fine: Mapping[str, Any], channel: str
) -> dict[str, Any]:
    candidate_edges = np.asarray(candidate["z_edges"], dtype=np.float64)
    fine_edges = np.asarray(fine["z_edges"], dtype=np.float64)
    candidate_energy = np.asarray(candidate["channels"][channel], dtype=np.float64)
    fine_energy = np.asarray(fine["channels"][channel], dtype=np.float64)
    candidate_metrics = profile_metrics(candidate_edges, candidate_energy)
    fine_metrics = profile_metrics(fine_edges, fine_energy)
    common = union_edges(candidate_edges, fine_edges)
    candidate_common = conservative_remap(candidate_edges, candidate_energy, common)
    fine_common = conservative_remap(fine_edges, fine_energy, common)
    cumulative_error = float(
        np.max(np.abs(cumulative_curve(candidate_common) - cumulative_curve(fine_common)))
    )
    fine_pulse = fine_metrics["pulse_energy_J"]
    candidate_pulse = candidate_metrics["pulse_energy_J"]
    field_reference = max(float(candidate["field_in_J"]), float(fine["field_in_J"]))
    zero_channel = candidate_pulse == 0.0 and fine_pulse == 0.0
    negligible = max(candidate_pulse, fine_pulse) / field_reference < 1e-6
    if zero_channel:
        energy_error = 0.0
        energy_error_kind = "trivially_zero"
        energy_gate = True
    elif negligible:
        energy_error = abs(candidate_pulse - fine_pulse) / field_reference
        energy_error_kind = "absolute_over_field_in"
        energy_gate = energy_error <= 1e-6
    else:
        energy_error = abs(candidate_pulse - fine_pulse) / abs(fine_pulse)
        energy_error_kind = "relative"
        energy_gate = energy_error <= 0.01

    shifts: dict[str, float | None] = {}
    location_gate = True
    for key in ("z10_m", "z50_m", "z90_m", "centroid_m"):
        c_value = candidate_metrics[key]
        f_value = fine_metrics[key]
        shift = None if c_value is None or f_value is None else abs(c_value - f_value)
        shifts[key.replace("_m", "_shift_m")] = shift
        local = _local_dz(candidate_edges, c_value)
        if shift is not None and local is not None and shift > 2.0 * local + 1e-12:
            location_gate = False

    c_peak = candidate_metrics["peak_dE_dz_J_per_m"]
    f_peak = fine_metrics["peak_dE_dz_J_per_m"]
    peak_relative = (
        0.0 if c_peak == 0.0 and f_peak == 0.0
        else abs(c_peak - f_peak) / abs(f_peak) if f_peak != 0.0
        else math.inf
    )
    peak_z_shift = (
        None
        if candidate_metrics["peak_z_m"] is None or fine_metrics["peak_z_m"] is None
        else abs(candidate_metrics["peak_z_m"] - fine_metrics["peak_z_m"])
    )
    peak_local = _local_dz(candidate_edges, candidate_metrics["peak_z_m"])
    peak_gate = (
        peak_relative <= 0.05
        and (peak_z_shift is None or peak_local is None or peak_z_shift <= 2.0 * peak_local + 1e-12)
    )
    primary_pass = energy_gate and cumulative_error <= 0.02 and location_gate
    return {
        "channel": channel,
        "classification": "zero_channel" if zero_channel else "negligible_channel" if negligible else "authoritative_channel",
        "candidate": candidate_metrics,
        "fine": fine_metrics,
        "pulse_energy_error": energy_error,
        "pulse_energy_error_kind": energy_error_kind,
        "max_cumulative_shape_error": cumulative_error,
        **shifts,
        "peak_dE_dz_relative_difference": peak_relative,
        "peak_z_shift_m": peak_z_shift,
        "gate_A_pulse_energy": energy_gate,
        "gate_B_cumulative_shape": cumulative_error <= 0.02,
        "gate_C_locations": location_gate,
        "gate_D_peak_secondary": peak_gate,
        "primary_pass": primary_pass,
        "pass": primary_pass,
    }


def compare_runs(candidate: Mapping[str, Any], fine: Mapping[str, Any]) -> dict[str, Any]:
    comparisons = {
        channel: compare_channel(candidate, fine, channel)
        for channel in CHANNEL_KEYS
    }
    return {
        "schema": "khz_filament.hr2e.schedule_comparison.v1",
        "candidate_label": candidate["label"],
        "fine_label": fine["label"],
        "channels": comparisons,
        "primary_pass": all(comparisons[channel]["primary_pass"] for channel in PRIMARY_CHANNELS),
        "ib_limiting": not comparisons["ib"]["primary_pass"],
    }


def compare_triplet(
    coarse: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
    fine: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_vs_fine = compare_runs(candidate, fine)
    coarse_vs_candidate = None if coarse is None else compare_runs(coarse, candidate)
    trend: dict[str, Any] = {}
    if coarse_vs_candidate is not None:
        for channel in CHANNEL_KEYS:
            first = coarse_vs_candidate["channels"][channel]
            second = candidate_vs_fine["channels"][channel]
            trend[channel] = {
                "pulse_energy_error_decreased": (
                    second["pulse_energy_error"] <= first["pulse_energy_error"] + 1e-15
                ),
                "cumulative_shape_error_decreased": (
                    second["max_cumulative_shape_error"]
                    <= first["max_cumulative_shape_error"] + 1e-15
                ),
                "strict_monotonicity_required": False,
            }
    return {
        "schema": "khz_filament.hr2e.schedule_comparison_set.v1",
        "coarse_vs_candidate": coarse_vs_candidate,
        "candidate_vs_fine": candidate_vs_fine,
        "convergence_trend": trend,
        "primary_pass": candidate_vs_fine["primary_pass"],
    }
def historical_proposal(
    csv_path: str | Path,
    *,
    current_window: tuple[float, float] = (0.85, 1.05),
    proposed_window: tuple[float, float] = (0.75, 1.05),
) -> dict[str, Any]:
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"z_m", "E_dep_z", "dz_used_z"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path}: missing historical columns {sorted(required)}")
    z = np.asarray([float(row["z_m"]) for row in rows], dtype=np.float64)
    energy = np.asarray([float(row["E_dep_z"]) for row in rows], dtype=np.float64)
    dz = np.asarray([float(row["dz_used_z"]) for row in rows], dtype=np.float64)
    if (
        not np.all(np.isfinite(z))
        or not np.all(np.isfinite(energy))
        or not np.all(np.isfinite(dz))
        or np.any(np.diff(z) <= 0.0)
        or np.any(dz <= 0.0)
        or np.any(energy < -1e-18)
    ):
        raise ValueError(f"{path}: invalid historical diagnostics")
    energy = np.maximum(energy, 0.0)
    total = float(energy.sum())
    if total <= 0.0:
        raise ValueError(f"{path}: historical deposition proxy is zero")
    density = energy / dz
    peak_index = int(np.argmax(density))
    normalized_gradient = np.abs(np.diff(density) / np.diff(z)) / float(density[peak_index])
    cumulative = np.cumsum(energy) / total

    def qloc(q: float) -> float:
        return float(z[int(np.searchsorted(cumulative, q, side="left"))])

    def coverage(window: tuple[float, float]) -> float:
        return float(energy[(z >= window[0]) & (z <= window[1])].sum() / total)

    return {
        "schema": "khz_filament.hr2e.historical_schedule_proposal.v1",
        "proposal_only": True,
        "authoritative_convergence_evidence": False,
        "source_csv": str(path),
        "z_quantiles_m": {f"z{int(q * 100):02d}": qloc(q) for q in (0.01, 0.10, 0.50, 0.90, 0.99)},
        "peak_dE_dz_z_m": float(z[peak_index]),
        "max_normalized_gradient_per_m": float(np.max(normalized_gradient)),
        "current_window_m": list(current_window),
        "current_window_energy_fraction": coverage(current_window),
        "proposed_window_m": list(proposed_window),
        "proposed_window_energy_fraction": coverage(proposed_window),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(dict(payload)), indent=2) + "\n", encoding="utf-8")


def _write_comparison_csv(path: Path, comparison_set: Mapping[str, Any]) -> None:
    fields = [
        "comparison", "channel", "classification", "pulse_energy_error", "pulse_energy_error_kind",
        "max_cumulative_shape_error", "z10_shift_m", "z50_shift_m", "z90_shift_m",
        "centroid_shift_m", "peak_dE_dz_relative_difference", "peak_z_shift_m",
        "gate_A_pulse_energy", "gate_B_cumulative_shape", "gate_C_locations",
        "gate_D_peak_secondary", "primary_pass",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for comparison_name in ("coarse_vs_candidate", "candidate_vs_fine"):
            comparison = comparison_set.get(comparison_name)
            if comparison is None:
                continue
            for row in comparison["channels"].values():
                output = {key: row.get(key) for key in fields}
                output["comparison"] = comparison_name
                writer.writerow(output)


def _write_plots(
    output_dir: Path,
    runs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"status": "unavailable", "reason": type(exc).__name__, "files": []}
    files: list[str] = []
    for channel in CHANNEL_KEYS:
        figure, axes = plt.subplots(2, 1, figsize=(7.2, 7.5), sharex=True)
        for label, run in runs.items():
            edges = np.asarray(run["z_edges"], dtype=np.float64)
            energy = np.asarray(run["channels"][channel], dtype=np.float64)
            dz = np.diff(edges)
            midpoint = 0.5 * (edges[:-1] + edges[1:])
            axes[0].plot(midpoint, energy / dz, label=label, linewidth=1.2)
            axes[1].plot(edges, cumulative_curve(energy), label=label, linewidth=1.2)
        axes[0].set_ylabel("dE_dep/dz [J/m]")
        axes[1].set_ylabel("cumulative fraction")
        axes[1].set_xlabel("z [m]")
        axes[0].set_title(f"HR-2E {channel} longitudinal deposition")
        axes[0].legend()
        axes[1].legend()
        axes[0].grid(alpha=0.25)
        axes[1].grid(alpha=0.25)
        figure.tight_layout()
        path = output_dir / f"{channel}_longitudinal_convergence.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        files.append(path.name)
    return {"status": "created", "files": files}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    historical = subparsers.add_parser("historical", help="proposal-only historical analysis")
    historical.add_argument("--input", action="append", required=True, type=Path)
    historical.add_argument("--output", required=True, type=Path)
    compare = subparsers.add_parser("compare", help="compare canonical candidate and fine NPZ")
    compare.add_argument("--coarse", type=Path)
    compare.add_argument("--coarse-metadata", type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument("--candidate-metadata", required=True, type=Path)
    compare.add_argument("--fine", required=True, type=Path)
    compare.add_argument("--fine-metadata", required=True, type=Path)
    compare.add_argument("--manifest", required=True, type=Path)
    compare.add_argument("--execution-manifest", required=True, type=Path)
    compare.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.command == "historical":
        records = [historical_proposal(path) for path in args.input]
        worst = max(records, key=lambda row: row["max_normalized_gradient_per_m"])
        _write_json(args.output, {
            "schema": "khz_filament.hr2e.historical_schedule_proposal_set.v1",
            "proposal_only": True,
            "records": records,
            "worst_case_source_csv": worst["source_csv"],
            "suggested_focus_center_m": 0.90,
            "suggested_focus_halfwidth_m": 0.15,
        })
        return 0

    if (args.coarse is None) != (args.coarse_metadata is None):
        parser.error("--coarse and --coarse-metadata must be supplied together")
    pulse = "40fs" if "40fs" in args.candidate.name else "120fs" if "120fs" in args.candidate.name else None
    if pulse is None:
        raise ValueError("candidate filename must identify 40fs or 120fs")
    coarse = None if args.coarse is None else load_case_with_provenance(
        args.coarse, args.coarse_metadata, args.manifest, case_id=f"hr2e_{pulse}_coarse"
    )
    candidate = load_case_with_provenance(
        args.candidate, args.candidate_metadata, args.manifest, case_id=f"hr2e_{pulse}_candidate"
    )
    fine = load_case_with_provenance(
        args.fine, args.fine_metadata, args.manifest, case_id=f"hr2e_{pulse}_fine"
    )
    provenance_runs = {
        key: value for key, value in (("coarse", coarse), ("candidate", candidate), ("fine", fine)) if value is not None
    }
    validate_comparison_provenance(provenance_runs)
    validate_execution_manifest(provenance_runs, args.manifest, args.execution_manifest)
    result = compare_triplet(coarse, candidate, fine)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result["plots"] = _write_plots(
        args.output_dir,
        {key: value for key, value in (("coarse", coarse), ("candidate", candidate), ("fine", fine)) if value is not None},
    )
    _write_json(args.output_dir / "schedule_comparison.json", result)
    _write_comparison_csv(args.output_dir / "schedule_comparison.csv", result)
    return 0 if result["primary_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
