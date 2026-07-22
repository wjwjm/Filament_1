#!/usr/bin/env python3
"""Read-only Phase 8C filament-onset metric and effect-ledger builder.

The tool deliberately accepts incomplete inventories.  Missing curves stay missing:
it never edits source files, smooths data, shifts the focus, or infers a curve from
report prose.  All derived artifacts are written below the caller's output folder.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

CONTRACT_VERSION = "phase8c.metric_contract.v1"
THRESHOLDS = (1e19, 1e20, 1e21, 1e22)
FOCUS_M = 0.95
EPSILON_X_CM = 0.10


def focus_coordinate_cm(z_m: np.ndarray | float) -> np.ndarray | float:
    return 100.0 * (z_m - FOCUS_M)


def _none(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def _read_curve(repo_root: Path, record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source = record.get("curve_source")
    if not source:
        raise FileNotFoundError("no_curve_source")
    path = repo_root / source["path"]
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if source.get("format", "csv") != "csv":
        raise ValueError(f"unsupported curve format: {source.get('format')}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    rho_key = source["rho_key"]
    x_key = source.get("x_key")
    z_key = source.get("z_key")
    rho = np.asarray([float(row[rho_key]) for row in rows], dtype=float) * float(source.get("rho_scale_to_m3", 1.0))
    if z_key:
        z = np.asarray([float(row[z_key]) for row in rows], dtype=float)
        x = focus_coordinate_cm(z)
        x_source = f"derived:100*({z_key}-{FOCUS_M})"
        if x_key:
            archived_x = np.asarray([float(row[x_key]) for row in rows], dtype=float)
            mismatch = float(np.nanmax(np.abs(archived_x - x)))
        else:
            mismatch = None
    elif x_key:
        x = np.asarray([float(row[x_key]) for row in rows], dtype=float)
        x_source = x_key
        mismatch = None
    else:
        raise ValueError("curve source requires z_key or x_key")
    if len(x) != len(rho) or len(x) < 2:
        raise ValueError("curve has fewer than two aligned samples")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x axis is not strictly increasing")
    provenance = {
        "source_file": source["path"],
        "source_array_key": rho_key,
        "x_source": x_source,
        "archived_x_mismatch_cm": mismatch,
    }
    return x, rho, provenance


def _linear_cross(x0: float, y0: float, x1: float, y1: float, target: float) -> float:
    if y1 == y0:
        return x0
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


def first_crossing(x: np.ndarray, y: np.ndarray, target: float) -> tuple[float | None, str]:
    """First ascending crossing; an already-high first sample is left-censored."""
    finite = np.isfinite(x) & np.isfinite(y)
    first = int(np.flatnonzero(finite)[0]) if np.any(finite) else None
    if first is None:
        return None, "no_finite_samples"
    if y[first] >= target:
        return float(x[first]), "left_censored_at_first_sample"
    for index in range(first, len(x) - 1):
        if not (finite[index] and finite[index + 1]):
            continue
        if y[index] < target <= y[index + 1]:
            return _linear_cross(x[index], y[index], x[index + 1], y[index + 1], target), "interpolated"
    return None, "not_crossed"


def _descending_cross(x: np.ndarray, y: np.ndarray, target: float, start: int) -> tuple[float | None, str]:
    finite = np.isfinite(x) & np.isfinite(y)
    for index in range(start, len(x) - 1):
        if not (finite[index] and finite[index + 1]):
            continue
        if y[index] >= target > y[index + 1]:
            return _linear_cross(x[index], y[index], x[index + 1], y[index + 1], target), "interpolated"
    return None, "not_crossed"


def _value_at(x: np.ndarray, y: np.ndarray, point: float) -> float | None:
    if point < x[0] or point > x[-1]:
        return None
    index = int(np.searchsorted(x, point))
    if index == 0:
        return float(y[0]) if math.isfinite(float(y[0])) else None
    if index == len(x):
        return float(y[-1]) if math.isfinite(float(y[-1])) else None
    if not (math.isfinite(float(y[index - 1])) and math.isfinite(float(y[index]))):
        return None
    return _linear_cross(float(y[index - 1]), float(x[index - 1]), float(y[index]), float(x[index]), point)


def _interpolate_value(x: np.ndarray, y: np.ndarray, point: float) -> float | None:
    if point < x[0] or point > x[-1]:
        return None
    index = int(np.searchsorted(x, point))
    if index == 0:
        return _none(float(y[0]))
    if index == len(x):
        return _none(float(y[-1]))
    if not (math.isfinite(float(y[index - 1])) and math.isfinite(float(y[index]))):
        return None
    x0, x1 = float(x[index - 1]), float(x[index])
    y0, y1 = float(y[index - 1]), float(y[index])
    return float(y0 + (point - x0) * (y1 - y0) / (x1 - x0))


def _tail_integral(x: np.ndarray, y: np.ndarray, start: float) -> float | None:
    value = _interpolate_value(x, y, start)
    if value is None:
        return None
    index = int(np.searchsorted(x, start))
    tail_x = np.concatenate(([start], x[index:]))
    tail_y = np.concatenate(([value], y[index:]))
    if not np.all(np.isfinite(tail_y)):
        return None
    return float(np.trapezoid(tail_y, tail_x))


def curve_metrics(x: np.ndarray, rho: np.ndarray) -> dict[str, Any]:
    flags: list[str] = []
    finite = np.isfinite(x) & np.isfinite(rho)
    if not np.all(finite):
        flags.append("contains_nan_or_inf")
    if not np.any(finite):
        return {"calculation_status": "no_finite_samples", "data_quality_flags": flags}
    safe = np.where(finite, rho, -np.inf)
    peak_index = int(np.argmax(safe))
    peak = float(safe[peak_index])
    plateau = np.flatnonzero(finite & (rho == peak))
    plateau_left, plateau_right = int(plateau[0]), int(plateau[-1])
    plateau_width = float(x[plateau_right] - x[plateau_left])
    metrics: dict[str, Any] = {
        "calculation_status": "calculated",
        "data_quality_flags": flags,
        "peak_density_m3": peak,
        "peak_position_cm": float(x[peak_index]),
        "peak_plateau_width_cm": plateau_width,
    }
    for threshold in THRESHOLDS:
        value, status = first_crossing(x, rho, threshold)
        suffix = f"{threshold:.0e}".replace("+", "")
        metrics[f"crossing_{suffix}_cm"] = value
        metrics[f"crossing_{suffix}_status"] = status
    half = peak * 0.5
    left, left_status = first_crossing(x[: peak_index + 1], rho[: peak_index + 1], half)
    right, right_status = _descending_cross(x, rho, half, plateau_right)
    metrics.update({
        "left_halfmax_crossing_cm": left,
        "right_halfmax_crossing_cm": right,
        "fwhm_cm": None if left is None or right is None else float(right - left),
        "fwhm_status": "calculated" if left is not None and right is not None else f"left={left_status};right={right_status}",
    })
    rise10, rise10_status = first_crossing(x[: peak_index + 1], rho[: peak_index + 1], peak * 0.10)
    rise90, rise90_status = first_crossing(x[: peak_index + 1], rho[: peak_index + 1], peak * 0.90)
    fall90, fall90_status = _descending_cross(x, rho, peak * 0.90, plateau_right)
    fall50, fall50_status = _descending_cross(x, rho, peak * 0.50, plateau_right)
    fall10, fall10_status = _descending_cross(x, rho, peak * 0.10, plateau_right)
    metrics.update({
        "rise_10_position_cm": rise10,
        "rise_90_position_cm": rise90,
        "rise_10_90_cm": None if rise10 is None or rise90 is None else float(rise90 - rise10),
        "rise_status": f"10={rise10_status};90={rise90_status}",
        "fall_90_position_cm": fall90,
        "fall_50_position_cm": fall50,
        "fall_10_position_cm": fall10,
        "fall_90_10_cm": None if fall90 is None or fall10 is None else float(fall10 - fall90),
        "fall_status": f"90={fall90_status};50={fall50_status};10={fall10_status}",
    })
    for label, offset in (("after_peak", 0.0), ("peak_plus_5cm", 5.0), ("peak_plus_10cm", 10.0)):
        start = float(x[peak_index] + offset)
        metrics[f"tail_integral_{label}_m3_cm"] = _tail_integral(x, rho, start)
    metrics["rho_at_peak_plus_5cm_m3"] = _interpolate_value(x, rho, float(x[peak_index] + 5.0))
    metrics["rho_at_peak_plus_10cm_m3"] = _interpolate_value(x, rho, float(x[peak_index] + 10.0))
    return metrics


def comparison_metrics(x: np.ndarray, y: np.ndarray, px: np.ndarray, py: np.ndarray) -> dict[str, Any]:
    lo, hi = max(float(x[0]), float(px[0])), min(float(x[-1]), float(px[-1]))
    if lo >= hi:
        return {"rmse_linear_vs_pycap": None, "rmse_log_vs_pycap": None, "median_abs_log10_error_vs_pycap": None, "pycap_comparison_status": "no_common_interval"}
    grid = np.unique(np.concatenate((x[(x >= lo) & (x <= hi)], px[(px >= lo) & (px <= hi)])))
    a, b = np.interp(grid, x, y), np.interp(grid, px, py)
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if valid.sum() < 2:
        return {"rmse_linear_vs_pycap": None, "rmse_log_vs_pycap": None, "median_abs_log10_error_vs_pycap": None, "pycap_comparison_status": "insufficient_positive_common_samples"}
    log_error = np.log10(a[valid]) - np.log10(b[valid])
    return {
        "rmse_linear_vs_pycap": float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2))),
        "rmse_log_vs_pycap": float(np.sqrt(np.mean(log_error ** 2))),
        "median_abs_log10_error_vs_pycap": float(np.median(np.abs(log_error))),
        "pycap_comparison_status": "calculated_linear_interpolation_on_merged_common_grid",
    }


def _flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    return {key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else _none(value) for key, value in row.items()}


def normalize_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    defaults = inventory.get("defaults", {})
    return {**inventory, "results": [{**defaults, **record} for record in inventory["results"]]}


def build_metrics(repo_root: Path, inventory: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    provenances: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for record in inventory["results"]:
        row = {key: record.get(key) for key in ("result_id", "phase", "job_id", "physics_use_status", "data_integrity", "energy_admission")}
        row.update({"metric_definition_version": CONTRACT_VERSION, "source_file": None, "source_array_key": None})
        try:
            x, rho, provenance = _read_curve(repo_root, record)
            curves[record["result_id"]] = (x, rho)
            provenances[record["result_id"]] = provenance
            row.update(curve_metrics(x, rho))
            row.update(provenance)
        except (FileNotFoundError, ValueError) as exc:
            row.update({"calculation_status": "missing_or_incomplete", "data_quality_flags": [str(exc)]})
        rows.append(row)
    pycap_id = inventory.get("pycap_result_id")
    if pycap_id in curves:
        px, py = curves[pycap_id]
        pycap_metric = next(row for row in rows if row["result_id"] == pycap_id)
        for row in rows:
            if row["result_id"] == pycap_id or row.get("calculation_status") != "calculated":
                continue
            x, rho = curves[row["result_id"]]
            row.update(comparison_metrics(x, rho, px, py))
            row["delta_peak_position_vs_pycap_cm"] = _none_difference(row.get("peak_position_cm"), pycap_metric.get("peak_position_cm"))
            for threshold in THRESHOLDS:
                suffix = f"{threshold:.0e}".replace("+", "")
                row[f"delta_crossing_{suffix}_vs_pycap_cm"] = _none_difference(row.get(f"crossing_{suffix}_cm"), pycap_metric.get(f"crossing_{suffix}_cm"))
    return rows, curves


def _none_difference(left: Any, right: Any) -> float | None:
    return None if left is None or right is None else float(left - right)


def _effect_value(metrics: dict[str, dict[str, Any]], baseline: str, comparison: str, key: str) -> float | None:
    return _none_difference(metrics.get(comparison, {}).get(key), metrics.get(baseline, {}).get(key))


def build_effects(metrics_rows: list[dict[str, Any]], definitions: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = {row["result_id"]: row for row in metrics_rows}
    current_id = definitions["current_production_result_id"]
    pycap_id = definitions["pycap_result_id"]
    rows: list[dict[str, Any]] = []
    for definition in definitions["effects"]:
        baseline, comparison = definition.get("baseline_result_id"), definition.get("comparison_result_id")
        row = dict(definition)
        for threshold in THRESHOLDS:
            suffix = f"{threshold:.0e}".replace("+", "")
            delta = _effect_value(metrics, baseline, comparison, f"crossing_{suffix}_cm")
            total = _effect_value(metrics, pycap_id, current_id, f"crossing_{suffix}_cm")
            row[f"delta_crossing_{suffix}_cm"] = delta
            pycap_status = metrics.get(pycap_id, {}).get(f"crossing_{suffix}_status")
            current_status = metrics.get(current_id, {}).get(f"crossing_{suffix}_status")
            if delta is None or total is None or abs(total) < EPSILON_X_CM or delta * total <= 0 or pycap_status != "interpolated" or current_status != "interpolated":
                row[f"fraction_of_total_pycap_offset_{suffix}"] = None
            else:
                row[f"fraction_of_total_pycap_offset_{suffix}"] = float(delta / total)
        row["delta_peak_position_cm"] = _effect_value(metrics, baseline, comparison, "peak_position_cm")
        base_peak = metrics.get(baseline, {}).get("peak_density_m3")
        comp_peak = metrics.get(comparison, {}).get("peak_density_m3")
        row["peak_density_ratio"] = None if base_peak in (None, 0) or comp_peak is None else float(comp_peak / base_peak)
        row["delta_fwhm_cm"] = _effect_value(metrics, baseline, comparison, "fwhm_cm")
        row["delta_rise_width_cm"] = _effect_value(metrics, baseline, comparison, "rise_10_90_cm")
        row["delta_tail_metric"] = _effect_value(metrics, baseline, comparison, "tail_integral_after_peak_m3_cm")
        if row.get("causal_pair_quality") != "strict_single_delta" and row.get("confidence") == "high":
            row["confidence"] = "medium"
            row["limitations"] = (row.get("limitations", "") + "; confidence capped because this is not a strict single-delta pair").strip("; ")
        rows.append(row)
    return rows


def plot_artifacts(output_dir: Path, curves: dict[str, tuple[np.ndarray, np.ndarray]], metrics: list[dict[str, Any]], effects: list[dict[str, Any]], inventory: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {record["result_id"]: record.get("plot_label", record["result_id"]) for record in inventory["results"]}
    plot_ids = [record["result_id"] for record in inventory["results"] if record.get("plot_curve") and record["result_id"] in curves]
    fig, axis = plt.subplots(figsize=(9, 5.2))
    for result_id in plot_ids:
        x, rho = curves[result_id]
        axis.plot(x, rho, label=labels[result_id], linewidth=1.4)
    axis.set(xlabel=r"$x_{focus}$ (cm)", ylabel=r"$\rho_{max}$ (m$^{-3}$)", yscale="log", title="Available 120 fs density curves (unshifted)")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "figure1_curves_vs_pycap.png", dpi=180)
    fig.savefig(output_dir / "figure1_curves_vs_pycap.pdf")
    plt.close(fig)

    usable = [row for row in metrics if row.get("calculation_status") == "calculated"]
    fig, axis = plt.subplots(figsize=(9, 4.8))
    for index, row in enumerate(usable):
        for threshold, color in zip(THRESHOLDS, ("#4c78a8", "#f58518", "#54a24b", "#e45756")):
            suffix = f"{threshold:.0e}".replace("+", "")
            value = row.get(f"crossing_{suffix}_cm")
            if value is not None:
                axis.scatter(value, index, color=color, s=26)
    axis.set_yticks(range(len(usable)), [labels.get(row["result_id"], row["result_id"]) for row in usable])
    axis.set(xlabel=r"$x_{focus}$ (cm)", title="First density crossings (unshifted)")
    axis.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "figure2_crossing_positions.png", dpi=180)
    fig.savefig(output_dir / "figure2_crossing_positions.pdf")
    plt.close(fig)

    effect_rows = [row for row in effects if row.get("delta_crossing_1e21_cm") is not None]
    fig, axis = plt.subplots(figsize=(9, max(3.2, 0.55 * len(effects) + 1.4)))
    for index, row in enumerate(effects):
        value = row.get("delta_crossing_1e21_cm")
        if value is None:
            continue
        strict = row.get("causal_pair_quality") == "strict_single_delta"
        axis.scatter(value, index, marker="o" if strict else "o", facecolors="#4c78a8" if strict else "none", edgecolors="#4c78a8" if strict else "#777777", s=52)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_yticks(range(len(effects)), [row["effect_id"] for row in effects])
    axis.set(xlabel=r"$\Delta x$ at $10^{21}$ m$^{-3}$ (cm; comparison - baseline)", title="Effect ledger: onset shifts (not additive)")
    axis.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "figure3_effect_intervals.png", dpi=180)
    fig.savefig(output_dir / "figure3_effect_intervals.pdf")
    plt.close(fig)

    confidence_rank = {"high": 3, "medium": 2, "low": 1, "not_interpretable": 0}
    fig, axis = plt.subplots(figsize=(9, max(3.2, 0.55 * len(effects) + 1.4)))
    for index, row in enumerate(effects):
        value = row.get("delta_crossing_1e21_cm")
        marker = "o" if row.get("causal_pair_quality") == "strict_single_delta" else "s"
        axis.scatter(confidence_rank[row["confidence"]], index, s=64, marker=marker, color="#59a14f" if value is not None else "#9d9d9d")
        axis.text(3.12, index, "curve metric available" if value is not None else "missing/confounded", va="center", fontsize=8)
    axis.set(xlim=(-0.4, 5.7), xticks=[0, 1, 2, 3], xticklabels=["not interpretable", "low", "medium", "high"], yticks=range(len(effects)), yticklabels=[row["effect_id"] for row in effects], title="Effect-chain confidence and data state")
    axis.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "figure4_effect_confidence.png", dpi=180)
    fig.savefig(output_dir / "figure4_effect_confidence.pdf")
    plt.close(fig)
    _write_csv(output_dir / "plot_source_metrics.csv", [_flatten_for_csv(row) for row in metrics])
    _write_csv(output_dir / "plot_source_effects.csv", [_flatten_for_csv(row) for row in effects])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--effect-definitions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    inventory = normalize_inventory(json.loads(args.inventory.read_text(encoding="utf-8")))
    definitions = json.loads(args.effect_definitions.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, curves = build_metrics(repo_root, inventory)
    effects = build_effects(metrics, definitions)
    _write_json(args.output_dir / "effect_metrics_by_result.json", {"metric_definition_version": CONTRACT_VERSION, "rows": metrics})
    _write_csv(args.output_dir / "effect_metrics_by_result.csv", [_flatten_for_csv(row) for row in metrics])
    _write_json(args.output_dir / "result_inventory.json", inventory)
    _write_csv(args.output_dir / "result_inventory.csv", [_flatten_for_csv(row) for row in inventory["results"]])
    _write_json(args.output_dir / "filament_onset_effect_ledger.json", {"metric_definition_version": CONTRACT_VERSION, "rows": effects})
    _write_csv(args.output_dir / "filament_onset_effect_ledger.csv", [_flatten_for_csv(row) for row in effects])
    if args.plots:
        plot_artifacts(args.output_dir, curves, metrics, effects, inventory)


if __name__ == "__main__":
    main()
