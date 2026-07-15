#!/usr/bin/env python3
"""Create a controlled-comparison report for the transverse-profile stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.size != 1:
        return array
    value = array.reshape(()).item()
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def _finite_array(data: dict[str, Any], key: str) -> np.ndarray | None:
    if key not in data:
        return None
    values = np.asarray(data[key], dtype=float).reshape(-1)
    return values if values.size else None


def _filament_metrics(result: dict[str, Any], threshold_m3: float) -> dict[str, Any]:
    z = _finite_array(result, "z_axis")
    rho = _finite_array(result, "rho_onaxis_max_z")
    imax = _finite_array(result, "I_max_z")
    fwhm = _finite_array(result, "fwhm_plasma_z")
    if z is None or rho is None or z.size != rho.size:
        return {"status": "missing_rho_or_z"}
    mask = np.isfinite(rho) & (rho >= threshold_m3)
    hit = np.flatnonzero(mask)
    peak_index = int(np.nanargmax(rho)) if np.any(np.isfinite(rho)) else None
    metrics: dict[str, Any] = {
        "status": "detected" if hit.size else "threshold_not_reached",
        "rho_threshold_m3": threshold_m3,
        "rho_peak_m3": float(rho[peak_index]) if peak_index is not None else None,
        "z_rho_peak_m": float(z[peak_index]) if peak_index is not None else None,
        "I_max_peak_W_m2": float(np.nanmax(imax)) if imax is not None and np.any(np.isfinite(imax)) else None,
        "fwhm_plasma_at_rho_peak_m": float(fwhm[peak_index]) if fwhm is not None and peak_index is not None and fwhm.size == z.size else None,
    }
    if hit.size:
        metrics.update({
            "z_on_m": float(z[hit[0]]),
            "z_end_m": float(z[hit[-1]]),
            "filament_length_m": float(z[hit[-1]] - z[hit[0]]),
        })
    else:
        metrics.update({"z_on_m": None, "z_end_m": None, "filament_length_m": 0.0})
    return metrics


def _quality_gates(result: dict[str, Any], gates: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    z = _finite_array(result, "z_axis")
    if gates.get("require_strictly_increasing_z") and (z is None or not np.all(np.diff(z) > 0.0)):
        failures.append("z_axis is not strictly increasing")
    energy = _finite_array(result, "U_z")
    if energy is None or energy.size == 0 or not np.isfinite(energy[0]) or energy[0] == 0.0:
        failures.append("U_z is unavailable for energy gate")
    elif np.nanmax(energy) / energy[0] - 1.0 > float(gates["maximum_energy_growth_fraction"]):
        failures.append("energy growth exceeds the configured limit")
    imax = _finite_array(result, "I_max_z")
    if imax is None or imax.size < 2:
        failures.append("I_max_z is unavailable for intensity-growth gate")
    else:
        positive = imax[:-1] > 0.0
        if np.any(positive) and np.nanmax(imax[1:][positive] / imax[:-1][positive]) > float(gates["maximum_adjacent_intensity_growth"]):
            failures.append("adjacent intensity growth exceeds the configured limit")
    rho = _finite_array(result, "rho_onaxis_max_z")
    if rho is None or not np.any(np.isfinite(rho)):
        failures.append("rho_onaxis_max_z is unavailable")
    elif np.nanmax(rho) > float(gates["maximum_electron_density_m3"]):
        failures.append("electron density exceeds the configured neutral-density envelope")
    fwhm = _finite_array(result, "fwhm_plasma_z")
    if gates.get("require_positive_fwhm") and (fwhm is None or not np.all(np.isfinite(fwhm) & (fwhm > 0.0))):
        failures.append("fwhm_plasma_z contains non-positive or non-finite values")
    return failures


def _profile_figure(cases: list[tuple[dict[str, Any], dict[str, Any]]], output: Path) -> str:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for case, result in cases:
        x = _finite_array(result, "input_profile_x")
        intensity = _finite_array(result, "input_profile_center_I")
        if x is None or intensity is None or x.size != intensity.size or not np.any(np.isfinite(intensity)):
            continue
        peak = float(np.nanmax(intensity))
        if peak > 0.0:
            ax.plot(x * 1e3, intensity / peak, linewidth=1.5, label=case["label"])
    ax.set(xlabel="x (mm)", ylabel="normalized center-line input intensity", title="Input transverse profiles")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output.name


def finalize_stage(stage_dir: str | Path) -> dict[str, Any]:
    stage_dir = Path(stage_dir)
    spec = json.loads((stage_dir / "stage_spec_snapshot.json").read_text(encoding="utf-8"))
    threshold = float(spec["filament_threshold_m3"])
    reports_dir = stage_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    case_results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    case_reports: dict[str, Any] = {}
    complete = True
    quality_pass = True
    for case in spec["cases"]:
        case_dir = stage_dir / "cases" / case["case_id"]
        metadata_path = case_dir / "run_metadata.json"
        npz_path = case_dir / "result.npz"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        if metadata.get("status") != "completed" or not npz_path.is_file():
            complete = False
            case_reports[case["case_id"]] = {"metadata": metadata, "status": "incomplete"}
            continue
        result = _load_npz(npz_path)
        profile_type = _scalar(result.get("input_profile_type", case["profile_type"]))
        input_summary = {
            key: _scalar(result[key]) if key in result else None
            for key in (
                "input_peak_power_W", "input_peak_intensity_W_m2", "input_effective_area_m2",
                "input_second_moment_radius_m", "input_r50_m", "input_r90_m", "input_boundary_I_fraction",
            )
        }
        failures = _quality_gates(result, spec["quality_gates"])
        quality_pass = quality_pass and not failures
        case_reports[case["case_id"]] = {
            "metadata": metadata, "profile_type": profile_type, "input": input_summary,
            "filament_metrics": _filament_metrics(result, threshold), "quality_gate_failures": failures,
        }
        case_results.append((case, result))

    comparison_dir = stage_dir / "comparison"
    comparison_summary = comparison_dir / "comparison_summary.json"
    if not comparison_summary.is_file():
        complete = False
    figure_name = _profile_figure(case_results, reports_dir / "input_profiles.png") if len(case_results) == 2 else None
    report = {
        "stage_id": spec["stage_id"], "stage_name": spec["stage_name"], "objective": spec["objective"],
        "technical_status": "completed" if complete else "incomplete",
        "quality_gate_status": "passed" if complete and quality_pass else ("failed" if complete else "not_evaluated"),
        "scientific_interpretation_status": "controlled_comparison_only",
        "interpretation_limit": "No experimental reference curve is supplied; the report quantifies differences only and does not assign causal importance.",
        "filament_threshold_m3": threshold, "cases": case_reports,
        "comparison_summary": "comparison/comparison_summary.json" if comparison_summary.is_file() else None,
        "input_profile_figure": f"reports/{figure_name}" if figure_name else None,
    }
    (reports_dir / "transverse_profile_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        f"# {spec['display_name']}", "", "## Status", "",
        f"- Technical: {report['technical_status']}", f"- Quality gates: {report['quality_gate_status']}",
        "- Interpretation: controlled comparison only; no experimental reference curve was supplied.", "",
        "## Input normalization", "",
        "| Case | Profile | Peak power (W) | Peak intensity (W/m²) | Effective area (m²) | r50 (m) | r90 (m) | Boundary I fraction |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in spec["cases"]:
        item = case_reports.get(case["case_id"], {})
        data = item.get("input", {})
        lines.append(
            f"| {case['label']} | {item.get('profile_type', '')} | {data.get('input_peak_power_W', '')} | "
            f"{data.get('input_peak_intensity_W_m2', '')} | {data.get('input_effective_area_m2', '')} | "
            f"{data.get('input_r50_m', '')} | {data.get('input_r90_m', '')} | {data.get('input_boundary_I_fraction', '')} |"
        )
    lines.extend(["", "## Filament metrics", "", "| Case | Status | z_on (m) | z_peak (m) | rho_peak (m⁻³) | z_end (m) | Length (m) |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for case in spec["cases"]:
        metrics = case_reports.get(case["case_id"], {}).get("filament_metrics", {})
        lines.append(
            f"| {case['label']} | {metrics.get('status', '')} | {metrics.get('z_on_m', '')} | "
            f"{metrics.get('z_rho_peak_m', '')} | {metrics.get('rho_peak_m3', '')} | "
            f"{metrics.get('z_end_m', '')} | {metrics.get('filament_length_m', '')} |"
        )
    lines.extend(["", "## Outputs", "", "- reports/input_profiles.png", "- comparison/comparison_overview.png", "- comparison/rho_onaxis_max_z.png", "- comparison/I_max_z.png", "- comparison/fwhm_plasma_z.png", "", "## Interpretation limit", "", report["interpretation_limit"]])
    (reports_dir / "transverse_profile_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", required=True)
    args = parser.parse_args()
    report = finalize_stage(args.stage_dir)
    return 0 if report["technical_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
