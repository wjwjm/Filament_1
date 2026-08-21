#!/usr/bin/env python3
"""Compare the 0.85-electronic-Kerr candidate with all 120 fs references."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import compare_raman_phase_causality as cc  # noqa: E402


RHO_THRESHOLDS = (1e19, 1e20, 1e21, 1e22)
INTENSITY_THRESHOLDS = (1e16, 3e16, 1e17, 3e17, 5e17)
LABELS = {
    "production": "Current production",
    "raman_off": "Raman phase OFF",
    "historical_fr_mixture": "Historical f_R mixture",
    "raman_off_kerr085": "Raman OFF + 0.85 electronic Kerr",
}
COLORS = {
    "production": "#b91c1c", "raman_off": "#475569",
    "historical_fr_mixture": "#0369a1", "raman_off_kerr085": "#15803d",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def plot_all(out_dir: Path, x: np.ndarray, series: dict, px: np.ndarray, py: np.ndarray) -> None:
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 9})
    for filename, field, ylabel, semilogy in (
        ("rho_vs_x.png", "rho_max_z", r"peak electron density (m$^{-3}$)", True),
        ("Imax_vs_x.png", "I_max_z", r"I$_{max}$ (W m$^{-2}$)", True),
        ("energy_change_vs_x.png", "U_rel_change_z", "relative pulse-energy change", False),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        for name, data in series.items():
            method = ax.semilogy if semilogy else ax.plot
            method(x, data[field], label=LABELS[name], color=COLORS[name])
        if field == "rho_max_z":
            ax.semilogy(px, py, "k--", linewidth=1.2, label="PyCAP digitization")
        ax.set(xlabel="x relative to focus (cm)", ylabel=ylabel)
        ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(out_dir / filename); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for name, data in series.items():
        ax.semilogy(x, data["rho_max_z"], label=LABELS[name], color=COLORS[name])
    ax.semilogy(px, py, "k--", linewidth=1.2, label="PyCAP digitization")
    ax.axhline(1e22, color="#111827", linewidth=0.8, linestyle=":")
    ax.set(xlim=(-17.5, -11.5), ylim=(5e20, 8e22), xlabel="x relative to focus (cm)",
           ylabel=r"peak electron density (m$^{-3}$)")
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out_dir / "onset_zoom_1e22.png"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("production", "production_raman", "raman_off", "raman_off_raman",
                 "historical", "historical_raman", "candidate", "candidate_raman", "pycap"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    series = {
        "production": cc.merge(cc.read(args.production), cc.read(args.production_raman)),
        "raman_off": cc.merge(cc.read(args.raman_off), cc.read(args.raman_off_raman)),
        "historical_fr_mixture": cc.merge(cc.read(args.historical), cc.read(args.historical_raman)),
        "raman_off_kerr085": cc.merge(cc.read(args.candidate), cc.read(args.candidate_raman)),
    }
    paper = cc.read(args.pycap)
    px, py = paper["x_focus_cm"], paper["rho_1e16_cm3"] * 1e22
    x = series["production"]["x_focus_cm"]
    for name, data in series.items():
        if not np.array_equal(x, data["x_focus_cm"]):
            raise ValueError(f"{name} x_focus_cm axis differs from production")
    epsilon = max(0.1, 3.0 * float(np.median(np.diff(x))))

    density = {name: cc.density(x, data["rho_max_z"]) for name, data in series.items()}
    pycap_density = cc.density(px, py)
    crossings: list[dict] = []
    for threshold in RHO_THRESHOLDS:
        key = str(int(threshold))
        row = {"threshold_m-3": threshold, "x_pycap_cm": pycap_density["crossings"][key]}
        for name in series:
            row[f"x_{name}_cm"] = density[name]["crossings"][key]
        row["candidate_minus_raman_off_cm"] = (
            row["x_raman_off_kerr085_cm"] - row["x_raman_off_cm"])
        row["candidate_minus_historical_cm"] = (
            row["x_raman_off_kerr085_cm"] - row["x_historical_fr_mixture_cm"])
        crossings.append(row)

    onset = next(row for row in crossings if row["threshold_m-3"] == 1e22)
    candidate_onset = onset["x_raman_off_kerr085_cm"]
    historical_onset = onset["x_historical_fr_mixture_cm"]
    off_onset = onset["x_raman_off_cm"]
    supports_electronic_origin = abs(candidate_onset - historical_onset) <= epsilon

    peaks = []
    rmses = {}
    for name, metrics in density.items():
        rmses[name] = cc.rmse(x, series[name]["rho_max_z"], px, py)
        peaks.append({
            "case": name, "rho_peak_m-3": metrics["rho_peak_m3"],
            "peak_x_cm": metrics["peak_x_cm"], "peak_top_center_cm": metrics["peak_top_center_cm"],
            "fwhm_cm": metrics["fwhm_cm"], "rmse_vs_pycap": rmses[name],
        })
    peaks.append({
        "case": "pycap", "rho_peak_m-3": pycap_density["rho_peak_m3"],
        "peak_x_cm": pycap_density["peak_x_cm"],
        "peak_top_center_cm": pycap_density["peak_top_center_cm"],
        "fwhm_cm": pycap_density["fwhm_cm"], "rmse_vs_pycap": 0.0,
    })

    intensity = [{
        "case": name, "threshold_W_m-2": threshold,
        "x_cm": cc.cross(x, data["I_max_z"], threshold),
    } for threshold in INTENSITY_THRESHOLDS for name, data in series.items()]
    intensity_peaks = []
    for name, data in series.items():
        imax = np.asarray(data["I_max_z"], float)
        peak_index = int(np.argmax(imax))
        intensity_peaks.append({
            "case": name,
            "I_max_peak_W_m-2": float(imax[peak_index]),
            "peak_x_cm": float(x[peak_index]),
        })
    energy = [{
        "case": name, "metric": field, "final": float(np.asarray(data[field])[-1]),
        "max_abs": float(np.max(np.abs(np.asarray(data[field])))),
    } for name, data in series.items()
      for field in ("U_rel_change_z", "E_dep_cumulative_z", "E_loss_from_input_z")]
    numerical = [{
        "case": name, "dz_min_m": float(np.min(data["dz_used_z"])),
        "dz_max_m": float(np.max(data["dz_used_z"])),
        "rejections_max": float(np.max(data["adaptive_rejection_count_z"])),
        "safety_max": float(np.max(data["safety_mode_trigger_count_z"])),
    } for name, data in series.items()]
    phase = [{
        "case": name, "metric": field,
        "peak": float(np.max(np.abs(np.asarray(data.get(field, []), float))))
        if np.asarray(data.get(field, [])).size else None,
    } for name, data in series.items()
      for field in ("dphi_elec_applied_max_abs_z", "dphi_rot_applied_max_abs_z", "alpha_R_applied_max_z")]

    summary = {
        "schema": "khz_filament.raman_off_kerr085.comparison.v1",
        "coordinate_definition": "x_focus_cm = 100 * (z_m - 0.95)",
        "epsilon_cm": epsilon,
        "rho_threshold_crossings": crossings,
        "onset_1e22": {
            "raman_off_cm": off_onset, "candidate_cm": candidate_onset,
            "historical_fr_mixture_cm": historical_onset, "pycap_cm": onset["x_pycap_cm"],
            "candidate_minus_raman_off_cm": candidate_onset - off_onset,
            "candidate_minus_historical_cm": candidate_onset - historical_onset,
            "supports_electronic_kerr_origin_within_epsilon": supports_electronic_origin,
        },
        "peak_and_width": peaks, "intensity_threshold_crossings": intensity,
        "intensity_peaks": intensity_peaks,
        "energy_change": energy, "numerical_path": numerical,
        "phase_and_absorption_diagnostics": phase, "rmse_vs_pycap": rmses,
    }
    (args.out_dir / "raman_off_kerr085_comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(args.out_dir / "rho_threshold_crossings.csv", crossings)
    write_csv(args.out_dir / "peak_and_width.csv", peaks)
    write_csv(args.out_dir / "intensity_threshold_crossings.csv", intensity)
    write_csv(args.out_dir / "intensity_peaks.csv", intensity_peaks)
    write_csv(args.out_dir / "energy_change.csv", energy)
    write_csv(args.out_dir / "numerical_path.csv", numerical)
    write_csv(args.out_dir / "phase_and_absorption_diagnostics.csv", phase)
    plot_all(args.out_dir, x, series, px, py)

    verdict = (
        "SUPPORTED: candidate onset is within the fixed epsilon of the historical mixture."
        if supports_electronic_origin else
        "NOT SUPPORTED: candidate onset is not within the fixed epsilon of the historical mixture."
    )
    candidate_peak = next(row for row in peaks if row["case"] == "raman_off_kerr085")
    candidate_intensity_peak = next(
        row for row in intensity_peaks if row["case"] == "raman_off_kerr085"
    )
    report = [
        "# Raman phase OFF + 0.85 electronic Kerr causal comparison", "",
        f"- 1e22 onset: Raman-OFF={off_onset:.6f} cm, candidate={candidate_onset:.6f} cm, historical={historical_onset:.6f} cm, PyCAP={onset['x_pycap_cm']:.6f} cm.",
        f"- Candidate shift from Raman-OFF: {candidate_onset - off_onset:+.6f} cm.",
        f"- Candidate residual to historical mixture: {candidate_onset - historical_onset:+.6f} cm (epsilon={epsilon:.3f} cm).",
        f"- Candidate peak density: {candidate_peak['rho_peak_m-3']:.4e} m^-3 at {candidate_peak['peak_x_cm']:.3f} cm.",
        f"- Candidate peak I_max: {candidate_intensity_peak['I_max_peak_W_m-2']:.4e} W/m^2 at {candidate_intensity_peak['peak_x_cm']:.3f} cm.",
        f"- Core criterion: **{verdict}**", "",
    ]
    (args.out_dir / "raman_off_kerr085_comparison_report.md").write_text(
        "\n".join(report), encoding="utf-8")
    print(json.dumps(summary["onset_1e22"], indent=2))


if __name__ == "__main__":
    main()
