#!/usr/bin/env python3
"""Compare historical_fr_mixture against production, Raman-OFF, and PyCAP.

The effect chain is identical to the Phase 6 Raman causality comparison: the
same fixed-coordinate onset thresholds, peak metrics, and numerical-path
checks, with the historical_fr_mixture run as the new causal series.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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


RHO = (1e19, 1e20, 1e21, 1e22)
INT = (1e16, 3e16, 1e17, 3e17, 5e17)
CASE_LABELS = {
    "production": "Current production (legacy_split)",
    "raman_off": "Raman phase OFF",
    "historical_fr_mixture": "Historical f_R mixture",
}
CASE_COLORS = {
    "production": "#b91c1c",
    "raman_off": "#475569",
    "historical_fr_mixture": "#0369a1",
}


def _plot_comparison(out_dir: Path, x: np.ndarray, series: dict[str, dict[str, np.ndarray]],
                     px: np.ndarray, py: np.ndarray) -> None:
    """Write the standard causal-comparison figures from the aligned diagnostics."""
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 9})

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for name, data in series.items():
        ax.semilogy(x, data["rho_max_z"], label=CASE_LABELS[name], color=CASE_COLORS[name])
    ax.semilogy(px, py, "k--", linewidth=1.2, label="PyCAP digitization")
    ax.set(xlabel="x relative to focus (cm)", ylabel=r"peak electron density (m$^{-3}$)")
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out_dir / "rho_vs_x.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for name, data in series.items():
        ax.semilogy(x, data["rho_max_z"], label=CASE_LABELS[name], color=CASE_COLORS[name])
    ax.semilogy(px, py, "k--", linewidth=1.2, label="PyCAP digitization")
    ax.axhline(1e22, color="#111827", linewidth=0.8, linestyle=":")
    ax.set(xlim=(-18.5, -10.0), ylim=(3e20, 1e23), xlabel="x relative to focus (cm)",
           ylabel=r"peak electron density (m$^{-3}$)")
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out_dir / "onset_zoom_1e22.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for name, data in series.items():
        ax.semilogy(x, data["I_max_z"], label=CASE_LABELS[name], color=CASE_COLORS[name])
    ax.set(xlabel="x relative to focus (cm)", ylabel=r"I$_{max}$ (W m$^{-2}$)")
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out_dir / "Imax_vs_x.png"); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True)
    for name, data in series.items():
        axes[0].semilogy(x, np.abs(data["dphi_rot_applied_max_abs_z"]),
                         label=CASE_LABELS[name], color=CASE_COLORS[name])
        axes[1].semilogy(x, np.abs(data["alpha_R_applied_max_z"]),
                         label=CASE_LABELS[name], color=CASE_COLORS[name])
    axes[0].set_ylabel(r"max $|\Delta\phi_R|$ (rad)")
    axes[1].set(xlabel="x relative to focus (cm)", ylabel=r"max $|\alpha_R|$ (m$^{-1}$)")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8); fig.tight_layout()
    fig.savefig(out_dir / "raman_phase_diagnostics.png"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--production-raman", type=Path, required=True)
    parser.add_argument("--raman-off", type=Path, required=True)
    parser.add_argument("--raman-off-raman", type=Path, required=True)
    parser.add_argument("--mixture", type=Path, required=True)
    parser.add_argument("--mixture-raman", type=Path, required=True)
    parser.add_argument("--pycap", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    production = cc.merge(cc.read(args.production), cc.read(args.production_raman))
    raman_off = cc.merge(cc.read(args.raman_off), cc.read(args.raman_off_raman))
    mixture = cc.merge(cc.read(args.mixture), cc.read(args.mixture_raman))
    paper = cc.read(args.pycap)
    px, py = paper["x_focus_cm"], paper["rho_1e16_cm3"] * 1e22

    series = {
        "production": production,
        "raman_off": raman_off,
        "historical_fr_mixture": mixture,
    }
    x = production["x_focus_cm"]
    for name, data in series.items():
        if not np.array_equal(x, data["x_focus_cm"]):
            raise ValueError(f"{name} x_focus_cm axis differs from production")

    eps = max(0.1, 3.0 * float(np.median(np.diff(x))))
    density_metrics = {name: cc.density(x, data["rho_max_z"]) for name, data in series.items()}
    pycap_density = cc.density(px, py)

    thresholds = []
    for t in RHO:
        key = str(int(t))
        row = {"family": "rho_total", "threshold": t}
        for name in series:
            row[f"x_{name}_cm"] = density_metrics[name]["crossings"][key]
        row["x_pycap_cm"] = pycap_density["crossings"][key]
        mix_x = row["x_historical_fr_mixture_cm"]
        prod_x = row["x_production_cm"]
        row["onset_shift_mixture_minus_production_cm"] = (
            None if mix_x is None or prod_x is None else mix_x - prod_x)
        if t == 1e22:
            for name in series:
                row[f"error_{name}_to_pycap_cm"] = cc.err(
                    density_metrics[name]["crossings"][key], pycap_density["crossings"][key])
        thresholds.append(row)

    intensity = []
    for t in INT:
        for name in series:
            intensity.append({
                "threshold_W_m2": t,
                "case": name,
                "x_cm": cc.cross(x, series[name]["I_max_z"], t),
            })

    raman = []
    raman_fields = (
        "IR_max_z", "delta_n_rot_max_z", "delta_n_rot_applied_max_z",
        "dphi_rot_max_abs_z", "dphi_rot_applied_max_abs_z", "alpha_R_applied_max_z",
    )
    for name, data in series.items():
        for field in raman_fields:
            values = np.asarray(data.get(field, []), dtype=float)
            raman.append({
                "case": name, "metric": field,
                "peak": float(np.max(np.abs(values))) if values.size else None,
                "final": float(values[-1]) if values.size else None,
            })

    energy = []
    for name, data in series.items():
        for field in ("U_rel_change_z", "E_dep_cumulative_z", "E_loss_from_input_z"):
            values = np.asarray(data[field], dtype=float)
            energy.append({
                "case": name, "metric": field,
                "final": float(values[-1]),
                "max_abs": float(np.max(np.abs(values))),
            })

    numeric = []
    for name, data in series.items():
        numeric.append({
            "case": name,
            "dz_min_m": float(np.asarray(data["dz_used_z"]).min()),
            "dz_max_m": float(np.asarray(data["dz_used_z"]).max()),
            "rejections_max": float(np.asarray(data["adaptive_rejection_count_z"]).max()),
            "safety_max": float(np.asarray(data["safety_mode_trigger_count_z"]).max()),
        })

    rmse = {}
    for name, data in series.items():
        rmse[name] = cc.rmse(x, data["rho_max_z"], px, py)

    peak_rows = []
    for name, dm in density_metrics.items():
        peak_rows.append({
            "case": name,
            "rho_peak_m3": dm["rho_peak_m3"],
            "peak_x_cm": dm["peak_x_cm"],
            "peak_top_center_cm": dm["peak_top_center_cm"],
            "fwhm_cm": dm["fwhm_cm"],
            "tail_area_above_half_m3_cm": dm["tail_area_above_half_m3_cm"],
            "rmse_vs_pycap": rmse[name],
        })
    peak_rows.append({
        "case": "pycap",
        "rho_peak_m3": pycap_density["rho_peak_m3"],
        "peak_x_cm": pycap_density["peak_x_cm"],
        "peak_top_center_cm": pycap_density["peak_top_center_cm"],
        "fwhm_cm": pycap_density["fwhm_cm"],
        "tail_area_above_half_m3_cm": pycap_density["tail_area_above_half_m3_cm"],
        "rmse_vs_pycap": 0.0,
    })

    onset_1e22 = next(r for r in thresholds if r["threshold"] == 1e22)
    summary = {
        "schema": "khz_filament.historical_fr_mixture.comparison.v1",
        "coordinate_definition": "x_focus_cm = 100 * (z_m - 0.95)",
        "epsilon_x_cm": eps,
        "series": {
            "production": str(args.production),
            "raman_off": str(args.raman_off),
            "historical_fr_mixture": str(args.mixture),
            "pycap": str(args.pycap),
        },
        "rho_threshold_crossings": thresholds,
        "onset_1e22_shift": {
            "mixture_minus_production_cm": onset_1e22["onset_shift_mixture_minus_production_cm"],
            "mixture_error_to_pycap_cm": onset_1e22.get("error_historical_fr_mixture_to_pycap_cm"),
            "production_error_to_pycap_cm": onset_1e22.get("error_production_to_pycap_cm"),
            "raman_off_error_to_pycap_cm": onset_1e22.get("error_raman_off_to_pycap_cm"),
        },
        "intensity_threshold_crossings": intensity,
        "raman_diagnostics": raman,
        "energy_change": energy,
        "numerical_path": numeric,
        "peak_and_width": peak_rows,
        "rmse_vs_pycap": rmse,
    }
    (args.out_dir / "historical_fr_mixture_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plot_comparison(args.out_dir, x, series, px, py)

    def write_csv(name, rows):
        rows = list(rows)
        keys = sorted({k for r in rows for k in r})
        with (args.out_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader(); writer.writerows(rows)

    write_csv("rho_threshold_crossings.csv", thresholds)
    write_csv("intensity_threshold_crossings.csv", intensity)
    write_csv("raman_diagnostics.csv", raman)
    write_csv("energy_change.csv", energy)
    write_csv("numerical_path.csv", numeric)
    write_csv("peak_and_width.csv", peak_rows)

    prod_1e22 = onset_1e22["x_production_cm"]
    mix_1e22 = onset_1e22["x_historical_fr_mixture_cm"]
    pycap_1e22 = onset_1e22["x_pycap_cm"]
    shift = onset_1e22["onset_shift_mixture_minus_production_cm"]
    direction = "later (toward PyCAP)" if shift and shift > eps else "earlier" if shift and shift < -eps else "within epsilon"
    mix_peak = next(row for row in peak_rows if row["case"] == "historical_fr_mixture")
    prod_peak = next(row for row in peak_rows if row["case"] == "production")
    pycap_peak = next(row for row in peak_rows if row["case"] == "pycap")
    lines = [
        "# historical_fr_mixture 120 fs causal comparison",
        "",
        f"- 1e22 onset: production={prod_1e22} cm, mixture={mix_1e22} cm, PyCAP={pycap_1e22} cm.",
        f"- Mixture minus production shift at 1e22: **{shift} cm** ({direction}, epsilon={eps:.3f} cm).",
        f"- Peak rho: production={prod_peak['rho_peak_m3']:.4e} m^-3 at {prod_peak['peak_x_cm']:.3f} cm; mixture={mix_peak['rho_peak_m3']:.4e} m^-3 at {mix_peak['peak_x_cm']:.3f} cm; PyCAP={pycap_peak['rho_peak_m3']:.4e} m^-3 at {pycap_peak['peak_x_cm']:.3f} cm.",
        f"- RMSE vs PyCAP (rho_max_z): production={rmse['production']:.4e}, mixture={rmse['historical_fr_mixture']:.4e}, Raman-OFF={rmse['raman_off']:.4e}.",
        "",
        "Core question: with every other model component frozen, does swapping the Raman phase operator to the pre-April f_R mixture move the 120 fs onset back toward PyCAP? See `historical_fr_mixture_comparison_summary.json` for the full effect-chain metrics.",
        "",
    ]
    (args.out_dir / "historical_fr_mixture_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"shift_1e22_cm": shift, "rmse": rmse}, indent=2))


if __name__ == "__main__":
    main()
