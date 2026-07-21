#!/usr/bin/env python3
"""Small-grid production-chain audit of full Raman/non-Raman operator ordering."""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "isaacs_raman_closure" / "phase8b_preflight"


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def components(split_order, dz, total_z):
    from KHz_filament.config import (
        BeamConfig, GridConfig, HeatConfig, IonizationConfig,
        PropagationConfig, RamanConfig, RunConfig,
    )

    species = [
        {"name": "N2", "rate": "ppt_talebpour_i_lut", "reference_model": "ppt_talebpour_i_full_reference",
         "Ip_eV": 15.6, "Z": 1, "l": 0, "m": 0, "fraction": .8, "Ip_eV_eff": 15.6, "Zeff": .9},
        {"name": "O2", "rate": "ppt_talebpour_i_lut", "reference_model": "ppt_talebpour_i_full_reference",
         "Ip_eV": 12.1, "Z": 1, "l": 0, "m": 0, "fraction": .2, "Ip_eV_eff": 12.55, "Zeff": .53},
    ]
    rate_table = {
        "enabled": True, "reuse_cache": True,
        "cache_dir": str(ROOT / "tmp" / "phase8b_rate_tables"),
        "force_rebuild": False, "I_min_SI": 1e8, "I_max_SI": 1e19,
        "n_samples": 32, "spacing": "log", "interp_mode": "loglog",
        "ref_cycle_avg_samples": 8, "popruzhenko_sum_tol": 1e-6,
        "popruzhenko_max_terms": 256,
    }
    return dict(
        grid=GridConfig(Nx=16, Ny=16, Nt=384, Lx=1.2e-3, Ly=1.2e-3, Twin=960e-15),
        # Focus-local approximation: retain 17 GW and 120 fs while sampling a
        # 150 um high-intensity transverse spot with the production operators.
        beam=BeamConfig(
            w0=150e-6, tau_fwhm=120e-15, energy_J=None, P0_peak=17e9,
            focal_length=None,
        ),
        prop=PropagationConfig(
            z_max=total_z, dz=dz, linear_model="bk_nee", auto_substep=False,
            focus_window_step=False, limit_focus_window=False,
            progress_every_z=0, energy_probe_every=0, diag_extra=False,
            use_raman_phase=False, use_raman_full_operator=True,
            use_raman_absorption=False,
        ),
        ion=IonizationConfig(
            species=species, rate_table=rate_table, time_mode="full",
            integrator="rk4", cycle_avg_samples=32, mean_clip_frac=0.0,
            I_cap=1e19, W_cap=1e19, sigma_ib=0.0, nu_ei_const=0.0,
        ),
        heat=HeatConfig(), run=RunConfig(Npulses=1),
        raman=RamanConfig(
            enabled=True, model="isaacs_rot_sinexp", f_R=None, T_R=None,
            T2=None, tau2=None, n_R=2.3e-23, omega_R=1.6e13,
            Gamma_R=1.3e13, operator_mode="full_isaacs_eq27",
            operator_convention="isaacs_eq27",
            iir_sampling="exact_piecewise_linear", operator_integrator="heun",
            nonlinear_split_order=split_order, absorption=False,
        ),
    )


def metrics(result):
    field = np.asarray(result["E_final"])
    diag = result["diagnostics"]
    t = np.asarray(result["axes"]["t"])
    center = field[:, field.shape[1]//2, field.shape[2]//2]
    intensity = np.abs(center) ** 2
    temporal_centroid = float(np.sum(t*intensity)/max(np.sum(intensity), 1e-300))
    omega = np.fft.fftshift(2*np.pi*np.fft.fftfreq(len(t), t[1]-t[0]))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(center)))**2
    spectral_centroid = float(np.sum(omega*spectrum)/max(np.sum(spectrum), 1e-300))
    spectral_width = float(np.sqrt(np.sum((omega-spectral_centroid)**2*spectrum)/max(np.sum(spectrum), 1e-300)))
    return {
        "field": field,
        "pulse_energy_J": float(diag["U_z"][-1]),
        "I_max_W_m2": float(diag["I_max_z"][-1]),
        "rho_max_m3": float(diag["rho_max_z"][-1]),
        "raman_deposited_energy_J": float(diag["raman_actual_loss_cumulative_J"][-1]),
        "ionization_deposited_energy_J": float(np.sum(diag["E_dep_z"])),
        "temporal_centroid_s": temporal_centroid,
        "spectral_centroid_rad_s": spectral_centroid,
        "spectral_width_rad_s": spectral_width,
        "finite": bool(np.isfinite(field).all()),
    }


def relative(value, reference):
    return abs(float(value)-float(reference))/max(abs(float(reference)), 1e-300)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    from KHz_filament.runner import run_demo

    total_z, production_dz = 4e-4, 1e-4
    cases = [
        ("after_other", "after_other", production_dz),
        ("before_other", "before_other", production_dz),
        ("strang", "strang", production_dz),
        ("strang_dz2", "strang", production_dz/2),
        ("strang_dz4", "strang", production_dz/4),
        ("strang_dz8", "strang", production_dz/8),
        ("strang_dz16_reference", "strang", production_dz/16),
    ]
    results = {}
    with tempfile.TemporaryDirectory(dir=args.out_dir, prefix="combined_operator_scratch_") as scratch:
        scratch = Path(scratch)
        for label, order, dz in cases:
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_demo(
                    **components(order, dz, total_z),
                    out_path=str(scratch / f"{label}.npz"),
                    dtype="fp64", return_results=True,
                )
            results[label] = metrics(result)

    reference = results["strang_dz16_reference"]
    order_rows = []
    for label, order, dz in cases[:-1]:
        item = results[label]
        order_rows.append({
            "case": label, "nonlinear_split_order": order, "dz_m": dz,
            "field_l2_error_to_strang_dz16": float(np.linalg.norm(item["field"]-reference["field"])/np.linalg.norm(reference["field"])),
            "pulse_energy_J": item["pulse_energy_J"], "I_max_W_m2": item["I_max_W_m2"],
            "rho_max_m3": item["rho_max_m3"], "raman_deposited_energy_J": item["raman_deposited_energy_J"],
            "ionization_deposited_energy_J": item["ionization_deposited_energy_J"],
            "temporal_centroid_s": item["temporal_centroid_s"],
            "spectral_centroid_rad_s": item["spectral_centroid_rad_s"],
            "spectral_width_rad_s": item["spectral_width_rad_s"], "finite": item["finite"],
        })
    write_csv(args.out_dir / "combined_operator_order_comparison.csv", order_rows)

    convergence = [row for row in order_rows if row["case"].startswith("strang")]
    convergence.sort(key=lambda row: row["dz_m"], reverse=True)
    for index, row in enumerate(convergence):
        if index+1 < len(convergence):
            next_error = convergence[index+1]["field_l2_error_to_strang_dz16"]
            row["estimated_order"] = float(np.log(row["field_l2_error_to_strang_dz16"]/next_error)/np.log(2)) if next_error > 0 else ""
        else:
            row["estimated_order"] = ""
    write_csv(args.out_dir / "combined_operator_dz_convergence.csv", convergence)

    production = results["strang"]
    dz2 = results["strang_dz2"]
    after = results["after_other"]
    observable_rows = []
    for label, item, ref in (
        ("strang_vs_dz2", production, dz2),
        ("after_other_vs_strang", after, production),
    ):
        observable_rows.append({
            "comparison": label,
            "field_l2_difference": float(np.linalg.norm(item["field"]-ref["field"])/np.linalg.norm(ref["field"])),
            "pulse_energy_relative_difference": relative(item["pulse_energy_J"], ref["pulse_energy_J"]),
            "I_max_relative_difference": relative(item["I_max_W_m2"], ref["I_max_W_m2"]),
            "rho_max_relative_difference": relative(item["rho_max_m3"], ref["rho_max_m3"]),
            "raman_loss_relative_difference": relative(item["raman_deposited_energy_J"], ref["raman_deposited_energy_J"]),
            "ionization_loss_relative_difference": relative(item["ionization_deposited_energy_J"], ref["ionization_deposited_energy_J"]),
            "temporal_centroid_difference_fs": (item["temporal_centroid_s"]-ref["temporal_centroid_s"])*1e15,
            "spectral_centroid_difference_rad_s": item["spectral_centroid_rad_s"]-ref["spectral_centroid_rad_s"],
            "spectral_width_relative_difference": relative(item["spectral_width_rad_s"], ref["spectral_width_rad_s"]),
        })
    write_csv(args.out_dir / "combined_operator_observable_comparison.csv", observable_rows)
    summary = {
        "grid": {"Nt": 384, "Nx": 16, "Ny": 16},
        "pulse": "120 fs, 17 GW focus-local approximation",
        "species": ["N2 Talebpour", "O2 Talebpour"],
        "production_dz_m": production_dz,
        "minimum_estimated_order": min(float(row["estimated_order"]) for row in convergence if row["estimated_order"] != ""),
        "refined_estimated_order": float([row for row in convergence if row["estimated_order"] != ""][-1]["estimated_order"]),
        "production_vs_dz2": observable_rows[0],
        "after_other_vs_strang": observable_rows[1],
    }
    (args.out_dir / "combined_operator_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
