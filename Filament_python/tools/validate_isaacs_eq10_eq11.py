#!/usr/bin/env python3
"""Independent FFT, IIR, and Isaacs Eq. (10)/(11) convergence audit."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from KHz_filament.raman import raman_convolve_intensity, raman_convolve_intensity_fft_linear
from KHz_filament.raman_isaacs_reference import C0, eq11_alpha, isaacs_kernel

NR, OMEGA, GAMMA, I0 = 2.3e-23, 1.6e13, 1.3e13, 5e17


def write_csv(path, rows):
    rows = list(rows)
    keys = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def rel_linf(actual, reference):
    return float(np.max(np.abs(actual - reference)) / max(np.max(np.abs(reference)), 1e-300))


def pulse_case(t, case, center):
    x = t - center
    if case == "40fs_gaussian":
        return np.exp(-4 * np.log(2) * (x / 40e-15) ** 2)
    base = np.exp(-4 * np.log(2) * (x / 120e-15) ** 2)
    if case == "120fs_gaussian":
        return base
    if case == "120fs_chirped":
        return base * (1.0 + 0.20 * x / 120e-15)
    if case == "120fs_asymmetric_tail":
        return base + .18 * np.exp(-((x - 130e-15) / 75e-15) ** 2)
    if case == "impulse":
        result = np.zeros_like(t); result[int(round(center / (t[1] - t[0])))] = 1.0 / (t[1] - t[0]); return result
    if case == "constant":
        return np.ones_like(t)
    raise ValueError(case)


def fft_direct_rows():
    rows = []
    for dtype, tolerance in ((np.float64, 1e-10), (np.float32, 1e-5)):
        dt = dtype(.625e-15)
        t = np.arange(2048, dtype=dtype) * dt
        h = isaacs_kernel(t.astype(float), OMEGA, GAMMA).astype(dtype)
        for case in ("40fs_gaussian", "120fs_gaussian", "120fs_chirped", "120fs_asymmetric_tail", "impulse", "constant"):
            intensity = pulse_case(t, case, dtype(400) * dt).astype(dtype)
            direct = np.convolve(intensity, h, mode="full")[: len(t)] * dt
            fft = np.asarray(raman_convolve_intensity_fft_linear(intensity[:, None, None], h, dt=float(dt)))[:, 0, 0]
            pre_samples = 400 if case not in ("constant",) else 0
            pre_response = float(np.max(np.abs(fft[:pre_samples]))) if pre_samples else 0.0
            impulse_pre_response = float(np.max(np.abs(fft[:390]))) if case == "impulse" else 0.0
            pre_response_relative = impulse_pre_response / max(float(np.max(np.abs(fft))), 1e-300)
            rows.append({
                "case": case, "dtype": np.dtype(dtype).name,
                "relative_linf_error": rel_linf(fft, direct), "threshold": tolerance,
                "pre_response_abs_max": pre_response,
                "pre_response_relative": pre_response_relative,
                "wraparound_detected": bool(case == "impulse" and pre_response_relative >= tolerance),
            })
    return rows


def convergence_rows():
    validation, iir_rows = [], []
    for tau_fs in (40, 120, 200, 500, 1000):
        tau = tau_fs * 1e-15
        for dt_fs in (2.5, 1.25, .625, .3125, .15625):
            dt = dt_fs * 1e-15
            samples = int(round(tau / dt))
            nt = int(round((tau + 1.5e-12) / dt)) + 1
            t = np.arange(nt) * dt
            intensity = np.zeros(nt); intensity[:samples + 1] = I0
            h = isaacs_kernel(t, OMEGA, GAMMA)
            direct = fftconvolve(intensity, h, mode="full")[:nt] * dt
            fft = np.asarray(raman_convolve_intensity_fft_linear(intensity[:, None, None], h, dt=dt))[:, 0, 0]
            alpha11 = eq11_alpha(I0, tau, n_R=NR, omega_R=OMEGA, Gamma_R=GAMMA)
            alpha_direct = (NR / C0) * direct[samples] / tau
            alpha_fft = (NR / C0) * fft[samples] / tau
            variant_results = {}
            for sampling in ("legacy_right_hold", "left_hold", "trapezoidal", "exact_piecewise_linear"):
                iir = np.asarray(raman_convolve_intensity(
                    intensity[:, None, None], method="iir", dt=dt, omega_R=OMEGA,
                    Gamma_R=GAMMA, iir_sampling=sampling))[:, 0, 0]
                error = rel_linf(iir, direct)
                shift = int(np.argmax(iir) - np.argmax(direct))
                alpha_iir = (NR / C0) * iir[samples] / tau
                variant_results[sampling] = (error, alpha_iir, shift)
                iir_rows.append({
                    "pulse_fs": tau_fs, "dt_fs": dt_fs, "iir_sampling": sampling,
                    "iir_vs_direct_error": error,
                    "iir_vs_eq11_error": abs(alpha_iir - alpha11) / max(abs(alpha11), 1e-300),
                    "peak_time_shift_samples": shift,
                })
            selected_error, selected_alpha, selected_shift = variant_results["exact_piecewise_linear"]
            validation.append({
                "pulse_fs": tau_fs, "dt_fs": dt_fs,
                "direct_vs_eq11_error": abs(alpha_direct - alpha11) / max(abs(alpha11), 1e-300),
                "fft_vs_eq11_error": abs(alpha_fft - alpha11) / max(abs(alpha11), 1e-300),
                "iir_vs_eq11_error": abs(selected_alpha - alpha11) / max(abs(alpha11), 1e-300),
                "iir_vs_direct_error": selected_error,
                "iir_peak_time_shift_samples": selected_shift,
                "selected_iir_sampling": "exact_piecewise_linear",
                "boxcar_edge_method": "analytic_distributional_flux",
            })
    return validation, iir_rows


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv); args.out_dir.mkdir(parents=True, exist_ok=True)
    fft_rows = fft_direct_rows()
    validation, iir_rows = convergence_rows()
    write_csv(args.out_dir / "raman_fft_direct_comparison.csv", fft_rows)
    write_csv(args.out_dir / "raman_iir_direct_convergence.csv", iir_rows)
    write_csv(args.out_dir / "eq10_eq11_validation_v2.csv", validation)
    write_csv(args.out_dir / "eq10_eq11_convergence_v2.csv", validation)
    # Corrected schema also replaces the old-named live audit; historical Phase 8A output is untouched.
    write_csv(args.out_dir / "eq10_eq11_validation.csv", validation)
    write_csv(args.out_dir / "eq10_eq11_convergence.csv", validation)
    fig, ax = plt.subplots()
    for pulse_fs in (40, 120):
        values = [row for row in validation if row["pulse_fs"] == pulse_fs]
        ax.loglog([row["dt_fs"] for row in values], [row["iir_vs_direct_error"] for row in values], "o-", label=f"{pulse_fs} fs")
    ax.axhline(.01, color="k", ls="--", label="1% gate")
    ax.set(xlabel="dt (fs)", ylabel="IIR vs direct relative L-inf error", title="Strict Isaacs IIR convergence")
    ax.legend(); fig.tight_layout(); fig.savefig(args.out_dir / "eq10_eq11_comparison_v2.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
