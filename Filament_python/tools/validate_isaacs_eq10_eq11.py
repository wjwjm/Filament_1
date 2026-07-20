#!/usr/bin/env python3
"""Static Eq. (10)/Eq. (11) closure audit; never launches propagation."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from KHz_filament.raman import raman_convolve_intensity, raman_convolve_intensity_fft_linear
from KHz_filament.raman_isaacs_reference import C0, eq11_alpha, isaacs_kernel, signed_energy_from_response

NR, OMEGA, GAMMA, I0 = 2.3e-23, 1.6e13, 1.3e13, 5e17


def write_csv(path: Path, rows):
    rows = list(rows); keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def one_case(tau: float, dt: float):
    samples = int(round(tau / dt))
    t = np.arange(max(samples + 1, int(round((tau + 1.5e-12) / dt)))) * dt
    intensity = np.zeros_like(t); intensity[:samples] = I0
    h = isaacs_kernel(t, OMEGA, GAMMA)
    direct = np.convolve(intensity, h, mode="full")[: len(t)] * dt
    fft = np.asarray(raman_convolve_intensity_fft_linear(intensity[:, None, None], h, dt=dt))[:, 0, 0]
    iir = np.asarray(raman_convolve_intensity(intensity[:, None, None], method="iir", dt=dt, omega_R=OMEGA, Gamma_R=GAMMA))[:, 0, 0]
    alpha11 = eq11_alpha(I0, tau, n_R=NR, omega_R=OMEGA, Gamma_R=GAMMA)
    rows = []
    for name, response in (("direct", direct), ("fft_linear", fft), ("iir", iir)):
        # Analytic boxcar edge flux: I_R(0)=0 and I_R(tau) is sampled at end.
        u = -(NR / C0) * I0 * response[samples]
        alpha = -u / (I0 * tau)
        rows.append({"pulse_fs": tau * 1e15, "dt_fs": dt * 1e15, "path": name,
                     "alpha_eq10_m_inv": alpha, "alpha_eq11_m_inv": alpha11,
                     "relative_error": abs(alpha - alpha11) / max(abs(alpha11), 1e-300),
                     "u_R_signed_J_m3": u, "boxcar_edge_method": "analytic_distributional_flux"})
    signed = signed_energy_from_response(intensity, direct, dt, n_R=NR)
    return rows, {"pulse_fs": tau * 1e15, "dt_fs": dt * 1e15, "signed_finite_difference_J_m3": signed.u_R_signed,
                  "q_R_positive_J_m3": signed.q_R_positive, "legacy_clipped_J_m3": signed.legacy_clipped_result,
                  "legacy_to_corrected_ratio": signed.legacy_to_corrected_ratio}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, signed = [], []
    for tau_fs in (40, 120, 200, 500, 1000):
        for dt_fs in (2.5, 1.25, .625, .3125):
            current, energy = one_case(tau_fs * 1e-15, dt_fs * 1e-15); rows.extend(current); signed.append(energy)
    write_csv(args.out_dir / "eq10_eq11_validation.csv", rows)
    write_csv(args.out_dir / "eq10_eq11_convergence.csv", rows)
    write_csv(args.out_dir / "eq10_signed_energy_validation.csv", signed)
    fig, ax = plt.subplots()
    for path in ("direct", "fft_linear", "iir"):
        values = [r for r in rows if r["pulse_fs"] == 120.0 and r["path"] == path]
        ax.loglog([r["dt_fs"] for r in values], [max(r["relative_error"], 1e-18) for r in values], "o-", label=path)
    ax.axhline(.01, color="k", ls="--", label="1% gate")
    ax.set(xlabel="dt (fs)", ylabel="relative Eq.10/Eq.11 error", title="120 fs boxcar edge-flux convergence")
    ax.legend(); fig.tight_layout(); fig.savefig(args.out_dir / "eq10_eq11_comparison.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
