#!/usr/bin/env python3
"""Audit time-derivative and operator signs under the repository FFT convention."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from KHz_filament.nonlinear import operator_correct_scalar


def relative_linf(actual, expected):
    return float(np.max(np.abs(actual - expected)) / max(np.max(np.abs(expected)), 1e-300))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    nt, dt, mode_index = 4096, 1e-15, 8
    omega_a = 2.0 * np.pi * mode_index / (nt * dt)
    t = (np.arange(nt) - nt // 2) * dt
    omega = 2.0 * np.pi * np.fft.fftfreq(nt, d=dt)
    omega0 = 2.0 * np.pi * 299792458.0 / 800e-9
    rows, traces = [], {}
    cases = (
        ("cos_positive", np.cos(omega_a * t), -omega_a * np.sin(omega_a * t)),
        ("sin_positive", np.sin(omega_a * t), omega_a * np.cos(omega_a * t)),
        ("cos_negative", np.cos(-omega_a * t), omega_a * np.sin(-omega_a * t)),
        ("sin_negative", np.sin(-omega_a * t), -omega_a * np.cos(-omega_a * t)),
    )
    for name, signal, analytic in cases:
        fft_derivative = np.fft.ifft(1j * omega * np.fft.fft(signal)).real
        tdiff_derivative = (np.roll(signal, -1) - np.roll(signal, 1)) / (2.0 * dt)
        q = signal[:, None, None]
        corrected_tdiff = np.asarray(operator_correct_scalar(
            q, omega, omega0, dt=dt, method="tdiff", operator_convention="isaacs_eq27"))[:, 0, 0]
        corrected_fft = np.asarray(operator_correct_scalar(
            q, omega, omega0, method="fft", operator_convention="isaacs_eq27"))[:, 0, 0]
        expected_corrected = signal - analytic / omega0
        rows.append({
            "case": name,
            "analytic_vs_fft_derivative_error": relative_linf(fft_derivative, analytic),
            "analytic_vs_tdiff_derivative_error": relative_linf(tdiff_derivative, analytic),
            "operator_tdiff_error": relative_linf(corrected_tdiff, expected_corrected),
            "operator_fft_error": relative_linf(corrected_fft, expected_corrected),
            "tdiff_fft_operator_error": relative_linf(corrected_tdiff, corrected_fft),
            "omega_a_rad_s": omega_a,
            "dt_s": dt,
        })
        traces[name] = (analytic, fft_derivative, tdiff_derivative)
    with (args.out_dir / "time_derivative_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "fft_forward_convention": "F(Omega)=sum_n f(t_n) exp(-i Omega t_n)",
        "F[d_tau f]": "+i Omega F[f]",
        "Omega_array_ordering": "2*pi*fftfreq(Nt, dt), unshifted FFT order",
        "fft_ifft_normalization": "fft unnormalized; ifft includes 1/N",
        "tau_axis_direction": "increasing sample index means increasing tau",
        "retarded_time_definition": "tau=t-z/v_g",
        "Eq.27_operator_frequency_multiplier": "1-Omega/omega0 for (1+i/omega0*d_tau)",
        "legacy_operator_correct_scalar_tdiff_multiplier": "1-i*Omega/omega0",
        "legacy_operator_correct_scalar_fft_multiplier": "1+i*Omega/omega0",
        "isaacs_eq27_scalar_split_multiplier": "1-i*Omega/omega0",
        "tdiff_fft_mutually_consistent": max(row["tdiff_fft_operator_error"] for row in rows) < 1e-4,
        "legacy_behavior_changed": False,
        "strict_isaacs_convention": "isaacs_eq27",
    }
    (args.out_dir / "time_derivative_convention.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    analytic, fft_derivative, tdiff_derivative = traces["cos_positive"]
    fig, ax = plt.subplots()
    window = slice(nt // 2 - 160, nt // 2 + 160)
    ax.plot(t[window] * 1e15, analytic[window], label="analytic")
    ax.plot(t[window] * 1e15, fft_derivative[window], "--", label="FFT")
    ax.plot(t[window] * 1e15, tdiff_derivative[window], ":", label="central difference")
    ax.set(xlabel="tau (fs)", ylabel="df/dtau", title="Repository time-derivative convention")
    ax.legend(); fig.tight_layout(); fig.savefig(args.out_dir / "time_derivative_validation.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
