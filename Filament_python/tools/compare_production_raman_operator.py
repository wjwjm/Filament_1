#!/usr/bin/env python3
"""Compare the actual production split call chain with full Isaacs Eq. (27)."""
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
from KHz_filament.constants import c0, eps0
from KHz_filament.diagnostics import intensity as code_intensity
from KHz_filament.nonlinear import apply_nonlinear, kerr_phase_from_deltan, shock_intensity
from KHz_filament.raman import raman_convolve_intensity
from KHz_filament.raman_isaacs_reference import apply_isaacs_raman_reference_step, isaacs_raman_rhs

N0, NR, WR, GR, LAM = 1.00027, 2.3e-23, 1.6e13, 1.3e13, 800e-9
OMEGA0 = 2 * np.pi * c0 / LAM
K_MEDIUM = N0 * OMEGA0 / c0
K_VAC = OMEGA0 / c0
IPEAK = 5e17


def pulse(t, kind):
    center = 0.0
    x = t - center
    width = 40e-15 if kind == "40fs_tl" else 120e-15
    envelope = np.exp(-2 * np.log(2) * (x / width) ** 2)
    phase = np.zeros_like(t)
    if kind == "120fs_positive_chirp": phase = 2.5e27 * x * x
    if kind == "120fs_negative_chirp": phase = -2.5e27 * x * x
    if kind == "120fs_tail": envelope += .20 * np.exp(-((x - 130e-15) / 75e-15) ** 2)
    if kind == "120fs_double": envelope += .25 * np.exp(-((x - 210e-15) / 45e-15) ** 2)
    envelope /= envelope.max()
    amplitude = np.sqrt(2 * IPEAK / (eps0 * c0 * N0))
    return amplitude * np.sqrt(envelope).astype(complex) * np.exp(1j * phase)


def production_split_step(field, dz, dt):
    """Invoke the real production split functions; no surrogate source formula."""
    shaped = field[:, None, None]
    omega = 2 * np.pi * np.fft.fftfreq(len(field), dt)
    local_i = np.asarray(code_intensity(shaped, N0))
    response = np.asarray(raman_convolve_intensity(
        local_i, method="iir", dt=dt, omega_R=WR, Gamma_R=GR,
        iir_sampling="exact_piecewise_linear"))
    delta_n = NR * response
    corrected = np.asarray(shock_intensity(
        delta_n, omega, OMEGA0, dt=dt, method="tdiff",
        operator_convention="isaacs_eq27"))
    phase = np.asarray(kerr_phase_from_deltan(corrected, K_MEDIUM, dz))
    updated = shaped.copy()
    apply_nonlinear(updated, phase, np.zeros_like(phase), dz)
    return updated[:, 0, 0]


def centroid(t, field):
    weight = np.abs(field) ** 2
    return float(np.sum(t * weight) / np.sum(weight))


def spectrum(field, dt):
    omega = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(len(field), dt))
    power = np.abs(np.fft.fftshift(np.fft.fft(field))) ** 2
    mean = float(np.sum(omega * power) / np.sum(power))
    rms = float(np.sqrt(np.sum((omega - mean) ** 2 * power) / np.sum(power)))
    return mean, rms


def asymmetry(t, field):
    center = centroid(t, field)
    weight = np.abs(field) ** 2
    front = np.trapezoid(weight[t < center], t[t < center])
    back = np.trapezoid(weight[t >= center], t[t >= center])
    return float(front / max(back, 1e-300))


def phase_difference(a, b):
    mask = np.abs(a) > .01 * np.max(np.abs(a))
    return float(np.max(np.abs(np.angle(b[mask] * np.conj(a[mask])))))


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv); args.out_dir.mkdir(parents=True, exist_ok=True)
    dt, nt = .3125e-15, 4096
    t = (np.arange(nt) - nt // 2) * dt
    dz = .01 / (K_VAC * NR * IPEAK)
    rows = []
    for kind in ("40fs_tl", "120fs_tl", "120fs_positive_chirp", "120fs_negative_chirp", "120fs_tail", "120fs_double"):
        field = pulse(t, kind)
        source_dz = dz * 1e-5
        production_source = (production_split_step(field, source_dz, dt) - field) / source_dz
        full_source = isaacs_raman_rhs(field[:, None, None], dt=dt, omega0=OMEGA0, n0=N0,
                                       n_R=NR, omega_R=WR, Gamma_R=GR)[:, 0, 0]
        production = production_split_step(field, dz, dt)
        full = apply_isaacs_raman_reference_step(field[:, None, None], dz, dt=dt, omega0=OMEGA0,
                                                  n0=N0, n_R=NR, omega_R=WR, Gamma_R=GR,
                                                  integrator="heun")[:, 0, 0]
        norm = lambda x: float(np.linalg.norm(x))
        source_error = norm(production_source - full_source) / max(norm(full_source), 1e-300)
        update_error = norm(production - full) / max(norm(full), 1e-300)
        energy = lambda x: float(np.trapezoid(np.asarray(code_intensity(x[:, None, None], N0))[:, 0, 0], t))
        sp0, bw0 = spectrum(production, dt); sf0, bwf = spectrum(full, dt)
        observable_error = max(
            abs(energy(production) - energy(full)) / max(energy(full), 1e-300),
            abs(centroid(t, production) - centroid(t, full)) / max(120e-15, abs(centroid(t, full)), 1e-300),
            abs(bw0 - bwf) / max(bwf, 1e-300),
        )
        rows.append({
            "waveform": kind, "source_relative_l2_error": source_error,
            "one_step_field_relative_error": update_error,
            "one_step_energy_difference": (energy(production) - energy(full)) / max(energy(full), 1e-300),
            "temporal_centroid_difference_fs": (centroid(t, production) - centroid(t, full)) * 1e15,
            "spectral_centroid_difference_rad_s": sp0 - sf0,
            "rms_bandwidth_difference": (bw0 - bwf) / max(bwf, 1e-300),
            "front_back_asymmetry_difference": asymmetry(t, production) - asymmetry(t, full),
            "peak_phase_difference_rad": phase_difference(full, production),
            "peak_amplitude_difference": float(np.max(np.abs(production) - np.abs(full)) / np.max(np.abs(full))),
            "principal_observable_error": observable_error,
            "gate_error": max(source_error, update_error, observable_error),
            "production_split_update": "raman_convolve_intensity->shock_intensity->kerr_phase_from_deltan->apply_nonlinear",
            "full_reference_update": "Eq27 Heun with stage recomputation",
        })
    with (args.out_dir / "production_split_vs_full_operator.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (args.out_dir / "production_operator_waveform_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    derivation = {
        "paper_prefactor": "i*(omega0^2/c^2)*(n0*n_R/k0_paper)",
        "paper_k0_definition": "n0*omega0/c",
        "code_prefactor": K_MEDIUM * NR,
        "full_reference_prefactor": K_VAC * NR,
        "relative_difference": abs(K_MEDIUM - K_VAC) / K_VAC,
        "selected_candidate_prefactor": K_VAC * NR,
        "reason": "n0 cancels after substituting p_rot into Eq.7/Eq.27",
        "field_envelope_mapping": "paper A and code E are complex electric-field envelopes",
        "intensity_mapping": "I=0.5*eps0*c*n0*|E|^2",
        "cross_cutting_electronic_kerr_operator_issue": True,
    }
    (args.out_dir / "isaacs_operator_prefactor.json").write_text(json.dumps(derivation, indent=2) + "\n")
    (args.out_dir / "isaacs_operator_prefactor_derivation.md").write_text(
        "# Isaacs rotational operator prefactor\n\n"
        "Using `p_rot=(n0 n_R/2pi) I_R A` in Eq. (7)/(27) and paper "
        "`k0=n0 omega0/c`, the `n0` factors cancel. The rotational field RHS is "
        "`i (omega0/c) n_R (1+i/omega0 d_tau)[I_R A]`. The existing split phase "
        "uses the medium wavenumber and differs by `n0-1`; the full candidate uses "
        "the vacuum wavenumber. The analogous electronic-Kerr product-operator issue "
        "is recorded but not changed in Phase 8A.1.\n", encoding="utf-8")
    fig, ax = plt.subplots()
    ax.bar([row["waveform"] for row in rows], [row["gate_error"] for row in rows])
    ax.axhline(.01, color="k", ls="--", label="TL 1% gate")
    ax.axhline(.02, color="gray", ls=":", label="chirped/asymmetric 2% gate")
    ax.tick_params(axis="x", rotation=35); ax.set(ylabel="maximum comparator error", title="Actual production split vs full Isaacs operator")
    ax.legend(); fig.tight_layout(); fig.savefig(args.out_dir / "production_split_vs_full_operator.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
