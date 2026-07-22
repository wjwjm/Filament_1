#!/usr/bin/env python3
"""Reference comparison for individually selectable BK-NEE precision strategies."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from KHz_filament.linear import step_linear_bk_nee_factorized  # noqa: E402


def _axes(nt=32, nxy=16):
    dt, dx = 2.5e-15, 12e-6
    t = (np.arange(nt) - nt // 2) * dt
    x = (np.arange(nxy) - nxy // 2) * dx
    xx, yy = np.meshgrid(x, x, indexing="xy")
    omega = 2.0 * np.pi * np.fft.fftfreq(nt, dt)
    k = 2.0 * np.pi * np.fft.fftfreq(nxy, dx)
    return t, xx, yy, omega, k[:, None] ** 2 + k[None, :] ** 2, dt, dx


def _field(kind: str):
    t, xx, yy, omega, k2, dt, dx = _axes()
    temporal = np.exp(-0.5 * (t / 45e-15) ** 2)
    if kind == "positive_chirp":
        temporal = temporal * np.exp(1j * 2.2e27 * t**2)
    elif kind == "negative_chirp":
        temporal = temporal * np.exp(-1j * 2.2e27 * t**2)
    elif kind == "asymmetric":
        temporal = temporal * (1.0 + 0.35 * np.tanh((t + 12e-15) / 18e-15))
    spatial = np.exp(-(xx**2 + yy**2) / (2.0 * (42e-6) ** 2))
    if kind == "high_k":
        spatial = spatial * np.exp(1j * 0.55 * np.pi * (xx + yy) / 12e-6)
    elif kind == "edge_localized":
        spatial = np.exp(-((xx - 70e-6) ** 2 + (yy + 55e-6) ** 2) / (2.0 * (20e-6) ** 2))
    elif kind == "production_shape":
        spatial = np.maximum(0.0, 1.0 - (xx**2 + yy**2) / (92e-6) ** 2)
    return (temporal[:, None, None] * spatial[None, :, :]).astype(np.complex64), omega, k2, t, dt, dx


def _run(field, omega, k2, strategy: str, repeats: int):
    out = field.copy()
    omega0 = 2.0 * np.pi * 299792458.0 / 800e-9
    for _ in range(repeats):
        out = step_linear_bk_nee_factorized(
            out, Omega=omega, kperp2=k2, k0=7.856e6, omega0=omega0,
            dz=5e-5, beta2=0.2e-28, precision_strategy=strategy,
        )
    return np.asarray(out)


def _metrics(reference, candidate, t, dt, dx):
    ref = np.asarray(reference)
    got = np.asarray(candidate)
    denom = max(float(np.linalg.norm(ref.ravel())), 1e-300)
    l2 = float(np.linalg.norm((got - ref).ravel()) / denom)
    weight = np.abs(ref) * np.abs(got)
    phase = np.angle(got * np.conj(ref))
    phase_error = float(np.sqrt(np.sum(weight * phase**2) / max(float(np.sum(weight)), 1e-300)))
    intensity = np.abs(got) ** 2
    energy = float(np.sum(intensity, dtype=np.float64) * dt * dx * dx)
    ref_energy = float(np.sum(np.abs(ref) ** 2, dtype=np.float64) * dt * dx * dx)
    temporal = np.sum(intensity, axis=(1, 2), dtype=np.float64)
    norm = max(float(np.sum(temporal)), 1e-300)
    temporal_centroid = float(np.sum(t * temporal) / norm)
    temporal_width = float(np.sqrt(np.sum((t - temporal_centroid) ** 2 * temporal) / norm))
    spectrum = np.sum(np.abs(np.fft.fft(got, axis=0)) ** 2, axis=(1, 2), dtype=np.float64)
    omega = 2 * np.pi * np.fft.fftfreq(t.size, dt)
    snorm = max(float(np.sum(spectrum)), 1e-300)
    spectral_centroid = float(np.sum(omega * spectrum) / snorm)
    spectral_width = float(np.sqrt(np.sum((omega - spectral_centroid) ** 2 * spectrum) / snorm))
    return {
        "relative_l2": l2,
        "phase_rms_rad": phase_error,
        "relative_energy_error": float(abs(energy - ref_energy) / max(abs(ref_energy), 1e-300)),
        "temporal_centroid_s": temporal_centroid,
        "temporal_width_s": temporal_width,
        "spectral_centroid_rad_s": spectral_centroid,
        "spectral_width_rad_s": spectral_width,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    cases = ("tl_gaussian", "positive_chirp", "negative_chirp", "asymmetric", "high_k", "edge_localized", "production_shape")
    candidates = ("baseline_complex64", "orthonormal_fft", "mixed_precision", "unitary_projection")
    rows = []
    for case in cases:
        field, omega, k2, t, dt, dx = _field(case)
        for repeats in (1, 40, 400):  # half steps corresponding to 1/20/200 full z steps
            reference = _run(field.astype(np.complex128), omega, k2, "baseline_complex64", repeats)
            for candidate in candidates:
                metrics = _metrics(reference, _run(field, omega, k2, candidate, repeats), t, dt, dx)
                rows.append({"case": case, "full_steps": repeats // 2 if repeats > 1 else 0.5,
                             "halfsteps": repeats, "candidate": candidate, **metrics})
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    assessment = {}
    for candidate in candidates:
        subset = [r for r in rows if r["candidate"] == candidate]
        one = [r["relative_l2"] for r in subset if r["halfsteps"] == 1]
        long = [r["relative_l2"] for r in subset if r["halfsteps"] == 400]
        assessment[candidate] = {
            "max_one_halfstep_relative_l2": max(one),
            "max_200_step_relative_l2": max(long),
            "one_halfstep_gate_lt": 1e-6,
            "two_hundred_step_gate_lt": 1e-4,
            "passed": max(one) < 1e-6 and max(long) < 1e-4,
        }
    args.json.write_text(json.dumps({"schema": "khz_filament.phase8b_r.r5_operator_reference.v1", "assessment": assessment}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
