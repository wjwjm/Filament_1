from __future__ import annotations

import csv
import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from KHz_filament.nonlinear import operator_correct_scalar


def test_legacy_fft_behavior_is_preserved_and_strict_mode_is_consistent():
    nt, dt = 1024, 1e-15
    omega = 2 * np.pi * np.fft.fftfreq(nt, dt)
    signal = np.sin(2 * np.pi * 5 * np.arange(nt) / nt)[:, None, None]
    legacy = np.asarray(operator_correct_scalar(signal, omega, 2e15, method="fft", operator_convention="legacy"))
    expected_legacy = np.fft.ifft((1 + 1j * omega / 2e15) * np.fft.fft(signal[:, 0, 0])).real
    assert np.allclose(legacy[:, 0, 0], expected_legacy)
    strict_fft = np.asarray(operator_correct_scalar(signal, omega, 2e15, method="fft", operator_convention="isaacs_eq27"))
    strict_tdiff = np.asarray(operator_correct_scalar(signal, omega, 2e15, dt=dt, method="tdiff", operator_convention="isaacs_eq27"))
    assert np.max(np.abs(strict_fft - strict_tdiff)) / np.max(np.abs(strict_fft)) < 1e-4


def test_time_derivative_audit_closes_positive_and_negative_frequency(tmp_path):
    out = tmp_path / "audit"
    subprocess.run([sys.executable, str(ROOT / "tools" / "audit_time_derivative_convention.py"), "--out-dir", str(out)], check=True)
    rows = list(csv.DictReader((out / "time_derivative_validation.csv").open()))
    assert len(rows) == 4
    assert max(float(row["analytic_vs_fft_derivative_error"]) for row in rows) < 1e-10
    assert max(float(row["analytic_vs_tdiff_derivative_error"]) for row in rows) < 1e-4
    summary = json.loads((out / "time_derivative_convention.json").read_text())
    assert summary["F[d_tau f]"] == "+i Omega F[f]"
    assert summary["tdiff_fft_mutually_consistent"] is True
