from __future__ import annotations

import csv
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_corrected_fft_eq11_and_iir_evidence_are_separate(tmp_path):
    out = tmp_path / "closure"
    subprocess.run([sys.executable, str(ROOT / "tools" / "validate_isaacs_eq10_eq11.py"), "--out-dir", str(out)], check=True)
    fft = list(csv.DictReader((out / "raman_fft_direct_comparison.csv").open()))
    assert max(float(row["relative_linf_error"]) for row in fft if row["dtype"] == "float64") < 1e-10
    assert max(float(row["relative_linf_error"]) for row in fft if row["dtype"] == "float32") < 1e-5
    validation = list(csv.DictReader((out / "eq10_eq11_validation_v2.csv").open()))
    required = {"direct_vs_eq11_error", "fft_vs_eq11_error", "iir_vs_eq11_error", "iir_vs_direct_error"}
    assert required.issubset(validation[0]) and "relative_error" not in validation[0]
    refined = [row for row in validation if abs(float(row["dt_fs"]) - .15625) < 1e-12 and int(float(row["pulse_fs"])) in (40, 120)]
    assert max(float(row["iir_vs_direct_error"]) for row in refined) < .01
    assert max(abs(int(row["iir_peak_time_shift_samples"])) for row in refined) <= 1


def test_exact_piecewise_linear_iir_improves_with_refinement(tmp_path):
    out = tmp_path / "closure"
    subprocess.run([sys.executable, str(ROOT / "tools" / "validate_isaacs_eq10_eq11.py"), "--out-dir", str(out)], check=True)
    rows = list(csv.DictReader((out / "raman_iir_direct_convergence.csv").open()))
    selected = [row for row in rows if row["iir_sampling"] == "exact_piecewise_linear" and row["pulse_fs"] == "40"]
    selected.sort(key=lambda row: float(row["dt_fs"]), reverse=True)
    assert float(selected[-1]["iir_vs_direct_error"]) < float(selected[0]["iir_vs_direct_error"])
