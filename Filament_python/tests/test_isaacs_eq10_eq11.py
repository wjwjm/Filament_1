from __future__ import annotations

import csv
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_eq10_eq11_static_audit_writes_converged_artifacts(tmp_path):
    out = tmp_path / "closure"
    subprocess.run([sys.executable, str(ROOT / "tools" / "validate_isaacs_eq10_eq11.py"), "--out-dir", str(out)], check=True)
    rows = list(csv.DictReader((out / "eq10_eq11_validation.csv").open()))
    refined = [float(r["relative_error"]) for r in rows if abs(float(r["dt_fs"]) - .3125) < 1e-9 and r["path"] in ("direct", "fft_linear")]
    assert refined and max(refined) < .01
    assert all((out / name).is_file() for name in ("eq10_signed_energy_validation.csv", "eq10_eq11_convergence.csv", "eq10_eq11_comparison.png"))
