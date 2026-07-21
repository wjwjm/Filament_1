from __future__ import annotations

import csv
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_production_comparator_invokes_real_split_and_full_product(tmp_path):
    out = tmp_path / "operator"
    subprocess.run([sys.executable, str(ROOT / "tools" / "compare_production_raman_operator.py"), "--out-dir", str(out)], check=True)
    rows = list(csv.DictReader((out / "production_split_vs_full_operator.csv").open()))
    assert len(rows) == 6
    assert all("shock_intensity" in row["production_split_update"] for row in rows)
    assert all(row["full_reference_update"].startswith("Eq27 Heun") for row in rows)
    assert all(float(row["front_back_asymmetry_difference"]) == float(row["front_back_asymmetry_difference"]) for row in rows)
    prefactor = json.loads((out / "isaacs_operator_prefactor.json").read_text())
    assert prefactor["relative_difference"] < 1e-3
    assert prefactor["selected_candidate_prefactor"] == prefactor["full_reference_prefactor"]


def test_old_surrogate_comparator_has_no_production_gate_output():
    source = (ROOT / "tools" / "compare_isaacs_raman_operator.py").read_text()
    assert "production_split_vs_full_operator.csv" not in source
