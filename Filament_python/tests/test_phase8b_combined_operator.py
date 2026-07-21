from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "isaacs_raman_closure" / "phase8b_preflight"


def test_combined_operator_artifacts_pass_refined_order_and_production_step_gates():
    summary = json.loads((OUT / "combined_operator_summary.json").read_text(encoding="utf-8"))
    assert summary["refined_estimated_order"] >= 1.5
    production = summary["production_vs_dz2"]
    assert production["field_l2_difference"] < 1e-3
    assert production["I_max_relative_difference"] < 2e-3
    assert production["rho_max_relative_difference"] < 5e-3
    assert production["raman_loss_relative_difference"] < 5e-3
    assert all((OUT / name).is_file() for name in (
        "combined_operator_order_comparison.csv",
        "combined_operator_dz_convergence.csv",
        "combined_operator_observable_comparison.csv",
    ))


def test_formal_configs_select_strang_but_remain_single_factor():
    config_dir = ROOT / "configs" / "isaacs_raman_closure"
    on = json.loads((config_dir / "120fs_talebpour_isaacs_full_operator_on.json").read_text())
    off = json.loads((config_dir / "120fs_talebpour_isaacs_full_operator_feedback_off.json").read_text())
    assert on["raman"]["nonlinear_split_order"] == "strang"
    assert off["raman"]["nonlinear_split_order"] == "strang"
    diff = json.loads((OUT / "full_operator_on_vs_off_config_diff.json").read_text())
    assert [row["path"] for row in diff["differences"]] == ["propagation.use_raman_full_operator"]


def test_split_order_validation_rejects_unknown_value():
    from KHz_filament.config_normalize import normalize_config

    with pytest.raises(ValueError, match="nonlinear_split_order"):
        normalize_config({"raman": {"nonlinear_split_order": "frozen_fake_strang"}})


def test_convergence_csv_records_preasymptotic_and_refined_behavior():
    rows = list(csv.DictReader((OUT / "combined_operator_dz_convergence.csv").open()))
    orders = [float(row["estimated_order"]) for row in rows if row["estimated_order"]]
    assert orders[-1] >= 1.5
    assert min(orders) < orders[-1]
