from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _tool():
    path = ROOT / "tools" / "analyze_phase8b_job1_energy_budget.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_signed_energy_budget_reconstructs_channel_and_residual_histories():
    tool = _tool()
    data = {
        "z_axis": np.array([0.5, 1.0, 1.3]),
        "U_z": np.array([0.97, 0.93, 0.90]),
        "U_step_change_z": np.array([-0.03, -0.04, -0.03]),
        "E_dep_z": np.array([0.02, 0.03, 0.02]),
        "raman_actual_loss_step_J": np.array([0.01, 0.005, 0.01]),
        "raman_target_loss_step_J": np.array([0.01, 0.005, 0.01]),
        "raman_target_loss_cumulative_J": np.array([0.01, 0.015, 0.025]),
        "raman_actual_loss_cumulative_J": np.array([0.01, 0.015, 0.025]),
        "E_dep_total_z": np.array([0.03, 0.035, 0.03]),
        "E_dep_cumulative_z": np.array([0.03, 0.065, 0.095]),
        "alpha_R_applied_max_z": np.zeros(3),
        "alpha_ib_max_z": np.zeros(3),
    }
    budget = tool.build_energy_budget(data)
    assert budget["initial_energy_J"] == 1.0
    np.testing.assert_allclose(budget["field_loss_step_J"], [0.03, 0.04, 0.03])
    np.testing.assert_allclose(budget["total_accounted_step_J"], [0.03, 0.035, 0.03])
    np.testing.assert_allclose(budget["unaccounted_boundary_filter_or_numerical_step_J"], [0.0, 0.005, 0.0], atol=1e-15)
    np.testing.assert_allclose(budget["unaccounted_boundary_filter_or_numerical_cumulative_J"], [0.0, 0.005, 0.005], atol=1e-15)
    summary = tool.summarize_budget(budget)
    assert summary["final"]["relative_closure"] == pytest.approx(0.005)
    assert summary["ib_channel"].startswith("zero")
    assert summary["legacy_alpha_channel"].startswith("zero")
