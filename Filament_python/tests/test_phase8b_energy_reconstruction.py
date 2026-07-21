from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _tool():
    path = ROOT / "tools" / "audit_phase8b_energy_reconstruction.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reconstruction_audit_keeps_runtime_consistency_separate_from_missing_field_history():
    tool = _tool()
    data = {
        "z_axis": np.array([0.5, 1.0]),
        "U_z": np.array([0.98, 0.95]),
        "U_step_change_z": np.array([-0.02, -0.03]),
        "U_rel_change_z": np.array([-0.02, -0.05]),
        "E_loss_from_input_z": np.array([0.02, 0.05]),
        "E_dep_total_z": np.array([0.015, 0.025]),
        "E_dep_cumulative_z": np.array([0.015, 0.04]),
        "dz_used_z": np.array([0.5, 0.5]),
        "t_axis": np.array([-1.0, 0.0, 1.0]),
        "x": np.array([-1.0, 1.0]),
        "y": np.array([-1.0, 1.0]),
        "I_out_center_t": np.ones(3),
    }
    result = tool.audit_reconstruction(data, expected_audit={"checks": [{"name": "total_energy_final", "actual": 0.01}]})
    assert not result["archive_field_reintegration"]["available"]
    assert result["archive_field_reintegration"]["status"] == "inconclusive_missing_full_field_history"
    assert result["runtime_energy_diagnostics"]["max_U_step_change_mismatch_J"] == pytest.approx(0.0, abs=1e-15)
    assert result["runtime_energy_diagnostics"]["reconstructed_final_total_closure"] == pytest.approx(0.01)
    assert result["alignment"]["last_step"]["last_step_deposition_matches_cumulative_increment_J"] == pytest.approx(0.0, abs=1e-15)
