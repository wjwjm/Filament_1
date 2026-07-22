from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _tool():
    path = ROOT / "tools" / "run_r4_linear_halfstep_smoke.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_smoke_analysis_reports_unaccounted_linear_loss_with_fixed_sign():
    tool = _tool()
    n = 3
    data = {"z_axis": np.arange(n)}
    for half in (1, 2):
        data[f"linear_halfstep_{half}_energy_before_J"] = np.full(n, 2.0)
        data[f"linear_halfstep_{half}_field_delta_J"] = np.full(n, -1e-7)
        data[f"linear_halfstep_{half}_unaccounted_residual_J"] = np.full(n, -1e-7)
        for key in ("explicit_boundary_loss_J", "explicit_spectral_filter_loss_J", "explicit_crop_loss_J", "explicit_evanescent_loss_J", "explicit_other_loss_J"):
            data[f"linear_halfstep_{half}_{key}"] = np.zeros(n)
    result = tool.analyze(data, mode="pure_linear")
    assert result["linear_cumulative_unaccounted_residual_J"] < 0.0
    assert result["linear_cumulative_unaccounted_relative"] > 0.0
    assert result["pure_lossless_contract"]["passed"] is True
