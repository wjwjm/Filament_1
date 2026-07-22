from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _tool():
    path = ROOT / "tools" / "audit_linear_transfer_kernel.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bk_kernel_is_nonzero_and_near_unit_modulus():
    tool = _tool()
    result = tool.bk_kernel_stats(
        omega=np.array([-2.0, 0.0, 2.0]),
        kperp2=np.array([[0.0, 1.0], [2.0, 3.0]]),
        k0=10.0, omega0=100.0, dz_eff=0.01,
        beta2=0.0, denom_floor=1e-4, dtype=np.complex64,
    )
    assert result["zero_bins"] == 0
    assert result["max_abs_abs_H_minus_1"] < 1e-5
    assert result["below_unity"]["1e-05"]["count"] == 0
