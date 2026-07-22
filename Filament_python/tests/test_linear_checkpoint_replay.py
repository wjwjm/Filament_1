from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _tool():
    path = ROOT / "tools" / "audit_linear_checkpoint_replay.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_scalar_energy_archive_is_not_misrepresented_as_field_replay(tmp_path):
    tool = _tool()
    n = 8
    path = tmp_path / "scalar_only.npz"
    np.savez(
        path,
        z_axis=np.linspace(0.1, 0.8, n),
        I_max_z=np.array([1, 2, 5, 10, 8, 4, 2, 1.0]),
        energy_step_start_J=np.full(n, 1.0),
        energy_after_linear_half1_J=np.full(n, 0.99),
        energy_after_raman_post_J=np.full(n, 0.98),
        energy_after_linear_half2_J=np.full(n, 0.97),
    )
    rows, summary = tool.audit(path)
    assert summary["status"] == "inconclusive_missing_field_checkpoints"
    assert summary["field_replay_performed"] is False
    assert len(rows) == 12
    assert {row["replay_status"] for row in rows} == {"inconclusive_missing_field_checkpoint"}
    assert all(row["U_after_forward_fft_J"] is None for row in rows)
