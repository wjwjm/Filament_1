from __future__ import annotations

import json
import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for item in (ROOT, TOOLS):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from KHz_filament.diagnostics import Z_HISTORY_TRACE_KEYS
from validate_current_observability_baseline import REQUIRED_SCALARS, REQUIRED_Z_FIELDS, validate_npz


def _valid_npz(path: pathlib.Path) -> None:
    arrays = {"z_axis": np.array([0.9, 1.3])}
    for key in Z_HISTORY_TRACE_KEYS:
        arrays[key] = np.array([0.0, 0.0])
    arrays.update({
        "rho_N2_max_z": np.array([0.0, 1e20]), "rho_O2_max_z": np.array([0.0, 1e20]),
        "rho_O2_fraction_at_rho_total_max_z": np.array([0.0, 0.5]), "dz_used_z": np.array([1e-4, 1e-4]),
        "adaptive_rejection_count_z": np.array([0, 0]), "safety_mode_trigger_count_z": np.array([0, 0]),
        "safety_mode_event_summary": np.asarray("{}"), "propagation_observability_schema": np.asarray("khz_filament.propagation_observability.v1"),
        "diagnostic_validation_passed": np.asarray(True),
    })
    np.savez(path, **arrays)


def test_current_observability_audit_accepts_complete_synthetic_npz(tmp_path):
    config = json.loads((ROOT / "configs" / "profile_validation" / "flat_top_90_120fs.json").read_text(encoding="utf-8"))
    config["propagation"]["z_max"] = 1.3
    config_path = tmp_path / "config.json"; config_path.write_text(json.dumps(config), encoding="utf-8")
    npz_path = tmp_path / "result.npz"; _valid_npz(npz_path)
    metadata = tmp_path / "run_metadata.json"; metadata.write_text(json.dumps({"execution_git_sha": "test", "config_sha256": __import__("hashlib").sha256(config_path.read_bytes()).hexdigest()}), encoding="utf-8")
    result = validate_npz(npz_path, config_path, metadata)
    assert result["passed"] is True


def test_current_observability_audit_rejects_missing_mandatory_field(tmp_path):
    config = json.loads((ROOT / "configs" / "profile_validation" / "flat_top_90_120fs.json").read_text(encoding="utf-8"))
    config["propagation"]["z_max"] = 1.3
    config_path = tmp_path / "config.json"; config_path.write_text(json.dumps(config), encoding="utf-8")
    npz_path = tmp_path / "bad.npz"; _valid_npz(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "dz_used_z"}
    np.savez(npz_path, **arrays)
    metadata = tmp_path / "run_metadata.json"; metadata.write_text(json.dumps({"execution_git_sha": "test", "config_sha256": __import__("hashlib").sha256(config_path.read_bytes()).hexdigest()}), encoding="utf-8")
    result = validate_npz(npz_path, config_path, metadata)
    assert result["passed"] is False
    assert any("dz_used_z" in item for item in result["failures"])
