from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "tools"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from KHz_filament.diagnostics import Z_HISTORY_TRACE_KEYS
from validate_raman_phase_ablation import validate_raman_phase_off


def _config(tmp_path: pathlib.Path) -> pathlib.Path:
    data = json.loads((ROOT / "configs" / "raman_phase_causality" / "120fs_talebpour_full_model_raman_phase_off.json").read_text(encoding="utf-8"))
    data["propagation"]["z_max"] = 1.3
    path = tmp_path / "config.json"; path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _npz(path: pathlib.Path, *, applied: float = 0.0) -> None:
    arrays = {"z_axis": np.asarray([0.9, 1.3])}
    for key in Z_HISTORY_TRACE_KEYS:
        arrays[key] = np.zeros(2)
    arrays.update({
        "rho_N2_max_z": np.asarray([0.0, 1e20]), "rho_O2_max_z": np.asarray([0.0, 1e20]),
        "rho_O2_fraction_at_rho_total_max_z": np.asarray([0.0, 0.5]), "dz_used_z": np.asarray([1e-4, 1e-4]),
        "adaptive_rejection_count_z": np.asarray([0, 0]), "safety_mode_trigger_count_z": np.asarray([0, 0]),
        "safety_mode_event_summary": np.asarray("{}"), "propagation_observability_schema": np.asarray("khz_filament.propagation_observability.v1"),
        "diagnostic_validation_passed": np.asarray(True), "IR_max_z": np.asarray([1.0, 2.0]),
        "delta_n_rot_max_z": np.asarray([1e-7, 2e-7]), "delta_n_rot_applied_max_z": np.asarray([applied, applied]),
        "dphi_rot_max_abs_z": np.asarray([1e-4, 2e-4]), "dphi_rot_applied_max_abs_z": np.asarray([applied, applied]),
        "alpha_R_raw_max_z": np.asarray([1e-4, 2e-4]), "alpha_R_applied_max_z": np.asarray([1e-4, 2e-4]),
    })
    np.savez(path, **arrays)


def _metadata(path: pathlib.Path, config: pathlib.Path) -> None:
    path.write_text(json.dumps({"execution_git_sha": "8dcd01e", "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()}), encoding="utf-8")


def test_raman_phase_off_audit_accepts_raw_and_absorption_with_zero_applied_phase(tmp_path):
    config = _config(tmp_path); npz = tmp_path / "result.npz"; _npz(npz); metadata = tmp_path / "metadata.json"; _metadata(metadata, config)
    assert validate_raman_phase_off(npz, config, metadata)["passed"] is True


def test_raman_phase_off_audit_rejects_nonzero_applied_phase(tmp_path):
    config = _config(tmp_path); npz = tmp_path / "result.npz"; _npz(npz, applied=1e-7); metadata = tmp_path / "metadata.json"; _metadata(metadata, config)
    result = validate_raman_phase_off(npz, config, metadata)
    assert result["passed"] is False
    assert any("applied diagnostic is not zero" in item for item in result["failures"])
