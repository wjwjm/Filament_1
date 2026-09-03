from __future__ import annotations

import json

import numpy as np
import pytest

from KHz_filament.device import to_cpu
from KHz_filament.hr4e_real_spatial import build_e2c_validation_state
from KHz_filament.hr4e_timestep import (
    e1a_geometry,
    e1b_geometry_translation,
    e1b_source_grid,
    sha256_array,
    sha256_file,
)
from tools.preflight_hr4e2c_real import rows


def test_validation_adapter_is_deterministic_and_preserves_zero_velocity(tmp_path):
    source = np.zeros((351, 301), dtype=np.float64)
    source[175, 150] = -1.0e-5
    source_path = tmp_path / "screen_peak_delta_n.npy"
    np.save(source_path, source, allow_pickle=False)
    manifest = {
        "source_grid": e1b_source_grid(), "target_grid": e1a_geometry(),
        "geometry_translation": e1b_geometry_translation(), "n0": 1.00027,
        "source_dtype": "float64", "source_backend": "cpu", "source_git_sha": "frozen-source",
        "hr3b_state_file_sha256": "state-file", "hr3b_state_sha256": "state-array",
        "screens": {"peak": {"array_path": source_path.name, "file_sha256": sha256_file(source_path), "array_sha256": sha256_array(source), "shape": [351, 301], "dtype": "float64", "index": 7, "z_m": 0.8}},
    }
    manifest_path = tmp_path / "post_reference_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    identity = {"screen_id": "peak", "screen_index": 7, "screen_z_m": 0.8}
    first = build_e2c_validation_state(str(source_path), source_manifest_path=str(manifest_path), screen_identity=identity, spacing_m=5.0e-6)
    second = build_e2c_validation_state(str(source_path), source_manifest_path=str(manifest_path), screen_identity=identity, spacing_m=5.0e-6)
    grids = [build_e2c_validation_state(str(source_path), source_manifest_path=str(manifest_path), screen_identity=identity, spacing_m=spacing) for spacing in (20.0e-6, 10.0e-6, 5.0e-6)]
    assert first["target_state_sha256"] == second["target_state_sha256"]
    assert first["validation_representation"]["production_multigrid_mapping_modified"] is False
    assert first["validation_representation"]["scope_is_hydro_only_validation"] is True
    assert "proof_of_no_10um_upsampling" not in first["validation_representation"]
    assert np.all(to_cpu(first["state"]["vx"]) == 0.0)
    assert np.all(to_cpu(first["state"]["vy"]) == 0.0)
    assert all(item["source_provenance"] == first["source_provenance"] for item in grids)
    assert len({tuple((item["geometry"][key] for key in ("x_min_m", "x_max_m", "y_min_m", "y_max_m"))) for item in grids}) == 1


def test_validation_adapter_rejects_incomplete_identity(tmp_path):
    source = np.zeros((351, 301), dtype=np.float64)
    source_path = tmp_path / "screen_peak_delta_n.npy"
    np.save(source_path, source, allow_pickle=False)
    manifest = {
        "source_grid": e1b_source_grid(), "target_grid": e1a_geometry(),
        "geometry_translation": e1b_geometry_translation(), "n0": 1.00027,
        "source_dtype": "float64", "source_backend": "cpu", "source_git_sha": "frozen-source",
        "hr3b_state_file_sha256": "state-file", "hr3b_state_sha256": "state-array",
        "screens": {"peak": {"array_path": source_path.name, "file_sha256": sha256_file(source_path), "array_sha256": sha256_array(source), "shape": [351, 301], "dtype": "float64", "index": 7, "z_m": 0.8}},
    }
    manifest_path = tmp_path / "post_reference_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="complete screen_id"):
        build_e2c_validation_state(str(source_path), source_manifest_path=str(manifest_path), screen_identity={"screen_id": "peak"}, spacing_m=5.0e-6)


def _preflight_state(**metrics):
    return {"initial_metrics": metrics}


def test_preflight_accepts_tolerated_nonmonotonic_bilinear_widths():
    states = [
        _preflight_state(xc_m=0.0, yc_m=0.0, sigma_x_m=100.0e-6, sigma_y_m=100.0e-6, min_delta_n=-1.0e-5, M0_negative_index_m2=1.0e-12),
        _preflight_state(xc_m=0.0, yc_m=0.0, sigma_x_m=100.1e-6, sigma_y_m=100.1e-6, min_delta_n=-1.0e-5, M0_negative_index_m2=1.0e-12),
        _preflight_state(xc_m=0.0, yc_m=0.0, sigma_x_m=100.6e-6, sigma_y_m=100.5e-6, min_delta_n=-1.001e-5, M0_negative_index_m2=1.001e-12),
    ]
    sigma_x = next(row for row in rows(states) if row["observable"] == "sigma_x_m")
    assert sigma_x["mapping_consistency_pass"] is True
    assert sigma_x["diagnostic_warning"] == "WARNING_NONMONOTONIC_WITHIN_TOLERANCE"
    assert rows(states) == rows(states)


def test_preflight_rejects_materially_out_of_tolerance_mapping():
    states = [
        _preflight_state(xc_m=0.0, yc_m=0.0, sigma_x_m=100.0e-6, sigma_y_m=100.0e-6, min_delta_n=-1.0e-5, M0_negative_index_m2=1.0e-12),
        _preflight_state(xc_m=0.0, yc_m=0.0, sigma_x_m=100.0e-6, sigma_y_m=100.0e-6, min_delta_n=-1.0e-5, M0_negative_index_m2=1.0e-12),
        _preflight_state(xc_m=0.0, yc_m=0.0, sigma_x_m=102.0e-6, sigma_y_m=100.0e-6, min_delta_n=-1.0e-5, M0_negative_index_m2=1.0e-12),
    ]
    sigma_x = next(row for row in rows(states) if row["observable"] == "sigma_x_m")
    assert sigma_x["mapping_consistency_pass"] is False
