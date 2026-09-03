from __future__ import annotations

import json

import numpy as np

from KHz_filament.device import to_cpu
from KHz_filament.hr4e_real_spatial import build_e2c_validation_state
from KHz_filament.hr4e_timestep import (
    e1a_geometry,
    e1b_geometry_translation,
    e1b_source_grid,
    sha256_array,
    sha256_file,
)


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
    assert first["target_state_sha256"] == second["target_state_sha256"]
    assert first["validation_representation"]["production_multigrid_mapping_modified"] is False
    assert first["validation_representation"]["scope_is_hydro_only_validation"] is True
    assert np.all(to_cpu(first["state"]["vx"]) == 0.0)
    assert np.all(to_cpu(first["state"]["vy"]) == 0.0)
