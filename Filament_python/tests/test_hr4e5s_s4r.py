from __future__ import annotations

import hashlib
import json

import numpy as np

from KHz_filament.hr4e5s_s4r import S4R_SCREEN_COUNT, prepare_input_manifest
from KHz_filament.hr4e_timestep import sha256_array


def _file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s4r_manifest_selects_genuine_peak_centred_384_screen_window(tmp_path):
    state = np.zeros((15000, 1, 1), dtype=np.float64)
    state[8022, 0, 0] = -1.0
    state_path = tmp_path / "state.npy"; np.save(state_path, state)
    config = tmp_path / "config.json"; config.write_text("{}\n", encoding="utf-8")
    source = {"hr3b_state_file_sha256": _file_sha(state_path), "hr3b_state_sha256": sha256_array(state),
              "config_sha256": _file_sha(config), "source_z_positions_m": list(np.arange(15000, dtype=np.float64)), "n0": 1.00027}
    source_path = tmp_path / "source.json"; source_path.write_text(json.dumps(source), encoding="utf-8")
    output = tmp_path / "input.json"
    result = prepare_input_manifest(source_manifest_path=source_path, source_state_path=state_path, config_path=config, out_path=output)
    assert result["screen_indices"] == list(range(7830, 8214))
    assert len(result["screen_records"]) == S4R_SCREEN_COUNT
    assert result["screen_records"][0]["source_index"] == 7830
    assert result["screen_records"][-1]["source_index"] == 8213
    assert 7998 in result["screen_indices"] and 8045 in result["screen_indices"]
    assert result["hydro"]["block_size"] == 8
