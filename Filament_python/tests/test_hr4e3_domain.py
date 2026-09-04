from __future__ import annotations

import json

import numpy as np
import pytest

from KHz_filament.hr4e_domain import (
    E3_DOMAINS, E3_SNAPSHOT_TIMES_S, build_e3_real_post_state, build_e3_synthetic_state,
    common_d0_field_metrics, e3_geometry, initial_edge_validity, nested_d0_slices,
    read_e3_checkpoint, write_e3_checkpoint,
)
from KHz_filament.hr4e_timestep import e1a_geometry, e1b_geometry_translation, e1b_source_grid, sha256_array, sha256_file
from tools.summarize_hr4e3_domain import _difference, classify_final


def _source_fixture(tmp_path, *, edge_value: float = 0.0):
    source = np.zeros((351, 301), dtype=np.float64)
    source[175, 150] = -1.0e-5
    source[1, 150] = edge_value
    source_path = tmp_path / "screen_peak_delta_n.npy"
    np.save(source_path, source, allow_pickle=False)
    manifest = {"source_grid": e1b_source_grid(), "target_grid": e1a_geometry(), "geometry_translation": e1b_geometry_translation(), "n0": 1.00027, "source_dtype": "float64", "source_backend": "cpu", "source_git_sha": "frozen", "hr3b_state_file_sha256": "full-file", "hr3b_state_sha256": "full-array", "screens": {"peak": {"array_path": source_path.name, "file_sha256": sha256_file(source_path), "array_sha256": sha256_array(source), "shape": [351, 301], "dtype": "float64", "index": 7, "z_m": 0.8}}}
    manifest_path = tmp_path / "post_reference_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source, source_path, manifest_path


def test_e3_domain_geometries_are_inclusive_and_exactly_nested():
    assert {name: (e3_geometry(name)["Nx"], e3_geometry(name)["Ny"]) for name in E3_DOMAINS} == {"D0": (301, 351), "D1": (401, 351), "D2": (301, 501), "D3": (401, 501)}
    for name in E3_DOMAINS:
        assert nested_d0_slices(e3_geometry(name))["aligned"] is True


def test_synthetic_initial_state_is_directly_evaluated_on_each_domain():
    values = [build_e3_synthetic_state(e3_geometry(name)) for name in E3_DOMAINS]
    assert all(item["delta_n"].dtype == np.dtype("float64") for item in values)
    assert all(np.array_equal(np.asarray(item["vx"]), np.zeros_like(np.asarray(item["vx"]))) for item in values)


def test_real_post_zero_padding_preserves_d0_and_exterior_zero(tmp_path):
    source, source_path, manifest_path = _source_fixture(tmp_path)
    prepared = build_e3_real_post_state(str(source_path), source_manifest_path=str(manifest_path), screen_identity={"screen_id": "peak", "screen_index": 7, "screen_z_m": 0.8}, geometry=e3_geometry("D3"))
    slices = prepared["extension"]["d0_slices"]
    padded = np.asarray(prepared["state"]["delta_n"])
    assert np.array_equal(padded[slices["y_start"]:slices["y_stop"], slices["x_start"]:slices["x_stop"]], source)
    assert np.count_nonzero(padded[:slices["y_start"], :]) == 0
    assert prepared["initial_edge_validity"]["status"] == "PASS"


def test_real_post_rejects_non_negligible_first_inner_ring(tmp_path):
    _, source_path, manifest_path = _source_fixture(tmp_path, edge_value=-2.0e-8)
    with pytest.raises(ValueError, match="E3B_INVALID_INITIAL_DOMAIN_TRUNCATION"):
        build_e3_real_post_state(str(source_path), source_manifest_path=str(manifest_path), screen_identity={"screen_id": "peak", "screen_index": 7, "screen_z_m": 0.8}, geometry=e3_geometry("D1"))


def test_checkpoint_preserves_float64_hashable_arrays_and_exact_readback(tmp_path):
    geometry = e3_geometry("D0")
    state = {"delta_n": np.zeros((351, 301), dtype=np.float64), "vx": np.ones((351, 301), dtype=np.float64), "vy": np.full((351, 301), -2.0, dtype=np.float64)}
    receipt = write_e3_checkpoint(state, geometry, 100e-6, tmp_path / "checkpoint.npz")
    replay = read_e3_checkpoint(tmp_path / "checkpoint.npz")
    assert receipt["sha256"]
    assert all(replay[name].dtype == np.dtype("float64") and np.array_equal(replay[name], state[name]) for name in ("delta_n", "vx", "vy"))


def test_common_d0_field_restriction_and_norms(tmp_path):
    d0, d3 = e3_geometry("D0"), e3_geometry("D3")
    reference = np.zeros((351, 301), dtype=np.float64)
    larger = np.zeros((501, 401), dtype=np.float64)
    sl = nested_d0_slices(d3)
    larger[sl["y_start"]:sl["y_stop"], sl["x_start"]:sl["x_stop"]] = reference
    write_e3_checkpoint({"delta_n": reference, "vx": reference, "vy": reference}, d0, 100e-6, tmp_path / "d0.npz")
    write_e3_checkpoint({"delta_n": larger, "vx": larger, "vy": larger}, d3, 100e-6, tmp_path / "d3.npz")
    report = common_d0_field_metrics(tmp_path / "d0.npz", tmp_path / "d3.npz")
    assert report["alignment"]["aligned"] is True
    assert report["fields"]["delta_n"]["relative_L2"] == 0.0


def test_scalar_tolerance_and_final_decision_do_not_invent_field_thresholds():
    assert _difference(1.0e-3, 1.001e-3, "sigma_x_m")["pass"] is True
    assert _difference(1.0e-3, 1.02e-3, "sigma_x_m")["pass"] is False
    assert classify_final(True)["final_class"] == "PENDING_MANUAL_FIELD_REVIEW"
    assert classify_final(False)["final_class"] == "E3-E_INVALID_OR_INCONCLUSIVE"


def test_checkpoint_policy_is_limited_to_the_three_authorized_sparse_times():
    assert E3_SNAPSHOT_TIMES_S == (0.0, 100.0e-6, 1.0e-3)
