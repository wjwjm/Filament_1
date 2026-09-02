from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import KHz_filament.hr4e_timestep as hr4e_timestep
from KHz_filament.hr4e_timestep import (
    E1A_AMPLITUDE,
    E1A_SIGMA_M,
    build_e1a_initial_state,
    build_snapshot_step_schedule,
    e1a_geometry,
    e1b_geometry_translation,
    e1b_source_grid,
    load_e1b_screen,
    sha256_array,
    sha256_file,
    thermal_channel_metrics,
)
from tools.generate_hr4e1_post_reference import select_hr3b_screens
from tools.summarize_hr4e1_timestep import config_diff_guard, convergence_rows
from KHz_filament.hr4e_spatial import (
    E2_COMMON_DT_S,
    build_e2_synthetic_state,
    e2_geometry,
    run_e2_case,
)
from tools.summarize_hr4e2_spatial import spatial_report, temporal_guard


def _case(dt_us: float, *, y_shift: float = 0.0) -> dict:
    from KHz_filament.hr4e_timestep import benchmark_configuration

    snapshots = []
    for time_us in (100.0, 1000.0):
        snapshots.append(
            {
                "time_us": time_us,
                "yc_m": 3.0e-7 + y_shift,
                "sigma_x_m": 80.0e-6 + dt_us * 0.1e-6,
                "sigma_y_m": 81.0e-6 + dt_us * 0.1e-6,
                "min_delta_n": -1.0e-5 - dt_us * 1.0e-8,
                "max_abs_vy_m_s": 1.0e-2 + dt_us * 1.0e-5,
                "max_abs_v_m_s": 1.2e-2 + dt_us * 1.0e-5,
            }
        )
    return {
        "case_id": f"E1A_dt{dt_us:g}us",
        "status": "PASS",
        "configuration": benchmark_configuration(
            dt_hydro=dt_us * 1.0e-6, dtype="float64", backend="numpy", git_sha="test-sha"
        ),
        "stability": {"overall_pass": True},
        "initial_state_sha256": "synthetic-initial-state",
        "dtype": "float64",
        "backend": "numpy",
        "git_sha": "test-sha",
        "snapshots": snapshots,
    }


def test_e1a_geometry_and_gaussian_are_inclusive_nodal():
    geometry = e1a_geometry()
    assert (geometry["Nx"], geometry["Ny"]) == (301, 351)
    state = build_e1a_initial_state(dtype=np.float64)
    assert state["delta_n"].shape == (351, 301)
    assert state["vx"].shape == state["vy"].shape == (351, 301)
    assert state["delta_n"][100, 150] == pytest.approx(-E1A_AMPLITUDE)
    assert np.all(state["vx"] == 0.0) and np.all(state["vy"] == 0.0)


def test_e2_grid_family_preserves_domain_and_evaluates_gaussian_per_grid():
    expected = {20.0e-6: (176, 151), 10.0e-6: (351, 301), 5.0e-6: (701, 601)}
    for spacing, shape in expected.items():
        geometry = e2_geometry(spacing)
        assert geometry["x_min_m"] == pytest.approx(-1.5e-3)
        assert geometry["x_max_m"] == pytest.approx(1.5e-3)
        assert geometry["y_min_m"] == pytest.approx(-1.0e-3)
        assert geometry["y_max_m"] == pytest.approx(2.5e-3)
        state = build_e2_synthetic_state(spacing)
        assert state["delta_n"].shape == shape
        center = int(round(-geometry["y_min_m"] / spacing)), int(round(-geometry["x_min_m"] / spacing))
        assert state["delta_n"][center] == pytest.approx(-E1A_AMPLITUDE)


def test_e2_short_case_records_physical_m0_and_stability():
    case = run_e2_case(
        family="E2-A", spacing_m=20.0e-6, dt_hydro=E2_COMMON_DT_S,
        snapshot_times_s=(0.0, E2_COMMON_DT_S),
    )
    assert case["status"] == "PASS"
    assert case["stability"]["overall_pass"] is True
    assert case["snapshots"][-1]["M0_negative_index_m2"] > 0.0


def _e2_case(dx_m: float, dt_s: float, scale: float) -> dict:
    grid = e2_geometry(dx_m)
    initial = {"kind": "analytic_gaussian", "delta_n_sha256": "same-source"}
    configuration = {
        "family": "E2-A", "grid": grid, "dt_hydro_s": dt_s,
        "operator": {"frozen": True}, "snapshot_times_s": [0.0, 100.0e-6], "initial_state": initial,
    }
    snapshot = {
        "time_us": 100.0, "boundary_contaminated": False,
        "xc_m": 0.0, "yc_m": scale * 1.0e-7,
        "sigma_x_m": 80.0e-6 + scale * 1.0e-8, "sigma_y_m": 81.0e-6 + scale * 1.0e-8,
        "min_delta_n": -1.0e-5 + scale * 1.0e-9, "max_abs_vx_m_s": 0.0,
        "max_abs_vy_m_s": 1.0e-3 + scale * 1.0e-7, "max_abs_v_m_s": 1.0e-3 + scale * 1.0e-7,
        "M0_negative_index_m2": 1.0e-12 + scale * 1.0e-15,
    }
    return {"case_id": f"E2_{dx_m}", "status": "PASS", "configuration": configuration, "stability": {"overall_pass": True}, "snapshots": [snapshot]}


def test_e2_spatial_and_temporal_reports_are_deterministic():
    cases = [_e2_case(20e-6, E2_COMMON_DT_S, 4.0), _e2_case(10e-6, E2_COMMON_DT_S, 2.0), _e2_case(5e-6, E2_COMMON_DT_S, 1.0)]
    report = spatial_report(cases, horizons_us=(100.0,))
    assert report["status"] == "PASS"
    fine = _e2_case(5e-6, 0.0625e-6, 0.9)
    guard = temporal_guard(cases[-1], fine, report)
    assert guard["status"] == "PASS"


def test_metrics_use_negative_weight_and_separate_second_moments():
    delta_n = np.zeros((5, 5), dtype=np.float64)
    delta_n[2, 2] = -2.0
    delta_n[1, 1] = -1.0
    vx = np.zeros_like(delta_n)
    vy = np.zeros_like(delta_n)
    vy[2, 2] = 3.0
    result = thermal_channel_metrics(
        delta_n, vx, vy, dx=1.0, dy=1.0, x_min=0.0, y_min=0.0, x_max=4.0, y_max=4.0
    )
    assert result["thermal_channel_defined"] is True
    assert result["xc_m"] == pytest.approx((2 * 2 + 1 * 1) / 3)
    assert result["yc_m"] == pytest.approx((2 * 2 + 1 * 1) / 3)
    assert result["sigma_x_m"] == pytest.approx(result["sigma_y_m"])
    assert result["min_delta_n"] == -2.0
    assert result["max_abs_vy_m_s"] == 3.0
    assert result["formal_edge_boundary_ratio"] == 0.0
    assert result["first_interior_ring_ratio"] == pytest.approx(0.5)


def test_snapshot_schedule_is_exact_and_rejects_nonintegral_time():
    schedule = build_snapshot_step_schedule(0.5e-6)
    assert schedule[0] == (0.0, 0)
    assert schedule[3] == (100.0e-6, 200)
    assert schedule[-1] == (1000.0e-6, 2000)
    with pytest.raises(ValueError, match="integer number"):
        build_snapshot_step_schedule(3.0e-6, (0.0, 25.0e-6))


def test_config_guard_allows_only_dt_and_optional_screen_identity():
    cases = [_case(1.0), _case(0.5), _case(0.25)]
    assert config_diff_guard(cases)["pass"] is True
    altered = _case(0.5)
    altered["configuration"]["operator"]["nu_m2_s"] = 9.0
    assert config_diff_guard([cases[0], altered])["pass"] is False
    screen_cases = []
    for dt_us, screen_id in ((1.0, "peak"), (0.5, "front"), (0.25, "rear")):
        item = _case(dt_us)
        item["configuration"]["screen_identity"] = {"screen_id": screen_id}
        screen_cases.append(item)
    assert config_diff_guard(screen_cases, allow_screen_identity=True)["pass"] is False
    assert config_diff_guard(screen_cases, allow_screen_identity=False)["pass"] is False


def test_convergence_rows_classify_case_a_and_report_d1_d2():
    report = convergence_rows([_case(1.0), _case(0.5), _case(0.25)])
    assert report["classification"] == "A"
    assert report["status"] == "PASS"
    rows = [row for row in report["rows"] if row["horizon_us"] == 100.0]
    yc = next(row for row in rows if row["observable"] == "yc_m")
    assert yc["D1_1p0_vs_0p5"] == 0.0
    assert yc["D2_0p5_vs_0p25"] == 0.0
    assert yc["trend_D2_lt_D1"] is True
    assert report["candidate_checks"]["1p0"]["pass"] is True


def test_convergence_rows_classify_case_b_when_one_us_breaks_centroid_tolerance():
    report = convergence_rows([_case(1.0, y_shift=2.0e-6), _case(0.5), _case(0.25)])
    assert report["classification"] == "B"
    assert report["status"] == "PASS"
    assert report["candidate_checks"]["1p0"]["pass"] is False
    assert report["candidate_checks"]["0p5"]["pass"] is True


def test_selection_uses_contiguous_twenty_percent_support_and_absent_sides():
    selected = select_hr3b_screens([-0.1, -0.5, -1.0, -0.4], [0.0, 1.0, 2.0, 3.0])
    assert selected["peak"]["index"] == 2
    assert selected["front"]["index"] == 1
    assert selected["rear"]["index"] == 3
    isolated = select_hr3b_screens([-1.0, -0.1, -0.05], [0.0, 1.0, 2.0])
    assert isolated["front"] is None
    assert isolated["rear"] is None


def test_e1b_loader_requires_exact_grid_and_zeroes_velocities(tmp_path):
    source_path = tmp_path / "screen.npy"
    source = np.zeros((351, 301), dtype=np.float64)
    source[100, 150] = -1.0e-5
    np.save(source_path, source, allow_pickle=False)
    before = sha256_file(source_path)
    manifest = {
        "source_grid": e1b_source_grid(),
        "target_grid": e1a_geometry(),
        "geometry_translation": e1b_geometry_translation(),
        "n0": 1.00027,
        "source_dtype": "float64",
        "source_backend": "numpy",
        "source_git_sha": "test",
        "hr3b_state_file_sha256": "state-file",
        "hr3b_state_sha256": "state-array",
        "screens": {
            "peak": {
                "array_path": source_path.name,
                "file_sha256": sha256_file(source_path),
                "array_sha256": sha256_array(source),
                "shape": [351, 301],
                "dtype": "float64",
                "index": 0,
                "z_m": 0.0,
            }
        },
    }
    (tmp_path / "post_reference_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_e1b_screen(source_path)
    assert loaded["delta_n"].shape == (351, 301)
    assert np.all(loaded["vx"] == 0.0) and np.all(loaded["vy"] == 0.0)
    assert sha256_file(source_path) == before
    bad_path = tmp_path / "bad.npy"
    np.save(bad_path, np.zeros((3, 3), dtype=np.float64), allow_pickle=False)
    with pytest.raises(ValueError, match="match exactly one immutable"):
        load_e1b_screen(bad_path)


def test_e1b_partial_screen_identity_is_checked_against_immutable_manifest(monkeypatch):
    loaded = {
        "screen_identity": {"screen_id": "peak", "screen_index": 7, "screen_z_m": 0.8},
        "n0": 1.00027,
    }
    captured = {}

    monkeypatch.setattr(hr4e_timestep, "load_e1b_screen", lambda path: loaded)

    def fake_run_timestep_case(**kwargs):
        captured.update(kwargs)
        return {"status": "PASS"}

    monkeypatch.setattr(hr4e_timestep, "run_timestep_case", fake_run_timestep_case)
    result = hr4e_timestep.run_e1b_case("peak.npy", screen_identity={"screen_id": "peak"})
    assert result["status"] == "PASS"
    assert captured["screen_identity"] == loaded["screen_identity"]

    with pytest.raises(ValueError, match="screen identity"):
        hr4e_timestep.run_e1b_case("peak.npy", screen_identity={"screen_id": "front"})


def test_post_reference_generator_writes_selected_copies_and_preserves_source(tmp_path):
    from tools.generate_hr4e1_post_reference import generate_post_reference

    config_path = tmp_path / "source.json"
    config_path.write_text(
        json.dumps(
            {
                "grid": {"Nx": 301, "Ny": 351, "Nt": 8, "Lx": 3.01e-3, "Ly": 3.51e-3, "Twin": 80e-15},
                "propagation": {"use_raman_full_operator": True},
                "heat": {"hr3b_enabled": True, "hr3c_enabled": False},
                "run": {"Npulses": 1},
                "raman": {
                    "enabled": True,
                    "operator_mode": "full_isaacs_eq27",
                    "operator_convention": "isaacs_eq27",
                    "operator_integrator": "heun",
                    "nonlinear_split_order": "strang",
                },
            }
        ),
        encoding="utf-8",
    )
    source_output = tmp_path / "runner.npz"
    source_state_path = source_output.with_suffix(".hr3b_delta_n_th.npy")
    state = np.zeros((4, 351, 301), dtype=np.float64)
    state[0, 100, 150] = -1.0e-5
    state[1, 100, 150] = -0.5e-5
    state[2, 100, 150] = -0.8e-5
    state[3, 100, 150] = -0.1e-5
    expected_source_array_sha256 = sha256_array(state)

    def fake_runner(**kwargs):
        Path(kwargs["out_path"]).write_bytes(b"source-output")
        np.save(source_state_path, state, allow_pickle=False)
        return {
            "diagnostics": {
                "hr3b_authoritative": True,
                "authoritative_hr3a_thermal_source_available": True,
                "z_edges": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
            }
        }

    output_dir = tmp_path / "post_reference"
    before = sha256_file(source_state_path) if source_state_path.exists() else None
    manifest = generate_post_reference(
        config_path,
        output_dir,
        runner_output_path=source_output,
        runner=fake_runner,
    )
    assert manifest["source_untouched"] is True
    assert manifest["screens"]["peak"]["index"] == 0
    assert manifest["screens"]["front"] is None
    assert manifest["screens"]["rear"]["index"] == 2
    assert manifest["hr3b_state_sha256"] == expected_source_array_sha256
    assert manifest["hr3b_state_sha256"] == sha256_array(np.load(source_state_path, allow_pickle=False))
    assert manifest["runner_output_sha256"] == sha256_file(source_output)
    assert (output_dir / "screen_peak_delta_n.npy").is_file()
    assert (output_dir / "screen_rear_delta_n.npy").is_file()
    assert not (output_dir / "screen_front_delta_n.npy").exists()
