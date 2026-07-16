from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str):
    path = ROOT / "tools" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_stage_keeps_geometric_focus_and_explicitly_disables_nonlinearity() -> None:
    stage = json.loads((ROOT / "stages" / "vacuum_focus_profile_scan.json").read_text(encoding="utf-8"))
    assert "z_m - 0.95" in stage["coordinate_definition"]
    assert all(value is False for value in stage["nonlinear_terms"].values())
    assert stage["common"]["dz_output_m"] <= 0.00025


def test_all_window_grids_keep_transverse_spacing_and_cover_p1_to_p6() -> None:
    stage = json.loads((ROOT / "stages" / "vacuum_focus_profile_scan.json").read_text(encoding="utf-8"))
    windows = stage["window_scan"]
    assert {item["id"] for item in windows} == {"8mm_512", "10mm_640", "12mm_768", "14mm_896"}
    dx = [item["Lx_m"] / item["Nx"] for item in windows]
    assert max(dx) - min(dx) < 1e-12
    assert len(stage["profiles"]) == 6


def test_bundle_generates_every_profile_window_case(tmp_path: Path) -> None:
    tool = _load_tool("submit_vacuum_focus_profile_scan.py")
    stage = ROOT / "stages" / "vacuum_focus_profile_scan.json"
    old_argv = __import__("sys").argv
    try:
        __import__("sys").argv = ["submit", "--stage", str(stage), "--bundle-dir", str(tmp_path)]
        assert tool.main() == 0
    finally:
        __import__("sys").argv = old_argv
    manifest = json.loads((tmp_path / "profile_scan_manifest.json").read_text(encoding="utf-8"))
    windows = [case for case in manifest["cases"] if case["kind"] == "window"]
    assert len(windows) == 24
    assert all(case["fresnel_radial_samples"] >= 16385 for case in windows)
    assert {case["profile_id"] for case in windows} == {item["id"] for item in json.loads(stage.read_text(encoding="utf-8"))["profiles"]}


def test_p6_solver_matches_p1_discrete_second_moment() -> None:
    tool = _load_tool("submit_vacuum_focus_profile_scan.py")
    stage = json.loads((ROOT / "stages" / "vacuum_focus_profile_scan.json").read_text(encoding="utf-8"))
    entries = {item["id"]: item for item in stage["profiles"]}
    grid = {"Nx": 256, "Ny": 256, "Lx_m": 0.008, "Ly_m": 0.008}
    p6, scale, target = tool._solve_p6(entries, stage["common"], grid)
    x = (np.arange(grid["Nx"]) - grid["Nx"] // 2) * grid["Lx_m"] / grid["Nx"]
    got = tool._second_moment(x, x, p6)
    assert 1.0 < scale < 1.5
    assert abs(got - target) < 2e-9


def test_parabolic_focus_and_axial_shape_metrics() -> None:
    tool = _load_tool("run_vacuum_focus_profile_case.py")
    z = np.array([0.9495, 0.95, 0.9505])
    values = 1.0 - 2.0e6 * (z - 0.95017) ** 2
    peak = tool._parabolic_vertex(z, values)
    assert abs(peak["z_parabolic_m"] - 0.95017) < 1e-10
    x = np.linspace(-6.0, 6.0, 2401)
    y = np.exp(-0.5 * (x / 1.0) ** 2) + 0.2 * np.exp(-0.5 * ((x + 3.5) / 0.18) ** 2)
    shape = tool._axial_shape_metrics(x, y)
    assert 2.2 < shape["axial_fwhm_cm"] < 2.5
    assert 0.45 <= shape["prefocus_sidelobe_max_ratio"] <= 0.51


def test_continuous_fresnel_returns_finite_focus_for_ft90() -> None:
    tool = _load_tool("run_vacuum_focus_profile_case.py")
    profile = {"kind": "cosine", "flat_radius_m": 0.0017811, "zero_radius_m": 0.001979}
    common = {"wavelength_m": 8e-7, "focal_length_m": 0.95}
    z = np.arange(0.75, 1.151, 0.001)
    result = tool._fresnel_onaxis_crosscheck(profile, common, z, amplitude=1.0, n_r=16385)
    assert np.isfinite(result["x_focus_cm"])
    assert result["radial_samples"] == 16385


def test_aggregator_uses_fresnel_status_not_resolution_as_crosscheck() -> None:
    source = (ROOT / "tools" / "aggregate_vacuum_focus_profile_scan.py").read_text(encoding="utf-8")
    assert "independent_fresnel_crosscheck_ok" in source
    assert "fft_onaxis_fresnel_ok" in source
    assert 'crosscheck_ok = abs(refined' not in source
