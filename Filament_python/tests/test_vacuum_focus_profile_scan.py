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


def test_parabolic_focus_interpolation_recovers_subsample_peak() -> None:
    tool = _load_tool("run_vacuum_focus_profile_case.py")
    z = np.array([0.9495, 0.95, 0.9505])
    values = 1.0 - 2.0e6 * (z - 0.95017) ** 2
    peak = tool._parabolic_vertex(z, values)
    assert abs(peak["z_parabolic_m"] - 0.95017) < 1e-10
