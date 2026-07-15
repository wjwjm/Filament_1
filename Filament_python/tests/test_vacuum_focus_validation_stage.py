from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _get(value: dict, dotted: str):
    for part in dotted.split("."):
        value = value[part]
    return value


def test_vacuum_focus_stage_keeps_geometry_and_disables_all_nonlinearity() -> None:
    stage = json.loads((ROOT / "stages" / "vacuum_focus_validation_ft90.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "configs" / "vacuum_focus_validation" / "flat_top_90_vacuum.json").read_text(encoding="utf-8"))
    for dotted, expected in stage["required_invariants"].items():
        assert _get(config, dotted) == expected
    assert all(value is False for value in stage["nonlinear_terms"].values())
    assert stage["coordinate_definition"] == "x_focus_cm = 100 * (z_m - 0.95); zero is permanently the geometric thin-lens focus."
    assert config["propagation"]["dz_output_m"] <= 2.5e-4
