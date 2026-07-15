from __future__ import annotations

import json
from pathlib import Path

import pytest

import submit_stage


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "stages" / "stage1_single_pulse_optimization.json"


def test_stage1_spec_enforces_single_pulse_same_peak_power_and_cases() -> None:
    spec = submit_stage.load_stage_spec(SPEC_PATH)
    base = json.loads((SPEC_PATH.parent / spec["base_config"]).read_text(encoding="utf-8"))
    submit_stage.validate_stage_invariants(base, spec)
    configs = [submit_stage.build_case_config(base, case) for case in spec["cases"]]
    submit_stage.validate_case_differences(configs)
    assert [case["case_id"] for case in spec["cases"]] == ["40fs", "120fs"]
    assert base["run"]["Npulses"] == 1
    assert base["beam"]["P0_peak"] == 17e9
    assert base["beam"]["energy_J"] is None
    assert configs[0]["beam"]["tau_fwhm"] == pytest.approx(40e-15)
    assert configs[1]["beam"]["tau_fwhm"] == pytest.approx(120e-15)


def test_unapproved_case_difference_is_rejected() -> None:
    spec = submit_stage.load_stage_spec(SPEC_PATH)
    base = json.loads((SPEC_PATH.parent / spec["base_config"]).read_text(encoding="utf-8"))
    configs = [submit_stage.build_case_config(base, case) for case in spec["cases"]]
    configs[1]["beam"]["w0"] *= 2
    with pytest.raises(ValueError, match="outside allowed"):
        submit_stage.validate_case_differences(configs)
