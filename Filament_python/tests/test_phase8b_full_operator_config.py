from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from KHz_filament.config import PropagationConfig, RamanConfig, resolve_nonlinear_switches
from KHz_filament.config_normalize import normalize_config


def _strict(enabled=True):
    return {
        "propagation": {
            "use_raman_phase": False,
            "use_raman_full_operator": enabled,
            "use_raman_absorption": False,
        },
        "raman": {
            "enabled": True,
            "model": "isaacs_rot_sinexp",
            "n_R": 2.3e-23,
            "omega_R": 1.6e13,
            "Gamma_R": 1.3e13,
            "operator_mode": "full_isaacs_eq27",
            "operator_convention": "isaacs_eq27",
            "iir_sampling": "exact_piecewise_linear",
            "operator_integrator": "heun",
            "absorption": False,
        },
    }


def test_full_operator_switch_is_explicit_and_raw_convolution_stays_enabled():
    normalized = normalize_config(_strict(False))
    prop = PropagationConfig(**normalized["propagation"])
    raman = RamanConfig(**normalized["raman"])
    switches = resolve_nonlinear_switches(prop, raman, None)
    assert switches.use_raman_phase is False
    assert switches.use_raman_full_operator is False
    assert switches.compute_raman_convolution is True


@pytest.mark.parametrize(
    "section,field,value,match",
    [
        ("propagation", "use_raman_phase", True, "rejects propagation.use_raman_phase"),
        ("propagation", "use_raman_absorption", True, "rejects legacy Raman absorption"),
        ("raman", "absorption", True, "rejects legacy Raman absorption"),
    ],
)
def test_full_operator_rejects_legacy_split_or_absorption(section, field, value, match):
    config = _strict()
    config[section][field] = value
    with pytest.raises(ValueError, match=match):
        normalize_config(config)


def test_legacy_split_switch_resolution_is_unchanged():
    switches = resolve_nonlinear_switches(
        PropagationConfig(use_raman_phase=None, use_raman_full_operator=None),
        RamanConfig(enabled=True, operator_mode="legacy_split", absorption=True),
        None,
    )
    assert switches.use_raman_phase is True
    assert switches.use_raman_full_operator is False
    assert switches.use_raman_absorption is True


def test_phase8b_configs_are_baseline_locked_and_single_factor(tmp_path):
    config_dir = tmp_path / "configs"
    out_dir = tmp_path / "audit"
    subprocess.run([
        sys.executable,
        str(ROOT / "tools" / "prepare_phase8b_preflight_configs.py"),
        "--config-dir", str(config_dir),
        "--out-dir", str(out_dir),
    ], check=True)
    baseline_diff = json.loads((out_dir / "baseline_to_full_operator_config_diff.json").read_text())
    on_off_diff = json.loads((out_dir / "full_operator_on_vs_off_config_diff.json").read_text())
    assert baseline_diff["status"] == "passed" and not baseline_diff["unexpected_paths"]
    assert on_off_diff["status"] == "passed" and not on_off_diff["unexpected_paths"]
    assert [row["path"] for row in on_off_diff["differences"]] == ["propagation.use_raman_full_operator"]
    on = normalize_config(json.loads((config_dir / "120fs_talebpour_isaacs_full_operator_on.json").read_text()))
    off = normalize_config(json.loads((config_dir / "120fs_talebpour_isaacs_full_operator_feedback_off.json").read_text()))
    assert on["propagation"]["use_raman_full_operator"] is True
    assert off["propagation"]["use_raman_full_operator"] is False
    assert on["raman"]["absorption"] is False
    assert off["raman"]["absorption"] is False
