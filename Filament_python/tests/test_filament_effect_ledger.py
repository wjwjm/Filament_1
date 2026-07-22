from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_filament_effect_ledger import (  # noqa: E402
    build_effects,
    comparison_metrics,
    curve_metrics,
    first_crossing,
    focus_coordinate_cm,
)


def test_fixed_focus_coordinate_conversion() -> None:
    assert focus_coordinate_cm(0.95) == 0.0
    assert np.allclose(focus_coordinate_cm(np.array([0.90, 0.95, 1.00])), [-5.0, 0.0, 5.0])


def test_first_crossing_uses_linear_interpolation() -> None:
    value, status = first_crossing(np.array([0.0, 2.0]), np.array([0.0, 4.0]), 1.0)
    assert value == 0.5
    assert status == "interpolated"


def test_missing_crossing_is_null() -> None:
    value, status = first_crossing(np.array([0.0, 1.0]), np.array([0.0, 0.5]), 1.0)
    assert value is None
    assert status == "not_crossed"


def test_fwhm_is_null_when_falling_side_is_missing() -> None:
    metrics = curve_metrics(np.array([0.0, 1.0, 2.0]), np.array([0.0, 2.0, 2.0]))
    assert metrics["left_halfmax_crossing_cm"] == 0.5
    assert metrics["right_halfmax_crossing_cm"] is None
    assert metrics["fwhm_cm"] is None


def test_peak_plateau_records_first_peak_and_width() -> None:
    metrics = curve_metrics(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([0.0, 2.0, 2.0, 2.0, 0.0]))
    assert metrics["peak_position_cm"] == 1.0
    assert metrics["peak_plateau_width_cm"] == 2.0


def test_pycap_comparison_uses_common_interval() -> None:
    metrics = comparison_metrics(np.array([0.0, 1.0, 2.0]), np.array([1.0, 10.0, 100.0]), np.array([1.0, 2.0, 3.0]), np.array([10.0, 100.0, 1000.0]))
    assert metrics["pycap_comparison_status"].startswith("calculated")
    assert metrics["rmse_linear_vs_pycap"] is not None
    assert metrics["rmse_log_vs_pycap"] is not None


def test_confounded_effect_cannot_be_high_confidence() -> None:
    definitions = {
        "current_production_result_id": "current",
        "pycap_result_id": "pycap",
        "effects": [{
            "effect_id": "confounded",
            "baseline_result_id": "baseline",
            "comparison_result_id": "current",
            "changed_factor": "multiple_changes",
            "causal_pair_quality": "multiple_delta_confounded",
            "confidence": "high",
            "limitations": "multiple changes",
        }],
    }
    rows = [
        {"result_id": "pycap", "crossing_1e19_cm": 0.0, "crossing_1e20_cm": 0.0, "crossing_1e21_cm": 0.0, "crossing_1e22_cm": 0.0, "crossing_1e19_status": "interpolated", "crossing_1e20_status": "interpolated", "crossing_1e21_status": "interpolated", "crossing_1e22_status": "interpolated"},
        {"result_id": "baseline", "crossing_1e19_cm": 2.0, "crossing_1e20_cm": 2.0, "crossing_1e21_cm": 2.0, "crossing_1e22_cm": 2.0},
        {"result_id": "current", "crossing_1e19_cm": -2.0, "crossing_1e20_cm": -2.0, "crossing_1e21_cm": -2.0, "crossing_1e22_cm": -2.0, "crossing_1e19_status": "interpolated", "crossing_1e20_status": "interpolated", "crossing_1e21_status": "interpolated", "crossing_1e22_status": "interpolated"},
    ]
    effect = build_effects(rows, definitions)[0]
    assert effect["confidence"] == "medium"
    assert effect["fraction_of_total_pycap_offset_1e21"] == 2.0
