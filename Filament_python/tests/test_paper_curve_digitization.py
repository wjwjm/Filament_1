from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import digitize_paper_density_curve as digitizer  # noqa: E402


def test_multitick_vertical_calibration_is_within_gate() -> None:
    calibration = digitizer.fit_pixel_calibration(digitizer.Y_TICKS)
    assert calibration["max_abs_residual"] <= 0.03
    assert calibration["slope"] < 0.0


def test_panel_top_is_above_y6_tick_and_trace_retains_above_six() -> None:
    assert digitizer.PANEL["top"] < min(pixel for pixel, _ in digitizer.Y_TICKS)
    image_path = ROOT / "tmp" / "pdfs" / "isaacs2022-11.png"
    if not image_path.exists():
        # The integration render is intentionally not a test prerequisite in a
        # clean clone; the geometry invariant above catches the old defect.
        return
    from PIL import Image
    image = np.asarray(Image.open(image_path).convert("RGB"))
    _, y_pixel, _ = digitizer.trace_curve(image, "black")
    y_cal = digitizer.fit_pixel_calibration(digitizer.Y_TICKS)
    assert float(np.max(y_cal["slope"] * y_pixel + y_cal["intercept"])) > 6.0


def test_legend_exclusion_rectangle_is_right_of_black_trace() -> None:
    assert digitizer.EXCLUSIONS["legend"]["x0"] > digitizer.PANEL["left"] + 300
