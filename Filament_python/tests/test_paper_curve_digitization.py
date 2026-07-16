from __future__ import annotations

import sys
import json
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


def test_archived_fig5b_digitization_regression_is_locked() -> None:
    result = ROOT / "results" / "density_translation_width" / "density_translation_width_20260715_002"
    metadata = json.loads((result / "paper_digitization_metadata.json").read_text(encoding="utf-8"))
    assert "Fig. 5(b)" in metadata["source"]
    for name in ("paper_pycap_120fs.csv", "paper_pycap_40fs.csv", "paper_digitization_pixel_overlay.png"):
        assert (result / name).is_file()
    fit = json.loads((result / "translation_width_fit.json").read_text(encoding="utf-8"))["cases"]
    paper120 = fit["120 fs"]["paper_features"]
    paper40 = fit["40 fs"]["paper_features"]
    assert 6.30 <= paper120["peak_rho_1e16_cm3"] <= 6.60
    assert -12.19 <= paper120["peak_interval_center_cm"] <= -11.89
    assert -8.25 <= paper40["peak_interval_center_cm"] <= -7.95
    assert -14.85 <= paper120["absolute_0.2_rising_cm"] <= -14.48
    assert -12.50 <= paper40["absolute_0.2_rising_cm"] <= -12.10
