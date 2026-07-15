from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import analyze_density_translation_width as analysis  # noqa: E402


def test_peak_interval_and_pivoted_scale_fit_recover_synthetic_shift_and_width() -> None:
    x = np.linspace(-20.0, 20.0, 2001)
    paper = analysis.Curve("paper", x, np.exp(-0.5 * ((x + 8.0) / 2.0) ** 2))
    # Exact model form with x_ref=-8, delta=-3, scale=1.25.
    sim = analysis.Curve("sim", x, 1.2 * np.exp(-0.5 * (((x + 8.0 + 3.0) / 1.25) / 2.0) ** 2))
    features = analysis.peak_interval(paper)
    assert features["peak_interval_center_cm"] == pytest.approx(-8.0, abs=0.03)
    fit = analysis._fit_models(paper, sim, features["peak_interval_center_cm"])
    values = fit["translation_plus_scale"]["parameters"]
    assert values["delta_x_cm"] == pytest.approx(-3.0, abs=0.06)
    assert values["scale_s"] == pytest.approx(1.25, abs=0.03)


def test_fixed_geometric_coordinate_is_preserved() -> None:
    x = np.array([-1.0, 0.0, 1.0])
    curve = analysis.Curve("fixed", x, np.array([0.0, 1.0, 0.0]))
    assert analysis.evaluate(curve, np.array([0.0]))[0] == pytest.approx(1.0)
