from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import analyze_density_translation_width as analysis  # noqa: E402


def test_crossings_and_translation_scale_fit_recover_synthetic_shift() -> None:
    x = np.linspace(-20.0, 20.0, 801)
    paper = analysis.Curve("paper", x, np.exp(-0.5 * ((x + 8.0) / 2.0) ** 2))
    sim = analysis.Curve("sim", x, 1.2 * np.exp(-0.5 * ((x + 11.0) / 2.0) ** 2))
    assert analysis.crossing(paper, 0.5, "rising") is not None
    fit = analysis._fit_models(paper, sim)
    assert fit["pure_translation"]["parameters"]["delta_x_cm"] == pytest.approx(-3.0, abs=0.05)
