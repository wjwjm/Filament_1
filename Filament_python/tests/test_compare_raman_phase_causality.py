from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path: sys.path.insert(0, str(ROOT / "tools"))

from compare_raman_phase_causality import classify


def _metrics(onset, center, fwhm=1.0, peak=1e22, tail=1.0):
    return {"rho_peak_m3": peak, "peak_top_center_cm": center, "fwhm_cm": fwhm, "tail_area_above_half_m3_cm": tail, "threshold_crossings_cm": {"1000000000000000000000": onset}}


def test_classify_supports_resolved_noncollapsing_improvement():
    full, off, paper = _metrics(-1.0, 0.0), _metrics(-1.3, -0.3), _metrics(-1.0, 0.0)
    label, decision = classify(full, off, paper, epsilon_x_cm=0.1, numerical_ok=True)
    assert label == "raman_phase_supported" and decision["effect_resolved"]


def test_classify_marks_no_resolved_effect_not_supported():
    full, off, paper = _metrics(-1.0, 0.0), _metrics(-1.03, 0.02), _metrics(-1.0, 0.0)
    assert classify(full, off, paper, epsilon_x_cm=0.1, numerical_ok=True)[0] == "raman_phase_not_supported"


def test_classify_marks_bad_numerical_path_inconclusive():
    full, off, paper = _metrics(-1.0, 0.0), _metrics(-1.3, -0.3), _metrics(-1.0, 0.0)
    assert classify(full, off, paper, epsilon_x_cm=0.1, numerical_ok=False)[0] == "raman_phase_inconclusive"
