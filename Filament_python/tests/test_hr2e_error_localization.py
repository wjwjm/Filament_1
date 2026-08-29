from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "hr2e_error_localization.py"
SPEC = importlib.util.spec_from_file_location("hr2e_error_localization", TOOL)
assert SPEC and SPEC.loader
localization = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(localization)


def _run(label, edges, ion, raman):
    values = {"ion": np.asarray(ion, dtype=float), "raman": np.asarray(raman, dtype=float)}
    values["total"] = values["ion"] + values["raman"]
    return {"label": label, "z_edges": np.asarray(edges, dtype=float), "channels": values}


def test_region_energy_splits_intervals_conservatively_at_focus_boundaries():
    edges = np.array([0.0, 1.3])
    energy = np.array([13.0])
    assert localization._region_energy(edges, energy, 0.0, 0.75) == 7.5
    assert localization._region_energy(edges, energy, 0.75, 1.05) == 3.0
    assert localization._region_energy(edges, energy, 1.05, 1.3) == 2.5


def test_localization_attributes_candidate_fine_difference_to_focus_region():
    coarse = _run("coarse", [0.0, 0.75, 1.05, 1.3], [1, 1, 1], [1, 1, 1])
    candidate = _run("candidate", [0.0, 0.75, 1.05, 1.3], [1, 3, 1], [1, 3, 1])
    fine = _run("fine", [0.0, 0.75, 1.05, 1.3], [1, 2, 1], [1, 2, 1])
    report, curves = localization.build_report(coarse, candidate, fine)
    channel = report["comparisons"]["candidate_to_fine"]["channels"]["total"]
    assert channel["full_left_minus_right_J"] == 2.0
    assert [row["left_minus_right_J"] for row in channel["segments"]] == [0.0, 2.0, 0.0]
    assert channel["cumulative"]["change_focus_J"] == 2.0
    np.testing.assert_allclose(curves["candidate_to_fine"]["total"]["cumulative"][-1], 2.0)
