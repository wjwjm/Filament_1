from __future__ import annotations

import numpy as np
import pytest

from KHz_filament.hr4e2c_velocity_diagnostic import (
    classify_velocity_screen, field_norms, nested_node_indices, peak_audit,
    quadratic_subgrid_peak, restrict_nested_field, velocity_energy,
)
from tools.audit_hr4e2c_velocity_fields import sha_audit


def test_exact_nested_node_restriction_is_index_selection():
    coarse = np.array((-1.0, 0.0, 1.0))
    fine = np.linspace(-1.0, 1.0, 5)
    index = nested_node_indices(coarse, fine)
    assert np.array_equal(index, np.array((0, 2, 4)))
    field = np.arange(25.0).reshape(5, 5)
    assert np.array_equal(restrict_nested_field(field, index, index), field[::2, ::2])
    with pytest.raises(ValueError, match="align"):
        nested_node_indices(coarse, fine + 1.0e-4)


def test_area_weighted_norms_and_energy_are_correct():
    coarse = np.zeros((2, 2))
    fine = np.ones((2, 2))
    report = field_norms(coarse, fine, dA=0.25)
    assert report["E_L1"] == pytest.approx(1.0)
    assert report["E_L2"] == pytest.approx(1.0)
    assert report["E_Linf"] == pytest.approx(1.0)
    assert velocity_energy(fine, dA=0.25) == pytest.approx(1.0)


def test_peak_audit_and_quadratic_subgrid_reconstruction():
    x = y = np.array((-1.0, 0.0, 1.0))
    yy, xx = np.meshgrid(y, x, indexing="ij")
    field = 10.0 - (xx - 0.25) ** 2 - (yy + 0.25) ** 2
    peak = peak_audit(field, x, y)
    reconstructed = quadratic_subgrid_peak(field, x, y, peak)
    assert peak["index_yx"] == [1, 1]
    assert reconstructed["status"] == "VALID"
    assert reconstructed["x_m"] == pytest.approx(0.25)
    assert reconstructed["y_m"] == pytest.approx(-0.25)


def test_boundary_subgrid_fit_is_invalid():
    field = np.array(((4.0, 1.0), (1.0, 0.0)))
    peak = peak_audit(field, np.array((0.0, 1.0)), np.array((0.0, 1.0)))
    assert quadratic_subgrid_peak(field, np.array((0.0, 1.0)), np.array((0.0, 1.0)), peak)["status"] == "SUBGRID_PEAK_INVALID"


def test_velocity_classification_covers_v1_v2_v3_v4():
    peak = {"raw_max_abs_vy": 1.0}
    valid = {"status": "VALID", "reconstructed_max_abs_vy": 1.0}
    assert classify_velocity_screen(norms_20_10={"E_L1": 0.0, "E_L2": 0.0, "E_Linf_rel": 0.0}, norms_10_5={"E_L1": 0.0, "E_L2": 0.0, "E_Linf_rel": 0.0}, energy_20=1.0, energy_10=1.0, energy_5=1.0, raw_peaks=[peak] * 3, subgrid_peaks=[valid] * 3).startswith("V4")
    assert classify_velocity_screen(norms_20_10={"E_L1": 0.4, "E_L2": 0.4, "E_Linf_rel": 0.4}, norms_10_5={"E_L1": 0.1, "E_L2": 0.1, "E_Linf_rel": 0.1}, energy_20=3.0, energy_10=2.0, energy_5=1.8, raw_peaks=[{"raw_max_abs_vy": v} for v in (1.0, 2.0, 3.2)], subgrid_peaks=[{"status": "VALID", "reconstructed_max_abs_vy": v} for v in (1.0, 1.4, 1.5)]).startswith("V1")
    assert classify_velocity_screen(norms_20_10={"E_L1": 0.1, "E_L2": 0.1, "E_Linf_rel": 0.1}, norms_10_5={"E_L1": 0.2, "E_L2": 0.2, "E_Linf_rel": 0.2}, energy_20=1.0, energy_10=2.0, energy_5=4.0, raw_peaks=[peak] * 3, subgrid_peaks=[{"status": "SUBGRID_PEAK_INVALID"}] * 3).startswith("V2")
    assert classify_velocity_screen(norms_20_10={"E_L1": 0.4, "E_L2": 0.1, "E_Linf_rel": 0.4}, norms_10_5={"E_L1": 0.1, "E_L2": 0.2, "E_Linf_rel": 0.1}, energy_20=3.0, energy_10=2.0, energy_5=1.8, raw_peaks=[peak] * 3, subgrid_peaks=[{"status": "SUBGRID_PEAK_INVALID"}] * 3).startswith("V3")


def test_sha_audit_is_submission_wrapper_only_and_tool_has_no_submission_path():
    report = sha_audit(__import__("pathlib").Path(__file__).resolve().parents[2])
    assert report["SHA_EQUIVALENT_FOR_E2C"] == "PASS"
    assert {item["classification"] for item in report["changed_files"]} == {"SUBMISSION_WRAPPER_ONLY"}
    tool_source = (__import__("pathlib").Path(__file__).resolve().parents[1] / "tools" / "audit_hr4e2c_velocity_fields.py").read_text(encoding="utf-8")
    assert "sbatch" not in tool_source
    assert "np.save" not in tool_source
