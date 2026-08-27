from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from KHz_filament.longitudinal import build_longitudinal_schedule


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "hr2e_schedule_convergence.py"
SPEC = importlib.util.spec_from_file_location("hr2e_schedule_convergence", TOOL_PATH)
assert SPEC and SPEC.loader
hr2e = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hr2e)


def _schedule(base: float, focus: float):
    return build_longitudinal_schedule(
        base,
        1.3,
        focus_window_step=True,
        focus_center_m=0.90,
        focus_halfwidth_m=0.15,
        dz_focus=focus,
    )


def _canonical_mapping(*, raman_source="actual_field_fluence_loss", operator=True):
    edges = np.array([0.0, 0.4, 1.0], dtype=float)
    ion = np.array([1.0, 2.0])
    ib = np.zeros(2)
    raman = np.array([0.5, 0.75])
    total = ion + ib + raman
    return {
        "z_edges": edges,
        "dz_intervals": np.diff(edges),
        "n_intervals": 2,
        "E_field_in_J": 10.0,
        "E_dep_ion_interval_J": ion,
        "E_dep_ib_interval_J": ib,
        "E_dep_raman_interval_J": raman,
        "E_dep_total_interval_J": total,
        "E_dep_ion_pulse_J": ion.sum(),
        "E_dep_ib_pulse_J": ib.sum(),
        "E_dep_raman_pulse_J": raman.sum(),
        "E_dep_total_pulse_J": total.sum(),
        "deposition_level1_all_available_mechanism_closure_pass": True,
        "deposition_level2_all_available_mechanism_closure_pass": True,
        "E_dep_total_level2_closure_status": "pass",
        "total_deposition_authoritative": True,
        "deposition_raman_authoritative": True,
        "raman_deposition_source": raman_source,
        "raman_operator_applied": np.array([operator, operator]),
        "deposition_raman_level1_closure_status": "pass",
        "deposition_raman_level2_closure_status": "pass",
        "field_energy_bookkeeping_authoritative": True,
        "field_energy_bookkeeping_status": "available",
        "E_field_out_J": 5.75,
        "E_field_loss_J": 4.25,
        "E_dep_accounted_authoritative_J": 4.25,
        "E_field_energy_bookkeeping_residual_J": 0.0,
        "E_field_energy_bookkeeping_relative_residual": 0.0,
    }


def test_nested_schedule_family_is_valid_and_roundoff_tail_is_only_metadata():
    coarse = _schedule(2e-4, 1e-4)
    candidate = _schedule(1e-4, 5e-5)
    fine = _schedule(5e-5, 2.5e-5)
    for schedule in (coarse, candidate, fine):
        assert schedule.z_edges[0] == 0.0
        assert schedule.z_edges[-1] == pytest.approx(1.3)
        assert np.all(np.diff(schedule.z_edges) > 0.0)
    assert coarse.n_intervals < candidate.n_intervals < fine.n_intervals
    coarse_summary = hr2e.schedule_summary(
        coarse.z_edges, coarse.dz_intervals, base_dz=2e-4, focus_dz=1e-4
    )
    assert coarse_summary["roundoff_tail_present"]
    assert coarse_summary["roundoff_tail_count"] == 1
    assert candidate.n_intervals == 16000
    assert fine.n_intervals == 32000


def test_conservative_remap_conserves_integrated_energy():
    source_edges = np.array([0.0, 0.25, 0.75, 1.0])
    source_energy = np.array([1.0, 4.0, 2.0])
    target_edges = np.array([0.0, 0.1, 0.6, 0.9, 1.0])
    remapped = hr2e.conservative_remap(source_edges, source_energy, target_edges)
    assert remapped.sum() == pytest.approx(source_energy.sum(), rel=1e-12, abs=1e-15)
    assert np.all(remapped >= 0.0)


def test_identical_piecewise_constant_profile_has_zero_cumulative_error():
    candidate_edges = np.array([0.0, 0.5, 1.0])
    fine_edges = np.array([0.0, 0.2, 0.5, 0.7, 1.0])
    candidate_energy = 3.0 * np.diff(candidate_edges)
    fine_energy = 3.0 * np.diff(fine_edges)
    common = hr2e.union_edges(candidate_edges, fine_edges)
    candidate_common = hr2e.conservative_remap(candidate_edges, candidate_energy, common)
    fine_common = hr2e.conservative_remap(fine_edges, fine_energy, common)
    error = np.max(
        np.abs(hr2e.cumulative_curve(candidate_common) - hr2e.cumulative_curve(fine_common))
    )
    assert error == pytest.approx(0.0, abs=1e-14)


def test_zero_and_negligible_channels_do_not_divide_by_zero_or_false_fail():
    candidate = hr2e.validate_canonical_mapping(_canonical_mapping(), label="candidate")
    fine = hr2e.validate_canonical_mapping(_canonical_mapping(), label="fine")
    ib = hr2e.compare_channel(candidate, fine, "ib")
    assert ib["classification"] == "zero_channel"
    assert ib["pulse_energy_error"] == 0.0
    assert ib["primary_pass"]

    candidate["channels"]["ib"][:] = [2e-8, 0.0]
    candidate["pulse"]["ib"] = 2e-8
    fine["channels"]["ib"][:] = [1e-8, 0.0]
    fine["pulse"]["ib"] = 1e-8
    tiny = hr2e.compare_channel(candidate, fine, "ib")
    assert tiny["classification"] == "negligible_channel"
    assert tiny["pulse_energy_error_kind"] == "absolute_over_field_in"
    assert np.isfinite(tiny["pulse_energy_error"])


def test_legacy_total_and_operator_not_applied_are_rejected():
    legacy = _canonical_mapping(raman_source="legacy_unavailable")
    legacy["total_deposition_authoritative"] = False
    with pytest.raises(ValueError, match="total deposition is not authoritative"):
        hr2e.validate_canonical_mapping(legacy, label="legacy")

    not_applied = _canonical_mapping(raman_source="operator_not_applied", operator=False)
    with pytest.raises(ValueError, match="Raman source is not actual_field_fluence_loss"):
        hr2e.validate_canonical_mapping(not_applied, label="control")


def test_authoritative_gate_requires_level1_and_level2_closure():
    failed = _canonical_mapping()
    failed["deposition_level2_all_available_mechanism_closure_pass"] = False
    with pytest.raises(ValueError, match="Level-2 deposition closure failed"):
        hr2e.validate_canonical_mapping(failed, label="failed")

    valid = hr2e.validate_canonical_mapping(_canonical_mapping(), label="valid")
    assert valid["n_intervals"] == 2
    np.testing.assert_allclose(
        valid["channels"]["total"],
        valid["channels"]["ion"] + valid["channels"]["ib"] + valid["channels"]["raman"],
    )


def test_authoritative_gate_rejects_short_operator_trace_and_level3_failure():
    short = _canonical_mapping()
    short["raman_operator_applied"] = np.array([True])
    with pytest.raises(ValueError, match="operator was not applied throughout"):
        hr2e.validate_canonical_mapping(short, label="short")

    unavailable = _canonical_mapping()
    unavailable["field_energy_bookkeeping_authoritative"] = False
    with pytest.raises(ValueError, match="field-energy bookkeeping is not authoritative"):
        hr2e.validate_canonical_mapping(unavailable, label="unavailable")


def test_comparison_provenance_rejects_mismatched_git_or_pulse():
    base = {
        "schedule": "candidate",
        "git_sha": "a" * 40,
        "dtype": "fp32",
        "pulse_width": "120fs",
        "raman_mode": "full_isaacs_eq27",
    }
    fine = dict(base, schedule="fine")
    hr2e.validate_comparison_provenance({"candidate": base, "fine": fine})
    with pytest.raises(ValueError, match="git_sha"):
        hr2e.validate_comparison_provenance(
            {"candidate": base, "fine": dict(fine, git_sha="b" * 40)}
        )


def test_compare_triplet_returns_both_comparisons_and_trend():
    candidate = hr2e.validate_canonical_mapping(_canonical_mapping(), label="candidate")
    candidate["label"] = "candidate"
    fine = hr2e.validate_canonical_mapping(_canonical_mapping(), label="fine")
    fine["label"] = "fine"
    coarse = hr2e.validate_canonical_mapping(_canonical_mapping(), label="coarse")
    coarse["label"] = "coarse"
    result = hr2e.compare_triplet(coarse, candidate, fine)
    assert result["coarse_vs_candidate"] is not None
    assert result["candidate_vs_fine"] is not None
    assert result["primary_pass"]
    assert set(result["convergence_trend"]) == {"ion", "ib", "raman", "total"}
