from __future__ import annotations

import json
from pathlib import Path

from finalize_stage1 import build_stage1_report
from finalize_stage1 import evaluate_case_quality
from plot_khzfil_out import FIGURE_SPECS


def _summary() -> dict[str, object]:
    return {"generated_figures": list(FIGURE_SPECS.values()), "sanity_warnings": [], "quality_observations": {"z_strictly_increasing": True, "max_energy_growth_fraction": 0.01, "max_adjacent_intensity_growth": 2.0, "max_electron_density_m3": 1e20, "fwhm_all_positive_finite": True}}


def test_finalize_complete_stage_requires_review(tmp_path: Path) -> None:
    spec = {"stage_id": "stage1", "stage_name": "single_pulse_filament_optimization", "display_name": "Stage 1", "objective": "compare", "comparison_mode": "same_peak_power_different_pulse_duration", "required_invariants": {"beam.P0_peak": 17e9}, "quality_gates": {"require_strictly_increasing_z": True, "maximum_energy_growth_fraction": 0.1, "maximum_adjacent_intensity_growth": 10.0, "maximum_electron_density_m3": 1e25, "require_positive_fwhm": True}, "cases": [{"case_id": "40fs"}, {"case_id": "120fs"}]}
    (tmp_path / "stage_spec_snapshot.json").write_text(json.dumps(spec), encoding="utf-8"); (tmp_path / "submission_manifest.json").write_text("{}", encoding="utf-8")
    for case in ("40fs", "120fs"):
        directory = tmp_path / "cases" / case / "figures"; directory.mkdir(parents=True); (directory / "diagnostic_summary.json").write_text(json.dumps(_summary()), encoding="utf-8"); (directory.parent / "run_metadata.json").write_text('{"status": "completed"}', encoding="utf-8")
    (tmp_path / "comparison").mkdir(); (tmp_path / "comparison" / "comparison_summary.json").write_text('{"generated_figures": ["comparison_overview.png"]}', encoding="utf-8"); (tmp_path / "comparison" / "comparison_metrics.csv").write_text("case_id,case_label\n40fs,40 fs\n120fs,120 fs\n", encoding="utf-8")
    report = build_stage1_report(tmp_path)
    assert report["technical_status"] == "completed"
    assert report["quality_gate_status"] == "passed"
    assert report["scientific_interpretation_status"] == "requires_review"
    assert "40fs is optimal" not in (tmp_path / "reports" / "stage1_report.md").read_text(encoding="utf-8").lower()


def test_finalize_missing_case_is_not_completed(tmp_path: Path) -> None:
    (tmp_path / "stage_spec_snapshot.json").write_text(json.dumps({"stage_id": "stage1", "stage_name": "single", "display_name": "S", "objective": "x", "comparison_mode": "x", "required_invariants": {"beam.P0_peak": 17e9}, "quality_gates": {"require_strictly_increasing_z": True, "maximum_energy_growth_fraction": .1, "maximum_adjacent_intensity_growth": 10, "maximum_electron_density_m3": 1e25, "require_positive_fwhm": True}, "cases": [{"case_id": "40fs"}, {"case_id": "120fs"}]}), encoding="utf-8")
    (tmp_path / "submission_manifest.json").write_text("{}", encoding="utf-8")
    assert build_stage1_report(tmp_path)["technical_status"] == "incomplete"


def test_quality_failure_is_distinct_from_missing_artifacts() -> None:
    summary = _summary()
    summary["quality_observations"]["max_electron_density_m3"] = 2e25
    result = evaluate_case_quality(summary, {"require_strictly_increasing_z": True, "maximum_energy_growth_fraction": .1, "maximum_adjacent_intensity_growth": 10, "maximum_electron_density_m3": 1e25, "require_positive_fwhm": True})
    assert result["technical_pass"] is False
    assert any("electron" in failure for failure in result["failures"])
