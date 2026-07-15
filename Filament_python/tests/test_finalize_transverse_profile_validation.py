from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import finalize_transverse_profile_validation as finalizer


ROOT = Path(__file__).resolve().parents[1]


def _result(profile_type: str, scale: float) -> dict[str, np.ndarray]:
    return {
        "z_axis": np.array([0.8, 0.9, 1.0]),
        "U_z": np.array([1e-3, 0.99e-3, 0.98e-3]),
        "I_max_z": scale * np.array([1e13, 2e13, 1.5e13]),
        "rho_onaxis_max_z": scale * np.array([1e20, 2e21, 1e21]),
        "fwhm_plasma_z": np.array([20e-6, 30e-6, 25e-6]),
        "input_profile_x": np.array([-1e-3, 0.0, 1e-3]),
        "input_profile_center_I": scale * np.array([0.1e13, 1e13, 0.1e13]),
        "input_profile_type": np.asarray(profile_type),
        "input_peak_power_W": np.asarray(17e9),
        "input_peak_intensity_W_m2": np.asarray(scale * 1e13),
        "input_effective_area_m2": np.asarray(1e-6),
        "input_second_moment_radius_m": np.asarray(1e-3),
        "input_r50_m": np.asarray(0.5e-3),
        "input_r90_m": np.asarray(0.9e-3),
        "input_boundary_I_fraction": np.asarray(0.0),
    }


def test_profile_validation_finalizer_writes_controlled_comparison_report(tmp_path: Path) -> None:
    spec = json.loads((ROOT / "stages" / "transverse_profile_validation.json").read_text(encoding="utf-8"))
    (tmp_path / "stage_spec_snapshot.json").write_text(json.dumps(spec), encoding="utf-8")
    comparison = tmp_path / "comparison"
    comparison.mkdir()
    (comparison / "comparison_summary.json").write_text("{}", encoding="utf-8")

    for case, profile_type, scale in (("profile_g_120", "gaussian", 1.0), ("profile_ft90_120", "flat_top_cosine", 0.8)):
        case_dir = tmp_path / "cases" / case
        case_dir.mkdir(parents=True)
        (case_dir / "run_metadata.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        np.savez(case_dir / "result.npz", **_result(profile_type, scale))

    report = finalizer.finalize_stage(tmp_path)
    assert report["technical_status"] == "completed"
    assert report["quality_gate_status"] == "passed"
    assert report["scientific_interpretation_status"] == "controlled_comparison_only"
    assert (tmp_path / "reports" / "transverse_profile_validation.md").is_file()
    assert (tmp_path / "reports" / "input_profiles.png").is_file()
    assert "- comparison/rho_max_z.png" in (tmp_path / "reports" / "transverse_profile_validation.md").read_text(encoding="utf-8")
