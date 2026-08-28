from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

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


def _canonical_mapping(*, raman_source="eq10_heun_positive_rotational_energy", operator=True):
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
        "deposition_raman_deposition_reduction_closure_status": "pass",
        "deposition_raman_operator_energy_closure_status": "pass",
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
    with pytest.raises(ValueError, match="Raman source is not Eq.10/Heun rotational deposition"):
        hr2e.validate_canonical_mapping(not_applied, label="control")


def test_hr2c_r_reconstructs_existing_scalar_target_without_raw_npz_mutation(tmp_path):
    legacy = _canonical_mapping(raman_source="actual_field_fluence_loss")
    legacy.update({
        "deposition_level1_all_available_mechanism_closure_pass": False,
        "deposition_raman_level1_closure_status": "failed",
        "raman_target_loss_step_J": np.array([0.5, 0.75]),
        "raman_actual_loss_step_J": np.array([0.4999, 0.7498]),
        "raman_closure_residual_step": np.array([2e-4, 3e-4]),
        "raman_cumulative_closure_residual": np.array([2e-4, 3e-4]),
    })
    path = tmp_path / "existing_120fs.npz"
    np.savez(path, **legacy)
    reconstructed = hr2e.load_canonical_npz(path, label="existing")
    assert reconstructed["hr2c_r_contract_reconstructed"]
    np.testing.assert_allclose(reconstructed["channels"]["raman"], [0.5, 0.75])


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


def test_classified_hr2e_metadata_and_execution_manifest_are_bound(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    config = repo / "Filament_python" / "configs" / "case.json"
    planning_path = (
        repo
        / "Filament_python"
        / "results"
        / "hr2e_schedule_convergence"
        / "stage1_preflight"
        / "hr2e_stage1_preflight_manifest.json"
    )
    config.parent.mkdir(parents=True)
    planning_path.parent.mkdir(parents=True)
    config.write_text('{"case": "candidate"}\n', encoding="utf-8")
    config_rel = config.relative_to(repo).as_posix()
    planning_rel = planning_path.relative_to(repo).as_posix()
    planning = {
        "schema": "khz_filament.hr2e.stage1_preflight.v2",
        "hash_scope": "classified_by_record",
        "provenance_manifest_required": True,
        "provenance_manifest_schema": "filament.provenance.v2",
        "manifest_provenance_path": planning_rel,
        "tracked_paths": sorted([config_rel, planning_rel]),
        "cases": [{
            "case_id": "hr2e_120fs_candidate",
            "config_path": "configs/case.json",
            "config_provenance_path": config_rel,
            "dtype": "fp32",
            "pulse_width": "120fs",
            "schedule": "candidate",
            "raman_mode": "full_isaacs_eq27",
        }],
    }
    planning_path.write_text(json.dumps(planning) + "\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "init", "-b", "main"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "--", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "fixture"], check=True, capture_output=True)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    provenance_path = tmp_path / "provenance.json"
    provenance = hr2e.provenance_v2.create_manifest(
        repo,
        provenance_path,
        planning["tracked_paths"],
        [],
    )
    config_record = hr2e.provenance_v2.lookup_record(
        provenance,
        config_rel,
        classification="tracked_text",
        require_hash_scope=True,
    )
    preflight_record = hr2e.provenance_v2.lookup_record(
        provenance,
        planning_rel,
        classification="tracked_text",
        require_hash_scope=True,
    )

    npz_path = tmp_path / "hr2e_120fs_candidate.npz"
    np.savez(npz_path, **_canonical_mapping())
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "schema": "khz_filament.hr2e.job_metadata.v2",
        "hash_scope": "classified_by_record",
        "case_id": "hr2e_120fs_candidate",
        "git_sha": head,
        "config_path": str(config),
        "config_provenance_path": config_record["path"],
        "config_classification": config_record["classification"],
        "config_hash_scope": config_record["hash_scope"],
        "config_git_blob_oid": config_record["git_blob_oid"],
        "config_canonical_lf_sha256": config_record["canonical_lf_sha256"],
        "dtype": "fp32",
    }) + "\n", encoding="utf-8")

    monkeypatch.setattr(hr2e, "REPO_ROOT", repo)
    run = hr2e.load_case_with_provenance(
        npz_path,
        metadata_path,
        planning_path,
        provenance_path,
        case_id="hr2e_120fs_candidate",
    )
    assert run["config_binding"] == config_record

    provenance_raw = hr2e.provenance_v2.raw_sha256_file(provenance_path)
    execution_path = tmp_path / "execution.json"
    execution = {
        "schema": "khz_filament.hr2e.execution_manifest.v2",
        "hash_scope": "classified_by_record",
        "expected_git_sha": head,
        "preflight_manifest_path": str(planning_path),
        "preflight_manifest_record": preflight_record,
        "provenance_manifest_path": str(provenance_path),
        "provenance_manifest_sha256": provenance_raw,
        "provenance_manifest_hash_scope": "raw_bytes",
        "records": [{
            "path": str(provenance_path),
            "classification": "external",
            "hash_scope": "raw_bytes",
            "raw_sha256": provenance_raw,
        }, *provenance["records"]],
        "config_records": [{"case_id": run["case_id"], **config_record}],
        "case_ids": [run["case_id"]],
    }
    execution_path.write_text(json.dumps(execution) + "\n", encoding="utf-8")
    hr2e.validate_execution_manifest(
        {"candidate": run},
        planning_path,
        execution_path,
        provenance_path,
    )

    execution["config_records"][0]["canonical_lf_sha256"] = "0" * 64
    execution_path.write_text(json.dumps(execution) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config record mismatch"):
        hr2e.validate_execution_manifest(
            {"candidate": run}, planning_path, execution_path, provenance_path
        )
