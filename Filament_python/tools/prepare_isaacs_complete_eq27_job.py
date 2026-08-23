#!/usr/bin/env python3
"""Prepare the single 120 fs complete Isaacs Eq. (27) causal job.

The candidate is deliberately derived by deep-copying the locked Raman-ON
full-operator configuration.  The only configuration difference is the
opt-in electronic operator mode ``full_isaacs_eq27_complete``.  This tool
does not submit a job or run a propagation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "isaacs_raman_closure" / "120fs_talebpour_isaacs_full_operator_on.json"
OUT = ROOT / "results" / "isaacs_complete_eq27"
CANDIDATE_NAME = "120fs_talebpour_isaacs_complete_eq27.json"
PARENT_C1_COMMIT = "459dd108b9873b0e8b18fe83111f386993cf5b9f"
COMPLETE_MODE = "full_isaacs_eq27_complete"
SOURCE_MODE = "full_isaacs_eq27"
LOCKED_BASE_SHA256 = "942adca964f50b689fa5985c9af46f294da7948646b246c39ca0d50238a1b02a"
REMOTE_CAMPAIGN_ROOT = "/data/run01/scvi806/user_Wangjimin/isaacs_complete_eq27_c2"
CAMPAIGN_ID = "isaacs_complete_eq27_c2"
EXPECTED_GPU_MODEL = "NVIDIA GeForce RTX 5090"
PYCAP_REL = "results/density_translation_width/density_translation_width_20260715_002/paper_pycap_120fs.csv"
PYCAP_SHA256 = "9b43e75ebc08ccb0a7796829e45c6727b42ab12cd661b9a3d8d235ef89d31461"
C1_COMMIT = "459dd108b9873b0e8b18fe83111f386993cf5b9f"
C1_SUMMARY_REL = "results/isaacs_complete_eq27/c1_closure_summary.json"
C1_SUMMARY_SHA256 = "ccf6f865042651894e747f1272c5371cad8bc4bb7fd6abd11b61684a795ebcdc"
C1_REPORT_REL = "results/isaacs_complete_eq27/c1_operator_report.md"
C1_REPORT_SHA256 = "fe8b7fe99a88dde5d4c987d88d1a87dd5208461bb70ff25af6e365ef4ac7b21d"
FALLBACK_EVIDENCE = {
    "current_full_eq27": {
        "job_id": "180748",
        "case": "on",
        "npz_path": "/data/run01/scvi806/user_Wangjimin/phase8c_b_runs/test_a_f0a7b5d5ac10_fallback/on/test_a_on.npz",
        "npz_sha256": "68d846d4815cd8387c7a4c4934b26dfe48bcef77cc9140d2f06d2fa8e929a218",
        "metadata_path": "/data/run01/scvi806/user_Wangjimin/phase8c_b_runs/test_a_f0a7b5d5ac10_fallback/on/test_a_on_job_metadata.json",
        "metadata_sha256": "0b057fed4763bb2719d7b8288e820d30cf4f458b3752632d65a026cf1eee9f21",
        "config_sha256": "aafec917d06c252617e5bfdd2ce3a73dd276401c271c33380d59e0172055cf78",
        "execution_sha": "f0a7b5d5ac103546bd693378e8f8efb4f07c6c27",
        "gpu_model": EXPECTED_GPU_MODEL,
    },
    "raman_off": {
        "job_id": "180749",
        "case": "off",
        "npz_path": "/data/run01/scvi806/user_Wangjimin/phase8c_b_runs/test_a_f0a7b5d5ac10_fallback/off/test_a_off.npz",
        "npz_sha256": "e85b8dbbc0fd20b50f6c8234d3de677119ff46f4acaf459e43b1b8ff5e5dc6f9",
        "metadata_path": "/data/run01/scvi806/user_Wangjimin/phase8c_b_runs/test_a_f0a7b5d5ac10_fallback/off/test_a_off_job_metadata.json",
        "metadata_sha256": "d2bd43c85099a03c2b3f226127829c07b99fc955c486989d443d09c08d21716a",
        "config_sha256": "1c1415941d4497a6caaf6a37ee8559bbd8b8b20a9eeee6377a8dbbc7d28f41ef",
        "execution_sha": "f0a7b5d5ac103546bd693378e8f8efb4f07c6c27",
        "gpu_model": EXPECTED_GPU_MODEL,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            result.update(_flatten(child, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: value}


def config_diff(base: dict[str, Any], derived: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a deterministic, flattened diff for provenance and tests."""
    before, after = _flatten(base), _flatten(derived)
    return [
        {
            "path": key,
            "full_isaacs_eq27": before.get(key),
            "full_isaacs_eq27_complete": after.get(key),
        }
        for key in sorted(set(before) | set(after))
        if before.get(key) != after.get(key)
    ]


def _assert_fixed(config: dict[str, Any]) -> None:
    """Assert the C2 fixed-condition contract before and after the copy."""
    grid = config["grid"]
    beam = config["beam"]
    prop = config["propagation"]
    ion = config["ionization"]
    raman = config["raman"]

    assert (grid["Nx"], grid["Ny"], grid["Nt"], grid["Lx"], grid["Ly"], grid["Twin"]) == (
        512, 512, 384, 0.008, 0.008, 9.6e-13
    )
    assert (
        beam["lam0"], beam["n0"], beam["w0"], beam["tau_fwhm"],
        beam["P0_peak"], beam["focal_length"], beam["n2_air"],
    ) == (8e-7, 1.00027, 0.001979, 120e-15, 17e9, 0.95, 7.8e-24)
    assert beam["E0_peak"] == 0.0 and beam["energy_J"] is None
    assert beam["transverse_profile"] == {
        "type": "flat_top_cosine", "radius_m": 0.001979, "edge_start_fraction": 0.9,
    }

    assert prop["linear_model"] == "bk_nee"
    assert prop["z_max"] == 1.3 and prop["dz"] == 1e-4
    assert prop["strang"] is True
    assert prop["linear_chunk_t"] == 8
    assert prop["auto_substep"] is True
    assert prop["dz_min"] == 2.5e-5 and prop["grow_factor"] == 1.5
    assert prop["precheck_kerr"] is True and prop["max_precheck_iter"] == 8
    assert prop["use_self_steepening"] is True
    assert prop["use_electronic_kerr"] is True
    assert prop["use_raman_phase"] is False
    assert prop["use_plasma_phase"] is True
    assert prop["use_ionization_loss"] is True
    assert prop["use_raman_absorption"] is False
    assert prop["use_ionization_solver"] is True
    assert prop["full_linear_factorize"] is False
    assert prop["self_steepening_method"] == "tdiff"
    assert prop["safety_mode"] == "on"
    assert prop["focus_window_step"] is True
    assert prop["focus_center_m"] == 0.95 and prop["focus_halfwidth_m"] == 0.1
    assert prop["dz_focus"] == 5e-5 and prop["limit_focus_window"] is False
    assert prop["use_raman_full_operator"] is True

    assert ion["time_mode"] == "full" and ion["integrator"] == "rk4"
    assert len(ion["species"]) == 2
    for species, name, fraction, ip, zeff in (
        (ion["species"][0], "N2", 0.8, 15.6, 0.9),
        (ion["species"][1], "O2", 0.2, 12.1, 0.53),
    ):
        assert species["name"] == name
        assert species["rate"] == "ppt_talebpour_i_lut"
        assert species["reference_model"] == "ppt_talebpour_i_full_reference"
        assert species["fraction"] == fraction and species["Ip_eV"] == ip
        assert species["Z"] == 1 and species["l"] == 0 and species["m"] == 0
        assert species["Zeff"] == zeff
    assert ion["rate_table"]["enabled"] is True
    assert ion["rate_table"]["reuse_cache"] is True
    assert ion["rate_table"]["force_rebuild"] is False

    assert raman["enabled"] is True
    assert raman["model"] == "isaacs_rot_sinexp"
    assert raman["method"] == "iir"
    assert raman["absorption"] is False
    assert raman["omega_R"] == 1.6e13 and raman["Gamma_R"] == 1.3e13
    assert raman["n_R"] == 2.3e-23
    assert raman["operator_convention"] == "isaacs_eq27"
    assert raman["iir_sampling"] == "exact_piecewise_linear"
    assert raman["operator_integrator"] == "heun"
    assert raman["nonlinear_split_order"] == "strang"
    if raman.get("operator_mode") not in {SOURCE_MODE, COMPLETE_MODE}:
        raise AssertionError(f"unexpected fixed Raman operator mode: {raman.get('operator_mode')!r}")
    if prop.get("use_raman_full_operator") is not True:
        raise AssertionError("fixed C2 contract requires use_raman_full_operator=true")


def build(base: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the candidate and verify that exactly one field changed."""
    _assert_fixed(base)
    if base["raman"].get("operator_mode") != SOURCE_MODE:
        raise ValueError(f"source config must use {SOURCE_MODE!r}")
    derived = copy.deepcopy(base)
    derived["raman"]["operator_mode"] = COMPLETE_MODE
    _assert_fixed(derived)
    differences = config_diff(base, derived)
    expected = [{
        "path": "raman.operator_mode",
        "full_isaacs_eq27": SOURCE_MODE,
        "full_isaacs_eq27_complete": COMPLETE_MODE,
    }]
    if differences != expected:
        raise ValueError(f"complete Eq.27 configuration is not single-variable: {differences}")
    return derived, differences


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _assert_locked_base(path: Path) -> None:
    actual = sha256(path)
    if actual != LOCKED_BASE_SHA256:
        raise ValueError(
            "locked C2 base configuration SHA256 mismatch: "
            f"expected={LOCKED_BASE_SHA256} actual={actual} path={path}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)

    args.base = args.base.resolve()
    args.out_dir = args.out_dir.resolve()
    _assert_locked_base(args.base)
    base = json.loads(args.base.read_text(encoding="utf-8"))
    derived, differences = build(base)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.out_dir / CANDIDATE_NAME
    config_bytes = (json.dumps(derived, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    config_path.write_bytes(config_bytes)
    derived_sha = hashlib.sha256(config_bytes).hexdigest()

    diff_payload = {
        "schema": "khz_filament.isaacs_complete_eq27.c2_config_diff.v1",
        "base_config": _rel(args.base),
        "derived_config": _rel(config_path),
        "differences": differences,
        "single_causal_variable": "raman.operator_mode: full_isaacs_eq27 -> full_isaacs_eq27_complete",
        "source_operator_mode": SOURCE_MODE,
        "candidate_operator_mode": COMPLETE_MODE,
        "status": "passed",
    }
    (args.out_dir / "c2_config_diff.json").write_text(
        json.dumps(diff_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # These fixed raw NPZ/metadata records are the only accepted fallback
    # comparators.  CSVs are generated from them by the fallback-audit tool;
    # caller-supplied CSVs are never an authorization input.
    fallback = {
        "provenance_class": "fallback_verified_non_strict",
        "strict_sha_locked": True,
        "classification_qualification": (
            "physical classification is allowed only with the fixed raw NPZ and metadata "
            "chain; this remains a non-strict cross-run comparator"
        ),
        "comparators": FALLBACK_EVIDENCE,
        "excluded_invalid_jobs": ["179706", "179988"],
        "exclusion_reason": "both failed/invalid for physical onset classification; retained only as numerical/audit provenance",
    }

    manifest = {
        "schema": "khz_filament.isaacs_complete_eq27.c2_submission_manifest.v1",
        "campaign_id": CAMPAIGN_ID,
        "remote_campaign_root": REMOTE_CAMPAIGN_ROOT,
        "status": "prepared_not_submitted",
        "parent_c1_commit": PARENT_C1_COMMIT,
        "c1_gate": {
            "commit": C1_COMMIT,
            "summary_path": C1_SUMMARY_REL,
            "summary_sha256": C1_SUMMARY_SHA256,
            "report_path": C1_REPORT_REL,
            "report_sha256": C1_REPORT_SHA256,
            "overall": "PASS",
        },
        "prepared_from_git_sha": _git_head(),
        # The final source commit is not known until this preparation is
        # committed.  Bind it later with the external execution lock rather
        # than making the manifest self-referential.
        "expected_git_sha": None,
        "execution_lock_required": True,
        "locked_base_config_sha256": LOCKED_BASE_SHA256,
        "expected_git_sha_resolution": "external execution_lock generated after final source commit",
        "source_config": _rel(args.base),
        "source_config_sha256": sha256(args.base),
        "derived_config": _rel(config_path),
        "derived_config_sha256": derived_sha,
        "strict_config_diff": differences,
        "single_causal_variable": "electronic Eq.27 operator form",
        "causal_interpretation": (
            "complete combined Eq.27 implementation: electronic Kerr moves from the central "
            "scalar phase/shock approximation into the combined electronic+rotational Strang "
            "half-stages; coefficients and all other configuration fields remain fixed"
        ),
        "causal_limit": (
            "the finite-dz result does not separately identify derivative algebra, electronic "
            "stage placement, or electronic-rotational Heun coupling"
        ),
        "operator_modes": {"source": SOURCE_MODE, "candidate": COMPLETE_MODE},
        "jobs_authorized": 1,
        "jobs_submitted": 0,
        "full_jobs_authorized": 1,
        "full_propagation_jobs_authorized": 1,
        "full_production_jobs_submitted": 0,
        "scan_jobs_authorized": 0,
        "parameter_scan_authorized": False,
        "profiling_jobs_authorized": 0,
        "profiling_authorized": False,
        "additional_profiling_smokes_authorized": 0,
        "optimization_jobs_authorized": 0,
        "walltime_policy": {
            "partition": "gpu",
            "requested_time": "15:00:00",
            "max_time": "UNLIMITED",
        },
        "resources": {
            "partition": "gpu",
            "gpu_count": 1,
            "cpu_threads": 8,
            "requested_time": "15:00:00",
            "expected_gpu_model": EXPECTED_GPU_MODEL,
            "memory_policy": "site default per requested GPU; batch metadata records peak allocated/reserved bytes",
            "peak_allocated_bytes_from_existing_metadata": None,
            "peak_reserved_bytes_from_existing_metadata": None,
        },
        "shared_resources": {
            "gpu_count": 1,
            "cpu_threads": 8,
            "expected_gpu_model": EXPECTED_GPU_MODEL,
            "site_default_memory_mb_per_gpu": 126000,
        },
        "fallback_provenance": fallback,
        "comparison_qualification": (
            "jobs 180748/180749 are fallback_verified_non_strict and used mixed_precision, "
            "whereas the locked mother/candidate configuration retains its baseline default "
            "linear precision; classification must retain both qualifications"
        ),
        "comparison_inputs": {
            "current_full_eq27_job": "180748",
            "raman_off_job": "180749",
            "candidate_case_id": "complete_eq27",
            "pycap_120fs": PYCAP_REL,
            "pycap_120fs_sha256": PYCAP_SHA256,
            "coordinate_definition": "x_focus_cm = 100 * (z_m - 0.95)",
        },
        "raw_npz_policy": "raw candidate NPZ stays in RUN_DIR/HPC and is not copied into the repository",
    }
    (args.out_dir / "submission_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "config": _rel(config_path),
        "config_sha256": derived_sha,
        "differences": differences,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
