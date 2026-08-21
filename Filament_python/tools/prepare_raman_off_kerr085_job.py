#!/usr/bin/env python3
"""Prepare the single 120 fs Raman-phase-OFF + 0.85 electronic-Kerr job."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT / "results" / "raman_phase_causality" / "raman_phase_off_120fs_20260718T201000Z"
    / "120fs_talebpour_full_model_raman_phase_off.json"
)
OUT = ROOT / "results" / "raman_off_kerr085_causality"
BASE_N2 = 7.8e-24
KERR_SCALE = 0.85
CANDIDATE_N2 = 6.63e-24
EXECUTED_PARENT_SHA256 = "d57aadda4c75999722f63919ac92d6a7a42c743d9c3ae2837d502e98176a49b5"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(value, prefix="") -> dict[str, object]:
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, child in value.items():
            result.update(flatten(child, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: value}


def config_diff(base: dict, derived: dict) -> list[dict]:
    flat_base, flat_derived = flatten(base), flatten(derived)
    return [
        {"path": key, "raman_phase_off": flat_base.get(key), "raman_off_kerr085": flat_derived.get(key)}
        for key in sorted(set(flat_base) | set(flat_derived))
        if flat_base.get(key) != flat_derived.get(key)
    ]


def assert_phase_off_parent(config: dict) -> None:
    grid, beam = config["grid"], config["beam"]
    prop, raman, ion = config["propagation"], config["raman"], config["ionization"]
    assert (grid["Nx"], grid["Ny"], grid["Nt"], grid["Twin"]) == (512, 512, 384, 9.6e-13)
    assert (beam["P0_peak"], beam["tau_fwhm"], beam["focal_length"]) == (17e9, 120e-15, 0.95)
    assert beam["transverse_profile"] == {
        "type": "flat_top_cosine", "radius_m": 0.001979, "edge_start_fraction": 0.9,
    }
    assert prop["linear_model"] == "bk_nee"
    assert prop["use_electronic_kerr"] is True and prop["use_raman_phase"] is False
    assert prop["use_raman_absorption"] is True and raman["absorption"] is True
    assert raman["absorption_model"] == "conv_deriv"
    assert prop["use_self_steepening"] is True
    assert prop["use_plasma_phase"] is True and prop["use_ionization_loss"] is True
    assert prop["use_ionization_solver"] is True
    assert ion["time_mode"] == "full" and ion["integrator"] == "rk4"
    assert [item["rate"] for item in ion["species"]] == ["ppt_talebpour_i_lut", "ppt_talebpour_i_lut"]


def build(base: dict) -> tuple[dict, list[dict]]:
    assert_phase_off_parent(base)
    if float(base["beam"]["n2_air"]) != BASE_N2:
        raise ValueError("Raman-phase-OFF parent n2_air is not the locked 7.8e-24 value")
    derived = copy.deepcopy(base)
    derived["beam"]["n2_air"] = CANDIDATE_N2
    assert_phase_off_parent(derived)
    diff = config_diff(base, derived)
    expected = [{
        "path": "beam.n2_air",
        "raman_phase_off": BASE_N2,
        "raman_off_kerr085": CANDIDATE_N2,
    }]
    if diff != expected:
        raise ValueError(f"candidate configuration is not single-variable relative to Raman-phase-OFF: {diff}")
    if abs(CANDIDATE_N2 / BASE_N2 - KERR_SCALE) > 1e-15:
        raise ValueError("candidate n2 does not equal 0.85 times the phase-OFF parent")
    return derived, diff


def executed_parent_crlf_sha256(path: Path) -> str:
    """Reproduce the CRLF bytes recorded by completed job 176915."""
    lf = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    base = json.loads(args.base.read_text(encoding="utf-8"))
    if executed_parent_crlf_sha256(args.base) != EXECUTED_PARENT_SHA256:
        raise ValueError("archived parent does not reproduce job 176915 config SHA256")
    derived, differences = build(base)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.out_dir / "raman_off_kerr085_120fs_config.json"
    config_bytes = (json.dumps(derived, indent=2) + "\n").encode("utf-8")
    config_path.write_bytes(config_bytes)
    config_sha = hashlib.sha256(config_bytes).hexdigest()

    diff_payload = {
        "schema": "khz_filament.raman_off_kerr085.config_diff.v1",
        "direct_parent": str(args.base.relative_to(ROOT)),
        "derived_config": str(config_path.relative_to(ROOT)),
        "differences": differences,
        "single_physical_variable": "beam.n2_air",
        "electronic_kerr_scale": KERR_SCALE,
        "status": "passed",
    }
    (args.out_dir / "config_diff.json").write_text(
        json.dumps(diff_payload, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema": "khz_filament.raman_off_kerr085.submission_manifest.v1",
        "status": "prepared_not_submitted",
        "jobs_authorized": 1,
        "full_production_jobs_submitted": 0,
        "direct_parent_worktree_lf_sha256": sha256(args.base),
        "direct_parent_executed_crlf_sha256": EXECUTED_PARENT_SHA256,
        "direct_parent_line_ending_note": "JSON semantics are identical; job 176915 recorded CRLF bytes, while Git checkout enforces LF",
        "derived_config_sha256": config_sha,
        "strict_config_diff": differences,
        "physics_statement": "Raman phase OFF with electronic Kerr coefficient scaled to 0.85 of the phase-OFF parent",
        "resources": {
            "partition": "gpu", "gpu_count": 1, "cpu_threads": 8,
            "site_default_memory_mb_per_gpu": 126000, "requested_time": "08:00:00",
            "expected_gpu_model": "NVIDIA GeForce RTX 5090",
        },
        "comparison_series": {
            "production": "results/ionization_model_propagation/talebpour_120fs_20260717T114321Z/baseline_axial_diagnostics.csv",
            "raman_phase_off": "results/raman_phase_causality/raman_phase_off_120fs_20260718T201000Z/raman_phase_off_axial_diagnostics.csv",
            "historical_fr_mixture": "results/historical_fr_mixture_causality/postprocess_215812/historical_fr_mixture_axial_diagnostics.csv",
            "pycap": "results/density_translation_width/density_translation_width_20260715_002/paper_pycap_120fs.csv",
        },
    }
    (args.out_dir / "submission_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
