#!/usr/bin/env python3
"""Prepare the single 120 fs historical_fr_mixture causal job.

The only physical difference from the frozen production configuration is
``raman.operator_mode = "historical_fr_mixture"``.  Everything else, including
the absorption path, is inherited from the current 120 fs Talebpour baseline.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "ionization_model_propagation" / "120fs_talebpour_full_model.json"
OUT = ROOT / "results" / "historical_fr_mixture_causality"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(_flatten(child, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: value}


def config_diff(base: dict, derived: dict) -> list[dict]:
    flat_base, flat_derived = _flatten(base), _flatten(derived)
    return [
        {"path": key, "base": flat_base.get(key), "historical_fr_mixture": flat_derived.get(key)}
        for key in sorted(set(flat_base) | set(flat_derived))
        if flat_base.get(key) != flat_derived.get(key)
    ]


def _assert_common(config: dict) -> None:
    g = config["grid"]
    b = config["beam"]
    p = config["propagation"]
    r = config["raman"]
    ion = config["ionization"]
    assert (g["Nx"], g["Ny"], g["Nt"], g["Lx"], g["Ly"], g["Twin"]) == (512, 512, 384, 0.008, 0.008, 9.6e-13)
    assert (b["lam0"], b["P0_peak"], b["tau_fwhm"], b["focal_length"]) == (8e-7, 17e9, 120e-15, 0.95)
    assert b["transverse_profile"] == {"type": "flat_top_cosine", "radius_m": 0.001979, "edge_start_fraction": 0.9}
    assert (p["z_max"], p["dz"], p["focus_center_m"], p["focus_halfwidth_m"], p["dz_focus"]) == (1.3, 1e-4, 0.95, 0.10, 5e-5)
    assert p["linear_model"] == "bk_nee"
    assert p.get("linear_precision_strategy") is None
    assert p["use_electronic_kerr"] is True and p["use_self_steepening"] is True
    assert p["use_raman_phase"] is True and p["use_raman_absorption"] is True
    assert p["use_plasma_phase"] is True and p["use_ionization_loss"] is True and p["use_ionization_solver"] is True
    assert r["enabled"] is True and r["model"] == "rot_sinexp"
    assert r["method"] == "iir" and r["absorption"] is True
    assert r["absorption_model"] == "conv_deriv"
    assert r["f_R"] == 0.15 and r["T2"] == 8e-11 and r["T_R"] == 8.4e-12
    assert r["omega_R"] == 1.6e13 and r["Gamma_R"] == 1.3e13 and r["n_R"] == 2.3e-23
    assert ion["time_mode"] == "full" and ion["integrator"] == "rk4"
    assert ion["species"][0]["rate"] == "ppt_talebpour_i_lut"
    assert ion["species"][1]["rate"] == "ppt_talebpour_i_lut"


def build(base: dict) -> tuple[dict, list[dict]]:
    _assert_common(base)
    derived = copy.deepcopy(base)
    derived["raman"]["operator_mode"] = "historical_fr_mixture"
    _assert_common(derived)
    diff = config_diff(base, derived)
    expected = [{
        "path": "raman.operator_mode",
        "base": None,
        "historical_fr_mixture": "historical_fr_mixture",
    }]
    if diff != expected:
        raise ValueError(f"historical_fr_mixture configuration is not single-variable: {diff}")
    return derived, diff


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    base = json.loads(args.base.read_text(encoding="utf-8"))
    derived, differences = build(base)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.out_dir / "historical_fr_mixture_120fs_config.json"
    # Write LF bytes explicitly so the recorded sha256 matches the LF blob that
    # Git stores (see .gitattributes: *.json text eol=lf) and matches sha256sum
    # on the Linux HPC node.
    config_bytes = (json.dumps(derived, indent=2) + "\n").encode("utf-8")
    config_path.write_bytes(config_bytes)
    derived_config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    diff = {
        "schema": "khz_filament.historical_fr_mixture.config_diff.v1",
        "base_config": str(args.base.relative_to(ROOT)),
        "derived_config": str(config_path.relative_to(ROOT)),
        "differences": differences,
        "single_physical_variable": "raman.operator_mode",
        "status": "passed",
    }
    (args.out_dir / "config_diff.json").write_text(json.dumps(diff, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema": "khz_filament.historical_fr_mixture.submission_manifest.v1",
        "status": "prepared_not_submitted",
        "jobs_authorized": 1,
        "additional_profiling_smokes_authorized": 0,
        "optimization_jobs_authorized": 0,
        "walltime_policy": {"partition": "gpu", "max_time": "UNLIMITED", "requested_time": "15:00:00"},
        "shared_resources": {
            "gpu_count": 1,
            "cpu_threads": 8,
            "site_default_memory_mb_per_gpu": 126000,
            "expected_gpu_model": "NVIDIA GeForce RTX 5090",
        },
        "source_config_sha256": _sha256(args.base),
        "derived_config_sha256": derived_config_sha256,
        "strict_config_diff": differences,
        "full_production_jobs_submitted": 0,
        "comparison_series": {
            "current_production": "results/ionization_model_propagation/talebpour_120fs_20260717T114321Z/baseline_axial_diagnostics.csv",
            "raman_phase_off": "results/raman_phase_causality/raman_phase_off_120fs_20260718T201000Z/raman_phase_off_axial_diagnostics.csv",
            "pycap_120fs": "results/density_translation_width/density_translation_width_20260715_002/paper_pycap_120fs.csv",
        },
    }
    (args.out_dir / "submission_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
