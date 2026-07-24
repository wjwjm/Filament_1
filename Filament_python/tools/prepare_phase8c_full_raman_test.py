#!/usr/bin/env python3
"""Prepare the strict full-Eq.27 Raman feedback ON/OFF Test A pair."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "isaacs_raman_closure" / "120fs_talebpour_isaacs_full_operator_on_energy_audit.json"
OUT = ROOT / "results" / "phase8c_full_raman_causality"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(_flatten(child, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: value}


def config_diff(on: dict, off: dict) -> list[dict]:
    flat_on, flat_off = _flatten(on), _flatten(off)
    return [
        {"path": key, "on": flat_on.get(key), "off": flat_off.get(key)}
        for key in sorted(set(flat_on) | set(flat_off))
        if flat_on.get(key) != flat_off.get(key)
    ]


def _assert_common(config: dict) -> None:
    p, b, r, g, ion = (config["propagation"], config["beam"], config["raman"], config["grid"], config["ionization"])
    assert (g["Nx"], g["Ny"], g["Nt"], g["Lx"], g["Ly"], g["Twin"]) == (512, 512, 384, 0.008, 0.008, 9.6e-13)
    assert (b["lam0"], b["P0_peak"], b["tau_fwhm"], b["focal_length"]) == (800e-9, 17e9, 120e-15, 0.95)
    assert b["transverse_profile"] == {"type": "flat_top_cosine", "radius_m": 0.001979, "edge_start_fraction": 0.9}
    assert (p["z_max"], p["dz"], p["focus_center_m"], p["focus_halfwidth_m"], p["dz_focus"]) == (1.3, 1e-4, 0.95, 0.10, 5e-5)
    assert p["linear_model"] == "bk_nee" and p["linear_precision_strategy"] == "mixed_precision"
    assert p["use_electronic_kerr"] is True and p["use_self_steepening"] is True
    assert p["use_plasma_phase"] is True and p["use_ionization_loss"] is True and p["use_ionization_solver"] is True
    assert p["use_raman_phase"] is False and p["use_raman_absorption"] is False and r["absorption"] is False
    assert (r["model"], r["operator_mode"], r["operator_integrator"], r["nonlinear_split_order"], r["iir_sampling"]) == ("isaacs_rot_sinexp", "full_isaacs_eq27", "heun", "strang", "exact_piecewise_linear")
    assert (r["omega_R"], r["Gamma_R"], r["n_R"]) == (1.6e13, 1.3e13, 2.3e-23)
    assert ion["species"][0]["rate"] == "ppt_talebpour_i_lut" and ion["species"][1]["rate"] == "ppt_talebpour_i_lut"


def build(base: dict) -> tuple[dict, dict, list[dict]]:
    on = copy.deepcopy(base)
    p = on["propagation"]
    p["linear_precision_strategy"] = "mixed_precision"
    p["use_raman_full_operator"] = True
    # These values already exist in the frozen production configuration; make
    # their intended state explicit for the strict-pair audit.
    p["use_raman_phase"] = False
    p["use_raman_absorption"] = False
    on["raman"]["absorption"] = False
    _assert_common(on)
    off = copy.deepcopy(on)
    off["propagation"]["use_raman_full_operator"] = False
    _assert_common(off)
    diff = config_diff(on, off)
    expected = [{"path": "propagation.use_raman_full_operator", "on": True, "off": False}]
    if diff != expected:
        raise ValueError(f"strict Test A configuration pair is not single-variable: {diff}")
    return on, off, diff


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    base = json.loads(args.base.read_text(encoding="utf-8"))
    on, off, differences = build(base)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    on_path, off_path = args.out_dir / "test_a_on_config.json", args.out_dir / "test_a_off_config.json"
    on_path.write_text(json.dumps(on, indent=2) + "\n", encoding="utf-8")
    off_path.write_text(json.dumps(off, indent=2) + "\n", encoding="utf-8")
    diff = {
        "schema": "phase8c.full_eq27_raman.test_a.config_diff.v1",
        "base_config": str(args.base.relative_to(ROOT)),
        "on_config": str(on_path.relative_to(ROOT)),
        "off_config": str(off_path.relative_to(ROOT)),
        "differences": differences,
        "single_physical_variable": "propagation.use_raman_full_operator",
        "status": "passed",
    }
    (args.out_dir / "test_a_config_diff.json").write_text(json.dumps(diff, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema": "phase8c.full_eq27_raman.test_a.submission_manifest.v1",
        "status": "prepared_not_submitted",
        "jobs_authorized": 2,
        "additional_profiling_smokes_authorized": 0,
        "optimization_jobs_authorized": 0,
        "walltime_policy": {"partition": "gpu", "max_time": "UNLIMITED", "requested_time": "15:00:00"},
        "shared_resources": {"gpu_count": 1, "cpu_threads": 8, "site_default_memory_mb_per_gpu": 126000, "nodelist": "g0609", "expected_gpu_model": "NVIDIA GeForce RTX 5090"},
        "source_config_sha256": _sha256(args.base),
        "on_config_sha256": _sha256(on_path),
        "off_config_sha256": _sha256(off_path),
        "strict_config_diff": differences,
        "full_production_jobs_submitted": 0,
    }
    (args.out_dir / "test_a_submission_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
