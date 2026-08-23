#!/usr/bin/env python3
"""Prepare the one permitted Hybrid Propagation 0.60 m paired campaign.

The mother configuration is intentionally read-only.  Two derived configs are
written under ``Filament_python/results/hybrid_propagation_validation``:

* ``reference`` keeps the historical full nonlinear path;
* ``hybrid`` enables the opt-in linear preamble and starts the nonlinear
  operator at the absolute coordinate 0.60 m.

This tool only prepares JSON/configuration provenance.  It never invokes
``sbatch`` and never runs a propagation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


FILAMENT_ROOT = Path(__file__).resolve().parents[1]
REPO = FILAMENT_ROOT.parent
MOTHER = FILAMENT_ROOT / "configs" / "isaacs_raman_closure" / "120fs_talebpour_isaacs_full_operator_on.json"
OUTPUT = FILAMENT_ROOT / "results" / "hybrid_propagation_validation"
REFERENCE_NAME = "120fs_talebpour_isaacs_hybrid_reference.json"
HYBRID_NAME = "120fs_talebpour_isaacs_hybrid_0p60.json"
DIFF_NAME = "hybrid_config_diff.json"
MANIFEST_NAME = "submission_manifest.json"

CAMPAIGN_ID = "hybrid_propagation_validation_0p60"
REMOTE_CAMPAIGN_ROOT = "/data/run01/scvi806/user_Wangjimin/hybrid_propagation_validation_0p60"
REMOTE_ACCOUNT_ROOT = "/data/run01/scvi806/user_Wangjimin"
EXPECTED_GPU_MODEL = "NVIDIA GeForce RTX 5090"
EXPECTED_CPU_THREADS = 8
EXPECTED_GPU_COUNT = 1
REQUESTED_TIME = "15:00:00"
Z_NL_START_M = 0.60
SOURCE_CONFIG_REL = "configs/isaacs_raman_closure/120fs_talebpour_isaacs_full_operator_on.json"
SCHEMA = "khz_filament.hybrid_propagation_validation.submission_manifest.v1"
DIFF_SCHEMA = "khz_filament.hybrid_propagation_validation.config_diff.v1"


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
            result.update(_flatten(child, f"{prefix}.{key}" if prefix else str(key)))
        return result
    return {prefix: value}


def config_diff(reference: dict[str, Any], hybrid: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a deterministic A/B flattened diff.

    The labels deliberately name the two cases rather than the mother config;
    this prevents a later tool from silently treating a third configuration as
    part of this campaign.
    """
    before, after = _flatten(reference), _flatten(hybrid)
    return [
        {
            "path": key,
            "reference": before.get(key),
            "hybrid": after.get(key),
        }
        for key in sorted(set(before) | set(after))
        if before.get(key) != after.get(key)
    ]


def _assert_mother(config: dict[str, Any]) -> None:
    """Assert the fixed mother contract without changing scientific values."""
    propagation = config.get("propagation")
    if not isinstance(propagation, dict):
        raise ValueError("mother config lacks propagation object")
    if propagation.get("limit_focus_window") is not False:
        raise ValueError("Hybrid v1 requires mother propagation.limit_focus_window=false")
    if float(propagation.get("z_max", 0.0)) <= Z_NL_START_M:
        raise ValueError("mother propagation.z_max must be greater than 0.60 m")
    if "raman" not in config or "ionization" not in config or "beam" not in config:
        raise ValueError("mother config is missing a required physics section")


def build_pair(mother: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Build reference/hybrid configs and enforce the two-field A/B diff."""
    _assert_mother(mother)
    reference = copy.deepcopy(mother)
    hybrid = copy.deepcopy(mother)
    for payload in (reference, hybrid):
        prop = payload.setdefault("propagation", {})
        # The field was added after the frozen mother JSON.  It is intentionally
        # inserted into both cases, so it is not an A/B causal difference.
        prop["measure_performance"] = True
        # The formal campaign requires every saved numerical diagnostic to be
        # finite.  Enabling operator-energy probes avoids the legacy disabled
        # NaN sentinels while leaving the propagated field unchanged.
        prop["diag_operator_energy"] = True
        prop["limit_focus_window"] = False
    reference["propagation"]["propagation_mode"] = "full_nonlinear_from_z0"
    reference["propagation"]["z_nl_start"] = 0.0
    hybrid["propagation"]["propagation_mode"] = "hybrid"
    hybrid["propagation"]["z_nl_start"] = Z_NL_START_M
    differences = config_diff(reference, hybrid)
    expected = [
        {
            "path": "propagation.propagation_mode",
            "reference": "full_nonlinear_from_z0",
            "hybrid": "hybrid",
        },
        {
            "path": "propagation.z_nl_start",
            "reference": 0.0,
            "hybrid": Z_NL_START_M,
        },
    ]
    if differences != expected:
        raise ValueError(f"Hybrid pair must have exactly two propagation differences: {differences}")
    return reference, hybrid, differences


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(FILAMENT_ROOT.resolve()).as_posix()
    except ValueError:
        # Unit tests may direct generated artifacts to a temporary directory;
        # production preparation remains inside Filament_python/results.
        return path.resolve().as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(out_dir: Path = OUTPUT, *, mother_path: Path = MOTHER) -> dict[str, Any]:
    mother_path = mother_path.resolve()
    if mother_path != MOTHER.resolve():
        raise ValueError(f"mother config is fixed to {MOTHER}")
    if not mother_path.is_file():
        raise FileNotFoundError(mother_path)
    mother = json.loads(mother_path.read_text(encoding="utf-8"))
    if not isinstance(mother, dict):
        raise ValueError("mother config must be a JSON object")
    reference, hybrid, differences = build_pair(mother)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_path = out_dir / REFERENCE_NAME
    hybrid_path = out_dir / HYBRID_NAME
    _write_json(reference_path, reference)
    _write_json(hybrid_path, hybrid)
    reference_sha = sha256(reference_path)
    hybrid_sha = sha256(hybrid_path)
    mother_sha = sha256(mother_path)

    diff_payload = {
        "schema": DIFF_SCHEMA,
        "status": "passed",
        "mother_config": SOURCE_CONFIG_REL,
        "reference_config": _rel(reference_path),
        "hybrid_config": _rel(hybrid_path),
        "differences": differences,
        "single_causal_variable": "absolute nonlinear start plane at 0.60 m",
        "measure_performance": True,
        "diag_operator_energy": True,
    }
    diff_path = out_dir / DIFF_NAME
    _write_json(diff_path, diff_payload)

    manifest = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "remote_campaign_root": REMOTE_CAMPAIGN_ROOT,
        "remote_account_root": REMOTE_ACCOUNT_ROOT,
        "status": "prepared_not_submitted",
        "expected_git_sha": None,
        "execution_lock_required": True,
        "expected_git_sha_resolution": "external execution_lock generated after final source commit",
        "prepared_from_git_sha": _git_head(),
        "mother_config": SOURCE_CONFIG_REL,
        "mother_config_sha256": mother_sha,
        "reference_config": _rel(reference_path),
        "reference_config_sha256": reference_sha,
        "hybrid_config": _rel(hybrid_path),
        "hybrid_config_sha256": hybrid_sha,
        "config_diff_path": _rel(diff_path),
        "config_diff_sha256": sha256(diff_path),
        "strict_config_diff": differences,
        "cases": {
            "reference": {
                "case_id": "reference",
                "propagation_mode": "full_nonlinear_from_z0",
                "z_nl_start_m": 0.0,
                "config": _rel(reference_path),
                "config_sha256": reference_sha,
            },
            "hybrid": {
                "case_id": "hybrid",
                "propagation_mode": "hybrid",
                "z_nl_start_m": Z_NL_START_M,
                "config": _rel(hybrid_path),
                "config_sha256": hybrid_sha,
            },
        },
        "execution": {
            "allocation_count": 1,
            "case_order": ["reference", "hybrid"],
            "sequential": True,
            "warmup": "small_cuda_import_fft_only_not_a_candidate",
            "retry_policy": "no_retry",
            "dtype": "fp32",
            "measure_performance": True,
            "diag_operator_energy": True,
            "raw_npz_policy": "raw NPZ remains in remote RUN_DIR/HPC and is never committed",
        },
        "resources": {
            "partition": "gpu",
            "gpu_count": EXPECTED_GPU_COUNT,
            "cpu_threads": EXPECTED_CPU_THREADS,
            "requested_time": REQUESTED_TIME,
            "expected_gpu_model": EXPECTED_GPU_MODEL,
        },
        "comparison": {
            "g1_onset_threshold_m3": 1.0e22,
            "g1_max_abs_shift_cm": 0.10,
            "g2_max_peak_relative_difference": 0.02,
            "rho_thresholds_m3": [1.0e19, 1.0e20, 1.0e21, 1.0e22],
            "intensity_superlevel_fractions": [0.10, 0.50, 0.90],
            "component_max_position_difference_cm": 0.10,
            "curve_nrmse_max": 0.02,
            "curve_correlation_min": 0.995,
            "peak_prominence_fraction": 0.05,
            "peak_distance_cm": 0.10,
            "performance_reduction_min_fraction": 0.01,
            "visual_veto_is_explicit_input": True,
        },
        "jobs_authorized": 1,
        "jobs_submitted": 0,
        "full_propagation_jobs_authorized": 1,
        "parameter_scan_authorized": False,
        "profiling_authorized": False,
        "additional_start_planes_authorized": [],
        "pulse_train_authorized": False,
        "round_2_authorized": False,
    }
    manifest_path = out_dir / MANIFEST_NAME
    _write_json(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "campaign_id": CAMPAIGN_ID,
        "mother_config": SOURCE_CONFIG_REL,
        "reference_config": _rel(reference_path),
        "hybrid_config": _rel(hybrid_path),
        "differences": differences,
        "manifest": _rel(manifest_path),
        "reference_config_sha256": reference_sha,
        "hybrid_config_sha256": hybrid_sha,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    parser.add_argument("--mother", type=Path, default=MOTHER)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(prepare(args.out_dir, mother_path=args.mother), ensure_ascii=False, indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
