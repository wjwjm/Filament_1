#!/usr/bin/env python3
"""Prepare and submit the controlled Gaussian-versus-FT90 profile stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_CASE_IDS = ("profile_g_120", "profile_ft90_120")


def _get(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        value = value[part]
    return value


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            output.update(_flatten(child, child_prefix))
        return output
    return {prefix: value}


def _is_profile_path(path: str) -> bool:
    return path == "beam.transverse_profile" or path.startswith("beam.transverse_profile.")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def load_stage_spec(path: str | Path) -> dict[str, Any]:
    spec = _read_json(Path(path))
    required = ("stage_id", "stage_name", "cases", "required_invariants", "simulation_resources", "postprocess_resources")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"profile validation stage spec missing: {', '.join(missing)}")
    cases = spec["cases"]
    if not isinstance(cases, list) or tuple(case.get("case_id") for case in cases) != EXPECTED_CASE_IDS:
        raise ValueError("profile validation must contain Gaussian and FT90 120 fs cases in the declared order")
    for case in cases:
        for key in ("case_id", "label", "profile_type", "config"):
            if not case.get(key):
                raise ValueError(f"profile validation case missing {key}")
    resources = spec["simulation_resources"]
    for key in ("partition", "gpus", "cpus_per_task", "memory", "time"):
        if not resources.get(key):
            raise ValueError(f"profile validation simulation_resources missing {key}")
    if int(resources["gpus"]) != 1:
        raise ValueError("profile validation requires exactly one GPU per simulation case")
    post_resources = spec["postprocess_resources"]
    for key in ("partition", "gpus", "cpus_per_task", "memory", "time"):
        if not post_resources.get(key):
            raise ValueError(f"profile validation postprocess_resources missing {key}")
    if int(post_resources["gpus"]) != 1:
        raise ValueError("profile validation requires exactly one GPU for postprocessing")
    return spec


def load_and_validate_case_configs(spec: dict[str, Any], spec_path: str | Path) -> dict[str, dict[str, Any]]:
    spec_path = Path(spec_path).resolve()
    configs: dict[str, dict[str, Any]] = {}
    for case in spec["cases"]:
        path = (spec_path.parent / case["config"]).resolve()
        config = _read_json(path)
        for invariant_path, expected in spec["required_invariants"].items():
            actual = _get(config, invariant_path)
            if actual != expected:
                raise ValueError(
                    f"{case['case_id']} violates stage invariant: {invariant_path}={actual!r}, expected {expected!r}"
                )
        profile = config.get("beam", {}).get("transverse_profile")
        if not isinstance(profile, dict) or profile.get("type") != case["profile_type"]:
            raise ValueError(f"{case['case_id']} profile does not match declared profile_type")
        configs[case["case_id"]] = config

    flattened = [_flatten(configs[case["case_id"]]) for case in spec["cases"]]
    reference = flattened[0]
    for candidate in flattened[1:]:
        for path in set(reference) | set(candidate):
            if reference.get(path) != candidate.get(path) and not _is_profile_path(path):
                raise ValueError(f"profile validation configurations differ outside beam.transverse_profile: {path}")
    return configs


def _stage_root(script_dir: Path, spec: dict[str, Any], run_id: str) -> Path:
    return script_dir / "outputs" / spec["stage_name"] / run_id


def prepare_stage_directory(spec: dict[str, Any], run_id: str, script_dir: Path, configs: dict[str, dict[str, Any]]) -> tuple[Path, dict[str, Path]]:
    root = _stage_root(script_dir, spec, run_id)
    if root.exists():
        raise FileExistsError(f"run ID already exists: {root}")
    root.mkdir(parents=True)
    for relative in ("configs", "comparison", "reports", "logs"):
        (root / relative).mkdir()

    config_paths: dict[str, Path] = {}
    for case in spec["cases"]:
        case_id = case["case_id"]
        (root / "cases" / case_id / "figures").mkdir(parents=True)
        config_path = root / "configs" / f"{case_id}.json"
        config_path.write_text(json.dumps(configs[case_id], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        config_paths[case_id] = config_path
    (root / "stage_spec_snapshot.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return root, config_paths


def _paths(root: Path, case_id: str) -> dict[str, Path]:
    case_dir = root / "cases" / case_id
    return {
        "case_dir": case_dir,
        "npz": case_dir / "result.npz",
        "mat": case_dir / "result.mat",
        "figures": case_dir / "figures",
        "metadata": case_dir / "run_metadata.json",
    }


def _sbatch(command: list[str], cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=True)
    return result.stdout.strip().split(";", 1)[0]


def submit_simulation_jobs(spec: dict[str, Any], root: Path, config_paths: dict[str, Path], script_dir: Path) -> dict[str, str]:
    jobs: dict[str, str] = {}
    resources = spec["simulation_resources"]
    for case in spec["cases"]:
        case_id = case["case_id"]
        paths = _paths(root, case_id)
        exports = {
            "STAGE_ID": spec["stage_id"], "STAGE_NAME": spec["stage_name"], "RUN_ID": root.name,
            "CASE_ID": case_id, "CASE_LABEL": case["label"], "PULSE_WIDTH_FS": "120",
            "PROFILE_TYPE": case["profile_type"], "RUN_METADATA": str(paths["metadata"]),
            "CFG": str(config_paths[case_id]), "OUT": str(paths["npz"]), "MAT_DIR": str(paths["case_dir"]),
            "MAT_NAME": "result.mat", "FIG_DIR": str(paths["figures"]), "FIG_DPI": str(spec["figure_dpi"]),
            "Z_SHIFT_CM": str(spec["z_shift_cm"]), "DTYPE": str(spec.get("dtype", "fp32")),
            "CONVERT_TO_MAT": "1", "REMOVE_NPZ": "0", "GENERATE_FIGURES": "1",
        }
        command = [
            "sbatch", "--parsable", f"--job-name=pv_{case_id}",
            f"--partition={resources['partition']}", f"--gres=gpu:{resources['gpus']}",
            f"--cpus-per-task={resources['cpus_per_task']}", f"--mem={resources['memory']}",
            f"--time={resources['time']}",
            f"--output={root / 'logs' / f'{case_id}-%j.out'}",
            "--export=" + "ALL," + ",".join(f"{key}={value}" for key, value in exports.items()),
            str(script_dir / "sub.sh"),
        ]
        jobs[case_id] = _sbatch(command, script_dir)
    return jobs


def submit_postprocess_job(spec: dict[str, Any], root: Path, jobs: dict[str, str], script_dir: Path) -> str:
    resources = spec["postprocess_resources"]
    dependency = "afterok:" + ":".join(jobs[case["case_id"]] for case in spec["cases"])
    command = [
        "sbatch", "--parsable", "--job-name=pv_post_g_ft90", f"--dependency={dependency}",
        f"--partition={resources['partition']}", f"--gres=gpu:{resources['gpus']}",
        f"--cpus-per-task={resources['cpus_per_task']}", f"--mem={resources['memory']}",
        f"--time={resources['time']}",
        f"--output={root / 'logs' / 'profile-post-%j.out'}", f"--export=ALL,STAGE_DIR={root}",
        str(script_dir / "sub_profile_validation_postprocess.sh"),
    ]
    return _sbatch(command, script_dir)


def build_manifest(spec: dict[str, Any], root: Path, config_paths: dict[str, Path], jobs: dict[str, str | None], post_job: str | None, script_dir: Path, dry_run: bool) -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=script_dir, text=True, capture_output=True, check=True).stdout.strip()
    return {
        "stage_id": spec["stage_id"], "stage_name": spec["stage_name"], "run_id": root.name,
        "git_commit": commit, "submitted_at_utc": datetime.now(timezone.utc).isoformat(), "dry_run": dry_run,
        "comparison_mode": "same_peak_power_same_geometry_different_transverse_profile",
        "config_sha256": {case_id: _sha256(path) for case_id, path in config_paths.items() if path.exists()},
        "simulation_job_ids": jobs, "stage_postprocess_job_id": post_job,
        "paths": {case["case_id"]: {key: str(value.relative_to(root)) for key, value in _paths(root, case["case_id"]).items()} for case in spec["cases"]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and submit the transverse-profile validation stage")
    parser.add_argument("--spec", default="stages/transverse_profile_validation.json")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parents[1]
    spec_path = Path(args.spec).resolve()
    spec = load_stage_spec(spec_path)
    configs = load_and_validate_case_configs(spec, spec_path)
    run_id = args.run_id or f"{spec['stage_id']}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    root = _stage_root(script_dir, spec, run_id)
    if args.dry_run:
        manifest = build_manifest(
            spec,
            root,
            {case_id: root / "configs" / f"{case_id}.json" for case_id in configs},
            {case["case_id"]: None for case in spec["cases"]},
            None,
            script_dir,
            True,
        )
        manifest["stage_directory"] = str(root)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    root, config_paths = prepare_stage_directory(spec, run_id, script_dir, configs)
    jobs = submit_simulation_jobs(spec, root, config_paths, script_dir)
    post_job = submit_postprocess_job(spec, root, jobs, script_dir)
    manifest = build_manifest(spec, root, config_paths, jobs, post_job, script_dir, False)
    (root / "submission_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
