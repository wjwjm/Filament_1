#!/usr/bin/env python3
"""Prepare and submit reproducible filamentation stages."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_CASE_IDS = {"40fs", "120fs"}
ALLOWED_DIFFERENCES = {"beam.tau_fwhm", "raman.tau_fwhm"}


def load_stage_spec(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    for key in ("stage_id", "stage_name", "base_config", "cases"):
        if not spec.get(key): raise ValueError(f"stage spec missing {key}")
    cases = spec["cases"]
    if spec["stage_id"] == "stage1":
        if len(cases) != 2 or {case.get("case_id") for case in cases} != EXPECTED_CASE_IDS:
            raise ValueError("Stage 1 must contain exactly 40fs and 120fs cases")
    seen: set[str] = set(); labels: set[str] = set(); widths: set[float] = set()
    for case in cases:
        case_id, label, width = case.get("case_id"), case.get("label"), case.get("tau_fwhm_fs")
        if not case_id or not label or not isinstance(width, (int, float)) or width <= 0: raise ValueError("invalid case definition")
        if case_id in seen or label in labels or float(width) in widths: raise ValueError("case ID, label, and pulse width must be unique")
        seen.add(case_id); labels.add(label); widths.add(float(width))
    return spec


def _get(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for key in path.split("."): value = value[key]
    return value


def validate_stage_invariants(base_config: dict[str, Any], stage_spec: dict[str, Any]) -> None:
    expected = stage_spec["required_invariants"]
    for path, expected_value in expected.items():
        actual = _get(base_config, path)
        if actual != expected_value: raise ValueError(f"stage invariant failed: {path}={actual!r}, expected {expected_value!r}")
    if stage_spec["stage_id"] == "stage1" and Path(stage_spec["base_config"]).name == "config_ref.json":
        raise ValueError("Stage 1 must not use config_ref.json")


def build_case_config(base_config: dict[str, Any], case_spec: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["beam"]["tau_fwhm"] = float(case_spec["tau_fwhm_fs"]) * 1e-15
    raman = config.get("raman", {})
    if raman.get("absorption_model") == "closed_form" and "tau_fwhm" in raman:
        raman["tau_fwhm"] = config["beam"]["tau_fwhm"]
    return config


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, child in value.items(): flattened.update(_flatten(child, f"{prefix}.{key}" if prefix else key))
        return flattened
    return {prefix: value}


def validate_case_differences(configs: list[dict[str, Any]], allowed_paths: set[str] = ALLOWED_DIFFERENCES) -> None:
    reference = _flatten(configs[0])
    for config in configs[1:]:
        candidate = _flatten(config)
        for path in set(reference) | set(candidate):
            if reference.get(path) != candidate.get(path) and path not in allowed_paths:
                raise ValueError(f"case configurations differ outside allowed paths: {path}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(cwd: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, text=True, capture_output=True, check=True).stdout.strip()


def _stage_root(script_dir: Path, spec: dict[str, Any], run_id: str) -> Path:
    return script_dir / "outputs" / spec["stage_name"] / run_id


def prepare_stage_directory(stage_spec: dict[str, Any], run_id: str, script_dir: Path, case_configs: dict[str, dict[str, Any]]) -> tuple[Path, dict[str, Path]]:
    root = _stage_root(script_dir, stage_spec, run_id)
    if root.exists(): raise FileExistsError(f"run ID already exists: {root}")
    root.mkdir(parents=True)
    for relative in ("configs", "comparison", "reports", "logs"):
        (root / relative).mkdir()
    config_paths: dict[str, Path] = {}
    for case in stage_spec["cases"]:
        case_id = case["case_id"]
        (root / "cases" / case_id / "figures").mkdir(parents=True)
        path = root / "configs" / f"{case_id}.json"
        path.write_text(json.dumps(case_configs[case_id], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        config_paths[case_id] = path
    (root / "stage_spec_snapshot.json").write_text(json.dumps(stage_spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return root, config_paths


def _paths(root: Path, case_id: str) -> dict[str, Path]:
    case_dir = root / "cases" / case_id
    return {"case_dir": case_dir, "npz": case_dir / "result.npz", "mat": case_dir / "result.mat", "figures": case_dir / "figures", "metadata": case_dir / "run_metadata.json"}


def _sbatch(args: list[str], cwd: Path) -> str:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=True)
    return result.stdout.strip().split(";", 1)[0]


def submit_simulation_jobs(stage_spec: dict[str, Any], root: Path, config_paths: dict[str, Path], script_dir: Path) -> dict[str, str]:
    jobs: dict[str, str] = {}
    for case in stage_spec["cases"]:
        case_id = case["case_id"]; paths = _paths(root, case_id)
        exports = {"STAGE_ID": stage_spec["stage_id"], "STAGE_NAME": stage_spec["stage_name"], "RUN_ID": root.name, "CASE_ID": case_id, "CASE_LABEL": case["label"], "PULSE_WIDTH_FS": str(case["tau_fwhm_fs"]), "RUN_METADATA": str(paths["metadata"]), "CFG": str(config_paths[case_id]), "OUT": str(paths["npz"]), "MAT_DIR": str(paths["case_dir"]), "MAT_NAME": "result.mat", "FIG_DIR": str(paths["figures"]), "FIG_DPI": str(stage_spec["figure_dpi"]), "Z_SHIFT_CM": str(stage_spec["z_shift_cm"]), "CONVERT_TO_MAT": "1", "REMOVE_NPZ": "1", "GENERATE_FIGURES": "1"}
        command = ["sbatch", "--parsable", f"--job-name=s1_single_{case_id}", f"--output={root / 'logs' / f'{case_id}-%j.out'}", "--export=" + "ALL," + ",".join(f"{key}={value}" for key, value in exports.items()), str(script_dir / "sub.sh")]
        jobs[case_id] = _sbatch(command, script_dir)
    return jobs


def submit_stage_postprocess_job(stage_spec: dict[str, Any], root: Path, simulation_jobs: dict[str, str], script_dir: Path) -> str:
    resources = stage_spec["postprocess_resources"]
    dependency = "afterok:" + ":".join(simulation_jobs[case["case_id"]] for case in stage_spec["cases"])
    command = ["sbatch", "--parsable", "--job-name=s1_post_40_120", f"--dependency={dependency}", f"--partition={resources['partition']}", f"--gres=gpu:{resources['gpus']}", f"--cpus-per-task={resources['cpus_per_task']}", f"--time={resources['time']}", f"--output={root / 'logs' / 'stage1-post-%j.out'}", f"--export=ALL,STAGE_DIR={root}", str(script_dir / "sub_stage_postprocess.sh")]
    return _sbatch(command, script_dir)


def write_submission_manifest(stage_spec: dict[str, Any], root: Path, base_config_path: Path, config_paths: dict[str, Path], simulation_jobs: dict[str, str | None], post_job: str | None, script_dir: Path, dry_run: bool) -> dict[str, Any]:
    manifest = {"stage_id": stage_spec["stage_id"], "stage_name": stage_spec["stage_name"], "run_id": root.name, "git_commit": _git_commit(script_dir), "base_config": str(base_config_path), "comparison_mode": stage_spec["comparison_mode"], "submitted_at_utc": datetime.now(timezone.utc).isoformat(), "dry_run": dry_run, "config_sha256": {case: _sha256(path) for case, path in config_paths.items() if path.exists()}, "simulation_job_ids": simulation_jobs, "stage_postprocess_job_id": post_job, "paths": {case["case_id"]: {key: str(value.relative_to(root)) for key, value in _paths(root, case["case_id"]).items()} for case in stage_spec["cases"]}}
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit a defined filamentation stage")
    parser.add_argument("--spec", required=True); parser.add_argument("--run-id", default=None); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); script_dir = Path(__file__).resolve().parent; spec_path = Path(args.spec).resolve()
    spec = load_stage_spec(spec_path); base_config_path = (spec_path.parent / spec["base_config"]).resolve(); base = json.loads(base_config_path.read_text(encoding="utf-8")); validate_stage_invariants(base, spec)
    configs = {case["case_id"]: build_case_config(base, case) for case in spec["cases"]}; validate_case_differences(list(configs.values()))
    run_id = args.run_id or f"{spec['stage_id']}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    root = _stage_root(script_dir, spec, run_id)
    if args.dry_run:
        virtual_paths = {case_id: root / "configs" / f"{case_id}.json" for case_id in configs}
        manifest = {"stage_id": spec["stage_id"], "stage_name": spec["stage_name"], "run_id": run_id, "dry_run": True, "comparison_mode": spec["comparison_mode"], "base_config": str(base_config_path), "cases": [case["case_id"] for case in spec["cases"]], "stage_directory": str(root), "simulation_job_ids": {case["case_id"]: None for case in spec["cases"]}, "stage_postprocess_job_id": None, "planned_config_paths": {key: str(value) for key, value in virtual_paths.items()}}
        print(json.dumps(manifest, indent=2, ensure_ascii=False)); return 0
    root, config_paths = prepare_stage_directory(spec, run_id, script_dir, configs)
    jobs = submit_simulation_jobs(spec, root, config_paths, script_dir)
    post_job = submit_stage_postprocess_job(spec, root, jobs, script_dir)
    manifest = write_submission_manifest(spec, root, base_config_path, config_paths, jobs, post_job, script_dir, False)
    (root / "submission_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
