#!/usr/bin/env python3
"""Generate nonlinearity-ablation configs and a manifest; never submit jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
FILAMENT_ROOT = SCRIPT_DIR.parent
if str(FILAMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(FILAMENT_ROOT))

from KHz_filament.config import resolve_nonlinear_switches  # noqa: E402
from KHz_filament.confio import load_all  # noqa: E402


REQUIRED_SWITCHES = (
    "use_electronic_kerr",
    "use_raman_phase",
    "use_plasma_phase",
    "use_ionization_loss",
    "use_raman_absorption",
    "use_self_steepening",
    "use_ionization_solver",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit_sha(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict):
            existing = result.get(key)
            if existing is None:
                existing = {}
            if not isinstance(existing, dict):
                raise ValueError(f"cannot apply nested override through non-object key: {key}")
            result[key] = _deep_update(existing, value)
        else:
            result[key] = deepcopy(value)
    return result


def _leaf_paths(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key, child in value.items():
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        result.update(_leaf_paths(child, child_prefix))
    return result


def _validate_override_scope(base: dict[str, Any], generated: dict[str, Any], overrides: dict[str, Any]) -> None:
    allowed = set(_leaf_paths(overrides))
    before, after = _leaf_paths(base), _leaf_paths(generated)
    all_paths = set(before) | set(after)
    unexpected = sorted(path for path in all_paths if before.get(path) != after.get(path) and path not in allowed)
    if unexpected:
        raise ValueError(f"generated config changed fields outside declared overrides: {unexpected}")


def _effective_switches(path: Path) -> dict[str, bool]:
    _grid, _beam, prop, ion, _heat, _run, raman = load_all(str(path))
    effective = asdict(resolve_nonlinear_switches(prop, raman, ion))
    return {name: bool(effective[name]) for name in REQUIRED_SWITCHES}


def _validate_stage_spec(spec: dict[str, Any]) -> None:
    for key in ("stage_id", "stage_name", "duration_cases", "variants"):
        if key not in spec:
            raise ValueError(f"nonlinear ablation stage spec missing {key}")
    durations = spec["duration_cases"]
    variants = spec["variants"]
    if [item.get("case_id") for item in durations] != ["40fs", "120fs"]:
        raise ValueError("stage requires duration cases in the fixed order: 40fs, 120fs")
    expected_variants = [
        "vacuum",
        "electronic_kerr_only",
        "electronic_kerr_plus_raman_phase",
        "kerr_raman_ionization_plasma_no_loss",
        "kerr_raman_ionization_plasma_with_ionization_loss",
        "full_model",
    ]
    if [item.get("name") for item in variants] != expected_variants:
        raise ValueError(f"stage variants must be exactly {expected_variants}")
    for variant in variants:
        overrides = variant.get("overrides", {})
        if not isinstance(overrides, dict) or not isinstance(overrides.get("propagation"), dict):
            raise ValueError(f"variant {variant.get('name')} requires propagation overrides")
        missing = [name for name in REQUIRED_SWITCHES if name not in overrides["propagation"]]
        if missing:
            raise ValueError(f"variant {variant.get('name')} is missing nonlinear overrides: {missing}")


def generate_ablation_configs(stage_path: Path, output_dir: Path) -> dict[str, Any]:
    stage_path = stage_path.resolve()
    spec = _load_json(stage_path)
    _validate_stage_spec(spec)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    configs_dir = output_dir / "configs"
    configs_dir.mkdir()

    repository_root = FILAMENT_ROOT.parent
    manifest_cases: list[dict[str, Any]] = []
    base_records: list[dict[str, Any]] = []
    for duration in spec["duration_cases"]:
        base_path = (stage_path.parent / str(duration["base_config"])).resolve()
        base = _load_json(base_path)
        base_records.append({
            "case_id": duration["case_id"],
            "pulse_width_fs": float(duration["pulse_width_fs"]),
            "base_config": str(base_path),
            "base_config_sha256": _sha256(base_path),
        })
        for variant in spec["variants"]:
            overrides = deepcopy(variant["overrides"])
            generated = _deep_update(base, overrides)
            _validate_override_scope(base, generated, overrides)
            config_name = f"{duration['case_id']}__{variant['name']}.json"
            config_path = configs_dir / config_name
            config_path.write_text(json.dumps(generated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            effective = _effective_switches(config_path)
            declared = overrides["propagation"]
            mismatch = {key: (declared[key], effective[key]) for key in REQUIRED_SWITCHES if bool(declared[key]) != effective[key]}
            if mismatch:
                raise ValueError(f"effective switch mismatch for {config_name}: {mismatch}")
            manifest_cases.append({
                "case_id": f"{duration['case_id']}__{variant['name']}",
                "duration_case": duration["case_id"],
                "pulse_width_fs": float(duration["pulse_width_fs"]),
                "variant": variant["name"],
                "description": variant["description"],
                "base_config": str(base_path),
                "base_config_sha256": _sha256(base_path),
                "overrides": overrides,
                "effective_nonlinear_switches": effective,
                "ionization_solver_enabled": effective["use_ionization_solver"],
                "config_file": str(config_path.relative_to(output_dir)),
                "config_sha256": _sha256(config_path),
                "output_filename": f"nonlinear_ablation__{duration['case_id']}__{variant['name']}.npz",
            })

    manifest = {
        "schema": "khz_filament.nonlinear_ablation_configs.v1",
        "stage_id": spec["stage_id"],
        "stage_name": spec["stage_name"],
        "stage_spec": str(stage_path),
        "stage_spec_sha256": _sha256(stage_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit_sha": _git_commit_sha(repository_root),
        "job_submission": "not_supported_by_this_generator",
        "base_configs": base_records,
        "cases": manifest_cases,
    }
    manifest_path = output_dir / "nonlinear_ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=FILAMENT_ROOT / "stages" / "nonlinear_ablation_stage1.json")
    parser.add_argument("--out-dir", type=Path, required=True, help="new directory for generated configs and manifest")
    args = parser.parse_args()
    manifest = generate_ablation_configs(args.stage, args.out_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
