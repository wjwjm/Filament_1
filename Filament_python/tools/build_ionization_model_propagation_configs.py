#!/usr/bin/env python3
"""Create the controlled Popruzhenko-to-Talebpour Phase-5 configuration pair.

This tool only writes configuration and provenance artifacts.  It never submits a
job, changes the propagator, or creates an O2-off case.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
FILAMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = FILAMENT_ROOT.parent
BASE_CONFIGS = {
    "120fs": FILAMENT_ROOT / "configs" / "profile_validation" / "flat_top_90_120fs.json",
    "40fs": FILAMENT_ROOT / "configs" / "profile_validation" / "flat_top_90_40fs.json",
}
CASE_IDS = {
    "120fs": "120fs_talebpour_full_model",
    "40fs": "40fs_talebpour_full_model",
}
TAL_PARAMS = {
    "N2": {"Ip_eV_eff": 15.6, "Zeff": 0.9},
    "O2": {"Ip_eV_eff": 12.55, "Zeff": 0.53},
}
ALLOWED_LEAVES = {
    "ionization.species[0].rate", "ionization.species[0].reference_model",
    "ionization.species[0].Ip_eV_eff", "ionization.species[0].Zeff",
    "ionization.species[1].rate", "ionization.species[1].reference_model",
    "ionization.species[1].Ip_eV_eff", "ionization.species[1].Zeff",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        # Unit tests may deliberately use an isolated temporary directory.  Real
        # Phase-5 invocations write beneath REPO_ROOT and therefore remain
        # repository-relative as required by the manifest contract.
        return resolved.as_posix()


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            output.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return output
    if isinstance(value, list):
        output = {}
        for index, item in enumerate(value):
            output.update(_flatten(item, f"{prefix}[{index}]"))
        return output
    return {prefix: value}


def config_differences(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    before, after = _flatten(base), _flatten(candidate)
    return {key: (before.get(key), after.get(key)) for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)}


def make_talebpour_config(base: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    species = result["ionization"]["species"]
    if {entry.get("name") for entry in species} != {"N2", "O2"}:
        raise ValueError("Phase-5 Talebpour configuration requires exactly N2 and O2 species")
    for entry in species:
        name = str(entry["name"])
        entry["rate"] = "ppt_talebpour_i_lut"
        entry["reference_model"] = "ppt_talebpour_i_full_reference"
        entry.update(TAL_PARAMS[name])
    differences = config_differences(base, result)
    unexpected = sorted(set(differences) - ALLOWED_LEAVES)
    if unexpected:
        raise AssertionError(f"unexpected physical configuration differences: {unexpected}")
    if set(differences) != ALLOWED_LEAVES:
        raise AssertionError(f"incomplete Talebpour transformation: {sorted(differences)}")
    return result


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=True).stdout.strip()


def build(out_dir: Path) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out_dir}")
    out_dir.mkdir(parents=True)
    cases = []
    for width, base_path in BASE_CONFIGS.items():
        base = json.loads(base_path.read_text(encoding="utf-8"))
        tal = make_talebpour_config(base)
        out_path = out_dir / f"{CASE_IDS[width]}.json"
        out_path.write_text(json.dumps(tal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cases.append({
            "case_id": CASE_IDS[width], "pulse_width": width,
            "base_config": repo_relative(base_path), "base_config_sha256": sha256(base_path),
            "generated_config": repo_relative(out_path), "generated_config_sha256": sha256(out_path),
            "allowed_differences": sorted(config_differences(base, tal)),
            "talebpour_effective_parameters": TAL_PARAMS,
            "submission_authorized": False,
        })
    manifest = {
        "schema": "khz_filament.phase5.ionization_model_propagation_configs.v1",
        "generation_git_sha": _git_sha(),
        "execution_git_sha_requirement": "8dcd01ee38adf2167a2fd6083ae4785e94de89a0",
        "coordinate_definition": "x_focus_cm = 100 * (z_m - 0.95)",
        "cases": cases,
        "prohibited_submissions": ["40fs_talebpour_full_model", "120fs_O2_off"],
        "next_gate": "Only preflight and submit 120fs_talebpour_full_model after the formal Popruzhenko baseline gate is passed.",
    }
    (out_dir / "ionization_model_propagation_config_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = build(args.out_dir)
    print(json.dumps({"cases": [item["case_id"] for item in manifest["cases"]], "submission_authorized": False}))


if __name__ == "__main__":
    main()
