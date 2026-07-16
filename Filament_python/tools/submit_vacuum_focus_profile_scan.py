#!/usr/bin/env python3
"""Materialize one manifest and one Slurm-array script per FT90 focus case.

The script has no network side effects.  It prepares the bundle that is copied
to the GPU account, where the generated ``submit_profile_scan.sh`` is submitted
with ``sbatch``.  Every array index maps to exactly one manifest/case directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _gI(x: np.ndarray, y: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    X, Y = np.meshgrid(x, y, indexing="xy"); r = np.sqrt(X**2 + Y**2)
    flat, zero = float(profile["flat_radius_m"]), float(profile["zero_radius_m"])
    if profile["kind"] == "hard":
        return (r <= zero).astype(float)
    taper = 0.5 * (1.0 + np.cos(np.pi * (r - flat) / (zero - flat)))
    return np.where(r <= flat, 1.0, np.where(r < zero, taper, 0.0))


def _second_moment(x: np.ndarray, y: np.ndarray, profile: dict[str, Any]) -> float:
    g = _gI(x, y, profile); X, Y = np.meshgrid(x, y, indexing="xy")
    return float(np.sqrt(2.0 * (g * (X**2 + Y**2)).sum() / g.sum()))


def _profile_from_stage(entry: dict[str, Any], common: dict[str, Any], *, scale: float = 1.0) -> dict[str, Any]:
    R = float(common["nominal_radius_m"]) * scale
    return {"id": entry["id"], "kind": entry["kind"], "flat_radius_m": R * float(entry["flat_fraction"]), "zero_radius_m": R * float(entry["zero_fraction"]), "nominal_radius_scale": scale}


def _solve_p6(entries: dict[str, dict[str, Any]], common: dict[str, Any], grid: dict[str, float]) -> tuple[dict[str, Any], float, float]:
    x = (np.arange(int(grid["Nx"])) - int(grid["Nx"]) // 2) * float(grid["Lx_m"]) / int(grid["Nx"])
    y = (np.arange(int(grid["Ny"])) - int(grid["Ny"]) // 2) * float(grid["Ly_m"]) / int(grid["Ny"])
    target = _second_moment(x, y, _profile_from_stage(entries["P1_current_ft90"], common))
    entry = entries["P6_P2_second_moment_matched"]
    lo, hi = 0.5, 1.8
    for _ in range(56):
        mid = 0.5 * (lo + hi)
        current = _second_moment(x, y, _profile_from_stage(entry, common, scale=mid))
        if current < target: lo = mid
        else: hi = mid
    scale = 0.5 * (lo + hi); solved = _profile_from_stage(entry, common, scale=scale)
    return solved, scale, target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--bundle-dir", required=True)
    args = parser.parse_args()
    stage = json.loads(Path(args.stage).read_text(encoding="utf-8"))
    out = Path(args.bundle_dir); cases_dir = out / "cases"; cases_dir.mkdir(parents=True, exist_ok=True)
    common = stage["common"]; entries = {item["id"]: item for item in stage["profiles"]}
    base_grid = {"Nx": int(common["Nx"]), "Ny": int(common["Ny"]), "Lx_m": float(common["Lx_m"]), "Ly_m": float(common["Ly_m"])}
    p6, scale, target = _solve_p6(entries, common, base_grid)
    profiles = {key: (_profile_from_stage(value, common) if key != "P6_P2_second_moment_matched" else p6) for key, value in entries.items()}
    generated: list[dict[str, Any]] = []
    for entry in stage["profiles"]:
        pid = entry["id"]
        generated.append({"case_id": pid, "label": entry["label"], "profile": profiles[pid], "grid": base_grid, "kind": "profile"})
    for item in stage["convergence"]:
        profile = profiles[item["profile_id"]]
        if item["id"] == "P1_current_ft90_baseline":
            continue  # profile P1 already provides the baseline metrics.
        generated.append({"case_id": item["id"], "label": item["id"], "profile": profile, "grid": {key: item[key] for key in ("Nx", "Ny", "Lx_m", "Ly_m")}, "kind": "convergence"})
    for index, item in enumerate(generated):
        item.update({"coordinate_definition": stage["coordinate_definition"], "common": common, "index": index})
        (cases_dir / f"{index:02d}_{item['case_id']}.json").write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
    manifest = {"stage_id": stage["stage_id"], "coordinate_definition": stage["coordinate_definition"], "source_stage": str(Path(args.stage)), "p6_nominal_radius_scale": scale, "p1_discrete_second_moment_radius_m": target, "cases": generated, "resources": stage["resources"], "quality_gates": stage["quality_gates"], "nonlinear_terms": stage["nonlinear_terms"]}
    (out / "profile_scan_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    array_limit = int(stage["resources"]["array_concurrency"])
    script = f'''#!/bin/bash
#SBATCH -p {stage["resources"]["partition"]}
#SBATCH --gres=gpu:{stage["resources"]["gpus"]}
#SBATCH --cpus-per-task={stage["resources"]["cpus_per_task"]}
#SBATCH --time={stage["resources"]["time"]}
#SBATCH --array=0-{len(generated)-1}%{array_limit}
set -euo pipefail
cd "${{SLURM_SUBMIT_DIR}}"
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export UPPE_USE_GPU=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}" OPENBLAS_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}" MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}"
CASE=$(printf 'cases/%02d_' "$SLURM_ARRAY_TASK_ID")
CASE_FILE=$(compgen -G "${{CASE}}*.json")
CASE_ID=$(basename "$CASE_FILE" .json | cut -c4-)
EXTRA=()
if [[ "$CASE_ID" == "P1_current_ft90" ]]; then EXTRA+=(--save-focus-plane); fi
mkdir -p "results/$CASE_ID"
python tools/run_vacuum_focus_profile_case.py --case "$CASE_FILE" --out-dir "results/$CASE_ID" --gpu "${{EXTRA[@]}}"
'''
    path = out / "submit_profile_scan.sh"; path.write_text(script, encoding="utf-8", newline="\n"); path.chmod(0o755)
    print(json.dumps({"bundle": str(out), "cases": len(generated), "p6_scale": scale}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
