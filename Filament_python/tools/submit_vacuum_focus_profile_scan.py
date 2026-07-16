#!/usr/bin/env python3
"""Materialize independent GPU cases for FT90 window closure and Fresnel checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _gI(x: np.ndarray, y: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    X, Y = np.meshgrid(x, y, indexing="xy"); radius = np.sqrt(X**2 + Y**2)
    flat, zero = float(profile["flat_radius_m"]), float(profile["zero_radius_m"])
    if profile["kind"] == "hard":
        return (radius <= zero).astype(float)
    taper = 0.5 * (1.0 + np.cos(np.pi * (radius - flat) / (zero - flat)))
    return np.where(radius <= flat, 1.0, np.where(radius < zero, taper, 0.0))


def _second_moment(x: np.ndarray, y: np.ndarray, profile: dict[str, Any]) -> float:
    g = _gI(x, y, profile); X, Y = np.meshgrid(x, y, indexing="xy")
    return float(np.sqrt(2.0 * (g * (X**2 + Y**2)).sum() / g.sum()))


def _profile_from_stage(entry: dict[str, Any], common: dict[str, Any], *, scale: float = 1.0) -> dict[str, Any]:
    radius = float(common["nominal_radius_m"]) * scale
    return {"id": entry["id"], "kind": entry["kind"], "flat_radius_m": radius * float(entry["flat_fraction"]), "zero_radius_m": radius * float(entry["zero_fraction"]), "nominal_radius_scale": scale}


def _solve_p6(entries: dict[str, dict[str, Any]], common: dict[str, Any], grid: dict[str, float]) -> tuple[dict[str, Any], float, float]:
    x = (np.arange(int(grid["Nx"])) - int(grid["Nx"]) // 2) * float(grid["Lx_m"]) / int(grid["Nx"])
    y = (np.arange(int(grid["Ny"])) - int(grid["Ny"]) // 2) * float(grid["Ly_m"]) / int(grid["Ny"])
    target = _second_moment(x, y, _profile_from_stage(entries["P1_current_ft90"], common))
    entry = entries["P6_P2_second_moment_matched"]
    lo, hi = 0.5, 1.8
    for _ in range(56):
        mid = 0.5 * (lo + hi)
        current = _second_moment(x, y, _profile_from_stage(entry, common, scale=mid))
        if current < target:
            lo = mid
        else:
            hi = mid
    scale = 0.5 * (lo + hi)
    return _profile_from_stage(entry, common, scale=scale), scale, target


def _grid(item: dict[str, Any]) -> dict[str, float]:
    return {key: item[key] for key in ("Nx", "Ny", "Lx_m", "Ly_m")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--bundle-dir", required=True)
    args = parser.parse_args()
    stage = json.loads(Path(args.stage).read_text(encoding="utf-8"))
    out = Path(args.bundle_dir); cases_dir = out / "cases"; cases_dir.mkdir(parents=True, exist_ok=True)
    common = stage["common"]; entries = {item["id"]: item for item in stage["profiles"]}
    p6_scales: dict[str, float] = {}; p1_moments: dict[str, float] = {}

    def profile_for(profile_id: str, grid: dict[str, float]) -> dict[str, Any]:
        if profile_id != "P6_P2_second_moment_matched":
            return _profile_from_stage(entries[profile_id], common)
        profile, scale, target = _solve_p6(entries, common, grid)
        key = f"{int(grid['Nx'])}x{int(grid['Ny'])}_{grid['Lx_m']:.3g}m"
        p6_scales[key] = scale; p1_moments[key] = target
        return profile

    generated: list[dict[str, Any]] = []
    for window in stage["window_scan"]:
        grid = _grid(window)
        for entry in stage["profiles"]:
            profile_id = entry["id"]
            generated.append({
                "case_id": f"{profile_id}__{window['id']}", "label": f"{entry['label']} @ {window['id']}",
                "kind": "window", "window_id": window["id"], "profile_id": profile_id,
                "profile": profile_for(profile_id, grid), "grid": grid,
            })
    resolution = stage["resolution_check"]; resolution_grid = _grid(resolution)
    generated.append({
        "case_id": f"{resolution['profile_id']}__{resolution['id']}", "label": f"P1 resolution @ {resolution['id']}",
        "kind": "resolution", "window_id": resolution["id"], "profile_id": resolution["profile_id"],
        "profile": profile_for(resolution["profile_id"], resolution_grid), "grid": resolution_grid,
    })
    for index, item in enumerate(generated):
        item.update({"coordinate_definition": stage["coordinate_definition"], "common": common, "fresnel_radial_samples": int(stage["fresnel_radial_samples"]), "index": index})
        (cases_dir / f"{index:02d}_{item['case_id']}.json").write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "stage_id": stage["stage_id"], "coordinate_definition": stage["coordinate_definition"], "source_stage": str(Path(args.stage)),
        "cases": generated, "resources": stage["resources"], "quality_gates": stage["quality_gates"], "nonlinear_terms": stage["nonlinear_terms"],
        "p6_nominal_radius_scales": p6_scales, "p1_discrete_second_moment_radius_m": p1_moments,
    }
    (out / "profile_scan_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    resources = stage["resources"]
    script = f'''#!/bin/bash
#SBATCH -p {resources["partition"]}
#SBATCH --gres=gpu:{resources["gpus"]}
#SBATCH --cpus-per-task={resources["cpus_per_task"]}
#SBATCH --time={resources["time"]}
#SBATCH --array=0-{len(generated)-1}%{resources["array_concurrency"]}
set -euo pipefail
cd "${{SLURM_SUBMIT_DIR}}"
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export UPPE_USE_GPU=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}" OPENBLAS_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}" MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-4}}"
CASE=$(printf 'cases/%02d_' "$SLURM_ARRAY_TASK_ID")
CASE_FILE=$(compgen -G "${{CASE}}*.json")
CASE_ID=$(basename "$CASE_FILE" .json | cut -c4-)
mkdir -p "results/$CASE_ID"
python tools/run_vacuum_focus_profile_case.py --case "$CASE_FILE" --out-dir "results/$CASE_ID" --gpu --fresnel-crosscheck
'''
    path = out / "submit_profile_scan.sh"; path.write_text(script, encoding="utf-8", newline="\n"); path.chmod(0o755)
    print(json.dumps({"bundle": str(out), "cases": len(generated), "p6_scales": p6_scales}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
