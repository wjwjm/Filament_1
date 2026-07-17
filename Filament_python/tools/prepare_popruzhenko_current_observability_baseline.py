#!/usr/bin/env python3
"""Prepare and preflight the sole authorized current-observability 120 fs Pop run."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
FILAMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = FILAMENT_ROOT.parent
if str(FILAMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(FILAMENT_ROOT))

from KHz_filament.confio import load_all  # noqa: E402
from KHz_filament.config import GridConfig  # noqa: E402
from KHz_filament.constants import c0  # noqa: E402
from KHz_filament.ionization import make_Wfunc, prepare_ionization_lut_cache  # noqa: E402
from KHz_filament.runner import run_demo  # noqa: E402
from validate_current_observability_baseline import REQUIRED_SCALARS, REQUIRED_Z_FIELDS, sha256  # noqa: E402


DEFAULT_CONFIG = FILAMENT_ROOT / "configs" / "profile_validation" / "flat_top_90_120fs.json"
CASE_ID = "120fs_popruzhenko_full_model_current_observability"


def _git(command: list[str]) -> str:
    return subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=True).stdout.strip()


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _verify_config(config: dict[str, Any]) -> None:
    beam, propagation, ion = config["beam"], config["propagation"], config["ionization"]
    assert math.isclose(float(beam["tau_fwhm"]), 120e-15, rel_tol=0.0, abs_tol=1e-27)
    assert float(beam["P0_peak"]) == 17e9 and float(beam["focal_length"]) == 0.95
    assert beam["transverse_profile"] == {"type": "flat_top_cosine", "radius_m": 0.001979, "edge_start_fraction": 0.9}
    assert str(ion["time_mode"]) == "full" and str(ion["integrator"]) == "rk4"
    for species in ion["species"]:
        assert species["rate"] == "popruzhenko_atom_i_lut"
        assert species["reference_model"] == "popruzhenko_atom_i_full_reference"
    for key in ("use_electronic_kerr", "use_raman_phase", "use_plasma_phase", "use_ionization_loss", "use_raman_absorption", "use_self_steepening", "use_ionization_solver"):
        assert propagation[key] is True


def _write_remote_job(path: Path, remote_run_root: str) -> None:
    script = f'''#!/bin/bash
#SBATCH -J pop120obs
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00
#SBATCH -o {remote_run_root}/logs/pop120obs-%j.out
#SBATCH -e {remote_run_root}/logs/pop120obs-%j.err
set -euo pipefail
RUN_ROOT="{remote_run_root}"
CODE_DIR="$RUN_ROOT/source/Filament_python"
CFG="$RUN_ROOT/configs/120fs_popruzhenko_full_model_current_observability.json"
OUT="$RUN_ROOT/cases/{CASE_ID}/result.npz"
ANALYSIS="$RUN_ROOT/analysis"
META="$RUN_ROOT/cases/{CASE_ID}/run_metadata.json"
mkdir -p "$(dirname "$OUT")" "$RUN_ROOT/logs" "$RUN_ROOT/cases/{CASE_ID}/figures" "$ANALYSIS"
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export CUDA_DEVICE_ORDER=PCI_BUS_ID UPPE_USE_GPU=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}" OPENBLAS_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}" MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}" NUMEXPR_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}"
CODE_SHA="$(git -C "$CODE_DIR" rev-parse HEAD)"
CFG_SHA="$(sha256sum "$CFG" | awk '{{print $1}}')"
write_meta() {{
  STATUS="$1" RC="$2" CODE_SHA="$CODE_SHA" CFG_SHA="$CFG_SHA" python - "$META" <<'PY'
import json, os, sys
from pathlib import Path
p=Path(sys.argv[1]); data={{}}
if p.exists(): data=json.loads(p.read_text(encoding='utf-8'))
data.update({{'case_id':'{CASE_ID}','status':os.environ['STATUS'],'exit_code':int(os.environ['RC']),
             'slurm_job_id':os.environ.get('SLURM_JOB_ID',''),'execution_git_sha':os.environ['CODE_SHA'],
             'config_sha256':os.environ['CFG_SHA'],'config_path':'configs/120fs_popruzhenko_full_model_current_observability.json',
             'output_npz':'cases/{CASE_ID}/result.npz','diagnostic_schema':'khz_filament.propagation_observability.v1'}})
p.write_text(json.dumps(data,indent=2)+"\\n",encoding='utf-8')
PY
}}
trap 'rc=$?; write_meta failed "$rc"; exit "$rc"' ERR
write_meta running 0
cd "$CODE_DIR"
python test_run.py --cfg "$CFG" --gpu --dtype fp32 --out "$OUT" --fig-dir "$RUN_ROOT/cases/{CASE_ID}/figures" --fig-dpi 200
write_meta completed 0
python tools/validate_current_observability_baseline.py --npz "$OUT" --config "$CFG" --run-metadata "$META" --out-dir "$ANALYSIS"
'''
    path.write_text(script, encoding="utf-8", newline="\n")


def prepare(config_path: Path, out_dir: Path, remote_run_root: str) -> dict[str, Any]:
    config_path, out_dir = config_path.resolve(), out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"preflight output already exists: {out_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8")); _verify_config(config)
    out_dir.mkdir(parents=True)
    config_snapshot = out_dir / f"{CASE_ID}.json"
    config_snapshot.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    grid, beam, prop, ion, heat, run, raman = load_all(str(config_snapshot))
    ion_for_lut = copy.deepcopy(ion)
    ion_for_lut.rate_table = dict(ion.rate_table)
    ion_for_lut.rate_table.update({"cache_dir": str(out_dir / "lut_cache"), "save_tables": True})
    omega0 = 2.0 * math.pi * float(c0) / float(beam.lam0)
    tables = prepare_ionization_lut_cache(ion_for_lut, omega0, float(beam.n0))
    Wfunc = make_Wfunc("phase5_preflight", ion_for_lut, omega0, float(beam.n0))
    rate_probe = np.asarray(Wfunc(np.asarray([1e16, 1e17, 1e18], dtype=float)), dtype=float)
    effective_species = [{key: entry.get(key) for key in ("name", "rate", "fraction", "Ip_eV", "Ui_J")} for entry in Wfunc._species_entries]
    dry_grid = GridConfig(Nx=8, Ny=8, Nt=16, Lx=8e-4, Ly=8e-4, Twin=160e-15)
    dry_beam = dataclasses.replace(beam, energy_J=1e-9, P0_peak=None, focal_length=None)
    dry_prop = dataclasses.replace(prop, z_max=2e-4, dz=1e-4, linear_model="paraxial", auto_substep=False, focus_window_step=False, limit_focus_window=False, progress_every_z=0, energy_probe_every=0, diag_extra=False)
    dry_out = out_dir / "dry_run.npz"
    run_demo(grid=dry_grid, beam=dry_beam, prop=dry_prop, ion=ion_for_lut, heat=heat, run=run, raman=raman, out_path=str(dry_out), dtype="fp32")
    with np.load(dry_out, allow_pickle=False) as data:
        missing_dry = [key for key in REQUIRED_Z_FIELDS + REQUIRED_SCALARS if key not in data.files]
        bad = [key for key in REQUIRED_Z_FIELDS if key in data.files and (data[key].ndim != 1 or data[key].size != data["z_axis"].size or not np.all(np.isfinite(data[key])))]
    job_script = out_dir / "submit_popruzhenko_120fs_current_observability.sh"
    _write_remote_job(job_script, remote_run_root)
    status = not missing_dry and not bad and bool(np.all(np.isfinite(rate_probe)))
    manifest = {
        "schema": "khz_filament.phase5.popruzhenko_current_observability_preflight.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": CASE_ID,
        "git_sha": _git(["git", "rev-parse", "HEAD"]),
        "worktree_dirty": bool(_git(["git", "status", "--porcelain"])),
        "config_path": _repo_relative(config_path), "config_sha256": sha256(config_path),
        "config_snapshot": _repo_relative(config_snapshot), "config_snapshot_sha256": sha256(config_snapshot),
        "remote_run_root": remote_run_root,
        "effective_species": effective_species,
        "lut_tables": [{"model": table["model_name"], "species": table["species_name"], "metadata": table.get("metadata", {})} for table in tables],
        "rate_probe_s-1": rate_probe.tolist(),
        "required_diagnostic_fields": list(REQUIRED_Z_FIELDS + REQUIRED_SCALARS),
        "dry_run": {"npz": _repo_relative(dry_out), "missing_fields": missing_dry, "invalid_fields": bad, "passed": status},
        "single_job_script": _repo_relative(job_script),
        "expected_remote_output": f"cases/{CASE_ID}/result.npz",
        "preflight_passed": status,
    }
    (out_dir / "popruzhenko_120fs_current_observability_preflight.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Popruzhenko 120 fs current-observability preflight", "", f"Preflight: **{'passed' if status else 'failed'}**.", "", f"- Git SHA: `{manifest['git_sha']}`", f"- Worktree clean: `{not manifest['worktree_dirty']}`", f"- Config: `{manifest['config_path']}`", f"- Config SHA256: `{manifest['config_sha256']}`", f"- Remote output root: `{remote_run_root}`", f"- Single-job script: `{manifest['single_job_script']}`", f"- Dry run: `{status}`; missing diagnostics: `{missing_dry}`; invalid diagnostics: `{bad}`.", "", "No Talebpour, 40 fs, or O2-off job is present in the generated submission script."]
    (out_dir / "popruzhenko_120fs_current_observability_preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not status:
        raise RuntimeError("preflight failed")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--remote-run-root", required=True)
    args = parser.parse_args()
    result = prepare(args.config, args.out_dir, args.remote_run_root)
    print(f"preflight_passed={result['preflight_passed']}")


if __name__ == "__main__":
    main()
