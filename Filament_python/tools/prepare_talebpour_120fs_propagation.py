#!/usr/bin/env python3
"""Preflight exactly one 120 fs Talebpour full-model propagation; never submit it."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

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


CASE_ID = "120fs_talebpour_full_model"
DEFAULT_CONFIG = FILAMENT_ROOT / "configs" / "ionization_model_propagation" / f"{CASE_ID}.json"
EXECUTION_SHA = "8dcd01ee38adf2167a2fd6083ae4785e94de89a0"


def _git(args: list[str]) -> str:
    return subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True, check=True).stdout.strip()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _verify(config: dict) -> None:
    beam, prop, ion = config["beam"], config["propagation"], config["ionization"]
    assert math.isclose(float(beam["tau_fwhm"]), 120e-15, abs_tol=1e-27)
    assert float(beam["P0_peak"]) == 17e9 and float(beam["focal_length"]) == 0.95
    assert beam["transverse_profile"] == {"type": "flat_top_cosine", "radius_m": 0.001979, "edge_start_fraction": 0.9}
    assert ion["time_mode"] == "full" and ion["integrator"] == "rk4"
    expected = {"N2": (15.6, 0.9), "O2": (12.55, 0.53)}
    for sp in ion["species"]:
        assert sp["rate"] == "ppt_talebpour_i_lut" and sp["reference_model"] == "ppt_talebpour_i_full_reference"
        assert (float(sp["Ip_eV_eff"]), float(sp["Zeff"])) == expected[sp["name"]]
    for key in ("use_electronic_kerr", "use_raman_phase", "use_plasma_phase", "use_ionization_loss", "use_raman_absorption", "use_self_steepening", "use_ionization_solver"):
        assert prop[key] is True


def _job_script(remote_root: str) -> str:
    return f'''#!/bin/bash
#SBATCH -J tal120
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00
#SBATCH -o {remote_root}/logs/tal120-%j.out
#SBATCH -e {remote_root}/logs/tal120-%j.err
set -euo pipefail
RUN_ROOT="{remote_root}"
CODE_DIR="$RUN_ROOT/source/Filament_python"
CFG="$RUN_ROOT/configs/{CASE_ID}.json"
OUT="$RUN_ROOT/cases/{CASE_ID}/result.npz"
META="$RUN_ROOT/cases/{CASE_ID}/run_metadata.json"
mkdir -p "$(dirname "$OUT")" "$RUN_ROOT/logs" "$RUN_ROOT/cases/{CASE_ID}/figures" "$RUN_ROOT/analysis"
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export CUDA_DEVICE_ORDER=PCI_BUS_ID UPPE_USE_GPU=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}" OPENBLAS_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}" MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}" NUMEXPR_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}"
export CODE_SHA="$(cat "$RUN_ROOT/EXECUTION_GIT_SHA")" CFG_SHA="$(sha256sum "$CFG" | awk '{{print $1}}')"
STATUS=running RC=0
python - "$META" <<'PY'
import json, os, sys
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps({{'case_id':'{CASE_ID}','status':'running','exit_code':0,'slurm_job_id':os.environ.get('SLURM_JOB_ID',''),'execution_git_sha':os.environ['CODE_SHA'],'config_sha256':os.environ['CFG_SHA'],'diagnostic_schema':'khz_filament.propagation_observability.v1'}},indent=2)+'\\n')
PY
cd "$CODE_DIR"
python test_run.py --cfg "$CFG" --gpu --dtype fp32 --out "$OUT" --fig-dir "$RUN_ROOT/cases/{CASE_ID}/figures" --fig-dpi 200
STATUS=completed python - "$META" <<'PY'
import json, os, sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text()); d['status']=os.environ['STATUS']; p.write_text(json.dumps(d,indent=2)+'\\n')
PY
python tools/validate_current_observability_baseline.py --npz "$OUT" --config "$CFG" --run-metadata "$META" --out-dir "$RUN_ROOT/analysis"
'''


def prepare(config_path: Path, out_dir: Path, remote_run_root: str, *, remote_target_verified_empty: bool) -> dict:
    if not remote_run_root.startswith("/data/run01/scvi806/"):
        raise ValueError("unsafe remote root")
    dirty = bool(_git(["git", "status", "--porcelain"]))
    if out_dir.exists():
        raise FileExistsError(out_dir)
    config = json.loads(config_path.read_text(encoding="utf-8")); _verify(config)
    out_dir.mkdir(parents=True)
    snapshot = out_dir / f"{CASE_ID}.json"; snapshot.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    grid, beam, prop, ion, heat, run, raman = load_all(str(snapshot))
    ion_lut = copy.deepcopy(ion); ion_lut.rate_table = dict(ion.rate_table); ion_lut.rate_table.update({"cache_dir": str(out_dir / "lut_cache"), "save_tables": True})
    omega0 = 2 * math.pi * float(c0) / float(beam.lam0)
    tables = prepare_ionization_lut_cache(ion_lut, omega0, float(beam.n0))
    if len(tables) != 2 or any("talebpour" not in str(t.get("model_name", "")) for t in tables):
        raise RuntimeError("Talebpour LUT isolation check failed")
    for table in tables:
        meta = table.get("metadata", {})
        if float(meta.get("I_min_SI", float("inf"))) > 1e8 or float(meta.get("I_max_SI", 0.0)) < 1e19:
            raise RuntimeError("Talebpour LUT intensity range does not cover 1e8-1e19 W/m2")
    W = make_Wfunc("phase5_talebpour_preflight", ion_lut, omega0, float(beam.n0))
    probe = np.asarray(W(np.asarray([1e8, 1e16, 1e18, 1e19])), dtype=float)
    dry_grid = GridConfig(Nx=8, Ny=8, Nt=16, Lx=8e-4, Ly=8e-4, Twin=160e-15)
    dry_beam = dataclasses.replace(beam, energy_J=1e-9, P0_peak=None, focal_length=None)
    dry_prop = dataclasses.replace(prop, z_max=2e-4, dz=1e-4, linear_model="paraxial", auto_substep=False, focus_window_step=False, limit_focus_window=False, progress_every_z=0, energy_probe_every=0, diag_extra=False)
    dry = out_dir / "dry_run.npz"; run_demo(grid=dry_grid, beam=dry_beam, prop=dry_prop, ion=ion_lut, heat=heat, run=run, raman=raman, out_path=str(dry), dtype="fp32")
    with np.load(dry, allow_pickle=False) as data:
        missing = [k for k in REQUIRED_Z_FIELDS + REQUIRED_SCALARS if k not in data.files]
        bad = [k for k in REQUIRED_Z_FIELDS if k in data.files and (data[k].ndim != 1 or data[k].size != data["z_axis"].size or not np.all(np.isfinite(data[k])))]
    job = out_dir / "submit_talebpour_120fs.sh"; job.write_text(_job_script(remote_run_root), encoding="utf-8", newline="\n")
    (out_dir / "EXECUTION_GIT_SHA").write_text(EXECUTION_SHA + "\n", encoding="utf-8")
    passed = (not dirty) and remote_target_verified_empty and not missing and not bad and bool(np.all(np.isfinite(probe)))
    manifest = {"schema":"khz_filament.phase5.talebpour_120fs_preflight.v1","case_id":CASE_ID,"generation_git_sha":_git(["git","rev-parse","HEAD"]),"execution_git_sha":EXECUTION_SHA,"worktree_dirty":dirty,"config_path":_relative(config_path),"config_sha256":sha256(config_path),"config_snapshot":_relative(snapshot),"config_snapshot_sha256":sha256(snapshot),"remote_run_root":remote_run_root,"remote_target_verified_empty":remote_target_verified_empty,"lut_tables":[{"model":t["model_name"],"species":t["species_name"],"metadata":t.get("metadata",{})} for t in tables],"rate_probe_s-1":probe.tolist(),"dry_run":{"passed":not missing and not bad,"missing_fields":missing,"invalid_fields":bad},"required_diagnostics":list(REQUIRED_Z_FIELDS+REQUIRED_SCALARS),"single_job_script":_relative(job),"preflight_passed":passed,"prohibited_jobs":["40fs_talebpour_full_model","120fs_O2_off"]}
    (out_dir / "talebpour_120fs_preflight.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    (out_dir / "talebpour_120fs_preflight_report.md").write_text(f"# Talebpour 120 fs preflight\\n\\nPreflight: **{'passed' if passed else 'failed'}**.\\n\\n- Execution Git SHA: `{EXECUTION_SHA}`\\n- Config SHA256: `{manifest['config_sha256']}`\\n- Remote target verified empty: `{remote_target_verified_empty}`\\n- LUTs: Talebpour-only N2/O2 tables covering `1e8-1e19 W/m2`.\\n- Dry run passed: `{manifest['dry_run']['passed']}`.\\n- Script contains one 120 fs Talebpour job only; it does not submit it.\\n",encoding="utf-8")
    if not passed: raise RuntimeError("Talebpour preflight failed")
    return manifest


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",type=Path,default=DEFAULT_CONFIG); p.add_argument("--out-dir",type=Path,required=True); p.add_argument("--remote-run-root",required=True); p.add_argument("--remote-target-verified-empty",action="store_true"); a=p.parse_args()
    print(f"preflight_passed={prepare(a.config,a.out_dir,a.remote_run_root,remote_target_verified_empty=a.remote_target_verified_empty)['preflight_passed']}")


if __name__ == "__main__": main()
