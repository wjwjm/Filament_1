#!/usr/bin/env bash
#SBATCH --job-name=hr2e-convergence
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
# N50 assigns 126 GB host memory per requested GPU; do not override it.
#SBATCH --time=15:00:00

set -euo pipefail
: "${EXPECTED_GIT_SHA:?missing EXPECTED_GIT_SHA}"
: "${REPO_DIR:?missing REPO_DIR}"
: "${RUN_DIR:?missing RUN_DIR}"
: "${CASE_ID:?missing CASE_ID}"
: "${CONFIG_PATH:?missing CONFIG_PATH}"
: "${EXPECTED_CONFIG_SHA256:?missing EXPECTED_CONFIG_SHA256}"
: "${EXPECTED_GPU_MODEL:?missing EXPECTED_GPU_MODEL}"

cd "${REPO_DIR}"
actual_sha="$(git rev-parse HEAD)"
test "${actual_sha}" = "${EXPECTED_GIT_SHA}" || { echo "FATAL: Git SHA mismatch" >&2; exit 21; }
test -z "$(git status --porcelain=v1)" || { echo "FATAL: staging worktree is not clean" >&2; exit 22; }
actual_config_sha="$(sha256sum "${CONFIG_PATH}" | awk '{print $1}')"
test "${actual_config_sha}" = "${EXPECTED_CONFIG_SHA256}" || { echo "FATAL: config SHA mismatch" >&2; exit 23; }

mkdir -p "${RUN_DIR}"
export PYTHONPYCACHEPREFIX="${RUN_DIR}/pycache"
export CUPY_CACHE_DIR="${RUN_DIR}/cupy_cache"
export XDG_CACHE_HOME="${RUN_DIR}/xdg_cache"
unset PYTHONPATH
export PYTHONPATH="${REPO_DIR}/Filament_python"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export UPPE_USE_GPU=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python

cd "${RUN_DIR}"
python - "${RUN_DIR}/${CASE_ID}_job_metadata.json" <<'PY'
import json, os, subprocess, sys
import cupy as cp
from KHz_filament.device import xp
if xp.__name__ != "cupy" or cp.cuda.runtime.getDeviceCount() < 1:
    raise SystemExit("FATAL: CUDA/CuPy backend is required")
props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
if name != os.environ["EXPECTED_GPU_MODEL"]:
    raise SystemExit(f"FATAL: GPU model mismatch expected={os.environ['EXPECTED_GPU_MODEL']!r} actual={name!r}")
payload = {
    "schema": "khz_filament.hr2e.job_metadata.v1",
    "case_id": os.environ["CASE_ID"],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "git_sha": subprocess.check_output(["git", "-C", os.environ["REPO_DIR"], "rev-parse", "HEAD"], text=True).strip(),
    "config_path": os.environ["CONFIG_PATH"],
    "config_sha256": os.environ["EXPECTED_CONFIG_SHA256"],
    "gpu_model": name,
    "cpu_threads": int(os.environ["SLURM_CPUS_PER_TASK"]),
    "dtype": "fp32",
}
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(payload, indent=2) + "\n")
PY

python -c 'from KHz_filament.runner import run_from_file; import os; run_from_file(os.environ["CONFIG_PATH"], out_path=os.path.join(os.environ["RUN_DIR"], os.environ["CASE_ID"] + ".npz"), dtype="fp32")'
