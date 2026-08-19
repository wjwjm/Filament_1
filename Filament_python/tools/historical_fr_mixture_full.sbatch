#!/usr/bin/env bash
#SBATCH --job-name=hist-fr-mixture
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
# N50 assigns 126 GB host memory per requested GPU; do not override it.
#SBATCH --nodelist=g0609
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
if [ "${actual_sha}" != "${EXPECTED_GIT_SHA}" ]; then
  echo "FATAL: SHA mismatch expected=${EXPECTED_GIT_SHA} actual=${actual_sha}" >&2
  exit 21
fi
if [ -n "$(git status --porcelain=v1)" ]; then
  echo "FATAL: staging worktree is not clean" >&2
  git status --short >&2
  exit 22
fi
actual_config_sha="$(sha256sum "${CONFIG_PATH}" | awk '{print $1}')"
if [ "${actual_config_sha}" != "${EXPECTED_CONFIG_SHA256}" ]; then
  echo "FATAL: configuration SHA mismatch" >&2
  exit 23
fi

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
python - "${RUN_DIR}/historical_fr_mixture_${CASE_ID}_job_metadata.json" <<'PY'
import json, os, subprocess, sys
import cupy as cp
from KHz_filament.device import xp
if xp.__name__ != "cupy" or cp.cuda.runtime.getDeviceCount() < 1:
    raise SystemExit("FATAL: CUDA/CuPy backend is required")
device = cp.cuda.Device()
props = cp.cuda.runtime.getDeviceProperties(device.id)
name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
expected = os.environ["EXPECTED_GPU_MODEL"]
if name != expected:
    raise SystemExit(f"FATAL: GPU model mismatch expected={expected!r} actual={name!r}")
payload = {
  "schema": "khz_filament.historical_fr_mixture.job_metadata.v1",
  "case_id": os.environ["CASE_ID"],
  "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
  "expected_sha": os.environ["EXPECTED_GIT_SHA"],
  "actual_sha": subprocess.check_output(
      ["git", "-C", os.environ["REPO_DIR"], "rev-parse", "HEAD"], text=True
  ).strip(),
  "sha_match": True, "git_status_clean": True,
  "config_path": os.environ["CONFIG_PATH"], "config_sha256": os.environ["EXPECTED_CONFIG_SHA256"],
  "gpu_model": name, "gpu_total_memory_bytes": int(props["totalGlobalMem"]),
  "cpu_threads": int(os.environ["SLURM_CPUS_PER_TASK"]), "staging_directory": os.environ["REPO_DIR"],
}
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(payload, indent=2) + "\n")
PY

# Running from RUN_DIR keeps LUT/cache/output writes out of the immutable source tree.
python -c 'from KHz_filament.runner import run_from_file; import os; case=os.environ["CASE_ID"]; run_from_file(os.environ["CONFIG_PATH"], out_path=os.path.join(os.environ["RUN_DIR"], f"historical_fr_mixture_{case}.npz"), dtype="fp32")'
