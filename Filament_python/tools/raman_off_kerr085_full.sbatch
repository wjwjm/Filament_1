#!/usr/bin/env bash
#SBATCH --job-name=roff-kerr085
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

set -euo pipefail
: "${EXPECTED_GIT_SHA:?missing EXPECTED_GIT_SHA}"
: "${REPO_DIR:?missing REPO_DIR}"
: "${RUN_DIR:?missing RUN_DIR}"
: "${CONFIG_PATH:?missing CONFIG_PATH}"
: "${EXPECTED_CONFIG_SHA256:?missing EXPECTED_CONFIG_SHA256}"
: "${EXPECTED_GPU_MODEL:?missing EXPECTED_GPU_MODEL}"

cd "${REPO_DIR}"
actual_sha="$(git rev-parse HEAD)"
[[ "${actual_sha}" == "${EXPECTED_GIT_SHA}" ]] || { echo "FATAL: SHA mismatch" >&2; exit 21; }
if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "FATAL: staging worktree is not clean" >&2
  git status --short >&2
  exit 22
fi
actual_config_sha="$(sha256sum "${CONFIG_PATH}" | awk '{print $1}')"
[[ "${actual_config_sha}" == "${EXPECTED_CONFIG_SHA256}" ]] || { echo "FATAL: configuration SHA mismatch" >&2; exit 23; }

mkdir -p "${RUN_DIR}"
export PYTHONPYCACHEPREFIX="${RUN_DIR}/pycache"
export CUPY_CACHE_DIR="${RUN_DIR}/cupy_cache"
export XDG_CACHE_HOME="${RUN_DIR}/xdg_cache"
export PYTHONPATH="${REPO_DIR}/Filament_python"
export CUDA_DEVICE_ORDER=PCI_BUS_ID UPPE_USE_GPU=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python

META="${RUN_DIR}/raman_off_kerr085_job_metadata.json"
OUT="${RUN_DIR}/raman_off_kerr085.npz"
cd "${RUN_DIR}"
python - "${META}" <<'PY'
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
    "schema": "khz_filament.raman_off_kerr085.job_metadata.v1",
    "case_id": "raman_off_kerr085", "status": "running", "exit_code": 0,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "execution_git_sha": subprocess.check_output(
        ["git", "-C", os.environ["REPO_DIR"], "rev-parse", "HEAD"], text=True).strip(),
    "expected_git_sha": os.environ["EXPECTED_GIT_SHA"],
    "config_path": os.environ["CONFIG_PATH"],
    "config_sha256": os.environ["EXPECTED_CONFIG_SHA256"],
    "gpu_model": name, "gpu_total_memory_bytes": int(props["totalGlobalMem"]),
    "cpu_threads": int(os.environ["SLURM_CPUS_PER_TASK"]),
    "diagnostic_schema": "khz_filament.propagation_observability.v1",
}
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(payload, indent=2) + "\n")
PY

set +e
python -c 'from KHz_filament.runner import run_from_file; import os; run_from_file(os.environ["CONFIG_PATH"], out_path=os.path.join(os.environ["RUN_DIR"], "raman_off_kerr085.npz"), dtype="fp32")'
rc=$?
set -e
RUN_STATUS="$([[ ${rc} -eq 0 ]] && echo completed || echo failed)" RUN_RC="${rc}" python - "${META}" <<'PY'
import json, os, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
d["status"] = os.environ["RUN_STATUS"]
d["exit_code"] = int(os.environ["RUN_RC"])
open(p, "w", encoding="utf-8").write(json.dumps(d, indent=2) + "\n")
PY
exit "${rc}"
