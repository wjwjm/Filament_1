#!/bin/bash
#SBATCH --job-name=phase8b_job1_on
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

set -euo pipefail

: "${RUN_SOURCE:?RUN_SOURCE is required}"
: "${EXPECTED_GIT_SHA:?EXPECTED_GIT_SHA is required}"
: "${EXPECTED_CONFIG_SHA256:?EXPECTED_CONFIG_SHA256 is required}"
: "${MIN_GPU_MEMORY_BYTES:?MIN_GPU_MEMORY_BYTES is required}"

if [[ ! -f "${RUN_SOURCE}/BASE_GIT_SHA" ]]; then
  echo "[fatal] missing ${RUN_SOURCE}/BASE_GIT_SHA"
  exit 3
fi
if [[ "$(tr -d '[:space:]' < "${RUN_SOURCE}/BASE_GIT_SHA")" != "${EXPECTED_GIT_SHA}" ]]; then
  echo "[fatal] source Git SHA does not match the locked execution SHA"
  exit 3
fi

cd "${RUN_SOURCE}/Filament_python"
CFG="${CFG:-configs/isaacs_raman_closure/120fs_talebpour_isaacs_full_operator_on.json}"
if [[ "$(sha256sum "${CFG}" | awk '{print $1}')" != "${EXPECTED_CONFIG_SHA256}" ]]; then
  echo "[fatal] formal Job 1 configuration SHA256 mismatch"
  exit 3
fi

source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export UPPE_USE_GPU=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# sub.sh intentionally begins in SLURM_SUBMIT_DIR.  The batch script itself
# lives next to the run metadata rather than inside the archived source, so
# pin that directory to the verified source tree before delegating to it.
export SLURM_SUBMIT_DIR="${RUN_SOURCE}/Filament_python"

python - <<'PY'
import os
import cupy as cp
from KHz_filament.device import xp

if os.environ.get("UPPE_USE_GPU") != "1":
    raise SystemExit("UPPE_USE_GPU must be 1")
if xp.__name__ != "cupy":
    raise SystemExit(f"NumPy fallback forbidden; backend is {xp.__name__}")
if cp.cuda.runtime.getDeviceCount() < 1:
    raise SystemExit("no visible CUDA device")
device = cp.cuda.Device()
props = cp.cuda.runtime.getDeviceProperties(device.id)
name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
total = int(props["totalGlobalMem"])
minimum = int(os.environ["MIN_GPU_MEMORY_BYTES"])
print(f"[phase8b-r2] backend=cupy device={device.id}:{name} total_memory_bytes={total}")
if total < minimum:
    raise SystemExit(f"GPU memory {total} < required smoke-node memory {minimum}")
PY

exec bash sub.sh
