#!/usr/bin/env bash
#SBATCH --job-name=hr4e1-post
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

set -euo pipefail

readonly FIXED_PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
: "${EXPECTED_GIT_SHA:?missing EXPECTED_GIT_SHA}"
: "${REPO_DIR:?missing REPO_DIR}"
: "${POST_DIR:?missing POST_DIR}"
: "${RUNNER_OUTPUT:?missing RUNNER_OUTPUT}"
: "${BASE_CONFIG:?missing BASE_CONFIG}"

cd "${REPO_DIR}"
test "$(git rev-parse HEAD)" = "${EXPECTED_GIT_SHA}"
test -z "$(git status --porcelain=v1)"
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python

export PYTHONPYCACHEPREFIX="${POST_DIR}/pycache"
export CUPY_CACHE_DIR="${POST_DIR}/cupy_cache"
export XDG_CACHE_HOME="${POST_DIR}/xdg_cache"
export PYTHONPATH="${REPO_DIR}/Filament_python"
export UPPE_USE_GPU=1 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mkdir -p "${POST_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "${POST_DIR}/gpu_info.txt"
"${FIXED_PYTHON}" "${REPO_DIR}/Filament_python/tools/generate_hr4e1_post_reference.py" \
    --base-config "${BASE_CONFIG}" --out-dir "${POST_DIR}" \
    --runner-output "${RUNNER_OUTPUT}" --dtype fp64
