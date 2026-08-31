#!/usr/bin/env bash
#SBATCH --job-name=hr4e1-dt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

set -euo pipefail

readonly FIXED_PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
: "${EXPECTED_GIT_SHA:?missing EXPECTED_GIT_SHA}"
: "${REPO_DIR:?missing REPO_DIR}"
: "${CASE_DIR:?missing CASE_DIR}"
: "${CASE_ID:?missing CASE_ID}"
: "${DT_US:?missing DT_US}"
: "${BENCHMARK:?missing BENCHMARK}"

cd "${REPO_DIR}"
test "$(git rev-parse HEAD)" = "${EXPECTED_GIT_SHA}"
test -z "$(git status --porcelain=v1)"
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python

export PYTHONPYCACHEPREFIX="${CASE_DIR}/pycache"
export CUPY_CACHE_DIR="${CASE_DIR}/cupy_cache"
export XDG_CACHE_HOME="${CASE_DIR}/xdg_cache"
export PYTHONPATH="${REPO_DIR}/Filament_python"
export UPPE_USE_GPU=1 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mkdir -p "${CASE_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "${CASE_DIR}/gpu_info.txt"
case_args=(--benchmark "${BENCHMARK}" --dt-us "${DT_US}" --out-dir "${CASE_DIR}" --case-id "${CASE_ID}" --dtype fp64)
if [[ "${BENCHMARK}" == E1-B ]]; then
    : "${E1B_SCREEN:?missing E1B_SCREEN}"
    : "${SCREEN_ID:?missing SCREEN_ID}"
    case_args+=(--e1b-screen "${E1B_SCREEN}" --screen-id "${SCREEN_ID}")
fi
"${FIXED_PYTHON}" "${REPO_DIR}/Filament_python/tools/run_hr4e1_timestep_case.py" "${case_args[@]}"
