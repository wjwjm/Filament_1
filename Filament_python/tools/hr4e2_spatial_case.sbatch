#!/usr/bin/env bash
#SBATCH --job-name=hr4e2-grid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
set -euo pipefail
readonly FIXED_PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
: "${EXPECTED_GIT_SHA:?}"; : "${REPO_DIR:?}"; : "${CASE_DIR:?}"; : "${CASE_ID:?}"; : "${FAMILY:?}"; : "${DX_UM:?}"; : "${DT_US:?}"
cd "${REPO_DIR}"
test "$(git rev-parse HEAD)" = "${EXPECTED_GIT_SHA}"
test -z "$(git status --porcelain=v1)"
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export PYTHONPYCACHEPREFIX="${CASE_DIR}/pycache" CUPY_CACHE_DIR="${CASE_DIR}/cupy_cache" XDG_CACHE_HOME="${CASE_DIR}/xdg_cache" PYTHONPATH="${REPO_DIR}/Filament_python"
export UPPE_USE_GPU=1 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p "${CASE_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "${CASE_DIR}/gpu_info.txt"
"${FIXED_PYTHON}" "${REPO_DIR}/Filament_python/tools/run_hr4e2_spatial_case.py" --family "${FAMILY}" --dx-um "${DX_UM}" --dt-us "${DT_US}" --out-dir "${CASE_DIR}" --case-id "${CASE_ID}"
