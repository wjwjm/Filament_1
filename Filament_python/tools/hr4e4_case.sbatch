#!/usr/bin/env bash
#SBATCH --job-name=hr4e4-limit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
set -euo pipefail
readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
: "${EXPECTED_GIT_SHA:?}" "${REPO_DIR:?}" "${CASE_DIR:?}" "${CASE_ID:?}"
cd "${REPO_DIR}"; test "$(git rev-parse HEAD)" = "${EXPECTED_GIT_SHA}"; test -z "$(git status --porcelain=v1)"
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh; conda activate Filament_python
export UPPE_USE_GPU=1 PYTHONPATH="${REPO_DIR}/Filament_python" PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX="${CASE_DIR}/pycache" CUPY_CACHE_DIR="${CASE_DIR}/cupy_cache" XDG_CACHE_HOME="${CASE_DIR}/xdg_cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p "${CASE_DIR}"
"${PYTHON}" -c 'from KHz_filament.device import debug_backend; assert debug_backend()["backend"] == "cupy", debug_backend()'
"${PYTHON}" "${REPO_DIR}/Filament_python/tools/run_hr4e4_case.py" --case-id "${CASE_ID}" --out-dir "${CASE_DIR}"
