#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=320G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
Q3_ROOT="/publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_q3_20261015_01a1332e"
GIT_WRAPPER_DIR="/publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_p1_20261013_01a1332e/ops/bin"
export PATH="${GIT_WRAPPER_DIR}:${PATH}"
if [[ ! -x "${GIT_WRAPPER_DIR}/git" ]]; then
    echo "required Git wrapper is not executable: ${GIT_WRAPPER_DIR}/git" >&2
    exit 2
fi
RESOLVED_GIT="$(command -v git)"
if [[ "${RESOLVED_GIT}" != "${GIT_WRAPPER_DIR}/git" ]]; then
    echo "Git wrapper resolution mismatch: ${RESOLVED_GIT}" >&2
    exit 2
fi
"${RESOLVED_GIT}" --version
"${RESOLVED_GIT}" -C "${PWD}" status --porcelain=v1 >/dev/null
export PYTHONPATH="${PWD}/Filament_python"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 UPPE_USE_GPU=0
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=24 NUMBA_CPU_NAME=generic NUMBA_THREADING_LAYER=workqueue
export NUMBA_CACHE_DIR="${Q3_ROOT}/environment/numba-cache"
mkdir -p "${NUMBA_CACHE_DIR}"
exec /publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_q1_20261014_110706/environment/venv/bin/python -s -B -m tools.cpu_o1.q3_t0_real "$@"
