#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=320G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
export PATH="/publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_p1_20261013_01a1332e/ops/bin:${PATH}"
export PYTHONPATH="${PWD}/Filament_python"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 UPPE_USE_GPU=0
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=24 NUMBA_CPU_NAME=generic NUMBA_THREADING_LAYER=workqueue
export NUMBA_CACHE_DIR="/publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_q1_20261014_110706/environment/numba-cache"
exec /publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_q1_20261014_110706/environment/venv/bin/python -s -B -m tools.cpu_o1.q1_t0_real "$@"
