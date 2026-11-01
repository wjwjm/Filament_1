#!/usr/bin/env bash
# CPU-S4-6F C full-size capture/mechanism job; submission remains parent-controlled.
#SBATCH --job-name=cpu-s46f-c-mech
#SBATCH --account=bscc-t
#SBATCH --partition=v6_384
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail

: "${S46F_C_FULL_SOURCE:?}" "${S46F_C_FULL_REPORT:?}" "${S46F_C_FULL_STAGE_ROOT:?}"
: "${S46F_C_FULL_EXECUTION_SHA256:?}" "${S46F_C_FULL_HPC_PLAN_SHA256:?}" "${CPU_S4_DATA_ROOT:?}"

source /publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_q1_20261014_110706/environment/venv/bin/activate
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export UPPE_USE_GPU=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=48
export NUMBA_CPU_NAME=generic
export NUMBA_THREADING_LAYER=workqueue
export S46F_C_FULL_EXECUTION_SHA256
export S46F_C_FULL_HPC_PLAN_SHA256
export PATH="$S46F_C_FULL_STAGE_ROOT/runtime_bin:$PATH"
export S43G2_SOURCE_ROOT="$S46F_C_FULL_SOURCE" S43F_SOURCE_ROOT="$S46F_C_FULL_SOURCE" S43C_SOURCE_ROOT="$S46F_C_FULL_SOURCE"
cd "$S46F_C_FULL_SOURCE/Filament_python"

exec srun --ntasks=1 --cpus-per-task=96 --cpu-bind=cores /publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_q1_20261014_110706/environment/venv/bin/python \
  -s -B -m tools.cpu_s4_6f_fullsize_parallel.c_fullsize_hpc \
  --report "$S46F_C_FULL_REPORT"
