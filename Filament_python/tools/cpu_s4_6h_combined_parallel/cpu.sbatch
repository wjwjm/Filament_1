#!/usr/bin/env bash
# CPU-S4-6H combined serial four-window qualification.  Submission remains a
# parent-controlled action; this file is only the audited batch entry.
#SBATCH --job-name=cpu-s46h-joint
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
: "${S46F_SOURCE:?}" "${S46F_REPORT:?}" "${S46F_STAGE_ROOT:?}" "${S46F_HPC_PLAN_SHA256:?}" "${S46F_EXECUTION_SHA256:?}" "${S46F_CANDIDATE:?}" "${CPU_S4_DATA_ROOT:?}" "${CPU_BOUNDARY_HISTORICAL_RECEIPT:?}"
source /publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_q1_20261014_110706/environment/venv/bin/activate
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UPPE_USE_GPU=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=48 NUMBA_CPU_NAME=generic NUMBA_THREADING_LAYER=workqueue
export PATH="$S46F_STAGE_ROOT/runtime_bin:$PATH"
export S43G2_SOURCE_ROOT="$S46F_SOURCE" S43F_SOURCE_ROOT="$S46F_SOURCE" S43C_SOURCE_ROOT="$S46F_SOURCE"
export CPU_S4_DATA_ROOT="$CPU_S4_DATA_ROOT" CPU_BOUNDARY_HISTORICAL_RECEIPT="$CPU_BOUNDARY_HISTORICAL_RECEIPT"
cd "$S46F_SOURCE/Filament_python"
exec srun --ntasks=1 --cpus-per-task=96 --cpu-bind=cores /publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_q1_20261014_110706/environment/venv/bin/python -s -B -m tools.cpu_s4_6h_combined_parallel.hpc --report "$S46F_REPORT" --candidate "$S46F_CANDIDATE"
