#!/usr/bin/env bash
#SBATCH --job-name=cpu-s46e-c14
#SBATCH --account=bscc-t
#SBATCH --partition=v6_384
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -euo pipefail
: "${S46E_SOURCE:?}" "${S46E_REPORT:?}" "${S46E_STAGE_ROOT:?}" "${S46E_C14_HPC_PLAN_SHA256:?}" "${S46E_C14_EXECUTION_SHA256:?}"
source /publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_q1_20261014_110706/environment/venv/bin/activate
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UPPE_USE_GPU=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=48 NUMBA_CPU_NAME=generic NUMBA_THREADING_LAYER=workqueue
export PATH="$S46E_STAGE_ROOT/runtime_bin:$PATH"
export S43G2_SOURCE_ROOT="$S46E_SOURCE" S43F_SOURCE_ROOT="$S46E_SOURCE" S43C_SOURCE_ROOT="$S46E_SOURCE"
export CPU_S4_DATA_ROOT=/publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_s46a_20261029_r2/report/external_data/input_bundle
export CPU_BOUNDARY_HISTORICAL_RECEIPT=/publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_s46a_20261029_r2/report/external_data/G2_BOUNDARY_RECEIPT.json
cd "$S46E_SOURCE/Filament_python"
exec srun --ntasks=1 --cpus-per-task=96 --cpu-bind=cores /publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_q1_20261014_110706/environment/venv/bin/python -s -B -m tools.cpu_s4_6e_array_propagation.c_formal14_controller --report "$S46E_REPORT"
