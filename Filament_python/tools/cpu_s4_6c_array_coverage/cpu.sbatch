#!/usr/bin/env bash
set -euo pipefail
: "${S46C_SOURCE:?}" "${S46C_REPORT:?}" "${S46C_STAGE_ROOT:?}" "${S46C_PLAN_SHA256:?}"
source /publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_q1_20261014_110706/environment/venv/bin/activate
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UPPE_USE_GPU=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=48 NUMBA_CPU_NAME=generic NUMBA_THREADING_LAYER=workqueue
export NUMBA_CACHE_DIR="$S46C_REPORT/numba_cache"
export PATH="$S46C_STAGE_ROOT/runtime_bin:$PATH"
export S43G2_SOURCE_ROOT="$S46C_SOURCE" S43F_SOURCE_ROOT="$S46C_SOURCE" S43C_SOURCE_ROOT="$S46C_SOURCE"
export CPU_S4_DATA_ROOT=/publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_s46a_20261029_r2/report/external_data/input_bundle
export CPU_BOUNDARY_HISTORICAL_RECEIPT=/publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_s46a_20261029_r2/report/external_data/G2_BOUNDARY_RECEIPT.json
cd "$S46C_SOURCE/Filament_python"
exec srun --ntasks=1 --cpus-per-task=96 --cpu-bind=cores /publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_q1_20261014_110706/environment/venv/bin/python -s -B -m tools.cpu_s4_6c_array_coverage.node --report "$S46C_REPORT"
