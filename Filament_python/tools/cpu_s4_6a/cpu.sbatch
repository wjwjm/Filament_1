#!/usr/bin/env bash
# Resource flags are supplied by the separately audited, budget-checked submitter.
set -euo pipefail
: "${S46A_SOURCE:?}" "${S46A_REPORT:?}" "${S46A_SIZE:?}" "${S46A_PLAN_SHA256:?}" "${S46A_WALLTIME_S:?}" "${S46A_STAGE_ROOT:?}" "${S46A_BUDGET_SHA256:?}"
source /publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_q1_20261014_110706/environment/venv/bin/activate
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UPPE_USE_GPU=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=48 NUMBA_CPU_NAME=generic NUMBA_THREADING_LAYER=workqueue
export PATH="$S46A_STAGE_ROOT/runtime_bin:$PATH"
export S43G2_SOURCE_ROOT="$S46A_SOURCE" S43F_SOURCE_ROOT="$S46A_SOURCE" S43C_SOURCE_ROOT="$S46A_SOURCE"
export CPU_S4_DATA_ROOT="$S46A_REPORT/external_data/input_bundle"
export CPU_BOUNDARY_HISTORICAL_RECEIPT="$S46A_REPORT/external_data/G2_BOUNDARY_RECEIPT.json"
cd "$S46A_SOURCE/Filament_python"
exec srun --ntasks=1 --cpus-per-task=96 --cpu-bind=cores /publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_q1_20261014_110706/environment/venv/bin/python -s -B -m tools.cpu_s4_6a.matrix --backend CPU --size "$S46A_SIZE" --report "$S46A_REPORT"
