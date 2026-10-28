#!/usr/bin/env bash
# Strict 1GPU/8CPU resource flags, g0603 exclusion, supplied by the submitter.
set -euo pipefail
: "${S46A_SOURCE:?}" "${S46A_REPORT:?}" "${S46A_PLAN_SHA256:?}" "${S46A_WALLTIME_S:?}" "${S46A_BUDGET_SHA256:?}"
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UPPE_USE_GPU=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 BLIS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export S43G2_SOURCE_ROOT="$S46A_SOURCE" S43F_SOURCE_ROOT="$S46A_SOURCE" S43C_SOURCE_ROOT="$S46A_SOURCE"
export CPU_BOUNDARY_HISTORICAL_RECEIPT="$S46A_REPORT/external_data/G2_BOUNDARY_RECEIPT.json"
cd "$S46A_SOURCE/Filament_python"
exec srun --ntasks=1 --gpus-per-task=1 --cpus-per-task=8 --cpu-bind=cores /data/home/scvi806/.conda/envs/Filament_python/bin/python -s -B -m tools.cpu_s4_6a.matrix --backend GPU --size both --report "$S46A_REPORT"
