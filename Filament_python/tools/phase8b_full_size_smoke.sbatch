#!/bin/bash
#SBATCH -J p8b-smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
# Memory is assigned by the N50 site policy per requested GPU (126 GB/GPU).
#SBATCH -t 00:30:00
#SBATCH -o phase8b-smoke-%j.out
#SBATCH -e phase8b-smoke-%j.err

set -euo pipefail
: "${PHASE8B_SMOKE_ROOT:?missing PHASE8B_SMOKE_ROOT}"
: "${PHASE8B_SMOKE_CASE:?missing PHASE8B_SMOKE_CASE}"
: "${PHASE8B_SMOKE_CONFIG:?missing PHASE8B_SMOKE_CONFIG}"

source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export UPPE_USE_GPU=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PHASE8B_SCHEDULER_MEMORY_MB="$(
  scontrol show job "${SLURM_JOB_ID}" \
    | sed -n 's/.*AllocTRES=.*mem=\([0-9][0-9]*\)M.*/\1/p' \
    | head -n 1
)"

cd "${PHASE8B_SMOKE_ROOT}/source/Filament_python"
python tools/run_phase8b_full_size_smoke.py \
  --config "${PHASE8B_SMOKE_CONFIG}" \
  --out-dir "${PHASE8B_SMOKE_ROOT}/cases/${PHASE8B_SMOKE_CASE}" \
  --case "${PHASE8B_SMOKE_CASE}" \
  --steps 20 \
  --production-steps 15000 \
  --production-slurm-limit-s 28800 \
  --legacy-full-runtime-s 7488
