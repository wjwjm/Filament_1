#!/bin/bash
#SBATCH -p gpu

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" 

CFG="${CFG:?CFG is required}"
STAGE_SPEC="${STAGE_SPEC:?STAGE_SPEC is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"

source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python

export UPPE_USE_GPU=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

mkdir -p "$OUT_DIR"
python tools/run_vacuum_focus_validation.py --config "$CFG" --stage-spec "$STAGE_SPEC" --out-dir "$OUT_DIR" --gpu
python tools/analyze_vacuum_focus.py --out-dir "$OUT_DIR"
