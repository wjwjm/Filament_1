#!/bin/bash
set -euo pipefail

STAGE_DIR="${STAGE_DIR:?STAGE_DIR is required}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
unset UPPE_USE_GPU CUDA_VISIBLE_DEVICES

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

for result in "$STAGE_DIR/cases/profile_g_120/result.npz" "$STAGE_DIR/cases/profile_ft90_120/result.npz"; do
  [[ -f "$result" ]] || { echo "[fatal] missing result: $result"; exit 3; }
done

python compare_khzfil_outputs.py \
  --inputs "$STAGE_DIR/cases/profile_g_120/result.npz" "$STAGE_DIR/cases/profile_ft90_120/result.npz" \
  --labels "Gaussian, 120 fs" "FT90, 120 fs" \
  --case-ids profile_g_120 profile_ft90_120 \
  --out-dir "$STAGE_DIR/comparison" \
  --stage-spec "$STAGE_DIR/stage_spec_snapshot.json"
python finalize_transverse_profile_validation.py --stage-dir "$STAGE_DIR"
