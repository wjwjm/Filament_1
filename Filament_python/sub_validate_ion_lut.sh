#!/bin/bash
#SBATCH -J ionlut_check
#SBATCH -p gpu
#SBATCH -x g0601,g0605
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:20:00
#SBATCH -o ionlut_check.%j.out
#SBATCH -e ionlut_check.%j.err

set -euo pipefail

# 进入提交目录（Slurm 推荐），兜底到脚本目录
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

CFG="${CFG:-khz_config.json}"
OUTDIR="${OUTDIR:-runs/ion_lut_check_${SLURM_JOB_ID:-manual}}"
BACKEND="${BACKEND:-numpy}"
NUM_POINTS="${NUM_POINTS:-3000}"

# onset 自动窗口参数（W_reference）
ONSET_W_MIN="${ONSET_W_MIN:-1e4}"
ONSET_W_MAX="${ONSET_W_MAX:-1e10}"

# 可选手动 onset 强度窗口；若同时给了 I_MIN+I_MAX，将优先使用手动窗口
ONSET_I_MIN="${ONSET_I_MIN:-}"
ONSET_I_MAX="${ONSET_I_MAX:-}"

# 可选物种筛选：例如 SPECIES="N2 O2"
SPECIES="${SPECIES:-}"

if [[ ! -f "$CFG" ]]; then
  echo "[fatal] config not found: $CFG"
  exit 3
fi

if [[ ! -f "../validate_ion_lut.py" ]]; then
  echo "[fatal] validate_ion_lut.py not found at ../validate_ion_lut.py"
  exit 3
fi

module load miniforge/25.3.0-3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Filament_python

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export UPPE_USE_GPU="${UPPE_USE_GPU:-1}"
export PYTHONUNBUFFERED=1

# 与作业线程数对齐
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
fi

echo "[ionlut-check] CFG=$CFG"
echo "[ionlut-check] OUTDIR=$OUTDIR"
echo "[ionlut-check] BACKEND=$BACKEND"
echo "[ionlut-check] NUM_POINTS=$NUM_POINTS"
echo "[ionlut-check] ONSET_W_MIN=$ONSET_W_MIN ONSET_W_MAX=$ONSET_W_MAX"
if [[ -n "$ONSET_I_MIN" || -n "$ONSET_I_MAX" ]]; then
  echo "[ionlut-check] MANUAL onset I-window: [$ONSET_I_MIN, $ONSET_I_MAX]"
fi
if [[ -n "$SPECIES" ]]; then
  echo "[ionlut-check] SPECIES=$SPECIES"
fi

CMD=(
  python ../validate_ion_lut.py
  --config "$CFG"
  --outdir "$OUTDIR"
  --backend "$BACKEND"
  --num-points "$NUM_POINTS"
  --onset-W-min "$ONSET_W_MIN"
  --onset-W-max "$ONSET_W_MAX"
)

if [[ -n "$ONSET_I_MIN" && -n "$ONSET_I_MAX" ]]; then
  CMD+=(--onset-I-min "$ONSET_I_MIN" --onset-I-max "$ONSET_I_MAX")
fi

if [[ -n "$SPECIES" ]]; then
  # shellcheck disable=SC2206
  SP_ARR=($SPECIES)
  CMD+=(--species "${SP_ARR[@]}")
fi

"${CMD[@]}"

echo "[ionlut-check] done"
echo "[ionlut-check] summary: $OUTDIR/summary.json"
