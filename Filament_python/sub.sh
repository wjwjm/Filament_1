#!/bin/bash
#SBATCH -p gpu
#SBATCH -x g0601,g0605

set -euo pipefail

# 进入提交目录（Slurm 批处理推荐），兜底到脚本目录
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# 可按需覆盖：CFG/OUT/DTYPE
CFG="${CFG:-khz_config.json}"
OUT="${OUT:-khzfil_out.npz}"
DTYPE="${DTYPE:-fp32}"
MAT_DIR="${MAT_DIR:-}"
MAT_NAME="${MAT_NAME:-}"
REMOVE_NPZ="${REMOVE_NPZ:-1}"

if [[ ! -f "$CFG" ]]; then
  echo "[fatal] config not found: $CFG"
  exit 3
fi

if [[ ! -f "test_run.py" ]]; then
  echo "[fatal] test_run.py not found in $(pwd)"
  exit 3
fi

module load miniforge/25.3.0-3

# 兼容非交互 shell 的 conda 激活方式
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Filament_python

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export UPPE_USE_GPU=1
export PYTHONUNBUFFERED=1

# 与作业申请线程数对齐（若设置了 --cpus-per-task）
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
  export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
fi

python - <<'PY'
import os, sys
try:
    import cupy as cp
    n = cp.cuda.runtime.getDeviceCount()
    if n > 0:
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        print(
            f"[预检] 运行环境: UPPE_USE_GPU={os.environ.get('UPPE_USE_GPU')} | "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} | "
            f"SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK')} | "
            f"device_count={n} | using={dev.id}:{name}"
        )
    else:
        print("[预检] 未检测到可见GPU，任务终止。")
        sys.exit(2)
except Exception as e:
    print(f"[预检] CuPy/驱动初始化失败: {e}")
    sys.exit(1)
PY

CMD=(python test_run.py --cfg "$CFG" --gpu --dtype "$DTYPE" --out "$OUT")
if [[ -n "$MAT_DIR" ]]; then
  CMD+=(--mat-dir "$MAT_DIR")
  if [[ -n "$MAT_NAME" ]]; then
    CMD+=(--mat-name "$MAT_NAME")
  fi
  if [[ "$REMOVE_NPZ" == "1" ]]; then
    CMD+=(--remove-npz)
  fi
fi
"${CMD[@]}"
