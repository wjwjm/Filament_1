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
CONVERT_TO_MAT="${CONVERT_TO_MAT:-1}"
MAT_DIR="${MAT_DIR:-matlab保存数据}"
MAT_NAME="${MAT_NAME:-}"
REMOVE_NPZ="${REMOVE_NPZ:-1}"
GENERATE_FIGURES="${GENERATE_FIGURES:-1}"
FIG_DIR="${FIG_DIR:-figures}"
FIG_SELECT="${FIG_SELECT:-all}"
FIG_DPI="${FIG_DPI:-200}"
Z_SHIFT_CM="${Z_SHIFT_CM:-0}"

if [[ ! -f "$CFG" ]]; then
  echo "[fatal] config not found: $CFG"
  exit 3
fi

if [[ ! -f "test_run.py" ]]; then
  echo "[fatal] test_run.py not found in $(pwd)"
  exit 3
fi

if [[ "$REMOVE_NPZ" == "1" && "$CONVERT_TO_MAT" != "1" ]]; then
  echo "[fatal] REMOVE_NPZ=1 requires CONVERT_TO_MAT=1; the raw NPZ must not be deleted when no MAT exists."
  exit 3
fi

# Use cluster miniforge directly, no module required
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export UPPE_USE_GPU=1
export PYTHONUNBUFFERED=1

# Fail before the long propagation if optional post-processing dependencies are absent.
python - "$CONVERT_TO_MAT" "$GENERATE_FIGURES" <<'PY'
import sys

convert_to_mat, generate_figures = sys.argv[1:]
checks = []
if convert_to_mat == "1":
    checks.append(("scipy", "CONVERT_TO_MAT=1"))
if generate_figures == "1":
    checks.append(("matplotlib", "GENERATE_FIGURES=1"))

for module, reason in checks:
    try:
        __import__(module)
    except Exception as exc:
        print(f"[预检] {reason} requires Python package {module!r}: {exc}")
        raise SystemExit(4)
    print(f"[预检] {module} available ({reason})")
PY

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
if [[ "$GENERATE_FIGURES" == "1" ]]; then
  CMD+=(--fig-dir "$FIG_DIR" --fig-select "$FIG_SELECT" --fig-dpi "$FIG_DPI" --z-shift-cm "$Z_SHIFT_CM")
else
  CMD+=(--no-plots)
fi
if [[ "$CONVERT_TO_MAT" == "1" ]]; then
  CMD+=(--mat-dir "$MAT_DIR")
  if [[ -n "$MAT_NAME" ]]; then
    CMD+=(--mat-name "$MAT_NAME")
  fi
  if [[ "$REMOVE_NPZ" == "1" ]]; then
    CMD+=(--remove-npz)
  fi
fi
"${CMD[@]}"
