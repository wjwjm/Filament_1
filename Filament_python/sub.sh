#!/bin/bash
#SBATCH -p gpu
#SBATCH -x g0601,g0605

set -euo pipefail

# 进入脚本所在目录，避免 sbatch 从其它 cwd 提交时找不到配置文件
cd "$(dirname "$0")"

# 可按需覆盖：CFG/OUT/DTYPE
CFG="${CFG:-khz_config.json}"
OUT="${OUT:-khzfil_out.npz}"
DTYPE="${DTYPE:-fp32}"

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
print("[env] UPPE_USE_GPU =", os.environ.get("UPPE_USE_GPU"))
print("[env] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("[env] SLURM_CPUS_PER_TASK =", os.environ.get("SLURM_CPUS_PER_TASK"))
try:
    import cupy as cp
    n = cp.cuda.runtime.getDeviceCount()
    print("[precheck] CuPy OK. device_count =", n)
    if n > 0:
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        print(f"[precheck] using device {dev.id}: {name}")
    else:
        print("[precheck] NO GPU visible. exit.")
        sys.exit(2)
except Exception as e:
    print("[precheck] CuPy import/driver FAILED:", e)
    sys.exit(1)
PY

python test_run.py --cfg "$CFG" --gpu --dtype "$DTYPE" --out "$OUT"
