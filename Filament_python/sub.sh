#!/bin/bash
#SBATCH -x g0601,g0605
module load miniforge/25.3.0-3
source activate Filament_python

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export UPPE_USE_GPU=1
export PYTHONUNBUFFERED=1

python - <<'PY'
import os, sys
print("[env] UPPE_USE_GPU =", os.environ.get("UPPE_USE_GPU"))
print("[env] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
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
        print("[precheck] NO GPU visible. Will fallback to CPU.")
except Exception as e:
    print("[precheck] CuPy import/driver FAILED:", e)
    sys.exit(1)  
PY

python test_run.py --cfg khz_config.json --gpu --dtype fp32
