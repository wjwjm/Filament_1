from __future__ import annotations
#通过环境变量 UPPE_USE_GPU=1 自动切换 NumPy/CuPy，统一为 xp 接口。

#提供 to_cpu()、as_xp() 实用函数，便于 I/O 和后处理。
import os, numpy as np
xp = np
USE_GPU = False
cp = None
REASON = "UPPE_USE_GPU unset"

if os.environ.get("UPPE_USE_GPU", "0") == "1":
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            xp = cp
            USE_GPU = True
    except Exception:
        USE_GPU = False
        xp = np

def debug_backend():
    return {"USE_GPU": USE_GPU, "backend": "cupy" if USE_GPU else "numpy", "reason": REASON}

def gpu_hard_gc():
    """强制释放 CuPy 内存池（安全可反复调用）。"""
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

def as_xp(a, dtype=None):
    return xp.asarray(a, dtype=dtype)

def to_cpu(a):
    if USE_GPU and cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)
