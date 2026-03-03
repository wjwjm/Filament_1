#!/usr/bin/env python3             python test_run.py --cfg khz_config.json --gpu --dtype fp32
import os
import argparse
import time

# --------- CLI 参数 ---------
p = argparse.ArgumentParser(description="KHz filamentation runner")
p.add_argument("--cfg", type=str, default=None, help="Path to config file (json/yaml/toml)")
p.add_argument("--gpu", action="store_true", help="Use GPU (sets UPPE_USE_GPU=1)")
p.add_argument("--threads", type=int, default=None, help="Set OMP/MKL/OPENBLAS threads")
p.add_argument("--dtype", type=str, default="fp32", choices=["fp32","fp64"], help="Computation dtype")
p.add_argument("--out", type=str, default="khzfil_out.npz", help="Output npz path")
args = p.parse_args()

# --------- 环境变量（在导入 numpy/cupy 前设置） ---------
if args.gpu:
    os.environ["UPPE_USE_GPU"] = "1"   # 你的 device.py 会读取它并切到 CuPy,
# 线程绑定（与 SLURM --cpus-per-task 对齐）
if args.threads:
    n = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = n
    os.environ["OPENBLAS_NUM_THREADS"] = n
    os.environ["MKL_NUM_THREADS"] = n
    os.environ["NUMEXPR_NUM_THREADS"] = n
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")

# --------- 导入你的包 ---------
from KHz_filament.cli import run_demo
from KHz_filament.confio import load_all
from KHz_filament.device import xp, USE_GPU,debug_backend
print("[backend-debug]", debug_backend())
# --------- 加载配置并根据参数微调 ---------
if args.cfg:
    grid, beam, prop, ion, heat, run, *maybe_raman = load_all(args.cfg)
    raman = maybe_raman[0] if maybe_raman else None

    # 如果 --gpu：切到 3D 一次性线性步（GPU 上更快）
    if getattr(args, "force_uppe", False):
        try:
            prop.linear_model = "uppe"
        except Exception:
            pass
        # 3D 一次性 FFT（避免 factorized 的 Python 循环）
        try:
            prop.full_linear_factorize = False
        except Exception:
            pass

    # 进度打印（没有就忽略）
    try:
        if not hasattr(prop, "progress_every_z"):
            prop.progress_every_z = 100
        if not hasattr(prop, "show_eta"):
            prop.show_eta = True
    except Exception:
        pass
else:
    # 没有外部配置就用 run_demo 的默认参数
    grid = beam = prop = ion = heat = run = raman = None

# --------- 运行 ---------
t0 = time.perf_counter()
kw = dict(out_path=args.out)

# 把 dtype 与 raman（如果有）传进去；下面的 run_demo 需要支持 dtype 形参（见下一节补丁）
if raman is not None:
    kw["raman"] = raman

kw["dtype"] = args.dtype  # "fp32" | "fp64"


# ---- Device sanity print ----


if all(v is not None for v in (grid, beam, prop, ion, heat, run)):
    run_demo(grid=grid, beam=beam, prop=prop, ion=ion, heat=heat, run=run, **kw)
else:
    run_demo(**kw)

print(f"[total] {time.perf_counter() - t0:6.2f}s")
