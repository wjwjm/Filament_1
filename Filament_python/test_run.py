#!/usr/bin/env python3
"""CLI wrapper for running KHz filament simulations locally/HPC."""

from __future__ import annotations

import argparse
import os
import pathlib
import time


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KHz filamentation runner")
    p.add_argument("--cfg", type=str, default="khz_config.json", help="Path to config file (json/yaml/toml)")
    p.add_argument("--gpu", action="store_true", help="Use GPU (sets UPPE_USE_GPU=1)")
    p.add_argument("--threads", type=int, default=None, help="Set OMP/MKL/OPENBLAS thread count")
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp64"], help="Computation dtype")
    p.add_argument("--out", type=str, default="khzfil_out.npz", help="Output npz path")
    p.add_argument("--force-uppe", action="store_true", help="Force linear_model=uppe and disable factorized linear step")
    return p


def _setup_runtime_env(args: argparse.Namespace) -> None:
    if args.gpu:
        os.environ["UPPE_USE_GPU"] = "1"

    threads = args.threads
    if threads is None:
        slurm_threads = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_threads:
            try:
                threads = int(slurm_threads)
            except ValueError:
                threads = None

    if threads and threads > 0:
        n = str(threads)
        os.environ["OMP_NUM_THREADS"] = n
        os.environ["OPENBLAS_NUM_THREADS"] = n
        os.environ["MKL_NUM_THREADS"] = n
        os.environ["NUMEXPR_NUM_THREADS"] = n
        os.environ.setdefault("OMP_PROC_BIND", "close")
        os.environ.setdefault("OMP_PLACES", "cores")
        print(f"[threads] using {threads} threads")


def main() -> int:
    args = _build_parser().parse_args()
    _setup_runtime_env(args)

    from KHz_filament.cli import run_demo
    from KHz_filament.confio import load_all
    from KHz_filament.device import debug_backend

    print("[backend-debug]", debug_backend())

    cfg_path = pathlib.Path(args.cfg) if args.cfg else None
    if cfg_path and not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    grid = beam = prop = ion = heat = run = raman = None
    if cfg_path:
        grid, beam, prop, ion, heat, run, *maybe_raman = load_all(str(cfg_path))
        raman = maybe_raman[0] if maybe_raman else None

        if args.force_uppe:
            prop.linear_model = "uppe"
            prop.full_linear_factorize = False

        if not hasattr(prop, "progress_every_z"):
            prop.progress_every_z = 100
        if not hasattr(prop, "show_eta"):
            prop.show_eta = True

    t0 = time.perf_counter()
    run_kw = {"out_path": args.out, "dtype": args.dtype}
    if raman is not None:
        run_kw["raman"] = raman

    if all(v is not None for v in (grid, beam, prop, ion, heat, run)):
        run_demo(grid=grid, beam=beam, prop=prop, ion=ion, heat=heat, run=run, **run_kw)
    else:
        run_demo(**run_kw)

    print(f"[total] {time.perf_counter() - t0:6.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
