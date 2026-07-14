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
    p.add_argument("--mat-dir", type=str, default=None, help="If set, convert npz to mat in this directory")
    p.add_argument("--mat-name", type=str, default=None, help="Output mat file name (default: <out stem>.mat)")
    p.add_argument("--fig-dir", type=str, default=None, help="If set, write diagnostic PNGs and JSON summary to this directory")
    p.add_argument("--fig-select", type=str, default="all", help="Comma-separated figures: all, intensity, plasma, beam, energy, fwhm, rho_tz")
    p.add_argument("--fig-dpi", type=int, default=200, help="PNG resolution for Slurm/headless diagnostics")
    p.add_argument("--z-shift-cm", type=float, default=0.0, help="Manual plotted z-axis shift in cm")
    p.add_argument("--no-plots", action="store_true", help="Skip diagnostic PNG generation even if --fig-dir is set")
    p.add_argument("--remove-npz", action="store_true", help="Remove npz only after all enabled plots and MAT conversion succeed")
    p.add_argument("--verbose-backend", action="store_true", help="Print backend debug details")
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


def _validate_postprocess_args(args: argparse.Namespace) -> None:
    if args.fig_dpi <= 0:
        raise ValueError("--fig-dpi must be positive")
    if not args.no_plots and args.fig_dir is not None and not str(args.fig_dir).strip():
        raise ValueError("--fig-dir must not be empty")
    if args.remove_npz and not args.mat_dir:
        raise ValueError("--remove-npz requires --mat-dir so a successful MAT file remains available")


def _postprocess_output(args: argparse.Namespace) -> None:
    """Run ordered post-processing without deleting the source on any failure."""
    out_npz = pathlib.Path(args.out)
    if not out_npz.is_file():
        raise FileNotFoundError(f"Simulation did not produce expected NPZ: {out_npz}")

    if args.fig_dir and not args.no_plots:
        from plot_khzfil_out import generate_figures

        summary = generate_figures(
            npz_path=out_npz,
            figure_dir=args.fig_dir,
            selected_figures=args.fig_select,
            z_shift_cm=args.z_shift_cm,
            dpi=args.fig_dpi,
        )
        for name in summary["generated_figures"]:
            print(f"[figures] wrote: {pathlib.Path(args.fig_dir) / name}")
        print(f"[figures] summary: {summary['summary_path']}")

    if args.mat_dir:
        mat_dir = pathlib.Path(args.mat_dir)
        mat_name = args.mat_name or f"{out_npz.stem}.mat"
        mat_path = mat_dir / mat_name

        from npz2mat import convert_npz_to_mat

        convert_npz_to_mat(out_npz, mat_path)

    if args.remove_npz:
        # _validate_postprocess_args enforces MAT conversion as a prerequisite.
        out_npz.unlink()
        print(f"[postprocess] removed source NPZ after successful plots and MAT conversion: {out_npz}")


def main() -> int:
    args = _build_parser().parse_args()
    _validate_postprocess_args(args)
    _setup_runtime_env(args)

    from KHz_filament.cli import run_demo
    from KHz_filament.confio import load_all
    from KHz_filament.device import debug_backend

    if args.verbose_backend:
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

    _postprocess_output(args)

    print(f"[total] {time.perf_counter() - t0:6.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
