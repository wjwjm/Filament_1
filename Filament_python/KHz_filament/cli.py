from __future__ import annotations

import argparse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KHz filament CLI")
    parser.add_argument("cfg", nargs="?", default=None, help="Config path. If omitted, run built-in demo defaults.")
    parser.add_argument("--out", default="khzfil_out.npz", help="Output npz path.")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp64"], help="Computation dtype.")
    return parser


def run_demo(*args, **kwargs):
    from .runner import run_demo as _run_demo

    return _run_demo(*args, **kwargs)


def run_from_file(*args, **kwargs):
    from .runner import run_from_file as _run_from_file

    return _run_from_file(*args, **kwargs)


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cfg:
        run_from_file(args.cfg, out_path=args.out, dtype=args.dtype)
    else:
        run_demo(out_path=args.out, dtype=args.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
