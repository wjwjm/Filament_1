#!/usr/bin/env python3
"""Convert simulation npz output to MATLAB .mat format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io


def convert_npz_to_mat(npz_path: str | Path, mat_path: str | Path, remove_npz: bool = False) -> Path:
    npz_p = Path(npz_path)
    mat_p = Path(mat_path)

    if not npz_p.exists():
        raise FileNotFoundError(f"npz file not found: {npz_p}")

    mat_p.parent.mkdir(parents=True, exist_ok=True)

    with np.load(npz_p, allow_pickle=False) as data:
        dic = {k: data[k] for k in data.files}
    scipy.io.savemat(mat_p, dic)
    print(f"[npz2mat] wrote: {mat_p}")

    if remove_npz:
        npz_p.unlink()
        print(f"[npz2mat] removed source: {npz_p}")

    return mat_p


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert .npz file to MATLAB .mat")
    p.add_argument("--npz", required=True, help="Input npz file path")
    p.add_argument("--mat", required=True, help="Output mat file path")
    p.add_argument("--remove-npz", action="store_true", help="Delete source npz after successful conversion")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    convert_npz_to_mat(args.npz, args.mat, remove_npz=args.remove_npz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
