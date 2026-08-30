"""Tiny local HR-3C-B streaming benchmark; never allocates production state."""

from __future__ import annotations

import argparse
import json
import tempfile

import numpy as np

from KHz_filament.config import HeatConfig
from KHz_filament.device import debug_backend
from KHz_filament.grids import make_axes
from KHz_filament.slow_state_pingpong import PingPongSlowStateStore, diffuse_current_to_next


def _parse_batch_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes or min(sizes) <= 0:
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    return sizes


def _fill_current(store, axes) -> None:
    x, y = np.asarray(axes.x), np.asarray(axes.y)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    profile = -1.0e-4 * np.exp(-(xx**2 + yy**2) / (0.2 * max(abs(x.max()), abs(y.max())))**2)
    for index in range(store.n_intervals):
        store.update_current_interval(index, profile * (1.0 + 0.01 * index))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervals", type=int, default=32)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default=(1, 2, 4, 8, 16, 32))
    args = parser.parse_args()
    if args.intervals <= 0 or args.grid <= 2:
        raise ValueError("--intervals must be positive and --grid must exceed 2")

    axes = make_axes(args.grid, args.grid, 8, 4e-3, 4e-3, 80e-15)
    heat = HeatConfig()
    rows = []
    with tempfile.TemporaryDirectory(prefix="hr3c_benchmark_") as directory:
        for batch_size in args.batch_sizes:
            store = PingPongSlowStateStore(
                output_path=f"{directory}/batch_{batch_size}.npz",
                n_intervals=args.intervals, shape=(args.grid, args.grid), dtype=np.float32,
            )
            try:
                _fill_current(store, axes)
                summary = diffuse_current_to_next(
                    store, kperp2=axes.kperp2, D_th=heat.D_th, f_rep=heat.f_rep,
                    batch_intervals=batch_size,
                )
                walltime = float(summary["walltime_s"])
                rows.append({
                    "batch_intervals": batch_size,
                    "walltime_s": walltime,
                    "throughput_MiB_s": (float(summary["bytes_read"]) / 1024**2) / walltime,
                    "n_batches": int(summary["n_batches"]),
                    "complete": bool(summary["complete"]),
                })
            finally:
                store.close()
    print(json.dumps({"backend": debug_backend(), "intervals": args.intervals, "grid": args.grid, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
