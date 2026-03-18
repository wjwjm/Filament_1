from __future__ import annotations
import math
import time
import numpy as np
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.constants import c0
from KHz_filament.ionization import (
    cycle_average_ppt_talebpour_legacy_from_I,
    cycle_average_ppt_talebpour_full_from_I,
    build_rate_table,
    eval_rate_from_table,
)


def tcall(fn, *args, repeat=3, **kwargs):
    best = 1e99
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    lam0 = 800e-9
    n0 = 1.00027
    omega0 = 2.0 * math.pi * c0 / lam0

    I_scalar = 3e14
    I_block = np.logspace(12, 15, 64 * 64).reshape(64, 64)
    I_time = np.logspace(12, 15, 32 * 64 * 64).reshape(32, 64, 64)

    taleb_kw = dict(Ip_eV=15.6, Zeff=0.9, l=0, m=0)
    ref_opts = {"cycle_avg_samples_ref": 64, "popruzhenko_sum_tol": 1e-6, "popruzhenko_max_terms": 256}
    table = build_rate_table(
        model_name="ppt_talebpour_i_full_reference",
        species_params={"name": "N2", **taleb_kw},
        omega0_SI=omega0,
        n0=n0,
        I_min_SI=1e8,
        I_max_SI=1e19,
        n_samples=3000,
        spacing="log",
        reference_opts=ref_opts,
    )

    t_legacy_scalar = tcall(cycle_average_ppt_talebpour_legacy_from_I, I_scalar, n0=n0, samples=16, **taleb_kw)
    t_ref_scalar = tcall(cycle_average_ppt_talebpour_full_from_I, I_scalar, n0=n0, omega0_SI=omega0, samples=64, **taleb_kw)
    t_lut_scalar = tcall(eval_rate_from_table, I_scalar, table, "loglog")

    t_legacy_block = tcall(cycle_average_ppt_talebpour_legacy_from_I, I_block, n0=n0, samples=16, **taleb_kw)
    t_ref_block = tcall(cycle_average_ppt_talebpour_full_from_I, I_block, n0=n0, omega0_SI=omega0, samples=64, **taleb_kw)
    t_lut_block = tcall(eval_rate_from_table, I_block, table, "loglog")

    t_legacy_time = tcall(cycle_average_ppt_talebpour_legacy_from_I, I_time, n0=n0, samples=16, **taleb_kw, repeat=1)
    t_ref_time = tcall(cycle_average_ppt_talebpour_full_from_I, I_time, n0=n0, omega0_SI=omega0, samples=64, **taleb_kw, repeat=1)
    t_lut_time = tcall(eval_rate_from_table, I_time, table, "loglog", repeat=1)

    print("=== Ionization rate evaluator benchmark ===")
    print(f"single-point legacy={t_legacy_scalar:.6e}s reference={t_ref_scalar:.6e}s lut={t_lut_scalar:.6e}s")
    print(f"small-tensor legacy={t_legacy_block:.6e}s reference={t_ref_block:.6e}s lut={t_lut_block:.6e}s")
    print(f"full-time-estimate legacy={t_legacy_time:.6e}s reference={t_ref_time:.6e}s lut={t_lut_time:.6e}s")


if __name__ == "__main__":
    main()
