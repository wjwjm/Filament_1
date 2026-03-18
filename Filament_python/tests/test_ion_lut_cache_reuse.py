from __future__ import annotations
import math
import tempfile
from types import SimpleNamespace
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.constants import c0
from KHz_filament.ionization import prepare_ionization_lut_cache


def main():
    with tempfile.TemporaryDirectory() as td:
        ion = SimpleNamespace(
            species=[
                {
                    "name": "N2",
                    "rate": "ppt_talebpour_i_lut",
                    "reference_model": "ppt_talebpour_i_full_reference",
                    "Ip_eV": 15.6,
                    "Zeff": 0.9,
                    "fraction": 0.8,
                }
            ],
            rate_table={
                "enabled": True,
                "reuse_cache": True,
                "cache_dir": td,
                "rebuild_if_missing": True,
                "force_rebuild": False,
                "save_tables": True,
                "I_min_SI": 1e8,
                "I_max_SI": 1e19,
                "n_samples": 800,
                "spacing": "log",
                "interp_mode": "loglog",
                "ref_cycle_avg_samples": 32,
                "popruzhenko_sum_tol": 1e-6,
                "popruzhenko_max_terms": 128,
            },
        )
        omega0 = 2.0 * math.pi * c0 / 800e-9
        print("[cache-test] first run (expect build)")
        prepare_ionization_lut_cache(ion, omega0_SI=omega0, n0=1.00027)
        print("[cache-test] second run (expect cache hit)")
        prepare_ionization_lut_cache(ion, omega0_SI=omega0, n0=1.00027)


if __name__ == "__main__":
    main()
