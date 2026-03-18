#!/usr/bin/env python3
from __future__ import annotations
import argparse
import math

from Filament_python.KHz_filament.confio import load_all
from Filament_python.KHz_filament.constants import c0
from Filament_python.KHz_filament.ionization import prepare_ionization_lut_cache


def main() -> int:
    ap = argparse.ArgumentParser(description="Build and cache ionization LUT tables only.")
    ap.add_argument("--config", required=True, help="Path to simulation config (json/yaml/toml).")
    args = ap.parse_args()

    _grid, beam, _prop, ion, _heat, _run, _raman = load_all(args.config)
    omega0 = 2.0 * math.pi * float(c0) / float(beam.lam0)
    print(f"[build_ion_lut] config={args.config}")
    print(f"[build_ion_lut] lam0={beam.lam0:.6e} m omega0={omega0:.6e} rad/s n0={beam.n0:.6f}")
    tables = prepare_ionization_lut_cache(ion, omega0_SI=omega0, n0=float(beam.n0))
    print(f"[build_ion_lut] prepared_tables={len(tables)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
