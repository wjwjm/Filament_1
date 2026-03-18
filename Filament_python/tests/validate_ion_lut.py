from __future__ import annotations
import math
import numpy as np
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.ionization import (
    build_rate_table,
    eval_rate_from_table,
    validate_rate_table,
    _get_reference_evaluator,
)
from KHz_filament.constants import c0


def run_validation_for_species(name: str, model: str, species: dict, lam0: float = 800e-9, n0: float = 1.00027):
    omega0 = 2.0 * math.pi * c0 / lam0
    ref_opts = {
        "cycle_avg_samples_ref": 64,
        "popruzhenko_sum_tol": 1e-6,
        "popruzhenko_max_terms": 256,
    }
    table = build_rate_table(
        model_name=model,
        species_params=species,
        omega0_SI=omega0,
        n0=n0,
        I_min_SI=1e8,
        I_max_SI=1e19,
        n_samples=3000,
        spacing="log",
        reference_opts=ref_opts,
    )
    I_test = np.logspace(8, 19, 256)
    ref_eval = _get_reference_evaluator(model, species, omega0, n0, ref_opts)
    lut_eval = lambda I: eval_rate_from_table(I, table, method="loglog")
    metrics = validate_rate_table(ref_eval, lut_eval, I_test)
    print(f"[{name}] model={model}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6e}")


def main():
    run_validation_for_species(
        "N2-talebpour",
        "ppt_talebpour_i_full_reference",
        {"name": "N2", "Ip_eV": 15.6, "Zeff": 0.9, "l": 0, "m": 0},
    )
    run_validation_for_species(
        "O2-popruzhenko",
        "popruzhenko_atom_i_full_reference",
        {"name": "O2", "Ip_eV": 12.1, "Z": 1, "l": 0, "m": 0},
    )


if __name__ == "__main__":
    main()
