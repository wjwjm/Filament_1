from __future__ import annotations

import math
import numpy as np

from KHz_filament.constants import c0
from KHz_filament.ionization import (
    _ion_rate_table_defaults,
    _resolve_rate,
    cycle_average_ppt_talebpour_full_from_I,
    make_Wfunc,
    prepare_ionization_lut_for_species,
)


class _IonConf:
    def __init__(self, species):
        self.W_cap = 1e17
        self.W_scale = 1.0
        self.time_mode = "full"
        self.integrator = "rk4"
        self.cycle_avg_samples = 64
        self.species = species
        self.rate_table = {
            "enabled": True,
            "reuse_cache": True,
            "force_rebuild": False,
            "save_tables": False,
            "I_min_SI": 1e10,
            "I_max_SI": 1e16,
            "n_samples": 200,
            "ref_cycle_avg_samples": 24,
            "popruzhenko_max_terms": 64,
        }


def test_split_runtime_curve_matches_reference_formula():
    lam0 = 800e-9
    n0 = 1.00027
    omega0 = 2.0 * math.pi * c0 / lam0
    I = np.logspace(12, 15, 64)

    species = {"name": "N2", "rate": "ppt_talebpour_i_full_reference", "Ip_eV": 15.6, "Zeff": 0.9, "l": 0, "m": 0, "fraction": 1.0}
    conf = _IonConf([species])

    W_new = make_Wfunc("ppt", conf, omega0, n0)(I)
    W_old_like = cycle_average_ppt_talebpour_full_from_I(I, n0=n0, omega0_SI=omega0, Ip_eV=15.6, Zeff=0.9, l=0, m=0, samples=64)

    np.testing.assert_allclose(W_new, W_old_like, rtol=1e-10, atol=0.0)


def test_split_lut_cache_hit_behavior():
    lam0 = 800e-9
    n0 = 1.00027
    omega0 = 2.0 * math.pi * c0 / lam0
    species = {"name": "N2", "rate": "ppt_talebpour_i_lut", "Ip_eV": 15.6, "Zeff": 0.9, "l": 0, "m": 0, "fraction": 1.0}
    conf = _IonConf([species])

    cfg = _ion_rate_table_defaults(conf)
    sp = dict(species)
    sp["rate"] = _resolve_rate(sp, conf)

    t1 = prepare_ionization_lut_for_species(sp, omega0_SI=omega0, n0=n0, rate_table_cfg=cfg)
    t2 = prepare_ionization_lut_for_species(sp, omega0_SI=omega0, n0=n0, rate_table_cfg=cfg)

    assert t1 is t2
