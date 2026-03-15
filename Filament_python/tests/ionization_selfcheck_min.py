"""Minimal ionization model self-check script.

Compares legacy bridged-ADK vs Talebpour molecular PPT vs Popruzhenko atomic model
at 800 nm using cycle-averaged W(I) curves for N2/O2/Xe examples.
"""
from __future__ import annotations

import numpy as np

from KHz_filament.ionization import (
    cycle_average_popruzhenko_from_I,
    cycle_average_ppt_talebpour_from_I,
    make_Wfunc,
)
class _IonConf:
    def __init__(self):
        self.W_cap = 1e17
        self.a_gamma = 0.75
        self.W_scale = 1.0
        self.time_mode = "full"
        self.integrator = "rk4"
        self.cycle_avg_samples = 64
        self.species = [
            {"name": "N2", "rate": "ppt_i_legacy", "Ip_eV": 15.6, "Z": 1, "l": 0, "m": 0, "fraction": 1.0},
        ]


def main():
    lam0 = 800e-9
    c0 = 299792458.0
    omega0 = 2.0 * np.pi * c0 / lam0
    n0 = 1.00027

    I = np.logspace(16, 18, 48)

    ion_legacy = _IonConf()
    W_legacy = make_Wfunc("ppt", ion_legacy, omega0, n0)(I)

    W_n2_taleb = cycle_average_ppt_talebpour_from_I(I, n0=n0, Ip_eV=15.6, Zeff=0.9, l=0, m=0, samples=64)
    W_o2_taleb = cycle_average_ppt_talebpour_from_I(I, n0=n0, Ip_eV=12.55, Zeff=0.53, l=0, m=0, samples=64)

    W_xe_popr = cycle_average_popruzhenko_from_I(I, n0=n0, omega0_SI=omega0, Ip_eV=12.13, Z=1, samples=64, n_terms=96)

    print("# I[W/m^2] W_legacy_N2 W_taleb_N2 W_taleb_O2 W_popruzhenko_Xe")
    for k in (0, 8, 16, 24, 32, 40, 47):
        print(f"{I[k]:.6e} {float(W_legacy[k]):.6e} {float(W_n2_taleb[k]):.6e} {float(W_o2_taleb[k]):.6e} {float(W_xe_popr[k]):.6e}")


if __name__ == "__main__":
    main()
