"""
khzfil: Minimal kHz filamentation baseline (modular).

This package provides a compact, publishable-baseline solver for high-repetition-rate
filamentation in gases using a split-step angular spectrum (paraxial) model with
nonlinear Kerr phase, ionization (rate equation), and slow-time heat/density-hole diffusion.

CPU (NumPy) by default; GPU (CuPy) when environment variable UPPE_USE_GPU=1 is set
and CuPy is available.
"""

from . import (
    constants,
    device,
    config,
    grids,
    linear,
    ionization,
    nonlinear,
    heat,
    propagate,
    diagnostics,
    utils,
    air_dispersion,linear_full, raman
)

__all__ = [
    "constants",
    "device",
    "config",
    "grids",
    "linear",
    "ionization",
    "nonlinear",
    "heat",
    "propagate",
    "diagnostics",
    "utils",
    "air_dispersion", "linear_full","raman"
]
