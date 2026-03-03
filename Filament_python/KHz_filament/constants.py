from __future__ import annotations
import numpy as _np
from .device import xp

__all__ = [
    "c0", "eps0", "mu0", "e", "me",
    "n0_air", "n2_air", "Ui_N2", "N0_air",
    "rho_crit",
    "hbar", "a0", "Eh", "eV",
    "Ip_eV_to_au", "E_SI_to_au", "omega_SI_to_au",
    "field_from_intensity_SI",
]

# ----------------- Physical constants (SI) -----------------
c0   = 2.99792458e8         # m/s
eps0 = 8.854187817e-12      # F/m
mu0  = 4e-7 * _np.pi        # H/m   (注意：这里用 _np.pi，只做常数，不涉数组)
e    = 1.602176634e-19      # C
me   = 9.10938356e-31       # kg

# Typical air properties near ~800 nm
n0_air = 1.00027
n2_air = 0.78e-23        # 0.78e-23   3.2e-23  m^2/W  (Kerr coefficient)，采用较新的实验文献
Ui_N2  = 15.6 * e           # J      (ionization potential ~ N2)
N0_air = 2.5e25             # 1/m^3  (molecular number density ~ STP)

def rho_crit(omega):
    """
    Critical electron density for angular frequency omega (SI).

    - 若传入标量 (int/float)，返回 Python float（避免 CuPy 0-d 到 float 的禁止隐式转换问题）。
    - 若传入数组（np/cp），返回 xp.ndarray，保持后端一致性。
    """
    if _np.isscalar(omega):
        return eps0 * me * (omega ** 2) / (e ** 2)
    omega_xp = xp.asarray(omega)
    return eps0 * me * (omega_xp ** 2) / (e ** 2)

# ----------------- Atomic units & conversions -----------------
# Atomic units references:
#   1 a.u. of electric field:  Eh / (e * a0) ≈ 5.142206747e11 V/m
#   1 a.u. of angular frequency: Eh / ħ ≈ 4.134137333e16 rad/s
#   1 a.u. of energy (Hartree): Eh = 4.3597447222071e-18 J
#   Bohr radius a0 = 5.29177210903e-11 m
hbar = 1.054571817e-34      # J*s
a0   = 5.29177210903e-11    # m
Eh   = 4.3597447222071e-18  # J
eV   = 1.602176634e-19      # J

_EFIELD_AU_SI = 5.142206747e11     # V/m
_OMEGA_AU_SI  = 4.134137333e16     # rad/s

def Ip_eV_to_au(Ip_eV: float):
    """Convert ionization potential from eV to atomic units (Hartree=1)."""
    return (Ip_eV * eV) / Eh

def E_SI_to_au(E_SI):
    """Convert electric field amplitude |E| from SI (V/m) to atomic units."""
    return xp.asarray(E_SI) / _EFIELD_AU_SI

def omega_SI_to_au(omega):
    """Convert angular frequency (rad/s) to atomic units."""
    return xp.asarray(omega) / _OMEGA_AU_SI

def field_from_intensity_SI(I_SI, n0: float):
    """
    Compute |E| from intensity I (SI):
        |E| = sqrt(2 I / (eps0 c0 n0))
    支持标量与数组输入；返回与后端一致的 xp.ndarray 或 Python float。
    """
    I_xp = xp.asarray(I_SI)
    I_xp = xp.maximum(I_xp, 0.0)
    return xp.sqrt(2.0 * I_xp / (eps0 * c0 * n0))
