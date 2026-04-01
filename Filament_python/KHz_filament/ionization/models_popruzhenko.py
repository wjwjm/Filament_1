from __future__ import annotations

import math
import numpy as _np

from ..device import xp
from ..constants import eps0, c0
from ..constants import Ip_eV_to_au, E_SI_to_au, omega_SI_to_au
from .common import W_CAP_DEFAULT, _as_real_like, _minmax_inplace, _safe_exp_inplace
from .models_ppt import _dawson_xp, _ppt_g_gamma, _sum_tail_adaptive


def popruzhenko_coulomb_Q_full(F, gamma, n_star):
    F = _minmax_inplace(_as_real_like(F), 1e-30, None)
    gamma = _minmax_inplace(_as_real_like(gamma), 0.0, 1e6)
    n_star = float(n_star)
    lnQ = (2.0 * n_star) * xp.log(2.0 / F) - (2.0 * n_star) * xp.log(1.0 + 2.0 * math.exp(-1.0) * gamma)
    Q = _safe_exp_inplace(lnQ, neg_only=False)
    return _minmax_inplace(Q, 0.0, 1e300)


def popruzhenko_short_range_wSR_full(F, gamma, Ip_au: float, omega_au: float, n_star: float, max_terms: int = 4096, sum_rel_tol: float = 1e-9, W_cap: float = W_CAP_DEFAULT):
    F = _minmax_inplace(_as_real_like(F), 1e-30, None)
    gamma = _minmax_inplace(_as_real_like(gamma), 1e-12, 1e6)

    K0 = float(Ip_au) / max(float(omega_au), 1e-30)
    n_th = K0 * (1.0 + 1.0 / (2.0 * gamma * gamma))
    asinh_g = xp.arcsinh(gamma)
    sqrt_1g2 = xp.sqrt(1.0 + gamma * gamma)
    g_gamma = _ppt_g_gamma(gamma)
    c1 = asinh_g - gamma / sqrt_1g2
    beta = 2.0 * gamma / sqrt_1g2

    C2 = (2.0 ** (2.0 * n_star - 2.0)) / (math.gamma(n_star + 1.0) ** 2)
    pref = (2.0 * C2 / math.pi) * float(Ip_au) * (K0**-1.5) * xp.sqrt(beta)

    n0 = xp.floor(n_th) + 1.0

    def _term(n):
        dn = n - n_th
        expo_arg = -2.0 * c1 * dn
        expo = _safe_exp_inplace(expo_arg, neg_only=True)
        return expo * _dawson_xp(xp.sqrt(beta * dn))

    sum_term = _sum_tail_adaptive(_term, n0, max_terms=max_terms, rel_tol=sum_rel_tol)

    w_sr_au = pref * _safe_exp_inplace(-2.0 * g_gamma / (3.0 * F), neg_only=True) * sum_term
    w_sr = _as_real_like(w_sr_au * 4.134137333e16)
    return _minmax_inplace(w_sr, 0.0, W_cap)


def w_popruzhenko_atom_full_from_E(E_SI, omega0_SI: float, Ip_eV: float, Z: int, max_terms: int = 4096, sum_rel_tol: float = 1e-9, W_cap: float = W_CAP_DEFAULT):
    Eabs = _as_real_like(xp.abs(E_SI), like=E_SI)
    _minmax_inplace(Eabs, 1e-30, None)
    Ip_au = float(Ip_eV_to_au(Ip_eV))
    omega_au = float(omega_SI_to_au(omega0_SI))
    E_au = _as_real_like(E_SI_to_au(Eabs), like=Eabs)
    _minmax_inplace(E_au, 1e-30, None)

    n_star = float(Z) / math.sqrt(2.0 * Ip_au)
    F = E_au / ((2.0 * Ip_au) ** 1.5)
    gamma = omega_au * math.sqrt(2.0 * Ip_au) / E_au

    Q = popruzhenko_coulomb_Q_full(F=F, gamma=gamma, n_star=n_star)
    w_sr = popruzhenko_short_range_wSR_full(F=F, gamma=gamma, Ip_au=Ip_au, omega_au=omega_au, n_star=n_star, max_terms=max_terms, sum_rel_tol=sum_rel_tol, W_cap=W_cap)
    w = Q * w_sr
    return xp.nan_to_num(xp.clip(w, 0.0, W_cap), nan=0.0, posinf=W_cap, neginf=0.0)


def cycle_average_popruzhenko_atom_full_from_I(I_SI, n0: float, omega0_SI: float, Ip_eV: float, Z: int, samples: int = 64, max_terms: int = 4096, sum_rel_tol: float = 1e-9, W_cap: float = W_CAP_DEFAULT):
    phase = _np.linspace(0.0, _np.pi, max(8, int(samples)), endpoint=False)
    cos_abs = _np.abs(_np.cos(phase))
    fac_E_from_I = (2.0 / (float(n0) * float(eps0) * float(c0))) ** 0.5
    I = xp.asarray(I_SI)
    E0 = xp.sqrt(xp.maximum(I, 0.0)) * fac_E_from_I
    acc = 0.0
    for cs in cos_abs:
        acc = acc + w_popruzhenko_atom_full_from_E(E0 * float(cs), omega0_SI=omega0_SI, Ip_eV=Ip_eV, Z=Z, max_terms=max_terms, sum_rel_tol=sum_rel_tol, W_cap=W_cap)
    W_mean = acc / len(cos_abs)
    return xp.nan_to_num(xp.clip(W_mean, 0.0, W_cap), nan=0.0, posinf=W_cap, neginf=0.0)
