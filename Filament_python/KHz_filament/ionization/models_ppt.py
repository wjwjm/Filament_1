from __future__ import annotations

import math
import numpy as _np

from ..device import xp
from ..constants import eps0, c0
from ..constants import Ip_eV_to_au, E_SI_to_au, omega_SI_to_au
from .common import W_CAP_DEFAULT, _as_real_like, _minmax_inplace, _safe_exp_inplace

try:
    from scipy import special as _sp_special  # type: ignore
except Exception:
    _sp_special = None


def _W_mpa_factorial(I_SI, omega0_SI: float, I_mp: float, ell: int, W_cap: float):
    I = xp.asarray(I_SI)
    I = xp.maximum(I, xp.asarray(1e-300, dtype=I.dtype))
    Gamma0 = 2.0 * math.pi * float(omega0_SI) / math.factorial(max(ell - 1, 1))
    W = Gamma0 * (I / float(I_mp)) ** int(ell)
    W = xp.clip(W, 0.0, float(W_cap))
    W = xp.nan_to_num(W, nan=0.0, posinf=float(W_cap), neginf=0.0)
    return W


def _factorial_safe(n: int) -> float:
    if n < 0:
        return 1.0
    return float(math.factorial(int(n)))


def _ppt_prefactor_C2(n_star: float, l: int) -> float:
    nsl = max(float(n_star - l), 1e-8)
    return (2.0 ** (2.0 * n_star)) / (float(n_star) * math.gamma(n_star + l + 1.0) * math.gamma(nsl))


def w_ppt_talebpour_legacy_from_E(E_SI, Ip_eV: float, Zeff: float, l: int, m: int, W_cap: float = W_CAP_DEFAULT):
    Eabs = _as_real_like(xp.abs(E_SI), like=E_SI)
    _minmax_inplace(Eabs, 1e-30, None)
    Ip_au = float(Ip_eV_to_au(Ip_eV))
    kappa = math.sqrt(2.0 * Ip_au)
    n_star = float(Zeff) / max(kappa, 1e-30)
    E_au = _as_real_like(E_SI_to_au(Eabs), like=Eabs)
    _minmax_inplace(E_au, 1e-30, None)

    am = abs(int(m))
    C2 = _ppt_prefactor_C2(n_star, int(l))
    ang = (2.0 * int(l) + 1.0) * _factorial_safe(int(l) + am)
    ang /= max((2.0 ** am) * _factorial_safe(am) * _factorial_safe(int(l) - am), 1e-30)

    power_base = 2.0 * (kappa**3) / _minmax_inplace(E_au, 1e-30, None)
    _minmax_inplace(power_base, 1e-30, None)
    power = 2.0 * n_star - am - 1.0
    power_term = xp.exp(power * xp.log(power_base))
    expo = -2.0 * (kappa**3) / (3.0 * E_au)
    expo_term = _safe_exp_inplace(expo, neg_only=True)

    W_au = float(C2 * ang) * power_term * expo_term
    W_SI = _as_real_like(W_au * 4.134137333e16, like=Eabs)
    _minmax_inplace(W_SI, 0.0, W_cap)
    return W_SI


def cycle_average_ppt_talebpour_legacy_from_I(I_SI, n0: float, Ip_eV: float, Zeff: float, l: int, m: int, samples: int = 64, W_cap: float = W_CAP_DEFAULT):
    phase = _np.linspace(0.0, _np.pi, max(8, int(samples)), endpoint=False)
    cos_abs = _np.abs(_np.cos(phase))
    fac_E_from_I = (2.0 / (float(n0) * float(eps0) * float(c0))) ** 0.5
    I = xp.asarray(I_SI)
    E0 = xp.sqrt(xp.maximum(I, 0.0)) * fac_E_from_I
    acc = 0.0
    for cs in cos_abs:
        acc = acc + w_ppt_talebpour_legacy_from_E(E0 * float(cs), Ip_eV=Ip_eV, Zeff=Zeff, l=l, m=m, W_cap=W_cap)
    W_mean = acc / len(cos_abs)
    return xp.nan_to_num(xp.clip(W_mean, 0.0, W_cap), nan=0.0, posinf=W_cap, neginf=0.0)


def _dawson_xp(x):
    arr = xp.asarray(x)
    cupy_backend = xp.__name__ == "cupy"
    if cupy_backend:
        try:
            import cupyx.scipy.special as _csp  # type: ignore
            return _csp.dawsn(arr)
        except Exception:
            pass
    elif _sp_special is not None:
        arr_np = _np.asarray(arr)
        return xp.asarray(_sp_special.dawsn(arr_np), dtype=arr.dtype)

    ax = xp.abs(arr)
    small = ax < 0.2
    large = ax > 4.0
    mid = ~(small | large)
    out = xp.zeros_like(arr, dtype=arr.dtype)

    xs = arr[small]
    out[small] = xs * (1.0 - (2.0 / 3.0) * xs * xs + (4.0 / 15.0) * xs * xs * xs * xs)
    xl = arr[large]
    inv = 1.0 / xp.maximum(xl, 1e-30)
    out[large] = 0.5 * inv + 0.25 * inv**3 + 0.375 * inv**5

    xm = arr[mid]
    x2 = xm * xm
    out[mid] = xm * (1.0 + 0.10499349 * x2 + 0.04240606 * x2 * x2) / (
        1.0 + 0.77154710 * x2 + 0.29097386 * x2 * x2 + 0.06945558 * x2**3
    )
    return out


def _ppt_g_gamma(gamma):
    gamma = _minmax_inplace(_as_real_like(gamma), 1e-12, 1e6)
    asinh_g = xp.arcsinh(gamma)
    sqrt_1g2 = xp.sqrt(1.0 + gamma * gamma)
    return (3.0 / (2.0 * gamma)) * ((1.0 + 1.0 / (2.0 * gamma * gamma)) * asinh_g - sqrt_1g2 / (2.0 * gamma))


def _ppt_alpha(gamma):
    gamma = _minmax_inplace(_as_real_like(gamma), 1e-12, 1e6)
    return 2.0 * (xp.arcsinh(gamma) - gamma / xp.sqrt(1.0 + gamma * gamma))


def _sum_tail_adaptive(term_fn, first_idx, max_terms=4096, rel_tol=1e-9):
    s = None
    for k in range(int(max_terms)):
        t = term_fn(first_idx + float(k))
        s = t if s is None else (s + t)
        if k > 8:
            denom = xp.maximum(xp.abs(s), 1e-300)
            if bool(xp.all(xp.abs(t) <= rel_tol * denom)):
                break
    return s if s is not None else 0.0


def _ppt_Am_series_m0(gamma, nu, *, max_terms=4096, rel_tol=1e-9):
    gamma = _minmax_inplace(_as_real_like(gamma), 1e-12, 1e6)
    beta = 2.0 * gamma / xp.sqrt(1.0 + gamma * gamma)
    alpha = _ppt_alpha(gamma)
    n0 = xp.floor(nu) + 1.0

    def _term(n):
        dn = n - nu
        return _dawson_xp(xp.sqrt(beta * dn)) * _safe_exp_inplace(-alpha * dn, neg_only=True)

    s = _sum_tail_adaptive(_term, n0, max_terms=max_terms, rel_tol=rel_tol)
    pref = (4.0 / math.sqrt(3.0 * math.pi)) * (gamma * gamma) / (1.0 + gamma * gamma)
    return pref * s


def w_ppt_talebpour_full_from_E(E_SI, omega0_SI: float, Ip_eV: float, Zeff: float, l: int, m: int, *, max_terms: int = 4096, sum_rel_tol: float = 1e-9, W_cap: float = W_CAP_DEFAULT):
    Eabs = _minmax_inplace(_as_real_like(xp.abs(E_SI), like=E_SI), 1e-30, None)
    F = _minmax_inplace(_as_real_like(E_SI_to_au(Eabs), like=Eabs), 1e-30, None)
    Ip_au = float(Ip_eV_to_au(Ip_eV))
    omega_au = max(float(omega_SI_to_au(omega0_SI)), 1e-30)
    kappa = math.sqrt(2.0 * Ip_au)
    n_star = float(Zeff) / max(kappa, 1e-30)
    am = abs(int(m))

    gamma = _minmax_inplace(omega_au * kappa / F, 1e-12, 1e8)
    nu = float(Ip_au) * (1.0 + 1.0 / (2.0 * gamma * gamma)) / omega_au

    C2 = _ppt_prefactor_C2(n_star, int(l))
    f_lm = (2.0 * int(l) + 1.0) * _factorial_safe(int(l) + am)
    f_lm /= max((2.0**am) * _factorial_safe(am) * _factorial_safe(int(l) - am), 1e-30)

    A_m0 = _ppt_Am_series_m0(gamma, nu, max_terms=max_terms, rel_tol=sum_rel_tol)
    A_m = A_m0 if am == 0 else A_m0 * (2.0 * gamma / xp.sqrt(1.0 + gamma * gamma)) ** am
    g_gamma = _ppt_g_gamma(gamma)
    expo = _safe_exp_inplace(-2.0 * (kappa**3) * g_gamma / (3.0 * F), neg_only=True)
    sqrt_term = xp.sqrt((3.0 / (math.pi * F)) * ((1.0 + gamma * gamma) ** (-am / 2.0 + 0.75)) * xp.maximum(A_m, 0.0))
    w_au = float(C2 * f_lm * Ip_au) * sqrt_term * expo
    w_si = _as_real_like(w_au * 4.134137333e16, like=Eabs)
    return xp.nan_to_num(xp.clip(w_si, 0.0, W_cap), nan=0.0, posinf=W_cap, neginf=0.0)


def cycle_average_ppt_talebpour_full_from_I(I_SI, n0: float, omega0_SI: float, Ip_eV: float, Zeff: float, l: int, m: int, samples: int = 64, max_terms: int = 4096, sum_rel_tol: float = 1e-9, W_cap: float = W_CAP_DEFAULT):
    phase = _np.linspace(0.0, _np.pi, max(8, int(samples)), endpoint=False)
    cos_abs = _np.abs(_np.cos(phase))
    fac_E_from_I = (2.0 / (float(n0) * float(eps0) * float(c0))) ** 0.5
    I = xp.asarray(I_SI)
    E0 = xp.sqrt(xp.maximum(I, 0.0)) * fac_E_from_I
    acc = 0.0
    for cs in cos_abs:
        acc = acc + w_ppt_talebpour_full_from_E(
            E0 * float(cs), omega0_SI=omega0_SI, Ip_eV=Ip_eV, Zeff=Zeff, l=l, m=m,
            max_terms=max_terms, sum_rel_tol=sum_rel_tol, W_cap=W_cap
        )
    W_mean = acc / len(cos_abs)
    return xp.nan_to_num(xp.clip(W_mean, 0.0, W_cap), nan=0.0, posinf=W_cap, neginf=0.0)
