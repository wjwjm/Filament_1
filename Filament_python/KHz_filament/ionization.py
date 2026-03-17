from __future__ import annotations
from .device import xp
from .constants import eps0, c0
from .constants import Ip_eV_to_au, E_SI_to_au, omega_SI_to_au
import math
import numpy as _np

try:
    from scipy import special as _sp_special  # type: ignore
except Exception:
    _sp_special = None

# ----------------- Global safety caps -----------------
I_CAP_DEFAULT = 1e19    # W/m^2, cap intensity
W_CAP_DEFAULT = 1e16    # 1/s,  cap ionization rate
DRHO_FRAC     = 0.25    # per time step, max fractional increase of N0

# ----------------- Small helpers (OOM-safe) -----------------
def _nan_inf_to_num_inplace(arr, nan=0.0, posinf=None, neginf=None, *, chunk_elems=1<<20):
    a = xp.asarray(arr)
    flat = a.reshape(-1)
    n = flat.size
    dtype = a.dtype
    if posinf is None:
        posinf_val = xp.finfo(dtype).max if dtype.kind == 'f' else 0
    else:
        posinf_val = dtype.type(posinf)
    if neginf is None:
        neginf_val = -xp.finfo(dtype).max if dtype.kind == 'f' else 0
    else:
        neginf_val = dtype.type(neginf)
    nan_val = dtype.type(nan) if nan is not None else None
    step = int(max(1024, chunk_elems))
    for j in range(0, n, step):
        v = flat[j:j+step]
        if nan_val is not None:
            m_nan = xp.isnan(v); v[m_nan] = nan_val; del m_nan
        m_pos = xp.isposinf(v)
        if xp.any(m_pos): v[m_pos] = posinf_val
        del m_pos
        m_neg = xp.isneginf(v)
        if xp.any(m_neg): v[m_neg] = neginf_val
        del m_neg
    return a

def _as_real_like(a, like=None):
    arr = xp.asarray(a)
    if xp.iscomplexobj(arr):
        arr = xp.real(arr)
    if like is not None and getattr(like, "dtype", None) in (xp.float32, xp.complex64):
        return arr.astype(xp.float32, copy=False)
    return arr.astype(xp.float64 if arr.dtype == xp.float64 else xp.float32, copy=False)

def _minmax_inplace(x, lo=None, hi=None):
    arr = xp.asarray(x)
    if lo is not None: xp.maximum(arr, xp.asarray(lo, dtype=arr.dtype), out=arr)
    if hi is not None: xp.minimum(arr, xp.asarray(hi, dtype=arr.dtype), out=arr)
    return arr

def _safe_exp_inplace(arg, neg_only=True):
    arr = xp.asarray(arg)
    if arr.dtype not in (xp.float32, xp.float64):
        arr = arr.astype(xp.float32, copy=False)
    if arr.dtype == xp.float32:
        lo, hi = (-80.0, 0.0) if neg_only else (-80.0, 50.0)
    else:
        lo, hi = (-700.0, 0.0) if neg_only else (-700.0, 50.0)
    xp.maximum(arr, xp.asarray(lo, dtype=arr.dtype), out=arr)
    xp.minimum(arr, xp.asarray(hi, dtype=arr.dtype), out=arr)
    xp.exp(arr, out=arr)
    return arr

def _g(sp, key, default=None):
    if isinstance(sp, dict):
        return sp.get(key, default)
    return getattr(sp, key, default)

# ----------------- Basic optics helpers -----------------
def intensity(E, n0: float, I_cap: float = I_CAP_DEFAULT):
    I = 0.5 * eps0 * c0 * n0 * (xp.abs(E) ** 2)
    return _minmax_inplace(I, 0.0, I_cap)

def field_amplitude_from_intensity(I, n0: float):
    return xp.sqrt(_minmax_inplace(2.0 * I / (eps0 * c0 * n0), 0.0))

# ----------------- MPA with factorial prefactor (paper-consistent) -----------------
def _W_mpa_factorial(I_SI, omega0_SI: float, I_mp: float, ell: int, W_cap: float):
    """W(I) = Gamma0 * (I / I_mp)^ell,  Gamma0 = 2π ω0 / ((ell-1)!)"""
    I = xp.asarray(I_SI); I = xp.maximum(I, xp.asarray(1e-300, dtype=I.dtype))
    Gamma0 = 2.0 * math.pi * float(omega0_SI) / math.factorial(max(ell - 1, 1))
    W = Gamma0 * (I / float(I_mp))**int(ell)
    W = xp.clip(W, 0.0, float(W_cap))
    W = xp.nan_to_num(W, nan=0.0, posinf=float(W_cap), neginf=0.0)
    return W

def _factorial_safe(n: int) -> float:
    if n < 0:
        return 1.0
    return float(math.factorial(int(n)))


def _ppt_prefactor_C2(n_star: float, l: int) -> float:
    """PPT/ADK-like C_{nl}^2 prefactor used by Talebpour-style molecular branch."""
    nsl = max(float(n_star - l), 1e-8)
    return (2.0 ** (2.0 * n_star)) / (float(n_star) * math.gamma(n_star + l + 1.0) * math.gamma(nsl))


def w_ppt_talebpour_legacy_from_E(E_SI, Ip_eV: float, Zeff: float, l: int, m: int,
                                  W_cap: float = W_CAP_DEFAULT):
    """
    Legacy simplified PPT/ADK-like branch (NOT Talebpour Appendix-A full form).

    Notes:
    - Uses a PPT/ADK-form instantaneous rate with molecular effective charge Zeff
      entering n* = Zeff / sqrt(2*Ip_au).
    - This is for molecular semi-empirical PPT usage (e.g., N2/O2), and is NOT
      Popruzhenko-2008 arbitrary-gamma Coulomb-corrected atomic formula.
    """
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

    # PPT/ADK-like tunneling exponent and power law (instantaneous rate)
    power_base = 2.0 * (kappa ** 3) / _minmax_inplace(E_au, 1e-30, None)
    _minmax_inplace(power_base, 1e-30, None)
    power = 2.0 * n_star - am - 1.0
    power_term = xp.exp(power * xp.log(power_base))
    expo = -2.0 * (kappa ** 3) / (3.0 * E_au)
    expo_term = _safe_exp_inplace(expo, neg_only=True)

    W_au = float(C2 * ang) * power_term * expo_term
    W_SI = _as_real_like(W_au * 4.134137333e16, like=Eabs)
    _minmax_inplace(W_SI, 0.0, W_cap)
    return W_SI


def cycle_average_ppt_talebpour_legacy_from_I(I_SI, n0: float, Ip_eV: float, Zeff: float, l: int, m: int,
                                              samples: int = 64, W_cap: float = W_CAP_DEFAULT):
    """
    Cycle-averaged legacy simplified PPT/ADK-like molecular rate:
        W_bar(I) = (1/pi)∫_0^pi W_PPT_mol(E0*|cos(phi)|) dphi
    where E0 = sqrt(2I/(n0*eps0*c0)).
    """
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
    """Dawson function with scipy/cupyx path and a fallback approximation."""
    arr = xp.asarray(x)
    try:
        if xp.__name__ == "cupy":
            import cupyx.scipy.special as _csp  # type: ignore
            return _csp.dawsn(arr)
    except Exception:
        pass

    if _sp_special is not None:
        arr_np = _np.asarray(arr)
        return xp.asarray(_sp_special.dawsn(arr_np), dtype=arr.dtype)

    # fallback approximation: low-order near zero + asymptotic tail + rational mid-range
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
    """
    Appendix-A-like A_m series core for m=0.
    needs_formula_verification: OCR 对 A_m 的 m 依赖项存在歧义，此处先严格实现 m=0 主项。
    """
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


def w_ppt_talebpour_full_from_E(E_SI, omega0_SI: float, Ip_eV: float, Zeff: float, l: int, m: int,
                                *, max_terms: int = 4096, sum_rel_tol: float = 1e-9,
                                W_cap: float = W_CAP_DEFAULT):
    """Talebpour 1999 Appendix-A PPT molecular branch (full-form implementation with Zeff substitution)."""
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
    f_lm /= max((2.0 ** am) * _factorial_safe(am) * _factorial_safe(int(l) - am), 1e-30)

    A_m0 = _ppt_Am_series_m0(gamma, nu, max_terms=max_terms, rel_tol=sum_rel_tol)
    if am == 0:
        A_m = A_m0
    else:
        # needs_formula_verification: Appendix-A OCR 对 A_m 的 m>0 权重项不清晰。
        A_m = A_m0 * (2.0 * gamma / xp.sqrt(1.0 + gamma * gamma)) ** am

    g_gamma = _ppt_g_gamma(gamma)
    expo = _safe_exp_inplace(-2.0 * (kappa ** 3) * g_gamma / (3.0 * F), neg_only=True)
    sqrt_term = xp.sqrt((3.0 / (math.pi * F)) * ((1.0 + gamma * gamma) ** (-am / 2.0 + 0.75)) * xp.maximum(A_m, 0.0))
    w_au = float(C2 * f_lm * Ip_au) * sqrt_term * expo
    w_si = _as_real_like(w_au * 4.134137333e16, like=Eabs)
    return xp.nan_to_num(xp.clip(w_si, 0.0, W_cap), nan=0.0, posinf=W_cap, neginf=0.0)


def cycle_average_ppt_talebpour_full_from_I(I_SI, n0: float, omega0_SI: float, Ip_eV: float, Zeff: float, l: int, m: int,
                                            samples: int = 64, max_terms: int = 4096, sum_rel_tol: float = 1e-9,
                                            W_cap: float = W_CAP_DEFAULT):
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


def popruzhenko_coulomb_Q_full(F, gamma, n_star):
    """Popruzhenko 2008 Coulomb correction Q for atomic/ionic targets."""
    F = _minmax_inplace(_as_real_like(F), 1e-30, None)
    gamma = _minmax_inplace(_as_real_like(gamma), 0.0, 1e6)
    n_star = float(n_star)
    lnQ = (2.0 * n_star) * xp.log(2.0 / F) - (2.0 * n_star) * xp.log(1.0 + 2.0 * math.exp(-1.0) * gamma)
    Q = _safe_exp_inplace(lnQ, neg_only=False)
    return _minmax_inplace(Q, 0.0, 1e300)


def popruzhenko_short_range_wSR_full(F, gamma, Ip_au: float, omega_au: float, n_star: float,
                                     max_terms: int = 4096, sum_rel_tol: float = 1e-9,
                                     W_cap: float = W_CAP_DEFAULT):
    """Popruzhenko 2008 short-range rate term w_SR (atomic/ionic; linear polarization)."""
    F = _minmax_inplace(_as_real_like(F), 1e-30, None)
    gamma = _minmax_inplace(_as_real_like(gamma), 1e-12, 1e6)

    K0 = float(Ip_au) / max(float(omega_au), 1e-30)
    n_th = K0 * (1.0 + 1.0 / (2.0 * gamma * gamma))
    asinh_g = xp.arcsinh(gamma)
    sqrt_1g2 = xp.sqrt(1.0 + gamma * gamma)
    g_gamma = _ppt_g_gamma(gamma)
    c1 = asinh_g - gamma / sqrt_1g2
    beta = 2.0 * gamma / sqrt_1g2

    # Hartree approximation for s-states: C^2 = 2^(2n*-2) / (Gamma(n*+1)^2)
    C2 = (2.0 ** (2.0 * n_star - 2.0)) / (math.gamma(n_star + 1.0) ** 2)
    pref = (2.0 * C2 / math.pi) * float(Ip_au) * (K0 ** -1.5) * xp.sqrt(beta)

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


def w_popruzhenko_atom_full_from_E(E_SI, omega0_SI: float, Ip_eV: float, Z: int,
                                   max_terms: int = 4096, sum_rel_tol: float = 1e-9,
                                   W_cap: float = W_CAP_DEFAULT):
    """
    Popruzhenko 2008 arbitrary-gamma Coulomb-corrected atomic/ionic rate.
    Assumptions: non-resonant, single-active-electron, non-relativistic,
    primarily linear polarization.
    """
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
    w_sr = popruzhenko_short_range_wSR_full(F=F, gamma=gamma, Ip_au=Ip_au, omega_au=omega_au,
                                            n_star=n_star, max_terms=max_terms,
                                            sum_rel_tol=sum_rel_tol, W_cap=W_cap)
    w = Q * w_sr
    return xp.nan_to_num(xp.clip(w, 0.0, W_cap), nan=0.0, posinf=W_cap, neginf=0.0)


def cycle_average_popruzhenko_atom_full_from_I(I_SI, n0: float, omega0_SI: float, Ip_eV: float, Z: int,
                                                samples: int = 64, max_terms: int = 4096,
                                                sum_rel_tol: float = 1e-9, W_cap: float = W_CAP_DEFAULT):
    """Cycle average of Popruzhenko atomic rate over E0*|cos(phi)|."""
    phase = _np.linspace(0.0, _np.pi, max(8, int(samples)), endpoint=False)
    cos_abs = _np.abs(_np.cos(phase))
    fac_E_from_I = (2.0 / (float(n0) * float(eps0) * float(c0))) ** 0.5
    I = xp.asarray(I_SI)
    E0 = xp.sqrt(xp.maximum(I, 0.0)) * fac_E_from_I
    acc = 0.0
    for cs in cos_abs:
        acc = acc + w_popruzhenko_atom_full_from_E(E0 * float(cs), omega0_SI=omega0_SI,
                                                   Ip_eV=Ip_eV, Z=Z, max_terms=max_terms,
                                                   sum_rel_tol=sum_rel_tol, W_cap=W_cap)
    W_mean = acc / len(cos_abs)
    return xp.nan_to_num(xp.clip(W_mean, 0.0, W_cap), nan=0.0, posinf=W_cap, neginf=0.0)


# ----------------- Rate resolver (new) -----------------
def _removed_rate_error(rate: str, *, context: str) -> ValueError:
    r = str(rate).lower()
    repl = "ppt_talebpour_i_legacy / ppt_talebpour_i_full / popruzhenko_atom_i_full / mpa_fact / off"
    if r in ("ppt_e", "adk_e"):
        hint = "该旧模型为 |E| 域模型，现已下线；请改用 I 域模型并传入强度 I。"
    else:
        hint = "请改用保留模型。"
    return ValueError(
        f"[ionization] {context}: rate='{rate}' 已被移除。"
        f"建议替代: {repl}。{hint}"
    )


def _resolve_rate(sp, ion_conf):
    """Resolve species ionization rate with strict deprecation handling."""
    removed_rates = {"ppt_e", "ppt_i", "ppt_i_legacy", "adk_e", "powerlaw", "mpa"}
    r = str(_g(sp, "rate", None) or "").lower().replace("ppt-i", "ppt_i")
    alias_map = {
        "ppt_talebpour_i": "ppt_talebpour_i_full",
        "popruzhenko_atom_i": "popruzhenko_atom_i_full",
    }
    if r:
        if r in removed_rates:
            raise _removed_rate_error(r, context="species.rate")
        if r in ("none", "zero"):
            return "off"
        if r in alias_map:
            print(f"[ionization] rate alias '{r}' -> '{alias_map[r]}'")
            return alias_map[r]
        if r in ("mpa_fact", "ppt_talebpour_i_legacy", "ppt_talebpour_i_full", "popruzhenko_atom_i_full", "off"):
            return r
        raise ValueError(
            "[ionization] 未识别的 rate='{}'。允许值仅为: "
            "ppt_talebpour_i_legacy / ppt_talebpour_i_full / popruzhenko_atom_i_full / mpa_fact / off".format(r)
        )

    # backward compatibility: infer from legacy model + cycle_avg
    m = str(_g(sp, "model", getattr(ion_conf, "model", "off"))).lower()
    cyc = bool(_g(sp, "cycle_avg", getattr(ion_conf, "cycle_avg", False)))
    if m in ("none", "off", "zero", ""):
        return "off"
    if m in ("mpa_fact", "mpa_factorial", "multiphoton_factorial"):
        return "mpa_fact"
    if m in ("ppt", "ppt_cycleavg", "adk", "powerlaw", "mpa"):
        ctx = f"legacy model='{m}' with cycle_avg={cyc}"
        raise _removed_rate_error(m, context=ctx)
    raise ValueError(
        f"[ionization] 无法从 legacy model='{m}' 推断可用电离模型；"
        "请在 species 中显式设置 rate 为: ppt_talebpour_i_legacy / ppt_talebpour_i_full / popruzhenko_atom_i_full / mpa_fact / off"
    )


def _talebpour_defaults(name: str, Ip_eV: float | None, Ip_eV_eff: float | None, Zeff: float | None):
    n = str(name).upper()
    if n == "O2":
        return (
            float(Ip_eV_eff if Ip_eV_eff is not None else 12.55),
            float(Zeff if Zeff is not None else 0.53),
        )
    if n == "N2":
        return (
            float(Ip_eV if Ip_eV is not None else 15.576),
            float(Zeff if Zeff is not None else 0.9),
        )
    return (
        float(Ip_eV_eff if Ip_eV_eff is not None else (Ip_eV if Ip_eV is not None else 15.6)),
        float(Zeff if Zeff is not None else 1.0),
    )


def _ion_model_family(rate: str) -> str:
    r = str(rate).lower()
    if r == "ppt_talebpour_i_legacy":
        return "legacy_simplified"
    if r == "ppt_talebpour_i_full":
        return "talebpour_molecule_full"
    if r == "popruzhenko_atom_i_full":
        return "popruzhenko_atom"
    return "other"

# ----------------- Factory (species is REQUIRED now) -----------------
def make_Wfunc(model_or_conf, ion_conf, omega0_SI: float, n0: float):
    """
    生成电离速率函数 Wfunc(inp)。本实现要求 ion_conf.species 非空。
    - 每个物种 'rate' 决定模型与输入域：
        'ppt_talebpour_i_legacy' | 'ppt_talebpour_i_full' | 'popruzhenko_atom_i_full' | 'mpa_fact' | 'off'（均为 I 域）
    - 顶层 time_mode: 'full' | 'qs_peak' | 'qs_mean' | 'qs_mean_esq'
      （仅挂到 Wfunc 供 evolve_rho_time 读取；不改变函数签名）
    - 顶层 integrator: 'rk4' | 'euler' （仅在 time_mode='full' 时生效）
    返回：
      - Wfunc(inp)  →  总速率（按物种 fraction 线性加权后，再封顶）
      - Wfunc._expects ∈ {'uses_E','uses_I','none'}
      - Wfunc._species_entries：每个物种 {'name','fraction','W_s','tag'}
    """
    W_cap_global = float(getattr(ion_conf, "W_cap", W_CAP_DEFAULT))
    species = getattr(ion_conf, "species", None)
    assert species and isinstance(species, (list, tuple)) and len(species) > 0, \
        "简化后的 ionization 需要提供非空的 species[]"

    # PPT/Popruzhenko 周期平均：相位采样（给 *_i 用）
    _phase_count = int(getattr(ion_conf, "cycle_avg_samples", 64))
    _phase = _np.linspace(0.0, _np.pi, max(8, _phase_count), endpoint=False)
    _cos_abs = _np.abs(_np.cos(_phase))

    # 顶层默认参数（可被每个 species 覆盖）
    W_scale_def   = float(getattr(ion_conf, "W_scale", 1.0))

    def _mk_W_by_rate(rate, sp, W_cap):
        rate = str(rate).lower()
        # 物种级参数（可覆盖顶层）
        W_scale   = float(_g(sp, "W_scale",  W_scale_def))

        if rate in ("ppt_talebpour_i_legacy", "ppt_talebpour_i_full"):
            name = str(_g(sp, "name", "")).strip()
            Ip_raw = _g(sp, "Ip_eV", None)
            Ip_eff = _g(sp, "Ip_eV_eff", None)
            Zeff_raw = _g(sp, "Zeff", None)
            l = int(_g(sp, "l", 0))
            m = int(_g(sp, "m", 0))
            Ip_use, Zeff_use = _talebpour_defaults(name=name, Ip_eV=Ip_raw, Ip_eV_eff=Ip_eff, Zeff=Zeff_raw)
            max_terms = int(_g(sp, "max_terms", 4096))
            sum_rel_tol = float(_g(sp, "sum_rel_tol", 1e-9))

            def W_s(I_SI, Ip_eV=Ip_use, Zeff=Zeff_use, l=l, m=m, W_cap=W_cap, W_scale=W_scale,
                    rate=rate, max_terms=max_terms, sum_rel_tol=sum_rel_tol):
                if rate == "ppt_talebpour_i_full":
                    W_mean = cycle_average_ppt_talebpour_full_from_I(
                        I_SI, n0=n0, omega0_SI=omega0_SI, Ip_eV=Ip_eV, Zeff=Zeff, l=l, m=m,
                        samples=_phase_count, max_terms=max_terms, sum_rel_tol=sum_rel_tol, W_cap=W_cap
                    )
                else:
                    W_mean = cycle_average_ppt_talebpour_legacy_from_I(
                        I_SI, n0=n0, Ip_eV=Ip_eV, Zeff=Zeff, l=l, m=m, samples=_phase_count, W_cap=W_cap
                    )
                if W_scale != 1.0:
                    W_mean = W_mean * W_scale
                return xp.nan_to_num(xp.clip(W_mean, 0.0, W_cap), nan=0.0, posinf=W_cap, neginf=0.0)

            tag = "uses_I"

        elif rate == "popruzhenko_atom_i_full":
            Ip = float(_g(sp, "Ip_eV", 15.6))
            Z = int(_g(sp, "Z", 1))
            max_terms = int(_g(sp, "max_terms", 4096))
            sum_rel_tol = float(_g(sp, "sum_rel_tol", 1e-9))

            def W_s(I_SI, Ip_eV=Ip, Z=Z, max_terms=max_terms, sum_rel_tol=sum_rel_tol,
                    W_cap=W_cap, W_scale=W_scale):
                W_mean = cycle_average_popruzhenko_atom_full_from_I(
                    I_SI, n0=n0, omega0_SI=omega0_SI, Ip_eV=Ip_eV, Z=Z, samples=_phase_count,
                    max_terms=max_terms, sum_rel_tol=sum_rel_tol, W_cap=W_cap
                )
                if W_scale != 1.0:
                    W_mean = W_mean * W_scale
                return xp.nan_to_num(xp.clip(W_mean, 0.0, W_cap), nan=0.0, posinf=W_cap, neginf=0.0)

            tag = "uses_I"

        elif rate == "mpa_fact":
            ell = int(_g(sp, "ell", 8));
            Imp = float(_g(sp, "I_mp", 1e18))

            def W_s(I_SI, ell=ell, Imp=Imp, W_cap=W_cap):
                W = _W_mpa_factorial(I_SI, omega0_SI, Imp, ell, W_cap)
                if W_scale != 1.0: W = W * W_scale
                return xp.nan_to_num(xp.clip(W, 0.0, W_cap), nan=0.0, posinf=W_cap, neginf=0.0)
            tag = "uses_I"
        else:  # 'off' or unknown
            def W_s(inp):
                return xp.zeros_like(inp, dtype=(xp.float32 if xp.asarray(inp).dtype == xp.complex64 else xp.float64))

            tag = "none"
        return W_s, tag

    entries, flags, species_meta = [], set(), []
    for sp in species:
        name = str(_g(sp, "name", f"sp{len(species_meta)}"))
        frac = float(_g(sp, "fraction", 1.0))
        W_cap = float(_g(sp, "W_cap", W_cap_global))
        rate = _resolve_rate(sp, ion_conf)
        W_s, tag = _mk_W_by_rate(rate, sp, W_cap)
        entries.append((frac, W_s, tag))
        model_family = _ion_model_family(rate)
        ip = _g(sp, "Ip_eV", None)
        ip_eff = _g(sp, "Ip_eV_eff", None)
        z = _g(sp, "Z", None)
        zeff = _g(sp, "Zeff", None)
        l = _g(sp, "l", None)
        m = _g(sp, "m", None)
        if rate in ("ppt_talebpour_i_legacy", "ppt_talebpour_i_full"):
            ip, zeff = _talebpour_defaults(name=name, Ip_eV=ip, Ip_eV_eff=ip_eff, Zeff=zeff)
            ip_eff = ip
        print(
            f"[ionization] species={name} rate={rate} family={model_family} "
            f"Ip_eV={ip} Ip_eV_eff={ip_eff} Z={z} Zeff={zeff} l={l} m={m} "
            f"cycle_avg_samples={_phase_count} time_mode={getattr(ion_conf, 'time_mode', '')} "
            f"integrator={getattr(ion_conf, 'integrator', 'rk4')}"
        )
        if str(name).upper() in ("N2", "O2") and rate == "popruzhenko_atom_i_full":
            print("Warning: using atomic Popruzhenko FULL model for molecular species; this is an atomic proxy, not strict molecular model.")

        species_meta.append({
            "name": name, "fraction": frac, "W_s": W_s, "tag": tag,
            "rate": rate, "family": model_family,
            "Ip_eV": ip, "Ip_eV_eff": ip_eff, "Z": z, "Zeff": zeff, "l": l, "m": m,
        })
        flags.add(tag)

    if "uses_E" in flags and "uses_I" in flags:
        raise ValueError("Ionization species 输入域冲突：存在既需要 |E| 又需要 I 的通道，请统一。")

    def Wfunc(inp):
        Wtot = None
        for frac, W_s, _tag in entries:
            Ws = W_s(inp)
            Wtot = Ws * frac if Wtot is None else (Wtot + frac * Ws)
        cap = float(getattr(ion_conf, "W_cap", W_cap_global))
        return xp.nan_to_num(xp.clip(Wtot, 0.0, cap), nan=0.0, posinf=cap, neginf=0.0)

    Wfunc._expects = "uses_I" if "uses_I" in flags else ("uses_E" if "uses_E" in flags else "none")
    Wfunc._species_entries = tuple(species_meta)
    Wfunc._species_fractions = tuple([m["fraction"] for m in species_meta])

    # 新增：把时间模式与积分器挂到 Wfunc（evolve_rho_time 会读取）
    Wfunc._time_mode  = str(getattr(ion_conf, "time_mode", "")).lower()      # "", "full","qs_peak","qs_mean","qs_mean_esq"
    Wfunc._integrator = str(getattr(ion_conf, "integrator", "rk4")).lower()  # "rk4"|"euler"
    return Wfunc

def _ion_input_domain(ion_conf):
    """
    返回 'I'。当前保留电离模型均是 I 域。
    对已移除模型给出友好报错，避免后续域错配导致崩溃。
    """
    species = getattr(ion_conf, "species", None)
    assert species and len(species) > 0, "要求提供 species[]"
    for sp in species:
        _ = _resolve_rate(sp, ion_conf)
    return "I"


# ----------------- rho(t) evolution (species-resolved + quasi-static/full) -----------------
def evolve_rho_time(input_array,dt: float,N0: float,beta_rec: float,Wfunc, *,
                    quasi_static_time: bool = False,time_stat: str = "peak",mean_clip_frac: float = 1e-3,
                    expects: str | None = None):
    """
    物种分辨的电子密度演化：
      - time_mode（由 Wfunc._time_mode 控制；若空则使用显式参数）：
        'full' → 全时域；'qs_peak'/'qs_mean'/'qs_mean_esq' → 准稳态
      - integrator（仅 'full' 下生效）：'rk4'（默认）或 'euler'
    接口保持与旧版一致。
    """
    # 覆盖时间模式（若 Wfunc 挂了 _time_mode）
    tm = str(getattr(Wfunc, "_time_mode", "")).lower()
    if tm in ("qs_peak","qs_mean","qs_mean_esq"):
        quasi_static_time = True
        time_stat = {"qs_peak":"peak","qs_mean":"mean","qs_mean_esq":"mean_esq"}[tm]
    elif tm == "full":
        quasi_static_time = False

    inp = _as_real_like(input_array, like=input_array)   # [Nt, Ny, Nx]
    Nt  = int(inp.shape[0])
    rdtype = xp.float32 if inp.dtype == xp.complex64 else xp.float64
    expects = (expects or getattr(Wfunc, "_expects", "E")).upper()

    # ===== 物种路径（推荐；本简化版假定一定存在） =====
    sp_entries = getattr(Wfunc, "_species_entries", None)
    if sp_entries and len(sp_entries) > 0:
        fracs = xp.asarray([max(0.0, float(s["fraction"])) for s in sp_entries], dtype=rdtype)
        ssum  = float(fracs.sum())
        if not xp.isfinite(ssum) or ssum <= 0.0:
            fracs = xp.ones_like(fracs) / float(len(sp_entries))
        else:
            fracs = fracs / ssum
        N0_j_list = [float(N0) * float(fracs[j]) for j in range(len(sp_entries))]

        # ---- 全时域推进 ----
        if not quasi_static_time:
            Wt_list = [ s["W_s"](inp).astype(rdtype, copy=False) for s in sp_entries ]  # 每个：[Nt,Ny,Nx]
            beta_list = [ float(beta_rec) * float(N0_j) for N0_j in N0_j_list ]
            u_list = [ xp.zeros_like(inp, dtype=rdtype) for _ in sp_entries ]  # ρ_j/N0_j

            mode = str(getattr(Wfunc, "_integrator", "rk4")).lower()
            for it in range(Nt - 1):
                if mode == "rk4":
                    dt_step = dt
                    for j in range(len(sp_entries)):
                        u  = u_list[j][it]
                        W1 = Wt_list[j][it]
                        W4 = Wt_list[j][it+1]
                        W2 = 0.5 * (W1 + W4)
                        W3 = W2
                        betaN = beta_list[j]
                        k1 = W1*(1.0 - u) - betaN*(u*u)
                        u2 = u + 0.5*dt_step*k1
                        k2 = W2*(1.0 - u2) - betaN*(u2*u2)
                        u3 = u + 0.5*dt_step*k2
                        k3 = W2*(1.0 - u3) - betaN*(u3*u3)
                        u4 = u + dt_step*k3
                        k4 = W4*(1.0 - u4) - betaN*(u4*u4)
                        u_next = u + (dt_step/6.0)*(k1 + 2*k2 + 2*k3 + k4)
                        _minmax_inplace(u_next, 0.0, 1.0)
                        u_list[j][it+1] = u_next
                else:
                    _DRHO_FRAC = globals().get("DRHO_FRAC", 0.05)
                    Wmax_frame = max(float(xp.max(Wt_list[j][it])) for j in range(len(sp_entries)))
                    max_du_dt  = max(Wmax_frame + max(beta_list), 0.0)
                    if (max_du_dt <= 0.0) or (not math.isfinite(max_du_dt)):
                        n_sub = 1
                    else:
                        val = dt * max_du_dt / max(_DRHO_FRAC, 1e-6)
                        n_sub = int(val) if val == int(val) else int(val) + 1
                        n_sub = 1 if n_sub < 1 else (1000 if n_sub > 1000 else n_sub)
                    dt_sub = dt / n_sub
                    for _ in range(n_sub):
                        for j in range(len(sp_entries)):
                            u  = u_list[j][it]
                            Wf = Wt_list[j][it]
                            betaN = beta_list[j]
                            du = dt_sub * (Wf * (1.0 - u) - betaN * (u * u))
                            u  = u + du
                            _minmax_inplace(u, 0.0, 1.0)
                            u_list[j][it] = u
                    for j in range(len(sp_entries)):
                        u_list[j][it+1] = u_list[j][it]

            rho_list = [ (u_list[j] * float(N0_j_list[j])).astype(rdtype, copy=False) for j in range(len(sp_entries)) ]
            rho_sum  = rho_list[0]
            for j in range(1, len(sp_entries)):
                rho_sum = rho_sum + rho_list[j]

            # 合成诊断速率（按分数加权）
            Wt_total = Wt_list[0] * float(fracs[0])
            for j in range(1, len(sp_entries)):
                Wt_total = Wt_total + Wt_list[j] * float(fracs[j])
            return rho_sum, Wt_total

        # ---- 准稳态：为每个物种构建 Wc_j 并解析推进 ----
        stat = str(time_stat).lower()
        Wc_list = []

        if stat in ("peak", "max"):
            for s in sp_entries:
                Wt_full = s["W_s"](inp).astype(rdtype, copy=False)
                Wc_list.append(xp.max(Wt_full, axis=0))

        elif stat in ("mean", "avg"):
            for s in sp_entries:
                Wt_full = s["W_s"](inp).astype(rdtype, copy=False)
                if mean_clip_frac and mean_clip_frac > 0.0:
                    peak_inp = xp.max(inp, axis=0) + 1e-30
                    mask = inp >= (mean_clip_frac * peak_inp)[None, ...]
                    Nt_tot = inp.shape[0]
                    Wc = (Wt_full * mask).sum(axis=0) / Nt_tot
                    # cnt  = xp.maximum(mask.sum(axis=0), 1)    这两行注释替代上面的全部
                    # Wc   = (Wt_full * mask).sum(axis=0) / cnt
                else:
                    Wc = xp.mean(Wt_full, axis=0)
                Wc_list.append(Wc.astype(rdtype, copy=False))

        elif stat in ("mean_esq", "rms_e", "e_rms"):
            if expects == "E":
                Esq_mean = xp.mean(inp * inp, axis=0)  # <|E|^2>
                E_rms    = xp.sqrt(Esq_mean + 0.0)
                for s in sp_entries:
                    Wc_list.append(s["W_s"](E_rms).astype(rdtype, copy=False))
            else:  # expects == "I"
                I_mean = xp.mean(inp, axis=0)
                for s in sp_entries:
                    Wc_list.append(s["W_s"](I_mean).astype(rdtype, copy=False))
        else:
            raise ValueError("time_stat 只能是 'peak' | 'mean' | 'mean_Esq'")

        # 解析推进 ρ_j(t)=N0_j(1-exp(-Wc_j t))
        t_idx = (xp.arange(1, Nt + 1, dtype=rdtype) * float(dt))[:, None, None]
        rho_list = [ float(N0_j_list[j]) * (1.0 - xp.exp(-Wc_list[j][None, ...] * t_idx)) for j in range(len(sp_entries)) ]
        rho_sum  = rho_list[0]
        for j in range(1, len(sp_entries)):
            rho_sum = rho_sum + rho_list[j]
        rho_sum = xp.asarray(rho_sum, dtype=rdtype)

        # 合成 Wt（广播到全时间轴）
        Wt_total = Wc_list[0] * float(fracs[0])
        for j in range(1, len(sp_entries)):
            Wt_total = Wt_total + Wc_list[j] * float(fracs[j])
        Wt_total = xp.broadcast_to(Wt_total[None, ...], inp.shape).astype(rdtype, copy=False)

        return rho_sum, Wt_total

    # ===== 兼容：若没有 species（不建议）则退回旧行为 =====
    if not quasi_static_time:
        Wt = Wfunc(inp)
        u = xp.zeros_like(inp, dtype=rdtype)           # u = rho/N0
        _DRHO_FRAC = globals().get("DRHO_FRAC", 0.05)
        betaN0 = float(beta_rec) * float(N0)
        for it in range(Nt - 1):
            Wmax = float(xp.max(Wt[it])); max_du_dt = Wmax + betaN0
            if (max_du_dt <= 0.0) or (not xp.isfinite(max_du_dt)):
                n_sub = 1
            else:
                val = dt * max_du_dt / max(_DRHO_FRAC, 1e-6)
                n_sub = int(val) if val == int(val) else int(val) + 1
                n_sub = 1 if n_sub < 1 else (1000 if n_sub > 1000 else n_sub)
            dt_sub = dt / n_sub
            u_t = u[it]
            for _ in range(n_sub):
                du = dt_sub * (Wt[it] * (1.0 - u_t) - betaN0 * (u_t * u_t))
                u_t = u_t + du
                _minmax_inplace(u_t, 0.0, 1.0)
            u[it + 1] = u_t
        rho = (u * float(N0)).astype(rdtype, copy=False)
        return rho, Wt

    # ---- Quasi-static (no species) ----
    stat = str(time_stat).lower()
    if stat in ("peak", "max"):
        Wt_full = Wfunc(inp).astype(rdtype, copy=False)
        Wc = xp.max(Wt_full, axis=0)
        Wt = xp.broadcast_to(Wc[None, ...], inp.shape)
    elif stat in ("mean", "avg"):
        Wt_full = Wfunc(inp).astype(rdtype, copy=False)
        if mean_clip_frac and mean_clip_frac > 0.0:
            peak_inp = xp.max(inp, axis=0) + 1e-30
            mask = inp >= (mean_clip_frac * peak_inp)[None, ...]
            cnt  = xp.maximum(mask.sum(axis=0), 1)
            Wc   = (Wt_full * mask).sum(axis=0) / cnt
        else:
            Wc = xp.mean(Wt_full, axis=0)
        Wt = xp.broadcast_to(Wc[None, ...], inp.shape)
    elif stat in ("mean_esq", "rms_e", "e_rms"):
        if expects == "E":
            Esq_mean = xp.mean(inp * inp, axis=0)
            E_rms = xp.sqrt(Esq_mean + 0.0)
            Wc = Wfunc(E_rms).astype(rdtype, copy=False)
        else:
            I_mean = xp.mean(inp, axis=0)
            Wc = Wfunc(I_mean).astype(rdtype, copy=False)
        Wt = xp.broadcast_to(Wc[None, ...], inp.shape)
    else:
        raise ValueError("time_stat 只能是 'peak' | 'mean' | 'mean_Esq'")

    t_idx = (xp.arange(1, Nt + 1, dtype=rdtype) * float(dt))[:, None, None]
    rho_t = float(N0) * (1.0 - xp.exp(-Wc[None, ...] * t_idx))
    return rho_t.astype(rdtype, copy=False), Wt
