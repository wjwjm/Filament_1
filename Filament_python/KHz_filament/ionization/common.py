from __future__ import annotations

import numpy as _np
from ..device import xp
from ..constants import eps0, c0

I_CAP_DEFAULT = 1e19
W_CAP_DEFAULT = 1e16
DRHO_FRAC = 0.25
ION_LUT_SCHEMA_VERSION = "ion-lut-v1"
_ION_LUT_MEMORY_CACHE = {}


def _nan_inf_to_num_inplace(arr, nan=0.0, posinf=None, neginf=None, *, chunk_elems=1 << 20):
    a = xp.asarray(arr)
    flat = a.reshape(-1)
    n = flat.size
    dtype = a.dtype
    if posinf is None:
        posinf_val = xp.finfo(dtype).max if dtype.kind == "f" else 0
    else:
        posinf_val = dtype.type(posinf)
    if neginf is None:
        neginf_val = -xp.finfo(dtype).max if dtype.kind == "f" else 0
    else:
        neginf_val = dtype.type(neginf)
    nan_val = dtype.type(nan) if nan is not None else None
    step = int(max(1024, chunk_elems))
    for j in range(0, n, step):
        v = flat[j : j + step]
        if nan_val is not None:
            m_nan = xp.isnan(v)
            v[m_nan] = nan_val
            del m_nan
        m_pos = xp.isposinf(v)
        if xp.any(m_pos):
            v[m_pos] = posinf_val
        del m_pos
        m_neg = xp.isneginf(v)
        if xp.any(m_neg):
            v[m_neg] = neginf_val
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
    if lo is not None:
        xp.maximum(arr, xp.asarray(lo, dtype=arr.dtype), out=arr)
    if hi is not None:
        xp.minimum(arr, xp.asarray(hi, dtype=arr.dtype), out=arr)
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


def intensity(E, n0: float, I_cap: float = I_CAP_DEFAULT):
    I = 0.5 * eps0 * c0 * n0 * (xp.abs(E) ** 2)
    return _minmax_inplace(I, 0.0, I_cap)


def field_amplitude_from_intensity(I, n0: float):
    return xp.sqrt(_minmax_inplace(2.0 * I / (eps0 * c0 * n0), 0.0))


def _to_numpy(a):
    if xp.__name__ == "cupy":
        return xp.asnumpy(a)
    return _np.asarray(a)
