from __future__ import annotations
from .device import xp
#lin_propagator(kperp2,k0,dz) 构建角谱传播因子。

#step_linear(E, prop) 对 [Nt, Ny, Nx] 的每个时间切片执行 2D FFT 传播。
# def lin_propagator(kperp2, k0: float, dz: float):
#     """Angular spectrum paraxial propagator for a single z-step."""
#     return xp.exp(1j * (-kperp2) * dz / (2.0 * k0))
#
# def step_linear(E, prop):
#     """Apply 2D (x,y) FFT-based linear propagation to a [Nt, Ny, Nx] field E."""
#     Ew = xp.fft.fft2(E, axes=(-2, -1))
#     Ew *= prop  # broadcast over time axis
#     return xp.fft.ifft2(Ew, axes=(-2, -1))


def lin_propagator(kperp2, k0, dz, *, ctype=None):
    if ctype is None:
        ctype = xp.complex64  # 或由上层传 E.dtype
    rdtype = xp.float32 if ctype == xp.complex64 else xp.float64
    onej   = xp.array(1j, dtype=ctype)

    k2 = xp.asarray(kperp2, dtype=rdtype)
    phase = (-k2) * (dz / (2.0 * float(k0)))           # rdtype
    return xp.exp(onej * phase).astype(ctype)          # ctype

def step_linear(E, prop):
    if prop.dtype != E.dtype:
        prop = prop.astype(E.dtype, copy=False)
    Ew = xp.fft.fft2(E, axes=(-2, -1))
    Ew *= prop
    return xp.fft.ifft2(Ew, axes=(-2, -1))
