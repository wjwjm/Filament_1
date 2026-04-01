from __future__ import annotations
from .device import xp
from .constants import c0

def _K02_from_nw(omega_abs, n_w):
    return (n_w * omega_abs / c0)**2  # shape [Nt]

def step_linear_full_factorized(E, K02_w, kperp2, dz_eff):
    """
    线性半步/整步（factorize版）：对每个 ω 切片做 2D FFT，乘 exp(i kz dz)，再 ifft2。
    内存低（不需 [Nt,Ny,Nx] 级别的prop缓存）。
    """
    ctype  = E.dtype
    rdtype = xp.float32 if ctype == xp.complex64 else xp.float64
    onej   = xp.array(1j, dtype=ctype)

    Ew = xp.fft.fft(E, axis=0)  # [Nt,Ny,Nx]
    Nt = Ew.shape[0]
    kperp2 = xp.asarray(kperp2, dtype=rdtype)

    for i in range(Nt):
        kval = xp.asarray(K02_w[i], dtype=rdtype)
        rad = kval - kperp2
        kz = xp.sqrt(rad.astype(ctype))  # ctype
        prop2d = xp.exp(onej * kz * dz_eff)

        S = xp.fft.fft2(Ew[i], axes=(-2,-1))
        S *= prop2d
        Ew[i] = xp.fft.ifft2(S, axes=(-2,-1))
    return xp.fft.ifft(Ew, axis=0)

# linear_full.py
def step_linear_full_3d(E, K02_w, kperp2, dz_eff, ctype=None):
    """
    正确的 UPPE 线性步（时间+空间频域）:
        EwΩxy = FFT_t( FFT2_xy(E) )
        kz(Ω,k⊥) = sqrt( K0^2(Ω) - k⊥^2 )    # K0^2(Ω) = (n(ω)·ω/c)^2
        EwΩxy *= exp( i · kz(Ω,k⊥) · dz_eff )
        E = IFFT2_xy( IFFT_t(EwΩxy) )
    E: [Nt, Ny, Nx]
    K02_w: [Nt]  (Ω 轴对应的 (n(ω)ω/c)^2 )
    kperp2: [Ny, Nx]
    """
    if ctype is None:
        ctype = E.dtype
    onej = xp.array(1j, dtype=ctype)

    # 1) t-FFT 到 Ω 域
    Ew = xp.fft.fft(E, axis=0)                 # [Nt,Ny,Nx]

    # 2) xy-FFT 到 (kx,ky)
    Ew = xp.fft.fft2(Ew, axes=(-2, -1))        # [Nt,Ny,Nx]

    # 3) 传播相位
    #    注意 K02_w 广播到 [Nt,1,1]，kperp2 广播到 [1,Ny,Nx]
    kz2 = K02_w[:, None, None] - kperp2[None, :, :]
    kz  = xp.sqrt(xp.maximum(kz2, 0.0))
    Ew *= xp.exp(onej * kz * dz_eff).astype(ctype, copy=False)

    # 4) 回到时空域
    Ew = xp.fft.ifft2(Ew, axes=(-2, -1))
    E  = xp.fft.ifft(Ew, axis=0)

    return E.astype(ctype, copy=False)

def _smooth_taper(r, r0, r1):
    """cos^2 软过渡: r<=r0 -> 1, r>=r1 -> 0, 其间平滑下降"""
    w = (r - r0) / (r1 - r0 + 1e-30)
    w = xp.clip(w, 0.0, 1.0)
    return xp.cos(0.5 * xp.pi * w) ** 2

def step_linear_full_3d_chunked(
    E, K02_w, kperp2, dz_eff,
    chunk_t=32, ctype=None,
    mask_frac=0.985,        # 建议 0.98~0.995
    ramp_frac=0.02,         # 软过渡宽度 (相对于 k0)
    evanescent_mode="decay" # "decay" 或 "zero"
):
    """
    UPPE 线性步：FFT_t -> 按 Ω 分片做 FFT2_xy / 相位乘 / IFFT2_xy -> IFFT_t
    - 对 k_perp≈k0(ω) 采用 cos^2 软掩膜，避免硬截断导致的振铃；
    - 对虚衍射（k_perp>k0）:
        evanescent_mode="decay": 乘 e^{-κ dz}（物理上指数衰减）
        evanescent_mode="zero" : 直接置零
    """
    if ctype is None:
        ctype = E.dtype
    onej = xp.array(1j, dtype=ctype)

    Nt = E.shape[0]
    Ew = xp.fft.fft(E, axis=0)  # [Nt,Ny,Nx]

    kperp = xp.sqrt(kperp2)[None, :, :]  # [1,Ny,Nx]

    for s in range(0, Nt, int(max(1, chunk_t))):
        e = Ew[s:s+chunk_t]                         # [m,Ny,Nx]
        e = xp.fft.fft2(e, axes=(-2, -1))          # -> (Ω,kx,ky)

        K02_blk = K02_w[s:s+chunk_t][:, None, None] # [m,1,1]
        k0_blk  = xp.sqrt(xp.maximum(K02_blk, 0.0)) # [m,1,1]

        # 软掩膜: 相对截止半径 rp = k_perp / k0(ω)
        rp = kperp / (k0_blk + 1e-30)               # [m,Ny,Nx]
        r0 = float(mask_frac)
        r1 = float(mask_frac + ramp_frac)
        wmask = xp.where(rp <= r0, 1.0,
                 xp.where(rp >= r1, 0.0, _smooth_taper(rp, r0, r1)))
        wmask = wmask.astype(ctype, copy=False)

        kz2 = K02_blk - kperp2[None, :, :]          # [m,Ny,Nx]
        real_mask = kz2 >= 0.0
        kz   = xp.sqrt(xp.maximum(kz2, 0.0))
        ph_r = xp.exp(onej * kz * dz_eff).astype(ctype, copy=False)

        if evanescent_mode == "decay":
            kappa = xp.sqrt(xp.maximum(-kz2, 0.0))
            ph_e = xp.exp(-kappa * dz_eff).astype(ctype, copy=False)
        else:  # "zero"
            ph_e = xp.zeros_like(ph_r)

        phase = xp.where(real_mask, ph_r, ph_e)
        e *= (phase * wmask)

        e = xp.fft.ifft2(e, axes=(-2, -1))
        Ew[s:s+chunk_t] = e

    E = xp.fft.ifft(Ew, axis=0)
    return E.astype(ctype, copy=False)