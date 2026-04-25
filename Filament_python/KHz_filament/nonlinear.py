from __future__ import annotations
from .device import xp
from .constants import rho_crit

# ——数值安全上限——
ALPHA_DZ_CAP = 20.0   # 限制 alpha*dz 的最大值，避免 exp(-0.5*alpha*dz) 下溢/溢出
PHASE_CAP    = 5e3    # 可选：限制相位幅值，避免极端相位（通常不需要很小）

def kerr_phase(I, k0, n2, dz,*,rdtype=None):
    if rdtype is None: rdtype = I.dtype
    return (float(k0) * float(n2) * I * float(dz)).astype(rdtype, copy=False)
    # return k0 * n2 * I * dz

def kerr_phase_from_deltan(delta_n, k0, dz, *, rdtype=None):
    """Kerr phase from explicit refractive-index perturbation Δn(t,x,y)."""
    if rdtype is None:
        rdtype = delta_n.dtype
    return (float(k0) * delta_n * float(dz)).astype(rdtype, copy=False)

def plasma_phase(rho, k0, omega0, dz, *, rdtype=None):
    # rhoc = rho_crit(omega0)
    # return -k0 * (rho / (2.0 * rhoc + 1e-300)) * dz
    from .constants import rho_crit
    if rdtype is None: rdtype = rho.dtype
    rhoc = float(rho_crit(omega0))
    return (-float(k0) * (rho / (2.0 * rhoc)) * float(dz)).astype(rdtype, copy=False)

def ib_alpha(rho, sigma_ib,*,rdtype=None):
    if rdtype is None: rdtype = rho.dtype
    a = (float(sigma_ib) * rho).astype(rdtype, copy=False)
    return  xp.clip(a, 0.0, xp.inf)
    # return xp.maximum(0.0, sigma_ib) * xp.maximum(rho, 0.0)

def apply_nonlinear(E, phase, alpha, dz,*, dn_gas=None, k0=None):

    ctype  = E.dtype
    rdtype = xp.float32 if ctype == xp.complex64 else xp.float64
    onej   = xp.array(1j, dtype=ctype)

    # 叠加慢时间 Δn_gas
    if dn_gas is not None and k0 is not None:
        phase = (phase + float(k0) * xp.asarray(dn_gas, dtype=rdtype) * float(dz)).astype(rdtype, copy=False)
        #phase = phase + (k0 * dn_gas)[None, ...] * dz
    else:
        phase = xp.asarray(phase, dtype=rdtype)

    # 相位可选裁剪（避免极端大相位造成 exp 里 NaN）
    #phase = xp.clip(phase, -PHASE_CAP, PHASE_CAP)

    # 吸收指数因子裁剪在 alpha*dz <= ALPHA_DZ_CAP
    #alpha = xp.clip(alpha, 0.0, ALPHA_DZ_CAP / max(dz, 1e-30))
    #att = xp.exp(-0.5 * alpha * dz)

    # 组合
    # E = E * xp.exp(1j * phase) * att
    # 把潜在的 NaN/Inf 清掉（保守做法，便于继续往下跑）
    # E = xp.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)
    alpha = xp.asarray(alpha, dtype=rdtype)
    alpha = xp.clip(alpha, 0.0, xp.inf)

    E *= xp.exp(onej * phase).astype(ctype, copy=False)
    E *= xp.exp((-0.5 * alpha * float(dz))).astype(ctype, copy=False)
    return E

# def apply_self_steepening(E, omega0, Omega):
#     """
#     一阶自陡峭算符：在频域乘 (1 + iΩ/ω0)。低代价，建议配宽带启用。
#     """
#     ctype  = E.dtype
#     onej   = xp.array(1j, dtype=ctype)
#     Ew = xp.fft.fft(E, axis=0)
#     # S = (1.0 + 1j * (Omega / max(omega0, 1e-12)))[:, None, None]
#     S = (1.0 + onej * (xp.asarray(Omega, dtype=xp.float64) / float(omega0)))[:, None, None]
#     Ew *= S.astype(ctype, copy=False)
#     return xp.fft.ifft(Ew, axis=0)

# def shock_intensity(I, Omega, omega0):
#     """
#     计算 I_shock = Re( F^{-1}{ (1 + iΩ/ω0) * F{I} } )
#     作为 Kerr 相位的等效强度；不直接作用在 E 上，因而不改变能量模长。
#     """
#     Iw = xp.fft.fft(I, axis=0)                  # [Nt,Ny,Nx]
#     S  = (1.0 + 1j * (Omega / float(omega0)))   # [Nt]
#     Rw = Iw * S[:, None, None]
#     Ishock = xp.real(xp.fft.ifft(Rw, axis=0))
#     return xp.nan_to_num(Ishock, nan=0.0, posinf=0.0, neginf=0.0)

def operator_correct_scalar(Q, Omega, omega0, *, dt=None, method="auto", chunk_pixels=None):
    """
    通用包络算子修正（实标量场）：
      Q_eff = Q - (1/ω0) ∂Q/∂t   <=>   F{Q_eff} = (1 + iΩ/ω0) F{Q}
    与 shock_intensity 保持同一符号与 FFT 约定。
    """
    Nt, Ny, Nx = Q.shape
    dtype = Q.dtype

    if method in ("tdiff", "auto"):
        if dt is None and method == "tdiff":
            raise ValueError("operator_correct_scalar(method='tdiff') 需要 dt")
        if method == "tdiff" or dt is not None:
            dQdt = (xp.roll(Q, -1, axis=0) - xp.roll(Q, 1, axis=0)) / (2.0 * (dt if dt else 1.0))
            Qeff = Q - dQdt / float(omega0)
            return Qeff.astype(dtype, copy=False)

    if chunk_pixels is None:
        chunk_pixels = min(Ny * Nx, 65536)

    Q2 = Q.reshape(Nt, Ny * Nx)
    out = xp.empty_like(Q2, dtype=dtype)
    mult = (1.0 + 1j * (Omega.astype(dtype) / float(omega0)))[:, None]

    for j in range(0, Ny * Nx, chunk_pixels):
        sl = Q2[:, j:j + chunk_pixels].astype(xp.complex64 if dtype == xp.float32 else xp.complex128, copy=False)
        Rw = xp.fft.fft(sl, axis=0)
        Rw *= mult
        qeff = xp.fft.ifft(Rw, axis=0).real
        out[:, j:j + chunk_pixels] = qeff.astype(dtype, copy=False)

    return out.reshape(Nt, Ny, Nx)


def shock_intensity(I, Omega, omega0, *, dt=None, method="auto", chunk_pixels=None):
    """
    自陡峭强度修正：I_shock = I - (1/ω0) ∂I/∂t  （等价于频域乘子 1 + iΩ/ω0）
    参数：
      - method: "auto" | "tdiff" | "fft"
      - dt:     时间步长（仅 tdiff 需要）
      - chunk_pixels: 频域法时的空间分块大小（Ny*Nx 里每块的像素数）
    形状：
      I: [Nt, Ny, Nx]（float32/float64）
      Omega: [Nt]
    返回：
      Ishock: [Nt, Ny, Nx]，与 I 同 dtype
    模式选择：
    method="tdiff"：时域中心差分，几乎不占显存，推荐在 GPU 上使用。
    method="fft"：频域法，但分块处理，避免一次性分配数百 MB。
    method="auto"：若提供了 dt，直接走 tdiff；否则走 fft（保留兼容性）。
    """
    return operator_correct_scalar(I, Omega, omega0, dt=dt, method=method, chunk_pixels=chunk_pixels)
