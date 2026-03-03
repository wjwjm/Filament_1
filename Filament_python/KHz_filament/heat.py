from __future__ import annotations
from .device import xp

Q_CAP = 1e20  # W/m^3 的安全上限，避免热功率溢出（可按需调）

def heat_Q_per_z(Wt, I, rho, Ui: float, alpha_ib, dt: float, N0: float):
    # Q = Ui*W*(N0 - rho) + alpha_ib*I；所有项裁剪，最后再时间积分
    term_ion = Ui * Wt * xp.clip(N0 - rho, 0.0, N0)
    term_ib  = alpha_ib * I
    Q = term_ion + term_ib
    Q = xp.clip(Q, -Q_CAP, Q_CAP)           # 极端保护（理论上不该为负）
    Q = xp.nan_to_num(Q, nan=0.0, posinf=Q_CAP, neginf=0.0)
    Qslice = xp.sum(Q, axis=0) * dt
    return Qslice


def _nan_inf_to_num_inplace(arr, nan=0.0, posinf=0.0, neginf=0.0):
    """
    原地清理 NaN / ±Inf。arr 可为 numpy/cupy，内部统一到当前 xp 后端。
    不返回新数组，直接修改并返回同一引用。
    """
    a = xp.asarray(arr)
    # NaN -> nan
    m = xp.isnan(a)
    if xp.any(m):
        a[m] = nan
    # +Inf -> posinf
    m = xp.isposinf(a)
    if xp.any(m):
        a[m] = posinf
    # -Inf -> neginf
    m = xp.isneginf(a)
    if xp.any(m):
        a[m] = neginf
    return a

def diffuse_dn_gas(dn_gas, Q2D, D_gas, delta_t_pulse, kperp2, gamma_heat):
    """
    一次脉间扩散：Δn(t+Δt) = e^{-D k^2 Δt} * Δn + gamma_heat * Q2D
    - dn_gas: [Ny,Nx]  上一脉后的 Δn（numpy 或 cupy 都可）
    - Q2D:    [Ny,Nx]  本脉沉积能量密度时间积分（numpy 或 cupy 都可）
    说明：
      * 先用 xp.asarray 统一到当前后端
      * NaN/Inf 清理用原地方法，避免 xp.nan_to_num 的后端/显存问题
      * 尽量原地运算，降低额外内存
    """
    # 统一到当前后端
    dn = xp.asarray(dn_gas)
    Q  = xp.asarray(Q2D)
    k2 = xp.asarray(kperp2)

    # 注入项：gamma_heat * Q2D，并原地清理异常值
    delta_n_inj = gamma_heat * Q
    _nan_inf_to_num_inplace(delta_n_inj, nan=0.0, posinf=0.0, neginf=0.0)

    # 频域扩散：Δn_hat <- e^{-D k^2 Δt} * Δn_hat + FFT(delta_n_inj)
    dn_hat = xp.fft.fft2(dn)
    damp   = xp.exp(-float(D_gas) * k2 * float(delta_t_pulse))  # [Ny,Nx]
    dn_hat *= damp                                              # 原地
    dn_hat += xp.fft.fft2(delta_n_inj)

    dn_new = xp.real(xp.fft.ifft2(dn_hat))
    return dn_new
