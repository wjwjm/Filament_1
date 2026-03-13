# raman.py
from __future__ import annotations
from .device import xp
from types import SimpleNamespace as _NS
import numpy as _np


# ------------------------- 轻量工具 -------------------------
def _as_obj(cfg):
    """Allow dict or object; return object with attribute access."""
    return _NS(**cfg) if isinstance(cfg, dict) else cfg

def _get(cfg, key, default=None):
    """Safe getter for dict or object."""
    return cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)

def _heaviside_like(t):
    return (t >= 0).astype(xp.float64)


# ------------------------- 核生成 -------------------------
def make_raman_kernel(t, cfg) -> xp.ndarray:
    """
    生成时间域拉曼核 h(t)，只随 t 变化；返回 shape=[Nt] 的数组（xp.ndarray，float64）。
    - model="rot_sinexp": h(t) = ((ω_R^2+Γ_R^2)/ω_R) * e^{-Γ_R t} * sin(ω_R t) * u(t)
        首选字段: omega_R, Gamma_R
        若缺省：omega_R = 2π/T_R（把 T_R 当“周期”而非时间常数）；Gamma_R = 1/T2
    - model="exp"/"debye": h(t) = (1/T_R) * e^{-t/T_R} * u(t)
        字段: T_R（时间常数）
    核做 1 阶归一化（∑ h dt = 1），以便延迟 Kerr 的比例仅由 f_R 决定，而不是核面积漂移。
    """
    cfg = _as_obj(cfg)
    model = str(_get(cfg, "model", "rot_sinexp")).lower()

    t = xp.asarray(t, dtype=xp.float64)
    dt = float(t[1] - t[0])
    tt = xp.maximum(t, 0.0)  # 因果核

    if model in ("rot_sinexp", "rot-sinexp", "rot", "sinexp"):
        omega_R = _get(cfg, "omega_R", None)
        Gamma_R = _get(cfg, "Gamma_R", None)

        if omega_R is None:
            T_R = float(max(_get(cfg, "T_R", 8.4e-12), 1e-30))  # 文献常用“周期”≈8.4 ps
            omega_R = 2.0 * _np.pi / T_R
        else:
            omega_R = float(omega_R)

        if Gamma_R is None:
            T2 = float(max(_get(cfg, "T2", 8.0e-11), 1e-30))  # 去相干时间 ~ 80 ps
            Gamma_R = 1.0 / T2
        else:
            Gamma_R = float(Gamma_R)

        # 未归一化核（随后按面积归一）
        pref = (omega_R * omega_R + Gamma_R * Gamma_R) / max(omega_R, 1e-30)
        h_raw = pref * xp.exp(-Gamma_R * tt) * xp.sin(omega_R * tt) * _heaviside_like(t)

        area = float(xp.sum(h_raw) * dt)
        if not _np.isfinite(area) or area <= 0.0:
            area = 1.0
        h = h_raw / area
        # 数值修正
        area2 = float(xp.sum(h) * dt)
        if _np.isfinite(area2) and abs(area2 - 1.0) > 1e-3:
            h = h / (area2 + 1e-30)
        return h.astype(xp.float64)

    # 指数（Debye）核：h = e^{-t/T_R}/T_R * u(t)，天然已归一
    elif model in ("exp", "debye"):
        T_R = float(max(_get(cfg, "T_R", 8.4e-12), 1e-30))
        h = xp.exp(-tt / T_R) * (1.0 / T_R) * _heaviside_like(t)
        # 数值安全：再轻微校正到 ∑h dt ≈ 1
        area = float(xp.sum(h) * dt)
        if _np.isfinite(area) and abs(area - 1.0) > 1e-3:
            h = h / (area + 1e-30)
        return h.astype(xp.float64)

    # 未知模型：返回零核，避免崩溃
    return xp.zeros_like(t, dtype=xp.float64)


def precompute_kernel_fft(h: xp.ndarray) -> xp.ndarray:
    """频域核 H(Ω)（与 xp 后端一致的 FFT），沿时间轴变换。"""
    return xp.fft.fft(h.astype(xp.float64), axis=0)


# ------------------------- 强度卷积（I ⊗ h_R） -------------------------
def raman_convolve_intensity(I, H_w=None, *, method="iir", dt=None, T2=None, T_R=None, chunk_pixels=65536):
    """
    计算 IR = (h_R * I)(t)，仅沿 t 轴卷积；I/IR 形状 [Nt,Ny,Nx]。
    - method="iir": 省显存时域递推。
        * 若同时提供 T2 与 T_R ：按“旋转拉曼核”递推，参数映射
              Γ_R = 1/T2,   ω_R = 2π / T_R   （注意：此处 T_R 解释为“周期”）
          与核生成式保持一致，避免 2π 漏乘。
        * 若仅提供 T_R ：按 Debye 核递推（h = e^{-t/T_R}/T_R）。
    - method="fft": 频域法，需要预先给 H_w = FFT(h)；空间按列分块避免 OOM。
    """
    Nt, Ny, Nx = I.shape
    dtype = I.dtype
    method = str(method).lower()

    # ---------------- IIR：旋转拉曼（sin-exp）/ Debye 两种递推 ----------------
    if method == "iir":
        if dt is None:
            raise ValueError("raman_convolve_intensity(method='iir') 需要 dt")

        I2 = I.reshape(Nt, Ny * Nx)
        ctype = xp.complex64 if dtype == xp.float32 else xp.complex128
        IR = xp.empty_like(I2, dtype=dtype)

        if (T2 is not None) and (T_R is not None):
            # ===== 旋转拉曼核：h(t) = pref * e^{-Γ t} sin(ω t) =====
            T2 = float(max(T2, 1e-30))
            T_R = float(max(T_R, 1e-30))  # 解释为“周期”
            Gamma = 1.0 / T2
            omega = 2.0 * _np.pi / T_R  # ★ 避免 2π 漏乘

            # 用 xp 标量数组保证 dtype 与后端一致
            a = xp.array(Gamma - 1j * omega, dtype=ctype)          # Γ - iω
            r = xp.exp(-a * dt)                                    # e^{-a dt}
            c = (1.0 - r) / a                                      # (1 - r)/a

            # k 使 ∫h dt = 1：Im(k/a) = 1 -> k = 1 / Im(1/a)
            inv_a = 1.0 / a                                        # complex scalar
            denom = xp.imag(inv_a) + xp.array(1e-300, dtype=inv_a.real.dtype)
            k = 1.0 / denom                                        # real scalar (xp array)

            S = xp.zeros((Ny * Nx,), dtype=ctype)
            for n in range(Nt):
                S = r * S + c * I2[n]
                # k 是实标量；(k*S) 与 S 同 dtype → 取虚部后再 cast 回 dtype
                IR[n] = xp.imag(k * S).astype(dtype, copy=False)

            return IR.reshape(Nt, Ny, Nx)

        # ===== 仅 T_R 给出：Debye 核 IIR =====
        if T_R is None:
            raise ValueError("raman_convolve_intensity(method='iir'): 请提供 (T2 与 T_R) 或至少 T_R")
        T_R = float(max(T_R, 1e-30))
        r = _np.exp(-dt / T_R)  # 纯实数（用 numpy 算标量没关系）
        c = (1.0 - r)

        S = xp.zeros((Ny * Nx,), dtype=dtype)
        for n in range(Nt):
            S = r * S + c * I2[n]
            IR[n] = S  # 直接实数
        return IR.reshape(Nt, Ny, Nx)

    # ---------------- FFT：按空间列分块 ----------------
    if H_w is None:
        raise ValueError("raman_convolve_intensity(method='fft') 需要 H_w=FFT(h)")

    I2 = I.reshape(Nt, Ny * Nx)
    out = xp.empty_like(I2, dtype=dtype)
    chunk = int(max(1, min(chunk_pixels, Ny * Nx)))

    # 匹配频域 dtype
    H_w = H_w.astype(xp.complex64 if dtype == xp.float32 else xp.complex128, copy=False)[:, None]

    for j in range(0, Ny * Nx, chunk):
        sl = I2[:, j:j + chunk].astype(H_w.dtype, copy=False)
        Rw = xp.fft.fft(sl, axis=0)
        Rw *= H_w
        ish = xp.fft.ifft(Rw, axis=0).real
        out[:, j:j + chunk] = ish.astype(dtype, copy=False)

    return out.reshape(Nt, Ny, Nx)
