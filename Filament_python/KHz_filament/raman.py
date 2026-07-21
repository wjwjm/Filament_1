# raman.py
from __future__ import annotations
from .device import xp
from .constants import c0, eps0
from types import SimpleNamespace as _NS
import numpy as _np
import time as _time


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
    rot_sinexp 保持解析 prefactor（由 ω_R, Γ_R 决定）；不引入基于 f_R 的幅值缩放。
    """
    cfg = _as_obj(cfg)
    model = str(_get(cfg, "model", "rot_sinexp")).lower()

    t = xp.asarray(t, dtype=xp.float64)
    dt = float(t[1] - t[0])
    tt = xp.maximum(t, 0.0)  # 因果核

    if model in ("isaacs_rot_sinexp",):
        omega_R = _get(cfg, "omega_R", None)
        Gamma_R = _get(cfg, "Gamma_R", None)
        if omega_R is None or Gamma_R is None:
            raise ValueError("isaacs_rot_sinexp requires explicit omega_R and Gamma_R")
        omega_R = float(omega_R)
        Gamma_R = float(Gamma_R)
        if omega_R <= 0.0 or Gamma_R < 0.0:
            raise ValueError("isaacs_rot_sinexp requires omega_R > 0 and Gamma_R >= 0")
        pref = (omega_R * omega_R + Gamma_R * Gamma_R) / omega_R
        return (pref * xp.exp(-Gamma_R * tt) * xp.sin(omega_R * tt) * _heaviside_like(t)).astype(xp.float64)

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

        # 解析核（保持论文形式的 prefactor）
        pref = (omega_R * omega_R + Gamma_R * Gamma_R) / max(omega_R, 1e-30)
        h = pref * xp.exp(-Gamma_R * tt) * xp.sin(omega_R * tt) * _heaviside_like(t)
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


def precompute_kernel_fft(h: xp.ndarray, n_fft: int | None = None) -> xp.ndarray:
    """频域核 H(Ω)（与 xp 后端一致的 FFT），沿时间轴变换。"""
    return xp.fft.fft(h.astype(xp.float64), n=n_fft, axis=0)


def raman_convolve_intensity_fft_linear(I, h, *, dt, chunk_pixels=65536):
    """Causal linear convolution ``(h * I) dt`` along the time axis."""
    if dt is None:
        raise ValueError("raman_convolve_intensity_fft_linear requires dt")
    Nt, Ny, Nx = I.shape
    h = xp.asarray(h, dtype=I.dtype)
    Nh = int(h.shape[0])
    if Nh < 1:
        raise ValueError("raman causal kernel must contain at least one sample")
    n_fft = Nt + Nh - 1
    ctype = xp.complex64 if I.dtype == xp.float32 else xp.complex128
    H = xp.fft.fft(h.astype(ctype, copy=False), n=n_fft)[:, None]
    I2 = I.reshape(Nt, Ny * Nx)
    out = xp.empty_like(I2, dtype=I.dtype)
    chunk = int(max(1, min(chunk_pixels, Ny * Nx)))
    for j in range(0, Ny * Nx, chunk):
        values = I2[:, j:j + chunk].astype(ctype, copy=False)
        conv = xp.fft.ifft(xp.fft.fft(values, n=n_fft, axis=0) * H, axis=0).real
        out[:, j:j + chunk] = (conv[:Nt] * dt).astype(I.dtype, copy=False)
    return out.reshape(Nt, Ny, Nx)


# ------------------------- 强度卷积（I ⊗ h_R） -------------------------
def resolve_raman_rot_params(*, T2=None, T_R=None, omega_R=None, Gamma_R=None):
    """统一解析旋转拉曼参数：优先 omega_R/Gamma_R，其次 T_R/T2。"""
    if omega_R is None:
        if T_R is None:
            raise ValueError("resolve_raman_rot_params: omega_R 或 T_R 至少提供一个")
        T_R = float(max(T_R, 1e-30))
        omega_R = 2.0 * _np.pi / T_R
    else:
        omega_R = float(omega_R)

    if Gamma_R is None:
        if T2 is None:
            raise ValueError("resolve_raman_rot_params: Gamma_R 或 T2 至少提供一个")
        T2 = float(max(T2, 1e-30))
        Gamma_R = 1.0 / T2
    else:
        Gamma_R = float(Gamma_R)

    return float(omega_R), float(Gamma_R)


def raman_convolve_intensity(I, H_w=None, *, method="iir", dt=None, T2=None, T_R=None,
                             omega_R=None, Gamma_R=None, chunk_pixels=65536,
                             iir_sampling="legacy_right_hold"):
    """
    计算 IR = (h_R * I)(t)，仅沿 t 轴卷积；I/IR 形状 [Nt,Ny,Nx]。
    - method="iir": 省显存时域递推。
        * 若提供 (omega_R 与 Gamma_R) 或 (T2 与 T_R) ：按“旋转拉曼核”递推
              Γ_R = Gamma_R 或 1/T2,   ω_R = omega_R 或 2π/T_R
          与核生成式保持一致，避免 2π 漏乘。
        * 若仅提供 T_R ：按 Debye 核递推（h = e^{-t/T_R}/T_R）。
    - method="fft": 频域法，需要预先给 H_w = FFT(h)；空间按列分块避免 OOM。
    """
    Nt, Ny, Nx = I.shape
    dtype = I.dtype
    method = str(method).lower()

    if method == "fft":
        if H_w is None:
            raise ValueError("raman_convolve_intensity(method='fft') requires sampled causal kernel h")
        return raman_convolve_intensity_fft_linear(I, H_w, dt=dt, chunk_pixels=chunk_pixels)

    # ---------------- IIR：旋转拉曼（sin-exp）/ Debye 两种递推 ----------------
    if method == "iir":
        if dt is None:
            raise ValueError("raman_convolve_intensity(method='iir') 需要 dt")

        I2 = I.reshape(Nt, Ny * Nx)
        ctype = xp.complex64 if dtype == xp.float32 else xp.complex128
        IR = xp.empty_like(I2, dtype=dtype)

        if ((omega_R is not None) and (Gamma_R is not None)) or ((T2 is not None) and (T_R is not None)):
            # ===== 旋转拉曼核：h(t) = pref * e^{-Γ t} sin(ω t) =====
            omega, Gamma = resolve_raman_rot_params(T2=T2, T_R=T_R, omega_R=omega_R, Gamma_R=Gamma_R)

            # 用 xp 标量数组保证 dtype 与后端一致
            a = xp.array(Gamma - 1j * omega, dtype=ctype)          # Γ - iω
            r = xp.exp(-a * dt)                                    # e^{-a dt}
            c = (1.0 - r) / a                                      # (1 - r)/a

            # k 使 ∫h dt = 1：Im(k/a) = 1 -> k = 1 / Im(1/a)
            inv_a = 1.0 / a                                        # complex scalar
            denom = xp.imag(inv_a) + xp.array(1e-300, dtype=inv_a.real.dtype)
            k = 1.0 / denom                                        # real scalar (xp array)

            S = xp.zeros((Ny * Nx,), dtype=ctype)
            sampling = str(iir_sampling).lower()
            if sampling != "legacy_right_hold":
                IR[0] = xp.asarray(0.0, dtype=dtype)
                if sampling == "left_hold":
                    for n in range(1, Nt):
                        S = r * S + c * I2[n - 1]
                        IR[n] = xp.imag(k * S).astype(dtype, copy=False)
                elif sampling == "trapezoidal":
                    for n in range(1, Nt):
                        S = r * S + c * 0.5 * (I2[n - 1] + I2[n])
                        IR[n] = xp.imag(k * S).astype(dtype, copy=False)
                elif sampling == "exact_piecewise_linear":
                    c1 = c - (1.0 - r * (1.0 + a * dt)) / (a * a * dt)
                    c0 = c - c1
                    for n in range(1, Nt):
                        S = r * S + c0 * I2[n - 1] + c1 * I2[n]
                        IR[n] = xp.imag(k * S).astype(dtype, copy=False)
                else:
                    raise ValueError(
                        "iir_sampling must be legacy_right_hold, left_hold, trapezoidal, "
                        "or exact_piecewise_linear"
                    )
                return IR.reshape(Nt, Ny, Nx)
            for n in range(Nt):
                S = r * S + c * I2[n]
                # k 是实标量；(k*S) 与 S 同 dtype → 取虚部后再 cast 回 dtype
                IR[n] = xp.imag(k * S).astype(dtype, copy=False)

            return IR.reshape(Nt, Ny, Nx)

        # ===== 仅 T_R 给出：Debye 核 IIR =====
        if T_R is None:
            raise ValueError("raman_convolve_intensity(method='iir'): 请提供 (omega_R 与 Gamma_R) 或 (T2 与 T_R) 或至少 T_R")
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


def isaacs_raman_stage(E, *, Omega, dt, omega0, n0, n_R, omega_R, Gamma_R,
                       method="iir", chunk_pixels=65536,
                       iir_sampling="exact_piecewise_linear",
                       return_response=True, return_energy=True):
    """Evaluate one Eq. (27) stage with exactly one Raman convolution."""
    started = _time.perf_counter()
    rdtype = xp.float32 if E.dtype == xp.complex64 else xp.float64
    intensity = (0.5 * float(eps0) * float(c0) * float(n0) * xp.abs(E) ** 2).astype(rdtype, copy=False)
    if method == "fft":
        tau = xp.arange(E.shape[0], dtype=xp.float64) * float(dt)
        kernel = make_raman_kernel(tau, {
            "model": "isaacs_rot_sinexp", "omega_R": omega_R, "Gamma_R": Gamma_R,
        })
        response = raman_convolve_intensity(
            intensity, kernel, method="fft", dt=dt, chunk_pixels=chunk_pixels)
    else:
        response = raman_convolve_intensity(
            intensity, method="iir", dt=dt, omega_R=omega_R, Gamma_R=Gamma_R,
            chunk_pixels=chunk_pixels, iir_sampling=iir_sampling)
    product = response.astype(rdtype, copy=False) * E
    ctype = E.dtype
    # Evaluate (1/omega0)*d_tau as one dimensionless spectral multiplier.
    # Forming Omega*FFT(I_R*A) first can overflow complex64 even though the
    # final Eq. (27) prefactor makes the physical RHS finite.
    multiplier = (1j * xp.asarray(Omega, dtype=rdtype) / float(omega0)).astype(ctype, copy=False)
    normalized_derivative = xp.fft.ifft(
        multiplier[:, None, None] * xp.fft.fft(product, axis=0), axis=0)
    prefactor = (float(omega0) / float(c0)) * float(n_R)
    rhs = (1j * prefactor * product - prefactor * normalized_derivative).astype(ctype, copy=False)
    result = {
        "rhs": rhs,
        "rhs_l2_norm": float(xp.linalg.norm(rhs)),
        "IR_max": float(xp.max(xp.abs(response))),
        "convolution_count": 1,
    }
    if return_response:
        result["I_R"] = response.astype(rdtype, copy=False)
    if return_energy:
        intensity_scale = float(xp.max(xp.abs(intensity)))
        if intensity_scale > 0.0:
            # Normalize before multiplying I_R*dI.  The unscaled float32
            # intermediate is O(I^2/dt) and can overflow even though the
            # integrated Eq. (10) energy is finite.
            derivative_increment_norm = xp.fft.ifft(
                (1j * xp.asarray(Omega, dtype=rdtype) * float(dt))[:, None, None]
                * xp.fft.fft(intensity, axis=0), axis=0).real / intensity_scale
            response_norm = response / intensity_scale
            energy_scale = (float(n_R) / float(c0)) * intensity_scale * intensity_scale
            u_signed = energy_scale * xp.sum(
                response_norm * derivative_increment_norm, axis=0)
        else:
            u_signed = xp.zeros(intensity.shape[1:], dtype=rdtype)
        result["u_R_signed"] = u_signed.astype(rdtype, copy=False)
        result["q_R_positive"] = xp.maximum(-u_signed, 0.0).astype(rdtype, copy=False)
    result["walltime_s"] = float(_time.perf_counter() - started)
    return result


def isaacs_raman_field_rhs(E, *, Omega, dt, omega0, n0, n_R, omega_R, Gamma_R,
                           method="iir", chunk_pixels=65536,
                           iir_sampling="exact_piecewise_linear"):
    """Full complex rotational RHS derived from Isaacs Eq. (27)."""
    return isaacs_raman_stage(
        E, Omega=Omega, dt=dt, omega0=omega0, n0=n0, n_R=n_R,
        omega_R=omega_R, Gamma_R=Gamma_R, method=method,
        chunk_pixels=chunk_pixels, iir_sampling=iir_sampling,
        return_response=False, return_energy=False,
    )["rhs"]


def isaacs_raman_signed_energy_density(E, *, Omega, dt, n0, n_R, omega_R, Gamma_R,
                                       method="iir", chunk_pixels=65536,
                                       iir_sampling="exact_piecewise_linear"):
    """Return local signed Eq. (10) exchange ``u_R`` and deposition ``q_R``."""
    stage = isaacs_raman_stage(
        E, Omega=Omega, dt=dt, omega0=1.0, n0=n0, n_R=n_R,
        omega_R=omega_R, Gamma_R=Gamma_R, method=method,
        chunk_pixels=chunk_pixels, iir_sampling=iir_sampling,
        return_response=False, return_energy=True,
    )
    return stage["u_R_signed"], stage["q_R_positive"]


def apply_isaacs_raman_operator_step(E, dz, *, Omega, dt, omega0, n0, n_R,
                                     omega_R, Gamma_R, integrator="heun",
                                     method="iir", chunk_pixels=65536,
                                     iir_sampling="exact_piecewise_linear",
                                     return_diagnostics=False, stage1=None,
                                     transverse_cell_area=1.0):
    """Opt-in full Isaacs Raman step; Heun recomputes I and I_R at stage two."""
    kwargs = dict(
        Omega=Omega, dt=dt, omega0=omega0, n0=n0, n_R=n_R,
        omega_R=omega_R, Gamma_R=Gamma_R, method=method,
        chunk_pixels=chunk_pixels, iir_sampling=iir_sampling,
    )
    started = _time.perf_counter()
    before_fluence = xp.sum(
        0.5 * float(eps0) * float(c0) * float(n0) * xp.abs(E) ** 2,
        axis=0) * float(dt)
    stage1 = stage1 or isaacs_raman_stage(E, **kwargs)
    k1, q1 = stage1["rhs"], stage1["q_R_positive"]
    if str(integrator).lower() == "euler":
        result = (E + float(dz) * k1).astype(E.dtype, copy=False)
        stage2 = None
        target_stage1 = float(dz) * q1
        target_stage2 = xp.zeros_like(target_stage1)
        target_heun = target_stage1
    elif str(integrator).lower() == "heun":
        predictor = (E + float(dz) * k1).astype(E.dtype, copy=False)
        stage2 = isaacs_raman_stage(predictor, **kwargs)
        result = (E + 0.5 * float(dz) * (k1 + stage2["rhs"])).astype(E.dtype, copy=False)
        target_stage1 = float(dz) * q1
        target_stage2 = float(dz) * stage2["q_R_positive"]
        target_heun = 0.5 * (target_stage1 + target_stage2)
    else:
        raise ValueError("Raman full-operator integrator must be 'euler' or 'heun'")
    if return_diagnostics:
        after_fluence = xp.sum(
            0.5 * float(eps0) * float(c0) * float(n0) * xp.abs(result) ** 2,
            axis=0) * float(dt)
        actual_local = before_fluence - after_fluence
        local_residual_map = xp.abs(actual_local - target_heun) / xp.maximum(
            target_heun, xp.maximum(before_fluence * 1e-15, 1e-300))
        area = float(transverse_cell_area)
        target_global = float(xp.sum(target_heun) * area)
        actual_global = float(xp.sum(actual_local) * area)
        before_global = float(xp.sum(before_fluence) * area)
        global_residual = abs(actual_global - target_global) / max(
            target_global, before_global * 1e-15, 1e-300)
        diagnostics = {
            "target_local_fluence_loss_stage1": target_stage1,
            "target_local_fluence_loss_stage2": target_stage2,
            "target_local_fluence_loss_heun": target_heun,
            "target_local_fluence_loss": target_heun,
            "target_global_energy_loss_J": target_global,
            "actual_local_fluence_loss": actual_local,
            "actual_global_energy_loss_J": actual_global,
            "local_closure_residual": float(xp.max(local_residual_map)),
            "global_closure_residual": float(global_residual),
            "rhs_l2_norm_stage1": stage1["rhs_l2_norm"],
            "rhs_l2_norm_stage2": stage2["rhs_l2_norm"] if stage2 else 0.0,
            "IR_max_stage1": stage1["IR_max"],
            "IR_max_stage2": stage2["IR_max"] if stage2 else 0.0,
            "I_R_stage1": stage1.get("I_R"),
            "I_R_stage2": stage2.get("I_R") if stage2 else None,
            "convolution_count": stage1["convolution_count"] + (stage2["convolution_count"] if stage2 else 0),
            "operator_walltime_s": float(_time.perf_counter() - started),
            "finite": bool(xp.all(xp.isfinite(result))) and math_isfinite(global_residual),
            "clipping_count": 0,
        }
        return result, diagnostics
    return result


def math_isfinite(value):
    return bool(_np.isfinite(float(value)))
