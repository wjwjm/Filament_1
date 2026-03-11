from __future__ import annotations
from .device import xp
from .linear import lin_propagator, step_linear, step_linear_bk_nee_factorized
from .ionization import (
    intensity as inten_ion,
    make_Wfunc,
    field_amplitude_from_intensity,
    evolve_rho_time,_ion_input_domain
)
from .nonlinear import kerr_phase, plasma_phase, ib_alpha, apply_nonlinear,shock_intensity
from .heat import heat_Q_per_z
from .linear_full import step_linear_full_factorized, step_linear_full_3d
from .air_dispersion import n_of_omega
from .constants import c0
from .raman import make_raman_kernel, precompute_kernel_fft, raman_convolve_intensity
from .diagnostics import intensity, pulse_energy, second_moment_radius,_fwhm_time_1d,parabola_peak,_fwhm_diameter_xy_center

def _linear_phase_per_meter(linear_model, k0, axes, K02_w=None, omega0=None, nee_denom_floor=1e-4):
    """返回线性传播子对应的 |kz| (rad/m) 的 max 值，用于估计 Δφ_linear = kz_max * dz"""
    if linear_model == "paraxial":
        # 相位：exp(i * (-k⊥^2) dz / (2 k0))，幅角/米 = k⊥^2 / (2 k0)
        kz_abs_max = float(xp.max(axes.kperp2) / (2.0 * k0))
    elif linear_model == "bk_nee":
        # Brabec–Krausz NEE 线性项主导的横向衍射相位估计
        if omega0 is None:
            omega0 = float(getattr(axes, "omega0"))
        rel = axes.Omega / float(omega0)
        denom = 1.0 + rel
        denom_abs = xp.maximum(xp.abs(denom), float(nee_denom_floor))
        denom_sign = xp.where(denom >= 0.0, 1.0, -1.0)
        denom = denom_sign * denom_abs
        kz_abs_max = float(xp.max(xp.abs(axes.kperp2[None, ...] / (2.0 * k0 * denom[:, None, None]))))
    else:  # UPPE
        # K02_w = (n(ω) ω / c)^2, kz = sqrt(K02_w - k⊥^2)；取实部的最大值
        if K02_w is None:
            # 退化：用 n0 * ω/c 的中心值做下界估计（更保守些）
            omega_tot = axes.omega0 + axes.Omega
            k0w = (axes.n_w * omega_tot / c0) if hasattr(axes, "n_w") else (k0 * (omega_tot / axes.omega0))
            kz = xp.sqrt(xp.maximum(k0w[:, None, None]**2 - axes.kperp2[None, ...], 0.0))
        else:
            kz = xp.sqrt(xp.maximum(K02_w - axes.kperp2[None, ...], 0.0))
        kz_abs_max = float(xp.max(xp.abs(xp.real(kz))))
    return kz_abs_max


# --- 轻量 CPU 侧 FWHM 计算：对 2D map 做圆平均，再找 0.5×峰值的半径 ---
def _fwhm_circular_cpu(map2d_cpu, x_cpu, y_cpu, floor_rel=1e-12, nbins=256):
    import numpy as np
    m = np.asarray(map2d_cpu, dtype=np.float64)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(m.max())
    if peak <= 0.0:
        return 0.0
    m = np.where(m < floor_rel * peak, 0.0, m)

    X, Y = np.meshgrid(x_cpu, y_cpu, indexing="xy")
    r = np.sqrt(X*X + Y*Y)
    rmax = float(r.max())
    if rmax <= 0:
        return 0.0
    bins = np.linspace(0.0, rmax, nbins+1)
    idx = np.digitize(r.ravel(), bins) - 1
    idx = np.clip(idx, 0, nbins-1)

    sumv = np.bincount(idx, weights=m.ravel(), minlength=nbins)
    cnt  = np.bincount(idx, minlength=nbins)
    prof = np.divide(sumv, np.maximum(cnt, 1), out=np.zeros_like(sumv), where=cnt>0)
    rmid = 0.5*(bins[:-1] + bins[1:])
    # 找到 prof 降到 0.5×peak 的位置做线性插值
    half = 0.5 * float(prof.max())
    below = np.where(prof <= half)[0]
    if below.size == 0:
        return 0.0
    i = int(below[0])
    if i == 0:
        r_half = float(rmid[0])
    else:
        x1, y1 = rmid[i-1], prof[i-1]
        x2, y2 = rmid[i],   prof[i]
        if y2 == y1:
            r_half = float(x2)
        else:
            r_half = float(x1 + (half - y1) * (x2 - x1) / (y2 - y1))
    return 2.0 * r_half  # 直径 FWHM

def propagate_one_pulse(
    E,
    *,
    kperp2,
    k0: float,
    omega0: float,
    dz: float,
    z_max: float,
    n0: float,
    n2: float,
    Ui: float,
    N0: float,
    ion_conf,          # IonizationConfig
    dn_gas=None,
    dt: float,
    axes=None, prop_conf=None, raman_conf=None,
    record_onaxis_rho_time: bool = True,
    record_every_z: int = 1,
):
    """极简稳定版：固定一步只做一次安全缩步（可选近焦加密），标准 Strang 分裂。
       产出统一的 diag 契约（见函数尾部）。"""
    import time
    import numpy as _np

    # ---------- dtype ----------
    ctype = E.dtype
    rdtype = xp.float32 if ctype == xp.complex64 else xp.float64
    rdtype_np = _np.float32 if ctype == xp.complex64 else _np.float64

    # ---------- 近焦缩步 ----------
    p = prop_conf
    dz_base = float(dz)
    use_focus_win = bool(getattr(p, "focus_window_step", False))
    z_center = getattr(p, "focus_center_m", None) or getattr(p, "z_focus_hint", None)
    z_half = float(getattr(p, "focus_halfwidth_m", 0.0))
    dz_focus = float(getattr(p, "dz_focus", dz_base))

    # ---------- 索引/几何 ----------
    Ny, Nx = E.shape[-2], E.shape[-1]
    y0, x0 = Ny // 2, Nx // 2
    t_arr = axes.t
    t0_idx = int(xp.argmin(xp.abs(t_arr)))

    save_every = max(1, int(record_every_z))
    save_count = 0


    # ---------- 线性分支 ----------
    linear_model = str(getattr(p, "linear_model", "uppe")).lower()
    use_uppe = (linear_model == "uppe")
    use_bk_nee = (linear_model == "bk_nee")

    if use_uppe:
        Omega = axes.Omega
        omega_tot = omega0 + Omega
        omega_safe = xp.where(xp.abs(omega_tot) < 1e-9 * omega0,
                              xp.sign(omega_tot) * 1e-9 * omega0,
                              omega_tot)
        n_w = n_of_omega(omega_safe,
                         P=getattr(p, "air_P", 101325.0),
                         T=getattr(p, "air_T", 293.15))
        K02_w = (n_w * omega_safe / c0) ** 2
        use_factor = bool(getattr(p, "full_linear_factorize", False))

    # ---------- 拉曼（延迟 Kerr + 可选吸收模型） ----------
    use_raman = bool(getattr(raman_conf, "enabled", False)) if (raman_conf is not None) else False
    raman_absorb_on = False
    absorption_model = "poynting"
    omega_R = Gamma_R = None
    tau_fwhm_cfg = None
    n_rot_frac = 0.99
    R0_mode = "mom"
    R0_fixed = None

    if use_raman:
        h = make_raman_kernel(axes.t, raman_conf)
        H_w = precompute_kernel_fft(h)
        fR = float(getattr(raman_conf, "f_R", 0.15))
        r_method = str(getattr(raman_conf, "method", "iir")).lower()
        r_chunk = int(getattr(raman_conf, "chunk_pixels", 65536))

        # 可选：仅关闭吸收，保留延迟 Kerr
        absorption_model = str(getattr(raman_conf, "absorption_model", "poynting")).lower()
        raman_absorb_on = bool(getattr(raman_conf, "absorption", True))

        # closed_form 需要的参数（都有默认）
        omega_R = float(getattr(raman_conf, "omega_R", 2.0 * _np.pi / 8.4e-12))
        Gamma_R = float(getattr(raman_conf, "Gamma_R", 1.0 / 8.0e-11))
        tau_fwhm_cfg = getattr(raman_conf, "tau_fwhm", None)
        n_rot_frac = float(getattr(raman_conf, "n_rot_frac", 0.99))
        R0_mode = str(getattr(raman_conf, "R0_mode", "mom")).lower()
        R0_fixed = float(getattr(raman_conf, "R0_fixed_m", 2.0e-4))
    else:
        H_w, fR, r_method, r_chunk = None, 0.0, "iir", 65536

    # ---------- 电离速率 ----------


    Wfunc = make_Wfunc(getattr(ion_conf, "model", "none"), ion_conf, omega0, n0)
    ion_input = getattr(Wfunc, "_expects", None)
    if ion_input in ("uses_E", "E"):
        ion_input = "E"
    elif ion_input in ("uses_I", "I"):
        ion_input = "I"
    else:
        ion_input = _ion_input_domain(ion_conf)  # 兜底
    ion_off = str(getattr(ion_conf, "model", "none")).lower() in ("none", "off", "zero") and not getattr(ion_conf, "species", None)

    # ---------- 基线能量 ----------
    I0 = intensity(E, n0)
    U0_baseline = float(pulse_energy(I0, dt, axes.dx, axes.dy)) + 1e-30
    energy_print_every = int(getattr(prop_conf, "energy_probe_every", 1))
    if energy_print_every > 0:
        print(f"[U] z={0.000:0.3f} m  U={U0_baseline: .3e} J  Δrel={0.00:.2f}%")

    # ---------- 诊断收集 ----------
    z_axis_list, U_z_list = [], []
    I_max_z_list, rho_max_z_list = [], []
    I_onaxis_max_z_list, I_center_t0_z_list = [], []
    w_mom_z_list, rho_onaxis_max_list = [], []

    E_dep_z_list, E_dep_rot_z_list = [], []  # ← 新增：拉曼沉积
    fwhm_plasma_z_list, fwhm_fluence_z_list = [], []
    rho_onaxis_time_list = [] if record_onaxis_rho_time else None
    I_onaxis_max_interp_list,alpha_R_mean_z_list,alpha_R_closed_z_list,IR_max_z_list = [],[],[],[]
    alpha_R_max_z_list = []
    # ---------- 主循环 ----------
    z = 0.0
    # Qacc 累积到面密度 J/m^2（把体功率密度在 t,z 两方向积分）
    Qacc = xp.zeros((Ny, Nx), dtype=rdtype)
    t0 = time.perf_counter()

    while z < z_max - 1e-16:
        dz_remain = z_max - z
        if use_focus_win and (z_center is not None) and (z_half > 0.0):
            z_mid = z + 0.5 * dz_base
            dz_try = min(dz_focus if abs(z_mid - float(z_center)) <= z_half else dz_base, dz_remain)
        else:
            dz_try = min(dz_base, dz_remain)

        # 线性半步
        if use_uppe:
            if use_factor:
                E = step_linear_full_factorized(E, K02_w, kperp2, dz_try / 2)
            else:
                E = step_linear_full_3d(E, K02_w, kperp2, dz_try / 2)
        elif use_bk_nee:
            E = step_linear_bk_nee_factorized(
                E,
                Omega=axes.Omega,
                kperp2=kperp2,
                k0=k0,
                omega0=omega0,
                dz=dz_try / 2,
                beta2=float(getattr(p, "nee_beta2", 0.0)),
                denom_floor=float(getattr(p, "nee_denom_floor", 1e-4)),
            )
        else:
            prop_xh = xp.sqrt(lin_propagator(kperp2, k0, dz_try, ctype=ctype)).astype(ctype)
            E = step_linear(E, prop_xh)

        # 非线性整步
        I = inten_ion(E, n0, I_cap=getattr(ion_conf, "I_cap", 1e19))

        # 延迟 Kerr：I_nl 用于 Kerr 相位；同时得到 IR 供后续吸收
        if use_raman:
            IR = raman_convolve_intensity(
                I, H_w if r_method == "fft" else None,
                method=r_method, dt=dt, T2=float(raman_conf.T2), T_R=float(raman_conf.T_R),
                chunk_pixels=r_chunk
            )
            I_nl = (1.0 - fR) * I + fR * IR
        else:
            IR = None
            I_nl = I

        if bool(getattr(p, "use_self_steepening", False)):
            I_kerr = shock_intensity(I_nl, axes.Omega, omega0, dt=dt,
                                     method=str(getattr(p, "self_steepening_method", "tdiff")).lower())
        else:
            I_kerr = I_nl

        # —— 电离/IB 吸收（分开记账）——
        if ion_off:
            rho = xp.zeros_like(I, dtype=rdtype)
            Wt  = xp.zeros_like(I, dtype=rdtype)
            alpha_ib = xp.zeros_like(I, dtype=rdtype)
        else:
            X = field_amplitude_from_intensity(I, n0) if ion_input == "E" else I
            rho, Wt = evolve_rho_time(
                X, dt, N0, ion_conf.beta_rec, Wfunc,
                quasi_static_time=bool(getattr(ion_conf, "quasi_static_time", False)),
                time_stat=str(getattr(ion_conf, "time_stat", "peak"))
            )
            xp.maximum(rho, 0.0, out=rho)
            xp.minimum(rho, N0, out=rho)


            # Drude IB：优先 nu_ei_const，退回 sigma_ib·rho
            sigma_ib = float(getattr(ion_conf, "sigma_ib", 0.0))
            nu_ei = getattr(ion_conf, "nu_ei_const", None)
            if nu_ei is not None:
                from .constants import e as qe, me, eps0 as _eps0, c0 as _c0
                sigma_ib = (qe * qe / (_eps0 * me * _c0)) * (float(nu_ei) / (float(nu_ei) ** 2 + omega0 ** 2))
            alpha_ib = ib_alpha(rho.astype(rdtype, copy=False), sigma_ib, rdtype=rdtype)
            xp.maximum(alpha_ib, 0.0, out=alpha_ib)

        # 电离吸收 α_ion（只参与传播与热，避免在 Q 中重复计 IB）
        I_floor = 1e-6 * float(getattr(ion_conf, "I_cap", 1e19))
        d_rho_dt = Wt * xp.clip(N0 - rho, 0.0, N0)
        alpha_ion = (Ui * d_rho_dt) / (I + I_floor)
        xp.maximum(alpha_ion, 0.0, out=alpha_ion)

        # —— 拉曼吸收 —— 两种模型二选一

        alpha_R_eff = 0.0                    # 标量/2D 等效吸收，用于传播
        E_dep_rot_step = 0.0                 # 本 z 步旋转拉曼总沉积能量 [J]
        IR_max = 0.0


        if use_raman and raman_absorb_on:
            # 文献参数（允许在 config 中覆盖）
            omega_R_use = float(getattr(raman_conf, "omega_R", 1.6e13))
            Gamma_R_use = float(getattr(raman_conf, "Gamma_R", 1.3e13))
            n_R = float(getattr(raman_conf, "n_R", 2.3e-23))  # m^2/W （2.3e-19 cm^2/W）

            # 因果核 Ω(τ)（离散化在 make_raman_kernel 里已实现 rot_sinexp）
            # 这里确保核用上述频率/阻尼
            h = make_raman_kernel(axes.t, dict(
                model="rot_sinexp", omega_R=omega_R_use, Gamma_R=Gamma_R_use
            ))
            H_w = precompute_kernel_fft(h)

            # 入射能量（用于能量守恒折算）
            U_in = float(pulse_energy(I, dt, axes.dx, axes.dy)) + 1e-30

            scheme = str(getattr(raman_conf, "absorption_model", "conv_deriv")).lower()
            if scheme == "poynting":
                # 兼容名：把 poynting 指到 conv_deriv（同一物理口径）
                scheme = "conv_deriv"

            if scheme in ("conv_deriv", "conv-deriv", "convolution"):
                # ===== (A) 文献式：u_R(τ) = (n_R/c) ∫ Ω(τ') [ I(τ-τ')·∂τ I(τ) ] dτ' =====
                # 计算 I_R = Ω * I（因果卷积）
                IR = raman_convolve_intensity(
                    I, H_w if r_method == "fft" else None,
                    method=r_method, dt=dt, T2=float(raman_conf.T2), T_R=float(raman_conf.T_R),
                    chunk_pixels=r_chunk
                )
                # ∂I/∂τ（中心差分）
                dIdt = xp.empty_like(I)
                dIdt[1:-1] = (I[2:] - I[:-2]) / (2.0 * dt)
                dIdt[0]    = (I[1] - I[0]) / dt
                dIdt[-1]   = (I[-1] - I[-2]) / dt

                # 经验门限（抑制远尾噪声）
                mask_frac = float(getattr(raman_conf, "abs_mask_frac", 5e-3))
                Ipk = xp.max(I, axis=0) + 1e-30
                mask = I >= (mask_frac * Ipk)[None, ...]

                # 按文献式累积：对 τ' 的离散卷积已经体现在 IR 中 → u_R ≈ (n_R/c) · IR · dI/dt
                w_R = (n0 / c0) * n_R * IR * dIdt               # W/m^3
                w_R = xp.where(mask, w_R, 0.0)

                # 脉内时间积分 → 体能量密度（正值，表示净注入分子转动）
                Q_rot_vol = xp.sum(xp.maximum(w_R, 0.0), axis=0) * dt   # J/m^3
                # 总沉积能量（再乘体元素体积在 z 上的长度）
                E_dep_rot_step = float(xp.sum(Q_rot_vol) * axes.dx * axes.dy * dz_try)

                # 能量守恒折成等效 α_R（用于传播）
                alpha_R_eff = E_dep_rot_step / (U_in * dz_try)
                alpha_R_eff = float(max(0.0, alpha_R_eff))
                # 单步上限（防抽干）
                max_alpha_dz = float(getattr(raman_conf, "max_alpha_dz", 1e-3))
                if alpha_R_eff * dz_try > max_alpha_dz:
                    alpha_R_eff = max_alpha_dz / dz_try

                IR_max = float(IR.max())

            elif scheme in ("alpha_local", "closed_form", "closed-form"):
                # ===== (B) 封闭式：α_R(x,y) = (n_R/(c τ_p)) I0 · bracket（时间常数，逐像素）=====
                # 本地峰值与本地脉宽（用 F/I0 的高斯关系，抗噪&可矢量化）
                I0_xy = xp.max(I, axis=0) + 1e-30                      # W/m^2
                F_xy  = xp.sum(I, axis=0) * dt                          # J/m^2
                C_gauss = _np.sqrt(4.0*_np.log(2.0)/_np.pi)             # ≈0.939
                tau_p_xy = C_gauss * (F_xy / I0_xy)                     # s

                phi_xy = omega_R_use * tau_p_xy
                g_xy   = Gamma_R_use * tau_p_xy
                bracket_xy = 1.0 - xp.exp(-g_xy) * (xp.cos(phi_xy) + (Gamma_R_use/omega_R_use) * xp.sin(phi_xy))

                alpha_map = (n_R / (c0 * tau_p_xy)) * I0_xy * bracket_xy   # 1/m
                alpha_map = xp.clip(alpha_map, 0.0, _np.inf)

                # 单步上限
                max_alpha_dz = float(getattr(raman_conf, "max_alpha_dz", 1e-3))
                alpha_map = xp.minimum(alpha_map, max_alpha_dz / dz_try)

                # 折成本步能量（对面积积分 × dz）
                E_dep_rot_step = float(xp.sum(alpha_map * F_xy) * axes.dx * axes.dy * dz_try)

                # 等效标量 α（用面积-能量加权平均，参与传播）
                # alpha_eff = (∫α F dA)/(∫F dA)
                denomF = float(xp.sum(F_xy) * axes.dx * axes.dy) + 1e-30
                alpha_R_eff = float(xp.sum(alpha_map * F_xy) * axes.dx * axes.dy) / denomF

            else:
                # 未知方案：关闭吸收，仅延迟 Kerr
                alpha_R_eff = 0.0
                E_dep_rot_step = 0.0

        # —— 把等效 α_R_eff 加入总吸收参与传播 ——
        alpha_total = alpha_ib + alpha_ion + float(alpha_R_eff)

        # 相位
        dphi_k = kerr_phase(I_kerr.astype(rdtype, copy=False), k0, n2, dz_try, rdtype=rdtype)
        dphi_p = plasma_phase(rho.astype(rdtype, copy=False), k0, omega0, dz_try, rdtype=rdtype)
        phase  = dphi_k + dphi_p

        E = apply_nonlinear(E, phase, alpha_total, dz_try, dn_gas=dn_gas, k0=k0)

        # —— 热沉积：分量分别记账 ——
        # 电离 + IB：Qslice (J/m^3)；注意仅用 IB alpha 进入 Q（电离能量用 Ui·W 计算）
        Qslice = heat_Q_per_z(Wt, I, rho, Ui, alpha_ib, dt, N0)
        Qacc += xp.asarray(Qslice, dtype=Qacc.dtype) * dz_try
        # Poynting 模式：如需把拉曼也积入慢时间面密度（J/m^2），可把体能量密度乘 dz_try 累起来
        if use_raman and raman_absorb_on and (absorption_model == "poynting") and (Q_rot_vol is not None):
            Qacc += xp.asarray(Q_rot_vol, dtype=Qacc.dtype) * dz_try

        # 第二个线性半步
        if use_uppe:
            if use_factor:
                E = step_linear_full_factorized(E, K02_w, kperp2, dz_try / 2)
            else:
                E = step_linear_full_3d(E, K02_w, kperp2, dz_try / 2)
        elif use_bk_nee:
            E = step_linear_bk_nee_factorized(
                E,
                Omega=axes.Omega,
                kperp2=kperp2,
                k0=k0,
                omega0=omega0,
                dz=dz_try / 2,
                beta2=float(getattr(p, "nee_beta2", 0.0)),
                denom_floor=float(getattr(p, "nee_denom_floor", 1e-4)),
            )
        else:
            prop_xh = xp.sqrt(lin_propagator(kperp2, k0, dz_try, ctype=ctype)).astype(ctype)
            E = step_linear(E, prop_xh)

        # --- 每步诊断 ---
        save_count += 1
        if (save_count % save_every) == 0 or (z + dz_try >= z_max - 1e-16):
            z_now = float(z + dz_try)
            z_axis_list.append(z_now)

            I_now = intensity(E, n0)
            U_now = float(pulse_energy(I_now, dt, axes.dx, axes.dy))
            U_z_list.append(U_now)
            I_max_z_list.append(float(I_now.max()))

            # on-axis(t)
            I_onax_t = I_now[:, y0, x0]
            I_onaxis_max_z_list.append(float(I_onax_t.max()))
            I_center_t0_z_list.append(float(I_onax_t[t0_idx]))
            kpk = int(xp.argmax(I_onax_t))
            if 0 < kpk < (I_onax_t.shape[0] - 1):
                I_onaxis_peak_interp = float(parabola_peak(I_onax_t[kpk - 1], I_onax_t[kpk], I_onax_t[kpk + 1]))
            else:
                I_onaxis_peak_interp = float(I_onax_t[kpk])
            I_onaxis_max_interp_list.append(I_onaxis_peak_interp)

            # 二阶矩束腰（对 t 积分后求 2D 矩）
            F2D = xp.trapz(I_now, dx=dt, axis=0)
            w_mom = second_moment_radius(
                I_now, axes.x, axes.y, dt=dt,
                frac_keep=getattr(prop_conf, "mom_frac_keep", 0.999),
                rel_floor=getattr(prop_conf, "mom_rel_floor", 1e-8),
            )
            w_mom_z_list.append(float(w_mom))

            # 电子密度统计
            rho_onax_t = rho[:, y0, x0].astype(rdtype, copy=False)
            rho_onaxis_max_list.append(float(xp.max(rho_onax_t)))
            if record_onaxis_rho_time:
                rho_onaxis_time_list.append(_np.array(xp.asnumpy(rho_onax_t)))

            rho_maxt = xp.max(rho, axis=0)
            rho_max_z_list.append(float(rho_maxt.max()))

            # 步内沉积能量（J）
            E_dep = float(xp.sum(Qslice) * axes.dx * axes.dy * dz_try)  # 电离+IB
            E_dep_z_list.append(E_dep)
            # 拉曼沉积
            if use_raman and raman_absorb_on:
                E_dep_rot = E_dep_rot_step
                alphaR_max = float(alpha_R_eff)
                alphaR_mean = float(alpha_R_eff)
            else:
                E_dep_rot = alphaR_max = alphaR_mean = 0.0

            E_dep_rot_z_list.append(E_dep_rot)
            alpha_R_max_z_list.append(alphaR_max)
            alpha_R_mean_z_list.append(alphaR_mean)

            # FWHM：等离子体通道 与 能量密度
            fwhm_plasma = _fwhm_diameter_xy_center(rho_maxt, axes, x0, y0)
            fwhm_flu    = _fwhm_diameter_xy_center(F2D,      axes, x0, y0)
            fwhm_plasma_z_list.append(float(fwhm_plasma))
            fwhm_fluence_z_list.append(float(fwhm_flu))

            # 能量哨兵打印

            if energy_print_every > 0:
                steps_done = len(z_axis_list)
                if (steps_done % energy_print_every == 0) or (z_now >= z_max - 1e-16):
                    drel = 100.0 * (U_now - U0_baseline) / U0_baseline
                    print(f"[U] z={z_now:0.3f} m  U={U_now: .3e} J  Δrel={drel:.2f}%")
                    # --- monitor PPT_i cap-hit (after evolve_rho_time) ---
                    W_cap_used = float(getattr(ion_conf, "W_cap", 0.0))
                    if W_cap_used > 0.0:
                        thr = 0.999 * W_cap_used
                        hits = 0
                        total = 0
                        # 可选抽样以进一步降开销：t_stride=1 表示不抽样
                        t_stride = 1
                        for it in range(0, Wt.shape[0], t_stride):
                            frm = Wt[it]  # [Ny,Nx]，此处只会临时分配一个小掩码
                            hits += int(xp.count_nonzero(frm >= thr))
                            total += frm.size
                        hit_frac = hits / max(1, total)
                        print(f"[z={z:.3f} m] PPT_i cap-hit = {hit_frac * 100:.3f}% (cap={W_cap_used:.2e})")

        # 前进 z & 进度
        z += dz_try
        pe = int(getattr(p, "progress_every_z", 0) or 0)
        if pe and ((len(z_axis_list) % pe) == 0 or z >= z_max - 1e-16):
            frac = z / z_max
            elapsed = time.perf_counter() - t0
            eta = elapsed / max(frac, 1e-9) * (1.0 - frac)
            print(f"[z] {z:.3f}/{z_max:.3f} m ({frac*100:6.2f}%)  elapsed {elapsed:6.1f}s  ETA {eta:6.1f}s")

    # ---------- 打包 ----------
    diag = {
        "z_axis":                   _np.asarray(z_axis_list,              dtype=rdtype_np),
        "U_z":                      _np.asarray(U_z_list,                 dtype=rdtype_np),
        "I_max_z":                  _np.asarray(I_max_z_list,             dtype=rdtype_np),
        "I_onaxis_max_z":           _np.asarray(I_onaxis_max_z_list,      dtype=rdtype_np),
        "I_center_t0_z":            _np.asarray(I_center_t0_z_list,       dtype=rdtype_np),
        "w_mom_z":                  _np.asarray(w_mom_z_list,             dtype=rdtype_np),
        "rho_max_z":                _np.asarray(rho_max_z_list,           dtype=rdtype_np),
        "rho_onaxis_max_z":         _np.asarray(rho_onaxis_max_list,      dtype=rdtype_np),
        "E_dep_z":                  _np.asarray(E_dep_z_list,             dtype=rdtype_np),   # 电离+IB

        "fwhm_plasma_z":            _np.asarray(fwhm_plasma_z_list,       dtype=rdtype_np),
        "fwhm_fluence_z":           _np.asarray(fwhm_fluence_z_list,      dtype=rdtype_np),
        "I_onaxis_max_interp_list": _np.asarray(I_onaxis_max_interp_list, dtype=rdtype_np),
        "raman_absorption_on":      bool(use_raman and raman_absorb_on),  # ← 新增：用于 summary 的 ON/OFF
        "E_dep_rot_z": _np.asarray(E_dep_rot_z_list, dtype=rdtype_np),
        "alpha_R_max_z": _np.asarray(alpha_R_max_z_list, dtype=rdtype_np),
        "alpha_R_mean_z": _np.asarray(alpha_R_mean_z_list, dtype=rdtype_np),
        "alpha_R_closed_z": _np.asarray(alpha_R_closed_z_list, dtype=rdtype_np),
        "IR_max_z": _np.asarray(IR_max_z_list, dtype=rdtype_np),
        "raman_absorption_model": absorption_model,  # 便于外部读
    }
    if record_onaxis_rho_time and (rho_onaxis_time_list is not None and len(rho_onaxis_time_list) > 0):
        diag["rho_onaxis_t_z"] = _np.stack(rho_onaxis_time_list, axis=0).astype(rdtype_np, copy=False)


    # Qacc 是 2D（J/m^2）：用于慢时间热扩散
    return E, xp.asnumpy(Qacc).astype(rdtype_np, copy=False), diag



