from __future__ import annotations
import sys, os, time

from .air_dispersion import n_of_omega
from .device import xp,to_cpu
from .constants import c0, n0_air, n2_air, Ui_N2, N0_air
from .config import GridConfig, BeamConfig, PropagationConfig, IonizationConfig, HeatConfig, RunConfig,RamanConfig
from .grids import make_axes
from .utils import gaussian_beam_xy, gaussian_pulse_t
from .diagnostics import intensity, peak_intensity, pulse_energy, save_npz
from .propagate import propagate_one_pulse
from .heat import diffuse_dn_gas
from .confio import E0_from_energy, E0_from_peak_power
import dataclasses


def _linear_advance(E, dz, *, axes, kperp2, k0, prop,beam):
    """
    仅线性推进 E -> z+dz（一次性），用于把起始面“跳”到 z_start。
    UPPE 用 step_linear_full_3d / factorized；paraxial 用 lin_propagator+step_linear。
    """
    if abs(float(dz)) < 1e-16:
        return E
    from .linear import lin_propagator, step_linear, step_linear_bk_nee_factorized
    from .linear_full import step_linear_full_factorized, step_linear_full_3d

    linear_model = str(getattr(prop, "linear_model", "uppe")).lower()
    if linear_model == "uppe":
        # 逐频折射率
        Omega = axes.Omega
        omega0 = 2.0 * xp.pi * c0 / float(getattr(beam, "lam0"))
        omega_tot = omega0 + Omega
        omega_safe = xp.where(xp.abs(omega_tot) < 1e-9 * omega0,
                              xp.sign(omega_tot) * 1e-9 * omega0,
                              omega_tot)
        n_w = n_of_omega(omega_safe,
                         P=getattr(prop, "air_P", 101325.0),
                         T=getattr(prop, "air_T", 293.15))
        K02_w = (n_w * omega_safe / c0) ** 2  # [Nt]
        use_factor = bool(getattr(prop, "full_linear_factorize", False))
        if use_factor:
            return step_linear_full_factorized(E, K02_w, kperp2, dz)
        else:
            return step_linear_full_3d(E, K02_w, kperp2, dz)
    elif linear_model == "bk_nee":
        omega0 = 2.0 * xp.pi * c0 / float(getattr(beam, "lam0"))
        return step_linear_bk_nee_factorized(
            E,
            Omega=axes.Omega,
            kperp2=kperp2,
            k0=k0,
            omega0=omega0,
            dz=dz,
            beta2=float(getattr(prop, "nee_beta2", 0.0)),
            denom_floor=float(getattr(prop, "nee_denom_floor", 1e-4)),
        )
    else:
        prop_x = lin_propagator(kperp2, k0, dz, ctype=E.dtype)
        return step_linear(E, prop_x)


def print_sim_summary(*, grid, beam, prop, ion, heat, run, axes, E, n2_used=None,
                      raman=None, dtype_str=None):
    import numpy as _np
    from .diagnostics import intensity, pulse_energy

    backend = getattr(xp, "__name__", "numpy")
    using_gpu = (backend == "cupy")
    dtype_str = (dtype_str or (str(E.dtype)))
    dev_extra = ""
    try:
        if using_gpu:
            import cupy as cp
            dev = cp.cuda.runtime.getDevice()
            name = cp.cuda.runtime.getDeviceProperties(dev)["name"].decode()
            dev_extra = f" [{name}]"
    except Exception:
        pass

    Nx, Ny, Nt = int(grid.Nx), int(grid.Ny), int(grid.Nt)
    Lx, Ly, Twin = float(grid.Lx), float(grid.Ly), float(grid.Twin)
    dx, dy = Lx / Nx, Ly / Ny
    dt     = Twin / Nt

    lam0 = float(beam.lam0)
    n0   = float(beam.n0)
    w0   = float(beam.w0)
    tau  = float(beam.tau_fwhm)
    fL   = getattr(beam, "focal_length", None)
    fL_s = f"{float(fL):.3f} m" if fL else "(none)"
    E0p  = float(getattr(beam, "E0_peak", 0.0))
    Ucfg = float(getattr(beam, "energy_J", 0.0) or 0.0)
    P0cfg = getattr(beam, "P0_peak", None)
    I0legacy = getattr(beam, "I0_peak", None)

    I0  = intensity(E, n0)
    Uin = float(pulse_energy(I0, dt, dx, dy))

    z_max = float(prop.z_max)
    dz    = float(prop.dz)
    auto  = bool(getattr(prop, "auto_substep", True))
    dzmin = float(getattr(prop, "dz_min", 0.0))
    grow  = float(getattr(prop, "grow_factor", 0.0))

    safety_mode   = str(getattr(prop, "safety_mode", "off")).lower()
    precheck_kerr = bool(getattr(prop, "precheck_kerr", False))
    max_pre_iter  = int(getattr(prop, "max_precheck_iter", 0))

    focus_step   = bool(getattr(prop, "focus_window_step", False))
    z_center     = getattr(prop, "focus_center_m", getattr(prop, "z_focus_hint", None))
    z_half       = float(getattr(prop, "focus_halfwidth_m", 0.0))
    dz_focus     = float(getattr(prop, "dz_focus", dz))

    th_lin  = float(getattr(prop, "max_linear_phase", 0.0))
    th_abs  = float(getattr(prop, "max_alpha_dz", 0.0))
    th_kerr = float(getattr(prop, "max_kerr_phase", 0.0))
    g_lim   = float(getattr(prop, "imax_growth_limit", 0.5))

    linear_model = str(getattr(prop, "linear_model", "uppe")).lower()
    factorize    = bool(getattr(prop, "full_linear_factorize", False))
    chunk_t      = int(getattr(prop, "linear_chunk_t", 8))
    lens_chunk_t = int(getattr(prop, "lens_chunk_t", 0))
    lens_mode    = "achromatic-per-ω" if (linear_model == "uppe" and fL) else "thin-lens (center ω)"

    n2_used = n2_used if n2_used is not None else float(getattr(prop, "n2", getattr(beam, "n2", 3.2e-23)))
    kerr_on = (n2_used > 0.0)

    use_shock    = bool(getattr(prop, "use_self_steepening", False))
    shock_method = str(getattr(prop, "self_steepening_method", "tdiff")).lower()
    shock_chunk  = int(getattr(prop, "shock_chunk_pixels", 65536))

    raman_on = bool(getattr(raman, "enabled", False)) if (raman is not None) else False
    if raman_on:
        fR = float(getattr(raman, "f_R", 0.15))
        T2 = float(getattr(raman, "T2", 8e-11))
        TR = float(getattr(raman, "T_R", 8e-12))
        raman_model  = str(getattr(raman, "model", "rot_sinexp"))
        raman_method = str(getattr(raman, "method", "iir")).lower()
        raman_chunk  = int(getattr(raman, "chunk_pixels", 65536))
        # 吸收设置
        absorption_model = str(getattr(raman, "absorption_model", "poynting")).lower()
        absorption_on    = bool(getattr(raman, "absorption", True))
        omega_R = getattr(raman, "omega_R", None)
        Gamma_R = getattr(raman, "Gamma_R", None)
        tau_fwhm_cfg = getattr(raman, "tau_fwhm", None)
        n_rot_frac   = getattr(raman, "n_rot_frac", None)
        R0_mode      = str(getattr(raman, "R0_mode", "mom")).lower()
        R0_fixed     = getattr(raman, "R0_fixed_m", None)


    else:
        fR = 0.0
        T2 = 0.0
        TR = 0.0
        raman_model = "off"
        raman_method = "off"
        raman_chunk = 0
        absorption_model = "off"
        absorption_on = False
        omega_R = None
        Gamma_R = None
        tau_fwhm_cfg = None
        n_rot_frac = None
        R0_mode = "off"
        R0_fixed = None
        print(f"Raman       : OFF")

    # ------- Ionization config -------

    # ------- Ionization config (species-based printing) -------
    # 兼容新旧字段：优先使用 time_mode / integrator / cycle_avg_samples / mean_clip_frac
    W_cap       = float(getattr(ion, "W_cap", 0.0))
    I_cap       = float(getattr(ion, "I_cap", 0.0))
    beta_rec    = float(getattr(ion, "beta_rec", 0.0))
    sigma_ib    = float(getattr(ion, "sigma_ib", 0.0))
    nu_ei_const = getattr(ion, "nu_ei_const", None)
    # time_mode 兼容旧字段（quasi_static_time+time_stat）
    _tm = str(getattr(ion, "time_mode", "") or "").lower()
    if not _tm:
        _qs = bool(getattr(ion, "quasi_static_time", False))
        _ts = str(getattr(ion, "time_stat", "peak")).lower()
        _tm = f"qs_{_ts}" if _qs else "full"
    time_mode   = _tm
    integrator  = str(getattr(ion, "integrator", "rk4")).lower()
    cav_samples = int(getattr(ion, "cycle_avg_samples", 64))
    mean_clip   = float(getattr(ion, "mean_clip_frac", 1e-3))

    species = getattr(ion, "species", None)

    # ---- helpers ----
    def _g(sp, key, default=None):
        return sp.get(key, default) if isinstance(sp, dict) else getattr(sp, key, default)

    def _resolve_rate(sp):
        """rate 优先；否则从旧的 model+cycle_avg 推断。已下线模型会抛错提示迁移。"""
        r = str(_g(sp, "rate", "") or "").lower().replace("ppt-i", "ppt_i")
        alias_map = {
            "ppt_talebpour_i": "ppt_talebpour_i_full",
            "popruzhenko_atom_i": "popruzhenko_atom_i_full",
        }
        removed = {"ppt_e", "ppt_i", "ppt_i_legacy", "adk_e", "powerlaw", "mpa"}
        if r:
            if r in removed:
                raise ValueError(
                    f"[ionization] species.rate='{r}' 已移除，请改用: "
                    "ppt_talebpour_i_legacy / ppt_talebpour_i_full / popruzhenko_atom_i_full / mpa_fact / off"
                )
            if r in alias_map:
                return alias_map[r]
            if r in ("none", "zero"):
                return "off"
            return r
        m = str(_g(sp, "model", getattr(ion, "model", "off"))).lower()
        cyc = bool(_g(sp, "cycle_avg", getattr(ion, "cycle_avg", False)))
        if m in ("none", "off", "zero", ""):
            return "off"
        if m in ("mpa_fact", "mpa_factorial", "multiphoton_factorial"):
            return "mpa_fact"
        if m in ("ppt", "ppt_cycleavg", "adk", "powerlaw", "mpa"):
            raise ValueError(
                f"[ionization] 旧字段 model='{m}'(cycle_avg={cyc}) 将推断到已移除模型；"
                "请在 species.rate 中显式设置为: "
                "ppt_talebpour_i_legacy / ppt_talebpour_i_full / popruzhenko_atom_i_full / mpa_fact / off"
            )
        raise ValueError(f"[ionization] 未识别 model/rate: model='{m}'")

    def _expects_for_rate(rate: str) -> str:
        rate = (rate or "").lower()
        if rate in ("ppt_talebpour_i_legacy", "ppt_talebpour_i_full", "popruzhenko_atom_i_full", "mpa_fact"):
            return "I"
        return "—"


    try:
        Pcr = 0.148 * lam0 * lam0 / (n0 * n2_used)
    except Exception:
        Pcr = _np.nan

    def onoff(b): return "ON " if b else "OFF"
    def fmt(v, unit=""):
        try: fv = float(v)
        except: return str(v)
        if abs(fv) >= 1e3 or (abs(fv) > 0 and abs(fv) < 1e-2):
            s = f"{fv:.3e}"
        else:
            s = f"{fv:.6f}".rstrip("0").rstrip(".")
        return f"{s}{unit}"

    print("\n=== Simulation Summary ===========================================")
    print(f"Backend(后端)        : {backend}{dev_extra} | dtype={dtype_str}  # 计算设备与精度")
    print(f"Grid(网格)           : Nx={Nx}, Ny={Ny}, Nt={Nt} | Lx={fmt(Lx,' m')}, Ly={fmt(Ly,' m')}, Twin={fmt(Twin,' s')}  # 时空采样规模")
    print(f"Steps(z步进)         : z_max={fmt(z_max,' m')}, dz={fmt(dz,' m')} | AutoSubstep={onoff(auto)} (dz_min={fmt(dzmin,' m')}, grow×{fmt(grow)})  # 传播步长控制")
    print(f"Safety(稳定性)       : mode={safety_mode.upper()} | precheck_kerr={onoff(precheck_kerr)} max_iter={max_pre_iter}  # 数值安全设置")
    if focus_step:
        zc = "None" if (z_center is None) else f"{float(z_center):.3f} m"
        print(f"FocusWin(焦区加密)   : {onoff(True)} center={zc} halfwidth={fmt(z_half,' m')} dz_focus={fmt(dz_focus,' m')}  # 焦点附近细步长")
    else:
        print(f"FocusWin(焦区加密)   : OFF  # 关闭焦区细步长")
    print(f"Thresholds(阈值)      : lin_phase≤{th_lin}, alpha·dz≤{th_abs}, kerr_phase≤{th_kerr}, Imax_growth≤{g_lim*100:.0f}%  # 自适应/回退判据")
    print(f"Linear(线性传播)      : model={linear_model} | factorize={onoff(factorize)} | chunk_t={chunk_t}  # 线性算子设置")
    print(f"Lens(透镜)           : {lens_mode} | f={fL_s} | lens_chunk_t={lens_chunk_t}  # 聚焦模型")
    print(f"Beam(入射光束)       : λ0={fmt(lam0,' m')} n0={fmt(n0)} w0={fmt(w0,' m')} τ_FWHM={fmt(tau,' s')}  # 初始脉冲参数")
    print(f"Energy(能量)         : config={fmt(Ucfg,' J')} | actual(after norm)={fmt(Uin,' J')} | E0_peak={fmt(E0p,' V/m')}  # 目标与归一化")
    norm_source = "E0_peak_direct"
    if P0cfg is not None:
        print(f"P0_peak(峰值功率)    : {fmt(float(P0cfg),' W')}  # 与 energy_J 二选一")
        norm_source = "P0_peak"
    elif I0legacy is not None:
        print(f"I0_peak(旧键,峰值功率): {fmt(float(I0legacy),' W')}  # 兼容旧配置")
        norm_source = "I0_peak_legacy"
    elif Ucfg > 0.0:
        norm_source = "energy_J"
    print(f"Repetition(重频)      : f_rep={fmt(float(getattr(heat,'f_rep',0.0)),' Hz')} | pulses={int(getattr(run,'Npulses',1))}  # 脉冲序列")
    print(f"Kerr(克尔效应)       : {onoff(kerr_on)}  n2={fmt(n2_used,' m^2/W')} | P_cr≈{fmt(Pcr,' W')}  # 自聚焦关键参数")
    print(f"Self-steep.(自陡峭)   : {onoff(use_shock)}  method={shock_method}  chunk_px={shock_chunk}  # 脉冲前沿陡化")
    # print(f"Raman       : {'ON ' if raman_on else 'OFF'}" + (f"  f_R={fmt(fR)}  model={raman_model}  T2={fmt(T2,' s')}  T_R={fmt(TR,' s')}  method={raman_method}  chunk_px={raman_chunk}" if raman_on else ""))
    print(
        f"Raman       : {'ON ' if raman_on else 'OFF'} f_R={fmt(fR)}  model={raman_model}  T2={fmt(T2, ' s')}  T_R={fmt(TR, ' s')}  method={raman_method}  chunk_px={raman_chunk}")
    print(f"  Absorption(吸收)   : {'ON ' if absorption_on else 'OFF'}  scheme={absorption_model}  # 拉曼吸收模型")
    if absorption_model == "closed_form":
        wR = fmt(omega_R, ' s^-1') if omega_R is not None else '(auto)'
        gR = fmt(Gamma_R, ' s^-1') if Gamma_R is not None else '(auto)'
        tp = fmt(tau_fwhm_cfg, ' s') if tau_fwhm_cfg is not None else '(measure on-axis)'
        nr = fmt(n_rot_frac) if n_rot_frac is not None else '(def 0.99)'
        if R0_mode == "fixed":
            r0s = f"fixed, R0={fmt(R0_fixed, ' m')}"
        else:
            r0s = "mom (second-moment radius)"
        print(f"    Params  : ω_R={wR}  Γ_R={gR}  τ_FWHM={tp}  n_rot_frac={nr}  R0_mode={r0s}")

        # 等效时间常数（用于核参数核对）
        if (omega_R is not None) and (Gamma_R is not None):
            Teq_R = (2.0 * _np.pi / float(omega_R)) if float(omega_R) > 0.0 else _np.nan
            Teq_2 = (1.0 / float(Gamma_R)) if float(Gamma_R) > 0.0 else _np.nan
            print(f"    Effective: T_R(eq)={fmt(Teq_R,' s')}  T2(eq)={fmt(Teq_2,' s')}  # 由 ω_R/Γ_R 反算")

    print("FinalEffective(最终生效参数):")
    print(f"  - ionization.time_mode={time_mode}  integrator={integrator}")
    print(f"  - propagation.auto_substep={auto}  dz={fmt(dz,' m')}  dz_focus={fmt(dz_focus,' m')}")
    print(f"  - beam.energy_J={'null' if getattr(beam, 'energy_J', None) is None else fmt(getattr(beam, 'energy_J'), ' J')}  P0_peak={'null' if P0cfg is None else fmt(P0cfg, ' W')}  norm_source={norm_source}")

    # ------------ Ionization printing (dict/object safe) ------------


    species_ok = (species and isinstance(species, (list, tuple)) and len(species) > 0)
    if species_ok:
        # 汇总输入域
        exp_set = set()
        has_ppt_i = False
        for sp in species:
            r = _resolve_rate(sp)
            exp_set.add(_expects_for_rate(r))
            if r in ("ppt_talebpour_i_legacy", "ppt_talebpour_i_full", "popruzhenko_atom_i_full"):
                has_ppt_i = True

        exp_tag = "mixed" if (len(exp_set) > 1) else (next(iter(exp_set)) if exp_set else "none")
        print(f"Ionization(电离)      : SPECIES({len(species)})  caps: W≤{fmt(W_cap,' s^-1')}, I≤{fmt(I_cap,' W/m^2')}  expects={exp_tag}  # 电离输入域")

        print(f"  Species(组分)      :")
        for sp in species:
            name = _g(sp, "name", "?")
            frac = float(_g(sp, "fraction", 1.0))
            rate = _resolve_rate(sp)
            expx = _expects_for_rate(rate)

            if rate in ("ppt_talebpour_i_legacy", "ppt_talebpour_i_full"):
                Ip = _g(sp, "Ip_eV", getattr(ion, "Ip_eV", None))
                Ipe = _g(sp, "Ip_eV_eff", None)
                Zeff = _g(sp, "Zeff", None)
                l  = _g(sp, "l", 0)
                m  = _g(sp, "m", 0)
                Wc = _g(sp, "W_cap", None)
                cap_note = f", W_cap={fmt(Wc,' s^-1')}" if Wc is not None else ""
                print(f"    - {name:6s} rate={rate.upper()} Ip={Ip} eV, Ip_eff={Ipe} eV, Zeff={Zeff}, l={l}, m={m}, fraction={frac:.3f}, expects={expx}{cap_note}")

            elif rate == "popruzhenko_atom_i_full":
                Ip = _g(sp, "Ip_eV", getattr(ion, "Ip_eV", None))
                Z  = _g(sp, "Z", getattr(ion, "Z", None))
                l  = _g(sp, "l", 0)
                m  = _g(sp, "m", 0)
                nt = _g(sp, "max_terms", 4096)
                Wc = _g(sp, "W_cap", None)
                cap_note = f", W_cap={fmt(Wc,' s^-1')}" if Wc is not None else ""
                print(f"    - {name:6s} rate=POPRUZHENKO_ATOM_I_FULL Ip={Ip} eV, Z={Z}, l={l}, m={m}, max_terms={nt}, fraction={frac:.3f}, expects={expx}{cap_note}")

            elif rate == "mpa_fact":
                ell = _g(sp, "ell", getattr(ion, "ell", None))
                Imp = _g(sp, "I_mp", getattr(ion, "I_mp", None))
                Wc  = _g(sp, "W_cap", None)
                cap_note = f", W_cap={fmt(Wc,' s^-1')}" if Wc is not None else ""
                print(f"    - {name:6s} rate=MPA_FACT ell={ell}, I_mp={fmt(Imp,' W/m^2')}, fraction={frac:.3f}, expects={expx}{cap_note}")

            elif rate == "off":
                print(f"    - {name:6s} rate=OFF      fraction={frac:.3f}, expects=—")
            else:
                print(f"    - {name:6s} rate={rate.upper():<8} fraction={frac:.3f}, expects={expx}")

        # Time-mode / integrator / extra params
        if time_mode.startswith("qs_"):
            print(f"  TimeMode(时间近似) : QS ({time_mode[3:]})  mean_clip={mean_clip}  # 准稳态近似")
        else:
            print(f"  TimeMode(时间近似) : FULL  integrator={integrator.upper()}  # 全时域积分")

        if has_ppt_i:
            print(f"  CycleAvg(周期平均) : samples={cav_samples}  (for *_i rates)  # 周期采样数")

        if nu_ei_const is not None:
            print(f"  Drude ν_ei(碰撞频) : {fmt(float(nu_ei_const),' s^-1')}")

        print(f"  Drude/IB(等离子体) : β_rec={fmt(beta_rec, ' m^3/s')}  σ_ib={fmt(sigma_ib, ' m^2')}  # 复合与吸收")

    else:
        # 兼容：若用户没有提供 species（不推荐），回退到旧打印
        ion_model = str(getattr(ion, "model", "none")).lower()
        if ion_model in ("mpa_fact","mpa_factorial"):
            ell = getattr(ion, "ell", None); Imp = getattr(ion, "I_mp", None)
            print(f"Ionization  : MPA_FACT ell={ell}, I_mp={fmt(Imp,' W/m^2')} | caps: W≤{fmt(W_cap,' s^-1')}, I≤{fmt(I_cap,' W/m^2')}  expects=I")
        else:
            print("Ionization  : OFF")
        if nu_ei_const is not None:
            print(f"  Drude ν_ei(碰撞频) : {fmt(float(nu_ei_const),' s^-1')}")
        print(f"  Drude/IB(等离子体) : β_rec={fmt(beta_rec, ' m^3/s')}  σ_ib={fmt(sigma_ib, ' m^2')}  # 复合与吸收")

    guard_en    = bool(getattr(prop, "energy_guard_enabled", True))
    guard_every = int(getattr(prop, "energy_guard_every", 50))
    guard_skip  = int(getattr(prop, "energy_guard_skip", 5))
    blowup_fac  = float(getattr(prop, "energy_guard_blowup", 50.0))
    diag_extra  = bool(getattr(prop, "diag_extra", False))

    print(f"EnergyGuard(能量守护): {onoff(guard_en)} every={guard_every} skip-first={guard_skip}× | blowup×{fmt(blowup_fac)}  # 异常能量监测")
    print(f"Diagnostics(诊断输出): extra={onoff(diag_extra)}  # 扩展诊断开关")
    print("===================================================================\n")



# --- 频率域薄透镜：对每个 ω 乘相位  exp(-i k(ω) r^2 / 2f)
def apply_thin_lens_achromatic(E, axes, beam, prop, chunk_t: int = 0):
    """
    频率域薄透镜：对每个频率分量乘 exp(-i k(|ω|) r^2 / (2f))
    - 只在 UPPE 里用；paraxial 可继续用中心频率版本
    - 支持按时间轴分块，降低显存峰值
    """
    Omega   = axes.Omega                       # [Nt]
    omega0  = 2.0 * xp.pi * c0 / beam.lam0
    omega   = omega0 + Omega                   # [Nt]
    omega_a = xp.abs(omega)                    # 关键：用 |ω|

    n_w = n_of_omega(omega_a,
                     P=getattr(prop, "air_P", 101325.0),
                     T=getattr(prop, "air_T", 293.15))
    k_w = n_w * omega_a / c0                  # [Nt], 始终非负，与线性传播子一致

    X, Y = xp.meshgrid(axes.x, axes.y, indexing='xy')
    r2   = (X**2 + Y**2)[xp.newaxis, :, :]    # [1,Ny,Nx]
    f    = float(beam.focal_length)
    onej = xp.array(1j, dtype=E.dtype)

    Nt = E.shape[0]
    if not chunk_t or chunk_t <= 0:
        Ew = xp.fft.fft(E, axis=0)            # [Nt,Ny,Nx]
        phase_w = -(k_w[:, None, None] / (2.0 * f)) * r2
        Ew *= xp.exp(onej * phase_w).astype(Ew.dtype, copy=False)
        E = xp.fft.ifft(Ew, axis=0)
        return E
    else:
        # 分块（按时间轴）
        out = xp.empty_like(E)
        for i0 in range(0, Nt, chunk_t):
            i1 = min(Nt, i0 + chunk_t)
            Ew = xp.fft.fft(E[i0:i1, ...], axis=0)            # [B,Ny,Nx]
            phase_w = -(k_w[i0:i1, None, None] / (2.0 * f)) * r2
            Ew *= xp.exp(onej * phase_w).astype(Ew.dtype, copy=False)
            out[i0:i1, ...] = xp.fft.ifft(Ew, axis=0)
            # 释放临时
            Ew = phase_w = None
        return out

def energy_probe_cb(z, U_now, U0, tol):
    rel = abs(U_now - U0) / max(U0, 1e-30)
    print(f"[U] z={z:.3f} m  U={U_now:.3e} J  Δrel={rel:.2%}")
    if rel > tol:
        print(f"[U-WARN] energy deviation > {tol:.1%} at z={z:.3f} m")

def run_demo(
    grid: GridConfig = GridConfig(),
    beam: BeamConfig = BeamConfig(n0=n0_air),
    prop: PropagationConfig = PropagationConfig(),
    ion: IonizationConfig = IonizationConfig(),
    heat: HeatConfig = HeatConfig(),
    run: RunConfig = RunConfig(),
    raman: RamanConfig = RamanConfig(),
    out_path: str = "khzfil_out.npz",
    dtype: str = "fp32",
):
    dmap = {"fp32": xp.complex64, "fp64": xp.complex128}
    ctype = dmap.get(str(dtype).lower(), xp.complex64)
    rmap = {"fp32": xp.float32, "fp64": xp.float64}
    rtype = rmap.get(str(dtype).lower(), xp.float32)

    import math, numpy as _np
    omega0 = 2 * math.pi * c0 / beam.lam0
    k0 = beam.n0 * omega0 / c0
    axes = make_axes(grid.Nx, grid.Ny, grid.Nt, grid.Lx, grid.Ly, grid.Twin)

    has_energy = getattr(beam, "energy_J", None) is not None
    has_p0 = getattr(beam, "P0_peak", None) is not None
    has_i0_legacy = getattr(beam, "I0_peak", None) is not None
    if has_energy and has_p0:
        raise ValueError("Beam energy_J and P0_peak are mutually exclusive; please keep only one.")
    if has_energy and has_i0_legacy:
        raise ValueError("Beam energy_J and legacy I0_peak are mutually exclusive; please keep only one.")
    if has_p0 and has_i0_legacy:
        raise ValueError("Beam P0_peak and legacy I0_peak cannot both be set.")

    if (getattr(beam, "E0_peak", 0.0) == 0.0) and has_energy:
        beam.E0_peak = E0_from_energy(float(beam.energy_J), float(beam.w0), float(beam.tau_fwhm), float(beam.n0))
        print(f"[derive] E0_peak <= 0, derived from energy_J: E0_peak={beam.E0_peak:.3e} V/m")
    elif (getattr(beam, "E0_peak", 0.0) == 0.0) and has_p0:
        beam.E0_peak = E0_from_peak_power(float(beam.P0_peak), float(beam.w0), float(beam.n0))
        print(f"[derive] E0_peak <= 0, derived from P0_peak: E0_peak={beam.E0_peak:.3e} V/m")
    elif (getattr(beam, "E0_peak", 0.0) == 0.0) and has_i0_legacy:
        beam.E0_peak = E0_from_peak_power(float(beam.I0_peak), float(beam.w0), float(beam.n0))
        print(f"[derive] E0_peak <= 0, derived from legacy I0_peak(as peak power): E0_peak={beam.E0_peak:.3e} V/m")
    elif getattr(beam, "E0_peak", 0.0) == 0.0:
        raise ValueError("Beam E0_peak is 0 and no energy_J/P0_peak provided; cannot build input field.")

    E_xy = gaussian_beam_xy(axes.x, axes.y, beam.w0)[None, ...]
    E_t  = gaussian_pulse_t(axes.t, beam.tau_fwhm)
    E = (beam.E0_peak * E_t * E_xy).astype(ctype)

    if getattr(beam, "focal_length", None):
        if str(getattr(prop, "linear_model", "uppe")).lower() == "uppe":
            E = apply_thin_lens_achromatic(E, axes, beam, prop, chunk_t=getattr(prop, "lens_chunk_t", 0))
        else:
            X, Y = xp.meshgrid(axes.x, axes.y, indexing='xy')
            phase_lens = -(k0 / (2.0 * float(beam.focal_length))) * (X ** 2 + Y ** 2)
            onej = xp.array(1j, dtype=E.dtype)
            E *= xp.exp(onej * xp.asarray(phase_lens, dtype=(xp.float32 if E.dtype == xp.complex64 else xp.float64)))

    U_target = getattr(beam, "energy_J", None)
    if U_target is not None:
        I_in = intensity(E, beam.n0)
        U_now = float(pulse_energy(I_in, axes.dt, axes.dx, axes.dy)) + 1e-30
        if abs(U_now - U_target) / U_target > 1e-3:
            scale = (float(U_target) / U_now) ** 0.5
            E *= scale
            print(f"[norm] input energy: {U_now:.3e} J -> target {U_target:.3e} J (scale {scale:.3e})")

    n2_used = float(getattr(prop, "n2", getattr(beam, "n2_air", n2_air)))
    print_sim_summary(grid=grid, beam=beam, prop=prop, ion=ion, heat=heat, run=run,
                      axes=axes, E=E, n2_used=n2_used, raman=raman, dtype_str=dtype)

    # 线性基准腰斑测量（提示）
    def measure_input_waist(E0, axes):
        it0 = E0.shape[0] // 2
        I0 = xp.abs(E0[it0]) ** 2
        Ix = xp.sum(I0, axis=0) * axes.dy
        Iy = xp.sum(I0, axis=1) * axes.dx
        Sx = float(xp.sum(Ix) * axes.dx) + 1e-30
        Sy = float(xp.sum(Iy) * axes.dy) + 1e-30
        x2 = float(xp.sum((axes.x ** 2) * Ix) * axes.dx) / Sx
        y2 = float(xp.sum((axes.y ** 2) * Iy) * axes.dy) / Sy
        w_x = (4.0 * x2) ** 0.5
        w_y = (4.0 * y2) ** 0.5
        return 0.5 * (w_x + w_y)

    w_meas = measure_input_waist(E, axes)
    lam_med = beam.lam0 / beam.n0
    zR = math.pi * (w_meas ** 2) / lam_med
    f = float(getattr(beam, "focal_length", float("inf")))
    z_pred = (f / (1.0 + (f / zR) ** 2)) if _np.isfinite(f) else float("inf")
    print(f"[waist] measured w_in={w_meas:.3e} m  -> z_R={zR:.3e} m  -> z_focus_pred≈{z_pred:.4f} m (thin-lens)")

    # ====== 围焦点限窗（可极大节省时间）======
    # 配置字段（如果没有则走默认 False）
    limit_win = bool(getattr(prop, "limit_focus_window", False))
    win_half  = float(getattr(prop, "window_halfwidth_m", 0.0))   # 例如 0.30
    # 焦点中心：优先 Prop 提供；否则用线性薄透镜预估值
    z_focus_hint = getattr(prop, "focus_center_m", getattr(prop, "z_focus_hint", None))

    # 线性薄透镜预估（供兜底）
    def _predict_focus_linear(E0, axes):
        # 用你之前 Summary 里同一套测腰+计算 zR 的办法
        it0 = E0.shape[0] // 2
        I0 = xp.abs(E0[it0]) ** 2
        Ix = xp.sum(I0, axis=0) * axes.dy
        Iy = xp.sum(I0, axis=1) * axes.dx
        Sx = float(xp.sum(Ix) * axes.dx) + 1e-30
        Sy = float(xp.sum(Iy) * axes.dy) + 1e-30
        x2 = float(xp.sum((axes.x ** 2) * Ix) * axes.dx) / Sx
        y2 = float(xp.sum((axes.y ** 2) * Iy) * axes.dy) / Sy
        w_x = (4.0 * x2) ** 0.5
        w_y = (4.0 * y2) ** 0.5
        w_meas = 0.5 * (w_x + w_y)
        lam_med = beam.lam0 / beam.n0
        zR = xp.pi * (w_meas ** 2) / lam_med
        fL = float(getattr(beam, "focal_length", _np.inf))
        return (fL / (1.0 + (fL / zR) ** 2)) if _np.isfinite(fL) else 0.0

    if limit_win and win_half > 0.0:
        if z_focus_hint is None:
            z_focus_hint = _predict_focus_linear(E, axes)
        z_start = max(0.0, float(z_focus_hint) - win_half)
        z_end   = float(z_focus_hint) + win_half
        # 一次性线性把起始场推进到 z_start
        if z_start > 0.0:
            E = _linear_advance(E, z_start, axes=axes, kperp2=axes.kperp2, k0=k0, prop=prop,beam=beam)
            print(f"[window] Linear pre-advance: z_start={z_start:.4f} m  (center={float(z_focus_hint):.4f} m, half={win_half:.4f} m)")
        # 覆盖本次传播的 z_max：只在窗内跑非线性
        z_window = max(1e-9, z_end - z_start)
        prop = dataclasses.replace(prop, z_max=z_window)

    dn_gas = xp.zeros((grid.Ny, grid.Nx), dtype=rtype)
    delta_t_pulse = 1.0 / heat.f_rep

    t_all = time.perf_counter()
    last_diag = None
    for i in range(run.Npulses):
        t_p = time.perf_counter()
        E, Q2D, diag = propagate_one_pulse(
            E,
            kperp2=axes.kperp2,
            k0=k0, omega0=omega0,
            dz=prop.dz, z_max=prop.z_max,
            n0=beam.n0, n2=n2_used,
            Ui=Ui_N2, N0=N0_air,
            ion_conf=ion, dn_gas=dn_gas,
            dt=axes.dt, axes=axes, prop_conf=prop, raman_conf=raman,
            record_onaxis_rho_time=True,
            record_every_z=1,
        )
        dn_gas = diffuse_dn_gas(dn_gas, Q2D, heat.D_gas, delta_t_pulse, axes.kperp2, heat.gamma_heat)
        mn, mx = float(xp.min(dn_gas)), float(xp.max(dn_gas))
        print(f"Pulse {i+1}/{run.Npulses}: Δn_gas min/max = {mn:.3e}/{mx:.3e}  (elapsed {time.perf_counter()-t_p:.1f}s)")
        last_diag = diag

    print(f"[total] {time.perf_counter() - t_all:5.1f}s")

    I_out = intensity(E, beam.n0)
    Ipk = peak_intensity(I_out)
    energy = pulse_energy(I_out, axes.dt, axes.dx, axes.dy)
    print(f"Peak intensity out: {Ipk:.3e} W/m^2; Pulse energy ~ {energy:.3e} J")

    # 焦点（若有 w_mom_z）
    try:
        if last_diag and "w_mom_z" in last_diag:
            z_axis_cpu = to_cpu(last_diag["z_axis"])
            wcpu = to_cpu(last_diag["w_mom_z"])
            if len(z_axis_cpu) == len(wcpu) and len(wcpu) > 0:
                iz_min = int(_np.argmin(wcpu))
                print(f"[focus] estimated z_of_focus ≈ {z_axis_cpu[iz_min]:.4f} m (min second-moment radius)")
    except Exception:
        pass

    # 组织输出（存在就写）
    out = {
        "x": to_cpu(axes.x), "y": to_cpu(axes.y), "t": to_cpu(axes.t),
        "I_out_center_t": to_cpu(I_out[:, grid.Ny//2, grid.Nx//2]),
        "dn_gas": to_cpu(dn_gas),
    }
    # 统一把 diag 中以下键转存（存在就收）
    diag_keys = [
        "z_axis", "U_z", "I_max_z", "I_onaxis_max_z", "I_center_t0_z",
        "w_mom_z", "rho_onaxis_max_z","rho_max_z",
        "E_dep_z","E_dep_rot_z" ,"fwhm_plasma_z", "fwhm_fluence_z", "rho_onaxis_t_z","I_onaxis_max_interp_list"
    ]
    if last_diag:
        for k in diag_keys:
            if k in last_diag and last_diag[k] is not None:
                out[k] = to_cpu(last_diag[k])

    out = {k: v for k, v in out.items() if v is not None}
    save_npz(out_path, **out)
    print(f"Saved diagnostics to {out_path}")


def run_from_file(cfg_path: str, out_path: str = "khzfil_out.npz"):
    from .confio import load_all
    grid, beam, prop, ion, heat, run, raman = load_all(cfg_path)
    print(f"[config] Loaded: {os.path.abspath(cfg_path)}")
    return run_demo(grid=grid, beam=beam, prop=prop, ion=ion, heat=heat, run=run, raman=raman, out_path=out_path)

if __name__ == "__main__":
    # 用法：
    #   python -m KHz_filament.cli                    # 用默认参数
    #   python -m KHz_filament.cli path/to/config部分说明 # 从外部配置跑
    if len(sys.argv) > 1:
        cfg = sys.argv[1]
        run_from_file(cfg)
    else:
        run_demo()
