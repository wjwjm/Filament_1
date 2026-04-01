from __future__ import annotations

from .device import xp
from .ionization.rate_registry import RATE_ALIAS_MAP


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
    dt = Twin / Nt

    lam0 = float(beam.lam0)
    n0 = float(beam.n0)
    w0 = float(beam.w0)
    tau = float(beam.tau_fwhm)
    fL = getattr(beam, "focal_length", None)
    fL_s = f"{float(fL):.3f} m" if fL else "(none)"
    E0p = float(getattr(beam, "E0_peak", 0.0))
    Ucfg = float(getattr(beam, "energy_J", 0.0) or 0.0)
    P0cfg = getattr(beam, "P0_peak", None)

    I0 = intensity(E, n0)
    Uin = float(pulse_energy(I0, dt, dx, dy))

    z_max = float(prop.z_max)
    dz = float(prop.dz)
    auto = bool(getattr(prop, "auto_substep", True))
    dzmin = float(getattr(prop, "dz_min", 0.0))
    grow = float(getattr(prop, "grow_factor", 0.0))

    safety_mode = str(getattr(prop, "safety_mode", "off")).lower()
    precheck_kerr = bool(getattr(prop, "precheck_kerr", False))
    max_pre_iter = int(getattr(prop, "max_precheck_iter", 0))

    focus_step = bool(getattr(prop, "focus_window_step", False))
    z_center = getattr(prop, "focus_center_m", getattr(prop, "z_focus_hint", None))
    z_half = float(getattr(prop, "focus_halfwidth_m", 0.0))
    dz_focus = float(getattr(prop, "dz_focus", dz))

    th_lin = float(getattr(prop, "max_linear_phase", 0.0))
    th_abs = float(getattr(prop, "max_alpha_dz", 0.0))
    th_kerr = float(getattr(prop, "max_kerr_phase", 0.0))
    g_lim = float(getattr(prop, "imax_growth_limit", 0.5))

    linear_model = str(getattr(prop, "linear_model", "uppe")).lower()
    factorize = bool(getattr(prop, "full_linear_factorize", False))
    chunk_t = int(getattr(prop, "linear_chunk_t", 8))
    lens_chunk_t = int(getattr(prop, "lens_chunk_t", 0))
    lens_mode = "achromatic-per-ω" if (linear_model == "uppe" and fL) else "thin-lens (center ω)"

    n2_used = n2_used if n2_used is not None else float(getattr(prop, "n2", getattr(beam, "n2", 3.2e-23)))
    kerr_on = (n2_used > 0.0)

    use_shock = bool(getattr(prop, "use_self_steepening", False))
    shock_method = str(getattr(prop, "self_steepening_method", "tdiff")).lower()
    shock_chunk = int(getattr(prop, "shock_chunk_pixels", 65536))

    raman_on = bool(getattr(raman, "enabled", False)) if (raman is not None) else False
    if raman_on:
        fR = float(getattr(raman, "f_R", 0.15))
        T2 = float(getattr(raman, "T2", 8e-11))
        TR = float(getattr(raman, "T_R", 8e-12))
        raman_model = str(getattr(raman, "model", "rot_sinexp"))
        raman_method = str(getattr(raman, "method", "iir")).lower()
        raman_chunk = int(getattr(raman, "chunk_pixels", 65536))
        absorption_model = str(getattr(raman, "absorption_model", "poynting")).lower()
        absorption_on = bool(getattr(raman, "absorption", True))
        omega_R = getattr(raman, "omega_R", None)
        Gamma_R = getattr(raman, "Gamma_R", None)
        tau_fwhm_cfg = getattr(raman, "tau_fwhm", None)
        n_rot_frac = getattr(raman, "n_rot_frac", None)
        R0_mode = str(getattr(raman, "R0_mode", "mom")).lower()
        R0_fixed = getattr(raman, "R0_fixed_m", None)
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
        print("Raman       : OFF")

    W_cap = float(getattr(ion, "W_cap", 0.0))
    I_cap = float(getattr(ion, "I_cap", 0.0))
    beta_rec = float(getattr(ion, "beta_rec", 0.0))
    sigma_ib = float(getattr(ion, "sigma_ib", 0.0))
    nu_ei_const = getattr(ion, "nu_ei_const", None)
    _tm = str(getattr(ion, "time_mode", "") or "").lower()
    if not _tm:
        _qs = bool(getattr(ion, "quasi_static_time", False))
        _ts = str(getattr(ion, "time_stat", "peak")).lower()
        _tm = f"qs_{_ts}" if _qs else "full"
    time_mode = _tm
    integrator = str(getattr(ion, "integrator", "rk4")).lower()
    cav_samples = int(getattr(ion, "cycle_avg_samples", 64))
    mean_clip = float(getattr(ion, "mean_clip_frac", 1e-3))

    species = getattr(ion, "species", None)

    def _g(sp, key, default=None):
        return sp.get(key, default) if isinstance(sp, dict) else getattr(sp, key, default)

    def _resolve_rate(sp):
        r = str(_g(sp, "rate", "off") or "off").lower()
        return RATE_ALIAS_MAP.get(r, r)

    def _expects_for_rate(rate: str) -> str:
        if (rate or "").lower() in ("ppt_talebpour_i_legacy", "ppt_talebpour_i_full", "popruzhenko_atom_i_full", "mpa_fact"):
            return "I"
        return "—"

    try:
        Pcr = 0.148 * lam0 * lam0 / (n0 * n2_used)
    except Exception:
        Pcr = _np.nan

    def onoff(b): return "ON " if b else "OFF"
    def fmt(v, unit=""):
        try:
            fv = float(v)
        except Exception:
            return str(v)
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
        print("FocusWin(焦区加密)   : OFF  # 关闭焦区细步长")
    print(f"Thresholds(阈值)      : lin_phase≤{th_lin}, alpha·dz≤{th_abs}, kerr_phase≤{th_kerr}, Imax_growth≤{g_lim*100:.0f}%  # 自适应/回退判据")
    print(f"Linear(线性传播)      : model={linear_model} | factorize={onoff(factorize)} | chunk_t={chunk_t}  # 线性算子设置")
    print(f"Lens(透镜)           : {lens_mode} | f={fL_s} | lens_chunk_t={lens_chunk_t}  # 聚焦模型")
    print(f"Beam(入射光束)       : λ0={fmt(lam0,' m')} n0={fmt(n0)} w0={fmt(w0,' m')} τ_FWHM={fmt(tau,' s')}  # 初始脉冲参数")
    print(f"Energy(能量)         : config={fmt(Ucfg,' J')} | actual(after norm)={fmt(Uin,' J')} | E0_peak={fmt(E0p,' V/m')}  # 目标与归一化")
    norm_source = "E0_peak_direct"
    if P0cfg is not None:
        print(f"P0_peak(峰值功率)    : {fmt(float(P0cfg),' W')}  # 与 energy_J 二选一")
        norm_source = "P0_peak"
    elif Ucfg > 0.0:
        norm_source = "energy_J"
    print(f"Repetition(重频)      : f_rep={fmt(float(getattr(heat,'f_rep',0.0)),' Hz')} | pulses={int(getattr(run,'Npulses',1))}  # 脉冲序列")
    print(f"Kerr(克尔效应)       : {onoff(kerr_on)}  n2={fmt(n2_used,' m^2/W')} | P_cr≈{fmt(Pcr,' W')}  # 自聚焦关键参数")
    print(f"Self-steep.(自陡峭)   : {onoff(use_shock)}  method={shock_method}  chunk_px={shock_chunk}  # 脉冲前沿陡化")
    print(f"Raman       : {'ON ' if raman_on else 'OFF'} f_R={fmt(fR)}  model={raman_model}  T2={fmt(T2, ' s')}  T_R={fmt(TR, ' s')}  method={raman_method}  chunk_px={raman_chunk}")
    print(f"  Absorption(吸收)   : {'ON ' if absorption_on else 'OFF'}  scheme={absorption_model}  # 拉曼吸收模型")
    if absorption_model == "closed_form":
        wR = fmt(omega_R, ' s^-1') if omega_R is not None else '(auto)'
        gR = fmt(Gamma_R, ' s^-1') if Gamma_R is not None else '(auto)'
        tp = fmt(tau_fwhm_cfg, ' s') if tau_fwhm_cfg is not None else '(measure on-axis)'
        nr = fmt(n_rot_frac) if n_rot_frac is not None else '(def 0.99)'
        r0s = f"fixed, R0={fmt(R0_fixed, ' m')}" if R0_mode == "fixed" else "mom (second-moment radius)"
        print(f"    Params  : ω_R={wR}  Γ_R={gR}  τ_FWHM={tp}  n_rot_frac={nr}  R0_mode={r0s}")
        if (omega_R is not None) and (Gamma_R is not None):
            Teq_R = (2.0 * _np.pi / float(omega_R)) if float(omega_R) > 0.0 else _np.nan
            Teq_2 = (1.0 / float(Gamma_R)) if float(Gamma_R) > 0.0 else _np.nan
            print(f"    Effective: T_R(eq)={fmt(Teq_R,' s')}  T2(eq)={fmt(Teq_2,' s')}  # 由 ω_R/Γ_R 反算")

    print("FinalEffective(最终生效参数):")
    print(f"  - ionization.time_mode={time_mode}  integrator={integrator}")
    print(f"  - propagation.auto_substep={auto}  dz={fmt(dz,' m')}  dz_focus={fmt(dz_focus,' m')}")
    print(f"  - beam.energy_J={'null' if getattr(beam, 'energy_J', None) is None else fmt(getattr(beam, 'energy_J'), ' J')}  P0_peak={'null' if P0cfg is None else fmt(P0cfg, ' W')}  norm_source={norm_source}")

    species_ok = (species and isinstance(species, (list, tuple)) and len(species) > 0)
    if species_ok:
        exp_set = set()
        has_ppt_i = False
        for sp in species:
            r = _resolve_rate(sp)
            exp_set.add(_expects_for_rate(r))
            if r in ("ppt_talebpour_i_legacy", "ppt_talebpour_i_full", "popruzhenko_atom_i_full"):
                has_ppt_i = True

        exp_tag = "mixed" if (len(exp_set) > 1) else (next(iter(exp_set)) if exp_set else "none")
        print(f"Ionization(电离)      : SPECIES({len(species)})  caps: W≤{fmt(W_cap,' s^-1')}, I≤{fmt(I_cap,' W/m^2')}  expects={exp_tag}  # 电离输入域")

        print("  Species(组分)      :")
        for sp in species:
            name = _g(sp, "name", "?")
            frac = float(_g(sp, "fraction", 1.0))
            rate = _resolve_rate(sp)
            expx = _expects_for_rate(rate)

            if rate in ("ppt_talebpour_i_legacy", "ppt_talebpour_i_full"):
                Ip = _g(sp, "Ip_eV", getattr(ion, "Ip_eV", None))
                Ipe = _g(sp, "Ip_eV_eff", None)
                Zeff = _g(sp, "Zeff", None)
                l = _g(sp, "l", 0)
                m = _g(sp, "m", 0)
                Wc = _g(sp, "W_cap", None)
                cap_note = f", W_cap={fmt(Wc,' s^-1')}" if Wc is not None else ""
                print(f"    - {name:6s} rate={rate.upper()} Ip={Ip} eV, Ip_eff={Ipe} eV, Zeff={Zeff}, l={l}, m={m}, fraction={frac:.3f}, expects={expx}{cap_note}")
            elif rate == "popruzhenko_atom_i_full":
                Ip = _g(sp, "Ip_eV", getattr(ion, "Ip_eV", None))
                Z = _g(sp, "Z", getattr(ion, "Z", None))
                l = _g(sp, "l", 0)
                m = _g(sp, "m", 0)
                nt = _g(sp, "max_terms", 4096)
                Wc = _g(sp, "W_cap", None)
                cap_note = f", W_cap={fmt(Wc,' s^-1')}" if Wc is not None else ""
                print(f"    - {name:6s} rate=POPRUZHENKO_ATOM_I_FULL Ip={Ip} eV, Z={Z}, l={l}, m={m}, max_terms={nt}, fraction={frac:.3f}, expects={expx}{cap_note}")
            elif rate == "mpa_fact":
                ell = _g(sp, "ell", getattr(ion, "ell", None))
                Imp = _g(sp, "I_mp", getattr(ion, "I_mp", None))
                Wc = _g(sp, "W_cap", None)
                cap_note = f", W_cap={fmt(Wc,' s^-1')}" if Wc is not None else ""
                print(f"    - {name:6s} rate=MPA_FACT ell={ell}, I_mp={fmt(Imp,' W/m^2')}, fraction={frac:.3f}, expects={expx}{cap_note}")
            elif rate == "off":
                print(f"    - {name:6s} rate=OFF      fraction={frac:.3f}, expects=—")
            else:
                print(f"    - {name:6s} rate={rate.upper():<8} fraction={frac:.3f}, expects={expx}")

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
        ion_model = str(getattr(ion, "model", "none")).lower()
        if ion_model in ("mpa_fact", "mpa_factorial"):
            ell = getattr(ion, "ell", None)
            Imp = getattr(ion, "I_mp", None)
            print(f"Ionization  : MPA_FACT ell={ell}, I_mp={fmt(Imp,' W/m^2')} | caps: W≤{fmt(W_cap,' s^-1')}, I≤{fmt(I_cap,' W/m^2')}  expects=I")
        else:
            print("Ionization  : OFF")
        if nu_ei_const is not None:
            print(f"  Drude ν_ei(碰撞频) : {fmt(float(nu_ei_const),' s^-1')}")
        print(f"  Drude/IB(等离子体) : β_rec={fmt(beta_rec, ' m^3/s')}  σ_ib={fmt(sigma_ib, ' m^2')}  # 复合与吸收")

    guard_en = bool(getattr(prop, "energy_guard_enabled", True))
    guard_every = int(getattr(prop, "energy_guard_every", 50))
    guard_skip = int(getattr(prop, "energy_guard_skip", 5))
    blowup_fac = float(getattr(prop, "energy_guard_blowup", 50.0))
    diag_extra = bool(getattr(prop, "diag_extra", False))

    print(f"EnergyGuard(能量守护): {onoff(guard_en)} every={guard_every} skip-first={guard_skip}× | blowup×{fmt(blowup_fac)}  # 异常能量监测")
    print(f"Diagnostics(诊断输出): extra={onoff(diag_extra)}  # 扩展诊断开关")
    print("===================================================================\n")
