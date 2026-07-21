from __future__ import annotations
import json
from .device import xp, to_cpu
from .linear import lin_propagator, step_linear, step_linear_bk_nee_factorized
from .ionization import (
    intensity as inten_ion,
    make_Wfunc,
    field_amplitude_from_intensity,
    evolve_rho_time,_ion_input_domain
)
from .nonlinear import kerr_phase_from_deltan, plasma_phase, ib_alpha, apply_nonlinear, shock_intensity, operator_correct_scalar
from .heat import heat_Q_per_z
from .linear_full import step_linear_full_factorized, step_linear_full_3d
from .air_dispersion import n_of_omega
from .constants import c0
from .config import resolve_nonlinear_switches
from .raman import apply_isaacs_raman_operator_step, isaacs_raman_stage, make_raman_kernel, precompute_kernel_fft, raman_convolve_intensity, resolve_raman_rot_params
from .diagnostics import (
    intensity,
    pulse_energy,
    second_moment_radius,
    _fwhm_time_1d,
    parabola_peak,
    _fwhm_diameter_xy_center,
    validate_nonlinear_diagnostics,
)

def _linear_phase_per_meter(linear_model, k0, axes, K02_w=None, omega0=None, nee_denom_floor=1e-4):
    """è¿”å›çº¿æ€§ä¼ æ’­å­å¯¹åº”çš„ |kz| (rad/m) çš„ max å€¼ï¼Œç”¨äºä¼°è®¡ Î”Ï†_linear = kz_max * dz"""
    if linear_model == "paraxial":
        # ç›¸ä½ï¼šexp(i * (-kâŠ¥^2) dz / (2 k0))ï¼Œå¹…è§’/ç±³ = kâŠ¥^2 / (2 k0)
        kz_abs_max = float(xp.max(axes.kperp2) / (2.0 * k0))
    elif linear_model == "bk_nee":
        # Brabecâ€“Krausz NEE çº¿æ€§é¡¹ä¸»å¯¼çš„æ¨ªå‘è¡å°„ç›¸ä½ä¼°è®¡
        if omega0 is None:
            omega0 = float(getattr(axes, "omega0"))
        rel = axes.Omega / float(omega0)
        denom = 1.0 + rel
        denom_abs = xp.maximum(xp.abs(denom), float(nee_denom_floor))
        denom_sign = xp.where(denom >= 0.0, 1.0, -1.0)
        denom = denom_sign * denom_abs
        kz_abs_max = float(xp.max(xp.abs(axes.kperp2[None, ...] / (2.0 * k0 * denom[:, None, None]))))
    else:  # UPPE
        # K02_w = (n(Ï‰) Ï‰ / c)^2, kz = sqrt(K02_w - kâŠ¥^2)ï¼›å–å®éƒ¨çš„æœ€å¤§å€¼
        if K02_w is None:
            # é€€åŒ–ï¼šç”¨ n0 * Ï‰/c çš„ä¸­å¿ƒå€¼åšä¸‹ç•Œä¼°è®¡ï¼ˆæ›´ä¿å®ˆäº›ï¼‰
            omega_tot = axes.omega0 + axes.Omega
            k0w = (axes.n_w * omega_tot / c0) if hasattr(axes, "n_w") else (k0 * (omega_tot / axes.omega0))
            kz = xp.sqrt(xp.maximum(k0w[:, None, None]**2 - axes.kperp2[None, ...], 0.0))
        else:
            kz = xp.sqrt(xp.maximum(K02_w - axes.kperp2[None, ...], 0.0))
        kz_abs_max = float(xp.max(xp.abs(xp.real(kz))))
    return kz_abs_max


def _combine_raman_operator_diagnostics(parts, *, reference_energy):
    """Combine one or two isolated Raman operator substeps into one z-step record."""
    if not parts:
        return None
    if len(parts) == 1:
        result = dict(parts[0])
        result["operator_substep_count"] = 1
        result["energy_projection_max_scale_deviation"] = abs(
            float(result.get("energy_projection_scale", 1.0)) - 1.0)
        return result
    target_local = sum((part["target_local_fluence_loss_heun"] for part in parts))
    actual_local = sum((part["actual_local_fluence_loss"] for part in parts))
    target_global = sum(float(part["target_global_energy_loss_J"]) for part in parts)
    actual_global = sum(float(part["actual_global_energy_loss_J"]) for part in parts)
    local_residual = float(xp.max(
        xp.abs(actual_local - target_local)
        / xp.maximum(target_local, 1e-300)))
    global_residual = abs(actual_global - target_global) / max(
        target_global, float(reference_energy) * 1e-15, 1e-300)
    return {
        "target_local_fluence_loss_stage1": sum((part["target_local_fluence_loss_stage1"] for part in parts)),
        "target_local_fluence_loss_stage2": sum((part["target_local_fluence_loss_stage2"] for part in parts)),
        "target_local_fluence_loss_heun": target_local,
        "target_local_fluence_loss": target_local,
        "target_global_energy_loss_J": target_global,
        "actual_local_fluence_loss": actual_local,
        "actual_global_energy_loss_J": actual_global,
        "local_closure_residual": local_residual,
        "global_closure_residual": global_residual,
        "rhs_l2_norm_stage1": max(float(part["rhs_l2_norm_stage1"]) for part in parts),
        "rhs_l2_norm_stage2": max(float(part["rhs_l2_norm_stage2"]) for part in parts),
        "IR_max_stage1": max(float(part["IR_max_stage1"]) for part in parts),
        "IR_max_stage2": max(float(part["IR_max_stage2"]) for part in parts),
        "I_R_stage1": parts[-1].get("I_R_stage1"),
        "I_R_stage2": parts[-1].get("I_R_stage2"),
        "convolution_count": sum(int(part["convolution_count"]) for part in parts),
        "operator_substep_count": len(parts),
        "operator_walltime_s": sum(float(part["operator_walltime_s"]) for part in parts),
        "finite": all(bool(part["finite"]) for part in parts),
        "clipping_count": sum(int(part.get("clipping_count", 0)) for part in parts),
        "actual_loss_evaluation": parts[0].get(
            "actual_loss_evaluation", "legacy_fluence_subtraction"),
        "energy_projection_applied": any(
            bool(part.get("energy_projection_applied", False)) for part in parts),
        "energy_projection_iterations": sum(
            int(part.get("energy_projection_iterations", 0)) for part in parts),
        "energy_projection_max_scale_deviation": max(
            abs(float(part.get("energy_projection_scale", 1.0)) - 1.0)
            for part in parts),
        "energy_projection_initial_residual": max(
            float(part.get("energy_projection_initial_residual", 0.0))
            for part in parts),
    }


def _performance_sync(enabled):
    if enabled and getattr(xp, "__name__", "numpy") == "cupy":
        xp.cuda.Stream.null.synchronize()


# --- è½»é‡ CPU ä¾§ FWHM è®¡ç®—ï¼šå¯¹ 2D map åšåœ†å¹³å‡ï¼Œå†æ‰¾ 0.5Ã—å³°å€¼çš„åŠå¾„ ---
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
    # æ‰¾åˆ° prof é™åˆ° 0.5Ã—peak çš„ä½ç½®åšçº¿æ€§æ’å€¼
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
    return 2.0 * r_half  # ç›´å¾„ FWHM

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
    """æç®€ç¨³å®šç‰ˆï¼šå›ºå®šä¸€æ­¥åªåšä¸€æ¬¡å®‰å…¨ç¼©æ­¥ï¼ˆå¯é€‰è¿‘ç„¦åŠ å¯†ï¼‰ï¼Œæ ‡å‡† Strang åˆ†è£‚ã€‚
       äº§å‡ºç»Ÿä¸€çš„ diag å¥‘çº¦ï¼ˆè§å‡½æ•°å°¾éƒ¨ï¼‰ã€‚"""
    import time
    import numpy as _np

    # ---------- dtype ----------
    ctype = E.dtype
    rdtype = xp.float32 if ctype == xp.complex64 else xp.float64
    rdtype_np = _np.float32 if ctype == xp.complex64 else _np.float64

    # ---------- è¿‘ç„¦ç¼©æ­¥ ----------
    p = prop_conf
    switches = resolve_nonlinear_switches(p, raman_conf, ion_conf)
    measure_performance = bool(getattr(p, "measure_performance", False))
    dz_base = float(dz)
    use_focus_win = bool(getattr(p, "focus_window_step", False))
    z_center = getattr(p, "focus_center_m", None) or getattr(p, "z_focus_hint", None)
    z_half = float(getattr(p, "focus_halfwidth_m", 0.0))
    dz_focus = float(getattr(p, "dz_focus", dz_base))
    print(
        f"[focus-step] focus_center_m(local)={z_center if z_center is not None else 'None'}, "
        f"focus_halfwidth_m={z_half:.4e}, dz_base={dz_base:.4e}, dz_focus={dz_focus:.4e}"
    )

    # ---------- ç´¢å¼•/å‡ ä½• ----------
    Ny, Nx = E.shape[-2], E.shape[-1]
    y0, x0 = Ny // 2, Nx // 2
    t_arr = axes.t
    t0_idx = int(xp.argmin(xp.abs(t_arr)))

    save_every = max(1, int(record_every_z))
    save_count = 0


    # ---------- çº¿æ€§åˆ†æ”¯ ----------
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

    # ---------- æ‹‰æ›¼ï¼ˆå»¶è¿Ÿ Kerr + å¯é€‰å¸æ”¶æ¨¡å‹ï¼‰ ----------
    # The convolution and the field-feedback switches are intentionally
    # separate: an OFF feedback switch must not erase its raw diagnostic.
    use_raman = switches.compute_raman_convolution
    raman_absorb_on = switches.use_raman_absorption
    raman_absorption_compute = switches.compute_raman_absorption
    absorption_model = "poynting"
    omega_R = Gamma_R = None
    tau_fwhm_cfg = None
    n_rot_frac = 0.99
    R0_mode = "mom"
    R0_fixed = None
    n_R = 0.0
    n2_elec = float(n2)
    fR_ignored_rot_sinexp = False
    r_operator_mode = "legacy_split"

    if use_raman:
        fR_value = getattr(raman_conf, "f_R", None)
        fR = float(fR_value) if fR_value is not None else 0.0
        r_method = str(getattr(raman_conf, "method", "iir")).lower()
        r_chunk = int(getattr(raman_conf, "chunk_pixels", 65536))
        r_model = str(getattr(raman_conf, "model", "rot_sinexp")).lower()
        n_R = float(getattr(raman_conf, "n_R", 2.3e-23))
        r_operator_mode = str(getattr(raman_conf, "operator_mode", "legacy_split")).lower()
        fR_ignored_rot_sinexp = (r_model == "rot_sinexp")
        if fR_ignored_rot_sinexp:
            print(f"[Raman] model=rot_sinexp uses explicit n_R={n_R:.3e} m^2/W for phase/absorption; f_R={fR:.3g} ignored in phase channel.")
        if (r_method == "fft"):
            h = make_raman_kernel(axes.t, raman_conf)
            H_w = h
        else:
            H_w = None

        # The resolved Phase-2 switch controls whether this raw coefficient is
        # applied; the legacy raman fields only provide compatibility defaults.
        absorption_model = str(getattr(raman_conf, "absorption_model", "poynting")).lower()

        # closed_form éœ€è¦çš„å‚æ•°ï¼ˆéƒ½æœ‰é»˜è®¤ï¼‰
        omega_R, Gamma_R = resolve_raman_rot_params(
            T2=getattr(raman_conf, "T2", None),
            T_R=getattr(raman_conf, "T_R", None),
            omega_R=getattr(raman_conf, "omega_R", None),
            Gamma_R=getattr(raman_conf, "Gamma_R", None),
        )
        tau_fwhm_cfg = getattr(raman_conf, "tau_fwhm", None)
        n_rot_frac = float(getattr(raman_conf, "n_rot_frac", 0.99))
        R0_mode = str(getattr(raman_conf, "R0_mode", "mom")).lower()
        R0_fixed = float(getattr(raman_conf, "R0_fixed_m", 2.0e-4))
    else:
        H_w, fR, r_method, r_chunk = None, 0.0, "iir", 65536
    full_isaacs_mode = bool(use_raman and r_operator_mode == "full_isaacs_eq27")
    r_nonlinear_split_order = str(
        getattr(raman_conf, "nonlinear_split_order", "after_other")
    ).lower()

    # ---------- ç”µç¦»é€Ÿç‡ ----------


    ion_off = not switches.use_ionization_solver
    if ion_off:
        Wfunc, ion_input = None, "none"
    else:
        Wfunc = make_Wfunc(getattr(ion_conf, "model", "none"), ion_conf, omega0, n0)
        ion_input = getattr(Wfunc, "_expects", None)
        if ion_input in ("uses_E", "E"):
            ion_input = "E"
        elif ion_input in ("uses_I", "I"):
            ion_input = "I"
        else:
            ion_input = _ion_input_domain(ion_conf)  # å…œåº•
    use_ion_op_corr = bool(getattr(ion_conf, "use_ionization_operator_correction", False))
    ion_op_method = str(getattr(ion_conf, "ionization_operator_method", "tdiff")).lower()

    # ---------- åŸºçº¿èƒ½é‡ ----------
    I0 = intensity(E, n0)
    U0_baseline = float(pulse_energy(I0, dt, axes.dx, axes.dy)) + 1e-30
    energy_print_every = int(getattr(prop_conf, "energy_probe_every", 1))
    if energy_print_every > 0:
        print(f"[U] z={0.000:0.3f} m  U={U0_baseline: .3e} J  Î”rel={0.00:.2f}%")

    # ---------- è¯Šæ–­æ”¶é›† ----------
    z_axis_list, U_z_list = [], []
    I_max_z_list, rho_max_z_list = [], []
    I_onaxis_max_z_list, I_center_t0_z_list = [], []
    w_mom_z_list, rho_onaxis_max_list = [], []

    E_dep_z_list, E_dep_rot_z_list = [], []  # ç”µç¦»+IBã€æ‹‰æ›¼æ²‰ç§¯
    E_dep_total_z_list, E_dep_cumulative_z_list = [], []
    U_rel_change_z_list, U_step_change_z_list, E_loss_from_input_z_list = [], [], []
    fwhm_plasma_z_list, fwhm_fluence_z_list = [], []
    rho_onaxis_time_list = [] if record_onaxis_rho_time else None
    I_onaxis_max_interp_list,alpha_R_mean_z_list,alpha_R_closed_z_list,IR_max_z_list = [],[],[],[]
    delta_n_elec_max_z_list, delta_n_rot_max_z_list = [], []
    delta_n_elec_peak_z_list, delta_n_rot_peak_z_list = [], []
    alpha_R_max_z_list = []
    alpha_ion_raw_max_z_list, alpha_ion_corr_max_z_list = [], []
    alpha_ib_max_z_list, alpha_total_max_z_list, alpha_R_eff_z_list = [], [], []
    delta_n_plasma_min_z_list = []
    dphi_kerr_max_abs_z_list, dphi_elec_max_abs_z_list = [], []
    dphi_rot_max_abs_z_list, dphi_plasma_max_abs_z_list = [], []
    IR_abs_max_z_list = []
    delta_n_elec_applied_max_z_list, delta_n_rot_applied_max_z_list = [], []
    delta_n_plasma_applied_min_z_list = []
    dphi_elec_applied_max_abs_z_list, dphi_rot_applied_max_abs_z_list = [], []
    dphi_plasma_raw_max_abs_z_list, dphi_plasma_applied_max_abs_z_list = [], []
    alpha_ion_applied_max_z_list, alpha_R_raw_max_z_list, alpha_R_applied_max_z_list = [], [], []
    rho_N2_max_z_list, rho_O2_max_z_list = [], []
    rho_N2_at_rho_total_max_z_list, rho_O2_at_rho_total_max_z_list = [], []
    rho_O2_fraction_at_rho_total_max_z_list = []
    dz_used_z_list,÷M½¶‰Ëkºwµçd¤¤(€€€€€€€€€€€É…µ…¹}•¹•Éå}ÁÉ½©•Ñ¥½¹}Í…±•}‘•Ù¥…Ñ¥½¹}é}±¥ÍĞ¹…ÁÁ•¹¡™±½…Ğ (€€€€€€€€€€€€€€€É…µ…¹}ÍÑ•Á}‘¥…œ¹•Ğ ‰•¹•Éå}ÁÉ½©•Ñ¥½¹}µ…á}Í…±•}‘•Ù¥…Ñ¥½¸ˆ°€À¸À¤¤¤(€€€€€€€€€€€É…µ…¹}•¹•Éå}ÁÉ½©•Ñ¥½¹}¥¹¥Ñ¥…±}É•Í¥‘Õ…±}é}±¥ÍĞ¹…ÁÁ•¹¡™±½…Ğ (€€€€€€€€€€€€€€€É…µ…¹}ÍÑ•Á}‘¥…œ¹•Ğ ‰•¹•Éå}ÁÉ½©•Ñ¥½¹}¥¹¥Ñ¥…±}É•Í¥‘Õ…°ˆ°€À¸À¤¤¤(€€€€€€€€€€€±¥¹•…É}İ…±±Ñ¥µ•}ÍÑ•Á}é}±¥ÍĞ¹…ÁÁ•¹¡™±½…Ğ¡±¥¹•…É}İ…±±Ñ¥µ•}ÍÑ•À¤¤(€€€€€€€€€€€¥½¹¥é…Ñ¥½¹}İ…±±Ñ¥µ•}ÍÑ•Á}é}±¥ÍĞ¹…ÁÁ•¹¡™±½…Ğ¡¥½¹¥é…Ñ¥½¹}İ…±±Ñ¥µ•}ÍÑ•À¤¤(€€€€€€€€€€€Ñ½Ñ…±}İ…±±Ñ¥µ•}ÍÑ•Á}é}±¥ÍĞ¹…ÁÁ•¹¡™±½…Ğ¡Ñ½Ñ…±}İ…±±Ñ¥µ•}ÍÑ•À¤¤(€€€€€€€€€€€ÁÕ}…±±½…Ñ•‘}ÍÑ•Á}‰åÑ•Í}±¥ÍĞ¹…ÁÁ•¹¡¥¹Ğ¡ÁÕ}…±±½…Ñ•‘}ÍÑ•À¤¤(€€€€€€€€€€€ÁÕ}É•Í•ÉÙ•‘}ÍÑ•Á}‰åÑ•Í}±¥ÍĞ¹…ÁÁ•¹¡¥¹Ğ¡ÁÕ}É•Í•ÉÙ•‘}ÍÑ•À¤¤(€€€€€€€€€€€‘•±Ñ…}¹}Á±…Íµ…}µ¥¹}é}±¥ÍĞ¹…ÁÁ•¹¡™±½…Ğ¡áÀ¹µ¥¸¡‘Á¡¥}Á}É…Ü¤¤€¼€¡™±½…Ğ¡¬À¤€¨‘é}ÑÉä¤¤(€€€€€€€€€€€‘•±Ñ…}¹}•±•}…ÁÁ±¥•‘}µ…á}é}±¥ÍĞ¹…ÁÁ•¹¡™±½…Ğ¡áÀ¹µ…à¡‘•±Ñ…}¹}•±•}…ÁÁ±¥•¤¤¤(€€€€€€€€€€€‘•±Ñ…}¹}É½Ñ}…ÁÁ±¥•‘}µ…á}é}±¥ÍĞ¹…ÁÁ•¹¡™±½…Ğ¡áÀ¹µ…à¡‘•±Ñ…}¹}É½Ñ}…ÁÁ±¥•¤¤¤(€€€€€€€€€€€‘•±Ñ…}¹}Á±…Íµ…}…ÁÁ±¥•‘}µ¥¹}é}±¥ÍĞ¹…ÁÁ•¹¡™±½…Ğ¡áÀ¹µ¥¸¡‘Á¡¥}À¤¤€¼€¡™±½…Ğ¡¬À¤€¨‘é}ÑÉä¤¤(€€€€€€€€€€€‘Á¡¥}­•ÉÉ}µ…á}…‰Í}é}±¥ÍĞ¹…ÁÁ•¹¡™±½…Ğ¡áÀ¹µ…à¡áÀ¹…‰Ì¡‘Á¡¥}¬¤¤¤¤(€€€€€€€€€€€‘Á¡¥}•±•}µ…á}…‰Í}é}±¥ÍĞ¹…ÁÁ•¹¡™±½…Ğ¡áÀ¹µ…à¡áÀ¹…‰Ì¡™±½…Ğ¡¬À¤€¨‘•±Ñ…}¹}•±•Œ€¨‘é}ÑÉä¤¤¤¤(€€€€€€€€€€€‘Á¡¥}É½Ñ}µ…á}…‰Í}é}±¥ÍĞ¹…ÁÁ•¹¡™±½…Ğ¡áÀ¹µ…à¡áÀ¹…‰Ì¡™±½…Ğ¡¬À¤€¨‘•±Ñ…}¹}É½Ğ€¨‘é}ÑÉä¤¤¤¤(€€€€€€€€€€€‘Á¡¥}•±•}…ÁÁ±¥•‘}µ…á}…‰Í}é}±¥ÍĞ¹…ÁÁ•¹¡™±½…Ğ¡áÀ¹µ…à¡áÀ¹…‰Ì¡™±½…Ğ¡¬À¤€¨‘•±Ñ…}¹}•±•}…ÁÁ±¥•€¨‘é}ÑÉä¤¤¤¤(€€€€€€€€€€€‘Á¡¥}É½Ñ}…ÁÁ±¥•‘}µ…á}…‰Í}é}±¥ÍĞ¹…ÁÁ•¹¡™±½…Ğ¡áÀ¹µ…à¡áÀ¹…‰Ì¡™±½…Ğ¡¬À¤€¨‘•±Ñ…}¹}É½Ñ}…ÁÁ±¥•€¨‘é}ÑÉä¤¤¤¤(€€€€€€€€€€€‘Á¡¥}Á±…Íµ…}É…İ}µ…á}…‰Í}é}±¥ÍĞ¹…ÁÁ•¹¡™±½…Ğ¡áÀ¹µ…à¡áÀ¹…‰Ì¡‘Á¡¥}Á}É…Ü¤¤¤¤(€€€€€€€€€€€‘Á¡¥}Á±…Íµ…}…ÁÁ±¥•‘}µ…á}…‰Í}é}±¥ÍĞ¹…ÁÁ•¹¡™±½…Ğ¡áÀ¹µ…à¡áÀ¹…‰Ì¡‘Á¡¥}À¤¤¤¤(€€€€€€€€€€€‘Á¡¥}Á±…Íµ…}µ…á}…‰Í}é}±¥ÍĞ¹…ÁÁ•¹¡™±½…Ğ¡áÀ¹µ…à¡áÀ¹…‰Ì¡‘Á¡¥}À¤¤¤¤((€€€€€€€€€€€€Œ]!7¾òk¶'šï–¶C’öO¦k¦Lƒ’â8ƒ¢÷¦?–¾–ê˜(€€€€€€€€€€€™İ¡µ}Á±…Íµ„€ô}™İ¡µ}‘¥…µ•Ñ•É}áå}•¹Ñ•È¡É¡½}µ…áĞ°…á•Ì°àÀ°äÀ¤(€€€€€€€€€€€™İ¡µ}™±Ô€€€€ô}™İ¡µ}‘¥…µ•Ñ•É}áå}•¹Ñ•È¡É°€€€€€…á•Ì°àÀ°äÀ¤(€€€€€€€€€€€™İ¡µ}Á±…Íµ…}é}±¥ÍĞ¹…ÁÁ•¹¡™±½…Ğ¡™İ¡µ}Á±…Íµ„¤¤(€€€€€€€€€€€™İ¡µ}™±Õ•¹•}é}±¥ÍĞ¹…ÁÁ•¹¡™±½…Ğ¡™İ¡µ}™±Ô¤¤((€€€€€€€€€€€€Œƒ¢÷¦?–N£–×š&O–6À((€€€€€€€€€€€¥˜•¹•Éå}ÁÉ¥¹Ñ}•Ù•Éä€ø€Àè(€€€€€€€€€€€€€€€ÍÑ•ÁÍ}‘½¹”€ô±•¸¡é}…á¥Í}±¥ÍĞ¤(€€€€€€€€€€€€€€€¥˜€¡ÍÑ•ÁÍ}‘½¹”€”•¹•Éå}ÁÉ¥¹Ñ}•Ù•Éä€ôô€À¤½È€¡é}¹½Ü€øôé}µ…à€´€Å”´ÄØ¤è(€€€€€€€€€€€€€€€€€€€‘É•°€ô€ÄÀÀ¸À€¨€¡U}¹½Ü€´TÁ}‰…Í•±¥¹”¤€¼TÁ}‰…Í•±¥¹”(€€€€€€€€€€€€€€€€€€€ÁÉ¥¹Ğ¡˜‰mUtèõíé}¹½ÜèÀ¸Í™ô´€TõíU}¹½Üè€¸Í•ô(€ƒ:QÉ•°õí‘É•°è¸É™ô”ˆ¤(€€€€€€€€€€€€€€€€€€€¥˜‰½½°¡•Ñ…ÑÑÈ¡À°€‰‘¥…}•áÑÉ„ˆ°…±Í”¤¤è(€€€€€€€€€€€€€€€€€€€€€€€ÁÉ¥¹Ğ (€€€€€€€€€€€€€€€€€€€€€€€€€€€˜‰m91tèõíé}¹½ÜèÀ¸Í™ô´€%Á¬õí%}µ…á}é}±¥ÍÑl´Åtè¸Í•ô\½µxÈ€€ˆ(€€€€€€€€€€€€€€€€€€€€€€€€€€€˜‹:Q¹}”õí‘•±Ñ…}¹}•±•}µ…á}é}±¥ÍÑl´Åtè¸Í•ô€ƒ:Q¹}Hõí‘•±Ñ…}¹}É½Ñ}µ…á}é}±¥ÍÑl´Åtè¸Í•ô€€ˆ(€€€€€€€€€€€€€€€€€€€€€€€€€€€˜‰%Hõí%I}…‰Í}µ…á}é}±¥ÍÑl´Åtè¸Í•ô\½µxÈ€€ˆ(€€€€€€€€€€€€€€€€€€€€€€€€€€€˜‹:Q¹}À±µ¥¸õí‘•±Ñ…}¹}Á±…Íµ…}µ¥¹}é}±¥ÍÑl´Åtè¸Í•ô€€ˆ(€€€€€€€€€€€€€€€€€€€€€€€€€€€˜‰ó>}-ğõí‘Á¡¥}­•ÉÉ}µ…á}…‰Í}é}±¥ÍÑl´Åtè¸Í•ôÉ…€€ˆ(€€€€€€€€€€€€€€€€€€€€€€€€€€€˜‰ó>}Áğõí‘Á¡¥}Á±…Íµ…}µ…á}…‰Í}é}±¥ÍÑl´Åtè¸Í•ôÉ…€€ˆ(€€€€€€€€€€€€€€€€€€€€€€€€€€€˜‹:Å}¥½¸¡É…Ü½…ÁÁ°¤õí…±Á¡…}¥½¹}½ÉÉ}µ…á}é}±¥ÍÑl´Åtè¸Í•ô½í…±Á¡…}¥½¹}…ÁÁ±¥•‘}µ…á}é}±¥ÍÑl´Åtè¸Í•ô€€ˆ(€€€€€€€€€€€€€€€€€€€€€€€€€€€˜‹:Å}%õí…±Á¡…}¥‰}µ…á}é}±¥ÍÑl´Åtè¸Í•ô€ƒ:Å}H¡É…Ü½…ÁÁ°¤õí…±Á¡…}I}É…İ}µ…á}é}±¥ÍÑl´Åtè¸Í•ô½í…±Á¡…}I}…ÁÁ±¥•‘}µ…á}é}±¥ÍÑl´Åtè¸Í•ôµx´Ä€€ˆ(€€€€€€€€€€€€€€€€€€€€€€€€€€€˜‰‘•Àõí}‘•Á}Ñ½Ñ…±}é}±¥ÍÑl´Åtè¸Í•ô(ˆ(€€€€€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€€Œ€´´´µ½¹¥Ñ½ÈAAQ}¤…Àµ¡¥Ğ€¡…™Ñ•È•Ù½±Ù•}É¡½}Ñ¥µ”¤€´´´(€€€€€€€€€€€€€€€€€€€]}…Á}ÕÍ•€ô™±½…Ğ¡•Ñ…ÑÑÈ¡¥½¹}½¹˜°€‰]}…Àˆ°€À¸À¤¤(€€€€€€€€€€€€€€€€€€€¥˜]}…Á}ÕÍ•€ø€À¸Àè(€€€€€€€€€€€€€€€€€€€€€€€Ñ¡È€ô€À¸äää€¨]}…Á}ÕÍ•(€€€€€€€€€€€€€€€€€€€€€€€¡¥ÑÌ€ô€À(€€€€€€€€€€€€€€€€€€€€€€€Ñ½Ñ…°€ô€À(€€€€€€€€€€€€€€€€€€€€€€€€Œƒ–>¿¦'š*÷š‚ß’î—¢şo’âš¶—¦f7–ò¦R¾òiÑ}ÍÑÉ¥‘”ôÄƒ¢†£’ë’â7š*÷š‚Ü(€€€€€€€€€€€€€€€€€€€€€€€Ñ}ÍÑÉ¥‘”€ô€Ä(€€€€€€€€€€€€€€€€€€€€€€€™½È¥Ğ¥¸É…¹” À°]Ğ¹Í¡…Á•lÁt°Ñ}ÍÑÉ¥‘”¤è(€€€€€€€€€€€€€€€€€€€€€€€€€€€™É´€ô]Ñm¥Ñt€€Œm9ä±9áw¾ò3š¶“–’–>«’òk’âÓš^Û–"¦7’â’â«–Â?š:§‚(€€€€€€€€€€€€€€€€€€€€€€€€€€€¡¥ÑÌ€¬ô¥¹Ğ¡áÀ¹½Õ¹Ñ}¹½¹é•É¼¡™É´€øôÑ¡È¤¤(€€€€€€€€€€€€€€€€€€€€€€€€€€€Ñ½Ñ…°€¬ô™É´¹Í¥é”(€€€€€€€€€€€€€€€€€€€€€€€¡¥Ñ}™É…Œ€ô¡¥ÑÌ€¼µ…à Ä°Ñ½Ñ…°¤(€€€€€€€€€€€€€€€€€€€€€€€ÁÉ¥¹Ğ¡˜‰mèõíèè¸Í™ôµtAAQ}¤…Àµ¡¥Ğ€ôí¡¥Ñ}™É…Œ€¨€ÄÀÀè¸Í™ô”€¡…Àõí]}…Á}ÕÍ•è¸É•ô¤ˆ¤((€€€€€€€€Œƒ–&7¢şlè€˜ƒ¢şo–ê˜(€€€€€€€è€¬ô‘é}ÑÉä(€€€€€€€Á”€ô¥¹Ğ¡•Ñ…ÑÑÈ¡À°€‰ÁÉ½É•ÍÍ}•Ù•Éå}èˆ°€À¤½È€À¤(€€€€€€€¥˜Á”…¹€ ¡±•¸¡é}…á¥Í}±¥ÍĞ¤€”Á”¤€ôô€À½Èè€øôé}µ…à€´€Å”´ÄØ¤è(€€€€€€€€€€€™É…Œ€ôè€¼é}µ…à(€€€€€€€€€€€•±…ÁÍ•€ôÑ¥µ”¹Á•É™}½Õ¹Ñ•È ¤€´ĞÀ(€€€€€€€€€€€•Ñ„€ô•±…ÁÍ•€¼µ…à¡™É…Œ°€Å”´ä¤€¨€ Ä¸À€´™É…Œ¤(€€€€€€€€€€€ÁÉ¥¹Ğ¡˜‰métíèè¸Í™ô½íé}µ…àè¸Í™ô´€¡í™É…Œ¨ÄÀÀèØ¸É™ô”¤€•±…ÁÍ•í•±…ÁÍ•èØ¸Å™õÌ€Qí•Ñ„èØ¸Å™õÌˆ¤((€€€€Œ€´´´´´´´´´´ƒš&O–2€´´´´´´´´´´(€€€‘¥…œ€ôì(€€€€€€€€‰é}…á¥Ìˆè€€€€€€€€€€€€€€€€€€}¹À¹…Í…ÉÉ…ä¡é}…á¥Í}±¥ÍĞ°€€€€€€€€€€€€€‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰U}èˆè€€€€€€€€€€€€€€€€€€€€€}¹À¹…Í…ÉÉ…ä¡U}é}±¥ÍĞ°€€€€€€€€€€€€€€€€‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰%}µ…á}èˆè€€€€€€€€€€€€€€€€€}¹À¹…Í…ÉÉ…ä¡%}µ…á}é}±¥ÍĞ°€€€€€€€€€€€€‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰%}½¹…á¥Í}µ…á}èˆè€€€€€€€€€€}¹À¹…Í…ÉÉ…ä¡%}½¹…á¥Í}µ…á}é}±¥ÍĞ°€€€€€‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰%}•¹Ñ•É}ĞÁ}èˆè€€€€€€€€€€€}¹À¹…Í…ÉÉ…ä¡%}•¹Ñ•É}ĞÁ}é}±¥ÍĞ°€€€€€€‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰İ}µ½µ}èˆè€€€€€€€€€€€€€€€€€}¹À¹…Í…ÉÉ…ä¡İ}µ½µ}é}±¥ÍĞ°€€€€€€€€€€€€‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰É¡½}µ…á}èˆè€€€€€€€€€€€€€€€}¹À¹…Í…ÉÉ…ä¡É¡½}µ…á}é}±¥ÍĞ°€€€€€€€€€€‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰É¡½}½¹…á¥Í}µ…á}èˆè€€€€€€€€}¹À¹…Í…ÉÉ…ä¡É¡½}½¹…á¥Í}µ…á}±¥ÍĞ°€€€€€‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰}‘•Á}èˆè€€€€€€€€€€€€€€€€€}¹À¹…Í…ÉÉ…ä¡}‘•Á}é}±¥ÍĞ°€€€€€€€€€€€€‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°€€€ŒƒR×šì­%(€€€€€€€€‰}‘•Á}Ñ½Ñ…±}èˆè€€€€€€€€€€€}¹À¹…Í…ÉÉ…ä¡}‘•Á}Ñ½Ñ…±}é}±¥ÍĞ°€€€€€€‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰}‘•Á}ÕµÕ±…Ñ¥Ù•}èˆè€€€€€€}¹À¹…Í…ÉÉ…ä¡}‘•Á}ÕµÕ±…Ñ¥Ù•}é}±¥ÍĞ°€‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰U}É•±}¡…¹•}èˆè€€€€€€€€€€}¹À¹…Í…ÉÉ…ä¡U}É•±}¡…¹•}é}±¥ÍĞ°€€€€€‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰U}ÍÑ•Á}¡…¹•}èˆè€€€€€€€€€}¹À¹…Í…ÉÉ…ä¡U}ÍÑ•Á}¡…¹•}é}±¥ÍĞ°€€€€‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰}±½ÍÍ}™É½µ}¥¹ÁÕÑ}èˆè€€€€€}¹À¹…Í…ÉÉ…ä¡}±½ÍÍ}™É½µ}¥¹ÁÕÑ}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰É¡½}8É}µ…á}èˆè}¹À¹…Í…ÉÉ…ä¡É¡½}8É}µ…á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰É¡½}<É}µ…á}èˆè}¹À¹…Í…ÉÉ…ä¡É¡½}<É}µ…á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰É¡½}8É}…Ñ}É¡½}Ñ½Ñ…±}µ…á}èˆè}¹À¹…Í…ÉÉ…ä¡É¡½}8É}…Ñ}É¡½}Ñ½Ñ…±}µ…á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰É¡½}<É}…Ñ}É¡½}Ñ½Ñ…±}µ…á}èˆè}¹À¹…Í…ÉÉ…ä¡É¡½}<É}…Ñ}É¡½}Ñ½Ñ…±}µ…á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰É¡½}<É}™É…Ñ¥½¹}…Ñ}É¡½}Ñ½Ñ…±}µ…á}èˆè}¹À¹…Í…ÉÉ…ä¡É¡½}<É}™É…Ñ¥½¹}…Ñ}É¡½}Ñ½Ñ…±}µ…á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰‘é}ÕÍ•‘}èˆè}¹À¹…Í…ÉÉ…ä¡‘é}ÕÍ•‘}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰…‘…ÁÑ¥Ù•}É•©•Ñ¥½¹}½Õ¹Ñ}èˆè}¹À¹…Í…ÉÉ…ä¡…‘…ÁÑ¥Ù•}É•©•Ñ¥½¹}½Õ¹Ñ}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰Í…™•Ñå}µ½‘•}ÑÉ¥•É}½Õ¹Ñ}èˆè}¹À¹…Í…ÉÉ…ä¡Í…™•Ñå}µ½‘•}ÑÉ¥•É}½Õ¹Ñ}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰Í…™•Ñå}µ½‘•}•Ù•¹Ñ}ÍÕµµ…Éäˆè}¹À¹…Í…ÉÉ…ä¡©Í½¸¹‘ÕµÁÌ¡ì(€€€€€€€€€€€€‰½¹™¥ÕÉ•‘}µ½‘”ˆèÍÑÈ¡•Ñ…ÑÑÈ¡À°€‰Í…™•Ñå}µ½‘”ˆ°€‰½™˜ˆ¤¤°(€€€€€€€€€€€€‰ÑÉ¥•É}½Õ¹Ğˆè¥¹Ğ¡Í…™•Ñå}µ½‘•}ÑÉ¥•É}½Õ¹Ğ¤°(€€€€€€€€€€€€‰É•©•Ñ¥½¹}½Õ¹Ğˆè¥¹Ğ¡…‘…ÁÑ¥Ù•}É•©•Ñ¥½¹}½Õ¹Ğ¤°(€€€€€€€€€€€€‰Í½ÕÉ”ˆè€‰±¥Ù”ÁÉ½Á……Ñ¥½¸µ±½½À½Õ¹Ñ•ÉÌˆ°(€€€€€€€ô°Í½ÉÑ}­•åÌõQÉÕ”¤¤°(€€€€€€€€‰ÁÉ½Á……Ñ¥½¹}½‰Í•ÉÙ…‰¥±¥Ñå}Í¡•µ„ˆè}¹À¹…Í…ÉÉ…ä ‰­¡é}™¥±…µ•¹Ğ¹ÁÉ½Á……Ñ¥½¹}½‰Í•ÉÙ…‰¥±¥Ñä¹ØÄˆ¤°((€€€€€€€€‰™İ¡µ}Á±…Íµ…}èˆè€€€€€€€€€€€}¹À¹…Í…ÉÉ…ä¡™İ¡µ}Á±…Íµ…}é}±¥ÍĞ°€€€€€€‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰™İ¡µ}™±Õ•¹•}èˆè€€€€€€€€€€}¹À¹…Í…ÉÉ…ä¡™İ¡µ}™±Õ•¹•}é}±¥ÍĞ°€€€€€‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰%}½¹…á¥Í}µ…á}¥¹Ñ•ÉÁ}±¥ÍĞˆè}¹À¹…Í…ÉÉ…ä¡%}½¹…á¥Í}µ…á}¥¹Ñ•ÉÁ}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰É…µ…¹}…‰Í½ÉÁÑ¥½¹}½¸ˆè€€€€€‰½½°¡ÕÍ•}É…µ…¸…¹É…µ…¹}…‰Í½É‰}½¸¤°(€€€€€€€€‰É…µ…¹}…‰Í½ÉÁÑ¥½¹}…±Õ±…Ñ•ˆè‰½½°¡ÕÍ•}É…µ…¸…¹É…µ…¹}…‰Í½ÉÁÑ¥½¹}½µÁÕÑ”¤°(€€€€€€€€‰É…µ…¹}½Á•É…Ñ½É}µ½‘”ˆè}¹À¹…Í…ÉÉ…ä¡É}½Á•É…Ñ½É}µ½‘”¤°(€€€€€€€€‰É…µ…¹}½Á•É…Ñ½É}™••‘‰…­}•¹…‰±•ˆè‰½½°¡Íİ¥Ñ¡•Ì¹ÕÍ•}É…µ…¹}™Õ±±}½Á•É…Ñ½È¤°(€€€€€€€€‰É…µ…¹}½Á•É…Ñ½É}…ÁÁ±¥•ˆè}¹À¹…Í…ÉÉ…ä¡É…µ…¹}½Á•É…Ñ½É}…ÁÁ±¥•‘}é}±¥ÍĞ°‘ÑåÁ”õ}¹À¹‰½½±|¤°(€€€€€€€€‰É…µ…¹}É¡Í}°É}¹½É´ˆè}¹À¹…Í…ÉÉ…ä¡É…µ…¹}É¡Í}°É}¹½Éµ}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰É…µ…¹}%I}µ…á}É…Üˆè}¹À¹…Í…ÉÉ…ä¡É…µ…¹}%I}µ…á}É…İ}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰É…µ…¹}Ñ…É•Ñ}±½ÍÍ}ÍÑ•Á}(ˆè}¹À¹…Í…ÉÉ…ä¡É…µ…¹}Ñ…É•Ñ}±½ÍÍ}ÍÑ•Á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰É…µ…¹}…ÑÕ…±}±½ÍÍ}ÍÑ•Á}(ˆè}¹À¹…Í…ÉÉ…ä¡É…µ…¹}…ÑÕ…±}±½ÍÍ}ÍÑ•Á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰É…µ…¹}±½ÍÕÉ•}É•Í¥‘Õ…±}ÍÑ•Àˆè}¹À¹…Í…ÉÉ…ä¡É…µ…¹}±½ÍÕÉ•}É•Í¥‘Õ…±}ÍÑ•Á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰É…µ…¹}Ñ…É•Ñ}±½ÍÍ}ÕµÕ±…Ñ¥Ù•}(ˆè}¹À¹…Í…ÉÉ…ä¡É…µ…¹}Ñ…É•Ñ}±½ÍÍ}ÕµÕ±…Ñ¥Ù•}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰É…µ…¹}…ÑÕ…±}±½ÍÍ}ÕµÕ±…Ñ¥Ù•}(ˆè}¹À¹…Í…ÉÉ…ä¡É…µ…¹}…ÑÕ…±}±½ÍÍ}ÕµÕ±…Ñ¥Ù•}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰É…µ…¹}ÕµÕ±…Ñ¥Ù•}±½ÍÕÉ•}É•Í¥‘Õ…°ˆè}¹À¹…Í…ÉÉ…ä¡É…µ…¹}ÕµÕ±…Ñ¥Ù•}±½ÍÕÉ•}É•Í¥‘Õ…±}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰É…µ…¹}½¹Ù½±ÕÑ¥½¹}½Õ¹Ñ}ÍÑ•Àˆè}¹À¹…Í…ÉÉ…ä¡É…µ…¹}½¹Ù½±ÕÑ¥½¹}½Õ¹Ñ}ÍÑ•Á}é}±¥ÍĞ°‘ÑåÁ”õ}¹À¹¥¹ĞØĞ¤°(€€€€€€€€‰É…µ…¹}½Á•É…Ñ½É}ÍÕ‰ÍÑ•Á}½Õ¹Ğˆè}¹À¹…Í…ÉÉ…ä¡É…µ…¹}½Á•É…Ñ½É}ÍÕ‰ÍÑ•Á}½Õ¹Ñ}é}±¥ÍĞ°‘ÑåÁ”õ}¹À¹¥¹ĞØĞ¤°(€€€€€€€€‰É…µ…¹}½Á•É…Ñ½É}İ…±±Ñ¥µ•}ÍÑ•Á}Ìˆè}¹À¹…Í…ÉÉ…ä¡É…µ…¹}½Á•É…Ñ½É}İ…±±Ñ¥µ•}ÍÑ•Á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰É…µ…¹}•¹•Éå}ÁÉ½©•Ñ¥½¹}¥Ñ•É…Ñ¥½¹Ìˆè}¹À¹…Í…ÉÉ…ä (€€€€€€€€€€€É…µ…¹}•¹•Éå}ÁÉ½©•Ñ¥½¹}¥Ñ•É…Ñ¥½¹Í}é}±¥ÍĞ°‘ÑåÁ”õ}¹À¹¥¹ĞØĞ¤°(€€€€€€€€‰É…µ…¹}•¹•Éå}ÁÉ½©•Ñ¥½¹}Í…±•}‘•Ù¥…Ñ¥½¸ˆè}¹À¹…Í…ÉÉ…ä (€€€€€€€€€€€É…µ…¹}•¹•Éå}ÁÉ½©•Ñ¥½¹}Í…±•}‘•Ù¥…Ñ¥½¹}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰É…µ…¹}•¹•Éå}ÁÉ½©•Ñ¥½¹}¥¹¥Ñ¥…±}É•Í¥‘Õ…°ˆè}¹À¹…Í…ÉÉ…ä (€€€€€€€€€€€É…µ…¹}•¹•Éå}ÁÉ½©•Ñ¥½¹}¥¹¥Ñ¥…±}É•Í¥‘Õ…±}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰±¥¹•…É}İ…±±Ñ¥µ•}ÍÑ•Á}Ìˆè}¹À¹…Í…ÉÉ…ä¡±¥¹•…É}İ…±±Ñ¥µ•}ÍÑ•Á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰¥½¹¥é…Ñ¥½¹}İ…±±Ñ¥µ•}ÍÑ•Á}Ìˆè}¹À¹…Í…ÉÉ…ä¡¥½¹¥é…Ñ¥½¹}İ…±±Ñ¥µ•}ÍÑ•Á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰Ñ½Ñ…±}İ…±±Ñ¥µ•}ÍÑ•Á}Ìˆè}¹À¹…Í…ÉÉ…ä¡Ñ½Ñ…±}İ…±±Ñ¥µ•}ÍÑ•Á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰ÁÕ}…±±½…Ñ•‘}ÍÑ•Á}‰åÑ•Ìˆè}¹À¹…Í…ÉÉ…ä¡ÁÕ}…±±½…Ñ•‘}ÍÑ•Á}‰åÑ•Í}±¥ÍĞ°‘ÑåÁ”õ}¹À¹¥¹ĞØĞ¤°(€€€€€€€€‰ÁÕ}É•Í•ÉÙ•‘}ÍÑ•Á}‰åÑ•Ìˆè}¹À¹…Í…ÉÉ…ä¡ÁÕ}É•Í•ÉÙ•‘}ÍÑ•Á}‰åÑ•Í}±¥ÍĞ°‘ÑåÁ”õ}¹À¹¥¹ĞØĞ¤°(€€€€€€€€‰Á•É™½Éµ…¹•}µ•…ÍÕÉ•µ•¹Ñ}•¹…‰±•ˆè‰½½°¡µ•…ÍÕÉ•}Á•É™½Éµ…¹”¤°(€€€€€€€€‰‘•±Ñ…}¹}É½Ñ}…ÁÁ±¥•‘}Í•µ…¹Ñ¥Ìˆè}¹À¹…Í…ÉÉ…ä (€€€€€€€€€€€€‰¹½Ñ}…ÁÁ±¥…‰±•}™Õ±±}½µÁ±•á}½Á•É…Ñ½Èˆ¥˜™Õ±±}¥Í……Í}µ½‘”•±Í”€‰ÍÁ±¥Ñ}‘•±…å•‘}¥¹‘•á}…ÁÁ±¥•ˆ(€€€€€€€€¤°(€€€€€€€€‰É…µ…¹}±½ÍÕÉ•}É•Í¥‘Õ…±}Í•µ…¹Ñ¥Ìˆè}¹À¹…Í…ÉÉ…ä (€€€€€€€€€€€€‰™¥•±‘}ÙÍ}•ÄÄÀˆ¥˜™Õ±±}¥Í……Í}½¸•±Í”€‰¹½Ñ}…ÁÁ±¥…‰±•}™••‘‰…­}½™™}½É}±•…äˆ(€€€€€€€€¤°(€€€€€€€€‰É…µ…¹}…ÑÕ…±}±½ÍÍ}•Ù…±Õ…Ñ¥½¸ˆè}¹À¹…Í…ÉÉ…ä (€€€€€€€€€€€€‰ÍÑ…‰±•}½µÁ½¹•¹Ñ}‘¥™™•É•¹•}™±½…ĞØĞˆ¥˜™Õ±±}¥Í……Í}µ½‘”(€€€€€€€€€€€•±Í”€‰±•…å}™±Õ•¹•}ÍÕ‰ÑÉ…Ñ¥½¸ˆ(€€€€€€€€¤°(€€€€€€€€‰E…}É…µ…¸ˆèÑ½}ÁÔ¡E…}É…µ…¸¤¹…ÍÑåÁ”¡É‘ÑåÁ•}¹À°½Áäõ…±Í”¤°(€€€€€€€€‰¥½¹¥é…Ñ¥½¹}±½ÍÍ}½¸ˆè‰½½°¡Íİ¥Ñ¡•Ì¹ÕÍ•}¥½¹¥é…Ñ¥½¹}±½ÍÌ¤°(€€€€€€€€‰¥½¹¥é…Ñ¥½¹}Í½±Ù•É}½¸ˆè‰½½°¡Íİ¥Ñ¡•Ì¹ÕÍ•}¥½¹¥é…Ñ¥½¹}Í½±Ù•È¤°(€€€€€€€€‰}‘•Á}É½Ñ}èˆè}¹À¹…Í…ÉÉ…ä¡}‘•Á}É½Ñ}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰…±Á¡…}I}µ…á}èˆè}¹À¹…Í…ÉÉ…ä¡…±Á¡…}I}µ…á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰…±Á¡…}I}µ•…¹}èˆè}¹À¹…Í…ÉÉ…ä¡…±Á¡…}I}µ•…¹}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰…±Á¡…}I}•™™}èˆè}¹À¹…Í…ÉÉ…ä¡…±Á¡…}I}•™™}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰…±Á¡…}I}±½Í•‘}èˆè}¹À¹…Í…ÉÉ…ä¡…±Á¡…}I}±½Í•‘}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰…±Á¡…}I}É…İ}µ…á}èˆè}¹À¹…Í…ÉÉ…ä¡…±Á¡…}I}É…İ}µ…á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰…±Á¡…}I}…ÁÁ±¥•‘}µ…á}èˆè}¹À¹…Í…ÉÉ…ä¡…±Á¡…}I}…ÁÁ±¥•‘}µ…á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰%I}µ…á}èˆè}¹À¹…Í…ÉÉ…ä¡%I}µ…á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰%I}…‰Í}µ…á}èˆè}¹À¹…Í…ÉÉ…ä¡%I}…‰Í}µ…á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰‘•±Ñ…}¹}•±•}µ…á}èˆè}¹À¹…Í…ÉÉ…ä¡‘•±Ñ…}¹}•±•}µ…á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰‘•±Ñ…}¹}É½Ñ}µ…á}èˆè}¹À¹…Í…ÉÉ…ä¡‘•±Ñ…}¹}É½Ñ}µ…á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰‘•±Ñ…}¹}•±•}Á•…­}èˆè}¹À¹…Í…ÉÉ…ä¡‘•±Ñ…}¹}•±•}Á•…­}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰‘•±Ñ…}¹}É½Ñ}Á•…­}èˆè}¹À¹…Í…ÉÉ…ä¡‘•±Ñ…}¹}É½Ñ}Á•…­}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰‘•±Ñ…}¹}•±•}…ÁÁ±¥•‘}µ…á}èˆè}¹À¹…Í…ÉÉ…ä¡‘•±Ñ…}¹}•±•}…ÁÁ±¥•‘}µ…á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰‘•±Ñ…}¹}É½Ñ}…ÁÁ±¥•‘}µ…á}èˆè}¹À¹…Í…ÉÉ…ä¡‘•±Ñ…}¹}É½Ñ}…ÁÁ±¥•‘}µ…á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰…±Á¡…}¥½¹}É…İ}µ…á}èˆè}¹À¹…Í…ÉÉ…ä¡…±Á¡…}¥½¹}É…İ}µ…á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰…±Á¡…}¥½¹}½ÉÉ}µ…á}èˆè}¹À¹…Í…ÉÉ…ä¡…±Á¡…}¥½¹}½ÉÉ}µ…á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰…±Á¡…}¥½¹}…ÁÁ±¥•‘}µ…á}èˆè}¹À¹…Í…ÉÉ…ä¡…±Á¡…}¥½¹}…ÁÁ±¥•‘}µ…á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰…±Á¡…}¥‰}µ…á}èˆè}¹À¹…Í…ÉÉ…ä¡…±Á¡…}¥‰}µ…á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰…±Á¡…}Ñ½Ñ…±}µ…á}èˆè}¹À¹…Í…ÉÉ…ä¡…±Á¡…}Ñ½Ñ…±}µ…á}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰‘•±Ñ…}¹}Á±…Íµ…}µ¥¹}èˆè}¹À¹…Í…ÉÉ…ä¡‘•±Ñ…}¹}Á±…Íµ…}µ¥¹}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰‘•±Ñ…}¹}Á±…Íµ…}…ÁÁ±¥•‘}µ¥¹}èˆè}¹À¹…Í…ÉÉ…ä¡‘•±Ñ…}¹}Á±…Íµ…}…ÁÁ±¥•‘}µ¥¹}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰‘Á¡¥}­•ÉÉ}µ…á}…‰Í}èˆè}¹À¹…Í…ÉÉ…ä¡‘Á¡¥}­•ÉÉ}µ…á}…‰Í}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰‘Á¡¥}•±•}µ…á}…‰Í}èˆè}¹À¹…Í…ÉÉ…ä¡‘Á¡¥}•±•}µ…á}…‰Í}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰‘Á¡¥}É½Ñ}µ…á}…‰Í}èˆè}¹À¹…Í…ÉÉ…ä¡‘Á¡¥}É½Ñ}µ…á}…‰Í}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰‘Á¡¥}•±•}…ÁÁ±¥•‘}µ…á}…‰Í}èˆè}¹À¹…Í…ÉÉ…ä¡‘Á¡¥}•±•}…ÁÁ±¥•‘}µ…á}…‰Í}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰‘Á¡¥}É½Ñ}…ÁÁ±¥•‘}µ…á}…‰Í}èˆè}¹À¹…Í…ÉÉ…ä¡‘Á¡¥}É½Ñ}…ÁÁ±¥•‘}µ…á}…‰Í}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰‘Á¡¥}Á±…Íµ…}É…İ}µ…á}…‰Í}èˆè}¹À¹…Í…ÉÉ…ä¡‘Á¡¥}Á±…Íµ…}É…İ}µ…á}…‰Í}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰‘Á¡¥}Á±…Íµ…}…ÁÁ±¥•‘}µ…á}…‰Í}èˆè}¹À¹…Í…ÉÉ…ä¡‘Á¡¥}Á±…Íµ…}…ÁÁ±¥•‘}µ…á}…‰Í}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰‘Á¡¥}Á±…Íµ…}µ…á}…‰Í}èˆè}¹À¹…Í…ÉÉ…ä¡‘Á¡¥}Á±…Íµ…}µ…á}…‰Í}é}±¥ÍĞ°‘ÑåÁ”õÉ‘ÑåÁ•}¹À¤°(€€€€€€€€‰¸É}•±•}ÕÍ•ˆè™±½…Ğ¡¸É}•±•Œ¤°(€€€€€€€€‰¹}I}ÕÍ•ˆè™±½…Ğ¡¹}H¤°(€€€€€€€€‰™}I}¥¹½É•‘}É½Ñ}Í¥¹•áÀˆè‰½½°¡™I}¥¹½É•‘}É½Ñ}Í¥¹•áÀ¤°(€€€€€€€€‰É…µ…¹}…‰Í½ÉÁÑ¥½¹}µ½‘•°ˆè…‰Í½ÉÁÑ¥½¹}µ½‘•°°€€Œƒ’úÿ’ê;–’[¦£¢¾ì(€€€€€€€€‰¥½¹¥é…Ñ¥½¹}½Á•É…Ñ½É}½ÉÉ•Ñ¥½¹}½¸ˆè‰½½°¡ÕÍ•}¥½¹}½Á}½ÉÈ¤°(€€€€€€€€‰¥½¹¥é…Ñ¥½¹}½Á•É…Ñ½É}µ•Ñ¡½ˆè¥½¹}½Á}µ•Ñ¡½°(€€€€€€€€‰¹½¹±¥¹•…É}ÕÍ•}•±•ÑÉ½¹¥}­•ÉÈˆè‰½½°¡Íİ¥Ñ¡•Ì¹ÕÍ•}•±•ÑÉ½¹¥}­•ÉÈ¤°(€€€€€€€€‰¹½¹±¥¹•…É}ÕÍ•}É…µ…¹}Á¡…Í”ˆè‰½½°¡Íİ¥Ñ¡•Ì¹ÕÍ•}É…µ…¹}Á¡…Í”¤°(€€€€€€€€‰¹½¹±¥¹•…É}ÕÍ•}É…µ…¹}™Õ±±}½Á•É…Ñ½Èˆè‰½½°¡Íİ¥Ñ¡•Ì¹ÕÍ•}É…µ…¹}™Õ±±}½Á•É…Ñ½È¤°(€€€€€€€€‰¹½¹±¥¹•…É}ÕÍ•}Á±…Íµ…}Á¡…Í”ˆè‰½½°¡Íİ¥Ñ¡•Ì¹ÕÍ•}Á±…Íµ…}Á¡…Í”¤°(€€€€€€€€‰¹½¹±¥¹•…É}ÕÍ•}¥½¹¥é…Ñ¥½¹}±½ÍÌˆè‰½½°¡Íİ¥Ñ¡•Ì¹ÕÍ•}¥½¹¥é…Ñ¥½¹}±½ÍÌ¤°(€€€€€€€€‰¹½¹±¥¹•…É}ÕÍ•}É…µ…¹}…‰Í½ÉÁÑ¥½¸ˆè‰½½°¡Íİ¥Ñ¡•Ì¹ÕÍ•}É…µ…¹}…‰Í½ÉÁÑ¥½¸¤°(€€€€€€€€‰¹½¹±¥¹•…É}ÕÍ•}Í•±™}ÍÑ••Á•¹¥¹œˆè‰½½°¡Íİ¥Ñ¡•Ì¹ÕÍ•}Í•±™}ÍÑ••Á•¹¥¹œ¤°(€€€€€€€€‰¹½¹±¥¹•…É}ÕÍ•}¥½¹¥é…Ñ¥½¹}Í½±Ù•Èˆè‰½½°¡Íİ¥Ñ¡•Ì¹ÕÍ•}¥½¹¥é…Ñ¥½¹}Í½±Ù•È¤°(€€€ô(€€€¥˜É•½É‘}½¹…á¥Í}É¡½}Ñ¥µ”…¹€¡É¡½}½¹…á¥Í}Ñ¥µ•}±¥ÍĞ¥Ì¹½Ğ9½¹”…¹±•¸¡É¡½}½¹…á¥Í}Ñ¥µ•}±¥ÍĞ¤€ø€À¤è(€€€€€€€‘¥…l‰É¡½}½¹…á¥Í}Ñ}è‰t€ô}¹À¹ÍÑ…¬¡É¡½}½¹…á¥Í}Ñ¥µ•}±¥ÍĞ°…á¥ÌôÀ¤¹…ÍÑåÁ”¡É‘ÑåÁ•}¹À°½Áäõ…±Í”¤((€€€Ù…±¥‘…Ñ¥½¸€ôÙ…±¥‘…Ñ•}¹½¹±¥¹•…É}‘¥…¹½ÍÑ¥Ì¡‘¥…œ¤(€€€‘¥…l‰‘¥…¹½ÍÑ¥}Ù…±¥‘…Ñ¥½¹}Á…ÍÍ•‰t€ô}¹À¹‰½½±|¡Ù…±¥‘…Ñ¥½¹l‰Á…ÍÍ•‰t¤(€€€‘¥…l‰‘¥…¹½ÍÑ¥}Ù…±¥‘…Ñ¥½¹}ÑÉ…•}½Õ¹Ğ‰t€ô}¹À¹¥¹ĞØĞ¡Ù…±¥‘…Ñ¥½¹l‰é}É•½É‘Ì‰t¤(€€€‘¥…l‰‘¥…¹½ÍÑ¥}…±±}é•É½}ÑÉ…•Ì‰t€ô}¹À¹…Í…ÉÉ…ä¡Ù…±¥‘…Ñ¥½¹l‰…±±}é•É½}ÑÉ…•Ì‰t°‘ÑåÁ”ô‰TØĞˆ¤(((€€€€ŒE…Œƒšb¼€É¾ò!(½µxË¾ò'¾òkR£’ê;š‹š^Û¦^Ó·š&§šVŒ(€€€É•ÑÕÉ¸°Ñ½}ÁÔ¡E…Œ¤¹…ÍÑåÁ”¡É‘ÑåÁ•}¹À°½Áäõ…±Í”¤°‘¥…œ(