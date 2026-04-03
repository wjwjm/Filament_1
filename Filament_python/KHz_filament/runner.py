from __future__ import annotations

import dataclasses
import os
import time

from .air_dispersion import n_of_omega
from .constants import N0_air, Ui_N2, c0, n0_air, n2_air
from .config import (
    BeamConfig,
    GridConfig,
    HeatConfig,
    IonizationConfig,
    PropagationConfig,
    RamanConfig,
    RunConfig,
)
from .device import to_cpu, xp
from .diagnostics import intensity, peak_intensity, pulse_energy, save_npz
from .grids import make_axes
from .heat import diffuse_dn_gas
from .propagate import propagate_one_pulse
from .summary import print_sim_summary
from .utils import gaussian_beam_xy, gaussian_pulse_t


def _linear_advance(E, dz, *, axes, kperp2, k0, prop, beam):
    """仅线性推进 E -> z+dz（一次性），用于把起始面“跳”到 z_start。"""
    if abs(float(dz)) < 1e-16:
        return E

    from .linear import lin_propagator, step_linear, step_linear_bk_nee_factorized
    from .linear_full import step_linear_full_3d, step_linear_full_factorized

    linear_model = str(getattr(prop, "linear_model", "uppe")).lower()
    if linear_model == "uppe":
        omega0 = 2.0 * xp.pi * c0 / float(getattr(beam, "lam0"))
        omega_tot = omega0 + axes.Omega
        omega_safe = xp.where(xp.abs(omega_tot) < 1e-9 * omega0, xp.sign(omega_tot) * 1e-9 * omega0, omega_tot)
        n_w = n_of_omega(omega_safe, P=getattr(prop, "air_P", 101325.0), T=getattr(prop, "air_T", 293.15))
        K02_w = (n_w * omega_safe / c0) ** 2
        if bool(getattr(prop, "full_linear_factorize", False)):
            return step_linear_full_factorized(E, K02_w, kperp2, dz)
        return step_linear_full_3d(E, K02_w, kperp2, dz)

    if linear_model == "bk_nee":
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

    prop_x = lin_propagator(kperp2, k0, dz, ctype=E.dtype)
    return step_linear(E, prop_x)


def apply_thin_lens_achromatic(E, axes, beam, prop, chunk_t: int = 0):
    """频率域薄透镜：对每个频率分量乘 exp(-i k(|ω|) r^2 / (2f))。"""
    Omega = axes.Omega
    omega0 = 2.0 * xp.pi * c0 / beam.lam0
    omega = omega0 + Omega
    omega_a = xp.abs(omega)

    n_w = n_of_omega(omega_a, P=getattr(prop, "air_P", 101325.0), T=getattr(prop, "air_T", 293.15))
    k_w = n_w * omega_a / c0

    X, Y = xp.meshgrid(axes.x, axes.y, indexing="xy")
    r2 = (X ** 2 + Y ** 2)[xp.newaxis, :, :]
    f = float(beam.focal_length)
    onej = xp.array(1j, dtype=E.dtype)

    Nt = E.shape[0]
    if not chunk_t or chunk_t <= 0:
        Ew = xp.fft.fft(E, axis=0)
        phase_w = -(k_w[:, None, None] / (2.0 * f)) * r2
        Ew *= xp.exp(onej * phase_w).astype(Ew.dtype, copy=False)
        return xp.fft.ifft(Ew, axis=0)

    out = xp.empty_like(E)
    for i0 in range(0, Nt, chunk_t):
        i1 = min(Nt, i0 + chunk_t)
        Ew = xp.fft.fft(E[i0:i1, ...], axis=0)
        phase_w = -(k_w[i0:i1, None, None] / (2.0 * f)) * r2
        Ew *= xp.exp(onej * phase_w).astype(Ew.dtype, copy=False)
        out[i0:i1, ...] = xp.fft.ifft(Ew, axis=0)
    return out


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

    import math
    import numpy as _np

    omega0 = 2 * math.pi * c0 / beam.lam0
    k0 = beam.n0 * omega0 / c0
    axes = make_axes(grid.Nx, grid.Ny, grid.Nt, grid.Lx, grid.Ly, grid.Twin)

    if getattr(beam, "E0_peak", 0.0) == 0.0:
        raise ValueError("Beam E0_peak is 0 and no energy_J/P0_peak provided; cannot build input field.")

    E_xy = gaussian_beam_xy(axes.x, axes.y, beam.w0)[None, ...]
    E_t = gaussian_pulse_t(axes.t, beam.tau_fwhm)
    E = (beam.E0_peak * E_t * E_xy).astype(ctype)

    if getattr(beam, "focal_length", None):
        if str(getattr(prop, "linear_model", "uppe")).lower() == "uppe":
            E = apply_thin_lens_achromatic(E, axes, beam, prop, chunk_t=getattr(prop, "lens_chunk_t", 0))
        else:
            X, Y = xp.meshgrid(axes.x, axes.y, indexing="xy")
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

    def measure_input_waist(E0, axes):
        it0 = E0.shape[0] // 2
        I0 = xp.abs(E0[it0]) ** 2
        Ix = xp.sum(I0, axis=0) * axes.dy
        Iy = xp.sum(I0, axis=1) * axes.dx
        Sx = float(xp.sum(Ix) * axes.dx) + 1e-30
        Sy = float(xp.sum(Iy) * axes.dy) + 1e-30
        x2 = float(xp.sum((axes.x ** 2) * Ix) * axes.dx) / Sx
        y2 = float(xp.sum((axes.y ** 2) * Iy) * axes.dy) / Sy
        return 0.5 * ((4.0 * x2) ** 0.5 + (4.0 * y2) ** 0.5)

    w_meas = measure_input_waist(E, axes)
    lam_med = beam.lam0 / beam.n0
    zR = math.pi * (w_meas ** 2) / lam_med
    fL0 = getattr(beam, "focal_length", float("inf"))
    f = float("inf") if fL0 is None else float(fL0)
    z_pred = (f / (1.0 + (f / zR) ** 2)) if _np.isfinite(f) else float("inf")
    print(f"[waist] measured w_in={w_meas:.3e} m  -> z_R={zR:.3e} m  -> z_focus_pred≈{z_pred:.4f} m (thin-lens)")

    limit_win = bool(getattr(prop, "limit_focus_window", False))
    win_half = float(getattr(prop, "window_halfwidth_m", 0.0))
    z_focus_hint = getattr(prop, "focus_center_m", getattr(prop, "z_focus_hint", None))

    def _predict_focus_linear(E0, axes):
        it0 = E0.shape[0] // 2
        I0 = xp.abs(E0[it0]) ** 2
        Ix = xp.sum(I0, axis=0) * axes.dy
        Iy = xp.sum(I0, axis=1) * axes.dx
        Sx = float(xp.sum(Ix) * axes.dx) + 1e-30
        Sy = float(xp.sum(Iy) * axes.dy) + 1e-30
        x2 = float(xp.sum((axes.x ** 2) * Ix) * axes.dx) / Sx
        y2 = float(xp.sum((axes.y ** 2) * Iy) * axes.dy) / Sy
        w_meas = 0.5 * ((4.0 * x2) ** 0.5 + (4.0 * y2) ** 0.5)
        lam_med = beam.lam0 / beam.n0
        zR = xp.pi * (w_meas ** 2) / lam_med
        fL0 = getattr(beam, "focal_length", _np.inf)
        fL = float("inf") if fL0 is None else float(fL0)
        return (fL / (1.0 + (fL / zR) ** 2)) if _np.isfinite(fL) else 0.0

    z_start = 0.0
    prop_for_pulse = prop
    if limit_win and win_half > 0.0:
        if z_focus_hint is None:
            z_focus_hint = _predict_focus_linear(E, axes)
        z_start = max(0.0, float(z_focus_hint) - win_half)
        z_end = float(z_focus_hint) + win_half
        focus_center_local_m = float(z_focus_hint) - z_start
        if z_start > 0.0:
            E = _linear_advance(E, z_start, axes=axes, kperp2=axes.kperp2, k0=k0, prop=prop, beam=beam)
            print(f"[window] Linear pre-advance: z_start={z_start:.4f} m  (center={float(z_focus_hint):.4f} m, half={win_half:.4f} m)")
        window_has_focus = (0.0 <= focus_center_local_m <= max(1e-9, z_end - z_start))
        print(
            f"[window] absolute focus={float(z_focus_hint):.4f} m, z_start={z_start:.4f} m, "
            f"z_end={z_end:.4f} m, local focus={focus_center_local_m:.4f} m, "
            f"dz_focus_window_active={window_has_focus}"
        )
        print(
            f"[window] propagate_one_pulse z-axis is local (starts at 0). "
            f"Absolute z requires z_local + z_start ({z_start:.4f} m)."
        )
        prop_for_pulse = dataclasses.replace(
            prop,
            z_max=max(1e-9, z_end - z_start),
            focus_center_m=focus_center_local_m,
        )

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
            dz=prop_for_pulse.dz, z_max=prop_for_pulse.z_max,
            n0=beam.n0, n2=n2_used,
            Ui=Ui_N2, N0=N0_air,
            ion_conf=ion, dn_gas=dn_gas,
            dt=axes.dt, axes=axes, prop_conf=prop_for_pulse, raman_conf=raman,
            record_onaxis_rho_time=True,
            record_every_z=1,
        )
        dn_gas = diffuse_dn_gas(dn_gas, Q2D, heat.D_gas, delta_t_pulse, axes.kperp2, heat.gamma_heat)
        mn, mx = float(xp.min(dn_gas)), float(xp.max(dn_gas))
        print(f"Pulse {i + 1}/{run.Npulses}: Δn_gas min/max = {mn:.3e}/{mx:.3e}  (elapsed {time.perf_counter() - t_p:.1f}s)")
        last_diag = diag

    print(f"[total] {time.perf_counter() - t_all:5.1f}s")

    I_out = intensity(E, beam.n0)
    Ipk = peak_intensity(I_out)
    energy = pulse_energy(I_out, axes.dt, axes.dx, axes.dy)
    print(f"Peak intensity out: {Ipk:.3e} W/m^2; Pulse energy ~ {energy:.3e} J")

    try:
        if last_diag and "w_mom_z" in last_diag:
            z_axis_cpu = to_cpu(last_diag["z_axis"])
            wcpu = to_cpu(last_diag["w_mom_z"])
            if len(z_axis_cpu) == len(wcpu) and len(wcpu) > 0:
                iz_min = int(_np.argmin(wcpu))
                print(f"[focus] estimated z_of_focus ≈ {z_axis_cpu[iz_min]:.4f} m (min second-moment radius)")
    except Exception:
        pass

    out = {
        "x": to_cpu(axes.x), "y": to_cpu(axes.y), "t": to_cpu(axes.t),
        "I_out_center_t": to_cpu(I_out[:, grid.Ny // 2, grid.Nx // 2]),
        "dn_gas": to_cpu(dn_gas),
    }
    diag_keys = [
        "z_axis", "U_z", "I_max_z", "I_onaxis_max_z", "I_center_t0_z",
        "w_mom_z", "rho_onaxis_max_z", "rho_max_z",
        "E_dep_z", "E_dep_rot_z", "fwhm_plasma_z", "fwhm_fluence_z", "rho_onaxis_t_z", "I_onaxis_max_interp_list",
    ]
    if last_diag:
        for k in diag_keys:
            if k in last_diag and last_diag[k] is not None:
                out[k] = to_cpu(last_diag[k])

    save_npz(out_path, **{k: v for k, v in out.items() if v is not None})
    print(f"Saved diagnostics to {out_path}")


def run_from_file(cfg_path: str, out_path: str = "khzfil_out.npz", dtype: str = "fp32"):
    from .confio import load_all

    grid, beam, prop, ion, heat, run, raman = load_all(cfg_path)
    print(f"[config] Loaded: {os.path.abspath(cfg_path)}")
    return run_demo(grid=grid, beam=beam, prop=prop, ion=ion, heat=heat, run=run, raman=raman, out_path=out_path, dtype=dtype)
