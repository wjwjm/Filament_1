from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path
import numpy as _np

from .air_dispersion import n_of_omega
from .constants import N0_air, Ui_N2, c0, eps0, n0_air, n2_air
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
from .diagnostics import (
    intensity,
    peak_intensity,
    pulse_energy,
    save_npz,
    write_nonlinear_diagnostic_report,
)
from .grids import make_axes
from .heat import diffuse_dn_gas
from .longitudinal import build_deposition_contract, build_longitudinal_schedule
from .propagate import propagate_one_pulse
from .slow_state import (
    HR3BDiagnosticSink,
    ThermalSlowStateStore,
    validate_hr3b_parameters,
)
from .hr3c_state_machine import HR3CStateController
from .thermalization import ThermalDiagnosticSink, build_physical_sample_plan
from .summary import print_sim_summary
from .utils import gaussian_pulse_t, transverse_intensity_profile


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
            precision_strategy=getattr(prop, "linear_precision_strategy", "baseline_complex64"),
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


def _encircled_radius(radius_m, weights, fraction: float) -> float:
    """Return the discrete radius enclosing ``fraction`` of weighted power."""
    order = _np.argsort(radius_m.ravel())
    sorted_radius = radius_m.ravel()[order]
    cumulative = _np.cumsum(weights.ravel()[order])
    if cumulative.size == 0 or cumulative[-1] <= 0.0:
        return float("nan")
    index = min(int(_np.searchsorted(cumulative, float(fraction) * cumulative[-1], side="left")), cumulative.size - 1)
    return float(sorted_radius[index])


def build_transverse_input_field(axes, beam, ctype):
    """Build the pre-lens field and discrete input-profile diagnostics.

    Peak-power normalization is deliberately performed here, after the
    transverse profile has been sampled on the actual simulation grid.
    """
    import numpy as _np

    gI = transverse_intensity_profile(
        axes.x,
        axes.y,
        getattr(beam, "transverse_profile", None),
        fallback_w0=float(beam.w0),
    )
    pref = 0.5 * eps0 * c0 * float(beam.n0)
    area_eff = float(xp.sum(gI)) * axes.dx * axes.dy
    if not _np.isfinite(area_eff) or area_eff <= 0.0:
        raise ValueError("transverse profile has non-positive effective area on this grid")

    p0_target = getattr(beam, "P0_peak", None)
    energy_target = getattr(beam, "energy_J", None)
    if p0_target is not None:
        amplitude = float(float(p0_target) / (pref * area_eff)) ** 0.5
        norm_source = "P0_peak"
    elif energy_target is not None:
        amplitude = 1.0
        norm_source = "energy_J"
    else:
        amplitude = float(getattr(beam, "E0_peak", 0.0) or 0.0)
        if amplitude <= 0.0:
            raise ValueError("Beam E0_peak is 0 and no energy_J/P0_peak provided; cannot build input field.")
        norm_source = "E0_peak_direct"

    E_xy = amplitude * xp.sqrt(gI)
    E_t = gaussian_pulse_t(axes.t, beam.tau_fwhm)
    E = (E_t * E_xy[None, ...]).astype(ctype)

    if energy_target is not None:
        I_in = intensity(E, beam.n0)
        U_now = float(pulse_energy(I_in, axes.dt, axes.dx, axes.dy))
        if not _np.isfinite(U_now) or U_now <= 0.0:
            raise ValueError("cannot normalize a transverse input with zero or non-finite energy")
        scale = (float(energy_target) / U_now) ** 0.5
        E *= scale
        E_xy *= scale
        amplitude *= scale

    profile = getattr(beam, "transverse_profile", None) or {}
    profile_type = str(profile.get("type", "gaussian"))
    profile_radius = float(profile.get("radius_m", beam.w0))
    edge_start = profile.get("edge_start_fraction", None)
    iy_center = int(xp.argmin(xp.abs(axes.y)))
    gI_cpu = _np.asarray(to_cpu(gI), dtype=float)
    x_cpu = _np.asarray(to_cpu(axes.x), dtype=float)
    y_cpu = _np.asarray(to_cpu(axes.y), dtype=float)
    X_cpu, Y_cpu = _np.meshgrid(x_cpu, y_cpu, indexing="xy")
    r_cpu = _np.sqrt(X_cpu ** 2 + Y_cpu ** 2)
    g_sum = float(gI_cpu.sum())
    r2_mean = float((r_cpu ** 2 * gI_cpu).sum() / g_sum) if g_sum > 0.0 else float("nan")
    boundary = _np.concatenate((gI_cpu[0, :], gI_cpu[-1, :], gI_cpu[1:-1, 0], gI_cpu[1:-1, -1]))
    boundary_fraction = float(_np.max(boundary) / max(float(_np.max(gI_cpu)), 1e-30))
    actual_p0 = pref * float(xp.sum(xp.abs(E_xy) ** 2)) * axes.dx * axes.dy

    diagnostics = {
        "input_profile_x": to_cpu(axes.x),
        "input_profile_center_I": to_cpu(pref * xp.abs(E_xy[iy_center, :]) ** 2),
        "input_peak_power_W": actual_p0,
        "input_peak_intensity_W_m2": pref * amplitude ** 2,
        "input_effective_area_m2": area_eff,
        "input_second_moment_radius_m": (2.0 * r2_mean) ** 0.5,
        "input_r50_m": _encircled_radius(r_cpu, gI_cpu, 0.50),
        "input_r90_m": _encircled_radius(r_cpu, gI_cpu, 0.90),
        "input_boundary_I_fraction": boundary_fraction,
        "input_profile_type": _np.asarray(profile_type),
        "input_profile_radius_m": profile_radius,
        "input_profile_edge_start_fraction": float(edge_start) if edge_start is not None else _np.nan,
        "input_field_amplitude_V_m": amplitude,
        "input_normalization_source": _np.asarray(norm_source),
    }
    return E, diagnostics


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
    return_results: bool = False,
):
    dmap = {"fp32": xp.complex64, "fp64": xp.complex128}
    ctype = dmap.get(str(dtype).lower(), xp.complex64)
    rmap = {"fp32": xp.float32, "fp64": xp.float64}
    rtype = rmap.get(str(dtype).lower(), xp.float32)

    import math

    omega0 = 2 * math.pi * c0 / beam.lam0
    k0 = beam.n0 * omega0 / c0
    axes = make_axes(grid.Nx, grid.Ny, grid.Nt, grid.Lx, grid.Ly, grid.Twin)

    E, input_profile = build_transverse_input_field(axes, beam, ctype)

    if getattr(beam, "focal_length", None):
        if str(getattr(prop, "linear_model", "uppe")).lower() == "uppe":
            E = apply_thin_lens_achromatic(E, axes, beam, prop, chunk_t=getattr(prop, "lens_chunk_t", 0))
        else:
            X, Y = xp.meshgrid(axes.x, axes.y, indexing="xy")
            phase_lens = -(k0 / (2.0 * float(beam.focal_length))) * (X ** 2 + Y ** 2)
            onej = xp.array(1j, dtype=E.dtype)
            E *= xp.exp(onej * xp.asarray(phase_lens, dtype=(xp.float32 if E.dtype == xp.complex64 else xp.float64)))

    n2_used = float(getattr(prop, "n2", getattr(beam, "n2_air", n2_air)))
    print_sim_summary(grid=grid, beam=beam, prop=prop, ion=ion, heat=heat, run=run,
                      axes=axes, E=E, n2_used=n2_used, raman=raman, dtype_str=dtype,
                      input_profile=input_profile)

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
    schedule_z_end = float(prop_for_pulse.z_max)
    if limit_win and win_half > 0.0:
        if z_focus_hint is None:
            z_focus_hint = _predict_focus_linear(E, axes)
        z_start = max(0.0, float(z_focus_hint) - win_half)
        z_end = float(z_focus_hint) + win_half
        schedule_z_end = z_end
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

    # Build one absolute, deterministic schedule and one metadata-only
    # deposition contract after the propagation window is finalized.  Every
    # pulse receives these exact objects; no pulse-dependent z grid is built.
    longitudinal_schedule = build_longitudinal_schedule(
        dz=float(prop_for_pulse.dz),
        z_max=float(schedule_z_end),
        z_start=float(z_start),
        focus_window_step=bool(getattr(prop_for_pulse, "focus_window_step", False)),
        focus_center_m=(None if z_focus_hint is None else float(z_focus_hint)),
        focus_halfwidth_m=float(getattr(prop_for_pulse, "focus_halfwidth_m", 0.0)),
        dz_focus=float(getattr(prop_for_pulse, "dz_focus", prop_for_pulse.dz)),
    )
    deposition_contract = build_deposition_contract(
        longitudinal_schedule, axes=axes
    )
    focus_enabled = bool(getattr(prop_for_pulse, "focus_window_step", False))
    focal_plane = getattr(beam, "focal_length", None)
    thermal_sample_plan = build_physical_sample_plan(
        longitudinal_schedule,
        focus_center_m=(None if z_focus_hint is None else float(z_focus_hint)),
        focus_halfwidth_m=float(getattr(prop_for_pulse, "focus_halfwidth_m", 0.0)),
        focus_enabled=focus_enabled,
        focal_plane_m=(None if focal_plane is None else float(focal_plane)),
    )
    thermal_sink = ThermalDiagnosticSink(
        plan=thermal_sample_plan,
        output_path=out_path,
        shape=(grid.Ny, grid.Nx),
        dtype=(_np.float32 if str(dtype).lower() == "fp32" else _np.float64),
        enabled=(int(run.Npulses) == 1),
    )
    hr3b_enabled = bool(getattr(heat, "hr3b_enabled", False))
    hr3c_enabled = bool(getattr(heat, "hr3c_enabled", False))
    if hr3c_enabled and not hr3b_enabled:
        raise ValueError("HR-3C requires heat.hr3b_enabled=true")
    hr3b_parameters = None
    thermal_slow_state = None
    hr3b_sink = None
    hr3c_controller = None
    if hr3b_enabled:
        beta_th = validate_hr3b_parameters(
            rho0=float(heat.rho0),
            Cv=float(heat.Cv),
            T0=float(prop_for_pulse.air_T),
            n0=float(beam.n0),
        )
        hr3b_parameters = {
            "rho0": float(heat.rho0),
            "Cv": float(heat.Cv),
            "T0": float(prop_for_pulse.air_T),
            "n0": float(beam.n0),
            "beta_th": float(beta_th),
        }
        if hr3c_enabled:
            hr3c_controller = HR3CStateController(
                output_path=out_path, n_intervals=longitudinal_schedule.n_intervals,
                shape=(grid.Ny, grid.Nx), dtype=(_np.float32 if str(dtype).lower() == "fp32" else _np.float64),
                z_edges=longitudinal_schedule.z_edges, dx=axes.dx, dy=axes.dy,
                D_th=float(heat.D_th), f_rep=float(heat.f_rep), edge_threshold=1.0e-3,
                batch_intervals=int(heat.hr3c_batch_intervals), npulses=int(run.Npulses),
                resume=bool(heat.resume_hr3c),
            )
            hr3c_controller.attach_grid(axes.kperp2)
        else:
            thermal_slow_state = ThermalSlowStateStore(
                output_path=out_path, n_intervals=longitudinal_schedule.n_intervals,
                shape=(grid.Ny, grid.Nx), dtype=(_np.float32 if str(dtype).lower() == "fp32" else _np.float64),
            )
            hr3b_sink = HR3BDiagnosticSink(
                plan=thermal_sample_plan, output_path=out_path, shape=(grid.Ny, grid.Nx),
                dtype=(_np.float32 if str(dtype).lower() == "fp32" else _np.float64),
                enabled=(int(run.Npulses) == 1),
            )

    # The pulse-independent source is defined at the exact input plane of
    # propagate_one_pulse: after the lens and optional linear pre-advance.
    # It is never passed to propagation directly, so every pulse receives an
    # independent working copy while only the slow medium persists.
    E_source = E

    dn_gas = xp.zeros((grid.Ny, grid.Nx), dtype=rtype) if not hr3b_enabled else None
    delta_t_pulse = 1.0 / heat.f_rep

    t_all = time.perf_counter()
    last_diag = None
    pulse_index_list = []
    pulse_dn_gas_min_list = []
    pulse_dn_gas_max_list = []
    pulse_delta_n_th_min_list = []
    hr3c_diffusion_passes = 0
    hr3c_post_commits = 0
    while hr3c_controller is not None and not bool(hr3c_controller.manifest["run_complete"]):
        if hr3c_controller.manifest["physical_stage"] == "post_pulse":
            hr3c_controller.diffuse_to_next_pre()
            hr3c_diffusion_passes += 1
            continue
        i = int(hr3c_controller.manifest["next_pulse_index"])
        t_p = time.perf_counter()
        E_pulse = E_source.copy()
        transaction = hr3c_controller.begin_pulse()
        try:
            E, Q2D, diag = propagate_one_pulse(
            E_pulse,
            kperp2=axes.kperp2,
            k0=k0, omega0=omega0,
            dz=prop_for_pulse.dz, z_max=prop_for_pulse.z_max,
            n0=beam.n0, n2=n2_used,
            Ui=Ui_N2, N0=N0_air,
            ion_conf=ion, dn_gas=dn_gas,
            dt=axes.dt, axes=axes, prop_conf=prop_for_pulse, raman_conf=raman,
            record_onaxis_rho_time=True,
            record_every_z=1,
            longitudinal_schedule=longitudinal_schedule,
            deposition_contract=deposition_contract,
            thermal_sink=thermal_sink,
            thermal_slow_state=transaction,
            hr3b_parameters=hr3b_parameters,
            hr3b_sink=None,
            )
            hr3c_controller.commit_post_pulse(transaction, i)
        except Exception:
            transaction.store.mark_next_invalid()
            raise
        hr3c_post_commits += 1
        mn = float(_np.min(diag["delta_n_state_min_after_update"]))
        pulse_delta_n_th_min_list.append(mn)
        print(f"Pulse {i + 1}/{run.Npulses}: HR-3C post Δn_th min = {mn:.3e}  (elapsed {time.perf_counter() - t_p:.1f}s)")
        pulse_index_list.append(i + 1)
        last_diag = diag
        if i == int(run.Npulses) - 1:
            break
        hr3c_controller.diffuse_to_next_pre()
        hr3c_diffusion_passes += 1

    if hr3c_controller is None:
      for i in range(run.Npulses):
        t_p = time.perf_counter()
        E_pulse = E_source.copy()
        E, Q2D, diag = propagate_one_pulse(
            E_pulse,
            kperp2=axes.kperp2, k0=k0, omega0=omega0,
            dz=prop_for_pulse.dz, z_max=prop_for_pulse.z_max,
            n0=beam.n0, n2=n2_used, Ui=Ui_N2, N0=N0_air,
            ion_conf=ion, dn_gas=dn_gas, dt=axes.dt, axes=axes, prop_conf=prop_for_pulse, raman_conf=raman,
            record_onaxis_rho_time=True, record_every_z=1,
            longitudinal_schedule=longitudinal_schedule, deposition_contract=deposition_contract,
            thermal_sink=thermal_sink, thermal_slow_state=thermal_slow_state,
            hr3b_parameters=hr3b_parameters, hr3b_sink=hr3b_sink,
        )
        if hr3b_enabled:
            mn = float(_np.min(diag["delta_n_state_min_after_update"]))
            pulse_delta_n_th_min_list.append(mn)
            print(f"Pulse {i + 1}/{run.Npulses}: HR-3B Δn_th min = {mn:.3e}  (elapsed {time.perf_counter() - t_p:.1f}s)")
        else:
            dn_gas = diffuse_dn_gas(dn_gas, Q2D, heat.D_gas, delta_t_pulse, axes.kperp2, heat.gamma_heat)
            mn, mx = float(xp.min(dn_gas)), float(xp.max(dn_gas))
            pulse_dn_gas_min_list.append(mn)
            pulse_dn_gas_max_list.append(mx)
            print(f"Pulse {i + 1}/{run.Npulses}: Δn_gas min/max = {mn:.3e}/{mx:.3e}  (elapsed {time.perf_counter() - t_p:.1f}s)")
        pulse_index_list.append(i + 1)
        last_diag = diag

    # Idempotently close a sink even when a test double bypassed propagation.
    thermal_sink.finalize()
    if hr3b_sink is not None:
        hr3b_sink.finalize()
    if thermal_slow_state is not None:
        thermal_slow_state.finalize()
    if hr3c_controller is not None:
        hr3c_controller.close()

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
        "x": to_cpu(axes.x), "y": to_cpu(axes.y), "t": to_cpu(axes.t), "t_axis": to_cpu(axes.t),
        "I_out_center_t": to_cpu(I_out[:, grid.Ny // 2, grid.Nx // 2]),
        "pulse_index": _np.asarray(pulse_index_list, dtype=_np.int64),
    }
    if hr3b_enabled:
        out["pulse_delta_n_th_min"] = _np.asarray(pulse_delta_n_th_min_list, dtype=float)
    if hr3c_controller is not None:
        manifest = hr3c_controller.manifest
        out.update({
            "hr3c_enabled": True,
            "hr3c_manifest_filename": _np.asarray(hr3c_controller.manifest_path.name),
            "hr3c_final_authoritative_filename": _np.asarray(manifest["authoritative_filename"]),
            "hr3c_final_physical_stage": _np.asarray(manifest["physical_stage"]),
            "hr3c_final_pulse_index": _np.int64(manifest["pulse_index"]),
            "hr3c_run_complete": bool(manifest["run_complete"]),
            "n_fresh_pulses_completed": _np.int64(manifest["n_fresh_pulses_completed_total"]),
            "n_hr3b_post_commits": _np.int64(manifest["n_hr3b_post_commits_total"]),
            "n_hr3c_diffusion_passes": _np.int64(manifest["n_hr3c_diffusion_passes_total"]),
            "this_invocation_fresh_pulses": _np.int64(len(pulse_index_list)),
            "this_invocation_post_commits": _np.int64(hr3c_post_commits),
            "this_invocation_diffusions": _np.int64(hr3c_diffusion_passes),
            "hr3c_D_th_m2_s": float(heat.D_th),
            "hr3c_dt_interpulse_s": 1.0 / float(heat.f_rep),
            "hr3c_edge_threshold": 1.0e-3,
            "hr3c_batch_intervals": _np.int64(heat.hr3c_batch_intervals),
        })
    else:
        out.update({
            "dn_gas": to_cpu(dn_gas),
            "pulse_dn_gas_min": _np.asarray(pulse_dn_gas_min_list, dtype=float),
            "pulse_dn_gas_max": _np.asarray(pulse_dn_gas_max_list, dtype=float),
        })
    out.update(input_profile)
    if last_diag:
        # Propagation owns the diagnostic schema.  Copy every returned field so
        # newly added observability traces cannot be calculated but omitted
        # from the NPZ by a stale allow-list here.
        for k, value in last_diag.items():
            if value is not None:
                out[k] = to_cpu(value)

    save_npz(out_path, **{k: v for k, v in out.items() if v is not None})
    print(f"Saved diagnostics to {out_path}")
    if last_diag:
        report_path = Path(out_path).with_suffix(".diagnostic_report.json")
        report = write_nonlinear_diagnostic_report(report_path, last_diag, npz_path=out_path)
        print(
            f"[diagnostic-check] PASS: {report['validation']['z_records']} z records; "
            f"report={report_path}"
        )
    if return_results:
        return {
            "E_final": to_cpu(E),
            "I_final": to_cpu(I_out),
            "diagnostics": {key: to_cpu(value) for key, value in (last_diag or {}).items()},
            "pulse_summary": {
                "pulse_index": _np.asarray(pulse_index_list, dtype=_np.int64),
                **(
                    {"delta_n_th_min": _np.asarray(pulse_delta_n_th_min_list, dtype=float)}
                    if hr3b_enabled else {
                        "dn_gas_min": _np.asarray(pulse_dn_gas_min_list, dtype=float),
                        "dn_gas_max": _np.asarray(pulse_dn_gas_max_list, dtype=float),
                    }
                ),
            },
            "axes": {"x": to_cpu(axes.x), "y": to_cpu(axes.y), "t": to_cpu(axes.t)},
        }


def run_from_file(cfg_path: str, out_path: str = "khzfil_out.npz", dtype: str = "fp32"):
    from .confio import load_all

    grid, beam, prop, ion, heat, run, raman = load_all(cfg_path)
    print(f"[config] Loaded: {os.path.abspath(cfg_path)}")
    return run_demo(grid=grid, beam=beam, prop=prop, ion=ion, heat=heat, run=run, raman=raman, out_path=out_path, dtype=dtype)
