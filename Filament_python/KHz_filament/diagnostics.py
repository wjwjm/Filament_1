from __future__ import annotations
from .device import xp, to_cpu
from .constants import eps0, c0
import json
import math
from pathlib import Path

import numpy as _np


# Every entry below is a one-dimensional history indexed by ``z_axis``.  Keep
# this registry close to the validation code so additions cannot silently miss
# the output/checking path.
NONLINEAR_DIAGNOSTIC_METADATA = {
    "delta_n_elec_max_z": {
        "meaning": "maximum instantaneous electronic-Kerr refractive-index increment",
        "source": "propagate.py: delta_n_elec = n2_elec * I",
        "unit": "1",
        "use": "electronic Kerr strength and self-focusing onset",
    },
    "delta_n_rot_max_z": {
        "meaning": "maximum rotational-Raman refractive-index increment",
        "source": "propagate.py: delta_n_rot = n_R * (Raman convolution of I)",
        "unit": "1",
        "use": "delayed Kerr contribution and Raman ablation studies",
    },
    "IR_max_z": {
        "meaning": "maximum Raman-convolved intensity response",
        "source": "propagate.py: raman_convolve_intensity(I, ...)",
        "unit": "W/m^2",
        "use": "verify Raman convolution magnitude along propagation",
    },
    "IR_abs_max_z": {
        "meaning": "maximum absolute Raman-convolved intensity response",
        "source": "propagate.py: abs(raman_convolve_intensity(I, ...))",
        "unit": "W/m^2",
        "use": "detect signed Raman response even when its positive maximum is small",
    },
    "fwhm_time_z": {
        "meaning": "on-axis temporal intensity full width at half maximum",
        "source": "propagate.py: _fwhm_time_1d(I_now[:, y0, x0], dt)",
        "unit": "s",
        "use": "pulse-duration evolution along propagation",
    },
    "delta_n_plasma_min_z": {
        "meaning": "most negative plasma refractive-index increment",
        "source": "propagate.py: plasma_phase(rho, ...) divided by k0*dz",
        "unit": "1",
        "use": "plasma defocusing strength",
    },
    "dphi_kerr_max_abs_z": {
        "meaning": "maximum absolute applied total Kerr phase per nonlinear step",
        "source": "propagate.py: dphi_k after optional self-steepening correction",
        "unit": "rad",
        "use": "nonlinear phase budget and step-size diagnosis",
    },
    "dphi_elec_max_abs_z": {
        "meaning": "maximum absolute electronic-Kerr phase estimate before optional self-steepening correction",
        "source": "propagate.py: k0 * delta_n_elec * dz",
        "unit": "rad",
        "use": "separate electronic Kerr from rotational Raman in a phase budget",
    },
    "dphi_rot_max_abs_z": {
        "meaning": "maximum absolute rotational-Raman phase estimate before optional self-steepening correction",
        "source": "propagate.py: k0 * delta_n_rot * dz",
        "unit": "rad",
        "use": "separate delayed Raman Kerr from electronic Kerr in a phase budget",
    },
    "dphi_plasma_max_abs_z": {
        "meaning": "maximum absolute applied plasma phase per nonlinear step",
        "source": "propagate.py: plasma_phase(rho, ...)",
        "unit": "rad",
        "use": "plasma defocusing phase budget",
    },
    "delta_n_elec_applied_max_z": {
        "meaning": "maximum electronic-Kerr refractive-index increment admitted to the propagation phase",
        "source": "propagate.py: delta_n_elec gated by use_electronic_kerr",
        "unit": "1",
        "use": "verify electronic-Kerr switch isolation",
    },
    "delta_n_rot_applied_max_z": {
        "meaning": "maximum rotational-Raman refractive-index increment admitted to the propagation phase",
        "source": "propagate.py: delta_n_rot gated by use_raman_phase",
        "unit": "1",
        "use": "verify Raman-phase switch isolation",
    },
    "delta_n_plasma_applied_min_z": {
        "meaning": "most negative plasma refractive-index increment admitted to the propagation phase",
        "source": "propagate.py: plasma phase gated by use_plasma_phase",
        "unit": "1",
        "use": "verify plasma-phase switch isolation",
    },
    "dphi_elec_applied_max_abs_z": {
        "meaning": "maximum electronic-Kerr phase estimate admitted before optional self-steepening",
        "source": "propagate.py: applied delta_n_elec",
        "unit": "rad",
        "use": "separate raw and applied electronic Kerr",
    },
    "dphi_rot_applied_max_abs_z": {
        "meaning": "maximum rotational-Raman phase estimate admitted before optional self-steepening",
        "source": "propagate.py: applied delta_n_rot",
        "unit": "rad",
        "use": "separate raw and applied rotational Raman",
    },
    "dphi_plasma_raw_max_abs_z": {
        "meaning": "maximum plasma phase calculated from rho before switch gating",
        "source": "propagate.py: plasma_phase(rho, ...)",
        "unit": "rad",
        "use": "verify rho-derived plasma diagnostics when feedback is OFF",
    },
    "dphi_plasma_applied_max_abs_z": {
        "meaning": "maximum plasma phase admitted to propagation",
        "source": "propagate.py: raw plasma phase gated by use_plasma_phase",
        "unit": "rad",
        "use": "verify plasma-phase switch isolation",
    },
    "alpha_ion_corr_max_z": {
        "meaning": "maximum ionization-loss coefficient used in propagation",
        "source": "propagate.py: alpha_ion after optional operator correction",
        "unit": "m^-1",
        "use": "ionization absorption and operator-correction comparisons",
    },
    "alpha_ion_raw_max_z": {
        "meaning": "maximum ionization-loss coefficient before optional operator correction",
        "source": "propagate.py: ion_source_raw / (I + I_floor)",
        "unit": "m^-1",
        "use": "quantify the effect of the ionization operator correction",
    },
    "alpha_ib_max_z": {
        "meaning": "maximum inverse-Bremsstrahlung absorption coefficient",
        "source": "propagate.py: ib_alpha(rho, sigma_ib)",
        "unit": "m^-1",
        "use": "separate plasma collisional loss from ionization loss",
    },
    "alpha_R_eff_z": {
        "meaning": "effective Raman absorption coefficient used in propagation",
        "source": "propagate.py: Raman absorption model",
        "unit": "m^-1",
        "use": "Raman-absorption ablation and loss budget",
    },
    "alpha_R_closed_z": {
        "meaning": "effective Raman coefficient from the closed-form absorption branch; zero for other Raman models",
        "source": "propagate.py: closed_form / alpha_local Raman absorption branch",
        "unit": "m^-1",
        "use": "distinguish the closed-form Raman-loss branch from convolution-derivative loss",
    },
    "alpha_total_max_z": {
        "meaning": "maximum total absorption coefficient applied to the field",
        "source": "propagate.py: alpha_ib + alpha_ion + alpha_R_eff",
        "unit": "m^-1",
        "use": "total nonlinear attenuation budget and step-size diagnosis",
    },
    "alpha_ion_applied_max_z": {
        "meaning": "maximum ionization-loss coefficient admitted to total field attenuation",
        "source": "propagate.py: alpha_ion gated by use_ionization_loss",
        "unit": "m^-1",
        "use": "separate potential ionization loss from propagated loss",
    },
    "alpha_R_raw_max_z": {
        "meaning": "maximum Raman absorption coefficient calculated before switch gating",
        "source": "propagate.py: selected Raman absorption model",
        "unit": "m^-1",
        "use": "retain Raman-loss diagnostics when propagation feedback is OFF",
    },
    "alpha_R_applied_max_z": {
        "meaning": "maximum Raman absorption coefficient admitted to total field attenuation",
        "source": "propagate.py: alpha_R_raw gated by use_raman_absorption",
        "unit": "m^-1",
        "use": "verify Raman-absorption switch isolation",
    },
    "E_dep_z": {
        "meaning": "ionization plus inverse-Bremsstrahlung energy deposited in each recorded z step",
        "source": "propagate.py: heat_Q_per_z",
        "unit": "J",
        "use": "ionization/plasma heat-deposition history",
    },
    "E_dep_rot_z": {
        "meaning": "rotational-Raman energy deposited in each recorded z step",
        "source": "propagate.py: Raman absorption model",
        "unit": "J",
        "use": "Raman energy-transfer history",
    },
    "E_dep_total_z": {
        "meaning": "sum of ionization/IB and Raman deposited energy for each recorded z step",
        "source": "propagate.py: E_dep_z + E_dep_rot_z",
        "unit": "J",
        "use": "total nonlinear deposition budget",
    },
    "E_dep_cumulative_z": {
        "meaning": "cumulative nonlinear deposited energy",
        "source": "propagate.py: cumulative sum of E_dep_total_z",
        "unit": "J",
        "use": "compare cumulative material deposition along z",
    },
    "U_rel_change_z": {
        "meaning": "pulse-energy change relative to the input propagation plane",
        "source": "propagate.py: (U_z - U0)/U0",
        "unit": "1",
        "use": "energy-conservation and loss monitoring",
    },
    "U_step_change_z": {
        "meaning": "pulse-energy change over each recorded propagation interval",
        "source": "propagate.py: U_z difference from previous record (first uses U0)",
        "unit": "J",
        "use": "localize sudden energy-loss or numerical-instability events",
    },
    "E_loss_from_input_z": {
        "meaning": "observed pulse-energy loss relative to the input propagation plane",
        "source": "propagate.py: U0 - U_z",
        "unit": "J",
        "use": "compare field-energy loss against deposited-energy diagnostics",
    },
    "rho_N2_max_z": {
        "meaning": "maximum N2 electron density over time and transverse coordinates",
        "source": "propagate.py: existing per-species rho_list returned by evolve_rho_time",
        "unit": "m^-3",
        "use": "separate N2 contribution from total plasma density",
    },
    "rho_O2_max_z": {
        "meaning": "maximum O2 electron density over time and transverse coordinates",
        "source": "propagate.py: existing per-species rho_list returned by evolve_rho_time",
        "unit": "m^-3",
        "use": "separate O2 contribution from total plasma density",
    },
    "rho_O2_fraction_at_rho_total_max_z": {
        "meaning": "O2 density fraction at the spatiotemporal maximum of total electron density",
        "source": "propagate.py: rho_O2 / rho_total at argmax(rho_total)",
        "unit": "1",
        "use": "identify species controlling the local total-density maximum",
    },
    "dz_used_z": {
        "meaning": "actual accepted propagation step used to reach each recorded z sample",
        "source": "propagate.py: live dz_try in the propagation loop",
        "unit": "m",
        "use": "distinguish physical changes from actual propagation-step changes",
    },
    "adaptive_rejection_count_z": {
        "meaning": "cumulative count of actual propagation-step rejections at each recorded z sample",
        "source": "propagate.py: live rejection counter",
        "unit": "count",
        "use": "verify whether rejection/retry logic affected a comparison",
    },
    "safety_mode_trigger_count_z": {
        "meaning": "cumulative count of actual safety-mode triggers at each recorded z sample",
        "source": "propagate.py: live safety-mode trigger counter",
        "unit": "count",
        "use": "verify whether safety-mode control altered a comparison",
    },
}

NONLINEAR_DIAGNOSTIC_METADATA.update({
    "raman_operator_applied": {
        "meaning": "whether the complete Isaacs Raman field operator was applied at each z step",
        "source": "propagate.py: explicit use_raman_full_operator gate",
        "unit": "bool",
        "use": "distinguish raw Raman diagnostics from field feedback",
    },
    "raman_rhs_l2_norm": {
        "meaning": "maximum stage L2 norm of the applied full Raman RHS",
        "source": "raman.isaacs_raman_stage",
        "unit": "field/m",
        "use": "verify nonzero full-operator feedback",
    },
    "raman_IR_max_raw": {
        "meaning": "raw maximum delayed Raman intensity response",
        "source": "raman.isaacs_raman_stage",
        "unit": "W/m^2",
        "use": "retain Raman response in ON and feedback-OFF controls",
    },
    "raman_target_loss_step_J": {
        "meaning": "Eq. (10) target rotational energy loss for the current step",
        "source": "raman.isaacs_raman_stage / Heun quadrature",
        "unit": "J",
        "use": "per-step Raman energy closure",
    },
    "raman_actual_loss_step_J": {
        "meaning": "actual field-energy loss caused by the full Raman operator",
        "source": "field fluence before/after the isolated Raman update",
        "unit": "J",
        "use": "per-step Raman energy accounting",
    },
    "raman_closure_residual_step": {
        "meaning": "relative Eq. (10) target-versus-actual Raman loss residual",
        "source": "raman.apply_isaacs_raman_operator_step",
        "unit": "1",
        "use": "Raman energy-closure gate",
    },
    "raman_target_loss_cumulative_J": {
        "meaning": "cumulative Eq. (10) target rotational loss",
        "source": "propagate.py cumulative sum",
        "unit": "J",
        "use": "global Raman energy audit",
    },
    "raman_actual_loss_cumulative_J": {
        "meaning": "cumulative actual Raman field-energy loss",
        "source": "propagate.py cumulative sum",
        "unit": "J",
        "use": "global Raman energy audit",
    },
    "raman_cumulative_closure_residual": {
        "meaning": "relative cumulative target-versus-actual Raman loss residual",
        "source": "propagate.py cumulative energy accounting",
        "unit": "1",
        "use": "full-job Raman closure contract",
    },
    "raman_convolution_count_step": {
        "meaning": "number of Raman convolutions evaluated for the current full step",
        "source": "raman stage diagnostics",
        "unit": "count",
        "use": "verify Heun convolution reuse",
    },
    "raman_operator_walltime_step_s": {
        "meaning": "wall time spent in full Raman stage/update calculations",
        "source": "raman stage timer",
        "unit": "s",
        "use": "runtime projection",
    },
    "raman_operator_substep_count": {
        "meaning": "number of isolated full Raman operator applications in one propagation step",
        "source": "propagate.py nonlinear split order",
        "unit": "count",
        "use": "interpret total convolution count for Strang splitting",
    },
    "linear_walltime_step_s": {
        "meaning": "synchronized wall time of both linear half steps",
        "source": "propagate.py performance instrumentation",
        "unit": "s",
        "use": "full-size runtime projection",
    },
    "ionization_walltime_step_s": {
        "meaning": "synchronized ionization/plasma source calculation wall time",
        "source": "propagate.py performance instrumentation",
        "unit": "s",
        "use": "full-size runtime projection",
    },
    "total_walltime_step_s": {
        "meaning": "synchronized total propagation-step wall time",
        "source": "propagate.py performance instrumentation",
        "unit": "s",
        "use": "full-job walltime estimate",
    },
    "gpu_allocated_step_bytes": {
        "meaning": "CuPy memory-pool bytes actively allocated after each step",
        "source": "cupy default memory pool",
        "unit": "byte",
        "use": "GPU memory gate",
    },
    "gpu_reserved_step_bytes": {
        "meaning": "CuPy memory-pool bytes reserved after each step",
        "source": "cupy default memory pool",
        "unit": "byte",
        "use": "GPU memory gate",
    },
    "raman_energy_projection_iterations": {
        "meaning": "complex64 energy-projection iterations used by the full Eq.27 step",
        "source": "raman._project_complex64_heun_energy",
        "unit": "count",
        "use": "audit storage-rounding closure correction",
    },
    "raman_energy_projection_scale_deviation": {
        "meaning": "maximum absolute deviation of the projection scale from unity",
        "source": "raman._project_complex64_heun_energy",
        "unit": "1",
        "use": "bound the numerical correction applied to the field",
    },
    "raman_energy_projection_initial_residual": {
        "meaning": "stored-field closure residual before the quantization projection",
        "source": "raman._project_complex64_heun_energy",
        "unit": "1",
        "use": "separate Eq.27 closure from complex64 storage error",
    },
})

# Scalar semantic tags are stored alongside a run as text fields.  They are
# deliberately kept out of ``NONLINEAR_TRACE_KEYS``: unlike z histories they
# are not one-dimensional records and must not become mandatory validation
# arrays for legacy or feedback-off runs.
NONLINEAR_DIAGNOSTIC_SEMANTICS = {
    "delta_n_elec_applied_semantics": {
        "meaning": "whether the electronic applied trace is scalar phase feedback or an equivalent n2*I record",
        "unit": "text",
        "use": "interpret delta_n_elec_applied_max_z in the complete complex-operator mode",
    },
    "dphi_kerr_semantics": {
        "meaning": "whether dphi_kerr_max_abs_z is a scalar Kerr phase or is not applicable",
        "unit": "text",
        "use": "prevent treating the zero scalar trace as absence of complete-operator Kerr",
    },
    "self_steepening_semantics": {
        "meaning": "self-steepening representation used by the nonlinear field update",
        "unit": "text",
        "use": "distinguish scalar shock_intensity from the complete product derivative",
    },
}


NONLINEAR_TRACE_KEYS = tuple(NONLINEAR_DIAGNOSTIC_METADATA)

# Existing and Phase-1 histories that must all have one record per z_axis
# entry.  Scalars, text configuration tags, and rho_onaxis_t_z (which has a
# z-leading 2D shape) intentionally do not belong here.
Z_HISTORY_TRACE_KEYS = (
    "U_z",
    "I_max_z",
    "I_onaxis_max_z",
    "I_center_t0_z",
    "w_mom_z",
    "rho_max_z",
    "rho_onaxis_max_z",
    "E_dep_z",
    "fwhm_plasma_z",
    "fwhm_fluence_z",
    "fwhm_time_z",
    "I_onaxis_max_interp_list",
    "E_dep_rot_z",
    "alpha_R_max_z",
    "alpha_R_mean_z",
    "alpha_R_eff_z",
    "alpha_R_closed_z",
    "alpha_R_raw_max_z",
    "alpha_R_applied_max_z",
    "IR_max_z",
    "IR_abs_max_z",
    "delta_n_elec_max_z",
    "delta_n_rot_max_z",
    "delta_n_elec_peak_z",
    "delta_n_rot_peak_z",
    "delta_n_elec_applied_max_z",
    "delta_n_rot_applied_max_z",
    "alpha_ion_raw_max_z",
    "alpha_ion_corr_max_z",
    "alpha_ion_applied_max_z",
    "alpha_ib_max_z",
    "alpha_total_max_z",
    "delta_n_plasma_min_z",
    "delta_n_plasma_applied_min_z",
    "dphi_kerr_max_abs_z",
    "dphi_elec_max_abs_z",
    "dphi_rot_max_abs_z",
    "dphi_plasma_max_abs_z",
    "dphi_elec_applied_max_abs_z",
    "dphi_rot_applied_max_abs_z",
    "dphi_plasma_raw_max_abs_z",
    "dphi_plasma_applied_max_abs_z",
    "E_dep_total_z",
    "E_dep_cumulative_z",
    "U_rel_change_z",
    "U_step_change_z",
    "E_loss_from_input_z",
    "rho_N2_max_z",
    "rho_O2_max_z",
    "rho_O2_fraction_at_rho_total_max_z",
    "dz_used_z",
    "adaptive_rejection_count_z",
    "safety_mode_trigger_count_z",
    "raman_operator_applied",
    "raman_rhs_l2_norm",
    "raman_IR_max_raw",
    "raman_target_loss_step_J",
    "raman_actual_loss_step_J",
    "raman_closure_residual_step",
    "raman_target_loss_cumulative_J",
    "raman_actual_loss_cumulative_J",
    "raman_cumulative_closure_residual",
    "raman_convolution_count_step",
    "raman_operator_walltime_step_s",
    "raman_operator_substep_count",
    "linear_walltime_step_s",
    "ionization_walltime_step_s",
    "total_walltime_step_s",
    "gpu_allocated_step_bytes",
    "gpu_reserved_step_bytes",
    "raman_energy_projection_iterations",
    "raman_energy_projection_scale_deviation",
    "raman_energy_projection_initial_residual",
)


def validate_nonlinear_diagnostics(diag: dict) -> dict:
    """Validate the z-history diagnostics without changing propagation data.

    Empty traces, length mismatches, and non-finite values are hard failures.
    A trace containing only zeros is reported but not failed: zero is physically
    valid when the corresponding nonlinear switch or density is zero.
    """
    if "z_axis" not in diag:
        raise ValueError("diagnostic validation requires z_axis")
    z_axis = _np.asarray(to_cpu(diag["z_axis"]))
    n_records = int(z_axis.size)
    if n_records <= 0:
        raise ValueError("diagnostic validation found an empty z_axis")
    if not _np.all(_np.isfinite(z_axis)):
        raise ValueError("diagnostic validation found non-finite z_axis values")
    # The legacy loop may emit one final record separated only by floating-point
    # roundoff when z_max is reached.  It is still a valid completed history;
    # reject actual backward motion, not a numerically coincident endpoint.
    if n_records > 1 and _np.any(_np.diff(z_axis) < 0.0):
        raise ValueError("diagnostic validation requires a non-decreasing z_axis")

    checked = list(Z_HISTORY_TRACE_KEYS)
    all_zero = []
    for key in checked:
        if key not in diag:
            raise ValueError(f"diagnostic validation missing required trace: {key}")
        values = _np.asarray(to_cpu(diag[key]))
        if values.ndim != 1 or values.size != n_records:
            raise ValueError(
                f"diagnostic validation length mismatch for {key}: "
                f"expected ({n_records},), got {values.shape}"
            )
        if not _np.all(_np.isfinite(values)):
            raise ValueError(f"diagnostic validation found NaN/Inf in {key}")
        if _np.all(values == 0.0):
            all_zero.append(key)

    rho_tz = diag.get("rho_onaxis_t_z")
    if rho_tz is not None:
        rho_tz = _np.asarray(to_cpu(rho_tz))
        if rho_tz.ndim != 2 or rho_tz.shape[0] != n_records:
            raise ValueError(
                "diagnostic validation length mismatch for rho_onaxis_t_z: "
                f"expected first dimension {n_records}, got {rho_tz.shape}"
            )
        if not _np.all(_np.isfinite(rho_tz)):
            raise ValueError("diagnostic validation found NaN/Inf in rho_onaxis_t_z")

    U_z = _np.asarray(to_cpu(diag["U_z"]), dtype=float)
    U_step = _np.asarray(to_cpu(diag["U_step_change_z"]), dtype=float)
    U0 = float(U_z[0] - U_step[0])
    if not _np.isfinite(U0) or U0 <= 0.0:
        raise ValueError("diagnostic validation reconstructed a non-positive input energy")
    expected_rel = (U_z - U0) / (U0 if abs(U0) > 1e-30 else 1e-30)
    if not _np.allclose(_np.asarray(to_cpu(diag["U_rel_change_z"]), dtype=float), expected_rel, rtol=2e-5, atol=2e-8):
        raise ValueError("diagnostic validation failed U_rel_change_z consistency")
    expected_step = _np.diff(_np.concatenate(([U0], U_z)))
    if not _np.allclose(U_step, expected_step, rtol=2e-5, atol=2e-10):
        raise ValueError("diagnostic validation failed U_step_change_z consistency")
    expected_deposition = _np.asarray(to_cpu(diag["E_dep_z"]), dtype=float) + _np.asarray(to_cpu(diag["E_dep_rot_z"]), dtype=float)
    if not _np.allclose(_np.asarray(to_cpu(diag["E_dep_total_z"]), dtype=float), expected_deposition, rtol=2e-5, atol=2e-12):
        raise ValueError("diagnostic validation failed E_dep_total_z consistency")
    if not _np.allclose(_np.asarray(to_cpu(diag["E_dep_cumulative_z"]), dtype=float), _np.cumsum(expected_deposition), rtol=2e-5, atol=2e-12):
        raise ValueError("diagnostic validation failed E_dep_cumulative_z consistency")

    for key in ("rho_N2_max_z", "rho_O2_max_z"):
        values = _np.asarray(to_cpu(diag[key]), dtype=float)
        if _np.any(values < -1e-12):
            raise ValueError(f"diagnostic validation found negative species density in {key}")
    o2_fraction = _np.asarray(to_cpu(diag["rho_O2_fraction_at_rho_total_max_z"]), dtype=float)
    if _np.any((o2_fraction < -1e-8) | (o2_fraction > 1.0 + 1e-8)):
        raise ValueError("diagnostic validation found O2 fraction outside [0, 1]")
    dz_used = _np.asarray(to_cpu(diag["dz_used_z"]), dtype=float)
    if _np.any(dz_used <= 0.0):
        raise ValueError("diagnostic validation requires positive actual dz_used_z")
    for key in ("adaptive_rejection_count_z", "safety_mode_trigger_count_z"):
        values = _np.asarray(to_cpu(diag[key]), dtype=float)
        if _np.any(values < 0.0) or _np.any(_np.diff(values) < 0.0):
            raise ValueError(f"diagnostic validation requires non-decreasing non-negative {key}")

    return {
        "passed": True,
        "z_records": n_records,
        "checked_traces": checked,
        "all_zero_traces": all_zero,
    }


def write_nonlinear_diagnostic_report(path: str | Path, diag: dict, *, npz_path: str | Path) -> dict:
    """Write a compact, self-describing JSON report next to a result NPZ."""
    validation = validate_nonlinear_diagnostics(diag)
    effective_switches = {
        key.removeprefix("nonlinear_"): bool(_np.asarray(to_cpu(value)).item())
        for key, value in diag.items()
        if key.startswith("nonlinear_use_")
    }
    report = {
        "schema": "khz_filament.nonlinear_diagnostics.v1",
        "npz_path": str(npz_path),
        "validation": validation,
        "effective_nonlinear_switches": effective_switches,
        "variables": NONLINEAR_DIAGNOSTIC_METADATA,
        "semantic_fields": NONLINEAR_DIAGNOSTIC_SEMANTICS,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

def intensity(E, n0: float):
    """Compute intensity (W/m^2) from complex field E."""
    return 0.5 * eps0 * c0 * n0 * xp.abs(E)**2

def peak_intensity(I):
    """Return scalar peak intensity."""
    return float(xp.max(I))

def pulse_energy(I, dt: float, dx: float, dy: float):
    """
    Energy in Joules from intensity:
      E = ∭ I(x,y,t) dt dx dy
    """
    return float(xp.sum(I) * dt * dx * dy)

def save_npz(path: str, **arrays):
    """Save arrays to .npz on CPU."""

    cpu_arrays = {k: to_cpu(v) for k, v in arrays.items()}
    xp.savez(path, **cpu_arrays)

"""
Itxy: [Nt,Ny,Nx] 强度
算法：
  1) 对时间积分 -> I2D
  2) 去掉极小背景 (rel_floor)
  3) 按能量累积分位 (frac_keep) 做阈值截断
  4) 仅用被保留区域计算二阶矩半径: w = sqrt( 2 * <r^2> )
"""
def second_moment_radius(I3D, x, y, *, dt,
                         frac_keep: float = 0.999,
                         rel_floor: float = 1e-8) -> float:
    """
    由三维强度 I3D[t,y,x] 计算二阶矩等效半径 w。
    约定：理想高斯 I ∝ exp(-2 r^2 / w^2) 时，本函数返回的 w 即该式中的 w。

    参数
    ----
    I3D : [Nt, Ny, Nx]  强度（非负）
    x   : [Nx]          x 坐标（均匀步进）
    y   : [Ny]          y 坐标（均匀步进）
    dt  : float         时间步长
    frac_keep : (0,1]   仅用累计能量前 frac_keep 的主能量区做二阶矩（抗噪）
    rel_floor : float   按峰值的相对阈值，小于该阈值的像素清零（进一步抗噪）
    """
    # ---- 1) 沿 t 积分得到 fluence F2D[y,x] ----
    F = (xp.trapezoid if hasattr(xp, "trapezoid") else xp.trapz)(xp.asarray(I3D), dx=float(dt), axis=0)  # [Ny, Nx]
    F = xp.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    F = xp.maximum(F, 0.0)

    # ---- 2) 基础过滤：相对门限 ----
    fmax = float(F.max())
    if not math.isfinite(fmax) or fmax <= 0.0:
        return 0.0
    if rel_floor > 0.0:
        F = xp.where(F >= rel_floor * fmax, F, 0.0)

    Ny, Nx = F.shape
    x = xp.asarray(x)
    y = xp.asarray(y)
    dx = float(x[1] - x[0]) if Nx > 1 else 1.0
    dy = float(y[1] - y[0]) if Ny > 1 else 1.0

    # ---- 3) 主能量区选择（避免 CuPy 的 searchsorted 限制）----
    # 扁平化并按强度降序
    if 0.0 < frac_keep < 1.0:
        flat = F.ravel()
        order = xp.argsort(flat)[::-1]
        flat_sorted = flat[order]
        csum = xp.cumsum(flat_sorted, dtype=xp.float64)
        total = float(csum[-1])
        if total <= 0.0:
            return 0.0
        target = frac_keep * total
        # CuPy 对标量 searchsorted 有兼容问题；用计数实现等价功能
        k = int(xp.count_nonzero(csum < target)) + 1  # 第一个使累计≥target的位置（1基）
        k = max(1, min(k, flat_sorted.size))
        thr = float(flat_sorted[k - 1])
        F = xp.where(F >= thr, F, 0.0)

    # ---- 4) 二阶矩计算（矢量化，避免大 meshgrid）----
    S = float(xp.sum(F, dtype=xp.float64) * dx * dy)
    if S <= 0.0 or not math.isfinite(S):
        return 0.0

    x2 = x * x                  # [Nx]
    y2 = y * y                  # [Ny]
    Fx_sum_y = xp.sum(F, axis=0, dtype=xp.float64)  # [Nx]
    Fy_sum_x = xp.sum(F, axis=1, dtype=xp.float64)  # [Ny]
    mom_x2 = float(xp.sum(x2 * Fx_sum_y) * dx * dy)
    mom_y2 = float(xp.sum(y2 * Fy_sum_x) * dx * dy)

    mean_x2 = mom_x2 / S
    mean_y2 = mom_y2 / S

    # 高斯关系：<x^2>=w^2/4, <y^2>=w^2/4 => w = sqrt(2*(<x^2>+<y^2>))
    w_sq = 2.0 * (mean_x2 + mean_y2)
    if w_sq <= 0.0 or not math.isfinite(w_sq):
        return 0.0
    return math.sqrt(w_sq)


def second_moment_radius_from_2d(F2D, x, y, *,
                                 dx=None, dy=None,
                                 frac_keep: float = 0.999,
                                 rel_floor: float = 1e-8) -> float:
    """
    由 2D 面密度(如 fluence) 计算二阶矩等效半径 w。
    约定：理想高斯 I ∝ exp(-2 r^2 / w^2) 时，本函数返回的 w 即该式中的 w。

    参数
    ----
    F2D : [Ny, Nx]  非负实数数组（例如对 I(t,x,y) 沿 t 积分后的 fluence）
    x   : [Nx]      x 坐标
    y   : [Ny]      y 坐标
    dx,dy : float   网格间距（可省略；若省略则以 x、y 的差分首个间距近似）
    frac_keep : (0,1]  仅使用累计能量占比前 frac_keep 的主能量区进行统计
    rel_floor : 相对峰值的强度阈值，小于阈值的像素置零

    返回
    ----
    w : float  二阶矩等效半径（米）
    """
    F = xp.asarray(F2D)
    # 基本清理：非负、去 NaN/Inf、小于阈值置零
    F = xp.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    F = xp.maximum(F, 0.0)
    fmax = float(F.max())
    if not math.isfinite(fmax) or fmax <= 0.0:
        return 0.0
    if rel_floor > 0.0:
        F = xp.where(F >= rel_floor * fmax, F, 0.0)

    Ny, Nx = F.shape
    x = xp.asarray(x)
    y = xp.asarray(y)
    if dx is None:
        dx = float(x[1] - x[0]) if Nx > 1 else 1.0
    if dy is None:
        dy = float(y[1] - y[0]) if Ny > 1 else 1.0

    # 如果启用 frac_keep < 1：仅保留累计能量占比前 frac_keep 的像素
    if 0.0 < frac_keep < 1.0:
        flat = F.ravel()
        # 按强度降序排序，取前 K 使得累计和 >= frac_keep * 总和
        order = xp.argsort(flat)[::-1]
        flat_sorted = flat[order]
        csum = xp.cumsum(flat_sorted, dtype=xp.float64)
        total = float(csum[-1])
        if total <= 0.0:
            return 0.0
        target = frac_keep * total
        # 找到第一个累计超过目标的位置
        k = int(xp.searchsorted(csum, target, side="left"))
        k = max(1, min(k, flat_sorted.size))
        thr = float(flat_sorted[k-1])  # 阈值=第 k 个值
        # 保留 >= 阈值 的像素（注意：可能略多于 frac_keep，对结果影响很小）
        F = xp.where(F >= thr, F, 0.0)

    # 二阶矩：<x^2> 与 <y^2>（对 F 权重）
    # 先构造网格上的 x^2, y^2（矢量外积避免大额中间数组）
    # S = ∬ F dx dy
    S = float(xp.sum(F, dtype=xp.float64) * dx * dy)
    if S <= 0.0 or not math.isfinite(S):
        return 0.0

    x2 = x * x           # [Nx]
    y2 = y * y           # [Ny]
    # ∬ F x^2 dx dy  = (∑_x x^2 ∑_y F) dx dy
    Fx_sum_y = xp.sum(F, axis=0, dtype=xp.float64)             # [Nx]
    Fy_sum_x = xp.sum(F, axis=1, dtype=xp.float64)             # [Ny]
    mom_x2 = float(xp.sum(x2 * Fx_sum_y) * dx * dy)            # ∬ F x^2 dx dy
    mom_y2 = float(xp.sum(y2 * Fy_sum_x) * dx * dy)            # ∬ F y^2 dx dy

    mean_x2 = mom_x2 / S
    mean_y2 = mom_y2 / S
    # 高斯关系：<x^2>=w^2/4, <y^2>=w^2/4 => w = sqrt( 2*(<x^2>+<y^2>) )
    w_sq = 2.0 * (mean_x2 + mean_y2)
    if w_sq <= 0.0 or not math.isfinite(w_sq):
        return 0.0
    return math.sqrt(w_sq)
def parabola_peak(y_minus, y0v, y_plus):
    denom = (y_minus - 2 * y0v + y_plus) + 1e-30
    x_peak = 0.5 * (y_minus - y_plus) / denom
    return y0v - 0.5 * (y_minus - y_plus) * x_peak

def _fwhm_1d_centerline(v, x, i0):
    v = xp.asarray(v)
    vmax = float(v[i0])
    if vmax <= 0.0 or not _np.isfinite(vmax):
        return 0.0
    thr = 0.5 * vmax
    L, R = i0, i0
    n = v.size
    while L > 0 and v[L] >= thr:
        L -= 1
    while R < n - 1 and v[R] >= thr:
        R += 1
    xL = float(x[L]); xR = float(x[R])
    if L < i0 and v[L] < thr:
        w = (thr - float(v[L])) / (float(v[L+1]) - float(v[L]) + 1e-30)
        xL = float(x[L] + w * (x[L+1] - x[L]))
    if R > i0 and v[R] < thr:
        w = (thr - float(v[R])) / (float(v[R-1]) - float(v[R]) + 1e-30)
        xR = float(x[R] + w * (x[R-1] - x[R]))
    return max(0.0, xR - xL)

def _fwhm_diameter_xy_center(r2d, axes, x0i, y0i):
    row = r2d[y0i, :]
    col = r2d[:, x0i]
    fwhm_x = _fwhm_1d_centerline(row, axes.x, x0i)
    fwhm_y = _fwhm_1d_centerline(col, axes.y, y0i)
    return 0.5 * (fwhm_x + fwhm_y)

def _fwhm_time_1d(vt, dt):
    vt = xp.asarray(vt)
    vmax = float(vt.max())
    if vmax <= 0.0 or not _np.isfinite(vmax):
        return 1e-15
    thr = 0.5 * vmax
    idx = xp.where(vt >= thr)[0]
    if idx.size < 2:
        return 1e-15
    tL = float((idx[0]) * dt)
    tR = float((idx[-1]) * dt)
    return max(1e-15, tR - tL)
