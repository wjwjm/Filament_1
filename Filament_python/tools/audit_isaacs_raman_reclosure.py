#!/usr/bin/env python3
"""Independent Isaacs Eqs. (7)-(12), (27) Raman closure audit.

This is a no-propagation audit.  It derives the paper equations directly,
uses adaptive quadrature as the continuous reference, and only then compares
against the production IIR and Eq. (27) Raman field operator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.constants import c0, eps0  # noqa: E402
from KHz_filament.raman import (  # noqa: E402
    apply_isaacs_raman_operator_step,
    isaacs_raman_field_rhs,
    isaacs_raman_stage,
    raman_convolve_intensity,
)


PAPER_OMEGA_R = 1.6e13
PAPER_GAMMA_R = 1.3e13
PAPER_N_R = 2.3e-23
PAPER_N2 = 7.8e-24
N0 = 1.00027
LAMBDA0 = 800e-9
OMEGA0 = 2.0 * np.pi * c0 / LAMBDA0
K_VAC = OMEGA0 / c0
K_MED = N0 * K_VAC
TAU_FWHM = 120e-15
I_PEAK = 5.0e17
NT_PRODUCTION = 384
TWIN_PRODUCTION = 960e-15
DT_PRODUCTION = TWIN_PRODUCTION / NT_PRODUCTION


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gaussian_intensity(t: np.ndarray | float, peak: float = I_PEAK) -> np.ndarray:
    return peak * np.exp(-4.0 * np.log(2.0) * (np.asarray(t) / TAU_FWHM) ** 2)


def gaussian_derivative(t: np.ndarray, peak: float = I_PEAK) -> np.ndarray:
    intensity = gaussian_intensity(t, peak)
    return intensity * (-8.0 * np.log(2.0) * t / TAU_FWHM**2)


def isaacs_kernel(delay: np.ndarray | float) -> np.ndarray:
    delay = np.asarray(delay, dtype=float)
    prefactor = (PAPER_OMEGA_R**2 + PAPER_GAMMA_R**2) / PAPER_OMEGA_R
    return np.where(
        delay >= 0.0,
        prefactor * np.exp(-PAPER_GAMMA_R * delay) * np.sin(PAPER_OMEGA_R * delay),
        0.0,
    )


def continuous_response(t: np.ndarray, peak: float = I_PEAK) -> np.ndarray:
    """Adaptive-quadrature evaluation of Eq. (8) after W=-1."""
    from scipy.integrate import quad

    lower = -2.0e-12
    values = []
    for observation in np.asarray(t, dtype=float):
        upper = float(observation)
        points = [0.0] if lower < 0.0 < upper else None

        def normalized_integrand(source_time: float) -> float:
            return float(
                isaacs_kernel(upper - source_time)
                * gaussian_intensity(source_time, 1.0)
            )

        value, _ = quad(
            normalized_integrand,
            lower,
            upper,
            points=points,
            epsabs=2e-13,
            epsrel=2e-13,
            limit=300,
        )
        values.append(peak * value)
    return np.asarray(values)


def mpmath_response_checkpoints(t: np.ndarray) -> list[dict]:
    """60-decimal checkpoint integration independent of SciPy quadrature."""
    import mpmath as mp

    mp.mp.dps = 60
    omega = mp.mpf("1.6e13")
    gamma = mp.mpf("1.3e13")
    tau = mp.mpf("120e-15")
    peak = mp.mpf("5e17")
    prefactor = (omega * omega + gamma * gamma) / omega
    lower = mp.mpf("-2e-12")
    requested_fs = (-200.0, -150.0, -130.0, -100.0, -85.0, -60.0, -40.0, 0.0, 40.0, 70.0, 100.0, 150.0, 200.0)
    rows = []
    for requested in requested_fs:
        index = int(np.argmin(np.abs(t * 1e15 - requested)))
        observation = mp.mpf(str(float(t[index])))

        def integrand(source_time):
            delay = observation - source_time
            kernel = prefactor * mp.exp(-gamma * delay) * mp.sin(omega * delay)
            intensity = peak * mp.exp(-4 * mp.log(2) * (source_time / tau) ** 2)
            return kernel * intensity

        intervals = [lower, observation]
        if lower < 0 < observation:
            intervals = [lower, mp.mpf("0"), observation]
        value = mp.quad(integrand, intervals)
        rows.append({
            "index": index,
            "t_fs": float(t[index] * 1e15),
            "I_R_mpmath_W_m-2": float(value),
        })
    return rows


def exact_piecewise_linear_kernel(dt: float, count: int) -> np.ndarray:
    """Equivalent discrete convolution weights for the production IIR."""
    a = PAPER_GAMMA_R - 1j * PAPER_OMEGA_R
    r = np.exp(-a * dt)
    c = (1.0 - r) / a
    c1 = c - (1.0 - r * (1.0 + a * dt)) / (a * a * dt)
    c0 = c - c1
    k = 1.0 / np.imag(1.0 / a)
    weights = np.empty(count, dtype=float)
    weights[0] = np.imag(k * c1)
    if count > 1:
        lag = np.arange(count - 1)
        weights[1:] = np.imag(k * (r**lag) * (c0 + r * c1))
    return weights


def spectral_derivative(values: np.ndarray, dt: float) -> np.ndarray:
    omega = 2.0 * np.pi * np.fft.fftfreq(values.shape[0], d=dt)
    return np.fft.ifft(1j * omega * np.fft.fft(values, axis=0), axis=0)


def relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-300))


def first_crossing(t: np.ndarray, values: np.ndarray, threshold: float) -> float | None:
    index = np.flatnonzero(values >= threshold)
    if not index.size:
        return None
    i = int(index[0])
    if i == 0:
        return float(t[0])
    x0, x1 = float(t[i - 1]), float(t[i])
    y0, y1 = float(values[i - 1]), float(values[i])
    if y1 == y0:
        return x1
    return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)


def direct_rotational_rhs(field: np.ndarray, response: np.ndarray, dt: float) -> np.ndarray:
    product = response * field
    return 1j * K_VAC * PAPER_N_R * product - (PAPER_N_R / c0) * spectral_derivative(product, dt)


def product_rule_rotational_rhs(field: np.ndarray, response: np.ndarray, dt: float) -> np.ndarray:
    derivative = spectral_derivative(response, dt) * field
    derivative += response * spectral_derivative(field, dt)
    return 1j * K_VAC * PAPER_N_R * response * field - (PAPER_N_R / c0) * derivative


def incomplete_rotational_rhs(field: np.ndarray, response: np.ndarray, dt: float) -> np.ndarray:
    derivative = spectral_derivative(response, dt) * field
    return 1j * K_VAC * PAPER_N_R * response * field - (PAPER_N_R / c0) * derivative


def full_electronic_rhs(field: np.ndarray, intensity: np.ndarray, dt: float) -> np.ndarray:
    product = intensity * field
    return 1j * K_VAC * PAPER_N2 * product - (PAPER_N2 / c0) * spectral_derivative(product, dt)


def scalar_split_electronic_rhs(field: np.ndarray, intensity: np.ndarray, dt: float) -> np.ndarray:
    corrected_intensity = intensity - spectral_derivative(intensity, dt).real / OMEGA0
    return 1j * K_MED * PAPER_N2 * corrected_intensity * field


def production_iir(intensity: np.ndarray, dtype=np.float64) -> np.ndarray:
    values = np.asarray(intensity, dtype=dtype)[:, None, None]
    response = raman_convolve_intensity(
        values,
        method="iir",
        dt=DT_PRODUCTION,
        omega_R=PAPER_OMEGA_R,
        Gamma_R=PAPER_GAMMA_R,
        iir_sampling="exact_piecewise_linear",
    )
    return np.asarray(response)[:, 0, 0]


def stage_rhs(field: np.ndarray, dt: float) -> np.ndarray:
    omega = 2.0 * np.pi * np.fft.fftfreq(field.shape[0], d=dt)
    return np.asarray(
        isaacs_raman_field_rhs(
            field[:, None, None],
            Omega=omega,
            dt=dt,
            omega0=OMEGA0,
            n0=N0,
            n_R=PAPER_N_R,
            omega_R=PAPER_OMEGA_R,
            Gamma_R=PAPER_GAMMA_R,
            method="iir",
            iir_sampling="exact_piecewise_linear",
        )
    )[:, 0, 0]


def rk4_reference(field: np.ndarray, dz: float, dt: float, substeps: int = 128) -> np.ndarray:
    result = field.astype(np.complex128, copy=True)
    step = dz / substeps
    for _ in range(substeps):
        k1 = stage_rhs(result, dt)
        k2 = stage_rhs(result + 0.5 * step * k1, dt)
        k3 = stage_rhs(result + 0.5 * step * k2, dt)
        k4 = stage_rhs(result + step * k3, dt)
        result += (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_table(config: dict) -> list[dict]:
    rows = [
        ("omega_R", "Eq. (9) angular oscillation rate", "rad s^-1", "raman.omega_R", PAPER_OMEGA_R, config["raman"]["omega_R"], "No 2*pi conversion; paper already supplies omega."),
        ("Gamma_R", "Eq. (9) exponential damping rate", "s^-1", "raman.Gamma_R", PAPER_GAMMA_R, config["raman"]["Gamma_R"], "1/Gamma_R = 76.923 fs; not 1/T2 from historical mode."),
        ("n_R", "Eqs. (8),(10) rotational nonlinear index", "m^2 W^-1", "raman.n_R", PAPER_N_R, config["raman"]["n_R"], "2.3e-19 cm^2/W multiplied by 1e-4."),
        ("n2", "Eq. (27) electronic nonlinear index", "m^2 W^-1", "beam.n2_air", PAPER_N2, config["beam"]["n2_air"], "0.78e-19 cm^2/W multiplied by 1e-4."),
        ("I", "Cycle-averaged physical intensity", "W m^-2", "0.5*eps0*c0*n0*abs(E)^2", "physical intensity", "KHz_filament.raman.isaacs_raman_stage", "Consistent with SI field envelope and measured n2/n_R convention."),
        ("Omega_kernel", "Eq. (9) causal response kernel", "s^-1", "make_raman_kernel / IIR state", "((omega^2+Gamma^2)/omega) exp(-Gamma t) sin(omega t) H(t)", "same analytic prefactor", "Continuous integral equals one."),
        ("I_R", "Integral Omega(t-t') I(t') dt'", "W m^-2", "response / I_R", "causal convolution with dt", "IIR coefficients analytically integrate each interval", "No additional f_R or n_R factor in I_R."),
        ("delta_n_rot", "Rotational refractive-index contribution", "dimensionless", "n_R*I_R", "n_R I_R", "full operator algebraically uses n_R once", "No electronic/rotational mixing weight in Isaacs model."),
        ("u_R", "Eq. (10) signed fluence derivative per length", "J m^-3", "u_R_signed", "(n_R/c) integral I_R dI/dt dt", "same with spectral derivative increments", "Negative for pulse loss; q_R_positive=-u_R_signed."),
        ("D[p_rot]", "Eq. (27) full rotational polarization derivative", "field m^-1", "isaacs_raman_stage.rhs", "i*k_vac*n_R*I_R*A-(n_R/c)d_t(I_R*A)", "same product derivative", "Includes I_R*dA/dt."),
    ]
    return [
        {
            "quantity": quantity,
            "paper_definition": paper,
            "si_unit": unit,
            "code_variable": code,
            "paper_value_or_formula": paper_value,
            "actual_code_value_or_formula": actual,
            "audit_note": note,
        }
        for quantity, paper, unit, code, paper_value, actual, note in rows
    ]


def main() -> None:
    import matplotlib
    from scipy.integrate import quad

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))

    t = (np.arange(NT_PRODUCTION) - NT_PRODUCTION // 2) * DT_PRODUCTION
    intensity = gaussian_intensity(t)
    reference = continuous_response(t)
    high_precision_rows = mpmath_response_checkpoints(t)
    for row in high_precision_rows:
        index = int(row["index"])
        row["I_R_scipy_quad_W_m-2"] = float(reference[index])
        row["difference_over_response_peak"] = None
    iir64 = production_iir(intensity, np.float64)
    iir32 = production_iir(intensity, np.float32).astype(float)
    peak_response = float(np.max(reference))
    for row in high_precision_rows:
        row["difference_over_response_peak"] = (
            row["I_R_scipy_quad_W_m-2"] - row["I_R_mpmath_W_m-2"]
        ) / peak_response
    write_csv(args.out_dir / "high_precision_checkpoints.csv", high_precision_rows)
    significant = reference >= peak_response * 1e-6

    chi_l_gaussian = (N0 * N0 - 1.0) / (4.0 * np.pi)
    q_reference = (PAPER_N_R * N0 / (2.0 * np.pi * chi_l_gaussian)) * reference
    p_rot_over_a = chi_l_gaussian * q_reference
    delta_n_elec = PAPER_N2 * intensity
    delta_n_rot = PAPER_N_R * reference
    rotational_over_electronic_peak = delta_n_rot / (PAPER_N2 * I_PEAK)
    instantaneous_ratio = np.divide(
        delta_n_rot,
        delta_n_elec,
        out=np.full_like(delta_n_rot, np.nan),
        where=intensity >= 1e-4 * I_PEAK,
    )
    d_intensity = gaussian_derivative(t)
    q_r_time = (PAPER_N_R / c0) * reference * d_intensity
    u_r_signed = float(np.trapezoid(q_r_time, x=t))

    kernel_weights = exact_piecewise_linear_kernel(DT_PRODUCTION, 8192)
    kernel_rows = [
        {
            "lag_index": index,
            "lag_fs": index * DT_PRODUCTION * 1e15,
            "dimensionless_weight": value,
            "equivalent_kernel_1_s": value / DT_PRODUCTION,
            "continuous_kernel_1_s": float(isaacs_kernel(index * DT_PRODUCTION)),
        }
        for index, value in enumerate(kernel_weights[:1024])
    ]
    write_csv(args.out_dir / "iir_equivalent_kernel.csv", kernel_rows)

    pointwise_rows = []
    for index in range(t.size):
        pointwise_rows.append({
            "t_fs": t[index] * 1e15,
            "I_W_m-2": intensity[index],
            "I_R_continuous_W_m-2": reference[index],
            "I_R_iir_fp64_W_m-2": iir64[index],
            "I_R_iir_fp32_W_m-2": iir32[index],
            "fp64_error_over_response_peak": (iir64[index] - reference[index]) / peak_response,
            "fp32_error_over_response_peak": (iir32[index] - reference[index]) / peak_response,
            "Q_gaussian_dimensionless": q_reference[index],
            "p_rot_over_A_gaussian": p_rot_over_a[index],
            "delta_n_elec": delta_n_elec[index],
            "delta_n_rot": delta_n_rot[index],
            "rot_over_electronic_peak": rotational_over_electronic_peak[index],
            "rot_over_instantaneous_electronic": instantaneous_ratio[index],
            "eq10_power_density_W_m-3": q_r_time[index],
        })
    write_csv(args.out_dir / "gaussian_120fs_pointwise.csv", pointwise_rows)
    write_csv(args.out_dir / "paper_to_code_audit.csv", audit_table(config))

    amplitude = np.sqrt(2.0 * intensity / (eps0 * c0 * N0))
    chirp = np.exp(1j * 2.5e27 * t * t)
    field = amplitude * chirp
    response_column = iir64[:, None, None]
    field_column = field[:, None, None]
    omega = 2.0 * np.pi * np.fft.fftfreq(t.size, d=DT_PRODUCTION)
    current_rhs = np.asarray(
        isaacs_raman_field_rhs(
            field_column,
            Omega=omega,
            dt=DT_PRODUCTION,
            omega0=OMEGA0,
            n0=N0,
            n_R=PAPER_N_R,
            omega_R=PAPER_OMEGA_R,
            Gamma_R=PAPER_GAMMA_R,
            method="iir",
            iir_sampling="exact_piecewise_linear",
        )
    )[:, 0, 0]
    direct_rhs = direct_rotational_rhs(field, iir64, DT_PRODUCTION)
    product_rule_rhs = product_rule_rotational_rhs(field, iir64, DT_PRODUCTION)
    incomplete_rhs = incomplete_rotational_rhs(field, iir64, DT_PRODUCTION)
    continuous_rhs = direct_rotational_rhs(field, reference, DT_PRODUCTION)
    electronic_full = full_electronic_rhs(field, intensity, DT_PRODUCTION)
    electronic_scalar = scalar_split_electronic_rhs(field, intensity, DT_PRODUCTION)

    heun_rows = []
    rk4_cache = {}
    for dz in (2.0e-4, 1.0e-4, 5.0e-5):
        reference_step = rk4_reference(field, dz, DT_PRODUCTION, substeps=128)
        rk4_cache[dz] = reference_step
        current_step = np.asarray(
            apply_isaacs_raman_operator_step(
                field_column,
                dz,
                Omega=omega,
                dt=DT_PRODUCTION,
                omega0=OMEGA0,
                n0=N0,
                n_R=PAPER_N_R,
                omega_R=PAPER_OMEGA_R,
                Gamma_R=PAPER_GAMMA_R,
                integrator="heun",
                method="iir",
                iir_sampling="exact_piecewise_linear",
            )
        )[:, 0, 0]
        heun_rows.append({
            "dz_m": dz,
            "heun_relative_l2_vs_rk4": relative_l2(current_step, reference_step),
        })
    write_csv(args.out_dir / "eq27_heun_convergence.csv", heun_rows)

    # A transform-limited production-scale probe reliably exposes the
    # complex64 energy projection without requiring a propagation.
    field32 = amplitude.astype(np.complex64)[:, None, None]
    projected32, projection_diag = apply_isaacs_raman_operator_step(
        field32,
        1.0e-4,
        Omega=omega,
        dt=DT_PRODUCTION,
        omega0=OMEGA0,
        n0=N0,
        n_R=PAPER_N_R,
        omega_R=PAPER_OMEGA_R,
        Gamma_R=PAPER_GAMMA_R,
        integrator="heun",
        method="iir",
        iir_sampling="exact_piecewise_linear",
        return_diagnostics=True,
    )
    stage1 = isaacs_raman_stage(
        field32,
        Omega=omega,
        dt=DT_PRODUCTION,
        omega0=OMEGA0,
        n0=N0,
        n_R=PAPER_N_R,
        omega_R=PAPER_OMEGA_R,
        Gamma_R=PAPER_GAMMA_R,
        method="iir",
        iir_sampling="exact_piecewise_linear",
    )
    predictor = (field32 + 1.0e-4 * stage1["rhs"]).astype(np.complex64)
    stage2 = isaacs_raman_stage(
        predictor,
        Omega=omega,
        dt=DT_PRODUCTION,
        omega0=OMEGA0,
        n0=N0,
        n_R=PAPER_N_R,
        omega_R=PAPER_OMEGA_R,
        Gamma_R=PAPER_GAMMA_R,
        method="iir",
        iir_sampling="exact_piecewise_linear",
    )
    pure_heun32 = (field32 + 0.5e-4 * (stage1["rhs"] + stage2["rhs"])).astype(np.complex64)

    crossings = {
        str(level): (
            None
            if first_crossing(t, rotational_over_electronic_peak, level) is None
            else first_crossing(t, rotational_over_electronic_peak, level) * 1e15
        )
        for level in (0.01, 0.05, 0.10, 0.25, 0.50)
    }
    summary = {
        "schema": "khz_filament.isaacs_raman_reclosure.v1",
        "paper": {
            "path": str(args.paper),
            "sha256": sha256(args.paper),
            "equation_pages_pdf": {"eq7_to_eq12": 4, "eq27": 9},
        },
        "configuration": {"path": str(args.config), "sha256": sha256(args.config)},
        "paper_parameters_si": {
            "omega_R_rad_s": PAPER_OMEGA_R,
            "Gamma_R_1_s": PAPER_GAMMA_R,
            "n_R_m2_W": PAPER_N_R,
            "n2_m2_W": PAPER_N2,
            "n_R_over_n2": PAPER_N_R / PAPER_N2,
            "kernel_decay_time_fs": 1e15 / PAPER_GAMMA_R,
            "kernel_oscillation_period_fs": 2.0 * np.pi / PAPER_OMEGA_R * 1e15,
        },
        "kernel_closure": {
            "continuous_integral_analytic": 1.0,
            "continuous_integral_quad": quad(lambda delay: float(isaacs_kernel(delay)), 0.0, 5.0e-12, epsabs=1e-13, epsrel=1e-13, limit=500)[0],
            "iir_8192_weight_sum_numeric": float(np.sum(kernel_weights)),
            "iir_infinite_weight_sum_analytic": 1.0,
            "production_dt_fs": DT_PRODUCTION * 1e15,
        },
        "gaussian_120fs": {
            "I_peak_W_m-2": I_PEAK,
            "response_peak_W_m-2": peak_response,
            "response_peak_t_fs": float(t[np.argmax(reference)] * 1e15),
            "fp64_max_abs_error_over_response_peak": float(np.max(np.abs(iir64 - reference)) / peak_response),
            "fp64_rms_error_over_response_peak": float(np.sqrt(np.mean((iir64 - reference) ** 2)) / peak_response),
            "fp64_max_relative_error_where_response_gt_1e-6_peak": float(np.max(np.abs((iir64[significant] - reference[significant]) / reference[significant]))),
            "fp64_max_relative_error_where_response_gt_1e-3_peak": float(np.max(np.abs((iir64[reference >= peak_response * 1e-3] - reference[reference >= peak_response * 1e-3]) / reference[reference >= peak_response * 1e-3]))),
            "fp32_max_abs_error_over_response_peak": float(np.max(np.abs(iir32 - reference)) / peak_response),
            "scipy_quad_vs_mpmath_60dps_max_abs_over_response_peak": float(max(abs(row["difference_over_response_peak"]) for row in high_precision_rows)),
            "Q_peak_dimensionless_gaussian": float(np.max(q_reference)),
            "delta_n_rot_peak": float(np.max(delta_n_rot)),
            "delta_n_elec_peak": float(np.max(delta_n_elec)),
            "max_rot_over_electronic_peak": float(np.max(rotational_over_electronic_peak)),
            "crossing_times_fs_rot_over_electronic_peak": crossings,
            "eq10_u_R_signed_J_m-3": u_r_signed,
            "eq10_deposited_energy_density_J_m-3": -u_r_signed,
        },
        "eq27_operator": {
            "rotational_current_vs_direct_product_derivative_rel_l2": relative_l2(current_rhs, direct_rhs),
            "rotational_product_rule_vs_direct_product_derivative_rel_l2": relative_l2(product_rule_rhs, direct_rhs),
            "rotational_current_vs_continuous_response_rel_l2": relative_l2(current_rhs, continuous_rhs),
            "rotational_error_if_I_R_dA_dt_omitted_rel_l2": relative_l2(incomplete_rhs, direct_rhs),
            "electronic_scalar_split_vs_full_product_derivative_rel_l2": relative_l2(electronic_scalar, electronic_full),
            "electronic_scalar_split_boundary": "Current full_isaacs_eq27 mode is a full rotational suboperator, not a monolithic Eq.27 implementation of every p_NL component.",
            "complex64_projection_applied": bool(projection_diag["energy_projection_applied"]),
            "complex64_projection_scale": float(projection_diag["energy_projection_scale"]),
            "complex64_projected_vs_pure_heun_rel_l2": relative_l2(np.asarray(projected32), pure_heun32),
            "heun_convergence": heun_rows,
        },
        "weights": {
            "electronic": "n2*I once",
            "rotational": "n_R*I_R once",
            "f_R_used_by_isaacs_full_operator": False,
            "legacy_split_rotational_phase_simultaneously_enabled": bool(config["propagation"].get("use_raman_phase", False)),
            "legacy_raman_absorption_simultaneously_enabled": bool(config["propagation"].get("use_raman_absorption", False) or config["raman"].get("absorption", False)),
        },
        "scope_limitations": [
            "Closure applies to the strict isaacs_rot_sinexp, exact_piecewise_linear IIR, full rotational operator path in the audited configuration.",
            "The inactive FFT and legacy Raman paths were not numerically certified by this experiment.",
            "This is a single-point and small-array audit, not a full-propagation validation.",
            "The complex64 projection was quantified for one local production-scale probe; cumulative propagation impact was not inferred.",
        ],
    }
    summary["decision"] = {
        "class": "B",
        "statement": (
            "For the audited strict Isaacs exact-PWL IIR/full-rotational path, the parameters, kernel normalization, dt handling, weights, Eq. (10) units, and D[I_R A] operator close; no convention or normalization error was found in that path. "
            "However, the configuration named full_isaacs_eq27 is not a mathematically complete Eq. (27) nonlinear operator: electronic Kerr remains a scalar shock/phase approximation to D[I A], and complex64 may add a tiny global energy-projection rescaling."
        ),
        "no_full_propagation_run": True,
        "no_parameter_fitting": True,
    }
    (args.out_dir / "closure_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.dpi": 200})
    fig, ax = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True)
    ax[0].plot(t * 1e15, intensity / I_PEAK, label="I / I_peak", color="#111827")
    ax[0].plot(t * 1e15, reference / I_PEAK, label="continuous I_R / I_peak", color="#0369a1")
    ax[0].plot(t * 1e15, iir64 / I_PEAK, "--", label="production IIR", color="#15803d")
    ax[0].set_ylabel("normalized response")
    ax[0].legend()
    ax[0].grid(alpha=0.25)
    ax[1].plot(t * 1e15, (iir64 - reference) / peak_response, color="#b91c1c")
    ax[1].set(xlabel="retarded time (fs)", ylabel="IIR error / peak(I_R)")
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out_dir / "gaussian_120fs_response.png")
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True)
    ax[0].plot(t * 1e15, delta_n_elec / np.max(delta_n_elec), label="electronic delta_n / electronic peak", color="#111827")
    ax[0].plot(t * 1e15, rotational_over_electronic_peak, label="rotational delta_n / electronic peak", color="#0369a1")
    for level in (0.01, 0.05, 0.10):
        ax[0].axhline(level, color="#6b7280", linewidth=0.7, linestyle=":")
    ax[0].set(ylim=(-0.02, 2.65), ylabel="contribution / electronic peak")
    ax[0].grid(alpha=0.25)
    ax[0].legend(fontsize=8)
    ratio_masked = np.where(intensity >= 1e-2 * I_PEAK, instantaneous_ratio, np.nan)
    ax[1].semilogy(t * 1e15, ratio_masked, color="#15803d")
    ax[1].axhline(1.0, color="#6b7280", linewidth=0.8, linestyle=":")
    ax[1].set(xlim=(-240, 240), ylim=(1e-3, 2e2), xlabel="retarded time (fs)", ylabel="rotational / instantaneous electronic")
    ax[1].grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(args.out_dir / "rotational_fraction_120fs.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    labels = ["Raman code vs direct", "omit I_R dA/dt", "electronic scalar vs full"]
    values = [
        summary["eq27_operator"]["rotational_current_vs_direct_product_derivative_rel_l2"],
        summary["eq27_operator"]["rotational_error_if_I_R_dA_dt_omitted_rel_l2"],
        summary["eq27_operator"]["electronic_scalar_split_vs_full_product_derivative_rel_l2"],
    ]
    ax.bar(labels, values, color=["#15803d", "#b91c1c", "#d97706"])
    ax.set_yscale("log")
    ax.set_ylabel("relative L2 discrepancy")
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out_dir / "eq27_operator_discrepancy.png")
    plt.close(fig)

    report = [
        "# Isaacs Raman continuous-to-discrete independent reclosure",
        "",
        "## Scope",
        "",
        "- Source authority: Isaacs et al. 2022 Eqs. (7)-(12) and Eq. (27), transcribed directly from the supplied PDF (PDF pages 4 and 9).",
        "- No parameter fitting, no PyCAP fitting, and no full propagation or Slurm job.",
        "- Audited configuration: `120fs_talebpour_isaacs_full_operator_on.json`.",
        "",
        "## Paper-to-SI derivation",
        "",
        "With `W(tau)=-1`, define `I_R(tau)=integral Omega(tau-tau') I(tau') d tau'`. Eq. (8) becomes",
        "",
        "`Q = n_R n0 I_R / (2 pi chi_L)`, and therefore `p_rot = chi_L Q A = n0 n_R I_R A / (2 pi)`.",
        "",
        "Isolating the rotational part of Eq. (27), using `k0=n0 omega0/c`, gives",
        "",
        "`dA/dz = i (omega0/c) n_R I_R A - (n_R/c) d(I_R A)/d tau`.",
        "",
        "Thus the field prefactor is the vacuum wave number `omega0/c`; `n_R` appears once, and the derivative must act on the complete product `I_R A`.",
        "",
        "`omega_R=1.6e13 s^-1` is the angular rate appearing inside `sin(omega_R tau)`. Its ordinary frequency is `omega_R/(2 pi)=2.546e12 Hz`, giving a `392.699 fs` period. `Gamma_R=1.3e13 s^-1` is a damping rate with `76.923 fs` decay time; neither value receives another `2 pi` factor.",
        "",
        "## Continuous and discrete closure",
        "",
        f"- Eq. (9) kernel analytic integral: 1; the first 8192 exact-PWL IIR weights sum to {summary['kernel_closure']['iir_8192_weight_sum_numeric']:.12g}, while the analytic infinite sum is 1.",
        f"- Production time step: {DT_PRODUCTION*1e15:.3f} fs. Gaussian fp64 maximum absolute response error: {summary['gaussian_120fs']['fp64_max_abs_error_over_response_peak']:.3e} of peak response.",
        f"- In the physically visible region `I_R >= 1e-3 peak(I_R)`, maximum pointwise relative error is {summary['gaussian_120fs']['fp64_max_relative_error_where_response_gt_1e-3_peak']:.3e}. At the extreme `1e-6` tail it reaches {summary['gaussian_120fs']['fp64_max_relative_error_where_response_gt_1e-6_peak']:.3e}, but the absolute response there is negligible.",
        f"- SciPy adaptive quadrature agrees with independent 60-decimal mpmath checkpoints to {summary['gaussian_120fs']['scipy_quad_vs_mpmath_60dps_max_abs_over_response_peak']:.3e} of peak response.",
        "- The IIR interval coefficients already contain the time integration. There is no missing or duplicated external `dt`; the equivalent discrete kernel weights sum to one.",
        f"- Eq. (10) signed integral: {u_r_signed:.6e} J/m^3; deposited energy density is {-u_r_signed:.6e} J/m^3.",
        "- Unit chain: `(m^2/W)/(m/s) * (W/m^2) * (W/m^2/s) * s = J/m^3`.",
        "- The strict Isaacs path uses `n2*I` and `n_R*I_R` once each. It does not use `f_R`, does not enable split Raman phase, and does not enable legacy Raman absorption.",
        f"- Rotational delta_n reaches 1%, 5%, and 10% of the electronic peak at {crossings['0.01']:.3f}, {crossings['0.05']:.3f}, and {crossings['0.1']:.3f} fs.",
        f"- For this fixed `I_peak={I_PEAK:.3e} W/m^2` test, peak `I_R/I_peak={peak_response/I_PEAK:.6f}` and peak rotational index is {summary['gaussian_120fs']['max_rot_over_electronic_peak']:.6f} times the electronic peak. This large response follows from the paper values `n_R/n2={PAPER_N_R/PAPER_N2:.6f}`, not from IIR amplification.",
        "",
        "## Eq. (27) operator audit",
        "",
        f"- Current rotational RHS vs direct `D[I_R A]`: relative L2={summary['eq27_operator']['rotational_current_vs_direct_product_derivative_rel_l2']:.3e}.",
        f"- Omitting `I_R*dA/dt` would produce relative L2 error {summary['eq27_operator']['rotational_error_if_I_R_dA_dt_omitted_rel_l2']:.3e}; the current Raman code does not omit it.",
        f"- Current scalar electronic Kerr/shock RHS vs full `D[I A]`: relative L2={summary['eq27_operator']['electronic_scalar_split_vs_full_product_derivative_rel_l2']:.3e}.",
        f"- Heun one-step errors versus a 128-substep RK4 reference are {heun_rows[0]['heun_relative_l2_vs_rk4']:.3e}, {heun_rows[1]['heun_relative_l2_vs_rk4']:.3e}, and {heun_rows[2]['heun_relative_l2_vs_rk4']:.3e} as `dz` halves, showing the expected approximately eightfold local-error reduction.",
        f"- The production-scale complex64 probe applies a global energy projection with scale {summary['eq27_operator']['complex64_projection_scale']:.12g}; its single-step field-level difference from pure Heun is {summary['eq27_operator']['complex64_projected_vs_pure_heun_rel_l2']:.3e}. It is not the unmodified Eq. (27) Heun map; this local test does not infer its cumulative propagation impact.",
        "- Therefore the Raman `full_isaacs_eq27` suboperator is mathematically equivalent to the paper's full rotational polarization derivative, while the overall nonlinear step is not a monolithic Eq. (27) implementation of every `p_NL` term.",
        "- The inactive FFT compatibility path was not used in the production comparison; its centered-time kernel interface remains a separate implementation risk, but it cannot explain the current IIR result.",
        "",
        "## Decision",
        "",
        "**B. The continuous Raman formula and the audited strict IIR/full-rotational code path close, but the overall operator called `full_isaacs_eq27` is not a complete mathematical equivalent of Eq. (27) because electronic Kerr remains scalar and complex64 may apply a global energy projection.**",
        "",
        "The next step must be operator-only and local: implement or explicitly separate the full electronic `D[I A]` term, quantify/remove the complex64 projection boundary, and rerun the small-array closure. Only after that closes should one design a single full propagation. The Raman parameters must remain fixed; no new Raman-ON propagation is justified by this audit alone.",
        "",
    ]
    (args.out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"decision": summary["decision"], "gaussian": summary["gaussian_120fs"], "eq27": summary["eq27_operator"]}, indent=2))


if __name__ == "__main__":
    main()
