"""CPU-only reference calculations for Isaacs Eqs. (9)-(11).

This module is deliberately independent from production propagation.  In
particular, ``q_R_positive`` is applied only after the complete signed time
integral has been evaluated; it is not a per-sample clipping rule.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

C0 = 299792458.0


def isaacs_kernel(tau: np.ndarray, omega_R: float, Gamma_R: float) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    if omega_R <= 0.0 or Gamma_R < 0.0:
        raise ValueError("omega_R must be > 0 and Gamma_R must be >= 0")
    pref = (omega_R * omega_R + Gamma_R * Gamma_R) / omega_R
    return pref * np.exp(-Gamma_R * np.maximum(tau, 0.0)) * np.sin(omega_R * np.maximum(tau, 0.0)) * (tau >= 0.0)


def causal_convolution_direct(intensity: np.ndarray, kernel: np.ndarray, dt: float) -> np.ndarray:
    return np.convolve(np.asarray(intensity, dtype=float), np.asarray(kernel, dtype=float), mode="full")[: len(intensity)] * float(dt)


def boxcar_edge_signed_energy(intensity0: float, pulse_duration: float, *, n_R: float, omega_R: float, Gamma_R: float) -> float:
    """Eq. (10) using the analytic distributional derivative of a boxcar.

    ``dI/dtau = I0 [delta(tau) - delta(tau-tau_p)]`` and causal ``I_R(0)=0``
    give the signed energy density injected into rotational modes.
    """
    bracket = 1.0 - np.exp(-Gamma_R * pulse_duration) * (
        np.cos(omega_R * pulse_duration) + (Gamma_R / omega_R) * np.sin(omega_R * pulse_duration)
    )
    return -(n_R / C0) * intensity0 * intensity0 * bracket


def eq11_alpha(intensity0: float, pulse_duration: float, *, n_R: float, omega_R: float, Gamma_R: float) -> float:
    if pulse_duration <= 0.0:
        raise ValueError("pulse_duration must be positive")
    bracket = 1.0 - np.exp(-Gamma_R * pulse_duration) * (
        np.cos(omega_R * pulse_duration) + (Gamma_R / omega_R) * np.sin(omega_R * pulse_duration)
    )
    return (n_R * intensity0 / (C0 * pulse_duration)) * bracket


@dataclass(frozen=True)
class SignedEnergy:
    u_R_signed: float
    q_R_positive: float
    positive_time_contribution: float
    negative_time_contribution: float
    legacy_clipped_result: float
    legacy_to_corrected_ratio: float


def signed_energy_from_response(intensity: np.ndarray, response: np.ndarray, dt: float, *, n_R: float) -> SignedEnergy:
    """Evaluate Eq. (10) and diagnostic legacy clipping on a sampled pulse."""
    intensity = np.asarray(intensity, dtype=float)
    response = np.asarray(response, dtype=float)
    dIdt = np.gradient(intensity, float(dt), edge_order=2)
    w_R = (n_R / C0) * response * dIdt
    u_R = float(np.sum(w_R) * dt)
    positive = float(np.sum(np.maximum(w_R, 0.0)) * dt)
    negative = float(np.sum(np.minimum(w_R, 0.0)) * dt)
    q = max(-u_R, 0.0)
    return SignedEnergy(
        u_R_signed=u_R,
        q_R_positive=q,
        positive_time_contribution=positive,
        negative_time_contribution=negative,
        legacy_clipped_result=positive,
        legacy_to_corrected_ratio=positive / q if q > 0.0 else float("inf"),
    )
