from __future__ import annotations

from .device import xp


def _complex_real_dtypes(ctype):
    """Return matching real dtype for a complex dtype."""
    rdtype = xp.float32 if ctype == xp.complex64 else xp.float64
    return ctype, rdtype


def lin_propagator(kperp2, k0, dz, *, ctype=None):
    """Paraxial angular-spectrum propagator exp(i * (-k_perp^2) dz / (2k0))."""
    if ctype is None:
        ctype = xp.complex64
    ctype, rdtype = _complex_real_dtypes(ctype)

    onej = xp.array(1j, dtype=ctype)
    k2 = xp.asarray(kperp2, dtype=rdtype)
    phase = (-k2) * (dz / (2.0 * float(k0)))
    return xp.exp(onej * phase).astype(ctype)


def step_linear(E, prop):
    """Apply a 2D (x,y) FFT-based linear propagation to [Nt, Ny, Nx]."""
    if prop.dtype != E.dtype:
        prop = prop.astype(E.dtype, copy=False)
    Ew = xp.fft.fft2(E, axes=(-2, -1))
    Ew *= prop
    return xp.fft.ifft2(Ew, axes=(-2, -1))


def step_linear_bk_nee_factorized(
    E,
    *,
    Omega,
    kperp2,
    k0,
    omega0,
    dz,
    beta2=0.0,
    denom_floor=1e-4,
    return_energy_diagnostics=False,
    energy_scale=None,
):
    """Brabec–Krausz NEE linear step (factorized over frequency slices).

    Uses the linear operator in frequency domain:
      dA/dz = i [ -k_perp^2/(2 k0 (1+Omega/omega0)) + (beta2/2) Omega^2 ] A
    and applies exp(i * phase * dz) per Omega slice.
    """
    ctype, rdtype = _complex_real_dtypes(E.dtype)
    onej = xp.array(1j, dtype=ctype)

    Omega = xp.asarray(Omega, dtype=rdtype)
    kperp2 = xp.asarray(kperp2, dtype=rdtype)

    # FFT_t first, then per-slice FFT2_xy to keep memory usage lower than full 3D operator.
    if return_energy_diagnostics and energy_scale is None:
        raise ValueError("BK-NEE energy diagnostics require energy_scale")

    def _norm2(value):
        return xp.sum(xp.abs(value) ** 2, dtype=xp.float64)

    input_norm2 = _norm2(E) if return_energy_diagnostics else None
    Ew = xp.fft.fft(E, axis=0)  # [Nt, Ny, Nx]
    forward_norm2 = _norm2(Ew) if return_energy_diagnostics else None

    rel = Omega / float(omega0)
    denom = 1.0 + rel
    # Avoid singularity near Omega ~= -omega0.
    denom_abs = xp.maximum(xp.abs(denom), float(denom_floor))
    denom_sign = xp.where(denom >= 0.0, 1.0, -1.0)
    denom = denom_sign * denom_abs

    coeff_diff = -1.0 / (2.0 * float(k0) * denom)          # [Nt]
    coeff_gvd = 0.5 * float(beta2) * (Omega ** 2)          # [Nt]

    Nt = Ew.shape[0]
    nxy = int(Ew.shape[-2] * Ew.shape[-1])
    transfer_norm2 = xp.asarray(0.0, dtype=xp.float64) if return_energy_diagnostics else None
    for i in range(Nt):
        phase_xy = coeff_diff[i] * kperp2 + coeff_gvd[i]
        prop2d = xp.exp(onej * phase_xy * float(dz)).astype(ctype, copy=False)

        S = xp.fft.fft2(Ew[i], axes=(-2, -1))
        S *= prop2d
        if return_energy_diagnostics:
            transfer_norm2 += _norm2(S)
        Ew[i] = xp.fft.ifft2(S, axes=(-2, -1))

    inverse_spatial_norm2 = _norm2(Ew) if return_energy_diagnostics else None
    out = xp.fft.ifft(Ew, axis=0).astype(ctype, copy=False)
    if not return_energy_diagnostics:
        return out
    inverse_time_norm2 = _norm2(out)
    scale = float(energy_scale)
    return out, {
        "energy_before_J": float(scale * input_norm2),
        "energy_after_forward_fft_J": float(scale * forward_norm2 / Nt),
        "energy_after_transfer_J": float(scale * transfer_norm2 / (Nt * nxy)),
        "energy_after_inverse_fft_J": float(scale * inverse_spatial_norm2 / Nt),
        "energy_after_J": float(scale * inverse_time_norm2),
        "explicit_boundary_loss_J": 0.0,
        "explicit_spectral_filter_loss_J": 0.0,
        "explicit_crop_loss_J": 0.0,
        "explicit_evanescent_loss_J": 0.0,
        "explicit_other_loss_J": 0.0,
    }
