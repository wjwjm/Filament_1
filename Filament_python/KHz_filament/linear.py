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
    precision_strategy="baseline_complex64",
    return_energy_diagnostics=False,
    energy_scale=None,
):
    """Brabec–Krausz NEE linear step (factorized over frequency slices).

    Uses the linear operator in frequency domain:
      dA/dz = i [ -k_perp^2/(2 k0 (1+Omega/omega0)) + (beta2/2) Omega^2 ] A
    and applies exp(i * phase * dz) per Omega slice.
    """
    allowed_strategies = ("baseline_complex64", "orthonormal_fft", "mixed_precision", "unitary_projection")
    strategy = str(precision_strategy or "baseline_complex64").lower()
    if strategy not in allowed_strategies:
        raise ValueError(f"unknown BK-NEE precision strategy {strategy!r}; allowed: {allowed_strategies}")

    output_ctype, output_rdtype = _complex_real_dtypes(E.dtype)
    work_ctype = xp.complex128 if strategy == "mixed_precision" else output_ctype
    _, work_rdtype = _complex_real_dtypes(work_ctype)
    onej = xp.array(1j, dtype=work_ctype)

    Omega = xp.asarray(Omega, dtype=work_rdtype)
    kperp2 = xp.asarray(kperp2, dtype=work_rdtype)

    # FFT_t first, then per-slice FFT2_xy to keep memory usage lower than full 3D operator.
    if return_energy_diagnostics and energy_scale is None:
        raise ValueError("BK-NEE energy diagnostics require energy_scale")

    def _norm2(value):
        return xp.sum(xp.abs(value) ** 2, dtype=xp.float64)

    needs_projection_norm = strategy == "unitary_projection"
    input_norm2 = _norm2(E) if (return_energy_diagnostics or needs_projection_norm) else None
    work_input = E.astype(work_ctype, copy=False)
    input_cast_norm2 = _norm2(work_input) if return_energy_diagnostics else None
    fft_kwargs = {"norm": "ortho"} if strategy == "orthonormal_fft" else {}
    Ew = xp.fft.fft(work_input, axis=0, **fft_kwargs)  # [Nt, Ny, Nx]
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
    forward_factor = 1 if strategy == "orthonormal_fft" else Nt
    transfer_factor = 1 if strategy == "orthonormal_fft" else Nt * nxy
    transfer_norm2 = xp.asarray(0.0, dtype=xp.float64) if return_energy_diagnostics else None
    for i in range(Nt):
        phase_xy = coeff_diff[i] * kperp2 + coeff_gvd[i]
        prop2d = xp.exp(onej * phase_xy * float(dz)).astype(work_ctype, copy=False)

        S = xp.fft.fft2(Ew[i], axes=(-2, -1), **fft_kwargs)
        S *= prop2d
        if return_energy_diagnostics:
            transfer_norm2 += _norm2(S)
        Ew[i] = xp.fft.ifft2(S, axes=(-2, -1), **fft_kwargs)

    inverse_spatial_norm2 = _norm2(Ew) if return_energy_diagnostics else None
    internal_out = xp.fft.ifft(Ew, axis=0, **fft_kwargs)
    internal_norm2 = _norm2(internal_out) if return_energy_diagnostics else None
    candidate_out = internal_out.astype(output_ctype, copy=False)
    output_cast_norm2 = _norm2(candidate_out) if (return_energy_diagnostics or needs_projection_norm) else None
    projection_scale = 1.0
    if strategy == "unitary_projection":
        candidate_energy = float(output_cast_norm2)
        if not candidate_energy > 0.0:
            raise FloatingPointError("cannot apply BK-NEE unitary projection to a non-positive field norm")
        projection_scale = (float(input_norm2) / candidate_energy) ** 0.5
        out = (candidate_out * xp.asarray(projection_scale, dtype=output_rdtype)).astype(output_ctype, copy=False)
    else:
        out = candidate_out
    if not return_energy_diagnostics:
        return out
    inverse_time_norm2 = _norm2(out)
    scale = float(energy_scale)
    return out, {
        "energy_before_J": float(scale * input_norm2),
        "energy_after_input_cast_J": float(scale * input_cast_norm2),
        "energy_after_forward_fft_J": float(scale * forward_norm2 / forward_factor),
        "energy_after_transfer_J": float(scale * transfer_norm2 / transfer_factor),
        "energy_after_inverse_fft_J": float(scale * inverse_spatial_norm2 / forward_factor),
        "energy_after_internal_linear_J": float(scale * internal_norm2),
        "energy_after_output_cast_J": float(scale * output_cast_norm2),
        "energy_after_J": float(scale * inverse_time_norm2),
        "output_cast_field_delta_J": float(scale * (output_cast_norm2 - internal_norm2)),
        "unitary_projection_scale": float(projection_scale),
        "unitary_projection_scale_deviation": float(abs(projection_scale - 1.0)),
        "explicit_boundary_loss_J": 0.0,
        "explicit_spectral_filter_loss_J": 0.0,
        "explicit_crop_loss_J": 0.0,
        "explicit_evanescent_loss_J": 0.0,
        "explicit_other_loss_J": 0.0,
    }
