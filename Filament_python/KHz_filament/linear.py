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
    Ew = xp.fft.fft(E, axis=0)  # [Nt, Ny, Nx]

    rel = Omega / float(omega0)
    denom = 1.0 + rel
    # Avoid singularity near Omega ~= -omega0.
    denom_abs = xp.maximum(xp.abs(denom), float(denom_floor))
    denom_sign = xp.where(denom >= 0.0, 1.0, -1.0)
    denom = denom_sign * denom_abs

    coeff_diff = -1.0 / (2.0 * float(k0) * denom)          # [Nt]
    coeff_gvd = 0.5 * float(beta2) * (Omega ** 2)          # [Nt]

    Nt = Ew.shape[0]
    for i in range(Nt):
        phase_xy = coeff_diff[i] * kperp2 + coeff_gvd[i]
        prop2d = xp.exp(onej * phase_xy * float(dz)).astype(ctype, copy=False)

        S = xp.fft.fft2(Ew[i], axes=(-2, -1))
        S *= prop2d
        Ew[i] = xp.fft.ifft2(S, axes=(-2, -1))

    return xp.fft.ifft(Ew, axis=0).astype(ctype, copy=False)
