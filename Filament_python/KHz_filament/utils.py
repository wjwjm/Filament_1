from __future__ import annotations
from .device import xp


def _profile_parameters(profile, fallback_w0):
    """Return canonical runtime parameters for a transverse profile."""
    if profile is None:
        return "gaussian", float(fallback_w0), {}
    if not isinstance(profile, dict):
        raise ValueError("transverse_profile must be a mapping or None")
    profile_type = str(profile.get("type", "gaussian")).strip().lower()
    radius = float(profile.get("radius_m", fallback_w0))
    if radius <= 0.0:
        raise ValueError("transverse profile radius_m must be positive")
    return profile_type, radius, profile


def transverse_intensity_profile(x, y, profile=None, fallback_w0=None):
    """Build a dimensionless transverse intensity profile ``gI(x, y)``.

    ``gI`` is normalized to one at the physical beam center.  The returned
    quantity is intensity-level, so callers must use ``sqrt(gI)`` to build an
    electric-field amplitude.
    """
    if fallback_w0 is None:
        raise ValueError("fallback_w0 is required")
    profile_type, radius, options = _profile_parameters(profile, fallback_w0)
    X, Y = xp.meshgrid(x, y, indexing="xy")
    r2 = X ** 2 + Y ** 2

    if profile_type == "gaussian":
        gI = xp.exp(-2.0 * r2 / (radius ** 2))
    elif profile_type == "flat_top_cosine":
        edge_start = float(options.get("edge_start_fraction", 0.9))
        if not 0.0 < edge_start < 1.0:
            raise ValueError("flat_top_cosine edge_start_fraction must satisfy 0 < value < 1")
        r = xp.sqrt(r2)
        taper = 0.5 * (1.0 + xp.cos(xp.pi * (r - edge_start * radius) / ((1.0 - edge_start) * radius)))
        gI = xp.where(r <= edge_start * radius, 1.0, xp.where(r < radius, taper, 0.0))
    elif profile_type == "super_gaussian":
        order = float(options.get("order", 4.0))
        if order <= 0.0:
            raise ValueError("super_gaussian order must be positive")
        gI = xp.exp(-2.0 * (r2 / (radius ** 2)) ** order)
    else:
        raise ValueError(f"unsupported transverse profile type: {profile_type!r}")
    return xp.clip(gI, 0.0, 1.0)


#构造入射场
def gaussian_beam_xy(x, y, w0):
    """
    Return a [Ny, Nx] transverse Gaussian (1/e field radius = w0).
    """
    X, Y = xp.meshgrid(x, y, indexing='xy')
    R2 = X**2 + Y**2
    return xp.exp(-R2 / (w0**2))

def gaussian_pulse_t(t, tau_fwhm):
    """
    Return a [Nt, 1, 1] temporal Gaussian (field-level) with FWHM = tau_fwhm.
    """
    import math
    tau = tau_fwhm / math.sqrt(2.0 * math.log(2.0))  # field sigma
    return xp.exp(-(t[:, None, None]**2) / (tau**2))
