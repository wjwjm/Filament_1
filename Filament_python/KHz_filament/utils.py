from __future__ import annotations
from .device import xp
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
