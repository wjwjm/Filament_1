"""Top-level package for KHz_filament.

This module intentionally avoids importing heavy runtime dependencies at import-time
(e.g. NumPy/CuPy) so that package discovery and lightweight sanity checks can run
in constrained environments.
"""

from __future__ import annotations

__all__ = [
    "constants",
    "device",
    "config",
    "grids",
    "linear",
    "ionization",
    "nonlinear",
    "heat",
    "propagate",
    "diagnostics",
    "utils",
    "air_dispersion",
    "linear_full",
    "raman",
]

# Expose module names without eagerly importing submodules.
# This keeps `import KHz_filament` lightweight and avoids failing
# when optional/heavy dependencies are not yet installed.
for _name in __all__:
    globals()[_name] = _name

del _name
