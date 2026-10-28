"""Private Eq10 pointwise kernels for CPU-S4-5C.

The package deliberately exposes only the two materialization operations used
by the candidate. FFT/IFFT, reductions, and all scientific stage control stay
in the source-bound adapter.
"""

from .kernels import (
    CANDIDATE_IDENTITY,
    KernelExecutor,
    KernelUnsupported,
    source_identity,
)

__all__ = [
    "CANDIDATE_IDENTITY",
    "KernelExecutor",
    "KernelUnsupported",
    "source_identity",
]
