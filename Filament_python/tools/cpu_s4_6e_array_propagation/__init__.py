"""Private A/P24 propagation candidate.

The package is deliberately separate from the production optical modules and
from the earlier S4-5B timing candidate. It owns only the source-bound stage
clone, the bounded pointwise worker pool, and the L1 role adapter.
"""

from .candidate import (
    CANDIDATE_SCHEMA,
    DEFAULT_VARIANT,
    CandidateUnsupported,
    SourceBindingError,
    build_role,
    source_identity,
)

__all__ = [
    "CANDIDATE_SCHEMA",
    "DEFAULT_VARIANT",
    "CandidateUnsupported",
    "SourceBindingError",
    "build_role",
    "source_identity",
]
