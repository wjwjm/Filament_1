"""G3 small-grid lifecycle candidate namespace."""

from .adapter import (
    ADAPTER_SCHEMA,
    BOUNDARY_CONTRACT,
    LINEAR_RETURN_FORMS,
    build_role,
)
from .candidate import (
    CANDIDATE_SCHEMA,
    DEFAULT_VARIANT,
    LOCAL_ENGINEERING_CONTRACT,
    POOL_CONTRACT,
    SMALLGRID_MODES,
    SMALLGRID_ROLES,
    CandidateUnsupported,
    HandoffRejected,
    SourceBindingError,
    StageBindingError,
    make_factory,
    source_identity,
)
from .controlled_callsite import ControlledLifetimeError, ControlledLifetimeIdentity

__all__ = [
    "CANDIDATE_SCHEMA",
    "DEFAULT_VARIANT",
    "POOL_CONTRACT",
    "LOCAL_ENGINEERING_CONTRACT",
    "ADAPTER_SCHEMA",
    "BOUNDARY_CONTRACT",
    "LINEAR_RETURN_FORMS",
    "SMALLGRID_MODES",
    "SMALLGRID_ROLES",
    "CandidateUnsupported",
    "HandoffRejected",
    "SourceBindingError",
    "StageBindingError",
    "ControlledLifetimeError",
    "ControlledLifetimeIdentity",
    "build_role",
    "make_factory",
    "source_identity",
]
