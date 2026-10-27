"""Private CPU-S4-5B steady propagation attribution namespace."""

from .adapter import (
    DOMAIN,
    MODES,
    PURE_OFF_SMALLGRID,
    QUALIFY_BOUNDARY,
    ROLE_OFF,
    ROLE_PROFILE,
    ROLES,
    RecorderPair,
    make_recorder_pair,
)
from .candidate import build_role, make_recorder, source_identity
from .timing import (
    DEFAULT_MAX_RECORDS,
    MAX_METADATA_BYTES,
    NEW_A8_LABELS,
    NEW_B8_LABELS,
    OLD_SCOPE_LABELS,
    SCOPE_LABELS,
    TimingConfigurationError,
    TimingContext,
    TimingError,
    TimingRecorder,
    context,
)

__all__ = [
    "DEFAULT_MAX_RECORDS",
    "DOMAIN",
    "MAX_METADATA_BYTES",
    "MODES",
    "NEW_A8_LABELS",
    "NEW_B8_LABELS",
    "OLD_SCOPE_LABELS",
    "PURE_OFF_SMALLGRID",
    "QUALIFY_BOUNDARY",
    "ROLE_OFF",
    "ROLE_PROFILE",
    "ROLES",
    "RecorderPair",
    "SCOPE_LABELS",
    "TimingConfigurationError",
    "TimingContext",
    "TimingError",
    "TimingRecorder",
    "build_role",
    "context",
    "make_recorder",
    "make_recorder_pair",
    "source_identity",
]
