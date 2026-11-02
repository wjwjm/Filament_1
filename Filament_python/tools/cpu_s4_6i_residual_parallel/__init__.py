"""Private S4-6I residual pointwise-cost instrumentation layer."""

from .cost_factory import (
    COST_SCHEMA,
    CostFactoryError,
    ResidualCostHook,
    build_candidate,
    build_role,
    finalize_factory,
    make_cost_hook,
    metadata,
    source_identity,
    take_cost_snapshot,
    verify_bindings,
)
from .instrument import (
    INSTRUMENT_SCHEMA,
    ResidualInstrumentationError,
    instrument_heat_ast,
    instrument_residual_cost_ast,
    strip_residual_cost_ast,
)
from .registry import registry_snapshot, validate_registry

__all__ = [
    "COST_SCHEMA",
    "INSTRUMENT_SCHEMA",
    "CostFactoryError",
    "ResidualCostHook",
    "ResidualInstrumentationError",
    "build_candidate",
    "build_role",
    "finalize_factory",
    "instrument_heat_ast",
    "instrument_residual_cost_ast",
    "make_cost_hook",
    "metadata",
    "source_identity",
    "registry_snapshot",
    "strip_residual_cost_ast",
    "take_cost_snapshot",
    "validate_registry",
    "verify_bindings",
]
