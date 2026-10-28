"""Bounded, non-scientific records and contract checks for CPU-S4-6A.

The package deliberately contains no propagation, backend, or numerical
implementation.  It validates the lightweight evidence records used by the
matched CPU/GPU optical timing task.
"""

from .records import (
    CASE_ORDER,
    BUDGET_CONTRACTS,
    BudgetError,
    ClockError,
    RecordError,
    TimingGateError,
    budget_check,
    expected_case_order,
    load_json_strict,
    resolve_record_path,
    timing_gate,
    validate_case_order,
    validate_clock_records,
    validate_source_records,
)

__all__ = [
    "CASE_ORDER",
    "BUDGET_CONTRACTS",
    "BudgetError",
    "ClockError",
    "RecordError",
    "TimingGateError",
    "budget_check",
    "expected_case_order",
    "load_json_strict",
    "resolve_record_path",
    "timing_gate",
    "validate_case_order",
    "validate_clock_records",
    "validate_source_records",
]
