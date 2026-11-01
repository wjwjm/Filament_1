"""CPU-S4-6G Eq10 variability diagnostics.

This package observes scalar process and scheduler state around the frozen
S4-6F controller. It does not alter the propagation entry or its clocks.
"""

DIAGNOSTIC_TASK = "CPU-S4-6G-EQ10-VARIABILITY-20261101"
DIAGNOSTIC_SCHEMA = "filament.cpu_s46g.diagnostic_context.v1"

__all__ = ["DIAGNOSTIC_SCHEMA", "DIAGNOSTIC_TASK"]
