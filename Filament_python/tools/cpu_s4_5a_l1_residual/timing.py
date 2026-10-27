"""Bounded scalar timing for the CPU-S4-5A L1 residual profile.

The recorder is deliberately small and source independent.  It observes an
already selected execution path; it does not wrap arbitrary callbacks, inspect
arrays, or change the numerical route.  A recorder is owned by the thread that
created it and refuses all other callers.
"""

from __future__ import annotations

import os
import sys
import threading
from collections import defaultdict
from contextlib import AbstractContextManager
from time import perf_counter_ns, process_time_ns
from typing import Any, NamedTuple


SCHEMA = "filament.cpu_s4_5a.scalar_timing.v1"
DEFAULT_MAX_SCOPES = 4096

SCOPE_LABELS = (
    "APPLICATION",
    "PROPAGATION",
    "INTERVAL",
    "LINEAR",
    "RAMAN_HALF",
    "RAMAN_STAGE",
    "RAMAN_INTENSITY",
    "RAMAN_COEFFICIENTS",
    "RAMAN_IIR",
    "RAMAN_ENERGY",
    "HEUN_ASSEMBLY",
    "FFT",
    "MATH_KERNELS",
    "NONLINEAR",
    "ION_SETUP",
    "ION_RECURSION",
    "NONLINEAR_ARRAYS",
    "DIAGNOSTICS",
    "OUTPUT",
    "INPUT_PREPARATION",
    "SOURCE_FACTORY",
    "SOURCE_VALIDATION",
    "L1_FACTORY",
    "MANAGEMENT",
)
_SCOPE_LABEL_SET = frozenset(SCOPE_LABELS)
_PHASES = frozenset(("none", "pre", "post"))
_STAGES = frozenset(("none", "stage1", "stage2"))


class TimingError(RuntimeError):
    """A timing scope or recorder contract was violated."""


class TimingConfigurationError(TimingError):
    """The process configuration cannot run this instrumentation."""


class TimingContext(NamedTuple):
    """Immutable scalar scope metadata; no runtime objects are retained."""

    interval: int
    phase: str = "none"
    stage: str = "none"


def context(interval: int, phase: str = "none", stage: str = "none") -> TimingContext:
    """Create validated scalar metadata for a timing scope."""

    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 0:
        raise TimingConfigurationError("TIMING_CONTEXT_INTERVAL_MUST_BE_NONNEGATIVE_INT")
    if phase not in _PHASES:
        raise TimingConfigurationError("TIMING_CONTEXT_PHASE_INVALID")
    if stage not in _STAGES:
        raise TimingConfigurationError("TIMING_CONTEXT_STAGE_INVALID")
    return TimingContext(int(interval), str(phase), str(stage))


_context_factory = context


class _Frame:
    __slots__ = (
        "token",
        "label",
        "context",
        "parent_token",
        "depth",
        "wall_start_ns",
        "cpu_start_ns",
        "child_wall_ns",
        "child_cpu_ns",
    )

    def __init__(
        self,
        token: int,
        label: str,
        metadata: TimingContext,
        parent_token: int | None,
        depth: int,
        wall_start_ns: int,
        cpu_start_ns: int,
    ) -> None:
        self.token = token
        self.label = label
        self.context = metadata
        self.parent_token = parent_token
        self.depth = depth
        self.wall_start_ns = wall_start_ns
        self.cpu_start_ns = cpu_start_ns
        self.child_wall_ns = 0
        self.child_cpu_ns = 0


class _Scope(AbstractContextManager["_Scope"]):
    __slots__ = ("_recorder", "_label", "_context", "_token", "_entered")

    def __init__(self, recorder: "TimingRecorder", label: str, metadata: TimingContext) -> None:
        self._recorder = recorder
        self._label = label
        self._context = metadata
        self._token: int | None = None
        self._entered = False

    def __enter__(self) -> "_Scope":
        if self._entered:
            raise TimingError("SCOPE_CONTEXT_REENTERED")
        self._token = self._recorder.begin(self._label, self._context)
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        if not self._entered or self._token is None:
            raise TimingError("SCOPE_CONTEXT_EXIT_WITHOUT_ENTER")
        token = self._token
        # Deliberately inspect only the exception presence.  Never retain the
        # exception, traceback, or any frame reachable through them.
        failed = exc_type is not None
        try:
            self._recorder.end(token, failed=failed)
        finally:
            self._entered = False
            self._token = None
        return False


class TimingRecorder:
    """Single-owner, bounded, nested scalar timing recorder."""

    __slots__ = (
        "enabled",
        "max_scopes",
        "owner_thread_id",
        "_stack",
        "_records",
        "_ended_tokens",
        "_started_scopes",
        "_next_token",
        "events_dropped",
        "_incomplete",
    )

    def __init__(self, enabled: bool = True, *, max_scopes: int = DEFAULT_MAX_SCOPES) -> None:
        if os.environ.get("PYTHONOPTIMIZE") or sys.flags.optimize:
            raise TimingConfigurationError("PYTHONOPTIMIZE_FORBIDDEN")
        if isinstance(max_scopes, bool) or not isinstance(max_scopes, int) or max_scopes <= 0:
            raise TimingConfigurationError("TIMING_MAX_SCOPES_INVALID")
        if max_scopes > DEFAULT_MAX_SCOPES:
            raise TimingConfigurationError("TIMING_MAX_SCOPES_EXCEEDS_FIXED_BOUND")
        self.enabled = bool(enabled)
        self.max_scopes = int(max_scopes)
        self.owner_thread_id = threading.get_ident()
        self._stack: list[_Frame] = []
        self._records: list[dict[str, Any]] = []
        self._ended_tokens: set[int] = set()
        self._started_scopes = 0
        self._next_token = 1
        self.events_dropped = 0
        self._incomplete = []

    def _check_owner(self) -> None:
        if threading.get_ident() != self.owner_thread_id:
            raise TimingError("TIMING_OWNER_THREAD_MISMATCH")

    @staticmethod
    def _label(label: str) -> str:
        if not isinstance(label, str) or label not in _SCOPE_LABEL_SET:
            raise TimingConfigurationError("TIMING_SCOPE_LABEL_INVALID")
        return label

    @staticmethod
    def _metadata(metadata: TimingContext | None) -> TimingContext:
        if metadata is None:
            return context(0)
        if not isinstance(metadata, TimingContext):
            raise TimingConfigurationError("TIMING_CONTEXT_OBJECT_INVALID")
        return context(metadata.interval, metadata.phase, metadata.stage)

    def context(self, interval: int, phase: str = "none", stage: str = "none") -> TimingContext:
        self._check_owner()
        return context(interval, phase, stage)

    def begin(
        self,
        label: str,
        metadata: TimingContext | None = None,
        *,
        context: TimingContext | None = None,
    ) -> int:
        """Begin one scalar scope and return its integer token."""

        self._check_owner()
        self._label(label)
        if metadata is not None and context is not None:
            raise TimingConfigurationError("TIMING_BEGIN_CONTEXT_DUPLICATED")
        if context is not None:
            metadata = context
        if not self.enabled:
            return 0
        if self._started_scopes >= self.max_scopes:
            # Refusal occurs before reading either clock.  Nothing is dropped.
            raise TimingError("TIMING_SCOPE_CAPACITY_EXCEEDED")
        checked_context = self._metadata(metadata)
        token = self._next_token
        self._next_token += 1
        self._started_scopes += 1
        wall_start = perf_counter_ns()
        cpu_start = process_time_ns()
        parent = self._stack[-1] if self._stack else None
        self._stack.append(
            _Frame(
                token,
                label,
                checked_context,
                None if parent is None else parent.token,
                len(self._stack),
                wall_start,
                cpu_start,
            )
        )
        return token

    def scope(
        self,
        label: str,
        metadata: TimingContext | None = None,
        *,
        context: TimingContext | None = None,
        interval: int | None = None,
        phase: str = "none",
        stage: str = "none",
    ) -> _Scope:
        """Return a context manager for one nested scalar scope."""

        self._check_owner()
        self._label(label)
        if metadata is not None and context is not None:
            raise TimingConfigurationError("TIMING_SCOPE_CONTEXT_DUPLICATED")
        if context is not None:
            metadata = context
        if metadata is not None and interval is not None:
            raise TimingConfigurationError("TIMING_SCOPE_CONTEXT_DUPLICATED")
        if interval is not None:
            metadata = _context_factory(interval, phase, stage)
        else:
            metadata = self._metadata(metadata)
        return _Scope(self, label, metadata)

    def end(self, token: int, *, failed: bool = False) -> dict[str, Any] | None:
        """End the top scope, calculating inclusive and exclusive clocks."""

        self._check_owner()
        if not self.enabled:
            if token != 0:
                raise TimingError("TIMING_DISABLED_TOKEN_INVALID")
            return None
        if not isinstance(token, int) or token <= 0:
            raise TimingError("TIMING_SCOPE_TOKEN_INVALID")
        if not self._stack:
            if token in self._ended_tokens:
                raise TimingError("TIMING_SCOPE_ALREADY_ENDED")
            raise TimingError("TIMING_SCOPE_TOKEN_UNKNOWN")
        frame = self._stack[-1]
        if frame.token != token:
            raise TimingError("TIMING_SCOPE_EXIT_MISMATCH")
        wall_end = perf_counter_ns()
        cpu_end = process_time_ns()
        wall_inclusive = wall_end - frame.wall_start_ns
        cpu_inclusive = cpu_end - frame.cpu_start_ns
        if wall_inclusive < 0 or cpu_inclusive < 0:
            raise TimingError("TIMING_CLOCK_WENT_BACKWARDS")
        wall_exclusive = wall_inclusive - frame.child_wall_ns
        cpu_exclusive = cpu_inclusive - frame.child_cpu_ns
        if wall_exclusive < 0 or cpu_exclusive < 0:
            raise TimingError("TIMING_NEGATIVE_EXCLUSIVE")
        self._stack.pop()
        self._ended_tokens.add(token)
        record = {
            "token": frame.token,
            "label": frame.label,
            "context": {
                "interval": frame.context.interval,
                "phase": frame.context.phase,
                "stage": frame.context.stage,
            },
            "parent_token": frame.parent_token,
            "depth": frame.depth,
            "start_wall_ns": frame.wall_start_ns,
            "end_wall_ns": wall_end,
            "start_cpu_ns": frame.cpu_start_ns,
            "end_cpu_ns": cpu_end,
            "wall_inclusive_ns": wall_inclusive,
            "wall_exclusive_ns": wall_exclusive,
            "cpu_inclusive_ns": cpu_inclusive,
            "cpu_exclusive_ns": cpu_exclusive,
            "direct_child_wall_inclusive_ns": frame.child_wall_ns,
            "direct_child_cpu_inclusive_ns": frame.child_cpu_ns,
            "failed": bool(failed),
            "status": "failed" if failed else "complete",
        }
        self._records.append(record)
        parent = self._stack[-1] if self._stack else None
        if parent is not None:
            parent.child_wall_ns += wall_inclusive
            parent.child_cpu_ns += cpu_inclusive
        return dict(record)

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-safe records and closure/grouped scalar summaries."""

        self._check_owner()
        if self.enabled and self._stack:
            raise TimingError("TIMING_INCOMPLETE_SCOPE_STACK")
        records = [dict(row, context=dict(row["context"])) for row in self._records]
        grouped: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "call_count": 0,
                "failed_count": 0,
                "wall_inclusive_ns": 0,
                "wall_exclusive_ns": 0,
                "cpu_inclusive_ns": 0,
                "cpu_exclusive_ns": 0,
            }
        )
        closures = []
        roots = []
        for row in records:
            group = grouped[row["label"]]
            group["call_count"] += 1
            group["failed_count"] += int(row["failed"])
            for name in ("wall_inclusive_ns", "wall_exclusive_ns", "cpu_inclusive_ns", "cpu_exclusive_ns"):
                group[name] += row[name]
            wall_error = row["wall_inclusive_ns"] - row["wall_exclusive_ns"] - row["direct_child_wall_inclusive_ns"]
            cpu_error = row["cpu_inclusive_ns"] - row["cpu_exclusive_ns"] - row["direct_child_cpu_inclusive_ns"]
            if wall_error < 0 or cpu_error < 0:
                raise TimingError("TIMING_NEGATIVE_CLOSURE")
            closures.append({"token": row["token"], "wall_error_ns": wall_error, "cpu_error_ns": cpu_error})
            if row["parent_token"] is None:
                roots.append(row["token"])
        root_wall = max((row["wall_inclusive_ns"] for row in records if row["parent_token"] is None), default=0)
        tolerance_ns = max(1000, int(root_wall * 1e-8))
        max_wall_error = max((row["wall_error_ns"] for row in closures), default=0)
        max_cpu_error = max((row["cpu_error_ns"] for row in closures), default=0)
        return {
            "schema": SCHEMA,
            "enabled": self.enabled,
            "owner_thread_id": self.owner_thread_id,
            "max_scopes": self.max_scopes,
            "started_scopes": self._started_scopes,
            "completed_scopes": len(records),
            "events_dropped": self.events_dropped,
            "incomplete_scopes": [dict(row) for row in self._incomplete],
            "records": records,
            "summary_by_label": {key: dict(value) for key, value in sorted(grouped.items())},
            "roots": roots,
            "clock_closure": {
                "root_wall_inclusive_ns": root_wall,
                "tolerance_ns": tolerance_ns,
                "max_wall_error_ns": max_wall_error,
                "max_cpu_error_ns": max_cpu_error,
                "pass": max_wall_error == 0 and max_cpu_error == 0,
            },
        }

    to_jsonable = snapshot

    def retire_incomplete(self, reason: str) -> None:
        """Retire scalar frames without clocks if a clock/closure fault prevents end."""
        self._check_owner()
        if not isinstance(reason,str):raise TimingError("INCOMPLETE_REASON_MUST_BE_SCALAR_TEXT")
        self._incomplete.extend(dict(token=f.token,label=f.label,parent_token=f.parent_token,
            interval=f.context.interval,phase=f.context.phase,stage=f.context.stage,
            start_wall_ns=f.wall_start_ns,start_cpu_ns=f.cpu_start_ns,status='incomplete',reason=reason) for f in self._stack)
        self._stack.clear()


def empty_scope_control_value(*, enabled: bool = True) -> dict[str, Any]:
    """Measure one empty scalar control interval; never a subtraction input."""

    if not enabled:
        return {"enabled": False, "wall_ns": 0, "cpu_ns": 0, "subtract_from_scopes": False}
    wall_start = perf_counter_ns()
    cpu_start = process_time_ns()
    recorder = TimingRecorder(enabled=True)
    with recorder.scope("MANAGEMENT"):
        pass
    wall_end = perf_counter_ns()
    cpu_end = process_time_ns()
    wall_ns = wall_end - wall_start
    cpu_ns = cpu_end - cpu_start
    if wall_ns < 0 or cpu_ns < 0:
        raise TimingError("TIMING_EMPTY_CONTROL_CLOCK_WENT_BACKWARDS")
    empty = recorder.snapshot()["records"][0]
    return {"enabled": True, "wall_ns": wall_ns, "cpu_ns": cpu_ns,
            "empty_scope_wall_ns": empty["wall_inclusive_ns"],
            "empty_scope_cpu_ns": empty["cpu_inclusive_ns"],
            "definition": "One actual empty recorder context, including construction and scope bookkeeping in outer control",
            "subtract_from_scopes": False}


__all__ = [
    "DEFAULT_MAX_SCOPES",
    "SCHEMA",
    "SCOPE_LABELS",
    "TimingConfigurationError",
    "TimingContext",
    "TimingError",
    "TimingRecorder",
    "context",
    "empty_scope_control_value",
]
