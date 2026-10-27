"""Private scalar timing recorder for the CPU-S4-5B measurement seam.

This module is deliberately independent of the accepted S4-5A recorder.  The
S4-5A role still receives its exact recorder type through :mod:`adapter`; B's
own recorder is used for the new A8/B8 observations and for the common shells
that must be present in both OFF and PROFILE.

Only immutable scalar metadata is retained.  A recorder is single-owner,
bounded, and fail-closed: capacity exhaustion, an unclosed scope, a repeated
end, an invalid context, or an owner-thread violation raises instead of
silently dropping an event.
"""

from __future__ import annotations

import os
import sys
import threading
from collections import defaultdict
from contextlib import AbstractContextManager
from time import perf_counter_ns, process_time_ns
from typing import Any, NamedTuple


SCHEMA = "filament.cpu_s4_5b.scalar_timing.v1"
DEFAULT_MAX_RECORDS = 8192
MAX_METADATA_BYTES = 64 * 1024 * 1024
# The reservation is intentionally conservative and fixed.  It bounds only
# recorder metadata, including a later merged-tree row for each raw event;
# clocks and scientific arrays are never retained here.
RESERVED_BYTES_PER_RECORD = MAX_METADATA_BYTES // (DEFAULT_MAX_RECORDS * 2)
MAX_SOURCE_SITE_LENGTH = 160

OLD_SCOPE_LABELS = (
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

NEW_A8_LABELS = (
    "PROP_ENTRY_INIT",
    "INTERVAL_BODY",
    "INTERVAL_PREP_AND_OTHER_ARRAYS",
    "INTERVAL_ENERGY_BOOKKEEPING",
    "INTERVAL_DEPOSITION_AND_QACC",
    "INTERVAL_POST_COMMIT",
    "INTERVAL_SAVED_DIAGNOSTICS",
    "PROP_FINALIZE_EXIT",
)

NEW_B8_LABELS = (
    "EQ10_SCALE_MAX",
    "EQ10_DERIVATIVE_EXPR",
    "EQ10_FFT",
    "EQ10_IFFT",
    "EQ10_NORMALIZE_SCALE",
    "EQ10_PRODUCT_REDUCE_SCALE",
    "EQ10_SUM",
    "EQ10_MATERIALIZE",
)

SCOPE_LABELS = OLD_SCOPE_LABELS + NEW_A8_LABELS + NEW_B8_LABELS
_SCOPE_LABEL_SET = frozenset(SCOPE_LABELS)
_COMMON_LABELS = frozenset(
    {"APPLICATION", "SOURCE_VALIDATION", "L1_FACTORY", "INTERVAL_BODY"}
)
OCCURRENCE_CLASSES = (
    "WINDOW_ONCE",
    "FIRST_INTERVAL_SETUP",
    "PER_INTERVAL",
    "SAVE_BRANCH",
    "WINDOW_FINALIZE",
)
_OCCURRENCE_SET = frozenset(OCCURRENCE_CLASSES)
_PHASES = frozenset(("none", "pre", "post"))
_STAGES = frozenset(("none", "stage1", "stage2"))
_MARKER_KINDS = frozenset(("ZERO_INTENSITY",))


class TimingError(RuntimeError):
    """A recorder lifecycle, capacity, or closure contract failed."""


class TimingConfigurationError(TimingError):
    """The process configuration cannot run this instrumentation."""


class TimingContext(NamedTuple):
    """Scalar context for one scope.

    ``interval`` and ``ordinal`` are the original ``enumerate`` index and are
    therefore zero based.  ``ordinal_1based`` is included explicitly so a
    report cannot accidentally confuse the two conventions.  A window-level
    scope uses zero for the index and its occurrence class identifies that it
    is not an interval event.
    """

    interval: int
    ordinal: int
    ordinal_1based: int
    attempt: int
    phase: str = "none"
    stage: str = "none"
    occurrence_class: str = "PER_INTERVAL"
    source_site: str = ""


def _source_site(value: str | None) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TimingConfigurationError("TIMING_SOURCE_SITE_MUST_BE_SCALAR_TEXT")
    if len(value) > MAX_SOURCE_SITE_LENGTH:
        raise TimingConfigurationError("TIMING_SOURCE_SITE_TOO_LONG")
    return value


def context(
    interval: int = 0,
    phase: str = "none",
    stage: str = "none",
    *,
    ordinal: int | None = None,
    attempt: int = 1,
    occurrence_class: str = "PER_INTERVAL",
    source_site: str = "",
) -> TimingContext:
    """Create validated scalar timing metadata.

    ``interval`` is the original zero-based index.  Callers may supply
    ``ordinal`` only to make the mapping explicit; it must remain zero based.
    """

    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 0:
        raise TimingConfigurationError("TIMING_CONTEXT_INTERVAL_MUST_BE_NONNEGATIVE_INT")
    if ordinal is None:
        ordinal = int(interval)
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise TimingConfigurationError("TIMING_CONTEXT_ORDINAL_MUST_BE_NONNEGATIVE_INT")
    if int(ordinal) != int(interval):
        raise TimingConfigurationError("TIMING_CONTEXT_INTERVAL_ORDINAL_MISMATCH")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt != 1:
        raise TimingConfigurationError("TIMING_CONTEXT_ATTEMPT_MUST_BE_ONE")
    if phase not in _PHASES:
        raise TimingConfigurationError("TIMING_CONTEXT_PHASE_INVALID")
    if stage not in _STAGES:
        raise TimingConfigurationError("TIMING_CONTEXT_STAGE_INVALID")
    if occurrence_class not in _OCCURRENCE_SET:
        raise TimingConfigurationError("TIMING_CONTEXT_OCCURRENCE_CLASS_INVALID")
    return TimingContext(
        int(interval),
        int(ordinal),
        int(ordinal) + 1,
        int(attempt),
        str(phase),
        str(stage),
        str(occurrence_class),
        _source_site(source_site),
    )


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
    __slots__ = ("_recorder", "_label", "_context", "_common", "_token", "_entered")

    def __init__(
        self,
        recorder: "TimingRecorder",
        label: str,
        metadata: TimingContext,
        common: bool,
    ) -> None:
        self._recorder = recorder
        self._label = label
        self._context = metadata
        self._common = bool(common)
        self._token: int | None = None
        self._entered = False

    def __enter__(self) -> "_Scope":
        if self._entered:
            raise TimingError("SCOPE_CONTEXT_REENTERED")
        self._token = self._recorder.begin(
            self._label, self._context, common=self._common
        )
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        if not self._entered or self._token is None:
            raise TimingError("SCOPE_CONTEXT_EXIT_WITHOUT_ENTER")
        token = self._token
        failed = exc_type is not None
        try:
            self._recorder.end(token, failed=failed)
        finally:
            self._entered = False
            self._token = None
        return False


class TimingRecorder:
    """Bounded, single-owner, nested scalar timing recorder.

    ``enabled`` controls the new fine-grained PROFILE scopes.  Common shells
    are enabled in both OFF and PROFILE when their label is one of
    :data:`_COMMON_LABELS`; callers may also pass ``common=True`` explicitly.
    """

    __slots__ = (
        "enabled",
        "common_shells_enabled",
        "max_records",
        "owner_thread_id",
        "_stack",
        "_records",
        "_markers",
        "_ended_tokens",
        "_started_scopes",
        "_event_count",
        "_next_token",
        "events_dropped",
        "_incomplete",
        "_companion_recorder",
        "_companion_profile",
        "_companion_identity",
    )

    def __init__(
        self,
        enabled: bool = True,
        *,
        max_records: int = DEFAULT_MAX_RECORDS,
        common_shells_enabled: bool = True,
    ) -> None:
        if os.environ.get("PYTHONOPTIMIZE") or sys.flags.optimize:
            raise TimingConfigurationError("PYTHONOPTIMIZE_FORBIDDEN")
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records <= 0:
            raise TimingConfigurationError("TIMING_MAX_RECORDS_INVALID")
        if max_records > DEFAULT_MAX_RECORDS:
            raise TimingConfigurationError("TIMING_MAX_RECORDS_EXCEEDS_FIXED_BOUND")
        self.enabled = bool(enabled)
        self.common_shells_enabled = bool(common_shells_enabled)
        self.max_records = int(max_records)
        self.owner_thread_id = threading.get_ident()
        self._stack: list[_Frame] = []
        self._records: list[dict[str, Any]] = []
        self._markers: list[dict[str, Any]] = []
        self._ended_tokens: set[int] = set()
        self._started_scopes = 0
        self._event_count = 0
        self._next_token = 1
        self.events_dropped = 0
        self._incomplete: list[dict[str, Any]] = []
        self._companion_recorder: Any = None
        self._companion_profile: bool | None = None
        self._companion_identity: str | None = None

    @property
    def fine_enabled(self) -> bool:
        return bool(self.enabled)

    @property
    def metadata_reserved_bytes(self) -> int:
        return int(self._event_count * RESERVED_BYTES_PER_RECORD)

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
            return context(0, occurrence_class="WINDOW_ONCE")
        if not isinstance(metadata, TimingContext):
            raise TimingConfigurationError("TIMING_CONTEXT_OBJECT_INVALID")
        return context(
            metadata.interval,
            metadata.phase,
            metadata.stage,
            ordinal=metadata.ordinal,
            attempt=metadata.attempt,
            occurrence_class=metadata.occurrence_class,
            source_site=metadata.source_site,
        )

    def context(
        self,
        interval: int = 0,
        phase: str = "none",
        stage: str = "none",
        *,
        ordinal: int | None = None,
        attempt: int = 1,
        occurrence_class: str = "PER_INTERVAL",
        source_site: str = "",
    ) -> TimingContext:
        self._check_owner()
        return context(
            interval,
            phase,
            stage,
            ordinal=ordinal,
            attempt=attempt,
            occurrence_class=occurrence_class,
            source_site=source_site,
        )

    def bind_companion(
        self,
        companion: Any,
        *,
        profile: bool,
        identity: str,
    ) -> None:
        """Bind the exact S4-5A scalar recorder used as the phase bridge.

        The bridge accepts only the frozen recorder class and scalar identity;
        it never accepts a callback, scientific array, or arbitrary context
        provider.  The companion is retained only as the private pair member
        needed to read its active scalar frame.
        """

        self._check_owner()
        if not isinstance(profile, bool) or profile is not self.enabled:
            raise TimingConfigurationError("TIMING_COMPANION_PROFILE_MISMATCH")
        if not isinstance(identity, str) or not identity or len(identity) > 160:
            raise TimingConfigurationError("TIMING_COMPANION_IDENTITY_INVALID")
        module_name = "tools.cpu_s4_5a_l1_residual.timing"
        try:
            module = __import__(module_name, fromlist=["TimingRecorder"])
            expected = getattr(module, "TimingRecorder", None)
        except Exception as exc:  # pragma: no cover - import failure is explicit
            raise TimingConfigurationError("TIMING_COMPANION_MODULE_MISSING") from exc
        if expected is None or type(companion) is not expected:
            raise TimingConfigurationError("TIMING_COMPANION_EXACT_TYPE_REQUIRED")
        if getattr(companion, "owner_thread_id", None) != self.owner_thread_id:
            raise TimingConfigurationError("TIMING_COMPANION_OWNER_MISMATCH")
        if getattr(companion, "enabled", object()) is not profile:
            raise TimingConfigurationError("TIMING_COMPANION_PROFILE_SWITCH_MISMATCH")
        self._companion_recorder = companion
        self._companion_profile = bool(profile)
        self._companion_identity = identity

    def bridged_context(
        self,
        *,
        require_companion: bool,
        source_site: str = "",
    ) -> TimingContext:
        """Return B's interval frame with A's current scalar phase/stage.

        This is intentionally a fixed scalar bridge.  In PROFILE, Eq10 calls
        must be enclosed by the active exact S4-5A frame; no scientific state,
        array, callback, or arbitrary context source is accepted.
        """

        self._check_owner()
        current = self.current_context()
        companion = self._companion_recorder
        if companion is None:
            if require_companion and self.enabled:
                raise TimingConfigurationError("TIMING_COMPANION_BRIDGE_MISSING")
            return self._metadata(
                context(
                    current.interval,
                    current.phase,
                    current.stage,
                    ordinal=current.ordinal,
                    attempt=current.attempt,
                    occurrence_class=current.occurrence_class,
                    source_site=source_site or current.source_site,
                )
            )
        if getattr(companion, "owner_thread_id", None) != self.owner_thread_id:
            raise TimingConfigurationError("TIMING_COMPANION_OWNER_MISMATCH")
        if getattr(companion, "enabled", object()) is not self.enabled:
            raise TimingConfigurationError("TIMING_COMPANION_PROFILE_SWITCH_MISMATCH")
        stack = getattr(companion, "_stack", None)
        if self.enabled and require_companion and not stack:
            raise TimingConfigurationError("TIMING_COMPANION_SCALAR_FRAME_MISSING")
        if stack:
            frame = stack[-1]
            a_context = getattr(frame, "context", None)
            phase = getattr(a_context, "phase", "none")
            stage = getattr(a_context, "stage", "none")
        else:
            phase = "none"
            stage = "none"
        return context(
            current.interval,
            str(phase),
            str(stage),
            ordinal=current.ordinal,
            attempt=current.attempt,
            occurrence_class=current.occurrence_class,
            source_site=source_site or current.source_site,
        )

    def current_context(self) -> TimingContext:
        """Return the active scalar context, or a window-level default."""

        self._check_owner()
        if self._stack:
            frame = self._stack[-1]
            return self._metadata(frame.context)
        return context(0, occurrence_class="WINDOW_ONCE")

    def _is_enabled(self, label: str, common: bool) -> bool:
        return bool(self.enabled or common or (self.common_shells_enabled and label in _COMMON_LABELS))

    def _reserve_event(self) -> None:
        if self._event_count >= self.max_records:
            # Refusal happens before a clock read.  No event is dropped.
            raise TimingError("TIMING_RECORD_CAPACITY_EXCEEDED")
        if (self._event_count + 1) * RESERVED_BYTES_PER_RECORD > MAX_METADATA_BYTES:
            raise TimingError("TIMING_METADATA_BUDGET_EXCEEDED")
        self._event_count += 1

    def begin(
        self,
        label: str,
        metadata: TimingContext | None = None,
        *,
        context: TimingContext | None = None,
        common: bool = False,
    ) -> int:
        """Begin one scope and return an integer token."""

        self._check_owner()
        self._label(label)
        if metadata is not None and context is not None:
            raise TimingConfigurationError("TIMING_BEGIN_CONTEXT_DUPLICATED")
        if context is not None:
            metadata = context
        if not self._is_enabled(label, common):
            return 0
        checked_context = self._metadata(metadata)
        self._reserve_event()
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
        ordinal: int | None = None,
        attempt: int = 1,
        occurrence_class: str = "PER_INTERVAL",
        common: bool = False,
        source_site: str = "",
    ) -> _Scope:
        """Return one validated scalar context manager."""

        self._check_owner()
        self._label(label)
        if metadata is not None and context is not None:
            raise TimingConfigurationError("TIMING_SCOPE_CONTEXT_DUPLICATED")
        if context is not None:
            metadata = context
        if metadata is not None and interval is not None:
            raise TimingConfigurationError("TIMING_SCOPE_CONTEXT_DUPLICATED")
        if interval is not None:
            metadata = _context_factory(
                interval,
                phase,
                stage,
                ordinal=ordinal,
                attempt=attempt,
                occurrence_class=occurrence_class,
                source_site=source_site,
            )
        else:
            metadata = self._metadata(metadata)
        return _Scope(self, label, metadata, common)

    def interval_scope(
        self,
        label: str,
        interval_index: int,
        *,
        phase: str = "none",
        stage: str = "none",
        occurrence_class: str = "PER_INTERVAL",
        common: bool = False,
        source_site: str = "",
    ) -> _Scope:
        """Create a scope whose context is tied to original enumerate index."""

        if isinstance(interval_index, bool) or not isinstance(interval_index, int):
            raise TimingConfigurationError("TIMING_INTERVAL_INDEX_INVALID")
        return self.scope(
            label,
            interval=interval_index,
            phase=phase,
            stage=stage,
            ordinal=interval_index,
            attempt=1,
            occurrence_class=occurrence_class,
            common=common,
            source_site=source_site,
        )

    def end(self, token: int, *, failed: bool = False) -> dict[str, Any] | None:
        """End the top scope and calculate inclusive/exclusive clocks."""

        self._check_owner()
        if token == 0:
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
            "token": int(frame.token),
            "label": str(frame.label),
            "context": {
                "interval_index": int(frame.context.interval),
                "ordinal": int(frame.context.ordinal),
                "ordinal_1based": int(frame.context.ordinal_1based),
                "attempt": int(frame.context.attempt),
                "phase": str(frame.context.phase),
                "stage": str(frame.context.stage),
                "occurrence_class": str(frame.context.occurrence_class),
                "source_site": str(frame.context.source_site),
            },
            "parent_token": (
                None if frame.parent_token is None else int(frame.parent_token)
            ),
            "depth": int(frame.depth),
            "start_wall_ns": int(frame.wall_start_ns),
            "end_wall_ns": int(wall_end),
            "start_cpu_ns": int(frame.cpu_start_ns),
            "end_cpu_ns": int(cpu_end),
            "wall_inclusive_ns": int(wall_inclusive),
            "wall_exclusive_ns": int(wall_exclusive),
            "cpu_inclusive_ns": int(cpu_inclusive),
            "cpu_exclusive_ns": int(cpu_exclusive),
            "direct_child_wall_inclusive_ns": int(frame.child_wall_ns),
            "direct_child_cpu_inclusive_ns": int(frame.child_cpu_ns),
            "failed": bool(failed),
            "status": "failed" if failed else "complete",
        }
        self._records.append(record)
        parent = self._stack[-1] if self._stack else None
        if parent is not None:
            parent.child_wall_ns += wall_inclusive
            parent.child_cpu_ns += cpu_inclusive
        return dict(record)

    def marker(
        self,
        kind: str,
        *,
        interval_index: int = 0,
        phase: str = "none",
        stage: str = "none",
        source_site: str = "",
    ) -> None:
        """Record one bounded scalar marker; markers have no new scope label."""

        self._check_owner()
        if kind not in _MARKER_KINDS:
            raise TimingConfigurationError("TIMING_MARKER_KIND_INVALID")
        if not self.enabled:
            return
        metadata = context(
            interval_index,
            phase,
            stage,
            ordinal=interval_index,
            attempt=1,
            occurrence_class="PER_INTERVAL",
            source_site=source_site,
        )
        self._reserve_event()
        self._markers.append(
            {
                "kind": str(kind),
                "interval_index": int(metadata.interval),
                "ordinal": int(metadata.ordinal),
                "ordinal_1based": int(metadata.ordinal_1based),
                "attempt": int(metadata.attempt),
                "phase": str(metadata.phase),
                "stage": str(metadata.stage),
                "source_site": str(metadata.source_site),
            }
        )

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-safe records, markers, grouping, and closure checks."""

        self._check_owner()
        if self._stack:
            raise TimingError("TIMING_INCOMPLETE_SCOPE_STACK")
        records = [
            dict(row, context=dict(row["context"])) for row in self._records
        ]
        markers = [dict(row) for row in self._markers]
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
        closures: list[dict[str, int]] = []
        roots: list[int] = []
        for row in records:
            group = grouped[row["label"]]
            group["call_count"] += 1
            group["failed_count"] += int(row["failed"])
            for name in (
                "wall_inclusive_ns",
                "wall_exclusive_ns",
                "cpu_inclusive_ns",
                "cpu_exclusive_ns",
            ):
                group[name] += int(row[name])
            wall_error = (
                row["wall_inclusive_ns"]
                - row["wall_exclusive_ns"]
                - row["direct_child_wall_inclusive_ns"]
            )
            cpu_error = (
                row["cpu_inclusive_ns"]
                - row["cpu_exclusive_ns"]
                - row["direct_child_cpu_inclusive_ns"]
            )
            if wall_error < 0 or cpu_error < 0:
                raise TimingError("TIMING_NEGATIVE_CLOSURE")
            closures.append(
                {
                    "token": int(row["token"]),
                    "wall_error_ns": int(wall_error),
                    "cpu_error_ns": int(cpu_error),
                }
            )
            if row["parent_token"] is None:
                roots.append(int(row["token"]))
        root_wall = max(
            (row["wall_inclusive_ns"] for row in records if row["parent_token"] is None),
            default=0,
        )
        tolerance_ns = max(1000, int(root_wall * 1e-8))
        max_wall_error = max((row["wall_error_ns"] for row in closures), default=0)
        max_cpu_error = max((row["cpu_error_ns"] for row in closures), default=0)
        return {
            "schema": SCHEMA,
            "enabled": bool(self.enabled),
            "profile": bool(self.enabled),
            "common_shells_enabled": bool(self.common_shells_enabled),
            "owner_thread_id": int(self.owner_thread_id),
            "clock_domain": "perf_counter_ns/process_time_ns",
            "max_records": int(self.max_records),
            "max_metadata_bytes": int(MAX_METADATA_BYTES),
            "reserved_bytes_per_record": int(RESERVED_BYTES_PER_RECORD),
            "metadata_reserved_bytes": int(self.metadata_reserved_bytes),
            "started_scopes": int(self._started_scopes),
            "completed_scopes": int(len(records)),
            "event_count": int(self._event_count),
            "events_dropped": int(self.events_dropped),
            "marker_count": int(len(markers)),
            "markers": markers,
            "incomplete_scopes": [dict(row) for row in self._incomplete],
            "records": records,
            "summary_by_label": {key: dict(value) for key, value in sorted(grouped.items())},
            "roots": roots,
            "clock_closure": {
                "root_wall_inclusive_ns": int(root_wall),
                "tolerance_ns": int(tolerance_ns),
                "max_wall_error_ns": int(max_wall_error),
                "max_cpu_error_ns": int(max_cpu_error),
                "pass": max_wall_error == 0 and max_cpu_error == 0,
            },
            "companion_bridge": {
                "bound": bool(self._companion_recorder is not None),
                "profile": self._companion_profile,
                "identity": self._companion_identity,
                "owner_thread_id": (
                    None
                    if self._companion_recorder is None
                    else int(getattr(self._companion_recorder, "owner_thread_id", -1))
                ),
            },
        }

    to_jsonable = snapshot

    def retire_incomplete(self, reason: str) -> None:
        """Retire open frames without clocks after an exception/limit."""

        self._check_owner()
        if not isinstance(reason, str) or not reason:
            raise TimingError("INCOMPLETE_REASON_MUST_BE_SCALAR_TEXT")
        self._incomplete.extend(
            {
                "token": int(frame.token),
                "label": str(frame.label),
                "parent_token": (
                    None if frame.parent_token is None else int(frame.parent_token)
                ),
                "interval_index": int(frame.context.interval),
                "ordinal": int(frame.context.ordinal),
                "ordinal_1based": int(frame.context.ordinal_1based),
                "attempt": int(frame.context.attempt),
                "phase": str(frame.context.phase),
                "stage": str(frame.context.stage),
                "occurrence_class": str(frame.context.occurrence_class),
                "source_site": str(frame.context.source_site),
                "start_wall_ns": int(frame.wall_start_ns),
                "start_cpu_ns": int(frame.cpu_start_ns),
                "status": "incomplete",
                "reason": str(reason),
            }
            for frame in self._stack
        )
        self._stack.clear()


def empty_scope_control_value(*, enabled: bool = True) -> dict[str, Any]:
    """Measure one empty B scope; it is never a subtraction input."""

    if not enabled:
        return {
            "enabled": False,
            "wall_ns": 0,
            "cpu_ns": 0,
            "subtract_from_scopes": False,
        }
    wall_start = perf_counter_ns()
    cpu_start = process_time_ns()
    recorder = TimingRecorder(enabled=True)
    with recorder.scope("MANAGEMENT", occurrence_class="WINDOW_ONCE"):
        pass
    wall_end = perf_counter_ns()
    cpu_end = process_time_ns()
    wall_ns = wall_end - wall_start
    cpu_ns = cpu_end - cpu_start
    if wall_ns < 0 or cpu_ns < 0:
        raise TimingError("TIMING_EMPTY_CONTROL_CLOCK_WENT_BACKWARDS")
    empty = recorder.snapshot()["records"][0]
    return {
        "enabled": True,
        "wall_ns": int(wall_ns),
        "cpu_ns": int(cpu_ns),
        "empty_scope_wall_ns": int(empty["wall_inclusive_ns"]),
        "empty_scope_cpu_ns": int(empty["cpu_inclusive_ns"]),
        "definition": "One actual empty B recorder context; never subtracted from scientific scopes",
        "subtract_from_scopes": False,
    }


__all__ = [
    "DEFAULT_MAX_RECORDS",
    "MAX_METADATA_BYTES",
    "NEW_A8_LABELS",
    "NEW_B8_LABELS",
    "OCCURRENCE_CLASSES",
    "OLD_SCOPE_LABELS",
    "RESERVED_BYTES_PER_RECORD",
    "SCHEMA",
    "SCOPE_LABELS",
    "TimingConfigurationError",
    "TimingContext",
    "TimingError",
    "TimingRecorder",
    "context",
    "empty_scope_control_value",
]
