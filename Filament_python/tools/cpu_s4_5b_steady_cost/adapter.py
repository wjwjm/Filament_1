"""Private adapters that bridge S4-5A's exact recorder and G2's clone seam.

The accepted S4-5A candidate intentionally rejects subclasses of its
``TimingRecorder``.  S4-5B therefore never changes that check: this module
creates an exact S4-5A recorder for the old scopes and a separate B-owned
recorder for A8/B8.  The private G2 compile interceptor is supplied to a
cloned G2 namespace before its final compile; it never mutates
``__g2_inner__`` after the guard is captured.
"""

from __future__ import annotations

import ast
import builtins
import copy
import importlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType
from typing import Any, Callable, Mapping

from .instrument import (
    INSTRUMENT_SCHEMA,
    InstrumentationBindingError,
    instrument_propagation_ast,
    make_marker_helper,
    make_scope_helper,
    strip_generated_nodes,
)
from .timing import (
    DEFAULT_MAX_RECORDS,
    RESERVED_BYTES_PER_RECORD,
    SCHEMA,
    TimingRecorder,
)
from tools.hpc_ops import provenance_v2 as _provenance


ADAPTER_SCHEMA = "filament.cpu_s4_5b.private_adapter.v1"
DOMAIN = "S45B_SMALLGRID_L1"
ROLE_OFF = "L1_S45B_OFF"
ROLE_PROFILE = "L1_S45B_PROFILE"
ROLES = (ROLE_OFF, ROLE_PROFILE)
QUALIFY_BOUNDARY = "QUALIFY_BOUNDARY"
PURE_OFF_SMALLGRID = "PURE_OFF_SMALLGRID"
MODES = (QUALIFY_BOUNDARY, PURE_OFF_SMALLGRID)


class AdapterBindingError(RuntimeError):
    """A private adapter or exact source/guard contract failed closed."""


@dataclass
class RecorderPair:
    """The two non-interchangeable recorders used by one S45B role."""

    b: TimingRecorder
    s45a: Any
    profile: bool
    identity_token: str = "S45B_RECORDER_PAIR_V1"
    max_records: int = DEFAULT_MAX_RECORDS

    def __post_init__(self) -> None:
        if type(self.b) is not TimingRecorder:
            raise AdapterBindingError("B_RECORDER_EXACT_TYPE_REQUIRED")
        self.profile = _normalise_profile(self.profile)
        self.s45a = accept_s45a_recorder(self.s45a, profile=self.profile)
        if getattr(self.b, "owner_thread_id", None) != getattr(
            self.s45a, "owner_thread_id", None
        ):
            raise AdapterBindingError("S45B_RECORDER_PAIR_OWNER_MISMATCH")
        if not isinstance(self.identity_token, str) or not self.identity_token:
            raise AdapterBindingError("S45B_RECORDER_PAIR_IDENTITY_INVALID")
        if (
            isinstance(self.max_records, bool)
            or not isinstance(self.max_records, int)
            or self.max_records < 2
            or self.max_records > DEFAULT_MAX_RECORDS
        ):
            raise AdapterBindingError("S45B_GLOBAL_MAX_RECORDS_INVALID")
        self.b.bind_companion(
            self.s45a,
            profile=self.profile,
            identity=self.identity_token,
        )

    def snapshot(self) -> dict[str, Any]:
        """Return both scalar snapshots without merging their event trees."""

        b_snapshot = self.b.snapshot()
        a_snapshot = self.s45a.snapshot()
        raw_count = int(a_snapshot.get("started_scopes", 0)) + int(
            b_snapshot.get("event_count", b_snapshot.get("started_scopes", 0))
        )
        if raw_count > int(getattr(self, "max_records", DEFAULT_MAX_RECORDS)):
            raise AdapterBindingError("S45B_COMBINED_RECORD_CAPACITY_EXCEEDED")
        merged = _merge_snapshots(
            b_snapshot,
            a_snapshot,
            profile=self.profile,
        )
        merged_count = len(merged.get("records", []))
        if (raw_count + merged_count) * _reserved_bytes_per_record() > 64 * 1024 * 1024:
            raise AdapterBindingError("S45B_COMBINED_METADATA_BUDGET_EXCEEDED")
        return {
            "schema": ADAPTER_SCHEMA + ".recorders",
            "profile": bool(self.profile),
            "global_max_records": int(getattr(self, "max_records", raw_count)),
            "raw_scope_count": int(raw_count),
            "raw_marker_count": int(b_snapshot.get("marker_count", 0)),
            "metadata_reserved_bytes": int(
                (raw_count + merged_count) * _reserved_bytes_per_record()
            ),
            "b": b_snapshot,
            "s45a": a_snapshot,
            "merged": merged,
            "bridge": copy.deepcopy(b_snapshot.get("companion_bridge", {})),
            "s45a_exact_type": type(self.s45a).__name__,
            "s45a_type_is_subclass_accepted": False,
        }


def _reserved_bytes_per_record() -> int:
    return int(RESERVED_BYTES_PER_RECORD)


def _clock_domain(snapshot: Mapping[str, Any]) -> str:
    value = snapshot.get("clock_domain", "perf_counter_ns/process_time_ns")
    if not isinstance(value, str) or not value:
        raise AdapterBindingError("S45B_CLOCK_DOMAIN_INVALID")
    return value


def _normalised_row(origin: str, row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        raw_token = int(row["token"])
        parent = row.get("parent_token")
        raw_parent = None if parent is None else int(parent)
        start_wall = int(row["start_wall_ns"])
        end_wall = int(row["end_wall_ns"])
        start_cpu = int(row["start_cpu_ns"])
        end_cpu = int(row["end_cpu_ns"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AdapterBindingError("S45B_RAW_RECORD_SCALAR_INVALID") from exc
    if min(start_wall, end_wall, start_cpu, end_cpu) < 0:
        raise AdapterBindingError("S45B_RAW_RECORD_CLOCK_NEGATIVE")
    if end_wall < start_wall or end_cpu < start_cpu:
        raise AdapterBindingError("S45B_RAW_RECORD_CLOCK_ORDER_INVALID")
    ctx = row.get("context")
    if not isinstance(ctx, Mapping):
        raise AdapterBindingError("S45B_RAW_CONTEXT_MISSING")
    interval = int(ctx.get("interval_index", ctx.get("interval", 0)))
    ordinal = int(ctx.get("ordinal", interval))
    ordinal_1based = int(ctx.get("ordinal_1based", ordinal + 1))
    attempt = int(ctx.get("attempt", 1))
    source_site = ctx.get("source_site", "")
    if not isinstance(source_site, str) or len(source_site) > 160:
        raise AdapterBindingError("S45B_RAW_SOURCE_SITE_INVALID")
    raw_context = {
        "interval_index": interval,
        "ordinal": ordinal,
        "ordinal_1based": ordinal_1based,
        "attempt": attempt,
        "phase": str(ctx.get("phase", "none")),
        "stage": str(ctx.get("stage", "none")),
        "occurrence_class": str(ctx.get("occurrence_class", "PER_INTERVAL")),
        "source_site": source_site,
    }
    normalized = {
        "start_wall_ns": start_wall,
        "end_wall_ns": end_wall,
        "start_cpu_ns": start_cpu,
        "end_cpu_ns": end_cpu,
        "wall_duration_ns": end_wall - start_wall,
        "cpu_duration_ns": end_cpu - start_cpu,
        "interval_index": interval,
        "ordinal": ordinal,
        "ordinal_1based": ordinal_1based,
        "attempt": attempt,
        "phase": raw_context["phase"],
        "stage": raw_context["stage"],
        "occurrence_class": raw_context["occurrence_class"],
        "source_site": source_site,
    }
    return {
        "origin": str(origin),
        "raw_token": raw_token,
        "raw_parent_token": raw_parent,
        "raw_context": copy.deepcopy(dict(ctx)),
        "normalized_interval": normalized,
        "label": str(row.get("label", "")),
        "start_wall_ns": start_wall,
        "end_wall_ns": end_wall,
        "start_cpu_ns": start_cpu,
        "end_cpu_ns": end_cpu,
        "wall_inclusive_ns": end_wall - start_wall,
        "cpu_inclusive_ns": end_cpu - start_cpu,
        "failed": bool(row.get("failed", False)),
        "status": str(row.get("status", "complete")),
    }


def _strict_contains(parent: Mapping[str, Any], child: Mapping[str, Any]) -> bool:
    return (
        int(parent["start_wall_ns"]) < int(child["start_wall_ns"])
        and int(child["end_wall_ns"]) < int(parent["end_wall_ns"])
        and int(parent["start_cpu_ns"]) <= int(child["start_cpu_ns"])
        and int(child["end_cpu_ns"]) <= int(parent["end_cpu_ns"])
    )


def _wall_strict_contains(parent: Mapping[str, Any], child: Mapping[str, Any]) -> bool:
    return (
        int(parent["start_wall_ns"]) < int(child["start_wall_ns"])
        and int(child["end_wall_ns"]) < int(parent["end_wall_ns"])
    )


def _overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def _validate_pair_clock_relation(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    wall_overlap = _overlap(
        int(left["start_wall_ns"]),
        int(left["end_wall_ns"]),
        int(right["start_wall_ns"]),
        int(right["end_wall_ns"]),
    )
    if not wall_overlap:
        return
    wall_equal = (
        int(left["start_wall_ns"]) == int(right["start_wall_ns"])
        and int(left["end_wall_ns"]) == int(right["end_wall_ns"])
    )
    if wall_equal:
        raise AdapterBindingError("S45B_AMBIGUOUS_EQUAL_SPAN")
    wall_left = (
        int(left["start_wall_ns"]) < int(right["start_wall_ns"])
        and int(right["end_wall_ns"]) < int(left["end_wall_ns"])
    )
    wall_right = (
        int(right["start_wall_ns"]) < int(left["start_wall_ns"])
        and int(left["end_wall_ns"]) < int(right["end_wall_ns"])
    )
    if not (wall_left or wall_right):
        raise AdapterBindingError("S45B_CROSSING_CLOCK_SPANS")
    wall_parent, wall_child = (
        (left, right) if wall_left else (right, left)
    )
    if not (
        int(wall_parent["start_cpu_ns"]) <= int(wall_child["start_cpu_ns"])
        and int(wall_child["end_cpu_ns"]) <= int(wall_parent["end_cpu_ns"])
    ):
        raise AdapterBindingError("S45B_CROSSING_CPU_SPANS")


def _merge_snapshots(
    b_snapshot: Mapping[str, Any],
    a_snapshot: Mapping[str, Any],
    *,
    profile: bool,
) -> dict[str, Any]:
    """Merge two raw scalar trees without modifying either source tree."""

    if bool(b_snapshot.get("profile")) is not bool(profile) or bool(
        a_snapshot.get("enabled", a_snapshot.get("profile", False))
    ) is not bool(profile):
        raise AdapterBindingError("S45B_MERGE_PROFILE_SWITCH_MISMATCH")
    b_owner = b_snapshot.get("owner_thread_id")
    a_owner = a_snapshot.get("owner_thread_id")
    if b_owner != a_owner:
        raise AdapterBindingError("S45B_MERGE_OWNER_MISMATCH")
    if not isinstance(b_owner, int):
        raise AdapterBindingError("S45B_MERGE_OWNER_ID_INVALID")
    b_domain = _clock_domain(b_snapshot)
    a_domain = _clock_domain(a_snapshot)
    if b_domain != a_domain:
        raise AdapterBindingError("S45B_MERGE_CLOCK_DOMAIN_MISMATCH")
    for snapshot in (b_snapshot, a_snapshot):
        if int(snapshot.get("events_dropped", 0)) != 0:
            raise AdapterBindingError("S45B_MERGE_EVENTS_DROPPED")
        if snapshot.get("incomplete_scopes"):
            raise AdapterBindingError("S45B_MERGE_INCOMPLETE_SCOPES")
    rows: list[dict[str, Any]] = []
    seen: dict[str, set[int]] = {"A": set(), "B": set()}
    for origin, snapshot in (("A", a_snapshot), ("B", b_snapshot)):
        raw_rows = snapshot.get("records", [])
        if not isinstance(raw_rows, list):
            raise AdapterBindingError("S45B_MERGE_RECORDS_INVALID")
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                raise AdapterBindingError("S45B_MERGE_RECORD_INVALID")
            row = _normalised_row(origin, raw)
            token = int(row["raw_token"])
            if token in seen[origin]:
                raise AdapterBindingError("S45B_MERGE_DUPLICATE_RAW_TOKEN")
            seen[origin].add(token)
            row["_raw_key"] = (origin, token)
            rows.append(row)
    by_key = {row["_raw_key"]: row for row in rows}
    # The old A segment counter is retained raw. Only actual B for-body spans
    # establish source interval ordinals in the merged analysis view.
    interval_rows = [row for row in rows if row["origin"] == "B" and row["label"] == "INTERVAL_BODY"]
    for row in rows:
        enclosing = [body for body in interval_rows if body is row or _strict_contains(body, row)]
        if len(enclosing) > 1:
            raise AdapterBindingError("S45B_MERGE_AMBIGUOUS_INTERVAL_BODY")
        normal = row["normalized_interval"]
        if enclosing:
            body_context = enclosing[0]["raw_context"]
            for name in ("interval_index", "ordinal", "ordinal_1based", "attempt"):
                normal[name] = int(body_context[name])
            if row["origin"] == "A":
                normal["occurrence_class"] = "PER_INTERVAL"
            normal["interval_identity_basis"] = "ACTUAL_B_FOR_BODY_CONTAINMENT"
        else:
            for name in ("interval_index", "ordinal", "ordinal_1based"):
                normal[name] = None
            normal["interval_identity_basis"] = "OUTSIDE_ACTUAL_INTERVAL_BODY"
            if row["origin"] == "A":
                normal["occurrence_class"] = "WINDOW_ONCE"
    for row in rows:
        parent = row["raw_parent_token"]
        if parent is None:
            continue
        parent_row = by_key.get((row["origin"], int(parent)))
        if parent_row is None:
            raise AdapterBindingError("S45B_MERGE_RAW_PARENT_MISSING")
        if not _strict_contains(parent_row, row):
            raise AdapterBindingError("S45B_MERGE_RAW_PARENT_NOT_CONTAINING")
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            _validate_pair_clock_relation(left, right)

    merged_parent: dict[tuple[str, int], tuple[str, int] | None] = {}
    for row in rows:
        key = row["_raw_key"]
        candidates = [
            other
            for other in rows
            if other is not row and _wall_strict_contains(other, row)
        ]
        innermost = [
            candidate
            for candidate in candidates
            if not any(
                candidate is not other and _wall_strict_contains(candidate, other)
                for other in candidates
            )
        ]
        if len(innermost) > 1:
            raise AdapterBindingError("S45B_MERGE_AMBIGUOUS_PARENT")
        merged_parent[key] = None if not innermost else innermost[0]["_raw_key"]

    # A raw parent edge must survive as an ancestor, but a cross-tree wall
    # container may be the direct merged parent.  This check therefore walks
    # the merged chain rather than forcing raw parents to be direct children.
    for row in rows:
        raw_parent = row["raw_parent_token"]
        if raw_parent is None:
            continue
        expected_parent = (row["origin"], int(raw_parent))
        cursor = merged_parent[row["_raw_key"]]
        ancestors: set[tuple[str, int]] = set()
        while cursor is not None:
            if cursor in ancestors:
                raise AdapterBindingError("S45B_MERGE_PARENT_CYCLE")
            ancestors.add(cursor)
            if cursor == expected_parent:
                break
            cursor = merged_parent.get(cursor)
        if cursor != expected_parent:
            raise AdapterBindingError("S45B_MERGE_RAW_PARENT_NOT_ANCESTOR")

    ordered = list(rows)
    merged_token_by_key = {
        row["_raw_key"]: f"M{index + 1}" for index, row in enumerate(ordered)
    }
    children: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        parent_key = merged_parent[row["_raw_key"]]
        if parent_key is not None:
            children[parent_key].append(row)
    result_rows: list[dict[str, Any]] = []
    depth_cache: dict[tuple[str, int], int] = {}

    def depth(key: tuple[str, int], trail: set[tuple[str, int]] | None = None) -> int:
        if key in depth_cache:
            return depth_cache[key]
        trail = set() if trail is None else set(trail)
        if key in trail:
            raise AdapterBindingError("S45B_MERGE_PARENT_CYCLE")
        trail.add(key)
        parent = merged_parent[key]
        value = 0 if parent is None else depth(parent, trail) + 1
        depth_cache[key] = value
        return value

    for row in ordered:
        key = row["_raw_key"]
        child_rows = children.get(key, [])
        child_wall = sum(int(child["wall_inclusive_ns"]) for child in child_rows)
        child_cpu = sum(int(child["cpu_inclusive_ns"]) for child in child_rows)
        wall_inclusive = int(row["wall_inclusive_ns"])
        cpu_inclusive = int(row["cpu_inclusive_ns"])
        wall_exclusive = wall_inclusive - child_wall
        cpu_exclusive = cpu_inclusive - child_cpu
        if wall_exclusive < 0 or cpu_exclusive < 0:
            raise AdapterBindingError("S45B_MERGE_NEGATIVE_EXCLUSIVE")
        parent_key = merged_parent[key]
        result_rows.append(
            {
                "merged_token": merged_token_by_key[key],
                "merged_parent_token": (
                    None if parent_key is None else merged_token_by_key[parent_key]
                ),
                "origin": row["origin"],
                "raw_token": int(row["raw_token"]),
                "raw_parent_token": row["raw_parent_token"],
                "raw_context": copy.deepcopy(row["raw_context"]),
                "normalized_interval": copy.deepcopy(row["normalized_interval"]),
                "label": row["label"],
                "depth": int(depth(key)),
                "start_wall_ns": int(row["start_wall_ns"]),
                "end_wall_ns": int(row["end_wall_ns"]),
                "start_cpu_ns": int(row["start_cpu_ns"]),
                "end_cpu_ns": int(row["end_cpu_ns"]),
                "wall_inclusive_ns": wall_inclusive,
                "wall_exclusive_ns": wall_exclusive,
                "cpu_inclusive_ns": cpu_inclusive,
                "cpu_exclusive_ns": cpu_exclusive,
                "direct_child_wall_inclusive_ns": child_wall,
                "direct_child_cpu_inclusive_ns": child_cpu,
                "failed": bool(row["failed"]),
                "status": row["status"],
            }
        )
    roots = [
        row["merged_token"]
        for row in result_rows
        if row["merged_parent_token"] is None
    ]
    return {
        "schema": ADAPTER_SCHEMA + ".merged_tree",
        "profile": bool(profile),
        "owner_thread_id": int(b_owner),
        "clock_domain": b_domain,
        "raw_rows_retained": len(result_rows),
        "records": result_rows,
        "roots": roots,
        "forest_allowed": True,
        "all_raw_parent_edges_preserved": True,
        "clock_closure": {
            "pass": all(
                int(row["wall_inclusive_ns"])
                == int(row["wall_exclusive_ns"])
                + int(row["direct_child_wall_inclusive_ns"])
                for row in result_rows
            )
            and all(
                int(row["cpu_inclusive_ns"])
                == int(row["cpu_exclusive_ns"])
                + int(row["direct_child_cpu_inclusive_ns"])
                for row in result_rows
            )
        },
    }


def _normalise_profile(profile: bool) -> bool:
    if not isinstance(profile, bool):
        raise AdapterBindingError("S45B_PROFILE_MUST_BE_BOOL")
    return bool(profile)


def make_recorder(profile: bool, *, max_records: int = DEFAULT_MAX_RECORDS) -> TimingRecorder:
    """Create the B-owned bounded recorder for one role."""

    return TimingRecorder(enabled=_normalise_profile(profile), max_records=max_records)


def accept_s45a_recorder(recorder: Any, *, profile: bool) -> Any:
    """Accept only the exact frozen S4-5A recorder class.

    This is deliberately private to B.  It does not loosen or monkey-patch
    the S4-5A candidate's own exact-type check.
    """

    profile = _normalise_profile(profile)
    module = importlib.import_module("tools.cpu_s4_5a_l1_residual.timing")
    expected = getattr(module, "TimingRecorder", None)
    if expected is None or type(recorder) is not expected:
        raise AdapterBindingError("S45A_RECORDER_EXACT_TYPE_REQUIRED")
    if getattr(recorder, "enabled", object()) is not profile:
        raise AdapterBindingError("S45A_RECORDER_PROFILE_SWITCH_MISMATCH")
    return recorder


def make_recorder_pair(profile: bool, *, max_records: int = DEFAULT_MAX_RECORDS) -> RecorderPair:
    """Create B plus exact A under one global raw-record budget."""

    profile = _normalise_profile(profile)
    if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 2:
        raise AdapterBindingError("S45B_GLOBAL_MAX_RECORDS_INVALID")
    if max_records > DEFAULT_MAX_RECORDS:
        raise AdapterBindingError("S45B_GLOBAL_MAX_RECORDS_EXCEEDS_FIXED_BOUND")
    a_limit = min(4096, max(1, max_records // 2))
    b_limit = max_records - a_limit
    b = make_recorder(profile, max_records=b_limit)
    s45a_module = importlib.import_module("tools.cpu_s4_5a_l1_residual.timing")
    s45a_type = getattr(s45a_module, "TimingRecorder", None)
    if s45a_type is None:
        raise AdapterBindingError("S45A_TIMING_MODULE_MISSING")
    s45a = s45a_type(enabled=profile, max_scopes=a_limit)
    accept_s45a_recorder(s45a, profile=profile)
    return RecorderPair(
        b=b,
        s45a=s45a,
        profile=profile,
        identity_token=f"S45B_RECORDER_PAIR_V1:{id(b)}:{id(s45a)}",
        max_records=int(max_records),
    )


def bind_runtime_helpers(recorder: TimingRecorder) -> dict[str, Callable[..., Any]]:
    """Return the fixed propagation helper names for a cloned G2 namespace."""

    if type(recorder) is not TimingRecorder:
        raise AdapterBindingError("B_RECORDER_EXACT_TYPE_REQUIRED")
    return {
        "__cpu_s45b_scope__": make_scope_helper(recorder),
        "__cpu_s45b_marker__": make_marker_helper(recorder),
    }


def attach_with_b(
    raw_propagator: Callable[..., Any],
    dispatch: Any,
    *,
    role: str,
    mode: str,
    capture_root: str | Path,
    owned_identity: Mapping[str, Any],
    recorder: TimingRecorder,
) -> tuple[Any, Any, Any, dict[str, Any], Any, Any]:
    """Attach a B-instrumented G2 clone using the existing six-tuple API.

    This is the only runtime seam owned by B.  It mirrors the small S4-5A
    runtime adapter, but inserts B helpers into the private G2 clone before
    ``build_role`` compiles and captures the actual code.  The original G2
    module, its globals, and its wrapper are never mutated.
    """

    if type(recorder) is not TimingRecorder:
        raise AdapterBindingError("B_RECORDER_EXACT_TYPE_REQUIRED")
    if not callable(raw_propagator):
        raise AdapterBindingError("S45B_RAW_PROPAGATOR_REQUIRED")
    role_name = str(role).upper()
    if role_name not in ("G_CONTROLLED_INPUT", "I0"):
        raise AdapterBindingError(f"S45B_G2_ROLE_INVALID:{role_name}")
    mode_name = str(mode).upper()
    if mode_name not in MODES:
        raise AdapterBindingError(f"S45B_G2_MODE_INVALID:{mode_name}")

    from types import FunctionType, SimpleNamespace

    from tools.cpu_s4_3g3_smallgrid.scientific_context import frozen_tools
    from tools.cpu_s4_checkpoint_support import g2_adapter_path, load_g2_adapter

    repo = Path(__file__).resolve().parents[3]
    frozen_tools(repo)
    original = load_g2_adapter()
    expected_adapter = g2_adapter_path()
    if Path(getattr(original, "__file__", "")).resolve() != expected_adapter.resolve(strict=True):
        raise AdapterBindingError("S45B_FROZEN_G2_MODULE_MISMATCH")
    namespace = clone_g2_namespace(original)
    common = SimpleNamespace(**vars(original._common))
    common.factory = lambda unused_role: (lambda unused_base: (raw_propagator, dispatch))
    namespace["_common"] = common

    old_external = namespace["_external_frozen_source_binding"]
    own = {
        str(Path(record["path"]).resolve()): copy.deepcopy(record)
        for record in owned_identity.get("files", {}).values()
        if isinstance(record, Mapping) and record.get("path")
    }
    old_tracked = namespace["_frozen_tracked_record"]
    own_tracked = {
        record["relative_path"]: {
            key: record[key]
            for key in ("classification", "hash_scope", "git_blob_oid", "canonical_lf_sha256")
        } | {"path": record["relative_path"]}
        for record in own.values()
        if record.get("classification") == "tracked_text"
    }

    def frozen_tracked_record(relative: str) -> dict[str, Any] | None:
        record = own_tracked.get(relative)
        if record is None:
            return old_tracked(relative)
        # The unchanged G2 method binder performs authoritative PV2 validation.
        return copy.deepcopy(record)

    namespace["_frozen_tracked_record"] = frozen_tracked_record

    def external_binding(path: Any, raw: bytes) -> dict[str, Any]:
        actual_path = Path(path).resolve(strict=True)
        record = own.get(str(actual_path))
        if record is None:
            return old_external(path, raw)
        digest = _provenance.raw_sha256(raw)
        if digest != record.get("raw_sha256") or len(raw) != int(record.get("size_bytes", -1)):
            raise AdapterBindingError("S45B_OWNED_SOURCE_CHANGED_BEFORE_CALL")
        return {
            "relative_path": actual_path.relative_to(repo).as_posix(),
            "filename": str(actual_path),
            "classification": "external",
            "hash_scope": "raw_bytes",
            "git_blob_oid": None,
            "external_frozen_source": True,
            "external_binding_report": "S45B_OWNED_PRE_CALL_SOURCE_RECORDS",
            "expected_raw_sha256": record["raw_sha256"],
            "source_raw_sha256": digest,
            "source_size_bytes": len(raw),
            "external_source_path_match": "exact_owned_namespace",
        }

    namespace["_external_frozen_source_binding"] = external_binding
    b_helpers = bind_runtime_helpers(recorder)

    def compile_interceptor(*, role: str, mode: str, bindings: tuple[tuple[str, str], ...]):
        return make_g2_compile_interceptor(
            role=role,
            mode=mode,
            bindings=bindings,
            g2_namespace=namespace,
        )

    namespace["_make_compile_interceptor"] = compile_interceptor

    original_clone_g = namespace.get("_clone_g")
    original_clone_i0 = namespace.get("_clone_i0")
    if not callable(original_clone_g) or not callable(original_clone_i0):
        raise AdapterBindingError("S45B_G2_CLONE_ENTRY_MISSING")

    def clone_g(*args: Any, helpers: Mapping[str, Any], **kwargs: Any):
        extended = dict(helpers)
        extended.update(b_helpers)
        return original_clone_g(*args, helpers=extended, **kwargs)

    def clone_i0(*args: Any, helpers: Mapping[str, Any], **kwargs: Any):
        extended = dict(helpers)
        extended.update(b_helpers)
        return original_clone_i0(*args, helpers=extended, **kwargs)

    namespace["_clone_g"] = clone_g
    namespace["_clone_i0"] = clone_i0
    frozen_role = role_name
    frozen_mode = QUALIFY_BOUNDARY if mode_name == QUALIFY_BOUNDARY else "PURE_OFF_REAL8"
    wrapped, actual_dispatch, clock, binding, audit, lease = namespace["build_role"](
        frozen_role,
        frozen_mode,
        capture_root=Path(capture_root).resolve(),
    )
    if actual_dispatch is not dispatch:
        raise AdapterBindingError("S45B_ACTUAL_DISPATCH_BINDING_CHANGED")
    binding = copy.deepcopy(binding)
    binding.update(
        {
            "s45b_domain": DOMAIN,
            "s45b_mode": mode_name,
            "s45b_role": ROLE_PROFILE if recorder.enabled else ROLE_OFF,
            "s45b_instrumentation_schema": INSTRUMENT_SCHEMA,
            "s45b_runtime_helpers": sorted(b_helpers),
            "s45b_g2_compile_seam": "PRIVATE_NAMESPACE_BEFORE_FINAL_COMPILE",
            "s45b_old_g2_wrapper_untouched": True,
            "s45b_owned_candidate": copy.deepcopy(dict(owned_identity)),
        }
    )
    return wrapped, actual_dispatch, clock, binding, audit, lease


def make_g2_compile_interceptor(
    *,
    role: str,
    mode: str,
    bindings: tuple[tuple[str, str], ...],
    g2_namespace: Mapping[str, Any],
) -> Callable[..., Any]:
    """Create a private G2 interceptor that adds B AST before final compile.

    ``g2_namespace`` must be the private clone namespace used by the active
    G2 adapter, not the original module dictionary.  Existing G2 timing and
    (when requested) boundary insertion are applied first.  B scopes are then
    inserted, and the old timing/boundary stripper is used to prove that only
    B's generated nodes differ from the baseline.  The resulting clone is
    therefore captured by the normal G2 wrapper guards at construction time.
    """

    role = str(role)
    mode = str(mode).upper()
    if mode == "PURE_OFF_REAL8":
        mode = PURE_OFF_SMALLGRID
    if mode not in MODES:
        raise AdapterBindingError(f"S45B_G2_MODE_INVALID:{mode}")
    timing_module = g2_namespace.get("_timing")
    inject_timing = getattr(timing_module, "_inject_timing", None)
    insert_boundary = g2_namespace.get("_insert_boundary")
    strip_boundary = g2_namespace.get("_strip_boundary")
    structure_without_timing = g2_namespace.get("_structure_without_timing_ast")
    if not callable(inject_timing) or not callable(structure_without_timing):
        raise AdapterBindingError("S45B_G2_TIMING_SEAM_UNAVAILABLE")
    if mode == QUALIFY_BOUNDARY and not callable(insert_boundary):
        raise AdapterBindingError("S45B_G2_BOUNDARY_INSERT_SEAM_UNAVAILABLE")
    if mode == QUALIFY_BOUNDARY and not callable(strip_boundary):
        raise AdapterBindingError("S45B_G2_BOUNDARY_STRIP_SEAM_UNAVAILABLE")

    injected = False

    def interceptor(
        source: Any,
        filename: Any,
        compile_mode: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        nonlocal injected
        if not injected and isinstance(source, ast.Module):
            baseline = copy.deepcopy(source)
            timing_receipt = inject_timing(source, role=role)
            boundary_receipt = None
            if mode == QUALIFY_BOUNDARY:
                boundary_receipt = insert_boundary(source, bindings)
            function_node = _find_function(source, "propagate_one_pulse")
            b_receipt = instrument_propagation_ast(function_node)
            # Remove only B nodes first.  The old timing/boundary tree must be
            # exactly what the existing G2 validation expects.
            without_b = strip_generated_nodes(source)
            if mode == QUALIFY_BOUNDARY:
                without_old_timing, count = strip_boundary(
                    structure_without_timing(without_b)
                )
                if count != 1:
                    raise AdapterBindingError(f"S45B_BOUNDARY_STRIP_COUNT:{count}")
                post_strip = without_old_timing
            else:
                post_strip = structure_without_timing(without_b)
            if _ast_dump(post_strip) != _ast_dump(baseline):
                raise AdapterBindingError("S45B_G2_AST_CHANGED_BEYOND_B_SCOPES")
            interceptor.static_receipt = {
                "schema": ADAPTER_SCHEMA + ".g2_static",
                "role": role,
                "mode": mode,
                "timing": copy.deepcopy(timing_receipt),
                "boundary": copy.deepcopy(boundary_receipt),
                "b_propagation": copy.deepcopy(b_receipt),
                "baseline_ast_sha256": _ast_hash(baseline),
                "post_strip_ast_sha256": _ast_hash(post_strip),
                "b_stripped_ast_sha256": _ast_hash(without_b),
                "old_g2_tree_preserved": True,
                "actual_clone_guard_seam": "BEFORE_FINAL_COMPILE",
            }
            injected = True
        return builtins.compile(source, filename, compile_mode, *args, **kwargs)

    interceptor.static_receipt = None
    return interceptor


def _find_function(module: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise AdapterBindingError(f"S45B_G2_FUNCTION_COUNT:{name}:{len(matches)}")
    return matches[0]


def _ast_dump(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _ast_hash(node: ast.AST) -> str:
    return _provenance.canonical_lf_sha256(_ast_dump(node).encode("utf-8"))


def clone_g2_namespace(original: Any) -> dict[str, Any]:
    """Build a narrow private G2 globals seam.

    Only ``build_role``, ``_clone_g`` and ``_clone_i0`` are cloned with
    :class:`types.FunctionType`; their other dependencies remain references to
    the frozen G2 module.  This is intentionally not a copy of the whole G2
    framework.  The caller may replace ``_make_compile_interceptor`` and the
    two clone entry points before invoking the private ``build_role``.
    """

    if original is None or not isinstance(getattr(original, "__dict__", None), dict):
        raise AdapterBindingError("S45B_G2_MODULE_REQUIRED")
    source_globals = vars(original)
    namespace: dict[str, Any] = {
        name: ({} if name.endswith("CACHE") and isinstance(value, dict) else value)
        for name, value in source_globals.items()
    }
    for name in ("build_role", "_clone_g", "_clone_i0"):
        value = source_globals.get(name)
        if not isinstance(value, FunctionType) or value.__globals__ is not source_globals:
            raise AdapterBindingError(f"S45B_G2_TARGETED_FUNCTION_MISSING:{name}")
        clone = FunctionType(
            value.__code__,
            namespace,
            value.__name__,
            value.__defaults__,
            value.__closure__,
        )
        clone.__kwdefaults__ = getattr(value, "__kwdefaults__", None)
        clone.__annotations__ = dict(getattr(value, "__annotations__", {}) or {})
        clone.__qualname__ = getattr(value, "__qualname__", value.__name__)
        clone.__module__ = getattr(value, "__module__", None)
        clone.__doc__ = getattr(value, "__doc__", None)
        namespace[name] = clone
    return namespace


def source_identity(path: str | Path) -> dict[str, Any]:
    """Return raw-byte identity for one private B source file."""

    resolved = Path(path).resolve(strict=True)
    raw = resolved.read_bytes()
    return {
        "path": str(resolved),
        "classification": "external",
        "hash_scope": "raw_bytes",
        "size_bytes": int(len(raw)),
        "raw_sha256": _provenance.raw_sha256(raw),
    }


__all__ = [
    "ADAPTER_SCHEMA",
    "AdapterBindingError",
    "DOMAIN",
    "MODES",
    "PURE_OFF_SMALLGRID",
    "QUALIFY_BOUNDARY",
    "ROLE_OFF",
    "ROLE_PROFILE",
    "ROLES",
    "RecorderPair",
    "accept_s45a_recorder",
    "bind_runtime_helpers",
    "clone_g2_namespace",
    "make_g2_compile_interceptor",
    "make_recorder",
    "make_recorder_pair",
    "_merge_snapshots",
    "source_identity",
]
