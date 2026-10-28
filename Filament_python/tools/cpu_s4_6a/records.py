"""Strict record checks for CPU-S4-6A.

This module is intentionally independent of the propagation implementation.
It validates paths, clocks, case ordering, observer timing, and bounded
resource ledgers.  It does not create a new hash convention: raw-byte hashes
are delegated to :mod:`tools.hpc_ops.provenance_v2`.
"""

from __future__ import annotations

import json
import math
import ntpath
import os
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from tools.hpc_ops import provenance_v2 as _provenance


CASE_ORDER = (
    "QUAL_OFF",
    "QUAL_LIGHT",
    "P1_OFF",
    "P1_LIGHT",
    "P2_LIGHT",
    "P2_OFF",
    "P3_OFF",
    "P3_LIGHT",
)


BUDGET_CONTRACTS: dict[str, dict[str, Any]] = {
    "CPU": {
        "max_job_ids": 3,
        "max_core_hours": 240.0,
        "cpus": 96,
        "gpus": 0,
        "max_inflight": 1,
        "max_walltime_s": {"small": 1200.0, "large": 5400.0, "repair": 5400.0},
        "repair_walltime_s": None,
    },
    "GPU": {
        "max_job_ids": 2,
        "max_gpu_hours": 2.0,
        "max_core_hours": 16.0,
        "gpus": 1,
        "cpus": 8,
        "max_inflight": 1,
        "max_walltime_s": {"main": 5400.0, "repair": 1800.0},
    },
}


class RecordError(ValueError):
    """Malformed or unsafe evidence record."""


class ClockError(RecordError):
    """Invalid, incomplete, overlapping, or negative clock record."""


class TimingGateError(RecordError):
    """Invalid OFF/LIGHT timing pair or failed observer gate."""


class BudgetError(RecordError):
    """Invalid resource ledger or exceeded task budget."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RecordError(f"DUPLICATE_JSON_KEY:{key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise RecordError(f"NONFINITE_JSON_CONSTANT:{value}")


def _ensure_finite_json(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RecordError(f"NONFINITE_JSON_VALUE:{path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _ensure_finite_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _ensure_finite_json(child, f"{path}[{index}]")


def load_json_strict(path: str | os.PathLike[str]) -> Any:
    """Load UTF-8/UTF-8-BOM JSON while rejecting duplicate keys and NaN."""

    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise RecordError(f"JSON_READ_FAILED:{source}") from exc
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
        _ensure_finite_json(value)
        return value
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise RecordError(f"JSON_INVALID:{source}") from exc


def _normal_relative(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise RecordError("SOURCE_PATH_REQUIRED")
    candidate = value.replace("\\", "/")
    # Reject both Windows and POSIX absolute forms before Path semantics vary.
    drive, _ = ntpath.splitdrive(candidate)
    if drive or candidate.startswith("/") or candidate.startswith("//"):
        raise RecordError(f"SOURCE_PATH_ABSOLUTE:{value}")
    parts = candidate.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise RecordError(f"SOURCE_PATH_COMPONENT:{value}")
    return "/".join(parts)


def _raw_sha256(path: Path) -> str:
    # Use the repository's authoritative raw-byte implementation.
    try:
        return str(_provenance.raw_sha256_file(path)).lower()
    except OSError as exc:
        raise RecordError(f"SOURCE_HASH_READ_FAILED:{path}") from exc


def _ensure_inside(root: Path, path: Path) -> Path:
    # Walk the lexical path before resolving it.  Resolving first would erase
    # the symlink component that must be rejected for provenance safety.
    lexical_root = root.absolute()
    lexical_path = path.absolute()
    try:
        lexical_relative = lexical_path.relative_to(lexical_root)
    except ValueError as exc:
        raise RecordError(f"SOURCE_PATH_OUTSIDE_ROOT:{path}") from exc
    cursor = lexical_root
    if cursor.is_symlink():
        raise RecordError(f"SOURCE_PATH_SYMLINK:{cursor}")
    for part in lexical_relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RecordError(f"SOURCE_PATH_SYMLINK:{cursor}")
    try:
        resolved_root = lexical_root.resolve(strict=True)
        resolved = lexical_path.resolve(strict=True)
    except OSError as exc:
        raise RecordError(f"SOURCE_FILE_MISSING:{path}") from exc
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise RecordError(f"SOURCE_PATH_OUTSIDE_ROOT:{path}") from exc
    return resolved


def resolve_record_path(
    root: str | os.PathLike[str],
    relative_path: str,
    *,
    expected_raw_sha256: str | None = None,
    all_paths: Iterable[str] | None = None,
) -> Path:
    """Resolve one source path with portable path and raw-hash guards.

    A basename-only record is accepted only when exactly one matching file is
    present below ``root``.  Callers with a manifest should pass ``all_paths``;
    duplicate basenames there are rejected before resolution.
    """

    normalized = _normal_relative(relative_path)
    root_path = Path(root).absolute()
    try:
        if not root_path.exists():
            raise FileNotFoundError(root_path)
    except OSError as exc:
        raise RecordError(f"SOURCE_ROOT_MISSING:{root}") from exc
    if all_paths is not None:
        normalized_paths = [_normal_relative(item) for item in all_paths]
        basename = Path(normalized).name
        matches = [item for item in normalized_paths if Path(item).name == basename]
        if "/" not in normalized and len(matches) > 1:
            raise RecordError(f"SOURCE_BASENAME_AMBIGUOUS:{basename}")
    if "/" not in normalized:
        candidates = sorted(
            item
            for item in root_path.rglob(normalized)
            if item.is_file() and not item.is_symlink()
        )
        if len(candidates) != 1:
            reason = "MISSING" if not candidates else "AMBIGUOUS"
            raise RecordError(f"SOURCE_BASENAME_{reason}:{normalized}")
        target = candidates[0]
    else:
        target = root_path.joinpath(*normalized.split("/"))
    resolved = _ensure_inside(root_path, target)
    if not resolved.is_file():
        raise RecordError(f"SOURCE_FILE_MISSING:{normalized}")
    if expected_raw_sha256 is not None:
        expected = str(expected_raw_sha256).lower()
        actual = _raw_sha256(resolved)
        if actual != expected:
            raise RecordError(f"SOURCE_RAW_SHA256_MISMATCH:{normalized}")
    return resolved


def validate_source_records(
    root: str | os.PathLike[str], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Validate a manifest's paths, unique basenames, and raw-byte hashes."""

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise RecordError("SOURCE_RECORDS_MUST_BE_SEQUENCE")
    paths = [_normal_relative(record.get("path")) for record in records]
    if len(set(paths)) != len(paths):
        raise RecordError("SOURCE_PATH_DUPLICATE")
    resolved: list[dict[str, Any]] = []
    for record, path in zip(records, paths):
        scope = record.get("hash_scope", "raw_bytes")
        if scope != "raw_bytes":
            raise RecordError(f"SOURCE_HASH_SCOPE_UNSUPPORTED:{scope}")
        digest = record.get("raw_sha256", record.get("sha256"))
        if not isinstance(digest, str) or len(digest) != 64:
            raise RecordError(f"SOURCE_RAW_SHA256_REQUIRED:{path}")
        resolved_path = resolve_record_path(root, path, expected_raw_sha256=digest, all_paths=paths)
        resolved.append({"path": path, "resolved": str(resolved_path), "raw_sha256": digest.lower()})
    return {"status": "PASS_SOURCE_RECORDS", "records": resolved, "default": "I0", "production_qualified": False, "DEFAULT": "I0", "PRODUCTION_QUALIFIED": False}


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ClockError(f"CLOCK_INTEGER_REQUIRED:{field}")
    return value


def _clock_rows(rows: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(rows, Mapping):
        result = []
        for name, value in rows.items():
            if not isinstance(value, Mapping):
                raise ClockError(f"CLOCK_ROW_INVALID:{name}")
            row = dict(value)
            row.setdefault("scope", name)
            result.append(row)
        return result
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        if any(not isinstance(row, Mapping) for row in rows):
            raise ClockError("CLOCK_ROW_INVALID")
        return [dict(row) for row in rows]
    raise ClockError("CLOCK_ROWS_REQUIRED")


def validate_clock_records(
    rows: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    require_core_scopes: bool = True,
) -> dict[str, Any]:
    """Validate integer-nanosecond completed clocks and disjoint core scopes."""

    materialized = _clock_rows(rows)
    if not materialized:
        raise ClockError("CLOCK_ROWS_EMPTY")
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, row in enumerate(materialized):
        scope = row.get("scope", row.get("name"))
        if not isinstance(scope, str) or not scope:
            raise ClockError(f"CLOCK_SCOPE_REQUIRED:{index}")
        if scope in names:
            raise ClockError(f"CLOCK_SCOPE_DUPLICATE:{scope}")
        names.add(scope)
        if row.get("completed") is not True:
            raise ClockError(f"CLOCK_NOT_COMPLETED:{scope}")
        start = _integer(row.get("start_ns"), f"{scope}.start_ns")
        end = _integer(row.get("end_ns"), f"{scope}.end_ns")
        if start < 0 or end < 0 or end < start:
            raise ClockError(f"CLOCK_RANGE_INVALID:{scope}")
        duration = end - start
        if "duration_ns" in row and _integer(row["duration_ns"], f"{scope}.duration_ns") != duration:
            raise ClockError(f"CLOCK_DURATION_MISMATCH:{scope}")
        normalized.append({"scope": scope, "start_ns": start, "end_ns": end, "duration_ns": duration})
    by_scope = {row["scope"]: row for row in normalized}
    required = ("window", "interval1", "later")
    if require_core_scopes and any(name not in by_scope for name in required):
        raise ClockError("CLOCK_CORE_SCOPES_REQUIRED")
    if all(name in by_scope for name in required):
        window, first, later = (by_scope[name] for name in required)
        for child in (first, later):
            if child["start_ns"] < window["start_ns"] or child["end_ns"] > window["end_ns"]:
                raise ClockError(f"CLOCK_SCOPE_OUTSIDE_WINDOW:{child['scope']}")
        if first["start_ns"] < later["end_ns"] and later["start_ns"] < first["end_ns"]:
            raise ClockError("CLOCK_INTERVAL1_LATER_OVERLAP")
    # Validate explicit parent links and sibling disjointness without clipping.
    children: dict[str, list[dict[str, Any]]] = {}
    for row in materialized:
        parent = row.get("parent")
        if parent is not None:
            if parent not in by_scope:
                raise ClockError(f"CLOCK_PARENT_MISSING:{row.get('scope')}")
            children.setdefault(str(parent), []).append(by_scope[str(row["scope"])])
    for parent, siblings in children.items():
        parent_row = by_scope[parent]
        ordered = sorted(siblings, key=lambda item: (item["start_ns"], item["end_ns"]))
        for child in ordered:
            if child["start_ns"] < parent_row["start_ns"] or child["end_ns"] > parent_row["end_ns"]:
                raise ClockError(f"CLOCK_CHILD_OUTSIDE_PARENT:{child['scope']}")
        for left, right in zip(ordered, ordered[1:]):
            if left["end_ns"] > right["start_ns"]:
                raise ClockError(f"CLOCK_SIBLING_OVERLAP:{parent}")
    return {
        "status": "PASS_CLOCK_RECORDS",
        "scopes": normalized,
        "durations_ns": {row["scope"]: row["duration_ns"] for row in normalized},
        "default": "I0",
        "production_qualified": False,
        "DEFAULT": "I0",
        "PRODUCTION_QUALIFIED": False,
    }


def expected_case_order(_backend: str | None = None, _shape: Sequence[int] | None = None) -> tuple[str, ...]:
    """Return the frozen eight-case order; backend/shape are descriptive only."""

    return CASE_ORDER


def validate_case_order(rows: Sequence[Mapping[str, Any]], *, backend: str | None = None, shape: Sequence[int] | None = None) -> dict[str, Any]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise RecordError("CASE_ROWS_REQUIRED")
    actual = tuple(row.get("case") for row in rows)
    if actual != CASE_ORDER:
        raise RecordError(f"CASE_ORDER_MISMATCH:{actual}")
    if backend is not None and any(row.get("backend") not in (None, backend) for row in rows):
        raise RecordError("CASE_BACKEND_MISMATCH")
    if shape is not None:
        expected_shape = tuple(int(value) for value in shape)
        for row in rows:
            if row.get("shape") is not None and tuple(row["shape"]) != expected_shape:
                raise RecordError("CASE_SHAPE_MISMATCH")
    return {
        "status": "PASS_CASE_ORDER",
        "backend": backend,
        "shape": list(shape) if shape is not None else None,
        "cases": list(CASE_ORDER),
        "default": "I0",
        "production_qualified": False,
        "DEFAULT": "I0",
        "PRODUCTION_QUALIFIED": False,
    }


def _finite_positive(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TimingGateError(f"TIMING_NUMBER_REQUIRED:{field}") from exc
    if not math.isfinite(number) or number <= 0:
        raise TimingGateError(f"TIMING_FINITE_POSITIVE_REQUIRED:{field}")
    return number


def timing_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = ("application", "window", "later"),
    required_pairs: Sequence[str] = ("P1", "P2", "P3"),
    median_bounds: tuple[float, float] = (0.95, 1.05),
    each_bounds: tuple[float, float] = (0.90, 1.10),
) -> dict[str, Any]:
    """Compute LIGHT/OFF observer ratios and apply the preregistered gate."""

    if not rows:
        raise TimingGateError("TIMING_ROWS_EMPTY")
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for index, row in enumerate(rows):
        raw_case = str(row.get("kind", row.get("case", ""))).upper()
        pair = str(row.get("pair", row.get("batch", "")))
        kind = raw_case
        if kind not in {"OFF", "LIGHT"} and kind.endswith(("_OFF", "_LIGHT")):
            suffix = "_OFF" if kind.endswith("_OFF") else "_LIGHT"
            kind = suffix[1:]
            if not pair:
                pair = raw_case[: -len(suffix)]
        if not pair or kind not in {"OFF", "LIGHT"}:
            raise TimingGateError(f"TIMING_PAIR_KIND_REQUIRED:{index}")
        if kind in grouped.setdefault(pair, {}):
            raise TimingGateError(f"TIMING_DUPLICATE_PAIR_KIND:{pair}:{kind}")
        grouped[pair][kind] = row
    expected_pairs = tuple(str(pair) for pair in required_pairs)
    if set(grouped) != set(expected_pairs) or len(grouped) != len(expected_pairs):
        raise TimingGateError(f"TIMING_REQUIRED_PAIRS_MISMATCH:{tuple(sorted(grouped))}")
    ratios: list[dict[str, Any]] = []
    for pair, values in grouped.items():
        if set(values) != {"OFF", "LIGHT"}:
            raise TimingGateError(f"TIMING_PAIR_INCOMPLETE:{pair}")
        ratio_row: dict[str, Any] = {"pair": pair}
        for metric in metrics:
            off = _finite_positive(values["OFF"].get(f"{metric}_s", values["OFF"].get(metric)), f"{pair}.OFF.{metric}")
            light = _finite_positive(values["LIGHT"].get(f"{metric}_s", values["LIGHT"].get(metric)), f"{pair}.LIGHT.{metric}")
            ratio = light / off
            ratio_row[metric] = ratio
            ratio_row[f"{metric}_off_s"] = off
            ratio_row[f"{metric}_light_s"] = light
        ratios.append(ratio_row)
    summaries: dict[str, Any] = {}
    all_pass = True
    for metric in metrics:
        values = [float(row[metric]) for row in ratios]
        med = float(median(values))
        each_pass = all(each_bounds[0] <= value <= each_bounds[1] for value in values)
        median_pass = median_bounds[0] <= med <= median_bounds[1]
        summaries[metric] = {"ratios": values, "median": med, "each_pass": each_pass, "median_pass": median_pass, "pass": each_pass and median_pass}
        all_pass = all_pass and each_pass and median_pass
    return {
        "status": "PASS_TIMING_GATE" if all_pass else "TIMING_GATE_FAIL",
        "pairs": ratios,
        "metrics": summaries,
        "median_bounds": list(median_bounds),
        "each_bounds": list(each_bounds),
        "default": "I0",
        "production_qualified": False,
        "DEFAULT": "I0",
        "PRODUCTION_QUALIFIED": False,
    }


_VALID_STATES = {
    "completed",
    "cancelled",
    "failed",
    "pending",
    "running",
    "configuring",
    "completing",
    "inflight",
}
_INFLIGHT_STATES = {"pending", "running", "configuring", "completing", "inflight"}


def _normalise_job(row: Mapping[str, Any], resource: str, *, is_new: bool = False) -> dict[str, Any]:
    value = dict(row)
    category = str(value.get("category", "")).lower()
    allowed = {"batch", "extern", "repair"} | ({"small", "large"} if resource == "CPU" else {"main"})
    if category not in allowed:
        raise BudgetError(f"BUDGET_CATEGORY_UNSUPPORTED:{category}")
    if is_new and not value.get("state"):
        value["state"] = "pending"
    state = str(value.get("state", "")).lower()
    if state not in _VALID_STATES:
        raise BudgetError(f"BUDGET_STATE_UNSUPPORTED:{state}")
    value["category"] = category
    value["state"] = state
    return value


def _job_chargeable(row: Mapping[str, Any]) -> bool:
    category = str(row.get("category", "")).lower()
    if category in {"batch", "extern"}:
        return False
    return bool(row.get("chargeable", category not in {"batch", "extern"}))


def _job_id(row: Mapping[str, Any], index: int) -> str:
    value = row.get("job_id", row.get("id"))
    if value is None or str(value) == "":
        raise BudgetError(f"BUDGET_JOB_ID_REQUIRED:{index}")
    return str(value)


def _hours(row: Mapping[str, Any], *, resource: str, worst: bool) -> tuple[float, float]:
    wall = row.get("worst_walltime_s" if worst else "actual_walltime_s", row.get("walltime_s"))
    if wall is None:
        raise BudgetError("BUDGET_WALLTIME_REQUIRED")
    try:
        wall_s = _finite_positive(wall, "budget.walltime_s")
    except TimingGateError as exc:
        try:
            wall_s = float(wall)
        except (TypeError, ValueError) as parse_error:
            raise BudgetError(str(exc)) from parse_error
        if not math.isfinite(wall_s) or wall_s < 0 or (wall_s == 0 and row.get("state") != "cancelled"):
            raise BudgetError(str(exc)) from exc
    expected_cpus = float(BUDGET_CONTRACTS[resource]["cpus"])
    expected_gpus = float(BUDGET_CONTRACTS[resource].get("gpus", 0))
    try:
        cpus = float(row.get("cpus", expected_cpus))
        gpus = float(row.get("gpus", expected_gpus))
    except (TypeError, ValueError) as exc:
        raise BudgetError("BUDGET_RESOURCE_COUNT_INVALID") from exc
    if not math.isfinite(cpus) or not math.isfinite(gpus) or cpus != expected_cpus or gpus != expected_gpus:
        raise BudgetError("BUDGET_RESOURCE_COUNT_MISMATCH")
    core_hours = wall_s * cpus / 3600.0
    gpu_hours = wall_s * gpus / 3600.0
    return core_hours, gpu_hours


def cpu_completed_metrics(snapshot: Mapping[str, Any], intervals: int) -> dict[str, Any]:
    """Read the real S45B merged schema, with one completed propagation shell."""
    if intervals not in (2, 8):
        raise ClockError("CPU_COMPLETED_INTERVAL_COUNT")
    try:
        records = snapshot["merged"]["records"]
        bodies = sorted((row for row in records if row.get("label") == "INTERVAL_BODY"), key=lambda row: row["start_wall_ns"])
        windows = [row for row in records if row.get("label") == "PROPAGATION"]
        indices = [row["raw_context"]["interval_index"] for row in bodies]
    except (KeyError, TypeError) as exc:
        raise ClockError("CPU_MERGED_TIMING_SCHEMA") from exc
    if indices != list(range(intervals)) or len(windows) != 1:
        raise ClockError("CPU_COMPLETED_WINDOW_OR_INTERVALS_MISSING")
    window = windows[0]
    for row in [window, *bodies]:
        keys = ("start_wall_ns", "end_wall_ns", "start_cpu_ns", "end_cpu_ns", "wall_inclusive_ns", "cpu_inclusive_ns")
        if any(type(row.get(key)) is not int for key in keys) or row.get("failed"):
            raise ClockError("CPU_INCOMPLETE_INTEGER_CLOCK")
        if row["end_wall_ns"] - row["start_wall_ns"] != row["wall_inclusive_ns"] or row["end_cpu_ns"] - row["start_cpu_ns"] != row["cpu_inclusive_ns"]:
            raise ClockError("CPU_RAW_CLOCK_ARITHMETIC")
        if row["wall_inclusive_ns"] <= 0 or row["cpu_inclusive_ns"] < 0:
            raise ClockError("CPU_NEGATIVE_COMPLETED_CLOCK")
    if window["start_wall_ns"] > bodies[0]["start_wall_ns"] or window["end_wall_ns"] < bodies[-1]["end_wall_ns"]:
        raise ClockError("CPU_BODY_OUTSIDE_COMPLETED_WINDOW")
    if any(left["end_wall_ns"] > right["start_wall_ns"] for left, right in zip(bodies, bodies[1:])):
        raise ClockError("CPU_INTERVAL_OVERLAP")
    return dict(window_completed_s=window["wall_inclusive_ns"] / 1e9,
                window_process_cpu_s=window["cpu_inclusive_ns"] / 1e9,
                first_interval_s=bodies[0]["wall_inclusive_ns"] / 1e9,
                later_body_sum_s=sum(row["wall_inclusive_ns"] for row in bodies[1:]) / 1e9,
                later_body_s=(bodies[-1]["end_wall_ns"] - bodies[1]["start_wall_ns"]) / 1e9,
                completed_clock_basis="RAW_S45B_MERGED_PROPAGATION_AND_RAW_CONTEXT_INTERVALS")


def budget_check(
    ledger: Sequence[Mapping[str, Any]],
    resource: str,
    *,
    new_request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a CPU/GPU ledger against the fixed S4-6A resource contract."""

    key = str(resource).upper()
    if key not in BUDGET_CONTRACTS:
        raise BudgetError(f"BUDGET_RESOURCE_UNSUPPORTED:{resource}")
    contract = BUDGET_CONTRACTS[key]
    if not isinstance(ledger, Sequence) or isinstance(ledger, (str, bytes)):
        raise BudgetError("BUDGET_LEDGER_REQUIRED")
    if any(not isinstance(row, Mapping) for row in ledger):
        raise BudgetError("BUDGET_JOB_ROW_INVALID")
    rows = [_normalise_job(row, key) for row in ledger]
    charged = [row for row in rows if _job_chargeable(row)]
    ids = [_job_id(row, index) for index, row in enumerate(charged)]
    if len(set(ids)) != len(ids):
        raise BudgetError("BUDGET_DUPLICATE_JOB_ID")
    reasons: list[str] = []
    if len(ids) > int(contract["max_job_ids"]):
        reasons.append("MAX_JOB_IDS")
    inflight = [row for row in charged if row["state"] in _INFLIGHT_STATES]
    if len(inflight) > int(contract["max_inflight"]):
        reasons.append("MAX_INFLIGHT")
    for row in charged:
        category = str(row.get("category", "")).lower()
        wall = row.get("walltime_s", row.get("worst_walltime_s"))
        if wall is None:
            reasons.append(f"WALLTIME_REQUIRED:{row.get('job_id')}")
            continue
        try:
            wall_s = float(wall)
        except (TypeError, ValueError) as exc:
            raise BudgetError(f"WALLTIME_INVALID:{row.get('job_id')}") from exc
        if not math.isfinite(wall_s) or wall_s < 0 or (wall_s == 0 and row["state"] != "cancelled"):
            raise BudgetError(f"WALLTIME_INVALID:{row.get('job_id')}")
        if row["state"] == "cancelled" and wall_s == 0:
            pass
        elif wall_s <= 0:
            reasons.append(f"WALLTIME_INVALID:{row.get('job_id')}")
        if key == "CPU" and category in contract["max_walltime_s"] and wall_s > contract["max_walltime_s"][category]:
            reasons.append(f"WALLTIME_CAP:{row.get('job_id')}")
        if key == "GPU" and category in contract["max_walltime_s"] and wall_s > contract["max_walltime_s"][category]:
            reasons.append(f"WALLTIME_CAP:{row.get('job_id')}")
    consumed_core = consumed_gpu = 0.0
    for row in charged:
        worst = row["state"] in _INFLIGHT_STATES
        core, gpu = _hours(row, resource=key, worst=worst)
        consumed_core += core
        consumed_gpu += gpu
    if new_request is not None:
        if not isinstance(new_request, Mapping):
            raise BudgetError("BUDGET_NEW_REQUEST_INVALID")
        request = _normalise_job(new_request, key, is_new=True)
        new_request_reserved = False
        if _job_chargeable(request):
            request_id = str(request.get("job_id", "")).strip()
            new_request_reserved = True
            if request_id and request_id in ids:
                reasons.append("NEW_DUPLICATE_JOB_ID")
            request["worst_walltime_s"] = request.get("worst_walltime_s", request.get("walltime_s"))
            new_core, new_gpu = _hours(request, resource=key, worst=True)
            consumed_core += new_core
            consumed_gpu += new_gpu
            category = str(request.get("category", "")).lower()
            cap = contract["max_walltime_s"].get(category)
            try:
                wall = float(request["worst_walltime_s"])
            except (TypeError, ValueError) as exc:
                raise BudgetError("NEW_WALLTIME_INVALID") from exc
            if cap is not None and wall > cap:
                reasons.append("NEW_WALLTIME_CAP")
            if len(ids) + 1 > int(contract["max_job_ids"]):
                reasons.append("NEW_MAX_JOB_IDS")
            if request["state"] in _INFLIGHT_STATES and len(inflight) + 1 > int(contract["max_inflight"]):
                reasons.append("NEW_MAX_INFLIGHT")
    else:
        new_request_reserved = False
    max_core = float(contract["max_core_hours"])
    if consumed_core > max_core + 1e-12:
        reasons.append("CORE_HOURS_BUDGET")
    max_gpu = contract.get("max_gpu_hours")
    if max_gpu is not None and consumed_gpu > float(max_gpu) + 1e-12:
        reasons.append("GPU_HOURS_BUDGET")
    return {
        "status": "PASS_BUDGET" if not reasons else "BUDGET_BLOCKED",
        "resource": key,
        "charged_job_ids": ids,
        "ignored_categories": ["batch", "extern"],
        "inflight_job_ids": [_job_id(row, index) for index, row in enumerate(inflight)],
        "new_request_reserved": new_request_reserved,
        "core_hours_worst_total": consumed_core,
        "gpu_hours_worst_total": consumed_gpu,
        "limits": dict(contract),
        "reasons": reasons,
        "default": "I0",
        "production_qualified": False,
        "DEFAULT": "I0",
        "PRODUCTION_QUALIFIED": False,
    }
