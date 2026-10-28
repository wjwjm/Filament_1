"""Small, read-only S46A evidence replay.

This module deliberately stays independent of the production package.  It
replays only the fixed S46A metadata schema (matrix result JSON, raw timing
JSON, manifests, and resource ledgers); it never opens scientific arrays and
never writes to the report directory during normal operation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "filament.cpu_s4_6a.replay.v1"
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
PAIRS = ("P1", "P2", "P3")
METRICS = ("application", "window", "later")
MAX_METADATA_BYTES = 8 * 1024 * 1024
SKIP_DIRS = {".git", "__pycache__", "cache", "outputs", "figures", "tmp"}
ARRAY_SUFFIXES = {".npy", ".npz", ".mat", ".h5", ".hdf5"}
DEFAULT_CURRENT_SCOPE = ("tmp/windows_micro_r2",)
SCOPE_KEYS = {
    "datasets": ("datasets", "matrix", "matrix_results"),
    "timings": ("current_raw_timings", "raw_timings", "timings", "raw"),
    "manifests": ("portable_metadata_manifest", "portable_manifests", "manifests", "source_input_manifests"),
    "ledgers": ("ledger", "resource_ledger", "ledgers", "scheduler"),
}


class ReplayError(ValueError):
    """Malformed replay metadata, never a production-science exception."""


def _duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ReplayError(f"DUPLICATE_JSON_KEY:{key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> Any:
    raise ReplayError(f"NONFINITE_JSON_CONSTANT:{value}")


def _finite_tree(value: Any, path: str = "$", *, max_nodes: int = 200_000) -> None:
    """Reject non-finite values, including overflow parsed from ``1e999``."""

    count = 0

    def visit(item: Any, location: str) -> None:
        nonlocal count
        count += 1
        if count > max_nodes:
            raise ReplayError("JSON_METADATA_TOO_LARGE")
        if isinstance(item, float) and not math.isfinite(item):
            raise ReplayError(f"NONFINITE_JSON_VALUE:{location}")
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f"{location}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{location}[{index}]")

    visit(value, path)


def load_json_strict(path: Path, *, max_bytes: int = MAX_METADATA_BYTES) -> Any:
    """Load a small UTF-8-sig JSON metadata file with strict semantics."""

    size = path.stat().st_size
    if size > max_bytes:
        raise ReplayError(f"METADATA_TOO_LARGE:{path.name}:{size}")
    try:
        text = path.read_text(encoding="utf-8-sig")
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except ReplayError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReplayError(f"JSON_READ_FAILED:{path.name}:{exc}") from exc
    _finite_tree(value)
    return value


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ReplayError(f"NUMBER_REQUIRED:{field}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReplayError(f"NUMBER_REQUIRED:{field}") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise ReplayError(f"FINITE_{'POSITIVE_' if positive else ''}NUMBER_REQUIRED:{field}")
    return number


def _integer_ns(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayError(f"INTEGER_NS_REQUIRED:{field}")
    if value < 0:
        raise ReplayError(f"NEGATIVE_NS:{field}")
    return value


def _report_root(root: os.PathLike[str] | str) -> Path:
    path = Path(root)
    if not path.exists() or not path.is_dir():
        raise ReplayError(f"REPORT_ROOT_REQUIRED:{root}")
    if path.is_symlink():
        raise ReplayError("REPORT_ROOT_SYMLINK_REJECTED")
    return path.absolute()


def _scope_values(data: Mapping[str, Any], kind: str) -> list[str]:
    for key in SCOPE_KEYS[kind]:
        value = data.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ReplayError(f"REPLAY_SCOPE_LIST_REQUIRED:{key}")
        return list(value)
    return []


def _scope_paths(base: Path) -> dict[str, Any]:
    """Return an explicit report-local whitelist, never a broad tree scan."""

    scope_path = base / "REPLAY_SCOPE.json"
    if scope_path.is_file():
        data = load_json_strict(scope_path)
        if not isinstance(data, Mapping):
            raise ReplayError("REPLAY_SCOPE_OBJECT_REQUIRED")
        configured = True
    else:
        data = {}
        configured = False
    selected: dict[str, list[Path]] = {kind: [] for kind in SCOPE_KEYS}
    issues: list[str] = []
    for kind in SCOPE_KEYS:
        values = _scope_values(data, kind) if configured else []
        if not configured and kind in {"timings", "manifests"}:
            values = list(DEFAULT_CURRENT_SCOPE)
        for raw in values:
            try:
                path = _resolve_qualified(base, _relative_parts(raw))
            except ReplayError as exc:
                issues.append(f"{kind}:{raw}:{exc}")
                continue
            if path.is_dir():
                # Explicit directory entries are allowed, but discovery remains
                # schema/name limited and does not follow symlinks or arrays.
                for child in _walk_report(path):
                    if child.suffix.lower() in ARRAY_SUFFIXES:
                        continue
                    selected[kind].append(child)
            elif path.is_file():
                selected[kind].append(path)
            else:
                issues.append(f"{kind}:{raw}:PATH_MISSING")
    # A default scope selects only current micro metadata; matrix/ledger files
    # are intentionally absent until a package supplies an explicit whitelist.
    if not configured:
        selected["timings"] = [
            path for path in selected["timings"]
            if path.name.upper() in {"RAW_CPU_TIMING.JSON", "CASE_RECEIPT.JSON"}
        ]
        selected["manifests"] = [
            path for path in selected["manifests"]
            if path.name.upper() in {"INPUT_MANIFEST.JSON", "REPLAY_SCOPE.JSON"}
        ]
    for kind in selected:
        selected[kind] = sorted({path for path in selected[kind] if not path.is_symlink()}, key=str)
    return {
        "path": "REPLAY_SCOPE.json" if configured else None,
        "configured": configured,
        "selected": {kind: [path.relative_to(base).as_posix() for path in paths] for kind, paths in selected.items()},
        "issues": sorted(set(issues)),
        "paths": selected,
    }


def _relative_parts(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str) or not value.strip():
        raise ReplayError("RELATIVE_PATH_REQUIRED")
    text = value.replace("\\", "/")
    if text.startswith("/") or text.startswith("//") or re.match(r"^[A-Za-z]:", text):
        raise ReplayError(f"ABSOLUTE_PATH_REJECTED:{value}")
    if text.startswith("./"):
        text = text[2:]
    parts = tuple(text.split("/"))
    if any(not part or part in {".", ".."} for part in parts):
        raise ReplayError(f"UNSAFE_RELATIVE_PATH:{value}")
    return parts


def _walk_report(root: Path) -> Iterable[Path]:
    """Yield report files while excluding old run trees and array outputs."""

    for directory, dirs, files in os.walk(root, topdown=True, followlinks=False):
        dirs[:] = sorted(
            name for name in dirs
            if name not in SKIP_DIRS and not (Path(directory) / name).is_symlink()
        )
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_symlink():
                continue
            yield path


def _resolve_qualified(root: Path, parts: Sequence[str]) -> Path:
    current = root
    for part in parts:
        if current.is_symlink():
            raise ReplayError("SYMLINK_COMPONENT_REJECTED")
        current = current / part
        if current.is_symlink():
            raise ReplayError("SYMLINK_COMPONENT_REJECTED")
    root_resolved = root.resolve()
    candidate_resolved = current.resolve(strict=False)
    try:
        inside = os.path.commonpath((str(root_resolved), str(candidate_resolved))) == str(root_resolved)
    except ValueError:
        inside = False
    if not inside:
        raise ReplayError("PATH_ESCAPES_REPORT_ROOT")
    return current


def resolve_report_path(root: os.PathLike[str] | str, value: Any) -> Path:
    """Resolve a qualified relative path or an unambiguous basename.

    Exact qualified paths are allowed to share a basename.  Basename fallback
    is intentionally strict and rejects both missing and ambiguous matches.
    """

    base = _report_root(root)
    explicit_root_path = isinstance(value, str) and value.replace("\\", "/").startswith("./")
    parts = _relative_parts(value)
    if len(parts) == 1 and not explicit_root_path:
        matches = [path for path in _walk_report(base) if path.name == parts[0]]
        if not matches:
            raise ReplayError(f"PATH_MISSING:{value}")
        if len(matches) != 1:
            raise ReplayError(f"BASENAME_AMBIGUOUS:{value}")
        return _resolve_qualified(base, matches[0].relative_to(base).parts)
    return _resolve_qualified(base, parts)


def sha256_raw(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest_records(
    root: os.PathLike[str] | str,
    records: Sequence[Mapping[str, Any]],
    *,
    manifest_kind: str = "source_or_input",
    record_root: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Validate packaged metadata records without reading scientific arrays."""

    base = _report_root(root)
    location = base if record_root is None else _report_root(record_root)
    if not location.is_relative_to(base):
        raise ReplayError("MANIFEST_RECORD_ROOT_OUTSIDE_REPORT_ROOT")
    result: dict[str, Any] = {
        "manifest_kind": manifest_kind,
        "records": [],
        "issues": [],
        "status": "INCOMPLETE",
        "arrays_not_rechecked": 0,
    }
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        result["issues"].append("MANIFEST_RECORDS_REQUIRED")
        return result
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            result["issues"].append(f"RECORD_NOT_OBJECT:{index}")
            continue
        raw_path = row.get("path", row.get("relative_path", row.get("file")))
        checked: dict[str, Any] = {"index": index, "path": raw_path}
        try:
            if not isinstance(raw_path, str):
                raise ReplayError("RELATIVE_PATH_REQUIRED")
            path_parts = _relative_parts(raw_path)
            suffix = Path(raw_path.replace("\\", "/")).suffix.lower()
            if suffix in ARRAY_SUFFIXES or isinstance(row.get("array_names"), list):
                # The review package may retain a manifest while canonical
                # arrays remain on the producing host.  Never read them here.
                checked["qualified_path"] = "/".join(path_parts)
                checked["status"] = "NOT_RECHECKED_ARRAY"
                checked["reason"] = "SCIENTIFIC_ARRAY_BYTES_OUT_OF_REPLAY_SCOPE"
                result["arrays_not_rechecked"] += 1
                result["records"].append(checked)
                continue
            resolved = resolve_report_path(location, raw_path)
            checked["qualified_path"] = resolved.relative_to(base).as_posix() if resolved.is_relative_to(base) else str(resolved)
            if not resolved.is_file():
                raise ReplayError("PATH_MISSING")
            checked["exists"] = True
            expected = row.get("raw_sha256", row.get("sha256", row.get("hash")))
            scope = row.get("hash_scope")
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                raise ReplayError("RAW_SHA256_REQUIRED")
            if scope != "raw_bytes":
                raise ReplayError("RAW_HASH_SCOPE_REQUIRED")
            actual = sha256_raw(resolved)
            checked["raw_sha256"] = actual
            if actual.lower() != expected.lower():
                raise ReplayError("RAW_SHA256_MISMATCH")
            checked["status"] = "VALID"
        except (ReplayError, OSError) as exc:
            checked["status"] = "INVALID"
            checked["error"] = str(exc)
            result["issues"].append(f"{index}:{exc}")
        result["records"].append(checked)
    if result["records"] and not result["issues"]:
        result["status"] = "MANIFEST_METADATA_REPLAYED" if result["arrays_not_rechecked"] else "MANIFEST_REPLAYED"
    return result


def _name_matches(path: Path, kind: str) -> bool:
    upper = path.name.upper()
    if path.suffix.lower() not in {".json", ".jsonl", ".csv"}:
        return False
    if kind == "matrix":
        return upper.startswith("MATRIX_RESULT") and path.suffix.lower() == ".json"
    if kind == "timing":
        return (upper.startswith("RAW_CPU_TIMING") or upper.startswith("RAW_GPU_TIMING")) and path.suffix.lower() == ".json"
    if kind == "ledger":
        return (
            ("RESOURCE" in upper and "LEDGER" in upper)
            or "SCHEDULER" in upper
            or upper.startswith("SLURM_")
            or upper.startswith("JOB_RECEIPT")
        )
    if kind == "manifest":
        return "MANIFEST" in upper or "SOURCE_RECORD" in upper or "INPUT_RECORD" in upper
    return False


def _discover(root: Path, kind: str) -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    skipped: list[str] = []
    for path in _walk_report(root):
        if not _name_matches(path, kind):
            continue
        if kind == "manifest":
            relative_parts = {part.upper() for part in path.relative_to(root).parts[:-1]}
            name_upper = path.name.upper()
            # The review replay consumes exported raw/input manifests.  Old
            # deployment source indexes are repository-wide references and
            # would only create thousands of report-root-missing rows here.
            allowed_named = {"INPUT_MANIFEST.JSON", "SOURCE_MANIFEST.JSON", "SOURCE_RECORDS.JSON", "INPUT_RECORDS.JSON"}
            if "DEPLOY" in relative_parts or name_upper.startswith("EXECUTION_SOURCE_MANIFEST"):
                continue
            if not ({"RAW", "EXTERNAL_DATA"} & relative_parts) and name_upper not in allowed_named:
                continue
        if path.stat().st_size > MAX_METADATA_BYTES:
            skipped.append(f"METADATA_TOO_LARGE:{path.relative_to(root).as_posix()}")
            continue
        paths.append(path)
    return paths, skipped


def _value_from(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    containers: list[Mapping[str, Any]] = [row]
    for key in ("timings", "metrics", "timing", "measurements"):
        value = row.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    for container in containers:
        for name in names:
            if name in container:
                value = container[name]
                if isinstance(value, Mapping):
                    for inner in ("s", "seconds", "value"):
                        if inner in value:
                            return value[inner]
                return value
    return None


_METRIC_NAMES = {
    "application": ("application_service_s", "application_s", "application"),
    "window": ("window_completed_s", "optical_window_completed_s", "window_s", "window"),
    "later": ("later_body_s", "later_body_sum_s", "later_s", "later_body"),
}


def _metric(row: Mapping[str, Any], metric: str) -> tuple[float | None, str | None]:
    raw = _value_from(row, _METRIC_NAMES[metric])
    if raw is None:
        return None, f"MISSING_{metric}_S"
    try:
        return _number(raw, metric, positive=True), None
    except ReplayError as exc:
        return None, str(exc)


def recompute_timing_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute the fixed three-pair bilateral LIGHT/OFF gate."""

    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    issues: list[str] = []
    for index, row in enumerate(rows):
        case = str(row.get("case", "")).upper()
        match = re.fullmatch(r"(P[123])_(OFF|LIGHT)", case)
        if not match:
            continue
        pair, kind = match.groups()
        if kind in grouped.setdefault(pair, {}):
            issues.append(f"DUPLICATE_CASE:{case}")
        else:
            grouped[pair][kind] = row
    missing_pairs = [pair for pair in PAIRS if set(grouped.get(pair, {})) != {"OFF", "LIGHT"}]
    issues.extend(f"MISSING_PAIR:{pair}" for pair in missing_pairs)
    pair_results: list[dict[str, Any]] = []
    for pair in PAIRS:
        value: dict[str, Any] = {"pair": pair}
        if pair not in grouped or set(grouped[pair]) != {"OFF", "LIGHT"}:
            pair_results.append(value)
            continue
        for metric in METRICS:
            off, off_error = _metric(grouped[pair]["OFF"], metric)
            light, light_error = _metric(grouped[pair]["LIGHT"], metric)
            if off_error:
                issues.append(f"{pair}.OFF.{metric}:{off_error}")
            if light_error:
                issues.append(f"{pair}.LIGHT.{metric}:{light_error}")
            if off is not None and light is not None:
                value[f"{metric}_off_s"] = off
                value[f"{metric}_light_s"] = light
                ratio = light / off
                if not math.isfinite(ratio) or ratio <= 0:
                    issues.append(f"{pair}.{metric}:NONFINITE_RATIO")
                else:
                    value[metric] = ratio
        pair_results.append(value)
    summaries: dict[str, Any] = {}
    complete = not issues and len(pair_results) == len(PAIRS)
    for metric in METRICS:
        values = [row[metric] for row in pair_results if metric in row]
        each_pass = len(values) == 3 and all(0.90 <= item <= 1.10 for item in values)
        median_value = statistics.median(values) if len(values) == 3 else None
        median_pass = median_value is not None and 0.95 <= median_value <= 1.05
        summaries[metric] = {
            "ratios": values,
            "median": median_value,
            "each_pass": each_pass,
            "median_pass": median_pass,
            "pass": bool(each_pass and median_pass),
        }
        if len(values) != 3:
            complete = False
    gate_pass = complete and all(item["pass"] for item in summaries.values())
    return {
        "status": "PASS_TIMING_GATE" if gate_pass else ("INCOMPLETE" if not complete else "TIMING_GATE_FAIL"),
        "pairs": pair_results,
        "metrics": summaries,
        "median_bounds": [0.95, 1.05],
        "each_bounds": [0.90, 1.10],
        "missing": sorted(set(issues)),
    }


def off_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return three OFF values, median, range, and MAD for every metric."""

    by_pair = {str(row.get("case", "")).upper(): row for row in rows}
    result: dict[str, Any] = {"status": "COMPLETE", "metrics": {}, "missing": []}
    for metric in METRICS:
        values: list[float] = []
        errors: list[str] = []
        for pair in PAIRS:
            row = by_pair.get(f"{pair}_OFF")
            if row is None:
                errors.append(f"MISSING_{pair}_OFF")
                continue
            value, error = _metric(row, metric)
            if error or value is None:
                errors.append(f"{pair}_OFF.{metric}:{error}")
            else:
                values.append(value)
        if len(values) != 3:
            result["status"] = "INCOMPLETE"
            result["missing"].extend(errors or [f"MISSING_THREE_OFF_VALUES:{metric}"])
            result["metrics"][metric] = {"values_s": values, "count": len(values)}
            continue
        median_value = float(statistics.median(values))
        range_value = max(values) - min(values)
        mad_value = float(statistics.median([abs(item - median_value) for item in values]))
        if not math.isfinite(range_value) or not math.isfinite(mad_value):
            result["status"] = "INCOMPLETE"
            result["missing"].append(f"NONFINITE_OFF_STATISTICS:{metric}")
        result["metrics"][metric] = {
            "values_s": values,
            "count": 3,
            "median_s": median_value,
            "min_s": min(values),
            "max_s": max(values),
            "range_s": range_value,
            "mad_s": mad_value,
        }
    return result


def _shape_id(data: Mapping[str, Any], path: Path) -> str:
    shape = data.get("shape")
    if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)) and shape:
        try:
            return "shape:" + "x".join(str(int(value)) for value in shape)
        except (TypeError, ValueError):
            pass
    labels = [part.upper() for part in path.parts]
    for label in reversed(labels):
        if label in {"CPU_SMALL", "GPU_SMALL", "SMALL"}:
            return "SMALL"
        if label in {"CPU_FULL", "GPU_FULL", "FULL"}:
            return "FULL"
    return "FULL"


def _backend(data: Mapping[str, Any], path: Path) -> str:
    value = str(data.get("backend", "")).upper()
    if value in {"CPU", "GPU"}:
        return value
    upper = path.as_posix().upper()
    if "GPU" in upper:
        return "GPU"
    if "CPU" in upper:
        return "CPU"
    return "UNKNOWN"


def _case_rows(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("cases", "rows", "records", "results"):
        value = data.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, Mapping)]
        if isinstance(value, Mapping):
            result: list[Mapping[str, Any]] = []
            for case, row in value.items():
                if isinstance(row, Mapping):
                    item = dict(row)
                    item.setdefault("case", case)
                    result.append(item)
            return result
    return []


def _replay_matrix_file(root: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    data = load_json_strict(path)
    if not isinstance(data, Mapping):
        raise ReplayError("MATRIX_RESULT_OBJECT_REQUIRED")
    backend = _backend(data, path)
    shape = _shape_id(data, path)
    source_rows = _case_rows(data)
    rows: list[dict[str, Any]] = []
    order: list[str] = []
    issues: list[str] = []
    for row in source_rows:
        case = str(row.get("case", row.get("name", ""))).upper()
        if case not in CASE_ORDER:
            continue
        order.append(case)
        item: dict[str, Any] = {"case": case}
        for metric in METRICS:
            value, error = _metric(row, metric)
            if value is not None:
                item[f"{metric}_s"] = value
            if error:
                item[f"{metric}_error"] = error
        rows.append(item)
    if tuple(order) != CASE_ORDER:
        issues.append("CASE_ORDER_MISMATCH")
    gate = recompute_timing_gate(rows)
    off = off_statistics(rows)
    issues.extend(gate["missing"])
    issues.extend(off["missing"])
    return {
        "path": relative,
        "backend": backend,
        "shape": shape,
        "case_order": order,
        "case_order_ok": tuple(order) == CASE_ORDER,
        "cases": rows,
        "timing_gate": gate,
        "off_statistics": off,
        "issues": sorted(set(issues)),
        "status": "REPLAYED" if not issues else "INCOMPLETE",
    }


def _sample_id(path: Path) -> str:
    for part in reversed(path.parts):
        match = re.fullmatch(r"test_s46a_accepted_l1_to_off_(t\d+)", part, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return path.parent.name.upper()


def _case_ns_pair(clocks: Mapping[str, Any], *, start: str, end: str) -> tuple[int | None, int | None, str | None]:
    try:
        first = _integer_ns(clocks.get(start), f"application_clocks.{start}")
        last = _integer_ns(clocks.get(end), f"application_clocks.{end}")
    except ReplayError as exc:
        return None, None, str(exc)
    if last < first:
        return first, last, "APPLICATION_CLOCK_RANGE_INVALID"
    return first, last, None


def _replay_case_metadata(root: Path, case_path: Path) -> dict[str, Any]:
    relative = case_path.relative_to(root).as_posix()
    issues: list[str] = []
    receipt = load_json_strict(case_path)
    if not isinstance(receipt, Mapping):
        raise ReplayError("CASE_RECEIPT_OBJECT_REQUIRED")
    raw_path = case_path.with_name("RAW_CPU_TIMING.json")
    raw_data: Mapping[str, Any] | None = None
    if raw_path.is_file() and not raw_path.is_symlink():
        raw_value = load_json_strict(raw_path)
        if isinstance(raw_value, Mapping):
            raw_data = raw_value
        else:
            issues.append("RAW_TIMING_OBJECT_REQUIRED")
    else:
        issues.append("RAW_TIMING_MISSING")
    clocks = receipt.get("application_clocks")
    if not isinstance(clocks, Mapping):
        issues.append("APPLICATION_CLOCKS_MISSING")
        app_s = None
    else:
        app_start, app_end, error = _case_ns_pair(clocks, start="start_ns", end="application_end_ns")
        if error:
            issues.append(error)
        app_s = None if app_start is None or app_end is None else (app_end - app_start) / 1e9
    closure: dict[str, Any] = {"status": "PENDING", "row_count": 0, "scopes": {}, "issues": ["RAW_TIMING_MISSING"]}
    raw_rows: list[dict[str, Any]] = []
    raw_kind = "missing"
    if raw_data is not None:
        raw_kind = "merged" if _merged_clock_rows(raw_data) else "gpu" if _gpu_clock_rows(raw_data) else "generic"
        raw_rows = _merged_clock_rows(raw_data) if raw_kind == "merged" else _gpu_clock_rows(raw_data)
        expected_intervals = receipt.get("intervals")
        if isinstance(expected_intervals, bool) or not isinstance(expected_intervals, int) or expected_intervals < 1:
            expected_intervals = None
            issues.append("INTERVAL_COUNT_MISSING")
        closure = validate_clock_closure(raw_rows, expected_intervals=expected_intervals) if raw_rows else _raw_timing_replay(raw_data)
        if closure["status"] != "PASS_CLOCK_CLOSURE":
            issues.extend(f"CLOCK:{issue}" for issue in closure.get("issues", []))
    interval_rows = [
        row for row in raw_rows
        if str(row.get("label", "")).upper() == "INTERVAL_BODY"
        and isinstance(row.get("interval_index"), int)
        and not isinstance(row.get("interval_index"), bool)
    ]
    interval_rows.sort(key=lambda row: int(row["interval_index"]))
    metrics: dict[str, float | None] = {"application": app_s, "window": None, "later": None}
    metric_sources: dict[str, str] = {"application": "CASE.application_clocks"}
    propagation = [row for row in raw_rows if str(row.get("label", "")).upper() == "PROPAGATION"]
    if len(propagation) == 1:
        row = propagation[0]
        try:
            start = _integer_ns(row.get("start_ns"), "PROPAGATION.start_ns")
            end = _integer_ns(row.get("end_ns"), "PROPAGATION.end_ns")
            if end < start:
                raise ReplayError("PROPAGATION_CLOCK_RANGE_INVALID")
            metrics["window"] = (end - start) / 1e9
            metric_sources["window"] = "RAW.merged.PROPAGATION.host_wall_ns"
        except ReplayError as exc:
            issues.append(str(exc))
    elif raw_kind == "merged":
        issues.append("PROPAGATION_SCOPE_MISSING_NOT_APPLICATION_SUBSTITUTE")
    elif raw_kind == "gpu":
        # The window is the explicit PROPAGATION host interval.  A CASE
        # receipt's legacy/application duration is not a substitute when an
        # OFF or incomplete GPU record omits that scope.
        issues.append("PROPAGATION_SCOPE_MISSING_NOT_APPLICATION_SUBSTITUTE")
    elif raw_kind == "missing":
        issues.append("PROPAGATION_SCOPE_UNAVAILABLE")
    expected = receipt.get("intervals")
    if isinstance(expected, int) and not isinstance(expected, bool) and expected >= 1:
        observed = [int(row["interval_index"]) for row in interval_rows]
        if observed != list(range(expected)):
            issues.append(f"INTERVAL_SEQUENCE_MISMATCH:{observed}:{list(range(expected))}")
        later = [row for row in interval_rows if int(row["interval_index"]) >= 1]
        if expected > 1 and len(later) == expected - 1:
            try:
                start = _integer_ns(later[0].get("start_ns"), "later.start_ns")
                end = _integer_ns(later[-1].get("end_ns"), "later.end_ns")
                if end < start:
                    raise ReplayError("LATER_CLOCK_RANGE_INVALID")
                metrics["later"] = (end - start) / 1e9
                metric_sources["later"] = (
                    f"RAW.merged.INTERVAL_BODY.ns[start interval_index=1,end interval_index={expected - 1}]"
                    if raw_kind == "merged" else f"RAW.records.host_wall_ns[start interval=1,end interval={expected - 1}]"
                )
            except ReplayError as exc:
                issues.append(str(exc))
        elif expected > 1:
            issues.append("LATER_INTERVALS_MISSING")
    else:
        issues.append("LATER_INTERVAL_COUNT_UNKNOWN")
    device_spans = [row.get("device_event_span_ms") for row in raw_rows if row.get("device_event_span_ms") is not None]
    return {
        "path": relative,
        "raw_timing_path": raw_path.relative_to(root).as_posix() if raw_path.is_file() else None,
        "sample_id": _sample_id(case_path),
        "backend": str(receipt.get("backend", "UNKNOWN")).upper(),
        "shape": _shape_id(receipt, case_path),
        "case": str(receipt.get("case", receipt.get("mode", "UNKNOWN"))).upper(),
        "mode": str(receipt.get("mode", "UNKNOWN")).upper(),
        "intervals": expected,
        "metrics_s": metrics,
        "metric_sources": metric_sources,
        "legacy_application_service_s": receipt.get("legacy_application_service_s"),
        "case_optical_compute_window_s": receipt.get("optical_compute_window_s"),
        "application_process_cpu_s": receipt.get("application_process_cpu_s"),
        "clock_closure": closure,
        "deferred_device_event_spans_ms": device_spans,
        "issues": sorted(set(issues)),
        "status": "REPLAYED" if not issues else "INCOMPLETE",
        "DEFAULT": "I0",
        "PRODUCTION_QUALIFIED": False,
        "MANUAL_REVIEW_PENDING": True,
    }


def _metadata_case_replay(root: Path, paths: Sequence[Path], skipped: Sequence[str]) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    issues = list(skipped)
    for path in paths:
        if path.name.upper() != "CASE_RECEIPT.JSON":
            continue
        # The accepted micro evidence is the paired OFF/LIGHT view.  The
        # REFERENCE tree has the same receipt vocabulary but is not one of
        # the current timing cases and must not be silently promoted here.
        if path.parent.name.upper() not in {"OFF", "LIGHT"}:
            continue
        try:
            run = _replay_case_metadata(root, path)
            runs.append(run)
            issues.extend(f"{run['path']}:{issue}" for issue in run["issues"])
        except (ReplayError, OSError) as exc:
            issues.append(f"{path.relative_to(root).as_posix()}:{exc}")
    return {
        "status": "REPLAYED" if runs and not issues else ("INCOMPLETE" if runs else "PENDING"),
        "runs": runs,
        "issues": sorted(set(issues)),
    }


def _clock_candidates(value: Any, *, location: str = "$") -> Iterable[dict[str, Any]]:
    """Find clock rows in the small raw timing vocabulary."""

    if isinstance(value, Mapping):
        direct = (
            ("start_ns" in value and "end_ns" in value)
            or ("start_wall_ns" in value and "end_wall_ns" in value)
        )
        if direct:
            start = value.get("start_ns", value.get("start_wall_ns"))
            end = value.get("end_ns", value.get("end_wall_ns"))
            label = value.get("scope", value.get("name", value.get("label")))
            if not label:
                label = re.split(r"[.\[]", location.rsplit(".", 1)[-1], maxsplit=1)[0]
            context = value.get("context", value.get("raw_context"))
            interval_index = value.get("interval", value.get("interval_index"))
            if interval_index is None and isinstance(context, Mapping):
                interval_index = context.get("interval_index")
            identifier = value.get("id", value.get("token", value.get("merged_token", value.get("scope", label or location))))
            parent = value.get("parent", value.get("parent_token", value.get("merged_parent_token", value.get("raw_parent_token"))))
            completed = value.get("completed") is True
            if "completed" not in value and str(value.get("status", "")).lower() in {"complete", "completed"} and value.get("failed") is not True:
                completed = True
            yield {
                "id": str(identifier),
                "parent": None if parent is None else str(parent),
                "label": str(label or ""),
                "interval_index": interval_index,
                "completed": completed,
                "start_ns": start,
                "end_ns": end,
                "location": location,
            }
            # The normalized_interval nested under a raw row repeats the same
            # clock; do not count it twice when an outer row already had clocks.
            children = ((key, child) for key, child in value.items() if key != "normalized_interval")
        else:
            children = value.items()
        for key, child in children:
            if isinstance(child, (Mapping, list)):
                yield from _clock_candidates(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, (Mapping, list)):
                yield from _clock_candidates(child, location=f"{location}[{index}]")


def validate_clock_closure(
    rows: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    *,
    expected_intervals: int | None = None,
) -> dict[str, Any]:
    """Validate integer-ns completion, parent containment, and sibling closure."""

    if isinstance(rows, Mapping):
        rows = [dict(value, scope=key) for key, value in rows.items() if isinstance(value, Mapping)]
    normalized: list[dict[str, Any]] = []
    issues: list[str] = []
    for index, row in enumerate(rows):
        try:
            start = _integer_ns(row.get("start_ns"), f"{index}.start_ns")
            end = _integer_ns(row.get("end_ns"), f"{index}.end_ns")
            if end < start:
                raise ReplayError(f"CLOCK_RANGE_INVALID:{index}")
        except ReplayError as exc:
            issues.append(str(exc))
            continue
        if row.get("completed") is not True:
            issues.append(f"CLOCK_NOT_COMPLETED:{index}")
        identifier = str(row.get("id", row.get("merged_token", index)))
        parent = row.get("parent", row.get("merged_parent_token"))
        normalized.append({
            **dict(row),
            "id": identifier,
            "parent": None if parent is None else str(parent),
            "start_ns": start,
            "end_ns": end,
            "duration_ns": end - start,
        })
    ids: dict[str, dict[str, Any]] = {}
    for row in normalized:
        if row["id"] in ids:
            issues.append(f"CLOCK_DUPLICATE_ID:{row['id']}")
        ids[row["id"]] = row
        for alias in (row.get("label"), row.get("scope")):
            alias_text = str(alias or "")
            if alias_text and alias_text.lower() not in ids:
                ids[alias_text.lower()] = row
    children: dict[str, list[dict[str, Any]]] = {}
    for row in normalized:
        parent = row.get("parent")
        if parent is None:
            continue
        parent_row = ids.get(str(parent)) or ids.get(str(parent).lower())
        if parent_row is None:
            issues.append(f"CLOCK_PARENT_MISSING:{row['id']}")
            continue
        if row["start_ns"] < parent_row["start_ns"] or row["end_ns"] > parent_row["end_ns"]:
            issues.append(f"CLOCK_CHILD_OUTSIDE_PARENT:{row['id']}")
        children.setdefault(str(parent), []).append(row)
    for parent, siblings in children.items():
        ordered = sorted(siblings, key=lambda item: (item["start_ns"], item["end_ns"], item["id"]))
        for left, right in zip(ordered, ordered[1:]):
            if left["end_ns"] > right["start_ns"]:
                issues.append(f"CLOCK_SIBLING_OVERLAP:{parent}")

    window: dict[str, Any] | None = None
    first: dict[str, Any] | None = None
    later_rows: list[dict[str, Any]] = []
    propagation = [
        row for row in normalized
        if str(row.get("label", row.get("scope", ""))).upper() == "PROPAGATION"
    ]
    if propagation:
        window = max(propagation, key=lambda item: item["duration_ns"])
    for row in normalized:
        label = str(row.get("label", row.get("scope", ""))).upper()
        scope = label.replace("-", "_").lower()
        if window is None and scope in {"window", "optical_window"}:
            if window is None or row["duration_ns"] > window["duration_ns"]:
                window = row
        index = row.get("interval_index")
        valid_interval = isinstance(index, int) and not isinstance(index, bool)
        if scope in {"interval1", "interval_1"} or (label == "INTERVAL_BODY" and valid_interval and index == 0):
            first = row
        if scope == "later" or (label == "INTERVAL_BODY" and valid_interval and index >= 1):
            later_rows.append(row)
    core: dict[str, Any] = {}
    if window is not None:
        core["window"] = {"start_ns": window["start_ns"], "end_ns": window["end_ns"], "duration_ns": window["duration_ns"]}
    if first is not None:
        core["interval1"] = {"start_ns": first["start_ns"], "end_ns": first["end_ns"], "duration_ns": first["duration_ns"]}
    if later_rows:
        start = min(row["start_ns"] for row in later_rows)
        end = max(row["end_ns"] for row in later_rows)
        core["later"] = {"start_ns": start, "end_ns": end, "duration_ns": end - start, "interval_count": len(later_rows)}
    for name, child in (("interval1", first), ("later", core.get("later"))):
        if window is not None and child is not None:
            start = child["start_ns"] if isinstance(child, Mapping) else child["start_ns"]
            end = child["end_ns"] if isinstance(child, Mapping) else child["end_ns"]
            if start < window["start_ns"] or end > window["end_ns"]:
                issues.append(f"CLOCK_CORE_OUTSIDE_WINDOW:{name}")
    if first is not None and later_rows:
        if first["end_ns"] > min(row["start_ns"] for row in later_rows):
            issues.append("CLOCK_INTERVAL1_LATER_OVERLAP")
    if expected_intervals is not None:
        if isinstance(expected_intervals, bool) or expected_intervals < 1:
            issues.append("EXPECTED_INTERVAL_COUNT_INVALID")
        else:
            observed = sorted(
                row.get("interval_index")
                for row in normalized
                if isinstance(row.get("interval_index"), int)
                and not isinstance(row.get("interval_index"), bool)
                and str(row.get("label", "")).upper() == "INTERVAL_BODY"
            )
            expected = list(range(int(expected_intervals)))
            if observed != expected:
                issues.append(f"CLOCK_INTERVAL_SEQUENCE_MISMATCH:{observed}:{expected}")
            if expected_intervals > 1 and len(later_rows) != expected_intervals - 1:
                issues.append("CLOCK_LATER_INTERVAL_COUNT_MISMATCH")
    required = {"window", "interval1", "later"}
    status = "PASS_CLOCK_CLOSURE" if not issues and required <= set(core) else ("CLOCK_INVALID" if issues else "PARTIAL_CLOCK_CLOSURE")
    return {
        "status": status,
        "row_count": len(normalized),
        "scopes": core,
        "issues": sorted(set(issues)),
    }


def _merged_clock_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    merged = data.get("merged")
    if not isinstance(merged, Mapping) or not isinstance(merged.get("records"), list):
        return []
    rows: list[dict[str, Any]] = []
    for row in merged["records"]:
        if not isinstance(row, Mapping):
            continue
        normalized = row.get("normalized_interval")
        normalized = normalized if isinstance(normalized, Mapping) else {}
        context = row.get("raw_context")
        context = context if isinstance(context, Mapping) else {}
        interval = normalized.get("interval_index", context.get("interval_index"))
        completed = row.get("completed") is True
        if "completed" not in row:
            completed = str(row.get("status", "")).lower() in {"complete", "completed"} and row.get("failed") is not True
        rows.append({
            "id": row.get("merged_token", row.get("raw_token")),
            "parent": row.get("merged_parent_token"),
            "label": row.get("label", ""),
            "interval_index": interval,
            "completed": completed,
            "start_ns": row.get("start_wall_ns", normalized.get("start_wall_ns")),
            "end_ns": row.get("end_wall_ns", normalized.get("end_wall_ns")),
            "wall_inclusive_ns": row.get("wall_inclusive_ns"),
            "wall_exclusive_ns": row.get("wall_exclusive_ns"),
            "origin": row.get("origin"),
        })
    return rows


def _gpu_clock_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = data.get("records")
    if not isinstance(records, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in records:
        if not isinstance(row, Mapping):
            continue
        context = row.get("context")
        context = context if isinstance(context, Mapping) else {}
        completed = row.get("completed") is True
        if "completed" not in row:
            completed = row.get("failed") is not True and "end_wall_ns" in row
        rows.append({
            "id": row.get("token"),
            "parent": row.get("parent", row.get("parent_token")),
            "label": row.get("label", ""),
            "interval_index": row.get("interval", row.get("interval_index", context.get("interval_index"))),
            "completed": completed,
            "start_ns": row.get("start_wall_ns"),
            "end_ns": row.get("end_wall_ns"),
            "wall_ns": row.get("wall_ns"),
            "device_event_span_ms": row.get("device_event_span_ms"),
        })
    return rows


def _raw_timing_replay(data: Any) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        return {"status": "INCOMPLETE", "row_count": 0, "scopes": {}, "issues": ["TIMING_OBJECT_REQUIRED"]}
    merged_rows = _merged_clock_rows(data)
    if merged_rows:
        summary = validate_clock_closure(merged_rows)
        summary["clock_source"] = "merged.records"
        summary["overlap_views_ignored"] = [name for name in ("b", "s45a") if isinstance(data.get(name), Mapping)]
        return summary
    gpu_rows = _gpu_clock_rows(data)
    if gpu_rows:
        summary = validate_clock_closure(gpu_rows)
        summary["clock_source"] = "records.host_wall_ns"
        spans = [row.get("device_event_span_ms") for row in gpu_rows if row.get("device_event_span_ms") is not None]
        summary["deferred_device_event_spans_ms"] = spans
        summary["device_spans_not_subtracted_from_host"] = True
        return summary
    candidates = list(_clock_candidates(data))
    if not candidates:
        return {"status": "PENDING", "row_count": 0, "scopes": {}, "issues": ["CLOCK_ROWS_MISSING"]}
    summary = validate_clock_closure(candidates)
    summary["clock_source"] = "generic_metadata"
    return summary


def _clock_replay(root: Path, paths: Sequence[Path], skipped: Sequence[str]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    issues = list(skipped)
    for path in paths:
        relative = path.relative_to(root).as_posix()
        try:
            data = load_json_strict(path)
            summary = _raw_timing_replay(data)
            files.append({"path": relative, **summary})
            if summary["status"] != "PASS_CLOCK_CLOSURE":
                issues.extend(f"{relative}:{item}" for item in summary["issues"] or [summary["status"]])
        except ReplayError as exc:
            files.append({"path": relative, "status": "INCOMPLETE", "row_count": 0, "scopes": {}, "issues": [str(exc)]})
            issues.append(f"{relative}:{exc}")
    return {"status": "REPLAYED" if files and not issues else ("INCOMPLETE" if files else "PENDING"), "files": files, "issues": sorted(set(issues))}


_RESOURCE_LIMITS = {
    "CPU": {"ids": 3, "core_hours": 240.0, "cpus": 96, "gpus": 0, "categories": {"small": 1200.0, "large": 5400.0, "repair": 5400.0, "main": 5400.0}},
    "GPU": {"ids": 2, "gpu_hours": 2.0, "core_hours": 16.0, "cpus": 8, "gpus": 1, "categories": {"main": 5400.0, "repair": 1800.0}},
}


def _ledger_rows(data: Any) -> list[Mapping[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, Mapping)]
    if isinstance(data, Mapping):
        new_request = data.get("new_request", data.get("new"))
        for key in ("rows", "records", "jobs", "ledger", "resources"):
            value = data.get(key)
            if isinstance(value, list):
                rows = [row for row in value if isinstance(row, Mapping)]
                if isinstance(new_request, Mapping):
                    item = dict(new_request)
                    item.setdefault("new", True)
                    item.setdefault("state", "pending")
                    rows.append(item)
                return rows
        for key in ("backend_ledgers", "by_backend"):
            value = data.get(key)
            if isinstance(value, Mapping):
                rows: list[Mapping[str, Any]] = []
                for backend, entries in value.items():
                    if isinstance(entries, list):
                        for row in entries:
                            if isinstance(row, Mapping):
                                item = dict(row)
                                item.setdefault("backend", backend)
                                rows.append(item)
                if rows:
                    if isinstance(new_request, Mapping):
                        item = dict(new_request)
                        item.setdefault("new", True)
                        item.setdefault("state", "pending")
                        rows.append(item)
                    return rows
        markers = {"job_id", "jobid", "state", "category", "cpus", "gpus", "walltime_s", "actual_walltime_s"}
        if any(key in data for key in ("backend", "resource")) and any(key in data for key in markers):
            return [data]
        if isinstance(new_request, Mapping):
            item = dict(new_request)
            item.setdefault("new", True)
            item.setdefault("state", "pending")
            return [item]
    return []


def _field(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
        for key, value in row.items():
            if str(key).lower() == name.lower():
                return value
    return None


def _normalize_sacct_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Map a raw sacct CSV row to the small ledger vocabulary."""

    if _field(row, "backend", "resource") is not None:
        return row
    job_id = _field(row, "JobIDRaw", "JobID")
    job_name = _field(row, "JobName")
    if job_id is None and job_name is None:
        return row
    name = str(job_name or "").lower()
    backend = "GPU" if "gpu" in name or "gres/gpu" in str(_field(row, "AllocTRES", "AllocGRES") or "").lower() else "CPU"
    category = "repair" if "repair" in name else "small" if "small" in name else "large" if "large" in name else "main"
    state = str(_field(row, "State", "Status") or "").split()[0].lower()
    alloc_gres = str(_field(row, "AllocTRES", "AllocGRES") or "").lower()
    gpu_count = 1 if backend == "GPU" else 0
    gpu_match = re.search(r"gpu(?::|=)(\d+)", alloc_gres)
    if gpu_match:
        gpu_count = int(gpu_match.group(1))
    return {
        "backend": backend,
        "job_id": job_id,
        "job_name": job_name,
        "category": category,
        "state": state,
        "actual_walltime_s": _field(row, "ElapsedRaw", "Elapsed"),
        "worst_walltime_s": _field(row, "TimelimitRaw", "Timelimit"),
        "cpus": _field(row, "AllocCPUS", "CPUS"),
        "gpus": gpu_count,
    }


def _duration_seconds(value: Any, field: str) -> float:
    if isinstance(value, str) and ":" in value:
        text = value.strip()
        day = 0
        if "-" in text.split(" ", 1)[0]:
            prefix, text = text.split("-", 1)
            try:
                day = int(prefix)
            except ValueError as exc:
                raise ReplayError(f"DURATION_REQUIRED:{field}") from exc
        parts = text.split(":")
        try:
            if len(parts) == 3:
                seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                seconds = int(parts[0]) * 60 + float(parts[1])
            else:
                raise ValueError
        except ValueError as exc:
            raise ReplayError(f"DURATION_REQUIRED:{field}") from exc
        return _number(day * 86400 + seconds, field)
    return _number(value, field)


def replay_resource_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute CPU/GPU actual and worst-case resource ledgers."""

    result: dict[str, Any] = {"status": "INCOMPLETE", "backends": {}, "issues": []}
    grouped: dict[str, list[Mapping[str, Any]]] = {"CPU": [], "GPU": []}
    for row in rows:
        backend = str(_field(row, "backend", "resource")).upper()
        if backend in grouped:
            grouped[backend].append(row)
        else:
            result["issues"].append("UNKNOWN_BACKEND")
    complete = True
    for backend, entries in grouped.items():
        limits = _RESOURCE_LIMITS[backend]
        ids: set[str] = set()
        pre_id = 0
        inflight = 0
        actual_core = actual_gpu = 0.0
        inflight_core = inflight_gpu = 0.0
        new_worst_core = new_worst_gpu = 0.0
        backend_issues: list[str] = []
        seen_main_ids: set[str] = set()
        for index, row in enumerate(entries):
            job_id_raw = _field(row, "job_id", "jobid", "job_id_raw", "id", "JobIDRaw")
            job_id_text = "" if job_id_raw is None else str(job_id_raw).strip()
            if any(job_id_text.lower().endswith(suffix) for suffix in (".batch", ".extern")):
                continue
            if "." in job_id_text and not (
                _field(row, "new", "prospective", "new_request") is True
                or str(_field(row, "new", "prospective", "new_request")).lower() in {"new", "prospective", "true"}
            ):
                # sacct step rows (123.batch, 123.extern, 123.0) are not a
                # second allocation and must never be charged again.
                continue
            category = str(_field(row, "category", "job_category")).lower().replace("-", "_")
            state = str(_field(row, "state", "status", "job_state")).lower()
            if category in {"batch", "extern", "external"}:
                continue
            if category.startswith("cpu_") or category.startswith("gpu_"):
                category = category.split("_", 1)[1]
            if category not in limits["categories"]:
                backend_issues.append(f"UNKNOWN_CATEGORY:{index}")
                continue
            if state not in {"completed", "cancelled", "failed", "pending", "running", "configuring", "completing", "inflight"}:
                backend_issues.append(f"UNKNOWN_STATE:{index}")
                continue
            role = str(_field(row, "charge_role", "role", "kind")).lower()
            new_marker = _field(row, "new", "prospective", "new_request")
            is_new = new_marker is True or str(new_marker).lower() in {"new", "prospective", "true"} or role in {"new", "prospective", "new_request"}
            cpus_raw = _field(row, "cpus", "cpu_count", "requested_cpus", "alloc_cpus")
            gpus_raw = _field(row, "gpus", "gpu_count", "requested_gpus", "alloc_gpus")
            if cpus_raw is None or gpus_raw is None:
                backend_issues.append(f"RESOURCE_COUNTS_MISSING:{index}")
                continue
            try:
                cpus = _number(cpus_raw, f"{backend}.{index}.cpus")
                gpus = _number(gpus_raw, f"{backend}.{index}.gpus")
            except ReplayError as exc:
                backend_issues.append(str(exc))
                continue
            if cpus != limits["cpus"] or gpus != limits["gpus"]:
                backend_issues.append(f"RESOURCE_COUNTS_MISMATCH:{index}")
            job_id = job_id_text
            if not job_id or job_id.lower() in {"none", "null", "pending"}:
                pre_id += 1
            else:
                if job_id in seen_main_ids:
                    backend_issues.append(f"DUPLICATE_MAIN_JOB_ID:{job_id}")
                    continue
                seen_main_ids.add(job_id)
                ids.add(str(job_id))
            if state in {"pending", "running", "configuring", "completing", "inflight"}:
                inflight += 1
            actual_raw = _field(row, "actual_walltime_s", "elapsed_s", "elapsed", "walltime_s", "duration_s", "runtime_s")
            if state == "cancelled" and actual_raw is None:
                actual = 0.0
            elif state in {"pending", "running", "configuring", "completing", "inflight"} and actual_raw is None:
                actual = 0.0
            elif actual_raw is None:
                backend_issues.append(f"ACTUAL_WALLTIME_MISSING:{index}")
                continue
            else:
                try:
                    actual = _duration_seconds(actual_raw, f"{backend}.{index}.actual_walltime_s")
                except ReplayError as exc:
                    backend_issues.append(str(exc))
                    continue
            if actual < 0:
                backend_issues.append(f"NEGATIVE_WALLTIME:{index}")
            if state in {"completed", "failed"} and actual <= 0:
                backend_issues.append(f"NONCANCELLED_ZERO_WALLTIME:{index}")
            if actual > limits["categories"][category]:
                backend_issues.append(f"WALLTIME_CAP_EXCEEDED:{index}")
            worst_raw = _field(row, "worst_walltime_s", "requested_walltime_s", "time_limit_s", "walltime_limit_s")
            if state in {"pending", "running", "configuring", "completing", "inflight"}:
                if worst_raw is None:
                    worst = limits["categories"][category]
                else:
                    try:
                        worst = _duration_seconds(worst_raw, f"{backend}.{index}.worst_walltime_s")
                    except ReplayError as exc:
                        backend_issues.append(str(exc))
                        continue
                if worst > limits["categories"][category]:
                    backend_issues.append(f"WORST_WALLTIME_CAP_EXCEEDED:{index}")
            else:
                if worst_raw is None:
                    worst = actual
                else:
                    try:
                        worst = _duration_seconds(worst_raw, f"{backend}.{index}.worst_walltime_s")
                    except ReplayError as exc:
                        backend_issues.append(str(exc))
                        continue
            if worst < 0:
                backend_issues.append(f"NEGATIVE_WORST_WALLTIME:{index}")
            if worst > limits["categories"][category]:
                backend_issues.append(f"WORST_WALLTIME_CAP_EXCEEDED:{index}")
            if not is_new and state not in {"pending", "running", "configuring", "completing", "inflight"}:
                actual_core += actual * cpus / 3600.0
                actual_gpu += actual * gpus / 3600.0
            if is_new:
                new_worst_core += worst * cpus / 3600.0
                new_worst_gpu += worst * gpus / 3600.0
            elif state in {"pending", "running", "configuring", "completing", "inflight"}:
                inflight_core += worst * cpus / 3600.0
                inflight_gpu += worst * gpus / 3600.0
        if len(ids) > limits["ids"]:
            backend_issues.append(f"JOB_ID_CAP_EXCEEDED:{len(ids)}")
        if inflight > 1:
            backend_issues.append(f"INFLIGHT_CAP_EXCEEDED:{inflight}")
        core_limit = float(limits["core_hours"])
        gpu_limit = float(limits.get("gpu_hours", 0.0))
        worst_core = actual_core + inflight_core
        worst_gpu = actual_gpu + inflight_gpu
        worst_core += new_worst_core
        worst_gpu += new_worst_gpu
        if worst_core > core_limit + 1e-12:
            backend_issues.append("CORE_HOUR_BUDGET_EXCEEDED")
        if gpu_limit and worst_gpu > gpu_limit + 1e-12:
            backend_issues.append("GPU_HOUR_BUDGET_EXCEEDED")
        result["backends"][backend] = {
            "job_ids": sorted(ids),
            "pre_id_rows": pre_id,
            "inflight_count": inflight,
            "actual_core_hours": actual_core,
            "inflight_worst_core_hours": inflight_core,
            "new_worst_core_hours": new_worst_core,
            "worst_case_core_hours": worst_core,
            "core_hour_limit": core_limit,
            "actual_gpu_hours": actual_gpu,
            "inflight_worst_gpu_hours": inflight_gpu,
            "new_worst_gpu_hours": new_worst_gpu,
            "worst_case_gpu_hours": worst_gpu,
            "gpu_hour_limit": gpu_limit,
            "issues": sorted(set(backend_issues)),
            "status": "REPLAYED" if not backend_issues else "INCOMPLETE",
        }
        result["issues"].extend(f"{backend}:{issue}" for issue in backend_issues)
        complete = complete and not backend_issues
    if not rows:
        result["status"] = "PENDING"
    elif result["issues"]:
        result["status"] = "BUDGET_BLOCKED" if any("BUDGET" in item or "CAP_EXCEEDED" in item for item in result["issues"]) else "INCOMPLETE"
    else:
        result["status"] = "REPLAYED" if complete else "INCOMPLETE"
    result["issues"] = sorted(set(result["issues"]))
    return result


def _sacct_rows_from_text(text: str) -> list[Mapping[str, Any]]:
    """Decode the fixed raw sacct fields used by the S46A scheduler receipt."""

    rows: list[Mapping[str, Any]] = []
    for line in text.splitlines():
        fields = [part.strip() for part in line.split("|")]
        if len(fields) < 6 or not re.match(r"^\d+(?:\.\d+|\.(?:batch|extern))?$", fields[0]):
            continue
        job_id, job_name, state, _exit_code, elapsed, alloc_cpus = fields[:6]
        name = job_name.lower()
        backend = "GPU" if "gpu" in name else "CPU"
        category = "repair" if "repair" in name else "small" if "small" in name else "large" if "large" in name else "main"
        rows.append({
            "backend": backend,
            "job_id": job_id,
            "category": category,
            "state": state.split()[0].lower(),
            "actual_walltime_s": elapsed,
            "cpus": alloc_cpus,
            "gpus": 1 if backend == "GPU" else 0,
        })
    return rows


def _sacct_rows_from_value(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, str):
        return _sacct_rows_from_text(value)
    if isinstance(value, Mapping):
        rows: list[Mapping[str, Any]] = []
        for child in value.values():
            rows.extend(_sacct_rows_from_value(child))
        return rows
    if isinstance(value, list):
        rows: list[Mapping[str, Any]] = []
        for child in value:
            rows.extend(_sacct_rows_from_value(child))
        return rows
    return []


def _resource_replay(root: Path, paths: Sequence[Path], skipped: Sequence[str]) -> dict[str, Any]:
    rows: list[Mapping[str, Any]] = []
    issues = list(skipped)
    files: list[str] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        try:
            if path.suffix.lower() == ".csv":
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    parsed = list(csv.DictReader(handle))
                rows.extend(_normalize_sacct_row(row) for row in parsed if isinstance(row, Mapping))
            elif path.suffix.lower() in {".log", ".txt"}:
                text = path.read_text(encoding="utf-8", errors="replace")
                try:
                    # Scheduler receipts commonly wrap raw sacct stdout in a
                    # small JSON object.  Decode that wrapper first so the
                    # main JobID row is charged once while batch/extern/step
                    # rows remain excluded by replay_resource_rows().
                    value = load_json_strict(path)
                except ReplayError:
                    rows.extend(_sacct_rows_from_text(text))
                else:
                    rows.extend(_sacct_rows_from_value(value))
            else:
                value = load_json_strict(path)
                parsed = _ledger_rows(value)
                rows.extend(parsed if parsed else _sacct_rows_from_value(value))
            files.append(relative)
        except (OSError, ReplayError, csv.Error) as exc:
            issues.append(f"{relative}:{exc}")
    summary = replay_resource_rows(rows)
    summary["files"] = files
    summary["issues"] = sorted(set(issues + summary["issues"]))
    if issues and summary["status"] == "REPLAYED":
        summary["status"] = "INCOMPLETE"
    return summary


def _historical_replay(root: Path) -> dict[str, Any]:
    path = root / "historical" / "GPU_S4_RUNTIME_EVENTS_ORIGINAL.csv"
    if not path.is_file() or path.is_symlink():
        return {"status": "PENDING", "issues": ["HISTORICAL_CSV_MISSING"]}
    rows: list[dict[str, str]] = []
    issues: list[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if isinstance(row, dict)]
    except (OSError, UnicodeError, csv.Error) as exc:
        return {"status": "INCOMPLETE", "issues": [f"HISTORICAL_CSV_READ_FAILED:{exc}"]}
    events: list[tuple[str, float, dict[str, str]]] = []
    for index, row in enumerate(rows):
        try:
            value = _number(row.get("monotonic_s"), f"historical.{index}.monotonic_s")
        except ReplayError as exc:
            issues.append(str(exc))
            continue
        events.append((str(row.get("event", "")), value, row))
    posts = sorted((item for item in events if item[0] == "POST_COMMITTED"), key=lambda item: item[1])
    starts = [value for event, value, _ in events if event == "OPTICAL_START"]
    completes = [value for event, value, _ in events if event == "OPTICAL_COMPLETE"]
    backpressure = [value for event, value, _ in events if event == "PRODUCER_BACKPRESSURE_BEGIN"]
    backpressure_end = [value for event, value, _ in events if event == "PRODUCER_BACKPRESSURE_END"]
    if not posts:
        issues.append("HISTORICAL_POST_ROWS_MISSING")
    elif len(posts) != 48:
        issues.append(f"HISTORICAL_POST_COUNT_EXPECTED_48:{len(posts)}")
    selected_span = None
    screens_per_s = None
    if posts:
        selected_span = posts[-1][1] - posts[0][1]
        if selected_span <= 0:
            issues.append("HISTORICAL_POST_SPAN_INVALID")
        else:
            screens_per_s = len(posts) / selected_span
    active_s = None
    if starts and completes:
        active_s = completes[0] - starts[0]
        if active_s < 0:
            issues.append("HISTORICAL_ACTIVE_SPAN_INVALID")
    backpressure_s = None
    if backpressure and backpressure_end:
        backpressure_s = backpressure_end[0] - backpressure[0]
        if backpressure_s < 0:
            issues.append("HISTORICAL_BACKPRESSURE_SPAN_INVALID")
    receipt_path = root / "historical" / "HISTORICAL_RECOMPUTATION.json"
    receipt = None
    if receipt_path.is_file() and not receipt_path.is_symlink():
        try:
            receipt = load_json_strict(receipt_path)
            if isinstance(receipt, Mapping):
                expected = receipt.get("export_raw_sha256")
                actual = sha256_raw(path)
                if isinstance(expected, str) and expected.lower() != actual.lower():
                    issues.append("HISTORICAL_CSV_RAW_SHA256_MISMATCH")
        except (ReplayError, OSError) as exc:
            issues.append(f"HISTORICAL_RECEIPT_UNAVAILABLE:{exc}")
    return {
        "status": "REPLAYED" if not issues and posts and active_s is not None else "INCOMPLETE",
        "path": path.relative_to(root).as_posix(),
        "raw_sha256": sha256_raw(path),
        "post_count": len(posts),
        "first_post_s": posts[0][1] if posts else None,
        "last_post_s": posts[-1][1] if posts else None,
        "selected_post_span_s": selected_span,
        "selected_screens_per_s": screens_per_s,
        "optical_start_s": starts[0] if starts else None,
        "optical_complete_s": completes[0] if completes else None,
        "optical_active_s": active_s,
        "selected_backpressure_s": backpressure_s,
        "formula": "post_count / (last_post_s - first_post_s)",
        "receipt_status": receipt.get("status") if isinstance(receipt, Mapping) else None,
        "issues": sorted(set(issues)),
        "historical_not_matched_speedup": True,
    }


def _manifest_records(data: Any) -> list[Mapping[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, Mapping) and any(key in row for key in ("path", "relative_path", "file"))]
    if not isinstance(data, Mapping):
        return []
    for key in ("records", "source_records", "input_records", "files", "manifest"):
        value = data.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, Mapping) and any(name in row for name in ("path", "relative_path", "file"))]
    return []


def _manifest_replay(root: Path, paths: Sequence[Path], skipped: Sequence[str]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    issues = list(skipped)
    for path in paths:
        relative = path.relative_to(root).as_posix()
        try:
            records = _manifest_records(load_json_strict(path))
            if not records:
                continue
            checked = validate_manifest_records(
                root,
                records,
                manifest_kind=path.stem,
                record_root=path.parent,
            )
            files.append({"path": relative, **checked})
            issues.extend(f"{relative}:{issue}" for issue in checked["issues"])
        except ReplayError as exc:
            issues.append(f"{relative}:{exc}")
    arrays_not_rechecked = sum(int(item.get("arrays_not_rechecked", 0)) for item in files)
    metadata_partial = arrays_not_rechecked > 0
    status = (
        "MANIFEST_METADATA_ONLY"
        if files and not issues and metadata_partial
        else "MANIFEST_REPLAYED"
        if files and not issues
        else "INCOMPLETE"
        if files
        else "PENDING"
    )
    return {"status": status, "files": files, "issues": sorted(set(issues)), "arrays_not_rechecked": arrays_not_rechecked}


def _cpu_gpu_ratios(datasets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    duplicates: list[str] = []
    for dataset in datasets:
        key = (str(dataset.get("backend")), str(dataset.get("shape")))
        if key in by_key:
            duplicates.append(f"DUPLICATE_MATRIX:{key[0]}:{key[1]}")
        else:
            by_key[key] = dataset
    ratios: list[dict[str, Any]] = []
    missing: list[str] = []
    shapes = sorted({shape for backend, shape in by_key if backend in {"CPU", "GPU"}})
    for shape in shapes:
        cpu = by_key.get(("CPU", shape))
        gpu = by_key.get(("GPU", shape))
        if cpu is None or gpu is None:
            missing.append(f"CPU_GPU_PAIR_MISSING:{shape}")
            continue
        row: dict[str, Any] = {"shape": shape, "backend_pair": ["CPU", "GPU"], "metrics": {}}
        cpu_off = cpu.get("off_statistics", {}).get("metrics", {})
        gpu_off = gpu.get("off_statistics", {}).get("metrics", {})
        for metric in METRICS:
            c = cpu_off.get(metric, {}).get("median_s")
            g = gpu_off.get(metric, {}).get("median_s")
            if c is None or g is None:
                missing.append(f"CPU_GPU_RATIO_MISSING:{shape}:{metric}")
            else:
                ratio = c / g
                if not math.isfinite(ratio) or ratio <= 0:
                    missing.append(f"CPU_GPU_RATIO_NONFINITE:{shape}:{metric}")
                else:
                    row["metrics"][metric] = {"cpu_off_median_s": c, "gpu_off_median_s": g, "cpu_over_gpu": ratio}
        ratios.append(row)
    return {"status": "REPLAYED" if ratios and not missing and not duplicates else ("INCOMPLETE" if missing or duplicates else "PENDING"), "ratios": ratios, "missing": sorted(set(missing + duplicates))}


def _executed_source_replay(root: Path) -> dict[str, Any]:
    """Verify packaged original Git blobs, using the frozen PV2 LF function."""
    import sys
    from types import ModuleType
    source_root = root / "executed_source"
    manifest_path = root / "EXECUTION_SOURCE_MANIFEST.json"
    if not source_root.is_dir():
        return {"status": "NOT_PACKAGED", "issues": [], "records": []}
    issues: list[str] = []
    rows: list[dict[str, Any]] = []
    try:
        manifest = load_json_strict(manifest_path)
        records = manifest["records"]
        helper_relative = "Filament_python/tools/hpc_ops/provenance_v2.py"
        helper_record = next(row for row in records if row["path"] == helper_relative)
        def read_blob(row: Mapping[str, Any]) -> tuple[Path, bytes]:
            path = _resolve_qualified(source_root, _relative_parts(row["path"]))
            raw = path.read_bytes()
            header = b"blob " + str(len(raw)).encode("ascii") + b"\0"
            oid = str(row["git_blob_oid"])
            actual = (hashlib.sha1 if len(oid) == 40 else hashlib.sha256)(header + raw).hexdigest()
            if actual != oid:
                raise ReplayError("PACKAGED_GIT_BLOB_OID_MISMATCH:" + str(row["path"]))
            return path, raw
        helper, helper_bytes = read_blob(helper_record)
        module = ModuleType("_s46a_frozen_pv2_replay")
        module.__file__ = str(helper)
        sys.modules[module.__name__] = module
        exec(compile(helper_bytes, str(helper), "exec"), module.__dict__)
        for row in records:
            if row.get("classification") != "tracked_text" or row.get("hash_scope") != module.TRACKED_HASH_SCOPE:
                raise ReplayError("PACKAGED_TRACKED_HASH_SCOPE")
            _, raw = read_blob(row)
            canonical = module.canonical_lf_sha256(raw)
            if canonical != row["canonical_lf_sha256"]:
                raise ReplayError("PACKAGED_CANONICAL_LF_MISMATCH:" + row["path"])
            rows.append({"path": row["path"], "git_blob_oid": row["git_blob_oid"], "canonical_lf_sha256": canonical})
        return {"status": "SOURCE_IDENTITIES_REPLAYED", "head": manifest["head"], "records": rows, "issues": []}
    except (ReplayError, OSError, KeyError, StopIteration) as exc:
        issues.append(str(exc))
        return {"status": "SOURCE_IDENTITY_INVALID", "records": rows, "issues": issues}


def replay_report(root: os.PathLike[str] | str) -> dict[str, Any]:
    """Read and recompute the available report metadata, without writing."""

    base = _report_root(root)
    scope = _scope_paths(base)
    selected = scope["paths"]

    # All normal replay inputs come from the explicit whitelist.  In the
    # absence of one, only the current windows_micro_r2 metadata scope is
    # selected above; old versions and broad report-tree scans are excluded.
    matrix_paths = sorted(
        (path for path in selected["datasets"] if _name_matches(path, "matrix")),
        key=str,
    )
    timing_candidates = [
        path for path in selected["timings"]
        if "REFERENCE" not in {part.upper() for part in path.relative_to(base).parts}
    ]
    timing_paths = sorted(
        (path for path in timing_candidates if _name_matches(path, "timing")),
        key=str,
    )
    case_paths = sorted(
        (path for path in timing_candidates if path.name.upper() == "CASE_RECEIPT.JSON"),
        key=str,
    )
    ledger_paths = sorted(
        (
            path for path in selected["ledgers"]
            if path.suffix.lower() in {".json", ".csv", ".log", ".txt"}
            and "REFERENCE" not in {part.upper() for part in path.relative_to(base).parts}
        ),
        key=str,
    )
    manifest_paths = sorted(
        (path for path in selected["manifests"] if _name_matches(path, "manifest")),
        key=str,
    )
    datasets: list[dict[str, Any]] = []
    issues: list[str] = [f"SCOPE:{issue}" for issue in scope["issues"]]
    for path in matrix_paths:
        try:
            dataset = _replay_matrix_file(base, path)
            datasets.append(dataset)
            issues.extend(f"{dataset['path']}:{issue}" for issue in dataset["issues"])
        except ReplayError as exc:
            issues.append(f"{path.relative_to(base).as_posix()}:{exc}")
    gates = [{"backend": d["backend"], "shape": d["shape"], **d["timing_gate"]} for d in datasets]
    case_metadata = _metadata_case_replay(base, case_paths, [])
    clocks = _clock_replay(base, timing_paths, [])
    resources = _resource_replay(base, ledger_paths, [])
    manifests = _manifest_replay(base, manifest_paths, [])
    historical = _historical_replay(base)
    executed_source = _executed_source_replay(base)
    ratios = _cpu_gpu_ratios(datasets)
    missing: list[str] = []
    unverified: list[str] = []
    issues.extend(f"CASE:{issue}" for issue in case_metadata["issues"])
    issues.extend(f"CLOCK:{issue}" for issue in clocks["issues"])
    issues.extend(f"RESOURCE:{issue}" for issue in resources["issues"])
    issues.extend(f"MANIFEST:{issue}" for issue in manifests["issues"])
    issues.extend(f"HISTORICAL:{issue}" for issue in historical.get("issues", []))
    issues.extend(f"SOURCE:{issue}" for issue in executed_source.get("issues", []))
    issues.extend(f"RATIO:{issue}" for issue in ratios["missing"])
    if not datasets:
        missing.append("MATRIX_RESULT")
    if not gates or any(gate.get("status") != "PASS_TIMING_GATE" for gate in gates):
        missing.append("TIMING_GATE")
    if ratios["status"] != "REPLAYED":
        missing.append("CPU_GPU_RATIO")
    if clocks["status"] != "REPLAYED":
        missing.append("RAW_TIMING_CLOCKS")
    if case_metadata["status"] != "REPLAYED":
        missing.append("CPU_CASE_METADATA_REPLAY")
    if resources["status"] != "REPLAYED":
        missing.append("RESOURCE_LEDGER")
    if manifests["status"] != "MANIFEST_REPLAYED":
        missing.append("SOURCE_INPUT_MANIFEST")
    if manifests.get("arrays_not_rechecked"):
        unverified.append("SCIENTIFIC_ARRAY_BYTES_OUT_OF_REPLAY_SCOPE")
        missing.append("SOURCE_INPUT_ARRAY_BYTES_NOT_RECHECKED")
    if historical.get("status") != "REPLAYED":
        missing.append("HISTORICAL_RUNTIME_REPLAY")
    if not timing_paths and not case_metadata["runs"]:
        unverified.append("CURRENT_RAW_TIMINGS_NOT_PACKAGED")
    if not datasets:
        unverified.append("MATRIX_RESULTS_NOT_PACKAGED")
    evidence_present = bool(datasets or case_metadata["runs"] or timing_paths or manifest_paths or ledger_paths)
    status = "REPLAY_COMPLETE_REVIEW_PENDING" if not missing and not issues else ("INCOMPLETE" if evidence_present else "PENDING")
    scope_public = {key: value for key, value in scope.items() if key != "paths"}
    return {
        "schema": SCHEMA,
        "status": status,
        "root": str(base),
        "scope": scope_public,
        "matrix": datasets,
        "timing_gates": gates,
        "cpu_gpu_ratios": ratios,
        "case_metadata": case_metadata,
        "clocks": clocks,
        "resources": resources,
        "manifests": manifests,
        "historical": historical,
        "executed_source": executed_source,
        "missing": sorted(set(missing)),
        "issues": sorted(set(issues)),
        "unverified": sorted(set(unverified)),
        "DEFAULT": "I0",
        "PRODUCTION_QUALIFIED": False,
        "MANUAL_REVIEW_PENDING": True,
        "review_gate": "MANUAL_REVIEW_PENDING",
    }


def self_test() -> dict[str, Any]:
    """Run a tiny in-memory/temp replay; the user report is never modified."""

    with tempfile.TemporaryDirectory(prefix="s46a-replay-") as directory:
        root = Path(directory)
        (root / "raw" / "CPU_FULL").mkdir(parents=True)
        (root / "raw" / "GPU").mkdir(parents=True)
        rows = []
        for case, value in (
            ("QUAL_OFF", 1.0), ("QUAL_LIGHT", 1.0),
            ("P1_OFF", 10.0), ("P1_LIGHT", 10.0),
            ("P2_LIGHT", 20.0), ("P2_OFF", 20.0),
            ("P3_OFF", 30.0), ("P3_LIGHT", 30.0),
        ):
            rows.append({"case": case, "application_service_s": value, "window_completed_s": value, "later_body_s": value})
        matrix = {"backend": "CPU", "shape": [1, 1, 1], "cases": rows}
        (root / "raw" / "CPU_FULL" / "MATRIX_RESULT_CPU_FULL.json").write_text(json.dumps(matrix), encoding="utf-8")
        matrix["backend"] = "GPU"
        matrix["cases"] = [{**row, "application_service_s": row["application_service_s"] / 2, "window_completed_s": row["window_completed_s"] / 2, "later_body_s": row["later_body_s"] / 2} for row in rows]
        (root / "raw" / "GPU" / "MATRIX_RESULT_GPU.json").write_text(json.dumps(matrix), encoding="utf-8")
        (root / "REPLAY_SCOPE.json").write_text(
            json.dumps({"datasets": ["raw"], "timings": [], "ledgers": [], "manifests": []}),
            encoding="utf-8",
        )
        replay = replay_report(root)
        checks = {
            "matrix_count": len(replay["matrix"]) == 2,
            "gate_count": all(item["status"] == "PASS_TIMING_GATE" for item in replay["timing_gates"]),
            "off_median": replay["matrix"][0]["off_statistics"]["metrics"]["application"]["median_s"] == 20.0,
            "cpu_gpu_ratio": replay["cpu_gpu_ratios"]["ratios"][0]["metrics"]["application"]["cpu_over_gpu"] == 2.0,
            "lifecycle": replay["DEFAULT"] == "I0" and replay["PRODUCTION_QUALIFIED"] is False and replay["MANUAL_REVIEW_PENDING"] is True,
        }
        if not all(checks.values()):
            raise ReplayError(f"SELF_TEST_FAILED:{checks}")
    return {
        "schema": SCHEMA,
        "status": "SELF_TEST_PASS",
        "checks": checks,
        "DEFAULT": "I0",
        "PRODUCTION_QUALIFIED": False,
        "MANUAL_REVIEW_PENDING": True,
        "review_gate": "MANUAL_REVIEW_PENDING",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only S46A metadata replay")
    parser.add_argument("--root", required=True, help="report directory")
    parser.add_argument("--self-test", action="store_true", help="use a temporary self-test fixture")
    args = parser.parse_args(argv)
    try:
        result = self_test() if args.self_test else replay_report(args.root)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:  # CLI must remain machine-readable on malformed evidence.
        print(json.dumps({"schema": SCHEMA, "status": "REPLAY_ERROR", "error": str(exc), "DEFAULT": "I0", "PRODUCTION_QUALIFIED": False, "MANUAL_REVIEW_PENDING": True}, ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
