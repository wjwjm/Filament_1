"""Small, read-only S46A evidence replay.

This module deliberately stays independent of the production package.  It
replays only the fixed S46A metadata schema (matrix result JSON, raw timing
JSON, manifests, and resource ledgers); it never opens scientific arrays and
never writes to the report directory during normal operation.
"""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from decimal import Decimal, InvalidOperation, localcontext
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
DEFAULT_CURRENT_SCOPE = ("raw",)
FALLBACK_CURRENT_SCOPE = ("terminal_evidence/raw/CPU_R2_FAILED",)
# These are the only continuation evidence roots that may be discovered when
# no REPLAY_SCOPE.json has been supplied.  They are deliberately named rather
# than inferred from every directory in the report: old versions, failure
# bundles, and historical tables must not silently become current inputs.
DEFAULT_FORMAL_METADATA_ROOTS = (
    "raw",
    "evidence_CPU_SMALL",
    "evidence_CPU_FULL",
    "evidence_GPU_SMALL",
    "evidence_GPU_FULL",
    "evidence_GPU",
)
LEDGER_DEFAULT_FILES = ("RESOURCE_LEDGER.csv", "RESOURCE_AND_STOP_STATE.json")
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


def load_json_strict(
    path: Path,
    *,
    max_bytes: int = MAX_METADATA_BYTES,
    max_nodes: int = 200_000,
) -> Any:
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
    _finite_tree(value, max_nodes=max_nodes)
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
        if not configured and kind in {"timings", "manifests", "ledgers"}:
            if kind == "ledgers":
                # The continuation will carry its own raw ledger when it is
                # synchronized.  Until then, retain the report-level ledger
                # so historical CPU allocations remain charged exactly once.
                report_ledger = base / LEDGER_DEFAULT_FILES[0]
                if report_ledger.is_file():
                    # SCHEDULER_TERMINAL.json in the old evidence tree is a
                    # second view of these same rows; the resource CSV is
                    # the authoritative compact ledger for the default path.
                    values = [LEDGER_DEFAULT_FILES[0]]
                else:
                    values = [
                        candidate
                        for candidate in DEFAULT_FORMAL_METADATA_ROOTS
                        if (base / candidate).is_dir() or (base / candidate).is_file()
                    ]
                    if not values:
                        for candidate in (*DEFAULT_CURRENT_SCOPE, *FALLBACK_CURRENT_SCOPE, *LEDGER_DEFAULT_FILES[1:]):
                            candidate_path = base / candidate
                            if candidate_path.is_dir() or candidate_path.is_file():
                                values.append(candidate)
            else:
                values = [
                    candidate
                    for candidate in DEFAULT_FORMAL_METADATA_ROOTS
                    if (base / candidate).is_dir() or (base / candidate).is_file()
                ]
                if not values:
                    values = list(DEFAULT_CURRENT_SCOPE if (base / DEFAULT_CURRENT_SCOPE[0]).is_dir() else FALLBACK_CURRENT_SCOPE)
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
    # A default scope selects only current formal metadata.  Matrix files are
    # selected only through an explicit dataset whitelist or CASE receipts;
    # old report versions are never scanned implicitly.
    if not configured:
        selected["timings"] = [
            path for path in selected["timings"]
            if path.name.upper() in {"RAW_CPU_TIMING.JSON", "CASE_RECEIPT.JSON"}
        ]
        selected["manifests"] = [
            path for path in selected["manifests"]
            if path.name.upper() in {
                "INPUT_MANIFEST.JSON",
                "PORTABLE_METADATA_MANIFEST.JSON",
                "REPLAY_SCOPE.JSON",
            }
        ]
        selected["ledgers"] = [
            path for path in selected["ledgers"]
            if _name_matches(path, "ledger")
        ]
    else:
        # Explicit directories still obey the fixed metadata vocabulary; a
        # directory entry is not permission to ingest logs, output trees, or
        # terminal JSONL files.
        selected["datasets"] = [
            path for path in selected["datasets"] if _name_matches(path, "matrix")
        ]
        selected["timings"] = [
            path for path in selected["timings"]
            if path.name.upper() in {"RAW_CPU_TIMING.JSON", "RAW_GPU_TIMING.JSON", "CASE_RECEIPT.JSON"}
        ]
        selected["ledgers"] = [
            path for path in selected["ledgers"] if _name_matches(path, "ledger")
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
            or upper.startswith(("SLURM_", "SLURM-"))
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


def _replay_matrix_file(
    root: Path,
    path: Path,
    *,
    max_bytes: int = MAX_METADATA_BYTES,
    max_nodes: int = 200_000,
) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    data = load_json_strict(path, max_bytes=max_bytes, max_nodes=max_nodes)
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
    if gate["status"] != "PASS_TIMING_GATE":
        issues.append(f"TIMING_GATE:{gate['status']}")
    return {
        "path": relative,
        "backend": backend,
        "shape": shape,
        "case_order": order,
        "case_order_ok": tuple(order) == CASE_ORDER,
        "cases": rows,
        "timing_gate": gate,
        "off_statistics": off,
        "matrix_status": data.get("status"),
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


def _completed_window_pair(clocks: Mapping[str, Any]) -> tuple[int | None, int | None, str | None]:
    """Read an explicitly recorded completed-window endpoint pair only."""

    def pick(names: Sequence[str], field: str) -> int:
        for name in names:
            if name not in clocks:
                continue
            value = clocks[name]
            if isinstance(value, Mapping):
                value = value.get("ns", value.get("value"))
            return _integer_ns(value, field)
        raise ReplayError(f"INTEGER_NS_REQUIRED:completed_window_clocks.{field}")

    try:
        first = pick(("start_ns", "start_wall_ns", "window_start_ns", "completed_window_start_ns"), "start_ns")
        last = pick(("end_ns", "end_wall_ns", "window_end_ns", "completed_window_end_ns"), "end_ns")
    except ReplayError as exc:
        return None, None, str(exc)
    if last < first:
        return first, last, "COMPLETED_WINDOW_CLOCK_RANGE_INVALID"
    return first, last, None


def _optional_clock_duration(
    clocks: Mapping[str, Any],
    starts: Sequence[str],
    ends: Sequence[str],
    field: str,
) -> tuple[float | None, str | None]:
    """Return a duration only when both integer-ns endpoints are recorded.

    A missing pair is a metadata omission (not a fabricated zero).  A partial
    pair or an invalid/range-reversed pair is an evidence error.
    """

    def pick(names: Sequence[str]) -> tuple[Any, bool]:
        for name in names:
            if name in clocks:
                value = clocks[name]
                if isinstance(value, Mapping):
                    value = value.get("ns", value.get("value"))
                return value, True
        return None, False

    start_value, has_start = pick(starts)
    end_value, has_end = pick(ends)
    if not has_start and not has_end:
        return None, None
    if not has_start or not has_end:
        return None, f"PROCESS_CPU_CLOCK_INCOMPLETE:{field}"
    try:
        start = _integer_ns(start_value, f"{field}.start_ns")
        end = _integer_ns(end_value, f"{field}.end_ns")
    except ReplayError as exc:
        return None, str(exc)
    if end < start:
        return None, f"PROCESS_CPU_CLOCK_RANGE_INVALID:{field}"
    return (end - start) / 1e9, None


def _post_metadata(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Read explicitly recorded POST timing metadata without clock mixing."""

    timeline = receipt.get("POST_timeline")
    if not isinstance(timeline, Mapping):
        return {
            "status": "PENDING",
            "source": None,
            "first_POST_from_application_s": None,
            "adjacent_POST_gaps_s": [],
            "adjacent_POST_gap_count": 0,
            "adjacent_POST_gap_median_s": None,
            "adjacent_POST_gap_min_s": None,
            "adjacent_POST_gap_max_s": None,
        }
    try:
        first_raw = timeline.get("first_POST_from_application_s")
        first = None if first_raw is None else _number(first_raw, "POST_timeline.first_POST_from_application_s", positive=True)
        gaps_raw = timeline.get("adjacent_POST_gaps_s", [])
        if not isinstance(gaps_raw, list):
            raise ReplayError("POST_timeline.adjacent_POST_gaps_s_LIST_REQUIRED")
        gaps = [
            _number(value, f"POST_timeline.adjacent_POST_gaps_s[{index}]", positive=True)
            for index, value in enumerate(gaps_raw)
        ]
    except ReplayError as exc:
        return {
            "status": "INCOMPLETE",
            "source": "CASE.POST_timeline",
            "first_POST_from_application_s": None,
            "adjacent_POST_gaps_s": [],
            "adjacent_POST_gap_count": 0,
            "adjacent_POST_gap_median_s": None,
            "adjacent_POST_gap_min_s": None,
            "adjacent_POST_gap_max_s": None,
            "issues": [str(exc)],
        }
    return {
        "status": "REPLAYED" if first is not None and gaps else "PENDING",
        "source": "CASE.POST_timeline",
        "first_POST_from_application_s": first,
        "adjacent_POST_gaps_s": gaps,
        "adjacent_POST_gap_count": len(gaps),
        "adjacent_POST_gap_median_s": statistics.median(gaps) if gaps else None,
        "adjacent_POST_gap_min_s": min(gaps) if gaps else None,
        "adjacent_POST_gap_max_s": max(gaps) if gaps else None,
    }


def _replay_case_metadata(root: Path, case_path: Path) -> dict[str, Any]:
    relative = case_path.relative_to(root).as_posix()
    issues: list[str] = []
    receipt = load_json_strict(case_path)
    if not isinstance(receipt, Mapping):
        raise ReplayError("CASE_RECEIPT_OBJECT_REQUIRED")
    backend_hint = str(receipt.get("backend", "")).upper()
    if backend_hint == "CPU":
        raw_candidates = [case_path.with_name("RAW_CPU_TIMING.json")]
    elif backend_hint == "GPU":
        raw_candidates = [case_path.with_name("RAW_GPU_TIMING.json")]
    else:
        raw_candidates = [
            case_path.with_name("RAW_GPU_TIMING.json"),
            case_path.with_name("RAW_CPU_TIMING.json"),
        ]
    raw_path = next((path for path in raw_candidates if path.is_file() and not path.is_symlink()), raw_candidates[0])
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
    application_cpu_s: float | None = None
    application_cpu_error: str | None = None
    case_completed_window_pair: tuple[int, int] | None = None
    completed_window_value = receipt.get("completed_window_clocks")
    if isinstance(completed_window_value, Mapping):
        case_start, case_end, case_window_error = _completed_window_pair(completed_window_value)
        if case_window_error:
            issues.append(case_window_error)
        elif case_start is not None and case_end is not None:
            case_completed_window_pair = (case_start, case_end)
    if not isinstance(clocks, Mapping):
        issues.append("APPLICATION_CLOCKS_MISSING")
        app_s = None
    else:
        app_start, app_end, error = _case_ns_pair(clocks, start="start_ns", end="application_end_ns")
        if error:
            issues.append(error)
        app_s = None if app_start is None or app_end is None else (app_end - app_start) / 1e9
        if app_s is not None and app_s <= 0:
            issues.append("APPLICATION_DURATION_NOT_POSITIVE")
        application_cpu_s, application_cpu_error = _optional_clock_duration(
            clocks,
            ("start_cpu_ns", "application_start_cpu_ns"),
            ("application_end_cpu_ns", "end_cpu_ns"),
            "application",
        )
        if application_cpu_error:
            issues.append(application_cpu_error)
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
        closure_rows = list(raw_rows)
        completed_window = receipt.get("completed_window_clocks")
        if isinstance(completed_window, Mapping) and not any(
            str(row.get("label", "")).upper() == "PROPAGATION" for row in raw_rows
        ):
            start, end, window_error = _completed_window_pair(completed_window)
            if window_error is None and start is not None and end is not None:
                closure_rows.append({
                    "id": "CASE.completed_window_clocks",
                    "label": "PROPAGATION",
                    "start_ns": start,
                    "end_ns": end,
                    "completed": True,
                })
            else:
                issues.append(window_error or "COMPLETED_WINDOW_CLOCKS_INVALID")
        closure = validate_clock_closure(closure_rows, expected_intervals=expected_intervals) if closure_rows else _raw_timing_replay(raw_data)
        if closure["status"] != "PASS_CLOCK_CLOSURE":
            issues.extend(f"CLOCK:{issue}" for issue in closure.get("issues", []))
    host_clock_exclusive = host_clock_exclusive_ledger(
        raw_rows,
        backend=str(receipt.get("backend", backend_hint or "UNKNOWN")),
        mode=str(receipt.get("mode", "UNKNOWN")),
        clock_source=("merged.records" if raw_kind == "merged" else "records.host_wall_ns" if raw_kind == "gpu" else "case.raw"),
    )
    if host_clock_exclusive["status"] != "PASS_HOST_CLOCK_EXCLUSIVE":
        issues.extend(f"HOST_CLOCK:{issue}" for issue in host_clock_exclusive.get("issues", []))
    interval_rows = [
        row for row in raw_rows
        if str(row.get("label", "")).upper() == "INTERVAL_BODY"
        and isinstance(row.get("interval_index"), int)
        and not isinstance(row.get("interval_index"), bool)
    ]
    interval_rows.sort(key=lambda row: int(row["interval_index"]))
    metrics: dict[str, float | None] = {"application": app_s, "window": None, "later": None}
    process_metrics: dict[str, float | None] = {
        "application": application_cpu_s,
        "window": None,
        "later": None,
    }
    first_interval_s: float | None = None
    first_interval_process_cpu_s: float | None = None
    process_metric_errors: list[str] = []
    metric_sources: dict[str, str] = {"application": "CASE.application_clocks"}
    propagation = [row for row in raw_rows if str(row.get("label", "")).upper() == "PROPAGATION"]
    completed_window_pair: tuple[int | None, int | None] | None = None
    if len(propagation) == 1:
        row = propagation[0]
        try:
            start = _integer_ns(row.get("start_ns"), "PROPAGATION.start_ns")
            end = _integer_ns(row.get("end_ns"), "PROPAGATION.end_ns")
            if end < start:
                raise ReplayError("PROPAGATION_CLOCK_RANGE_INVALID")
            metrics["window"] = (end - start) / 1e9
            metric_sources["window"] = (
                "RAW.merged.PROPAGATION.host_wall_ns"
                if raw_kind == "merged" else "RAW.records.PROPAGATION.host_wall_ns"
            )
            completed_window_pair = (start, end)
            if case_completed_window_pair is not None and case_completed_window_pair != (start, end):
                issues.append(
                    "RAW_CASE_COMPLETED_WINDOW_MISMATCH:"
                    f"{case_completed_window_pair!r}:{(start, end)!r}"
                )
            window_cpu, window_cpu_error = _optional_clock_duration(
                row,
                ("start_cpu_ns",),
                ("end_cpu_ns",),
                "window",
            )
            process_metrics["window"] = window_cpu
            if window_cpu_error:
                process_metric_errors.append(window_cpu_error)
        except ReplayError as exc:
            issues.append(str(exc))
    elif len(propagation) > 1:
        issues.append("PROPAGATION_SCOPE_AMBIGUOUS")
    else:
        completed_window = receipt.get("completed_window_clocks")
        if isinstance(completed_window, Mapping):
            start, end, error = _completed_window_pair(completed_window)
            if error:
                issues.append(error)
            elif start is not None and end is not None:
                metrics["window"] = (end - start) / 1e9
                metric_sources["window"] = "CASE.completed_window_clocks"
                completed_window_pair = (start, end)
                window_cpu, window_cpu_error = _optional_clock_duration(
                    completed_window,
                    ("start_cpu_ns",),
                    ("end_cpu_ns",),
                    "window",
                )
                process_metrics["window"] = window_cpu
                if window_cpu_error:
                    process_metric_errors.append(window_cpu_error)
        if completed_window_pair is None:
            issues.append("PROPAGATION_SCOPE_MISSING_NOT_APPLICATION_SUBSTITUTE")
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
                later_cpu, later_cpu_error = _optional_clock_duration(
                    {
                        "start_cpu_ns": later[0].get("start_cpu_ns"),
                        "end_cpu_ns": later[-1].get("end_cpu_ns"),
                    },
                    ("start_cpu_ns",),
                    ("end_cpu_ns",),
                    "later",
                )
                process_metrics["later"] = later_cpu
                if later_cpu_error:
                    process_metric_errors.append(later_cpu_error)
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
    if interval_rows:
        first_row = next((row for row in interval_rows if int(row["interval_index"]) == 0), None)
        if first_row is not None:
            try:
                first_start = _integer_ns(first_row.get("start_ns"), "interval1.start_ns")
                first_end = _integer_ns(first_row.get("end_ns"), "interval1.end_ns")
                if first_end < first_start:
                    raise ReplayError("INTERVAL1_CLOCK_RANGE_INVALID")
                first_interval_s = (first_end - first_start) / 1e9
                first_interval_process_cpu_s, first_cpu_error = _optional_clock_duration(
                    first_row,
                    ("start_cpu_ns",),
                    ("end_cpu_ns",),
                    "interval1",
                )
                if first_cpu_error:
                    process_metric_errors.append(first_cpu_error)
            except ReplayError as exc:
                process_metric_errors.append(str(exc))
        else:
            process_metric_errors.append("INTERVAL1_MISSING")
    post = _post_metadata(receipt)
    post_issues = post.get("issues", [])
    if post_issues:
        issues.extend(f"POST:{issue}" for issue in post_issues)
    for error in process_metric_errors:
        issues.append(f"PROCESS_METRIC:{error}")
    backend = str(receipt.get("backend", "UNKNOWN")).upper()
    case_name = str(receipt.get("case", receipt.get("mode", "UNKNOWN"))).upper()
    mode = str(receipt.get("mode", "UNKNOWN")).upper()
    status_text = str(receipt.get("status", receipt.get("state", ""))).upper()
    failed_marker = (
        receipt.get("failed") is True
        or receipt.get("error") is not None
        or any(token in status_text for token in ("FAIL", "CANCEL", "INCOMPLETE", "PENDING", "RUNNING"))
    )
    matrix_eligible = True
    exclusion_reason: str | None = None
    if mode in {"AUX_SERIALIZED", "SERIALIZED"} or case_name == "AUX_SERIALIZED":
        matrix_eligible = False
        exclusion_reason = "AUX_SERIALIZED_DIAGNOSTIC_ONLY"
    elif case_name not in CASE_ORDER:
        matrix_eligible = False
        exclusion_reason = "CASE_NOT_IN_FORMAL_CASE_ORDER"
    elif failed_marker:
        matrix_eligible = False
        exclusion_reason = "FAILED_OR_INCOMPLETE_RECEIPT"
    elif issues:
        matrix_eligible = False
        exclusion_reason = "CLOCK_OR_METADATA_REPLAY_INCOMPLETE"
    device_spans = [row.get("device_event_span_ms") for row in raw_rows if row.get("device_event_span_ms") is not None]
    return {
        "path": relative,
        "raw_timing_path": raw_path.relative_to(root).as_posix() if raw_path.is_file() and not raw_path.is_symlink() else None,
        "sample_id": (
            _sample_id(case_path)
            if _sample_id(case_path) not in {"WORK", "OFF", "LIGHT"}
            else str(receipt.get("case", receipt.get("mode", "UNKNOWN"))).upper()
        ),
        "backend": backend,
        "shape": _shape_id(receipt, case_path),
        "case": case_name,
        "mode": mode,
        "intervals": expected,
        "metrics_s": metrics,
        "metric_sources": metric_sources,
        "recorded_application_service_s": receipt.get("application_service_s"),
        "recorded_window_completed_s": receipt.get("window_completed_s"),
        "recorded_later_body_s": receipt.get("later_body_s"),
        "process_metrics_s": process_metrics,
        "first_interval_s": first_interval_s,
        "first_interval_process_cpu_s": first_interval_process_cpu_s,
        "post_timeline": post,
        "legacy_application_service_s": receipt.get("legacy_application_service_s"),
        "case_optical_compute_window_s": receipt.get("optical_compute_window_s"),
        "application_process_cpu_s": receipt.get("application_process_cpu_s"),
        "replayed_application_process_cpu_s": application_cpu_s,
        "completed_window_clocks": receipt.get("completed_window_clocks"),
        "completed_window_consistency": {
            "status": (
                "PASS_COMPLETED_WINDOW_ENDPOINT_MATCH"
                if case_completed_window_pair is not None
                and completed_window_pair == case_completed_window_pair
                else "PENDING_CASE_OR_RAW_COMPLETED_WINDOW_ENDPOINT"
                if case_completed_window_pair is None or completed_window_pair is None
                else "INCOMPLETE_COMPLETED_WINDOW_ENDPOINT_MISMATCH"
            ),
            "case_pair_ns": list(case_completed_window_pair) if case_completed_window_pair is not None else None,
            "raw_pair_ns": list(completed_window_pair) if completed_window_pair is not None else None,
        },
        "clock_closure": closure,
        "host_clock_exclusive": host_clock_exclusive,
        "deferred_device_event_spans_ms": device_spans,
        "matrix_eligible": matrix_eligible,
        "matrix_exclusion_reason": exclusion_reason,
        "receipt_status": status_text or None,
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
        # Formal evidence is runs/.../{case}/work/CASE_RECEIPT.json.  Keep
        # compatibility with the earlier micro OFF/LIGHT layout, while never
        # promoting a REFERENCE receipt into the current matrix.
        if path.parent.name.upper() not in {"WORK", "OFF", "LIGHT"}:
            continue
        try:
            run = _replay_case_metadata(root, path)
            if run["case"] == "REFERENCE":
                continue
            runs.append(run)
            issues.extend(f"{run['path']}:{issue}" for issue in run["issues"])
        except (ReplayError, OSError) as exc:
            issues.append(f"{path.relative_to(root).as_posix()}:{exc}")
    return {
        "status": "REPLAYED" if runs and not issues else ("INCOMPLETE" if runs else "PENDING"),
        "runs": runs,
        "issues": sorted(set(issues)),
    }


def _datasets_from_case_runs(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build matrix-shaped records from formal CASE receipts only."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for run in runs:
        case = str(run.get("case", "")).upper()
        if case not in CASE_ORDER:
            continue
        # AUX_SERIALIZED, failed, and other diagnostic receipts remain in the
        # case metadata/ledger, but can never fill a formal matrix sample.
        if run.get("matrix_eligible", True) is not True:
            continue
        key = (str(run.get("backend", "UNKNOWN")).upper(), str(run.get("shape", "UNKNOWN")))
        grouped.setdefault(key, []).append(run)
    datasets: list[dict[str, Any]] = []
    order_index = {case: index for index, case in enumerate(CASE_ORDER)}
    for (backend, shape), entries in sorted(grouped.items()):
        issues: list[str] = []
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for run in sorted(entries, key=lambda item: (order_index.get(str(item.get("case", "")).upper(), len(CASE_ORDER)), str(item.get("path", "")))):
            case = str(run.get("case", "")).upper()
            if case in seen:
                issues.append(f"DUPLICATE_CASE:{case}")
            seen.add(case)
            metrics = run.get("metrics_s")
            metrics = metrics if isinstance(metrics, Mapping) else {}
            rows.append({
                "case": case,
                "application_service_s": metrics.get("application"),
                "window_completed_s": metrics.get("window"),
                "later_body_s": metrics.get("later"),
            })
        order = [str(row.get("case", "")).upper() for row in rows]
        if tuple(order) != CASE_ORDER:
            issues.append("CASE_ORDER_MISMATCH")
        gate = recompute_timing_gate(rows)
        off = off_statistics(rows)
        issues.extend(gate["missing"])
        issues.extend(off["missing"])
        if gate["status"] != "PASS_TIMING_GATE":
            issues.append(f"TIMING_GATE:{gate['status']}")
        datasets.append({
            "path": f"formal_case_receipts/{backend}/{shape}",
            "backend": backend,
            "shape": shape,
            "source": "CASE_RECEIPT.application_clocks+RAW.PROPAGATION_or_completed_window_clocks+RAW_INTERVAL_BODY",
            "case_order": order,
            "case_order_ok": tuple(order) == CASE_ORDER,
            "cases": rows,
            "timing_gate": gate,
            "off_statistics": off,
            "issues": sorted(set(issues)),
            "status": "REPLAYED" if not issues else "INCOMPLETE",
        })
    return datasets


def _compare_matrix_case_datasets(
    matrix_datasets: Sequence[Mapping[str, Any]],
    case_datasets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Cross-check metadata values without reopening any scientific arrays."""

    matrix_by_key = {
        (str(item.get("backend")), str(item.get("shape"))): item
        for item in matrix_datasets
    }
    case_by_key = {
        (str(item.get("backend")), str(item.get("shape"))): item
        for item in case_datasets
    }
    checks: list[dict[str, Any]] = []
    issues: list[str] = []
    for key in sorted(set(matrix_by_key) | set(case_by_key)):
        matrix = matrix_by_key.get(key)
        cases = case_by_key.get(key)
        if matrix is None or cases is None:
            checks.append({
                "backend": key[0],
                "shape": key[1],
                "status": "PENDING",
                "reason": "MATRIX_OR_CASE_DATASET_MISSING",
            })
            continue
        matrix_rows = {
            str(row.get("case", "")).upper(): row
            for row in matrix.get("cases", [])
            if isinstance(row, Mapping)
        }
        case_rows = {
            str(row.get("case", "")).upper(): row
            for row in cases.get("cases", [])
            if isinstance(row, Mapping)
        }
        local_issues: list[str] = []
        for case in CASE_ORDER:
            matrix_row = matrix_rows.get(case)
            case_row = case_rows.get(case)
            if matrix_row is None or case_row is None:
                local_issues.append(f"MISSING_CASE:{case}")
                continue
            for metric in METRICS:
                matrix_value = matrix_row.get(f"{metric}_s")
                case_key = {
                    "application": "application_service_s",
                    "window": "window_completed_s",
                    "later": "later_body_s",
                }[metric]
                case_value = case_row.get(case_key)
                if matrix_value != case_value:
                    local_issues.append(
                        f"VALUE_MISMATCH:{case}:{metric}:{matrix_value!r}:{case_value!r}"
                    )
        status = "PASS_MATRIX_CASE_METADATA_EXACT" if not local_issues else "INCOMPLETE"
        checks.append({
            "backend": key[0],
            "shape": key[1],
            "status": status,
            "matrix_path": matrix.get("path"),
            "matrix_status": matrix.get("matrix_status"),
            "matrix_timing_gate": matrix.get("timing_gate", {}).get("status"),
            "case_timing_gate": cases.get("timing_gate", {}).get("status"),
            "issues": sorted(set(local_issues)),
        })
        issues.extend(f"{key[0]}:{key[1]}:{item}" for item in local_issues)
    return {
        "status": "REPLAYED" if checks and not issues else ("INCOMPLETE" if checks else "PENDING"),
        "checks": checks,
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
            "start_cpu_ns": row.get("start_cpu_ns", normalized.get("start_cpu_ns")),
            "end_cpu_ns": row.get("end_cpu_ns", normalized.get("end_cpu_ns")),
            "wall_inclusive_ns": row.get("wall_inclusive_ns"),
            "wall_exclusive_ns": row.get("wall_exclusive_ns"),
            "cpu_inclusive_ns": row.get("cpu_inclusive_ns"),
            "cpu_exclusive_ns": row.get("cpu_exclusive_ns"),
            "direct_child_wall_inclusive_ns": row.get("direct_child_wall_inclusive_ns"),
            "direct_child_cpu_inclusive_ns": row.get("direct_child_cpu_inclusive_ns"),
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
        token = row.get("token")
        label = row.get("label", row.get("scope", row.get("name", "")))
        # GPU persistence names the completed application propagation window
        # by token; retain the explicit token while normalizing its scope for
        # the fixed closure/gate vocabulary.
        if not label and str(token) == "completed_propagation_window":
            label = "PROPAGATION"
        completed = row.get("completed") is True
        if "completed" not in row:
            completed = row.get("failed") is not True and "end_wall_ns" in row
        rows.append({
            "id": token,
            "parent": row.get("parent", row.get("parent_token")),
            "label": label,
            "interval_index": row.get("interval", row.get("interval_index", context.get("interval_index"))),
            "completed": completed,
            "start_ns": row.get("start_wall_ns"),
            "end_ns": row.get("end_wall_ns"),
            "start_cpu_ns": row.get("start_cpu_ns"),
            "end_cpu_ns": row.get("end_cpu_ns"),
            "wall_ns": row.get("wall_ns"),
            "device_event_span_ms": row.get("device_event_span_ms"),
        })
    return rows


_HOST_CLOCK_LABELS = {
    "linear": frozenset({"LINEAR"}),
    "Raman_Heun": frozenset({
        "RAMAN_STAGE", "RAMAN_HALF", "HEUN_ASSEMBLY", "RAMAN_INTENSITY",
        "RAMAN_COEFFICIENTS", "RAMAN_IIR", "RAMAN_ENERGY", "FFT",
        "MATH_KERNELS", "EQ10_SCALE_MAX", "EQ10_DERIVATIVE_EXPR",
        "EQ10_FFT", "EQ10_IFFT", "EQ10_NORMALIZE_SCALE",
        "EQ10_PRODUCT_REDUCE_SCALE", "EQ10_SUM", "EQ10_MATERIALIZE",
    }),
    "ion": frozenset({"ION_SETUP", "ION_RECURSION"}),
    "otherNL": frozenset({"NONLINEAR", "NONLINEAR_ARRAYS"}),
    "deposition_POST": frozenset({
        "INTERVAL_DEPOSITION_AND_QACC", "INTERVAL_ENERGY_BOOKKEEPING",
        "INTERVAL_POST_COMMIT",
    }),
    "diagnostics_output": frozenset({
        "DIAGNOSTICS", "OUTPUT", "INTERVAL_SAVED_DIAGNOSTICS",
    }),
}
_HOST_CLOCK_MANAGEMENT_LABELS = frozenset({
    "APPLICATION", "INPUT_PREPARATION", "PROP_ENTRY_INIT", "PROP_FINALIZE_EXIT",
    "INTERVAL_PREP_AND_OTHER_ARRAYS",
})
_HOST_CLOCK_REMAINDER_LABELS = frozenset({"PROPAGATION", "INTERVAL_BODY"})
_HOST_CLOCK_CATEGORY_ORDER = tuple((*_HOST_CLOCK_LABELS, "management_remainder_unknown"))


def recompute_host_clock_exclusive(
    rows: Sequence[Mapping[str, Any]],
    *,
    backend: str,
    mode: str | None = None,
    clock_source: str = "normalized_rows",
) -> dict[str, Any]:
    """Recompute a seven-class host-clock exclusive ledger from integer clocks.

    Every included row contributes once as ``duration - direct-child durations``;
    no stored exclusive field is trusted and negative results are reported, not
    clipped.  GPU host clocks remain an asynchronous submission/wait view and
    CUDA event spans remain a separate idle-inclusive diagnostic view.
    """

    backend_upper = str(backend).upper()
    mode_upper = None if mode is None else str(mode).upper()
    issues: list[str] = []
    normalized: list[dict[str, Any]] = []
    ids: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            issues.append(f"HOST_CLOCK_ROW_NOT_OBJECT:{index}")
            continue
        identifier = str(raw.get("id", raw.get("token", raw.get("merged_token", index))))
        if identifier in ids:
            issues.append(f"HOST_CLOCK_DUPLICATE_ID:{identifier}")
            continue
        parent_raw = raw.get("parent", raw.get("merged_parent_token", raw.get("parent_token")))
        parent = None if parent_raw is None else str(parent_raw)
        label = str(raw.get("label", raw.get("scope", raw.get("name", "")))).upper()
        try:
            start = _integer_ns(raw.get("start_ns"), f"host.{identifier}.start_ns")
            end = _integer_ns(raw.get("end_ns"), f"host.{identifier}.end_ns")
            if end < start:
                raise ReplayError(f"HOST_CLOCK_RANGE_INVALID:{identifier}")
        except ReplayError as exc:
            issues.append(str(exc))
            continue
        cpu_start = raw.get("start_cpu_ns")
        cpu_end = raw.get("end_cpu_ns")
        cpu_duration: int | None = None
        if cpu_start is None and cpu_end is None:
            if backend_upper in {"CPU", "GPU"}:
                issues.append(f"PROCESS_CPU_CLOCK_MISSING:{identifier}")
        elif cpu_start is None or cpu_end is None:
            issues.append(f"PROCESS_CPU_CLOCK_INCOMPLETE:{identifier}")
        else:
            try:
                cpu_start = _integer_ns(cpu_start, f"host.{identifier}.start_cpu_ns")
                cpu_end = _integer_ns(cpu_end, f"host.{identifier}.end_cpu_ns")
                if cpu_end < cpu_start:
                    raise ReplayError(f"PROCESS_CPU_CLOCK_RANGE_INVALID:{identifier}")
                cpu_duration = cpu_end - cpu_start
            except ReplayError as exc:
                issues.append(str(exc))
        row = {
            **dict(raw),
            "id": identifier,
            "parent": parent,
            "label": label,
            "start_ns": start,
            "end_ns": end,
            "wall_duration_ns": end - start,
            "process_cpu_duration_ns": cpu_duration,
        }
        if row.get("completed") is not True:
            issues.append(f"HOST_CLOCK_NOT_COMPLETED:{identifier}")
        normalized.append(row)
        ids[identifier] = row

    roots = [row for row in normalized if row["label"] == "PROPAGATION"]
    if len(roots) != 1:
        issues.append("PROPAGATION_ROOT_REQUIRED" if not roots else "PROPAGATION_ROOT_AMBIGUOUS")
        return {
            "status": "PENDING" if not roots else "INCOMPLETE",
            "backend": backend_upper,
            "mode": mode_upper,
            "clock_source": clock_source,
            "categories": {name: {"labels": [], "row_count": 0, "wall_exclusive_ns": 0, "process_cpu_exclusive_ns": 0} for name in _HOST_CLOCK_CATEGORY_ORDER},
            "issues": sorted(set(issues)),
            "label_policy": {
                "fixed": {name: sorted(labels) for name, labels in _HOST_CLOCK_LABELS.items()},
                "management_labels": sorted(_HOST_CLOCK_MANAGEMENT_LABELS),
                "remainder_labels": sorted(_HOST_CLOCK_REMAINDER_LABELS),
                "unknown_labels": [],
            },
        }

    root = roots[0]
    included: list[dict[str, Any]] = []
    for row in normalized:
        if row["id"] == root["id"]:
            included.append(row)
            continue
        current = row
        seen: set[str] = set()
        belongs = False
        while current.get("parent") is not None:
            parent = str(current["parent"])
            if parent in seen:
                issues.append(f"HOST_CLOCK_PARENT_CYCLE:{row['id']}")
                break
            seen.add(parent)
            if parent == root["id"]:
                belongs = True
                break
            current = ids.get(parent)
            if current is None:
                issues.append(f"HOST_CLOCK_PARENT_MISSING:{row['id']}:{parent}")
                break
        if belongs:
            included.append(row)

    included_ids = {row["id"] for row in included}
    children: dict[str, list[dict[str, Any]]] = {row["id"]: [] for row in included}
    for row in included:
        parent = row.get("parent")
        if parent is None or parent not in included_ids:
            continue
        parent_row = ids.get(str(parent))
        if parent_row is None:
            issues.append(f"HOST_CLOCK_PARENT_MISSING:{row['id']}:{parent}")
            continue
        if row["start_ns"] < parent_row["start_ns"] or row["end_ns"] > parent_row["end_ns"]:
            issues.append(f"HOST_CLOCK_CHILD_OUTSIDE_PARENT:{row['id']}")
        children[str(parent)].append(row)
    for parent, siblings in children.items():
        ordered = sorted(siblings, key=lambda item: (item["start_ns"], item["end_ns"], item["id"]))
        for left, right in zip(ordered, ordered[1:]):
            if left["end_ns"] > right["start_ns"]:
                issues.append(f"HOST_CLOCK_SIBLING_OVERLAP:{parent}")

    categories = {
        name: {"labels": set(), "row_count": 0, "wall_exclusive_ns": 0, "process_cpu_exclusive_ns": 0}
        for name in _HOST_CLOCK_CATEGORY_ORDER
    }
    unknown_labels: set[str] = set()
    event_spans: list[dict[str, Any]] = []
    process_cpu_complete = True
    for row in included:
        label = row["label"]
        category = next((name for name, labels in _HOST_CLOCK_LABELS.items() if label in labels), None)
        if category is None:
            category = "management_remainder_unknown"
            if label not in _HOST_CLOCK_MANAGEMENT_LABELS and label not in _HOST_CLOCK_REMAINDER_LABELS:
                unknown_labels.add(label)
                issues.append(f"UNKNOWN_HOST_CLOCK_LABEL:{label}")
        children_rows = children.get(row["id"], [])
        wall_child_ns = sum(child["wall_duration_ns"] for child in children_rows)
        wall_exclusive_ns = row["wall_duration_ns"] - wall_child_ns
        if wall_exclusive_ns < 0:
            issues.append(f"NEGATIVE_WALL_EXCLUSIVE:{row['id']}")
        cpu_children = [child["process_cpu_duration_ns"] for child in children_rows]
        if row["process_cpu_duration_ns"] is None or any(value is None for value in cpu_children):
            process_cpu_complete = False
            cpu_exclusive_ns = None
        else:
            cpu_exclusive_ns = row["process_cpu_duration_ns"] - sum(value for value in cpu_children if value is not None)
            if cpu_exclusive_ns < 0:
                issues.append(f"NEGATIVE_PROCESS_CPU_EXCLUSIVE:{row['id']}")
        item = categories[category]
        item["labels"].add(label)
        item["row_count"] += 1
        item["wall_exclusive_ns"] += wall_exclusive_ns
        if cpu_exclusive_ns is not None:
            item["process_cpu_exclusive_ns"] += cpu_exclusive_ns
        if row.get("device_event_span_ms") is not None:
            try:
                span = _number(row["device_event_span_ms"], f"device_event_span_ms.{row['id']}")
                event_spans.append({"id": row["id"], "label": label, "span_ms": span})
            except ReplayError as exc:
                issues.append(str(exc))

    wall_total = sum(item["wall_exclusive_ns"] for item in categories.values())
    cpu_total = sum(item["process_cpu_exclusive_ns"] for item in categories.values()) if process_cpu_complete else None
    if wall_total != root["wall_duration_ns"]:
        issues.append("HOST_WALL_EXCLUSIVE_SUM_MISMATCH")
    if process_cpu_complete and cpu_total != root["process_cpu_duration_ns"]:
        issues.append("PROCESS_CPU_EXCLUSIVE_SUM_MISMATCH")
    for item in categories.values():
        item["labels"] = sorted(item["labels"])
    if backend_upper == "GPU":
        host_semantics = "async submission/wait host wall clock; not an additive cross-backend operator cost"
    elif backend_upper == "CPU":
        host_semantics = "synchronized CPU host wall clock"
    else:
        host_semantics = "unknown host clock semantics"
    aux_serialized = backend_upper == "GPU" and mode_upper in {"AUX_SERIALIZED", "SERIALIZED"}
    return {
        "status": "PASS_HOST_CLOCK_EXCLUSIVE" if not issues else "INCOMPLETE",
        "backend": backend_upper,
        "mode": mode_upper,
        "clock_source": clock_source,
        "host_wall_semantics": host_semantics,
        "cross_backend_operator_difference_allowed": False,
        "diagnostic_only": aux_serialized,
        "can_fill_off_or_p_samples": not aux_serialized,
        "cuda_event_spans_ms": event_spans,
        "cuda_event_spans_are_idle_inclusive_not_busy": backend_upper == "GPU",
        "cuda_event_spans_not_subtracted_from_host": backend_upper == "GPU",
        "propagation": {
            "id": root["id"],
            "wall_inclusive_ns": root["wall_duration_ns"],
            "process_cpu_inclusive_ns": root["process_cpu_duration_ns"],
            "row_count": len(included),
        },
        "categories": categories,
        "exclusive_sum": {
            "wall_ns": wall_total,
            "process_cpu_ns": cpu_total,
            "wall_equals_propagation": wall_total == root["wall_duration_ns"],
            "process_cpu_equals_propagation": process_cpu_complete and cpu_total == root["process_cpu_duration_ns"],
        },
        "label_policy": {
            "fixed": {name: sorted(labels) for name, labels in _HOST_CLOCK_LABELS.items()},
            "management_labels": sorted(_HOST_CLOCK_MANAGEMENT_LABELS),
            "remainder_labels": sorted(_HOST_CLOCK_REMAINDER_LABELS),
            "unknown_labels": sorted(unknown_labels),
        },
        "issues": sorted(set(issues)),
    }


def host_clock_exclusive_ledger(
    data_or_rows: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    backend: str,
    mode: str | None = None,
    clock_source: str = "normalized_rows",
) -> dict[str, Any]:
    """Normalize a raw timing object and return its metadata-only ledger."""

    if isinstance(data_or_rows, Mapping):
        merged_rows = _merged_clock_rows(data_or_rows)
        if merged_rows:
            return recompute_host_clock_exclusive(
                merged_rows, backend=backend, mode=mode, clock_source="merged.records"
            )
        gpu_rows = _gpu_clock_rows(data_or_rows)
        if gpu_rows:
            return recompute_host_clock_exclusive(
                gpu_rows, backend=backend, mode=mode, clock_source="records.host_wall_ns"
            )
        return recompute_host_clock_exclusive([], backend=backend, mode=mode, clock_source=clock_source)
    return recompute_host_clock_exclusive(
        data_or_rows, backend=backend, mode=mode, clock_source=clock_source
    )


def _raw_timing_replay(
    data: Any,
    *,
    backend: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        return {"status": "INCOMPLETE", "row_count": 0, "scopes": {}, "issues": ["TIMING_OBJECT_REQUIRED"]}
    merged_rows = _merged_clock_rows(data)
    if merged_rows:
        summary = validate_clock_closure(merged_rows)
        summary["clock_source"] = "merged.records"
        summary["overlap_views_ignored"] = [name for name in ("b", "s45a") if isinstance(data.get(name), Mapping)]
        ledger = host_clock_exclusive_ledger(
            merged_rows,
            backend=backend or str(data.get("backend", "CPU")),
            mode=mode or data.get("mode"),
            clock_source="merged.records",
        )
        summary["host_clock_exclusive"] = ledger
        summary["issues"] = sorted(set(summary.get("issues", []) + ledger.get("issues", [])))
        return summary
    gpu_rows = _gpu_clock_rows(data)
    if gpu_rows:
        summary = validate_clock_closure(gpu_rows)
        summary["clock_source"] = "records.host_wall_ns"
        spans = [row.get("device_event_span_ms") for row in gpu_rows if row.get("device_event_span_ms") is not None]
        summary["deferred_device_event_spans_ms"] = spans
        summary["device_spans_not_subtracted_from_host"] = True
        ledger = host_clock_exclusive_ledger(
            gpu_rows,
            backend=backend or str(data.get("backend", "GPU")),
            mode=mode or data.get("mode"),
            clock_source="records.host_wall_ns",
        )
        summary["host_clock_exclusive"] = ledger
        summary["issues"] = sorted(set(summary.get("issues", []) + ledger.get("issues", [])))
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
    "CPU": {"ids": 4, "core_hours": 240.0, "cpus": 96, "gpus": 0, "categories": {"small": 1200.0, "large": 5400.0, "repair": 5400.0, "main": 5400.0}},
    "GPU": {"ids": 4, "gpu_hours": 2.0, "core_hours": 16.0, "cpus": 8, "gpus": 1, "categories": {"main": 5400.0, "repair": 1800.0}},
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


def _sacct_rows_from_text(
    text: str,
    *,
    legacy_state_format: bool = False,
    backend_hint: str | None = None,
) -> list[Mapping[str, Any]]:
    """Decode the fixed raw sacct fields used by the S46A scheduler receipt.

    Current receipts include ``JobName``.  The earlier compact receipt used
    ``JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,TimelimitRaw`` and therefore
    has no job-name column; keep that format explicit rather than shifting its
    state into the wrong field.
    """

    rows: list[Mapping[str, Any]] = []
    for line in text.splitlines():
        fields = [part.strip() for part in line.split("|")]
        if len(fields) < 6 or not re.match(r"^\d+(?:\.\d+|\.(?:batch|extern))?$", fields[0]):
            continue
        if legacy_state_format:
            job_id, state, _exit_code, elapsed, alloc_cpus, timelimit = fields[:6]
            job_name = ""
            backend = str(backend_hint or "CPU").upper()
            category = "main"
        else:
            job_id, job_name, state, _exit_code, elapsed, alloc_cpus = fields[:6]
            timelimit = ""
            name = job_name.lower()
            backend = "GPU" if "gpu" in name else "CPU"
            category = "repair" if "repair" in name else "small" if "small" in name else "large" if "large" in name else "main"
        rows.append({
            "backend": backend,
            "job_id": job_id,
            "category": category,
            "state": state.split()[0].lower(),
            "actual_walltime_s": elapsed,
            "worst_walltime_s": timelimit or None,
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
                if isinstance(value, Mapping) and isinstance(value.get("stdout"), str):
                    argv = value.get("argv")
                    argv_text = " ".join(str(item) for item in argv) if isinstance(argv, list) else ""
                    legacy = "JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,TimelimitRaw" in argv_text
                    backend_hint = "GPU" if "GPU" in relative.upper() else "CPU"
                    rows.extend(
                        _sacct_rows_from_text(
                            value["stdout"],
                            legacy_state_format=legacy,
                            backend_hint=backend_hint,
                        )
                    )
                else:
                    parsed = _ledger_rows(value)
                    rows.extend(parsed if parsed else _sacct_rows_from_value(value))
            files.append(relative)
        except (OSError, ReplayError, csv.Error) as exc:
            issues.append(f"{relative}:{exc}")
    # Multiple scheduler views may carry the same main JobID.  Retain the
    # most specific row (small/large/repair over generic main) once; step rows
    # are still passed through and excluded by replay_resource_rows().
    selected_rows: list[Mapping[str, Any]] = []
    main_rows: dict[tuple[str, str], tuple[int, Mapping[str, Any]]] = {}
    for row in rows:
        job_id = _field(row, "job_id", "jobid", "job_id_raw", "id", "JobIDRaw")
        job_text = "" if job_id is None else str(job_id).strip()
        if job_text and "." not in job_text:
            backend = str(_field(row, "backend", "resource") or "").upper()
            category = str(_field(row, "category", "job_category") or "").lower()
            score = int(category in {"small", "large", "repair"})
            score += int(_field(row, "cpus", "cpu_count", "requested_cpus", "alloc_cpus") not in (None, ""))
            key = (backend, job_text)
            prior = main_rows.get(key)
            if prior is None or score > prior[0]:
                main_rows[key] = (score, row)
        else:
            selected_rows.append(row)
    selected_rows.extend(row for _score, row in main_rows.values())
    summary = replay_resource_rows(selected_rows)
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


def _ns_stat_summary(values: Sequence[Any]) -> dict[str, Any]:
    """Summarize already validated integer-ns values without changing units."""

    numbers = [value for value in values if isinstance(value, int) and not isinstance(value, bool)]
    if len(numbers) != len(values) or not numbers:
        return {
            "status": "INCOMPLETE",
            "count": len(numbers),
            "values_ns": numbers,
            "values_s": [value / 1e9 for value in numbers],
            "median_ns": None,
            "range_ns": None,
            "mad_ns": None,
            "issues": ["INTEGER_NS_VALUES_REQUIRED"],
        }
    median_ns = statistics.median(numbers)
    range_ns = max(numbers) - min(numbers)
    mad_ns = statistics.median([abs(value - median_ns) for value in numbers])
    return {
        "status": "REPLAYED",
        "count": len(numbers),
        "values_ns": numbers,
        "values_s": [value / 1e9 for value in numbers],
        "median_ns": median_ns,
        "median_s": median_ns / 1e9,
        "range_ns": range_ns,
        "range_s": range_ns / 1e9,
        "mad_ns": mad_ns,
        "mad_s": mad_ns / 1e9,
        "issues": [],
    }


def _float_stat_summary(values: Sequence[Any]) -> dict[str, Any]:
    numbers = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))]
    if len(numbers) != len(values) or not numbers:
        return {
            "status": "INCOMPLETE",
            "count": len(numbers),
            "values": numbers,
            "median": None,
            "range": None,
            "mad": None,
            "issues": ["FINITE_FLOAT_VALUES_REQUIRED"],
        }
    median_value = float(statistics.median(numbers))
    range_value = max(numbers) - min(numbers)
    mad_value = float(statistics.median([abs(item - median_value) for item in numbers]))
    return {
        "status": "REPLAYED",
        "count": len(numbers),
        "values": numbers,
        "median": median_value,
        "range": range_value,
        "mad": mad_value,
        "issues": [],
    }


def _dataset_by_key(datasets: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(item.get("backend", "")).upper(), str(item.get("shape", ""))): item
        for item in datasets
        if isinstance(item, Mapping)
    }


def light_exclusive_summaries(
    runs: Sequence[Mapping[str, Any]],
    datasets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize the fixed seven host-clock classes for formal LIGHT P cases.

    This is a metadata ledger only.  CPU ranks are process-CPU cost ranks;
    GPU asynchronous host clocks are retained separately and never converted
    into a cross-backend operator difference.
    """

    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for run in runs:
        if not isinstance(run, Mapping) or run.get("matrix_eligible") is not True:
            continue
        if str(run.get("mode", "")).upper() != "LIGHT":
            continue
        case = str(run.get("case", "")).upper()
        if case not in {f"{pair}_LIGHT" for pair in PAIRS}:
            continue
        key = (str(run.get("backend", "")).upper(), str(run.get("shape", "")))
        grouped.setdefault(key, {})[case] = run
    by_dataset = _dataset_by_key(datasets)
    result: list[dict[str, Any]] = []
    for (backend, shape), by_case in sorted(grouped.items()):
        dataset = by_dataset.get((backend, shape), {})
        gate = dataset.get("timing_gate") if isinstance(dataset, Mapping) else {}
        gate = gate if isinstance(gate, Mapping) else {}
        category_rows: dict[str, dict[str, Any]] = {}
        closure_rows: list[dict[str, Any]] = []
        issues: list[str] = []
        for case in ("P1_LIGHT", "P2_LIGHT", "P3_LIGHT"):
            run = by_case.get(case)
            ledger = run.get("host_clock_exclusive") if isinstance(run, Mapping) else None
            ledger = ledger if isinstance(ledger, Mapping) else {}
            categories = ledger.get("categories")
            categories = categories if isinstance(categories, Mapping) else {}
            exclusive = ledger.get("exclusive_sum")
            exclusive = exclusive if isinstance(exclusive, Mapping) else {}
            propagation = ledger.get("propagation")
            propagation = propagation if isinstance(propagation, Mapping) else {}
            wall_sum = exclusive.get("wall_ns")
            wall_prop = propagation.get("wall_inclusive_ns")
            cpu_sum = exclusive.get("process_cpu_ns")
            cpu_prop = propagation.get("process_cpu_inclusive_ns")
            wall_residual = wall_sum - wall_prop if isinstance(wall_sum, int) and isinstance(wall_prop, int) else None
            cpu_residual = cpu_sum - cpu_prop if isinstance(cpu_sum, int) and isinstance(cpu_prop, int) else None
            closure_ok = bool(
                run
                and ledger.get("status") == "PASS_HOST_CLOCK_EXCLUSIVE"
                and exclusive.get("wall_equals_propagation") is True
                and exclusive.get("process_cpu_equals_propagation") is True
                and wall_residual == 0
                and cpu_residual == 0
            )
            closure_rows.append({
                "case": case,
                "source_path": run.get("path") if isinstance(run, Mapping) else None,
                "ledger_status": ledger.get("status"),
                "propagation_wall_ns": wall_prop,
                "exclusive_wall_ns": wall_sum,
                "wall_residual_ns": wall_residual,
                "propagation_process_cpu_ns": cpu_prop,
                "exclusive_process_cpu_ns": cpu_sum,
                "process_cpu_residual_ns": cpu_residual,
                "closure_pass": closure_ok,
                "issues": ledger.get("issues", []) if isinstance(ledger.get("issues"), list) else [],
            })
            if not closure_ok:
                issues.append(f"CLOSURE:{case}")
            for category in _HOST_CLOCK_CATEGORY_ORDER:
                item = categories.get(category)
                item = item if isinstance(item, Mapping) else {}
                category_rows.setdefault(category, {"wall": [], "process_cpu": [], "occupancy": []})["wall"].append(item.get("wall_exclusive_ns"))
                category_rows.setdefault(category, {"wall": [], "process_cpu": [], "occupancy": []})["process_cpu"].append(item.get("process_cpu_exclusive_ns"))
                wall_value = item.get("wall_exclusive_ns")
                cpu_value = item.get("process_cpu_exclusive_ns")
                if isinstance(wall_value, int) and wall_value > 0 and isinstance(cpu_value, int) and cpu_value >= 0:
                    category_rows.setdefault(category, {"wall": [], "process_cpu": [], "occupancy": []})["occupancy"].append(cpu_value / wall_value)
        categories_out: list[dict[str, Any]] = []
        for category in _HOST_CLOCK_CATEGORY_ORDER:
            values = category_rows.get(category, {"wall": [], "process_cpu": []})
            wall_values = [value for value in values["wall"] if isinstance(value, int) and not isinstance(value, bool)]
            cpu_values = [value for value in values["process_cpu"] if isinstance(value, int) and not isinstance(value, bool)]
            occupancy_values = values.get("occupancy", [])
            if len(wall_values) != 3:
                issues.append(f"CATEGORY_WALL_COUNT:{category}:{len(wall_values)}")
            if len(cpu_values) != 3:
                issues.append(f"CATEGORY_PROCESS_CPU_COUNT:{category}:{len(cpu_values)}")
            category_output = {
                "category": category,
                "wall_exclusive": _ns_stat_summary(wall_values),
                "process_cpu_exclusive": _ns_stat_summary(cpu_values),
                "occupancy_process_cpu_over_wall": _float_stat_summary(occupancy_values),
                "labels": sorted({
                    label
                    for case in by_case.values()
                    for label in (
                        (case.get("host_clock_exclusive") or {}).get("categories", {}).get(category, {}).get("labels", [])
                        if isinstance(case.get("host_clock_exclusive"), Mapping)
                        and isinstance((case.get("host_clock_exclusive") or {}).get("categories"), Mapping)
                        and isinstance(((case.get("host_clock_exclusive") or {}).get("categories") or {}).get(category), Mapping)
                        else []
                    )
                }),
            }
            categories_out.append(category_output)
        ranked = sorted(
            categories_out,
            key=lambda item: (
                item["process_cpu_exclusive"].get("median_ns") is not None,
                item["process_cpu_exclusive"].get("median_ns") or -1,
            ),
            reverse=True,
        )
        for rank, item in enumerate(ranked, 1):
            item["cpu_cost_rank"] = rank if backend == "CPU" else None
        wall_ranked = sorted(
            categories_out,
            key=lambda item: (
                item["wall_exclusive"].get("median_ns") is not None,
                item["wall_exclusive"].get("median_ns") or -1,
            ),
            reverse=True,
        )
        for rank, item in enumerate(wall_ranked, 1):
            item["wall_cost_rank"] = rank if backend == "CPU" else None
        propagation_wall_values = [
            row.get("propagation_wall_ns")
            for row in closure_rows
            if isinstance(row.get("propagation_wall_ns"), int)
        ]
        propagation_wall_median = statistics.median(propagation_wall_values) if propagation_wall_values else None
        for item in categories_out:
            wall_median = item["wall_exclusive"].get("median_ns")
            if backend == "CPU" and isinstance(wall_median, (int, float)) and isinstance(propagation_wall_median, (int, float)) and propagation_wall_median > 0:
                fraction = wall_median / propagation_wall_median
                item["wall_fraction_of_propagation"] = fraction
                item["amdahl_speedup_2x"] = 1.0 / ((1.0 - fraction) + fraction / 2.0)
                item["amdahl_speedup_4x"] = 1.0 / ((1.0 - fraction) + fraction / 4.0)
            else:
                item["wall_fraction_of_propagation"] = None
                item["amdahl_speedup_2x"] = None
                item["amdahl_speedup_4x"] = None
        gate_status = gate.get("status")
        result.append({
            "backend": backend,
            "shape": shape,
            "cases": ["P1_LIGHT", "P2_LIGHT", "P3_LIGHT"],
            "case_count": len(by_case),
            "status": "REPLAYED" if len(by_case) == 3 and not issues else "INCOMPLETE",
            "timing_gate": gate_status,
            "timing_gate_metrics": gate.get("metrics"),
            "host_wall_semantics": (
                "synchronized CPU host wall clock"
                if backend == "CPU"
                else "async submission/wait host wall clock; not an additive cross-backend operator cost"
            ),
            "occupancy_definition": "process CPU exclusive ns / host wall exclusive ns; diagnostic only and may exceed 1 under parallel CPU execution",
            "amdahl_definition": "1 / ((1-f) + f/s), f=category wall-median / propagation wall-median, s=2 or 4",
            "categories": categories_out,
            "rank_by_process_cpu_median": [item["category"] for item in ranked] if backend == "CPU" else None,
            "rank_scope": "CPU process-CPU median only" if backend == "CPU" else "NOT_COMPUTED_FOR_GPU",
            "closure": closure_rows,
            "closure_all_pass": bool(closure_rows) and all(row["closure_pass"] for row in closure_rows),
            "native_attribution_allowed": backend == "CPU" and gate_status == "PASS_TIMING_GATE" and not issues,
            "cross_backend_operator_difference_allowed": False,
            "scientific_interpretation_allowed": False,
            "issues": sorted(set(issues)),
        })
    return result


def aux_serialized_summaries(
    runs: Sequence[Mapping[str, Any]],
    datasets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep GPU AUX_SERIALIZED ratios separate from formal OFF/P samples."""

    by_key = {
        (str(run.get("backend", "")).upper(), str(run.get("shape", "")), str(run.get("case", "")).upper()): run
        for run in runs
        if isinstance(run, Mapping)
    }
    datasets_by_key = _dataset_by_key(datasets)
    result: list[dict[str, Any]] = []
    for key, aux in sorted(by_key.items()):
        backend, shape, case = key
        if backend != "GPU" or case != "AUX_SERIALIZED":
            continue
        p1 = by_key.get((backend, shape, "P1_OFF"))
        aux_metrics = aux.get("metrics_s") if isinstance(aux.get("metrics_s"), Mapping) else {}
        p1_metrics = p1.get("metrics_s") if isinstance(p1, Mapping) and isinstance(p1.get("metrics_s"), Mapping) else {}
        aux_application = aux.get("recorded_application_service_s")
        if aux_application is None:
            aux_application = aux_metrics.get("application")
        p1_application = p1.get("recorded_application_service_s") if isinstance(p1, Mapping) else None
        if p1_application is None:
            p1_application = p1_metrics.get("application")
        dataset = datasets_by_key.get((backend, shape), {})
        off = dataset.get("off_statistics") if isinstance(dataset, Mapping) else {}
        off_metrics = off.get("metrics") if isinstance(off, Mapping) else {}
        off_application = off_metrics.get("application") if isinstance(off_metrics, Mapping) else {}
        off_median = off_application.get("median_s") if isinstance(off_application, Mapping) else None
        issues: list[str] = []
        ratio_p1 = None
        ratio_off = None
        if isinstance(aux_application, (int, float)) and isinstance(p1_application, (int, float)) and p1_application > 0:
            ratio_p1 = aux_application / p1_application
        else:
            issues.append("AUX_OR_P1_OFF_APPLICATION_MISSING")
        if isinstance(aux_application, (int, float)) and isinstance(off_median, (int, float)) and off_median > 0:
            ratio_off = aux_application / off_median
        else:
            issues.append("AUX_OR_OFF_MEDIAN_APPLICATION_MISSING")
        result.append({
            "backend": backend,
            "shape": shape,
            "case": "AUX_SERIALIZED",
            "mode": aux.get("mode"),
            "status": "REPLAYED" if not issues else "INCOMPLETE",
            "source_path": aux.get("path"),
            "p1_off_source_path": p1.get("path") if isinstance(p1, Mapping) else None,
            "aux_application_s": aux_application,
            "p1_off_application_s": p1_application,
            "off_application_median_s": off_median,
            "aux_to_p1_off_application_ratio": ratio_p1,
            "aux_to_off_application_median_ratio": ratio_off,
            "diagnostic_only": True,
            "matrix_eligible": False,
            "not_off_median": True,
            "not_formal_p_sample": True,
            "scientific_interpretation_allowed": False,
            "issues": sorted(set(issues)),
        })
    return result


_NATIVE_OFF_LEAVES = ("outside", "window_remainder", "later")


def _decimal_seconds(value: Any, field: str) -> Decimal:
    if isinstance(value, bool):
        raise ReplayError(f"NATIVE_OFF_NUMBER_REQUIRED:{field}")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ReplayError(f"NATIVE_OFF_NUMBER_REQUIRED:{field}") from exc
    if not number.is_finite() or number < 0:
        raise ReplayError(f"NATIVE_OFF_FINITE_NONNEGATIVE_REQUIRED:{field}")
    return number


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _fraction_text(value: Fraction, *, digits: int = 50) -> str:
    with localcontext() as context:
        context.prec = digits
        return format(Decimal(value.numerator) / Decimal(value.denominator), "f")


def native_off_macro_summaries(datasets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Recompute independent completed-wall OFF means and three additive leaves.

    CPU and GPU each use their own three OFF records.  No P-case or LIGHT
    timing is used, and the resulting CPU-minus-GPU values are descriptive
    independent-mean differences rather than round-paired attribution.
    """

    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for dataset in datasets:
        if not isinstance(dataset, Mapping):
            continue
        backend = str(dataset.get("backend", "")).upper()
        shape = str(dataset.get("shape", ""))
        if backend in {"CPU", "GPU"} and shape:
            grouped.setdefault(shape, {})[backend] = dataset
    result: list[dict[str, Any]] = []
    for shape, by_backend in sorted(grouped.items()):
        backend_out: dict[str, Any] = {}
        pair_issues: list[str] = []
        for backend in ("CPU", "GPU"):
            dataset = by_backend.get(backend)
            off = dataset.get("off_statistics") if isinstance(dataset, Mapping) else None
            metrics = off.get("metrics") if isinstance(off, Mapping) else None
            metrics = metrics if isinstance(metrics, Mapping) else {}
            means: dict[str, Decimal] = {}
            issues: list[str] = []
            metric_out: dict[str, Any] = {}
            for metric in METRICS:
                item = metrics.get(metric)
                item = item if isinstance(item, Mapping) else {}
                values = item.get("values_s")
                if not isinstance(values, list) or len(values) != 3:
                    issues.append(f"{backend}:{metric}:OFF_THREE_VALUES_REQUIRED")
                    continue
                try:
                    decimals = [_decimal_seconds(value, f"{backend}.{shape}.{metric}.{index}") for index, value in enumerate(values)]
                except ReplayError as exc:
                    issues.append(str(exc))
                    continue
                with localcontext() as context:
                    context.prec = 50
                    mean = sum(decimals, Decimal(0)) / Decimal(3)
                means[metric] = mean
                metric_out[metric] = {
                    "values_s": values,
                    "mean_s": float(mean),
                    "mean_decimal_s": _decimal_text(mean),
                }
            leaves: dict[str, Any] = {}
            if set(means) == set(METRICS):
                with localcontext() as context:
                    context.prec = 50
                    leaf_values = {
                        "outside": means["application"] - means["window"],
                        "window_remainder": means["window"] - means["later"],
                        "later": means["later"],
                    }
                    leaf_sum = sum(leaf_values.values(), Decimal(0))
                    residual = leaf_sum - means["application"]
                for leaf in _NATIVE_OFF_LEAVES:
                    leaves[leaf] = {
                        "mean_s": float(leaf_values[leaf]),
                        "mean_decimal_s": _decimal_text(leaf_values[leaf]),
                    }
                closure_pass = residual == 0
                if not closure_pass:
                    issues.append(f"{backend}:NATIVE_OFF_LEAF_CLOSURE:{_decimal_text(residual)}")
            else:
                closure_pass = False
            gate = dataset.get("timing_gate") if isinstance(dataset, Mapping) else None
            gate_status = gate.get("status") if isinstance(gate, Mapping) else None
            backend_out[backend] = {
                "status": "REPLAYED" if not issues and closure_pass else "INCOMPLETE",
                "off_status": off.get("status") if isinstance(off, Mapping) else "PENDING",
                "timing_gate": gate_status,
                "metrics": metric_out,
                "leaves": leaves,
                "closure": {
                    "application_mean_s": float(means["application"]) if "application" in means else None,
                    "leaf_sum_s": float(sum((item["mean_s"] for item in leaves.values()), 0.0)) if leaves else None,
                    "residual_decimal_s": "0" if closure_pass else None,
                    "pass": closure_pass,
                },
                "source_path": dataset.get("path") if isinstance(dataset, Mapping) else None,
                "issues": sorted(set(issues)),
            }
            pair_issues.extend(issues)
        cpu = backend_out.get("CPU", {})
        gpu = backend_out.get("GPU", {})
        delta_leaves: dict[str, Any] = {}
        if all(isinstance(backend_out.get(name), Mapping) and backend_out[name].get("status") == "REPLAYED" for name in ("CPU", "GPU")):
            def exact_mean(backend: Mapping[str, Any], metric: str) -> Fraction:
                values = backend["metrics"][metric]["values_s"]
                return sum((Fraction(str(value)) for value in values), Fraction(0)) / 3

            cpu_means = {metric: exact_mean(cpu, metric) for metric in METRICS}
            gpu_means = {metric: exact_mean(gpu, metric) for metric in METRICS}
            application_delta_fraction = cpu_means["application"] - gpu_means["application"]
            delta_fraction_by_leaf = {
                "outside": (cpu_means["application"] - cpu_means["window"]) - (gpu_means["application"] - gpu_means["window"]),
                "window_remainder": (cpu_means["window"] - cpu_means["later"]) - (gpu_means["window"] - gpu_means["later"]),
                "later": cpu_means["later"] - gpu_means["later"],
            }
            for leaf in _NATIVE_OFF_LEAVES:
                delta = delta_fraction_by_leaf[leaf]
                delta_leaves[leaf] = {
                    "delta_s": float(delta),
                    "delta_decimal_s": _fraction_text(delta),
                    "delta_fraction": f"{delta.numerator}/{delta.denominator}",
                }
            leaf_delta_sum_fraction = sum(delta_fraction_by_leaf.values(), Fraction(0))
            delta_residual_fraction = leaf_delta_sum_fraction - application_delta_fraction
            application_delta = Decimal(application_delta_fraction.numerator) / Decimal(application_delta_fraction.denominator)
            leaf_delta_sum = Decimal(leaf_delta_sum_fraction.numerator) / Decimal(leaf_delta_sum_fraction.denominator)
            delta_residual = Decimal(delta_residual_fraction.numerator) / Decimal(delta_residual_fraction.denominator)
            delta_closed = delta_residual_fraction == 0
        else:
            application_delta = None
            leaf_delta_sum = None
            delta_residual = None
            application_delta_fraction = None
            leaf_delta_sum_fraction = None
            delta_residual_fraction = None
            delta_closed = False
        gate_ok = all(
            isinstance(backend_out.get(name), Mapping)
            and backend_out[name].get("timing_gate") == "PASS_TIMING_GATE"
            for name in ("CPU", "GPU")
        )
        # OFF application/window/later endpoints are independently complete
        # even when the separate LIGHT qualification gate fails.  Keep the
        # completed-wall macro delta available, while marking operator-level
        # attribution UNKNOWN below.
        pair_status = "REPLAYED_METADATA_ONLY" if not pair_issues and delta_closed else "INCOMPLETE"
        result.append({
            "shape": shape,
            "shape_class": "SMALL" if "117" in shape else "FULL",
            "status": pair_status,
            "backends": backend_out,
            "delta": {
                "leaves": delta_leaves,
                "application_delta_s": float(application_delta) if application_delta is not None else None,
                "application_delta_decimal_s": _fraction_text(application_delta_fraction) if application_delta_fraction is not None else None,
                "application_delta_fraction": f"{application_delta_fraction.numerator}/{application_delta_fraction.denominator}" if application_delta_fraction is not None else None,
                "leaf_delta_sum_s": float(leaf_delta_sum) if leaf_delta_sum is not None else None,
                "leaf_delta_sum_decimal_s": _fraction_text(leaf_delta_sum_fraction) if leaf_delta_sum_fraction is not None else None,
                "leaf_delta_sum_fraction": f"{leaf_delta_sum_fraction.numerator}/{leaf_delta_sum_fraction.denominator}" if leaf_delta_sum_fraction is not None else None,
                "closure_residual_decimal_s": _fraction_text(delta_residual_fraction) if delta_residual_fraction is not None else None,
                "closure_residual_fraction": f"{delta_residual_fraction.numerator}/{delta_residual_fraction.denominator}" if delta_residual_fraction is not None else None,
                "closure_pass": delta_closed,
            },
            "delta_definition": "CPU independent three-OFF arithmetic mean minus GPU independent three-OFF arithmetic mean",
            "clock_domain": "native completed-wall OFF records; no round-paired CPU/GPU comparison",
            "native_operator_delta_status": "DESCRIPTIVE_MACRO_ONLY" if gate_ok else "UNKNOWN",
            "scientific_interpretation_allowed": False,
            "issues": sorted(set(pair_issues)),
        })
    return result


def _executed_source_replay(root: Path) -> dict[str, Any]:
    """Verify packaged original Git blobs, using the frozen PV2 LF function."""
    import sys
    from types import ModuleType
    def verify_manifest(
        bundle_root: Path,
        *,
        role: str | None = None,
        expected_head: str | None = None,
    ) -> dict[str, Any]:
        """Verify one PV2 source bundle without introducing hash semantics."""

        source_root = _resolve_qualified(bundle_root, ("executed_source",))
        manifest_path = _resolve_qualified(bundle_root, ("EXECUTION_SOURCE_MANIFEST.json",))
        if not source_root.is_dir() or source_root.is_symlink():
            raise ReplayError("PACKAGED_SOURCE_ROOT_REQUIRED")
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ReplayError("PACKAGED_SOURCE_MANIFEST_REQUIRED")
        manifest = load_json_strict(manifest_path)
        records = manifest["records"]
        if not isinstance(records, list):
            raise ReplayError("PACKAGED_RECORDS_REQUIRED")
        if expected_head is not None and manifest.get("head") != expected_head:
            raise ReplayError(f"SOURCE_BUNDLE_HEAD_MISMATCH:{role or 'UNKNOWN'}")
        helper_relative = "Filament_python/tools/hpc_ops/provenance_v2.py"
        helper_record = next(row for row in records if row["path"] == helper_relative)

        def read_blob(row: Mapping[str, Any]) -> tuple[Path, bytes]:
            path = _resolve_qualified(source_root, _relative_parts(row["path"]))
            raw = path.read_bytes()
            header = b"blob " + str(len(raw)).encode("ascii") + b"\0"
            oid = str(row["git_blob_oid"])
            if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", oid):
                raise ReplayError("PACKAGED_GIT_BLOB_OID_REQUIRED:" + str(row["path"]))
            actual = (hashlib.sha1 if len(oid) == 40 else hashlib.sha256)(header + raw).hexdigest()
            if actual != oid:
                raise ReplayError("PACKAGED_GIT_BLOB_OID_MISMATCH:" + str(row["path"]))
            return path, raw

        helper, helper_bytes = read_blob(helper_record)
        module_name = "_s46a_frozen_pv2_replay" if role is None else f"_s46a_frozen_pv2_replay_{role.lower()}"
        module = ModuleType(module_name)
        module.__file__ = str(helper)
        sys.modules[module.__name__] = module
        exec(compile(helper_bytes, str(helper), "exec"), module.__dict__)
        rows: list[dict[str, Any]] = []
        for row in records:
            if row.get("classification") != "tracked_text" or row.get("hash_scope") != module.TRACKED_HASH_SCOPE:
                raise ReplayError("PACKAGED_TRACKED_HASH_SCOPE")
            _, raw = read_blob(row)
            canonical = module.canonical_lf_sha256(raw)
            if canonical != row["canonical_lf_sha256"]:
                raise ReplayError("PACKAGED_CANONICAL_LF_MISMATCH:" + str(row["path"]))
            rows.append({"path": row["path"], "git_blob_oid": row["git_blob_oid"], "canonical_lf_sha256": canonical})
        return {
            "role": role,
            "root": bundle_root.relative_to(root).as_posix(),
            "head": manifest["head"],
            "record_count": len(rows),
            "records": rows,
            "issues": [],
        }

    index_path = root / "SOURCE_BUNDLE_INDEX.json"
    if index_path.is_file() and not index_path.is_symlink():
        issues: list[str] = []
        bundles_out: list[dict[str, Any]] = []
        try:
            index = load_json_strict(index_path)
            if not isinstance(index, Mapping) or index.get("schema") != "filament.cpu.s46a.source_bundles.v1":
                raise ReplayError("SOURCE_BUNDLE_INDEX_SCHEMA_INVALID")
            bundles = index.get("bundles")
            if not isinstance(bundles, list) or len(bundles) != 2:
                raise ReplayError("SOURCE_BUNDLE_INDEX_REQUIRES_CPU_GPU")
            seen_roles: set[str] = set()
            seen_roots: set[str] = set()
            for item in bundles:
                if not isinstance(item, Mapping):
                    raise ReplayError("SOURCE_BUNDLE_ENTRY_REQUIRED")
                role = str(item.get("role", "")).upper()
                raw_root = item.get("root")
                commit = item.get("execution_commit")
                if role not in {"CPU", "GPU"} or role in seen_roles:
                    raise ReplayError("SOURCE_BUNDLE_ROLES_MUST_BE_CPU_GPU_ONCE")
                if not isinstance(raw_root, str) or not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", commit):
                    raise ReplayError("SOURCE_BUNDLE_ROOT_AND_COMMIT_REQUIRED")
                bundle_root = _resolve_qualified(root, _relative_parts(raw_root))
                normalized_root = bundle_root.relative_to(root).as_posix()
                if normalized_root in seen_roots or not bundle_root.is_dir() or bundle_root.is_symlink():
                    raise ReplayError("SOURCE_BUNDLE_ROOT_INVALID_OR_DUPLICATE")
                seen_roles.add(role)
                seen_roots.add(normalized_root)
                bundles_out.append(verify_manifest(bundle_root, role=role, expected_head=commit))
            if seen_roles != {"CPU", "GPU"}:
                raise ReplayError("SOURCE_BUNDLE_ROLES_MUST_BE_CPU_GPU_ONCE")
            return {
                "status": "SOURCE_BUNDLES_REPLAYED",
                "index": "SOURCE_BUNDLE_INDEX.json",
                "roles": bundles_out,
                "record_count": sum(item["record_count"] for item in bundles_out),
                "issues": [],
            }
        except (ReplayError, OSError, KeyError, StopIteration) as exc:
            issues.append(str(exc))
            return {"status": "SOURCE_BUNDLE_INVALID", "index": "SOURCE_BUNDLE_INDEX.json", "roles": bundles_out, "issues": issues}

    source_root = root / "executed_source"
    manifest_path = root / "EXECUTION_SOURCE_MANIFEST.json"
    if not source_root.is_dir():
        return {"status": "NOT_PACKAGED", "issues": [], "records": []}
    issues: list[str] = []
    try:
        verified = verify_manifest(root)
        return {"status": "SOURCE_IDENTITIES_REPLAYED", "head": verified["head"], "records": verified["records"], "issues": []}
    except (ReplayError, OSError, KeyError, StopIteration) as exc:
        issues.append(str(exc))
        return {"status": "SOURCE_IDENTITY_INVALID", "records": [], "issues": issues}


def replay_report(root: os.PathLike[str] | str) -> dict[str, Any]:
    """Read and recompute the available report metadata, without writing."""

    base = _report_root(root)
    scope = _scope_paths(base)
    selected = scope["paths"]

    # All normal replay inputs come from the explicit whitelist.  In the
    # absence of one, only current formal raw metadata (or the frozen terminal
    # evidence fallback) is selected above; old versions are never scanned.
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
    case_metadata = _metadata_case_replay(base, case_paths, [])
    matrix_datasets: list[dict[str, Any]] = []
    matrix_deferred: list[dict[str, Any]] = []
    issues: list[str] = [f"SCOPE:{issue}" for issue in scope["issues"]]
    for path in matrix_paths:
        try:
            dataset = _replay_matrix_file(base, path)
            matrix_datasets.append(dataset)
            issues.extend(f"{dataset['path']}:{issue}" for issue in dataset["issues"])
        except ReplayError as exc:
            error = str(exc)
            relative = path.relative_to(base).as_posix()
            if error.startswith("METADATA_TOO_LARGE:"):
                matrix_deferred.append({"path": relative, "status": "DEFERRED", "reason": error})
            else:
                issues.append(f"{relative}:{error}")
    case_datasets = _datasets_from_case_runs(case_metadata["runs"])
    datasets = case_datasets if case_datasets else matrix_datasets
    for dataset in case_datasets:
        issues.extend(f"{dataset['path']}:{issue}" for issue in dataset["issues"])
    matrix_case_consistency = _compare_matrix_case_datasets(matrix_datasets, case_datasets)
    issues.extend(f"MATRIX_CASE:{issue}" for issue in matrix_case_consistency["issues"])
    gates = [{"backend": d["backend"], "shape": d["shape"], **d["timing_gate"]} for d in datasets]
    clocks = _clock_replay(base, timing_paths, [])
    resources = _resource_replay(base, ledger_paths, [])
    manifests = _manifest_replay(base, manifest_paths, [])
    historical = _historical_replay(base)
    executed_source = _executed_source_replay(base)
    ratios = _cpu_gpu_ratios(datasets)
    light_summaries = light_exclusive_summaries(case_metadata["runs"], datasets)
    aux_summaries = aux_serialized_summaries(case_metadata["runs"], datasets)
    native_off_summaries = native_off_macro_summaries(datasets)
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
    if matrix_deferred:
        unverified.extend(
            f"MATRIX_METADATA_DEFERRED:{item['path']}" for item in matrix_deferred
        )
    evidence_present = bool(datasets or case_metadata["runs"] or timing_paths or manifest_paths or ledger_paths)
    status = "REPLAY_COMPLETE_REVIEW_PENDING" if not missing and not issues else ("INCOMPLETE" if evidence_present else "PENDING")
    scope_public = {key: value for key, value in scope.items() if key != "paths"}
    return {
        "schema": SCHEMA,
        "status": status,
        "root": str(base),
        "scope": scope_public,
        "matrix": datasets,
        "matrix_sources": {
            "parsed": matrix_datasets,
            "deferred": matrix_deferred,
            "case_consistency": matrix_case_consistency,
        },
        "timing_gates": gates,
        "cpu_gpu_ratios": ratios,
        "light_exclusive_summaries": light_summaries,
        "aux_serialized_summaries": aux_summaries,
        "native_off_macro_summaries": native_off_summaries,
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


def _gpu_replay_payload(replay: Mapping[str, Any]) -> dict[str, Any]:
    """Project the current GPU metadata replay into a compact JSON receipt."""

    runs = [
        run for run in replay.get("case_metadata", {}).get("runs", [])
        if isinstance(run, Mapping) and str(run.get("backend", "")).upper() == "GPU"
    ]
    datasets = [
        dataset for dataset in replay.get("matrix", [])
        if isinstance(dataset, Mapping) and str(dataset.get("backend", "")).upper() == "GPU"
    ]
    formal_runs: list[dict[str, Any]] = []
    aux_runs: list[dict[str, Any]] = []
    for run in runs:
        ledger = run.get("host_clock_exclusive")
        ledger = ledger if isinstance(ledger, Mapping) else {}
        item = {
            "shape": run.get("shape"),
            "case": run.get("case"),
            "mode": run.get("mode"),
            "status": run.get("status"),
            "matrix_eligible": run.get("matrix_eligible"),
            "matrix_exclusion_reason": run.get("matrix_exclusion_reason"),
            "source_path": run.get("path"),
            "raw_timing_path": run.get("raw_timing_path"),
            "metrics_s": run.get("metrics_s"),
            "process_metrics_s": run.get("process_metrics_s"),
            "first_interval_s": run.get("first_interval_s"),
            "post_timeline": run.get("post_timeline"),
            "completed_window_clocks": run.get("completed_window_clocks"),
            "completed_window_consistency": run.get("completed_window_consistency"),
            "clock_closure": run.get("clock_closure"),
            "host_clock_exclusive": {
                "status": ledger.get("status"),
                "host_wall_semantics": ledger.get("host_wall_semantics"),
                "diagnostic_only": ledger.get("diagnostic_only"),
                "can_fill_off_or_p_samples": ledger.get("can_fill_off_or_p_samples"),
                "propagation": ledger.get("propagation"),
                "categories": ledger.get("categories"),
                "exclusive_sum": ledger.get("exclusive_sum"),
                "cuda_event_spans_ms": ledger.get("cuda_event_spans_ms"),
                "cuda_event_spans_are_idle_inclusive_not_busy": ledger.get(
                    "cuda_event_spans_are_idle_inclusive_not_busy"
                ),
                "cuda_event_spans_not_subtracted_from_host": ledger.get(
                    "cuda_event_spans_not_subtracted_from_host"
                ),
            },
            "deferred_device_event_spans_ms": run.get("deferred_device_event_spans_ms"),
            "issues": run.get("issues"),
        }
        if run.get("matrix_eligible") is True:
            formal_runs.append(item)
        else:
            aux_runs.append(item)
    off_statistics = [
        {
            "backend": dataset.get("backend"),
            "shape": dataset.get("shape"),
            "status": dataset.get("status"),
            "timing_gate": dataset.get("timing_gate"),
            "off_statistics": dataset.get("off_statistics"),
            "matrix_status": dataset.get("matrix_status"),
            "path": dataset.get("path"),
            "issues": dataset.get("issues"),
        }
        for dataset in datasets
    ]
    gate_failures = [
        {
            "shape": dataset.get("shape"),
            "path": dataset.get("path"),
            "status": dataset.get("timing_gate", {}).get("status"),
            "metrics": dataset.get("timing_gate", {}).get("metrics"),
        }
        for dataset in datasets
        if dataset.get("timing_gate", {}).get("status") != "PASS_TIMING_GATE"
    ]
    resource = replay.get("resources", {}).get("backends", {}).get("GPU")
    resource = resource if isinstance(resource, Mapping) else {"status": "PENDING"}
    result_status = (
        "REPLAYED_WITH_TIMING_GATE_FAILURE"
        if gate_failures
        else "REPLAYED_METADATA_ONLY"
    )
    return {
        "schema": "filament.cpu.s4_6a.gpu_replay.v1",
        "status": result_status,
        "report_status": replay.get("status"),
        "scope": replay.get("scope"),
        "matrix_sources": replay.get("matrix_sources"),
        "formal_gpu_runs": formal_runs,
        "aux_serialized_diagnostics": aux_runs,
        "gpu_off_statistics_and_gates": off_statistics,
        "gpu_light_gate_failures": gate_failures,
        "light_exclusive_summaries": replay.get("light_exclusive_summaries"),
        "aux_serialized_summaries": replay.get("aux_serialized_summaries"),
        "native_off_macro_summaries": replay.get("native_off_macro_summaries"),
        "matrix_case_consistency": replay.get("matrix_sources", {}).get("case_consistency"),
        "resource_ledger": resource,
        "limitations": [
            "GPU host clocks are asynchronous submission/wait wall clocks.",
            "CUDA/device event spans are retained separately and never subtracted from host wall clocks.",
            "AUX_SERIALIZED is diagnostic-only and is not an OFF median or P1-P3 sample.",
            "Full-shape LIGHT timing gate failure is retained; OFF statistics are not discarded.",
            "No scientific arrays were opened by this replay.",
        ],
        "missing": replay.get("missing"),
        "issues": replay.get("issues"),
        "unverified": replay.get("unverified"),
        "DEFAULT": "I0",
        "PRODUCTION_QUALIFIED": False,
        "MANUAL_REVIEW_PENDING": True,
        "review_gate": "MANUAL_REVIEW_PENDING",
    }


def write_gpu_replay(root: os.PathLike[str] | str, output: os.PathLike[str] | str) -> dict[str, Any]:
    """Write an explicitly requested compact GPU metadata replay receipt."""

    base = _report_root(root)
    target = Path(output)
    if not target.is_absolute():
        target = base / target
    target = target.absolute()
    if not target.is_relative_to(base):
        raise ReplayError("GPU_REPLAY_OUTPUT_OUTSIDE_REPORT_ROOT")
    replay = replay_report(base)
    payload = _gpu_replay_payload(replay)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "path": target.relative_to(base).as_posix(),
        "status": payload["status"],
        "formal_gpu_runs": len(payload["formal_gpu_runs"]),
        "aux_serialized_diagnostics": len(payload["aux_serialized_diagnostics"]),
        "gpu_off_statistics_and_gates": len(payload["gpu_off_statistics_and_gates"]),
        "DEFAULT": "I0",
        "PRODUCTION_QUALIFIED": False,
        "MANUAL_REVIEW_PENDING": True,
    }


def _all_backends_replay_payload(replay: Mapping[str, Any]) -> dict[str, Any]:
    datasets = replay.get("matrix", [])
    datasets = datasets if isinstance(datasets, list) else []
    gate_failures = [
        {
            "backend": dataset.get("backend"),
            "shape": dataset.get("shape"),
            "status": (dataset.get("timing_gate") or {}).get("status") if isinstance(dataset.get("timing_gate"), Mapping) else None,
        }
        for dataset in datasets
        if isinstance(dataset, Mapping)
        and (dataset.get("timing_gate") or {}).get("status") != "PASS_TIMING_GATE"
    ]
    return {
        "schema": "filament.cpu.s4_6a.all_backends_replay.v1",
        "status": "REPLAYED_WITH_GPU_TIMING_GATE_FAILURE" if any(
            item.get("backend") == "GPU" for item in gate_failures
        ) else "REPLAYED_METADATA_ONLY",
        "report_status": replay.get("status"),
        "scope": replay.get("scope"),
        "datasets": [
            {
                "backend": item.get("backend"),
                "shape": item.get("shape"),
                "status": item.get("status"),
                "path": item.get("path"),
                "matrix_status": item.get("matrix_status"),
                "case_order": item.get("case_order"),
                "case_order_ok": item.get("case_order_ok"),
                "timing_gate": item.get("timing_gate"),
                "off_statistics": item.get("off_statistics"),
                "issues": item.get("issues"),
            }
            for item in datasets
            if isinstance(item, Mapping)
        ],
        "matrix_sources": replay.get("matrix_sources"),
        "executed_source": replay.get("executed_source"),
        "timing_gate_failures": gate_failures,
        "cpu_gpu_ratios": replay.get("cpu_gpu_ratios"),
        "native_off_macro_summaries": replay.get("native_off_macro_summaries"),
        "light_exclusive_summaries": replay.get("light_exclusive_summaries"),
        "aux_serialized_summaries": replay.get("aux_serialized_summaries"),
        "resources": replay.get("resources"),
        "missing": replay.get("missing"),
        "issues": replay.get("issues"),
        "unverified": replay.get("unverified"),
        "limitations": [
            "No scientific arrays were opened; CPU/GPU matrix bytes remain deferred metadata evidence.",
            "CPU full MATRIX_RESULT_CPU_351.json is deferred because it exceeds the 8 MiB metadata parser bound; CASE_RECEIPT and RAW_CPU_TIMING are authoritative for this replay.",
            "GPU full LIGHT timing-gate failure is retained; its three OFF statistics and ratios are not discarded.",
            "GPU host clocks are asynchronous submission/wait wall clocks; CUDA event spans are retained separately and never subtracted from host clocks.",
            "AUX_SERIALIZED is a separate diagnostic ledger and is not an OFF median or formal P sample.",
            "Native OFF macro leaves use independent three-OFF completed-wall arithmetic means; they are not round-paired CPU/GPU attribution.",
            "Full-shape native operator delta remains UNKNOWN because the GPU full LIGHT timing gate failed.",
            "The seven-class ranks are CPU process-CPU median cost ranks only; no GPU-native operator delta is inferred.",
            "This receipt is metadata-only and does not qualify production or establish scientific equivalence.",
        ],
        "DEFAULT": "I0",
        "PRODUCTION_QUALIFIED": False,
        "MANUAL_REVIEW_PENDING": True,
        "review_gate": "MANUAL_REVIEW_PENDING",
    }


def write_all_backends_replay(root: os.PathLike[str] | str, output: os.PathLike[str] | str) -> dict[str, Any]:
    """Write the current four-shape metadata-only replay receipt."""

    base = _report_root(root)
    target = Path(output)
    if not target.is_absolute():
        target = base / target
    target = target.absolute()
    if not target.is_relative_to(base):
        raise ReplayError("ALL_BACKENDS_REPLAY_OUTPUT_OUTSIDE_REPORT_ROOT")
    payload = _all_backends_replay_payload(replay_report(base))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "path": target.relative_to(base).as_posix(),
        "status": payload["status"],
        "report_status": payload["report_status"],
        "dataset_count": len(payload["datasets"]),
        "DEFAULT": "I0",
        "PRODUCTION_QUALIFIED": False,
        "MANUAL_REVIEW_PENDING": True,
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
    parser.add_argument("--gpu-replay-output", default=None, help="write an explicit compact GPU replay receipt under --root")
    parser.add_argument("--all-backends-output", default=None, help="write the four backend/shape metadata replay receipt under --root")
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            result = self_test()
        elif args.gpu_replay_output:
            result = write_gpu_replay(args.root, args.gpu_replay_output)
        elif args.all_backends_output:
            result = write_all_backends_replay(args.root, args.all_backends_output)
        else:
            result = replay_report(args.root)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:  # CLI must remain machine-readable on malformed evidence.
        print(json.dumps({"schema": SCHEMA, "status": "REPLAY_ERROR", "error": str(exc), "DEFAULT": "I0", "PRODUCTION_QUALIFIED": False, "MANUAL_REVIEW_PENDING": True}, ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
