"""Build bounded S46A metadata-only measurement tables.

The builder consumes the local replay output and writes only the three
continuation CSVs. GAP_CONTRIBUTIONS is the native completed-wall OFF macro
ledger; PROFILE_DIAGNOSTIC_COSTS retains the asynchronous LIGHT profile rows.
It never opens matrix arrays or replaces historical report tables. Missing
backend/shape evidence is represented by DRAFT rows; no sample or historical
value is synthesized.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from replay_readonly import METRICS, PAIRS, replay_report


LEDGER_LABELS = (
    "linear",
    "Raman_Heun",
    "ion",
    "otherNL",
    "deposition_POST",
    "diagnostics_output",
    "management_remainder_unknown",
)
EXPECTED_BACKEND_SHAPES = (
    ("CPU", "SMALL"),
    ("CPU", "FULL"),
    ("GPU", "SMALL"),
    ("GPU", "FULL"),
)

RUNTIME_FIELDS = (
    "schema",
    "table_status",
    "row_kind",
    "status",
    "backend",
    "shape",
    "shape_class",
    "sample_id",
    "case",
    "mode",
    "pair",
    "matrix_eligible",
    "matrix_exclusion_reason",
    "source_path",
    "raw_timing_path",
    "application_s",
    "window_s",
    "later_s",
    "first_interval_s",
    "application_process_cpu_s",
    "window_process_cpu_s",
    "later_process_cpu_s",
    "first_interval_process_cpu_s",
    "post_status",
    "post_first_POST_from_application_s",
    "post_adjacent_gap_count",
    "post_adjacent_gap_median_s",
    "post_adjacent_gap_min_s",
    "post_adjacent_gap_max_s",
    "post_adjacent_gaps_s",
    "host_clock_status",
    "host_wall_semantics",
    "host_wall_exclusive_ns",
    "host_process_cpu_exclusive_ns",
    "host_clock_scope",
    "host_clock_diagnostic_only",
    "host_clock_can_fill_off_or_p_samples",
    "cuda_event_spans_ms",
    "summary_category",
    "summary_wall_values_ns",
    "summary_wall_median_ns",
    "summary_wall_range_ns",
    "summary_wall_mad_ns",
    "summary_process_cpu_values_ns",
    "summary_process_cpu_median_ns",
    "summary_process_cpu_range_ns",
    "summary_process_cpu_mad_ns",
    "summary_occupancy_values",
    "summary_occupancy_median",
    "summary_occupancy_range",
    "summary_occupancy_mad",
    "summary_wall_fraction_of_propagation",
    "summary_amdahl_speedup_2x",
    "summary_amdahl_speedup_4x",
    "summary_cpu_cost_rank",
    "summary_wall_cost_rank",
    "summary_closure_all_pass",
    "summary_aux_to_p1_off_application_ratio",
    "summary_aux_to_off_application_median_ratio",
    "dataset_status",
    "dataset_path",
    "gate_application",
    "gate_window",
    "gate_later",
    "off_application_values_s",
    "off_application_median_s",
    "off_application_range_s",
    "off_application_mad_s",
    "off_window_values_s",
    "off_window_median_s",
    "off_window_range_s",
    "off_window_mad_s",
    "off_later_values_s",
    "off_later_median_s",
    "off_later_range_s",
    "off_later_mad_s",
    "resource_actual_core_hours",
    "resource_worst_core_hours",
    "resource_actual_gpu_hours",
    "resource_worst_gpu_hours",
    "resource_job_ids",
    "missing_reason",
    "DEFAULT",
    "PRODUCTION_QUALIFIED",
    "MANUAL_REVIEW_PENDING",
)

GAP_FIELDS = (
    "schema",
    "table_status",
    "status",
    "shape",
    "shape_class",
    "pair",
    "scope",
    "CPU_comparable_s",
    "GPU_comparable_s",
    "delta_s",
    "delta_definition",
    "clock_domain",
    "CPU_status",
    "GPU_status",
    "CPU_source",
    "GPU_source",
    "CPU_mode",
    "GPU_mode",
    "CPU_matrix_eligible",
    "GPU_matrix_eligible",
    "CPU_diagnostic_only",
    "GPU_diagnostic_only",
    "native_operator_delta_status",
    "profile_semantics",
    "missing_reason",
    "scientific_interpretation_allowed",
    "DEFAULT",
    "PRODUCTION_QUALIFIED",
    "MANUAL_REVIEW_PENDING",
)

NATIVE_GAP_FIELDS = (
    "schema",
    "table_status",
    "status",
    "shape",
    "shape_class",
    "scope",
    "CPU_mean_s",
    "GPU_mean_s",
    "delta_s",
    "delta_status",
    "delta_definition",
    "clock_domain",
    "CPU_values_s",
    "GPU_values_s",
    "CPU_leaf_min_s",
    "CPU_leaf_max_s",
    "CPU_leaf_range_s",
    "CPU_leaf_mad_s",
    "GPU_leaf_min_s",
    "GPU_leaf_max_s",
    "GPU_leaf_range_s",
    "GPU_leaf_mad_s",
    "delta_share_of_mean_application_gap",
    "source_workload_scope",
    "leaf_anchor",
    "mutually_exclusive",
    "CPU_source",
    "GPU_source",
    "CPU_off_status",
    "GPU_off_status",
    "CPU_timing_gate",
    "GPU_timing_gate",
    "CPU_leaf_closure",
    "GPU_leaf_closure",
    "pair_leaf_closure",
    "native_operator_delta_status",
    "scientific_interpretation_allowed",
    "missing_reason",
    "DEFAULT",
    "PRODUCTION_QUALIFIED",
    "MANUAL_REVIEW_PENDING",
)


def _shape_class(value: Mapping[str, Any]) -> str:
    text = " ".join(
        str(value.get(key, ""))
        for key in ("shape", "path", "source", "sample_id")
    ).upper()
    return "SMALL" if "SMALL" in text or "117" in text else "FULL"


def _json_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _metric_value(run: Mapping[str, Any], metric: str) -> Any:
    metrics = run.get("metrics_s")
    return metrics.get(metric) if isinstance(metrics, Mapping) else None


def _post(run: Mapping[str, Any]) -> Mapping[str, Any]:
    value = run.get("post_timeline")
    return value if isinstance(value, Mapping) else {}


def _ledger_category(ledger: Mapping[str, Any] | None, category: str, field: str = "wall_exclusive_ns") -> Any:
    if not isinstance(ledger, Mapping):
        return None
    categories = ledger.get("categories")
    row = categories.get(category) if isinstance(categories, Mapping) else None
    return row.get(field) if isinstance(row, Mapping) else None


def _native_leaf_values(backend: Mapping[str, Any], scope: str) -> list[float] | None:
    """Return the three per-OFF-case values for one additive native leaf."""

    metrics = backend.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    values_by_metric: dict[str, list[float]] = {}
    for metric in ("application", "window", "later"):
        item = metrics.get(metric)
        values = item.get("values_s") if isinstance(item, Mapping) else None
        if not isinstance(values, list) or len(values) != 3:
            return None
        try:
            converted = [float(value) for value in values]
        except (TypeError, ValueError, OverflowError):
            return None
        if any(not math.isfinite(value) for value in converted):
            return None
        values_by_metric[metric] = converted
    if scope == "outside":
        return [a - w for a, w in zip(values_by_metric["application"], values_by_metric["window"])]
    if scope == "window_remainder":
        return [w - l for w, l in zip(values_by_metric["window"], values_by_metric["later"])]
    if scope == "later":
        return list(values_by_metric["later"])
    return None


def _leaf_stats(values: Sequence[float] | None) -> dict[str, Any]:
    if not values or any(not math.isfinite(float(value)) for value in values):
        return {"min_s": None, "max_s": None, "range_s": None, "mad_s": None}
    numbers = [float(value) for value in values]
    median = float(statistics.median(numbers))
    return {
        "min_s": min(numbers),
        "max_s": max(numbers),
        "range_s": max(numbers) - min(numbers),
        "mad_s": float(statistics.median([abs(value - median) for value in numbers])),
    }


def _case_row(
    run: Mapping[str, Any],
    *,
    table_status: str,
    dataset: Mapping[str, Any] | None = None,
    row_kind: str = "CASE",
) -> dict[str, Any]:
    post = _post(run)
    ledger = run.get("host_clock_exclusive")
    ledger = ledger if isinstance(ledger, Mapping) else {}
    process = run.get("process_metrics_s")
    process = process if isinstance(process, Mapping) else {}
    row: dict[str, Any] = {
        "schema": "filament.cpu.s4_6a.measurement_tables.v1",
        "table_status": table_status,
        "row_kind": row_kind,
        "status": run.get("status"),
        "backend": run.get("backend"),
        "shape": run.get("shape"),
        "shape_class": _shape_class(run),
        "sample_id": run.get("sample_id"),
        "case": run.get("case"),
        "mode": run.get("mode"),
        "matrix_eligible": run.get("matrix_eligible"),
        "matrix_exclusion_reason": run.get("matrix_exclusion_reason"),
        "source_path": run.get("path"),
        "raw_timing_path": run.get("raw_timing_path"),
        "application_s": _metric_value(run, "application"),
        "window_s": _metric_value(run, "window"),
        "later_s": _metric_value(run, "later"),
        "first_interval_s": run.get("first_interval_s"),
        "application_process_cpu_s": process.get("application"),
        "window_process_cpu_s": process.get("window"),
        "later_process_cpu_s": process.get("later"),
        "first_interval_process_cpu_s": run.get("first_interval_process_cpu_s"),
        "post_status": post.get("status"),
        "post_first_POST_from_application_s": post.get("first_POST_from_application_s"),
        "post_adjacent_gap_count": post.get("adjacent_POST_gap_count"),
        "post_adjacent_gap_median_s": post.get("adjacent_POST_gap_median_s"),
        "post_adjacent_gap_min_s": post.get("adjacent_POST_gap_min_s"),
        "post_adjacent_gap_max_s": post.get("adjacent_POST_gap_max_s"),
        "post_adjacent_gaps_s": post.get("adjacent_POST_gaps_s"),
        "host_clock_status": ledger.get("status"),
        "host_wall_semantics": ledger.get("host_wall_semantics"),
        "host_wall_exclusive_ns": {
            label: _ledger_category(ledger, label) for label in LEDGER_LABELS
        },
        "host_process_cpu_exclusive_ns": {
            label: _ledger_category(ledger, label, "process_cpu_exclusive_ns")
            for label in LEDGER_LABELS
        },
        "host_clock_scope": ledger.get("clock_source"),
        "host_clock_diagnostic_only": ledger.get("diagnostic_only"),
        "host_clock_can_fill_off_or_p_samples": ledger.get("can_fill_off_or_p_samples"),
        "cuda_event_spans_ms": run.get("deferred_device_event_spans_ms"),
        "dataset_status": dataset.get("status") if isinstance(dataset, Mapping) else None,
        "dataset_path": dataset.get("path") if isinstance(dataset, Mapping) else None,
        "DEFAULT": "I0",
        "PRODUCTION_QUALIFIED": False,
        "MANUAL_REVIEW_PENDING": True,
    }
    return row


def _off_row(dataset: Mapping[str, Any], *, table_status: str) -> dict[str, Any]:
    off = dataset.get("off_statistics")
    off = off if isinstance(off, Mapping) else {}
    metrics = off.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    gate = dataset.get("timing_gate")
    gate = gate if isinstance(gate, Mapping) else {}
    gate_metrics = gate.get("metrics")
    gate_metrics = gate_metrics if isinstance(gate_metrics, Mapping) else {}
    row: dict[str, Any] = {
        "schema": "filament.cpu.s4_6a.measurement_tables.v1",
        "table_status": table_status,
        "row_kind": "OFF_STATS",
        "status": dataset.get("status"),
        "backend": dataset.get("backend"),
        "shape": dataset.get("shape"),
        "shape_class": _shape_class(dataset),
        "dataset_status": dataset.get("status"),
        "dataset_path": dataset.get("path"),
        "DEFAULT": "I0",
        "PRODUCTION_QUALIFIED": False,
        "MANUAL_REVIEW_PENDING": True,
    }
    for metric in METRICS:
        item = metrics.get(metric)
        item = item if isinstance(item, Mapping) else {}
        row[f"off_{metric}_values_s"] = item.get("values_s")
        row[f"off_{metric}_median_s"] = item.get("median_s")
        row[f"off_{metric}_range_s"] = item.get("range_s")
        row[f"off_{metric}_mad_s"] = item.get("mad_s")
        gate_item = gate_metrics.get(metric)
        gate_item = gate_item if isinstance(gate_item, Mapping) else {}
        row[f"gate_{metric}"] = gate_item.get("pass")
    return row


def _resource_rows(report: Mapping[str, Any], *, table_status: str) -> list[dict[str, Any]]:
    result = report.get("resources")
    result = result if isinstance(result, Mapping) else {}
    backends = result.get("backends")
    backends = backends if isinstance(backends, Mapping) else {}
    rows: list[dict[str, Any]] = []
    for backend, resource in sorted(backends.items()):
        if not isinstance(resource, Mapping):
            continue
        job_ids = resource.get("job_ids")
        has_jobs = isinstance(job_ids, list) and bool(job_ids)
        resource_status = resource.get("status") if has_jobs else "PENDING"
        resource_issues = list(resource.get("issues", [])) if isinstance(resource.get("issues"), list) else []
        if not has_jobs:
            resource_issues.append("NO_RESOURCE_LEDGER_ROWS_FOR_BACKEND")
        rows.append(
            {
                "schema": "filament.cpu.s4_6a.measurement_tables.v1",
                "table_status": table_status,
                "row_kind": "RESOURCE_LEDGER",
                "status": resource_status,
                "backend": backend,
                "resource_actual_core_hours": resource.get("actual_core_hours"),
                "resource_worst_core_hours": resource.get("worst_case_core_hours"),
                "resource_actual_gpu_hours": resource.get("actual_gpu_hours"),
                "resource_worst_gpu_hours": resource.get("worst_case_gpu_hours"),
                "resource_job_ids": job_ids,
                "missing_reason": ";".join(resource_issues),
                "DEFAULT": "I0",
                "PRODUCTION_QUALIFIED": False,
                "MANUAL_REVIEW_PENDING": True,
            }
        )
    return rows


def _summary_rows(report: Mapping[str, Any], *, table_status: str) -> list[dict[str, Any]]:
    """Expose replayed seven-class and AUX summaries without inventing samples."""

    rows: list[dict[str, Any]] = []
    summaries = report.get("light_exclusive_summaries")
    if isinstance(summaries, list):
        for summary in summaries:
            if not isinstance(summary, Mapping):
                continue
            categories = summary.get("categories")
            categories = categories if isinstance(categories, list) else []
            for item in categories:
                if not isinstance(item, Mapping):
                    continue
                wall = item.get("wall_exclusive")
                wall = wall if isinstance(wall, Mapping) else {}
                process = item.get("process_cpu_exclusive")
                process = process if isinstance(process, Mapping) else {}
                occupancy = item.get("occupancy_process_cpu_over_wall")
                occupancy = occupancy if isinstance(occupancy, Mapping) else {}
                rows.append(
                    {
                        "schema": "filament.cpu.s4_6a.measurement_tables.v1",
                        "table_status": table_status,
                        "row_kind": "LIGHT_EXCLUSIVE_SUMMARY",
                        "status": summary.get("status"),
                        "backend": summary.get("backend"),
                        "shape": summary.get("shape"),
                        "shape_class": _shape_class(summary),
                        "sample_id": "P_LIGHT_SUMMARY",
                        "case": "P1_LIGHT|P2_LIGHT|P3_LIGHT",
                        "mode": "LIGHT",
                        "matrix_eligible": True,
                        "source_path": ";".join(
                            str(item.get("source_path"))
                            for item in summary.get("closure", [])
                            if isinstance(item, Mapping) and item.get("source_path")
                        ),
                        "host_clock_status": "PASS_HOST_CLOCK_EXCLUSIVE" if summary.get("closure_all_pass") else "INCOMPLETE",
                        "host_wall_semantics": summary.get("host_wall_semantics"),
                        "host_wall_exclusive_ns": wall,
                        "host_process_cpu_exclusive_ns": process,
                        "host_clock_scope": "normalized_rows/parent_closed",
                        "host_clock_diagnostic_only": False,
                        "host_clock_can_fill_off_or_p_samples": True,
                        "summary_category": item.get("category"),
                        "summary_wall_values_ns": wall.get("values_ns"),
                        "summary_wall_median_ns": wall.get("median_ns"),
                        "summary_wall_range_ns": wall.get("range_ns"),
                        "summary_wall_mad_ns": wall.get("mad_ns"),
                        "summary_process_cpu_values_ns": process.get("values_ns"),
                        "summary_process_cpu_median_ns": process.get("median_ns"),
                        "summary_process_cpu_range_ns": process.get("range_ns"),
                        "summary_process_cpu_mad_ns": process.get("mad_ns"),
                        "summary_occupancy_values": occupancy.get("values"),
                        "summary_occupancy_median": occupancy.get("median"),
                        "summary_occupancy_range": occupancy.get("range"),
                        "summary_occupancy_mad": occupancy.get("mad"),
                        "summary_wall_fraction_of_propagation": item.get("wall_fraction_of_propagation"),
                        "summary_amdahl_speedup_2x": item.get("amdahl_speedup_2x"),
                        "summary_amdahl_speedup_4x": item.get("amdahl_speedup_4x"),
                        "summary_cpu_cost_rank": item.get("cpu_cost_rank"),
                        "summary_wall_cost_rank": item.get("wall_cost_rank"),
                        "summary_closure_all_pass": summary.get("closure_all_pass"),
                        "dataset_status": summary.get("status"),
                        "dataset_path": f"formal_case_receipts/{summary.get('backend')}/{summary.get('shape')}",
                        "missing_reason": ";".join(summary.get("issues", [])) if isinstance(summary.get("issues"), list) else None,
                        "DEFAULT": "I0",
                        "PRODUCTION_QUALIFIED": False,
                        "MANUAL_REVIEW_PENDING": True,
                    }
                )
    aux_summaries = report.get("aux_serialized_summaries")
    if isinstance(aux_summaries, list):
        for summary in aux_summaries:
            if not isinstance(summary, Mapping):
                continue
            rows.append(
                {
                    "schema": "filament.cpu.s4_6a.measurement_tables.v1",
                    "table_status": table_status,
                    "row_kind": "AUX_SERIALIZED_SUMMARY",
                    "status": summary.get("status"),
                    "backend": summary.get("backend"),
                    "shape": summary.get("shape"),
                    "shape_class": _shape_class(summary),
                    "sample_id": "AUX_SERIALIZED",
                    "case": "AUX_SERIALIZED",
                    "mode": "SERIALIZED",
                    "matrix_eligible": False,
                    "source_path": summary.get("source_path"),
                    "summary_aux_to_p1_off_application_ratio": summary.get("aux_to_p1_off_application_ratio"),
                    "summary_aux_to_off_application_median_ratio": summary.get("aux_to_off_application_median_ratio"),
                    "host_clock_diagnostic_only": True,
                    "host_clock_can_fill_off_or_p_samples": False,
                    "dataset_status": "DIAGNOSTIC_ONLY",
                    "missing_reason": ";".join(summary.get("issues", [])) if isinstance(summary.get("issues"), list) else None,
                    "DEFAULT": "I0",
                    "PRODUCTION_QUALIFIED": False,
                    "MANUAL_REVIEW_PENDING": True,
                }
            )
    return rows


def _missing_rows(
    datasets: Sequence[Mapping[str, Any]],
    runs: Sequence[Mapping[str, Any]],
    *,
    table_status: str,
) -> list[dict[str, Any]]:
    present = {
        (str(item.get("backend", "")).upper(), _shape_class(item))
        for item in datasets
    }
    present.update(
        (str(item.get("backend", "")).upper(), _shape_class(item))
        for item in runs
        if item.get("matrix_eligible") is True
    )
    rows: list[dict[str, Any]] = []
    for backend, shape_class in EXPECTED_BACKEND_SHAPES:
        if (backend, shape_class) in present:
            continue
        rows.append(
            {
                "schema": "filament.cpu.s4_6a.measurement_tables.v1",
                "table_status": table_status,
                "row_kind": "DRAFT_MISSING",
                "status": "DRAFT",
                "backend": backend,
                "shape_class": shape_class,
                "missing_reason": "NO_CURRENT_FORMAL_CASE_RECEIPTS_PACKAGED",
                "DEFAULT": "I0",
                "PRODUCTION_QUALIFIED": False,
                "MANUAL_REVIEW_PENDING": True,
            }
        )
    return rows


def _ledger_run_map(
    runs: Sequence[Mapping[str, Any]],
    dataset_by_key: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for run in runs:
        if run.get("matrix_eligible") is not True:
            continue
        if str(run.get("status", "REPLAYED")).upper() != "REPLAYED":
            continue
        closure = run.get("clock_closure")
        if isinstance(closure, Mapping) and closure.get("status") != "PASS_CLOCK_CLOSURE":
            continue
        if str(run.get("mode", "")).upper() != "LIGHT":
            continue
        case = str(run.get("case", "")).upper()
        if case not in {f"{pair}_LIGHT" for pair in PAIRS}:
            continue
        if dataset_by_key is not None:
            dataset = dataset_by_key.get(
                (str(run.get("backend", "")).upper(), _shape_class(run))
            )
            if dataset is None:
                continue
            gate = dataset.get("timing_gate")
            if dataset.get("status") != "REPLAYED" or not isinstance(gate, Mapping):
                continue
            if gate.get("status") != "PASS_TIMING_GATE":
                continue
        result.setdefault(
            (str(run.get("backend", "")).upper(), _shape_class(run), case),
            run,
        )
    return result


def _gap_rows(
    runs: Sequence[Mapping[str, Any]],
    *,
    table_status: str,
    datasets: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    dataset_by_key = None
    blocked_dataset_keys: set[tuple[str, str]] = set()
    if datasets is not None:
        dataset_by_key = {}
        for dataset in datasets:
            if not isinstance(dataset, Mapping):
                continue
            key = (str(dataset.get("backend", "")).upper(), _shape_class(dataset))
            dataset_by_key[key] = dataset
            gate = dataset.get("timing_gate")
            if dataset.get("status") != "REPLAYED" or not isinstance(gate, Mapping) or gate.get("status") != "PASS_TIMING_GATE":
                blocked_dataset_keys.add(key)
    mapping = _ledger_run_map(runs, dataset_by_key)
    shape_classes = sorted(
        set(shape for _, shape, _ in mapping)
        | {shape for _, shape in EXPECTED_BACKEND_SHAPES}
    )
    rows: list[dict[str, Any]] = []
    for shape_class in shape_classes:
        for pair in PAIRS:
            cpu = mapping.get(("CPU", shape_class, f"{pair}_LIGHT"))
            gpu = mapping.get(("GPU", shape_class, f"{pair}_LIGHT"))
            for scope in LEDGER_LABELS:
                cpu_ledger = cpu.get("host_clock_exclusive") if cpu else None
                gpu_ledger = gpu.get("host_clock_exclusive") if gpu else None
                cpu_status = cpu_ledger.get("status") if isinstance(cpu_ledger, Mapping) else "PENDING"
                gpu_status = gpu_ledger.get("status") if isinstance(gpu_ledger, Mapping) else "PENDING"
                cpu_ns = _ledger_category(cpu_ledger, scope)
                gpu_ns = _ledger_category(gpu_ledger, scope)
                cpu_value = (
                    cpu_ns / 1e9
                    if cpu_status == "PASS_HOST_CLOCK_EXCLUSIVE"
                    and isinstance(cpu_ledger, Mapping)
                    and cpu_ledger.get("diagnostic_only") is not True
                    and cpu_ns is not None
                    else None
                )
                gpu_value = (
                    gpu_ns / 1e9
                    if gpu_status == "PASS_HOST_CLOCK_EXCLUSIVE"
                    and isinstance(gpu_ledger, Mapping)
                    and gpu_ledger.get("diagnostic_only") is not True
                    and gpu_ns is not None
                    else None
                )
                comparable = cpu_value is not None and gpu_value is not None
                missing_reasons = []
                if not comparable:
                    missing_reasons.append("GPU_OR_CPU_FORMAL_LIGHT_LEDGER_MISSING_OR_INVALID")
                    if ("CPU", shape_class) in blocked_dataset_keys:
                        missing_reasons.append("CPU_DATASET_TIMING_GATE_FAIL")
                    if ("GPU", shape_class) in blocked_dataset_keys:
                        missing_reasons.append("GPU_DATASET_TIMING_GATE_FAIL")
                rows.append(
                    {
                        "schema": "filament.cpu.s4_6a.measurement_tables.v1",
                        "table_status": table_status,
                        "status": "PROFILE_DIAGNOSTIC_ONLY" if comparable else "DRAFT",
                        "shape": cpu.get("shape") if cpu else gpu.get("shape") if gpu else None,
                        "shape_class": shape_class,
                        "pair": pair,
                        "scope": scope,
                        "CPU_comparable_s": cpu_value,
                        "GPU_comparable_s": gpu_value,
                        "delta_s": None,
                        "delta_definition": "UNKNOWN: CPU completed-wall and GPU asynchronous-host profile clocks are not a cross-backend delta",
                        "clock_domain": (
                            "PROFILE_DIAGNOSTIC_ONLY_CPU_SYNC_GPU_ASYNC_NO_CROSS_BACKEND_DELTA"
                            if comparable
                            else "NO_MATCHED_COMPLETED_NONOVERLAP_DOMAIN"
                        ),
                        "CPU_status": cpu_status,
                        "GPU_status": gpu_status,
                        "CPU_source": cpu.get("path") if cpu else None,
                        "GPU_source": gpu.get("path") if gpu else None,
                        "CPU_mode": cpu.get("mode") if cpu else None,
                        "GPU_mode": gpu.get("mode") if gpu else None,
                        "CPU_matrix_eligible": cpu.get("matrix_eligible") if cpu else None,
                        "GPU_matrix_eligible": gpu.get("matrix_eligible") if gpu else None,
                        "CPU_diagnostic_only": cpu_ledger.get("diagnostic_only") if isinstance(cpu_ledger, Mapping) else None,
                        "GPU_diagnostic_only": gpu_ledger.get("diagnostic_only") if isinstance(gpu_ledger, Mapping) else None,
                        "native_operator_delta_status": "UNKNOWN",
                        "profile_semantics": "LIGHT seven-class exclusive diagnostic; GPU host clock is asynchronous and never a native operator delta",
                        "missing_reason": ";".join(missing_reasons) if missing_reasons else None,
                        "scientific_interpretation_allowed": False,
                        "DEFAULT": "I0",
                        "PRODUCTION_QUALIFIED": False,
                        "MANUAL_REVIEW_PENDING": True,
                    }
                )
    return rows


def _native_gap_rows(report: Mapping[str, Any], *, table_status: str) -> list[dict[str, Any]]:
    """Build the main completed-wall OFF macro leaf-addition table."""

    summaries = report.get("native_off_macro_summaries")
    if not isinstance(summaries, list):
        return []
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        if not isinstance(summary, Mapping):
            continue
        backends = summary.get("backends")
        backends = backends if isinstance(backends, Mapping) else {}
        cpu = backends.get("CPU") if isinstance(backends.get("CPU"), Mapping) else {}
        gpu = backends.get("GPU") if isinstance(backends.get("GPU"), Mapping) else {}
        delta = summary.get("delta")
        delta = delta if isinstance(delta, Mapping) else {}
        delta_leaves = delta.get("leaves")
        delta_leaves = delta_leaves if isinstance(delta_leaves, Mapping) else {}
        for scope in ("outside", "window_remainder", "later"):
            cpu_leaf = (cpu.get("leaves") or {}).get(scope, {})
            gpu_leaf = (gpu.get("leaves") or {}).get(scope, {})
            cpu_values = _native_leaf_values(cpu, scope)
            gpu_values = _native_leaf_values(gpu, scope)
            cpu_stats = _leaf_stats(cpu_values)
            gpu_stats = _leaf_stats(gpu_values)
            leaf_delta = delta_leaves.get(scope) if isinstance(delta_leaves.get(scope), Mapping) else {}
            application_delta = delta.get("application_delta_s")
            leaf_delta_value = leaf_delta.get("delta_s")
            delta_share = None
            if (
                isinstance(application_delta, (int, float))
                and not isinstance(application_delta, bool)
                and math.isfinite(float(application_delta))
                and float(application_delta) != 0.0
                and isinstance(leaf_delta_value, (int, float))
                and not isinstance(leaf_delta_value, bool)
                and math.isfinite(float(leaf_delta_value))
            ):
                delta_share = float(leaf_delta_value) / float(application_delta)
            leaf_anchor = {
                "outside": "application-window",
                "window_remainder": "window-later",
                "later": "later",
            }[scope]
            status = summary.get("status")
            missing = list(summary.get("issues", [])) if isinstance(summary.get("issues"), list) else []
            if cpu_values is None:
                missing.append(f"CPU_{scope.upper()}_LEAF_VALUES_MISSING")
            if gpu_values is None:
                missing.append(f"GPU_{scope.upper()}_LEAF_VALUES_MISSING")
            if summary.get("native_operator_delta_status") == "UNKNOWN":
                missing.append("GPU_FULL_LIGHT_TIMING_GATE_FAIL_NATIVE_OPERATOR_DELTA_UNKNOWN")
            rows.append(
                {
                    "schema": "filament.cpu.s4_6a.measurement_tables.v1",
                    "table_status": table_status,
                    "status": status,
                    "shape": summary.get("shape"),
                    "shape_class": summary.get("shape_class"),
                    "scope": scope,
                    "CPU_mean_s": cpu_leaf.get("mean_s"),
                    "GPU_mean_s": gpu_leaf.get("mean_s"),
                    "delta_s": leaf_delta.get("delta_s"),
                    "delta_status": "REPLAYED_METADATA_ONLY" if status == "REPLAYED_METADATA_ONLY" else status,
                    "delta_definition": summary.get("delta_definition"),
                    "clock_domain": summary.get("clock_domain"),
                    "CPU_values_s": cpu_values,
                    "GPU_values_s": gpu_values,
                    "CPU_leaf_min_s": cpu_stats["min_s"],
                    "CPU_leaf_max_s": cpu_stats["max_s"],
                    "CPU_leaf_range_s": cpu_stats["range_s"],
                    "CPU_leaf_mad_s": cpu_stats["mad_s"],
                    "GPU_leaf_min_s": gpu_stats["min_s"],
                    "GPU_leaf_max_s": gpu_stats["max_s"],
                    "GPU_leaf_range_s": gpu_stats["range_s"],
                    "GPU_leaf_mad_s": gpu_stats["mad_s"],
                    "delta_share_of_mean_application_gap": delta_share,
                    "source_workload_scope": "fixed three OFF cases P1_OFF,P2_OFF,P3_OFF per backend/shape; excludes LIGHT,AUX,REFERENCE,failed",
                    "leaf_anchor": leaf_anchor,
                    "mutually_exclusive": True,
                    "CPU_source": cpu.get("source_path"),
                    "GPU_source": gpu.get("source_path"),
                    "CPU_off_status": cpu.get("off_status"),
                    "GPU_off_status": gpu.get("off_status"),
                    "CPU_timing_gate": cpu.get("timing_gate"),
                    "GPU_timing_gate": gpu.get("timing_gate"),
                    "CPU_leaf_closure": cpu.get("closure"),
                    "GPU_leaf_closure": gpu.get("closure"),
                    "pair_leaf_closure": delta.get("closure_pass"),
                    "native_operator_delta_status": summary.get("native_operator_delta_status"),
                    "scientific_interpretation_allowed": False,
                    "missing_reason": ";".join(sorted(set(missing))) if missing else None,
                    "DEFAULT": "I0",
                    "PRODUCTION_QUALIFIED": False,
                    "MANUAL_REVIEW_PENDING": True,
                }
            )
    return rows


def build_tables(root: os.PathLike[str] | str) -> dict[str, Any]:
    report = replay_report(root)
    datasets = report.get("matrix", [])
    datasets = datasets if isinstance(datasets, list) else []
    case_metadata = report.get("case_metadata")
    case_metadata = case_metadata if isinstance(case_metadata, Mapping) else {}
    runs = case_metadata.get("runs", [])
    runs = runs if isinstance(runs, list) else []
    # This is a derived, metadata-only table status.  It does not promote the
    # replay lifecycle, qualification gate, or scientific interpretation.
    table_status = "FINAL_MEASURED_LIMITED"
    runtime_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        if isinstance(dataset, Mapping):
            runtime_rows.append(_off_row(dataset, table_status=table_status))
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        dataset = next(
            (
                item for item in datasets
                if isinstance(item, Mapping)
                and item.get("backend") == run.get("backend")
                and item.get("shape") == run.get("shape")
            ),
            None,
        )
        runtime_rows.append(
            _case_row(
                run,
                table_status=table_status,
                dataset=dataset,
                row_kind=(
                    "AUX_SERIALIZED_LEDGER"
                    if str(run.get("mode", "")).upper() in {"AUX_SERIALIZED", "SERIALIZED"}
                    or str(run.get("case", "")).upper() == "AUX_SERIALIZED"
                    else "LIGHT_LEDGER"
                    if str(run.get("mode", "")).upper() == "LIGHT"
                    else "CASE"
                ),
            )
        )
    runtime_rows.extend(_summary_rows(report, table_status=table_status))
    runtime_rows.extend(_resource_rows(report, table_status=table_status))
    runtime_rows.extend(_missing_rows(datasets, runs, table_status=table_status))
    gap_rows = _native_gap_rows(report, table_status=table_status)
    profile_rows = _gap_rows(runs, table_status=table_status, datasets=datasets)
    return {
        "schema": "filament.cpu.s4_6a.measurement_tables.v1",
        "status": table_status,
        "runtime_rows": runtime_rows,
        "gap_rows": gap_rows,
        "profile_rows": profile_rows,
        "report_status": report.get("status"),
        "missing": report.get("missing", []),
        "unverified": report.get("unverified", []),
        "DEFAULT": "I0",
        "PRODUCTION_QUALIFIED": False,
        "MANUAL_REVIEW_PENDING": True,
    }


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise RuntimeError(f"OUTPUT_EXISTS_USE_OVERWRITE:{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_cell(row.get(field)) for field in fields})
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build S46A metadata-only measurement tables")
    parser.add_argument("--root", required=True, help="continuation report root")
    parser.add_argument("--output-dir", default=None, help="table output directory (default: root)")
    parser.add_argument("--overwrite", action="store_true", help="replace only these generated CSVs")
    args = parser.parse_args(argv)
    result = build_tables(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.root)
    runtime_path = output_dir / "MATCHED_RUNTIME_TABLE.csv"
    gap_path = output_dir / "GAP_CONTRIBUTIONS.csv"
    profile_path = output_dir / "PROFILE_DIAGNOSTIC_COSTS.csv"
    _write_csv(runtime_path, RUNTIME_FIELDS, result["runtime_rows"], overwrite=args.overwrite)
    _write_csv(gap_path, NATIVE_GAP_FIELDS, result["gap_rows"], overwrite=args.overwrite)
    _write_csv(profile_path, GAP_FIELDS, result["profile_rows"], overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "runtime_path": str(runtime_path),
                "gap_path": str(gap_path),
                "profile_path": str(profile_path),
                "runtime_rows": len(result["runtime_rows"]),
                "gap_rows": len(result["gap_rows"]),
                "profile_rows": len(result["profile_rows"]),
                "missing": result["missing"],
                "unverified": result["unverified"],
                "DEFAULT": "I0",
                "PRODUCTION_QUALIFIED": False,
                "MANUAL_REVIEW_PENDING": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
