"""Bounded P4 single-GPU block-scaling orchestration and closeout helpers.

This module deliberately only consumes the qualified E5-P worker artifacts.
It neither imports the HR-4 screen update nor changes any numerical setting.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .hr4e5_parallel import HR4C_FIELDS, compare_store_states, open_store_from_spec
from .hr4e_timestep import json_safe, sha256_array, sha256_file
from .hr4e5p_launcher import P3_INDICES, P3_SOURCE_FILE_SHA256


P4_BLOCK_SIZES = (1, 2, 4, 8, 16, 48)
P4_REPETITIONS = (1, 2)
P4_ADDITIONAL_INDICES = (
    208, 625, 1041, 1458, 1875, 2291, 2708, 3125, 3541, 3958, 4375, 4791,
    5208, 5625, 6041, 6458, 6875, 7291, 7708, 8125, 8541, 8958, 9375,
    9791, 10208, 10625, 11041, 11458, 11875, 12291, 12708, 13125, 13541,
    13958, 14375, 14791,
)
P4_SOURCE_INDICES = tuple(sorted((*P3_INDICES, *P4_ADDITIONAL_INDICES)))
assert len(P4_SOURCE_INDICES) == 48 and len(set(P4_SOURCE_INDICES)) == 48


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(dict(value)), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def p4_case_id(block_size: int, repetition: int) -> str:
    if block_size not in P4_BLOCK_SIZES or repetition not in P4_REPETITIONS:
        raise ValueError("P4 requires one authorized block size and repetition")
    return f"p4_b{block_size:02d}_r{repetition:02d}"


def build_input_manifest(validation_input: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the contractually defined P3-plus-stratified 48-screen input set."""
    records = validation_input.get("screen_records")
    if not isinstance(records, list) or len(records) != 48:
        raise ValueError("P4 validation input must contain exactly 48 screens")
    source_sha = str(validation_input.get("source_state_file_sha256", ""))
    if source_sha != P3_SOURCE_FILE_SHA256 or validation_input.get("dtype") != "float64":
        raise ValueError("P4 validation input does not retain the qualified source provenance")
    ordered = sorted(records, key=lambda item: int(item["source_index"]))
    observed = tuple(int(item["source_index"]) for item in ordered)
    if observed != P4_SOURCE_INDICES:
        raise ValueError("P4 screen list is not the frozen P3-plus-stratified set")
    store = open_store_from_spec(validation_input["parallel_state"])
    try:
        frozen = []
        for ordinal, record in enumerate(ordered):
            payload = store.read_authoritative_batch(int(record["ordinal"]), int(record["ordinal"]) + 1)
            frozen.append({
                "ordinal": ordinal, "screen_id": str(record["screen_id"]), "source_index": int(record["source_index"]),
                "z_m": float(record["z_m"]), "source_array_sha256": str(record["source_array_sha256"]),
                "input_delta_n_sha256": sha256_array(payload["delta_n"][0]),
                "input_vx_sha256": sha256_array(payload["vx"][0]),
                "input_vy_sha256": sha256_array(payload["vy"][0]),
            })
    finally:
        store.close()
    return {
        "schema": "khz_filament.hr4e5p.p4_input_manifest.v1", "status": "FROZEN",
        "selection_rule": "all 12 P3 indices plus 36 fixed equally spaced longitudinal-bin centers; ascending source-index order",
        "p3_source_indices": list(P3_INDICES), "additional_stratified_source_indices": list(P4_ADDITIONAL_INDICES),
        "source_state_file_sha256": source_sha, "source_state_array_sha256": validation_input["source_state_array_sha256"],
        "source_manifest_sha256": validation_input["source_manifest_sha256"], "dtype": "float64",
        "geometry": validation_input["geometry"], "screens": frozen,
    }


def write_input_manifest(validation_input_path: str | Path, json_out: str | Path, csv_out: str | Path) -> dict[str, Any]:
    manifest = build_input_manifest(_read_json(validation_input_path))
    _write_json(json_out, manifest)
    _write_csv(csv_out, manifest["screens"], (
        "ordinal", "screen_id", "source_index", "z_m", "source_array_sha256",
        "input_delta_n_sha256", "input_vx_sha256", "input_vy_sha256",
    ))
    return manifest


def _input_hashes(case_dir: Path) -> dict[tuple[int, str], dict[str, str]]:
    result: dict[tuple[int, str], dict[str, str]] = {}
    for path in sorted(case_dir.glob("block_*.npz")):
        with np.load(path, allow_pickle=False) as block:
            metadata = json.loads(str(block["metadata_json"].item()))
        for item in metadata.get("input_screens", []):
            key = (int(item["ordinal"]), str(item["screen_id"]))
            if key in result:
                raise ValueError("P4 worker artifacts repeat one input screen")
            hashes = item.get("array_sha256")
            if not isinstance(hashes, Mapping) or set(hashes) != set(HR4C_FIELDS):
                raise ValueError("P4 worker artifact lacks complete input hashes")
            result[key] = {field: str(hashes[field]) for field in HR4C_FIELDS}
    return result


def _load_case(root: Path, block_size: int, repetition: int, receipt: Mapping[str, str]) -> dict[str, Any]:
    case_id = p4_case_id(block_size, repetition)
    directory = root / case_id
    required = (directory / "validation_input.json", directory / "partition.json", directory / "worker_0000.json", directory / "gather_result.json")
    if any(not path.is_file() for path in required):
        raise FileNotFoundError(f"P4 case artifacts incomplete: {case_id}")
    validation = _read_json(required[0]); partition = _read_json(required[1]); worker = _read_json(required[2]); gather = _read_json(required[3])
    if int(partition.get("block_size", -1)) != block_size or int(partition.get("n_workers", -1)) != 1:
        raise ValueError("P4 partition resource or block size differs from its case identity")
    records = validation.get("screen_records", [])
    if len(records) != 48 or tuple(int(item["source_index"]) for item in records) != P4_SOURCE_INDICES:
        raise ValueError("P4 case input manifest differs from frozen 48 screens")
    if gather.get("status") != "PASS" or int(gather.get("n_screens", -1)) != 48:
        raise ValueError("P4 gather did not complete all 48 screens")
    if str(validation.get("source_state_file_sha256", "")) != P3_SOURCE_FILE_SHA256:
        raise ValueError("P4 case source hash differs from P3 prerequisite")
    if case_id not in receipt:
        raise ValueError("P4 case is missing from submission receipt")
    resource_samples = directory / "gpu_memory_mib.csv"
    if not resource_samples.is_file():
        raise FileNotFoundError(f"P4 GPU resource monitor is missing: {case_id}")
    gpu_mib = [int(line.strip()) for line in resource_samples.read_text(encoding="utf-8").splitlines() if line.strip().isdigit()]
    if not gpu_mib:
        raise ValueError("P4 GPU resource monitor did not yield a numeric sample")
    outputs = worker.get("outputs", [])
    timings = [float(item["walltime_s"]) for item in outputs]
    if len(outputs) != 48 // block_size or len(worker.get("screen_timings", [])) != 48:
        raise ValueError("P4 worker block or screen timing count is invalid")
    memory = dict(worker.get("memory", {}))
    worker_time = float(worker["walltime_s"]); gather_time = float(gather["walltime_s"])
    return {
        "case_id": case_id, "block_size": block_size, "repetition": repetition, "job_id": receipt[case_id],
        "directory": str(directory), "validation": validation, "gather": gather, "worker": worker,
        "input_hashes": _input_hashes(directory), "executor_walltime_s": worker_time + gather_time,
        "scientific_execution_time_s": worker_time, "gather_time_s": gather_time, "screens_per_s": 48.0 / worker_time,
        "seconds_per_screen": worker_time / 48.0, "n_blocks": len(outputs),
        "first_block_time_s": timings[0], "median_block_time_s": statistics.median(timings),
        "output_write_time_s": None, "gpu_peak_memory_bytes": max(gpu_mib) * 1024 * 1024,
        "gpu_pool_peak_bytes": memory.get("gpu_pool_total_bytes"),
        "gpu_device_total_bytes": memory.get("gpu_device_total_bytes"), "host_peak_rss_bytes": memory.get("cpu_max_rss_bytes"),
        "backend": memory.get("backend"), "output_manifest_status": gather.get("status"),
    }


def _receipt(path: Path) -> dict[str, str]:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline=""), delimiter="\t"))
    expected = {p4_case_id(block, repeat) for block in P4_BLOCK_SIZES for repeat in P4_REPETITIONS}
    found = {str(row.get("case_id", "")): str(row.get("job_id", "")) for row in rows}
    if set(found) != expected or any(not value.isdecimal() for value in found.values()):
        raise ValueError("P4 submission receipt must contain exactly the 12 authorized numeric job IDs")
    return found


def _comparison(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = compare_store_states(
        reference_state=reference["gather"]["output_state"], candidate_state=candidate["gather"]["output_state"],
        records=reference["validation"]["screen_records"],
    )
    input_equal = reference["input_hashes"] == candidate["input_hashes"]
    rows: list[dict[str, Any]] = []
    for screen in report["screens"]:
        for field, comparison in screen["fields"].items():
            rows.append({
                "reference_case_id": reference["case_id"], "candidate_case_id": candidate["case_id"],
                "ordinal": screen["ordinal"], "screen_id": screen["screen_id"], "field": field,
                "shape_equal": comparison["shape_equal"], "dtype_equal": comparison["dtype_equal"],
                "input_hash_equal": input_equal, "canonical_output_hash_equal": comparison.get("reference_sha256") == comparison.get("candidate_sha256"),
                "array_equal": comparison["exact_equal"], "reference_sha256": comparison.get("reference_sha256"),
                "candidate_sha256": comparison.get("candidate_sha256"), "differing_elements": comparison.get("differing_elements"),
            })
    exact = bool(report["exact_equal"]) and input_equal and all(row["canonical_output_hash_equal"] for row in rows)
    return {"candidate_case_id": candidate["case_id"], "exact_equal": exact, "raw_comparison": report}, rows


def finalize_p4(root_path: str | Path, manifest_path: str | Path, receipt_path: str | Path) -> dict[str, Any]:
    """Produce all required P4 result artifacts, fail closed on incomplete evidence."""
    root = Path(root_path); manifest = _read_json(manifest_path); receipt = _receipt(Path(receipt_path))
    if manifest.get("schema") != "khz_filament.hr4e5p.p4_input_manifest.v1" or len(manifest.get("screens", [])) != 48:
        raise ValueError("P4 input manifest is invalid")
    cases = [_load_case(root, block, repeat, receipt) for block in P4_BLOCK_SIZES for repeat in P4_REPETITIONS]
    reference = next(case for case in cases if case["case_id"] == p4_case_id(1, 1))
    equivalence, flattened = [], []
    for case in cases:
        if case is reference:
            continue
        report, rows = _comparison(reference, case)
        equivalence.append(report); flattened.extend(rows)
    if not all(item["exact_equal"] for item in equivalence):
        raise RuntimeError("P4_BLOCK_SIZE_EQUIVALENCE_FAIL")
    timing_rows = [{key: case[key] for key in (
        "case_id", "block_size", "repetition", "job_id", "executor_walltime_s", "scientific_execution_time_s", "gather_time_s",
        "screens_per_s", "seconds_per_screen", "n_blocks", "first_block_time_s", "median_block_time_s", "output_write_time_s", "output_manifest_status",
    )} for case in cases]
    resource_rows = [{key: case[key] for key in ("case_id", "block_size", "repetition", "job_id", "backend", "gpu_peak_memory_bytes", "gpu_pool_peak_bytes", "gpu_device_total_bytes", "host_peak_rss_bytes")} for case in cases]
    summaries = []
    for block in P4_BLOCK_SIZES:
        subset = [case for case in cases if case["block_size"] == block]
        summaries.append({
            "block_size": block, "median_executor_walltime_s": statistics.median(case["executor_walltime_s"] for case in subset),
            "median_scientific_execution_time_s": statistics.median(case["scientific_execution_time_s"] for case in subset),
            "median_screens_per_s": statistics.median(case["screens_per_s"] for case in subset),
            "median_seconds_per_screen": statistics.median(case["seconds_per_screen"] for case in subset),
            "throughput_spread_screens_per_s": max(case["screens_per_s"] for case in subset) - min(case["screens_per_s"] for case in subset),
            "max_gpu_peak_memory_bytes": max(case["gpu_peak_memory_bytes"] for case in subset),
            "max_gpu_pool_peak_bytes": max(case["gpu_pool_peak_bytes"] or 0 for case in subset),
            "max_host_peak_rss_bytes": max(case["host_peak_rss_bytes"] or 0 for case in subset),
        })
    best = max(row["median_screens_per_s"] for row in summaries); plateau = [row for row in summaries if row["median_screens_per_s"] >= 0.95 * best]
    candidate = min(plateau, key=lambda row: row["block_size"])["block_size"]
    baseline = next(row["median_screens_per_s"] for row in summaries if row["block_size"] == 1)
    for row in summaries:
        row["relative_throughput_vs_block_1"] = row["median_screens_per_s"] / baseline
        row["within_95_percent_of_best"] = row in plateau
    _write_json(root / "p4_run_manifest.json", {"schema": "khz_filament.hr4e5p.p4_run_manifest.v1", "input_manifest_sha256": sha256_file(manifest_path), "receipt": receipt, "runs": timing_rows})
    _write_json(root / "p4_timing_results.json", {"schema": "khz_filament.hr4e5p.p4_timing.v1", "runs": timing_rows}); _write_csv(root / "p4_timing_results.csv", timing_rows, tuple(timing_rows[0]))
    _write_json(root / "p4_resource_results.json", {"schema": "khz_filament.hr4e5p.p4_resource.v1", "runs": resource_rows}); _write_csv(root / "p4_resource_results.csv", resource_rows, tuple(resource_rows[0]))
    _write_json(root / "p4_equivalence_results.json", {"schema": "khz_filament.hr4e5p.p4_equivalence.v1", "reference_case_id": reference["case_id"], "comparisons": equivalence, "field_comparisons": flattened})
    _write_csv(root / "p4_equivalence_results.csv", flattened, tuple(flattened[0]))
    _write_csv(root / "p4_block_scaling_summary.csv", summaries, tuple(summaries[0]))
    decision = {"schema": "khz_filament.hr4e5p.p4_final_decision.v1", "decision": "HR-4E-5P-P4 = CLOSED / SINGLE_GPU_BLOCK_SCALING_PASS", "candidate_block_size": candidate, "best_median_screens_per_s": best, "plateau_threshold_screens_per_s": 0.95 * best, "plateau_block_sizes": [row["block_size"] for row in plateau], "exact_field_comparisons": len(flattened), "mismatch_count": 0, "p5_started": False}
    _write_json(root / "p4_final_decision.json", decision)
    table = "\n".join(f"| {row['block_size']} | {row['median_scientific_execution_time_s']:.3f} | {row['median_screens_per_s']:.4f} | {row['relative_throughput_vs_block_1']:.3f} | {row['max_gpu_peak_memory_bytes']} |" for row in summaries)
    (root / "HR4E5P_P4_CLOSEOUT.md").write_text("# HR-4E-5P P4 Closeout\n\n**HR-4E-5P-P4 = CLOSED / SINGLE_GPU_BLOCK_SCALING_PASS**\n\nP3 prerequisite: `EXACT_EQUIVALENCE_PASS`; P4 used the fixed 48-screen manifest and two one-GPU runs per block size.\n\n| block | median scientific s | screens/s | relative to block 1 | peak GPU bytes |\n|---:|---:|---:|---:|---:|\n" + table + f"\n\nSelected block size: **{candidate}** (smallest within 95% of best median throughput).\n\nExact field comparisons: {len(flattened)}; mismatches: 0. P5 was not run.\n", encoding="utf-8", newline="\n")
    return decision
