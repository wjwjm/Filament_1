"""P5 fixed-workload multi-GPU scaling orchestration and CPU adjudication."""

from __future__ import annotations

import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .hr4e5_parallel import HR4C_FIELDS, compare_store_states, open_store_from_spec
from .hr4e5p_launcher import P3_SOURCE_FILE_SHA256
from .hr4e5p_p4 import P4_SOURCE_INDICES
from .hr4e_timestep import json_safe, sha256_array, sha256_file


P5_GPU_COUNTS = (1, 2, 4)
P5_REPETITIONS = (1, 2)
P5_BLOCK_SIZE = 8
P5_ADDITIONAL_INDICES = tuple(int(np.floor((index + 0.5) * 15000 / 144)) for index in range(144))
P5_SOURCE_INDICES = tuple(sorted((*P4_SOURCE_INDICES, *P5_ADDITIONAL_INDICES)))
assert len(P5_SOURCE_INDICES) == 192 and len(set(P5_SOURCE_INDICES)) == 192


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(dict(value)), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, target)


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader(); writer.writerows(rows)


def _read_json(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def p5_case_id(gpu_count: int, repetition: int) -> str:
    if gpu_count not in P5_GPU_COUNTS or repetition not in P5_REPETITIONS:
        raise ValueError("P5 requires an authorized GPU count and repetition")
    return f"p5_g{gpu_count:02d}_r{repetition:02d}"


def build_input_manifest(validation_input: Mapping[str, Any]) -> dict[str, Any]:
    records = validation_input.get("screen_records")
    if not isinstance(records, list) or len(records) != 192 or validation_input.get("dtype") != "float64":
        raise ValueError("P5 validation input must contain exactly 192 float64 screens")
    if str(validation_input.get("source_state_file_sha256", "")) != P3_SOURCE_FILE_SHA256:
        raise ValueError("P5 input does not retain the qualified source file provenance")
    ordered = sorted(records, key=lambda item: int(item["source_index"]))
    if tuple(int(item["source_index"]) for item in ordered) != P5_SOURCE_INDICES:
        raise ValueError("P5 screen list is not the frozen P4-plus-stratified set")
    store = open_store_from_spec(validation_input["parallel_state"])
    try:
        screens = []
        for ordinal, record in enumerate(ordered):
            payload = store.read_authoritative_batch(int(record["ordinal"]), int(record["ordinal"]) + 1)
            screens.append({
                "ordinal": ordinal, "screen_id": str(record["screen_id"]), "source_index": int(record["source_index"]),
                "z_m": float(record["z_m"]), "source_array_sha256": str(record["source_array_sha256"]),
                "input_delta_n_sha256": sha256_array(payload["delta_n"][0]), "input_vx_sha256": sha256_array(payload["vx"][0]),
                "input_vy_sha256": sha256_array(payload["vy"][0]), "shape": list(payload["delta_n"][0].shape), "dtype": str(payload["delta_n"][0].dtype),
            })
    finally:
        store.close()
    return {
        "schema": "khz_filament.hr4e5p.p5_input_manifest.v1", "status": "FROZEN", "block_size": P5_BLOCK_SIZE,
        "selection_rule": "all 48 frozen P4 screens plus 144 fixed equal longitudinal-bin centers; ascending source-index order",
        "p4_source_indices": list(P4_SOURCE_INDICES), "additional_stratified_source_indices": list(P5_ADDITIONAL_INDICES),
        "source_manifest_sha256": validation_input["source_manifest_sha256"], "source_state_file_sha256": validation_input["source_state_file_sha256"],
        "source_state_array_sha256": validation_input["source_state_array_sha256"], "geometry": validation_input["geometry"], "dtype": "float64", "screens": screens,
    }


def write_input_manifest(input_path: str | Path, json_out: str | Path, csv_out: str | Path) -> dict[str, Any]:
    manifest = build_input_manifest(_read_json(input_path))
    _write_json(json_out, manifest)
    _write_csv(csv_out, manifest["screens"], ("ordinal", "screen_id", "source_index", "z_m", "source_array_sha256", "shape", "dtype", "input_delta_n_sha256", "input_vx_sha256", "input_vy_sha256"))
    return manifest


def _frozen_inputs(manifest: Mapping[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    screens = manifest.get("screens")
    if not isinstance(screens, list) or len(screens) != 192 or manifest.get("block_size") != P5_BLOCK_SIZE:
        raise ValueError("P5 frozen manifest is invalid")
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for ordinal, item in enumerate(screens):
        key = (int(item.get("ordinal", -1)), str(item.get("screen_id", "")))
        hashes = {field: str(item.get(f"input_{field}_sha256", "")) for field in HR4C_FIELDS}
        if key[0] != ordinal or not key[1] or key in result or any(len(value) != 64 for value in hashes.values()):
            raise ValueError("P5 manifest identity or input hash is invalid")
        result[key] = {"source_index": int(item["source_index"]), "z_m": float(item["z_m"]), "source_array_sha256": str(item["source_array_sha256"]), "shape": list(item["shape"]), "dtype": str(item["dtype"]), "hashes": hashes}
    if tuple(item["source_index"] for item in result.values()) != P5_SOURCE_INDICES:
        raise ValueError("P5 manifest source order is invalid")
    return result


def _input_hashes(case_dir: Path) -> dict[tuple[int, str], dict[str, str]]:
    result: dict[tuple[int, str], dict[str, str]] = {}
    for path in sorted(case_dir.glob("block_*.npz")):
        with np.load(path, allow_pickle=False) as block:
            metadata = json.loads(str(block["metadata_json"].item()))
        for item in metadata.get("input_screens", []):
            key = (int(item["ordinal"]), str(item["screen_id"]))
            hashes = item.get("array_sha256")
            if key in result or not isinstance(hashes, Mapping) or set(hashes) != set(HR4C_FIELDS):
                raise ValueError("P5 worker input artifact is invalid")
            result[key] = {field: str(hashes[field]) for field in HR4C_FIELDS}
    return result


def _resource_peaks(path: Path, gpu_count: int) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError("P5 GPU monitor artifact is missing")
    result: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            result[parts[0]] = max(result.get(parts[0], 0), int(parts[1]))
    if len(result) != gpu_count:
        raise ValueError("P5 GPU monitor does not contain one peak for every allocated GPU")
    return result


def _receipt(path: Path) -> dict[str, str]:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline=""), delimiter="\t"))
    expected = {p5_case_id(gpu, repeat) for gpu in P5_GPU_COUNTS for repeat in P5_REPETITIONS}
    found = {str(row.get("case_id", "")): str(row.get("job_id", "")) for row in rows}
    if set(found) != expected or any(not item.isdecimal() for item in found.values()):
        raise ValueError("P5 receipt must contain exactly six numeric mandatory job IDs")
    return found


def _load_case(root: Path, gpu_count: int, repetition: int, receipt: Mapping[str, str], frozen: Mapping[tuple[int, str], Mapping[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    case_id = p5_case_id(gpu_count, repetition); directory = root / case_id
    required = (directory / "validation_input.json", directory / "partition.json", directory / "gather_result.json", directory / "p5_gpu_memory_mib.csv")
    if any(not item.is_file() for item in required):
        raise FileNotFoundError(f"P5 case artifacts incomplete: {case_id}")
    validation, partition, gather = (_read_json(required[0]), _read_json(required[1]), _read_json(required[2]))
    workers = [_read_json(item) for item in sorted(directory.glob("worker_*.json"))]
    if len(workers) != gpu_count or case_id not in receipt or int(partition.get("block_size", -1)) != P5_BLOCK_SIZE or int(partition.get("n_workers", -1)) != gpu_count:
        raise ValueError("P5 case resource or worker layout is invalid")
    records = validation.get("screen_records", []); keys = [(int(item["ordinal"]), str(item["screen_id"])) for item in records]
    if keys != list(frozen) or tuple(int(item["source_index"]) for item in records) != P5_SOURCE_INDICES:
        raise ValueError("P5 case screen identity differs from the frozen manifest")
    for record, key in zip(records, keys, strict=True):
        expected = frozen[key]
        if int(record["source_index"]) != expected["source_index"] or float(record["z_m"]) != expected["z_m"] or str(record["source_array_sha256"]) != expected["source_array_sha256"]:
            raise ValueError("P5 case source identity differs from the frozen manifest")
    for key in ("source_manifest_sha256", "source_state_file_sha256", "source_state_array_sha256", "geometry", "dtype"):
        if validation.get(key) != manifest.get(key):
            raise ValueError("P5 case provenance/grid differs from the frozen manifest")
    if gather.get("status") != "PASS" or int(gather.get("n_screens", -1)) != 192:
        raise ValueError("P5 gather is incomplete")
    input_hashes = _input_hashes(directory)
    if set(input_hashes) != set(frozen) or any(input_hashes[key] != frozen[key]["hashes"] for key in frozen):
        raise ValueError("P5 worker input hashes differ from frozen manifest")
    expected_blocks = 24 // gpu_count; balance = []
    for index, worker in enumerate(workers):
        outputs = worker.get("outputs", [])
        if int(worker.get("worker_index", -1)) != index or len(outputs) != expected_blocks:
            raise ValueError("P5 worker block allocation is imbalanced or malformed")
        balance.append({"case_id": case_id, "job_id": receipt[case_id], "gpu_count": gpu_count, "repeat_index": repetition, "worker_index": index, "blocks_assigned": expected_blocks, "blocks_completed": len(outputs), "worker_scientific_runtime_s": float(worker["walltime_s"]), "worker_idle_time_s": None})
    runtime = [row["worker_scientific_runtime_s"] for row in balance]; memory = [dict(item.get("memory", {})) for item in workers]
    peaks = _resource_peaks(required[3], gpu_count)
    return {
        "case_id": case_id, "job_id": receipt[case_id], "gpu_count": gpu_count, "repeat_index": repetition, "directory": str(directory),
        "validation": validation, "gather": gather, "input_hashes": input_hashes, "input_provenance_status": "PASS", "balance": balance,
        "scientific_execution_time_s": max(runtime), "gather_time_s": float(gather["walltime_s"]), "executor_walltime_s": max(runtime) + float(gather["walltime_s"]),
        "screens_per_s": 192.0 / max(runtime), "seconds_per_screen": max(runtime) / 192.0, "total_blocks": 24, "blocks_per_gpu": expected_blocks,
        "slowest_worker_runtime_s": max(runtime), "fastest_worker_runtime_s": min(runtime), "imbalance": (max(runtime) - min(runtime)) / max(runtime),
        "gpu_peak_memory_mib_by_device": peaks, "peak_gpu_memory_mib_per_gpu": max(peaks.values()), "peak_host_rss_bytes": max(item.get("cpu_max_rss_bytes") or 0 for item in memory),
        "peak_gpu_pool_bytes_per_gpu": max(item.get("gpu_pool_total_bytes") or 0 for item in memory), "completion_status": gather["status"],
        "validation_input_sha256": sha256_file(required[0]), "gather_result_sha256": sha256_file(required[2]),
    }


def _compare(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = compare_store_states(reference_state=reference["gather"]["output_state"], candidate_state=candidate["gather"]["output_state"], records=reference["validation"]["screen_records"])
    input_equal = reference["input_hashes"] == candidate["input_hashes"] and reference["input_provenance_status"] == candidate["input_provenance_status"] == "PASS"
    shape_left, shape_right = list(reference["gather"]["output_state"]["shape"]), list(candidate["gather"]["output_state"]["shape"])
    dtype_left, dtype_right = str(reference["gather"]["output_state"]["dtype"]), str(candidate["gather"]["output_state"]["dtype"])
    rows = []
    for screen in report["screens"]:
        for field, value in screen["fields"].items():
            hash_equal = value.get("reference_sha256") == value.get("candidate_sha256")
            passed = input_equal and value["shape_equal"] and value["dtype_equal"] and value["exact_equal"] and hash_equal
            rows.append({"reference_job_id": reference["job_id"], "candidate_job_id": candidate["job_id"], "reference_case_id": reference["case_id"], "candidate_case_id": candidate["case_id"], "candidate_gpu_count": candidate["gpu_count"], "candidate_repeat_index": candidate["repeat_index"], "ordinal": screen["ordinal"], "screen_id": screen["screen_id"], "field": field, "reference_shape": shape_left, "candidate_shape": shape_right, "shape_equal": value["shape_equal"], "reference_dtype": dtype_left, "candidate_dtype": dtype_right, "dtype_equal": value["dtype_equal"], "reference_canonical_hash": value.get("reference_sha256"), "candidate_canonical_hash": value.get("candidate_sha256"), "canonical_hash_equal": hash_equal, "array_equal": value["exact_equal"], "input_provenance_status": "PASS" if input_equal else "FAIL", "input_hash_equal": input_equal, "final_field_status": "PASS" if passed else "FAIL", "mismatch_reason": "" if passed else "exact_comparison_or_input_provenance_mismatch"})
    return {"candidate_case_id": candidate["case_id"], "candidate_job_id": candidate["job_id"], "comparisons_expected": 576, "comparisons_completed": len(rows), "array_equal_pass_count": sum(bool(row["array_equal"]) for row in rows), "canonical_hash_pass_count": sum(bool(row["canonical_hash_equal"]) for row in rows), "mismatch_count": sum(row["final_field_status"] != "PASS" for row in rows), "provenance_mismatch_count": sum(not bool(row["input_hash_equal"]) for row in rows)}, rows


def finalize_p5(root_path: str | Path, manifest_path: str | Path, receipt_path: str | Path) -> dict[str, Any]:
    root = Path(root_path); manifest = _read_json(manifest_path); frozen = _frozen_inputs(manifest); receipt = _receipt(Path(receipt_path))
    cases = [_load_case(root, gpu, repeat, receipt, frozen, manifest) for gpu in P5_GPU_COUNTS for repeat in P5_REPETITIONS]
    reference = next(item for item in cases if item["case_id"] == p5_case_id(1, 1)); comparisons, fields = [], []
    for case in cases:
        if case is reference:
            continue
        summary, rows = _compare(reference, case); comparisons.append(summary); fields.extend(rows)
    if len(fields) != 2880 or any(row["final_field_status"] != "PASS" for row in fields):
        raise RuntimeError("P5_MULTI_GPU_EQUIVALENCE_FAIL")
    timing = [{key: case[key] for key in ("case_id", "job_id", "gpu_count", "repeat_index", "scientific_execution_time_s", "executor_walltime_s", "gather_time_s", "screens_per_s", "seconds_per_screen", "total_blocks", "blocks_per_gpu", "slowest_worker_runtime_s", "fastest_worker_runtime_s", "imbalance", "completion_status")} for case in cases]
    resources = [{key: case[key] for key in ("case_id", "job_id", "gpu_count", "repeat_index", "gpu_peak_memory_mib_by_device", "peak_gpu_memory_mib_per_gpu", "peak_gpu_pool_bytes_per_gpu", "peak_host_rss_bytes")} for case in cases]
    balance = [row for case in cases for row in case["balance"]]
    _write_json(root / "p5_run_manifest.json", {"schema": "khz_filament.hr4e5p.p5_run_manifest.v1", "input_manifest_sha256": sha256_file(manifest_path), "receipt": receipt, "runs": timing})
    _write_json(root / "p5_timing_results.json", {"schema": "khz_filament.hr4e5p.p5_timing.v1", "runs": timing}); _write_csv(root / "p5_timing_results.csv", timing, tuple(timing[0]))
    _write_csv(root / "p5_resource_results.csv", resources, tuple(resources[0])); _write_csv(root / "p5_worker_balance.csv", balance, tuple(balance[0]))
    exact = {"schema": "khz_filament.hr4e5p.p5_exact_equivalence.v1", "canonical_reference": {"case_id": reference["case_id"], "job_id": reference["job_id"], "gpu_count": 1, "repeat_index": 1, "output_directory": reference["directory"], "validation_input_sha256": reference["validation_input_sha256"], "gather_result_sha256": reference["gather_result_sha256"]}, "per_run": comparisons, "field_comparisons": fields}
    _write_json(root / "p5_exact_equivalence_results.json", exact); _write_csv(root / "p5_exact_equivalence_results.csv", fields, tuple(fields[0]))
    exact_summary = {"schema": "khz_filament.hr4e5p.p5_exact_equivalence_summary.v1", "compared_run_count": 5, "expected_comparison_count": 2880, "completed_comparison_count": len(fields), "shape_mismatch_count": sum(not bool(row["shape_equal"]) for row in fields), "dtype_mismatch_count": sum(not bool(row["dtype_equal"]) for row in fields), "canonical_hash_mismatch_count": sum(not bool(row["canonical_hash_equal"]) for row in fields), "array_mismatch_count": sum(not bool(row["array_equal"]) for row in fields), "provenance_mismatch_count": sum(not bool(row["input_hash_equal"]) for row in fields), "status": "PASS"}
    _write_json(root / "p5_exact_equivalence_summary.json", exact_summary)
    scaling = []
    for gpu in P5_GPU_COUNTS:
        subset = [case for case in cases if case["gpu_count"] == gpu]
        scaling.append({"gpu_count": gpu, "repeats": 2, "median_scientific_execution_time_s": statistics.median(item["scientific_execution_time_s"] for item in subset), "median_screens_per_s": statistics.median(item["screens_per_s"] for item in subset), "median_seconds_per_screen": statistics.median(item["seconds_per_screen"] for item in subset), "median_gather_time_s": statistics.median(item["gather_time_s"] for item in subset), "peak_gpu_memory_mib_per_gpu": max(item["peak_gpu_memory_mib_per_gpu"] for item in subset), "peak_host_rss_bytes": max(item["peak_host_rss_bytes"] for item in subset), "median_imbalance": statistics.median(item["imbalance"] for item in subset)})
    baseline = scaling[0]["median_screens_per_s"]
    for row in scaling:
        row["speedup"] = row["median_screens_per_s"] / baseline; row["parallel_efficiency"] = row["speedup"] / row["gpu_count"]
    projections = [{"gpu_count": row["gpu_count"], "screens_per_s": row["median_screens_per_s"], "projected_15000_s": 15000.0 / row["median_screens_per_s"], "projected_15000_min": 250.0 / row["median_screens_per_s"], "projected_15000_h": 250.0 / (60.0 * row["median_screens_per_s"]), "screens_per_hour": 3600.0 * row["median_screens_per_s"], "scope": "P5 empirical full-z hydro walltime projection; excludes optical, mapping, queue, external checkpoint I/O and streaming overlap"} for row in scaling]
    _write_csv(root / "p5_scaling_summary.csv", scaling, tuple(scaling[0])); _write_csv(root / "p5_fullz_projection.csv", projections, tuple(projections[0]))
    preflight = _read_json(root / "p5_submission_preflight.json")
    decision = {"schema": "khz_filament.hr4e5p.p5_final_decision.v1", "decision": "HR-4E-5P-P5 = CLOSED / MULTI_GPU_SCALING_QUALIFIED", "p5_status": "CLOSED / MULTI_GPU_SCALING_QUALIFIED", "parent_status": "HR-4E-5P = CLOSED / PARALLEL_FULL_Z_EXECUTION_QUALIFIED", "scientific_execution_sha": preflight["git_sha"], "frozen_input_manifest_sha256": sha256_file(manifest_path), "canonical_reference": exact["canonical_reference"], "expected_comparison_count": 2880, "completed_comparison_count": len(fields), "shape_mismatch_count": 0, "dtype_mismatch_count": 0, "canonical_hash_mismatch_count": 0, "array_mismatch_count": 0, "provenance_mismatch_count": 0, "scaling": scaling, "fullz_projection": projections, "optional_8gpu_status": "NOT_ATTEMPTED", "streaming_started": False, "full_e5_started": False}
    _write_json(root / "p5_final_decision.json", decision)
    rows = "\n".join(f"| {row['gpu_count']} | {row['median_scientific_execution_time_s']:.3f} | {row['median_screens_per_s']:.4f} | {row['speedup']:.3f} | {row['parallel_efficiency']:.3f} | {row['median_imbalance']:.4f} | {next(item['projected_15000_h'] for item in projections if item['gpu_count'] == row['gpu_count']):.2f} |" for row in scaling)
    (root / "HR4E5P_P5_CLOSEOUT.md").write_text("# HR-4E-5P P5 Closeout\n\n**HR-4E-5P-P5 = CLOSED / MULTI_GPU_SCALING_QUALIFIED**\n\n| GPUs | median scientific s | screens/s | speedup | efficiency | imbalance | projected 15k h |\n|---:|---:|---:|---:|---:|---:|---:|\n" + rows + f"\n\nExact comparisons: {len(fields)} / 2880; all shape, dtype, canonical-hash, array and provenance mismatch counts are zero.\n\nOptional 8-GPU point: not attempted. Streaming and full E5 were not started.\n", encoding="utf-8", newline="\n")
    return decision
