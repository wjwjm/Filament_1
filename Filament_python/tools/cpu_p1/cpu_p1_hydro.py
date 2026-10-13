#!/usr/bin/env python3
"""CPU-only HR-4 screen diagnostics; never a production qualification."""
from __future__ import annotations

import argparse
import csv
import contextlib
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

os.environ["UPPE_USE_GPU"] = "0"

from tools.cpu_p1 import cpu_p1_fft as common

SCHEMA = "filament.cpu_p1_hydro_benchmark.v1"
CLASSIFICATION = "P1_DIAGNOSTIC_ONLY"
QUALIFICATION = "NOT_PRODUCTION_QUALIFIED"
MANIFEST_SCHEMA = "cpu_p1.historical_real_screen_inputs.v1"
INPUT_IDENTITY = "HISTORICAL_S5_REFERENCE_48_REAL_POST_NOT_S4R_384"
STREAM_SCHEMA = "khz_filament.hr4e5s.streaming.v1"
FIELDS = ("delta_n", "vx", "vy")
EXPECTED_SHAPE = (351, 301)
EXPECTED_DTYPE = "float64"
EXPECTED_MANIFEST_RAW_SHA256 = "f273cfeb2f112de62be9a787d58bc2a7c40ac44f5d7e9978c057dfb9d3aa552a"
EXPECTED_DX_M = 1e-5
EXPECTED_DY_M = 1e-5
EXPECTED_HYDRO_PARAMETERS = {
    "block_size": 8, "cfl_limit": 1.0, "chi": 2.17e-5, "dt_hydro": 1e-6,
    "gravity_x": 0.0, "gravity_y": -9.81, "n0": 1.00027, "n_hydro_steps": 1000,
    "nu": 1.5e-5, "queue_depth": 16,
}
WORKERS = (1, 8, 24, 48, 96)
TASKS_PER_WORKER = 4
SCREEN_COUNT = 48
STEPS = 1000
MAX_SECONDS = 3300
A0_MAX_SECONDS = 900
MEMORY_BYTES = 320 * 1024**3
RSS_FRACTION = 0.80


class HydroError(RuntimeError):
    pass


def sync_json_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def sync_jsonl_append(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def task_ordinal(flat_task_index: int, task_count: int) -> int:
    if task_count <= 0 or not 0 <= flat_task_index < task_count:
        raise ValueError("flat task index/count is invalid")
    if task_count <= SCREEN_COUNT:
        return math.floor(flat_task_index * SCREEN_COUNT / task_count)
    return flat_task_index % SCREEN_COUNT


def worker_plan(worker_count: int) -> list[dict[str, object]]:
    if worker_count not in WORKERS:
        raise HydroError(f"unsupported worker count: {worker_count}")
    plan = []
    for worker_index in range(worker_count):
        ordinals = [task_ordinal(worker_index * TASKS_PER_WORKER + task_index,
                                 worker_count * TASKS_PER_WORKER)
                    for task_index in range(TASKS_PER_WORKER)]
        plan.append({
            "worker_index": worker_index,
            "task_count": TASKS_PER_WORKER,
            "task_ordinals": ordinals,
        })
    return plan


def input_scaling_class(worker_count: int) -> str:
    return ("REPEATED_REAL_INPUT_SCALING_ONLY"
            if worker_count * TASKS_PER_WORKER > SCREEN_COUNT
            else "REAL_INPUT_FIXED_TASKS_PER_WORKER")


def load_manifest(path: Path, input_root: Path, raw_file_hash) -> tuple[dict, dict[str, dict[int, dict]]]:
    if not path.is_file():
        raise HydroError(f"input manifest unavailable: {path}")
    raw_sha = raw_file_hash(path)
    if raw_sha != EXPECTED_MANIFEST_RAW_SHA256:
        raise HydroError(f"input manifest raw SHA256 mismatch: {raw_sha}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HydroError(f"input manifest cannot be read: {type(exc).__name__}: {exc}") from exc
    expected = {
        "schema": MANIFEST_SCHEMA,
        "identity": INPUT_IDENTITY,
        "diagnostic_only": True,
        "config_sha256": "eaec83ad326a29af95881912db76a7e5c26943dec9e0d6876ca9c9cc2a100c22",
        "source_manifest_sha256": "3e2c7557e5e8027ac1d9fb42411fe885fe3ea13bef9ee60e91538462adaf32d2",
        "dtype": EXPECTED_DTYPE,
        "shape": list(EXPECTED_SHAPE),
        "unique_real_screens": SCREEN_COUNT,
    }
    mismatches = {key: {"expected": value, "actual": manifest.get(key)}
                  for key, value in expected.items() if manifest.get(key) != value}
    hydro = manifest.get("hydro", {})
    if hydro != EXPECTED_HYDRO_PARAMETERS:
        mismatches["hydro_parameters"] = {"expected": EXPECTED_HYDRO_PARAMETERS,
                                           "actual": hydro}
    if manifest.get("dx_m") != EXPECTED_DX_M or manifest.get("dy_m") != EXPECTED_DY_M:
        mismatches["spacing"] = {"expected_dx_m": EXPECTED_DX_M,
                                  "expected_dy_m": EXPECTED_DY_M,
                                  "actual_dx_m": manifest.get("dx_m"),
                                  "actual_dy_m": manifest.get("dy_m")}
    if mismatches:
        raise HydroError("input manifest identity rejected: " + json.dumps(mismatches, sort_keys=True))
    if not input_root.is_dir():
        raise HydroError(f"input root unavailable: {input_root}")
    by_namespace: dict[str, dict[int, dict]] = {name: {} for name in ("current", "post", "next")}
    for record in manifest.get("records", []):
        namespace = str(record.get("namespace", "")).lower()
        ordinal = record.get("ordinal")
        if namespace not in by_namespace or not isinstance(ordinal, int) or ordinal in by_namespace[namespace]:
            raise HydroError("input manifest contains unknown or duplicate namespace/ordinal")
        rel = Path(str(record.get("relative_path", "")))
        if rel.is_absolute() or ".." in rel.parts:
            raise HydroError("input manifest path is not a safe relative path")
        resolved = (input_root / rel).resolve()
        if input_root.resolve() not in resolved.parents:
            raise HydroError("input manifest path escapes input root")
        if record.get("hash_scope") != "raw_bytes" or not isinstance(record.get("sha256"), str):
            raise HydroError("input file hash scope or value is invalid")
        if not isinstance(record.get("field_sha256"), dict):
            raise HydroError("input canonical field hashes are missing")
        by_namespace[namespace][ordinal] = record
    wanted = set(range(SCREEN_COUNT))
    if any(set(records) != wanted for records in by_namespace.values()):
        raise HydroError("input manifest does not contain all 48 current/post/next records")
    return {"manifest": manifest, "raw_sha256": raw_sha}, by_namespace


def record_path(input_root: Path, record: dict) -> Path:
    rel = Path(record["relative_path"])
    path = (input_root / rel).resolve()
    if input_root.resolve() not in path.parents:
        raise HydroError("input path escapes the staged root")
    return path


def load_screen(record: dict, input_root: Path, raw_file_hash, canonical_array_hash) -> tuple[dict, dict]:
    import numpy as np

    path = record_path(input_root, record)
    if not path.is_file():
        raise HydroError(f"input screen missing: {record['relative_path']}")
    if path.stat().st_size != int(record.get("size_bytes", -1)):
        raise HydroError(f"input screen byte size changed: {record['relative_path']}")
    started = time.perf_counter()
    actual_raw = raw_file_hash(path)
    raw_hash_s = time.perf_counter() - started
    if actual_raw != record["sha256"]:
        raise HydroError(f"raw SHA256 mismatch: {record['relative_path']}")
    started = time.perf_counter()
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not set((*FIELDS, "metadata_json")).issubset(archive.files):
                raise HydroError(f"NPZ fields missing: {record['relative_path']}")
            metadata_value = archive["metadata_json"]
            metadata = json.loads(str(metadata_value.item()))
            arrays = {name: np.array(archive[name], copy=True, order="K") for name in FIELDS}
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise HydroError(f"NPZ unreadable: {record['relative_path']}: {type(exc).__name__}: {exc}") from exc
    load_s = time.perf_counter() - started
    expected_identity = {
        "schema": STREAM_SCHEMA,
        "namespace": str(record["namespace"]).upper(),
        "ordinal": int(record["ordinal"]),
        "screen_id": record["screen_id"],
        "z_m": float(record["z_m"]),
        "shape": list(EXPECTED_SHAPE),
        "dtype": EXPECTED_DTYPE,
    }
    identity_bad = {key: {"expected": value, "actual": metadata.get(key)}
                    for key, value in expected_identity.items() if metadata.get(key) != value}
    if identity_bad:
        raise HydroError("embedded screen metadata mismatch: " + json.dumps(identity_bad, sort_keys=True))
    expected_hashes = record["field_sha256"]
    if set(expected_hashes) != set(FIELDS) or metadata.get("field_sha256") != expected_hashes:
        raise HydroError("embedded/manifest field hashes differ")
    hash_started = time.perf_counter()
    for name, array in arrays.items():
        if array.shape != EXPECTED_SHAPE or array.dtype != np.dtype(EXPECTED_DTYPE):
            raise HydroError(f"{name} shape/dtype mismatch: {array.shape}/{array.dtype}")
        if not array.flags.c_contiguous:
            raise HydroError(f"{name} is not C-contiguous")
        if not np.isfinite(array).all():
            raise HydroError(f"{name} contains NaN or Inf")
        if canonical_array_hash(array) != expected_hashes[name]:
            raise HydroError(f"canonical field SHA256 mismatch: {name}")
    canonical_s = time.perf_counter() - hash_started
    return arrays, {
        "raw_sha256": actual_raw,
        "canonical_field_sha256": expected_hashes,
        "metadata": metadata,
        "raw_hash_seconds": raw_hash_s,
        "load_seconds": load_s,
        "canonical_hash_seconds": canonical_s,
    }


def field_error_metrics(candidate, reference) -> dict[str, float]:
    import numpy as np

    got, ref = np.asarray(candidate), np.asarray(reference)
    if got.shape != ref.shape or got.dtype != np.float64 or ref.dtype != np.float64:
        raise HydroError("numerical comparison requires matching float64 fields")
    if not np.isfinite(got).all() or not np.isfinite(ref).all():
        raise HydroError("numerical comparison encountered NaN or Inf")
    diff = got - ref
    max_abs = float(np.max(np.abs(diff), initial=0.0))
    max_ref = float(np.max(np.abs(ref), initial=0.0))
    err_norm = float(np.linalg.norm(diff.ravel()))
    ref_norm = float(np.linalg.norm(ref.ravel()))
    tiny = float(np.finfo(np.float64).tiny)
    values = {"max_abs": max_abs, "max_rel": max_abs / max(max_ref, tiny),
              "l2_rel": err_norm / max(ref_norm, tiny)}
    if not all(math.isfinite(value) for value in values.values()):
        raise HydroError("numerical comparison produced a non-finite metric")
    return values


def compare_fields(candidate: dict, reference: dict) -> dict:
    return {name: field_error_metrics(candidate[name], reference[name]) for name in FIELDS}


def _numeric_observable_delta(candidate, reference, prefix="") -> dict:
    output = {}
    for key in sorted(set(candidate) & set(reference)):
        left, right = candidate[key], reference[key]
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(left, dict) and isinstance(right, dict):
            output.update(_numeric_observable_delta(left, right, name))
        elif (isinstance(left, (int, float)) and not isinstance(left, bool)
              and isinstance(right, (int, float)) and not isinstance(right, bool)):
            a, b = float(left), float(right)
            if not math.isfinite(a) or not math.isfinite(b):
                raise HydroError(f"non-finite observable: {name}")
            output[name] = {"candidate": a, "reference": b, "delta": a - b,
                            "max_rel": abs(a - b) / max(abs(b), float.fromhex("0x1.0p-1022"))}
    return output


def _jsonable(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise HydroError("non-finite value in scientific result")
    return value


def invoke_hydro(advance, fields: dict, manifest: dict) -> dict:
    params = manifest["hydro"]
    result = advance(fields["delta_n"], fields["vx"], fields["vy"],
                     dx=float(manifest["dx_m"]), dy=float(manifest["dy_m"]),
                     dt_hydro=float(params["dt_hydro"]), chi=float(params["chi"]),
                     nu=float(params["nu"]), n0=float(params["n0"]),
                     gravity_x=float(params["gravity_x"]), gravity_y=float(params["gravity_y"]),
                     cfl_limit=float(params["cfl_limit"]), n_steps=STEPS, require_stable=True)
    if result.get("steps") != STEPS or result.get("performance", {}).get("backend") != "numpy":
        raise HydroError("frozen HR-4 call returned unexpected step count or backend")
    return result


def _load_science():
    os.environ["UPPE_USE_GPU"] = "0"
    from KHz_filament.device import debug_backend
    from KHz_filament.hr4 import advance_hr4_single_screen
    from KHz_filament.hr4e_timestep import sha256_array
    from tools.hpc_ops.provenance_v2 import raw_sha256_file
    import threadpoolctl

    backend = debug_backend()
    if backend.get("backend") != "numpy":
        raise HydroError(f"CPU backend required, got {backend}")
    return advance_hr4_single_screen, sha256_array, raw_sha256_file, threadpoolctl, backend


def _write_result_arrays(path: Path, fields: dict) -> tuple[float, float, float, str, dict]:
    import numpy as np
    from tools.hpc_ops.provenance_v2 import raw_sha256_file
    from KHz_filament.hr4e_timestep import sha256_array

    write_started = time.perf_counter()
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("xb") as stream:
        np.savez(stream, **{name: fields[name] for name in FIELDS})
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(path)
    os.rename(temp, path)
    write_s = time.perf_counter() - write_started
    start = time.perf_counter()
    raw_hash = raw_sha256_file(path)
    raw_hash_s = time.perf_counter() - start
    start = time.perf_counter()
    field_hashes = {name: sha256_array(fields[name]) for name in FIELDS}
    canonical_hash_s = time.perf_counter() - start
    return write_s, raw_hash_s, canonical_hash_s, raw_hash, field_hashes


def run_one_task(task: dict, *, input_root: Path, output_root: Path, manifest: dict,
                 records: dict[str, dict[int, dict]], previous_fields: dict | None,
                 advance, canonical_array_hash, raw_file_hash, worker_cpu: int | None) -> tuple[dict, dict]:
    import numpy as np

    task_started = time.perf_counter()
    post, post_timing = load_screen(records["post"][task["ordinal"]], input_root,
                                    raw_file_hash, canonical_array_hash)
    next_fields, next_timing = load_screen(records["next"][task["ordinal"]], input_root,
                                           raw_file_hash, canonical_array_hash)
    load_s = post_timing["load_seconds"] + next_timing["load_seconds"]
    raw_hash_s = post_timing["raw_hash_seconds"] + next_timing["raw_hash_seconds"]
    canonical_s = post_timing["canonical_hash_seconds"] + next_timing["canonical_hash_seconds"]
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    result = invoke_hydro(advance, post, manifest)
    kernel_wall, kernel_cpu = time.perf_counter() - start_wall, time.process_time() - start_cpu
    output_fields = {name: np.asarray(result[name]) for name in FIELDS}
    for name, array in output_fields.items():
        if array.shape != EXPECTED_SHAPE or array.dtype != np.dtype(EXPECTED_DTYPE):
            raise HydroError(f"HR-4 output {name} shape/dtype mismatch")
        if not array.flags.c_contiguous or not np.isfinite(array).all():
            raise HydroError(f"HR-4 output {name} layout/numeric validation failed")
    reference_comparison = compare_fields(output_fields, next_fields)
    repeat_comparison = compare_fields(output_fields, previous_fields) if previous_fields is not None else None
    output_path = output_root / f"{task['task_id']}.npz"
    write_s, output_hash_s, output_canonical_hash_s, output_raw_hash, output_field_hashes = \
        _write_result_arrays(output_path, output_fields)
    observable = _jsonable(result.get("observables", {}))
    reference_observable = _jsonable(_load_reference_observables(next_fields, manifest))
    elapsed = time.perf_counter() - task_started
    receipt = {
        "schema": SCHEMA, "status": "DONE", "classification": CLASSIFICATION,
        "production_qualification": QUALIFICATION,
        "comparison_scope": "HISTORICAL_GPU_NEXT_SAME_POST_DIAGNOSTIC_ONLY",
        "comparison_service_note": "S4R_SERVICE_NOT_DIRECTLY_COMPARABLE",
        "task_id": task["task_id"], "case_id": task["case_id"],
        "worker_index": task["worker_index"], "worker_cpu": worker_cpu,
        "ordinal": task["ordinal"], "screen_id": task["screen_id"], "z_m": task["z_m"],
        "input_post": {"relative_path": records["post"][task["ordinal"]]["relative_path"],
                       "raw_sha256": post_timing["raw_sha256"],
                       "canonical_field_sha256": post_timing["canonical_field_sha256"]},
        "input_historical_next": {"relative_path": records["next"][task["ordinal"]]["relative_path"],
                                  "raw_sha256": next_timing["raw_sha256"],
                                  "canonical_field_sha256": next_timing["canonical_field_sha256"]},
        "timing_seconds": {"post_load": post_timing["load_seconds"],
                            "next_load": next_timing["load_seconds"], "load_total": load_s,
                            "raw_hash_total": raw_hash_s, "canonical_hash_total": canonical_s,
                            "kernel_wall": kernel_wall, "kernel_cpu": kernel_cpu,
                            "kernel_reported_wall": result["performance"]["wall_time_s_total"],
                            "output_write": write_s, "output_raw_hash": output_hash_s,
                            "output_canonical_hash": output_canonical_hash_s,
                            "service_total": elapsed},
        "steps": result["steps"], "stability": _jsonable(result.get("stability")),
        "performance": _jsonable(result["performance"]),
        "observables": observable,
        "historical_next_field_comparison": reference_comparison,
        "historical_next_observable_comparison": _numeric_observable_delta(
            observable, reference_observable),
        "cpu_repeat_comparison": repeat_comparison,
        "output": {"relative_path": output_path.name, "raw_sha256": output_raw_hash,
                   "canonical_field_sha256": output_field_hashes},
    }
    path = output_root / f"{task['task_id']}.json"
    sync_json_new(path, receipt)
    return receipt, output_fields


def _load_reference_observables(fields, manifest):
    """Compute scalar observables for the historical NEXT arrays with frozen HR-4 diagnostics."""
    from KHz_filament.hr4 import thermal_channel_observables
    return thermal_channel_observables(fields["delta_n"], fields["vx"], fields["vy"],
                                       dx=float(manifest["dx_m"]), dy=float(manifest["dy_m"]))


def _task_error_receipt(task: dict, exc: BaseException) -> dict:
    return {"schema": SCHEMA, "status": "FAIL", "classification": CLASSIFICATION,
            "production_qualification": QUALIFICATION, "task_id": task["task_id"],
            "case_id": task["case_id"], "ordinal": task["ordinal"],
            "screen_id": task["screen_id"],
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _worker_main(worker_index: int, worker_cpu: int | None, tasks: list[dict], *,
                 input_root: str, output_root: str, manifest: dict, raw_hash, canonical_hash,
                 advance, threadpoolctl) -> None:
    output = Path(output_root)
    if worker_cpu is not None:
        if not hasattr(os, "sched_setaffinity"):
            raise HydroError("Linux CPU affinity API unavailable")
        os.sched_setaffinity(0, {worker_cpu})
        if set(os.sched_getaffinity(0)) != {worker_cpu}:
            raise HydroError("worker CPU affinity did not bind to its assigned physical core")
    case_name = tasks[0]["case_id"] if tasks else "empty"
    sync_json_new(output / f"{case_name}-worker-{worker_index:03d}-started.json", {
        "worker_index": worker_index, "worker_cpu": worker_cpu,
        "started_monotonic": time.monotonic(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    })
    previous_by_ordinal = {}
    with threadpoolctl.threadpool_limits(limits=1):
        for task in tasks:
            try:
                receipt, previous = run_one_task(
                    task, input_root=Path(input_root), output_root=output, manifest=manifest,
                    records=task["_records"],
                    previous_fields=previous_by_ordinal.get(task["ordinal"]), advance=advance,
                    canonical_array_hash=canonical_hash, raw_file_hash=raw_hash,
                    worker_cpu=worker_cpu)
                previous_by_ordinal[task["ordinal"]] = previous
            except BaseException as exc:
                sync_json_new(output / f"{task['task_id']}.json", _task_error_receipt(task, exc))
                raise


def _case_tasks(case_id: str, worker_count: int, records: dict[str, dict[int, dict]]) -> list[list[dict]]:
    workers = []
    for worker_index in range(worker_count):
        tasks = []
        for task_index in range(TASKS_PER_WORKER):
            flat_index = worker_index * TASKS_PER_WORKER + task_index
            ordinal = task_ordinal(flat_index, worker_count * TASKS_PER_WORKER)
            source = records["post"][ordinal]
            task_id = f"{case_id}-w{worker_index:03d}-t{task_index:02d}-s{ordinal:03d}"
            tasks.append({"task_id": task_id, "case_id": case_id,
                          "worker_index": worker_index, "task_index": task_index,
                          "ordinal": ordinal, "screen_id": source["screen_id"],
                          "z_m": source["z_m"], "_records": records})
        workers.append(tasks)
    return workers


def _run_a0_child(args, manifest, records, advance, canonical_hash, raw_hash, threadpoolctl) -> dict:
    record = records["post"][24]
    tasks = [{"task_id": f"A0-repeat-{index + 1}-s024", "case_id": "A0",
              "worker_index": 0, "task_index": index, "ordinal": 24,
              "screen_id": record["screen_id"], "z_m": record["z_m"], "_records": records}
             for index in range(2)]
    cpu = int(args.case_cpus.split(",")[0]) if args.case_cpus else None
    _worker_main(0, cpu, tasks, input_root=args.input_root, output_root=args.task_root,
                 manifest=manifest, raw_hash=raw_hash, canonical_hash=canonical_hash,
                 advance=advance, threadpoolctl=threadpoolctl)
    receipts = [json.loads((Path(args.task_root) / f"{task['task_id']}.json").read_text(encoding="utf-8"))
                for task in tasks]
    gate = _a0_repeat_gate(receipts)
    status = "COMPLETE" if gate["hash_match"] else "CPU_REPEAT_DRIFT"
    return {"case_id": "A0", "status": status, "workers": 1,
            "input_mode": "SAME_REAL_POST_REPEAT_2", "completed_tasks": 2,
            "cpu_repeat_hash_match": gate["hash_match"],
            "cpu_repeat_canonical_hashes": gate["hashes"],
            "stop_reason": None if gate["hash_match"] else "CPU_REPEAT_DRIFT"}


def _a0_repeat_gate(receipts: list[dict]) -> dict:
    hashes = [receipt.get("output", {}).get("canonical_field_sha256") for receipt in receipts]
    valid = len(hashes) == 2 and all(isinstance(value, dict) and set(value) == set(FIELDS)
                                     for value in hashes)
    return {"hash_match": bool(valid and hashes[0] == hashes[1]), "hashes": hashes}


def _stop_processes(processes) -> None:
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except (AssertionError, OSError):
            pass
    for process in processes:
        try:
            process.join(timeout=3)
        except (AssertionError, OSError):
            pass
    for process in processes:
        try:
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
        except (AssertionError, OSError):
            pass


def _start_processes(ctx, process_specs: list[tuple[tuple, dict]]) -> list:
    started = []
    try:
        for args, kwargs in process_specs:
            process = ctx.Process(*args, **kwargs)
            process.start()
            started.append(process)
    except BaseException:
        _stop_processes(started)
        raise
    return started


def _run_a1_child(args, manifest, records, advance, canonical_hash, raw_hash, threadpoolctl) -> dict:
    count = int(args.case_workers)
    cores = [int(item) for item in args.case_cpus.split(",") if item]
    if len(cores) < count:
        raise HydroError(f"case requires {count} physical CPUs; received {len(cores)}")
    groups = _case_tasks(f"A1-p{count:02d}", count, records)
    if not hasattr(os, "sched_setaffinity") or "fork" not in mp.get_all_start_methods():
        raise HydroError("Linux fork and CPU affinity are required for A1")
    ctx = mp.get_context("fork")
    process_specs = []
    start = time.monotonic()
    for index in range(count):
        process_specs.append(((), {
            "target": _worker_main, "args": (index, cores[index], groups[index]),
            "kwargs": {"input_root": args.input_root, "output_root": args.task_root,
                       "manifest": manifest, "raw_hash": raw_hash,
                       "canonical_hash": canonical_hash, "advance": advance,
                       "threadpoolctl": threadpoolctl}}))
    processes = _start_processes(ctx, process_specs)
    deadline = float(args.case_deadline)
    failure = None
    while any(process.is_alive() for process in processes):
        failed = [process for process in processes if process.exitcode not in (None, 0)]
        if failed:
            failure = f"worker process failed: {[(p.pid, p.exitcode) for p in failed]}"
            break
        if time.monotonic() >= deadline:
            failure = "CASE_DEADLINE_REACHED"
            break
        time.sleep(0.25)
    if failure:
        _stop_processes(processes)
    else:
        for process in processes:
            process.join()
    case_id = f"A1-p{count:02d}"
    completed = sum(1 for path in Path(args.task_root).glob(f"{case_id}-*.json")
                    if json.loads(path.read_text(encoding="utf-8")).get("status") == "DONE")
    started_files = list(Path(args.task_root).glob(f"{case_id}-worker-*-started.json"))
    first_started = min((json.loads(p.read_text(encoding="utf-8"))["started_monotonic"]
                         for p in started_files), default=start)
    last_started = max((json.loads(p.read_text(encoding="utf-8"))["started_monotonic"]
                        for p in started_files), default=start)
    wall = time.monotonic() - start
    block8, trailing = divmod(completed, 8)
    status = "COMPLETE" if failure is None and completed == count * TASKS_PER_WORKER else "PARTIAL"
    return {"case_id": f"A1-p{count:02d}", "status": status, "workers": count,
            "tasks_planned": count * TASKS_PER_WORKER, "completed_tasks": completed,
            "unique_screens_planned": len({item["ordinal"] for worker in groups for item in worker}),
            "reused_screen_tasks": count * TASKS_PER_WORKER - len({item["ordinal"] for worker in groups for item in worker}),
            "input_scaling_class": input_scaling_class(count),
            "startup_seconds": max(0.0, last_started - start),
            "steady_execution_seconds": max(0.0, time.monotonic() - last_started),
            "case_wall_seconds": wall,
            "screens_per_second": completed / wall if wall > 0 else None,
            "hydro_steps_per_second": completed * STEPS / wall if wall > 0 else None,
            "block_size_8": {"full_blocks": block8, "trailing_tasks": trailing,
                             "blocks_per_second": block8 / wall if wall > 0 else None},
            "stop_reason": failure}


def run_case(args) -> int:
    common.add_vendor(args.vendor_dir)
    advance, canonical_hash, raw_hash, threadpoolctl, backend = _load_science()
    details, records = load_manifest(Path(args.manifest), Path(args.input_root), raw_hash)
    manifest = details["manifest"]
    if not hasattr(threadpoolctl, "threadpool_limits"):
        raise HydroError("threadpoolctl.threadpool_limits unavailable")
    if args.case_mode == "A0":
        summary = _run_a0_child(args, manifest, records, advance, canonical_hash, raw_hash, threadpoolctl)
    else:
        summary = _run_a1_child(args, manifest, records, advance, canonical_hash, raw_hash, threadpoolctl)
    summary.update({"schema": SCHEMA, "classification": CLASSIFICATION,
                    "production_qualification": QUALIFICATION,
                    "backend": backend, "manifest_sha256": details["raw_sha256"]})
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
    return 0 if summary["status"] == "COMPLETE" else 1


CSV_FIELDS = ("case_id", "task_id", "status", "worker_index", "worker_cpu", "ordinal", "screen_id",
              "kernel_wall", "kernel_cpu", "service_total", "tree_peak_rss_bytes",
              "tree_cpu_seconds", "historical_delta_n_max_abs", "historical_vx_max_abs",
              "historical_vy_max_abs", "output_sha256", "note")


def _case_summary_from_stdout(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema") == SCHEMA:
            return value
    return None


def _supervise_case(command: list[str], root: Path, case_id: str, deadline: float,
                    memory_limit: int) -> dict:
    out_path, err_path = root / f"{case_id}.stdout.txt", root / f"{case_id}.stderr.txt"
    started = time.monotonic()
    peak_rss, max_cpu, outcome, monitored = 0, 0.0, None, False
    with out_path.open("x", encoding="utf-8", newline="\n") as out, \
         err_path.open("x", encoding="utf-8", newline="\n") as err:
        kwargs = {"stdout": out, "stderr": err}
        if os.name != "nt":
            kwargs["start_new_session"] = True
        process = subprocess.Popen(command, **kwargs)
        while process.poll() is None:
            snapshot = common.tree_snapshot(process.pid)
            if snapshot is not None:
                monitored = True
                peak_rss = max(peak_rss, snapshot["rss"])
                max_cpu = max(max_cpu, snapshot["cpu"])
                if peak_rss >= memory_limit * RSS_FRACTION:
                    outcome = "STOPPED_RSS_80_PERCENT"
                    common.kill_child(process)
                    break
            if time.monotonic() >= deadline:
                outcome = "LIMIT_REACHED"
                common.kill_child(process)
                break
            time.sleep(0.25)
        exit_code = process.wait()
    if exit_code != 0 and os.name != "nt":
        _cleanup_case_process_group(process.pid)
    stdout = out_path.read_text(encoding="utf-8", errors="replace")
    stderr = err_path.read_text(encoding="utf-8", errors="replace")
    return {"case_id": case_id, "exit_code": exit_code, "outcome": outcome,
            "wall_seconds": time.monotonic() - started, "peak_tree_rss_bytes": peak_rss,
            "tree_cpu_seconds": max_cpu, "tree_monitor_available": monitored,
            "stdout_summary": _case_summary_from_stdout(stdout),
            "stdout_path": out_path.name, "stderr_path": err_path.name,
            "stderr_tail": "\n".join(stderr.splitlines()[-30:])}


def _cleanup_case_process_group(pgid: int, grace_seconds: float = 2.0) -> None:
    """Stop only descendants that still occupy this case's dedicated process group."""
    if os.name == "nt":
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _environment(root: Path, vendor_dir: str | None, affinity: dict, manifest_details: dict) -> dict:
    import numpy as np
    import threadpoolctl

    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.__config__.show()
    lscpu = common.optional_command("lscpu", ["-e"])
    numa = common.optional_command("numactl", ["--hardware"])
    common.write_new(root / "lscpu_e.txt", lscpu["stdout"])
    common.write_new(root / "numactl_hardware.txt", numa["stdout"])
    status = Path("/proc/self/status")
    common.write_new(root / "proc_status.txt", status.read_text(errors="replace") if status.is_file() else "UNAVAILABLE\n")
    common.write_new(root / "numpy_config.txt", config.getvalue())
    pools = {"status": "AVAILABLE", "version": getattr(threadpoolctl, "__version__", None),
             "pools": threadpoolctl.threadpool_info()}
    common.write_json(root / "threadpool_info.json", pools)
    np_probe = {
        "complex64_input_dtype": "complex64",
        "complex64_fft_output_dtype": str(np.fft.fft(np.ones(8, dtype=np.complex64)).dtype),
        "complex128_input_dtype": "complex128",
        "complex128_fft_output_dtype": str(np.fft.fft(np.ones(8, dtype=np.complex128)).dtype),
    }
    safe_slurm = common.slurm_metadata(os.environ)
    safe_slurm["gpu_allocation"] = {key: os.environ[key] for key in (
        "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE") if os.environ.get(key)}
    common.write_json(root / "slurm_environment.json", safe_slurm)
    common.write_json(root / "affinity.json", affinity)
    environment = {
        "schema": SCHEMA, "classification": CLASSIFICATION,
        "production_qualification": QUALIFICATION, "python": sys.version,
        "python_executable": sys.executable, "platform": sys.platform,
        "numpy_version": np.__version__, "threadpoolctl_version": pools["version"],
        "threadpool_info_status": pools["status"], "numpy_fft_dtype_probe": np_probe,
        "lscpu_e_status": lscpu["status"], "numactl_status": numa["status"],
        "vendor_dir": vendor_dir, "manifest_raw_sha256": manifest_details["raw_sha256"],
        "input_manifest_sha256": manifest_details["manifest"].get("input_manifest_sha256"),
        "config_sha256": manifest_details["manifest"].get("config_sha256"),
        "source_manifest_sha256": manifest_details["manifest"].get("source_manifest_sha256"),
        "input_identity": manifest_details["manifest"].get("identity"),
        "hydro_parameters": manifest_details["manifest"].get("hydro"),
        "comparison": manifest_details["manifest"].get("comparison"),
    }
    common.write_json(root / "environment.json", environment)
    return environment


def _publish_task_receipts(root: Path, seen: set[str], accounting: dict) -> tuple[int, int]:
    import numpy as np

    jsonl = root / "benchmark_results.jsonl"
    csv_path = root / "performance_summary.csv"
    new_count, done_count = 0, 0
    with csv_path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        for path in sorted((root / "tasks").glob("A*.json")):
            if path.name in seen or "-started" in path.name:
                continue
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            seen.add(path.name)
            sync_jsonl_append(jsonl, receipt)
            perf = receipt.get("timing_seconds", {})
            metrics = receipt.get("historical_next_field_comparison", {})
            output = receipt.get("output", {})
            writer.writerow({
                "case_id": receipt.get("case_id"), "task_id": receipt.get("task_id"),
                "status": receipt.get("status"), "worker_index": receipt.get("worker_index"),
                "worker_cpu": receipt.get("worker_cpu"), "ordinal": receipt.get("ordinal"),
                "screen_id": receipt.get("screen_id"), "kernel_wall": perf.get("kernel_wall"),
                "kernel_cpu": perf.get("kernel_cpu"), "service_total": perf.get("service_total"),
                "tree_peak_rss_bytes": accounting.get("peak_tree_rss_bytes"),
                "tree_cpu_seconds": accounting.get("tree_cpu_seconds"),
                "historical_delta_n_max_abs": metrics.get("delta_n", {}).get("max_abs"),
                "historical_vx_max_abs": metrics.get("vx", {}).get("max_abs"),
                "historical_vy_max_abs": metrics.get("vy", {}).get("max_abs"),
                "output_sha256": output.get("raw_sha256"), "note": receipt.get("error", ""),
            })
            stream.flush()
            os.fsync(stream.fileno())
            new_count += 1
            done_count += receipt.get("status") == "DONE"
    return new_count, done_count


def _make_child_command(args, root: Path, case_id: str, case_mode: str,
                        workers: int, cpus: list[int], deadline: float) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()),
            "--case-mode", case_mode, "--case-workers", str(workers),
            "--case-cpus", ",".join(map(str, cpus)),
            "--case-deadline", str(deadline), "--input-root", args.input_root,
            "--manifest", args.manifest, "--task-root", str(root / "tasks"),
            "--vendor-dir", args.vendor_dir or ""]


def run_parent(args) -> int:
    vendor = common.add_vendor(args.vendor_dir)
    if args.smoke:
        return run_smoke(args)
    common.validate_full_allocation(os.environ)
    if not Path("/proc").is_dir():
        raise HydroError("Linux /proc required for RSS/CPU time monitoring")
    if not 0 < args.deadline_seconds <= MAX_SECONDS or not 0 < args.a0_deadline_seconds <= A0_MAX_SECONDS:
        raise HydroError("overall/A0 deadlines exceed the authorized budget")
    if not 0 < args.memory_limit_bytes <= MEMORY_BYTES:
        raise HydroError("memory limit exceeds the authorized node budget")
    advance, canonical_hash, raw_hash, threadpoolctl, backend = _load_science()
    if backend.get("backend") != "numpy":
        raise HydroError("NumPy backend is mandatory")
    if getattr(threadpoolctl, "__version__", None) != "3.6.0":
        raise HydroError("frozen threadpoolctl 3.6.0 vendor is required")
    input_root = Path(args.input_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest_details, records = load_manifest(manifest_path, input_root, raw_hash)
    topo = common.cpu_topology()
    if not common.physical_topology_verified(topo) or len(topo["physical_core_representatives"]) < 96:
        output = common.create_output_root(Path(args.output_root))
        common.write_json(output / "status.json", {
            "schema": SCHEMA, "classification": CLASSIFICATION,
            "production_qualification": QUALIFICATION,
            "status": "BLOCKED_PHYSICAL_TOPOLOGY_UNAVAILABLE",
            "physical_core_count": topo.get("physical_core_count"),
            "topology_source": topo.get("topology_source"),
            "topology_error": topo.get("topology_error"),
        })
        print(json.dumps({"status": "BLOCKED_PHYSICAL_TOPOLOGY_UNAVAILABLE",
                          "output_root": str(output)}, sort_keys=True), flush=True)
        return 1
    if len(topo["allowed_logical_cpus"]) != 96:
        raise HydroError(f"expected 96 allowed allocation CPUs, found {len(topo['allowed_logical_cpus'])}")
    core_pool = common.select_physical_cpus(96, topology=topo)
    output = common.create_output_root(Path(args.output_root))
    task_root = output / "tasks"
    task_root.mkdir(mode=0o750)
    process_root = output / "process_logs"
    process_root.mkdir(mode=0o750)
    csv_path = output / "performance_summary.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=CSV_FIELDS).writeheader()
        stream.flush()
        os.fsync(stream.fileno())
    (output / "benchmark_results.jsonl").open("x", encoding="utf-8").close()
    _environment(output, args.vendor_dir, topo, manifest_details)
    common.write_json(output / "input_acceptance.json", {
        "schema": SCHEMA, "classification": CLASSIFICATION,
        "production_qualification": QUALIFICATION,
        "status": "ACCEPTED_FOR_DIAGNOSTIC_EXECUTION",
        "manifest_raw_sha256": manifest_details["raw_sha256"],
        "manifest_declared_sha256": manifest_details["manifest"].get("input_manifest_sha256"),
        "input_identity": manifest_details["manifest"]["identity"],
        "actual_records_loaded": 0, "note": "Each task revalidates raw and canonical hashes at read time.",
    })
    started, deadline = time.monotonic(), time.monotonic() + args.deadline_seconds
    seen, cases, status, total_done, total_cpu, total_peak = set(), [], "COMPLETE", 0, 0.0, 0
    sequence = [("A0", 1)] + [(f"A1-p{n:02d}", n) for n in WORKERS]
    for case_id, count in sequence:
        if time.monotonic() >= deadline:
            status = "LIMIT_REACHED"
            break
        case_mode = "A0" if case_id == "A0" else "A1"
        case_deadline = min(deadline, time.monotonic() + args.a0_deadline_seconds) if case_mode == "A0" else deadline
        cpus = [core_pool[0]] if case_mode == "A0" else core_pool[:count]
        command = _make_child_command(args, output, case_id, case_mode, count, cpus, case_deadline)
        accounting = _supervise_case(command, process_root, case_id, case_deadline, args.memory_limit_bytes)
        new_count, done_count = _publish_task_receipts(output, seen, accounting)
        total_done += done_count
        total_cpu += accounting["tree_cpu_seconds"]
        total_peak = max(total_peak, accounting["peak_tree_rss_bytes"])
        child = accounting.get("stdout_summary") or {}
        case_status = child.get("status")
        if accounting["outcome"]:
            case_status = accounting["outcome"]
        if not case_status:
            case_status = "COMPLETE" if accounting["exit_code"] == 0 else "FAIL"
        case_record = {**accounting, "status": case_status,
                       "published_task_receipts": new_count,
                       "completed_tasks": child.get("completed_tasks", done_count),
                       "allocated_workers": count, "case_mode": case_mode}
        common.write_json(output / f"{case_id}.json", case_record)
        cases.append(case_record)
        print(json.dumps({"case_id": case_id, "status": case_status,
                          "completed_tasks": case_record["completed_tasks"],
                          "peak_tree_rss_bytes": accounting["peak_tree_rss_bytes"]}, sort_keys=True), flush=True)
        if case_status != "COMPLETE":
            status = case_status if case_status in ("LIMIT_REACHED", "STOPPED_RSS_80_PERCENT",
                                                    "CPU_REPEAT_DRIFT") \
                     else ("PARTIAL" if total_done else case_status)
            break
        if case_mode == "A0" and done_count != 2:
            status = "A0_FAILED_STOP_A1"
            break
    wall = time.monotonic() - started
    summary = {
        "schema": SCHEMA, "classification": CLASSIFICATION,
        "production_qualification": QUALIFICATION, "status": status,
        "input_kind": "HISTORICAL_REAL_POST_AND_NEXT_48_SCREEN_SET",
        "scientific_qualification": False, "case_count": len(cases),
        "completed_tasks": total_done, "wall_seconds": wall,
        "tree_cpu_seconds": total_cpu, "peak_tree_rss_bytes": total_peak,
        "allocated_core_hours": wall * 96 / 3600,
        "historical_service_comparison": "NOT_DIRECTLY_COMPARABLE",
        "A2_384_screen_strong_scaling": "NOT_RUN_INPUT_SET_HAS_48_SCREENS",
        "cases": cases,
    }
    common.write_json(output / "status.json", summary)
    print(json.dumps({"status": status, "output_root": str(output),
                      "completed_tasks": total_done, "wall_seconds": wall}, sort_keys=True), flush=True)
    return 0 if status == "COMPLETE" else 1


def run_smoke(args) -> int:
    import numpy as np

    shape = tuple(args.smoke_shape)
    if len(shape) != 2 or any(int(value) < 1 or int(value) > 64 for value in shape):
        raise HydroError("smoke shape must be two dimensions in 1..64")
    output = common.create_output_root(Path(args.output_root))
    source = {name: np.arange(math.prod(shape), dtype=np.float64).reshape(shape)
              for name in FIELDS}
    fake_callable = lambda values: {name: value.copy() for name, value in values.items()}
    result = fake_callable(source)
    with (output / "synthetic_smoke.npz").open("xb") as stream:
        np.savez(stream, **result)
        stream.flush()
        os.fsync(stream.fileno())
    summary = {"schema": SCHEMA, "status": "COMPLETE", "classification": CLASSIFICATION,
               "production_qualification": QUALIFICATION,
               "input_kind": "SYNTHETIC_ENGINEERING_SMOKE_ONLY",
               "scientific_inputs_used": False, "fake_callable_used": True,
               "shape": shape, "fields_written": list(FIELDS)}
    common.write_json(output / "status.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root")
    parser.add_argument("--manifest")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--vendor-dir")
    parser.add_argument("--deadline-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--a0-deadline-seconds", type=float, default=A0_MAX_SECONDS)
    parser.add_argument("--memory-limit-bytes", type=int, default=MEMORY_BYTES)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-shape", nargs=2, type=int, default=(8, 6))
    parser.add_argument("--case-mode", choices=("A0", "A1"), help=argparse.SUPPRESS)
    parser.add_argument("--case-workers", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--case-cpus", default="", help=argparse.SUPPRESS)
    parser.add_argument("--case-deadline", type=float, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--task-root", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.smoke:
            return run_smoke(args)
        if args.case_mode:
            if not all((args.input_root, args.manifest, args.task_root)):
                raise HydroError("case worker input paths are required")
            return run_case(args)
        if not args.input_root:
            raise HydroError("--input-root is required outside synthetic smoke")
        if not args.manifest:
            args.manifest = str(Path(args.input_root) / "candidate_input_manifest.json")
        return run_parent(args)
    except BaseException as exc:
        print(json.dumps({"schema": SCHEMA, "status": "FAIL",
                          "classification": CLASSIFICATION,
                          "production_qualification": QUALIFICATION,
                          "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
