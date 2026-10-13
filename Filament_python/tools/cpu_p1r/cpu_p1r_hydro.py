#!/usr/bin/env python3
"""P1R CPU-only HR-4 retest. Every result is diagnostic, never production-qualified."""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback

_INITIAL_GPU_REQUEST = os.environ.get("UPPE_USE_GPU")
os.environ["UPPE_USE_GPU"] = "0"

from tools.cpu_p1 import cpu_p1_fft as common
from tools.cpu_p1 import cpu_p1_hydro as p1
from tools.hpc_ops.provenance_v2 import raw_sha256 as provenance_raw_sha256

SCHEMA = "filament.cpu_p1r_hydro_retest.v1"
WORKLOAD_SCHEMA = "filament.cpu_p1r_hydro_workload.v1"
CLASSIFICATION = "P1R_DIAGNOSTIC_ONLY"
QUALIFICATION = "NOT_PRODUCTION_QUALIFIED"
WORKER_COUNT = 96
NODE_COUNT = 2
SOURCE_COUNT = 48
REPETITIONS = 16
TASK_COUNT = SOURCE_COUNT * REPETITIONS
H2_TASK_COUNT = 96
MAX_ALLOCATION_SECONDS = 5400
FINAL_RESERVE_SECONDS = 300
STAGE_RESERVE_SECONDS = 300
MEMORY_LIMIT_BYTES = 320 * 1024**3
RSS_FRACTION = 0.80
HISTORICAL_STAGE_ESTIMATE_SECONDS = 394.0
STAGE_ESTIMATE_SAFETY_FACTOR = 1.25
FIELDS = p1.FIELDS
FIXED_PYTHON = "/publicfs01/fs1-share/soft/amd-soft/miniforge/24.11/bin/python"
STAGES = (("1N-A", 1, "H1"), ("2N-A", 2, "H1"),
          ("2N-B", 2, "H1"), ("1N-B", 1, "H1"))
H2_STAGES = (("H2-A1-NUMA0", "NUMA0"), ("H2-B1-STRIPED", "STRIPED"),
             ("H2-B2-STRIPED", "STRIPED"), ("H2-A2-NUMA0", "NUMA0"))
CSV_FIELDS = ("stage", "task_id", "status", "rank", "worker_index", "worker_cpu",
              "ordinal", "repetition", "kernel_wall", "kernel_cpu", "service_wall",
              "historical_next_delta_n_max_abs", "historical_next_vx_max_abs",
              "historical_next_vy_max_abs", "output_sha256")


class P1RError(RuntimeError):
    pass


def build_workload(manifest_details: dict, records: dict) -> dict:
    """Build the frozen 16-repetition-major x 48-source workload; never run-time substitute it."""
    manifest = manifest_details["manifest"]
    frozen_parameters = {"hydro": manifest["hydro"], "dx_m": manifest["dx_m"],
                         "dy_m": manifest["dy_m"], "config_sha256": manifest["config_sha256"],
                         "source_manifest_sha256": manifest["source_manifest_sha256"]}
    parameter_json = json.dumps(frozen_parameters, sort_keys=True,
                                separators=(",", ":"), allow_nan=False).encode("utf-8")
    parameter_sha = provenance_raw_sha256(parameter_json)
    tasks = []
    for index in range(TASK_COUNT):
        repetition, ordinal = divmod(index, SOURCE_COUNT)
        source = records["post"][ordinal]
        tasks.append({"index": index, "task_id": f"task-{index:04d}",
                      "repetition": repetition, "ordinal": ordinal,
                      "screen_id": source["screen_id"], "z_m": source["z_m"],
                      "post_raw_sha256": source["sha256"],
                      "post_field_sha256": source["field_sha256"],
                      "hydro_parameter_sha256": parameter_sha,
                      "assignment_1n": {"rank": 0, "worker_index": index % WORKER_COUNT},
                      "assignment_2n": {"rank": (index % (2 * WORKER_COUNT)) // WORKER_COUNT,
                                        "worker_index": index % WORKER_COUNT}})
    return {"schema": WORKLOAD_SCHEMA, "classification": CLASSIFICATION,
            "hash_scope": "classified_by_record",
            "hash_encoding": "raw bytes for input files; canonical array hash for fields; UTF-8 sorted compact JSON bytes for frozen hydro parameters",
            "input_manifest_raw_sha256": manifest_details["raw_sha256"],
            "ordering": "repetition_major_source_minor", "source_count": SOURCE_COUNT,
            "repetitions": REPETITIONS, "task_count": TASK_COUNT,
            "hydro_parameter_sha256": parameter_sha,
            "frozen_parameters": frozen_parameters,
            "assignment_summary": {"1N": {"nodes": 1, "workers_per_node": 96, "tasks_per_worker": 8},
                                   "2N": {"nodes": 2, "workers_per_node": 96, "tasks_per_worker": 4}},
            "tasks": tasks}


def validate_workload(path: Path, manifest_details: dict, records: dict, raw_hash) -> tuple[dict, str]:
    if not path.is_file():
        raise P1RError(f"frozen workload unavailable: {path}")
    workload_sha = raw_hash(path)
    try:
        workload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise P1RError(f"frozen workload unreadable: {type(exc).__name__}: {exc}") from exc
    expected = build_workload(manifest_details, records)
    if workload != expected:
        raise P1RError("frozen workload differs from the approved 48x16 repetition-major assignment")
    return workload, workload_sha


def worker_tasks(workload: dict, node_count: int, rank: int) -> list[list[dict]]:
    """Return local worker queues with globally stable task IDs and exactly-once assignment."""
    if node_count not in (1, 2) or rank < 0 or rank >= node_count:
        raise ValueError("node_count/rank outside the approved 1N or 2N layout")
    workers_here = WORKER_COUNT
    queues: list[list[dict]] = [[] for _ in range(workers_here)]
    limit = TASK_COUNT if node_count == 1 else TASK_COUNT
    for task in workload["tasks"][:limit]:
        global_worker = int(task["index"]) % (WORKER_COUNT * node_count)
        task_rank = 0 if node_count == 1 else global_worker // WORKER_COUNT
        expected_assignment = task["assignment_1n"] if node_count == 1 else task["assignment_2n"]
        if int(expected_assignment["rank"]) != task_rank or int(expected_assignment["worker_index"]) != global_worker % WORKER_COUNT:
            raise P1RError(f"frozen assignment mismatch for {task['task_id']}")
        if task_rank == rank:
            local_worker = global_worker if node_count == 1 else global_worker % WORKER_COUNT
            queues[local_worker].append({**task, "global_worker": global_worker,
                                         "worker_index": local_worker, "rank": rank})
    return queues


def h2_tasks(workload: dict) -> list[list[dict]]:
    queues: list[list[dict]] = [[] for _ in range(24)]
    for task in workload["tasks"][:H2_TASK_COUNT]:
        worker = int(task["index"]) % 24
        queues[worker].append({**task, "global_worker": worker,
                               "worker_index": worker, "rank": 0})
    return queues


def validate_allocation_2n(env: dict) -> dict:
    required = {"SLURM_JOB_NUM_NODES": "2", "SLURM_NTASKS": "2",
                "SLURM_NTASKS_PER_NODE": "1", "SLURM_CPUS_PER_TASK": "96",
                "SLURM_JOB_PARTITION": "v6_384", "SLURM_JOB_ACCOUNT": "bscc-t",
                "SLURM_JOB_QOS": "normal", "SLURM_MEM_PER_NODE": "327680",
                "UPPE_USE_GPU": "0"}
    mismatches = {key: {"expected": value, "actual": env.get(key)}
                  for key, value in required.items() if env.get(key) != value}
    gpu_keys = ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE")
    allocated_gpu = {key: env.get(key) for key in gpu_keys if env.get(key)}
    tres_gpu = {key: env.get(key) for key in ("SLURM_TRES_PER_NODE", "SLURM_TRES_PER_JOB")
                if env.get(key) and re.search(r"(?:gres/)?gpu", env[key], re.IGNORECASE)}
    visible = env.get("CUDA_VISIBLE_DEVICES", "")
    if (_INITIAL_GPU_REQUEST not in (None, "0", "false", "False", "") or allocated_gpu or tres_gpu or
            (visible and visible not in ("NoDevFiles", "-1"))):
        mismatches["gpu_allocation"] = {**allocated_gpu, **tres_gpu,
                                         "UPPE_USE_GPU_at_process_entry": _INITIAL_GPU_REQUEST,
                                         **({"CUDA_VISIBLE_DEVICES": visible} if visible else {})}
    if mismatches:
        raise P1RError("P1R 2N/192-core CPU allocation rejected: " +
                       json.dumps(mismatches, sort_keys=True))
    return {key: env.get(key) for key in (*required, *gpu_keys, "SLURM_TRES_PER_NODE",
                                          "SLURM_TRES_PER_JOB", "CUDA_VISIBLE_DEVICES") if env.get(key)}


def validate_numpy_call(advance, fields: dict, manifest: dict) -> dict:
    """Thin named seam for verifying the unchanged HR-4 invocation contract."""
    return p1.invoke_hydro(advance, fields, manifest)


def _parse_cpu_list(text: str) -> set[int]:
    result: set[int] = set()
    for item in text.strip().split(","):
        if not item:
            continue
        if "-" in item:
            start, stop = map(int, item.split("-", 1))
            result.update(range(start, stop + 1))
        else:
            result.add(int(item))
    return result


def numa_physical_cpus(topology: dict, sysfs: Path = Path("/sys/devices/system")) -> dict[int, list[int]]:
    allowed_physical = set(topology["physical_core_representatives"])
    nodes: dict[int, list[int]] = {}
    for path in sorted((sysfs / "node").glob("node[0-9]*")):
        node_id = int(path.name[4:])
        cpu_list = path / "cpulist"
        if cpu_list.is_file():
            cpus = sorted(_parse_cpu_list(cpu_list.read_text()) & allowed_physical)
            if cpus:
                nodes[node_id] = cpus
    return nodes


def select_h2_cpus(layout: str, topology: dict, numa_nodes: dict[int, list[int]]) -> list[int]:
    if layout == "NUMA0":
        if 0 not in numa_nodes or len(numa_nodes[0]) < 24:
            raise P1RError("H2 NUMA0 placement unavailable: fewer than 24 verified physical cores")
        return numa_nodes[0][:24]
    if layout == "STRIPED":
        node_ids = sorted(numa_nodes)
        if len(node_ids) < 4 or any(len(numa_nodes[node]) < 6 for node in node_ids[:4]):
            raise P1RError("H2 striped placement unavailable: four NUMA nodes with six cores each required")
        return [cpu for node in node_ids[:4] for cpu in numa_nodes[node][:6]]
    raise ValueError(f"unknown H2 layout: {layout}")


def _probe_record() -> dict:
    if not hasattr(os, "sched_getaffinity"):
        raise P1RError("Linux affinity API is required")
    allowed = sorted(os.sched_getaffinity(0))
    topology = common.cpu_topology(allowed)
    if (not common.physical_topology_verified(topology) or len(allowed) != 96 or
            topology.get("physical_core_count") != 96):
        raise P1RError("srun probe did not receive 96 verified physical cores: " +
                       json.dumps(topology, sort_keys=True))
    return {"record_type": "P1R_NODE_PROBE", "rank": int(os.environ.get("SLURM_PROCID", "-1")),
            "hostname": socket.gethostname(), "slurm_node_name": os.environ.get("SLURMD_NODENAME"),
            "affinity": topology,
            "slurm_step_id": os.environ.get("SLURM_STEP_ID")}


def write_probe_record(probe_dir: Path) -> dict:
    record = _probe_record()
    p1.sync_json_new(probe_dir / f"probe-rank-{record['rank']}.json", record)
    print(json.dumps({"record_type": "P1R_PROBE_WRITTEN", "rank": record["rank"]},
                     separators=(",", ":")), flush=True)
    return record


def _load_runtime(args):
    _validate_interpreter()
    common.add_vendor(args.vendor_dir)
    advance, canonical_hash, raw_hash, threadpoolctl, backend = p1._load_science()
    if backend.get("backend") != "numpy":
        raise P1RError(f"CPU NumPy backend required, got {backend}")
    if getattr(threadpoolctl, "__version__", None) != "3.6.0":
        raise P1RError("frozen threadpoolctl 3.6.0 is required")
    details, records = p1.load_manifest(Path(args.manifest), Path(args.input_root), raw_hash)
    workload, workload_sha = validate_workload(Path(args.workload), details, records, raw_hash)
    return advance, canonical_hash, raw_hash, threadpoolctl, backend, details, records, workload, workload_sha


def _validate_interpreter() -> None:
    if os.path.abspath(sys.executable) != FIXED_PYTHON:
        raise P1RError(f"fixed HPC interpreter required: expected {FIXED_PYTHON}, got {sys.executable}")


def _task_receipt(task: dict, *, stage: str, node_root: Path, input_root: Path,
                  manifest: dict, records: dict, advance, canonical_hash, raw_hash,
                  worker_cpu: int, rank: int) -> dict:
    import numpy as np

    started = time.perf_counter()
    post_rec = records["post"][task["ordinal"]]
    next_rec = records["next"][task["ordinal"]]
    post, post_timing = p1.load_screen(post_rec, input_root, raw_hash, canonical_hash)
    next_fields, next_timing = p1.load_screen(next_rec, input_root, raw_hash, canonical_hash)
    if (post_timing["raw_sha256"] != task["post_raw_sha256"] or
            post_timing["canonical_field_sha256"] != task["post_field_sha256"]):
        raise P1RError(f"task input identity differs from frozen workload: {task['task_id']}")
    kernel_wall_start, kernel_cpu_start = time.perf_counter(), time.process_time()
    result = validate_numpy_call(advance, post, manifest)
    kernel_wall = time.perf_counter() - kernel_wall_start
    kernel_cpu = time.process_time() - kernel_cpu_start
    output_fields = {name: np.asarray(result[name]) for name in FIELDS}
    for name, array in output_fields.items():
        if array.shape != p1.EXPECTED_SHAPE or array.dtype != np.dtype("float64"):
            raise P1RError(f"HR-4 {name} output shape/dtype rejected")
        if not array.flags.c_contiguous or not np.isfinite(array).all():
            raise P1RError(f"HR-4 {name} output is not finite C-contiguous float64")
    historical = p1.compare_fields(output_fields, next_fields)
    out_path = node_root / "tasks" / f"{task['task_id']}.npz"
    write_s, raw_hash_s, canonical_hash_s, output_raw, output_fields_hash = p1._write_result_arrays(out_path, output_fields)
    observables = p1._jsonable(result.get("observables", {}))
    ref_observables = p1._jsonable(p1._load_reference_observables(next_fields, manifest))
    receipt = {
        "schema": SCHEMA, "status": "DONE", "classification": CLASSIFICATION,
        "production_qualification": QUALIFICATION,
        "comparison_scope": "HISTORICAL_NEXT_ARRAY_DIAGNOSTIC_ONLY",
        "stage": stage, "task_id": task["task_id"], "task_index": task["index"],
        "repetition": task["repetition"], "ordinal": task["ordinal"],
        "hydro_parameter_sha256": task["hydro_parameter_sha256"],
        "screen_id": task["screen_id"], "z_m": task["z_m"], "rank": rank,
        "worker_index": task["worker_index"], "worker_cpu": worker_cpu,
        "input_post": {"relative_path": post_rec["relative_path"],
                        "raw_sha256": post_timing["raw_sha256"],
                        "canonical_field_sha256": post_timing["canonical_field_sha256"]},
        "input_historical_next": {"relative_path": next_rec["relative_path"],
                                  "raw_sha256": next_timing["raw_sha256"],
                                  "canonical_field_sha256": next_timing["canonical_field_sha256"]},
        "timing_seconds": {
            "post_load": post_timing["load_seconds"], "next_load": next_timing["load_seconds"],
            "input_raw_hash": post_timing["raw_hash_seconds"] + next_timing["raw_hash_seconds"],
            "input_canonical_hash": post_timing["canonical_hash_seconds"] + next_timing["canonical_hash_seconds"],
            "kernel_wall": kernel_wall, "kernel_cpu": kernel_cpu,
            "kernel_reported_wall": result["performance"]["wall_time_s_total"],
            "output_write": write_s, "output_raw_hash": raw_hash_s,
            "output_canonical_hash": canonical_hash_s,
            "service_wall": time.perf_counter() - started},
        "steps": result["steps"], "stability": p1._jsonable(result.get("stability")),
        "performance": p1._jsonable(result["performance"]), "observables": observables,
        "historical_next_field_comparison": historical,
        "historical_next_observable_comparison": p1._numeric_observable_delta(observables, ref_observables),
        "cpu_backend": result["performance"].get("backend"),
        "historical_next_backend_identity": "NOT_ASSERTED_BY_ACCEPTED_INPUT_MANIFEST",
        "output": {"relative_path": f"stages/{stage}/node-{rank:02d}/tasks/{out_path.name}",
                   "raw_sha256": output_raw,
                   "canonical_field_sha256": output_fields_hash},
    }
    p1.sync_json_new(node_root / "tasks" / f"{task['task_id']}.json", receipt)
    return receipt


def _worker_main(worker_index: int, cpu: int, tasks: list[dict], *, stage: str,
                 node_root: str, input_root: str, manifest: dict, records: dict,
                 advance, canonical_hash, raw_hash, threadpoolctl, rank: int) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if not hasattr(os, "sched_setaffinity"):
        raise P1RError("Linux affinity API is required")
    os.sched_setaffinity(0, {cpu})
    if set(os.sched_getaffinity(0)) != {cpu}:
        raise P1RError("worker did not bind to exactly one verified physical core")
    root = Path(node_root)
    p1.sync_json_new(root / f"worker-{worker_index:03d}-started.json", {
        "stage": stage, "rank": rank, "worker_index": worker_index, "cpu": cpu,
        "affinity": sorted(os.sched_getaffinity(0)), "started_monotonic": time.monotonic()})
    with threadpoolctl.threadpool_limits(limits=1):
        pool_info = threadpoolctl.threadpool_info()
        oversized = [item for item in pool_info
                     if int(item.get("num_threads", 1)) > 1]
        if oversized:
            raise P1RError(f"worker thread pool exceeds one thread: {oversized}")
        p1.sync_json_new(root / f"worker-{worker_index:03d}-threadpools.json", {
            "threadpool_info": pool_info, "thread_limit": 1,
            "status": "SINGLE_THREADED"})
        for task in tasks:
            try:
                _task_receipt(task, stage=stage, node_root=root, input_root=Path(input_root),
                              manifest=manifest, records=records, advance=advance,
                              canonical_hash=canonical_hash, raw_hash=raw_hash,
                              worker_cpu=cpu, rank=rank)
            except BaseException as exc:
                failure = {"schema": SCHEMA, "status": "FAIL", "classification": CLASSIFICATION,
                           "production_qualification": QUALIFICATION,
                           "stage": stage, "task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()}
                p1.sync_json_new(root / "tasks" / f"{task['task_id']}.json", failure)
                raise


def _node_stage(args) -> int:
    rank = int(os.environ.get("SLURM_PROCID", "-1"))
    node_count = int(args.node_count)
    if node_count == 1 and rank != 0:
        raise P1RError("1N stage expected Slurm rank 0")
    if node_count == 2 and rank not in (0, 1):
        raise P1RError("2N stage expected Slurm rank 0 or 1")
    expected_probe = json.loads(Path(args.probe_json).read_text(encoding="utf-8"))
    expected_host = expected_probe["rank_to_hostname"][str(rank)]
    if socket.gethostname() != expected_host:
        raise P1RError("srun stage host differs from the verified probe host")
    allowed = sorted(os.sched_getaffinity(0))
    topology = common.cpu_topology(allowed)
    if (len(allowed) != 96 or topology.get("physical_core_count") != 96 or
            not common.physical_topology_verified(topology)):
        raise P1RError("stage rank lacks 96 verified physical cores")
    numa_nodes = numa_physical_cpus(topology)
    if args.layout == "H1":
        cpus = common.select_physical_cpus(96, topology=topology)
        queues = worker_tasks(args._workload_value, node_count, rank)
    else:
        cpus = select_h2_cpus(args.layout, topology, numa_nodes)
        queues = h2_tasks(args._workload_value)
    advance, canonical_hash, raw_hash, threadpoolctl, backend, details, records, _, workload_sha = _load_runtime(args)
    _validate_interpreter()
    if backend.get("backend") != "numpy":
        raise P1RError("CPU NumPy backend is not active on node rank")
    node_root = Path(args.output_root) / f"node-{rank:02d}"
    tasks_root = node_root / "tasks"
    if not tasks_root.is_dir():
        raise P1RError(f"stage task output directory missing: {tasks_root}")
    if len(cpus) != len(queues):
        raise P1RError("physical CPU and worker queue counts differ")
    # Replace temporary task shell objects with independently verified screen records.
    for worker_queue in queues:
        for task in worker_queue:
            task["_records"] = records
    started = time.monotonic()
    deadline = started + args.stage_budget
    failure = None
    stop_signal = {"received": None}
    old_handlers = {}
    def request_stop(signum, _frame):
        stop_signal["received"] = signum
    for signum in (signal.SIGTERM, signal.SIGINT):
        old_handlers[signum] = signal.signal(signum, request_stop)
    ctx = mp.get_context("fork")
    specs = []
    for worker_index, queue in enumerate(queues):
        if queue:
            specs.append(((), {"target": _worker_main,
                               "args": (worker_index, cpus[worker_index], queue),
                               "kwargs": {"stage": args.stage, "node_root": str(node_root),
                                          "input_root": args.input_root, "manifest": details["manifest"],
                                          "records": records, "advance": advance,
                                          "canonical_hash": canonical_hash, "raw_hash": raw_hash,
                                          "threadpoolctl": threadpoolctl, "rank": rank}}))
    processes = []
    peak_rss = 0
    peak_cpu = 0.0
    monitor_available = False
    try:
        processes = p1._start_processes(ctx, specs)
        while any(proc.is_alive() for proc in processes):
            failed = [proc for proc in processes if proc.exitcode not in (None, 0)]
            snapshot = common.tree_snapshot(os.getpid())
            if snapshot is not None:
                monitor_available = True
                peak_rss = max(peak_rss, int(snapshot["rss"]))
                peak_cpu = max(peak_cpu, float(snapshot["cpu"]))
            if failed:
                failure = f"WORKER_FAILED:{[(proc.pid, proc.exitcode) for proc in failed]}"
                break
            if peak_rss >= args.memory_limit_bytes * RSS_FRACTION:
                failure = "STOPPED_NODE_RSS_80_PERCENT"
                break
            if stop_signal["received"] is not None:
                failure = f"STOPPED_BY_SIGNAL_{stop_signal['received']}"
                break
            if time.monotonic() >= deadline:
                failure = "LIMIT_REACHED"
                break
            time.sleep(0.25)
        if failure:
            p1._stop_processes(processes)
        else:
            for proc in processes:
                proc.join()
        if not monitor_available:
            failure = failure or "NODE_PROCESS_TREE_MONITOR_UNAVAILABLE"
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        p1._stop_processes(processes)
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    task_count = sum(len(queue) for queue in queues)
    done = sum(1 for path in tasks_root.glob("task-*.json")
               if json.loads(path.read_text(encoding="utf-8")).get("status") == "DONE")
    node_summary = {"record_type": "P1R_NODE_STAGE", "schema": SCHEMA,
                    "classification": CLASSIFICATION, "stage": args.stage,
                    "rank": rank, "hostname": socket.gethostname(),
                    "status": "COMPLETE" if failure is None and done == task_count else "PARTIAL",
                    "stop_reason": failure, "tasks_planned": task_count, "tasks_completed": done,
                    "workers_started": len(processes), "physical_workers_requested": len(cpus),
                    "affinity": topology, "numa_node_cpu_counts": {str(k): len(v) for k, v in numa_nodes.items()},
                    "layout": args.layout, "workload_sha256": workload_sha,
                    "node_wall_seconds": time.monotonic() - started,
                    "peak_local_process_tree_rss_bytes": peak_rss,
                    "local_process_tree_cpu_seconds_sampled_peak_not_cumulative": peak_cpu,
                    "local_tree_monitor_available": monitor_available,
                    "local_worker_children_reaped": all(not proc.is_alive() for proc in processes),
                    "remote_step_cleanup_proven": False}
    p1.sync_json_new(node_root / "node_summary.json", node_summary)
    print(json.dumps({"record_type": "P1R_NODE_STAGE", "stage": args.stage,
                      "rank": rank, "status": node_summary["status"],
                      "tasks_completed": done}, separators=(",", ":")), flush=True)
    return 0 if node_summary["status"] == "COMPLETE" else 1


def _terminate_srun(process, pid: int, known_step_id: str | None = None) -> dict:
    """Bounded TERM, then cancel only this named step; never kill the whole job."""
    step_id = known_step_id or _find_named_step(process.args)
    targeted_cancel = False
    if process.poll() is None:
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            process.terminate()
    try:
        process.wait(timeout=75)
    except subprocess.TimeoutExpired:
        if step_id:
            try:
                cancel = subprocess.run(["scancel", "--signal=KILL", step_id], check=False,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                targeted_cancel = cancel.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                targeted_cancel = False
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except OSError:
                    process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
    exited = process.poll() is not None
    p1._cleanup_case_process_group(pid)
    remote_verified = exited and _step_absent(step_id) if step_id else exited and _named_step_absent(process.args)
    return {"srun_exited_after_stop": exited, "step_id": step_id or "UNAVAILABLE",
            "targeted_step_cancel_issued": targeted_cancel,
            "remote_step_teardown_verified": remote_verified}


def _find_named_step(command: list[str]) -> str | None:
    """Resolve only this controller's named Slurm step; never cancel a whole job/account."""
    if "--job-name" not in command:
        return None
    try:
        name = command[command.index("--job-name") + 1]
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            return None
        result = subprocess.run(["squeue", "--steps", "--noheader", "--jobs", job_id,
                                 "--format=%i %j"], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", check=False, timeout=5)
        for line in result.stdout.splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[1] == name and parts[0].startswith(job_id + "."):
                return parts[0]
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return None
    return None


def _step_absent(step_id: str) -> bool:
    try:
        job_id = step_id.split(".", 1)[0]
        result = subprocess.run(["squeue", "--steps", "--noheader", "--jobs", job_id,
                                 "--format=%i"], capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", check=False, timeout=5)
        return result.returncode == 0 and step_id not in result.stdout.split()
    except (OSError, subprocess.TimeoutExpired):
        return False


def _named_step_absent(command: list[str]) -> bool:
    if "--job-name" not in command:
        return False
    try:
        name = command[command.index("--job-name") + 1]
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            return False
        result = subprocess.run(["squeue", "--steps", "--noheader", "--jobs", job_id,
                                 "--format=%i %j"], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", check=False, timeout=5)
        return result.returncode == 0 and not any(
            len(parts := line.split(maxsplit=1)) == 2 and parts[1] == name
            for line in result.stdout.splitlines())
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return False


def _run_srun(command: list[str], log_root: Path, name: str, deadline: float) -> dict:
    stdout_path, stderr_path = log_root / f"{name}.stdout.txt", log_root / f"{name}.stderr.txt"
    started = time.monotonic()
    process = None
    teardown = {"remote_step_teardown_verified": None, "step_id": None,
                "srun_exited_after_stop": None, "targeted_step_cancel_issued": False}
    outcome = None
    exit_code = -998
    launch_error = None
    termination_attempted = False
    with stdout_path.open("x", encoding="utf-8", newline="\n") as stdout, \
            stderr_path.open("x", encoding="utf-8", newline="\n") as stderr:
        try:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            teardown["step_id"] = _find_named_step(command)
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    outcome = "LIMIT_REACHED"
                    teardown.update(_terminate_srun(process, process.pid, teardown["step_id"]))
                    termination_attempted = True
                    break
                time.sleep(0.25)
            exit_code = process.poll() if process is not None else -998
            if exit_code is None:
                # The bounded teardown failed. Do not block forever or launch another step.
                exit_code = -999
                outcome = outcome or "REMOTE_STEP_TEARDOWN_UNRESOLVED"
            if teardown["remote_step_teardown_verified"] is None:
                step_id = teardown["step_id"]
                teardown["remote_step_teardown_verified"] = (
                    _step_absent(step_id) if step_id else _named_step_absent(command))
        except Exception as exc:
            launch_error = f"{type(exc).__name__}: {exc}"
            outcome = "SRUN_EXCEPTION"
            if process is not None and process.poll() is None:
                teardown.update(_terminate_srun(process, process.pid, teardown["step_id"]))
                termination_attempted = True
            exit_code = process.poll() if process is not None and process.poll() is not None else -998
        finally:
            if process is not None and process.poll() is None and not termination_attempted:
                teardown.update(_terminate_srun(process, process.pid, teardown["step_id"]))
                termination_attempted = True
                exit_code = process.poll() if process.poll() is not None else -999
            stdout.flush()
            stderr.flush()
    return {"name": name, "exit_code": exit_code, "outcome": outcome,
            "wall_seconds": time.monotonic() - started,
            "stdout_path": stdout_path.name, "stderr_path": stderr_path.name,
            "launch_or_run_error": launch_error,
            **teardown,
            "stderr_tail": "\n".join(stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])}


def _srun_command(nodes: int, hostlist: str, python: str, script: Path,
                  inner_args: list[str], name: str) -> list[str]:
    cmd = ["srun", "--nodes", str(nodes), "--ntasks", str(nodes),
           "--ntasks-per-node", "1", "--cpus-per-task", "96",
           "--cpu-bind=cores", "--exact", "--kill-on-bad-exit=1", "--job-name", name]
    if hostlist:
        cmd.extend(["--nodelist", hostlist])
    return cmd + [python, str(script), *inner_args]


def _validate_probe_dir(probe_dir: Path) -> dict:
    by_rank = {}
    for rank in (0, 1):
        path = probe_dir / f"probe-rank-{rank}.json"
        if not path.is_file():
            raise P1RError(f"shared probe directory missing rank receipt: {path.name}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("record_type") != "P1R_NODE_PROBE" or int(row.get("rank", -1)) != rank:
            raise P1RError(f"shared probe rank receipt malformed: {path.name}")
        by_rank[rank] = row
    if len({row["hostname"] for row in by_rank.values()}) != 2:
        raise P1RError("two-node srun probe did not prove distinct hostnames")
    allocation_expr = os.environ.get("SLURM_JOB_NODELIST")
    if not allocation_expr:
        raise P1RError("allocation NodeList is missing")
    query = subprocess.run(["scontrol", "show", "hostnames", allocation_expr],
                           capture_output=True, text=True, check=True, timeout=10)
    allocation_nodes = query.stdout.split()
    rank_nodes = [by_rank[rank].get("slurm_node_name") for rank in (0, 1)]
    if len(allocation_nodes) != 2 or len(set(allocation_nodes)) != 2 or rank_nodes != allocation_nodes:
        raise P1RError("Slurm NodeName/rank order differs from allocation host list")
    return {"rank_to_slurm_node": {str(rank): rank_nodes[rank] for rank in (0, 1)},
            "allocation_nodes_in_order": allocation_nodes,
            "rank_to_hostname": {str(rank): by_rank[rank]["hostname"] for rank in (0, 1)},
            "ranks": by_rank, "shared_directory_proof": "TWO_RANK_EXCLUSIVE_RECEIPTS_VISIBLE"}


def _run_probe(args, root: Path, deadline: float) -> tuple[dict, dict]:
    logs = root / "process_logs"
    probe_dir = logs / "probe_nodes"
    probe_dir.mkdir(mode=0o750)
    script = Path(__file__).resolve()
    command = _srun_command(2, "", sys.executable, script,
                            ["--mode", "probe", "--probe-dir", str(probe_dir)], "p1r-probe")
    accounting = _run_srun(command, logs, "two_node_probe", deadline)
    common.write_json(root / "srun_probe.json", accounting)
    if (accounting["exit_code"] != 0 or accounting["outcome"] or
            accounting.get("remote_step_teardown_verified") is not True):
        raise P1RError("two-node srun probe failed: " + json.dumps(accounting, sort_keys=True))
    probe = _validate_probe_dir(probe_dir)
    common.write_json(root / "probe.json", probe)
    return probe, accounting


def _stage_tasks(stage: str, node_count: int, rank: int, workload: dict,
                 layout: str) -> list[list[dict]]:
    return h2_tasks(workload) if layout in ("NUMA0", "STRIPED") else worker_tasks(workload, node_count, rank)


def _stage_run(args, root: Path, probe: dict, stage: str, node_count: int,
               layout: str, budget: float, deadline: float,
               raw_hash, canonical_hash) -> dict:
    stage_dir = root / "stages" / stage
    stage_dir.mkdir(mode=0o750)
    for rank in range(node_count):
        (stage_dir / f"node-{rank:02d}").mkdir(mode=0o750)
        (stage_dir / f"node-{rank:02d}" / "tasks").mkdir(mode=0o750)
    hosts = ",".join(probe["rank_to_slurm_node"][str(rank)] for rank in range(node_count))
    inner = ["--mode", "stage", "--stage", stage, "--layout", layout,
             "--node-count", str(node_count), "--stage-budget", str(budget),
             "--output-root", str(stage_dir), "--input-root", args.input_root,
             "--manifest", args.manifest, "--workload", args.workload,
             "--vendor-dir", args.vendor_dir or "", "--probe-json", str(root / "probe.json"),
             "--memory-limit-bytes", str(args.memory_limit_bytes)]
    script = Path(__file__).resolve()
    command = _srun_command(node_count, hosts, sys.executable, script, inner,
                            "p1r-" + stage.lower().replace("-", "-"))
    service_started = time.monotonic()
    accounting = _run_srun(command, root / "process_logs", f"stage-{stage}", deadline)
    node_summaries = []
    for rank in range(node_count):
        path = stage_dir / f"node-{rank:02d}" / "node_summary.json"
        if path.is_file():
            node_summaries.append(json.loads(path.read_text(encoding="utf-8")))
    planned = H2_TASK_COUNT if layout in ("NUMA0", "STRIPED") else TASK_COUNT
    completed = sum(int(item.get("tasks_completed", 0)) for item in node_summaries)
    output_validation = validate_stage_outputs(stage_dir, args._workload_value,
                                               node_count if layout == "H1" else 1,
                                               layout, raw_hash, canonical_hash, deadline)
    service_wall = time.monotonic() - service_started
    status = "COMPLETE" if accounting["exit_code"] == 0 and not accounting["outcome"] and \
        accounting.get("remote_step_teardown_verified") is True and \
        len(node_summaries) == node_count and all(item.get("status") == "COMPLETE" for item in node_summaries) and completed == planned and output_validation["status"] == "COMPLETE" else "PARTIAL"
    record = {"schema": SCHEMA, "classification": CLASSIFICATION,
              "production_qualification": QUALIFICATION, "stage": stage,
              "layout": layout, "node_count": node_count, "status": status,
              "tasks_planned": planned, "tasks_completed": completed,
              "coordinator_wall_seconds": accounting["wall_seconds"],
              "coordinator_service_wall_seconds": service_wall,
              "output_validation": output_validation,
              "coordinator_deadline_outcome": accounting["outcome"],
              "srun_exit_code": accounting["exit_code"], "srun_stdout": accounting["stdout_path"],
              "srun_stderr": accounting["stderr_path"], "node_summaries": node_summaries}
    return record


def _load_valid_output(stage_dir: Path, task_id: str, receipt: dict, raw_hash, canonical_hash):
    import numpy as np
    path = stage_dir / f"node-{int(receipt['rank']):02d}" / "tasks" / f"{task_id}.npz"
    if not path.is_file() or raw_hash(path) != receipt["output"]["raw_sha256"]:
        raise P1RError(f"output raw hash missing/mismatch for {task_id} in {stage_dir.name}")
    with np.load(path, allow_pickle=False) as data:
        fields = {name: np.array(data[name], copy=True, order="K") for name in FIELDS}
    for name, arr in fields.items():
        if arr.shape != p1.EXPECTED_SHAPE or arr.dtype != np.dtype("float64") or not arr.flags.c_contiguous or not np.isfinite(arr).all():
            raise P1RError(f"output integrity failed for {task_id}:{name}")
        if canonical_hash(arr) != receipt["output"]["canonical_field_sha256"][name]:
            raise P1RError(f"output canonical hash mismatch for {task_id}:{name}")
    return fields


def validate_stage_outputs(stage_dir: Path, workload: dict, node_count: int, layout: str,
                           raw_hash, canonical_hash, deadline: float | None = None) -> dict:
    expected_tasks = workload["tasks"][:H2_TASK_COUNT] if layout in ("NUMA0", "STRIPED") else workload["tasks"]
    receipt_map = _receipts(stage_dir)
    errors = []
    validated = 0
    for task in expected_tasks:
        if deadline is not None and time.monotonic() >= deadline:
            errors.append("OUTPUT_VALIDATION_LIMIT_REACHED")
            break
        task_id = task["task_id"]
        receipt = receipt_map.get(task_id)
        if receipt is None:
            errors.append(f"MISSING_VALID_TASK:{task_id}")
            continue
        if layout == "H1":
            assignment = task["assignment_1n"] if node_count == 1 else task["assignment_2n"]
            expected_rank, expected_worker = int(assignment["rank"]), int(assignment["worker_index"])
        else:
            expected_rank, expected_worker = 0, int(task["index"]) % 24
        if int(receipt["rank"]) != expected_rank or int(receipt["worker_index"]) != expected_worker:
            errors.append(f"ASSIGNMENT_RECEIPT_MISMATCH:{task_id}")
            continue
        try:
            _load_valid_output(stage_dir, task_id, receipt, raw_hash, canonical_hash)
        except (OSError, ValueError, KeyError, P1RError, json.JSONDecodeError) as exc:
            errors.append(f"OUTPUT_INVALID:{task_id}:{type(exc).__name__}:{exc}")
            continue
        validated += 1
    return {"status": "COMPLETE" if validated == len(expected_tasks) and not errors else "PARTIAL",
            "expected_tasks": len(expected_tasks), "validated_tasks": validated,
            "raw_and_canonical_hashes_verified": True, "errors": errors[:50]}


def _receipts(stage_dir: Path) -> dict[str, dict]:
    found = {}
    for path in stage_dir.glob("node-*/tasks/task-*.json"):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("status") == "DONE" and receipt.get("classification") == CLASSIFICATION:
            found[receipt["task_id"]] = receipt
    return found


def _append_stage_csv(root: Path, stage: str, stage_dir: Path) -> int:
    receipt_map = _receipts(stage_dir)
    with (root / "performance_summary.csv").open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        for task_id, receipt in sorted(receipt_map.items()):
            perf = receipt["timing_seconds"]
            cmp = receipt["historical_next_field_comparison"]
            writer.writerow({"stage": stage, "task_id": task_id, "status": receipt["status"],
                             "rank": receipt["rank"], "worker_index": receipt["worker_index"],
                             "worker_cpu": receipt["worker_cpu"], "ordinal": receipt["ordinal"],
                             "repetition": receipt["repetition"], "kernel_wall": perf["kernel_wall"],
                             "kernel_cpu": perf["kernel_cpu"], "service_wall": perf["service_wall"],
                             "historical_next_delta_n_max_abs": cmp["delta_n"]["max_abs"],
                             "historical_next_vx_max_abs": cmp["vx"]["max_abs"],
                             "historical_next_vy_max_abs": cmp["vy"]["max_abs"],
                             "output_sha256": receipt["output"]["raw_sha256"]})
        stream.flush()
        os.fsync(stream.fileno())
    return len(receipt_map)


def _contains_nonzero_delta(value) -> bool:
    if isinstance(value, dict):
        if "delta" in value:
            try:
                if float(value["delta"]) != 0.0:
                    return True
            except (TypeError, ValueError):
                return True
        return any(_contains_nonzero_delta(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_nonzero_delta(item) for item in value)
    return False


def _cross_compare_with_gate(root: Path, baseline_stage: str, current_stage: str,
                             task_ids: list[str], raw_hash, canonical_hash) -> dict:
    baseline_dir, current_dir = root / "stages" / baseline_stage, root / "stages" / current_stage
    baseline, current = _receipts(baseline_dir), _receipts(current_dir)
    path = root / "cross_stage_comparison.jsonl"
    compared = hash_diff = observable_diff = 0
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for task_id in task_ids:
            if task_id not in baseline or task_id not in current:
                continue
            left = _load_valid_output(baseline_dir, task_id, baseline[task_id], raw_hash, canonical_hash)
            right = _load_valid_output(current_dir, task_id, current[task_id], raw_hash, canonical_hash)
            hash_match = baseline[task_id]["output"]["canonical_field_sha256"] == current[task_id]["output"]["canonical_field_sha256"]
            observable_error = p1._numeric_observable_delta(
                current[task_id].get("observables", {}), baseline[task_id].get("observables", {}))
            obs_changed = _contains_nonzero_delta(observable_error)
            row = {"schema": SCHEMA, "classification": CLASSIFICATION,
                   "comparison_scope": "SAME_TASK_ACROSS_P1R_CPU_LAYOUTS_DIAGNOSTIC_ONLY",
                   "baseline_stage": baseline_stage, "stage": current_stage,
                   "task_id": task_id, "canonical_hash_match": hash_match,
                   "field_error": p1.compare_fields(right, left),
                   "observable_error": observable_error,
                   "observable_difference": obs_changed}
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            compared += 1
            hash_diff += not hash_match
            observable_diff += obs_changed
    gate = "CONTINUE" if compared == len(task_ids) and hash_diff == 0 and observable_diff == 0 else \
        "NUMERICAL_DIFFERENCE_STOP_OPTIONAL_AND_REMAINING_STAGES"
    return {"baseline_stage": baseline_stage, "stage": current_stage,
            "planned_task_comparisons": len(task_ids), "completed_task_comparisons": compared,
            "canonical_hash_differences": hash_diff, "observable_differences": observable_diff,
            "continue_gate": gate, "status": "COMPLETE" if compared == len(task_ids) else "PARTIAL"}


def _environment(root: Path, probe: dict, allocation: dict, details: dict,
                 workload_sha: str, backend: dict, vendor_dir: str | None) -> dict:
    import numpy as np
    import threadpoolctl
    import contextlib
    import io
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.__config__.show()
    common.write_new(root / "numpy_config.txt", config.getvalue())
    pools = {"version": getattr(threadpoolctl, "__version__", None),
             "pools": threadpoolctl.threadpool_info()}
    common.write_json(root / "threadpool_info.json", pools)
    lscpu = common.optional_command("lscpu", ["-e"])
    common.write_new(root / "lscpu_e.txt", lscpu["stdout"])
    srun_version = common.optional_command("srun", ["--version"])
    scheduler_config = common.optional_command("scontrol", ["show", "config"])
    config_lines = scheduler_config["stdout"].splitlines()
    filtered = [line.strip() for line in config_lines
                if line.strip().startswith(("ProctrackType", "KillWait"))]
    common.write_new(root / "slurm_step_contract.txt",
                     "srun --exact and --kill-on-bad-exit=1 are required by this runner.\n" +
                     "".join(line + "\n" for line in filtered) +
                     f"srun_version_status={srun_version['status']}\n{srun_version['stdout']}")
    metadata = {"schema": SCHEMA, "classification": CLASSIFICATION,
                "production_qualification": QUALIFICATION,
                "python": sys.version, "python_executable": sys.executable,
                "numpy_version": np.__version__, "threadpoolctl_version": pools["version"],
                "backend": backend, "vendor_dir": vendor_dir,
                "input_manifest_raw_sha256": details["raw_sha256"],
                "workload_raw_sha256": workload_sha,
                "input_identity": details["manifest"].get("identity"),
                "hydro_parameters": details["manifest"].get("hydro"),
                "allocation_resource_environment": allocation,
                "probe_rank_to_hostname": probe["rank_to_hostname"],
                "lscpu_status": lscpu["status"]}
    metadata["slurm_version_status"] = srun_version["status"]
    metadata["slurm_proctrack_killwait_status"] = scheduler_config["status"]
    common.write_json(root / "environment.json", metadata)
    return metadata


def _entry_smoke_worker(output_root: str, advance, threadpoolctl, canonical_hash) -> None:
    """Tiny synthetic execution of the real frozen function through a forked P1R writer."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if not hasattr(os, "sched_getaffinity"):
        raise P1RError("Linux affinity API is required for entry smoke")
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise P1RError("entry smoke received empty CPU affinity")
    os.sched_setaffinity(0, {allowed[0]})
    import numpy as np
    fields = {name: np.zeros((6, 5), dtype=np.float64, order="C") for name in FIELDS}
    manifest = {"hydro": p1.EXPECTED_HYDRO_PARAMETERS,
                "dx_m": p1.EXPECTED_DX_M, "dy_m": p1.EXPECTED_DY_M}
    with threadpoolctl.threadpool_limits(limits=1):
        pools = threadpoolctl.threadpool_info()
        if any(int(item.get("num_threads", 1)) > 1 for item in pools):
            raise P1RError("entry-smoke worker thread pool exceeds one thread")
        result = p1.invoke_hydro(advance, fields, manifest)
    output = {name: np.asarray(result[name]) for name in FIELDS}
    if any(arr.shape != (6, 5) or arr.dtype != np.dtype("float64") or
           not arr.flags.c_contiguous or not np.isfinite(arr).all() for arr in output.values()):
        raise P1RError("entry-smoke frozen HR-4 output integrity failed")
    root = Path(output_root)
    data_path = root / "software_smoke.npz"
    write_s, raw_s, canonical_s, raw_digest, field_digests = p1._write_result_arrays(data_path, output)
    if field_digests != {name: canonical_hash(array) for name, array in output.items()}:
        raise P1RError("entry-smoke writer canonical hashes differ from repository array hashing")
    receipt = {"schema": SCHEMA, "status": "COMPLETE", "classification": CLASSIFICATION,
               "production_qualification": QUALIFICATION,
               "input_kind": "SOFTWARE_SMOKE_ONLY_SYNTHETIC",
               "scientific_inputs_used": False,
               "call_contract": {"n_steps": p1.STEPS, "require_stable": True,
                                 "backend": result["performance"].get("backend"),
                                 "shape": [6, 5], "dtype": "float64"},
               "worker_affinity": sorted(os.sched_getaffinity(0)),
               "worker_threadpools": pools,
               "steps": result["steps"], "stability": p1._jsonable(result.get("stability")),
               "timing_seconds": {"kernel_reported": result["performance"]["wall_time_s_total"],
                                  "output_write": write_s, "output_raw_hash": raw_s,
                                  "output_canonical_hash": canonical_s},
               "output": {"relative_path": data_path.name, "raw_sha256": raw_digest,
                          "canonical_field_sha256": field_digests}}
    p1.sync_json_new(root / "software_smoke.json", receipt)


def run_entry_smoke(args) -> int:
    _validate_interpreter()
    common.add_vendor(args.vendor_dir)
    advance, canonical_hash, raw_hash, threadpoolctl, backend = p1._load_science()
    if backend.get("backend") != "numpy":
        raise P1RError("entry smoke requires actual NumPy CPU backend")
    output = common.create_output_root(Path(args.output_root))
    if "fork" not in mp.get_all_start_methods():
        raise P1RError("Linux fork unavailable for P1R entry smoke")
    process = mp.get_context("fork").Process(target=_entry_smoke_worker,
                                               args=(str(output), advance, threadpoolctl, canonical_hash))
    started = time.monotonic()
    process.start()
    outcome = None
    while process.is_alive():
        if time.monotonic() - started >= args.smoke_timeout_seconds:
            outcome = "LIMIT_REACHED"
            p1._stop_processes([process])
            break
        time.sleep(0.1)
    process.join(timeout=1)
    root_receipt = output / "software_smoke.json"
    status = "COMPLETE" if process.exitcode == 0 and root_receipt.is_file() and outcome is None else "FAIL"
    verification = {"status": status, "classification": CLASSIFICATION,
                    "production_qualification": QUALIFICATION,
                    "input_kind": "SOFTWARE_SMOKE_ONLY_SYNTHETIC",
                    "wall_seconds": time.monotonic() - started,
                    "child_exit_code": process.exitcode, "outcome": outcome,
                    "local_process_tree_cleanup_verified": not process.is_alive()}
    if status == "COMPLETE":
        receipt = json.loads(root_receipt.read_text(encoding="utf-8"))
        data_path = output / receipt["output"]["relative_path"]
        if raw_hash(data_path) != receipt["output"]["raw_sha256"]:
            raise P1RError("parent entry-smoke raw output revalidation failed")
        import numpy as np
        with np.load(data_path, allow_pickle=False) as archive:
            output_fields = {name: np.array(archive[name], copy=True, order="K") for name in FIELDS}
        parent_hashes = {name: canonical_hash(array) for name, array in output_fields.items()}
        if parent_hashes != receipt["output"]["canonical_field_sha256"]:
            raise P1RError("parent entry-smoke canonical output revalidation failed")
        verification["output_raw_sha256"] = receipt["output"]["raw_sha256"]
        verification["output_canonical_field_sha256"] = parent_hashes
    common.write_json(output / "status.json", verification)
    print(json.dumps({"status": status, "output_root": str(output),
                      "classification": CLASSIFICATION,
                      "input_kind": "SOFTWARE_SMOKE_ONLY_SYNTHETIC"}, sort_keys=True), flush=True)
    return 0 if status == "COMPLETE" else 1


def _allocation_start(args) -> float:
    value = args.allocation_start_epoch or os.environ.get("P1R_ALLOCATION_START_EPOCH")
    if not value:
        raise P1RError("allocation start epoch missing; refusing to reset the 90-minute budget")
    start = float(value)
    if not math.isfinite(start) or start > time.time() + 5:
        raise P1RError("allocation start epoch is invalid")
    return start


def run_controller(args) -> int:
    _validate_interpreter()
    allocation = validate_allocation_2n(os.environ)
    if not Path("/proc").is_dir():
        raise P1RError("Linux /proc is required for node-local RSS and CPU monitoring")
    common.add_vendor(args.vendor_dir)
    from tools.hpc_ops.provenance_v2 import raw_sha256_file
    input_root = Path(args.input_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    details, records = p1.load_manifest(manifest_path, input_root, raw_sha256_file)
    workload, workload_sha = validate_workload(Path(args.workload), details, records, raw_sha256_file)
    args._workload_value = workload
    allocation_start = _allocation_start(args)
    hard_budget = MAX_ALLOCATION_SECONDS - FINAL_RESERVE_SECONDS - max(0.0, time.time() - allocation_start)
    if hard_budget <= 0:
        raise P1RError("allocation has passed its conservative 90-minute execution deadline")
    output = common.create_output_root(Path(args.output_root))
    (output / "stages").mkdir(mode=0o750)
    (output / "process_logs").mkdir(mode=0o750)
    with (output / "performance_summary.csv").open("x", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=CSV_FIELDS).writeheader()
    (output / "cross_stage_comparison.jsonl").open("x", encoding="utf-8").close()
    coordinator_started = time.monotonic()
    hard_deadline = coordinator_started + hard_budget
    probe, probe_accounting = _run_probe(args, output, hard_deadline)
    advance, canonical_hash, raw_hash, threadpoolctl, backend = p1._load_science()
    if backend.get("backend") != "numpy" or getattr(threadpoolctl, "__version__", None) != "3.6.0":
        raise P1RError("coordinator CPU backend/threadpoolctl verification failed")
    _environment(output, probe, allocation, details, workload_sha, backend, args.vendor_dir)
    common.write_json(output / "input_acceptance.json", {
        "schema": SCHEMA, "classification": CLASSIFICATION,
        "production_qualification": QUALIFICATION, "status": "ACCEPTED_FOR_P1R_DIAGNOSTIC",
        "input_manifest_raw_sha256": details["raw_sha256"], "workload_raw_sha256": workload_sha,
        "workload_tasks": TASK_COUNT, "sources": SOURCE_COUNT, "repetitions": REPETITIONS,
        "input_identity": details["manifest"]["identity"],
        "note": "Each worker rechecks source raw/canonical hashes when loading each task."})
    stage_records = []
    stage_plan = list(STAGES)
    observed_stage_seconds = {1: HISTORICAL_STAGE_ESTIMATE_SECONDS,
                              2: HISTORICAL_STAGE_ESTIMATE_SECONDS}
    numerical_difference = False
    for i, (stage, nodes, layout) in enumerate(stage_plan):
        remaining_after = len(stage_plan) - i - 1
        reserve = STAGE_RESERVE_SECONDS * remaining_after
        stage_deadline = hard_deadline - reserve
        budget = stage_deadline - time.monotonic()
        available = hard_deadline - time.monotonic()
        estimate = observed_stage_seconds[nodes]
        if available < estimate + reserve:
            record = {"stage": stage, "status": "SKIPPED_NO_BUDGET",
                      "tasks_planned": TASK_COUNT, "tasks_completed": 0,
                      "estimated_stage_seconds": estimate, "remaining_seconds": max(0.0, budget),
                      "future_stage_reserve_seconds": reserve}
            common.write_json(output / "stages" / f"{stage}.json", record)
            stage_records.append(record)
            break
        record = _stage_run(args, output, probe, stage, nodes, layout, budget, stage_deadline,
                            raw_sha256_file, canonical_hash)
        record["published_task_receipts"] = _append_stage_csv(output, stage, output / "stages" / stage)
        if stage != "1N-A" and stage_records and stage_records[0].get("status") == "COMPLETE":
            comparison = _cross_compare_with_gate(
                output, "1N-A", stage, [f"task-{index:04d}" for index in range(TASK_COUNT)],
                raw_sha256_file, canonical_hash)
            record["cross_stage_comparison"] = comparison
            if comparison["continue_gate"].startswith("NUMERICAL_DIFFERENCE"):
                numerical_difference = True
                record["status"] = "NUMERICAL_DIFFERENCE"
            elif comparison["continue_gate"] != "CONTINUE" and record["status"] == "COMPLETE":
                record["status"] = "PARTIAL"
                record["stop_reason"] = "CROSS_STAGE_COMPARISON_INCOMPLETE"
        common.write_json(output / "stages" / f"{stage}.json", record)
        stage_records.append(record)
        if record.get("status") == "COMPLETE":
            observed_stage_seconds[nodes] = max(
                HISTORICAL_STAGE_ESTIMATE_SECONDS,
                float(record["coordinator_service_wall_seconds"]) * STAGE_ESTIMATE_SAFETY_FACTOR)
        if record.get("status") != "COMPLETE":
            break
    h2_records = []
    h2_ready = len(stage_records) == 4 and all(item.get("status") == "COMPLETE" for item in stage_records)
    remaining = hard_deadline - time.monotonic()
    h2_skipped_reason = None
    if args.include_h2 and h2_ready:
        h2_stage_estimate = max(STAGE_RESERVE_SECONDS,
                                observed_stage_seconds[1] * H2_TASK_COUNT / TASK_COUNT)
        for i, (stage, layout) in enumerate(H2_STAGES):
            remaining_after = len(H2_STAGES) - i - 1
            reserve = STAGE_RESERVE_SECONDS * remaining_after
            deadline = hard_deadline - reserve
            budget = deadline - time.monotonic()
            available = hard_deadline - time.monotonic()
            if available < h2_stage_estimate + reserve:
                h2_skipped_reason = "SKIPPED_NO_BUDGET"
                for pending_stage, _ in H2_STAGES[i:]:
                    skipped = {"stage": pending_stage, "status": h2_skipped_reason,
                               "tasks_planned": H2_TASK_COUNT, "tasks_completed": 0,
                               "estimated_stage_seconds": h2_stage_estimate,
                               "future_stage_reserve_seconds": reserve}
                    common.write_json(output / "stages" / f"{pending_stage}.json", skipped)
                    h2_records.append(skipped)
                break
            record = _stage_run(args, output, probe, stage, 1, layout, budget, deadline,
                                raw_sha256_file, canonical_hash)
            record["published_task_receipts"] = _append_stage_csv(output, stage, output / "stages" / stage)
            comparison = _cross_compare_with_gate(
                output, "1N-A", stage, [f"task-{index:04d}" for index in range(H2_TASK_COUNT)],
                raw_sha256_file, canonical_hash)
            record["cross_stage_comparison"] = comparison
            if comparison["continue_gate"].startswith("NUMERICAL_DIFFERENCE"):
                numerical_difference = True
                record["status"] = "NUMERICAL_DIFFERENCE"
            elif comparison["continue_gate"] != "CONTINUE" and record["status"] == "COMPLETE":
                record["status"] = "PARTIAL"
                record["stop_reason"] = "CROSS_STAGE_COMPARISON_INCOMPLETE"
            common.write_json(output / "stages" / f"{stage}.json", record)
            h2_records.append(record)
            if record.get("status") != "COMPLETE":
                break
    else:
        h2_skipped_reason = ("SKIPPED_NOT_REQUESTED" if not args.include_h2 else
                             "SKIPPED_H1_INCOMPLETE_OR_NUMERICAL_GATE")
    if h2_skipped_reason or not h2_records or any(item.get("status") != "COMPLETE" for item in h2_records):
        common.write_json(output / "H2_status.json", {"schema": SCHEMA, "classification": CLASSIFICATION,
                          "status": h2_skipped_reason or "PARTIAL",
                          "remaining_seconds": max(0.0, hard_deadline - time.monotonic()),
                          "required_reserve_seconds_per_remaining_stage": STAGE_RESERVE_SECONDS,
                          "stages": h2_records})
    h1_complete = len(stage_records) == 4 and all(item.get("status") == "COMPLETE" for item in stage_records)
    h2_failed = bool(h2_records and any(item.get("status") not in ("COMPLETE", "SKIPPED_NO_BUDGET") for item in h2_records))
    completed_tasks = sum(int(item.get("tasks_completed", 0)) for item in stage_records)
    status = ("NUMERICAL_DIFFERENCE" if numerical_difference else
              "COMPLETE" if h1_complete and not h2_failed else
              "PARTIAL" if completed_tasks else "INCOMPLETE")
    summary = {"schema": SCHEMA, "classification": CLASSIFICATION,
               "production_qualification": QUALIFICATION, "status": status,
               "workload_raw_sha256": workload_sha, "input_manifest_raw_sha256": details["raw_sha256"],
               "h1_stage_count": len(stage_records), "h1_stages": stage_records,
               "h2_stages": h2_records or "SKIPPED_OR_NOT_RUN",
               "coordinator_wall_seconds": time.monotonic() - coordinator_started,
               "estimated_elapsed_allocated_core_hours_not_slurm_accounting":
                   max(0.0, time.time() - allocation_start) * 192 / 3600,
               "h1_complete": h1_complete,
               "qualification_statement": "P1R_DIAGNOSTIC_ONLY; NOT_PRODUCTION_QUALIFIED"}
    common.write_json(output / "status.json", summary)
    print(json.dumps({"status": status, "output_root": str(output),
                      "h1_complete": h1_complete, "h2_stage_count": len(h2_records)}, sort_keys=True), flush=True)
    return 0 if status == "COMPLETE" else 1


def run_stage(args) -> int:
    if args.stage_budget <= 0 or args.stage_budget > MAX_ALLOCATION_SECONDS:
        raise P1RError("stage budget is outside the authorized allocation window")
    args._workload_value = json.loads(Path(args.workload).read_text(encoding="utf-8-sig"))
    return _node_stage(args)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("controller", "probe", "stage"), default="controller")
    parser.add_argument("--input-root")
    parser.add_argument("--manifest")
    parser.add_argument("--workload")
    parser.add_argument("--output-root")
    parser.add_argument("--vendor-dir")
    parser.add_argument("--allocation-start-epoch")
    parser.add_argument("--memory-limit-bytes", type=int, default=MEMORY_LIMIT_BYTES)
    parser.add_argument("--include-h2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--entry-smoke", action="store_true")
    parser.add_argument("--smoke-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--stage")
    parser.add_argument("--layout", default="H1")
    parser.add_argument("--node-count", type=int, default=1)
    parser.add_argument("--stage-budget", type=float, default=0)
    parser.add_argument("--probe-json")
    parser.add_argument("--probe-dir")
    args = parser.parse_args(argv)
    try:
        if args.mode == "probe":
            os.environ["UPPE_USE_GPU"] = "0"
            if not args.probe_dir:
                raise P1RError("probe mode requires a shared --probe-dir")
            write_probe_record(Path(args.probe_dir))
            return 0
        if args.entry_smoke:
            if not args.output_root:
                raise P1RError("--entry-smoke requires --output-root")
            if not 0 < args.smoke_timeout_seconds <= 60:
                raise P1RError("entry-smoke timeout must be in 1..60 seconds")
            return run_entry_smoke(args)
        if args.mode == "stage":
            required = (args.stage, args.output_root, args.input_root, args.manifest,
                        args.workload, args.probe_json, args.vendor_dir)
            if not all(required):
                raise P1RError("stage mode requires stage, input, output, workload, probe, and vendor paths")
            return run_stage(args)
        required = (args.input_root, args.manifest, args.workload, args.output_root)
        if not all(required):
            raise P1RError("controller requires --input-root, --manifest, --workload and --output-root")
        if not 0 < args.memory_limit_bytes <= MEMORY_LIMIT_BYTES:
            raise P1RError("per-node memory limit exceeds the approved 320 GiB allocation")
        return run_controller(args)
    except BaseException as exc:
        print(json.dumps({"schema": SCHEMA, "status": "FAIL",
                          "classification": CLASSIFICATION,
                          "production_qualification": QUALIFICATION,
                          "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
