"""C/P24 propagation, capture, and independent local mechanism entry."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback
from typing import Any
from types import SimpleNamespace

import numpy as np

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_3g3_smallgrid.candidate import unoptimized
from tools.cpu_s4_5b_steady_cost.entry import read, require
from tools.hpc_ops import provenance_v2 as pv

from .c_capture import CaptureStore
from .c_mechanism import run_local_mechanism
from .c_formal14_protocol import BASE as C14_BASE, EXECUTION_SCHEMA as C14_EXECUTION_SCHEMA, HPC_EXECUTION_SCHEMA as C14_HPC_EXECUTION_SCHEMA, MECHANISM_EXECUTION as C14_MECHANISM_EXECUTION, TASK as C14_TASK, plan as c14_plan
from .c_protocol import BASE, CAPTURE_INTERVALS, CAPTURE_TOTAL_CALLS, HPC_EXECUTION_SCHEMA, MICRO_SHAPE, SCIENCE_TREE, SHAPE, TASK, plan


SCHEMA = "filament.cpu_s4_6e.c_entry.v1"


def admission(*, local_engineering: bool, shape: tuple[int, ...], intervals: int, mode: str, role: str, capture: bool = False) -> None:
    require(str(role).upper() in ("R", "C"), "S46E_C_FIXED_ROLES")
    require((intervals, mode) in ((2, "QUALIFY_BOUNDARY"), (8, "PURE_OFF_SMALLGRID")), "S46E_C_TWO_QUAL_OR_EIGHT_OFF_ONLY")
    if capture:
        require(str(role).upper() == "R" and intervals == 8 and mode == "PURE_OFF_SMALLGRID", "S46E_C_CAPTURE_ONLY_R8")
    if local_engineering:
        require(not os.environ.get("SLURM_JOB_ID") and tuple(shape) == MICRO_SHAPE, "S46E_C_LOCAL_NT384_MICRO_ONLY")
    else:
        require(os.environ.get("SLURM_JOB_ID", "").isdigit() and tuple(shape) == SHAPE, "S46E_C_ACTUAL_SLURM_FIXED_SMALLGRID_ONLY")


def verify_source(report: str | Path, *, local_engineering: bool) -> dict:
    unoptimized()
    report = Path(report).resolve(strict=True)
    binding = read(report / "EXECUTION.json")
    is_c14 = binding.get("schema") == C14_EXECUTION_SCHEMA and binding.get("task") == C14_TASK
    is_c = binding.get("schema") == "filament.cpu_s4_6e_c.execution.v1" and binding.get("task") == TASK
    require(is_c14 or is_c, "S46E_C_EXECUTION_SCHEMA")
    expected_plan = c14_plan() if is_c14 else plan()
    expected_base = C14_BASE if is_c14 else BASE
    require(binding["baseline_commit"] == expected_base and binding["science_tree"] == SCIENCE_TREE, "S46E_C_FROZEN_SOURCE_CHAIN")
    require(read(report / "PLAN.json") == expected_plan, "S46E_C_FIXED_PLAN_IDENTITY")
    require(pv.raw_sha256_file(report / "PLAN.json") == binding["plan_raw_sha256"], "S46E_C_PLAN_IDENTITY")
    repo = Path(__file__).resolve().parents[3]
    if not local_engineering:
        from tools.cpu_s4_checkpoint_support.git_readonly_compat import verify_runtime_query_tool

        execution_plan = read(report / "HPC_EXECUTION_PLAN.json")
        expected_hpc_schema = C14_HPC_EXECUTION_SCHEMA if is_c14 else HPC_EXECUTION_SCHEMA
        hpc_hash_env = "S46E_C14_HPC_PLAN_SHA256" if is_c14 else "S46E_HPC_PLAN_SHA256"
        execution_hash_env = "S46E_C14_EXECUTION_SHA256" if is_c14 else "S46E_EXECUTION_SHA256"
        require(execution_plan.get("schema") == expected_hpc_schema, "S46E_C_HPC_PLAN_SCHEMA")
        require(pv.raw_sha256_file(report / "HPC_EXECUTION_PLAN.json") == os.environ.get(hpc_hash_env), "S46E_C_HPC_PLAN_IDENTITY")
        verify_runtime_query_tool(execution_plan)
        require(pv.raw_sha256_file(report / "EXECUTION.json") == os.environ.get(execution_hash_env), "S46E_C_BINDING_IDENTITY")
    require(pv.raw_sha256_file(report / "EXECUTION_SOURCE_MANIFEST.json") == binding["source_manifest_raw_sha256"], "S46E_C_SOURCE_MANIFEST_IDENTITY")
    source = read(report / "EXECUTION_SOURCE_MANIFEST.json")
    require(pv._git(repo, "rev-parse", "HEAD") == binding["execution_commit"], "S46E_C_COMMITTED_EXECUTION_HEAD")
    pv._ensure_clean(repo)
    for row in source["records"]:
        require(row["classification"] == "tracked_text" and row["hash_scope"] == pv.TRACKED_HASH_SCOPE, "S46E_C_SOURCE_HASH_SCOPE", row.get("path"))
        path = repo / row["path"]
        require(not Path(row["path"]).is_absolute() and path.resolve().is_relative_to(repo), "S46E_C_SOURCE_CONFINEMENT")
        require(pv._git(repo, "rev-parse", "HEAD:" + row["path"]) == row["git_blob_oid"] and pv.canonical_lf_sha256(path.read_bytes()) == row["canonical_lf_sha256"], "S46E_C_SOURCE_BLOB_CANONICAL", row["path"])
    require(source["head"] == binding["execution_commit"] and len(source["records"]) >= 252, "S46E_C_COMPLETE_SOURCE_CLOSURE")
    require(pv._git(repo, "rev-parse", "HEAD:Filament_python/KHz_filament") == SCIENCE_TREE, "S46E_C_UNCHANGED_SCIENCE_SUBTREE")
    require(pv.raw_sha256_file(report / "INPUT_CONTRACT.json") == binding["input_contract_raw_sha256"] and pv.raw_sha256_file(report / "REFERENCE_INPUT_RAW_RECORDS.json") == binding["reference_contract_raw_sha256"], "S46E_C_FROZEN_INPUT_CONTRACTS")
    if is_c14:
        require(binding.get("mechanism_execution_commit") == C14_MECHANISM_EXECUTION, "C14_MECHANISM_EXECUTION_BINDING")
        admission_path = report / "MECHANISM_ADMISSION.json"
        require(admission_path.is_file() and pv.raw_sha256_file(admission_path) == binding.get("mechanism_admission_raw_sha256"), "C14_MECHANISM_ADMISSION_IDENTITY")
        admission = read(admission_path)
        require(admission.get("status") == "PASS_C14_MECHANISM_ADMISSION" and admission.get("mechanism_execution_commit") == C14_MECHANISM_EXECUTION and int(admission.get("formal_calls", -1)) == 90 and int(admission.get("warmup_calls", -1)) == 36 and admission.get("all_configs_exact") is True, "C14_MECHANISM_ADMISSION_GATE")
        gains = admission.get("p24_block_gains")
        require(isinstance(gains, dict) and len(gains) == 6 and all(float(value) >= 0.05 for value in gains.values()), "C14_MECHANISM_GAIN_GATE")
        from .c_formal14_evidence import SOURCE_IDENTITY_PATHS

        current_source = {str(row.get("path")): row.get("canonical_lf_sha256") for row in source["records"] if isinstance(row, dict)}
        admitted_source = admission.get("source_identities")
        require(isinstance(admitted_source, dict), "C14_MECHANISM_SOURCE_IDENTITIES")
        for relative in SOURCE_IDENTITY_PATHS:
            require(admitted_source.get(relative) == current_source.get(relative), "C14_MECHANISM_SOURCE_IDENTITY", relative)
    return binding


def verify_execution(report: str | Path, *, local_engineering: bool, shape: tuple[int, ...], intervals: int, mode: str, role: str, capture: bool = False) -> dict:
    admission(local_engineering=local_engineering, shape=shape, intervals=intervals, mode=mode, role=role, capture=capture)
    binding = verify_source(report, local_engineering=local_engineering)
    if not local_engineering:
        gate = read(Path(report) / "NODE_GATE.json")
        require(gate["status"] == "PASS_S46E_C_SMALLGRID_SOURCE_RESOURCE_INPUT_GATES" and gate["job_id"] == os.environ["SLURM_JOB_ID"], "S46E_C_ACTUAL_NODE_GATE")
    return binding


def input_domain(manifest: dict, *, local_engineering: bool) -> None:
    expected = MICRO_SHAPE if local_engineering else SHAPE
    require(manifest.get("shape") == list(expected), "S46E_C_ACTUAL_MANIFEST_SHAPE_DOMAIN")


def _source_identity_digest() -> str:
    from .c_candidate import source_identity

    payload = json.dumps(source_identity(), sort_keys=True, default=str).encode("utf-8")
    return pv.raw_sha256(payload)


def _write_entry_failure(work: Path, error: BaseException) -> dict:
    """Persist a bounded failure receipt before the child exits."""

    text = repr(error)
    trace = traceback.format_exc()
    trace_path = work / "ENTRY_FAILURE.traceback.txt"
    trace_path.write_text(trace, encoding="utf-8")
    science = any(
        tag in text
        for tag in (
            "STOP_FIRST_SCIENCE_NONEXACT",
            "STOP_FIRST_STAGE_NONEXACT",
            "S46E_ORIGINAL_108_EXACT_4_TIMING_CLASSIFICATION",
            "NONEXACT",
        )
    )
    failure_index = copy.deepcopy(getattr(error, "failure", None))
    exact_paths = [
        str(path.relative_to(work)).replace("\\", "/")
        for path in sorted(work.rglob("*.jsonl"))
        if "EXACT" in path.name.upper()
    ]
    payload = {
        "schema": SCHEMA + ".failure.v1",
        "classification": "SCIENTIFIC_NONEXACT" if science else "ENGINEERING_OR_RESOURCE_FAILURE",
        "stop_reason": "STOP_FIRST_SCIENCE_NONEXACT" if "STOP_FIRST_SCIENCE_NONEXACT" in text else ("STOP_FIRST_STAGE_NONEXACT" if "STOP_FIRST_STAGE_NONEXACT" in text else ("S46E_C_NONEXACT" if science else None)),
        "error": text,
        "traceback": trace,
        "traceback_path": str(trace_path),
        "failure_index": failure_index,
        "failure_index_path": None if not isinstance(failure_index, dict) else failure_index.get("path"),
        "exact_comparison_paths": exact_paths,
    }
    sg.write(work / "ENTRY_FAILURE.json", payload)
    return payload


def _save_capture(store: CaptureStore, root: Path, *, provenance: dict) -> dict:
    root = root.resolve()
    require(not root.exists(), "S46E_C_NON_OVERWRITING_CAPTURE_ROOT")
    required_provenance = ("execution_json_sha256", "input_manifest_sha256", "source_identity_digest", "source_binding", "case_source")
    require(isinstance(provenance, dict) and all(key in provenance for key in required_provenance), "S46E_C_CAPTURE_PROVENANCE_REQUIRED")
    root.mkdir(parents=True)
    rows = store.require_complete()
    metadata = store.metadata()
    require(metadata["selected_interval_indices"] == list(CAPTURE_INTERVALS) and metadata["expected_total_calls"] == CAPTURE_TOTAL_CALLS and metadata["interval_sequence"] == list(range(CAPTURE_TOTAL_CALLS)), "S46E_C_CAPTURE_SEQUENCE_CONTRACT")
    saved_records = {}
    array_names = ("E_input", "phase", "alpha", "dn_gas", "output")
    for interval, row in sorted(rows.items()):
        saved = {key: value for key, value in row.items() if key not in array_names}
        saved["arrays"] = {}
        for name in array_names:
            value = row.get(name)
            if value is None:
                saved["arrays"][name] = None
                continue
            filename = f"interval_{int(interval):02d}_{name}.npy"
            with (root / filename).open("xb") as stream:
                np.save(stream, value, allow_pickle=False)
            saved["arrays"][name] = filename
        saved_records[str(interval)] = saved
    files = [
        {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "classification": "external",
            "hash_scope": "raw_bytes",
            "size_bytes": int(path.stat().st_size),
            "raw_sha256": pv.raw_sha256_file(path),
        }
        for path in sorted(root.glob("*.npy"))
    ]
    package = {
        "schema": "filament.cpu_s4_6e.c_capture.package.v1",
        "metadata": metadata,
        "records": saved_records,
        "files": files,
        "provenance": copy.deepcopy(provenance),
        "array_storage": "NPY_ALLOW_PICKLE_FALSE",
        "live_original_array_refs_retained": False,
        "captured_owned_arrays": True,
        "runtime_store_cleanup_required": True,
    }
    sg.write(root / "CAPTURE.json", package)
    return package


def _capture_runtime_binding(binding: dict, *, role: str, mode: str) -> dict:
    """Extract and validate the factory's source-bound runtime wrapper."""

    require(isinstance(binding, dict), "S46E_C_CAPTURE_RUNTIME_BINDING_REQUIRED")
    runtime_binding = binding.get("runtime_wrapper")
    require(isinstance(runtime_binding, dict), "S46E_C_CAPTURE_RUNTIME_WRAPPER_REQUIRED")
    require(str(role).upper() == "R" and runtime_binding.get("c_role") == "R", "S46E_C_CAPTURE_RUNTIME_ROLE")
    require(runtime_binding.get("c_mode") == mode, "S46E_C_CAPTURE_RUNTIME_MODE")
    require(isinstance(runtime_binding.get("c_candidate_schema"), str) and runtime_binding["c_candidate_schema"], "S46E_C_CAPTURE_RUNTIME_SCHEMA")
    source_record = runtime_binding.get("c_source_binding")
    require(isinstance(source_record, dict), "S46E_C_CAPTURE_RUNTIME_SOURCE_RECORD")
    required_source_fields = (
        "path",
        "relative_path",
        "classification",
        "hash_scope",
        "canonical_lf_sha256",
        "source_blob",
        "function",
        "function_first_line",
        "raw_sha256",
    )
    require(all(source_record.get(field) not in (None, "") for field in required_source_fields), "S46E_C_CAPTURE_RUNTIME_SOURCE_RECORD")
    return runtime_binding


def _load_capture(
    root: Path,
    *,
    expected_execution_sha: str | None = None,
    expected_input_manifest_sha: str | None = None,
    expected_source_identity_digest: str | None = None,
    expected_shape: tuple[int, ...] | None = None,
) -> dict:
    root = Path(root).resolve(strict=True)
    package = read(root / "CAPTURE.json")
    require(package.get("schema") == "filament.cpu_s4_6e.c_capture.package.v1", "S46E_C_CAPTURE_PACKAGE_SCHEMA")
    metadata = package.get("metadata")
    records = package.get("records")
    files = package.get("files")
    provenance = package.get("provenance")
    require(isinstance(metadata, dict) and isinstance(records, dict) and isinstance(files, list) and isinstance(provenance, dict), "S46E_C_CAPTURE_PACKAGE_RECORDS")
    source_binding = provenance.get("source_binding")
    case_source = provenance.get("case_source")
    require(
        isinstance(source_binding, dict)
        and isinstance(case_source, dict)
        and isinstance(provenance.get("source_identity_digest"), str)
        and bool(provenance.get("source_identity_digest")),
        "S46E_C_CAPTURE_SOURCE_PROVENANCE",
    )
    require(
        isinstance(provenance.get("execution_json_sha256"), str)
        and bool(provenance.get("execution_json_sha256"))
        and isinstance(provenance.get("input_manifest_sha256"), str)
        and bool(provenance.get("input_manifest_sha256")),
        "S46E_C_CAPTURE_EXECUTION_MANIFEST_PROVENANCE",
    )
    require(source_binding.get("role") == "R", "S46E_C_CAPTURE_SOURCE_ROLE")
    source_record = source_binding.get("source")
    require(isinstance(source_record, dict) and bool(source_record.get("canonical_lf_sha256")), "S46E_C_CAPTURE_SOURCE_SHA")
    require(
        case_source.get("role") == "R"
        and int(case_source.get("intervals", -1)) == 8
        and case_source.get("capture") is True,
        "S46E_C_CAPTURE_CASE_SOURCE",
    )
    if expected_shape is not None:
        require(case_source.get("shape") == list(expected_shape), "S46E_C_CAPTURE_CASE_SHAPE")
    require(package.get("live_original_array_refs_retained") is False and package.get("captured_owned_arrays") is True, "S46E_C_CAPTURE_RETENTION_METADATA")
    if expected_execution_sha is not None:
        require(provenance.get("execution_json_sha256") == expected_execution_sha, "S46E_C_CAPTURE_EXECUTION_BINDING")
    if expected_input_manifest_sha is not None:
        require(provenance.get("input_manifest_sha256") == expected_input_manifest_sha, "S46E_C_CAPTURE_INPUT_MANIFEST_BINDING")
    if expected_source_identity_digest is not None:
        require(provenance.get("source_identity_digest") == expected_source_identity_digest, "S46E_C_CAPTURE_SOURCE_IDENTITY_BINDING")
    for record in files:
        require(isinstance(record, dict) and record.get("classification") == "external" and record.get("hash_scope") == "raw_bytes", "S46E_C_CAPTURE_FILE_PROVENANCE")
        path = (root / str(record.get("path"))).resolve()
        require(path.is_relative_to(root) and path.is_file(), "S46E_C_CAPTURE_FILE_PATH")
        require(int(path.stat().st_size) == int(record.get("size_bytes", -1)) and pv.raw_sha256_file(path) == record.get("raw_sha256"), "S46E_C_CAPTURE_FILE_HASH")
    restored = {}
    for key, row in records.items():
        row = copy.deepcopy(row)
        arrays = row.pop("arrays")
        require(isinstance(arrays, dict), "S46E_C_CAPTURE_ARRAY_INDEX")
        for name, filename in arrays.items():
            if filename is None:
                row[name] = None
                continue
            path = (root / filename).resolve()
            require(path.is_relative_to(root) and path.is_file(), "S46E_C_CAPTURE_ARRAY_PATH")
            row[name] = np.array(np.load(path, allow_pickle=False), copy=True, order="C")
        restored[int(key)] = row
    return {"metadata": metadata, "records": restored, "provenance": provenance}


def _mechanism(args) -> dict:
    require(args.allow_p24, "S46E_C_P24_AUTHORIZATION_FLAG_REQUIRED")
    require(not args.local_engineering, "S46E_C_P24_HPC_ONLY")
    verify_execution(args.report, local_engineering=False, shape=SHAPE, intervals=8, mode="PURE_OFF_SMALLGRID", role="R", capture=True)
    execution_sha = pv.raw_sha256_file(Path(args.report) / "EXECUTION.json")
    input_manifest_sha = pv.raw_sha256_file(Path(args.report) / "BENCHMARK_INPUT_MANIFEST.json")
    bundle = _load_capture(
        args.capture_root,
        expected_execution_sha=execution_sha,
        expected_input_manifest_sha=input_manifest_sha,
        expected_source_identity_digest=_source_identity_digest(),
        expected_shape=MICRO_SHAPE if args.local_engineering else SHAPE,
    )
    g1 = bundle
    g2 = copy.deepcopy(bundle)
    g2["metadata"]["group"] = "G2"
    for row in g2["records"].values():
        row["group"] = "G2"
    result = run_local_mechanism({"G1": g1, "G2": g2})
    if result.get("status") == "SCIENTIFIC_NONEXACT":
        sg.write(args.output, result)
        return result
    require(result.get("formal_calls") == 90 and result.get("warmup_calls") == 36, "S46E_C_MECHANISM_CALL_COUNT")
    if result.get("mechanism_gate") is not True:
        result["status"] = "SCIENTIFIC_INSUFFICIENT_P24_GAIN"
        result["classification"] = "SCIENTIFIC_INSUFFICIENT_GAIN"
    sg.write(args.output, result)
    return result


def _qualify_boundary(args: Any, work: Path, result: dict, built: Any, support_module: Any) -> str | None:
    """Validate and, for the two-interval gate, compare the disk boundary."""

    boundary = work / "formal" / "BOUNDARY_RECEIPT.json"
    if int(args.intervals) != 2:
        result["boundary_path"] = None
        return None
    require(
        boundary.is_file()
        and result["boundary"]["retained_live_array_refs"] == 0
        and built[5].transparency().get("retired") is True,
        "S46E_C_RETIRED_CAPTURE_DISK_BOUNDARY",
    )
    exact = support_module("g2_hpc_exact")
    metadata = exact.inspect_boundary_metadata(boundary, require_members=True, header_only=True)
    require(metadata["status"] == "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY", "S46E_ORIGINAL_BOUNDARY_JSON")
    comparison = None
    reference_boundary = getattr(args, "reference_boundary", None)
    if reference_boundary:
        comparison = exact.compare_current_boundary_pair(
            Path(reference_boundary), boundary, work / "formal" / "BOUNDARY_EXACT.jsonl"
        )
        require(
            comparison["status"] == "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT",
            "S46E_ORIGINAL_108_EXACT_4_TIMING_CLASSIFICATION",
        )
        result.setdefault("comparisons", {})["boundary"] = comparison
    sg.write(
        work / "DISK_BOUNDARY_ROUNDTRIP.json",
        dict(
            path=str(boundary),
            collector_retired=True,
            live_array_references=0,
            metadata=metadata,
            comparison=comparison,
            both_comparison_inputs="ACTUAL_PATHS_AFTER_COLLECTOR_AND_VIEW_CLOSE",
        ),
    )
    result["boundary_path"] = str(boundary)
    return str(boundary)


def run(args) -> dict:
    from .c_candidate import build_role, verify_bindings
    from .measurement import run_window
    from tools.cpu_s4_5b_steady_cost.adapter import make_recorder_pair
    from tools.cpu_s4_checkpoint_support import support_module

    role = str(args.role).upper()
    mode = "QUALIFY_BOUNDARY" if args.intervals == 2 else "PURE_OFF_SMALLGRID"
    capture_store = None
    if args.capture:
        capture_store = CaptureStore()
        capture_store.begin("G1", source_role="R")
    application_started = time.perf_counter()
    pair = make_recorder_pair(False)
    timer = pair.b
    application_token = timer.begin("APPLICATION", common=True, context=timer.context(0, occurrence_class="WINDOW_ONCE", source_site="c_entry.run:before-source-and-factory"))
    built = None
    work = Path(args.work).resolve()
    try:
        with timer.scope("SOURCE_FACTORY", common=True):
            with timer.scope("SOURCE_VALIDATION", common=True):
                binding = verify_execution(args.report, local_engineering=args.local_engineering, shape=args.shape, intervals=args.intervals, mode=mode, role=role, capture=args.capture)
                input_domain(read(args.input_manifest or Path(args.report) / "BENCHMARK_INPUT_MANIFEST.json"), local_engineering=args.local_engineering)
            require(not work.exists(), "S46E_C_NON_OVERWRITING_CASE")
            work.mkdir(parents=True)
            sg.frozen_tools(args.report)
            support_module("g2_hpc_entry")._quiet_runtime()
            with timer.scope("L1_FACTORY", common=True):
                built = build_role(mode, capture_root=work / "BOUNDARY", role=role, local_engineering=args.local_engineering, pool_override=1 if args.local_engineering else None, recorder_pair=pair, capture_store=capture_store)
            initial_cache = built[1].cpu_s4_4a_cache_snapshot()
            initial_stats = initial_cache["stats"]
            require(initial_stats["factor_requests"] == initial_stats["plane_builds"] == initial_stats["cache_bytes"] == 0, "S46E_C_FRESH_EMPTY_LINEAR_CACHE")
        initialized = time.perf_counter()
        request = SimpleNamespace(report=Path(args.report).resolve(), work=work, role="G3_SMALL", reported_role=role, mode=mode, intervals=args.intervals, local_engineering=args.local_engineering, input_manifest=args.input_manifest, reference_output=args.reference_output, reference_boundary=None, reference_stages=args.reference_stages)
        result = run_window(request, built, timer=timer, application_token=application_token, label="formal", intervals=args.intervals, capture=args.intervals == 2, application_started=application_started)
        binding_after = verify_bindings(built[1])
        sg.write(work / "BINDING_AFTER.json", binding_after)
    except BaseException as error:
        if built is not None:
            try:
                built[1].close_c_pool()
            except BaseException as cleanup:
                error.add_note("S46E_C_CLOSE_FAILURE:" + type(cleanup).__name__)
        for recorder in (pair.s45a, pair.b):
            try:
                while recorder._stack:
                    recorder.end(recorder._stack[-1].token, failed=True)
            except BaseException as cleanup:
                recorder.retire_incomplete(type(cleanup).__name__)
        if work.is_dir():
            sg.write(work / "INCOMPLETE_TIMING.json", dict(a=pair.s45a.snapshot(), b=pair.b.snapshot()))
            _write_entry_failure(work, error)
        raise

    try:
        _qualify_boundary(args, work, result, built, support_module)
    except BaseException as error:
        if work.is_dir():
            _write_entry_failure(work, error)
        raise

    nH = 2 * int(args.intervals)
    route = result["route"]
    require(route["nH"] == nH and route["fresh_responses"] == 2 * nH and route["math_dispatches"] == 8 * nH and route["fallback"] == 0, "S46E_C_ORIGINAL_WINDOW_COUNTS")
    require(all(value == nH for value in route["lifecycle"].values()) and not route.get("active_lifecycle_identity"), "S46E_C_RETIRED_CAPABILITIES")
    cache = built[1].cpu_s4_4a_cache_snapshot()
    stats = cache["stats"]
    require(stats["factor_requests"] == 384 * nH and stats["plane_builds"] == 384 and stats["hits"] == 384 * (nH - 1) and stats["key_misses"] == 1, "S46E_C_CACHE_COUNTS")
    require(stats["invalidations"] == stats["evictions"] == stats["failed_requests"] == 0 and cache["events_dropped"] == 0 and cache["max_bytes"] == 268435456 and cache["max_tables"] == 2, "S46E_C_CACHE_POLICY")
    finalized = getattr(built[1], "_cpu_s46e_finalized_c_pool", None)
    require(isinstance(finalized, dict) and finalized["pool"]["closed"] and finalized["pool"]["active_thread_ids"] == [], "S46E_C_POOL_FINALIZED_BEFORE_OUTPUT")
    sg.write(work / "C_POOL.json", finalized)
    sg.write(work / "LINEAR_CACHE_SIDECAR.json", cache)
    reference_integrity = None
    if role == "R" and args.intervals == 2:
        for name in ("final_E", "scientific_Qacc"):
            values = np.load(result["normal_outputs"][name], mmap_mode="r", allow_pickle=False)
            require(np.isfinite(values).all(), "S46E_C_REFERENCE_CRITICAL_NONFINITE:" + name)
            del values
        with np.load(result["normal_outputs"]["final_POST"], allow_pickle=False) as post:
            require(all(np.isfinite(post[name]).all() for name in post.files), "S46E_C_REFERENCE_POST_NONFINITE")
        reference_integrity = dict(critical_field_Qacc_POST_finite=True, historical_two_interval_terminal_exact=False, terminal_basis="CURRENT_SOURCE_BOUND_C_REFERENCE_AWAIT_C2_PAIR", native_layout_and_alias_metadata_recorded=True)
    with np.load(result["normal_outputs"]["diagnostics"], allow_pickle=False) as diagnostic:
        if reference_integrity is not None:
            require("U_z" in diagnostic.files and np.isfinite(diagnostic["U_z"]).all(), "S46E_C_REFERENCE_ENERGY_NONFINITE")
            reference_integrity["energy_finite"] = True
        timings = {name: diagnostic[name].tolist() for name in ("total_walltime_step_s", "linear_walltime_step_s", "ionization_walltime_step_s", "raman_operator_walltime_step_s")}
    require(all(len(values) == args.intervals for values in timings.values()), "S46E_C_ORIGINAL_DIAGNOSTIC_CLOCK_COUNT")
    posts = [row for row in read(work / "formal/PROPAGATION/state/streaming_manifest.json")["rate_events"] if row.get("event") == "POST_COMMITTED"]
    post_values = [row["time_s"] for row in posts]
    end = application_started + result["optical_application_window_s"]
    require(len(post_values) == args.intervals and application_started < post_values[0] < post_values[-1] < end and all(b > a for a, b in zip(post_values, post_values[1:])), "S46E_C_ORIGINAL_POST_CLOCK_DOMAIN")
    gaps = [b - a for a, b in zip(post_values, post_values[1:])]
    pair_snapshot = pair.snapshot()
    raw_b, raw_a = pair.b.snapshot(), pair.s45a.snapshot()
    merged = pair_snapshot["merged"]
    require(not raw_a["enabled"] and not raw_b["enabled"] and not raw_a["records"], "S46E_C_FINE_TIMERS_OFF_BOTH_ROLES")
    require(merged["clock_closure"]["pass"] and pair_snapshot["raw_scope_count"] <= 8192 and pair_snapshot["metadata_reserved_bytes"] <= 67108864, "S46E_C_COMMON_CLOCK_CLOSURE_BUDGET")
    require(not raw_a["incomplete_scopes"] and not raw_b["incomplete_scopes"] and not any(row["failed"] for row in merged["records"]), "S46E_C_COMPLETE_SCALAR_SCOPES")
    require(set(row["label"] for row in merged["records"]) <= {"APPLICATION", "SOURCE_FACTORY", "SOURCE_VALIDATION", "L1_FACTORY", "INTERVAL_BODY"}, "S46E_C_OFF_ONLY_COMMON_SCOPES")
    body = sorted((row for row in raw_b["records"] if row["label"] == "INTERVAL_BODY"), key=lambda row: row["context"]["ordinal"])
    require([row["context"]["interval_index"] for row in body] == list(range(args.intervals)) and all(row["context"]["ordinal_1based"] == i + 1 and row["context"]["attempt"] == 1 for i, row in enumerate(body)), "S46E_C_ORIGINAL_FULL_BODY_ORDINALS")
    body_s = [row["wall_inclusive_ns"] / 1e9 for row in body]
    require(all(math.isfinite(value) and value > 0.0 for value in body_s), "S46E_C_FINITE_BODY_CLOCKS")
    sg.write(work / "RAW_TIMING.json", dict(b=raw_b, a=raw_a, merged=merged))
    sg.write(work / "INTERVAL_CLOCKS.json", dict(rows=body, body_s=body_s, main_denominator="sum body ordinals 2..8 for eight-interval formal windows"))
    sg.write(work / "OPTICAL_PACING_SIDECAR.json", dict(original_interval_timings=timings, POST_COMMITTED=posts, application_started_monotonic_s=application_started, application_finished_monotonic_s=end, adjacent_POST_gaps_s=gaps, definition="Original immutable POST before second linear half; no clock subtraction"))
    if capture_store is not None:
        package_root = Path(args.capture_root) if args.capture_root else work / "CAPTURE"
        input_manifest_path = Path(args.input_manifest) if args.input_manifest else Path(args.report) / "BENCHMARK_INPUT_MANIFEST.json"
        runtime_binding = _capture_runtime_binding(built[3], role=role, mode=mode)
        package = _save_capture(
            capture_store,
            package_root,
            provenance={
                "execution_json_sha256": pv.raw_sha256_file(Path(args.report) / "EXECUTION.json"),
                "input_manifest_sha256": pv.raw_sha256_file(input_manifest_path),
                "source_identity_digest": _source_identity_digest(),
                "source_binding": {
                    "candidate_schema": runtime_binding.get("c_candidate_schema"),
                    "role": runtime_binding.get("c_role"),
                    "mode": runtime_binding.get("c_mode"),
                    "source": runtime_binding.get("c_source_binding"),
                },
                "case_source": {
                    "entry": "tools.cpu_s4_6e_array_propagation.c_entry.run",
                    "role": role,
                    "intervals": int(args.intervals),
                    "shape": list(args.shape),
                    "capture": True,
                    "input_manifest": str(input_manifest_path.resolve()),
                },
            },
        )
        result["capture_package"] = str(package_root)
        result["capture_metadata"] = package["metadata"]
        capture_store.release_owned_arrays()
        del capture_store
    result.update(schema=SCHEMA + ".case.v1", execution_commit=read(Path(args.report) / "EXECUTION.json")["execution_commit"], role=role, intervals=args.intervals, capture=bool(args.capture), execution_domain="S46E_LOCAL_MICRO" if args.local_engineering else "S46E_SMALLGRID", cache=cache, cache_fallback=0, initial_cache=initial_cache, fresh_responses=2 * nH, original_math_calls=8 * nH, boundary_path=result.get("boundary_path"), default="I0", production_qualified=False, review_gate="MANUAL_REVIEW_PENDING", initialization_within_application_s=initialized - application_started, application_service_s=result["optical_application_window_s"], research_service_s=time.perf_counter() - application_started, later_body_s=sum(body_s[1:]), first_body_s=body_s[0], all_body_s=sum(body_s), first_POST_latency_s=post_values[0] - application_started, post_gap_median_s=statistics.median(gaps), clock_domain_pass=True, c_pool=finalized, pointwise_calls=2 * int(args.intervals) if role == "C" else 0, reference_integrity=reference_integrity, profile_enabled=False)
    sg.write(work / "CASE_RECEIPT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--role", choices=("R", "C"))
    parser.add_argument("--intervals", type=int, choices=(2, 8))
    parser.add_argument("--local-engineering", action="store_true")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--mechanism", action="store_true")
    parser.add_argument("--allow-p24", action="store_true")
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--reference-boundary", type=Path)
    parser.add_argument("--reference-stages", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    args = parser.parse_args()
    if args.mechanism:
        require(args.capture_root is not None and args.output is not None, "S46E_C_MECHANISM_PATHS_REQUIRED")
        value = _mechanism(args)
    else:
        require(args.work is not None and args.role is not None and args.intervals is not None, "S46E_C_PROPAGATION_ARGS_REQUIRED")
        args.shape = MICRO_SHAPE if args.local_engineering else SHAPE
        try:
            value = run(args)
        except BaseException as error:
            if args.work is not None and args.work.is_dir() and not (args.work / "ENTRY_FAILURE.json").exists():
                try:
                    _write_entry_failure(args.work, error)
                except BaseException as failure_error:
                    error.add_note("S46E_C_ENTRY_FAILURE_INDEX_WRITE:" + repr(failure_error))
            raise
    print(json.dumps({key: value.get(key) for key in ("status", "role", "intervals", "formal_calls", "warmup_calls", "mechanism_gate") if key in value}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
