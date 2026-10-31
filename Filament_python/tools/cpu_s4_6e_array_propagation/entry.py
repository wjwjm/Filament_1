"""One source-bound L1 or A/P24 window, with local/HPC admission."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_3g3_smallgrid.candidate import unoptimized
from tools.cpu_s4_5b_steady_cost.entry import read, require
from tools.hpc_ops import provenance_v2 as pv

from .protocol import BASE, MICRO_SHAPE, SCIENCE_TREE, SHAPE, TASK, plan


SCHEMA = "filament.cpu_s4_6e.entry.v1"


def admission(*, local_engineering, shape, intervals, mode):
    require((intervals, mode) in ((2, "QUALIFY_BOUNDARY"), (8, "PURE_OFF_SMALLGRID")), "S46E_TWO_QUAL_OR_EIGHT_OFF_ONLY")
    if local_engineering:
        require(not os.environ.get("SLURM_JOB_ID") and tuple(shape) == MICRO_SHAPE, "S46E_LOCAL_NT384_MICRO_ONLY")
    else:
        require(os.environ.get("SLURM_JOB_ID", "").isdigit() and tuple(shape) == SHAPE, "S46E_ACTUAL_SLURM_FIXED_SMALLGRID_ONLY")


def verify_source(report, *, local_engineering):
    unoptimized()
    report = Path(report).resolve(strict=True)
    binding = read(report / "EXECUTION.json")
    require(binding["schema"] == "filament.cpu_s4_6e.execution.v1" and binding["task"] == TASK, "S46E_EXECUTION_SCHEMA")
    require(binding["baseline_commit"] == BASE and binding["science_tree"] == SCIENCE_TREE, "S46E_FROZEN_SOURCE_CHAIN")
    require(read(report / "PLAN.json") == plan(), "S46E_FIXED_FOURTEEN_WINDOW_PROTOCOL")
    require(pv.raw_sha256_file(report / "PLAN.json") == binding["plan_raw_sha256"], "S46E_PRE_DATA_PLAN_IDENTITY")
    repo = Path(__file__).resolve().parents[3]
    if not local_engineering:
        from tools.cpu_s4_checkpoint_support.git_readonly_compat import verify_runtime_query_tool

        execution_plan = read(report / "HPC_EXECUTION_PLAN.json")
        require(pv.raw_sha256_file(report / "HPC_EXECUTION_PLAN.json") == os.environ.get("S46E_HPC_PLAN_SHA256"), "S46E_HPC_PLAN_IDENTITY")
        verify_runtime_query_tool(execution_plan)
        require(pv.raw_sha256_file(report / "EXECUTION.json") == os.environ.get("S46E_EXECUTION_SHA256"), "S46E_BINDING_IDENTITY")
    require(pv.raw_sha256_file(report / "EXECUTION_SOURCE_MANIFEST.json") == binding["source_manifest_raw_sha256"], "S46E_SOURCE_MANIFEST_IDENTITY")
    source = read(report / "EXECUTION_SOURCE_MANIFEST.json")
    require(pv._git(repo, "rev-parse", "HEAD") == binding["execution_commit"], "S46E_COMMITTED_EXECUTION_HEAD")
    pv._ensure_clean(repo)
    for row in source["records"]:
        require(row["classification"] == "tracked_text" and row["hash_scope"] == pv.TRACKED_HASH_SCOPE, "S46E_SOURCE_HASH_SCOPE")
        path = repo / row["path"]
        require(not Path(row["path"]).is_absolute() and path.resolve().is_relative_to(repo), "S46E_SOURCE_CONFINEMENT")
        require(pv._git(repo, "rev-parse", "HEAD:" + row["path"]) == row["git_blob_oid"] and pv.canonical_lf_sha256(path.read_bytes()) == row["canonical_lf_sha256"], "S46E_SOURCE_BLOB_CANONICAL", row["path"])
    require(source["head"] == binding["execution_commit"] and len(source["records"]) >= 252, "S46E_COMPLETE_COMMITTED_SOURCE_CLOSURE")
    require(pv._git(repo, "rev-parse", "HEAD:Filament_python/KHz_filament") == SCIENCE_TREE, "S46E_UNCHANGED_SCIENCE_SUBTREE")
    require(pv.raw_sha256_file(report / "INPUT_CONTRACT.json") == binding["input_contract_raw_sha256"] and pv.raw_sha256_file(report / "REFERENCE_INPUT_RAW_RECORDS.json") == binding["reference_contract_raw_sha256"], "S46E_FROZEN_INPUT_REFERENCE_CONTRACTS")
    return binding


def verify_execution(report, *, local_engineering, shape, intervals, mode, require_node=True):
    admission(local_engineering=local_engineering, shape=shape, intervals=intervals, mode=mode)
    report = Path(report).resolve(strict=True)
    binding = verify_source(report, local_engineering=local_engineering)
    if not local_engineering and require_node:
        gate = read(report / "NODE_GATE.json")
        require(gate["status"] == "PASS_S46E_SMALLGRID_SOURCE_RESOURCE_INPUT_GATES" and gate["job_id"] == os.environ["SLURM_JOB_ID"], "S46E_ACTUAL_NODE_GATE")
    return binding


def input_domain(manifest, *, local_engineering):
    expected = MICRO_SHAPE if local_engineering else SHAPE
    require(manifest.get("shape") == list(expected), "S46E_ACTUAL_MANIFEST_SHAPE_DOMAIN")


def run(args):
    application_started = time.perf_counter()
    require(args.role in ("R", "A"), "S46E_FIXED_ROLES")
    from .candidate import build_role, verify_bindings
    from .measurement import run_window
    from .pool import POINTWISE_LABELS
    from tools.cpu_s4_5b_steady_cost.adapter import make_recorder_pair
    from tools.cpu_s4_checkpoint_support import support_module

    pair = make_recorder_pair(False)
    timer = pair.b
    token = timer.begin("APPLICATION", common=True, context=timer.context(0, occurrence_class="WINDOW_ONCE", source_site="entry.run:before-source-and-factory"))
    built = None
    work = Path(args.work).resolve()
    try:
        with timer.scope("SOURCE_FACTORY", common=True):
            with timer.scope("SOURCE_VALIDATION", common=True):
                binding = verify_execution(args.report, local_engineering=args.local_engineering, shape=args.shape, intervals=args.intervals, mode=args.mode)
                input_domain(read(args.input_manifest or Path(args.report) / "BENCHMARK_INPUT_MANIFEST.json"), local_engineering=args.local_engineering)
            report = Path(args.report).resolve(strict=True)
            require(not work.exists(), "S46E_NON_OVERWRITING_CASE")
            work.mkdir(parents=True)
            sg.frozen_tools(report)
            support_module("g2_hpc_entry")._quiet_runtime()
            with timer.scope("L1_FACTORY", common=True):
                built = build_role(args.mode, capture_root=work / "BOUNDARY", role=args.role, local_engineering=args.local_engineering, pool_override=1 if args.local_engineering else None, recorder_pair=pair)
            require(built[1].cpu_s4_5b_recorder_pair is pair, "S46E_ONE_COMMON_RECORDER_PAIR")
            stats = built[1].cpu_s4_4a_cache_snapshot()["stats"]
            require(stats["factor_requests"] == stats["plane_builds"] == stats["cache_bytes"] == 0, "S46E_FRESH_EMPTY_LINEAR_CACHE")
        initialized = time.perf_counter()
        request = SimpleNamespace(report=report, work=work, role="G3_SMALL", reported_role=args.role, mode=args.mode, intervals=args.intervals, local_engineering=args.local_engineering, input_manifest=args.input_manifest, reference_output=args.reference_output, reference_boundary=None, reference_stages=args.reference_stages)
        result = run_window(request, built, timer=timer, application_token=token, label="formal", intervals=args.intervals, capture=args.mode == "QUALIFY_BOUNDARY", application_started=application_started)
        sg.write(work / "BINDING_AFTER.json", verify_bindings(built[1]))
    except BaseException as error:
        if built is not None:
            try:
                built[1].close_intensity_pool()
            except BaseException as cleanup:
                error.add_note("S46E_CLOSE_FAILURE:" + type(cleanup).__name__)
        for recorder in (pair.s45a, pair.b):
            try:
                while recorder._stack:
                    recorder.end(recorder._stack[-1].token, failed=True)
            except BaseException as cleanup:
                recorder.retire_incomplete(type(cleanup).__name__)
        if work.is_dir():
            sg.write(work / "INCOMPLETE_TIMING.json", dict(a=pair.s45a.snapshot(), b=pair.b.snapshot()))
        raise

    boundary = work / "formal/BOUNDARY_RECEIPT.json"
    if args.mode == "QUALIFY_BOUNDARY":
        require(boundary.is_file() and result["boundary"]["retained_live_array_refs"] == 0 and built[5].transparency().get("retired") is True, "S46E_RETIRED_CAPTURE_DISK_BOUNDARY")
        exact = support_module("g2_hpc_exact")
        metadata = exact.inspect_boundary_metadata(boundary, require_members=True, header_only=True)
        require(metadata["status"] == "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY", "S46E_ORIGINAL_BOUNDARY_JSON")
        if args.reference_boundary:
            comparison = exact.compare_current_boundary_pair(Path(args.reference_boundary), boundary, work / "formal/BOUNDARY_EXACT.jsonl")
            require(comparison["status"] == "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT", "S46E_ORIGINAL_108_EXACT_4_TIMING_CLASSIFICATION")
            result["comparisons"]["boundary"] = comparison
        sg.write(work / "DISK_BOUNDARY_ROUNDTRIP.json", dict(path=str(boundary), collector_retired=True, live_array_references=0, metadata=metadata, comparison=result["comparisons"].get("boundary")))

    nH = 2 * args.intervals
    route = result["route"]
    require(route["nH"] == nH and route["fresh_responses"] == 2 * nH and route["math_dispatches"] == 8 * nH and route["fallback"] == 0, "S46E_ORIGINAL_WINDOW_COUNTS")
    cache = built[1].cpu_s4_4a_cache_snapshot()
    stats = cache["stats"]
    require(stats["factor_requests"] == 384 * nH and stats["plane_builds"] == 384 and stats["hits"] == 384 * (nH - 1) and stats["key_misses"] == 1, "S46E_ORIGINAL_CACHE_WORKLOAD")
    require(stats["invalidations"] == stats["evictions"] == stats["failed_requests"] == 0 and cache["events_dropped"] == 0 and cache["max_bytes"] == 268435456 and cache["max_tables"] == 2, "S46E_ORIGINAL_CACHE_POLICY")
    finalized = getattr(built[1], "_cpu_s46e_finalized_pool", None)
    require(isinstance(finalized, dict) and "pool" in finalized and "activity" in finalized, "S46E_POOL_FINALIZATION_MISSING")
    pool = finalized["pool"]
    activity = finalized["activity"]
    require(pool["closed"] and pool["close_error"] is None, "S46E_JOINED_POOL")
    count = 4 * args.intervals if args.role == "A" else 0
    require(pool["operation_counts"] == {label: count for label in POINTWISE_LABELS} and len(activity) == 3 * count, "S46E_SEPARATE_POINTWISE_COUNTS")
    sg.write(work / "INTENSITY_POOL.json", dict(pool=pool, activity=activity, counts_separate_from_original_math_dispatches=True))
    sg.write(work / "LINEAR_CACHE_SIDECAR.json", cache)

    import numpy as np

    reference_integrity = None
    if args.role == "R" and args.intervals == 2:
        for name in ("final_E", "scientific_Qacc"):
            values = np.load(result["normal_outputs"][name], mmap_mode="r", allow_pickle=False)
            require(np.isfinite(values).all(), "S46E_REFERENCE_CRITICAL_NONFINITE:" + name)
            del values
        with np.load(result["normal_outputs"]["final_POST"], allow_pickle=False) as post:
            require(all(np.isfinite(post[name]).all() for name in post.files), "S46E_REFERENCE_POST_NONFINITE")
        reference_integrity = dict(critical_field_Qacc_POST_finite=True, historical_two_interval_terminal_exact=False, terminal_basis="CURRENT_SOURCE_BOUND_L1_REFERENCE_AWAIT_A2_PAIR", native_layout_and_alias_metadata_recorded=True)
    with np.load(result["normal_outputs"]["diagnostics"], allow_pickle=False) as diagnostic:
        if reference_integrity is not None:
            require("U_z" in diagnostic.files and np.isfinite(diagnostic["U_z"]).all(), "S46E_REFERENCE_ENERGY_NONFINITE")
            reference_integrity["energy_finite"] = True
        timings = {name: diagnostic[name].tolist() for name in ("total_walltime_step_s", "linear_walltime_step_s", "ionization_walltime_step_s", "raman_operator_walltime_step_s")}
    require(all(len(values) == args.intervals for values in timings.values()), "S46E_ORIGINAL_DIAGNOSTIC_CLOCK_COUNT")
    posts = [row for row in read(work / "formal/PROPAGATION/state/streaming_manifest.json")["rate_events"] if row.get("event") == "POST_COMMITTED"]
    values = [row["time_s"] for row in posts]
    end = application_started + result["optical_application_window_s"]
    require(len(values) == args.intervals and application_started < values[0] < values[-1] < end and all(b > a for a, b in zip(values, values[1:])), "S46E_ORIGINAL_POST_CLOCK_DOMAIN")
    gaps = [b - a for a, b in zip(values, values[1:])]
    pair_snapshot = pair.snapshot()
    raw_b, raw_a = pair.b.snapshot(), pair.s45a.snapshot()
    merged = pair_snapshot["merged"]
    require(not raw_a["enabled"] and not raw_b["enabled"] and not raw_a["records"], "S46E_FINE_TIMERS_OFF_BOTH_ROLES")
    require(merged["clock_closure"]["pass"] and pair_snapshot["raw_scope_count"] <= 8192 and pair_snapshot["metadata_reserved_bytes"] <= 67108864, "S46E_COMMON_CLOCK_CLOSURE_BUDGET")
    require(not raw_a["incomplete_scopes"] and not raw_b["incomplete_scopes"] and not any(row["failed"] for row in merged["records"]), "S46E_COMPLETE_SCALAR_SCOPES")
    require(set(row["label"] for row in merged["records"]) <= {"APPLICATION", "SOURCE_FACTORY", "SOURCE_VALIDATION", "L1_FACTORY", "INTERVAL_BODY"}, "S46E_OFF_ONLY_COMMON_SCOPES")
    body = sorted((row for row in raw_b["records"] if row["label"] == "INTERVAL_BODY"), key=lambda row: row["context"]["ordinal"])
    require([row["context"]["interval_index"] for row in body] == list(range(args.intervals)) and all(row["context"]["ordinal_1based"] == i + 1 and row["context"]["attempt"] == 1 for i, row in enumerate(body)), "S46E_ORIGINAL_FULL_BODY_ORDINALS")
    body_s = [row["wall_inclusive_ns"] / 1e9 for row in body]
    require(all(math.isfinite(v) and v > 0 for v in body_s), "S46E_FINITE_BODY_CLOCKS")
    sg.write(work / "RAW_TIMING.json", dict(b=raw_b, a=raw_a, merged=merged))
    sg.write(work / "INTERVAL_CLOCKS.json", dict(rows=body, body_s=body_s, main_denominator="sum body ordinals 2..8 for eight-interval formal windows"))
    sg.write(work / "OPTICAL_PACING_SIDECAR.json", dict(original_interval_timings=timings, POST_COMMITTED=posts, application_started_monotonic_s=application_started, application_finished_monotonic_s=end, adjacent_POST_gaps_s=gaps, definition="Original immutable POST before second linear half; no clock subtraction"))
    result.update(schema=SCHEMA + ".case.v1", status="CURRENT_L1_REFERENCE_GENERATED" if not args.reference_output else "PASS_S46E_TERMINAL_NATIVE_EXACT", execution_commit=binding["execution_commit"], execution_domain="S46E_LOCAL_MICRO" if args.local_engineering else "S46E_SMALLGRID", application_service_s=result["optical_application_window_s"], initialization_within_application_s=initialized - application_started, research_service_s=time.perf_counter() - application_started, later_body_s=sum(body_s[1:]), first_body_s=body_s[0], all_body_s=sum(body_s), first_POST_latency_s=values[0] - application_started, post_gap_median_s=statistics.median(gaps), clock_domain_pass=True, intensity_pool=pool, cache=cache, cache_fallback=0, L1_executed=True, profile_enabled=False, boundary_path=str(boundary) if args.mode == "QUALIFY_BOUNDARY" else None, pointwise_calls=3 * count, original_math_calls=8 * nH, default="I0", production_qualified=False, review_gate="MANUAL_REVIEW_PENDING")
    result["reference_integrity"] = reference_integrity
    sg.write(work / "CASE_RECEIPT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--role", choices=("R", "A"), required=True)
    parser.add_argument("--intervals", type=int, choices=(2, 8), required=True)
    parser.add_argument("--local-engineering", action="store_true")
    for name in ("reference-output", "reference-boundary", "reference-stages", "input-manifest"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    args.shape = MICRO_SHAPE if args.local_engineering else SHAPE
    args.mode = "QUALIFY_BOUNDARY" if args.intervals == 2 else "PURE_OFF_SMALLGRID"
    try:
        result = run(args)
    except BaseException as error:
        import traceback

        text = repr(error)
        science = any(tag in text for tag in ("STOP_FIRST_SCIENCE_NONEXACT", "STOP_FIRST_STAGE_NONEXACT", "S46E_ORIGINAL_108_EXACT_4_TIMING_CLASSIFICATION"))
        if args.work.is_dir():
            sg.write(args.work / "ENTRY_FAILURE.json", dict(classification="SCIENTIFIC_NONEXACT" if science else "ENGINEERING_OR_RESOURCE_FAILURE", error=text, traceback=traceback.format_exc()))
        raise
    print(json.dumps({key: result[key] for key in ("status", "role", "intervals", "application_service_s")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
