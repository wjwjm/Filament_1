"""One original-grid OFF or COST window on the same four-component baseline."""
from __future__ import annotations
import argparse
import json
import math
import os
import statistics
import time
import traceback
from pathlib import Path
from types import FunctionType, SimpleNamespace
from typing import Any

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_checkpoint_support import support_module
from tools.cpu_s4_6f_fullsize_parallel import entry as parent
from tools.cpu_s4_6f_fullsize_parallel.entry import (
    read, require, _canonical_bundle_from_contract, _canonical_window,
    input_domain, _timer_receipts,
)
from .measurement import run_window
from .role_spec import build_role, verify_role
from .protocol import BASE, CANDIDATE, MICRO_SHAPE, SCIENCE_TREE, SHAPE, TASK, candidate_spec, exception_classification, plan

SCHEMA = "filament.cpu_s4_6j.entry.v1"
FULL_GATE_STATUS = "PASS_S46J_FULLSIZE_SOURCE_RESOURCE_INPUT_GATES"


def admission(*, candidate, local_engineering, shape, intervals, mode, **unused):
    candidate_spec(candidate)
    require((intervals, mode) in ((2, "QUALIFY_BOUNDARY"), (8, "PURE_OFF_FULLSIZE")),
            "S46J_TWO_QUAL_OR_EIGHT_OFF_ONLY")
    if local_engineering:
        require(not os.environ.get("SLURM_JOB_ID") and tuple(shape) == MICRO_SHAPE,
                "S46J_LOCAL_MICRO_ONLY")
    else:
        require(os.environ.get("SLURM_JOB_ID", "").isdigit() and tuple(shape) == SHAPE,
                "S46J_HPC_ORIGINAL_GRID_ONLY")


def verify_source(*args, **kwargs):
    namespace = dict(parent.verify_source.__globals__)
    namespace.update(BASE=BASE, TASK=TASK, SCIENCE_TREE=SCIENCE_TREE, SHAPE=SHAPE,
                     MICRO_SHAPE=MICRO_SHAPE, candidate_spec=candidate_spec, plan=plan,
                     FULL_GATE_STATUS=FULL_GATE_STATUS)
    namespace["_read_plan"] = FunctionType(parent._read_plan.__code__, namespace)
    function = FunctionType(parent.verify_source.__code__, namespace,
                            argdefs=parent.verify_source.__defaults__)
    function.__kwdefaults__ = parent.verify_source.__kwdefaults__
    return function(*args, **kwargs)


def run(args):
    application_started = time.perf_counter()
    candidate = str(args.candidate).upper()
    spec = candidate_spec(candidate)
    role = str(args.role).upper()
    require(role in ("R", "X"), "S46J_ROLE_R_OR_X")
    cost_mode = str(args.cost_mode).upper()
    require(cost_mode in ("OFF", "COST"), "S46J_COST_MODE")
    shape = tuple(args.shape)
    mode = "QUALIFY_BOUNDARY" if int(args.intervals) == 2 else "PURE_OFF_FULLSIZE"
    admission(candidate=candidate, local_engineering=bool(args.local_engineering), shape=shape, intervals=args.intervals, mode=mode, allow_c_mechanism=(not bool(args.local_engineering) and spec["deferred"]))
    from tools.cpu_s4_5b_steady_cost.adapter import make_recorder_pair

    pair = make_recorder_pair(False)
    timer = pair.b
    token = timer.begin("APPLICATION", common=True, context=timer.context(0, occurrence_class="WINDOW_ONCE", source_site="S46J.entry.run:before-source-and-factory"))
    report = Path(args.report).resolve(strict=True)
    work = Path(args.work).resolve()
    built = None
    try:
        with timer.scope("SOURCE_FACTORY", common=True):
            with timer.scope("SOURCE_VALIDATION", common=True):
                binding = verify_source(report, candidate=candidate, local_engineering=bool(args.local_engineering), shape=shape, intervals=args.intervals, mode=mode)
                input_domain(read(args.input_manifest or report / ("BENCHMARK_INPUT_MANIFEST.json" if args.local_engineering else "FULL_INPUT_MANIFEST.json")), local_engineering=bool(args.local_engineering))
            require(not work.exists(), "S46J_NON_OVERWRITING_CASE")
            work.mkdir(parents=True)
            sg.frozen_tools(report)
            support_module("g2_hpc_entry")._quiet_runtime()
            with timer.scope("L1_FACTORY", common=True):
                built = build_role(candidate, mode, role=role, capture_root=work / "BOUNDARY", local_engineering=bool(args.local_engineering), shape=shape, pool_override=1 if args.local_engineering else None, recorder_pair=pair, cost_mode=cost_mode)
            require(getattr(built[1], "cpu_s4_5b_recorder_pair", pair) is pair, "S46J_ONE_COMMON_RECORDER_PAIR")
            stats = built[1].cpu_s4_4a_cache_snapshot()["stats"]
            require(stats["factor_requests"] == stats["plane_builds"] == stats["cache_bytes"] == 0, "S46J_FRESH_EMPTY_LINEAR_CACHE")
            cache = built[1].cpu_s4_4a_cache_snapshot()
            expected_bytes = 256 * 1024 * 1024 if args.local_engineering else 1_610_612_736
            require(cache["max_tables"] == 2 and cache["max_bytes"] == expected_bytes, "S46J_CACHE_CAPACITY")
        initialized = time.perf_counter()
        source_factory_preparation_s = initialized - application_started
        canonical_setup_started = initialized
        if args.local_engineering:
            window = sg.run_window
            request_manifest = args.input_manifest
            canonical_setup_s = 0.0
        else:
            input_contract = read(report / "INPUT_CONTRACT.json")
            canonical_bundle = _canonical_bundle_from_contract(input_contract)
            window, _ctx = _canonical_window(report, canonical_bundle, work)
            request_manifest = work / "CANONICAL_INPUT_VIEW.json"
            canonical_setup_s = time.perf_counter() - canonical_setup_started
        request = SimpleNamespace(report=report, work=work, role="G3_SMALL", reported_role="R" if role == "R" else spec["old_role"], semantic_candidate=candidate, mode=mode, intervals=args.intervals, local_engineering=args.local_engineering, input_manifest=request_manifest, reference_output=args.reference_output, reference_boundary=args.reference_boundary, reference_stages=args.reference_stages, execution_domain="S46J_LOCAL_MICRO" if args.local_engineering else "S46J_FULLSIZE")
        result = run_window(request, built, timer=timer, application_token=token, window=window, label="formal", intervals=args.intervals, capture=args.intervals == 2, application_started=application_started)
        finalized_after = getattr(built[1], "_cpu_s4_6f_finalized_pool", None)
        require(isinstance(finalized_after, dict) and isinstance(finalized_after.get("pool"), dict), "S46J_POOL_FINALIZATION_FOR_BINDING")
        binding_after = verify_role(built[1], candidate)
        live_pool = finalized_after["pool"]
        binding_after["pool_liveness"] = dict(
            initial_thread_ids=list(live_pool.get("initial_thread_ids", ())),
            active_thread_ids=list(live_pool.get("active_thread_ids", ())),
            initial_thread_ids_semantics="historical_worker_identity_snapshot",
            active_thread_ids_semantics="live_native_ids_after_close_and_join",
        )
        sg.write(work / "BINDING_AFTER.json", binding_after)
    except BaseException as error:
        if built is not None:
            try:
                from .role_spec import close_role
                close_role(built[1], candidate)
            except BaseException as cleanup:
                error.add_note("S46J_CLOSE_FAILURE:" + type(cleanup).__name__)
        for recorder in (pair.s45a, pair.b):
            try:
                while recorder._stack:
                    recorder.end(recorder._stack[-1].token, failed=True)
            except BaseException as cleanup:
                recorder.retire_incomplete(type(cleanup).__name__)
        if work.is_dir() and not (work / "ENTRY_FAILURE.json").exists():
            sg.write(work / "INCOMPLETE_TIMING.json", dict(a=pair.s45a.snapshot(), b=pair.b.snapshot()))
        raise

    if int(args.intervals) == 2:
        boundary = work / "formal/BOUNDARY_RECEIPT.json"
        require(boundary.is_file(), "S46J_BOUNDARY_RECEIPT_MISSING")
        require(result.get("boundary", {}).get("retained_live_array_refs") == 0, "S46J_BOUNDARY_LIVE_ARRAY_REFS")
        if built[5] is not None:
            require(built[5].transparency().get("retired") is True, "S46J_BOUNDARY_COLLECTOR_NOT_RETIRED")
        exact = support_module("g2_hpc_exact")
        metadata = exact.inspect_boundary_metadata(boundary, require_members=True, header_only=True)
        require(metadata.get("status") == "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY", "S46J_BOUNDARY_METADATA")
        sg.write(work / "DISK_BOUNDARY_ROUNDTRIP.json", dict(path=str(boundary), collector_retired=None if built[5] is None else built[5].transparency().get("retired"), live_array_references=0, metadata=metadata, comparison=result.get("comparisons", {}).get("boundary")))

    n_half = 2 * int(args.intervals)
    require(result["route"]["nH"] == n_half and result["route"]["fresh_responses"] == 2 * n_half and result["route"]["math_dispatches"] == 8 * n_half and result["route"]["fallback"] == 0, "S46J_ORIGINAL_WINDOW_COUNTS")
    cache = built[1].cpu_s4_4a_cache_snapshot()
    stats = cache["stats"]
    expected_cache = {2: (1536, 384, 1152), 8: (6144, 384, 5760)}[int(args.intervals)]
    require((stats["factor_requests"], stats["plane_builds"], stats["hits"]) == expected_cache and stats["invalidations"] == stats["evictions"] == stats["failed_requests"] == 0 and cache["events_dropped"] == 0, "S46J_CACHE_WORKLOAD")
    finalized = getattr(built[1], "_cpu_s4_6f_finalized_pool", None)
    require(isinstance(finalized, dict) and "pool" in finalized and "activity" in finalized, "S46J_POOL_FINALIZATION_MISSING")
    sg.write(work / "POINTWISE_POOL.json", dict(semantic_candidate=candidate, pool=finalized["pool"], activity=finalized["activity"], counts_separate_from_original_math_dispatches=True))
    sg.write(work / "LINEAR_CACHE_SIDECAR.json", cache)
    import numpy as np

    reference_integrity = None
    if role == "R" and int(args.intervals) == 2:
        for name in ("final_E", "scientific_Qacc"):
            values = np.load(result["normal_outputs"][name], mmap_mode="r", allow_pickle=False)
            require(np.isfinite(values).all(), "S46J_REFERENCE_CRITICAL_NONFINITE:" + name)
            del values
        with np.load(result["normal_outputs"]["final_POST"], allow_pickle=False) as post:
            require(all(np.isfinite(post[name]).all() for name in post.files), "S46J_REFERENCE_POST_NONFINITE")
        reference_integrity = dict(critical_field_Qacc_POST_finite=True, historical_two_interval_terminal_exact=False, terminal_basis="CURRENT_SOURCE_BOUND_JOINT_REFERENCE", native_layout_and_alias_metadata_recorded=True)
    body_s, posts = _timer_receipts(pair, work, application_started, result, int(args.intervals))
    raw_b_after = pair.b.snapshot()
    propagation_records = [row for row in raw_b_after["records"] if row["label"] == "PROPAGATION"]
    require(len(propagation_records) == 1, "S46J_PROPAGATION_CLOCK_REQUIRED")
    propagation_start_s = float(propagation_records[0]["start_wall_ns"]) / 1e9
    initialization_total_before_propagation_s = propagation_start_s - application_started
    require(math.isfinite(initialization_total_before_propagation_s) and initialization_total_before_propagation_s >= source_factory_preparation_s, "S46J_INITIALIZATION_CLOCK_ORDER")
    sg.write(work / "INITIALIZATION_CLOCKS.json", dict(
        source_factory_preparation_s=source_factory_preparation_s,
        canonical_setup_s=canonical_setup_s,
        initialization_total_before_propagation_s=initialization_total_before_propagation_s,
        propagation_start_monotonic_s=propagation_start_s,
        application_start_monotonic_s=application_started,
        legacy_initialization_within_application_s=source_factory_preparation_s,
        legacy_initialization_scope="SOURCE_FACTORY_ONLY",
    ))
    post_values = [float(row["time_s"]) for row in posts]
    gaps = [b - a for a, b in zip(post_values, post_values[1:])]
    pointwise_per_label = int(args.intervals) if spec["pointwise_operation"] == "PHASE_ABSORB" else 4 * int(args.intervals)
    result.update(schema=SCHEMA + ".case.v1", status="PASS_S46J_WINDOW_EXACT" if args.reference_output else "CURRENT_S46J_REFERENCE_GENERATED", execution_commit=binding["execution_commit"], execution_domain="S46J_LOCAL_MICRO" if args.local_engineering else "S46J_FULLSIZE", semantic_candidate=candidate, fullsize_new_candidate=candidate, role=role, old_role=spec["old_role"], application_service_s=result["optical_application_window_s"], initialization_within_application_s=source_factory_preparation_s, initialization_within_application_scope="SOURCE_FACTORY_ONLY_LEGACY_LABEL", source_factory_preparation_s=source_factory_preparation_s, canonical_setup_s=canonical_setup_s, initialization_total_before_propagation_s=initialization_total_before_propagation_s, research_service_s=time.perf_counter() - application_started, later_body_s=sum(body_s[1:]), first_body_s=body_s[0], all_body_s=sum(body_s), first_POST_latency_s=posts[0]["time_s"] - application_started, first_POST_diagnosis=dict(clock_domain="application_started_to_POST_COMMITTED", first_POST_includes_initialization=True, initialization_total_before_propagation_s=initialization_total_before_propagation_s, partial_initialization_not_used_as_first_POST=False), post_gap_median_s=statistics.median(gaps), body_s=body_s, clock_domain_pass=True, pointwise_calls=38 * int(args.intervals), original_math_calls=8 * n_half, pointwise_pool=finalized, cache=cache, cache_fallback=0, L1_executed=True, profile_enabled=False, bk_nee_profile_enabled=True, boundary_path=str(work / "formal/BOUNDARY_RECEIPT.json") if args.intervals == 2 else None, default="I0", production_qualified=False, review_gate="MANUAL_REVIEW_PENDING")
    from .cost_analysis import validate_cost_records
    cost_receipt = validate_cost_records(raw_b_after, cost_mode=cost_mode, intervals=int(args.intervals))
    sg.write(work / "RESIDUAL_COST_RECORDS.json", cost_receipt)
    result.update(pool_role="X", cost_mode=cost_mode, residual_cost_records=cost_receipt["cost_record_count"], factory_cost_records=cost_receipt["factory_record_count"],
                  propagation_service_s=propagation_records[0]["wall_inclusive_ns"] / 1e9)
    result["reference_integrity"] = reference_integrity
    sg.write(work / "CASE_RECEIPT.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--candidate", choices=(CANDIDATE,), default=CANDIDATE)
    parser.add_argument("--role", choices=("R", "X"), required=True)
    parser.add_argument("--intervals", type=int, choices=(2, 8), required=True)
    parser.add_argument("--cost-mode", choices=("OFF", "COST"), required=True)
    parser.add_argument("--local-engineering", action="store_true")
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--reference-boundary", type=Path)
    parser.add_argument("--reference-stages", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    args = parser.parse_args(argv)
    args.shape = MICRO_SHAPE if args.local_engineering else SHAPE
    try:
        result = run(args)
        print(json.dumps({name: result[name] for name in (
            "status", "semantic_candidate", "role", "intervals", "application_service_s"
        )}))
        return 0
    except BaseException as error:
        if args.work.is_dir() and not (args.work / "ENTRY_FAILURE.json").exists():
            sg.write(args.work / "ENTRY_FAILURE.json", dict(classification=exception_classification(error), error=repr(error), notes=list(getattr(error, "__notes__", ())), traceback=traceback.format_exc()))
        raise

if __name__ == "__main__":
    raise SystemExit(main())
