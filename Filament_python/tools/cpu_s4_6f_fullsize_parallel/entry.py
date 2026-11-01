"""One bounded CPU-S4-6F reference or semantic-candidate window."""

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
from tools.cpu_s4_3g3_smallgrid.candidate import unoptimized
from tools.cpu_s4_5b_steady_cost.entry import read, require
from tools.cpu_s4_checkpoint_support import support_module
from tools.hpc_ops import provenance_v2 as pv

from .measurement import run_window
from .protocol import BASE, MICRO_SHAPE, SCIENCE_TREE, SHAPE, TASK, candidate_spec, plan
from .role_spec import build_role, pool_contract, verify_role

SCHEMA = "filament.cpu_s4_6f.entry.v1"
FULL_GATE_STATUS = "PASS_S46F_FULLSIZE_SOURCE_RESOURCE_INPUT_GATES"


def admission(*, candidate: str, local_engineering: bool, shape: tuple[int, int, int], intervals: int, mode: str) -> None:
    require((int(intervals), str(mode)) in ((2, "QUALIFY_BOUNDARY"), (8, "PURE_OFF_FULLSIZE")), "S46F_TWO_QUAL_OR_EIGHT_OFF_ONLY")
    if local_engineering:
        require(not os.environ.get("SLURM_JOB_ID") and tuple(shape) == MICRO_SHAPE, "S46F_LOCAL_MICRO_ONLY")
    else:
        if candidate_spec(candidate)["deferred"]:
            raise RuntimeError("S46F_C_FULL_MECHANISM_PHASE2_NOT_IMPLEMENTED")
        require(os.environ.get("SLURM_JOB_ID", "").isdigit() and tuple(shape) == SHAPE and int(intervals) in (2, 8), "S46F_HPC_FIXED_FULLSIZE_DOMAIN")


def _read_plan(report: Path, candidate: str) -> dict[str, Any]:
    value = read(report / "PLAN.json")
    require(value == plan(candidate), "S46F_FIXED_CANDIDATE_PLAN")
    return value


def _is_external_absolute(value: Any) -> bool:
    text = str(value)
    return text.startswith("/") or Path(text).is_absolute()


def _restore_original_cpu_diagnostic_profile(ctx: dict[str, Any]) -> dict[str, Any]:
    """Restore the accepted CPU BK-NEE diagnostic setting after 6A loading."""

    prop = ctx["prop"]
    before = bool(getattr(prop, "diag_bk_nee_profile", False))
    prop.diag_bk_nee_profile = True
    return dict(
        field="prop.diag_bk_nee_profile",
        loader_value=before,
        accepted_original_cpu_value=True,
        restored=True,
        acquisition_only=True,
        mathematical_expressions_changed=0,
    )


def verify_source(report: str | Path, *, candidate: str, local_engineering: bool, shape: tuple[int, int, int], intervals: int, mode: str, require_node: bool = True) -> dict[str, Any]:
    unoptimized()
    report = Path(report).resolve(strict=True)
    spec = candidate_spec(candidate)
    binding = read(report / "EXECUTION.json")
    require(binding.get("schema") == "filament.cpu_s4_6f.execution.v1" and binding.get("task") == TASK, "S46F_EXECUTION_SCHEMA")
    require(binding.get("candidate") == spec["semantic_id"] and binding.get("old_role") == spec["old_role"], "S46F_SEMANTIC_CANDIDATE_BINDING")
    require(binding.get("baseline_commit") == BASE and binding.get("science_tree") == SCIENCE_TREE, "S46F_FROZEN_SOURCE_CHAIN")
    _read_plan(report, candidate)
    require(pv.raw_sha256_file(report / "PLAN.json") == binding.get("plan_raw_sha256"), "S46F_PLAN_IDENTITY")
    manifest_path = report / "EXECUTION_SOURCE_MANIFEST.json"
    require(manifest_path.is_file() and pv.raw_sha256_file(manifest_path) == binding.get("source_manifest_raw_sha256"), "S46F_SOURCE_MANIFEST_IDENTITY")
    execution_plan: dict[str, Any] = {}
    if local_engineering:
        require(tuple(shape) == MICRO_SHAPE, "S46F_LOCAL_SHAPE")
    else:
        from tools.cpu_s4_checkpoint_support.git_readonly_compat import verify_runtime_query_tool

        execution_plan_path = report / "HPC_EXECUTION_PLAN.json"
        require(execution_plan_path.is_file(), "S46F_HPC_EXECUTION_PLAN_MISSING")
        require(pv.raw_sha256_file(execution_plan_path) == os.environ.get("S46F_HPC_PLAN_SHA256"), "S46F_HPC_PLAN_IDENTITY")
        execution_plan = read(execution_plan_path)
        verify_runtime_query_tool(execution_plan)
        require(pv.raw_sha256_file(report / "EXECUTION.json") == os.environ.get("S46F_EXECUTION_SHA256"), "S46F_EXECUTION_IDENTITY")
    repo = Path(__file__).resolve().parents[3]
    require(pv._git(repo, "rev-parse", "HEAD") == binding.get("execution_commit"), "S46F_COMMITTED_EXECUTION_HEAD")
    pv._ensure_clean(repo)
    source = read(manifest_path)
    for row in source.get("records", ()):  # PV2 tracked-text identity is checked per window.
        require(row.get("classification") == "tracked_text" and row.get("hash_scope") == pv.TRACKED_HASH_SCOPE, "S46F_SOURCE_HASH_SCOPE")
        path = repo / row["path"]
        require(not Path(row["path"]).is_absolute() and path.resolve().is_relative_to(repo), "S46F_SOURCE_CONFINEMENT")
        require(pv._git(repo, "rev-parse", "HEAD:" + row["path"]) == row["git_blob_oid"] and pv.canonical_lf_sha256(path.read_bytes()) == row["canonical_lf_sha256"], "S46F_SOURCE_BLOB_CANONICAL")
    require(source.get("head") == binding.get("execution_commit"), "S46F_SOURCE_HEAD")
    require(pv._git(repo, "rev-parse", "HEAD:Filament_python/KHz_filament") == SCIENCE_TREE, "S46F_UNCHANGED_SCIENCE_SUBTREE")
    input_contract_path = report / "INPUT_CONTRACT.json"
    reference_contract_path = report / "REFERENCE_INPUT_RAW_RECORDS.json"
    full_input_manifest_path = report / "FULL_INPUT_MANIFEST.json"
    for path in (input_contract_path, reference_contract_path, full_input_manifest_path):
        require(path.is_file(), "S46F_INPUT_REFERENCE_CONTRACT_MISSING:" + path.name)
    input_contract = read(input_contract_path)
    reference_contract = read(reference_contract_path)
    full_input_manifest = read(full_input_manifest_path)
    require(pv.raw_sha256_file(input_contract_path) == binding.get("input_contract_raw_sha256"), "S46F_INPUT_CONTRACT_IDENTITY")
    require(pv.raw_sha256_file(reference_contract_path) == binding.get("reference_contract_raw_sha256"), "S46F_REFERENCE_CONTRACT_IDENTITY")
    require(pv.raw_sha256_file(full_input_manifest_path) == binding.get("full_input_manifest_raw_sha256"), "S46F_FULL_INPUT_MANIFEST_IDENTITY")
    require(input_contract.get("schema") == "filament.cpu_s4_6f.input_contract.v1" and input_contract.get("classification") == "external" and input_contract.get("hash_scope") == "raw_bytes", "S46F_INPUT_CONTRACT_SCHEMA")
    require(input_contract.get("shape") == list(SHAPE) and input_contract.get("dtype") == "complex128" and input_contract.get("native_layout") == "C", "S46F_INPUT_CONTRACT_DOMAIN")
    require(full_input_manifest.get("schema") == "filament.cpu_s4_6f.full_input_manifest.v1" and full_input_manifest.get("classification") == "external" and full_input_manifest.get("hash_scope") == "raw_bytes", "S46F_FULL_INPUT_MANIFEST_SCHEMA")
    require(full_input_manifest.get("shape") == list(SHAPE) and full_input_manifest.get("dtype") == "complex128" and full_input_manifest.get("native_layout") == "C", "S46F_FULL_INPUT_MANIFEST_DOMAIN")
    require(full_input_manifest.get("fingerprints") == input_contract.get("expected_input_fingerprints"), "S46F_FULL_INPUT_FINGERPRINT_BINDING")
    require(full_input_manifest.get("canonical_manifest_raw_sha256") == input_contract.get("canonical_manifest_raw_sha256") == binding.get("canonical_manifest_raw_sha256"), "S46F_CANONICAL_MANIFEST_BINDING")
    require(reference_contract.get("schema") == "filament.cpu_s4_5a.reference_files.v1" and reference_contract.get("hash_scope") == "raw_bytes", "S46F_REFERENCE_CONTRACT_SCHEMA")
    reference = reference_contract.get("reference", {})
    require(reference.get("shape") == list(SHAPE) and reference.get("intervals") == 8 and reference.get("role") == "L1_LINEAR_FACTOR_CACHE", "S46F_REFERENCE_CONTRACT_DOMAIN")
    for key in ("output", "boundary_path", "stages"):
        require(_is_external_absolute(reference.get(key)), "S46F_REFERENCE_PATH_BINDING:" + key)
    require(binding.get("reference_shape") == list(SHAPE) and binding.get("reference_intervals") == 8 and binding.get("reference_role") == "L1_LINEAR_FACTOR_CACHE", "S46F_REFERENCE_BINDING")
    if not local_engineering and spec["deferred"]:
        # Phase-2 C full mechanism validation is not implemented in this
        # entry. A copied file is not an admission receipt.
        raise RuntimeError("S46F_C_FULL_MECHANISM_PHASE2_NOT_IMPLEMENTED")
    if not local_engineering and require_node:
        gate = read(report / "NODE_GATE.json")
        require(gate.get("status") == FULL_GATE_STATUS and gate.get("job_id") == os.environ["SLURM_JOB_ID"], "S46F_ACTUAL_NODE_GATE")
    # The frozen 6A loader expects the full-input fields from the execution
    # plan, while source identity remains owned by EXECUTION.json.
    merged = dict(binding)
    merged.update(execution_plan)
    merged.setdefault("execution_commit", binding.get("execution_commit"))
    merged.setdefault("science_tree", SCIENCE_TREE)
    return merged


def _canonical_window(report: Path, bundle: Path, work: Path):
    """Use the frozen 6A canonical bundle and its sealed-axis prepare hook."""

    from tools.cpu_s4_6a.entry import prepare_function
    from tools.cpu_s4_6a.inputs import context as canonical_context

    ctx = canonical_context(bundle, verify_raw=True)
    diagnostic_profile = _restore_original_cpu_diagnostic_profile(ctx)
    from tools.cpu_p1r import cpu_p1r_optical as original_cpu_helper

    original_cpu_source = Path(original_cpu_helper.__file__).resolve()
    diagnostic_profile["original_cpu_helper"] = dict(
        module=original_cpu_helper.__name__,
        source_path=str(original_cpu_source),
        source_raw_sha256=pv.raw_sha256_file(original_cpu_source),
        accepted_setting="diag_bk_nee_profile=True",
    )
    cases, exact = sg.frozen_tools(report)
    prepare, prepare_binding = prepare_function(cases.prepare, ctx)
    prepare_binding = dict(prepare_binding, diagnostic_profile=diagnostic_profile)
    namespace = dict(vars(sg))
    namespace.update(
        context=lambda _report, shape=None: ctx,
        SHAPE=SHAPE,
        frozen_tools=lambda _report: (SimpleNamespace(prepare=prepare, finish=cases.finish), exact),
    )
    namespace["_run_window"] = FunctionType(sg._run_window.__code__, namespace, sg._run_window.__name__, sg._run_window.__defaults__, sg._run_window.__closure__)
    namespace["_run_window"].__kwdefaults__ = sg._run_window.__kwdefaults__
    window = FunctionType(sg.run_window.__code__, namespace, sg.run_window.__name__, sg.run_window.__defaults__, sg.run_window.__closure__)
    window.__kwdefaults__ = sg.run_window.__kwdefaults__
    sg.write(work / "PREPARE_ACQUISITION_BINDING.json", prepare_binding)
    sg.write(work / "CANONICAL_INPUT_VIEW.json", dict(shape=list(ctx["source"].shape), fingerprints=sg.fingerprints(ctx), canonical_bundle=str(bundle), manifest=str(bundle / "INPUT_MANIFEST.json"), canonical_manifest_raw_sha256=pv.raw_sha256_file(bundle / "INPUT_MANIFEST.json"), diagnostic_profile=diagnostic_profile))
    return window, ctx


def _canonical_bundle_from_contract(contract: dict[str, Any]) -> Path:
    """Resolve only explicit 6A bundle fields; never infer a science root."""

    manifest_value = contract.get("canonical_root") or contract.get("canonical_input_manifest")
    require(manifest_value is not None, "S46F_CANONICAL_ROOT_FIELD_REQUIRED")
    manifest = Path(manifest_value).resolve()
    if manifest.is_dir():
        bundle = manifest
    elif manifest.is_file() and manifest.name == "INPUT_MANIFEST.json":
        bundle = manifest.parent
    elif manifest.is_file():
        metadata = read(manifest)
        value = metadata.get("bundle_root") or metadata.get("canonical_bundle") or metadata.get("full_input_root")
        require(value is not None, "S46F_CANONICAL_BUNDLE_FIELD_REQUIRED")
        bundle = Path(value).resolve()
    else:
        value = contract.get("full_input_root")
        require(value is not None, "S46F_CANONICAL_BUNDLE_PATH_MISSING")
        bundle = Path(value).resolve()
    require((bundle / "INPUT_MANIFEST.json").is_file(), "S46F_CANONICAL_INPUT_MANIFEST_MISSING")
    expected_hash = contract.get("canonical_manifest_raw_sha256") or contract.get("canonical_input_manifest_raw_sha256")
    if expected_hash is not None:
        require(pv.raw_sha256_file(bundle / "INPUT_MANIFEST.json") == expected_hash, "S46F_CANONICAL_INPUT_MANIFEST_HASH")
    return bundle


def input_domain(manifest: dict[str, Any], *, local_engineering: bool) -> None:
    expected = list(MICRO_SHAPE if local_engineering else SHAPE)
    require(manifest.get("shape") == expected, "S46F_INPUT_MANIFEST_SHAPE_DOMAIN")
    require(manifest.get("dtype") == "complex128" and manifest.get("native_layout") == "C", "S46F_INPUT_MANIFEST_DTYPE_LAYOUT")


def _timer_receipts(pair, work: Path, application_started: float, result: dict[str, Any], intervals: int) -> tuple[list[float], list[dict[str, Any]]]:
    diagnostic_path = result["normal_outputs"].get("diagnostics")
    timings: dict[str, list[Any]] = {}
    if diagnostic_path and Path(diagnostic_path).is_file():
        import numpy as np

        with np.load(diagnostic_path, allow_pickle=False) as diagnostic:
            for name in ("total_walltime_step_s", "linear_walltime_step_s", "ionization_walltime_step_s", "raman_operator_walltime_s", "raman_operator_walltime_step_s"):
                if name in diagnostic.files:
                    timings[name] = diagnostic[name].tolist()
    stream = Path(work / "formal/PROPAGATION/state/streaming_manifest.json")
    posts = [row for row in read(stream)["rate_events"] if row.get("event") == "POST_COMMITTED"] if stream.is_file() else []
    values = [float(row["time_s"]) for row in posts]
    end = application_started + float(result["optical_application_window_s"])
    require(len(values) == intervals and application_started < values[0] < values[-1] < end and all(b > a for a, b in zip(values, values[1:])), "S46F_POST_CLOCK_DOMAIN")
    raw_b, raw_a = pair.b.snapshot(), pair.s45a.snapshot()
    merged = pair.snapshot()["merged"]
    require(not raw_a["enabled"] and not raw_b["enabled"] and not raw_a["records"], "S46F_FINE_TIMERS_OFF")
    require(merged["clock_closure"]["pass"] and not raw_a["incomplete_scopes"] and not raw_b["incomplete_scopes"], "S46F_COMMON_CLOCK_CLOSURE")
    body = sorted((row for row in raw_b["records"] if row["label"] == "INTERVAL_BODY"), key=lambda row: row["context"]["ordinal"])
    require([row["context"]["interval_index"] for row in body] == list(range(intervals)), "S46F_BODY_ORDINALS")
    body_s = [float(row["wall_inclusive_ns"]) / 1e9 for row in body]
    require(all(math.isfinite(v) and v > 0 for v in body_s), "S46F_FINITE_BODY_CLOCKS")
    sg.write(work / "RAW_TIMING.json", dict(a=raw_a, b=raw_b, merged=merged))
    sg.write(work / "INTERVAL_CLOCKS.json", dict(rows=body, body_s=body_s, main_denominator="sum body ordinals 2..8"))
    sg.write(work / "OPTICAL_PACING_SIDECAR.json", dict(original_interval_timings=timings, POST_COMMITTED=posts, application_started_monotonic_s=application_started, application_finished_monotonic_s=end, adjacent_POST_gaps_s=[b - a for a, b in zip(values, values[1:])], first_POST_domain="original application clock"))
    return body_s, posts


def run(args):
    application_started = time.perf_counter()
    candidate = str(args.candidate).upper()
    spec = candidate_spec(candidate)
    role = str(args.role).upper()
    require(role in ("R", "X"), "S46F_ROLE_R_OR_X")
    shape = tuple(args.shape)
    mode = "QUALIFY_BOUNDARY" if int(args.intervals) == 2 else "PURE_OFF_FULLSIZE"
    admission(candidate=candidate, local_engineering=bool(args.local_engineering), shape=shape, intervals=args.intervals, mode=mode)
    from tools.cpu_s4_5b_steady_cost.adapter import make_recorder_pair

    pair = make_recorder_pair(False)
    timer = pair.b
    token = timer.begin("APPLICATION", common=True, context=timer.context(0, occurrence_class="WINDOW_ONCE", source_site="S46F.entry.run:before-source-and-factory"))
    report = Path(args.report).resolve(strict=True)
    work = Path(args.work).resolve()
    built = None
    try:
        with timer.scope("SOURCE_FACTORY", common=True):
            with timer.scope("SOURCE_VALIDATION", common=True):
                binding = verify_source(report, candidate=candidate, local_engineering=bool(args.local_engineering), shape=shape, intervals=args.intervals, mode=mode)
                input_domain(read(args.input_manifest or report / ("BENCHMARK_INPUT_MANIFEST.json" if args.local_engineering else "FULL_INPUT_MANIFEST.json")), local_engineering=bool(args.local_engineering))
            require(not work.exists(), "S46F_NON_OVERWRITING_CASE")
            work.mkdir(parents=True)
            sg.frozen_tools(report)
            support_module("g2_hpc_entry")._quiet_runtime()
            with timer.scope("L1_FACTORY", common=True):
                built = build_role(candidate, mode, role=role, capture_root=work / "BOUNDARY", local_engineering=bool(args.local_engineering), shape=shape, pool_override=1 if args.local_engineering else None, recorder_pair=pair)
            require(getattr(built[1], "cpu_s4_5b_recorder_pair", pair) is pair, "S46F_ONE_COMMON_RECORDER_PAIR")
            stats = built[1].cpu_s4_4a_cache_snapshot()["stats"]
            require(stats["factor_requests"] == stats["plane_builds"] == stats["cache_bytes"] == 0, "S46F_FRESH_EMPTY_LINEAR_CACHE")
            cache = built[1].cpu_s4_4a_cache_snapshot()
            expected_bytes = 256 * 1024 * 1024 if args.local_engineering else 1_610_612_736
            require(cache["max_tables"] == 2 and cache["max_bytes"] == expected_bytes, "S46F_CACHE_CAPACITY")
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
        request = SimpleNamespace(report=report, work=work, role="G3_SMALL", reported_role="R" if role == "R" else spec["old_role"], semantic_candidate=candidate, mode=mode, intervals=args.intervals, local_engineering=args.local_engineering, input_manifest=request_manifest, reference_output=args.reference_output, reference_boundary=args.reference_boundary, reference_stages=args.reference_stages, execution_domain="S46F_LOCAL_MICRO" if args.local_engineering else "S46F_FULLSIZE")
        result = run_window(request, built, timer=timer, application_token=token, window=window, label="formal", intervals=args.intervals, capture=args.intervals == 2, application_started=application_started)
        finalized_after = getattr(built[1], "_cpu_s4_6f_finalized_pool", None)
        require(isinstance(finalized_after, dict) and isinstance(finalized_after.get("pool"), dict), "S46F_POOL_FINALIZATION_FOR_BINDING")
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
                error.add_note("S46F_CLOSE_FAILURE:" + type(cleanup).__name__)
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
        require(boundary.is_file(), "S46F_BOUNDARY_RECEIPT_MISSING")
        require(result.get("boundary", {}).get("retained_live_array_refs") == 0, "S46F_BOUNDARY_LIVE_ARRAY_REFS")
        if built[5] is not None:
            require(built[5].transparency().get("retired") is True, "S46F_BOUNDARY_COLLECTOR_NOT_RETIRED")
        exact = support_module("g2_hpc_exact")
        metadata = exact.inspect_boundary_metadata(boundary, require_members=True, header_only=True)
        require(metadata.get("status") == "PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY", "S46F_BOUNDARY_METADATA")
        sg.write(work / "DISK_BOUNDARY_ROUNDTRIP.json", dict(path=str(boundary), collector_retired=None if built[5] is None else built[5].transparency().get("retired"), live_array_references=0, metadata=metadata, comparison=result.get("comparisons", {}).get("boundary")))

    n_half = 2 * int(args.intervals)
    require(result["route"]["nH"] == n_half and result["route"]["fresh_responses"] == 2 * n_half and result["route"]["math_dispatches"] == 8 * n_half and result["route"]["fallback"] == 0, "S46F_ORIGINAL_WINDOW_COUNTS")
    cache = built[1].cpu_s4_4a_cache_snapshot()
    stats = cache["stats"]
    expected_cache = {2: (1536, 384, 1152), 8: (6144, 384, 5760)}[int(args.intervals)]
    require((stats["factor_requests"], stats["plane_builds"], stats["hits"]) == expected_cache and stats["invalidations"] == stats["evictions"] == stats["failed_requests"] == 0 and cache["events_dropped"] == 0, "S46F_CACHE_WORKLOAD")
    finalized = getattr(built[1], "_cpu_s4_6f_finalized_pool", None)
    require(isinstance(finalized, dict) and "pool" in finalized and "activity" in finalized, "S46F_POOL_FINALIZATION_MISSING")
    sg.write(work / "POINTWISE_POOL.json", dict(semantic_candidate=candidate, pool=finalized["pool"], activity=finalized["activity"], counts_separate_from_original_math_dispatches=True))
    sg.write(work / "LINEAR_CACHE_SIDECAR.json", cache)
    import numpy as np

    reference_integrity = None
    if role == "R" and int(args.intervals) == 2:
        for name in ("final_E", "scientific_Qacc"):
            values = np.load(result["normal_outputs"][name], mmap_mode="r", allow_pickle=False)
            require(np.isfinite(values).all(), "S46F_REFERENCE_CRITICAL_NONFINITE:" + name)
            del values
        with np.load(result["normal_outputs"]["final_POST"], allow_pickle=False) as post:
            require(all(np.isfinite(post[name]).all() for name in post.files), "S46F_REFERENCE_POST_NONFINITE")
        reference_integrity = dict(critical_field_Qacc_POST_finite=True, historical_two_interval_terminal_exact=False, terminal_basis="CURRENT_SOURCE_BOUND_L1_REFERENCE", native_layout_and_alias_metadata_recorded=True)
    body_s, posts = _timer_receipts(pair, work, application_started, result, int(args.intervals))
    raw_b_after = pair.b.snapshot()
    propagation_records = [row for row in raw_b_after["records"] if row["label"] == "PROPAGATION"]
    require(len(propagation_records) == 1, "S46F_PROPAGATION_CLOCK_REQUIRED")
    propagation_start_s = float(propagation_records[0]["start_wall_ns"]) / 1e9
    initialization_total_before_propagation_s = propagation_start_s - application_started
    require(math.isfinite(initialization_total_before_propagation_s) and initialization_total_before_propagation_s >= source_factory_preparation_s, "S46F_INITIALIZATION_CLOCK_ORDER")
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
    result.update(schema=SCHEMA + ".case.v1", status="PASS_S46F_WINDOW_EXACT" if args.reference_output else "CURRENT_S46F_REFERENCE_GENERATED", execution_commit=binding["execution_commit"], execution_domain="S46F_LOCAL_MICRO" if args.local_engineering else "S46F_FULLSIZE", semantic_candidate=candidate, fullsize_new_candidate=candidate, role=role, old_role=spec["old_role"], application_service_s=result["optical_application_window_s"], initialization_within_application_s=source_factory_preparation_s, initialization_within_application_scope="SOURCE_FACTORY_ONLY_LEGACY_LABEL", source_factory_preparation_s=source_factory_preparation_s, canonical_setup_s=canonical_setup_s, initialization_total_before_propagation_s=initialization_total_before_propagation_s, research_service_s=time.perf_counter() - application_started, later_body_s=sum(body_s[1:]), first_body_s=body_s[0], all_body_s=sum(body_s), first_POST_latency_s=posts[0]["time_s"] - application_started, first_POST_diagnosis=dict(clock_domain="application_started_to_POST_COMMITTED", first_POST_includes_initialization=True, initialization_total_before_propagation_s=initialization_total_before_propagation_s, partial_initialization_not_used_as_first_POST=False), post_gap_median_s=statistics.median(gaps), body_s=body_s, clock_domain_pass=True, pointwise_calls=spec["pointwise_labels"] * pointwise_per_label if role == "X" else 0, original_math_calls=8 * n_half, pointwise_pool=finalized, cache=cache, cache_fallback=0, L1_executed=True, profile_enabled=False, bk_nee_profile_enabled=True, boundary_path=str(work / "formal/BOUNDARY_RECEIPT.json") if args.intervals == 2 else None, default="I0", production_qualified=False, review_gate="MANUAL_REVIEW_PENDING")
    result["reference_integrity"] = reference_integrity
    sg.write(work / "CASE_RECEIPT.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--candidate", choices=tuple(sorted(("EQ10_P48", "INTENSITY_P24", "PHASE_ABSORB_P24"))), required=True)
    parser.add_argument("--role", choices=("R", "X"), required=True)
    parser.add_argument("--intervals", type=int, choices=(2, 8), required=True)
    parser.add_argument("--local-engineering", action="store_true")
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--reference-boundary", type=Path)
    parser.add_argument("--reference-stages", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    args = parser.parse_args(argv)
    args.shape = MICRO_SHAPE if args.local_engineering else SHAPE
    try:
        return run(args)
    except BaseException as error:
        if args.work.is_dir() and not (args.work / "ENTRY_FAILURE.json").exists():
            science = any(tag in repr(error) for tag in ("STOP_FIRST_SCIENCE_NONEXACT", "STOP_FIRST_STAGE_NONEXACT"))
            sg.write(args.work / "ENTRY_FAILURE.json", dict(classification="SCIENTIFIC_NONEXACT" if science else "ENGINEERING_OR_RESOURCE_FAILURE", error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
