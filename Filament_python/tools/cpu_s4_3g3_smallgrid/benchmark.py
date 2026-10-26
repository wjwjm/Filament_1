"""One fixed optical SG window; original operators, I/O and exact comparators."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import FunctionType

from .candidate import unoptimized
from .scientific_context import SHAPE, context, data_root, frozen_tools, require


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=str)
        stream.write("\n")


def versions():
    import numpy, scipy, numba, llvmlite, psutil, threadpoolctl
    return dict(python=sys.version, executable=sys.executable, platform=sys.platform,
        optimize=sys.flags.optimize, modules={m.__name__: dict(version=m.__version__, path=m.__file__) for m in (numpy, scipy, numba, llvmlite, psutil, threadpoolctl)})


def fingerprints(ctx):
    from KHz_filament.hr4e5_evidence import sha256_array
    return dict(source=sha256_array(ctx["source"]), PRE={key: sha256_array(value) for key, value in sorted(ctx["pre_fields"].items())})


def input_manifest(report, *, suffix="", shape=SHAPE):
    """Generate directly on the fixed grid, once, and freeze its exact identity."""
    import numpy as np
    from tools.hpc_ops import provenance_v2 as pv
    report = Path(report)
    ctx = context(report, shape=shape)
    target = report / ("BENCHMARK_INPUTS" + suffix)
    target.mkdir(exist_ok=False)
    with (target / "source.npy").open("xb") as stream:
        np.save(stream, ctx["source"], allow_pickle=False)
    with (target / "PRE.npz").open("xb") as stream:
        np.savez(stream, **ctx["pre_fields"])
    records = [dict(path=str(path.relative_to(report)), classification="external", hash_scope="raw_bytes", raw_sha256=pv.raw_sha256_file(path), size_bytes=path.stat().st_size) for path in sorted(target.iterdir())]
    value = dict(schema="filament.g3_sg.inputs.v1", hash_scope="raw_bytes", benchmark_id="OPTICAL_SMALL384_117_129_V1" if tuple(shape) == SHAPE else "MICRO_CONTRACT_ONLY_NOT_PERFORMANCE",
        shape=list(shape), dtype="complex128", native_layout="C", fingerprints=fingerprints(ctx), records=records,
        config_raw_sha256=pv.raw_sha256_file(data_root(report) / "SMALLGRID_CONFIG.json"), schedule=ctx["schedule"].as_metadata(),
        input_kind=ctx["benchmark_input_kind"], acquisition="Each window directly regenerates the same frozen source/PRE values from these config/generator parameters; all generation, hashes and ownership acquisition are inside application time.",
        generator=dict(path=str(Path(context.__code__.co_filename).resolve()), classification="external", hash_scope="raw_bytes", raw_sha256=pv.raw_sha256_file(Path(context.__code__.co_filename))),
        production_fullsize_field_constructed=False, Hydro_called=False)
    write(report / ("BENCHMARK_INPUT_MANIFEST" + suffix + ".json"), value)
    return value


def delta(left, right, key):
    return int(right.get(key, 0)) - int(left.get(key, 0))


def map_delta(left, right):
    return {key: int(right.get(key, 0)) - int(left.get(key, 0)) for key in sorted(set(left) | set(right))}


def route(role, dispatch, before, after, executions, *, local, intervals, expected_halves=None):
    import numba
    expected_mask = 1 if local else 24
    require(numba.get_num_threads() == expected_mask, "ACTUAL_OUTER_MASK_RESTORED")
    expected_workers = dict(temporal_workers=1, spatial_workers=1) if local else dict(temporal_workers=48, spatial_workers=24)
    require(dispatch.report().get("workers") == expected_workers, "ACTUAL_FFT_WORKER_SETTINGS")
    if role == "I0_SMALL":
        return dict(role=role, route="FROZEN_I0", mask_after=expected_mask, workers=expected_workers)
    nH = delta(before, after, "controlled_full_calls")
    require(nH >= 2 * intervals, "REAL_RAMAN_HALVES_REQUIRED")
    if expected_halves is not None:
        require(nH == int(expected_halves), "FROZEN_QUALIFIED_STEP_COUNT")
    roles = map_delta(before.get("controlled_hash_role_counts", {}), after.get("controlled_hash_role_counts", {}))
    operations = map_delta(before.get("operation_counts", {}), after.get("operation_counts", {}))
    routes = map_delta(before.get("route_counts", {}), after.get("route_counts", {}))
    require(routes.get("G_ACTUAL") == nH and routes.get("F_STRICT_FALLBACK", 0) == 0 and routes.get("REJECTED", 0) == 0, "ONLY_ACTUAL_CONTROLLED_ROUTE")
    require(all(roles.get(name) == nH for name in ("Omega_stage1", "Omega_apply", "Omega_stage2")), "THREE_REAL_OMEGA_HASHES")
    require(roles.get("E_stage1", 0) == (nH if role == "G2_SMALL" else 0) and roles.get("E_apply", 0) == 0, "HONEST_E_HASH_ROLE_COUNTS")
    expected_ops = dict(R_PRODUCT=2*nH, R_DERIVATIVE=2*nH, R_RHS=2*nH, H_PREDICTOR=nH, H_CORRECTOR=nH)
    require(operations == expected_ops and len(executions) == 8*nH, "FRESH_TWO_RESPONSE_EIGHT_DISPATCH_CONTRACT")
    workers = 1 if local else 48
    for row in executions:
        require(row.get("active_thread_ids") == list(range(workers)), "REAL_GET_THREAD_ID_WORKERS")
    lifecycle = None
    if role == "G3_SMALL":
        lifecycle = {name: delta(before, after, name) for name in ("lifecycle_identity_created", "lifecycle_identity_validated", "lifecycle_identity_retired")}
        require(all(value == nH for value in lifecycle.values()) and not after.get("active_lifecycle_identity"), "LIFECYCLE_CREATED_VALIDATED_RETIRED")
    return dict(role=role, route="G_ACTUAL", nH=nH, hash_roles=roles, operation_counts=operations, fresh_responses=2*nH,
                math_dispatches=len(executions), actual_numba_kernel_executions=executions, route_deltas=routes,
                lifecycle=lifecycle, mask_after=expected_mask, workers=expected_workers, fallback=0, rejected=0)


def normal_writer(report, intervals):
    from tools.cpu_s4_checkpoint_support import support_module
    old_entry = support_module("g2_hpc_entry")
    original = old_entry._write_normal_outputs
    namespace = dict(original.__globals__)
    namespace["INTERVALS"] = intervals
    return FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)


def run_window(args, built, *, label, intervals, capture, application_started=None):
    state = getattr(built[1], "cpu_s4_3g_state", None)
    try:
        return _run_window(args, built, label=label, intervals=intervals,
                           capture=capture, application_started=application_started)
    finally:
        if state is not None and capture:
            state._g_set_qualification_capture(False)


def _run_window(args, built, *, label, intervals, capture, application_started=None):
    import numpy as np
    from KHz_filament.hr4e5_evidence import sha256_array
    from tools.hpc_ops import provenance_v2 as pv
    report = Path(args.report).resolve()
    work = Path(args.work).resolve() / label
    require(not work.exists(), "NON_OVERWRITING_WINDOW")
    work.mkdir(parents=True)
    prop, dispatch, clock, binding, audit, lease = built
    cases, exact = frozen_tools(report)
    from tools.cpu_s4_checkpoint_support import support_module
    extended = support_module("g2_hpc_exact")
    required_input = read(args.input_manifest or report / "BENCHMARK_INPUT_MANIFEST.json")
    writer = normal_writer(report, intervals)
    state = getattr(dispatch, "cpu_s4_3g_state", None)
    if state is not None and capture:
        state._g_set_qualification_capture(True)
    before = {} if state is None else state.metrics()
    runtime_start = 0 if state is None else len(state.r8_state._runtime.get("executions", []))
    fft_before = copy.deepcopy(dispatch.report().get("transform_counts", {}))
    application_started = time.perf_counter() if application_started is None else application_started
    selected_shape = tuple(required_input["shape"])
    require(args.local_engineering or selected_shape == SHAPE, "HPC_ONLY_FIXED_MAIN_SMALLGRID")
    ctx = context(report, shape=selected_shape)
    ingress = fingerprints(ctx)
    require(ingress == required_input["fingerprints"], "SAME_FROZEN_SMALLGRID_SOURCE_AND_PRE")
    compute_started = time.perf_counter()
    field, kwargs, view = cases.prepare(ctx, work / "PROPAGATION", intervals=intervals)
    require(not np.shares_memory(field, ctx["source"]) and sha256_array(field) == ingress["source"], "FRESH_FIELD_OWNERSHIP_AND_CONTENT")
    current_before = view.current_content_sha256
    try:
        result = prop(field, **kwargs)
        bundle = cases.finish(result, view, work / "PROPAGATION", intervals=intervals)
    finally:
        view.close()
    after_inputs = fingerprints(ctx)
    require(after_inputs == ingress and view.current_content_sha256 == current_before, "SOURCE_PRE_IMMUTABLE")
    after = {} if state is None else state.metrics()
    executions = [] if state is None else copy.deepcopy(state.r8_state._runtime.get("executions", [])[runtime_start:])
    expected_nH = None
    qualified = report / "QUALIFIED_STEP_CONTRACT.json"
    if qualified.is_file() and intervals == 8:
        expected_nH = read(qualified)["nH"]
    actual_route = route(args.role, dispatch, before, after, executions, local=args.local_engineering, intervals=intervals, expected_halves=expected_nH)
    compute_finished = time.perf_counter()
    ordinary = writer(bundle, work / "PROPAGATION", work, role=args.role, raw_diagnostics=result[2])
    application_finished = time.perf_counter()
    selected = None
    if state is not None and capture:
        captures = state._g_take_qualification_capture()
        require(len(captures) == actual_route["nH"], "REAL_ALL_HALF_STAGE_CAPTURE_COUNT")
        require([row["scope"] for row in captures[:2]] == ["pre", "post"], "ACTUAL_FIRST_PRE_POST_STAGES")
        selected = [{key: row[key] for key in ("field", "stage1", "predictor", "stage2")} for row in captures[:2]]
        state._g_set_qualification_capture(False)
        del captures
    native = extended.native_metadata(bundle, pointer=True)
    output = Path(exact.save_tree(work / "OUTPUT", bundle, compressed=True))
    comparisons = {}
    if label == "formal" and args.reference_output:
        comparisons["terminal"] = extended.compare_full_output(args.reference_output, bundle, work / "TERMINAL_EXACT.jsonl")
        require(comparisons["terminal"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS", "STOP_FIRST_SCIENCE_NONEXACT")
    boundary = None if lease is None else lease.result()
    if capture:
        require(boundary is not None and boundary["field_count"] == 112 and boundary["retained_live_array_refs"] == 0, "BOUND_FIRST_112_CAPTURE_NO_ESCAPE")
        write(work / "BOUNDARY_RECEIPT.json", boundary)
        if args.reference_boundary:
            comparisons["boundary"] = extended.compare_current_boundary_pair(args.reference_boundary, boundary, work / "BOUNDARY_EXACT.jsonl")
    stage_output = None
    activity = {}
    if selected is not None:
        activity = {f"half_{i}_{stage}": dict(I_R_max=float(np.max(np.abs(row[stage]["I_R"]))), rhs_max=float(np.max(np.abs(row[stage]["rhs"])))) for i, row in enumerate(selected) for stage in ("stage1", "stage2")}
        require(all(row["I_R_max"] > 0 and row["rhs_max"] > 0 for row in activity.values()), "NONZERO_REAL_RAMAN_RESPONSE_RHS")
        stage_output = Path(exact.save_tree(work / "STAGES", selected, compressed=True))
        if args.reference_stages:
            comparisons["stages"] = exact.compare(Path(args.reference_stages), selected, work / "STAGES_EXACT.jsonl", disk=True)
            require(comparisons["stages"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS", "STOP_FIRST_STAGE_NONEXACT")
        del selected
    require(capture or lease is None, "PURE_OFF_MUST_NOT_CAPTURE")
    raw_index = extended.save_file_index(work / "OUTPUT_RAW_INDEX.json", {"OUTPUT": output, "PROPAGATION": work / "PROPAGATION", "SCIENTIFIC_QACC": ordinary["scientific_Qacc"], "SCIENTIFIC_SCALARS": ordinary["scientific_scalar_diagnostics"]})
    full_confirmation = getattr(args, "execution_domain", "SMALLGRID_BENCHMARK") == "FULLSIZE_CONFIRMATION"
    if full_confirmation:
        require(not args.local_engineering and selected_shape == (384, 351, 301), "EXPLICIT_FULLSIZE_METADATA_SCOPE")
    status = "EXECUTED_WARMUP_NOT_ACCEPTANCE" if label == "warmup" else (("PASS_FULLSIZE_WINDOW_EXACT" if full_confirmation else "PASS_SMALLGRID_WINDOW_EXACT") if args.reference_output else "I0_SMALL_ORACLE_GENERATED")
    receipt = dict(status=status,
        role=getattr(args, "reported_role", args.role), mode=args.mode, label=label, shape=list(selected_shape), intervals=intervals,
        scope="LOCAL_ENGINEERING_ONLY" if args.local_engineering else ("FULLSIZE_ENTRY_EIGHT_ONLY" if full_confirmation else "SMALLGRID_ENGINEERING_ONLY"),
        optical_compute_window_s=compute_finished-compute_started,
        optical_application_window_s=application_finished-application_started,
        necessary_input_hashes=dict(before=ingress, after=after_inputs, current_PRE=current_before),
        output=str(output), normal_outputs=ordinary, native_metadata=native, boundary=boundary,
        stages=None if stage_output is None else str(stage_output), stage_activity=activity,
        comparisons=comparisons, route=actual_route, binding=binding,
        output_raw_index=str(work / "OUTPUT_RAW_INDEX.json"),
        FFT_transform_counts=map_delta(fft_before, dispatch.report().get("transform_counts", {})),
        fullsize_new_candidate="EXECUTED_ENTRY8" if full_confirmation else "NOT_RUN", Hydro_called=False, default="I0", production_qualified=False)
    write(work / "WINDOW_RECEIPT.json", receipt)
    return receipt


def run(args):
    from .adapter import build_role
    from tools.hpc_ops import provenance_v2 as pv
    unoptimized()
    require(args.role in ("I0_SMALL", "G2_SMALL", "G3_SMALL"), "NAMED_SMALLGRID_ROLE")
    require(args.intervals in (1, 2, 8), "BOUNDED_INTERVAL_SET")
    require(not args.local_engineering or not os.environ.get("SLURM_JOB_ID"), "LOCAL_PATH_FORBIDDEN_IN_SLURM")
    if not args.local_engineering:
        require(os.environ.get("SLURM_JOB_ID") and args.intervals == 8, "HPC_ALLOCATION_AND_EIGHT_INTERVALS")
        plan = read(Path(args.report) / "BENCHMARK_PLAN.json")
        require(pv.raw_sha256_file(Path(args.report) / "BENCHMARK_PLAN.json") == os.environ.get("S43SG_PLAN_SHA256"), "EXACT_PREREGISTERED_PLAN")
        for row in plan["code_records"]:
            path = Path(row["path"])
            require(pv.raw_sha256_file(path) == row["raw_sha256"] and path.stat().st_size == row["size_bytes"], "RUN_SOURCE_RAW_IDENTITY")
    started = time.perf_counter()
    frozen_tools(args.report)
    from tools.cpu_s4_checkpoint_support import support_module
    support_module("g2_hpc_entry")._quiet_runtime()
    work = Path(args.work).resolve()
    require(not work.exists(), "NON_OVERWRITING_CASE")
    work.mkdir(parents=True)
    built = build_role(args.role, args.mode, capture_root=work / "BOUNDARY", local_engineering=args.local_engineering, pool_override=1 if args.local_engineering else None)
    initialized = time.perf_counter()
    warm = None
    if args.warmup:
        require(args.mode == "PURE_OFF_SMALLGRID", "WARMUP_ONLY_IN_PURE_OFF_PROCESS")
        warm = run_window(args, built, label="warmup", intervals=1, capture=False)
        state = getattr(built[1], "cpu_s4_3g_state", None)
        require(state is None or (state._g_pending_cap is None and getattr(state._local, "context", None) is None and not state._g_propagation_active), "WARMUP_CONTEXT_CLEARED")
    formal = run_window(args, built, label="formal", intervals=args.intervals, capture=args.mode == "QUALIFY_BOUNDARY")
    formal.update(runtime_initialization_s=initialized-started, research_service_s=time.perf_counter()-started,
                  warmup=None if warm is None else dict(receipt=str(work / "warmup/WINDOW_RECEIPT.json"), optical_application_window_s=warm["optical_application_window_s"]), environment=versions())
    write(work / "CASE_RECEIPT.json", formal)
    return formal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--role", choices=("I0_SMALL", "G2_SMALL", "G3_SMALL"))
    parser.add_argument("--mode", choices=("QUALIFY_BOUNDARY", "PURE_OFF_SMALLGRID"), default="QUALIFY_BOUNDARY")
    parser.add_argument("--intervals", type=int, default=8)
    parser.add_argument("--local-engineering", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--reference-boundary", type=Path)
    parser.add_argument("--reference-stages", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--prepare-inputs", action="store_true")
    args = parser.parse_args()
    unoptimized()
    value = input_manifest(args.report) if args.prepare_inputs else run(args)
    print(json.dumps(dict(status=value.get("status", "INPUTS_FROZEN"), role=value.get("role"), report=str(args.report))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
