"""C full-size capture and offline REF/P1/P24 mechanism entry."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import traceback
from typing import Any

import numpy as np

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_5b_steady_cost.adapter import make_recorder_pair
from tools.cpu_s4_5b_steady_cost.entry import read, require
from tools.hpc_ops import provenance_v2 as pv

from . import entry as full_entry
from .c_fullsize_protocol import CAPTURE_INTERVALS, CAPTURE_TOTAL_CALLS, EXECUTION_SCHEMA, MICRO_SHAPE, RESULT_SCHEMA, SCIENCE_TREE, SHAPE, TASK, plan
from .measurement import run_window
from .role_spec import build_role, close_role, verify_role

from tools.cpu_s4_6e_array_propagation.c_capture import CaptureStore
from tools.cpu_s4_6e_array_propagation.c_entry import (
    _capture_runtime_binding,
    _load_capture,
    _save_capture,
    _source_identity_digest,
)
from tools.cpu_s4_6e_array_propagation import c_mechanism


SCHEMA = "filament.cpu_s4_6f.c_fullsize_entry.v1"


def _validate_full_input_contracts(binding: dict[str, Any], input_contract: dict[str, Any], full_manifest: dict[str, Any], reference_contract: dict[str, Any]) -> None:
    """Check the existing 6F input/reference schemas without loading arrays."""

    require(input_contract.get("schema") == "filament.cpu_s4_6f.input_contract.v1" and input_contract.get("classification") == "external" and input_contract.get("hash_scope") == "raw_bytes", "S46F_C_FULL_INPUT_CONTRACT_SCHEMA")
    require(input_contract.get("shape") == list(SHAPE) and input_contract.get("dtype") == "complex128" and input_contract.get("native_layout") == "C", "S46F_C_FULL_INPUT_CONTRACT_DOMAIN")
    require(full_manifest.get("schema") == "filament.cpu_s4_6f.full_input_manifest.v1" and full_manifest.get("classification") == "external" and full_manifest.get("hash_scope") == "raw_bytes", "S46F_C_FULL_INPUT_MANIFEST_SCHEMA")
    require(full_manifest.get("shape") == list(SHAPE) and full_manifest.get("dtype") == "complex128" and full_manifest.get("native_layout") == "C", "S46F_C_FULL_INPUT_MANIFEST_DOMAIN")
    fingerprints = input_contract.get("expected_input_fingerprints")
    require(isinstance(fingerprints, dict) and full_manifest.get("fingerprints") == fingerprints, "S46F_C_FULL_INPUT_FINGERPRINT_BINDING")
    canonical_hash = input_contract.get("canonical_manifest_raw_sha256")
    require(isinstance(canonical_hash, str) and bool(canonical_hash) and full_manifest.get("canonical_manifest_raw_sha256") == canonical_hash == binding.get("canonical_manifest_raw_sha256"), "S46F_C_FULL_CANONICAL_MANIFEST_BINDING")
    # The accepted 5A contract has no top-level classification; each raw
    # record carries its own external/raw-bytes identity.
    require(reference_contract.get("schema") == "filament.cpu_s4_5a.reference_files.v1" and reference_contract.get("hash_scope") == "raw_bytes", "S46F_C_FULL_REFERENCE_CONTRACT_SCHEMA")
    reference = reference_contract.get("reference", {})
    require(reference.get("shape") == list(SHAPE) and int(reference.get("intervals", -1)) == 8 and reference.get("role") == "L1_LINEAR_FACTOR_CACHE", "S46F_C_FULL_REFERENCE_CONTRACT_DOMAIN")
    for key in ("output", "boundary_path", "stages"):
        require(full_entry._is_external_absolute(reference.get(key)), "S46F_C_FULL_REFERENCE_PATH_BINDING:" + key)
    records = reference_contract.get("records")
    if records is not None:
        require(isinstance(records, list) and records, "S46F_C_FULL_REFERENCE_RECORDS")
        for row in records:
            require(isinstance(row, dict) and row.get("classification") == "external" and row.get("hash_scope") == "raw_bytes" and row.get("path") and row.get("raw_sha256") and int(row.get("size_bytes", -1)) >= 0, "S46F_C_FULL_REFERENCE_RECORD_IDENTITY")
    require(binding.get("reference_shape") == list(SHAPE) and binding.get("reference_intervals") == 8 and binding.get("reference_role") == "L1_LINEAR_FACTOR_CACHE", "S46F_C_FULL_REFERENCE_BINDING")


def _verify_mechanism_source(report: Path, *, local_engineering: bool, shape: tuple[int, ...]) -> dict[str, Any]:
    """Verify the private mechanism report before any factory or array work."""

    binding = read(report / "EXECUTION.json")
    require(binding.get("schema") == EXECUTION_SCHEMA and binding.get("task") == TASK, "S46F_C_FULL_EXECUTION_SCHEMA")
    require(binding.get("science_tree") == SCIENCE_TREE, "S46F_C_FULL_SCIENCE_TREE")
    plan_path = report / "PLAN.json"
    require(read(plan_path) == plan(), "S46F_C_FULL_PLAN_IDENTITY")
    require(binding.get("plan_raw_sha256") == pv.raw_sha256_file(plan_path), "S46F_C_FULL_PLAN_HASH")
    repo = Path(__file__).resolve().parents[3]
    require(pv._git(repo, "rev-parse", "HEAD") == binding.get("execution_commit"), "S46F_C_FULL_COMMITTED_EXECUTION_HEAD")
    pv._ensure_clean(repo)
    source_path = report / "EXECUTION_SOURCE_MANIFEST.json"
    source = read(source_path)
    require(source.get("head") == binding.get("execution_commit"), "S46F_C_FULL_SOURCE_HEAD")
    require(binding.get("source_manifest_raw_sha256") == pv.raw_sha256_file(source_path), "S46F_C_FULL_SOURCE_MANIFEST_HASH")
    for row in source.get("records", ()):
        require(row.get("classification") == "tracked_text" and row.get("hash_scope") == pv.TRACKED_HASH_SCOPE, "S46F_C_FULL_SOURCE_SCOPE", row.get("path"))
        path = (repo / str(row["path"])).resolve(strict=True)
        require(path.is_relative_to(repo.resolve()) and pv._git(repo, "rev-parse", "HEAD:" + row["path"]) == row.get("git_blob_oid") and pv.canonical_lf_sha256(path.read_bytes()) == row.get("canonical_lf_sha256"), "S46F_C_FULL_SOURCE_IDENTITY", row.get("path"))
    require(pv._git(repo, "rev-parse", "HEAD:Filament_python/KHz_filament") == SCIENCE_TREE, "S46F_C_FULL_UNCHANGED_SCIENCE")
    input_contract_path = report / "INPUT_CONTRACT.json"
    full_manifest_path = report / "FULL_INPUT_MANIFEST.json"
    reference_path = report / "REFERENCE_INPUT_RAW_RECORDS.json"
    for path in (input_contract_path, full_manifest_path, reference_path):
        require(path.is_file(), "S46F_C_FULL_INPUT_CONTRACT_MISSING", path.name)
    input_contract = read(input_contract_path)
    full_manifest = read(full_manifest_path)
    reference_contract = read(reference_path)
    require(binding.get("input_contract_raw_sha256") == pv.raw_sha256_file(input_contract_path) and binding.get("full_input_manifest_raw_sha256") == pv.raw_sha256_file(full_manifest_path) and binding.get("reference_contract_raw_sha256") == pv.raw_sha256_file(reference_path), "S46F_C_FULL_INPUT_CONTRACT_HASH")
    _validate_full_input_contracts(binding, input_contract, full_manifest, reference_contract)
    if not local_engineering:
        from tools.cpu_s4_checkpoint_support.git_readonly_compat import verify_runtime_query_tool

        execution_plan_path = report / "HPC_EXECUTION_PLAN.json"
        require(execution_plan_path.is_file(), "S46F_C_FULL_HPC_PLAN_MISSING")
        hpc_plan_hash = os.environ.get("S46F_C_FULL_HPC_PLAN_SHA256") or os.environ.get("S46F_HPC_PLAN_SHA256")
        require(hpc_plan_hash and pv.raw_sha256_file(execution_plan_path) == hpc_plan_hash, "S46F_C_FULL_HPC_PLAN_HASH")
        verify_runtime_query_tool(read(execution_plan_path))
        execution_hash = os.environ.get("S46F_C_FULL_EXECUTION_SHA256") or os.environ.get("S46F_EXECUTION_SHA256")
        require(execution_hash and pv.raw_sha256_file(report / "EXECUTION.json") == execution_hash, "S46F_C_FULL_EXECUTION_HASH")
    if not local_engineering:
        hpc_plan = report / "HPC_EXECUTION_PLAN.json"
        require(hpc_plan.is_file(), "S46F_C_FULL_HPC_PLAN_MISSING")
        plan_hash = os.environ.get("S46F_C_FULL_HPC_PLAN_SHA256") or os.environ.get("S46F_HPC_PLAN_SHA256")
        require(plan_hash and pv.raw_sha256_file(hpc_plan) == plan_hash, "S46F_C_FULL_HPC_PLAN_HASH")
    return binding


def _write_failure(work: Path, error: BaseException) -> None:
    if not work.is_dir() or (work / "ENTRY_FAILURE.json").exists():
        return
    payload = {
        "schema": SCHEMA + ".failure.v1",
        "classification": "SCIENTIFIC_NONEXACT" if "NONEXACT" in repr(error) else "ENGINEERING_OR_RESOURCE_FAILURE",
        "error": repr(error),
        "traceback": traceback.format_exc(),
    }
    sg.write(work / "ENTRY_FAILURE.json", payload)


def _manifest_for_request(args: Any, report: Path, work: Path, *, local_engineering: bool):
    if local_engineering:
        path = Path(args.input_manifest or report / "BENCHMARK_INPUT_MANIFEST.json")
        return sg.run_window, path
    contract = read(report / "INPUT_CONTRACT.json")
    bundle = full_entry._canonical_bundle_from_contract(contract)
    window, _ctx = full_entry._canonical_window(report, bundle, work)
    return window, work / "CANONICAL_INPUT_VIEW.json"


def _finite_outputs(result: dict[str, Any]) -> None:
    for name in ("final_E", "scientific_Qacc"):
        path = result.get("normal_outputs", {}).get(name)
        if not path:
            continue
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            require(np.isfinite(values).all(), "S46F_C_FULL_CAPTURE_NONFINITE:" + name)
        finally:
            del values
    post_path = result.get("normal_outputs", {}).get("final_POST")
    if post_path:
        with np.load(post_path, allow_pickle=False) as post:
            require(all(np.isfinite(post[name]).all() for name in post.files), "S46F_C_FULL_CAPTURE_POST_NONFINITE")


def _capture_runtime_receipts(result: dict[str, Any], dispatch: Any, *, local_engineering: bool) -> dict[str, Any]:
    """Record the original route/cache and C-pool scalar contracts."""

    intervals = 8
    n_half = 2 * intervals
    route = result.get("route")
    require(isinstance(route, dict), "S46F_C_FULL_CAPTURE_ROUTE_MISSING")
    require(route.get("nH") == n_half and route.get("fresh_responses") == 2 * n_half and route.get("math_dispatches") == 8 * n_half and route.get("fallback") == 0, "S46F_C_FULL_CAPTURE_ROUTE_COUNTS")
    lifecycle = route.get("lifecycle", {})
    if lifecycle:
        require(all(int(value) == n_half for value in lifecycle.values()) and not route.get("active_lifecycle_identity"), "S46F_C_FULL_CAPTURE_ROUTE_LIFECYCLE")
    cache = dispatch.cpu_s4_4a_cache_snapshot()
    stats = cache.get("stats", {})
    require((stats.get("factor_requests"), stats.get("plane_builds"), stats.get("hits")) == (6144, 384, 5760), "S46F_C_FULL_CAPTURE_CACHE_COUNTS")
    expected_bytes = 256 * 1024 * 1024 if local_engineering else 1_610_612_736
    require(cache.get("max_tables") == 2 and cache.get("max_bytes") == expected_bytes and cache.get("events_dropped") == 0, "S46F_C_FULL_CAPTURE_CACHE_POLICY")
    require(stats.get("invalidations") == stats.get("evictions") == stats.get("failed_requests") == 0, "S46F_C_FULL_CAPTURE_CACHE_FAILURES")
    finalized = getattr(dispatch, "_cpu_s4_6f_finalized_pool", None)
    require(isinstance(finalized, dict) and isinstance(finalized.get("pool"), dict) and isinstance(finalized.get("activity"), list), "S46F_C_FULL_CAPTURE_POOL_MISSING")
    pool = finalized["pool"]
    require(pool.get("closed") is True and pool.get("active_thread_ids") == [] and pool.get("pending_scalar_records") == 0, "S46F_C_FULL_CAPTURE_POOL_LIFECYCLE")
    require(len(finalized["activity"]) == 0, "S46F_C_FULL_CAPTURE_POOL_ACTIVITY")
    pool_evidence = dict(pool, records_drained=True, second_activity_drain_empty=True)
    return dict(route=copy.deepcopy(route), cache=copy.deepcopy(cache), cache_fallback=0, original_math_calls=8 * n_half, fresh_responses=2 * n_half, pointwise_pool=dict(pool=pool_evidence, activity=[]))


def _run_mechanism(groups: Any, *, local_engineering: bool) -> dict[str, Any]:
    """Use W1 only for local engineering; keep frozen HPC P1/P24 at 1/24."""

    if not local_engineering:
        return c_mechanism.run_local_mechanism(groups)
    original_pool = c_mechanism.CPool

    def local_pool(_count: int, *args: Any, **kwargs: Any):
        return original_pool(1, *args, **kwargs)

    c_mechanism.CPool = local_pool
    try:
        return c_mechanism.run_local_mechanism(groups)
    finally:
        c_mechanism.CPool = original_pool


def run_capture(args: Any) -> dict[str, Any]:
    local = bool(args.local_engineering)
    shape = MICRO_SHAPE if local else SHAPE
    mode = "PURE_OFF_FULLSIZE"
    full_entry.admission(candidate="PHASE_ABSORB_P24", local_engineering=local, shape=shape, intervals=8, mode=mode, allow_c_mechanism=True)
    report = Path(args.report).resolve(strict=True)
    work = Path(args.work).resolve()
    require(not work.exists(), "S46F_C_FULL_CAPTURE_NON_OVERWRITING_WORK")
    work.mkdir(parents=True)
    input_manifest = Path(args.input_manifest).resolve(strict=True) if args.input_manifest else report / ("BENCHMARK_INPUT_MANIFEST.json" if local else "FULL_INPUT_MANIFEST.json")
    full_entry.input_domain(read(input_manifest), local_engineering=local)
    pair = make_recorder_pair(False)
    timer = pair.b
    application_started = time.perf_counter()
    application_token = timer.begin("APPLICATION", common=True, context=timer.context(0, occurrence_class="WINDOW_ONCE", source_site="c_fullsize_entry.run_capture:before-source-and-factory"))
    store = CaptureStore() if bool(args.capture) else None
    if store is not None:
        store.begin("G1", source_role="R")
    built = None
    try:
        with timer.scope("SOURCE_FACTORY", common=True):
            with timer.scope("SOURCE_VALIDATION", common=True):
                binding = _verify_mechanism_source(report, local_engineering=local, shape=shape)
            sg.frozen_tools(report)
            with timer.scope("L1_FACTORY", common=True):
                built = build_role("PHASE_ABSORB_P24", mode, capture_root=work / "BOUNDARY", role="R", local_engineering=local, recorder_pair=pair, shape=shape, pool_override=1 if local else None, capture_store=store)
            cache = built[1].cpu_s4_4a_cache_snapshot()
            expected_bytes = 256 * 1024 * 1024 if local else 1_610_612_736
            require(cache["stats"]["factor_requests"] == cache["stats"]["plane_builds"] == cache["stats"]["cache_bytes"] == 0 and cache["max_tables"] == 2 and cache["max_bytes"] == expected_bytes, "S46F_C_FULL_CAPTURE_FRESH_CACHE")
        window, request_manifest = _manifest_for_request(args, report, work, local_engineering=local)
        request = SimpleNamespace(
            report=report,
            work=work,
            role="G3_SMALL",
            reported_role="R",
            semantic_candidate="PHASE_ABSORB_P24",
            mode=mode,
            intervals=8,
            local_engineering=local,
            input_manifest=request_manifest,
            reference_output=args.reference_output,
            reference_boundary=None,
            reference_stages=None,
            execution_domain="S46F_LOCAL_MICRO" if local else "FULLSIZE_CONFIRMATION",
        )
        result = run_window(request, built, timer=timer, application_token=application_token, window=window, label="formal", intervals=8, capture=False, application_started=application_started)
        require(result.get("comparisons", {}).get("terminal", {}).get("status") == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS" if args.reference_output else True, "S46F_C_FULL_CAPTURE_TERMINAL_NONEXACT")
        _finite_outputs(result)
        binding_after = verify_role(built[1], "PHASE_ABSORB_P24")
        sg.write(work / "BINDING_AFTER.json", binding_after)
        runtime = _capture_runtime_receipts(result, built[1], local_engineering=local)
    except BaseException as error:
        if built is not None:
            try:
                close_role(built[1], "PHASE_ABSORB_P24")
            except BaseException as cleanup:
                error.add_note("S46F_C_FULL_CAPTURE_CLOSE_FAILURE:" + type(cleanup).__name__)
        if store is not None:
            try:
                store.release_owned_arrays()
            except BaseException as cleanup:
                error.add_note("S46F_C_FULL_CAPTURE_RELEASE_FAILURE:" + type(cleanup).__name__)
        if work.is_dir():
            _write_failure(work, error)
        raise
    if store is not None:
        try:
            rows = store.require_complete()
            # Source verification returns EXECUTION metadata.  The actual
            # source-bound C wrapper is the fourth factory tuple item.
            runtime_binding = _capture_runtime_binding(built[3], role="R", mode="PURE_OFF_SMALLGRID")
            canonical_view = work / "CANONICAL_INPUT_VIEW.json"
            package = _save_capture(
                store,
                Path(args.capture_root).resolve() if args.capture_root else work / "CAPTURE",
                provenance={
                    "execution_json_sha256": pv.raw_sha256_file(report / "EXECUTION.json"),
                    "input_manifest_sha256": pv.raw_sha256_file(report / ("BENCHMARK_INPUT_MANIFEST.json" if local else "FULL_INPUT_MANIFEST.json")),
                    "canonical_input_view_sha256": pv.raw_sha256_file(canonical_view) if canonical_view.is_file() else None,
                    "source_identity_digest": _source_identity_digest(),
                    "source_binding": {
                        "candidate_schema": runtime_binding.get("c_candidate_schema"),
                        "role": runtime_binding.get("c_role"),
                        "mode": runtime_binding.get("c_mode"),
                        "source": runtime_binding.get("c_source_binding"),
                        "source_identity_digest": _source_identity_digest(),
                    },
                    "case_source": {
                        "entry": "tools.cpu_s4_6f_fullsize_parallel.c_fullsize_entry.run_capture",
                        "role": "R",
                        "intervals": 8,
                        "shape": list(shape),
                        "capture": True,
                        "selected_interval_indices": list(CAPTURE_INTERVALS),
                        "input_manifest": str(input_manifest),
                    },
                },
            )
            capture_root = Path(args.capture_root).resolve() if args.capture_root else work / "CAPTURE"
            receipt_source_binding = copy.deepcopy(package.get("provenance", {}).get("source_binding", {}))
            receipt_source_binding.update({"role": "R", "source_identity_digest": _source_identity_digest()})
            sg.write(work / "CAPTURE_RECEIPT.json", dict(
                schema=SCHEMA + ".capture_receipt.v1",
                shape=list(shape),
                selected_interval_indices=list(CAPTURE_INTERVALS),
                expected_total_calls=CAPTURE_TOTAL_CALLS,
                source_binding=receipt_source_binding,
                capture_package=str(capture_root),
                capture_metadata=package["metadata"],
                raw_files=package["files"],
                terminal_status=result.get("comparisons", {}).get("terminal", {}).get("status"),
                result_output=result.get("output"),
            ))
        finally:
            store.release_owned_arrays()
    result.update(
        schema=SCHEMA + ".case.v1",
        execution_schema=EXECUTION_SCHEMA,
        execution_commit=read(report / "EXECUTION.json")["execution_commit"],
        execution_domain="S46F_LOCAL_MICRO" if local else "S46F_FULLSIZE",
        role="R",
        intervals=8,
        capture=bool(args.capture),
        selected_interval_indices=list(CAPTURE_INTERVALS) if args.capture else [],
        cpu_diagnostics_profile=True,
        diagnostic_outputs=True,
        fine_performance_profile=False,
        production_qualified=False,
        default="I0",
        review_gate="MANUAL_REVIEW_PENDING",
    )
    result.update(runtime)
    sg.write(work / "CASE_RECEIPT.json", result)
    return result


def run_mechanism(args: Any) -> dict[str, Any]:
    local = bool(args.local_engineering)
    shape = MICRO_SHAPE if local else SHAPE
    report = Path(args.report).resolve(strict=True)
    _verify_mechanism_source(report, local_engineering=local, shape=shape)
    execution_sha = pv.raw_sha256_file(report / "EXECUTION.json")
    manifest_name = "BENCHMARK_INPUT_MANIFEST.json" if local else "FULL_INPUT_MANIFEST.json"
    input_sha = pv.raw_sha256_file(report / manifest_name)
    bundle = _load_capture(Path(args.capture_root), expected_execution_sha=execution_sha, expected_input_manifest_sha=input_sha, expected_source_identity_digest=_source_identity_digest(), expected_shape=shape)
    g2 = copy.deepcopy(bundle)
    g2["metadata"]["group"] = "G2"
    for row in g2["records"].values():
        row["group"] = "G2"
    mechanism = _run_mechanism({"G1": bundle, "G2": g2}, local_engineering=local)
    if mechanism.get("status") == "SCIENTIFIC_NONEXACT":
        status = "SCIENTIFIC_NONEXACT"
    elif mechanism.get("mechanism_gate") is not True:
        status = "SCIENTIFIC_INSUFFICIENT_P24_GAIN"
    elif local:
        # W1 is an engineering smoke path only; its timing cannot satisfy
        # the full-size P24 admission contract.
        status = "PASS_C_LOCAL_ENGINEERING_ONLY"
    else:
        status = "PASS_C_FULLSIZE_MECHANISM_EXACT_GAIN_GATE"
    result = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "execution_commit": read(report / "EXECUTION.json")["execution_commit"],
        "shape": list(shape),
        "capture_root": str(Path(args.capture_root).resolve()),
        "selected_interval_indices": list(CAPTURE_INTERVALS),
        "mechanism": mechanism,
        "production_qualified": False,
        "default": "I0",
        "review_gate": "MANUAL_REVIEW_PENDING",
    }
    sg.write(Path(args.output).resolve(), result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--role", choices=("R",))
    parser.add_argument("--intervals", type=int, choices=(8,))
    parser.add_argument("--local-engineering", action="store_true")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--mechanism", action="store_true")
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    args = parser.parse_args(argv)
    if args.mechanism:
        require(args.capture_root is not None and args.output is not None, "S46F_C_FULL_MECHANISM_PATHS_REQUIRED")
        try:
            result = run_mechanism(args)
        except BaseException as error:
            _write_failure(Path(args.output).resolve().parent, error)
            raise
    else:
        require(args.work is not None and args.role == "R" and args.intervals == 8, "S46F_C_FULL_CAPTURE_ARGS_REQUIRED")
        try:
            result = run_capture(args)
        except BaseException as error:
            if args.work is not None:
                _write_failure(Path(args.work).resolve(), error)
            raise
    print(json.dumps({key: result.get(key) for key in ("status", "schema", "capture", "formal_calls", "warmup_calls") if key in result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
