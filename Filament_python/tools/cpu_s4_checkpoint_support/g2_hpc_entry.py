"""Private G2 HPC role entry.

This entry binds the accepted A07 adapter to the frozen G1 context and
propagation helpers.  It does not submit jobs, alter production modules, or
introduce a second numerical implementation.  Each invocation is one fresh
role process and one contiguous application-service clock.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping


ROLES = ("I0", "G_CONTROLLED_INPUT")
MODES = ("QUALIFY_BOUNDARY", "PURE_OFF_REAL8")
INTERVALS = 8
FULL_SHAPE = (384, 351, 301)
EXPECTED_WORKERS = {"temporal_workers": 48, "spatial_workers": 24}
EXPECTED_G_HASH_ROLES = ("E_stage1", "Omega_stage1", "Omega_apply", "Omega_stage2")


def _portable_build_role():
    """Resolve G2 through the repository-local source binding bridge."""
    try:
        from portable_binding import load_g2_adapter
    except ImportError:  # pragma: no cover - package-style import
        from .portable_binding import load_g2_adapter

    return load_g2_adapter().build_role


def _require(condition: bool, reason: str, context: Any = None) -> None:
    if not condition:
        raise RuntimeError({"gate": str(reason), "context": context})


def _unoptimized() -> None:
    _require(sys.flags.optimize == 0 and not os.environ.get("PYTHONOPTIMIZE"), "G2_HPC_OPTIMIZATION_FORBIDDEN")


def _allocation_identity() -> dict[str, Any]:
    """Record scheduler allocation state without implying child submission."""
    job_id = os.environ.get("SLURM_JOB_ID")
    environment = {
        "slurm_job_id": job_id,
        "hostname": socket.gethostname(),
        "in_hpc_allocation": bool(job_id),
        "scheduler_submitted_by_child": False,
    }
    return {
        "in_hpc_allocation": bool(job_id),
        "job_id": job_id,
        "scheduler_submitted_by_child": False,
        "execution_environment": environment,
    }


def _quiet_runtime() -> dict[str, Any]:
    """Reject ambient observers before importing scientific code."""
    _require(sys.gettrace() is None, "G2_TRACE_FORBIDDEN")
    _require(sys.getprofile() is None, "G2_PROFILE_FORBIDDEN")
    monitoring: list[dict[str, Any]] = []
    monitor = getattr(sys, "monitoring", None)
    if monitor is not None:
        for tool_id in range(8):
            try:
                registered = monitor.get_tool(tool_id)
                events = int(monitor.get_events(tool_id))
            except BaseException:
                registered, events = None, 0
            monitoring.append({"id": tool_id, "registered": registered, "events": events})
        _require(all(row["registered"] is None and row["events"] == 0 for row in monitoring), "G2_MONITORING_FORBIDDEN", monitoring)
    return {"trace": None, "profile": None, "monitoring": monitoring}


def _ensure_work(path: str | os.PathLike[str]) -> Path:
    work = Path(path).resolve()
    if work.exists():
        _require(work.is_dir() and not any(work.iterdir()), "G2_WORK_NONEMPTY", str(work))
    else:
        work.mkdir(parents=True, exist_ok=False)
    return work


def _check_numba(pool: int) -> dict[str, Any]:
    import numba

    _require(int(numba.config.NUMBA_NUM_THREADS) == int(pool), "G2_ROLE_POOL", {
        "expected": int(pool), "actual": int(numba.config.NUMBA_NUM_THREADS)
    })
    _require(
        str(numba.config.CPU_NAME) == "generic" and str(numba.config.THREADING_LAYER) == "workqueue",
        "G2_NUMBA_GENERIC_WORKQUEUE",
        {"CPU_NAME": numba.config.CPU_NAME, "THREADING_LAYER": numba.config.THREADING_LAYER},
    )
    numba.set_num_threads(24)
    active_layer = str(numba.threading_layer())
    _require(active_layer == "workqueue", "G2_NUMBA_ACTIVE_WORKQUEUE", active_layer)
    return {
        "configured_pool": int(pool),
        "compiler_cpu": str(numba.config.CPU_NAME),
        "threading_layer": str(numba.config.THREADING_LAYER),
        "active_threading_layer": active_layer,
        "outer_mask": int(numba.get_num_threads()),
    }


def _blas_contract() -> list[dict[str, Any]]:
    from threadpoolctl import threadpool_info

    pools = threadpool_info()
    _require(
        all(int(item.get("num_threads", 1)) == 1 for item in pools if item.get("user_api") == "blas"),
        "G2_BLAS_POOL1",
        pools,
    )
    return pools


def _context(root: Path, plan: Mapping[str, Any], work: Path, pool: int, *, tiny: bool) -> dict[str, Any]:
    if tiny:
        _require(plan.get("local_test_only") is True, "G2_TINY_REQUIRES_LOCAL_TEST_PLAN")
        from g2_cases import local_context

        return local_context()
    from g1_common import context

    return context(work, dict(plan), pool)


def _expected_pre0_hash(plan: Mapping[str, Any]) -> str:
    """Resolve the plan's explicit frozen PRE0 canonical identity."""
    candidates: list[Any] = []
    for key in ("pre0_canonical_sha256", "pre0_hash", "pre0_canonical_hash"):
        candidates.append(plan.get(key))
    for container_name in ("pre0_identity", "source_pre_identity", "input_identity"):
        container = plan.get(container_name)
        if isinstance(container, Mapping):
            for key in ("pre0_canonical_sha256", "pre0_hash", "canonical_sha256"):
                candidates.append(container.get(key))
    config = plan.get("run_config")
    if isinstance(config, Mapping):
        for key in ("pre0_canonical_sha256", "pre0_hash", "pre0_canonical_hash"):
            candidates.append(config.get(key))
    values = [str(value) for value in candidates if value not in (None, "")]
    _require(bool(values), "G2_PLAN_PRE0_IDENTITY_MISSING")
    _require(len(set(values)) == 1, "G2_PLAN_PRE0_IDENTITY_AMBIGUOUS", values)
    return values[0]


def _loader_identity(root: Path, work: Path, pv: Any, plan: Mapping[str, Any], context_value: Mapping[str, Any], *, tiny: bool) -> dict[str, Any]:
    """Require real loader provenance and the plan-bound PRE0 hash for full-size cases."""
    import numpy as np
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1

    if tiny:
        return {
            "status": "TINY_LOCAL_EXEMPT",
            "real_input_used": False,
            "pre0_canonical_sha256": None,
            "expected_pre0_canonical_sha256": None,
            "loader_receipt": None,
        }
    _require(context_value.get("real_input_used") is True, "G2_REAL_INPUT_REQUIRED", context_value.get("real_input_used"))
    assets = context_value.get("assets")
    pre_fields = context_value.get("pre_fields")
    _require(isinstance(assets, Mapping), "G2_REAL_ASSETS_MISSING")
    _require(isinstance(pre_fields, Mapping) and isinstance(pre_fields.get("delta_n"), np.ndarray), "G2_REAL_PRE_FIELDS_MISSING")
    expected = _expected_pre0_hash(plan)
    actual = j1._canonical_hash(pre_fields["delta_n"])
    context_hash = context_value.get("pre0_hash") or assets.get("pre0_canonical_sha256")
    _require(str(context_hash) == actual, "G2_CONTEXT_PRE0_IDENTITY", {"context": context_hash, "actual": actual})
    _require(actual == expected, "G2_PLAN_PRE0_IDENTITY", {"expected": expected, "actual": actual})
    # g1_common.context receives this case's work directory, not plan root.
    path = work / "LOADER_RECEIPT.json"
    _require(path.is_file(), "G2_LOADER_RECEIPT_MISSING", str(path))
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError({"gate": "G2_LOADER_RECEIPT_PARSE", "context": str(path)}) from exc
    snapshot = path
    snapshot_raw_sha256 = str(pv.raw_sha256_file(snapshot))
    _require(snapshot_raw_sha256 == str(pv.raw_sha256(raw)), "G2_LOADER_RECEIPT_SNAPSHOT_DRIFT", str(snapshot))
    return {
        "status": "PASS_REAL_LOADER_PRE0_IDENTITY",
        "real_input_used": True,
        "pre0_canonical_sha256": actual,
        "expected_pre0_canonical_sha256": expected,
        "loader_receipt": {
            "path": str(snapshot),
            "source_path": str(path),
            "raw_sha256": snapshot_raw_sha256,
            "size_bytes": int(len(raw)),
            "payload": payload,
            "case_local_original_receipt_retained": True,
        },
    }


def _source_and_pre_fingerprints(context_value: Mapping[str, Any]) -> dict[str, Any]:
    from g1_exact import fingerprints
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1

    source = context_value.get("source")
    _require(source is not None, "G2_SOURCE_MISSING")
    return {
        "source": {
            "canonical_sha256": j1._canonical_hash(source),
            "shape": [int(item) for item in source.shape],
            "dtype": str(source.dtype),
            "strides": [int(item) for item in source.strides],
            "C": bool(source.flags.c_contiguous),
        },
        "PRE": fingerprints(context_value.get("pre_fields", {})),
    }


def _current_pre_field_identity(view: Any, *, intervals: int) -> dict[str, Any]:
    """Hash the fresh disk-backed CURRENT PRE fields before propagation."""
    import numpy as np
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle
    import g2_storage_bridge as bridge

    lifecycle = StreamingLifecycle.open(view.root)
    fields = {
        name: np.stack(
            [np.asarray(lifecycle.current_fields(index)[name], dtype=np.float64) for index in range(int(intervals))],
            axis=0,
        )
        for name in bridge.PRE_FIELDS
    }
    return bridge.current_pre_field_identity(fields, intervals=int(intervals))


def _write_normal_outputs(
    bundle: Mapping[str, Any],
    propagation_directory: Path,
    work: Path,
    *,
    role: str,
    raw_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist the ordinary scientific files before the application clock ends."""
    import numpy as np
    from g2_hpc_exact import POST_FIELDS

    returned = tuple(bundle.get("returned", ()))
    _require(len(returned) >= 3, "G2_RETURNED_BUNDLE_INCOMPLETE")
    final_e, qacc, diagnostics = returned[:3]
    _require(isinstance(diagnostics, Mapping), "G2_DIAGNOSTICS_MAPPING")
    _require("Qacc_raman" in diagnostics, "G2_QACC_RAMAN_DIAGNOSTIC_MISSING")
    propagation_directory = Path(propagation_directory)
    propagation_directory.mkdir(parents=True, exist_ok=True)

    final_e_path = propagation_directory / "final_E.npy"
    qacc_path = Path(work) / "scientific_Qacc.npy"
    final_post_path = propagation_directory / "final_POST.npz"
    diagnostics_path = propagation_directory / "diagnostics.npz"
    scalar_diagnostics_path = Path(work) / "scientific_scalar_diagnostics.json"
    for path in (final_e_path, qacc_path, final_post_path, diagnostics_path, scalar_diagnostics_path):
        _require(not path.exists(), "G2_NORMAL_OUTPUT_OVERWRITE", str(path))
    with final_e_path.open("xb") as stream:
        np.save(stream, np.asarray(final_e), allow_pickle=False)
    with qacc_path.open("xb") as stream:
        np.save(stream, np.asarray(qacc), allow_pickle=False)

    post_records = dict(bundle.get("POST", {}))
    _require(tuple(sorted(post_records, key=lambda item: int(item))) == tuple(str(i) for i in range(INTERVALS)), "G2_POST_OUTPUT_INTERVALS")
    post_names = tuple(POST_FIELDS)
    final_post = {
        name: np.stack(
            [np.asarray(post_records[str(i)][name]) for i in range(INTERVALS)],
            axis=0,
        )
        for name in post_names
    }
    with final_post_path.open("xb") as stream:
        np.savez(stream, **final_post)

    numeric_diagnostics = {}
    for name, value in diagnostics.items():
        if isinstance(value, (np.ndarray, list, tuple)):
            array = np.asarray(value)
            if array.dtype.kind in "biufc":
                numeric_diagnostics[str(name)] = array
    with diagnostics_path.open("xb") as stream:
        np.savez(stream, **numeric_diagnostics)

    # Preserve the original endpoint convention for numeric scalar leaves;
    # these 102 fields are separate from the ndarray diagnostics archive.
    from tools.cpu_o1 import q1_t0

    scalar_source = raw_diagnostics if isinstance(raw_diagnostics, Mapping) else diagnostics
    scalar_payload, _ = q1_t0._numeric_scalar_diagnostics(dict(scalar_source))
    q1_t0._write_scalar_diagnostics(scalar_diagnostics_path, str(role), scalar_payload)

    sink_paths = sorted(path for path in propagation_directory.glob("sinks.*.npy") if path.is_file())
    profile_sidecar = propagation_directory / "RAW_LINEAR_PROFILE_SIDECARS.json"
    return {
        "propagation_root": str(propagation_directory),
        "final_E": str(final_e_path),
        "scientific_Qacc": str(qacc_path),
        "final_POST": str(final_post_path),
        "diagnostics": str(diagnostics_path),
        "scientific_scalar_diagnostics": str(scalar_diagnostics_path),
        "scalar_diagnostic_fields": sorted(scalar_payload),
        "scalar_diagnostic_field_count": len(scalar_payload),
        "diagnostic_fields": sorted(numeric_diagnostics),
        "profile_sidecar": str(profile_sidecar) if profile_sidecar.is_file() else None,
        "sinks": [str(path) for path in sink_paths],
    }


def _role_contract(
    role: str,
    route: Mapping[str, Any],
    fft_delta: Mapping[str, Any],
    *,
    intervals: int,
    fullsize: bool,
) -> dict[str, Any]:
    """Check only the accepted A07/G1 route evidence, with derived counts."""
    _require(int(route.get("mask_after", -1)) == 24, "G2_MASK_AFTER_ROUTE", route)
    if role == "I0":
        return {
            "role": "I0",
            "route": "I0",
            "halves": int(route.get("halves", 0)),
            "fft_transform_counts": dict(fft_delta),
            "fallback": 0,
            "route_contract": "FROZEN_I0_ORIGINAL_FACTORY",
        }

    expected_halves = 2 * intervals
    hash_rows = list(route.get("hash_evidence", []))
    _require(len(hash_rows) == expected_halves, "G2_G_HASH_HALF_COUNT", len(hash_rows))
    _require(all(tuple(row.get("hash_roles", ())) == EXPECTED_G_HASH_ROLES for row in hash_rows), "G2_G_HASH_ROLE_DRIFT", hash_rows)
    dispatches = list(route.get("actual_numba_kernel_executions", []))
    _require(len(dispatches) == 128 if intervals == 8 else len(dispatches) == 16 * intervals, "G2_G_DISPATCH_COUNT", len(dispatches))
    counters = dict(route.get("actual_counter_deltas", {}))
    g_route = dict(route.get("actual_G_route_deltas", {}))
    _require(int(counters.get("controlled_full_calls", -1)) == expected_halves, "G2_G_CONTROLLED_CALL_COUNT", counters)
    _require(int(counters.get("strict_full_calls", -1)) == 0 and int(counters.get("failed_full_calls", -1)) == 0, "G2_G_FALLBACK_OR_FAILURE", counters)
    _require(int(g_route.get("F_STRICT_FALLBACK", -1)) == 0 and int(g_route.get("REJECTED", -1)) == 0, "G2_G_FALLBACK_OR_REJECTED", g_route)
    operations = dict(route.get("actual_operation_deltas", {}))
    response_operation_counts = {
        name: int(operations.get(name, -1))
        for name in ("R_PRODUCT", "R_DERIVATIVE", "R_RHS")
    }
    _require(
        len(set(response_operation_counts.values())) == 1
        and next(iter(response_operation_counts.values())) == 4 * intervals,
        "G2_G_RESPONSE_OPERATION_COUNT",
        response_operation_counts,
    )
    response_count = next(iter(response_operation_counts.values()))
    _require(int(operations.get("H_PREDICTOR", -1)) == expected_halves and int(operations.get("H_CORRECTOR", -1)) == expected_halves, "G2_G_HEUN_COUNT", operations)
    if fullsize:
        _require(all(row.get("active_thread_ids") == list(range(48)) for row in dispatches), "G2_G_ALL_48_WORKERS", dispatches)
    _require(int(counters.get("linear_producer_calls", -1)) == expected_halves, "G2_G_LINEAR_PRODUCER_COUNT", counters)
    _require(int(g_route.get("G_ACTUAL", -1)) == expected_halves, "G2_G_ACTUAL_ROUTE_COUNT", g_route)
    return {
        "role": "G_CONTROLLED_INPUT",
        "route": "G_ACTUAL",
        "halves": expected_halves,
        "fresh_response_count": response_count,
        "fresh_response_derivation": "equal_actual_R_PRODUCT_R_DERIVATIVE_R_RHS_deltas",
        "response_operation_deltas": response_operation_counts,
        "actual_numba_kernel_dispatches": len(dispatches),
        "E_stage1": expected_halves,
        "E_apply": 0,
        "Omega_dispatches": 3 * expected_halves,
        "fallback": 0,
        "rejected": 0,
        "fft_transform_counts": dict(fft_delta),
        "route_contract": "A07_SOURCE_BOUND_G_CONTROLLED_INPUT",
    }


def _boundary_mapping_path() -> Path | None:
    candidates = (
        Path(__file__).resolve().parents[1] / "inputs" / "HISTORICAL_BOUNDARY_FIELD_MAP_112.csv",
        Path(__file__).resolve().parents[1] / "BOUNDARY_FIELD_MAP_112.csv",
    )
    return next((path for path in candidates if path.is_file()), None)


def _historical_compare(root: Path, plan: Mapping[str, Any], current_boundary: Mapping[str, Any], work: Path, identity: Any) -> dict[str, Any]:
    import g2_hpc_exact as exact

    oracle = plan.get("historical_oracle_root")
    _require(oracle is not None, "BLOCKED_INPUT_OR_ORACLE_MISSING")
    scalar = exact._resolve_scalar_path(plan, root)
    mapping_value = plan.get("historical_boundary_mapping") or _boundary_mapping_path()
    _require(mapping_value is not None, "HISTORICAL_BOUNDARY_MAPPING_REQUIRED")
    mapping = Path(str(mapping_value))
    if not mapping.is_absolute():
        mapping = root / mapping
    historical = exact.load_historical_boundary(oracle, scalar, mapping_path=mapping)
    result = exact.compare_historical_boundary(
        historical,
        current_boundary,
        work / "FULL_EXACT_COMPARISONS" / "HISTORICAL_FIRST_BOUNDARY.jsonl",
        candidate_identity=identity,
    )
    exact.write_json(work / "FULL_EXACT_COMPARISONS" / "HISTORICAL_FIRST_BOUNDARY_RECEIPT.json", result)
    _require(result["status"] == "PASS_HISTORICAL_BOUNDARY_112_EXACT", "STOP_FIRST_BOUNDARY_FIELD_OR_SEMANTIC_GAP", result)
    return result


def _storage_bridge_contract(root: Path, plan: Mapping[str, Any]) -> dict[str, Any] | None:
    """Load the explicitly deployed storage bridge contract, if present."""
    row = plan.get("storage_bridge_contract")
    if row is None:
        return None
    _require(isinstance(row, Mapping) and row.get("path") == "STORAGE_PROTOCOL_BRIDGE_CONTRACT.json",
             "G2_STORAGE_BRIDGE_PLAN_RECORD", row)
    import g2_storage_bridge as bridge

    path = Path(str(row["path"]))
    if not path.is_absolute():
        path = root / path
    return bridge.load_contract(path)


def _historical_terminal_compare(
    root: Path,
    plan: Mapping[str, Any],
    bundle: Mapping[str, Any],
    work: Path,
    identity: Any,
    current_storage_protocol: Mapping[str, Any],
    current_pre_field_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    import numpy as np
    import g2_hpc_exact as exact

    if not plan.get("historical_terminal_required") and not plan.get("historical_scientific_Qacc"):
        return None
    import g2_storage_bridge as bridge

    contract = _storage_bridge_contract(root, plan)
    historical_artifacts = None
    historical_metadata = None
    if contract is not None:
        historical_artifacts = bridge.verify_historical_artifacts(plan["historical_oracle_root"], contract)
        historical_metadata = historical_artifacts["historical_metadata"]
    records = exact.historical_terminal_records(plan["historical_oracle_root"])
    _require(
        len(records.get("diagnostics", {})) == exact.EXPECTED_TERMINAL_DIAGNOSTIC_ARRAYS,
        "HISTORICAL_TERMINAL_DIAGNOSTIC_ARRAY_COUNT",
        len(records.get("diagnostics", {})),
    )
    _require(
        len(records.get("scalar_diagnostics", {})) == exact.EXPECTED_TERMINAL_SCALARS,
        "HISTORICAL_TERMINAL_SCALAR_COUNT",
        len(records.get("scalar_diagnostics", {})),
    )
    qacc_path = plan.get("historical_scientific_Qacc")
    if qacc_path:
        path = Path(str(qacc_path))
        if not path.is_absolute():
            path = root / path
        path = path.resolve(strict=True)
    else:
        path = Path(plan["historical_oracle_root"]).resolve(strict=True) / "scientific_Qacc.npy"
    _require(path.is_file(), "HISTORICAL_SCIENTIFIC_QACC_MISSING", str(path))
    records["scientific_Qacc"] = np.load(path, mmap_mode="r", allow_pickle=False)
    _require(len(records.get("sinks", {})) >= 6, "HISTORICAL_SIX_SINKS_MISSING", len(records.get("sinks", {})))
    raw_result = exact.compare_terminal_records(
        records,
        bundle,
        work / "FULL_EXACT_COMPARISONS" / "HISTORICAL_TERMINAL.jsonl",
        candidate_identity=identity,
    )
    exact.write_json(work / "FULL_EXACT_COMPARISONS" / "HISTORICAL_TERMINAL_RECEIPT.json", raw_result)
    if contract is not None:
        _require(isinstance(historical_metadata, Mapping), "STORAGE_BRIDGE_HISTORICAL_PROTOCOL_MISSING")
        result = bridge.bridge_terminal_comparison(
            raw_result,
            work / "FULL_EXACT_COMPARISONS" / "HISTORICAL_TERMINAL.jsonl",
            historical_metadata,
            current_storage_protocol,
            contract,
            historical_artifacts=historical_artifacts,
            current_pre_field_identity=current_pre_field_identity,
        )
        exact.write_json(
            work / "FULL_EXACT_COMPARISONS" / "HISTORICAL_TERMINAL_STORAGE_BRIDGE_RECEIPT.json",
            result,
        )
    else:
        # A missing bridge authorization deliberately retains the original
        # strict comparator gate; it never creates an implicit exception.
        _require(raw_result["status"] == "PASS_HISTORICAL_TERMINAL_EXACT", "STOP_SCIENCE_NONEXACT", raw_result)
        result = raw_result
    _require(
        int(result.get("comparison_row_count", -1)) == exact.EXPECTED_TERMINAL_COMPARISON_ROWS,
        "HISTORICAL_TERMINAL_ROW_COUNT",
        result,
    )
    return result


def _entry_smoke(root: Path, plan: Mapping[str, Any], args: argparse.Namespace, work: Path) -> dict[str, Any]:
    _unoptimized()
    pool = 24 if args.role == "I0" else 48
    runtime = _check_numba(pool)
    build_role = _portable_build_role()

    prop, dispatch, clock, binding, audit, lease = build_role(args.role, args.mode, capture_root=work / "BOUNDARY")
    workers = dict(dispatch.report().get("workers", {}))
    _require(workers == EXPECTED_WORKERS, "G2_FFT_WORKER_CONTRACT", workers)
    receipt = {
        "status": "PASS_G2_HPC_ENTRY_SMOKE",
        "scientific_arrays_loaded": False,
        "role": args.role,
        "mode": args.mode,
        "fullsize": False,
        "pool": runtime,
        "workers": workers,
        "lease": lease is not None,
        "binding": binding,
        "audit": audit.as_dict(),
        "clock_rows": copy.deepcopy(clock.rows),
        "clock_half_rows": copy.deepcopy(clock.half_rows),
        **_allocation_identity(),
    }
    from g2_hpc_common import write

    write(work / "CASE_RECEIPT.json", receipt)
    return receipt


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run one fresh role case; the caller is responsible for process isolation."""
    _unoptimized()
    _require(args.role in ROLES, "G2_ROLE_UNSUPPORTED", args.role)
    _require(args.mode in MODES, "G2_MODE_UNSUPPORTED", args.mode)
    application_started = time.perf_counter()
    observer_state = _quiet_runtime()
    import g2_hpc_common as common

    root, plan, pv = common.bootstrap(args.plan)
    tiny = bool(args.tiny)
    fullsize = not tiny
    work = _ensure_work(args.work)
    if args.entry_smoke:
        return _entry_smoke(root, plan, args, work)

    pool = 24 if args.role == "I0" else 48
    runtime = _check_numba(pool)
    context_value = _context(root, plan, work, pool, tiny=tiny)
    import numpy as np
    import numba
    import scipy
    import llvmlite

    source = context_value.get("source")
    _require(type(source) is np.ndarray and source.dtype == np.complex128 and source.ndim == 3 and source.flags.c_contiguous, "G2_SOURCE_DTYPE_LAYOUT")
    if fullsize:
        _require(tuple(int(item) for item in source.shape) == FULL_SHAPE, "G2_REAL_SOURCE_GRID_SHAPE", list(source.shape))
    source_pre_before = _source_and_pre_fingerprints(context_value)
    loader_identity = _loader_identity(root, work, pv, plan, context_value, tiny=tiny)

    from g1_common import metrics
    from g1_exact import fingerprints, save_tree
    from g1_segment import RoleAudit, verify_route
    build_role = _portable_build_role()
    from g2_cases import finish, prepare
    import g2_storage_bridge as storage_bridge
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1

    prop, dispatch, clock, binding, g2_audit, lease = build_role(args.role, args.mode, capture_root=work / "BOUNDARY")
    route_audit = RoleAudit(prop.__g2_inner__, dispatch, args.role, clock)
    workers = dict(dispatch.report().get("workers", {}))
    _require(workers == EXPECTED_WORKERS, "G2_FFT_WORKER_CONTRACT", workers)
    blas_pools = _blas_contract()
    _require(numba.get_num_threads() == 24, "G2_OUTER_MASK_BEFORE", numba.get_num_threads())

    propagation_directory = work / "PROPAGATION"
    view = None
    storage_protocol: Mapping[str, Any] | None = None
    storage_protocol_pre_fields: Mapping[str, Any] | None = None
    try:
        field, kwargs, view = prepare(context_value, propagation_directory, intervals=INTERVALS)
        storage_protocol_contract = _storage_bridge_contract(root, plan)
        storage_protocol = storage_bridge.validate_current_metadata(view.metadata(), storage_protocol_contract)
        storage_protocol_pre_fields = _current_pre_field_identity(view, intervals=INTERVALS)
        _require(not np.shares_memory(field, source), "G2_FRESH_SOURCE_OWNERSHIP")
        source_before_hash = source_pre_before["source"]["canonical_sha256"]
        input_hash = j1._canonical_hash(field)
        _require(input_hash == source_before_hash, "G2_FRESH_SOURCE_HASH")
        pre_current_before = str(view.current_content_sha256)
        before = metrics(dispatch, args.role)
        if args.role == "G_CONTROLLED_INPUT":
            before["_g1_runtime_execution_count"] = len(dispatch.cpu_s4_3e_state.r8_state._runtime.get("executions", []))
        fft_before = dict(dispatch.report().get("transform_counts", {}))
        _quiet_runtime()

        propagation_started = time.perf_counter()
        propagation_cpu_started = time.process_time()
        result = prop(field, **kwargs)
        propagation_wall_s = time.perf_counter() - propagation_started
        propagation_process_cpu_s = time.process_time() - propagation_cpu_started
        after = metrics(dispatch, args.role)
        route = verify_route(args.role, dispatch, before, after, route_audit, clock, intervals=INTERVALS, fullsize=fullsize)
        if args.role == "G_CONTROLLED_INPUT":
            route["actual_counter_deltas"] = {
                name: int(after.get(name, 0)) - int(before.get(name, 0))
                for name in (
                    "controlled_full_calls",
                    "strict_full_calls",
                    "strict_route_calls",
                    "failed_full_calls",
                    "linear_producer_calls",
                    "post_capability_issues",
                )
            }
            route["actual_operation_deltas"] = {
                name: int(after["operation_counts"][name]) - int(before["operation_counts"][name])
                for name in ("R_PRODUCT", "R_DERIVATIVE", "R_RHS", "H_PREDICTOR", "H_CORRECTOR")
            }
            route["actual_G_route_deltas"] = {
                name: int(after.get("route_counts", {}).get(name, 0))
                - int(before.get("route_counts", {}).get(name, 0))
                for name in ("G_ACTUAL", "F_STRICT_FALLBACK", "REJECTED")
            }
        fft_after = dict(dispatch.report().get("transform_counts", {}))
        fft_delta = {
            str(key): int(fft_after.get(key, 0)) - int(fft_before.get(key, 0))
            for key in sorted(set(fft_before) | set(fft_after))
        }
        route_contract = _role_contract(args.role, route, fft_delta, intervals=INTERVALS, fullsize=fullsize)
        bundle = finish(result, view, propagation_directory, intervals=INTERVALS)
    except BaseException:
        try:
            clock.abort()
        finally:
            if view is not None:
                view.close()
        raise
    finally:
        if view is not None and getattr(view, "complete", False):
            view.close()

    _require(numba.get_num_threads() == 24, "G2_OUTER_MASK_AFTER", numba.get_num_threads())
    _require(str(view.current_content_sha256) == pre_current_before, "G2_PRE_CURRENT_IDENTITY", {
        "before": pre_current_before,
        "after": str(view.current_content_sha256),
    })

    # Mandatory post-propagation source/PRE immutability checks remain inside
    # the application service span, before the ordinary output flush ends it.
    source_after_hashes = _source_and_pre_fingerprints(context_value)
    _require(source_after_hashes["source"]["canonical_sha256"] == source_before_hash, "G2_SOURCE_CHANGED")
    source_pre_identity = {
        "before": source_pre_before,
        "after": source_after_hashes,
        "input_copy_canonical_sha256": input_hash,
        "PRE_current_before": pre_current_before,
        "PRE_current_after": str(view.current_content_sha256),
        "current_PRE_field_identity": dict(storage_protocol_pre_fields or {}),
        "source_unchanged": True,
        "PRE_unchanged": source_pre_before.get("PRE") == source_after_hashes.get("PRE"),
    }
    _require(source_pre_identity["PRE_unchanged"], "G2_PRE_IDENTITY_CHANGED", source_pre_identity)

    # Persist ordinary scientific outputs before the application clock ends.
    import g2_hpc_exact as exact

    normal_output = _write_normal_outputs(
        bundle,
        propagation_directory,
        work,
        role=args.role,
        raw_diagnostics=result[2] if isinstance(result, (tuple, list)) and len(result) >= 3 else None,
    )
    if fullsize:
        _require(
            int(normal_output.get("scalar_diagnostic_field_count", -1)) == exact.EXPECTED_TERMINAL_SCALARS,
            "G2_CURRENT_SCALAR_DIAGNOSTIC_COUNT",
            normal_output.get("scalar_diagnostic_field_count"),
        )
    application_finished = time.perf_counter()

    # Research archive and native metadata are deliberately outside the
    # application service span.  Capture native identity before save_tree
    # makes any archive copies.
    native_live = exact.native_metadata(bundle, pointer=True)
    output_tree = Path(save_tree(work / "OUTPUT", bundle, compressed=True))
    exact.write_json(work / "CURRENT_NATIVE_OUTPUT_METADATA.json", native_live)
    exact.write_json(work / "CURRENT_OUTPUT_FINGERPRINTS.json", fingerprints(bundle))
    exact.save_file_index(
        work / "OUTPUT_RAW_INDEX.json",
        {
            "OUTPUT": output_tree,
            "PROPAGATION": propagation_directory,
            "SCIENTIFIC_QACC": normal_output["scientific_Qacc"],
            "SCIENTIFIC_SCALAR_DIAGNOSTICS": normal_output["scientific_scalar_diagnostics"],
        },
    )

    identity = {
        "role": args.role,
        "mode": args.mode,
        "fullsize": fullsize,
        "plan": str(Path(args.plan).resolve()),
        "plan_sha256": os.environ.get("G2_HPC_PLAN_SHA256"),
        "implementation_sha256": plan.get("harness_implementation_sha256"),
        "frozen_A07_composite_sha256": "d8e610109a7d666f2b8d9bd4b225a6cd75d34462847dc89d2a4558020d642e6d",
        "source_shape": [int(item) for item in source.shape],
        "source_dtype": str(source.dtype),
    }

    boundary_receipt = None if lease is None else lease.result()
    if args.mode == "QUALIFY_BOUNDARY":
        _require(isinstance(boundary_receipt, dict), "G2_BOUNDARY_RECEIPT_MISSING")
        _require(int(boundary_receipt.get("field_count", -1)) == 112, "G2_BOUNDARY_FIELD_COUNT", boundary_receipt)
        mode_capture = {
            "enabled": True,
            "lease": "retired_after_one_capture",
            "receipt": boundary_receipt,
            "field_count": int(boundary_receipt.get("field_count", -1)),
        }
    else:
        _require(lease is None and boundary_receipt is None, "G2_PURE_OFF_CAPTURE_FORBIDDEN")
        mode_capture = {"enabled": False, "lease": "none", "receipt": None, "field_count": 0}

    comparisons: dict[str, Any] = {}
    if args.reference_output is not None:
        comparisons["terminal_current_pair"] = exact.compare_full_output(
            Path(args.reference_output).resolve(strict=True),
            bundle,
            work / "FULL_EXACT_COMPARISONS" / "CURRENT_TERMINAL.jsonl",
        )
        _require(comparisons["terminal_current_pair"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS", "STOP_SCIENCE_NONEXACT", comparisons["terminal_current_pair"])
    elif args.mode == "PURE_OFF_REAL8" or args.role == "G_CONTROLLED_INPUT":
        _require(False, "G2_REFERENCE_OUTPUT_REQUIRED")

    if args.reference_boundary is not None:
        _require(args.mode == "QUALIFY_BOUNDARY" and boundary_receipt is not None, "G2_REFERENCE_BOUNDARY_MODE")
        comparisons["current_boundary_pair"] = exact.compare_current_boundary_pair(
            args.reference_boundary,
            boundary_receipt,
            work / "FULL_EXACT_COMPARISONS" / "CURRENT_BOUNDARY_PAIR.jsonl",
        )
    elif args.role == "G_CONTROLLED_INPUT" and args.mode == "QUALIFY_BOUNDARY":
        _require(False, "G2_REFERENCE_BOUNDARY_REQUIRED")

    if fullsize and args.mode == "QUALIFY_BOUNDARY":
        current_boundary = exact.load_boundary_capture(boundary_receipt)
        comparisons["historical_boundary"] = _historical_compare(root, plan, current_boundary, work, identity)
    elif fullsize:
        comparisons["historical_boundary"] = {"status": "NOT_REQUESTED_PURE_OFF"}
    else:
        comparisons["historical_boundary"] = {"status": "SKIPPED_TINY_LOCAL_NO_HISTORICAL_ORACLE" if tiny else "NOT_REQUESTED_PURE_OFF"}
    comparisons["historical_terminal"] = (
        _historical_terminal_compare(
            root,
            plan,
            bundle,
            work,
            identity,
            storage_protocol,
            storage_protocol_pre_fields,
        )
        if fullsize
        else None
    )

    _require(len(bundle.get("POST", {})) == INTERVALS, "G2_POST_INTERVAL_COUNT")
    required_sinks = 6 if fullsize else 4
    _require(len(bundle.get("sinks", {})) >= required_sinks, "G2_SCIENTIFIC_SINK_COUNT", {
        "required": required_sinks,
        "actual": len(bundle.get("sinks", {})),
    })
    z_axis = bundle["returned"][2]["z_axis"]
    _require(len(z_axis) == INTERVALS, "G2_CONTIGUOUS_Z_AXIS", len(z_axis))
    receipt = {
        "status": "PASS_G2_HPC_ROLE_EXACT",
        "role": args.role,
        "mode": args.mode,
        "intervals": INTERVALS,
        "fullsize": fullsize,
        "tiny_local_only": tiny,
        "normal_output": normal_output,
        "output_tree": str(output_tree),
        "application_service_s": float(application_finished - application_started),
        "application_clock_scope": "one contiguous perf_counter span from before bootstrap through ordinary final_E/scientific_Qacc/final_POST/diagnostics persistence; source/ownership checks, fresh PRE/sinks, propagation, finish, and mandatory post-propagation source/PRE immutability checks included; native metadata, research archive, comparison-bundle fingerprints, raw index and exact comparisons after stop",
        "propagation_wall_s": float(propagation_wall_s),
        "propagation_process_cpu_s": float(propagation_process_cpu_s),
        "controlled_segment_wall_s": float(sum(float(row["wall_s"]) for row in clock.rows)),
        "controlled_segment_process_cpu_s": float(sum(float(row["process_cpu_s"]) for row in clock.rows)),
        "heun_core_wall_s": float(sum(float(row["wall_s"]) for row in clock.half_rows)),
        "heun_core_process_cpu_s": float(sum(float(row["process_cpu_s"]) for row in clock.half_rows)),
        "clock_rows": copy.deepcopy(clock.rows),
        "clock_half_rows": copy.deepcopy(clock.half_rows),
        "heun_rows": copy.deepcopy(clock.half_rows),
        "route": {**route, "contract": route_contract},
        "FFT_transform_counts": dict(fft_delta),
        "FFT_actual_workers": workers,
        "mode_capture": mode_capture,
        "source/PRE_after_identity": source_pre_identity,
        "real_input_identity": loader_identity,
        "loader_receipt": loader_identity.get("loader_receipt"),
        "continuous_z_axis": z_axis.tolist() if hasattr(z_axis, "tolist") else list(z_axis),
        "G2_lifecycle_audit": g2_audit.as_dict(),
        "runtime": runtime,
        "observer_state": observer_state,
        "blas_pools": blas_pools,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "numba": numba.__version__,
            "llvmlite": llvmlite.__version__,
        },
        "binding": binding,
        "storage_protocol": dict(storage_protocol or {}),
        "storage_protocol_pre_fields": dict(storage_protocol_pre_fields or {}),
        "storage_bridge_contract": None if plan.get("storage_bridge_contract") is None else dict(plan["storage_bridge_contract"]),
        "current_native_output_metadata": "CURRENT_NATIVE_OUTPUT_METADATA.json",
        "current_native_pointer_policy": "pointers recorded for current identity; pointer excluded from current pair comparison",
        "historical_native_alias_layout": "UNOBSERVED_IN_ORIGINAL_HISTORY",
        "exact": comparisons,
        "historical_boundary": comparisons["historical_boundary"],
        "historical_terminal": comparisons["historical_terminal"],
        "source_identity": identity,
        "batch": args.batch,
        "order": args.order,
        **_allocation_identity(),
        "research_only": {
            "fingerprints": "CURRENT_OUTPUT_FINGERPRINTS.json",
            "raw_index": "OUTPUT_RAW_INDEX.json",
            "qualification_clock_not_performance": args.mode == "QUALIFY_BOUNDARY",
        },
        "production_qualified": False,
        "review_gate": "MANUAL_REVIEW_PENDING",
    }
    common.write(work / "CASE_RECEIPT.json", receipt)
    numba.set_num_threads(24)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-S4-3G2 private HPC role harness")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--reference-boundary", type=Path)
    parser.add_argument("--batch")
    parser.add_argument("--order", type=int)
    parser.add_argument("--tiny", action="store_true", help="local synthetic context only; never a real-size qualification")
    parser.add_argument("--entry-smoke", action="store_true", help="construct the approved role without scientific propagation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except BaseException as exc:
        target = Path(args.work).resolve()
        target.mkdir(parents=True, exist_ok=True)
        failure = target / "FAILURE.json"
        if not failure.exists():
            with failure.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump({
                    "status": "FAIL_G2_HPC_ROLE",
                    "exception_type": type(exc).__name__,
                    "reason": str(exc),
                    "traceback": traceback.format_exc(),
                    **_allocation_identity(),
                }, stream, indent=2)
                stream.write("\n")
        raise
    return 0


if __name__ == "__main__":
    main()


__all__ = ["ROLES", "MODES", "build_parser", "run", "main"]
