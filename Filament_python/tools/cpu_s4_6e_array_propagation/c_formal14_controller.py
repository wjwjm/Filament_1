"""Serial C14 small-grid controller for one authorized CPU allocation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.hpc_ops.audit_batch_entry import audit_batch_entry

from . import c_entry, c_hpc
from .c_formal14_analysis import analyze
from .c_formal14_evidence import validate_mechanism_evidence
from .c_formal14_protocol import MATRIX, REMOTE_PYTHON, RESOURCES, SHAPE, THREAD_ENV, VERSIONS, plan


def _require(ok: bool, reason: str, detail=None) -> None:
    if not ok:
        raise RuntimeError(reason if detail is None else f"{reason}:{detail}")


def _mechanism_gate(report: Path) -> dict:
    admission = c_entry.read(report / "MECHANISM_ADMISSION.json")
    _require(admission.get("status") == "PASS_C14_MECHANISM_ADMISSION", "C14_MECHANISM_GATE_MISSING")
    _require(admission.get("mechanism_execution_commit") == plan()["mechanism_execution_commit"], "C14_MECHANISM_EXECUTION_MISMATCH")
    _require(int(admission.get("formal_calls", -1)) == 90 and int(admission.get("warmup_calls", -1)) == 36 and admission.get("all_configs_exact") is True, "C14_MECHANISM_GATE_COUNTS_OR_EXACT")
    gains = admission.get("p24_block_gains")
    _require(isinstance(gains, dict) and len(gains) == 6 and all(float(value) >= 0.05 for value in gains.values()), "C14_MECHANISM_GATE_GAINS")
    verified = validate_mechanism_evidence(report / "evidence_C_mechanism", repo=Path(__file__).resolve().parents[3])
    for key in ("mechanism_execution_commit", "mechanism_result_raw_sha256", "mechanism_source_manifest_raw_sha256", "formal_calls", "warmup_calls", "all_configs_exact", "p24_block_gains"):
        _require(verified.get(key) == admission.get(key), "C14_MECHANISM_ADMISSION_REPLAY_MISMATCH", key)
    return admission


def _own_batch_gate() -> dict:
    audit, code = audit_batch_entry(Path(__file__).with_name("c_formal14_cpu.sbatch"), REMOTE_PYTHON, "bash")
    _require(code == 0 and audit.get("status") == "passed", "C14_BATCH_ENTRY_AUDIT", audit)
    return audit


def preflight(report: str | Path) -> dict:
    """Audit C14 source, mechanism admission, and its own batch entry."""

    report = Path(report).resolve(strict=True)
    admission = _mechanism_gate(report)
    audit = _own_batch_gate()
    common = c_hpc.preflight(report)
    return dict(status="PASS_C14_SOURCE_MECHANISM_BATCH_PREFLIGHT", execution_commit=common["execution_commit"], preflight=common, mechanism=admission, batch_audit=audit, propagation_called=False)


def _launch_case(report: Path, *, case: str, role: str, intervals: int, kind: str, reference: dict | None, deadline: float, node: dict) -> dict:
    _require(time.monotonic() < deadline, "C14_DEADLINE_BEFORE_NEXT_WINDOW")
    folder = report / "runs/C14" / case
    folder.mkdir(parents=True, exist_ok=False)
    work = folder / "work"
    argv = [sys.executable, "-s", "-B", "-m", "tools.cpu_s4_6e_array_propagation.c_entry", "--report", str(report), "--work", str(work), "--role", role, "--intervals", str(intervals), "--input-manifest", str(report / "BENCHMARK_INPUT_MANIFEST.json")]
    if reference:
        if reference.get("output"):
            argv.extend(("--reference-output", reference["output"]))
        if intervals == 2:
            if reference.get("boundary_path"):
                argv.extend(("--reference-boundary", reference["boundary_path"]))
            if reference.get("stages"):
                argv.extend(("--reference-stages", reference["stages"]))
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env["NUMBA_CACHE_DIR"] = str(report / "cache" / role)
    sg.write(folder / "LAUNCH.json", dict(argv=argv, case=case, role=role, intervals=intervals, kind=kind, capture=False, UTC=dt.datetime.now(dt.timezone.utc).isoformat(), warmup_intervals=0, factor_cache="EMPTY_PRIVATE_NEW_PROCESS", JIT_cache=env["NUMBA_CACHE_DIR"]))
    with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    try:
        receipt = c_hpc._supervise_process(process, node=node, deadline=deadline, receipt_path=folder / "SUPERVISOR_RECEIPT.json")
    except BaseException:
        failure_path = work / "ENTRY_FAILURE.json"
        if failure_path.is_file():
            failure = c_entry.read(failure_path)
            if failure.get("classification") == "SCIENTIFIC_NONEXACT":
                return dict(schema="filament.cpu_s4_6e_c14.case_failure.v1", case_id=case, role=role, intervals=intervals, status="SCIENTIFIC_NONEXACT", execution_domain="S46E_SMALLGRID", shape=list(SHAPE), entry_failure=failure, supervisor=c_entry.read(folder / "SUPERVISOR_RECEIPT.json"))
        raise
    result = c_entry.read(work / "CASE_RECEIPT.json")
    result.update(case_id=case, kind=kind, supervisor=receipt)
    _require(result.get("execution_domain") == "S46E_SMALLGRID" and result.get("shape") == list(SHAPE), "C14_CASE_DOMAIN")
    return result


def _exact(result: dict, *, terminal: bool, boundary: bool, stages: bool) -> None:
    comparisons = result.get("comparisons", {})
    if terminal:
        _require(comparisons.get("terminal", {}).get("status") == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS", "C14_STOP_TERMINAL_NONEXACT")
    if boundary:
        _require(comparisons.get("boundary", {}).get("status") == "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT", "C14_STOP_BOUNDARY_NONEXACT")
    if stages:
        _require(comparisons.get("stages", {}).get("status") == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS", "C14_STOP_STAGES_NONEXACT")
    intervals = int(result["intervals"])
    n_h = 2 * intervals
    route = result["route"]
    stats = result["cache"]["stats"]
    _require(route["nH"] == n_h and route["fresh_responses"] == 2 * n_h and route["math_dispatches"] == 8 * n_h and route["fallback"] == 0, "C14_ORIGINAL_ROUTE_COUNTS")
    _require(all(value == n_h for value in route["lifecycle"].values()) and not route.get("active_lifecycle_identity"), "C14_ORIGINAL_RETIRED_CAPABILITIES")
    _require(stats["factor_requests"] == 384 * n_h and stats["plane_builds"] == 384 and stats["hits"] == 384 * (n_h - 1) and result["cache_fallback"] == 0, "C14_CACHE_COUNTS")
    pool = result.get("c_pool")
    _require(isinstance(pool, dict) and pool["pool"]["closed"] and pool["pool"]["active_thread_ids"] == [] and pool["pool"]["pending_scalar_records"] == 0, "C14_C_POOL_LIFECYCLE")
    expected_count = intervals if result["role"] == "C" else 0
    _require(pool["pool"]["operation_counts"] == {"C_PHASE": expected_count, "C_ABSORPTION": expected_count}, "C14_C_POOL_COUNTS")
    activity = pool["activity"]
    _require(len(activity) == 2 * expected_count, "C14_C_POOL_ACTIVITY_COUNT")
    _require(all(row["requested_workers"] == row["actual_workers"] == 24 and sum(worker["items"] for worker in row["workers"]) == 384 for row in activity), "C14_C_POOL_WORKER_PARTITION")
    result["exact_gate_pass"] = True
    result["terminal_exact_basis"] = "COMPARED_CURRENT_WINDOW_TO_REFERENCE" if terminal else "R2_SOURCE_BOUND_BOUNDARY_STAGE_ONLY_NO_HISTORICAL_TWO_INTERVAL_TERMINAL"


def _scientific_stop(report: Path, node: dict, started: float, cases: dict, *, stage: str) -> dict:
    result = dict(schema="filament.cpu_s4_6e_c14.result.v1", status="SCIENTIFIC_NONEXACT", classification="SCIENTIFIC_NONEXACT", scientific_stop_stage=stage, cases=cases, default="I0", production_qualified=False, resource_ledger=dict(job_id=node.get("job_id"), jobs_used=1, retry_authorized=False, controller_service_s=time.monotonic() - started, allocated_core_hours_max=48, scheduler_terminal_state="AWAIT_SACCT"))
    sg.write(report / "RESULT.json", result)
    return result


def run(report: str | Path) -> dict:
    report = Path(report).resolve(strict=True)
    started = time.monotonic()
    cases: dict[str, dict] = {}
    try:
        admission = _mechanism_gate(report)
        batch_audit = _own_batch_gate()
        execution_commit = c_entry.read(report / "EXECUTION.json")["execution_commit"]
        node = c_hpc.node_gate(report)
        sg.write(report / "C14_BATCH_AUDIT.json", batch_audit)
        sg.write(report / "C14_MECHANISM_GATE.json", dict(status="PASS_C14_MECHANISM_GATE", admission=admission, batch_audit=batch_audit, execution_commit=execution_commit))
        inputs = sg.input_manifest(report, shape=SHAPE)
        expected_inputs = c_entry.read(report / "INPUT_CONTRACT.json")["expected_input_fingerprints"]
        _require(inputs["fingerprints"] == expected_inputs, "C14_FROZEN_DIRECT_SMALLGRID_INPUTS")
        deadline = started + 1680
        historical = c_entry.read(report / "REFERENCE_INPUT_RAW_RECORDS.json")["reference"]
        _require(historical.get("boundary_path") and historical.get("stages"), "C14_HISTORICAL_R2_BOUNDARY_STAGE_REQUIRED")
        for case, role, intervals, kind in MATRIX:
            if case == "QUAL_R2":
                reference = dict(boundary_path=historical["boundary_path"], stages=historical.get("stages"))
            elif case == "CTRL_EARLY_R1":
                reference = historical
            elif case == "QUAL_C2":
                reference = cases["QUAL_R2"]
            else:
                reference = cases["CTRL_EARLY_R1"]
            result = _launch_case(report, case=case, role=role, intervals=intervals, kind=kind, reference=reference, deadline=deadline, node=node)
            if result.get("status") == "SCIENTIFIC_NONEXACT":
                return _scientific_stop(report, node, started, cases | {case: result}, stage=case)
            if case == "QUAL_R2":
                _exact(result, terminal=False, boundary=True, stages=True)
            elif case == "QUAL_C2":
                _exact(result, terminal=True, boundary=True, stages=True)
            else:
                _exact(result, terminal=True, boundary=False, stages=False)
            cases[case] = result
            sg.write(report / "runs/C14" / case / "ACCEPTED_CASE.json", result)
            if case == "QUAL_C2":
                sg.write(report / "QUALIFIED_STEP_CONTRACT.json", dict(nH=16, intervals=8, qualification_nH=4, qualification_intervals=2, qualification_fresh_responses=8, qualification_original_math_calls=32, qualification_cases=["QUAL_R2", "QUAL_C2"], r2_historical_terminal_exact=False, r2_source_bound_boundary_stage_exact=True, c2_terminal_exact_against_r2=True, qualification_gate=True))
        result = analyze(cases)
        sg.write(report / "ACCEPTED_CASES.json", cases)
        sg.write(report / "RESULT.json", result)
        sg.write(report / "RESOURCE_LEDGER.json", dict(job_id=node.get("job_id"), jobs_used=1, retry_authorized=False, serial_windows=14, controller_service_s=time.monotonic() - started, allocated_core_hours_max=48, scheduler_terminal_state="AWAIT_SACCT", actual_allocated_core_hours="AWAIT_SACCT"))
        return result
    except BaseException as error:
        sg.write(report / "JOB_FAILURE.json", dict(classification="SCIENTIFIC_NONEXACT" if "NONEXACT" in repr(error) else "ENGINEERING_OR_RESOURCE_FAILURE", error=repr(error), traceback=traceback.format_exc(), completed_cases=list(cases), default="I0", production_qualified=False, automatic_retry=False))
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    value = preflight(args.report.resolve(strict=True)) if args.preflight else run(args.report.resolve(strict=True))
    print(json.dumps(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
