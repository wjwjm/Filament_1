"""Serial full-size C capture/mechanism controller."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import getpass
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_5b_steady_cost.hpc import command, seal_reference_files
from tools.cpu_o1.cpu_s4.memory_limits import snapshot_cgroup_memory
from tools.cpu_s4_fullsize_confirmation.controller import _tree_rss
from tools.hpc_ops import provenance_v2 as pv

from . import c_fullsize_entry
from . import hpc as full_hpc
from .c_fullsize_evidence import _gain_map, validate_mechanism_evidence
from .c_fullsize_protocol import EVIDENCE_SCHEMA, EXECUTION_SCHEMA, FORMAL_CALLS, MICRO_SHAPE, REMOTE_PYTHON, RESOURCES, RESULT_SCHEMA, SHAPE, TASK, THREAD_ENV, VERSIONS, WARMUP_CALLS, plan


def _require(ok: bool, reason: str, detail=None) -> None:
    if not ok:
        raise RuntimeError(reason if detail is None else f"{reason}:{detail}")


def _fresh(report: Path) -> None:
    for name in ("RESULT.json", "JOB_FAILURE.json", "NODE_GATE.json", "runs", "evidence_C_fullsize_mechanism"):
        _require(not (report / name).exists(), "S46F_C_FULL_CONTROLLER_ALREADY_EXECUTED", name)


def preflight(report: str | Path) -> dict:
    """Read-only source, interpreter, batch, and factory audit."""

    from tools.hpc_ops.audit_batch_entry import audit_batch_entry

    report = Path(report).resolve(strict=True)
    _require(getpass.getuser() == "t0s000727", "S46F_C_FULL_T0_ONLY")
    _require(Path(sys.executable).absolute() == Path(REMOTE_PYTHON), "S46F_C_FULL_FIXED_INTERPRETER")
    _require(sys.version_info[:3] == (3, 12, 8), "S46F_C_FULL_PYTHON_PATCH")
    versions = {name: importlib.import_module(name).__version__ for name in VERSIONS}
    _require(versions == VERSIONS, "S46F_C_FULL_FROZEN_DEPENDENCIES", versions)
    _require(os.environ.get("UPPE_USE_GPU") == "0" and all(os.environ.get(key) == value for key, value in THREAD_ENV.items()), "S46F_C_FULL_THREAD_ENV")
    binding = c_fullsize_entry._verify_mechanism_source(report, local_engineering=False, shape=SHAPE)
    repo = Path(__file__).resolve().parents[3]
    pv.validate_manifest(repo, report / "EXECUTION_SOURCE_MANIFEST.json", require_clean=True, require_hash_scope=True)
    audit, code = audit_batch_entry(Path(__file__).with_name("c_fullsize_cpu.sbatch"), REMOTE_PYTHON, "bash")
    _require(code == 0 and audit.get("status") == "passed", "S46F_C_FULL_BATCH_ENTRY_AUDIT", audit)
    help_result = command([sys.executable, "-s", "-B", "-m", "tools.cpu_s4_6f_fullsize_parallel.c_fullsize_entry", "--help"])
    from .role_spec import build_role, close_role, snapshot_role, verify_role

    factories = {}
    for role in ("R", "X"):
        root = report / f"UNUSED_C_FULL_PREFLIGHT_{role}"
        _require(not root.exists(), "S46F_C_FULL_PREFLIGHT_CAPTURE_ROOT_PREEXISTS", str(root))
        built = build_role("PHASE_ABSORB_P24", "PURE_OFF_FULLSIZE", capture_root=root, role=role, local_engineering=False, shape=SHAPE, recorder_pair=None, capture_store=None)
        try:
            checked = verify_role(built[1], "PHASE_ABSORB_P24")
        finally:
            close_role(built[1], "PHASE_ABSORB_P24")
        snapshot = snapshot_role(built[1], "PHASE_ABSORB_P24")
        _require(snapshot.get("closed") is True and snapshot.get("active_thread_ids") == [], "S46F_C_FULL_PREFLIGHT_FACTORY_CLOSED")
        _require(not root.exists(), "S46F_C_FULL_PREFLIGHT_CAPTURE_ROOT_CREATED", str(root))
        factories[role] = dict(binding=checked, post_close_pool=snapshot)
    return dict(schema=EXECUTION_SCHEMA + ".preflight.v1", status="PASS_S46F_C_FULL_MECHANISM_READONLY_PREFLIGHT", execution_commit=binding["execution_commit"], executable=sys.executable, versions=versions, batch_audit=audit, entry_help_checked=bool(help_result), factories=factories, propagation_called=False)


def node_gate(report: str | Path) -> dict:
    import psutil

    report = Path(report).resolve(strict=True)
    checked = preflight(report)
    authorization_path = report / "RESOURCE_AUTHORIZATION.json"
    _require(authorization_path.is_file(), "S46F_C_FULL_RESOURCE_AUTHORIZATION_MISSING")
    authorization = sg.read(authorization_path)
    _require(authorization.get("authorized") is True and authorization.get("execution_commit") == checked["execution_commit"] and authorization.get("resources") == RESOURCES and bool(authorization.get("user_instruction")), "S46F_C_FULL_RESOURCE_AUTHORIZATION")
    job = os.environ.get("SLURM_JOB_ID", "")
    _require(job.isdigit() and os.environ.get("SLURM_RESTART_COUNT", "0") == "0", "S46F_C_FULL_ONE_NEW_NON_REQUEUED_JOB")
    _require(os.environ.get("SLURM_CPUS_PER_TASK") == "96" and os.environ.get("SLURM_JOB_NUM_NODES") == "1", "S46F_C_FULL_NODE_CPU96")
    description = command(["scontrol", "show", "job", job, "-o"])
    fields = dict(word.split("=", 1) for word in description.split() if "=" in word)
    _require(fields.get("NumCPUs") == "96" and fields.get("NumNodes") == "1" and fields.get("TimeLimit") == "01:00:00" and fields.get("Account") == "bscc-t" and fields.get("Partition") == "v6_384", "S46F_C_FULL_EXACT_RESOURCES", description)
    tres = dict(word.split("=", 1) for word in fields.get("AllocTRES", "").split(",") if "=" in word)
    _require(tres.get("mem") in ("128G", "131072M") and "gres/gpu" not in description and not os.environ.get("SLURM_JOB_GPUS"), "S46F_C_FULL_128G_ZERO_GPU")
    affinity = psutil.Process().cpu_affinity()
    hardware = command(["lscpu"])
    topology = command(["lscpu", "-p=CPU,CORE,SOCKET,NODE"])
    topology_receipt = full_hpc.validate_physical_topology(hardware, topology, affinity)
    memory = snapshot_cgroup_memory(os.getpid(), allocation_bytes=128 * 1024**3)
    _require(memory.get("status") == "READ" and int(memory.get("effective_limit_bytes", 0)) > 0, "S46F_C_FULL_ACTUAL_CGROUP")
    guard = int(0.8 * min(128 * 1024**3, memory["effective_limit_bytes"]))
    _require(guard > 80 * 1024**3, "S46F_C_FULL_MEMORY_GUARD")
    leaf = Path(memory["levels_leaf_to_mount_root"][0]["path"])
    peak_path = leaf / ("memory.max_usage_in_bytes" if memory["version"] == 1 else "memory.peak")
    _require(int(peak_path.read_text().strip()) < guard, "S46F_C_FULL_INITIAL_MEMORY_GUARD")
    sealed = full_hpc._seal_canonical_files(report)
    refs = seal_reference_files(report)
    node = dict(schema=EXECUTION_SCHEMA + ".node_gate.v1", status="PASS_S46F_C_FULL_MECHANISM_SOURCE_RESOURCE_INPUT_GATES", job_id=job, preflight=checked, allocation=description, hardware=hardware, topology=topology, **topology_receipt, memory=memory, process_tree_stop_bytes=guard, memory_peak_path=str(peak_path), sealed_canonical_inputs=sealed, sealed_reference_files=refs, default="I0", production_qualified=False)
    sg.write(report / "NODE_GATE.json", node)
    return node


def _supervise_process(process, *, node: dict, deadline: float, receipt_path: Path) -> dict:
    import psutil

    from tools.cpu_o1.cpu_s4.supervisor import terminate

    created = psutil.Process(process.pid).create_time()
    started = time.monotonic()
    peak = kernel_peak = samples = 0
    reason = None
    limit, peak_path = node["process_tree_stop_bytes"], Path(node["memory_peak_path"])
    try:
        while process.poll() is None:
            child = _tree_rss(process)
            if child < 0:
                reason = "PROCESS_IDENTITY_LOST"
                break
            peak = max(peak, child + psutil.Process().memory_info().rss)
            kernel_peak = max(kernel_peak, int(peak_path.read_text().strip()))
            samples += 1
            if max(peak, kernel_peak) >= limit:
                reason = "ACTUAL_MEMORY_GUARD_80_PERCENT"
                break
            if time.monotonic() >= deadline:
                reason = "WORK_DEADLINE_3480_SECONDS"
                break
            time.sleep(1)
    finally:
        if process.poll() is None:
            terminate(process.pid, created)
    code = process.wait(timeout=30)
    finished = time.monotonic()
    kernel_peak = max(kernel_peak, int(peak_path.read_text().strip()))
    if reason is None and kernel_peak >= limit:
        reason = "ACTUAL_MEMORY_GUARD_80_PERCENT_FINAL"
    if reason is None and finished >= deadline:
        reason = "WORK_DEADLINE_3480_SECONDS_FINAL"
    receipt = dict(exit_code=code, stop_reason=reason, process_service_s=finished - started, peak_tree_rss_bytes=peak, kernel_cgroup_peak_bytes=kernel_peak, memory_samples=samples)
    sg.write(receipt_path, receipt)
    return receipt


def _launch(report: Path, *, case: str, capture: bool, reference_output: str | None, deadline: float, node: dict, input_manifest: Path) -> dict:
    folder = report / "runs" / "C_FULLSIZE" / case
    folder.mkdir(parents=True, exist_ok=False)
    work = folder / "work"
    argv = [sys.executable, "-s", "-B", "-m", "tools.cpu_s4_6f_fullsize_parallel.c_fullsize_entry", "--report", str(report), "--work", str(work), "--role", "R", "--intervals", "8", "--input-manifest", str(input_manifest)]
    if capture:
        argv.extend(("--capture", "--capture-root", str(folder / "CAPTURE")))
    if reference_output:
        argv.extend(("--reference-output", reference_output))
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env["NUMBA_CACHE_DIR"] = str(report / "cache" / "C_FULLSIZE" / case)
    sg.write(folder / "LAUNCH.json", dict(argv=argv, case=case, role="R", intervals=8, capture=capture, UTC=dt.datetime.now(dt.timezone.utc).isoformat(), warmup_intervals=0, factor_cache="EMPTY_PRIVATE_NEW_PROCESS", JIT_cache=env["NUMBA_CACHE_DIR"]))
    with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    receipt = _supervise_process(process, node=node, deadline=deadline, receipt_path=folder / "SUPERVISOR_RECEIPT.json")
    failure = work / "ENTRY_FAILURE.json"
    failure_value = c_fullsize_entry.read(failure) if failure.is_file() else None
    classification = full_hpc._child_failure_classification(receipt, failure_value)
    if classification == "SCIENTIFIC_NONEXACT":
        return dict(schema=RESULT_SCHEMA + ".case_failure.v1", status="SCIENTIFIC_NONEXACT", case_id=case, execution_domain="S46F_FULLSIZE", shape=list(SHAPE), entry_failure=failure_value, supervisor=receipt)
    if classification == "ENGINEERING_OR_RESOURCE_FAILURE":
        raise RuntimeError(f"S46F_C_FULL_CHILD_FAILURE:{receipt}")
    result = c_fullsize_entry.read(work / "CASE_RECEIPT.json")
    result.update(case_id=case, supervisor=receipt, work_root=str(work), reference_output=reference_output)
    _require(result.get("execution_domain") == "S46F_FULLSIZE" and result.get("shape") == list(SHAPE), "S46F_C_FULL_CASE_DOMAIN")
    return result


def _launch_mechanism(report: Path, *, capture_root: Path, output: Path, deadline: float, node: dict, input_manifest: Path) -> dict:
    folder = output.parent
    folder.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, "-s", "-B", "-m", "tools.cpu_s4_6f_fullsize_parallel.c_fullsize_entry", "--report", str(report), "--mechanism", "--capture-root", str(capture_root), "--output", str(output), "--input-manifest", str(input_manifest)]
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env["NUMBA_CACHE_DIR"] = str(report / "cache" / "C_FULLSIZE" / "MECHANISM")
    sg.write(folder / "LAUNCH.json", dict(argv=argv, case="MECHANISM", UTC=dt.datetime.now(dt.timezone.utc).isoformat(), capture=False, factor_cache="NONE_OFFLINE_MECHANISM", JIT_cache=env["NUMBA_CACHE_DIR"]))
    with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    receipt = _supervise_process(process, node=node, deadline=deadline, receipt_path=folder / "SUPERVISOR_RECEIPT.json")
    failure = output.parent / "ENTRY_FAILURE.json"
    failure_value = c_fullsize_entry.read(failure) if failure.is_file() else None
    classification = full_hpc._child_failure_classification(receipt, failure_value)
    if classification == "SCIENTIFIC_NONEXACT":
        if output.is_file():
            return c_fullsize_entry.read(output)
        return dict(schema=RESULT_SCHEMA, status="SCIENTIFIC_NONEXACT", execution_domain="S46F_FULLSIZE", entry_failure=failure_value, supervisor=receipt, production_qualified=False, default="I0")
    if classification == "ENGINEERING_OR_RESOURCE_FAILURE":
        raise RuntimeError(f"S46F_C_FULL_MECHANISM_CHILD_FAILURE:{receipt}")
    return c_fullsize_entry.read(output)


def _write_evidence(report: Path, *, node: dict, uncaptured: dict, captured: dict, mechanism: dict) -> Path:
    root = report / "evidence_C_fullsize_mechanism"
    root.mkdir(parents=True, exist_ok=False)
    for name in ("EXECUTION.json", "EXECUTION_SOURCE_MANIFEST.json", "INPUT_CONTRACT.json", "FULL_INPUT_MANIFEST.json", "REFERENCE_INPUT_RAW_RECORDS.json"):
        shutil.copyfile(report / name, root / name)
    sg.write(root / "RESULT.json", mechanism)
    captured_receipt = c_fullsize_entry.read(Path(captured["work_root"]) / "CAPTURE_RECEIPT.json") if captured.get("work_root") else {}
    capture_package_path = Path(captured_receipt["capture_package"]).resolve(strict=True) if captured_receipt.get("capture_package") else None
    capture_receipt = dict(
        schema="filament.cpu_s4_6f.c_fullsize_capture_receipt.v1",
        shape=list(SHAPE),
        selected_interval_indices=[0, 3, 7],
        expected_total_calls=8,
        historical_terminal_status=uncaptured.get("comparisons", {}).get("terminal", {}).get("status"),
        uncaptured_terminal_status=uncaptured.get("comparisons", {}).get("terminal", {}).get("status"),
        captured_terminal_status=captured.get("comparisons", {}).get("terminal", {}).get("status"),
        historical_reference_output=uncaptured.get("reference_output"),
        captured_reference_output=captured.get("reference_output"),
        captured_reference_is_uncaptured_output=captured.get("reference_output") == uncaptured.get("output"),
        diagnostic_outputs=captured.get("diagnostic_outputs") is True,
        cpu_diagnostics_profile=captured.get("cpu_diagnostics_profile") is True,
        fine_performance_profile=captured.get("fine_performance_profile") is True,
        source_binding=captured_receipt.get("source_binding", {}),
        capture_metadata=captured_receipt.get("capture_metadata", {}),
        raw_files=captured_receipt.get("raw_files", []),
        capture_package_manifest=None if capture_package_path is None else dict(path=str(capture_package_path / "CAPTURE.json"), raw_sha256=pv.raw_sha256_file(capture_package_path / "CAPTURE.json"), size_bytes=(capture_package_path / "CAPTURE.json").stat().st_size, classification="external", hash_scope="raw_bytes"),
        uncaptured_output=uncaptured.get("output"),
        captured_output=captured.get("output"),
    )
    sg.write(root / "CAPTURE_RECEIPT.json", capture_receipt)
    sg.write(root / "RESOURCE_RECEIPT.json", dict(physical_cpus=96, memory_GiB=128, gpus=0, terminal=False, node_gate_status=node.get("status")))
    scheduler = dict(job_id=node.get("job_id"), scheduler_terminal_state="AWAIT_SACCT", exit_code=None, allocated_cpus=96, requested_memory="128G", gpus=0)
    sg.write(root / "SCHEDULER_ACCOUNTING.json", scheduler)
    admission = dict(schema=EVIDENCE_SCHEMA, status="PENDING_SCHEDULER_TERMINAL", execution_commit=mechanism.get("execution_commit"), shape=list(SHAPE), selected_interval_indices=[0, 3, 7], expected_total_calls=8, formal_calls=mechanism.get("mechanism", {}).get("formal_calls"), warmup_calls=mechanism.get("mechanism", {}).get("warmup_calls"), p24_block_gains=mechanism.get("mechanism", {}).get("p24_block_gains"), source_manifest_raw_sha256=pv.raw_sha256_file(root / "EXECUTION_SOURCE_MANIFEST.json"), input_contract_raw_sha256=pv.raw_sha256_file(root / "INPUT_CONTRACT.json"), full_input_manifest_raw_sha256=pv.raw_sha256_file(root / "FULL_INPUT_MANIFEST.json"), reference_contract_raw_sha256=pv.raw_sha256_file(root / "REFERENCE_INPUT_RAW_RECORDS.json"))
    sg.write(root / "ADMISSION.json", admission)
    return root


def finalize_evidence(evidence_root: str | Path, scheduler: dict) -> dict:
    """Attach terminal scheduler evidence and perform the final strict gate."""

    root = Path(evidence_root).resolve(strict=True)
    _require(scheduler.get("scheduler_terminal_state") == "COMPLETED" and scheduler.get("exit_code") == "0:0", "S46F_C_FULL_SCHEDULER_NOT_TERMINAL")
    # Pending receipts are immutable: append terminal receipts instead of
    # reopening the exclusive-write files created before sacct is available.
    sg.write(root / "SCHEDULER_ACCOUNTING_TERMINAL.json", scheduler)
    sg.write(root / "RESOURCE_RECEIPT_TERMINAL.json", dict(physical_cpus=96, memory_GiB=128, gpus=0, terminal=True, scheduler_terminal_state="COMPLETED"))
    result = c_fullsize_entry.read(root / "RESULT.json")
    mechanism = result.get("mechanism", {})
    admission = c_fullsize_entry.read(root / "ADMISSION.json")
    admission.update(status="PASS_C_FULLSIZE_MECHANISM_EXACT_GAIN_GATE", execution_commit=result.get("execution_commit"), formal_calls=mechanism.get("formal_calls"), warmup_calls=mechanism.get("warmup_calls"), p24_block_gains=_gain_map(result), source_manifest_raw_sha256=pv.raw_sha256_file(root / "EXECUTION_SOURCE_MANIFEST.json"))
    sg.write(root / "ADMISSION_TERMINAL.json", admission)
    return validate_mechanism_evidence(root)


def run(report: str | Path) -> dict:
    report = Path(report).resolve(strict=True)
    _fresh(report)
    try:
        return _run_fresh(report)
    except BaseException as error:
        failure = report / "JOB_FAILURE.json"
        if not failure.exists():
            sg.write(failure, dict(schema=RESULT_SCHEMA + ".failure.v1", classification="SCIENTIFIC_NONEXACT" if "NONEXACT" in repr(error) else "ENGINEERING_OR_RESOURCE_FAILURE", error=repr(error), traceback=traceback.format_exc(), default="I0", production_qualified=False, automatic_retry=False))
        raise


def _run_fresh(report: Path) -> dict:
    report = Path(report).resolve(strict=True)
    started = time.monotonic()
    node = node_gate(report)
    input_manifest = report / "FULL_INPUT_MANIFEST.json"
    _require(input_manifest.is_file(), "S46F_C_FULL_INPUT_MANIFEST_MISSING")
    deadline = started + int(RESOURCES["work_deadline_s"])
    reference = c_fullsize_entry.read(report / "REFERENCE_INPUT_RAW_RECORDS.json")["reference"]
    _require(reference.get("output"), "S46F_C_FULL_REFERENCE_OUTPUT_REQUIRED")
    cases = {}
    uncaptured = _launch(report, case="R8_UNCAPTURED", capture=False, reference_output=reference["output"], deadline=deadline, node=node, input_manifest=input_manifest)
    cases["R8_UNCAPTURED"] = uncaptured
    if uncaptured.get("status") == "SCIENTIFIC_NONEXACT":
        result = dict(schema=RESULT_SCHEMA, status="SCIENTIFIC_NONEXACT", cases=cases, default="I0", production_qualified=False)
        sg.write(report / "RESULT.json", result)
        return result
    captured = _launch(report, case="R8_CAPTURED", capture=True, reference_output=uncaptured["output"], deadline=deadline, node=node, input_manifest=input_manifest)
    cases["R8_CAPTURED"] = captured
    if captured.get("status") == "SCIENTIFIC_NONEXACT":
        result = dict(schema=RESULT_SCHEMA, status="SCIENTIFIC_NONEXACT", cases=cases, default="I0", production_qualified=False)
        sg.write(report / "RESULT.json", result)
        return result
    mechanism = _launch_mechanism(report, capture_root=report / "runs" / "C_FULLSIZE" / "R8_CAPTURED" / "CAPTURE", output=report / "runs" / "C_FULLSIZE" / "MECHANISM" / "RESULT.json", deadline=deadline, node=node, input_manifest=input_manifest)
    cases["MECHANISM"] = mechanism
    evidence_root = _write_evidence(report, node=node, uncaptured=uncaptured, captured=captured, mechanism=mechanism)
    result = dict(schema=RESULT_SCHEMA, status=mechanism.get("status"), cases=cases, mechanism=mechanism.get("mechanism"), evidence_root=str(evidence_root), resource_ledger=dict(job_id=node.get("job_id"), jobs_used=1, retry_authorized=False, controller_service_s=time.monotonic() - started, allocated_core_hours_max=RESOURCES["allocated_core_hours_max"], scheduler_terminal_state="AWAIT_SACCT"), default="I0", production_qualified=False, review_gate="MANUAL_REVIEW_PENDING")
    sg.write(report / "RESULT.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    value = preflight(args.report.resolve(strict=True)) if args.preflight else run(args.report)
    print(json.dumps(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
