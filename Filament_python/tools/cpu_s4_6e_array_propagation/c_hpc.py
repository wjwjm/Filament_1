"""Read-only preflight and one-allocation C capture/mechanism supervisor."""

from __future__ import annotations

import datetime as dt
import argparse
import getpass
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_5b_steady_cost.hpc import command, seal_reference_files
from tools.cpu_s4_fullsize_confirmation.controller import _tree_rss
from tools.hpc_ops import provenance_v2 as pv

from . import c_entry
from .c_protocol import HPC_CASES, REMOTE_PYTHON, SHAPE, THREAD_ENV, VERSIONS, plan


def _require(ok: bool, reason: str, detail=None) -> None:
    if not ok:
        raise RuntimeError(reason if detail is None else f"{reason}:{detail}")


def preflight(report):
    """Read-only source/environment/batch audit; no run root or scheduler call."""

    from tools.hpc_ops.audit_batch_entry import audit_batch_entry

    _require(getpass.getuser() == "t0s000727", "S46E_C_T0_ONLY")
    _require(Path(sys.executable).absolute() == Path(REMOTE_PYTHON), "S46E_C_FIXED_INTERPRETER")
    _require(sys.version_info[:3] == (3, 12, 8), "S46E_C_PYTHON_PATCH")
    versions = {name: importlib.import_module(name).__version__ for name in VERSIONS}
    _require(versions == VERSIONS, "S46E_C_FROZEN_DEPENDENCIES", versions)
    _require(os.environ.get("UPPE_USE_GPU") == "0" and all(os.environ.get(key) == value for key, value in THREAD_ENV.items()), "S46E_C_FROZEN_CPU_THREAD_ENV")
    binding = c_entry.verify_source(report, local_engineering=False)
    repo = Path(__file__).resolve().parents[3]
    pv.validate_manifest(repo, Path(report) / "EXECUTION_SOURCE_MANIFEST.json", require_clean=True, require_hash_scope=True)
    audit, code = audit_batch_entry(Path(__file__).with_name("c_cpu.sbatch"), REMOTE_PYTHON, "bash")
    _require(code == 0 and audit["status"] == "passed", "S46E_C_BATCH_ENTRY_AUDIT", audit)
    help_result = command([sys.executable, "-s", "-B", "-m", "tools.cpu_s4_6e_array_propagation.c_entry", "--help"])
    from .c_candidate import build_role, verify_bindings

    factories = {}
    for role in ("R", "C"):
        built = build_role("PURE_OFF_SMALLGRID", capture_root=Path(report) / ("UNUSED_PREFLIGHT_" + role), role=role, local_engineering=False)
        try:
            factories[role] = verify_bindings(built[1])
        finally:
            built[1].close_c_pool()
        _require(built[1].c_pool_snapshot()["closed"], "S46E_C_PREFLIGHT_FACTORY_CLOSED")
    return dict(status="PASS_READONLY_C_SOURCE_RESOURCE_ENVIRONMENT_PREFLIGHT", execution_commit=binding["execution_commit"], executable=sys.executable, versions=versions, batch_audit=audit, entry_help_checked=bool(help_result), factories=factories, propagation_called=False)


def node_gate(report):
    import psutil

    from tools.cpu_o1.cpu_s4.memory_limits import snapshot_cgroup_memory

    checked = preflight(report)
    authorization = c_entry.read(Path(report) / "RESOURCE_AUTHORIZATION.json")
    resources = plan()["resources"]
    _require(authorization.get("authorized") is True and authorization.get("execution_commit") == checked["execution_commit"] and authorization.get("resources") == resources and bool(authorization.get("user_instruction")), "S46E_C_NEW_EXPLICIT_RESOURCE_AUTHORIZATION_REQUIRED")
    job = os.environ.get("SLURM_JOB_ID", "")
    _require(job.isdigit() and os.environ.get("SLURM_RESTART_COUNT", "0") == "0", "S46E_C_ONE_NEW_NON_REQUEUED_JOB")
    _require(os.environ.get("SLURM_CPUS_PER_TASK") == "96" and os.environ.get("SLURM_JOB_NUM_NODES") == "1", "S46E_C_ONE_NODE_CPU96")
    description = command(["scontrol", "show", "job", job, "-o"])
    fields = dict(word.split("=", 1) for word in description.split() if "=" in word)
    _require(fields.get("NumCPUs") == "96" and fields.get("NumNodes") == "1" and fields.get("TimeLimit") == "00:30:00" and fields.get("Account") == "bscc-t" and fields.get("Partition") == "v6_384", "S46E_C_EXACT_RESOURCES", description)
    tres = dict(word.split("=", 1) for word in fields["AllocTRES"].split(","))
    _require(tres.get("mem") == "32G" and "gres/gpu" not in description and not os.environ.get("SLURM_JOB_GPUS"), "S46E_C_32G_ZERO_GPU")
    affinity = psutil.Process().cpu_affinity()
    topology = command(["lscpu", "-p=CPU,CORE,SOCKET,NODE"])
    rows = [line.split(",") for line in topology.splitlines() if line and not line.startswith("#")]
    cores = {(row[2], row[1]) for row in rows if int(row[0]) in affinity}
    numas = {row[3] for row in rows if int(row[0]) in affinity}
    hardware = command(["lscpu"])
    _require(len(affinity) == len(cores) == 96 and len(numas) == 4 and "Intel(R) Xeon(R) Platinum 9242" in hardware, "S46E_C_FROZEN_96_PHYSICAL_CORE_NODE")
    memory = snapshot_cgroup_memory(os.getpid(), allocation_bytes=32 * 1024**3)
    _require(memory.get("status") == "READ" and memory["effective_limit_bytes"] > 0, "S46E_C_ACTUAL_CGROUP")
    guard = int(0.8 * min(32 * 1024**3, memory["effective_limit_bytes"]))
    _require(guard > 20 * 1024**3, "S46E_C_MEMORY_GUARD")
    leaf = Path(memory["levels_leaf_to_mount_root"][0]["path"])
    peak_path = leaf / ("memory.max_usage_in_bytes" if memory["version"] == 1 else "memory.peak")
    _require(int(peak_path.read_text().strip()) < guard, "S46E_C_INITIAL_MEMORY_GUARD")
    data = Path(os.environ["CPU_S4_DATA_ROOT"]).resolve(strict=True)
    contract = c_entry.read(Path(report) / "INPUT_CONTRACT.json")
    _require(pv.raw_sha256_file(data / "SMALLGRID_CONFIG.json") == contract["input_contract"]["config_raw_sha256"], "S46E_C_FROZEN_CONFIG")
    for row in c_entry.read(data / "raw/FROZEN_LUT_RETRIEVAL.json"):
        path = data / "inputs" / row["path"]
        _require(row["classification"] == "external" and row["hash_scope"] == "raw_bytes" and path.resolve().is_relative_to(data) and path.stat().st_size == row["size_bytes"] and pv.raw_sha256_file(path) == row["raw_sha256"], "S46E_C_FROZEN_LUT", row["path"])
    refs = seal_reference_files(report)
    node = dict(status="PASS_S46E_C_SMALLGRID_SOURCE_RESOURCE_INPUT_GATES", job_id=job, preflight=checked, allocation=description, hardware=hardware, affinity=affinity, physical_cores=len(cores), numa_nodes=sorted(numas), memory=memory, process_tree_stop_bytes=guard, memory_peak_path=str(peak_path), sealed_reference_files=refs, default="I0", production_qualified=False)
    sg.write(Path(report) / "NODE_GATE.json", node)
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
                reason = "WORK_DEADLINE_1680_SECONDS"
                break
            time.sleep(1)
    finally:
        if process.poll() is None:
            terminate(process.pid, created)
    code = process.wait(timeout=15)
    finished = time.monotonic()
    kernel_peak = max(kernel_peak, int(peak_path.read_text().strip()))
    if reason is None and kernel_peak >= limit:
        reason = "ACTUAL_MEMORY_GUARD_80_PERCENT_FINAL"
    if reason is None and finished >= deadline:
        reason = "WORK_DEADLINE_1680_SECONDS_FINAL"
    receipt = dict(exit_code=code, stop_reason=reason, process_service_s=time.monotonic() - started, peak_tree_rss_bytes=peak, kernel_cgroup_peak_bytes=kernel_peak, memory_samples=samples)
    sg.write(receipt_path, receipt)
    _require(code == 0 and reason is None, "S46E_C_STOP_CHILD_FAILURE", receipt)
    return receipt


def _launch(report: Path, *, case: str, role: str, capture: bool, reference_output: str | None, deadline: float, node: dict) -> dict:
    folder = report / "runs/C01" / case
    folder.mkdir(parents=True, exist_ok=False)
    work = folder / "work"
    argv = [sys.executable, "-s", "-B", "-m", "tools.cpu_s4_6e_array_propagation.c_entry", "--report", str(report), "--work", str(work), "--role", role, "--intervals", "8", "--input-manifest", str(report / "BENCHMARK_INPUT_MANIFEST.json")]
    if capture:
        argv.extend(("--capture", "--capture-root", str(folder / "CAPTURE")))
    if reference_output:
        argv.extend(("--reference-output", reference_output))
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env["NUMBA_CACHE_DIR"] = str(report / "cache" / role)
    sg.write(folder / "LAUNCH.json", dict(argv=argv, case=case, role=role, intervals=8, capture=capture, UTC=dt.datetime.now(dt.timezone.utc).isoformat(), warmup_intervals=0, factor_cache="EMPTY_PRIVATE_NEW_PROCESS", JIT_cache=env["NUMBA_CACHE_DIR"]))
    with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    try:
        receipt = _supervise_process(process, node=node, deadline=deadline, receipt_path=folder / "SUPERVISOR_RECEIPT.json")
    except BaseException:
        failure_path = work / "ENTRY_FAILURE.json"
        if failure_path.is_file():
            failure = c_entry.read(failure_path)
            if failure.get("classification") == "SCIENTIFIC_NONEXACT":
                return {
                    "schema": "filament.cpu_s4_6e_c.case_failure.v1",
                    "case_id": case,
                    "role": role,
                    "intervals": 8,
                    "status": "SCIENTIFIC_NONEXACT",
                    "execution_domain": "S46E_SMALLGRID",
                    "shape": list(SHAPE),
                    "entry_failure": failure,
                    "supervisor": c_entry.read(folder / "SUPERVISOR_RECEIPT.json"),
                }
        raise
    result = c_entry.read(work / "CASE_RECEIPT.json")
    result.update(case_id=case, supervisor=receipt)
    _require(result["execution_domain"] == "S46E_SMALLGRID" and result["shape"] == list(SHAPE), "S46E_C_CASE_DOMAIN")
    return result


def _launch_mechanism(report: Path, capture_root: Path, deadline: float, node: dict) -> dict:
    _require(time.monotonic() < deadline, "S46E_C_DEADLINE_BEFORE_MECHANISM")
    output = report / "runs/C01/MECHANISM" / "MECHANISM.json"
    output.parent.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, "-s", "-B", "-m", "tools.cpu_s4_6e_array_propagation.c_entry", "--report", str(report), "--mechanism", "--allow-p24", "--capture-root", str(capture_root), "--output", str(output)]
    with (output.parent / "stdout.log").open("xb") as stdout, (output.parent / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(argv, env=os.environ.copy(), stdout=stdout, stderr=stderr, start_new_session=True)
    _supervise_process(process, node=node, deadline=deadline, receipt_path=output.parent / "SUPERVISOR_RECEIPT.json")
    return c_entry.read(output)


def _write_scientific_stop(report: Path, node: dict, started: float, cases: dict, *, stage: str) -> dict:
    result = dict(
        schema="filament.cpu_s4_6e_c.result.v1",
        status="SCIENTIFIC_NONEXACT",
        classification="SCIENTIFIC_NONEXACT",
        scientific_stop_stage=stage,
        cases=cases,
        default="I0",
        production_qualified=False,
        resource_ledger=dict(
            job_id=node["job_id"],
            jobs_used=1,
            retry_authorized=False,
            controller_service_s=time.monotonic() - started,
            allocated_core_hours_max=48,
            scheduler_terminal_state="AWAIT_SACCT",
        ),
    )
    sg.write(report / "RESULT.json", result)
    return result


def run(report):
    report = Path(report).resolve(strict=True)
    started = time.monotonic()
    node = node_gate(report)
    inputs = sg.input_manifest(report, shape=SHAPE)
    expected_inputs = c_entry.read(report / "INPUT_CONTRACT.json")["expected_input_fingerprints"]
    _require(inputs["fingerprints"] == expected_inputs, "S46E_C_FROZEN_DIRECT_SMALLGRID_INPUTS")
    deadline = started + 1680
    cases = {}
    try:
        frozen_reference = c_entry.read(report / "REFERENCE_INPUT_RAW_RECORDS.json")["reference"]
        _require(frozen_reference.get("output"), "S46E_C_FROZEN_REFERENCE_OUTPUT_REQUIRED")
        uncaptured = _launch(report, case="R8_UNCAPTURED", role="R", capture=False, reference_output=frozen_reference["output"], deadline=deadline, node=node)
        cases["R8_UNCAPTURED"] = uncaptured
        if uncaptured.get("status") == "SCIENTIFIC_NONEXACT":
            return _write_scientific_stop(report, node, started, cases, stage="R8_UNCAPTURED")
        _require(uncaptured["comparisons"]["terminal"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS", "S46E_C_UNCAPTURED_R8_TERMINAL_NONEXACT")
        captured = _launch(report, case="R8_CAPTURED", role="R", capture=True, reference_output=uncaptured["output"], deadline=deadline, node=node)
        cases["R8_CAPTURED"] = captured
        if captured.get("status") == "SCIENTIFIC_NONEXACT":
            return _write_scientific_stop(report, node, started, cases, stage="R8_CAPTURED")
        _require(captured["comparisons"]["terminal"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS", "S46E_C_CAPTURED_R8_TERMINAL_NONEXACT")
        mechanism = _launch_mechanism(report, Path(captured["capture_package"]), deadline, node)
        cases["MECHANISM"] = mechanism
        if mechanism.get("status") == "SCIENTIFIC_NONEXACT":
            final_status = "SCIENTIFIC_NONEXACT"
            classification = "SCIENTIFIC_NONEXACT"
        elif mechanism.get("mechanism_gate") is not True:
            final_status = "SCIENTIFIC_INSUFFICIENT_P24_GAIN"
            classification = "SCIENTIFIC_INSUFFICIENT_GAIN"
        else:
            final_status = "C_CAPTURE_AND_MECHANISM_EXACT_GATE_PASS_MANUAL_REVIEW_PENDING"
            classification = "SCIENTIFIC_QUALIFICATION_CANDIDATE"
        result = dict(schema="filament.cpu_s4_6e_c.result.v1", status=final_status, classification=classification, cases=cases, mechanism=mechanism, resource_ledger=dict(job_id=node["job_id"], jobs_used=1, retry_authorized=False, controller_service_s=time.monotonic() - started, allocated_core_hours_max=48, scheduler_terminal_state="AWAIT_SACCT"), default="I0", production_qualified=False)
        sg.write(report / "RESULT.json", result)
        return result
    except BaseException as error:
        sg.write(report / "JOB_FAILURE.json", dict(classification="SCIENTIFIC_NONEXACT" if "NONEXACT" in repr(error) else "ENGINEERING_OR_RESOURCE_FAILURE", error=repr(error), traceback=traceback.format_exc(), completed_cases=list(cases), default="I0", production_qualified=False, automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    value = preflight(args.report.resolve(strict=True)) if args.preflight else run(args.report)
    print(json.dumps(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
