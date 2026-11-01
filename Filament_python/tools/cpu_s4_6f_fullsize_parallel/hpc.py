"""Read-only preflight and one serial CPU-S4-6F full-size controller."""

from __future__ import annotations

import argparse
import datetime as dt
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
from tools.cpu_o1.cpu_s4.memory_limits import snapshot_cgroup_memory
from tools.hpc_ops import provenance_v2 as pv

from . import entry
from .analysis import analyze
from .protocol import CANDIDATES, REMOTE_PYTHON, RESOURCE, SHAPE, TASK, THREAD_ENV, VERSIONS, matrix, plan

POINTWISE_LABELS = {
    "EQ10_P48": ("B_SPECTRAL_MULTIPLY", "B_DERIVATIVE_DIVIDE", "B_RESPONSE_DIVIDE", "B_PRODUCT"),
    "INTENSITY_P24": ("A_ABS", "A_SQUARE", "A_SCALE"),
    "PHASE_ABSORB_P24": ("C_PHASE", "C_ABSORPTION"),
}


def _require(ok: bool, reason: str, detail=None) -> None:
    if not ok:
        raise RuntimeError(reason if detail is None else f"{reason}:{detail}")


def validate_physical_topology(hardware: str, topology: str, affinity: list[int]) -> dict[str, object]:
    """Require the accepted 96-distinct-core, four-NUMA Xeon 9242 node."""

    rows = [line.split(",") for line in topology.splitlines() if line and not line.startswith("#")]
    _require(all(len(row) == 4 for row in rows), "S46F_CPU_TOPOLOGY_RECORDS")
    allowed = set(int(value) for value in affinity)
    cores = {(row[2], row[1]) for row in rows if int(row[0]) in allowed}
    numas = {row[3] for row in rows if int(row[0]) in allowed}
    _require(len(affinity) == len(cores) == 96 and len(numas) == 4, "S46F_FROZEN_96_PHYSICAL_CORE_FOUR_NUMA")
    _require("Intel(R) Xeon(R) Platinum 9242" in hardware, "S46F_VALIDATED_CPU_MODEL")
    return dict(physical_cores=len(cores), numa_nodes=sorted(numas), affinity=list(affinity))


def _seal_canonical_files(report: Path) -> dict[str, object]:
    """Rehash every canonical manifest record without deserializing arrays."""

    contract = entry.read(report / "INPUT_CONTRACT.json")
    bundle = entry._canonical_bundle_from_contract(contract)
    manifest_path = bundle / "INPUT_MANIFEST.json"
    _require(pv.raw_sha256_file(manifest_path) == contract.get("canonical_manifest_raw_sha256"), "S46F_CANONICAL_MANIFEST_RAW_IDENTITY")
    manifest = entry.read(manifest_path)
    records = []
    for row in manifest.get("records", ()):
        relative = Path(str(row.get("path", "")))
        _require(not relative.is_absolute() and ".." not in relative.parts, "S46F_CANONICAL_RECORD_PATH")
        path = bundle / relative
        _require(path.is_file() and not path.is_symlink() and path.resolve(strict=True).is_relative_to(bundle.resolve(strict=True)), "S46F_CANONICAL_RECORD_FILE")
        _require(path.stat().st_size == int(row["size_bytes"]), "S46F_CANONICAL_RECORD_SIZE", row["path"])
        digest = pv.raw_sha256_file(path)
        _require(digest == row["raw_sha256"], "S46F_CANONICAL_RECORD_RAW_IDENTITY", row["path"])
        _require(row.get("classification") == "external" and row.get("hash_scope") == "raw_bytes", "S46F_CANONICAL_RECORD_PROVENANCE", row["path"])
        records.append(dict(path=str(path), hash_scope="raw_bytes", raw_sha256=digest, size_bytes=path.stat().st_size))
    _require(records and len(records) == len(manifest.get("records", ())), "S46F_CANONICAL_RECORDS_COMPLETE")
    return dict(status="PASS_COMPLETE_RAW_CANONICAL_IDENTITIES_BEFORE_PROPAGATION", manifest_path=str(manifest_path), manifest_raw_sha256=pv.raw_sha256_file(manifest_path), records=records, arrays_deserialized=0)


def _validate_reference_paths(report: Path) -> dict[str, str]:
    contract = entry.read(report / "REFERENCE_INPUT_RAW_RECORDS.json")
    root = Path(contract["source_root"]).resolve(strict=True)
    reference = contract["reference"]
    paths = {}
    for key, expect_directory in (("output", True), ("boundary_path", False), ("stages", True)):
        path = Path(reference[key]).resolve(strict=True)
        _require(path.is_relative_to(root), "S46F_REFERENCE_PATH_OUTSIDE_ROOT", key)
        _require(path.is_dir() if expect_directory else path.is_file(), "S46F_REFERENCE_PATH_TYPE", key)
        paths[key] = str(path)
    return paths


def _preflight_factories(report: Path, candidate: str) -> dict[str, object]:
    """Build and close R/X full-size factories without entering propagation."""

    mechanism_admission = None
    if candidate == "PHASE_ABSORB_P24":
        from .c_fullsize_evidence import require_fullsize_mechanism_admission

        mechanism_admission = require_fullsize_mechanism_admission(report, repo=Path(__file__).resolve().parents[3])
    from .role_spec import build_role, close_role, snapshot_role, verify_role

    factories = {}
    for role in ("R", "X"):
        capture_root = report / f"UNUSED_PREFLIGHT_{candidate}_{role}"
        _require(not capture_root.exists(), "S46F_PREFLIGHT_CAPTURE_ROOT_PREEXISTS", str(capture_root))
        built = build_role(candidate, "PURE_OFF_FULLSIZE", capture_root=capture_root, role=role, local_engineering=False, shape=SHAPE, recorder_pair=None)
        try:
            checked = verify_role(built[1], candidate)
        finally:
            close_role(built[1], candidate)
        snapshot = snapshot_role(built[1], candidate)
        _require(snapshot.get("closed") is True and snapshot.get("active_thread_ids") == [], "S46F_PREFLIGHT_FACTORY_CLOSED")
        _require(not capture_root.exists(), "S46F_PREFLIGHT_CAPTURE_ROOT_CREATED", str(capture_root))
        factories[role] = dict(binding=checked, post_close_pool=snapshot)
    return dict(status="PASS_READONLY_FULLSIZE_FACTORY_PREFLIGHT", propagation_called=False, factories=factories, mechanism_admission=mechanism_admission)


def _classify_failure(error: BaseException) -> str:
    text = repr(error)
    if "S46F_STOP_CHILD_FAILURE" in text:
        return "ENGINEERING_OR_RESOURCE_FAILURE"
    science_tags = ("STOP_FIRST_SCIENCE_NONEXACT", "STOP_FIRST_STAGE_NONEXACT", "S46F_STOP_TERMINAL_NONEXACT", "S46F_STOP_BOUNDARY_NONEXACT", "S46F_STOP_STAGES_NONEXACT", "SCIENTIFIC_NONEXACT")
    return "SCIENTIFIC_NONEXACT" if any(tag in text for tag in science_tags) else "ENGINEERING_OR_RESOURCE_FAILURE"


def _require_fresh_controller(report: Path) -> None:
    for name in ("RESULT.json", "JOB_FAILURE.json", "RESOURCE_LEDGER.json", "ACCEPTED_CASES.json", "NODE_GATE.json", "runs"):
        _require(not (report / name).exists(), "S46F_CONTROLLER_ALREADY_EXECUTED", name)


def preflight(report: str | Path, candidate: str) -> dict:
    """Perform source/interpreter/batch checks without creating run state."""

    from tools.hpc_ops.audit_batch_entry import audit_batch_entry

    report = Path(report).resolve(strict=True)
    _require(getpass.getuser() == "t0s000727", "S46F_T0_ONLY")
    _require(Path(sys.executable).absolute() == Path(REMOTE_PYTHON), "S46F_FIXED_INTERPRETER")
    _require(sys.version_info[:3] == (3, 12, 8), "S46F_PYTHON_PATCH")
    versions = {name: importlib.import_module(name).__version__ for name in VERSIONS}
    _require(versions == VERSIONS, "S46F_FROZEN_DEPENDENCIES", versions)
    _require(os.environ.get("UPPE_USE_GPU") == "0" and all(os.environ.get(key) == value for key, value in THREAD_ENV.items()), "S46F_FROZEN_CPU_THREAD_ENV")
    binding = entry.verify_source(report, candidate=candidate, local_engineering=False, shape=SHAPE, intervals=8, mode="PURE_OFF_FULLSIZE", require_node=False)
    repo = Path(__file__).resolve().parents[3]
    pv.validate_manifest(repo, report / "EXECUTION_SOURCE_MANIFEST.json", require_clean=True, require_hash_scope=True)
    audit, code = audit_batch_entry(Path(__file__).with_name("cpu.sbatch"), REMOTE_PYTHON, "bash")
    _require(code == 0 and audit.get("status") == "passed", "S46F_BATCH_ENTRY_AUDIT", audit)
    help_result = command([sys.executable, "-s", "-B", "-m", "tools.cpu_s4_6f_fullsize_parallel.entry", "--help"])
    factories = _preflight_factories(report, str(candidate).upper())
    return dict(status="PASS_S46F_SOURCE_RESOURCE_ENVIRONMENT_PREFLIGHT", execution_commit=binding["execution_commit"], candidate=str(candidate).upper(), executable=sys.executable, versions=versions, batch_audit=audit, entry_help_checked=bool(help_result), factories=factories, propagation_called=False)


def node_gate(report: str | Path, candidate: str) -> dict:
    import psutil

    report = Path(report).resolve(strict=True)
    checked = preflight(report, candidate)
    authorization_path = report / "RESOURCE_AUTHORIZATION.json"
    _require(authorization_path.is_file(), "S46F_RESOURCE_AUTHORIZATION_MISSING")
    authorization = entry.read(authorization_path)
    _require(authorization.get("authorized") is True and authorization.get("candidate") == str(candidate).upper() and authorization.get("execution_commit") == checked["execution_commit"] and authorization.get("resources") == RESOURCE and bool(authorization.get("user_instruction")), "S46F_RESOURCE_AUTHORIZATION")
    job = os.environ.get("SLURM_JOB_ID", "")
    _require(job.isdigit() and os.environ.get("SLURM_RESTART_COUNT", "0") == "0", "S46F_ONE_NEW_NON_REQUEUED_JOB")
    _require(os.environ.get("SLURM_CPUS_PER_TASK") == "96" and os.environ.get("SLURM_JOB_NUM_NODES") == "1", "S46F_ONE_NODE_CPU96")
    description = command(["scontrol", "show", "job", job, "-o"])
    fields = dict(word.split("=", 1) for word in description.split() if "=" in word)
    _require(fields.get("NumCPUs") == "96" and fields.get("NumNodes") == "1" and fields.get("TimeLimit") == "02:00:00" and fields.get("Account") == "bscc-t" and fields.get("Partition") == "v6_384", "S46F_EXACT_RESOURCES", description)
    tres = dict(word.split("=", 1) for word in fields.get("AllocTRES", "").split(",") if "=" in word)
    _require(tres.get("mem") in ("128G", "131072M") and "gres/gpu" not in description and not os.environ.get("SLURM_JOB_GPUS"), "S46F_128G_ZERO_GPU")
    affinity = psutil.Process().cpu_affinity()
    hardware = command(["lscpu"])
    topology = command(["lscpu", "-p=CPU,CORE,SOCKET,NODE"])
    topology_receipt = validate_physical_topology(hardware, topology, affinity)
    memory = snapshot_cgroup_memory(os.getpid(), allocation_bytes=128 * 1024**3)
    _require(memory.get("status") == "READ" and memory.get("effective_limit_bytes", 0) > 0, "S46F_ACTUAL_CGROUP")
    guard = int(0.8 * min(128 * 1024**3, memory["effective_limit_bytes"]))
    _require(guard > 80 * 1024**3, "S46F_MEMORY_GUARD")
    leaf = Path(memory["levels_leaf_to_mount_root"][0]["path"])
    peak_path = leaf / ("memory.max_usage_in_bytes" if memory["version"] == 1 else "memory.peak")
    _require(int(peak_path.read_text().strip()) < guard, "S46F_INITIAL_MEMORY_GUARD")
    reference_paths = _validate_reference_paths(report)
    sealed_inputs = _seal_canonical_files(report)
    sealed_refs = seal_reference_files(report)
    _require(len(sealed_refs.get("records", ())) == 118, "S46F_REFERENCE_RECORD_COUNT")
    node = dict(status="PASS_S46F_FULLSIZE_SOURCE_RESOURCE_INPUT_GATES", job_id=job, candidate=str(candidate).upper(), preflight=checked, allocation=description, hardware=hardware, topology=topology, **topology_receipt, memory=memory, process_tree_stop_bytes=guard, memory_peak_path=str(peak_path), reference_paths=reference_paths, sealed_canonical_inputs=sealed_inputs, sealed_reference_files=sealed_refs, default="I0", production_qualified=False)
    sg.write(report / "NODE_GATE.json", node)
    return node


def _supervise_process(process, *, node: dict, deadline: float, receipt_path: Path) -> dict:
    import psutil
    from tools.cpu_o1.cpu_s4.supervisor import terminate
    from tools.cpu_s4_fullsize_confirmation.controller import _tree_rss

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
                reason = "WORK_DEADLINE_7080_SECONDS"
                break
            time.sleep(1)
    finally:
        if process.poll() is None:
            terminate(process.pid, created)
    code = process.wait(timeout=30)
    finished = time.monotonic()
    kernel_peak = max(kernel_peak, int(peak_path.read_text().strip()))
    if reason is None and kernel_peak >= limit:
        reason = "KERNEL_MEMORY_HIGH_WATER_GUARD"
    if reason is None and time.monotonic() >= deadline:
        reason = "WORK_DEADLINE_7080_SECONDS"
    receipt = dict(exit_code=code, stop_reason=reason, process_service_s=finished - started, peak_tree_rss_bytes=peak, kernel_cgroup_peak_bytes=kernel_peak, memory_samples=samples)
    sg.write(receipt_path, receipt)
    return receipt


def _child_failure_classification(receipt: dict, failure: dict | None) -> str | None:
    """Classify a failed child only after supervisor priority is preserved."""

    if receipt.get("stop_reason") is not None:
        return "ENGINEERING_OR_RESOURCE_FAILURE"
    if isinstance(failure, dict) and failure.get("classification") == "SCIENTIFIC_NONEXACT":
        return "SCIENTIFIC_NONEXACT"
    if int(receipt.get("exit_code", -1)) != 0:
        return "ENGINEERING_OR_RESOURCE_FAILURE"
    return None


def launch(report: Path, candidate: str, spec: dict, reference: dict | None, *, deadline: float, node: dict) -> dict:
    case, role, intervals, kind = spec["case"], spec["role"], int(spec["intervals"]), spec["kind"]
    _require(time.monotonic() < deadline, "S46F_DEADLINE_BEFORE_NEXT_WINDOW")
    folder = report / "runs" / "S46F" / str(candidate).upper() / case
    folder.mkdir(parents=True, exist_ok=False)
    work = folder / "work"
    argv = [sys.executable, "-s", "-B", "-m", "tools.cpu_s4_6f_fullsize_parallel.entry", "--report", str(report), "--work", str(work), "--candidate", str(candidate).upper(), "--role", role, "--intervals", str(intervals), "--input-manifest", str(report / "FULL_INPUT_MANIFEST.json")]
    if reference:
        if reference.get("output") and (intervals == 8 or role == "X"):
            argv.extend(("--reference-output", reference["output"]))
        if intervals == 2:
            for flag, key in (("--reference-boundary", "boundary_path"), ("--reference-stages", "stages")):
                if reference.get(key):
                    argv.extend((flag, reference[key]))
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env["NUMBA_CACHE_DIR"] = str(report / "cache" / str(candidate).upper() / role)
    sg.write(folder / "LAUNCH.json", dict(argv=argv, case=case, candidate=str(candidate).upper(), role=role, intervals=intervals, kind=kind, UTC=dt.datetime.now(dt.timezone.utc).isoformat(), warmup_intervals=0, factor_cache="EMPTY_PRIVATE_NEW_PROCESS", JIT_cache=env["NUMBA_CACHE_DIR"]))
    with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    receipt = _supervise_process(process, node=node, deadline=deadline, receipt_path=folder / "SUPERVISOR_RECEIPT.json")
    failure = work / "ENTRY_FAILURE.json"
    failure_value = entry.read(failure) if failure.is_file() else None
    classification = _child_failure_classification(receipt, failure_value)
    if classification == "SCIENTIFIC_NONEXACT":
        return dict(schema="filament.cpu_s4_6f.case_failure.v1", case_id=case, candidate=str(candidate).upper(), role=role, intervals=intervals, status="SCIENTIFIC_NONEXACT", execution_domain="S46F_FULLSIZE", shape=list(SHAPE), entry_failure=failure_value, supervisor=receipt)
    if classification == "ENGINEERING_OR_RESOURCE_FAILURE":
        _require(False, "S46F_STOP_CHILD_FAILURE", dict(supervisor=receipt, entry_failure=failure_value))
    _require((work / "CASE_RECEIPT.json").is_file(), "S46F_CASE_RECEIPT_MISSING")
    result = entry.read(work / "CASE_RECEIPT.json")
    result.update(case_id=case, kind=kind, supervisor=receipt)
    _require(result.get("execution_domain") == "S46F_FULLSIZE" and result.get("shape") == list(SHAPE), "S46F_CASE_DOMAIN")
    return result


def _exact(result: dict, *, terminal: bool, boundary: bool, stages: bool, candidate: str) -> None:
    comparisons = result.get("comparisons", {})
    if terminal:
        _require(comparisons.get("terminal", {}).get("status") == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS", "S46F_STOP_TERMINAL_NONEXACT")
    if boundary:
        _require(comparisons.get("boundary", {}).get("status") == "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT", "S46F_STOP_BOUNDARY_NONEXACT")
    if stages:
        _require(comparisons.get("stages", {}).get("status") == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS", "S46F_STOP_STAGES_NONEXACT")
    n_h = 2 * int(result["intervals"])
    route, stats = result["route"], result["cache"]["stats"]
    _require(route["nH"] == n_h and route["fresh_responses"] == 2 * n_h and route["math_dispatches"] == 8 * n_h and route["fallback"] == 0, "S46F_ORIGINAL_ROUTE_COUNTS")
    expected_cache = {2: (1536, 384, 1152), 8: (6144, 384, 5760)}[int(result["intervals"])]
    _require((stats["factor_requests"], stats["plane_builds"], stats["hits"]) == expected_cache and result["cache_fallback"] == 0, "S46F_CACHE_COUNTS")
    pool = result.get("pointwise_pool", {})
    _require(pool.get("pool", {}).get("closed") is True and pool["pool"].get("active_thread_ids") == [] and pool["pool"].get("pending_scalar_records") == 0, "S46F_POINTWISE_POOL_LIFECYCLE")
    spec = CANDIDATES[str(candidate).upper()]
    count = 0 if result["role"] == "R" else (int(result["intervals"]) if spec["pointwise_operation"] == "PHASE_ABSORB" else 4 * int(result["intervals"]))
    _require(pool["pool"].get("operation_counts") == {label: count for label in POINTWISE_LABELS[str(candidate).upper()]}, "S46F_POINTWISE_POOL_COUNTS")
    _require(len(pool.get("activity", ())) == spec["pointwise_labels"] * count, "S46F_POINTWISE_ACTIVITY_COUNT")
    result["exact_gate_pass"] = True
    result["terminal_exact_basis"] = "COMPARED_CURRENT_WINDOW_TO_REFERENCE" if terminal else "R2_SOURCE_BOUND_BOUNDARY_STAGE_ONLY_NO_HISTORICAL_TWO_INTERVAL_TERMINAL"


def run(report: str | Path, candidate: str) -> dict:
    report = Path(report).resolve(strict=True)
    candidate = str(candidate).upper()
    _require(candidate in CANDIDATES, "S46F_UNKNOWN_CANDIDATE")
    _require_fresh_controller(report)
    started = time.monotonic()
    cases: dict[str, dict] = {}
    try:
        node = node_gate(report, candidate)
        deadline = started + int(RESOURCE["work_deadline_s"])
        manifest = entry.read(report / "FULL_INPUT_MANIFEST.json")
        expected = entry.read(report / "INPUT_CONTRACT.json").get("expected_input_fingerprints")
        _require(manifest.get("shape") == list(SHAPE) and manifest.get("fingerprints") == expected, "S46F_FROZEN_FULLSIZE_INPUTS")
        reference_contract = entry.read(report / "REFERENCE_INPUT_RAW_RECORDS.json")
        historical = reference_contract.get("reference", reference_contract)
        _require(historical.get("output"), "S46F_FROZEN_REFERENCE_OUTPUT_REQUIRED")
        for row in matrix(candidate):
            case = row["case"]
            if case == "QUAL_R2":
                reference = {"boundary_path": historical.get("boundary_path"), "stages": historical.get("stages")}
            elif case == "QUAL_X2":
                reference = cases["QUAL_R2"]
            elif case == "CTRL_EARLY_R1":
                reference = historical
            elif case == "CTRL_LATE_R1":
                reference = cases["CTRL_EARLY_R1"]
            else:
                reference = cases["CTRL_EARLY_R1"]
            result = launch(report, candidate, row, reference, deadline=deadline, node=node)
            if result.get("status") == "SCIENTIFIC_NONEXACT":
                sg.write(report / "RESULT.json", dict(schema="filament.cpu_s4_6f.result.v1", status="SCIENTIFIC_NONEXACT", classification="SCIENTIFIC_NONEXACT", scientific_stop_stage=case, cases=cases | {case: result}, default="I0", production_qualified=False))
                return result
            if case == "QUAL_R2":
                _exact(result, terminal=False, boundary=True, stages=True, candidate=candidate)
            elif case == "QUAL_X2":
                _exact(result, terminal=True, boundary=True, stages=True, candidate=candidate)
            else:
                _exact(result, terminal=True, boundary=False, stages=False, candidate=candidate)
            cases[case] = result
            sg.write(report / "runs" / "S46F" / candidate / case / "ACCEPTED_CASE.json", result)
        result = analyze(cases, candidate)
        sg.write(report / "ACCEPTED_CASES.json", cases)
        sg.write(report / "RESULT.json", result)
        sg.write(report / "RESOURCE_LEDGER.json", dict(candidate=candidate, job_id=node["job_id"], jobs_used=1, retry_authorized=False, serial_windows=14, controller_service_s=time.monotonic() - started, per_job_allocated_core_hours=RESOURCE["per_job_allocated_core_hours"], study_allocated_core_hours_max=RESOURCE["study_allocated_core_hours_max"], scheduler_terminal_state="AWAIT_SACCT", actual_allocated_core_hours="AWAIT_SACCT"))
        return result
    except BaseException as error:
        failure = report / "JOB_FAILURE.json"
        if not failure.exists():
            sg.write(failure, dict(classification=_classify_failure(error), error=repr(error), traceback=traceback.format_exc(), completed_cases=list(cases), candidate=candidate, default="I0", production_qualified=False, automatic_retry=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate", choices=tuple(sorted(CANDIDATES)), required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    value = preflight(args.report, args.candidate) if args.preflight else run(args.report, args.candidate)
    print(json.dumps(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
