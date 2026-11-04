"""Serial original-size S4-6J alpha fourteen-window controller."""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_5b_steady_cost.hpc import command, seal_reference_files
from tools.cpu_o1.cpu_s4.memory_limits import snapshot_cgroup_memory
from tools.hpc_ops import provenance_v2 as pv
from tools.cpu_s4_6f_fullsize_parallel import hpc as parent
from tools.cpu_s4_6f_fullsize_parallel.hpc import (
    _child_failure_classification,
    _require,
    _require_fresh_controller,
    _seal_canonical_files,
    _supervise_process,
    _validate_reference_paths,
    validate_physical_topology,
)

from . import alpha_formal_analysis as analysis
from . import alpha_formal_entry as entry
from .alpha_formal_protocol import (
    BASE,
    CANDIDATE,
    COMPONENTS,
    MICRO_SHAPE,
    REMOTE_PYTHON,
    SHAPE,
    TASK,
    THREAD_ENV,
    VERSIONS,
    candidate_spec,
    exception_classification,
    matrix,
    plan,
)
from .alpha_measurement import KEYS, validate_pools
from .alpha_role_spec import build_role, close_role, snapshot_role, verify_role


RESOURCE = plan()["resources"]
SCHEMA = "filament.cpu.s4_6j.alpha_formal_hpc.v1"


def _preflight_factories(report: Path, candidate: str) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for role in ("R", "X"):
        root = report / ("UNUSED_ALPHA_FORMAL_PREFLIGHT_" + role)
        _require(not root.exists(), "S46J_ALPHA_FORMAL_PREFLIGHT_TARGET_EXISTS")
        built = build_role(
            candidate,
            "PURE_OFF_FULLSIZE",
            capture_root=root,
            role=role,
            local_engineering=False,
            shape=SHAPE,
            recorder_pair=None,
            cost_mode="OFF",
        )
        try:
            checked = verify_role(built[1], candidate)
        finally:
            close_role(built[1], candidate)
        snapshot = snapshot_role(built[1], candidate)
        _require(snapshot["closed"] and not snapshot["active_thread_ids"], "S46J_ALPHA_FORMAL_PREFLIGHT_JOIN")
        _require(not root.exists(), "S46J_ALPHA_FORMAL_PREFLIGHT_CAPTURE_CREATED")
        results[role] = {"binding": checked, "post_close_pool": snapshot}
    return {"status": "PASS_S46J_ALPHA_FORMAL_FACTORY_PREFLIGHT", "propagation_called": False, "factories": results}


def preflight(report: str | Path, candidate: str = CANDIDATE) -> dict[str, Any]:
    from tools.hpc_ops.audit_batch_entry import audit_batch_entry

    report_path = Path(report).resolve(strict=True)
    _require(getpass.getuser() == "t0s000727", "S46J_ALPHA_FORMAL_T0_ONLY")
    _require(Path(sys.executable).absolute() == Path(REMOTE_PYTHON), "S46J_ALPHA_FORMAL_FIXED_INTERPRETER")
    _require(sys.version_info[:3] == (3, 12, 8), "S46J_ALPHA_FORMAL_PYTHON_PATCH")
    versions = {name: importlib.import_module(name).__version__ for name in VERSIONS}
    _require(versions == VERSIONS, "S46J_ALPHA_FORMAL_FROZEN_DEPENDENCIES")
    _require(
        os.environ.get("UPPE_USE_GPU") == "0"
        and all(os.environ.get(key) == value for key, value in THREAD_ENV.items()),
        "S46J_ALPHA_FORMAL_FROZEN_CPU_THREAD_ENV",
    )
    binding = entry.verify_source(
        report_path,
        candidate=candidate,
        local_engineering=False,
        shape=SHAPE,
        intervals=8,
        mode="PURE_OFF_FULLSIZE",
        require_node=False,
    )
    repo = Path(__file__).resolve().parents[3]
    pv.validate_manifest(repo, report_path / "EXECUTION_SOURCE_MANIFEST.json", require_clean=True, require_hash_scope=True)
    audit, code = audit_batch_entry(Path(__file__).with_name("alpha_formal_cpu.sbatch"), REMOTE_PYTHON, "bash")
    _require(code == 0 and audit.get("status") == "passed", "S46J_ALPHA_FORMAL_BATCH_ENTRY_AUDIT")
    help_result = command([sys.executable, "-s", "-B", "-m", "tools.cpu_s4_6j_alpha_factory.alpha_formal_entry", "--help"])
    factories = _preflight_factories(report_path, str(candidate).upper())
    return {
        "status": "PASS_S46J_ALPHA_FORMAL_SOURCE_RESOURCE_ENVIRONMENT_PREFLIGHT",
        "execution_commit": binding["execution_commit"],
        "candidate": str(candidate).upper(),
        "executable": sys.executable,
        "versions": versions,
        "batch_audit": audit,
        "entry_help_checked": bool(help_result),
        "factories": factories,
        "propagation_called": False,
    }


def node_gate(report: str | Path, candidate: str = CANDIDATE) -> dict[str, Any]:
    import psutil

    report_path = Path(report).resolve(strict=True)
    checked = preflight(report_path, candidate)
    authorization_path = report_path / "RESOURCE_AUTHORIZATION.json"
    _require(authorization_path.is_file(), "S46J_ALPHA_FORMAL_RESOURCE_AUTHORIZATION_MISSING")
    authorization = entry.read(authorization_path)
    _require(
        authorization.get("authorized") is True
        and authorization.get("candidate") == str(candidate).upper()
        and authorization.get("execution_commit") == checked["execution_commit"]
        and authorization.get("resources") == RESOURCE
        and bool(authorization.get("user_instruction")),
        "S46J_ALPHA_FORMAL_RESOURCE_AUTHORIZATION",
    )
    job = os.environ.get("SLURM_JOB_ID", "")
    _require(job.isdigit() and os.environ.get("SLURM_RESTART_COUNT", "0") == "0", "S46J_ALPHA_FORMAL_ONE_NON_REQUEUED_JOB")
    _require(os.environ.get("SLURM_CPUS_PER_TASK") == "96" and os.environ.get("SLURM_JOB_NUM_NODES") == "1", "S46J_ALPHA_FORMAL_CPU_TOPOLOGY")
    description = command(["scontrol", "show", "job", job, "-o"])
    fields = dict(word.split("=", 1) for word in description.split() if "=" in word)
    _require(fields.get("NumCPUs") == "96" and fields.get("NumNodes") == "1" and fields.get("TimeLimit") == "02:00:00" and fields.get("Account") == "bscc-t" and fields.get("Partition") == "v6_384", "S46J_ALPHA_FORMAL_EXACT_RESOURCES")
    tres = dict(word.split("=", 1) for word in fields.get("AllocTRES", "").split(",") if "=" in word)
    _require(tres.get("mem") in ("128G", "131072M") and "gres/gpu" not in description and not os.environ.get("SLURM_JOB_GPUS"), "S46J_ALPHA_FORMAL_128G_ZERO_GPU")
    affinity = psutil.Process().cpu_affinity()
    hardware = command(["lscpu"])
    topology = command(["lscpu", "-p=CPU,CORE,SOCKET,NODE"])
    topology_receipt = validate_physical_topology(hardware, topology, affinity)
    memory = snapshot_cgroup_memory(os.getpid(), allocation_bytes=128 * 1024**3)
    _require(memory.get("status") == "READ" and memory.get("effective_limit_bytes", 0) > 0, "S46J_ALPHA_FORMAL_ACTUAL_CGROUP")
    guard = int(0.8 * min(128 * 1024**3, memory["effective_limit_bytes"]))
    _require(guard > 80 * 1024**3, "S46J_ALPHA_FORMAL_MEMORY_GUARD")
    leaf = Path(memory["levels_leaf_to_mount_root"][0]["path"])
    peak_path = leaf / ("memory.max_usage_in_bytes" if memory["version"] == 1 else "memory.peak")
    _require(int(peak_path.read_text().strip()) < guard, "S46J_ALPHA_FORMAL_INITIAL_MEMORY_GUARD")
    reference_paths = _validate_reference_paths(report_path)
    sealed_inputs = _seal_canonical_files(report_path)
    sealed_refs = seal_reference_files(report_path)
    _require(len(sealed_refs.get("records", ())) == 118, "S46J_ALPHA_FORMAL_REFERENCE_RECORD_COUNT")
    node = {
        "status": "PASS_S46J_ALPHA_FORMAL_SOURCE_RESOURCE_INPUT_GATES",
        "job_id": job,
        "candidate": str(candidate).upper(),
        "preflight": checked,
        "allocation": description,
        "hardware": hardware,
        "topology": topology,
        **topology_receipt,
        "memory": memory,
        "process_tree_stop_bytes": guard,
        "memory_peak_path": str(peak_path),
        "reference_paths": reference_paths,
        "sealed_canonical_inputs": sealed_inputs,
        "sealed_reference_files": sealed_refs,
        "default": "I0",
        "production_qualified": False,
    }
    sg.write(report_path / "NODE_GATE.json", node)
    return node


def launch(report: Path, candidate: str, spec: dict[str, Any], reference: dict[str, Any] | None, *, deadline: float, node: dict[str, Any]) -> dict[str, Any]:
    case, role, intervals, kind = spec["case"], spec["role"], int(spec["intervals"]), spec["kind"]
    _require(time.monotonic() < deadline, "S46J_ALPHA_FORMAL_DEADLINE")
    folder = report / "runs" / "S46J_ALPHA_FORMAL" / str(candidate).upper() / case
    folder.mkdir(parents=True, exist_ok=False)
    work = folder / "work"
    argv = [sys.executable, "-s", "-B", "-m", "tools.cpu_s4_6j_alpha_factory.alpha_formal_entry", "--report", str(report), "--work", str(work), "--candidate", str(candidate).upper(), "--role", role, "--intervals", str(intervals), "--input-manifest", str(report / "FULL_INPUT_MANIFEST.json")]
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
    if kind == "QUALIFICATION":
        _require(not Path(env["NUMBA_CACHE_DIR"]).exists(), "S46J_ALPHA_FORMAL_ROLE_JIT_NOT_EMPTY")
    sg.write(folder / "LAUNCH.json", {"argv": argv, "case": case, "candidate": str(candidate).upper(), "role": role, "intervals": intervals, "kind": kind, "UTC": dt.datetime.now(dt.timezone.utc).isoformat(), "warmup_intervals": 0, "factor_cache": "EMPTY_PRIVATE_NEW_PROCESS", "JIT_cache": env["NUMBA_CACHE_DIR"]})
    with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    receipt = _supervise_process(process, node=node, deadline=deadline, receipt_path=folder / "SUPERVISOR_RECEIPT.json")
    failure = work / "FORMAL_ENTRY_FAILURE.json"
    failure_value = entry.read(failure) if failure.is_file() else None
    classification = _child_failure_classification(receipt, failure_value)
    if classification == "SCIENTIFIC_NONEXACT":
        return {"schema": SCHEMA + ".case_failure.v1", "case_id": case, "candidate": str(candidate).upper(), "role": role, "intervals": intervals, "status": "SCIENTIFIC_NONEXACT", "execution_domain": "S46J_FULLSIZE", "shape": list(SHAPE), "entry_failure": failure_value, "supervisor": receipt}
    if classification == "ENGINEERING_OR_RESOURCE_FAILURE":
        return {"case_id": case, "status": classification, "supervisor": receipt, "entry_failure": failure_value}
    _require((work / "FORMAL_CASE_RECEIPT.json").is_file(), "S46J_ALPHA_FORMAL_CASE_RECEIPT_MISSING")
    result = entry.read(work / "FORMAL_CASE_RECEIPT.json")
    result.update(case_id=case, kind=kind, supervisor=receipt)
    _require(result.get("execution_domain") == "S46J_FULLSIZE" and result.get("shape") == list(SHAPE), "S46J_ALPHA_FORMAL_CASE_DOMAIN")
    return result


def _exact(result: dict[str, Any], *, terminal: bool, boundary: bool, stages: bool) -> None:
    snapshot = result["pointwise_pool"]["pool"]
    activity = result["pointwise_pool"]["activity"]
    for name, key in KEYS.items():
        component = dict(result, role="X")
        component["pointwise_pool"] = dict(pool=snapshot["components"][key], activity=[row for row in activity if row["label"] in parent.POINTWISE_LABELS[name]])
        parent._exact(component, terminal=terminal, boundary=boundary, stages=stages, candidate=name)
    validate_pools(CANDIDATE, result["role"], result["intervals"], snapshot, activity, local_engineering=False)
    result["exact_gate_pass"] = True
    result["terminal_exact_basis"] = "CURRENT_REFERENCE" if terminal else "CURRENT_R2_BOUNDARY_STAGE_NO_HISTORICAL_R2_TERMINAL"


def run(report: str | Path, candidate: str = CANDIDATE) -> dict[str, Any]:
    _require(candidate == CANDIDATE, "S46J_ALPHA_FORMAL_CANDIDATE_ONLY")
    report_path = Path(report).resolve(strict=True)
    _require_fresh_controller(report_path)
    started = time.monotonic()
    cases: dict[str, Any] = {}
    active_case = None
    failed_case = None
    try:
        node = node_gate(report_path, candidate)
        deadline = started + int(RESOURCE["work_deadline_s"])
        manifest = entry.read(report_path / "FULL_INPUT_MANIFEST.json")
        expected = entry.read(report_path / "INPUT_CONTRACT.json")["expected_input_fingerprints"]
        _require(manifest["shape"] == list(SHAPE) and manifest["fingerprints"] == expected, "S46J_ALPHA_FORMAL_FROZEN_INPUTS")
        historical = entry.read(report_path / "REFERENCE_INPUT_RAW_RECORDS.json")["reference"]
        for spec in matrix(candidate):
            name = spec["case"]
            active_case = name
            if name == "QUAL_R2":
                reference = {key: historical[key] for key in ("boundary_path", "stages")}
            elif name == "QUAL_X2":
                reference = cases["QUAL_R2"]
            elif name == "CTRL_EARLY_R1":
                reference = historical
            else:
                reference = cases["CTRL_EARLY_R1"]
            value = launch(report_path, candidate, spec, reference, deadline=deadline, node=node)
            if value.get("status") in ("SCIENTIFIC_NONEXACT", "ENGINEERING_OR_RESOURCE_FAILURE"):
                failed_case = value
                raise RuntimeError("S46J_ALPHA_FORMAL_STOP_" + value["status"] + ":" + name)
            _exact(value, terminal=name != "QUAL_R2", boundary=int(spec["intervals"]) == 2, stages=int(spec["intervals"]) == 2)
            _require(value.get("capture_enabled") is False and value.get("mechanism_enabled") is False and value.get("cost_mode") == "OFF" and value.get("residual_cost_records") == 0, "S46J_ALPHA_FORMAL_INSTRUMENTATION_NOT_OFF")
            cases[name] = value
            sg.write(report_path / "runs" / "S46J_ALPHA_FORMAL" / candidate / name / "ACCEPTED_CASE.json", value)
        result = analysis.analyze(cases, candidate)
        result.update(execution_commit=node["preflight"]["execution_commit"], shape=list(SHAPE), science_exact=True, scheduler_terminal_state="AWAIT_SACCT")
        sg.write(report_path / "ACCEPTED_CASES.json", cases)
        sg.write(report_path / "RESULT.json", result)
        sg.write(report_path / "RESOURCE_LEDGER.json", {"job_id": node["job_id"], "jobs_used": 1, "serial_propagation_windows": 14, "retry_authorized": False, "controller_service_s": time.monotonic() - started, "reserved_allocated_core_hours": 192, "scheduler_terminal_state": "AWAIT_SACCT"})
        return result
    except BaseException as error:
        if not (report_path / "JOB_FAILURE.json").exists():
            sg.write(report_path / "JOB_FAILURE.json", {"classification": exception_classification(error), "error": repr(error), "traceback": traceback.format_exc(), "completed_cases": list(cases), "active_case": active_case, "failed_case": failed_case, "automatic_retry": False})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate", choices=(CANDIDATE,), default=CANDIDATE)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = preflight(args.report, args.candidate) if args.preflight_only else run(args.report, args.candidate)
    print(json.dumps({"status": result["status"], "candidate": args.candidate}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "node_gate", "preflight", "run"]
