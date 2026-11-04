"""Original-grid power cleanup capture and fixed local mechanism."""
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
from tools.cpu_s4_6f_fullsize_parallel import hpc as parent
from tools.cpu_s4_6f_fullsize_parallel.hpc import (
    _require, _seal_canonical_files, _validate_reference_paths, _supervise_process,
    _child_failure_classification, _require_fresh_controller, validate_physical_topology,
)
from . import alpha_entry as entry
from .alpha_measurement import KEYS, _snapshot, validate_pools
from .alpha_protocol import CANDIDATE, COMPONENTS, REMOTE_PYTHON, SHAPE, THREAD_ENV, VERSIONS, plan
from .alpha_role_spec import build_role, close_role, verify_role

RESOURCE = plan()["resources"]


def _preflight_factories(report, candidate):
    results = {}
    for role in ("R", "X"):
        root = report / ("UNUSED_COMBINED_PREFLIGHT_" + role)
        _require(not root.exists(), "S46J_ALPHA_PREFLIGHT_TARGET_EXISTS")
        built = build_role(candidate, "PURE_OFF_FULLSIZE", capture_root=root,
                           role=role, local_engineering=False, shape=SHAPE, recorder_pair=None)
        try:
            checked = verify_role(built[1], candidate)
        finally:
            close_role(built[1], candidate)
        snapshot = _snapshot(built[1], candidate)
        _require(snapshot["closed"] and not snapshot["active_thread_ids"]
                 and snapshot["close_error"] is None, "S46J_ALPHA_PREFLIGHT_JOIN")
        _require(not root.exists(), "S46J_ALPHA_PREFLIGHT_CAPTURE_CREATED")
        results[role] = dict(binding=checked, post_close_pool=snapshot)
    return dict(status="PASS_READONLY_COMBINED_FACTORY_PREFLIGHT", propagation_called=False,
                factories=results)


def preflight(report: str | Path, candidate: str) -> dict:
    """Perform source/interpreter/batch checks without creating run state."""

    from tools.hpc_ops.audit_batch_entry import audit_batch_entry

    report = Path(report).resolve(strict=True)
    _require(getpass.getuser() == "t0s000727", "S46J_ALPHA_T0_ONLY")
    _require(Path(sys.executable).absolute() == Path(REMOTE_PYTHON), "S46J_ALPHA_FIXED_INTERPRETER")
    _require(sys.version_info[:3] == (3, 12, 8), "S46J_ALPHA_PYTHON_PATCH")
    versions = {name: importlib.import_module(name).__version__ for name in VERSIONS}
    _require(versions == VERSIONS, "S46J_ALPHA_FROZEN_DEPENDENCIES", versions)
    _require(os.environ.get("UPPE_USE_GPU") == "0" and all(os.environ.get(key) == value for key, value in THREAD_ENV.items()), "S46J_ALPHA_FROZEN_CPU_THREAD_ENV")
    binding = entry.verify_source(report, candidate=candidate, local_engineering=False, shape=SHAPE, intervals=8, mode="PURE_OFF_FULLSIZE", require_node=False)
    repo = Path(__file__).resolve().parents[3]
    pv.validate_manifest(repo, report / "EXECUTION_SOURCE_MANIFEST.json", require_clean=True, require_hash_scope=True)
    audit, code = audit_batch_entry(Path(__file__).with_name("alpha_capture_cpu.sbatch"), REMOTE_PYTHON, "bash")
    _require(code == 0 and audit.get("status") == "passed", "S46J_ALPHA_BATCH_ENTRY_AUDIT", audit)
    help_result = command([sys.executable, "-s", "-B", "-m", "tools.cpu_s4_6j_alpha_factory.alpha_entry", "--help"])
    factories = _preflight_factories(report, str(candidate).upper())
    return dict(status="PASS_S46J_ALPHA_SOURCE_RESOURCE_ENVIRONMENT_PREFLIGHT", execution_commit=binding["execution_commit"], candidate=str(candidate).upper(), executable=sys.executable, versions=versions, batch_audit=audit, entry_help_checked=bool(help_result), factories=factories, propagation_called=False)


def node_gate(report: str | Path, candidate: str) -> dict:
    import psutil

    report = Path(report).resolve(strict=True)
    checked = preflight(report, candidate)
    authorization_path = report / "RESOURCE_AUTHORIZATION.json"
    _require(authorization_path.is_file(), "S46J_ALPHA_RESOURCE_AUTHORIZATION_MISSING")
    authorization = entry.read(authorization_path)
    _require(authorization.get("authorized") is True and authorization.get("candidate") == str(candidate).upper() and authorization.get("execution_commit") == checked["execution_commit"] and authorization.get("resources") == RESOURCE and bool(authorization.get("user_instruction")), "S46J_ALPHA_RESOURCE_AUTHORIZATION")
    job = os.environ.get("SLURM_JOB_ID", "")
    _require(job.isdigit() and os.environ.get("SLURM_RESTART_COUNT", "0") == "0", "S46J_ALPHA_ONE_NEW_NON_REQUEUED_JOB")
    _require(os.environ.get("SLURM_CPUS_PER_TASK") == "96" and os.environ.get("SLURM_JOB_NUM_NODES") == "1", "S46J_ALPHA_ONE_NODE_CPU96")
    description = command(["scontrol", "show", "job", job, "-o"])
    fields = dict(word.split("=", 1) for word in description.split() if "=" in word)
    _require(fields.get("NumCPUs") == "96" and fields.get("NumNodes") == "1" and fields.get("TimeLimit") == "02:00:00" and fields.get("Account") == "bscc-t" and fields.get("Partition") == "v6_384", "S46J_ALPHA_EXACT_RESOURCES", description)
    tres = dict(word.split("=", 1) for word in fields.get("AllocTRES", "").split(",") if "=" in word)
    _require(tres.get("mem") in ("128G", "131072M") and "gres/gpu" not in description and not os.environ.get("SLURM_JOB_GPUS"), "S46J_ALPHA_128G_ZERO_GPU")
    affinity = psutil.Process().cpu_affinity()
    hardware = command(["lscpu"])
    topology = command(["lscpu", "-p=CPU,CORE,SOCKET,NODE"])
    topology_receipt = validate_physical_topology(hardware, topology, affinity)
    memory = snapshot_cgroup_memory(os.getpid(), allocation_bytes=128 * 1024**3)
    _require(memory.get("status") == "READ" and memory.get("effective_limit_bytes", 0) > 0, "S46J_ALPHA_ACTUAL_CGROUP")
    guard = int(0.8 * min(128 * 1024**3, memory["effective_limit_bytes"]))
    _require(guard > 80 * 1024**3, "S46J_ALPHA_MEMORY_GUARD")
    leaf = Path(memory["levels_leaf_to_mount_root"][0]["path"])
    peak_path = leaf / ("memory.max_usage_in_bytes" if memory["version"] == 1 else "memory.peak")
    _require(int(peak_path.read_text().strip()) < guard, "S46J_ALPHA_INITIAL_MEMORY_GUARD")
    reference_paths = _validate_reference_paths(report)
    sealed_inputs = _seal_canonical_files(report)
    sealed_refs = seal_reference_files(report)
    _require(len(sealed_refs.get("records", ())) == 118, "S46J_ALPHA_REFERENCE_RECORD_COUNT")
    node = dict(status="PASS_S46J_ALPHA_FULLSIZE_SOURCE_RESOURCE_INPUT_GATES", job_id=job, candidate=str(candidate).upper(), preflight=checked, allocation=description, hardware=hardware, topology=topology, **topology_receipt, memory=memory, process_tree_stop_bytes=guard, memory_peak_path=str(peak_path), reference_paths=reference_paths, sealed_canonical_inputs=sealed_inputs, sealed_reference_files=sealed_refs, default="I0", production_qualified=False)
    sg.write(report / "NODE_GATE.json", node)
    return node


def launch(report: Path, candidate: str, spec: dict, reference: dict | None, *, deadline: float, node: dict) -> dict:
    case, role, intervals, kind = spec["case"], spec["role"], int(spec["intervals"]), spec["kind"]
    _require(time.monotonic() < deadline, "S46J_ALPHA_DEADLINE_BEFORE_NEXT_WINDOW")
    folder = report / "runs" / "S46J_ALPHA" / str(candidate).upper() / case
    folder.mkdir(parents=True, exist_ok=False)
    work = folder / "work"
    argv = [sys.executable, "-s", "-B", "-m", "tools.cpu_s4_6j_alpha_factory.alpha_entry", "--report", str(report), "--work", str(work), "--candidate", str(candidate).upper(), "--role", role, "--intervals", str(intervals), "--input-manifest", str(report / "FULL_INPUT_MANIFEST.json")]
    if spec.get("capture"):
        argv.append("--capture")
    if spec.get("mechanism"):
        argv.extend(("--mechanism", "--capture-root", spec["capture_root"]))
    if reference:
        if reference.get("output") and (intervals == 8 or role == "X"):
            argv.extend(("--reference-output", reference["output"]))
        if intervals == 2:
            for flag, key in (("--reference-boundary", "boundary_path"), ("--reference-stages", "stages")):
                if reference.get(key):
                    argv.extend((flag, reference[key]))
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env["NUMBA_CACHE_DIR"] = str(report / "cache" / str(candidate).upper() / case)
    sg.write(folder / "LAUNCH.json", dict(argv=argv, case=case, candidate=str(candidate).upper(), role=role, intervals=intervals, kind=kind, UTC=dt.datetime.now(dt.timezone.utc).isoformat(), warmup_intervals=0, factor_cache="EMPTY_PRIVATE_NEW_PROCESS", JIT_cache=env["NUMBA_CACHE_DIR"]))
    with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    receipt = _supervise_process(process, node=node, deadline=deadline, receipt_path=folder / "SUPERVISOR_RECEIPT.json")
    failure = work / "ENTRY_FAILURE.json"
    failure_value = entry.read(failure) if failure.is_file() else None
    classification = _child_failure_classification(receipt, failure_value)
    if classification == "SCIENTIFIC_NONEXACT":
        return dict(schema="filament.cpu_s4_6j.alpha.case_failure.v1", case_id=case, candidate=str(candidate).upper(), role=role, intervals=intervals, status="SCIENTIFIC_NONEXACT", execution_domain="S46J_FULLSIZE", shape=list(SHAPE), entry_failure=failure_value, supervisor=receipt)
    if classification == "ENGINEERING_OR_RESOURCE_FAILURE":
        _require(False, "S46J_ALPHA_STOP_CHILD_FAILURE", dict(supervisor=receipt, entry_failure=failure_value))
    if spec.get("mechanism"):
        value = entry.read(work / "MECHANISM_RESULT.json")
        value.update(supervisor=receipt,case_id=case)
        return value
    _require((work / "ALPHA_CASE_RECEIPT.json").is_file(), "S46J_ALPHA_CASE_RECEIPT_MISSING")
    result = entry.read(work / "ALPHA_CASE_RECEIPT.json")
    result.update(case_id=case, kind=kind, supervisor=receipt)
    _require(result.get("execution_domain") == "S46J_FULLSIZE" and result.get("shape") == list(SHAPE), "S46J_ALPHA_CASE_DOMAIN")
    return result

def _exact(result, *, terminal, boundary, stages):
    snapshot = result["pointwise_pool"]["pool"]
    activity = result["pointwise_pool"]["activity"]
    for name, key in KEYS.items():
        _require(result.get("pool_role") == "X", "S46J_ALPHA_ACTIVE_JOINT_POOLS_BOTH_ROLES")
        component = dict(result, role="X")
        component["pointwise_pool"] = dict(pool=snapshot["components"][key],
            activity=[row for row in activity if row["label"] in parent.POINTWISE_LABELS[name]])
        parent._exact(component, terminal=terminal, boundary=boundary, stages=stages, candidate=name)
    validate_pools(CANDIDATE,result["role"],result["intervals"],snapshot,activity,local_engineering=False)
    result["exact_gate_pass"] = True
    result["terminal_exact_basis"] = ("CURRENT_REFERENCE" if terminal else
        "CURRENT_R2_BOUNDARY_STAGE_NO_HISTORICAL_R2_TERMINAL")


def _classify_failure(error):
    return entry.exception_classification(error)


def run(report,candidate=CANDIDATE):
    _require(candidate==CANDIDATE,'S46J_ALPHA_SELECTED_ONLY')
    report=Path(report).resolve(strict=True)
    _require_fresh_controller(report)
    started=time.monotonic()
    cases={}
    try:
        node=node_gate(report,candidate)
        deadline=started+RESOURCE['work_deadline_s']
        manifest=entry.read(report/'FULL_INPUT_MANIFEST.json')
        expected=entry.read(report/'INPUT_CONTRACT.json')['expected_input_fingerprints']
        _require(manifest['shape']==list(SHAPE) and manifest['fingerprints']==expected,'S46J_ALPHA_FROZEN_FULL_INPUTS')
        historical=entry.read(report/'REFERENCE_INPUT_RAW_RECORDS.json')['reference']
        for name,capture in (('R8_UNCAPTURED',False),('R8_CAPTURED',True)):
            reference=cases['R8_UNCAPTURED'] if capture else historical
            spec=dict(case=name,role='R',intervals=8,kind='CAPTURE_QUALIFICATION',capture=capture)
            value=launch(report,candidate,spec,reference,deadline=deadline,node=node)
            _require(value.get('status')!='SCIENTIFIC_NONEXACT','SCIENTIFIC_NONEXACT:'+name)
            _exact(value,terminal=True,boundary=False,stages=False)
            _require(value['capture_enabled'] is capture and value['residual_cost_records']==0,'S46J_ALPHA_CAPTURE_ROLE_BINDING')
            cases[name]=value
            sg.write(report/'runs/S46J_ALPHA'/candidate/name/'ACCEPTED_CASE.json',value)
        spec=dict(case='MECHANISM',role='R',intervals=8,kind='MECHANISM',mechanism=True,
                  capture_root=cases['R8_CAPTURED']['capture_package'])
        mechanism=launch(report,candidate,spec,None,deadline=deadline,node=node)
        _require(mechanism.get('status')!='SCIENTIFIC_NONEXACT','SCIENTIFIC_NONEXACT:MECHANISM')
        _require(mechanism['inputs']==3 and mechanism['warmup_calls']==36 and mechanism['formal_calls']==90 and
                 mechanism['all_exact'] is True and len(mechanism['blocks'])==6,'S46J_ALPHA_MECHANISM_COMPLETENESS')
        allowed=all(row['p24_gain']>=0.05 for row in mechanism['blocks'])
        _require(mechanism['formal14_allowed'] is allowed,'S46J_ALPHA_MECHANISM_GATE_BINDING')
        result=dict(schema='filament.cpu_s4_6j.alpha.capture_stage.v1',status=mechanism['status'],
            candidate=candidate,execution_commit=node['preflight']['execution_commit'],
            shape=list(SHAPE),science_exact=True,formal14_allowed=allowed,
            cases=cases,mechanism=mechanism,default='I0',production_qualified=False,
            scheduler_terminal_state='AWAIT_SACCT',review_gate='MANUAL_REVIEW_PENDING')
        sg.write(report/'ACCEPTED_CASES.json',cases)
        sg.write(report/'RESULT.json',result)
        sg.write(report/'RESOURCE_LEDGER.json',dict(job_id=node['job_id'],jobs_used=1,
            serial_propagation_windows=2,mechanism_inputs=3,retry_authorized=False,
            controller_service_s=time.monotonic()-started,reserved_allocated_core_hours=192,
            scheduler_terminal_state='AWAIT_SACCT'))
        return result
    except BaseException as exc:
        if not (report/'JOB_FAILURE.json').exists():
            sg.write(report/'JOB_FAILURE.json',dict(classification=_classify_failure(exc),
                error=repr(exc),traceback=traceback.format_exc(),completed_cases=list(cases),automatic_retry=False))
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--candidate',choices=(CANDIDATE,),default=CANDIDATE)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    result=preflight(args.report,args.candidate) if args.preflight_only else run(args.report,args.candidate)
    print(json.dumps(dict(status=result['status'],candidate=args.candidate)))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
