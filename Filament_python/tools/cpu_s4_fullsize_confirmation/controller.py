"""Bounded supervisor for the already-authorized FULLSIZE confirmation."""
from __future__ import annotations
import argparse, csv, datetime as dt, json, math, os, signal, statistics, subprocess, sys, time, traceback
from pathlib import Path
from typing import Any
from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.hpc_ops import provenance_v2 as pv

SCHEMA = "filament.cpu_s4.fullsize_confirmation.controller.v1"
FULL_SHAPE = (384, 351, 301)
FORMAL_ORDER = {"B1": ("G2_FULL", "G3_FULL"), "B2": ("G3_FULL", "G2_FULL"), "B3": ("G2_FULL", "G3_FULL")}
VERSIONS = {"numpy": "2.2.5", "scipy": "1.15.2", "numba": "0.61.2", "llvmlite": "0.44.0", "psutil": "7.0.0", "threadpoolctl": "3.6.0"}
BLAS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

def require(ok: bool, reason: str, context: Any = None) -> None:
    if not ok: raise RuntimeError({"gate": reason, "context": context})

def command(argv: list[str], *, cwd: str | Path | None = None) -> str:
    result = subprocess.run(argv, cwd=None if cwd is None else str(cwd), capture_output=True, text=True, check=False, timeout=30)
    require(result.returncode == 0, "NODE_METADATA_COMMAND", {"argv": argv, "stderr": result.stderr[-2000:]})
    return result.stdout

def _path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value)
    if not path.is_absolute() and base is not None: path = base / path
    return path.resolve(strict=True)

def _plan(report: Path) -> dict[str, Any]:
    require(sys.flags.optimize==0 and not os.environ.get("PYTHONOPTIMIZE"),"FULL_OPTIMIZATION_FORBIDDEN")
    plan = sg.read(report / "FULLSIZE_CONFIRMATION_PLAN.json")
    require(plan.get("task")=="CPU-S4-3G3-2-CHECKPOINT-FULLSIZE-20261026","FULL_TASK_AUTHORITY")
    require(isinstance(plan, dict), "FULL_PLAN_OBJECT")
    require(plan.get("execution_domain") == "FULLSIZE_CONFIRMATION" and plan.get("shape") == list(FULL_SHAPE) and plan.get("intervals") == 8, "EXPLICIT_FULLSIZE_ENTRY8_ONLY")
    require(plan.get("warmup_intervals_per_formal_case") == 0, "PREREGISTERED_SYMMETRIC_NO_EXTRA_WARMUP")
    budget = tuple(plan.get(key) for key in ("max_submissions", "allocated_core_hours_limit", "physical_cpus", "nodes", "gpus", "memory_GiB"))
    spent, wall = float(plan["spent_core_hours"]), int(plan["walltime_s"])
    require(budget == (2, 144, 96, 1, 0, 320) and plan.get("ordinal") in (1,2) and math.isfinite(spent) and spent >= 0 and 60 < wall <= 4500 and spent+96*wall/3600 <= 144, "FULL_BUDGET_CONTRACT")
    require(plan.get("formal_order") == {key: list(value) for key, value in FORMAL_ORDER.items()}, "FULL_FORMAL_ORDER")
    return plan

def _hash_record(row: dict[str, Any], base: Path | None = None) -> dict[str, Any]:
    path = _path(row["path"], base)
    require(path.is_file() and not path.is_symlink(), "SEALED_INPUT_REGULAR_FILE", str(path))
    size, digest = path.stat().st_size, pv.raw_sha256_file(path)
    require(size == int(row["size_bytes"]) and digest == str(row["raw_sha256"]), "SEALED_INPUT_RAW_IDENTITY", str(path))
    return {"path": str(path), "raw_sha256": digest, "size_bytes": size}

def _sealed_inputs(plan: dict[str, Any]) -> dict[str, Any]:
    root, records = _path(plan["full_input_root"]), list(plan.get("full_input_records", ()))
    require(len(records) == 14, "FULL_INPUT_RECORD_COUNT")
    inputs = []
    for row in records:
        relative = Path(row["relative_path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "FULL_INPUT_RELATIVE_PATH")
        require((root / relative).resolve(strict=True).is_relative_to(root), "FULL_INPUT_ROOT_ESCAPE")
        inputs.append(_hash_record({"path": relative, "raw_sha256": row["raw_sha256"], "size_bytes": row["size_bytes"]}, root))
    require(all(Path(row["path"]).is_absolute() for row in plan.get("reference_records", ())), "FULL_REFERENCE_ABSOLUTE_PATHS")
    references = [_hash_record(row) for row in plan.get("reference_records", ())]
    require(references, "FULL_REFERENCE_RECORDS")
    require(Path(plan["task_record"]["path"]).is_absolute(), "FULL_TASK_ABSOLUTE_PATH")
    return {"full_input_root": str(root), "full_input_records": inputs, "reference_records": references, "task_record": _hash_record(plan["task_record"])}

def _source_gate(plan: dict[str, Any]) -> dict[str, Any]:
    repo, source = Path(__file__).resolve().parents[3], _path(plan["source_root"])
    require(repo == source, "FULL_SOURCE_ROOT")
    manifest = pv.validate_manifest(source, _path(plan["source_provenance"]), require_clean=True, require_hash_scope=True)
    require(manifest["head"] == plan["execution_commit"], "REAL_CHECKPOINT_HEAD")
    require(plan["science_tree"] == "821fec5bdacd832b0347e1f2e84285cb54172e57", "FROZEN_SCIENCE_TREE_AUTHORITY")
    require(pv._git(source, "rev-parse", "HEAD:Filament_python/KHz_filament") == plan["science_tree"], "FROZEN_SCIENCE_TREE")
    records = []
    for row in plan.get("code_records", ()):
        relative, path = str(row["path"]), source / Path(*str(row["path"]).split("/"))
        require(not Path(relative).is_absolute() and path.is_file(), "FULL_COMMITTED_SOURCE_MISSING", relative)
        require(pv._git(source, "rev-parse", "HEAD:" + relative) == row["git_blob_oid"], "FULL_COMMITTED_SOURCE_BLOB", relative)
        require(pv.canonical_lf_sha256(path.read_bytes()) == row["canonical_lf_sha256"], "FULL_COMMITTED_SOURCE_CANONICAL", relative)
        records.append({"path": relative, "git_blob_oid": row["git_blob_oid"], "canonical_lf_sha256": row["canonical_lf_sha256"]})
    require(records, "FULL_CODE_RECORDS")
    return {"manifest": manifest, "records": records, "repo": str(source)}

def _runtime_gate(plan: dict[str, Any]) -> dict[str, Any]:
    import pwd, psutil, threadpoolctl, numpy, scipy, numba, llvmlite
    require(sys.platform.startswith("linux"), "ACTUAL_LINUX_NODE")
    require(pwd.getpwuid(os.geteuid()).pw_name == plan["account_username"], "ACTUAL_ALLOCATION_ACCOUNT")
    require(Path(sys.executable).absolute() == Path(plan["python"]).absolute(), "FIXED_INTERPRETER_PATH")
    require(Path(plan["system_git"]).is_file(), "FIXED_SYSTEM_GIT")
    job = os.environ.get("SLURM_JOB_ID", ""); require(job.isdigit(), "ACTUAL_SLURM_ALLOCATION")
    versions = {"numpy": numpy.__version__, "scipy": scipy.__version__, "numba": numba.__version__, "llvmlite": llvmlite.__version__, "psutil": psutil.__version__, "threadpoolctl": threadpoolctl.__version__}
    require(versions == VERSIONS and sys.version_info[:3] == (3, 12, 8), "FROZEN_DEPENDENCY_VERSIONS", versions)
    require(all(os.environ.get(name) == "1" for name in BLAS), "ACTUAL_BLAS1")
    require(os.environ.get("NUMBA_CPU_NAME") == "generic" and os.environ.get("NUMBA_THREADING_LAYER") == "workqueue", "FROZEN_GENERIC_WORKQUEUE")
    require(all(row.get("num_threads") == 1 for row in threadpoolctl.threadpool_info() if row.get("user_api") == "blas"), "LOADED_BLAS1")
    scheduler, description = plan["scheduler"], command(["scontrol", "show", "job", "-o", job])
    wall=int(plan["walltime_s"]); wall_label=f"{wall//3600:02d}:{wall%3600//60:02d}:{wall%60:02d}"
    required = (f"Partition={scheduler['partition']}", f"Account={scheduler['account']}", f"QOS={scheduler['qos']}", "NumCPUs=96", "TimeLimit="+wall_label)
    require(all(token in description for token in required), "ACTUAL_FULL_ALLOCATION", description)
    require(any(token in description for token in ("MinMemoryNode=320G", "MinMemoryNode=320Gn")), "ACTUAL_FULL_MEMORY_ALLOCATION", description)
    fields=dict(word.split("=",1) for word in description.split() if "=" in word)
    nodes=command(["scontrol","show","hostnames",fields["NodeList"]]).splitlines()
    require(fields.get("NumNodes")=="1" and os.environ.get("SLURM_JOB_NUM_NODES")=="1" and len(nodes)==1,"EXACTLY_ONE_COMPUTE_NODE")
    require(not os.environ.get("SLURM_JOB_GPUS") and "gres/gpu" not in description, "ZERO_GPU_ALLOCATION")
    hardware, topology, affinity = command(["lscpu"]), command(["lscpu", "-p=CPU,CORE,SOCKET,NODE"]), psutil.Process().cpu_affinity()
    rows = [line.split(",") for line in topology.splitlines() if line and not line.startswith("#")]
    cores, numas = {(row[2], row[1]) for row in rows if int(row[0]) in affinity}, {row[3] for row in rows if int(row[0]) in affinity}
    require(len(affinity) == len(cores) == 96 and len(numas) == 4, "96_DISTINCT_PHYSICAL_CORES_FOUR_NUMA")
    require("Intel(R) Xeon(R) Platinum 9242" in hardware, "VALIDATED_CPU_MODEL")
    from tools.cpu_o1.cpu_s4 import memory_limits
    memory = memory_limits.snapshot_cgroup_memory(os.getpid(), allocation_bytes=320 * 1024**3)
    require(memory.get("status") == "READ" and int(memory["effective_limit_bytes"]) > 0, "ACTUAL_CGROUP_MEMORY_READ")
    require(all("current_bytes" in row and "hard_limit_bytes" in row for row in memory.get("levels_leaf_to_mount_root", ())), "CGROUP_HIGH_WATER_FILES_READABLE")
    stop = int(.8 * int(memory["effective_limit_bytes"])); require(stop > 0, "ACTUAL_MEMORY_STOP_LIMIT")
    leaf=Path(memory["levels_leaf_to_mount_root"][0]["path"])
    peak_path=leaf/("memory.max_usage_in_bytes" if memory["version"]==1 else "memory.peak")
    peak_value=peak_path.read_text(encoding="ascii").strip()
    require(peak_value.isdecimal() and int(peak_value)<=stop,"ACTUAL_CGROUP_HIGH_WATER_READABLE_BELOW_LIMIT")
    return {"job_id": job, "environment": {"python": sys.version, "executable": sys.executable, "modules": versions}, "allocation": description, "hardware": hardware, "cpu_affinity": affinity, "physical_cores": len(cores), "numa_nodes": sorted(numas), "cgroup": memory, "process_tree_stop_bytes": stop,"allocated_nodes":nodes,"memory_peak_path":str(peak_path),"initial_kernel_cgroup_peak_bytes":int(peak_value)}

def node_gate(report: str | Path, plan: dict[str, Any] | None = None) -> dict[str, Any]:
    report = _path(report); plan = _plan(report) if plan is None else plan
    expected = os.environ.get("CPU_G3_FULL_PLAN_SHA256")
    require(expected and pv.raw_sha256_file(report / "FULLSIZE_CONFIRMATION_PLAN.json") == expected, "FULL_PLAN_RAW_IDENTITY")
    source, inputs, runtime = _source_gate(plan), _sealed_inputs(plan), _runtime_gate(plan)
    result = {"schema": SCHEMA + ".node_gate.v1", "status": "PASS_FULLSIZE_NODE_SOURCE_AND_INPUT_GATES", "job_id": runtime["job_id"], "source": source, "sealed_inputs": inputs, "runtime": runtime, "fullsize_arrays_created": 0, "Hydro_called": False}
    sg.write(report / "NODE_GATE.json", result); return result

def _tree_rss(process: Any) -> int:
    import psutil
    try: root = psutil.Process(process.pid)
    except psutil.Error: return 0 if process.poll() is not None else -1
    try: members = [root, *root.children(recursive=True)]
    except psutil.Error: members = [root]
    total = 0
    for member in members:
        try: total += int(member.memory_info().rss)
        except psutil.NoSuchProcess: pass
    return total

def launch(report: str | Path, run_root: str | Path, plan: dict[str, Any], case_id: str, role: str, mode: str, *, reference: str | Path | None = None, boundary: str | Path | None = None, stages: str | Path | None = None, deadline: float, limit: int) -> dict[str, Any]:
    import psutil
    from tools.cpu_o1.cpu_s4.supervisor import terminate
    report, run_root, folder = _path(report), Path(run_root).resolve(), Path(run_root).resolve() / case_id
    folder.mkdir(exist_ok=False); env = os.environ.copy(); env.pop("PYTEST_ADDOPTS", None); env["NUMBA_NUM_THREADS"] = "48"
    env["NUMBA_CACHE_DIR"] = str(Path(plan.get("numba_cache_root", str(report / "cache"))).resolve() / role)
    argv = [sys.executable, "-s", "-B", "-m", "tools.cpu_s4_fullsize_confirmation.entry", "--report", str(report), "--work", str(folder / "work"), "--role", role, "--mode", mode]
    for flag, value in (("--reference-output", reference), ("--reference-boundary", boundary), ("--reference-stages", stages)):
        if value is not None: argv.extend((flag, str(value)))
    sg.write(folder / "LAUNCH.json", {"schema": SCHEMA + ".launch.v1", "argv": argv, "role": role, "mode": mode, "source_plan_sha256": os.environ.get("CPU_G3_FULL_PLAN_SHA256"), "UTC": dt.datetime.now(dt.timezone.utc).isoformat()})
    with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open("xb") as stderr: process = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    created, peak, samples, reason, started = psutil.Process(process.pid).create_time(), 0, 0, None, time.monotonic()
    peak_path=Path(sg.read(report/"NODE_GATE.json")["runtime"]["memory_peak_path"]); kernel_peak=0
    try:
        while process.poll() is None:
            child_rss = _tree_rss(process)
            if child_rss < 0: reason = "PROCESS_IDENTITY_LOST"; break
            rss = child_rss+psutil.Process().memory_info().rss
            peak, samples = max(peak, rss), samples + 1
            kernel_peak=max(kernel_peak,int(peak_path.read_text(encoding="ascii").strip()))
            if rss > limit or kernel_peak > limit: reason = "PROCESS_TREE_OR_KERNEL_CGROUP_MEMORY_LIMIT"; break
            if time.monotonic() >= deadline: reason = "ALLOCATION_DEADLINE_RESERVE"; break
            time.sleep(1.0)
    finally:
        if process.poll() is None: terminate(process.pid, created)
    code = process.wait(timeout=10)
    kernel_peak=max(kernel_peak,int(peak_path.read_text(encoding="ascii").strip()))
    if kernel_peak>limit:reason="KERNEL_CGROUP_HIGH_WATER_LIMIT"
    receipt = {"schema": SCHEMA + ".supervisor.v1", "exit_code": code, "stop_reason": reason, "process_service_s": time.monotonic() - started, "peak_tree_rss_bytes": peak, "memory_sample_seconds": 1.0, "memory_sample_count": samples,"kernel_cgroup_peak_bytes":kernel_peak,"peak_scope":"Actual job cgroup high-water including charged cache; process-tree RSS sampled separately"}
    sg.write(folder / "SUPERVISOR_RECEIPT.json", receipt); require(code == 0 and reason is None, "STOP_CHILD_FAILURE:" + case_id, receipt)
    result = sg.read(folder / "work/CASE_RECEIPT.json"); require(result.get("execution_domain") == "FULLSIZE_CONFIRMATION", "FULLSIZE_CASE_DOMAIN")
    for execution in result.get("route", {}).get("actual_numba_kernel_executions", ()):
        require(execution.get("requested_workers") == 48 and execution.get("thread_mask_before") == 24 and execution.get("thread_mask_inside") == 48 and execution.get("thread_mask_after") == 24, "REAL_W48_OUTER24_MASKS")
    result.update(case_id=case_id, supervisor=receipt)
    print(json.dumps(dict(case=case_id,status=result["status"],application_service_s=result["application_service_s"],process_service_s=receipt["process_service_s"])),flush=True)
    return result

def _boundary(run_root: Path, case_id: str) -> Path: return run_root / case_id / "work" / "formal" / "BOUNDARY_RECEIPT.json"

def _exact(result: dict[str, Any], *names: str) -> None:
    expected = {"terminal": "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS", "boundary": "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT", "stages": "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"}
    for name in names: require(result.get("comparisons", {}).get(name, {}).get("status") == expected[name], "FULLSIZE_EXACT_REQUIRED:" + name)

def _csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({key: row.get(key) for key in fields})

def run(report: str | Path) -> dict[str, Any]:
    service_started=time.monotonic()
    report, rows, qualified = _path(report), [], False
    try:
        plan, gate = _plan(report), None; gate = node_gate(report, plan); root = report / "runs" / "A01"; root.mkdir(parents=True, exist_ok=False)
        deadline, limit = service_started + int(plan["walltime_s"]) - 60, gate["runtime"]["process_tree_stop_bytes"]
        from .entry import SOURCE_SHA,PRE_SHA
        sg.write(report/"FULL_INPUT_MANIFEST.json",dict(schema="filament.fullsize.frozen_inputs.v1",shape=list(FULL_SHAPE),dtype="complex128",native_layout="C",fingerprints=dict(source=SOURCE_SHA,PRE=PRE_SHA),input_kind="ORIGINAL_SEALED_G2_SOURCE_PRE",source_root=plan["full_input_root"],full_input_records=plan["full_input_records"]))
        g2 = launch(report, root, plan, "QUALIFY_G2_FULL", "G2_FULL", "QUALIFY_BOUNDARY", reference=plan["reference_output"], boundary=plan["reference_boundary"], stages=plan.get("reference_stages"), deadline=deadline, limit=limit); rows.append(g2)
        _exact(g2, "terminal", "boundary")
        expected_nh = int(plan.get("expected_nH", 16)); require(g2["route"]["nH"] == expected_nh, "FROZEN_G2_NH", g2["route"])
        sg.write(report / "QUALIFIED_STEP_CONTRACT.json", {"nH": expected_nh, "intervals": 8, "source": "Actual G2_FULL qualification"})
        g3 = launch(report, root, plan, "QUALIFY_G3_FULL", "G3_FULL", "QUALIFY_BOUNDARY", reference=g2["output"], boundary=_boundary(root, "QUALIFY_G2_FULL"), stages=g2.get("stages"), deadline=deadline, limit=limit); rows.append(g3)
        _exact(g3, "terminal", "boundary", "stages")
        require(g3["route"]["nH"] == expected_nh, "FROZEN_G3_NH", g3["route"]); qualified = True
        for batch, order in FORMAL_ORDER.items():
            previous = None
            for role in order:
                current = launch(report, root, plan, batch + "_" + role, role, "PURE_OFF_SMALLGRID", reference=g2["output"], deadline=deadline, limit=limit); rows.append(current)
                _exact(current, "terminal")
                previous = current
        paired = []
        for batch in FORMAL_ORDER:
            values = {row["role"]: row for row in rows if row["case_id"].startswith(batch + "_")}; require(set(values) == {"G2_FULL", "G3_FULL"}, "FORMAL_PAIR_COMPLETE", batch); line = {"batch": batch}
            for clock in ("optical_application_window_s", "optical_compute_window_s"):
                g2t, g3t = float(values["G2_FULL"][clock]), float(values["G3_FULL"][clock]); require(math.isfinite(g2t) and math.isfinite(g3t) and g2t >= .2 and g3t >= .2, "FULLSIZE_TIMING_RESOLUTION", batch); line["G2_FULL_" + clock], line["G3_FULL_" + clock] = g2t, g3t; line["g_incremental_" + clock] = (g2t - g3t) / g2t
            paired.append(line)
        gains = [row["g_incremental_optical_application_window_s"] for row in paired]; positive = sum(value > 0 for value in gains)
        result = {"schema": SCHEMA + ".result.v1", "status": "FULLSIZE_REAL8_INCREMENTAL_GAIN_PASS" if statistics.median(gains) >= .03 and positive >= 2 else "FULLSIZE_REAL8_NO_QUALIFYING_GAIN", "paired": paired, "incremental_median": statistics.median(gains), "positive_batches": positive, "gains": gains, "completed_cases": [row["case_id"] for row in rows], "qualification": "FULLSIZE_EXACT_PASS", "execution_domain": "FULLSIZE_CONFIRMATION", "warmup_intervals_per_formal_case": 0, "default": "I0", "i0_not_run": True, "production_qualified": False, "review_gate": "MANUAL_REVIEW_PENDING"}
        _csv(report / "PAIRED_RESULTS.csv", ("batch", "G2_FULL_optical_application_window_s", "G3_FULL_optical_application_window_s", "g_incremental_optical_application_window_s", "G2_FULL_optical_compute_window_s", "G3_FULL_optical_compute_window_s", "g_incremental_optical_compute_window_s"), paired); sg.write(report / "FULLSIZE_RESULT.json", result); return result
    except BaseException as error:
        status = "FULLSIZE_EXACT_PASS/PERFORMANCE_NOT_COMPLETED" if qualified else "FULLSIZE_QUALIFICATION_FAILED"; sg.write(report / "JOB_FAILURE.json", {"schema": SCHEMA + ".failure.v1", "status": status, "error": repr(error), "traceback": traceback.format_exc(), "completed_cases": [row.get("case_id") for row in rows], "i0_not_run": True}); raise
    finally:
        if rows: _csv(report / "PER_CASE_TIMINGS.csv", ("case_id", "role", "mode", "application_service_s", "optical_application_window_s", "optical_compute_window_s", "initialization_within_application_s", "research_service_s"), rows)

def main(argv: list[str] | None = None) -> int:
    def interrupted(signum,frame):raise InterruptedError("JOB_TERMINATION_SIGNAL:"+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--report", type=Path, required=True); result = run(parser.parse_args(argv).report); print(json.dumps({"status": result["status"], "completed_cases": result["completed_cases"]})); return 0

if __name__ == "__main__": raise SystemExit(main())
