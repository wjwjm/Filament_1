#!/usr/bin/env python3
"""Synthetic CPU-only FFT benchmark. Results are P1 diagnostics, never scientific runs."""
from __future__ import annotations
import argparse, contextlib, csv, hashlib, importlib.util, io, json, math, os
from pathlib import Path
import platform, shutil, signal, statistics, subprocess, sys, time, traceback
from datetime import datetime, timezone

FULL_SHAPE = (384, 351, 301)
SEED = 20261013
METHODS = ("slice_numpy", "batch_numpy", "batch_scipy")
CORE_ORDER = (1, 96, 12, 24, 48)
SCHEMA = "filament.cpu_p1_fft_benchmark.v1"
DIAGNOSTIC = "P1_DIAGNOSTIC_ONLY"
PRODUCTION = "NOT_PRODUCTION_QUALIFIED"
RSS_FRACTION = 0.80
MAX_SECONDS = 5400
MEMORY_BYTES = 320 * 1024**3
SLURM_SAFE_KEYS = {
    "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NUM_NODES", "SLURM_NTASKS",
    "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE",
    "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT", "SLURM_JOB_QOS", "SLURM_JOB_NODELIST",
    "SLURM_JOB_CONSTRAINTS", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_CPU_BIND",
}

def slurm_metadata(env):
    return {k: v for k, v in env.items() if k in SLURM_SAFE_KEYS or k in {"UPPE_USE_GPU", "CUDA_VISIBLE_DEVICES"}}

class BenchmarkError(RuntimeError):
    pass

def add_vendor(vendor_dir):
    if not vendor_dir:
        return []
    root = Path(vendor_dir).expanduser()
    if not root.exists():
        raise BenchmarkError(f"vendor path does not exist: {root}")
    wheels = sorted(root.glob("threadpoolctl-*.whl")) if root.is_dir() else [root]
    paths = [str(p.resolve()) for p in wheels]
    if root.is_dir():
        paths.append(str(root.resolve()))
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    return paths

def write_new(path, data):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        f.write(data)

def write_json(path, data):
    write_new(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

def validate_full_allocation(env):
    expected = {
        "SLURM_JOB_NUM_NODES": "1", "SLURM_NTASKS": "1",
        "SLURM_CPUS_PER_TASK": "96", "SLURM_JOB_PARTITION": "v6_384", "SLURM_JOB_ACCOUNT": "bscc-t",
        "SLURM_JOB_QOS": "normal", "UPPE_USE_GPU": "0",
    }
    bad = {k: {"expected": v, "actual": env.get(k)} for k, v in expected.items()
           if env.get(k) != v}
    for key in ("SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"):
        value = env.get(key)
        if value and value.split(",", 1)[0].split("(", 1)[0] != "96":
            bad[key] = {"expected": "96", "actual": value}
    gpu = {k: env[k] for k in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS",
           "SLURM_GPUS", "SLURM_GPUS_ON_NODE") if env.get(k, "").strip()}
    visible = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "-1":
        gpu["CUDA_VISIBLE_DEVICES"] = visible
    if gpu:
        bad["GPU_ALLOCATION"] = gpu
    if bad:
        raise BenchmarkError("CPU allocation rejected: " + json.dumps(bad, sort_keys=True))

def _parse_cpu_list(value):
    """Parse Linux list syntax such as ``0-3,8,10-11``."""
    result = set()
    for part in value.strip().split(","):
        if not part:
            raise ValueError("empty CPU-list component")
        bounds = part.split("-")
        if len(bounds) == 1:
            first = last = int(bounds[0])
        elif len(bounds) == 2:
            first, last = map(int, bounds)
        else:
            raise ValueError("invalid CPU-list component")
        if first < 0 or last < first:
            raise ValueError("invalid CPU-list range")
        result.update(range(first, last + 1))
    return result


def cpu_topology(allowed=None, sysfs=Path("/sys/devices/system/cpu")):
    if allowed is None:
        allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
            else list(range(os.cpu_count() or 1))
    allowed = sorted({int(cpu) for cpu in allowed})
    try:
        if not allowed:
            raise ValueError("empty CPU affinity")
        groups = {}
        for cpu in allowed:
            topo = sysfs / f"cpu{cpu}" / "topology" / "thread_siblings_list"
            siblings = _parse_cpu_list(topo.read_text().strip())
            if cpu not in siblings:
                raise ValueError(f"CPU {cpu} absent from its sibling list")
            visible = frozenset(siblings.intersection(allowed))
            if not visible:
                raise ValueError(f"CPU {cpu} has no allowed sibling")
            groups[cpu] = visible
        for cpu, siblings in groups.items():
            for sibling in siblings:
                if groups.get(sibling) != siblings:
                    raise ValueError("thread_siblings_list is asymmetric or inconsistent")
        unique = set(groups.values())
        covered = set().union(*unique)
        if covered != set(allowed):
            raise ValueError("thread sibling groups do not cover affinity")
        if sum(len(group) for group in unique) != len(allowed):
            raise ValueError("thread sibling groups overlap")
        cores = sorted(min(group) for group in unique)
        return {"allowed_logical_cpus": allowed,
                "physical_core_representatives": cores,
                "physical_core_count": len(cores),
                "thread_sibling_groups": [sorted(group) for group in sorted(unique, key=min)],
                "topology_method": "sysfs_thread_siblings_list",
                "topology_source": "sysfs_physical_core"}
    except (OSError, ValueError) as exc:
        return {"allowed_logical_cpus": allowed,
                "physical_core_representatives": list(allowed),
                "physical_core_count": None, "thread_sibling_groups": None,
                "topology_method": "unavailable",
                "topology_source": "logical_cpu_fallback",
                "topology_error": f"{type(exc).__name__}: {exc}"}

def physical_topology_verified(affinity):
    return affinity.get("topology_source") == "sysfs_physical_core"

def select_physical_cpus(count, allowed=None, topology=None):
    if count <= 0:
        raise ValueError("count must be positive")
    cores = (topology or cpu_topology(allowed))["physical_core_representatives"]
    if count > len(cores):
        raise BenchmarkError(f"requested {count} physical cores, available={len(cores)}")
    return list(cores[:count])

def build_synthetic_array(shape):
    import numpy as np
    if len(shape) != 3 or any(int(d) <= 0 for d in shape):
        raise ValueError("shape must be three positive dimensions")
    rng = np.random.default_rng(SEED)
    arr = np.empty(shape, dtype=np.complex128, order="C")
    arr.real[...] = rng.standard_normal(shape, dtype=np.float64)
    arr.imag[...] = rng.standard_normal(shape, dtype=np.float64)
    if arr.dtype != np.complex128 or not arr.flags.c_contiguous:
        raise BenchmarkError("synthetic array dtype/layout mismatch")
    return arr

def fingerprint(arr):
    return hashlib.sha256(memoryview(arr).cast("B")).hexdigest()

def error_metrics(expected, got):
    import numpy as np
    if expected.shape != got.shape or expected.dtype != np.complex128 or got.dtype != np.complex128:
        raise BenchmarkError("error comparison requires matching complex128 arrays")
    for start in range(0, expected.shape[0], 16):
        if not np.isfinite(expected[start:start+16]).all() or not np.isfinite(got[start:start+16]).all():
            raise BenchmarkError("error comparison encountered NaN or Inf")
    max_abs = max_ref = err2 = ref2 = 0.0
    for start in range(0, expected.shape[0], 16):
        ref = expected[start:start+16]
        diff = got[start:start+16] - ref
        max_abs = max(max_abs, float(np.max(np.abs(diff), initial=0)))
        max_ref = max(max_ref, float(np.max(np.abs(ref), initial=0)))
        err2 += float(np.vdot(diff, diff).real)
        ref2 += float(np.vdot(ref, ref).real)
    tiny = float(np.finfo(np.float64).tiny)
    return {"max_abs": max_abs, "max_rel": max_abs / max(max_ref, tiny),
            "l2_rel": math.sqrt(err2) / max(math.sqrt(ref2), tiny)}
def pool_info():
    try:
        import threadpoolctl
        return {"status": "AVAILABLE", "version": getattr(threadpoolctl, "__version__", None),
                "pools": threadpoolctl.threadpool_info()}
    except Exception as exc:
        return {"status": "UNAVAILABLE", "version": None, "pools": None,
                "error": f"{type(exc).__name__}: {exc}"}

def optional_command(name, args):
    exe = shutil.which(name)
    if not exe:
        return {"status": "UNAVAILABLE", "stdout": "UNAVAILABLE\n"}
    try:
        run = subprocess.run([exe, *args], capture_output=True, text=True,
                             errors="replace", timeout=20, check=False)
        return {"status": "CAPTURED" if run.returncode == 0 else "FAILED",
                "stdout": run.stdout, "stderr": run.stderr, "exit_code": run.returncode}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "UNAVAILABLE", "stdout": "UNAVAILABLE\n", "error": str(exc)}

def environment(root, vendor, shape):
    import numpy as np
    cfg = io.StringIO()
    with contextlib.redirect_stdout(cfg):
        np.__config__.show()
    slurm = slurm_metadata(os.environ)
    write_json(root / "slurm_environment.json", slurm)
    for name, source in (("proc_status.txt", "/proc/self/status"),
                         ("proc_cpuinfo.txt", "/proc/cpuinfo"),
                         ("proc_meminfo.txt", "/proc/meminfo")):
        p = Path(source)
        write_new(root / name, p.read_text(errors="replace") if p.is_file() else "UNAVAILABLE\n")
    lscpu, numa = optional_command("lscpu", []), optional_command("numactl", ["--hardware"])
    write_new(root / "lscpu.txt", lscpu["stdout"])
    write_new(root / "numactl_hardware.txt", numa["stdout"])
    write_new(root / "numpy_config.txt", cfg.getvalue())
    pools = pool_info()
    write_json(root / "threadpool_info.json", pools)
    topo = cpu_topology()
    affinity = {**topo, "full_run_core_order": list(CORE_ORDER),
                "shape_slice_y_x": list(shape), "dtype": "complex128"}
    write_json(root / "affinity.json", affinity)
    try:
        import scipy
        scipy_ver = scipy.__version__
    except Exception:
        scipy_ver = None
    meta = {
        "schema": SCHEMA, "classification": DIAGNOSTIC, "production_qualification": PRODUCTION,
        "input_kind": "DETERMINISTIC_SYNTHETIC_ONLY", "scientific_inputs_used": False,
        "seed": SEED, "array_shape_slice_y_x": list(shape), "array_dtype": "complex128",
        "array_layout": "C contiguous; slices axis 0; FFT axes (-2,-1)",
        "fingerprint": "SHA256 of C-order array bytes", "python": sys.version,
        "python_executable": sys.executable, "platform": platform.platform(),
        "numpy_version": np.__version__, "scipy_version": scipy_ver,
        "threadpoolctl": pools, "vendor_paths": vendor, "lscpu": lscpu,
        "numactl": numa, "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "environment.json", meta)
    return affinity

def fft_functions(method, workers):
    import numpy as np
    if method == "slice_numpy":
        def fwd(a):
            out = np.empty_like(a)
            for i in range(a.shape[0]):
                out[i] = np.fft.fft2(a[i], axes=(-2,-1))
            return out
        def inv(a):
            out = np.empty_like(a)
            for i in range(a.shape[0]):
                out[i] = np.fft.ifft2(a[i], axes=(-2,-1))
            return out
        return fwd, inv, "THREADING_NOT_SUPPORTED_BY_BASELINE"
    if method == "batch_numpy":
        return (lambda a: np.fft.fft2(a, axes=(-2,-1)),
                lambda a: np.fft.ifft2(a, axes=(-2,-1)),
                "THREADING_NOT_SUPPORTED_BY_BASELINE")
    from scipy import fft
    return (lambda a: fft.fft2(a, axes=(-2,-1), workers=workers),
            lambda a: fft.ifft2(a, axes=(-2,-1), workers=workers),
            "SCIPY_WORKERS_EXPLICIT")

def timed_pair(arr, fwd, inv, reference_spectrum):
    import numpy as np
    pwall, pcpu = time.perf_counter(), time.process_time()
    fw, fc = time.perf_counter(), time.process_time()
    spectrum = fwd(arr)
    fw, fc = time.perf_counter()-fw, time.process_time()-fc
    if spectrum.shape != arr.shape or spectrum.dtype != np.complex128:
        raise BenchmarkError(f"forward FFT shape/dtype mismatch: {spectrum.shape}/{spectrum.dtype}")
    iw, ic = time.perf_counter(), time.process_time()
    recovered = inv(spectrum)
    iw, ic = time.perf_counter()-iw, time.process_time()-ic
    pair = time.perf_counter()-pwall
    cpupair = time.process_time()-pcpu
    if recovered.shape != arr.shape or recovered.dtype != np.complex128:
        raise BenchmarkError("inverse FFT shape/dtype mismatch")
    return {"forward_wall_seconds": fw, "forward_cpu_seconds": fc,
            "inverse_wall_seconds": iw, "inverse_cpu_seconds": ic,
            "pair_wall_seconds": pair, "pair_cpu_seconds": cpupair,
            "roundtrip_error": error_metrics(arr, recovered),
            "forward_spectrum_error": error_metrics(reference_spectrum, spectrum)}
def child_case(args):
    vendor = add_vendor(args.vendor_dir)
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    topo = cpu_topology(allowed)
    cpus = select_physical_cpus(args.workers, allowed, topo)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpus))
    import numpy as np
    arr = build_synthetic_array(tuple(args.shape))
    fp = fingerprint(arr)
    if args.expected_fingerprint and fp != args.expected_fingerprint:
        raise BenchmarkError("same-array fingerprint mismatch")
    fwd, inv, threading = fft_functions(args.method, args.workers)
    reference_fwd, _, _ = fft_functions("slice_numpy", 1)
    ref_wall, ref_cpu = time.perf_counter(), time.process_time()
    reference_spectrum = reference_fwd(arr)
    reference_wall = time.perf_counter() - ref_wall
    reference_cpu = time.process_time() - ref_cpu
    if reference_spectrum.shape != arr.shape or reference_spectrum.dtype != np.complex128:
        raise BenchmarkError("slice NumPy reference spectrum shape/dtype mismatch")
    error_metrics(reference_spectrum, reference_spectrum)
    pools = pool_info()
    def emit_progress(label, sample):
        print(json.dumps({"status": "PROGRESS", "sample": label, "method": args.method,
                          "workers_requested": args.workers,
                          "array_fingerprint_sha256": fp, "sample_result": sample}, sort_keys=True),
              flush=True)
    warmup = timed_pair(arr, fwd, inv, reference_spectrum)
    emit_progress("warmup", warmup)
    reps = []
    for index in range(3):
        sample = timed_pair(arr, fwd, inv, reference_spectrum)
        reps.append(sample)
        emit_progress(f"rep-{index+1}", sample)
    metrics = {}
    for key in ("forward_wall_seconds", "forward_cpu_seconds", "inverse_wall_seconds",
                "inverse_cpu_seconds", "pair_wall_seconds", "pair_cpu_seconds"):
        vals = sorted(x[key] for x in reps)
        metrics[key] = {"p50": statistics.median(vals), "p90_nearest_rank": vals[2]}
    try:
        import scipy
        scipy_ver = scipy.__version__
    except Exception:
        scipy_ver = None
    return {"schema": SCHEMA, "status": "DONE", "classification": DIAGNOSTIC,
            "production_qualification": PRODUCTION, "input_kind": "DETERMINISTIC_SYNTHETIC_ONLY",
            "method": args.method, "workers_requested": args.workers, "threading_status": threading,
            "affinity_cpus": cpus, "topology_source": topo["topology_source"],
            "array_shape_slice_y_x": list(arr.shape), "array_dtype": str(arr.dtype),
            "array_fingerprint_sha256": fp, "warmup": warmup, "repetitions": reps,
            "metrics": metrics, "forward_reference_method": "slice_numpy",
            "reference_generation": {"wall_seconds": reference_wall, "cpu_seconds": reference_cpu},
            "threadpool_info": pools, "vendor_paths": vendor,
            "numpy_version": np.__version__, "scipy_version": scipy_ver}

def proc_entry(pid):
    try:
        base = Path("/proc") / str(pid)
        fields = dict(line.split(":",1) for line in (base/"status").read_text().splitlines() if ":" in line)
        stat = (base/"stat").read_text()
        parts = stat[stat.rfind(")")+2:].split()
        return {"pid": pid, "ppid": int(parts[1]),
                "rss": int(fields.get("VmRSS","0 kB").split()[0])*1024,
                "cpu": (int(parts[11])+int(parts[12]))/os.sysconf("SC_CLK_TCK")}
    except (OSError, ValueError, IndexError):
        return None

def tree_snapshot(pid):
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    entries = {}
    for path in proc.iterdir():
        if path.name.isdigit():
            item = proc_entry(int(path.name))
            if item:
                entries[item["pid"]] = item
    pids, changed = {pid}, True
    while changed:
        changed = False
        for child, item in entries.items():
            if item["ppid"] in pids and child not in pids:
                pids.add(child); changed = True
    group = [entries[x] for x in pids if x in entries]
    return {"rss": sum(x["rss"] for x in group), "cpu": sum(x["cpu"] for x in group)}

def kill_child(proc):
    if proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass

CSV_FIELDS = ("case_id,status,method,workers,threading_status,shape,dtype,fingerprint,"
"forward_wall_p50,forward_wall_p90,forward_cpu_p50,forward_cpu_p90,"
"inverse_wall_p50,inverse_wall_p90,inverse_cpu_p50,inverse_cpu_p90,"
"pair_wall_p50,pair_wall_p90,pair_cpu_p50,pair_cpu_p90,roundtrip_max_abs,roundtrip_max_rel,roundtrip_l2_rel,forward_spectrum_max_abs,forward_spectrum_max_rel,forward_spectrum_l2_rel,"
"peak_tree_rss_bytes,tree_cpu_seconds,child_exit_code,note").split(",")

def csv_row(cid, result, accounting, shape):
    row = {"case_id":cid, "status":result.get("status","FAIL"), "method":result.get("method",""),
           "workers":result.get("workers_requested",""), "threading_status":result.get("threading_status",""),
           "shape":"x".join(map(str,shape)), "dtype":result.get("array_dtype","complex128"),
           "fingerprint":result.get("array_fingerprint_sha256",""),
           "peak_tree_rss_bytes":accounting.get("peak_tree_rss_bytes"),
           "tree_cpu_seconds":accounting.get("tree_cpu_seconds"),
           "child_exit_code":accounting.get("child_exit_code"), "note":result.get("note","")}
    last = (result.get("repetitions") or [{}])[-1]
    for label, key in (("roundtrip", "roundtrip_error"), ("forward_spectrum", "forward_spectrum_error")):
        for metric in ("max_abs", "max_rel", "l2_rel"):
            row[f"{label}_{metric}"] = (last.get(key) or {}).get(metric, "")
    for name in ("forward_wall","forward_cpu","inverse_wall","inverse_cpu","pair_wall","pair_cpu"):
        source = name.replace("_", "_") + "_seconds"
        values = result.get("metrics",{}).get(source,{})
        row[name+"_p50"] = values.get("p50","")
        row[name+"_p90"] = values.get("p90_nearest_rank","")
    return row

def append_result(jsonl, writer, ch, record, result, accounting, shape):
    jsonl.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+"\n"); jsonl.flush(); os.fsync(jsonl.fileno())
    writer.writerow(csv_row(record["case_id"],result,accounting,shape)); ch.flush(); os.fsync(ch.fileno())

def create_output_root(path):
    output = Path(path).expanduser().absolute()
    if not output.parent.is_dir():
        raise BenchmarkError("output parent must already exist")
    output.mkdir(mode=0o750, parents=False, exist_ok=False)
    return output

def run_parent(args):
    vendor = add_vendor(args.vendor_dir)
    if not args.smoke:
        validate_full_allocation(os.environ)
        if not Path("/proc").is_dir():
            raise BenchmarkError("Linux /proc required for RSS/CPU-time monitoring")
        if not 0 < args.deadline_seconds <= MAX_SECONDS:
            raise BenchmarkError(f"deadline must be <= {MAX_SECONDS} seconds")
        shape, counts = FULL_SHAPE, CORE_ORDER
        if pool_info()["version"] != "3.6.0":
            raise BenchmarkError("threadpoolctl 3.6.0 required; provide the frozen vendor module")
    else:
        shape, counts = tuple(args.smoke_shape), (1,)
        if len(shape) != 3 or any(d <= 0 for d in shape):
            raise BenchmarkError("smoke shape must have three positive dimensions")
    if args.deadline_seconds <= 0 or args.memory_limit_bytes <= 0:
        raise BenchmarkError("deadline and memory limit must be positive")
    if not importlib.util.find_spec("numpy"):
        raise BenchmarkError("NumPy required")
    output = create_output_root(Path(args.output_root))
    affinity = environment(output,vendor,shape)
    if not args.smoke and not physical_topology_verified(affinity):
        status={"schema":SCHEMA,"classification":DIAGNOSTIC,"production_qualification":PRODUCTION,
                "status":"BLOCKED_PHYSICAL_TOPOLOGY_UNAVAILABLE","case_count_including_skips":0,
                "topology_source":affinity["topology_source"],"scientific_inputs_used":False}
        write_json(output/"status.json",status)
        print(json.dumps({"status":status["status"],"output_root":str(output),"cases":0},sort_keys=True))
        return 1
    cores = affinity["physical_core_representatives"]
    started, deadline, expected_fp = time.monotonic(), time.monotonic()+args.deadline_seconds, None
    state, ncase, incomplete = "COMPLETE", 0, False
    jp, cp = output/"benchmark_results.jsonl", output/"performance_summary.csv"
    with jp.open("x",encoding="utf-8",newline="\n") as jf, cp.open("x",encoding="utf-8",newline="") as cf:
        writer = csv.DictWriter(cf,fieldnames=CSV_FIELDS); writer.writeheader(); cf.flush()
        for workers in counts:
            for method in METHODS:
                ncase += 1
                cid=f"case-{ncase:02d}-{method}-w{workers}"
                if time.monotonic() >= deadline:
                    state="LIMIT_REACHED"; result={"status":state,"note":"Overall deadline before case."}
                    append_result(jf,writer,cf,{"case_id":cid,**result,"classification":DIAGNOSTIC},result,{},shape)
                    break
                if workers > len(cores):
                    result={"status":"SKIPPED_INSUFFICIENT_PHYSICAL_CORES","method":method,
                            "workers_requested":workers,"note":f"physical cores available={len(cores)}"}
                    append_result(jf,writer,cf,{"case_id":cid,**result,"classification":DIAGNOSTIC},result,{},shape)
                    incomplete = True
                    continue
                if method=="batch_scipy" and not importlib.util.find_spec("scipy"):
                    result={"status":"SKIPPED_SCIPY_UNAVAILABLE","method":method,"workers_requested":workers}
                    append_result(jf,writer,cf,{"case_id":cid,**result,"classification":DIAGNOSTIC},result,{},shape)
                    incomplete = True
                    continue
                cmd=[sys.executable,str(Path(__file__).resolve()),"--case-method",method,
                     "--workers",str(workers),"--shape",*map(str,shape)]
                if args.vendor_dir: cmd += ["--vendor-dir",args.vendor_dir]
                if expected_fp: cmd += ["--expected-fingerprint",expected_fp]
                stdout_path, stderr_path = output/f"{cid}.stdout.txt", output/f"{cid}.stderr.txt"
                with stdout_path.open("x", encoding="utf-8", newline="\n") as stdout_file, \
                     stderr_path.open("x", encoding="utf-8", newline="\n") as stderr_file:
                    kw={"stdout":stdout_file,"stderr":stderr_file}
                    if os.name!="nt": kw["start_new_session"]=True
                    proc=subprocess.Popen(cmd,**kw)
                    peak,cpu_seen,monitor,outcome=0,0.0,False,None
                    while proc.poll() is None:
                        snap=tree_snapshot(proc.pid)
                        if snap is not None:
                            monitor=True; peak=max(peak,snap["rss"]); cpu_seen=max(cpu_seen,snap["cpu"])
                            if peak >= args.memory_limit_bytes*RSS_FRACTION:
                                outcome="STOPPED_RSS_80_PERCENT"; kill_child(proc); break
                        if time.monotonic() >= deadline:
                            outcome="LIMIT_REACHED"; kill_child(proc); break
                        time.sleep(0.25)
                stdout=stdout_path.read_text(encoding="utf-8",errors="replace")
                stderr=stderr_path.read_text(encoding="utf-8",errors="replace")
                parsed=[]
                for line in stdout.splitlines():
                    try:
                        item=json.loads(line)
                        if isinstance(item,dict): parsed.append(item)
                    except json.JSONDecodeError:
                        continue
                progress=[item for item in parsed if item.get("status")=="PROGRESS"]
                result=next((item for item in reversed(parsed)
                             if item.get("status") in {"DONE","FAIL"}),None)
                if not isinstance(result,dict):
                    result={"status":"FAIL","method":method,"workers_requested":workers,
                            "note":stderr.strip()[-2000:] or "Child emitted no final JSON."}
                if progress:
                    result["progress_samples"]=progress
                    result.setdefault("repetitions", [item["sample_result"] for item in progress
                                                       if item.get("sample", "").startswith("rep-")])
                if outcome:
                    result["status"]=outcome; result["note"]=outcome+"; partial stdout retained."
                    state="LIMIT_REACHED" if outcome=="LIMIT_REACHED" else "RESOURCE_STOPPED"
                elif proc.returncode:
                    result["status"]="FAIL"; state="PARTIAL_FAILURE"
                fp=result.get("array_fingerprint_sha256")
                if result.get("status")=="DONE":
                    if expected_fp and fp != expected_fp:
                        result["status"]="FAIL"; result["note"]="same-array fingerprint mismatch"; state="PARTIAL_FAILURE"
                    else:
                        expected_fp=fp or expected_fp
                accounting={"peak_tree_rss_bytes":peak if monitor else None,
                            "tree_cpu_seconds":cpu_seen if monitor else None,
                            "child_exit_code":proc.returncode,"process_tree_monitor":"PROCFS" if monitor else "UNAVAILABLE",
                            "memory_limit_bytes":args.memory_limit_bytes,
                            "rss_stop_threshold_bytes":int(args.memory_limit_bytes*RSS_FRACTION),
                            "overall_elapsed_seconds":time.monotonic()-started}
                append_result(jf,writer,cf,{"case_id":cid,**result,"parent_accounting":accounting},
                              result,accounting,shape)
                if state!="COMPLETE": break
            if state!="COMPLETE": break
    if state == "COMPLETE" and incomplete:
        state = "INCOMPLETE"
    status={"schema":SCHEMA,"classification":DIAGNOSTIC,"production_qualification":PRODUCTION,
            "status":state,"case_count_including_skips":ncase,"shape_slice_y_x":list(shape),
            "dtype":"complex128","input_kind":"DETERMINISTIC_SYNTHETIC_ONLY",
            "same_array_fingerprint_sha256":expected_fp,"elapsed_wall_seconds":time.monotonic()-started,
            "deadline_seconds":args.deadline_seconds,"memory_limit_bytes":args.memory_limit_bytes,
            "scientific_inputs_used":False,"fft_results_are_scientific_evidence":False}
    write_json(output/"status.json",status)
    print(json.dumps({"status":state,"output_root":str(output),"cases":ncase},sort_keys=True))
    return 0 if state=="COMPLETE" else 1

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root"); p.add_argument("--vendor-dir")
    p.add_argument("--smoke",action="store_true",help="tiny synthetic smoke; bypasses Slurm gate")
    p.add_argument("--smoke-shape",nargs=3,type=int,default=(16,12,8))
    p.add_argument("--deadline-seconds",type=float,default=4500)
    p.add_argument("--memory-limit-bytes",type=int,default=MEMORY_BYTES)
    p.add_argument("--case-method",choices=METHODS,help=argparse.SUPPRESS)
    p.add_argument("--workers",type=int,help=argparse.SUPPRESS)
    p.add_argument("--shape",nargs=3,type=int,help=argparse.SUPPRESS)
    p.add_argument("--expected-fingerprint",help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.case_method:
        args.method = args.case_method
        try:
            if not args.workers or not args.shape: raise BenchmarkError("incomplete child args")
            print(json.dumps(child_case(args),sort_keys=True)); return 0
        except Exception as exc:
            print(json.dumps({"status":"FAIL","method":args.case_method,
                              "error":f"{type(exc).__name__}: {exc}",
                              "traceback":traceback.format_exc()},sort_keys=True)); return 2
    if not args.output_root: p.error("--output-root required")
    try: return run_parent(args)
    except Exception as exc:
        print(f"cpu_p1_fft failed: {type(exc).__name__}: {exc}",file=sys.stderr); return 2

if __name__=="__main__":
    raise SystemExit(main())

