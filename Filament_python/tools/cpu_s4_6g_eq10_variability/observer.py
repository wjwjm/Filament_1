"""Low-frequency, scalar-only observations for one 6F child process.

The observer is deliberately independent of the propagation process. It reads
Linux procfs counters and writes bounded JSONL metadata; it never imports
NumPy, accesses scientific arrays, or changes process affinity/policy.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterable

try:
    import psutil
except ImportError:  # pragma: no cover - the HPC environment supplies psutil
    psutil = None


MAX_SAMPLES = 1450
MAX_THREADS = 512
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
DEFAULT_PERIOD_S = 5.0
SCHEMA = "filament.cpu_s46g.telemetry.v1"


class ObservationError(RuntimeError):
    """A sampler contract or process identity failure."""


def _clock_metadata() -> dict[str, Any]:
    monotonic = time.get_clock_info("monotonic")
    perf_counter = time.get_clock_info("perf_counter")
    return {
        "sample_timestamp": "time.monotonic_ns",
        "collection_duration": "time.perf_counter_ns",
        "monotonic_implementation": monotonic.implementation,
        "perf_counter_implementation": perf_counter.implementation,
        "same_implementation": monotonic.implementation == perf_counter.implementation,
        "monotonic_resolution_s": monotonic.resolution,
        "perf_counter_resolution_s": perf_counter.resolution,
    }


def _units_metadata() -> dict[str, Any]:
    try:
        ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
    except (AttributeError, OSError, ValueError):
        ticks_per_second = None
    try:
        page_size_bytes = int(os.sysconf("SC_PAGESIZE"))
    except (AttributeError, OSError, ValueError):
        page_size_bytes = None
    return {
        "proc_stat_ticks": "jiffies",
        "proc_stat_ticks_per_second": ticks_per_second,
        "rss_pages": "pages",
        "page_size_bytes": page_size_bytes,
        "status_memory": "kB",
        "schedstat": "ns",
        "wall_clock": "ns",
        "cpu_mhz": "MHz",
    }


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _unavailable(reason: str) -> dict[str, str]:
    return {"status": "UNAVAILABLE", "reason": str(reason)}


def _int_field(value: str, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ObservationError(f"PROC_FIELD_INVALID:{label}") from exc


def parse_proc_stat(text: str) -> dict[str, int | str]:
    """Parse /proc/PID/stat without breaking names containing spaces or ')'."""

    left = text.find("(")
    right = text.rfind(")")
    if left <= 0 or right <= left + 1:
        raise ObservationError("PROC_STAT_COMMAND_NAME_BOUNDARY")
    prefix = text[:left].strip()
    tail = text[right + 1 :].strip().split()
    if not prefix or len(tail) < 37:
        raise ObservationError("PROC_STAT_FIELD_COUNT")
    # tail[0] is field 3 (state); the remaining indexes are field-3.
    return {
        "pid": _int_field(prefix, "pid"),
        "state": tail[0],
        "utime_ticks": _int_field(tail[11], "utime"),
        "stime_ticks": _int_field(tail[12], "stime"),
        "minflt": _int_field(tail[7], "minflt"),
        "majflt": _int_field(tail[9], "majflt"),
        "starttime_ticks": _int_field(tail[19], "starttime"),
        "rss_pages": _int_field(tail[21], "rss"),
        "num_threads": _int_field(tail[17], "num_threads"),
        "last_cpu": _int_field(tail[36], "processor"),
    }


def parse_schedstat(text: str) -> dict[str, int]:
    fields = text.strip().split()
    if len(fields) < 3:
        raise ObservationError("PROC_SCHEDSTAT_FIELD_COUNT")
    return {
        "runtime_ns": _int_field(fields[0], "sched_runtime"),
        "runqueue_wait_ns": _int_field(fields[1], "sched_wait"),
        "timeslices": _int_field(fields[2], "sched_timeslices"),
    }


def parse_status(text: str, names: Iterable[str]) -> dict[str, Any]:
    wanted = set(names)
    value: dict[str, Any] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        if key not in wanted:
            continue
        raw = raw.strip()
        if key in {"VmRSS", "voluntary_ctxt_switches", "nonvoluntary_ctxt_switches"}:
            first = raw.split()[0] if raw else ""
            value[key] = _int_field(first, key) if first else _unavailable("EMPTY_STATUS_VALUE")
        else:
            value[key] = raw
    for key in wanted:
        value.setdefault(key, _unavailable("STATUS_FIELD_MISSING"))
    return value


def parse_cpu_stat(text: str) -> dict[str, int] | dict[str, str]:
    line = next((row for row in text.splitlines() if row.startswith("cpu ")), "")
    fields = line.split()
    if len(fields) < 6:
        raise ObservationError("CPU_STAT_FIELDS_MISSING")
    return {"user": _int_field(fields[1], "cpu_user"), "nice": _int_field(fields[2], "cpu_nice"),
            "system": _int_field(fields[3], "cpu_system"), "idle": _int_field(fields[4], "cpu_idle"),
            "iowait": _int_field(fields[5], "cpu_iowait")}


def summarize_numa_maps(text: str) -> dict[str, int] | dict[str, str]:
    """Aggregate only N<node>=<pages>; mapping addresses and paths are dropped."""

    total: dict[str, int] = {}
    for line in text.splitlines():
        for token in line.split():
            if not token.startswith("N") or "=" not in token:
                continue
            node, pages = token.split("=", 1)
            if not node[1:].isdigit() or not pages.isdigit():
                continue
            total[node] = total.get(node, 0) + int(pages)
    return total or _unavailable("NUMA_NODE_TOKENS_MISSING")


def _safe_file(path: Path, parser, *args) -> Any:
    try:
        return parser(_read(path), *args)
    except FileNotFoundError:
        return _unavailable("FILE_MISSING")
    except PermissionError:
        return _unavailable("PERMISSION_DENIED")
    except ObservationError as exc:
        return _unavailable(str(exc))
    except OSError as exc:
        return _unavailable(type(exc).__name__)


def _safe_text(path: Path) -> str | dict[str, str]:
    try:
        return _read(path).strip()
    except FileNotFoundError:
        return _unavailable("FILE_MISSING")
    except PermissionError:
        return _unavailable("PERMISSION_DENIED")
    except OSError as exc:
        return _unavailable(type(exc).__name__)


def _proc_uid(status: dict[str, Any]) -> int | None:
    raw = status.get("Uid")
    if not isinstance(raw, str):
        return None
    try:
        return int(raw.split()[0])
    except (ValueError, IndexError):
        return None


def _pid_identity(pid: int, proc_root: Path) -> tuple[float, int | None, int]:
    if psutil is None:
        raise ObservationError("PSUTIL_REQUIRED")
    try:
        process = psutil.Process(pid)
        create_time = float(process.create_time())
    except psutil.NoSuchProcess as exc:
        raise ObservationError("PID_NOT_OBSERVABLE") from exc
    except psutil.AccessDenied as exc:
        raise ObservationError("PID_ACCESS_DENIED") from exc
    try:
        status = parse_status(_read(proc_root / str(pid) / "status"), {"Uid"})
    except FileNotFoundError as exc:
        raise ObservationError("PID_NOT_OBSERVABLE") from exc
    uid = _proc_uid(status)
    own_uid = os.getuid() if hasattr(os, "getuid") else uid
    if uid is None or own_uid is not None and uid != own_uid:
        raise ObservationError("PID_OWNER_MISMATCH")
    try:
        stat = parse_proc_stat(_read(proc_root / str(pid) / "stat"))
    except FileNotFoundError as exc:
        raise ObservationError("PID_NOT_OBSERVABLE") from exc
    return create_time, uid, int(stat["starttime_ticks"])


def _thread_ids_with_bound(task_root: Path) -> tuple[list[int], int, bool, str | None]:
    try:
        values = sorted(int(path.name) for path in task_root.iterdir() if path.name.isdigit())
    except FileNotFoundError:
        return [], 0, False, "FILE_MISSING"
    except PermissionError:
        return [], 0, False, "PERMISSION_DENIED"
    except OSError as exc:
        return [], 0, False, type(exc).__name__
    if not values:
        return [], 0, False, "EMPTY"
    return values[:MAX_THREADS], len(values), len(values) > MAX_THREADS, None


def _thread_ids(task_root: Path) -> list[int]:
    return _thread_ids_with_bound(task_root)[0]


def _thread_sample(task_root: Path, tid: int) -> dict[str, Any]:
    root = task_root / str(tid)
    stat = _safe_file(root / "stat", parse_proc_stat)
    status = _safe_file(root / "status", parse_status, {"Cpus_allowed_list", "Mems_allowed_list", "voluntary_ctxt_switches", "nonvoluntary_ctxt_switches"})
    sched = _safe_file(root / "schedstat", parse_schedstat)
    return {"tid": tid, "stat": stat, "status": status, "schedstat": sched}


def _node_sample(proc_root: Path) -> dict[str, Any]:
    proc = proc_root
    cpuinfo = _safe_text(proc / "cpuinfo")
    mhz: list[float] = []
    processors: list[int] = []
    if isinstance(cpuinfo, str):
        for line in cpuinfo.splitlines():
            if line.lower().startswith("processor") and ":" in line:
                try:
                    processors.append(int(line.split(":", 1)[1].strip()))
                except ValueError:
                    pass
            if line.lower().startswith("cpu mhz") and ":" in line:
                try:
                    mhz.append(float(line.split(":", 1)[1].strip()))
                except ValueError:
                    pass
    if not mhz:
        cpu_mhz: dict[str, Any] = _unavailable("CPU_MHZ_FIELDS_MISSING" if processors else "CPUINFO_FIELDS_MISSING")
        cpu_mhz["processor_count"] = len(processors) or None
    else:
        cpu_mhz = {"status": "MEASURED", "processor_count": len(processors) or None,
                   "count": len(mhz), "min": min(mhz), "mean": sum(mhz) / len(mhz), "max": max(mhz)}
    pressure = {}
    for name in ("cpu", "memory", "io"):
        pressure[name] = _safe_text(proc / "pressure" / name)
    thermal = {}
    for path in glob.glob("/sys/devices/system/cpu/cpu*/thermal_throttle/*"):
        value = _safe_text(Path(path))
        parts = Path(path).parts
        cpu_name = next((part for part in parts if part.startswith("cpu") and part[3:].isdigit()), "cpu?")
        thermal[f"{cpu_name}/{Path(path).name}"] = value
    return {
        "proc_cpu_stat": _safe_file(proc / "stat", parse_cpu_stat),
        "cpu_mhz": cpu_mhz,
        "loadavg": _safe_text(proc / "loadavg"), "pressure": pressure,
        "sched_schedstats": _safe_text(proc / "sys/kernel/sched_schedstats"),
        "numa_balancing": _safe_text(proc / "sys/kernel/numa_balancing"),
        "thermal_throttle": thermal or _unavailable("NO_THERMAL_COUNTERS"),
    }


def probe_current_pid(proc_root: str | Path = "/proc", *, pid: int | None = None) -> dict[str, Any]:
    """Read-only capability probe; it creates no observer thread or output."""

    proc_root = Path(proc_root)
    pid = os.getpid() if pid is None else pid
    identity = _pid_identity(pid, proc_root)
    root = proc_root / str(pid)
    main = _safe_file(root / "stat", parse_proc_stat)
    if not isinstance(main, dict) or main.get("status") == "UNAVAILABLE" or "pid" not in main:
        raise ObservationError("PROBE_MAIN_STAT_UNAVAILABLE")
    main_status = _safe_file(root / "status", parse_status, {"Uid", "VmRSS"})
    if not isinstance(main_status, dict) or not isinstance(main_status.get("Uid"), str):
        raise ObservationError("PROBE_MAIN_UID_UNAVAILABLE")
    tids, total_tids, thread_cap_hit, thread_error = _thread_ids_with_bound(root / "task")
    if thread_error is not None or not tids:
        raise ObservationError("PROBE_THREAD_LIST_UNAVAILABLE:" + str(thread_error or "EMPTY"))
    thread = _thread_sample(root / "task", tids[0])
    thread_stat = thread.get("stat")
    if not isinstance(thread_stat, dict) or thread_stat.get("status") == "UNAVAILABLE" or "pid" not in thread_stat:
        raise ObservationError("PROBE_THREAD_STAT_UNAVAILABLE")
    return {
        "schema": SCHEMA + ".capability_probe.v1",
        "pid": pid,
        "identity_verified": True,
        "uid": identity[1],
        "create_time": identity[0],
        "starttime_ticks": identity[2],
        "required": {
            "main_stat": "AVAILABLE",
            "main_uid": "AVAILABLE",
            "thread_stat": "AVAILABLE",
            "thread_count_observed": len(tids),
            "thread_count_total": total_tids,
            "thread_cap": MAX_THREADS,
            "thread_cap_hit": thread_cap_hit,
        },
        "optional": {
            "main_status": main_status,
            "thread_status": thread.get("status"),
            "thread_schedstat": thread.get("schedstat"),
            "node": _node_sample(proc_root),
        },
        "clock": _clock_metadata(),
        "units": _units_metadata(),
        "science_arrays_retained": False,
    }


class ProcessObserver:
    """Observe one PID, with idempotent stop/join and bounded output."""

    def __init__(self, pid: int, output_dir: str | Path, *, period_s: float = DEFAULT_PERIOD_S,
                 numa_every_samples: int = 3, proc_root: str | Path = "/proc") -> None:
        if not isinstance(pid, int) or pid <= 0:
            raise ObservationError("PID_INVALID")
        if period_s <= 0 or numa_every_samples <= 0:
            raise ObservationError("OBSERVER_INTERVAL_INVALID")
        self.pid = pid
        self.output_dir = Path(output_dir).resolve()
        self.period_s = float(period_s)
        self.numa_every_samples = int(numa_every_samples)
        self.proc_root = Path(proc_root)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._started = False
        self._stopped = False
        self._errors: list[dict[str, str]] = []
        self._samples = 0
        self._bytes = 0
        self._status = "NOT_STARTED"
        self._start_wall_ns = 0
        self._end_wall_ns = 0
        self._start_perf_counter_ns = 0
        self._end_perf_counter_ns = 0
        self._collection_wall_ns = 0
        self._thread_cap_hits = 0
        self._thread_cpu_ns = 0
        self._identity: tuple[float, int | None, int] | None = None
        self._jsonl: Path | None = None

    def start(self) -> "ProcessObserver":
        with self._lock:
            if self._started:
                raise ObservationError("OBSERVER_ALREADY_STARTED")
            self.output_dir.mkdir(parents=True, exist_ok=False)
            self._identity = _pid_identity(self.pid, self.proc_root)
            self._jsonl = self.output_dir / "samples.jsonl"
            self._start_wall_ns = time.monotonic_ns()
            self._start_perf_counter_ns = time.perf_counter_ns()
            self._status = "RUNNING"
            self._started = True
            self._thread = threading.Thread(target=self._loop, name=f"s46g-observer-{self.pid}", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout_s: float = 30.0) -> dict[str, Any]:
        with self._lock:
            if not self._started:
                raise ObservationError("OBSERVER_NOT_STARTED")
            self._stop.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_s)
        with self._lock:
            if thread is not None and thread.is_alive():
                self._status = "FAILED"
                self._errors.append({"kind": "JOIN_TIMEOUT", "message": str(timeout_s)})
            self._stopped = True
            if self._end_wall_ns == 0:
                self._end_wall_ns = time.monotonic_ns()
            if self._end_perf_counter_ns == 0:
                self._end_perf_counter_ns = time.perf_counter_ns()
            receipt = self._receipt()
            receipt_path = self.output_dir / "OBSERVATION_RECEIPT.json"
            receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return receipt

    def _loop(self) -> None:
        cpu_start = time.thread_time_ns()
        try:
            while not self._stop.is_set():
                if self._samples >= MAX_SAMPLES:
                    self._errors.append({"kind": "SAMPLE_CAP", "message": str(MAX_SAMPLES)})
                    self._status = "FAILED"
                    break
                try:
                    collection_start = time.perf_counter_ns()
                    sample = self._sample(self._samples)
                    collection_end = time.perf_counter_ns()
                    sample["collection_start_perf_counter_ns"] = collection_start
                    sample["collection_end_perf_counter_ns"] = collection_end
                    sample["collection_wall_ns"] = max(0, collection_end - collection_start)
                    self._collection_wall_ns += sample["collection_wall_ns"]
                    encoded = (json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                    if self._bytes + len(encoded) > MAX_OUTPUT_BYTES:
                        self._errors.append({"kind": "OUTPUT_CAP", "message": str(MAX_OUTPUT_BYTES)})
                        self._status = "FAILED"
                        break
                    assert self._jsonl is not None
                    with self._jsonl.open("ab") as stream:
                        stream.write(encoded)
                    self._bytes += len(encoded)
                    self._samples += 1
                except ObservationError as exc:
                    if str(exc) == "PID_NOT_OBSERVABLE":
                        # The frozen supervisor owns child reaping. A normal
                        # child exit is an observation boundary, not a zero
                        # counter and not a sampler failure.
                        self._status = "PID_EXITED"
                    else:
                        self._errors.append({"kind": "SAMPLE_ERROR", "message": str(exc)})
                        self._status = "FAILED"
                    break
                except BaseException as exc:  # preserve sampler failure, never child failure
                    self._errors.append({"kind": "SAMPLE_EXCEPTION", "message": repr(exc)})
                    self._status = "FAILED"
                    break
                self._stop.wait(self.period_s)
        finally:
            self._thread_cpu_ns = time.thread_time_ns() - cpu_start
            self._end_wall_ns = time.monotonic_ns()
            self._end_perf_counter_ns = time.perf_counter_ns()
            if self._status == "RUNNING":
                self._status = "STOPPED" if self._stop.is_set() else "PID_EXITED"

    def _sample(self, index: int) -> dict[str, Any]:
        identity = _pid_identity(self.pid, self.proc_root)
        if self._identity is None or identity[1:] != self._identity[1:] or identity[0] != self._identity[0]:
            raise ObservationError("PID_IDENTITY_CHANGED")
        root = self.proc_root / str(self.pid)
        main = _safe_file(root / "stat", parse_proc_stat)
        status = _safe_file(root / "status", parse_status, {"Name", "VmRSS", "voluntary_ctxt_switches", "nonvoluntary_ctxt_switches", "Cpus_allowed_list", "Mems_allowed_list"})
        tids, total_tids, thread_cap_hit, thread_error = _thread_ids_with_bound(root / "task")
        if thread_error is not None:
            if not root.exists():
                raise ObservationError("PID_NOT_OBSERVABLE")
            raise ObservationError("THREAD_TASK_UNAVAILABLE:" + thread_error)
        if thread_cap_hit:
            self._thread_cap_hits += 1
        threads = [_thread_sample(root / "task", tid) for tid in tids]
        sample: dict[str, Any] = {"schema": SCHEMA + ".sample.v1", "index": index,
            "monotonic_ns": time.monotonic_ns(), "pid": self.pid, "main": main, "status": status,
            "thread_count": len(threads), "thread_count_total": total_tids,
            "thread_cap": MAX_THREADS, "thread_cap_hit": thread_cap_hit,
            "threads": threads, "node": _node_sample(self.proc_root),
            "clock": _clock_metadata(), "units": _units_metadata()}
        if index % self.numa_every_samples == 0:
            sample["numa_pages_by_node"] = _safe_file(root / "numa_maps", summarize_numa_maps)
        else:
            sample["numa_pages_by_node"] = {"status": "NOT_SAMPLED", "every_samples": self.numa_every_samples}
        return sample

    def _receipt(self) -> dict[str, Any]:
        return {"schema": SCHEMA + ".receipt.v1", "pid": self.pid, "uid": self._identity[1] if self._identity else None,
            "status": self._status, "samples": self._samples, "bytes": self._bytes,
            "max_samples": MAX_SAMPLES, "max_threads": MAX_THREADS, "max_output_bytes": MAX_OUTPUT_BYTES,
            "period_s": self.period_s, "numa_every_samples": self.numa_every_samples,
            "start_monotonic_ns": self._start_wall_ns, "end_monotonic_ns": self._end_wall_ns,
            "start_perf_counter_ns": self._start_perf_counter_ns, "end_perf_counter_ns": self._end_perf_counter_ns,
            "observer_wall_ns": max(0, self._end_wall_ns - self._start_wall_ns),
            "collection_wall_ns": self._collection_wall_ns,
            "thread_cap_hits": self._thread_cap_hits,
            "observer_thread_cpu_ns": self._thread_cpu_ns, "errors": list(self._errors),
            "science_arrays_retained": False, "short_lived_children_observed": False,
            "last_periodic_sample_only": True, "end_state_snapshot": False,
            "clock": _clock_metadata(), "units": _units_metadata(),
            "proc_root": str(self.proc_root)}


__all__ = ["DEFAULT_PERIOD_S", "MAX_OUTPUT_BYTES", "MAX_SAMPLES", "MAX_THREADS", "ObservationError", "ProcessObserver", "parse_cpu_stat", "parse_proc_stat", "parse_schedstat", "parse_status", "probe_current_pid", "summarize_numa_maps"]
