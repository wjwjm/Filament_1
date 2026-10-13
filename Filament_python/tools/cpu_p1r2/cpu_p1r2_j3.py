#!/usr/bin/env python3
"""Bounded CPU J3 baseline closed-loop diagnostic; never production-qualified."""
from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

os.environ["UPPE_USE_GPU"] = "0"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from tools.cpu_p1 import cpu_p1_fft as common
from tools.cpu_p1r import cpu_p1r_optical as p1r
from tools.cpu_p1r2 import cpu_p1r2_j1 as j1
from tools.hpc_ops.provenance_v2 import raw_sha256_file

SCHEMA = "filament.cpu_p1r2.j3_closed_loop_diagnostic.v1"
PHASE = "J3"
CLASSIFICATION = "P1R2_J3_BASELINE_DIAGNOSTIC_ONLY"
QUALIFICATION = "NOT_PRODUCTION_QUALIFIED"
FIELDS = ("delta_n", "vx", "vy")
HYDRO_STEPS = 1000
EXPECTED_F_REP = 1000.0
EXPECTED_DT_HYDRO = 1e-6
EXPECTED_N0 = 1.00027
EXPECTED_DX = EXPECTED_DY = 1e-5
OPTICAL_WINDOW_BASELINE_MEASURED_SECONDS = 514.142852
OPTICAL_WINDOW_SCHEDULING_MARGIN = 0.10
OPTICAL_WINDOW_ESTIMATE_SECONDS = max(512.0, OPTICAL_WINDOW_BASELINE_MEASURED_SECONDS) * (
    1.0 + OPTICAL_WINDOW_SCHEDULING_MARGIN)
HYDRO_STAGE_ALLOWANCE_SECONDS = 120.0  # scheduling allowance, not a measured prediction
HYDRO_STAGE_TIMEOUT_SECONDS = 180.0
FIRST_CHILD_TIMEOUT_SECONDS = 2 * OPTICAL_WINDOW_ESTIMATE_SECONDS + HYDRO_STAGE_TIMEOUT_SECONDS
RELOAD_CHILD_TIMEOUT_SECONDS = 600.0
J3_WORK_LIMIT_SECONDS = 2700.0
MEMORY_BYTES = 320 * 1024**3
RSS_WARN_FRACTION = 0.80
RSS_ABORT_FRACTION = 0.90


class J3Error(RuntimeError):
    pass


def _json_safe(value):
    return j1._json_safe(value)


def _json_bytes(value) -> bytes:
    return (json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True,
        indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_atomic_new(path: Path, payload: bytes) -> None:
    """Atomically publish a same-directory file without replacing an existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    except BaseException:
        try:
            temp.unlink()
        except OSError:
            pass
        raise
    temp.unlink()


def _write_json_new(path: Path, value) -> None:
    _write_atomic_new(path, _json_bytes(value))


def _npz_bytes(fields: dict[str, object]) -> bytes:
    import numpy as np
    stream = io.BytesIO()
    np.savez(stream, **fields)
    return stream.getvalue()


def _write_npz_new(path: Path, fields: dict[str, object]) -> str:
    _write_atomic_new(path, _npz_bytes(fields))
    return raw_sha256_file(path)


def _field_hashes(fields: dict[str, object]) -> dict[str, str]:
    return {name: j1._canonical_hash(fields[name]) for name in FIELDS}


def _validate_fields(fields: dict, *, shape=None, label="J3 fields") -> dict:
    import numpy as np
    if set(fields) != set(FIELDS):
        raise J3Error(f"{label} must contain exactly delta_n/vx/vy")
    result = {}
    expected_shape = None if shape is None else tuple(shape)
    for name in FIELDS:
        value = np.asarray(fields[name])
        if value.dtype != np.dtype("float64") or value.ndim not in (2, 3) or not value.flags.c_contiguous:
            raise J3Error(f"{label}.{name} must be C-contiguous float64 map/stack")
        if not np.isfinite(value).all():
            raise J3Error(f"{label}.{name} contains NaN/Inf")
        if expected_shape is None:
            expected_shape = value.shape
        if value.shape != expected_shape:
            raise J3Error(f"{label} field shapes differ")
        result[name] = value
    return result


def _load_npz(path: Path, *, expected_raw_sha256: str, expected_field_sha256: dict,
              shape=None, label="J3 checkpoint") -> dict:
    import numpy as np
    raw = raw_sha256_file(path)
    if raw != expected_raw_sha256:
        raise J3Error(f"{label} raw-byte SHA256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        fields = _validate_fields({name: archive[name].copy(order="C") for name in archive.files},
                                  shape=shape, label=label)
    hashes = _field_hashes(fields)
    if hashes != expected_field_sha256:
        raise J3Error(f"{label} canonical field hash mismatch")
    return fields


def _load_frozen_hydro_parameters(input_root: Path, context: dict) -> dict:
    from KHz_filament import hr4
    from KHz_filament.grids import make_axes
    from KHz_filament.hr4e5_formal_entry import (
        interpulse_worker_parameters, load_admission_identity,
    )
    identity_path = input_root / "N5_FORMAL_ADMISSION_IDENTITY.json"
    identity = load_admission_identity(identity_path, formal=True)
    f_rep = float(identity["f_rep"])
    dt_hydro = float(identity["dt_hydro"])
    if f_rep != EXPECTED_F_REP or dt_hydro != EXPECTED_DT_HYDRO:
        raise J3Error("sealed admission identity does not declare the approved J3 f_rep/dt_hydro")
    schedule = interpulse_worker_parameters(f_rep=f_rep, dt_hydro=dt_hydro)
    steps = int(schedule["n_hydro_steps"])
    if steps != HYDRO_STEPS:
        raise J3Error("sealed interpulse schedule does not resolve to exactly 1000 HR4 steps")
    grid, beam = context["grid"], context["beam"]
    axes = make_axes(grid.Nx, grid.Ny, grid.Nt, grid.Lx, grid.Ly, grid.Twin)
    dx, dy, n0 = float(axes.dx), float(axes.dy), float(beam.n0)
    if (not math.isclose(dx, EXPECTED_DX, rel_tol=0.0, abs_tol=1e-15)
            or not math.isclose(dy, EXPECTED_DY, rel_tol=0.0, abs_tol=1e-15)):
        raise J3Error("sealed optical grid spacing does not match frozen HR4 dx/dy")
    if n0 != EXPECTED_N0:
        raise J3Error("sealed beam configuration n0 does not match the frozen HR4 value")
    if (grid.Ny, grid.Nx) != (351, 301) or context["schedule"].n_intervals != 8:
        raise J3Error("J3 requires the sealed 351x301 physical maps and complete eight-interval window")
    return {
        "identity": identity,
        "identity_raw_sha256": raw_sha256_file(identity_path),
        "f_rep": f_rep, "dt_hydro": dt_hydro, "n_steps": steps,
        "dx": dx, "dy": dy, "n0": n0,
        "chi": hr4.HR4_CHI, "nu": hr4.HR4_NU,
        "gravity_x": hr4.HR4_GRAVITY_X, "gravity_y": hr4.HR4_GRAVITY_Y,
        "cfl_limit": hr4.HR4_CFL_LIMIT,
        "x_min": hr4.HR4_X_MIN, "y_min": hr4.HR4_Y_MIN,
        "x_max": hr4.HR4_X_MAX, "y_max": hr4.HR4_Y_MAX,
        "observable_coordinate_frames": {
            "optical_centered_y_extent_m": [-float(grid.Ly) / 2.0, float(grid.Ly) / 2.0],
            "hr4_legacy_observable_origin_m": {
                "x_min": hr4.HR4_X_MIN, "y_min": hr4.HR4_Y_MIN},
            "centroid_frames_directly_comparable": False,
            "note": "HR4 origin is the inherited thermal_channel_observables default; no shift, regrid, or coordinate remapping is applied"},
        "boundaries": {"delta_n": hr4.HR4_ALLOWED_DELTA_N_BOUNDARY,
            "velocity": hr4.HR4_ALLOWED_VELOCITY_BOUNDARY,
            "advection": hr4.HR4_ALLOWED_ADVECTION,
            "diffusion": hr4.HR4_ALLOWED_DIFFUSION,
            "integrator": hr4.HR4_ALLOWED_INTEGRATOR,
            "grid_layout": hr4.HR4_ALLOWED_GRID_LAYOUT},
        "identity_f_rep_dt_hydro_sha256": identity["identity_sha256"],
    }


def _invoke_hydro(advance, fields: dict, params: dict) -> dict:
    result = advance(fields["delta_n"], fields["vx"], fields["vy"],
        dx=float(params["dx"]), dy=float(params["dy"]),
        dt_hydro=float(params["dt_hydro"]), chi=float(params["chi"]),
        nu=float(params["nu"]), n0=float(params["n0"]),
        gravity_x=float(params["gravity_x"]), gravity_y=float(params["gravity_y"]),
        cfl_limit=float(params["cfl_limit"]), n_steps=int(params["n_steps"]),
        require_stable=True, x_min=float(params["x_min"]), y_min=float(params["y_min"]))
    if int(result.get("steps", -1)) != int(params["n_steps"]):
        raise J3Error("HR4 returned a step count different from the sealed interpulse schedule")
    if result.get("performance", {}).get("backend") != "numpy":
        raise J3Error("J3 HR4 task did not use the actual NumPy CPU backend")
    return result


def _verified_worker_cpus(count: int) -> tuple[dict, list[int]]:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise J3Error("Linux CPU affinity is required for J3 real HR4 workers")
    allowed = sorted(os.sched_getaffinity(0))
    topology = common.cpu_topology(allowed)
    if not common.physical_topology_verified(topology):
        raise J3Error("sysfs physical-core topology is unavailable")
    cpus = common.select_physical_cpus(count, allowed, topology)
    if len(set(cpus)) != count:
        raise J3Error("physical worker affinity selection contains duplicates")
    return topology, cpus


def _copy_post_screen(full: dict, ordinal: int) -> dict:
    import numpy as np
    if isinstance(ordinal, bool) or not 0 <= int(ordinal) < 8:
        raise J3Error("POST1 screen ordinal must be in [0, 7]")
    fields = {name: np.array(full[name][int(ordinal)], dtype=np.float64,
                             order="C", copy=True) for name in FIELDS}
    _validate_fields(fields, shape=tuple(full["delta_n"].shape[1:]),
                     label="independent POST1 screen copy")
    if any(np.shares_memory(fields[name], full[name]) for name in FIELDS):
        raise J3Error("worker screen copy unexpectedly aliases the POST stack")
    return fields


def _worker_task(task_path: Path) -> int:
    """Run one screen after binding, loading, and validating the immutable POST1."""
    worker_cpu_started = time.process_time()
    try:
        task = json.loads(task_path.read_text(encoding="utf-8"))
        if int(task.get("expected_pgid", -1)) != os.getpgrp():
            raise J3Error("HR4 worker did not inherit its chain child's process group")
        cpu = int(task["cpu"])
        allowed = sorted(os.sched_getaffinity(0))
        topology = common.cpu_topology(allowed)
        if not common.physical_topology_verified(topology) or cpu not in topology["physical_core_representatives"]:
            raise J3Error("worker CPU is not a verified physical-core representative")
        os.sched_setaffinity(0, {cpu})
        if sorted(os.sched_getaffinity(0)) != [cpu]:
            raise J3Error("worker affinity did not restrict to its assigned physical CPU")
        common.add_vendor(str(task["vendor_dir"]))
        import numpy as np
        import threadpoolctl
        from KHz_filament import hr4
        limit = threadpoolctl.threadpool_limits(limits=1)
        pools = threadpoolctl.threadpool_info()
        if any(int(row.get("num_threads", 1)) > 1 for row in pools):
            raise J3Error("worker native thread pool exceeds one thread")
        if hr4.debug_backend().get("backend") != "numpy":
            raise J3Error("worker HR4 backend is not NumPy")
        load_started, load_cpu_started = time.perf_counter(), time.process_time()
        full = _load_npz(Path(task["post_path"]),
            expected_raw_sha256=task["post_raw_sha256"],
            expected_field_sha256=task["post_field_sha256"],
            shape=tuple(task["post_shape"]), label="J3 POST1 worker input")
        ordinal = int(task["ordinal"])
        fields = _copy_post_screen(full, ordinal)
        input_hashes = {name: j1._canonical_hash(fields[name]) for name in FIELDS}
        load_seconds = time.perf_counter() - load_started
        load_cpu_seconds = time.process_time() - load_cpu_started
        kernel_wall, kernel_cpu = time.perf_counter(), time.process_time()
        result = _invoke_hydro(hr4.advance_hr4_single_screen, fields, task["hydro_parameters"])
        kernel_wall = time.perf_counter() - kernel_wall
        kernel_cpu = time.process_time() - kernel_cpu
        validated_output = _validate_fields({name: result[name] for name in FIELDS},
            shape=fields["delta_n"].shape, label="J3 NEXT1 worker output")
        output_fields = {name: np.array(validated_output[name], dtype=np.float64,
            order="C", copy=True) for name in FIELDS}
        output_hashes = _field_hashes(output_fields)
        write_started, write_cpu_started = time.perf_counter(), time.process_time()
        output_path = Path(task["output_npz"])
        output_raw = _write_npz_new(output_path, output_fields)
        write_seconds = time.perf_counter() - write_started
        write_cpu_seconds = time.process_time() - write_cpu_started
        receipt = {"schema": SCHEMA, "phase": PHASE, "classification": CLASSIFICATION,
            "production_qualification": QUALIFICATION, "status": "DONE",
            "execution_scope": task["execution_scope"], "ordinal": ordinal,
            "assigned_cpu": cpu, "worker_affinity": sorted(os.sched_getaffinity(0)),
            "physical_topology": topology, "threadpool_info": pools,
            "hr4_backend": hr4.debug_backend(), "n_steps": HYDRO_STEPS,
            "input_POST_raw_sha256": task["post_raw_sha256"],
            "input_POST_screen_field_sha256": input_hashes,
            "output_NEXT_raw_sha256": output_raw,
            "output_NEXT_screen_field_sha256": output_hashes,
            "load_seconds": load_seconds, "kernel_wall_seconds": kernel_wall,
            "load_process_cpu_seconds": load_cpu_seconds,
            "kernel_process_cpu_seconds": kernel_cpu, "write_hash_seconds": write_seconds,
            "write_hash_process_cpu_seconds": write_cpu_seconds,
            "worker_process_cpu_seconds_before_receipt_write":
                time.process_time() - worker_cpu_started,
            "service_wall_seconds": load_seconds + kernel_wall + write_seconds,
            "screen_observables": j1._json_safe(result.get("observables", {})),
            "stability": j1._json_safe(result.get("stability", {})),
            "performance": j1._json_safe(result.get("performance", {}))}
        _write_json_new(Path(task["output_receipt"]), receipt)
        del limit
        return 0
    except BaseException as exc:
        failure = {"schema": SCHEMA, "phase": PHASE, "status": "FAIL",
            "ordinal": task.get("ordinal") if "task" in locals() else None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()}
        try:
            _write_json_new(Path(task["failure_receipt"]), failure)
        except Exception:
            pass
        traceback.print_exc()
        return 1


def _prepare_real_context(args) -> tuple[dict, dict]:
    if args.phase != PHASE or any(value is None for value in
            (args.bench_root, args.input_root, args.vendor_dir)):
        raise J3Error("real J3 requires --phase J3 and bench/input/vendor roots")
    if sys.version_info < (3, 12) or not hasattr(sys, "monitoring"):
        raise J3Error("frozen optical observer requires Python 3.12+ sys.monitoring")
    allocation = p1r._check_allocation()
    common.add_vendor(str(args.vendor_dir))
    import threadpoolctl
    limit = threadpoolctl.threadpool_limits(limits=1)
    if getattr(threadpoolctl, "__version__", None) != "3.6.0":
        raise J3Error("the frozen threadpoolctl 3.6.0 vendor is required")
    if sys.executable != p1r.FIXED_PYTHON:
        raise J3Error(f"fixed Python required: {p1r.FIXED_PYTHON}; got {sys.executable}")
    if any(int(row.get("num_threads", 1)) > 1 for row in threadpoolctl.threadpool_info()):
        raise J3Error("J3 coordinator native thread pool exceeds one thread")
    j1._assert_cpu_backend()
    p1r._source_binding(args.bench_root)
    context = p1r._load_context(args.input_root, args.bench_root, args.vendor_dir)
    if context["source_hash"] != p1r.SOURCE_ARRAY_SHA256:
        raise J3Error("loaded source differs from the sealed canonical source hash")
    if context["assets"]["pre0_canonical_sha256"] != p1r.PRE0_ARRAY_SHA256:
        raise J3Error("loaded PRE0 differs from its sealed canonical hash")
    hydro = _load_frozen_hydro_parameters(args.input_root, context)
    del limit
    return {"context": context, "hydro": hydro}, allocation


def _window(context: dict, pre_fields: dict, *, output_root: Path,
            stage_name: str, initial_pre0: bool, source, source_hash: str,
            pre_hash: str | None, execution_scope: str) -> dict:
    """Invoke the frozen complete-eight J1 propagator and verify all persisted outputs."""
    import numpy as np
    output_root.parent.mkdir(parents=True, exist_ok=True)
    receipt = j1._run_window(source=source, pre_fields=pre_fields,
        grid=context["grid"], beam=context["beam"], prop=context["prop"],
        ion=context["ion"], heat=context["heat"], raman=context["raman"],
        schedule=context["schedule"], output_dir=output_root,
        source_label="sealed_PRE0_first_eight" if initial_pre0 else "J3_NEXT1_memory",
        initial_pre0=initial_pre0, profile=False, source_hash=source_hash,
        pre0_hash=pre_hash, stage_name=stage_name, execution_scope=execution_scope)
    is_smoke = execution_scope == "SOFTWARE_SMOKE_ONLY"
    _write_json_new(output_root / "j3_provenance_receipt.json", {
        "schema": SCHEMA, "phase": PHASE, "execution_scope": execution_scope,
        "source_raw_sha256": None if is_smoke else p1r.SOURCE_RAW_SHA256,
        "source_canonical_sha256": source_hash,
        "pre0_raw_sha256": None if is_smoke or not initial_pre0 else p1r.PRE0_RAW_SHA256,
        "pre0_canonical_sha256": pre_hash if initial_pre0 else None,
        "transfer_mapping_sha256": None if is_smoke else context["assets"]["mapping_sha256"],
        "input_PRE_fields_canonical_sha256": receipt["input_PRE_fields_canonical_sha256"],
        "output_raw_sha256": {"final_E": receipt["final_E_raw_sha256"],
            "final_POST": receipt["final_POST_raw_sha256"],
            "diagnostics": receipt["diagnostics_raw_sha256"]}})
    if receipt.get("status") != "COMPLETE_DIAGNOSTIC" or receipt.get("interval_count") != 8:
        raise J3Error(f"{stage_name} did not complete exactly eight optical intervals")
    final_path = output_root / "final_E.npy"
    final = np.load(final_path, mmap_mode="r", allow_pickle=False)
    final_hash = p1r.verify_array_contract(final, shape=tuple(source.shape),
        dtype="complex128", expected_hash=receipt["final_E_canonical_sha256"],
        label=f"{stage_name} final E")
    final_raw = raw_sha256_file(final_path)
    if final_raw != receipt["final_E_raw_sha256"]:
        raise J3Error(f"{stage_name} final E raw hash mismatch")
    post_path = output_root / "final_POST.npz"
    post_hashes = {}
    with np.load(post_path, allow_pickle=False) as archive:
        if set(archive.files) != set(FIELDS):
            raise J3Error(f"{stage_name} POST fields are incomplete")
        for name in FIELDS:
            post_hashes[name] = p1r.verify_array_contract(archive[name],
                shape=(8, context["grid"].Ny, context["grid"].Nx), dtype="float64",
                expected_hash=receipt["final_POST_canonical_sha256"][name],
                label=f"{stage_name} POST {name}")
    post_raw = raw_sha256_file(post_path)
    if post_raw != receipt["final_POST_raw_sha256"]:
        raise J3Error(f"{stage_name} POST raw hash mismatch")
    diagnostics_path = output_root / "diagnostics.npz"
    diagnostics_hashes = {}
    with np.load(diagnostics_path, allow_pickle=False) as archive:
        for name in archive.files:
            diagnostics_hashes[name] = j1._canonical_hash(archive[name])
    if diagnostics_hashes != receipt["diagnostics_canonical_sha256"]:
        raise J3Error(f"{stage_name} diagnostic canonical hashes mismatch")
    diagnostics_raw = raw_sha256_file(diagnostics_path)
    if diagnostics_raw != receipt["diagnostics_raw_sha256"]:
        raise J3Error(f"{stage_name} diagnostic raw hash mismatch")
    if len(receipt.get("interval_rows", [])) != 8:
        raise J3Error(f"{stage_name} scalar observer missed a complete interval")
    return {"status": "VALIDATED", "receipt": receipt,
        "final_E_path": str(final_path), "final_E_raw_sha256": final_raw,
        "final_E_canonical_sha256": final_hash,
        "final_POST_path": str(post_path), "final_POST_raw_sha256": post_raw,
        "final_POST_canonical_sha256": post_hashes,
        "diagnostics_path": str(diagnostics_path),
        "diagnostics_raw_sha256": diagnostics_raw,
        "diagnostics_canonical_sha256": diagnostics_hashes}


def _allocation_start_epoch() -> float:
    values = [os.environ.get("P1R2_ALLOCATION_START_EPOCH"),
              os.environ.get("P1R_ALLOCATION_START_EPOCH")]
    values = [float(value) for value in values if value not in (None, "")]
    if not values or (len(values) == 2 and abs(values[0] - values[1]) > 1e-6):
        raise J3Error("matching P1R2/P1R allocation start epoch variables are required")
    return values[0]


def _child_command(args, child_mode: str, child_root: Path, **extra):
    command = [sys.executable, "-m", "tools.cpu_p1r2.cpu_p1r2_j3",
        "--phase", PHASE, "--bench-root", str(args.bench_root),
        "--input-root", str(args.input_root), "--vendor-dir", str(args.vendor_dir),
        "--output-root", str(child_root), "--child-mode", child_mode]
    for key, value in extra.items():
        command.extend((f"--{key.replace('_', '-')}", str(value)))
    return command


def _supervise_child(args, stage: str, child_mode: str, *, timeout_seconds: float,
                     extra=None) -> dict:
    stage_dir = args.output_root / "stages" / stage
    stage_dir.mkdir(parents=True, exist_ok=False)
    child_root = stage_dir / "child"
    env = j1._child_environment(args.bench_root)
    command = _child_command(args, child_mode, child_root, **(extra or {}))
    started = time.perf_counter()
    supervisor = j1._spawn_and_monitor(command, stage_dir=stage_dir,
        cwd=args.bench_root / "Filament_python", env=env,
        timeout_seconds=timeout_seconds, memory_bytes=MEMORY_BYTES)
    result = {"stage": stage, "status": "DONE" if supervisor.get("exit_code") == 0
        and supervisor.get("stop") is None else "FAIL_CHILD", "supervisor": supervisor,
        "child_root": str(child_root)}
    if supervisor.get("timed_out"):
        result["status"] = "LIMIT_REACHED"
    elif supervisor.get("exit_code") is None or (supervisor.get("stop") or {}).get("status") == "STOP_UNRESOLVED":
        result["status"] = "STOP_UNRESOLVED"
    elif supervisor.get("stop"):
        result["status"] = "STOPPED_RESOURCE_GATE"
    if result["status"] == "DONE":
        try:
            result["child_receipt"] = json.loads((child_root / "j3_child_receipt.json").read_text(encoding="utf-8"))
        except BaseException as exc:
            result.update({"status": "FAIL_CHILD_RECEIPT",
                "receipt_error": f"{type(exc).__name__}: {exc}"})
    result["service_wall_seconds_including_receipt_read"] = time.perf_counter() - started
    _write_json_new(stage_dir / "stage_receipt.json", result)
    return result


def _terminate_worker_batch(states: list[dict], reason: str) -> None:
    active = [state["proc"] for state in states if state["proc"].poll() is None]
    for proc in active:
        try:
            proc.terminate()
        except OSError:
            pass
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and any(proc.poll() is None for proc in active):
        time.sleep(0.05)
    still_active = [proc for proc in active if proc.poll() is None]
    for proc in still_active:
        try:
            proc.kill()
        except OSError:
            pass
    kill_deadline = time.monotonic() + 5.0
    while time.monotonic() < kill_deadline and any(proc.poll() is None for proc in still_active):
        time.sleep(0.05)
    for state in states:
        proc = state["proc"]
        if proc.poll() is None:
            state["stop"] = {"reason": reason, "status": "STOP_UNRESOLVED"}
        elif proc in active:
            state["stop"] = {"reason": reason, "status": "TERMINATED_WITH_BATCH"}
        try:
            proc.wait(timeout=0.1)
        except (subprocess.TimeoutExpired, OSError):
            if proc.poll() is None:
                state["stop"] = {"reason": reason, "status": "STOP_UNRESOLVED"}


def _write_worker_batch_failure(states: list[dict], stage: str, exc: BaseException) -> None:
    failure = {"schema": SCHEMA, "phase": PHASE, "status": "FAIL",
        "stage": stage, "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc()}
    for state in states:
        try:
            _write_json_new(Path(state["spec"]["task_dir"]) / "batch_failure.json", failure)
        except Exception:
            pass


def _supervise_worker_batch(specs: list[dict], *, timeout_seconds: float,
                            memory_bytes: int) -> dict:
    """Launch all screen workers before waiting; monitor and stop the batch as a unit."""
    if len(specs) != 8:
        raise J3Error("J3 hydro stage must launch exactly eight concurrent worker processes")
    started = time.monotonic()
    expected_pgid = os.getpgrp()
    states = []
    try:
        for spec in specs:
            task_dir = Path(spec["task_dir"])
            stdout_path, stderr_path = task_dir / "stdout.log", task_dir / "stderr.log"
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                proc = subprocess.Popen(list(spec["command"]), cwd=str(spec["cwd"]),
                    env=spec["env"], stdout=stdout, stderr=stderr)
            try:
                worker_pgid = os.getpgid(proc.pid)
            except OSError:
                worker_pgid = None
            state = {"spec": spec, "proc": proc, "pgid": worker_pgid,
                "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
                "started": time.monotonic(), "peak_rss": 0, "peak_cpu": 0.0,
                "pss_samples": [], "next_pss": 0.0, "stop": None}
            states.append(state)
            if worker_pgid != expected_pgid:
                raise J3Error("HR4 worker process escaped the chain process group")
    except BaseException as exc:
        _terminate_worker_batch(states, "BATCH_LAUNCH_FAILURE")
        _write_worker_batch_failure(states, "launch", exc)
        raise

    stop_reason = None
    peak_aggregate_rss = 0
    peak_sampled_tree_cpu = 0.0
    rss_80_latched = False
    pss_samples = []
    next_pss = 0.0
    try:
        while any(state["proc"].poll() is None for state in states):
            if time.monotonic() - started >= float(timeout_seconds):
                stop_reason = "HYDRO_STAGE_TIMEOUT"
                break
            failed_state = None
            now = time.monotonic()
            for state in states:
                code = state["proc"].poll()
                if code not in (None, 0) and failed_state is None:
                    failed_state = state
            snapshot = common.tree_snapshot(os.getpid())
            if snapshot is None:
                stop_reason = "PROCESS_TREE_RSS_UNAVAILABLE"
                break
            aggregate_rss = int(snapshot["rss"])
            peak_aggregate_rss = max(peak_aggregate_rss, aggregate_rss)
            peak_sampled_tree_cpu = max(peak_sampled_tree_cpu, float(snapshot.get("cpu", 0.0)))
            if now - started >= next_pss:
                pss_samples.append({"elapsed_seconds": now - started,
                                    **j1._pss_tree_snapshot(os.getpid())})
                next_pss = now - started + j1.PSS_SAMPLE_SECONDS
            policy = j1._rss_action(aggregate_rss, memory_bytes)
            rss_80_latched = rss_80_latched or bool(policy["warn_80"])
            if policy["abort_90"]:
                stop_reason = "RSS_90_PERCENT"
                break
            if failed_state is not None:
                stop_reason = f"WORKER_EXIT_{failed_state['proc'].returncode}"
                break
            time.sleep(0.25)
    except BaseException as exc:
        _terminate_worker_batch(states, "BATCH_MONITOR_EXCEPTION")
        _write_worker_batch_failure(states, "monitor", exc)
        raise

    if stop_reason is not None:
        _terminate_worker_batch(states, stop_reason)
    # Reap every direct child with a bound; never leave a worker behind.
    reap_deadline = time.monotonic() + 5.0
    for state in states:
        proc = state["proc"]
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
        try:
            state["exit_code"] = proc.wait(timeout=max(0.1, reap_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            state["exit_code"] = None
            state["stop"] = {"reason": stop_reason or "REAP_TIMEOUT",
                              "status": "STOP_UNRESOLVED"}
    rows = []
    for state in states:
        row = {"exit_code": state.get("exit_code"), "stop": state["stop"],
            "elapsed_wall_seconds": time.monotonic() - state["started"],
            "process_group_id": state["pgid"],
            "inherited_chain_process_group": state["pgid"] == expected_pgid,
            "stdout_path": state["stdout_path"], "stderr_path": state["stderr_path"]}
        _write_json_new(Path(state["spec"]["task_dir"]) / "supervisor_receipt.json", row)
        rows.append(row)
    return {"status": "DONE" if all(row["exit_code"] == 0 and row["stop"] is None
                                      for row in rows) else "INCOMPLETE",
        "worker_rows": rows, "worker_processes_launched": len(states),
        "worker_processes_completed": sum(row["exit_code"] == 0 for row in rows),
        "elapsed_wall_seconds": time.monotonic() - started,
        "aggregate_peak_process_tree_rss_bytes": peak_aggregate_rss or None,
        "peak_sampled_process_tree_cpu_seconds_not_cumulative": peak_sampled_tree_cpu,
        "process_tree_pss_samples": pss_samples,
        "process_tree_sampling_seconds": 0.25,
        "rss_80_percent_latched": rss_80_latched or bool(
            j1._rss_action(peak_aggregate_rss or None, memory_bytes)["warn_80"]),
        "rss_80_fraction": RSS_WARN_FRACTION, "rss_90_fraction": RSS_ABORT_FRACTION,
        "process_group_id": expected_pgid, "all_workers_inherited_chain_process_group": all(
            row["inherited_chain_process_group"] for row in rows)}


def _process_hydro_workers(args, *, child_root: Path, post_path: Path,
                           post_raw_sha256: str, post_hashes: dict,
                           hydro_parameters: dict, execution_scope: str) -> dict:
    import numpy as np
    topology, cpus = _verified_worker_cpus(8)
    if len(set(cpus)) != 8:
        raise J3Error("eight concurrent HR4 workers require eight distinct physical CPUs")
    task_root = child_root / "hydro" / "tasks"
    task_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    completed, rows, specs = [], [], []
    env = j1._child_environment(args.bench_root)
    for ordinal, cpu in enumerate(cpus):
        task_dir = task_root / f"screen-{ordinal:02d}"
        task_dir.mkdir(exist_ok=False)
        task = {"ordinal": ordinal, "cpu": cpu, "vendor_dir": str(args.vendor_dir),
            "expected_pgid": os.getpgrp(),
            "post_path": str(post_path), "post_raw_sha256": post_raw_sha256,
            "post_field_sha256": post_hashes, "post_shape": [8, 351, 301],
            "hydro_parameters": hydro_parameters, "execution_scope": execution_scope,
            "output_npz": str(task_dir / "next_screen.npz"),
            "output_receipt": str(task_dir / "task_receipt.json"),
            "failure_receipt": str(task_dir / "failure_receipt.json")}
        task_path = task_dir / "task.json"
        _write_json_new(task_path, task)
        specs.append({"ordinal": ordinal, "cpu": cpu, "task_dir": task_dir,
            "cwd": args.bench_root / "Filament_python", "env": env,
            "command": [sys.executable, "-m", "tools.cpu_p1r2.cpu_p1r2_j3",
                "--phase", PHASE, "--worker-task", str(task_path)]})
    batch = _supervise_worker_batch(specs, timeout_seconds=HYDRO_STAGE_TIMEOUT_SECONDS,
                                    memory_bytes=MEMORY_BYTES)
    for spec, supervisor in zip(specs, batch["worker_rows"], strict=True):
        ordinal, cpu, task_dir = spec["ordinal"], spec["cpu"], Path(spec["task_dir"])
        row = {"ordinal": ordinal, "assigned_cpu": cpu, "supervisor": supervisor,
               "status": "DONE" if supervisor.get("exit_code") == 0
                   and not supervisor.get("stop") else "FAIL"}
        if row["status"] == "DONE":
            receipt = json.loads((task_dir / "task_receipt.json").read_text(encoding="utf-8"))
            if receipt.get("ordinal") != ordinal or receipt.get("n_steps") != HYDRO_STEPS:
                raise J3Error("HR4 worker receipt does not prove its assigned complete 1000-step task")
            output_path = task_dir / "next_screen.npz"
            if raw_sha256_file(output_path) != receipt["output_NEXT_raw_sha256"]:
                raise J3Error("HR4 worker output raw hash mismatch")
            with np.load(output_path, allow_pickle=False) as archive:
                screen = _validate_fields({name: archive[name].copy(order="C") for name in archive.files},
                    shape=(351, 301), label=f"NEXT screen {ordinal}")
            if _field_hashes(screen) != receipt["output_NEXT_screen_field_sha256"]:
                raise J3Error("HR4 worker output canonical hashes mismatch")
            row.update({"receipt": receipt, "output_path": str(output_path),
                        "output_hashes": _field_hashes(screen)})
            completed.append((ordinal, screen, receipt))
        rows.append(row)
    if len(completed) != 8:
        partial = {"schema": SCHEMA, "phase": PHASE, "status": "INCOMPLETE",
            "completed_screen_count": len(completed), "worker_rows": rows,
            "timeout_seconds": HYDRO_STAGE_TIMEOUT_SECONDS,
            "service_wall_seconds": time.perf_counter() - started,
            "physical_topology": topology, "worker_batch_supervisor": batch}
        _write_json_new(child_root / "hydro_partial_receipt.json", partial)
        raise J3Error(f"J3 HR4 transition incomplete: {len(completed)}/8 screens")
    outputs = {name: np.stack([screen[name] for _, screen, _ in completed], axis=0)
               for name in FIELDS}
    outputs = _validate_fields({key: np.ascontiguousarray(value, dtype=np.float64)
                                for key, value in outputs.items()}, shape=(8, 351, 301),
                               label="J3 NEXT1")
    next_path = child_root / "NEXT1_checkpoint.npz"
    next_raw = _write_npz_new(next_path, outputs)
    next_hashes = _field_hashes(outputs)
    receipt = {"schema": SCHEMA, "phase": PHASE, "classification": CLASSIFICATION,
        "production_qualification": QUALIFICATION, "status": "COMPLETE_DIAGNOSTIC",
        "execution_scope": execution_scope, "source_POST_raw_sha256": post_raw_sha256,
        "source_POST_field_sha256": post_hashes, "NEXT1_raw_sha256": next_raw,
        "NEXT1_field_sha256": next_hashes, "screen_count": 8,
        "worker_processes_launched": batch["worker_processes_launched"],
        "worker_processes_completed": len(completed),
        "worker_execution_model": "EIGHT_CONCURRENT_PROCESSES_ONE_PHYSICAL_CORE_EACH",
        "worker_assignments": [{"ordinal": spec["ordinal"], "cpu": spec["cpu"]}
                                for spec in specs],
        "physical_topology": topology, "worker_cpus": cpus, "worker_rows": rows,
        "rss_80_percent_latched": batch["rss_80_percent_latched"],
        "aggregate_peak_process_tree_rss_bytes": batch["aggregate_peak_process_tree_rss_bytes"],
        "worker_batch_supervisor": batch,
        "service_wall_seconds": time.perf_counter() - started,
        "worker_cpu_accounting": {
            "sum_worker_process_cpu_seconds": sum(float(item[2].get(
                "worker_process_cpu_seconds_before_receipt_write", 0.0))
                for item in completed),
            "sum_worker_kernel_cpu_seconds": sum(float(item[2].get(
                "kernel_process_cpu_seconds", 0.0)) for item in completed),
            "sum_worker_load_cpu_seconds": sum(float(item[2].get(
                "load_process_cpu_seconds", 0.0)) for item in completed),
            "sum_worker_write_cpu_seconds": sum(float(item[2].get(
                "write_hash_process_cpu_seconds", 0.0)) for item in completed),
            "subphase_cpu_fields_are_included_in_whole_worker_cpu": True,
            "aggregate_rule": "Use whole-worker CPU as aggregate; load/kernel/write are breakdowns and must not be added again"},
        "hydro_timing_classification": "MEASURED_8_SCREEN_SERVICE",
        "historical_768_SCREEN_SERVICE_SECONDS": [411.493585, 411.424327],
        "historical_comparability": "NOT_COMPARABLE_TO_EIGHT_SCREEN_J3_WORKLOAD"}
    _write_json_new(child_root / "NEXT1_checkpoint_receipt.json", receipt)
    return {"path": next_path, "raw_sha256": next_raw,
            "field_sha256": next_hashes, "receipt": receipt, "fields": outputs}


def _initial_pre_fields(context: dict) -> dict:
    from KHz_filament.hr4e5_formal_entry import pre0_fields_from_delta_n
    fields = pre0_fields_from_delta_n(context["assets"]["pre0"], expected_count=8)
    values = _validate_fields(fields, shape=(8, 351, 301), label="sealed PRE0")
    if _field_hashes(values)["delta_n"] != p1r.PRE0_ARRAY_SHA256:
        raise J3Error("generated PRE0 fields do not match the sealed canonical delta_n")
    for name in ("vx", "vy"):
        import numpy as np
        if not np.equal(values[name], 0.0).all() or np.signbit(values[name]).any():
            raise J3Error(f"sealed PRE0 {name} is not exact positive float64 zero")
    return values


def _child_failure(output_root: Path, mode: str, exc: BaseException) -> int:
    try:
        _write_json_new(output_root / "j3_child_failure.json", {
            "schema": SCHEMA, "phase": PHASE, "child_mode": mode,
            "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
            "status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()})
    except Exception:
        pass
    traceback.print_exc()
    return 1


def _current_tree_rss_gate() -> dict:
    snapshot = common.tree_snapshot(os.getpid())
    if snapshot is None:
        raise J3Error("Linux process-tree RSS snapshot is unavailable at a safe stage boundary")
    rss = int(snapshot["rss"])
    return {"rss_bytes": rss, "rss_fraction_of_320_GiB": rss / MEMORY_BYTES,
            "latch_80_percent": rss >= int(MEMORY_BYTES * RSS_WARN_FRACTION),
            "stop_90_percent": rss >= int(MEMORY_BYTES * RSS_ABORT_FRACTION),
            "sampled_process_tree_cpu_seconds_not_cumulative": float(snapshot.get("cpu", 0.0))}


def _real_chain_child(args) -> int:
    child_cpu_started = time.process_time()
    try:
        prepared, allocation = _prepare_real_context(args)
        context, hydro_parameters = prepared["context"], prepared["hydro"]
        args.output_root.mkdir(parents=True, exist_ok=False)
        pre0_fields = _initial_pre_fields(context)
        initial = _window(context, pre0_fields, output_root=args.output_root / "pulse1",
            stage_name="J3-pulse1-from-PRE0", initial_pre0=True,
            source=context["source"], source_hash=context["source_hash"],
            pre_hash=p1r.PRE0_ARRAY_SHA256, execution_scope="J3_REAL_INPUT")
        import numpy as np
        with np.load(initial["final_POST_path"], allow_pickle=False) as archive:
            post1 = _validate_fields({name: np.array(archive[name], order="C", copy=True)
                for name in FIELDS}, shape=(8, 351, 301), label="J3 POST1")
        post_hashes = _field_hashes(post1)
        if post_hashes != initial["final_POST_canonical_sha256"]:
            raise J3Error("POST1 canonical fields differ from the first optical window receipt")
        post_path = args.output_root / "POST1_checkpoint.npz"
        post_raw = _write_npz_new(post_path, post1)
        post_receipt = {"schema": SCHEMA, "phase": PHASE, "status": "CHECKPOINT_VALIDATED",
            "checkpoint": str(post_path), "raw_sha256": post_raw,
            "field_canonical_sha256": post_hashes, "source_pulse": "pulse1",
            "interval_count": 8, "screen_count": 8}
        _write_json_new(args.output_root / "POST1_checkpoint_receipt.json", post_receipt)
        pulse1_rss = _current_tree_rss_gate()
        if pulse1_rss["latch_80_percent"]:
            _write_json_new(args.output_root / "j3_child_receipt.json", {
                "schema": SCHEMA, "phase": PHASE, "child_mode": "chain",
                "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
                "status": "BLOCKED_RSS_80_PERCENT_LATCH", "execution_scope": "J3_REAL_INPUT",
                "safe_boundary": "PULSE1_COMPLETE_POST1_CHECKPOINTED",
                "rss_gate": pulse1_rss, "pulse1_window": initial,
                "POST1_checkpoint": post_receipt,
                "lineage": {"fresh_source_loads": 1, "optical_pulses": 1,
                    "POST_generations": 1, "hydro_transitions": 0}})
            return 1
        hydro = _process_hydro_workers(args, child_root=args.output_root,
            post_path=post_path, post_raw_sha256=post_raw, post_hashes=post_hashes,
            hydro_parameters=hydro_parameters, execution_scope="J3_REAL_INPUT")
        if hydro["receipt"]["worker_processes_completed"] != 8:
            raise J3Error("J3 requires exactly eight completed single-screen HR4 worker processes")
        hydro_rss = _current_tree_rss_gate()
        hydro_rss["latch_80_percent"] = bool(
            hydro_rss["latch_80_percent"]
            or hydro["receipt"].get("rss_80_percent_latched", False))
        hydro_rss["worker_batch_peak_rss_bytes"] = hydro["receipt"].get(
            "aggregate_peak_process_tree_rss_bytes")
        if hydro_rss["latch_80_percent"]:
            _write_json_new(args.output_root / "j3_child_receipt.json", {
                "schema": SCHEMA, "phase": PHASE, "child_mode": "chain",
                "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
                "status": "BLOCKED_RSS_80_PERCENT_LATCH", "execution_scope": "J3_REAL_INPUT",
                "safe_boundary": "NEXT1_CHECKPOINTED_BEFORE_PULSE2",
                "rss_gate": hydro_rss, "pulse1_window": initial,
                "POST1_checkpoint": post_receipt, "hydro": hydro["receipt"],
                "NEXT1": {"raw_sha256": hydro["raw_sha256"],
                    "field_canonical_sha256": hydro["field_sha256"]},
                "lineage": {"fresh_source_loads": 1, "optical_pulses": 1,
                    "POST_generations": 1, "hydro_transitions": 1}})
            return 1
        # Reload the sealed source from disk before the second optical pulse.
        source2, source2_hash, _ = p1r.validate_frozen_source(args.input_root, context["assets"])
        if source2_hash != context["source_hash"] or source2_hash != p1r.SOURCE_ARRAY_SHA256:
            raise J3Error("freshly reloaded source for pulse2 differs from the sealed source")
        if np.shares_memory(source2, context["source"]):
            raise J3Error("pulse2 source reload aliases pulse1 source")
        pulse2 = _window(context, hydro["fields"], output_root=args.output_root / "pulse2_chain",
            stage_name="J3-pulse2-from-in-memory-NEXT1", initial_pre0=False,
            source=source2, source_hash=source2_hash, pre_hash=None,
            execution_scope="J3_REAL_INPUT")
        receipt = {"schema": SCHEMA, "phase": PHASE, "child_mode": "chain",
            "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
            "status": "COMPLETE_DIAGNOSTIC", "execution_scope": "J3_REAL_INPUT",
            "source_binding": context["source_binding"],
            "source_canonical_sha256": context["source_hash"],
            "source_raw_sha256": p1r.SOURCE_RAW_SHA256,
            "pre0_canonical_sha256": p1r.PRE0_ARRAY_SHA256,
            "hydro_parameters": hydro_parameters,
            "coordinate_frame_convention": hydro_parameters["observable_coordinate_frames"],
            "allocation": allocation,
            "lineage": {"fresh_source_loads": 2, "optical_pulses": 2,
                "POST_generations": 2, "hydro_transitions": 1,
                "branch": "POST1_TO_EIGHT_REAL_HR4_WORKERS_TO_NEXT1"},
            "pulse1": {"final_E_canonical_sha256": initial["final_E_canonical_sha256"],
                "POST_raw_sha256": post_raw, "POST_field_sha256": post_hashes,
                "checkpoint_receipt": post_receipt},
            "hydro": hydro["receipt"],
            "NEXT1": {"raw_sha256": hydro["raw_sha256"],
                "field_canonical_sha256": hydro["field_sha256"],
                "checkpoint_receipt": hydro["receipt"]},
            "pulse2_window": pulse2,
            "chain_coordinator_process_cpu_seconds_excluding_worker_processes":
                time.process_time() - child_cpu_started,
            "process_cpu_aggregation": {
                "chain_process_cpu_excludes_hr4_worker_processes": True,
                "hr4_worker_process_cpu_sum_seconds":
                    hydro["receipt"]["worker_cpu_accounting"]["sum_worker_process_cpu_seconds"],
                "worker_kernel_load_write_cpu_are_breakdowns_not_additional_totals": True},
            "runtime_config_delta": context["runtime_config_delta"],
            "status_semantics": "DIAGNOSTIC_ONLY_NO_PRODUCTION_QUALIFICATION"}
        _write_json_new(args.output_root / "j3_child_receipt.json", receipt)
        return 0
    except BaseException as exc:
        return _child_failure(args.output_root, "chain", exc)


def _real_reload_child(args) -> int:
    child_cpu_started = time.process_time()
    try:
        prepared, allocation = _prepare_real_context(args)
        context = prepared["context"]
        args.output_root.mkdir(parents=True, exist_ok=False)
        if args.next1_path is None or args.next1_receipt is None:
            raise J3Error("reload child requires the chain's immutable NEXT1 path and receipt")
        checkpoint_receipt = json.loads(args.next1_receipt.read_text(encoding="utf-8"))
        if checkpoint_receipt.get("status") != "COMPLETE_DIAGNOSTIC":
            raise J3Error("NEXT1 checkpoint receipt is not complete")
        next1 = _load_npz(args.next1_path,
            expected_raw_sha256=checkpoint_receipt["NEXT1_raw_sha256"],
            expected_field_sha256=checkpoint_receipt["NEXT1_field_sha256"],
            shape=(8, 351, 301), label="J3 independently reloaded NEXT1")
        source, source_hash, _ = p1r.validate_frozen_source(args.input_root, context["assets"])
        if source_hash != context["source_hash"] or source_hash != p1r.SOURCE_ARRAY_SHA256:
            raise J3Error("reload-branch source differs from the sealed source")
        pulse2 = _window(context, next1, output_root=args.output_root / "pulse2_reload",
            stage_name="J3-independent-pulse2-from-disk-NEXT1", initial_pre0=False,
            source=source, source_hash=source_hash, pre_hash=None,
            execution_scope="J3_REAL_INPUT")
        receipt = {"schema": SCHEMA, "phase": PHASE, "child_mode": "reload",
            "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
            "status": "COMPLETE_DIAGNOSTIC", "execution_scope": "J3_REAL_INPUT",
            "source_binding": context["source_binding"], "source_canonical_sha256": source_hash,
            "source_raw_sha256": p1r.SOURCE_RAW_SHA256,
            "NEXT1_input": {"raw_sha256": checkpoint_receipt["NEXT1_raw_sha256"],
                "field_canonical_sha256": checkpoint_receipt["NEXT1_field_sha256"],
                "independent_disk_reload": True},
            "allocation": allocation,
            "lineage": {"fresh_source_loads": 1, "optical_pulses": 1,
                "POST_generations": 1, "hydro_transitions": 0,
                "branch": "INDEPENDENT_DISK_RELOAD_OF_CHAIN_NEXT1"},
            "pulse2_window": pulse2,
            "reload_child_process_cpu_seconds": time.process_time() - child_cpu_started,
            "status_semantics": "DIAGNOSTIC_ONLY_NO_PRODUCTION_QUALIFICATION"}
        _write_json_new(args.output_root / "j3_child_receipt.json", receipt)
        return 0
    except BaseException as exc:
        return _child_failure(args.output_root, "reload", exc)


def _mock_hydro_advance(delta_n, vx, vy, **kwargs):
    """Software-smoke placeholder; never represents an HR4 computation."""
    import numpy as np
    if int(kwargs.get("n_steps", -1)) != HYDRO_STEPS:
        raise AssertionError("J3 software smoke did not request exactly 1000 HR4 steps")
    return {"delta_n": np.array(delta_n, dtype=np.float64, order="C", copy=True),
        "vx": np.array(vx, dtype=np.float64, order="C", copy=True),
        "vy": np.array(vy, dtype=np.float64, order="C", copy=True),
        "steps": HYDRO_STEPS, "stability": {"overall_pass": True},
        "observables": {"smoke_placeholder": True},
        "performance": {"backend": "numpy", "mocked": True}}


def _smoke_context():
    context = j1._tiny_context(operator_mode="full_isaacs_eq27")
    context["assets"] = {"mapping_sha256": None}
    return context


def _smoke_chain_child(args) -> int:
    try:
        import numpy as np
        context = _smoke_context()
        args.output_root.mkdir(parents=True, exist_ok=False)
        source_hash = j1._canonical_hash(context["source"])
        pre_fields = _validate_fields(context["pre_fields"], shape=(8, 8, 8),
                                      label="software PRE0")
        pre_hash = j1._canonical_hash(pre_fields["delta_n"])
        start_cpu = time.process_time()
        pulse1 = _window(context, pre_fields, output_root=args.output_root / "pulse1",
            stage_name="J3-smoke-pulse1", initial_pre0=True, source=context["source"],
            source_hash=source_hash, pre_hash=pre_hash, execution_scope="SOFTWARE_SMOKE_ONLY")
        with np.load(pulse1["final_POST_path"], allow_pickle=False) as archive:
            post1 = _validate_fields({name: np.array(archive[name], order="C", copy=True)
                for name in FIELDS}, shape=(8, 8, 8), label="software POST1")
        post_path = args.output_root / "POST1_checkpoint.npz"
        post_raw = _write_npz_new(post_path, post1)
        post_hashes = _field_hashes(post1)
        next_fields = {name: [] for name in FIELDS}
        calls = []
        for ordinal in range(8):
            screen = {name: np.array(post1[name][ordinal], order="C", copy=True)
                      for name in FIELDS}
            result = _invoke_hydro(_mock_hydro_advance, screen, {
                "dx": 1e-5, "dy": 1e-5, "dt_hydro": 1e-6, "chi": 21.7e-6,
                "nu": 1.5e-5, "n0": 1.00027, "gravity_x": 0.0,
                "gravity_y": -9.81, "cfl_limit": 1.0, "n_steps": HYDRO_STEPS,
                "x_min": -4e-5, "y_min": -4e-5})
            calls.append({"ordinal": ordinal, "n_steps": int(result["steps"])})
            for name in FIELDS:
                next_fields[name].append(np.array(result[name], order="C", copy=True))
        next_fields = {name: np.ascontiguousarray(np.stack(values), dtype=np.float64)
                       for name, values in next_fields.items()}
        next_hashes = _field_hashes(next_fields)
        next_path = args.output_root / "NEXT1_checkpoint.npz"
        next_raw = _write_npz_new(next_path, next_fields)
        pulse2_source = np.array(context["source"], dtype=np.complex128, order="C", copy=True)
        pulse2 = _window(context, next_fields, output_root=args.output_root / "pulse2_chain",
            stage_name="J3-smoke-pulse2-chain", initial_pre0=False, source=pulse2_source,
            source_hash=source_hash, pre_hash=None, execution_scope="SOFTWARE_SMOKE_ONLY")
        receipt = {"schema": SCHEMA, "phase": PHASE, "child_mode": "smoke-chain",
            "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
            "execution_scope": "SOFTWARE_SMOKE_ONLY", "status": "COMPLETE_DIAGNOSTIC",
            "source_canonical_sha256": source_hash,
            "POST1": {"raw_sha256": post_raw, "field_canonical_sha256": post_hashes},
            "NEXT1": {"raw_sha256": next_raw, "field_canonical_sha256": next_hashes,
                "checkpoint_receipt": {"status": "COMPLETE_DIAGNOSTIC"}},
            "hydro_smoke": {"method": "MOCKED_NO_HR4_PHYSICS",
                "worker_processes_launched": 0, "mocked_advance_calls": len(calls),
                "calls": calls, "requested_n_steps": HYDRO_STEPS},
            "lineage": {"fresh_source_loads": 2, "optical_pulses": 2,
                "POST_generations": 2, "hydro_transitions": 1,
                "branch": "SOFTWARE_SMOKE_ONLY"},
            "pulse2_window": pulse2,
            "child_process_cpu_seconds_excluding_children": time.process_time() - start_cpu}
        _write_json_new(args.output_root / "NEXT1_checkpoint_receipt.json", {
            "schema": SCHEMA, "phase": PHASE, "status": "COMPLETE_DIAGNOSTIC",
            "NEXT1_raw_sha256": next_raw, "NEXT1_field_sha256": next_hashes,
            "execution_scope": "SOFTWARE_SMOKE_ONLY"})
        _write_json_new(args.output_root / "j3_child_receipt.json", receipt)
        return 0
    except BaseException as exc:
        return _child_failure(args.output_root, "smoke-chain", exc)


def _smoke_reload_child(args) -> int:
    try:
        context = _smoke_context()
        args.output_root.mkdir(parents=True, exist_ok=False)
        if args.next1_path is None or args.next1_receipt is None:
            raise J3Error("smoke reload requires NEXT1 path and receipt")
        checkpoint = json.loads(args.next1_receipt.read_text(encoding="utf-8"))
        next_fields = _load_npz(args.next1_path,
            expected_raw_sha256=checkpoint["NEXT1_raw_sha256"],
            expected_field_sha256=checkpoint["NEXT1_field_sha256"],
            shape=(8, 8, 8), label="software smoke reloaded NEXT1")
        import numpy as np
        source = np.array(context["source"], dtype=np.complex128, order="C", copy=True)
        source_hash = j1._canonical_hash(source)
        pulse2 = _window(context, next_fields, output_root=args.output_root / "pulse2_reload",
            stage_name="J3-smoke-pulse2-reload", initial_pre0=False, source=source,
            source_hash=source_hash, pre_hash=None, execution_scope="SOFTWARE_SMOKE_ONLY")
        receipt = {"schema": SCHEMA, "phase": PHASE, "child_mode": "smoke-reload",
            "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
            "execution_scope": "SOFTWARE_SMOKE_ONLY", "status": "COMPLETE_DIAGNOSTIC",
            "source_canonical_sha256": source_hash,
            "NEXT1_input": {"raw_sha256": checkpoint["NEXT1_raw_sha256"],
                "field_canonical_sha256": checkpoint["NEXT1_field_sha256"],
                "independent_disk_reload": True},
            "lineage": {"fresh_source_loads": 1, "optical_pulses": 1,
                "POST_generations": 1, "hydro_transitions": 0,
                "branch": "SOFTWARE_SMOKE_ONLY_INDEPENDENT_RELOAD"},
            "pulse2_window": pulse2}
        _write_json_new(args.output_root / "j3_child_receipt.json", receipt)
        return 0
    except BaseException as exc:
        return _child_failure(args.output_root, "smoke-reload", exc)


def _smoke_command(args, mode: str, child_root: Path, **extra):
    command = [sys.executable, "-m", "tools.cpu_p1r2.cpu_p1r2_j3",
               "--phase", PHASE, "--output-root", str(child_root),
               "--smoke-child", mode]
    for key, value in extra.items():
        command.extend((f"--{key.replace('_', '-')}", str(value)))
    return command


def _entry_smoke(args) -> int:
    try:
        if args.bench_root is None:
            raise J3Error("entry smoke requires --bench-root to bind its isolated child imports")
        args.output_root.mkdir(parents=True, exist_ok=False)
        stages, memory_latched = [], False
        chain_dir = args.output_root / "chain-stage"
        chain_dir.mkdir(exist_ok=False)
        chain_root = chain_dir / "child"
        env = j1._child_environment(args.bench_root)
        chain_sup = j1._spawn_and_monitor(_smoke_command(args, "chain", chain_root),
            stage_dir=chain_dir, cwd=args.bench_root / "Filament_python", env=env,
            timeout_seconds=900, memory_bytes=MEMORY_BYTES)
        chain_status = "DONE" if chain_sup.get("exit_code") == 0 and not chain_sup.get("stop") else "FAIL"
        stages.append({"stage": "smoke-chain", "status": chain_status, "supervisor": chain_sup})
        memory_latched = bool(chain_sup.get("rss_80_percent_latched"))
        if chain_status != "DONE" or memory_latched:
            raise J3Error("software smoke chain failed or latched the 80 percent RSS stop gate")
        chain_receipt = json.loads((chain_root / "j3_child_receipt.json").read_text(encoding="utf-8"))
        next1_receipt = chain_root / "NEXT1_checkpoint_receipt.json"
        reload_dir = args.output_root / "reload-stage"
        reload_dir.mkdir(exist_ok=False)
        reload_root = reload_dir / "child"
        reload_sup = j1._spawn_and_monitor(_smoke_command(args, "reload", reload_root,
            next1_path=chain_root / "NEXT1_checkpoint.npz", next1_receipt=next1_receipt),
            stage_dir=reload_dir, cwd=args.bench_root / "Filament_python", env=env,
            timeout_seconds=900, memory_bytes=MEMORY_BYTES)
        reload_status = "DONE" if reload_sup.get("exit_code") == 0 and not reload_sup.get("stop") else "FAIL"
        stages.append({"stage": "smoke-reload", "status": reload_status, "supervisor": reload_sup})
        if reload_status != "DONE":
            raise J3Error("software smoke independent reload child failed")
        reload_receipt = json.loads((reload_root / "j3_child_receipt.json").read_text(encoding="utf-8"))
        comparison = j1._compare_stage_files(chain_receipt["pulse2_window"],
            reload_receipt["pulse2_window"], "smoke_chain_pulse2_vs_independent_reload")
        exact = bool(comparison.get("canonical_exact_match"))
        result = {"schema": SCHEMA, "phase": PHASE,
            "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
            "execution_scope": "SOFTWARE_SMOKE_ONLY",
            "status": "SOFTWARE_SMOKE_EXACT" if exact else "SOFTWARE_SMOKE_DIFFERENCE",
            "stages": stages, "comparison": comparison,
            "mocked_hr4_calls": chain_receipt["hydro_smoke"]["mocked_advance_calls"],
            "real_hr4_physics_executed": False, "worker_processes": 0}
        _write_json_new(args.output_root / "entry_smoke_receipt.json", result)
        return 0 if exact else 1
    except BaseException as exc:
        traceback.print_exc()
        try:
            _write_json_new(args.output_root / "entry_smoke_failure.json", {
                "schema": SCHEMA, "phase": PHASE, "execution_scope": "SOFTWARE_SMOKE_ONLY",
                "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
        return 1


def _run_j3(args) -> int:
    if args.phase != PHASE:
        raise J3Error("J3 coordinator requires explicit --phase J3")
    prepared, allocation = _prepare_real_context(args)
    context = prepared["context"]
    hydro_parameters = prepared["hydro"]
    allocation_start = _allocation_start_epoch()
    allocation_deadline = allocation_start + 7200.0 - 900.0
    work_deadline = min(allocation_start + J3_WORK_LIMIT_SECONDS, allocation_deadline)
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(exist_ok=False)
    p1r._environment_receipt(args.output_root, vendor_dir=args.vendor_dir, allocation=allocation)
    input_summary = {"source_raw_sha256": p1r.SOURCE_RAW_SHA256,
        "source_canonical_sha256": context["source_hash"],
        "pre0_raw_sha256": p1r.PRE0_RAW_SHA256,
        "pre0_canonical_sha256": context["assets"]["pre0_canonical_sha256"],
        "transfer_mapping_sha256": context["assets"]["mapping_sha256"],
        "schedule": j1._json_safe(context["schedule"].as_metadata()),
        "runtime_config_delta": j1._json_safe(context["runtime_config_delta"]),
        "hydro_parameters": hydro_parameters}
    _write_json_new(args.output_root / "input_validation_receipt.json", input_summary)
    del prepared, context, hydro_parameters
    import gc
    gc.collect()
    start_cpu = time.process_time()
    stages = []
    comparison = None
    remaining = work_deadline - time.time()
    predicted_chain = FIRST_CHILD_TIMEOUT_SECONDS
    predicted_reload = OPTICAL_WINDOW_ESTIMATE_SECONDS
    if remaining < predicted_chain + predicted_reload:
        stages.append({"stage": "chain", "status": "SKIPPED_INSUFFICIENT_TIME",
            "remaining_seconds": remaining, "required_predicted_seconds":
                predicted_chain + predicted_reload})
    else:
        chain = _supervise_child(args, "chain", "chain",
            timeout_seconds=min(FIRST_CHILD_TIMEOUT_SECONDS,
                                max(1.0, remaining - predicted_reload)))
        stages.append(chain)
        memory_latched = bool(chain.get("supervisor", {}).get("rss_80_percent_latched"))
        if chain.get("status") == "DONE" and not memory_latched:
            remaining = work_deadline - time.time()
            if remaining < predicted_reload:
                stages.append({"stage": "reload", "status": "SKIPPED_INSUFFICIENT_TIME",
                    "remaining_seconds": remaining,
                    "predicted_seconds": predicted_reload})
            else:
                chain_root = Path(chain["child_root"])
                checkpoint_receipt = json.loads(
                    (chain_root / "NEXT1_checkpoint_receipt.json").read_text(encoding="utf-8"))
                reload = _supervise_child(args, "reload", "reload",
                    timeout_seconds=min(RELOAD_CHILD_TIMEOUT_SECONDS,
                        max(1.0, remaining)),
                    extra={"next1_path": chain_root / "NEXT1_checkpoint.npz",
                           "next1_receipt": chain_root / "NEXT1_checkpoint_receipt.json"})
                stages.append(reload)
                if reload.get("status") == "DONE":
                    chain_receipt = chain["child_receipt"]
                    reload_receipt = reload["child_receipt"]
                    comparison = j1._compare_stage_files(
                        chain_receipt["pulse2_window"], reload_receipt["pulse2_window"],
                        "J3_chain_pulse2_vs_independent_NEXT1_reload")
                    comparison["NEXT1_checkpoint_identity_equal"] = (
                        chain_receipt["NEXT1"]["raw_sha256"]
                        == reload_receipt["NEXT1_input"]["raw_sha256"]
                        and chain_receipt["NEXT1"]["field_canonical_sha256"]
                        == reload_receipt["NEXT1_input"]["field_canonical_sha256"])
                    comparison["canonical_exact_match"] = bool(
                        comparison["canonical_exact_match"]
                        and comparison["NEXT1_checkpoint_identity_equal"])
                    comparison["classification"] = (
                        "EXACT_DIAGNOSTIC_MATCH" if comparison["canonical_exact_match"]
                        else "NUMERICAL_DIFFERENCE")
                    _write_json_new(args.output_root / "chain_vs_reload_comparison.json",
                                    comparison)
        elif memory_latched:
            stages.append({"stage": "reload", "status": "BLOCKED_RSS_80_PERCENT_LATCH"})
        else:
            stages.append({"stage": "reload", "status": "BLOCKED_AFTER_CHAIN_FAILURE"})
    chain_done = any(row.get("stage") == "chain" and row.get("status") == "DONE"
                     for row in stages)
    reload_done = any(row.get("stage") == "reload" and row.get("status") == "DONE"
                      for row in stages)
    if chain_done and reload_done and comparison is not None:
        status = ("COMPLETE_DIAGNOSTIC" if comparison["canonical_exact_match"]
                  else "NUMERICAL_DIFFERENCE")
    elif any(row.get("status") == "LIMIT_REACHED" for row in stages):
        status = "LIMIT_REACHED"
    else:
        status = "INCOMPLETE"
    receipt = {"schema": SCHEMA, "phase": PHASE,
        "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
        "status": status, "execution_mode": "SERIAL_CHAIN_THEN_INDEPENDENT_RELOAD",
        "allocation": allocation, "allocation_start_epoch": allocation_start,
        "allocation_wall_limit_seconds": 7200, "allocation_reserve_seconds": 900,
        "j3_work_limit_seconds": J3_WORK_LIMIT_SECONDS, "work_deadline_epoch": work_deadline,
        "stage_order": ["chain", "reload"], "stages": stages,
        "comparison": comparison, "input_validation": input_summary,
        "optical_scheduling_model": {
            "recent_J1_baseline_service_seconds": OPTICAL_WINDOW_BASELINE_MEASURED_SECONDS,
            "historical_prior_estimate_seconds": 512.0,
            "scheduling_window_seconds": OPTICAL_WINDOW_ESTIMATE_SECONDS,
            "scheduling_margin_fraction": OPTICAL_WINDOW_SCHEDULING_MARGIN,
            "interpretation": "Conservative stage-scheduling allowance, not a wall-clock ETA"},
        "hydro_scheduling_model": {
            "eight_screen_allowance_seconds": HYDRO_STAGE_ALLOWANCE_SECONDS,
            "bounded_stage_timeout_seconds": HYDRO_STAGE_TIMEOUT_SECONDS,
            "interpretation": "Allowance is not a measured prediction; actual eight-screen service is measured in this run",
            "historical_768_screen_service_seconds_not_comparable": [411.493585, 411.424327]},
        "process_cpu_accounting": {
            "coordinator_process_cpu_seconds": time.process_time() - start_cpu,
            "child_process_cpu_is_reported_in_each_child_receipt": True,
            "hr4_worker_process_cpu_is_reported_as_a_separate_sum": True,
            "subphase_cpu_is_not_added_to_parent_or_worker_totals": True},
        "rss_policy": {"80_percent": "LATCH_AND_BLOCK_LATER_STAGE",
            "90_percent": "BOUNDED_TERMINATION_OF_CURRENT_CHILD",
            "memory_limit_bytes": MEMORY_BYTES},
        "diagnostic_boundary": "P1R2_J3_BASELINE_DIAGNOSTIC_ONLY"}
    _write_json_new(args.output_root / "run_receipt.json", receipt)
    return 0 if status == "COMPLETE_DIAGNOSTIC" else 1


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=(PHASE,))
    parser.add_argument("--bench-root", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--vendor-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--child-mode", choices=("chain", "reload"), help=argparse.SUPPRESS)
    parser.add_argument("--smoke-child", choices=("chain", "reload"), help=argparse.SUPPRESS)
    parser.add_argument("--next1-path", type=Path)
    parser.add_argument("--next1-receipt", type=Path)
    parser.add_argument("--worker-task", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--entry-smoke", action="store_true")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.worker_task is not None:
            return _worker_task(args.worker_task)
        if args.output_root is None:
            raise J3Error("--output-root is required")
        if args.smoke_child == "chain":
            return _smoke_chain_child(args)
        if args.smoke_child == "reload":
            return _smoke_reload_child(args)
        if args.entry_smoke:
            return _entry_smoke(args)
        if args.child_mode == "chain":
            return _real_chain_child(args)
        if args.child_mode == "reload":
            return _real_reload_child(args)
        return _run_j3(args)
    except BaseException as exc:
        traceback.print_exc()
        if args.output_root is not None:
            try:
                target = args.output_root / "coordinator_failure.json"
                if args.output_root.is_dir():
                    _write_json_new(target, {"schema": SCHEMA, "phase": PHASE,
                        "status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()})
            except Exception:
                pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
