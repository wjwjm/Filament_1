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
OPTICAL_WINDOW_BASELINE_MEASURED_SECONDS = 518.0
OPTICAL_WINDOW_SCHEDULING_MARGIN = 0.30
OPTICAL_WINDOW_ESTIMATE_SECONDS = OPTICAL_WINDOW_BASELINE_MEASURED_SECONDS * (
    1.0 + OPTICAL_WINDOW_SCHEDULING_MARGIN)
HYDRO_STAGE_ALLOWANCE_SECONDS = 120.0  # scheduling allowance, not a measured prediction
HYDRO_STAGE_TIMEOUT_SECONDS = 180.0
FIRST_CHILD_TIMEOUT_SECONDS = 2700.0
RELOAD_CHILD_TIMEOUT_SECONDS = 2700.0
J3_WORK_LIMIT_SECONDS = 6300.0
J3_TAIL_RESERVE_SECONDS = 900.0
MEMORY_BYTES = 320 * 1024**3
RSS_WARN_FRACTION = 0.80
RSS_ABORT_FRACTION = 0.90
J2_TOOL_COMMIT = "5bdc064aaa8f7458ea65134a28ac9b8265041681"
J2_MODULE_RELATIVE_PATH = "Filament_python/tools/cpu_p1r2/cpu_p1r2_j2.py"
J2_ADMISSION_HASH_SCOPE = {"descriptor": "raw_bytes", "j2_result": "raw_bytes",
    "j2_module": "git_blob_oid+canonical_lf_sha256"}


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


def _window_source_label(initial_pre0: bool, state_origin: str | None = None) -> str:
    return state_origin or ("sealed_PRE0_first_eight" if initial_pre0 else "J3_NEXT1_memory")


def _window(context: dict, pre_fields: dict, *, output_root: Path,
            stage_name: str, initial_pre0: bool, source, source_hash: str,
            pre_hash: str | None, execution_scope: str,
            propagator_factory=None, state_origin: str | None = None) -> dict:
    """Invoke the frozen complete-eight J1 propagator and verify all persisted outputs."""
    import numpy as np
    output_root.parent.mkdir(parents=True, exist_ok=True)
    receipt = j1._run_window(source=source, pre_fields=pre_fields,
        grid=context["grid"], beam=context["beam"], prop=context["prop"],
        ion=context["ion"], heat=context["heat"], raman=context["raman"],
        schedule=context["schedule"], output_dir=output_root,
        source_label=_window_source_label(initial_pre0, state_origin),
        initial_pre0=initial_pre0, profile=False, source_hash=source_hash,
        pre0_hash=pre_hash, stage_name=stage_name, execution_scope=execution_scope,
        propagator_factory=propagator_factory)
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


def _validate_candidate_admission(args) -> dict:
    """Verify the external J2 admission and the actual J2 receipt it names."""
    import json as _json
    from tools.cpu_p1r2 import cpu_p1r2_j2 as j2
    from tools.hpc_ops import provenance_v2

    if args.candidate_descriptor is None or args.candidate_admission is None:
        raise J3Error("candidate descriptor and candidate admission must be supplied together")
    descriptor_path = args.candidate_descriptor
    admission_path = args.candidate_admission
    if not descriptor_path.is_absolute() or descriptor_path.is_symlink() or not descriptor_path.is_file():
        raise J3Error("candidate descriptor must be an existing absolute non-symlink file")
    if admission_path.is_symlink() or not admission_path.is_file():
        raise J3Error("candidate admission must be an existing non-symlink file")
    descriptor_raw = provenance_v2.raw_sha256_file(descriptor_path)
    descriptor = j2._validate_descriptor(_json.loads(descriptor_path.read_text(encoding="utf-8")))
    admission = _json.loads(admission_path.read_text(encoding="utf-8"))
    required_true = ("eligible", "repeat_exact", "hard_gates")
    if admission.get("schema") != "filament.cpu_p1r2.j3_candidate_admission.v1":
        raise J3Error("candidate admission schema is unsupported")
    if any(admission.get(key) is not True for key in required_true):
        raise J3Error("candidate admission must set eligible/repeat_exact/hard_gates true")
    if admission.get("hash_scope") != J2_ADMISSION_HASH_SCOPE:
        raise J3Error("candidate admission hash_scope does not match the frozen byte scopes")
    if admission.get("descriptor_raw_sha256") != descriptor_raw:
        raise J3Error("candidate descriptor raw SHA256 does not match admission")
    if admission.get("j2_tool_git_commit") != J2_TOOL_COMMIT:
        raise J3Error("candidate admission is not bound to the frozen J2 tool commit")

    record = {"path": J2_MODULE_RELATIVE_PATH,
        "classification": "tracked_text", "hash_scope": provenance_v2.TRACKED_HASH_SCOPE,
        "git_blob_oid": admission.get("j2_module_git_blob_oid"),
        "canonical_lf_sha256": admission.get("j2_module_canonical_lf_sha256")}
    provenance_v2._validate_tracked(args.bench_root, record, strict=True)
    commit_blob = provenance_v2._git(args.bench_root, "rev-parse", "--verify",
        f"{J2_TOOL_COMMIT}:{J2_MODULE_RELATIVE_PATH}")
    if commit_blob.lower() != str(admission["j2_module_git_blob_oid"]).lower():
        raise J3Error("J2 tracked module blob differs from the admitted J2 commit object")

    result_path = Path(admission.get("j2_result_path", ""))
    if not result_path.is_absolute() or result_path.is_symlink() or not result_path.is_file():
        raise J3Error("admitted J2 result must be an existing absolute non-symlink file")
    result_raw = provenance_v2.raw_sha256_file(result_path)
    if result_raw != admission.get("j2_result_raw_sha256"):
        raise J3Error("J2 result raw SHA256 does not match admission")
    j2_receipt = _json.loads(result_path.read_text(encoding="utf-8"))
    comparisons = j2_receipt.get("comparisons")
    hard_gates_ok = (isinstance(comparisons, dict) and len(comparisons) == 4
        and all(isinstance(row, dict) and row.get("hard_gate_pass") is True
                for row in comparisons.values()))
    if (j2_receipt.get("phase") != "J2"
            or j2_receipt.get("status") != "COMPLETE_DIAGNOSTIC"
            or j2_receipt.get("eligible_for_J3_bounded_diagnostic") is not True
            or j2_receipt.get("candidate_repeat_exact") is not True
            or not hard_gates_ok
            or j2_receipt.get("candidate_descriptor") != descriptor):
        raise J3Error("actual J2 receipt does not satisfy repeat, descriptor, and hard-gate criteria")
    return {"status": "ADMITTED", "schema": admission["schema"],
        "admission_path": str(admission_path), "descriptor_path": str(descriptor_path),
        "descriptor": descriptor, "descriptor_raw_sha256": descriptor_raw,
        "j2_result_path": str(result_path), "j2_result_raw_sha256": result_raw,
        "j2_tool_git_commit": J2_TOOL_COMMIT,
        "j2_module_git_blob_oid": record["git_blob_oid"],
        "j2_module_canonical_lf_sha256": record["canonical_lf_sha256"],
        "j2_actual_eligible": True, "j2_actual_repeat_exact": True,
        "j2_actual_comparison_hard_gates": True}


def _supervise_child(args, stage: str, child_mode: str, *, timeout_seconds: float,
                     extra=None, stage_root: Path | None = None,
                     candidate_descriptor: Path | None = None) -> dict:
    root = stage_root or args.output_root
    stage_dir = root / "stages" / stage
    stage_dir.mkdir(parents=True, exist_ok=False)
    child_root = stage_dir / "child"
    env = j1._child_environment(args.bench_root)
    latch_path = (stage_dir / "rss_80_percent.latch").resolve()
    env["P1R2_RSS_LATCH"] = str(latch_path)
    child_args = dict(extra or {})
    if candidate_descriptor is not None:
        child_args["candidate_descriptor"] = candidate_descriptor
        child_args["candidate_descriptor_raw_sha256"] = args.candidate_descriptor_raw_sha256
    command = _child_command(args, child_mode, child_root, **child_args)
    started = time.perf_counter()
    supervisor = j1._spawn_and_monitor(command, stage_dir=stage_dir,
        cwd=args.bench_root / "Filament_python", env=env,
        timeout_seconds=timeout_seconds, memory_bytes=MEMORY_BYTES,
        rss_latch_path=latch_path)
    result = {"stage": stage, "status": "DONE" if supervisor.get("exit_code") == 0
        and supervisor.get("stop") is None else "FAIL_CHILD", "supervisor": supervisor,
        "child_root": str(child_root), "rss_latch_path": str(latch_path),
        "rss_latch_exists": latch_path.is_file()}
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
    marker = os.environ.get("P1R2_RSS_LATCH")
    marker_exists = bool(marker and Path(marker).is_file())
    return {"rss_bytes": rss, "rss_fraction_of_320_GiB": rss / MEMORY_BYTES,
            "latch_marker_path": marker, "latch_marker_exists": marker_exists,
            "latch_80_percent": (marker_exists
                or rss >= int(MEMORY_BYTES * RSS_WARN_FRACTION)),
            "stop_90_percent": rss >= int(MEMORY_BYTES * RSS_ABORT_FRACTION),
            "sampled_process_tree_cpu_seconds_not_cumulative": float(snapshot.get("cpu", 0.0))}


def _candidate_factory(args):
    if args.candidate_descriptor is None:
        return None
    import json as _json
    from tools.cpu_p1r2 import cpu_p1r2_j2 as j2
    actual_raw = raw_sha256_file(args.candidate_descriptor)
    expected_raw = getattr(args, "candidate_descriptor_raw_sha256", None)
    if expected_raw is not None and actual_raw != expected_raw:
        raise J3Error("candidate descriptor changed after admission validation")
    descriptor = _json.loads(args.candidate_descriptor.read_text(encoding="utf-8"))
    return j2.make_propagator_factory(descriptor)


def _real_chain_child(args) -> int:
    child_cpu_started = time.process_time()
    try:
        prepared, allocation = _prepare_real_context(args)
        context, hydro_parameters = prepared["context"], prepared["hydro"]
        args.output_root.mkdir(parents=True, exist_ok=False)
        propagator_factory = _candidate_factory(args)
        pre0_fields = _initial_pre_fields(context)
        initial = _window(context, pre0_fields, output_root=args.output_root / "pulse1",
            stage_name="J3-pulse1-from-PRE0", initial_pre0=True,
            source=context["source"], source_hash=context["source_hash"],
            pre_hash=p1r.PRE0_ARRAY_SHA256, execution_scope="J3_REAL_INPUT",
            propagator_factory=propagator_factory)
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
            execution_scope="J3_REAL_INPUT", propagator_factory=propagator_factory,
            state_origin="J3_NEXT1_MEMORY")
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
            "pulse1_window": initial,
            "pulse2_window": pulse2,
            "implementation": ("SCIPY_PRIVATE_DISPATCH_CANDIDATE"
                if propagator_factory is not None else "FROZEN_NUMPY_BASELINE"),
            "candidate_descriptor": (json.loads(args.candidate_descriptor.read_text(encoding="utf-8"))
                if args.candidate_descriptor is not None else None),
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
        # _load_context has already read and sealed-validated this child process's
        # fresh source. Reuse that independent child copy instead of loading the
        # same 649 MB asset a second time.
        source = context["source"]
        source_hash = context["source_hash"]
        source_validation = context["source_validation"]
        if source_hash != context["source_hash"] or source_hash != p1r.SOURCE_ARRAY_SHA256:
            raise J3Error("reload-branch source differs from the sealed source")
        if (source_validation.get("canonical_array_sha256") != source_hash
                or source_validation.get("file_sha256") != p1r.SOURCE_RAW_SHA256):
            raise J3Error("reload context source-validation receipt differs from sealed identity")
        pulse2 = _window(context, next1, output_root=args.output_root / "pulse2_reload",
            stage_name="J3-independent-pulse2-from-disk-NEXT1", initial_pre0=False,
            source=source, source_hash=source_hash, pre_hash=None,
            execution_scope="J3_REAL_INPUT", propagator_factory=_candidate_factory(args),
            state_origin="J3_NEXT1_DISK_RELOADED")
        receipt = {"schema": SCHEMA, "phase": PHASE, "child_mode": "reload",
            "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
            "status": "COMPLETE_DIAGNOSTIC", "execution_scope": "J3_REAL_INPUT",
            "source_binding": context["source_binding"], "source_canonical_sha256": source_hash,
            "source_raw_sha256": p1r.SOURCE_RAW_SHA256,
            "source_validation": j1._json_safe(source_validation),
            "NEXT1_input": {"raw_sha256": checkpoint_receipt["NEXT1_raw_sha256"],
                "field_canonical_sha256": checkpoint_receipt["NEXT1_field_sha256"],
                "independent_disk_reload": True,
                "state_origin": "J3_NEXT1_DISK_RELOADED"},
            "allocation": allocation,
            "lineage": {"fresh_source_loads": 1, "optical_pulses": 1,
                "POST_generations": 1, "hydro_transitions": 0,
                "branch": "INDEPENDENT_DISK_RELOAD_OF_CHAIN_NEXT1"},
            "pulse2_window": pulse2,
            "implementation": ("SCIPY_PRIVATE_DISPATCH_CANDIDATE"
                if args.candidate_descriptor is not None else "FROZEN_NUMPY_BASELINE"),
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
        propagator_factory = _candidate_factory(args)
        pre_fields = _validate_fields(context["pre_fields"], shape=(8, 8, 8),
                                      label="software PRE0")
        pre_hash = j1._canonical_hash(pre_fields["delta_n"])
        start_cpu = time.process_time()
        pulse1 = _window(context, pre_fields, output_root=args.output_root / "pulse1",
            stage_name="J3-smoke-pulse1", initial_pre0=True, source=context["source"],
            source_hash=source_hash, pre_hash=pre_hash, execution_scope="SOFTWARE_SMOKE_ONLY",
            propagator_factory=propagator_factory)
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
            source_hash=source_hash, pre_hash=None, execution_scope="SOFTWARE_SMOKE_ONLY",
            propagator_factory=propagator_factory, state_origin="J3_NEXT1_MEMORY")
        receipt = {"schema": SCHEMA, "phase": PHASE, "child_mode": "smoke-chain",
            "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
            "execution_scope": "SOFTWARE_SMOKE_ONLY", "status": "COMPLETE_DIAGNOSTIC",
            "source_canonical_sha256": source_hash,
            "POST1": {"raw_sha256": post_raw, "field_canonical_sha256": post_hashes},
            "NEXT1": {"raw_sha256": next_raw, "field_canonical_sha256": next_hashes,
                "checkpoint_receipt": {"status": "COMPLETE_DIAGNOSTIC"}},
            "pulse1_window": pulse1,
            "hydro_smoke": {"method": "MOCKED_NO_HR4_PHYSICS",
                "worker_processes_launched": 0, "mocked_advance_calls": len(calls),
                "calls": calls, "requested_n_steps": HYDRO_STEPS},
            "lineage": {"fresh_source_loads": 2, "optical_pulses": 2,
                "POST_generations": 2, "hydro_transitions": 1,
                "branch": "SOFTWARE_SMOKE_ONLY"},
            "pulse2_window": pulse2,
            "implementation": ("SCIPY_PRIVATE_DISPATCH_CANDIDATE"
                if propagator_factory is not None else "FROZEN_NUMPY_BASELINE"),
            "candidate_factory_window_count": 2 if propagator_factory is not None else 0,
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
        propagator_factory = _candidate_factory(args)
        pulse2 = _window(context, next_fields, output_root=args.output_root / "pulse2_reload",
            stage_name="J3-smoke-pulse2-reload", initial_pre0=False, source=source,
            source_hash=source_hash, pre_hash=None, execution_scope="SOFTWARE_SMOKE_ONLY",
            propagator_factory=propagator_factory,
            state_origin="J3_NEXT1_DISK_RELOADED")
        receipt = {"schema": SCHEMA, "phase": PHASE, "child_mode": "smoke-reload",
            "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
            "execution_scope": "SOFTWARE_SMOKE_ONLY", "status": "COMPLETE_DIAGNOSTIC",
            "source_canonical_sha256": source_hash,
            "NEXT1_input": {"raw_sha256": checkpoint["NEXT1_raw_sha256"],
                "field_canonical_sha256": checkpoint["NEXT1_field_sha256"],
                "independent_disk_reload": True,
                "state_origin": "J3_NEXT1_DISK_RELOADED"},
            "lineage": {"fresh_source_loads": 1, "optical_pulses": 1,
                "POST_generations": 1, "hydro_transitions": 0,
                "branch": "SOFTWARE_SMOKE_ONLY_INDEPENDENT_RELOAD"},
            "pulse2_window": pulse2,
            "implementation": ("SCIPY_PRIVATE_DISPATCH_CANDIDATE"
                if propagator_factory is not None else "FROZEN_NUMPY_BASELINE"),
            "candidate_factory_window_count": 1 if propagator_factory is not None else 0}
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


def _run_smoke_implementation(args, implementation: str,
                             candidate_descriptor: Path | None) -> dict:
    implementation_root = args.output_root / implementation
    implementation_root.mkdir(exist_ok=False)
    env = j1._child_environment(args.bench_root)
    child_args = {}
    if candidate_descriptor is not None:
        child_args = {"candidate_descriptor": candidate_descriptor,
            "candidate_descriptor_raw_sha256": args.candidate_descriptor_raw_sha256}
    chain_dir = implementation_root / "chain-stage"
    chain_dir.mkdir(exist_ok=False)
    chain_root = chain_dir / "child"
    chain_sup = j1._spawn_and_monitor(_smoke_command(args, "chain", chain_root, **child_args),
        stage_dir=chain_dir, cwd=args.bench_root / "Filament_python", env=env,
        timeout_seconds=900, memory_bytes=MEMORY_BYTES)
    chain_status = "DONE" if chain_sup.get("exit_code") == 0 and not chain_sup.get("stop") else "FAIL"
    chain = {"stage": "chain", "status": chain_status, "supervisor": chain_sup,
        "child_root": str(chain_root)}
    reload = {"stage": "reload", "status": "BLOCKED_AFTER_CHAIN_FAILURE"}
    comparison = None
    if chain_status == "DONE" and not chain_sup.get("rss_80_percent_latched"):
        chain_receipt = json.loads((chain_root / "j3_child_receipt.json").read_text(encoding="utf-8"))
        reload_dir = implementation_root / "reload-stage"
        reload_dir.mkdir(exist_ok=False)
        reload_root = reload_dir / "child"
        reload_sup = j1._spawn_and_monitor(_smoke_command(args, "reload", reload_root,
            next1_path=chain_root / "NEXT1_checkpoint.npz",
            next1_receipt=chain_root / "NEXT1_checkpoint_receipt.json", **child_args),
            stage_dir=reload_dir, cwd=args.bench_root / "Filament_python", env=env,
            timeout_seconds=900, memory_bytes=MEMORY_BYTES)
        reload_status = "DONE" if reload_sup.get("exit_code") == 0 and not reload_sup.get("stop") else "FAIL"
        reload = {"stage": "reload", "status": reload_status, "supervisor": reload_sup,
            "child_root": str(reload_root)}
        if reload_status == "DONE":
            reload_receipt = json.loads((reload_root / "j3_child_receipt.json").read_text(encoding="utf-8"))
            comparison = j1._compare_stage_files(chain_receipt["pulse2_window"],
                reload_receipt["pulse2_window"], f"{implementation}_smoke_chain_vs_disk_reload")
            chain_pre = _verify_window_pre_lineage(chain_receipt["pulse2_window"],
                chain_receipt["NEXT1"]["field_canonical_sha256"], "smoke chain pulse2")
            reload_pre = _verify_window_pre_lineage(reload_receipt["pulse2_window"],
                reload_receipt["NEXT1_input"]["field_canonical_sha256"], "smoke reload pulse2")
            comparison["same_implementation_PRE_fields_equal"] = chain_pre == reload_pre
            comparison["reload_state_origin_is_disk_NEXT1"] = (
                reload_receipt.get("NEXT1_input", {}).get("state_origin")
                == "J3_NEXT1_DISK_RELOADED")
            comparison["canonical_exact_match"] = bool(comparison["canonical_exact_match"]
                and comparison["same_implementation_PRE_fields_equal"]
                and comparison["reload_state_origin_is_disk_NEXT1"])
    return {"implementation": implementation, "chain": chain, "reload": reload,
        "comparison": comparison,
        "repeat_exact": bool(comparison and comparison.get("canonical_exact_match")),
        "candidate_factory_window_count": (chain_receipt.get("candidate_factory_window_count", 0)
            + reload_receipt.get("candidate_factory_window_count", 0)
            if chain_status == "DONE" and reload.get("status") == "DONE" else 0)}


def _entry_smoke(args) -> int:
    try:
        if args.bench_root is None:
            raise J3Error("entry smoke requires --bench-root to bind its isolated child imports")
        candidate_requested = (args.candidate_descriptor is not None
            or args.candidate_admission is not None)
        candidate_gate = {"status": "NOT_REQUESTED"}
        if candidate_requested:
            candidate_gate = _validate_candidate_admission(args)
            args.candidate_descriptor_raw_sha256 = candidate_gate["descriptor_raw_sha256"]
        args.output_root.mkdir(parents=True, exist_ok=False)
        baseline = _run_smoke_implementation(args, "baseline", None)
        candidate = None
        cross = None
        if candidate_requested:
            candidate = _run_smoke_implementation(args, "candidate", args.candidate_descriptor)
            from tools.cpu_p1r2 import cpu_p1r2_j2 as j2
            base_chain = json.loads((Path(baseline["chain"]["child_root"])
                / "j3_child_receipt.json").read_text(encoding="utf-8"))
            cand_chain = json.loads((Path(candidate["chain"]["child_root"])
                / "j3_child_receipt.json").read_text(encoding="utf-8"))
            base_reload = json.loads((Path(baseline["reload"]["child_root"])
                / "j3_child_receipt.json").read_text(encoding="utf-8"))
            cand_reload = json.loads((Path(candidate["reload"]["child_root"])
                / "j3_child_receipt.json").read_text(encoding="utf-8"))
            cross = {"pulse1": _cross_pulse1_compare(base_chain["pulse1_window"],
                    cand_chain["pulse1_window"], "smoke_baseline_vs_candidate_pulse1"),
                "pulse2_chain": _cross_lineaged_pulse2_compare(
                    base_chain["pulse2_window"], cand_chain["pulse2_window"],
                    base_chain["NEXT1"]["field_canonical_sha256"],
                    cand_chain["NEXT1"]["field_canonical_sha256"],
                    "smoke_baseline_vs_candidate_chain_pulse2"),
                "pulse2_reload": _cross_lineaged_pulse2_compare(
                    base_reload["pulse2_window"], cand_reload["pulse2_window"],
                    base_reload["NEXT1_input"]["field_canonical_sha256"],
                    cand_reload["NEXT1_input"]["field_canonical_sha256"],
                    "smoke_baseline_vs_candidate_reload_pulse2")}
            cross["hard_gates_pass"] = all(cross[name].get("hard_gate_pass") is True
                for name in ("pulse1", "pulse2_chain", "pulse2_reload"))
        exact = bool(baseline["repeat_exact"] and (candidate is None
            or (candidate["repeat_exact"] and candidate["candidate_factory_window_count"] == 3
                and cross["hard_gates_pass"])))
        result = {"schema": SCHEMA, "phase": PHASE,
            "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
            "execution_scope": "SOFTWARE_SMOKE_ONLY",
            "status": "SOFTWARE_SMOKE_EXACT" if exact else "SOFTWARE_SMOKE_DIFFERENCE",
            "baseline": baseline, "candidate": candidate,
            "candidate_admission": candidate_gate, "cross_implementation": cross,
            "mocked_hr4_calls": 8,
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


def _run_implementation(args, implementation: str, stage_root: Path, *,
                        candidate_descriptor: Path | None, deadline: float) -> dict:
    stage_root.mkdir(parents=True, exist_ok=False)
    predicted_chain = 2 * OPTICAL_WINDOW_ESTIMATE_SECONDS + HYDRO_STAGE_TIMEOUT_SECONDS
    predicted_reload = OPTICAL_WINDOW_ESTIMATE_SECONDS
    stages = []
    comparison = None
    remaining = deadline - time.time()
    if remaining < predicted_chain + predicted_reload:
        stages.append({"stage": "chain", "status": "SKIPPED_INSUFFICIENT_TIME",
            "remaining_seconds": remaining, "predicted_chain_seconds": predicted_chain,
            "predicted_reload_seconds": predicted_reload,
            "remaining_in_work_budget": True})
    else:
        chain_timeout = min(FIRST_CHILD_TIMEOUT_SECONDS,
            max(1.0, remaining - predicted_reload))
        chain = _supervise_child(args, "chain", "chain", stage_root=stage_root,
            candidate_descriptor=candidate_descriptor, timeout_seconds=chain_timeout)
        stages.append(chain)
        marker_latched = bool(chain.get("rss_latch_exists")
            or chain.get("supervisor", {}).get("rss_80_percent_latched"))
        if chain.get("status") == "DONE" and not marker_latched:
            remaining = deadline - time.time()
            if remaining < max(predicted_reload, J3_TAIL_RESERVE_SECONDS):
                stages.append({"stage": "reload", "status": "SKIPPED_INSUFFICIENT_TIME",
                    "remaining_seconds": remaining, "predicted_seconds": predicted_reload,
                    "tail_reserve_seconds_within_work_budget": J3_TAIL_RESERVE_SECONDS})
            else:
                chain_root = Path(chain["child_root"])
                reload_timeout = min(RELOAD_CHILD_TIMEOUT_SECONDS,
                    max(1.0, remaining))
                reload = _supervise_child(args, "reload", "reload",
                    stage_root=stage_root, candidate_descriptor=candidate_descriptor,
                    timeout_seconds=reload_timeout,
                    extra={"next1_path": chain_root / "NEXT1_checkpoint.npz",
                           "next1_receipt": chain_root / "NEXT1_checkpoint_receipt.json"})
                stages.append(reload)
                if reload.get("status") == "DONE":
                    chain_receipt = chain["child_receipt"]
                    reload_receipt = reload["child_receipt"]
                    chain_pre = _verify_window_pre_lineage(chain_receipt["pulse2_window"],
                        chain_receipt["NEXT1"]["field_canonical_sha256"],
                        f"{implementation} chain pulse2")
                    reload_pre = _verify_window_pre_lineage(reload_receipt["pulse2_window"],
                        reload_receipt["NEXT1_input"]["field_canonical_sha256"],
                        f"{implementation} reload pulse2")
                    comparison = j1._compare_stage_files(
                        chain_receipt["pulse2_window"], reload_receipt["pulse2_window"],
                        f"J3_{implementation}_chain_pulse2_vs_independent_disk_NEXT1_reload")
                    comparison["same_implementation_PRE_fields_equal"] = chain_pre == reload_pre
                    comparison["NEXT1_checkpoint_identity_equal"] = (
                        chain_receipt["NEXT1"]["raw_sha256"]
                        == reload_receipt["NEXT1_input"]["raw_sha256"]
                        and chain_receipt["NEXT1"]["field_canonical_sha256"]
                        == reload_receipt["NEXT1_input"]["field_canonical_sha256"])
                    comparison["reload_state_origin_is_disk_NEXT1"] = (
                        reload_receipt["NEXT1_input"].get("state_origin")
                        == "J3_NEXT1_DISK_RELOADED")
                    comparison["canonical_exact_match"] = bool(
                        comparison["canonical_exact_match"]
                        and comparison["NEXT1_checkpoint_identity_equal"]
                        and comparison["same_implementation_PRE_fields_equal"]
                        and comparison["reload_state_origin_is_disk_NEXT1"])
                    comparison["classification"] = (
                        "EXACT_DIAGNOSTIC_MATCH" if comparison["canonical_exact_match"]
                        else "NUMERICAL_DIFFERENCE")
                    _write_json_new(stage_root / "chain_vs_reload_comparison.json", comparison)
        elif marker_latched:
            stages.append({"stage": "reload", "status": "BLOCKED_RSS_80_PERCENT_LATCH"})
        else:
            stages.append({"stage": "reload", "status": "BLOCKED_AFTER_CHAIN_FAILURE"})
    chain = next((row for row in stages if row.get("stage") == "chain"), {})
    reload = next((row for row in stages if row.get("stage") == "reload"), {})
    chain_done = chain.get("status") == "DONE"
    reload_done = reload.get("status") == "DONE"
    complete = bool(chain_done and reload_done and comparison is not None)
    repeat_exact = bool(complete and comparison.get("canonical_exact_match"))
    status = ("COMPLETE_DIAGNOSTIC" if repeat_exact else
        "NUMERICAL_DIFFERENCE" if complete else
        "LIMIT_REACHED" if any(row.get("status") == "LIMIT_REACHED" for row in stages)
        else "INCOMPLETE")
    result = {"implementation": implementation, "status": status,
        "stages": stages, "chain_vs_reload_comparison": comparison,
        "repeat_exact": repeat_exact,
        "rss_80_percent_latched": any(bool(row.get("rss_latch_exists")
            or row.get("supervisor", {}).get("rss_80_percent_latched")) for row in stages),
        "predicted_chain_seconds": predicted_chain,
        "predicted_reload_seconds": predicted_reload,
        "prediction_basis": "518_seconds_per_window_times_1.3_plus_180_second_HR4_budget"}
    _write_json_new(stage_root / "implementation_receipt.json", result)
    return result


def _next1_comparison(args, baseline: dict, candidate: dict) -> dict:
    import numpy as np
    from tools.cpu_p1r2 import cpu_p1r2_j2 as j2
    baseline_chain = next(row for row in baseline["stages"] if row.get("stage") == "chain")
    candidate_chain = next(row for row in candidate["stages"] if row.get("stage") == "chain")
    baseline_root = Path(baseline_chain["child_root"])
    candidate_root = Path(candidate_chain["child_root"])
    left_receipt = json.loads((baseline_root / "NEXT1_checkpoint_receipt.json").read_text(encoding="utf-8"))
    right_receipt = json.loads((candidate_root / "NEXT1_checkpoint_receipt.json").read_text(encoding="utf-8"))
    left = _load_npz(baseline_root / "NEXT1_checkpoint.npz",
        expected_raw_sha256=left_receipt["NEXT1_raw_sha256"],
        expected_field_sha256=left_receipt["NEXT1_field_sha256"],
        shape=(8, 351, 301), label="J3 baseline NEXT1 comparison")
    right = _load_npz(candidate_root / "NEXT1_checkpoint.npz",
        expected_raw_sha256=right_receipt["NEXT1_raw_sha256"],
        expected_field_sha256=right_receipt["NEXT1_field_sha256"],
        shape=(8, 351, 301), label="J3 candidate NEXT1 comparison")
    metrics = {}
    for name in FIELDS:
        metrics[name] = j2._array_error_metrics(np.asarray(left[name]), np.asarray(right[name]))
    return {"label": "J3_baseline_vs_candidate_NEXT1_fields",
        "fields": metrics, "all_fields_canonical_exact": all(
            item.get("canonical_exact", False) for item in metrics.values()),
        "interpretation": "NUMERICAL_DIFFERENCE_IS_DIAGNOSTIC_ONLY"}


def _window_pre_fields(window: dict) -> dict:
    value = window.get("receipt", {}).get("input_PRE_fields_canonical_sha256")
    return value if isinstance(value, dict) else {}


def _verify_window_pre_lineage(window: dict, checkpoint_fields: dict, label: str) -> dict:
    actual = _window_pre_fields(window)
    receipt = window.get("receipt", {})
    pre0_expected = receipt.get("pre0_expected_canonical_sha256")
    pre0_actual = receipt.get("pre0_canonical_sha256")
    pre0_contract_ok = (pre0_actual == actual.get("delta_n") if pre0_expected is not None
        else pre0_actual is None)
    valid = (set(actual) == set(FIELDS) and set(checkpoint_fields) == set(FIELDS)
        and actual == checkpoint_fields and pre0_contract_ok)
    if not valid:
        raise J3Error(f"{label} optical PRE hashes do not match their verified NEXT1 lineage")
    return actual


def _cross_pulse1_compare(left: dict, right: dict, label: str) -> dict:
    from tools.cpu_p1r2 import cpu_p1r2_j2 as j2
    left_pre, right_pre = _window_pre_fields(left), _window_pre_fields(right)
    raw = j2._compare_stage_files(left, right, label)
    result = dict(raw)
    errors = list(raw.get("hard_gate_errors", []))
    same_pre0 = set(left_pre) == set(FIELDS) and left_pre == right_pre
    if not same_pre0:
        errors.append("pre0_full_field_hashes_changed")
    result["raw_j2_hard_gate_errors"] = list(raw.get("hard_gate_errors", []))
    result["effective_hard_gate_errors"] = errors
    result["pre0_full_field_hashes_equal"] = same_pre0
    result["hard_gate_errors"] = errors
    result["hard_gate_pass"] = not errors
    return result


def _cross_lineaged_pulse2_compare(left: dict, right: dict,
                                    left_next_fields: dict, right_next_fields: dict,
                                    label: str) -> dict:
    """Preserve J2 evidence; reclassify only verified separately derived PRE2."""
    from tools.cpu_p1r2 import cpu_p1r2_j2 as j2
    left_pre = _verify_window_pre_lineage(left, left_next_fields, f"{label} left")
    right_pre = _verify_window_pre_lineage(right, right_next_fields, f"{label} right")
    raw = j2._compare_stage_files(left, right, label)
    raw_errors = list(raw.get("hard_gate_errors", []))
    left_source = left.get("receipt", {}).get("source_canonical_sha256")
    right_source = right.get("receipt", {}).get("source_canonical_sha256")
    source_equal = bool(left_source and left_source == right_source)
    distinct_pre = left_pre != right_pre
    allowed_pre_error = (distinct_pre and source_equal
        and "pre0_canonical_sha256_changed" in raw_errors)
    effective = [error for error in raw_errors
        if not (allowed_pre_error and error == "pre0_canonical_sha256_changed")]
    if not source_equal and "source_fresh_input_changed" not in effective:
        effective.append("source_fresh_input_changed")
    result = dict(raw)
    result.update({
        "raw_j2_comparison": raw,
        "raw_j2_hard_gate_errors": raw_errors,
        "effective_hard_gate_errors": effective,
        "hard_gate_errors": effective,
        "hard_gate_pass": not effective,
        "pre2_lineage_exception": {
            "classification": ("EXPECTED_DISTINCT_LINEAGED_PRE2_INPUT" if distinct_pre
                else "SAME_LINEAGED_PRE2_INPUT"),
            "left_input_PRE_fields_canonical_sha256": left_pre,
            "right_input_PRE_fields_canonical_sha256": right_pre,
            "left_verified_NEXT1_field_canonical_sha256": left_next_fields,
            "right_verified_NEXT1_field_canonical_sha256": right_next_fields,
            "left_PRE_matches_own_NEXT1": True,
            "right_PRE_matches_own_NEXT1": True,
            "same_fresh_source_E": source_equal,
            "only_pre_hash_gate_reclassified": bool(allowed_pre_error),
            "other_raw_hard_gates_preserved": True},
    })
    return result


def _run_j3(args) -> int:
    if args.phase != PHASE:
        raise J3Error("J3 coordinator requires explicit --phase J3")
    if any(value is None for value in (args.bench_root, args.input_root, args.vendor_dir)):
        raise J3Error("full J3 requires bench-root, input-root and vendor-dir")
    start_cpu = time.process_time()
    prepared, allocation = _prepare_real_context(args)
    context = prepared["context"]
    candidate_requested = args.candidate_descriptor is not None or args.candidate_admission is not None
    candidate_gate = {"status": "NOT_REQUESTED"}
    if candidate_requested:
        try:
            candidate_gate = _validate_candidate_admission(args)
            if candidate_gate.get("status") == "ADMITTED":
                args.candidate_descriptor_raw_sha256 = candidate_gate["descriptor_raw_sha256"]
        except Exception as exc:
            candidate_gate = {"status": "BLOCKED_INVALID_ADMISSION",
                "error": f"{type(exc).__name__}: {exc}"}
    allocation_start = _allocation_start_epoch()
    allocation_deadline = allocation_start + 7200.0 - 900.0
    work_deadline = min(allocation_start + J3_WORK_LIMIT_SECONDS, allocation_deadline)
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(exist_ok=False)
    p1r._environment_receipt(args.output_root, vendor_dir=args.vendor_dir, allocation=allocation)
    input_summary = {"source_raw_sha256": p1r.SOURCE_RAW_SHA256,
        "source_canonical_sha256": context["source_hash"],
        "source_validation": j1._json_safe(context["source_validation"]),
        "pre0_raw_sha256": p1r.PRE0_RAW_SHA256,
        "pre0_canonical_sha256": context["assets"]["pre0_canonical_sha256"],
        "transfer_mapping_sha256": context["assets"]["mapping_sha256"],
        "schedule": j1._json_safe(context["schedule"].as_metadata()),
        "runtime_config_delta": j1._json_safe(context["runtime_config_delta"]),
        "hydro_parameters": prepared["hydro"]}
    _write_json_new(args.output_root / "input_validation_receipt.json", input_summary)
    _write_json_new(args.output_root / "candidate_admission_receipt.json", candidate_gate)
    del prepared, context
    import gc
    gc.collect()
    predicted_chain = 2 * OPTICAL_WINDOW_ESTIMATE_SECONDS + HYDRO_STAGE_TIMEOUT_SECONDS
    predicted_reload = OPTICAL_WINDOW_ESTIMATE_SECONDS
    baseline = _run_implementation(args, "baseline", args.output_root / "baseline",
        candidate_descriptor=None, deadline=work_deadline)
    candidate = None
    cross_comparison = None
    candidate_status = candidate_gate["status"]
    baseline_latched = baseline.get("rss_80_percent_latched", False)
    if candidate_requested:
        if candidate_gate.get("status") != "ADMITTED":
            candidate_status = "BLOCKED_INVALID_ADMISSION"
        elif baseline_latched:
            candidate_status = "BLOCKED_RSS_80_PERCENT_LATCH"
        elif baseline.get("repeat_exact") is not True:
            candidate_status = "BLOCKED_BASELINE_REPEAT_NOT_EXACT"
        else:
            remaining = work_deadline - time.time()
            predicted_candidate_total = predicted_chain + predicted_reload
            if remaining < max(predicted_candidate_total, J3_TAIL_RESERVE_SECONDS):
                candidate_status = "SKIPPED_INSUFFICIENT_TIME"
            else:
                candidate = _run_implementation(args, "candidate",
                    args.output_root / "candidate",
                    candidate_descriptor=args.candidate_descriptor,
                    deadline=work_deadline)
                candidate_status = candidate["status"]
                if (candidate.get("status") == "COMPLETE_DIAGNOSTIC"
                        and baseline.get("status") == "COMPLETE_DIAGNOSTIC"):
                    from tools.cpu_p1r2 import cpu_p1r2_j2 as j2
                    baseline_chain = next(row for row in baseline["stages"]
                        if row.get("stage") == "chain")
                    candidate_chain = next(row for row in candidate["stages"]
                        if row.get("stage") == "chain")
                    baseline_reload = next(row for row in baseline["stages"]
                        if row.get("stage") == "reload")
                    candidate_reload = next(row for row in candidate["stages"]
                        if row.get("stage") == "reload")
                    cross_comparison = {
                        "pulse1": _cross_pulse1_compare(
                            baseline_chain["child_receipt"]["pulse1_window"],
                            candidate_chain["child_receipt"]["pulse1_window"],
                            "J3_baseline_vs_candidate_pulse1"),
                        "pulse2_chain": _cross_lineaged_pulse2_compare(
                            baseline_chain["child_receipt"]["pulse2_window"],
                            candidate_chain["child_receipt"]["pulse2_window"],
                            baseline_chain["child_receipt"]["NEXT1"]["field_canonical_sha256"],
                            candidate_chain["child_receipt"]["NEXT1"]["field_canonical_sha256"],
                            "J3_baseline_vs_candidate_chain_pulse2"),
                        "pulse2_independent_reload": _cross_lineaged_pulse2_compare(
                            baseline_reload["child_receipt"]["pulse2_window"],
                            candidate_reload["child_receipt"]["pulse2_window"],
                            baseline_reload["child_receipt"]["NEXT1_input"]["field_canonical_sha256"],
                            candidate_reload["child_receipt"]["NEXT1_input"]["field_canonical_sha256"],
                            "J3_baseline_vs_candidate_disk_reload_pulse2"),
                        "NEXT1_fields": _next1_comparison(args, baseline, candidate)}
                    cross_comparison["all_hard_gates_pass"] = all(
                        cross_comparison[key].get("hard_gate_pass") is True
                        for key in ("pulse1", "pulse2_chain", "pulse2_independent_reload"))
                    cross_comparison["nonzero_numerical_differences_remain_diagnostic_only"] = True
                    _write_json_new(args.output_root / "baseline_vs_candidate_comparison.json",
                                    cross_comparison)
                    if not cross_comparison["all_hard_gates_pass"]:
                        candidate_status = "NUMERICAL_DIFFERENCE_HARD_GATE"
                    elif not candidate["repeat_exact"]:
                        candidate_status = "CANDIDATE_REPEAT_NOT_EXACT"
    if candidate_status not in ("ADMITTED", "NOT_REQUESTED") and candidate is None:
        overall_status = "COMPLETE_DIAGNOSTIC_BASELINE_ONLY" if baseline.get("repeat_exact") else baseline.get("status", "INCOMPLETE")
    elif candidate is not None and candidate_status == "COMPLETE_DIAGNOSTIC":
        overall_status = "COMPLETE_DIAGNOSTIC"
    elif candidate is not None and candidate_status == "NUMERICAL_DIFFERENCE_HARD_GATE":
        overall_status = "NUMERICAL_DIFFERENCE"
    elif candidate is not None and candidate_status == "CANDIDATE_REPEAT_NOT_EXACT":
        overall_status = "NUMERICAL_DIFFERENCE"
    elif candidate is not None and candidate_status != "COMPLETE_DIAGNOSTIC":
        overall_status = candidate_status if candidate_status in {
            "NUMERICAL_DIFFERENCE", "LIMIT_REACHED", "INCOMPLETE"} else "INCOMPLETE"
    else:
        overall_status = baseline.get("status", "INCOMPLETE")
    receipt = {"schema": SCHEMA, "phase": PHASE,
        "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
        "status": overall_status,
        "execution_mode": "SERIAL_BASELINE_CHAIN_RELOAD_THEN_ADMITTED_CANDIDATE_CHAIN_RELOAD",
        "allocation": allocation, "allocation_start_epoch": allocation_start,
        "allocation_wall_limit_seconds": 7200, "allocation_reserve_seconds": 900,
        "j3_work_limit_seconds": J3_WORK_LIMIT_SECONDS, "work_deadline_epoch": work_deadline,
        "coordinator_admission_source_loads": 1,
        "coordinator_process_cpu_includes_admission_and_source_validation": True,
        "tail_reserve_seconds": J3_TAIL_RESERVE_SECONDS,
        "stage_order": ["baseline.chain", "baseline.reload", "candidate.chain", "candidate.reload"],
        "baseline": baseline, "candidate": candidate,
        "candidate_admission": candidate_gate, "candidate_status": candidate_status,
        "cross_implementation_comparison": cross_comparison,
        "lineage_accounting": {"per_implementation": {
            "primary_chain": {"fresh_source_loads": 2, "optical_pulses": 2,
                "POST_generations": 2, "hydro_transitions": 1},
            "independent_disk_reload": {"fresh_source_loads": 1, "optical_pulses": 1,
                "POST_generations": 1, "hydro_transitions": 0},
            "actual_optical_windows_if_both_stages_complete": 3,
            "reload_second_pulse_is_a_separate_execution_not_a_third_chain_pulse": True}},
        "input_validation": input_summary,
        "optical_scheduling_model": {
            "recent_J1_baseline_service_seconds": OPTICAL_WINDOW_BASELINE_MEASURED_SECONDS,
            "scheduling_basis_seconds_per_window": 518.0,
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
        "diagnostic_boundary": "P1R2_J3_BASELINE_AND_OPTIONAL_SCIPY_CANDIDATE_DIAGNOSTIC_ONLY",
        "cross_implementation_nonzero_differences_are_not_scientific_acceptance": True}
    _write_json_new(args.output_root / "run_receipt.json", receipt)
    return 0 if overall_status in ("COMPLETE_DIAGNOSTIC", "COMPLETE_DIAGNOSTIC_BASELINE_ONLY") else 1


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
    parser.add_argument("--candidate-descriptor", type=Path)
    parser.add_argument("--candidate-admission", type=Path)
    parser.add_argument("--candidate-descriptor-raw-sha256")
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
