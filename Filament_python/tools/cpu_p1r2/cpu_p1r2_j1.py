#!/usr/bin/env python3
"""Bounded J1 CPU diagnostics: two fresh baseline windows and one profiled window."""
from __future__ import annotations

import argparse
from collections import defaultdict
import functools
import importlib
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import FunctionType

_INITIAL_GPU = os.environ.get("UPPE_USE_GPU")
os.environ["UPPE_USE_GPU"] = "0"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from tools.cpu_p1 import cpu_p1_fft as common
from tools.cpu_p1r import cpu_p1r_optical as p1r
from tools.hpc_ops.provenance_v2 import raw_sha256_file

SCHEMA = "filament.cpu_p1r2.j1_diagnostic.v1"
CLASSIFICATION = "P1R2_J1_DIAGNOSTIC_ONLY"
QUALIFICATION = "NOT_PRODUCTION_QUALIFIED"
PHASE = "J1"
STAGES = ("baseline-a", "baseline-b", "profile")
WINDOW_SECONDS = 2700
B0_SECONDS = 900
ALLOCATION_SECONDS = 7200
RESERVE_SECONDS = 900
RSS_WARN_FRACTION = 0.80
RSS_ABORT_FRACTION = 0.90
PSS_SAMPLE_SECONDS = 10.0
TIMING_DIAGNOSTIC_FIELDS = frozenset({
    "raman_operator_walltime_step_s", "linear_walltime_step_s",
    "ionization_walltime_step_s", "total_walltime_step_s",
})


class J1Error(RuntimeError):
    pass


def _jsonable(value):
    return p1r._jsonable(value)


def _json_new(path: Path, value) -> None:
    p1r._json_new(path, value)


def _append_event(path: Path, value) -> None:
    p1r._append_event(path, value)


def _canonical_hash(array) -> str:
    return p1r._canonical_hash(array)


def _pre0_hash_gate(actual_delta_n_hash: str, expected_hash: str | None, *,
                    initial_pre0: bool, execution_scope: str) -> str | None:
    if not initial_pre0:
        return None
    if expected_hash is None and execution_scope == "J1_REAL_INPUT":
        raise J1Error("real PRE0 execution requires the sealed canonical hash")
    if expected_hash is not None and actual_delta_n_hash != expected_hash:
        raise J1Error("actual PRE0 delta_n canonical hash differs from sealed PRE0")
    return actual_delta_n_hash


def _effective_precision_strategy(prop) -> str:
    strategy = str(getattr(prop, "linear_precision_strategy", "baseline_complex64")
                   or "baseline_complex64").lower()
    if strategy == "unitary_projection":
        raise J1Error("J1 does not admit linear_precision_strategy=unitary_projection")
    return strategy


def _json_safe(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _array_specs(value, prefix="arg", depth=0):
    import numpy as np
    out = []
    if isinstance(value, np.ndarray):
        out.append({"name": prefix, "shape": list(value.shape), "dtype": str(value.dtype)})
    elif depth < 2 and isinstance(value, (tuple, list)):
        for i, item in enumerate(value[:8]):
            out.extend(_array_specs(item, f"{prefix}[{i}]", depth + 1))
    elif depth < 2 and isinstance(value, dict):
        for key in sorted(value)[:8]:
            out.extend(_array_specs(value[key], f"{prefix}.{key}", depth + 1))
    return out


class MemoryPREView:
    """In-memory PRE/POST adapter with the formal read/update/post_fields protocol."""

    def __init__(self, fields, *, state_origin: str, initial_pre0: bool = False):
        import numpy as np
        required = {"delta_n", "vx", "vy"}
        if set(fields) != required:
            raise J1Error("PRE field set must be exactly delta_n/vx/vy")
        copied = {}
        expected_shape = None
        for key in ("delta_n", "vx", "vy"):
            value = np.asarray(fields[key])
            if value.dtype != np.dtype("float64") or value.ndim != 3 or not value.flags.c_contiguous:
                raise J1Error(f"PRE {key} must be C-contiguous float64 [K,Ny,Nx]")
            if not np.isfinite(value).all():
                raise J1Error(f"PRE {key} contains NaN/Inf")
            if expected_shape is None:
                expected_shape = value.shape
            elif value.shape != expected_shape:
                raise J1Error("PRE fields have inconsistent shapes")
            copied[key] = np.array(value, dtype=np.float64, order="C", copy=True)
        if expected_shape[0] != 8:
            raise J1Error("J1 requires exactly eight PRE intervals")
        if initial_pre0:
            for key in ("vx", "vy"):
                if not np.equal(copied[key], 0.0).all() or np.signbit(copied[key]).any():
                    raise J1Error("initial PRE0 velocities must be exact positive float64 zero")
        self.fields = copied
        self.shape = tuple(expected_shape[1:])
        self.state_origin = str(state_origin)
        self.initial_pre0 = bool(initial_pre0)
        self._read = set()
        self._updated = set()
        self._after = {}
        # Hash before propagation; metadata() is called by the unchanged propagator.
        self._field_hashes = {key: _canonical_hash(value) for key, value in copied.items()}

    def read_interval(self, index):
        if index in self._read or not 0 <= int(index) < 8:
            raise ValueError("PRE interval must be read exactly once")
        self._read.add(int(index))
        return self.fields["delta_n"][int(index)].copy()

    def update_interval(self, index, increment):
        import numpy as np
        index = int(index)
        if index not in self._read or index in self._updated:
            raise ValueError("PRE update requires one prior read")
        inc = np.asarray(increment)
        if inc.dtype != np.dtype("float64") or inc.shape != self.shape or not np.isfinite(inc).all():
            raise ValueError("HR-3B increment must be finite float64 with the PRE map shape")
        after = np.asarray(self.fields["delta_n"][index] + inc, dtype=np.float64)
        if not np.isfinite(after).all():
            raise ValueError("POST delta_n contains NaN/Inf")
        self._after[index] = np.array(after, dtype=np.float64, order="C", copy=True)
        self._updated.add(index)
        return self._after[index].copy()

    def post_fields(self, index):
        index = int(index)
        if index not in self._updated:
            raise ValueError("POST requested before PRE update")
        return {"delta_n": self._after[index].copy(),
                "vx": self.fields["vx"][index].copy(),
                "vy": self.fields["vy"][index].copy()}

    @property
    def complete(self):
        return self._read == set(range(8)) and self._updated == self._read

    def final_fields(self):
        import numpy as np
        if not self.complete:
            raise J1Error("PRE/POST view is incomplete")
        return {"delta_n": np.stack([self._after[i] for i in range(8)]),
                "vx": np.array(self.fields["vx"], copy=True),
                "vy": np.array(self.fields["vy"], copy=True)}

    def metadata(self):
        return {"hr3b_state_schema": "filament.cpu_p1r2.j1_diagnostic_pre_memory.v1",
            "hr3b_state_filename": "", "hr3b_state_dtype": "float64",
            "hr3b_state_shape": tuple(self.fields["delta_n"].shape),
            "hr3b_state_interval_centered": True, "hr3b_state_disk_backed": False,
            "schema": "filament.cpu_p1r2.j1_diagnostic_pre_memory.v1",
            "diagnostic_only": True, "state_origin": self.state_origin,
            "storage": "in_memory_copy", "disk_backed": False,
            "interval_centered": True, "shape": list(self.fields["delta_n"].shape),
            "dtype": "float64", "field_canonical_sha256": self._field_hashes}

    def close(self):
        return None


def _last(locals_, key, default=None):
    value = locals_.get(key)
    if isinstance(value, (list, tuple)) and value:
        return value[-1]
    return default


class ScalarBoundaryObserver:
    """Source-bound low-overhead observer at the loop and accepted-z anchors."""

    def __init__(self, code, progress_path: Path, *, b0_limit=B0_SECONDS):
        self.code = code
        self.progress_path = progress_path
        self.b0_limit = float(b0_limit)
        self.starts, self.cpu_starts, self.rows = {}, {}, []
        self.tool_id = None

    def _callback(self, code, line):
        frame = sys._getframe(1)
        while frame is not None and frame.f_code is not code:
            frame = frame.f_back
        if frame is None:
            raise J1Error("sys.monitoring could not recover frozen propagate frame")
        if line == p1r.INTERVAL_START_LINE:
            index = int(frame.f_locals["interval_index"])
            self.starts[index] = time.perf_counter()
            self.cpu_starts[index] = time.process_time()
            if index == 0:
                _append_event(self.progress_path, {"event": "first_interval_started",
                    "interval_index": index, "monotonic": self.starts[index], "utc": time.time()})
            return None
        if line == p1r.INTERVAL_END_LINE:
            boundary = time.perf_counter()
            cpu_boundary = time.process_time()
            loc = frame.f_locals
            index = int(loc["interval_index"])
            if index not in self.starts or index != len(self.rows):
                raise J1Error("accepted-boundary observer saw a duplicate or out-of-order interval")
            row = {"interval_index": index,
                "z_m": float(loc["z"]), "dz_m": float(_last(loc, "dz_used_z_list")),
                "interval_wall_seconds": boundary - self.starts[index],
                "interval_process_cpu_seconds": cpu_boundary - self.cpu_starts[index],
                "U_z_J": float(_last(loc, "U_z_list")),
                "I_max_W_m2": float(_last(loc, "I_max_z_list")),
                "rho_N2_max_m3": float(_last(loc, "rho_N2_max_z_list")),
                "rho_O2_max_m3": float(_last(loc, "rho_O2_max_z_list")),
                "linear_walltime_step_seconds": float(_last(loc, "linear_walltime_step_z_list")),
                "propagation_step_wall_seconds": float(_last(loc, "total_walltime_step_z_list")),
                "adaptive_rejection_count": int(_last(loc, "adaptive_rejection_count_z_list", 0)),
                "safety_mode_trigger_count": int(_last(loc, "safety_mode_trigger_count_z_list", 0)),
                "raman_operator_substep_count": int(_last(loc, "raman_operator_substep_count_z_list", 0)),
                "raman_convolution_count": int(_last(loc, "raman_convolution_count_step_z_list", 0)),
                "boundary_monotonic": boundary, "boundary_utc": time.time()}
            if index == 0:
                row["B0_status"] = "DONE" if row["interval_wall_seconds"] <= self.b0_limit else "LIMIT_REACHED"
            row["observer_callback_output_seconds"] = time.perf_counter() - boundary
            self.rows.append(row)
            _append_event(self.progress_path, {"event": "complete_interval", **row})
            if index == 0 and row["B0_status"] != "DONE":
                raise TimeoutError("J1 first complete interval exceeded 900 seconds")
            return None
        return sys.monitoring.DISABLE

    def __enter__(self):
        mon = sys.monitoring
        for candidate in range(6):
            if mon.get_tool(candidate) is None:
                self.tool_id = candidate
                break
        if self.tool_id is None:
            raise J1Error("no free sys.monitoring tool ID")
        mon.use_tool_id(self.tool_id, "filament_cpu_p1r2_j1")
        try:
            mon.register_callback(self.tool_id, mon.events.LINE, self._callback)
            mon.set_local_events(self.tool_id, self.code, mon.events.LINE)
        except BaseException:
            self._cleanup()
            raise
        return self

    def _cleanup(self):
        if self.tool_id is None:
            return
        mon = sys.monitoring
        tool = self.tool_id
        try:
            mon.set_local_events(tool, self.code, 0)
            mon.set_events(tool, 0)
            mon.register_callback(tool, mon.events.LINE, None)
            mon.restart_events()
        finally:
            mon.free_tool_id(tool)
            self.tool_id = None

    def __exit__(self, exc_type, exc, tb):
        self._cleanup()
        return False


class _FunctionProfile:
    """Inclusive/exclusive timer for explicitly dispatched scientific functions."""
    def __init__(self):
        self.stack = []
        self.rows = defaultdict(lambda: {"calls": 0, "inclusive_wall_seconds": 0.0,
            "exclusive_wall_seconds": 0.0, "inclusive_process_cpu_seconds": 0.0,
            "exclusive_process_cpu_seconds": 0.0, "failures": 0})

    def wrap(self, label, function):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            caller = inspect.currentframe().f_back
            callsite = f"{caller.f_code.co_filename}:{caller.f_lineno}:{caller.f_code.co_name}"
            specs = _array_specs(args) + _array_specs(kwargs, "kwargs")
            axes = kwargs.get("axes")
            axis_value = kwargs.get("axis")
            if axis_value is None and isinstance(axes, (int, tuple, list)):
                axis_value = axes
            scalar_kwargs = {key: kwargs[key] for key in ("method", "iir_sampling")
                             if isinstance(kwargs.get(key), (str, int, float, bool))}
            signature = json.dumps({"function": label, "callsite": callsite,
                "arrays": specs, "axis": _jsonable(axis_value),
                "scalar_kwargs": scalar_kwargs,
                "axes_shape": (list(axes.kperp2.shape) if hasattr(axes, "kperp2") else None)},
                sort_keys=True, separators=(",", ":"))
            started_wall, started_cpu = time.perf_counter(), time.process_time()
            frame = {"child_wall": 0.0, "child_cpu": 0.0}
            self.stack.append(frame)
            failed = False
            try:
                return function(*args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                wall = time.perf_counter() - started_wall
                cpu = time.process_time() - started_cpu
                self.stack.pop()
                if self.stack:
                    self.stack[-1]["child_wall"] += wall
                    self.stack[-1]["child_cpu"] += cpu
                row = self.rows[signature]
                row["function"] = label
                row["callsite"] = callsite
                row["array_shapes_dtypes"] = specs
                row["axes"] = _jsonable(axis_value)
                row["scalar_kwargs"] = scalar_kwargs
                row["calls"] += 1
                row["inclusive_wall_seconds"] += wall
                row["exclusive_wall_seconds"] += max(0.0, wall - frame["child_wall"])
                row["inclusive_process_cpu_seconds"] += cpu
                row["exclusive_process_cpu_seconds"] += max(0.0, cpu - frame["child_cpu"])
                row["failures"] += int(failed)
        measured.__j1_inner_code__ = function.__code__
        return measured

    def record_fft(self, name, function, args, kwargs):
        @functools.wraps(function)
        def measured(*call_args, **call_kwargs):
            caller = inspect.currentframe().f_back
            callsite = f"{caller.f_code.co_filename}:{caller.f_lineno}:{caller.f_code.co_name}"
            specs = _array_specs(call_args)
            axes = call_kwargs.get("axis", call_kwargs.get("axes"))
            transform_names = {"fft", "ifft", "fft2", "ifft2", "fftn", "ifftn",
                "rfft", "irfft", "rfft2", "irfft2", "rfftn", "irfftn"}
            category = "fft_transform" if name in transform_names else "fft_aux"
            label = f"{category}.{name}"
            key = json.dumps({"function": label, "callsite": callsite,
                "arrays": specs, "axes": _jsonable(axes),
                "n": _jsonable(call_kwargs.get("n")),
                "norm": call_kwargs.get("norm")}, sort_keys=True, separators=(",", ":"))
            started_wall, started_cpu = time.perf_counter(), time.process_time()
            frame = {"child_wall": 0.0, "child_cpu": 0.0}
            self.stack.append(frame)
            failed = False
            try:
                return function(*call_args, **call_kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                wall, cpu = time.perf_counter() - started_wall, time.process_time() - started_cpu
                self.stack.pop()
                if self.stack:
                    self.stack[-1]["child_wall"] += wall
                    self.stack[-1]["child_cpu"] += cpu
                row = self.rows[key]
                row["function"] = label
                row["callsite"] = callsite
                row["array_shapes_dtypes"] = specs
                row["axes"] = _jsonable(axes)
                row["n"] = _jsonable(call_kwargs.get("n"))
                row["norm"] = call_kwargs.get("norm")
                row["calls"] += 1
                row["inclusive_wall_seconds"] += wall
                row["exclusive_wall_seconds"] += max(0.0, wall - frame["child_wall"])
                row["inclusive_process_cpu_seconds"] += cpu
                row["exclusive_process_cpu_seconds"] += max(0.0, cpu - frame["child_cpu"])
                row["failures"] += int(failed)
        return measured

    def report(self):
        rows = [dict(value) for value in self.rows.values()]
        rows.sort(key=lambda row: (row["function"], row["callsite"],
                                  json.dumps(row["array_shapes_dtypes"], sort_keys=True)))
        return {"schema": "filament.cpu_p1r2.j1_function_profile.v1",
                "mode": "PRIVATE_FUNCTION_NAMESPACE_AND_XP_FFT_PROXY",
                "global_numpy_fft_modified": False,
                "global_linear_raman_xp_modified": False,
                "records": rows,
                "unattributed_propagator_time": "UNATTRIBUTED",
                "fft_call_count": sum(r["calls"] for r in rows if r["function"].startswith("fft_transform.")),
                "fft_inclusive_wall_seconds": sum(r["inclusive_wall_seconds"] for r in rows if r["function"].startswith("fft_transform.")),
                "fft_auxiliary_call_count": sum(r["calls"] for r in rows if r["function"].startswith("fft_aux."))}


class _FFTProxy:
    def __init__(self, module, collector):
        self._module, self._collector = module, collector

    def __getattr__(self, name):
        value = getattr(self._module, name)
        if callable(value):
            return self._collector.record_fft(name, value, (), {})
        return value


class _XPProxy:
    def __init__(self, xp, collector):
        self._xp = xp
        self.fft = _FFTProxy(xp.fft, collector)

    def __getattr__(self, name):
        if name == "fft":
            return self.fft
        return getattr(self._xp, name)


def _clone_module_functions(module, proxy, profiler, selected):
    namespace = dict(vars(module))
    namespace["xp"] = proxy
    clones = {}
    for name, value in vars(module).items():
        if inspect.isfunction(value) and value.__module__ == module.__name__:
            clones[name] = FunctionType(value.__code__, namespace, value.__name__,
                value.__defaults__, value.__closure__)
            clones[name].__kwdefaults__ = value.__kwdefaults__
            clones[name].__annotations__ = value.__annotations__
            clones[name].__module__ = value.__module__
            clones[name].__qualname__ = value.__qualname__
    for name, function in tuple(clones.items()):
        if name in selected:
            clones[name] = profiler.wrap(f"{module.__name__}.{name}", function)
    namespace.update(clones)
    return namespace


def _profiled_propagator():
    """Clone only dispatch namespaces; production module globals stay untouched."""
    linear = importlib.import_module("KHz_filament.linear")
    raman = importlib.import_module("KHz_filament.raman")
    runtime = importlib.import_module("KHz_filament.ionization.runtime")
    propagate = importlib.import_module("KHz_filament.propagate")
    from KHz_filament.device import xp
    profiler = _FunctionProfile()
    proxy = _XPProxy(xp, profiler)
    linear_ns = _clone_module_functions(linear, proxy, profiler,
        {"step_linear_bk_nee_factorized", "step_linear", "lin_propagator"})
    raman_selected = {"isaacs_complete_eq27_stage", "apply_isaacs_complete_eq27_operator_step",
        "isaacs_complete_eq27_field_rhs", "isaacs_raman_stage",
        "apply_isaacs_raman_operator_step", "raman_convolve_intensity",
        "raman_convolve_intensity_fft_linear", "precompute_kernel_fft", "make_raman_kernel"}
    raman_ns = _clone_module_functions(raman, proxy, profiler, raman_selected)
    runtime_ns = _clone_module_functions(runtime, proxy, profiler,
        {"make_Wfunc", "evolve_rho_time"})
    runtime_ns["eval_rate_from_table"] = profiler.wrap(
        "KHz_filament.ionization.lut.eval_rate_from_table", runtime.eval_rate_from_table)
    source = propagate.propagate_one_pulse
    private = dict(source.__globals__)
    replacements = {
        "step_linear_bk_nee_factorized": linear_ns["step_linear_bk_nee_factorized"],
        "step_linear": linear_ns["step_linear"], "lin_propagator": linear_ns["lin_propagator"],
        "make_Wfunc": runtime_ns["make_Wfunc"],
        "evolve_rho_time": runtime_ns["evolve_rho_time"],
    }
    for name in ("apply_isaacs_raman_operator_step", "apply_isaacs_complete_eq27_operator_step",
                 "historical_fr_mixture_response", "isaacs_raman_stage",
                 "isaacs_complete_eq27_stage", "make_raman_kernel",
                 "precompute_kernel_fft", "raman_convolve_intensity",
                 "resolve_raman_rot_params"):
        if name in raman_ns:
            replacements[name] = raman_ns[name]
    for name in ("apply_nonlinear", "build_unified_deposition_ledger",
                 "thermalize_interval", "map_post_acoustic_increment", "heat_Q_per_z",
                 "direct_interval_energy", "interval_energy_from_fluence_gain",
                 "interval_energy_from_q", "intensity", "pulse_energy",
                 "second_moment_radius", "_fwhm_time_1d",
                 "_fwhm_diameter_xy_center", "shock_intensity"):
        function = private.get(name)
        if inspect.isfunction(function):
            replacements[name] = profiler.wrap(f"{function.__module__}.{name}", function)
    private.update(replacements)
    clone = FunctionType(source.__code__, private, source.__name__, source.__defaults__, source.__closure__)
    clone.__kwdefaults__ = source.__kwdefaults__
    clone.__annotations__ = source.__annotations__
    clone.__module__ = source.__module__
    clone.__qualname__ = source.__qualname__
    measured = profiler.wrap("KHz_filament.propagate.propagate_one_pulse", clone)
    return measured, profiler


def _assert_cpu_backend():
    import numpy as np
    from KHz_filament.device import USE_GPU, xp
    if bool(USE_GPU) or xp is not np:
        raise J1Error("J1 requires the actual NumPy CPU backend")


def _save_array(path: Path, array) -> None:
    import io
    import numpy as np
    payload = io.BytesIO()
    np.save(payload, np.asarray(array), allow_pickle=False)
    p1r._write_new(path, payload.getvalue())


def _save_npz(path: Path, fields) -> None:
    p1r._write_new(path, p1r._npz_bytes(fields))


def _run_window(*, source, pre_fields, grid, beam, prop, ion, heat, raman,
                schedule, output_dir: Path, source_label: str,
                initial_pre0: bool, profile: bool, source_hash: str | None = None,
                pre0_hash: str | None = None, stage_name: str = "tiny",
                execution_scope: str = "J1_REAL_INPUT", observe: bool = True) -> dict:
    """Execute exactly eight frozen schedule intervals on a fresh source/PRE copy."""
    import numpy as np
    from contextlib import nullcontext
    from KHz_filament.constants import N0_air, Ui_N2, c0, n2_air
    from KHz_filament.device import xp
    from KHz_filament.grids import make_axes
    from KHz_filament.longitudinal import build_deposition_contract
    from KHz_filament.slow_state import HR3BDiagnosticSink, validate_hr3b_parameters
    from KHz_filament.thermalization import ThermalDiagnosticSink
    from KHz_filament.hr4e5_formal_entry import _all_interval_sample_plan
    from KHz_filament.hr4e5_evidence import sha256_array
    from KHz_filament import propagate as propagate_module

    if schedule.n_intervals != 8:
        raise J1Error("J1 propagator requires an exact complete eight-interval schedule")
    if xp is not np:
        raise J1Error("J1 propagation backend changed from NumPy")
    if not bool(getattr(prop, "measure_performance", False)) or not bool(getattr(prop, "diag_bk_nee_profile", False)):
        raise J1Error("all J1 stages must retain measure_performance and diag_bk_nee_profile")
    precision_strategy = _effective_precision_strategy(prop)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    original_source = np.asarray(source)
    calculated_source_hash = _validate_source_array(original_source, expected_hash=source_hash)
    E = np.array(original_source, dtype=np.complex128, order="C", copy=True)
    if np.shares_memory(E, original_source):
        raise J1Error("fresh J1 source aliases the frozen source")
    fresh_source_hash = str(sha256_array(E))
    if fresh_source_hash != calculated_source_hash:
        raise J1Error("fresh source copy differs from the frozen canonical source")

    view = MemoryPREView(pre_fields, state_origin=source_label, initial_pre0=initial_pre0)
    input_pre_hashes = dict(view._field_hashes)
    actual_pre0_hash = _pre0_hash_gate(input_pre_hashes["delta_n"], pre0_hash,
        initial_pre0=initial_pre0, execution_scope=execution_scope)
    if tuple(view.fields["delta_n"].shape[1:]) != (int(grid.Ny), int(grid.Nx)):
        raise J1Error("PRE maps do not match the frozen transverse grid")
    if view.fields["delta_n"].shape[0] != 8:
        raise J1Error("PRE input does not contain exactly eight intervals")
    axes = make_axes(grid.Nx, grid.Ny, grid.Nt, grid.Lx, grid.Ly, grid.Twin)
    omega0 = 2.0 * xp.pi * c0 / float(beam.lam0)
    k0 = float(beam.n0) * float(omega0) / c0
    n2_used = float(getattr(prop, "n2", getattr(beam, "n2_air", n2_air)))
    beta = validate_hr3b_parameters(rho0=float(heat.rho0), Cv=float(heat.Cv),
        T0=float(prop.air_T), n0=float(beam.n0))
    hr3b_parameters = {"rho0": float(heat.rho0), "Cv": float(heat.Cv),
        "T0": float(prop.air_T), "n0": float(beam.n0), "beta_th": float(beta)}
    plan = _all_interval_sample_plan(schedule)
    thermal_sink = ThermalDiagnosticSink(plan=plan, output_path=str(output_dir / "sinks"),
        shape=(grid.Ny, grid.Nx), dtype=np.float64, enabled=True, mode="validation")
    hr3b_sink = HR3BDiagnosticSink(plan=plan, output_path=str(output_dir / "sinks"),
        shape=(grid.Ny, grid.Nx), dtype=np.float64, enabled=True)
    if profile:
        propagator, profiler = _profiled_propagator()
    else:
        propagator, profiler = propagate_module.propagate_one_pulse, None
    observer_code = getattr(propagator, "__j1_inner_code__", propagator.__code__)
    observer = (ScalarBoundaryObserver(observer_code, output_dir / "observer_progress.jsonl")
                if observe else None)
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    try:
        with (observer if observer is not None else nullcontext()):
            try:
                final_E, _, diag = propagator(
                    E, kperp2=axes.kperp2, k0=k0, omega0=float(omega0),
                    dz=float(schedule.dz_intervals[0]),
                    z_max=float(schedule.z_end - schedule.z_start),
                    n0=float(beam.n0), n2=n2_used, Ui=Ui_N2, N0=N0_air,
                    ion_conf=ion, dn_gas=None, dt=axes.dt, axes=axes, prop_conf=prop,
                    raman_conf=raman, record_onaxis_rho_time=True, record_every_z=1,
                    longitudinal_schedule=schedule,
                    deposition_contract=build_deposition_contract(schedule, axes=axes),
                    thermal_sink=thermal_sink, thermal_slow_state=view,
                    hr3b_parameters=hr3b_parameters, hr3b_sink=hr3b_sink)
                # Capture the scientific-call clocks immediately at return.
                propagation_wall = time.perf_counter() - start_wall
                propagation_cpu = time.process_time() - start_cpu
            except BaseException:
                raise
    except BaseException:
        try: thermal_sink.finalize()
        except Exception: pass
        try: hr3b_sink.finalize()
        except Exception: pass
        view.close()
        raise
    finally:
        view.close()

    post_call_started = time.perf_counter()
    if not view.complete or (observe and len(observer.rows) != 8):
        raise J1Error("J1 propagation did not read/update/observe all eight intervals")
    if len(diag.get("z_axis", ())) != 8 or not np.allclose(
            np.asarray(diag["z_axis"], dtype=np.float64), np.asarray(schedule.z_edges[1:]),
            rtol=0.0, atol=1e-12):
        raise J1Error("propagation returned an incomplete or mismatched eight-interval z history")
    closure = p1r._diagnostic_closure(diag)
    p1r.verify_array_contract(final_E, shape=original_source.shape, dtype="complex128", label="J1 final E")
    _save_array(output_dir / "final_E.npy", final_E)
    final_post = view.final_fields()
    for key, value in final_post.items():
        p1r.verify_array_contract(value, shape=(8, grid.Ny, grid.Nx), dtype="float64",
                                  label=f"J1 final POST {key}")
    _save_npz(output_dir / "final_POST.npz", final_post)
    numeric = {str(key): np.asarray(value) for key, value in diag.items()
        if isinstance(value, (np.ndarray, list, tuple))
        and np.asarray(value).dtype.kind in "biufc"}
    _save_npz(output_dir / "diagnostics.npz", numeric)
    diag_hashes = {key: _canonical_hash(value) for key, value in numeric.items()}
    profile_report = profiler.report() if profiler is not None else None
    if profile_report is not None:
        _json_new(output_dir / "function_fft_profile.json", profile_report)
    _json_new(output_dir / "linear_halfstep_profiles.json", {
        "halfstep1_json": _json_safe(diag.get("bk_nee_profile_halfstep_1_json", ())),
        "halfstep2_json": _json_safe(diag.get("bk_nee_profile_halfstep_2_json", ())),
        "common_instrumentation": {"measure_performance": True,
            "diag_bk_nee_profile": True}})
    final_hash = str(sha256_array(np.asarray(final_E)))
    receipt = {"schema": SCHEMA, "phase": PHASE, "stage": stage_name,
        "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
        "execution_scope": execution_scope,
        "status": "COMPLETE_DIAGNOSTIC", "interval_count": 8,
        "input_fresh_copy_canonical_sha256": fresh_source_hash,
        "source_canonical_sha256": source_hash or fresh_source_hash,
        "effective_linear_precision_strategy": precision_strategy,
        "pre0_canonical_sha256": actual_pre0_hash,
        "pre0_expected_canonical_sha256": pre0_hash,
        "input_PRE_fields_canonical_sha256": input_pre_hashes,
        "final_E_canonical_sha256": final_hash,
        "final_E_raw_sha256": raw_sha256_file(output_dir / "final_E.npy"),
        "final_POST_canonical_sha256": {k: _canonical_hash(v) for k, v in final_post.items()},
        "final_POST_raw_sha256": raw_sha256_file(output_dir / "final_POST.npz"),
        "diagnostics_canonical_sha256": diag_hashes,
        "diagnostics_raw_sha256": raw_sha256_file(output_dir / "diagnostics.npz"),
        "interval_rows": observer.rows if observer is not None else [],
        "observer_enabled": bool(observe), "PRE_complete": view.complete,
        "PRE_metadata": view.metadata(), "diagnostic_closure": closure,
        "propagation_wall_seconds": propagation_wall,
        "propagation_process_cpu_seconds": propagation_cpu,
        "validation_output_seconds": time.perf_counter() - post_call_started,
        "runtime_instrumentation": {"measure_performance": True,
            "diag_bk_nee_profile": True,
            "additional_profile_enabled": bool(profile),
            "additional_profile_attribution": "UNATTRIBUTED"},
        "profile": profile_report,
        "observables": {key: _json_safe(diag.get(key)) for key in
            ("z_axis", "U_z", "I_max_z", "rho_N2_max_z", "rho_O2_max_z",
             "dz_used_z", "adaptive_rejection_count_z", "safety_mode_trigger_count_z")
            if key in diag}}
    _json_new(output_dir / "child_receipt.json", receipt)
    return receipt


def _proc_tree_pids(root_pid: int):
    proc = Path("/proc")
    entries = {}
    if not proc.is_dir():
        return None
    for path in proc.iterdir():
        if not path.name.isdigit():
            continue
        try:
            status = (path / "status").read_text(encoding="ascii", errors="ignore")
            ppid = int(next(line.split()[1] for line in status.splitlines() if line.startswith("PPid:")))
            entries[int(path.name)] = ppid
        except (OSError, StopIteration, ValueError, IndexError):
            continue
    pids, changed = {int(root_pid)}, True
    while changed:
        changed = False
        for child, parent in entries.items():
            if parent in pids and child not in pids:
                pids.add(child)
                changed = True
    return sorted(pids)


def _pss_tree_snapshot(root_pid: int):
    pids = _proc_tree_pids(root_pid)
    if pids is None:
        return {"pss_bytes": None, "process_count": None, "source": "UNAVAILABLE"}
    total = 0
    available = 0
    for pid in pids:
        try:
            lines = (Path("/proc") / str(pid) / "smaps_rollup").read_text(
                encoding="ascii", errors="ignore").splitlines()
            value = next(int(line.split()[1]) for line in lines if line.startswith("Pss:"))
            total += value * 1024
            available += 1
        except (OSError, StopIteration, ValueError, IndexError):
            continue
    return {"pss_bytes": total if available else None,
            "process_count": len(pids), "pss_processes_read": available,
            "source": "/proc/*/smaps_rollup"}


def _rss_action(rss_bytes, memory_bytes):
    if rss_bytes is None or memory_bytes is None or memory_bytes <= 0:
        return {"warn_80": False, "abort_90": False, "fraction": None}
    fraction = float(rss_bytes) / float(memory_bytes)
    return {"warn_80": fraction >= RSS_WARN_FRACTION,
            "abort_90": fraction >= RSS_ABORT_FRACTION,
            "fraction": fraction}


def _stage_block_reason(memory_latched: bool, previous_stages: list[dict],
                        baseline_repeat: dict | None = None) -> str | None:
    if memory_latched:
        return "BLOCKED_RSS_80_PERCENT_LATCH"
    if baseline_repeat is not None and not baseline_repeat.get("canonical_exact_match", False):
        return "BLOCKED_BASELINE_REPEAT_DIFFERENCE"
    if any(row.get("status") == "SKIPPED_INSUFFICIENT_TIME" for row in previous_stages):
        return "SKIPPED_INSUFFICIENT_TIME"
    if any(row.get("status") != "DONE" for row in previous_stages):
        return "BLOCKED_AFTER_PRIOR_STAGE_FAILURE"
    return None


def _terminate_child_bounded(proc, *, grace_seconds=5.0) -> dict:
    if proc.poll() is not None:
        return {"status": "ALREADY_EXITED", "exit_code": proc.returncode}
    method = "process_group_term_then_kill" if os.name != "nt" else "process_term_then_kill"
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        try:
            proc.wait(timeout=float(grace_seconds))
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait(timeout=5.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "STOP_UNRESOLVED", "method": method,
                "error": f"{type(exc).__name__}: {exc}"}
    return {"status": "STOPPED", "method": method, "exit_code": proc.returncode}


def _spawn_and_monitor(command, *, stage_dir: Path, cwd: Path, env: dict,
                       timeout_seconds: float, memory_bytes: int | None,
                       poll_seconds=0.5) -> dict:
    """Run a child with file-backed logs, bounded stop, RSS polling and low-rate PSS."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = stage_dir / "stdout.log", stage_dir / "stderr.log"
    peak_rss = 0
    peak_sampled_tree_cpu = 0.0
    pss_samples = []
    warn_latched = False
    last_pss = 0.0
    started = time.monotonic()
    kwargs = {"cwd": str(cwd), "env": env, "stdout": stdout_path.open("xb"),
              "stderr": stderr_path.open("xb")}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(list(command), **kwargs)
    except BaseException:
        kwargs["stdout"].close(); kwargs["stderr"].close()
        raise
    kwargs["stdout"].close(); kwargs["stderr"].close()
    stop = None
    while proc.poll() is None:
        snapshot = common.tree_snapshot(proc.pid)
        rss = int(snapshot["rss"]) if snapshot is not None else None
        if rss is not None:
            peak_rss = max(peak_rss, rss)
            peak_sampled_tree_cpu = max(peak_sampled_tree_cpu, float(snapshot.get("cpu", 0.0)))
        policy = _rss_action(rss, memory_bytes)
        if policy["warn_80"]:
            warn_latched = True
        now = time.monotonic()
        if now - last_pss >= PSS_SAMPLE_SECONDS:
            pss_samples.append({"monotonic_elapsed_seconds": now - started,
                                **_pss_tree_snapshot(proc.pid)})
            last_pss = now
        if policy["abort_90"]:
            stop = {"reason": "RSS_90_PERCENT", **_terminate_child_bounded(proc)}
            break
        if now - started >= float(timeout_seconds):
            stop = {"reason": "STAGE_TIMEOUT", **_terminate_child_bounded(proc)}
            break
        time.sleep(float(poll_seconds))
    try:
        exit_code = proc.wait(timeout=7.0)
    except subprocess.TimeoutExpired:
        stop = {"reason": "POST_STOP_WAIT_EXPIRED", **_terminate_child_bounded(proc)}
        try:
            exit_code = proc.wait(timeout=7.0)
        except subprocess.TimeoutExpired:
            exit_code = None
    final_tree = common.tree_snapshot(proc.pid)
    if final_tree is not None:
        peak_rss = max(peak_rss, int(final_tree["rss"]))
        peak_sampled_tree_cpu = max(peak_sampled_tree_cpu, float(final_tree.get("cpu", 0.0)))
    pss_samples.append({"monotonic_elapsed_seconds": time.monotonic() - started,
                        **_pss_tree_snapshot(proc.pid)})
    elapsed = time.monotonic() - started
    result = {"exit_code": exit_code, "elapsed_wall_seconds": elapsed,
        "timed_out": bool(stop and stop.get("reason") == "STAGE_TIMEOUT"),
        "stop": stop, "peak_process_tree_rss_bytes": peak_rss or None,
        "peak_sampled_process_tree_cpu_seconds_not_cumulative": peak_sampled_tree_cpu,
        "rss_80_percent_latched": warn_latched,
        "rss_80_fraction": RSS_WARN_FRACTION, "rss_90_fraction": RSS_ABORT_FRACTION,
        "process_tree_pss_samples": pss_samples,
        "pss_sample_interval_seconds": PSS_SAMPLE_SECONDS,
        "rss_poll_interval_seconds": float(poll_seconds),
        "rss_pss_method": "RSS summed from /proc process tree; PSS summed from smaps_rollup at 10s intervals and stage end"}
    _json_new(stage_dir / "supervisor_receipt.json", result)
    return result


def _stage_child_command(args, stage_name: str, child_output: Path):
    return [sys.executable, "-m", "tools.cpu_p1r2.cpu_p1r2_j1",
        "--phase", PHASE, "--bench-root", str(args.bench_root),
        "--input-root", str(args.input_root), "--vendor-dir", str(args.vendor_dir),
        "--output-root", str(child_output), "--child-stage", stage_name]


def _child_environment(bench_root: Path):
    env = os.environ.copy()
    root = str((bench_root / "Filament_python").resolve())
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONNOUSERSITE"] = "1"
    env["UPPE_USE_GPU"] = "0"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    return env


def _validate_stage(stage_dir: Path) -> dict:
    import numpy as np
    child_dir = stage_dir / "propagation"
    receipt = json.loads((child_dir / "child_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("phase") != PHASE or receipt.get("status") != "COMPLETE_DIAGNOSTIC" or receipt.get("interval_count") != 8:
        raise J1Error("child receipt does not certify a complete eight-interval J1 window")
    provenance_path = child_dir / "child_provenance_receipt.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    production_identity_ok = (provenance.get("source_raw_sha256") == p1r.SOURCE_RAW_SHA256
        and provenance.get("pre0_raw_sha256") == p1r.PRE0_RAW_SHA256
        and provenance.get("transfer_mapping_sha256") == p1r.MAPPING_SHA256)
    smoke_identity_ok = receipt.get("execution_scope") == "SOFTWARE_SMOKE_ONLY"
    if (provenance.get("phase") != PHASE
            or provenance.get("source_canonical_sha256") != receipt.get("source_canonical_sha256")
            or provenance.get("pre0_canonical_sha256") != receipt.get("pre0_canonical_sha256")
            or not (production_identity_ok or smoke_identity_ok)):
        raise J1Error("J1 provenance supplement disagrees with the frozen source/PRE0 transfer identity")
    final_path = child_dir / "final_E.npy"
    final = np.load(final_path, mmap_mode="r", allow_pickle=False)
    final_hash = p1r.verify_array_contract(final, shape=final.shape,
        dtype="complex128", label="J1 parent final E")
    final_raw = raw_sha256_file(final_path)
    if final_hash != receipt["final_E_canonical_sha256"] or final_raw != receipt["final_E_raw_sha256"]:
        raise J1Error("J1 final E failed parent raw/canonical verification")
    post_path = child_dir / "final_POST.npz"
    post_hashes = {}
    with np.load(post_path, allow_pickle=False) as archive:
        if set(archive.files) != {"delta_n", "vx", "vy"}:
            raise J1Error("J1 endpoint POST is missing required fields")
        for name in archive.files:
            value = archive[name]
            post_hashes[name] = p1r.verify_array_contract(value,
                shape=(8, *final.shape[-2:]), dtype="float64", label=f"J1 POST {name}")
    post_raw = raw_sha256_file(post_path)
    if post_hashes != receipt["final_POST_canonical_sha256"] or post_raw != receipt["final_POST_raw_sha256"]:
        raise J1Error("J1 final POST failed parent raw/canonical verification")
    diag_path = child_dir / "diagnostics.npz"
    diag_hashes = {}
    with np.load(diag_path, allow_pickle=False) as archive:
        for name in archive.files:
            diag_hashes[name] = _canonical_hash(archive[name])
    diag_raw = raw_sha256_file(diag_path)
    if diag_hashes != receipt["diagnostics_canonical_sha256"] or diag_raw != receipt["diagnostics_raw_sha256"]:
        raise J1Error("J1 diagnostics failed parent raw/canonical verification")
    rows = receipt.get("interval_rows", [])
    if len(rows) != 8 or [int(row["interval_index"]) for row in rows] != list(range(8)):
        raise J1Error("J1 scalar observer missed one or more accepted intervals")
    return {"status": "VALIDATED", "receipt": receipt, "provenance": provenance,
        "final_E_path": str(final_path), "final_E_raw_sha256": final_raw,
        "final_E_canonical_sha256": final_hash,
        "final_POST_path": str(post_path), "final_POST_raw_sha256": post_raw,
        "final_POST_canonical_sha256": post_hashes,
        "diagnostics_path": str(diag_path), "diagnostics_raw_sha256": diag_raw,
        "diagnostics_canonical_sha256": diag_hashes}


def _compare_stage_files(reference: dict, candidate: dict, label: str) -> dict:
    import numpy as np
    comparisons = {}
    equal = True
    for key, left_path, right_path in (
        ("final_E", reference["final_E_path"], candidate["final_E_path"]),
        ("final_POST", reference["final_POST_path"], candidate["final_POST_path"]),
        ("diagnostics", reference["diagnostics_path"], candidate["diagnostics_path"])):
        if key == "final_E":
            left = np.load(left_path, mmap_mode="r", allow_pickle=False)
            right = np.load(right_path, mmap_mode="r", allow_pickle=False)
            left_items, right_items = {"E": left}, {"E": right}
        else:
            with np.load(left_path, allow_pickle=False) as archive:
                left_items = {name: archive[name].copy() for name in archive.files}
            with np.load(right_path, allow_pickle=False) as archive:
                right_items = {name: archive[name].copy() for name in archive.files}
        if set(left_items) != set(right_items):
            comparisons[key] = {"same_fields": False, "left_fields": sorted(left_items),
                "right_fields": sorted(right_items)}
            equal = False
            continue
        component = {}
        for name in sorted(left_items):
            a, b = left_items[name], right_items[name]
            hash_a, hash_b = _canonical_hash(a), _canonical_hash(b)
            item = {"left_canonical_sha256": hash_a, "right_canonical_sha256": hash_b,
                "canonical_equal": hash_a == hash_b}
            if key == "diagnostics" and name in TIMING_DIAGNOSTIC_FIELDS:
                item["comparison_scope"] = "TIMING_ONLY_EXCLUDED_FROM_NUMERICAL_EQUALITY"
                component[name] = item
                continue
            if hash_a != hash_b:
                equal = False
                try:
                    item["error_metrics"] = p1r._array_metrics(a, b)
                except Exception as exc:
                    item["error_metrics"] = {"unavailable": f"{type(exc).__name__}: {exc}"}
            component[name] = item
        comparisons[key] = component
    return {"label": label, "canonical_exact_match": equal,
        "classification": "EXACT_DIAGNOSTIC_MATCH" if equal else "NUMERICAL_DIFFERENCE",
        "error_metrics": comparisons}


def _validate_source_array(source, *, expected_hash=None):
    import numpy as np
    value = np.asarray(source)
    if (value.dtype != np.dtype("complex128") or value.ndim != 3
            or not value.flags.c_contiguous or not np.isfinite(value).all()):
        raise J1Error("J1 source must be finite C-contiguous complex128 [Nt,Ny,Nx]")
    digest = _canonical_hash(value)
    if expected_hash is not None and digest != expected_hash:
        raise J1Error("J1 source canonical hash mismatch")
    return digest


def _require_eight_schedule(schedule):
    if int(getattr(schedule, "n_intervals", -1)) != 8:
        raise J1Error("J1 requires exactly eight complete schedule intervals")
    return schedule


def _tiny_context(*, operator_mode: str = "full_isaacs_eq27"):
    """Deterministic full-operator fixture used only by software smoke/tests."""
    import numpy as np
    from KHz_filament.config import (BeamConfig, GridConfig, HeatConfig,
        IonizationConfig, PropagationConfig, RamanConfig)
    from KHz_filament.grids import make_axes
    from KHz_filament.longitudinal import build_longitudinal_schedule
    from KHz_filament.hr4e5_formal_entry import pre0_fields_from_delta_n
    from KHz_filament.runner import build_transverse_input_field
    from KHz_filament.hr4e5_formal_entry import build_prefix_schedule

    grid = GridConfig(Nx=8, Ny=8, Nt=64, Lx=8e-4, Ly=8e-4, Twin=640e-15)
    beam = BeamConfig(w0=1.5e-4, tau_fwhm=120e-15, E0_peak=1.2e9,
        energy_J=None, P0_peak=None, focal_length=None)
    dz = 1e-5
    prop = PropagationConfig(z_max=8 * dz, dz=dz, linear_model="bk_nee",
        auto_substep=False, focus_window_step=False, limit_focus_window=False,
        progress_every_z=0, show_eta=False, energy_probe_every=0, diag_extra=False,
        use_self_steepening=True, use_electronic_kerr=True, use_raman_phase=False,
        use_raman_full_operator=True, use_raman_absorption=False,
        use_plasma_phase=True, use_ionization_loss=True, use_ionization_solver=True,
        measure_performance=True, diag_bk_nee_profile=True)
    ion = IonizationConfig(species=[{"name": "N2", "rate": "ppt_talebpour_i_full_reference",
        "Ip_eV": 15.6, "Zeff": 0.9, "l": 0, "m": 0, "fraction": 1.0}],
        time_mode="full", integrator="rk4")
    heat = HeatConfig(hr3b_enabled=True)
    raman = RamanConfig(enabled=True, model="isaacs_rot_sinexp", n_R=2.3e-23,
        omega_R=1.6e13, Gamma_R=1.3e13, T_R=None, T2=None,
        operator_mode=str(operator_mode), operator_convention="isaacs_eq27",
        iir_sampling="exact_piecewise_linear", operator_integrator="heun",
        absorption=False, nonlinear_split_order="strang")
    full = build_longitudinal_schedule(dz=dz, z_max=8 * dz, z_start=0.0,
        focus_window_step=False, focus_center_m=0.0, focus_halfwidth_m=0.0, dz_focus=dz)
    schedule = build_prefix_schedule(full, 8)
    axes = make_axes(grid.Nx, grid.Ny, grid.Nt, grid.Lx, grid.Ly, grid.Twin)
    source, _ = build_transverse_input_field(axes, beam, np.complex128)
    source = np.ascontiguousarray(source, dtype=np.complex128)
    pre0 = np.zeros((8, grid.Ny, grid.Nx), dtype=np.float64, order="C")
    return {"source": source, "pre0": pre0,
        "pre_fields": pre0_fields_from_delta_n(pre0, expected_count=8),
        "grid": grid, "beam": beam, "prop": prop, "ion": ion, "heat": heat,
        "raman": raman, "schedule": schedule}


def _tiny_child(args) -> int:
    try:
        _assert_cpu_backend()
        if args.vendor_dir is not None:
            common.add_vendor(str(args.vendor_dir))
        import threadpoolctl
        limit = threadpoolctl.threadpool_limits(limits=1)
        if any(int(row.get("num_threads", 1)) > 1 for row in threadpoolctl.threadpool_info()):
            raise J1Error("entry-smoke child thread pool is not single-threaded")
        context = _tiny_context()
        result = _run_window(**{key: context[key] for key in ("source", "pre_fields", "grid", "beam",
                "prop", "ion", "heat", "raman", "schedule")},
            output_dir=args.output_root, source_label="synthetic_tiny_PRE0",
            initial_pre0=True, profile=(args.smoke_child == "profile"),
            stage_name=f"software-smoke-{args.smoke_child}",
            execution_scope="SOFTWARE_SMOKE_ONLY")
        _json_new(args.output_root / "child_provenance_receipt.json", {
            "schema": SCHEMA, "phase": PHASE,
            "source_canonical_sha256": result["source_canonical_sha256"],
            "pre0_canonical_sha256": result["pre0_canonical_sha256"],
            "execution_scope": "SOFTWARE_SMOKE_ONLY"})
        del limit
        return 0
    except BaseException as exc:
        failure_path = args.output_root.parent / f"{args.smoke_child}_failure.json"
        try:
            _json_new(failure_path, {"schema": SCHEMA, "phase": PHASE,
                "classification": CLASSIFICATION, "execution_scope": "SOFTWARE_SMOKE_ONLY",
                "status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()})
        except Exception:
            pass
        traceback.print_exc()
        return 1


def _tiny_smoke_pair(args) -> int:
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(exist_ok=False)
    env = os.environ.copy()
    env["UPPE_USE_GPU"] = "0"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    if args.vendor_dir:
        vend = str(args.vendor_dir.resolve())
        env["PYTHONPATH"] = vend + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    records = []
    for stage in ("baseline", "profile"):
        stage_dir = args.output_root / stage
        stage_dir.mkdir(exist_ok=False)
        output = stage_dir / "propagation"
        command = [sys.executable, "-m", "tools.cpu_p1r2.cpu_p1r2_j1",
            "--output-root", str(output), "--smoke-child", stage]
        work = args.bench_root / "Filament_python" if args.bench_root else Path.cwd()
        if args.vendor_dir:
            env["PYTHONPATH"] = str(work.resolve()) + os.pathsep + env["PYTHONPATH"]
        result = _spawn_and_monitor(command, stage_dir=stage_dir, cwd=work,
            env=env, timeout_seconds=240.0, memory_bytes=None)
        if result["exit_code"] != 0 or result["stop"]:
            records.append({"stage": stage, "status": "FAIL", "supervisor": result})
            break
        validation = _validate_stage(stage_dir)
        records.append({"stage": stage, "status": "DONE", "supervisor": result,
                        "validation": validation})
    comparison = (_compare_stage_files(records[0]["validation"], records[1]["validation"],
        "tiny_baseline_vs_profile") if len(records) == 2 and all(r["status"] == "DONE" for r in records) else None)
    profile_receipt = None
    if len(records) == 2 and records[1]["status"] == "DONE":
        profile_receipt = records[1]["validation"]["receipt"].get("profile")
    fft_calls = int((profile_receipt or {}).get("fft_call_count", 0))
    exact = bool(comparison and comparison["canonical_exact_match"] and fft_calls > 0)
    receipt = {"schema": SCHEMA, "phase": PHASE,
        "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
        "execution_scope": "SOFTWARE_SMOKE_ONLY", "status": "PASS" if exact else "FAIL",
        "independent_fresh_children": True, "stages": records,
        "baseline_profile_comparison": comparison,
        "profile_fft_call_count": fft_calls,
        "full_raman_eq27_ionization_hr3b_deposition_fixture": True}
    _json_new(args.output_root / "entry_smoke_receipt.json", receipt)
    return 0 if exact else 1


def _allocation_start_epoch() -> float:
    values = [os.environ.get("P1R2_ALLOCATION_START_EPOCH"),
              os.environ.get("P1R_ALLOCATION_START_EPOCH")]
    values = [float(value) for value in values if value not in (None, "")]
    if not values:
        raise J1Error("P1R2_ALLOCATION_START_EPOCH is required for J1 coordination")
    if len(values) == 2 and abs(values[0] - values[1]) > 1e-6:
        raise J1Error("P1R2 and P1R allocation start epoch variables disagree")
    return values[0]


def _supervise_stage(args, stage: str, *, timeout_seconds: float, memory_bytes: int):
    stage_dir = args.output_root / "stages" / stage
    stage_dir.mkdir(parents=True, exist_ok=False)
    command = _stage_child_command(args, stage, stage_dir / "propagation")
    env = _child_environment(args.bench_root)
    started = time.perf_counter()
    supervisor = _spawn_and_monitor(command, stage_dir=stage_dir,
        cwd=args.bench_root / "Filament_python", env=env,
        timeout_seconds=timeout_seconds, memory_bytes=memory_bytes)
    result = {"stage": stage, "supervisor": supervisor}
    if supervisor.get("exit_code") == 0 and supervisor.get("stop") is None:
        try:
            validation = _validate_stage(stage_dir)
            result.update(validation)
            result["status"] = "DONE"
        except BaseException as exc:
            result.update({"status": "FAIL_OUTPUT_VALIDATION",
                "validation_error": f"{type(exc).__name__}: {exc}"})
    elif (supervisor.get("stop") or {}).get("status") == "STOP_UNRESOLVED" or supervisor.get("exit_code") is None:
        result["status"] = "STOP_UNRESOLVED"
    elif supervisor.get("timed_out"):
        result["status"] = "LIMIT_REACHED"
    elif supervisor.get("stop"):
        result["status"] = "STOPPED_RESOURCE_GATE"
    else:
        result["status"] = "FAIL_CHILD"
    result["service_wall_seconds_including_output_validation"] = time.perf_counter() - started
    result["partial_progress_path"] = str(stage_dir / "propagation" / "observer_progress.jsonl")
    _json_new(stage_dir / "stage_supervisor_receipt.json", result)
    return result


def _run_j1(args) -> int:
    if args.phase != PHASE:
        raise J1Error("J1 coordinator requires explicit --phase J1")
    if any(value is None for value in (args.bench_root, args.input_root, args.vendor_dir)):
        raise J1Error("full J1 requires bench-root, input-root and vendor-dir")
    allocation = p1r._check_allocation()
    common.add_vendor(str(args.vendor_dir))
    import threadpoolctl
    thread_limit = threadpoolctl.threadpool_limits(limits=1)
    if getattr(threadpoolctl, "__version__", None) != "3.6.0":
        raise J1Error("the frozen threadpoolctl 3.6.0 vendor is required")
    if sys.executable != p1r.FIXED_PYTHON:
        raise J1Error(f"fixed Python required: {p1r.FIXED_PYTHON}; got {sys.executable}")
    if any(int(row.get("num_threads", 1)) > 1 for row in threadpoolctl.threadpool_info()):
        raise J1Error("coordinator native thread pool exceeds one thread")
    context = p1r._load_context(args.input_root, args.bench_root, args.vendor_dir)
    del context
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(exist_ok=False)
    p1r._environment_receipt(args.output_root, vendor_dir=args.vendor_dir, allocation=allocation)
    allocation_start = _allocation_start_epoch()
    deadline = allocation_start + ALLOCATION_SECONDS
    stages, memory_latched = [], False
    baseline_repeat_comparison = None
    base_wall = None
    for index, stage in enumerate(STAGES):
        block_reason = _stage_block_reason(memory_latched, stages,
            baseline_repeat=baseline_repeat_comparison)
        if block_reason is not None:
            stages.append({"stage": stage, "status": block_reason})
            continue
        remaining = deadline - time.time()
        if index == 0:
            predicted = WINDOW_SECONDS
        elif stage == "profile":
            predicted = min(WINDOW_SECONDS, max(60.0, float(base_wall or WINDOW_SECONDS) * 1.75))
        else:
            predicted = min(WINDOW_SECONDS, max(60.0, float(base_wall or WINDOW_SECONDS) * 1.25))
        if remaining < predicted + RESERVE_SECONDS:
            stages.append({"stage": stage, "status": "SKIPPED_INSUFFICIENT_TIME",
                "remaining_seconds": remaining, "predicted_seconds": predicted,
                "required_reserve_seconds": RESERVE_SECONDS})
            continue
        result = _supervise_stage(args, stage,
            timeout_seconds=min(WINDOW_SECONDS, max(1.0, remaining - RESERVE_SECONDS)),
            memory_bytes=p1r.MEMORY_LIMIT_BYTES)
        stages.append(result)
        _append_event(args.output_root / "run_progress.jsonl", {
            "phase": PHASE, "stage": stage, "status": result.get("status"),
            "service_wall_seconds": result.get("service_wall_seconds_including_output_validation"),
            "rss_80_percent_latched": result.get("supervisor", {}).get("rss_80_percent_latched")})
        memory_latched = bool(result.get("supervisor", {}).get("rss_80_percent_latched"))
        if index == 0 and result.get("status") == "DONE":
            base_wall = float(result["receipt"]["propagation_wall_seconds"])
        if stage == "baseline-b" and result.get("status") == "DONE" and stages:
            baseline_a = next((row for row in stages[:-1]
                               if row.get("stage") == "baseline-a" and row.get("status") == "DONE"), None)
            if baseline_a is not None:
                baseline_repeat_comparison = _compare_stage_files(
                    baseline_a, result, "baseline_a_vs_baseline_b")
                _json_new(args.output_root / "baseline_repeat_comparison.json",
                          baseline_repeat_comparison)

    done = {row["stage"]: row for row in stages if row.get("status") == "DONE"}
    comparisons = {}
    if "baseline-a" in done and "baseline-b" in done:
        comparisons["baseline_repeat"] = (baseline_repeat_comparison or
            _compare_stage_files(done["baseline-a"], done["baseline-b"],
                                 "baseline_a_vs_baseline_b"))
    if "baseline-a" in done and "profile" in done:
        comparisons["profile_vs_baseline"] = _compare_stage_files(done["baseline-a"],
            done["profile"], "baseline_a_vs_profile")
    all_complete = len(done) == 3 and len(stages) == 3
    exact = all_complete and all(item.get("canonical_exact_match") for item in comparisons.values())
    base_times = [float(done[name]["receipt"]["propagation_wall_seconds"])
                  for name in ("baseline-a", "baseline-b") if name in done]
    profile_time = (float(done["profile"]["receipt"]["propagation_wall_seconds"])
                    if "profile" in done else None)
    baseline_mean = sum(base_times) / len(base_times) if base_times else None
    profile_assessment = {"profile_propagation_wall_seconds": profile_time,
        "baseline_mean_propagation_wall_seconds": baseline_mean,
        "additional_profile_overhead_ratio": ((profile_time - baseline_mean) / baseline_mean
            if profile_time is not None and baseline_mean else None),
        "attribution": "UNATTRIBUTED"}
    profile_records = done.get("profile", {}).get("receipt", {}).get("profile") or {}
    fft_fraction = (float(profile_records.get("fft_inclusive_wall_seconds", 0.0)) / profile_time
                    if profile_time and profile_time > 0 else None)
    diagnostic_status = ("COMPLETE_DIAGNOSTIC" if exact else
        ("NUMERICAL_DIFFERENCE" if baseline_repeat_comparison is not None
         and not baseline_repeat_comparison.get("canonical_exact_match", False)
         else ("NUMERICAL_DIFFERENCE" if all_complete else "INCOMPLETE")))
    receipt = {"schema": SCHEMA, "phase": PHASE,
        "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
        "status": diagnostic_status,
        "execution_order": list(STAGES), "execution_mode": "SERIAL_FRESH_CHILD_PER_STAGE",
        "allocation": allocation, "allocation_start_epoch": allocation_start,
        "allocation_wall_limit_seconds": ALLOCATION_SECONDS,
        "reserve_seconds": RESERVE_SECONDS, "work_deadline_epoch": deadline,
        "fresh_source_and_pre0_each_stage": True,
        "source_canonical_sha256_each_stage": {name: done[name]["receipt"].get("source_canonical_sha256") for name in done},
        "pre0_canonical_sha256_each_stage": {name: done[name]["receipt"].get("pre0_canonical_sha256") for name in done},
        "common_instrumentation_flags": {"measure_performance": True,
            "diag_bk_nee_profile": True},
        "stages": stages, "comparisons": comparisons,
        "profile_overhead_assessment": profile_assessment,
        "baseline_repeat_gate": baseline_repeat_comparison,
        "profile_fft_inclusive_wall_fraction": fft_fraction,
        "profile_hotspot_records_path": str(args.output_root / "stages" / "profile" / "propagation" / "function_fft_profile.json"),
        "rss_policy": {"80_percent": "LATCH_AND_STOP_LAUNCHING_LATER_STAGES",
            "90_percent": "BOUNDED_TERMINATION_OF_CURRENT_CHILD",
            "memory_bytes": p1r.MEMORY_LIMIT_BYTES,
            "pss_sample_interval_seconds": PSS_SAMPLE_SECONDS}}
    _json_new(args.output_root / "run_receipt.json", receipt)
    del thread_limit
    return 0 if exact else 1


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=(PHASE,))
    parser.add_argument("--bench-root", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--vendor-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--child-stage", choices=STAGES, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-child", choices=("baseline", "profile"), help=argparse.SUPPRESS)
    parser.add_argument("--entry-smoke", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.smoke_child:
            return _tiny_child(args)
        if args.entry_smoke:
            return _tiny_smoke_pair(args)
        if args.child_stage:
            return _real_child(args)
        return _run_j1(args)
    except BaseException as exc:
        traceback.print_exc()
        try:
            failure_path = (args.output_root / "coordinator_failure.json"
                if args.output_root.is_dir() else args.output_root.with_suffix(".failure.json"))
            _json_new(failure_path, {
                "schema": SCHEMA, "phase": PHASE, "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()})
        except Exception:
            pass
        return 2


def _real_child(args) -> int:
    try:
        if args.phase != PHASE:
            raise J1Error("J1 child requires explicit --phase J1")
        if any(value is None for value in (args.bench_root, args.input_root, args.vendor_dir)):
            raise J1Error("J1 child requires bench-root, input-root and vendor-dir")
        if sys.version_info < (3, 12) or not hasattr(sys, "monitoring"):
            raise J1Error("Python 3.12+ sys.monitoring is required")
        allocation = p1r._check_allocation()
        common.add_vendor(str(args.vendor_dir))
        import threadpoolctl
        limit = threadpoolctl.threadpool_limits(limits=1)
        if getattr(threadpoolctl, "__version__", None) != "3.6.0":
            raise J1Error("the frozen threadpoolctl 3.6.0 vendor is required")
        if sys.executable != p1r.FIXED_PYTHON:
            raise J1Error(f"fixed Python required: {p1r.FIXED_PYTHON}; got {sys.executable}")
        if any(int(row.get("num_threads", 1)) > 1 for row in threadpoolctl.threadpool_info()):
            raise J1Error("J1 child thread pool exceeds one thread")
        _assert_cpu_backend()
        p1r._source_binding(args.bench_root)
        context = p1r._load_context(args.input_root, args.bench_root, args.vendor_dir)
        source_hash = str(context["source_hash"])
        pre0_hash = str(context["assets"]["pre0_canonical_sha256"])
        pre_fields = __import__("KHz_filament.hr4e5_formal_entry", fromlist=["pre0_fields_from_delta_n"]).pre0_fields_from_delta_n(
            context["assets"]["pre0"], expected_count=8)
        receipt = _run_window(source=context["source"], pre_fields=pre_fields,
            grid=context["grid"], beam=context["beam"], prop=context["prop"],
            ion=context["ion"], heat=context["heat"], raman=context["raman"],
            schedule=context["schedule"], output_dir=args.output_root,
            source_label="sealed_pre0_first_8", initial_pre0=True,
            profile=(args.child_stage == "profile"), source_hash=source_hash,
            pre0_hash=pre0_hash, stage_name=args.child_stage)
        receipt.update({"source_raw_sha256": p1r.SOURCE_RAW_SHA256,
            "pre0_raw_sha256": p1r.PRE0_RAW_SHA256,
            "transfer_mapping_sha256": context["assets"]["mapping_sha256"],
            "source_binding": context["source_binding"],
            "source_validation": _json_safe(context["source_validation"]),
            "lut_validation": _json_safe(context["lut"]),
            "schedule": _json_safe(context["schedule"].as_metadata()),
            "runtime_config_delta": _json_safe(context["runtime_config_delta"]),
            "effective_linear_precision_strategy": _effective_precision_strategy(context["prop"]),
            "child_allocation": allocation,
            "threadpool_info": threadpoolctl.threadpool_info()})
        # child_receipt was already created with the output arrays. Replace is forbidden;
        # provenance supplement is a separate non-overwriting receipt.
        _json_new(args.output_root / "child_provenance_receipt.json", receipt)
        del limit
        return 0
    except BaseException as exc:
        failure = {"schema": SCHEMA, "phase": PHASE, "stage": args.child_stage,
            "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
            "status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()}
        try:
            _json_new(args.output_root.parent / "child_failure.json", failure)
        except Exception:
            pass
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
