"""Private, diagnostic-only timing around the frozen CPU-O1 I propagator.

S4 instruments one factory-created private clone. It does not edit scientific
modules, replace numerical expressions, retain full arrays, or submit jobs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import functools
import importlib
import inspect
from itertools import islice
import json
import math
import os
from pathlib import Path
import platform
import sys
import threading
import time
import traceback
from types import SimpleNamespace


SCHEMA = "filament.cpu_s4.private_i_profile.v1"
TIMING_CONTRACT_SCHEMA = "filament.cpu_s4.profile_timing_contract.v3"
ARRAY_RECORD_THRESHOLD_BYTES = 1024 * 1024
FIXED_EXECUTION = {
    "identity": "I",
    "temporal_workers": 48,
    "spatial_workers": 24,
    "kernel_threads": 24,
    "layout": "NtB",
    "block_pixels": 32,
}
TRANSFORMS = frozenset((
    "fft", "ifft", "fft2", "ifft2", "fftn", "ifftn",
    "rfft", "irfft", "rfft2", "irfft2", "rfftn", "irfftn",
))
LINEAR_FUNCTIONS = frozenset((
    "lin_propagator", "step_linear", "step_linear_bk_nee_factorized",
))
RAMAN_FUNCTIONS = frozenset((
    "make_raman_kernel", "precompute_kernel_fft",
    "raman_convolve_intensity_fft_linear", "raman_convolve_intensity",
    "isaacs_raman_stage", "isaacs_raman_field_rhs",
    "isaacs_raman_signed_energy_density", "stable_field_fluence_and_loss",
    "apply_isaacs_raman_operator_step",
))
PROPAGATE_FUNCTIONS = {
    "evolve_rho_time": "ionization.full_time_rk4_candidate",
    "apply_nonlinear": "nonlinear.apply_nonlinear",
    "shock_intensity": "nonlinear.shock_intensity",
    "kerr_phase_from_deltan": "nonlinear.kerr_phase_from_deltan",
    "plasma_phase": "nonlinear.plasma_phase",
    "ib_alpha": "nonlinear.ib_alpha",
    "intensity": "diagnostics.intensity",
    "pulse_energy": "diagnostics.pulse_energy",
    "build_unified_deposition_ledger": "deposition.build_unified_deposition_ledger",
    "thermalize_interval": "thermalization.thermalize_interval",
    "map_post_acoustic_increment": "thermalization.map_post_acoustic_increment",
    "heat_Q_per_z": "deposition.heat_Q_per_z",
    "direct_interval_energy": "deposition.direct_interval_energy",
    "interval_energy_from_fluence_gain": "deposition.interval_energy_from_fluence_gain",
    "interval_energy_from_q": "deposition.interval_energy_from_q",
    "second_moment_radius": "diagnostics.second_moment_radius",
    "_fwhm_time_1d": "diagnostics._fwhm_time_1d",
    "_fwhm_diameter_xy_center": "diagnostics._fwhm_diameter_xy_center",
}


def _json_safe(value):
    """Convert profiler metadata to strict-JSON-safe scalars without arrays."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - runtime has NumPy
        np = None
    if np is not None and isinstance(value, np.ndarray):
        return {"array_metadata_only": True, "shape": [int(x) for x in value.shape],
                "dtype": str(value.dtype), "nbytes": int(value.nbytes)}
    if np is not None and isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def _strict_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         allow_nan=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _strict_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(_json_safe(row), ensure_ascii=False,
                sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _walk_arrays(value, *, depth=0):
    """Yield array objects from shallow call metadata; never materialize them."""
    if depth > 2:
        return
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return
    if isinstance(value, np.ndarray):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value[:64]:
            yield from _walk_arrays(item, depth=depth + 1)
    elif isinstance(value, dict):
        for item in islice(value.values(), 64):
            yield from _walk_arrays(item, depth=depth + 1)


def _array_bytes(value) -> int:
    return sum(int(item.nbytes) for item in _walk_arrays(value))


def _named_arrays(value, *, prefix="value", depth=0):
    """Yield bounded array metadata paths without retaining the arrays."""
    if depth > 3:
        return
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return
    if isinstance(value, np.ndarray):
        yield prefix, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value[:64]):
            yield from _named_arrays(item, prefix=f"{prefix}[{index}]", depth=depth + 1)
    elif isinstance(value, dict):
        for key, item in list(value.items())[:64]:
            yield from _named_arrays(item, prefix=f"{prefix}.{key}", depth=depth + 1)


def _current_interval(frame):
    cursor = frame
    while cursor is not None:
        code = cursor.f_code
        if (code.co_name == "propagate_one_pulse"
                and code.co_filename.replace("\\", "/").endswith(
                    "/KHz_filament/propagate.py")):
            value = cursor.f_locals.get("interval_index")
            if isinstance(value, int) and not isinstance(value, bool):
                return int(value)
            return None
        cursor = cursor.f_back
    return None


class _TimerScope:
    def __init__(self, timer, label, callsite, interval, metadata, input_values=None):
        self.timer = timer
        self.label = str(label)
        self.callsite = str(callsite)
        self.interval = interval
        self.metadata = metadata
        self.input_bytes = _array_bytes(input_values) if input_values is not None else 0
        self.return_bytes = 0
        self.child_wall = 0.0
        self.child_cpu = 0.0
        self.inclusive_wall_seconds = 0.0
        self.exclusive_wall_seconds = 0.0
        self.inclusive_process_cpu_seconds = 0.0
        self.exclusive_process_cpu_seconds = 0.0
        self._started = False
        self._failed = False
        self._operations = []

    def __enter__(self):
        self.timer._start_scope(self)
        self.started_wall = time.perf_counter()
        self.started_cpu = time.process_time()
        self._started = True
        return self

    def observe_arrays(self, *, operation, inputs, output, kind,
                       disposition=None):
        if not self._started:
            raise RuntimeError("array metadata can only be recorded inside an active timer scope")
        self._operations.append({"operation": str(operation), "inputs": inputs,
            "output": output, "kind": str(kind), "disposition": disposition})

    def set_return(self, result):
        self.return_bytes = _array_bytes(result)

    def __exit__(self, exc_type, exc, tb):
        self._failed = exc_type is not None
        wall = time.perf_counter() - self.started_wall
        cpu = time.process_time() - self.started_cpu
        self.timer._finish_scope(self, wall, cpu)
        return False


class NestedTimer:
    """Nested wall/process-CPU scopes, source roles, and bounded array metadata."""

    def __init__(self, *, array_threshold_bytes=ARRAY_RECORD_THRESHOLD_BYTES):
        if isinstance(array_threshold_bytes, bool) or not isinstance(array_threshold_bytes, int) or array_threshold_bytes < 1:
            raise ValueError("array threshold must be a positive integer")
        self.array_threshold_bytes = int(array_threshold_bytes)
        self._rows = defaultdict(lambda: {
            "calls": 0, "failures": 0,
            "inclusive_wall_seconds": 0.0, "exclusive_wall_seconds": 0.0,
            "inclusive_process_cpu_seconds": 0.0,
            "exclusive_process_cpu_seconds": 0.0,
            "input_array_bytes_observed": 0, "return_array_bytes_observed": 0,
        })
        self._array_rows = defaultdict(lambda: {
            "calls": 0, "input_logical_bytes": 0, "output_logical_bytes": 0,
            "inclusive_wall_seconds": 0.0, "exclusive_wall_seconds": 0.0,
            "inclusive_process_cpu_seconds": 0.0,
            "exclusive_process_cpu_seconds": 0.0,
        })
        self._array_omitted = defaultdict(int)
        self._instrumentation_bindings = []
        self._local = threading.local()
        self._lock = threading.Lock()
        self._thread_ids = set()
        self._nested_calls = 0
        self._cross_thread_overlapping_calls = 0
        self._active_calls = 0
        self._active_by_thread = defaultdict(int)

    def _stack(self):
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    def register_instrumentation(self, binding):
        self._instrumentation_bindings.append(_json_safe(binding))

    def current_scope_metadata(self):
        stack = self._stack()
        return dict(stack[-1].metadata) if stack else {}

    def _start_scope(self, scope):
        stack = self._stack()
        if stack and "parent_stage" not in scope.metadata:
            scope.metadata["parent_stage"] = stack[-1].label
        stack.append(scope)
        thread_id = threading.get_ident()
        with self._lock:
            if self._active_calls:
                if any(active_id != thread_id and count > 0
                       for active_id, count in self._active_by_thread.items()):
                    self._cross_thread_overlapping_calls += 1
                elif self._active_by_thread[thread_id] > 0:
                    self._nested_calls += 1
            self._active_calls += 1
            self._active_by_thread[thread_id] += 1
            self._thread_ids.add(thread_id)
        scope.thread_id = thread_id

    def _finish_scope(self, scope, wall, cpu):
        try:
            stack = self._stack()
            if not stack or stack[-1] is not scope:
                raise RuntimeError("nested timer scope stack is corrupt")
            stack.pop()
            scope.inclusive_wall_seconds = float(wall)
            scope.exclusive_wall_seconds = max(0.0, float(wall) - scope.child_wall)
            scope.inclusive_process_cpu_seconds = float(cpu)
            scope.exclusive_process_cpu_seconds = max(0.0, float(cpu) - scope.child_cpu)
            if stack:
                stack[-1].child_wall += wall
                stack[-1].child_cpu += cpu
            metadata = dict(scope.metadata)
            metadata_json = json.dumps(_json_safe(metadata), sort_keys=True,
                                      allow_nan=False, separators=(",", ":"))
            key = (scope.label, scope.callsite, scope.interval, metadata_json)
            with self._lock:
                row = self._rows[key]
                row["calls"] += 1
                row["failures"] += int(scope._failed)
                row["inclusive_wall_seconds"] += wall
                row["exclusive_wall_seconds"] += max(0.0, wall - scope.child_wall)
                row["inclusive_process_cpu_seconds"] += cpu
                row["exclusive_process_cpu_seconds"] += max(0.0, cpu - scope.child_cpu)
                row["input_array_bytes_observed"] += scope.input_bytes
                row["return_array_bytes_observed"] += scope.return_bytes
                self._active_calls -= 1
                self._active_by_thread[scope.thread_id] -= 1
                if self._active_by_thread[scope.thread_id] == 0:
                    del self._active_by_thread[scope.thread_id]
            for operation in tuple(scope._operations):
                try:
                    self._record_array_operation(scope, operation)
                finally:
                    operation["inputs"] = ()
                    operation["output"] = None
        finally:
            # The AST-injected `with ... as scope` local remains in the source
            # function frame after __exit__; metadata must not retain arrays.
            scope._operations.clear()

    def _record_array_operation(self, scope, operation):
        named_inputs = []
        for index, value in enumerate(operation["inputs"]):
            named_inputs.extend(_named_arrays(value, prefix=f"input{index}"))
        named_outputs = list(_named_arrays(operation["output"], prefix="output"))
        if not named_inputs and not named_outputs:
            return
        input_bytes = sum(int(array.nbytes) for _, array in named_inputs)
        output_bytes = sum(int(array.nbytes) for _, array in named_outputs)
        if max(input_bytes, output_bytes) < self.array_threshold_bytes:
            with self._lock:
                self._array_omitted[operation["operation"]] += 1
            return
        input_meta = tuple((name, tuple(int(x) for x in array.shape), str(array.dtype),
                            int(array.nbytes)) for name, array in named_inputs)
        output_meta = tuple((name, tuple(int(x) for x in array.shape), str(array.dtype),
                             int(array.nbytes)) for name, array in named_outputs)
        shared_flags = []
        if named_inputs and named_outputs:
            import numpy as np
            for _, output in named_outputs:
                for _, input_array in named_inputs:
                    try:
                        shared_flags.append(bool(np.shares_memory(output, input_array)))
                    except Exception:
                        shared_flags.append(None)
        if operation["disposition"] is not None:
            storage = str(operation["disposition"])
        elif any(flag is None for flag in shared_flags):
            storage = "storage_sharing_indeterminate"
        elif not shared_flags or not any(shared_flags):
            storage = "distinct_output_storage_observed"
        elif all(shared_flags):
            storage = "shared_or_view_storage_observed"
        else:
            storage = "mixed_storage_observed"
        parent = scope.metadata.get("parent_stage")
        role = scope.metadata.get("stage_role")
        timing_scope = ("whole_statement" if "source_bound_statement" in scope.metadata
                        else "whole_function")
        key = (operation["operation"], scope.label, scope.callsite, scope.interval,
               role, parent, operation["kind"], storage, timing_scope,
               input_meta, output_meta)
        with self._lock:
            row = self._array_rows[key]
            row["calls"] += 1
            row["input_logical_bytes"] += input_bytes
            row["output_logical_bytes"] += output_bytes
            row["inclusive_wall_seconds"] += scope.inclusive_wall_seconds
            row["exclusive_wall_seconds"] += scope.exclusive_wall_seconds
            row["inclusive_process_cpu_seconds"] += scope.inclusive_process_cpu_seconds
            row["exclusive_process_cpu_seconds"] += scope.exclusive_process_cpu_seconds

    def statement(self, label: str, *, callsite: str, stage_role: str,
                  parent_stage: str):
        frame = inspect.currentframe().f_back
        metadata = {"stage_role": str(stage_role), "parent_stage": str(parent_stage),
                    "source_bound_statement": str(callsite)}
        return _TimerScope(self, label, callsite, _current_interval(frame), metadata)

    def wrap(self, label: str, function, *, metadata=None, array_observer=None):
        if not callable(function):
            raise TypeError(f"cannot profile non-callable {label}")

        @functools.wraps(function)
        def measured(*args, **kwargs):
            caller = inspect.currentframe().f_back
            callsite = f"{caller.f_code.co_filename}:{caller.f_lineno}:{caller.f_code.co_name}"
            interval = _current_interval(caller)
            call_metadata = (metadata(args, kwargs, caller) if callable(metadata)
                             else dict(metadata or {}))
            scope = _TimerScope(self, label, callsite, interval,
                                dict(call_metadata), input_values=(args, kwargs))
            with scope:
                result = function(*args, **kwargs)
                scope.set_return(result)
                if array_observer is not None:
                    observed = array_observer(args, kwargs, result, caller)
                    if observed:
                        scope.observe_arrays(**observed)
                return result

        measured.__s4_inner_code__ = getattr(function, "__s4_inner_code__", function.__code__)
        return measured

    def report(self):
        with self._lock:
            records = []
            for (label, callsite, interval, metadata_json), values in sorted(
                    self._rows.items(), key=lambda item: (
                        item[0][0], item[0][2] if item[0][2] is not None else -1,
                        item[0][1], item[0][3])):
                records.append({"label": label, "callsite": callsite,
                    "interval_index": interval, **json.loads(metadata_json), **dict(values)})
            array_records = []
            for key, values in sorted(self._array_rows.items(), key=lambda item: repr(item[0])):
                (operation, timer_label, callsite, interval, role, parent, kind,
                 storage, timing_scope, inputs, outputs) = key
                array_records.append({"operation": operation, "timer_label": timer_label,
                    "callsite": callsite, "interval_index": interval,
                    "stage_role": role, "parent_stage": parent, "kind": kind,
                    "storage_observation": storage,
                    "timing_scope": timing_scope,
                    "aggregation_warning": "alternative_view_do_not_sum_with_nested_timers",
                    "input_arrays": [{"path": name, "shape": list(shape), "dtype": dtype,
                                      "nbytes": nbytes} for name, shape, dtype, nbytes in inputs],
                    "output_arrays": [{"path": name, "shape": list(shape), "dtype": dtype,
                                       "nbytes": nbytes} for name, shape, dtype, nbytes in outputs],
                    **dict(values)})
            omitted = dict(self._array_omitted)
            thread_ids = sorted(int(value) for value in self._thread_ids)
            nested_calls = int(self._nested_calls)
            cross_thread_overlaps = int(self._cross_thread_overlapping_calls)
        root_rows = [row for row in records
                     if row["label"] == "propagate.propagate_one_pulse"]
        root_wall = sum(row["inclusive_wall_seconds"] for row in root_rows)
        root_cpu = sum(row["inclusive_process_cpu_seconds"] for row in root_rows)
        root_exclusive_wall = sum(row["exclusive_wall_seconds"] for row in root_rows)
        root_exclusive_cpu = sum(row["exclusive_process_cpu_seconds"] for row in root_rows)
        exclusive_wall = sum(row["exclusive_wall_seconds"] for row in records)
        exclusive_cpu = sum(row["exclusive_process_cpu_seconds"] for row in records)
        return {
            "schema": "filament.cpu_s4.nested_timer.v1",
            "clock": {"wall": "time.perf_counter", "process_cpu": "time.process_time",
                      "process_cpu_parallel_threads": "aggregate process CPU time"},
            "single_python_caller_expected": True,
            "caller_thread_ids": thread_ids,
            "nested_profile_calls_observed": nested_calls,
            "cross_thread_overlapping_profile_calls_observed": cross_thread_overlaps,
            "records": records,
            "calls": sum(row["calls"] for row in records),
            "failures": sum(row["failures"] for row in records),
            "reconciliation": {
                "propagator_inclusive_wall_seconds": root_wall,
                "propagator_inclusive_process_cpu_seconds": root_cpu,
                "propagator_exclusive_wall_seconds_unprofiled_direct_work": root_exclusive_wall,
                "propagator_exclusive_process_cpu_seconds_unprofiled_direct_work": root_exclusive_cpu,
                "sum_all_profiled_exclusive_wall_seconds": exclusive_wall,
                "sum_all_profiled_exclusive_process_cpu_seconds": exclusive_cpu,
                "wall_reconciliation_delta_seconds": root_wall - exclusive_wall if root_rows else None,
                "process_cpu_reconciliation_delta_seconds": root_cpu - exclusive_cpu if root_rows else None,
                "interpretation": "exclusive rows partition nested instrumented scopes; root exclusive time remains direct work and wrapper overhead",
            },
            "array_accounting": {
                "threshold_bytes": self.array_threshold_bytes,
                "operations": array_records,
                "omitted_below_threshold_calls": omitted,
                "byte_values_are": "logical input/output array sizes, not memory traffic or allocation bytes",
                "storage_observation": "NumPy storage-sharing check at the observed call boundary; does not infer hidden internal copies",
                "full_array_retention": False,
            },
            "instrumentation_bindings": list(self._instrumentation_bindings),
            "unmeasured": ["Python scheduling and synchronization wait", "opaque NumPy/SciPy/Numba internal allocations", "memory bandwidth and NUMA traffic"],
        }


class _ProfileReceipt:
    def __init__(self, timer: NestedTimer, dispatch, propagator, *, operator_mode):
        self.timer = timer
        self.dispatch = dispatch
        self.propagator = propagator
        self.operator_mode = str(operator_mode)

    def report(self):
        execution = dict(getattr(self.propagator, "_cpu_o1_execution", {}))
        workspace = getattr(self.propagator, "_cpu_o1_rate_workspace", None)
        timers = self.timer.report()
        records = timers["records"]
        stage_calls = [row for row in records
                       if row["label"] == "raman.isaacs_raman_stage"]
        operator_calls = [row for row in records
                          if row["label"] == "raman.apply_isaacs_raman_operator_step"]
        roles = defaultdict(int)
        for row in stage_calls:
            role = row.get("stage_role")
            if role:
                roles[str(role)] += int(row["calls"])
        contract = {
            "schema": TIMING_CONTRACT_SCHEMA,
            "raman_mode": self.operator_mode,
            "active_path": ["isaacs_raman_stage", "apply_isaacs_raman_operator_step"],
            "inactive_complete_path": "isaacs_complete_eq27_stage_not_instrumented",
            "inactive_complete_runtime_count": None,
            "inactive_complete_evidence": "excluded_by_frozen_mode_and_source_not_timed",
            "active_function_call_counts": {
                "isaacs_raman_stage": sum(int(row["calls"]) for row in stage_calls),
                "apply_isaacs_raman_operator_step": sum(int(row["calls"]) for row in operator_calls),
            },
            "active_support_function_call_counts": {
                "stable_field_fluence_and_loss": sum(
                    int(row["calls"]) for row in records
                    if row["label"] == "raman.stable_field_fluence_and_loss"),
            },
            "stage_call_roles": dict(sorted(roles.items())),
            "stage1_reused_counts": {
                "supplied": sum(int(row["calls"]) for row in operator_calls
                                 if row.get("stage1_supplied") is True),
                "recomputed_inside_operator": sum(int(row["calls"]) for row in operator_calls
                    if row.get("stage1_supplied") is False),
            },
            "array_record_threshold_bytes": self.timer.array_threshold_bytes,
            "source_bindings": list(self.timer._instrumentation_bindings),
            "exclusive_accounting_source": "nested_timers.records.exclusive_*; array metadata has no additive timing",
            "unmeasured": ["Python scheduling and synchronization wait", "opaque NumPy/SciPy internal allocation traffic", "memory bandwidth and NUMA traffic"],
        }
        return {
            "schema": SCHEMA,
            "mode": "PROFILE",
            "implementation": "I",
            "execution": _json_safe(execution),
            "fft_dispatch": _json_safe(self.dispatch.report()),
            "nested_timers": timers,
            "ionization_candidate_events": _json_safe(list(
                getattr(self.propagator, "_cpu_o1_candidate_events", ()))),
            "rate_workspace": ({
                "allocations": int(workspace.allocations),
                "reuses": int(workspace.reuses),
                "live_buffer_bytes": int(workspace.array.nbytes) if workspace.array is not None else 0,
                "allocated_bytes_if_single_allocation": (int(workspace.array.nbytes)
                    if workspace.allocations == 1 and workspace.array is not None else None),
                "scope": "observable CPU-O1 rate-packing workspace only",
            } if workspace is not None else {"status": "NOT_PRESENT"}),
            "raman_heun_split": {
                "status": "SOURCE_BOUND_STATEMENT_CONTEXTS",
                "predictor_update": "raman.heun.predictor_update",
                "stage2_response": "raman.heun.stage2_recomputed",
                "corrector_update": "raman.heun.corrector_update",
                "stage1_response": "caller_supplied_and_reused; existing stage timer is reused",
            },
            "timing_contract": contract,
            "global_science_modules_modified": False,
            "full_arrays_retained": False,
        }


def _profile_private_clone(propagator, dispatch, timer: NestedTimer, *, operator_mode):
    """Wrap selected symbols in the one factory-created clone's namespaces."""
    from tools.cpu_o1.cpu_s4.timing_instrumentation import instrument_active_raman_function

    if str(operator_mode) != "full_isaacs_eq27":
        raise RuntimeError("S4 PROFILE supports only the frozen full_isaacs_eq27 Raman path")
    if getattr(propagator, "_cpu_o1_identity", None) != "I":
        raise RuntimeError("S4 PROFILE requires the fixed I propagator identity")
    private = propagator.__globals__
    replacements = {}
    seen_namespaces = set()

    def stage_metadata(args, kwargs, caller):
        parent = timer.current_scope_metadata()
        if parent.get("stage_role") == "stage2_recomputed":
            role = "stage2_recomputed"
        elif caller.f_code.co_name == "_full_raman_substep":
            role = "stage1_supplied"
        else:
            role = "unclassified"
        return {"stage_role": role, "operator_mode": str(operator_mode)}

    def operator_metadata(args, kwargs, caller):
        supplied = kwargs.get("stage1") is not None
        return {"stage_role": "heun_operator", "operator_mode": str(operator_mode),
                "stage1_supplied": bool(supplied),
                "stage1_recomputed_inside_operator": not bool(supplied)}

    def stage_array_observer(args, kwargs, result, caller):
        return {"operation": "raman.stage.output_bundle", "inputs": (args[0],),
                "output": result, "kind": "function_output_bundle"}

    def convolution_array_observer(args, kwargs, result, caller):
        return {"operation": "raman.response.convolution_output", "inputs": (args, kwargs),
                "output": result, "kind": "convolution_output"}

    for key in ("step_linear_bk_nee_factorized", "step_linear", "lin_propagator",
                "isaacs_raman_stage", "apply_isaacs_raman_operator_step"):
        function = private.get(key)
        if inspect.isfunction(function):
            namespace = function.__globals__
            module_name = str(namespace.get("__name__", ""))
            if module_name not in {"KHz_filament.linear", "KHz_filament.raman"}:
                continue
            module = importlib.import_module(module_name)
            if namespace is module.__dict__ or id(namespace) in seen_namespaces:
                continue
            seen_namespaces.add(id(namespace))
            selected = LINEAR_FUNCTIONS if module_name.endswith(".linear") else RAMAN_FUNCTIONS
            for name in sorted(selected):
                candidate = namespace.get(name)
                if inspect.isfunction(candidate) and candidate.__globals__ is namespace:
                    original_candidate = candidate
                    label = f"{'linear' if module_name.endswith('.linear') else 'raman'}.{name}"
                    metadata = None
                    array_observer = None
                    if module_name.endswith(".raman") and name in {
                            "isaacs_raman_stage", "apply_isaacs_raman_operator_step",
                            "stable_field_fluence_and_loss"}:
                        candidate, binding = instrument_active_raman_function(candidate, timer)
                        timer.register_instrumentation(binding)
                        if name == "isaacs_raman_stage":
                            metadata = stage_metadata
                            array_observer = stage_array_observer
                        elif name == "apply_isaacs_raman_operator_step":
                            metadata = operator_metadata
                        else:
                            metadata = {"stage_role": "energy_diagnostic",
                                        "operator_mode": str(operator_mode)}
                    elif module_name.endswith(".raman") and name == "raman_convolve_intensity":
                        array_observer = convolution_array_observer
                    wrapped = timer.wrap(label, candidate, metadata=metadata,
                                         array_observer=array_observer)
                    namespace[name] = wrapped
                    replacements[id(original_candidate)] = wrapped
                    replacements[id(candidate)] = wrapped

    # The propagator imports direct helpers from scientific modules. Bind timed
    # wrappers only in its already-private clone namespace.
    for name, label in PROPAGATE_FUNCTIONS.items():
        function = private.get(name)
        if inspect.isfunction(function) and id(function) not in replacements:
            wrapped = timer.wrap(label, function)
            replacements[id(function)] = wrapped
            private[name] = wrapped

    # evolve_rho_time was wrapped exactly once by PROPAGATE_FUNCTIONS above.
    for name, function in tuple(private.items()):
        replacement = replacements.get(id(function))
        if replacement is not None:
            private[name] = replacement

    for transform in sorted(TRANSFORMS):
        current = getattr(dispatch, transform, None)
        if callable(current):
            setattr(dispatch, transform, timer.wrap(f"fft.{transform}", current))

    workspace = getattr(dispatch, "cpu_o1_rate_workspace", None)
    if workspace is not None:
        acquire = getattr(workspace, "acquire", None)
        if not callable(acquire):
            raise RuntimeError("S4 rate workspace has no acquire method")

        def workspace_array_observer(args, kwargs, result, caller):
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("rate workspace acquire return contract changed")
            array, reused = result
            return {"operation": "ionization.rate_workspace.acquire", "inputs": (),
                    "output": array, "kind": "workspace_buffer",
                    "disposition": "reused" if reused else "allocated"}

        workspace.acquire = timer.wrap("ionization.rate_workspace.acquire", acquire,
            metadata={"stage_role": "rate_workspace_acquire"},
            array_observer=workspace_array_observer)

    measured = timer.wrap("propagate.propagate_one_pulse", propagator)
    measured.__j1_inner_code__ = getattr(
        propagator, "__j1_inner_code__", propagator.__code__)
    measured.__cpu_s4_private_profile__ = True
    propagator._cpu_o1_rate_workspace = getattr(
        dispatch, "cpu_o1_rate_workspace", None)
    return measured


def make_profiled_i_factory(*, enabled: bool = True, kernel_threads: int = 24,
                            operator_mode="full_isaacs_eq27",
                            array_threshold_bytes=ARRAY_RECORD_THRESHOLD_BYTES):
    """Create the fixed I private factory, optionally wrapping its private clone."""
    from tools.cpu_o1.cpu_o1 import make_joint_propagator_factory

    base_factory = make_joint_propagator_factory("I", **{
        "kernel_threads": kernel_threads,
        "block_pixels": FIXED_EXECUTION["block_pixels"],
        "layout": FIXED_EXECUTION["layout"],
        "temporal_workers": FIXED_EXECUTION["temporal_workers"],
        "spatial_workers": FIXED_EXECUTION["spatial_workers"],
    })
    timer = NestedTimer(array_threshold_bytes=array_threshold_bytes) if enabled else None

    def factory(frozen_propagator):
        propagator, dispatch = base_factory(frozen_propagator)
        execution = dict(getattr(propagator, "_cpu_o1_execution", {}))
        expected = {**FIXED_EXECUTION, "kernel_threads": kernel_threads}
        if any(execution.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"S4 fixed I execution contract mismatch: {execution}")
        if enabled:
            returned_propagator = _profile_private_clone(
                propagator, dispatch, timer, operator_mode=operator_mode)
            reporter = _ProfileReceipt(timer, dispatch, propagator,
                                       operator_mode=operator_mode)
        else:
            returned_propagator = propagator
            reporter = dispatch
        factory.last_propagator = propagator
        factory.last_dispatch = dispatch
        factory.last_rate_workspace = getattr(base_factory, "last_rate_workspace", None)
        factory.profile_receipt = reporter
        return returned_propagator, reporter

    factory.last_propagator = None
    factory.last_dispatch = None
    factory.last_rate_workspace = None
    factory.profile_receipt = None
    factory.profile_timer = timer
    return factory


def _memory_snapshot():
    snapshot = {"rss_current_bytes": None, "rss_current_status": "UNAVAILABLE",
                "process_lifetime_peak_rss_bytes": None,
                "process_lifetime_peak_rss_status": "UNAVAILABLE",
                "process_thread_count": None, "python_thread_count": int(threading.active_count())}
    try:
        import psutil
        process = psutil.Process()
        memory = process.memory_info()
        snapshot["rss_current_bytes"] = int(memory.rss)
        snapshot["rss_current_status"] = "MEASURED_ENDPOINT"
        snapshot["process_thread_count"] = int(process.num_threads())
        snapshot["cpu_affinity"] = list(process.cpu_affinity())
        if hasattr(memory, "peak_wset"):
            snapshot["process_lifetime_peak_rss_bytes"] = int(memory.peak_wset)
            snapshot["process_lifetime_peak_rss_status"] = "WINDOWS_PROCESS_PEAK_WORKING_SET_NOT_MODE_SPECIFIC"
    except Exception as exc:
        snapshot["rss_current_reason"] = f"{type(exc).__name__}: {exc}"
    try:
        import resource
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes. S4 real entry is Linux-only.
        if sys.platform.startswith("linux"):
            peak *= 1024
        snapshot["process_lifetime_peak_rss_bytes"] = peak
        snapshot["process_lifetime_peak_rss_status"] = "PROCESS_LIFETIME_HIGH_WATER_NOT_MODE_SPECIFIC"
    except Exception as exc:
        if snapshot["process_lifetime_peak_rss_bytes"] is None:
            snapshot["process_lifetime_peak_rss_reason"] = f"{type(exc).__name__}: {exc}"
    snapshot["thread_environment"] = {key: os.environ.get(key) for key in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS",
        "NUMBA_NUM_THREADS", "NUMBA_CPU_NAME", "NUMBA_THREADING_LAYER")}
    return snapshot


def _validate_profile_receipt(payload, *, kernel_threads=24):
    if not isinstance(payload, dict) or payload.get("mode") != "PROFILE":
        raise RuntimeError("missing or invalid S4 PROFILE receipt")
    if payload.get("implementation") != "I" or any(
            payload.get("execution", {}).get(k) != v for k, v in
            {**FIXED_EXECUTION, "kernel_threads": kernel_threads}.items()):
        raise RuntimeError("PROFILE execution identity mismatch")
    timer = payload.get("nested_timers", {})
    records = timer.get("records", [])
    covered = {r.get("interval_index") for r in records if r.get("calls", 0) > 0}
    if not set(range(8)).issubset(covered):
        raise RuntimeError("PROFILE must cover all eight intervals")
    if timer.get("failures", 0) or timer.get("cross_thread_overlapping_profile_calls_observed", 0):
        raise RuntimeError("PROFILE timer failed or Python callers overlapped")
    if not any(r.get("label") == "propagate.propagate_one_pulse" and r.get("calls", 0) for r in records):
        raise RuntimeError("PROFILE root timing missing")
    if not payload.get("ionization_candidate_events"):
        raise RuntimeError("PROFILE candidate events missing")
    if payload.get("schema") == SCHEMA:
        _validate_timing_contract(payload, records)
    return sorted(x for x in covered if x is not None)


def _validate_timing_contract(payload, records):
    contract = payload.get("timing_contract")
    if not isinstance(contract, dict) or contract.get("schema") != TIMING_CONTRACT_SCHEMA:
        raise RuntimeError("PROFILE source-bound timing contract missing")
    if contract.get("raman_mode") != "full_isaacs_eq27":
        raise RuntimeError("PROFILE timing contract has the wrong Raman mode")
    expected_path = ["isaacs_raman_stage", "apply_isaacs_raman_operator_step"]
    if contract.get("active_path") != expected_path:
        raise RuntimeError("PROFILE active Raman path differs from the frozen contract")
    if contract.get("inactive_complete_path") != "isaacs_complete_eq27_stage_not_instrumented":
        raise RuntimeError("PROFILE complete Raman path classification is missing")
    if (contract.get("inactive_complete_runtime_count", "missing") is not None
            or contract.get("inactive_complete_evidence") != "excluded_by_frozen_mode_and_source_not_timed"):
        raise RuntimeError("PROFILE inactive complete path must not claim measured calls")
    counts = contract.get("active_function_call_counts")
    if (not isinstance(counts, dict)
            or counts.get("isaacs_raman_stage", 0) <= 0
            or counts.get("apply_isaacs_raman_operator_step", 0) <= 0
            or set(counts) != set(expected_path)):
        raise RuntimeError("PROFILE did not observe the expected active Raman functions")
    if any("complete_eq27" in str(row.get("label", ""))
           for row in records if row.get("calls", 0)):
        raise RuntimeError("PROFILE unexpectedly timed the inactive complete Raman path")
    support_counts = contract.get("active_support_function_call_counts")
    if (not isinstance(support_counts, dict)
            or support_counts.get("stable_field_fluence_and_loss", 0) <= 0):
        raise RuntimeError("PROFILE did not observe the active stable Raman energy helper")
    stage_roles = contract.get("stage_call_roles")
    if (not isinstance(stage_roles, dict)
            or stage_roles.get("stage1_supplied", 0) <= 0
            or stage_roles.get("stage2_recomputed", 0) <= 0
            or sum(int(value) for value in stage_roles.values()) != counts["isaacs_raman_stage"]):
        raise RuntimeError("PROFILE stage1/stage2 Raman response coverage is incomplete")
    reuse = contract.get("stage1_reused_counts")
    if (not isinstance(reuse, dict) or reuse.get("supplied", 0) != counts[
            "apply_isaacs_raman_operator_step"]
            or reuse.get("recomputed_inside_operator") != 0):
        raise RuntimeError("PROFILE did not preserve caller-supplied stage1 reuse")

    bindings = contract.get("source_bindings")
    expected_binding_functions = {*expected_path, "stable_field_fluence_and_loss"}
    if not isinstance(bindings, list) or len(bindings) != len(expected_binding_functions):
        raise RuntimeError("PROFILE source binding inventory is incomplete")
    by_function = {item.get("function"): item for item in bindings if isinstance(item, dict)}
    if set(by_function) != expected_binding_functions:
        raise RuntimeError("PROFILE source binding function set is invalid")
    expected_locations = {
        "isaacs_raman_stage": {
            "raman.stage.intensity_generation", "raman.stage.response_field_product",
            "raman.stage.spectral_derivative_output", "raman.stage.rhs_output",
        },
        "apply_isaacs_raman_operator_step": {
            "raman.heun.predictor_update", "raman.heun.stage2_recomputed",
            "raman.heun.corrector_update",
        },
        "stable_field_fluence_and_loss": {
            "raman.energy.before_real.float64_cast",
            "raman.energy.after_real.float64_cast",
            "raman.energy.before_imag.float64_cast",
            "raman.energy.after_imag.float64_cast",
        },
    }
    from tools.hpc_ops.provenance_v2 import canonical_lf_sha256
    from tools.cpu_o1.cpu_o1 import P1R2_SOURCE_BLOBS
    import subprocess
    repo_root = Path(__file__).resolve().parents[4]
    expected_blob = P1R2_SOURCE_BLOBS["Filament_python/KHz_filament/raman.py"]
    expected_source = subprocess.run(["git", "cat-file", "blob", expected_blob],
        cwd=repo_root, capture_output=True, check=False)
    if expected_source.returncode != 0:
        raise RuntimeError("PROFILE frozen Raman source blob cannot be read")
    expected_sha = canonical_lf_sha256(expected_source.stdout)
    for name, binding in by_function.items():
        source_path = Path(str(binding.get("source_path", "")))
        source_sha = binding.get("source_canonical_lf_sha256")
        if (binding.get("frozen_git_blob_oid") != expected_blob
                or source_sha != expected_sha
                or not source_path.is_file()
                or canonical_lf_sha256(source_path.read_bytes()) != expected_sha):
            raise RuntimeError(f"PROFILE frozen source binding mismatch: {name}")
        locations = binding.get("statement_locations")
        if not isinstance(locations, list) or {
                item.get("operation") for item in locations if isinstance(item, dict)
        } != expected_locations[name]:
            raise RuntimeError(f"PROFILE statement location set mismatch: {name}")
        if len(locations) != len(expected_locations[name]):
            raise RuntimeError(f"PROFILE duplicate statement locations: {name}")
        if binding.get("transformation") != "private_ast_timer_context_v1":
            raise RuntimeError(f"PROFILE source transformation mismatch: {name}")

    if sum(int(row.get("calls", 0)) for row in records
           if row.get("label") == "raman.stable_field_fluence_and_loss") != support_counts[
               "stable_field_fluence_and_loss"]:
        raise RuntimeError("PROFILE stable Raman energy helper count mismatch")

    labels = {str(row.get("label")): row for row in records if row.get("calls", 0)}
    for name, operations in expected_locations.items():
        for operation in operations:
            if not any(row.get("label") == operation and row.get("calls", 0) > 0
                       for row in records):
                raise RuntimeError(f"PROFILE statement timing missing: {operation}")
    array_report = payload.get("nested_timers", {}).get("array_accounting")
    threshold = contract.get("array_record_threshold_bytes")
    if (not isinstance(array_report, dict) or isinstance(threshold, bool)
            or not isinstance(threshold, int) or threshold < 1
            or array_report.get("threshold_bytes") != threshold
            or not isinstance(array_report.get("operations"), list)
            or not isinstance(array_report.get("omitted_below_threshold_calls"), dict)):
        raise RuntimeError("PROFILE array operation inventory is incomplete")
    for item in array_report["operations"]:
        if (not isinstance(item, dict) or not isinstance(item.get("calls"), int)
                or item["calls"] <= 0 or not isinstance(item.get("input_arrays"), list)
                or not isinstance(item.get("output_arrays"), list)
                or item.get("timing_scope") not in {"whole_statement", "whole_function"}
                or item.get("aggregation_warning") != (
                    "alternative_view_do_not_sum_with_nested_timers")
                or any(isinstance(item.get(name), bool)
                       or not isinstance(item.get(name), (int, float))
                       or not math.isfinite(item[name]) or item[name] < 0
                       for name in ("inclusive_wall_seconds", "exclusive_wall_seconds",
                                    "inclusive_process_cpu_seconds",
                                    "exclusive_process_cpu_seconds"))
                or any(not isinstance(array.get("shape"), list)
                       or not isinstance(array.get("dtype"), str)
                       or not isinstance(array.get("nbytes"), int)
                       for array in (*item["input_arrays"], *item["output_arrays"]))):
            raise RuntimeError("PROFILE array operation metadata is invalid")


def _run_one(context: dict, case_root: Path, *, profile: bool,
             case_id: str | None = None,
             first_boundary_reference: tuple[str, Path] | None = None):
    import numpy as np
    from tools.cpu_o1 import q1_t0
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1

    kernel_threads = int(context.get("_local_kernel_threads", 24))
    if context.get("real_input_used") and kernel_threads != 24:
        raise RuntimeError("real S4 requires 24 kernel threads")
    operator_mode = str(getattr(context.get("raman"), "operator_mode", ""))
    factory = make_profiled_i_factory(enabled=profile, kernel_threads=kernel_threads,
        operator_mode=operator_mode or "full_isaacs_eq27",
        array_threshold_bytes=int(context.get("_s4_array_threshold_bytes",
                                             ARRAY_RECORD_THRESHOLD_BYTES)))
    capture = {}
    wrapped_factory = q1_t0._scalar_capture_factory(factory, capture)
    source = context["source"]
    pre_fields = context.get("pre_fields")
    if pre_fields is None:
        from KHz_filament.hr4e5_formal_entry import pre0_fields_from_delta_n
        pre_fields = pre0_fields_from_delta_n(context["assets"]["pre0"], expected_count=8)
    before_source_hash = j1._canonical_hash(source)
    before_pre_hashes = {key: j1._canonical_hash(value)
                         for key, value in pre_fields.items()}
    memory_before = _memory_snapshot()
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    private_window = j1._run_window
    boundary_observer_type = None
    if case_id is not None:
        from tools.cpu_o1 import q1_t0_real
        propagation_source = Path(
            __import__("KHz_filament.propagate", fromlist=["propagate_one_pulse"]).__file__
        ).resolve(strict=True)
        diagnostic_contract = q1_t0_real._frozen_diag_list_contract(propagation_source)
        references = ([first_boundary_reference] if first_boundary_reference is not None
                      else [(case_id, case_root / "first_boundary")])
        boundary_observer_type = q1_t0_real._observer_class(
            j1, identity="I", case_id=case_id, case_root=case_root,
            comparison_boundary_roots=references,
            diagnostic_contract=diagnostic_contract)
        base_observer = boundary_observer_type
        class SealedBoundaryObserver(base_observer):
            def _callback(self, code, line):
                result = super()._callback(code, line)
                if self.q1_first_gate is not None and not hasattr(self, "s4_boundary_hashes"):
                    from tools.hpc_ops import provenance_v2 as pv
                    self.s4_boundary_hashes = {name: pv.raw_sha256_file(case_root / "first_boundary" / name)
                        for name in ("Qacc.npy", "Qacc_raman.npy")}
                return result
        boundary_observer_type = SealedBoundaryObserver
        private_window = q1_t0_real._private_run_window_with_observer(
            j1, boundary_observer_type)
    receipt = private_window(
        **{key: context[key] for key in (
            "source", "grid", "beam", "prop", "ion", "heat", "raman", "schedule")},
        pre_fields=pre_fields,
        output_dir=case_root / "propagation",
        source_label="S4_sealed_PRE0",
        initial_pre0=True,
        profile=False,
        source_hash=context.get("source_hash", before_source_hash),
        pre0_hash=context.get("assets", {}).get("pre0_canonical_sha256",
            context.get("pre0_hash")),
        stage_name="S4_I_PROFILE" if profile else "S4_I_OFF",
        execution_scope=("S4_REAL_INPUT_PROFILE" if context.get("real_input_used")
                         else "S4_LOCAL_TINY_PROFILE"),
        observe=True,
        propagator_factory=wrapped_factory,
    )
    wall_seconds = time.perf_counter() - wall_start
    process_cpu_seconds = time.process_time() - cpu_start
    observer_rows = list(receipt.get("interval_rows", ()))
    if len(observer_rows) != 8 or [int(row.get("interval_index", -1))
                                  for row in observer_rows] != list(range(8)):
        raise RuntimeError("S4 run did not observe all eight accepted intervals")
    first_boundary_gate = None
    if boundary_observer_type is not None:
        observer = boundary_observer_type.last_instance
        if (observer is None or observer.q1_first_boundary is None
                or observer.q1_first_gate is None
                or observer.q1_first_gate.get("proceed_to_next_identity") is not True):
            raise RuntimeError("S4 first accepted-boundary exact gate did not pass")
        first_boundary_gate = {
            "case_id": case_id,
            "implementation": "I",
            "classification": observer.q1_first_gate.get("classification"),
            "proceed_to_next_identity": True,
            "receipt_path": str((case_root / "first_boundary_receipt.json").resolve()),
            "snapshot": observer.q1_first_boundary,
            "gate": observer.q1_first_gate,
            "capture_hashes": observer.s4_boundary_hashes,
        }
    propagation_wall_raw = float(receipt["propagation_wall_seconds"])
    first_boundary_cross_seconds = float(
        (first_boundary_gate or {}).get("gate", {}).get(
            "cross_identity_comparison_seconds", 0.0)) if profile else 0.0
    if (not math.isfinite(propagation_wall_raw)
            or not math.isfinite(first_boundary_cross_seconds)
            or propagation_wall_raw < 0 or first_boundary_cross_seconds < 0
            or first_boundary_cross_seconds > propagation_wall_raw):
        raise RuntimeError("S4 propagation/cross-window timing boundary is invalid")
    propagation_wall_adjusted = propagation_wall_raw - first_boundary_cross_seconds
    for index, row in enumerate(observer_rows):
        row["s4_interval_wall_seconds"] = float(row["interval_wall_seconds"])
        row["s4_interval_process_cpu_seconds"] = float(row["interval_process_cpu_seconds"])
        row["profiled_call_count"] = sum(
            item["calls"] for item in (factory.profile_timer.report()["records"]
                if profile else ()) if item["interval_index"] == index)
    if not {"qacc", "payload", "values"}.issubset(capture):
        raise RuntimeError("S4 run lost the private Q1 scalar/Qacc return capture")
    q1_t0._write_new_array(case_root / "scientific_Qacc.npy", capture["qacc"])
    q1_t0._write_scalar_diagnostics(
        case_root / "scientific_scalar_diagnostics.json", "I", capture["payload"])
    dispatch_check = q1_t0._check_identity_dispatch("I", factory,
        kernel_threads=kernel_threads,
        temporal_workers=FIXED_EXECUTION["temporal_workers"],
        spatial_workers=FIXED_EXECUTION["spatial_workers"])
    events = list(getattr(factory.last_propagator, "_cpu_o1_candidate_events", ()))
    _strict_jsonl(case_root / "candidate_events.jsonl", events)
    if dispatch_check["candidate_event_count"] <= 0:
        raise RuntimeError("S4 I execution did not invoke the compiled ionization candidate")
    if j1._canonical_hash(source) != before_source_hash:
        raise RuntimeError("S4 changed the sealed source input")
    after_pre_hashes = {key: j1._canonical_hash(value)
                        for key, value in pre_fields.items()}
    if after_pre_hashes != before_pre_hashes:
        raise RuntimeError("S4 changed a sealed PRE input field")
    memory_after = _memory_snapshot()
    profile_path = case_root / "propagation" / "function_fft_profile.json"
    profile_data = (json.loads(profile_path.read_text(encoding="utf-8"))
                    if profile_path.is_file() else None)
    if profile:
        _validate_profile_receipt(profile_data, kernel_threads=kernel_threads)
        if not all(row["profiled_call_count"] > 0 for row in observer_rows):
            raise RuntimeError("PROFILE interval call coverage missing")
    return {
        "mode": "PROFILE" if profile else "OFF",
        "status": "COMPLETE",
        "window_call_wall_seconds": float(wall_seconds),
        "window_timing_boundary": "J1 _run_window entry-to-return, observer-inclusive (snapshot IO and validation); normalized propagation subtracts PROFILE cross-window comparison only; excludes source/PRE prehash, scalar/Qacc sidecars, process startup/loading/exit",
        "full_service_wall_seconds": None,
        "full_service_status": "NOT_MEASURED_IN_SAME_PROCESS_LOCAL_PAIR",
        "propagation_wall_seconds": propagation_wall_adjusted,
        "propagation_wall_seconds_raw_including_cross_window_comparison": propagation_wall_raw,
        "propagation_cross_window_comparison_seconds": first_boundary_cross_seconds,
        "propagation_process_cpu_seconds": float(receipt["propagation_process_cpu_seconds"]),
        "process_cpu_seconds_including_outputs": float(process_cpu_seconds),
        "interval_rows": observer_rows,
        "interval_wall_seconds_sum": sum(float(row["interval_wall_seconds"])
                                          for row in observer_rows),
        "interval_process_cpu_seconds_sum": sum(float(row["interval_process_cpu_seconds"])
                                                for row in observer_rows),
        "dispatch": dispatch_check,
        "profile": profile_data,
        "candidate_event_count": len(events),
        "input_source_canonical_sha256": before_source_hash,
        "input_pre_canonical_sha256": before_pre_hashes,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "peak_thread_count": None,
        "peak_thread_count_status": "NOT_SAMPLED_TO_AVOID_A_MONITOR_THREAD",
        "science_source_globals_modified": False,
        "receipt": receipt,
        "first_boundary_gate": first_boundary_gate,
        "output_root": str(case_root.resolve()),
        "numpy_version": np.__version__,
    }


def run_validated_context_pair(context: dict, output_root: str | os.PathLike, *,
                               preflight_receipt: dict | None = None):
    """Run frozen I OFF then PROFILE for an already validated eight-interval context."""
    from tools.cpu_o1 import q1_t0
    from tools.cpu_o1.numeric_gate import classify_comparison
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1

    root = Path(output_root).resolve(strict=False)
    if root.exists():
        raise FileExistsError(f"S4 output root must be new: {root}")
    if int(getattr(context.get("schedule"), "n_intervals", -1)) != 8:
        raise ValueError("S4 accepts only a complete eight-interval validated context")
    if str(getattr(context.get("raman"), "operator_mode", "")) != "full_isaacs_eq27":
        raise ValueError("S4 requires the frozen full_isaacs_eq27 Raman path")
    ion = context.get("ion")
    if str(getattr(ion, "time_mode", "")) != "full" or str(
            getattr(ion, "integrator", "")).lower() != "rk4":
        raise ValueError("S4 requires full-time RK4 ionization")
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    modes = {}
    if not context.get("real_input_used"):
        from tools.cpu_o1.cpu_o1 import available_numba_threads
        context = {**context, "_local_kernel_threads": min(24, available_numba_threads(), os.cpu_count() or 1)}
    try:
        modes["off"] = _run_one(context, root / "off", profile=False)
        modes["profile"] = _run_one(context, root / "profile", profile=True)
        comparison_path = root / "comparison.jsonl"
        rows = q1_t0._compare_window_outputs(
            root / "off", root / "profile", "I",
            comparison_path=comparison_path,
            scalar_reference=_read_scalar_fields(root / "off" / "scientific_scalar_diagnostics.json"),
            scalar_candidate=_read_scalar_fields(root / "profile" / "scientific_scalar_diagnostics.json"),
            case_id="PROFILE", reference_case_id="OFF",
            comparison_stage="s4_off_profile")
        classifications = [classify_comparison(row) for row in rows]
        exact = bool(rows) and all(row.get("classification") in {
            "CURRENT_LINUX_LOCAL_EXACT", "TIMING_EXCLUDED"} for row in classifications)
        local_classifications = [{
            **row,
            "frozen_classification": row.get("classification"),
            "classification": (
                "LOCAL_EXACT_FROZEN_CONTRACT"
                if row.get("classification") == "CURRENT_LINUX_LOCAL_EXACT"
                else row.get("classification")),
        } for row in classifications]
        comparison = {
            "schema": "filament.cpu_s4.off_profile_comparison.v1",
            "scope": "LOCAL_TINY_SYNTHETIC_ONLY" if not context.get("real_input_used")
                     else "REAL_INPUT_DIAGNOSTIC_ONLY",
            "status": "EXACT" if exact else "BLOCKED_NUMERIC_GATE",
            "numeric_acceptance_tolerance_authorized": False,
            "rows": local_classifications,
            "raw_comparison_jsonl": str(comparison_path),
            "raw_row_count": len(rows),
            "timing_fields_use_frozen_exclusion_list": True,
            "nan_placeholders_use_frozen_numeric_gate_whitelist": True,
        }
        _strict_json(root / "comparison_gate.json", comparison)
        _strict_json(root / "S4_PROFILE_SUMMARY.json", {
            "schema": SCHEMA,
            "status": "LOCAL_PASS" if exact else "NUMERIC_DIAGNOSTIC_ONLY",
            "production_qualification": "NOT_PRODUCTION_QUALIFIED",
            "hpc_submission": "NOT_SUBMITTED",
            "source_binding": _json_safe(context.get("source_binding")),
            "input_identity": {
                "source_canonical_sha256": modes["off"]["input_source_canonical_sha256"],
                "pre_canonical_sha256": modes["off"]["input_pre_canonical_sha256"],
            },
            "fixed_execution": FIXED_EXECUTION,
            "actual_kernel_threads": context.get("_local_kernel_threads", 24),
            "local_thread_cap_only": not bool(context.get("real_input_used")),
            "preflight": _json_safe(preflight_receipt),
            "modes": {key: {item: value for item, value in row.items()
                if item not in {"receipt", "profile"}}
                for key, row in modes.items()},
            "profile_receipt_paths": {
                "OFF": str(root / "off" / "propagation" / "function_fft_profile.json"),
                "PROFILE": str(root / "profile" / "propagation" / "function_fft_profile.json"),
            },
            "comparison_gate": comparison,
            "elapsed_wall_seconds_including_comparison": time.perf_counter() - started,
            "interpretation": "tiny software evidence only; full real optical service and instrumentation overhead remain unmeasured",
        })
        if not exact:
            raise RuntimeError("S4 OFF/PROFILE scientific comparison did not pass the frozen exact gate")
        return {"status": "LOCAL_PASS", "output_root": str(root),
                "summary": str(root / "S4_PROFILE_SUMMARY.json"),
                "comparison": comparison, "modes": modes}
    except BaseException as exc:
        failure_path = root / "S4_FAILURE.json"
        if not failure_path.exists():
            _strict_json(failure_path, {
                "schema": SCHEMA, "status": "FAIL",
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(),
                "modes_completed": sorted(modes),
                "hpc_submission": "NOT_SUBMITTED",
            })
        raise


def _read_scalar_fields(path: Path):
    import numpy as np
    payload = json.loads(path.read_text(encoding="utf-8"))
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        raise ValueError(f"invalid Q1 scalar sidecar: {path}")
    result = {}
    for name, item in fields.items():
        value = item.get("value")
        kind = item.get("dtype", "")
        if kind == "bool":
            result[name] = value
        elif isinstance(value, dict) and set(value) == {"real", "imag"}:
            result[name] = complex(float.fromhex(value["real"]), float.fromhex(value["imag"]))
        elif isinstance(value, str) and value in {"NaN", "+Infinity", "-Infinity"}:
            result[name] = {"NaN": float("nan"), "+Infinity": float("inf"),
                            "-Infinity": float("-inf")}[value]
        elif kind.startswith("float"):
            result[name] = float.fromhex(value)
        elif kind.startswith("complex"):
            result[name] = complex(float.fromhex(value["real"]), float.fromhex(value["imag"]))
        elif kind.startswith("int") or kind.startswith("uint"):
            result[name] = int(value)
        else:
            result[name] = value
    # Reconstruct the recorded dtype, including signed zero, instead of silently
    # widening float32/int32 scalar leaves to Python float/int.
    return {name: np.asarray(value, dtype=fields[name]["dtype"])[()]
            for name, value in result.items()}


def _load_real_context(args):
    """Reuse Q1's manifest, environment, source and frozen asset validators."""
    from tools.cpu_o1 import q1_t0_real
    from tools.cpu_p1r import cpu_p1r_optical as p1r
    from tools.hpc_ops import provenance_v2 as pv

    if args.output_root.exists():
        raise FileExistsError(f"S4 output root must be new: {args.output_root}")
    if not q1_t0_real._resolved_under(args.output_root.parent, args.storage_root):
        raise ValueError("S4 output-root must be inside the audited storage-root")
    q1_t0_real._check_expected_python(args.expected_python)
    q1_t0_real._validate_storage_caps(args.storage_cap_bytes, args.max_files)
    storage = q1_t0_real._check_storage_growth(args.storage_root,
        growth_cap_bytes=args.storage_cap_bytes, max_files=args.max_files)
    linux = importlib.import_module("tools.cpu_o1.linux_q1_preflight").run_linux_q1_preflight()
    if not linux.get("eligible"):
        raise RuntimeError("Q1 Linux runtime preflight failed: " + json.dumps(
            _json_safe(linux.get("reasons", [])), allow_nan=False))
    manifest = pv.validate_manifest(args.bench_root, args.provenance,
        require_clean=True, require_hash_scope=True)
    external_paths = q1_t0_real._validate_external_input_set(manifest, args.input_root)
    expected_binding = q1_t0_real._load_expected_source_binding(
        args.expected_source_binding)
    actual_binding = p1r._source_binding(args.bench_root.resolve(strict=True))
    if actual_binding != expected_binding:
        raise RuntimeError("S4 current source binding differs from the reviewed Q1 identity")
    incoming_gpu = os.environ.get("UPPE_USE_GPU")
    if incoming_gpu not in (None, "0", "false", "False"):
        raise RuntimeError("S4 real profile requires UPPE_USE_GPU=0")
    if os.environ.get("NUMBA_THREADING_LAYER") != "workqueue":
        raise RuntimeError("S4 real profile requires NUMBA_THREADING_LAYER=workqueue")
    if os.environ.get("NUMBA_CPU_NAME") != "generic":
        raise RuntimeError("S4 real profile requires NUMBA_CPU_NAME=generic")
    if os.environ.get("NUMBA_NUM_THREADS") != "24":
        raise RuntimeError("S4 real profile requires NUMBA_NUM_THREADS=24")
    for name in q1_t0_real.BLAS_THREAD_VARIABLES:
        if os.environ.get(name) != "1":
            raise RuntimeError(f"S4 real profile requires {name}=1")
    import threadpoolctl
    native_limit = threadpoolctl.threadpool_limits(limits=1)
    context = p1r._load_context(args.input_root.resolve(strict=True),
                                args.bench_root.resolve(strict=True),
                                args.vendor_dir.resolve(strict=True))
    if context.get("source_binding") != expected_binding:
        raise RuntimeError("S4 loaded context source binding changed after Q1 validation")
    try:
        import numba
        if numba.config.THREADING_LAYER != "workqueue" or int(
                numba.config.NUMBA_NUM_THREADS) < 24:
            raise RuntimeError("S4 Numba workqueue/thread ceiling does not meet the fixed plan")
        numba.set_num_threads(24)
        if int(numba.get_num_threads()) != 24 or numba.threading_layer() != "workqueue":
            raise RuntimeError("S4 actual Numba layer/thread count differs from 24/workqueue")
    except Exception as exc:
        raise RuntimeError(f"S4 Numba runtime qualification failed: {exc}") from exc
    context["real_input_used"] = True
    context["pre0_hash"] = context["assets"]["pre0_canonical_sha256"]
    context["pre_fields"] = __import__(
        "KHz_filament.hr4e5_formal_entry", fromlist=["pre0_fields_from_delta_n"]
    ).pre0_fields_from_delta_n(context["assets"]["pre0"], expected_count=8)
    return context, {
        "status": "Q1_CONTEXT_AND_LINUX_PREFLIGHT_VALIDATED",
        "linux_environment": _json_safe(linux),
        "source_binding": _json_safe(actual_binding),
        "input_hashes": {
            "source_canonical_sha256": context["source_hash"],
            "pre0_canonical_sha256": context["assets"]["pre0_canonical_sha256"],
            "external_input_count": len(external_paths),
        },
        "storage_inventory": _json_safe(storage),
        "native_threadpools": _json_safe(threadpoolctl.threadpool_info()),
        "native_thread_limit_held": native_limit is not None,
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tiny", action="store_true",
                      help="run only the deterministic local eight-interval fixture")
    mode.add_argument("--real-pair", action="store_true",
                      help="run Q1-preflighted real I OFF then PROFILE; no scheduler submission")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="new output directory; existing paths are refused")
    parser.add_argument("--allow-real-optical-execution", action="store_true",
                        help="required explicit acknowledgement for real eight-interval execution")
    for name in ("bench-root", "input-root", "vendor-dir", "provenance",
                 "expected-source-binding", "expected-python", "storage-root"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--storage-cap-bytes", type=int)
    parser.add_argument("--max-files", type=int, default=50000)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.tiny:
        receipt = run_tiny_off_profile(args.output_root)
        print(json.dumps({"status": receipt["status"], "output_root": receipt["output_root"],
                          "hpc_submission": "NOT_SUBMITTED"}, allow_nan=False))
        return 0
    if not args.allow_real_optical_execution:
        raise SystemExit("--real-pair requires --allow-real-optical-execution")
    required = ("bench_root", "input_root", "vendor_dir", "provenance",
                "expected_source_binding", "expected_python", "storage_root",
                "storage_cap_bytes")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit("--real-pair missing: " + ", ".join(missing))
    context, preflight = _load_real_context(args)
    receipt = run_validated_context_pair(context, args.output_root,
                                         preflight_receipt=_json_safe(preflight))
    print(json.dumps({"status": receipt["status"], "output_root": receipt["output_root"],
                      "hpc_submission": "NOT_SUBMITTED"}, allow_nan=False))
    return 0


def run_tiny_off_profile(output_root: str | os.PathLike):
    """Run only the deterministic eight-interval local fixture in both modes."""
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1

    context = j1._tiny_context(operator_mode="full_isaacs_eq27")
    context["real_input_used"] = False
    context["source_hash"] = j1._canonical_hash(context["source"])
    context["assets"] = {"pre0": context["pre0"],
        "pre0_canonical_sha256": j1._canonical_hash(context["pre0"])}
    return run_validated_context_pair(context, output_root)


if __name__ == "__main__":  # pragma: no cover - exercised by the runner smoke
    raise SystemExit(main())
