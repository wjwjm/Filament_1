#!/usr/bin/env python3
"""Bounded J2 CPU FFT-dispatch diagnostic; never production-qualified."""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib
import json
import math
import statistics
import os
from pathlib import Path
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

SCHEMA = "filament.cpu_p1r2.j2_fft_dispatch_diagnostic.v1"
PHASE = "J2"
CLASSIFICATION = "P1R2_J2_DIAGNOSTIC_ONLY"
QUALIFICATION = "NOT_PRODUCTION_QUALIFIED"
STAGES = ("capture", "baseline", "microbench", "candidate-a", "candidate-b")
WINDOW_SECONDS = 2700
B0_SECONDS = 900
RESERVE_SECONDS = 900
ALLOCATION_SECONDS = 7200
MEMORY_BYTES = 320 * 1024**3
MAX_CAPTURE_SPAN = 1024**3
TIMING_FIELDS = j1.TIMING_DIAGNOSTIC_FIELDS
DISCRETE_FIELDS = frozenset(("z_axis", "dz_used_z", "adaptive_rejection_count_z",
    "safety_mode_trigger_count_z", "raman_operator_applied",
    "raman_convolution_count_step", "raman_operator_substep_count",
    "raman_energy_projection_iterations"))
TRANSFORMS = frozenset(("fft", "ifft", "fft2", "ifft2", "fftn", "ifftn",
    "rfft", "irfft", "rfft2", "irfft2", "rfftn", "irfftn"))


class J2Error(RuntimeError):
    pass


def _json_new(path: Path, value) -> None:
    j1._json_new(path, value)


def _sha_array(value) -> str:
    return j1._canonical_hash(value)


def _validate_descriptor(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {"temporal_workers", "spatial_workers"}:
        raise J2Error("candidate descriptor must contain temporal_workers and spatial_workers")
    result = {}
    for key, raw in value.items():
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise J2Error("SciPy FFT worker counts must be integer values")
        result[key] = int(raw)
    if any(v not in {1, 8, 24, 48, 96} for v in result.values()):
        raise J2Error("SciPy FFT workers must be one of 1/8/24/48/96")
    return result


def _worker_group(name: str, args, kwargs) -> str:
    if name in ("fft2", "ifft2", "rfft2", "irfft2"):
        return "spatial"
    axes = kwargs.get("axes")
    if axes is not None and len(tuple(axes)) > 1:
        return "spatial"
    return "temporal"


class _CaptureDispatch:
    """NumPy-delegating capture shim; arrays are retained only until full-window success."""
    def __init__(self, numpy_fft):
        self.numpy_fft = numpy_fft
        self.arrays = {}
        self.layouts = defaultdict(dict)

    def __getattr__(self, name):
        target = getattr(self.numpy_fft, name)
        if name not in TRANSFORMS or not callable(target):
            return target
        def call(*args, **kwargs):
            import numpy as np
            value = args[0] if args else kwargs.get("a", kwargs.get("x"))
            if isinstance(value, np.ndarray):
                frame = sys._getframe(1)
                caller = f"{frame.f_globals.get('__name__', '')}.{frame.f_code.co_name}"
                interval_index = _current_propagate_interval(frame)
                layout = {"shape": list(value.shape), "dtype": str(value.dtype),
                    "strides": list(value.strides), "callsite": caller,
                    "source_file": frame.f_code.co_filename,
                    "source_line": int(frame.f_lineno),
                    "observed_interval_index": interval_index}
                bucket = self.layouts[f"{name}:{value.dtype}:{value.ndim}"]
                signature = json.dumps(layout, sort_keys=True)
                if signature in bucket:
                    bucket[signature]["calls"] += 1
                else:
                    bucket[signature] = {**layout, "calls": 1}
                key = _capture_class(name, value)
                if key is not None and key not in self.arrays and interval_index == 0:
                    if any(s <= 0 for s in value.strides):
                        raise J2Error("capture requires strictly positive strides")
                    span = value.dtype.itemsize + sum((n - 1) * s
                        for n, s in zip(value.shape, value.strides))
                    if span > MAX_CAPTURE_SPAN:
                        raise J2Error(f"capture backing span exceeds 1 GiB: {span}")
                    spec = {**layout, "transform": name,
                        "call_args_tail": [_json_scalar(v) for v in args[1:]],
                        "call_kwargs": {str(k): _json_scalar(v) for k, v in kwargs.items()},
                        "backing_span_bytes": int(span)}
                    spec["compact_values"] = np.array(value, copy=True, order="C")
                    self.arrays[key] = spec
            return target(*args, **kwargs)
        return call

    def report(self):
        return {"capture_classes": sorted(self.arrays),
            "observed_layouts": {k: list(v.values()) for k, v in self.layouts.items()},
            "capture_published_only_after_complete_window": True}


def _capture_class(name, value):
    import numpy as np
    if name == "fft" and value.ndim == 3 and value.dtype == np.dtype("complex128"):
        return "fft_complex3"
    if name == "fft" and value.ndim == 3 and value.dtype == np.dtype("float64"):
        return "fft_float3"
    if name == "ifft" and value.ndim == 3 and value.dtype == np.dtype("complex128"):
        return "ifft_complex3"
    if name in ("fft2", "ifft2") and value.ndim == 2 and value.dtype == np.dtype("complex128"):
        return f"{name}_plane_complex"
    return None


def _current_propagate_interval(frame):
    cursor = frame
    while cursor is not None:
        if (cursor.f_code.co_name == "propagate_one_pulse"
                and cursor.f_code.co_filename.replace("\\", "/").endswith("/KHz_filament/propagate.py")):
            value = cursor.f_locals.get("interval_index")
            return int(value) if isinstance(value, int) else None
        cursor = cursor.f_back
    return None


class _ScipyDispatch:
    def __init__(self, numpy_fft, scipy_fft, descriptor):
        self.numpy_fft = numpy_fft
        self.scipy_fft = scipy_fft
        self.descriptor = _validate_descriptor(descriptor)
        self.counts = defaultdict(int)

    def __getattr__(self, name):
        if name not in TRANSFORMS:
            return getattr(self.numpy_fft, name)
        target = getattr(self.scipy_fft, name, None)
        if target is None:
            raise J2Error(f"scipy.fft lacks transform {name}")
        def call(*args, **kwargs):
            group = _worker_group(name, args, kwargs)
            self.counts[f"{group}:{name}"] += 1
            options = dict(kwargs)
            options["workers"] = self.descriptor[f"{group}_workers"]
            options["overwrite_x"] = False
            return target(*args, **options)
        return call

    def report(self):
        return {"candidate": "scipy.fft", "workers": dict(self.descriptor),
            "transform_counts": dict(self.counts),
            "global_numpy_fft_modified": False,
            "global_linear_raman_xp_modified": False}


class _FFTNamespace:
    def __init__(self, dispatch):
        self.dispatch = dispatch
    def __getattr__(self, name):
        return getattr(self.dispatch, name)


class _XPNamespace:
    def __init__(self, xp, dispatch):
        self.xp, self.fft = xp, _FFTNamespace(dispatch)
    def __getattr__(self, name):
        return self.fft if name == "fft" else getattr(self.xp, name)


def _clone_propagator(dispatch):
    """Wire only cloned frozen linear/Raman symbols; leave science modules untouched."""
    import inspect
    from types import FunctionType
    linear = importlib.import_module("KHz_filament.linear")
    raman = importlib.import_module("KHz_filament.raman")
    propagate = importlib.import_module("KHz_filament.propagate")
    from KHz_filament.device import xp
    proxy = _XPNamespace(xp, dispatch)
    linear_ns = j1._clone_module_functions(linear, proxy, None, set())
    raman_ns = j1._clone_module_functions(raman, proxy, None, set())
    source = propagate.propagate_one_pulse
    private = dict(source.__globals__)
    replacements = {name: linear_ns[name] for name in
        ("step_linear_bk_nee_factorized", "step_linear", "lin_propagator")
        if name in linear_ns}
    for name in ("apply_isaacs_raman_operator_step", "apply_isaacs_complete_eq27_operator_step",
        "historical_fr_mixture_response", "isaacs_raman_stage", "isaacs_complete_eq27_stage",
        "make_raman_kernel", "precompute_kernel_fft", "raman_convolve_intensity",
        "raman_convolve_intensity_fft_linear", "resolve_raman_rot_params"):
        if name in raman_ns:
            replacements[name] = raman_ns[name]
    private.update(replacements)
    clone = FunctionType(source.__code__, private, source.__name__, source.__defaults__, source.__closure__)
    clone.__kwdefaults__ = source.__kwdefaults__
    clone.__annotations__ = source.__annotations__
    clone.__module__, clone.__qualname__ = source.__module__, source.__qualname__
    clone.__j1_inner_code__ = source.__code__
    if linear.xp is not xp or raman.xp is not xp:
        raise J2Error("private FFT dispatch unexpectedly changed a science module xp")
    return clone


def make_propagator_factory(descriptor: dict):
    """Factory API: descriptor -> (fresh cloned propagate callable, dispatch report)."""
    selected = _validate_descriptor(descriptor)
    def factory(_frozen_propagator):
        import scipy.fft
        from KHz_filament.device import xp
        dispatch = _ScipyDispatch(xp.fft, scipy.fft, selected)
        return _clone_propagator(dispatch), dispatch
    return factory


def make_capture_factory(collector: _CaptureDispatch | None = None):
    holder = collector if collector is not None else {"dispatch": None}
    def factory(_frozen_propagator):
        from KHz_filament.device import xp
        dispatch = _CaptureDispatch(xp.fft)
        if isinstance(holder, dict):
            holder["dispatch"] = dispatch
        else:
            holder.__dict__.update(dispatch.__dict__)
            dispatch = holder
        return _clone_propagator(dispatch), dispatch
    return factory


def _backing_span(shape, strides, itemsize):
    if not shape or len(shape) != len(strides) or any(int(s) <= 0 for s in strides):
        raise J2Error("layout restore requires nonempty shape and positive strides")
    return int(itemsize + sum((int(n) - 1) * int(s) for n, s in zip(shape, strides)))


def restore_layout(compact, *, shape, dtype, strides, expected_hash=None):
    """Recreate original positive strides over a fresh same-sized backing allocation."""
    import numpy as np
    value = np.asarray(compact)
    dtype = np.dtype(dtype)
    shape, strides = tuple(map(int, shape)), tuple(map(int, strides))
    if value.shape != shape or value.dtype != dtype:
        raise J2Error("captured logical values do not match stored shape/dtype")
    span = _backing_span(shape, strides, dtype.itemsize)
    if span > MAX_CAPTURE_SPAN:
        raise J2Error("layout restore backing span exceeds 1 GiB")
    backing = np.empty(span, dtype=np.uint8)
    restored = np.ndarray(shape, dtype=dtype, buffer=backing, strides=strides)
    restored[...] = value
    if not restored.flags.writeable or restored.strides != strides:
        raise J2Error("restored array layout differs from captured positive strides")
    if expected_hash is not None and _sha_array(restored) != expected_hash:
        raise J2Error("restored logical array canonical hash mismatch")
    return restored


def _json_scalar(value):
    if isinstance(value, (tuple, list)):
        return [_json_scalar(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise J2Error("FFT call metadata contains a non-finite scalar")
        return value
    raise J2Error(f"FFT parameter is not a scalar/axis tuple: {type(value).__name__}")


def _array_error_metrics(reference, candidate):
    import numpy as np
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape or a.dtype != b.dtype:
        raise J2Error("FFT output shape/dtype changed")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise J2Error("FFT output contains NaN/Inf")
    ah, bh = _sha_array(a), _sha_array(b)
    if a.dtype.kind not in "biufc":
        raise J2Error("comparison array dtype is not numeric")
    a_numeric = a.astype(np.complex128 if a.dtype.kind == "c" else np.float64, copy=False)
    b_numeric = b.astype(np.complex128 if b.dtype.kind == "c" else np.float64, copy=False)
    diff = np.abs(b_numeric - a_numeric)
    max_abs = float(np.max(diff, initial=0.0))
    ref_abs = np.abs(a)
    ref_max = float(np.max(ref_abs, initial=0.0))
    normalized = (0.0 if max_abs == 0.0 else
                  (max_abs / ref_max if ref_max > 0.0 else None))
    scale = ref_max
    if scale == 0.0:
        relative_l2 = 0.0 if max_abs == 0.0 else None
    else:
        relative_l2 = float(np.linalg.norm((diff / scale).ravel()) /
                            np.linalg.norm((ref_abs / scale).ravel()))
    nonzero = ref_abs > 0
    zero_count = int(nonzero.size - np.count_nonzero(nonzero))
    rel = (float(np.max((diff[nonzero] / ref_abs[nonzero]), initial=0.0))
           if np.any(nonzero) else None)
    zero_error = float(np.max(diff[~nonzero], initial=0.0)) if zero_count else 0.0
    bitwise_equal = a.tobytes(order="C") == b.tobytes(order="C")
    return {"reference_canonical_sha256": ah, "candidate_canonical_sha256": bh,
        "canonical_exact": ah == bh, "max_abs": max_abs,
        "bitwise_equal": bitwise_equal,
        "normalized_Linf": normalized,
        "normalized_Linf_undefined_reason": ("reference_amplitude_zero" if normalized is None else None),
        "relative_L2": relative_l2,
        "relative_L2_undefined_reason": ("reference_norm_zero" if relative_l2 is None else None),
        "elementwise_relative_max_nonzero_reference": rel,
        "zero_reference_element_count": zero_count,
        "max_abs_error_on_zero_reference_elements": zero_error}


def _load_capture_index(capture_root: Path):
    index_path = capture_root / "capture_manifest.json"
    manifest = json.loads(index_path.read_text(encoding="utf-8"))
    captures = manifest.get("captures", {})
    if not captures:
        raise J2Error("captured FFT input manifest is empty")
    return manifest, captures


def _restore_capture(capture_root: Path, record: dict):
    import numpy as np
    path = capture_root / record["values_file"]
    compact = np.load(path, mmap_mode="r", allow_pickle=False)
    if raw_sha256_file(path) != record["values_raw_sha256"]:
        raise J2Error(f"captured input raw SHA mismatch: {path.name}")
    return restore_layout(compact, shape=record["shape"], dtype=record["dtype"],
        strides=record["strides"], expected_hash=record["input_canonical_sha256"])


def _publish_capture(dispatch: _CaptureDispatch, target_dir: Path, receipt: dict,
                    provenance: dict | None = None):
    import numpy as np
    observed = receipt.get("observables", {})
    for field in ("adaptive_rejection_count_z", "safety_mode_trigger_count_z"):
        values = np.asarray(observed.get(field, []))
        if values.size == 0 or not np.isfinite(values).all() or np.any(values != 0):
            raise J2Error(f"capture stage requires zero {field} across the full window")
    if len(receipt.get("interval_rows", [])) != 8:
        raise J2Error("capture stage requires eight accepted interval observations")
    if target_dir.exists():
        raise FileExistsError(target_dir)
    target_dir.mkdir(parents=True, exist_ok=False)
    captures = {}
    for key, record in sorted(dispatch.arrays.items()):
        name = f"{key}.npy"
        path = target_dir / name
        compact = record.pop("compact_values")
        with path.open("xb") as stream:
            np.save(stream, compact, allow_pickle=False)
        item = {name: value for name, value in record.items() if name != "compact_values"}
        item.update({"hash_scope": "raw_bytes",
            "input_canonical_sha256": _sha_array(compact),
            "values_file": name, "values_raw_sha256": raw_sha256_file(path)})
        captures[key] = item
    expected = {"fft_complex3", "fft_float3", "ifft_complex3",
                "fft2_plane_complex", "ifft2_plane_complex"}
    missing = sorted(expected - set(captures))
    manifest = {"schema": SCHEMA + ".captures.v1", "phase": PHASE,
        "hash_scope": "raw_bytes",
        "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
        "source_canonical_sha256": receipt.get("source_canonical_sha256"),
        "pre0_canonical_sha256": receipt.get("pre0_canonical_sha256"),
        "source_raw_sha256": (provenance or {}).get("source_raw_sha256"),
        "pre0_raw_sha256": (provenance or {}).get("pre0_raw_sha256"),
        "transfer_mapping_sha256": (provenance or {}).get("transfer_mapping_sha256"),
        "accepted_window_intervals": int(receipt.get("interval_count", 0)),
        "adaptive_rejection_count": receipt.get("observables", {}).get("adaptive_rejection_count_z"),
        "safety_mode_trigger_count": receipt.get("observables", {}).get("safety_mode_trigger_count_z"),
        "capture_published_after_complete_window": True,
        "captures": captures,
        "observed_layouts": {k: list(v.values()) for k, v in dispatch.layouts.items()},
        "missing_required_classes": missing,
        "capture_status": "COMPLETE" if not missing else "INCOMPLETE_COVERAGE"}
    _json_new(target_dir / "capture_manifest.json", manifest)
    if missing:
        raise J2Error(f"accepted window did not exercise required FFT capture classes: {missing}")
    return manifest


def _capture_transform(record, value, backend, workers=1):
    import numpy as np
    kwargs = dict(record.get("call_kwargs", {}))
    tail = list(record.get("call_args_tail", []))
    if backend == "numpy":
        function = getattr(np.fft, record["transform"])
        return function(value, *tail, **kwargs)
    import scipy.fft
    function = getattr(scipy.fft, record["transform"])
    kwargs["workers"] = int(workers)
    kwargs["overwrite_x"] = False
    return function(value, *tail, **kwargs)


def _one_micro_trial(capture_root, record, backend, workers, reference_output=None):
    import numpy as np
    service_start, cpu_start = time.perf_counter(), time.process_time()
    restore_start = time.perf_counter()
    value = _restore_capture(capture_root, record)
    restore_wall = time.perf_counter() - restore_start
    kernel_start, kernel_cpu = time.perf_counter(), time.process_time()
    output = _capture_transform(record, value, backend, workers)
    kernel_wall = time.perf_counter() - kernel_start
    kernel_cpu_seconds = time.process_time() - kernel_cpu
    output = np.asarray(output)
    if not np.isfinite(output).all():
        raise J2Error("microbenchmark FFT output contains NaN/Inf")
    metrics = (_array_error_metrics(reference_output, output)
               if reference_output is not None else None)
    output_hash = _sha_array(output)
    process_rss = common.tree_snapshot(os.getpid())
    service_wall = time.perf_counter() - service_start
    result = {"backend": backend, "workers": workers if backend == "scipy.fft" else None,
        "service_wall_seconds_restore_included": service_wall,
        "service_process_cpu_seconds_restore_included": time.process_time() - cpu_start,
        "process_rss_bytes_after_trial": (int(process_rss["rss"]) if process_rss else None),
        "layout_restore_wall_seconds": restore_wall,
        "kernel_wall_seconds": kernel_wall, "kernel_process_cpu_seconds": kernel_cpu_seconds,
        "input_shape": list(value.shape), "input_dtype": str(value.dtype),
        "input_strides": list(value.strides), "output_shape": list(output.shape),
        "output_dtype": str(output.dtype), "output_canonical_sha256": output_hash,
        "error_metrics_vs_numpy_reference": metrics}
    del value, output
    return result


def _percentile90(values):
    ordered = sorted(float(v) for v in values)
    return ordered[max(0, math.ceil(0.90 * len(ordered)) - 1)]


def _summarize_samples(samples, key):
    values = [float(item[key]) for item in samples]
    return {"samples": values, "p50": float(statistics.median(values)),
        "p90": _percentile90(values), "p50_method": "ordinary_median",
        "p90_method": "nearest_rank", "min": min(values), "max": max(values)}


def _class_group(key):
    return "spatial" if "plane" in key else "temporal"


def _class_weight(key):
    return {"fft_complex3": 48, "fft_float3": 32, "ifft_complex3": 80,
        "fft2_plane_complex": 6144, "ifft2_plane_complex": 6144}.get(key, 1)


def _microbenchmark(capture_root: Path, output_dir: Path) -> dict:
    import numpy as np
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    _, captures = _load_capture_index(capture_root)
    trials, scores = {}, defaultdict(dict)
    for key, record in sorted(captures.items()):
        ref_input = _restore_capture(capture_root, record)
        cold_start, cold_cpu = time.perf_counter(), time.process_time()
        reference = _capture_transform(record, ref_input, "numpy")
        first_reference = {"wall_seconds": time.perf_counter() - cold_start,
            "process_cpu_seconds": time.process_time() - cold_cpu,
            "scope": "FIRST_REFERENCE_CALL_FOR_CLASS_NOT_GUARANTEED_COLD_PROCESS"}
        j1._append_event(output_dir / "microbenchmark_progress.jsonl",
            {"class": key, "sample_kind": "first_numpy_reference", "trial": first_reference})
        if not np.isfinite(reference).all():
            raise J2Error(f"NumPy FFT reference is non-finite: {key}")
        row = {"class": key, "group": _class_group(key), "weight": _class_weight(key),
            "first_numpy_reference": first_reference,
            "callsite": record["callsite"], "shape": record["shape"],
            "dtype": record["dtype"], "strides": record["strides"],
            "call_kwargs": record.get("call_kwargs", {}),
            "call_args_tail": record.get("call_args_tail", []), "methods": {}}
        methods = [("numpy", None), *(('scipy.fft', n) for n in (1, 8, 24, 48))]
        for backend, workers in methods:
            warmup = _one_micro_trial(capture_root, record, backend, workers, reference)
            j1._append_event(output_dir / "microbenchmark_progress.jsonl", {
                "class": key, "method": backend, "workers": workers,
                "sample_kind": "warmup", "trial": warmup})
            samples = []
            for repeat in range(2):
                sample = _one_micro_trial(capture_root, record, backend, workers, reference)
                samples.append(sample)
                j1._append_event(output_dir / "microbenchmark_progress.jsonl", {
                    "class": key, "method": backend, "workers": workers,
                    "sample_kind": "measured", "sample_index": repeat, "trial": sample})
            label = "numpy" if workers is None else f"scipy_workers_{workers}"
            row["methods"][label] = {"warmup_count": 1, "measured_count": 2,
                "kernel_wall": _summarize_samples(samples, "kernel_wall_seconds"),
                "kernel_cpu": _summarize_samples(samples, "kernel_process_cpu_seconds"),
                "service_wall": _summarize_samples(samples, "service_wall_seconds_restore_included"),
                "layout_restore_wall": _summarize_samples(samples, "layout_restore_wall_seconds"),
                "warmup": warmup, "samples": samples}
            if workers is not None:
                scores.setdefault(row["group"], {}).setdefault(workers, []).append(
                    (row["weight"], row["methods"][label]["kernel_wall"]["p50"]))
        trials[key] = row
        del ref_input, reference
    selected = {}
    for group in ("temporal", "spatial"):
        group_scores = {}
        for workers, points in scores[group].items():
            total_weight = sum(weight for weight, _ in points)
            group_scores[workers] = sum(weight * median for weight, median in points) / total_weight
        if not group_scores:
            raise J2Error(f"capture set has no {group} transform class")
        if group_scores.get(48, math.inf) < group_scores.get(24, math.inf):
            for key, row in trials.items():
                if row["group"] != group:
                    continue
                record = captures[key]
                base = _restore_capture(capture_root, record)
                ref = _capture_transform(record, base, "numpy")
                warmup = _one_micro_trial(capture_root, record, "scipy.fft", 96, ref)
                j1._append_event(output_dir / "microbenchmark_progress.jsonl",
                    {"class": key, "workers": 96, "sample_kind": "warmup", "trial": warmup})
                samples = []
                for repeat in range(2):
                    sample = _one_micro_trial(capture_root, record, "scipy.fft", 96, ref)
                    samples.append(sample)
                    j1._append_event(output_dir / "microbenchmark_progress.jsonl",
                        {"class": key, "workers": 96, "sample_kind": "measured",
                         "sample_index": repeat, "trial": sample})
                row["methods"]["scipy_workers_96"] = {"warmup_count": 1, "warmup": warmup,
                    "measured_count": 2,
                    "kernel_wall": _summarize_samples(samples, "kernel_wall_seconds"),
                    "kernel_cpu": _summarize_samples(samples, "kernel_process_cpu_seconds"),
                    "service_wall": _summarize_samples(samples, "service_wall_seconds_restore_included"),
                    "layout_restore_wall": _summarize_samples(samples, "layout_restore_wall_seconds"),
                    "samples": samples}
                group_scores[96] = group_scores.get(96, 0.0) + row["weight"] * row["methods"]["scipy_workers_96"]["kernel_wall"]["p50"]
                del base, ref
            total_weight = sum(row["weight"] for row in trials.values() if row["group"] == group)
            group_scores[96] /= total_weight
        winner = min(group_scores, key=group_scores.get)
        selected[group] = int(winner)
        for key, row in trials.items():
            if row["group"] != group:
                continue
            record = captures[key]
            base = _restore_capture(capture_root, record)
            ref = _capture_transform(record, base, "numpy")
            extra = _one_micro_trial(capture_root, record, "scipy.fft", winner, ref)
            j1._append_event(output_dir / "microbenchmark_progress.jsonl",
                {"class": key, "workers": winner, "sample_kind": "winner_extra", "trial": extra})
            label = f"scipy_workers_{winner}"
            row["methods"][label]["samples"].append(extra)
            row["methods"][label]["measured_count"] += 1
            for metric, field in (("kernel_wall", "kernel_wall_seconds"),
                ("kernel_cpu", "kernel_process_cpu_seconds"),
                ("service_wall", "service_wall_seconds_restore_included"),
                ("layout_restore_wall", "layout_restore_wall_seconds")):
                row["methods"][label][metric] = _summarize_samples(
                    row["methods"][label]["samples"], field)
            del base, ref
    result = {"schema": SCHEMA + ".microbenchmark.v1", "phase": PHASE,
        "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
        "capture_manifest_raw_sha256": raw_sha256_file(capture_root / "capture_manifest.json"),
        "warmups_per_method": 1, "initial_measured_samples_per_method": 2,
        "selected_method_additional_sample_count": 1,
        "worker_candidates": [1, 8, 24, 48, 96],
        "worker_96_rule": "only if weighted scipy 48 median beats 24",
        "call_weights": {key: _class_weight(key) for key in trials},
        "selected_candidate_workers": {"temporal_workers": selected["temporal"],
            "spatial_workers": selected["spatial"]},
        "selection_metric": "weighted_mean_per_class_kernel_wall_median",
        "representative_layouts_not_full_callsite_coverage": True,
        "cases": trials}
    _json_new(output_dir / "microbenchmark.json", result)
    return result


def _load_real_context(args):
    allocation = p1r._check_allocation()
    common.add_vendor(str(args.vendor_dir))
    import threadpoolctl
    thread_limit = threadpoolctl.threadpool_limits(limits=1)
    if getattr(threadpoolctl, "__version__", None) != "3.6.0":
        raise J2Error("the frozen threadpoolctl 3.6.0 vendor is required")
    if sys.executable != p1r.FIXED_PYTHON:
        raise J2Error(f"fixed Python required: {p1r.FIXED_PYTHON}; got {sys.executable}")
    pools = threadpoolctl.threadpool_info()
    if any(int(row.get("num_threads", 1)) > 1 for row in pools):
        raise J2Error("BLAS/native pool exceeds one thread")
    j1._assert_cpu_backend()
    binding = p1r._source_binding(args.bench_root)
    context = p1r._load_context(args.input_root, args.bench_root, args.vendor_dir)
    return allocation, thread_limit, pools, binding, context


def _pre_fields(context):
    module = __import__("KHz_filament.hr4e5_formal_entry", fromlist=["pre0_fields_from_delta_n"])
    return module.pre0_fields_from_delta_n(context["assets"]["pre0"], expected_count=8)


def _provenance_supplement(args, stage, receipt, allocation, pools, binding, context):
    value = {"schema": SCHEMA + ".child_provenance.v1", "phase": j1.PHASE,
        "parent_phase": PHASE,
        "stage": stage, "classification": CLASSIFICATION,
        "production_qualification": QUALIFICATION,
        "source_raw_sha256": p1r.SOURCE_RAW_SHA256,
        "pre0_raw_sha256": p1r.PRE0_RAW_SHA256,
        "transfer_mapping_sha256": context["assets"]["mapping_sha256"],
        "source_canonical_sha256": receipt["source_canonical_sha256"],
        "pre0_canonical_sha256": receipt["pre0_canonical_sha256"],
        "source_binding": binding,
        "source_validation": j1._json_safe(context["source_validation"]),
        "lut_validation": j1._json_safe(context["lut"]),
        "schedule": j1._json_safe(context["schedule"].as_metadata()),
        "runtime_config_delta": j1._json_safe(context["runtime_config_delta"]),
        "effective_linear_precision_strategy": j1._effective_precision_strategy(context["prop"]),
        "child_allocation": allocation, "threadpool_info": pools,
        "stage_metrics": receipt.get("stage_metrics"),
        "candidate_descriptor": receipt.get("candidate_descriptor"),
        "capture_manifest_raw_sha256": receipt.get("capture_manifest_raw_sha256")}
    _json_new(args.output_root / "child_provenance_receipt.json", value)


def _child_runtime(args, *, load_context):
    allocation = p1r._check_allocation()
    common.add_vendor(str(args.vendor_dir))
    import threadpoolctl
    limit = threadpoolctl.threadpool_limits(limits=1)
    if getattr(threadpoolctl, "__version__", None) != "3.6.0":
        raise J2Error("the frozen threadpoolctl 3.6.0 vendor is required")
    if sys.executable != p1r.FIXED_PYTHON:
        raise J2Error(f"fixed Python required: {p1r.FIXED_PYTHON}; got {sys.executable}")
    pools = threadpoolctl.threadpool_info()
    if any(int(row.get("num_threads", 1)) > 1 for row in pools):
        raise J2Error("BLAS/native pool exceeds one thread")
    j1._assert_cpu_backend()
    if not load_context:
        return allocation, limit, pools, None, None
    binding = p1r._source_binding(args.bench_root)
    context = p1r._load_context(args.input_root, args.bench_root, args.vendor_dir)
    return allocation, limit, pools, binding, context


def _real_child(args):
    setup_wall, setup_cpu = time.perf_counter(), time.process_time()
    stage = args.child_stage
    allocation, limit, pools, binding, context = _child_runtime(
        args, load_context=stage != "microbench")
    setup_metrics = {"context_setup_wall_seconds": time.perf_counter() - setup_wall,
        "context_setup_process_cpu_seconds": time.process_time() - setup_cpu}
    try:
        if stage == "microbench":
            result = _microbenchmark(args.capture_root, args.output_root)
            _json_new(args.output_root.parent / "microbenchmark_child_receipt.json", {
                "schema": SCHEMA + ".child.v1", "phase": PHASE,
                "stage": stage, "status": "COMPLETE_DIAGNOSTIC",
                "classification": CLASSIFICATION,
                "production_qualification": QUALIFICATION,
                "process_cpu_seconds": time.process_time(),
                "microbenchmark_raw_sha256": raw_sha256_file(args.output_root / "microbenchmark.json"),
                "selected_candidate_workers": result["selected_candidate_workers"]})
            return 0
        source_hash = str(context["source_hash"])
        pre0_hash = str(context["assets"]["pre0_canonical_sha256"])
        fields = _pre_fields(context)
        holder = {}
        factory = None
        descriptor = None
        if stage == "capture":
            factory = make_capture_factory(holder)
        elif stage.startswith("candidate"):
            descriptor = _validate_descriptor({"temporal_workers": args.temporal_workers,
                "spatial_workers": args.spatial_workers})
            factory = make_propagator_factory(descriptor)
        started_cpu = time.process_time()
        receipt = j1._run_window(source=context["source"], pre_fields=fields,
            grid=context["grid"], beam=context["beam"], prop=context["prop"],
            ion=context["ion"], heat=context["heat"], raman=context["raman"],
            schedule=context["schedule"], output_dir=args.output_root,
            source_label="sealed_pre0_first_8", initial_pre0=True, profile=False,
            source_hash=source_hash, pre0_hash=pre0_hash, stage_name=f"J2_{stage}",
            execution_scope="J2_REAL_INPUT", propagator_factory=factory)
        receipt["stage_metrics"] = dict(setup_metrics,
            window_call_process_cpu_seconds=time.process_time() - started_cpu,
            process_cpu_seconds_since_child_entry=time.process_time() - setup_cpu)
        if stage == "capture":
            dispatch = holder.get("dispatch")
            if not isinstance(dispatch, _CaptureDispatch):
                raise J2Error("capture propagator did not return its private collector")
            manifest = _publish_capture(dispatch, args.output_root.parent / "captured_inputs", receipt,
                {"source_raw_sha256": p1r.SOURCE_RAW_SHA256,
                 "pre0_raw_sha256": p1r.PRE0_RAW_SHA256,
                 "transfer_mapping_sha256": context["assets"]["mapping_sha256"]})
            receipt["capture_manifest_raw_sha256"] = raw_sha256_file(
                args.output_root.parent / "captured_inputs" / "capture_manifest.json")
            receipt["capture_status"] = manifest["capture_status"]
        elif descriptor:
            receipt["candidate_descriptor"] = descriptor
        _provenance_supplement(args, stage, receipt, allocation, pools, binding, context)
        return 0
    finally:
        del limit


def _child_main(args):
    try:
        if args.phase != PHASE or args.child_stage not in STAGES:
            raise J2Error("J2 child requires explicit --phase J2 and valid child stage")
        if any(value is None for value in (args.bench_root, args.input_root,
                args.vendor_dir, args.output_root)):
            raise J2Error("J2 child requires bench/input/vendor/output roots")
        if args.child_stage == "microbench" and args.capture_root is None:
            raise J2Error("microbenchmark child requires --capture-root")
        return _real_child(args)
    except BaseException as exc:
        traceback.print_exc()
        failure_root = args.output_root.parent if args.child_stage == "microbench" else args.output_root.parent
        try:
            _json_new(failure_root / "child_failure.json", {
                "schema": SCHEMA + ".child_failure.v1", "phase": PHASE,
                "stage": args.child_stage, "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()})
        except Exception:
            pass
        return 1


def _compare_stage_files(reference: dict, candidate: dict, label: str) -> dict:
    import numpy as np
    comparisons, exact, hard_gate_errors = {}, True, []
    for key, left_path, right_path in (
        ("final_E", reference["final_E_path"], candidate["final_E_path"]),
        ("final_POST", reference["final_POST_path"], candidate["final_POST_path"]),
        ("diagnostics", reference["diagnostics_path"], candidate["diagnostics_path"])):
        if key == "final_E":
            left_items = {"E": np.load(left_path, mmap_mode="r", allow_pickle=False)}
            right_items = {"E": np.load(right_path, mmap_mode="r", allow_pickle=False)}
        else:
            with np.load(left_path, allow_pickle=False) as a:
                left_items = {name: a[name].copy() for name in a.files}
            with np.load(right_path, allow_pickle=False) as b:
                right_items = {name: b[name].copy() for name in b.files}
        if set(left_items) != set(right_items):
            hard_gate_errors.append(f"{key}_field_set_changed")
            exact = False
            continue
        rows = {}
        for name in sorted(left_items):
            a, b = left_items[name], right_items[name]
            optional_energy = {"energy_step_start_J", "energy_after_linear_half1_J",
                "energy_after_raman_pre_J", "energy_after_nonraman_J",
                "energy_after_raman_post_J", "energy_after_linear_half2_J"}
            if (key == "diagnostics" and name in optional_energy and a.shape == b.shape
                    and a.dtype == b.dtype and np.isnan(a).all() and np.isnan(b).all()):
                same = _sha_array(a) == _sha_array(b)
                rows[name] = {"comparison_scope": "FROZEN_DISABLED_OPERATOR_ENERGY_NAN_PLACEHOLDER",
                    "canonical_exact": same, "bitwise_equal": a.tobytes() == b.tobytes(),
                    "reference_canonical_sha256": _sha_array(a),
                    "candidate_canonical_sha256": _sha_array(b)}
                exact = exact and same
                if not same:
                    hard_gate_errors.append(f"disabled_placeholder_{name}_changed")
                continue
            if a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all() or not np.isfinite(b).all():
                hard_gate_errors.append(f"{key}.{name}_shape_dtype_or_finite_changed")
                exact = False
                rows[name] = {"hard_gate": "FAIL", "left_shape": list(a.shape),
                    "right_shape": list(b.shape), "left_dtype": str(a.dtype), "right_dtype": str(b.dtype)}
                continue
            metrics = _array_error_metrics(a, b)
            metrics["comparison_scope"] = ("TIMING_ONLY_EXCLUDED_FROM_NUMERICAL_EQUALITY"
                if key == "diagnostics" and name in TIMING_FIELDS else "NUMERICAL")
            rows[name] = metrics
            if key == "diagnostics" and name in TIMING_FIELDS:
                continue
            if not metrics["canonical_exact"]:
                exact = False
                if key == "diagnostics" and (name in DISCRETE_FIELDS or a.dtype.kind in "biu"):
                    hard_gate_errors.append(f"discrete_{name}_changed")
        comparisons[key] = rows
    left_receipt, right_receipt = reference["receipt"], candidate["receipt"]
    for field in ("source_canonical_sha256", "pre0_canonical_sha256"):
        if left_receipt.get(field) != right_receipt.get(field):
            hard_gate_errors.append(f"{field}_changed")
    for name in ("z_axis", "adaptive_rejection_count_z", "safety_mode_trigger_count_z"):
        a = np.asarray(left_receipt.get("observables", {}).get(name, []))
        b = np.asarray(right_receipt.get("observables", {}).get(name, []))
        if a.shape != b.shape or not np.array_equal(a, b):
            hard_gate_errors.append(f"topology_{name}_changed")
    return {"label": label, "canonical_exact_match": bool(exact),
        "hard_gate_pass": not hard_gate_errors, "hard_gate_errors": hard_gate_errors,
        "classification": "EXACT_DIAGNOSTIC_MATCH" if exact and not hard_gate_errors
            else "NUMERICAL_DIFFERENCE", "metrics": comparisons}


def _stage_command(args, stage, stage_dir, descriptor=None):
    output = stage_dir / ("measurements" if stage == "microbench" else "propagation")
    command = [sys.executable, "-m", "tools.cpu_p1r2.cpu_p1r2_j2",
        "--phase", PHASE, "--bench-root", str(args.bench_root),
        "--input-root", str(args.input_root), "--vendor-dir", str(args.vendor_dir),
        "--output-root", str(output), "--child-stage", stage]
    if stage == "microbench":
        command.extend(["--capture-root", str(args.output_root / "stages" / "capture" / "captured_inputs")])
    if descriptor is not None:
        command.extend(["--temporal-workers", str(descriptor["temporal_workers"]),
            "--spatial-workers", str(descriptor["spatial_workers"])])
    return command


def _supervise_stage(args, stage, *, timeout_seconds, descriptor=None):
    stage_dir = args.output_root / "stages" / stage
    stage_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    supervisor = j1._spawn_and_monitor(_stage_command(args, stage, stage_dir, descriptor),
        stage_dir=stage_dir, cwd=args.bench_root / "Filament_python",
        env=j1._child_environment(args.bench_root), timeout_seconds=timeout_seconds,
        memory_bytes=MEMORY_BYTES)
    result = {"stage": stage, "supervisor": supervisor}
    if supervisor.get("exit_code") == 0 and supervisor.get("stop") is None:
        try:
            if stage == "microbench":
                data = stage_dir / "measurements" / "microbenchmark.json"
                child = stage_dir / "microbenchmark_child_receipt.json"
                if not data.is_file() or not child.is_file():
                    raise J2Error("microbenchmark child omitted a complete result receipt")
                result.update({"status": "DONE", "microbenchmark": json.loads(data.read_text(encoding="utf-8")),
                    "child_receipt": json.loads(child.read_text(encoding="utf-8"))})
            else:
                validation = j1._validate_stage(stage_dir)
                result.update({"status": "DONE", "validation": validation,
                    "receipt": validation["receipt"]})
                if stage == "capture":
                    result["capture_manifest"] = json.loads((stage_dir / "captured_inputs" / "capture_manifest.json").read_text(encoding="utf-8"))
        except BaseException as exc:
            result.update({"status": "FAIL_OUTPUT_VALIDATION",
                "validation_error": f"{type(exc).__name__}: {exc}"})
    elif supervisor.get("exit_code") is None:
        result["status"] = "STOP_UNRESOLVED"
    elif supervisor.get("timed_out"):
        result["status"] = "LIMIT_REACHED"
    elif supervisor.get("stop"):
        result["status"] = "STOPPED_RESOURCE_GATE"
    elif supervisor.get("exit_code") is None:
        result["status"] = "STOP_UNRESOLVED"
    else:
        result["status"] = "FAIL_CHILD"
    result["service_wall_seconds_including_validation"] = time.perf_counter() - started
    _json_new(stage_dir / "stage_supervisor_receipt.json", result)
    j1._append_event(args.output_root / "run_progress.jsonl", {
        "phase": PHASE, "stage": stage, "status": result["status"],
        "service_wall_seconds": result["service_wall_seconds_including_validation"],
        "rss_80_percent_latched": supervisor.get("rss_80_percent_latched")})
    return result


def _tiny_child(args):
    context = j1._tiny_context()
    source_hash = j1._validate_source_array(context["source"])
    pre0_hash = _sha_array(context["pre_fields"]["delta_n"])
    descriptor = {"temporal_workers": args.temporal_workers or 1,
        "spatial_workers": args.spatial_workers or 1}
    holder = {}
    factory = (None if args.smoke_child == "baseline" else
        make_capture_factory(holder) if args.smoke_child == "capture" else make_propagator_factory(descriptor))
    receipt = j1._run_window(source=context["source"], pre_fields=context["pre_fields"],
        grid=context["grid"], beam=context["beam"], prop=context["prop"],
        ion=context["ion"], heat=context["heat"], raman=context["raman"],
        schedule=context["schedule"], output_dir=args.output_root,
        source_label="synthetic_tiny_PRE0", initial_pre0=True, profile=False,
        source_hash=source_hash, pre0_hash=pre0_hash, stage_name=f"tiny_{args.smoke_child}",
        execution_scope="SOFTWARE_SMOKE_ONLY", propagator_factory=factory)
    if args.smoke_child == "capture":
        _publish_capture(holder["dispatch"], args.output_root.parent / "captured_inputs", receipt,
            {"execution_scope": "SOFTWARE_SMOKE_ONLY"})
    _json_new(args.output_root / "child_provenance_receipt.json", {
        "schema": SCHEMA + ".smoke_provenance.v1", "phase": j1.PHASE,
        "execution_scope": "SOFTWARE_SMOKE_ONLY",
        "source_canonical_sha256": receipt["source_canonical_sha256"],
        "pre0_canonical_sha256": receipt["pre0_canonical_sha256"]})
    return 0


def _entry_smoke(args):
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=False)
    records = {}
    for stage in ("baseline", "capture", "candidate"):
        stage_dir = args.output_root / "stages" / stage
        stage_dir.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, "-m", "tools.cpu_p1r2.cpu_p1r2_j2",
            "--phase", PHASE, "--bench-root", str(args.bench_root),
            "--output-root", str(stage_dir / "propagation"), "--smoke-child", stage]
        if stage == "candidate":
            command.extend(["--temporal-workers", "1", "--spatial-workers", "1"])
        run = j1._spawn_and_monitor(command, stage_dir=stage_dir,
            cwd=args.bench_root / "Filament_python",
            env=j1._child_environment(args.bench_root), timeout_seconds=900,
            memory_bytes=MEMORY_BYTES)
        row = {"stage": stage, "supervisor": run}
        if run.get("exit_code") == 0 and not run.get("stop"):
            validation = j1._validate_stage(stage_dir)
            row.update({"status": "DONE", "validation": validation})
        else:
            row["status"] = "FAIL"
        records[stage] = row
    comparison = (_compare_stage_files(records["baseline"]["validation"],
        records["candidate"]["validation"], "tiny_baseline_vs_scipy")
        if all(records[name]["status"] == "DONE" for name in records) else None)
    dispatch = (records.get("candidate", {}).get("validation", {}).get("receipt", {}).get("profile") or {})
    count = sum(dispatch.get("transform_counts", {}).values())
    capture_comparison = (_compare_stage_files(records["baseline"]["validation"],
        records["capture"]["validation"], "tiny_capture_vs_baseline")
        if records["capture"]["status"] == "DONE" and records["baseline"]["status"] == "DONE" else None)
    passed = bool(comparison and comparison["hard_gate_pass"] and count > 0
        and capture_comparison and capture_comparison["canonical_exact_match"]
        and capture_comparison["hard_gate_pass"])
    _json_new(args.output_root / "entry_smoke_receipt.json", {
        "schema": SCHEMA + ".entry_smoke.v1", "phase": PHASE,
        "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
        "execution_scope": "SOFTWARE_SMOKE_ONLY",
        "status": "PASS_SOFTWARE_ONLY" if passed else "FAIL",
        "independent_fresh_children": True, "full_eight_interval_windows": True,
        "stages": records, "comparison": comparison, "capture_comparison": capture_comparison,
        "candidate_transform_count": count,
        "nonzero_numerical_differences_remain_diagnostic": True})
    return 0 if passed else 1


def _run_j2(args):
    if args.phase != PHASE:
        raise J2Error("J2 coordinator requires explicit --phase J2")
    if any(value is None for value in (args.bench_root, args.input_root, args.vendor_dir)):
        raise J2Error("full J2 requires bench-root, input-root and vendor-dir")
    allocation, thread_limit, pools, binding, context = _load_real_context(args)
    del context, binding
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=False)
    p1r._environment_receipt(args.output_root, vendor_dir=args.vendor_dir, allocation=allocation)
    allocation_start = j1._allocation_start_epoch()
    deadline = allocation_start + ALLOCATION_SECONDS
    initial_window_estimate = 566.0
    predictions = {"capture": 900.0, "baseline": initial_window_estimate,
        "microbench": 900.0, "candidate-a": initial_window_estimate,
        "candidate-b": initial_window_estimate}
    stages, comparisons, rss_latched = [], {}, False
    descriptor = None
    baseline = candidate_a = None
    capture_identity_ok = True
    for index, stage in enumerate(STAGES):
        prior = {row["stage"]: row for row in stages}
        block = None
        if rss_latched:
            block = "BLOCKED_RSS_80_PERCENT_LATCH"
        elif stage != "capture" and prior.get("capture", {}).get("status") != "DONE":
            block = "BLOCKED_CAPTURE_STAGE"
        elif stage in ("microbench", "candidate-a", "candidate-b") and prior.get("baseline", {}).get("status") != "DONE":
            block = "BLOCKED_BASELINE_STAGE"
        elif stage in ("microbench", "candidate-a", "candidate-b") and not capture_identity_ok:
            block = "BLOCKED_CAPTURE_BASELINE_EXACT_GATE"
        elif stage in ("candidate-a", "candidate-b") and prior.get("microbench", {}).get("status") != "DONE":
            block = "BLOCKED_MICROBENCH_STAGE"
        elif stage == "candidate-b" and (candidate_a is None or not candidate_a.get("hard_gate_pass")):
            block = "BLOCKED_CANDIDATE_A_HARD_GATE"
        if block:
            stages.append({"stage": stage, "status": block})
            continue
        remaining = deadline - time.time()
        future = sum(predictions[name] for name in STAGES[index:])
        if remaining < future + RESERVE_SECONDS:
            stages.append({"stage": stage, "status": "SKIPPED_INSUFFICIENT_TIME",
                "remaining_seconds": remaining, "projected_remaining_stage_seconds": future,
                "reserve_seconds": RESERVE_SECONDS})
            continue
        timeout = (900.0 if stage == "microbench" else WINDOW_SECONDS)
        timeout = min(timeout, max(1.0, remaining - RESERVE_SECONDS))
        result = _supervise_stage(args, stage, timeout_seconds=timeout, descriptor=descriptor)
        stages.append(result)
        rss_latched = bool(result.get("supervisor", {}).get("rss_80_percent_latched"))
        if stage == "capture" and result.get("status") == "DONE":
            capture_identity_ok = (result["receipt"].get("source_canonical_sha256") is not None
                and result["receipt"].get("pre0_canonical_sha256") is not None
                and result.get("capture_manifest", {}).get("missing_required_classes") == [])
        if stage == "baseline" and result.get("status") == "DONE":
            baseline = result["validation"]
            capture_comparison = _compare_stage_files(prior["capture"]["validation"], baseline,
                "capture_vs_clean_baseline")
            comparisons["capture_vs_clean_baseline"] = capture_comparison
            capture_identity_ok = bool(capture_identity_ok and capture_comparison["hard_gate_pass"]
                and capture_comparison["canonical_exact_match"])
            predictions["candidate-a"] = predictions["candidate-b"] = max(
                initial_window_estimate,
                float(result["receipt"].get("propagation_wall_seconds", 0.0)) * 1.1)
        if stage == "microbench" and result.get("status") == "DONE":
            descriptor = _validate_descriptor(result["microbenchmark"]["selected_candidate_workers"])
            _json_new(args.output_root / "candidate_descriptor.json", descriptor)
        if stage == "candidate-a" and result.get("status") == "DONE":
            candidate_a = _compare_stage_files(baseline, result["validation"], "candidate_a_vs_baseline")
            comparisons["candidate_a_vs_baseline"] = candidate_a
        if stage == "candidate-b" and result.get("status") == "DONE":
            comparisons["candidate_b_vs_baseline"] = _compare_stage_files(
                baseline, result["validation"], "candidate_b_vs_baseline")
            comparisons["candidate_a_vs_candidate_b"] = _compare_stage_files(
                stages[-2]["validation"], result["validation"], "candidate_a_vs_candidate_b")
    done = {row["stage"]: row for row in stages if row.get("status") == "DONE"}
    complete = len(done) == len(STAGES)
    repeat_exact = bool(comparisons.get("candidate_a_vs_candidate_b", {}).get("canonical_exact_match"))
    hard_gates = bool(complete and capture_identity_ok and len(comparisons) == 4
        and all(item.get("hard_gate_pass", False) for item in comparisons.values()))
    exact = all(item.get("canonical_exact_match", False) for item in comparisons.values())
    baseline_time = (float(done["baseline"]["receipt"]["propagation_wall_seconds"])
        if "baseline" in done else None)
    candidate_times = {stage: float(done[stage]["receipt"]["propagation_wall_seconds"])
        for stage in ("candidate-a", "candidate-b") if stage in done}
    performance = {stage: {"candidate_propagation_wall_seconds": value,
        "baseline_propagation_wall_seconds": baseline_time,
        "speedup": baseline_time / value if baseline_time and value else None,
        "improvement_fraction": 1.0 - value / baseline_time if baseline_time and value else None}
        for stage, value in candidate_times.items()}
    status = ("INCOMPLETE" if not complete else
        ("NUMERICAL_DIFFERENCE" if not hard_gates or not repeat_exact else "COMPLETE_DIAGNOSTIC"))
    receipt = {"schema": SCHEMA, "phase": PHASE,
        "classification": CLASSIFICATION, "production_qualification": QUALIFICATION,
        "status": status, "execution_order": list(STAGES),
        "execution_mode": "SERIAL_FRESH_CHILD_PER_STAGE",
        "allocation": allocation, "allocation_start_epoch": allocation_start,
        "allocation_wall_limit_seconds": ALLOCATION_SECONDS,
        "reserve_seconds": RESERVE_SECONDS,
        "source_and_PRE0_reloaded_for_each_optical_window": True,
        "stages": stages, "comparisons": comparisons,
        "all_candidate_numerical_differences_are_diagnostic_only": True,
        "candidate_repeat_exact": repeat_exact,
        "eligible_for_J3_bounded_diagnostic": bool(complete and hard_gates and repeat_exact),
        "acceptance_tolerance": "ACCEPTANCE_TOLERANCE_NOT_AUTHORIZED",
        "candidate_performance": performance,
        "candidate_descriptor": descriptor,
        "capture_identity_verified": capture_identity_ok,
        "capture_callsite_coverage": (done.get("capture", {}).get("capture_manifest", {}).get("observed_layouts")
            if "capture" in done else None),
        "rss_policy": {"80_percent": "LATCH_AND_STOP_LAUNCHING_LATER_STAGES",
            "90_percent": "BOUNDED_TERMINATION_OF_CURRENT_CHILD",
            "memory_bytes": MEMORY_BYTES,
            "pss_sample_interval_seconds": j1.PSS_SAMPLE_SECONDS}}
    _json_new(args.output_root / "run_receipt.json", receipt)
    del thread_limit
    return 0 if complete and hard_gates and repeat_exact else 1


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=(PHASE,))
    parser.add_argument("--bench-root", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--vendor-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--child-stage", choices=STAGES, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-child", choices=("baseline", "capture", "candidate"), help=argparse.SUPPRESS)
    parser.add_argument("--temporal-workers", type=int)
    parser.add_argument("--spatial-workers", type=int)
    parser.add_argument("--entry-smoke", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.smoke_child:
            return _tiny_child(args)
        if args.entry_smoke:
            if args.bench_root is None:
                raise J2Error("entry smoke requires --bench-root")
            return _entry_smoke(args)
        if args.child_stage:
            return _child_main(args)
        return _run_j2(args)
    except BaseException as exc:
        traceback.print_exc()
        try:
            failure_path = (args.output_root / "coordinator_failure.json"
                if args.output_root.is_dir() else args.output_root.with_suffix(".failure.json"))
            _json_new(failure_path, {"schema": SCHEMA + ".failure.v1", "phase": PHASE,
                "status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()})
        except Exception:
            pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


