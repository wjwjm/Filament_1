"""Bounded CPU-O1 local prototypes around frozen Filament operators.

This module never patches production globals and is not a production backend.
The compiled path accepts only the full-time, full-species RK4 contract; LUT
evaluation and all physical configuration handling remain in the frozen
runtime adapter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
import hashlib
import os
import platform
from pathlib import Path
import subprocess
import time
import zlib
from typing import Literal

import numpy as np


Implementation = Literal["reference", "numba_serial", "numba_parallel"]
Layout = Literal["NtB", "BNt"]
_NUMBA_KERNELS = None
_NUMBA_PHOTO_CLIP = None
_RAMAN_KERNELS = None
MAX_BOUNDED_COMPARISON_BYTES = 64 * 1024 * 1024
MAX_BOUNDED_DIFFERENCE_EXAMPLES = 128

# The private P1-R2 clone and its source modules are pinned to the reviewed
# pre-joint checkpoint. The binding covers the clone implementation, helper
# namespace transformation, and the two frozen science modules it invokes.
P1R2_SOURCE_ANCHOR_COMMIT = "3d4cbb54bc216c3fcfcd85bdd9c77a250e4bb0c3"
P1R2_SOURCE_ANCHOR_TREE = "41dfa1808a4290b5a1f674213759ddde6e7d4008"
P1R2_SOURCE_BLOBS = {
    "Filament_python/KHz_filament/propagate.py": "f156111398415c2b92e4609ccc1da075fcf5d1e3",
    "Filament_python/KHz_filament/raman.py": "5bb2702c109103ff882ee3642fedc748040c41b8",
    "Filament_python/tools/cpu_p1r2/cpu_p1r2_j1.py": "384fa5aebf6fa2905926c6d855521b641f66ba65",
    "Filament_python/tools/cpu_p1r2/cpu_p1r2_j2.py": "4a506accfb26cc90aacdd84b8525ca2f527ae821",
}
SOURCE_SHA = "c80aa3f744712d5035e84e9fa9a22be750a78ef578c15917e37150e66fa73aa6"


def _clip_photo_argument(value, lower, upper):
    """Scalar equivalent of NumPy clip for finite bounds, preserving ties."""
    if value < lower:
        value = lower
    if value > upper:
        value = upper
    return value
prange = range


@dataclass(frozen=True)
class CPUExecutionPlan:
    """Explicit settings for one local diagnostic invocation."""

    implementation: Implementation = "reference"
    temporal_workers: int = 1
    spatial_workers: int = 1
    kernel_threads: int = 1
    block_pixels: int = 32
    layout: Layout = "NtB"
    workspace_owner: Literal["per_call", "per_factory"] = "per_call"
    reduction_policy: Literal["ordered_species_reference_global"] = (
        "ordered_species_reference_global"
    )

    def __post_init__(self) -> None:
        if self.implementation not in {"reference", "numba_serial", "numba_parallel"}:
            raise ValueError("unsupported CPU-O1 implementation")
        for name, value in (
            ("temporal_workers", self.temporal_workers),
            ("spatial_workers", self.spatial_workers),
        ):
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value not in {1, 8, 24, 48, 96}):
                raise ValueError(f"{name} must be one of 1/8/24/48/96")
        if isinstance(self.kernel_threads, bool) or not isinstance(self.kernel_threads, int):
            raise ValueError("kernel_threads must be an integer")
        if self.kernel_threads < 1 or self.kernel_threads > 96:
            raise ValueError("kernel_threads must be in [1, 96]")
        if isinstance(self.block_pixels, bool) or not isinstance(self.block_pixels, int):
            raise ValueError("block_pixels must be an integer")
        if self.block_pixels < 1:
            raise ValueError("block_pixels must be positive")
        if self.layout not in {"NtB", "BNt"}:
            raise ValueError("layout must be 'NtB' or 'BNt'")
        if self.workspace_owner not in {"per_call", "per_factory"}:
            raise ValueError("workspace_owner must be per_call or per_factory")
        if self.reduction_policy != "ordered_species_reference_global":
            raise ValueError("unsupported reduction policy")
        if self.implementation in {"reference", "numba_serial"} and self.kernel_threads != 1:
            raise ValueError("reference and serial implementations require kernel_threads=1")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class WorkspaceStamp:
    """Identity required before a caller may reuse a derived workspace."""

    field_version: int
    shape: tuple[int, ...]
    dtype: str
    layout: Layout


class _IonizationRateScratch:
    """One propagator's reusable nonescaping species-rate packing buffer."""

    def __init__(self):
        self.array = None
        self.key = None
        self.allocations = 0
        self.reuses = 0

    def acquire(self, shape, dtype, *, layout):
        key = (tuple(int(value) for value in shape), np.dtype(dtype).str, str(layout))
        if self.array is not None and self.key == key:
            self.reuses += 1
            return self.array, True
        self.array = np.empty(key[0], dtype=np.dtype(dtype))
        self.key = key
        self.allocations += 1
        return self.array, False


def workspace_matches(
    stamp: WorkspaceStamp,
    *,
    field_version: int,
    shape: tuple[int, ...],
    dtype: str,
    layout: str,
) -> bool:
    return (
        stamp.field_version == int(field_version)
        and stamp.shape == tuple(shape)
        and stamp.dtype == str(dtype)
        and stamp.layout == str(layout)
    )


def compare_numeric(reference, candidate) -> dict:
    """Return exactness and scale-explicit error metrics without a tolerance."""
    from tools.cpu_p1r2.cpu_p1r2_j1 import _canonical_hash

    ref = np.asarray(reference)
    got = np.asarray(candidate)
    if ref.shape != got.shape:
        raise ValueError(f"shape mismatch: {ref.shape} != {got.shape}")
    diff = np.abs(got - ref)
    ref_abs = np.abs(ref)
    ref_max = float(np.max(ref_abs)) if ref.size else 0.0
    ref_l2 = float(np.linalg.norm(ref.ravel())) if ref.size else 0.0
    diff_max = float(np.max(diff)) if diff.size else 0.0
    diff_l2 = float(np.linalg.norm(diff.ravel())) if diff.size else 0.0
    if ref_max > 0.0:
        normalized_linf = diff_max / ref_max
        normalized_linf_reason = None
    elif diff_max == 0.0:
        normalized_linf = 0.0
        normalized_linf_reason = None
    else:
        normalized_linf = None
        normalized_linf_reason = "zero_reference_scale_with_nonzero_difference"
    if ref_l2 > 0.0:
        relative_l2 = diff_l2 / ref_l2
        relative_l2_reason = None
    elif diff_l2 == 0.0:
        relative_l2 = 0.0
        relative_l2_reason = None
    else:
        relative_l2 = None
        relative_l2_reason = "zero_reference_scale_with_nonzero_difference"
    zero_mask = ref == 0
    elementwise_relative = np.full(ref.shape, np.nan, dtype=np.float64)
    np.divide(diff, ref_abs, out=elementwise_relative, where=~zero_mask)
    reference_hash = _canonical_hash(ref)
    candidate_hash = _canonical_hash(got)
    return {
        "shape": list(ref.shape),
        "element_count": int(ref.size),
        "reference_shape": list(ref.shape),
        "candidate_shape": list(got.shape),
        "reference_dtype": str(ref.dtype),
        "candidate_dtype": str(got.dtype),
        "dtype_equal": bool(ref.dtype == got.dtype),
        "reference_canonical_sha256": reference_hash,
        "candidate_canonical_sha256": candidate_hash,
        "canonical_exact": bool(reference_hash == candidate_hash),
        "value_equal": bool(np.array_equal(ref, got)),
        "bitwise_equal": bool(ref.dtype == got.dtype and ref.tobytes() == got.tobytes()),
        "all_finite": bool(np.all(np.isfinite(ref)) and np.all(np.isfinite(got))),
        "max_abs": diff_max,
        "reference_max_abs": ref_max,
        "normalized_linf": normalized_linf,
        "normalized_linf_reason": normalized_linf_reason,
        "relative_l2": relative_l2,
        "relative_l2_reason": relative_l2_reason,
        "elementwise_relative_error": elementwise_relative,
        "zero_reference_mask": zero_mask,
        "zero_reference_count": int(np.count_nonzero(zero_mask)),
    }


def compare_numeric_bounded(
    reference,
    candidate,
    *,
    max_working_bytes: int = 64 * 1024 * 1024,
    max_examples: int = 128,
    zero_mask_path: str | os.PathLike | None = None,
) -> dict:
    """Compare arrays using bounded chunks and a compact zero-reference mask.

    The authoritative canonical hash helper is still used for both arrays.
    Difference, norm, and zero-mask work is streamed in chunks sized so the
    comparison temporaries stay below ``max_working_bytes``. For large masks,
    pass a new sidecar path; this avoids building a large JSON field. Hash-phase
    Python/NumPy traced allocations are measured when this call can own
    tracemalloc; process RSS is reported only as non-isolated endpoint samples.
    """
    import math
    import tracemalloc

    from tools.cpu_p1r2.cpu_p1r2_j1 import _canonical_hash

    if (isinstance(max_working_bytes, bool) or not isinstance(max_working_bytes, int)
            or not 1024 <= max_working_bytes <= MAX_BOUNDED_COMPARISON_BYTES):
        raise ValueError("max_working_bytes must be an integer in [1024, 64 MiB]")
    if (isinstance(max_examples, bool) or not isinstance(max_examples, int)
            or not 0 <= max_examples <= MAX_BOUNDED_DIFFERENCE_EXAMPLES):
        raise ValueError("max_examples must be an integer in [0, 128]")
    ref = np.asarray(reference)
    got = np.asarray(candidate)
    if ref.shape != got.shape:
        raise ValueError(f"shape mismatch: {ref.shape} != {got.shape}")
    if ref.dtype.hasobject or got.dtype.hasobject:
        raise TypeError("bounded comparison does not accept object arrays")
    if ref.dtype.kind not in "fc" or got.dtype.kind not in "fc":
        raise TypeError("bounded comparison accepts only floating or complex arrays")
    packed_mask_bytes = (int(ref.size) + 7) // 8
    inline_mask_limit = 4 * 1024 * 1024
    if zero_mask_path is None and packed_mask_bytes > inline_mask_limit:
        raise ValueError("large zero-reference masks require a new zero_mask_path sidecar")

    rss_process = None
    rss_before = None
    rss_after = None
    rss_error = None
    try:
        import psutil
    except ImportError:
        psutil = None
        rss_error = "psutil_not_installed"
    if psutil is not None:
        try:
            rss_process = psutil.Process(os.getpid())
            rss_before = int(rss_process.memory_info().rss)
        except (psutil.Error, OSError) as exc:
            rss_process = None
            rss_error = type(exc).__name__

    tracemalloc_was_active = tracemalloc.is_tracing()
    tracemalloc_started_here = False
    traced_baseline = None
    traced_peak_delta = None
    traced_current_delta = None
    tracemalloc_status = "UNMEASURED_EXTERNAL_TRACEMALLOC_ACTIVE"
    tracemalloc_unmeasured_reason = (
        "tracemalloc was already active; its caller-owned peak was not reset or read"
        if tracemalloc_was_active else None
    )
    if not tracemalloc_was_active:
        try:
            tracemalloc.start()
            tracemalloc_started_here = tracemalloc.is_tracing()
            if tracemalloc_started_here:
                traced_baseline, _ = tracemalloc.get_traced_memory()
                tracemalloc.reset_peak()
        except RuntimeError as exc:
            tracemalloc_status = "UNMEASURED_TRACEMALLOC_START_FAILED"
            tracemalloc_unmeasured_reason = f"tracemalloc_start_failed:{type(exc).__name__}"

    hash_started = time.perf_counter()
    try:
        reference_hash = _canonical_hash(ref)
        candidate_hash = _canonical_hash(got)
        hash_s = float(time.perf_counter() - hash_started)
        if tracemalloc_started_here:
            traced_current, traced_peak = tracemalloc.get_traced_memory()
            traced_current_delta = max(0, int(traced_current - traced_baseline))
            traced_peak_delta = max(0, int(traced_peak - traced_baseline))
            tracemalloc_status = "MEASURED_ISOLATED_TRACED_PEAK"
    finally:
        if tracemalloc_started_here and tracemalloc.is_tracing():
            tracemalloc.stop()
    if rss_process is not None:
        try:
            rss_after = int(rss_process.memory_info().rss)
        except (psutil.Error, OSError) as exc:
            rss_error = type(exc).__name__
    rss_delta = (int(rss_after - rss_before)
                 if rss_before is not None and rss_after is not None else None)

    # nditer buffers non-contiguous inputs instead of materializing full
    # flattened copies. Leave headroom for abs, difference, and mask arrays.
    item_bytes = max(ref.dtype.itemsize, got.dtype.itemsize, 1)
    elements_per_chunk = max(8, max_working_bytes // (item_bytes * 16))
    elements_per_chunk -= elements_per_chunk % 8
    iterator = np.nditer(
        [ref, got],
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=[["readonly"], ["readonly"]],
        order="C",
        buffersize=elements_per_chunk,
    )
    ref_max = 0.0
    diff_max = 0.0
    reference_lassq_scale = 0.0
    reference_lassq_sumsq = 1.0
    difference_lassq_scale = 0.0
    difference_lassq_sumsq = 1.0
    zero_count = 0
    finite_pair_count = 0
    nonfinite_mismatch_count = 0
    reference_operand_nonfinite_count = 0
    candidate_operand_nonfinite_count = 0
    reference_operand_infinity_count = 0
    candidate_operand_infinity_count = 0
    same_nonfinite_value_count = 0
    finite_subtraction_overflow_count = 0
    reference_magnitude_overflow_count = 0
    candidate_magnitude_overflow_count = 0
    difference_magnitude_overflow_count = 0
    normalization_overflow_count = 0
    zero_reference_nonzero_difference_count = 0
    value_equal = True
    bitwise_equal = ref.dtype == got.dtype
    all_finite = True
    reference_all_nan = bool(ref.size)
    candidate_all_nan = bool(got.size)
    mask_hash = hashlib.sha256()
    mask_compressor = zlib.compressobj(level=6)
    compressed_mask_parts = []
    mask_target = None
    mask_stream = None
    if zero_mask_path is not None:
        mask_target = os.path.abspath(os.fspath(zero_mask_path))
        os.makedirs(os.path.dirname(mask_target) or ".", exist_ok=True)
        mask_stream = open(mask_target, "xb")
    worst = []
    flat_offset = 0
    pending_mask_bits = np.empty((0,), dtype=bool)

    def component_equal(left, right):
        with np.errstate(invalid="ignore"):
            return (left == right) | (np.isnan(left) & np.isnan(right))

    def same_value_mask(left, right):
        if left.dtype.kind == "c" or right.dtype.kind == "c":
            return (component_equal(np.real(left), np.real(right))
                    & component_equal(np.imag(left), np.imag(right)))
        return component_equal(left, right)

    def stable_ratio(numerator, denominator):
        """Return a representable positive ratio, or an explicit reason."""
        if numerator == 0.0:
            return 0.0, None
        if denominator == 0.0:
            return None, "zero_reference_scale_with_nonzero_difference"
        if not (math.isfinite(numerator) and math.isfinite(denominator)):
            return None, "nonfinite_operand_or_magnitude"
        n_mantissa, n_exponent = math.frexp(numerator)
        d_mantissa, d_exponent = math.frexp(denominator)
        try:
            value = math.ldexp(n_mantissa / d_mantissa, n_exponent - d_exponent)
        except OverflowError:
            return None, "normalization_overflow"
        if not math.isfinite(value):
            return None, "normalization_overflow"
        if value == 0.0:
            return None, "normalization_underflow"
        return float(value), None

    def stable_l2_ratio(diff_scale, diff_sumsq, ref_scale, ref_sumsq):
        if diff_scale == 0.0:
            return 0.0, None
        if ref_scale == 0.0:
            return None, "zero_reference_scale_with_nonzero_difference"
        log_ratio = (math.log(diff_scale) - math.log(ref_scale)
                     + 0.5 * (math.log(diff_sumsq) - math.log(ref_sumsq)))
        if log_ratio > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
            return None, "normalization_overflow"
        if log_ratio < math.log(float.fromhex("0x0.0000000000001p-1022")):
            return None, "normalization_underflow"
        value = math.exp(log_ratio)
        if not math.isfinite(value):
            return None, "normalization_overflow"
        if value == 0.0:
            return None, "normalization_underflow"
        return float(value), None

    def emit_mask_bits(bits):
        packed = np.packbits(bits, bitorder="little").tobytes()
        if not packed:
            return
        mask_hash.update(packed)
        compressed = mask_compressor.compress(packed)
        if mask_stream is not None:
            if compressed:
                mask_stream.write(compressed)
        elif compressed:
            compressed_mask_parts.append(compressed)

    def lassq_update(scale, sumsq, values):
        magnitudes = np.abs(values).astype(np.float64, copy=False).ravel()
        finite = magnitudes[np.isfinite(magnitudes)]
        if finite.size == 0:
            return scale, sumsq
        local = float(np.max(finite))
        if local == 0.0:
            return scale, sumsq
        local_sumsq = float(np.sum((finite / local) ** 2, dtype=np.float64))
        if scale == 0.0:
            return local, local_sumsq
        if scale < local:
            return local, local_sumsq + sumsq * (scale / local) ** 2
        return scale, sumsq + local_sumsq * (local / scale) ** 2

    for ref_chunk_raw, got_chunk_raw in iterator:
        ref_chunk = np.asarray(ref_chunk_raw)
        got_chunk = np.asarray(got_chunk_raw)
        zero_mask = np.equal(ref_chunk, 0)
        zero_count += int(np.count_nonzero(zero_mask))
        mask_bits = np.concatenate((pending_mask_bits, zero_mask.ravel()))
        complete_bits = (mask_bits.size // 8) * 8
        emit_mask_bits(mask_bits[:complete_bits])
        pending_mask_bits = mask_bits[complete_bits:].copy()

        with np.errstate(invalid="ignore", over="ignore"):
            ref_mag = np.abs(ref_chunk)
            got_mag = np.abs(got_chunk)
        finite_ref = np.isfinite(ref_chunk)
        finite_got = np.isfinite(got_chunk)
        finite_pair = finite_ref & finite_got
        reference_operand_nonfinite_count += int(np.count_nonzero(~finite_ref))
        candidate_operand_nonfinite_count += int(np.count_nonzero(~finite_got))
        reference_operand_infinity_count += int(np.count_nonzero(np.isinf(ref_chunk)))
        candidate_operand_infinity_count += int(np.count_nonzero(np.isinf(got_chunk)))
        with np.errstate(invalid="ignore"):
            reference_all_nan = reference_all_nan and bool(ref_chunk.size and np.isnan(ref_chunk).all())
            candidate_all_nan = candidate_all_nan and bool(got_chunk.size and np.isnan(got_chunk).all())
        reference_magnitude_overflow = finite_ref & ~np.isfinite(ref_mag)
        candidate_magnitude_overflow = finite_got & ~np.isfinite(got_mag)
        reference_magnitude_overflow_count += int(np.count_nonzero(reference_magnitude_overflow))
        candidate_magnitude_overflow_count += int(np.count_nonzero(candidate_magnitude_overflow))
        all_finite = all_finite and bool(np.all(finite_ref)) and bool(np.all(finite_got))
        same_value = same_value_mask(ref_chunk, got_chunk)
        value_equal = value_equal and bool(np.all(same_value))
        if bitwise_equal:
            # nditer exposes bounded chunks even for strided source arrays.
            bitwise_equal = (
                np.ascontiguousarray(ref_chunk).tobytes()
                == np.ascontiguousarray(got_chunk).tobytes()
            )
        finite_pair_count += int(np.count_nonzero(finite_pair))
        with np.errstate(invalid="ignore", over="ignore"):
            raw_difference = got_chunk - ref_chunk
            difference = np.abs(raw_difference)
        finite_subtraction_overflow = finite_pair & ~np.isfinite(raw_difference)
        finite_subtraction_overflow_count += int(np.count_nonzero(finite_subtraction_overflow))
        difference_magnitude_overflow = (
            finite_pair & np.isfinite(raw_difference) & ~np.isfinite(difference)
        )
        difference_magnitude_overflow_count += int(np.count_nonzero(difference_magnitude_overflow))
        diff_finite = np.isfinite(difference)
        same_nonfinite = ~finite_pair & same_value
        same_nonfinite_value_count += int(np.count_nonzero(same_nonfinite))
        nonfinite_mismatch = (~finite_pair) & ~same_nonfinite
        nonfinite_mismatch_count += int(np.count_nonzero(nonfinite_mismatch))
        finite_ref_magnitude = finite_ref & np.isfinite(ref_mag)
        if np.any(finite_ref_magnitude):
            ref_max = max(ref_max, float(np.max(ref_mag[finite_ref_magnitude])))
            reference_lassq_scale, reference_lassq_sumsq = lassq_update(
                reference_lassq_scale, reference_lassq_sumsq, ref_chunk[finite_ref_magnitude]
            )
        if np.any(diff_finite):
            diff_max = max(diff_max, float(np.max(difference[diff_finite])))
            difference_lassq_scale, difference_lassq_sumsq = lassq_update(
                difference_lassq_scale, difference_lassq_sumsq, difference[diff_finite]
            )

        denominator = np.where(zero_mask, 1.0, ref_mag)
        relative = np.zeros(difference.shape, dtype=np.float64)
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            np.divide(difference, denominator, out=relative, where=~zero_mask)
        zero_diff = zero_mask & (difference != 0)
        zero_reference_nonzero_difference_count += int(np.count_nonzero(zero_diff))
        relative[zero_diff] = np.inf
        normalization_overflow = (
            finite_pair & ~zero_mask & np.isfinite(difference) & ~np.isfinite(relative)
        )
        normalization_overflow_count += int(np.count_nonzero(normalization_overflow))
        bad = (finite_pair & (difference != 0)) | nonfinite_mismatch
        if max_examples and np.any(bad):
            local_indices = np.flatnonzero(bad)
            if local_indices.size > max_examples * 4:
                local_scores = np.nan_to_num(
                    relative[local_indices], nan=np.inf, posinf=np.inf, neginf=np.inf
                )
                keep = np.argpartition(local_scores, -max_examples)[-max_examples:]
                local_indices = local_indices[keep]
            for local_index in local_indices:
                score = float(relative[local_index])
                flat_index = flat_offset + int(local_index)
                record = (
                    score,
                    flat_index,
                    float(difference[local_index]) if np.isfinite(difference[local_index]) else None,
                )
                worst.append(record)
            if len(worst) > max_examples * 2:
                worst = sorted(worst, key=lambda item: (item[0], item[1]))[-max_examples:]
        flat_offset += int(ref_chunk.size)

    if pending_mask_bits.size:
        emit_mask_bits(pending_mask_bits)
    tail = mask_compressor.flush()
    if mask_stream is not None:
        if tail:
            mask_stream.write(tail)
        mask_stream.flush()
        mask_stream.close()
    elif tail:
        compressed_mask_parts.append(tail)
    packed_mask_blob = b"".join(compressed_mask_parts) if mask_stream is None else None
    if ref_max > 0.0:
        normalized_linf, normalized_reason = stable_ratio(diff_max, ref_max)
    elif diff_max == 0.0:
        normalized_linf, normalized_reason = 0.0, None
    else:
        normalized_linf, normalized_reason = None, "zero_reference_scale_with_nonzero_difference"
    if reference_lassq_scale > 0.0:
        relative_l2, relative_reason = stable_l2_ratio(
            difference_lassq_scale, difference_lassq_sumsq,
            reference_lassq_scale, reference_lassq_sumsq,
        )
    elif difference_lassq_scale == 0.0:
        relative_l2, relative_reason = 0.0, None
    else:
        relative_l2, relative_reason = None, "zero_reference_scale_with_nonzero_difference"
    metrics_have_nonfinite_operands = bool(
        reference_operand_nonfinite_count or candidate_operand_nonfinite_count
    )
    metrics_have_overflow = bool(
        finite_subtraction_overflow_count or reference_magnitude_overflow_count
        or candidate_magnitude_overflow_count or difference_magnitude_overflow_count
    )
    if metrics_have_nonfinite_operands:
        max_abs, max_abs_reason = None, "nonfinite_operands_present"
        reference_max_abs, reference_max_abs_reason = None, "nonfinite_operands_present"
        normalized_linf, normalized_reason = None, "nonfinite_operands_present"
        relative_l2, relative_reason = None, "nonfinite_operands_present"
    elif metrics_have_overflow:
        max_abs, max_abs_reason = None, "magnitude_or_difference_overflow"
        reference_max_abs, reference_max_abs_reason = None, "magnitude_or_difference_overflow"
        normalized_linf, normalized_reason = None, "magnitude_or_difference_overflow"
        relative_l2, relative_reason = None, "magnitude_or_difference_overflow"
    else:
        max_abs, max_abs_reason = float(diff_max), None
        reference_max_abs, reference_max_abs_reason = float(ref_max), None
    normalization_overflow_count += int(normalized_reason == "normalization_overflow")
    normalization_overflow_count += int(relative_reason == "normalization_overflow")
    sorted_worst = sorted(worst, key=lambda item: (item[0], item[1]), reverse=True)[:max_examples]
    examples = []
    for score, flat_index, absolute in sorted_worst:
        index = tuple(int(v) for v in np.unravel_index(flat_index, ref.shape)) if ref.shape else ()
        ref_value = ref[index].item() if ref.shape else ref.item()
        got_value = got[index].item() if got.shape else got.item()
        examples.append({
            "flat_index": flat_index,
            "index": list(index),
            "reference": repr(ref_value),
            "candidate": repr(got_value),
            "max_relative_error": score if np.isfinite(score) else None,
            "absolute_error": absolute,
            "zero_reference": bool(ref_value == 0),
        })
    return {
        "shape": list(ref.shape),
        "element_count": int(ref.size),
        "reference_shape": list(ref.shape),
        "candidate_shape": list(got.shape),
        "reference_dtype": str(ref.dtype),
        "candidate_dtype": str(got.dtype),
        "reference_canonical_sha256": reference_hash,
        "candidate_canonical_sha256": candidate_hash,
        "canonical_exact": bool(reference_hash == candidate_hash),
        "value_equal": bool(value_equal),
        "bitwise_equal": bool(bitwise_equal),
        "all_finite": bool(all_finite),
        "reference_all_nan": bool(reference_all_nan),
        "candidate_all_nan": bool(candidate_all_nan),
        "finite_pair_count": finite_pair_count,
        "nonfinite_mismatch_count": nonfinite_mismatch_count,
        "dtype_equal": bool(ref.dtype == got.dtype),
        "reference_operand_nonfinite_count": reference_operand_nonfinite_count,
        "candidate_operand_nonfinite_count": candidate_operand_nonfinite_count,
        "reference_operand_infinity_count": reference_operand_infinity_count,
        "candidate_operand_infinity_count": candidate_operand_infinity_count,
        "same_nonfinite_value_count": same_nonfinite_value_count,
        "finite_subtraction_overflow_count": finite_subtraction_overflow_count,
        "reference_magnitude_overflow_count": reference_magnitude_overflow_count,
        "candidate_magnitude_overflow_count": candidate_magnitude_overflow_count,
        "difference_magnitude_overflow_count": difference_magnitude_overflow_count,
        "normalization_overflow_count": normalization_overflow_count,
        "zero_reference_nonzero_difference_count": zero_reference_nonzero_difference_count,
        "max_abs": max_abs,
        "max_abs_reason": max_abs_reason,
        "reference_max_abs": reference_max_abs,
        "reference_max_abs_reason": reference_max_abs_reason,
        "normalized_linf": normalized_linf,
        "normalized_linf_reason": normalized_reason,
        "relative_l2": relative_l2,
        "relative_l2_reason": relative_reason,
        "zero_reference_count": zero_count,
        "zero_reference_mask_encoding": "zlib+base64(packbits_little)",
        "zero_reference_mask_shape": list(ref.shape),
        "zero_reference_mask_sha256": mask_hash.hexdigest(),
        "zero_reference_mask_zlib_base64": (
            base64.b64encode(packed_mask_blob).decode("ascii")
            if packed_mask_blob is not None else None
        ),
        "zero_reference_mask_sidecar": mask_target,
        "zero_reference_mask_status": "SIDECAR_WRITTEN" if mask_target else "INLINE_COMPRESSED",
        "worst_differences": examples,
        "comparison_chunk_limit_bytes": max_working_bytes,
        "canonical_hash_seconds": hash_s,
        "canonical_hash_input_bytes": int(ref.nbytes + got.nbytes),
        "canonical_hash_tracemalloc_status": tracemalloc_status,
        "canonical_hash_tracemalloc_unmeasured_reason": tracemalloc_unmeasured_reason,
        "canonical_hash_traced_peak_bytes": traced_peak_delta,
        "canonical_hash_traced_current_delta_bytes": traced_current_delta,
        "canonical_hash_traced_memory_scope": (
            "tracemalloc-tracked allocations; not process RSS or all native allocations"
        ),
        "canonical_hash_process_rss_before_bytes": rss_before,
        "canonical_hash_process_rss_after_bytes": rss_after,
        "canonical_hash_process_rss_delta_bytes": rss_delta,
        "canonical_hash_process_rss_status": (
            "ENDPOINTS_SAMPLED_NOT_HASH_PEAK" if rss_delta is not None else "UNAVAILABLE"
        ),
        "canonical_hash_process_rss_unavailable_reason": rss_error,
        "canonical_hash_process_rss_scope": (
            "whole-process RSS endpoints; not isolated to hash allocations and not a peak"
        ),
        "comparison_working_memory_scope": (
            "configured limit applies to metric chunks only; hash phase has separate traced-allocation and RSS endpoint fields"
        ),
    }


def local_environment_fingerprint() -> dict:
    """Return the toolchain identity used by CPU-O1 exact local fixtures."""
    import importlib.metadata
    import sys
    import sysconfig

    import numba
    try:
        import threadpoolctl
        threadpools = threadpoolctl.threadpool_info()
        threadpoolctl_version = importlib.metadata.version("threadpoolctl")
    except (ImportError, importlib.metadata.PackageNotFoundError):
        threadpools = None
        threadpoolctl_version = None
    try:
        psutil_version = importlib.metadata.version("psutil")
    except importlib.metadata.PackageNotFoundError:
        psutil_version = None

    versions = {}
    for package in ("numpy", "scipy", "numba", "llvmlite", "pytest"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        active_threading_layer = str(numba.threading_layer())
    except ValueError:
        active_threading_layer = "NOT_INITIALIZED"
    runtime = {
        "numba_cpu_target": str(numba.config.CPU_NAME),
        "numba_cpu_target_env": os.environ.get("NUMBA_CPU_NAME"),
        "numba_threading_layer_config": str(numba.config.THREADING_LAYER),
        "numba_threading_layer_env": os.environ.get("NUMBA_THREADING_LAYER"),
        "numba_active_threading_layer": active_threading_layer,
        "numba_threading_layer_priority": list(numba.config.THREADING_LAYER_PRIORITY),
        "numba_threads_available": int(numba.config.NUMBA_NUM_THREADS),
        "numba_threads_current": int(numba.get_num_threads()),
        "parallel_call_organization": "NOT_OBSERVED_BY_ENVIRONMENT_FINGERPRINT",
        "parallel_call_policy": (
            "Q1 serial identity sequence; one active child; Numba kernel and FFT workers are not nested"
        ),
    }
    expected_versions = {
        "numpy": "2.2.5", "scipy": "1.15.2", "numba": "0.61.2",
        "llvmlite": "0.44.0", "pytest": "8.3.5",
    }
    qualification_reasons = []
    if platform.python_version() != "3.12.8":
        qualification_reasons.append("python_version_mismatch")
    if platform.system() != "Windows":
        qualification_reasons.append("platform_not_windows")
    if platform.machine().lower() not in {"amd64", "x86_64"}:
        qualification_reasons.append("machine_not_amd64")
    if versions != expected_versions:
        qualification_reasons.append("package_versions_mismatch")
    if runtime["numba_cpu_target"] != "generic" or runtime["numba_cpu_target_env"] != "generic":
        qualification_reasons.append("numba_target_not_generic")
    if (runtime["numba_threading_layer_config"] != "workqueue"
            or runtime["numba_threading_layer_env"] != "workqueue"):
        qualification_reasons.append("numba_threading_layer_not_workqueue")
    if (active_threading_layer not in {"NOT_INITIALIZED", "workqueue"}):
        qualification_reasons.append("active_threading_layer_mismatch")
    glibc = platform.libc_ver()
    linux_reasons = []
    expected_linux_versions = {
        **expected_versions, "threadpoolctl": "3.6.0", "psutil": "7.0.0",
    }
    observed_linux_versions = dict(versions)
    observed_linux_versions.update({"threadpoolctl": threadpoolctl_version, "psutil": psutil_version})
    if platform.python_version() != "3.12.8":
        linux_reasons.append("python_version_mismatch")
    if platform.python_implementation() != "CPython":
        linux_reasons.append("python_implementation_mismatch")
    if platform.system() != "Linux":
        linux_reasons.append("platform_not_linux")
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        linux_reasons.append("machine_not_x86_64")
    if sys.byteorder != "little":
        linux_reasons.append("byteorder_not_little")
    if observed_linux_versions != expected_linux_versions:
        linux_reasons.append("locked_package_versions_mismatch")
    if glibc[0].lower() != "glibc" or not glibc[1]:
        linux_reasons.append("glibc_identity_unavailable")
    if not sysconfig.get_config_var("SOABI") or not sysconfig.get_config_var("EXT_SUFFIX"):
        linux_reasons.append("python_abi_identity_unavailable")
    if runtime["numba_cpu_target"] != "generic" or runtime["numba_cpu_target_env"] != "generic":
        linux_reasons.append("numba_target_not_generic")
    if (runtime["numba_threading_layer_config"] != "workqueue"
            or runtime["numba_threading_layer_env"] != "workqueue"):
        linux_reasons.append("numba_threading_layer_not_workqueue")
    if active_threading_layer != "workqueue":
        linux_reasons.append("numba_active_workqueue_not_proven")
    cache_value = os.environ.get("NUMBA_CACHE_DIR")
    cache_path = Path(cache_value).expanduser() if cache_value else None
    cache_writable = bool(cache_path is not None and cache_path.is_dir()
                          and os.access(cache_path, os.W_OK))
    if not cache_writable:
        linux_reasons.append("numba_cache_not_writable_or_unconfigured")
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "platform_system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "versions": versions,
        "additional_runtime_versions": {
            "threadpoolctl": threadpoolctl_version, "psutil": psutil_version,
        },
        "byteorder": sys.byteorder,
        "runtime": runtime,
        "linux_runtime_identity": {
            "libc_name": glibc[0], "libc_version": glibc[1],
            "python_soabi": sysconfig.get_config_var("SOABI"),
            "python_ext_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
            "threadpool_info": threadpools,
            "numba_cache_dir": str(cache_path) if cache_path is not None else None,
            "numba_cache_dir_exists": bool(cache_path is not None and cache_path.is_dir()),
            "numba_cache_dir_writable": cache_writable,
        },
        "exact_regression_qualification": {
            "eligible": not qualification_reasons,
            "status": "PINNED_WINDOWS_AMD64_GENERIC_WORKQUEUE"
            if not qualification_reasons else "UNQUALIFIED_ENVIRONMENT",
            "reasons": qualification_reasons,
        },
        "linux_q1_qualification": {
            "eligible": not linux_reasons,
            "status": "PINNED_LINUX_X86_64_GENERIC_WORKQUEUE"
            if not linux_reasons else "LINUX_Q1_NOT_QUALIFIED",
            "reasons": linux_reasons,
            "expected_locked_versions": expected_linux_versions,
        },
    }


def make_base_cpufft_factory(plan: CPUExecutionPlan | None = None):
    """Return the existing scoped BASE_CPUFFT factory without global mutation.

    The default dispatch preserves the measured BASE_CPUFFT identity (48
    temporal / 24 spatial workers). An explicit plan may select another pair.
    """
    from tools.cpu_p1r2.cpu_p1r2_j2 import make_propagator_factory

    selected = plan or CPUExecutionPlan(temporal_workers=48, spatial_workers=24)
    return make_propagator_factory({
        "temporal_workers": selected.temporal_workers,
        "spatial_workers": selected.spatial_workers,
    })


def _validate_p1r2_source_binding() -> dict:
    """Fail closed if any source used by the P1-R2 clone drifted."""
    repo_root = Path(__file__).resolve().parents[3]

    def git(*args):
        result = subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True, text=True,
            encoding="utf-8", errors="strict", check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "P1-R2 source binding could not be verified: "
                f"git {' '.join(args)} exited {result.returncode}: {result.stderr.strip()}"
            )
        return result.stdout.strip()

    anchor_tree = git("rev-parse", "--verify", f"{P1R2_SOURCE_ANCHOR_COMMIT}^{{tree}}")
    if anchor_tree != P1R2_SOURCE_ANCHOR_TREE:
        raise RuntimeError("P1-R2 source anchor tree identity changed")
    source_blobs = {}
    for relative_path, expected_blob in sorted(P1R2_SOURCE_BLOBS.items()):
        actual_blob = git("rev-parse", f"{P1R2_SOURCE_ANCHOR_COMMIT}:{relative_path}")
        if actual_blob != expected_blob:
            raise RuntimeError(f"P1-R2 source anchor blob mismatch: {relative_path}")
        source_blobs[relative_path] = actual_blob
    drift = subprocess.run(
        ["git", "diff", "--quiet", P1R2_SOURCE_ANCHOR_COMMIT, "--",
         *sorted(P1R2_SOURCE_BLOBS)],
        cwd=repo_root, capture_output=True, check=False,
    )
    if drift.returncode != 0:
        raise RuntimeError("P1-R2 clone or frozen source files drifted from the pinned anchor")
    return {
        "anchor_commit": P1R2_SOURCE_ANCHOR_COMMIT,
        "anchor_tree": anchor_tree,
        "source_blobs": source_blobs,
        "SOURCE_SHA": SOURCE_SHA,
        "working_tree_matches_anchor": True,
        "transformation": "P1R2_FunctionType_private_clone_no_AST_splice",
    }


def make_joint_propagator_factory(
    identity: Literal["B", "I", "R", "IR"],
    *,
    kernel_threads: int = 1,
    block_pixels: int = 32,
    layout: Layout = "NtB",
    temporal_workers: int = 48,
    spatial_workers: int = 24,
):
    """Return a fresh private propagator for B/I/R/IR local comparisons.

    The default identity uses the measured 48/24 temporal/spatial FFT
    dispatch. Explicit 1/1 is reserved for the Linux login-node tiny gate;
    the real Q1 coordinator requests 48/24.
    Ionization and Raman replacements are installed only in the cloned
    propagator globals returned by the P1-R2 factory.
    """
    if identity not in {"B", "I", "R", "IR"}:
        raise ValueError("identity must be one of B/I/R/IR")
    if isinstance(kernel_threads, bool) or not isinstance(kernel_threads, int):
        raise ValueError("kernel_threads must be an integer")
    if not 1 <= kernel_threads <= 96:
        raise ValueError("kernel_threads must be in [1, 96]")
    if isinstance(block_pixels, bool) or not isinstance(block_pixels, int) or block_pixels < 1:
        raise ValueError("block_pixels must be a positive integer")
    if layout not in {"NtB", "BNt"}:
        raise ValueError("layout must be NtB or BNt")

    if (isinstance(temporal_workers, bool) or not isinstance(temporal_workers, int)
            or temporal_workers < 1 or temporal_workers > 96):
        raise ValueError("temporal_workers must be an integer in [1, 96]")
    if (isinstance(spatial_workers, bool) or not isinstance(spatial_workers, int)
            or spatial_workers < 1 or spatial_workers > 96):
        raise ValueError("spatial_workers must be an integer in [1, 96]")
    plan = CPUExecutionPlan(
        implementation="numba_parallel" if kernel_threads > 1 else "numba_serial",
        temporal_workers=temporal_workers,
        spatial_workers=spatial_workers,
        kernel_threads=kernel_threads,
        block_pixels=block_pixels,
        layout=layout,
        workspace_owner=("per_factory" if identity in {"I", "IR"} else "per_call"),
    )
    base_factory = make_base_cpufft_factory(
        CPUExecutionPlan(temporal_workers=temporal_workers,
                         spatial_workers=spatial_workers)
    )

    def factory(frozen_propagator):
        source_binding = _validate_p1r2_source_binding()
        propagator, dispatch = base_factory(frozen_propagator)
        from KHz_filament.propagate import propagate_one_pulse as source_propagator

        if getattr(propagator, "__j1_inner_code__", None) is not source_propagator.__code__:
            raise RuntimeError("P1-R2 private clone no longer carries the anchored propagator code")
        rate_workspace = _IonizationRateScratch() if identity in {"I", "IR"} else None
        events = []
        propagator._cpu_o1_candidate_events = events
        propagator._cpu_o1_identity = identity
        dispatch.cpu_o1_candidate_events = events
        if identity in {"I", "IR"}:
            original_plan = plan

            def evolve_candidate(*args, **kwargs):
                metrics = {}
                if "metrics_out" in kwargs:
                    raise TypeError("P1-R2 propagator unexpectedly supplied metrics_out")
                result = evolve_rho_time_cpu_o1(
                    *args, plan=original_plan,
                    rate_workspace=rate_workspace, metrics_out=metrics, **kwargs,
                )
                events.append({"kind": "ionization", **metrics})
                return result

            propagator.__globals__["evolve_rho_time"] = evolve_candidate
        if identity in {"R", "IR"}:
            stage_reference = propagator.__globals__.get("isaacs_raman_stage")
            raman_ns = getattr(stage_reference, "__globals__", None)
            if not isinstance(raman_ns, dict) or "raman_convolve_intensity" not in raman_ns:
                raise RuntimeError("P1-R2 clone has no private Raman namespace")
            from KHz_filament.raman import isaacs_raman_stage as source_raman_stage

            if stage_reference.__code__ is not source_raman_stage.__code__:
                raise RuntimeError("P1-R2 private Raman stage drifted from its anchored source")
            dispatch.cpu_o1_raman_namespace = raman_ns

            def stage_candidate(field, **kwargs):
                metrics = {}
                result = isaacs_raman_stage_cpu_o1(
                    field, raman_namespace=raman_ns, plan=plan,
                    metrics_out=metrics, **kwargs,
                )
                events.append({"kind": "raman_stage", **metrics})
                return result

            def apply_candidate(field, dz, **kwargs):
                metrics = {}
                result = apply_isaacs_raman_step_cpu_o1(
                    field, dz, raman_namespace=raman_ns, plan=plan,
                    stage_function=stage_candidate, metrics_out=metrics, **kwargs,
                )
                events.append({"kind": "raman_apply", **metrics})
                return result

            propagator.__globals__["isaacs_raman_stage"] = stage_candidate
            propagator.__globals__["apply_isaacs_raman_operator_step"] = apply_candidate
        propagator._cpu_o1_execution = {
            "identity": identity,
            "temporal_workers": temporal_workers,
            "spatial_workers": spatial_workers,
            "kernel_threads": plan.kernel_threads,
            "block_pixels": plan.block_pixels,
            "layout": plan.layout,
            "workspace_owner": plan.workspace_owner,
            "source": "private_clone_only",
            "source_binding": source_binding,
            "propagator_bytecode_sha256": hashlib.sha256(
                propagator.__code__.co_code
            ).hexdigest(),
        }
        dispatch.cpu_o1_execution = dict(propagator._cpu_o1_execution)
        dispatch.cpu_o1_rate_workspace = rate_workspace
        dispatch.cpu_o1_candidate_events = events
        factory.last_propagator = propagator
        factory.last_dispatch = dispatch
        factory.last_rate_workspace = rate_workspace
        return propagator, dispatch

    factory.last_propagator = None
    factory.last_dispatch = None
    factory.last_rate_workspace = None
    return factory


def available_numba_threads() -> int:
    """Load the isolated compiler and report its configured thread ceiling."""
    numba, _, _ = _load_numba_kernels()
    return int(numba.config.NUMBA_NUM_THREADS)


def _load_numba_kernels():
    global _NUMBA_KERNELS, _NUMBA_PHOTO_CLIP
    if _NUMBA_KERNELS is not None:
        return _NUMBA_KERNELS

    # Numba reads its threading-layer setting during import. Scope the variable
    # to import time, then restore the caller's environment exactly.
    old_value = os.environ.get("NUMBA_THREADING_LAYER")
    had_value = "NUMBA_THREADING_LAYER" in os.environ
    if not had_value:
        os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
    try:
        import numba
    except Exception as exc:  # dependency and binary-load errors are surfaced clearly
        raise RuntimeError("CPU-O1 compiled path requires a working Numba installation") from exc
    finally:
        if had_value:
            os.environ["NUMBA_THREADING_LAYER"] = old_value  # type: ignore[arg-type]
        else:
            os.environ.pop("NUMBA_THREADING_LAYER", None)

    layer_setting = str(numba.config.THREADING_LAYER).lower()
    if layer_setting != "workqueue":
        raise RuntimeError(
            "CPU-O1 requires NUMBA_THREADING_LAYER=workqueue before Numba import; "
            f"current setting is {layer_setting!r}"
        )

    global prange
    prange = numba.prange
    photo_clip = numba.njit(_clip_photo_argument, cache=True, fastmath=False)

    def _kernel(rates, fractions, n0_species, beta_species, ui_species,
                block_pixels, layout_flag, diagnose_flag, zero, one, half_value,
                half_dt, rk_scale, two, dt_value, beta_rec_value):
        species_count = rates.shape[0]
        if layout_flag == 0:
            nt = rates.shape[1]
            pixels = rates.shape[2]
            species_values = np.zeros((species_count, nt, pixels), dtype=rates.dtype)
            rho_sum = np.zeros((nt, pixels), dtype=rates.dtype)
            wt_total = np.zeros((nt, pixels), dtype=rates.dtype)
            net_energy = np.zeros((nt, pixels), dtype=rates.dtype)
            photo_energy = np.zeros((nt, pixels), dtype=rates.dtype)
        else:
            pixels = rates.shape[1]
            nt = rates.shape[2]
            species_values = np.zeros((species_count, pixels, nt), dtype=rates.dtype)
            rho_sum = np.zeros((pixels, nt), dtype=rates.dtype)
            wt_total = np.zeros((pixels, nt), dtype=rates.dtype)
            net_energy = np.zeros((pixels, nt), dtype=rates.dtype)
            photo_energy = np.zeros((pixels, nt), dtype=rates.dtype)

        block_count = (pixels + block_pixels - 1) // block_pixels
        if diagnose_flag:
            step_min = np.empty((species_count, block_count), dtype=np.float64)
            step_max = np.empty((species_count, block_count), dtype=np.float64)
            intermediate_min = np.empty((species_count, block_count), dtype=np.float64)
            intermediate_max = np.empty((species_count, block_count), dtype=np.float64)
            step_clips = np.zeros((species_count, block_count), dtype=np.int64)
            intermediate_violations = np.zeros((species_count, block_count), dtype=np.int64)
        else:
            step_min = np.empty((species_count, 0), dtype=np.float64)
            step_max = np.empty((species_count, 0), dtype=np.float64)
            intermediate_min = np.empty((species_count, 0), dtype=np.float64)
            intermediate_max = np.empty((species_count, 0), dtype=np.float64)
            step_clips = np.empty((species_count, 0), dtype=np.int64)
            intermediate_violations = np.empty((species_count, 0), dtype=np.int64)
        for block_id in prange(block_count):
            start = block_id * block_pixels
            stop = min(start + block_pixels, pixels)
            if diagnose_flag:
                local_step_min = np.full(species_count, np.inf, dtype=np.float64)
                local_step_max = np.full(species_count, -np.inf, dtype=np.float64)
                local_intermediate_min = np.full(species_count, np.inf, dtype=np.float64)
                local_intermediate_max = np.full(species_count, -np.inf, dtype=np.float64)
                local_step_clips = np.zeros(species_count, dtype=np.int64)
                local_intermediate_violations = np.zeros(species_count, dtype=np.int64)

            for pixel in range(start, stop):
                for species in range(species_count):
                    u = zero
                    if layout_flag == 0:
                        species_values[species, 0, pixel] = zero
                    else:
                        species_values[species, pixel, 0] = zero
                    for it in range(nt - 1):
                        if layout_flag == 0:
                            w1 = rates[species, it, pixel]
                            w4 = rates[species, it + 1, pixel]
                        else:
                            w1 = rates[species, pixel, it]
                            w4 = rates[species, pixel, it + 1]
                        w2 = half_value * (w1 + w4)
                        beta_n = beta_species[species]
                        k1 = w1 * (one - u) - beta_n * (u * u)
                        u2 = u + half_dt * k1
                        k2 = w2 * (one - u2) - beta_n * (u2 * u2)
                        u3 = u + half_dt * k2
                        k3 = w2 * (one - u3) - beta_n * (u3 * u3)
                        u4 = u + dt_value * k3
                        k4 = w4 * (one - u4) - beta_n * (u4 * u4)
                        u_next = u + rk_scale * (k1 + two * k2 + two * k3 + k4)

                        if diagnose_flag:
                            for value in (u2, u3, u4):
                                if value < local_intermediate_min[species]:
                                    local_intermediate_min[species] = value
                                if value > local_intermediate_max[species]:
                                    local_intermediate_max[species] = value
                                if value < 0.0 or value > 1.0:
                                    local_intermediate_violations[species] += 1
                            if u_next < local_step_min[species]:
                                local_step_min[species] = u_next
                            if u_next > local_step_max[species]:
                                local_step_max[species] = u_next
                            if u_next < 0.0 or u_next > 1.0:
                                local_step_clips[species] += 1
                        u_next = np.maximum(u_next, zero)
                        u_next = np.minimum(u_next, one)
                        u = u_next
                        if layout_flag == 0:
                            species_values[species, it + 1, pixel] = u
                        else:
                            species_values[species, pixel, it + 1] = u

                for it in range(nt):
                    if layout_flag == 0:
                        rho0 = species_values[0, it, pixel] * n0_species[0]
                        wt0 = rates[0, it, pixel] * fractions[0]
                    else:
                        rho0 = species_values[0, pixel, it] * n0_species[0]
                        wt0 = rates[0, pixel, it] * fractions[0]
                    rho_value = rho0
                    wt_value = wt0
                    net_value = zero
                    photo_value = zero
                    for species in range(species_count):
                        if layout_flag == 0:
                            u_value = species_values[species, it, pixel]
                            rate_value = rates[species, it, pixel]
                        else:
                            u_value = species_values[species, pixel, it]
                            rate_value = rates[species, pixel, it]
                        rho_species = u_value * n0_species[species]
                        if layout_flag == 0:
                            species_values[species, it, pixel] = rho_species
                        if layout_flag == 1:
                            species_values[species, pixel, it] = rho_species
                        if species > 0:
                            rho_value = rho_value + rho_species
                            wt_value = wt_value + rate_value * fractions[species]
                        n0_value = n0_species[species]
                        lower = zero
                        upper = n0_value
                        photo_arg = photo_clip(n0_value - rho_species, lower, upper)
                        photo_creation = rate_value * photo_arg
                        net_density = photo_creation - beta_rec_value * (rho_species * rho_species)
                        net_value = net_value + ui_species[species] * net_density
                        photo_value = photo_value + ui_species[species] * photo_creation
                    photo_value = np.maximum(photo_value, zero)
                    if layout_flag == 0:
                        rho_sum[it, pixel] = rho_value
                        wt_total[it, pixel] = wt_value
                        net_energy[it, pixel] = net_value
                        photo_energy[it, pixel] = photo_value
                    else:
                        rho_sum[pixel, it] = rho_value
                        wt_total[pixel, it] = wt_value
                        net_energy[pixel, it] = net_value
                        photo_energy[pixel, it] = photo_value

            if diagnose_flag:
                for species in range(species_count):
                    step_min[species, block_id] = local_step_min[species]
                    step_max[species, block_id] = local_step_max[species]
                    intermediate_min[species, block_id] = local_intermediate_min[species]
                    intermediate_max[species, block_id] = local_intermediate_max[species]
                    step_clips[species, block_id] = local_step_clips[species]
                    intermediate_violations[species, block_id] = local_intermediate_violations[species]

        return (species_values, rho_sum, wt_total, net_energy, photo_energy,
                step_min, step_max, intermediate_min, intermediate_max,
                step_clips, intermediate_violations)

    # prange lowers to a serial range when parallel=False. Keeping one kernel
    # body ensures the scalar operation order is shared by both candidates.
    _serial = numba.njit(_kernel, cache=True, fastmath=False)
    _parallel = numba.njit(_kernel, cache=True, fastmath=False, parallel=True)
    _NUMBA_PHOTO_CLIP = photo_clip
    _NUMBA_KERNELS = (numba, _serial, _parallel)
    return _NUMBA_KERNELS


def _raman_product_body(response, field):
    ny, nx = field.shape[1], field.shape[2]
    output = np.empty(field.shape, dtype=field.dtype)
    for flat in prange(field.size):
        it = flat // (ny * nx)
        iy = (flat // nx) % ny
        ix = flat % nx
        output[it, iy, ix] = response[it, iy, ix] * field[it, iy, ix]
    return output


def _raman_rhs_body(product, normalized_derivative, prefactor):
    ny, nx = product.shape[1], product.shape[2]
    output = np.empty(product.shape, dtype=product.dtype)
    for flat in prange(product.size):
        it = flat // (ny * nx)
        iy = (flat // nx) % ny
        ix = flat % nx
        first = 1j * prefactor * product[it, iy, ix]
        second = prefactor * normalized_derivative[it, iy, ix]
        output[it, iy, ix] = first - second
    return output


def _raman_predictor_body(field, k1, dz):
    ny, nx = field.shape[1], field.shape[2]
    output = np.empty(field.shape, dtype=field.dtype)
    for flat in prange(field.size):
        it = flat // (ny * nx)
        iy = (flat // nx) % ny
        ix = flat % nx
        increment = dz * k1[it, iy, ix]
        output[it, iy, ix] = field[it, iy, ix] + increment
    return output


def _raman_heun_body(field, k1, k2, dz):
    ny, nx = field.shape[1], field.shape[2]
    output = np.empty(field.shape, dtype=field.dtype)
    half_dz = 0.5 * dz
    for flat in prange(field.size):
        it = flat // (ny * nx)
        iy = (flat // nx) % ny
        ix = flat % nx
        slope_sum = k1[it, iy, ix] + k2[it, iy, ix]
        increment = half_dz * slope_sum
        output[it, iy, ix] = field[it, iy, ix] + increment
    return output


def _raman_stable_expression_body(before, after):
    before_square = np.empty(before.shape, dtype=np.float64)
    loss_expression = np.empty(before.shape, dtype=np.float64)
    ny, nx = before.shape[1], before.shape[2]
    for flat in prange(before.size):
        it = flat // (ny * nx)
        iy = (flat // nx) % ny
        ix = flat % nx
        before_value = before[it, iy, ix]
        after_value = after[it, iy, ix]
        before_square[it, iy, ix] = before_value * before_value
        loss_expression[it, iy, ix] = (
            (before_value - after_value) * (before_value + after_value)
        )
    return before_square, loss_expression


def _load_raman_kernels():
    global _RAMAN_KERNELS
    if _RAMAN_KERNELS is not None:
        return _RAMAN_KERNELS
    numba, _, _ = _load_numba_kernels()
    bodies = {
        "product": _raman_product_body,
        "rhs": _raman_rhs_body,
        "predictor": _raman_predictor_body,
        "heun": _raman_heun_body,
        "stable_expression": _raman_stable_expression_body,
    }
    kernels = {"numba": numba}
    for name, body in bodies.items():
        kernels[name] = (
            numba.njit(body, cache=True, fastmath=False),
            numba.njit(body, cache=True, fastmath=False, parallel=True),
        )
    _RAMAN_KERNELS = kernels
    return kernels


def _call_raman_kernel(name, args, *, threads):
    numba, _, _ = _load_numba_kernels()
    kernels = _load_raman_kernels()
    serial_kernel, parallel_kernel = kernels[name]
    kernel = serial_kernel if threads == 1 else parallel_kernel
    signature = tuple(numba.typeof(argument) for argument in args)
    compiled_before = signature in kernel.signatures
    compilation_started = time.perf_counter()
    if not compiled_before:
        kernel.compile(signature)
    compilation_s = time.perf_counter() - compilation_started
    old_threads = None
    if threads > 1:
        if threads > int(numba.config.NUMBA_NUM_THREADS):
            raise ValueError(
                f"requested {threads} threads exceeds Numba limit "
                f"{numba.config.NUMBA_NUM_THREADS}"
            )
        old_threads = int(numba.get_num_threads())
        numba.set_num_threads(threads)
    started = time.perf_counter()
    try:
        result = kernel(*args)
    finally:
        if old_threads is not None:
            numba.set_num_threads(old_threads)
    return result, {
        "kernel": name,
        "kernel_threads": threads,
        "cold_start": not compiled_before,
        "compile_or_cache_load_s": float(compilation_s),
        "kernel_s": float(time.perf_counter() - started),
        "threading_layer": (
            str(numba.threading_layer()) if threads > 1 else "workqueue_not_initialized"
        ),
    }


def _require_raman_candidate_inputs(field, kwargs, plan, raman_namespace):
    if not isinstance(field, np.ndarray) or field.ndim != 3:
        raise ValueError("CPU-O1 Raman candidate requires a NumPy [Nt,Ny,Nx] field")
    if field.dtype != np.dtype("complex128"):
        raise ValueError("CPU-O1 Raman candidate currently supports complex128 only")
    if any(int(size) < 1 for size in field.shape):
        raise ValueError("CPU-O1 Raman candidate requires positive field dimensions")
    if not np.all(np.isfinite(field)):
        raise ValueError("CPU-O1 Raman candidate requires finite field values")
    if str(kwargs.get("method", "iir")).lower() not in {"iir", "fft"}:
        raise ValueError("CPU-O1 Raman candidate supports only the frozen IIR or FFT convolution")
    if kwargs.get("iir_sampling", "exact_piecewise_linear") != "exact_piecewise_linear":
        raise ValueError("CPU-O1 Raman candidate requires exact_piecewise_linear IIR semantics")
    xp_proxy = raman_namespace.get("xp")
    xp = getattr(xp_proxy, "xp", xp_proxy)
    if getattr(xp, "__name__", None) != "numpy":
        raise ValueError("CPU-O1 Raman candidate requires the NumPy backend")
    if plan.implementation not in {"numba_serial", "numba_parallel"}:
        raise ValueError("CPU-O1 Raman candidate requires an explicit compiled implementation")
    return xp


def isaacs_raman_stage_cpu_o1(
    field,
    *,
    raman_namespace: dict,
    plan: CPUExecutionPlan,
    metrics_out: dict | None = None,
    **kwargs,
):
    """Private local Raman stage with compiled pointwise product and RHS.

    The frozen exact-PWL IIR, FFTs, global reductions, intensity expression,
    and energy accumulation remain in the cloned reference namespace.
    """
    from KHz_filament.constants import c0, eps0

    if metrics_out is not None and not isinstance(metrics_out, dict):
        raise TypeError("metrics_out must be a dictionary when supplied")
    xp = _require_raman_candidate_inputs(field, kwargs, plan, raman_namespace)
    xp_proxy = raman_namespace.get("xp")
    fft = getattr(xp_proxy, "fft", xp.fft)
    required = {"Omega", "dt", "omega0", "n0", "n_R", "omega_R", "Gamma_R"}
    missing = required.difference(kwargs)
    if missing:
        raise TypeError(f"missing Raman stage arguments: {sorted(missing)}")
    started = time.perf_counter()
    rdtype = xp.float64
    intensity_started = time.perf_counter()
    intensity = (
        0.5 * float(eps0) * float(c0) * float(kwargs["n0"]) * xp.abs(field) ** 2
    ).astype(rdtype, copy=False)
    intensity_s = time.perf_counter() - intensity_started
    convolution_started = time.perf_counter()
    method = str(kwargs.get("method", "iir")).lower()
    if method == "fft":
        tau = xp.arange(field.shape[0], dtype=xp.float64) * float(kwargs["dt"])
        kernel = raman_namespace["make_raman_kernel"](tau, {
            "model": "isaacs_rot_sinexp",
            "omega_R": kwargs["omega_R"],
            "Gamma_R": kwargs["Gamma_R"],
        })
        response = raman_namespace["raman_convolve_intensity"](
            intensity, kernel, method="fft", dt=kwargs["dt"],
            chunk_pixels=kwargs.get("chunk_pixels", 65536),
        )
    else:
        response = raman_namespace["raman_convolve_intensity"](
            intensity,
            method="iir",
            dt=kwargs["dt"],
            omega_R=kwargs["omega_R"],
            Gamma_R=kwargs["Gamma_R"],
            chunk_pixels=kwargs.get("chunk_pixels", 65536),
            iir_sampling=kwargs.get("iir_sampling", "exact_piecewise_linear"),
        )
    convolution_s = time.perf_counter() - convolution_started
    product, product_metrics = _call_raman_kernel(
        "product", (response, field), threads=plan.kernel_threads,
    )
    derivative_started = time.perf_counter()
    multiplier = (
        1j * xp.asarray(kwargs["Omega"], dtype=rdtype) / float(kwargs["omega0"])
    ).astype(field.dtype, copy=False)
    normalized_derivative = fft.ifft(
        multiplier[:, None, None] * fft.fft(product, axis=0), axis=0
    )
    derivative_s = time.perf_counter() - derivative_started
    prefactor = (float(kwargs["omega0"]) / float(c0)) * float(kwargs["n_R"])
    rhs, rhs_metrics = _call_raman_kernel(
        "rhs", (product, normalized_derivative, float(prefactor)),
        threads=plan.kernel_threads,
    )
    reduction_started = time.perf_counter()
    rhs_l2_norm = float(xp.linalg.norm(rhs))
    ir_max = float(xp.max(xp.abs(response)))
    energy_metrics = None
    if kwargs.get("return_energy", True):
        intensity_scale = float(xp.max(xp.abs(intensity)))
        if intensity_scale > 0.0:
            derivative_increment_norm = fft.ifft(
                (1j * xp.asarray(kwargs["Omega"], dtype=rdtype) * float(kwargs["dt"]))[:, None, None]
                * fft.fft(intensity, axis=0), axis=0,
            ).real / intensity_scale
            response_norm = response / intensity_scale
            energy_scale = (float(kwargs["n_R"]) / float(c0)) * intensity_scale * intensity_scale
            u_signed = energy_scale * xp.sum(
                response_norm * derivative_increment_norm, axis=0,
            )
        else:
            u_signed = xp.zeros(intensity.shape[1:], dtype=rdtype)
        u_signed = u_signed.astype(rdtype, copy=False)
        q_positive = xp.maximum(-u_signed, 0.0).astype(rdtype, copy=False)
        energy_metrics = {
            "intensity_scale": intensity_scale,
            "energy_scale": float((float(kwargs["n_R"]) / float(c0))
                                   * intensity_scale * intensity_scale),
        }
    reduction_s = time.perf_counter() - reduction_started
    result = {
        "rhs": rhs,
        "rhs_l2_norm": rhs_l2_norm,
        "IR_max": ir_max,
        "convolution_count": 1,
    }
    if kwargs.get("return_response", True):
        result["I_R"] = response.astype(rdtype, copy=False)
    if kwargs.get("return_energy", True):
        result["u_R_signed"] = u_signed
        result["q_R_positive"] = q_positive
    result["walltime_s"] = float(time.perf_counter() - started)
    if metrics_out is not None:
        metrics_out.update({
            "implementation": "cpu_o1_raman_pointwise_numba",
            "shape": list(field.shape),
            "dtype": str(field.dtype),
            "intensity_s": float(intensity_s),
            "iir_or_convolution_s": float(convolution_s),
            "product": product_metrics,
            "derivative_fft_s": float(derivative_s),
            "rhs": rhs_metrics,
            "reduction_s": float(reduction_s),
            "energy": energy_metrics,
            "stage_s": float(time.perf_counter() - started),
        })
    return result


def _stable_field_fluence_and_loss_cpu_o1(
    before, after, *, dt, n0, raman_namespace, plan, metrics_out=None, chunk_t=32,
):
    from KHz_filament.constants import c0, eps0

    if before.shape != after.shape:
        raise ValueError("before/after field shapes must match")
    if before.ndim != 3:
        raise ValueError("Raman field energy diagnostic expects (Nt, Ny, Nx)")
    if int(chunk_t) != 32:
        raise ValueError("CPU-O1 stable Raman energy path preserves chunk_t=32")
    xp_proxy = raman_namespace.get("xp")
    xp = getattr(xp_proxy, "xp", xp_proxy)
    before_sum = xp.zeros(before.shape[1:], dtype=xp.float64)
    loss_sum = xp.zeros(before.shape[1:], dtype=xp.float64)
    expression_metrics = []
    started = time.perf_counter()
    for start in range(0, before.shape[0], 32):
        stop = min(start + 32, before.shape[0])
        before_real = before[start:stop].real.astype(xp.float64, copy=False)
        after_real = after[start:stop].real.astype(xp.float64, copy=False)
        (before_real_square, real_loss), kernel_metrics = _call_raman_kernel(
            "stable_expression", (before_real, after_real),
            threads=plan.kernel_threads,
        )
        expression_metrics.append(kernel_metrics)
        before_sum += xp.sum(before_real_square, axis=0, dtype=xp.float64)
        loss_sum += xp.sum(real_loss, axis=0, dtype=xp.float64)
        del before_real, after_real, before_real_square, real_loss

        before_imag = before[start:stop].imag.astype(xp.float64, copy=False)
        after_imag = after[start:stop].imag.astype(xp.float64, copy=False)
        (before_imag_square, imag_loss), kernel_metrics = _call_raman_kernel(
            "stable_expression", (before_imag, after_imag),
            threads=plan.kernel_threads,
        )
        expression_metrics.append(kernel_metrics)
        before_sum += xp.sum(before_imag_square, axis=0, dtype=xp.float64)
        loss_sum += xp.sum(imag_loss, axis=0, dtype=xp.float64)
        del before_imag, after_imag, before_imag_square, imag_loss
    scale = 0.5 * float(eps0) * float(c0) * float(n0) * float(dt)
    if metrics_out is not None:
        metrics_out.update({
            "implementation": "cpu_o1_stable_energy_expression_numba",
            "chunk_t": 32,
            "expression_kernel_calls": len(expression_metrics),
            "expression_kernel_metrics": expression_metrics,
            "stable_energy_s": float(time.perf_counter() - started),
        })
    return before_sum * scale, loss_sum * scale


def apply_isaacs_raman_step_cpu_o1(
    field,
    dz,
    *,
    raman_namespace: dict,
    plan: CPUExecutionPlan,
    stage_function,
    metrics_out: dict | None = None,
    Omega,
    dt,
    omega0,
    n0,
    n_R,
    omega_R,
    Gamma_R,
    integrator="heun",
    method="iir",
    chunk_pixels=65536,
    iir_sampling="exact_piecewise_linear",
    return_diagnostics=False,
    stage1=None,
    transverse_cell_area=1.0,
):
    """Apply the supported complex128 full-operator Raman Heun candidate."""
    if str(integrator).lower() != "heun":
        raise ValueError("CPU-O1 Raman candidate currently supports Heun only")
    if field.dtype != np.dtype("complex128"):
        raise ValueError("CPU-O1 Raman candidate currently supports complex128 only")
    if metrics_out is not None and not isinstance(metrics_out, dict):
        raise TypeError("metrics_out must be a dictionary when supplied")
    kwargs = {
        "Omega": Omega, "dt": dt, "omega0": omega0, "n0": n0, "n_R": n_R,
        "omega_R": omega_R, "Gamma_R": Gamma_R, "method": method,
        "chunk_pixels": chunk_pixels, "iir_sampling": iir_sampling,
    }
    started = time.perf_counter()
    stage1_started = stage1 is None
    if stage1 is None:
        stage1 = stage_function(field, **kwargs)
    k1, q1 = stage1["rhs"], stage1["q_R_positive"]
    predictor, predictor_metrics = _call_raman_kernel(
        "predictor", (field, k1, float(dz)), threads=plan.kernel_threads,
    )
    stage2 = stage_function(predictor, **kwargs)
    result, heun_metrics = _call_raman_kernel(
        "heun", (field, k1, stage2["rhs"], float(dz)),
        threads=plan.kernel_threads,
    )
    target_stage1 = float(dz) * q1
    target_stage2 = float(dz) * stage2["q_R_positive"]
    target_heun = 0.5 * (target_stage1 + target_stage2)
    diagnostics = None
    stable_metrics = None
    if return_diagnostics:
        before_fluence, actual_local = _stable_field_fluence_and_loss_cpu_o1(
            field, result, dt=dt, n0=n0, raman_namespace=raman_namespace,
            plan=plan, metrics_out=(stable_metrics := {}), chunk_t=32,
        )
        xp_proxy = raman_namespace.get("xp")
        xp = getattr(xp_proxy, "xp", xp_proxy)
        local_residual_map = xp.abs(actual_local - target_heun) / xp.maximum(
            target_heun, xp.maximum(before_fluence * 1e-15, 1e-300)
        )
        area = float(transverse_cell_area)
        target_global = float(xp.sum(target_heun, dtype=xp.float64) * area)
        actual_global = float(xp.sum(actual_local, dtype=xp.float64) * area)
        before_global = float(xp.sum(before_fluence, dtype=xp.float64) * area)
        global_residual = abs(actual_global - target_global) / max(
            target_global, before_global * 1e-15, 1e-300
        )
        diagnostics = {
            "target_local_fluence_loss_stage1": target_stage1,
            "target_local_fluence_loss_stage2": target_stage2,
            "target_local_fluence_loss_heun": target_heun,
            "target_local_fluence_loss": target_heun,
            "target_global_energy_loss_J": target_global,
            "actual_local_fluence_loss": actual_local,
            "actual_global_energy_loss_J": actual_global,
            "local_closure_residual": float(xp.max(local_residual_map)),
            "global_closure_residual": float(global_residual),
            "rhs_l2_norm_stage1": stage1["rhs_l2_norm"],
            "rhs_l2_norm_stage2": stage2["rhs_l2_norm"],
            "IR_max_stage1": stage1["IR_max"],
            "IR_max_stage2": stage2["IR_max"],
            "I_R_stage1": stage1.get("I_R"),
            "I_R_stage2": stage2.get("I_R"),
            "convolution_count": stage1["convolution_count"] + stage2["convolution_count"],
            "operator_walltime_s": float(time.perf_counter() - started),
            "finite": bool(xp.all(xp.isfinite(result))) and bool(np.isfinite(global_residual)),
            "clipping_count": 0,
            "actual_loss_evaluation": "stable_component_difference_float64",
            "energy_projection_applied": False,
            "energy_projection_iterations": 0,
            "energy_projection_scale": 1.0,
            "energy_projection_initial_residual": 0.0,
        }
    if metrics_out is not None:
        metrics_out.update({
            "implementation": "cpu_o1_raman_heun_numba",
            "identity": "R_or_IR",
            "shape": list(field.shape),
            "dtype": str(field.dtype),
            "stage1_computed_here": bool(stage1_started),
            "predictor": predictor_metrics,
            "heun": heun_metrics,
            "stable_energy": stable_metrics,
            "complete_call_s": float(time.perf_counter() - started),
        })
    return (result, diagnostics) if return_diagnostics else result


def _restore_internal_layout(value, *, layout: str, shape: tuple[int, int, int]):
    nt, ny, nx = shape
    pixels = ny * nx
    if layout == "NtB":
        return value.reshape((nt, ny, nx))
    return value.T.reshape((nt, ny, nx))


def _restore_species_layout(value, *, layout: str, shape: tuple[int, int, int]):
    nt, ny, nx = shape
    pixels = ny * nx
    if layout == "NtB":
        return value.reshape((value.shape[0], nt, ny, nx))
    return value.transpose((0, 2, 1)).reshape((value.shape[0], nt, ny, nx))


def _merge_stability(species_entries, arrays):
    step_min, step_max, inter_min, inter_max, clips, violations = arrays
    records = {}
    for species, entry in enumerate(species_entries):
        name = str(entry.get("name", f"species[{species}]"))
        record = records.setdefault(name, {
            "preclip_step_min": float("inf"),
            "preclip_step_max": float("-inf"),
            "preclip_intermediate_min": float("inf"),
            "preclip_intermediate_max": float("-inf"),
            "step_clip_count": 0,
            "intermediate_violation_count": 0,
        })
        for block in range(step_min.shape[1]):
            record["preclip_step_min"] = min(record["preclip_step_min"], float(step_min[species, block]))
            record["preclip_step_max"] = max(record["preclip_step_max"], float(step_max[species, block]))
            record["preclip_intermediate_min"] = min(
                record["preclip_intermediate_min"], float(inter_min[species, block])
            )
            record["preclip_intermediate_max"] = max(
                record["preclip_intermediate_max"], float(inter_max[species, block])
            )
            record["step_clip_count"] += int(clips[species, block])
            record["intermediate_violation_count"] += int(violations[species, block])
    return records


def evolve_rho_time_cpu_o1(
    input_array,
    dt: float,
    N0: float,
    beta_rec: float,
    Wfunc,
    *,
    plan: CPUExecutionPlan | None = None,
    quasi_static_time: bool = False,
    time_stat: str = "peak",
    mean_clip_frac: float = 1e-3,
    expects: str | None = None,
    return_species_terms: bool = False,
    return_species_densities: bool = False,
    diagnose_integrator_stability: bool = False,
    metrics_out: dict | None = None,
    rate_workspace: _IonizationRateScratch | None = None,
):
    """Run the frozen reference or an explicit local compiled RK4 candidate.

    The return tuple and species-term keys match ``runtime.evolve_rho_time``.
    ``metrics_out`` is diagnostic-only and does not change that return contract.
    """
    from KHz_filament.ionization import runtime

    selected = plan or CPUExecutionPlan()
    if metrics_out is not None and not isinstance(metrics_out, dict):
        raise TypeError("metrics_out must be a dictionary when supplied")
    if selected.workspace_owner == "per_factory" and rate_workspace is None:
        raise ValueError("per_factory workspace ownership requires a private rate workspace")
    if rate_workspace is not None and selected.workspace_owner != "per_factory":
        raise ValueError("a private rate workspace requires workspace_owner='per_factory'")
    start_total = time.perf_counter()
    if selected.implementation == "reference":
        started = time.perf_counter()
        result = runtime.evolve_rho_time(
            input_array, dt, N0, beta_rec, Wfunc,
            quasi_static_time=quasi_static_time, time_stat=time_stat,
            mean_clip_frac=mean_clip_frac, expects=expects,
            return_species_terms=return_species_terms,
            return_species_densities=return_species_densities,
            diagnose_integrator_stability=diagnose_integrator_stability,
        )
        if metrics_out is not None:
            metrics_out.update({"implementation": "reference", "runtime_s": time.perf_counter() - started})
        return result

    scalar_inputs = np.asarray([dt, N0, beta_rec], dtype=np.float64)
    if not np.all(np.isfinite(scalar_inputs)):
        raise ValueError("CPU-O1 requires finite dt, N0, and beta_rec")

    mode = str(getattr(Wfunc, "_time_mode", "")).lower()
    integrator = str(getattr(Wfunc, "_integrator", "rk4")).lower()
    entries = getattr(Wfunc, "_species_entries", None)
    if mode in {"qs_peak", "qs_mean", "qs_mean_esq"}:
        quasi_static_time = True
    elif mode == "full":
        quasi_static_time = False
    if quasi_static_time:
        raise ValueError("CPU-O1 compiled prototype only supports full-time ionization")
    if integrator != "rk4":
        raise ValueError("CPU-O1 compiled prototype only supports the frozen RK4 integrator")
    if not entries:
        raise ValueError("CPU-O1 compiled prototype requires full species Wfunc entries")
    if mode not in {"", "full"}:
        raise ValueError(f"CPU-O1 compiled prototype does not support time mode {mode!r}")
    if runtime.xp.__name__ != "numpy":
        raise ValueError("CPU-O1 compiled prototype requires the NumPy backend")
    inp = runtime._as_real_like(input_array, like=input_array)
    if not isinstance(inp, np.ndarray):
        raise ValueError("CPU-O1 compiled prototype accepts NumPy arrays only")
    if inp.ndim != 3:
        raise ValueError("CPU-O1 input must have shape [Nt, Ny, Nx]")
    nt, ny, nx = (int(v) for v in inp.shape)
    if nt < 1 or ny < 1 or nx < 1:
        raise ValueError("CPU-O1 input dimensions must be positive")
    if not np.all(np.isfinite(inp)):
        raise ValueError("CPU-O1 input must be finite")
    rdtype = np.float32 if inp.dtype == np.complex64 else np.float64
    if inp.dtype not in {np.dtype("float32"), np.dtype("float64")}:
        inp = inp.astype(rdtype, copy=False)

    fractions = np.asarray([max(0.0, float(entry["fraction"])) for entry in entries], dtype=rdtype)
    fraction_sum = float(fractions.sum())
    if np.isfinite(fraction_sum) and fraction_sum > 0.0:
        fractions = fractions / fraction_sum
    else:
        fractions = np.ones_like(fractions) / float(len(entries))
    n0_species = np.asarray([float(N0) * float(fractions[j]) for j in range(len(entries))], dtype=rdtype)
    beta_species = np.asarray([float(beta_rec) * float(n0_species[j]) for j in range(len(entries))], dtype=rdtype)
    if not np.all(np.isfinite(fractions)):
        raise ValueError("CPU-O1 normalized species fractions must remain finite")
    if not np.all(np.isfinite(n0_species)) or not np.all(np.isfinite(beta_species)):
        raise ValueError("CPU-O1 derived species constants must remain finite")
    if return_species_terms:
        ui_species = np.asarray(
            [runtime._species_ui_joule(entry, j) for j, entry in enumerate(entries)],
            dtype=rdtype,
        )
    else:
        ui_species = np.zeros(len(entries), dtype=rdtype)
    if not np.all(np.isfinite(ui_species)):
        raise ValueError("CPU-O1 species ionization energies must remain finite")
    rate_started = time.perf_counter()
    pixels = ny * nx
    rate_shape = (
        (len(entries), nt, pixels)
        if selected.layout == "NtB"
        else (len(entries), pixels, nt)
    )
    if rate_workspace is None:
        rates = np.empty(rate_shape, dtype=rdtype)
        rate_workspace_reused = False
    else:
        rates, rate_workspace_reused = rate_workspace.acquire(
            rate_shape, rdtype, layout=selected.layout,
        )
    rate_eval_s = 0.0
    rate_pack_started = time.perf_counter()
    for species_index, entry in enumerate(entries):
        evaluator_started = time.perf_counter()
        evaluator = entry.get("W_runtime", entry["W_s"])
        rate = evaluator(inp).astype(rdtype, copy=False)
        if not isinstance(rate, np.ndarray) or rate.shape != inp.shape:
            raise ValueError("W_runtime must return a NumPy array matching [Nt, Ny, Nx]")
        if rate.dtype != np.dtype(rdtype):
            rate = rate.astype(rdtype, copy=False)
        if not np.all(np.isfinite(rate)):
            raise ValueError("CPU-O1 compiled prototype requires finite W_runtime output")
        rate_eval_s += time.perf_counter() - evaluator_started
        if selected.layout == "NtB":
            np.copyto(rates[species_index], rate.reshape((nt, pixels)), casting="no")
        else:
            np.copyto(rates[species_index], rate.reshape((nt, pixels)).T, casting="no")
        del rate
    rate_pack_s = time.perf_counter() - rate_pack_started - rate_eval_s
    layout_s = 0.0
    if selected.layout == "NtB":
        layout_flag = 0
    else:
        layout_flag = 1

    dependency_load_started = time.perf_counter()
    numba, serial_kernel, parallel_kernel = _load_numba_kernels()
    dependency_load_s = time.perf_counter() - dependency_load_started
    kernel = serial_kernel if selected.implementation == "numba_serial" else parallel_kernel
    if selected.implementation == "numba_parallel":
        if selected.kernel_threads > int(numba.config.NUMBA_NUM_THREADS):
            raise ValueError(
                f"requested {selected.kernel_threads} threads exceeds Numba limit "
                f"{numba.config.NUMBA_NUM_THREADS}"
            )
        old_threads = int(numba.get_num_threads())
        numba.set_num_threads(selected.kernel_threads)
    else:
        old_threads = None
    scalar_type = rates.dtype.type
    call_args = (
        rates, fractions, n0_species, beta_species, ui_species,
        selected.block_pixels, layout_flag, bool(diagnose_integrator_stability),
        scalar_type(0.0), scalar_type(1.0), scalar_type(0.5),
        scalar_type(0.5 * float(dt)), scalar_type(float(dt) / 6.0),
        scalar_type(2.0), scalar_type(float(dt)), scalar_type(beta_rec),
    )
    signature = tuple(numba.typeof(argument) for argument in call_args)
    compiled_before = signature in kernel.signatures
    compilation_started = time.perf_counter()
    try:
        if not compiled_before:
            kernel.compile(signature)
    finally:
        if old_threads is not None:
            numba.set_num_threads(old_threads)
    compilation_or_cache_load_s = time.perf_counter() - compilation_started
    if selected.implementation == "numba_parallel":
        old_threads = int(numba.get_num_threads())
        numba.set_num_threads(selected.kernel_threads)
    kernel_started = time.perf_counter()
    try:
        outputs = kernel(*call_args)
    finally:
        if selected.implementation == "numba_parallel":
            numba.set_num_threads(old_threads)
    kernel_s = time.perf_counter() - kernel_started
    (species_values, rho_sum_internal, wt_total_internal, net_internal,
     photo_internal, step_min, step_max, inter_min, inter_max,
     clips, violations) = outputs
    convert_started = time.perf_counter()
    rho_sum = _restore_internal_layout(rho_sum_internal, layout=selected.layout, shape=inp.shape).astype(rdtype, copy=False)
    wt_total = _restore_internal_layout(wt_total_internal, layout=selected.layout, shape=inp.shape).astype(rdtype, copy=False)
    net_terms = _restore_internal_layout(net_internal, layout=selected.layout, shape=inp.shape).astype(rdtype, copy=False)
    photo_terms = _restore_internal_layout(photo_internal, layout=selected.layout, shape=inp.shape).astype(rdtype, copy=False)
    species_density = _restore_species_layout(species_values, layout=selected.layout, shape=inp.shape).astype(rdtype, copy=False)
    species_density_views = [species_density[j] for j in range(len(entries))]
    convert_s = time.perf_counter() - convert_started
    if len(entries) == 1:
        # The frozen reference returns the sole species array itself as rho_sum.
        rho_sum = species_density_views[0]
    if not all(np.all(np.isfinite(array)) for array in (
        rho_sum, wt_total, net_terms, photo_terms, species_density,
    )):
        raise ValueError("CPU-O1 generated a non-finite output array")
    if nt > 1 and not all(np.all(np.isfinite(array)) for array in (
        step_min, step_max, inter_min, inter_max,
    )):
        raise ValueError("CPU-O1 generated non-finite RK4 stability observations")

    stability = None
    if diagnose_integrator_stability:
        stability = {
            "integrator": integrator,
            "species": _merge_stability(
                entries, (step_min, step_max, inter_min, inter_max, clips, violations)
            ),
        }
    if return_species_terms:
        terms = {
            "drho_dt_u_sum": net_terms,
            "photoionization_energy_rate": photo_terms,
        }
        if return_species_densities:
            species_by_name = {
                str(entry.get("name", f"species[{j}]")): species_density_views[j]
                for j, entry in enumerate(entries)
            }
            terms["rho_by_species"] = species_by_name
        result = (rho_sum, wt_total, terms, stability) if stability is not None else (rho_sum, wt_total, terms)
    else:
        result = (rho_sum, wt_total, stability) if stability is not None else (rho_sum, wt_total)

    if metrics_out is not None:
        metrics_out.update({
            "implementation": selected.implementation,
            "settings": selected.as_dict(),
            "rate_evaluation_s": float(rate_eval_s),
            "rate_packing_s": float(rate_pack_s),
            "rate_workspace_reused": bool(rate_workspace_reused),
            "rate_workspace_allocations": (
                int(rate_workspace.allocations) if rate_workspace is not None else 0
            ),
            "rate_workspace_reuses": (
                int(rate_workspace.reuses) if rate_workspace is not None else 0
            ),
            "layout_preparation_s": float(layout_s),
            "numba_dependency_load_s": float(dependency_load_s),
            "kernel_cold_start": not compiled_before,
            "kernel_compilation_or_cache_load_s": float(compilation_or_cache_load_s),
            "kernel_s": float(kernel_s),
            "output_restore_s": float(convert_s),
            "runtime_s": float(time.perf_counter() - start_total),
            "numba_threading_layer": (
                str(numba.threading_layer()) if selected.implementation == "numba_parallel" else None
            ),
            "numba_threads": (
                int(selected.kernel_threads) if selected.implementation == "numba_parallel" else 1
            ),
        })
    return result


@dataclass(frozen=True)
class CPUExecutor:
    """Thin plan-bound adapter for APIs that expect ``cpu_executor`` methods."""

    plan: CPUExecutionPlan = CPUExecutionPlan()

    def evolve_rho_time(self, *args, **kwargs):
        if "plan" in kwargs:
            raise TypeError("CPUExecutor binds its plan; omit the plan keyword")
        return evolve_rho_time_cpu_o1(*args, plan=self.plan, **kwargs)
