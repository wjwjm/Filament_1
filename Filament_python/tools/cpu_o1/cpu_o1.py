"""Bounded CPU-O1 local prototypes around frozen Filament operators.

This module never patches production globals and is not a production backend.
The compiled path accepts only the full-time, full-species RK4 contract; LUT
evaluation and all physical configuration handling remain in the frozen
runtime adapter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import time
from typing import Literal

import numpy as np


Implementation = Literal["reference", "numba_serial", "numba_parallel"]
Layout = Literal["NtB", "BNt"]
_NUMBA_KERNELS = None
_NUMBA_PHOTO_CLIP = None


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
    workspace_owner: Literal["per_call"] = "per_call"
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
        if self.workspace_owner != "per_call":
            raise ValueError("the local prototype only supports per-call workspaces")
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
        "reference_dtype": str(ref.dtype),
        "candidate_dtype": str(got.dtype),
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
                block_pixels, layout_flag, zero, one, half_value,
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
        step_min = np.empty((species_count, block_count), dtype=np.float64)
        step_max = np.empty((species_count, block_count), dtype=np.float64)
        intermediate_min = np.empty((species_count, block_count), dtype=np.float64)
        intermediate_max = np.empty((species_count, block_count), dtype=np.float64)
        step_clips = np.zeros((species_count, block_count), dtype=np.int64)
        intermediate_violations = np.zeros((species_count, block_count), dtype=np.int64)
        for block_id in prange(block_count):
            start = block_id * block_pixels
            stop = min(start + block_pixels, pixels)
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
):
    """Run the frozen reference or an explicit local compiled RK4 candidate.

    The return tuple and species-term keys match ``runtime.evolve_rho_time``.
    ``metrics_out`` is diagnostic-only and does not change that return contract.
    """
    from KHz_filament.ionization import runtime

    selected = plan or CPUExecutionPlan()
    if metrics_out is not None and not isinstance(metrics_out, dict):
        raise TypeError("metrics_out must be a dictionary when supplied")
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
    rate_fields = []
    for entry in entries:
        evaluator = entry.get("W_runtime", entry["W_s"])
        rate = evaluator(inp).astype(rdtype, copy=False)
        if not isinstance(rate, np.ndarray) or rate.shape != inp.shape:
            raise ValueError("W_runtime must return a NumPy array matching [Nt, Ny, Nx]")
        if rate.dtype != np.dtype(rdtype):
            rate = rate.astype(rdtype, copy=False)
        if not np.all(np.isfinite(rate)):
            raise ValueError("CPU-O1 compiled prototype requires finite W_runtime output")
        rate_fields.append(rate)
    rate_eval_s = time.perf_counter() - rate_started
    rates_stack = np.stack(rate_fields, axis=0).astype(rdtype, copy=False)
    pixels = ny * nx
    if selected.layout == "NtB":
        rates = np.ascontiguousarray(rates_stack.reshape((len(entries), nt, pixels)))
        layout_flag = 0
    else:
        rates = np.ascontiguousarray(rates_stack.reshape((len(entries), nt, pixels)).transpose((0, 2, 1)))
        layout_flag = 1
    layout_s = time.perf_counter() - rate_started - rate_eval_s

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
        selected.block_pixels, layout_flag,
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
