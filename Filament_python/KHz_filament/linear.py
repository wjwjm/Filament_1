from __future__ import annotations

import time

from .device import xp


class _BkNeeProfiler:
    """Opt-in, synchronised CuPy stage timer; no calls occur when disabled."""
    def __init__(self, enabled):
        self.enabled = bool(enabled)
        self.stages = {}
        self.peak_allocated_bytes = 0
        self.peak_reserved_bytes = 0
        self.sync_walltime_s = 0.0

    def _memory(self):
        if not self.enabled or getattr(xp, "__name__", "numpy") != "cupy":
            return 0, 0
        pool = xp.get_default_memory_pool()
        allocated, reserved = int(pool.used_bytes()), int(pool.total_bytes())
        self.peak_allocated_bytes = max(self.peak_allocated_bytes, allocated)
        self.peak_reserved_bytes = max(self.peak_reserved_bytes, reserved)
        return allocated, reserved

    def _sync(self):
        if getattr(xp, "__name__", "numpy") == "cupy":
            t0 = time.perf_counter(); xp.cuda.Stream.null.synchronize()
            self.sync_walltime_s += time.perf_counter() - t0

    def run(self, name, operation):
        if not self.enabled:
            return operation()
        self._sync(); before = self._memory(); t0 = time.perf_counter()
        result = operation()
        self._sync(); elapsed = time.perf_counter() - t0; after = self._memory()
        item = self.stages.setdefault(name, {"walltime_s": 0.0, "calls": 0, "allocated_before_bytes": before[0], "reserved_before_bytes": before[1], "allocated_after_bytes": after[0], "reserved_after_bytes": after[1]})
        item["walltime_s"] += elapsed; item["calls"] += 1
        item["allocated_after_bytes"], item["reserved_after_bytes"] = after
        return result

    def report(self):
        return {"stages": self.stages, "explicit_synchronization_walltime_s": self.sync_walltime_s,
                "peak_allocated_gpu_memory_bytes": self.peak_allocated_bytes,
                "peak_reserved_gpu_memory_bytes": self.peak_reserved_bytes,
                "temporary_array_count_where_measurable": None}


def _complex_real_dtypes(ctype):
    """Return matching real dtype for a complex dtype."""
    rdtype = xp.float32 if ctype == xp.complex64 else xp.float64
    return ctype, rdtype


def lin_propagator(kperp2, k0, dz, *, ctype=None):
    """Paraxial angular-spectrum propagator exp(i * (-k_perp^2) dz / (2k0))."""
    if ctype is None:
        ctype = xp.complex64
    ctype, rdtype = _complex_real_dtypes(ctype)

    onej = xp.array(1j, dtype=ctype)
    k2 = xp.asarray(kperp2, dtype=rdtype)
    phase = (-k2) * (dz / (2.0 * float(k0)))
    return xp.exp(onej * phase).astype(ctype)


def step_linear(E, prop):
    """Apply a 2D (x,y) FFT-based linear propagation to [Nt, Ny, Nx]."""
    if prop.dtype != E.dtype:
        prop = prop.astype(E.dtype, copy=False)
    Ew = xp.fft.fft2(E, axes=(-2, -1))
    Ew *= prop
    return xp.fft.ifft2(Ew, axes=(-2, -1))


def step_linear_bk_nee_factorized(
    E,
    *,
    Omega,
    kperp2,
    k0,
    omega0,
    dz,
    beta2=0.0,
    denom_floor=1e-4,
    precision_strategy="baseline_complex64",
    return_energy_diagnostics=False,
    energy_scale=None,
    return_profile_diagnostics=False,
):
    """Brabec–Krausz NEE linear step (factorized over frequency slices).

    Uses the linear operator in frequency domain:
      dA/dz = i [ -k_perp^2/(2 k0 (1+Omega/omega0)) + (beta2/2) Omega^2 ] A
    and applies exp(i * phase * dz) per Omega slice.
    """
    allowed_strategies = ("baseline_complex64", "orthonormal_fft", "mixed_precision", "unitary_projection")
    strategy = str(precision_strategy or "baseline_complex64").lower()
    if strategy not in allowed_strategies:
        raise ValueError(f"unknown BK-NEE precision strategy {strategy!r}; allowed: {allowed_strategies}")

    profiler = _BkNeeProfiler(return_profile_diagnostics)
    output_ctype, output_rdtype = _complex_real_dtypes(E.dtype)
    work_ctype = xp.complex128 if strategy == "mixed_precision" else output_ctype
    _, work_rdtype = _complex_real_dtypes(work_ctype)
    onej = xp.array(1j, dtype=work_ctype)

    Omega, kperp2 = profiler.run("allocation_workspace_preparation", lambda: (xp.asarray(Omega, dtype=work_rdtype), xp.asarray(kperp2, dtype=work_rdtype)))

    # FFT_t first, then per-slice FFT2_xy to keep memory usage lower than full 3D operator.
    if return_energy_diagnostics and energy_scale is None:
        raise ValueError("BK-NEE energy diagnostics require energy_scale")

    def _norm2(value):
        return xp.sum(xp.abs(value) ** 2, dtype=xp.float64)

    needs_projection_norm = strategy == "unitary_projection"
    input_norm2 = _norm2(E) if (return_energy_diagnostics or needs_projection_norm) else None
    work_input = profiler.run("cast_input_to_complex128", lambda: E.astype(work_ctype, copy=False))
    input_cast_norm2 = _norm2(work_input) if return_energy_diagnostics else None
    fft_kwargs = {"norm": "ortho"} if strategy == "orthonormal_fft" else {}
    Ew = profiler.run("temporal_fft", lambda: xp.fft.fft(work_input, axis=0, **fft_kwargs))  # [Nt, Ny, Nx]
    # ``Ew`` owns the complete transformed field.  Keeping the complex128
    # cast alive until the inverse temporal FFT needlessly costs one full
    # volume (1.61 GB for the Phase 8C production grid) at the peak.
    del work_input
    forward_norm2 = _norm2(Ew) if return_energy_diagnostics else None

    rel = Omega / float(omega0)
    denom = 1.0 + rel
    # Avoid singularity near Omega ~= -omega0.
    denom_abs = xp.maximum(xp.abs(denom), float(denom_floor))
    denom_sign = xp.where(denom >= 0.0, 1.0, -1.0)
    denom = denom_sign * denom_abs

    coeff_diff = -1.0 / (2.0 * float(k0) * denom)          # [Nt]
    coeff_gvd = 0.5 * float(beta2) * (Omega ** 2)          # [Nt]

    Nt = Ew.shape[0]
    nxy = int(Ew.shape[-2] * Ew.shape[-1])
    forward_factor = 1 if strategy == "orthonormal_fft" else Nt
    transfer_factor = 1 if strategy == "orthonormal_fft" else Nt * nxy
    transfer_norm2 = xp.asarray(0.0, dtype=xp.float64) if return_energy_diagnostics else None
    for i in range(Nt):
        phase_xy = coeff_diff[i] * kperp2 + coeff_gvd[i]
        prop2d = profiler.run("transfer_kernel_preparation", lambda: xp.exp(onej * phase_xy * float(dz)).astype(work_ctype, copy=False))
        S = profiler.run("spatial_fft2", lambda: xp.fft.fft2(Ew[i], axes=(-2, -1), **fft_kwargs))
        profiler.run("transfer_multiply", lambda: S.__imul__(prop2d))
        if return_energy_diagnostics:
            transfer_norm2 += _norm2(S)
        Ew[i] = profiler.run("inverse_spatial_fft2", lambda: xp.fft.ifft2(S, axes=(-2, -1), **fft_kwargs))

    inverse_spatial_norm2 = _norm2(Ew) if return_energy_diagnostics else None
    # The last slice's work buffers are no longer required once copied back
    # into ``Ew``.  Release their Python references before the full-volume
    # complex128 inverse temporal transform is allocated.
    del S, prop2d
    internal_out = profiler.run("inverse_temporal_fft", lambda: xp.fft.ifft(Ew, axis=0, **fft_kwargs))
    internal_norm2 = _norm2(internal_out) if return_energy_diagnostics else None
    candidate_out = profiler.run("cast_output_to_complex64", lambda: internal_out.astype(output_ctype, copy=False))
    output_cast_norm2 = _norm2(candidate_out) if (return_energy_diagnostics or needs_projection_norm) else None
    projection_scale = 1.0
    if strategy == "unitary_projection":
        candidate_energy = float(output_cast_norm2)
        if not candidate_energy > 0.0:
            raise FloatingPointError("cannot apply BK-NEE unitary projection to a non-positive field norm")
        projection_scale = (float(input_norm2) / candidate_energy) ** 0.5
        out = (candidate_out * xp.asarray(projection_scale, dtype=output_rdtype)).astype(output_ctype, copy=False)
    else:
        out = candidate_out
    profile = profiler.report() if return_profile_diagnostics else None
    if not return_energy_diagnostics:
        return (out, profile) if return_profile_diagnostics else out
    inverse_time_norm2 = _norm2(out)
    scale = float(energy_scale)
    energy = {
        "energy_before_J": float(scale * input_norm2),
        "energy_after_input_cast_J": float(scale * input_cast_norm2),
        "energy_after_forward_fft_J": float(scale * forward_norm2 / forward_factor),
        "energy_after_transfer_J": float(scale * transfer_norm2 / transfer_factor),
        "energy_after_inverse_fft_J": float(scale * inverse_spatial_norm2 / forward_factor),
        "energy_after_internal_linear_J": float(scale * internal_norm2),
        "energy_after_output_cast_J": float(scale * output_cast_norm2),
        "energy_after_J": float(scale * inverse_time_norm2),
        "output_cast_field_delta_J": float(scale * (output_cast_norm2 - internal_norm2)),
        "unitary_projection_scale": float(projection_scale),
        "unitary_projection_scale_deviation": float(abs(projection_scale - 1.0)),
        "explicit_boundary_loss_J": 0.0,
        "explicit_spectral_filter_loss_J": 0.0,
        "explicit_crop_loss_J": 0.0,
        "explicit_evanescent_loss_J": 0.0,
        "explicit_other_loss_J": 0.0,
    }
    return (out, energy, profile) if return_profile_diagnostics else (out, energy)
