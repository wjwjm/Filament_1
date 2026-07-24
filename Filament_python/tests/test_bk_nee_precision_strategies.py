from __future__ import annotations

import numpy as np
import pytest


def _problem():
    rng = np.random.default_rng(51)
    field = (rng.normal(size=(24, 10, 10)) + 1j * rng.normal(size=(24, 10, 10))).astype(np.complex64)
    omega = 2.0 * np.pi * np.fft.fftfreq(24, 2.5e-15)
    k = 2.0 * np.pi * np.fft.fftfreq(10, 12e-6)
    return field, omega, k[:, None] ** 2 + k[None, :] ** 2


def _step(field, strategy, *, diagnostics=False):
    from KHz_filament.linear import step_linear_bk_nee_factorized

    omega0 = 2.0 * np.pi * 3e8 / 800e-9
    field0, omega, kperp2 = _problem()
    return step_linear_bk_nee_factorized(
        field, Omega=omega, kperp2=kperp2, k0=7e6, omega0=omega0,
        dz=5e-5, precision_strategy=strategy,
        return_energy_diagnostics=diagnostics, energy_scale=1.0 if diagnostics else None,
    )


def test_precision_strategy_normalization_is_strict():
    from KHz_filament.config_normalize import normalize_config

    assert normalize_config({"propagation": {"linear_precision_strategy": "mixed_precision"}})["propagation"]["linear_precision_strategy"] == "mixed_precision"
    with pytest.raises(ValueError, match="linear_precision_strategy"):
        normalize_config({"propagation": {"linear_precision_strategy": "surprise"}})


def test_candidates_preserve_shape_dtype_and_emit_float64_energy_audit():
    field, _, _ = _problem()
    for strategy in ("baseline_complex64", "orthonormal_fft", "mixed_precision", "unitary_projection"):
        out, audit = _step(field, strategy, diagnostics=True)
        assert out.dtype == np.complex64
        assert out.shape == field.shape
        assert all(np.isfinite(value) for value in audit.values())
        assert audit["explicit_boundary_loss_J"] == 0.0
        assert audit["explicit_spectral_filter_loss_J"] == 0.0
        assert audit["explicit_crop_loss_J"] == 0.0
        assert audit["explicit_evanescent_loss_J"] == 0.0


def test_mixed_precision_reduces_reference_error_without_changing_storage_dtype():
    field, _, _ = _problem()
    reference = _step(field.astype(np.complex128), "baseline_complex64")
    baseline = _step(field, "baseline_complex64")
    mixed = _step(field, "mixed_precision")
    denom = np.linalg.norm(reference.ravel())
    baseline_error = np.linalg.norm((baseline - reference).ravel()) / denom
    mixed_error = np.linalg.norm((mixed - reference).ravel()) / denom
    assert mixed_error < baseline_error
    assert mixed.dtype == np.complex64


def test_unitary_projection_is_opt_in_and_does_not_worsen_reference_error():
    field, _, _ = _problem()
    reference = _step(field.astype(np.complex128), "baseline_complex64")
    baseline, baseline_audit = _step(field, "baseline_complex64", diagnostics=True)
    projected, projected_audit = _step(field, "unitary_projection", diagnostics=True)
    denom = np.linalg.norm(reference.ravel())
    baseline_error = np.linalg.norm((baseline - reference).ravel()) / denom
    projected_error = np.linalg.norm((projected - reference).ravel()) / denom
    assert projected_audit["unitary_projection_scale_deviation"] >= 0.0
    assert projected_error <= baseline_error * (1.0 + 1e-6)
    assert abs(projected_audit["energy_after_J"] - projected_audit["energy_before_J"]) <= abs(baseline_audit["energy_after_J"] - baseline_audit["energy_before_J"])


def test_bk_nee_profile_does_not_change_field():
    from KHz_filament.linear import step_linear_bk_nee_factorized
    field, omega, kperp2 = _problem()
    plain = step_linear_bk_nee_factorized(field, Omega=omega, kperp2=kperp2, k0=7e6,
                                           omega0=2.35e15, dz=5e-5,
                                           precision_strategy="mixed_precision")
    profiled, _ = step_linear_bk_nee_factorized(field, Omega=omega, kperp2=kperp2, k0=7e6,
                                                 omega0=2.35e15, dz=5e-5,
                                                 precision_strategy="mixed_precision",
                                                 return_profile_diagnostics=True)
    assert np.array_equal(plain, profiled)


def test_bk_nee_profile_reports_all_stages():
    from KHz_filament.linear import step_linear_bk_nee_factorized
    field, omega, kperp2 = _problem()
    _, profile = step_linear_bk_nee_factorized(field, Omega=omega, kperp2=kperp2, k0=7e6,
                                                omega0=2.35e15, dz=5e-5,
                                                precision_strategy="mixed_precision",
                                                return_profile_diagnostics=True)
    required = {"allocation_workspace_preparation", "cast_input_to_complex128", "temporal_fft",
                "spatial_fft2", "transfer_kernel_preparation", "transfer_multiply",
                "inverse_spatial_fft2", "inverse_temporal_fft", "cast_output_to_complex64"}
    assert required <= set(profile["stages"])
    assert all(profile["stages"][name]["calls"] > 0 for name in required)


def test_bk_nee_profile_records_reserved_memory():
    from KHz_filament.linear import step_linear_bk_nee_factorized
    field, omega, kperp2 = _problem()
    _, profile = step_linear_bk_nee_factorized(field, Omega=omega, kperp2=kperp2, k0=7e6,
                                                omega0=2.35e15, dz=5e-5,
                                                precision_strategy="mixed_precision",
                                                return_profile_diagnostics=True)
    assert profile["peak_allocated_gpu_memory_bytes"] >= 0
    assert profile["peak_reserved_gpu_memory_bytes"] >= 0
