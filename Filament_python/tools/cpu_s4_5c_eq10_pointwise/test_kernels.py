"""Focused C1 tests for the two Eq10 pointwise materializations."""

import json
import os

import numpy as np
import pytest

from .kernels import KernelExecutor, KernelUnsupported, source_identity


def _ordered_bytes(value: np.ndarray) -> bytes:
    """Compare logical element encodings without changing native layout."""
    return value.tobytes(order="C")


def _first_mismatch(actual: np.ndarray, expected: np.ndarray):
    for index in np.ndindex(actual.shape):
        actual_bytes = np.asarray(actual[index], dtype=actual.dtype).tobytes()
        expected_bytes = np.asarray(expected[index], dtype=expected.dtype).tobytes()
        if actual_bytes != expected_bytes:
            return index, actual_bytes.hex(), expected_bytes.hex()
    return None


def _require_exact(actual, expected, *, kernel, tmp_path):
    if _ordered_bytes(actual) == _ordered_bytes(expected):
        return
    first = _first_mismatch(actual, expected)
    payload = {
        "status": "FIRST_NONEXACT_STOP",
        "kernel": kernel,
        "first_index": list(first[0]) if first else None,
        "actual_bits_hex": first[1] if first else None,
        "expected_bits_hex": first[2] if first else None,
        "actual_layout": {
            "shape": list(actual.shape), "dtype": str(actual.dtype),
            "strides": list(actual.strides), "C": bool(actual.flags.c_contiguous),
            "F": bool(actual.flags.f_contiguous), "OWNDATA": bool(actual.flags.owndata),
        },
        "expected_layout": {
            "shape": list(expected.shape), "dtype": str(expected.dtype),
            "strides": list(expected.strides), "C": bool(expected.flags.c_contiguous),
            "F": bool(expected.flags.f_contiguous), "OWNDATA": bool(expected.flags.owndata),
        },
        "kernel_source_identity": source_identity(),
    }
    failure = tmp_path / "S45C_FIRST_NONEXACT.json"
    failure.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    pytest.fail(f"{kernel} first nonexact recorded at {failure}: {payload}")


def _complex_fixture(shape=(384, 5, 37)):
    rng = np.random.default_rng(20261028)
    spectrum = (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ).astype(np.complex128)
    coeff = np.linspace(-0.75, 0.9, shape[0], dtype=np.float64).astype(np.complex128)
    coeff = (1j * coeff).reshape(shape[0], 1, 1)
    spectrum[0, 0, 0] = complex(-0.0, 0.0)
    spectrum[1, 0, 1] = complex(np.nextafter(0.0, 1.0), -np.nextafter(0.0, 1.0))
    coeff[0, 0, 0] = complex(-0.0, 1.0)
    return coeff, spectrum


def test_k1_matches_source_operand_bitwise_and_preserves_inputs(tmp_path):
    coeff, spectrum = _complex_fixture()
    coeff_before = coeff.copy()
    spectrum_before = spectrum.copy()
    expected = coeff * spectrum
    result = KernelExecutor(workers=1).k1(coeff, spectrum)
    _require_exact(result, expected, kernel="K1", tmp_path=tmp_path)
    assert _ordered_bytes(coeff) == _ordered_bytes(coeff_before)
    assert _ordered_bytes(spectrum) == _ordered_bytes(spectrum_before)
    assert result.flags.owndata
    assert not np.shares_memory(result, spectrum)
    assert result.shape == spectrum.shape
    assert result.dtype == np.dtype("complex128")


def test_k2_matches_real_view_bitwise_without_contiguous_copy(tmp_path):
    rng = np.random.default_rng(20261029)
    backing = rng.standard_normal((384, 5, 74)).astype(np.float64)
    real_view = backing[:, :, ::2]
    assert not real_view.flags.c_contiguous
    before = real_view.copy()
    scale = np.float64(1.125)
    expected = real_view / scale
    executor = KernelExecutor(workers=1)
    result = executor.k2(real_view, scale)
    _require_exact(result, expected, kernel="K2", tmp_path=tmp_path)
    assert _ordered_bytes(real_view) == _ordered_bytes(before)
    assert result.flags.owndata
    assert not np.shares_memory(result, real_view)


def test_f_order_and_tail_block_are_supported(tmp_path):
    coeff, spectrum = _complex_fixture((7, 3, 35))
    spectrum = np.asfortranarray(spectrum)
    expected = coeff * spectrum
    result = KernelExecutor(workers=1).k1(coeff, spectrum)
    _require_exact(result, expected, kernel="K1_F_ORDER", tmp_path=tmp_path)
    assert result.flags.f_contiguous


def test_real_numpy_fft_operands_and_nonfinite_classification(tmp_path):
    rng = np.random.default_rng(20261030)
    intensity = rng.standard_normal((384, 3, 35)).astype(np.float64)
    omega = (np.arange(384, dtype=np.float64) - 192.0) * 1.0e12
    coeff = (1j * np.asarray(omega, dtype=np.float64) * float(2.5e-16))[:, None, None]
    spectrum = np.fft.fft(intensity.astype(np.complex128), axis=0)
    expected_product = coeff * spectrum
    executor = KernelExecutor(workers=1)
    product = executor.k1(coeff, spectrum)
    _require_exact(product, expected_product, kernel="K1_REAL_FFT", tmp_path=tmp_path)
    transformed = np.fft.ifft(product, axis=0)
    real_view = transformed.real
    expected_real = real_view / np.float64(1.25)
    normalized = executor.k2(real_view, np.float64(1.25))
    _require_exact(normalized, expected_real, kernel="K2_REAL_IFFT", tmp_path=tmp_path)
    nonfinite = np.zeros((4, 2, 3), dtype=np.complex128)
    nonfinite[0, 0, 0] = complex(np.inf, -np.inf)
    nonfinite[1, 0, 1] = complex(np.nan, 1.0)
    nonfinite_expected = coeff[:4] * nonfinite
    nonfinite_actual = executor.k1(coeff[:4], nonfinite)
    assert np.array_equal(np.isfinite(nonfinite_actual), np.isfinite(nonfinite_expected))
    assert np.array_equal(np.isnan(nonfinite_actual), np.isnan(nonfinite_expected))
    assert np.array_equal(np.isinf(nonfinite_actual), np.isinf(nonfinite_expected))


def test_snapshot_records_dispatch_and_restored_mask():
    coeff, spectrum = _complex_fixture((8, 2, 33))
    executor = KernelExecutor(workers=1)
    executor.k1(coeff, spectrum)
    executor.k2(spectrum.real, np.float64(2.0))
    snapshot = executor.snapshot()
    assert snapshot["dispatch_count"] == 2
    assert [row["operation"] for row in snapshot["dispatches"]] == ["k1", "k2"]
    for row in snapshot["dispatches"]:
        assert row["status"] == "PASS"
        assert row["thread_mask_inside"] == 1
        assert row["thread_mask_after"] == row["thread_mask_before"]
        assert row["requested_workers"] == 1
        assert row["active_thread_ids"] == [0]
        assert row["present_thread_ids"] == [0]
    assert snapshot["cache_identity"]["parallel"] == {"serial": False, "parallel": True}
    assert snapshot["cache_identity"]["compiled_signatures"]["k1"]
    assert snapshot["cache_identity"]["compiled_signatures"]["k2"]


def test_actual_two_worker_local_diagnostic_and_mask_restore():
    coeff, spectrum = _complex_fixture((8, 3, 37))
    executor = KernelExecutor(workers=2, diagnostic_local=True)
    result = executor.k1(coeff, spectrum)
    expected = coeff * spectrum
    assert _ordered_bytes(result) == _ordered_bytes(expected)
    row = executor.snapshot()["dispatches"][0]
    assert row["requested_workers"] == 2
    assert row["thread_mask_inside"] == 2
    assert row["thread_mask_after"] == row["thread_mask_before"]
    assert row["active_thread_ids"] == [0, 1]


def test_fixed_domain_and_optimized_python_guards(monkeypatch):
    coeff, spectrum = _complex_fixture((4, 2, 3))
    with pytest.raises(KernelUnsupported, match="WORKERS"):
        KernelExecutor(workers=2)
    with pytest.raises(KernelUnsupported, match="BLOCK_SIZE"):
        KernelExecutor(block_size=16)
    with pytest.raises(KernelUnsupported, match="ROLE"):
        KernelExecutor(role="I0")
    monkeypatch.setenv("PYTHONOPTIMIZE", "1")
    with pytest.raises(KernelUnsupported, match="PYTHONOPTIMIZE"):
        KernelExecutor()
    monkeypatch.delenv("PYTHONOPTIMIZE")
    assert coeff.shape == (spectrum.shape[0], 1, 1)

