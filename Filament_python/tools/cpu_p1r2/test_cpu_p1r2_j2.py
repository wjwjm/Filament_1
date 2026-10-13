from __future__ import annotations

import numpy as np
import pytest

from tools.cpu_p1r2 import cpu_p1r2_j1 as j1
from tools.cpu_p1r2 import cpu_p1r2_j2 as j2


def test_candidate_worker_descriptor_is_strict_and_bounded():
    assert j2._validate_descriptor({"temporal_workers": 24, "spatial_workers": 1}) == {
        "temporal_workers": 24, "spatial_workers": 1}
    for value in (0, 2, 97, True, 1.9):
        with pytest.raises(j2.J2Error):
            j2._validate_descriptor({"temporal_workers": value, "spatial_workers": 1})
    with pytest.raises(j2.J2Error):
        j2._validate_descriptor({"temporal_workers": 1})


def test_layout_restore_preserves_positive_strides_and_values():
    base = np.arange(35, dtype=np.float64).reshape(5, 7)
    view = base[::2, ::2]
    restored = j2.restore_layout(np.array(view, copy=True), shape=view.shape,
        dtype=view.dtype, strides=view.strides, expected_hash=j2._sha_array(view))
    assert restored.strides == view.strides
    assert restored.flags.writeable
    assert not np.shares_memory(restored, view)
    np.testing.assert_array_equal(restored, view)
    with pytest.raises(j2.J2Error, match="positive strides"):
        j2.restore_layout(np.array(view), shape=view.shape, dtype=view.dtype,
            strides=(view.strides[0], 0))


def test_scoped_scipy_dispatch_preserves_axes_and_does_not_mutate_science_globals():
    import KHz_filament.linear as linear
    import KHz_filament.raman as raman
    from KHz_filament.device import xp

    class FakeScipyFFT:
        def __init__(self):
            self.calls = []
        def __getattr__(self, name):
            def call(value, *args, workers=None, overwrite_x=None, **kwargs):
                self.calls.append((name, workers, overwrite_x, kwargs.copy()))
                return getattr(np.fft, name)(value, *args, **kwargs)
            return call

    before = (linear.xp, raman.xp, np.fft)
    fake = FakeScipyFFT()
    dispatch = j2._ScipyDispatch(xp.fft, fake,
        {"temporal_workers": 8, "spatial_workers": 24})
    value = np.arange(24, dtype=np.complex128).reshape(2, 3, 4)
    result = dispatch.fft(value, axis=0, norm="ortho")
    np.testing.assert_array_equal(result, np.fft.fft(value, axis=0, norm="ortho"))
    plane = np.arange(12, dtype=np.complex128).reshape(3, 4)
    result2 = dispatch.fft2(plane, axes=(0, 1), norm="forward")
    np.testing.assert_array_equal(result2, np.fft.fft2(plane, axes=(0, 1), norm="forward"))
    assert fake.calls == [
        ("fft", 8, False, {"axis": 0, "norm": "ortho"}),
        ("fft2", 24, False, {"axes": (0, 1), "norm": "forward"})]
    assert (linear.xp, raman.xp, np.fft) == before


def test_private_propagator_clone_wires_candidate_dispatch_only():
    import KHz_filament.linear as linear
    import KHz_filament.raman as raman
    from KHz_filament.device import xp

    class FakeFFT:
        def __getattr__(self, name):
            return getattr(np.fft, name)

    dispatch = j2._ScipyDispatch(xp.fft, FakeFFT(),
        {"temporal_workers": 1, "spatial_workers": 1})
    cloned = j2._clone_propagator(dispatch)
    from KHz_filament.propagate import propagate_one_pulse
    assert callable(cloned)
    assert cloned.__code__ is propagate_one_pulse.__code__
    assert linear.xp is xp and raman.xp is xp


def test_fft_call_parameters_and_comparison_metrics_keep_zero_reference_explicit():
    record = {"transform": "fft", "call_args_tail": [],
        "call_kwargs": {"axis": 0, "norm": "ortho"}}
    values = np.arange(12, dtype=np.complex128).reshape(3, 4)
    np.testing.assert_array_equal(j2._capture_transform(record, values, "numpy"),
        np.fft.fft(values, axis=0, norm="ortho"))
    zeros = np.zeros(4, dtype=np.complex128)
    same = j2._array_error_metrics(zeros, zeros.copy())
    assert same["canonical_exact"] and same["bitwise_equal"]
    assert same["normalized_Linf"] == 0.0 and same["relative_L2"] == 0.0
    changed = zeros.copy(); changed[1] = 1.0
    metrics = j2._array_error_metrics(zeros, changed)
    assert not metrics["canonical_exact"]
    assert metrics["normalized_Linf"] is None
    assert metrics["relative_L2"] is None
    assert metrics["zero_reference_element_count"] == 4
    assert metrics["max_abs_error_on_zero_reference_elements"] == 1.0
    with pytest.raises(j2.J2Error, match="NaN/Inf"):
        j2._array_error_metrics(zeros, np.array([np.nan] * 4, dtype=np.complex128))


def test_j2_median_and_strided_boolean_metrics():
    assert j2._summarize_samples([{"s": 2}, {"s": 8}], "s")["p50"] == 5
    a = np.array([[True, False], [False, True]])[:, ::-1]
    m = j2._array_error_metrics(a, a.copy())
    assert m["bitwise_equal"] and m["canonical_exact"]
    b = a.copy(); b[0, 0] = not b[0, 0]
    assert j2._array_error_metrics(a, b)["max_abs"] == 1


def test_j2_actual_parent_child_capture_and_candidate_entry(tmp_path):
    from pathlib import Path
    repo = Path(j2.__file__).resolve().parents[3]
    assert j2.main(["--phase", "J2", "--entry-smoke", "--bench-root", str(repo),
        "--output-root", str(tmp_path / "entry")]) == 0
