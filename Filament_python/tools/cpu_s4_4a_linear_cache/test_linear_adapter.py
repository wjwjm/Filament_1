from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest

from . import candidate, linear_adapter


def test_source_ast_has_exact_transfer_and_phase_anchors():
    path = Path(__file__).resolve().parents[2] / "KHz_filament" / "linear.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == linear_adapter.LINEAR_FUNCTION_NAME)
    assert linear_adapter._find_transfer_lambda(fn) is not None
    assert sum(isinstance(node, ast.Return) for node in ast.walk(fn)) == 3


def test_unknown_producer_rejected_before_any_cache_or_fft_work():
    def wrong_producer(value):
        return value

    with pytest.raises(linear_adapter.LinearBindingError):
        linear_adapter.make_cached_linear(wrong_producer, object())


def _extract(result, *, energy: bool, profile: bool):
    if not energy and not profile:
        return result, None, None
    if energy and profile:
        return result[0], result[1], result[2]
    if energy:
        return result[0], result[1], None
    return result[0], None, result[1]


@pytest.mark.skipif(
    os.environ.get("CPU_S4_4A_RUN_REAL_LINEAR") != "1",
    reason="real J2 four-return-form check is enabled by the frozen runner",
)
def test_j2_cached_linear_four_return_forms_exact_layout_and_fft_counts(tmp_path):
    """Use the actual private G3 J2 original; no surrogate module is accepted."""

    import KHz_filament.propagate as propagate_module

    factory = candidate.make_factory(
        local_engineering=True,
        kernel_threads=1,
        temporal_workers=1,
        spatial_workers=1,
        raman_workers=1,
    )
    _, dispatch = factory(propagate_module.propagate_one_pulse)
    state = dispatch.cpu_s4_4a_state
    original = state._g_linear_function
    cached = state._cpu_s4_4a_cached_linear
    assert callable(original) and callable(cached)
    assert cached.__globals__ is not original.__globals__
    assert cached.__code__.co_name == linear_adapter.CACHED_FUNCTION_NAME
    assert str(cached.__code__.co_filename) == str(Path(linear_adapter.__file__).resolve())

    rng = np.random.default_rng(44)
    nt, ny, nx = 32, 3, 4
    field = np.asarray(rng.normal(size=(nt, ny, nx)) + 1j * rng.normal(size=(nt, ny, nx)), dtype=np.complex128)
    omega = np.linspace(-2.0e14, 2.0e14, nt, dtype=np.float64)
    kperp2 = np.arange(ny * nx, dtype=np.float64).reshape(ny, nx) * 1.0e4
    common = dict(
        Omega=omega,
        kperp2=kperp2,
        k0=1.0e7,
        omega0=2.0e15,
        dz=2.0e-4,
        beta2=1.0e-30,
        denom_floor=1.0e-4,
        precision_strategy="baseline_complex64",
    )
    forms = ((False, False), (True, False), (False, True), (True, True))
    for energy, profile in forms:
        kwargs = dict(common, return_energy_diagnostics=energy, return_profile_diagnostics=profile)
        baseline_before = dict(getattr(dispatch, "counts", {}))
        baseline_result = original(field.copy(), energy_scale=1.0, **kwargs) if energy else original(field.copy(), **kwargs)
        baseline_after = dict(getattr(dispatch, "counts", {}))
        baseline_delta = {
            key: int(baseline_after.get(key, 0)) - int(baseline_before.get(key, 0))
            for key in set(baseline_before) | set(baseline_after)
        }
        cached_before = dict(getattr(dispatch, "counts", {}))
        try:
            cached_result = cached(field.copy(), energy_scale=1.0, **kwargs) if energy else cached(field.copy(), **kwargs)
        except BaseException:
            state._cpu_s4_4a_cache.finish_half(success=False)
            raise
        state._cpu_s4_4a_cache.finish_half(success=True)
        cached_after = dict(getattr(dispatch, "counts", {}))
        cached_delta = {
            key: int(cached_after.get(key, 0)) - int(cached_before.get(key, 0))
            for key in set(cached_before) | set(cached_after)
        }
        baseline_field, baseline_energy, baseline_profile = _extract(baseline_result, energy=energy, profile=profile)
        cached_field, cached_energy, cached_profile = _extract(cached_result, energy=energy, profile=profile)
        np.testing.assert_array_equal(cached_field, baseline_field)
        assert cached_field.dtype == np.dtype(np.complex128)
        assert cached_field.flags.c_contiguous
        assert not np.shares_memory(cached_field, field)
        assert baseline_delta == cached_delta
        assert (baseline_energy is None) == (cached_energy is None)
        if baseline_energy is not None:
            assert baseline_energy.keys() == cached_energy.keys()
            for key in baseline_energy:
                assert cached_energy[key] == baseline_energy[key]
        assert (baseline_profile is None) == (cached_profile is None)

    private = state._cpu_s4_4a_private_binding
    helper_name = private["helper_name"]
    original_helper = private["helper"]
    private["namespace"][helper_name] = lambda *args: original_helper(*args)
    with pytest.raises(linear_adapter.LinearBindingError):
        linear_adapter._validate_cached_runtime(state, original, cached)
    private["namespace"][helper_name] = original_helper
    linear_adapter._validate_cached_runtime(state, original, cached)
