"""Focused exactness tests for the CPU-S4-5D private fluence helper."""

from __future__ import annotations

import ast
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from KHz_filament.diagnostics import (
    _fwhm_diameter_xy_center,
    second_moment_radius,
    second_moment_radius_from_2d,
)

from .adapter import (
    AdapterBindingError,
    BodyClock,
    FluenceCounter,
    _StripS45D,
    _apply_fluence_nodes,
    bind_helper_slots,
    verify_helper_slots,
)
from .fluence import postprocess_source_proof, second_moment_radius_from_f2d


def _integral(I3D: np.ndarray, dt: float) -> np.ndarray:
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return trapezoid(I3D, dx=dt, axis=0)


@pytest.mark.parametrize('fraction',(0.75,0.8))
def test_disambiguating_fixture_uses_original_three_d_threshold_rule(fraction) -> None:
    I3D = np.stack([[[5.0, 3.0, 2.0]], [[5.0, 3.0, 2.0]]]).astype(np.float64)
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    y = np.array([0.0], dtype=np.float64)
    F2D = _integral(I3D, 1.0)

    original = second_moment_radius(I3D, x, y, dt=1.0, frac_keep=fraction, rel_floor=0.0)
    private = second_moment_radius_from_f2d(F2D, x, y, frac_keep=fraction, rel_floor=0.0)
    public_2d = second_moment_radius_from_2d(F2D, x, y, frac_keep=fraction, rel_floor=0.0)

    assert original == 1.118033988749895
    assert private == original
    assert public_2d == 1.4142135623730951
    assert private != public_2d


@pytest.mark.parametrize("frac_keep", [0.0, 1.0, 0.999, 0.75, 0.8])
@pytest.mark.parametrize("rel_floor", [0.0, 1e-8, 1.0])
@pytest.mark.parametrize("shape", [(384, 3, 4), (5, 2, 3), (4, 1, 1)])
def test_private_post_integral_matches_original_oracle(
    frac_keep: float, rel_floor: float, shape: tuple[int, int, int]
) -> None:
    rng = np.random.default_rng(20261028 + sum(shape))
    I3D = rng.random(shape, dtype=np.float64)
    # Include a small asymmetric component so both coordinate moments are used.
    I3D[:, 0, 0] *= 1.25
    x = np.linspace(-1.0, 1.0, shape[2], dtype=np.float64)
    y = np.linspace(-0.5, 0.5, shape[1], dtype=np.float64)
    dt = float(1.0 / max(shape[0] - 1, 1))
    F2D = _integral(I3D, dt)

    original = second_moment_radius(
        I3D, x, y, dt=dt, frac_keep=frac_keep, rel_floor=rel_floor
    )
    private = second_moment_radius_from_f2d(
        F2D, x, y, frac_keep=frac_keep, rel_floor=rel_floor
    )
    assert private == original


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((4, 2, 3), dtype=np.float64),
        np.full((4, 2, 3), np.finfo(np.float64).tiny),
        np.array([[[0.0, -0.0, -1.0], [np.nan, np.inf, -np.inf]]] * 4),
        np.array([[[np.nextafter(0.0, 1.0), 2.0, 3.0], [4.0, 5.0, 6.0]]] * 4),
    ],
)
def test_zero_tiny_subnormal_signedzero_negative_nan_inf_policy(values: np.ndarray) -> None:
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    y = np.array([-1.0, 1.0], dtype=np.float64)
    F2D = _integral(values, 1.0)
    original = second_moment_radius(values, x, y, dt=1.0, frac_keep=0.999, rel_floor=1e-8)
    private = second_moment_radius_from_f2d(F2D, x, y, frac_keep=0.999, rel_floor=1e-8)
    assert private == original


def test_f2d_dtype_strides_and_input_bytes_are_unchanged_after_helper_and_fwhm() -> None:
    I3D = np.arange(2 * 3 * 5, dtype=np.float64).reshape(2, 3, 5) + 1.0
    F2D = _integral(I3D, 0.125)
    F2D = F2D[:, ::2]  # non-contiguous supported input layout
    x = np.array([-2.0, 0.0, 2.0], dtype=np.float64)
    y = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    before = F2D.tobytes(order="A")
    before_dtype = F2D.dtype
    before_shape = F2D.shape
    before_strides = F2D.strides

    _ = second_moment_radius_from_f2d(F2D, x, y, frac_keep=0.8, rel_floor=0.0)
    _ = _fwhm_diameter_xy_center(F2D, type("Axes", (), {"x": x, "y": y})(), 1, 1)
    _ = second_moment_radius_from_f2d(F2D, x, y, frac_keep=1.0, rel_floor=1e-8)

    assert F2D.tobytes(order="A") == before
    assert F2D.dtype == before_dtype
    assert F2D.shape == before_shape
    assert F2D.strides == before_strides


def test_singleton_axes_and_odd_even_tail_shapes_match_oracle() -> None:
    for shape in ((3, 1, 1), (4, 1, 3), (5, 2, 1), (6, 3, 4), (7, 2, 5)):
        I3D = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + 1.0
        x = np.arange(shape[2], dtype=np.float64) if shape[2] > 1 else np.array([0.0])
        y = np.arange(shape[1], dtype=np.float64) if shape[1] > 1 else np.array([0.0])
        dt = 0.25
        F2D = _integral(I3D, dt)
        original = second_moment_radius(I3D, x, y, dt=dt, frac_keep=0.999, rel_floor=0.0)
        private = second_moment_radius_from_f2d(F2D, x, y, frac_keep=0.999, rel_floor=0.0)
        assert private == original


def test_unsupported_backend_or_dtype_fails_closed_without_fallback() -> None:
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    y = np.array([0.0], dtype=np.float64)
    with pytest.raises(TypeError, match="F2D_FLOAT64"):
        second_moment_radius_from_f2d(np.ones((1, 3), dtype=np.float32), x, y)
    with pytest.raises(TypeError, match="F2D_FLOAT64"):
        second_moment_radius_from_f2d([[1.0, 2.0, 3.0]], x, y)


def test_ast_change_is_reversible_for_reference_and_reuse() -> None:
    repo = Path(__file__).resolve().parents[3]
    source_path = repo / "Filament_python/KHz_filament/propagate.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    original = copy.deepcopy(module)
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "propagate_one_pulse")

    for candidate_applied in (False, True):
        transformed = copy.deepcopy(original)
        transformed_function = next(
            node
            for node in transformed.body
            if isinstance(node, ast.FunctionDef) and node.name == "propagate_one_pulse"
        )
        receipt = _apply_fluence_nodes(transformed_function, candidate_applied=candidate_applied)
        assert receipt["candidate_applied"] is candidate_applied
        restored = _StripS45D().visit(transformed)
        assert ast.dump(restored, include_attributes=False) == ast.dump(
            original, include_attributes=False
        )


def test_scalar_counter_reports_reference_and_reuse_contract_without_arrays() -> None:
    reference = FluenceCounter(role="L1_REF_5D", candidate_applied=False)
    reuse = FluenceCounter(role="D_FLUENCE_REUSE_V1", candidate_applied=True)
    for interval in range(8):
        reference.saved_event(interval, 0.25)
        reference.radius_event(1.0)
        reference.fwhm_event(1.0)
        reuse.saved_event(interval, 0.25)
        reuse.radius_event(1.0)
        reuse.fwhm_event(1.0)
    a = reference.snapshot()
    d = reuse.snapshot()
    assert (a["saved_calls"], a["outer_integrals"], a["radius_calls"], a["fwhm_calls"]) == (8, 8, 8, 8)
    assert (d["saved_calls"], d["outer_integrals"], d["radius_calls"], d["fwhm_calls"]) == (8, 8, 8, 8)
    assert a["fwhm_calls_total_known"] == d["fwhm_calls_total_known"] == 16
    assert a["total_fwhm_known2S"] == d["total_fwhm_known2S"] == 16
    assert a["inner_integrals"] == 8
    assert d["inner_integrals"] == 0
    assert a["arrays_retained"] is d["arrays_retained"] is False
    assert a["unobserved_counts"] == d["unobserved_counts"] == [
        "inner_integrals_backend_trapezoid_not_directly_hooked"
    ]


def test_private_postprocess_ast_is_bound_to_original_diagnostics() -> None:
    proof = postprocess_source_proof()
    assert proof["post_ast_equal"] is True
    assert proof["original_post_ast_sha256"] == proof["private_post_ast_sha256"]
    assert proof["public_2d_helper_references"] == 0


def test_private_clone_helper_slots_are_sealed_and_mutation_fails_closed() -> None:
    body_clock = BodyClock()
    counter = FluenceCounter(role="D_FLUENCE_REUSE_V1", candidate_applied=True)
    from .fluence import second_moment_radius_from_f2d

    helpers = {
        "__cpu_s45d_body__": body_clock.body,
        "__cpu_s45d_saved_event__": counter.saved_event,
        "__cpu_s45d_radius_event__": counter.radius_event,
        "__cpu_s45d_fwhm_event__": counter.fwhm_event,
        "__cpu_s45d_second_moment_radius__": second_moment_radius_from_f2d,
    }
    clone = SimpleNamespace(__globals__=dict(helpers))
    bindings = bind_helper_slots(
        clone,
        helpers,
        expected_owners={
            "__cpu_s45d_body__": body_clock,
            "__cpu_s45d_saved_event__": counter,
            "__cpu_s45d_radius_event__": counter,
            "__cpu_s45d_fwhm_event__": counter,
            "__cpu_s45d_second_moment_radius__": None,
        },
    )
    verify_helper_slots(clone, bindings)
    clone.__globals__["__cpu_s45d_second_moment_radius__"] = lambda *args, **kwargs: 0.0
    with pytest.raises(AdapterBindingError, match="S45D_HELPER_SLOT_CHANGED"):
        verify_helper_slots(clone, bindings)


def test_private_namespace_import_does_not_load_eq10_package() -> None:
    forbidden_prefix = "tools.cpu_s4_5" + "c_eq10_pointwise"
    assert not any(name.startswith(forbidden_prefix) for name in sys.modules)
