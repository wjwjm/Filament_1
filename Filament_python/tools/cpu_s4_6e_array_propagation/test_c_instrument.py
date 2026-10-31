"""Focused source-bound C nonlinear math checks."""

from __future__ import annotations

import types

import numpy as np
import pytest

from KHz_filament.nonlinear import apply_nonlinear

from .c_instrument import (
    CInstrumentationError,
    compile_private_apply,
    instrument_apply_nonlinear,
    validate_cpu_backend,
    verify_private_apply,
)
from .c_pool import CPool, CPoolError


def _arrays(shape=(7, 4, 3)):
    rng = np.random.default_rng(108)
    E = np.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape), dtype=np.complex128, order="C")
    phase = np.asarray(rng.normal(size=shape), dtype=np.float64, order="C")
    alpha = np.asarray(rng.normal(size=shape), dtype=np.float64, order="C")
    dn = np.asarray(rng.normal(size=shape[1:]), dtype=np.float64, order="C")
    return E, phase, alpha, dn


@pytest.mark.parametrize("workers", (1, 24))
def test_s46e_c_private_apply_exact_identity_and_prelude_broadcast(workers):
    E, phase, alpha, dn = _arrays()
    phase_before, alpha_before, dn_before = phase.copy(), alpha.copy(), dn.copy()
    ref = E.copy()
    expected = apply_nonlinear(ref, phase, alpha, 0.015, dn_gas=dn, k0=0.3)
    pool = CPool(workers)
    candidate, receipt = compile_private_apply(apply_nonlinear, pool)
    result = candidate(E, phase, alpha, 0.015, dn_gas=dn, k0=0.3)
    assert result is E
    assert np.array_equal(result, expected)
    assert np.array_equal(phase, phase_before)
    assert np.array_equal(alpha, alpha_before)
    assert np.array_equal(dn, dn_before)
    assert receipt["prelude_preserved"] is True
    assert receipt["replacement_counts"] == {"phase": 1, "absorption": 1}
    assert verify_private_apply(candidate)["two_rounds"] is True
    pool.close()


def test_s46e_c_private_apply_zero_nan_inf_and_return_object():
    E, phase, alpha, _ = _arrays((5, 2, 2))
    phase[:] = np.nan
    alpha[0, 0, 0] = np.inf
    alpha[1, 0, 0] = -np.inf
    ref = E.copy()
    expected = apply_nonlinear(ref, phase, alpha, 0.0)
    pool = CPool(1)
    candidate, _ = compile_private_apply(apply_nonlinear, pool)
    out = candidate(E, phase, alpha, 0.0)
    assert out is E
    assert np.array_equal(out, expected, equal_nan=True)
    pool.close()


def test_s46e_c_compiled_and_helper_seals_reject_tampering():
    E, phase, alpha, _ = _arrays((3, 2, 2))
    pool = CPool(1)
    candidate, _ = compile_private_apply(apply_nonlinear, pool)
    function_seal = candidate.__cpu_s46e_c_function_seal__
    original_code = function_seal["code"]
    function_seal["code"] = object()
    with pytest.raises(CInstrumentationError, match="CODE_CHANGED"):
        verify_private_apply(candidate)
    function_seal["code"] = original_code
    helper_seals = candidate.__cpu_s46e_c_helper_seals__
    guard_seal = helper_seals[next(iter(helper_seals))]
    closure_cells = guard_seal["closure_cells"]
    assert closure_cells
    original_value = closure_cells[0]["value"]
    closure_cells[0]["value"] = object()
    with pytest.raises(CInstrumentationError, match="CLOSURE_VALUE_CHANGED"):
        verify_private_apply(candidate)
    closure_cells[0]["value"] = original_value
    assert verify_private_apply(candidate)["two_rounds"] is True
    pool.close()


def test_s46e_c_rejects_alias_readonly_and_bad_backend_before_write():
    E, phase, alpha, _ = _arrays()
    pool = CPool(1)
    candidate, _ = compile_private_apply(apply_nonlinear, pool)
    original = E.copy()
    alias = E.view(np.float64).reshape(E.shape + (2,))[..., 0]
    with pytest.raises(CPoolError, match="ALPHA_E_ALIAS"):
        candidate(E, phase, alias, 0.1)
    assert np.array_equal(E, original)
    readonly = E.copy()
    readonly.flags.writeable = False
    with pytest.raises(CPoolError, match="WRITABLE"):
        candidate(readonly, phase, alpha, 0.1)
    assert np.array_equal(E, original)
    bad = types.SimpleNamespace(xp=np, abs=np.add)
    with pytest.raises(CInstrumentationError, match="ABS"):
        validate_cpu_backend(bad)
    old_abs = np.abs
    try:
        np.abs = np.add
        with pytest.raises(CInstrumentationError, match="ABS"):
            candidate(E, phase, alpha, 0.1)
    finally:
        np.abs = old_abs
    private_globals = candidate.__globals__
    original_xp = private_globals["xp"]
    try:
        private_globals["xp"] = object()
        with pytest.raises(CInstrumentationError, match="XP_SLOT"):
            candidate(E, phase, alpha, 0.1)
    finally:
        private_globals["xp"] = original_xp
    pool.close()


def test_s46e_c_ast_inverse_proof_requires_two_original_updates():
    import ast
    from pathlib import Path

    source = Path(apply_nonlinear.__code__.co_filename).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "apply_nonlinear")
    receipt = instrument_apply_nonlinear(function)
    assert receipt["stripped_ast_sha256"] == receipt["source_ast_sha256"]
    assert receipt["math_reordering"] is False


def test_s46e_c_workers_are_source_ast_slices_and_sealed():
    pool = CPool(1)
    candidate, receipt = compile_private_apply(apply_nonlinear, pool)
    workers = candidate.__cpu_s46e_c_workers__
    worker_receipts = candidate.__cpu_s46e_c_worker_receipts__
    assert set(workers) == {"phase", "absorption"}
    assert set(worker_receipts) == {"phase", "absorption"}
    for kind, array_name in (("phase", "phase"), ("absorption", "alpha")):
        worker = workers[kind]
        worker_receipt = worker_receipts[kind]
        assert worker.__globals__ is candidate.__globals__
        assert candidate.__globals__[worker.__name__] is worker
        assert worker_receipt["array_name"] == array_name
        assert worker_receipt["axis"] == 0
        assert worker_receipt["astype_copy_false"] is True
        assert worker_receipt["original_ast_sha256"] != worker_receipt["sliced_ast_sha256"]
        assert "Subscript" in worker_receipt["sliced_ast"]
        assert receipt["workers"][kind]["sliced_ast_sha256"] == worker_receipt["sliced_ast_sha256"]
        assert candidate.__cpu_s46e_c_instrumentation__["workers"][kind]["original_ast_sha256"] == worker_receipt["original_ast_sha256"]
    assert verify_private_apply(candidate)["workers"]["phase"]["original_ast_sha256"]
    pool.close()


def test_s46e_c_worker_code_tamper_is_rejected():
    pool = CPool(1)
    candidate, _ = compile_private_apply(apply_nonlinear, pool)
    worker = candidate.__cpu_s46e_c_workers__["phase"]
    original_code = worker.__code__
    worker.__code__ = (lambda E, phase, onej, ctype, lo, hi: None).__code__
    try:
        with pytest.raises(CInstrumentationError, match="C_SEAL_CODE_CHANGED"):
            verify_private_apply(candidate)
    finally:
        worker.__code__ = original_code
    original_hash = candidate.__cpu_s46e_c_worker_receipts__["phase"]["sliced_ast_sha256"]
    candidate.__cpu_s46e_c_worker_receipts__["phase"]["sliced_ast_sha256"] = "tampered"
    try:
        with pytest.raises(CInstrumentationError, match="C_WORKER_RECEIPT_CHANGED"):
            verify_private_apply(candidate)
    finally:
        candidate.__cpu_s46e_c_worker_receipts__["phase"]["sliced_ast_sha256"] = original_hash
    assert verify_private_apply(candidate)["two_rounds"] is True
    pool.close()


def test_s46e_c_worker_expression_shape_rejects_deformation():
    import ast
    from pathlib import Path

    source = Path(apply_nonlinear.__code__.co_filename).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "apply_nonlinear")
    update = next(
        node for node in function.body
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "E"
    )
    update.value.func.value.func.attr = "log"
    with pytest.raises(CInstrumentationError, match="C_AUGASSIGN_EXPRESSION_SHAPE"):
        instrument_apply_nonlinear(function)


def test_s46e_c_cast_dtype_shape_rejects_non_ctype_name():
    import ast
    from pathlib import Path

    source = Path(apply_nonlinear.__code__.co_filename).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "apply_nonlinear")
    update = next(
        node for node in function.body
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "E"
    )
    update.value.args[0] = ast.Name(id="wrong_dtype", ctx=ast.Load())
    with pytest.raises(CInstrumentationError, match="C_AUGASSIGN_EXPRESSION_SHAPE"):
        instrument_apply_nonlinear(function)
