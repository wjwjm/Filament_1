"""Low-cost private Eq10/P48 contract tests; no propagation or HPC access."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from .candidate import (
    ROLE_CANDIDATE,
    ROLE_REFERENCE,
    _owned_source_identity,
    build_role,
    verify_bindings,
)
from .instrument import HELPER_NAMES, _helpers, instrument_eq10_stage
from .pool import Eq10Pool, PoolError, SafeWorkers


def _stage_node() -> ast.FunctionDef:
    source = Path(__file__).resolve().parents[2] / "KHz_filament" / "raman.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "isaacs_raman_stage")


def test_s46d_ast_replaces_exactly_four_pointwise_segments() -> None:
    receipt = instrument_eq10_stage(_stage_node())
    assert receipt["replacement_counts"] == {
        "spectral": 1,
        "derivative": 1,
        "response": 1,
        "product": 1,
    }
    assert receipt["stripped_ast_sha256"] == receipt["source_ast_sha256"]
    assert receipt["fft_calls"] == receipt["ifft_calls"] == receipt["sum_calls"] == 1


def test_s46d_return_energy_false_and_zero_branch_remain_unmodified() -> None:
    stage = _stage_node()
    receipt = instrument_eq10_stage(stage)
    calls = [
        node.func.id
        for node in ast.walk(stage)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert all(name in calls for name in HELPER_NAMES.values())
    return_energy = next(
        node
        for node in stage.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "return_energy"
    )
    positive = next(
        node
        for node in return_energy.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "intensity_scale"
    )
    zero_branch = positive.orelse
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in set(HELPER_NAMES.values())
        for branch in zero_branch
        for node in ast.walk(branch)
    )
    assert receipt["private_guard_before_helper_calls"] is True


def test_s46d_pool_exact_pointwise_result_and_idempotent_close() -> None:
    pool = Eq10Pool(1)
    source = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    output = np.empty_like(source)
    pool.ufunc("B_RESPONSE_DIVIDE", np.divide, (source, 2.0), output)
    np.testing.assert_array_equal(output, source / 2.0)
    pool.close()
    pool.close()
    assert pool.snapshot()["close_ns"] is not None
    assert pool.snapshot()["lifetime_ns"] >= pool.snapshot()["close_ns"]
    with pytest.raises(PoolError, match="RETIRED"):
        pool.ufunc("B_RESPONSE_DIVIDE", np.divide, (source, 2.0), output)


def test_s46d_pool_failure_retires_after_join() -> None:
    pool = Eq10Pool(1)
    source = np.ones((2, 3, 4), dtype=np.float64)
    output = np.empty_like(source)
    with pytest.raises(Exception):
        pool.ufunc("B_RESPONSE_DIVIDE", np.divide, (source, "not-a-number"), output)
    assert pool.snapshot()["closed"] is True


def test_s46d_four_helpers_exact_real_view_input_and_output_lifetime() -> None:
    pool = Eq10Pool(1)
    helpers = _helpers(pool)
    complex_source = np.ascontiguousarray(
        (np.arange(24, dtype=np.float64).reshape(4, 3, 2) + 1j),
        dtype=np.complex128,
    )
    multiplier = np.arange(4, dtype=np.float64).reshape(4, 1, 1) + 1j
    spectrum = complex_source * (2.0 - 0.5j)
    first = helpers[HELPER_NAMES["spectral"]](multiplier, spectrum)
    assert first.flags.owndata and first.flags.c_contiguous
    np.testing.assert_array_equal(first, np.multiply(multiplier, spectrum))

    inverse = np.ascontiguousarray(first)
    real_view = inverse.real
    assert np.shares_memory(real_view, inverse)
    derivative = helpers[HELPER_NAMES["derivative"]](inverse, 2.5)
    np.testing.assert_array_equal(derivative, real_view / 2.5)
    response = np.arange(24, dtype=np.float64).reshape(4, 3, 2)
    response_out = helpers[HELPER_NAMES["response"]](response, 2.5)
    product_out = helpers[HELPER_NAMES["product"]](response_out, derivative)
    np.testing.assert_array_equal(response_out, response / 2.5)
    np.testing.assert_array_equal(product_out, (response / 2.5) * (real_view / 2.5))
    assert first is not response_out and response_out is not product_out
    np.testing.assert_array_equal(first, np.multiply(multiplier, spectrum))
    assert pool.snapshot()["operation_counts"] == {
        "B_SPECTRAL_MULTIPLY": 1,
        "B_DERIVATIVE_DIVIDE": 1,
        "B_RESPONSE_DIVIDE": 1,
        "B_PRODUCT": 1,
    }
    pool.close()


def test_s46d_actual_scipy_ifft_real_view_exact():
    from scipy import fft
    source = np.arange(384*3*2, dtype=np.float64).reshape(384, 3, 2)
    before = source.copy()
    inverse = fft.ifft(fft.fft(source, axis=0, workers=1), axis=0, workers=1)
    assert np.shares_memory(inverse.real, inverse) and not inverse.real.flags.c_contiguous
    with_values = inverse.copy()
    pool = Eq10Pool(1)
    try:
        helper = _helpers(pool)[HELPER_NAMES['derivative']]
        first = helper(inverse, 7.0)
        saved = first.copy()
        second = helper(inverse, 3.0)
        assert first.tobytes() == (inverse.real/7.0).tobytes()
        assert first.tobytes() == saved.tobytes()
        assert not np.shares_memory(first, inverse) and not np.shares_memory(first, second)
        assert source.tobytes() == before.tobytes() and inverse.tobytes() == with_values.tobytes()
    finally:
        pool.close()


@pytest.mark.parametrize('kind', ('zero', 'nan', 'inf', 'energy_off'))
def test_s46d_original_energy_branch_execution(kind):
    import copy
    import scipy.fft
    from KHz_filament.constants import c0
    class XP:
        fft = scipy.fft
        def __getattr__(self, name):
            return getattr(np, name)
    def compile_energy(stage, helpers):
        branch = next(node for node in stage.body if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'return_energy')
        template = ast.parse('def energy(intensity,response,Omega,dt,n_R,return_energy):\n rdtype=xp.float64\n result={}\n pass\n return result\n')
        template.body[0].body[2:3] = [copy.deepcopy(branch)]
        scope = dict(xp=XP(), c0=c0, **helpers)
        exec(compile(ast.fix_missing_locations(template), '<source-bound-energy-test>', 'exec'), scope)
        return scope['energy']
    stage = _stage_node()
    reference = compile_energy(stage, {})
    instrument_eq10_stage(stage)
    pool = Eq10Pool(1)
    candidate = compile_energy(stage, _helpers(pool))
    source = np.ones((8, 3, 2), dtype=np.float64)
    if kind == 'zero': source.fill(0)
    if kind == 'nan': source[0, 0, 0] = np.nan
    if kind == 'inf': source[0, 0, 0] = np.inf
    frozen = source.tobytes()
    args = (source, np.ones_like(source), np.arange(8, dtype=float), 1e-15, 1e-23, kind != 'energy_off')
    try:
        with np.errstate(all='ignore'):
            left, right = reference(*args), candidate(*args)
        assert left.keys() == right.keys()
        for key in left:
            np.testing.assert_equal(left[key], right[key])
        assert source.tobytes() == frozen
        if kind in ('zero', 'nan', 'energy_off'):
            assert sum(pool.snapshot()['operation_counts'].values()) == 0
    finally:
        pool.close()


def test_s46d_zero_nan_inf_and_return_energy_off_are_native_transparent() -> None:
    pool = Eq10Pool(1)
    helpers = _helpers(pool)
    values = np.array(
        [[[0.0, np.nan], [np.inf, -np.inf]], [[1.0, -2.0], [3.0, 4.0]]],
        dtype=np.float64,
    )
    with np.errstate(all="ignore"):
        result = helpers[HELPER_NAMES["response"]](values, 2.0)
    np.testing.assert_equal(result, np.divide(values, 2.0))
    values32 = np.ascontiguousarray(values, dtype=np.float32)
    result32 = helpers[HELPER_NAMES["response"]](values32, np.float32(2.0))
    assert result32.dtype == np.dtype("float32")
    np.testing.assert_array_equal(result32, np.divide(values32, np.float32(2.0)))
    zero = np.zeros((2, 2, 2), dtype=np.float64)
    product = helpers[HELPER_NAMES["product"]](zero, zero)
    np.testing.assert_array_equal(product, zero)
    pool.close()
    assert pool.snapshot()["closed"] is True


def test_s46d_pool_rejects_alias_and_fixed_backend_changes() -> None:
    pool = Eq10Pool(1)
    values = np.ones((2, 2, 2), dtype=np.float64)
    with pytest.raises(PoolError, match="ALIAS"):
        pool.ufunc("B_RESPONSE_DIVIDE", np.divide, (values, 2.0), values)
    with pytest.raises(PoolError, match="BACKEND"):
        pool.ufunc("B_RESPONSE_DIVIDE", np.add, (values, 2.0), np.empty_like(values))
    pool.close()


def test_s46d_owned_source_identity_includes_chains_pv2_record() -> None:
    identity = _owned_source_identity()
    files = identity["files"]
    assert "chains.py" in files
    chains = files["chains.py"]
    assert chains["classification"] == "tracked_text"
    assert chains["hash_scope"] == "git_blob_oid+canonical_lf_sha256"
    assert isinstance(chains["git_blob_oid"], str) and len(chains["git_blob_oid"]) == 40
    assert isinstance(chains["canonical_lf_sha256"], str) and len(chains["canonical_lf_sha256"]) == 64


def test_s46d_null_pool_close_is_idempotent_and_marks_retired() -> None:
    from .pool import NullEq10Pool

    pool = NullEq10Pool()
    assert pool.snapshot()["closed"] is False
    pool.close()
    pool.close()
    assert pool.snapshot()["closed"] is True


def test_s46d_constructor_failure_reclaims_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    import concurrent.futures
    import tools.cpu_s4_6d_eq10_propagation.pool as pool_module

    executor_box: dict[str, concurrent.futures.ThreadPoolExecutor] = {}

    def fail_init(self: object, count: int) -> None:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor_box["executor"] = executor
        self.executor = executor
        raise RuntimeError("constructor failure")

    monkeypatch.setattr(pool_module.Workers, "__init__", fail_init)
    with pytest.raises(RuntimeError, match="constructor failure"):
        SafeWorkers(1)
    assert executor_box["executor"]._shutdown is True


def test_s46d_task_exception_closes_and_rejects_reuse() -> None:
    pool = Eq10Pool(1)
    values = np.ones((2, 2, 2), dtype=np.float64)
    output = np.empty_like(values)
    with pytest.raises(Exception):
        pool.ufunc("B_RESPONSE_DIVIDE", np.divide, (values, "not-a-number"), output)
    assert pool.snapshot()["closed"] is True
    with pytest.raises(PoolError, match="RETIRED"):
        pool.ufunc("B_RESPONSE_DIVIDE", np.divide, (values, 2.0), output)


def _set_local_factory_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = Path(__file__).resolve().parents[3]
    for name in ("S43G2_SOURCE_ROOT", "S43F_SOURCE_ROOT", "S43C_SOURCE_ROOT"):
        monkeypatch.setenv(name, str(repo))
    monkeypatch.setenv("UPPE_USE_GPU", "0")
    monkeypatch.setenv("NUMBA_NUM_THREADS", "1")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    historical = Path(
        r"D:\Filament_1\.worktrees\cpu_optimization\reports\CPU_S4_5B_STEADY_COST_20261027_223422\external_data\G2_BOUNDARY_RECEIPT.json"
    )
    if historical.is_file():
        monkeypatch.setenv("CPU_BOUNDARY_HISTORICAL_RECEIPT", str(historical))


@pytest.mark.parametrize("role", (ROLE_REFERENCE, ROLE_CANDIDATE))
def test_s46d_factory_roles_construct_without_propagation(
    role: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("numba", reason="local-test environment lacks numba; factory construction is deferred")
    _set_local_factory_environment(monkeypatch)
    dispatch = None
    built = build_role(
        "PURE_OFF_SMALLGRID",
        capture_root=tmp_path / role,
        role=role,
        local_engineering=True,
        pool_override=1,
    )
    dispatch = built[1]
    try:
        assert len(built) == 6
        assert dispatch.cpu_s4_5b_recorder_pair is not None
        assert dispatch.cpu_s4_6d_role == role
        assert dispatch.cpu_s4_6d_binding["common_body_instrumentation_bound"] is True
        check = verify_bindings(dispatch)
        assert check["common_body_instrumentation_bound"] is True
        if role == ROLE_CANDIDATE:
            assert check["stage_replaced"] is True
            assert check["helper_count"] == 4
        else:
            assert check["stage_replaced"] is False
    finally:
        if dispatch is not None:
            dispatch.close_eq10_pool()


def test_s46d_factory_rejects_stage_slot_and_backend_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("numba", reason="local-test environment lacks numba; factory construction is deferred")
    _set_local_factory_environment(monkeypatch)
    built = build_role(
        "PURE_OFF_SMALLGRID",
        capture_root=tmp_path / "B",
        role=ROLE_CANDIDATE,
        local_engineering=True,
        pool_override=1,
    )
    dispatch = built[1]
    state = dispatch.cpu_s4_6d_state
    stage = state._internal_stage_inner
    helper_name = next(iter(HELPER_NAMES.values()))
    original_helper = stage.__globals__[helper_name]
    original_xp = stage.__globals__["xp"]
    original_scipy = original_xp.fft.dispatch.scipy_fft
    original_defaults = dict(stage.__kwdefaults__)
    required = dict(Omega=None, dt=None, omega0=None, n0=None, n_R=None, omega_R=None, Gamma_R=None)
    try:
        stage.__globals__[helper_name] = lambda *args: None
        with pytest.raises(Exception, match="SLOT"):
            verify_bindings(dispatch)
        with pytest.raises(Exception, match="HELPER_SLOT"):
            stage(None, **required)
        stage.__globals__[helper_name] = original_helper
        original_xp.fft.dispatch.scipy_fft = object()
        with pytest.raises(Exception, match="SCIPY_FFT"):
            stage(None, **required)
        original_xp.fft.dispatch.scipy_fft = original_scipy
        stage.__kwdefaults__['return_energy'] = False
        with pytest.raises(Exception, match="KWDEFAULTS"):
            stage(None, **required)
        stage.__kwdefaults__.update(original_defaults)
        stage.__globals__["xp"] = object()
        with pytest.raises(Exception, match="CPU_XP_NAMESPACE_TYPE"):
            verify_bindings(dispatch)
        stage.__globals__["xp"] = original_xp
        state._internal_stage_inner = lambda *args, **kwargs: None
        with pytest.raises(Exception, match="ACTIVE_STAGE"):
            verify_bindings(dispatch)
    finally:
        state._internal_stage_inner = stage
        stage.__globals__[helper_name] = original_helper
        stage.__globals__["xp"] = original_xp
        original_xp.fft.dispatch.scipy_fft = original_scipy
        stage.__kwdefaults__.clear()
        stage.__kwdefaults__.update(original_defaults)
        dispatch.close_eq10_pool()
