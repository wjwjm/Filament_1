"""Low-cost A/P24 contract tests; no propagation or HPC access."""

from __future__ import annotations

import ast
import threading
from pathlib import Path

import numpy as np
import pytest

from .candidate import _owned_source_identity
from .instrument import HELPER_NAMES, _helpers, instrument_intensity_stage
from .pool import IntensityPool, PoolError, SafeWorkers


def _stage_node() -> ast.FunctionDef:
    source = Path(__file__).resolve().parents[2] / "KHz_filament" / "raman.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "isaacs_raman_stage")


def test_s46e_ast_replaces_three_pointwise_segments_and_restores_source() -> None:
    receipt = instrument_intensity_stage(_stage_node())
    assert receipt["replacement_counts"] == {"intensity": 1, "abs": 1, "square": 1, "scale": 1}
    assert receipt["stripped_ast_sha256"] == receipt["source_ast_sha256"]
    assert receipt["pointwise_segments"] == ["complex_absolute", "square", "coefficient_scale"]


def test_s46e_source_ast_tamper_and_return_energy_branch_are_rejected_or_untouched() -> None:
    stage = _stage_node()
    intensity = next(node for node in ast.walk(stage) if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "intensity")
    abs_call = intensity.value.func.value.right.left
    abs_call.func.attr = "real"
    with pytest.raises(Exception, match="ABS_SHAPE"):
        instrument_intensity_stage(stage)
    stage = _stage_node()
    instrument_intensity_stage(stage)
    branch = next(node for node in stage.body if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "return_energy")
    helper = HELPER_NAMES["intensity"]
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == helper for node in ast.walk(branch))


def _compile_stage(stage: ast.FunctionDef, namespace: dict[str, object]):
    module = ast.fix_missing_locations(ast.Module(body=[stage], type_ignores=[]))
    local: dict[str, object] = {}
    exec(compile(module, "<s46e-source-bound-stage-test>", "exec"), namespace, local)
    return local[stage.name]


def _assert_stage_results_equal(reference: dict[str, object], candidate: dict[str, object]) -> None:
    assert reference.keys() == candidate.keys()
    for name in reference:
        if name == "walltime_s":
            continue
        left, right = reference[name], candidate[name]
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right


def test_s46e_compiled_stage_return_energy_false_and_zero_field_exact() -> None:
    import KHz_filament.raman as raman

    pool = IntensityPool(1)
    try:
        reference_node = _stage_node()
        candidate_node = _stage_node()
        instrument_intensity_stage(candidate_node)
        reference = _compile_stage(reference_node, dict(vars(raman)))
        candidate_namespace = dict(vars(raman))
        candidate_namespace.update(_helpers(pool))
        candidate = _compile_stage(candidate_node, candidate_namespace)
        kwargs = dict(
            Omega=np.linspace(-2.0e14, 2.0e14, 8, dtype=np.float64),
            dt=1.0e-15,
            omega0=2.0e15,
            n0=1.0,
            n_R=0.1,
            omega_R=8.0e12,
            Gamma_R=1.0e11,
            method="iir",
            chunk_pixels=4,
            return_response=True,
        )
        rng = np.random.default_rng(20261101)
        field = np.ascontiguousarray(rng.standard_normal((8, 2, 2)) + 1j * rng.standard_normal((8, 2, 2)), dtype=np.complex128)
        reference_input = field.copy()
        candidate_input = field.copy()
        reference_result = reference(reference_input, return_energy=False, **kwargs)
        candidate_result = candidate(candidate_input, return_energy=False, **kwargs)
        _assert_stage_results_equal(reference_result, candidate_result)
        np.testing.assert_array_equal(reference_input, field)
        np.testing.assert_array_equal(candidate_input, field)

        zero = np.zeros((8, 2, 2), dtype=np.complex128)
        reference_zero = reference(zero.copy(), return_energy=True, **kwargs)
        candidate_zero_input = zero.copy()
        candidate_zero = candidate(candidate_zero_input, return_energy=True, **kwargs)
        _assert_stage_results_equal(reference_zero, candidate_zero)
        np.testing.assert_array_equal(candidate_zero_input, zero)
        assert pool.snapshot()["operation_counts"] == {"A_ABS": 2, "A_SQUARE": 2, "A_SCALE": 2}
    finally:
        pool.close()


def test_s46e_helper_exact_input_unchanged_output_lifetime_and_return_dtype() -> None:
    pool = IntensityPool(1)
    helper = _helpers(pool)[HELPER_NAMES["intensity"]]
    field = np.ascontiguousarray((np.arange(4 * 3 * 2, dtype=np.float64).reshape(4, 3, 2) + 1j), dtype=np.complex128)
    frozen = field.copy()
    result = helper(field, 2.5, np.float64)
    expected = (2.5 * np.abs(field) ** 2).astype(np.float64, copy=False)
    np.testing.assert_array_equal(result, expected)
    assert result.flags.c_contiguous and result.flags.owndata and result.dtype == np.dtype("float64")
    assert not np.shares_memory(result, field)
    np.testing.assert_array_equal(field, frozen)
    assert pool.snapshot()["operation_counts"] == {"A_ABS": 1, "A_SQUARE": 1, "A_SCALE": 1}
    assert len(pool.take_records()) == 3
    assert pool.take_records() == []
    pool.close()


def test_s46e_helper_bitwise_matches_original_abs_power_expression() -> None:
    pool = IntensityPool(1)
    helper = _helpers(pool)[HELPER_NAMES["intensity"]]
    rng = np.random.default_rng(46)
    field = np.ascontiguousarray((rng.standard_normal((4, 3, 2)) + 1j * rng.standard_normal((4, 3, 2))), dtype=np.complex128)
    with np.errstate(all="ignore"):
        got = helper(field, 0.5 * 1.0000000000000002, np.float64)
        expected = (0.5 * 1.0000000000000002 * np.abs(field) ** 2).astype(np.float64, copy=False)
    assert np.array_equal(got, expected)
    pool.close()


def test_s46e_zero_nan_inf_policy_is_native_numpy() -> None:
    pool = IntensityPool(1)
    helper = _helpers(pool)[HELPER_NAMES["intensity"]]
    values = np.array([0.0, np.nan, np.inf, -np.inf, 2.0], dtype=np.float64).reshape(5, 1, 1).astype(np.complex128)
    values[1, 0, 0] = np.nan + 1j
    values[2, 0, 0] = np.inf + 1j
    with np.errstate(all="ignore"):
        got = helper(values, 0.5, np.float64)
        expected = (0.5 * np.abs(values) ** 2).astype(np.float64, copy=False)
    np.testing.assert_equal(got, expected)
    pool.close()


def test_s46e_pool_rejects_alias_wrong_backend_and_call_after_close() -> None:
    pool = IntensityPool(1)
    values = np.ones((2, 2, 2), dtype=np.complex128)
    output = np.empty(values.shape, dtype=np.float64)
    magnitude = np.ones(values.shape, dtype=np.float64)
    with pytest.raises(PoolError, match="ALIAS"):
        pool.ufunc("A_SQUARE", np.square, (magnitude,), magnitude)
    with pytest.raises(PoolError, match="BACKEND"):
        pool.ufunc("A_ABS", np.square, (values,), output)
    pool.close()
    with pytest.raises(PoolError, match="RETIRED"):
        pool.ufunc("A_ABS", np.absolute, (values,), output)


def test_s46e_pool_idempotent_close_and_scalar_record_bound() -> None:
    pool = IntensityPool(1, max_records=3)
    source = np.ones((2, 2, 2), dtype=np.complex128)
    magnitude = np.empty(source.shape, dtype=np.float64)
    pool.ufunc("A_ABS", np.absolute, (source,), magnitude)
    live = pool.snapshot()
    assert set(live["active_thread_ids"]) == set(live["initial_thread_ids"])
    pool.close()
    again = pool.close()
    assert again["closed"] is True and again["close_error"] is None
    assert again["lifetime_ns"] >= again["close_ns"]
    assert again["active_thread_ids"] == []
    assert again["pending_scalar_records"] == 1


def test_s46e_task_exception_closes_and_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = IntensityPool(1)
    source = np.ones((2, 2, 2), dtype=np.complex128)
    output = np.empty(source.shape, dtype=np.float64)

    def fail(*args, **kwargs):
        raise RuntimeError("worker failure")

    monkeypatch.setattr(pool._workers, "ufunc", fail)
    with pytest.raises(RuntimeError, match="worker failure"):
        pool.ufunc("A_ABS", np.absolute, (source,), output)
    assert pool.snapshot()["closed"] is True
    with pytest.raises(PoolError, match="RETIRED"):
        pool.ufunc("A_ABS", np.absolute, (source,), output)


def test_s46e_workers_async_task_exception_waits_for_all_tasks() -> None:
    workers = SafeWorkers(8)
    completed = [threading.Event() for _ in range(8)]

    def operation(lo: int, hi: int) -> None:
        try:
            if lo == 0:
                raise RuntimeError("one worker failed")
        finally:
            completed[lo].set()

    try:
        with pytest.raises(RuntimeError, match="one worker failed"):
            workers.run("A_ASYNC_FAILURE", 8, operation)
        assert all(event.is_set() for event in completed)
    finally:
        workers.close()
    assert all(not thread.is_alive() for thread in workers.executor._threads)


def test_s46e_constructor_failure_reclaims_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    import concurrent.futures
    import tools.cpu_s4_6e_array_propagation.pool as pool_module

    box: dict[str, concurrent.futures.ThreadPoolExecutor] = {}

    def fail_init(self: object, count: int) -> None:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        box["executor"] = executor
        self.executor = executor
        raise RuntimeError("constructor failure")

    monkeypatch.setattr(pool_module.Workers, "__init__", fail_init)
    with pytest.raises(RuntimeError, match="constructor failure"):
        SafeWorkers(1)
    assert box["executor"]._shutdown is True


def test_s46e_null_pool_close_is_idempotent() -> None:
    from .pool import NullIntensityPool

    pool = NullIntensityPool()
    assert pool.snapshot()["closed"] is False
    pool.close()
    pool.close()
    assert pool.snapshot()["closed"] is True


def test_s46e_owned_source_identity_includes_reused_workers() -> None:
    identity = _owned_source_identity()
    chains = identity["files"]["chains.py"]
    assert chains["classification"] == "tracked_text"
    assert chains["hash_scope"] == "git_blob_oid+canonical_lf_sha256"
    assert len(chains["git_blob_oid"]) == 40 and len(chains["canonical_lf_sha256"]) == 64


def test_s46e_xp_abs_binding_tamper_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("numba", reason="local-test environment lacks numba")
    from .candidate import build_role, verify_bindings

    _set_local_factory_environment(monkeypatch)
    import numba

    numba.config.NUMBA_NUM_THREADS = 1
    numba.set_num_threads(1)
    built = build_role("PURE_OFF_SMALLGRID", capture_root=tmp_path / "A", role="A", local_engineering=True, pool_override=1)
    dispatch = built[1]
    stage = dispatch.cpu_s4_6e_state._internal_stage_inner
    xp = stage.__globals__["xp"]
    original_abs = xp.abs
    try:
        xp.abs = object()
        with pytest.raises(Exception, match="ABS_BINDING"):
            verify_bindings(dispatch)
    finally:
        xp.abs = original_abs
        dispatch.close_intensity_pool()


def _set_local_factory_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = Path(__file__).resolve().parents[3]
    for name in ("S43G2_SOURCE_ROOT", "S43F_SOURCE_ROOT", "S43C_SOURCE_ROOT"):
        monkeypatch.setenv(name, str(repo))
    monkeypatch.setenv("UPPE_USE_GPU", "0")
    monkeypatch.setenv("NUMBA_NUM_THREADS", "1")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    historical = Path(r"D:\Filament_1\.worktrees\cpu_optimization\reports\CPU_S4_5B_STEADY_COST_20261027_223422\external_data\G2_BOUNDARY_RECEIPT.json")
    if historical.is_file():
        monkeypatch.setenv("CPU_BOUNDARY_HISTORICAL_RECEIPT", str(historical))


@pytest.mark.parametrize("role", ("R", "A"))
def test_s46e_factory_roles_construct_without_propagation(role: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("numba", reason="local-test environment lacks numba")
    from .candidate import build_role, verify_bindings

    _set_local_factory_environment(monkeypatch)
    import numba

    numba.config.NUMBA_NUM_THREADS = 1
    numba.set_num_threads(1)
    built = build_role("PURE_OFF_SMALLGRID", capture_root=tmp_path / role, role=role, local_engineering=True, pool_override=1)
    dispatch = built[1]
    try:
        assert len(built) == 6
        assert dispatch.cpu_s4_6e_role == role
        assert verify_bindings(dispatch)["common_body_instrumentation_bound"] is True
        assert verify_bindings(dispatch)["stage_replaced"] is (role == "A")
    finally:
        dispatch.close_intensity_pool()
