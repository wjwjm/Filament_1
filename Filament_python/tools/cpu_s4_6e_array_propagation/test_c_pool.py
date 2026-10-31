"""Focused C/P24 worker-pool checks."""

from __future__ import annotations

import threading
import concurrent.futures

import numpy as np
import pytest

from . import c_pool as pool_module
from .c_pool import CPool, CPoolError, SafeWorkers


def _arrays(shape=(8, 3, 2)):
    rng = np.random.default_rng(104)
    E = np.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape), dtype=np.complex128, order="C")
    phase = np.asarray(rng.normal(size=shape), dtype=np.float64, order="C")
    alpha = np.asarray(rng.normal(size=shape), dtype=np.float64, order="C")
    return E, phase, alpha


def test_s46e_c_pool_two_rounds_exact_and_bounded_records():
    E, phase, alpha = _arrays()
    before = E.copy()
    pool = CPool(1)
    pool.validate_domain(E, phase, alpha)
    def phase_worker(lo, hi):
        E[lo:hi] *= np.exp(1j * phase[lo:hi]).astype(np.complex128, copy=False)

    def absorption_worker(lo, hi):
        E[lo:hi] *= np.exp((-0.5 * alpha[lo:hi] * float(0.02))).astype(np.complex128, copy=False)

    pool.phase_update(E, phase_worker, phase=phase, ctype=np.dtype(np.complex128))
    pool.absorption_update(E, absorption_worker, alpha=alpha, ctype=np.dtype(np.complex128))
    expected = before * np.exp(1j * phase).astype(np.complex128, copy=False)
    expected *= np.exp(-0.5 * alpha * 0.02).astype(np.complex128, copy=False)
    assert E is not before
    assert np.array_equal(E, expected)
    snap = pool.snapshot()
    assert snap["operation_counts"] == {"C_PHASE": 1, "C_ABSORPTION": 1}
    assert snap["pending_scalar_records"] == 2
    assert snap["array_values_retained"] is False
    records = pool.take_records()
    assert len(records) == 2
    assert all(isinstance(item["items"], int) for row in records for item in row["workers"])
    pool.close()
    pool.close()
    assert pool.snapshot()["active_thread_ids"] == []
    with pytest.raises(CPoolError, match="RETIRED"):
        pool.phase_update(E, phase_worker, phase=phase, ctype=np.dtype(np.complex128))


def test_s46e_c_pool_rejects_alias_and_unsupported_domain_before_write():
    E, phase, alpha = _arrays()
    pool = CPool(1)
    original = E.copy()
    alias = E.view(np.float64).reshape(E.shape + (2,))[..., 0]
    with pytest.raises(CPoolError, match="ALIAS"):
        pool.validate_domain(E, alias, alpha)
    assert np.array_equal(E, original)
    with pytest.raises(CPoolError, match="ALPHA_DOMAIN"):
        pool.validate_domain(E, phase, np.asarray(alpha, dtype=np.float32))
    assert np.array_equal(E, original)
    pool.close()


def test_s46e_c_pool_waits_for_all_tasks_before_retiring_on_error():
    E, _, _ = _arrays((16, 2, 2))
    pool = CPool(8)
    completed: list[int] = []
    lock = threading.Lock()

    def operation(lo: int, hi: int) -> None:
        if lo == 0:
            raise RuntimeError("synthetic-task-error")
        with lock:
            completed.append(int(lo))

    with pytest.raises(RuntimeError, match="synthetic-task-error"):
        pool._run("C_PHASE", E, operation)
    assert len(completed) == 7
    assert pool.closed is True
    assert pool.snapshot()["active_thread_ids"] == []


def test_s46e_c_pool_rejects_missing_ast_worker():
    E, phase, alpha = _arrays()
    pool = CPool(1)
    with pytest.raises(CPoolError, match="WORKER"):
        pool.phase_update(E, None, phase=phase, ctype=np.dtype(np.complex128))
    pool.close()


def test_s46e_c_pool_constructor_failure_closes_partial_executor(monkeypatch):
    executor_box = {}

    def fail_init(self, count):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor_box["executor"] = executor
        self.executor = executor
        raise RuntimeError("synthetic-constructor-error")

    monkeypatch.setattr(pool_module.Workers, "__init__", fail_init)
    with pytest.raises(RuntimeError, match="synthetic-constructor-error"):
        SafeWorkers(1)
    assert executor_box["executor"]._shutdown is True
