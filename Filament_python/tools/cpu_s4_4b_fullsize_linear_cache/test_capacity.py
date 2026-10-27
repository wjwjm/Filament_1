from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from . import contract
from .cache import CacheContractError, FullsizeLinearFactorCache


SHAPE = contract.MICRO_SHAPE_32


def _arrays(*, omega_shift: float = 0.0, k_shift: float = 0.0):
    omega = np.linspace(-0.2, 0.3, SHAPE[0], dtype=np.float64) + omega_shift
    kperp2 = np.arange(SHAPE[1] * SHAPE[2], dtype=np.float64).reshape(SHAPE[1], SHAPE[2])
    kperp2 = kperp2 / 10.0 + k_shift
    return omega, kperp2


def _begin(cache, omega, kperp2, *, dz: float = 0.25, beta2: float = 0.1):
    cache.begin_half(
        omega=omega,
        kperp2=kperp2,
        geometry=SHAPE,
        work_dtype=np.complex128,
        output_dtype=np.complex128,
        k0=1.0,
        omega0=2.0,
        beta2=beta2,
        denom_floor=1e-4,
        dz=dz,
        precision_strategy="baseline_complex64",
        norm_strategy="baseline_complex64",
        backend=np,
        source_version="toy-source-v1",
        allow_toy_backend=True,
    )


def _run(cache, *, omega_shift=0.0, k_shift=0.0, dz=0.25, beta2=0.1, fail_at=None, view=False):
    omega, kperp2 = _arrays(omega_shift=omega_shift, k_shift=k_shift)
    _begin(cache, omega, kperp2, dz=dz, beta2=beta2)
    for index in range(SHAPE[0]):
        def builder(index=index):
            if fail_at is not None and index == fail_at:
                raise RuntimeError("synthetic builder failure")
            array = np.ones((SHAPE[1], SHAPE[2]), dtype=np.complex128)
            if view:
                array = array.view()
                array.setflags(write=False)
            return array

        cache.factor(index, builder)
    cache.finish_half(success=True)


def _cache(**kwargs):
    return FullsizeLinearFactorCache(
        owner_identity="test-owner",
        algorithm_source_identity="toy-source",
        resource_scope=contract.LOCAL_ENGINEERING_SCOPE,
        configured_shape=SHAPE,
        **kwargs,
    )


def test_fullsize_capacity_arithmetic_is_allocation_free():
    assert contract.payload_bytes(contract.FULL_SHAPE) == contract.FULL_PAYLOAD_BYTES
    assert contract.key_guard_bytes(contract.FULL_SHAPE) == (384 + 351 * 301) * 8
    assert contract.persistent_reserved_bytes(contract.FULL_SHAPE, 2) < contract.FULLSIZE_MAX_BYTES
    assert contract.persistent_reserved_bytes(contract.FULL_SHAPE, 1) < contract.persistent_reserved_bytes(
        contract.FULL_SHAPE, 2
    )
    assert contract.SMALLGRID_DEFAULT_MAX_BYTES == 256 * 1024 * 1024
    assert contract.SMALLGRID_DEFAULT_MAX_TABLES == 2
    assert _cache().max_bytes == contract.SMALLGRID_DEFAULT_MAX_BYTES
    metadata = FullsizeLinearFactorCache(owner_identity='metadata',algorithm_source_identity='metadata',configured_shape=contract.FULL_SHAPE)
    assert metadata.max_bytes == contract.FULLSIZE_MAX_BYTES
    assert metadata.snapshot()['stats']['cache_bytes'] == 0


def test_micro_miss_hit_exact_keys_and_in_place_changes():
    cache = _cache()
    _run(cache)
    _run(cache)
    omega, kperp2 = _arrays()
    omega[1] = np.nextafter(omega[1], np.inf)
    _begin(cache, omega, kperp2)
    for index in range(SHAPE[0]):
        cache.factor(index, lambda: np.ones((SHAPE[1], SHAPE[2]), dtype=np.complex128))
    cache.finish_half(success=True)
    _run(cache, k_shift=np.nextafter(0.0, np.inf))
    _run(cache, dz=np.nextafter(0.25, np.inf))
    _run(cache, beta2=np.nextafter(0.1, np.inf))
    snapshot = cache.snapshot()
    assert snapshot["stats"]["key_misses"] >= 5
    assert snapshot["stats"]["hits"] >= SHAPE[0]
    assert snapshot["owned_object_audit"]["status"] == "PASS"


def test_partial_failure_external_view_rejected_and_recovery():
    cache = _cache()
    _begin(cache, *_arrays())
    cache.factor(0, lambda: np.ones((SHAPE[1], SHAPE[2]), dtype=np.complex128))
    with pytest.raises(CacheContractError, match="incomplete"):
        cache.finish_half(success=True)
    assert cache.snapshot()["stats"]["entries"] == 0
    with pytest.raises(RuntimeError, match="synthetic builder failure"):
        _run(cache, fail_at=2)
    assert cache.snapshot()["stats"]["entries"] == 0
    with pytest.raises(CacheContractError, match="external view"):
        _run(cache, view=True)
    assert cache.snapshot()["stats"]["entries"] == 0
    _run(cache)
    assert cache.snapshot()["stats"]["entries"] == 1


def test_lru_third_key_rebuild_and_old_table_release():
    cache = _cache()
    _run(cache, omega_shift=0.0)
    first_entry = next(iter(cache._entries.values()))
    old_reference = weakref.ref(first_entry.table[0])
    del first_entry
    _run(cache, omega_shift=1.0)
    _run(cache, omega_shift=2.0)
    gc.collect()
    assert cache.snapshot()["stats"]["entries"] == 2
    assert cache.snapshot()["stats"]["evictions"] >= 1
    assert old_reference() is None
    before = cache.snapshot()["stats"]["key_misses"]
    _run(cache, omega_shift=0.0)
    assert cache.snapshot()["stats"]["key_misses"] == before + 1


def test_budget_scope_dtype_and_reserve_fail_closed_then_recover():
    cache = _cache()
    with pytest.raises(CacheContractError, match="capacity"):
        cache._evict_for(contract.FULLSIZE_MAX_BYTES, 0)
    with pytest.raises(CacheContractError, match="unknown CPU-S4-4B resource scope"):
        FullsizeLinearFactorCache(
            owner_identity="x",
            algorithm_source_identity="x",
            resource_scope="UNKNOWN",
        )
    with pytest.raises(CacheContractError, match="unsupported CPU-S4-4B shape"):
        bad = FullsizeLinearFactorCache(
            owner_identity="x",
            algorithm_source_identity="x",
            resource_scope=contract.LOCAL_ENGINEERING_SCOPE,
        )
        omega = np.ones(3, dtype=np.float64)
        kperp = np.ones((2, 3), dtype=np.float64)
        bad.begin_half(
            omega=omega,
            kperp2=kperp,
            geometry=(3, 2, 3),
            work_dtype=np.complex128,
            output_dtype=np.complex128,
            k0=1.0,
            omega0=2.0,
            beta2=0.1,
            denom_floor=1e-4,
            dz=0.25,
            precision_strategy="x",
            norm_strategy="x",
            backend=np,
            source_version="x",
            allow_toy_backend=True,
        )
    with pytest.raises(CacheContractError, match="native complex128"):
        cache.begin_half(
            omega=np.ones(SHAPE[0], dtype=np.float64),
            kperp2=np.ones((SHAPE[1], SHAPE[2]), dtype=np.float64),
            geometry=SHAPE,
            work_dtype=np.complex64,
            output_dtype=np.complex128,
            k0=1.0,
            omega0=2.0,
            beta2=0.1,
            denom_floor=1e-4,
            dz=0.25,
            precision_strategy="x",
            norm_strategy="x",
            backend=np,
            source_version="x",
            allow_toy_backend=True,
        )
    _run(cache)
    cache._events.append("x" * 5_000_000)
    with pytest.raises(CacheContractError, match="underestimated reserve"):
        _run(cache, omega_shift=3.0)
    assert cache.snapshot()["stats"]["entries"] == 0
    cache._events = [event for event in cache._events if not isinstance(event, str)]
    _run(cache)
    assert cache.snapshot()["stats"]["entries"] == 1


def test_cache_instances_are_isolated():
    left = _cache()
    right = _cache()
    _run(left)
    _run(right)
    assert left.snapshot()["stats"]["key_misses"] == 1
    assert right.snapshot()["stats"]["key_misses"] == 1
    assert left.owner_identity == right.owner_identity
    assert left is not right
