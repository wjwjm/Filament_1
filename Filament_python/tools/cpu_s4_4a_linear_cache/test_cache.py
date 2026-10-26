from __future__ import annotations

import json

import numpy as np
import pytest

from .cache import CacheContractError, LinearFactorCache


def _inputs(nt: int = 3):
    omega = np.linspace(-0.2, 0.3, nt, dtype=np.float64)
    kperp2 = np.arange(6, dtype=np.float64).reshape(2, 3) / 10.0
    return omega, kperp2


def _begin(cache, omega, kperp2, **scalars):
    cache.begin_half(
        omega=omega,
        kperp2=kperp2,
        geometry=(len(omega), kperp2.shape[0], kperp2.shape[1]),
        work_dtype=np.complex128,
        output_dtype=np.complex128,
        k0=scalars.get("k0", 1.0),
        omega0=scalars.get("omega0", 2.0),
        beta2=scalars.get("beta2", 0.1),
        denom_floor=scalars.get("denom_floor", 1e-4),
        dz=scalars.get("dz", 0.25),
        precision_strategy=scalars.get("precision_strategy", "baseline_complex64"),
        norm_strategy=scalars.get("norm_strategy", "baseline_complex64"),
        backend=np,
        source_version="toy-source-v1",
        allow_toy_backend=True,
    )


def _run_half(cache, omega, kperp2, *, shift=0.0, fail_at=None, **scalars):
    _begin(cache, omega, kperp2, **scalars)
    for index in range(len(omega)):
        def builder(index=index):
            if fail_at is not None and index == fail_at:
                raise RuntimeError("synthetic builder failure")
            phase = (-0.8 + 0.1 * index) * kperp2 + (0.01 + shift)
            return np.exp(np.asarray(1j, dtype=np.complex128) * phase * 0.25).astype(np.complex128, copy=False)

        cache.factor(index, builder)
    cache.finish_half(success=True)


def test_complete_half_miss_then_hit_and_raw_bit_restore():
    cache = LinearFactorCache(owner_identity="test", algorithm_source_identity="toy")
    omega, kperp2 = _inputs()
    _run_half(cache, omega, kperp2)
    first = cache.snapshot()
    assert first["stats"]["key_misses"] == 1
    assert first["stats"]["factor_requests"] == len(omega)
    assert first["stats"]["plane_builds"] == len(omega)
    assert first["stats"]["misses"] == len(omega)
    assert first["stats"]["hits"] == 0
    _run_half(cache, omega, kperp2)
    hit = cache.snapshot()
    assert hit["stats"]["hits"] == len(omega)
    omega[1] = np.nextafter(omega[1], np.inf)
    _run_half(cache, omega, kperp2)
    assert cache.snapshot()["stats"]["key_misses"] == 2
    omega[1] = np.linspace(-0.2, 0.3, len(omega), dtype=np.float64)[1]
    _run_half(cache, omega, kperp2)
    assert cache.snapshot()["stats"]["hits"] >= 2 * len(omega)


@pytest.mark.parametrize("field", ["k0", "omega0", "beta2", "denom_floor", "dz"])
def test_every_converted_scalar_bit_is_keyed(field):
    cache = LinearFactorCache(owner_identity="test", algorithm_source_identity="toy")
    omega, kperp2 = _inputs()
    _run_half(cache, omega, kperp2)
    values = {"k0": 1.0, "omega0": 2.0, "beta2": 0.1, "denom_floor": 1e-4, "dz": 0.25}
    values[field] = np.nextafter(values[field], np.inf)
    _run_half(cache, omega, kperp2, **values)
    assert cache.snapshot()["stats"]["key_misses"] == 2


def test_partial_failure_does_not_publish_and_third_key_evicts():
    cache = LinearFactorCache(owner_identity="test", algorithm_source_identity="toy", max_tables=2)
    omega, kperp2 = _inputs()
    _begin(cache, omega, kperp2)
    cache.factor(0, lambda: np.ones_like(kperp2, dtype=np.complex128))
    with pytest.raises(CacheContractError):
        cache.finish_half(success=True)
    failed = cache.snapshot()
    assert failed["stats"]["entries"] == 0
    assert failed["stats"]["invalidations"] == 1
    with pytest.raises(RuntimeError, match="synthetic builder failure"):
        _run_half(cache, omega, kperp2, fail_at=1)
    assert cache.snapshot()["stats"]["entries"] == 0
    for shift in range(3):
        _run_half(cache, omega + shift, kperp2, shift=float(shift))
    snapshot = cache.snapshot()
    assert snapshot["stats"]["entries"] == 2
    assert snapshot["stats"]["evictions"] >= 1
    assert snapshot["events"] and len(snapshot["events"]) <= 64
    json.dumps(snapshot, allow_nan=False)


def test_in_place_and_layout_contracts_and_isolation():
    cache = LinearFactorCache(owner_identity="test", algorithm_source_identity="toy")
    omega, kperp2 = _inputs()
    _run_half(cache, omega, kperp2)
    same_content = omega.copy()
    _run_half(cache, same_content, kperp2.copy())
    assert cache.snapshot()["stats"]["hits"] >= len(omega)
    with pytest.raises(CacheContractError, match="positive-stride native C"):
        _begin(cache, omega, np.asfortranarray(kperp2))
    with pytest.raises(CacheContractError, match="positive-stride native C"):
        _begin(cache, omega[::-1], kperp2)
    with pytest.raises(CacheContractError, match="native float64"):
        _begin(cache, omega.astype(np.float32), kperp2)
    with pytest.raises(CacheContractError, match="native float64"):
        _begin(cache, omega, kperp2.astype(object))
    isolated = LinearFactorCache(owner_identity="isolated", algorithm_source_identity="toy")
    assert isolated.snapshot()["stats"]["factor_requests"] == 0
    snapshot = cache.snapshot()
    def metadata_only(value):
        if isinstance(value, dict):
            return all(isinstance(key, str) and metadata_only(item) for key,item in value.items())
        if isinstance(value, (list,tuple)):
            return all(metadata_only(item) for item in value)
        return value is None or type(value) in (str,int,float,bool)
    assert metadata_only(snapshot)
