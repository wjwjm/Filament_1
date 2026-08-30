from __future__ import annotations

import numpy as np
import pytest


def _axes(nx: int = 32, length_m: float = 4.0e-3):
    from KHz_filament.grids import make_axes

    return make_axes(nx, nx, 8, length_m, length_m, 80e-15)


def _gaussian_stack(k: int, axes, dtype):
    x, y = np.asarray(axes.x), np.asarray(axes.y)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    radius_squared = xx**2 + yy**2
    return np.stack([
        -float(index + 1) * 1.0e-4 * np.exp(-radius_squared / (3.0e-4)**2)
        for index in range(k)
    ]).astype(dtype)


def _store_with_current(tmp_path, *, k: int, axes, dtype):
    from KHz_filament.slow_state_pingpong import PingPongSlowStateStore

    store = PingPongSlowStateStore(
        output_path=str(tmp_path / "run.npz"), n_intervals=k,
        shape=(axes.y.size, axes.x.size), dtype=dtype,
    )
    initial = _gaussian_stack(k, axes, dtype)
    for index, state in enumerate(initial):
        store.update_current_interval(index, state)
    return store, initial


def test_cb1_pingpong_layout_nominal_estimates_and_preflight(tmp_path, monkeypatch):
    from KHz_filament.slow_state_pingpong import (
        PingPongSlowStateStore,
        estimate_pingpong_bytes,
        estimate_state_bytes,
    )
    import KHz_filament.slow_state_pingpong as pingpong

    store = PingPongSlowStateStore(
        output_path=str(tmp_path / "run.npz"), n_intervals=3, shape=(4, 5), dtype=np.float32,
    )
    assert store.current_path != store.next_path
    assert store.current_path.name.endswith(".hr3c_delta_n_th_current.npy")
    assert store.next_path.name.endswith(".hr3c_delta_n_th_next.npy")
    assert isinstance(store.read_current_batch(0, 1), np.memmap)
    assert store.state_shape == (3, 4, 5)
    assert store.dtype == np.dtype(np.float32)
    assert estimate_state_bytes(n_intervals=16000, shape=(512, 512), dtype=np.float32) == 16_777_216_000
    assert estimate_pingpong_bytes(n_intervals=16000, shape=(512, 512), dtype=np.float32) == 33_554_432_000
    assert estimate_state_bytes(n_intervals=16000, shape=(512, 512), dtype=np.float32) / 1024**3 == pytest.approx(15.625)
    assert estimate_pingpong_bytes(n_intervals=16000, shape=(512, 512), dtype=np.float32) / 1024**3 == pytest.approx(31.25)
    store.close()

    class NoSpace:
        free = 0

    monkeypatch.setattr(pingpong.shutil, "disk_usage", lambda _: NoSpace())
    blocked = tmp_path / "blocked.npz"
    with pytest.raises(OSError, match="preflight"):
        PingPongSlowStateStore(
            output_path=str(blocked), n_intervals=1, shape=(4, 4), dtype=np.float32,
        )
    assert not blocked.with_suffix("").with_name("blocked.hr3c_delta_n_th_current.npy").exists()
    assert not blocked.with_suffix("").with_name("blocked.hr3c_delta_n_th_next.npy").exists()


@pytest.mark.parametrize("dtype,rtol", [(np.float32, 3e-6), (np.float64, 2e-12)])
def test_cb3_batch_matches_hr3ca_slice_reference_and_partial_final_batch(dtype, rtol):
    from KHz_filament.config import HeatConfig
    from KHz_filament.slow_diffusion import diffuse_batch_2d, diffuse_interval_2d

    axes = _axes()
    heat = HeatConfig()
    states = _gaussian_stack(5, axes, dtype)
    evolved = np.asarray(diffuse_batch_2d(
        states, kperp2=axes.kperp2, D_th=heat.D_th, f_rep=heat.f_rep,
    ))
    for index, state in enumerate(states):
        reference = np.asarray(diffuse_interval_2d(
            state, kperp2=axes.kperp2, D_th=heat.D_th, f_rep=heat.f_rep,
        ))
        np.testing.assert_allclose(evolved[index], reference, rtol=rtol, atol=1e-12)


def test_cb2_cb4_cb7_streaming_keeps_current_immutable_and_reopens_next(tmp_path):
    from KHz_filament.config import HeatConfig
    from KHz_filament.slow_diffusion import diffuse_interval_2d
    from KHz_filament.slow_state_pingpong import diffuse_current_to_next

    axes = _axes()
    heat = HeatConfig()
    store, initial = _store_with_current(tmp_path, k=5, axes=axes, dtype=np.float32)
    current_before = np.array(store.read_current_batch(0, 5), copy=True)
    summary = diffuse_current_to_next(
        store, kperp2=axes.kperp2, D_th=heat.D_th, f_rep=heat.f_rep,
        batch_intervals=2,
    )
    assert summary["complete"]
    assert summary["next_authoritative"] is False
    assert summary["n_batches"] == 3
    assert summary["bytes_read"] == summary["bytes_written"] == initial.nbytes
    np.testing.assert_array_equal(store.read_current_batch(0, 5), current_before)
    store.close()

    reopened_next = np.load(tmp_path / "run.hr3c_delta_n_th_next.npy", mmap_mode="r")
    assert reopened_next.shape == initial.shape
    for index, state in enumerate(initial):
        reference = np.asarray(diffuse_interval_2d(
            state, kperp2=axes.kperp2, D_th=heat.D_th, f_rep=heat.f_rep,
        ))
        np.testing.assert_allclose(reopened_next[index], reference, rtol=3e-6, atol=1e-12)


def test_cb5_edge_failure_leaves_current_intact_and_next_unpromoted(tmp_path):
    from KHz_filament.config import HeatConfig
    from KHz_filament.slow_diffusion import EdgeContaminationError
    from KHz_filament.slow_state_pingpong import diffuse_current_to_next

    axes = _axes()
    heat = HeatConfig()
    store, initial = _store_with_current(tmp_path, k=4, axes=axes, dtype=np.float64)
    store.update_current_interval(3, np.full((32, 32), -1.0e-4))
    current_before = np.array(store.read_current_batch(0, 4), copy=True)
    with pytest.raises(EdgeContaminationError) as failed:
        diffuse_current_to_next(
            store, kperp2=axes.kperp2, D_th=heat.D_th, f_rep=heat.f_rep,
            batch_intervals=2,
        )
    assert failed.value.interval_index == 3
    assert failed.value.R_edge > failed.value.threshold
    assert not store.next_complete
    assert not store.next_valid
    np.testing.assert_array_equal(store.read_current_batch(0, 4), current_before)
    assert np.any(np.asarray(store._next[:2]) != 0.0)
    store.close()


def test_cb6_streaming_reads_only_batches_and_builds_one_kernel(monkeypatch, tmp_path):
    from KHz_filament import slow_diffusion
    from KHz_filament.config import HeatConfig
    from KHz_filament.slow_state_pingpong import diffuse_current_to_next

    axes = _axes()
    heat = HeatConfig()
    store, _ = _store_with_current(tmp_path, k=5, axes=axes, dtype=np.float32)
    original_read = store.read_current_batch
    original_kernel = slow_diffusion.build_diffusion_kernel
    batch_lengths, kernel_calls = [], []

    def checked_read(start, stop):
        batch_lengths.append(stop - start)
        assert stop - start <= 2
        return original_read(start, stop)

    def counted_kernel(*args, **kwargs):
        kernel_calls.append(1)
        return original_kernel(*args, **kwargs)

    monkeypatch.setattr(store, "read_current_batch", checked_read)
    monkeypatch.setattr(slow_diffusion, "build_diffusion_kernel", counted_kernel)
    summary = diffuse_current_to_next(
        store, kperp2=axes.kperp2, D_th=heat.D_th, f_rep=heat.f_rep,
        batch_intervals=2,
    )
    assert batch_lengths == [2, 2, 1]
    assert len(kernel_calls) == 1
    assert summary["n_batches"] == 3
    store.close()


def test_cb6_rejects_invalid_batch_size_without_touching_store(tmp_path):
    from KHz_filament.config import HeatConfig
    from KHz_filament.slow_state_pingpong import diffuse_current_to_next

    axes = _axes()
    heat = HeatConfig()
    store, initial = _store_with_current(tmp_path, k=2, axes=axes, dtype=np.float32)
    with pytest.raises(ValueError, match="positive"):
        diffuse_current_to_next(
            store, kperp2=axes.kperp2, D_th=heat.D_th, f_rep=heat.f_rep,
            batch_intervals=0,
        )
    np.testing.assert_allclose(store.read_current_batch(0, 2), initial)
    assert not store.next_complete
    store.close()
