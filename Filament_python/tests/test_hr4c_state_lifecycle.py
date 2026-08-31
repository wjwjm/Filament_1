from __future__ import annotations

import json

import numpy as np
import pytest


DX = DY = 10.0e-6
DT = 1.0e-6
CHI = 21.7e-6
NU = 1.5e-5
N0 = 1.00027


def _store(tmp_path, *, name="run", k=5, shape=(9, 9), dtype=np.float64):
    from KHz_filament.hr4c_state import HR4CThreeFieldStore

    z_edges = np.linspace(0.0, k * 1.0e-4, k + 1)
    return HR4CThreeFieldStore(
        output_path=str(tmp_path / f"{name}.npz"), n_intervals=k, shape=shape,
        dtype=dtype, z_edges=z_edges, dx=DX, dy=DY,
    ), z_edges


def _initial_fields(k, shape, dtype=np.float64):
    yy, xx = np.indices(shape)
    fields = {
        "delta_n": np.zeros((k, *shape), dtype=dtype),
        "vx": np.zeros((k, *shape), dtype=dtype),
        "vy": np.zeros((k, *shape), dtype=dtype),
    }
    for index in range(k):
        cx = shape[1] // 2 + (index % 2)
        cy = shape[0] // 2
        fields["delta_n"][index] = -1.0e-4 * (index + 1) * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / 4.0
        )
    if k > 2:
        fields["vx"][2, shape[0] // 2, shape[1] // 2] = 1.0e-3
    return fields


def _seed(store, fields, *, batch=2):
    store.begin_staging()
    try:
        for start in range(0, store.n_intervals, batch):
            stop = min(start + batch, store.n_intervals)
            store.write_staging_batch(start, {field: values[start:stop] for field, values in fields.items()})
        store.commit_staging({"operation": "test_seed", "batch_intervals": batch})
    except Exception:
        store.abort_staging(reason="test_seed_failure")
        raise


def _evolve(store, *, batch=2, steps=2, failure_injector=None):
    from KHz_filament.hr4c_state import evolve_hr4_full_z

    return evolve_hr4_full_z(
        store, dt_hydro=DT, n_hydro_steps=steps, batch_intervals=batch,
        chi=CHI, nu=NU, n0=N0, failure_injector=failure_injector,
    )


def _reference(fields, *, steps=2):
    from KHz_filament.hr4 import advance_hr4_single_screen

    out = {field: np.empty_like(values) for field, values in fields.items()}
    for index in range(fields["delta_n"].shape[0]):
        result = advance_hr4_single_screen(
            fields["delta_n"][index], fields["vx"][index], fields["vy"][index],
            dx=DX, dy=DY, dt_hydro=DT, chi=CHI, nu=NU, n0=N0, n_steps=steps,
        )
        for field in out:
            out[field][index] = result[field]
    return out


def test_c1_three_field_create_write_close_reopen_and_metadata(tmp_path):
    from KHz_filament.hr4c_state import HR4C_FIELDS, HR4CThreeFieldStore

    store, z_edges = _store(tmp_path, k=3, shape=(5, 6), dtype=np.float32)
    fields = _initial_fields(3, (5, 6), np.float32)
    _seed(store, fields, batch=2)
    metadata = store.authoritative_metadata()
    assert metadata["fields"] == HR4C_FIELDS
    assert metadata["state_shape"] == (3, 5, 6)
    assert len(list(tmp_path.glob("run.hr4c_*_*.npy"))) == 6
    store.close()

    reopened = HR4CThreeFieldStore.open_existing(
        output_path=str(tmp_path / "run.npz"), n_intervals=3, shape=(5, 6),
        dtype=np.float32, z_edges=z_edges, dx=DX, dy=DY,
    )
    loaded = reopened.read_authoritative_batch(0, 3)
    for field in HR4C_FIELDS:
        np.testing.assert_array_equal(loaded[field], fields[field])
    reopened.close()


def test_c2_legacy_delta_n_initialization_preserves_source_and_zeroes_velocity(tmp_path):
    from KHz_filament.hr4c_state import HR4CThreeFieldStore

    k, shape = 4, (7, 7)
    legacy_path = tmp_path / "legacy.npy"
    legacy = np.lib.format.open_memmap(legacy_path, mode="w+", dtype=np.float32, shape=(k, *shape))
    source = _initial_fields(k, shape, np.float32)["delta_n"]
    legacy[:] = source
    legacy.flush()
    before = np.array(legacy, copy=True)
    z_edges = np.linspace(0.0, k * 1e-4, k + 1)
    store = HR4CThreeFieldStore.initialize_from_legacy_delta_n(
        output_path=str(tmp_path / "migrated.npz"), legacy_delta_n_path=str(legacy_path),
        n_intervals=k, shape=shape, dtype=np.float32, z_edges=z_edges, dx=DX, dy=DY,
        batch_intervals=2,
    )
    migrated = store.read_authoritative_batch(0, k)
    np.testing.assert_array_equal(migrated["delta_n"], source)
    np.testing.assert_array_equal(migrated["vx"], np.zeros_like(source))
    np.testing.assert_array_equal(migrated["vy"], np.zeros_like(source))
    np.testing.assert_array_equal(np.load(legacy_path, mmap_mode="r"), before)
    assert store.manifest["last_evolution"]["operation"] == "legacy_delta_n_initialization"
    store.close()


def test_c3_batch_size_equivalence_and_c4_full_memory_reference(tmp_path):
    fields = _initial_fields(5, (9, 9))
    first, _ = _store(tmp_path, name="batch1")
    second, _ = _store(tmp_path, name="batch2")
    _seed(first, fields, batch=1)
    _seed(second, fields, batch=4)
    _evolve(first, batch=1, steps=3)
    _evolve(second, batch=4, steps=3)
    expected = _reference(fields, steps=3)
    for field in expected:
        np.testing.assert_allclose(first.read_authoritative_batch(0, 5)[field], expected[field], rtol=2e-12, atol=1e-18)
        np.testing.assert_allclose(second.read_authoritative_batch(0, 5)[field], expected[field], rtol=2e-12, atol=1e-18)
    first.close()
    second.close()


def test_c5_last_partial_batch_and_c6_z_screen_independence(tmp_path):
    store, _ = _store(tmp_path, k=10, shape=(7, 7))
    fields = {field: np.zeros((10, 7, 7), dtype=np.float64) for field in ("delta_n", "vx", "vy")}
    fields["delta_n"][6, 3, 3] = -1.0e-4
    _seed(store, fields, batch=4)
    summary = _evolve(store, batch=4, steps=2)
    assert summary["n_batches"] == 3
    final = store.read_authoritative_batch(0, 10)
    assert final["vy"][6, 3, 3] > 0.0
    for field in final:
        np.testing.assert_array_equal(final[field][:6], np.zeros_like(final[field][:6]))
        np.testing.assert_array_equal(final[field][7:], np.zeros_like(final[field][7:]))
    store.close()


def test_c7_mid_batch_failure_preserves_old_generation_and_c9_reopen_discards_staging(tmp_path):
    from KHz_filament.hr4c_state import HR4CThreeFieldStore

    store, z_edges = _store(tmp_path, k=5)
    fields = _initial_fields(5, (9, 9))
    _seed(store, fields)
    before = {field: np.array(values, copy=True) for field, values in store.read_authoritative_batch(0, 5).items()}

    def fail_on_second_batch(start, stop):
        if start == 2:
            raise RuntimeError("injected batch failure")

    with pytest.raises(RuntimeError, match="injected"):
        _evolve(store, batch=2, failure_injector=fail_on_second_batch)
    assert store.manifest["transaction_status"] == "committed"
    assert store.manifest["generation"] == 1
    assert store.manifest["last_abort"] == "RuntimeError"
    for field in before:
        np.testing.assert_array_equal(store.read_authoritative_batch(0, 5)[field], before[field])

    store.begin_staging()
    store.write_staging_field_batch("delta_n", 0, fields["delta_n"][:2])
    store.close()
    reopened = HR4CThreeFieldStore.open_existing(
        output_path=str(tmp_path / "run.npz"), n_intervals=5, shape=(9, 9),
        dtype=np.float64, z_edges=z_edges, dx=DX, dy=DY,
    )
    assert reopened.manifest["transaction_status"] == "committed"
    assert reopened.manifest["generation"] == 1
    assert reopened.manifest["last_abort"] == "reopen_discarded_incomplete_staging"
    for field in before:
        np.testing.assert_array_equal(reopened.read_authoritative_batch(0, 5)[field], before[field])
    reopened.close()


def test_c8_incomplete_staging_and_c10_nonfinite_staging_fail_closed(tmp_path):
    store, _ = _store(tmp_path, k=3, shape=(5, 5))
    fields = _initial_fields(3, (5, 5))
    store.begin_staging()
    store.write_staging_field_batch("delta_n", 0, fields["delta_n"])
    with pytest.raises(ValueError, match="completeness"):
        store.commit_staging({"batch_intervals": 3})
    assert store.manifest["transaction_status"] == "committed"
    assert store.manifest["last_abort"] == "ValueError"
    assert store.manifest["generation"] == 0

    store.begin_staging()
    for field in ("delta_n", "vx", "vy"):
        store.write_staging_field_batch(field, 0, fields[field])
    store._next["vy"][1, 2, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        store.commit_staging({"batch_intervals": 2})
    assert store.manifest["transaction_status"] == "committed"
    assert store.manifest["last_abort"] == "ValueError"
    assert store.manifest["generation"] == 0
    store.close()


def test_c11_manifest_layout_grid_and_z_ordering_mismatch_rejected(tmp_path):
    from KHz_filament.hr4c_state import HR4CThreeFieldStore

    store, z_edges = _store(tmp_path, k=3)
    store.close()
    with pytest.raises(ValueError, match="mismatch"):
        HR4CThreeFieldStore.open_existing(
            output_path=str(tmp_path / "run.npz"), n_intervals=4, shape=(9, 9),
            dtype=np.float64, z_edges=np.linspace(0.0, 4e-4, 5), dx=DX, dy=DY,
        )
    with pytest.raises(ValueError, match="mismatch"):
        HR4CThreeFieldStore.open_existing(
            output_path=str(tmp_path / "run.npz"), n_intervals=3, shape=(9, 9),
            dtype=np.float64, z_edges=z_edges, dx=2.0 * DX, dy=DY,
        )
    manifest_path = tmp_path / "run.hr4c_state_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["z_ordering"] = "reversed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="z_ordering"):
        HR4CThreeFieldStore.open_existing(
            output_path=str(tmp_path / "run.npz"), n_intervals=3, shape=(9, 9),
            dtype=np.float64, z_edges=z_edges, dx=DX, dy=DY,
        )


def test_c12_bounded_memory_accounting_is_independent_of_k_and_c13_has_no_hydro_history(tmp_path):
    from KHz_filament.hr4c_state import estimate_hr4c_working_set_bytes

    assert estimate_hr4c_working_set_bytes(batch_intervals=2, shape=(9, 9), dtype=np.float64) == (
        6 * 2 + 12
    ) * 9 * 9 * 8
    small, _ = _store(tmp_path, name="small", k=3)
    large, _ = _store(tmp_path, name="large", k=17)
    _seed(small, _initial_fields(3, (9, 9)))
    _seed(large, _initial_fields(17, (9, 9)))
    small_summary = _evolve(small, batch=2, steps=1)
    observed_reads, observed_writes = [], []
    read_batch = large.read_authoritative_batch
    write_batch = large.write_staging_batch

    def record_read(start, stop):
        observed_reads.append((start, stop))
        return read_batch(start, stop)

    def record_write(start, values):
        observed_writes.append((start, start + values["delta_n"].shape[0]))
        return write_batch(start, values)

    large.read_authoritative_batch = record_read
    large.write_staging_batch = record_write
    large_summary = _evolve(large, batch=2, steps=5)
    assert small_summary["working_set_estimate_bytes"] == large_summary["working_set_estimate_bytes"]
    assert small_summary["slow_time_history_stored"] is False
    assert large_summary["slow_time_history_stored"] is False
    assert large_summary["n_intervals"] > small_summary["n_intervals"]
    assert observed_reads == observed_writes == [(start, min(start + 2, 17)) for start in range(0, 17, 2)]
    small.close()
    large.close()


def test_c14_end_to_end_small_full_z_reopen_matches_screen_reference(tmp_path):
    from KHz_filament.hr4c_state import HR4CThreeFieldStore

    fields = _initial_fields(4, (9, 9))
    fields["delta_n"][0].fill(0.0)
    fields["vy"][3, 4, 4] = 2.0e-4
    expected = _reference(fields, steps=2)
    store, z_edges = _store(tmp_path, k=4)
    _seed(store, fields, batch=2)
    summary = _evolve(store, batch=2, steps=2)
    assert summary["generation"] == 2
    assert summary["source_generation"] == 1
    assert summary["full_z_materialized"] is False
    assert summary["bytes_read"] == summary["bytes_written"] == 3 * fields["delta_n"].nbytes
    store.close()

    reopened = HR4CThreeFieldStore.open_existing(
        output_path=str(tmp_path / "run.npz"), n_intervals=4, shape=(9, 9),
        dtype=np.float64, z_edges=z_edges, dx=DX, dy=DY,
    )
    for field in expected:
        np.testing.assert_allclose(reopened.read_authoritative_batch(0, 4)[field], expected[field], rtol=2e-12, atol=1e-18)
    manifest = json.loads((tmp_path / "run.hr4c_state_manifest.json").read_text(encoding="utf-8"))
    assert manifest["transaction_status"] == "committed"
    assert manifest["last_evolution"]["z_scan_order"] == "z_batch_outer_then_screen_then_all_hydro_steps"
    reopened.close()
