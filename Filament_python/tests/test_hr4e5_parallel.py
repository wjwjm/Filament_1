from __future__ import annotations

import numpy as np
import pytest


def _make_store(tmp_path, name: str):
    from KHz_filament.hr4c_state import HR4CThreeFieldStore

    store = HR4CThreeFieldStore(
        output_path=str(tmp_path / f"{name}.npz"), n_intervals=5, shape=(9, 9),
        dtype=np.float64, z_edges=np.arange(6, dtype=np.float64), dx=1.0e-4, dy=1.0e-4,
    )
    coordinate = np.linspace(0.0, np.pi, 9)
    mode = np.sin(coordinate)[:, None] * np.sin(coordinate)[None, :]
    store.begin_staging()
    for index in range(store.n_intervals):
        store.write_staging_batch(index, {
            "delta_n": (-1.0e-5 * (index + 1) * mode)[None, ...],
            "vx": (0.02 * mode)[None, ...],
            "vy": np.zeros((1, 9, 9), dtype=np.float64),
        })
    store.commit_staging({"batch_intervals": 2, "operation": "test_initialization"})
    return store


def _records():
    return [
        {"ordinal": 0, "screen_id": "z000"}, {"ordinal": 1, "screen_id": "z007"},
        {"ordinal": 2, "screen_id": "z042"}, {"ordinal": 3, "screen_id": "z105"},
        {"ordinal": 4, "screen_id": "z999"},
    ]


def test_partition_preserves_order_and_rejects_ambiguity():
    from KHz_filament.hr4e5_parallel import build_screen_blocks, validate_block_manifest

    manifest = build_screen_blocks(_records(), block_size=2, n_workers=2)
    assert [item for block in manifest["blocks"] for item in block["screen_ids"]] == [item["screen_id"] for item in _records()]
    validate_block_manifest(manifest)
    invalid = dict(manifest)
    invalid["blocks"] = [dict(block) for block in manifest["blocks"]]
    invalid["blocks"][1]["screen_records"] = list(invalid["blocks"][0]["screen_records"])
    invalid["blocks"][1]["ordinals"] = list(invalid["blocks"][0]["ordinals"])
    invalid["blocks"][1]["screen_ids"] = list(invalid["blocks"][0]["screen_ids"])
    with pytest.raises(ValueError, match="reconstruct"):
        validate_block_manifest(invalid)


def test_parallel_block_gather_is_exactly_equivalent_to_serial(tmp_path):
    from KHz_filament.hr4c_state import evolve_hr4_full_z
    from KHz_filament.hr4e5_parallel import (
        build_screen_blocks, compare_store_states, execute_worker, gather_worker_outputs, store_spec,
    )

    serial, parallel = _make_store(tmp_path, "serial"), _make_store(tmp_path, "parallel")
    try:
        evolve_hr4_full_z(
            serial, dt_hydro=1.0e-6, n_hydro_steps=2, batch_intervals=2,
            chi=21.7e-6, nu=1.5e-5, n0=1.00027,
        )
        serial_state = store_spec(serial)
        input_state = store_spec(parallel)
        partition = build_screen_blocks(_records(), block_size=2, n_workers=2)
        workers = []
        output_dir = tmp_path / "parallel_blocks"
        for worker_index in range(2):
            assigned = [block for block in partition["blocks"] if block["worker_index"] == worker_index]
            workers.append(execute_worker(
                worker_index=worker_index, blocks=assigned, state=input_state, out_dir=output_dir,
                dt_hydro=1.0e-6, n_hydro_steps=2, chi=21.7e-6, nu=1.5e-5, n0=1.00027,
            ))
        gathered = gather_worker_outputs(
            state=input_state, partition=partition, worker_manifests=workers, batch_intervals=2,
        )
        assert gathered["status"] == "PASS"
        # The coordinator promoted the parallel store through a separate
        # handle; reopen before deriving its now-current generation spec.
        parallel.close()
        from KHz_filament.hr4e5_parallel import open_store_from_spec
        promoted_spec = dict(input_state)
        promoted_spec["input_generation"] = gathered["generation"]
        promoted_spec["authoritative_filenames"] = gathered["authoritative_filenames"]
        parallel = open_store_from_spec(promoted_spec)
        report = compare_store_states(
            reference_state=serial_state, candidate_state=store_spec(parallel), records=_records(),
        )
        assert report["status"] == "P3_EXACT_EQUIVALENCE_PASS"
        assert report["exact_equal"] is True
    finally:
        serial.close()
        parallel.close()


def test_gather_rejects_partial_worker_results(tmp_path):
    from KHz_filament.hr4e5_parallel import build_screen_blocks, execute_worker, gather_worker_outputs, store_spec

    store = _make_store(tmp_path, "partial")
    try:
        state = store_spec(store)
        partition = build_screen_blocks(_records(), block_size=1, n_workers=2)
        only_zero = [block for block in partition["blocks"] if block["worker_index"] == 0]
        worker = execute_worker(
            worker_index=0, blocks=only_zero, state=state, out_dir=tmp_path / "partial_blocks",
            dt_hydro=1.0e-6, n_hydro_steps=1, chi=21.7e-6, nu=1.5e-5, n0=1.00027,
        )
        with pytest.raises(ValueError, match="partial"):
            gather_worker_outputs(state=state, partition=partition, worker_manifests=[worker], batch_intervals=1)
    finally:
        store.close()


def test_screen_independence_audit_is_explicit():
    from KHz_filament.hr4e5_parallel import screen_independence_audit

    audit = screen_independence_audit()
    assert audit["status"] == "P1_SCREEN_INDEPENDENCE_CONFIRMED"
    assert audit["evidence"]["sequential_time_steps_within_screen"] is True
