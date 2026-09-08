from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest


def _records(count: int):
    return [{"ordinal": index, "screen_id": f"z{index:05d}", "z_m": index * 1.0e-4} for index in range(count)]


def _lifecycle(tmp_path, *, count=16, queue_depth=16):
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle

    coordinate = np.linspace(-1.0, 1.0, 8)
    mode = np.outer(1.0 - coordinate**2, 1.0 - coordinate**2)
    current = {
        "delta_n": np.stack([-1.0e-6 * (index + 1) * mode for index in range(count)]).astype(np.float64),
        "vx": np.zeros((count, 8, 8), dtype=np.float64),
        "vy": np.zeros((count, 8, 8), dtype=np.float64),
    }
    return StreamingLifecycle.create(root=tmp_path / "stream", current=current, screen_records=_records(count), current_generation="pre-p7-g12", dx_m=1.0e-4, dy_m=1.0e-4, queue_depth=queue_depth), current


def _commit(lifecycle, ordinal: int, *, enqueue=True):
    lifecycle.deposition_finalized(ordinal)
    current = lifecycle.current_fields(ordinal)
    lifecycle.commit_post_from_delta_n(ordinal, current["delta_n"] - 1.0e-8)
    if enqueue:
        lifecycle.enqueue_post(ordinal)


def _complete(lifecycle):
    for ordinal in range(lifecycle.manifest["expected_screen_count"]):
        _commit(lifecycle, ordinal)
    while lifecycle.run_one_hydro_block(dt_hydro=1.0e-6, n_hydro_steps=1, chi=0.0, nu=0.0, n0=1.00027, gravity_y=0.0):
        pass


def test_current_post_next_are_distinct_and_post_commit_precedes_enqueue(tmp_path):
    lifecycle, original = _lifecycle(tmp_path)
    assert lifecycle.manifest["current_generation"] != lifecycle.manifest["next_generation"]
    current = lifecycle.current_fields(0)
    current["delta_n"][...] = 42.0
    np.testing.assert_array_equal(lifecycle.current_fields(0)["delta_n"], original["delta_n"][0])
    lifecycle.deposition_finalized(0)
    with pytest.raises(ValueError, match="float64"):
        lifecycle.commit_post(0, {"delta_n": np.zeros((8, 8), dtype=np.float32), "vx": np.zeros((8, 8), dtype=np.float32), "vy": np.zeros((8, 8), dtype=np.float32)})
    assert lifecycle.manifest["records"][0]["post"] is None
    lifecycle.commit_post_from_delta_n(0, lifecycle.current_fields(0)["delta_n"] - 1.0e-8)
    lifecycle.enqueue_post(0)
    record = lifecycle.manifest["records"][0]
    assert record["post"] is not record["current"]
    assert record["state"] == "HYDRO_QUEUED"


def test_duplicate_post_queue_and_next_are_fail_closed(tmp_path):
    from KHz_filament.hr4e5s_streaming import DuplicateCommitError

    lifecycle, _ = _lifecycle(tmp_path, count=8, queue_depth=8)
    _commit(lifecycle, 0)
    with pytest.raises((DuplicateCommitError, Exception), match="POST|state"):
        lifecycle.commit_post_from_delta_n(0, lifecycle.current_fields(0)["delta_n"])
    with pytest.raises(DuplicateCommitError, match="duplicate queue"):
        lifecycle.enqueue_post(0)
    for ordinal in range(1, 8):
        _commit(lifecycle, ordinal)
    block = lifecycle.claim_block()
    for ordinal in block:
        fields = lifecycle._artifact_fields(lifecycle.manifest["records"][ordinal]["post"], namespace="POST")
        lifecycle.commit_next(ordinal, fields)
        with pytest.raises((DuplicateCommitError, Exception), match="NEXT|running"):
            lifecycle.commit_next(ordinal, fields)


def test_backpressure_preserves_committed_post_and_deterministic_blocks(tmp_path):
    from KHz_filament.hr4e5s_streaming import BackpressureError

    lifecycle, _ = _lifecycle(tmp_path, count=16, queue_depth=8)
    for ordinal in reversed(range(8)):
        _commit(lifecycle, ordinal)
    assert lifecycle.claim_block() == list(range(8))
    for ordinal in range(8):
        fields = lifecycle._artifact_fields(lifecycle.manifest["records"][ordinal]["post"], namespace="POST")
        lifecycle.commit_next(ordinal, fields)
    for ordinal in range(8, 16):
        lifecycle.deposition_finalized(ordinal)
        lifecycle.commit_post_from_delta_n(ordinal, lifecycle.current_fields(ordinal)["delta_n"])
    for ordinal in range(8, 16):
        lifecycle.enqueue_post(ordinal)
    assert lifecycle.claim_block() == list(range(8, 16))
    isolated, _ = _lifecycle(tmp_path / "pressure", count=9, queue_depth=8)
    for ordinal in range(9):
        isolated.deposition_finalized(ordinal)
        isolated.commit_post_from_delta_n(ordinal, isolated.current_fields(ordinal)["delta_n"])
        if ordinal < 8:
            isolated.enqueue_post(ordinal)
    with pytest.raises(BackpressureError, match="full"):
        isolated.enqueue_post(8)
    assert isolated.manifest["records"][8]["state"] == "POST_COMMITTED"


def test_backpressure_pauses_hook_style_producer_until_a_block_is_claimed(tmp_path):
    lifecycle, _ = _lifecycle(tmp_path, count=16, queue_depth=8)
    for ordinal in range(9):
        lifecycle.deposition_finalized(ordinal)
        lifecycle.commit_post_from_delta_n(ordinal, lifecycle.current_fields(ordinal)["delta_n"])
        if ordinal < 8:
            lifecycle.enqueue_post(ordinal)

    errors = []
    producer = threading.Thread(
        target=lambda: _enqueue_waiting(lifecycle, 8, errors), daemon=True,
    )
    producer.start()
    time.sleep(0.05)
    assert producer.is_alive()
    assert lifecycle.manifest["records"][8]["state"] == "POST_COMMITTED"
    assert lifecycle.claim_block() == list(range(8))
    producer.join(timeout=2.0)
    assert not producer.is_alive() and not errors
    assert lifecycle.manifest["records"][8]["state"] == "HYDRO_QUEUED"
    assert any(event["event"] == "BACKPRESSURE" for event in lifecycle.manifest["rate_events"])


def test_concurrent_consumers_claim_disjoint_deterministic_blocks(tmp_path):
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle

    lifecycle, _ = _lifecycle(tmp_path, count=16, queue_depth=16)
    for ordinal in range(16):
        _commit(lifecycle, ordinal)
    left = StreamingLifecycle.open(lifecycle.root)
    right = StreamingLifecycle.open(lifecycle.root)
    gate = threading.Barrier(2)
    claims, errors = [], []

    def claim(worker):
        try:
            gate.wait(timeout=1.0)
            claims.append(worker.claim_block())
        except Exception as error:  # pragma: no cover - asserted by the caller
            errors.append(error)

    threads = [threading.Thread(target=claim, args=(worker,), daemon=True) for worker in (left, right)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)
    assert not errors and all(not thread.is_alive() for thread in threads)
    assert sorted(claims) == [list(range(8)), list(range(8, 16))]


def _enqueue_waiting(lifecycle, ordinal, errors):
    try:
        lifecycle.enqueue_post(ordinal, wait_for_capacity=True, timeout_s=1.0)
    except Exception as error:  # pragma: no cover - asserted by the caller
        errors.append(error)


def test_barrier_rejects_missing_and_wrong_generation_then_promotes_atomically(tmp_path):
    from KHz_filament.hr4e5s_streaming import BarrierError

    lifecycle, _ = _lifecycle(tmp_path, count=8, queue_depth=8)
    for ordinal in range(8):
        _commit(lifecycle, ordinal)
    block = lifecycle.claim_block()
    for ordinal in block[:-1]:
        fields = lifecycle._artifact_fields(lifecycle.manifest["records"][ordinal]["post"], namespace="POST")
        lifecycle.commit_next(ordinal, fields)
    with pytest.raises(BarrierError, match="not_next"):
        lifecycle.validate_barrier()
    fields = lifecycle._artifact_fields(lifecycle.manifest["records"][7]["post"], namespace="POST")
    lifecycle.commit_next(7, fields)
    manifest = json.loads(lifecycle.manifest_path.read_text(encoding="utf-8"))
    manifest["current_generation"] = "wrong-generation"
    lifecycle.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BarrierError, match="provenance"):
        lifecycle.validate_barrier()
    manifest["current_generation"] = "pre-p7-g12"
    lifecycle.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    lifecycle = type(lifecycle).open(lifecycle.root)
    barrier = lifecycle.validate_barrier()
    pointer = lifecycle.promote_next_to_current()
    assert barrier["status"] == "PASS" and pointer["authoritative_namespace"] == "NEXT"
    assert pointer["authoritative_generation"] == lifecycle.manifest["next_generation"]


def test_barrier_rejects_duplicate_next_identity(tmp_path):
    from KHz_filament.hr4e5s_streaming import BarrierError

    lifecycle, _ = _lifecycle(tmp_path, count=8, queue_depth=8)
    _complete(lifecycle)
    manifest = json.loads(lifecycle.manifest_path.read_text(encoding="utf-8"))
    manifest["records"][1]["next"] = dict(manifest["records"][0]["next"])
    lifecycle.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BarrierError, match="identity"):
        lifecycle.validate_barrier()


def test_barrier_rejects_tampered_next_artifact_hash(tmp_path):
    from KHz_filament.hr4e5s_streaming import BarrierError

    lifecycle, _ = _lifecycle(tmp_path, count=8, queue_depth=8)
    _complete(lifecycle)
    artifact = lifecycle.root / lifecycle.manifest["records"][0]["next"]["artifact"]
    with artifact.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(BarrierError, match="hash-mismatched"):
        lifecycle.validate_barrier()


def test_restart_reconstructs_only_missing_next_work(tmp_path):
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle

    lifecycle, _ = _lifecycle(tmp_path, count=16, queue_depth=16)
    for ordinal in range(16):
        _commit(lifecycle, ordinal)
    first = lifecycle.run_one_hydro_block(dt_hydro=1.0e-6, n_hydro_steps=1, chi=0.0, nu=0.0, n0=1.00027, gravity_y=0.0)
    reopened = StreamingLifecycle.open(lifecycle.root)
    assert reopened.reconstruct_queue() == list(range(8, 16))
    assert all(reopened.manifest["records"][ordinal]["state"] == "NEXT_COMMITTED" for ordinal in first)


def test_restart_rejects_post_from_another_current_generation(tmp_path):
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle, StreamingLifecycleError

    lifecycle, _ = _lifecycle(tmp_path, count=8, queue_depth=8)
    _commit(lifecycle, 0, enqueue=False)
    manifest = json.loads(lifecycle.manifest_path.read_text(encoding="utf-8"))
    manifest["current_generation"] = "wrong-generation"
    lifecycle.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    reopened = StreamingLifecycle.open(lifecycle.root)
    with pytest.raises(StreamingLifecycleError, match="provenance"):
        reopened.reconstruct_queue()


def test_optional_optical_hook_commits_only_after_the_frontier(tmp_path):
    from KHz_filament.hr4e5s_streaming import make_post_commit_hook

    lifecycle, _ = _lifecycle(tmp_path, count=8, queue_depth=8)
    hook = make_post_commit_hook(lifecycle)
    hook(interval=SimpleNamespace(index=0), state_after=lifecycle.current_fields(0)["delta_n"] - 1.0e-8, hr3a_authoritative=True, hr3b_authoritative=True)
    assert lifecycle.manifest["records"][0]["state"] == "HYDRO_QUEUED"


def test_real_optical_path_completes_optical_post_hydro_barrier_and_promotion(tmp_path):
    from KHz_filament.config import BeamConfig, GridConfig, IonizationConfig, PropagationConfig, RamanConfig
    from KHz_filament.constants import N0_air, Ui_N2, c0, n2_air
    from KHz_filament.device import xp
    from KHz_filament.grids import make_axes
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle, make_post_commit_hook
    from KHz_filament.longitudinal import build_deposition_contract, build_longitudinal_schedule
    from KHz_filament.propagate import propagate_one_pulse
    from KHz_filament.runner import build_transverse_input_field
    from KHz_filament.slow_state import validate_hr3b_parameters

    class Transaction:
        def __init__(self, count):
            self.current = np.zeros((count, 8, 8), dtype=np.float64)
            self.next = np.zeros_like(self.current)

        def read_interval(self, index):
            return self.current[index]

        def update_interval(self, index, increment):
            self.next[index] = self.current[index] + np.asarray(increment, dtype=np.float64)
            return self.next[index]

        def metadata(self):
            return {"hr3b_state_schema": "test.streaming.frontier.v1", "hr3b_state_filename": "transaction.npy", "hr3b_state_dtype": "float64", "hr3b_state_shape": (2, 8, 8), "hr3b_state_interval_centered": True, "hr3b_state_disk_backed": False}

    grid = GridConfig(Nx=8, Ny=8, Nt=8, Lx=8e-4, Ly=8e-4, Twin=80e-15)
    beam = BeamConfig(w0=1.5e-4, tau_fwhm=40e-15, energy_J=1e-10, focal_length=None)
    prop = PropagationConfig(z_max=8e-4, dz=1e-4, linear_model="paraxial", auto_substep=False, focus_window_step=False, limit_focus_window=False, progress_every_z=0, energy_probe_every=0, diag_extra=False, use_electronic_kerr=False, use_raman_phase=False, use_raman_absorption=False, use_plasma_phase=False, use_ionization_loss=False, use_ionization_solver=False)
    axes = make_axes(grid.Nx, grid.Ny, grid.Nt, grid.Lx, grid.Ly, grid.Twin)
    source, _ = build_transverse_input_field(axes, beam, xp.complex64)
    schedule = build_longitudinal_schedule(dz=prop.dz, z_max=prop.z_max)
    current = {"delta_n": np.zeros((8, 8, 8), dtype=np.float64), "vx": np.zeros((8, 8, 8), dtype=np.float64), "vy": np.zeros((8, 8, 8), dtype=np.float64)}
    lifecycle = StreamingLifecycle.create(root=tmp_path / "frontier", current=current, screen_records=_records(8), current_generation="pre-p1-g0", dx_m=axes.dx, dy_m=axes.dy)
    beta = validate_hr3b_parameters(rho0=1.23, Cv=1000.0 / 1.4, T0=prop.air_T, n0=beam.n0)
    omega0 = 2.0 * np.pi * c0 / beam.lam0
    propagate_one_pulse(source, kperp2=axes.kperp2, k0=beam.n0 * omega0 / c0, omega0=omega0, dz=prop.dz, z_max=prop.z_max, n0=beam.n0, n2=n2_air, Ui=Ui_N2, N0=N0_air, ion_conf=IonizationConfig(species=[]), dn_gas=xp.zeros((8, 8), dtype=xp.float32), dt=axes.dt, axes=axes, prop_conf=prop, raman_conf=RamanConfig(enabled=False, absorption=False), record_onaxis_rho_time=False, record_every_z=1, longitudinal_schedule=schedule, deposition_contract=build_deposition_contract(schedule, axes=axes), thermal_slow_state=Transaction(8), hr3b_parameters={"rho0": 1.23, "Cv": 1000.0 / 1.4, "T0": prop.air_T, "n0": beam.n0, "beta_th": beta}, post_commit_hook=make_post_commit_hook(lifecycle))
    assert [record["state"] for record in lifecycle.manifest["records"]] == ["HYDRO_QUEUED"] * 8
    assert lifecycle.run_one_hydro_block(dt_hydro=1.0e-6, n_hydro_steps=1, chi=0.0, nu=0.0, n0=beam.n0, gravity_y=0.0) == list(range(8))
    assert lifecycle.validate_barrier()["status"] == "PASS"
    assert lifecycle.promote_next_to_current()["authoritative_generation"] == lifecycle.manifest["next_generation"]


def test_s2_bounded_16_screen_smoke_uses_the_frozen_single_screen_worker(tmp_path):
    lifecycle, _ = _lifecycle(tmp_path, count=16, queue_depth=16)
    _complete(lifecycle)
    barrier = lifecycle.validate_barrier()
    pointer = lifecycle.promote_next_to_current()
    metrics = lifecycle.rate_metrics()
    assert barrier["status"] == "PASS"
    assert pointer["authoritative_namespace"] == "NEXT"
    assert metrics["post_committed_count"] == 16
    reopened = type(lifecycle).open(lifecycle.root)
    assert reopened._authoritative_generation == lifecycle.manifest["next_generation"]
    np.testing.assert_array_equal(reopened.current_fields(0)["delta_n"], lifecycle._artifact_fields(lifecycle.manifest["records"][0]["next"], namespace="NEXT")["delta_n"])
