from __future__ import annotations

import json
import numpy as np
import pytest


def _controller(tmp_path, *, npulses=3, resume=False, dtype=np.float64):
    from KHz_filament.grids import make_axes
    from KHz_filament.hr3c_state_machine import HR3CStateController

    axes = make_axes(8, 8, 8, 2e-3, 2e-3, 80e-15)
    controller = HR3CStateController(
        output_path=str(tmp_path / "run.npz"), n_intervals=2, shape=(8, 8), dtype=dtype,
        z_edges=np.array([0.0, 1e-4, 2e-4]), dx=axes.dx, dy=axes.dy,
        D_th=21.7e-6, f_rep=1e3, edge_threshold=1.0, batch_intervals=1,
        npulses=npulses, resume=resume,
    )
    controller.attach_grid(axes.kperp2)
    return controller


def _commit_one(controller, pulse):
    tx = controller.begin_pulse()
    coordinates = np.linspace(-1.0, 1.0, 8)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    increment = -1e-6 * (pulse + 1) * np.exp(-8.0 * (xx**2 + yy**2))
    for index in range(2):
        pre = tx.read_interval(index)
        np.testing.assert_array_equal(pre, np.asarray(controller.store.read_current_interval(index)))
        tx.update_interval(index, increment)
    controller.commit_post_pulse(tx, pulse)


def test_cc1_cc3_cc4_transactional_counts_and_final_pulse_has_no_diffusion(tmp_path):
    controller = _controller(tmp_path, npulses=3)
    pre0 = np.array(controller.store.read_current_batch(0, 2), copy=True)
    post_commits = diffusion_passes = 0
    for pulse in range(3):
        _commit_one(controller, pulse)
        post_commits += 1
        if pulse < 2:
            controller.diffuse_to_next_pre()
            diffusion_passes += 1
    assert post_commits == 3 and diffusion_passes == 2
    assert controller.manifest["physical_stage"] == "post_pulse"
    assert controller.manifest["run_complete"]
    np.testing.assert_array_equal(pre0, np.zeros_like(pre0))
    controller.close()


def test_cc6_pulse_crash_leaves_pre_manifest_and_resume_retries_fresh_pulse(tmp_path):
    controller = _controller(tmp_path, npulses=1)
    pre = np.array(controller.store.read_current_batch(0, 2), copy=True)
    tx = controller.begin_pulse()
    tx.read_interval(0)
    tx.update_interval(0, np.full((8, 8), -1e-6))
    tx.store.mark_next_invalid()  # simulate propagation failure before finalize
    assert controller.manifest["physical_stage"] == "pre_pulse"
    np.testing.assert_array_equal(controller.store.read_current_batch(0, 2), pre)
    controller.close()

    resumed = _controller(tmp_path, npulses=1, resume=True)
    assert resumed.manifest["physical_stage"] == "pre_pulse"
    _commit_one(resumed, 0)
    assert resumed.manifest["run_complete"]
    resumed.close()


def test_cc7_cc8_cc9_atomic_manifest_reopen_and_fingerprint_fail_closed(tmp_path, monkeypatch):
    import KHz_filament.hr3c_state_machine as machine

    controller = _controller(tmp_path, npulses=2)
    _commit_one(controller, 0)
    post = np.array(controller.store.read_current_batch(0, 2), copy=True)
    controller.close()
    resumed = _controller(tmp_path, npulses=2, resume=True)
    np.testing.assert_array_equal(resumed.store.read_current_batch(0, 2), post)
    resumed.diffuse_to_next_pre()
    resumed.close()

    manifest_path = tmp_path / "run.hr3c_state_manifest.json"
    old = manifest_path.read_text(encoding="utf-8")
    monkeypatch.setattr(machine.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("interrupt")))
    with pytest.raises(OSError):
        machine._atomic_json(manifest_path, {"bad": True})
    assert manifest_path.read_text(encoding="utf-8") == old
    data = json.loads(old)
    data["D_th"] = 9.0
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _controller(tmp_path, npulses=2, resume=True)


def test_final_post_is_atomic_completed_state_and_resume_is_read_only(tmp_path):
    controller = _controller(tmp_path, npulses=1)
    _commit_one(controller, 0)
    manifest = dict(controller.manifest)
    final_state = np.array(controller.store.read_current_batch(0, 2), copy=True)
    assert manifest["physical_stage"] == "post_pulse"
    assert manifest["run_complete"]
    assert manifest["next_pulse_index"] == 1
    assert manifest["n_fresh_pulses_completed_total"] == 1
    assert manifest["n_hr3b_post_commits_total"] == 1
    assert manifest["n_hr3c_diffusion_passes_total"] == 0
    controller.close()

    resumed = _controller(tmp_path, npulses=1, resume=True)
    np.testing.assert_array_equal(resumed.store.read_current_batch(0, 2), final_state)
    with pytest.raises(ValueError, match="not ready for diffusion"):
        resumed.diffuse_to_next_pre()
    with pytest.raises(ValueError, match="not ready for a fresh pulse"):
        resumed.begin_pulse()
    assert resumed.manifest == manifest
    resumed.close()


@pytest.mark.parametrize("field,value", [
    ("authoritative_filename", "unknown.npy"),
    ("scratch_filename", "unknown.npy"),
    ("scratch_filename", "run.hr3c_delta_n_th_current.npy"),
    ("next_pulse_index", 99),
])
def test_manifest_slot_and_stage_invariants_fail_closed(tmp_path, field, value):
    controller = _controller(tmp_path, npulses=2)
    path = tmp_path / "run.hr3c_state_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data[field] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="invariant"):
        _controller(tmp_path, npulses=2, resume=True)
    controller.close()


def test_diffusion_interruption_resume_matches_uninterrupted_final_state_and_totals(tmp_path, monkeypatch):
    import KHz_filament.hr3c_state_machine as machine

    reference_root = tmp_path / "reference"
    interrupted_root = tmp_path / "interrupted"
    reference_root.mkdir()
    interrupted_root.mkdir()
    reference = _controller(reference_root, npulses=2)
    _commit_one(reference, 0)
    reference.diffuse_to_next_pre()
    _commit_one(reference, 1)
    reference_state = np.array(reference.store.read_current_batch(0, 2), copy=True)
    reference_manifest = dict(reference.manifest)
    reference.close()

    controller = _controller(interrupted_root, npulses=2)
    _commit_one(controller, 0)
    post = np.array(controller.store.read_current_batch(0, 2), copy=True)
    original = machine.diffuse_current_to_next
    monkeypatch.setattr(machine, "diffuse_current_to_next", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("interrupt")))
    with pytest.raises(RuntimeError, match="interrupt"):
        controller.diffuse_to_next_pre()
    assert controller.manifest["physical_stage"] == "post_pulse"
    assert controller.manifest["n_fresh_pulses_completed_total"] == 1
    assert controller.manifest["n_hr3b_post_commits_total"] == 1
    assert controller.manifest["n_hr3c_diffusion_passes_total"] == 0
    np.testing.assert_array_equal(controller.store.read_current_batch(0, 2), post)
    controller.close()

    monkeypatch.setattr(machine, "diffuse_current_to_next", original)
    resumed = _controller(interrupted_root, npulses=2, resume=True)
    resumed.diffuse_to_next_pre()
    _commit_one(resumed, 1)
    np.testing.assert_allclose(resumed.store.read_current_batch(0, 2), reference_state, rtol=3e-12, atol=1e-15)
    for key in (
        "n_fresh_pulses_completed_total", "n_hr3b_post_commits_total",
        "n_hr3c_diffusion_passes_total", "physical_stage", "pulse_index",
        "next_pulse_index", "run_complete",
    ):
        assert resumed.manifest[key] == reference_manifest[key]
    resumed.close()


def test_hr3c_uses_exactly_two_full_state_files(tmp_path):
    controller = _controller(tmp_path, npulses=1)
    _commit_one(controller, 0)
    controller.close()
    state_files = sorted(tmp_path.glob("run.hr3c_delta_n_th_*.npy"))
    assert [path.name for path in state_files] == [
        "run.hr3c_delta_n_th_current.npy", "run.hr3c_delta_n_th_next.npy",
    ]
    assert not (tmp_path / "run.hr3b_delta_n_th.npy").exists()
