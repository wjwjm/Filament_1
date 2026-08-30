from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest


def _components(*, npulses: int, resume: bool = False, hr3c: bool = True):
    from KHz_filament.config import (
        BeamConfig, GridConfig, HeatConfig, IonizationConfig,
        PropagationConfig, RamanConfig, RunConfig,
    )

    return dict(
        grid=GridConfig(Nx=16, Ny=16, Nt=8, Lx=8e-3, Ly=8e-3, Twin=80e-15),
        beam=BeamConfig(w0=2e-4, tau_fwhm=40e-15, energy_J=1e-10, focal_length=None),
        prop=PropagationConfig(
            z_max=1e-4, dz=1e-4, linear_model="paraxial", auto_substep=False,
            focus_window_step=False, limit_focus_window=False, progress_every_z=0,
            energy_probe_every=0, diag_extra=False, use_electronic_kerr=False,
            use_raman_phase=False, use_raman_absorption=False, use_plasma_phase=False,
            use_ionization_loss=False, use_ionization_solver=False,
        ),
        ion=IonizationConfig(species=[]),
        heat=HeatConfig(hr3b_enabled=True, hr3c_enabled=hr3c, resume_hr3c=resume, hr3c_batch_intervals=1),
        run=RunConfig(Npulses=npulses), raman=RamanConfig(enabled=False, absorption=False),
        dtype="fp32",
    )


def _authoritative_state(output: Path) -> np.ndarray:
    manifest = json.loads(output.with_suffix(".hr3c_state_manifest.json").read_text(encoding="utf-8"))
    return np.array(np.load(output.with_name(manifest["authoritative_filename"]), mmap_mode="r"), copy=True)


def _npz_count(output: Path, key: str) -> int:
    with np.load(output) as data:
        return int(data[key])


def _install_deterministic_propagator(monkeypatch, *, fail_on=None, pulse_offset=0):
    runner = importlib.import_module("KHz_filament.runner")
    fields, calls = [], []

    def fake_propagate(E, *, thermal_slow_state, longitudinal_schedule, **kwargs):
        fields.append(np.array(E, copy=True))
        pulse = pulse_offset + len(calls)
        coordinates = np.linspace(-1.0, 1.0, E.shape[-1])
        xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
        increment = -1e-6 * (pulse + 1) * np.exp(-24.0 * (xx**2 + yy**2))
        for interval in longitudinal_schedule.intervals:
            thermal_slow_state.read_interval(interval.index)
            thermal_slow_state.update_interval(interval.index, increment)
            if fail_on == (pulse, interval.index):
                raise RuntimeError("injected pulse interruption")
        calls.append(pulse)
        E[...] = pulse + 1
        return E, np.zeros(E.shape[-2:], dtype=np.float32), {
            "delta_n_state_min_after_update": np.array([increment.min()]),
        }

    monkeypatch.setattr(runner, "propagate_one_pulse", fake_propagate)
    monkeypatch.setattr(runner, "write_nonlinear_diagnostic_report", lambda *a, **k: {"validation": {"z_records": 0}})
    return runner, fields, calls


@pytest.mark.parametrize("npulses,expected_diffusions", [(1, 0), (2, 1), (3, 2)])
def test_runner_counts_fresh_source_legacy_isolation_and_two_slots(monkeypatch, tmp_path, npulses, expected_diffusions):
    runner, fields, calls = _install_deterministic_propagator(monkeypatch)
    monkeypatch.setattr(runner, "diffuse_dn_gas", lambda *a, **k: (_ for _ in ()).throw(AssertionError("legacy path")))
    output = tmp_path / f"n{npulses}.npz"
    result = runner.run_demo(**_components(npulses=npulses), out_path=str(output), return_results=True)
    manifest = json.loads(output.with_suffix(".hr3c_state_manifest.json").read_text(encoding="utf-8"))
    assert calls == list(range(npulses))
    assert all(np.array_equal(field, fields[0]) for field in fields[1:])
    assert manifest["physical_stage"] == "post_pulse" and manifest["run_complete"]
    assert (manifest["pulse_index"], manifest["next_pulse_index"]) == (npulses - 1, npulses)
    assert manifest["n_fresh_pulses_completed_total"] == npulses
    assert manifest["n_hr3b_post_commits_total"] == npulses
    assert manifest["n_hr3c_diffusion_passes_total"] == expected_diffusions
    assert _npz_count(output, "n_hr3c_diffusion_passes") == expected_diffusions
    assert sorted(path.name for path in tmp_path.glob(f"n{npulses}.hr3c_delta_n_th_*.npy")) == [
        f"n{npulses}.hr3c_delta_n_th_current.npy", f"n{npulses}.hr3c_delta_n_th_next.npy",
    ]
    assert not output.with_suffix(".hr3b_delta_n_th.npy").exists()


def test_pulse_interruption_resume_matches_uninterrupted_runner(monkeypatch, tmp_path):
    runner, _, _ = _install_deterministic_propagator(monkeypatch)
    monkeypatch.setattr(runner, "diffuse_dn_gas", lambda *a, **k: (_ for _ in ()).throw(AssertionError("legacy path")))
    reference = tmp_path / "reference.npz"
    runner.run_demo(**_components(npulses=2), out_path=str(reference))
    reference_state = _authoritative_state(reference)
    reference_manifest = json.loads(reference.with_suffix(".hr3c_state_manifest.json").read_text(encoding="utf-8"))

    interrupted = tmp_path / "interrupted.npz"
    runner, _, _ = _install_deterministic_propagator(monkeypatch, fail_on=(1, 0))
    monkeypatch.setattr(runner, "diffuse_dn_gas", lambda *a, **k: (_ for _ in ()).throw(AssertionError("legacy path")))
    with pytest.raises(RuntimeError, match="injected pulse interruption"):
        runner.run_demo(**_components(npulses=2), out_path=str(interrupted))
    partial = json.loads(interrupted.with_suffix(".hr3c_state_manifest.json").read_text(encoding="utf-8"))
    assert partial["physical_stage"] == "pre_pulse"
    assert partial["next_pulse_index"] == 1
    assert partial["n_fresh_pulses_completed_total"] == 1

    runner, _, _ = _install_deterministic_propagator(monkeypatch, pulse_offset=1)
    monkeypatch.setattr(runner, "diffuse_dn_gas", lambda *a, **k: (_ for _ in ()).throw(AssertionError("legacy path")))
    runner.run_demo(**_components(npulses=2, resume=True), out_path=str(interrupted))
    resumed_manifest = json.loads(interrupted.with_suffix(".hr3c_state_manifest.json").read_text(encoding="utf-8"))
    np.testing.assert_allclose(_authoritative_state(interrupted), reference_state, rtol=3e-6, atol=1e-12)
    for key in ("n_fresh_pulses_completed_total", "n_hr3b_post_commits_total", "n_hr3c_diffusion_passes_total", "physical_stage", "pulse_index", "next_pulse_index", "run_complete"):
        assert resumed_manifest[key] == reference_manifest[key]


def test_completed_runner_resume_does_not_reexecute(monkeypatch, tmp_path):
    runner, _, _ = _install_deterministic_propagator(monkeypatch)
    output = tmp_path / "completed.npz"
    runner.run_demo(**_components(npulses=3), out_path=str(output))
    state_before = _authoritative_state(output)
    manifest_before = json.loads(output.with_suffix(".hr3c_state_manifest.json").read_text(encoding="utf-8"))
    runner, _, _ = _install_deterministic_propagator(monkeypatch)
    monkeypatch.setattr(runner, "propagate_one_pulse", lambda *a, **k: (_ for _ in ()).throw(AssertionError("pulse rerun")))
    import KHz_filament.hr3c_state_machine as machine
    monkeypatch.setattr(machine, "diffuse_current_to_next", lambda *a, **k: (_ for _ in ()).throw(AssertionError("diffusion rerun")))
    resumed = runner.run_demo(**_components(npulses=3, resume=True), out_path=str(output), return_results=True)
    np.testing.assert_array_equal(_authoritative_state(output), state_before)
    assert json.loads(output.with_suffix(".hr3c_state_manifest.json").read_text(encoding="utf-8")) == manifest_before
    assert _npz_count(output, "n_fresh_pulses_completed") == 3
    assert _npz_count(output, "n_hr3b_post_commits") == 3
    assert _npz_count(output, "n_hr3c_diffusion_passes") == 2


def test_standalone_hr3b_remains_single_file_without_hr3c_manifest(tmp_path):
    runner = importlib.import_module("KHz_filament.runner")
    output = tmp_path / "standalone.npz"
    runner.run_demo(**_components(npulses=1, hr3c=False), out_path=str(output))
    assert output.with_suffix(".hr3b_delta_n_th.npy").is_file()
    assert not output.with_suffix(".hr3c_state_manifest.json").exists()
    assert not list(tmp_path.glob("standalone.hr3c_delta_n_th_*.npy"))
