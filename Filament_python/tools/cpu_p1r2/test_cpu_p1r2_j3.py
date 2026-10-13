from __future__ import annotations

import os
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools.cpu_p1r2 import cpu_p1r2_j3 as j3
from tools.cpu_p1r2 import cpu_p1r2_j1 as j1
from tools.cpu_p1r2 import cpu_p1r2_j2 as j2


def test_hydro_adapter_keeps_1000_steps_and_frozen_arguments():
    fields = {name: np.zeros((3, 4), dtype=np.float64, order="C")
              for name in j3.FIELDS}
    params = {"dx": 1e-5, "dy": 1e-5, "dt_hydro": 1e-6, "chi": 21.7e-6,
        "nu": 1.5e-5, "n0": 1.00027, "gravity_x": 0.0, "gravity_y": -9.81,
        "cfl_limit": 1.0, "n_steps": 1000, "x_min": -1.5e-3, "y_min": -1.0e-3}
    calls = {}

    def advance(delta_n, vx, vy, **kwargs):
        calls.update(kwargs)
        return {"delta_n": delta_n, "vx": vx, "vy": vy,
                "steps": kwargs["n_steps"], "performance": {"backend": "numpy"}}

    result = j3._invoke_hydro(advance, fields, params)
    assert result["steps"] == 1000
    assert calls["n_steps"] == 1000 and calls["require_stable"] is True
    assert calls["dt_hydro"] == 1e-6 and calls["cfl_limit"] == 1.0
    assert calls["x_min"] == -1.5e-3 and calls["y_min"] == -1.0e-3


def test_worker_screen_copy_is_independent_and_preserves_values():
    full = {name: np.arange(8 * 3 * 4, dtype=np.float64).reshape(8, 3, 4).copy(order="C")
            for name in j3.FIELDS}
    copied = j3._copy_post_screen(full, 5)
    for name in j3.FIELDS:
        np.testing.assert_array_equal(copied[name], full[name][5])
        assert copied[name].dtype == np.float64 and copied[name].flags.c_contiguous
        assert not np.shares_memory(copied[name], full[name])
    copied["delta_n"][0, 0] = -1.0
    assert full["delta_n"][5, 0, 0] == 60.0


def test_worker_cpu_selection_produces_eight_distinct_assignments(monkeypatch):
    monkeypatch.setattr(j3.os, "sched_getaffinity", lambda _pid: set(range(16)), raising=False)
    monkeypatch.setattr(j3.os, "sched_setaffinity", lambda _pid, _cpus: None, raising=False)
    monkeypatch.setattr(j3.common, "cpu_topology", lambda allowed: {
        "physical_core_representatives": list(allowed)[::2]})
    monkeypatch.setattr(j3.common, "physical_topology_verified", lambda _topology: True)
    monkeypatch.setattr(j3.common, "select_physical_cpus",
        lambda count, allowed, topology: topology["physical_core_representatives"][:count])
    _topology, cpus = j3._verified_worker_cpus(8)
    assert len(cpus) == 8
    assert len(set(cpus)) == 8


def test_small_real_hr4_call_uses_numpy_and_returns_finite_fields():
    from KHz_filament import hr4

    zeros = np.zeros((3, 4), dtype=np.float64, order="C")
    result = hr4.advance_hr4_single_screen(zeros, zeros.copy(order="C"), zeros.copy(order="C"),
        dx=1e-5, dy=1e-5, dt_hydro=1e-6, chi=hr4.HR4_CHI, nu=hr4.HR4_NU,
        n0=1.00027, cfl_limit=1.0, n_steps=2, require_stable=True)
    assert result["steps"] == 2
    assert result["performance"]["backend"] == "numpy"
    for name in j3.FIELDS:
        assert np.isfinite(result[name]).all()


def test_reload_state_origin_is_explicitly_disk_loaded_next1():
    assert j3._window_source_label(False, "J3_NEXT1_DISK_RELOADED") == "J3_NEXT1_DISK_RELOADED"
    assert j3._window_source_label(False) == "J3_NEXT1_memory"


def test_parser_registers_candidate_options_once():
    args = j3._parser().parse_args(["--phase", "J3", "--candidate-descriptor", "d.json",
        "--candidate-admission", "a.json", "--candidate-descriptor-raw-sha256", "abc"])
    assert args.candidate_descriptor == Path("d.json")
    assert args.candidate_admission == Path("a.json")
    assert args.candidate_descriptor_raw_sha256 == "abc"


def test_cross_pulse2_reclassifies_only_verified_distinct_pre_hash_gate(monkeypatch):
    left_pre = {"delta_n": "ld", "vx": "lvx", "vy": "lvy"}
    right_pre = {"delta_n": "rd", "vx": "rvx", "vy": "rvy"}
    left = {"receipt": {"input_PRE_fields_canonical_sha256": left_pre,
        "pre0_canonical_sha256": None, "pre0_expected_canonical_sha256": None,
        "source_canonical_sha256": "source"}}
    right = {"receipt": {"input_PRE_fields_canonical_sha256": right_pre,
        "pre0_canonical_sha256": None, "pre0_expected_canonical_sha256": None,
        "source_canonical_sha256": "source"}}
    errors = ["pre0_canonical_sha256_changed"]
    monkeypatch.setattr(j2, "_compare_stage_files", lambda *_args: {
        "label": "cross", "canonical_exact_match": False,
        "hard_gate_pass": False, "hard_gate_errors": errors})

    result = j3._cross_lineaged_pulse2_compare(left, right, left_pre, right_pre, "cross")
    assert result["pre2_lineage_exception"]["classification"] == (
        "EXPECTED_DISTINCT_LINEAGED_PRE2_INPUT")
    assert result["raw_j2_hard_gate_errors"] == ["pre0_canonical_sha256_changed"]
    assert result["effective_hard_gate_errors"] == []
    assert result["hard_gate_pass"] is True
    assert result["raw_j2_comparison"]["hard_gate_errors"] == errors

    errors[:] = ["pre0_canonical_sha256_changed", "topology_z_axis_changed"]
    result = j3._cross_lineaged_pulse2_compare(left, right, left_pre, right_pre, "cross")
    assert result["effective_hard_gate_errors"] == ["topology_z_axis_changed"]
    assert result["hard_gate_pass"] is False


def test_cross_pulse2_rejects_pre_not_matching_its_own_next1(monkeypatch):
    pre = {"delta_n": "d", "vx": "x", "vy": "y"}
    left = {"receipt": {"input_PRE_fields_canonical_sha256": pre,
        "pre0_canonical_sha256": None, "pre0_expected_canonical_sha256": None,
        "source_canonical_sha256": "source"}}
    right = {"receipt": {"input_PRE_fields_canonical_sha256": pre,
        "pre0_canonical_sha256": None, "pre0_expected_canonical_sha256": None,
        "source_canonical_sha256": "source"}}
    monkeypatch.setattr(j2, "_compare_stage_files", lambda *_args: {
        "hard_gate_pass": True, "hard_gate_errors": []})
    with pytest.raises(j3.J3Error, match="PRE hashes"):
        j3._cross_lineaged_pulse2_compare(left, right, pre,
            {"delta_n": "wrong", "vx": "x", "vy": "y"}, "cross")


def test_chain_and_reload_route_candidate_factory_only_to_candidate_windows(tmp_path, monkeypatch):
    sentinel = object()
    calls = []
    monkeypatch.setattr(j3, "_candidate_factory",
        lambda args: sentinel if args.candidate_descriptor is not None else None)

    def fake_window(context, pre_fields, *, output_root, **kwargs):
        output_root.mkdir(parents=True, exist_ok=False)
        post = {name: np.zeros((8, 8, 8), dtype=np.float64, order="C") for name in j3.FIELDS}
        post_path = output_root / "final_POST.npz"
        np.savez(post_path, **post)
        calls.append(kwargs.get("propagator_factory"))
        return {"status": "COMPLETE_DIAGNOSTIC", "interval_count": 8,
            "final_POST_path": str(post_path),
            "final_POST_canonical_sha256": j3._field_hashes(post)}

    monkeypatch.setattr(j3, "_window", fake_window)
    roots = {}
    for implementation, descriptor in (("baseline", None), ("candidate", tmp_path / "descriptor.json")):
        chain_root = tmp_path / f"{implementation}-chain"
        chain_args = SimpleNamespace(output_root=chain_root, candidate_descriptor=descriptor,
            candidate_descriptor_raw_sha256="test-hash")
        assert j3._smoke_chain_child(chain_args) == 0
        reload_root = tmp_path / f"{implementation}-reload"
        reload_args = SimpleNamespace(output_root=reload_root, candidate_descriptor=descriptor,
            candidate_descriptor_raw_sha256="test-hash",
            next1_path=chain_root / "NEXT1_checkpoint.npz",
            next1_receipt=chain_root / "NEXT1_checkpoint_receipt.json")
        assert j3._smoke_reload_child(reload_args) == 0
        roots[implementation] = (chain_root, reload_root)

    assert calls[:3] == [None, None, None]
    assert calls[3:] == [sentinel, sentinel, sentinel]
    candidate_chain = json.loads((roots["candidate"][0] / "j3_child_receipt.json").read_text())
    candidate_reload = json.loads((roots["candidate"][1] / "j3_child_receipt.json").read_text())
    assert candidate_chain["candidate_factory_window_count"] == 2
    assert candidate_reload["candidate_factory_window_count"] == 1


@pytest.mark.skipif(not (os.name == "posix" and Path("/proc").is_dir()),
                    reason="the process-tree RSS monitor requires Linux /proc")
def test_rss_80_latch_persists_after_sample_falls_to_60_percent(tmp_path, monkeypatch):
    snapshots = iter([81, 60, 60, 60, 60, 60])
    monkeypatch.setattr(j1.common, "tree_snapshot", lambda _pid: {
        "rss": next(snapshots, 60), "cpu": 0.0, "pids": []})
    stage_dir = tmp_path / "monitor-stage"
    stage_dir.mkdir()
    latch_path = stage_dir / "rss_80_percent.latch"
    result = j1._spawn_and_monitor([sys.executable, "-c", "import time; time.sleep(.2)"],
        stage_dir=stage_dir, cwd=tmp_path, env=os.environ.copy(),
        timeout_seconds=3.0, memory_bytes=100, poll_seconds=0.01,
        rss_latch_path=latch_path)
    assert result["exit_code"] == 0
    assert result["rss_80_percent_latched"] is True
    assert latch_path.is_file()
    assert json.loads(latch_path.read_text())["observed_rss_bytes"] == 81

    monkeypatch.setattr(j3.common, "tree_snapshot", lambda _pid: {
        "rss": 60, "cpu": 0.0, "pids": []})
    monkeypatch.setenv("P1R2_RSS_LATCH", str(latch_path))
    assert j3._current_tree_rss_gate()["latch_80_percent"] is True


@pytest.mark.skipif(not (os.name == "posix" and hasattr(os, "getpgid")
                         and Path("/proc").is_dir()),
                    reason="the J3 process-group supervisor is Linux-specific")
def test_worker_batch_launches_eight_children_before_waiting(tmp_path):
    shared = tmp_path / "start-markers"
    shared.mkdir()
    specs = []
    env = os.environ.copy()
    env["J3_BATCH_TEST_DIR"] = str(shared)
    child_code = """
import os
import pathlib
import time
p = pathlib.Path(os.environ['J3_BATCH_TEST_DIR'])
(p / str(os.getpid())).write_text('started')
end = time.monotonic() + 12
while len(list(p.iterdir())) < 8 and time.monotonic() < end:
    time.sleep(.02)
raise SystemExit(0 if len(list(p.iterdir())) == 8 else 4)
"""
    for ordinal in range(8):
        task_dir = tmp_path / f"worker-{ordinal}"
        task_dir.mkdir()
        specs.append({"task_dir": task_dir, "cwd": tmp_path, "env": env,
            "command": [sys.executable, "-c", child_code]})

    result = j3._supervise_worker_batch(specs, timeout_seconds=20.0,
                                        memory_bytes=10**12)
    assert result["status"] == "DONE"
    assert result["worker_processes_launched"] == 8
    assert result["worker_processes_completed"] == 8
    assert result["all_workers_inherited_chain_process_group"] is True
    assert len(list(shared.iterdir())) == 8
    assert all(Path(row["stdout_path"]).is_file() and Path(row["stderr_path"]).is_file()
               for row in result["worker_rows"])


@pytest.mark.skipif(not (os.name == "posix" and hasattr(os, "getpgid")
                         and Path("/proc").is_dir()),
                    reason="the J3 process-group supervisor is Linux-specific")
def test_worker_batch_timeout_reaps_all_launched_children(tmp_path):
    specs = []
    env = os.environ.copy()
    for ordinal in range(8):
        task_dir = tmp_path / f"timeout-{ordinal}"
        task_dir.mkdir()
        specs.append({"task_dir": task_dir, "cwd": tmp_path, "env": env,
            "command": [sys.executable, "-c", "import time; time.sleep(20)"]})

    result = j3._supervise_worker_batch(specs, timeout_seconds=0.3,
                                        memory_bytes=10**12)
    assert result["status"] == "INCOMPLETE"
    assert result["worker_processes_launched"] == 8
    assert all(row["exit_code"] is not None for row in result["worker_rows"])
    assert all(row["stop"] is not None for row in result["worker_rows"])

