from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pytest

from tools.cpu_p1r2 import cpu_p1r2_j3 as j3


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

