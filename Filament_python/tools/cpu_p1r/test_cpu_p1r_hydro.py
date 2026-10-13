"""Focused contract tests for the P1R CPU HR-4 diagnostic adapter."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pytest

from tools.cpu_p1r import cpu_p1r_hydro as p1r
from tools.cpu_p1 import cpu_p1_hydro as p1


def _inputs():
    manifest = {"hydro": dict(p1.EXPECTED_HYDRO_PARAMETERS), "dx_m": p1.EXPECTED_DX_M,
                "dy_m": p1.EXPECTED_DY_M, "config_sha256": "c" * 64,
                "source_manifest_sha256": "s" * 64}
    details = {"manifest": manifest, "raw_sha256": "m" * 64}
    records = {"post": {ordinal: {"sha256": f"{ordinal:064x}",
                                   "field_sha256": {name: f"{ordinal + i:064x}" for i, name in enumerate(p1.FIELDS)},
                                   "screen_id": f"screen-{ordinal:02d}", "z_m": float(ordinal)}
                            for ordinal in range(48)}}
    return details, records


def test_workload_has_frozen_repetition_major_1n_and_2n_assignments():
    details, records = _inputs()
    workload = p1r.build_workload(details, records)
    assert workload["hash_scope"] == "classified_by_record"
    assert workload["task_count"] == 768
    assert [(t["repetition"], t["ordinal"]) for t in workload["tasks"][:50]] == [
        (0, ordinal) for ordinal in range(48)] + [(1, 0), (1, 1)]
    assert all(t["assignment_1n"]["worker_index"] == t["index"] % 96 for t in workload["tasks"])
    assert all(t["assignment_2n"]["rank"] == (t["index"] % 192) // 96 and
               t["assignment_2n"]["worker_index"] == t["index"] % 96 for t in workload["tasks"])
    assert all(len(queue) == 8 for queue in p1r.worker_tasks(workload, 1, 0))
    assert all(len(queue) == 4 for queue in p1r.worker_tasks(workload, 2, 0))
    assert all(len(queue) == 4 for queue in p1r.worker_tasks(workload, 2, 1))


def test_frozen_workload_rejects_mutation(tmp_path: Path):
    details, records = _inputs()
    workload = p1r.build_workload(details, records)
    path = tmp_path / "WORKLOAD_768.json"
    path.write_text(json.dumps(workload), encoding="utf-8")
    verified, digest = p1r.validate_workload(path, details, records,
                                            lambda item: hashlib.sha256(item.read_bytes()).hexdigest())
    assert verified == workload and len(digest) == 64
    workload["tasks"][0]["ordinal"] = 47
    path.write_text(json.dumps(workload), encoding="utf-8")
    with pytest.raises(p1r.P1RError, match="frozen workload differs"):
        p1r.validate_workload(path, details, records,
                              lambda item: hashlib.sha256(item.read_bytes()).hexdigest())


def test_allocation_gate_requires_two_nodes_192_cpu_320g_and_no_gpu():
    env = {"SLURM_JOB_NUM_NODES": "2", "SLURM_NTASKS": "2",
           "SLURM_NTASKS_PER_NODE": "1", "SLURM_CPUS_PER_TASK": "96",
           "SLURM_JOB_PARTITION": "v6_384", "SLURM_JOB_ACCOUNT": "bscc-t",
           "SLURM_JOB_QOS": "normal", "SLURM_MEM_PER_NODE": "327680",
           "UPPE_USE_GPU": "0"}
    assert p1r.validate_allocation_2n(env)["SLURM_JOB_NUM_NODES"] == "2"
    for key, value in (("SLURM_JOB_NUM_NODES", "1"), ("SLURM_MEM_PER_NODE", "320000"),
                       ("SLURM_JOB_GPUS", "1"), ("UPPE_USE_GPU", "1")):
        rejected = dict(env)
        rejected[key] = value
        with pytest.raises(p1r.P1RError):
            p1r.validate_allocation_2n(rejected)


def test_hr4_call_keeps_frozen_1000_step_numpy_contract():
    observed = {}
    manifest = {"hydro": dict(p1.EXPECTED_HYDRO_PARAMETERS), "dx_m": 1e-5, "dy_m": 1e-5}
    fields = {name: np.zeros((2, 3), dtype=np.float64) for name in p1.FIELDS}

    def fake_advance(delta_n, vx, vy, **kwargs):
        observed["kwargs"] = kwargs
        observed["shapes"] = (delta_n.shape, vx.shape, vy.shape)
        return {"steps": 1000, "performance": {"backend": "numpy"}}

    result = p1r.validate_numpy_call(fake_advance, fields, manifest)
    assert result["steps"] == 1000
    assert observed["shapes"] == ((2, 3),) * 3
    assert observed["kwargs"] == {"dx": 1e-5, "dy": 1e-5,
                                  "dt_hydro": p1.EXPECTED_HYDRO_PARAMETERS["dt_hydro"],
                                  "chi": p1.EXPECTED_HYDRO_PARAMETERS["chi"],
                                  "nu": p1.EXPECTED_HYDRO_PARAMETERS["nu"],
                                  "n0": p1.EXPECTED_HYDRO_PARAMETERS["n0"],
                                  "gravity_x": p1.EXPECTED_HYDRO_PARAMETERS["gravity_x"],
                                  "gravity_y": p1.EXPECTED_HYDRO_PARAMETERS["gravity_y"],
                                  "cfl_limit": p1.EXPECTED_HYDRO_PARAMETERS["cfl_limit"],
                                  "n_steps": 1000, "require_stable": True}


def test_cross_stage_gate_stops_on_hash_or_observable_difference(tmp_path: Path, monkeypatch):
    import tools.cpu_p1r.cpu_p1r_hydro as module

    root = tmp_path
    for stage, raw_hash in (("base", "a"), ("next", "b")):
        task_root = root / "stages" / stage / "node-00" / "tasks"
        task_root.mkdir(parents=True)
        (task_root / "task-0000.json").write_text(json.dumps({
            "status": "DONE", "classification": module.CLASSIFICATION, "task_id": "task-0000",
            "rank": 0, "worker_index": 0,
            "output": {"canonical_field_sha256": {name: raw_hash for name in p1.FIELDS}},
            "observables": {"rho": 1.0 if stage == "base" else 2.0}}), encoding="utf-8")
    fields = {name: np.ones((2, 3), dtype=np.float64) for name in p1.FIELDS}
    monkeypatch.setattr(module, "_load_valid_output", lambda *_args: fields)
    gate = module._cross_compare_with_gate(root, "base", "next", ["task-0000"],
                                           lambda _path: "unused", lambda _array: "unused")
    assert gate["canonical_hash_differences"] == 1
    assert gate["observable_differences"] == 1
    assert gate["continue_gate"].startswith("NUMERICAL_DIFFERENCE")


def test_srun_entry_uses_exact_allocation_and_one_named_step():
    command = p1r._srun_command(2, "nodeA,nodeB", p1r.FIXED_PYTHON,
                                Path("runner.py"), ["--mode", "probe"], "p1r-probe")
    assert "--exact" in command and "--kill-on-bad-exit=1" in command
    assert command[command.index("--nodelist") + 1] == "nodeA,nodeB"
    assert command[command.index("--job-name") + 1] == "p1r-probe"


def test_step_query_failure_is_not_interpreted_as_cleaned(monkeypatch):
    import subprocess
    monkeypatch.setattr(p1r.subprocess, "run", lambda *_args, **_kwargs:
                        subprocess.CompletedProcess([], 1, stdout="", stderr="squeue failed"))
    assert p1r._step_absent("12345.0") is False


def test_srun_timeout_persists_file_logs_and_returns_partial_receipt(tmp_path: Path, monkeypatch):
    captured = {}

    class RunningProcess:
        args = ["srun", "--job-name", "p1r-timeout"]
        pid = 4242
        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        return RunningProcess()

    monkeypatch.setattr(p1r.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(p1r, "_find_named_step", lambda _command: None)
    monkeypatch.setattr(p1r, "_terminate_srun", lambda *_args: {
        "srun_exited_after_stop": False, "step_id": "UNAVAILABLE",
        "targeted_step_cancel_issued": False, "remote_step_teardown_verified": False})
    receipt = p1r._run_srun(["srun"], tmp_path, "timeout", time.monotonic() - 1)
    assert receipt["outcome"] == "LIMIT_REACHED" and receipt["exit_code"] == -999
    assert receipt["remote_step_teardown_verified"] is False
    assert (tmp_path / receipt["stdout_path"]).is_file()
    assert (tmp_path / receipt["stderr_path"]).is_file()
    assert captured["stdout"] is not p1r.subprocess.PIPE and captured["stderr"] is not p1r.subprocess.PIPE


def test_srun_step_cleanup_uses_bounded_waits(monkeypatch):
    import subprocess
    timeouts = []

    class StoppingProcess:
        args = ["srun", "--job-name", "p1r-test"]
        returncode = None
        def poll(self):
            return self.returncode
        def wait(self, timeout):
            timeouts.append(timeout)
            if timeout in (75, 10):
                raise subprocess.TimeoutExpired("srun", timeout)
            return self.returncode
        def kill(self):
            self.returncode = -9

    process = StoppingProcess()
    monkeypatch.setattr(p1r.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(p1r.os, "killpg", lambda _pid, sig: process.kill() if sig == 9 else None, raising=False)
    monkeypatch.setattr(p1r.p1, "_cleanup_case_process_group", lambda *_args: None)
    monkeypatch.setattr(p1r.subprocess, "run", lambda args, **_kwargs:
                        subprocess.CompletedProcess(args, 0, stdout="", stderr=""))
    result = p1r._terminate_srun(process, 4242, "12345.9")
    assert timeouts == [75, 10, 5]
    assert result["srun_exited_after_stop"] is True
    assert result["remote_step_teardown_verified"] is True


def test_probe_fails_closed_when_step_teardown_is_unverified(tmp_path: Path, monkeypatch):
    (tmp_path / "process_logs").mkdir()
    monkeypatch.setattr(p1r, "_srun_command", lambda *_args: ["srun", "--job-name", "probe"])
    monkeypatch.setattr(p1r, "_run_srun", lambda *_args: {
        "exit_code": 0, "outcome": None, "remote_step_teardown_verified": False})
    with pytest.raises(p1r.P1RError, match="two-node srun probe failed"):
        p1r._run_probe(type("Args", (), {})(), tmp_path, time.monotonic() + 10)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="controller integration fixture requires Linux")
def test_controller_publishes_each_exclusive_stage_receipt_once(tmp_path: Path, monkeypatch):
    module = p1r
    details, records = _inputs()
    details["manifest"]["identity"] = "SYNTHETIC_CONTROLLER_FIXTURE"
    workload = module.build_workload(details, records)
    input_root = tmp_path / "input"
    input_root.mkdir()
    output_root = tmp_path / "controller-output"
    monkeypatch.setattr(module, "_validate_interpreter", lambda: None)
    monkeypatch.setattr(module, "validate_allocation_2n", lambda _env: {"fixture": True})
    monkeypatch.setattr(module, "_allocation_start", lambda _args: time.time())
    monkeypatch.setattr(module.common, "add_vendor", lambda _path: None)
    monkeypatch.setattr(module.p1, "load_manifest", lambda *_args: (details, records))
    monkeypatch.setattr(module, "validate_workload", lambda *_args: (workload, "w" * 64))
    monkeypatch.setattr(module, "_run_probe", lambda _args, root, _deadline: (
        {"rank_to_hostname": {"0": "node-a", "1": "node-b"}}, {"exit_code": 0}))
    monkeypatch.setattr(module.p1, "_load_science", lambda: (
        object(), lambda _array: "field", lambda _path: "raw", type("ThreadPoolFixture", (), {"__version__": "3.6.0"})(), {"backend": "numpy"}))
    monkeypatch.setattr(module, "_environment", lambda *_args: {})

    def fake_stage(_args, root, _probe, stage, nodes, layout, _budget, _deadline,
                   _raw_hash, _canonical_hash):
        (root / "stages" / stage).mkdir()
        return {"stage": stage, "node_count": nodes, "layout": layout,
                "status": "COMPLETE", "tasks_planned": module.TASK_COUNT,
                "tasks_completed": module.TASK_COUNT, "coordinator_service_wall_seconds": 1.0}

    monkeypatch.setattr(module, "_stage_run", fake_stage)
    monkeypatch.setattr(module, "_append_stage_csv", lambda *_args: 0)
    monkeypatch.setattr(module, "_cross_compare_with_gate", lambda *_args: {
        "continue_gate": "CONTINUE", "status": "COMPLETE"})
    args = type("Args", (), {"input_root": str(input_root), "manifest": str(tmp_path / "manifest.json"),
                              "workload": str(tmp_path / "WORKLOAD_768.json"),
                              "output_root": str(output_root), "vendor_dir": None,
                              "allocation_start_epoch": str(time.time()), "memory_limit_bytes": module.MEMORY_LIMIT_BYTES,
                              "include_h2": False})()
    assert module.run_controller(args) == 0
    for stage, _nodes, _layout in module.STAGES:
        path = output_root / "stages" / f"{stage}.json"
        assert path.is_file()
        assert json.loads(path.read_text(encoding="utf-8"))["status"] == "COMPLETE"
    assert json.loads((output_root / "status.json").read_text(encoding="utf-8"))["status"] == "COMPLETE"


def test_probe_maps_fqdn_to_verified_slurm_nodename_and_rank_order(tmp_path, monkeypatch):
    import subprocess
    nodes = ["cb0108", "cc0807"]
    for rank, node in enumerate(nodes):
        (tmp_path / f"probe-rank-{rank}.json").write_text(json.dumps({
            "record_type": "P1R_NODE_PROBE", "rank": rank,
            "hostname": node + ".para.bscc", "slurm_node_name": node}), encoding="utf-8")
    monkeypatch.setenv("SLURM_JOB_NODELIST", ",".join(nodes))
    monkeypatch.setattr(p1r.subprocess, "run", lambda *_a, **_k:
                        subprocess.CompletedProcess([], 0, stdout="cb0108\ncc0807\n"))
    probe = p1r._validate_probe_dir(tmp_path)
    assert probe["rank_to_slurm_node"] == {"0": "cb0108", "1": "cc0807"}
    assert probe["rank_to_hostname"]["0"] == "cb0108.para.bscc"
    row = json.loads((tmp_path / "probe-rank-0.json").read_text())
    row["slurm_node_name"] = "cc0807"
    (tmp_path / "probe-rank-0.json").write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(p1r.P1RError, match="NodeName/rank order"):
        p1r._validate_probe_dir(tmp_path)


@pytest.mark.parametrize("nodes,expected", [(1, "cb0108"), (2, "cb0108,cc0807")])
def test_stage_nodelist_uses_scheduler_names_not_fqdn(tmp_path, monkeypatch, nodes, expected):
    from types import SimpleNamespace
    (tmp_path / "stages").mkdir()
    probe = {"rank_to_hostname": {"0": "cb0108.para.bscc", "1": "cc0807.para.bscc"},
             "rank_to_slurm_node": {"0": "cb0108", "1": "cc0807"}}
    def capture(n, hosts, *_args):
        assert n == nodes and hosts == expected
        raise RuntimeError("captured_before_launch")
    monkeypatch.setattr(p1r, "_srun_command", capture)
    args = SimpleNamespace(input_root="inputs", manifest="manifest", workload="workload",
                           vendor_dir="vendor", memory_limit_bytes=p1r.MEMORY_LIMIT_BYTES)
    with pytest.raises(RuntimeError, match="captured_before_launch"):
        p1r._stage_run(args, tmp_path, probe, "fixture", nodes, "H1", 10,
                      time.monotonic()+10, lambda _: "raw", lambda _: "canonical")


def test_allocation_probe_exercises_explicit_1n_and_2n_stage_placement(tmp_path, monkeypatch):
    (tmp_path / "process_logs").mkdir()
    probe = {"rank_to_hostname": {"0": "cb0108.para.bscc", "1": "cc0807.para.bscc"},
             "rank_to_slurm_node": {"0": "cb0108", "1": "cc0807"}}
    observed = []
    def run(command, _logs, name, _deadline):
        nodes = int(command[command.index("--nodes")+1])
        hostlist = command[command.index("--nodelist")+1] if "--nodelist" in command else None
        observed.append((nodes, hostlist))
        target = Path(command[command.index("--probe-dir")+1])
        for rank in range(nodes):
            (target / f"probe-rank-{rank}.json").write_text(json.dumps({
                "rank": rank, "hostname": probe["rank_to_hostname"][str(rank)],
                "slurm_node_name": probe["rank_to_slurm_node"][str(rank)]}), encoding="utf-8")
        return {"exit_code": 0, "outcome": None, "remote_step_teardown_verified": True}
    monkeypatch.setattr(p1r, "_run_srun", run)
    monkeypatch.setattr(p1r, "_validate_probe_dir", lambda _path: probe)
    p1r._run_probe(object(), tmp_path, time.monotonic()+10)
    assert observed == [(2, None), (1, "cb0108"), (2, "cb0108,cc0807")]
    assert (tmp_path / "explicit-1n-probe.json").is_file()
    assert (tmp_path / "explicit-2n-probe.json").is_file()
