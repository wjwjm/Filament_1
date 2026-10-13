from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools.cpu_p1 import cpu_p1_fft as bench
from tools.cpu_p1 import cpu_p1_hydro as hydro


def valid_slurm_env() -> dict[str, str]:
    return {
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NTASKS": "1",
        "SLURM_CPUS_PER_TASK": "96",
        "SLURM_CPUS_ON_NODE": "96",
        "SLURM_JOB_PARTITION": "v6_384",
        "SLURM_JOB_ACCOUNT": "bscc-t",
        "SLURM_JOB_QOS": "normal",
        "UPPE_USE_GPU": "0",
    }


def test_full_allocation_accepts_only_cpu_96_core_contract() -> None:
    bench.validate_full_allocation(valid_slurm_env())
    for key, value in (("SLURM_CPUS_PER_TASK", "48"),
                       ("SLURM_JOB_PARTITION", "other"),
                       ("SLURM_JOB_ACCOUNT", "other"),
                       ("SLURM_JOB_NUM_NODES", "2"),
                       ("SLURM_JOB_GPUS", "1"),
                       ("CUDA_VISIBLE_DEVICES", "0")):
        env = valid_slurm_env()
        env[key] = value
        with pytest.raises(bench.BenchmarkError):
            bench.validate_full_allocation(env)


def test_slurm_receipt_uses_resource_whitelist() -> None:
    env = valid_slurm_env()
    env["SLURM_JOB_TOKEN"] = "secret"
    env["SLURM_UNRELATED"] = "do-not-record"
    recorded = bench.slurm_metadata(env)
    assert recorded["SLURM_JOB_PARTITION"] == "v6_384"
    assert recorded["SLURM_JOB_ACCOUNT"] == "bscc-t"
    assert "SLURM_JOB_TOKEN" not in recorded
    assert "SLURM_UNRELATED" not in recorded


def test_batch_literals_match_confirmed_t0_route() -> None:
    text = Path(bench.__file__).with_suffix(".sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --partition=v6_384" in text
    assert "#SBATCH --account=bscc-t" in text
    assert "#SBATCH --qos=normal" in text
    assert "--constraint" not in text
    assert "FIXED_PYTHON=/publicfs01/fs1-share/soft/amd-soft/miniforge/24.11/bin/python" in text
    assert 'export PYTHONPATH="$BENCH_ROOT/Filament_python:$VENDOR_DIR"' in text
    assert "export PYTHONNOUSERSITE=1" in text
    assert "$PYTHONPATH" not in text

def test_synthetic_array_shape_dtype_layout_and_fingerprint() -> None:
    first = bench.build_synthetic_array((4, 3, 2))
    second = bench.build_synthetic_array((4, 3, 2))
    assert first.shape == (4, 3, 2)
    assert first.dtype == np.complex128
    assert first.flags.c_contiguous
    assert bench.fingerprint(first) == bench.fingerprint(second)


def test_physical_affinity_uses_first_available_core_representatives() -> None:
    info = {
        "physical_core_representatives": [2, 4, 8],
        "topology_source": "test",
    }
    assert bench.select_physical_cpus(2, topology=info) == [2, 4]
    with pytest.raises(bench.BenchmarkError):
        bench.select_physical_cpus(4, topology=info)


def test_output_root_is_created_without_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "run"
    assert bench.create_output_root(target) == target
    with pytest.raises(FileExistsError):
        bench.create_output_root(target)


def test_roundtrip_error_metrics_report_dtype_and_perturbation() -> None:
    value = np.ones((3, 4, 5), dtype=np.complex128)
    exact = bench.error_metrics(value, value.copy())
    changed = value.copy()
    changed[0, 0, 0] += 1e-8
    perturbed = bench.error_metrics(value, changed)
    assert exact["max_abs"] == 0
    assert exact["l2_rel"] == 0
    assert perturbed["max_abs"] > 0
    assert perturbed["max_rel"] > 0


def test_error_metrics_reject_nonfinite_values() -> None:
    value = np.ones((2, 3, 4), dtype=np.complex128)
    value[1, 1, 1] = complex(np.nan, 0)
    with pytest.raises(bench.BenchmarkError, match="NaN or Inf"):
        bench.error_metrics(value, value.copy())


def test_forward_spectrum_error_is_separate_from_roundtrip() -> None:
    value = bench.build_synthetic_array((2, 3, 4))
    reference = np.fft.fft2(value, axes=(-2, -1))
    altered = lambda array: np.fft.fft2(array, axes=(-2, -1)) + 1e-8
    inverse = lambda array: np.fft.ifft2(array, axes=(-2, -1))
    result = bench.timed_pair(value, altered, inverse, reference)
    assert result["roundtrip_error"]["max_abs"] > 0
    assert result["forward_spectrum_error"]["max_abs"] > 0


def test_full_topology_requires_sysfs_physical_core_evidence() -> None:
    assert bench.physical_topology_verified({"topology_source": "sysfs_physical_core"})
    assert not bench.physical_topology_verified({"topology_source": "logical_cpu_fallback"})


def _write_cpu_topology(sysfs: Path, cpu: int, siblings: str, *, core_id: int = 0) -> None:
    topology = sysfs / f"cpu{cpu}" / "topology"
    topology.mkdir(parents=True)
    (topology / "thread_siblings_list").write_text(siblings, encoding="ascii")
    (topology / "physical_package_id").write_text("0", encoding="ascii")
    (topology / "core_id").write_text(str(core_id), encoding="ascii")


def test_topology_uses_siblings_when_multidie_core_ids_repeat(tmp_path: Path) -> None:
    sysfs = tmp_path / "sys"
    for cpu in (0, 1, 48, 49):
        _write_cpu_topology(sysfs, cpu, str(cpu), core_id=cpu % 2)
    info = bench.cpu_topology([0, 1, 48, 49], sysfs=sysfs)
    assert info["physical_core_representatives"] == [0, 1, 48, 49]
    assert info["physical_core_count"] == 4
    assert info["thread_sibling_groups"] == [[0], [1], [48], [49]]
    assert bench.physical_topology_verified(info)


def test_topology_collapses_sysfs_smt_sibling_pairs(tmp_path: Path) -> None:
    sysfs = tmp_path / "sys"
    for cpu, siblings in ((0, "0,48"), (48, "0,48"), (1, "1,49"), (49, "1,49")):
        _write_cpu_topology(sysfs, cpu, siblings)
    info = bench.cpu_topology([0, 1, 48, 49], sysfs=sysfs)
    assert info["physical_core_representatives"] == [0, 1]
    assert info["thread_sibling_groups"] == [[0, 48], [1, 49]]


def test_inconsistent_sysfs_sibling_lists_do_not_verify_physical_cores(tmp_path: Path) -> None:
    sysfs = tmp_path / "sys"
    _write_cpu_topology(sysfs, 0, "0,1")
    _write_cpu_topology(sysfs, 1, "1")
    info = bench.cpu_topology([0, 1], sysfs=sysfs)
    assert info["topology_source"] == "logical_cpu_fallback"
    assert info["topology_method"] == "unavailable"
    assert not bench.physical_topology_verified(info)


def test_skipped_physical_core_slots_make_smoke_incomplete(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "incomplete-run"
    monkeypatch.setattr(bench, "environment", lambda *args: {
        "physical_core_representatives": [], "topology_source": "logical_cpu_fallback"})
    args = argparse.Namespace(smoke=True, smoke_shape=(2, 2, 2), deadline_seconds=30,
                              memory_limit_bytes=1024**3, vendor_dir=None,
                              output_root=str(root))
    assert bench.run_parent(args) == 1
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "INCOMPLETE"

def test_tiny_subprocess_smoke_preserves_outputs(tmp_path: Path) -> None:
    root = tmp_path / "tiny-run"
    script = Path(bench.__file__).resolve()
    command = [
        sys.executable, str(script), "--smoke", "--output-root", str(root),
        "--smoke-shape", "6", "5", "4", "--deadline-seconds", "180",
        "--memory-limit-bytes", str(1024**3),
    ]
    run = subprocess.run(command, capture_output=True, text=True, timeout=240, check=False)
    assert run.returncode == 0, run.stderr + run.stdout
    result = json.loads(run.stdout.strip().splitlines()[-1])
    assert result["status"] == "COMPLETE"
    assert (root / "environment.json").is_file()
    assert (root / "affinity.json").is_file()
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["input_kind"] == "DETERMINISTIC_SYNTHETIC_ONLY"
    assert status["scientific_inputs_used"] is False
    with (root / "performance_summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all(row["threading_status"] != "" or row["status"].startswith("SKIPPED")
               for row in rows)
    records = [json.loads(line) for line in
               (root / "benchmark_results.jsonl").read_text(encoding="utf-8").splitlines()]
    done = [row for row in records if row["status"] == "DONE"]
    assert len(done) >= 2
    assert len({row["array_fingerprint_sha256"] for row in done}) == 1
    assert all(len(row['progress_samples']) == 4 for row in done)
    assert all(row['repetitions'][-1]['forward_spectrum_error']['max_abs'] >= 0 for row in done)
    # A second invocation cannot replace the completed result directory.
    again = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    assert again.returncode == 2
    assert json.loads((root / "status.json").read_text(encoding="utf-8")) == status


def test_hydro_flat_task_sampling_and_reuse_contract() -> None:
    p1 = hydro.worker_plan(1)
    assert [ordinal for worker in p1 for ordinal in worker["task_ordinals"]] == [0, 12, 24, 36]
    p8 = hydro.worker_plan(8)
    p8_ordinals = [ordinal for worker in p8 for ordinal in worker["task_ordinals"]]
    assert len(set(p8_ordinals)) == 32
    assert p8_ordinals == [int(i * 48 / 32) for i in range(32)]
    for workers, expected_unique, expected_reused in ((24, 48, 48), (48, 48, 144), (96, 48, 336)):
        ordinals = [ordinal for worker in hydro.worker_plan(workers)
                    for ordinal in worker["task_ordinals"]]
        assert len(ordinals) == workers * 4
        assert len(set(ordinals)) == expected_unique
        assert len(ordinals) - len(set(ordinals)) == expected_reused
    assert hydro.input_scaling_class(8) == "REAL_INPUT_FIXED_TASKS_PER_WORKER"
    assert hydro.input_scaling_class(24) == "REPEATED_REAL_INPUT_SCALING_ONLY"


def test_hydro_invoke_preserves_frozen_1000_step_contract() -> None:
    fields = {name: np.ones((2, 3), dtype=np.float64) for name in hydro.FIELDS}
    captured = {}

    def fake_advance(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {**dict(zip(hydro.FIELDS, args)), "steps": 1000,
                "performance": {"backend": "numpy"}, "observables": {}}

    manifest = {"dx_m": 1e-5, "dy_m": 2e-5,
                "hydro": {"dt_hydro": 1e-6, "chi": 2.17e-5, "nu": 1.5e-5,
                          "n0": 1.00027, "gravity_x": 0.0, "gravity_y": -9.81,
                          "cfl_limit": 1.0}}
    result = hydro.invoke_hydro(fake_advance, fields, manifest)
    assert result["steps"] == 1000
    assert captured["kwargs"]["n_steps"] == 1000
    assert captured["kwargs"]["require_stable"] is True
    assert captured["kwargs"]["dx"] == 1e-5
    assert captured["kwargs"] == {
        "dx": 1e-5, "dy": 2e-5, "dt_hydro": 1e-6, "chi": 2.17e-5,
        "nu": 1.5e-5, "n0": 1.00027, "gravity_x": 0.0,
        "gravity_y": -9.81, "cfl_limit": 1.0, "n_steps": 1000,
        "require_stable": True,
    }


def test_hydro_manifest_requires_accepted_raw_sha_and_frozen_parameters(tmp_path: Path) -> None:
    manifest = {
        "schema": hydro.MANIFEST_SCHEMA, "identity": hydro.INPUT_IDENTITY,
        "diagnostic_only": True,
        "config_sha256": "eaec83ad326a29af95881912db76a7e5c26943dec9e0d6876ca9c9cc2a100c22",
        "source_manifest_sha256": "3e2c7557e5e8027ac1d9fb42411fe885fe3ea13bef9ee60e91538462adaf32d2",
        "dtype": "float64", "shape": [351, 301], "unique_real_screens": 48,
        "dx_m": 1e-5, "dy_m": 1e-5, "hydro": dict(hydro.EXPECTED_HYDRO_PARAMETERS),
        "records": [],
    }
    for namespace in ("current", "post", "next"):
        for ordinal in range(48):
            manifest["records"].append({
                "namespace": namespace, "ordinal": ordinal,
                "relative_path": f"{namespace}/{ordinal:03d}.npz",
                "hash_scope": "raw_bytes", "sha256": "a" * 64,
                "size_bytes": 100, "field_sha256": {name: "b" * 64 for name in hydro.FIELDS},
                "screen_id": f"source_index_{ordinal:05d}", "z_m": ordinal * 1e-3,
            })
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    path = input_root / "candidate_input_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    accepted_hash = lambda _path: hydro.EXPECTED_MANIFEST_RAW_SHA256
    loaded, records = hydro.load_manifest(path, input_root, accepted_hash)
    assert loaded["raw_sha256"] == hydro.EXPECTED_MANIFEST_RAW_SHA256
    assert len(records["post"]) == 48
    with pytest.raises(hydro.HydroError, match="manifest raw SHA256 mismatch"):
        hydro.load_manifest(path, input_root, lambda _path: "0" * 64)
    manifest["hydro"]["chi"] = 1.0
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(hydro.HydroError, match="hydro_parameters"):
        hydro.load_manifest(path, input_root, accepted_hash)


def test_hydro_a0_repeat_gate_requires_exact_canonical_hashes_and_keeps_metrics() -> None:
    hashes = {name: name * 64 for name in hydro.FIELDS}
    receipts = [
        {"output": {"canonical_field_sha256": dict(hashes)},
         "historical_next_field_comparison": {"delta_n": {"max_abs": 1e-9}}},
        {"output": {"canonical_field_sha256": dict(hashes)},
         "cpu_repeat_comparison": {"delta_n": {"max_abs": 0.0}}},
    ]
    assert hydro._a0_repeat_gate(receipts)["hash_match"]
    receipts[1]["output"]["canonical_field_sha256"]["vx"] = "c" * 64
    assert not hydro._a0_repeat_gate(receipts)["hash_match"]
    assert receipts[0]["historical_next_field_comparison"]["delta_n"]["max_abs"] == 1e-9


def test_hydro_a1_start_failure_terminates_already_started_processes() -> None:
    class FakeProcess:
        def __init__(self, fail=False):
            self.fail = fail
            self.alive = False
            self.terminated = False
            self.joined = False

        def start(self):
            if self.fail:
                raise RuntimeError("synthetic process start failure")
            self.alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

        def join(self, timeout=None):
            self.joined = True

        def kill(self):
            self.alive = False

    class FakeContext:
        def __init__(self):
            self.processes = [FakeProcess(), FakeProcess(fail=True)]

        def Process(self, *args, **kwargs):
            return self.processes.pop(0)

    context = FakeContext()
    already_started = context.processes[0]
    with pytest.raises(RuntimeError, match="synthetic process start failure"):
        hydro._start_processes(context, [((), {}), ((), {})])
    assert already_started.terminated and already_started.joined


def test_hydro_supervisor_cleans_only_failed_cases_process_group(tmp_path: Path, monkeypatch) -> None:
    called = []

    class FakeProcess:
        pid = 43127

        def __init__(self, command, **kwargs):
            assert kwargs["start_new_session"] is True

        def poll(self):
            return 17

        def wait(self):
            return 17

    monkeypatch.setattr(hydro.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(hydro, "_cleanup_case_process_group", called.append)
    monkeypatch.setattr(hydro.os, "name", "posix")
    result = hydro._supervise_case(["fake"], tmp_path, "case-fail", hydro.time.monotonic() + 5,
                                   1024**3)
    assert result["exit_code"] == 17
    assert called == [43127]


def test_hydro_case_command_starts_and_rejects_invalid_manifest_before_argparse(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    manifest = input_root / "candidate_input_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    output_root = tmp_path / "parent-output-not-created"
    vendor_dir = tmp_path / "vendor"
    vendor_dir.mkdir()
    (vendor_dir / "threadpoolctl.py").write_text(
        "from contextlib import nullcontext\n"
        "__version__ = '3.6.0'\n"
        "def threadpool_limits(limits=1): return nullcontext()\n",
        encoding="utf-8")
    args = argparse.Namespace(input_root=str(input_root), manifest=str(manifest),
                              vendor_dir=str(vendor_dir))
    command = hydro._make_child_command(
        args, output_root, "A0", "A0", 1, [0], hydro.time.monotonic() + 30)

    result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)

    assert result.returncode == 2
    assert "required: --output-root" not in result.stderr
    failure = json.loads(result.stderr.strip().splitlines()[-1])
    assert failure["status"] == "FAIL"
    assert "input manifest raw SHA256 mismatch" in failure["error"]
    assert not output_root.exists()


def test_hydro_npz_loader_checks_raw_canonical_hashes_and_metadata(tmp_path: Path) -> None:
    from KHz_filament.hr4e_timestep import sha256_array
    from tools.hpc_ops.provenance_v2 import raw_sha256_file

    root = tmp_path / "input"
    root.mkdir()
    shape = hydro.EXPECTED_SHAPE
    arrays = {"delta_n": np.zeros(shape, dtype=np.float64),
              "vx": np.ones(shape, dtype=np.float64),
              "vy": -np.ones(shape, dtype=np.float64)}
    hashes = {name: sha256_array(value) for name, value in arrays.items()}
    metadata = {"schema": hydro.STREAM_SCHEMA, "namespace": "POST", "ordinal": 24,
                "screen_id": "source_index_08022", "z_m": 0.802249999999928,
                "shape": list(shape), "dtype": "float64", "field_sha256": hashes}
    path = root / "screen.npz"
    with path.open("xb") as stream:
        np.savez(stream, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    record = {"ordinal": 24, "namespace": "post", "screen_id": "source_index_08022",
              "z_m": metadata["z_m"], "relative_path": "screen.npz",
              "sha256": raw_sha256_file(path), "hash_scope": "raw_bytes",
              "size_bytes": path.stat().st_size, "field_sha256": hashes}
    loaded, receipt = hydro.load_screen(record, root, raw_sha256_file, sha256_array)
    assert all(np.array_equal(loaded[name], arrays[name]) for name in hydro.FIELDS)
    assert receipt["raw_sha256"] == record["sha256"]
    assert receipt["canonical_field_sha256"] == hashes
    record["sha256"] = "0" * 64
    with pytest.raises(hydro.HydroError, match="raw SHA256 mismatch"):
        hydro.load_screen(record, root, raw_sha256_file, sha256_array)


def test_hydro_sbatch_literals_and_cpu_only_contract() -> None:
    path = Path(hydro.__file__).with_suffix(".sbatch")
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --partition=v6_384" in text
    assert "#SBATCH --account=bscc-t" in text
    assert "#SBATCH --qos=normal" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --cpus-per-task=96" in text
    assert "#SBATCH --time=01:00:00" in text
    assert "--constraint" not in text
    assert "FIXED_PYTHON=/publicfs01/fs1-share/soft/amd-soft/miniforge/24.11/bin/python" in text
    assert "export UPPE_USE_GPU=0" in text
    assert "--deadline-seconds 3300 --a0-deadline-seconds 900" in text


def test_hydro_synthetic_fake_callable_subprocess_smoke_is_nonoverwriting(tmp_path: Path) -> None:
    root = tmp_path / "hydro-smoke"
    command = [sys.executable, str(Path(hydro.__file__).resolve()), "--smoke",
               "--output-root", str(root), "--smoke-shape", "5", "4"]
    first = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    assert first.returncode == 0, first.stderr + first.stdout
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "COMPLETE"
    assert status["input_kind"] == "SYNTHETIC_ENGINEERING_SMOKE_ONLY"
    assert status["scientific_inputs_used"] is False
    assert status["fake_callable_used"] is True
    assert (root / "synthetic_smoke.npz").is_file()
    second = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    assert second.returncode == 2
    assert json.loads((root / "status.json").read_text(encoding="utf-8")) == status

