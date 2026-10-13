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

