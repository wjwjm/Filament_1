from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "isaacs_raman_closure" / "phase8b_preflight"


def _load_smoke_summary_module():
    path = ROOT / "tools" / "summarize_phase8b_full_size_smoke.py"
    spec = importlib.util.spec_from_file_location("phase8b_smoke_summary", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_smoke_slurm_resources_and_serial_case_contract_are_explicit():
    script = (ROOT / "tools" / "phase8b_full_size_smoke.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH -p gpu" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem" not in script
    assert "site policy per requested GPU" in script
    assert "PHASE8B_SCHEDULER_MEMORY_MB" in script
    assert "#SBATCH -t 00:30:00" in script
    assert "PHASE8B_SMOKE_CASE" in script
    assert "--steps 20" in script


def test_smoke_tool_help_does_not_require_local_cupy():
    subprocess.run([
        sys.executable,
        str(ROOT / "tools" / "run_phase8b_full_size_smoke.py"),
        "--help",
    ], check=True, capture_output=True, text=True)


def test_smoke_tool_records_on_off_physical_contract_gates():
    script = (ROOT / "tools" / "run_phase8b_full_size_smoke.py").read_text(encoding="utf-8")
    for gate in (
        "steps_complete",
        "energy_closure_finite",
        "target_loss_nonzero",
        "actual_loss_expected",
        "applied_rhs_expected",
        "convolution_reuse",
    ):
        assert f'"{gate}"' in script


def test_recorded_full_size_smokes_pass_resource_and_on_off_contracts():
    module = _load_smoke_summary_module()
    evidence = RESULTS / "smoke_evidence"
    load = lambda name: json.loads((evidence / name).read_text(encoding="utf-8"))
    metrics, runtime = module.summarize(
        load("phase8b_full_size_smoke_on_metrics.json"),
        load("phase8b_full_size_smoke_off_metrics.json"),
        load("phase8b_full_size_smoke_on_config_audit.json"),
        load("phase8b_full_size_smoke_off_config_audit.json"),
        scheduler_memory_mb=126000,
    )
    assert all(metrics["gates"].values())
    assert metrics["on_contract"]["two_convolutions_per_operator_substep"]
    assert metrics["off_contract"]["applied_rhs_zero"]
    assert metrics["off_contract"]["actual_loss_zero"]
    assert runtime["runtime_gate"]
