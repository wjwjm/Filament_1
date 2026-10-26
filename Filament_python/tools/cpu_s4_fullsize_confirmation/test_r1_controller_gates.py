"""Small subprocess checks for the explicit R1 lifecycle gates."""
from __future__ import annotations

import os
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from . import controller, entry


def test_r1_entry_accepts_the_new_node_receipt_status():
    entry.verify_node_gate_receipt(
        {"status": "PASS_FULLSIZE_NODE_SOURCE_INPUT_AND_REFERENCE_METADATA_GATES",
         "job_id": "UNIT_RECEIPT_ONLY"}, "UNIT_RECEIPT_ONLY")


@pytest.mark.parametrize("status,job", (
    ("PASS_FULLSIZE_NODE_SOURCE_AND_INPUT_GATES", "UNIT_RECEIPT_ONLY"),
    ("UNKNOWN_STATUS", "UNIT_RECEIPT_ONLY"),
    ("PASS_FULLSIZE_NODE_SOURCE_INPUT_AND_REFERENCE_METADATA_GATES", "WRONG_UNIT_RECEIPT"),
))
def test_r1_entry_rejects_old_unknown_or_wrong_job_node_receipts(status, job):
    with pytest.raises(RuntimeError, match="ACTUAL_FULLSIZE_NODE_GATE_REQUIRED"):
        entry.verify_node_gate_receipt({"status": status, "job_id": job}, "UNIT_RECEIPT_ONLY")


def _run(code: str, *, optimize: bool = False, env_update: dict[str, str] | None = None):
    env = os.environ.copy()
    if env_update:
        env.update(env_update)
    argv = [sys.executable]
    if optimize:
        argv.append("-O")
    argv.extend(("-s", "-B", "-c", code))
    return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", env=env, check=False)


def test_plan_rejects_optimized_subprocess_before_reading_plan(tmp_path: Path):
    code = (
        "from pathlib import Path; "
        "from tools.cpu_s4_fullsize_confirmation import controller; "
        "controller._plan(Path('definitely_missing_plan.json'))"
    )
    result = _run(code, optimize=True)
    assert result.returncode != 0
    assert "FULL_OPTIMIZATION_FORBIDDEN" in (result.stdout + result.stderr)


def test_plan_rejects_pythonoptimize_environment_before_reading_plan(tmp_path: Path):
    code = (
        "from pathlib import Path; "
        "from tools.cpu_s4_fullsize_confirmation import controller; "
        "controller._plan(Path('definitely_missing_plan.json'))"
    )
    result = _run(code, env_update={"PYTHONOPTIMIZE": "1"})
    assert result.returncode != 0
    assert "FULL_OPTIMIZATION_FORBIDDEN" in (result.stdout + result.stderr)


def _valid_r1_plan(tmp_path: Path) -> dict:
    reference = tmp_path / "BOUNDARY_RECEIPT.json"
    raw = b"reference"
    reference.write_bytes(raw)
    return {
        "task": controller.R1_TASK,
        "controller_schema": controller.SCHEMA,
        "execution_domain": "FULLSIZE_CONFIRMATION",
        "shape": list(controller.FULL_SHAPE),
        "intervals": 8,
        "warmup_intervals_per_formal_case": 0,
        "max_submissions": 2,
        "allocated_core_hours_limit": 144,
        "physical_cpus": 96,
        "nodes": 1,
        "gpus": 0,
        "memory_GiB": 320,
        "walltime_s": 4500,
        "prior_allocated_core_hours": 14.72,
        "additional_allocated_core_hours": 0,
        "inflight_reserved_core_hours": 0,
        "additional_allocated_core_hours_limit": 129.28,
        "next_submission_reserved_core_hours": 120.0,
        "r1_ordinal": 1,
        "ordinal": 1,
        "formal_order": {key: list(value) for key, value in controller.FORMAL_ORDER.items()},
        "reference_boundary": str(reference),
        "reference_boundary_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "reference_boundary_size_bytes": len(raw),
        "reference_records": [{
            "path": str(reference),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }],
    }


def test_plan_accepts_exact_r1_budget_and_reference_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    plan = _valid_r1_plan(tmp_path)
    monkeypatch.setattr(controller.sg, "read", lambda _: dict(plan))
    accepted = controller._plan(tmp_path)
    assert accepted["r1_ordinal"] == 1


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        (lambda plan: plan.update(additional_allocated_core_hours=9.29), "R1_BUDGET_CONTRACT"),
        (lambda plan: plan.update(inflight_reserved_core_hours=9.29), "R1_BUDGET_CONTRACT"),
        (lambda plan: plan.update(next_submission_reserved_core_hours=120.01), "R1_BUDGET_CONTRACT"),
        (lambda plan: plan.update(r1_ordinal="1"), "R1_ORDINAL_REQUIRED"),
        (lambda plan: plan.update(ordinal="1"), "R1_ORDINAL_REQUIRED"),
        (lambda plan: plan.update(reference_boundary_raw_sha256="0" * 64), "R1_REFERENCE_BOUNDARY_IDENTITY"),
        (lambda plan: plan.pop("additional_allocated_core_hours_limit"), "R1_BUDGET_FIELDS_REQUIRED"),
    ),
)
def test_plan_rejects_r1_budget_or_reference_faults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation, reason: str):
    plan = _valid_r1_plan(tmp_path)
    mutation(plan)
    monkeypatch.setattr(controller.sg, "read", lambda _: dict(plan))
    with pytest.raises(RuntimeError, match=reason):
        controller._plan(tmp_path)
