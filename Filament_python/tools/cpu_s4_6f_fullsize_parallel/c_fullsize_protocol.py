"""Fixed contract for the CPU-S4-6F C full-size capture/mechanism job."""

from __future__ import annotations

from copy import deepcopy

from .protocol import BASE, MICRO_SHAPE, REMOTE_PYTHON, SCIENCE_TREE, SHAPE, THREAD_ENV, VERSIONS


TASK = "CPU-S4-6F-C-FULLSIZE-MECHANISM-20261101"
EXECUTION_SCHEMA = "filament.cpu_s4_6f_c_mechanism.execution.v1"
PLAN_SCHEMA = "filament.cpu_s4_6f_c_mechanism.plan.v1"
RESULT_SCHEMA = "filament.cpu_s4_6f_c_mechanism.result.v1"
EVIDENCE_SCHEMA = "filament.cpu_s4_6f_c_mechanism.evidence.v1"
CAPTURE_INTERVALS = (0, 3, 7)
CAPTURE_TOTAL_CALLS = 8
WARMUP_CALLS = 36
FORMAL_CALLS = 90

RESOURCES = dict(
    account="t0s000727",
    slurm_account="bscc-t",
    partition="v6_384",
    qos="normal",
    nodes=1,
    physical_cpus=96,
    memory_GiB=128,
    gpus=0,
    walltime_s=3600,
    work_deadline_s=3480,
    memory_guard_fraction=0.8,
    max_new_jobids=1,
    max_inflight=1,
    requeue=False,
    allocated_core_hours_max=96,
)


def plan() -> dict:
    return {
        "schema": PLAN_SCHEMA,
        "task": TASK,
        "baseline_commit": BASE,
        "science_tree": SCIENCE_TREE,
        "shape": list(SHAPE),
        "micro_shape": list(MICRO_SHAPE),
        "capture": {
            "selected_interval_indices": list(CAPTURE_INTERVALS),
            "expected_total_calls": CAPTURE_TOTAL_CALLS,
            "source_role": "R",
            "array_storage": "NPY_ALLOW_PICKLE_FALSE",
        },
        "mechanism": {
            "configs": ["REF", "P1", "P24"],
            "groups": ["G1", "G2"],
            "warmup_calls": WARMUP_CALLS,
            "formal_calls": FORMAL_CALLS,
            "selected_block_count": 6,
            "p24_gain_min": 0.05,
            "timer": "restore_and_post_checks_outside_function_clock",
        },
        "processes": [
            "R8_UNCAPTURED",
            "R8_CAPTURED",
            "MECHANISM",
        ],
        "serial": True,
        "new_process_each_stage": True,
        "factor_cache": "EMPTY_PRIVATE_FACTORY_EACH_PROCESS",
        "fine_performance_profile": False,
        "cpu_diagnostics_profile": True,
        "diagnostic_outputs": True,
        "resources": deepcopy(RESOURCES),
        "python": REMOTE_PYTHON,
        "versions": deepcopy(VERSIONS),
        "thread_environment": deepcopy(THREAD_ENV),
        "authorization": "AUTHORIZED_WITHIN_APPROVED_4_JOB_CONDITIONAL_PLAN; per-attempt resource receipt required",
        "fullsize": True,
        "GPU": False,
        "default": "I0",
        "production_qualified": False,
        "review_gate": "MANUAL_REVIEW_PENDING",
    }


__all__ = [
    "BASE",
    "CAPTURE_INTERVALS",
    "CAPTURE_TOTAL_CALLS",
    "EVIDENCE_SCHEMA",
    "EXECUTION_SCHEMA",
    "FORMAL_CALLS",
    "MICRO_SHAPE",
    "PLAN_SCHEMA",
    "REMOTE_PYTHON",
    "RESULT_SCHEMA",
    "RESOURCES",
    "SCIENCE_TREE",
    "SHAPE",
    "TASK",
    "THREAD_ENV",
    "VERSIONS",
    "WARMUP_CALLS",
    "plan",
]
