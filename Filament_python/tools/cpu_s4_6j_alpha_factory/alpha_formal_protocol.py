"""Frozen S4-6J original-size fourteen-window protocol."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from . import alpha_protocol as base


TASK = base.TASK
BASE = base.BASE
CANDIDATE = base.CANDIDATE
SCIENCE_TREE = base.SCIENCE_TREE
SHAPE = tuple(base.SHAPE)
MICRO_SHAPE = tuple(base.MICRO_SHAPE)
REMOTE_PYTHON = base.REMOTE_PYTHON
VERSIONS = deepcopy(base.VERSIONS)
THREAD_ENV = deepcopy(base.THREAD_ENV)
COMPONENTS = deepcopy(base.COMPONENTS)

PHASE = "FORMAL_ALPHA_FOURTEEN"
FORMAL_SCHEMA = "filament.cpu_s4_6j.alpha_formal_protocol.v1"
NORMAL_JOB_COUNT = 2
NORMAL_ALLOCATED_CORE_HOURS_MAX = 384
L1_EMPTY_CAPACITY_BYTES = 1_610_612_736
ORIGINAL_COMPONENT_GROUPS = {"2": 76, "8": 304}
ALPHA_OPERATION_GROUPS = {"2": 10, "8": 40}
ROLE_COMPONENTS = {
    "R": ("intensity", "eq10", "c", "power_volume", "L1"),
    "X": ("intensity", "eq10", "c", "power_volume", "L1", "alpha"),
}


def candidate_spec(name: str = CANDIDATE) -> dict[str, Any]:
    value = deepcopy(base.candidate_spec(name))
    value.update(
        {
            "module": "tools.cpu_s4_6j_alpha_factory.alpha_role_spec",
            "measurement_module": "tools.cpu_s4_6j_alpha_factory.alpha_measurement",
            "pointwise_operation": "ION_ALPHA_CLEANUP",
            "pointwise_labels": 5,
            "workers": 24,
            "role_components": {key: list(items) for key, items in ROLE_COMPONENTS.items()},
            "deferred": False,
        }
    )
    return value


def matrix(candidate: str = CANDIDATE) -> tuple[dict[str, Any], ...]:
    candidate_spec(candidate)
    return tuple(deepcopy(row) for row in base.matrix(candidate))


def pairings(candidate: str = CANDIDATE) -> tuple[dict[str, Any], ...]:
    candidate_spec(candidate)
    return tuple(deepcopy(row) for row in base.pairings(candidate))


def exception_classification(error: BaseException) -> str:
    return base.exception_classification(error)


def plan(candidate: str = CANDIDATE) -> dict[str, Any]:
    spec = candidate_spec(candidate)
    value = deepcopy(base.plan(candidate))
    value.update(
        {
            "schema": FORMAL_SCHEMA + ".plan.v1",
            "phase": PHASE,
            "task": TASK,
            "baseline_commit": BASE,
            "candidate": CANDIDATE,
            "candidate_module": spec["module"],
            "candidate_spec": spec,
            "matrix": list(matrix(candidate)),
            "pairings": list(pairings(candidate)),
            "components": list(COMPONENTS),
            "reference": "R0_FOUR_COMPONENTS_PLUS_L1",
            "role_components": {key: list(items) for key, items in ROLE_COMPONENTS.items()},
            "original_component_groups": deepcopy(ORIGINAL_COMPONENT_GROUPS),
            "alpha_operation_groups": deepcopy(ALPHA_OPERATION_GROUPS),
            "formal14_allowed": "MECHANISM_ADMISSION_REQUIRED",
            "capture_enabled": False,
            "mechanism_enabled": False,
            "production_qualified": False,
            "submission_authorized": False,
            "review_gate": "MANUAL_REVIEW_PENDING",
        }
    )
    value["resources"] = dict(
        value.get("resources", {}),
        physical_cpus=96,
        memory_GiB=128,
        gpus=0,
        walltime_s=7200,
        work_deadline_s=7080,
        max_jobids=NORMAL_JOB_COUNT,
        normal_job_count=NORMAL_JOB_COUNT,
        per_job_allocated_core_hours=192,
        normal_allocated_core_hours_max=NORMAL_ALLOCATED_CORE_HOURS_MAX,
        study_allocated_core_hours_max=NORMAL_ALLOCATED_CORE_HOURS_MAX,
    )
    value["formal_acceptance"] = {
        "qualification_cases": ["QUAL_R2", "QUAL_X2"],
        "early_controls": ["CTRL_EARLY_R1", "CTRL_EARLY_R2"],
        "order_RXXR": ["K1_R1", "K1_X1", "K1_X2", "K1_R2"],
        "order_XRRX": ["K2_X1", "K2_R1", "K2_R2", "K2_X2"],
        "late_controls": ["CTRL_LATE_R1", "CTRL_LATE_R2"],
        "role_JIT": "EMPTY_AT_QUALIFICATION_THEN_REUSE_BY_ROLE",
        "l1_empty_capacity_bytes": L1_EMPTY_CAPACITY_BYTES,
        "original_component_groups": deepcopy(ORIGINAL_COMPONENT_GROUPS),
        "alpha_operation_groups": deepcopy(ALPHA_OPERATION_GROUPS),
        "body_paired_median_gain_strictly_greater_than": 0.0,
        "application_net_saved_median_strictly_greater_than": 0.0,
        "application_block_net_saved_strictly_positive": True,
        "first_POST": "UNEXPLAINED_DELAY_BLOCKS_MANUAL_ACCEPTANCE",
    }
    return value


__all__ = [
    "ALPHA_OPERATION_GROUPS",
    "BASE",
    "CANDIDATE",
    "COMPONENTS",
    "FORMAL_SCHEMA",
    "L1_EMPTY_CAPACITY_BYTES",
    "MICRO_SHAPE",
    "NORMAL_ALLOCATED_CORE_HOURS_MAX",
    "NORMAL_JOB_COUNT",
    "ORIGINAL_COMPONENT_GROUPS",
    "PHASE",
    "REMOTE_PYTHON",
    "ROLE_COMPONENTS",
    "SCIENCE_TREE",
    "SHAPE",
    "TASK",
    "THREAD_ENV",
    "VERSIONS",
    "candidate_spec",
    "exception_classification",
    "matrix",
    "pairings",
    "plan",
]
