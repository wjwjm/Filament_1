"""Frozen S4-6I formal fourteen-window protocol.

The formal phase reuses the reviewed 6F matrix and acceptance guards while
binding the already-selected S4-6I semantic candidate.  It owns protocol
metadata only; no propagation or scheduler side effects live here.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from tools.cpu_s4_6f_fullsize_parallel import protocol as fullsize

from . import selected_protocol


TASK = selected_protocol.TASK
BASE = selected_protocol.BASE
CANDIDATE = selected_protocol.CANDIDATE
SCIENCE_TREE = selected_protocol.SCIENCE_TREE
SHAPE = tuple(fullsize.SHAPE)
MICRO_SHAPE = tuple(fullsize.MICRO_SHAPE)
REMOTE_PYTHON = fullsize.REMOTE_PYTHON
VERSIONS = deepcopy(fullsize.VERSIONS)
THREAD_ENV = deepcopy(fullsize.THREAD_ENV)
COMPONENTS = deepcopy(getattr(selected_protocol, "COMPONENTS", ()))
CONTEXTS = tuple(getattr(selected_protocol, "CONTEXTS", ("q_ion", "q_ib", "direct_ion", "direct_ib")))

PHASE = "FORMAL_FOURTEEN"
FORMAL_SCHEMA = "filament.cpu_s4_6i.formal_protocol.v1"
NORMAL_JOB_COUNT = 3
NORMAL_ALLOCATED_CORE_HOURS_MAX = 480
L1_EMPTY_CAPACITY_BYTES = 1_610_612_736
SELECTED_OPERATION_GROUPS = {"2": 16, "8": 64}
ORIGINAL_COMPONENT_GROUPS = {"2": 60, "8": 240}
ROLE_COMPONENTS = {
    "R": ("intensity", "eq10", "c", "L1"),
    "X": ("intensity", "eq10", "c", "L1", "power_volume"),
}


def candidate_spec(name: str = CANDIDATE) -> dict[str, Any]:
    """Reuse the selected semantic specification without relabeling it."""

    value = deepcopy(selected_protocol.candidate_spec(name))
    value.update(
        {
            "semantic_id": CANDIDATE,
            "module": "tools.cpu_s4_6i_residual_parallel.selected_factory",
            "measurement_module": "tools.cpu_s4_6i_residual_parallel.selected_measurement",
            "old_role": "X",
            "workers": 24,
            "pointwise_operation": "POWER_VOLUME",
            "pointwise_labels": 2,
            "selected_operation_groups": deepcopy(SELECTED_OPERATION_GROUPS),
            "original_component_groups": deepcopy(ORIGINAL_COMPONENT_GROUPS),
            "role_components": {key: list(value) for key, value in ROLE_COMPONENTS.items()},
            "deferred": False,
        }
    )
    return value


def matrix(candidate: str = CANDIDATE) -> tuple[dict[str, Any], ...]:
    candidate_spec(candidate)
    # EQ10_P48 is only the 6F topology carrier; semantic identity remains the
    # selected S4-6I candidate in the returned plan and case receipts.
    return tuple(deepcopy(row) for row in fullsize.matrix("EQ10_P48"))


def pairings(candidate: str = CANDIDATE) -> tuple[dict[str, Any], ...]:
    candidate_spec(candidate)
    return tuple(deepcopy(row) for row in fullsize.pairings("EQ10_P48"))


def exception_classification(error: BaseException) -> str:
    return selected_protocol.exception_classification(error)


def _formal_resources() -> dict[str, Any]:
    resources = deepcopy(getattr(fullsize, "RESOURCE", {}))
    resources.update(
        {
            "physical_cpus": 96,
            "memory_GiB": 128,
            "gpus": 0,
            "walltime_s": 7200,
            "work_deadline_s": 7080,
            "normal_job_count": NORMAL_JOB_COUNT,
            "normal_allocated_core_hours_max": NORMAL_ALLOCATED_CORE_HOURS_MAX,
            "per_job_allocated_core_hours": 192,
            "max_jobids": NORMAL_JOB_COUNT,
            "study_allocated_core_hours_max": NORMAL_ALLOCATED_CORE_HOURS_MAX,
        }
    )
    return resources


def plan(candidate: str = CANDIDATE) -> dict[str, Any]:
    """Return the formal plan with the complete 6F execution contract.

    The 6F plan is the carrier for cache/count/resource fields consumed by the
    frozen entry.  Only the formal task, semantic candidate, matrix, pairings,
    acceptance metadata, and approved resource envelope are replaced.
    """

    spec = candidate_spec(candidate)
    carrier = deepcopy(fullsize.plan("EQ10_P48"))
    value = deepcopy(carrier)
    value.update(
        schema=FORMAL_SCHEMA + ".plan.v1",
        task=TASK,
        phase=PHASE,
        baseline_commit=BASE,
        science_tree=SCIENCE_TREE,
        candidate=CANDIDATE,
        old_role="X",
        candidate_module=spec["module"],
        candidate_spec=spec,
        shape=list(SHAPE),
        micro_shape=list(MICRO_SHAPE),
        matrix=list(matrix(candidate)),
        pairings=list(pairings(candidate)),
        components=list(COMPONENTS),
        reference="INTENSITY_P24_EQ10_P48_PHASE_ABSORB_P24_PLUS_L1",
        authorization="AUTHORIZED_WITHIN_S46I_3_JOB_480_CORE_HOUR_CONDITIONAL_PLAN",
        fullsize_cache=deepcopy(carrier["fullsize_cache"]),
        local_cache=deepcopy(carrier["local_cache"]),
        cache_counts=deepcopy(carrier["cache_counts"]),
        counts=deepcopy(carrier["counts"]),
        pointwise=dict(
            workers=24,
            labels=2,
            operation_count_per_label="4*intervals",
            groups_per_interval=4,
        ),
        pointwise_groups={
            "2": {
                "INTENSITY_P24": 24,
                "EQ10_P48": 32,
                "PHASE_ABSORB_P24": 4,
            },
            "8": {
                "INTENSITY_P24": 96,
                "EQ10_P48": 128,
                "PHASE_ABSORB_P24": 16,
            },
        },
        formal_pointwise_groups={
            "2": {
                "original_components": ORIGINAL_COMPONENT_GROUPS["2"],
                "POWER_VOLUME_CLEANUP_P24": SELECTED_OPERATION_GROUPS["2"],
            },
            "8": {
                "original_components": ORIGINAL_COMPONENT_GROUPS["8"],
                "POWER_VOLUME_CLEANUP_P24": SELECTED_OPERATION_GROUPS["8"],
            },
        },
        role_components={key: list(items) for key, items in ROLE_COMPONENTS.items()},
        resources=_formal_resources(),
        formal_acceptance={
            "qualification_cases": ["QUAL_R2", "QUAL_X2"],
            "early_controls": ["CTRL_EARLY_R1", "CTRL_EARLY_R2"],
            "order_RXXR": ["K1_R1", "K1_X1", "K1_X2", "K1_R2"],
            "order_XRRX": ["K2_X1", "K2_R1", "K2_R2", "K2_X2"],
            "late_controls": ["CTRL_LATE_R1", "CTRL_LATE_R2"],
            "role_JIT": "EMPTY_AT_QUALIFICATION_THEN_REUSE_BY_ROLE",
            "l1_empty_capacity_bytes": L1_EMPTY_CAPACITY_BYTES,
            "selected_operation_groups": deepcopy(SELECTED_OPERATION_GROUPS),
            "original_component_groups": deepcopy(ORIGINAL_COMPONENT_GROUPS),
            "role_components": {key: list(items) for key, items in ROLE_COMPONENTS.items()},
        },
        normal_study_job_count=NORMAL_JOB_COUNT,
        normal_study_allocated_core_hours_max=NORMAL_ALLOCATED_CORE_HOURS_MAX,
        normal_study_reserved_core_hours_max=NORMAL_ALLOCATED_CORE_HOURS_MAX,
        production_qualified=False,
        submission_authorized=False,
        review_gate="MANUAL_REVIEW_PENDING",
    )
    return value


__all__ = [
    "BASE",
    "CANDIDATE",
    "COMPONENTS",
    "CONTEXTS",
    "FORMAL_SCHEMA",
    "L1_EMPTY_CAPACITY_BYTES",
    "MICRO_SHAPE",
    "NORMAL_ALLOCATED_CORE_HOURS_MAX",
    "NORMAL_JOB_COUNT",
    "ORIGINAL_COMPONENT_GROUPS",
    "ROLE_COMPONENTS",
    "PHASE",
    "REMOTE_PYTHON",
    "SCIENCE_TREE",
    "SELECTED_OPERATION_GROUPS",
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
