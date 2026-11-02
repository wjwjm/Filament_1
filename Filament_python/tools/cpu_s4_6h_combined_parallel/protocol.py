"""Original-grid qualification of the user-selected three-chain CPU path."""
from copy import deepcopy

from tools.cpu_s4_6f_fullsize_parallel.protocol import (
    MICRO_SHAPE, REMOTE_PYTHON, SCIENCE_TREE, SHAPE, THREAD_ENV, VERSIONS,
    expected_counts,
)

TASK = "CPU-S4-6H-COMBINED-PARALLEL-20261102"
BASE = "5388bdba5965f779ec7bcc2232b7e935a0d215cf"
CANDIDATE = "INTENSITY_P24_EQ10_P48_PHASE_ABSORB_P24"
COMPONENTS = ("INTENSITY_P24", "EQ10_P48", "PHASE_ABSORB_P24")
_WINDOWS = (("QUAL_R2", "R", 2), ("QUAL_X2", "X", 2),
            ("QUAL_R8", "R", 8), ("QUAL_X8", "X", 8))


def candidate_spec(name):
    if str(name).upper() != CANDIDATE:
        raise ValueError("S46H_COMBINED_CANDIDATE_REQUIRED")
    return dict(semantic_id=CANDIDATE, old_role="X", deferred=False,
                module="tools.cpu_s4_6h_combined_parallel.combined_candidate",
                pointwise_operation="COMBINED", workers=96, pointwise_labels=9)


def matrix(candidate=CANDIDATE):
    candidate_spec(candidate)
    return tuple(dict(case=case, role=role, intervals=n, kind="JOINT_QUALIFICATION")
                 for case, role, n in _WINDOWS)


def exception_classification(error):
    text = repr(error) + "\n" + "\n".join(getattr(error, "__notes__", ()))
    if any(tag in text for tag in (
        "S46H_CLOSE_FAILURE", "S46H_POOL_FINALIZATION_FAILED",
        "S46H_STOP_CHILD_FAILURE", "S46F_STOP_CHILD_FAILURE",
    )):
        return "ENGINEERING_OR_RESOURCE_FAILURE"
    science_tags = (
        "STOP_FIRST_SCIENCE_NONEXACT", "STOP_FIRST_STAGE_NONEXACT",
        "S46F_STOP_TERMINAL_NONEXACT", "S46F_STOP_BOUNDARY_NONEXACT",
        "S46F_STOP_STAGES_NONEXACT", "SCIENTIFIC_NONEXACT",
    )
    return "SCIENTIFIC_NONEXACT" if any(tag in text for tag in science_tags) else "ENGINEERING_OR_RESOURCE_FAILURE"


def plan(candidate=CANDIDATE):
    candidate_spec(candidate)
    return dict(
        schema="filament.cpu_s4_6h.joint_qualification.plan.v1", task=TASK,
        baseline_commit=BASE, science_tree=SCIENCE_TREE, candidate=CANDIDATE,
        components=list(COMPONENTS), shape=list(SHAPE), micro_shape=list(MICRO_SHAPE),
        matrix=list(matrix()), serial=True, new_process_each_window=True,
        propagation_warmup_intervals=0, factor_cache="EMPTY_EACH_WINDOW",
        role_JIT_cache="FRESH_AT_QUALIFICATION_THEN_REUSED_BY_ROLE",
        fullsize_cache=dict(max_tables=2, max_bytes=1610612736),
        counts={str(n): expected_counts(n) for n in (2, 8)},
        pointwise_groups={str(n): dict(INTENSITY_P24=12*n, EQ10_P48=16*n,
                                     PHASE_ABSORB_P24=2*n) for n in (2, 8)},
        python=REMOTE_PYTHON, versions=deepcopy(VERSIONS),
        thread_environment=deepcopy(THREAD_ENV),
        resources=dict(account="t0s000727", slurm_account="bscc-t", partition="v6_384",
                       qos="normal", nodes=1, physical_cpus=96, memory_GiB=128,
                       gpus=0, walltime_s=3600, work_deadline_s=3480,
                       memory_guard_fraction=0.8, max_inflight=1, requeue=False,
                       max_jobids=1, per_job_allocated_core_hours=96),
        exact="TWO_INTERVAL_BOUNDARY_STAGE_TERMINAL_AND_EIGHT_INTERVAL_TERMINAL",
        performance_scope="ONE_EIGHT_INTERVAL_PAIR_DESCRIPTIVE_ONLY",
        adoption_basis="USER_SELECTED_JOINT_PATH_REQUIRES_SCIENTIFIC_EXACT",
        historical_default="I0", production_qualified=False,
    )
