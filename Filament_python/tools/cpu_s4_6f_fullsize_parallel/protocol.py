"""Frozen CPU-S4-6F full-size matrix and resource contract.

The three candidates are selected independently. ``R`` is the current
private L1 reference and ``X`` is the one selected semantic candidate; no
combination is admitted by this protocol.
"""

from __future__ import annotations

from copy import deepcopy

TASK = "CPU-S4-6F-FULLSIZE-PARALLEL-20261101"
BASE = "9bb61d0cbd2f93315c69a4b58fe90d2041947ae1"
CPU_BASE = BASE
SCIENCE_TREE = "821fec5bdacd832b0347e1f2e84285cb54172e57"
SHAPE = (384, 351, 301)
MICRO_SHAPE = (384, 17, 19)
REMOTE_PYTHON = "/publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_q1_20261014_110706/environment/venv/bin/python"
VERSIONS = dict(numpy="2.2.5", scipy="1.15.2", numba="0.61.2", llvmlite="0.44.0", psutil="7.0.0", threadpoolctl="3.6.0")
THREAD_ENV = dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", BLIS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", NUMBA_NUM_THREADS="48", NUMBA_CPU_NAME="generic", NUMBA_THREADING_LAYER="workqueue")

_WINDOWS = (
    ("QUAL_R2", "R", 2, "QUALIFICATION"),
    ("QUAL_X2", "X", 2, "QUALIFICATION"),
    ("CTRL_EARLY_R1", "R", 8, "RR_CONTROL"),
    ("CTRL_EARLY_R2", "R", 8, "RR_CONTROL"),
    ("K1_R1", "R", 8, "FORMAL"),
    ("K1_X1", "X", 8, "FORMAL"),
    ("K1_X2", "X", 8, "FORMAL"),
    ("K1_R2", "R", 8, "FORMAL"),
    ("K2_X1", "X", 8, "FORMAL"),
    ("K2_R1", "R", 8, "FORMAL"),
    ("K2_R2", "R", 8, "FORMAL"),
    ("K2_X2", "X", 8, "FORMAL"),
    ("CTRL_LATE_R1", "R", 8, "RR_CONTROL"),
    ("CTRL_LATE_R2", "R", 8, "RR_CONTROL"),
)
_PAIRS = (
    ("K1_P1", "K1_R1", "K1_X1", "RX"),
    ("K1_P2", "K1_R2", "K1_X2", "XR"),
    ("K2_P1", "K2_R1", "K2_X1", "XR"),
    ("K2_P2", "K2_R2", "K2_X2", "RX"),
)

CANDIDATES = {
    "EQ10_P48": dict(semantic_id="EQ10_P48", old_role="B", module="tools.cpu_s4_6d_eq10_propagation.candidate", measurement_module="tools.cpu_s4_6d_eq10_propagation.measurement", workers=48, pointwise_labels=4, pointwise_operation="EQ10", deferred=False),
    "INTENSITY_P24": dict(semantic_id="INTENSITY_P24", old_role="A", module="tools.cpu_s4_6e_array_propagation.candidate", measurement_module="tools.cpu_s4_6e_array_propagation.measurement", workers=24, pointwise_labels=3, pointwise_operation="INTENSITY", deferred=False),
    "PHASE_ABSORB_P24": dict(semantic_id="PHASE_ABSORB_P24", old_role="C", module="tools.cpu_s4_6e_array_propagation.c_candidate", measurement_module="tools.cpu_s4_6e_array_propagation.measurement", workers=24, pointwise_labels=2, pointwise_operation="PHASE_ABSORB", deferred=True),
}

RESOURCE = dict(
    account="t0s000727", slurm_account="bscc-t", partition="v6_384", qos="normal", nodes=1,
    physical_cpus=96, memory_GiB=128, gpus=0, walltime_s=7200, work_deadline_s=7080,
    memory_guard_fraction=0.8, max_inflight=1, requeue=False, max_jobids=4,
    per_job_allocated_core_hours=192, study_allocated_core_hours_max=672,
)
LOCAL_RESOURCE = dict(physical_cpus=1, memory_MiB=256, gpus=0, work_deadline_s=120, hpc_allowed=False)


def candidate_spec(name: str) -> dict:
    try:
        value = CANDIDATES[str(name).upper()]
    except KeyError as exc:
        raise ValueError(f"unknown CPU-S4-6F semantic candidate: {name!r}") from exc
    return deepcopy(value)


def matrix(candidate: str):
    candidate_spec(candidate)
    return tuple(dict(case=case, role=role, intervals=intervals, kind=kind) for case, role, intervals, kind in _WINDOWS)


def pairings(candidate: str):
    candidate_spec(candidate)
    return tuple(dict(pair=pair, reference=reference, candidate=selected, order=order) for pair, reference, selected, order in _PAIRS)


def expected_counts(intervals: int) -> dict[str, int]:
    n_half = 2 * int(intervals)
    return dict(half_steps=n_half, fresh_responses=2 * n_half, original_math_dispatches=8 * n_half)


def plan(candidate: str):
    spec = candidate_spec(candidate)
    return dict(
        schema="filament.cpu_s4_6f.fullsize_parallel.plan.v1", task=TASK, baseline_commit=BASE,
        historical_CPU_execution=CPU_BASE, science_tree=SCIENCE_TREE, candidate=spec["semantic_id"],
        old_role=spec["old_role"], candidate_module=spec["module"], shape=list(SHAPE), micro_shape=list(MICRO_SHAPE),
        matrix=list(matrix(candidate)), pairings=list(pairings(candidate)), serial=True, new_process_each_window=True,
        propagation_warmup_intervals=0, role_JIT_cache="FRESH_AT_QUALIFICATION_THEN_REUSED_BY_ROLE",
        factor_cache="EMPTY_PRIVATE_FACTORY_EACH_WINDOW", fullsize_cache=dict(max_tables=2, max_bytes=1_610_612_736, scope="FULLSIZE_PRIVATE_L1"),
        local_cache=dict(max_tables=2, max_bytes=256 * 1024 * 1024, scope="LOCAL_ENGINEERING_ONLY"),
        pointwise=dict(workers=spec["workers"], labels=spec["pointwise_labels"], operation_count_per_label="intervals" if spec["pointwise_operation"] == "PHASE_ABSORB" else "4*intervals", groups_per_interval=2 if spec["pointwise_operation"] == "PHASE_ABSORB" else 4),
        counts={"two_intervals": expected_counts(2), "eight_intervals": expected_counts(8)},
        cache_counts={"two_intervals": dict(requests=1536, builds=384, hits=1152, fallback=0), "eight_intervals": dict(requests=6144, builds=384, hits=5760, fallback=0)},
        python=REMOTE_PYTHON, versions=deepcopy(VERSIONS), thread_environment=deepcopy(THREAD_ENV), resources=deepcopy(RESOURCE), local_resources=deepcopy(LOCAL_RESOURCE),
        acceptance=dict(body_scope="sum complete body ordinals 2..8", paired_median_gain_min=0.03, both_blocks_positive=True, both_orders_positive=True, exceeds_observed_RR_log_variation=True, POST_both_blocks_positive=True, application_ratio_median_max=1.02, application_ratio_each_max=1.05, first_POST="UNEXPLAINED_DELAY_BLOCKS_MANUAL_ACCEPTANCE"),
        authorization="AUTHORIZED_WITHIN_APPROVED_4_JOB_CONDITIONAL_PLAN; per-attempt resource receipt required", fullsize=True, GPU=False, default="I0", production_qualified=False, review_gate="MANUAL_REVIEW_PENDING", deferred=bool(spec["deferred"]),
    )


__all__ = ["BASE", "CANDIDATES", "CPU_BASE", "MICRO_SHAPE", "REMOTE_PYTHON", "RESOURCE", "SCIENCE_TREE", "SHAPE", "TASK", "THREAD_ENV", "VERSIONS", "candidate_spec", "expected_counts", "matrix", "pairings", "plan"]
