"""Fixed C/P24 fourteen-window small-grid propagation contract."""

from __future__ import annotations


TASK = "CPU-S4-6E-C14-PROPAGATION-20261101"
BASE = "c19258163a2d570424fd680263421ce8b6f9cd1f"
MECHANISM_EXECUTION = BASE
APPROVED_RESEARCH_BASELINE = "4431ba1192ee0d14b0eddf2efb138fe2a549d49f"
SCIENCE_TREE = "821fec5bdacd832b0347e1f2e84285cb54172e57"
SHAPE = (384, 117, 129)
MICRO_SHAPE = (384, 17, 19)
REMOTE_PYTHON = "/publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_q1_20261014_110706/environment/venv/bin/python"
VERSIONS = dict(numpy="2.2.5", scipy="1.15.2", numba="0.61.2", llvmlite="0.44.0", psutil="7.0.0", threadpoolctl="3.6.0")
THREAD_ENV = dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", BLIS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", NUMBA_NUM_THREADS="48", NUMBA_CPU_NAME="generic", NUMBA_THREADING_LAYER="workqueue")
HPC_EXECUTION_SCHEMA = "filament.cpu_s4_6e.hpc_execution.v1"
EXECUTION_SCHEMA = "filament.cpu_s4_6e_c14.execution.v1"
PLAN_SCHEMA = "filament.cpu_s4_6e_c14.plan.v1"
RESULT_SCHEMA = "filament.cpu_s4_6e_c14.result.v1"

MATRIX = (
    ("QUAL_R2", "R", 2, "QUALIFICATION"),
    ("QUAL_C2", "C", 2, "QUALIFICATION"),
    ("CTRL_EARLY_R1", "R", 8, "RR_CONTROL"),
    ("CTRL_EARLY_R2", "R", 8, "RR_CONTROL"),
    ("K1_R1", "R", 8, "FORMAL"),
    ("K1_C1", "C", 8, "FORMAL"),
    ("K1_C2", "C", 8, "FORMAL"),
    ("K1_R2", "R", 8, "FORMAL"),
    ("K2_C1", "C", 8, "FORMAL"),
    ("K2_R1", "R", 8, "FORMAL"),
    ("K2_R2", "R", 8, "FORMAL"),
    ("K2_C2", "C", 8, "FORMAL"),
    ("CTRL_LATE_R1", "R", 8, "RR_CONTROL"),
    ("CTRL_LATE_R2", "R", 8, "RR_CONTROL"),
)
PAIRS = (
    ("K1_P1", "K1_R1", "K1_C1", "RC"),
    ("K1_P2", "K1_R2", "K1_C2", "CR"),
    ("K2_P1", "K2_R1", "K2_C1", "CR"),
    ("K2_P2", "K2_R2", "K2_C2", "RC"),
)
LOCAL_CASES = (
    ("QUAL_R2", "R", 2),
    ("QUAL_C2", "C", 2),
    ("R8_UNCAPTURED", "R", 8),
    ("C8", "C", 8),
)
RESOURCES = dict(
    account="t0s000727",
    slurm_account="bscc-t",
    partition="v6_384",
    qos="normal",
    nodes=1,
    physical_cpus=96,
    memory_GiB=32,
    gpus=0,
    walltime_s=1800,
    work_deadline_s=1680,
    max_new_jobids=1,
    allocated_core_hours_max=48,
    max_inflight=1,
    requeue=False,
    memory_guard_fraction=0.8,
)


def plan() -> dict:
    return {
        "schema": PLAN_SCHEMA,
        "task": TASK,
        "baseline_commit": BASE,
        "mechanism_execution_commit": MECHANISM_EXECUTION,
        "approved_research_baseline": APPROVED_RESEARCH_BASELINE,
        "science_tree": SCIENCE_TREE,
        "shape": list(SHAPE),
        "micro_shape": list(MICRO_SHAPE),
        "matrix": [dict(case=case, role=role, intervals=intervals, kind=kind) for case, role, intervals, kind in MATRIX],
        "pairings": [dict(pair=pair, reference=reference, candidate=candidate, order=order) for pair, reference, candidate, order in PAIRS],
        "local_cases": [dict(case=case, role=role, intervals=intervals) for case, role, intervals in LOCAL_CASES],
        "reference_basis": "FROZEN_A_L1_ORIGINAL112_BOUNDARY_STAGE_FOR_R2; CURRENT_C14_R2_FOR_C2_AND_FORMAL_RR",
        "mechanism_admission": {
            "execution_commit": MECHANISM_EXECUTION,
            "formal_calls": 90,
            "warmup_calls": 36,
            "all_configs_exact": True,
            "p24_gain_min": 0.05,
            "selected_block_count": 6,
            "scientific_arrays_loaded_by_preflight": 0,
        },
        "serial": True,
        "new_process_each_window": True,
        "propagation_warmup_intervals": 0,
        "factor_cache": "EMPTY_PRIVATE_FACTORY_EACH_WINDOW",
        "candidate": "C_NONLINEAR_PHASE_AND_ABSORPTION_POINTWISE_P24_ONLY",
        "counts": {
            "two_interval": {"half_steps": 4, "fresh_responses": 8, "original_math_calls": 32, "c_round_tasks": 4},
            "eight_interval": {"half_steps": 16, "fresh_responses": 32, "original_math_calls": 128, "c_round_tasks": 16},
        },
        "resources": RESOURCES,
        "python": REMOTE_PYTHON,
        "versions": VERSIONS,
        "thread_environment": THREAD_ENV,
        "acceptance": {
            "body_scope": "sum complete original body ordinals2..8",
            "paired_median_gain_min": 0.03,
            "both_blocks_positive": True,
            "both_orders_positive": True,
            "exceeds_observed_RR_log_variation": True,
            "POST_both_blocks_positive": True,
            "application_ratio_median_max": 1.02,
            "application_ratio_each_max": 1.05,
            "first_POST": "UNEXPLAINED_DELAY_BLOCKS_MANUAL_ACCEPTANCE",
            "RR_controls": 2,
        },
        "authorization": "AUTHORIZED_WITHIN_APPROVED_5_JOB_CONDITIONAL_PLAN; per-attempt resource receipt required",
        "fullsize": False,
        "GPU": False,
        "default": "I0",
        "production_qualified": False,
        "review_gate": "MANUAL_REVIEW_PENDING",
    }


__all__ = [
    "APPROVED_RESEARCH_BASELINE",
    "BASE",
    "EXECUTION_SCHEMA",
    "HPC_EXECUTION_SCHEMA",
    "LOCAL_CASES",
    "MATRIX",
    "MECHANISM_EXECUTION",
    "MICRO_SHAPE",
    "PAIRS",
    "PLAN_SCHEMA",
    "REMOTE_PYTHON",
    "RESULT_SCHEMA",
    "RESOURCES",
    "SCIENCE_TREE",
    "SHAPE",
    "TASK",
    "THREAD_ENV",
    "VERSIONS",
    "plan",
]
