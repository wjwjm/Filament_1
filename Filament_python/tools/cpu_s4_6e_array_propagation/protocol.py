"""The approved S4-6E A/P24 small-grid matrix; no submission authority."""

from __future__ import annotations

TASK = "CPU-S4-6E-INTENSITY-P24-PROPAGATION-20261101"
BASE = "4431ba1192ee0d14b0eddf2efb138fe2a549d49f"
CPU_BASE = BASE
MECHANISM_EXECUTION = "420ed7e82f09180da9a1b546f4ac69daeb795f59"
SCIENCE_TREE = "821fec5bdacd832b0347e1f2e84285cb54172e57"
SHAPE = (384, 117, 129)
MICRO_SHAPE = (384, 17, 19)
REMOTE_PYTHON = "/publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_q1_20261014_110706/environment/venv/bin/python"
VERSIONS = dict(numpy="2.2.5", scipy="1.15.2", numba="0.61.2", llvmlite="0.44.0", psutil="7.0.0", threadpoolctl="3.6.0")
THREAD_ENV = dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", BLIS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", NUMBA_NUM_THREADS="48", NUMBA_CPU_NAME="generic", NUMBA_THREADING_LAYER="workqueue")
MATRIX = (
    ("QUAL_R2", "R", 2, "QUALIFICATION"), ("QUAL_A2", "A", 2, "QUALIFICATION"),
    ("CTRL_EARLY_R1", "R", 8, "RR_CONTROL"), ("CTRL_EARLY_R2", "R", 8, "RR_CONTROL"),
    ("K1_R1", "R", 8, "FORMAL"), ("K1_A1", "A", 8, "FORMAL"),
    ("K1_A2", "A", 8, "FORMAL"), ("K1_R2", "R", 8, "FORMAL"),
    ("K2_A1", "A", 8, "FORMAL"), ("K2_R1", "R", 8, "FORMAL"),
    ("K2_R2", "R", 8, "FORMAL"), ("K2_A2", "A", 8, "FORMAL"),
    ("CTRL_LATE_R1", "R", 8, "RR_CONTROL"), ("CTRL_LATE_R2", "R", 8, "RR_CONTROL"),
)
PAIRS = (
    ("K1_P1", "K1_R1", "K1_A1", "RA"), ("K1_P2", "K1_R2", "K1_A2", "AR"),
    ("K2_P1", "K2_R1", "K2_A1", "AR"), ("K2_P2", "K2_R2", "K2_A2", "RA"),
)


def plan():
    return dict(
        schema="filament.cpu_s4_6e.plan.v1",
        task=TASK,
        baseline_commit=BASE,
        historical_CPU_execution=CPU_BASE,
        mechanism_execution=MECHANISM_EXECUTION,
        science_tree=SCIENCE_TREE,
        shape=list(SHAPE),
        micro_shape=list(MICRO_SHAPE),
        matrix=[dict(case=c, role=r, intervals=n, kind=k) for c, r, n, k in MATRIX],
        pairings=[dict(pair=p, reference=r, candidate=a, order=o) for p, r, a, o in PAIRS],
        stage_exact_scope="TWO_INTERVAL_QUALIFICATION; user confirmed eight-interval full terminal exact on 2026-11-01",
        serial=True,
        new_process_each_window=True,
        propagation_warmup_intervals=0,
        factor_cache="EMPTY_PRIVATE_FACTORY_EACH_WINDOW",
        role_JIT_cache="SEPARATE_ROLE_NAMESPACES_REUSED_AFTER_QUALIFICATION",
        candidate="THREE_INTENSITY_POINTWISE_SEGMENTS_P24_ONLY",
        python=REMOTE_PYTHON,
        versions=VERSIONS,
        thread_environment=THREAD_ENV,
        resources=dict(account="t0s000727", slurm_account="bscc-t", partition="v6_384", qos="normal", nodes=1, physical_cpus=96, memory_GiB=32, gpus=0, walltime_s=1800, work_deadline_s=1680, max_new_jobids=1, allocated_core_hours_max=48, max_inflight=1, requeue=False, memory_guard_fraction=0.8),
        acceptance=dict(body_scope="sum complete original body ordinals2..8", paired_median_gain_min=0.03, both_blocks_positive=True, both_orders_positive=True, exceeds_observed_RR_log_variation=True, POST_both_blocks_positive=True, application_ratio_median_max=1.02, POST_metric="Per-window median of adjacent POST gaps; balanced log ratio of the two formal pairs in each block", application_ratio_each_max=1.05, first_POST="UNEXPLAINED_DELAY_BLOCKS_MANUAL_ACCEPTANCE"),
        authorization="AUTHORIZED_WITHIN_APPROVED_5_JOB_CONDITIONAL_PLAN; per-attempt resource receipt required",
        fullsize=False,
        GPU=False,
        default="I0",
        production_qualified=False,
        review_gate="MANUAL_REVIEW_PENDING",
    )
