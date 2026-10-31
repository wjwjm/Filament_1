"""Fixed C/P24 mechanism and small-grid propagation contract."""

from __future__ import annotations


TASK = "CPU-S4-6E-C-MECHANISM-20261101"
BASE = "1c9f599d4f7fb0426374997981d0c52ffeaab1aa"
APPROVED_RESEARCH_BASELINE = "4431ba1192ee0d14b0eddf2efb138fe2a549d49f"
SOURCE_PARENT_COMMIT = BASE
SCIENCE_TREE = "821fec5bdacd832b0347e1f2e84285cb54172e57"
SHAPE = (384, 117, 129)
MICRO_SHAPE = (384, 17, 19)
REMOTE_PYTHON = "/publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_q1_20261014_110706/environment/venv/bin/python"
VERSIONS = dict(numpy="2.2.5", scipy="1.15.2", numba="0.61.2", llvmlite="0.44.0", psutil="7.0.0", threadpoolctl="3.6.0")
THREAD_ENV = dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", BLIS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", NUMBA_NUM_THREADS="48", NUMBA_CPU_NAME="generic", NUMBA_THREADING_LAYER="workqueue")
CAPTURE_INTERVALS = (0, 3, 7)
CAPTURE_TOTAL_CALLS = 8
HPC_EXECUTION_SCHEMA = "filament.cpu_s4_6e.hpc_execution.v1"
LOCAL_CASES = (
    ("QUAL_R2", "R", 2, False),
    ("QUAL_C2", "C", 2, False),
    ("R8_UNCAPTURED", "R", 8, False),
    ("R8_CAPTURED", "R", 8, True),
    ("C8", "C", 8, False),
)
HPC_CASES = (
    ("R8_UNCAPTURED", "R", 8, False),
    ("R8_CAPTURED", "R", 8, True),
)


def plan() -> dict:
    return {
        "schema": "filament.cpu_s4_6e_c.plan.v1",
        "task": TASK,
        "baseline_commit": SOURCE_PARENT_COMMIT,
        "approved_research_baseline": APPROVED_RESEARCH_BASELINE,
        "source_parent_commit": SOURCE_PARENT_COMMIT,
        "reference_basis": "FRESH_SOURCE_BOUND_L1_R8_UNCAPTURED_IN_THIS_C_CAMPAIGN",
        "a_result_composed_into_c_reference": False,
        "science_tree": SCIENCE_TREE,
        "shape": list(SHAPE),
        "micro_shape": list(MICRO_SHAPE),
        "local_cases": [dict(case=case, role=role, intervals=intervals, capture=capture) for case, role, intervals, capture in LOCAL_CASES],
        "hpc_cases": [dict(case=case, role=role, intervals=intervals, capture=capture) for case, role, intervals, capture in HPC_CASES],
        "capture": {
            "selected_interval_indices": list(CAPTURE_INTERVALS),
            "expected_total_calls": CAPTURE_TOTAL_CALLS,
            "expected_interval_sequence": list(range(CAPTURE_TOTAL_CALLS)),
            "arrays_persisted": "ORIGINAL_SIDE_ONLY_NPY",
            "capture_timing_in_performance_sample": False,
        },
        "mechanism": {
            "groups": ["G1", "G2"],
            "configs": {"G1": ["REF", "P1", "P24"], "G2": ["P24", "P1", "REF"]},
            "selected_inputs": list(CAPTURE_INTERVALS),
            "warmups_per_input_config": 2,
            "formal_per_input_config": 5,
            "formal_calls": 90,
            "warmup_calls": 36,
            "p24_gain_min": 0.05,
            "p1_is_mechanism_only": True,
            "local_p24_qualification": False,
        },
        "resources": {
            "account": "t0s000727",
            "slurm_account": "bscc-t",
            "partition": "v6_384",
            "qos": "normal",
            "nodes": 1,
            "physical_cpus": 96,
            "memory_GiB": 32,
            "gpus": 0,
            "walltime_s": 1800,
            "work_deadline_s": 1680,
            "max_new_jobids": 1,
            "allocated_core_hours_max": 48,
            "max_inflight": 1,
            "requeue": False,
            "memory_guard_fraction": 0.8,
        },
        "python": REMOTE_PYTHON,
        "versions": VERSIONS,
        "thread_environment": THREAD_ENV,
        "serial": True,
        "new_process_each_case": True,
        "warmup_intervals": 0,
        "default": "I0",
        "production_qualified": False,
        "review_gate": "MANUAL_REVIEW_PENDING",
        "authorization": "AUTHORIZED_WITHIN_APPROVED_5_JOB_CONDITIONAL_PLAN; per-attempt resource receipt required",
    }


__all__ = [
    "BASE",
    "APPROVED_RESEARCH_BASELINE",
    "CAPTURE_INTERVALS",
    "CAPTURE_TOTAL_CALLS",
    "HPC_CASES",
    "HPC_EXECUTION_SCHEMA",
    "LOCAL_CASES",
    "MICRO_SHAPE",
    "REMOTE_PYTHON",
    "SCIENCE_TREE",
    "SOURCE_PARENT_COMMIT",
    "SHAPE",
    "TASK",
    "THREAD_ENV",
    "VERSIONS",
    "plan",
]
