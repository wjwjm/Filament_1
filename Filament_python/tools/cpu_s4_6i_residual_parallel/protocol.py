"""Frozen original-grid six-window cost measurement on the joint baseline."""
from copy import deepcopy
from tools.cpu_s4_6h_combined_parallel import protocol as previous
from tools.cpu_s4_6f_fullsize_parallel.protocol import (
    MICRO_SHAPE, REMOTE_PYTHON, SCIENCE_TREE, SHAPE, THREAD_ENV, VERSIONS, expected_counts)

TASK='CPU-S4-6I-RESIDUAL-PARALLEL-20261102'
BASE='54d8c3f2c101a5268a197f475e68c75e04333562'
CANDIDATE='JOINT_RESIDUAL_COST'
COMPONENTS=previous.COMPONENTS
_WINDOWS=(('OFF2','R',2,'OFF'),('COST2','X',2,'COST'),
          ('OFF8_A','R',8,'OFF'),('COST8_A','X',8,'COST'),
          ('COST8_B','X',8,'COST'),('OFF8_B','R',8,'OFF'))

def candidate_spec(name):
    if str(name).upper()!=CANDIDATE:
        raise ValueError('S46I_COST_SIX_ONLY')
    return dict(semantic_id=CANDIDATE,old_role='X',deferred=False,
                module='tools.cpu_s4_6i_residual_parallel.cost_factory',
                pointwise_operation='COMBINED',workers=96,pointwise_labels=9)

def matrix(candidate=CANDIDATE):
    candidate_spec(candidate)
    return tuple(dict(case=case,role=role,intervals=n,cost_mode=cost,
                      kind='QUALIFICATION' if n==2 else 'COST_INTERFERENCE')
                 for case,role,n,cost in _WINDOWS)

def exception_classification(error):
    text=repr(error)+'\n'+'\n'.join(getattr(error,'__notes__',()))
    if 'S46I_STOP_CHILD_FAILURE' in text or 'S46I_CLOSE_FAILURE' in text:
        return 'ENGINEERING_OR_RESOURCE_FAILURE'
    return previous.exception_classification(error)

def plan(candidate=CANDIDATE):
    candidate_spec(candidate)
    value=deepcopy(previous.plan())
    value.update(schema='filament.cpu_s4_6i.cost_six.plan.v1',task=TASK,
                 baseline_commit=BASE,candidate=CANDIDATE,matrix=list(matrix()),
                 phase='COST_SIX',joint_baseline_both_roles=True,pool_role='X',
                 performance_scope='COST_TIMING_INTERFERENCE_ONLY',
                 adoption_basis='ONE_MAXIMAL_DISJOINT_POINTWISE_UNIT_AFTER_VALID_COST',
                 timing_interference=dict(metrics=['application','propagation','body_2_to_8'],
                     median_ratio_max=1.05,each_ratio_max=1.10),
                 ranking=dict(metric='median_of_two_later_interval_exclusive_wall_sums',
                     tie_break='preregistered_source_order',inactive_branches='EXCLUDED',
                     parent_child_costs_added=False,worker_cpu_time_used=False),
                 cost_pairs=[['OFF8_A','COST8_A'],['OFF8_B','COST8_B']],
                 normal_study_jobids_max=3,normal_study_reserved_core_hours_max=480)
    return value
