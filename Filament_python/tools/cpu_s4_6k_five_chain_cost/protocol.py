"""One measurement-only COST6 on the accepted five-chain reference."""
from copy import deepcopy
from tools.cpu_s4_6i_residual_parallel import protocol as previous
from tools.cpu_s4_6j_alpha_factory.alpha_protocol import COMPONENTS as ALPHA_BASE_COMPONENTS
from tools.cpu_s4_6f_fullsize_parallel.protocol import (
    MICRO_SHAPE, REMOTE_PYTHON, SCIENCE_TREE, SHAPE, THREAD_ENV, VERSIONS)

TASK = 'CPU-S4-6K-FIVE-CHAIN-COST-20261104'
BASE = 'b907561223167b9996559bc0ee4996c53a626441'
CANDIDATE = 'FIVE_CHAIN_RESIDUAL_COST'
COMPONENTS = tuple(ALPHA_BASE_COMPONENTS) + ('ION_ALPHA_CLEANUP_P24',)

def candidate_spec(name):
    if str(name).upper() != CANDIDATE:
        raise ValueError('S46K_COST_SIX_ONLY')
    return dict(semantic_id=CANDIDATE, old_role='X', deferred=False,
                module='tools.cpu_s4_6k_five_chain_cost.role_spec',
                pointwise_operation='COMBINED', workers=96, pointwise_labels=16)

def matrix(candidate=CANDIDATE):
    candidate_spec(candidate)
    return previous.matrix()

def exception_classification(error):
    return previous.exception_classification(error)

def plan(candidate=CANDIDATE):
    candidate_spec(candidate)
    value = deepcopy(previous.plan())
    value.update(schema='filament.cpu_s4_6k.cost_six.plan.v1', task=TASK,
        baseline_commit=BASE, candidate=CANDIDATE, phase='COST_SIX',
        matrix=list(matrix()), components=list(COMPONENTS),
        reference_components=list(COMPONENTS), five_components_both_roles=True,
        joint_baseline_both_roles=True, pool_role='X',
        adoption_basis='MEASUREMENT_ONLY_STOP_AFTER_RANKING',
        normal_study_jobids_max=1, normal_study_reserved_core_hours_max=96,
        new_parallel_candidate=False, factory_repair=False,
        historical_gates_rewritten=False)
    value['acceptance'] = dict(phase='COST_SIX', scientific_exact_required=True,
        two_interval_stage_exact=True, every_window_terminal_exact=True,
        metrics=['application','propagation','body_2_to_8'],
        paired_median_cost_over_off_max=1.05, each_cost_over_off_max=1.10,
        complete_scalar_counts_and_clock_closure=True,
        ranking_requires_all_interference_gates=True,
        optimization_adoption='NOT_AUTHORIZED_THIS_TASK')
    value['ranking'].update(phase_assembly='SUM_FOUR_DISJOINT_ARITHMETIC_CHILDREN',
        branch_zero_cost='SEPARATE_NOT_ADDED_TO_CANDIDATE', automatic_candidate_implementation=False)
    for n in (2, 8):
        value['pointwise_groups'][str(n)].update(POWER_VOLUME_CLEANUP_P24=8*n, ION_ALPHA_CLEANUP_P24=5*n)
    return value
