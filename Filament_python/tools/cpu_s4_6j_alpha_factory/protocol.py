"""Frozen S4-6J first-stage protocol, using the accepted four components."""
from copy import deepcopy
from tools.cpu_s4_6i_residual_parallel import protocol as previous
from tools.cpu_s4_6f_fullsize_parallel.protocol import (
    MICRO_SHAPE, REMOTE_PYTHON, SCIENCE_TREE, SHAPE, THREAD_ENV, VERSIONS)

TASK = 'CPU-S4-6J-ALPHA-FACTORY-20261103'
BASE = '04aafc1386a21412fdfc8139515a6cc64da8a1a2'
CANDIDATE = 'FOUR_COMPONENT_FACTORY_COST'
COMPONENTS = tuple(previous.COMPONENTS) + ('POWER_VOLUME_CLEANUP_P24',)

def candidate_spec(name):
    if str(name).upper() != CANDIDATE:
        raise ValueError('S46J_COST_SIX_ONLY')
    return dict(semantic_id=CANDIDATE, old_role='X', deferred=False,
                module='tools.cpu_s4_6j_alpha_factory.role_spec',
                pointwise_operation='COMBINED', workers=96, pointwise_labels=11)

def matrix(candidate=CANDIDATE):
    candidate_spec(candidate)
    return previous.matrix()

def exception_classification(error):
    return previous.exception_classification(error)

def plan(candidate=CANDIDATE):
    candidate_spec(candidate)
    value = deepcopy(previous.plan())
    value.update(schema='filament.cpu_s4_6j.cost_six.plan.v1', task=TASK,
        baseline_commit=BASE, candidate=CANDIDATE, phase='COST_SIX',
        matrix=list(matrix()), components=list(COMPONENTS),
        reference_components=list(COMPONENTS), four_components_both_roles=True,
        joint_baseline_both_roles=True, pool_role='X',
        adoption_basis='ONE_EVIDENCED_SAFE_FACTORY_DUPLICATE_THEN_INDEPENDENT_ALPHA',
        normal_study_jobids_max=4, normal_study_reserved_core_hours_max=672,
        ranking=dict(metric='median_of_two_factory_exclusive_wall_sums',
            tie_break='preregistered_source_order', parent_child_costs_added=False,
            worker_cpu_time_used=False, automatic_factory_repair_selection=False),
        alpha_target='ION_ALPHA_CLEANUP_P24', alpha_body_gain_min=0.0,
        historical_gates_rewritten=False)
    for n in (2, 8):
        value['pointwise_groups'][str(n)]['POWER_VOLUME_CLEANUP_P24'] = 8*n
    return value
