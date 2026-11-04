"""S4-6J alpha plan; the established 6F clock/matrix contracts are retained."""
from copy import deepcopy
from tools.cpu_s4_6f_fullsize_parallel import protocol as fullsize
from tools.cpu_s4_6f_fullsize_parallel.protocol import MICRO_SHAPE, REMOTE_PYTHON, SCIENCE_TREE, SHAPE, THREAD_ENV, VERSIONS
from .protocol import BASE, TASK, COMPONENTS, exception_classification

CANDIDATE = 'ION_ALPHA_CLEANUP_P24'

def candidate_spec(name=CANDIDATE):
    if str(name).upper() != CANDIDATE:
        raise ValueError('S46J_ALPHA_CANDIDATE_REQUIRED')
    return dict(semantic_id=CANDIDATE, old_role='X', deferred=False,
        module='tools.cpu_s4_6j_alpha_factory.alpha_role_spec',
        pointwise_operation='ION_ALPHA_CLEANUP', workers=24, pointwise_labels=5)

def matrix(candidate=CANDIDATE):
    candidate_spec(candidate)
    return tuple(deepcopy(row) for row in fullsize.matrix('EQ10_P48'))

def pairings(candidate=CANDIDATE):
    candidate_spec(candidate)
    return tuple(deepcopy(row) for row in fullsize.pairings('EQ10_P48'))

def plan(candidate=CANDIDATE):
    spec = candidate_spec(candidate)
    value = deepcopy(fullsize.plan('EQ10_P48'))
    value.update(schema='filament.cpu_s4_6j.alpha.plan.v1', task=TASK,
        phase='ALPHA_CAPTURE_THEN_CONDITIONAL_FORMAL14', baseline_commit=BASE,
        candidate=CANDIDATE, candidate_module=spec['module'], candidate_spec=spec,
        reference='R0_FOUR_COMPONENTS_PLUS_L1', reference_components=list(COMPONENTS),
        components=list(COMPONENTS), matrix=list(matrix()), pairings=list(pairings()),
        pointwise=dict(workers=24, labels=5, groups_per_interval=1),
        factory_repair_selected=False, historical_gates_rewritten=False)
    value['resources'].update(physical_cpus=96, memory_GiB=128, gpus=0,
        walltime_s=7200, work_deadline_s=7080, per_job_allocated_core_hours=192)
    value['acceptance'] = dict(scientific_exact_required=True, two_interval_stage_exact=True,
        every_window_terminal_exact=True, body_metric='complete_body_intervals_2_to_8',
        paired_body_median_gain_strictly_greater_than=0.0, both_blocks_positive=True,
        both_orders_positive=True, balanced_body_log_gain_exceeds_max_RR=True,
        POST_both_blocks_positive=True, application_ratio_median_max=1.02,
        application_ratio_each_max=1.05, application_median_saved_s_strictly_positive=True,
        both_application_blocks_net_saved_positive=True,
        unexplained_first_POST_delay_blocks_manual_acceptance=True)
    value['alpha'] = dict(calls={'2':2,'8':8}, rounds={'2':10,'8':40}, workers=24,
        capture_intervals=[0,3,7], input_count=3, arrays=12, warmup_calls=36,
        formal_calls=90, blocks=6, mechanism_gain_each_min=0.05)
    value['base_pointwise_groups'] = {'2':76,'8':304}
    value['normal_study_jobids_max']=4
    value['normal_study_reserved_core_hours_max']=672
    return value
