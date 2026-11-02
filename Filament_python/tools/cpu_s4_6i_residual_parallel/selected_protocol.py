"""Frozen single selected unit; stage two capture and mechanism protocol."""
from copy import deepcopy
from .protocol import (MICRO_SHAPE, REMOTE_PYTHON, SCIENCE_TREE, SHAPE, TASK,
                       THREAD_ENV, VERSIONS, COMPONENTS, exception_classification)
from .protocol import plan as cost_plan

BASE = '35cdca6391fa72f07a91cf1b7c02a4475acafbdc'
CANDIDATE = 'POWER_VOLUME_CLEANUP_P24'
CONTEXTS = ('q_ion', 'q_ib', 'direct_ion', 'direct_ib')
CAPTURE_INTERVALS = (0, 3, 7)
RESOURCES = dict(cost_plan()['resources'], walltime_s=7200,
                 work_deadline_s=7080, allocated_core_hours_max=192)

def candidate_spec(name):
    if str(name).upper() != CANDIDATE:
        raise ValueError('S46I_SELECTED_UNIT_ONLY')
    return dict(semantic_id=CANDIDATE, old_role='X', deferred=False,
                module='tools.cpu_s4_6i_residual_parallel.selected_factory',
                pointwise_operation='POWER_VOLUME', workers=24, pointwise_labels=2)

def plan(candidate=CANDIDATE):
    candidate_spec(candidate)
    value = deepcopy(cost_plan())
    for name in ('timing_interference', 'ranking', 'cost_pairs', 'matrix'):
        value.pop(name, None)
    value.update(schema='filament.cpu_s4_6i.selected_capture.plan.v1',
        baseline_commit=BASE, candidate=CANDIDATE, phase='CAPTURE_MECHANISM',
        performance_scope='SELECTED_ORIGINAL_GRID_LOCAL_MECHANISM',
        selected_unit='deposition_power_volume', selected_workers=24,
        source_contexts=list(CONTEXTS), capture_intervals=list(CAPTURE_INTERVALS),
        processes=['R8_UNCAPTURED', 'R8_CAPTURED', 'MECHANISM'],
        capture_input_count=12, capture_total_calls=32,
        groups=[['REF','P1','P24'], ['P24','P1','REF']],
        warmups_per_input_config_group=2, formal_per_input_config_group=5,
        warmup_calls=144, formal_calls=360, p24_gain_min=0.05,
        resources=deepcopy(RESOURCES), selected_operation_groups={'2':16,'8':64},
        unchanged_baseline_components=list(COMPONENTS),
        no_second_candidate=True, formal14_requires_this_mechanism_pass=True)
    return value
