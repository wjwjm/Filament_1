"""Original application denominator and close anchor for three/four pools."""
from types import FunctionType
from tools.cpu_s4_6f_fullsize_parallel import measurement as parent
from tools.cpu_s4_6h_combined_parallel import measurement as joint
from .selected_factory import finalize_factory
from .selected_protocol import CANDIDATE, COMPONENTS, CONTEXTS, candidate_spec
from .power_volume import NAN_TO_NUM_LABEL, MAXIMUM_LABEL

KEYS = joint.KEYS
POWER_LABELS = (NAN_TO_NUM_LABEL, MAXIMUM_LABEL)

def _closed(dispatch, semantic):
    candidate_spec(semantic)
    return finalize_factory(dispatch)

def _component_keys(dispatch):
    result = dict(KEYS)
    if dispatch.cpu_s4_6i_selected_role == 'X':
        result[CANDIDATE] = 'power_volume'
    return result

def _snapshot(dispatch, semantic):
    namespace = dict(joint._snapshot.__globals__, _closed=_closed,
                     KEYS=_component_keys(dispatch))
    return FunctionType(joint._snapshot.__code__, namespace)(dispatch, semantic)

def _take(dispatch, semantic):
    namespace = dict(joint._take.__globals__, _closed=_closed,
                     KEYS=_component_keys(dispatch))
    return FunctionType(joint._take.__code__, namespace)(dispatch, semantic)

def validate_pools(semantic, role, intervals, snapshot, activity, *, local_engineering):
    candidate_spec(semantic)
    if role not in ('R','X'):
        raise RuntimeError('S46I_SELECTED_ROLE')
    joint_rows = [row for row in activity if row['label'] not in POWER_LABELS]
    joint.validate_pools(joint.CANDIDATE, 'X', intervals, snapshot, joint_rows,
                         local_engineering=local_engineering)
    rows = [row for row in activity if row['label'] in POWER_LABELS]
    components = snapshot['components']
    if role == 'R':
        if rows or 'power_volume' in components:
            raise RuntimeError('S46I_REFERENCE_EXTRA_POOL')
        return
    pool = components['power_volume']
    if pool.get('dropped_records',0) or pool.get('invalid_records_dropped',0):
        raise RuntimeError('S46I_SELECTED_ACTIVITY_LOSS')
    # Reuse original lifecycle, item-partition, worker and scalar-size checks.
    namespace = dict(parent._validate_pool.__globals__, candidate_spec=candidate_spec,
                     POINTWISE_LABELS={CANDIDATE:POWER_LABELS})
    FunctionType(parent._validate_pool.__code__,namespace)(semantic,role,intervals,
        pool,rows,local_engineering=local_engineering)
    expected = [(i,context,label) for i in range(intervals)
                for context in CONTEXTS for label in POWER_LABELS]
    actual = [(row.get('interval_index'),row.get('context'),row['label']) for row in rows]
    if actual != expected:
        raise RuntimeError('S46I_SELECTED_CONTEXT_OPERATION_ORDER')

def run_window(args,built,**kwargs):
    namespace = dict(parent.run_window.__globals__,close_role=_closed,
        snapshot_role=_snapshot,take_role_records=_take,_validate_pool=validate_pools,
        pool_contract=lambda semantic:dict(candidate=CANDIDATE,
            components=list(_component_keys(built[1])),baseline_workers=[24,48,24],
            selected_workers=24 if built[1].cpu_s4_6i_selected_role=='X' else 0))
    return FunctionType(parent.run_window.__code__,namespace)(args,built,**kwargs)
