"""Reuse the accepted four-pool checks and the original close-before-output clock."""
from types import FunctionType
from tools.cpu_s4_6f_fullsize_parallel import measurement as parent
from tools.cpu_s4_6h_combined_parallel import measurement as joint
from tools.cpu_s4_6i_residual_parallel import selected_measurement as four
from tools.cpu_s4_6i_residual_parallel.selected_protocol import CANDIDATE as POWER
from .role_spec import close_role
from .protocol import CANDIDATE, COMPONENTS, candidate_spec

KEYS = joint.KEYS

def _closed(dispatch, semantic):
    return close_role(dispatch, semantic)

def _snapshot(dispatch, semantic):
    candidate_spec(semantic)
    namespace = dict(joint._snapshot.__globals__, _closed=_closed,
                     KEYS=four._component_keys(dispatch))
    return FunctionType(joint._snapshot.__code__, namespace)(dispatch, semantic)

def _take(dispatch, semantic):
    candidate_spec(semantic)
    namespace = dict(joint._take.__globals__, _closed=_closed,
                     KEYS=four._component_keys(dispatch))
    return FunctionType(joint._take.__code__, namespace)(dispatch, semantic)

def validate_pools(semantic, role, intervals, snapshot, activity, *, local_engineering):
    candidate_spec(semantic)
    if role not in ('R','X'):
        raise RuntimeError('S46J_EXTERNAL_ROLE')
    return four.validate_pools(POWER, 'X', intervals, snapshot, activity,
                               local_engineering=local_engineering)

def run_window(args, built, **kwargs):
    namespace = dict(parent.run_window.__globals__, close_role=_closed, snapshot_role=_snapshot,
        take_role_records=_take, _validate_pool=validate_pools,
        pool_contract=lambda semantic:dict(candidate=CANDIDATE, pool_role='X',
            components=list(COMPONENTS), workers=[24,48,24,24]))
    return FunctionType(parent.run_window.__code__, namespace)(args, built, **kwargs)
