"""The inherited close-before-output clock, with active joint pools in both roles."""
from types import FunctionType
from tools.cpu_s4_6f_fullsize_parallel import measurement as parent
from tools.cpu_s4_6h_combined_parallel import measurement as joint
from .cost_factory import finalize_factory
from .protocol import CANDIDATE, COMPONENTS, candidate_spec
KEYS=joint.KEYS

def _closed(dispatch,semantic):
    candidate_spec(semantic)
    return finalize_factory(dispatch)

def _snapshot(dispatch,semantic):
    namespace=dict(joint._snapshot.__globals__,_closed=_closed)
    return FunctionType(joint._snapshot.__code__,namespace)(dispatch,semantic)

def _take(dispatch,semantic):
    namespace=dict(joint._take.__globals__,_closed=_closed)
    return FunctionType(joint._take.__code__,namespace)(dispatch,semantic)

def validate_pools(semantic,role,intervals,snapshot,activity,*,local_engineering):
    candidate_spec(semantic)
    if role not in ('R','X'):
        raise ValueError('S46I_EXTERNAL_ROLE')
    return joint.validate_pools(joint.CANDIDATE,'X',intervals,snapshot,activity,
                                local_engineering=local_engineering)

def run_window(args,built,**kwargs):
    namespace=dict(parent.run_window.__globals__)
    namespace.update(close_role=_closed,snapshot_role=_snapshot,take_role_records=_take,
                     _validate_pool=validate_pools,
                     pool_contract=lambda semantic:dict(candidate=CANDIDATE,pool_role='X',
                         components=list(COMPONENTS),workers=[24,48,24]))
    return FunctionType(parent.run_window.__code__,namespace)(args,built,**kwargs)
