"""Same application clock and five-chain lifecycle on both timing roles."""
from types import FunctionType
from tools.cpu_s4_6f_fullsize_parallel import measurement as parent
from tools.cpu_s4_6j_alpha_factory import alpha_measurement as five
from tools.cpu_s4_6j_alpha_factory.alpha_protocol import CANDIDATE as ALPHA
from .protocol import CANDIDATE, candidate_spec
from .role_spec import close_role, snapshot_role, take_role_records, pool_contract

KEYS = five.KEYS

def _snapshot(dispatch, semantic=CANDIDATE):
    return snapshot_role(dispatch, semantic)

def _take(dispatch, semantic=CANDIDATE):
    if getattr(dispatch,'_s46k_measurement_records_taken',False):
        return []
    result = take_role_records(dispatch, semantic)
    dispatch._s46k_measurement_records_taken = True
    return result

def validate_pools(semantic, role, intervals, snapshot, activity, *, local_engineering):
    candidate_spec(semantic)
    if role not in ('R','X'):
        raise RuntimeError('S46K_EXTERNAL_ROLE')
    return five.validate_pools(ALPHA, 'X', intervals, snapshot, activity,
        local_engineering=local_engineering)

def run_window(args, built, **kwargs):
    candidate_spec(args.semantic_candidate)
    namespace = dict(parent.run_window.__globals__, close_role=close_role,
        snapshot_role=_snapshot, take_role_records=_take,
        _validate_pool=validate_pools, pool_contract=pool_contract)
    return FunctionType(parent.run_window.__code__, namespace)(args, built, **kwargs)
