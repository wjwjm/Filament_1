"""OFF/COST both use the accepted S4-6J X five-pool factory."""
import copy
from tools.cpu_s4_4b_fullsize_linear_cache import contract as cache_contract
from tools.cpu_s4_5b_steady_cost.adapter import make_recorder_pair
from tools.cpu_s4_6j_alpha_factory import alpha_factory, alpha_role_spec as alpha_roles
from tools.cpu_s4_6j_alpha_factory.alpha_protocol import CANDIDATE as ALPHA
from .protocol import CANDIDATE, COMPONENTS, MICRO_SHAPE, SHAPE, candidate_spec
from .cost_hook import FiveChainCostHook

def build_role(semantic, mode, *, capture_root, role, local_engineering,
               recorder_pair, cost_mode, shape=None, pool_override=None):
    candidate_spec(semantic)
    if (role, cost_mode) not in (('R','OFF'),('X','COST')):
        raise ValueError('S46K_ROLE_COST_MODE_BINDING')
    expected = MICRO_SHAPE if local_engineering else SHAPE
    if shape is not None and tuple(shape) != expected:
        raise ValueError('S46K_GRID_DOMAIN')
    cache_contract.validate_shape(expected, scope=(cache_contract.LOCAL_ENGINEERING_SCOPE
        if local_engineering else cache_contract.FULLSIZE_SCOPE))
    pair = recorder_pair if recorder_pair is not None else make_recorder_pair(False)
    hook = FiveChainCostHook(pair.b, enabled=cost_mode=='COST')
    built = alpha_factory.build_role(
        'PURE_OFF_SMALLGRID' if mode=='PURE_OFF_FULLSIZE' else mode,
        capture_root=capture_root, role='X', local_engineering=local_engineering,
        recorder_pair=pair, shape=expected, pool_override=pool_override,
        residual_cost_hook=hook)
    dispatch = built[1]
    try:
        dispatch.cpu_s4_6k_cost_hook = hook
        dispatch.cpu_s4_6k_external_role = role
        dispatch.cpu_s4_6k_cost_mode = cost_mode
        binding = copy.deepcopy(built[3])
        binding.update(semantic_id=CANDIDATE, components=list(COMPONENTS),
            external_role=role, internal_role='X', alpha_enabled=True,
            five_components_enabled=True, cost_mode=cost_mode,
            cost_binding=hook.verify_binding(getattr(dispatch,'cpu_s4_6e_c_active_globals',None)))
        return built[:3]+(binding,)+built[4:]
    except BaseException:
        alpha_factory.finalize_factory(dispatch, reason='COST_BINDING_FAILURE')
        raise

def verify_role(dispatch, semantic=CANDIDATE):
    candidate_spec(semantic)
    if dispatch.cpu_s4_6j_external_role != 'X' or dispatch.cpu_s4_6j_alpha_pool is None:
        raise RuntimeError('S46K_FIVE_CHAIN_REFERENCE_REQUIRED')
    check = alpha_factory.verify_bindings(dispatch)
    hook = dispatch.cpu_s4_6k_cost_hook
    if type(hook) is not FiveChainCostHook:
        raise RuntimeError('S46K_COST_HOOK_IDENTITY')
    return dict(five_chain=check, cost=hook.verify_binding(
        getattr(dispatch,'cpu_s4_6e_c_active_globals',None)), semantic_id=CANDIDATE)

def close_role(dispatch, semantic=CANDIDATE):
    candidate_spec(semantic)
    return alpha_factory.finalize_factory(dispatch)

def snapshot_role(dispatch, semantic=CANDIDATE):
    candidate_spec(semantic)
    return alpha_roles.snapshot_role(dispatch, ALPHA)

def take_role_records(dispatch, semantic=CANDIDATE):
    candidate_spec(semantic)
    return alpha_roles.take_role_records(dispatch, ALPHA)

def pool_contract(semantic=CANDIDATE):
    candidate_spec(semantic)
    value = alpha_roles.pool_contract(ALPHA)
    return dict(value, candidate=CANDIDATE, alpha_always_enabled=True)
