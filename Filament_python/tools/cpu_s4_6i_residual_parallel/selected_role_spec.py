"""Private selected role adapter; the six-tuple and L1 capacity are unchanged."""
import copy
from tools.cpu_s4_4b_fullsize_linear_cache import candidate as cache_candidate
from tools.cpu_s4_4b_fullsize_linear_cache import contract as cache_contract
from . import selected_factory as factory
from .selected_protocol import CANDIDATE, MICRO_SHAPE, SHAPE, candidate_spec

def build_role(semantic,mode,*,capture_root,role,local_engineering,recorder_pair,
               shape=None,pool_override=None,capture_store=None):
    candidate_spec(semantic)
    expected = MICRO_SHAPE if local_engineering else SHAPE
    if shape is not None and tuple(shape)!=expected:
        raise ValueError('S46I_SELECTED_GRID_DOMAIN')
    cache_contract.validate_shape(expected,scope=(cache_contract.LOCAL_ENGINEERING_SCOPE
                                  if local_engineering else cache_contract.FULLSIZE_SCOPE))
    built = factory.build_role('PURE_OFF_SMALLGRID' if mode=='PURE_OFF_FULLSIZE' else mode,
        capture_root=capture_root,role=role,local_engineering=local_engineering,
        recorder_pair=recorder_pair,pool_override=pool_override,capture_store=capture_store)
    try:
        cache = cache_candidate._replace_empty_cache(built[1],
                    local_engineering=local_engineering,configured_shape=expected)
        built[1].cpu_s4_6i_shape=expected
        binding=copy.deepcopy(built[3])
        binding.update(semantic_id=CANDIDATE,shape=list(expected),
                       fullsize_cache_binding=cache,pool_role='X',external_role=role)
        return built[:3]+(binding,)+built[4:]
    except BaseException:
        factory.finalize_factory(built[1],reason='CACHE_CONSTRUCTION_FAILURE')
        raise

def verify_role(dispatch,semantic=CANDIDATE):
    candidate_spec(semantic)
    result=factory.verify_bindings(dispatch)
    cache=dispatch.cpu_s4_4b_cache_snapshot()
    if tuple(cache.get('configured_shape',()))!=tuple(dispatch.cpu_s4_6i_shape):
        raise RuntimeError('S46I_SELECTED_CACHE_SHAPE_BINDING_CHANGED')
    return dict(combined=result,cache=cache,semantic_id=CANDIDATE)

def close_role(dispatch,semantic=CANDIDATE):
    candidate_spec(semantic)
    return factory.finalize_factory(dispatch)
