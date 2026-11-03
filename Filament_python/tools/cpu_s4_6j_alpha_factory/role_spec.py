"""Both external COST6 roles use the same four-component mathematical factory."""
import copy
import importlib
from types import FunctionType
from tools.cpu_s4_4b_fullsize_linear_cache import candidate as cache_candidate
from tools.cpu_s4_4b_fullsize_linear_cache import contract as cache_contract
from tools.cpu_s4_5b_steady_cost.adapter import make_recorder_pair
from tools.cpu_s4_6i_residual_parallel import selected_factory as selected
from .cost_hook import AlphaCostHook
from .factory_timing import FactoryTiming
from .protocol import CANDIDATE, MICRO_SHAPE, SHAPE, candidate_spec

INSTRUMENTED_MODULES = (
    'tools.cpu_s4_6i_residual_parallel.power_hook',
    'tools.cpu_s4_6i_residual_parallel.selected_factory',
    'tools.cpu_s4_6h_combined_parallel.combined_candidate',
    'tools.cpu_s4_6e_array_propagation.c_candidate',
)

def build_role(semantic, mode, *, capture_root, role, local_engineering,
               recorder_pair, cost_mode, shape=None, pool_override=None):
    candidate_spec(semantic)
    if role not in ('R','X') or cost_mode not in ('OFF','COST'):
        raise ValueError('S46J_ROLE_COST_DOMAIN')
    expected = MICRO_SHAPE if local_engineering else SHAPE
    if shape is not None and tuple(shape) != expected:
        raise ValueError('S46J_GRID_DOMAIN')
    cache_contract.validate_shape(expected, scope=(cache_contract.LOCAL_ENGINEERING_SCOPE
        if local_engineering else cache_contract.FULLSIZE_SCOPE))
    if recorder_pair is None:
        recorder_pair = make_recorder_pair(False)
    modules = [importlib.import_module(name) for name in INSTRUMENTED_MODULES]
    timing = FactoryTiming(recorder_pair.b, enabled=cost_mode=='COST',
        extra_source_paths=tuple(module.__file__ for module in modules)+(__file__,),
        instrumentation_aliases=tuple(name+'.factory_scope' for name in INSTRUMENTED_MODULES))
    def construct(pool, *, capture_store=None):
        return AlphaCostHook(pool, recorder=recorder_pair.b, enabled=cost_mode=='COST', capture_store=capture_store)
    namespace = dict(selected.build_role.__globals__, PowerVolumeHook=construct)
    factory = FunctionType(selected.build_role.__code__, namespace, argdefs=selected.build_role.__defaults__)
    factory.__kwdefaults__ = selected.build_role.__kwdefaults__
    built = None
    try:
        with timing.activate():
            built = factory('PURE_OFF_SMALLGRID' if mode=='PURE_OFF_FULLSIZE' else mode,
                capture_root=capture_root, role='X', local_engineering=local_engineering,
                recorder_pair=recorder_pair, pool_override=pool_override)
    except BaseException as error:
        if built is not None:
            try:
                selected.finalize_factory(built[1], reason='FACTORY_TIMING_FAILURE')
            except BaseException as cleanup:
                error.add_note('S46J_CLOSE_FAILURE:'+repr(cleanup))
        raise
    dispatch = built[1]
    dispatch.cpu_s4_6j_factory_timing = timing
    dispatch.cpu_s4_6j_shape = expected
    dispatch.cpu_s4_6j_cost_mode = cost_mode
    try:
        cache = cache_candidate._replace_empty_cache(dispatch, local_engineering=local_engineering,
                                                     configured_shape=expected)
        binding = copy.deepcopy(built[3])
        binding.update(semantic_id=CANDIDATE, shape=list(expected), fullsize_cache_binding=cache,
                       external_role=role, pool_role='X', four_components_enabled=True,
                       cost_mode=cost_mode, factory_timing_binding=timing.verify_binding())
        return built[:3] + (binding,) + built[4:]
    except BaseException:
        close_role(dispatch, semantic)
        raise

def verify_role(dispatch, semantic=CANDIDATE):
    candidate_spec(semantic)
    if dispatch.cpu_s4_6i_selected_role != 'X' or type(dispatch.cpu_s4_6i_power_hook) is not AlphaCostHook:
        raise RuntimeError('S46J_FOUR_COMPONENT_FACTORY_REQUIRED')
    result = selected.verify_bindings(dispatch)
    cache = dispatch.cpu_s4_4b_cache_snapshot()
    if tuple(cache.get('configured_shape', ())) != tuple(dispatch.cpu_s4_6j_shape):
        raise RuntimeError('S46J_CACHE_SHAPE_BINDING_CHANGED')
    return dict(combined=result, cache=cache, semantic_id=CANDIDATE,
                factory_timing=dispatch.cpu_s4_6j_factory_timing.verify_binding())

def close_role(dispatch, semantic=CANDIDATE):
    candidate_spec(semantic)
    if isinstance(getattr(dispatch, '_cpu_s4_6i_finalized', None), dict):
        return selected.finalize_factory(dispatch)
    try:
        with dispatch.cpu_s4_6j_factory_timing.activate():
            return selected.finalize_factory(dispatch)
    except BaseException as error:
        try:
            selected.finalize_factory(dispatch, reason='FACTORY_TIMING_FAILURE')
        except BaseException as cleanup:
            error.add_note('S46J_CLOSE_FAILURE:'+repr(cleanup))
        raise
