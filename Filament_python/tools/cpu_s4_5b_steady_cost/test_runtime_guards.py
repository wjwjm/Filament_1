"""One real private factory: PASS -> specific scalar mutation reject -> PASS."""
import gc,json,sys,weakref
import pytest
from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from .candidate import build_role,verify_bindings
from .timing import TimingRecorder,TimingError

def test_s45b_real_factory_negative_binding_triplets(tmp_path):
    built=build_role('PURE_OFF_SMALLGRID',capture_root=tmp_path/'capture',local_engineering=True,pool_override=1,profile=True)
    dispatch=built[1];state=dispatch.cpu_s4_5b_state;pair=dispatch.cpu_s4_5b_recorder_pair;records=[]
    def check():assert verify_bindings(dispatch)['status']=='PASS_S45B_PRIVATE_BINDINGS'
    def triplet(name,mutate,restore,pattern):
        check();mutate()
        try:
            with pytest.raises(RuntimeError,match=pattern):verify_bindings(dispatch)
        finally:restore()
        check();records.append(dict(test=name,normal='PASS',rejection=pattern,restored='PASS'))
    stage=state._internal_stage_inner;namespace=stage.__globals__;name='__cpu_s45b_eq10_fft__';original=namespace[name]
    triplet('actual_stage_global_slot',lambda:namespace.__setitem__(name,lambda *a,**k:None),lambda:namespace.__setitem__(name,original),'GLOBAL_SLOT')
    old_code=stage.__code__
    triplet('actual_stage_code',lambda:setattr(stage,'__code__',old_code.replace(co_name='changed_stage')),lambda:setattr(stage,'__code__',old_code),'CODE|ACTUAL')
    seal=state._cpu_s45b_eq10_backend_seals['EQ10_SUM']['implementation_seal'];implementation=seal['function'];old_code=implementation.__code__
    triplet('original_sum_implementation',lambda:setattr(implementation,'__code__',old_code.replace(co_name='changed_sum')),lambda:setattr(implementation,'__code__',old_code),'BACKEND')
    triplet('scalar_companion_identity',lambda:setattr(pair.b,'_companion_identity','wrong'),lambda:setattr(pair.b,'_companion_identity',pair.identity_token),'COMPANION')
    # Existing G3 hook guard is exercised without a scientific call.
    state._g_check_monitoring_state()
    monitoring=getattr(sys,'monitoring',None)
    if monitoring is not None:
        unused=next(i for i in range(6) if monitoring.get_tool(i) is None)
        monitoring.use_tool_id(unused,'s45b-negative')
        try:
            with pytest.raises(RuntimeError,match='MONITOR'):state._g_check_monitoring_state()
        finally:monitoring.free_tool_id(unused)
        state._g_check_monitoring_state();records.append(dict(test='registered_monitoring_tool',normal='PASS',rejection='MONITOR',restored='PASS'))
    # The inherited fixed math helper retains its original exception/mask policy.
    import numpy as np
    import numba
    helper=state._cpu_s45a_private_namespace['_s43c_product']
    response=np.ones((384,17,19),dtype=np.float64);field=np.ones(response.shape,dtype=np.complex128)
    references=[weakref.ref(response),weakref.ref(field)]
    result=helper(response,field,np.float64);assert np.array_equal(result,field);del result
    runtime=state.r8_state._runtime;original_kernel=runtime['kernels']['product']
    class FixedFailure:
        signatures=[]
        def __call__(self,*unused):raise RuntimeError('S45B_FIXED_INHERITED_KERNEL_FAILURE')
    try:
        runtime['kernels']['product']=(FixedFailure(),FixedFailure())
        with pytest.raises(RuntimeError,match='S45B_FIXED_INHERITED_KERNEL_FAILURE'):helper(response,field,np.float64)
        assert numba.get_num_threads()==1
    finally:runtime['kernels']['product']=original_kernel
    result=helper(response,field,np.float64);assert np.array_equal(result,field);del result,response,field
    gc.collect();assert all(ref() is None for ref in references)
    check();assert numba.get_num_threads()==1
    timing=pair.s45a.snapshot();assert not timing['incomplete_scopes'] and sum(row['failed'] for row in timing['records'])==1
    records.append(dict(test='inherited_kernel_exception_mask_and_array_lifetime',normal='PASS',rejection='S45B_FIXED_INHERITED_KERNEL_FAILURE',restored='PASS',mask_after=1,arrays_collected=True))
    sg.write(tmp_path/'BINDING_NEGATIVE_TRIPLETS.json',dict(status='PASS_REAL_FACTORY_NEGATIVE_TRIPLETS',records=records,scientific_propagation_calls=0,inherited_math_calls=3,raw_failed_scope_preserved=timing))

def test_s45b_scope_negative_triplets():
    recorder=TimingRecorder(enabled=True,max_records=3)
    with pytest.raises(TimingError,match='UNKNOWN'):recorder.end(1)
    token=recorder.begin('EQ10_SUM');recorder.end(token)
    with pytest.raises(TimingError,match='ALREADY_ENDED'):recorder.end(token)
    with pytest.raises(RuntimeError,match='INTERVAL|ORDINAL'):recorder.interval_scope('INTERVAL_BODY',-1).__enter__()
    with recorder.scope('EQ10_SUM'):pass
    assert recorder.snapshot()['clock_closure']['pass']
