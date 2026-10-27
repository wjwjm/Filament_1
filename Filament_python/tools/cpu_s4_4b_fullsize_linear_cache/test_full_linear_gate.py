"""All four actual J2 return forms use the unchanged full-element gate."""
import copy
import numpy as np
import pytest
from . import candidate
from tools.cpu_s4_4a_linear_cache import linear_adapter
from tools.cpu_s4_3g3_smallgrid.scientific_context import frozen_tools

def test_four_actual_linear_forms_original_numeric_layout_alias_and_recovery(tmp_path):
    import KHz_filament.propagate as propagation
    factory=candidate.make_factory(local_engineering=True,kernel_threads=1,temporal_workers=1,spatial_workers=1,raman_workers=1,shape=(32,17,19))
    _,dispatch=factory(propagation.propagate_one_pulse);state=dispatch.cpu_s4_4a_state
    original=state._g_linear_function;cached=state._cpu_s4_4a_cached_linear
    _,exact=frozen_tools(tmp_path)
    rng=np.random.default_rng(109);field=(rng.normal(size=(32,17,19))+1j*rng.normal(size=(32,17,19))).astype(np.complex128)
    kwargs=dict(Omega=np.linspace(-2e14,2e14,32),kperp2=np.arange(17*19,dtype=np.float64).reshape(17,19)*1e4,k0=1e7,omega0=2e15,dz=5e-5,beta2=0.,denom_floor=1e-4,precision_strategy='baseline_complex64',energy_scale=1.)
    held=[]
    def counts():return copy.deepcopy(dispatch.report()['transform_counts'])
    def delta(a,b):return {k:int(b.get(k,0))-int(a.get(k,0)) for k in set(a)|set(b) if int(b.get(k,0))-int(a.get(k,0))}
    def call_cached(options):
        linear_adapter._validate_cached_runtime(state,original,cached)
        try:result=cached(field.copy(),**options)
        except BaseException:state._cpu_s4_4a_cache.finish_half(success=False);raise
        state._cpu_s4_4a_cache.finish_half(success=True);return result
    for index,(energy,profile) in enumerate(((False,False),(True,False),(False,True),(True,True))):
        options=dict(kwargs,return_energy_diagnostics=energy,return_profile_diagnostics=profile)
        before=counts();reference=original(field.copy(),**options);reference_delta=delta(before,counts())
        before=counts();actual=call_cached(options);actual_delta=delta(before,counts())
        assert reference_delta and sum(reference_delta.values())==66 and actual_delta==reference_delta
        verdict=exact.compare(reference,actual,tmp_path/f'FORM_{index}_EXACT.jsonl')
        assert verdict['status']=='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
        output=actual if not energy and not profile else actual[0]
        held.append((output,output.copy()))
        for retained_index,(old,copy0) in enumerate(held):
            assert exact.compare(copy0,old,tmp_path/f'HELD_{index}_{retained_index}.jsonl')['status']=='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
    private=state._cpu_s4_4a_private_binding;helper_name=private['helper_name'];helper=private['helper']
    baseline=call_cached(dict(kwargs,return_energy_diagnostics=False,return_profile_diagnostics=False))
    before=counts();private['namespace'][helper_name]=lambda *args:None
    with pytest.raises(linear_adapter.LinearBindingError,match='CACHED_HELPER_CHANGED_OR_UNKNOWN'):call_cached(kwargs)
    assert counts()==before
    private['namespace'][helper_name]=helper;recovered=call_cached(kwargs)
    assert exact.compare(baseline,recovered,tmp_path/'RECOVERED_EXACT.jsonl')['status']=='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
