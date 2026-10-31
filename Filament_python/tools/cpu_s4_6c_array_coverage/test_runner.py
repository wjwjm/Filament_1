"""Small synthetic control-chain checks; no real-input qualification."""
import json
import numpy as np
import numba
import pytest
from . import runner,chains


def test_s46c_fresh_factory_and_reference_chain(tmp_path):
    from tools.cpu_s4_checkpoint_support import support_module
    exact=support_module('g1_exact')
    built,state,fft=runner.fresh_factory(tmp_path/'capture',local=True)
    refs=chains.source_references(fft)
    rng=np.random.default_rng(6)
    E=np.ascontiguousarray((rng.normal(size=(32,17,19))+1j*rng.normal(size=(32,17,19)))*1e5)
    kwargs=dict(Omega=np.asarray(2*np.pi*np.fft.fftfreq(32,1e-15)),dt=1e-15,omega0=2.35e15,n0=1.00027,
                n_R=2.3e-23,omega_R=1.6e13,Gamma_R=1.3e13,method='iir',chunk_pixels=65536,iir_sampling='exact_piecewise_linear')
    actual=state.stage(E,**kwargs)
    assert state.stats['reference_fallback_calls']==0
    I=refs[0](E,kwargs['n0'])
    energy=refs[1](I,actual['I_R'],kwargs['Omega'],kwargs['dt'],kwargs['n_R'])
    exact.compare({k:actual[k] for k in energy},energy,tmp_path/'energy.jsonl')
    data=(E,I,actual['I_R'],kwargs)
    for chain in ('A','B'):
        output=runner.call(chain,('REF',0),data,None,refs,fft)
        identity=exact.fingerprints(output)
        rows=runner.measure_block(chain,'synthetic',data,refs,fft,exact,tmp_path,[('REF',0),('P',1)],0,
                                  'ENGINEERING_ONLY',identity,expected_mask=1)
        assert len(rows)==10 and all(r['wall_s']>0 for r in rows)
    assert numba.get_num_threads()==1


def test_s46c_existing_nan_gate_is_not_relaxed(tmp_path):
    from tools.cpu_s4_checkpoint_support import support_module
    exact=support_module('g1_exact')
    with pytest.raises(RuntimeError,match='FIRST_SCIENTIFIC_NONEXACT'):
        exact.compare(np.array([np.nan]),np.array([np.nan]),tmp_path/'nan.jsonl')


def test_s46c_local48_requires_both_order_blocks():
    rows=[dict(chain='A',input='x',block=b,mode='P',workers=w,wall_s=t)
          for b,w,t in [(0,24,1),(0,48,.9),(1,24,1),(1,48,1.01)]]
    assert not runner.allow_local48(rows,'A','x')
    rows[-1]['wall_s']=.96
    assert runner.allow_local48(rows,'A','x')
