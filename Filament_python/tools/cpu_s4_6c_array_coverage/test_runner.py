"""Small synthetic control-chain checks; no real-input qualification."""
import json
import numpy as np
import numba
import pytest
from pathlib import Path
import zipfile
from . import runner,chains


def test_s46c_fresh_factory_and_reference_chain(tmp_path):
    from tools.cpu_s4_checkpoint_support import support_module
    exact=support_module('g1_exact')
    built,state,fft=runner.fresh_factory(tmp_path/'capture',local=True)
    refs=chains.source_references(fft)
    rng=np.random.default_rng(6)
    E=np.ascontiguousarray((rng.normal(size=(32,17,19))+1j*rng.normal(size=(32,17,19)))*1e5)
    # Load the actual frozen config through the original loader, not a
    # hand-built object that accidentally supplies dropped JSON attributes.
    from KHz_filament.confio import load_all
    from types import SimpleNamespace
    metadata=Path(__file__).resolve().parents[3]/'reports/CPU_S4_6C_REAL_ARRAY_COVERAGE_20261031/T0_INPUT_METADATA.stdout.json'
    config=tmp_path/'config.json';config.write_text(json.dumps(json.loads(metadata.read_text())['config']),encoding='utf8')
    grid,beam,prop,ion,heat,run,raman=load_all(str(config))
    assert not hasattr(raman,'method')
    kwargs=runner.stage_kwargs(dict(raman=raman,beam=beam,axes=SimpleNamespace(Omega=np.asarray(2*np.pi*np.fft.fftfreq(32,1e-15)),dt=1e-15)))
    assert kwargs['method']=='iir' and kwargs['chunk_pixels']==65536
    actual=state.stage(E,**kwargs)
    assert state.stats['reference_fallback_calls']==0
    snapshot=tmp_path/'snapshot'
    exact.save_tree(snapshot,dict(actual),compressed=True)
    with zipfile.ZipFile(snapshot/'ARRAYS.zip') as archive:
        restored=runner.decode(json.loads((snapshot/'TREE.json').read_text()),archive,exact)
    assert exact.fingerprints(restored)==json.loads((snapshot/'FINGERPRINTS.json').read_text())
    exact.compare(restored,dict(actual),tmp_path/'snapshot_roundtrip.jsonl')
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
