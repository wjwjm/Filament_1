"""Only the two S4 engineering gates; synthetic cgroups and frozen tiny outputs."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
from tools.hpc_ops import provenance_v2 as pv
from KHz_filament.hr4e_timestep import sha256_array_streaming

HERE=Path(__file__).parent

def load(name):
    spec=importlib.util.spec_from_file_location('s4_gate_'+name,HERE/(name+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

m=load('memory_limits');o=load('output_inventory')


def v2(tmp_path, leaf='max', parent='1000', usage=50, parent_usage=100):
    proc=tmp_path/'proc/41';proc.mkdir(parents=True)
    (proc/'cgroup').write_text('0::/job/step\n')
    (proc/'mountinfo').write_text('1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n')
    root=tmp_path/'fs/sys/fs/cgroup';root.mkdir(parents=True)
    (root/'cgroup.controllers').write_text('memory cpu\n')
    for name,limit,current in [('job',parent,parent_usage),('job/step',leaf,usage)]:
        p=root/name;p.mkdir(exist_ok=True)
        (p/'memory.max').write_text(str(limit));(p/'memory.high').write_text('max');(p/'memory.current').write_text(str(current))
    return proc,root


def snap(tmp_path,allocation=2000):
    return m.snapshot_cgroup_memory(41,allocation_bytes=allocation,proc_root=tmp_path/'proc',virtual_fs_root=tmp_path/'fs')


@pytest.mark.parametrize('leaf,parent,allocation,expected', [('max',1000,2000,1000),(500,1000,2000,500),(1500,1000,2000,1000),(1500,1000,700,700)])
def test_s4_two_gates_effective_conflicts(tmp_path,leaf,parent,allocation,expected):
    v2(tmp_path,leaf,parent)
    r=snap(tmp_path,allocation);assert r['effective_limit_bytes']==expected
    assert len(r['levels_leaf_to_mount_root'])==3


@pytest.mark.parametrize('fraction',[.79,.8,.9,.95])
def test_s4_two_gates_ancestor_usage_threshold(tmp_path,fraction):
    v2(tmp_path,500,1000,10,int(1000*fraction))
    r=snap(tmp_path);f=m.memory_fractions(rss_bytes=5,pss_bytes=2,cgroup=r,allocation_bytes=2000)
    assert f['max_enforced_fraction']==fraction
    assert (f['max_enforced_fraction']>=.8)==(fraction>=.8)
    assert (f['max_enforced_fraction']>=.9)==(fraction>=.9)


def test_s4_two_gates_memory_high_is_not_hard_limit(tmp_path):
    _,root=v2(tmp_path);(root/'job/memory.high').write_text('20')
    assert snap(tmp_path)['effective_limit_bytes']==1000


@pytest.mark.parametrize('fault',['unreadable','malformed','hidden','ambiguous','root_subtree','missing_controller'])
def test_s4_two_gates_cgroup_fail_closed(tmp_path,fault):
    proc,root=v2(tmp_path)
    if fault=='unreadable':(root/'job/memory.max').unlink()
    elif fault=='malformed':(root/'job/memory.current').write_text('nan')
    elif fault=='hidden':(proc/'mountinfo').write_text('1 0 0:1 /job /sys/fs/cgroup rw - cgroup2 cgroup rw\n')
    elif fault=='ambiguous':(proc/'mountinfo').write_text((proc/'mountinfo').read_text()*2)
    elif fault=='root_subtree':(root/'memory.max').write_text('100')
    else:(root/'cgroup.controllers').write_text('cpu')
    with pytest.raises(m.CgroupMemoryError):snap(tmp_path)


def v1(tmp_path,hierarchy,reported):
    proc=tmp_path/'proc/41';proc.mkdir(parents=True)
    (proc/'cgroup').write_text('4:memory:/job/step\n')
    (proc/'mountinfo').write_text('1 0 0:1 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n')
    root=tmp_path/'fs/sys/fs/cgroup/memory';root.mkdir(parents=True)
    for name,limit,current,flag in [('',max(m._V1_UNLIMITED),100,1),('job',500,400,hierarchy),('job/step',1000,50,hierarchy)]:
        p=root/name;p.mkdir(exist_ok=True)
        (p/'memory.limit_in_bytes').write_text(str(limit));(p/'memory.usage_in_bytes').write_text(str(current));(p/'memory.use_hierarchy').write_text(str(flag))
    (root/'job/step/memory.stat').write_text('hierarchical_memory_limit '+str(reported)+'\n')
    return root


@pytest.mark.parametrize('hierarchy,expected',[(0,1000),(1,500)])
def test_s4_two_gates_v1_hierarchy(tmp_path,hierarchy,expected):
    v1(tmp_path,hierarchy,expected)
    assert snap(tmp_path)['effective_limit_bytes']==expected


def test_s4_two_gates_v1_hidden_limit(tmp_path):
    v1(tmp_path,1,200)
    with pytest.raises(m.CgroupMemoryError,match='unlocated'):snap(tmp_path)


def test_s4_two_gates_32bit_sized_limit_not_unlimited_on_64bit(tmp_path):
    if sys.maxsize>2**32:
        p=tmp_path/'limit';p.write_text(str(2**31-4096));assert m._v1_limit(p)==2**31-4096


@pytest.fixture(scope='session')
def generated_tiny_pair(tmp_path_factory):
    import subprocess
    root=tmp_path_factory.mktemp('s4_inventory_source')/'independent'
    proc=subprocess.run([sys.executable,'-s','-B',str(HERE/'supervisor.py'),
        '--supervise-pair','--tiny-science','--output-root',str(root)],
        capture_output=True,text=True,timeout=150)
    assert proc.returncode==0,proc.stdout+proc.stderr+(root/'PAIR_RECEIPT.json').read_text(encoding='utf8')
    return root


def dump(path,x):path.write_text(json.dumps(x,allow_nan=False),encoding='utf8')

def seal(case):
    w=case/'work';mode=json.loads((w/'child_receipt.json').read_text())['mode']
    names=['window_receipt.json','input_output_contract.json','scientific_Qacc.npy','scientific_scalar_diagnostics.json','candidate_events.jsonl','propagation/child_receipt.json','propagation/final_E.npy','propagation/final_POST.npz','propagation/diagnostics.npz','propagation/observer_progress.jsonl','propagation/linear_halfstep_profiles.json']+['propagation/'+n for n in o.SINK_FILES]
    if mode=='PROFILE':names.append('propagation/function_fft_profile.json')
    dump(w/'OUTPUT_SEAL.json',dict(schema='filament.cpu_s4.output_seal.v1',files={n:pv.raw_sha256_file(w/n) for n in names}))
    child=json.loads((w/'child_receipt.json').read_text());child['output_seal_sha256']=pv.raw_sha256_file(w/'OUTPUT_SEAL.json');child.pop('ledger_closed',None);dump(w/'child_receipt.json',child)


@pytest.fixture
def output_case(tmp_path,generated_tiny_pair):
    case=tmp_path/'case';shutil.copytree(generated_tiny_pair/'I_PROFILE',case)
    w=case/'work'
    dump(w/'input_output_contract.json',dict(schema='filament.cpu_s4.input_output_contract.v1',mode='PROFILE',implementation='I',interval_count=8,source_shape=[64,8,8],source_dtype='complex128',post_shape=[8,8,8],qacc_shape=[8,8],qacc_dtype='float64'))
    sup=json.loads((case/'supervisor_receipt.json').read_text());sup['mode']='PROFILE';dump(case/'supervisor_receipt.json',sup)
    seal(case);return case


def verify(case,mode='PROFILE',raw=pv.raw_sha256_file):
    return o.verify_output_inventory(case,mode,raw_sha256_file=raw,canonical_array_sha256=sha256_array_streaming)


def test_s4_two_gates_valid_inventory(output_case):
    r=verify(output_case);assert r['status']=='PASS',r['failures']
    assert r['ledger_closed'] and r['raw_hash_calls']>0


REQUIRED=['supervisor_receipt.json','work/child_receipt.json','work/window_receipt.json','work/input_output_contract.json','work/OUTPUT_SEAL.json','work/scientific_Qacc.npy','work/scientific_scalar_diagnostics.json','work/candidate_events.jsonl','work/propagation/child_receipt.json','work/propagation/final_E.npy','work/propagation/final_POST.npz','work/propagation/diagnostics.npz','work/propagation/observer_progress.jsonl','work/propagation/linear_halfstep_profiles.json','work/propagation/function_fft_profile.json']+['work/propagation/'+n for n in o.SINK_FILES]


@pytest.mark.parametrize('name',REQUIRED)
def test_s4_two_gates_every_required_file_missing(output_case,name):
    (output_case/name).unlink();r=verify(output_case);assert r['status']!='PASS'


@pytest.mark.parametrize('name',['propagation/final_E.npy','propagation/final_POST.npz','scientific_scalar_diagnostics.json','candidate_events.jsonl'])
def test_s4_two_gates_truncated_write(output_case,name):
    p=output_case/'work'/name;b=p.read_bytes();p.write_bytes(b[:len(b)//2]);seal(output_case)
    assert verify(output_case)['status']!='PASS'


@pytest.mark.parametrize('name',['scientific_Qacc.npy']+['propagation/'+n for n in o.SINK_FILES])
def test_s4_two_gates_same_shape_finite_tamper(output_case,name):
    p=output_case/'work'/name;x=np.load(p);x.flat[0]+=1;np.save(p,x)
    assert verify(output_case)['status']!='PASS'


@pytest.mark.parametrize('fault',['shape','dtype','scalar','post','hash','false_claim','mutation','nan_json'])
def test_s4_two_gates_contract_failures(output_case,fault):
    w=output_case/'work'
    if fault in ('shape','dtype'):
        p=w/'scientific_Qacc.npy';x=np.load(p);np.save(p,x[:1] if fault=='shape' else x.astype(np.float32));seal(output_case)
    elif fault=='scalar':
        p=w/'scientific_scalar_diagnostics.json';x=json.loads(p.read_text());x['fields'].pop(next(iter(x['fields'])));dump(p,x);seal(output_case)
    elif fault=='post':
        p=w/'propagation/final_POST.npz';np.savez(p,delta_n=np.zeros((8,8,8)));seal(output_case)
    elif fault=='hash':
        p=w/'propagation/child_receipt.json';x=json.loads(p.read_text());x['final_E_raw_sha256']='0'*64;dump(p,x);seal(output_case)
    elif fault=='false_claim':
        p=w/'child_receipt.json';x=json.loads(p.read_text());x['ledger_closed']=True;dump(p,x);(w/'OUTPUT_SEAL.json').unlink()
    elif fault=='nan_json':
        (w/'window_receipt.json').write_text('{"bad":NaN}');seal(output_case)
    else:
        calls=[0]
        def raw(path):
            h=pv.raw_sha256_file(path);calls[0]+=1
            if calls[0]==3:
                p=w/'child_receipt.json';p.write_bytes(p.read_bytes()+b' ')
            return h
        assert verify(output_case,raw=raw)['status']!='PASS';return
    assert verify(output_case)['status']!='PASS'


@pytest.mark.parametrize('suffix',['json','jsonl'])
def test_s4_two_gates_strict_json_numeric_overflow(tmp_path,suffix):
    p=tmp_path/('x.'+suffix);p.write_text('{"value":1e999}\n')
    with pytest.raises(ValueError):
        (o._strict_json if suffix=='json' else o._strict_jsonl)(p)


def test_s4_two_gates_hybrid_does_not_hide_bad_v2(tmp_path):
    proc,root=v2(tmp_path)
    (proc/'cgroup').write_text('0::/job/step\n4:memory:/job/step\n')
    (root/'job/memory.max').write_text('corrupt')
    with pytest.raises(m.CgroupMemoryError,match='invalid_v2'):
        snap(tmp_path)



def test_s4_two_gates_progress_read_precedes_upper_timestamp(tmp_path,monkeypatch):
    import os,time
    supervisor=load('supervisor')
    # A child may publish a start while the progress file is being read.
    monkeypatch.setattr(supervisor,'progress_state',lambda path:(time.monotonic(),set(range(8))))
    monkeypatch.setattr(supervisor,'resource_snapshot',lambda pid:{'available':False})
    row=supervisor.supervise_child([sys.executable,'-s','-B','-c','import time;time.sleep(.1)'],tmp_path/'case',env=os.environ.copy(),sample_seconds=.01)
    assert row['status']=='COMPLETE',row

@pytest.mark.parametrize('dtype,bad',[('float64','invalid_hex'),('float64','+Infinity'),('bool','false'),('int64',1.5)])
def test_s4_two_gates_scalar_encoding(output_case,dtype,bad):
    p=output_case/'work/scientific_scalar_diagnostics.json'
    x=json.loads(p.read_text())
    name=next(k for k,v in x['fields'].items() if v['dtype']==dtype)
    x['fields'][name]['value']=bad
    dump(p,x);seal(output_case)
    result=verify(output_case)
    assert result['status']!='PASS' and result['failures'] and not result.get('ledger_closed', False)
