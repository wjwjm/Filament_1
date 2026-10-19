"""Additional real-entry gates; original regression files remain frozen."""
import json, math, importlib.util, sys
from pathlib import Path
import pytest
from tools.cpu_o1.cpu_s4.test_s4_two_gates import output_case, generated_tiny_pair, verify, load, o

def test_s4_two_gates_profile_receipt_size_cap(output_case,monkeypatch):
    monkeypatch.setattr(o,'MAX_PROFILE_RECEIPT_BYTES',16)
    receipt=output_case/'work/propagation/function_fft_profile.json'
    receipt.write_bytes(b'x'*17)
    result=verify(output_case)
    assert result['status']!='PASS'
    assert any('PROFILE_receipt_exceeds_512_MiB_cap' in item
               for item in result['failures'])

def test_s4_two_gates_storage_budget_uses_frozen_diagnostic_count():
    budget=load('storage_budget')
    inputs=[{'identity_status':'PASS','size_bytes':1} for _ in range(14)]
    result=budget.estimate_storage_budget(source_shape=(384,351,301),
        input_records=inputs,diagnostic_field_count=7,current_bytes=0,
        cap_bytes=budget.FROZEN_CAP_MAX_BYTES,
        filesystem_available_bytes=budget.FROZEN_CAP_MAX_BYTES,
        filesystem_available_inodes=1000,
        quota_available_bytes=budget.FROZEN_CAP_MAX_BYTES,
        quota_available_inodes=1000)
    assert result['diagnostic_field_count']==7
    assert result['diagnostic_archive_upper_bound_bytes']==7*351*301*16

@pytest.mark.parametrize('value',[None,0,-1,float('nan'),float('inf'),True])
def test_s4_real_entry_partial_ratios(value):
    s=load('supervisor')
    assert s._ratio_and_overhead(value,1)==(None,None)


def test_s4_real_entry_quota_and_no_linux(tmp_path,monkeypatch):
    from tools.cpu_o1.cpu_s4 import runtime_gate as g
    header='Filesystem type KB quota limit in_doubt grace | files quota limit in_doubt grace Remarks\n'
    row='publicfs01 USR 100 200 300 10 none | 5 20 30 2 none publicfs01.para.bscc'
    q=g.quota_available(header+row)
    assert q==dict(available_bytes=90*1024,available_inodes=13)
    for bad in (row,header.replace('KB','MB')+row,header+row.replace('|','bad')):
        with pytest.raises(ValueError):g.quota_available(bad)
    with pytest.raises(ValueError):g.quota_available('unavailable')
    monkeypatch.setattr(g.sys,'platform','win32')
    with pytest.raises(RuntimeError,match='LINUX_ALLOCATION'):
        g.run({},approval_path=tmp_path/'a',config_path=tmp_path/'c',output_root=tmp_path/'run')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('fault',['missing','skipped','failure'])
def test_s4_real_entry_junit_fail_closed(tmp_path,fault):
    from tools.cpu_o1.cpu_s4.runtime_gate import validate_junit
    p=tmp_path/'tests.xml';p.write_text('<testsuite><testcase name="tiny_independent_exact">'+('' if fault=='missing' else '<'+fault+'/>')+'</testcase></testsuite>')
    with pytest.raises(RuntimeError):validate_junit(p)


@pytest.mark.parametrize('bad',[True,1.5,float('nan'),-1])
def test_s4_real_entry_storage_rejects_invalid_counts(bad):
    b=load('storage_budget')
    with pytest.raises(ValueError):
        b.estimate_storage_budget(source_shape=(384,351,301),input_records=[dict(identity_status='PASS',size_bytes=1)]*14,
            diagnostic_field_count=7,current_bytes=bad,cap_bytes=b.FROZEN_CAP_MAX_BYTES,
            filesystem_available_bytes=100*1024**3,filesystem_available_inodes=1000,
            quota_available_bytes=100*1024**3,quota_available_inodes=1000)


def test_s4_real_entry_missing_current_overlay(tmp_path):
    from tools.cpu_o1.cpu_s4 import admission as a
    from tools.hpc_ops import provenance_v2 as pv
    (tmp_path/a.S4_TOOL_PREFIX).mkdir(parents=True)
    with pytest.raises(a.AdmissionBlocked):a._bind_s4_overlay_inventory(tmp_path,[],pv)


def test_s4_real_entry_sealed_boundary_tamper(tmp_path,generated_tiny_pair):
    import shutil
    output_case=tmp_path/"case"
    shutil.copytree(generated_tiny_pair/"I_PROFILE",output_case)
    import numpy as np
    from tools.hpc_ops import provenance_v2 as pv
    s=load('supervisor');p=output_case/'work/first_boundary/Qacc.npy'
    arr=np.load(p);arr.flat[0]+=1;np.save(p,arr)
    with pytest.raises(RuntimeError,match='HASH_MISMATCH'):
        s._verify_first_boundary_acceptance(output_case,'PROFILE',raw_sha256_file=pv.raw_sha256_file)


@pytest.mark.parametrize('value',[1.,float('nan')])
def test_s4_real_entry_first_boundary_stops_before_second(tmp_path,generated_tiny_pair,value):
    import shutil, numpy as np
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1
    from tools.cpu_o1.cpu_s4 import profile
    from tools.cpu_o1 import q1_t0_real
    reference=tmp_path/'reference';shutil.copytree(generated_tiny_pair/'I_OFF/work/first_boundary',reference)
    p=reference/'E.npy';x=np.load(p);x.flat[0]=value;np.save(p,x)
    context=j1._tiny_context(operator_mode='full_isaacs_eq27')
    context.update(real_input_used=False,_local_kernel_threads=1,source_hash=j1._canonical_hash(context['source']),
        assets={'pre0':context['pre0'],'pre0_canonical_sha256':j1._canonical_hash(context['pre0'])})
    case=tmp_path/'candidate'
    with pytest.raises(q1_t0_real.Q1ControlledStop):
        profile._run_one(context,case,profile=True,case_id='I_PROFILE',first_boundary_reference=('I_OFF',reference))
    rows=[json.loads(line) for line in (case/'propagation/observer_progress.jsonl').read_text().splitlines()]
    assert not any(row.get('interval_index',0)>0 for row in rows)
    assert not (case/'propagation/final_E.npy').exists()


def test_s4_real_entry_service_accounting(generated_tiny_pair):
    row=json.loads((generated_tiny_pair/'PAIR_RECEIPT.json').read_text())
    a,b=row['cases'];assert a['cross_window_comparison_seconds']==0
    for case in (a,b):
        assert case['full_service_wall_seconds']==pytest.approx(case['full_service_wall_seconds_raw_including_cross_window_comparison']-case['cross_window_comparison_seconds'])
        assert case['parent_output_acceptance_seconds']>0
    assert row['instrumentation_overhead_fraction']==pytest.approx(b['full_service_wall_seconds']/a['full_service_wall_seconds']-1)
    assert row['propagation_overhead_fraction']==pytest.approx(b['propagation_wall_seconds']/a['propagation_wall_seconds']-1)


def test_s4_real_entry_allocation_clock():
    from datetime import datetime
    from tools.cpu_o1.cpu_s4.runtime_gate import allocation_clock
    start=datetime.fromisoformat('2026-10-18T12:00:00').timestamp()
    raw='JobId=123 JobState=RUNNING TimeLimit=02:00:00 StartTime=2026-10-18T12:00:00'
    assert allocation_clock(raw,'123',start+10)==start
    for bad in (raw.replace('123','124'),raw.replace('RUNNING','PENDING'),raw.replace('02:00:00','UNLIMITED'),raw.replace('2026-10-18T12:00:00','Unknown'),raw.replace('StartTime','Missing')):
        with pytest.raises((ValueError,KeyError)):
            allocation_clock(bad,'123',start+10)
    for now in (start-1,start+7200):
        with pytest.raises(ValueError): allocation_clock(raw,'123',now)
