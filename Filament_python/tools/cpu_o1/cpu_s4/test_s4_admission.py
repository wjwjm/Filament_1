"""Tiny/fake-only S4 preparation tests; no real arrays or scheduler commands."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import psutil
import pytest

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('s4_supervisor_test',HERE/'supervisor.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
spec=importlib.util.spec_from_file_location('s4_admission_test',HERE/'admission.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def cli(*args, timeout=45):
    return subprocess.run([sys.executable,'-s','-B',str(HERE/'supervisor.py'),*map(str,args)],capture_output=True,text=True,timeout=timeout)

def test_s4_admission_light_parent_and_real_refusal(tmp_path):
    p=cli('--preflight-only');assert p.returncode==0
    assert json.loads(p.stdout)['parent_science_modules']==[]
    for mode in [('--single-mode','PROFILE'),('--supervise-pair',)]:
        p=cli(*mode,'--output-root',tmp_path/'real')
        assert p.returncode!=0 and 'BLOCKED_REAL_EXECUTION' in p.stderr
        assert not (tmp_path/'real').exists()

def test_s4_admission_fake_pair_and_overwrite(tmp_path):
    root=tmp_path/'pair';p=cli('--supervise-pair','--dry-run','--output-root',root,'--sample-seconds','.05')
    assert p.returncode==0,p.stdout+p.stderr
    row=json.loads((root/'PAIR_RECEIPT.json').read_text())
    assert row['status']=='DRY_RUN_PASS' and row['parent_science_modules_before_launch']==[]
    assert len({c['child_pid'] for c in row['cases']})==2
    assert all(c['complete_intervals']==list(range(8)) for c in row['cases'])
    assert cli('--supervise-pair','--dry-run','--output-root',root).returncode!=0

@pytest.mark.parametrize('fixture,expected',[('exit','NONZERO_CHILD_EXIT'),('hang','FIRST_INTERVAL_TIMEOUT'),('spawn','FIRST_INTERVAL_TIMEOUT'),('coverage','profile coverage'),('ledger','ledger'),('memory80','MEMORY_80'),('memory90','MEMORY_90')])
def test_s4_admission_failure_stops_matrix(tmp_path,fixture,expected):
    mode='PROFILE' if fixture=='coverage' else 'OFF'
    root=tmp_path/fixture
    p=cli('--supervise-pair','--dry-run','--output-root',root,'--fixture',fixture,'--fail-mode',mode,'--first-limit','2','--window-limit','10','--sample-seconds','.05')
    assert p.returncode!=0
    row=json.loads((root/'PAIR_RECEIPT.json').read_text())
    assert row['status']=='BLOCKED' and expected in json.dumps(row)
    if mode=='OFF':assert len(row['cases'])==1
    desc=root/'I_OFF/work/descendant.json'
    if desc.exists():assert not psutil.pid_exists(json.loads(desc.read_text())['pid'])

@pytest.mark.parametrize('method',['kill','terminate','interrupt'])
def test_s4_admission_parent_loss_cleans_child_tree(tmp_path,method):
    root=tmp_path/method
    command=[sys.executable,'-s','-B',str(HERE/'supervisor.py'),'--supervise-pair','--dry-run','--fixture','spawn','--output-root',str(root),'--sample-seconds','.05']
    p=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    childfile=root/'I_OFF/work/descendant.json';until=time.monotonic()+15
    try:
        while not childfile.exists() and time.monotonic()<until:
            assert p.poll() is None
            time.sleep(.05)
        assert childfile.exists()
        descendant=json.loads(childfile.read_text())['pid']
        guardian=json.loads((root/'I_OFF/work/guardian.json').read_text())['pid']
        if method=='interrupt' and os.name!='nt':p.send_signal(signal.SIGINT)
        elif method=='terminate':p.terminate()
        else:p.kill() # Windows kill/TERM: validates guardian rather than console Ctrl-C.
        p.wait(timeout=8)
        until=time.monotonic()+8
        while (psutil.pid_exists(descendant) or psutil.pid_exists(guardian)) and time.monotonic()<until:time.sleep(.05)
        assert not psutil.pid_exists(descendant)
        assert not psutil.pid_exists(guardian)
    finally:
        if p.poll() is None:
            for c in psutil.Process(p.pid).children(recursive=True):
                try:c.kill()
                except psutil.Error:pass
            p.kill();p.wait(timeout=5)

@pytest.mark.parametrize('change',[1.,-0.,float('nan')])
def test_s4_admission_numeric_nonexact_rejected(change):
    from tools.cpu_o1.cpu_o1 import compare_numeric_bounded
    from tools.cpu_o1.numeric_gate import classify_comparison
    metrics=compare_numeric_bounded(np.array([0.]),np.array([change]))
    assert not classify_comparison(dict(field_name='science',scope='science',metrics=metrics))['proceed']

def test_s4_admission_frozen_timing_and_missing_metrics():
    from tools.cpu_o1.numeric_gate import classify_comparison
    assert classify_comparison(dict(field_name='linear_walltime_step_s',scope='timing',metrics={}))['proceed']
    assert not classify_comparison(dict(field_name='science'))['proceed']

@pytest.mark.parametrize('field,value',[('implementation','R'),('implementation','IR'),('intervals',[0]),('candidate_event_count',0),('ledger_closed',False),('source_hash',None)])
def test_s4_admission_child_contract(field,value):
    row=dict(status='COMPLETE',mode='OFF',implementation='I',intervals=list(range(8)),candidate_event_count=8,ledger_closed=True,source_hash='x',pre_hash='y')
    row[field]=value
    with pytest.raises(ValueError):s.validate_child_receipt(row,'OFF')

def test_s4_admission_expected_artifact_substitution(tmp_path):
    from tools.hpc_ops import provenance_v2 as pv
    path=tmp_path/'substitute.json';path.write_text('{}')
    with pytest.raises(a.AdmissionBlocked):a._parse_expected_inputs(path.read_bytes(), expected_sha256=a.EXPECTED_INPUTS_RAW_SHA256)

def test_s4_admission_metadata_does_not_hash_and_full_hash_fails(tmp_path):
    file=tmp_path/'x';file.write_bytes(b'known')
    expected={'x':{'raw_sha256':'0'*64}};manifest={'x':{'raw_sha256':'0'*64,'hash_scope':'raw_bytes'}}
    def forbidden(p):raise AssertionError('unexpected bulk read')
    rows=a._verify_input_records(expected,manifest,tmp_path,verify_raw_hashes=False,raw_sha256_file=forbidden)
    assert rows[0]['actual_raw_sha256'] is None
    with pytest.raises(a.AdmissionBlocked):a._verify_input_records(expected,manifest,tmp_path,verify_raw_hashes=True,raw_sha256_file=lambda p:'1'*64)
    with pytest.raises(a.AdmissionBlocked):a._verify_input_records(expected,{},tmp_path,verify_raw_hashes=False,raw_sha256_file=forbidden)
    file.unlink()
    with pytest.raises(FileNotFoundError):a._verify_input_records(expected,manifest,tmp_path,verify_raw_hashes=False,raw_sha256_file=forbidden)

def test_s4_admission_tiny_independent_exact(tmp_path):
    root=tmp_path/'independent'
    p=cli('--supervise-pair','--tiny-science','--output-root',root,timeout=150)
    assert p.returncode==0,p.stdout+p.stderr+(root/'PAIR_RECEIPT.json').read_text()
    row=json.loads((root/'PAIR_RECEIPT.json').read_text());gate=json.loads((root/'comparison_gate.json').read_text())
    assert row['status']=='LOCAL_TINY_INDEPENDENT_EXACT_PASS'
    assert row['parent_science_modules_before_launch']==[]
    assert gate['status']=='EXACT' and gate['raw_rows']==235
    child=[json.loads((root/('I_'+m)/'work/child_receipt.json').read_text()) for m in ('OFF','PROFILE')]
    assert child[0]['pid']!=child[1]['pid']
    assert child[0]['source_hash']==child[1]['source_hash']
    assert all(c['local_kernel_threads']<=12 for c in child)
    # Actual saved scientific-output perturbation and missing scalar sidecar.
    altered=tmp_path/'altered';altered.mkdir()
    for m in ('I_OFF','I_PROFILE'):shutil.copytree(root/m,altered/m)
    arr=altered/'I_PROFILE/work/propagation/final_E.npy'
    data=np.load(arr);data.flat[0]+=1.;np.save(arr,data)
    assert s.compare_pair(altered)['status']=='NUMERIC_DIAGNOSTIC_ONLY'
    missing=tmp_path/'missing';missing.mkdir()
    for m in ('I_OFF','I_PROFILE'):shutil.copytree(root/m,missing/m)
    side=missing/'I_PROFILE/work/scientific_scalar_diagnostics.json'
    payload=json.loads(side.read_text());payload['fields'].pop(next(iter(payload['fields'])));side.write_text(json.dumps(payload))
    assert s.compare_pair(missing)['status']=='NUMERIC_DIAGNOSTIC_ONLY'


def test_s4_admission_start_failure_and_sigint_handler(tmp_path,monkeypatch):
    row=s.supervise_child([str(tmp_path/'missing-executable')],tmp_path/'missing',env=os.environ.copy(),sample_seconds=.05)
    assert row['status']=='BLOCKED_CHILD' and 'FileNotFoundError' in row['stop_reason']
    def interrupt(pid):
        signal.getsignal(signal.SIGINT)(signal.SIGINT,None)
    monkeypatch.setattr(s,'resource_snapshot',interrupt)
    row=s.supervise_child([sys.executable,'-s','-B','-c','import time;time.sleep(60)'],tmp_path/'sigint',env=os.environ.copy(),sample_seconds=.05)
    assert row['status']=='BLOCKED_CHILD' and 'TERMINATION_SIGNAL' in row['stop_reason']
    assert not psutil.pid_exists(row['child_pid'])

def test_s4_admission_strict_json_rejects_nan(tmp_path):
    with pytest.raises(ValueError):s.write(tmp_path/'bad.json',{'value':float('nan')})

def test_s4_admission_code_hash_tamper(tmp_path):
    from tools.hpc_ops import provenance_v2 as pv
    code=tmp_path/'module.py';code.write_text('pass\n')
    record={'path':str(code),'classification':'external','hash_scope':'raw_bytes','raw_sha256':pv.raw_sha256_file(code)}
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({**dict.fromkeys(pv.STRICT_TOP_LEVEL_KEYS), 'schema':pv.SCHEMA,'version':pv.VERSION,'hash_scope':'classified_by_record','records':[record]}))
    assert a._validate_adapter_raw_manifest(tmp_path,manifest,pv)
    code.write_text('raise RuntimeError\n')
    with pytest.raises(a.AdmissionBlocked):a._validate_adapter_raw_manifest(tmp_path,manifest,pv)


@pytest.mark.parametrize('rows',[
    [{'event':'first_interval_started','monotonic':float('nan')}],
    [{'event':'first_interval_started','monotonic':10},{'event':'complete_interval','boundary_monotonic':9,'interval_index':0}],
    [{'event':'complete_interval','boundary_monotonic':float('inf'),'interval_index':0}],
])
def test_s4_admission_invalid_progress(tmp_path,rows):
    path=tmp_path/'progress.jsonl';path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with pytest.raises(ValueError):s.progress_state(path)


def test_s4_admission_missing_start_deadline(tmp_path):
    row=s.supervise_child([sys.executable,'-s','-B','-c','import time;time.sleep(60)'],tmp_path/'missing',env=os.environ.copy(),first_limit=.15,window_limit=5,sample_seconds=.05)
    assert row['stop_reason']=='FIRST_INTERVAL_START_NOT_OBSERVED'
    assert not psutil.pid_exists(row['child_pid'])


def test_s4_admission_unlisted_snapshot_file(tmp_path):
    from tools.hpc_ops import provenance_v2 as pv
    code=tmp_path/'module.py';code.write_text('pass\n')
    record={'path':str(code),'classification':'external','hash_scope':'raw_bytes','raw_sha256':pv.raw_sha256_file(code)}
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({**dict.fromkeys(pv.STRICT_TOP_LEVEL_KEYS), 'schema':pv.SCHEMA,'version':pv.VERSION,'hash_scope':'classified_by_record','records':[record]}))
    (tmp_path/'hidden.py').write_text('pass\n')
    with pytest.raises(a.AdmissionBlocked):a._validate_adapter_raw_manifest(tmp_path,manifest,pv)


def test_s4_admission_execution_root_binding(tmp_path):
    with pytest.raises(RuntimeError,match='EXECUTING_ADAPTER_ROOT_MISMATCH'):
        s.bind_execution_snapshot({'adapter_root':str(tmp_path)})


def test_s4_admission_bad_progress_keeps_receipt(tmp_path,monkeypatch):
    def bad(path):raise ValueError('bad telemetry')
    monkeypatch.setattr(s,'progress_state',bad)
    case=tmp_path/'bad'
    row=s.supervise_child([sys.executable,'-s','-B','-c','import time;time.sleep(60)'],case,env=os.environ.copy(),sample_seconds=.05)
    assert row['status']=='BLOCKED_CHILD' and (case/'supervisor_receipt.json').exists()
