from pathlib import Path
import json
import os
import sys
import subprocess
import numpy as np
import pytest
from tools.cpu_p1r import cpu_p1r_optical as o


def test_optical_array_identity_and_layout():
    a=np.arange(24,dtype=np.float64).reshape(2,3,4)
    digest=o.verify_array_contract(a,shape=a.shape,dtype='float64')
    assert o.verify_array_contract(a.copy(),shape=a.shape,dtype='float64',expected_hash=digest)==digest
    with pytest.raises(o.OpticalError,match='hash'):
        o.verify_array_contract(a+1,shape=a.shape,dtype='float64',expected_hash=digest)
    with pytest.raises(o.OpticalError,match='layout'):
        o.verify_array_contract(np.asfortranarray(a),shape=a.shape,dtype='float64')
    with pytest.raises(o.OpticalError):
        o.verify_array_contract(a.astype('>f8'),shape=a.shape,dtype='float64')


def test_optical_pre_view_order_and_zero():
    fields={k:np.zeros((8,2,2),dtype=np.float64) for k in ('delta_n','vx','vy')}
    v=o.DiagnosticPREView(fields)
    with pytest.raises(ValueError): v.update_interval(0,np.zeros((2,2)))
    for i in range(8):
        v.read_interval(i); v.update_interval(i,np.full((2,2),-1e-9))
    assert v.complete
    with pytest.raises(ValueError): v.read_interval(0)
    assert all(np.count_nonzero(x)==0 for x in fields.values())
    fields['vx'][0,0,0]=-0.0
    with pytest.raises(o.OpticalError,match='positive-zero'): o.DiagnosticPREView(fields)


def test_optical_source_anchor_and_illegal_prefix():
    from KHz_filament.longitudinal import build_longitudinal_schedule
    from KHz_filament.hr4e5_formal_entry import build_prefix_schedule
    root=Path(__file__).resolve().parents[3]
    binding=o._source_binding(root)
    assert binding['interval_start_line']==628 and binding['accepted_boundary_line']==1503
    with pytest.raises(ValueError): build_prefix_schedule(build_longitudinal_schedule(1e-4,8e-4),1)


def test_optical_real_tiny_entry_observation_and_reset(tmp_path):
    root=Path(__file__).resolve().parents[3]
    args=o._parser().parse_args(['--bench-root',str(root),'--input-root',str(tmp_path),
        '--vendor-dir',str(tmp_path),'--output-root',str(tmp_path/'entry'),'--entry-smoke'])
    assert o._entry_smoke(args)==0
    result=json.loads((tmp_path/'entry/child/smoke_receipt.json').read_text())
    assert result['observed_vs_unobserved_bitexact']
    assert result['independent_repeat_bitexact']


def test_optical_fft_eligibility_uses_whole_computation():
    profile={'stages':{'temporal_fft':{'walltime_s':3.0}}}
    receipt={'interval_rows':[{'bk_nee_profile_halfstep_1':profile,
        'linear_walltime_step_seconds':4.,'interval_wall_seconds':20.}]}
    result=o._fft_share(receipt)
    assert result['fft_fraction_of_complete_interval_compute']==0.15
    assert result['status']=='SKIPPED_FFT_SHARE_BELOW_50_PERCENT'


def test_optical_partial_event_and_process_cleanup(tmp_path):
    path=tmp_path/'events.jsonl'
    path.write_text('{"event":"complete_interval","interval_index":0}\n{"event":',encoding='utf-8')
    events,offset=o._read_new_events(path,0)
    assert len(events)==1 and events[0]['interval_index']==0
    assert offset<path.stat().st_size
    with (tmp_path/'child.log').open('wb') as log:
        child=subprocess.Popen([sys.executable,'-s','-B','-c','import time; print("partial",flush=True); time.sleep(60)'],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name!='nt')
        result=o._stop_case(child)
    assert result['terminated'] and child.poll() is not None


def test_optical_supervisor_monitor_fault_retains_receipt(monkeypatch,tmp_path):
    root=Path(__file__).resolve().parents[3]
    def fail(*args): raise RuntimeError('injected monitor fault')
    monkeypatch.setattr(o,'_read_new_events',fail)
    result=o._supervise_case('A',input_root=tmp_path,bench_root=root,vendor_dir=tmp_path,output_root=tmp_path)
    assert result['stop_reason'].startswith('MONITOR_ERROR')
    assert result['stop_evidence']['terminated']
    assert (tmp_path/'A/supervisor_receipt.json').exists()
    assert (tmp_path/'A/child.log').exists()


def test_optical_load_context_uses_device_module(monkeypatch,tmp_path):
    root=Path(__file__).resolve().parents[3]
    def reached(*args): raise RuntimeError('asset validation reached after CPU backend check')
    monkeypatch.setattr(o,'verify_asset_mapping',reached)
    with pytest.raises(RuntimeError,match='asset validation reached'):
        o._load_context(tmp_path,root,tmp_path)