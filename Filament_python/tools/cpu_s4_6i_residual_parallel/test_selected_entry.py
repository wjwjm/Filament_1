"""Selected input provenance, failure cleanup and opt-in real W1 entry checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from KHz_filament import deposition
from . import capture, mechanism, selected_factory as factory
from .power_hook import PowerVolumeHook, PowerHookError
from .power_volume import make_power_volume_pool, power_volume_cleanup_p24
from .selected_protocol import CANDIDATE, CONTEXTS, MICRO_SHAPE, plan

def test_power_capture_local_mechanism_and_raw_identity(tmp_path):
    root=tmp_path/'captures'
    store=capture.CaptureStore(root,shape=(24,1,2))
    source=np.linspace(-2,3,48).reshape(24,1,2)
    source[0,0,0]=np.nan
    source[0,0,1]=np.inf
    for interval in range(8):
        for context in CONTEXTS:
            token=store.before_call(context=context,interval_index=interval,source=source,
                                    source_site=capture.SOURCE_SITE)
            store.after_call(token,source=source,output=deposition._power_volume(source))
    receipt=store.finish(binding=dict(execution_commit='engineering'))
    assert len(receipt['records'])==12 and receipt['retained_array_references']==0
    output=tmp_path/'MECHANISM_RESULT.json'
    result=mechanism.run(root,output,shape=source.shape,execution_commit='engineering',local_engineering=True)
    assert result['warmup_calls']==144 and result['formal_calls']==360
    assert len(result['blocks'])==24 and result['all_exact']
    assert result['status']=='LOCAL_ENGINEERING_ONLY' and not result['formal14_allowed']
    target=root/receipt['records'][0]['input_file']['path']
    with target.open('ab') as handle:
        handle.write(b'tampered')
    with pytest.raises(RuntimeError,match='ARTIFACT_IDENTITY'):
        capture.load_capture(root,shape=source.shape)

def test_power_capture_rejects_input_mutation_before_output(tmp_path):
    store=capture.CaptureStore(tmp_path/'capture',shape=(24,1,1))
    source=np.ones((24,1,1))
    token=store.before_call(context='q_ion',interval_index=0,source=source,source_site=capture.SOURCE_SITE)
    source[0]=2
    with pytest.raises(RuntimeError,match='INPUT_CHANGED'):
        store.after_call(token,source=source,output=deposition._power_volume(source))

def test_power_actual_backend_and_kernel_tamper_rejected(monkeypatch):
    from . import power_volume
    pool=make_power_volume_pool(1)
    try:
        hook=PowerVolumeHook(pool)
        real=power_volume._run_round
        monkeypatch.setattr(power_volume,'_run_round',lambda *a,**k:None)
        with pytest.raises(PowerHookError,match='KERNEL_BINDING_CHANGED'):
            hook.verify_binding()
        monkeypatch.setattr(power_volume,'_run_round',real)
        monkeypatch.setattr(np,'maximum',lambda *a,**k:None)
        with pytest.raises(PowerHookError,match='BACKEND'):
            hook.verify_binding()
    finally:
        pool.close()

def test_selected_construct_failure_retires_new_pool(monkeypatch,tmp_path):
    pool=make_power_volume_pool(1)
    monkeypatch.setattr(factory,'make_power_volume_pool',lambda count:pool)
    combined=factory._combined_module()
    def broken(*args,**kwargs):
        raise RuntimeError('construction_probe')
    monkeypatch.setattr(combined,'build_role',broken)
    with pytest.raises(RuntimeError,match='construction_probe'):
        factory.build_role('PURE_OFF_SMALLGRID',capture_root=tmp_path/'unused',role='X',local_engineering=True)
    assert pool.snapshot()['closed'] and not pool.snapshot()['active_thread_ids']
    with pytest.raises(RuntimeError):
        power_volume_cleanup_p24(np.ones((24,1,1)),pool=pool)

def test_selected_pool_identity_failure_closes_both_pools(tmp_path):
    built=factory.build_role('PURE_OFF_SMALLGRID',capture_root=tmp_path/'unused',role='X',local_engineering=True)
    dispatch=built[1]
    original=dispatch.cpu_s4_6i_power_pool
    replacement=make_power_volume_pool(1)
    dispatch.cpu_s4_6i_power_pool=replacement
    with pytest.raises(RuntimeError,match='POOL_IDENTITY_CHANGED'):
        factory.verify_bindings(dispatch)
    with pytest.raises(RuntimeError,match='POOL_FINALIZATION_FAILED'):
        factory.finalize_factory(dispatch)
    assert original.snapshot()['closed'] and replacement.snapshot()['closed']
    assert dispatch._cpu_s4_6i_finalization_failure['all_closed']

def test_selected_worker_failure_retires_all_four_pools(tmp_path,monkeypatch):
    from .power_hook import Q_ION_HELPER,INTERVAL_KW
    built=factory.build_role('PURE_OFF_SMALLGRID',capture_root=tmp_path/'unused',role='X',local_engineering=True)
    dispatch=built[1]
    pool=dispatch.cpu_s4_6i_power_pool
    original_run=pool._workers.run
    completed=[]
    def broken(label,count,operation):
        def fail(lo,hi):
            completed.append((lo,hi))
            raise RuntimeError('actual_worker_probe')
        return original_run(label,count,fail)
    monkeypatch.setattr(pool._workers,'run',broken)
    with pytest.raises(RuntimeError,match='actual_worker_probe'):
        dispatch.cpu_s4_6i_power_hook.runtime_helpers()[Q_ION_HELPER](
            np.ones((384,1,1)),1.0,**{INTERVAL_KW:0})
    assert completed==[(0,384)]
    receipt=factory.finalize_factory(dispatch)
    assert receipt['all_pools_retired_before_output']
    assert all(receipt[key]['pool']['closed'] and not receipt[key]['pool']['active_thread_ids']
               for key in ('intensity','eq10','c','power_volume'))

@pytest.mark.parametrize('failure_mode',['scientific_close_exception','task_pending','scientific_recorded_close'])
def test_mechanism_preserves_primary_failure_and_pending_activity(tmp_path,monkeypatch,failure_mode):
    from .power_volume import PowerVolumePool
    scientific=failure_mode.startswith('scientific')
    root=tmp_path/'captures'
    store=capture.CaptureStore(root,shape=(24,1,1))
    source=np.ones((24,1,1))
    for interval in range(8):
        for context in CONTEXTS:
            token=store.before_call(context=context,interval_index=interval,source=source,
                                    source_site=capture.SOURCE_SITE)
            store.after_call(token,source=source,output=deposition._power_volume(source))
    store.finish(binding=dict(execution_commit='engineering'))
    real=mechanism.power_volume_cleanup_p24
    def failing(*args,**kwargs):
        actual=real(*args,**kwargs)
        if failure_mode=='scientific_recorded_close':
            pool=kwargs['pool']
            workers_close=pool._workers.close
            def fail_after_join():
                workers_close()
                raise RuntimeError('recorded_worker_close')
            monkeypatch.setattr(pool._workers,'close',fail_after_join)
            with pytest.raises(RuntimeError,match='recorded_worker_close'):
                pool.close()
        if scientific:
            return actual+1
        raise RuntimeError('after_completed_rounds')
    monkeypatch.setattr(mechanism,'power_volume_cleanup_p24',failing)
    if failure_mode=='scientific_close_exception':
        close=PowerVolumePool.close
        def close_failure(self):
            close(self)
            raise RuntimeError('injected_close_error')
        monkeypatch.setattr(PowerVolumePool,'close',close_failure)
    expected='SCIENTIFIC_NONEXACT' if scientific else 'after_completed_rounds'
    with pytest.raises(RuntimeError,match=expected) as error:
        mechanism.run(root,tmp_path/'result.json',shape=source.shape,
                      execution_commit='engineering',local_engineering=True)
    assert any('POOL_RETIREMENT' in note for note in error.value.__notes__)
    receipt=json.loads((tmp_path/'MECHANISM_POOLS.json').read_text())
    assert receipt['retirement_failed']
    if scientific:
        assert 'SCIENTIFIC_NONEXACT' in receipt['primary_error']
        partial=json.loads((tmp_path/'MECHANISM_PARTIAL.json').read_text())
        assert len(partial['failure']['activity'])==2
        if failure_mode=='scientific_recorded_close':
            assert receipt['pools']['P1']['close_error']=='RuntimeError'
            assert any('recorded_close_error' in row['error'] for row in receipt['errors'])
    else:
        assert len(receipt['terminal_drain']['P1'])==2

@pytest.mark.skipif(not os.environ.get('S46I_SELECTED_REAL_REPORT'),reason='opt-in selected W1 entry')
def test_selected_real_micro_entry_opt_in():
    report=Path(os.environ['S46I_SELECTED_REAL_REPORT']).resolve(strict=True)
    work=Path(os.environ['S46I_SELECTED_REAL_WORK']).resolve()
    work.mkdir(parents=True,exist_ok=False)
    from tools.cpu_s4_3g3_smallgrid import benchmark as sg
    manifest=report/'BENCHMARK_INPUT_MANIFEST.json'
    if not manifest.exists():
        sg.input_manifest(report,shape=MICRO_SHAPE)
    cases={}
    for name,role,n,capturing in (('R2','R',2,False),('X2','X',2,False),
            ('R8','R',8,False),('X8','X',8,False),('CAPTURE8','R',8,True)):
        target=work/name
        argv=[sys.executable,'-s','-B','-m','tools.cpu_s4_6i_residual_parallel.selected_entry',
            '--report',str(report),'--work',str(target),'--role',role,'--intervals',str(n),
            '--local-engineering','--input-manifest',str(manifest)]
        if capturing:
            argv.append('--capture')
        reference=cases.get('R2' if n==2 else 'R8')
        if reference:
            argv+=['--reference-output',reference['output']]
            if n==2:
                argv+=['--reference-boundary',reference['boundary_path'],'--reference-stages',reference['stages']]
        with (work/(name+'.stdout.log')).open('xb') as out,(work/(name+'.stderr.log')).open('xb') as err:
            finished=subprocess.run(argv,env=os.environ.copy(),stdout=out,stderr=err)
        assert finished.returncode==0,f'{name}: see {work}'
        value=sg.read(target/'CASE_RECEIPT.json')
        assert value['baseline_pointwise_groups']==30*n and value['selected_operation_groups']==(8*n if role=='X' else 0)
        assert value['pointwise_pool']['pool']['closed'] and value['residual_cost_records']==0
        assert value['route']['math_dispatches']==16*n and value['route']['fallback']==0
        if reference:
            assert value['comparisons']['terminal']['status']=='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
            if n==2:
                assert value['comparisons']['stages']['status']=='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
                assert value['comparisons']['boundary']['status']=='PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT'
        cases[name]=value
    argv=[sys.executable,'-s','-B','-m','tools.cpu_s4_6i_residual_parallel.selected_entry',
        '--report',str(report),'--work',str(work/'MECHANISM'),'--role','R','--intervals','8',
        '--mechanism','--capture-root',cases['CAPTURE8']['capture_package'],'--local-engineering']
    with (work/'mechanism.stdout.log').open('xb') as out,(work/'mechanism.stderr.log').open('xb') as err:
        finished=subprocess.run(argv,env=os.environ.copy(),stdout=out,stderr=err)
    assert finished.returncode==0,f'mechanism: see {work}'
    value=sg.read(work/'MECHANISM/MECHANISM_RESULT.json')
    assert value['status']=='LOCAL_ENGINEERING_ONLY' and value['formal_calls']==360 and value['all_exact']
