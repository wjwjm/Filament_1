"""Actual Nt384 IIR/FFT/Heun, original4A -> OFF -> PROFILE, continuous2/8."""
import json,os,shutil,subprocess,sys
from pathlib import Path
import pytest
from tools.cpu_s4_3g3_smallgrid import benchmark as sg

@pytest.mark.parametrize('intervals',(2,8))
def test_s45a_actual_entry_original_l1_off_profile_complete_exact(intervals,tmp_path):
    origin=Path(os.environ['S45A_REPORT_ROOT']);report=tmp_path/'report';report.mkdir()
    for name in ('EXPERIMENT_PLAN.json','EXECUTION_BINDING.json','S45A_EXECUTION_BINDING.json','S45A_MEASUREMENT_PLAN.json'):
        shutil.copyfile(origin/name,report/name)
    sg.input_manifest(report,shape=(384,17,19));previous=None;records=[]
    for role,module in (('L1_LINEAR_FACTOR_CACHE','tools.cpu_s4_4a_linear_cache.entry'),('L1_OFF','tools.cpu_s4_5a_l1_residual.entry'),('L1_PROFILE','tools.cpu_s4_5a_l1_residual.entry')):
        work=tmp_path/role
        argv=[sys.executable,'-s','-B','-m',module,'--report',str(report),'--work',str(work),'--role',role,'--mode','QUALIFY_BOUNDARY','--local-engineering','--micro-shape','384,17,19','--intervals',str(intervals),'--input-manifest',str(report/'BENCHMARK_INPUT_MANIFEST.json')]
        env=os.environ.copy();env['NUMBA_CACHE_DIR']=str(tmp_path/'jit'/role)
        if previous:argv+=['--reference-output',previous['output'],'--reference-boundary',previous['boundary_path'],'--reference-stages',previous['stages']]
        result=subprocess.run(argv,env=env,capture_output=True,text=True,encoding='utf-8',timeout=600)
        (tmp_path/(role+'.stdout.log')).write_text(result.stdout,encoding='utf-8');(tmp_path/(role+'.stderr.log')).write_text(result.stderr,encoding='utf-8')
        assert result.returncode==0,result.stderr
        current=json.loads((work/'CASE_RECEIPT.json').read_text(encoding='utf-8'))
        route=current['route'];assert current['shape']==[384,17,19] and current['intervals']==intervals
        assert route['nH']==2*intervals and route['fresh_responses']==4*intervals and route['math_dispatches']==16*intervals
        assert route['fallback']==route['rejected']==0 and route['mask_after']==1
        assert all(value==2*intervals for value in route['lifecycle'].values())
        assert all(row['active_thread_ids']==[0] and row['requested_workers']==1 for row in route['actual_numba_kernel_executions'])
        assert route['hash_roles'].get('E_stage1',0)==route['hash_roles'].get('E_apply',0)==0
        assert all(route['hash_roles'][name]==2*intervals for name in ('Omega_stage1','Omega_apply','Omega_stage2'))
        cache=current['cache'];assert cache['max_bytes']==268435456 and cache['stats']['factor_requests']==384*2*intervals
        disk=json.loads((work/'DISK_BOUNDARY_ROUNDTRIP.json').read_text());assert disk['collector_retired'] and disk['live_array_references']==0
        assert disk['metadata']['status']=='PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY'
        stream=json.loads((work/'formal/PROPAGATION/state/streaming_manifest.json').read_text());assert len(stream['records'])==intervals and all(row['post'] is not None for row in stream['records'])
        if previous:
            for name,status in {'terminal':'PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS','boundary':'PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT','stages':'PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'}.items():assert current['comparisons'][name]['status']==status
            assert current['FFT_transform_counts']==previous['FFT_transform_counts']
        if role in ('L1_OFF','L1_PROFILE'):
            timing=json.loads((work/'SCALAR_TIMING.json').read_text())
            assert timing['events_dropped']==0 and timing['clock_closure']['pass']
            assert timing['enabled']==(role=='L1_PROFILE')
            if role=='L1_PROFILE':
                grouped=timing['summary_by_label']
                for name,count in {'RAMAN_IIR':4*intervals,'RAMAN_COEFFICIENTS':8*intervals,'RAMAN_STAGE':4*intervals,'RAMAN_HALF':2*intervals,'LINEAR':2*intervals,'ION_RECURSION':intervals}.items():assert grouped[name]['call_count']==count,(name,grouped)
                assert {row['context']['stage'] for row in timing['records'] if row['label']=='RAMAN_IIR'}=={'stage1','stage2'}
                assert {row['context']['phase'] for row in timing['records'] if row['label']=='RAMAN_IIR'}=={'pre','post'}
            else:assert timing['records']==[]
        records.append(current);previous=current
    (tmp_path/'ACTUAL_S45A_MICRO_EXACT_RECEIPT.json').write_text(json.dumps(dict(status='PASS_ACTUAL_ORIGINAL4A_OFF_PROFILE_NT384_EXACT',intervals=intervals,shape=[384,17,19],local_workers=1,HPC_W48='NOT_RUN_LOCAL',records=records),indent=2),encoding='utf-8')
