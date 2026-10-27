"""Actual accepted4A -> B OFF -> B PROFILE, original comparators, Nt384 W1."""
import json,os,shutil,subprocess,sys
from pathlib import Path
import pytest
from tools.cpu_s4_3g3_smallgrid import benchmark as sg

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def report_copy(tmp_path):
    origin=Path(os.environ['S45B_REPORT_ROOT']);report=tmp_path/'report';report.mkdir()
    for name in ('EXPERIMENT_PLAN.json','EXECUTION_BINDING.json','BASELINE_AND_EXECUTION_BINDING.json','S45B_MEASUREMENT_PLAN.json'):
        shutil.copyfile(origin/name,report/name)
    return report

def child(argv,tmp_path,role):
    env=os.environ.copy();env['NUMBA_CACHE_DIR']=str(tmp_path/'jit'/role)
    result=subprocess.run([sys.executable,'-s','-B',*argv],env=env,capture_output=True,text=True,encoding='utf-8',timeout=600)
    (tmp_path/(role+'.stdout.log')).write_text(result.stdout,encoding='utf-8');(tmp_path/(role+'.stderr.log')).write_text(result.stderr,encoding='utf-8')
    assert result.returncode==0,result.stderr

@pytest.mark.parametrize('intervals',(2,8))
def test_s45b_actual_original_l1_off_profile_exact(intervals,tmp_path):
    report=report_copy(tmp_path);sg.input_manifest(report,shape=(384,17,19));previous=None;records=[]
    for role,module in (('L1_LINEAR_FACTOR_CACHE','tools.cpu_s4_4a_linear_cache.entry'),('L1_S45B_OFF','tools.cpu_s4_5b_steady_cost.entry'),('L1_S45B_PROFILE','tools.cpu_s4_5b_steady_cost.entry')):
        work=tmp_path/role
        argv=['-m',module,'--report',str(report),'--work',str(work),'--role',role,'--mode','QUALIFY_BOUNDARY','--local-engineering','--micro-shape','384,17,19','--intervals',str(intervals),'--input-manifest',str(report/'BENCHMARK_INPUT_MANIFEST.json')]
        if previous:argv+=['--reference-output',previous['output'],'--reference-boundary',previous['boundary_path'],'--reference-stages',previous['stages']]
        child(argv,tmp_path,role);current=read(work/'CASE_RECEIPT.json');route=current['route']
        assert current['shape']==[384,17,19] and current['intervals']==intervals
        assert route['nH']==2*intervals and route['fresh_responses']==4*intervals and route['math_dispatches']==16*intervals
        assert route['fallback']==route['rejected']==0 and route['mask_after']==1
        assert all(value==2*intervals for value in route['lifecycle'].values())
        assert all(row['active_thread_ids']==[0] and row['requested_workers']==1 for row in route['actual_numba_kernel_executions'])
        assert route['hash_roles'].get('E_stage1',0)==route['hash_roles'].get('E_apply',0)==0
        assert all(route['hash_roles'][name]==2*intervals for name in ('Omega_stage1','Omega_apply','Omega_stage2'))
        cache=current['cache'];assert cache['max_bytes']==268435456 and cache['stats']['factor_requests']==384*2*intervals
        disk=read(work/'DISK_BOUNDARY_ROUNDTRIP.json');assert disk['collector_retired'] and disk['live_array_references']==0
        assert disk['metadata']['status']=='PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY'
        stream=read(work/'formal/PROPAGATION/state/streaming_manifest.json');assert len(stream['records'])==intervals and all(row['post'] is not None for row in stream['records'])
        if previous:
            for name,status in {'terminal':'PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS','boundary':'PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT','stages':'PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'}.items():assert current['comparisons'][name]['status']==status
            assert current['FFT_transform_counts']==previous['FFT_transform_counts']
        if role.startswith('L1_S45B_'):
            timing=read(work/'SCALAR_TIMING.json');clocks=read(work/'INTERVAL_CLOCKS.json');budget=read(work/'RECORDER_PAIR_BUDGET.json')
            assert timing['clock_closure']['pass'] and timing['all_raw_parent_edges_preserved'] and len(timing['roots'])==1
            assert len(clocks['rows'])==intervals and clocks['later_interval_body_s']>0
            assert budget['raw_scope_count']<=8192 and budget['metadata_reserved_bytes']<=67108864
            assert current['profile_enabled']==(role=='L1_S45B_PROFILE')
        records.append(dict(role=role,path=str(work/'CASE_RECEIPT.json'),comparisons=current['comparisons'],timing_records=current.get('timing_records'),case_status=current['status']))
        previous=current
    sg.write(tmp_path/'ACTUAL_S45B_MICRO_EXACT_RECEIPT.json',dict(status='PASS_ACTUAL_ORIGINAL4A_B_OFF_B_PROFILE_NT384_EXACT',intervals=intervals,shape=[384,17,19],local_workers=1,HPC_W48='NOT_RUN_LOCAL',records=records))

def test_s45b_actual_guarded_zero_intensity_transparency(tmp_path):
    report=report_copy(tmp_path);previous=None;records=[]
    for role in ('ORIGINAL4A','L1_S45B_OFF','L1_S45B_PROFILE'):
        work=tmp_path/role
        argv=['-m','tools.cpu_s4_5b_steady_cost.zero_fixture','--report',str(report),'--work',str(work),'--role',role]
        if previous:argv+=['--reference',previous['output']]
        child(argv,tmp_path,role);current=read(work/'ZERO_RECEIPT.json');records.append(dict(role=role,path=str(work/'ZERO_RECEIPT.json'),status=current['status'],zero_markers=current['zero_markers']));previous=current
    sg.write(tmp_path/'ACTUAL_ZERO_TRANSPARENCY.json',dict(status='PASS_ACTUAL_ORIGINAL_OFF_PROFILE_ZERO_NT384',records=records))
