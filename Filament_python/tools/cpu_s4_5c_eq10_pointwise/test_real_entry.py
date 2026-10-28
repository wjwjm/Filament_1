"""Actual original4A -> common-clock L1 -> C qualification and C OFF."""
import json,os,shutil,subprocess,sys
from pathlib import Path
import pytest
from tools.cpu_s4_3g3_smallgrid import benchmark as sg

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def report_copy(tmp_path):
    origin=Path(os.environ['S45C_REPORT_ROOT']);report=tmp_path/'report';report.mkdir()
    for name in ('EXPERIMENT_PLAN.json','EXECUTION_BINDING.json','C_EXECUTION_BINDING.json','S45C_EXECUTION_PLAN.json'):
        shutil.copyfile(origin/name,report/name)
    return report
def child(argv,tmp_path,label):
    env=os.environ.copy();env['NUMBA_NUM_THREADS']='1';env['NUMBA_CACHE_DIR']=str(tmp_path/'jit'/label)
    result=subprocess.run([sys.executable,'-s','-B',*argv],env=env,capture_output=True,text=True,encoding='utf-8',timeout=600)
    (tmp_path/(label+'.stdout.log')).write_text(result.stdout,encoding='utf-8')
    (tmp_path/(label+'.stderr.log')).write_text(result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError(dict(gate='ACTUAL_C_CHILD',label=label,exit=result.returncode,stderr=result.stderr[-6000:]))

@pytest.mark.parametrize('intervals',(2,8))
def test_s45c_actual_original_l1_candidate_qual_off(intervals,tmp_path):
    report=report_copy(tmp_path);sg.input_manifest(report,shape=(384,17,19));previous=None;records=[]
    for label,role,module,mode in (
        ('ORIGINAL4A','L1_LINEAR_FACTOR_CACHE','tools.cpu_s4_4a_linear_cache.entry','QUALIFY_BOUNDARY'),
        ('L1','L1_REF_5C','tools.cpu_s4_5c_eq10_pointwise.entry','QUALIFY_BOUNDARY'),
        ('C_QUAL','C_EQ10_V1','tools.cpu_s4_5c_eq10_pointwise.entry','QUALIFY_BOUNDARY'),
        ('C_OFF','C_EQ10_V1','tools.cpu_s4_5c_eq10_pointwise.entry','OFF')):
        work=tmp_path/label
        argv=['-m',module,'--report',str(report),'--work',str(work),'--role',role,'--mode',mode,
              '--local-engineering','--micro-shape','384,17,19','--intervals',str(intervals),
              '--input-manifest',str(report/'BENCHMARK_INPUT_MANIFEST.json')]
        if previous:
            argv+=['--reference-output',previous['output']]
            if mode=='QUALIFY_BOUNDARY':argv+=['--reference-boundary',previous['boundary_path'],'--reference-stages',previous['stages']]
        child(argv,tmp_path,label);value=read(work/'CASE_RECEIPT.json');route=value['route']
        assert route['nH']==2*intervals and route['fresh_responses']==4*intervals and route['math_dispatches']==16*intervals
        assert route['fallback']==route['rejected']==0 and route['mask_after']==1
        assert route['hash_roles'].get('E_stage1',0)==route['hash_roles'].get('E_apply',0)==0
        assert all(route['hash_roles'][name]==2*intervals for name in ('Omega_stage1','Omega_apply','Omega_stage2'))
        assert all(row['active_thread_ids']==[0] for row in route['actual_numba_kernel_executions'])
        if previous:
            assert value['comparisons']['terminal']['status']=='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
            assert value['FFT_transform_counts']==previous['FFT_transform_counts']
            if mode=='QUALIFY_BOUNDARY':
                assert value['comparisons']['boundary']['status']=='PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT'
                assert value['comparisons']['stages']['status']=='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
        if mode=='QUALIFY_BOUNDARY':
            disk=read(work/'DISK_BOUNDARY_ROUNDTRIP.json');assert disk['collector_retired'] and disk['live_array_references']==0
        if label!='ORIGINAL4A':
            bodies=read(work/'INTERVAL_BODY_CLOCKS.json');assert len(bodies['rows'])==intervals
            assert value['new_kernel_dispatches']==(8*intervals if role=='C_EQ10_V1' else 0)
        records.append(dict(label=label,role=role,mode=mode,path=str(work/'CASE_RECEIPT.json'),comparisons=value['comparisons']))
        previous=value
    sg.write(tmp_path/'ACTUAL_C_REAL_EXACT_RECEIPT.json',dict(status='PASS_ACTUAL_ORIGINAL4A_L1_C_QUAL_C_OFF',
        shape=[384,17,19],intervals=intervals,records=records,local_workers=1,HPC_W48='NOT_RUN_LOCAL'))

def test_s45c_actual_guarded_zero(tmp_path):
    report=report_copy(tmp_path);previous=None;records=[]
    for role in ('ORIGINAL4A','L1_REF_5C','C_EQ10_V1'):
        work=tmp_path/role;argv=['-m','tools.cpu_s4_5c_eq10_pointwise.zero_fixture','--report',str(report),'--work',str(work),'--role',role]
        if previous:argv+=['--reference',previous['output']]
        child(argv,tmp_path,role);previous=read(work/'ZERO_RECEIPT.json');records.append(previous)
    sg.write(tmp_path/'ACTUAL_C_ZERO_EXACT_RECEIPT.json',dict(status='PASS_ORIGINAL_L1_C_ACTUAL_ZERO',records=records))
