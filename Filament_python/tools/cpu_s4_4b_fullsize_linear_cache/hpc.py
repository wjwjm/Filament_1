"""The fixed two-qualification/six-formal 4B matrix; one bounded allocation."""
from __future__ import annotations
import argparse,datetime as dt,json,math,os,shutil,signal,statistics,subprocess,sys,time,traceback
from pathlib import Path
from . import entry
from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_fullsize_confirmation import controller as frozen_controller
from tools.hpc_ops import provenance_v2 as pv

ORDER={'B1':('G3_REF_FULLSIZE','L1_CACHE_FULLSIZE'),'B2':('L1_CACHE_FULLSIZE','G3_REF_FULLSIZE'),'B3':('G3_REF_FULLSIZE','L1_CACHE_FULLSIZE')}
SCHEMA='filament.cpu_s4_4b.hpc.v1'
def require(ok,why,context=None):entry.require(ok,why,context)

def node_gate(report):
    plan=entry.verify_execution(report,local_engineering=False,shape=entry.FULL_SHAPE,intervals=8,require_node=False)
    require(plan['walltime_s']==5400 and plan['max_new_jobids']==2 and plan['max_allocated_core_hours']==288 and plan['max_inflight']==1,'FROZEN_4B_RESOURCE_BUDGET')
    source=frozen_controller._source_gate(plan)
    sealed=frozen_controller._sealed_inputs(plan)
    runtime=frozen_controller._runtime_gate(plan)
    require(int(runtime['cgroup']['effective_limit_bytes'])<=320*1024**3,'ACTUAL_MEMORY_HARD_LIMIT_AT_MOST_320_GIB')
    runtime['process_tree_stop_bytes']=int(.8*min(320*1024**3,int(runtime['cgroup']['effective_limit_bytes'])))
    require(runtime['process_tree_stop_bytes']>int(plan['capacity_forecast']['required_guardian_headroom_bytes']),'ACTUAL_GUARDIAN_ABOVE_CONSERVATIVE_PEAK_FORECAST')
    from tools.cpu_s4_checkpoint_support import support_module
    metadata=support_module('g2_hpc_exact').inspect_boundary_metadata(plan['reference_boundary'],require_members=True,header_only=True,expected_receipt_path=plan['reference_boundary'],expected_raw_sha256=plan['reference_boundary_raw_sha256'],expected_size_bytes=plan['reference_boundary_size_bytes'])
    disk=shutil.disk_usage(report)
    require(disk.free>=int(plan['capacity_forecast']['required_free_disk_bytes']),'ACTUAL_DISK_FREE_FOR_COMPLETE_MATRIX')
    paths=[Path(plan['full_input_root'])/r['relative_path'] for r in plan['full_input_records']]+[Path(r['path']) for r in plan['reference_records']]
    permissions=[dict(path=str(p),mode=oct(p.stat().st_mode & 0o777),uid=p.stat().st_uid,gid=p.stat().st_gid,readable=os.access(p,os.R_OK),size_bytes=p.stat().st_size) for p in paths]
    require(all(p['readable'] for p in permissions),'ACTUAL_INPUT_AND_REFERENCE_READ_PERMISSIONS')
    node=dict(schema=SCHEMA+'.node.v1',status='PASS_S44B_FULLSIZE_SOURCE_RESOURCE_INPUT_GATES',job_id=os.environ['SLURM_JOB_ID'],source=source,sealed_inputs=sealed,runtime=runtime,reference_boundary_metadata=metadata,disk=dict(total_bytes=disk.total,used_bytes=disk.used,free_bytes=disk.free,quota_receipt=plan['disk_quota_receipt']),input_reference_permissions=permissions,fullsize_scientific_arrays_created=0,Hydro_called=False)
    sg.write(report/'NODE_GATE.json',node)
    sg.write(report/'FULL_INPUT_MANIFEST.json',dict(schema=SCHEMA+'.inputs.v1',shape=list(entry.FULL_SHAPE),dtype='complex128',native_layout='C',fingerprints=dict(source=entry.frozen_full.SOURCE_SHA,PRE=entry.frozen_full.PRE_SHA),input_kind='ORIGINAL_SEALED_G2_FULLSIZE_SOURCE_AND_PRE',source_root=plan['full_input_root'],records=plan['full_input_records']))
    return plan,node

def _pss(process):
    import psutil
    try:
        root=psutil.Process(process.pid);members=[root,*root.children(recursive=True)];values=[getattr(p.memory_full_info(),'pss',None) for p in members]
        return sum(values) if all(value is not None for value in values) else None
    except psutil.Error:return None

def launch(report,case,role,mode,reference,*,deadline,limit,peak_path):
    import psutil
    from tools.cpu_o1.cpu_s4.supervisor import terminate
    folder=report/'runs/A01'/case;folder.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-s','-B','-m','tools.cpu_s4_4b_fullsize_linear_cache.entry','--report',str(report),'--work',str(folder/'work'),'--role',role,'--mode',mode,'--input-manifest',str(report/'FULL_INPUT_MANIFEST.json')]
    if reference is not None:
        argv+=['--reference-output',str(reference['output'])]
        if mode=='QUALIFY_BOUNDARY':
            argv+=['--reference-boundary',str(reference['boundary_path'])]
            if reference.get('stages'):argv+=['--reference-stages',str(reference['stages'])]
    environment=os.environ.copy();environment.pop('PYTEST_ADDOPTS',None);environment['NUMBA_CACHE_DIR']=str(report/'cache'/role)
    sg.write(folder/'LAUNCH.json',dict(argv=argv,role=role,mode=mode,UTC=dt.datetime.now(dt.timezone.utc).isoformat(),warmup_intervals=0,factor_cache_start='EMPTY_PRIVATE_NEW_PROCESS',jit_cache_directory=environment['NUMBA_CACHE_DIR'],jit_cache_preexisting=Path(environment['NUMBA_CACHE_DIR']).exists()))
    with (folder/'stdout.log').open('xb') as so,(folder/'stderr.log').open('xb') as se:process=subprocess.Popen(argv,env=environment,stdout=so,stderr=se,start_new_session=True)
    created=psutil.Process(process.pid).create_time();started=time.monotonic();peak=0;kernel_peak=0;samples=0;pss_peak=0;pss_samples=0;reason=None
    try:
        while process.poll() is None:
            child=frozen_controller._tree_rss(process)
            if child<0:reason='PROCESS_IDENTITY_LOST';break
            rss=child+psutil.Process().memory_info().rss;peak=max(peak,rss);samples+=1
            kernel_peak=max(kernel_peak,int(peak_path.read_text(encoding='ascii').strip()))
            if samples%5==1:
                pss=_pss(process)
                if pss is not None:pss_peak=max(pss_peak,pss);pss_samples+=1
            if rss>limit or kernel_peak>limit:reason='PROCESS_TREE_OR_KERNEL_CGROUP_MEMORY_LIMIT';break
            if time.monotonic()>=deadline:reason='ALLOCATION_DEADLINE_RESERVE';break
            time.sleep(1)
    finally:
        if process.poll() is None:terminate(process.pid,created)
    code=process.wait(timeout=10);kernel_peak=max(kernel_peak,int(peak_path.read_text(encoding='ascii').strip()))
    if kernel_peak>limit:reason='KERNEL_CGROUP_HIGH_WATER_LIMIT'
    supervisor=dict(schema=SCHEMA+'.supervisor.v1',exit_code=code,stop_reason=reason,process_service_s=time.monotonic()-started,peak_tree_rss_bytes=peak,kernel_cgroup_peak_bytes=kernel_peak,memory_sample_s=1,memory_samples=samples,observed_child_tree_pss_peak_bytes=pss_peak if pss_samples else None,pss_samples=pss_samples,pss_sample_cadence_s=5,pss_scope='Child tree only; unavailable samples do not imply zero')
    sg.write(folder/'SUPERVISOR_RECEIPT.json',supervisor)
    require(code==0 and reason is None,'STOP_CHILD_FAILURE:'+case,supervisor)
    result=entry.read(folder/'work/CASE_RECEIPT.json');result.update(case_id=case,supervisor=supervisor)
    require(result['execution_domain']=='S44B_FULLSIZE_LINEAR_CACHE' and result['shape']==list(entry.FULL_SHAPE),'ACTUAL_FULLSIZE_ROLE_DOMAIN')
    route=result['route'];require(route['nH']==16 and route['fresh_responses']==32 and route['math_dispatches']==128 and route['fallback']==route['rejected']==0 and result['cache_fallback']==0,'ORIGINAL_G3_REAL_ROUTE')
    require(all(route['hash_roles'][name]==16 for name in ('Omega_stage1','Omega_apply','Omega_stage2')) and route['hash_roles'].get('E_stage1',0)==route['hash_roles'].get('E_apply',0)==0,'FROZEN_G3_HASH_ROLES')
    require(route['lifecycle'] and all(v==16 for v in route['lifecycle'].values()),'ORIGINAL_CONTROLLED_LIFECYCLE')
    for execution in route['actual_numba_kernel_executions']:
        require(execution['active_thread_ids']==list(range(48)) and execution['requested_workers']==48 and (execution['thread_mask_before'],execution['thread_mask_inside'],execution['thread_mask_after'])==(24,48,24),'ACTUAL_W48_MASK24_48_24')
    print(json.dumps(dict(case=case,status=result['status'],application_service_s=result['application_service_s'])),flush=True);return result

def exact(result,*,boundary=False,stages=False):
    for name in ('terminal',)+(('boundary',) if boundary else ())+ (('stages',) if stages else ()):
        expected='PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT' if name=='boundary' else 'PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
        require(result['comparisons'][name]['status']==expected,'ORIGINAL_FULLSIZE_EXACT:'+name)

def run(report):
    report=Path(report).resolve(strict=True);started=time.monotonic();completed=[];qualified=False
    try:
        plan,node=node_gate(report);deadline=started+5400-90;runtime=node['runtime']
        one=lambda case,role,mode,ref:launch(report,case,role,mode,ref,deadline=deadline,limit=runtime['process_tree_stop_bytes'],peak_path=Path(runtime['memory_peak_path']))
        historical=dict(output=plan['reference_output'],boundary_path=plan['reference_boundary'],stages=None)
        reference=one('QUAL_G3','G3_REF_FULLSIZE','QUALIFY_BOUNDARY',historical);completed.append(reference);exact(reference,boundary=True)
        candidate=one('QUAL_L1','L1_CACHE_FULLSIZE','QUALIFY_BOUNDARY',reference);completed.append(candidate);exact(candidate,boundary=True,stages=True)
        require(candidate['cache']['stats']['factor_requests']==6144,'REAL_FULLSIZE_CACHE_QUALIFICATION')
        require(candidate['FFT_transform_counts']==reference['FFT_transform_counts'],'SAME_ACTUAL_QUALIFICATION_FFT_COUNTS')
        sg.write(report/'QUALIFIED_STEP_CONTRACT.json',dict(nH=16,intervals=8,source='Current same-allocation G3/L1 fullsize original exact qualification'));qualified=True
        for batch,roles in ORDER.items():
            for role in roles:
                data=one(batch+'_'+('G3' if role=='G3_REF_FULLSIZE' else 'L1'),role,'PURE_OFF_FULLSIZE',reference);completed.append(data);exact(data)
                require(data['FFT_transform_counts']==reference['FFT_transform_counts'],'SAME_ACTUAL_FORMAL_FFT_COUNTS')
        pairs=[];gains=[]
        for batch in ORDER:
            values={x['role']:x for x in completed if x['case_id'].startswith(batch+'_')};g3=values['G3_REF_FULLSIZE']['application_service_s'];l1=values['L1_CACHE_FULLSIZE']['application_service_s']
            require(math.isfinite(g3) and math.isfinite(l1) and min(g3,l1)>=.2,'RESOLVED_APPLICATION_CLOCK')
            gain=(g3-l1)/g3;gains.append(gain);pairs.append(dict(batch=batch,G3_application_s=g3,L1_application_s=l1,gain=gain))
        med=statistics.median(gains);positive=sum(x>0 for x in gains);passed=med>=.03 and positive>=2
        result=dict(schema=SCHEMA+'.result.v1',status='S4_4B_FULLSIZE_CACHE_EXACT_AND_INCREMENTAL_GAIN_PASS' if passed else 'S4_4B_FULLSIZE_CACHE_EXACT_PASS_NO_QUALIFYING_GAIN',qualification='FULLSIZE_ENTRY8_EXACT_PASS',pairs=pairs,gains=gains,incremental_median=med,positive_batches=positive,min=min(gains),max=max(gains),MAD=statistics.median(abs(x-med) for x in gains),completed_cases=[x['case_id'] for x in completed],default='I0',I0_cumulative_gain='NOT_MEASURED',production_qualified=False,review_gate='MANUAL_REVIEW_PENDING')
        sg.write(report/'FULLSIZE_RESULT.json',result);return result
    except BaseException as error:
        sg.write(report/'JOB_FAILURE.json',dict(schema=SCHEMA+'.failure.v1',error=repr(error),traceback=traceback.format_exc(),qualification_completed=qualified,completed_cases=[x['case_id'] for x in completed],scientific_nonexact_must_stop=True));raise

def main():
    def interrupted(signum,frame):raise InterruptedError('JOB_TERMINATION_SIGNAL:'+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);result=run(p.parse_args().report);print(json.dumps(result));return 0

if __name__=='__main__':raise SystemExit(main())
