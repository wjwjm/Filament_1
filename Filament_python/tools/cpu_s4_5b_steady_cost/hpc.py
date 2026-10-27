"""Fixed serial S45B small-grid L1 OFF/PROFILE measurement; no new math kernel."""
from __future__ import annotations
import argparse,datetime as dt,importlib,json,math,os,statistics,subprocess,sys,time,traceback
from pathlib import Path
from . import entry
from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.hpc_ops import provenance_v2 as pv
from tools.cpu_s4_fullsize_confirmation.controller import VERSIONS,BLAS,_tree_rss
SCHEMA='filament.cpu_s4_5b.hpc.v1'
ORDER={'P1':('L1_S45B_OFF','L1_S45B_PROFILE'),'P2':('L1_S45B_PROFILE','L1_S45B_OFF'),'P3':('L1_S45B_OFF','L1_S45B_PROFILE')}
def require(ok,why,context=None):entry.require(ok,why,context)
def command(argv):
    result=subprocess.run(argv,capture_output=True,text=True,timeout=40)
    require(result.returncode==0,'NODE_METADATA_COMMAND',dict(argv=argv,stderr=result.stderr));return result.stdout
def seal_reference_files(report):
    contract=entry.read(report/'REFERENCE_INPUT_RAW_RECORDS.json');records=[]
    root=Path(contract['source_root']).resolve(strict=True)
    for row in contract['records']:
        path=Path(row['path']);require(path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink(),'FROZEN_REFERENCE_PATH')
        require(row['classification']=='external' and row['hash_scope']=='raw_bytes' and path.stat().st_size==row['size_bytes'] and os.access(path,os.R_OK),'FROZEN_REFERENCE_FILE_SIZE_PERMISSION')
        digest=pv.raw_sha256_file(path);require(digest==row['raw_sha256'],'COMPUTE_PRE_PROPAGATION_REFERENCE_RAW_IDENTITY',str(path))
        records.append(dict(path=str(path),hash_scope='raw_bytes',raw_sha256=digest,size_bytes=path.stat().st_size))
    return dict(status='PASS_COMPLETE_RAW_REFERENCE_IDENTITIES_BEFORE_PROPAGATION',records=records,arrays_deserialized=0)

def node_gate(report):
    import psutil
    from tools.cpu_o1.cpu_s4 import memory_limits
    binding=entry.verify_execution(report,local_engineering=False,shape=sg.SHAPE,intervals=8,require_node=False)
    require(sys.version_info[:3]==(3,12,8),'FROZEN_PYTHON_PATCH_VERSION')
    versions={name:importlib.import_module(name).__version__ for name in VERSIONS}
    require(versions==VERSIONS,'FROZEN_SCIENTIFIC_DEPENDENCIES',versions)
    require(all(os.environ.get(name)=='1' for name in BLAS) and os.environ.get('UPPE_USE_GPU')=='0','FROZEN_CPU_BLAS_ENV')
    require(os.environ.get('NUMBA_NUM_THREADS')=='48' and os.environ.get('NUMBA_CPU_NAME')=='generic' and os.environ.get('NUMBA_THREADING_LAYER')=='workqueue','FROZEN_NUMBA_ENV')
    job=os.environ['SLURM_JOB_ID'];description=command(['scontrol','show','job',job,'-o'])
    require('NumCPUs=96' in description and 'NumNodes=1' in description and 'TimeLimit=00:20:00' in description,'EXACT_JOB_RESOURCES',description)
    require('MinMemoryNode=32G' in description and not os.environ.get('SLURM_JOB_GPUS') and 'gres/gpu' not in description,'EXACT_32_GIB_ZERO_GPU',description)
    nodes=command(['scontrol','show','hostnames',os.environ['SLURM_JOB_NODELIST']]).splitlines()
    require(len(nodes)==1 and os.environ.get('SLURM_JOB_NUM_NODES')=='1','EXACTLY_ONE_COMPUTE_NODE')
    hardware=command(['lscpu']);topology=command(['lscpu','-p=CPU,CORE,SOCKET,NODE'])
    affinity=psutil.Process().cpu_affinity();rows=[line.split(',') for line in topology.splitlines() if line and not line.startswith('#')]
    cores={(row[2],row[1]) for row in rows if int(row[0]) in affinity};numas={row[3] for row in rows if int(row[0]) in affinity}
    require(len(affinity)==len(cores)==96 and len(numas)==4,'96_DISTINCT_PHYSICAL_CORES_FOUR_NUMA')
    require('Intel(R) Xeon(R) Platinum 9242' in hardware,'VALIDATED_CPU_MODEL')
    memory=memory_limits.snapshot_cgroup_memory(os.getpid(),allocation_bytes=32*1024**3)
    require(memory.get('status')=='READ' and int(memory['effective_limit_bytes'])>0,'ACTUAL_CGROUP_MEMORY_READ')
    stop=int(.8*min(32*1024**3,int(memory['effective_limit_bytes'])))
    leaf=Path(memory['levels_leaf_to_mount_root'][0]['path']);peak_path=leaf/('memory.max_usage_in_bytes' if memory['version']==1 else 'memory.peak')
    initial_peak=int(peak_path.read_text(encoding='ascii').strip());require(initial_peak<=stop,'INITIAL_CGROUP_PEAK_UNDER_GUARDIAN')
    plan=entry.read_strict(report/'S45B_MEASUREMENT_PLAN.json')
    data=Path(os.environ['CPU_S4_DATA_ROOT']).resolve(strict=True)
    require(pv.raw_sha256_file(data/'SMALLGRID_CONFIG.json')==plan['input_contract']['config_raw_sha256'],'R14_CONFIG_RAW_IDENTITY')
    for row in entry.read_strict(data/'raw/FROZEN_LUT_RETRIEVAL.json'):
        source=data/'inputs'/row['path']
        require(row['classification']=='external' and row['hash_scope']=='raw_bytes' and source.resolve().is_relative_to(data) and source.stat().st_size==row['size_bytes'] and pv.raw_sha256_file(source)==row['raw_sha256'],'FROZEN_LUT_RAW_IDENTITY',row['path'])
    require(int(memory['effective_limit_bytes'])<=32*1024**3 and stop>20*1024**3,'ACTUAL_32_GIB_LIMIT_AND_CONSERVATIVE_FORECAST')
    sealed_refs=seal_reference_files(report)
    node=dict(schema=SCHEMA+'.node.v1',status='PASS_S45B_SMALLGRID_SOURCE_RESOURCE_INPUT_GATES',job_id=job,execution_commit=binding['execution_commit'],environment=dict(python=sys.version,executable=sys.executable,modules=versions),allocation=description,hardware=hardware,cpu_affinity=affinity,physical_cores=len(cores),numa_nodes=sorted(numas),allocated_nodes=nodes,sealed_reference_files=sealed_refs,cgroup=memory,process_tree_stop_bytes=stop,memory_peak_path=str(peak_path),initial_kernel_cgroup_peak_bytes=initial_peak,fullsize_arrays_created=0,Hydro_called=False)
    sg.write(report/'NODE_GATE.json',node);return node


def launch(report,case,role,mode,reference,*,deadline,limit,peak_path):
    import psutil
    from tools.cpu_o1.cpu_s4.supervisor import terminate
    folder=report/'runs/A01'/case;folder.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-s','-B','-m','tools.cpu_s4_5b_steady_cost.entry','--report',str(report),'--work',str(folder/'work'),'--role',role,'--mode',mode,'--input-manifest',str(report/'BENCHMARK_INPUT_MANIFEST.json')]
    if reference is not None:
        argv.extend(('--reference-output',reference['output']))
        if mode=='QUALIFY_BOUNDARY':
            argv.extend(('--reference-boundary',reference['boundary_path']))
            if reference.get('stages'):argv.extend(('--reference-stages',reference['stages']))
    env=os.environ.copy();env.pop('PYTEST_ADDOPTS',None);env['NUMBA_CACHE_DIR']=str(report/'cache'/role)
    sg.write(folder/'LAUNCH.json',dict(argv=argv,role=role,mode=mode,UTC=dt.datetime.now(dt.timezone.utc).isoformat(),warmup_intervals=0,factor_cache_start='EMPTY_PRIVATE_NEW_PROCESS',jit_cache_directory=env['NUMBA_CACHE_DIR'],jit_cache_preexisting=Path(env['NUMBA_CACHE_DIR']).exists()))
    with (folder/'stdout.log').open('xb') as stdout,(folder/'stderr.log').open('xb') as stderr:
        process=subprocess.Popen(argv,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
    created=psutil.Process(process.pid).create_time();started=time.monotonic();peak=0;kernel_peak=0;samples=0;reason=None
    try:
        while process.poll() is None:
            child=_tree_rss(process)
            if child<0:reason='PROCESS_IDENTITY_LOST';break
            rss=child+psutil.Process().memory_info().rss;peak=max(peak,rss);samples+=1
            kernel_peak=max(kernel_peak,int(peak_path.read_text(encoding='ascii').strip()))
            if rss>limit or kernel_peak>limit:reason='PROCESS_TREE_OR_KERNEL_CGROUP_MEMORY_LIMIT';break
            if time.monotonic()>=deadline:reason='ALLOCATION_DEADLINE_RESERVE';break
            time.sleep(1)
    finally:
        if process.poll() is None:terminate(process.pid,created)
    code=process.wait(timeout=10);kernel_peak=max(kernel_peak,int(peak_path.read_text(encoding='ascii').strip()))
    if kernel_peak>limit:reason='KERNEL_CGROUP_HIGH_WATER_LIMIT'
    supervisor=dict(schema=SCHEMA+'.supervisor.v1',exit_code=code,stop_reason=reason,process_service_s=time.monotonic()-started,peak_tree_rss_bytes=peak,kernel_cgroup_peak_bytes=kernel_peak,memory_sample_seconds=1,memory_sample_count=samples)
    sg.write(folder/'SUPERVISOR_RECEIPT.json',supervisor)
    require(code==0 and reason is None,'STOP_CHILD_FAILURE:'+case,supervisor)
    result=entry.read_strict(folder/'work/CASE_RECEIPT.json');result['case_id']=case;result['supervisor']=supervisor
    require(result['route']['nH']==16 and result['route']['fresh_responses']==32 and result['route']['math_dispatches']==128 and result['route']['fallback']==0,'ORIGINAL_G3_ACTUAL_ROUTE')
    require(result['execution_domain']=='S45B_SMALLGRID_L1' and result['shape']==list(sg.SHAPE),'SMALLGRID_CASE_DOMAIN')
    print(json.dumps(dict(case=case,status=result['status'],application_service_s=result['application_service_s'])),flush=True)
    return result


def exact(result,*,boundary=False,stages=False):
    for name in ('terminal',)+(('boundary',) if boundary else ())+(('stages',) if stages else ()):
        expected='PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT' if name=='boundary' else 'PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
        require(result['comparisons'][name]['status']==expected,'ORIGINAL_COMPLETE_EXACT:'+name)
    cache=result['cache'];stats=cache['stats'];route=result['route']
    require(result['L1_executed'] and result['cache_fallback']==0,'ACTUAL_L1_BOTH_ROLES')
    require(stats['factor_requests']==6144 and stats['plane_builds']==384 and stats['hits']==5760 and stats['key_misses']==1 and stats['invalidations']==stats['evictions']==stats['failed_requests']==0,'SAME_ONE_KEY_CACHE_WORKLOAD')
    require(cache['max_bytes']==268435456 and cache['max_tables']==2,'ORIGINAL_SMALLGRID_CAPACITY')
    require(route['lifecycle'] and all(value==16 for value in route['lifecycle'].values()),'ORIGINAL_LIFECYCLE_16_CREATED_VALIDATED_RETIRED')
    require(all(route['hash_roles'].get(key)==16 for key in ('Omega_stage1','Omega_apply','Omega_stage2')) and route['hash_roles'].get('E_stage1',0)==route['hash_roles'].get('E_apply',0)==0,'ORIGINAL_THREE_OMEGA_NO_NEW_E_HASH')
    for row in route['actual_numba_kernel_executions']:
        require(row['requested_workers']==row['active_thread_count']==48 and row['active_thread_ids']==list(range(48)) and (row['thread_mask_before'],row['thread_mask_inside'],row['thread_mask_after'])==(24,48,24),'ACTUAL_128_CALLS_EACH_W48_MASK_RESTORED')

def run(report):
    report=Path(report).resolve(strict=True);started=time.monotonic();completed=[];qualified=False
    try:
        node=node_gate(report);deadline=started+1200-60
        manifest=sg.input_manifest(report);plan=entry.read(report/'S45B_MEASUREMENT_PLAN.json')
        require(manifest['fingerprints']==plan['expected_input_fingerprints'],'FROZEN_DIRECT_SMALLGRID_INPUT_IDENTITY')
        old=entry.read(report/'REFERENCE_INPUT_RAW_RECORDS.json')['reference']
        old=dict(output=old['output'],boundary_path=old['boundary_path'],stages=None)
        one=lambda case,role,mode,ref:launch(report,case,role,mode,ref,deadline=deadline,limit=node['process_tree_stop_bytes'],peak_path=Path(node['memory_peak_path']))
        reference=one('QUAL_L1_S45B_OFF','L1_S45B_OFF','QUALIFY_BOUNDARY',old);completed.append(reference);exact(reference,boundary=True)
        profile=one('QUAL_L1_S45B_PROFILE','L1_S45B_PROFILE','QUALIFY_BOUNDARY',reference);completed.append(profile);exact(profile,boundary=True,stages=True)
        require(profile['profile_enabled'] and not reference['profile_enabled'],'QUALIFICATION_ROLES_DIFFER_ONLY_IN_TIMING')
        sg.write(report/'QUALIFIED_STEP_CONTRACT.json',dict(nH=16,intervals=8,source='Current L1 OFF against accepted S44A then PROFILE original complete exact'))
        qualified=True
        for batch,roles in ORDER.items():
            for role in roles:
                value=one(batch+'_'+role.removeprefix('L1_S45B_'),role,'PURE_OFF_SMALLGRID',reference)
                completed.append(value);exact(value)
        pairs=[]
        for batch in ORDER:
            values={row['role']:row for row in completed if row['case_id'].startswith(batch+'_')}
            off=values['L1_S45B_OFF'];profile=values['L1_S45B_PROFILE']
            app_off=off['application_service_s'];app_on=profile['application_service_s'];prop_off=off['propagation_s'];prop_on=profile['propagation_s'];later_off=off['later_interval_body_s'];later_on=profile['later_interval_body_s']
            require(all(math.isfinite(t) and t>0 for t in (app_off,app_on,prop_off,prop_on,later_off,later_on)),'FINITE_ORIGINAL_APPLICATION_PROPAGATION_CLOCKS')
            pairs.append(dict(batch=batch,OFF_application_s=app_off,PROFILE_application_s=app_on,r_app=app_on/app_off,OFF_propagation_s=prop_off,PROFILE_propagation_s=prop_on,r_prop=prop_on/prop_off,OFF_later_interval_body_s=later_off,PROFILE_later_interval_body_s=later_on,r_later=later_on/later_off))
        app_med=statistics.median(row['r_app'] for row in pairs);prop_med=statistics.median(row['r_prop'] for row in pairs);later_med=statistics.median(row['r_later'] for row in pairs)
        valid=app_med<=1.05 and prop_med<=1.05 and later_med<=1.05 and all(row['r_app']<=1.10 and row['r_prop']<=1.10 and row['r_later']<=1.10 for row in pairs)
        result=dict(schema=SCHEMA+'.result.v1',status='S45B_TARGETED_EXACT_PASS',timing_status='TIMING_VALID' if valid else 'TIMING_VALIDITY_UNRESOLVED',pairs=pairs,r_app_median=app_med,r_prop_median=prop_med,r_later_median=later_med,completed_cases=[row['case_id'] for row in completed],default='I0',new_math_implemented=False,performance_qualified=False,production_qualified=False,review_gate='MANUAL_REVIEW_PENDING',I0_cumulative='NOT_MEASURED')
        sg.write(report/'S45B_RESULT.json',result);return result
    except BaseException as error:
        sg.write(report/'JOB_FAILURE.json',dict(schema=SCHEMA+'.failure.v1',error=repr(error),traceback=traceback.format_exc(),qualification_completed=qualified,completed_cases=[row['case_id'] for row in completed],scientific_nonexact_must_stop=True));raise

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True)
    result=run(parser.parse_args().report);print(json.dumps(result));return 0
if __name__=='__main__':raise SystemExit(main())
