"""Task-specific serial eight-window supervisor; no additional optimizer search."""
from __future__ import annotations
import argparse,datetime as dt,importlib,json,math,os,statistics,subprocess,sys,time,traceback
from pathlib import Path
from . import entry
from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.hpc_ops import provenance_v2 as pv
from tools.cpu_s4_fullsize_confirmation.controller import VERSIONS,BLAS,_tree_rss

SCHEMA='filament.cpu_s4_4a.hpc.v1'
ORDER={'B1':('G3_REF','L1_LINEAR_FACTOR_CACHE'),'B2':('L1_LINEAR_FACTOR_CACHE','G3_REF'),'B3':('G3_REF','L1_LINEAR_FACTOR_CACHE')}

def require(ok,why,context=None):entry.require(ok,why,context)
def command(argv):
    result=subprocess.run(argv,capture_output=True,text=True,timeout=40)
    require(result.returncode==0,'NODE_METADATA_COMMAND',dict(argv=argv,stderr=result.stderr));return result.stdout

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
    plan=entry.read_strict(report/'EXPERIMENT_PLAN.json')
    data=Path(os.environ['CPU_S4_DATA_ROOT']).resolve(strict=True)
    require(pv.raw_sha256_file(data/'SMALLGRID_CONFIG.json')==plan['input_contract']['config_raw_sha256'],'R14_CONFIG_RAW_IDENTITY')
    for row in entry.read_strict(data/'raw/FROZEN_LUT_RETRIEVAL.json'):
        source=data/'inputs'/row['path']
        require(row['classification']=='external' and row['hash_scope']=='raw_bytes' and source.resolve().is_relative_to(data) and source.stat().st_size==row['size_bytes'] and pv.raw_sha256_file(source)==row['raw_sha256'],'FROZEN_LUT_RAW_IDENTITY',row['path'])
    node=dict(schema=SCHEMA+'.node.v1',status='PASS_SMALLGRID_LINEAR_CACHE_NODE_GATE',job_id=job,execution_commit=binding['execution_commit'],environment=dict(python=sys.version,executable=sys.executable,modules=versions),allocation=description,hardware=hardware,cpu_affinity=affinity,physical_cores=len(cores),numa_nodes=sorted(numas),allocated_nodes=nodes,cgroup=memory,process_tree_stop_bytes=stop,memory_peak_path=str(peak_path),initial_kernel_cgroup_peak_bytes=initial_peak,fullsize_arrays_created=0,Hydro_called=False)
    sg.write(report/'NODE_GATE.json',node);return node

def launch(report,case,role,mode,reference,*,deadline,limit,peak_path):
    import psutil
    from tools.cpu_o1.cpu_s4.supervisor import terminate
    folder=report/'runs/A01'/case;folder.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-s','-B','-m','tools.cpu_s4_4a_linear_cache.entry','--report',str(report),'--work',str(folder/'work'),'--role',role,'--mode',mode,'--input-manifest',str(report/'BENCHMARK_INPUT_MANIFEST.json')]
    if reference is not None:
        argv.extend(('--reference-output',reference['output']))
        if mode=='QUALIFY_BOUNDARY':argv.extend(('--reference-boundary',reference['boundary_path'],'--reference-stages',reference['stages']))
    env=os.environ.copy();env.pop('PYTEST_ADDOPTS',None);env['NUMBA_CACHE_DIR']=str(report/'cache'/role)
    sg.write(folder/'LAUNCH.json',dict(argv=argv,role=role,mode=mode,UTC=dt.datetime.now(dt.timezone.utc).isoformat(),warmup_intervals=0,factor_cache_start='EMPTY_PRIVATE_NEW_PROCESS'))
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
    require(result['execution_domain']=='SMALLGRID_LINEAR_FACTOR_REUSE' and result['shape']==list(sg.SHAPE),'SMALLGRID_CASE_DOMAIN')
    print(json.dumps(dict(case=case,status=result['status'],application_service_s=result['application_service_s'])),flush=True)
    return result

def _exact(result,*,boundary=False):
    require(result['comparisons']['terminal']['status']=='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS','ORIGINAL_TERMINAL_EXACT')
    if boundary:
        require(result['comparisons']['boundary']['status']=='PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT' and result['comparisons']['stages']['status']=='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS','ORIGINAL_BOUNDARY_AND_STAGE_EXACT')

def run(report):
    report=Path(report).resolve(strict=True);started=time.monotonic();completed=[];qualified=False
    try:
        node=node_gate(report);deadline=started+1200-60
        manifest=sg.input_manifest(report)
        plan=entry.read_strict(report/'EXPERIMENT_PLAN.json')
        require(manifest['fingerprints']==plan['expected_input_fingerprints'],'ORIGINAL_R14_DIRECT_SMALLGRID_INPUTS')
        launch_one=lambda case,role,mode,ref:launch(report,case,role,mode,ref,deadline=deadline,limit=node['process_tree_stop_bytes'],peak_path=Path(node['memory_peak_path']))
        reference=launch_one('QUAL_G3_REF','G3_REF','QUALIFY_BOUNDARY',None);completed.append(reference)
        require(reference['status']=='G3_REF_SAME_GRID_STRICT_REFERENCE_GENERATED','CURRENT_G3_REFERENCE_GENERATED')
        candidate=launch_one('QUAL_L1','L1_LINEAR_FACTOR_CACHE','QUALIFY_BOUNDARY',reference);completed.append(candidate);_exact(candidate,boundary=True)
        require(candidate['cache']['stats']['invalidations']==0 and candidate['cache']['stats']['factor_requests']==6144,'REAL_CACHE_QUALIFICATION_PATH')
        sg.write(report/'QUALIFIED_STEP_CONTRACT.json',dict(nH=16,intervals=8,source='Current same-allocation G3_REF and L1 original exact qualification'))
        qualified=True
        for batch,roles in ORDER.items():
            for role in roles:
                value=launch_one(batch+'_'+('L1' if role=='L1_LINEAR_FACTOR_CACHE' else 'G3_REF'),role,'PURE_OFF_SMALLGRID',reference)
                completed.append(value);_exact(value)
        pairs=[];gains=[]
        for batch in ORDER:
            values={row['role']:row for row in completed if row['case_id'].startswith(batch+'_')}
            g3=values['G3_REF']['application_service_s'];l1=values['L1_LINEAR_FACTOR_CACHE']['application_service_s']
            require(math.isfinite(g3) and math.isfinite(l1) and min(g3,l1)>=.2,'RESOLVED_APPLICATION_CLOCKS')
            gain=(g3-l1)/g3;gains.append(gain);pairs.append(dict(batch=batch,G3_REF_application_s=g3,L1_application_s=l1,gain=gain))
        median=statistics.median(gains);positive=sum(x>0 for x in gains);passed=median>=.03 and positive>=2
        result=dict(schema=SCHEMA+'.result.v1',status='SMALLGRID_LINEAR_CACHE_EXACT_AND_GAIN_PASS' if passed else 'SMALLGRID_LINEAR_CACHE_EXACT_PASS_NO_QUALIFYING_GAIN',qualification='SMALLGRID_LINEAR_CACHE_EXACT_PASS',pairs=pairs,gains=gains,incremental_median=median,positive_batches=positive,min=min(gains),max=max(gains),MAD=statistics.median(abs(x-median) for x in gains),completed_cases=[x['case_id'] for x in completed],default='I0',fullsize_new_cache='NOT_RUN',I0_cumulative='NOT_MEASURED',production_qualified=False,review_gate='MANUAL_REVIEW_PENDING')
        sg.write(report/'SMALLGRID_RESULT.json',result);return result
    except BaseException as error:
        sg.write(report/'JOB_FAILURE.json',dict(schema=SCHEMA+'.failure.v1',error=repr(error),traceback=traceback.format_exc(),qualification_completed=qualified,completed_cases=[x['case_id'] for x in completed],scientific_nonexact_must_stop=True));raise

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True)
    result=run(parser.parse_args().report);print(json.dumps(result));return 0

if __name__=='__main__':raise SystemExit(main())
