"""Fixed S46A allocation entry; no submission, no automatic scientific fallback."""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import pwd
import signal
import subprocess
import sys
import time
from .inputs import require, seal, write
from .records import CASE_ORDER, load_json_strict, timing_gate, budget_check, cpu_completed_metrics
from tools.hpc_ops import provenance_v2 as pv


def command(argv):
    value=subprocess.run(argv,capture_output=True,text=True,timeout=40)
    return dict(argv=argv,exit_code=value.returncode,stdout=value.stdout,stderr=value.stderr)


def proc_tree(pid):
    """Linux procfs guardian for the fixed GPU environment without psutil."""
    root=Path('/proc')/str(pid)
    try:
        rss=int((root/'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        children=(root/'task'/str(pid)/'children').read_text().split()
        return rss+sum(proc_tree(int(child)) for child in children)
    except (FileNotFoundError,ProcessLookupError):
        return 0


def proc_identity(pid):
    return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]


def stop_gpu_child(process,identity):
    if process.poll() is not None:return
    require(proc_identity(process.pid)==identity,'OWN_CHILD_IDENTITY_BEFORE_TERMINATION')
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        require(proc_identity(process.pid)==identity,'OWN_CHILD_IDENTITY_BEFORE_KILL')
        os.killpg(process.pid,signal.SIGKILL)


def proc_thread_sample(pid):
    records=[];ticks=os.sysconf('SC_CLK_TCK')
    for path in (Path('/proc')/str(pid)/'task').glob('*/stat'):
        try:
            fields=path.read_text().rsplit(')',1)[1].split()
            records.append(dict(id=int(path.parent.name),user_s=int(fields[11])/ticks,system_s=int(fields[12])/ticks))
        except FileNotFoundError:pass
    return dict(wall_ns=time.perf_counter_ns(),threads=records,sampling_api='Linux procfs')


def node_gate(report,backend,size,plan):
    from tools.cpu_o1.cpu_s4.memory_limits import snapshot_cgroup_memory
    repo=Path(__file__).resolve().parents[3]
    if backend=='CPU':
        from tools.cpu_s4_checkpoint_support.git_readonly_compat import verify_runtime_query_tool
        git_identity=verify_runtime_query_tool(plan)
    else:git_identity=command(['git','--version'])
    source=pv.validate_manifest(repo,report/'EXECUTION_SOURCE_MANIFEST.json',require_clean=True,require_hash_scope=True)
    require(source['head']==plan['execution_commit'],'CLEAN_EXPECTED_EXECUTION_COMMIT')
    require(pv.raw_sha256_file(report/'MEASUREMENT_PLAN.json')==os.environ['S46A_PLAN_SHA256'],'PLAN_RAW_IDENTITY')
    admission=load_json_strict(report/'BUDGET_ADMISSION.json')
    require(pv.raw_sha256_file(report/'BUDGET_ADMISSION.json')==os.environ['S46A_BUDGET_SHA256'],'PRE_SUBMISSION_BUDGET_IDENTITY')
    budget=budget_check(admission['ledger'],backend,new_request=admission['request'])
    require(budget['status']=='PASS_BUDGET' and int(admission['request']['walltime_s'])==int(os.environ['S46A_WALLTIME_S']),'WORST_RESERVED_BUDGET')
    account='t0s000727' if backend=='CPU' else 'scvi806'
    require(pwd.getpwuid(os.geteuid()).pw_name==account,'ACTUAL_ALLOCATION_ACCOUNT')
    job=os.environ.get('SLURM_JOB_ID','');require(job.isdigit(),'REAL_ALLOCATION_ONLY')
    cpus=96 if backend=='CPU' else 8
    require(int(os.environ['SLURM_CPUS_PER_TASK'])==cpus and int(os.environ['SLURM_JOB_NUM_NODES'])==1,'ALLOCATION_TOPOLOGY')
    description=command(['scontrol','show','job','-o',job]);require(description['exit_code']==0,'SCONTROL_JOB')
    require(('NumCPUs='+str(cpus)) in description['stdout'],'SCHEDULER_ALLOCATED_CPUS')
    allocation_fields=dict(item.split('=',1) for item in description['stdout'].split() if '=' in item)
    hosts=command(['scontrol','show','hostnames',allocation_fields['NodeList']])
    require(hosts['exit_code']==0 and len(hosts['stdout'].splitlines())==1 and hosts['stdout'].strip()!='g0603','EXCLUDED_G0603_ACTUAL_NODE')
    require(int(allocation_fields['NumCPUs'])==cpus,'ACTUAL_SCHEDULER_CPU_COUNT')
    limit=allocation_fields['TimeLimit'].split('-');days=int(limit[0]) if len(limit)==2 else 0
    hours,minutes,seconds=map(int,limit[-1].split(':'))
    require(days*86400+hours*3600+minutes*60+seconds==int(os.environ['S46A_WALLTIME_S']),'ACTUAL_SCHEDULER_WALL_LIMIT')
    affinity=sorted(os.sched_getaffinity(0))
    topology=command(['lscpu','-p=CPU,CORE,SOCKET,NODE']);hardware=command(['lscpu'])
    require(topology['exit_code']==hardware['exit_code']==0,'NODE_CPU_TOPOLOGY')
    rows=[row.split(',') for row in topology['stdout'].splitlines() if row and not row.startswith('#')]
    physical={(r[2],r[1]) for r in rows if int(r[0]) in affinity}
    numas=sorted({r[3] for r in rows if int(r[0]) in affinity})
    if backend=='CPU':
        require(len(affinity)==len(physical)==96,'NINETY_SIX_PHYSICAL_CPU_CORES')
        require(not os.environ.get('SLURM_JOB_GPUS') and 'gres/gpu' not in description['stdout'],'CPU_ZERO_GPU')
        expected=dict(numpy='2.2.5',scipy='1.15.2',numba='0.61.2',llvmlite='0.44.0',psutil='7.0.0',threadpoolctl='3.6.0')
        versions={name:importlib.import_module(name).__version__ for name in expected}
        require(versions==expected and sys.version_info[:3]==(3,12,8),'ACCEPTED_CPU_ENVIRONMENT')
    else:
        import cupy as cp
        require(cp.cuda.runtime.getDeviceCount()==1,'SINGLE_GPU_ONLY')
        versions={name:importlib.import_module(name).__version__ for name in ('numpy','scipy','cupy')}
    allocation_gib=32 if backend=='CPU' and size=='small' else 320 if backend=='CPU' else 120
    memory=snapshot_cgroup_memory(os.getpid(),allocation_bytes=allocation_gib*1024**3)
    require(memory['status']=='READ' and memory['effective_limit_bytes']>0,'ACTUAL_CGROUP_MEMORY')
    stop=int(.8*memory['effective_limit_bytes'])
    leaf=Path(memory['levels_leaf_to_mount_root'][0]['path'])
    peak=leaf/('memory.max_usage_in_bytes' if memory['version']==1 else 'memory.peak')
    if backend=='GPU':
        write(report/'NODE_MEMORY_GATE.json',dict(memory=memory,expected_peak_path=str(peak),
            kernel_peak_available=peak.is_file(),guardian_fraction=.8,
            fallback='Existing process-tree RSS plus current values of every enforced cgroup limit; kernel peak unavailable is reported, not estimated'))
    require(backend=='GPU' or peak.is_file(),'CGROUP_PEAK_AVAILABLE')
    result=dict(status='PASS',job_id=job,backend=backend,size=size,source=source,allocation=description,hardware=hardware,
                topology=topology,affinity=affinity,physical_cores=len(physical),numa_nodes=numas,memory=memory,
                guardian_bytes=stop,memory_peak_path=str(peak) if peak.is_file() else None,
                memory_guardian_mode='KERNEL_PEAK_AND_PROCESS_TREE_RSS' if peak.is_file() else 'PROCESS_TREE_RSS_AND_CURRENT_CGROUP_LIMITS',
                versions=versions,python=sys.version,executable=sys.executable,
                load_average=os.getloadavg(),frequency=command(['lscpu','-e=CPU,CORE,SOCKET,NODE,MHZ']),
                NUMA_pages=Path('/proc/self/numa_maps').read_text(encoding='utf-8'),
                thread_environment={key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMBA_CPU_NAME','NUMBA_THREADING_LAYER')})
    result.update(budget=budget,git_identity=git_identity)
    if backend=='GPU':
        result.update(gpu=command(['nvidia-smi','-q','-i',cp.cuda.Device(0).pci_bus_id]),gpu_free_total_bytes=list(cp.cuda.runtime.memGetInfo()),
                      cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cuda_driver=cp.cuda.runtime.driverGetVersion(),
                      gpu_properties={key:str(value) for key,value in cp.cuda.runtime.getDeviceProperties(0).items()},
                      nsys=command(['bash','-c','command -v nsys || true']),gpu_worker_threads=cpus)
    write(report/'NODE_GATE.json',result)
    return result


def launch(report,backend,shape,case,mode,qualify,reference,node,deadline):
    import resource
    if backend=='CPU':
        import psutil
        from tools.cpu_o1.cpu_s4.supervisor import terminate
        from tools.cpu_s4_fullsize_confirmation.controller import _tree_rss
    folder=report/'runs'/backend/('x'.join(map(str,shape)))/case
    folder.mkdir(parents=True,exist_ok=False)
    plan=load_json_strict(report/'MEASUREMENT_PLAN.json')
    canonical=Path(plan.get('canonical_root',str(report/'canonical')))
    argv=[sys.executable,'-s','-B','-m','tools.cpu_s4_6a.entry','--report',str(report),'--backend',backend,
          '--bundle',str(canonical/('x'.join(map(str,shape)))),'--work',str(folder/'work'),'--case',case,'--mode',mode]
    if qualify:argv+=['--qualify']
    if reference:
        argv+=['--reference-output',reference['output']]
        if qualify:argv+=['--reference-boundary',reference['boundary_path']]
        if reference.get('stages'):argv+=['--reference-stages',reference['stages']]
    env=os.environ.copy()
    env.update(NUMBA_CACHE_DIR=str(report/'cache'/backend/mode),CUPY_CACHE_DIR=str(report/'cache'/backend/mode/'cupy'),
               S46A_INPUT_BINDINGS_SHA256=pv.raw_sha256_file(report/'INPUT_BINDINGS.json'))
    env.pop('PYTEST_ADDOPTS',None)
    write(folder/'LAUNCH.json',dict(argv=argv,mode=mode,case=case,qualification=qualify,
        fresh_process=True,warmup_intervals=0,numba_cache=env['NUMBA_CACHE_DIR'],cupy_cache=env['CUPY_CACHE_DIR'],
        factor_cache='NEW_EMPTY_FACTORY',source_manifest_sha256=pv.raw_sha256_file(report/'EXECUTION_SOURCE_MANIFEST.json')))
    children_before=resource.getrusage(resource.RUSAGE_CHILDREN)
    started=time.perf_counter_ns()
    with (folder/'stdout.log').open('xb') as so,(folder/'stderr.log').open('xb') as se:
        process=subprocess.Popen(argv,env=env,stdout=so,stderr=se,start_new_session=True)
    created=psutil.Process(process.pid).create_time() if backend=='CPU' else proc_identity(process.pid)
    rss_peak=0;reason=None;samples=0;thread_samples=[]
    try:
        while process.poll() is None:
            rss=_tree_rss(process) if backend=='CPU' else proc_tree(process.pid)
            rss_peak=max(rss_peak,rss);samples+=1
            if node['memory_peak_path'] is not None:
                cgroup_peak=int(Path(node['memory_peak_path']).read_text(encoding='ascii').strip())
                exceeded=max(rss,cgroup_peak)>node['guardian_bytes']
            else:
                from tools.cpu_o1.cpu_s4.memory_limits import snapshot_cgroup_memory,memory_fractions
                current=snapshot_cgroup_memory(os.getpid(),allocation_bytes=node['memory']['allocation_limit_bytes'])
                fractions=memory_fractions(rss_bytes=max(rss,0),pss_bytes=None,cgroup=current,
                    allocation_bytes=node['memory']['allocation_limit_bytes'])
                require(fractions['status']=='READ','GPU_CURRENT_MEMORY_GUARDIAN_REQUIRED')
                exceeded=fractions['max_enforced_fraction']>.8
            if rss<0 or exceeded:reason='HOST_GUARDIAN_80_PERCENT';break
            if time.monotonic()>deadline:reason='ALLOCATION_DEADLINE';break
            if mode=='LIGHT' and samples%5==1:
                if backend=='GPU':thread_samples.append(proc_thread_sample(process.pid))
                else:
                    try:
                        member=psutil.Process(process.pid)
                        thread_samples.append(dict(wall_ns=time.perf_counter_ns(),threads=[dict(id=t.id,user_s=t.user_time,system_s=t.system_time) for t in member.threads()],io=member.io_counters()._asdict()))
                    except psutil.Error:pass
            time.sleep(1)
    finally:
        if process.poll() is None:
            if backend=='CPU':terminate(process.pid,created)
            else:stop_gpu_child(process,created)
    exit_code=process.wait(timeout=15);finished=time.perf_counter_ns()
    children_after=resource.getrusage(resource.RUSAGE_CHILDREN)
    supervisor=dict(exit_code=exit_code,stop_reason=reason,start_ns=started,end_ns=finished,
                    process_service_s=(finished-started)/1e9,peak_tree_rss_bytes=rss_peak,sample_s=1,samples=samples)
    supervisor.update(memory_guardian_mode=node['memory_guardian_mode'],
                      kernel_cgroup_peak_available=node['memory_peak_path'] is not None)
    supervisor.update(waited_children_cpu_s=(children_after.ru_utime+children_after.ru_stime)-(children_before.ru_utime+children_before.ru_stime),
                      child_cpu_scope='RUSAGE_CHILDREN delta, entry process and descendants it waited for; full service, not application-only')
    write(folder/'SUPERVISOR_RECEIPT.json',supervisor)
    if thread_samples:write(folder/'THREAD_AND_IO_SAMPLES.json',dict(cadence_s=5,sampling_scope='LIGHT whole child service; sampled thread CPU time is not an exact per-core timeline',records=thread_samples))
    require(exit_code==0 and reason is None,'CASE_FAILED:'+case)
    result=load_json_strict(folder/'work/CASE_RECEIPT.json')
    result['supervisor']=supervisor;result['case']=case
    if backend=='CPU':
        route=result['route']
        require(route['nH']==16 and route['fresh_responses']==32 and route['math_dispatches']==128 and route['fallback']==0,'CPU_ORIGINAL_WORKLOAD')
        if mode!='REFERENCE':
            result.update(cpu_completed_metrics(load_json_strict(folder/'work/RAW_CPU_TIMING.json'), 8))
    write(folder/'MATRIX_CASE_SUMMARY.json',result)
    print(json.dumps({key:result.get(key) for key in ('backend','case','application_service_s','window_completed_s','later_body_s')}),flush=True)
    return result


def matrix(report,backend,shape,node,deadline):
    reference=launch(report,backend,shape,'REFERENCE','REFERENCE',True,None,node,deadline)
    complete=[]
    for case in CASE_ORDER:
        qualify=case.startswith('QUAL_'); mode=case.split('_')[-1]
        result=launch(report,backend,shape,case,mode,qualify,reference,node,deadline)
        complete.append(result)
        write(report/('PROGRESS_'+backend+'_'+str(shape[1])+'.json.'+case),dict(completed=[r['case'] for r in complete]))
    rows=[dict(pair=r['case'].split('_')[0],kind=r['case'].split('_')[1],application_s=r['application_service_s'],
               window_s=r['window_completed_s'],later_s=r['later_body_s']) for r in complete if r['case'].startswith('P')]
    gate=timing_gate(rows)
    write(report/('TIMING_GATE_'+backend+'_'+str(shape[1])+'.json'),gate)
    result=dict(status='TOTAL_TIMES_MEASURED' if gate['status']=='PASS_TIMING_GATE' else 'DETAIL_ATTRIBUTION_LIMITED',
                backend=backend,shape=list(shape),cases=complete,timing_gate=gate,reference=reference)
    write(report/('MATRIX_RESULT_'+backend+'_'+str(shape[1])+'.json'),result)
    if backend=='CPU':
        from tools.cpu_s4_checkpoint_support import support_module
        boundary=support_module('g2_hpc_exact').load_boundary_capture(complete[0]['boundary_path'])
        support_module('g1_exact').save_tree(report/('CROSS_FIRST_CPU_'+str(shape[1])),boundary['fields'],compressed=True)
    else:
        diagnostic=launch(report,backend,shape,'AUX_SERIALIZED','SERIALIZED',False,reference,node,deadline)
        write(report/('SERIALIZED_PHASE_DIAGNOSTIC_'+str(shape[1])+'.json'),dict(
            case=diagnostic,application_interference_ratio=diagnostic['application_service_s']/complete[2]['application_service_s'],
            warning='Separate completed-wall diagnostic. Does not replace OFF or native LIGHT events.'))
    return result


def cross_small(report,gpu_result):
    """Descriptive differences only; no new epsilon and no equivalence PASS."""
    import numpy as np
    from .gpu import read_snapshot
    cpu=load_json_strict(report/'CPU_SMALLGRID_TRANSPARENCY.json')
    require(cpu['cases'][0]['input_identity']==gpu_result['cases'][0]['input_identity'],'CROSS_BACKEND_IDENTICAL_COMPLETE_INPUT')
    left=read_snapshot(report/'CPU_SMALLGRID_OUTPUT')
    right=read_snapshot(gpu_result['cases'][0]['output'])
    first_left=read_snapshot(report/'CPU_SMALLGRID_FIRST_BOUNDARY')
    b=read_snapshot(gpu_result['cases'][0]['boundary_path'])
    first_right=b
    from tools.cpu_s4_checkpoint_support import support_module
    flatten=support_module('g2_hpc_exact').current_terminal_fields
    a,z=flatten(left),flatten(right)
    a.pop('scalar_diagnostics');z.pop('scalar_diagnostics')
    for key in ('z_axis','dz_used_z','adaptive_rejection_count_z','safety_mode_trigger_count_z','raman_operator_substep_count'):
        require(np.array_equal(a['diagnostics.'+key],z['diagnostics.'+key]),'MATCHED_ACCEPTED_SCHEDULE_AND_BRANCHES:'+key)
    fields=[]
    for domain,la,ra in (('terminal',a,z),('first_boundary',first_left,first_right)):
        require(set(la)==set(ra),'CROSS_FIELD_SET:'+domain)
        for key in sorted(la):
            x,y=np.asarray(la[key]),np.asarray(ra[key])
            require(x.shape==y.shape and x.dtype==y.dtype,'CROSS_SHAPE_PRECISION:'+key)
            if x.dtype.kind not in 'biufc':continue
            finite=np.isfinite(x)&np.isfinite(y)
            require(np.array_equal(np.isnan(x),np.isnan(y)) and np.array_equal(np.isinf(x),np.isinf(y)),'CROSS_NONFINITE_PATTERN:'+key)
            xv,yv=x[finite],y[finite]
            if xv.dtype.kind=='b':xv,yv=xv.astype(np.int8),yv.astype(np.int8)
            d=np.asarray(xv-yv)
            norm=float(np.linalg.norm(xv.ravel()))
            absolute=float(np.max(np.abs(d))) if d.size else 0.
            relative=float(np.linalg.norm(d.ravel())/norm) if norm else None
            fields.append(dict(domain=domain,field=key,shape=list(x.shape),dtype=str(x.dtype),max_abs=absolute,
                relative_L2=relative,relative_denominator='L2(CPU finite elements)',zero_denominator=norm==0,
                cpu_nan=int(np.isnan(x).sum()),gpu_nan=int(np.isnan(y).sum()),cpu_inf=int(np.isinf(x).sum()),gpu_inf=int(np.isinf(y).sum())))
    for key in ('scalars.interval_index','scalars.dz_try','scalars.adaptive_rejection_count','scalars.safety_mode_trigger_count'):
        require(np.array_equal(first_left[key],first_right[key]),'SAME_WORK_BRANCH:'+key)
    value=dict(status='CROSS_BACKEND_NUMERICAL_EQUIVALENCE_NOT_QUALIFIED',matched_input_workload=True,
               tolerance_contract=None,fields=fields,scientific_equivalence=False)
    write(report/'CROSS_BACKEND_NUMERIC_DIFFERENCES_SMALL.json',value)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--backend',choices=('CPU','GPU'),required=True);parser.add_argument('--size',choices=('small','full','both'),required=True)
    args=parser.parse_args();report=args.report.resolve(strict=True);plan=load_json_strict(report/'MEASUREMENT_PLAN.json')
    require((args.backend=='CPU' and args.size in ('small','full')) or (args.backend=='GPU' and args.size=='both'),'PREREGISTERED_RESOURCE_ROUTES_ONLY')
    started=time.monotonic();node=node_gate(report,args.backend,args.size,plan)
    deadline=started+int(os.environ['S46A_WALLTIME_S'])-60
    try:
        if args.backend=='CPU' and args.size=='small':
            canonical=Path(plan.get('canonical_root',str(report/'canonical')))
            for shape in ((384,117,129),(384,351,301)):
                target=canonical/('x'.join(map(str,shape)))
                if plan.get('reuse_sealed_canonical'):
                    from .records import validate_source_records
                    frozen=load_json_strict(target/'INPUT_MANIFEST.json')
                    require(frozen['shape']==list(shape) and frozen['generated_once'],'REUSED_FROZEN_CANONICAL_SHAPE')
                    validate_source_records(target,frozen['records'])
                else:
                    seal(target,report/'external_data/input_bundle',shape,full_input_root=plan['full_input_root'])
            manifests={','.join(map(str,shape)):pv.raw_sha256_file(canonical/('x'.join(map(str,shape)))/'INPUT_MANIFEST.json')
                       for shape in ((384,117,129),(384,351,301))}
            write(report/'INPUT_BINDINGS.json',dict(schema='filament.cpu_s46a.input_bindings.v1',hash_scope='raw_bytes',manifest_raw_sha256=manifests,frozen_before_any_performance_case=True))
        elif args.backend=='GPU':
            preposition=load_json_strict(report/'PREPOSITIONED_INPUTS_AND_CPU_REFERENCE.json')
            from .records import validate_source_records
            validate_source_records(report,preposition['records'])
            require((report/'CPU_SMALLGRID_TRANSPARENCY.json').is_file(),'CPU_SMALLGRID_REFERENCE_PREPOSITIONED')
        shape=(384,351,301) if args.size=='full' else (384,117,129)
        result=matrix(report,args.backend,shape,node,deadline)
        if args.backend=='GPU' and args.size=='both':
            cross_small(report,result)
            import cupy as cp
            free,total=cp.cuda.runtime.memGetInfo()
            required=plan['gpu_memory_forecast']['full_required_bytes']
            if free<required:
                write(report/'FULLSIZE_RESOURCE_BLOCKED.json',dict(reason='AVAILABLE_VRAM_BELOW_PREREGISTERED_LIFETIME_FORECAST',free_bytes=free,total_bytes=total,required_bytes=required))
                write(report/'ALLOCATION_PARTIAL.json',dict(status='MATCHED_SMALLGRID_ONLY',reason='FULLSIZE_RESOURCE_BLOCKED',backend=args.backend,job_id=os.environ['SLURM_JOB_ID']))
                return 0
            else:
                matrix(report,'GPU',(384,351,301),node,deadline)
        write(report/'ALLOCATION_COMPLETED.json',dict(status='COMPLETED',backend=args.backend,size=args.size,job_id=os.environ['SLURM_JOB_ID'],elapsed_s=time.monotonic()-started))
    except BaseException as error:
        write(report/'ALLOCATION_FAILED.json',dict(error=repr(error),backend=args.backend,size=args.size,job_id=os.environ['SLURM_JOB_ID'],elapsed_s=time.monotonic()-started,scientific_fallback=False))
        raise
    return 0


if __name__=='__main__':raise SystemExit(main())
