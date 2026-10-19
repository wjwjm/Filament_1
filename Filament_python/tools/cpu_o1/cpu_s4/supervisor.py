"""S4 preparation CLI. Invoke by file path to avoid eager cpu_o1 package imports.

Real execution is deliberately disabled. Fake and deterministic tiny fixtures
exercise the same fresh-process supervision; neither accepts external fields.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

REAL_EXECUTION_ENABLED = False
BLAS = ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
THREADS = dict(NUMBA_CPU_NAME='generic', NUMBA_THREADING_LAYER='workqueue', NUMBA_NUM_THREADS='24', UPPE_USE_GPU='0')
SCRIPT = Path(__file__).resolve()

def write(path, value):
    payload=json.dumps(value,allow_nan=False,indent=2)+'\n'
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f:
        f.write(payload)

def append(path, value):
    with Path(path).open('a', encoding='utf8') as f:
        f.write(json.dumps(value, allow_nan=False)+'\n'); f.flush()

def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def s4_module(filename, name):
    return load_module(SCRIPT.with_name(filename), name)

def science_path():
    root=str(SCRIPT.parents[3])
    if root not in sys.path:sys.path.insert(0,root)

def bind_execution_snapshot(config):
    adapter=Path(config['adapter_root']).resolve()
    expected=adapter/'Filament_python/tools/cpu_o1/cpu_s4/supervisor.py'
    if SCRIPT != expected.resolve():
        raise RuntimeError('EXECUTING_ADAPTER_ROOT_MISMATCH')
    pv=load_module(adapter/'Filament_python/tools/hpc_ops/provenance_v2.py','s4_binding_pv')
    admission=load_module(SCRIPT.with_name('admission.py'),'s4_binding_admission')
    admission._validate_adapter_raw_manifest(adapter,Path(config['adapter_provenance']),pv)
    for name,module in tuple(sys.modules.items()):
        if name=='tools' or name.startswith(('tools.','KHz_filament')):
            file=getattr(module,'__file__',None)
            if file and not Path(file).resolve().is_relative_to(adapter):
                raise RuntimeError('IMPORTED_ADAPTER_ROOT_MISMATCH:'+name)
    return adapter


def identity(pid, created):
    import psutil
    try:
        p=psutil.Process(pid)
        return p.is_running() and p.status()!=psutil.STATUS_ZOMBIE and abs(p.create_time()-created)<0.01
    except psutil.Error:return False

def terminate(pid, created):
    """Bounded kill for a launched process group; Windows job is held by guardian."""
    import psutil
    if not identity(pid,created):return
    root=psutil.Process(pid);children=root.children(recursive=True)
    if os.name!='nt':
        try:os.killpg(pid,signal.SIGTERM)
        except ProcessLookupError:pass
    for p in reversed(children+[root]):
        try:p.terminate()
        except psutil.Error:pass
    _,alive=psutil.wait_procs(children+[root],timeout=2)
    for p in alive:
        try:p.kill()
        except psutil.Error:pass
    psutil.wait_procs(alive,timeout=2)
    if os.name!='nt':
        try:os.killpg(pid,signal.SIGKILL)
        except ProcessLookupError:pass

def windows_job(pid):
    """Guardian owns a kill-on-close job before the child may launch science."""
    import ctypes as c
    from ctypes import wintypes as w
    class Basic(c.Structure):
        _fields_=[('process_time',c.c_int64),('job_time',c.c_int64),('flags',w.DWORD),('minimum',c.c_size_t),('maximum',c.c_size_t),('active',w.DWORD),('affinity',c.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
    class IO(c.Structure):_fields_=[(x,c.c_uint64) for x in ('ro','wo','oo','rb','wb','ob')]
    class Extended(c.Structure):
        _fields_=[('basic',Basic),('io',IO),('process_memory',c.c_size_t),('job_memory',c.c_size_t),('peak_process',c.c_size_t),('peak_job',c.c_size_t)]
    k=c.WinDLL('kernel32',use_last_error=True)
    k.CreateJobObjectW.argtypes=[c.c_void_p,w.LPCWSTR];k.CreateJobObjectW.restype=w.HANDLE
    k.SetInformationJobObject.argtypes=[w.HANDLE,c.c_int,c.c_void_p,w.DWORD];k.SetInformationJobObject.restype=w.BOOL
    k.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];k.OpenProcess.restype=w.HANDLE
    k.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE];k.AssignProcessToJobObject.restype=w.BOOL
    k.CloseHandle.argtypes=[w.HANDLE]
    job=k.CreateJobObjectW(None,None);info=Extended();info.basic.flags=0x2000
    process=None
    try:
        if not job or not k.SetInformationJobObject(job,9,c.byref(info),c.sizeof(info)):raise c.WinError(c.get_last_error())
        process=k.OpenProcess(0x0100|0x0001,False,pid) # SET_QUOTA | TERMINATE
        if not process or not k.AssignProcessToJobObject(job,process):raise c.WinError(c.get_last_error())
    except BaseException:
        if job:k.CloseHandle(job)
        raise
    finally:
        if process:k.CloseHandle(process)
    return lambda:k.CloseHandle(job)

def guardian(args):
    close=None
    try:
        if not identity(args.target_pid,args.target_created):raise RuntimeError('child disappeared before guardian')
        if os.name=='nt':close=windows_job(args.target_pid)
        write(args.ready,dict(status='GUARD_READY',pid=os.getpid(),mechanism='WINDOWS_KILL_ON_JOB_CLOSE' if close else 'POSIX_PROCESS_GROUP_GUARD'))
        while identity(args.target_pid,args.target_created) and identity(args.parent_pid,args.parent_created):time.sleep(.1)
    finally:
        if close:close()
        elif os.name!='nt':
            # Group may still contain descendants after the leader exited.
            try:os.killpg(args.target_pid,signal.SIGTERM)
            except ProcessLookupError:pass
            time.sleep(.2)
            try:os.killpg(args.target_pid,signal.SIGKILL)
            except ProcessLookupError:pass
    return 0

def guarded_child(args):
    import psutil
    root=args.output_root
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True)
    proc=psutil.Process();ready=root/'guardian.json'
    cmd=[sys.executable,'-s','-B',str(SCRIPT),'--guardian','--target-pid',str(proc.pid),'--target-created',str(proc.create_time()),'--parent-pid',str(args.parent_pid),'--parent-created',str(args.parent_created),'--ready',str(ready)]
    with (root/'guardian.log').open('xb') as log:
        guard=subprocess.Popen(cmd,stdout=log,stderr=log,start_new_session=True)
    until=time.monotonic()+8
    while not ready.exists():
        if guard.poll() is not None or time.monotonic()>until:raise RuntimeError('GUARDIAN_SETUP_FAILED')
        time.sleep(.02)
    if not identity(args.parent_pid,args.parent_created):raise RuntimeError('PARENT_LOST_BEFORE_START')
    return root

def validate_child_receipt(row, mode, *, require_first_boundary=False):
    if row.get('status')!='COMPLETE' or row.get('mode')!=mode or row.get('implementation')!='I':raise ValueError('child identity/status mismatch')
    if row.get('intervals')!=list(range(8)):raise ValueError('missing complete eight intervals')
    if row.get('candidate_event_count',0)<=0:raise ValueError('compiled I not called')
    if mode=='PROFILE' and row.get('profile_intervals')!=list(range(8)):raise ValueError('profile coverage incomplete')
    # A child claim of ledger_closed=True is deliberately ignored. The parent
    # closes the ledger only after independent output inventory acceptance.
    if row.get('ledger_closed') is False:raise ValueError('child reported incomplete output ledger')
    if not row.get('source_hash') or not row.get('pre_hash'):raise ValueError('input identity missing')
    if require_first_boundary:
        case_id='I_'+mode;boundary=row.get('first_boundary_gate')
        if (row.get('case_id')!=case_id or not isinstance(boundary,dict)
                or boundary.get('case_id')!=case_id or boundary.get('implementation')!='I'
                or boundary.get('proceed_to_next_identity') is not True
                or not isinstance(boundary.get('gate'),dict)
                or boundary['gate'].get('case_id')!=case_id
                or boundary['gate'].get('proceed_to_next_identity') is not True):
            raise ValueError('first-boundary identity/exact gate missing')
        if not isinstance(row.get('output_seal_sha256'),str) or len(row['output_seal_sha256'])!=64:
            raise ValueError('independent output seal binding missing')

def _require_scoped_real_authorization(args):
    from tools.cpu_o1.cpu_s4.launcher import validate_scoped_authorization
    if args.run_config is None or args.approval is None:
        raise RuntimeError('BLOCKED_REAL_EXECUTION_REQUIRES_TASK_BOUND_AUTHORIZATION')
    run_root = getattr(args, 'authorized_run_root', None) or args.output_root
    return validate_scoped_authorization(run_config_path=args.run_config,
        approval_path=args.approval,output_root=args.output_root,
        authorized_run_root=run_root)

def cache_inventory(path):
    path=Path(path)
    return [dict(path=str(p.relative_to(path)),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(path.rglob('*')) if p.is_file()] if path.exists() else []

def real_context(args, root):
    """Dormant real adapter; this preparation version cannot unlock it.

    A later reviewed unlock must bind a genuine approval and verified staged
    paths. It does not change the frozen loader or science window.
    """
    scoped = _require_scoped_real_authorization(args)
    science_path()
    from tools.hpc_ops import provenance_v2 as pv
    from types import SimpleNamespace
    approval=json.loads(args.approval.read_text())
    config=json.loads(args.run_config.read_text())
    run_id=os.environ.get('HPC_AUTHORIZED_RUN_ID')
    if not run_id or approval.get('run_id')!=run_id or approval.get('authorized') is not True:
        raise RuntimeError('APPROVAL_BINDING_FAILED')
    expected_resource=dict(nodes=1,physical_cpus=96,gpus=0,memory_GiB=320,seconds=7200,max_submissions=1)
    if approval.get('resources')!=expected_resource or approval.get('run_config_sha256')!=pv.raw_sha256_file(args.run_config):
        raise RuntimeError('APPROVAL_CONFIG_OR_RESOURCE_MISMATCH')
    if Path(approval['run_root']).resolve()!=args.output_root.resolve().parents[1]:
        raise RuntimeError('APPROVAL_RUN_ROOT_MISMATCH')
    for key in ('provenance','adapter_provenance','expected_inputs','expected_source_binding'):
        if approval.get(key+'_sha256')!=pv.raw_sha256_file(Path(config[key])):
            raise RuntimeError('APPROVAL_SOURCE_INPUT_BINDING_MISMATCH:'+key)
    admission=load_module(SCRIPT.with_name('admission.py'),'s4_real_admission')
    gate=admission.validate_preflight(bench_root=config['bench_root'],provenance_path=config['provenance'],adapter_root=config['adapter_root'],adapter_provenance_path=config['adapter_provenance'],input_root=config['input_root'],expected_inputs_path=config['expected_inputs'],verify_input_hashes=True)
    write(root/'input_source_admission.json',gate)
    if gate['status']!='FULL_INPUT_IDENTITY_PASS':raise RuntimeError('INPUT_SOURCE_ADMISSION_FAILED')
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK')!='96':
        raise RuntimeError('SLURM_ALLOCATION_REQUIRED')
    import psutil
    if len(psutil.Process().cpu_affinity())!=96:raise RuntimeError('CPUSET_96_REQUIRED')
    from tools.cpu_o1 import q1_t0_real
    write(root/'allocation_environment.json',q1_t0_real._allocation_environment())
    # Allocation lifecycle probe already ran once in the bound runtime gate.
    write(root/'allocation_probe_adapter.json',{'status':'VERIFIED_IN_RUNTIME_GATE','job_id':scoped['job_id']})
    from tools.cpu_o1.cpu_s4 import profile
    fields={key:Path(config[key]) for key in ('bench_root','input_root','vendor_dir','provenance','expected_source_binding','expected_python','storage_root')}
    # Frozen loader insists its nominated output path does not yet exist.
    storage_cap_bytes=int(config['storage_cap_bytes'])
    max_files=int(config.get('max_files',50000))
    ns=SimpleNamespace(**fields,output_root=root/'validated_window_target',
        storage_cap_bytes=storage_cap_bytes,max_files=max_files)
    context,receipt=profile._load_real_context(ns)
    write(root/'loader_admission.json',profile._json_safe(receipt))
    from tools.cpu_o1 import q1_t0_real
    storage_module=s4_module('storage_budget.py','s4_storage_budget')
    storage_root=Path(config['storage_root']).resolve(strict=True)
    inventory=q1_t0_real._storage_inventory(storage_root)
    statvfs=os.statvfs(storage_root)
    runtime_gate=scoped['runtime_gate']
    quota=runtime_gate.get('storage_quota')
    if not isinstance(quota,dict):
        raise RuntimeError('BLOCKED_RUNTIME_GATE_STORAGE_QUOTA_RECEIPT_MISSING')
    budget=storage_module.estimate_storage_budget(
        source_shape=context['source'].shape,input_records=gate['inputs'],
        diagnostic_field_count=len(q1_t0_real._frozen_diag_list_contract(
            Path(__import__("KHz_filament.propagate",
                fromlist=["propagate_one_pulse"]).__file__).resolve(strict=True))),
        current_bytes=inventory['bytes'],cap_bytes=storage_cap_bytes,
        filesystem_available_bytes=int(statvfs.f_bavail*statvfs.f_frsize),
        filesystem_available_inodes=int(statvfs.f_favail),
        quota_available_bytes=int(quota['available_bytes']),
        quota_available_inodes=int(quota['available_inodes']))
    write(root/'storage_budget.json',budget)
    if budget['status']!='PASS':
        raise RuntimeError('BLOCKED_STORAGE_BUDGET:'+json.dumps(budget,allow_nan=False))
    return context

def single(args):
    scoped=None
    if not (args.dry_run or args.tiny_science):scoped=_require_scoped_real_authorization(args)
    if args.implementation!='I':raise ValueError('only I permitted')
    child_clock = time.monotonic if args.dry_run else time.perf_counter
    started,cpu=child_clock(),time.process_time();root=guarded_child(args)
    initial=cache_inventory(os.environ.get('NUMBA_CACHE_DIR',root/'cache'))
    if args.dry_run:
        progress=root/'propagation';progress.mkdir();path=progress/'observer_progress.jsonl'
        append(path,dict(event='first_interval_started',interval_index=0,monotonic=time.monotonic()))
        if args.fixture in ('hang','spawn'):
            if args.fixture=='spawn':
                p=subprocess.Popen([sys.executable,'-s','-B','-c','import time; time.sleep(120)'])
                write(root/'descendant.json',dict(pid=p.pid))
            time.sleep(120)
        if args.fixture=='exit':raise RuntimeError('INJECTED_CHILD_EXIT')
        for i in range(8):append(path,dict(event='complete_interval',interval_index=i,boundary_monotonic=time.monotonic()))
        row=dict(status='COMPLETE',mode=args.single_mode,implementation='I',intervals=list(range(8)),profile_intervals=list(range(8)),candidate_event_count=8,ledger_closed=True,source_hash='FAKE_ONLY',pre_hash='FAKE_ONLY',fixture=True)
        if args.fixture=='coverage':row['profile_intervals']=[0]
        if args.fixture=='ledger':row['ledger_closed']=False
        time.sleep(.15)
    else:
        if not args.tiny_science:
            bind_execution_snapshot(json.loads(args.run_config.read_text()))
        science_path()
        from tools.cpu_p1r2 import cpu_p1r2_j1 as j1
        from tools.cpu_o1.cpu_s4 import profile
        from tools.cpu_o1.cpu_o1 import available_numba_threads
        if args.tiny_science:
            context=j1._tiny_context(operator_mode='full_isaacs_eq27')
            context.update(real_input_used=False,source_hash=j1._canonical_hash(context['source']),assets={'pre0':context['pre0'],'pre0_canonical_sha256':j1._canonical_hash(context['pre0'])},_local_kernel_threads=min(12,available_numba_threads(),os.cpu_count() or 1))
        else:
            context=real_context(args,root)
        if context['schedule'].n_intervals!=8 or context['raman'].operator_mode!='full_isaacs_eq27' or context['ion'].time_mode!='full' or context['ion'].integrator.lower()!='rk4':
            raise RuntimeError('FROZEN_SCIENTIFIC_MODE_REQUIRED')
        load_seconds=child_clock()-started
        write(root/'propagation_start_admission.json',dict(monotonic=child_clock(),mode=args.single_mode,clock='perf_counter'))
        grid=context['grid'];source=context['source'];intervals=int(context['schedule'].n_intervals)
        write(root/'input_output_contract.json',dict(schema='filament.cpu_s4.input_output_contract.v1',
            implementation='I',mode=args.single_mode,interval_count=intervals,
            source_shape=[int(x) for x in source.shape],source_dtype=str(source.dtype),
            post_shape=[intervals,int(grid.Ny),int(grid.Nx)],post_dtype='float64',
            qacc_shape=[int(grid.Ny),int(grid.Nx)],qacc_dtype='float64',
            sink_shape=[intervals,int(grid.Ny),int(grid.Nx)],sink_dtype='float64'))
        reference=None
        if args.first_boundary_reference_root is not None:
            if not args.first_boundary_reference_case_id:
                raise RuntimeError('FIRST_BOUNDARY_REFERENCE_CASE_ID_REQUIRED')
            reference=(args.first_boundary_reference_case_id,args.first_boundary_reference_root)
        result=profile._run_one(context,root,profile=args.single_mode=='PROFILE',
            case_id=args.case_id,first_boundary_reference=reference)
        window_payload={k:v for k,v in result.items() if k not in {'receipt','profile'}}
        window_payload.update(mode=args.single_mode,implementation='I',case_id=args.case_id)
        write(root/'window_receipt.json',profile._json_safe(window_payload))
        from tools.hpc_ops import provenance_v2 as pv
        seal_started=time.perf_counter()
        inventory=s4_module('output_inventory.py','s4_output_inventory')
        names=['window_receipt.json','input_output_contract.json','scientific_Qacc.npy',
            'scientific_scalar_diagnostics.json','candidate_events.jsonl',
            'propagation/child_receipt.json','propagation/final_E.npy',
            'propagation/final_POST.npz','propagation/diagnostics.npz',
            'propagation/observer_progress.jsonl','propagation/linear_halfstep_profiles.json']
        names += ['propagation/'+n for n in inventory.SINK_FILES]
        names.append('propagation/function_fft_profile.json')
        auxiliary_names=['first_boundary_receipt.json',
            'first_boundary/snapshot.json','first_boundary/NUMERIC_GATE.json',
            'first_boundary/E.npy','first_boundary/POST.npz',
            'first_boundary/Qacc.npy','first_boundary/Qacc_raman.npy',
            'first_boundary/diagnostics.npz','first_boundary/scalar_diagnostics.json']
        auxiliary={name:pv.raw_sha256_file(root/name) for name in auxiliary_names}
        write(root/'OUTPUT_SEAL.json',dict(schema='filament.cpu_s4.output_seal.v1',
            files={n:pv.raw_sha256_file(root/n) for n in names},
            auxiliary_files=auxiliary))
        seal_hash=pv.raw_sha256_file(root/'OUTPUT_SEAL.json')
        row=dict(status='COMPLETE',mode=args.single_mode,implementation='I',case_id=args.case_id,
            intervals=[x['interval_index'] for x in result['interval_rows']],
            profile_intervals=list(range(8)) if args.single_mode=='PROFILE' else [],
            candidate_event_count=result['candidate_event_count'],output_seal_sha256=seal_hash,
            output_sealing_seconds=time.perf_counter()-seal_started,
            source_hash=result['input_source_canonical_sha256'],
            pre_hash=result['input_pre_canonical_sha256'],fixture=False,
            first_boundary_gate=result.get('first_boundary_gate'),
            local_kernel_threads=context.get('_local_kernel_threads',24),
            load_identity_seconds=load_seconds,window_call_wall_seconds=result['window_call_wall_seconds'],
            propagation_wall_seconds=result['propagation_wall_seconds'],
            propagation_wall_seconds_raw_including_cross_window_comparison=result['propagation_wall_seconds_raw_including_cross_window_comparison'],
            propagation_cross_window_comparison_seconds=result['propagation_cross_window_comparison_seconds'],
            propagation_process_cpu_seconds=result['propagation_process_cpu_seconds'],
            first_jit_seconds='NOT_INSTRUMENTED')
    row.update(pid=os.getpid(),process_cpu_seconds=time.process_time()-cpu,child_wall_seconds=child_clock()-started,environment={k:os.environ.get(k) for k in (*BLAS,*THREADS)},cache_before=initial,cache_after=cache_inventory(os.environ.get('NUMBA_CACHE_DIR',root/'cache')))
    write(root/'child_receipt.json',row)
    return 0

def resource_snapshot(pid, *, simulation=False):
    import psutil
    try:root=psutil.Process(pid); processes=[root,*root.children(recursive=True)]
    except psutil.NoSuchProcess:return {'available':False}
    rss=threads=0;pss=0;pss_available=True;cpu=0.;pids=[]
    for p in processes:
        try:
            rss+=p.memory_info().rss;threads+=p.num_threads();t=p.cpu_times();cpu+=t.user+t.system;pids.append(p.pid)
            try:pss+=p.memory_full_info().pss
            except (psutil.Error,AttributeError):pss_available=False
        except psutil.NoSuchProcess:continue
    try:affinity=root.cpu_affinity()
    except (psutil.Error,AttributeError):affinity='NOT_MEASURED'
    allocation=320*1024**3
    if sys.platform.startswith('linux') and not simulation:
        limits=s4_module('memory_limits.py','s4_memory_limits')
        cg=limits.snapshot_cgroup_memory(pid,allocation_bytes=allocation)
        fractions=limits.memory_fractions(rss_bytes=rss,pss_bytes=pss if pss_available else None,cgroup=cg,allocation_bytes=allocation)
        qualification='LINUX_CGROUP_HIERARCHY_READ'
    else:
        # Fake/dry-run tests on Windows do not qualify Linux resource gates.
        cg={'status':'SIMULATION_ONLY' if simulation else 'NOT_LINUX',
            'reason':'linux_cgroup_qualification_not_available'}
        fractions={'tree_rss_bytes':rss,'tree_pss_bytes':pss if pss_available else None,
            'allocation_fraction_rss':rss/allocation,'effective_limit_bytes':allocation,
            'max_enforced_fraction':rss/allocation,'cgroup':cg,
            'status':'SIMULATION_ONLY','pss_is_report_only':True}
        qualification='SIMULATION_ONLY_NOT_LINUX_RESOURCE_QUALIFICATION'
    return dict(available=True,pids=pids,rss_bytes=rss,pss_bytes=pss if pss_available else 'NOT_MEASURED',threads=threads,process_tree_cpu_seconds=cpu,affinity=affinity,cgroup=cg,memory=fractions,resource_qualification=qualification)

def progress_state(path):
    if not path.exists():return None,set()
    lines=path.read_bytes().splitlines(keepends=True);first=None;complete=set()
    for line in lines:
        if not line.endswith(b'\n'):continue
        x=json.loads(line)
        if x.get('event')=='first_interval_started':
            first=float(x['monotonic'])
            if not math.isfinite(first) or first<=0:raise ValueError('invalid start timestamp')
        if x.get('event')=='complete_interval':
            boundary=float(x['boundary_monotonic'])
            if not math.isfinite(boundary) or boundary<=0 or (first is not None and boundary<first):raise ValueError('invalid boundary timestamp')
            complete.add(int(x['interval_index']))
    return first,complete

def supervise_child(command, case, *, env, first_limit=900., window_limit=2700., sample_seconds=1., fixture='ok', mode=None, simulation=False, progress_clock='monotonic'):
    import psutil
    if progress_clock not in ('monotonic', 'perf_counter'):
        raise ValueError('unsupported progress clock')
    clock = time.perf_counter if progress_clock == 'perf_counter' else time.monotonic
    case.mkdir();start=clock();proc=None;created=None;reason=None;latch=False;peak=0;peak_threads=0;known=[]
    old={}
    def interrupted(signum, frame):raise InterruptedError('TERMINATION_SIGNAL_'+str(signum))
    for sig in (signal.SIGINT,signal.SIGTERM):old[sig]=signal.signal(sig,interrupted)
    try:
        with (case/'stdout.log').open('xb') as out,(case/'stderr.log').open('xb') as err:
            proc=subprocess.Popen(command,cwd=SCRIPT.parents[4],env=env,stdout=out,stderr=err,start_new_session=True)
        created=psutil.Process(proc.pid).create_time()
        while True:
            running=proc.poll() is None
            first,complete=progress_state(case/'work/propagation/observer_progress.jsonl');now=clock()
            if first is not None and not start<=first<=now:raise ValueError('start timestamp outside child lifetime')
            if not running:break
            snap=resource_snapshot(proc.pid,simulation=True) if simulation else resource_snapshot(proc.pid)
            if snap.get('available'):
                fraction=snap['memory']['max_enforced_fraction'];known=snap['pids'];peak=max(peak,snap['rss_bytes']);peak_threads=max(peak_threads,snap['threads'])
                if fixture in ('memory80','memory90'):fraction=.81 if fixture=='memory80' else .91;snap['simulation']='INJECTED_MEMORY_FRACTION'
                if fraction is None:raise RuntimeError('MEMORY_TELEMETRY_UNAVAILABLE')
                latch |= fraction>=.8
                if fraction>=.9:reason='MEMORY_90_PERCENT'
                append(case/'resources.jsonl',dict(elapsed=now-start,**snap))
            if first is not None and 0 not in complete and now-first>=first_limit:reason='FIRST_INTERVAL_TIMEOUT'
            if first is None and now-start>=first_limit:reason='FIRST_INTERVAL_START_NOT_OBSERVED'
            phase=case/'work/propagation_start_admission.json'
            if first is None and phase.exists() and now-float(json.loads(phase.read_text())['monotonic'])>=first_limit:
                reason='FIRST_INTERVAL_START_NOT_OBSERVED'
            if now-start>=window_limit:reason='WINDOW_TIMEOUT'
            if reason:break
            time.sleep(sample_seconds)
    except BaseException as exc:reason=type(exc).__name__+': '+str(exc)
    finally:
        if proc and created and (proc.poll() is None or reason):terminate(proc.pid,created)
        for sig,handler in old.items():signal.signal(sig,handler)
    exit_code=None if proc is None else proc.wait(timeout=5)
    # Guardian and any sampled descendant must finish; no success with live leftovers.
    guard_path=case/'work/guardian.json'
    if guard_path.exists():known.append(json.loads(guard_path.read_text())['pid'])
    until=clock()+5
    while clock()<until:
        alive=[]
        for pid in set(known):
            try:
                p=psutil.Process(pid)
                if p.is_running() and p.status()!=psutil.STATUS_ZOMBIE:alive.append(p)
            except psutil.NoSuchProcess:pass
        if not alive:break
        time.sleep(.05)
    else:
        reason=reason or 'RESIDUAL_PROCESS'
        for p in alive:
            try:p.kill()
            except psutil.Error:pass
    try:
        first,complete=progress_state(case/'work/propagation/observer_progress.jsonl')
    except Exception as exc:
        first,complete=None,set();reason=reason or ('INVALID_PROGRESS: '+str(exc))
    result=dict(progress_clock=progress_clock,status='COMPLETE' if exit_code==0 and reason is None else 'BLOCKED_CHILD',mode=mode,exit_code=exit_code,stop_reason=reason,full_service_wall_seconds=clock()-start,first_interval_observed=first is not None,complete_intervals=sorted(complete),rss80_latched=latch,peak_tree_rss_bytes=peak,peak_threads=peak_threads,sample_seconds=sample_seconds,rss_semantics='sum process tree RSS; shared pages may be counted repeatedly',resource_semantics='sampled peak not continuous; not DRAM traffic',child_pid=proc.pid if proc else None)
    write(case/'supervisor_receipt.json',result);return result

def compare_pair(root):
    science_path()
    from tools.cpu_o1 import q1_t0
    from tools.cpu_o1.cpu_s4.profile import _read_scalar_fields
    from tools.cpu_o1.numeric_gate import classify_comparison
    off=root/'I_OFF/work';pro=root/'I_PROFILE/work'
    rows=q1_t0._compare_window_outputs(off,pro,'I',comparison_path=root/'comparison.jsonl',scalar_reference=_read_scalar_fields(off/'scientific_scalar_diagnostics.json'),scalar_candidate=_read_scalar_fields(pro/'scientific_scalar_diagnostics.json'),case_id='PROFILE',reference_case_id='OFF',comparison_stage='s4_independent_tiny')
    classified=[classify_comparison(x) for x in rows]
    good=bool(rows) and all(x.get('proceed') for x in classified)
    result=dict(status='EXACT' if good else 'NUMERIC_DIAGNOSTIC_ONLY',raw_rows=len(rows),classifications=classified,numeric_tolerance_authorized=False)
    write(root/'comparison_gate.json',result);return result

def _ratio_and_overhead(numerator, denominator):
    if (isinstance(numerator,bool) or isinstance(denominator,bool)
            or not isinstance(numerator,(int,float)) or not isinstance(denominator,(int,float))
            or not math.isfinite(float(numerator)) or not math.isfinite(float(denominator))
            or float(numerator)<=0 or float(denominator)<=0):
        return None,None
    ratio=float(numerator)/float(denominator)
    if not math.isfinite(ratio) or ratio<=0:return None,None
    return ratio,ratio-1.0

def _verify_first_boundary_acceptance(case, mode, *, raw_sha256_file):
    work=case/'work';snapshot_root=work/'first_boundary'
    required=(work/'first_boundary_receipt.json',snapshot_root/'snapshot.json',
        snapshot_root/'NUMERIC_GATE.json',snapshot_root/'E.npy',
        snapshot_root/'POST.npz',snapshot_root/'Qacc.npy',
        snapshot_root/'Qacc_raman.npy',snapshot_root/'diagnostics.npz',
        snapshot_root/'scalar_diagnostics.json')
    started=time.perf_counter();hash_seconds=0.0
    if any(not path.is_file() or path.is_symlink() for path in required):
        raise RuntimeError('FIRST_BOUNDARY_REQUIRED_ARTIFACT_MISSING')
    def strict(path):
        return json.loads(path.read_text(encoding='utf-8'),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    receipt=strict(required[0]);snapshot=strict(required[1]);gate=strict(required[2])
    child=strict(work/'child_receipt.json');window=strict(work/'window_receipt.json')
    contract=strict(work/'input_output_contract.json')
    seal=strict(work/'OUTPUT_SEAL.json')
    case_id='I_'+mode
    child_boundary=child.get('first_boundary_gate')
    window_boundary=window.get('first_boundary_gate')
    if (receipt.get('identity')!='I' or receipt.get('interval_index')!=0
            or receipt.get('B0_status')!='DONE'
            or receipt.get('gate_proceed') is not True
            or receipt.get('complete_E_POST_diagnostics_snapshot') is not True
            or receipt.get('Qacc_valid') is not True
            or receipt.get('Qacc_raman_valid') is not True
            or gate.get('identity')!='I' or gate.get('case_id')!=case_id
            or gate.get('proceed_to_next_identity') is not True):
        raise RuntimeError('FIRST_BOUNDARY_NUMERIC_GATE_RECEIPT_INVALID')
    if (child.get('case_id')!=case_id or window.get('case_id')!=case_id
            or window.get('mode')!=mode or window.get('implementation')!='I'
            or not isinstance(child_boundary,dict) or not isinstance(window_boundary,dict)
            or child_boundary.get('case_id')!=case_id or window_boundary.get('case_id')!=case_id
            or child_boundary.get('gate')!=gate or window_boundary.get('gate')!=gate
            or child_boundary.get('snapshot')!=receipt
            or window_boundary.get('snapshot')!=receipt):
        raise RuntimeError('FIRST_BOUNDARY_CHILD_BINDING_MISSING')
    # Q1 writes snapshot.json before adding post-snapshot gate fields to the
    # immutable receipt. Compare the persisted snapshot to that pre-gate view.
    post_gate_fields=('B0_status','gate_proceed','gate_classification',
        'cross_identity_comparison_seconds','self_validation_seconds')
    persisted_view={key:value for key,value in receipt.items() if key not in post_gate_fields}
    if snapshot!=persisted_view:
        raise RuntimeError('FIRST_BOUNDARY_SNAPSHOT_RECEIPT_MISMATCH')
    if gate!=strict(snapshot_root/'NUMERIC_GATE.json'):
        raise RuntimeError('FIRST_BOUNDARY_NUMERIC_GATE_FILE_MISMATCH')
    seal_names={
        'first_boundary_receipt.json','first_boundary/snapshot.json',
        'first_boundary/NUMERIC_GATE.json','first_boundary/E.npy',
        'first_boundary/POST.npz','first_boundary/Qacc.npy',
        'first_boundary/Qacc_raman.npy','first_boundary/diagnostics.npz',
        'first_boundary/scalar_diagnostics.json'}
    auxiliary=seal.get('auxiliary_files')
    if (seal.get('schema')!='filament.cpu_s4.output_seal.v1'
            or not isinstance(auxiliary,dict) or set(auxiliary)!=seal_names):
        raise RuntimeError('FIRST_BOUNDARY_OUTPUT_SEAL_MEMBERS_INVALID')
    if child.get('output_seal_sha256')!=raw_sha256_file(work/'OUTPUT_SEAL.json'):
        raise RuntimeError('FIRST_BOUNDARY_OUTPUT_SEAL_BINDING_MISMATCH')
    file_rows={}
    for relative in sorted(seal_names):
        path=work/relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError('FIRST_BOUNDARY_REQUIRED_ARTIFACT_MISSING:'+relative)
        before=time.perf_counter();digest=raw_sha256_file(path);hash_seconds+=time.perf_counter()-before
        if not isinstance(auxiliary.get(relative),str) or len(auxiliary[relative])!=64 or digest!=auxiliary[relative]:
            raise RuntimeError('FIRST_BOUNDARY_OUTPUT_SEAL_HASH_MISMATCH:'+relative)
        file_rows[relative]={'bytes':path.stat().st_size,'raw_sha256':digest}
    for name in ('Qacc.npy','Qacc_raman.npy'):
        expected=child_boundary.get('capture_hashes',{}).get(name)
        if (not expected or window_boundary.get('capture_hashes',{}).get(name)!=expected
                or file_rows['first_boundary/'+name]['raw_sha256']!=expected):
            raise RuntimeError('FIRST_BOUNDARY_CAPTURE_HASH_MISMATCH:'+name)
    expected_hashes={
        'E.npy':'first_boundary_E_raw_sha256',
        'POST.npz':'first_boundary_POST_raw_sha256',
        'diagnostics.npz':'first_boundary_diagnostics_raw_sha256',
        'scalar_diagnostics.json':'first_boundary_scalars_raw_sha256',
    }
    for name,key in expected_hashes.items():
        if file_rows['first_boundary/'+name]['raw_sha256']!=receipt.get(key):
            raise RuntimeError('FIRST_BOUNDARY_SNAPSHOT_HASH_MISMATCH:'+name)
    if (contract.get('implementation')!='I' or contract.get('mode')!=mode
            or contract.get('source_dtype')!='complex128'
            or contract.get('qacc_dtype')!='float64'
            or not isinstance(contract.get('source_shape'),list)
            or not isinstance(contract.get('qacc_shape'),list)):
        raise RuntimeError('FIRST_BOUNDARY_OUTPUT_CONTRACT_INVALID')
    import numpy as np
    inventory=s4_module('output_inventory.py','s4_output_inventory')
    source_shape=tuple(contract['source_shape']);qacc_shape=tuple(contract['qacc_shape'])
    if len(source_shape)!=3 or len(qacc_shape)!=2 or source_shape[-2:]!=qacc_shape:
        raise RuntimeError('FIRST_BOUNDARY_OUTPUT_SHAPE_CONTRACT_INVALID')
    validation_started=time.perf_counter()
    arrays=[]
    for name,shape,dtype in (('E.npy',source_shape,'complex128'),
            ('Qacc.npy',qacc_shape,'float64'),('Qacc_raman.npy',qacc_shape,'float64')):
        array=np.load(snapshot_root/name,mmap_mode='r',allow_pickle=False)
        if not isinstance(array,np.ndarray) or tuple(array.shape)!=shape or str(array.dtype)!=dtype:
            raise RuntimeError('FIRST_BOUNDARY_ARRAY_SHAPE_OR_DTYPE_INVALID:'+name)
        inventory._finite_bounded(array,np=np,label='first_boundary/'+name)
        arrays.append(name)
    with np.load(snapshot_root/'POST.npz',allow_pickle=False) as post:
        if set(post.files)!={'delta_n','vx','vy'}:
            raise RuntimeError('FIRST_BOUNDARY_POST_MEMBERS_INVALID')
        for name in post.files:
            arr=post[name]
            if tuple(arr.shape)!=qacc_shape or str(arr.dtype)!='float64':
                raise RuntimeError('FIRST_BOUNDARY_POST_SHAPE_OR_DTYPE_INVALID:'+name)
            inventory._finite_bounded(arr,np=np,label='first_boundary.POST.'+name)
    with np.load(snapshot_root/'diagnostics.npz',allow_pickle=False) as diagnostics:
        if sorted(diagnostics.files)!=receipt.get('diagnostic_array_fields'):
            raise RuntimeError('FIRST_BOUNDARY_DIAGNOSTIC_FIELDS_INVALID')
        for name in diagnostics.files:
            arr=diagnostics[name]
            if arr.dtype.hasobject or arr.dtype.kind not in 'biufcSU' or not arr.shape or any(x<=0 for x in arr.shape):
                raise RuntimeError('FIRST_BOUNDARY_DIAGNOSTIC_SHAPE_OR_DTYPE_INVALID:'+name)
            if arr.dtype.kind in 'biufc':
                inventory._finite_bounded(arr,np=np,label='first_boundary.diagnostics.'+name,
                    permit_all_nan=name in inventory.NAN_PLACEHOLDER_FIELDS)
    scalar=inventory._strict_json(snapshot_root/'scalar_diagnostics.json')
    if (scalar.get('schema')!='filament.cpu_q1.propagation_scalar_diagnostics.v1'
            or scalar.get('identity')!='I' or not isinstance(scalar.get('fields'),dict)):
        raise RuntimeError('FIRST_BOUNDARY_SCALAR_DIAGNOSTICS_INVALID')
    stability_hash_seconds=0.0
    for name in ('first_boundary_receipt.json','first_boundary/snapshot.json',
            'first_boundary/NUMERIC_GATE.json','first_boundary/E.npy',
            'first_boundary/POST.npz','first_boundary/Qacc.npy',
            'first_boundary/Qacc_raman.npy','first_boundary/diagnostics.npz',
            'first_boundary/scalar_diagnostics.json'):
        path=work/name;before=path.stat();hash_started=time.perf_counter();digest=raw_sha256_file(path);stability_hash_seconds+=time.perf_counter()-hash_started;after=path.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise RuntimeError('FIRST_BOUNDARY_OUTPUT_CHANGED_DURING_ACCEPTANCE:'+name)
        if digest!=auxiliary[name]:raise RuntimeError('FIRST_BOUNDARY_OUTPUT_CHANGED_DURING_ACCEPTANCE:'+name)
    hash_seconds+=stability_hash_seconds
    validation_seconds=time.perf_counter()-validation_started
    return {'status':'PASS','case_id':case_id,'implementation':'I',
        'snapshot_sha256':file_rows['first_boundary/snapshot.json']['raw_sha256'],
        'numeric_gate_sha256':file_rows['first_boundary/NUMERIC_GATE.json']['raw_sha256'],
        'first_boundary_gate_classification':gate.get('classification'),
        'validated_snapshot_files':len(required),'files':file_rows,
        'hash_seconds':hash_seconds,'validation_seconds':validation_seconds,
        'acceptance_seconds':time.perf_counter()-started}


def accept_child_outputs(case, mode):
    """Run fixed inventory acceptance in the parent after the child is gone."""
    science_path()
    from tools.hpc_ops import provenance_v2 as pv
    from KHz_filament.hr4e_timestep import sha256_array_streaming
    inventory=s4_module('output_inventory.py','s4_output_inventory')
    acceptance_started=time.perf_counter()
    result=inventory.verify_output_inventory(case,mode,
        raw_sha256_file=pv.raw_sha256_file,
        canonical_array_sha256=sha256_array_streaming)
    try:
        boundary_started=time.perf_counter()
        result['first_boundary_acceptance']=_verify_first_boundary_acceptance(
            Path(case),mode,raw_sha256_file=pv.raw_sha256_file)
        result['first_boundary_acceptance_seconds']=time.perf_counter()-boundary_started
    except Exception as exc:
        result['status']='BLOCKED_FIRST_BOUNDARY_INVENTORY'
        result['ledger_closed']=False
        result['failures'].append(f'{type(exc).__name__}: {exc}')
        result['first_boundary_acceptance_seconds']=time.perf_counter()-acceptance_started-result.get('inventory_seconds',0.0)
    boundary=result.get('first_boundary_acceptance') or {}
    result['parent_output_acceptance_seconds']=time.perf_counter()-acceptance_started
    result['hash_seconds']=result.get('hash_seconds',0.0)+boundary.get('hash_seconds',0.0)
    result['validation_seconds']=result.get('validation_seconds',0.0)+boundary.get('validation_seconds',0.0)
    return result

def pair(args):
    scoped=None
    if not (args.dry_run or args.tiny_science):scoped=_require_scoped_real_authorization(args)
    if args.implementation!='I':raise ValueError('only I supported')
    real=not (args.dry_run or args.tiny_science)
    allocation_start=None
    if real:
        if (args.first_limit,args.window_limit,args.sample_seconds)!=(900.,2700.,1.):raise ValueError('fixed real deadlines required')
        allocation_start=float(os.environ.get('SLURM_JOB_START_TIME','0'))
        if allocation_start<=0 or allocation_start>time.time():raise RuntimeError('ALLOCATION_START_TIME_REQUIRED')
    import psutil
    root=args.output_root
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True);start=time.monotonic();cases=[];status='BLOCKED';failure=None;comparison_seconds=None
    write(root/'PAIR_STARTED.json',dict(status='STARTED_NOT_ACCEPTED',parent_pid=os.getpid(),monotonic=start,modes=['OFF','PROFILE'],real_execution_enabled=bool(scoped),hpc_submission='NOT_SUBMITTED_BY_S4_CLI'))
    before=[x for x in ('numpy','scipy','numba','KHz_filament') if x in sys.modules]
    env=os.environ.copy();env.update(THREADS);env.update({x:'1' for x in BLAS});env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPATH=str(SCRIPT.parents[3]),NUMBA_CACHE_DIR=str(root/'jit_cache'),GIT_OPTIONAL_LOCKS='0')
    if args.tiny_science:env['NUMBA_NUM_THREADS']='12'
    parent=psutil.Process()
    try:
        for mode in ('OFF','PROFILE'):
            allocation_elapsed=(time.time()-allocation_start) if real else time.monotonic()-start
            required_remaining=(2700+900+300) if mode=='PROFILE' else (900+2700+900+300)
            if real and 7200-allocation_elapsed<required_remaining:
                raise RuntimeError('ALLOCATION_REMAINING_BUDGET:'+mode)
            case=root/('I_'+mode)
            fixture=args.fixture if mode==args.fail_mode else 'ok'
            case_service_started=time.perf_counter()
            command=[sys.executable,'-s','-B',str(SCRIPT),'--single-mode',mode,'--output-root',str(case/'work'),'--parent-pid',str(parent.pid),'--parent-created',str(parent.create_time()),'--fixture',fixture]
            if args.dry_run:command+=['--dry-run']
            elif args.tiny_science:command+=['--tiny-science']
            else:command+=['--run-config',str(args.run_config),'--approval',str(args.approval)]
            if not args.dry_run:
                command+=['--case-id','I_'+mode,'--authorized-run-root',str(root)]
                if mode=='PROFILE':
                    command+=['--first-boundary-reference-root',str(root/'I_OFF'/'work'/'first_boundary'),
                        '--first-boundary-reference-case-id','I_OFF']
            receipt=supervise_child(command,case,env=env,first_limit=args.first_limit,window_limit=args.window_limit,sample_seconds=args.sample_seconds,fixture=fixture,mode=mode,simulation=args.dry_run,progress_clock='monotonic' if args.dry_run else 'perf_counter')
            case_result=dict(receipt)
            cases.append(case_result)
            if receipt['status']!='COMPLETE':raise RuntimeError(receipt['stop_reason'] or 'NONZERO_CHILD_EXIT')
            child=json.loads((case/'work/child_receipt.json').read_text());validate_child_receipt(child,mode,require_first_boundary=not args.dry_run)
            if not args.dry_run:
                acceptance=accept_child_outputs(case,mode)
                write(case/'OUTPUT_ACCEPTANCE.json',acceptance)
                cases[-1]['independent_output_inventory_status']=acceptance['status']
                cases[-1]['output_acceptance_seconds']=acceptance['parent_output_acceptance_seconds']
                cases[-1]['output_hash_seconds']=acceptance['hash_seconds']
                if acceptance['status']!='PASS':raise RuntimeError('OUTPUT_INVENTORY_REJECTED:'+json.dumps(acceptance['failures'],allow_nan=False))
            if receipt['complete_intervals']!=list(range(8)) or not receipt['first_interval_observed']:raise RuntimeError('MISSING_PROGRESS_BOUNDARIES')
            if mode=='OFF':reference=child
            elif any(child[k]!=reference[k] for k in ('source_hash','pre_hash')):raise RuntimeError('INPUT_CHANGED_BETWEEN_MODES')
            if receipt['rss80_latched']:raise RuntimeError('MEMORY_80_PERCENT_NO_NEXT_WINDOW')
            case_result['child_supervision_wall_seconds']=receipt['full_service_wall_seconds']
            boundary_gate=child.get('first_boundary_gate') if not args.dry_run else None
            cross_seconds=float((boundary_gate or {}).get('gate',{}).get(
                'cross_identity_comparison_seconds',0.0)) if mode=='PROFILE' else 0.0
            raw_service=time.perf_counter()-case_service_started
            if not math.isfinite(cross_seconds) or cross_seconds<0 or cross_seconds>raw_service:
                raise RuntimeError('INVALID_FIRST_BOUNDARY_CROSS_WINDOW_TIMING')
            case_result['child_supervision_wall_seconds_raw_including_cross_window_comparison']=receipt['full_service_wall_seconds']
            case_result['child_supervision_wall_seconds']=max(0.0,receipt['full_service_wall_seconds']-cross_seconds)
            case_result['full_service_wall_seconds_raw_including_cross_window_comparison']=raw_service
            case_result['cross_window_comparison_seconds']=cross_seconds
            case_result['full_service_wall_seconds']=raw_service-cross_seconds
            case_result['full_service_boundary']='parent launch through child exit, guardian/log close, and parent output acceptance; cross-window first-boundary comparison excluded once in normalized value'
            case_result['parent_output_acceptance_seconds']=case_result.get('output_acceptance_seconds',0.0)
            case_result['child_receipt_path']=str((case/'work/child_receipt.json').resolve()) if not args.dry_run else None
            case_result['window_receipt_path']=str((case/'work/window_receipt.json').resolve()) if not args.dry_run else None
            case_result['output_acceptance_path']=str((case/'OUTPUT_ACCEPTANCE.json').resolve()) if not args.dry_run else None
            if not args.dry_run:
                raw_propagation=child.get('propagation_wall_seconds_raw_including_cross_window_comparison')
                propagation=child.get('propagation_wall_seconds')
                case_result['propagation_wall_seconds_raw_including_cross_window_comparison']=raw_propagation
                case_result['propagation_cross_window_comparison_seconds']=child.get('propagation_cross_window_comparison_seconds')
                case_result['propagation_wall_seconds']=propagation
                case_result['propagation_process_cpu_seconds']=child.get('propagation_process_cpu_seconds')
                case_result['process_cpu_seconds']=child.get('process_cpu_seconds')
        compare_started=time.monotonic()
        comparison=({'status':'FAKE_FIXTURE_ONLY_NO_SCIENTIFIC_PASS'} if args.dry_run else compare_pair(root))
        comparison_seconds=time.monotonic()-compare_started
        if not args.dry_run and comparison['status']!='EXACT':raise RuntimeError('NUMERIC_DIAGNOSTIC_ONLY')
        status='DRY_RUN_PASS' if args.dry_run else ('REAL8_OFF_PROFILE_EXACT_PASS' if real else 'LOCAL_TINY_INDEPENDENT_EXACT_PASS')
    except BaseException as exc:failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    service_ratio,service_overhead=_ratio_and_overhead(
        cases[1].get('full_service_wall_seconds') if len(cases)>1 else None,
        cases[0].get('full_service_wall_seconds') if cases else None)
    propagation_ratio,propagation_overhead=_ratio_and_overhead(
        cases[1].get('propagation_wall_seconds') if len(cases)>1 else None,
        cases[0].get('propagation_wall_seconds') if cases else None)
    first_boundary_cross_total=sum(float(row.get('cross_window_comparison_seconds',0.0))
        for row in cases if isinstance(row.get('cross_window_comparison_seconds'),(int,float)))
    summary=dict(status=status,cases=cases,error=failure,fixture=bool(args.dry_run),
        real8_profiled=bool(real and len(cases)==2 and cases[1].get('status')=='COMPLETE'),
        implementation='I',case_ids=['I_OFF','I_PROFILE'],
        hpc_submission='NOT_SUBMITTED_BY_S4_CLI',parent_science_modules_before_launch=before,
        elapsed_seconds=time.monotonic()-start,
        full_service_boundary='parent launch through child exit, guardian/log close, and parent output acceptance',
        comparison_seconds=comparison_seconds,
        first_boundary_cross_window_comparison_seconds=first_boundary_cross_total,
        endpoint_comparison_seconds=comparison_seconds,
        total_cross_window_comparison_seconds=first_boundary_cross_total+(comparison_seconds or 0.0),
        instrumentation_ratio=service_ratio,
        instrumentation_overhead_fraction=service_overhead,
        propagation_profile_over_off_ratio=propagation_ratio,
        propagation_overhead_fraction=propagation_overhead,
        service_pair_status='COMPLETE' if service_ratio is not None else 'PARTIAL_OR_UNAVAILABLE',
        propagation_pair_status='COMPLETE' if propagation_ratio is not None else 'PARTIAL_OR_UNAVAILABLE')
    write(root/'PAIR_RECEIPT.json',summary)
    return 0 if status.endswith('PASS') else 2

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--preflight-only',action='store_true');g.add_argument('--supervise-pair',action='store_true');g.add_argument('--single-mode',choices=['OFF','PROFILE']);g.add_argument('--guardian',action='store_true')
    p.add_argument('--output-root',type=Path);m=p.add_mutually_exclusive_group();m.add_argument('--dry-run',action='store_true');m.add_argument('--tiny-science',action='store_true')
    p.add_argument('--implementation',default='I');p.add_argument('--fixture',choices=['ok','hang','spawn','exit','coverage','ledger','memory80','memory90'],default='ok');p.add_argument('--fail-mode',choices=['OFF','PROFILE'],default='OFF')
    p.add_argument('--first-limit',type=float,default=900);p.add_argument('--window-limit',type=float,default=2700);p.add_argument('--sample-seconds',type=float,default=1)
    p.add_argument('--parent-pid',type=int);p.add_argument('--parent-created',type=float);p.add_argument('--target-pid',type=int);p.add_argument('--target-created',type=float);p.add_argument('--ready',type=Path)
    p.add_argument('--run-config',type=Path);p.add_argument('--approval',type=Path)
    p.add_argument('--case-id',choices=['I_OFF','I_PROFILE'])
    p.add_argument('--first-boundary-reference-root',type=Path)
    p.add_argument('--first-boundary-reference-case-id')
    p.add_argument('--authorized-run-root',type=Path)
    args=p.parse_args(argv)
    try:
        if args.guardian:return guardian(args)
        if args.preflight_only:
            print(json.dumps(dict(status='PREPARATION_ONLY_REAL_DISABLED',real_execution_enabled=False,HPC_SUBMITTED=False,source_input_validation='Use admission.py with reviewed manifests; default metadata only',allocation_runtime='DEFERRED',parent_science_modules=[x for x in ('numpy','scipy','numba','KHz_filament') if x in sys.modules]),allow_nan=False));return 0
        if not args.output_root:raise ValueError('output-root required')
        if args.sample_seconds<=0 or args.first_limit<=0 or args.window_limit<=0:raise ValueError('positive deadlines required')
        if args.tiny_science and args.fixture!='ok':raise ValueError('fake failure injection cannot be used with science')
        if args.single_mode:return single(args)
        return pair(args)
    except BaseException as exc:
        print(json.dumps(dict(status='BLOCKED',error=type(exc).__name__+': '+str(exc),HPC_SUBMITTED=False),allow_nan=False),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
