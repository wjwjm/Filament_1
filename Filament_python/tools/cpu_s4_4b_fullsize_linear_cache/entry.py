"""One source-bound 4B window; the accepted fullsize loader and science stay fixed."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
from types import FunctionType,SimpleNamespace
from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_3g3_smallgrid.candidate import unoptimized
from tools.cpu_s4_fullsize_confirmation import entry as frozen_full
from tools.hpc_ops import provenance_v2 as pv

TASK='CPU-S4-4B-FULLSIZE-LINEAR-FACTOR-CACHE-CONFIRM-20261027'
SCHEMA='filament.cpu_s4_4b.entry.v1'
BINDING_SCHEMA='filament.cpu_s4_4b.execution.v1'
SCIENCE_TREE='821fec5bdacd832b0347e1f2e84285cb54172e57'
FULL_SHAPE=(384,351,301);MICRO_SHAPES=((32,17,19),(384,17,19))
ROLES=('G3_REF_FULLSIZE','L1_CACHE_FULLSIZE')
BINDING_KEYS={'schema','task','execution_commit','baseline_commit','science_tree','shape','intervals','cache_max_bytes','cache_max_tables','source_records'}

def require(ok,why,context=None):
    if not ok:raise RuntimeError(dict(gate=why,context=context))

def _pairs(items):
    value={}
    for key,item in items:
        require(key not in value,'DUPLICATE_JSON_KEY',key);value[key]=item
    return value

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=_pairs)

def verify_execution(report,*,local_engineering,shape,intervals,require_node=True):
    unoptimized();report=Path(report).resolve(strict=True)
    binding=read(report/'S44B_EXECUTION_BINDING.json')
    require(isinstance(binding,dict) and set(binding)==BINDING_KEYS,'EXECUTION_BINDING_FIELDS')
    require(binding['schema']==BINDING_SCHEMA and binding['task']==TASK,'EXECUTION_BINDING_SCHEMA')
    require(binding['baseline_commit']=='16879445d9aff784a6c3f2d7edf33a0a464efc62' and binding['science_tree']==SCIENCE_TREE,'ACCEPTED_BASELINES')
    require(binding['shape']==list(FULL_SHAPE) and binding['intervals']==8 and binding['cache_max_bytes']==1610612736 and binding['cache_max_tables']==2,'FULLSIZE_CAPACITY_AND_DOMAIN')
    repo=Path(__file__).resolve().parents[3]
    if not local_engineering:
        plan=read(report/'HPC_EXPERIMENT_PLAN.json')
        require(plan['schema']=='filament.cpu_s4_4b.hpc_experiment.v1' and plan['task']==TASK and plan['shape']==list(FULL_SHAPE) and plan['intervals']==8 and plan['warmup_intervals']==0,'HPC_PLAN_DOMAIN')
        require(pv.raw_sha256_file(report/'HPC_EXPERIMENT_PLAN.json')==os.environ.get('S44B_HPC_PLAN_SHA256'),'HPC_PLAN_RAW_IDENTITY')
        from tools.cpu_s4_checkpoint_support.git_readonly_compat import verify_runtime_query_tool
        verify_runtime_query_tool(plan)
    require(pv._git(repo,'rev-parse','HEAD')==binding['execution_commit'],'ACTUAL_COMMITTED_EXECUTION_HEAD')
    pv._ensure_clean(repo)
    require(pv._git(repo,'rev-parse','HEAD:Filament_python/KHz_filament')==SCIENCE_TREE,'UNCHANGED_SCIENCE_TREE')
    require(isinstance(binding['source_records'],list) and binding['source_records'],'SOURCE_RECORDS_REQUIRED')
    for row in binding['source_records']:
        relative=row['path'];path=repo/relative
        require(not Path(relative).is_absolute() and path.resolve().is_relative_to(repo),'SOURCE_PATH_CONFINEMENT')
        require(row.get('classification')=='tracked_text' and row.get('hash_scope')==pv.TRACKED_HASH_SCOPE,'SOURCE_HASH_SCOPE')
        require(pv._git(repo,'rev-parse','HEAD:'+relative)==row['git_blob_oid'] and pv.canonical_lf_sha256(path.read_bytes())==row['canonical_lf_sha256'],'COMMITTED_SOURCE_BLOB_AND_CANONICAL_IDENTITY',relative)
    if local_engineering:
        require(not os.environ.get('SLURM_JOB_ID') and tuple(shape) in MICRO_SHAPES and intervals in (2,8),'LOCAL_MICRO_ONLY_NO_FULL_ARRAYS')
        return binding
    require(os.environ.get('SLURM_JOB_ID','').isdigit() and tuple(shape)==FULL_SHAPE and intervals==8,'ACTUAL_SLURM_FULLSIZE_ENTRY8')
    require(plan['execution_commit']==binding['execution_commit'] and pv.raw_sha256_file(report/'S44B_EXECUTION_BINDING.json')==os.environ.get('S44B_EXECUTION_BINDING_SHA256'),'HPC_EXECUTION_BINDING_RAW_IDENTITY')
    if require_node:
        node=read(report/'NODE_GATE.json')
        require(node['status']=='PASS_S44B_FULLSIZE_SOURCE_RESOURCE_INPUT_GATES' and node['job_id']==os.environ['SLURM_JOB_ID'],'ACTUAL_S44B_NODE_GATE')
    return plan

def full_context(report,*,shape=FULL_SHAPE):
    namespace=dict(frozen_full.full_context.__globals__)
    namespace['verify_execution']=lambda path:verify_execution(path,local_engineering=False,shape=FULL_SHAPE,intervals=8)
    loader=FunctionType(frozen_full.full_context.__code__,namespace,frozen_full.full_context.__name__,frozen_full.full_context.__defaults__,frozen_full.full_context.__closure__)
    loader.__kwdefaults__=frozen_full.full_context.__kwdefaults__
    return loader(report,shape=shape)

def private_window(context_function,timings):
    window=frozen_full.full_window(context_function)
    def writer_factory(report,intervals):
        original=sg.normal_writer(report,intervals)
        def writer(*args,**kwargs):
            started=time.perf_counter()
            try:return original(*args,**kwargs)
            finally:timings['normal_output_writer_s']+=time.perf_counter()-started
        return writer
    window.__globals__['normal_writer']=writer_factory
    return window

def _cache_unloaded():
    return not any(name.startswith('tools.cpu_s4_4a_linear_cache') or name in ('tools.cpu_s4_4b_fullsize_linear_cache.cache','tools.cpu_s4_4b_fullsize_linear_cache.candidate') for name in sys.modules)

def array_raw_index(work,result):
    """Reuse existing raw seals; hash only missing persisted array artifacts.

    This is application-external research provenance, never a live E/cache key
    guard or factor-table rescan. Large bytes are read only on a compute node
    (or for the explicitly admitted local micro test artifacts).
    """
    old=read(result['output_raw_index']);normal=result['normal_outputs']
    roots=dict(OUTPUT=Path(result['output']),PROPAGATION=Path(normal['propagation_root']),SCIENTIFIC_QACC=Path(normal['scientific_Qacc']).parent,SCIENTIFIC_SCALARS=Path(normal['scientific_scalar_diagnostics']).parent)
    known={}
    for name,rows in old['roots'].items():
        for row in rows:known[(roots[name]/row['path']).resolve()]=row
    rows=[]
    for path in sorted(work.rglob('*')):
        if path.suffix.lower() not in ('.npy','.npz','.zip') or not path.is_file():continue
        require(not path.is_symlink() and path.resolve().is_relative_to(work),'ARRAY_ARTIFACT_PATH_CONFINEMENT')
        stat=path.stat();existing=known.get(path.resolve())
        if existing is not None:
            require(existing['hash_scope']=='raw_bytes' and existing['size_bytes']==stat.st_size,'REUSED_ORIGINAL_RAW_SEAL_SIZE')
            digest=existing['sha256_raw'];origin='ORIGINAL_OUTPUT_RAW_INDEX_UNCHANGED_CLOSED_ARTIFACT'
        else:digest=pv.raw_sha256_file(path);origin='ONE_MISSING_PERSISTED_ARTIFACT_RAW_SEAL_RESEARCH_ONLY'
        rows.append(dict(path=str(path),classification='external',hash_scope='raw_bytes',raw_sha256=digest,size_bytes=stat.st_size,mode=oct(stat.st_mode & 0o777),uid=getattr(stat,'st_uid',None),gid=getattr(stat,'st_gid',None),readable=os.access(path,os.R_OK),writable=os.access(path,os.W_OK),identity_origin=origin))
    target=work/'ALL_ARRAY_RAW_INDEX.json';sg.write(target,dict(schema=SCHEMA+'.array_raw_index.v1',hash_scope='raw_bytes',records=rows,scientific_arrays_embedded=False,scope='Persisted artifacts only; no extra live E or factor-table hash'));return target

def run(args):
    started=time.perf_counter();report=Path(args.report).resolve(strict=True);shape=tuple(args.shape)
    require(args.role in ROLES and args.mode in ('QUALIFY_BOUNDARY','PURE_OFF_FULLSIZE'),'NAMED_ROLE_AND_MODE')
    identity=verify_execution(report,local_engineering=args.local_engineering,shape=shape,intervals=args.intervals)
    work=Path(args.work).resolve();require(not work.exists(),'NON_OVERWRITING_CASE');work.mkdir(parents=True)
    from tools.cpu_s4_checkpoint_support import support_module
    sg.frozen_tools(report);support_module('g2_hpc_entry')._quiet_runtime()
    mode='QUALIFY_BOUNDARY' if args.mode=='QUALIFY_BOUNDARY' else 'PURE_OFF_SMALLGRID'
    if args.role=='G3_REF_FULLSIZE':
        require(_cache_unloaded(),'G3_REF_MUST_NOT_IMPORT_CACHE')
        from tools.cpu_s4_3g3_smallgrid.adapter import build_role
        built=build_role('G3_SMALL',mode,capture_root=work/'BOUNDARY',local_engineering=args.local_engineering,pool_override=1 if args.local_engineering else None)
        capacity=dict(resource_scope='LOCAL_ENGINEERING_ONLY' if args.local_engineering else 'FULLSIZE_REFERENCE',factor_cache='NOT_LOADED',smallgrid_default_max_bytes=268435456,fullsize_candidate_max_bytes=1610612736,max_tables=2)
    else:
        from .candidate import build_role
        built=build_role(args.mode,capture_root=work/'BOUNDARY',local_engineering=args.local_engineering,pool_override=1 if args.local_engineering else None,shape=shape)
        empty=built[1].cpu_s4_4b_cache_snapshot()
        require(empty['stats']['factor_requests']==empty['stats']['entries']==empty['stats']['cache_bytes']==0,'NEW_FACTORY_EMPTY_FACTOR_CACHE')
        capacity=empty['fullsize_capacity']
    initialized=time.perf_counter();capture=args.mode=='QUALIFY_BOUNDARY';timings=dict(normal_output_writer_s=0.)
    request=SimpleNamespace(report=report,work=work,role='G3_SMALL',reported_role=args.role,mode=mode,intervals=args.intervals,local_engineering=args.local_engineering,input_manifest=args.input_manifest,reference_output=args.reference_output,reference_boundary=None,reference_stages=args.reference_stages,execution_domain='LOCAL_MICRO_S44B' if args.local_engineering else 'FULLSIZE_CONFIRMATION')
    context_function=sg.context if args.local_engineering else full_context
    result=private_window(context_function,timings)(request,built,label='formal',intervals=args.intervals,capture=capture,application_started=started)
    boundary=work/'formal/BOUNDARY_RECEIPT.json'
    if capture:
        require(built[5].transparency().get('retired') is True and result['boundary']['retained_live_array_refs']==0,'COLLECTOR_RETIRED_AND_REFERENCES_RELEASED')
        exact=support_module('g2_hpc_exact');meta=exact.inspect_boundary_metadata(boundary,require_members=True,header_only=True)
        require(meta['status']=='PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY','BOUNDARY_DISK_METADATA_READBACK')
        if args.reference_boundary:
            result['comparisons']['boundary']=exact.compare_current_boundary_pair(Path(args.reference_boundary),boundary,work/'formal/BOUNDARY_EXACT.jsonl')
            require(result['comparisons']['boundary']['status']=='PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT','ORIGINAL_DISK_BOUNDARY_EXACT')
        sg.write(work/'DISK_BOUNDARY_ROUNDTRIP.json',dict(collector_retired=True,live_array_references=0,both_inputs='ACTUAL_PATHS_AFTER_COLLECTOR_AND_VIEW_CLOSE',metadata=meta,comparison=result['comparisons'].get('boundary')))
    cache=None
    if args.role=='L1_CACHE_FULLSIZE':
        cache=built[1].cpu_s4_4b_cache_snapshot();stats=cache['stats'];cap=268435456 if args.local_engineering else 1610612736
        require(cache['max_bytes']==cap and stats['persistent_bytes_reserved']<=cap and stats['entries']<=2,'ACTUAL_SCOPE_CAPACITY')
        require(stats['factor_requests']==shape[0]*result['route']['nH'] and stats['plane_builds']+stats['hits']==stats['factor_requests'] and stats['misses']==stats['plane_builds'],'ACTUAL_CACHE_REQUEST_BUILD_HIT_COUNTS')
        require(stats['invalidations']==stats['failed_requests']==stats['owned_audit_rejections']==cache['events_dropped']==0,'NO_CACHE_FAILURE_OR_FALLBACK')
        require(cache['owned_object_audit']['status']=='PASS' and stats['owned_audit_samples']==stats['key_misses'],'NEW_TABLE_OWNED_OBJECT_AUDITS')
        sg.write(work/'LINEAR_FACTOR_CACHE_SIDECAR.json',dict(cache=cache,binding=built[1].cpu_s4_4b_binding,science_payload_modified=False,cache_bytes_are_process_peak=False))
        with (work/'LINEAR_FACTOR_CACHE_COUNTERS.jsonl').open('x',encoding='utf-8',newline='\n') as f:
            for event in cache['events']:f.write(json.dumps(event,allow_nan=False)+'\n')
    else:require(_cache_unloaded(),'G3_REF_CACHE_STAYED_UNLOADED')
    import numpy as np
    interval={}
    with np.load(result['normal_outputs']['diagnostics'],allow_pickle=False) as diagnostic:
        for name in ('total_walltime_step_s','linear_walltime_step_s','ionization_walltime_step_s','raman_operator_walltime_step_s'):
            require(name in diagnostic.files and diagnostic[name].shape==(args.intervals,),'ORIGINAL_INTERVAL_TIMING_FIELDS',name);interval[name]=diagnostic[name].tolist()
    posts=[event for event in read(work/'formal/PROPAGATION/state/streaming_manifest.json')['rate_events'] if event.get('event')=='POST_COMMITTED']
    require(len(posts)==args.intervals,'DURABLE_POST_EVENT_COUNT')
    pacing=dict(original_interval_timings=interval,POST_COMMITTED=posts,application_started_monotonic_s=started,application_service_s=result['optical_application_window_s'],propagation_walltime_s=sum(interval['total_walltime_step_s']),propagation_scope='Original step_started to after second BK half; includes original POST hook, excludes following diagnostic/save work',POST_event_definition='Immutable deposited source before the second BK half, not final optical field endpoint')
    sg.write(work/'OPTICAL_PACING_SIDECAR.json',pacing)
    raw_index=array_raw_index(work,result)
    result.update(schema=SCHEMA+'.case.v1',status=('PASS_S44B_MICRO_EXACT' if args.local_engineering else 'PASS_S44B_FULLSIZE_WINDOW_EXACT') if args.reference_output else 'CURRENT_G3_REFERENCE_GENERATED',execution_commit=identity['execution_commit'],execution_domain='S44B_LOCAL_MICRO_ENGINEERING' if args.local_engineering else 'S44B_FULLSIZE_LINEAR_CACHE',role=args.role,application_service_s=result['optical_application_window_s'],initialization_within_application_s=initialized-started,research_service_s=time.perf_counter()-started,propagation_walltime_s=pacing['propagation_walltime_s'],normal_output_writer_s=timings['normal_output_writer_s'],cache=cache,cache_capacity=capacity,L1_executed=args.role=='L1_CACHE_FULLSIZE',cache_fallback=0,warmup_intervals=0,boundary_path=str(boundary) if capture else None,actual_entry_path=str(Path(__file__).resolve()),actual_factory_binding=built[3],default='I0',production_qualified=False,review_gate='MANUAL_REVIEW_PENDING')
    result['all_array_raw_index']=str(raw_index)
    result['mode']=args.mode
    result['requested_mode']=args.mode
    result['internal_legacy_mode']=mode
    result['fullsize_new_candidate']='LOCAL_MICRO_ONLY' if args.local_engineering else ('L1_CACHE_EXECUTED_ENTRY8' if args.role=='L1_CACHE_FULLSIZE' else 'G3_REFERENCE_ENTRY8')
    sg.write(work/'CASE_RECEIPT.json',result);return result

def main():
    unoptimized();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--role',choices=ROLES,required=True);p.add_argument('--mode',choices=('QUALIFY_BOUNDARY','PURE_OFF_FULLSIZE'),required=True)
    p.add_argument('--intervals',type=int,default=8);p.add_argument('--local-engineering',action='store_true');p.add_argument('--micro-shape',choices=('32,17,19','384,17,19'))
    for flag in ('reference-output','reference-boundary','reference-stages','input-manifest'):p.add_argument('--'+flag,type=Path)
    a=p.parse_args();require(not a.micro_shape or a.local_engineering,'MICRO_ONLY_LOCAL');a.shape=FULL_SHAPE if not a.micro_shape else tuple(int(x) for x in a.micro_shape.split(','))
    value=run(a);print(json.dumps({k:value[k] for k in ('status','role','application_service_s')}));return 0

if __name__=='__main__':raise SystemExit(main())
