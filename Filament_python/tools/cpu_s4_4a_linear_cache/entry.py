"""One source-bound G3/L1 optical window, using the original SG science path."""
from __future__ import annotations
import argparse,copy,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_3g3_smallgrid.candidate import unoptimized
from tools.cpu_s4_3g3_smallgrid.scientific_context import SHAPE,MICRO_SHAPES
from tools.hpc_ops import provenance_v2 as pv

SCHEMA='filament.cpu_s4_4a.entry.v1'
BINDING_SCHEMA='filament.cpu_s4_4a.execution.v1'
SCIENCE_TREE='821fec5bdacd832b0347e1f2e84285cb54172e57'
ROLES=('G3_REF','L1_LINEAR_FACTOR_CACHE')
BINDING_KEYS={'schema','task','baseline_commit','execution_commit','science_tree','shape','intervals','warmup_intervals','experiment_plan_raw_sha256','source_records'}

def require(ok,reason,context=None):
    if not ok:raise RuntimeError({'gate':reason,'context':context})

def _pairs(items):
    result={}
    for key,value in items:
        require(key not in result,'DUPLICATE_JSON_KEY',key);result[key]=value
    return result

def read_strict(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=_pairs)

def verify_execution(report,*,local_engineering,shape,intervals,require_node=True):
    unoptimized();report=Path(report).resolve(strict=True)
    binding=read_strict(report/'EXECUTION_BINDING.json')
    require(isinstance(binding,dict) and set(binding)==BINDING_KEYS,'EXECUTION_BINDING_FIELDS')
    require(binding['schema']==BINDING_SCHEMA and binding['task']=='CPU-S4-4A-SMALLGRID-LINEAR-FACTOR-REUSE-20261026','EXECUTION_BINDING_SCHEMA')
    require(binding['shape']==list(SHAPE) and binding['intervals']==8 and binding['warmup_intervals']==0,'FIXED_SMALLGRID_PLAN_DOMAIN')
    require(binding['science_tree']==SCIENCE_TREE and binding['baseline_commit']=='41d0957eff04280b74fef839dc6a8d4676e2316a','ACCEPTED_BASELINE_IDENTITY')
    require(pv.raw_sha256_file(report/'EXPERIMENT_PLAN.json')==binding['experiment_plan_raw_sha256'],'EXPERIMENT_PLAN_RAW_IDENTITY')
    if not local_engineering:
        hpc=read_strict(report/'HPC_EXECUTION_PLAN.json')
        require(pv.raw_sha256_file(report/'HPC_EXECUTION_PLAN.json')==os.environ.get('S44A_HPC_PLAN_SHA256'),'HPC_PLAN_RAW_IDENTITY')
        from tools.cpu_s4_checkpoint_support.git_readonly_compat import verify_runtime_query_tool
        verify_runtime_query_tool(hpc)
    repo=Path(__file__).resolve().parents[3]
    require(pv._git(repo,'rev-parse','HEAD')==binding['execution_commit'],'ACTUAL_COMMITTED_EXECUTION_HEAD')
    pv._ensure_clean(repo)
    require(pv._git(repo,'rev-parse','HEAD:Filament_python/KHz_filament')==SCIENCE_TREE,'UNCHANGED_SCIENCE_TREE')
    require(isinstance(binding['source_records'],list) and binding['source_records'],'BOUND_EXECUTION_SOURCE_RECORDS')
    for row in binding['source_records']:
        require(row.get('classification')=='tracked_text' and row.get('hash_scope')==pv.TRACKED_HASH_SCOPE,'SOURCE_RECORD_HASH_SCOPE')
        relative=row['path'];path=repo/relative
        require(path.resolve().is_relative_to(repo) and not Path(relative).is_absolute(),'SOURCE_PATH_CONFINEMENT')
        require(pv._git(repo,'rev-parse','HEAD:'+relative)==row['git_blob_oid'] and pv.canonical_lf_sha256(path.read_bytes())==row['canonical_lf_sha256'],'COMMITTED_SOURCE_BLOB_AND_CANONICAL_IDENTITY',relative)
    if local_engineering:
        require(not os.environ.get('SLURM_JOB_ID') and tuple(shape) in MICRO_SHAPES and intervals in (2,8),'LOCAL_MICRO_DOMAIN_ONLY')
    else:
        require(os.environ.get('SLURM_JOB_ID','').isdigit() and tuple(shape)==SHAPE and intervals==8,'ACTUAL_SLURM_FIXED_SMALLGRID_ONLY')
        expected=os.environ.get('S44A_EXECUTION_BINDING_SHA256')
        require(expected and pv.raw_sha256_file(report/'EXECUTION_BINDING.json')==expected,'ACTUAL_HPC_EXECUTION_BINDING_RAW_IDENTITY')
        if require_node:
            node=read_strict(report/'NODE_GATE.json')
            require(node.get('status')=='PASS_SMALLGRID_LINEAR_CACHE_NODE_GATE' and node.get('job_id')==os.environ['SLURM_JOB_ID'],'ACTUAL_NODE_GATE_REQUIRED')
    return binding

def run(args):
    application_started=time.perf_counter();service_started=application_started
    require(args.role in ROLES and args.mode in ('QUALIFY_BOUNDARY','PURE_OFF_SMALLGRID'),'NAMED_ROLE_AND_MODE')
    shape=tuple(getattr(args,'shape',SHAPE))
    binding=verify_execution(args.report,local_engineering=args.local_engineering,shape=shape,intervals=args.intervals)
    report=Path(args.report).resolve(strict=True);work=Path(args.work).resolve()
    require(not work.exists(),'NON_OVERWRITING_CASE');work.mkdir(parents=True)
    from tools.cpu_s4_checkpoint_support import support_module
    sg.frozen_tools(report);support_module('g2_hpc_entry')._quiet_runtime()
    if args.role=='G3_REF':
        require(not any(name.startswith('tools.cpu_s4_4a_linear_cache.cache') or name.startswith('tools.cpu_s4_4a_linear_cache.linear_adapter') or name.startswith('tools.cpu_s4_4a_linear_cache.candidate') for name in sys.modules),'G3_REF_MUST_NOT_LOAD_CACHE_MODULES')
        from tools.cpu_s4_3g3_smallgrid.adapter import build_role
        built=build_role('G3_SMALL',args.mode,capture_root=work/'BOUNDARY',local_engineering=args.local_engineering,pool_override=1 if args.local_engineering else None)
    else:
        from .candidate import build_role
        built=build_role(args.mode,capture_root=work/'BOUNDARY',local_engineering=args.local_engineering,pool_override=1 if args.local_engineering else None)
        empty=built[1].cpu_s4_4a_cache_snapshot()['stats']
        require(empty['factor_requests']==empty['plane_builds']==empty['cache_bytes']==0,'INDEPENDENT_CASE_EMPTY_FACTOR_CACHE')
    initialized=time.perf_counter()
    request=SimpleNamespace(report=report,work=work,role='G3_SMALL',reported_role=args.role,mode=args.mode,intervals=args.intervals,local_engineering=args.local_engineering,input_manifest=args.input_manifest,reference_output=args.reference_output,reference_boundary=None,reference_stages=args.reference_stages)
    capture=args.mode=='QUALIFY_BOUNDARY'
    result=sg.run_window(request,built,label='formal',intervals=args.intervals,capture=capture,application_started=application_started)
    boundary_path=work/'formal/BOUNDARY_RECEIPT.json'
    if capture:
        require(boundary_path.is_file() and result['boundary']['retained_live_array_refs']==0 and built[5].transparency().get('retired') is True,'ACTUAL_COLLECTOR_RETIRED_DISK_BOUNDARY')
        exact=support_module('g2_hpc_exact')
        metadata=exact.inspect_boundary_metadata(boundary_path,require_members=True,header_only=True)
        require(metadata['status']=='PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY','CURRENT_BOUNDARY_METADATA_READBACK')
        if args.reference_boundary:
            result['comparisons']['boundary']=exact.compare_current_boundary_pair(Path(args.reference_boundary),boundary_path,work/'formal/BOUNDARY_EXACT.jsonl')
            require(result['comparisons']['boundary']['status']=='PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT','ORIGINAL_DISK_BOUNDARY_EXACT')
        sg.write(work/'DISK_BOUNDARY_ROUNDTRIP.json',dict(schema=SCHEMA+'.disk_boundary.v1',path=str(boundary_path),collector_retired=True,live_array_references=0,metadata=metadata,comparison=result['comparisons'].get('boundary'),both_comparison_inputs='ACTUAL_PATHS_AFTER_COLLECTOR_AND_VIEW_CLOSE',scientific_metadata_gate_alone='NOT_RUN_METADATA_ONLY'))
    cache=None
    if args.role=='L1_LINEAR_FACTOR_CACHE':
        cache=built[1].cpu_s4_4a_cache_snapshot();stats=cache['stats']
        require(stats['factor_requests']==shape[0]*result['route']['nH'],'REAL_FACTOR_REQUEST_COUNT')
        require(stats['plane_builds']+stats['hits']==stats['factor_requests'] and stats['misses']==stats['plane_builds'],'REAL_BUILD_HIT_ACCOUNTING')
        require(stats['persistent_bytes_reserved']<=256*1024**2 and stats['entries']<=2,'BOUNDED_FACTOR_CACHE')
        require(cache['events_dropped']==0,'COMPLETE_BOUNDED_CACHE_EVENT_LEDGER')
        sg.write(work/'LINEAR_CACHE_SIDECAR.json',dict(cache=cache,producer_binding=built[1].cpu_s4_4a_linear_binding,science_payload_modified=False,cache_bytes_are_process_peak=False))
        with (work/'CACHE_REQUEST_BUILD_HIT_LEDGER.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            for event in cache['events']:stream.write(json.dumps(event,allow_nan=False)+'\n')
    else:
        require(not any(name.startswith('tools.cpu_s4_4a_linear_cache.cache') for name in sys.modules),'G3_REF_CACHE_STAYED_UNLOADED')
    # Re-encode only existing small timing arrays; the scientific output is untouched.
    import numpy as np
    timings={}
    with np.load(result['normal_outputs']['diagnostics'],allow_pickle=False) as diagnostic_file:
        for name in ('total_walltime_step_s','linear_walltime_step_s','ionization_walltime_step_s','raman_operator_walltime_step_s'):
            if name in diagnostic_file.files:
                values=diagnostic_file[name]
                require(values.ndim==1 and values.size==args.intervals,'ORIGINAL_INTERVAL_TIMING_SHAPE',name)
                timings[name]=values.tolist()
    stream=read_strict(work/'formal/PROPAGATION/state/streaming_manifest.json')
    posts=[event for event in stream['rate_events'] if event.get('event')=='POST_COMMITTED']
    require(len(posts)==args.intervals,'ORIGINAL_POST_COMMIT_EVENT_COUNT')
    sg.write(work/'OPTICAL_PACING_SIDECAR.json',dict(original_interval_timings=timings,POST_COMMITTED=posts,application_started_monotonic_s=application_started,application_service_s=result['optical_application_window_s'],native_POST_event_definition='Frozen post_commit_hook emits immutable deposited source before the second BK half-step; this is not the final optical field endpoint',timing_extraction='Existing diagnostic arrays and streaming rate_events only; no propagation or Hydro re-execution'))
    result.update(schema=SCHEMA+'.case.v1',status='PASS_SMALLGRID_WINDOW_EXACT' if args.reference_output else 'G3_REF_SAME_GRID_STRICT_REFERENCE_GENERATED',role=args.role,execution_commit=binding['execution_commit'],execution_domain='SMALLGRID_LINEAR_FACTOR_REUSE',application_service_s=result['optical_application_window_s'],initialization_within_application_s=initialized-application_started,research_service_s=time.perf_counter()-service_started,warmup_intervals=0,cache=cache,cache_fallback=0,L1_executed=args.role=='L1_LINEAR_FACTOR_CACHE',boundary_path=str(boundary_path) if capture else None,default='I0',fullsize_new_cache='NOT_RUN',production_qualified=False,review_gate='MANUAL_REVIEW_PENDING')
    sg.write(work/'CASE_RECEIPT.json',result)
    return result

def main():
    unoptimized();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--role',choices=ROLES,required=True);parser.add_argument('--mode',choices=('QUALIFY_BOUNDARY','PURE_OFF_SMALLGRID'),required=True)
    parser.add_argument('--intervals',type=int,default=8);parser.add_argument('--local-engineering',action='store_true')
    parser.add_argument('--micro-shape',choices=('32,17,19','384,17,19'))
    for flag in ('reference-output','reference-boundary','reference-stages','input-manifest'):parser.add_argument('--'+flag,type=Path)
    args=parser.parse_args();require(not args.micro_shape or args.local_engineering,'MICRO_SHAPE_ONLY_LOCAL');args.shape=SHAPE if not args.micro_shape else tuple(int(value) for value in args.micro_shape.split(','))
    result=run(args);print(json.dumps(dict(status=result['status'],role=result['role'],application_service_s=result['application_service_s'])))
    return 0

if __name__=='__main__':raise SystemExit(main())
