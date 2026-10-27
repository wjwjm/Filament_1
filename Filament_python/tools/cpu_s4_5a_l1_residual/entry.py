"""Single S45A OFF/PROFILE window through the accepted small-grid L1 chain."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_3g3_smallgrid.candidate import unoptimized
from tools.hpc_ops import provenance_v2 as pv

SCHEMA='filament.cpu_s4_5a.entry.v1'
TASK='CPU-S4-5A-L1-SMALLGRID-RESIDUAL-HOTSPOT-SELECTION-20261027'
BASE='c2644d21f7f8fd201b5264d0245c7b4a37643899'
SCIENCE_TREE='821fec5bdacd832b0347e1f2e84285cb54172e57'
SHAPE=(384,117,129)
ROLES=('L1_OFF','L1_PROFILE')

def require(ok,why,context=None):
    if not ok:raise RuntimeError(dict(gate=why,context=context))

def _pairs(items):
    result={}
    for key,value in items:
        require(key not in result,'DUPLICATE_JSON_KEY',key);result[key]=value
    return result

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=_pairs)
read_strict=read

def verify_execution(report,*,local_engineering,shape,intervals,require_node=True):
    unoptimized();report=Path(report).resolve(strict=True)
    binding=read(report/'S45A_EXECUTION_BINDING.json')
    require(binding['schema']=='filament.cpu_s4_5a.execution.v1' and binding['task']==TASK,'S45A_EXECUTION_SCHEMA')
    require(binding['baseline_commit']==BASE and binding['science_tree']==SCIENCE_TREE,'ACCEPTED_L1_SOURCE_CHAIN')
    require(binding['shape']==list(SHAPE) and binding['intervals']==8 and binding['warmup_intervals']==0,'FIXED_SMALLGRID_8_PLAN')
    require(pv.raw_sha256_file(report/'S45A_MEASUREMENT_PLAN.json')==binding['measurement_plan_raw_sha256'],'PRE_DATA_PLAN_RAW_IDENTITY')
    repo=Path(__file__).resolve().parents[3]
    if not local_engineering:
        plan=read(report/'HPC_EXECUTION_PLAN.json')
        require(pv.raw_sha256_file(report/'HPC_EXECUTION_PLAN.json')==os.environ.get('S45A_HPC_PLAN_SHA256'),'HPC_PLAN_RAW_IDENTITY')
        from tools.cpu_s4_checkpoint_support.git_readonly_compat import verify_runtime_query_tool
        verify_runtime_query_tool(plan)
    require(pv._git(repo,'rev-parse','HEAD')==binding['execution_commit'],'REAL_COMMITTED_EXECUTION_HEAD')
    pv._ensure_clean(repo)
    require(pv._git(repo,'rev-parse','HEAD:Filament_python/KHz_filament')==SCIENCE_TREE,'UNCHANGED_SCIENCE_SUBTREE')
    require(isinstance(binding['source_records'],list) and len(binding['source_records'])>=207,'COMPLETE_BOUND_SOURCE_CLOSURE')
    for row in binding['source_records']:
        require(row['classification']=='tracked_text' and row['hash_scope']==pv.TRACKED_HASH_SCOPE,'SOURCE_HASH_SCOPE')
        path=repo/row['path'];require(not Path(row['path']).is_absolute() and path.resolve().is_relative_to(repo),'SOURCE_PATH_CONFINEMENT')
        require(pv._git(repo,'rev-parse','HEAD:'+row['path'])==row['git_blob_oid'] and pv.canonical_lf_sha256(path.read_bytes())==row['canonical_lf_sha256'],'COMMITTED_SOURCE_BLOB_CANONICAL',row['path'])
    if local_engineering:
        require(not os.environ.get('SLURM_JOB_ID') and tuple(shape)==(384,17,19) and intervals in (2,8),'LOCAL_REAL_NT384_MICRO_ONLY')
    else:
        require(os.environ.get('SLURM_JOB_ID','').isdigit() and tuple(shape)==SHAPE and intervals==8,'ACTUAL_SLURM_SMALLGRID_ONLY')
        require(pv.raw_sha256_file(report/'S45A_EXECUTION_BINDING.json')==os.environ.get('S45A_EXECUTION_BINDING_SHA256'),'NODE_EXECUTION_BINDING_RAW_IDENTITY')
        if require_node:
            gate=read(report/'NODE_GATE.json');require(gate['status']=='PASS_S45A_SMALLGRID_SOURCE_RESOURCE_INPUT_GATES' and gate['job_id']==os.environ['SLURM_JOB_ID'],'ACTUAL_NODE_GATE_REQUIRED')
    return binding

def run(args):
    application_started=time.perf_counter();service_started=application_started
    require(args.role in ROLES and args.mode in ('QUALIFY_BOUNDARY','PURE_OFF_SMALLGRID'),'NAMED_INSTRUMENTATION_ROLE_MODE')
    from .timing import TimingRecorder,empty_scope_control_value
    from .candidate import build_role,verify_bindings
    from .measurement import run_window
    timer=TimingRecorder(enabled=args.role=='L1_PROFILE');app_token=timer.begin('APPLICATION')
    try:
        with timer.scope('SOURCE_FACTORY'):
            binding=verify_execution(args.report,local_engineering=args.local_engineering,shape=args.shape,intervals=args.intervals)
            report=Path(args.report).resolve(strict=True);work=Path(args.work).resolve()
            require(not work.exists(),'NON_OVERWRITING_CASE');work.mkdir(parents=True)
            from tools.cpu_s4_checkpoint_support import support_module
            sg.frozen_tools(report);support_module('g2_hpc_entry')._quiet_runtime()
            built=build_role(args.mode,capture_root=work/'BOUNDARY',local_engineering=args.local_engineering,pool_override=1 if args.local_engineering else None,profile=args.role=='L1_PROFILE',recorder=timer)
            require(built[1].cpu_s45a_recorder is timer,'ONE_BOUNDED_RECORDER_FOR_ACTUAL_ROLE')
            empty=built[1].cpu_s4_4a_cache_snapshot()['stats']
            require(empty['factor_requests']==empty['plane_builds']==empty['cache_bytes']==0,'FRESH_EMPTY_ORIGINAL_FACTOR_CACHE')
        initialized=time.perf_counter()
        request=SimpleNamespace(report=report,work=work,role='G3_SMALL',reported_role=args.role,mode=args.mode,intervals=args.intervals,local_engineering=args.local_engineering,input_manifest=args.input_manifest,reference_output=args.reference_output,reference_boundary=None,reference_stages=args.reference_stages)
        result=run_window(request,built,timer=timer,application_token=app_token,label='formal',intervals=args.intervals,capture=args.mode=='QUALIFY_BOUNDARY',application_started=application_started)
        sg.write(work/'INSTRUMENTATION_BINDING_AFTER.json',verify_bindings(built[1].cpu_s45a_state))
    except BaseException:
        # Only scalar frames are unwound; exception/traceback objects are never retained.
        try:
            for unused in range(len(timer._stack)):
                timer.end(timer._stack[-1].token,failed=True)
        except BaseException as cleanup_error:
            timer.retire_incomplete(type(cleanup_error).__name__)
        if 'work' in locals() and work.is_dir():sg.write(work/'INCOMPLETE_TIMING.json',timer.snapshot())
        raise
    boundary=work/'formal/BOUNDARY_RECEIPT.json'
    if args.mode=='QUALIFY_BOUNDARY':
        require(boundary.is_file() and result['boundary']['retained_live_array_refs']==0 and built[5].transparency().get('retired') is True,'ACTUAL_CAPTURE_RETIRED_DISK_BOUNDARY')
        exact=support_module('g2_hpc_exact');metadata=exact.inspect_boundary_metadata(boundary,require_members=True,header_only=True)
        require(metadata['status']=='PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY','ORIGINAL_JSON_BOUNDARY_ROUNDTRIP')
        if args.reference_boundary:
            result['comparisons']['boundary']=exact.compare_current_boundary_pair(Path(args.reference_boundary),boundary,work/'formal/BOUNDARY_EXACT.jsonl')
            require(result['comparisons']['boundary']['status']=='PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT','ORIGINAL_108_EXACT_4_TIMING_BOUNDARY')
        sg.write(work/'DISK_BOUNDARY_ROUNDTRIP.json',dict(path=str(boundary),collector_retired=True,live_array_references=0,metadata=metadata,comparison=result['comparisons'].get('boundary'),both_comparison_inputs='ACTUAL_PATHS_AFTER_COLLECTOR_AND_VIEW_CLOSE'))
    cache=built[1].cpu_s4_4a_cache_snapshot();stats=cache['stats']
    require(stats['factor_requests']==args.shape[0]*result['route']['nH'] and stats['plane_builds']+stats['hits']==stats['factor_requests'],'ORIGINAL_FACTOR_WORKLOAD_ACTUAL_COUNTS')
    require(cache['max_bytes']==268435456 and cache['max_tables']==2 and stats['persistent_bytes_reserved']<=268435456 and cache['events_dropped']==0,'ORIGINAL_SMALLGRID_CACHE_POLICY')
    require(stats['invalidations']==stats['failed_requests']==0,'UNCHANGED_CACHE_ROUTE')
    sg.write(work/'LINEAR_CACHE_SIDECAR.json',dict(cache=cache,producer_binding=built[1].cpu_s4_4a_linear_binding,science_payload_modified=False))
    # Extract only the four pre-existing diagnostic timing vectors.
    import numpy as np
    timings={}
    with np.load(result['normal_outputs']['diagnostics'],allow_pickle=False) as diagnostic:
        for name in ('total_walltime_step_s','linear_walltime_step_s','ionization_walltime_step_s','raman_operator_walltime_step_s'):
            require(name in diagnostic.files,'ORIGINAL_INTERVAL_CLOCK_REQUIRED',name)
            values=diagnostic[name];require(values.shape==(args.intervals,),'ORIGINAL_INTERVAL_TIMING_SHAPE');timings[name]=values.tolist()
    stream=read(work/'formal/PROPAGATION/state/streaming_manifest.json')
    posts=[row for row in stream['rate_events'] if row.get('event')=='POST_COMMITTED'];require(len(posts)==args.intervals,'ORIGINAL_POST_EVENT_COUNT')
    sg.write(work/'OPTICAL_PACING_SIDECAR.json',dict(original_interval_timings=timings,POST_COMMITTED=posts,application_started_monotonic_s=application_started,application_service_s=result['optical_application_window_s'],definition='Immutable deposited source committed before second BK half; not final optical field endpoint'))
    snapshot=timer.snapshot();require(snapshot['events_dropped']==0 and snapshot['clock_closure']['pass'],'BOUNDED_TIMING_TREE_CLOSURE')
    require(snapshot['enabled']==(args.role=='L1_PROFILE'),'HONEST_TIMING_SWITCH')
    require(not snapshot['incomplete_scopes'] and not any(row['failed'] for row in snapshot['records']),'ONLY_COMPLETE_SUCCESSFUL_TIMING_SCOPES')
    if snapshot['enabled']:
        counts={name:row['call_count'] for name,row in snapshot['summary_by_label'].items()}
        expected=dict(APPLICATION=1,SOURCE_FACTORY=1,PROPAGATION=1,OUTPUT=1,DIAGNOSTICS=1,INPUT_PREPARATION=4,
            RAMAN_HALF=2*args.intervals,RAMAN_STAGE=4*args.intervals,RAMAN_INTENSITY=4*args.intervals,
            RAMAN_IIR=4*args.intervals,RAMAN_COEFFICIENTS=8*args.intervals,RAMAN_ENERGY=6*args.intervals,
            HEUN_ASSEMBLY=2*args.intervals,LINEAR=2*args.intervals,NONLINEAR=args.intervals,ION_SETUP=1,
            ION_RECURSION=args.intervals,FFT=4*args.intervals,MATH_KERNELS=12*args.intervals)
        require(all(counts.get(name)==value for name,value in expected.items()),'ACTUAL_TIMING_CATEGORY_WORKLOAD_COUNTS',dict(actual=counts,expected=expected))
        for phase in ('pre','post'):
            for stage in ('stage1','stage2'):
                require(sum(row['label']=='RAMAN_IIR' and row['context']['phase']==phase and row['context']['stage']==stage for row in snapshot['records'])==args.intervals,'ACTUAL_PRE_POST_STAGE1_STAGE2_IIR_COUNTS')
    else:require(not snapshot['records'],'OFF_ADDED_CLOCKS_DISABLED')
    sg.write(work/'SCALAR_TIMING.json',snapshot)
    sg.write(work/'EMPTY_SCOPE_CONTROL.json',empty_scope_control_value())
    original_events=getattr(built[1],'cpu_o1_candidate_events',[])
    sg.write(work/'ORIGINAL_CPU_O1_SCALAR_EVENTS.json',dict(events=original_events,semantics='Original already-enabled scalar subclocks; overlapping with instrumentation tree, never add to it'))
    result.update(schema=SCHEMA+'.case.v1',status='PASS_L1_SMALLGRID_INSTRUMENTATION_EXACT' if args.reference_output else 'LOCAL_L1_OFF_REFERENCE_GENERATED',role=args.role,execution_commit=binding['execution_commit'],execution_domain='S45A_SMALLGRID_L1',application_service_s=result['optical_application_window_s'],initialization_within_application_s=initialized-application_started,research_service_s=time.perf_counter()-service_started,cache=cache,cache_fallback=0,L1_executed=True,boundary_path=str(boundary) if args.mode=='QUALIFY_BOUNDARY' else None,propagation_s=sum(timings['total_walltime_step_s']),profile_enabled=snapshot['enabled'],timing_records=snapshot['completed_scopes'],default='I0',new_math_implemented=False,production_qualified=False,review_gate='MANUAL_REVIEW_PENDING')
    sg.write(work/'CASE_RECEIPT.json',result)
    return result

def main():
    unoptimized();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--role',choices=ROLES,required=True);parser.add_argument('--mode',choices=('QUALIFY_BOUNDARY','PURE_OFF_SMALLGRID'),required=True)
    parser.add_argument('--intervals',type=int,default=8);parser.add_argument('--local-engineering',action='store_true');parser.add_argument('--micro-shape',choices=('384,17,19',))
    for flag in ('reference-output','reference-boundary','reference-stages','input-manifest'):parser.add_argument('--'+flag,type=Path)
    args=parser.parse_args();require(not args.micro_shape or args.local_engineering,'LOCAL_MICRO_ONLY');args.shape=SHAPE if not args.micro_shape else (384,17,19)
    result=run(args);print(json.dumps(dict(status=result['status'],role=result['role'],application_service_s=result['application_service_s'])))
    return 0

if __name__=='__main__':raise SystemExit(main())
