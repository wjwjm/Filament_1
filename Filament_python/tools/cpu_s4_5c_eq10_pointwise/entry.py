"""A real L1/C window, common scalar body clock and original science comparators."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import FunctionType,SimpleNamespace

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_3g3_smallgrid.candidate import unoptimized
from tools.hpc_ops import provenance_v2 as pv

TASK='CPU-S4-5C-EQ10-POINTWISE-AUTO-REAL8-20261028'
BASE='4a5f3355ea2ca27135eb3c67d6de1c768b862a6d'
SCIENCE='821fec5bdacd832b0347e1f2e84285cb54172e57'
ROLES=('L1_REF_5C','C_EQ10_V1')
def require(ok,why,context=None):
    if not ok:raise RuntimeError(dict(gate=why,context=context))
def pairs(items):
    result={}
    for key,value in items:require(key not in result,'DUPLICATE_JSON_KEY',key);result[key]=value
    return result
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=pairs)

def verify_execution(report,*,local_engineering,shape,intervals,require_node=True):
    unoptimized();report=Path(report).resolve(strict=True);repo=Path(__file__).resolve().parents[3]
    binding=read(report/'C_EXECUTION_BINDING.json');plan=read(report/'S45C_EXECUTION_PLAN.json')
    require(set(binding)=={'schema','task','baseline_commit','execution_commit','science_tree','plan_raw_sha256','source_records'},'C_BINDING_EXACT_FIELDS')
    require(binding['schema']=='filament.cpu_s4_5c.execution.v1' and binding['task']==TASK==plan['task'],'C_EXECUTION_SCHEMA')
    require(binding['baseline_commit']==BASE and binding['science_tree']==SCIENCE,'C_ACCEPTED_BASELINE')
    require(pv.raw_sha256_file(report/'S45C_EXECUTION_PLAN.json')==binding['plan_raw_sha256'],'C_FROZEN_PLAN_RAW_IDENTITY')
    require(plan['execution_commit']==binding['execution_commit']==pv._git(repo,'rev-parse','HEAD'),'C_COMMITTED_EXECUTION_HEAD')
    pv._ensure_clean(repo)
    require(pv._git(repo,'rev-parse','HEAD:Filament_python/KHz_filament')==SCIENCE,'C_FROZEN_SCIENCE_TREE')
    for row in binding['source_records']:
        require(row['classification']=='tracked_text' and row['hash_scope']==pv.TRACKED_HASH_SCOPE,'C_SOURCE_RECORD_HASH_SCOPE')
        path=repo/row['path'];require(not Path(row['path']).is_absolute() and path.resolve().is_relative_to(repo),'C_SOURCE_PATH_CONFINEMENT')
        require(pv._git(repo,'rev-parse','HEAD:'+row['path'])==row['git_blob_oid'] and pv.canonical_lf_sha256(path.read_bytes())==row['canonical_lf_sha256'],'C_SOURCE_BLOB_CANONICAL',row['path'])
    if local_engineering:
        require(not os.environ.get('SLURM_JOB_ID') and tuple(shape)==(384,17,19) and intervals in (2,8),'C_LOCAL_MICRO_ONLY')
    else:
        require(os.environ.get('SLURM_JOB_ID','').isdigit() and tuple(shape) in ((384,117,129),(384,351,301)) and intervals==8,'C_HPC_REAL8_ONLY')
        require(pv.raw_sha256_file(report/'C_EXECUTION_BINDING.json')==os.environ.get('S45C_EXECUTION_BINDING_SHA256'),'C_NODE_BINDING_RAW')
        if require_node:
            node=read(report/'NODE_GATE.json')
            require(node['status']=='PASS_S45C_SOURCE_RESOURCE_INPUT_GATES' and node['job_id']==os.environ['SLURM_JOB_ID'],'C_ACTUAL_NODE_GATE')
        if tuple(shape)==(384,351,301):
            promotion=read(report/'SMALLGRID_PROMOTION_GATE.json')
            require(promotion['status']=='PASS' and promotion['execution_commit']==binding['execution_commit'],'C_AUTOMATIC_FULLSIZE_PROMOTION_BOUND_TO_THIS_SHA')
    return binding

def full_context(report,*,shape=(384,351,301)):
    from tools.cpu_s4_fullsize_confirmation import entry as frozen_full
    binding=verify_execution(report,local_engineering=False,shape=shape,intervals=8)
    location=read(Path(report)/'FULLSIZE_SOURCE_LOCATION.json')
    namespace=dict(frozen_full.full_context.__globals__)
    namespace['verify_execution']=lambda unused:dict(location,execution_commit=binding['execution_commit'],science_tree=SCIENCE)
    loader=FunctionType(frozen_full.full_context.__code__,namespace,frozen_full.full_context.__name__,frozen_full.full_context.__defaults__,frozen_full.full_context.__closure__)
    loader.__kwdefaults__=frozen_full.full_context.__kwdefaults__
    return loader(report,shape=shape)

def run(args):
    started=time.perf_counter();cpu_started=time.process_time();research_started=started
    require(args.role in ROLES and args.mode in ('QUALIFY_BOUNDARY','OFF'),'C_ROLE_MODE')
    binding=verify_execution(args.report,local_engineering=args.local_engineering,shape=args.shape,intervals=args.intervals)
    report=Path(args.report).resolve(strict=True);work=Path(args.work).resolve()
    require(not work.exists(),'C_NON_OVERWRITING_CASE');work.mkdir(parents=True)
    from tools.cpu_s4_checkpoint_support import support_module
    sg.frozen_tools(report);support_module('g2_hpc_entry')._quiet_runtime()
    from .candidate import build_role,verify_bindings
    built=build_role(args.mode,role=args.role,capture_root=work/'BOUNDARY',local_engineering=args.local_engineering,
        pool_override=1 if args.local_engineering else None,fullsize=tuple(args.shape)==(384,351,301))
    empty=built[1].cpu_s4_4a_cache_snapshot()['stats']
    require(empty['factor_requests']==empty['plane_builds']==empty['cache_bytes']==0,'C_FRESH_EMPTY_FACTOR_CACHE')
    initialized=time.perf_counter()
    request=SimpleNamespace(report=report,work=work,role='G3_SMALL',reported_role=args.role,mode='QUALIFY_BOUNDARY' if args.mode=='QUALIFY_BOUNDARY' else 'PURE_OFF_SMALLGRID',intervals=args.intervals,local_engineering=args.local_engineering,input_manifest=args.input_manifest,reference_output=args.reference_output,reference_boundary=None,reference_stages=args.reference_stages)
    if tuple(args.shape)==(384,351,301):
        from tools.cpu_s4_fullsize_confirmation.entry import full_window
        request.execution_domain='FULLSIZE_CONFIRMATION'
        runner=full_window(full_context)
    else:runner=sg.run_window
    result=runner(request,built,label='formal',intervals=args.intervals,capture=args.mode=='QUALIFY_BOUNDARY',application_started=started)
    sg.write(work/'C_BINDING_AFTER.json',verify_bindings(built[1]))
    boundary=work/'formal/BOUNDARY_RECEIPT.json'
    if args.mode=='QUALIFY_BOUNDARY':
        require(boundary.is_file() and result['boundary']['retained_live_array_refs']==0 and built[5].transparency().get('retired') is True,'C_CAPTURE_RETIRED_DISK_BOUNDARY')
        exact=support_module('g2_hpc_exact');metadata=exact.inspect_boundary_metadata(boundary,require_members=True,header_only=True)
        require(metadata['status']=='PASS_BOUNDARY_JSON_METADATA_COMPATIBILITY','C_ORIGINAL_JSON_DISK_ROUNDTRIP')
        if args.reference_boundary:
            result['comparisons']['boundary']=exact.compare_current_boundary_pair(Path(args.reference_boundary),boundary,work/'formal/BOUNDARY_EXACT.jsonl')
            require(result['comparisons']['boundary']['status']=='PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT','C_ORIGINAL_BOUNDARY_108_PLUS_4')
        sg.write(work/'DISK_BOUNDARY_ROUNDTRIP.json',dict(path=str(boundary),collector_retired=True,live_array_references=0,metadata=metadata,comparison=result['comparisons'].get('boundary'),both_comparison_inputs='ACTUAL_PATHS_AFTER_COLLECTOR_RETIREMENT_AND_VIEW_CLOSE'))
    cache=built[1].cpu_s4_4a_cache_snapshot();stats=cache['stats']
    cap=1610612736 if tuple(args.shape)==(384,351,301) else 268435456
    require(cache['max_bytes']==cap and cache['max_tables']==2 and stats['factor_requests']==384*result['route']['nH'],'C_SAME_FACTOR_POLICY_AND_REQUESTS')
    require(stats['plane_builds']+stats['hits']==stats['factor_requests'] and stats['failed_requests']==stats['invalidations']==0 and cache['events_dropped']==0,'C_FACTOR_ACCOUNTING_NO_FALLBACK')
    sg.write(work/'LINEAR_CACHE_SIDECAR.json',dict(cache=cache,producer_binding=built[1].cpu_s4_4a_linear_binding))
    clock=built[1].cpu_s45c_body_clock.snapshot()
    require(len(clock['rows'])==args.intervals and all(not row['failed'] and row['wall_ns']>=0 and row['cpu_ns']>=0 for row in clock['rows']),'C_COMPLETE_ACTUAL_BODY_CLOCKS')
    sg.write(work/'INTERVAL_BODY_CLOCKS.json',clock)
    executor=built[1].cpu_s45c_executor
    kernel=executor.snapshot() if executor is not None else dict(dispatch_count=0,dispatches=[],role=args.role,candidate_active=False)
    expected=4*args.intervals if executor is not None else 0
    require(kernel['dispatch_count']==2*expected,'C_NEW_DISPATCH_COUNT')
    for name in ('k1','k2'):
        rows=[row for row in kernel['dispatches'] if row['operation']==name]
        require(len(rows)==expected,'C_PER_KERNEL_ACTUAL_COUNT',name)
        if executor:
            require(sum(row['interval_index']>=1 for row in rows)==4*(args.intervals-1),'C_LATER_NEW_DISPATCH_COUNT',name)
            for row in rows:
                workers=1 if args.local_engineering else 48
                require(row['status']=='PASS' and row['requested_workers']==workers and row['active_thread_ids']==list(range(workers)),'C_ACTUAL_WORKER_PARTICIPATION',row)
                require(row['thread_mask_after']==row['thread_mask_before'] and row['thread_mask_inside']==workers,'C_NEW_MASK_RESTORE',row)
    sg.write(work/'EQ10_KERNEL_EXECUTIONS.json',kernel)
    import numpy as np
    timings={}
    with np.load(result['normal_outputs']['diagnostics'],allow_pickle=False) as diagnostic:
        for name in ('total_walltime_step_s','linear_walltime_step_s','ionization_walltime_step_s','raman_operator_walltime_step_s'):
            values=diagnostic[name];require(values.shape==(args.intervals,),'C_ORIGINAL_STEP_CLOCK_SHAPE');timings[name]=values.tolist()
    stream=read(work/'formal/PROPAGATION/state/streaming_manifest.json')
    posts=[row for row in stream['rate_events'] if row.get('event')=='POST_COMMITTED'];require(len(posts)==args.intervals,'C_REAL_POST_COUNT')
    sg.write(work/'OPTICAL_PACING_SIDECAR.json',dict(original_interval_timings=timings,POST_COMMITTED=posts,application_started_monotonic_s=started,application_service_s=result['optical_application_window_s'],definition='Original immutable deposited-source commit before second BK; seven actual gaps'))
    result.update(schema='filament.cpu_s4_5c.case.v1',status='PASS_CURRENT_C_L1_NATIVE_EXACT' if args.reference_output else 'CURRENT_L1_REFERENCE_GENERATED',role=args.role,execution_commit=binding['execution_commit'],application_service_s=result['optical_application_window_s'],initialization_within_application_s=initialized-started,research_service_s=time.perf_counter()-research_started,process_CPU_through_research_s=time.process_time()-cpu_started,propagation_s=sum(timings['total_walltime_step_s']),first_body_s=clock['first_body_s'],later_body_s=clock['later_body_s'],new_kernel_dispatches=kernel['dispatch_count'],boundary_path=str(boundary) if args.mode=='QUALIFY_BOUNDARY' else None,cache=cache,cache_fallback=0,L1_executed=True,candidate_executed=executor is not None,default='I0',production_qualified=False,review_gate='MANUAL_REVIEW_PENDING')
    sg.write(work/'CASE_RECEIPT.json',result);return result

def main():
    unoptimized();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--role',choices=ROLES,required=True);parser.add_argument('--mode',choices=('QUALIFY_BOUNDARY','OFF'),required=True)
    parser.add_argument('--intervals',type=int,default=8);parser.add_argument('--local-engineering',action='store_true');parser.add_argument('--micro-shape',choices=('384,17,19',));parser.add_argument('--fullsize',action='store_true')
    for flag in ('reference-output','reference-boundary','reference-stages','input-manifest'):parser.add_argument('--'+flag,type=Path)
    args=parser.parse_args();require(not args.micro_shape or args.local_engineering,'C_MICRO_LOCAL_ONLY')
    require(not(args.fullsize and args.local_engineering),'C_NO_LOCAL_FULLSIZE')
    args.shape=(384,17,19) if args.micro_shape else ((384,351,301) if args.fullsize else (384,117,129))
    result=run(args);print(json.dumps(dict(status=result['status'],role=args.role,application_service_s=result['application_service_s'])));return 0
if __name__=='__main__':raise SystemExit(main())
