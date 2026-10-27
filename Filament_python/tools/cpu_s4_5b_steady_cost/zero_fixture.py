"""Local Nt384 zero source through the actual guarded propagator; no HPC entry."""
from __future__ import annotations
import argparse,gc,json,sys,weakref
from pathlib import Path
from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_checkpoint_support import support_module

def require(ok,why):
    if not ok:raise RuntimeError(why)

def run(report,work,role,reference=None):
    import numpy as np
    from .entry import verify_execution
    verify_execution(report,local_engineering=True,shape=(384,17,19),intervals=2)
    require(role in ('ORIGINAL4A','L1_S45B_OFF','L1_S45B_PROFILE'),'FIXED_LOCAL_ZERO_ROLE')
    work.mkdir(parents=True,exist_ok=False);cases,exact=sg.frozen_tools(report)
    support_module('g2_hpc_entry')._quiet_runtime()
    ctx=sg.context(report,shape=(384,17,19))
    # Explicit test fixture only, before ownership admission and fingerprints.
    ctx['source']=np.zeros_like(ctx['source'])
    ingress=sg.fingerprints(ctx)
    sg.write(work/'LOCAL_ZERO_INPUT.json',dict(shape=[384,17,19],fingerprints=ingress,kind='SCRATCH_ZERO_SOURCE_BEFORE_ADMISSION',frozen_inputs_modified=False))
    pair=None
    if role=='ORIGINAL4A':
        from tools.cpu_s4_4a_linear_cache.candidate import build_role
        built=build_role('PURE_OFF_SMALLGRID',capture_root=work/'BOUNDARY',local_engineering=True,pool_override=1)
    else:
        from .candidate import build_role,make_recorder,verify_bindings
        pair=make_recorder(role=='L1_S45B_PROFILE');token=pair.b.begin('APPLICATION',common=True)
        built=build_role('PURE_OFF_SMALLGRID',capture_root=work/'BOUNDARY',local_engineering=True,pool_override=1,profile=pair.profile,recorder_pair=pair)
    field,kwargs,view=cases.prepare(ctx,work/'PROPAGATION',intervals=2)
    state=built[1].cpu_s4_3g_state;before=state.metrics();start=len(state.r8_state._runtime.get('executions',[]))
    try:
        result=built[0](field,**kwargs)
        bundle=cases.finish(result,view,work/'PROPAGATION',intervals=2)
    finally:view.close()
    after=state.metrics();route=sg.route('G3_SMALL',built[1],before,after,state.r8_state._runtime.get('executions',[])[start:],local=True,intervals=2)
    require(sg.fingerprints(ctx)==ingress,'ZERO_SOURCE_PRE_IMMUTABLE')
    extended=support_module('g2_hpc_exact')
    output=Path(exact.save_tree(work/'OUTPUT',bundle,compressed=True))
    comparison=None
    if reference:
        comparison=extended.compare_full_output(reference,bundle,work/'ZERO_TERMINAL_EXACT.jsonl')
        require(comparison['status']=='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS','ZERO_ORIGINAL_FULL_EXACT')
    timing=None
    if pair:
        verify_bindings(built[1]);pair.b.end(token)
        sg.write(work/'RAW_A_TIMING.json',pair.s45a.snapshot());sg.write(work/'RAW_B_TIMING.json',pair.b.snapshot())
        timing=pair.snapshot();sg.write(work/'ZERO_SCALAR_TIMING.json',timing)
        markers=timing['b']['markers']
        if pair.profile:
            from collections import Counter
            require(len(markers)==8 and all(row['kind']=='ZERO_INTENSITY' for row in markers),'ACTUAL_ZERO_EQ10_BRANCH_EIGHT_STAGES')
            require({(row['phase'],row['stage']) for row in markers}=={('pre','stage1'),('pre','stage2'),('post','stage1'),('post','stage2')},'ZERO_ACTUAL_PHASE_STAGE')
            require(set(Counter((row['phase'],row['stage']) for row in markers).values())=={2},'ZERO_EACH_PHASE_STAGE_EXACTLY_TWO')
            require(not any(row['label'] in ('EQ10_FFT','EQ10_IFFT','EQ10_SUM') for row in timing['b']['records']),'ZERO_BRANCH_RETAINS_NO_EQ10_FFT_OR_SUM')
    # No timing record may hold the returned scientific arrays alive.
    tracked=[weakref.ref(value) for value in (result[0],field)]
    del result,bundle,field,kwargs,view,ctx
    gc.collect();lifetime=dict(return_or_ingress_live=[ref() is not None for ref in tracked])
    require(not any(lifetime['return_or_ingress_live']),'NO_RETURN_OR_INGRESS_RETAINED_BY_INSTRUMENTATION')
    receipt=dict(status='PASS_ACTUAL_GUARDED_ZERO_SOURCE',role=role,output=str(output),comparison=comparison,route=route,lifetime=lifetime,zero_markers=0 if not pair else len(timing['b']['markers']),HPC_W48='NOT_RUN_LOCAL',intervals=2)
    sg.write(work/'ZERO_RECEIPT.json',receipt);return receipt

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);parser.add_argument('--work',type=Path,required=True);parser.add_argument('--role',required=True);parser.add_argument('--reference',type=Path)
    args=parser.parse_args();value=run(args.report,args.work,args.role,args.reference);print(json.dumps(dict(status=value['status'],role=value['role'])));return 0
if __name__=='__main__':raise SystemExit(main())
