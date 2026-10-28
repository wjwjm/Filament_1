"""Explicit scratch zero input through the original guarded propagation path."""
import argparse,gc,json,weakref
from pathlib import Path
from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_checkpoint_support import support_module
from .entry import verify_execution,require

def run(report,work,role,reference=None):
    import numpy as np
    verify_execution(report,local_engineering=True,shape=(384,17,19),intervals=2)
    require(role in ('ORIGINAL4A','L1_REF_5C','C_EQ10_V1'),'C_ZERO_FIXED_ROLE')
    work.mkdir(parents=True,exist_ok=False);cases,exact=sg.frozen_tools(report)
    support_module('g2_hpc_entry')._quiet_runtime();ctx=sg.context(report,shape=(384,17,19))
    ctx['source']=np.zeros_like(ctx['source']);ingress=sg.fingerprints(ctx)
    sg.write(work/'SCRATCH_ZERO_INPUT.json',dict(shape=[384,17,19],fingerprints=ingress,frozen_source_modified=False))
    if role=='ORIGINAL4A':
        from tools.cpu_s4_4a_linear_cache.candidate import build_role
        built=build_role('PURE_OFF_SMALLGRID',capture_root=work/'BOUNDARY',local_engineering=True,pool_override=1)
    else:
        from .candidate import build_role
        built=build_role('OFF',role=role,capture_root=work/'BOUNDARY',local_engineering=True,pool_override=1)
    field,kwargs,view=cases.prepare(ctx,work/'PROPAGATION',intervals=2)
    state=built[1].cpu_s4_3g_state;before=state.metrics();start=len(state.r8_state._runtime.get('executions',[]))
    try:
        result=built[0](field,**kwargs);bundle=cases.finish(result,view,work/'PROPAGATION',intervals=2)
    finally:view.close()
    after=state.metrics();route=sg.route('G3_SMALL',built[1],before,after,state.r8_state._runtime.get('executions',[])[start:],local=True,intervals=2)
    require(sg.fingerprints(ctx)==ingress,'C_ZERO_INPUT_IMMUTABLE')
    output=Path(exact.save_tree(work/'OUTPUT',bundle,compressed=True));comparison=None
    if reference:
        comparison=support_module('g2_hpc_exact').compare_full_output(reference,bundle,work/'ZERO_TERMINAL_EXACT.jsonl')
        require(comparison['status']=='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS','C_ZERO_ORIGINAL_FULL_EXACT')
    kernel=None
    if role!='ORIGINAL4A':
        from .candidate import verify_bindings
        verify_bindings(built[1]);executor=built[1].cpu_s45c_executor
        kernel=executor.snapshot() if executor is not None else dict(dispatch_count=0)
        require(kernel['dispatch_count']==0,'C_ACTUAL_ZERO_BRANCH_NO_NEW_DISPATCH')
        sg.write(work/'EQ10_KERNEL_EXECUTIONS.json',kernel)
    tracked=[weakref.ref(value) for value in (result[0],field)]
    del result,bundle,field,kwargs,view,ctx
    gc.collect();lifetime=dict(return_or_ingress_live=[ref() is not None for ref in tracked])
    require(not any(lifetime['return_or_ingress_live']),'C_ZERO_NO_RETURN_OR_INGRESS_RETENTION')
    receipt=dict(status='PASS_ACTUAL_GUARDED_ZERO_SOURCE',role=role,output=str(output),comparison=comparison,
        route=route,lifetime=lifetime,new_dispatches=0,intervals=2,HPC_W48='NOT_RUN_LOCAL')
    sg.write(work/'ZERO_RECEIPT.json',receipt);return receipt
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);parser.add_argument('--work',type=Path,required=True);parser.add_argument('--role',required=True);parser.add_argument('--reference',type=Path)
    args=parser.parse_args();value=run(args.report,args.work,args.role,args.reference);print(json.dumps(dict(status=value['status'],role=value['role'])));return 0
if __name__=='__main__':raise SystemExit(main())
