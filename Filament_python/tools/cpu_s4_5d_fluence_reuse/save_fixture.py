"""Local-only actual save-policy and previous public-return lifetime witness."""
import argparse,gc,json,weakref
from pathlib import Path
from .entry import verify_execution,require
from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_checkpoint_support import support_module

def run(args):
    import numpy as np
    verify_execution(args.report,local_engineering=True,shape=(384,17,19),intervals=args.intervals)
    require(args.role in ('ORIGINAL4A','L1_REF_5D','D_FLUENCE_REUSE_V1') and (args.intervals,args.save_every) in ((8,3),(2,8)),'D_FIXED_LOCAL_SAVE_FIXTURE')
    args.work.mkdir(parents=True,exist_ok=False);cases,exact=sg.frozen_tools(args.report)
    extended=support_module('g2_hpc_exact');support_module('g2_hpc_entry')._quiet_runtime()
    expected=[i for i in range(args.intervals) if (i+1)%args.save_every==0 or i==args.intervals-1]
    outputs=[];records=[];held=None;held_path=None;weak=[]
    for repeat in range(2):
        work=args.work/('CALL'+str(repeat));work.mkdir()
        ctx=sg.context(args.report,shape=(384,17,19));ingress=sg.fingerprints(ctx)
        # The original propagator forbids sparse saves when these optional
        # step-aligned diagnostics are enabled. This local-only fixture uses
        # its legal sparse-save domain identically for ORIGINAL4A/A/D.
        ctx['prop'].diag_operator_energy=False
        ctx['prop'].diag_linear_halfstep_energy=False
        if args.role=='ORIGINAL4A':
            from tools.cpu_s4_4a_linear_cache.candidate import build_role
            built=build_role('PURE_OFF_SMALLGRID',capture_root=work/'unused',local_engineering=True,pool_override=1)
        else:
            from .candidate import build_role
            built=build_role('OFF',role=args.role,capture_root=work/'unused',local_engineering=True,pool_override=1)
        require(built[5] is None,'D_LOCAL_OFF_HAS_NO_BOUNDARY_LEASE')
        field,kwargs,view=cases.prepare(ctx,work/'PROPAGATION',intervals=args.intervals)
        kwargs['record_every_z']=args.save_every
        state=built[1].cpu_s4_3g_state;before=state.metrics();start=len(state.r8_state._runtime.get('executions',[]))
        try:
            result=built[0](field,**kwargs)
            require(view.complete,'D_SPARSE_ALL_PRE_READS_COMPLETE')
            from KHz_filament.hr4e5s_streaming import StreamingLifecycle
            lifecycle=StreamingLifecycle.open(view.root)
            require(len(lifecycle.manifest['records'])==args.intervals,'D_SPARSE_ALL_INTERVALS_COMPLETE')
            post={}
            for i in range(args.intervals):
                require(lifecycle.has_authoritative_post(i),'D_SPARSE_EVERY_ORIGINAL_POST_PRESENT')
                post[str(i)]=lifecycle._artifact_fields(lifecycle.manifest['records'][i]['post'],namespace='POST')
            sinks={p.name:np.load(p,allow_pickle=False) for p in sorted((work/'PROPAGATION').glob('sinks.*.npy'))}
            require(len(sinks)>=4 and len(result[2]['z_axis'])==len(expected),'D_SPARSE_EXPECTED_SAVED_DIAGNOSTICS_AND_SINKS')
            require(np.array_equal(result[2]['z_axis'],np.asarray(ctx['schedule'].z_edges)[np.asarray(expected)+1]),'D_SPARSE_PERIODIC_AND_FORCED_FINAL_Z')
            bundle=exact.normalized_return(result,post,sinks,work/'PROPAGATION/RAW_LINEAR_PROFILE_SIDECARS.json')
        finally:view.close()
        require(sg.fingerprints(ctx)==ingress,'D_SPARSE_INPUT_UNCHANGED')
        route=sg.route('G3_SMALL',built[1],before,state.metrics(),state.r8_state._runtime.get('executions',[])[start:],local=True,intervals=args.intervals)
        output=Path(exact.save_tree(work/'OUTPUT',bundle,compressed=True));outputs.append(str(output))
        comparison=None
        if args.reference:
            comparison=extended.compare_full_output(args.reference,bundle,work/'SPARSE_TERMINAL_EXACT.jsonl')
            require(comparison['status']=='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS','D_SPARSE_ORIGINAL_ALL_ELEMENT_EXACT')
        old_comparison=None
        if held is not None:
            old_comparison=extended.compare_full_output(held_path,held,work/'PREVIOUS_RETURN_AFTER_NEXT_CALL_EXACT.jsonl')
            require(old_comparison['status']=='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS','D_PREVIOUS_PUBLIC_RETURN_STABLE_AFTER_NEXT_CALL')
        fluence=None
        if args.role!='ORIGINAL4A':
            from .candidate import verify_bindings
            verify_bindings(built[1]);fluence=built[1].cpu_s45d_fluence_snapshot()
            require(fluence['outer_integrals']==fluence['radius_calls']==fluence['fwhm_calls']==len(expected),'D_ACTUAL_SPARSE_S_COUNTS')
            require(fluence['saved_intervals']==expected,'D_ACTUAL_SPARSE_SOURCE_ORDINALS')
            require(fluence['inner_integrals']==(len(expected) if args.role=='L1_REF_5D' else 0) and fluence['fallback']==fluence['rejected']==0,'D_ACTUAL_SPARSE_ONLY_DUPLICATE_ELIMINATED')
        weak.extend([weakref.ref(result[0]),weakref.ref(field)])
        records.append(dict(repeat=repeat,expected_saved_ordinals=[i+1 for i in expected],comparison=comparison,old_return_comparison=old_comparison,route=route,fluence=fluence,outputs=str(output),factory='FRESH_NEW_INDEPENDENT_FACTORY'))
        held=bundle;held_path=output
        del result,bundle,field,kwargs,view,ctx,built,state
    del held
    gc.collect();lifetime=[ref() is not None for ref in weak]
    require(not any(lifetime),'D_NO_FINAL_OR_INGRESS_RETAINED_BY_PRIVATE_HELPERS')
    receipt=dict(status='PASS_ACTUAL_SPARSE_FINAL_AND_OLD_PUBLIC_RETURNS',role=args.role,intervals=args.intervals,save_every=args.save_every,expected_saved_ordinals=[i+1 for i in expected],outputs=outputs,records=records,lifetime=dict(return_or_ingress_live=lifetime),scientific_comparator='ORIGINAL_UNMODIFIED_ALL_ELEMENT_NATIVE_LAYOUT_ALIAS',full_interval_POST_and_sinks_preserved=True,local_only_legal_sparse_domain=dict(diag_operator_energy=False,diag_linear_halfstep_energy=False),formal_performance_save_or_diagnostics_changed=False)
    sg.write(args.work/'SAVE_FIXTURE_RECEIPT.json',receipt);return receipt

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);parser.add_argument('--work',type=Path,required=True);parser.add_argument('--role',required=True);parser.add_argument('--intervals',type=int,required=True);parser.add_argument('--save-every',type=int,required=True);parser.add_argument('--reference',type=Path)
    value=run(parser.parse_args());print(json.dumps(dict(status=value['status'],role=value['role'])));return 0
if __name__=='__main__':raise SystemExit(main())
