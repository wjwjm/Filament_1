"""Resume only unexecuted J3 phases after the proven source-receipt field defect."""
from pathlib import Path
import json,subprocess,time,os
from tools.cpu_p1r2 import cpu_p1r2_j3 as j3, cpu_p1r2_j1 as j1
from tools.cpu_p1r2.cpu_p1r2_finalize import require,read,verify_window,require_source_allocation
from tools.hpc_ops.provenance_v2 import raw_sha256_file,validate_manifest

EXPECTED_ERROR='J3Error: reload context source-validation receipt differs from sealed identity'

def validate_resume_source(args):
    src=args.resume_source_root.resolve()
    cmd=['sacct','-X','-nP','-j',args.resume_source_job,'--format=JobIDRaw,State,ExitCode']
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=30,check=True)
    row=next(x.split('|') for x in p.stdout.splitlines() if x.startswith(args.resume_source_job+'|'))
    require(row[1:3]==['FAILED','1:0'],'source must be the terminal FAILED 1:0 infrastructure case')
    require(read(src/'baseline/stages/reload/child/j3_child_failure.json')['error']==EXPECTED_ERROR,'different source failure')
    old=read(src/'run_receipt.json');require(old['status']=='INCOMPLETE' and old['candidate'] is None,'source has unexpected later phases')
    chain=next(s for s in old['baseline']['stages'] if s['stage']=='chain')
    failed=next(s for s in old['baseline']['stages'] if s['stage']=='reload')
    require(chain['status']=='DONE' and chain['supervisor']['exit_code']==0 and failed['status']=='FAIL_CHILD','source stage status mismatch')
    child=Path(chain['child_root']);require(child.resolve()==src/'baseline/stages/chain/child','source child root mismatch')
    require(not (src/'baseline/stages/reload/child/pulse2_reload').exists(),'reload already propagated; not a startup-only retry')
    c=chain['child_receipt'];require_source_allocation(c,args.resume_source_job)
    require(c['lineage']==dict(fresh_source_loads=2,optical_pulses=2,POST_generations=2,hydro_transitions=1,branch='POST1_TO_EIGHT_REAL_HR4_WORKERS_TO_NEXT1'),'source lineage mismatch')
    require(c['hydro_parameters']['n_steps']==1000 and c['hydro']['worker_processes_completed']==8,'incomplete frozen HR4')
    one=verify_window(child/'pulse1',initial_pre0=True);two=verify_window(child/'pulse2_chain',initial_pre0=False)
    require(c['pulse1']['POST_field_sha256']==one['final_POST_canonical_sha256'],'pulse1 POST mismatch')
    j3._load_npz(child/'POST1_checkpoint.npz',expected_raw_sha256=c['pulse1']['POST_raw_sha256'],expected_field_sha256=c['pulse1']['POST_field_sha256'],shape=(8,351,301),label='resumed POST1')
    j3._load_npz(child/'NEXT1_checkpoint.npz',expected_raw_sha256=c['NEXT1']['raw_sha256'],expected_field_sha256=c['NEXT1']['field_canonical_sha256'],shape=(8,351,301),label='resumed NEXT1')
    require(c['hydro']['source_POST_field_sha256']==c['pulse1']['POST_field_sha256'],'NEXT1 not from POST1')
    j3._verify_window_pre_lineage(two,c['NEXT1']['field_canonical_sha256'],'resumed baseline pulse2')
    c['pulse1_window']=one;c['pulse2_window']=two
    return chain,dict(source_job=args.resume_source_job,source_state='FAILED',source_exit_code='1:0',source_error=EXPECTED_ERROR,source_root=str(src),source_receipt_raw_sha256=raw_sha256_file(src/'run_receipt.json'),new_baseline_chain_executions=0,reused_baseline_optical_windows=2,reused_baseline_hydro_transitions=1,preflight_validation='PASS')

def resume_baseline(args,implementation,stage_root,*,candidate_descriptor,deadline):
    require(implementation=='baseline' and candidate_descriptor is None,'resume is baseline only')
    chain=args._verified_source_chain;root=stage_root;root.mkdir(parents=True,exist_ok=False)
    remaining=deadline-time.time();require(remaining>=j3.OPTICAL_WINDOW_ESTIMATE_SECONDS,'insufficient reload budget')
    reload=j3._supervise_child(args,'reload','reload',stage_root=root,timeout_seconds=min(j3.RELOAD_CHILD_TIMEOUT_SECONDS,remaining),extra={'next1_path':Path(chain['child_root'])/'NEXT1_checkpoint.npz','next1_receipt':Path(chain['child_root'])/'NEXT1_checkpoint_receipt.json'})
    comparison=None
    if reload['status']=='DONE':
        c=chain['child_receipt'];r=reload['child_receipt']
        a=j3._verify_window_pre_lineage(c['pulse2_window'],c['NEXT1']['field_canonical_sha256'],'reused chain pulse2')
        b=j3._verify_window_pre_lineage(r['pulse2_window'],r['NEXT1_input']['field_canonical_sha256'],'new disk reload')
        comparison=j1._compare_stage_files(c['pulse2_window'],r['pulse2_window'],'resumed_baseline_chain_vs_reload')
        comparison['same_implementation_PRE_fields_equal']=a==b
        comparison['NEXT1_checkpoint_identity_equal']=c['NEXT1']['raw_sha256']==r['NEXT1_input']['raw_sha256']
        comparison['reload_state_origin_is_disk_NEXT1']=r['NEXT1_input']['state_origin']=='J3_NEXT1_DISK_RELOADED'
        comparison['canonical_exact_match']=bool(comparison['canonical_exact_match'] and all(comparison[k] for k in ('same_implementation_PRE_fields_equal','NEXT1_checkpoint_identity_equal','reload_state_origin_is_disk_NEXT1')))
        j3._write_json_new(root/'chain_vs_reload_comparison.json',comparison)
    exact=bool(comparison and comparison['canonical_exact_match'])
    result={'implementation':'baseline','status':'COMPLETE_DIAGNOSTIC' if exact else 'NUMERICAL_DIFFERENCE' if comparison else 'INCOMPLETE','stages':[chain,reload],'chain_vs_reload_comparison':comparison,'repeat_exact':exact,'rss_80_percent_latched':bool(reload.get('rss_latch_exists') or reload.get('supervisor',{}).get('rss_80_percent_latched')),'reused_chain_from_job':args.resume_source_job,'source_chain_service_is_historical_not_current_job_cost':True}
    j3._write_json_new(root/'implementation_receipt.json',result)
    return result

def main():
    p=j3._parser();p.add_argument('--resume-source-root',type=Path,required=True);p.add_argument('--resume-source-job',required=True);p.add_argument('--resume-preflight-only',action='store_true');args=p.parse_args()
    chain,admission=validate_resume_source(args)
    if args.resume_preflight_only:
        print(json.dumps(admission));return 0
    require(not args.output_root.exists(),'resume output root exists')
    require(os.environ.get('SLURM_JOB_ID') not in (None,args.resume_source_job),'new allocation required')
    args._verified_source_chain=chain
    rc=j3._run_j3(args,baseline_runner=resume_baseline)
    j3._write_json_new(args.output_root/'RESUME_LINEAGE.json',admission)
    return rc
if __name__=='__main__':raise SystemExit(main())