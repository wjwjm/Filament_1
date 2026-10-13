"""One-shot read-only recovery of the known J3 pulse1_window aggregation defect."""
from pathlib import Path
import argparse, json, os, subprocess, time, traceback
import numpy as np
from tools.cpu_p1r2 import cpu_p1r2_j1 as j1, cpu_p1r2_j3 as j3
from tools.cpu_p1r import cpu_p1r_optical as p1r
from tools.hpc_ops.provenance_v2 import raw_sha256_file, validate_manifest

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)

def require_source_allocation(receipt, job):
    require(receipt.get('allocation',{}).get('SLURM_JOB_ID')==job,'source JobID does not match persisted allocation')

def verify_window(root, *, expected_shape=(384,351,301), production=True, initial_pre0=None):
    root=Path(root); receipt=read(root/'child_receipt.json'); provenance=read(root/'j3_provenance_receipt.json')
    require(receipt.get('status')=='COMPLETE_DIAGNOSTIC' and receipt.get('interval_count')==8,'incomplete optical window')
    require([r['interval_index'] for r in receipt['interval_rows']]==list(range(8)),'interval sequence mismatch')
    if production:
        require(receipt.get('execution_scope')=='J3_REAL_INPUT','not real J3 input')
        require(receipt['source_canonical_sha256']==p1r.SOURCE_ARRAY_SHA256,'source changed')
        require(provenance['source_raw_sha256']==p1r.SOURCE_RAW_SHA256 and provenance['transfer_mapping_sha256']==p1r.MAPPING_SHA256,'source mapping changed')
        require(isinstance(initial_pre0,bool),'caller must declare initial PRE0 role')
        require('pre0_expected_canonical_sha256' in receipt,'missing PRE0 contract field')
        if initial_pre0:
            require(receipt['pre0_expected_canonical_sha256']==receipt.get('pre0_canonical_sha256')==p1r.PRE0_ARRAY_SHA256,'frozen PRE0 identity mismatch')
            require(provenance.get('pre0_raw_sha256')==p1r.PRE0_RAW_SHA256,'frozen PRE0 raw identity mismatch')
            zeros=j1._canonical_hash(np.zeros((8,*expected_shape[-2:]),dtype=np.float64))
            require(receipt['input_PRE_fields_canonical_sha256']==dict(delta_n=p1r.PRE0_ARRAY_SHA256,vx=zeros,vy=zeros),'frozen PRE0 fields mismatch')
        else:
            require(receipt.get('pre0_expected_canonical_sha256') is None and receipt.get('pre0_canonical_sha256') is None and provenance.get('pre0_raw_sha256') is None,'non-PRE0 contract mismatch')
    require(provenance['source_canonical_sha256']==receipt['source_canonical_sha256'],'source provenance mismatch')
    require(provenance['input_PRE_fields_canonical_sha256']==receipt['input_PRE_fields_canonical_sha256'],'PRE provenance mismatch')
    wrapper={'status':'VALIDATED','receipt':receipt}
    for name,filename in [('final_E','final_E.npy'),('final_POST','final_POST.npz'),('diagnostics','diagnostics.npz')]:
        path=root/filename; raw=raw_sha256_file(path)
        require(raw==receipt[name+'_raw_sha256']==provenance['output_raw_sha256'][name],name+' raw hash mismatch')
        if name=='final_E':
            arr=np.load(path,mmap_mode='r',allow_pickle=False)
            canonical=p1r.verify_array_contract(arr,shape=expected_shape,dtype='complex128',expected_hash=receipt[name+'_canonical_sha256'],label='recovery E')
            del arr
        else:
            with np.load(path,allow_pickle=False) as z:
                if name=='final_POST':
                    require(set(z.files)==set(j3.FIELDS),'POST fields mismatch')
                    canonical={k:p1r.verify_array_contract(z[k],shape=(8,*expected_shape[-2:]),dtype='float64',expected_hash=receipt[name+'_canonical_sha256'][k],label='recovery POST '+k) for k in z.files}
                else:
                    canonical={k:j1._canonical_hash(z[k]) for k in z.files}
            require(canonical==receipt[name+'_canonical_sha256'],name+' canonical mismatch')
        wrapper[name+'_path']=str(path);wrapper[name+'_raw_sha256']=raw;wrapper[name+'_canonical_sha256']=canonical
    return wrapper

def recover(args):
    started=time.perf_counter();cpu_started=time.process_time();src=args.source_root.resolve();out=args.output_root.resolve()
    require(src.is_dir() and not out.exists() and src!=out and not out.is_relative_to(src),'invalid recovery roots')
    require(os.environ.get('SLURM_JOB_ID','').isdigit(),'allocation required')
    require(os.environ['SLURM_JOB_ID']!=args.source_job,'source allocation still in use')
    cmd=['sacct','-X','-nP','-j',args.source_job,'--format=JobIDRaw,State,ExitCode']
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=30,check=True)
    row=next(x.split('|') for x in p.stdout.splitlines() if x.startswith(args.source_job+'|'))
    require(row[1:3]==['FAILED','2:0'],'recovery accepts only the expected terminal FAILED 2:0')
    failure=read(src/'coordinator_failure.json')
    require(failure.get('error')=="KeyError: 'pulse1_window'",'different failure cannot use this recovery')
    old_repo=args.source_repo.resolve()
    require(subprocess.check_output(['git','-C',str(old_repo),'rev-parse','HEAD'],text=True).strip()=='943a899ed9817fe7b3175f3abe3e11d72106ebe1','wrong original execution commit')
    validate_manifest(old_repo,args.source_manifest,require_hash_scope=True)
    source_inventory={str(path.relative_to(src)):{'bytes':path.stat().st_size,'raw_sha256':raw_sha256_file(path)} for path in sorted(src.rglob('*')) if path.is_file()}
    out.mkdir(parents=True,exist_ok=False)
    write(out/'SOURCE_TERMINAL_AND_FAILURE.json',{'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'failure':failure,'original_job_status_preserved':'FAILED 2:0'})
    implementations={};windows={};next_fields={}
    for name in ('baseline','candidate'):
        impl=read(src/name/'implementation_receipt.json')
        require(impl['status']=='COMPLETE_DIAGNOSTIC' and impl['repeat_exact'] is True and not impl['rss_80_percent_latched'],name+' not complete and repeat exact')
        stages={s['stage']:s for s in impl['stages']}
        for stage in ('chain','reload'):
            s=stages[stage];require(s['status']=='DONE' and s['supervisor']['exit_code']==0 and not s['supervisor'].get('stop'),name+' stage invalid')
            require(Path(s['child_root']).resolve()==src/name/'stages'/stage/'child','unexpected child path')
        chain=stages['chain']['child_receipt'];reload=stages['reload']['child_receipt']
        require_source_allocation(chain,args.source_job);require_source_allocation(reload,args.source_job)
        require(chain['lineage']['fresh_source_loads']==2 and chain['lineage']['optical_pulses']==2 and chain['lineage']['hydro_transitions']==1,'invalid primary lineage')
        require(chain['hydro']['worker_processes_completed']==8,'incomplete HR4')
        for stage,suffix,key in [('chain','pulse1','pulse1'),('chain','pulse2_chain','pulse2'),('reload','pulse2_reload','reload')]:
            windows[(name,key)]=verify_window(Path(stages[stage]['child_root'])/suffix,initial_pre0=(key=='pulse1'))
        # Reconstruct in memory only; original receipts remain byte-for-byte unchanged.
        chain['pulse1_window']=windows[(name,'pulse1')]
        require(chain['pulse1']['POST_field_sha256']==windows[(name,'pulse1')]['final_POST_canonical_sha256'],'POST1 provenance mismatch')
        next_receipt=read(Path(stages['chain']['child_root'])/'NEXT1_checkpoint_receipt.json')
        require(next_receipt['source_POST_field_sha256']==chain['pulse1']['POST_field_sha256'],'Hydro not from this POST1')
        j3._load_npz(Path(stages['chain']['child_root'])/'NEXT1_checkpoint.npz',expected_raw_sha256=next_receipt['NEXT1_raw_sha256'],expected_field_sha256=next_receipt['NEXT1_field_sha256'],shape=(8,351,301),label='recovery NEXT1')
        next_fields[name]=next_receipt['NEXT1_field_sha256']
        j3._load_npz(Path(stages['chain']['child_root'])/'POST1_checkpoint.npz',expected_raw_sha256=chain['pulse1']['POST_raw_sha256'],expected_field_sha256=chain['pulse1']['POST_field_sha256'],shape=(8,351,301),label='recovery POST1')
        a=j3._verify_window_pre_lineage(windows[(name,'pulse2')],next_receipt['NEXT1_field_sha256'],name+' memory NEXT1')
        b=j3._verify_window_pre_lineage(windows[(name,'reload')],reload['NEXT1_input']['field_canonical_sha256'],name+' disk NEXT1')
        require(a==b and reload['NEXT1_input']['raw_sha256']==next_receipt['NEXT1_raw_sha256'],'reload checkpoint identity mismatch')
        repeat=j1._compare_stage_files(windows[(name,'pulse2')],windows[(name,'reload')],name+'_reverified_reload')
        require(repeat['canonical_exact_match'],name+' reload not exact')
        write(out/(name+'_reload_reverified.json'),repeat)
        implementations[name]=impl
    cross={'pulse1':j3._cross_pulse1_compare(windows[('baseline','pulse1')],windows[('candidate','pulse1')],'recovered_baseline_vs_candidate_pulse1')}
    for key in ('pulse2','reload'):
        left=windows[('baseline',key)];right=windows[('candidate',key)]
        cross[key]=j3._cross_lineaged_pulse2_compare(left,right,next_fields['baseline'],next_fields['candidate'],'recovered_baseline_vs_candidate_'+key)
    cross['NEXT1_fields']=j3._next1_comparison(args,implementations['baseline'],implementations['candidate'])
    cross['all_hard_gates_pass']=all(cross[k]['hard_gate_pass'] for k in ('pulse1','pulse2','reload'))
    write(out/'baseline_vs_candidate_comparison.json',cross)
    after={str(path.relative_to(src)):{'bytes':path.stat().st_size,'raw_sha256':raw_sha256_file(path)} for path in sorted(src.rglob('*')) if path.is_file()}
    require(source_inventory==after,'source evidence changed during read-only recovery')
    write(out/'SOURCE_READONLY_INVENTORY.json',{'hash_scope':'raw_bytes','records':source_inventory,'unchanged_after':True})
    receipt={'schema':'filament.cpu_p1r2.j3_readonly_finalization.v1','status':'COMPLETE_DIAGNOSTIC_RECOVERED_AGGREGATION' if cross['all_hard_gates_pass'] else 'FAIL_NUMERICAL_HARD_GATE','source_job':args.source_job,'source_job_state':'FAILED','source_job_exit_code':'2:0','source_failure':failure,'baseline':implementations['baseline'],'candidate':implementations['candidate'],'cross_implementation_comparison':cross,'new_optical_windows':0,'new_hydro_transitions':0,'source_arrays_modified':False,'repair':'restore missing pulse1_window wrapper from reverified persisted files; no numerical rerun','service_wall_seconds':time.perf_counter()-started,'process_cpu_seconds':time.process_time()-cpu_started,'classification':'NUMERIC_DIAGNOSTIC_ONLY / NOT_SCIENTIFICALLY_QUALIFIED','acceptance_tolerance':'ACCEPTANCE_TOLERANCE_NOT_AUTHORIZED'}
    write(out/'run_receipt.json',receipt)
    return 0 if cross['all_hard_gates_pass'] else 2

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--source-job',required=True);p.add_argument('--source-repo',type=Path,required=True);p.add_argument('--source-manifest',type=Path,required=True)
    args=p.parse_args()
    try:return recover(args)
    except BaseException as exc:
        traceback.print_exc()
        if args.output_root.is_dir():write(args.output_root/'finalization_failure.json',{'error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc()})
        return 2
if __name__=='__main__':raise SystemExit(main())