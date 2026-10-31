"""Standard-library, read-only S46C replay. Missing evidence is a failure."""
import argparse
import builtins
import csv
import importlib.util
import json
import math
from pathlib import Path, PureWindowsPath
import statistics
from types import SimpleNamespace


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('NONFINITE_JSON:'+x)))


def lines(path):
    return [json.loads(line,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('NONFINITE_JSON:'+x)))
            for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def require(value,why):
    if not value:raise ValueError(why)


def portable_leaf(path):
    return PureWindowsPath(path).name if '\\' in path else Path(path).name


def load_classifier(path,native_numpy=False):
    """Use the untouched gate, with only its three recorded dtype names decoded.

    This metadata-only adapter performs no arrays or scientific computation.
    Unknown dtype spellings are refused. Native NumPy is used only in the
    separately recorded Windows differential check, never required for replay.
    """
    spec=importlib.util.spec_from_file_location('s46c_frozen_numeric_gate',path)
    module=importlib.util.module_from_spec(spec)
    if not native_numpy:
        def dtype(name):
            kinds={'float64':'f','complex128':'c','int64':'i'}
            if name not in kinds:raise ValueError('UNREGISTERED_DTYPE_METADATA')
            return SimpleNamespace(kind=kinds[name])
        metadata=SimpleNamespace(dtype=dtype)
        def metadata_import(name,*args,**kwargs):
            return metadata if name=='numpy' else builtins.__import__(name,*args,**kwargs)
        module.__dict__['__builtins__']=dict(vars(builtins),__import__=metadata_import)
    spec.loader.exec_module(module)
    return module


def replay(root,native_numpy=False):
    root=Path(root);raw=root/'raw';plan=load(root/'PLAN.json')
    require((raw/'MECHANISM_COMPLETED.json').is_file(),'MISSING_MECHANISM_COMPLETION')
    completed=load(raw/'MECHANISM_COMPLETED.json')
    require(completed['input_unchanged'] and completed['version']==plan['version']
            and completed['inputs']==4 and completed['chains']==plan['chains']
            and completed['full_window_qualification'] is False,'FINAL_INPUT_AND_SCOPE')
    provenance_spec=importlib.util.spec_from_file_location('s46c_frozen_provenance',root/'execution_sources/provenance_v2.py')
    pv=importlib.util.module_from_spec(provenance_spec);provenance_spec.loader.exec_module(pv)
    delivery=root/'DELIVERY_MANIFEST.json'
    if delivery.is_file():
        for record in load(delivery)['records']:
            path=root/record['path']
            require(path.resolve().is_relative_to(root.resolve()),'DELIVERY_PATH_ESCAPE')
            require(pv.raw_sha256(path.read_bytes())==record['raw_sha256'],'DELIVERY_RAW_IDENTITY:'+record['path'])
    exported=load(root/'EXPORT_MANIFEST.json')
    require(exported['hash_scope']=='raw_bytes' and not exported['science_arrays_copied'],'METADATA_EXPORT_SCOPE')
    for record in exported['records']:
        path=root/record['path']
        require(path.resolve().is_relative_to(root.resolve()),'EXPORT_PATH_ESCAPE')
        payload=path.read_bytes()
        require(record['hash_scope']=='raw_bytes' and record['classification']=='external' and len(payload)==record['bytes']
                and pv.raw_sha256(payload)==record['raw_sha256'],'EXPORT_RAW_IDENTITY:'+record['path'])
    require(plan==load(raw/'PLAN.json'),'PUBLISHED_PLAN_MATCHES_EXECUTION')
    timings=lines(raw/'TIMINGS.jsonl')
    require(timings,'MISSING_TIMINGS')
    source=load(root/'EXECUTION.json')
    require(source==load(raw/'EXECUTION.json'),'PUBLISHED_EXECUTION_MATCHES_RAW')
    node=load(raw/'NODE_GATE.json');supervisor=load(raw/'SUPERVISOR_RECEIPT.json')
    require(node['source_commit']==source['execution_commit'],'EXECUTION_IDENTITY')
    require(supervisor['exit_code']==0 and supervisor['stop_reason'] is None,'SUPERVISOR_FAILED')
    classifier=root/'execution_sources/numeric_gate.py'
    manifest=load(raw/'EXECUTION_SOURCE_MANIFEST.json')
    bound=[r for r in manifest['records'] if r['path']=='Filament_python/tools/cpu_o1/numeric_gate.py']
    require(len(bound)==1 and pv.canonical_lf_sha256(classifier.read_bytes())==bound[0]['canonical_lf_sha256'],'FROZEN_NUMERIC_CLASSIFIER')
    source_paths={'raman.py':'Filament_python/KHz_filament/raman.py',
                  'provenance_v2.py':'Filament_python/tools/hpc_ops/provenance_v2.py'}
    source_paths.update({name:'Filament_python/tools/cpu_s4_6c_array_coverage/'+name
                        for name in ('chains.py','runner.py','node.py','cpu.sbatch','test_chains.py','test_runner.py')})
    for name,path in source_paths.items():
        binding=[r for r in manifest['records'] if r['path']==path]
        require(len(binding)==1 and pv.canonical_lf_sha256((root/'execution_sources'/name).read_bytes())==binding[0]['canonical_lf_sha256'],
                'EXECUTED_SOURCE_BINDING:'+name)
    module=load_classifier(classifier,native_numpy)
    admissions=lines(raw/'REAL_INPUT_ADMISSION.jsonl')
    require({r['input'] for r in admissions}==set(plan['inputs']) and len(admissions)==4,'FOUR_REAL_INPUTS')
    for name,digest in plan['stage_metadata_sha256'].items():
        require(pv.raw_sha256((root/'frozen_inputs'/('STAGES_'+name)).read_bytes())==digest,'FROZEN_INPUT_INDEX_BINDING')
    require(pv.raw_sha256((root/'frozen_inputs/INPUT_MANIFEST.json').read_bytes())==plan['canonical_manifest_sha256'],'CANONICAL_INPUT_BINDING')
    frozen_fingerprints=load(root/'frozen_inputs/STAGES_FINGERPRINTS.json')
    def check_exact(path):
        require(path.is_file(),'MISSING_EXACT:'+path.name)
        count=0
        for row in lines(path):
            if 'metrics' in row:
                require(all(row['metrics'].get(k) in ('float64','complex128','int64')
                            for k in ('reference_dtype','candidate_dtype')),'REGISTERED_DTYPE_SCOPE')
                require(module.classify_comparison(row)['proceed'],'RAW_NUMERIC_GATE:'+path.name)
                if row.get('reference_layout') is not None:
                    require(row['reference_layout']==row['candidate_layout'],'RAW_LAYOUT_GATE')
                count+=1
            else:require(row.get('classification') in ('TIMING_EXCLUDED','METADATA_EXACT'),'UNKNOWN_EXACT_ROW')
        require(count>0,'NO_NUMERIC_COMPARISONS')
        alias=load(path.with_suffix('.ALIASES.json'))
        require(alias['reference']==alias['candidate'],'RAW_ALIAS_GATE')
        return count
    numeric=0
    for a in admissions:
        require(a['fresh_response'] and not a['retired_capability_reused'],'FRESH_STAGE_ROUTE')
        for key,value in a['field_identity'].items():
            require(value==frozen_fingerprints[a['frozen_prefix']+key[len('bundle'):]],'REAL_FIELD_FROZEN_IDENTITY')
        numeric+=check_exact(raw/'exact'/portable_leaf(a['stage_gate']['comparison_path']))
        numeric+=check_exact(raw/'exact'/(a['input']+'_EXTRACTED_REF.jsonl'))
    qualification=lines(raw/'exact/QUALIFICATION.jsonl')
    expected_q={(i,c,m,w) for i in plan['inputs'] for c in plan['chains']
                for m,w in [('P',1),('P',8),('P',24),('P',48),('P_LOCAL',24),('P_LOCAL',48)]}
    require({(r['input'],r['chain'],r['mode'],r['workers']) for r in qualification}==expected_q
            and len(qualification)==len(expected_q),'COMPLETE_EXACT_MATRIX')
    output_ids={}
    field_ids={a['input']:a['field_identity']['bundle']['canonical_sha256'] for a in admissions}
    labels={('A','P'):['A_ABS','A_SQUARE','A_SCALE'],('A','P_LOCAL'):['A_LOCAL_ABS_SQUARE_SCALE'],
            ('B','P'):['B_SPECTRAL_MULTIPLY','B_DERIVATIVE_DIVIDE','B_RESPONSE_DIVIDE','B_PRODUCT'],
            ('B','P_LOCAL'):['B_SPECTRAL_MULTIPLY','B_LOCAL_DIVIDE_DIVIDE_PRODUCT']}
    def check_activity(record,calls=1):
        expected=[] if record['mode']=='REF' else labels[record['chain'],record['mode']]*calls
        require([a['label'] for a in record['target_activity']]==expected,'TARGET_OPERATION_COVERAGE')
        for a in record['target_activity']:
            require(a['actual_workers']==record['workers']==len(a['workers'])==len({t['thread_id'] for t in a['workers']}),'ACTUAL_DISTINCT_WORKERS')
            require(all(t['items']>0 and t['wall_ns']>0 and t['thread_cpu_ns']>=0 for t in a['workers']),'WORKER_ACTIVITY_DOMAIN')
    for q in qualification:
        require(q['input_unchanged'] and q['old_output_unchanged'] and q['fresh_no_alias'],'LIFETIME_OR_INPUT')
        require(q['mask_before']==q['mask_after']==24,'QUALIFICATION_MASK')
        require(all(a['actual_workers']==q['workers'] for a in q['target_activity']),'QUALIFICATION_WORKERS')
        check_activity(q,2)
        numeric+=check_exact(raw/'exact'/portable_leaf(q['gate']['comparison_path']))
        key=(q['input'],q['chain'])
        if key in output_ids:require(output_ids[key]==q['output_identity'],'QUALIFICATION_IDENTITY_AGREEMENT')
        output_ids[key]=q['output_identity']
    groups={};seen=set()
    for r in timings:
        group=(r['input'],r['chain'],r['factor'],r['mode'],r['workers'],r['block'])
        key=group+(r['repeat'],)
        require(key not in seen,'DUPLICATE_TIMING');seen.add(key)
        require(r['version']==plan['version'] and r['clock']=='COMPLETE_ISOLATED_CALL_WALL','VERSION_CLOCK')
        require(r['input'] in plan['inputs'] and r['chain'] in plan['chains'] and r['block'] in (0,1),'TIMING_SCOPE')
        require(r['warmup']==(r['repeat']<0),'WARMUP_CLASSIFICATION')
        require(math.isfinite(r['wall_s']) and r['wall_s']>0 and math.isfinite(r['process_cpu_s']) and r['process_cpu_s']>=0,'TIME_UNITS_FINITE')
        require(math.isclose(r['cpu_equivalent'],r['process_cpu_s']/r['wall_s'],rel_tol=1e-12),'CPU_EQUIVALENT')
        require(r['observation_before']['numba_mask']==r['observation_after']['numba_mask']==24,'MEASURED_MASK')
        require(r['output_identity']==output_ids[r['input'],r['chain']],'MEASURED_OUTPUT_IDENTITY')
        require(all(a['actual_workers']==r['workers'] for a in r['target_activity']),'MEASURED_TARGET_WORKERS')
        check_activity(r)
        groups.setdefault(group,[]).append(r)
    for key,rows in groups.items():
        require(sorted(r['repeat'] for r in rows)==list(range(-2,5)),'INCOMPLETE_FIXED_REPETITIONS:'+str(key))
    def med(i,c,f,m,w,b):
        require((i,c,f,m,w,b) in groups,'MISSING_CONFIGURATION:'+str((i,c,f,m,w,b)))
        return statistics.median(r['wall_s'] for r in groups[i,c,f,m,w,b] if not r['warmup'])
    expected_groups=set();matrix=[];eligible={}
    for i in plan['inputs']:
        for c in plan['chains']:
            for m,w in [('REF',0),('P',1),('P',8),('P',24),('P',48)]:
                for b in (0,1):
                    med(i,c,'THREAD',m,w,b);expected_groups.add((i,c,'THREAD',m,w,b))
            allow48=all(med(i,c,'THREAD','P',48,b)/med(i,c,'THREAD','P',24,b)<=.97 for b in (0,1))
            eligible[i,c]=allow48
            for w in ([24,48] if allow48 else [24]):
                for m in ('P','P_LOCAL'):
                    for b in (0,1):
                        med(i,c,'MATERIALIZATION',m,w,b);expected_groups.add((i,c,'MATERIALIZATION',m,w,b))
            ref=statistics.median([r['wall_s'] for b in (0,1) for r in groups[i,c,'THREAD','REF',0,b] if not r['warmup']])
            for m,w in [('REF',0),('P',1),('P',8),('P',24),('P',48)]+[('P_LOCAL',x) for x in ([24,48] if allow48 else [24])]:
                f='MATERIALIZATION' if m=='P_LOCAL' else 'THREAD'
                rows=[r for b in (0,1) for r in groups[i,c,f,m,w,b] if not r['warmup']]
                median=statistics.median(r['wall_s'] for r in rows)
                comparison=[med(i,c,'MATERIALIZATION','P',w,b) for b in (0,1)] if m=='P_LOCAL' else None
                matrix.append(dict(input=i,input_canonical_sha256=field_ids[i],chain=c,mode=m,workers=w,factor=f,mean_s=statistics.mean(r['wall_s'] for r in rows),
                  median_s=median,min_s=min(r['wall_s'] for r in rows),max_s=max(r['wall_s'] for r in rows),
                  relative_to_REF=median/ref,block0_median_s=med(i,c,f,m,w,0),block1_median_s=med(i,c,f,m,w,1),
                  process_cpu_mean_s=statistics.mean(r['process_cpu_s'] for r in rows),
                  cpu_equivalent_median=statistics.median(r['cpu_equivalent'] for r in rows),
                  target_worker_cpu_mean_s=statistics.mean(sum(t['thread_cpu_ns'] for a in r['target_activity'] for t in a['workers'])/1e9 for r in rows),
                  pool_initialization_median_s=statistics.median(r['pool_initialization_ns']/1e9 for r in rows),
                  same_thread_P_block0_s=comparison[0] if comparison else None,same_thread_P_block1_s=comparison[1] if comparison else None,
                  same_thread_P_ratio_block0=med(i,c,f,m,w,0)/comparison[0] if comparison else None,
                  same_thread_P_ratio_block1=med(i,c,f,m,w,1)/comparison[1] if comparison else None,
                  actual_target_workers_min=min((a['actual_workers'] for r in rows for a in r['target_activity']),default=0),
                  actual_target_workers_max=max((a['actual_workers'] for r in rows for a in r['target_activity']),default=0),
                  exact=True,samples=10,independent_nodes=1))
    require(set(groups)==expected_groups,'UNREGISTERED_CONFIGURATION')
    expected_order=[]
    for factor in ('THREAD','MATERIALIZATION'):
        for block in (0,1):
            for i in (plan['inputs'] if block==0 else list(reversed(plan['inputs']))):
                for c in (plan['chains'] if block==0 else list(reversed(plan['chains']))):
                    if factor=='THREAD':
                        configs=[('REF',0),('P',1),('P',8),('P',24),('P',48)]
                        if block:configs.reverse()
                    else:
                        configs=[]
                        for w in ([24,48] if eligible[i,c] else [24]):
                            pair=[('P',w),('P_LOCAL',w)]
                            configs.extend(pair if block==0 else list(reversed(pair)))
                    for m,w in configs:
                        expected_order.extend((i,c,factor,m,w,block,repeat) for repeat in range(-2,5))
    actual_order=[(r['input'],r['chain'],r['factor'],r['mode'],r['workers'],r['block'],r['repeat']) for r in timings]
    require(actual_order==expected_order,'PREREGISTERED_TIMING_ORDER')
    registered=load(raw/'LOCAL48_ADMISSION.json')['records']
    require({(x['input'],x['chain']):x['allowed'] for x in registered}==eligible,'LOCAL48_REPLAY')
    candidates=[]
    for c in plan['chains']:
        for m,w in [('P',1),('P',8),('P',24),('P',48),('P_LOCAL',24),('P_LOCAL',48)]:
            if m=='P_LOCAL' and w==48 and not all(eligible[i,c] for i in plan['inputs']):continue
            ratios=[]
            for i in plan['inputs']:
                for b in (0,1):
                    ratios.append(med(i,c,'MATERIALIZATION' if m=='P_LOCAL' else 'THREAD',m,w,b)/med(i,c,'THREAD','REF',0,b))
            if max(ratios)<=.95:
                anchor=12.528 if c=='A' else 34.388
                candidates.append(dict(chain=c,mode=m,workers=w,worst_input_block_ratio=max(ratios),
                      conservative_projected_old_CPU_s=anchor*(1-max(ratios)),projection_not_window_evidence=True))
    candidates.sort(key=lambda x:x['conservative_projected_old_CPU_s'],reverse=True)
    if delivery.is_file():
        with (root/'MATRIX.csv').open(encoding='utf-8-sig',newline='') as f:
            published=list(csv.DictReader(f))
        require(published==[{k:'' if v is None else str(v) for k,v in row.items()} for row in matrix],
                'PUBLISHED_MATRIX_RECOMPUTATION')
    return dict(status='PASS_READONLY_MECHANISM_REPLAY',timing_rows=len(timings),numeric_rows=numeric,matrix=matrix,
                conditional_candidate=candidates[0] if candidates else None,candidates=candidates,
                review_gate='MANUAL_REVIEW_PENDING',full_window_qualification=False,production_qualified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--full',action='store_true');args=p.parse_args()
    try:
        result=replay(args.root)
        print(json.dumps(result if args.full else {k:v for k,v in result.items() if k not in ('matrix','candidates')},ensure_ascii=False,allow_nan=False,indent=2))
    except (ValueError,KeyError,FileNotFoundError,TypeError) as e:
        print(json.dumps(dict(status='FAIL_READONLY_REPLAY',error=repr(e))));raise SystemExit(1)
