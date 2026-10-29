"""Read-only final-package acceptance, preserving declared limited outcomes."""
from __future__ import annotations
import argparse, json, math, platform, sys, time
from pathlib import Path

def require(condition, reason):
    if not condition: raise RuntimeError(reason)

def validate(root):
    sys.path.insert(0,str(root))
    from replay_readonly import replay_report
    start=time.perf_counter()
    result=replay_report(root)
    allowed=lambda issue: ('TIMING_GATE:TIMING_GATE_FAIL' in issue and ('GPU_351' in issue or 'GPU/shape:384x351x301' in issue))
    unexpected=[x for x in result['issues'] if not allowed(x)]
    require(not unexpected,'UNEXPECTED_REPLAY_ISSUES:'+json.dumps(unexpected))
    require(result['executed_source']['status']=='SOURCE_BUNDLES_REPLAYED','SOURCE_BUNDLES_NOT_VERIFIED')
    require(result['resources']['status']=='REPLAYED','RESOURCE_LEDGER_NOT_REPLAYED')
    require(result['manifests']['status']=='MANIFEST_METADATA_ONLY','INPUT_METADATA_STATUS')
    require(result['historical']['status']=='REPLAYED','HISTORY_NOT_REPLAYED')
    require(result['cpu_gpu_ratios']['status']=='REPLAYED','RATIOS_NOT_REPLAYED')
    require(len(result['matrix'])==4,'EXACTLY_FOUR_DATASETS_REQUIRED')
    for dataset in result['matrix']:
        require(dataset['case_order_ok'],'FORMAL_CASE_ORDER')
        require(all(dataset['off_statistics']['metrics'][m]['count']==3 for m in ('application','window','later')),'OFF_COUNT')
    gates={(x['backend'],x['shape']):x['status'] for x in result['timing_gates']}
    require(gates[('GPU','shape:384x351x301')]=='TIMING_GATE_FAIL','GPU_FULL_LIGHT_FAILURE_MUST_SURVIVE')
    require(sum(x=='PASS_TIMING_GATE' for x in gates.values())==3,'OTHER_THREE_LIGHT_GATES')
    require(len(result['native_off_macro_summaries'])==2,'MACRO_PAIR_COUNT')
    require(all(x['delta']['closure_pass'] and x['delta']['closure_residual_decimal_s']=='0' for x in result['native_off_macro_summaries']),'MACRO_CLOSURE')
    source=result['executed_source']
    resources=result['resources']['backends']
    require(len(resources['CPU']['job_ids'])==4 and len(resources['GPU']['job_ids'])==4,'EIGHT_JOB_HISTORY_REQUIRED')
    require(math.isclose(resources['CPU']['actual_core_hours'],150.77333333333334,abs_tol=1e-10),'CPU_CORE_HOURS')
    require(math.isclose(resources['GPU']['actual_gpu_hours'],2668/3600,abs_tol=1e-10),'GPU_HOURS')
    require(math.isclose(resources['GPU']['actual_core_hours'],2668*8/3600,abs_tol=1e-10),'GPU_HOST_HOURS')
    full_cpu=next(x for x in result['light_exclusive_summaries'] if x['backend']=='CPU' and '351' in x['shape'])
    require(full_cpu['closure_all_pass'] and len(full_cpu['categories'])==7,'CPU_EXCLUSIVE_CLOSURE')
    require(len(result['aux_serialized_summaries'])==2,'SEPARATE_AUX_REQUIRED')
    pv=sys.modules['_s46a_frozen_pv2_replay_gpu']
    cross_path=root/'CROSS_FULL_NUMERIC_A01/CROSS_BACKEND_NUMERIC_DIFFERENCES_FULL.json'
    require(pv.raw_sha256_file(cross_path)=='48d55398516b440132c643358210ca43a1ab96fd6a224cc10e58c74530054ae6','CROSS_RAW_TRANSFER_HASH')
    numeric=json.loads(cross_path.read_text(encoding='utf-8'))
    require(len(numeric['fields'])==251 and numeric['matched_input_workload'],'CROSS_FIELDS_AND_WORKLOAD')
    require(numeric['status']=='CROSS_BACKEND_NUMERICAL_EQUIVALENCE_NOT_QUALIFIED','CROSS_QUALIFICATION_BOUNDARY')
    require(result['DEFAULT']=='I0' and result['PRODUCTION_QUALIFIED'] is False and result['MANUAL_REVIEW_PENDING'] is True,'LIFECYCLE')
    expected_missing={'TIMING_GATE','SOURCE_INPUT_MANIFEST','SOURCE_INPUT_ARRAY_BYTES_NOT_RECHECKED'}
    require(set(result['missing'])==expected_missing,'UNEXPECTED_MISSING:'+json.dumps(result['missing']))
    return dict(status='PASS_EXPECTED_LIMITED_REPLAY',platform=platform.platform(),python=sys.version,
        root=str(root),wall_s=time.perf_counter()-start,replay_report_status=result['status'],
        conclusion='TOTAL_GAP_MEASURED / DETAIL_ATTRIBUTION_LIMITED',
        source_status=source['status'],source_bundles=source.get('bundles',source.get('roles')),
        cpu_gpu_ratios=result['cpu_gpu_ratios'],native_off_macro_summaries=result['native_off_macro_summaries'],
        timing_gates=gates_as_list(result['timing_gates']),resources=result['resources'],
        clocks_status=result['clocks']['status'],metadata_status=result['manifests']['status'],
        metadata_issues=result['manifests']['issues'],arrays_not_rechecked=result['manifests']['arrays_not_rechecked'],
        expected_issues=result['issues'],expected_missing=result['missing'],
        note='Replay success verifies original arithmetic/identity and expected failures; not scientific or LIGHT PASS',
        DEFAULT='I0',PRODUCTION_QUALIFIED=False,MANUAL_REVIEW_PENDING=True)

def gates_as_list(rows):
    return [dict(backend=x['backend'],shape=x['shape'],status=x['status']) for x in rows]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:
        value=validate(args.root.resolve())
        with args.output.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
        print(json.dumps(dict(status=value['status'],output=str(args.output),wall_s=value['wall_s'],resources=value['resources']['status'])))
    except Exception as exc:
        print(json.dumps(dict(status='FINAL_REPLAY_VALIDATION_FAILED',error=str(exc)),ensure_ascii=False));raise SystemExit(2)
