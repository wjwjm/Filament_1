"""Fixed offline summaries of the S46C matrix, after independent replay."""
import argparse
import csv
import json
from pathlib import Path
import statistics as st
from replay import replay, lines, load


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize(root):
    result=replay(root)
    write_json(root/'REPLAY_FULL.json',result)
    write_csv(root/'MATRIX.csv',result['matrix'])
    rows=[r for r in lines(root/'raw/TIMINGS.jsonl') if not r['warmup']]
    plan=load(root/'PLAN.json')
    summaries=[];activity=[];factors=[]
    def select(chain,factor,mode,workers,input_id=None,block=None):
        return [r for r in rows if r['chain']==chain and r['factor']==factor and r['mode']==mode and r['workers']==workers
                and (input_id is None or r['input']==input_id) and (block is None or r['block']==block)]
    def median(chain,factor,mode,workers,input_id,block):
        return st.median(r['wall_s'] for r in select(chain,factor,mode,workers,input_id,block))
    for c in plan['chains']:
        configs=sorted({(r['factor'],r['mode'],r['workers']) for r in rows if r['chain']==c})
        for f,m,w in configs:
            group=select(c,f,m,w)
            summaries.append(dict(chain=c,factor=f,mode=m,workers=w,inputs=len({r['input'] for r in group}),samples=len(group),
                mean_s=st.mean(r['wall_s'] for r in group),median_s=st.median(r['wall_s'] for r in group),
                min_s=min(r['wall_s'] for r in group),max_s=max(r['wall_s'] for r in group),
                process_cpu_s=st.mean(r['process_cpu_s'] for r in group),
                process_cpu_wall=sum(r['process_cpu_s'] for r in group)/sum(r['wall_s'] for r in group),
                pool_init_s=st.mean(r['pool_initialization_ns']/1e9 for r in group),
                observed_peak_rss_bytes=max(max(r['observation_before']['rss'],r['observation_after']['rss']) for r in group)))
            labels=sorted({a['label'] for r in group for a in r['target_activity']})
            for label in labels:
                acts=[a for r in group for a in r['target_activity'] if a['label']==label]
                activity.append(dict(chain=c,factor=f,mode=m,workers=w,label=label,calls=len(acts),
                    actual_workers_min=min(a['actual_workers'] for a in acts),actual_workers_max=max(a['actual_workers'] for a in acts),
                    worker_CPU_mean_s=st.mean(sum(t['thread_cpu_ns'] for t in a['workers'])/1e9 for a in acts),
                    longest_worker_wall_mean_s=st.mean(max(t['wall_ns'] for t in a['workers'])/1e9 for a in acts),
                    window='TARGET_OPERATION_ONLY_EXCLUDES_BARRIER_AND_HOST; NOT_ADDITIVE_WALL'))
        for i in plan['inputs']:
            for b in (0,1):
                ref=median(c,'THREAD','REF',0,i,b);p1=median(c,'THREAD','P',1,i,b)
                for w in (1,8,24,48):
                    p=median(c,'THREAD','P',w,i,b)
                    factors.append(dict(input=i,chain=c,block=b,factor='THREAD',workers=w,baseline='P1',
                        baseline_s=p1,candidate_s=p,saved_s=p1-p,ratio=p/p1,
                        REF_s=ref,P1_minus_REF_s=p1-ref,note='P1 vs REF is implementation and management, not multicore'))
                for w in (24,48):
                    if not select(c,'MATERIALIZATION','P_LOCAL',w,i,b):continue
                    p=median(c,'MATERIALIZATION','P',w,i,b);pl=median(c,'MATERIALIZATION','P_LOCAL',w,i,b)
                    factors.append(dict(input=i,chain=c,block=b,factor='MATERIALIZATION',workers=w,baseline='same-thread P',
                        baseline_s=p,candidate_s=pl,saved_s=p-pl,ratio=pl/p,
                        REF_s=ref,P1_minus_REF_s=p1-ref,note='Paired materialization round only; no DRAM attribution'))
    write_csv(root/'AGGREGATE_MATRIX.csv',summaries)
    write_csv(root/'THREAD_ACTIVITY.csv',activity)
    write_csv(root/'SEPARATED_FACTORS.csv',factors)
    allobs=[r[k] for r in rows for k in ('observation_before','observation_after')]
    analysis=dict(status=result['status'],formal_calls=len(rows),warmup_calls=len(lines(root/'raw/TIMINGS.jsonl'))-len(rows),
        independent_nodes=1,inputs=4,conditional_candidate=result['conditional_candidate'],
        aggregate_method='Arithmetic mean of all formal complete calls per configuration; unequal input coverage explicitly shown',
        block_decisions='Median of each input/block five repeats; two blocks are not independent nodes',
        summaries=summaries,local48=load(root/'raw/LOCAL48_ADMISSION.json'),
        observed_pss_max_bytes=max(x['pss'] for x in allobs if x['pss'] is not None),
        NUMA_observed_nodes=sorted({n for x in allobs for n,v in x['numa_pages'].items() if v}),
        NUMA_first_touch_attribution='UNKNOWN_AGGREGATE_PAGE_COUNTS_ONLY',
        affinity_constant=all(x['affinity']==allobs[0]['affinity'] for x in allobs),
        affinity=allobs[0]['affinity'],full_window_verified=False,GPU_measured=False)
    write_json(root/'ANALYSIS.json',analysis)
    print(json.dumps({k:v for k,v in analysis.items() if k not in ('summaries','affinity','local48')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    summarize(parser.parse_args().root)
