"""Scalar-only six-window clock validation; factory and alpha remain separate."""
import math
import statistics
from .factory_timing import FACTORY_TIMING_SITES
from .cost_hook import ALPHA_SITE, ALPHA_LABEL
from .scope_inventory import RUNTIME_COUNTS

def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)

def _ancestors(row, tokens):
    result = []
    token = row['parent_token']
    while token is not None:
        require(token in tokens and token not in result, 'S46J_TIMING_PARENT_TREE')
        result.append(token)
        token = tokens[token]['parent_token']
    return [tokens[token] for token in result]

def _factory_site(row):
    return str(row.get('context',{}).get('source_site','')).startswith('S46J.factory.')

def validate_cost_records(raw, *, cost_mode, intervals):
    require(cost_mode in ('OFF','COST'), 'S46J_COST_MODE')
    require(raw['enabled'] is False and raw['events_dropped']==0 and not raw['incomplete_scopes'],
            'S46J_COST_RECORD_LIFECYCLE')
    records = raw['records']
    tokens = {row['token']:row for row in records}
    require(len(tokens)==len(records), 'S46J_DUPLICATE_TIMING_TOKEN')
    roots = [row for row in records if row['label']=='L1_FACTORY' and not _factory_site(row)]
    require(len(roots)==1, 'S46J_ONE_ORIGINAL_FACTORY_CLOCK')
    root = roots[0]
    factory_rows = [row for row in records if _factory_site(row)]
    alpha_rows = [row for row in records if row['label']==ALPHA_LABEL]
    require((bool(factory_rows) and bool(alpha_rows)) if cost_mode=='COST' else not factory_rows and not alpha_rows,
            'S46J_COST_OFF_RECORD_COVERAGE')
    detail = {}
    close_detail = {}
    sites = set(FACTORY_TIMING_SITES.values())
    for row in factory_rows:
        site = row['context']['source_site']
        require(site in sites and row['label']=='L1_FACTORY', 'S46J_UNKNOWN_FACTORY_SOURCE_SITE')
        require(row['status']=='complete' and not row['failed'] and
                row['wall_inclusive_ns']>=row['wall_exclusive_ns']>=0, 'S46J_FACTORY_CLOCK_INVALID')
        ancestors = _ancestors(row, tokens)
        require(not any(a['label']=='INTERVAL_BODY' for a in ancestors), 'S46J_FACTORY_INSIDE_BODY')
        in_factory = root['token'] in {a['token'] for a in ancestors}
        if not in_factory:
            require('.pool_lifecycle.' in site, 'S46J_FACTORY_SCOPE_OUTSIDE_BUILD_OR_CLOSE')
        bucket = detail if in_factory else close_detail
        item = bucket.setdefault(site, dict(calls=0, exclusive_s=0.0, inclusive_s=0.0,
             category=site.removeprefix('S46J.factory.').split('.',1)[0], contexts=[]))
        item['calls'] += 1
        item['exclusive_s'] += row['wall_exclusive_ns']/1e9
        item['inclusive_s'] += row['wall_inclusive_ns']/1e9
        item['contexts'].append(dict(token=row['token'], parent_token=row['parent_token'],
            ancestor_sites=[a['context'].get('source_site','') for a in ancestors]))
    alpha = dict(calls_by_interval=[0]*intervals, exclusive_s_by_interval=[0.0]*intervals,
                 inclusive_s_by_interval=[0.0]*intervals)
    for row in alpha_rows:
        require(row['context']['source_site']==ALPHA_SITE, 'S46J_ONLY_DECLARED_ALPHA_COST')
        i = row['context']['ordinal']
        require(i==row['context']['interval_index'] and 0<=i<intervals,
                'S46J_ALPHA_INTERVAL_DOMAIN')
        require(row['status']=='complete' and not row['failed'] and
                row['wall_inclusive_ns']==row['wall_exclusive_ns']>=0, 'S46J_ALPHA_PURE_POINTWISE_CLOCK')
        bodies = [a for a in _ancestors(row,tokens) if a['label']=='INTERVAL_BODY']
        require(len(bodies)==1 and bodies[0]['context']['ordinal']==i, 'S46J_ALPHA_OUTSIDE_OWN_BODY')
        alpha['calls_by_interval'][i] += 1
        alpha['exclusive_s_by_interval'][i] += row['wall_exclusive_ns']/1e9
        alpha['inclusive_s_by_interval'][i] += row['wall_inclusive_ns']/1e9
    require(alpha['calls_by_interval']==([1]*intervals if cost_mode=='COST' else [0]*intervals),
            'S46J_ALPHA_COMPLETE_CALL_COUNTS')
    observed_counts = dict(build={site:item['calls'] for site,item in detail.items()},
                           close={site:item['calls'] for site,item in close_detail.items()})
    if cost_mode == 'COST':
        require(isinstance(RUNTIME_COUNTS, dict) and observed_counts == RUNTIME_COUNTS[str(intervals)],
                'S46J_FIXED_FACTORY_CALL_COUNTS')
    factory_s = root['wall_inclusive_ns']/1e9
    attributed_s = sum(row['exclusive_s'] for row in detail.values())
    require(attributed_s <= factory_s+1e-8, 'S46J_FACTORY_EXCLUSIVE_DOUBLE_COUNT')
    return dict(schema='filament.cpu_s4_6j.cost_records.v1', cost_mode=cost_mode,
        intervals=intervals, cost_record_count=len(alpha_rows), factory_record_count=len(factory_rows),
        factory_s=factory_s, factory_detail=detail, close_detail=close_detail,
        observed_factory_counts=observed_counts,
        factory_uncovered_s=factory_s-attributed_s, alpha=alpha,
        parent_child_costs_added=False, worker_cpu_time_used=False, scientific_arrays_read=False)

def analyze_cost_six(cases, read_timing):
    expected = ('OFF2','COST2','OFF8_A','COST8_A','COST8_B','OFF8_B')
    require(tuple(cases)==expected, 'S46J_FIXED_SIX_ORDER')
    summaries, propagation = {}, {}
    for name in expected:
        case = cases[name]
        require(case['exact_gate_pass'] and case['pool_role']=='X', 'S46J_ALL_SIX_EXACT_REQUIRED')
        require(case['pointwise_calls']==38*case['intervals'], 'S46J_FOUR_COMPONENT_COUNTS')
        timing = read_timing(name)
        require(timing['merged']['clock_closure']['pass'] and not timing['a']['enabled']
                and not timing['a']['records'], 'S46J_FINE_OFF_COMMON_CLOCK_CLOSURE')
        summaries[name] = validate_cost_records(timing['b'], cost_mode=case['cost_mode'], intervals=case['intervals'])
        rows = [row for row in timing['b']['records'] if row['label']=='PROPAGATION']
        require(len(rows)==1, 'S46J_COMPLETE_PROPAGATION_SCOPE')
        propagation[name] = rows[0]['wall_inclusive_ns']/1e9
    metrics = {}
    for metric,key in (('application','application_service_s'),('propagation',None),('body_2_to_8','later_body_s')):
        pairs = []
        for off,cost in (('OFF8_A','COST8_A'),('OFF8_B','COST8_B')):
            r = propagation[off] if key is None else float(cases[off][key])
            x = propagation[cost] if key is None else float(cases[cost][key])
            require(math.isfinite(r) and math.isfinite(x) and min(r,x)>0, 'S46J_FINITE_INTERFERENCE_CLOCK')
            pairs.append(dict(off_case=off,cost_case=cost,off_s=r,cost_s=x,added_s=x-r,cost_over_off=x/r))
        median = statistics.median(p['cost_over_off'] for p in pairs)
        metrics[metric] = dict(pairs=pairs,median_ratio=median,
                              pass_gate=median<=1.05 and all(p['cost_over_off']<=1.10 for p in pairs))
    passed = all(row['pass_gate'] for row in metrics.values())
    result = dict(schema='filament.cpu_s4_6j.cost_six_result.v1',
        status='PASS_COST_MEASUREMENT' if passed else 'STOP_TIMING_INTERFERENCE',
        classification='VALID_COST_MEASUREMENT' if passed else 'TIMING_INTERFERENCE_INVALID',
        ranking_allowed=passed,timing_interference=metrics,scalar_summaries=summaries,
        cost_ranking=[],selection=None)
    if not passed:
        return result
    order = {site:i for i,site in enumerate(FACTORY_TIMING_SITES.values())}
    sites = set(summaries['COST8_A']['factory_detail']) | set(summaries['COST8_B']['factory_detail'])
    ranking = []
    for site in sites:
        observed = [dict(case=name, **summaries[name]['factory_detail'].get(site,
            dict(calls=0,exclusive_s=0.,inclusive_s=0.,contexts=[]))) for name in ('COST8_A','COST8_B')]
        if all(row['calls']>0 for row in observed):
            ranking.append(dict(source_site=site,source_order=order[site],
                median_exclusive_s=statistics.median(row['exclusive_s'] for row in observed),observations=observed))
    ranking.sort(key=lambda row:(-row['median_exclusive_s'],row['source_order']))
    result['cost_ranking'] = ranking
    result['selection'] = dict(status='MEASURED_FACTORY_COST_REQUIRES_SAFE_DUPLICATE_REVIEW',
        automatic_repair=False, at_most_one_repair=True, no_source_guard_relaxation=True)
    observed = [dict(case=name, first_s=summaries[name]['alpha']['exclusive_s_by_interval'][0],
        later_s=sum(summaries[name]['alpha']['exclusive_s_by_interval'][1:]),
        all_s=sum(summaries[name]['alpha']['exclusive_s_by_interval'])) for name in ('COST8_A','COST8_B')]
    result['alpha_cost'] = dict(source_site=ALPHA_SITE, observations=observed,
        median_later_exclusive_s=statistics.median(row['later_s'] for row in observed),
        median_all_exclusive_s=statistics.median(row['all_s'] for row in observed),
        baseline='FOUR_COMPONENTS_PLUS_L1', savings_measured=False)
    return result
