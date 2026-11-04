"""Offline scalar COST6 replay; measurement does not authorize optimization."""
import math
import statistics
from .registry import registry_snapshot

def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)

def validate_cost_records(raw, *, cost_mode, intervals):
    registry = registry_snapshot()
    site_rows = registry['sites']
    sites = {row['source_site']:row for row in site_rows}
    require(len(sites)==len(site_rows), 'S46K_DUPLICATE_SITE')
    require(cost_mode in ('OFF','COST'), 'S46K_COST_MODE')
    require(raw['enabled'] is False and raw['events_dropped']==0 and not raw['incomplete_scopes'],
        'S46K_SCALAR_RECORD_LIFECYCLE')
    records = raw['records']
    tokens = {row['token']:row for row in records}
    require(len(tokens)==len(records), 'S46K_DUPLICATE_TOKEN')
    rows = [row for row in records if row['label']==registry['scope_label']]
    require(bool(rows)==(cost_mode=='COST'), 'S46K_COST_ON_OFF_RECORDS')
    eligible = {row['token'] for row in rows if sites.get(row['context']['source_site'],{}).get('eligible')}
    detail = {}
    for row in rows:
        site = row['context']['source_site']
        require(site in sites, 'S46K_UNREGISTERED_SITE:'+site)
        spec = sites[site]
        require(row['status']=='complete' and row['failed'] is False, 'S46K_FAILED_SCOPE')
        index = row['context']['interval_index']
        require(type(index) is int and 0<=index<intervals and row['context']['ordinal']==index,
            'S46K_INTERVAL_CONTEXT')
        require(row['wall_inclusive_ns']>=row['wall_exclusive_ns']>=0, 'S46K_WALL_CLOCK')
        ancestors=[]
        parent=row['parent_token']
        while parent is not None:
            require(parent in tokens and parent not in ancestors, 'S46K_CLOCK_TREE')
            ancestors.append(parent)
            parent=tokens[parent]['parent_token']
        bodies=[tokens[token] for token in ancestors if tokens[token]['label']=='INTERVAL_BODY']
        require(len(bodies)==1 and bodies[0]['context']['ordinal']==index, 'S46K_OWN_BODY')
        if row['token'] in eligible:
            require(not eligible.intersection(ancestors), 'S46K_PARENT_CHILD_DOUBLE_COUNT')
        values=detail.setdefault(site,dict(name=spec['name'],unit=spec['unit'],
            eligible=bool(spec['eligible']),scope_kind=spec['scope_kind'],
            calls_by_interval=[0]*intervals,exclusive_s_by_interval=[0.]*intervals,
            inclusive_s_by_interval=[0.]*intervals))
        values['calls_by_interval'][index]+=1
        values['exclusive_s_by_interval'][index]+=row['wall_exclusive_ns']/1e9
        values['inclusive_s_by_interval'][index]+=row['wall_inclusive_ns']/1e9
    if cost_mode=='COST':
        for site,spec in sites.items():
            expected=spec.get('expected_calls_per_interval')
            if expected is not None:
                require(detail.get(site,{}).get('calls_by_interval')==[expected]*intervals,
                    'S46K_EXPECTED_SITE_COUNTS:'+site)
    return dict(schema='filament.cpu_s4_6k.cost_records.v1',cost_mode=cost_mode,
        cost_record_count=len(rows),factory_record_count=0,intervals=intervals,
        source_detail=detail,inactive_sites=[site for site in sites if site not in detail],
        registry=registry,clock_domain='exclusive_wall_seconds_in_original_interval_body',
        parent_child_costs_added=False,worker_cpu_time_used=False,scientific_arrays_read=False)

def _observations(summaries, select):
    values=[]
    for name in ('COST8_A','COST8_B'):
        detail={site:row for site,row in summaries[name]['source_detail'].items() if select(row)}
        values.append(dict(case=name,contexts=detail,
            later_calls=sum(sum(row['calls_by_interval'][1:]) for row in detail.values()),
            first_s=sum(row['exclusive_s_by_interval'][0] for row in detail.values()),
            later_s=sum(sum(row['exclusive_s_by_interval'][1:]) for row in detail.values()),
            all_s=sum(sum(row['exclusive_s_by_interval']) for row in detail.values())))
    return values

def analyze_cost_six(cases, read_timing):
    expected=('OFF2','COST2','OFF8_A','COST8_A','COST8_B','OFF8_B')
    require(tuple(cases)==expected, 'S46K_FIXED_SIX_ORDER')
    summaries={}
    propagation={}
    for name in expected:
        case=cases[name]
        require(case['exact_gate_pass'] and case['pool_role']=='X'
            and case.get('alpha_enabled') is True, 'S46K_FIVE_CHAIN_SIX_EXACT')
        require(case['pointwise_calls']==43*case['intervals'] and case['alpha_rounds']==5*case['intervals'],
            'S46K_FIVE_CHAIN_COUNTS')
        raw=read_timing(name)
        require(raw['merged']['clock_closure']['pass'], 'S46K_CLOCK_CLOSURE')
        require(raw['a']['enabled'] is False and not raw['a']['records'], 'S46K_OLD_FINE_PROFILE_OFF')
        summaries[name]=validate_cost_records(raw['b'],cost_mode=case['cost_mode'],intervals=case['intervals'])
        rows=[row for row in raw['b']['records'] if row['label']=='PROPAGATION']
        require(len(rows)==1, 'S46K_PROPAGATION_CLOCK')
        propagation[name]=rows[0]['wall_inclusive_ns']/1e9
    metrics={}
    passed=True
    for metric,key in (('application','application_service_s'),('propagation',None),('body_2_to_8','later_body_s')):
        pairs=[]
        for off,cost in (('OFF8_A','COST8_A'),('OFF8_B','COST8_B')):
            r=propagation[off] if key is None else float(cases[off][key])
            x=propagation[cost] if key is None else float(cases[cost][key])
            require(math.isfinite(r) and math.isfinite(x) and min(r,x)>0, 'S46K_FINITE_TIMES')
            pairs.append(dict(off_case=off,cost_case=cost,off_s=r,cost_s=x,added_s=x-r,cost_over_off=x/r))
        median=statistics.median(row['cost_over_off'] for row in pairs)
        gate=median<=1.05 and all(row['cost_over_off']<=1.10 for row in pairs)
        metrics[metric]=dict(pairs=pairs,median_ratio=median,pass_gate=gate)
        passed=passed and gate
    result=dict(schema='filament.cpu_s4_6k.cost_six_result.v1',
        status='PASS_COST_MEASUREMENT_STOP' if passed else 'STOP_TIMING_INTERFERENCE',
        classification='VALID_COST_MEASUREMENT' if passed else 'TIMING_INTERFERENCE_INVALID',
        ranking_allowed=passed,timing_interference=metrics,scalar_summaries=summaries,
        cost_ranking=[],selection=None,baseline='REF_FIVE_CHAIN_L1',
        new_candidate_implemented=False,stop_after_this_measurement=True)
    if not passed:
        return result
    registry=registry_snapshot()
    units={}
    for row in registry['sites']:
        if row['eligible']:
            units[row['unit']]=min(units.get(row['unit'],row['source_order']),row['source_order'])
    for unit,order in units.items():
        observed=_observations(summaries,lambda row:row['eligible'] and row['unit']==unit)
        if all(row['later_calls']>0 for row in observed):
            result['cost_ranking'].append(dict(unit=unit,source_order=order,observations=observed,
                median_later_exclusive_s=statistics.median(row['later_s'] for row in observed),
                median_first_exclusive_s=statistics.median(row['first_s'] for row in observed),
                median_all_exclusive_s=statistics.median(row['all_s'] for row in observed)))
    result['cost_ranking'].sort(key=lambda row:(-row['median_later_exclusive_s'],row['source_order']))
    result['phase_subsegments']={row['name']:_observations(summaries,
        lambda value,name=row['name']:value['name']==name) for row in registry['sites']
        if row['unit']=='alpha_phase_assembly' or row.get('phase_related')}
    parents=[]
    for name in ('COST8_A','COST8_B'):
        spans=[row for row in summaries[name]['source_detail'].values()
            if row['name']=='alpha_phase_assembly']
        require(len(spans)==1, 'S46K_PHASE_PARENT_REQUIRED')
        span=spans[0]
        parents.append(dict(case=name,first_inclusive_s=span['inclusive_s_by_interval'][0],
            later_inclusive_s=sum(span['inclusive_s_by_interval'][1:]),
            all_inclusive_s=sum(span['inclusive_s_by_interval']),
            later_exclusive_wrapper_s=sum(span['exclusive_s_by_interval'][1:])))
    result['phase_parent_clock']=dict(observations=parents,
        median_later_inclusive_s=statistics.median(row['later_inclusive_s'] for row in parents),
        median_first_inclusive_s=statistics.median(row['first_inclusive_s'] for row in parents),
        median_all_inclusive_s=statistics.median(row['all_inclusive_s'] for row in parents),
        used_in_ranking=False,includes_child_scopes=True)
    result['selection']=dict(status='RANKING_ONLY_AWAIT_USER_INSTRUCTION',
        highest_observed_unit=result['cost_ranking'][0]['unit'] if result['cost_ranking'] else None,
        candidate_implementation_authorized=False)
    return result
