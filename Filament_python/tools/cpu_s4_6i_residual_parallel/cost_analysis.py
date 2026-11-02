"""Scalar-only cost-six validation and preregistered ranking; no arrays read."""
from __future__ import annotations

import math
import statistics

from .registry import registry_snapshot


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def _site_units():
    registry = registry_snapshot()
    units = list(registry['units']) + list(registry['pointwise_helper_units'])
    power = dict(registry['power_volume'])
    units.append(power)
    sites = {}
    for unit in units:
        contexts = unit.get('contexts', ())
        names = [unit['source_site'] + ':' + item for item in contexts] if contexts else [unit['source_site']]
        for source_site in names:
            require(source_site not in sites, 'S46I_DUPLICATE_SOURCE_SITE')
            sites[source_site] = unit
    for unit in registry.get('noncandidate_units', ()):
        require(not unit['eligible'], 'S46I_NONCANDIDATE_ELIGIBILITY')
        for site in ([unit['source_site'] + ':' + name for name in unit['contexts']]
                     if unit.get('contexts') else [unit['source_site']]):
            require(site not in sites, 'S46I_DUPLICATE_SOURCE_SITE')
            sites[site] = unit
    return registry, units, sites


def validate_cost_records(raw, *, cost_mode, intervals):
    registry, units, sites = _site_units()
    require(cost_mode in ('OFF', 'COST'), 'S46I_COST_MODE')
    require(raw['enabled'] is False and raw['events_dropped'] == 0
            and not raw['incomplete_scopes'], 'S46I_COST_RECORD_LIFECYCLE')
    records = raw['records']
    tokens = {row['token']: row for row in records}
    require(len(tokens) == len(records), 'S46I_DUPLICATE_TIMING_TOKEN')
    rows = [row for row in records if row['label'] == registry['scope_label']]
    if cost_mode == 'OFF':
        require(not rows, 'S46I_OFF_RESIDUAL_RECORDS')
    else:
        require(bool(rows), 'S46I_COST_RECORDS_MISSING')
    detail = {}
    eligible_tokens = {row['token'] for row in rows
                       if sites.get(row['context']['source_site'], {}).get('eligible')}
    for row in rows:
        site = row['context']['source_site']
        require(site in sites, 'S46I_UNREGISTERED_COST_SOURCE_SITE:' + site)
        require(row['status'] == 'complete' and row['failed'] is False,
                'S46I_COST_RECORD_FAILED')
        index = row['context']['ordinal']
        require(index == row['context']['interval_index'] and 0 <= index < intervals,
                'S46I_COST_INTERVAL_DOMAIN')
        require(row['wall_inclusive_ns'] >= row['wall_exclusive_ns'] >= 0,
                'S46I_COST_WALL_CLOCK')
        ancestors = []
        parent_token = row['parent_token']
        while parent_token is not None:
            require(parent_token in tokens and parent_token not in ancestors,
                    'S46I_COST_PARENT_TREE')
            ancestors.append(parent_token)
            parent_token = tokens[parent_token]['parent_token']
        bodies = [tokens[token] for token in ancestors if tokens[token]['label'] == 'INTERVAL_BODY']
        if sites[site].get('scope_kind') == 'outside_body':
            require(not bodies and not sites[site]['eligible'], 'S46I_INITIALIZATION_SCOPE_DOMAIN')
        else:
            require(len(bodies) == 1 and bodies[0]['context']['ordinal'] == index,
                    'S46I_COST_OUTSIDE_OWN_BODY')
        if row['token'] in eligible_tokens:
            require(not eligible_tokens.intersection(ancestors), 'S46I_ELIGIBLE_PARENT_CHILD_OVERLAP')
        values = detail.setdefault(site, dict(unit=sites[site]['name'],
            eligible=bool(sites[site]['eligible']), calls_by_interval=[0]*intervals,
            exclusive_s_by_interval=[0.0]*intervals, inclusive_s_by_interval=[0.0]*intervals))
        values['calls_by_interval'][index] += 1
        values['exclusive_s_by_interval'][index] += row['wall_exclusive_ns']/1e9
        values['inclusive_s_by_interval'][index] += row['wall_inclusive_ns']/1e9
    # Source coverage is checked structurally by the AST proof. Runtime branches
    # absent from the frozen input are reported inactive and never ranked.
    return dict(schema='filament.cpu_s4_6i.cost_records.v1', cost_mode=cost_mode,
        cost_record_count=len(rows), intervals=intervals, source_detail=detail,
        registered_sites=list(sites), inactive_sites=[site for site in sites if site not in detail],
        eligible_units=[unit['name'] for unit in units if unit['eligible']],
        clock_domain='exclusive_wall_seconds_in_original_interval_body',
        parent_child_costs_added=False, worker_cpu_time_used=False,
        scientific_arrays_read=False)


def analyze_cost_six(cases, read_timing):
    expected = ('OFF2','COST2','OFF8_A','COST8_A','COST8_B','OFF8_B')
    require(tuple(cases) == expected, 'S46I_FIXED_SIX_ORDER')
    summaries = {}
    propagation = {}
    for name in expected:
        case = cases[name]
        require(case['exact_gate_pass'] and case['pool_role'] == 'X', 'S46I_SIX_EXACT_REQUIRED')
        timing = read_timing(name)
        require(timing['merged']['clock_closure']['pass'], 'S46I_SIX_CLOCK_CLOSURE')
        require(timing['a']['enabled'] is False and not timing['a']['records'], 'S46I_FINE_PROFILE_OFF')
        summaries[name] = validate_cost_records(timing['b'], cost_mode=case['cost_mode'], intervals=case['intervals'])
        rows = [row for row in timing['b']['records'] if row['label'] == 'PROPAGATION']
        require(len(rows) == 1, 'S46I_COMPLETE_PROPAGATION_SCOPE')
        propagation[name] = rows[0]['wall_inclusive_ns']/1e9
    metrics = {}
    passed = True
    for name, key in (('application','application_service_s'),
                      ('propagation',None),('body_2_to_8','later_body_s')):
        pairs = []
        for off, cost in (('OFF8_A','COST8_A'),('OFF8_B','COST8_B')):
            r = propagation[off] if key is None else float(cases[off][key])
            x = propagation[cost] if key is None else float(cases[cost][key])
            require(math.isfinite(r) and math.isfinite(x) and min(r,x)>0, 'S46I_FINITE_INTERFERENCE_CLOCK')
            pairs.append(dict(off_case=off,cost_case=cost,off_s=r,cost_s=x,
                              added_s=x-r,cost_over_off=x/r))
        median = statistics.median(row['cost_over_off'] for row in pairs)
        gate = median <= 1.05 and all(row['cost_over_off'] <= 1.10 for row in pairs)
        metrics[name] = dict(pairs=pairs,median_ratio=median,pass_gate=gate)
        passed = passed and gate
    result = dict(schema='filament.cpu_s4_6i.cost_six_result.v1',
        status='PASS_COST_MEASUREMENT' if passed else 'STOP_TIMING_INTERFERENCE',
        classification='VALID_COST_MEASUREMENT' if passed else 'TIMING_INTERFERENCE_INVALID',
        ranking_allowed=passed, timing_interference=metrics,
        scalar_summaries=summaries, cost_ranking=[], selection=None)
    if not passed:
        return result
    registry, units, sites = _site_units()
    ranking = []
    for unit in units:
        if not unit['eligible']:
            continue
        observations = []
        active = True
        for name in ('COST8_A','COST8_B'):
            detail = {site:row for site,row in summaries[name]['source_detail'].items()
                      if row['unit'] == unit['name']}
            later_calls = sum(sum(row['calls_by_interval'][1:]) for row in detail.values())
            active = active and later_calls > 0
            observations.append(dict(case=name, later_calls=later_calls,
                first_s=sum(row['exclusive_s_by_interval'][0] for row in detail.values()),
                later_s=sum(sum(row['exclusive_s_by_interval'][1:]) for row in detail.values()),
                all_s=sum(sum(row['exclusive_s_by_interval']) for row in detail.values()),
                contexts=detail))
        if active:
            ranking.append(dict(unit=unit['name'],source_order=unit['source_order'],
                median_later_exclusive_s=statistics.median(row['later_s'] for row in observations),
                median_first_exclusive_s=statistics.median(row['first_s'] for row in observations),
                median_all_exclusive_s=statistics.median(row['all_s'] for row in observations),
                observations=observations))
    ranking.sort(key=lambda row:(-row['median_later_exclusive_s'],row['source_order']))
    result['cost_ranking'] = ranking
    result['selection'] = dict(status='SELECTED_ONE' if ranking else 'NO_ELIGIBLE_EXECUTED_UNIT',
        unit=ranking[0]['unit'] if ranking else None,
        source_order=ranking[0]['source_order'] if ranking else None,
        frozen_registry=registry, fixed_workers=24, fixed_axis=0,
        no_second_rank_retry=True, selection_basis='largest_median_exclusive_wall_sum_body_2_to_8',
        scientific_exact_does_not_establish_performance=True)
    if not ranking:
        result.update(status='STOP_NO_ELIGIBLE_UNIT',classification='VALID_NO_ELIGIBLE_UNIT')
    return result
