"""Count loss and timing interference must prevent any cost ranking."""
import copy
import pytest
from tools.cpu_s4_6j_alpha_factory.cost_analysis import validate_cost_records, analyze_cost_six
from tools.cpu_s4_6j_alpha_factory.scope_inventory import RUNTIME_COUNTS
from tools.cpu_s4_6j_alpha_factory.cost_hook import ALPHA_SITE, ALPHA_LABEL


def _raw(n=8, mode='COST'):
    records=[]
    def add(label, site='', parent=None, ordinal=0, ns=1):
        token=len(records)+1
        records.append(dict(token=token, parent_token=parent, label=label, status='complete', failed=False,
            wall_inclusive_ns=ns, wall_exclusive_ns=ns,
            context=dict(source_site=site,ordinal=ordinal,interval_index=ordinal)))
        return token
    factory=add('L1_FACTORY',ns=10**9)
    if mode=='COST':
        for stage,sites in RUNTIME_COUNTS[str(n)].items():
            for site,count in sites.items():
                for _ in range(count):
                    add('L1_FACTORY',site,parent=factory if stage=='build' else None)
    propagation=add('PROPAGATION',ns=200*10**9)
    for i in range(n):
        body=add('INTERVAL_BODY',parent=propagation,ordinal=i,ns=20*10**9)
        if mode=='COST':
            add(ALPHA_LABEL,ALPHA_SITE,parent=body,ordinal=i,ns=10**6)
    return dict(enabled=False,events_dropped=0,incomplete_scopes=[],records=records)


@pytest.mark.parametrize('n',[2,8])
def test_fixed_counts_reject_lost_or_duplicate_factory_scope(n):
    raw=_raw(n)
    assert validate_cost_records(raw,cost_mode='COST',intervals=n)['alpha']['calls_by_interval']==[1]*n
    lost=copy.deepcopy(raw)
    lost['records'].pop(1)
    with pytest.raises(RuntimeError,match='FIXED_FACTORY_CALL_COUNTS'):
        validate_cost_records(lost,cost_mode='COST',intervals=n)
    duplicate=copy.deepcopy(raw)
    extra=copy.deepcopy(duplicate['records'][1]); extra['token']=10000
    duplicate['records'].append(extra)
    with pytest.raises(RuntimeError,match='FIXED_FACTORY_CALL_COUNTS'):
        validate_cost_records(duplicate,cost_mode='COST',intervals=n)


def test_interference_failure_does_not_rank_or_select():
    cases={}; timings={}
    for name,n,mode in [('OFF2',2,'OFF'),('COST2',2,'COST'),('OFF8_A',8,'OFF'),
                        ('COST8_A',8,'COST'),('COST8_B',8,'COST'),('OFF8_B',8,'OFF')]:
        cases[name]=dict(exact_gate_pass=True,pool_role='X',pointwise_calls=38*n,intervals=n,
            cost_mode=mode,application_service_s=300.,later_body_s=100.)
        timings[name]=dict(merged=dict(clock_closure=dict(pass_=True)),a=dict(enabled=False,records=[]),b=_raw(n,mode))
        timings[name]['merged']['clock_closure']={'pass':True}
    passed=analyze_cost_six(cases,timings.__getitem__)
    assert passed['ranking_allowed'] and passed['cost_ranking']
    cases['COST8_A']['later_body_s']=120.
    failed=analyze_cost_six(cases,timings.__getitem__)
    assert not failed['ranking_allowed'] and not failed['cost_ranking'] and failed['selection'] is None
