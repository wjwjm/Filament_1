"""S4-6I scalar gates and opt-in W1 entry validation."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from . import entry, hpc
from .cost_analysis import analyze_cost_six, validate_cost_records
from .protocol import CANDIDATE, MICRO_SHAPE, SHAPE, matrix, plan
from .registry import registry_snapshot


def test_fixed_cost_six_domains_and_shared_pools(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    entry.admission(candidate=CANDIDATE,local_engineering=True,shape=MICRO_SHAPE,
                    intervals=2,mode='QUALIFY_BOUNDARY')
    with pytest.raises(RuntimeError,match='LOCAL_MICRO_ONLY'):
        entry.admission(candidate=CANDIDATE,local_engineering=True,shape=SHAPE,
                        intervals=8,mode='PURE_OFF_FULLSIZE')
    assert [row['cost_mode'] for row in matrix()] == ['OFF','COST','OFF','COST','COST','OFF']
    assert [row['intervals'] for row in matrix()] == [2,2,8,8,8,8]
    assert plan()['joint_baseline_both_roles'] and plan()['pool_role']=='X'
    assert plan()['resources']['work_deadline_s']==3480
    assert plan()['pointwise_groups']['8']==dict(INTENSITY_P24=96,EQ10_P48=128,PHASE_ABSORB_P24=16)


def _timing(n,cost,propagation_s=10.):
    rows=[dict(token=1,label='PROPAGATION',parent_token=None,wall_inclusive_ns=int(propagation_s*1e9),context={})]
    for i in range(n):
        body=2+i
        ctx=dict(ordinal=i,interval_index=i,source_site='body')
        rows.append(dict(token=body,label='INTERVAL_BODY',parent_token=1,context=ctx))
        if cost:
            rows.append(dict(token=100+i,label='NONLINEAR_ARRAYS',parent_token=body,
                context=dict(ctx,source_site='propagate.py:690:inten_ion'),
                wall_inclusive_ns=1000,wall_exclusive_ns=1000,status='complete',failed=False))
    return dict(a=dict(enabled=False,records=[]),b=dict(enabled=False,events_dropped=0,
        incomplete_scopes=[],records=rows),merged=dict(clock_closure={'pass': True}))


def _cases():
    cases={}
    timing={}
    for row in matrix():
        cases[row['case']]=dict(row,exact_gate_pass=True,pool_role='X',
                               application_service_s=12.,later_body_s=7.)
        timing[row['case']]=_timing(row['intervals'],row['cost_mode']=='COST')
    return cases,timing


def test_scalar_cost_order_interference_and_original_propagation_denominator():
    cases,timing=_cases()
    result=analyze_cost_six(cases,timing.__getitem__)
    assert result['selection']['unit']=='prep_intensity'
    assert result['cost_ranking'][0]['median_later_exclusive_s']==pytest.approx(7e-6)
    assert result['timing_interference']['propagation']['pairs'][0]['off_s']==10.
    assert result['timing_interference']['body_2_to_8']['pairs'][0]['off_s']==7.
    cases['COST8_A']['application_service_s']=13.3
    result=analyze_cost_six(cases,timing.__getitem__)
    assert not result['ranking_allowed'] and result['selection'] is None
    assert result['classification']=='TIMING_INTERFERENCE_INVALID'


def test_cost_record_validation_rejects_wrong_context_and_off_records():
    raw=_timing(2,True)['b']
    with pytest.raises(RuntimeError,match='OFF_RESIDUAL_RECORDS'):
        validate_cost_records(raw,cost_mode='OFF',intervals=2)
    broken=copy.deepcopy(raw)
    broken['records'][2]['context']['ordinal']=1
    broken['records'][2]['context']['interval_index']=1
    with pytest.raises(RuntimeError,match='OUTSIDE_OWN_BODY'):
        validate_cost_records(broken,cost_mode='COST',intervals=2)
    broken=copy.deepcopy(raw)
    broken['records'][2]['context']['source_site']='unregistered'
    with pytest.raises(RuntimeError,match='UNREGISTERED'):
        validate_cost_records(broken,cost_mode='COST',intervals=2)


@pytest.mark.skipif(not os.environ.get('S46I_REAL_ENTRY_REPORT'),reason='opt-in real W1 entry only')
def test_s46i_real_micro_entry_opt_in():
    report=Path(os.environ['S46I_REAL_ENTRY_REPORT']).resolve(strict=True)
    work=Path(os.environ['S46I_REAL_ENTRY_WORK']).resolve()
    from tools.cpu_s4_3g3_smallgrid import benchmark as sg
    manifest=report/'BENCHMARK_INPUT_MANIFEST.json'
    if not manifest.exists():
        sg.input_manifest(report,shape=MICRO_SHAPE)
    work.mkdir(parents=True,exist_ok=False)
    results={}
    for spec in matrix():
        label,role,n=spec['case'],spec['role'],spec['intervals']
        target=work/label
        argv=[sys.executable,'-s','-B','-m','tools.cpu_s4_6i_residual_parallel.entry',
              '--report',str(report),'--work',str(target),'--candidate',CANDIDATE,
              '--role',role,'--cost-mode',spec['cost_mode'],'--intervals',str(n),
              '--local-engineering','--input-manifest',str(manifest)]
        reference=results.get('OFF2' if n==2 else 'OFF8_A')
        if reference:
            argv += ['--reference-output',reference['output']]
            if n==2:
                argv += ['--reference-boundary',reference['boundary_path'],'--reference-stages',reference['stages']]
        with (work/(label+'.stdout.log')).open('xb') as out, (work/(label+'.stderr.log')).open('xb') as err:
            process=subprocess.run(argv,env=os.environ.copy(),stdout=out,stderr=err)
        assert process.returncode==0, f'{label} exit {process.returncode}; see {work}'
        result=json.loads((target/'CASE_RECEIPT.json').read_text(encoding='utf-8'))
        assert result['execution_domain']=='S46I_LOCAL_MICRO' and result['pool_role']=='X'
        assert result['role']==role and result['pointwise_calls']==30*n
        assert result['route']['math_dispatches']==16*n and result['route']['fallback']==0
        assert result['pointwise_pool']['pool']['closed'] and not result['pointwise_pool']['pool']['active_thread_ids']
        if spec['cost_mode']=='OFF':
            assert result['residual_cost_records']==0
        else:
            assert result['residual_cost_records']>0
        if reference:
            assert result['comparisons']['terminal']['status']=='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
            if n==2:
                assert result['comparisons']['boundary']['status']=='PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT'
                assert result['comparisons']['stages']['status']=='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
        results[label]=result
