"""Reject a lost clock, an inactive-chain baseline, or overlapping ranking."""
import copy
import pytest
from . import cost_analysis as ca
from .protocol import plan, matrix

def test_measurement_scope_is_fixed_and_stops():
    p=plan()
    assert len(matrix())==6 and p['normal_study_jobids_max']==1
    assert p['shape']==[384,351,301] and p['resources']['gpus']==0
    assert p['resources']['work_deadline_s']==3480
    assert len(p['components'])==5 and 'ION_ALPHA_CLEANUP_P24' in p['components']
    assert p['pointwise_groups']['8']['ION_ALPHA_CLEANUP_P24']==40
    assert p['acceptance']['ranking_requires_all_interference_gates']
    assert not p['ranking']['automatic_candidate_implementation']

def test_cost_record_rejects_loss_and_off_scope():
    raw=dict(enabled=False,events_dropped=1,incomplete_scopes=[],records=[])
    with pytest.raises(RuntimeError,match='SCALAR_RECORD_LIFECYCLE'):
        ca.validate_cost_records(raw,cost_mode='OFF',intervals=2)
    raw['events_dropped']=0
    raw['records']=[dict(token=1,label='NONLINEAR_ARRAYS',context={'source_site':'x'})]
    with pytest.raises(RuntimeError,match='COST_ON_OFF_RECORDS'):
        ca.validate_cost_records(raw,cost_mode='OFF',intervals=2)

def test_eligible_parent_child_cost_is_rejected(monkeypatch):
    sites=[dict(name=x,unit=x,source_site=x,source_order=i,eligible=True,scope_kind='eligible_scope')
        for i,x in enumerate(('outer','inner'))]
    monkeypatch.setattr(ca,'registry_snapshot',lambda:dict(sites=sites,scope_label='NONLINEAR_ARRAYS'))
    body=dict(token=1,label='INTERVAL_BODY',context={'ordinal':0},parent_token=None)
    def row(token,site,parent):
        return dict(token=token,label='NONLINEAR_ARRAYS',parent_token=parent,
            context=dict(source_site=site,ordinal=0,interval_index=0),status='complete',
            failed=False,wall_inclusive_ns=100,wall_exclusive_ns=50)
    raw=dict(enabled=False,events_dropped=0,incomplete_scopes=[],records=[body,row(2,'outer',1),row(3,'inner',2)])
    with pytest.raises(RuntimeError,match='PARENT_CHILD_DOUBLE_COUNT'):
        ca.validate_cost_records(raw,cost_mode='COST',intervals=2)

def test_old_four_chain_case_cannot_become_new_reference():
    rows={row['case']:dict(exact_gate_pass=True,pool_role='X',alpha_enabled=False) for row in matrix()}
    with pytest.raises(RuntimeError,match='FIVE_CHAIN_SIX_EXACT'):
        ca.analyze_cost_six(rows,lambda name:None)

def test_interference_failure_prevents_ranking(monkeypatch):
    rows={row['case']:dict(exact_gate_pass=True,pool_role='X',alpha_enabled=True,
        intervals=row['intervals'],pointwise_calls=43*row['intervals'],
        alpha_rounds=5*row['intervals'],cost_mode=row['cost_mode'],
        application_service_s=112 if row['case']=='COST8_A' else 100,
        later_body_s=100) for row in matrix()}
    monkeypatch.setattr(ca,'validate_cost_records',lambda *args,**kwargs:dict(source_detail={}))
    def raw(name):
        return dict(merged=dict(clock_closure={'pass':True}),a=dict(enabled=False,records=[]),
            b=dict(records=[dict(label='PROPAGATION',wall_inclusive_ns=100_000_000_000)]))
    result=ca.analyze_cost_six(rows,raw)
    assert not result['ranking_allowed'] and result['cost_ranking']==[]
    assert result['selection'] is None and result['stop_after_this_measurement']
    assert result['classification']=='TIMING_INTERFERENCE_INVALID'

def test_runtime_reference_requires_alpha_pool():
    from tools.cpu_s4_6j_alpha_factory.test_alpha_entry import _four_pool_case
    from . import hpc
    row=_four_pool_case()
    with pytest.raises(RuntimeError,match='ALPHA_POOL_SNAPSHOT_MISSING'):
        hpc._exact(row,terminal=True,boundary=False,stages=False)
