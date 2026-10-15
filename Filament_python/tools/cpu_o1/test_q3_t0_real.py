"""Bounded Q3 regression tests; synthetic fixtures only, no scheduler calls."""
from pathlib import Path
from types import SimpleNamespace
import copy,json,os,subprocess,sys,time
import numpy as np
import pytest
from tools.cpu_o1 import q3_t0_real as q3


def fixture_checkpoint(tmp_path):
    shape=(8,2,3)
    fields={k:np.full(shape,(i+1)*1e-9,dtype=np.float64) for i,k in enumerate(q3.FIELDS)}
    metadata={'checkpoint_kind':'NEXT1','branch':'B','completed_pulse':1,'next_pulse':2}
    path=tmp_path/'NEXT1.npz'
    published=q3._publish_checkpoint(path,fields,metadata=metadata,shape=shape)
    return path,fields,metadata,published


def test_q3_checkpoint_roundtrip_and_no_overwrite(tmp_path):
    path,fields,meta,published=fixture_checkpoint(tmp_path)
    disk,receipt=q3._load_checkpoint(path,expected_branch='B',expected_shape=(8,2,3),expected_metadata=meta)
    for k in fields:assert np.array_equal(fields[k].view(np.uint8),disk[k].view(np.uint8))
    assert receipt['metadata']==meta
    with pytest.raises(FileExistsError):q3._publish_checkpoint(path,fields,metadata=meta,shape=(8,2,3))


@pytest.mark.parametrize('defect',['marker','receipt','corrupt','branch','post','pulse','screen','dtype','shape'])
def test_q3_checkpoint_negative(tmp_path,defect):
    path,fields,meta,pub=fixture_checkpoint(tmp_path)
    rp=path.with_name('NEXT1.receipt.json');mp=path.with_name('NEXT1.COMPLETE.json')
    if defect in ('marker','receipt'):
        (mp if defect=='marker' else rp).unlink()
    elif defect=='corrupt':path.write_bytes(path.read_bytes()+b'corrupt')
    else:
        receipt=json.loads(rp.read_text());marker=json.loads(mp.read_text())
        key,value={'branch':('branch','I'),'post':('checkpoint_kind','POST1'),'pulse':('next_pulse',3),
            'screen':('screen_ids',list(reversed(q3.SCREEN_IDS))),'dtype':('field_dtype','float32'),
            'shape':('field_shape',[8,3,2])}[defect]
        receipt[key]=value;rp.write_bytes(q3._json_bytes(receipt))
        marker['receipt_sha256']=q3._sha256_file(rp);mp.write_bytes(q3._json_bytes(marker))
    with pytest.raises(Exception):q3._load_checkpoint(path,expected_branch='B',expected_shape=(8,2,3),expected_metadata=meta)


@pytest.mark.parametrize('defect',['pre0','missing_velocity','wrong_branch','wrong_screen','initial_pre0'])
def test_q3_actual_pre_lineage_negative(tmp_path,defect):
    from tools.cpu_p1r2 import cpu_p1r2_j3 as j3
    _,fields,_,_=fixture_checkpoint(tmp_path);expected=j3._field_hashes(fields)
    candidate={k:v.copy() for k,v in fields.items()};branch='B';initial=False
    if defect=='pre0':candidate={k:np.zeros_like(v) for k,v in fields.items()}
    if defect=='missing_velocity':del candidate['vx']
    if defect=='wrong_branch':branch='I'
    if defect=='wrong_screen':candidate['vy'][3,0,0]+=1e-8
    if defect=='initial_pre0':initial=True
    with pytest.raises(Exception):q3._require_pre_lineage(candidate,case_id='B_P2_CONT',expected_hashes=expected,
        initial_pre0=initial,branch=branch,expected_shape=(8,2,3))


@pytest.mark.parametrize('order',[list(range(7)),[0,1,2,3,4,5,6,6],list(reversed(range(8)))])
def test_q3_screen_order_rejected(order):
    with pytest.raises(q3.Q3Error):q3._validate_screen_ordinals([{'ordinal':x} for x in order])


def test_q3_contract_raw_once_and_missing_identity():
    contract=q3.load_static_contract();assert len(contract['group_map'])==31
    group=next(iter(contract['group_map'].values()));rows=[]
    for spec in q3._expected_field_rows(group):
        row=q3._array_row(field_name=spec['field_name'],reference=np.array([1.]),candidate=np.array([1.]),
            diagnostic_source=spec.get('diagnostic_source'),identity=group['implementation'],case_id=group['case_id'],
            reference_case_id=group['reference_case_id'],comparison_stage=group['comparison_stage'])
        row['scope']=spec['scope'];rows.append(row)
    q3._require_raw_group(group,rows)
    bad=copy.deepcopy(rows);del bad[0]['identity']
    with pytest.raises(q3.Q3Error):q3._require_raw_group(group,bad)
    with pytest.raises(q3.Q3Error):q3._require_raw_group(group,rows+rows[:1])
    with pytest.raises(q3.Q3Error):q3.build_q3_raw_once_gate(contract=contract,group_rows={},identity_checks={},
        linux_environment_eligible=True,resource_budget_respected=True,matrix_status='INCOMPLETE')


@pytest.mark.parametrize('value',[None,float('nan'),float('inf'),{'x':None}])
def test_q3_new_hydro_nonfinite_not_whitelisted(value):
    with pytest.raises(q3.Q3Error):q3._require_finite_json(value,'new_hydro_field')


def test_q3_stage_timeout_cleanup(tmp_path):
    phase=tmp_path/'phase.json'
    q3._set_phase(phase,phase='HYDRO',stage_id='TEST',identity='B',timeout_seconds=.1)
    result=q3._supervise_q3_stage([sys.executable,'-c','import time; time.sleep(30)'],stage_root=tmp_path,
        phase_path=phase,cwd=tmp_path,env=os.environ.copy(),allocation_deadline=time.monotonic()+10)
    assert result['stop'] and result['exit_code'] is not None
    assert (tmp_path/'stdout.log').exists() and (tmp_path/'stderr.log').exists()


def test_q3_storage_cap_and_pre_submission_cli():
    q3._validate_caps(12*2**30,50000)
    with pytest.raises(q3.Q3Error):q3._validate_caps(17*2**30,50000)
    with pytest.raises(SystemExit):q3.main([])


@pytest.mark.parametrize('branch',['B','I'])
def test_q3_real_tiny_two_pulse_independent_reload(tmp_path,branch):
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1
    chain=tmp_path/'chain';reload=tmp_path/'reload'
    env=os.environ.copy();env['UPPE_USE_GPU']='0'
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[2])
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
    base=[sys.executable,'-s','-B','-m','tools.cpu_o1.q3_t0_real','--branch',branch]
    for mode,out,extra in [('chain',chain,[]),('reload',reload,['--checkpoint-path',str(chain/'NEXT1.npz')])]:
        with (tmp_path/(mode+'.stdout.log')).open('xb') as so,(tmp_path/(mode+'.stderr.log')).open('xb') as se:
            proc=subprocess.run(base+['--tiny-mode',mode,'--output-root',str(out)]+extra,env=env,stdout=so,stderr=se,timeout=180)
        assert proc.returncode==0,(tmp_path/(mode+'.stderr.log')).read_text()
    c=json.loads((chain/'tiny_receipt.json').read_text());r=json.loads((reload/'tiny_receipt.json').read_text())
    assert c['pid']!=r['pid'] and c['source_hash']==r['source_hash'] and c['next_hashes']==r['next_hashes']
    assert len(c['hydro'])==8 and all(x['steps']==1000 and x['backend']=='numpy' for x in c['hydro'])
    comparison=j1._compare_stage_files(c['p2'],r['p2'],'Q3_TINY_RELOAD')
    assert comparison['canonical_exact_match'],comparison
    assert c['p2']['receipt']['input_PRE_fields_canonical_sha256']==c['next_hashes']
    assert r['p2']['receipt']['input_PRE_fields_canonical_sha256']==c['next_hashes']

def test_q3_real_window_seam_with_tiny_frozen_fixture(tmp_path,monkeypatch):
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1, cpu_p1r2_j3 as j3
    from tools.cpu_p1r import cpu_p1r_optical as p1r
    context=j1._tiny_context();context["source_hash"]=j1._canonical_hash(context["source"])
    context["assets"]={"pre0_canonical_sha256":j1._canonical_hash(context["pre0"])}
    monkeypatch.setenv("Q3_WORK_DEADLINE_EPOCH",str(time.time()+7200))
    monkeypatch.setattr(p1r,"validate_frozen_source",lambda *a:(context["source"].copy(),context["source_hash"],{"file_sha256":p1r.SOURCE_RAW_SHA256}))
    contract=q3.load_static_contract();groups={}
    args=SimpleNamespace(input_root=tmp_path)
    receipt=q3._run_optical_window(args=args,context=context,identity="B",case_id="B_P1",
        pre_fields=context["pre_fields"],initial_pre0=True,expected_pre_hashes=j3._field_hashes(context["pre_fields"]),
        output_root=tmp_path,phase_path=tmp_path/"phase.json",contract=contract,group_rows=groups)
    assert receipt["status"]=="COMPLETE_DIAGNOSTIC"
    assert receipt["propagation_receipt"]["interval_count"]==8
    assert set(groups)=={"B_P1:first_boundary:B_P1","B_P1:endpoint:B_P1"}
    assert receipt["fresh_source"]["fresh_copy"]
