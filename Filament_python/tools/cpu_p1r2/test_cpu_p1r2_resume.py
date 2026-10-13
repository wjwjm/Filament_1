from types import SimpleNamespace
import time
from pathlib import Path
import pytest
from tools.cpu_p1r2 import cpu_p1r2_resume as r

def test_resume_baseline_only_launches_unexecuted_reload(tmp_path,monkeypatch):
    fields={'delta_n':'d','vx':'x','vy':'y'}
    original={'receipt':{'input_PRE_fields_canonical_sha256':fields}}
    chain={'stage':'chain','status':'DONE','child_root':str(tmp_path/'old_chain'),'child_receipt':{'pulse2_window':original,'NEXT1':{'raw_sha256':'raw','field_canonical_sha256':fields}}}
    args=SimpleNamespace(_verified_source_chain=chain,resume_source_job='8818911')
    calls=[]
    def supervise(args,stage,mode,**kw):
        calls.append((stage,mode,kw));return {'stage':'reload','status':'DONE','supervisor':{'rss_80_percent_latched':False},'child_receipt':{'pulse2_window':original,'NEXT1_input':{'raw_sha256':'raw','field_canonical_sha256':fields,'state_origin':'J3_NEXT1_DISK_RELOADED'}}}
    monkeypatch.setattr(r.j3,'_supervise_child',supervise)
    monkeypatch.setattr(r.j1,'_compare_stage_files',lambda *a:{'canonical_exact_match':True})
    monkeypatch.setattr(r.j3,'_verify_window_pre_lineage',lambda w,f,l:f)
    result=r.resume_baseline(args,'baseline',tmp_path/'new_baseline',candidate_descriptor=None,deadline=time.time()+2000)
    assert result['repeat_exact'] and result['stages'][0] is chain
    assert len(calls)==1 and calls[0][:2]==('reload','reload')
    assert calls[0][2]['extra']['next1_path']==tmp_path/'old_chain/NEXT1_checkpoint.npz'

def test_resume_rejects_different_terminal_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(r.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='8818911|FAILED|2:0|\n'))
    with pytest.raises(RuntimeError,match='FAILED 1:0'):
        r.validate_resume_source(SimpleNamespace(resume_source_root=tmp_path,resume_source_job='8818911'))