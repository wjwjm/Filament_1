"""Actual new binding and optimized-Python refusals, before scientific start."""
import json,os,subprocess,sys
from pathlib import Path
import pytest
from . import entry

@pytest.mark.parametrize('fault,reason',(('missing','EXECUTION_BINDING_FIELDS'),('duplicate','DUPLICATE_JSON_KEY'),('schema','EXECUTION_BINDING_SCHEMA'),('source','ACTUAL_COMMITTED_EXECUTION_HEAD'),('scope','FULLSIZE_CAPACITY_AND_DOMAIN'),('blob','COMMITTED_SOURCE_BLOB_AND_CANONICAL_IDENTITY')))
def test_new_entry_baseline_fault_refusal_restore(fault,reason,tmp_path):
    original=(Path(os.environ['S44B_REPORT_ROOT'])/'S44B_EXECUTION_BINDING.json').read_bytes();path=tmp_path/'S44B_EXECUTION_BINDING.json';path.write_bytes(original)
    verify=lambda:entry.verify_execution(tmp_path,local_engineering=True,shape=(384,17,19),intervals=2)
    assert verify()['schema']==entry.BINDING_SCHEMA
    value=json.loads(original)
    if fault=='missing':value.pop('source_records')
    elif fault=='schema':value['schema']='unknown'
    elif fault=='source':value['execution_commit']='f'*40
    elif fault=='scope':value['cache_max_bytes']=320*1024**3
    elif fault=='blob':value['source_records'][0]['git_blob_oid']='f'*40
    raw=json.dumps(value)
    if fault=='duplicate':raw=raw[:-1]+',"schema":"unknown"}'
    path.write_text(raw,encoding='utf-8')
    with pytest.raises(RuntimeError,match=reason):verify()
    path.write_bytes(original);assert verify()['schema']==entry.BINDING_SCHEMA

@pytest.mark.parametrize('mode',('flag','environment'))
def test_new_entry_refuses_optimized_python(mode):
    command=[sys.executable,'-s','-B'];environment=os.environ.copy()
    if mode=='flag':command.append('-O')
    else:environment['PYTHONOPTIMIZE']='1'
    command+=['-m','tools.cpu_s4_4b_fullsize_linear_cache.entry','--help']
    result=subprocess.run(command,env=environment,capture_output=True,text=True,timeout=60)
    assert result.returncode!=0 and 'OPTIMIZATION_FORBIDDEN' in result.stderr
