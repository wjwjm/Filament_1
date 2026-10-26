"""Causal metadata/source admission checks; no scientific reruns."""
import copy,json,os,shutil,subprocess,sys
from pathlib import Path
import pytest
from . import entry

@pytest.fixture
def report(tmp_path):
    origin=Path(os.environ['S43SG_REPORT_ROOT']);target=tmp_path/'report';target.mkdir()
    for name in ('EXPERIMENT_PLAN.json','EXECUTION_BINDING.json'):shutil.copyfile(origin/name,target/name)
    return target

@pytest.mark.parametrize('fault,reason',(
    ('missing','EXECUTION_BINDING_FIELDS'),('extra','EXECUTION_BINDING_FIELDS'),
    ('duplicate','DUPLICATE_JSON_KEY'),('schema','EXECUTION_BINDING_SCHEMA'),
    ('plan_identity','EXPERIMENT_PLAN_RAW_IDENTITY'),('source_head','ACTUAL_COMMITTED_EXECUTION_HEAD'),
    ('source_blob','COMMITTED_SOURCE_BLOB_AND_CANONICAL_IDENTITY'),('source_canonical','COMMITTED_SOURCE_BLOB_AND_CANONICAL_IDENTITY'),
))
def test_entry_binding_baseline_reject_and_restore(fault,reason,report):
    path=report/'EXECUTION_BINDING.json';original=path.read_bytes()
    call=lambda:entry.verify_execution(report,local_engineering=True,shape=(384,17,19),intervals=2)
    assert call()['schema']==entry.BINDING_SCHEMA
    value=json.loads(original)
    if fault=='missing':value.pop('source_records')
    elif fault=='extra':value['unapproved_field']=0
    elif fault=='schema':value['schema']='unknown'
    elif fault=='plan_identity':value['experiment_plan_raw_sha256']='0'*64
    elif fault=='source_head':value['execution_commit']='f'*40
    elif fault=='source_blob':value['source_records'][0]['git_blob_oid']='f'*40
    elif fault=='source_canonical':value['source_records'][0]['canonical_lf_sha256']='0'*64
    raw=json.dumps(value)
    if fault=='duplicate':raw=raw[:-1]+',"schema":"'+entry.BINDING_SCHEMA+'"}'
    path.write_text(raw,encoding='utf-8')
    with pytest.raises(RuntimeError,match=reason):call()
    path.write_bytes(original);assert call()['schema']==entry.BINDING_SCHEMA

@pytest.mark.parametrize('optimized',('flag','environment'))
def test_entry_refuses_optimized_python_before_science(optimized):
    command=[sys.executable,'-s','-B'];environment=os.environ.copy()
    if optimized=='flag':command.append('-O')
    else:environment['PYTHONOPTIMIZE']='1'
    command.extend(('-m','tools.cpu_s4_4a_linear_cache.entry','--help'))
    result=subprocess.run(command,env=environment,capture_output=True,text=True,timeout=60)
    assert result.returncode!=0 and 'SMALLGRID_OPTIMIZATION_FORBIDDEN' in result.stderr

def test_hpc_cli_uses_real_entry_module():
    result=subprocess.run([sys.executable,'-s','-B','-m','tools.cpu_s4_4a_linear_cache.entry','--help'],capture_output=True,text=True,timeout=60)
    assert result.returncode==0 and 'L1_LINEAR_FACTOR_CACHE' in result.stdout and '--reference-boundary' in result.stdout
