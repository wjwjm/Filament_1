"""Negative identity and cross-host source-label checks, without science modules."""
import copy,hashlib,json
from pathlib import Path
import pytest
from .runtime_identity import RUNTIME_NAMES,manifest_identities,normalized_runtime_identities,source_label_basename
HEAD='17a6152260fb5b5b8dad44b22f9c20e48a129897'
def digest(value):return hashlib.sha256(value).hexdigest()
@pytest.fixture
def sample(tmp_path):
    folder=tmp_path/'runtime_source';folder.mkdir();payloads={name:('frozen:'+name).encode() for name in RUNTIME_NAMES}
    for name,value in payloads.items():(folder/name).write_bytes(value)
    def document(prefix):
        return dict(schema='filament.provenance.v2',version=2,repository='https://github.com/wjwjm/Filament_1.git',repository_path=prefix+'code',head=HEAD,branch='codex/cpu-opt-eq10-pointwise-20261028',hash_scope='classified_by_record',line_endings={'tracked_create':'canonical-LF-from-Git-blob','tracked_validate':'canonical-LF-worktree-match','external':'raw-bytes'},records=[dict(path=prefix+name,classification='external',hash_scope='raw_bytes',raw_sha256=digest(payloads[name])) for name in sorted(payloads)],created_at_utc='2026-10-28T07:05:54Z')
    class Records:
        index={'schema':'filament.compact_lossless_records.v1','hash_scope':'raw_bytes'}
        def __init__(self):
            self.before=document('/posix/frozen/');self.after=document('/posix/frozen/')
            self.records={name:dict(classification='external',hash_scope='raw_bytes',raw_sha256=digest(value),size_bytes=len(value)) for name,value in payloads.items()}
        def json(self,name):return self.before if name=='EXTERNAL_RUNTIME_PROVENANCE.json' else self.after
        def get(self,name):return payloads[name]
    return tmp_path,Records(),document('C:\\frozen\\')
def check(sample):
    root,records,local=sample
    return normalized_runtime_identities(root,records,local,expected_head=HEAD,sha=digest)
@pytest.mark.parametrize('path',['C:\\frozen\\HPC_DRIVER.py','/posix/frozen/HPC_DRIVER.py','\\\\host\\share\\HPC_DRIVER.py'])
def test_portable_label_basename(path):assert source_label_basename(path)=='HPC_DRIVER.py'
def test_all_five_windows_posix_payload_and_size_identities(sample):assert set(check(sample))==RUNTIME_NAMES
@pytest.mark.parametrize('which',['local','before','after'])
def test_duplicate_basename_rejected_on_both_sides(sample,which):
    root,records,local=sample;doc=local if which=='local' else getattr(records,which)
    doc['records'].append(dict(doc['records'][0],path='/another/'+source_label_basename(doc['records'][0]['path'])))
    with pytest.raises(RuntimeError,match='DUPLICATE_RUNTIME_BASENAME'):check(sample)
@pytest.mark.parametrize('kind',['missing','extra','hash','schema','row_schema','size'])
def test_bad_frozen_identity_rejected(sample,kind):
    root,records,local=sample
    if kind=='missing':local['records'].pop()
    elif kind=='extra':local['records'].append(dict(local['records'][0],path='/extra/UNEXPECTED.py'))
    elif kind=='hash':records.after['records'][0]['raw_sha256']='0'*64
    elif kind=='schema':records.before['schema']='other.schema'
    elif kind=='row_schema':local['records'][0]['unexpected']=True
    else:local['records'][0]['size_bytes']=999999
    with pytest.raises(RuntimeError):check(sample)
def test_wrong_logical_size_rejected(sample):
    sample[1].records['HPC_DRIVER.py']['size_bytes']+=1
    with pytest.raises(RuntimeError,match='SIZE_AND_RAW_BYTES'):check(sample)
def test_payload_content_mismatch_rejected(sample):
    (sample[0]/'runtime_source/HPC_DRIVER.py').write_bytes(b'wrong')
    with pytest.raises(RuntimeError,match='SIZE_AND_RAW_BYTES'):check(sample)
def test_source_label_never_opens_its_claimed_external_location(sample):
    sample[2]['records'][0]['path']='Z:\\unread\\'+source_label_basename(sample[2]['records'][0]['path'])
    assert set(check(sample))==RUNTIME_NAMES
@pytest.mark.parametrize('name',['../HPC_DRIVER.py','C:\\escape\\HPC_DRIVER.py','/escape/HPC_DRIVER.py'])
def test_container_member_allowlist_cannot_escape(sample,name):
    with pytest.raises(RuntimeError,match='TRAVERSAL|ALLOWLIST'):
        normalized_runtime_identities(sample[0],sample[1],sample[2],expected_head=HEAD,sha=digest,expected_names={name})
