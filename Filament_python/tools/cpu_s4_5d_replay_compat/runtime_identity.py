"""Compare frozen external runtime identities without host-dependent path parsing.

Source paths are labels only. File access uses a confined container root and a
separate trusted basename allowlist. Legacy v2 manifests are never rewritten;
their missing size field is checked against the original logical-record index
and actual packaged bytes instead.
"""
import re
from pathlib import Path

RUNTIME_NAMES=frozenset(('RUN.sbatch','HPC_DRIVER.py','bal14_analysis.py','FACTORY_PREFLIGHT.py','SEAL_CASE_ARRAYS.py'))
TOP_FIELDS=frozenset(('schema','version','repository','repository_path','head','branch','hash_scope','line_endings','records','created_at_utc'))
ROW_FIELDS=frozenset(('path','classification','hash_scope','raw_sha256'))
LINE_ENDINGS={'tracked_create':'canonical-LF-from-Git-blob','tracked_validate':'canonical-LF-worktree-match','external':'raw-bytes'}

def require(ok,why):
    if not ok:raise RuntimeError(why)

def source_label_basename(label):
    require(type(label) is str and label and '\x00' not in label,'SOURCE_PATH_LABEL_STRING')
    normalized=label.replace('\\','/');parts=normalized.split('/')
    require('..' not in parts,'SOURCE_PATH_LABEL_TRAVERSAL')
    name=parts[-1]
    require(name not in ('','.','..') and re.fullmatch(r'[A-Za-z0-9_.-]+',name) is not None,'SOURCE_LABEL_BASENAME')
    return name

def manifest_identities(document,*,expected_head,expected_names=RUNTIME_NAMES):
    require(type(document) is dict and set(document)==TOP_FIELDS,'EXTERNAL_RUNTIME_SCHEMA_FIELDS')
    require(document['schema']=='filament.provenance.v2' and type(document['version']) is int and document['version']==2 and document['hash_scope']=='classified_by_record','EXTERNAL_RUNTIME_SCHEMA_VERSION_SCOPE')
    require(document['head']==expected_head and re.fullmatch(r'[0-9a-f]{40}',expected_head) is not None and document['line_endings']==LINE_ENDINGS,'EXTERNAL_RUNTIME_SOURCE_HEADER')
    require(all(type(document[key]) is str and document[key] for key in ('repository','repository_path','branch','created_at_utc')),'EXTERNAL_RUNTIME_METADATA_TYPES')
    require(type(document['records']) is list,'EXTERNAL_RUNTIME_RECORD_LIST')
    identities={}
    for row in document['records']:
        require(type(row) is dict and set(row) in (ROW_FIELDS,ROW_FIELDS|{'size_bytes'}),'EXTERNAL_RUNTIME_RECORD_SCHEMA')
        require(row['classification']=='external' and row['hash_scope']=='raw_bytes' and type(row['raw_sha256']) is str and re.fullmatch(r'[0-9a-f]{64}',row['raw_sha256']) is not None,'EXTERNAL_RUNTIME_RECORD_IDENTITY')
        name=source_label_basename(row['path']);require(name not in identities,'DUPLICATE_RUNTIME_BASENAME:'+name)
        if 'size_bytes' in row:require(type(row['size_bytes']) is int and row['size_bytes']>=0,'EXTERNAL_RUNTIME_SIZE_TYPE')
        identities[name]=dict(row)
    require(set(identities)==set(expected_names),'EXACT_RUNTIME_BASENAME_COVERAGE')
    return identities

def normalized_runtime_identities(root,records,local_manifest,*,expected_head,sha,expected_names=RUNTIME_NAMES):
    """Check all three manifests, original logical bytes and packaged payloads."""
    root=Path(root).resolve(strict=True)
    require(records.index['schema']=='filament.compact_lossless_records.v1' and records.index['hash_scope']=='raw_bytes','ORIGINAL_LOGICAL_RECORD_SCHEMA')
    names=frozenset(expected_names)
    require(all(type(name) is str and source_label_basename(name)==name for name in names),'TRUSTED_CONTAINER_MEMBER_ALLOWLIST')
    manifests=[manifest_identities(document,expected_head=expected_head,expected_names=names) for document in (local_manifest,records.json('EXTERNAL_RUNTIME_PROVENANCE.json'),records.json('EXTERNAL_RUNTIME_PROTECTION_AFTER.json'))]
    result={}
    for name in sorted(names):
        identity=records.records[name]
        require(identity['classification']=='external' and identity['hash_scope']=='raw_bytes' and type(identity['size_bytes']) is int,'ORIGINAL_RUNTIME_LOGICAL_SCOPE_SIZE')
        logical=records.get(name)
        path=root/'runtime_source'/name
        require(not path.is_symlink() and path.resolve(strict=True).is_relative_to(root) and path.is_file(),'CONFINED_RUNTIME_CONTAINER_MEMBER')
        payload=path.read_bytes()
        require(payload==logical and len(payload)==identity['size_bytes'],'ORIGINAL_RUNTIME_SIZE_AND_RAW_BYTES:'+name)
        digest=sha(payload)
        require(digest==identity['raw_sha256'],'ORIGINAL_RUNTIME_LOGICAL_SHA:'+name)
        for mapping in manifests:
            row=mapping[name]
            require(row['raw_sha256']==digest,'EXTERNAL_RUNTIME_SHA_MISMATCH:'+name)
            require('size_bytes' not in row or row['size_bytes']==len(payload),'EXTERNAL_RUNTIME_SIZE_MISMATCH:'+name)
        result[name]=digest
    return result
