"""Restore this S46A review's byte-identical duplicate metadata files."""
from __future__ import annotations
import argparse, json, runpy, shutil, time
from pathlib import Path, PurePosixPath

def require(ok,reason):
    if not ok:raise RuntimeError(reason)
def confined(root,value):
    p=PurePosixPath(value)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in value,'UNSAFE_METADATA_PATH')
    result=root.joinpath(*p.parts)
    require(result.resolve().is_relative_to(root.resolve()),'METADATA_PATH_OUTSIDE_ROOT')
    return result

def main():
    parser=argparse.ArgumentParser(description='Restore the original S46A metadata review to a new folder')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=Path(__file__).resolve().parent
    index=json.loads((base/'LOGICAL_FILE_INDEX.json').read_text(encoding='utf-8'))
    require(index['schema']=='filament.cpu_s46a.lossless_metadata_dedup.v1','DEDUP_SCHEMA')
    output=args.output.resolve();require(not output.exists(),'RESTORE_OUTPUT_MUST_BE_NEW')
    require(not output.is_relative_to(base/'payload'),'RESTORE_OUTPUT_IN_PAYLOAD')
    # Original, packaged PV2 implementation; no replacement hash semantics.
    pv=runpy.run_path(str(confined(base,index['hash_implementation_payload'])))
    records=index['records'];names=[row['logical_path'] for row in records]
    require(len(names)==len(set(names))==len(set(x.casefold() for x in names)),'DUPLICATE_LOGICAL_PATH')
    started=time.perf_counter();verified={}
    for row in records:
        source=confined(base,row['payload_path'])
        key=row['payload_path']
        if key not in verified:
            require(source.is_file() and not source.is_symlink(),'RESTORE_SOURCE_REQUIRED')
            verified[key]=pv['raw_sha256_file'](source)
        require(verified[key]==row['raw_sha256'] and source.stat().st_size==row['size_bytes'],'RESTORE_SOURCE_BYTES')
        confined(output,row['logical_path'])
    output.mkdir(parents=True,exist_ok=False)
    for row in records:
        source=confined(base,row['payload_path']);target=confined(output,row['logical_path'])
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        require(pv['raw_sha256_file'](target)==row['raw_sha256'],'RESTORED_LOGICAL_FILE_HASH:'+row['logical_path'])
    receipt=dict(status='PASS_ALL_LOGICAL_FILES_RESTORED_RAW_BYTES',output=str(output),
        logical_files=len(records),unique_payload_files=len(verified),
        logical_bytes=sum(row['size_bytes'] for row in records),wall_s=time.perf_counter()-started,
        hash_scope='raw_bytes',hash_implementation='Original packaged provenance_v2.py',
        scientific_arrays=0,source_originals_modified=False)
    receipt_path=output.with_name(output.name+'_RESTORE_RECEIPT.json')
    with receipt_path.open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
