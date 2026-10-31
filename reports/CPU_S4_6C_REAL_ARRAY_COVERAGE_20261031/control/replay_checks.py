"""Negative checks on explicitly derived copies; original evidence stays read-only."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
from replay import replay,load,lines,portable_leaf


def main(root,scratch):
    root=root.resolve();scratch=scratch.resolve()
    assert not scratch.exists() and scratch.parent.is_dir()
    shutil.copytree(root,scratch)
    spec=importlib.util.spec_from_file_location('negative_test_pv',root/'execution_sources/provenance_v2.py')
    pv=importlib.util.module_from_spec(spec);spec.loader.exec_module(pv)
    path=scratch/'raw/TIMINGS.jsonl';original=path.read_bytes()
    manifest=scratch/'EXPORT_MANIFEST.json';original_manifest=manifest.read_bytes()
    results=[]
    def reject(name,expected,action):
        undo=action()
        try:
            try:replay(scratch)
            except (ValueError,KeyError,FileNotFoundError,TypeError) as e:
                message=str(e);assert expected in message,(name,message)
                results.append(dict(case=name,status='EXPECTED_REJECTION',reason=message))
            else:raise AssertionError('FALSE_PASS:'+name)
        finally:undo()
    def missing(relative):
        p=scratch/relative;payload=p.read_bytes();p.unlink()
        return lambda:p.write_bytes(payload)
    reject('completion_missing','MISSING_MECHANISM_COMPLETION',lambda:missing('raw/MECHANISM_COMPLETED.json'))
    exact=next((scratch/'raw/exact').glob('*_FRESH_STAGE.jsonl'))
    reject('exact_missing',exact.name,lambda:missing(exact.relative_to(scratch)))
    def changed(transform):
        rows=lines(path);transform(rows)
        payload=('\n'.join(json.dumps(x) for x in rows)+'\n').encode();path.write_bytes(payload)
        value=load(manifest)
        for record in value['records']:
            if record['path']=='raw/TIMINGS.jsonl':record.update(bytes=len(payload),raw_sha256=pv.raw_sha256(payload))
        manifest.write_text(json.dumps(value),encoding='utf-8')
        def undo():path.write_bytes(original);manifest.write_bytes(original_manifest)
        return undo
    reject('duplicate','DUPLICATE_TIMING',lambda:changed(lambda r:r.append(r[0])))
    reject('declared_clock','VERSION_CLOCK',lambda:changed(lambda r:r[0].update(clock='NANOSECONDS')))
    reject('out_of_order','PREREGISTERED_TIMING_ORDER',lambda:changed(lambda r:r.reverse()))
    reject('nonfinite','NONFINITE_JSON',lambda:changed(lambda r:r[0].update(wall_s=float('nan'))))
    def wrong_worker(rows):
        next(x for x in rows if x['mode']=='P')['target_activity'][0]['actual_workers']=0
    reject('worker_false_pass','MEASURED_TARGET_WORKERS',lambda:changed(wrong_worker))
    assert portable_leaf(r'C:\evidence\exact\example.jsonl')==portable_leaf('/evidence/exact/example.jsonl')=='example.jsonl'
    results.append(dict(case='Windows_POSIX_leaf',status='PASS'))
    print(json.dumps(dict(status='PASS_NEGATIVE_REPLAY_CHECKS',original_root=str(root),derived_copy=str(scratch),checks=results),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True)
    a=p.parse_args();main(a.root,a.scratch)
