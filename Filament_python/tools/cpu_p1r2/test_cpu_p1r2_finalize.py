import json
import numpy as np
import pytest
from tools.cpu_p1r2 import cpu_p1r2_finalize as f
from tools.cpu_p1r2 import cpu_p1r2_j1 as j1
from tools.hpc_ops.provenance_v2 import raw_sha256_file

def fixture_window(root):
    root.mkdir();shape=(4,3,2)
    np.save(root/'final_E.npy',np.ones(shape,dtype=np.complex128),allow_pickle=False)
    post={k:np.zeros((8,3,2),dtype=np.float64) for k in ('delta_n','vx','vy')}
    np.savez(root/'final_POST.npz',**post);np.savez(root/'diagnostics.npz',U_z=np.ones(8))
    r={'status':'COMPLETE_DIAGNOSTIC','interval_count':8,'interval_rows':[{'interval_index':i} for i in range(8)],'source_canonical_sha256':'smoke-source','input_PRE_fields_canonical_sha256':{k:j1._canonical_hash(v) for k,v in post.items()}}
    for name,ext in [('final_E','npy'),('final_POST','npz'),('diagnostics','npz')]:
        path=root/(name+'.'+ext);r[name+'_raw_sha256']=raw_sha256_file(path)
        if ext=='npy':r[name+'_canonical_sha256']=j1._canonical_hash(np.load(path,allow_pickle=False))
        else:
            with np.load(path,allow_pickle=False) as z:r[name+'_canonical_sha256']={k:j1._canonical_hash(z[k]) for k in z.files}
    p={'source_canonical_sha256':r['source_canonical_sha256'],'input_PRE_fields_canonical_sha256':r['input_PRE_fields_canonical_sha256'],'output_raw_sha256':{k:r[k+'_raw_sha256'] for k in ('final_E','final_POST','diagnostics')}}
    (root/'child_receipt.json').write_text(json.dumps(r));(root/'j3_provenance_receipt.json').write_text(json.dumps(p))
    return shape

def test_recovery_window_reconstructs_without_mutation(tmp_path):
    root=tmp_path/'window';shape=fixture_window(root)
    before={p.name:raw_sha256_file(p) for p in root.iterdir()}
    result=f.verify_window(root,expected_shape=shape,production=False)
    assert result['status']=='VALIDATED' and result['receipt']['interval_count']==8
    assert before=={p.name:raw_sha256_file(p) for p in root.iterdir()}
    with pytest.raises(RuntimeError,match='not real J3 input'):f.verify_window(root,expected_shape=shape)

def test_recovery_rejects_changed_array_and_incomplete_window(tmp_path):
    root=tmp_path/'window';shape=fixture_window(root)
    with (root/'final_E.npy').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(RuntimeError,match='raw hash mismatch'):f.verify_window(root,expected_shape=shape,production=False)
    r=json.loads((root/'child_receipt.json').read_text());r['interval_count']=7;(root/'child_receipt.json').write_text(json.dumps(r))
    with pytest.raises(RuntimeError,match='incomplete optical'):f.verify_window(root,expected_shape=shape,production=False)
def test_recovery_rejects_unrelated_source_job():
    f.require_source_allocation({'allocation':{'SLURM_JOB_ID':'8818911'}},'8818911')
    with pytest.raises(RuntimeError,match='source JobID'):
        f.require_source_allocation({'allocation':{'SLURM_JOB_ID':'other'}},'8818911')
def test_production_pulse1_cannot_relabel_itself_non_pre0(tmp_path):
    root=tmp_path/'window';shape=fixture_window(root)
    rp=root/'child_receipt.json';pp=root/'j3_provenance_receipt.json'
    r=json.loads(rp.read_text());p=json.loads(pp.read_text())
    r.update(execution_scope='J3_REAL_INPUT',source_canonical_sha256=f.p1r.SOURCE_ARRAY_SHA256,pre0_expected_canonical_sha256=None,pre0_canonical_sha256=None)
    p.update(source_canonical_sha256=f.p1r.SOURCE_ARRAY_SHA256,source_raw_sha256=f.p1r.SOURCE_RAW_SHA256,transfer_mapping_sha256=f.p1r.MAPPING_SHA256,pre0_raw_sha256=None)
    rp.write_text(json.dumps(r));pp.write_text(json.dumps(p))
    with pytest.raises(RuntimeError,match='frozen PRE0 identity'):
        f.verify_window(root,expected_shape=shape,production=True,initial_pre0=True)