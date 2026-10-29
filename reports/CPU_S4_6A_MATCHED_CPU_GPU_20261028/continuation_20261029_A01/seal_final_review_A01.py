"""Seal the one final S46A portable package; original evidence stays intact."""
from __future__ import annotations
import csv, html, json, shutil, sys, time, zipfile
from pathlib import Path, PurePosixPath

repo=Path(__file__).resolve().parents[3]
r=repo/'reports/CPU_S4_6A_MATCHED_CPU_GPU_20261028'
c=r/'continuation_20261029_A01'
out=r/'CPU_S4_6A_REVIEW_20261029_002'
archive=r/'CPU_S4_6A_REVIEW_20261029_002.zip'
sys.path.insert(0,str(repo/'Filament_python'))
from tools.hpc_ops import provenance_v2 as pv

def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,value):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def copy(src,dst):
    if dst.exists():
        require(pv.raw_sha256_file(src)==pv.raw_sha256_file(dst),'SEAL_COPY_WOULD_OVERWRITE:'+str(dst))
        return
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
def require(ok,reason):
    if not ok:raise RuntimeError(reason)

def main():
    started=time.perf_counter()
    copy(c/'validate_final_delivery_A01.py',out/'validate_final_delivery_A01.py')
    validation=[
        'WINDOWS_DELIVERY_COMPILE_CAPTURE_A02.json','WINDOWS_DELIVERY_BACKEND_A01.log','WINDOWS_DELIVERY_SANITY_A01.log',
        'WINDOWS_DELIVERY_TARGETED_A02.log','WINDOWS_DELIVERY_TARGETED_A02.xml',
        'LINUX_DELIVERY_BACKEND_A02.log','LINUX_DELIVERY_TARGETED_A03.log','LINUX_REPLAY_FINAL_PORTABLE_A03.xml',
        'WINDOWS_DELIVERY_REPLAY_PRESEAL_A03.json','LINUX_DELIVERY_REPLAY_A02.json','LINUX_DELIVERY_REPLAY_EXECUTION_A02.json',
        'FINAL_TABLES_FROM_FULL_SCOPE_A01.log',
    ]
    for name in validation:copy(c/name,out/'local_validation'/name)
    for name in ('LINUX_DELIVERY_TARGETED_A01.log','LINUX_DELIVERY_CONTEXT_A01.json','ZIP_SEAL_A01.log'):
        copy(c/name,out/'failure_history'/name)
    write(out/'local_validation/VALIDATION_SCOPE.json',dict(
        status='WINDOWS_AND_REAL_LINUX_PASS_EXPECTED_LIMITED_REPLAY',
        Windows=dict(backend='PASS',sanity='1 passed',replay_tests='21 passed',
            raw_metadata_replay='PASS_EXPECTED_LIMITED_REPLAY'),
        Linux=dict(backend='PASS',replay_tests='21 passed',raw_metadata_replay='PASS_EXPECTED_LIMITED_REPLAY',
            interpreter='/mnt/d/Filament_1/.worktrees/cpu_optimization/reports/CPU_S4_3B_FIXTURE_REPAIR_20261020_090443/linux/venv/bin/python',
            native_filesystem='/var/tmp/s46a-review-final-20261029-jm_rd1bp'),
        prior_failures='Prior /mnt/c slow run terminated (-15), not PASS; first /tmp staged copy vanished between invocations, preserved as failed validation',
        scope='Replay reconstructs ratios, integer-ns ledgers, expected full GPU LIGHT failure, inputs/source metadata and all-eight-job budget; no scientific arrays',
        final_zip_validation='External ZIP_FINAL_RECEIPT.json records sealed artifact hashes and final Windows/Linux replay to avoid self-reference',
        DEFAULT='I0',PRODUCTION_QUALIFIED=False,MANUAL_REVIEW_PENDING=True))
    write(out/'INDEPENDENT_REVIEW_A01.json',dict(
        role='filament_numerical_reviewer',mode='Read-only integrated report/contract/table review; no new propagation or tests',
        result='No blocking scientific or accounting defect reported after parent integration',
        checked=['Full GPU LIGHT failure retained','Median main ratios vs additive independent means',
            'Nonoverlap and CPU exclusive closure','No async-host/device mixed delta',
            'Unique stable-energy candidate with original NumPy sum preserved',
            'No cross-backend equivalence or hardware-limit claim'],
        corrected_preliminary_evidence=['Earlier wrong RAW layer claim of missing ION labels withdrawn',
            'No-stage RAMAN_ENERGY leaf differs from all-label parent/exclusive sum'],
        unverified=['Native full GPU operator attribution','Cross-backend scientific equivalence','Unimplemented next candidate']))
    # Identity and integrity of the byte-identical scheduler-name support copies.
    support=[]
    for job,name in [('290147','GPU_STATUS_A01.json'),('290163','GPU_REPAIR_STATUS_A01.json'),('290181','GPU_ADDITIONAL_STATUS_A02.json')]:
        old=out/'failure_history'/name;new=out/'failure_history'/('GPU_'+job+'_SCHEDULER_ORIGINAL.json')
        digest=pv.raw_sha256_file(old);require(pv.raw_sha256_file(new)==digest,'SCHEDULER_COPY_BYTES')
        support.append(dict(original=old.relative_to(out).as_posix(),copy=new.relative_to(out).as_posix(),
                            classification='external',hash_scope='raw_bytes',raw_sha256=digest))
    write(out/'SCHEDULER_SUPPORT_COPY_HASHES.json',dict(records=support,
        reason='Existing name-limited ledger reader; originals preserved, each main JobID charged once'))
    result=load(out/'MEASURED_RESULT_SUMMARY.json')
    ratios=result['cpu_gpu_ratios']['ratios']
    ratio_rows=''.join('<tr><td>'+html.escape(v['shape'].replace('shape:',''))+'</td><td>'+name+'</td><td>'+format(m['cpu_off_median_s'],'.6f')+'</td><td>'+format(m['gpu_off_median_s'],'.6f')+'</td><td>'+format(m['cpu_over_gpu'],'.6f')+'</td></tr>'
        for v in ratios for name,m in v['metrics'].items())
    links=[('中文主报告','MANUAL_REVIEW_REPORT.md'),('唯一下一CPU合同','NEXT_CPU_OPTIMIZATION_CONTRACT.md'),
           ('原始时钟与三次OFF','MATCHED_RUNTIME_TABLE.csv'),('互斥宏观差距表','GAP_CONTRIBUTIONS.csv'),
           ('受限PROFILE诊断','PROFILE_DIAGNOSTIC_COSTS.csv'),('资源账','RESOURCE_LEDGER.csv'),
           ('逐字段数值差异索引','CROSS_BACKEND_NUMERIC_DIFFERENCES.json'),('同后端exact索引','BACKEND_TRANSPARENCY_EXACT_INDEX.json'),
           ('实际执行身份','SOURCE_AND_BACKEND_BINDINGS.json'),('离线重放说明','REPLAY_INSTRUCTIONS.md')]
    content='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CPU-S4-6A 同工况耗时审阅</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1050px;margin:40px auto;padding:0 24px;color:#1b283a;background:#fbfcfe}h1{font-size:28px}.status{padding:16px;background:#fff4da;border-left:5px solid #d19b2a}table{border-collapse:collapse;width:100%;margin:24px 0}td,th{text-align:left;border-bottom:1px solid #d9e1eb;padding:9px}th{background:#eaf0f7}a{color:#164f91}li{margin:7px 0}code{background:#eaf0f7;padding:2px 5px}</style><h1>CPU-S4-6A 同工况 CPU/GPU 光学耗时</h1><p class="status"><b>TOTAL_GAP_MEASURED / DETAIL_ATTRIBUTION_LIMITED</b><br>原尺寸双端已完成。OFF总耗时有效；GPU原尺寸LIGHT干扰门失败，算子级差距仍UNKNOWN。DEFAULT=I0 · PRODUCTION_QUALIFIED=false · MANUAL_REVIEW_PENDING</p><p>原尺寸差距主要在第2–8区间（应用均值差的84.31%）。历史约27倍口径不匹配；当前完整传播窗口独立三次OFF中位之比为28.814954。</p><table><tr><th>shape</th><th>范围</th><th>CPU中位秒</th><th>GPU中位秒</th><th>CPU/GPU</th></tr>'''+ratio_rows+'''</table><p>矩阵P编号只在各后端内部检查LIGHT干扰，不构成跨系统逐轮配对。主比值使用中位；宏观加和分账使用独立均值。科学数组留原端，本包没有数组或环境。</p><ul>'''+''.join('<li><a href="'+html.escape(path)+'">'+html.escape(label)+'</a></li>' for label,path in links)+'''</ul><p>离线执行：<code>python -s -B replay_readonly.py --root .</code>。预期忠实返回有限终态，不能把GPU LIGHT失败改为PASS。Windows及真实Linux各21测试及元数据重放记录见local_validation/。</p></html>'''
    with (out/'REVIEW_INDEX.html').open('x',encoding='utf-8',newline='\n') as f:f.write(content)
    required=['MANUAL_REVIEW_REPORT.md','WORKLOAD_MATCH_CONTRACT.json','SOURCE_AND_BACKEND_BINDINGS.json',
        'SEMANTIC_DIFFERENCE_AUDIT.md','HISTORICAL_RATIO_RECONCILIATION.md','MEASUREMENT_PLAN.json',
        'CLOCK_BOUNDARY_CONTRACT.md','MATCHED_RUNTIME_TABLE.csv','GAP_CONTRIBUTIONS.csv',
        'PARALLELISM_TRANSFER_AND_MEMORY.csv','DIRECTED_SOURCE_TABLE.csv','BACKEND_TRANSPARENCY_EXACT_INDEX.json',
        'CROSS_BACKEND_NUMERIC_DIFFERENCES.json','NEXT_CPU_OPTIMIZATION_CONTRACT.md','RESOURCE_LEDGER.csv','replay_readonly.py']
    require(all((out/name).is_file() for name in required),'REQUIRED_ARTIFACT_MISSING')
    # The delivery manifest participates in replay; it excludes only itself.
    scope=load(out/'REPLAY_SCOPE.json');scope['manifests'].append('PACKAGE_METADATA_MANIFEST.json')
    (out/'REPLAY_SCOPE.json').write_text(json.dumps(scope,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    forbidden={'.npy','.npz','.mat','.h5','.hdf5','.zip','.gz','.bundle','.pyc'}
    records=[]
    for p in sorted(out.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(out).as_posix()
        require(not p.is_symlink() and p.suffix.lower() not in forbidden and '__pycache__' not in p.parts,'FORBIDDEN_PACKAGE_MEMBER:'+rel)
        records.append(dict(path='./'+rel if '/' not in rel else rel,classification='external',hash_scope='raw_bytes',
                            size_bytes=p.stat().st_size,raw_sha256=pv.raw_sha256_file(p)))
    write(out/'PACKAGE_METADATA_MANIFEST.json',dict(schema='filament.cpu_s46a.delivery_bytes.v1',hash_scope='raw_bytes',
        purpose='Exact delivery bytes only; original PV2 blob/canonical-LF identities remain separately verified',records=records))
    print(json.dumps(dict(stage='ZIP_COMPRESSION_START',files=len(records)+1,uncompressed_bytes=sum(x['size_bytes'] for x in records))),flush=True)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_LZMA) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out).as_posix())
    expected={row['path'].removeprefix('./'):row['raw_sha256'] for row in records}
    expected['PACKAGE_METADATA_MANIFEST.json']=pv.raw_sha256_file(out/'PACKAGE_METADATA_MANIFEST.json')
    print(json.dumps(dict(stage='ZIP_RAW_HASH_AND_CRC_VERIFICATION',size_bytes=archive.stat().st_size)),flush=True)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();require(len(names)==len(set(names))==len(set(n.casefold() for n in names)),'ZIP_DUPLICATE_PATH')
        require(set(names)==set(expected),'ZIP_MEMBER_SET')
        for name in names:
            path=PurePosixPath(name)
            require(not path.is_absolute() and '..' not in path.parts and '\\' not in name,'ZIP_UNSAFE_PATH')
            # Reading each whole member validates ZIP CRC and exact raw hash once.
            require(pv.raw_sha256(z.read(name))==expected[name],'ZIP_RAW_HASH:'+name)
    require(archive.stat().st_size<=20*1024**2,'ZIP_20MIB_CAP')
    receipt=dict(status='ZIP_BYTES_AND_CRC_PASS',path=str(archive),size_bytes=archive.stat().st_size,
        raw_sha256=pv.raw_sha256_file(archive),hash_scope='raw_bytes',compression='ZIP_LZMA',members=len(expected),
        uncompressed_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
        CRC='PASS_ALL_MEMBERS_READ',all_member_raw_sha256='PASS',scientific_arrays=0,nested_archives=0,
        wall_s=time.perf_counter()-started,conclusion='TOTAL_GAP_MEASURED / DETAIL_ATTRIBUTION_LIMITED',
        analysis_report_tip='Recorded after commit in external GIT_FINAL_DELIVERY_RECEIPT.json',
        DEFAULT='I0',PRODUCTION_QUALIFIED=False,MANUAL_REVIEW_PENDING=True)
    write(c/'ZIP_BUILD_RECEIPT_A01.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
