"""Task-allowed lossless dedup of the final review's repeated metadata."""
from __future__ import annotations
import json,sys,time,zipfile
from pathlib import Path

repo=Path(__file__).resolve().parents[3]
r=repo/'reports/CPU_S4_6A_MATCHED_CPU_GPU_20261028';c=r/'continuation_20261029_A01'
original=r/'CPU_S4_6A_REVIEW_20261029_002'
archive=r/'CPU_S4_6A_REVIEW_20261029_003.zip'
sys.path.insert(0,str(repo/'Filament_python'))
from tools.hpc_ops import provenance_v2 as pv

def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def require(ok,reason):
    if not ok:raise RuntimeError(reason)

def main():
    start=time.perf_counter();seen={};rows=[];payload={}
    source_manifest=json.loads((original/'PACKAGE_METADATA_MANIFEST.json').read_text(encoding='utf-8'))
    original_rows=source_manifest['records']+[dict(path='PACKAGE_METADATA_MANIFEST.json',
        raw_sha256=pv.raw_sha256_file(original/'PACKAGE_METADATA_MANIFEST.json'),
        size_bytes=(original/'PACKAGE_METADATA_MANIFEST.json').stat().st_size)]
    for row in sorted(original_rows,key=lambda x:x['path'].removeprefix('./')):
        logical=row['path'].removeprefix('./');digest=row['raw_sha256']
        require(pv.raw_sha256_file(original/logical)==digest,'ORIGINAL_REVIEW_BYTES_CHANGED:'+logical)
        if digest not in seen:
            seen[digest]='payload/'+logical;payload[seen[digest]]=original/logical
        rows.append(dict(logical_path=logical,payload_path=seen[digest],classification='external',
                         hash_scope='raw_bytes',raw_sha256=digest,size_bytes=row['size_bytes']))
    helper_logical='source_roles/CPU/executed_source/Filament_python/tools/hpc_ops/provenance_v2.py'
    index=dict(schema='filament.cpu_s46a.lossless_metadata_dedup.v1',hash_scope='raw_bytes',records=rows,
        purpose='Lossless duplicate-file storage only; restore every original logical path and hash before replay',
        hash_implementation_payload=next(x['payload_path'] for x in rows if x['logical_path']==helper_logical),
        original_logical_files=len(rows),unique_payload_files=len(payload),
        original_manifest_raw_sha256=pv.raw_sha256_file(original/'PACKAGE_METADATA_MANIFEST.json'),
        scientific_arrays=0,DEFAULT='I0',PRODUCTION_QUALIFIED=False,MANUAL_REVIEW_PENDING=True)
    index_bytes=(json.dumps(index,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    readme='''# CPU-S4-6A 最终审阅包\n\n终态：TOTAL_GAP_MEASURED / DETAIL_ATTRIBUTION_LIMITED。原尺寸应用/传播窗口/later CPU÷GPU为7.014243/28.814954/34.976333。GPU原尺寸LIGHT干扰失败保留。\n\n本ZIP按任务第11.2节无损去重，仅把完全相同的元数据文件存一份；全部原始逻辑文件、hash和恢复映射在LOGICAL_FILE_INDEX.json。没有科学数组或嵌套历史ZIP。\n\n先可直接阅读payload/MANUAL_REVIEW_REPORT.md。完整离线复算前，在解压目录执行（Python标准库）：\n\n```text\npython -s -B restore_review.py --output ./restored_review\npython -s -B ./restored_review/replay_readonly.py --root ./restored_review\n```\n\n恢复器只创建新目录，拒绝覆盖；使用原packaged provenance_v2.py核验每份恢复后的逻辑文件raw bytes。恢复后的原审阅树约1.14GiB，主要是重复alias/来源JSON，完全不含科学数组。\n\n普通重放预期忠实返回INCOMPLETE（GPU full LIGHT失败以及未在离线包内重检科学数组），同时提供有效OFF比值。进一步运行restored_review/validate_final_delivery_A01.py，预期PASS_EXPECTED_LIMITED_REPLAY。\n\nWindows和真实Linux的实际最终ZIP恢复/重放收据及Git报告tip在ZIP外ZIP_FINAL_RECEIPT.json/GIT_FINAL_DELIVERY_RECEIPT.json，避免自引用。DEFAULT=I0；PRODUCTION_QUALIFIED=false；MANUAL_REVIEW_PENDING。\n'''
    front='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CPU-S4-6A 最终审阅</title><style>body{font:17px/1.7 system-ui;max-width:900px;margin:45px auto;padding:0 24px;color:#172b44}aside{background:#fff3d7;padding:18px}code,pre{background:#edf2f8;padding:8px}li{margin:10px 0}</style><h1>CPU-S4-6A 同工况 CPU/GPU 对比</h1><aside><b>原尺寸：应用7.014倍，完整传播窗口28.815倍，第2–8区间34.976倍。</b><br>总耗时已测得；GPU原尺寸LIGHT干扰失败，细分归因受限，停在人工审阅。</aside><p>本包采用原任务允许的无损元数据去重，保留每份原始逻辑记录。打开主报告即可阅读；完整复算前先恢复原目录。</p><ul><li><a href="payload/MANUAL_REVIEW_REPORT.md">中文主报告</a></li><li><a href="payload/NEXT_CPU_OPTIMIZATION_CONTRACT.md">唯一下一CPU合同</a></li><li><a href="README.md">恢复及重放说明</a></li><li><a href="LOGICAL_FILE_INDEX.json">全部原始逻辑文件与校验映射</a></li></ul><pre>python -s -B restore_review.py --output ./restored_review\npython -s -B ./restored_review/replay_readonly.py --root ./restored_review</pre><p>不含科学数组或环境。DEFAULT=I0 · PRODUCTION_QUALIFIED=false · MANUAL_REVIEW_PENDING。</p></html>'''
    wrappers={'LOGICAL_FILE_INDEX.json':index_bytes,'README.md':readme.encode('utf-8'),
              'REVIEW_INDEX.html':front.encode('utf-8'),'restore_review.py':(c/'restore_compact_review.py').read_bytes()}
    print(json.dumps(dict(stage='LOSSLESS_COMPACT_ZIP_START',logical_files=len(rows),unique_payload_files=len(payload))),flush=True)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_LZMA) as z:
        for name,data in wrappers.items():z.writestr(name,data)
        for name,path in payload.items():z.write(path,name)
    expected={name:pv.raw_sha256(data) for name,data in wrappers.items()}
    expected.update({row['payload_path']:row['raw_sha256'] for row in rows})
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();require(set(names)==set(expected),'COMPACT_MEMBER_SET')
        require(len(names)==len(set(x.casefold() for x in names)),'COMPACT_CASEFOLD_DUPLICATE')
        for name in names:require(pv.raw_sha256(z.read(name))==expected[name],'COMPACT_RAW_HASH_AND_CRC:'+name)
    require(archive.stat().st_size<=20*1024**2,'COMPACT_ZIP_20MIB_CAP')
    receipt=dict(status='PASS_LOSSLESS_COMPACT_ZIP_BYTES',path=str(archive),hash_scope='raw_bytes',
        raw_sha256=pv.raw_sha256_file(archive),size_bytes=archive.stat().st_size,members=len(expected),
        logical_files=len(rows),unique_payload_files=len(payload),
        logical_uncompressed_bytes=sum(row['size_bytes'] for row in rows),
        unique_uncompressed_bytes=sum(path.stat().st_size for path in payload.values()),
        original_full_draft=dict(path=str(original.with_suffix('.zip')),size_bytes=original.with_suffix('.zip').stat().st_size,
            raw_sha256=pv.raw_sha256_file(original.with_suffix('.zip')),status='OVER_20MIB_PRESERVED_NOT_FINAL_DELIVERY'),
        CRC='PASS_ALL_UNIQUE_MEMBERS',all_unique_member_raw_sha256='PASS',scientific_arrays=0,nested_archives=0,
        wall_s=time.perf_counter()-start,conclusion='TOTAL_GAP_MEASURED / DETAIL_ATTRIBUTION_LIMITED',
        DEFAULT='I0',PRODUCTION_QUALIFIED=False,MANUAL_REVIEW_PENDING=True)
    write(c/'ZIP_COMPACT_BUILD_RECEIPT_A01.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
