"""Validate the extracted ZIP with a fresh read-only standard-library replay."""
import datetime,json,os,platform,subprocess,sys,tempfile,time,zipfile
from pathlib import Path
repo=Path(__file__).resolve().parents[3];r=repo/'reports/CPU_S4_6A_MATCHED_CPU_GPU_20261028';sys.path.insert(0,str(repo/'Filament_python'))
from tools.hpc_ops import provenance_v2 as pv
archive=r/'CPU_S4_6A_REVIEW_20261029_001.zip';receipt=json.loads((r/'ZIP_RECEIPT.json').read_text())
def require(ok,why):
 if not ok:raise RuntimeError(why)
require(pv.raw_sha256_file(archive)==receipt['raw_sha256'],'ZIP_RAW_IDENTITY')
if os.name=='nt':
 parent=r/'zip_validation';parent.mkdir(exist_ok=True);target=Path(tempfile.mkdtemp(prefix='windows_',dir=parent))
else:target=Path(tempfile.mkdtemp(prefix='cpu_s46a_review_'))
with zipfile.ZipFile(archive) as z:
 require(z.testzip() is None,'ZIP_CRC')
 for member in z.infolist():
  p=target/member.filename
  require(not member.filename.startswith('/') and p.resolve().is_relative_to(target.resolve()) and '..' not in Path(member.filename).parts and (member.external_attr>>16)&0o170000!=0o120000,'SAFE_QUALIFIED_EXTRACTION')
 z.extractall(target)
def inventory():
 return {p.relative_to(target).as_posix():pv.raw_sha256_file(p) for p in target.rglob('*') if p.is_file()}
before=inventory();started=time.perf_counter()
argv=[sys.executable,'-s','-B',str(target/'replay_readonly.py'),'--root',str(target)]
v=subprocess.run(argv,capture_output=True,text=True,encoding='utf8',timeout=120)
require(v.returncode==0,'REPLAY_PROCESS:'+v.stderr[-500:]);result=json.loads(v.stdout)
require(result['status']=='INCOMPLETE','EXPECTED_INCOMPLETE_NOT_FAKE_PERFORMANCE_PASS')
require(result['executed_source']['status']=='SOURCE_IDENTITIES_REPLAYED' and len(result['executed_source']['records'])==244,'OFFLINE_SOURCE_BLOBS_AND_CANONICAL_LF')
require(result['resources']['status']=='REPLAYED','RAW_RESOURCE_REPLAY')
require(abs(result['resources']['backends']['CPU']['actual_core_hours']-13.173333333333332)<1e-10,'RAW_PRIMARY_ACCOUNTING_TOTAL')
require(result['resources']['backends']['GPU']['actual_gpu_hours']==0,'ZERO_GPU_ALLOCATION')
require(result['manifests']['status']=='MANIFEST_REPLAYED','DELIVERY_RAW_BYTES')
require(result['historical']['status']=='REPLAYED' and result['historical']['post_count']==48,'HISTORICAL_ORIGINAL_ROWS')
require(not result['cpu_gpu_ratios']['ratios'],'NO_MATCHED_RATIO')
require(all(item.startswith(('CLOCK:','CASE:','RATIO:')) for item in result['issues']),'UNEXPECTED_REPLAY_ISSUES:'+str(result['issues']))
selftest=subprocess.run([*argv,'--self-test'],capture_output=True,text=True,encoding='utf8',timeout=30)
require(selftest.returncode==0 and json.loads(selftest.stdout)['status']=='SELF_TEST_PASS','SELF_TEST')
after=inventory();require(before==after,'READONLY_REPLAY_CHANGED_EXTRACTED_FILES')
result['_validation']=dict(status='PASS_REPLAY_REPRODUCES_INCOMPLETE',platform=platform.platform(),python=sys.version,executable=sys.executable,argv=argv,
 zip_raw_sha256=receipt['raw_sha256'],extracted_files=len(before),extracted_root=str(target),replay_and_selftest_s=time.perf_counter()-started,
 read_only_tree_before_after_equal=True,scientific_arrays_rechecked=False,expected_missing='No matrices/3 OFF samples/GPU; original OFF PROPAGATION missing',
 validation_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
print(json.dumps(result,ensure_ascii=False,sort_keys=True,allow_nan=False))
