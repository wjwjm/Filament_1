import datetime,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
repo=Path.cwd();r=repo/'reports/CPU_S4_6A_MATCHED_CPU_GPU_20261028';sys.path.insert(0,str(repo/'Filament_python'))
from tools.hpc_ops import provenance_v2 as pv
head='3aa457bef5866206b1c2742b9eb5b401f855a837';cpu='5c45586d12cb6450f5543b511ae179f87e58bcfa'
if pv._git(repo,'rev-parse','HEAD')!=head:raise RuntimeError('EXECUTION_SOURCE_CHANGED')
if set(pv._git(repo,'diff','--name-only',cpu,head).splitlines())!={'Filament_python/tools/cpu_s4_6a/gpu.py','Filament_python/tools/cpu_s4_6a/test_gpu_manager.py'}:raise RuntimeError('CPU_TESTED_BYTES_CHANGED')
tests=[]
for platform in ('WINDOWS','LINUX'):
 for kind,count in [('RECORDS_R2_FINAL',9),('MICRO_R2',3)]:
  p=r/(platform+'_'+kind+'.xml');root=ET.parse(p).getroot();suites=[root] if root.tag=='testsuite' else list(root)
  if sum(int(v.attrib['tests']) for v in suites)!=count or any(int(v.attrib.get(key,'0')) for v in suites for key in ('failures','errors','skipped')):raise RuntimeError('R2_LOCAL_TEST_FAILURE:'+p.name)
  tests.append(dict(path=p.name,tests=count,failures=0,errors=0,raw_sha256=pv.raw_sha256_file(p),hash_scope='raw_bytes'))
for platform in ('WINDOWS','LINUX'):
 data=(r/(platform+'_R2.log')).read_text(encoding='utf-8-sig')
 if not data.strip().endswith(cpu) or 'backend_probe_passed' not in data or '1 passed' not in data:raise RuntimeError('CPU_SOURCE_PINNED_BACKEND_SANITY')
value=dict(status='PASS_LOCAL_WINDOWS_LINUX_MICRO',execution_commit=head,instrument_revision=2,cpu_qualification_commit=cpu,
 CPU_executed_source_files_identical_after_qualification=True,post_qualification_diff=['GPU-only tool loader','GPU-only negative-dependency test'],
 science_tree=pv._git(repo,'rev-parse','HEAD:Filament_python/KHz_filament'),sealed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tests=tests,real_windows_per_platform=9,
 micro_cases=['Nt32/17/19 two intervals','Nt32/17/19 eight intervals','Nt384/17/19 eight intervals'],
 original_reference_and_OFF_and_LIGHT=True,backend_sanity_compileall=True,GPU_REAL_NOT_RUN=True,DEFAULT='I0',PRODUCTION_QUALIFIED=False,review_gate='MANUAL_REVIEW_PENDING')
with (r/'LOCAL_ACCEPTANCE.json').open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
print(json.dumps(value))
