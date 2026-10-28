"""Install one verified incremental source, no runs/locks/sbatch side effects."""
import json,os,pwd,subprocess,sys,tarfile,traceback
from pathlib import Path
ROOT=Path('/publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_s46a_20261028_r1')
BASE=ROOT.parent/'filament_cpu_s46a_20261028_a01/code'
HEAD='3aa457bef5866206b1c2742b9eb5b401f855a837';BASESHA='e134087bd07cc8748e5b6e000fe5592379e35e2c'
BRANCH='codex/cpu-gpu-optical-gap-20261028';COMMANDS=[]
def require(ok,why):
 if not ok:raise RuntimeError(why)
def write(p,d):
 with p.open('x',encoding='utf8') as f:json.dump(d,f,indent=2,allow_nan=False)
def call(argv,cwd=None,timeout=120):
 env=os.environ.copy();env.pop('GIT_DIR',None);env.pop('GIT_WORK_TREE',None)
 v=subprocess.run(argv,cwd=cwd,env=env,capture_output=True,text=True,timeout=timeout)
 row=dict(argv=argv,cwd=str(cwd) if cwd else None,exit_code=v.returncode,stdout=v.stdout,stderr=v.stderr);COMMANDS.append(row)
 require(v.returncode==0,'COMMAND_FAILED:'+str(row));return v.stdout.strip()
def main():
 require(pwd.getpwuid(os.geteuid()).pw_name=='t0s000727' and not os.environ.get('SLURM_JOB_ID'),'T0_LOGIN_IDENTITY')
 require(not ROOT.exists() and ROOT.parent.resolve()==ROOT.parent,'NEW_CONFINED_STAGE_ROOT')
 require(call(['/usr/bin/git','rev-parse','HEAD'],BASE)==BASESHA and call(['/usr/bin/git','status','--porcelain','--untracked-files=all'],BASE)=='','CLEAN_ACCEPTED_BASE')
 sys.path.insert(0,str(BASE/'Filament_python'))
 from tools.hpc_ops import provenance_v2 as pv
 archive=Path(str(ROOT)+'.tar.gz')
 require(pv.raw_sha256_file(archive)==sys.argv[1],'ORIGINAL_ARCHIVE_RAW_SHA256')
 ROOT.mkdir()
 with tarfile.open(archive,'r:gz') as t:
  for member in t.getmembers():
   path=ROOT/member.name
   require(not member.name.startswith('/') and path.resolve().is_relative_to(ROOT) and not member.issym() and not member.islnk(),'CONFINED_REGULAR_ARCHIVE')
  t.extractall(ROOT,filter='data')
 plan=json.loads((ROOT/'report/MEASUREMENT_PLAN.json').read_text())
 for row in json.loads((ROOT/'TRANSFER_MANIFEST.json').read_text())['records']:
  p=ROOT/row['path'];require(p.resolve().is_relative_to(ROOT) and p.stat().st_size==row['size_bytes'] and pv.raw_sha256_file(p)==row['raw_sha256'],'TRANSFER_RAW:'+row['path'])
 code=ROOT/'code'
 call(['/usr/bin/git','-c','gc.auto=0','clone','--shared','--no-checkout',str(BASE),str(code)])
 call(['/usr/bin/git','-c','gc.auto=0','remote','set-url','origin','https://github.com/wjwjm/Filament_1.git'],code)
 call(['/usr/bin/git','bundle','verify',str(ROOT/'SOURCE.bundle')],code)
 call(['/usr/bin/git','-c','gc.auto=0','fetch',str(ROOT/'SOURCE.bundle'),'refs/heads/'+BRANCH+':refs/heads/'+BRANCH],code)
 call(['/usr/bin/git','-c','gc.auto=0','checkout',BRANCH],code)
 require(call(['/usr/bin/git','rev-parse','HEAD'],code)==HEAD and call(['/usr/bin/git','status','--porcelain','--untracked-files=all'],code)=='','FRESH_CLEAN_EXPECTED_SOURCE')
 sys.path[:]=[str(code/'Filament_python')]+[p for p in sys.path if p!=str(BASE/'Filament_python')]
 for name in list(sys.modules):
  if name=='tools' or name.startswith('tools.'):del sys.modules[name]
 from tools.cpu_s4_checkpoint_support.git_readonly_compat import launcher_source,verify_runtime_query_tool
 runtime=ROOT/'runtime_bin';runtime.mkdir();launcher=runtime/'git'
 launcher.write_text(launcher_source(plan['python'],code/'Filament_python/tools/cpu_s4_checkpoint_support/git_readonly_compat.py','/usr/bin/git'),encoding='utf8');launcher.chmod(0o700)
 os.environ['PATH']=str(runtime)+os.pathsep+os.environ['PATH']
 from tools.hpc_ops import provenance_v2 as current
 identity=verify_runtime_query_tool(plan)
 from tools.hpc_ops.audit_batch_entry import audit_batch_entry
 audit,exit_code=audit_batch_entry(code/'Filament_python/tools/cpu_s4_6a/cpu.sbatch',plan['python'],'/bin/bash')
 require(exit_code==0 and audit['status']=='passed','BATCH_ENTRY_AUDIT')
 source=current.validate_manifest(code,ROOT/'report/EXECUTION_SOURCE_MANIFEST.json',require_clean=True,require_hash_scope=True)
 require(source['head']==HEAD and current._git(code,'rev-parse','HEAD:Filament_python/KHz_filament')==plan['science_tree'],'STRICT_SOURCE_SCIENCE')
 from tools.cpu_s4_6a.records import budget_check,validate_source_records
 transfer=json.loads((ROOT/'report/EXTERNAL_INPUT_RECORDS.json').read_text());validate_source_records(ROOT/'report',transfer['records'])
 admission=json.loads((ROOT/'report/BUDGET_ADMISSION.json').read_text())
 require(budget_check(admission['ledger'],'CPU',new_request=admission['request'])['status']=='PASS_BUDGET','BUDGET_BEFORE_ANY_RUN_LOCK_SUBMISSION')
 for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[n]='1'
 os.environ.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',UPPE_USE_GPU='0',NUMBA_NUM_THREADS='48',NUMBA_CPU_NAME='generic',NUMBA_THREADING_LAYER='workqueue',
   S43G2_SOURCE_ROOT=str(code),S43F_SOURCE_ROOT=str(code),S43C_SOURCE_ROOT=str(code),CPU_S4_DATA_ROOT=str(ROOT/'report/external_data/input_bundle'),CPU_BOUNDARY_HISTORICAL_RECEIPT=str(ROOT/'report/external_data/G2_BOUNDARY_RECEIPT.json'))
 import importlib
 versions={name:importlib.import_module(name).__version__ for name in ('numpy','scipy','numba','llvmlite','psutil','threadpoolctl')}
 require(sys.version_info[:3]==(3,12,8) and versions==dict(numpy='2.2.5',scipy='1.15.2',numba='0.61.2',llvmlite='0.44.0',psutil='7.0.0',threadpoolctl='3.6.0'),'ACTUAL_BATCH_ENV_IMPORTS')
 from tools.cpu_s4_5b_steady_cost import candidate
 pair=candidate.make_recorder(False);built=candidate.build_role('PURE_OFF_SMALLGRID',capture_root=ROOT/'report/FACTORY_CHECK_UNUSED',profile=False,recorder_pair=pair)
 binding=candidate.verify_bindings(built[1]);require(binding is not None,'ACTUAL_LINUX_FACTORY')
 write(ROOT/'report/BATCH_ENTRY_AUDIT.json',audit)
 flags=['--partition=v6_384','--account=bscc-t','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=96','--hint=nomultithread','--mem=32G','--time=00:20:00',str(code/'Filament_python/tools/cpu_s4_6a/cpu.sbatch')]
 call(['sbatch','--test-only',*flags],code,60)
 receipt=dict(status='PASS_S46A_LOGIN_PREFLIGHT',execution_commit=HEAD,baseline_commit=BASESHA,archive_raw_sha256=current.raw_sha256_file(archive),source_records=len(source['records']),
   plan_raw_sha256=current.raw_sha256_file(ROOT/'report/MEASUREMENT_PLAN.json'),budget_raw_sha256=current.raw_sha256_file(ROOT/'report/BUDGET_ADMISSION.json'),
   git_identity=identity,versions=versions,factory_binding=binding,commands=COMMANDS,new_jobids=0,propagation='NOT_RUN')
 write(ROOT/'report/LOGIN_PREFLIGHT.json',receipt);print(json.dumps({key:receipt[key] for key in ('status','execution_commit','plan_raw_sha256','budget_raw_sha256','new_jobids')}))
try:main()
except BaseException as error:
 if ROOT.is_dir() and not (ROOT/'INSTALL_FAILURE.json').exists():write(ROOT/'INSTALL_FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),commands=COMMANDS,new_jobids=0))
 raise
