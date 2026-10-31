"""Single S46C clean checkout; audit before run root, no submission."""
import importlib,json,os,pwd,subprocess,sys,tarfile,tempfile
from pathlib import Path
ROOT=Path('/publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_s46c_20261031_a02')
BASE=ROOT.parent/'filament_cpu_s46c_20261031_a01/code'
BASESHA='7c0ac61fa6de7892a2983e78835f1a3f326a28a1'
BRANCH='codex/cpu-real-array-coverage-20261031'
commands=[]
def require(ok,why):
 if not ok:raise RuntimeError(why)
def call(argv,cwd=None):
 p=subprocess.run(argv,cwd=cwd,capture_output=True,text=True,timeout=120)
 commands.append(dict(argv=argv,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr))
 require(p.returncode==0,'COMMAND_FAILED:'+str(commands[-1]));return p.stdout.strip()
def write(path,value):
 with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
require(pwd.getpwuid(os.geteuid()).pw_name=='t0s000727' and not os.environ.get('SLURM_JOB_ID'),'T0_LOGIN')
code=ROOT/'code';report=ROOT/'report'
require(ROOT.resolve()==ROOT and not (ROOT/'SUBMISSION_LOCK.json').exists() and not (report/'SUBMISSION_RECEIPT.json').exists(),'PRE_SUBMISSION_ONLY')
require({p.name for p in code.iterdir()}=={'.git'},'EMPTY_NO_CHECKOUT_SOURCE_ONLY')
require(call(['/usr/bin/git','rev-parse','HEAD'],code)==BASESHA,'EXACT_DETACH_BASE')
sys.path.insert(0,str(BASE/'Filament_python'))
from tools.hpc_ops import provenance_v2 as pv
from tools.hpc_ops.audit_batch_entry import audit_batch_entry
archive=Path(str(ROOT)+'.tar.gz')
require(pv.raw_sha256_file(archive)==sys.argv[1],'ORIGINAL_TRANSFER_ARCHIVE')
for row in json.loads((ROOT/'TRANSFER_MANIFEST.json').read_text())['records']:
 require(pv.raw_sha256_file(ROOT/row['path'])==row['raw_sha256'],'ORIGINAL_TRANSFER_RECORD')
execution=json.loads((report/'EXECUTION.json').read_text());plan=json.loads((report/'PLAN.json').read_text())
audit,ec=audit_batch_entry(ROOT/'cpu.sbatch',plan['python'],'/bin/bash')
require(ec==0 and audit['status']=='passed','BATCH_AUDIT')
write(report/'INSTALL_DETACH_RECOVERY.json',dict(reason='fetch refused current branch in empty no-checkout clone',empty_checkout_verified=True,baseline=BASESHA,original_archive_sha256=sys.argv[1],new_jobs=0))
call(['/usr/bin/git','checkout','--detach',BASESHA],code)
call(['/usr/bin/git','bundle','verify',str(ROOT/'SOURCE.bundle')],code)
call(['/usr/bin/git','-c','gc.auto=0','fetch',str(ROOT/'SOURCE.bundle'),'refs/heads/'+BRANCH+':refs/heads/'+BRANCH],code)
call(['/usr/bin/git','checkout',BRANCH],code)
require(call(['/usr/bin/git','rev-parse','HEAD'],code)==execution['execution_commit'] and not call(['/usr/bin/git','status','--porcelain','--untracked-files=all'],code),'CLEAN_EXPECTED_REPAIR_SOURCE')
sys.path[0]=str(code/'Filament_python')
for name in list(sys.modules):
 if name=='tools' or name.startswith('tools.'):del sys.modules[name]
from tools.hpc_ops import provenance_v2 as current
from tools.cpu_s4_checkpoint_support.git_readonly_compat import launcher_source,verify_runtime_query_tool
runtime=ROOT/'runtime_bin';runtime.mkdir();launcher=runtime/'git'
launcher.write_text(launcher_source(plan['python'],code/'Filament_python/tools/cpu_s4_checkpoint_support/git_readonly_compat.py','/usr/bin/git'),encoding='utf8');launcher.chmod(0o700)
os.environ['PATH']=str(runtime)+os.pathsep+os.environ['PATH']
identity=verify_runtime_query_tool(dict(system_git='/usr/bin/git',query_git_launcher=str(launcher),python=plan['python'],source_root=str(code),execution_commit=execution['execution_commit']))
manifest=current.validate_manifest(code,report/'EXECUTION_SOURCE_MANIFEST.json',require_clean=True,require_hash_scope=True)
require(current._git(code,'rev-parse','HEAD:Filament_python/KHz_filament')==plan['science_subtree'],'SCIENCE_SUBTREE')
require(current.raw_sha256_file(code/'Filament_python/tools/cpu_s4_6c_array_coverage/cpu.sbatch')==current.raw_sha256_file(ROOT/'cpu.sbatch'),'AUDITED_BATCH_EQUALS_EXECUTION')
for name,digest in plan['stage_metadata_sha256'].items():require(current.raw_sha256_file(Path(plan['stage_root'])/name)==digest,'STAGE_METADATA:'+name)
require(current.raw_sha256_file(Path(plan['canonical'])/'INPUT_MANIFEST.json')==plan['canonical_manifest_sha256'],'ORIGINAL_INPUT_MANIFEST')
os.environ.update(plan['thread_environment']);os.environ.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',UPPE_USE_GPU='0',S43G2_SOURCE_ROOT=str(code),S43F_SOURCE_ROOT=str(code),S43C_SOURCE_ROOT=str(code))
versions={n:importlib.import_module(n).__version__ for n in plan['versions']}
require(versions==plan['versions'],'ACTUAL_LIBRARY_VERSIONS')
from tools.cpu_s4_6c_array_coverage.runner import fresh_factory,stage_kwargs
from KHz_filament.confio import load_all
from types import SimpleNamespace
import numpy as np
loaded=load_all(str(Path(plan['canonical'])/'config.json'))
kw=stage_kwargs(dict(raman=loaded[-1],beam=loaded[1],axes=SimpleNamespace(Omega=np.empty(384),dt=loaded[0].Twin/loaded[0].Nt)))
require(kw['method']=='iir' and kw['chunk_pixels']==65536 and kw['iir_sampling']=='exact_piecewise_linear','ACTUAL_LOADED_CONFIG_ENTRY')
built,state,fft=fresh_factory(report/'UNUSED_LOGIN_CAPTURE')
require(state.stats['reference_fallback_calls']==0,'ACTUAL_FACTORY_IMPORT_ENTRY')
flags=['--partition=v6_384','--account=bscc-t','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=96','--hint=nomultithread','--mem=64G','--time=00:30:00']
call(['sbatch','--test-only',*flags,str(code/'Filament_python/tools/cpu_s4_6c_array_coverage/cpu.sbatch')],code)
write(report/'LOGIN_PREFLIGHT.json',dict(status='PASS',execution_commit=execution['execution_commit'],base_commit=BASESHA,
 archive_raw_sha256=current.raw_sha256_file(archive),bundle_raw_sha256=current.raw_sha256_file(ROOT/'SOURCE.bundle'),
 plan_raw_sha256=current.raw_sha256_file(report/'PLAN.json'),source_records=len(manifest['records']),audit=audit,git_identity=identity,
 versions=versions,factory_binding=built[3],commands=commands,new_jobs=0,science_payload_opened=False))
print(json.dumps(dict(status='PASS_LOGIN_PREFLIGHT',execution_commit=execution['execution_commit'],new_jobs=0)))
