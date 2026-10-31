"""Authorized S46C Job1, single exclusive lock and budget reservation."""
import datetime,json,os,pwd,re,subprocess,sys
from pathlib import Path
root=Path('/publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_s46c_20261031_a02');code=root/'code';report=root/'report'
def require(ok,why):
 if not ok:raise RuntimeError(why)
def write(path,value):
 with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
require(pwd.getpwuid(os.geteuid()).pw_name=='t0s000727' and not os.environ.get('SLURM_JOB_ID'),'T0_LOGIN')
sys.path.insert(0,str(code/'Filament_python'));os.environ['PATH']=str(root/'runtime_bin')+os.pathsep+os.environ['PATH']
from tools.hpc_ops import provenance_v2 as pv
plan=json.loads((report/'PLAN.json').read_text());pre=json.loads((report/'LOGIN_PREFLIGHT.json').read_text())
local=json.loads((report/'LOCAL_ACCEPTANCE.json').read_text())
require(pre['status']=='PASS' and local['status']=='PASS_LOCAL_ENGINEERING' and local['execution_commit']==pre['execution_commit'],'LOCAL_LOGIN_SOURCE')
require(pv.raw_sha256_file(report/'PLAN.json')==pre['plan_raw_sha256'],'PLAN_UNCHANGED')
pv.validate_manifest(code,report/'EXECUTION_SOURCE_MANIFEST.json',require_clean=True,require_hash_scope=True)
require(not (root/'SUBMISSION_LOCK.json').exists(),'NO_DUPLICATE_SUBMISSION')
queue=subprocess.run(['squeue','-h','-u','t0s000727','-o','%i|%j|%T'],capture_output=True,text=True,check=True)
require(not any('s46c' in line.lower() for line in queue.stdout.splitlines()),'NO_TASK_JOB_INFLIGHT')
prior=subprocess.run(['sacct','-j','8858616','-n','--parsable2','--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS'],capture_output=True,text=True,check=True)
main=[x.split('|') for x in prior.stdout.splitlines() if x.split('|')[0]=='8858616']
require(len(main)==1 and main[0][1]=='FAILED' and int(main[0][4])==96,'FIRST_JOB_TERMINAL')
used=int(main[0][3])*96/3600
require(plan['resources']['cpu_jobs_max']==2 and used+96*1800/3600<=96,'TWO_JOB_WORST_BUDGET')
exports=dict(S46C_SOURCE=str(code),S46C_REPORT=str(report),S46C_STAGE_ROOT=str(root),S46C_PLAN_SHA256=pre['plan_raw_sha256'])
argv=['sbatch','--parsable','--partition=v6_384','--account=bscc-t','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=96','--hint=nomultithread','--mem=64G','--time=00:30:00','--job-name=s46c-array-a2',
 '--output='+str(report/'slurm-%j.stdout.log'),'--error='+str(report/'slurm-%j.stderr.log'),'--export=ALL,'+','.join(k+'='+v for k,v in exports.items()),str(code/'Filament_python/tools/cpu_s4_6c_array_coverage/cpu.sbatch')]
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
write(root/'SUBMISSION_LOCK.json',dict(created_utc=stamp,argv=argv,reserved_core_hours=48,already_used_core_hours=used,job_limit=2))
p=subprocess.run(argv,cwd=code,capture_output=True,text=True,timeout=60)
receipt=dict(argv=argv,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,submitted_utc=stamp,core_hours_reserved=48,already_used_core_hours=used)
if p.returncode==0:
 job=p.stdout.strip().split(';')[0];require(re.fullmatch('[0-9]+',job),'PARSABLE_JOBID');receipt['job_id']=job
write(report/'SUBMISSION_RECEIPT.json',receipt);print(json.dumps(receipt))
require(p.returncode==0,'SBATCH_FAILED_NO_AUTOMATIC_IDENTICAL_RETRY')
