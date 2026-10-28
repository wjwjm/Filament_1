"""Parent-authorized one CPU small submission after frozen local/login gates."""
import datetime,json,os,pwd,re,subprocess,sys
from pathlib import Path
ROOT=Path('/publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_s46a_20261028_r1');report=ROOT/'report';code=ROOT/'code'
def require(ok,why):
 if not ok:raise RuntimeError(why)
def read(p):return json.loads(p.read_text())
def write(p,value):
 with p.open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
require(pwd.getpwuid(os.geteuid()).pw_name=='t0s000727' and not os.environ.get('SLURM_JOB_ID'),'T0_LOGIN')
sys.path.insert(0,str(code/'Filament_python'));os.environ['PATH']=str(ROOT/'runtime_bin')+os.pathsep+os.environ['PATH']
from tools.hpc_ops import provenance_v2 as pv
from tools.cpu_s4_6a.records import budget_check
plan=read(report/'MEASUREMENT_PLAN.json');pre=read(report/'LOGIN_PREFLIGHT.json');local=read(report/'LOCAL_ACCEPTANCE.json')
require(local['status']=='PASS_LOCAL_WINDOWS_LINUX_MICRO' and local['execution_commit']==plan['execution_commit']==pre['execution_commit'],'LOCAL_LOGIN_SAME_FROZEN_SOURCE')
pv.validate_manifest(code,report/'EXECUTION_SOURCE_MANIFEST.json',require_clean=True,require_hash_scope=True)
require(pv.raw_sha256_file(report/'MEASUREMENT_PLAN.json')==pre['plan_raw_sha256'] and pv.raw_sha256_file(report/'BUDGET_ADMISSION.json')==pre['budget_raw_sha256'],'PRE_DATA_PLAN_BUDGET_UNCHANGED')
admission=read(report/'BUDGET_ADMISSION.json');gate=budget_check(admission['ledger'],'CPU',new_request=admission['request']);require(gate['status']=='PASS_BUDGET','WORST_BUDGET')
v=subprocess.run(['squeue','-h','-u','t0s000727','-o','%i|%j|%T'],capture_output=True,text=True,check=True)
require(not any('s46a' in line.lower() for line in v.stdout.splitlines()),'NO_TASK_ALLOCATION_ALREADY_INFLIGHT')
exports=dict(S46A_SOURCE=str(code),S46A_REPORT=str(report),S46A_SIZE='small',S46A_STAGE_ROOT=str(ROOT),S46A_PLAN_SHA256=pre['plan_raw_sha256'],S46A_BUDGET_SHA256=pre['budget_raw_sha256'],S46A_WALLTIME_S='1200')
argv=['sbatch','--parsable','--partition=v6_384','--account=bscc-t','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=96','--hint=nomultithread','--mem=32G','--time=00:20:00','--job-name=s46a-cpu-small-r2',
 '--output='+str(report/'slurm-%j.stdout.log'),'--error='+str(report/'slurm-%j.stderr.log'),'--export=ALL,'+','.join(k+'='+value for k,value in exports.items()),str(code/'Filament_python/tools/cpu_s4_6a/cpu.sbatch')]
write(ROOT/'SUBMISSION_LOCK_SMALL.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),argv=argv,budget=gate,local_acceptance_raw_sha256=pv.raw_sha256_file(report/'LOCAL_ACCEPTANCE.json')))
p=subprocess.run(argv,cwd=code,capture_output=True,text=True,timeout=60)
receipt=dict(argv=argv,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exports=exports,budget=gate)
if p.returncode==0:
 job=p.stdout.strip().split(';')[0];require(re.fullmatch(r'[0-9]+',job),'PARSABLE_JOBID');receipt['job_id']=job
write(report/'SUBMISSION_RECEIPT.json',receipt)
require(p.returncode==0,'SBATCH_FAILED:'+p.stderr);print(json.dumps(receipt))
