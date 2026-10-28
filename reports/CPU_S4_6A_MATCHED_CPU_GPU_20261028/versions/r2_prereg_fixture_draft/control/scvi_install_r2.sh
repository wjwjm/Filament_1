#!/usr/bin/env bash
set -euo pipefail
stage=identity
trap 'printf "{\"schema\":\"filament.cpu_s46a.install.v1\",\"ok\":false,\"state\":\"failed\",\"stage\":\"%s\"}\n" "$stage"' ERR
test "$(id -un)" = scvi806
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 UPPE_USE_GPU=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
root=/data/run01/scvi806/user_Wangjimin/projects/filament_cpu_s46a_20261028_r1
stage=verified_new_source_store
/data/home/scvi806/.conda/envs/Filament_python/bin/python -s -B - "$1" <<'PY'
import json,subprocess,sys,tarfile
from pathlib import Path
root=Path('/data/run01/scvi806/user_Wangjimin/projects/filament_cpu_s46a_20261028_r1')
base=root.parent/'filament_cpu_s46a_20261028_a01/code'
def require(ok,why):
 if not ok:raise RuntimeError(why)
def call(argv,cwd=None):
 p=subprocess.run(argv,cwd=cwd,capture_output=True,text=True,timeout=90)
 require(p.returncode==0,'LOCAL_SOURCE_COMMAND:'+str(argv));return p.stdout.strip()
require(not root.exists() and root.parent.resolve()==root.parent,'NEW_CONFINED_ROOT')
require(call(['git','rev-parse','HEAD'],base)=='e134087bd07cc8748e5b6e000fe5592379e35e2c' and call(['git','status','--porcelain','--untracked-files=all'],base)=='','CLEAN_GPU_OBJECT_BASE')
require(call(['git','cat-file','-t','d4bd96ca21c5815b57f1a3c2ee8f8cec54dfb220'],base)=='commit','ACCEPTED_GPU_OBJECT')
archive=Path(str(root)+'.tar.gz')
require(call(['sha256sum',str(archive)]).split()[0]==sys.argv[1],'RAW_ARCHIVE_BOOTSTRAP_IDENTITY')
root.mkdir()
with tarfile.open(archive) as t:
 for m in t.getmembers():require(not m.name.startswith('/') and (root/m.name).resolve().is_relative_to(root) and not m.issym() and not m.islnk(),'CONFINED_REGULAR_TAR')
 t.extractall(root,filter='data')
call(['git','-c','gc.auto=0','clone','--shared','--no-checkout',str(base),str(root/'code')])
call(['git','remote','set-url','origin','https://github.com/wjwjm/Filament_1.git'],root/'code')
PY
stage=proxy_first_or_verified_bundle
source "$(dirname "$0")/hpc_proxy_env.sh"
transport=VERIFIED_BUNDLE_FALLBACK
if hpc_proxy_load /data/run01/scvi806/user_Wangjimin/.secrets/github_proxy.env && hpc_proxy_git_ls_remote https://github.com/wjwjm/Filament_1.git refs/heads/codex/cpu-gpu-optical-gap-20261028 3aa457bef5866206b1c2742b9eb5b401f855a837 30; then
  if timeout 120 git -c gc.auto=0 -C "$root/code" fetch --no-tags origin refs/heads/codex/cpu-gpu-optical-gap-20261028:refs/heads/codex/cpu-gpu-optical-gap-20261028 >/dev/null 2>&1; then
    transport=PROXY_GITHUB_VERIFIED_REF
  fi
fi
if test "$transport" = VERIFIED_BUNDLE_FALLBACK; then
  git -C "$root/code" bundle verify "$root/SOURCE.bundle" >/dev/null 2>&1
  git -c gc.auto=0 -C "$root/code" fetch "$root/SOURCE.bundle" refs/heads/codex/cpu-gpu-optical-gap-20261028:refs/heads/codex/cpu-gpu-optical-gap-20261028 >/dev/null 2>&1
fi
git -c gc.auto=0 -C "$root/code" checkout codex/cpu-gpu-optical-gap-20261028 >/dev/null 2>&1
export S46A_SOURCE_TRANSPORT="$transport"
stage=source_batch_and_environment_preflight
/data/home/scvi806/.conda/envs/Filament_python/bin/python -s -B - <<'PY'
import importlib,json,os,subprocess,sys,traceback
from pathlib import Path
root=Path('/data/run01/scvi806/user_Wangjimin/projects/filament_cpu_s46a_20261028_r1');code=root/'code';report=root/'report'
def require(ok,why):
 if not ok:raise RuntimeError(why)
def write(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
try:
 sys.path.insert(0,str(code/'Filament_python'))
 from tools.hpc_ops import provenance_v2 as pv
 from tools.cpu_s4_6a.records import budget_check,validate_source_records
 plan=json.loads((report/'MEASUREMENT_PLAN.json').read_text())
 source=pv.validate_manifest(code,report/'EXECUTION_SOURCE_MANIFEST.json',require_clean=True,require_hash_scope=True)
 require(source['head']==plan['execution_commit']=='3aa457bef5866206b1c2742b9eb5b401f855a837','EXPECTED_SOURCE')
 validate_source_records(root,json.loads((root/'TRANSFER_MANIFEST.json').read_text())['records'])
 admission=json.loads((report/'BUDGET_ADMISSION.json').read_text());require(budget_check(admission['ledger'],'GPU',new_request=admission['request'])['status']=='PASS_BUDGET','GPU_BUDGET')
 from tools.hpc_ops.audit_batch_entry import audit_batch_entry
 audit,exit_code=audit_batch_entry(code/'Filament_python/tools/cpu_s4_6a/gpu.sbatch',plan['python'],'/bin/bash');require(exit_code==0 and audit['status']=='passed','BATCH_AUDIT')
 write(report/'BATCH_ENTRY_AUDIT.json',audit)
 os.environ.update(S43G2_SOURCE_ROOT=str(code),S43F_SOURCE_ROOT=str(code),S43C_SOURCE_ROOT=str(code),CPU_BOUNDARY_HISTORICAL_RECEIPT=str(report/'external_data/G2_BOUNDARY_RECEIPT.json'))
 versions={name:importlib.import_module(name).__version__ for name in ('numpy','scipy','cupy')}
 import tools.cpu_s4_6a.gpu
 from tools.cpu_s4_6a.gpu import _preload_host_services
 _preload_host_services()
 import cupyx.scipy.signal,cupyx.scipy.fft
 flags=['--partition=gpu','--nodes=1','--ntasks=1','--gres=gpu:1','--cpus-per-task=8','--hint=nomultithread','--mem=120G','--time=01:30:00','--exclude=g0603',str(code/'Filament_python/tools/cpu_s4_6a/gpu.sbatch')]
 v=subprocess.run(['sbatch','--test-only',*flags],cwd=code,capture_output=True,text=True,timeout=60);require(v.returncode==0,'SLURM_TEST_ONLY:'+v.stderr)
 receipt=dict(status='PASS_S46A_GPU_LOGIN_PREFLIGHT',execution_commit=plan['execution_commit'],source_records=len(source['records']),
  plan_raw_sha256=pv.raw_sha256_file(report/'MEASUREMENT_PLAN.json'),budget_raw_sha256=pv.raw_sha256_file(report/'BUDGET_ADMISSION.json'),
  source_transport=os.environ['S46A_SOURCE_TRANSPORT'],versions=versions,python=sys.version,actual_interpreter=sys.executable,
  batch_entry_audit=audit,test_only=dict(exit_code=v.returncode,stdout=v.stdout,stderr=v.stderr),new_jobids=0,GPU_REAL_NOT_RUN=True)
 write(report/'LOGIN_PREFLIGHT.json',receipt)
 print(json.dumps(dict(schema='filament.cpu_s46a.install.v1',ok=True,state='completed',**{k:receipt[k] for k in ('status','execution_commit','source_transport','versions','new_jobids')})))
except BaseException as error:
 write(root/'INSTALL_FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),new_jobids=0))
 raise
PY
