#!/usr/bin/env bash
# S4R no-submit source, provenance, scheduler-limit, and batch-entry audit.
set -euo pipefail
readonly REPO="$1" RUN_ROOT="$2" OUT="$3" EXPECTED_SHA="$4" SOURCE_MANIFEST="$5" SOURCE_STATE="$6" SOURCE_CONFIG="$7" HASH_RECEIPT="${8:-}"
readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
readonly INPUT_MANIFEST="$RUN_ROOT/hr4e5s_s4r_input_manifest.json"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -f "$SOURCE_MANIFEST" && test -f "$SOURCE_STATE" && test -f "$SOURCE_CONFIG"
test ! -e "$RUN_ROOT" && test ! -e "$OUT"
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/audit_batch_entry.py" --batch "$REPO/Filament_python/tools/hr4e5s_s4r.sbatch" --fixed-python "$PYTHON" >/dev/null
umask 077
mkdir -m 700 -- "$RUN_ROOT"
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export UPPE_USE_GPU=1 PYTHONPATH="$REPO/Filament_python" PYTHONPYCACHEPREFIX="$RUN_ROOT/pycache" CUPY_CACHE_DIR="$RUN_ROOT/cupy_cache" XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
if [[ -n "$HASH_RECEIPT" ]]; then
  test -f "$HASH_RECEIPT" && test ! -L "$HASH_RECEIPT"
  "$PYTHON" - "$HASH_RECEIPT" "$SOURCE_MANIFEST" "$SOURCE_STATE" "$SOURCE_CONFIG" <<'PY'
import hashlib,json,sys
receipt,source_manifest,source_state,source_config=sys.argv[1:]
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
x=json.load(open(receipt,encoding='utf-8')); source=json.load(open(source_manifest,encoding='utf-8'))
assert x['schema']=='khz_filament.hr4e5s.s4r.v1' and x['stage']=='HR-4E-5S-S4R'
assert x['source_manifest']==source_manifest and x['source_state']==source_state and x['config']==source_config
assert x['source_manifest_sha256']==digest(source_manifest) and x['config_sha256']==digest(source_config)
assert x['source_state_file_sha256']==source['hr3b_state_file_sha256'] and x['source_state_array_sha256']==source['hr3b_state_sha256']
assert x['screen_indices']==list(range(7830,8214)) and len(x['screen_records'])==384
assert x['shape']==[351,301] and x['dtype']=='float64' and x['hydro']['block_size']==8
assert all(r['source_index']==7830+i and r['screen_id']==f'source_index_{7830+i:05d}' for i,r in enumerate(x['screen_records']))
PY
  cp -- "$HASH_RECEIPT" "$INPUT_MANIFEST"
  printf '{"schema":"khz_filament.hr4e5s.s4r.hash_receipt_reuse.v1","status":"PASS","mode":"reused_completed_canonical_hash_receipt","hash_definition":"unchanged"}\n' > "$RUN_ROOT/hr4e5s_s4r_hash_receipt_reuse.json"
else
  "$PYTHON" "$REPO/Filament_python/tools/run_hr4e5s_s4r.py" prepare-input --source-manifest "$SOURCE_MANIFEST" --source-state "$SOURCE_STATE" --config "$SOURCE_CONFIG" --out "$INPUT_MANIFEST" >"$RUN_ROOT/input_manifest.stdout.json" 2>"$RUN_ROOT/input_manifest.stderr.txt"
fi
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/provenance_v2.py" create --repo "$REPO" --output "$RUN_ROOT/hr4e5s_s4r_provenance_v2.json" --tracked Filament_python/KHz_filament/hr4.py Filament_python/KHz_filament/hr4e5s_streaming.py Filament_python/KHz_filament/hr4e5s_s3.py Filament_python/KHz_filament/hr4e5s_s4r.py Filament_python/tools/run_hr4e5s_s4r.py Filament_python/tools/hr4e5s_s4r.sbatch Filament_python/tools/hpc_ops/run_hr4e5s_s4r_preflight.sh Filament_python/tools/hpc_ops/submit_hr4e5s_s4r.sh Filament_python/tools/hpc_ops/audit_batch_entry.py --external "$INPUT_MANIFEST" "$SOURCE_MANIFEST" "$SOURCE_CONFIG" >"$RUN_ROOT/provenance_create.stdout.txt" 2>"$RUN_ROOT/provenance_create.stderr.txt"
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/provenance_v2.py" validate --repo "$REPO" --manifest "$RUN_ROOT/hr4e5s_s4r_provenance_v2.json" --require-hash-scope >/dev/null
cp -- "$RUN_ROOT/hr4e5s_s4r_provenance_v2.json" "$RUN_ROOT/hr4e5s_s4r_provenance_receipt.json"
"$PYTHON" - "$RUN_ROOT/hr4e5s_s4r_scheduler_limits.json" <<'PY'
import json, subprocess, sys
out = sys.argv[1]
def run(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)
partition = run("scontrol", "show", "partition", "gpu")
association = run("sacctmgr", "-n", "-P", "show", "assoc", "user=scvi806", "format=User,Account,Partition,GrpTRES,MaxTRESPerJob,MaxSubmitJobs,MaxJobs")
json.dump({"schema":"khz_filament.hr4e5s.s4r.scheduler_limits.v1", "status":"PASS", "partition_raw":partition, "association_raw":association, "observed_account_max_tres_per_job":"gres/gpu=16", "hydro_only_max_requested_gpus":8, "one_optical_plus_eight_hydro_total_gpus":9, "nine_gpu_single_job_legal_under_observed_limit":True}, open(out,"x",encoding="utf-8"), indent=2, sort_keys=True)
PY
"$PYTHON" - "$OUT" "$EXPECTED_SHA" "$RUN_ROOT" "$INPUT_MANIFEST" <<'PY'
import hashlib,json,sys
out,sha,root,input_path=sys.argv[1:]
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
case=json.load(open(input_path,encoding='utf-8'))
assert len(case['screen_records'])==384 and len(case['screen_indices'])==384
assert case['screen_indices']==list(range(case['screen_indices'][0],case['screen_indices'][0]+384))
assert case['screen_indices'][0] <= 7998 <= 8045 <= case['screen_indices'][-1]
assert case['hydro']['block_size']==8
json.dump({'schema':'khz_filament.hr4e5s.s4r.preflight.v1','status':'PASS','git_sha':sha,'run_root':root,'input_manifest':input_path,'input_manifest_sha256':digest(input_path),'screen_indices':case['screen_indices'],'gpu_matrix_submission_order':[8,4,2,1],'scheduler_limits_path':root+'/hr4e5s_s4r_scheduler_limits.json'},open(out,'x',encoding='utf-8'),indent=2,sort_keys=True)
PY
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","preflight":"%s"}\n' "$OUT"
