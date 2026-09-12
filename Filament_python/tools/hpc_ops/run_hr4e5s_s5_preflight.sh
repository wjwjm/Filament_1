#!/usr/bin/env bash
# Create a fresh S5-only evidence root only after strict source and batch audit.
set -euo pipefail
readonly REPO="$1" RUN_ROOT="$2" OUT="$3" EXPECTED_SHA="$4" SOURCE_MANIFEST="$5" SOURCE_STATE="$6" SOURCE_CONFIG="$7"
readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
readonly LUT_WORKSPACE="$RUN_ROOT/lut_workspace"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -f "$SOURCE_MANIFEST" && test -f "$SOURCE_STATE" && test -f "$SOURCE_CONFIG"
test ! -e "$RUN_ROOT" && test ! -e "$OUT"
umask 077
mkdir -m 700 -- "$RUN_ROOT"
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export UPPE_USE_GPU=1 PYTHONPATH="$REPO/Filament_python" PYTHONPYCACHEPREFIX="$RUN_ROOT/pycache" CUPY_CACHE_DIR="$RUN_ROOT/cupy_cache" XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/provenance_v2.py" create --repo "$REPO" --output "$RUN_ROOT/hr4e5s_s5_provenance_v2.json" --tracked Filament_python/KHz_filament/hr4e5s_streaming.py Filament_python/KHz_filament/hr4e5s_s3.py Filament_python/KHz_filament/hr4e5s_s5.py Filament_python/KHz_filament/hr4.py Filament_python/KHz_filament/hr4c_state.py Filament_python/KHz_filament/propagate.py Filament_python/KHz_filament/thermalization.py Filament_python/KHz_filament/slow_state.py Filament_python/tools/run_hr4e5s_s3.py Filament_python/tools/run_hr4e5s_s5.py Filament_python/tools/hr4e5s_s5.sbatch Filament_python/tools/hpc_ops/audit_hr4e5s_s3_lut_workspace.py Filament_python/tools/hpc_ops/run_hr4e5s_s5_preflight.sh Filament_python/tools/hpc_ops/submit_hr4e5s_s5.sh --external "$SOURCE_MANIFEST" "$SOURCE_STATE" "$SOURCE_CONFIG" >/dev/null
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/provenance_v2.py" validate --repo "$REPO" --manifest "$RUN_ROOT/hr4e5s_s5_provenance_v2.json" --require-hash-scope >/dev/null
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/audit_batch_entry.py" --batch "$REPO/Filament_python/tools/hr4e5s_s5.sbatch" --fixed-python "$PYTHON" >"$RUN_ROOT/hr4e5s_s5_batch_audit.json"
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/audit_hr4e5s_s3_lut_workspace.py" --config "$SOURCE_CONFIG" --workspace "$LUT_WORKSPACE" --out "$RUN_ROOT/hr4e5s_s5_lut_workspace.json" >/dev/null
"$PYTHON" "$REPO/Filament_python/tools/run_hr4e5s_s3.py" prepare --source-manifest "$SOURCE_MANIFEST" --source-state "$SOURCE_STATE" --config "$SOURCE_CONFIG" --out "$RUN_ROOT/hr4e5s_s5_input_manifest.json" >/dev/null
"$PYTHON" - "$OUT" "$RUN_ROOT" "$EXPECTED_SHA" "$SOURCE_MANIFEST" "$SOURCE_STATE" "$SOURCE_CONFIG" "$RUN_ROOT/hr4e5s_s5_input_manifest.json" "$RUN_ROOT/hr4e5s_s5_lut_workspace.json" <<'PY'
import hashlib,json,sys
out,root,sha,manifest,state,config,input_path,lut_path=sys.argv[1:]
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
source=json.load(open(manifest,encoding='utf-8')); case=json.load(open(input_path,encoding='utf-8')); lut=json.load(open(lut_path,encoding='utf-8'))
assert case['source_state_file_sha256']==source['hr3b_state_file_sha256'] and case['source_state_array_sha256']==source['hr3b_state_sha256']
assert case['config_sha256']==source['config_sha256'] and case['screen_indices']==list(range(7998,8046)) and len(case['screen_records'])==48
assert case['hydro']['block_size']==8 and case['hydro']['dt_hydro']==1e-6 and lut['status']=='PASS'
json.dump({'schema':'khz_filament.hr4e5s.s5.preflight.v1','status':'PASS','git_sha':sha,'run_root':root,'source_manifest':manifest,'source_manifest_sha256':digest(manifest),'source_state':state,'source_state_file_sha256':source['hr3b_state_file_sha256'],'source_state_array_sha256':source['hr3b_state_sha256'],'source_config':config,'source_config_sha256':source['config_sha256'],'input_manifest':'hr4e5s_s5_input_manifest.json','input_manifest_sha256':digest(input_path),'lut_workspace':'lut_workspace','lut_workspace_sha256':digest(lut_path),'screen_indices':case['screen_indices'],'resources':{'optical_gpus':1,'hydro_gpus':1},'fault_injection_default':'DISABLED'},open(out,'w',encoding='utf-8'),indent=2,sort_keys=True)
PY
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","preflight":"%s"}\n' "$OUT"
