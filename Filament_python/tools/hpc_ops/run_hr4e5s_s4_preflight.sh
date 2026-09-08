#!/usr/bin/env bash
# S4 no-submit provenance, telemetry-entry, and private-LUT validation.
set -euo pipefail
readonly REPO="$1" RUN_ROOT="$2" OUT="$3" EXPECTED_SHA="$4" INPUT_MANIFEST="$5" SOURCE_MANIFEST="$6" SOURCE_STATE="$7" SOURCE_CONFIG="$8"
readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
readonly LUT_WORKSPACE="$RUN_ROOT/lut_workspace"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -f "$INPUT_MANIFEST" && test -f "$SOURCE_MANIFEST" && test -f "$SOURCE_STATE" && test -f "$SOURCE_CONFIG"
test ! -e "$RUN_ROOT" && test ! -e "$OUT"
umask 077
mkdir -m 700 -- "$RUN_ROOT"
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export UPPE_USE_GPU=1 PYTHONPATH="$REPO/Filament_python" PYTHONPYCACHEPREFIX="$RUN_ROOT/pycache" CUPY_CACHE_DIR="$RUN_ROOT/cupy_cache" XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/provenance_v2.py" create --repo "$REPO" --output "$RUN_ROOT/hr4e5s_s4_provenance_v2.json" --tracked Filament_python/KHz_filament/hr4e5s_streaming.py Filament_python/KHz_filament/hr4e5s_s3.py Filament_python/KHz_filament/hr4e5s_s4.py Filament_python/KHz_filament/hr4.py Filament_python/tools/run_hr4e5s_s3.py Filament_python/tools/run_hr4e5s_s4.py Filament_python/tools/hr4e5s_s4.sbatch Filament_python/tools/hpc_ops/run_hr4e5s_s4_preflight.sh Filament_python/tools/hpc_ops/submit_hr4e5s_s4.sh Filament_python/tools/hpc_ops/audit_hr4e5s_s3_lut_workspace.py --external "$INPUT_MANIFEST" "$SOURCE_MANIFEST" "$SOURCE_STATE" "$SOURCE_CONFIG" >/dev/null
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/provenance_v2.py" validate --repo "$REPO" --manifest "$RUN_ROOT/hr4e5s_s4_provenance_v2.json" --require-hash-scope >/dev/null
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/audit_batch_entry.py" --batch "$REPO/Filament_python/tools/hr4e5s_s4.sbatch" --fixed-python "$PYTHON" >"$RUN_ROOT/hr4e5s_s4_batch_audit.json"
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/audit_hr4e5s_s3_lut_workspace.py" --config "$SOURCE_CONFIG" --workspace "$LUT_WORKSPACE" --out "$RUN_ROOT/hr4e5s_s4_lut_workspace.json" >"$RUN_ROOT/hr4e5s_s4_lut_workspace.stdout.json" 2>"$RUN_ROOT/hr4e5s_s4_lut_workspace.stderr.txt"
"$PYTHON" - "$OUT" "$EXPECTED_SHA" "$RUN_ROOT" "$INPUT_MANIFEST" "$LUT_WORKSPACE" <<'PY'
import hashlib,json,sys
out,sha,root,input_path,lut=sys.argv[1:]
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
case=json.load(open(input_path,encoding='utf-8'))
assert len(case['screen_records'])==48 and case['screen_indices']==list(range(7998,8046))
json.dump({'schema':'khz_filament.hr4e5s.s4.preflight.v1','status':'PASS','git_sha':sha,'run_root':root,'input_manifest':input_path,'input_manifest_sha256':digest(input_path),'lut_workspace':lut,'current_generation':case['current_generation'],'screen_indices':case['screen_indices'],'gpu_matrix':{'replay_hydro':[1,2,4],'streaming':'1 optical + selected hydro'}},open(out,'w',encoding='utf-8'),indent=2,sort_keys=True)
PY
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","preflight":"%s"}\n' "$OUT"
