#!/usr/bin/env bash
# S3 no-submit provenance and source validation.  The caller owns output paths.
set -euo pipefail
readonly REPO="$1" OUT="$2" EXPECTED_SHA="$3" SOURCE_MANIFEST="$4" SOURCE_STATE="$5" SOURCE_CONFIG="$6"
readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
readonly ERROR_FILE="${OUT%.json}_error.txt"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -f "$SOURCE_MANIFEST" && test -f "$SOURCE_STATE" && test -f "$SOURCE_CONFIG"
test ! -e "$OUT" && test ! -e "$ERROR_FILE"
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export UPPE_USE_GPU=1 PYTHONPATH="$REPO/Filament_python"
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/provenance_v2.py" create --repo "$REPO" --output "${OUT%.json}_provenance_v2.json" --tracked Filament_python/KHz_filament/hr4e5s_s3.py Filament_python/KHz_filament/hr4e5s_streaming.py Filament_python/KHz_filament/hr4.py Filament_python/KHz_filament/hr4c_state.py Filament_python/KHz_filament/propagate.py Filament_python/KHz_filament/thermalization.py Filament_python/KHz_filament/slow_state.py Filament_python/tools/run_hr4e5s_s3.py Filament_python/tools/hr4e5s_s3.sbatch Filament_python/tools/hpc_ops/run_hr4e5s_s3_preflight.sh Filament_python/tools/hpc_ops/submit_hr4e5s_s3.sh --external "$SOURCE_MANIFEST" "$SOURCE_STATE" "$SOURCE_CONFIG" >/dev/null 2>"$ERROR_FILE"
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/provenance_v2.py" validate --repo "$REPO" --manifest "${OUT%.json}_provenance_v2.json" --require-hash-scope >/dev/null 2>>"$ERROR_FILE"
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/audit_batch_entry.py" --batch "$REPO/Filament_python/tools/hr4e5s_s3.sbatch" --fixed-python "$PYTHON" >"${OUT%.json}_batch_audit.json"
tmpdir="$(mktemp -d)"
trap 'rm -rf -- "$tmpdir"' EXIT
"$PYTHON" "$REPO/Filament_python/tools/run_hr4e5s_s3.py" prepare --source-manifest "$SOURCE_MANIFEST" --source-state "$SOURCE_STATE" --config "$SOURCE_CONFIG" --out "$tmpdir/input.json" >/dev/null
"$PYTHON" - "$OUT" "$EXPECTED_SHA" "$SOURCE_MANIFEST" "$SOURCE_STATE" "$SOURCE_CONFIG" "$tmpdir/input.json" <<'PY'
import hashlib,json,sys
out,sha,manifest_path,state_path,config_path,input_path=sys.argv[1:]
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
source=json.load(open(manifest_path,encoding='utf-8'))
case=json.load(open(input_path,encoding='utf-8'))
assert case['source_state_file_sha256']==source['hr3b_state_file_sha256']
assert case['source_state_array_sha256']==source['hr3b_state_sha256']
assert case['config_sha256']==source['config_sha256']
assert len(case['screen_records'])==48 and case['screen_indices']==list(range(case['peak_source_index']-24,case['peak_source_index']+24))
json.dump({'schema':'khz_filament.hr4e5s.s3.preflight.v1','status':'PASS','git_sha':sha,'source_manifest':manifest_path,'source_manifest_sha256':digest(manifest_path),'source_state':state_path,'source_state_file_sha256':source['hr3b_state_file_sha256'],'source_state_array_sha256':source['hr3b_state_sha256'],'source_config':config_path,'source_config_sha256':source['config_sha256'],'s3_input_manifest_sha256':digest(input_path),'screen_indices':case['screen_indices'],'gpu_matrix':{'batch':1,'streaming_repeat_1':2,'streaming_repeat_2':2}},open(out,'w',encoding='utf-8'),indent=2,sort_keys=True)
PY
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","preflight":"%s"}\n' "$OUT"
