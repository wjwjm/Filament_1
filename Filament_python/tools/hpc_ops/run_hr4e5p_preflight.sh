#!/usr/bin/env bash
set -euo pipefail
readonly REPO="$1" OUT="$2" EXPECTED_SHA="$3"
readonly PYTHON="/data/home/scvi806/.conda/envs/Filament_python/bin/python"
readonly PROVENANCE="${OUT%.json}_provenance_v2.json"
readonly AUDIT="${OUT%.json}_screen_independence_audit.json"
readonly ERROR_FILE="${OUT%.json}_preflight_error.txt"
readonly SOURCE_ROOT="/data/run01/scvi806/user_Wangjimin/projects/hr4e1_runs_e1b_be280dc"
readonly SOURCE_MANIFEST="$SOURCE_ROOT/post_reference/post_reference_manifest.json"
readonly SOURCE_STATE="$SOURCE_ROOT/E1B_hr3b_source.hr3b_delta_n_th.npy"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test ! -e "$OUT" && test ! -e "$PROVENANCE" && test ! -e "$AUDIT" && test ! -e "$ERROR_FILE"
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export UPPE_USE_GPU=1 PYTHONPATH="$REPO/Filament_python"
if ! "$PYTHON" "$REPO/Filament_python/tools/hpc_ops/provenance_v2.py" create --repo "$REPO" --output "$PROVENANCE" \
  --tracked Filament_python/KHz_filament/hr4.py Filament_python/KHz_filament/hr4c_state.py Filament_python/KHz_filament/hr4e_domain.py Filament_python/KHz_filament/hr4e5_parallel.py Filament_python/KHz_filament/hr4e_timestep.py Filament_python/tools/run_hr4e5p.py Filament_python/tools/hr4e5p_parallel.sbatch Filament_python/tools/hpc_ops/run_hr4e5p_preflight.sh Filament_python/tools/hpc_ops/submit_hr4e5p_stage1.sh Filament_python/tools/hpc_ops/stage_hr4e5p_worktree.sh \
  --external "$SOURCE_MANIFEST" "$SOURCE_STATE" >/dev/null 2>"$ERROR_FILE"; then
  printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":false,"state":"provenance_create_failed","error_file":"%s"}\n' "$ERROR_FILE"
  exit 0
fi
if ! "$PYTHON" "$REPO/Filament_python/tools/hpc_ops/provenance_v2.py" validate --repo "$REPO" --manifest "$PROVENANCE" --require-hash-scope >/dev/null 2>"$ERROR_FILE"; then
  printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":false,"state":"provenance_validate_failed","error_file":"%s"}\n' "$ERROR_FILE"
  exit 0
fi
"$PYTHON" "$REPO/Filament_python/tools/run_hr4e5p.py" audit --out "$AUDIT"
"$PYTHON" - "$OUT" "$PROVENANCE" "$AUDIT" "$SOURCE_MANIFEST" "$SOURCE_STATE" "$EXPECTED_SHA" <<'PY'
import hashlib,json,sys
out,provenance,audit,manifest_path,state_path,sha=sys.argv[1:]
def file_sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as handle:
        for block in iter(lambda: handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()
manifest=json.load(open(manifest_path,encoding='utf-8'))
audit_value=json.load(open(audit,encoding='utf-8'))
assert audit_value['status']=='P1_SCREEN_INDEPENDENCE_CONFIRMED'
state_hash=manifest['hr3b_state_file_sha256']
assert state_hash=='70677c01564ad089d985214e2767a221f10c5e1ef97f42a9f367a89edf81f467'
json.dump({'schema':'khz_filament.hr4e5p.preflight.v1','status':'PASS','git_sha':sha,'provenance':provenance,'screen_independence_audit':audit,'source_manifest':manifest_path,'source_manifest_sha256':file_sha256(manifest_path),'source_state':state_path,'source_state_file_sha256':state_hash,'source_state_array_sha256':manifest['hr3b_state_sha256'],'source_screen_count':manifest['hr3b_state_shape'][0],'dtype':'float64','backend_required':'cupy','accepted_grid':'D0_301x351_10um','dt_hydro_s':1e-6},open(out,'w',encoding='utf-8'),indent=2,sort_keys=True)
PY
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","preflight":"%s","provenance":"%s","audit":"%s"}\n' "$OUT" "$PROVENANCE" "$AUDIT"
