#!/usr/bin/env bash
# Submit one S4 topology only after strict no-submit preflight.
set -euo pipefail
readonly REPO="$1" RUN_ROOT="$2" EXPECTED_SHA="$3" PREFLIGHT="$4" INPUT_MANIFEST="$5" BATCH_REFERENCE_ROOT="$6" MODE="$7" CASE_ID="$8" HYDRO_WORKERS="$9"
readonly BATCH="$REPO/Filament_python/tools/hr4e5s_s4.sbatch"
readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -f "$PREFLIGHT" && test -f "$INPUT_MANIFEST" && test -d "$BATCH_REFERENCE_ROOT/batch_optical" && test -d "$BATCH_REFERENCE_ROOT/batch_hydro"
[[ "$MODE" =~ ^(replay|stream)$ ]] && [[ "$HYDRO_WORKERS" =~ ^(1|2|4)$ ]]
test ! -e "$RUN_ROOT/${CASE_ID}_submission_receipt.tsv"
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/audit_batch_entry.py" --batch "$BATCH" --fixed-python "$PYTHON" >/dev/null
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
"$PYTHON" - "$PREFLIGHT" "$EXPECTED_SHA" "$RUN_ROOT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x['status']=='PASS' and x['git_sha']==sys.argv[2] and x['run_root']==sys.argv[3]
PY
if [[ "$MODE" == stream ]]; then gpus=$((HYDRO_WORKERS+1)); else gpus="$HYDRO_WORKERS"; fi
job="$(sbatch --parsable --job-name="e5s-s4-${CASE_ID}" --gres="gpu:${gpus}" --ntasks="$gpus" --output="$RUN_ROOT/${CASE_ID}-%j.out" --error="$RUN_ROOT/${CASE_ID}-%j.err" --export="ALL,EXPECTED_GIT_SHA=$EXPECTED_SHA,REPO_DIR=$REPO,RUN_ROOT=$RUN_ROOT,LUT_WORKSPACE=$RUN_ROOT/lut_workspace,MODE=$MODE,CASE_ID=$CASE_ID,HYDRO_WORKERS=$HYDRO_WORKERS,INPUT_MANIFEST=$INPUT_MANIFEST,BATCH_REFERENCE_ROOT=$BATCH_REFERENCE_ROOT" "$BATCH")"
job="${job%%;*}"; [[ "$job" =~ ^[0-9]+$ ]] || exit 1
printf 'case_id\tmode\thydro_workers\tgpu_count\tjob_id\n%s\t%s\t%s\t%s\t%s\n' "$CASE_ID" "$MODE" "$HYDRO_WORKERS" "$gpus" "$job" > "$RUN_ROOT/${CASE_ID}_submission_receipt.tsv"
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","job_id":"%s"}\n' "$job"
