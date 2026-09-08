#!/usr/bin/env bash
# Submit only the S3 1x batch + 2x two-GPU streaming matrix after a PASS preflight.
set -euo pipefail
readonly REPO="$1" RUN_ROOT="$2" EXPECTED_SHA="$3" PREFLIGHT="$4" SOURCE_MANIFEST="$5" SOURCE_STATE="$6" SOURCE_CONFIG="$7"
readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
readonly BATCH="$REPO/Filament_python/tools/hr4e5s_s3.sbatch"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test ! -e "$RUN_ROOT" && test -f "$PREFLIGHT" && test -f "$SOURCE_MANIFEST" && test -f "$SOURCE_STATE" && test -f "$SOURCE_CONFIG"
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/audit_batch_entry.py" --batch "$BATCH" --fixed-python "$PYTHON" >/dev/null
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export UPPE_USE_GPU=1 PYTHONPATH="$REPO/Filament_python"
"$PYTHON" - "$PREFLIGHT" "$EXPECTED_SHA" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x['status']=='PASS' and x['git_sha']==sys.argv[2]
assert x['gpu_matrix']=={'batch':1,'streaming_repeat_1':2,'streaming_repeat_2':2}
PY
mkdir -m 700 -- "$RUN_ROOT"
cp -- "$PREFLIGHT" "$RUN_ROOT/hr4e5s_s3_preflight.json"
"$PYTHON" "$REPO/Filament_python/tools/run_hr4e5s_s3.py" prepare --source-manifest "$SOURCE_MANIFEST" --source-state "$SOURCE_STATE" --config "$SOURCE_CONFIG" --out "$RUN_ROOT/hr4e5s_s3_input_manifest.json" >/dev/null
input="$RUN_ROOT/hr4e5s_s3_input_manifest.json"
batch_job="$(sbatch --parsable --job-name=e5s-s3-batch --gres=gpu:1 --ntasks=1 --output="$RUN_ROOT/batch-%j.out" --error="$RUN_ROOT/batch-%j.err" --export="ALL,EXPECTED_GIT_SHA=$EXPECTED_SHA,REPO_DIR=$REPO,RUN_ROOT=$RUN_ROOT,MODE=batch,INPUT_MANIFEST=$input" "$BATCH")"
batch_job="${batch_job%%;*}"; [[ "$batch_job" =~ ^[0-9]+$ ]] || exit 1
printf 'case_id\tjob_id\tgpu_count\tdependency\n' > "$RUN_ROOT/hr4e5s_s3_submission_attempt.tsv"
printf 'batch\t%s\t1\tnone\n' "$batch_job" >> "$RUN_ROOT/hr4e5s_s3_submission_attempt.tsv"
for stream in stream_repeat_1 stream_repeat_2; do
  job="$(sbatch --parsable --dependency="afterok:$batch_job" --job-name="e5s-s3-$stream" --gres=gpu:2 --ntasks=2 --output="$RUN_ROOT/$stream-%j.out" --error="$RUN_ROOT/$stream-%j.err" --export="ALL,EXPECTED_GIT_SHA=$EXPECTED_SHA,REPO_DIR=$REPO,RUN_ROOT=$RUN_ROOT,MODE=stream,STREAM_ID=$stream,INPUT_MANIFEST=$input" "$BATCH")"
  job="${job%%;*}"; [[ "$job" =~ ^[0-9]+$ ]] || exit 1
  printf '%s\t%s\t2\tafterok:%s\n' "$stream" "$job" "$batch_job" >> "$RUN_ROOT/hr4e5s_s3_submission_attempt.tsv"
done
mv -- "$RUN_ROOT/hr4e5s_s3_submission_attempt.tsv" "$RUN_ROOT/hr4e5s_s3_submission_receipt.tsv"
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","run_root":"%s","receipt":"%s"}\n' "$RUN_ROOT" "$RUN_ROOT/hr4e5s_s3_submission_receipt.tsv"
