#!/usr/bin/env bash
# Submit one S3 phase only: batch first, streaming only after terminal batch success.
set -euo pipefail
readonly REPO="$1" RUN_ROOT="$2" EXPECTED_SHA="$3" PREFLIGHT="$4" SOURCE_MANIFEST="$5" SOURCE_STATE="$6" SOURCE_CONFIG="$7" SUBMIT_PHASE="$8"
readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
readonly BATCH="$REPO/Filament_python/tools/hr4e5s_s3.sbatch"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -d "$RUN_ROOT" && test -f "$PREFLIGHT" && test -f "$SOURCE_MANIFEST" && test -f "$SOURCE_STATE" && test -f "$SOURCE_CONFIG"
test -f "$RUN_ROOT/hr4e5s_s3_input_manifest.json" && test -d "$RUN_ROOT/lut_workspace"
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/audit_batch_entry.py" --batch "$BATCH" --fixed-python "$PYTHON" >/dev/null
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export UPPE_USE_GPU=1 PYTHONPATH="$REPO/Filament_python"
"$PYTHON" - "$PREFLIGHT" "$EXPECTED_SHA" "$RUN_ROOT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x['status']=='PASS' and x['git_sha']==sys.argv[2]
assert x['run_root']==sys.argv[3]
assert x['gpu_matrix']=={'batch':1,'streaming_repeat_1':2,'streaming_repeat_2':2}
assert x['lut_workspace']=='lut_workspace' and len(x['lut_records']) >= 2
PY
input="$RUN_ROOT/hr4e5s_s3_input_manifest.json"
case "$SUBMIT_PHASE" in
  batch)
    test ! -e "$RUN_ROOT/hr4e5s_s3_batch_submission_receipt.tsv"
    batch_job="$(sbatch --parsable --job-name=e5s-s3-batch --gres=gpu:1 --ntasks=1 --output="$RUN_ROOT/batch-%j.out" --error="$RUN_ROOT/batch-%j.err" --export="ALL,EXPECTED_GIT_SHA=$EXPECTED_SHA,REPO_DIR=$REPO,RUN_ROOT=$RUN_ROOT,LUT_WORKSPACE=$RUN_ROOT/lut_workspace,MODE=batch,INPUT_MANIFEST=$input" "$BATCH")"
    batch_job="${batch_job%%;*}"; [[ "$batch_job" =~ ^[0-9]+$ ]] || exit 1
    printf 'case_id\tjob_id\tgpu_count\tdependency\n' > "$RUN_ROOT/hr4e5s_s3_batch_submission_attempt.tsv"
    printf 'batch\t%s\t1\tnone\n' "$batch_job" >> "$RUN_ROOT/hr4e5s_s3_batch_submission_attempt.tsv"
    mv -- "$RUN_ROOT/hr4e5s_s3_batch_submission_attempt.tsv" "$RUN_ROOT/hr4e5s_s3_batch_submission_receipt.tsv"
    ;;
  streaming)
    test -f "$RUN_ROOT/hr4e5s_s3_batch_submission_receipt.tsv"
    test ! -e "$RUN_ROOT/hr4e5s_s3_streaming_submission_receipt.tsv"
    batch_job="$(awk 'NR==2 {print $2}' "$RUN_ROOT/hr4e5s_s3_batch_submission_receipt.tsv")"
    [[ "$batch_job" =~ ^[0-9]+$ ]] || exit 1
    test "$(sacct -j "$batch_job" --format=State,ExitCode --parsable2 --noheader | head -n 1)" = 'COMPLETED|0:0'
    printf 'case_id\tjob_id\tgpu_count\tdependency\n' > "$RUN_ROOT/hr4e5s_s3_streaming_submission_attempt.tsv"
    for stream in stream_repeat_1 stream_repeat_2; do
      job="$(sbatch --parsable --job-name="e5s-s3-$stream" --gres=gpu:2 --ntasks=2 --output="$RUN_ROOT/$stream-%j.out" --error="$RUN_ROOT/$stream-%j.err" --export="ALL,EXPECTED_GIT_SHA=$EXPECTED_SHA,REPO_DIR=$REPO,RUN_ROOT=$RUN_ROOT,LUT_WORKSPACE=$RUN_ROOT/lut_workspace,MODE=stream,STREAM_ID=$stream,INPUT_MANIFEST=$input" "$BATCH")"
      job="${job%%;*}"; [[ "$job" =~ ^[0-9]+$ ]] || exit 1
      printf '%s\t%s\t2\tnone; batch sacct COMPLETED/0:0\n' "$stream" "$job" >> "$RUN_ROOT/hr4e5s_s3_streaming_submission_attempt.tsv"
    done
    mv -- "$RUN_ROOT/hr4e5s_s3_streaming_submission_attempt.tsv" "$RUN_ROOT/hr4e5s_s3_streaming_submission_receipt.tsv"
    ;;
  *) echo "invalid S3 submit phase=$SUBMIT_PHASE" >&2; exit 64 ;;
esac
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","run_root":"%s","phase":"%s"}\n' "$RUN_ROOT" "$SUBMIT_PHASE"
