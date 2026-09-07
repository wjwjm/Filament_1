#!/usr/bin/env bash
# Submit exactly the authorized P4 one-GPU block-size matrix, fail closed on receipt.
set -euo pipefail
readonly REPO="$1" RUN_ROOT="$2" EXPECTED_SHA="$3" PREFLIGHT="$4" MODE="${5:-submit}"
readonly PYTHON="/data/home/scvi806/.conda/envs/Filament_python/bin/python"
readonly BATCH="$REPO/Filament_python/tools/hr4e5p_parallel.sbatch"
readonly SOURCE_ROOT="/data/run01/scvi806/user_Wangjimin/projects/hr4e1_runs_e1b_be280dc"
readonly SOURCE_MANIFEST="$SOURCE_ROOT/post_reference/post_reference_manifest.json"
readonly SOURCE_STATE="$SOURCE_ROOT/E1B_hr3b_source.hr3b_delta_n_th.npy"
readonly INDICES="0,1,208,625,1041,1458,1875,2291,2500,2708,3125,3541,3958,4375,4791,5000,5208,5625,6041,6458,6875,7291,7708,7827,8022,8125,8541,8958,9000,9375,9791,10208,10338,10625,11041,11458,11875,12000,12291,12708,13125,13541,13958,14375,14791,14997,14998,14999"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -f "$BATCH" && test -f "$PREFLIGHT" && test -f "$SOURCE_MANIFEST" && test -f "$SOURCE_STATE" && test ! -e "$RUN_ROOT"
case "$MODE" in submit|no-submit) ;; *) exit 64 ;; esac
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/audit_batch_entry.py" --batch "$BATCH" --fixed-python "$PYTHON" >/dev/null
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export UPPE_USE_GPU=1 PYTHONPATH="$REPO/Filament_python"
mkdir -m 700 -- "$RUN_ROOT"
cp -- "$PREFLIGHT" "$RUN_ROOT/p4_submission_preflight.json"
"$PYTHON" - "$PREFLIGHT" "$EXPECTED_SHA" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding='utf-8'))
assert value['status']=='PASS' and value['git_sha']==sys.argv[2]
assert value['source_state_file_sha256']=='70677c01564ad089d985214e2767a221f10c5e1ef97f42a9f367a89edf81f467'
PY
for block in 1 2 4 8 16 48; do
  for repeat in 1 2; do
    name="p4_b$(printf '%02d' "$block")_r$(printf '%02d' "$repeat")"
    dir="$RUN_ROOT/$name"
    "$PYTHON" "$REPO/Filament_python/tools/run_hr4e5p.py" prepare --source-manifest "$SOURCE_MANIFEST" --source-state "$SOURCE_STATE" --verified-source-state-file-sha256 70677c01564ad089d985214e2767a221f10c5e1ef97f42a9f367a89edf81f467 --screen-index "$INDICES" --out-dir "$dir"
    "$PYTHON" "$REPO/Filament_python/tools/run_hr4e5p.py" partition --input "$dir/validation_input.json" --block-size "$block" --n-workers 1 --out "$dir/partition.json"
  done
done
"$PYTHON" "$REPO/Filament_python/tools/run_hr4e5p.py" p4-manifest --input "$RUN_ROOT/p4_b01_r01/validation_input.json" --out-json "$RUN_ROOT/p4_input_manifest.json" --out-csv "$RUN_ROOT/p4_input_manifest.csv"
if [[ "$MODE" == no-submit ]]; then
  printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","mode":"no_submit","run_root":"%s","stage":"P4"}\n' "$RUN_ROOT"
  exit 0
fi
attempt="$RUN_ROOT/p4_submission_attempt.tsv"; receipt="$RUN_ROOT/p4_submission_receipt.tsv"
printf 'case_id\tjob_id\tblock_size\trepetition\tgpu_count\n' > "$attempt"
for block in 1 2 4 8 16 48; do
  for repeat in 1 2; do
    name="p4_b$(printf '%02d' "$block")_r$(printf '%02d' "$repeat")"; dir="$RUN_ROOT/$name"
    job="$(sbatch --parsable --job-name="e5p-$name" --gres=gpu:1 --ntasks=1 --output="$dir/slurm-%j.out" --error="$dir/slurm-%j.err" --export="ALL,EXPECTED_GIT_SHA=$EXPECTED_SHA,REPO_DIR=$REPO,CASE_DIR=$dir,MODE=parallel,STATE_JSON=$dir/parallel_state.json,PARTITION_JSON=$dir/partition.json,GPU_COUNT=1,N_HYDRO_STEPS=1000,BATCH_INTERVALS=1,P4_RESOURCE_MONITOR_OUT=$dir/gpu_memory_mib.csv" "$BATCH")"
    job="${job%%;*}"; [[ "$job" =~ ^[0-9]+$ ]] || exit 1
    printf '%s\t%s\t%s\t%s\t1\n' "$name" "$job" "$block" "$repeat" >> "$attempt"
  done
done
mv -- "$attempt" "$receipt"
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","mode":"submit","run_root":"%s","receipt":"%s","stage":"P4"}\n' "$RUN_ROOT" "$receipt"
