#!/usr/bin/env bash
# Submit exactly the mandatory P5 1/2/4-GPU strong-scaling matrix.
set -euo pipefail
readonly REPO="$1" RUN_ROOT="$2" EXPECTED_SHA="$3" PREFLIGHT="$4" MODE="${5:-submit}"
readonly PYTHON="/data/home/scvi806/.conda/envs/Filament_python/bin/python"
readonly BATCH="$REPO/Filament_python/tools/hr4e5p_parallel.sbatch"
readonly SOURCE_ROOT="/data/run01/scvi806/user_Wangjimin/projects/hr4e1_runs_e1b_be280dc"
readonly SOURCE_MANIFEST="$SOURCE_ROOT/post_reference/post_reference_manifest.json"
readonly SOURCE_STATE="$SOURCE_ROOT/E1B_hr3b_source.hr3b_delta_n_th.npy"
readonly INDICES="0,1,52,156,208,260,364,468,572,625,677,781,885,989,1041,1093,1197,1302,1406,1458,1510,1614,1718,1822,1875,1927,2031,2135,2239,2291,2343,2447,2500,2552,2656,2708,2760,2864,2968,3072,3125,3177,3281,3385,3489,3541,3593,3697,3802,3906,3958,4010,4114,4218,4322,4375,4427,4531,4635,4739,4791,4843,4947,5000,5052,5156,5208,5260,5364,5468,5572,5625,5677,5781,5885,5989,6041,6093,6197,6302,6406,6458,6510,6614,6718,6822,6875,6927,7031,7135,7239,7291,7343,7447,7552,7656,7708,7760,7827,7864,7968,8022,8072,8125,8177,8281,8385,8489,8541,8593,8697,8802,8906,8958,9000,9010,9114,9218,9322,9375,9427,9531,9635,9739,9791,9843,9947,10052,10156,10208,10260,10338,10364,10468,10572,10625,10677,10781,10885,10989,11041,11093,11197,11302,11406,11458,11510,11614,11718,11822,11875,11927,12000,12031,12135,12239,12291,12343,12447,12552,12656,12708,12760,12864,12968,13072,13125,13177,13281,13385,13489,13541,13593,13697,13802,13906,13958,14010,14114,14218,14322,14375,14427,14531,14635,14739,14791,14843,14947,14997,14998,14999"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -f "$BATCH" && test -f "$PREFLIGHT" && test -f "$SOURCE_MANIFEST" && test -f "$SOURCE_STATE" && test ! -e "$RUN_ROOT"
case "$MODE" in submit|no-submit) ;; *) exit 64 ;; esac
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/audit_batch_entry.py" --batch "$BATCH" --fixed-python "$PYTHON" >/dev/null
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export UPPE_USE_GPU=1 PYTHONPATH="$REPO/Filament_python"
mkdir -m 700 -- "$RUN_ROOT"
cp -- "$PREFLIGHT" "$RUN_ROOT/p5_submission_preflight.json"
"$PYTHON" - "$PREFLIGHT" "$EXPECTED_SHA" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding='utf-8'))
assert value['status']=='PASS' and value['git_sha']==sys.argv[2]
assert value['source_state_file_sha256']=='70677c01564ad089d985214e2767a221f10c5e1ef97f42a9f367a89edf81f467'
PY
for gpu in 1 2 4; do
  for repeat in 1 2; do
    name="p5_g$(printf '%02d' "$gpu")_r$(printf '%02d' "$repeat")"; dir="$RUN_ROOT/$name"
    "$PYTHON" "$REPO/Filament_python/tools/run_hr4e5p.py" prepare --source-manifest "$SOURCE_MANIFEST" --source-state "$SOURCE_STATE" --verified-source-state-file-sha256 70677c01564ad089d985214e2767a221f10c5e1ef97f42a9f367a89edf81f467 --screen-index "$INDICES" --out-dir "$dir"
    "$PYTHON" "$REPO/Filament_python/tools/run_hr4e5p.py" partition --input "$dir/validation_input.json" --block-size 8 --n-workers "$gpu" --out "$dir/partition.json"
  done
done
"$PYTHON" "$REPO/Filament_python/tools/run_hr4e5p.py" p5-manifest --input "$RUN_ROOT/p5_g01_r01/validation_input.json" --out-json "$RUN_ROOT/p5_input_manifest.json" --out-csv "$RUN_ROOT/p5_input_manifest.csv"
if [[ "$MODE" == no-submit ]]; then
  printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","mode":"no_submit","run_root":"%s","stage":"P5"}\n' "$RUN_ROOT"
  exit 0
fi
attempt="$RUN_ROOT/p5_submission_attempt.tsv"; receipt="$RUN_ROOT/p5_submission_receipt.tsv"
printf 'case_id\tjob_id\tgpu_count\trepetition\tblock_size\n' > "$attempt"
for gpu in 1 2 4; do
  for repeat in 1 2; do
    name="p5_g$(printf '%02d' "$gpu")_r$(printf '%02d' "$repeat")"; dir="$RUN_ROOT/$name"
    job="$(sbatch --parsable --job-name="e5p-$name" --gres="gpu:$gpu" --ntasks="$gpu" --output="$dir/slurm-%j.out" --error="$dir/slurm-%j.err" --export="ALL,EXPECTED_GIT_SHA=$EXPECTED_SHA,REPO_DIR=$REPO,CASE_DIR=$dir,MODE=parallel,STATE_JSON=$dir/parallel_state.json,PARTITION_JSON=$dir/partition.json,GPU_COUNT=$gpu,N_HYDRO_STEPS=1000,BATCH_INTERVALS=1,P5_RESOURCE_MONITOR_OUT=$dir/p5_gpu_memory_mib.csv" "$BATCH")"
    job="${job%%;*}"; [[ "$job" =~ ^[0-9]+$ ]] || exit 1
    printf '%s\t%s\t%s\t%s\t8\n' "$name" "$job" "$gpu" "$repeat" >> "$attempt"
  done
done
mv -- "$attempt" "$receipt"
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","mode":"submit","run_root":"%s","receipt":"%s","stage":"P5"}\n' "$RUN_ROOT" "$receipt"
