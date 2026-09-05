#!/usr/bin/env bash
set -euo pipefail
readonly REPO="$1" RUN_ROOT="$2" EXPECTED_SHA="$3"
readonly PYTHON="/data/home/scvi806/.conda/envs/Filament_python/bin/python"
readonly BATCH="$REPO/Filament_python/tools/hr4e5p_parallel.sbatch"
readonly SOURCE_ROOT="/data/run01/scvi806/user_Wangjimin/projects/hr4e1_runs_e1b_be280dc"
readonly SOURCE_MANIFEST="$SOURCE_ROOT/post_reference/post_reference_manifest.json"
readonly SOURCE_STATE="$SOURCE_ROOT/E1B_hr3b_source.hr3b_delta_n_th.npy"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -f "$BATCH" && test -f "$SOURCE_MANIFEST" && test -f "$SOURCE_STATE" && test ! -e "$RUN_ROOT"
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/audit_batch_entry.py" --batch "$BATCH" --fixed-python "$PYTHON" >/dev/null
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export UPPE_USE_GPU=1 PYTHONPATH="$REPO/Filament_python"
mkdir -m 700 -- "$RUN_ROOT"
"$PYTHON" "$REPO/Filament_python/tools/run_hr4e5p.py" audit --out "$RUN_ROOT/e5p_screen_independence_audit.json"
"$PYTHON" "$REPO/Filament_python/tools/run_hr4e5p.py" dry-run --n-screens 15000 --block-size 32 --n-workers 7 --out "$RUN_ROOT/full_screen_partition_dry_run.json"
readonly P3_INDICES="0,1,2500,5000,7827,8022,9000,10338,12000,14997,14998,14999"
readonly P4_INDICES="0,238,476,714,952,1190,1428,1666,1904,2142,2380,2618,2856,3094,3332,3570,3808,4046,4284,4522,4760,4998,5236,5474,5712,5950,6188,6426,6664,6902,7140,7378,7616,7854,8092,8330,8568,8806,9044,9282,9520,9758,9996,10234,10472,10710,10948,11186,11424,11662,11900,12138,12376,12614,12852,13090,13328,13566,13804,14042,14280,14518,14756,14999"
prepare_case() {
  local name="$1" indices="$2" workers="$3" block="$4" dir="$RUN_ROOT/$name"
  "$PYTHON" "$REPO/Filament_python/tools/run_hr4e5p.py" prepare --source-manifest "$SOURCE_MANIFEST" --source-state "$SOURCE_STATE" --screen-index "$indices" --out-dir "$dir"
  "$PYTHON" "$REPO/Filament_python/tools/run_hr4e5p.py" partition --input "$dir/validation_input.json" --block-size "$block" --n-workers "$workers" --out "$dir/partition.json"
}
prepare_case p3_serial "$P3_INDICES" 1 4
prepare_case p3_parallel_g1 "$P3_INDICES" 1 4
prepare_case p3_parallel_g2 "$P3_INDICES" 2 4
prepare_case p3_parallel_g4 "$P3_INDICES" 4 4
prepare_case p3_parallel_g7 "$P3_INDICES" 7 4
for block in 1 4 8 16 32; do prepare_case "p4_block_${block}" "$P4_INDICES" 1 "$block"; done
printf 'case_id\tjob_id\tgpu_count\tmode\n' > "$RUN_ROOT/submission_receipt.tsv"
submit_case() {
  local name="$1" gpus="$2" mode="$3" dir="$RUN_ROOT/$name" state partition job
  state="$dir/$( [[ "$mode" == serial ]] && printf serial_state.json || printf parallel_state.json )"
  partition="$dir/partition.json"
  job="$(sbatch --parsable --job-name="e5p-$name" --gres="gpu:$gpus" --output="$dir/slurm-%j.out" --error="$dir/slurm-%j.err" --export="ALL,EXPECTED_GIT_SHA=$EXPECTED_SHA,REPO_DIR=$REPO,CASE_DIR=$dir,MODE=$mode,STATE_JSON=$state,PARTITION_JSON=$partition,GPU_COUNT=$gpus,N_HYDRO_STEPS=1000,BATCH_INTERVALS=1" "$BATCH")"
  job="${job%%;*}"; [[ "$job" =~ ^[0-9]+$ ]]
  printf '%s\t%s\t%s\t%s\n' "$name" "$job" "$gpus" "$mode" >> "$RUN_ROOT/submission_receipt.tsv"
}
submit_case p3_serial 1 serial
submit_case p3_parallel_g1 1 parallel
submit_case p3_parallel_g2 2 parallel
submit_case p3_parallel_g4 4 parallel
submit_case p3_parallel_g7 7 parallel
for block in 1 4 8 16 32; do submit_case "p4_block_${block}" 1 parallel; done
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","run_root":"%s","receipt":"%s"}\n' "$RUN_ROOT" "$RUN_ROOT/submission_receipt.tsv"
