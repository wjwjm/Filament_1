#!/usr/bin/env bash
# One S3 allocation: either the 1-GPU batch reference or a real 2-GPU stream.
#SBATCH --job-name=hr4e5s-s3
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
set -euo pipefail

readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
: "${EXPECTED_GIT_SHA:?}" "${REPO_DIR:?}" "${RUN_ROOT:?}" "${LUT_WORKSPACE:?}" "${MODE:?}" "${INPUT_MANIFEST:?}"
test "$(git -C "$REPO_DIR" rev-parse HEAD)" = "$EXPECTED_GIT_SHA"
test -z "$(git -C "$REPO_DIR" status --porcelain=v1 --untracked-files=all)"
test -f "$INPUT_MANIFEST"
test "$LUT_WORKSPACE" = "$RUN_ROOT/lut_workspace"
test -d "$LUT_WORKSPACE" && test -w "$LUT_WORKSPACE"
test "$(stat -c '%a' "$LUT_WORKSPACE")" = 700
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export UPPE_USE_GPU=1 PYTHONPATH="$REPO_DIR/Filament_python" PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="$RUN_ROOT/pycache" CUPY_CACHE_DIR="$RUN_ROOT/cupy_cache" XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$LUT_WORKSPACE"
test "$("$PYTHON" -c 'from KHz_filament.device import debug_backend; print(debug_backend()["backend"])')" = cupy

case "$MODE" in
  batch)
    test "${SLURM_GPUS_ON_NODE:-1}" -ge 1
    "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py" optical --input "$INPUT_MANIFEST" --out-dir "$RUN_ROOT/batch_optical"
    "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py" batch-hydro --input "$INPUT_MANIFEST" --optical-dir "$RUN_ROOT/batch_optical" --out-dir "$RUN_ROOT/batch_hydro"
    ;;
  stream)
    : "${STREAM_ID:?}" "${BATCH_REFERENCE_ROOT:?}"
    test -d "$BATCH_REFERENCE_ROOT/batch_optical" && test -d "$BATCH_REFERENCE_ROOT/batch_hydro"
    test "${SLURM_GPUS_ON_NODE:-0}" -ge 2
    stream_root="$RUN_ROOT/$STREAM_ID"
    test ! -e "$stream_root"
    "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py" initialize-stream --input "$INPUT_MANIFEST" --root "$stream_root/lifecycle"
    # run_optical_path owns its destination and atomically creates optical/.
    mkdir -m 700 -- "$stream_root/consumer"
    producer_status=0
    consumer_status=0
    srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py" optical --input "$INPUT_MANIFEST" --out-dir "$stream_root/optical" --stream-root "$stream_root/lifecycle" >"$stream_root/producer.out" 2>"$stream_root/producer.err" &
    producer_pid=$!
    srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py" consume --input "$INPUT_MANIFEST" --stream-root "$stream_root/lifecycle" --producer-complete "$stream_root/producer_complete" --out "$stream_root/consumer/result.json" >"$stream_root/consumer.out" 2>"$stream_root/consumer.err" &
    consumer_pid=$!
    # Reap whichever concurrent side exits first.  A consumer can fail before
    # the producer is complete, so both failure directions must release the
    # sibling srun promptly instead of holding the second GPU allocation.
    first_status=0
    finished_pid=""
    wait -n -p finished_pid "$producer_pid" "$consumer_pid" || first_status=$?
    if [[ "$finished_pid" == "$producer_pid" ]]; then
      producer_status="$first_status"
      if [[ "$producer_status" -ne 0 ]]; then
        kill "$consumer_pid" 2>/dev/null || true
        wait "$consumer_pid" || consumer_status=$?
        exit "$producer_status"
      fi
      touch "$stream_root/producer_complete"
      wait "$consumer_pid" || consumer_status=$?
    else
      consumer_status="$first_status"
      if [[ "$consumer_status" -ne 0 ]]; then
        kill "$producer_pid" 2>/dev/null || true
        wait "$producer_pid" || producer_status=$?
        exit "$consumer_status"
      fi
      # Successful consumption requires producer_complete, which is created
      # only after a successful producer exit above.
      wait "$producer_pid" || producer_status=$?
    fi
    test "$producer_status" -eq 0 && test "$consumer_status" -eq 0
    "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py" finalize-stream --stream-root "$stream_root/lifecycle" --out "$stream_root/hr4e5s_s3_stream_final.json"
    "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py" compare --input "$INPUT_MANIFEST" --batch-optical-dir "$BATCH_REFERENCE_ROOT/batch_optical" --batch-hydro-dir "$BATCH_REFERENCE_ROOT/batch_hydro" --stream-optical-dir "$stream_root/optical" --stream-root "$stream_root/lifecycle" --out-dir "$stream_root/comparison"
    ;;
  *) echo "invalid S3 MODE=$MODE" >&2; exit 64 ;;
esac
