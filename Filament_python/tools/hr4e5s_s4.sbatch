#!/usr/bin/env bash
# One S4 allocation: exact persisted-POST replay or multi-consumer streaming.
#SBATCH --job-name=hr4e5s-s4
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
set -euo pipefail

readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
: "${EXPECTED_GIT_SHA:?}" "${REPO_DIR:?}" "${RUN_ROOT:?}" "${LUT_WORKSPACE:?}" "${MODE:?}" "${INPUT_MANIFEST:?}" "${HYDRO_WORKERS:?}" "${CASE_ID:?}" "${BATCH_REFERENCE_ROOT:?}"
test "$(git -C "$REPO_DIR" rev-parse HEAD)" = "$EXPECTED_GIT_SHA"
test -z "$(git -C "$REPO_DIR" status --porcelain=v1 --untracked-files=all)"
test -f "$INPUT_MANIFEST" && test -d "$LUT_WORKSPACE" && test -w "$LUT_WORKSPACE"
test "$(stat -c '%a' "$LUT_WORKSPACE")" = 700
test -d "$BATCH_REFERENCE_ROOT/batch_optical" && test -d "$BATCH_REFERENCE_ROOT/batch_hydro"
[[ "$HYDRO_WORKERS" =~ ^(1|2|4)$ ]]
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export UPPE_USE_GPU=1 PYTHONPATH="$REPO_DIR/Filament_python" PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="$RUN_ROOT/pycache" CUPY_CACHE_DIR="$RUN_ROOT/cupy_cache" XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$LUT_WORKSPACE"
test "$("$PYTHON" -c 'from KHz_filament.device import debug_backend; print(debug_backend()["backend"])')" = cupy

case_root="$RUN_ROOT/$CASE_ID"
test ! -e "$case_root"
mkdir -m 700 -- "$case_root" "$case_root/consumers"
marker="$case_root/producer_complete"
consumer_pids=()
active_pids=()

launch_consumers() {
  local index
  for ((index=0; index<HYDRO_WORKERS; index++)); do
    HR4E5S_WORKER_ID="hydro_consumer_${index}" \
      srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
      env HR4E5S_WORKER_ID="hydro_consumer_${index}" "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s4.py" consume \
        --input "$INPUT_MANIFEST" --stream-root "$case_root/lifecycle" --producer-complete "$marker" \
        --actor "hydro_consumer_${index}" --out "$case_root/consumers/consumer_${index}.json" \
        >"$case_root/consumers/consumer_${index}.out" 2>"$case_root/consumers/consumer_${index}.err" &
    consumer_pids+=("$!")
  done
}

stop_children() {
  local child
  for child in "$@"; do kill "$child" 2>/dev/null || true; done
  for child in "$@"; do wait "$child" 2>/dev/null || true; done
}

case "$MODE" in
  replay)
    test "${SLURM_GPUS_ON_NODE:-0}" -ge "$HYDRO_WORKERS"
    "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s4.py" prepare-replay --input "$INPUT_MANIFEST" --batch-optical-dir "$BATCH_REFERENCE_ROOT/batch_optical" --root "$case_root/lifecycle" >"$case_root/replay_prepare.out"
    "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s4.py" enqueue-replay --stream-root "$case_root/lifecycle" --producer-complete "$marker" --out "$case_root/replay_enqueue.json" >"$case_root/replay_enqueue.out" 2>"$case_root/replay_enqueue.err" &
    producer_pid=$!
    launch_consumers
    ;;
  stream)
    test "${SLURM_GPUS_ON_NODE:-0}" -ge $((HYDRO_WORKERS + 1))
    "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py" initialize-stream --input "$INPUT_MANIFEST" --root "$case_root/lifecycle" >"$case_root/initialize.out"
    srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
      env HR4E5S_WORKER_ID="optical_producer" "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py" optical \
        --input "$INPUT_MANIFEST" --out-dir "$case_root/optical" --stream-root "$case_root/lifecycle" >"$case_root/producer.out" 2>"$case_root/producer.err" &
    producer_pid=$!
    launch_consumers
    ;;
  *) echo "invalid S4 MODE=$MODE" >&2; exit 64 ;;
esac

active_pids=("$producer_pid" "${consumer_pids[@]}")
producer_finished=0
while ((${#active_pids[@]})); do
  status=0
  finished_pid=""
  wait -n -p finished_pid "${active_pids[@]}" || status=$?
  if [[ "$finished_pid" == "$producer_pid" ]]; then
    producer_finished=1
    if [[ "$status" -ne 0 ]]; then
      stop_children "${consumer_pids[@]}"
      exit "$status"
    fi
    if [[ "$MODE" == stream ]]; then touch "$marker"; fi
  elif [[ "$status" -ne 0 ]]; then
    stop_children "$producer_pid" "${consumer_pids[@]}"
    exit "$status"
  fi
  next=()
  for child in "${active_pids[@]}"; do [[ "$child" == "$finished_pid" ]] || next+=("$child"); done
  active_pids=("${next[@]}")
done
test "$producer_finished" -eq 1

"$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s4.py" finalize --stream-root "$case_root/lifecycle" --out "$case_root/stream_final.json" >"$case_root/finalize.out"
"$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s4.py" summarize --stream-root "$case_root/lifecycle" --out-dir "$case_root/telemetry" >"$case_root/telemetry.out"
if [[ "$MODE" == replay ]]; then
  "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s4.py" compare-replay --batch-hydro-dir "$BATCH_REFERENCE_ROOT/batch_hydro" --stream-root "$case_root/lifecycle" --out "$case_root/exact_replay_next.json" >"$case_root/exact.out"
else
  "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py" compare --input "$INPUT_MANIFEST" --batch-optical-dir "$BATCH_REFERENCE_ROOT/batch_optical" --batch-hydro-dir "$BATCH_REFERENCE_ROOT/batch_hydro" --stream-optical-dir "$case_root/optical" --stream-root "$case_root/lifecycle" --out-dir "$case_root/comparison" >"$case_root/exact.out"
fi
