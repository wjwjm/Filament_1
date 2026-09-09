#!/usr/bin/env bash
# One S4R hydro-only, 384-screen, block-8 replay allocation.
#SBATCH --job-name=hr4e5s-s4r
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
set -euo pipefail

readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
: "${EXPECTED_GIT_SHA:?}" "${REPO_DIR:?}" "${RUN_ROOT:?}" "${INPUT_MANIFEST:?}" "${HYDRO_WORKERS:?}" "${CASE_ID:?}"
test "$(git -C "$REPO_DIR" rev-parse HEAD)" = "$EXPECTED_GIT_SHA"
test -z "$(git -C "$REPO_DIR" status --porcelain=v1 --untracked-files=all)"
test -f "$INPUT_MANIFEST"
[[ "$HYDRO_WORKERS" =~ ^(1|2|4|8)$ ]]
test "${SLURM_GPUS_ON_NODE:-0}" -ge "$HYDRO_WORKERS"

source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export UPPE_USE_GPU=1 PYTHONPATH="$REPO_DIR/Filament_python" PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="$RUN_ROOT/pycache" CUPY_CACHE_DIR="$RUN_ROOT/cupy_cache" XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
test "$("$PYTHON" -c 'from KHz_filament.device import debug_backend; print(debug_backend()["backend"])')" = cupy

case_root="$RUN_ROOT/$CASE_ID"
test ! -e "$case_root"
mkdir -m 700 -- "$case_root" "$case_root/consumers" "$case_root/timing"
export HR4E5S_TIMING_DIR="$case_root/timing"
export HR4E5S_WORKER_ID="replay_coordinator"
marker="$case_root/producer_complete"

"$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s4r.py" prepare-replay --input "$INPUT_MANIFEST" --root "$case_root/lifecycle" >"$case_root/replay_prepare.out" 2>"$case_root/replay_prepare.err"
"$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s4r.py" enqueue-replay --stream-root "$case_root/lifecycle" --producer-complete "$marker" --out "$case_root/replay_enqueue.json" >"$case_root/replay_enqueue.out" 2>"$case_root/replay_enqueue.err" &
producer_pid=$!

consumer_pids=()
for ((index=0; index<HYDRO_WORKERS; index++)); do
  srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    env HR4E5S_WORKER_ID="hydro_consumer_${index}" HR4E5S_TIMING_DIR="$case_root/timing" \
    "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s4r.py" consume \
      --input "$INPUT_MANIFEST" --stream-root "$case_root/lifecycle" --producer-complete "$marker" \
      --actor "hydro_consumer_${index}" --out "$case_root/consumers/consumer_${index}.json" \
      >"$case_root/consumers/consumer_${index}.out" 2>"$case_root/consumers/consumer_${index}.err" &
  consumer_pids+=("$!")
done

stop_children() {
  local child
  for child in "$@"; do kill "$child" 2>/dev/null || true; done
  for child in "$@"; do wait "$child" 2>/dev/null || true; done
}

active_pids=("$producer_pid" "${consumer_pids[@]}")
while ((${#active_pids[@]})); do
  status=0; finished_pid=""
  wait -n -p finished_pid "${active_pids[@]}" || status=$?
  if [[ "$status" -ne 0 ]]; then
    stop_children "$producer_pid" "${consumer_pids[@]}"
    exit "$status"
  fi
  next=()
  for child in "${active_pids[@]}"; do [[ "$child" == "$finished_pid" ]] || next+=("$child"); done
  active_pids=("${next[@]}")
done

"$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s4r.py" finalize --stream-root "$case_root/lifecycle" --out "$case_root/stream_final.json" >"$case_root/finalize.out" 2>"$case_root/finalize.err"
"$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s4r.py" summarize --stream-root "$case_root/lifecycle" --timing-dir "$case_root/timing" --out-dir "$case_root/telemetry" --hydro-workers "$HYDRO_WORKERS" >"$case_root/telemetry.out" 2>"$case_root/telemetry.err"
printf '{"schema":"khz_filament.hr4e5s.s4r.case.v1","status":"PASS","case_id":"%s","hydro_workers":%s,"completed_utc":"%s"}\n' "$CASE_ID" "$HYDRO_WORKERS" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$case_root/case_result.json"
