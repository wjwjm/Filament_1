#!/usr/bin/env bash
#SBATCH --job-name=e5-wp2-gpu-bind
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail

: "${EXPECTED_GIT_SHA:?}" "${REPO_DIR:?}" "${RUN_ROOT:?}" "${HYDRO_WORKERS:?}" "${CAMPAIGN_ID:?}"
[[ "$RUN_ROOT" == /data/run01/scvi806/* ]]
[[ "$HYDRO_WORKERS" == 2 || "$HYDRO_WORKERS" == 4 ]]
test "$(git -C "$REPO_DIR" rev-parse HEAD)" = "$EXPECTED_GIT_SHA"
test -z "$(git -C "$REPO_DIR" status --porcelain=v1 --untracked-files=all)"
test ! -e "$RUN_ROOT"
: "${SLURM_JOB_ID:?}" "${CUDA_VISIBLE_DEVICES:?}"
mkdir -m 700 -- "$RUN_ROOT"
mkdir -m 700 -- "$RUN_ROOT/runtime" "$RUN_ROOT/runtime/home" "$RUN_ROOT/runtime/tmp" \
  "$RUN_ROOT/runtime/xdg" "$RUN_ROOT/runtime/cupy" "$RUN_ROOT/runtime/pycache" \
  "$RUN_ROOT/case"
export HOME="$RUN_ROOT/runtime/home" TMPDIR="$RUN_ROOT/runtime/tmp"
export XDG_CACHE_HOME="$RUN_ROOT/runtime/xdg" CUPY_CACHE_DIR="$RUN_ROOT/runtime/cupy"
export PYTHONPYCACHEPREFIX="$RUN_ROOT/runtime/pycache"
export PYTHONPATH="$REPO_DIR/Filament_python" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export UPPE_USE_GPU=0
readonly ENV_PREFIX=/data/home/scvi806/.conda/envs/Filament_python
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"
readonly PYTHON="$ENV_PREFIX/bin/python"
test -x "$PYTHON"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$REPO_DIR"
printf 'job=%s host=%s sha=%s visible=%s job_gpus=%s H=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$EXPECTED_GIT_SHA" "$CUDA_VISIBLE_DEVICES" \
  "${SLURM_JOB_GPUS:-UNSET}" "$HYDRO_WORKERS"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu \
  --format=csv > "$RUN_ROOT/gpu_startup.csv"

sample_gpu() {
  while :; do
    printf 'timestamp_utc=%s\n' "$(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
      --format=csv,noheader || true
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory \
      --format=csv,noheader || true
    sleep 2
  done
}
sample_gpu > "$RUN_ROOT/gpu_samples.txt" 2>&1 &
sampler_pid=$!
on_exit() {
  local rc=$?
  trap - EXIT
  kill "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
  nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu \
    --format=csv > "$RUN_ROOT/gpu_terminal.csv" 2>&1 || true
  exit "$rc"
}
trap on_exit EXIT

"$PYTHON" -s -B "$REPO_DIR/Filament_python/tools/run_e5_1b_wp2_gpu_preflight.py" \
  --run-root "$RUN_ROOT/case" --campaign-id "$CAMPAIGN_ID" \
  --hydro-workers "$HYDRO_WORKERS"
