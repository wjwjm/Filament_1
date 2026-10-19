#!/usr/bin/env bash
# Fault-OFF, same-runtime clean reference for S5-FINAL 249485 comparison.
# This launcher is external evidence tooling: REPO_DIR must be an immutable,
# clean worktree at EXPECTED_GIT_SHA.  It has no controller, fault, restart, or
# recovery path.
#SBATCH --job-name=hr4e5s-s5-final-clean
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
# Memory is deliberately left to the site 126-GB-per-GPU default: with three
# GPUs this is the same 378000M class recorded for 249485.  An explicit total
# --mem request is rejected by this site's GPU submit policy.
#SBATCH --time=12:00:00
set -euo pipefail

readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
: "${EXPECTED_GIT_SHA:?}" "${REPO_DIR:?}" "${RUN_ROOT:?}" "${LUT_WORKSPACE:?}" "${INPUT_MANIFEST:?}" "${CLEAN_CASE_MODE:?}"
test "$CLEAN_CASE_MODE" = clean_reference
test "$(git -C "$REPO_DIR" rev-parse HEAD)" = "$EXPECTED_GIT_SHA"
test -z "$(git -C "$REPO_DIR" status --porcelain=v1 --untracked-files=all)"
test -f "$INPUT_MANIFEST" && test -d "$LUT_WORKSPACE" && test "$(stat -c '%a' "$LUT_WORKSPACE")" = 700
unset HR4_S5_FAULT_ID HR4_S5_FAULT_SCREEN HR4_S5_FAULT_ONCE || true
test -z "${HR4_S5_FAULT_ID-}" && test -z "${HR4_S5_FAULT_SCREEN-}" && test -z "${HR4_S5_FAULT_ONCE-}"

source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export PYTHON REPO_DIR RUN_ROOT EXPECTED_GIT_SHA
export UPPE_USE_GPU=1 PYTHONPATH="$REPO_DIR/Filament_python" PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="$RUN_ROOT/pycache" CUPY_CACHE_DIR="$RUN_ROOT/cupy_cache" XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$LUT_WORKSPACE"
test "$("$PYTHON" -c 'from KHz_filament.device import debug_backend; print(debug_backend()["backend"])')" = cupy

readonly CLEAN_ROOT="$RUN_ROOT/clean"
readonly FINAL_RUNNER="$REPO_DIR/Filament_python/tools/run_hr4e5s_s5_final.py"
readonly S3_RUNNER="$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py"
export FINAL_RUNNER S3_RUNNER
test ! -e "$CLEAN_ROOT"
mkdir -m 700 -- "$CLEAN_ROOT" "$CLEAN_ROOT/identities" "$CLEAN_ROOT/consumers"

children=()
cleanup() {
  local status=$?
  trap - EXIT HUP INT TERM
  if [[ "$status" -ne 0 ]]; then
    local child
    for child in "${children[@]}"; do kill "$child" 2>/dev/null || true; done
    for child in "${children[@]}"; do wait "$child" 2>/dev/null || true; done
  fi
  exit "$status"
}
trap cleanup EXIT HUP INT TERM

launch_worker() {
  local actor="$1" identity="$2" stdout="$3" stderr="$4"; shift 4
  srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c '
    set -euo pipefail
    actor="$1"; identity="$2"; shift 2
    : "${SLURM_JOB_ID:?}" "${SLURM_STEP_ID:?}" "${SLURMD_NODENAME:?}"
    env -u HR4_S5_FAULT_ID -u HR4_S5_FAULT_SCREEN -u HR4_S5_FAULT_ONCE HR4E5S_WORKER_ID="$actor" "$PYTHON" "$FINAL_RUNNER" write-identity \
      --out "$identity" --actor "$actor" --pid "$$" --step-id "$SLURM_STEP_ID" --job-id "$SLURM_JOB_ID" --node "$SLURMD_NODENAME" --phase clean
    exec env -u HR4_S5_FAULT_ID -u HR4_S5_FAULT_SCREEN -u HR4_S5_FAULT_ONCE HR4E5S_WORKER_ID="$actor" "$@"
  ' bash "$actor" "$identity" "$@" >"$stdout" 2>"$stderr" &
  children+=("$!")
}

"$PYTHON" "$S3_RUNNER" initialize-stream --input "$INPUT_MANIFEST" --root "$CLEAN_ROOT/lifecycle" >"$CLEAN_ROOT/initialize.out"
launch_worker optical_producer "$CLEAN_ROOT/identities/optical_producer.json" "$CLEAN_ROOT/producer.out" "$CLEAN_ROOT/producer.err" \
  "$PYTHON" "$S3_RUNNER" optical --input "$INPUT_MANIFEST" --out-dir "$CLEAN_ROOT/optical" --stream-root "$CLEAN_ROOT/lifecycle"
producer_pid="${children[-1]}"
launch_worker hydro_consumer_0 "$CLEAN_ROOT/identities/hydro_consumer_0.json" "$CLEAN_ROOT/consumers/consumer_0.out" "$CLEAN_ROOT/consumers/consumer_0.err" \
  "$PYTHON" "$FINAL_RUNNER" consume --input "$INPUT_MANIFEST" --stream-root "$CLEAN_ROOT/lifecycle" --producer-complete "$CLEAN_ROOT/producer_complete" --actor hydro_consumer_0 --out "$CLEAN_ROOT/consumers/consumer_0.json"
hydro_0_pid="${children[-1]}"
launch_worker hydro_consumer_1 "$CLEAN_ROOT/identities/hydro_consumer_1.json" "$CLEAN_ROOT/consumers/consumer_1.out" "$CLEAN_ROOT/consumers/consumer_1.err" \
  "$PYTHON" "$FINAL_RUNNER" consume --input "$INPUT_MANIFEST" --stream-root "$CLEAN_ROOT/lifecycle" --producer-complete "$CLEAN_ROOT/producer_complete" --actor hydro_consumer_1 --out "$CLEAN_ROOT/consumers/consumer_1.json"
hydro_1_pid="${children[-1]}"

wait "$producer_pid"
touch "$CLEAN_ROOT/producer_complete"
wait "$hydro_0_pid"
wait "$hydro_1_pid"
"$PYTHON" "$FINAL_RUNNER" finalize --stream-root "$CLEAN_ROOT/lifecycle" --out "$CLEAN_ROOT/final.json" >"$CLEAN_ROOT/finalize.out"
"$PYTHON" "$FINAL_RUNNER" snapshot --stream-root "$CLEAN_ROOT/lifecycle" --out "$CLEAN_ROOT/final_lifecycle_audit.json" >"$CLEAN_ROOT/final_audit.out"

"$PYTHON" "$REPO_DIR/Filament_python/tools/audit_hr4e5s_s5_final_clean.py" \
  --clean-root "$CLEAN_ROOT" --out "$RUN_ROOT/clean_reference_terminal_audit.json"
