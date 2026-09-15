#!/usr/bin/env bash
# One S5-FINAL allocation.  The initial job is externally interrupted once;
# the recovery job performs one serial reconstruction before its three workers.
#SBATCH --job-name=hr4e5s-s5-final
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
set -euo pipefail

readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
: "${EXPECTED_GIT_SHA:?}" "${REPO_DIR:?}" "${RUN_ROOT:?}" "${LUT_WORKSPACE:?}" "${INPUT_MANIFEST:?}" "${CASE_MODE:?}" "${REFERENCE_CASE_ROOT:?}"
case "$CASE_MODE" in initial|recovery) ;; *) echo "invalid S5-FINAL case mode" >&2; exit 64 ;; esac
test "$(git -C "$REPO_DIR" rev-parse HEAD)" = "$EXPECTED_GIT_SHA"
test -z "$(git -C "$REPO_DIR" status --porcelain=v1 --untracked-files=all)"
test -f "$INPUT_MANIFEST" && test -d "$LUT_WORKSPACE" && test "$(stat -c '%a' "$LUT_WORKSPACE")" = 700
test -d "$REFERENCE_CASE_ROOT/lifecycle" && test -d "$REFERENCE_CASE_ROOT/optical"
unset HR4_S5_FAULT_ID HR4_S5_FAULT_SCREEN HR4_S5_FAULT_ONCE || true
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export PYTHON REPO_DIR RUN_ROOT EXPECTED_GIT_SHA
export UPPE_USE_GPU=1 PYTHONPATH="$REPO_DIR/Filament_python" PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="$RUN_ROOT/pycache" CUPY_CACHE_DIR="$RUN_ROOT/cupy_cache" XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$LUT_WORKSPACE"
test "$("$PYTHON" -c 'from KHz_filament.device import debug_backend; print(debug_backend()["backend"])')" = cupy

readonly CASE_ROOT="$RUN_ROOT/scenario"
readonly FINAL_RUNNER="$REPO_DIR/Filament_python/tools/run_hr4e5s_s5_final.py"
readonly S3_RUNNER="$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py"
export FINAL_RUNNER S3_RUNNER

worker_pids=()
consumer_pids=()
producer_pid=""

launch_worker() {
  local actor="$1" identity="$2" phase="$3"; shift 3
  srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    bash -c '
      set -euo pipefail
      actor="$1"; identity="$2"; phase="$3"; shift 3
      : "${SLURM_JOB_ID:?}" "${SLURM_STEP_ID:?}" "${SLURMD_NODENAME:?}"
      env HR4E5S_WORKER_ID="$actor" "$PYTHON" "$FINAL_RUNNER" write-identity \
        --out "$identity" --actor "$actor" --pid "$$" --step-id "$SLURM_STEP_ID" \
        --job-id "$SLURM_JOB_ID" --node "$SLURMD_NODENAME" --phase "$phase"
      exec env HR4E5S_WORKER_ID="$actor" "$@"
    ' bash "$actor" "$identity" "$phase" "$@" &
  worker_pids+=("$!")
}

stop_children() {
  local child
  for child in "$@"; do kill "$child" 2>/dev/null || true; done
  for child in "$@"; do wait "$child" 2>/dev/null || true; done
}

wait_for_pair() {
  local marker="$1" phase="$2"; shift 2
  local -a active=("$@") next=()
  local status=0 finished="" producer_finished=0
  while ((${#active[@]})); do
    status=0; finished=""
    wait -n -p finished "${active[@]}" || status=$?
    if [[ "$finished" == "$producer_pid" ]]; then
      producer_finished=1
      if [[ "$status" -eq 0 ]]; then touch "$marker"; fi
    fi
    if [[ "$status" -ne 0 ]]; then
      printf '{"schema":"khz_filament.hr4e5s.s5_final.v1","kind":"failure_propagation","status":"FAIL","phase":"%s","failed_srun_pid":%s,"exit_code":%s}\n' "$phase" "$finished" "$status" >"$CASE_ROOT/${phase}_failure_propagation.json"
      stop_children "${active[@]}"
      return "$status"
    fi
    next=()
    for child in "${active[@]}"; do [[ "$child" == "$finished" ]] || next+=("$child"); done
    active=("${next[@]}")
  done
  test "$producer_finished" -eq 1
}

if [[ "$CASE_MODE" == initial ]]; then
  test ! -e "$CASE_ROOT"
  mkdir -m 700 -- "$CASE_ROOT" "$CASE_ROOT/identities" "$CASE_ROOT/consumers"
  "$PYTHON" "$S3_RUNNER" initialize-stream --input "$INPUT_MANIFEST" --root "$CASE_ROOT/lifecycle" >"$CASE_ROOT/initialize.out"
  launch_worker optical_producer "$CASE_ROOT/identities/optical_producer.json" initial \
    "$PYTHON" "$S3_RUNNER" optical --input "$INPUT_MANIFEST" --out-dir "$CASE_ROOT/optical" --stream-root "$CASE_ROOT/lifecycle" \
    >"$CASE_ROOT/producer.out" 2>"$CASE_ROOT/producer.err"
  producer_pid="${worker_pids[-1]}"
  for index in 0 1; do
    launch_worker "hydro_consumer_${index}" "$CASE_ROOT/identities/hydro_consumer_${index}.json" initial \
      "$PYTHON" "$FINAL_RUNNER" consume --input "$INPUT_MANIFEST" --stream-root "$CASE_ROOT/lifecycle" --producer-complete "$CASE_ROOT/producer_complete" \
      --actor "hydro_consumer_${index}" --out "$CASE_ROOT/consumers/consumer_${index}.json" \
      >"$CASE_ROOT/consumers/consumer_${index}.out" 2>"$CASE_ROOT/consumers/consumer_${index}.err"
    consumer_pids+=("${worker_pids[-1]}")
  done
  initial_status=0
  wait_for_pair "$CASE_ROOT/producer_complete" initial "$producer_pid" "${consumer_pids[@]}" || initial_status=$?
  if [[ "$initial_status" -eq 0 ]]; then
    printf '{"schema":"khz_filament.hr4e5s.s5_final.v1","kind":"scenario_coverage","status":"FAIL","reason":"worker_loss_not_triggered"}\n' >"$CASE_ROOT/scenario_coverage.json"
    exit 41
  fi
  exit "$initial_status"
fi

# Recovery cannot start until a separate controller has proven the old job
# quiescent and frozen the input snapshot/effects contract.
test -d "$CASE_ROOT/lifecycle" && test -f "$CASE_ROOT/old_job_quiescence.json"
test -f "$CASE_ROOT/interrupted_state_inventory.json" && test -f "$CASE_ROOT/expected_recovery_effects.json"
test ! -e "$CASE_ROOT/recovery"
mkdir -m 700 -- "$CASE_ROOT/recovery" "$CASE_ROOT/recovery/identities" "$CASE_ROOT/recovery/consumers"
"$PYTHON" "$FINAL_RUNNER" bootstrap --stream-root "$CASE_ROOT/lifecycle" --effects "$CASE_ROOT/expected_recovery_effects.json" --out "$CASE_ROOT/recovery/restart_reconstructed.json" --runtime-sha "$EXPECTED_GIT_SHA" >"$CASE_ROOT/recovery/bootstrap.out"
"$PYTHON" "$FINAL_RUNNER" validate-bootstrap --stream-root "$CASE_ROOT/lifecycle" --receipt "$CASE_ROOT/recovery/restart_reconstructed.json" --runtime-sha "$EXPECTED_GIT_SHA" >"$CASE_ROOT/recovery/bootstrap_validate.out"
launch_worker optical_producer "$CASE_ROOT/recovery/identities/optical_producer.json" recovery \
  "$PYTHON" "$FINAL_RUNNER" recovery-optical --input "$INPUT_MANIFEST" --stream-root "$CASE_ROOT/lifecycle" --out-dir "$CASE_ROOT/recovery/optical" --bootstrap-receipt "$CASE_ROOT/recovery/restart_reconstructed.json" \
  >"$CASE_ROOT/recovery/producer.out" 2>"$CASE_ROOT/recovery/producer.err"
producer_pid="${worker_pids[-1]}"
for index in 0 1; do
  launch_worker "hydro_consumer_${index}" "$CASE_ROOT/recovery/identities/hydro_consumer_${index}.json" recovery \
    "$PYTHON" "$FINAL_RUNNER" consume --input "$INPUT_MANIFEST" --stream-root "$CASE_ROOT/lifecycle" --producer-complete "$CASE_ROOT/recovery/producer_complete" \
    --actor "hydro_consumer_${index}" --out "$CASE_ROOT/recovery/consumers/consumer_${index}.json" \
    >"$CASE_ROOT/recovery/consumers/consumer_${index}.out" 2>"$CASE_ROOT/recovery/consumers/consumer_${index}.err"
  consumer_pids+=("${worker_pids[-1]}")
done
wait_for_pair "$CASE_ROOT/recovery/producer_complete" recovery "$producer_pid" "${consumer_pids[@]}"
"$PYTHON" "$FINAL_RUNNER" finalize --stream-root "$CASE_ROOT/lifecycle" --out "$CASE_ROOT/recovery/final.json" >"$CASE_ROOT/recovery/finalize.out"
"$PYTHON" "$FINAL_RUNNER" snapshot --stream-root "$CASE_ROOT/lifecycle" --out "$CASE_ROOT/recovery/final_lifecycle_audit.json" >"$CASE_ROOT/recovery/final_audit.out"
"$PYTHON" "$FINAL_RUNNER" compare --reference-lifecycle "$REFERENCE_CASE_ROOT/lifecycle" --reference-optical "$REFERENCE_CASE_ROOT/optical" --candidate-lifecycle "$CASE_ROOT/lifecycle" --candidate-optical "$CASE_ROOT/recovery/optical" --effects "$CASE_ROOT/expected_recovery_effects.json" --out-dir "$CASE_ROOT/comparison" >"$CASE_ROOT/recovery/exact.out"
