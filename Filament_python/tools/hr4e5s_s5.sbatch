#!/usr/bin/env bash
# One S5 lifecycle case: one optical GPU and one hydro GPU.  No scaling modes.
#SBATCH --job-name=hr4e5s-s5
#SBATCH --partition=gpu
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
set -euo pipefail

readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
: "${EXPECTED_GIT_SHA:?}" "${REPO_DIR:?}" "${RUN_ROOT:?}" "${CASE_ID:?}" "${CASE_MODE:?}" "${INPUT_MANIFEST:?}" "${LUT_WORKSPACE:?}"
test "$(git -C "$REPO_DIR" rev-parse HEAD)" = "$EXPECTED_GIT_SHA"
test -z "$(git -C "$REPO_DIR" status --porcelain=v1 --untracked-files=all)"
test -f "$INPUT_MANIFEST" && test -d "$LUT_WORKSPACE" && test "$(stat -c '%a' "$LUT_WORKSPACE")" = 700
case "$CASE_ID" in clean|F01|F02|F03|F04|F05|F06) ;; *) exit 64 ;; esac
case "$CASE_MODE" in clean|fault|recovery) ;; *) exit 64 ;; esac
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export UPPE_USE_GPU=1 PYTHONPATH="$REPO_DIR/Filament_python" PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="$RUN_ROOT/pycache" CUPY_CACHE_DIR="$RUN_ROOT/cupy_cache" XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
test "$("$PYTHON" -c 'from KHz_filament.device import debug_backend; print(debug_backend()["backend"])')" = cupy

readonly CASE_ROOT="$RUN_ROOT/$CASE_ID"
if [[ "$CASE_MODE" == recovery ]]; then
  test -d "$CASE_ROOT"
  test -f "$CASE_ROOT/injected/disk_state_audit.json"
  test -f "$CASE_ROOT/contract_check.json"
  # Contract failure must not create a recovery root, submit optical work, or
  # make this one-shot recovery namespace unusable for a later reviewed fix.
  "$PYTHON" - "$CASE_ROOT/contract_check.json" "$CASE_ID" <<'PY'
import json,sys
path,case_id=sys.argv[1:]
check=json.load(open(path,encoding='utf-8'))
assert check['status']=='PASS' and check['contract_match'] is True
assert str(check['fault_id']).startswith(case_id + '_')
PY
  test ! -e "$CASE_ROOT/recovery"
  # The recovery root is a one-shot transaction boundary.  Do not use mkdir -p:
  # an unexpected pre-existing root must fail closed rather than mix attempts.
  mkdir -m 700 -- "$CASE_ROOT/recovery"
else
  test ! -e "$CASE_ROOT"
  mkdir -m 700 -- "$CASE_ROOT"
fi

pair_run() {
  local lifecycle="$1" optical="$2" consumer="$3" marker="$4" resume="$5" status="$6" bootstrap_receipt="$7"
  mkdir -m 700 -- "$consumer"
  local -a optical_cmd=("$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py" optical --input "$INPUT_MANIFEST" --out-dir "$optical" --stream-root "$lifecycle")
  if [[ "$resume" == 1 ]]; then
    test -n "$bootstrap_receipt" && test -f "$bootstrap_receipt"
    "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py" validate-bootstrap --stream-root "$lifecycle" --receipt "$bootstrap_receipt" --runtime-sha "$EXPECTED_GIT_SHA" --case-id "$CASE_ID"
    optical_cmd+=(--resume --bootstrap-receipt "$bootstrap_receipt")
  else
    test -z "$bootstrap_receipt"
  fi
  srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" "${optical_cmd[@]}" >"$consumer/producer.out" 2>"$consumer/producer.err" &
  local producer_pid=$!
  srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py" consume --input "$INPUT_MANIFEST" --stream-root "$lifecycle" --producer-complete "$marker" --out "$consumer/result.json" >"$consumer/consumer.out" 2>"$consumer/consumer.err" &
  local consumer_pid=$! finished='' first_status=0 producer_status=0 consumer_status=0
  wait -n -p finished "$producer_pid" "$consumer_pid" || first_status=$?
  if [[ "$finished" == "$producer_pid" ]]; then
    producer_status="$first_status"
    if [[ "$producer_status" -ne 0 ]]; then kill "$consumer_pid" 2>/dev/null || true; wait "$consumer_pid" || consumer_status=$?; else touch "$marker"; wait "$consumer_pid" || consumer_status=$?; fi
  else
    consumer_status="$first_status"
    if [[ "$consumer_status" -ne 0 ]]; then kill "$producer_pid" 2>/dev/null || true; wait "$producer_pid" || producer_status=$?; else wait "$producer_pid" || producer_status=$?; if [[ "$producer_status" -eq 0 ]]; then touch "$marker"; fi; fi
  fi
  "$PYTHON" - "$status" "$producer_status" "$consumer_status" <<'PY'
import json,sys
path,producer,consumer=sys.argv[1:]
with open(path+'.tmp','w',encoding='utf-8',newline='\n') as f:
    json.dump({'producer_exit_code':int(producer),'consumer_exit_code':int(consumer)},f,sort_keys=True); f.write('\n'); f.flush()
import os
os.replace(path+'.tmp',path)
PY
  [[ "$producer_status" -eq 0 && "$consumer_status" -eq 0 ]]
}

initialize() {
  "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py" initialize-stream --input "$INPUT_MANIFEST" --root "$CASE_ROOT/injected/lifecycle"
}

if [[ "$CASE_MODE" == clean ]]; then
  unset HR4_S5_FAULT_ID HR4_S5_FAULT_SCREEN HR4_S5_FAULT_ONCE || true
  "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py" initialize-stream --input "$INPUT_MANIFEST" --root "$CASE_ROOT/lifecycle"
  if ! pair_run "$CASE_ROOT/lifecycle" "$CASE_ROOT/optical" "$CASE_ROOT/consumer" "$CASE_ROOT/producer_complete" 0 "$CASE_ROOT/pair_status.json" ""; then exit 1; fi
  "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py" finalize-stream --stream-root "$CASE_ROOT/lifecycle" --out "$CASE_ROOT/final.json"
  "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s5.py" inspect --lifecycle-root "$CASE_ROOT/lifecycle" --out "$CASE_ROOT/final_lifecycle_audit.json"
  exit 0
fi

if [[ "$CASE_MODE" == fault ]]; then
  : "${HR4_S5_FAULT_ID:?}" "${HR4_S5_FAULT_SCREEN:?}"
  initialize
  if [[ "$CASE_ID" == F06 ]]; then
    if ! pair_run "$CASE_ROOT/injected/lifecycle" "$CASE_ROOT/injected/optical" "$CASE_ROOT/injected/consumer" "$CASE_ROOT/injected/producer_complete" 0 "$CASE_ROOT/injected/pair_status.json" ""; then exit 1; fi
    if "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py" finalize-stream --stream-root "$CASE_ROOT/injected/lifecycle" --out "$CASE_ROOT/injected/final.json"; then exit 1; fi
  else
    if pair_run "$CASE_ROOT/injected/lifecycle" "$CASE_ROOT/injected/optical" "$CASE_ROOT/injected/consumer" "$CASE_ROOT/injected/producer_complete" 0 "$CASE_ROOT/injected/pair_status.json" ""; then exit 1; fi
  fi
  "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s5.py" inspect --lifecycle-root "$CASE_ROOT/injected/lifecycle" --out "$CASE_ROOT/injected/disk_state_audit.json"
  "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s5.py" contract --lifecycle-root "$CASE_ROOT/injected/lifecycle" --fault-id "$HR4_S5_FAULT_ID" --target-screen "$HR4_S5_FAULT_SCREEN" --out "$CASE_ROOT/contract_check.json"
  exit 0
fi

# Recovery is intentionally a separate Slurm job.  The fault hook is disabled;
# its persisted contract was checked before the recovery root was created.
unset HR4_S5_FAULT_ID HR4_S5_FAULT_SCREEN HR4_S5_FAULT_ONCE || true
if [[ "$CASE_ID" == F06 ]]; then
  "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py" finalize-stream --stream-root "$CASE_ROOT/injected/lifecycle" --out "$CASE_ROOT/recovery/final.json"
  candidate_optical="$CASE_ROOT/injected/optical"
else
  readonly BOOTSTRAP_RECEIPT="$CASE_ROOT/recovery/restart_reconstructed.json"
  "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py" bootstrap-recovery --stream-root "$CASE_ROOT/injected/lifecycle" --out "$BOOTSTRAP_RECEIPT" --runtime-sha "$EXPECTED_GIT_SHA" --case-id "$CASE_ID"
  "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py" validate-bootstrap --stream-root "$CASE_ROOT/injected/lifecycle" --receipt "$BOOTSTRAP_RECEIPT" --runtime-sha "$EXPECTED_GIT_SHA" --case-id "$CASE_ID"
  if ! pair_run "$CASE_ROOT/injected/lifecycle" "$CASE_ROOT/recovery/optical" "$CASE_ROOT/recovery/consumer" "$CASE_ROOT/recovery/producer_complete" 1 "$CASE_ROOT/recovery/pair_status.json" "$BOOTSTRAP_RECEIPT"; then exit 1; fi
  "$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py" finalize-stream --stream-root "$CASE_ROOT/injected/lifecycle" --out "$CASE_ROOT/recovery/final.json"
  candidate_optical="$CASE_ROOT/recovery/optical"
fi
"$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s5.py" inspect --lifecycle-root "$CASE_ROOT/injected/lifecycle" --out "$CASE_ROOT/recovery/final_lifecycle_audit.json"
: "${REFERENCE_CASE_ROOT:?}"
"$PYTHON" "$REPO_DIR/Filament_python/tools/run_hr4e5s_s5.py" compare --reference-lifecycle-root "$REFERENCE_CASE_ROOT/lifecycle" --reference-optical-dir "$REFERENCE_CASE_ROOT/optical" --candidate-lifecycle-root "$CASE_ROOT/injected/lifecycle" --candidate-optical-dir "$candidate_optical" --out-dir "$CASE_ROOT/comparison" --fault-contract "$CASE_ROOT/contract_check.json"
