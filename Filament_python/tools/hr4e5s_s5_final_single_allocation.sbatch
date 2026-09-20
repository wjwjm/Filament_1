#!/usr/bin/env bash
# One 3-GPU allocation: CPU-only Phase 0 first, then two fresh scientific epochs.
#SBATCH --job-name=hr4e5s-s5-final
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
set -euo pipefail

readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
: "${EXPECTED_GIT_SHA:?}" "${REPO_DIR:?}" "${RUN_ROOT:?}" "${LUT_WORKSPACE:?}" "${INPUT_MANIFEST:?}" "${CASE_MODE:?}" "${REFERENCE_CASE_ROOT:?}"
test "$CASE_MODE" = single_allocation
test "$(git -C "$REPO_DIR" rev-parse HEAD)" = "$EXPECTED_GIT_SHA"
test -z "$(git -C "$REPO_DIR" status --porcelain=v1 --untracked-files=all)"
test -f "$INPUT_MANIFEST" && test -d "$LUT_WORKSPACE" && test "$(stat -c '%a' "$LUT_WORKSPACE")" = 700
test -d "$REFERENCE_CASE_ROOT/lifecycle" && test -d "$REFERENCE_CASE_ROOT/optical"
unset HR4_S5_FAULT_ID HR4_S5_FAULT_SCREEN HR4_S5_FAULT_ONCE || true
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export PYTHON REPO_DIR RUN_ROOT EXPECTED_GIT_SHA PYTHONPATH="$REPO_DIR/Filament_python" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
umask 077

readonly PHASE0_ROOT="$RUN_ROOT/phase0"
readonly FINAL_RUNNER="$REPO_DIR/Filament_python/tools/run_hr4e5s_s5_final.py"
readonly S3_RUNNER="$REPO_DIR/Filament_python/tools/run_hr4e5s_s3.py"
readonly MONITOR="$REPO_DIR/Filament_python/tools/monitor_hr4e5s_s5_final.py"
readonly PHASE0_LAUNCHER="$REPO_DIR/Filament_python/tools/hr4e5s_s5_phase0.sh"
test -f "$PHASE0_LAUNCHER"

# Phase 0 is a separately versioned, stdlib-only program.  The batch invokes it
# directly so no Python source traverses an additional ``bash -c`` parser.
phase0_status=0
bash "$PHASE0_LAUNCHER" || phase0_status=$?
if [[ "$phase0_status" -ne 0 ]]; then
  exit "$phase0_status"
fi

# Only after Phase 0 PASS may the CUDA backend and scientific lifecycle be initialized.
export UPPE_USE_GPU=1 PYTHONPYCACHEPREFIX="$RUN_ROOT/pycache" CUPY_CACHE_DIR="$RUN_ROOT/cupy_cache" XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
cd "$LUT_WORKSPACE"; test "$("$PYTHON" -c 'from KHz_filament.device import debug_backend; print(debug_backend()["backend"])')" = cupy
readonly CASE_ROOT="$RUN_ROOT/scenario"; export FINAL_RUNNER S3_RUNNER
mkdir -m 700 -- "$CASE_ROOT" "$CASE_ROOT/initial" "$CASE_ROOT/initial/identities" "$CASE_ROOT/initial/arming" "$CASE_ROOT/initial/consumers"
atomic_json() { local destination="$1" value="$2" temporary="${1}.tmp.$$"; test ! -e "$destination"; printf '%s\n' "$value" >"$temporary"; mv -- "$temporary" "$destination"; }
wait_for_file_or_defect() { local wanted="$1" limit="$2" elapsed=0; while [[ ! -f "$wanted" ]]; do [[ -f "$RUN_ROOT/controller_defect.json" || "$elapsed" -ge "$limit" ]] && return 1; sleep 1; elapsed=$((elapsed+1)); done; }
launch_worker() {
  local actor="$1" epoch="$2" identity="$3"; shift 3
  srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c '
    set -euo pipefail; actor="$1"; epoch="$2"; identity="$3"; shift 3
    : "${SLURM_JOB_ID:?}" "${SLURM_STEP_ID:?}" "${SLURMD_NODENAME:?}"
    env HR4E5S_WORKER_ID="$actor" "$PYTHON" "$FINAL_RUNNER" write-identity --out "$identity" --actor "$actor" --pid "$$" --step-id "$SLURM_STEP_ID" --job-id "$SLURM_JOB_ID" --node "$SLURMD_NODENAME" --phase "$epoch" --execution-epoch "$epoch"
    exec env HR4E5S_WORKER_ID="$actor" "$@"
  ' bash "$actor" "$epoch" "$identity" "$@" & launched_pid="$!"
}
stop_children() { local child; for child in "$@"; do kill "$child" 2>/dev/null || true; done; for child in "$@"; do wait "$child" 2>/dev/null || true; done; }
readonly MODE=SINGLE_ALLOCATION_TWO_EXECUTION_EPOCHS MONITOR_MANIFEST="$RUN_ROOT/s5_final_single_allocation_manifest.json"
"$PYTHON" - "$MONITOR_MANIFEST" "$REPO_DIR" "$RUN_ROOT" "$EXPECTED_GIT_SHA" "$REFERENCE_CASE_ROOT" "$SLURM_JOB_ID" <<'PY'
import json,sys
out,repo,root,sha,reference,job=sys.argv[1:]
json.dump({'schema':'khz_filament.hr4e5s.s5_final.single_allocation_manifest.v1','execution_mode':'SINGLE_ALLOCATION_TWO_EXECUTION_EPOCHS','repo':repo,'run_root':root,'expected_sha':sha,'reference_case_root':reference,'allocation_id':job,'target_actor':'hydro_consumer_1'},open(out,'x',encoding='utf-8'),indent=2,sort_keys=True)
PY
"$PYTHON" "$MONITOR" --manifest "$MONITOR_MANIFEST" --single-allocation --resume --poll-seconds 5 >"$RUN_ROOT/controller.out" 2>"$RUN_ROOT/controller.err" & controller_pid="$!"
wait_for_file_or_defect "$CASE_ROOT/controller_ready.json" 30 || { wait "$controller_pid" || true; exit 40; }
"$PYTHON" "$S3_RUNNER" initialize-stream --input "$INPUT_MANIFEST" --root "$CASE_ROOT/lifecycle" >"$CASE_ROOT/initialize.out"
launch_worker optical_producer initial "$CASE_ROOT/initial/identities/optical_producer.json" "$PYTHON" "$S3_RUNNER" optical --input "$INPUT_MANIFEST" --out-dir "$CASE_ROOT/initial/optical" --stream-root "$CASE_ROOT/lifecycle"; initial_optical="$launched_pid"
launch_worker hydro_consumer_0 initial "$CASE_ROOT/initial/identities/hydro_consumer_0.json" "$PYTHON" "$FINAL_RUNNER" consume --input "$INPUT_MANIFEST" --stream-root "$CASE_ROOT/lifecycle" --producer-complete "$CASE_ROOT/initial/producer_complete" --actor hydro_consumer_0 --out "$CASE_ROOT/initial/consumers/consumer_0.json" --arming-dir "$CASE_ROOT/initial/arming" --execution-epoch initial; initial_hydro_0="$launched_pid"
launch_worker hydro_consumer_1 initial "$CASE_ROOT/initial/identities/hydro_consumer_1.json" "$PYTHON" "$FINAL_RUNNER" consume --input "$INPUT_MANIFEST" --stream-root "$CASE_ROOT/lifecycle" --producer-complete "$CASE_ROOT/initial/producer_complete" --actor hydro_consumer_1 --out "$CASE_ROOT/initial/consumers/consumer_1.json" --arming-dir "$CASE_ROOT/initial/arming" --execution-epoch initial; initial_hydro_1="$launched_pid"
initial_status=0; wait "$initial_hydro_1" || initial_status=$?
if [[ "$initial_status" -eq 0 ]] || ! wait_for_file_or_defect "$CASE_ROOT/initial/worker_loss_receipt.json" 30; then stop_children "$initial_optical" "$initial_hydro_0"; atomic_json "$CASE_ROOT/initial/initial_workers_stopped.json" '{"schema":"khz_filament.hr4e5s.s5_final.v1","status":"FAIL","reason":"unexpected_initial_termination"}'; wait "$controller_pid" || true; exit 41; fi
stop_children "$initial_optical" "$initial_hydro_0"; atomic_json "$CASE_ROOT/initial/initial_failure_propagation.json" "{\"schema\":\"khz_filament.hr4e5s.s5_final.v1\",\"status\":\"PASS\",\"target_actor\":\"hydro_consumer_1\",\"target_exit_code\":$initial_status}"; atomic_json "$CASE_ROOT/initial/initial_workers_stopped.json" '{"schema":"khz_filament.hr4e5s.s5_final.v1","status":"PASS","execution_epoch":"initial"}'
wait_for_file_or_defect "$CASE_ROOT/recovery_ready.json" 300 || { wait "$controller_pid" || true; exit 42; }
mkdir -m 700 -- "$CASE_ROOT/recovery" "$CASE_ROOT/recovery/identities" "$CASE_ROOT/recovery/consumers"
"$PYTHON" "$FINAL_RUNNER" bootstrap --stream-root "$CASE_ROOT/lifecycle" --effects "$CASE_ROOT/expected_recovery_effects.json" --out "$CASE_ROOT/recovery/restart_reconstructed.json" --runtime-sha "$EXPECTED_GIT_SHA" >"$CASE_ROOT/recovery/bootstrap.out"
"$PYTHON" "$FINAL_RUNNER" validate-bootstrap --stream-root "$CASE_ROOT/lifecycle" --receipt "$CASE_ROOT/recovery/restart_reconstructed.json" --runtime-sha "$EXPECTED_GIT_SHA" >"$CASE_ROOT/recovery/bootstrap_validate.out"
launch_worker optical_producer recovery "$CASE_ROOT/recovery/identities/optical_producer.json" "$PYTHON" "$FINAL_RUNNER" recovery-optical --input "$INPUT_MANIFEST" --stream-root "$CASE_ROOT/lifecycle" --out-dir "$CASE_ROOT/recovery/optical" --bootstrap-receipt "$CASE_ROOT/recovery/restart_reconstructed.json"; recovery_optical="$launched_pid"
launch_worker hydro_consumer_0 recovery "$CASE_ROOT/recovery/identities/hydro_consumer_0.json" "$PYTHON" "$FINAL_RUNNER" consume --input "$INPUT_MANIFEST" --stream-root "$CASE_ROOT/lifecycle" --producer-complete "$CASE_ROOT/recovery/producer_complete" --actor hydro_consumer_0 --out "$CASE_ROOT/recovery/consumers/consumer_0.json" --execution-epoch recovery; recovery_hydro_0="$launched_pid"
launch_worker hydro_consumer_1 recovery "$CASE_ROOT/recovery/identities/hydro_consumer_1.json" "$PYTHON" "$FINAL_RUNNER" consume --input "$INPUT_MANIFEST" --stream-root "$CASE_ROOT/lifecycle" --producer-complete "$CASE_ROOT/recovery/producer_complete" --actor hydro_consumer_1 --out "$CASE_ROOT/recovery/consumers/consumer_1.json" --execution-epoch recovery; recovery_hydro_1="$launched_pid"
wait "$recovery_optical" && touch "$CASE_ROOT/recovery/producer_complete"; wait "$recovery_hydro_0"; wait "$recovery_hydro_1"
"$PYTHON" "$FINAL_RUNNER" finalize --stream-root "$CASE_ROOT/lifecycle" --out "$CASE_ROOT/recovery/final.json" >"$CASE_ROOT/recovery/finalize.out"
"$PYTHON" "$FINAL_RUNNER" snapshot --stream-root "$CASE_ROOT/lifecycle" --out "$CASE_ROOT/recovery/final_lifecycle_audit.json" >"$CASE_ROOT/recovery/final_audit.out"
"$PYTHON" "$FINAL_RUNNER" compare --reference-lifecycle "$REFERENCE_CASE_ROOT/lifecycle" --reference-optical "$REFERENCE_CASE_ROOT/optical" --candidate-lifecycle "$CASE_ROOT/lifecycle" --candidate-optical "$CASE_ROOT/recovery/optical" --effects "$CASE_ROOT/expected_recovery_effects.json" --out-dir "$CASE_ROOT/comparison" >"$CASE_ROOT/recovery/exact.out"
wait "$controller_pid"
"$PYTHON" - "$CASE_ROOT/comparison/exact_comparison.json" "$RUN_ROOT/final_decision.json" "$RUN_ROOT/S5_FINAL_SINGLE_ALLOCATION_REPORT.md" <<'PY'
import json,sys
exact,out,report=sys.argv[1:]; x=json.load(open(exact,encoding='utf-8'))
ok=x.get('status')=='PASS' and x.get('expected_field_comparisons')==432 and x.get('completed_field_comparisons')==432 and x.get('mismatch_count')==0 and x.get('recovery_provenance',{}).get('status')=='PASS'
json.dump({'schema':'khz_filament.hr4e5s.s5_final.v1','status':'READY_FOR_S5_FINAL_MANUAL_REVIEW' if ok else 'READY_FOR_S5_FINAL_DEFECT_REVIEW','phase0':'PASS','candidate':'PASS' if ok else 'FAIL','single_allocation_scope':'fresh-process recovery within one allocation; excludes allocation/node/requeue loss'},open(out,'x',encoding='utf-8'),indent=2,sort_keys=True)
open(report,'x',encoding='utf-8').write('# S5-FINAL single-allocation report\n\n- Phase 0: PASS\n- Final state: '+('READY_FOR_S5_FINAL_MANUAL_REVIEW' if ok else 'READY_FOR_S5_FINAL_DEFECT_REVIEW')+'\n- Scope: fresh-process recovery within one allocation only.\n')
if not ok: raise SystemExit(43)
PY
