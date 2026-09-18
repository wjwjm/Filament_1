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
mkdir -m 700 -- "$PHASE0_ROOT" "$PHASE0_ROOT/identities" "$PHASE0_ROOT/heartbeats"

phase0_json() {
  local destination="$1"; shift
  "$PYTHON" - "$destination" "$@" <<'PY'
import json, os, sys, tempfile
out,*items=sys.argv[1:]; value={'schema':'khz_filament.hr4e5s.s5_final.phase0.v1'}
for item in items:
    key,raw=item.split('=',1); value[key]=(raw=='true' if raw in {'true','false'} else int(raw) if raw.lstrip('-').isdigit() else raw)
fd,tmp=tempfile.mkstemp(prefix=os.path.basename(out)+'.',dir=os.path.dirname(out))
with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as h: json.dump(value,h,indent=2,sort_keys=True); h.write('\n'); h.flush(); os.fsync(h.fileno())
os.replace(tmp,out)
PY
}
phase0_fail() {
  local reason="$1"
  phase0_json "$PHASE0_ROOT/phase0_observability_result.json" "status=FAIL" "reason=$reason" "controller_hostname=$(hostname)" "science_started=false"
  phase0_json "$RUN_ROOT/final_decision.json" "status=READY_FOR_S5_FINAL_DEFECT_REVIEW" "reason=PHASE0_${reason}" "science_started=false"
  printf '# S5-FINAL single-allocation report\n\n- Final status: `READY_FOR_S5_FINAL_DEFECT_REVIEW`\n- Phase 0: `FAIL` (%s)\n- Scientific workers: not started\n' "$reason" >"$RUN_ROOT/S5_FINAL_SINGLE_ALLOCATION_REPORT.md"
  exit 0
}
wait_for_file() { local path="$1" limit="$2" elapsed=0; while [[ ! -f "$path" && "$elapsed" -lt "$limit" ]]; do sleep 1; elapsed=$((elapsed+1)); done; test -f "$path"; }
read_json_field() { "$PYTHON" - "$1" "$2" <<'PY'
import json,sys
print(json.load(open(sys.argv[1],encoding='utf-8'))[sys.argv[2]])
PY
}
launch_phase0_probe() {
  local actor="$1" identity="$2" heartbeat="$3"
  # CVD is explicitly removed; the child only writes a receipt and a heartbeat.
  srun --exclusive --ntasks=1 --cpus-per-task=1 --gpus-per-task=0 env -u CUDA_VISIBLE_DEVICES -u UPPE_USE_GPU bash -c '
    set -euo pipefail
    py="$1"; identity="$2"; actor="$3"; heartbeat="$4"
    env -u CUDA_VISIBLE_DEVICES -u UPPE_USE_GPU "$py" - "$identity" "$actor" "$heartbeat" <<'"'"'PY'"'"'
import json,os,socket,sys,tempfile,time
out,actor,heartbeat=sys.argv[1:]
value={'schema':'khz_filament.hr4e5s.s5_final.phase0.v1','kind':'identity','actor':actor,'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'hostname':socket.gethostname(),'start_time':time.time(),'heartbeat_path':heartbeat,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES',''),'cupy_imported':False,'cuda_context_initialized':False}
fd,tmp=tempfile.mkstemp(prefix=os.path.basename(out)+'.',dir=os.path.dirname(out))
with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as h: json.dump(value,h,indent=2,sort_keys=True); h.write('\n'); h.flush(); os.fsync(h.fileno())
os.replace(tmp,out)
PY
    while true; do date +%s%N >"$heartbeat"; sleep 1; done
  ' bash "$PYTHON" "$identity" "$actor" "$heartbeat" &
  launched_pid="$!"
}
capture_listpids() {
  local name="$1" query="$2" stdout="$PHASE0_ROOT/${name}.stdout" stderr="$PHASE0_ROOT/${name}.stderr" rc=0
  scontrol listpids "$query" >"$stdout" 2>"$stderr" || rc=$?
  "$PYTHON" - "$PHASE0_ROOT/${name}.json" "$query" "$rc" "$stdout" "$stderr" <<'PY'
import json,os,socket,sys,tempfile,time
out,query,rc,stdout,stderr=sys.argv[1:]
value={'schema':'khz_filament.hr4e5s.s5_final.phase0.v1','kind':'listpids_evidence','command':['scontrol','listpids',query],'hostname':socket.gethostname(),'return_code':int(rc),'stdout':open(stdout,encoding='utf-8').read(),'stderr':open(stderr,encoding='utf-8').read(),'timestamp':time.time()}
fd,tmp=tempfile.mkstemp(prefix=os.path.basename(out)+'.',dir=os.path.dirname(out))
with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as h: json.dump(value,h,indent=2,sort_keys=True); h.write('\n'); h.flush(); os.fsync(h.fileno())
os.replace(tmp,out)
PY
  return "$rc"
}

# Phase 0.  No CuPy import, lifecycle directory, scientific worker, or GPU context precedes this block.
phase0_json "$PHASE0_ROOT/phase0_manifest.json" "status=RUNNING" "allocation_id=$SLURM_JOB_ID" "controller_hostname=$(hostname)" "allocated_gpus=3" "science_started=false"
launch_phase0_probe probe_survivor "$PHASE0_ROOT/identities/probe_survivor.json" "$PHASE0_ROOT/heartbeats/probe_survivor"; phase0_survivor_srun="$launched_pid"
launch_phase0_probe probe_target "$PHASE0_ROOT/identities/probe_target.json" "$PHASE0_ROOT/heartbeats/probe_target"; phase0_target_srun="$launched_pid"
wait_for_file "$PHASE0_ROOT/identities/probe_survivor.json" 30 && wait_for_file "$PHASE0_ROOT/identities/probe_target.json" 30 || phase0_fail IDENTITY_RECEIPTS_INCOMPLETE
readonly PHASE0_TARGET_JSON="$PHASE0_ROOT/identities/probe_target.json" PHASE0_SURVIVOR_JSON="$PHASE0_ROOT/identities/probe_survivor.json"
readonly PHASE0_TARGET_STEP="$(read_json_field "$PHASE0_TARGET_JSON" job_id).$(read_json_field "$PHASE0_TARGET_JSON" step_id)" PHASE0_SURVIVOR_STEP="$(read_json_field "$PHASE0_SURVIVOR_JSON" job_id).$(read_json_field "$PHASE0_SURVIVOR_JSON" step_id)"
readonly PHASE0_TARGET_PID="$(read_json_field "$PHASE0_TARGET_JSON" pid)" PHASE0_SURVIVOR_PID="$(read_json_field "$PHASE0_SURVIVOR_JSON" pid)"
test "$(read_json_field "$PHASE0_TARGET_JSON" hostname)" = "$(hostname)" && test "$(read_json_field "$PHASE0_SURVIVOR_JSON" hostname)" = "$(hostname)" || phase0_fail CONTROLLER_HOSTNAME_MISMATCH
test "$(read_json_field "$PHASE0_TARGET_JSON" cuda_visible_devices)" = "" && test "$(read_json_field "$PHASE0_SURVIVOR_JSON" cuda_visible_devices)" = "" || phase0_fail PROBE_GPU_ENV_PRESENT
capture_listpids listpids_before "$SLURM_JOB_ID" || phase0_fail LISTPIDS_JOB_RETURN_NONZERO
capture_listpids target_listpids_before "$PHASE0_TARGET_STEP" || phase0_fail TARGET_LISTPIDS_RETURN_NONZERO
capture_listpids survivor_listpids_before "$PHASE0_SURVIVOR_STEP" || phase0_fail SURVIVOR_LISTPIDS_RETURN_NONZERO
test -s "$PHASE0_ROOT/heartbeats/probe_target" && test -s "$PHASE0_ROOT/heartbeats/probe_survivor" || phase0_fail HEARTBEATS_NOT_LIVE
test ! -e "$PHASE0_ROOT/signal_intent.json" && test ! -e "$PHASE0_ROOT/signal_receipt.json" || phase0_fail DUPLICATE_SIGNAL_INTENT
test "$PHASE0_TARGET_STEP" != "$PHASE0_SURVIVOR_STEP" && test "$PHASE0_TARGET_PID" != "$PHASE0_SURVIVOR_PID" || phase0_fail IDENTITY_COLLISION
grep -F -- "$PHASE0_TARGET_PID" "$PHASE0_ROOT/target_listpids_before.stdout" >/dev/null || phase0_fail TARGET_PID_NOT_PROVEN
grep -F -- "$PHASE0_SURVIVOR_PID" "$PHASE0_ROOT/survivor_listpids_before.stdout" >/dev/null || phase0_fail SURVIVOR_PID_NOT_PROVEN
phase0_json "$PHASE0_ROOT/signal_intent.json" "status=INTENT" "target_step=$PHASE0_TARGET_STEP" "target_pid=$PHASE0_TARGET_PID" "signal=TERM" "signal_scope=single_step"
phase0_signal_rc=0; scancel --signal=TERM "$PHASE0_TARGET_STEP" || phase0_signal_rc=$?
phase0_json "$PHASE0_ROOT/signal_receipt.json" "status=SENT" "target_step=$PHASE0_TARGET_STEP" "target_pid=$PHASE0_TARGET_PID" "signal=TERM" "return_code=$phase0_signal_rc" "signal_scope=single_step"
test "$phase0_signal_rc" -eq 0 || phase0_fail TARGET_SIGNAL_FAILED
phase0_target_wait=0; wait "$phase0_target_srun" || phase0_target_wait=$?
test "$phase0_target_wait" -ne 0 || phase0_fail TARGET_DID_NOT_EXIT
kill -0 "$PHASE0_TARGET_PID" 2>/dev/null && phase0_fail TARGET_STILL_LIVE
kill -0 "$PHASE0_SURVIVOR_PID" 2>/dev/null || phase0_fail SIGNAL_SCOPE_TOO_BROAD
capture_listpids target_listpids_after "$PHASE0_TARGET_STEP" || true
grep -F -- "$PHASE0_TARGET_PID" "$PHASE0_ROOT/target_listpids_after.stdout" >/dev/null && phase0_fail TARGET_STILL_VISIBLE_AFTER_SIGNAL
phase0_json "$PHASE0_ROOT/target_quiescence_receipt.json" "status=PASS" "target_step=$PHASE0_TARGET_STEP" "target_pid=$PHASE0_TARGET_PID" "target_wait_nonzero=true"
phase0_json "$PHASE0_ROOT/survivor_status.json" "status=PASS" "survivor_step=$PHASE0_SURVIVOR_STEP" "survivor_pid=$PHASE0_SURVIVOR_PID"
kill "$phase0_survivor_srun" 2>/dev/null || true; wait "$phase0_survivor_srun" 2>/dev/null || true
phase0_json "$PHASE0_ROOT/phase0_observability_result.json" "status=PASS" "signal_count=1" "target_step=$PHASE0_TARGET_STEP" "survivor_step=$PHASE0_SURVIVOR_STEP" "controller_hostname=$(hostname)" "science_started=false"

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
