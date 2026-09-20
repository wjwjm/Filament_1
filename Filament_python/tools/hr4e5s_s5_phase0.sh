#!/usr/bin/env bash
# CPU-only, fail-closed S5-FINAL observability gate.  Called by the submitted batch.
set -euo pipefail

: "${PYTHON:?}" "${RUN_ROOT:?}" "${SLURM_JOB_ID:?}"
readonly ROOT="${S5_PHASE0_ROOT:-$RUN_ROOT/phase0}"
readonly TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly ACTOR="$TOOLS_DIR/hr4e5s_s5_probe_actor.py"
readonly LOCAL_FLAG="${S5_LOCAL_TEST:-}"
test -f "$ACTOR"

record() { "$PYTHON" "$ACTOR" write-record --out "$1" "${@:2}"; }
field() { "$PYTHON" "$ACTOR" field --path "$1" --key "$2"; }
capture() { "$PYTHON" "$ACTOR" capture-listpids --out "$ROOT/$1.json" --query "$2"; }
wait_for_file() { local path="$1" limit="$2" elapsed=0; while [[ ! -f "$path" && "$elapsed" -lt "$limit" ]]; do sleep 1; elapsed=$((elapsed+1)); done; test -f "$path"; }
cleanup() { for process in "${target_srun:-}" "${survivor_srun:-}"; do [[ -n "$process" ]] && kill "$process" 2>/dev/null || true; done; for process in "${target_srun:-}" "${survivor_srun:-}"; do [[ -n "$process" ]] && wait "$process" 2>/dev/null || true; done; }
if [[ -e "$ROOT" ]]; then
  record "$RUN_ROOT/phase0_preflight_defect.json" "status=FAIL" "reason=PHASE0_ROOT_ALREADY_EXISTS" "science_started=false"
  record "$RUN_ROOT/final_decision.json" "status=READY_FOR_S5_FINAL_DEFECT_REVIEW" "reason=PHASE0_ROOT_ALREADY_EXISTS" "science_started=false"
  exit 70
fi
mkdir -m 700 -- "$ROOT" "$ROOT/identities" "$ROOT/heartbeats"
fail() {
  local reason="$1"
  cleanup
  record "$ROOT/phase0_observability_result.json" "status=FAIL" "reason=$reason" "controller_hostname=$(hostname)" "science_started=false"
  record "$RUN_ROOT/final_decision.json" "status=READY_FOR_S5_FINAL_DEFECT_REVIEW" "reason=PHASE0_${reason}" "science_started=false"
  printf '# S5-FINAL single-allocation report\n\n- Final status: `READY_FOR_S5_FINAL_DEFECT_REVIEW`\n- Phase 0: `FAIL` (%s)\n- Scientific workers: not started\n' "$reason" >"$RUN_ROOT/S5_FINAL_SINGLE_ALLOCATION_REPORT.md"
  exit 70
}
launch_probe() {
  local actor="$1" identity="$2" heartbeat="$3" local_args=()
  [[ -n "$LOCAL_FLAG" ]] && local_args+=(--local-test)
  # This argv is intentionally direct: no nested shell and no embedded Python source.
  srun --exclusive --ntasks=1 --cpus-per-task=1 --gpus-per-task=0 \
    env -u CUDA_VISIBLE_DEVICES -u UPPE_USE_GPU "$PYTHON" "$ACTOR" actor \
    --actor "$actor" --identity "$identity" --heartbeat "$heartbeat" "${local_args[@]}" &
  launched_pid="$!"
}

record "$ROOT/phase0_manifest.json" "status=RUNNING" "allocation_id=$SLURM_JOB_ID" "controller_hostname=$(hostname)" "allocated_gpus=3" "science_started=false"
launch_probe probe_survivor "$ROOT/identities/probe_survivor.json" "$ROOT/heartbeats/probe_survivor"; survivor_srun="$launched_pid"
launch_probe probe_target "$ROOT/identities/probe_target.json" "$ROOT/heartbeats/probe_target"; target_srun="$launched_pid"
wait_for_file "$ROOT/identities/probe_survivor.json" 30 && wait_for_file "$ROOT/identities/probe_target.json" 30 || fail IDENTITY_RECEIPTS_INCOMPLETE
readonly TARGET_JSON="$ROOT/identities/probe_target.json" SURVIVOR_JSON="$ROOT/identities/probe_survivor.json"
readonly TARGET_STEP="$(field "$TARGET_JSON" job_id).$(field "$TARGET_JSON" step_id)" SURVIVOR_STEP="$(field "$SURVIVOR_JSON" job_id).$(field "$SURVIVOR_JSON" step_id)"
readonly TARGET_PID="$(field "$TARGET_JSON" pid)" SURVIVOR_PID="$(field "$SURVIVOR_JSON" pid)"
test "$(field "$TARGET_JSON" hostname)" = "$(hostname)" && test "$(field "$SURVIVOR_JSON" hostname)" = "$(hostname)" || fail CONTROLLER_HOSTNAME_MISMATCH
test "$(field "$TARGET_JSON" cuda_visible_devices)" = "" && test "$(field "$SURVIVOR_JSON" cuda_visible_devices)" = "" || fail PROBE_GPU_ENV_PRESENT
capture listpids_before "$SLURM_JOB_ID" || fail LISTPIDS_JOB_RETURN_NONZERO
capture target_listpids_before "$TARGET_STEP" || fail TARGET_LISTPIDS_RETURN_NONZERO
capture survivor_listpids_before "$SURVIVOR_STEP" || fail SURVIVOR_LISTPIDS_RETURN_NONZERO
test -s "$ROOT/heartbeats/probe_target" && test -s "$ROOT/heartbeats/probe_survivor" || fail HEARTBEATS_NOT_LIVE
test ! -e "$ROOT/signal_intent.json" && test ! -e "$ROOT/signal_receipt.json" || fail DUPLICATE_SIGNAL_INTENT
test "$TARGET_STEP" != "$SURVIVOR_STEP" && test "$TARGET_PID" != "$SURVIVOR_PID" || fail IDENTITY_COLLISION
grep -F -- "$TARGET_PID" "$ROOT/target_listpids_before.json" >/dev/null || fail TARGET_PID_NOT_PROVEN
grep -F -- "$SURVIVOR_PID" "$ROOT/survivor_listpids_before.json" >/dev/null || fail SURVIVOR_PID_NOT_PROVEN
record "$ROOT/signal_intent.json" "status=INTENT" "target_step=$TARGET_STEP" "target_pid=$TARGET_PID" "signal=TERM" "signal_scope=single_step"
signal_rc=0; scancel --signal=TERM "$TARGET_STEP" || signal_rc=$?
record "$ROOT/signal_receipt.json" "status=SENT" "target_step=$TARGET_STEP" "target_pid=$TARGET_PID" "signal=TERM" "return_code=$signal_rc" "signal_scope=single_step"
test "$signal_rc" -eq 0 || fail TARGET_SIGNAL_FAILED
target_wait=0; wait "$target_srun" || target_wait=$?
test "$target_wait" -ne 0 || fail TARGET_DID_NOT_EXIT
kill -0 "$TARGET_PID" 2>/dev/null && fail TARGET_STILL_LIVE
kill -0 "$SURVIVOR_PID" 2>/dev/null || fail SIGNAL_SCOPE_TOO_BROAD
capture target_listpids_after "$TARGET_STEP" || true
grep -F -- "$TARGET_PID" "$ROOT/target_listpids_after.json" >/dev/null && fail TARGET_STILL_VISIBLE_AFTER_SIGNAL
record "$ROOT/target_quiescence_receipt.json" "status=PASS" "target_step=$TARGET_STEP" "target_pid=$TARGET_PID" "target_wait_nonzero=true"
record "$ROOT/survivor_status.json" "status=PASS" "survivor_step=$SURVIVOR_STEP" "survivor_pid=$SURVIVOR_PID"
cleanup
record "$ROOT/phase0_observability_result.json" "status=PASS" "signal_count=1" "target_step=$TARGET_STEP" "survivor_step=$SURVIVOR_STEP" "controller_hostname=$(hostname)" "science_started=false"
