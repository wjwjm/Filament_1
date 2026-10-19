#!/usr/bin/env bash
# CPU-only site gate for S5-FINAL step/PID observability.  No science or GPU.
#SBATCH --job-name=hr4e5s-s5-observe
#SBATCH --partition=gpu
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
set -euo pipefail

: "${PROBE_ROOT:?}"
readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
readonly ROOT="$PROBE_ROOT"
umask 077
mkdir -m 700 -- "$ROOT" "$ROOT/heartbeats" "$ROOT/identities"
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python

write_json() {
  local destination="$1"; shift
  "$PYTHON" - "$destination" "$@" <<'PY'
import json,os,sys,tempfile
out,kind,*values=sys.argv[1:]
record={'schema':'khz_filament.hr4e5s.s5_final.site_observability.v1','kind':kind}
for item in values:
    key,value=item.split('=',1); record[key]=value
fd,tmp=tempfile.mkstemp(prefix=os.path.basename(out)+'.',dir=os.path.dirname(out)); os.close(fd)
with open(tmp,'w',encoding='utf-8',newline='\n') as h: json.dump(record,h,indent=2,sort_keys=True); h.write('\n'); h.flush(); os.fsync(h.fileno())
os.replace(tmp,out)
PY
}
capture_listpids() {
  local name="$1" step="$2" stdout="$ROOT/${name}.stdout" stderr="$ROOT/${name}.stderr" rc=0
  scontrol listpids "$step" >"$stdout" 2>"$stderr" || rc=$?
  "$PYTHON" - "$ROOT/${name}.json" "$step" "$rc" "$stdout" "$stderr" <<'PY'
import json,os,socket,sys,time,tempfile
out,step,rc,stdout,stderr=sys.argv[1:]
record={'schema':'khz_filament.hr4e5s.s5_final.site_observability.v1','kind':'listpids_evidence','command':['scontrol','listpids',step],'controller_hostname':socket.gethostname(),'return_code':int(rc),'stdout':open(stdout,encoding='utf-8').read(),'stderr':open(stderr,encoding='utf-8').read(),'timestamp_epoch':time.time()}
fd,tmp=tempfile.mkstemp(prefix=os.path.basename(out)+'.',dir=os.path.dirname(out)); os.close(fd)
with open(tmp,'w',encoding='utf-8',newline='\n') as h: json.dump(record,h,indent=2,sort_keys=True); h.write('\n'); h.flush(); os.fsync(h.fileno())
os.replace(tmp,out)
PY
  return "$rc"
}
launch_probe_step() {
  local actor="$1" identity="$2" heartbeat="$3"
  srun --exclusive --ntasks=1 --cpus-per-task=1 bash -c '
    set -euo pipefail
    actor="$1"; identity="$2"; heartbeat="$3"; root="$4"; py="$5"
    "$py" - "$identity" "$actor" "$heartbeat" <<'"'"'PY'"'"'
import json,os,socket,sys,tempfile,time
out,actor,heartbeat=sys.argv[1:]
r={'schema':'khz_filament.hr4e5s.s5_final.site_observability.v1','kind':'identity','actor':actor,'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'hostname':socket.gethostname(),'start_time':time.time(),'heartbeat_path':heartbeat,'receipt_version':'v1'}
fd,tmp=tempfile.mkstemp(prefix=os.path.basename(out)+'.',dir=os.path.dirname(out)); os.close(fd)
with open(tmp,'w',encoding='utf-8',newline='"'"'\n'"'"') as h: json.dump(r,h,sort_keys=True); h.write('"'"'\n'"'"'); h.flush(); os.fsync(h.fileno())
os.replace(tmp,out)
PY
    while true; do date +%s%N >"$heartbeat"; sleep 1; done
  ' bash "$actor" "$identity" "$heartbeat" "$ROOT" "$PYTHON" &
  launched_pid="$!"
}
wait_for_file() { local path="$1" n=0; while [[ ! -f "$path" && "$n" -lt 30 ]]; do sleep 1; n=$((n+1)); done; test -f "$path"; }
fail() { write_json "$ROOT/probe_result.json" probe_result "status=FAIL" "reason=$1"; exit 1; }

write_json "$ROOT/probe_manifest.json" probe_manifest "status=RUNNING" "allocation_id=$SLURM_JOB_ID" "partition=${SLURM_JOB_PARTITION:-unknown}" "gpu_requested=0" "controller_hostname=$(hostname)"
launch_probe_step probe_survivor "$ROOT/identities/survivor.json" "$ROOT/heartbeats/survivor"; survivor_srun="$launched_pid"
launch_probe_step probe_target "$ROOT/identities/target.json" "$ROOT/heartbeats/target"; target_srun="$launched_pid"
wait_for_file "$ROOT/identities/survivor.json" && wait_for_file "$ROOT/identities/target.json" || fail IDENTITY_RECEIPTS_INCOMPLETE

"$PYTHON" - "$ROOT/identities/survivor.json" "$ROOT/identities/target.json" "$ROOT/controller_identity.json" <<'PY'
import json,os,socket,sys,time,tempfile
s,t,out=(json.load(open(x,encoding='utf-8')) for x in sys.argv[1:])
assert s['job_id']==t['job_id']==os.environ['SLURM_JOB_ID'] and s['step_id']!=t['step_id'] and s['pid']!=t['pid']
r={'schema':'khz_filament.hr4e5s.s5_final.site_observability.v1','kind':'controller_identity','controller_hostname':socket.gethostname(),'target_hostname':t['hostname'],'survivor_hostname':s['hostname'],'target_step':f"{t['job_id']}.{t['step_id']}",'survivor_step':f"{s['job_id']}.{s['step_id']}",'timestamp_epoch':time.time()}
fd,tmp=tempfile.mkstemp(prefix=os.path.basename(out)+'.',dir=os.path.dirname(out)); os.close(fd)
with open(tmp,'w',encoding='utf-8',newline='\n') as h: json.dump(r,h,indent=2,sort_keys=True); h.write('\n'); h.flush(); os.fsync(h.fileno())
os.replace(tmp,out)
PY

target_step="$("$PYTHON" - "$ROOT/identities/target.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8')); print(f"{x['job_id']}.{x['step_id']}")
PY
)"
target_pid="$("$PYTHON" - "$ROOT/identities/target.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1],encoding='utf-8'))['pid'])
PY
)"
capture_listpids listpids_before "$target_step" || fail LISTPIDS_BEFORE_RETURN_NONZERO
"$PYTHON" - "$ROOT/listpids_before.json" "$target_pid" <<'PY' || fail TARGET_PID_NOT_PROVEN
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8')); assert str(sys.argv[2]) in x['stdout']
PY
test -s "$ROOT/heartbeats/target" && test -s "$ROOT/heartbeats/survivor" || fail HEARTBEATS_NOT_LIVE
write_json "$ROOT/signal_intent.json" signal_intent "status=INTENT" "target_step=$target_step" "target_pid=$target_pid" "signal=TERM"
test ! -e "$ROOT/signal_receipt.json" || fail DUPLICATE_SIGNAL_RECEIPT
signal_rc=0; scancel --signal=TERM "$target_step" || signal_rc=$?
write_json "$ROOT/signal_receipt.json" signal_receipt "status=SENT" "target_step=$target_step" "target_pid=$target_pid" "signal=TERM" "return_code=$signal_rc"
test "$signal_rc" -eq 0 || fail SIGNAL_FAILED
target_wait=0; wait "$target_srun" || target_wait=$?
test "$target_wait" -ne 0 || fail TARGET_DID_NOT_EXIT
kill -0 "$("$PYTHON" - "$ROOT/identities/survivor.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1],encoding='utf-8'))['pid'])
PY
)" 2>/dev/null || fail SIGNAL_SCOPE_TOO_BROAD
capture_listpids listpids_after "$target_step" || fail LISTPIDS_AFTER_RETURN_NONZERO
"$PYTHON" - "$ROOT/listpids_after.json" "$target_pid" <<'PY' || fail TARGET_STILL_VISIBLE_AFTER_SIGNAL
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8')); assert str(sys.argv[2]) not in x['stdout']
PY
write_json "$ROOT/target_quiescence_receipt.json" target_quiescence "status=PASS" "target_step=$target_step" "target_pid=$target_pid"
write_json "$ROOT/survivor_status.json" survivor_status "status=PASS" "survivor_srun_pid=$survivor_srun"
kill "$survivor_srun" 2>/dev/null || true; wait "$survivor_srun" 2>/dev/null || true
write_json "$ROOT/probe_result.json" probe_result "status=PASS" "duplicate_signal_count=1" "gpu_requested=0" "allocation_id=$SLURM_JOB_ID"
