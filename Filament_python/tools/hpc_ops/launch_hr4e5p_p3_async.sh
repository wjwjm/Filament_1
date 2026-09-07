#!/usr/bin/env bash
# Start a bounded P3 preflight/launcher chain without coupling it to an SSH
# foreground timeout.  The child writes durable status and logs; this wrapper
# returns only after the child PID has been recorded.
set -euo pipefail

readonly REPO="$1" RUN_ROOT="$2" EXPECTED_SHA="$3" PREFLIGHT_OUT="$4" LAUNCH_MODE="$5"
readonly PYTHON="/data/home/scvi806/.conda/envs/Filament_python/bin/python"
readonly PREFLIGHT_RUNNER="$REPO/Filament_python/tools/hpc_ops/run_hr4e5p_preflight.sh"
readonly P3_LAUNCHER="$REPO/Filament_python/tools/hpc_ops/submit_hr4e5p_p3.sh"
readonly STATUS="${PREFLIGHT_OUT%.json}_async_status.json"
readonly ASYNC_DIR="$(dirname -- "$PREFLIGHT_OUT")"
readonly PREFLIGHT_STDOUT="${PREFLIGHT_OUT%.json}_preflight.stdout"
readonly PREFLIGHT_STDERR="${PREFLIGHT_OUT%.json}_preflight.stderr"
readonly LAUNCHER_STDOUT="${PREFLIGHT_OUT%.json}_launcher.stdout"
readonly LAUNCHER_STDERR="${PREFLIGHT_OUT%.json}_launcher.stderr"

test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -x "$PYTHON" && test -f "$PREFLIGHT_RUNNER" && test -f "$P3_LAUNCHER"
test ! -e "$RUN_ROOT" && test ! -e "$ASYNC_DIR"
case "$LAUNCH_MODE" in submit|no-submit) ;; *) exit 64 ;; esac
mkdir -m 700 -- "$ASYNC_DIR"

write_status() {
  "$PYTHON" - "$STATUS" "$1" "$2" "$3" <<'PY'
import json,os,sys
path,state,stage,code=sys.argv[1:]
temporary=path + '.tmp'
with open(temporary,'w',encoding='utf-8',newline='\n') as handle:
    json.dump({'schema':'khz_filament.hr4e5p.p3_async_status.v1','state':state,'stage':stage,'exit_code':int(code)},handle,indent=2,sort_keys=True)
    handle.write('\n')
os.replace(temporary,path)
PY
}

write_status STARTED preflight 0
(
  trap '' HUP
  set +e
  bash "$PREFLIGHT_RUNNER" "$REPO" "$PREFLIGHT_OUT" "$EXPECTED_SHA" >"$PREFLIGHT_STDOUT" 2>"$PREFLIGHT_STDERR"
  preflight_rc=$?
  "$PYTHON" - "$PREFLIGHT_STDOUT" <<'PY'
import json,sys
lines=[line for line in open(sys.argv[1],encoding='utf-8') if line.strip()]
assert len(lines)==1
receipt=json.loads(lines[0])
assert receipt['ok'] is True and receipt['state']=='completed'
PY
  preflight_json_rc=$?
  if [[ "$preflight_rc" != 0 || "$preflight_json_rc" != 0 ]]; then
    failure_rc="$preflight_rc"
    if [[ "$failure_rc" == 0 ]]; then failure_rc=1; fi
    write_status FAILED preflight "$failure_rc"
    exit 0
  fi
  bash "$P3_LAUNCHER" "$REPO" "$RUN_ROOT" "$EXPECTED_SHA" "$PREFLIGHT_OUT" "$LAUNCH_MODE" >"$LAUNCHER_STDOUT" 2>"$LAUNCHER_STDERR"
  launcher_rc=$?
  if [[ "$launcher_rc" == 0 ]]; then write_status COMPLETED launcher 0; else write_status FAILED launcher "$launcher_rc"; fi
  if [[ -d "$RUN_ROOT" ]]; then cp -- "$ASYNC_DIR"/* "$RUN_ROOT/"; fi
  exit 0
) </dev/null >/dev/null 2>&1 &
child_pid=$!
"$PYTHON" - "$STATUS" "$child_pid" <<'PY'
import json,os,sys
path,pid=sys.argv[1:]
record=json.load(open(path,encoding='utf-8'))
record['pid']=int(pid)
temporary=path + '.tmp'
with open(temporary,'w',encoding='utf-8',newline='\n') as handle:
    json.dump(record,handle,indent=2,sort_keys=True)
    handle.write('\n')
os.replace(temporary,path)
PY
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","operation_state":"started","status":"%s","pid":%s,"run_root":"%s","stage":"P3"}\n' "$STATUS" "$child_pid" "$RUN_ROOT"
