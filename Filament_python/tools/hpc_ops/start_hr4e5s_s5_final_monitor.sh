#!/usr/bin/env bash
# Start one resumable S5-FINAL controller only after an initial numeric receipt.
set -euo pipefail
readonly REPO="$1" RUN_ROOT="$2" EXPECTED_SHA="$3" PREFLIGHT="$4" SUBMIT_SCRIPT="$5" REFERENCE_CASE_ROOT="$6"
readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
readonly RECEIPT="$RUN_ROOT/initial_submission_receipt.tsv" MANIFEST="$RUN_ROOT/s5_final_monitor_manifest.json" START="$RUN_ROOT/monitor_start_receipt.json"
test -f "$RECEIPT" && test -f "$PREFLIGHT" && test ! -e "$MANIFEST" && test ! -e "$START"
initial_job="$(awk -F '\t' 'NR==2 {print $3}' "$RECEIPT")"; [[ "$initial_job" =~ ^[0-9]+$ ]]
"$PYTHON" - "$MANIFEST" "$REPO" "$RUN_ROOT" "$EXPECTED_SHA" "$PREFLIGHT" "$SUBMIT_SCRIPT" "$REFERENCE_CASE_ROOT" "$initial_job" <<'PY'
import json,sys
out,repo,root,sha,preflight,submit,reference,job=sys.argv[1:]
value={'schema':'khz_filament.hr4e5s.s5_final.monitor_manifest.v1','repo':repo,'run_root':root,'expected_sha':sha,'preflight':preflight,'submit_script':submit,'reference_case_root':reference,'initial_job_id':job,'target_actor':'hydro_consumer_1'}
json.dump(value,open(out,'x',encoding='utf-8'),indent=2,sort_keys=True)
PY
nohup "$PYTHON" "$REPO/Filament_python/tools/monitor_hr4e5s_s5_final.py" --manifest "$MANIFEST" --resume --poll-seconds 15 >"$RUN_ROOT/monitor.log" 2>"$RUN_ROOT/monitor.err" < /dev/null &
pid="$!"
sleep 1
kill -0 "$pid"
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","monitor_pid":%s,"manifest":"%s","log":"%s"}\n' "$pid" "$MANIFEST" "$RUN_ROOT/monitor.log" >"$START"
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","monitor_pid":%s,"receipt":"%s"}\n' "$pid" "$START"
