#!/usr/bin/env bash
# Start an isolated S5-FINAL preflight durably; its receipt is outside RUN_ROOT
# so an interrupted setup cannot be mistaken for a completed preflight.
set -euo pipefail
readonly REPO="$1" RUN_ROOT="$2" OUT="$3" EXPECTED_SHA="$4" SOURCE_MANIFEST="$5" SOURCE_STATE="$6" SOURCE_CONFIG="$7"
readonly PREFLIGHT="$REPO/Filament_python/tools/hpc_ops/run_hr4e5s_s5_final_preflight.sh"
readonly STATUS_DIR="${RUN_ROOT}.s5_final_preflight_async"
test -x "$PREFLIGHT" || test -f "$PREFLIGHT"
test ! -e "$RUN_ROOT" && test ! -e "$STATUS_DIR"
umask 077
mkdir -m 700 -- "$STATUS_DIR"
printf '{"schema":"filament.hpc_ops.async_preflight.v1","status":"STARTING","run_root":"%s","expected_sha":"%s"}\n' "$RUN_ROOT" "$EXPECTED_SHA" >"$STATUS_DIR/start.json"
nohup bash "$PREFLIGHT" "$REPO" "$RUN_ROOT" "$OUT" "$EXPECTED_SHA" "$SOURCE_MANIFEST" "$SOURCE_STATE" "$SOURCE_CONFIG" >"$STATUS_DIR/preflight.stdout" 2>"$STATUS_DIR/preflight.stderr" < /dev/null &
pid="$!"
sleep 1
kill -0 "$pid"
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","pid":%s,"status_dir":"%s"}\n' "$pid" "$STATUS_DIR"
