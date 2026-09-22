#!/usr/bin/env bash
# Submit exactly one S5-FINAL initial or recovery job after durable receipts.
set -euo pipefail
readonly REPO="$1" RUN_ROOT="$2" EXPECTED_SHA="$3" PREFLIGHT="$4" CASE_MODE="$5" REFERENCE_CASE_ROOT="$6"
readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
readonly LEGACY_BATCH="$REPO/Filament_python/tools/hr4e5s_s5_final.sbatch"
readonly SINGLE_ALLOCATION_BATCH="$REPO/Filament_python/tools/hr4e5s_s5_final_single_allocation.sbatch"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA" && test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -d "$RUN_ROOT" && test -f "$PREFLIGHT" && test -f "$RUN_ROOT/s5_final_input_manifest.json" && test -d "$RUN_ROOT/lut_workspace"
test -d "$REFERENCE_CASE_ROOT/lifecycle" && test -d "$REFERENCE_CASE_ROOT/optical"
case "$CASE_MODE" in
  initial) BATCH="$LEGACY_BATCH"; RECEIPT="$RUN_ROOT/initial_submission_receipt.tsv"; test ! -e "$RECEIPT" && test ! -e "$RUN_ROOT/scenario" ;;
  recovery) BATCH="$LEGACY_BATCH"; RECEIPT="$RUN_ROOT/recovery_submission_receipt.tsv"; test -f "$RUN_ROOT/scenario/old_job_quiescence.json" && test -f "$RUN_ROOT/scenario/expected_recovery_effects.json" && test ! -e "$RECEIPT" && test ! -e "$RUN_ROOT/scenario/recovery" ;;
  single_allocation) BATCH="$SINGLE_ALLOCATION_BATCH"; RECEIPT="$RUN_ROOT/single_allocation_submission_receipt.tsv"; test ! -e "$RECEIPT" && test ! -e "$RUN_ROOT/scenario" ;;
  *) exit 64 ;;
esac
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/audit_batch_entry.py" --batch "$BATCH" --fixed-python "$PYTHON" >/dev/null
"$PYTHON" - "$PREFLIGHT" "$EXPECTED_SHA" "$RUN_ROOT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x['status']=='PASS' and x['git_sha']==sys.argv[2] and x['run_root']==sys.argv[3]
assert x['resources']=={'optical_gpus':1,'hydro_gpus':2} and x['fault_injection_default']=='DISABLED'
PY
"$PYTHON" "$REPO/Filament_python/tools/run_hr4e5s_s5_final.py" validate-reference --reference-root "$REFERENCE_CASE_ROOT" --input "$RUN_ROOT/s5_final_input_manifest.json" >/dev/null
export_args="ALL,EXPECTED_GIT_SHA=$EXPECTED_SHA,REPO_DIR=$REPO,RUN_ROOT=$RUN_ROOT,CASE_MODE=$CASE_MODE,INPUT_MANIFEST=$RUN_ROOT/s5_final_input_manifest.json,LUT_WORKSPACE=$RUN_ROOT/lut_workspace,REFERENCE_CASE_ROOT=$REFERENCE_CASE_ROOT"
job="$(sbatch --parsable --job-name="e5s-s5-final-${CASE_MODE}" --chdir="$RUN_ROOT" --gres=gpu:3 --ntasks=3 --output="$RUN_ROOT/${CASE_MODE}-%j.out" --error="$RUN_ROOT/${CASE_MODE}-%j.err" --export="$export_args" "$BATCH")"
job="${job%%;*}"; [[ "$job" =~ ^[0-9]+$ ]]
printf 'case_id\tcase_mode\tjob_id\toptical_gpus\thydro_gpus\texecution_sha\nS5_FINAL_WORKER_LOSS\t%s\t%s\t1\t2\t%s\n' "$CASE_MODE" "$job" "$EXPECTED_SHA" >"$RECEIPT.tmp"
mv -- "$RECEIPT.tmp" "$RECEIPT"
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","case_mode":"%s","job_id":"%s","receipt":"%s"}\n' "$CASE_MODE" "$job" "$RECEIPT"
