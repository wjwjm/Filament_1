#!/usr/bin/env bash
# Submit exactly one already-preflighted S5 clean, fault, or recovery case.
set -euo pipefail
readonly REPO="$1" RUN_ROOT="$2" EXPECTED_SHA="$3" PREFLIGHT="$4" CASE_ID="$5" CASE_MODE="$6" REFERENCE_CASE_ROOT="$7" FAULT_ID="${8:-}" FAULT_SCREEN="${9:-}"
readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
readonly BATCH="$REPO/Filament_python/tools/hr4e5s_s5.sbatch"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA" && test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -d "$RUN_ROOT" && test -f "$PREFLIGHT" && test -f "$RUN_ROOT/hr4e5s_s5_input_manifest.json" && test -d "$RUN_ROOT/lut_workspace"
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/audit_batch_entry.py" --batch "$BATCH" --fixed-python "$PYTHON" >/dev/null
case "$CASE_MODE" in
  clean)
    test "$CASE_ID" = clean && test -z "$REFERENCE_CASE_ROOT" && test -z "$FAULT_ID" && test -z "$FAULT_SCREEN"
    readonly RECEIPT="$RUN_ROOT/clean_submission_receipt.tsv"
    test ! -e "$RUN_ROOT/$CASE_ID" && test ! -e "$RECEIPT"
    ;;
  fault)
    [[ "$CASE_ID" =~ ^F0[1-6]$ ]] && [[ "$FAULT_ID" == "${CASE_ID}_"* ]] && test -n "$FAULT_SCREEN"
    readonly RECEIPT="$RUN_ROOT/${CASE_ID}_fault_submission_receipt.tsv"
    test ! -e "$RUN_ROOT/$CASE_ID" && test ! -e "$RECEIPT"
    ;;
  recovery)
    [[ "$CASE_ID" =~ ^F0[1-6]$ ]] && test -z "$FAULT_ID" && test -z "$FAULT_SCREEN"
    readonly RECEIPT="$RUN_ROOT/${CASE_ID}_recovery_submission_receipt.tsv"
    test -d "$RUN_ROOT/$CASE_ID" && test -f "$RUN_ROOT/$CASE_ID/injected/disk_state_audit.json"
    test -f "$RUN_ROOT/$CASE_ID/contract_check.json" && test ! -e "$RUN_ROOT/$CASE_ID/recovery" && test ! -e "$RECEIPT"
    ;;
  *) echo "invalid S5 case mode=$CASE_MODE" >&2; exit 64 ;;
esac
if [[ "$CASE_MODE" == fault || "$CASE_MODE" == recovery ]]; then
  test "$REFERENCE_CASE_ROOT" = "$RUN_ROOT/clean"
  test -d "$REFERENCE_CASE_ROOT/lifecycle" && test -d "$REFERENCE_CASE_ROOT/optical" && test -f "$REFERENCE_CASE_ROOT/final_lifecycle_audit.json"
  test -f "$RUN_ROOT/clean_submission_receipt.tsv"
  reference_job="$(awk -F '\t' 'NR==2 {print $5}' "$RUN_ROOT/clean_submission_receipt.tsv")"
  [[ "$reference_job" =~ ^[0-9]+$ ]]
  test "$(sacct -X -j "$reference_job" --format=State,ExitCode --parsable2 --noheader | head -n 1)" = 'COMPLETED|0:0'
  "$PYTHON" - "$REFERENCE_CASE_ROOT/final_lifecycle_audit.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
assert x['pointer_present'] is True
assert x['barrier']['status']=='PASS'
assert x['promotion']['authoritative_namespace']=='NEXT'
PY
fi
"$PYTHON" - "$PREFLIGHT" "$EXPECTED_SHA" "$RUN_ROOT" "$CASE_ID" "$CASE_MODE" "$REFERENCE_CASE_ROOT" "$FAULT_ID" "$FAULT_SCREEN" <<'PY'
import json,sys
p,sha,root,case,mode,reference,fault,screen=sys.argv[1:]
x=json.load(open(p,encoding='utf-8'))
assert x['status']=='PASS' and x['git_sha']==sha and x['run_root']==root and x['resources']=={'optical_gpus':1,'hydro_gpus':1} and x['fault_injection_default']=='DISABLED'
assert (
    (case=='clean' and mode=='clean' and not reference and not fault and not screen)
    or (case==fault.split('_',1)[0] and mode=='fault' and reference and fault and screen)
    or (case.startswith('F0') and mode=='recovery' and reference and not fault and not screen)
)
PY
if [[ "$CASE_MODE" == recovery ]]; then
  "$PYTHON" - "$RUN_ROOT/$CASE_ID/contract_check.json" "$CASE_ID" <<'PY'
import json,sys
path,case_id=sys.argv[1:]
check=json.load(open(path,encoding='utf-8'))
assert check['status']=='PASS' and check['contract_match'] is True
assert str(check['fault_id']).startswith(case_id + '_')
PY
fi
export_args="ALL,EXPECTED_GIT_SHA=$EXPECTED_SHA,REPO_DIR=$REPO,RUN_ROOT=$RUN_ROOT,CASE_ID=$CASE_ID,CASE_MODE=$CASE_MODE,INPUT_MANIFEST=$RUN_ROOT/hr4e5s_s5_input_manifest.json,LUT_WORKSPACE=$RUN_ROOT/lut_workspace,REFERENCE_CASE_ROOT=$REFERENCE_CASE_ROOT"
if [[ "$CASE_MODE" == fault ]]; then export_args+=",HR4_S5_FAULT_ID=$FAULT_ID,HR4_S5_FAULT_SCREEN=$FAULT_SCREEN,HR4_S5_FAULT_ONCE=1"; fi
job="$(sbatch --parsable --job-name="e5s-s5-${CASE_ID}-${CASE_MODE}" --chdir="$RUN_ROOT" --gres=gpu:2 --ntasks=2 --output="$RUN_ROOT/${CASE_ID}-${CASE_MODE}-%j.out" --error="$RUN_ROOT/${CASE_ID}-${CASE_MODE}-%j.err" --export="$export_args" "$BATCH")"
job="${job%%;*}"; [[ "$job" =~ ^[0-9]+$ ]]
attempt="$RECEIPT.tmp"
printf 'case_id\tcase_mode\tfault_id\tfault_screen\tjob_id\toptical_gpus\thydro_gpus\texecution_sha\n%s\t%s\t%s\t%s\t%s\t1\t1\t%s\n' "$CASE_ID" "$CASE_MODE" "$FAULT_ID" "$FAULT_SCREEN" "$job" "$EXPECTED_SHA" > "$attempt"
mv -- "$attempt" "$RECEIPT"
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","case_id":"%s","case_mode":"%s","job_id":"%s","receipt":"%s"}\n' "$CASE_ID" "$CASE_MODE" "$job" "$RECEIPT"
