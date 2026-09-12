#!/usr/bin/env bash
# Submit exactly one already-preflighted S5 clean or fault case.
set -euo pipefail
readonly REPO="$1" RUN_ROOT="$2" EXPECTED_SHA="$3" PREFLIGHT="$4" CASE_ID="$5" CASE_MODE="$6" REFERENCE_CASE_ROOT="$7" FAULT_ID="${8:-}" FAULT_SCREEN="${9:-}"
readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
readonly BATCH="$REPO/Filament_python/tools/hr4e5s_s5.sbatch"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA" && test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -d "$RUN_ROOT" && test -f "$PREFLIGHT" && test -f "$RUN_ROOT/hr4e5s_s5_input_manifest.json" && test -d "$RUN_ROOT/lut_workspace"
test ! -e "$RUN_ROOT/$CASE_ID" && test ! -e "$RUN_ROOT/${CASE_ID}_submission_receipt.tsv"
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/audit_batch_entry.py" --batch "$BATCH" --fixed-python "$PYTHON" >/dev/null
if [[ "$CASE_MODE" == fault ]]; then
  test "$REFERENCE_CASE_ROOT" = "$RUN_ROOT/clean"
  test -d "$REFERENCE_CASE_ROOT/lifecycle" && test -d "$REFERENCE_CASE_ROOT/optical" && test -f "$REFERENCE_CASE_ROOT/final_lifecycle_audit.json"
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
assert (case=='clean' and mode=='clean' and not fault and not screen) or (case==fault.split('_',1)[0] and mode=='fault' and reference and fault and screen)
PY
export_args="ALL,EXPECTED_GIT_SHA=$EXPECTED_SHA,REPO_DIR=$REPO,RUN_ROOT=$RUN_ROOT,CASE_ID=$CASE_ID,CASE_MODE=$CASE_MODE,INPUT_MANIFEST=$RUN_ROOT/hr4e5s_s5_input_manifest.json,LUT_WORKSPACE=$RUN_ROOT/lut_workspace,REFERENCE_CASE_ROOT=$REFERENCE_CASE_ROOT"
if [[ "$CASE_MODE" == fault ]]; then export_args+=",HR4_S5_FAULT_ID=$FAULT_ID,HR4_S5_FAULT_SCREEN=$FAULT_SCREEN,HR4_S5_FAULT_ONCE=1"; fi
job="$(sbatch --parsable --job-name="e5s-s5-$CASE_ID" --gres=gpu:2 --ntasks=2 --output="$RUN_ROOT/$CASE_ID-%j.out" --error="$RUN_ROOT/$CASE_ID-%j.err" --export="$export_args" "$BATCH")"
job="${job%%;*}"; [[ "$job" =~ ^[0-9]+$ ]]
printf 'case_id\tcase_mode\tfault_id\tfault_screen\tjob_id\toptical_gpus\thydro_gpus\texecution_sha\n%s\t%s\t%s\t%s\t%s\t1\t1\t%s\n' "$CASE_ID" "$CASE_MODE" "$FAULT_ID" "$FAULT_SCREEN" "$job" "$EXPECTED_SHA" > "$RUN_ROOT/${CASE_ID}_submission_receipt.tsv"
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","case_id":"%s","job_id":"%s"}\n' "$CASE_ID" "$job"
