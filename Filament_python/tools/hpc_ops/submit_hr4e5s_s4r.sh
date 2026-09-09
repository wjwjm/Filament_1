#!/usr/bin/env bash
# Submit exactly one S4R hydro replay after a successful no-submit preflight.
set -euo pipefail
readonly REPO="$1" RUN_ROOT="$2" EXPECTED_SHA="$3" PREFLIGHT="$4" INPUT_MANIFEST="$5" CASE_ID="$6" HYDRO_WORKERS="$7"
readonly BATCH="$REPO/Filament_python/tools/hr4e5s_s4r.sbatch"
readonly PYTHON=/data/home/scvi806/.conda/envs/Filament_python/bin/python
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -f "$PREFLIGHT" && test -f "$INPUT_MANIFEST"
[[ "$HYDRO_WORKERS" =~ ^(1|2|4|8)$ ]]
test ! -e "$RUN_ROOT/${CASE_ID}_submission_receipt.tsv"
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/audit_batch_entry.py" --batch "$BATCH" --fixed-python "$PYTHON" >/dev/null
"$PYTHON" - "$PREFLIGHT" "$EXPECTED_SHA" "$RUN_ROOT" "$INPUT_MANIFEST" "$HYDRO_WORKERS" <<'PY'
import json,sys
preflight,sha,root,input_path,workers=sys.argv[1:]
x=json.load(open(preflight,encoding='utf-8'))
assert x['status']=='PASS' and x['git_sha']==sha and x['run_root']==root and x['input_manifest']==input_path
assert int(workers) in (1,2,4,8)
PY
job="$(sbatch --parsable --job-name="e5s-s4r-${CASE_ID}" --gres="gpu:${HYDRO_WORKERS}" --ntasks="$HYDRO_WORKERS" --output="$RUN_ROOT/${CASE_ID}-%j.out" --error="$RUN_ROOT/${CASE_ID}-%j.err" --export="ALL,EXPECTED_GIT_SHA=$EXPECTED_SHA,REPO_DIR=$REPO,RUN_ROOT=$RUN_ROOT,CASE_ID=$CASE_ID,HYDRO_WORKERS=$HYDRO_WORKERS,INPUT_MANIFEST=$INPUT_MANIFEST" "$BATCH")"
job="${job%%;*}"; [[ "$job" =~ ^[0-9]+$ ]] || exit 1
printf 'case_id\thydro_workers\tgpu_count\tjob_id\n%s\t%s\t%s\t%s\n' "$CASE_ID" "$HYDRO_WORKERS" "$HYDRO_WORKERS" "$job" > "$RUN_ROOT/${CASE_ID}_submission_receipt.tsv"
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","job_id":"%s"}\n' "$job"
