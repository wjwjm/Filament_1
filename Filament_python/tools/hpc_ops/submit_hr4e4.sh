#!/usr/bin/env bash
set -euo pipefail
readonly REPO="$1" RUN_ROOT="$2" EXPECTED_SHA="$3" AUDIT="$4"
readonly BATCH="$REPO/Filament_python/tools/hr4e4_case.sbatch"; readonly PYTHON="/data/home/scvi806/.conda/envs/Filament_python/bin/python"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"; test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"; test -f "$BATCH" && test -f "$AUDIT" && test ! -e "$RUN_ROOT"
"$PYTHON" - "$AUDIT" <<'PY'
import json,sys
assert json.load(open(sys.argv[1],encoding='utf-8'))['status']=='ADVECTION_EVIDENCE_REUSE_PASS'
PY
mkdir -m 700 -- "$RUN_ROOT"; receipt="$RUN_ROOT/submission_receipt.tsv"; printf 'case_id\tjob_id\n' > "$receipt"
for case_id in E4A E4Bnu E4B0 E4Cminus E4Cplus E4D; do
 case_dir="$RUN_ROOT/$case_id"; mkdir -m 700 -- "$case_dir"; job="$(sbatch --parsable --job-name="$case_id" --output="$case_dir/slurm-%j.out" --error="$case_dir/slurm-%j.err" --export="ALL,EXPECTED_GIT_SHA=$EXPECTED_SHA,REPO_DIR=$REPO,CASE_DIR=$case_dir,CASE_ID=$case_id" "$BATCH")"; job="${job%%;*}"; [[ "$job" =~ ^[0-9]+$ ]]; printf '%s\t%s\n' "$case_id" "$job" >> "$receipt"
done
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","run_root":"%s","receipt":"%s"}\n' "$RUN_ROOT" "$receipt"
