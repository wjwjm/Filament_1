#!/usr/bin/env bash
# Submit only the independent four-domain E3-A synthetic block.
set -euo pipefail
readonly REPO="$1" RUN_ROOT="$2" EXPECTED_SHA="$3" PREFLIGHT="$4"
readonly PROVENANCE="$5"
readonly BATCH="$REPO/Filament_python/tools/hr4e3_domain_case.sbatch"
readonly PYTHON="/data/home/scvi806/.conda/envs/Filament_python/bin/python"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -x "$PYTHON" && test -f "$BATCH" && test -f "$PREFLIGHT" && test -f "$PROVENANCE" && test ! -e "$RUN_ROOT"
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/provenance_v2.py" validate --repo "$REPO" --manifest "$PROVENANCE" --require-hash-scope >/dev/null
"$PYTHON" - "$PREFLIGHT" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
assert report["status"] == "PASS"
assert report["backend"]["backend"] == "cupy"
PY
mkdir -m 700 -- "$RUN_ROOT"
receipt="$RUN_ROOT/submission_receipt.tsv"
printf 'case_id\tkind\tdomain\tjob_id\n' > "$receipt"
for domain in D0 D1 D2 D3; do
    case_id="E3A_${domain}_dx10um_dt1p0us"
    case_dir="$RUN_ROOT/$case_id"
    mkdir -m 700 -- "$case_dir"
    submission="$(sbatch --parsable --job-name="$case_id" --output="$case_dir/slurm-%j.out" --error="$case_dir/slurm-%j.err" --export="ALL,EXPECTED_GIT_SHA=$EXPECTED_SHA,REPO_DIR=$REPO,CASE_DIR=$case_dir,CASE_ID=$case_id,KIND=synthetic,DOMAIN_ID=$domain" "$BATCH")"
    job_id="${submission%%;*}"
    [[ "$job_id" =~ ^[0-9]+$ ]]
    printf '%s\t%s\t%s\t%s\n' "$case_id" synthetic "$domain" "$job_id" >> "$receipt"
done
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","run_root":"%s","receipt":"%s"}\n' "$RUN_ROOT" "$receipt"
