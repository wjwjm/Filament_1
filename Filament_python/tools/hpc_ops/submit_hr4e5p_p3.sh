#!/usr/bin/env bash
set -euo pipefail
readonly REPO="$1" RUN_ROOT="$2" EXPECTED_SHA="$3" PREFLIGHT="$4"
readonly LAUNCH_MODE="${5:-submit}"
readonly PYTHON="/data/home/scvi806/.conda/envs/Filament_python/bin/python"
readonly BATCH="$REPO/Filament_python/tools/hr4e5p_parallel.sbatch"
readonly SOURCE_ROOT="/data/run01/scvi806/user_Wangjimin/projects/hr4e1_runs_e1b_be280dc"
readonly SOURCE_MANIFEST="$SOURCE_ROOT/post_reference/post_reference_manifest.json"
readonly SOURCE_STATE="$SOURCE_ROOT/E1B_hr3b_source.hr3b_delta_n_th.npy"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -f "$BATCH" && test -f "$SOURCE_MANIFEST" && test -f "$SOURCE_STATE" && test -f "$PREFLIGHT" && test ! -e "$RUN_ROOT"
case "$LAUNCH_MODE" in submit|no-submit) ;; *) exit 64 ;; esac
"$PYTHON" "$REPO/Filament_python/tools/hpc_ops/audit_batch_entry.py" --batch "$BATCH" --fixed-python "$PYTHON" >/dev/null
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python
export UPPE_USE_GPU=1 PYTHONPATH="$REPO/Filament_python"
mkdir -m 700 -- "$RUN_ROOT"
cp -- "$PREFLIGHT" "$RUN_ROOT/p3_submission_preflight.json"
"$PYTHON" - "$PREFLIGHT" "$EXPECTED_SHA" <<'PY'
import json,sys
preflight=json.load(open(sys.argv[1],encoding='utf-8'))
assert preflight['status']=='PASS'
assert preflight['git_sha']==sys.argv[2]
assert preflight['source_state_file_sha256']=='70677c01564ad089d985214e2767a221f10c5e1ef97f42a9f367a89edf81f467'
PY
"$PYTHON" "$REPO/Filament_python/tools/run_hr4e5p.py" audit --out "$RUN_ROOT/e5p_screen_independence_audit.json"
"$PYTHON" "$REPO/Filament_python/tools/run_hr4e5p.py" dry-run --n-screens 15000 --block-size 32 --n-workers 7 --out "$RUN_ROOT/full_screen_partition_dry_run.json"
readonly P3_INDICES="0,1,2500,5000,7827,8022,9000,10338,12000,14997,14998,14999"
P3_NAMES=(p3_serial p3_parallel_g1 p3_parallel_g2 p3_parallel_g4 p3_parallel_g7)
P3_GPUS=(1 1 2 4 7)
P3_MODES=(serial parallel parallel parallel parallel)
prepare_case() {
  local name="$1"
  local workers="$2"
  local dir="$RUN_ROOT/$name"
  "$PYTHON" "$REPO/Filament_python/tools/run_hr4e5p.py" prepare --source-manifest "$SOURCE_MANIFEST" --source-state "$SOURCE_STATE" --verified-source-state-file-sha256 70677c01564ad089d985214e2767a221f10c5e1ef97f42a9f367a89edf81f467 --screen-index "$P3_INDICES" --out-dir "$dir"
  "$PYTHON" "$REPO/Filament_python/tools/run_hr4e5p.py" partition --input "$dir/validation_input.json" --block-size 4 --n-workers "$workers" --out "$dir/partition.json"
}
for i in "${!P3_NAMES[@]}"; do prepare_case "${P3_NAMES[$i]}" "${P3_GPUS[$i]}"; done
readonly PAYLOAD_VALIDATION="$RUN_ROOT/p3_payload_validation.json"
readonly PLAN="$RUN_ROOT/p3_submission_plan.json"
"$PYTHON" "$REPO/Filament_python/tools/run_hr4e5p.py" validate-p3 --case "p3_serial=$RUN_ROOT/p3_serial/validation_input.json" --case "p3_parallel_g1=$RUN_ROOT/p3_parallel_g1/validation_input.json" --case "p3_parallel_g2=$RUN_ROOT/p3_parallel_g2/validation_input.json" --case "p3_parallel_g4=$RUN_ROOT/p3_parallel_g4/validation_input.json" --case "p3_parallel_g7=$RUN_ROOT/p3_parallel_g7/validation_input.json" --out "$PAYLOAD_VALIDATION"
"$PYTHON" "$REPO/Filament_python/tools/run_hr4e5p.py" p3-plan --payload-validation "$PAYLOAD_VALIDATION" --expected-git-sha "$EXPECTED_SHA" --repo "$REPO" --batch "$BATCH" --out "$PLAN"
if [[ "$LAUNCH_MODE" == "no-submit" ]]; then
  printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","mode":"no_submit","run_root":"%s","plan":"%s","stage":"P3"}\n' "$RUN_ROOT" "$PLAN"
  exit 0
fi
"$PYTHON" "$REPO/Filament_python/tools/run_hr4e5p.py" submit-p3-plan --plan "$PLAN" --attempt "$RUN_ROOT/p3_submission_attempt.json" --receipt "$RUN_ROOT/submission_receipt.tsv"
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","mode":"submit","run_root":"%s","receipt":"%s","stage":"P3"}\n' "$RUN_ROOT" "$RUN_ROOT/submission_receipt.tsv"
