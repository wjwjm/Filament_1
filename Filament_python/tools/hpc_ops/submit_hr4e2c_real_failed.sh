#!/usr/bin/env bash
# Submit only the five E2-C cases that failed before application startup.
set -euo pipefail

readonly REPO="$1" RUN_ROOT="$2" EXPECTED_SHA="$3" PREFLIGHT="$4"
readonly BATCH="$REPO/Filament_python/tools/hr4e2c_real_case.sbatch"
readonly PYTHON="/data/home/scvi806/.conda/envs/Filament_python/bin/python"
readonly SOURCE_MANIFEST="/data/run01/scvi806/user_Wangjimin/projects/hr4e1_runs_e1b_be280dc/post_reference/post_reference_manifest.json"
readonly SOURCE_ROOT="/data/run01/scvi806/user_Wangjimin/projects/hr4e1_runs_e1b_be280dc/post_reference"
readonly DT_US="0.125"

test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test -x "$PYTHON" && test -f "$BATCH" && test -f "$PREFLIGHT"
"$PYTHON" - "$PREFLIGHT" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
assert report["status"] == "PASS"
assert report["scope_is_hydro_only_validation"] is True
assert report["production_multigrid_mapping_modified"] is False
assert report["full_chain_transverse_convergence_claimed"] is False
PY
test ! -e "$RUN_ROOT"

readonly -a SCREEN_IDS=(peak peak rear rear rear)
readonly -a SCREEN_INDEXES=(8022 8022 10338 10338 10338)
readonly -a SCREEN_Z_M=(0.802249999999928 0.802249999999928 0.941924999999913 0.941924999999913 0.941924999999913)
readonly -a DX_UM=(10 5 20 10 5)

mkdir -m 700 -- "$RUN_ROOT"
receipt="$RUN_ROOT/submission_receipt.tsv"
printf 'case_id\tscreen_id\tscreen_index\tscreen_z_m\tdx_um\tdt_us\tjob_id\n' > "$receipt"
for pos in "${!SCREEN_IDS[@]}"; do
    screen_id="${SCREEN_IDS[$pos]}"
    screen_index="${SCREEN_INDEXES[$pos]}"
    screen_z_m="${SCREEN_Z_M[$pos]}"
    dx="${DX_UM[$pos]}"
    screen_path="$SOURCE_ROOT/screen_${screen_id}_delta_n.npy"
    test -f "$screen_path"
    case_id="E2C_${screen_id}_dx${dx}um_dt0p125us"
    case_dir="$RUN_ROOT/$case_id"
    mkdir -m 700 -- "$case_dir"
    submission="$(sbatch --parsable --job-name="$case_id" --output="$case_dir/slurm-%j.out" --error="$case_dir/slurm-%j.err" --export="ALL,EXPECTED_GIT_SHA=$EXPECTED_SHA,REPO_DIR=$REPO,CASE_DIR=$case_dir,CASE_ID=$case_id,SCREEN_PATH=$screen_path,SOURCE_MANIFEST=$SOURCE_MANIFEST,SCREEN_ID=$screen_id,SCREEN_INDEX=$screen_index,SCREEN_Z_M=$screen_z_m,DX_UM=$dx,DT_US=$DT_US" "$BATCH")"
    job_id="${submission%%;*}"
    [[ "$job_id" =~ ^[0-9]+$ ]]
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$case_id" "$screen_id" "$screen_index" "$screen_z_m" "$dx" "$DT_US" "$job_id" >> "$receipt"
done
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","run_root":"%s","receipt":"%s"}\n' "$RUN_ROOT" "$receipt"
