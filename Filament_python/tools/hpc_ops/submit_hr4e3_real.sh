#!/usr/bin/env bash
# Submit the independent three-screen x four-domain E3-B confirmation block.
set -euo pipefail
readonly REPO="$1" RUN_ROOT="$2" EXPECTED_SHA="$3" PREFLIGHT="$4"
readonly PROVENANCE="$5"
readonly BATCH="$REPO/Filament_python/tools/hr4e3_domain_case.sbatch"
readonly PYTHON="/data/home/scvi806/.conda/envs/Filament_python/bin/python"
readonly SOURCE_MANIFEST="/data/run01/scvi806/user_Wangjimin/projects/hr4e1_runs_e1b_be280dc/post_reference/post_reference_manifest.json"
readonly SOURCE_ROOT="/data/run01/scvi806/user_Wangjimin/projects/hr4e1_runs_e1b_be280dc/post_reference"
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
assert all(item["edge_validity"]["status"] == "PASS" for item in report["real_post"])
PY
readonly -a SCREEN_IDS=(front peak rear)
readonly -a SCREEN_INDEXES=(7827 8022 10338)
readonly -a SCREEN_Z_M=(0.78274999999993 0.802249999999928 0.941924999999913)
mkdir -m 700 -- "$RUN_ROOT"
receipt="$RUN_ROOT/submission_receipt.tsv"
printf 'case_id\tscreen_id\tscreen_index\tscreen_z_m\tdomain\tjob_id\n' > "$receipt"
for screen_pos in "${!SCREEN_IDS[@]}"; do
    screen_id="${SCREEN_IDS[$screen_pos]}"; screen_index="${SCREEN_INDEXES[$screen_pos]}"; screen_z_m="${SCREEN_Z_M[$screen_pos]}"
    screen_path="$SOURCE_ROOT/screen_${screen_id}_delta_n.npy"
    test -f "$screen_path"
    for domain in D0 D1 D2 D3; do
        case_id="E3B_${screen_id}_${domain}_dx10um_dt1p0us"
        case_dir="$RUN_ROOT/$case_id"
        mkdir -m 700 -- "$case_dir"
        submission="$(sbatch --parsable --job-name="$case_id" --output="$case_dir/slurm-%j.out" --error="$case_dir/slurm-%j.err" --export="ALL,EXPECTED_GIT_SHA=$EXPECTED_SHA,REPO_DIR=$REPO,CASE_DIR=$case_dir,CASE_ID=$case_id,KIND=real_post,DOMAIN_ID=$domain,SCREEN_PATH=$screen_path,SOURCE_MANIFEST=$SOURCE_MANIFEST,SCREEN_ID=$screen_id,SCREEN_INDEX=$screen_index,SCREEN_Z_M=$screen_z_m" "$BATCH")"
        job_id="${submission%%;*}"
        [[ "$job_id" =~ ^[0-9]+$ ]]
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$case_id" "$screen_id" "$screen_index" "$screen_z_m" "$domain" "$job_id" >> "$receipt"
    done
done
printf '{"schema":"filament.hpc_ops.write_receipt.v1","ok":true,"state":"completed","run_root":"%s","receipt":"%s"}\n' "$RUN_ROOT" "$receipt"
