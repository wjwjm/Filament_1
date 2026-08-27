#!/usr/bin/env bash
# Submit exactly one HR-2E stage after all local/preflight gates have passed.

set -euo pipefail

FIXED_PYTHON="/data/home/scvi806/.conda/envs/Filament_python/bin/python"
REPO=""
RUN_ROOT=""
EXPECTED_HEAD=""
MANIFEST=""
STAGE=""
EXPECTED_GPU_MODEL="NVIDIA GeForce RTX 5090"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --repo) REPO="$2"; shift 2 ;;
        --run-root) RUN_ROOT="$2"; shift 2 ;;
        --expected-head) EXPECTED_HEAD="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        --expected-gpu-model) EXPECTED_GPU_MODEL="$2"; shift 2 ;;
        *) echo "unknown argument" >&2; exit 64 ;;
    esac
done

[[ -n "$REPO" && -n "$RUN_ROOT" && -n "$EXPECTED_HEAD" && -n "$MANIFEST" && -n "$STAGE" ]] || exit 64
[[ "$RUN_ROOT" == /data/run01/scvi806/* ]] || { echo "unsafe run root" >&2; exit 65; }
[[ "$STAGE" == stage2 || "$STAGE" == stage3 ]] || { echo "invalid stage" >&2; exit 64; }
[[ -d "$REPO" && ! -L "$REPO" ]] || { echo "repository missing or symlinked" >&2; exit 66; }
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$EXPECTED_HEAD" ]] || { echo "HEAD mismatch" >&2; exit 67; }
[[ "$(git -C "$REPO" branch --show-current)" == "codex-HR-2" ]] || { echo "branch mismatch" >&2; exit 67; }
[[ -z "$(git -C "$REPO" status --porcelain=v1)" ]] || { echo "repository is dirty" >&2; exit 67; }
[[ -f "$MANIFEST" ]] || { echo "manifest missing" >&2; exit 68; }
[[ ! -e "$RUN_ROOT" ]] || { echo "run root already exists" >&2; exit 69; }

# Mandatory batch-entry audit occurs before mkdir, lock, receipt, or sbatch.
"$FIXED_PYTHON" "$REPO/Filament_python/tools/hpc_ops/audit_batch_entry.py" \
    --batch "$REPO/Filament_python/tools/hr2e_schedule_convergence.sbatch" \
    --fixed-python "$FIXED_PYTHON" >/dev/null

mapfile -t CASE_ROWS < <("$FIXED_PYTHON" - "$MANIFEST" "$STAGE" "$REPO" <<'PY'
import hashlib, json, pathlib, sys
manifest_path, stage, repo = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
doc = json.loads(manifest_path.read_text(encoding="utf-8"))
if doc.get("schema") != "khz_filament.hr2e.stage1_preflight.v1":
    raise SystemExit("invalid manifest schema")
ids = doc["stage2_parallel_jobs"] if stage == "stage2" else doc["stage3_conditional_jobs"]
by_id = {case["case_id"]: case for case in doc["cases"]}
for case_id in ids:
    case = by_id[case_id]
    config = repo / "Filament_python" / case["config_path"]
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    if digest != case["config_sha256"]:
        raise SystemExit(f"config hash mismatch: {case_id}")
    print("\t".join((case_id, str(config), digest)))
PY
)
expected_count=3
[[ "$STAGE" == stage3 ]] && expected_count=2
[[ "${#CASE_ROWS[@]}" -eq "$expected_count" ]] || { echo "unexpected job count" >&2; exit 70; }

mkdir -p "$RUN_ROOT"
chmod 700 "$RUN_ROOT"
receipt="$RUN_ROOT/submission_receipt.tsv"
: >"$receipt"
chmod 600 "$receipt"
printf 'case_id\tjob_id\tconfig_path\tconfig_sha256\n' >>"$receipt"
"$FIXED_PYTHON" - "$RUN_ROOT/execution_manifest.json" "$MANIFEST" "$EXPECTED_HEAD" "$STAGE" "${CASE_ROWS[@]}" <<'PY'
import hashlib, json, pathlib, sys
output, manifest, expected_head, stage, *rows = sys.argv[1:]
case_ids = [row.split("\t", 1)[0] for row in rows]
payload = {
    "schema": "khz_filament.hr2e.execution_manifest.v1",
    "stage": stage,
    "expected_git_sha": expected_head,
    "preflight_manifest_path": manifest,
    "preflight_manifest_sha256": hashlib.sha256(pathlib.Path(manifest).read_bytes()).hexdigest(),
    "case_ids": case_ids,
}
pathlib.Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
chmod 600 "$RUN_ROOT/execution_manifest.json"

for row in "${CASE_ROWS[@]}"; do
    IFS=$'\t' read -r case_id config_path config_sha <<<"$row"
    case_dir="$RUN_ROOT/$case_id"
    mkdir -p "$case_dir"
    job_id="$(sbatch --parsable \
        --export=ALL,EXPECTED_GIT_SHA="$EXPECTED_HEAD",REPO_DIR="$REPO",RUN_DIR="$case_dir",CASE_ID="$case_id",CONFIG_PATH="$config_path",EXPECTED_CONFIG_SHA256="$config_sha",EXPECTED_GPU_MODEL="$EXPECTED_GPU_MODEL" \
        "$REPO/Filament_python/tools/hr2e_schedule_convergence.sbatch")"
    printf '%s\t%s\t%s\t%s\n' "$case_id" "$job_id" "$config_path" "$config_sha" >>"$receipt"
done
