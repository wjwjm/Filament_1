#!/usr/bin/env bash
# Submit exactly one HR-2E stage after all local/preflight gates have passed.

set -euo pipefail

FIXED_PYTHON="/data/home/scvi806/.conda/envs/Filament_python/bin/python"
REPO=""
RUN_ROOT=""
EXPECTED_HEAD=""
MANIFEST=""
PROVENANCE_MANIFEST=""
STAGE=""
EXPECTED_GPU_MODEL="NVIDIA GeForce RTX 5090"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --repo) REPO="$2"; shift 2 ;;
        --run-root) RUN_ROOT="$2"; shift 2 ;;
        --expected-head) EXPECTED_HEAD="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --provenance-manifest) PROVENANCE_MANIFEST="$2"; shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        --expected-gpu-model) EXPECTED_GPU_MODEL="$2"; shift 2 ;;
        *) echo "unknown argument" >&2; exit 64 ;;
    esac
done

[[ -n "$REPO" && -n "$RUN_ROOT" && -n "$EXPECTED_HEAD" && -n "$MANIFEST" && -n "$PROVENANCE_MANIFEST" && -n "$STAGE" ]] || exit 64
[[ "$RUN_ROOT" == /data/run01/scvi806/* ]] || { echo "unsafe run root" >&2; exit 65; }
[[ "$STAGE" == stage2 || "$STAGE" == stage3 ]] || { echo "invalid stage" >&2; exit 64; }
[[ -d "$REPO" && ! -L "$REPO" ]] || { echo "repository missing or symlinked" >&2; exit 66; }
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$EXPECTED_HEAD" ]] || { echo "HEAD mismatch" >&2; exit 67; }
[[ "$(git -C "$REPO" branch --show-current)" == "codex-HR-2" ]] || { echo "branch mismatch" >&2; exit 67; }
[[ -z "$(git -C "$REPO" status --porcelain=v1)" ]] || { echo "repository is dirty" >&2; exit 67; }
[[ -f "$MANIFEST" ]] || { echo "manifest missing" >&2; exit 68; }
[[ "$PROVENANCE_MANIFEST" == /* && -f "$PROVENANCE_MANIFEST" && ! -L "$PROVENANCE_MANIFEST" ]] || { echo "provenance manifest must be an absolute regular non-symlink file" >&2; exit 68; }
[[ ! -e "$RUN_ROOT" ]] || { echo "run root already exists" >&2; exit 69; }

# The strict provenance gate is deliberately before any run directory,
# receipt, execution manifest, case directory, or sbatch side effect.
"$FIXED_PYTHON" "$REPO/Filament_python/tools/hpc_ops/provenance_v2.py" validate \
    --repo "$REPO" --manifest "$PROVENANCE_MANIFEST" --require-hash-scope >/dev/null

# Mandatory batch-entry audit occurs before mkdir, lock, receipt, or sbatch.
"$FIXED_PYTHON" "$REPO/Filament_python/tools/hpc_ops/audit_batch_entry.py" \
    --batch "$REPO/Filament_python/tools/hr2e_schedule_convergence.sbatch" \
    --fixed-python "$FIXED_PYTHON" >/dev/null

mapfile -t CASE_ROWS < <("$FIXED_PYTHON" - "$MANIFEST" "$PROVENANCE_MANIFEST" "$STAGE" "$REPO" <<'PY'
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
provenance_path = pathlib.Path(sys.argv[2])
stage = sys.argv[3]
repo = pathlib.Path(sys.argv[4])
doc = json.loads(manifest_path.read_text(encoding="utf-8"))
if doc.get("schema") != "khz_filament.hr2e.stage1_preflight.v2":
    raise SystemExit("invalid manifest schema")
if doc.get("hash_scope") != "classified_by_record":
    raise SystemExit("planning manifest must declare classified_by_record hash_scope")
if doc.get("provenance_manifest_required") is not True:
    raise SystemExit("planning manifest must require a provenance manifest")
if doc.get("provenance_manifest_schema") != "filament.provenance.v2":
    raise SystemExit("planning manifest provenance schema is invalid")
tracked_paths = doc.get("tracked_paths")
if (
    not isinstance(tracked_paths, list)
    or not tracked_paths
    or tracked_paths != sorted(set(tracked_paths))
    or not all(isinstance(item, str) and item for item in tracked_paths)
):
    raise SystemExit("planning manifest tracked_paths is invalid")
manifest_provenance_path = doc.get("manifest_provenance_path")
if manifest_provenance_path not in tracked_paths:
    raise SystemExit("planning manifest does not provenance-bind itself")
expected_manifest = (repo / str(manifest_provenance_path)).resolve()
if manifest_path.resolve() != expected_manifest or not expected_manifest.is_file():
    raise SystemExit("planning manifest path is not canonical")
ids = doc["stage2_parallel_jobs"] if stage == "stage2" else doc["stage3_conditional_jobs"]
by_id = {case["case_id"]: case for case in doc["cases"]}
sys.path.insert(0, str(repo / "Filament_python" / "tools" / "hpc_ops"))
import provenance_v2

provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
provenance_v2.validate_manifest(repo, provenance_path, require_hash_scope=True)
for tracked_path in tracked_paths:
    provenance_v2.lookup_record(
        provenance, tracked_path,
        classification="tracked_text", require_hash_scope=True,
    )
for case_id in ids:
    case = by_id[case_id]
    config_rel = str(case["config_path"])
    config_provenance_path = str(case.get("config_provenance_path", "Filament_python/" + config_rel))
    if config_provenance_path not in tracked_paths:
        raise SystemExit(f"config is absent from planning tracked_paths: {case_id}")
    config = repo / "Filament_python" / config_rel
    if not config.is_file() or config.is_symlink() or config.resolve() != (repo / config_provenance_path).resolve():
        raise SystemExit(f"config is unavailable: {case_id}")
    record = provenance_v2.lookup_record(
        provenance, config_provenance_path,
        classification="tracked_text", require_hash_scope=True,
    )
    print("\t".join((
        case_id,
        str(config),
        config_provenance_path,
        str(record["git_blob_oid"]),
        str(record["canonical_lf_sha256"]),
    )))
PY
)
expected_count=3
[[ "$STAGE" == stage3 ]] && expected_count=2
[[ "${#CASE_ROWS[@]}" -eq "$expected_count" ]] || { echo "unexpected job count" >&2; exit 70; }

mkdir -p "$RUN_ROOT"
chmod 700 "$RUN_ROOT"
bound_provenance_manifest="$RUN_ROOT/provenance_manifest.json"
cp -- "$PROVENANCE_MANIFEST" "$bound_provenance_manifest"
chmod 600 "$bound_provenance_manifest"
receipt="$RUN_ROOT/submission_receipt.tsv"
: >"$receipt"
chmod 600 "$receipt"
printf 'case_id\tjob_id\tconfig_path\tconfig_provenance_path\tconfig_git_blob_oid\tconfig_canonical_lf_sha256\n' >>"$receipt"
"$FIXED_PYTHON" - "$RUN_ROOT/execution_manifest.json" "$MANIFEST" "$bound_provenance_manifest" "$EXPECTED_HEAD" "$STAGE" "$REPO" "${CASE_ROWS[@]}" <<'PY'
import json
import pathlib
import sys

output, manifest, provenance_manifest, expected_head, stage, repo, *rows = sys.argv[1:]
repo = pathlib.Path(repo)
sys.path.insert(0, str(repo / "Filament_python" / "tools" / "hpc_ops"))
import provenance_v2

planning = json.loads(pathlib.Path(manifest).read_text(encoding="utf-8"))
provenance = json.loads(pathlib.Path(provenance_manifest).read_text(encoding="utf-8"))
provenance_v2.validate_manifest(repo, provenance_manifest, require_hash_scope=True)
tracked_records = [
    dict(provenance_v2.lookup_record(
        provenance,
        tracked_path,
        classification="tracked_text",
        require_hash_scope=True,
    ))
    for tracked_path in planning["tracked_paths"]
]
preflight_record = provenance_v2.lookup_record(
    provenance,
    planning["manifest_provenance_path"],
    classification="tracked_text",
    require_hash_scope=True,
)

case_records = []
for row in rows:
    case_id, config_path, config_provenance_path, blob, canonical = row.split("\t")
    case_records.append({
        "case_id": case_id,
        "path": config_provenance_path,
        "classification": "tracked_text",
        "hash_scope": "git_blob_oid+canonical_lf_sha256",
        "git_blob_oid": blob,
        "canonical_lf_sha256": canonical,
    })
payload = {
    "schema": "khz_filament.hr2e.execution_manifest.v2",
    "hash_scope": "classified_by_record",
    "stage": stage,
    "expected_git_sha": expected_head,
    "preflight_manifest_path": manifest,
    "preflight_manifest_record": dict(preflight_record),
    "provenance_manifest_path": provenance_manifest,
    "provenance_manifest_sha256": provenance_v2.raw_sha256_file(pathlib.Path(provenance_manifest)),
    "provenance_manifest_hash_scope": "raw_bytes",
    "records": [
        {
            "path": provenance_manifest,
            "classification": "external",
            "hash_scope": "raw_bytes",
            "raw_sha256": provenance_v2.raw_sha256_file(pathlib.Path(provenance_manifest)),
        },
        *tracked_records,
    ],
    "config_records": case_records,
    "case_ids": [record["case_id"] for record in case_records],
}
pathlib.Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
chmod 600 "$RUN_ROOT/execution_manifest.json"

for row in "${CASE_ROWS[@]}"; do
    IFS=$'\t' read -r case_id config_path config_provenance_path config_blob config_canonical <<<"$row"
    case_dir="$RUN_ROOT/$case_id"
    mkdir -p "$case_dir"
    job_id="$(sbatch --parsable \
        --export=ALL,EXPECTED_GIT_SHA="$EXPECTED_HEAD",REPO_DIR="$REPO",RUN_DIR="$case_dir",CASE_ID="$case_id",CONFIG_PATH="$config_path",PROVENANCE_MANIFEST="$bound_provenance_manifest",CONFIG_PROVENANCE_PATH="$config_provenance_path",CONFIG_GIT_BLOB_OID="$config_blob",CONFIG_CANONICAL_LF_SHA256="$config_canonical",EXPECTED_GPU_MODEL="$EXPECTED_GPU_MODEL" \
        "$REPO/Filament_python/tools/hr2e_schedule_convergence.sbatch")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$case_id" "$job_id" "$config_path" "$config_provenance_path" "$config_blob" "$config_canonical" >>"$receipt"
done
