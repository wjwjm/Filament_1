#!/usr/bin/env bash
# Submit exactly one paired reference/hybrid allocation for the fixed 0.60 m campaign.
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 REPO_DIR RUN_DIR EXECUTION_LOCK_PATH PROVENANCE_V2_PATH" >&2
  exit 2
fi

readonly REPO_DIR_INPUT="$1"
readonly RUN_DIR_INPUT="$2"
readonly EXECUTION_LOCK_INPUT="$3"
readonly PROVENANCE_V2_INPUT="$4"
readonly CAMPAIGN_ID="hybrid_propagation_validation_0p60"
readonly REMOTE_ROOT="/data/run01/scvi806/user_Wangjimin/hybrid_propagation_validation_0p60"
readonly EXPECTED_GPU_MODEL="NVIDIA GeForce RTX 5090"
readonly EXPECTED_NODELIST="m4gn1401"
readonly MANIFEST_REL="Filament_python/results/hybrid_propagation_validation/submission_manifest.json"
readonly BATCH_REL="Filament_python/tools/hybrid_propagation_validation.sbatch"

REPO_DIR="$(cd -- "${REPO_DIR_INPUT}" && pwd -P)"
RUN_PARENT="$(dirname -- "${RUN_DIR_INPUT}")"
RUN_NAME="$(basename -- "${RUN_DIR_INPUT}")"
RUN_PARENT="$(cd -- "${RUN_PARENT}" && pwd -P)"
RUN_DIR="${RUN_PARENT}/${RUN_NAME}"
case "${RUN_DIR}" in
  "${REMOTE_ROOT}"/*) ;;
  *) echo "FATAL: RUN_DIR must be under ${REMOTE_ROOT}" >&2; exit 10 ;;
esac
[[ ! -e "${RUN_DIR}" && ! -L "${RUN_DIR}" ]] || { echo "FATAL: RUN_DIR already exists" >&2; exit 11; }

MANIFEST_PATH="${REPO_DIR}/${MANIFEST_REL}"
BATCH_PATH="${REPO_DIR}/${BATCH_REL}"
EXECUTION_LOCK_PATH="$(cd -- "$(dirname -- "${EXECUTION_LOCK_INPUT}")" && pwd -P)/$(basename -- "${EXECUTION_LOCK_INPUT}")"
PROVENANCE_V2_PATH="$(cd -- "$(dirname -- "${PROVENANCE_V2_INPUT}")" && pwd -P)/$(basename -- "${PROVENANCE_V2_INPUT}")"
for path in "${MANIFEST_PATH}" "${BATCH_PATH}" "${EXECUTION_LOCK_PATH}" "${PROVENANCE_V2_PATH}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || { echo "FATAL: required regular file missing: ${path}" >&2; exit 12; }
done

cd "${REPO_DIR}"
EXPECTED_GIT_SHA="$(git rev-parse HEAD)"
[[ -z "$(git status --porcelain=v1)" ]] || { echo "FATAL: source checkout is dirty" >&2; exit 13; }
EXPECTED_MANIFEST_SHA256="$(sha256sum "${MANIFEST_PATH}" | awk '{print $1}')"
EXPECTED_EXECUTION_LOCK_SHA256="$(sha256sum "${EXECUTION_LOCK_PATH}" | awk '{print $1}')"
EXPECTED_PROVENANCE_V2_SHA256="$(sha256sum "${PROVENANCE_V2_PATH}" | awk '{print $1}')"

VALIDATION="$({
  python - "${MANIFEST_PATH}" "${EXECUTION_LOCK_PATH}" "${EXPECTED_GIT_SHA}" <<'PY'
import importlib.util, json, sys
from pathlib import Path
manifest_path = Path(sys.argv[1])
lock_path = Path(sys.argv[2])
expected_sha = sys.argv[3]
module_path = Path.cwd() / "Filament_python/tools/create_hybrid_propagation_execution_lock.py"
spec = importlib.util.spec_from_file_location("hybrid_lock_submit", module_path)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load execution-lock validator")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
checked = module.validate_manifest_lock(manifest_path, expected_git_sha=expected_sha, require_clean=True, require_committed=True)
lock = json.loads(lock_path.read_text(encoding="utf-8"))
if lock.get("schema") != module.LOCK_SCHEMA or lock.get("status") != "authorized_not_consumed":
    raise SystemExit("execution lock schema/status invalid")
if lock.get("expected_git_sha") != expected_sha or lock.get("manifest_sha256") != checked["manifest_sha256"]:
    raise SystemExit("execution lock HEAD/manifest binding invalid")
if lock.get("nodelist") != module.EXPECTED_NODE or lock.get("expected_node") != module.EXPECTED_NODE:
    raise SystemExit("execution lock node binding invalid")
if lock.get("lut_build_cap_inactive_required") is not module.LUT_BUILD_CAP_INACTIVE_REQUIRED:
    raise SystemExit("execution lock LUT cap-inactive binding invalid")
if module.sha256(lock_path) == "":
    raise SystemExit("execution lock hash unavailable")
print(json.dumps({"head": expected_sha, "manifest_sha256": checked["manifest_sha256"]}))
PY
} 2>&1)" || { echo "FATAL: manifest/lock validation failed" >&2; exit 14; }
python Filament_python/tools/hpc_ops/provenance_v2.py validate --repo "${REPO_DIR}" --manifest "${PROVENANCE_V2_PATH}" >/dev/null

mkdir -p -- "${REMOTE_ROOT}"
GLOBAL_CONSUMED_LOCK="${REMOTE_ROOT}/.consumed.lock"
if ! mkdir -- "${RUN_DIR}"; then
  echo "FATAL: failed to reserve RUN_DIR" >&2
  exit 15
fi
RUN_OWNER_MARKER="${RUN_DIR}/.hybrid_run_owner"
printf '%s\n' "${EXPECTED_GIT_SHA}" > "${RUN_OWNER_MARKER}"

cleanup_pre_sbatch() {
  if [[ -f "${RUN_OWNER_MARKER}" ]]; then
    rm -f -- "${RUN_OWNER_MARKER}"
    rmdir -- "${RUN_DIR}" 2>/dev/null || true
  fi
  if [[ -f "${GLOBAL_CONSUMED_LOCK}/owner" ]] && [[ "$(cat "${GLOBAL_CONSUMED_LOCK}/owner")" == "${RUN_DIR}" ]]; then
    rm -f -- "${GLOBAL_CONSUMED_LOCK}/owner"
    rmdir -- "${GLOBAL_CONSUMED_LOCK}" 2>/dev/null || true
  fi
}
trap cleanup_pre_sbatch EXIT

if ! mkdir -- "${GLOBAL_CONSUMED_LOCK}"; then
  echo "FATAL: campaign submission already consumed" >&2
  exit 16
fi
printf '%s\n' "${RUN_DIR}" > "${GLOBAL_CONSUMED_LOCK}/owner"
chmod 700 "${GLOBAL_CONSUMED_LOCK}"

SUBMISSION_LOCK="${RUN_DIR}/SUBMISSION_LOCK"
JOB_RECEIPT_PATH="${RUN_DIR}/job_receipt.json"
python - "${SUBMISSION_LOCK}" "${CAMPAIGN_ID}" "${RUN_DIR}" "${EXPECTED_GIT_SHA}" \
  "${EXPECTED_MANIFEST_SHA256}" "${EXPECTED_EXECUTION_LOCK_SHA256}" "${EXPECTED_PROVENANCE_V2_SHA256}" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "schema": "khz_filament.hybrid_propagation_validation.submission_lock.v1",
    "campaign_id": sys.argv[2], "run_dir": sys.argv[3], "expected_git_sha": sys.argv[4],
    "manifest_sha256": sys.argv[5], "execution_lock_sha256": sys.argv[6],
    "provenance_v2_sha256": sys.argv[7], "state": "reserved_before_sbatch",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
chmod 444 "${SUBMISSION_LOCK}"

trap - EXIT
SBATCH_OUTPUT="$(sbatch --hold --parsable \
  --chdir="${RUN_DIR}" --output="${RUN_DIR}/slurm-%j.out" --error="${RUN_DIR}/slurm-%j.err" \
  --export=ALL,REPO_DIR="${REPO_DIR}",RUN_DIR="${RUN_DIR}",EXPECTED_GIT_SHA="${EXPECTED_GIT_SHA}",MANIFEST_PATH="${MANIFEST_PATH}",EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256}",EXECUTION_LOCK_PATH="${EXECUTION_LOCK_PATH}",EXPECTED_EXECUTION_LOCK_SHA256="${EXPECTED_EXECUTION_LOCK_SHA256}",PROVENANCE_V2_PATH="${PROVENANCE_V2_PATH}",EXPECTED_PROVENANCE_V2_SHA256="${EXPECTED_PROVENANCE_V2_SHA256}",JOB_RECEIPT_PATH="${JOB_RECEIPT_PATH}",SUBMISSION_LOCK="${SUBMISSION_LOCK}",GLOBAL_CONSUMED_LOCK="${GLOBAL_CONSUMED_LOCK}",EXPECTED_GPU_MODEL="${EXPECTED_GPU_MODEL}",EXPECTED_NODELIST="${EXPECTED_NODELIST}",CAMPAIGN_ID="${CAMPAIGN_ID}" \
  "${BATCH_PATH}")" || {
    printf '%s\n' "sbatch invocation failed or was ambiguous; locks retained" > "${RUN_DIR}/sbatch_failure_record.txt"
    exit 17
  }
JOB_ID="${SBATCH_OUTPUT%%;*}"
[[ "${JOB_ID}" =~ ^[0-9]+$ ]] || {
  printf '%s\n' "non-numeric sbatch output; locks retained" > "${RUN_DIR}/sbatch_failure_record.txt"
  exit 18
}

python - "${JOB_RECEIPT_PATH}" "${JOB_ID}" "${CAMPAIGN_ID}" "${RUN_DIR}" "${EXPECTED_GIT_SHA}" \
  "${EXPECTED_MANIFEST_SHA256}" "${EXPECTED_EXECUTION_LOCK_SHA256}" "${EXPECTED_PROVENANCE_V2_SHA256}" <<'PY'
import json, secrets, sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "schema": "khz_filament.hybrid_propagation_validation.job_receipt.v1",
    "job_id": sys.argv[2], "campaign_id": sys.argv[3], "run_dir": sys.argv[4],
    "expected_git_sha": sys.argv[5], "manifest_sha256": sys.argv[6],
    "execution_lock_sha256": sys.argv[7], "provenance_v2_sha256": sys.argv[8],
    "reservation_token": secrets.token_hex(16), "state": "held_receipt_recorded",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
chmod 444 "${JOB_RECEIPT_PATH}"
printf '%s\n' "${JOB_ID}" > "${RUN_DIR}/slurm_job_id.txt"
chmod 444 "${RUN_DIR}/slurm_job_id.txt"
rm -f -- "${RUN_OWNER_MARKER}"
chmod 500 "${GLOBAL_CONSUMED_LOCK}"

if ! scontrol release "${JOB_ID}"; then
  printf '%s\n' "held job receipt exists but release failed; manual scheduler review required" > "${RUN_DIR}/release_failure_record.txt"
  exit 19
fi
printf '%s\n' "${JOB_ID}"
