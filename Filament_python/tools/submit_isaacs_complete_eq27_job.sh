#!/usr/bin/env bash
# Submit exactly one complete Eq.27 120 fs job from an isolated run directory.
# The manifest and the campaign lock are the authorization boundary: a caller
# cannot select a different configuration, campaign root, or second RUN_DIR.
set -euo pipefail

: "${REPO_DIR:?missing REPO_DIR}"
: "${RUN_DIR:?missing RUN_DIR}"
: "${EXPECTED_GIT_SHA:?missing EXPECTED_GIT_SHA}"
: "${MANIFEST_PATH:?missing MANIFEST_PATH}"
: "${EXPECTED_MANIFEST_SHA256:?missing EXPECTED_MANIFEST_SHA256}"
: "${EXECUTION_LOCK_PATH:?missing EXECUTION_LOCK_PATH}"
: "${EXPECTED_EXECUTION_LOCK_SHA256:?missing EXPECTED_EXECUTION_LOCK_SHA256}"
: "${EXPECTED_GPU_MODEL:?missing EXPECTED_GPU_MODEL}"
: "${STAGING_PROVENANCE_PATH:?missing STAGING_PROVENANCE_PATH}"
: "${EXPECTED_STAGING_PROVENANCE_SHA256:?missing EXPECTED_STAGING_PROVENANCE_SHA256}"

readonly FIXED_REMOTE_CAMPAIGN_ROOT="/data/run01/scvi806/user_Wangjimin/isaacs_complete_eq27_c2"
readonly FIXED_CAMPAIGN_ID="isaacs_complete_eq27_c2"
readonly GLOBAL_CONSUMED_LOCK="${FIXED_REMOTE_CAMPAIGN_ROOT}/.consumed.lock"

# A caller-supplied campaign root is never authoritative.  Reject an explicit
# conflicting value so it cannot be mistaken for a supported override.
if [[ -n "${REMOTE_CAMPAIGN_ROOT:-}" && "${REMOTE_CAMPAIGN_ROOT}" != "${FIXED_REMOTE_CAMPAIGN_ROOT}" ]]; then
  echo "FATAL: remote campaign root is fixed to ${FIXED_REMOTE_CAMPAIGN_ROOT}" >&2
  exit 29
fi
REMOTE_CAMPAIGN_ROOT="${FIXED_REMOTE_CAMPAIGN_ROOT}"

CASE_ID="${CASE_ID:-complete_eq27}"
if [[ "${CASE_ID}" != "complete_eq27" ]]; then
  echo "FATAL: only CASE_ID=complete_eq27 is permitted" >&2
  exit 20
fi

REPO_DIR="$(cd -- "${REPO_DIR}" && pwd -P)" || {
  echo "FATAL: cannot resolve REPO_DIR: ${REPO_DIR}" >&2
  exit 2
}
[[ -d "${REPO_DIR}/.git" ]] || {
  echo "FATAL: REPO_DIR is not a Git worktree: ${REPO_DIR}" >&2
  exit 2
}

# Normalize the manifest path; relative paths are interpreted from
# REPO_DIR/Filament_python unless explicitly prefixed with Filament_python/.
if [[ "${MANIFEST_PATH}" != /* ]]; then
  case "${MANIFEST_PATH}" in
    Filament_python/*) MANIFEST_PATH="${REPO_DIR}/${MANIFEST_PATH}" ;;
    *) MANIFEST_PATH="${REPO_DIR}/Filament_python/${MANIFEST_PATH}" ;;
  esac
fi
MANIFEST_DIR="$(dirname -- "${MANIFEST_PATH}")"
MANIFEST_NAME="$(basename -- "${MANIFEST_PATH}")"
MANIFEST_DIR="$(cd -- "${MANIFEST_DIR}" && pwd -P)" || {
  echo "FATAL: cannot resolve MANIFEST_PATH: ${MANIFEST_PATH}" >&2
  exit 2
}
MANIFEST_PATH="${MANIFEST_DIR}/${MANIFEST_NAME}"
[[ -f "${MANIFEST_PATH}" ]] || {
  echo "FATAL: manifest not found: ${MANIFEST_PATH}" >&2
  exit 26
}
case "${MANIFEST_PATH}" in
  "${REPO_DIR}/Filament_python"|"${REPO_DIR}/Filament_python"/*) ;;
  *) echo "FATAL: MANIFEST_PATH must be inside REPO_DIR/Filament_python" >&2; exit 26 ;;
esac

actual_manifest_sha256="$(sha256sum "${MANIFEST_PATH}" | awk '{print $1}')"
if [[ "${actual_manifest_sha256}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
  echo "FATAL: manifest SHA mismatch expected=${EXPECTED_MANIFEST_SHA256} actual=${actual_manifest_sha256}" >&2
  exit 23
fi

# The execution lock is deliberately allowed outside REPO_DIR.  It is created
# only after the final source commit and carries the non-self-referential Git
# SHA used by this submission.
if [[ "${EXECUTION_LOCK_PATH}" != /* ]]; then
  EXECUTION_LOCK_DIR="$(dirname -- "${EXECUTION_LOCK_PATH}")"
  EXECUTION_LOCK_NAME="$(basename -- "${EXECUTION_LOCK_PATH}")"
  EXECUTION_LOCK_DIR="$(cd -- "${EXECUTION_LOCK_DIR}" && pwd -P)" || {
    echo "FATAL: cannot resolve EXECUTION_LOCK_PATH: ${EXECUTION_LOCK_PATH}" >&2
    exit 2
  }
  EXECUTION_LOCK_PATH="${EXECUTION_LOCK_DIR}/${EXECUTION_LOCK_NAME}"
fi
[[ -f "${EXECUTION_LOCK_PATH}" ]] || {
  echo "FATAL: execution lock not found: ${EXECUTION_LOCK_PATH}" >&2
  exit 26
}
actual_execution_lock_sha256="$(sha256sum "${EXECUTION_LOCK_PATH}" | awk '{print $1}')"
if [[ "${actual_execution_lock_sha256}" != "${EXPECTED_EXECUTION_LOCK_SHA256}" ]]; then
  echo "FATAL: execution lock SHA mismatch expected=${EXPECTED_EXECUTION_LOCK_SHA256} actual=${actual_execution_lock_sha256}" >&2
  exit 23
fi

# The verified-bundle provenance is an external, read-only input.  It is
# validated before any RUN_DIR/campaign-lock side effect and is intentionally
# not copied into the manifest or submission/global records.
if [[ "${STAGING_PROVENANCE_PATH}" != /* ]]; then
  STAGING_PROVENANCE_DIR="$(dirname -- "${STAGING_PROVENANCE_PATH}")"
  STAGING_PROVENANCE_NAME="$(basename -- "${STAGING_PROVENANCE_PATH}")"
  STAGING_PROVENANCE_DIR="$(cd -- "${STAGING_PROVENANCE_DIR}" && pwd -P)" || {
    echo "FATAL: cannot resolve STAGING_PROVENANCE_PATH: ${STAGING_PROVENANCE_PATH}" >&2
    exit 2
  }
  STAGING_PROVENANCE_PATH="${STAGING_PROVENANCE_DIR}/${STAGING_PROVENANCE_NAME}"
fi
[[ -f "${STAGING_PROVENANCE_PATH}" ]] || {
  echo "FATAL: staging provenance not found: ${STAGING_PROVENANCE_PATH}" >&2
  exit 26
}
actual_staging_provenance_sha256="$(sha256sum "${STAGING_PROVENANCE_PATH}" | awk '{print $1}')"
if [[ "${actual_staging_provenance_sha256}" != "${EXPECTED_STAGING_PROVENANCE_SHA256}" ]]; then
  echo "FATAL: staging provenance SHA mismatch expected=${EXPECTED_STAGING_PROVENANCE_SHA256} actual=${actual_staging_provenance_sha256}" >&2
  exit 23
fi
staging_validation="$(
  REPO_DIR="${REPO_DIR}" EXPECTED_GIT_SHA="${EXPECTED_GIT_SHA}" \
  EXECUTION_LOCK_PATH="${EXECUTION_LOCK_PATH}" \
  EXPECTED_STAGING_PROVENANCE_SHA256="${actual_staging_provenance_sha256}" \
  python3 - "${STAGING_PROVENANCE_PATH}" <<'PY'
import json
import os
import sys
from pathlib import Path

repo = Path(os.environ["REPO_DIR"]).resolve()
sys.path.insert(0, str(repo / "Filament_python" / "tools"))
from create_isaacs_complete_eq27_execution_lock import validate_staging_provenance

lock = json.loads(Path(os.environ["EXECUTION_LOCK_PATH"]).read_text(encoding="utf-8"))
if not isinstance(lock, dict) or lock.get("expected_git_sha") != os.environ["EXPECTED_GIT_SHA"]:
    raise SystemExit("staging provenance execution SHA does not match the execution lock")
binding = validate_staging_provenance(
    Path(sys.argv[1]),
    expected_sha256=os.environ["EXPECTED_STAGING_PROVENANCE_SHA256"],
    expected_git_sha=lock["expected_git_sha"],
    repo=repo,
)
print(f"{binding['method']}\t{binding['source_class']}\t{binding['branch']}")
PY
)" || {
  echo "FATAL: verified-bundle staging provenance preflight failed" >&2
  exit 2
}
IFS=$'\t' read -r STAGING_PROVENANCE_METHOD STAGING_PROVENANCE_SOURCE_CLASS STAGING_PROVENANCE_BRANCH <<< "${staging_validation}"
if [[ -z "${STAGING_PROVENANCE_METHOD}" || -z "${STAGING_PROVENANCE_SOURCE_CLASS}" || -z "${STAGING_PROVENANCE_BRANCH}" ]]; then
  echo "FATAL: staging provenance preflight returned incomplete binding" >&2
  exit 2
fi

# Re-run the shared side-effect-free validator before reserving RUN_DIR or the
# campaign lock.  This includes the real clean HEAD, C1 ancestry/artifacts,
# fixed resources/config delta, GPU, operator mode, and use_raman_full_operator
# checks; the older inline checks below remain a compatibility guard.
strict_manifest_validation="$(
  REPO_DIR="${REPO_DIR}" EXPECTED_MANIFEST_SHA256="${actual_manifest_sha256}" \
  EXPECTED_EXECUTION_LOCK_SHA256="${EXPECTED_EXECUTION_LOCK_SHA256}" \
  EXPECTED_GIT_SHA="${EXPECTED_GIT_SHA}" python3 - "${MANIFEST_PATH}" "${EXECUTION_LOCK_PATH}" <<'PY'
import os
import sys
from pathlib import Path

repo = Path(os.environ["REPO_DIR"]).resolve()
sys.path.insert(0, str(repo / "Filament_python" / "tools"))
from create_isaacs_complete_eq27_execution_lock import validate_manifest_lock

binding = validate_manifest_lock(
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    expected_manifest_sha256=os.environ["EXPECTED_MANIFEST_SHA256"],
    expected_lock_sha256=os.environ["EXPECTED_EXECUTION_LOCK_SHA256"],
    expected_git_sha=os.environ["EXPECTED_GIT_SHA"],
    require_clean=True,
    require_committed_manifest=True,
)
print(f"{binding['config_path']}\t{binding['config_sha256']}")
PY
)" || {
  echo "FATAL: shared manifest/execution-lock preflight failed" >&2
  exit 2
}
IFS=$'\t' read -r STRICT_CONFIG_PATH STRICT_CONFIG_SHA256 <<< "${strict_manifest_validation}"
if [[ -z "${STRICT_CONFIG_PATH}" || -z "${STRICT_CONFIG_SHA256}" ]]; then
  echo "FATAL: shared manifest/execution-lock preflight returned no config binding" >&2
  exit 2
fi
if [[ -n "${CONFIG_PATH:-}" && "${CONFIG_PATH}" != "${STRICT_CONFIG_PATH}" && "${REPO_DIR}/${CONFIG_PATH}" != "${STRICT_CONFIG_PATH}" ]]; then
  echo "FATAL: caller CONFIG_PATH does not match shared manifest binding" >&2
  exit 26
fi
if [[ -n "${EXPECTED_CONFIG_SHA256:-}" && "${EXPECTED_CONFIG_SHA256}" != "${STRICT_CONFIG_SHA256}" ]]; then
  echo "FATAL: caller EXPECTED_CONFIG_SHA256 does not match shared manifest binding" >&2
  exit 23
fi

# Python is the single parser/validator for the manifest.  It binds the
# expected Git target to the actual source HEAD and makes the manifest-derived
# config path/hash authoritative over optional legacy caller variables.
manifest_validation="$(
  REPO_DIR="${REPO_DIR}" \
  EXPECTED_GIT_SHA="${EXPECTED_GIT_SHA}" \
  MANIFEST_CONFIG_PATH="${CONFIG_PATH:-}" \
  MANIFEST_CONFIG_SHA256="${EXPECTED_CONFIG_SHA256:-}" \
  EXECUTION_LOCK_PATH="${EXECUTION_LOCK_PATH}" \
  FIXED_REMOTE_CAMPAIGN_ROOT="${FIXED_REMOTE_CAMPAIGN_ROOT}" \
  FIXED_CAMPAIGN_ID="${FIXED_CAMPAIGN_ID}" \
  EXPECTED_GPU_MODEL="${EXPECTED_GPU_MODEL}" \
  EXPECTED_MANIFEST_SHA256="${actual_manifest_sha256}" \
  python3 - "${MANIFEST_PATH}" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.environ["REPO_DIR"]) / "Filament_python" / "tools"))
try:
    from prepare_isaacs_complete_eq27_job import _assert_fixed, config_diff
except (ImportError, OSError) as exc:
    print(f"FATAL: cannot load fixed C2 configuration assertions: {exc}", file=sys.stderr)
    raise SystemExit(2)


def fail(message):
    print(f"FATAL: manifest validation failed: {message}", file=sys.stderr)
    raise SystemExit(2)


path = Path(sys.argv[1])
try:
    manifest = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    fail(f"cannot parse {path}: {exc}")
if not isinstance(manifest, dict):
    fail("top-level value is not an object")

repo = Path(os.environ["REPO_DIR"]).resolve()
filament_root = (repo / "Filament_python").resolve()
if manifest.get("expected_git_sha") is not None:
    fail("manifest expected_git_sha must be null; final SHA comes from execution lock")
if manifest.get("execution_lock_required") is not True:
    fail("manifest execution_lock_required must be true")
if manifest.get("expected_git_sha_resolution") != "external execution_lock generated after final source commit":
    fail("manifest expected_git_sha_resolution is incorrect")

lock_path = Path(os.environ["EXECUTION_LOCK_PATH"]).resolve()
try:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    fail(f"cannot parse execution lock {lock_path}: {exc}")
if not isinstance(lock, dict):
    fail("execution lock top-level value is not an object")
if lock.get("schema") != "khz_filament.isaacs_complete_eq27.c2_execution_lock.v1":
    fail("execution lock schema is invalid")
if lock.get("campaign_id") != os.environ["FIXED_CAMPAIGN_ID"]:
    fail("execution lock campaign_id is not the fixed campaign id")
if lock.get("remote_campaign_root") != os.environ["FIXED_REMOTE_CAMPAIGN_ROOT"]:
    fail("execution lock remote_campaign_root is not the fixed campaign root")
if lock.get("status") != "authorized_not_consumed":
    fail("execution lock status is not authorized_not_consumed")
if lock.get("operator_mode") != "full_isaacs_eq27_complete":
    fail("execution lock operator_mode is not complete Eq.27")
if lock.get("use_raman_full_operator") is not True:
    fail("execution lock requires use_raman_full_operator=true")
if lock.get("expected_gpu_model") != os.environ["EXPECTED_GPU_MODEL"]:
    fail("execution lock expected_gpu_model does not match EXPECTED_GPU_MODEL")
lock_sha = lock.get("expected_git_sha")
env_expected = os.environ.get("EXPECTED_GIT_SHA", "")
if not isinstance(lock_sha, str) or not lock_sha.strip():
    fail("execution lock expected_git_sha is null or empty")
if lock_sha != env_expected:
    fail("execution lock expected_git_sha does not match EXPECTED_GIT_SHA")
try:
    actual = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
except (OSError, subprocess.CalledProcessError) as exc:
    fail(f"cannot resolve actual source Git SHA: {exc}")
if lock_sha != actual.strip():
    fail(f"execution lock expected_git_sha does not match actual source HEAD {actual.strip()}")
try:
    clean = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain=v1"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
except (OSError, subprocess.CalledProcessError) as exc:
    fail(f"cannot verify clean source worktree: {exc}")
if clean:
    fail("source worktree is not clean")

manifest_rel = path.resolve().relative_to(repo).as_posix()
if lock.get("manifest_path") != manifest_rel:
    fail("execution lock manifest_path does not match MANIFEST_PATH")
if lock.get("manifest_sha256") != os.environ["EXPECTED_MANIFEST_SHA256"]:
    fail("execution lock manifest_sha256 does not match actual manifest hash")

if manifest.get("remote_campaign_root") != os.environ["FIXED_REMOTE_CAMPAIGN_ROOT"]:
    fail("remote_campaign_root is not the fixed campaign root")
if manifest.get("campaign_id") != os.environ["FIXED_CAMPAIGN_ID"]:
    fail("campaign_id is not the fixed campaign id")
if manifest.get("status") != "prepared_not_submitted":
    fail("status is not prepared_not_submitted")
for key, expected_count in {
    "jobs_authorized": 1,
    "jobs_submitted": 0,
    "full_jobs_authorized": 1,
    "full_propagation_jobs_authorized": 1,
    "full_production_jobs_submitted": 0,
    "scan_jobs_authorized": 0,
    "profiling_jobs_authorized": 0,
    "optimization_jobs_authorized": 0,
}.items():
    if manifest.get(key) != expected_count:
        fail(f"{key} must equal {expected_count!r}")

source_rel = manifest.get("source_config")
if source_rel != "configs/isaacs_raman_closure/120fs_talebpour_isaacs_full_operator_on.json":
    fail("source_config does not match the locked C2 source path")
if manifest.get("locked_base_config_sha256") != "942adca964f50b689fa5985c9af46f294da7948646b246c39ca0d50238a1b02a":
    fail("locked_base_config_sha256 is incorrect")
source = (filament_root / Path(source_rel)).resolve()
if not source.is_file():
    fail("locked source config does not exist")
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
if source_sha != manifest.get("source_config_sha256") or source_sha != manifest.get("locked_base_config_sha256"):
    fail("locked source config SHA does not match manifest and fixed baseline")
if lock.get("source_config_path") != source_rel or lock.get("source_config_sha256") != source_sha:
    fail("execution lock source config binding does not match the locked source")

derived = manifest.get("derived_config")
if not isinstance(derived, str) or not derived or Path(derived).is_absolute():
    fail("derived_config must be a non-empty relative path")
if derived != "results/isaacs_complete_eq27/120fs_talebpour_isaacs_complete_eq27.json":
    fail("derived_config does not match the locked C2 candidate path")
config = (filament_root / Path(derived)).resolve()
try:
    config.relative_to(filament_root)
except ValueError:
    fail("derived_config escapes REPO_DIR/Filament_python")
if not config.is_file():
    fail(f"derived config not found: {config}")
caller_config = os.environ.get("MANIFEST_CONFIG_PATH", "")
if caller_config:
    caller = Path(caller_config)
    if not caller.is_absolute():
        caller = repo / caller
    if caller.resolve() != config:
        fail("CONFIG_PATH does not resolve to manifest derived_config")

config_sha = manifest.get("derived_config_sha256")
if not isinstance(config_sha, str) or not config_sha:
    fail("derived_config_sha256 is null or empty")
actual_config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
if config_sha != actual_config_sha:
    fail("derived config SHA does not match derived_config_sha256")
if lock.get("config_path") != derived:
    fail("execution lock config_path does not match manifest derived_config")
if lock.get("config_sha256") != config_sha or lock.get("config_sha256") != actual_config_sha:
    fail("execution lock config_sha256 does not match manifest or actual derived config")
if lock.get("derived_config_path") != derived or lock.get("derived_config_sha256") != actual_config_sha:
    fail("execution lock derived config binding does not match manifest or actual config")
try:
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    derived_payload = json.loads(config.read_text(encoding="utf-8"))
    _assert_fixed(source_payload)
    _assert_fixed(derived_payload)
except (OSError, UnicodeError, json.JSONDecodeError, AssertionError, KeyError, TypeError) as exc:
    fail(f"fixed C2 configuration assertion failed: {exc}")
if derived_payload.get("propagation", {}).get("use_raman_full_operator") is not True:
    fail("derived config requires use_raman_full_operator=true")
expected_diff = [{
    "path": "raman.operator_mode",
    "full_isaacs_eq27": "full_isaacs_eq27",
    "full_isaacs_eq27_complete": "full_isaacs_eq27_complete",
}]
if config_diff(source_payload, derived_payload) != expected_diff:
    fail("flattened source/derived config diff is not the unique C2 operator delta")
if manifest.get("strict_config_diff") != expected_diff or lock.get("strict_config_diff") != expected_diff:
    fail("manifest/execution lock strict_config_diff is not the unique C2 operator delta")
if manifest.get("operator_modes") != {"source": "full_isaacs_eq27", "candidate": "full_isaacs_eq27_complete"}:
    fail("manifest operator_modes are not the fixed C2 modes")
if lock.get("c1_gate") != manifest.get("c1_gate"):
    fail("execution lock c1_gate does not match manifest")
if manifest.get("parent_c1_commit") != "459dd108b9873b0e8b18fe83111f386993cf5b9f":
    fail("manifest parent_c1_commit is not fixed")
c1_gate = manifest.get("c1_gate")
expected_c1_gate = {
    "commit": "459dd108b9873b0e8b18fe83111f386993cf5b9f",
    "summary_path": "results/isaacs_complete_eq27/c1_closure_summary.json",
    "summary_sha256": "ccf6f865042651894e747f1272c5371cad8bc4bb7fd6abd11b61684a795ebcdc",
    "report_path": "results/isaacs_complete_eq27/c1_operator_report.md",
    "report_sha256": "fe8b7fe99a88dde5d4c987d88d1a87dd5208461bb70ff25af6e365ef4ac7b21d",
    "overall": "PASS",
}
if c1_gate != expected_c1_gate or lock.get("c1_gate") != expected_c1_gate:
    fail("manifest/execution lock C1 gate binding is not fixed")
for rel, digest in ((expected_c1_gate["summary_path"], expected_c1_gate["summary_sha256"]), (expected_c1_gate["report_path"], expected_c1_gate["report_sha256"])):
    artifact = (filament_root / rel).resolve()
    if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
        fail(f"fixed C1 artifact is missing or has the wrong SHA256: {rel}")
summary = json.loads(((filament_root / expected_c1_gate["summary_path"]).resolve()).read_text(encoding="utf-8"))
if summary.get("overall") != "PASS":
    fail("fixed C1 machine-readable overall gate is not PASS")
resources = manifest.get("resources")
if not isinstance(resources, dict):
    fail("manifest resources are missing")
for key, expected in {
    "partition": "gpu", "gpu_count": 1, "cpu_threads": 8,
    "requested_time": "15:00:00", "expected_gpu_model": os.environ["EXPECTED_GPU_MODEL"],
}.items():
    if resources.get(key) != expected:
        fail(f"manifest resources.{key} does not match EXPECTED_GPU_MODEL/fixed C2 resources")
if lock.get("resources") != resources:
    fail("execution lock resources do not match manifest resources")
comparison = manifest.get("comparison_inputs")
if not isinstance(comparison, dict):
    fail("manifest comparison_inputs are missing")
pycap_rel = "results/density_translation_width/density_translation_width_20260715_002/paper_pycap_120fs.csv"
pycap_sha = "9b43e75ebc08ccb0a7796829e45c6727b42ab12cd661b9a3d8d235ef89d31461"
if comparison.get("pycap_120fs") != pycap_rel or comparison.get("pycap_120fs_sha256") != pycap_sha:
    fail("manifest PyCAP binding does not match the fixed input")
pycap = (filament_root / pycap_rel).resolve()
if not pycap.is_file() or hashlib.sha256(pycap.read_bytes()).hexdigest() != pycap_sha:
    fail("fixed PyCAP input is missing or has the wrong SHA256")
if lock.get("pycap_path") != pycap_rel or lock.get("pycap_sha256") != pycap_sha:
    fail("execution lock PyCAP binding does not match the fixed input")
caller_sha = os.environ.get("MANIFEST_CONFIG_SHA256", "")
if caller_sha and caller_sha != config_sha:
    fail("EXPECTED_CONFIG_SHA256 does not match manifest derived_config_sha256")
print(f"{config}\t{config_sha}")
PY
)" || {
  echo "FATAL: manifest validation could not be completed" >&2
  exit 2
}
IFS=$'\t' read -r CONFIG_PATH EXPECTED_CONFIG_SHA256 <<< "${manifest_validation}"
[[ -n "${CONFIG_PATH}" && -n "${EXPECTED_CONFIG_SHA256}" ]] || {
  echo "FATAL: manifest validator returned no derived config binding" >&2
  exit 2
}

RUN_PARENT="$(dirname -- "${RUN_DIR}")"
RUN_NAME="$(basename -- "${RUN_DIR}")"
RUN_PARENT="$(cd -- "${RUN_PARENT}" && pwd -P)" || {
  echo "FATAL: cannot resolve RUN_DIR parent: ${RUN_DIR}" >&2
  exit 2
}
RUN_DIR="${RUN_PARENT}/${RUN_NAME}"
case "${RUN_DIR}" in
  "${REPO_DIR}"|"${REPO_DIR}"/*)
    echo "FATAL: RUN_DIR must be an isolated staging directory outside REPO_DIR" >&2
    exit 25
    ;;
esac
if [[ -e "${RUN_DIR}" || -L "${RUN_DIR}" ]]; then
  echo "FATAL: RUN_DIR already exists; repeat submission is forbidden: ${RUN_DIR}" >&2
  exit 27
fi

SCRIPT="${REPO_DIR}/Filament_python/tools/isaacs_complete_eq27_full.sbatch"
[[ -f "${SCRIPT}" ]] || { echo "FATAL: batch script not found: ${SCRIPT}" >&2; exit 2; }

# Reserve a provisional, empty RUN_DIR first.  The global campaign lock is
# acquired only after this local preparation succeeds; a failed preparation
# removes only the directory and lock entries created by this invocation.
RUN_CREATED=0
GLOBAL_LOCK_CREATED=0
SBATCH_STARTED=0
OWNER_TOKEN="$(printf '%s' "$$-$(date +%s%N)" | sha256sum | awk '{print $1}')"
RUN_OWNER_MARKER="${RUN_DIR}/.c2_run_owner"
GLOBAL_LOCK_RECORD="${GLOBAL_CONSUMED_LOCK}/submission_record.txt"
SUBMISSION_LOCK="${RUN_DIR}/SUBMISSION_LOCK"
JOB_RECEIPT_PATH="${RUN_DIR}/job_receipt.json"

write_post_sbatch_failure() {
  local status="$1"
  local detail="${2:-}"
  local failure_path="${RUN_DIR}/${status}_record.txt"
  {
    printf 'status=%s\n' "${status}"
    printf 'job_id=%s\n' "${job_id:-}"
    printf 'reservation_token=%s\n' "${OWNER_TOKEN}"
    printf 'campaign_id=%s\n' "${FIXED_CAMPAIGN_ID}"
    printf 'remote_campaign_root=%s\n' "${FIXED_REMOTE_CAMPAIGN_ROOT}"
    printf 'manifest_path=%s\n' "${MANIFEST_PATH}"
    printf 'manifest_sha256=%s\n' "${actual_manifest_sha256}"
    printf 'execution_lock_path=%s\n' "${EXECUTION_LOCK_PATH}"
    printf 'execution_lock_sha256=%s\n' "${actual_execution_lock_sha256}"
    printf 'config_path=%s\n' "${CONFIG_PATH}"
    printf 'expected_config_sha256=%s\n' "${EXPECTED_CONFIG_SHA256}"
    printf 'expected_git_sha=%s\n' "${EXPECTED_GIT_SHA}"
    printf 'staging_provenance_path=%s\n' "${STAGING_PROVENANCE_PATH}"
    printf 'staging_provenance_sha256=%s\n' "${actual_staging_provenance_sha256}"
    printf 'run_dir=%s\n' "${RUN_DIR}"
    printf 'job_receipt_path=%s\n' "${JOB_RECEIPT_PATH}"
    [[ -n "${detail}" ]] && printf 'detail=%s\n' "${detail}"
  } > "${failure_path}" || true
  echo "FATAL: ${status}; campaign lock, RUN_DIR, and any held job are retained" >&2
}

cleanup_pre_sbatch() {
  local status=$?
  trap - EXIT
  if [[ "${SBATCH_STARTED}" -eq 0 && "${status}" -ne 0 ]]; then
    if [[ "${GLOBAL_LOCK_CREATED}" -eq 1 && -f "${GLOBAL_LOCK_RECORD}" ]] && grep -Fqx "reservation_token=${OWNER_TOKEN}" "${GLOBAL_LOCK_RECORD}"; then
      rm -f -- "${GLOBAL_LOCK_RECORD}" || true
      rmdir -- "${GLOBAL_CONSUMED_LOCK}" 2>/dev/null || true
    fi
    if [[ "${RUN_CREATED}" -eq 1 && -f "${RUN_OWNER_MARKER}" ]] && grep -Fqx "${OWNER_TOKEN}" "${RUN_OWNER_MARKER}"; then
      rm -f -- "${RUN_OWNER_MARKER}" "${SUBMISSION_LOCK}" 2>/dev/null || true
      rmdir -- "${RUN_DIR}" 2>/dev/null || true
    fi
  fi
  exit "${status}"
}
trap cleanup_pre_sbatch EXIT

if ! mkdir -- "${RUN_DIR}"; then
  echo "FATAL: failed to atomically reserve RUN_DIR: ${RUN_DIR}" >&2
  exit 27
fi
RUN_CREATED=1
printf '%s\n' "${OWNER_TOKEN}" > "${RUN_OWNER_MARKER}"

mkdir -p -- "${FIXED_REMOTE_CAMPAIGN_ROOT}" || {
  echo "FATAL: cannot create fixed remote campaign root: ${FIXED_REMOTE_CAMPAIGN_ROOT}" >&2
  exit 29
}
if ! mkdir -- "${GLOBAL_CONSUMED_LOCK}"; then
  echo "FATAL: global campaign submission already consumed: ${GLOBAL_CONSUMED_LOCK}" >&2
  exit 29
fi
GLOBAL_LOCK_CREATED=1
{
  printf 'campaign_id=%s\n' "${FIXED_CAMPAIGN_ID}"
  printf 'remote_campaign_root=%s\n' "${FIXED_REMOTE_CAMPAIGN_ROOT}"
  printf 'manifest_path=%s\n' "${MANIFEST_PATH}"
  printf 'manifest_sha256=%s\n' "${actual_manifest_sha256}"
  printf 'execution_lock_path=%s\n' "${EXECUTION_LOCK_PATH}"
  printf 'execution_lock_sha256=%s\n' "${actual_execution_lock_sha256}"
  printf 'expected_git_sha=%s\n' "${EXPECTED_GIT_SHA}"
  printf 'run_dir=%s\n' "${RUN_DIR}"
  printf 'reservation_token=%s\n' "${OWNER_TOKEN}"
  printf 'job_id_source=job_receipt\n'
} > "${GLOBAL_LOCK_RECORD}"

{
  printf 'case_id=%s\n' "${CASE_ID}"
  printf 'campaign_id=%s\n' "${FIXED_CAMPAIGN_ID}"
  printf 'remote_campaign_root=%s\n' "${FIXED_REMOTE_CAMPAIGN_ROOT}"
  printf 'manifest_path=%s\n' "${MANIFEST_PATH}"
  printf 'manifest_sha256=%s\n' "${actual_manifest_sha256}"
  printf 'execution_lock_path=%s\n' "${EXECUTION_LOCK_PATH}"
  printf 'execution_lock_sha256=%s\n' "${actual_execution_lock_sha256}"
  printf 'repo_dir=%s\n' "${REPO_DIR}"
  printf 'config_path=%s\n' "${CONFIG_PATH}"
  printf 'expected_config_sha256=%s\n' "${EXPECTED_CONFIG_SHA256}"
  printf 'expected_git_sha=%s\n' "${EXPECTED_GIT_SHA}"
  printf 'reservation_token=%s\n' "${OWNER_TOKEN}"
} > "${SUBMISSION_LOCK}"

export REPO_DIR RUN_DIR EXPECTED_GIT_SHA CONFIG_PATH EXPECTED_CONFIG_SHA256
export EXPECTED_GPU_MODEL CASE_ID SUBMISSION_LOCK MANIFEST_PATH EXPECTED_MANIFEST_SHA256
export CAMPAIGN_ID="${FIXED_CAMPAIGN_ID}"
export REMOTE_CAMPAIGN_ROOT="${FIXED_REMOTE_CAMPAIGN_ROOT}"
export GLOBAL_CONSUMED_LOCK EXECUTION_LOCK_PATH EXPECTED_EXECUTION_LOCK_SHA256 JOB_RECEIPT_PATH
export STAGING_PROVENANCE_PATH EXPECTED_STAGING_PROVENANCE_SHA256
export STAGING_PROVENANCE_METHOD STAGING_PROVENANCE_SOURCE_CLASS STAGING_PROVENANCE_BRANCH

# Submit held first.  The job receipt is the immutable job-id binding; the
# submission/global records are never edited after sbatch returns.
SBATCH_STARTED=1
if sbatch_output="$(sbatch --hold --parsable \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/slurm-%j.out" \
  --error="${RUN_DIR}/slurm-%j.err" \
  "${SCRIPT}")"; then
  :
else
  sbatch_status=$?
  write_post_sbatch_failure "ambiguous_sbatch_nonzero" "sbatch_exit_code=${sbatch_status}"
  exit "${sbatch_status}"
fi
if [[ "${sbatch_output}" == *$'\n'* || "${sbatch_output}" == *$'\r'* ]]; then
  write_post_sbatch_failure "ambiguous_sbatch_malformed_job_id" "sbatch_output_contains_newline=true"
  exit 28
fi
sbatch_output="${sbatch_output//$'\r'/}"
job_id="${sbatch_output%%;*}"
if [[ -z "${job_id}" || ! "${job_id}" =~ ^[0-9]+$ ]]; then
  write_post_sbatch_failure "ambiguous_sbatch_malformed_job_id" "sbatch_output=${sbatch_output}"
  exit 28
fi

if ! JOB_ID="${job_id}" RECEIPT_PATH="${JOB_RECEIPT_PATH}" \
  MANIFEST_PATH="${MANIFEST_PATH}" MANIFEST_SHA256="${actual_manifest_sha256}" \
  EXECUTION_LOCK_PATH="${EXECUTION_LOCK_PATH}" EXECUTION_LOCK_SHA256="${actual_execution_lock_sha256}" \
  CONFIG_PATH="${CONFIG_PATH}" CONFIG_SHA256="${EXPECTED_CONFIG_SHA256}" \
  EXPECTED_GIT_SHA="${EXPECTED_GIT_SHA}" CAMPAIGN_ID="${FIXED_CAMPAIGN_ID}" \
  REMOTE_CAMPAIGN_ROOT="${FIXED_REMOTE_CAMPAIGN_ROOT}" RUN_DIR="${RUN_DIR}" \
  RESERVATION_TOKEN="${OWNER_TOKEN}" STAGING_PROVENANCE_PATH="${STAGING_PROVENANCE_PATH}" \
  STAGING_PROVENANCE_SHA256="${actual_staging_provenance_sha256}" \
  STAGING_PROVENANCE_METHOD="${STAGING_PROVENANCE_METHOD}" \
  STAGING_PROVENANCE_SOURCE_CLASS="${STAGING_PROVENANCE_SOURCE_CLASS}" \
  STAGING_PROVENANCE_BRANCH="${STAGING_PROVENANCE_BRANCH}" python3 - <<'PY'
import json
import os
import stat
from pathlib import Path

path = Path(os.environ["RECEIPT_PATH"])
payload = {
    "schema": "khz_filament.isaacs_complete_eq27.job_receipt.v1",
    "state": "held",
    "job_id": os.environ["JOB_ID"],
    "reservation_token": os.environ["RESERVATION_TOKEN"],
    "campaign_id": os.environ["CAMPAIGN_ID"],
    "remote_campaign_root": os.environ["REMOTE_CAMPAIGN_ROOT"],
    "run_dir": os.path.realpath(os.environ["RUN_DIR"]),
    "manifest_path": os.path.realpath(os.environ["MANIFEST_PATH"]),
    "manifest_sha256": os.environ["MANIFEST_SHA256"],
    "execution_lock_path": os.path.realpath(os.environ["EXECUTION_LOCK_PATH"]),
    "execution_lock_sha256": os.environ["EXECUTION_LOCK_SHA256"],
    "config_path": os.path.realpath(os.environ["CONFIG_PATH"]),
    "config_sha256": os.environ["CONFIG_SHA256"],
    "expected_git_sha": os.environ["EXPECTED_GIT_SHA"],
    "staging_provenance_path": os.path.realpath(os.environ["STAGING_PROVENANCE_PATH"]),
    "staging_provenance_sha256": os.environ["STAGING_PROVENANCE_SHA256"],
    "staging_provenance_method": os.environ["STAGING_PROVENANCE_METHOD"],
    "staging_provenance_source_class": os.environ["STAGING_PROVENANCE_SOURCE_CLASS"],
    "staging_provenance_branch": os.environ["STAGING_PROVENANCE_BRANCH"],
    "method": os.environ["STAGING_PROVENANCE_METHOD"],
    "source_class": os.environ["STAGING_PROVENANCE_SOURCE_CLASS"],
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
PY
then
  write_post_sbatch_failure "ambiguous_job_receipt_failure" "receipt_write_or_readonly_chmod_failed=true"
  exit 30
fi
if [[ ! -f "${JOB_RECEIPT_PATH}" ]]; then
  write_post_sbatch_failure "ambiguous_job_receipt_failure" "receipt_missing_after_write=true"
  exit 30
fi
if ! JOB_RECEIPT_SHA256="$(sha256sum "${JOB_RECEIPT_PATH}" | awk '{print $1}')"; then
  write_post_sbatch_failure "ambiguous_job_receipt_failure" "receipt_hash_failed=true"
  exit 30
fi

if scontrol release "${job_id}"; then
  :
else
  release_status=$?
  write_post_sbatch_failure "release_failure" "scontrol_exit_code=${release_status}"
  exit "${release_status}"
fi

if ! printf '%s\n' "${job_id}" > "${RUN_DIR}/slurm_job_id.txt"; then
  write_post_sbatch_failure "ambiguous_post_sbatch_finalization" "slurm_job_id_record_failed=true"
  exit 31
fi
printf 'job_id=%s\nreceipt_path=%s\nreceipt_sha256=%s\n' "${job_id}" "${JOB_RECEIPT_PATH}" "${JOB_RECEIPT_SHA256}"
