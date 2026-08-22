#!/usr/bin/env bash
#SBATCH --job-name=isaacs-complete-eq27
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
# N50 assigns the site default host memory per requested GPU; do not override it.
#SBATCH --time=15:00:00

set -euo pipefail

: "${EXPECTED_GIT_SHA:?missing EXPECTED_GIT_SHA}"
: "${REPO_DIR:?missing REPO_DIR}"
: "${RUN_DIR:?missing RUN_DIR}"
: "${CASE_ID:?missing CASE_ID}"
: "${CONFIG_PATH:?missing CONFIG_PATH}"
: "${EXPECTED_CONFIG_SHA256:?missing EXPECTED_CONFIG_SHA256}"
: "${EXPECTED_GPU_MODEL:?missing EXPECTED_GPU_MODEL}"
: "${CAMPAIGN_ID:?missing CAMPAIGN_ID}"
: "${REMOTE_CAMPAIGN_ROOT:?missing REMOTE_CAMPAIGN_ROOT}"
: "${MANIFEST_PATH:?missing MANIFEST_PATH}"
: "${EXPECTED_MANIFEST_SHA256:?missing EXPECTED_MANIFEST_SHA256}"
: "${EXECUTION_LOCK_PATH:?missing EXECUTION_LOCK_PATH}"
: "${EXPECTED_EXECUTION_LOCK_SHA256:?missing EXPECTED_EXECUTION_LOCK_SHA256}"
: "${GLOBAL_CONSUMED_LOCK:?missing GLOBAL_CONSUMED_LOCK}"
: "${SUBMISSION_LOCK:?missing SUBMISSION_LOCK}"
: "${JOB_RECEIPT_PATH:?missing JOB_RECEIPT_PATH}"

readonly FIXED_CAMPAIGN_ID="isaacs_complete_eq27_c2"
readonly FIXED_REMOTE_CAMPAIGN_ROOT="/data/run01/scvi806/user_Wangjimin/isaacs_complete_eq27_c2"
readonly FIXED_CONFIG_REL="Filament_python/results/isaacs_complete_eq27/120fs_talebpour_isaacs_complete_eq27.json"
readonly FIXED_MANIFEST_REL="Filament_python/results/isaacs_complete_eq27/submission_manifest.json"
readonly JOB_RECEIPT_NAME="job_receipt.json"

if [[ "${CASE_ID}" != "complete_eq27" ]]; then
  echo "FATAL: only CASE_ID=complete_eq27 is permitted" >&2
  exit 20
fi
if [[ "${CAMPAIGN_ID}" != "${FIXED_CAMPAIGN_ID}" || "${REMOTE_CAMPAIGN_ROOT}" != "${FIXED_REMOTE_CAMPAIGN_ROOT}" ]]; then
  echo "FATAL: campaign binding is not the fixed C2 campaign" >&2
  exit 29
fi
if [[ "${SLURM_CPUS_PER_TASK:-8}" != "8" ]]; then
  echo "FATAL: C2 requires exactly 8 CPUs per task" >&2
  exit 24
fi

REPO_DIR="$(cd -- "${REPO_DIR}" && pwd -P)"
FIXED_CONFIG_PATH="${REPO_DIR}/${FIXED_CONFIG_REL}"
FIXED_MANIFEST_PATH="${REPO_DIR}/${FIXED_MANIFEST_REL}"
if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${REPO_DIR}/${CONFIG_PATH}"
fi
CONFIG_DIR="$(dirname -- "${CONFIG_PATH}")"
CONFIG_NAME="$(basename -- "${CONFIG_PATH}")"
CONFIG_DIR="$(cd -- "${CONFIG_DIR}" && pwd -P)"
CONFIG_PATH="${CONFIG_DIR}/${CONFIG_NAME}"
if [[ "${CONFIG_PATH}" != "${FIXED_CONFIG_PATH}" ]]; then
  echo "FATAL: CONFIG_PATH is not the fixed manifest-derived C2 config" >&2
  exit 26
fi
if [[ "${MANIFEST_PATH}" != "${FIXED_MANIFEST_PATH}" ]]; then
  echo "FATAL: MANIFEST_PATH is not the fixed C2 manifest" >&2
  exit 26
fi
if [[ "${GLOBAL_CONSUMED_LOCK}" != "${FIXED_REMOTE_CAMPAIGN_ROOT}/.consumed.lock" ]]; then
  echo "FATAL: GLOBAL_CONSUMED_LOCK is not the fixed campaign lock" >&2
  exit 29
fi
if [[ "${SUBMISSION_LOCK}" != "${RUN_DIR}/SUBMISSION_LOCK" ]]; then
  echo "FATAL: SUBMISSION_LOCK is not RUN_DIR/SUBMISSION_LOCK" >&2
  exit 27
fi
if [[ "${JOB_RECEIPT_PATH}" != "${RUN_DIR}/${JOB_RECEIPT_NAME}" ]]; then
  echo "FATAL: JOB_RECEIPT_PATH is not RUN_DIR/job_receipt.json" >&2
  exit 27
fi
RUN_DIR="$(cd -- "${RUN_DIR}" && pwd -P)"
case "${RUN_DIR}" in
  "${REPO_DIR}"|"${REPO_DIR}"/*)
    echo "FATAL: RUN_DIR must be an isolated staging directory outside REPO_DIR" >&2
    exit 25
    ;;
esac

META="${RUN_DIR}/complete_eq27_job_metadata.json"
OUT="${RUN_DIR}/complete_eq27.npz"
if [[ -e "${OUT}" || -L "${OUT}" ]]; then
  echo "FATAL: candidate NPZ already exists; repeat propagation is forbidden: ${OUT}" >&2
  exit 27
fi
if [[ -e "${META}" || -L "${META}" ]]; then
  echo "FATAL: job metadata already exists; repeat propagation is forbidden: ${META}" >&2
  exit 27
fi

cd "${REPO_DIR}"
actual_sha="$(git rev-parse HEAD)"
if [[ "${actual_sha}" != "${EXPECTED_GIT_SHA}" ]]; then
  echo "FATAL: SHA mismatch expected=${EXPECTED_GIT_SHA} actual=${actual_sha}" >&2
  exit 21
fi
if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "FATAL: staging worktree is not clean" >&2
  git status --short >&2
  exit 22
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "FATAL: config not found: ${CONFIG_PATH}" >&2
  exit 26
fi
actual_config_sha="$(sha256sum "${CONFIG_PATH}" | awk '{print $1}')"
if [[ "${actual_config_sha}" != "${EXPECTED_CONFIG_SHA256}" ]]; then
  echo "FATAL: configuration SHA mismatch expected=${EXPECTED_CONFIG_SHA256} actual=${actual_config_sha}" >&2
  exit 23
fi

export PYTHONPYCACHEPREFIX="${RUN_DIR}/pycache"
export CUPY_CACHE_DIR="${RUN_DIR}/cupy_cache"
export XDG_CACHE_HOME="${RUN_DIR}/xdg_cache"
export PYTHONPATH="${REPO_DIR}/Filament_python"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export UPPE_USE_GPU=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export CONFIG_PATH

source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python

cd "${RUN_DIR}"

# Exactly one Python process performs the CUDA guard, one full propagation, and
# metadata finalization.  The explicit case guard above prevents a loop or a
# second propagation from being smuggled in through environment variables.
python - <<'PY'
import json
import hashlib
import os
import subprocess
import sys
import traceback
from pathlib import Path

import cupy as cp
import numpy as np

from KHz_filament.device import xp
from KHz_filament.runner import run_from_file

sys.path.insert(0, str(Path(repo := os.environ["REPO_DIR"]) / "Filament_python" / "tools"))
from create_isaacs_complete_eq27_execution_lock import (  # noqa: E402
    ExecutionLockError,
    validate_manifest_lock,
)


meta_path = Path(os.environ["RUN_DIR"]) / "complete_eq27_job_metadata.json"
npz_path = Path(os.environ["RUN_DIR"]) / "complete_eq27.npz"
config = os.environ["CONFIG_PATH"]
threads = int(os.environ["SLURM_CPUS_PER_TASK"])

def git_head():
    return subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()

def gpu_name():
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    value = props["name"]
    return value.decode() if isinstance(value, bytes) else str(value)


def read_kv(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FATAL: cannot read job receipt {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("FATAL: job receipt must be a JSON object")
    return value


try:
    binding = validate_manifest_lock(
        Path(os.environ["MANIFEST_PATH"]),
        Path(os.environ["EXECUTION_LOCK_PATH"]),
        expected_manifest_sha256=os.environ["EXPECTED_MANIFEST_SHA256"],
        expected_lock_sha256=os.environ["EXPECTED_EXECUTION_LOCK_SHA256"],
        expected_git_sha=os.environ["EXPECTED_GIT_SHA"],
        require_clean=True,
        require_committed_manifest=True,
    )
except (ExecutionLockError, OSError, ValueError, KeyError, TypeError) as exc:
    raise SystemExit(f"FATAL: batch execution binding validation failed: {exc}") from exc

fixed_config = binding["config_path"].resolve()
if Path(config).resolve() != fixed_config:
    raise SystemExit("FATAL: batch CONFIG_PATH does not match the fixed manifest-derived config")
if binding["lock"].get("operator_mode") != "full_isaacs_eq27_complete":
    raise SystemExit("FATAL: execution lock operator_mode is not complete Eq.27")
if binding["lock"].get("use_raman_full_operator") is not True:
    raise SystemExit("FATAL: execution lock requires use_raman_full_operator=true")
if binding["manifest"].get("campaign_id") != os.environ["CAMPAIGN_ID"]:
    raise SystemExit("FATAL: batch manifest campaign_id mismatch")

submission_lock = Path(os.environ["SUBMISSION_LOCK"]).resolve()
if submission_lock != Path(os.environ["RUN_DIR"]).resolve() / "SUBMISSION_LOCK":
    raise SystemExit("FATAL: submission lock path is not RUN_DIR/SUBMISSION_LOCK")
if not submission_lock.is_file():
    raise SystemExit("FATAL: submission lock is missing")
submission_lock_sha = hashlib.sha256(submission_lock.read_bytes()).hexdigest()
submission = read_kv(submission_lock)
for key, expected in {
    "case_id": "complete_eq27",
    "campaign_id": os.environ["CAMPAIGN_ID"],
    "remote_campaign_root": os.environ["REMOTE_CAMPAIGN_ROOT"],
    "manifest_path": str(Path(os.environ["MANIFEST_PATH"]).resolve()),
    "manifest_sha256": os.environ["EXPECTED_MANIFEST_SHA256"],
    "execution_lock_path": str(Path(os.environ["EXECUTION_LOCK_PATH"]).resolve()),
    "execution_lock_sha256": os.environ["EXPECTED_EXECUTION_LOCK_SHA256"],
    "config_path": str(fixed_config),
    "expected_config_sha256": os.environ["EXPECTED_CONFIG_SHA256"],
    "expected_git_sha": os.environ["EXPECTED_GIT_SHA"],
}.items():
    if submission.get(key) != expected:
        raise SystemExit(f"FATAL: submission lock {key} mismatch")

global_lock = Path(os.environ["GLOBAL_CONSUMED_LOCK"]).resolve()
global_record_path = global_lock / "submission_record.txt"
if not global_record_path.is_file():
    raise SystemExit("FATAL: global consumed submission record is missing")
global_record = read_kv(global_record_path)
for key, expected in {
    "campaign_id": os.environ["CAMPAIGN_ID"],
    "remote_campaign_root": os.environ["REMOTE_CAMPAIGN_ROOT"],
    "manifest_path": str(Path(os.environ["MANIFEST_PATH"]).resolve()),
    "manifest_sha256": os.environ["EXPECTED_MANIFEST_SHA256"],
    "execution_lock_path": str(Path(os.environ["EXECUTION_LOCK_PATH"]).resolve()),
    "execution_lock_sha256": os.environ["EXPECTED_EXECUTION_LOCK_SHA256"],
    "expected_git_sha": os.environ["EXPECTED_GIT_SHA"],
    "run_dir": str(Path(os.environ["RUN_DIR"]).resolve()),
}.items():
    if global_record.get(key) != expected:
        raise SystemExit(f"FATAL: global consumed record {key} mismatch")

if "job_id" in submission or "job_id" in global_record:
    raise SystemExit("FATAL: submission/global records must not be edited after sbatch")
receipt_path = Path(os.environ["JOB_RECEIPT_PATH"]).resolve()
if receipt_path != Path(os.environ["RUN_DIR"]).resolve() / "job_receipt.json":
    raise SystemExit("FATAL: job receipt path is not RUN_DIR/job_receipt.json")
if not receipt_path.is_file():
    raise SystemExit("FATAL: held-job receipt is missing before propagation")
receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
receipt = read_object(receipt_path)
receipt_job_id = str(receipt.get("job_id") or "").strip()
if receipt_job_id != str(os.environ.get("SLURM_JOB_ID") or "").strip():
    raise SystemExit("FATAL: job receipt job_id does not match SLURM_JOB_ID")
receipt_token = str(receipt.get("reservation_token") or "").strip()
if not receipt_token:
    raise SystemExit("FATAL: job receipt reservation_token is empty")
if submission.get("reservation_token") != receipt_token or global_record.get("reservation_token") != receipt_token:
    raise SystemExit("FATAL: job receipt reservation_token does not match submission/global records")
for key, expected in {
    "schema": "khz_filament.isaacs_complete_eq27.job_receipt.v1",
    "state": "held",
    "campaign_id": os.environ["CAMPAIGN_ID"],
    "remote_campaign_root": os.environ["REMOTE_CAMPAIGN_ROOT"],
    "run_dir": str(Path(os.environ["RUN_DIR"]).resolve()),
    "manifest_path": str(Path(os.environ["MANIFEST_PATH"]).resolve()),
    "manifest_sha256": os.environ["EXPECTED_MANIFEST_SHA256"],
    "execution_lock_path": str(Path(os.environ["EXECUTION_LOCK_PATH"]).resolve()),
    "execution_lock_sha256": os.environ["EXPECTED_EXECUTION_LOCK_SHA256"],
    "config_path": str(fixed_config),
    "config_sha256": os.environ["EXPECTED_CONFIG_SHA256"],
    "expected_git_sha": os.environ["EXPECTED_GIT_SHA"],
}.items():
    if receipt.get(key) != expected:
        raise SystemExit(f"FATAL: job receipt {key} mismatch")

if xp.__name__ != "cupy" or cp.cuda.runtime.getDeviceCount() < 1:
    raise SystemExit("FATAL: CUDA/CuPy backend is required")
name = gpu_name()
if name != os.environ["EXPECTED_GPU_MODEL"]:
    raise SystemExit(
        f"FATAL: GPU model mismatch expected={os.environ['EXPECTED_GPU_MODEL']!r} actual={name!r}"
    )
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    if os.environ.get(key) != str(threads):
        raise SystemExit(f"FATAL: {key} is not consistent with SLURM_CPUS_PER_TASK")

props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
payload = {
    "schema": "khz_filament.isaacs_complete_eq27.job_metadata.v1",
    "case_id": os.environ["CASE_ID"],
    "status": "running",
    "exit_code": None,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "expected_git_sha": os.environ["EXPECTED_GIT_SHA"],
    "execution_git_sha": git_head(),
    "git_status_clean": True,
    "campaign_id": os.environ["CAMPAIGN_ID"],
    "remote_campaign_root": os.environ["REMOTE_CAMPAIGN_ROOT"],
    "manifest_path": os.environ["MANIFEST_PATH"],
    "manifest_sha256": os.environ["EXPECTED_MANIFEST_SHA256"],
    "execution_lock_path": os.environ["EXECUTION_LOCK_PATH"],
    "execution_lock_sha256": os.environ["EXPECTED_EXECUTION_LOCK_SHA256"],
    "submission_lock_path": str(submission_lock),
    "submission_lock_sha256": submission_lock_sha,
    "global_consumed_lock": os.environ["GLOBAL_CONSUMED_LOCK"],
    "job_receipt_path": str(receipt_path),
    "job_receipt_sha256": receipt_sha,
    "reservation_token": receipt_token,
    "config_path": config,
    "config_sha256": os.environ["EXPECTED_CONFIG_SHA256"],
    "gpu_model": name,
    "gpu_total_memory_bytes": int(props["totalGlobalMem"]),
    "cpu_threads": threads,
    "thread_environment": {key: os.environ[key] for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
    )},
    "staging_directory": repo,
    "run_directory": str(npz_path.parent),
    "operator_mode": "full_isaacs_eq27_complete",
    "use_raman_full_operator": True,
    "profiling_enabled": False,
    "propagation_invocations": 0,
    "peak_allocated_bytes": None,
    "peak_reserved_bytes": None,
    "peak_memory_source": "candidate NPZ gpu_allocated_step_bytes/gpu_reserved_step_bytes when available",
}
meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

failure = None
try:
    payload["propagation_invocations"] += 1
    run_from_file(config, out_path=str(npz_path), dtype="fp32")
    with np.load(npz_path, allow_pickle=False) as data:
        for source, target in (
            ("gpu_allocated_step_bytes", "peak_allocated_bytes"),
            ("gpu_reserved_step_bytes", "peak_reserved_bytes"),
        ):
            if source in data.files:
                values = np.asarray(data[source], dtype=np.float64)
                if values.size and np.all(np.isfinite(values)):
                    payload[target] = int(np.max(values))
        payload["npz_sha256"] = __import__("hashlib").sha256(npz_path.read_bytes()).hexdigest()
    payload["status"] = "completed"
    payload["exit_code"] = 0
except BaseException as exc:  # preserve failure provenance before re-raising
    failure = exc
    payload["status"] = "failed"
    payload["exit_code"] = 1
    payload["error_type"] = type(exc).__name__
    payload["error"] = str(exc)
    payload["traceback"] = traceback.format_exc()
finally:
    meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

if failure is not None:
    raise failure
PY
