#!/usr/bin/env bash
#SBATCH --job-name=hybrid-0p60-ab
#SBATCH --partition=gpu
#SBATCH --nodelist=m4gn1401
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=15:00:00

set -euo pipefail

: "${REPO_DIR:?missing REPO_DIR}"
: "${RUN_DIR:?missing RUN_DIR}"
: "${EXPECTED_GIT_SHA:?missing EXPECTED_GIT_SHA}"
: "${MANIFEST_PATH:?missing MANIFEST_PATH}"
: "${EXPECTED_MANIFEST_SHA256:?missing EXPECTED_MANIFEST_SHA256}"
: "${EXECUTION_LOCK_PATH:?missing EXECUTION_LOCK_PATH}"
: "${EXPECTED_EXECUTION_LOCK_SHA256:?missing EXPECTED_EXECUTION_LOCK_SHA256}"
: "${PROVENANCE_V2_PATH:?missing PROVENANCE_V2_PATH}"
: "${EXPECTED_PROVENANCE_V2_SHA256:?missing EXPECTED_PROVENANCE_V2_SHA256}"
: "${JOB_RECEIPT_PATH:?missing JOB_RECEIPT_PATH}"
: "${SUBMISSION_LOCK:?missing SUBMISSION_LOCK}"
: "${GLOBAL_CONSUMED_LOCK:?missing GLOBAL_CONSUMED_LOCK}"
: "${EXPECTED_GPU_MODEL:?missing EXPECTED_GPU_MODEL}"
: "${EXPECTED_NODELIST:?missing EXPECTED_NODELIST}"

readonly CAMPAIGN_ID="hybrid_propagation_validation_0p60"
readonly REMOTE_ROOT="/data/run01/scvi806/user_Wangjimin/hybrid_propagation_validation_0p60"
readonly FIXED_NODELIST="m4gn1401"
if [[ "${SLURM_CPUS_PER_TASK:-}" != "8" ]]; then
  echo "FATAL: paired campaign requires exactly 8 CPUs" >&2
  exit 20
fi
if [[ "${EXPECTED_NODELIST}" != "${FIXED_NODELIST}" || "${SLURM_JOB_NODELIST:-}" != "${FIXED_NODELIST}" ]]; then
  echo "FATAL: paired campaign requires exact node ${FIXED_NODELIST}" >&2
  exit 30
fi

REPO_DIR="$(cd -- "${REPO_DIR}" && pwd -P)"
RUN_DIR="$(cd -- "${RUN_DIR}" && pwd -P)"
case "${RUN_DIR}" in
  "${REMOTE_ROOT}"/*) ;;
  *) echo "FATAL: RUN_DIR is outside the fixed campaign root" >&2; exit 21 ;;
esac
if [[ "${JOB_RECEIPT_PATH}" != "${RUN_DIR}/job_receipt.json" || "${SUBMISSION_LOCK}" != "${RUN_DIR}/SUBMISSION_LOCK" ]]; then
  echo "FATAL: receipt/submission-lock paths are not fixed to RUN_DIR" >&2
  exit 22
fi
if [[ "${GLOBAL_CONSUMED_LOCK}" != "${REMOTE_ROOT}/.consumed.lock" ]]; then
  echo "FATAL: global campaign lock path is invalid" >&2
  exit 23
fi

cd "${REPO_DIR}"
actual_sha="$(git rev-parse HEAD)"
[[ "${actual_sha}" == "${EXPECTED_GIT_SHA}" ]] || { echo "FATAL: execution SHA mismatch" >&2; exit 24; }
[[ -z "$(git status --porcelain=v1)" ]] || { echo "FATAL: execution checkout is dirty" >&2; exit 25; }
for path in "${MANIFEST_PATH}" "${EXECUTION_LOCK_PATH}" "${PROVENANCE_V2_PATH}" "${JOB_RECEIPT_PATH}" "${SUBMISSION_LOCK}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || { echo "FATAL: required regular file missing: ${path}" >&2; exit 26; }
done
[[ -d "${GLOBAL_CONSUMED_LOCK}" && ! -L "${GLOBAL_CONSUMED_LOCK}" ]] || { echo "FATAL: global consumed lock is missing" >&2; exit 26; }
[[ "$(sha256sum "${MANIFEST_PATH}" | awk '{print $1}')" == "${EXPECTED_MANIFEST_SHA256}" ]] || { echo "FATAL: manifest SHA mismatch" >&2; exit 27; }
[[ "$(sha256sum "${EXECUTION_LOCK_PATH}" | awk '{print $1}')" == "${EXPECTED_EXECUTION_LOCK_SHA256}" ]] || { echo "FATAL: execution-lock SHA mismatch" >&2; exit 28; }
[[ "$(sha256sum "${PROVENANCE_V2_PATH}" | awk '{print $1}')" == "${EXPECTED_PROVENANCE_V2_SHA256}" ]] || { echo "FATAL: provenance-v2 SHA mismatch" >&2; exit 29; }

python Filament_python/tools/hpc_ops/provenance_v2.py validate \
  --repo "${REPO_DIR}" --manifest "${PROVENANCE_V2_PATH}"

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

source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate Filament_python

export REPO_DIR RUN_DIR EXPECTED_GIT_SHA MANIFEST_PATH EXECUTION_LOCK_PATH EXPECTED_NODELIST
export EXPECTED_EXECUTION_LOCK_SHA256 EXPECTED_GPU_MODEL JOB_RECEIPT_PATH
export SUBMISSION_LOCK GLOBAL_CONSUMED_LOCK CAMPAIGN_ID

python - <<'PY'
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

from KHz_filament.confio import load_all
from KHz_filament.constants import c0
from KHz_filament.ionization.common import W_CAP_DEFAULT
from KHz_filament.ionization.runtime import make_Wfunc, prepare_ionization_lut_cache
from KHz_filament.runner import run_from_file

repo = Path(os.environ["REPO_DIR"]).resolve()
run_dir = Path(os.environ["RUN_DIR"]).resolve()
manifest_path = Path(os.environ["MANIFEST_PATH"]).resolve()
lock_path = Path(os.environ["EXECUTION_LOCK_PATH"]).resolve()
receipt_path = Path(os.environ["JOB_RECEIPT_PATH"]).resolve()
LUT_AUDIT_SCHEMA = "khz_filament.hybrid_propagation_validation.lut_build_audit.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


lock_spec = importlib.util.spec_from_file_location(
    "hybrid_lock_batch", repo / "Filament_python/tools/create_hybrid_propagation_execution_lock.py"
)
if lock_spec is None or lock_spec.loader is None:
    raise RuntimeError("cannot load hybrid execution-lock validator")
lock_module = importlib.util.module_from_spec(lock_spec)
lock_spec.loader.exec_module(lock_module)
checked = lock_module.validate_manifest_lock(
    manifest_path, expected_git_sha=os.environ["EXPECTED_GIT_SHA"],
    require_clean=True, require_committed=True,
)
manifest = checked["manifest"]
lock = load_object(lock_path)
receipt = load_object(receipt_path)
submission = load_object(Path(os.environ["SUBMISSION_LOCK"]).resolve())
if receipt.get("schema") != "khz_filament.hybrid_propagation_validation.job_receipt.v1":
    raise RuntimeError("job receipt schema mismatch")
if submission.get("schema") != "khz_filament.hybrid_propagation_validation.submission_lock.v1":
    raise RuntimeError("submission lock schema mismatch")
if lock.get("schema") != lock_module.LOCK_SCHEMA or lock.get("status") != "authorized_not_consumed" or lock.get("expected_git_sha") != os.environ["EXPECTED_GIT_SHA"]:
    raise RuntimeError("execution lock is not authorized for this HEAD")
if lock.get("manifest_sha256") != checked["manifest_sha256"]:
    raise RuntimeError("execution lock is not bound to the manifest")
if lock.get("nodelist") != os.environ["EXPECTED_NODELIST"] or lock.get("expected_node") != os.environ["EXPECTED_NODELIST"]:
    raise RuntimeError("execution lock node binding is invalid")
if lock.get("lut_build_cap_inactive_required") is not True:
    raise RuntimeError("execution lock LUT cap-inactive requirement is missing")
if str(receipt.get("job_id")) != str(os.environ.get("SLURM_JOB_ID", "")):
    raise RuntimeError("job receipt does not match SLURM_JOB_ID")
for key, expected in {
    "campaign_id": os.environ["CAMPAIGN_ID"],
    "run_dir": str(run_dir),
    "expected_git_sha": os.environ["EXPECTED_GIT_SHA"],
    "manifest_sha256": os.environ["EXPECTED_MANIFEST_SHA256"],
    "execution_lock_sha256": os.environ["EXPECTED_EXECUTION_LOCK_SHA256"],
    "provenance_v2_sha256": os.environ["EXPECTED_PROVENANCE_V2_SHA256"],
    "state": "held_receipt_recorded",
}.items():
    if receipt.get(key) != expected:
        raise RuntimeError(f"job receipt {key} mismatch")
for key, expected in {
    "campaign_id": os.environ["CAMPAIGN_ID"],
    "run_dir": str(run_dir),
    "expected_git_sha": os.environ["EXPECTED_GIT_SHA"],
    "manifest_sha256": os.environ["EXPECTED_MANIFEST_SHA256"],
    "execution_lock_sha256": os.environ["EXPECTED_EXECUTION_LOCK_SHA256"],
    "provenance_v2_sha256": os.environ["EXPECTED_PROVENANCE_V2_SHA256"],
    "state": "reserved_before_sbatch",
}.items():
    if submission.get(key) != expected:
        raise RuntimeError(f"submission lock {key} mismatch")
owner_path = Path(os.environ["GLOBAL_CONSUMED_LOCK"]) / "owner"
if owner_path.read_text(encoding="utf-8").strip() != str(run_dir):
    raise RuntimeError("global consumed-lock owner mismatch")

gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"]
if isinstance(gpu_name, bytes):
    gpu_name = gpu_name.decode(errors="replace")
if gpu_name != os.environ["EXPECTED_GPU_MODEL"]:
    raise RuntimeError(f"GPU mismatch: {gpu_name!r}")
node_list = os.environ.get("SLURM_JOB_NODELIST", "")
if node_list != os.environ["EXPECTED_NODELIST"] or node_list != "m4gn1401":
    raise RuntimeError(f"node-list mismatch: {node_list!r}")


def audit_lut_tables(tables, ion_conf) -> dict:
    """Audit the exact cached/build tables before either candidate propagates."""
    builder_default_cap = float(W_CAP_DEFAULT)
    cap_threshold = 0.999 * builder_default_cap
    configured_global_cap = float(getattr(ion_conf, "W_cap", builder_default_cap))
    configured_by_species = {
        str(sp.get("name", "")): float(sp.get("W_cap", configured_global_cap))
        for sp in (getattr(ion_conf, "species", None) or [])
    }
    species = []
    for table in tables:
        name = str(table.get("species_name", ""))
        values = np.asarray(table.get("W_grid", []), dtype=np.float64).reshape(-1)
        finite = bool(values.size > 0 and np.all(np.isfinite(values)))
        maximum = float(np.max(values)) if values.size else float("nan")
        configured_cap = configured_by_species.get(name, configured_global_cap)
        configured_cap_valid = bool(np.isfinite(configured_cap) and configured_cap > 0.0)
        builder_cap_consistent = bool(np.isfinite(builder_default_cap) and builder_default_cap == float(W_CAP_DEFAULT))
        negative_step_count = 0
        max_relative_drop = None
        nondecreasing = False
        if finite and values.size >= 2:
            deltas = np.diff(values)
            negative = deltas < 0.0
            negative_step_count = int(np.count_nonzero(negative))
            nondecreasing = negative_step_count == 0
            if negative_step_count:
                previous = values[:-1][negative]
                relative_drop = -deltas[negative] / np.maximum(np.abs(previous), np.finfo(np.float64).tiny)
                max_relative_drop = float(np.max(relative_drop))
            else:
                max_relative_drop = 0.0
        cap_inactive = bool(finite and np.isfinite(maximum) and maximum < cap_threshold)
        species.append({
            "name": name,
            "W_grid_max": maximum,
            "configured_W_cap": configured_cap,
            "configured_cap_valid": configured_cap_valid,
            "builder_default_cap": builder_default_cap,
            "builder_cap_consistent": builder_cap_consistent,
            "finite": finite,
            "nondecreasing": nondecreasing,
            "negative_step_count": negative_step_count,
            "max_relative_drop": max_relative_drop,
            "cap_inactive": cap_inactive,
        })
    all_configured_caps_valid = bool(species) and all(item["configured_cap_valid"] for item in species)
    all_builder_caps_consistent = bool(species) and all(item["builder_cap_consistent"] for item in species)
    all_finite = bool(species) and all(item["finite"] for item in species)
    all_nondecreasing = bool(species) and all(item["nondecreasing"] for item in species)
    all_cap_inactive = bool(species) and all(item["cap_inactive"] for item in species)
    passed = bool(
        all_configured_caps_valid
        and all_builder_caps_consistent
        and all_finite
        and all_cap_inactive
    )
    audit = {
        "schema": LUT_AUDIT_SCHEMA,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "required": True,
        "builder_default_cap": builder_default_cap,
        "cap_threshold": cap_threshold,
        "all_configured_caps_valid": all_configured_caps_valid,
        "all_builder_caps_consistent": all_builder_caps_consistent,
        "all_finite": all_finite,
        "all_nondecreasing": all_nondecreasing,
        "all_cap_inactive": all_cap_inactive,
        "species": species,
    }
    if not audit["passed"]:
        raise RuntimeError(f"LUT build audit failed: {json.dumps(audit, sort_keys=True)}")
    return audit

# Common unmeasured warm-up: build/reuse the fixed LUT evaluator and initialize
# the CUDA/FFT path.  It does not propagate a second physical candidate.
warm_started = time.perf_counter()
_, beam, _, ion, _, _, _ = load_all(str(checked["reference_path"]))
omega0 = 2.0 * np.pi * c0 / beam.lam0
lut_tables = prepare_ionization_lut_cache(ion, omega0_SI=omega0, n0=beam.n0)
lut_build_audit = audit_lut_tables(lut_tables, ion)
rate = make_Wfunc(getattr(ion, "model", "none"), ion, omega0, beam.n0)
expects = getattr(rate, "_expects", "I")
sample = cp.asarray([1.0e16], dtype=cp.float64)
rate(sample)
cp.fft.fft(cp.ones(4096, dtype=cp.complex64))
cp.cuda.Stream.null.synchronize()
warmup_walltime = time.perf_counter() - warm_started

thread_env = {key: os.environ[key] for key in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
)}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_case(case_id: str, config_path: Path) -> dict:
    case_dir = run_dir / case_id
    case_dir.mkdir(mode=0o700)
    output = case_dir / f"{case_id}.npz"
    metadata_path = case_dir / f"{case_id}_job_metadata.json"
    if output.exists() or metadata_path.exists():
        raise RuntimeError(f"case output already exists: {case_id}")
    _, _, prop_config, _, _, _, _ = load_all(str(config_path))
    started_at = utc_now()
    started = time.perf_counter()
    metadata = {
        "schema": "khz_filament.hybrid_propagation_validation.case_metadata.v1",
        "campaign_id": os.environ["CAMPAIGN_ID"],
        "case_id": case_id,
        "status": "running",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "execution_git_sha": os.environ["EXPECTED_GIT_SHA"],
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "gpu_model": gpu_name,
        "nodelist": node_list,
        "expected_node": os.environ["EXPECTED_NODELIST"],
        "lut_build_audit": lut_build_audit,
        "cpu_threads": int(os.environ["SLURM_CPUS_PER_TASK"]),
        "thread_environment": thread_env,
        "warmup_rate_input": expects,
        "backend": "cupy",
        "dtype": "fp32",
        "linear_model": str(getattr(prop_config, "linear_model", "")),
        "linear_precision_strategy": str(getattr(prop_config, "linear_precision_strategy", "")),
        "started_at_utc": started_at,
    }
    try:
        run_from_file(str(config_path), out_path=str(output), dtype="fp32")
        cp.cuda.Stream.null.synchronize()
        metadata["case_total_walltime_s"] = time.perf_counter() - started
        with np.load(output, allow_pickle=False) as data:
            required = (
                "z_axis", "I_max_z", "rho_max_z", "U_z", "dz_used_z",
                "step_start_z_m", "step_end_z_m", "nonlinear_operator_applied",
                "nonlinear_operator_call_count_step", "ionization_solver_call_count_step",
                "linear_walltime_step_s", "nonlinear_walltime_step_s",
                "total_walltime_step_s", "raman_operator_substep_count",
                "raman_convolution_count_step", "diagnostic_validation_passed",
                "operator_energy_diagnostics_enabled", "energy_step_start_J",
                "energy_after_linear_half1_J", "energy_after_raman_pre_J",
                "energy_after_nonraman_J", "energy_after_raman_post_J",
                "energy_after_linear_half2_J",
                "propagation_mode", "z_nl_start_m",
            )
            missing = [key for key in required if key not in data.files]
            if missing:
                raise RuntimeError(f"case output missing fields: {missing}")
            for key in data.files:
                values = np.asarray(data[key])
                if values.dtype.kind in "biufc" and not np.all(np.isfinite(values)):
                    raise RuntimeError(f"case output contains nonfinite {key}")
            metadata["z_records"] = int(np.asarray(data["z_axis"]).size)
            metadata["propagation_mode"] = str(np.asarray(data["propagation_mode"]).item())
            metadata["z_nl_start_m"] = float(np.asarray(data["z_nl_start_m"]).item())
            metadata["diagnostic_validation_passed"] = bool(np.asarray(data["diagnostic_validation_passed"]).item())
        expected_mode = "full_nonlinear_from_z0" if case_id == "reference" else "hybrid"
        expected_start = 0.0 if case_id == "reference" else 0.6
        if metadata["propagation_mode"] != expected_mode or metadata["z_nl_start_m"] != expected_start:
            raise RuntimeError("case output mode/start mismatch")
        if not metadata["diagnostic_validation_passed"]:
            raise RuntimeError("case diagnostic validation did not pass")
        metadata.update(
            status="completed", exit_code=0, ended_at_utc=utc_now(),
            npz_path=str(output), npz_sha256=sha256(output),
        )
    except BaseException as exc:
        metadata.update(
            status="failed", exit_code=1,
            ended_at_utc=utc_now(),
            case_total_walltime_s=time.perf_counter() - started,
            error_type=type(exc).__name__, error_message=str(exc), traceback=traceback.format_exc(),
        )
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        raise
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


pair_started_at = utc_now()
pair_started = time.perf_counter()
reference = run_case("reference", checked["reference_path"])
hybrid = run_case("hybrid", checked["hybrid_path"])
pair = {
    "schema": "khz_filament.hybrid_propagation_validation.paired_metadata.v1",
    "campaign_id": os.environ["CAMPAIGN_ID"],
    "status": "completed",
    "exit_code": 0,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "execution_git_sha": os.environ["EXPECTED_GIT_SHA"],
    "gpu_model": gpu_name,
    "nodelist": node_list,
    "expected_node": os.environ["EXPECTED_NODELIST"],
    "lut_build_audit": lut_build_audit,
    "case_order": ["reference", "hybrid"],
    "allocation_count": 1,
    "started_at_utc": pair_started_at,
    "ended_at_utc": utc_now(),
    "warmup_walltime_s": warmup_walltime,
    "paired_cases_walltime_s": time.perf_counter() - pair_started,
    "manifest_path": str(manifest_path),
    "manifest_sha256": checked["manifest_sha256"],
    "execution_lock_path": str(lock_path),
    "execution_lock_sha256": os.environ["EXPECTED_EXECUTION_LOCK_SHA256"],
    "provenance_v2_path": os.environ["PROVENANCE_V2_PATH"],
    "provenance_v2_sha256": os.environ["EXPECTED_PROVENANCE_V2_SHA256"],
    "job_receipt_path": str(receipt_path),
    "job_receipt_sha256": sha256(receipt_path),
    "reference_metadata": reference,
    "hybrid_metadata": hybrid,
    "raw_npz_policy": "retained_on_HPC_RUN_DIR_not_repository",
}
(run_dir / "paired_job_metadata.json").write_text(json.dumps(pair, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": "completed", "job_id": pair["slurm_job_id"]}, indent=2))
PY
