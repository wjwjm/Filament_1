#!/usr/bin/env python3
"""Validate and postprocess a complete Isaacs Eq. (27) candidate NPZ.

The raw NPZ is read in place and is never copied into the repository.  The
postprocessor reuses the current-observability validator, then adds the
complete-operator mode, semantic-string, Raman-closure, energy, adaptive-step,
and memory-diagnostic checks needed for C2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
FILAMENT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(FILAMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(FILAMENT_ROOT))

from validate_current_observability_baseline import sha256, validate_npz  # noqa: E402
from create_isaacs_complete_eq27_execution_lock import (  # noqa: E402
    ExecutionLockError,
    validate_manifest_lock,
)


COMPLETE_MODE = "full_isaacs_eq27_complete"
RAMAN_Z_FIELDS = (
    "IR_max_z", "IR_abs_max_z", "delta_n_elec_max_z", "delta_n_rot_max_z",
    "delta_n_elec_applied_max_z", "delta_n_rot_applied_max_z",
    "dphi_elec_max_abs_z", "dphi_rot_max_abs_z", "dphi_elec_applied_max_abs_z",
    "dphi_rot_applied_max_abs_z", "alpha_R_raw_max_z", "alpha_R_applied_max_z",
    "raman_rhs_l2_norm", "raman_target_loss_step_J", "raman_actual_loss_step_J",
    "raman_closure_residual_step", "raman_target_loss_cumulative_J",
    "raman_actual_loss_cumulative_J", "raman_cumulative_closure_residual",
    "raman_convolution_count_step", "raman_operator_substep_count",
    "raman_operator_walltime_step_s", "raman_energy_projection_iterations",
    "raman_energy_projection_scale_deviation", "raman_energy_projection_initial_residual",
    "gpu_allocated_step_bytes", "gpu_reserved_step_bytes",
)
SEMANTIC_EXPECTATIONS = {
    "raman_operator_mode": COMPLETE_MODE,
    "delta_n_rot_applied_semantics": "not_applicable_full_complex_operator",
    "delta_n_elec_applied_semantics": "equivalent_n2_I_trace_full_complex_operator",
    "dphi_kerr_semantics": "not_applicable_scalar_phase_full_complex_operator",
    "self_steepening_semantics": "full_product_derivative_D_S_in_complete_complex_operator",
    "raman_closure_residual_semantics": "field_vs_eq10",
}
RAMAN_CLOSURE_STEP_P99_MAX = 1.0e-3
RAMAN_CLOSURE_CUMULATIVE_MAX = 5.0e-3
CAMPAIGN_ID = "isaacs_complete_eq27_c2"
REMOTE_CAMPAIGN_ROOT = "/data/run01/scvi806/user_Wangjimin/isaacs_complete_eq27_c2"
EXPECTED_GPU_MODEL = "NVIDIA GeForce RTX 5090"
EXPECTED_I_CAP = 1.0e19
LOCK_SCHEMA = "khz_filament.isaacs_complete_eq27.c2_execution_lock.v1"
FIXED_MANIFEST_REL = "results/isaacs_complete_eq27/submission_manifest.json"
FIXED_CONFIG_REL = "results/isaacs_complete_eq27/120fs_talebpour_isaacs_complete_eq27.json"
JOB_RECEIPT_SCHEMA = "khz_filament.isaacs_complete_eq27.job_receipt.v1"


def execution_git_sha(metadata: dict[str, Any]) -> str:
    """Return the exact execution SHA recorded by the C2 batch job."""
    return str(metadata.get("execution_git_sha") or "").strip()


def _job_id(metadata: dict[str, Any]) -> str:
    """Return the scheduler id without accepting an empty/whitespace value."""
    return str(metadata.get("slurm_job_id") or "").strip()


def _scalar(data: Any, key: str, default: Any = None) -> Any:
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    if value.size != 1:
        return default
    item = value.reshape(-1)[0]
    return item.item() if hasattr(item, "item") else item


def _finite_z_field(data: Any, key: str, n: int, failures: list[str]) -> np.ndarray | None:
    if key not in data.files:
        failures.append(f"missing complete Eq.27 diagnostic: {key}")
        return None
    try:
        values = np.asarray(data[key])
    except Exception as exc:
        failures.append(f"{key} cannot be read as an array: {type(exc).__name__}: {exc}")
        return None
    if values.ndim != 1 or values.size != n:
        failures.append(f"{key} is not z-aligned")
        return None
    try:
        numeric = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        failures.append(f"{key} is not numeric")
        return None
    if not np.all(np.isfinite(numeric)):
        failures.append(f"{key} contains NaN/Inf")
    return numeric


def _read_object(path: Path, label: str, failures: list[str]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        failures.append(f"{label} is unreadable: {exc}")
        return {}
    if not isinstance(payload, dict):
        failures.append(f"{label} must be a JSON object")
        return {}
    return payload


def _read_kv(path: Path, label: str, failures: list[str]) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        failures.append(f"{label} is unreadable: {exc}")
        return {}
    result: dict[str, str] = {}
    for line in lines:
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _path_inside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(FILAMENT_ROOT.resolve())
    except ValueError:
        return False
    return True


def _hash_record(path: Path, label: str, failures: list[str]) -> dict[str, str]:
    try:
        digest = sha256(path)
    except OSError as exc:
        failures.append(f"{label} cannot be hashed: {exc}")
        digest = ""
    return {"path": str(path.resolve()), "sha256": digest}


class _DataView(dict[str, Any]):
    @property
    def files(self) -> list[str]:
        return list(self)

    def __enter__(self) -> "_DataView":
        return self

    def __exit__(self, *_: Any) -> bool:
        return False


def _load_npz_safe(path: Path, failures: list[str]) -> _DataView:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            return _DataView({key: np.asarray(loaded[key]) for key in loaded.files})
    except Exception as exc:
        failures.append(f"candidate NPZ cannot be loaded safely: {type(exc).__name__}: {exc}")
        return _DataView()


def _numeric_array(data: Any, key: str, failures: list[str]) -> np.ndarray:
    if key not in data.files:
        return np.asarray([])
    try:
        return np.asarray(data[key], dtype=float)
    except Exception as exc:
        failures.append(f"{key} is not numeric: {type(exc).__name__}: {exc}")
        return np.asarray([])


def _validate_execution_binding(
    npz_path: Path,
    config_path: Path,
    metadata_path: Path | None,
    metadata: dict[str, Any],
    *,
    execution_lock_path: Path | None,
    submission_lock_path: Path | None,
    manifest_path: Path | None,
    expected_execution_lock_sha256: str | None,
    expected_execution_sha: str | None,
    job_receipt_path: Path | None,
    failures: list[str],
) -> dict[str, Any]:
    """Validate the immutable C2 execution/submit chain for one candidate."""
    binding: dict[str, Any] = {}
    fixed_manifest = (FILAMENT_ROOT / FIXED_MANIFEST_REL).resolve()
    fixed_config = (FILAMENT_ROOT / FIXED_CONFIG_REL).resolve()
    if config_path.resolve() != fixed_config:
        failures.append(f"candidate config path is not fixed: {config_path}")
    if not _path_inside_repo(config_path):
        failures.append("candidate config is outside the repository")
    if metadata_path is None or not metadata_path.is_file():
        failures.append("C2 run metadata file is required for the execution binding")
    if execution_lock_path is None:
        failures.append("C2 execution lock path is required")
    if submission_lock_path is None:
        failures.append("C2 submission lock path is required")
    if manifest_path is None:
        failures.append("C2 manifest path is required")
    if job_receipt_path is None:
        failures.append("C2 held-job receipt path is required")
    if metadata_path is None or execution_lock_path is None or submission_lock_path is None or manifest_path is None or job_receipt_path is None:
        return binding

    metadata_path = metadata_path.resolve()
    execution_lock_path = execution_lock_path.resolve()
    submission_lock_path = submission_lock_path.resolve()
    manifest_path = manifest_path.resolve()
    job_receipt_path = job_receipt_path.resolve()
    if manifest_path != fixed_manifest:
        failures.append(f"candidate manifest path is not fixed: {manifest_path}")
    if execution_lock_path == manifest_path or not execution_lock_path.is_file():
        failures.append("candidate execution lock does not exist")
    lock_sha = _hash_record(execution_lock_path, "candidate execution lock", failures)
    expected_lock_sha = str(expected_execution_lock_sha256 or metadata.get("execution_lock_sha256") or "").strip().lower()
    if not expected_lock_sha:
        failures.append("candidate execution lock SHA256 is required")
    elif lock_sha["sha256"] != expected_lock_sha:
        failures.append("candidate execution lock SHA256 does not match the expected binding")
    lock = _read_object(execution_lock_path, "candidate execution lock", failures)
    if lock.get("schema") != LOCK_SCHEMA:
        failures.append("candidate execution lock schema is invalid")
    for key, expected in {
        "campaign_id": CAMPAIGN_ID,
        "remote_campaign_root": REMOTE_CAMPAIGN_ROOT,
        "status": "authorized_not_consumed",
        "expected_gpu_model": EXPECTED_GPU_MODEL,
        "operator_mode": COMPLETE_MODE,
        "use_raman_full_operator": True,
    }.items():
        if lock.get(key) != expected:
            failures.append(f"candidate execution lock {key} does not match the fixed C2 binding")
    if lock.get("config_path") != FIXED_CONFIG_REL or lock.get("derived_config_path") != FIXED_CONFIG_REL:
        failures.append("candidate execution lock config path is not fixed")
    try:
        actual_config_sha = sha256(config_path)
    except OSError as exc:
        actual_config_sha = ""
        failures.append(f"candidate config cannot be hashed: {exc}")
    if lock.get("config_sha256") != actual_config_sha or lock.get("derived_config_sha256") != actual_config_sha:
        failures.append("candidate execution lock config SHA does not match the actual fixed config")
    if metadata.get("execution_lock_path") != str(execution_lock_path):
        failures.append("run metadata execution_lock_path does not match the supplied lock")
    if metadata.get("execution_lock_sha256") != lock_sha["sha256"]:
        failures.append("run metadata execution_lock_sha256 does not match the supplied lock")

    manifest_hash = _hash_record(manifest_path, "candidate manifest", failures)
    expected_manifest_sha = str(metadata.get("manifest_sha256") or lock.get("manifest_sha256") or "").strip().lower()
    if not expected_manifest_sha or manifest_hash["sha256"] != expected_manifest_sha:
        failures.append("candidate manifest SHA256 does not match the execution metadata/lock")
    manifest = _read_object(manifest_path, "candidate manifest", failures)
    if manifest.get("campaign_id") != CAMPAIGN_ID or manifest.get("remote_campaign_root") != REMOTE_CAMPAIGN_ROOT:
        failures.append("candidate manifest campaign binding is not fixed")
    if manifest.get("derived_config") != FIXED_CONFIG_REL or manifest.get("derived_config_sha256") != actual_config_sha:
        failures.append("candidate manifest derived config binding is not fixed")
    if metadata.get("manifest_path") != str(manifest_path):
        failures.append("run metadata manifest_path does not match the supplied manifest")
    if metadata.get("manifest_sha256") != manifest_hash["sha256"]:
        failures.append("run metadata manifest_sha256 does not match the supplied manifest")

    expected_submission = (npz_path.resolve().parent / "SUBMISSION_LOCK").resolve()
    if submission_lock_path != expected_submission:
        failures.append("candidate submission lock must be RUN_DIR/SUBMISSION_LOCK")
    submission_hash = _hash_record(submission_lock_path, "candidate submission lock", failures)
    submission = _read_kv(submission_lock_path, "candidate submission lock", failures)
    for key, expected in {
        "case_id": "complete_eq27",
        "campaign_id": CAMPAIGN_ID,
        "remote_campaign_root": REMOTE_CAMPAIGN_ROOT,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash["sha256"],
        "execution_lock_path": str(execution_lock_path),
        "execution_lock_sha256": lock_sha["sha256"],
        "config_path": str(config_path.resolve()),
        "expected_config_sha256": actual_config_sha,
        "expected_git_sha": str(metadata.get("execution_git_sha") or "").strip(),
    }.items():
        if submission.get(key) != expected:
            failures.append(f"candidate submission lock {key} does not match the execution binding")

    global_lock_value = metadata.get("global_consumed_lock")
    global_lock = Path(str(global_lock_value or "")).resolve()
    expected_global_lock = Path(REMOTE_CAMPAIGN_ROOT) / ".consumed.lock"
    if str(global_lock) != str(expected_global_lock):
        failures.append("candidate global consumed lock path is not fixed")
    global_record = global_lock / "submission_record.txt"
    global_hash = _hash_record(global_record, "candidate global submission record", failures)
    record = _read_kv(global_record, "candidate global submission record", failures)
    for key, expected in {
        "campaign_id": CAMPAIGN_ID,
        "remote_campaign_root": REMOTE_CAMPAIGN_ROOT,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash["sha256"],
        "execution_lock_path": str(execution_lock_path),
        "execution_lock_sha256": lock_sha["sha256"],
        "expected_git_sha": str(metadata.get("execution_git_sha") or "").strip(),
        "run_dir": str(npz_path.resolve().parent),
    }.items():
        if record.get(key) != expected:
            failures.append(f"candidate global submission record {key} does not match the execution binding")
    record_job = record.get("job_id", "")
    if "job_id" in submission or "job_id" in record:
        failures.append("submission/global records must not be edited after held sbatch")
    if record_job and record_job != _job_id(metadata):
        failures.append("candidate global submission record job_id does not match metadata")
    if metadata.get("submission_lock_path") != str(submission_lock_path):
        failures.append("run metadata submission_lock_path does not match the supplied lock")
    if metadata.get("submission_lock_sha256") != submission_hash["sha256"]:
        failures.append("run metadata submission_lock_sha256 does not match the supplied lock")
    if metadata.get("global_consumed_lock") != str(global_lock):
        failures.append("run metadata global_consumed_lock does not match the fixed campaign lock")

    expected_receipt = (npz_path.resolve().parent / "job_receipt.json").resolve()
    if job_receipt_path != expected_receipt:
        failures.append("candidate held-job receipt must be RUN_DIR/job_receipt.json")
    receipt_hash = _hash_record(job_receipt_path, "candidate held-job receipt", failures)
    receipt = _read_object(job_receipt_path, "candidate held-job receipt", failures)
    receipt_job = str(receipt.get("job_id") or "").strip()
    metadata_job = _job_id(metadata)
    if receipt.get("schema") != JOB_RECEIPT_SCHEMA or receipt.get("state") != "held":
        failures.append("candidate held-job receipt schema/state is invalid")
    if receipt_job != metadata_job:
        failures.append("candidate held-job receipt job_id does not match metadata")
    receipt_token = str(receipt.get("reservation_token") or "").strip()
    if not receipt_token:
        failures.append("candidate held-job receipt reservation_token is empty")
    if submission.get("reservation_token") != receipt_token or record.get("reservation_token") != receipt_token:
        failures.append("candidate held-job receipt reservation_token does not match submission/global records")
    for key, expected in {
        "campaign_id": CAMPAIGN_ID,
        "remote_campaign_root": REMOTE_CAMPAIGN_ROOT,
        "run_dir": str(npz_path.resolve().parent),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash["sha256"],
        "execution_lock_path": str(execution_lock_path),
        "execution_lock_sha256": lock_sha["sha256"],
        "config_path": str(config_path.resolve()),
        "config_sha256": actual_config_sha,
        "expected_git_sha": str(metadata.get("execution_git_sha") or "").strip(),
    }.items():
        if receipt.get(key) != expected:
            failures.append(f"candidate held-job receipt {key} does not match the execution binding")
    if metadata.get("job_receipt_path") != str(job_receipt_path):
        failures.append("run metadata job_receipt_path does not match the supplied receipt")
    if metadata.get("job_receipt_sha256") != receipt_hash["sha256"]:
        failures.append("run metadata job_receipt_sha256 does not match the supplied receipt")

    real_sha = str(metadata.get("execution_git_sha") or expected_execution_sha or "").strip()
    try:
        locked = validate_manifest_lock(
            manifest_path,
            execution_lock_path,
            expected_manifest_sha256=manifest_hash["sha256"],
            expected_lock_sha256=lock_sha["sha256"],
            expected_git_sha=real_sha,
            require_clean=True,
            require_committed_manifest=True,
        )
    except Exception as exc:
        failures.append(f"shared manifest/C1 execution-lock validation failed: {exc}")
    else:
        if locked["config_path"].resolve() != config_path.resolve():
            failures.append("shared manifest validator config path does not match candidate config")

    binding.update({
        "npz": _hash_record(npz_path, "candidate NPZ", failures),
        "metadata": _hash_record(metadata_path, "candidate metadata", failures),
        "config": {"path": str(config_path.resolve()), "sha256": actual_config_sha},
        "manifest": manifest_hash,
        "execution_lock": lock_sha,
        "submission_lock": submission_hash,
        "job_receipt": receipt_hash,
        "global_consumed_lock": {"path": str(global_record.resolve()), "sha256": global_hash["sha256"]},
        "campaign_id": CAMPAIGN_ID,
        "remote_campaign_root": REMOTE_CAMPAIGN_ROOT,
        "operator_mode": COMPLETE_MODE,
        "use_raman_full_operator": True,
    })
    return binding


def _check_monotonic_nonnegative(values: np.ndarray, key: str, failures: list[str]) -> None:
    if np.any(values < 0.0) or np.any(np.diff(values) < 0.0):
        failures.append(f"{key} is not non-negative and non-decreasing")


def _validate_impl(
    npz_path: Path,
    config_path: Path,
    metadata_path: Path | None = None,
    *,
    expected_execution_sha: str | None = None,
    execution_lock_path: Path | None = None,
    submission_lock_path: Path | None = None,
    manifest_path: Path | None = None,
    expected_execution_lock_sha256: str | None = None,
    job_receipt_path: Path | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    metadata: dict[str, Any] = {}
    if metadata_path is None:
        failures.append("C2 run metadata is required")
    elif not metadata_path.is_file():
        failures.append(f"C2 run metadata does not exist: {metadata_path}")
    else:
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                failures.append("C2 run metadata must be a JSON object")
            else:
                metadata = loaded
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"cannot read C2 run metadata: {exc}")

    provenance_binding = _validate_execution_binding(
        npz_path,
        config_path,
        metadata_path,
        metadata,
        execution_lock_path=execution_lock_path,
        submission_lock_path=submission_lock_path,
        manifest_path=manifest_path,
        expected_execution_lock_sha256=expected_execution_lock_sha256,
        expected_execution_sha=expected_execution_sha,
        job_receipt_path=job_receipt_path or (
            Path(str(metadata.get("job_receipt_path"))).resolve()
            if metadata.get("job_receipt_path") else None
        ),
        failures=failures,
    )

    try:
        base = validate_npz(npz_path, config_path, metadata_path)
    except Exception as exc:
        # Preserve a machine-readable failed audit for malformed/incomplete
        # candidate artifacts instead of turning an input failure into an
        # unlabelled traceback.
        base = {
            "passed": False,
            "failures": [f"base candidate validation failed: {exc}"],
            "metadata": metadata,
            "axial_rows": [],
        }
    failures = list(failures) + list(base["failures"])
    metadata = dict(metadata or base.get("metadata") or {})
    actual_execution_sha = execution_git_sha(metadata)
    if actual_execution_sha:
        failures = [item for item in failures if item != "run metadata lacks execution_git_sha"]
    if not expected_execution_sha or not str(expected_execution_sha).strip():
        failures.append("expected C2 execution SHA is required")
    elif actual_execution_sha != str(expected_execution_sha).strip():
        failures.append("run metadata execution_git_sha does not exactly match expected C2 execution SHA")
    if metadata.get("status") != "completed":
        failures.append("run metadata status is not completed")
    exit_code = metadata.get("exit_code")
    if isinstance(exit_code, bool) or exit_code != 0:
        failures.append("run metadata exit_code is not zero")
    invocation_count = metadata.get("propagation_invocations")
    if invocation_count != 1 or isinstance(invocation_count, bool):
        failures.append("run metadata propagation_invocations is not exactly 1")
    if not _job_id(metadata):
        failures.append("run metadata lacks a non-empty slurm job id")
    if metadata.get("profiling_enabled") is True:
        failures.append("profiling is enabled for the C2 candidate")

    try:
        actual_config_sha = sha256(config_path)
    except OSError as exc:
        failures.append(f"cannot hash supplied config: {exc}")
        actual_config_sha = ""
    metadata_config_sha = str(metadata.get("config_sha256") or "").strip().lower()
    if not metadata_config_sha:
        failures.append("run metadata lacks config_sha256")
    elif not actual_config_sha or metadata_config_sha != actual_config_sha:
        failures.append("run metadata config_sha256 does not match the supplied config")

    try:
        actual_npz_sha = sha256(npz_path)
    except OSError as exc:
        failures.append(f"cannot hash supplied NPZ: {exc}")
        actual_npz_sha = ""
    metadata_npz_sha = str(metadata.get("npz_sha256") or "").strip().lower()
    if not metadata_npz_sha:
        failures.append("run metadata lacks npz_sha256")
    elif not actual_npz_sha or metadata_npz_sha != actual_npz_sha:
        failures.append("run metadata npz_sha256 does not match the supplied NPZ")

    if metadata and metadata.get("case_id") not in (None, "complete_eq27"):
        failures.append("run metadata case_id is not complete_eq27")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"cannot read supplied config: {exc}")
        config = {}
    if config.get("raman", {}).get("operator_mode") != COMPLETE_MODE:
        failures.append(f"configuration is not raman.operator_mode={COMPLETE_MODE}")
    prop = config.get("propagation", {})
    ionization = config.get("ionization", {})
    raman = config.get("raman", {})
    if ionization.get("I_cap") != EXPECTED_I_CAP:
        failures.append(f"configuration ionization.I_cap is not the fixed {EXPECTED_I_CAP:g}")
    if prop.get("use_raman_full_operator") is not True:
        failures.append("complete candidate requires propagation.use_raman_full_operator=true")
    if prop.get("use_electronic_kerr") is not True or prop.get("use_self_steepening") is not True:
        failures.append("complete candidate requires electronic Kerr and self-steepening enabled")
    if prop.get("use_raman_absorption") is not False or raman.get("absorption") is not False:
        failures.append("complete candidate must keep Raman absorption disabled")

    semantic_values: dict[str, Any] = {}
    fields: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, float]] = []
    extras: list[dict[str, float]] = []
    closure: dict[str, Any] = {}
    with _load_npz_safe(npz_path, failures) as data:
        z = _numeric_array(data, "z_axis", failures)
        if z.ndim != 1:
            failures.append("z_axis is not one-dimensional")
            z = np.asarray([])
        n = int(z.size)
        for key, expected in SEMANTIC_EXPECTATIONS.items():
            value = _scalar(data, key)
            semantic_values[key] = value
            if value != expected:
                failures.append(f"{key}={value!r}, expected {expected!r}")
        for key in ("raman_operator_feedback_enabled", "raman_absorption_on", "raman_absorption_calculated"):
            if key in data.files:
                semantic_values[key] = bool(_scalar(data, key))
        if semantic_values.get("raman_operator_feedback_enabled") is not True:
            failures.append("raman_operator_feedback_enabled is not true")
        if semantic_values.get("raman_absorption_on") is True:
            failures.append("raman_absorption_on is unexpectedly true")
        if semantic_values.get("raman_absorption_calculated") is True:
            failures.append("raman_absorption_calculated is unexpectedly true")

        intensity = _finite_z_field(data, "I_max_z", n, failures)
        if intensity is not None and np.all(np.isfinite(intensity)):
            cap_margin = EXPECTED_I_CAP * (1.0 - 1.0e-6)
            if float(np.max(intensity)) >= cap_margin:
                failures.append(
                    f"I_max_z reaches ionization.I_cap margin: max={float(np.max(intensity)):g} "
                    f"cap_margin={cap_margin:g}"
                )

        applied = _finite_z_field(data, "raman_operator_applied", n, failures)
        if applied is not None and not np.all(applied.astype(bool)):
            failures.append("raman_operator_applied is not true for every accepted z step")
        rhs = _finite_z_field(data, "raman_rhs_l2_norm", n, failures)
        if rhs is not None and not np.any(rhs > 0.0):
            failures.append("complete Raman RHS is unexpectedly all zero")
        for key in RAMAN_Z_FIELDS:
            values = _finite_z_field(data, key, n, failures)
            if values is not None:
                fields[key] = {
                    "shape": list(np.asarray(data[key]).shape),
                    "finite": bool(np.all(np.isfinite(values))),
                    "max_abs": float(np.max(np.abs(values))) if values.size else None,
                }
        step_closure = _numeric_array(data, "raman_closure_residual_step", failures)
        cumulative_closure = _numeric_array(data, "raman_cumulative_closure_residual", failures)
        if step_closure.size:
            p99 = float(np.quantile(np.abs(step_closure), 0.99))
            closure["step_p99_abs"] = p99
            if p99 > RAMAN_CLOSURE_STEP_P99_MAX:
                failures.append(f"Raman per-step closure p99 exceeds {RAMAN_CLOSURE_STEP_P99_MAX:g}")
        if cumulative_closure.size:
            final = float(abs(cumulative_closure[-1]))
            closure["cumulative_final_abs"] = final
            if final > RAMAN_CLOSURE_CUMULATIVE_MAX:
                failures.append(f"Raman cumulative closure exceeds {RAMAN_CLOSURE_CUMULATIVE_MAX:g}")

        for key in ("U_z", "U_rel_change_z", "E_dep_cumulative_z", "E_loss_from_input_z", "dz_used_z", "adaptive_rejection_count_z", "safety_mode_trigger_count_z"):
            values = _finite_z_field(data, key, n, failures)
            if values is None:
                continue
            if key == "dz_used_z" and np.any(values <= 0.0):
                failures.append("dz_used_z contains non-positive values")
            if key in ("adaptive_rejection_count_z", "safety_mode_trigger_count_z"):
                _check_monotonic_nonnegative(values, key, failures)
            if key == "E_dep_cumulative_z" and np.any(np.diff(values) < -1e-10):
                failures.append("E_dep_cumulative_z is not non-decreasing")
        dz_values = _numeric_array(data, "dz_used_z", failures)
        if dz_values.size:
            if np.max(dz_values) > float(prop.get("dz", np.inf)) * (1.0 + 1e-6):
                failures.append("dz_used_z exceeds configured dz")
            if np.min(dz_values) < float(prop.get("dz_min", 0.0)) * (1.0 - 1e-6):
                failures.append("dz_used_z is below configured dz_min")

        available_extra = [
            key for key in RAMAN_Z_FIELDS
            if key in fields and np.asarray(data[key]).ndim == 1 and np.asarray(data[key]).size == n
        ]
        available_extra += [
            key for key in ("raman_operator_applied",)
            if key in data.files and _numeric_array(data, key, failures).ndim == 1 and _numeric_array(data, key, failures).size == n
        ]
        for index in range(n):
            extras.append({
                "z_m": float(z[index]),
                "x_focus_cm": 100.0 * (float(z[index]) - 0.95),
                **{key: float(np.asarray(data[key], dtype=float)[index]) for key in available_extra},
            })
        base_rows = base.get("axial_rows", [])
        rows = list(base_rows)

    # Scalars are part of the C2 contract as well as the semantic strings.
    with _load_npz_safe(npz_path, failures) as data:
        for key, expected, tolerance in (
            ("n2_elec_used", 7.8e-24, 1e-15),
            ("n_R_used", 2.3e-23, 1e-15),
        ):
            value = _scalar(data, key)
            try:
                numeric = None if value is None else float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                failures.append(f"{key} is not numeric: {type(exc).__name__}: {exc}")
                numeric = None
            if numeric is None or abs(numeric - expected) > tolerance * max(abs(expected), 1e-300):
                failures.append(f"{key} does not match locked C2 parameter")

    return {
        "passed": not failures,
        "failures": failures,
        "base": base,
        "metadata": metadata,
        "semantic_values": semantic_values,
        "field_status": fields,
        "closure": closure,
        "axial_rows": rows,
        "raman_rows": extras,
        "npz_sha256": actual_npz_sha,
        "config_sha256": actual_config_sha,
        "expected_execution_sha": expected_execution_sha,
        "provenance_binding": provenance_binding,
    }


def validate(
    npz_path: Path,
    config_path: Path,
    metadata_path: Path | None = None,
    *,
    expected_execution_sha: str | None = None,
    execution_lock_path: Path | None = None,
    submission_lock_path: Path | None = None,
    manifest_path: Path | None = None,
    expected_execution_lock_sha256: str | None = None,
    job_receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Return a failed audit for every malformed NPZ/type/index exception."""
    try:
        return _validate_impl(
            npz_path,
            config_path,
            metadata_path,
            expected_execution_sha=expected_execution_sha,
            execution_lock_path=execution_lock_path,
            submission_lock_path=submission_lock_path,
            manifest_path=manifest_path,
            expected_execution_lock_sha256=expected_execution_lock_sha256,
            job_receipt_path=job_receipt_path,
        )
    except Exception as exc:
        metadata: dict[str, Any] = {}
        if metadata_path is not None and metadata_path.is_file():
            try:
                loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    metadata = loaded
            except Exception:
                pass
        def safe_hash(path: Path) -> str:
            try:
                return sha256(path)
            except Exception:
                return ""
        return {
            "passed": False,
            "failures": [f"candidate validation failed safely: {type(exc).__name__}: {exc}"],
            "base": {"passed": False, "failures": [], "axial_rows": []},
            "metadata": metadata,
            "semantic_values": {},
            "field_status": {},
            "closure": {},
            "axial_rows": [],
            "raman_rows": [],
            "npz_sha256": safe_hash(npz_path),
            "config_sha256": safe_hash(config_path),
            "expected_execution_sha": expected_execution_sha,
            "provenance_binding": {},
        }


def write_audit(result: dict[str, Any], out_dir: Path, *, npz_path: Path, config_path: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # CSVs are the immutable comparison inputs.  Write and hash them before
    # publishing the reaudit JSON so the JSON can be consumed directly by the
    # comparison contract without reconstructing paths from a run directory.
    axial_path = out_dir / "isaacs_complete_eq27_axial_diagnostics.csv"
    extras_path = out_dir / "isaacs_complete_eq27_raman_extras.csv"
    if result["axial_rows"]:
        fields = list(result["axial_rows"][0])
        with axial_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result["axial_rows"])
    if result["raman_rows"]:
        fields = list(result["raman_rows"][0])
        with extras_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result["raman_rows"])

    artifacts: dict[str, dict[str, str]] = {}
    if axial_path.is_file():
        artifacts["axial"] = {"path": str(axial_path.resolve()), "sha256": sha256(axial_path)}
    if extras_path.is_file():
        artifacts["extras"] = {"path": str(extras_path.resolve()), "sha256": sha256(extras_path)}
    metadata = dict(result.get("metadata") or {})
    binding = dict(result.get("provenance_binding") or {})
    job_id = str(metadata.get("slurm_job_id") or "").strip()
    run_status = str(metadata.get("status") or "").strip()
    gate = "passed" if result["passed"] else "failed"
    numerical_admission = "passed" if result["passed"] else "failed"
    operator_state = "applied" if result["passed"] else "not_admitted"
    payload = {
        "schema": "khz_filament.isaacs_complete_eq27.c2_postprocess.v1",
        "job_id": job_id,
        "status": run_status,
        "gate": "passed" if result["passed"] else "failed",
        "audit_status": gate,
        "case_id": "complete_eq27",
        "role": "candidate_complete_eq27",
        "operator_mode": COMPLETE_MODE,
        "operator_state": operator_state,
        "operator": {"mode": COMPLETE_MODE, "state": operator_state},
        "provenance_class": "candidate_execution_verified",
        "provenance": {
            "class": "candidate_execution_verified",
            "execution_git_sha": execution_git_sha(metadata),
            "expected_execution_sha": result.get("expected_execution_sha"),
            "run_status": run_status,
        },
        "numerical_admission": numerical_admission,
        "numerical_admission_detail": {
            "status": numerical_admission,
            "passed": bool(result["passed"]),
            "failures": list(result["failures"]),
        },
        "artifacts": artifacts,
        "npz_path": str(npz_path),
        "config_path": str(config_path),
        "npz_sha256": result["npz_sha256"],
        "config_sha256": result["config_sha256"],
        "raw_source": {
            "generated_from_raw_npz": True,
            "npz": binding.get("npz", {"path": str(npz_path.resolve()), "sha256": result["npz_sha256"]}),
            "metadata": binding.get("metadata", {
                "path": str(metadata.get("metadata_path") or ""),
                "sha256": str(metadata.get("metadata_sha256") or ""),
            }),
            "config": binding.get("config", {"path": str(config_path.resolve()), "sha256": result["config_sha256"]}),
            "manifest": binding.get("manifest", {
                "path": str(metadata.get("manifest_path") or ""),
                "sha256": str(metadata.get("manifest_sha256") or ""),
            }),
            "execution_lock": binding.get("execution_lock", {
                "path": str(metadata.get("execution_lock_path") or ""),
                "sha256": str(metadata.get("execution_lock_sha256") or ""),
            }),
            "submission_lock": binding.get("submission_lock", {
                "path": str(metadata.get("submission_lock") or ""),
                "sha256": str(metadata.get("submission_lock_sha256") or ""),
            }),
        "global_consumed_lock": binding.get("global_consumed_lock", {
            "path": str(metadata.get("global_consumed_lock") or ""),
            "sha256": str(metadata.get("global_consumed_lock_sha256") or ""),
        }),
        "job_receipt": binding.get("job_receipt", {
            "path": str(metadata.get("job_receipt_path") or ""),
            "sha256": str(metadata.get("job_receipt_sha256") or ""),
        }),
            "campaign_id": CAMPAIGN_ID,
            "remote_campaign_root": REMOTE_CAMPAIGN_ROOT,
            "operator_mode": COMPLETE_MODE,
            "use_raman_full_operator": True,
            "job_id": job_id,
            "gpu_model": metadata.get("gpu_model"),
            "propagation_invocations": metadata.get("propagation_invocations"),
        },
        "run_metadata": result["metadata"],
        "semantic_values": result["semantic_values"],
        "field_status": result["field_status"],
        "raman_closure": result["closure"],
        "failures": result["failures"],
    }
    (out_dir / "isaacs_complete_eq27_reaudit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Complete Isaacs Eq. (27) candidate postprocess",
        "",
        f"Gate: **{payload['gate']}**.",
        "",
        "The candidate is required to use `full_isaacs_eq27_complete`, with the full complex Eq. (27) electronic and rotational RHS, no legacy Raman absorption, and fixed x_focus_cm = 100 * (z_m - 0.95).",
        "",
    ]
    if result["failures"]:
        lines.extend(["## Failures", "", *[f"- {item}" for item in result["failures"]]])
    else:
        lines.extend([
            "## Checks",
            "",
            "- Complete operator mode and semantic strings passed.",
            "- Operator-applied, energy, adaptive-step, safety, dz, finite-value, and Raman-closure checks passed.",
            "- Raw NPZ remains outside the repository; only CSV and audit artifacts are written here.",
        ])
    (out_dir / "isaacs_complete_eq27_reaudit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-metadata", type=Path, default=None)
    parser.add_argument("--expected-execution-sha", default=None)
    parser.add_argument("--execution-lock", required=True, type=Path)
    parser.add_argument("--expected-execution-lock-sha256", required=True)
    parser.add_argument("--submission-lock", required=True, type=Path)
    parser.add_argument("--job-receipt", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = validate(
        args.npz, args.config, args.run_metadata,
        expected_execution_sha=args.expected_execution_sha,
        execution_lock_path=args.execution_lock,
        submission_lock_path=args.submission_lock,
        manifest_path=args.manifest,
        expected_execution_lock_sha256=args.expected_execution_lock_sha256,
        job_receipt_path=args.job_receipt,
    )
    payload = write_audit(result, args.out_dir, npz_path=args.npz, config_path=args.config)
    print(f"complete_eq27_gate={payload['gate']}")
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
