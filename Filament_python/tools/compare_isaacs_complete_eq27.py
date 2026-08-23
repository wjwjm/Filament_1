#!/usr/bin/env python3
"""Compare the complete Eq. (27) candidate with fixed C2 comparators.

The comparison coordinate is always ``x_focus_cm = 100 * (z_m - 0.95)``.
Current full Eq. (27) job 180748 and Raman-OFF job 180749 are retained as
explicit fallback provenance; invalid jobs 179706 and 179988 never enter the
physical classification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
from create_isaacs_complete_eq27_execution_lock import (  # noqa: E402
    ExecutionLockError,
    STAGING_PROVENANCE_METHOD,
    STAGING_PROVENANCE_SOURCE_CLASS,
    validate_manifest_lock,
    validate_staging_provenance,
)


RHO_THRESHOLDS = (1e19, 1e20, 1e21, 1e22)
INTENSITY_THRESHOLDS = (1e16, 3e16, 1e17, 3e17, 5e17)
FOCUS_M = 0.95
ONSET_SIGNIFICANCE_CM = 0.1
SUPPORTED_IMPROVEMENT_CM = 0.5
RMSE_WORSENING_FRACTION = 0.10
PEAK_POSITION_WORSENING_CM = 0.5
PEAK_DENSITY_REL_ERR = 0.25
INVALID_JOB_IDS = frozenset({"179706", "179988"})
FALLBACK_PROVENANCE_CLASS = "fallback_verified_non_strict"
REQUIRED_SERIES_FIELDS = ("x_focus_cm", "rho_max_z", "I_max_z")
REQUIRED_NUMERICAL_FIELDS = (
    "U_rel_change_z",
    "E_dep_cumulative_z",
    "E_loss_from_input_z",
    "dz_used_z",
    "adaptive_rejection_count_z",
    "safety_mode_trigger_count_z",
)
CASE_LABELS = {
    "current_full_eq27": "Current full Eq.27 (job 180748 fallback, non-strict)",
    "raman_off": "Raman-OFF (job 180749 fallback, non-strict)",
    "candidate_complete_eq27": "Complete Eq.27 candidate",
    "pycap": "PyCAP 120 fs",
}
EXPECTED_GPU_MODEL = "NVIDIA GeForce RTX 5090"
EXPECTED_I_CAP = 1.0e19
JOB_RECEIPT_SCHEMA = "khz_filament.isaacs_complete_eq27.job_receipt.v1"
STAGING_PROVENANCE_SCHEMA = "khz_filament.isaacs_complete_eq27.staging_provenance.v1"
FIXED_CONFIG_PATH = Path(__file__).resolve().parents[1] / "results" / "isaacs_complete_eq27" / "120fs_talebpour_isaacs_complete_eq27.json"
FIXED_PYCAP_PATH = Path(__file__).resolve().parents[1] / "results" / "density_translation_width" / "density_translation_width_20260715_002" / "paper_pycap_120fs.csv"
FIXED_PYCAP_SHA256 = "9b43e75ebc08ccb0a7796829e45c6727b42ab12cd661b9a3d8d235ef89d31461"
FIXED_FALLBACK_RAW = {
    "current_full_eq27": {
        "job_id": "180748", "case": "on",
        "npz_path": "/data/run01/scvi806/user_Wangjimin/phase8c_b_runs/test_a_f0a7b5d5ac10_fallback/on/test_a_on.npz",
        "npz_sha256": "68d846d4815cd8387c7a4c4934b26dfe48bcef77cc9140d2f06d2fa8e929a218",
        "metadata_path": "/data/run01/scvi806/user_Wangjimin/phase8c_b_runs/test_a_f0a7b5d5ac10_fallback/on/test_a_on_job_metadata.json",
        "metadata_sha256": "0b057fed4763bb2719d7b8288e820d30cf4f458b3752632d65a026cf1eee9f21",
        "diagnostic_report_path": "/data/run01/scvi806/user_Wangjimin/phase8c_b_runs/test_a_f0a7b5d5ac10_fallback/on/test_a_on.diagnostic_report.json",
        "diagnostic_report_sha256": "5766a2454818f6ad495353f4abcdc6e0668016db2cdb84b1538c910799d8ed84",
        "config_sha256": "aafec917d06c252617e5bfdd2ce3a73dd276401c271c33380d59e0172055cf78",
    },
    "raman_off": {
        "job_id": "180749", "case": "off",
        "npz_path": "/data/run01/scvi806/user_Wangjimin/phase8c_b_runs/test_a_f0a7b5d5ac10_fallback/off/test_a_off.npz",
        "npz_sha256": "e85b8dbbc0fd20b50f6c8234d3de677119ff46f4acaf459e43b1b8ff5e5dc6f9",
        "metadata_path": "/data/run01/scvi806/user_Wangjimin/phase8c_b_runs/test_a_f0a7b5d5ac10_fallback/off/test_a_off_job_metadata.json",
        "metadata_sha256": "d2bd43c85099a03c2b3f226127829c07b99fc955c486989d443d09c08d21716a",
        "diagnostic_report_path": "/data/run01/scvi806/user_Wangjimin/phase8c_b_runs/test_a_f0a7b5d5ac10_fallback/off/test_a_off.diagnostic_report.json",
        "diagnostic_report_sha256": "17ab54bdef325e399618e12bc45141f8cfb4fe65ae2ed945b4573a3afbcd35a2",
        "config_sha256": "1c1415941d4497a6caaf6a37ee8559bbd8b8b20a9eeee6377a8dbbc7d28f41ef",
    },
}


class InsufficientEvidenceError(ValueError):
    """Raised when the C2 comparison cannot support a physical classification."""

    classification = "insufficient_evidence"

    def __init__(self, failures: str | Iterable[str]):
        if isinstance(failures, str):
            items = [failures]
        else:
            items = [str(item) for item in failures]
        self.failures = tuple(item for item in items if item)
        detail = "; ".join(self.failures) or "comparison evidence gate failed"
        super().__init__(f"{self.classification}: {detail}")


def _normalise_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_audit(path: Path, *, label: str) -> dict[str, Any]:
    if path is None:
        raise InsufficientEvidenceError(f"{label} audit JSON is required")
    if not path.is_file():
        raise InsufficientEvidenceError(f"{label} audit JSON does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InsufficientEvidenceError(f"{label} audit JSON is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise InsufficientEvidenceError(f"{label} audit JSON must be an object")
    return payload


def _read_raw_metadata(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InsufficientEvidenceError(f"{label} raw job metadata is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise InsufficientEvidenceError(f"{label} raw job metadata must be an object")
    return payload


def _scheduler_evidence(job_id: str) -> dict[str, str]:
    """Query Slurm again at comparison time; an audit's copy is not enough."""
    try:
        completed = subprocess.run(
            [
                "sacct", "-j", job_id, "-X", "-n", "-P",
                "-o", "JobID,State,ExitCode,Elapsed,NodeList,Submit,Start,End",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InsufficientEvidenceError(f"cannot query live Slurm evidence for {job_id}: {exc}") from exc
    rows = [line.strip().split("|") for line in completed.stdout.splitlines() if line.strip()]
    matches = [row for row in rows if len(row) >= 8 and row[0] == job_id]
    if len(matches) != 1:
        raise InsufficientEvidenceError(f"live Slurm evidence for {job_id} is missing or ambiguous")
    row = matches[0]
    if row[1] != "COMPLETED" or row[2] != "0:0":
        raise InsufficientEvidenceError(
            f"live Slurm job {job_id} is not COMPLETED/0:0: state={row[1]!r} exit={row[2]!r}"
        )
    return {
        "job_id": row[0], "state": row[1], "exit_code": row[2],
        "elapsed": row[3], "node_list": row[4], "submit_time": row[5],
        "start_time": row[6], "end_time": row[7], "source": "live_sacct",
    }


def _same_path(left: Any, right: Path) -> bool:
    try:
        return Path(str(left)).resolve() == right.resolve()
    except (OSError, TypeError, ValueError):
        return False


def _fallback_raw_chain(audit: dict[str, Any], *, label: str, expected_job_id: str) -> list[str]:
    """Require fixed raw NPZ/metadata hashes and identity before CSV use."""
    expected = FIXED_FALLBACK_RAW.get(label)
    if expected is None:
        return []
    source = audit.get("raw_source")
    failures: list[str] = []
    if not isinstance(source, dict) or source.get("generated_from_raw_npz") is not True:
        return [f"{label} audit lacks generated_from_raw_npz raw-source chain"]
    npz_record = source.get("npz")
    metadata_record = source.get("metadata")
    if not isinstance(npz_record, dict) or not isinstance(metadata_record, dict):
        return [f"{label} audit raw-source NPZ/metadata records are missing"]
    npz_path = Path(expected["npz_path"])
    metadata_path = Path(expected["metadata_path"])
    if not _same_path(npz_record.get("path"), npz_path) or npz_record.get("sha256") != expected["npz_sha256"]:
        failures.append(f"{label} audit raw NPZ path/SHA is not the fixed evidence")
    if not _same_path(metadata_record.get("path"), metadata_path) or metadata_record.get("sha256") != expected["metadata_sha256"]:
        failures.append(f"{label} audit metadata path/SHA is not the fixed evidence")
    if not npz_path.is_file() or _sha256(npz_path) != expected["npz_sha256"]:
        failures.append(f"{label} fixed raw NPZ is missing or has the wrong SHA256")
    if not metadata_path.is_file() or _sha256(metadata_path) != expected["metadata_sha256"]:
        failures.append(f"{label} fixed metadata is missing or has the wrong SHA256")
    diagnostic_path = Path(expected["diagnostic_report_path"])
    if not diagnostic_path.is_file() or _sha256(diagnostic_path) != expected["diagnostic_report_sha256"]:
        failures.append(f"{label} fixed diagnostic report is missing or has the wrong SHA256")
    if source.get("job_id") != expected_job_id or source.get("job_id") != expected["job_id"]:
        failures.append(f"{label} raw-source job id is not fixed to {expected_job_id}")
    if source.get("case") != expected["case"]:
        failures.append(f"{label} raw-source case is not fixed to {expected['case']}")
    if source.get("config_sha256") != expected["config_sha256"]:
        failures.append(f"{label} raw-source config SHA is not fixed")
    if source.get("expected_sha") != "f0a7b5d5ac103546bd693378e8f8efb4f07c6c27" or source.get("actual_sha") != "f0a7b5d5ac103546bd693378e8f8efb4f07c6c27":
        failures.append(f"{label} raw-source execution SHA is not fixed")
    if source.get("gpu_model") != EXPECTED_GPU_MODEL:
        failures.append(f"{label} raw-source GPU model is not fixed")
    if failures:
        return failures
    metadata = _read_raw_metadata(metadata_path, label=label)
    for key, expected_value in {
        "schema": "phase8c.full_eq27_raman.test_a.job_metadata.v1",
        "case_id": expected["case"],
        "slurm_job_id": expected_job_id,
        "config_sha256": expected["config_sha256"],
        "gpu_model": EXPECTED_GPU_MODEL,
        "expected_sha": "f0a7b5d5ac103546bd693378e8f8efb4f07c6c27",
        "actual_sha": "f0a7b5d5ac103546bd693378e8f8efb4f07c6c27",
    }.items():
        if str(metadata.get(key, "")) != expected_value:
            failures.append(f"{label} raw metadata {key} does not match fixed evidence")
    for key in ("sha_match", "git_status_clean"):
        if metadata.get(key) is not True:
            failures.append(f"{label} raw metadata {key} is not true")
    if str(source.get("config_path", "")).strip() != str(metadata.get("config_path", "")).strip():
        failures.append(f"{label} raw-source config path does not match metadata")
    scheduler = source.get("scheduler_evidence")
    if not isinstance(scheduler, dict) or scheduler.get("job_id") != expected_job_id or scheduler.get("state") != "COMPLETED" or scheduler.get("exit_code") != "0:0":
        failures.append(f"{label} scheduler evidence does not match fixed COMPLETED 0:0 record")
    report = scheduler.get("diagnostic_report") if isinstance(scheduler, dict) else None
    if not isinstance(report, dict) or not _same_path(report.get("path"), diagnostic_path) or report.get("sha256") != expected["diagnostic_report_sha256"]:
        failures.append(f"{label} scheduler diagnostic report binding is not fixed")
    if not failures:
        try:
            live = _scheduler_evidence(expected_job_id)
        except InsufficientEvidenceError as exc:
            failures.append(str(exc))
        else:
            for key in ("job_id", "state", "exit_code"):
                if scheduler.get(key) != live.get(key):
                    failures.append(f"{label} audit scheduler {key} does not match live sacct")
            if scheduler.get("source") != "live_sacct":
                failures.append(f"{label} audit scheduler source is not live_sacct")
    return failures


def _derived_csv_failures(
    audit: dict[str, Any], *, role: str, axial: Path | None, extras: Path | None,
) -> list[str]:
    """Re-derive fixed fallback CSVs from raw NPZ to reject CSV masquerading."""
    expected = FIXED_FALLBACK_RAW[role]
    extractor_path = Path(__file__).with_name("prepare_isaacs_eq27_fallback_comparator_audit.py")
    spec = importlib.util.spec_from_file_location("_c2_fallback_extractor_for_compare", extractor_path)
    if spec is None or spec.loader is None:
        return [f"{role} fixed raw extractor cannot be loaded"]
    extractor = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(extractor)
        raw = extractor._load_raw_npz(Path(expected["npz_path"]))
        with tempfile.TemporaryDirectory(prefix="c2-fallback-compare-") as temporary:
            derived_axial, derived_extras = extractor._derived_artifacts(role, raw, Path(temporary))
            expected_axial_sha = _sha256(derived_axial)
            expected_extras_sha = _sha256(derived_extras)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"{role} fixed raw-to-CSV derivation failed: {exc}"]
    failures: list[str] = []
    artifact_records = audit.get("artifacts")
    if not isinstance(artifact_records, dict):
        return [f"{role} audit artifacts are missing"]
    for artifact_name, supplied, expected_sha in (
        ("axial", axial, expected_axial_sha), ("extras", extras, expected_extras_sha),
    ):
        record = artifact_records.get(artifact_name)
        if not isinstance(record, dict) or record.get("sha256") != expected_sha:
            failures.append(f"{role} audit {artifact_name} is not the fixed raw-NPZ-derived CSV")
        if supplied is None or not supplied.is_file():
            failures.append(f"{role} {artifact_name} CSV is missing")
        else:
            try:
                actual = _sha256(supplied)
            except OSError as exc:
                failures.append(f"{role} {artifact_name} CSV cannot be hashed: {exc}")
            else:
                if actual != expected_sha:
                    failures.append(f"{role} {artifact_name} CSV is not the fixed raw-NPZ-derived artifact")
    return failures


def _candidate_raw_chain(
    audit: dict[str, Any],
    *,
    audit_path: Path,
    expected_job_id: str,
    axial: Path | None,
    extras: Path | None,
) -> list[str]:
    """Re-open the candidate raw chain and re-derive its CSVs before use."""
    failures: list[str] = []
    source = audit.get("raw_source")
    if not isinstance(source, dict) or source.get("generated_from_raw_npz") is not True:
        return ["candidate audit lacks generated_from_raw_npz raw-source chain"]
    if audit.get("provenance_class") != STAGING_PROVENANCE_SOURCE_CLASS:
        return ["candidate audit provenance_class is not verified_bundle_non_strict"]
    provenance = audit.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("class") != STAGING_PROVENANCE_SOURCE_CLASS:
        return ["candidate audit provenance class is not verified_bundle_non_strict"]
    if source.get("job_id") != expected_job_id:
        failures = ["candidate raw-source job id does not match the audit"]
    else:
        failures = []
    if source.get("operator_mode") != "full_isaacs_eq27_complete":
        failures.append("candidate raw-source operator_mode is not complete")
    if source.get("use_raman_full_operator") is not True:
        failures.append("candidate raw-source use_raman_full_operator is not true")
    if source.get("gpu_model") != EXPECTED_GPU_MODEL:
        failures.append("candidate raw-source GPU model is not fixed")
    if source.get("provenance_class") != STAGING_PROVENANCE_SOURCE_CLASS:
        failures.append("candidate raw-source provenance_class is not verified_bundle_non_strict")
    records: dict[str, dict[str, Any]] = {}
    for name in (
        "npz", "metadata", "config", "manifest", "execution_lock", "submission_lock",
        "global_consumed_lock", "job_receipt", "staging_provenance",
    ):
        value = source.get(name)
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("path"), str)
            or not value.get("path", "").strip()
            or not isinstance(value.get("sha256"), str)
            or not value.get("sha256", "").strip()
        ):
            failures.append(f"candidate raw-source {name} path/SHA record is missing")
        else:
            records[name] = value
    if failures:
        return failures
    paths = {name: Path(value["path"]).resolve() for name, value in records.items()}
    for name, record in records.items():
        path = paths[name]
        if not path.is_file():
            failures.append(f"candidate raw-source {name} does not exist: {path}")
            continue
        try:
            actual = _sha256(path)
        except OSError as exc:
            failures.append(f"candidate raw-source {name} cannot be hashed: {exc}")
        else:
            if actual != record["sha256"]:
                failures.append(f"candidate raw-source {name} SHA256 does not match its audit")
    if paths["npz"] != Path(str(audit.get("npz_path", ""))).resolve():
        failures.append("candidate audit npz_path does not match raw-source NPZ")
    if paths["config"] != (Path(__file__).resolve().parents[1] / "results" / "isaacs_complete_eq27" / "120fs_talebpour_isaacs_complete_eq27.json").resolve():
        failures.append("candidate raw-source config path is not the fixed C2 config")
    fixed_manifest = (Path(__file__).resolve().parents[1] / "results" / "isaacs_complete_eq27" / "submission_manifest.json").resolve()
    if paths["manifest"] != fixed_manifest:
        failures.append("candidate raw-source manifest path is not fixed")
    if paths["submission_lock"] != paths["npz"].parent / "SUBMISSION_LOCK":
        failures.append("candidate raw-source submission lock is not RUN_DIR/SUBMISSION_LOCK")
    if paths["job_receipt"] != paths["npz"].parent / "job_receipt.json":
        failures.append("candidate raw-source job receipt is not RUN_DIR/job_receipt.json")
    expected_global_record = Path("/data/run01/scvi806/user_Wangjimin/isaacs_complete_eq27_c2/.consumed.lock/submission_record.txt")
    if expected_global_record.anchor == "/" and paths["global_consumed_lock"] != expected_global_record:
        failures.append("candidate raw-source global consumed record path is not fixed")
    try:
        metadata = _read_raw_metadata(paths["metadata"], label="complete Eq.27 candidate")
    except InsufficientEvidenceError as exc:
        return [str(exc)]
    expected_sha = str(metadata.get("execution_git_sha") or "").strip()
    staging: dict[str, Any] | None = None
    try:
        staging = validate_staging_provenance(
            paths["staging_provenance"],
            expected_sha256=records["staging_provenance"]["sha256"],
            expected_git_sha=expected_sha,
            repo=Path(__file__).resolve().parents[2],
        )
    except (ExecutionLockError, OSError, ValueError, KeyError, TypeError) as exc:
        failures.append(f"candidate raw-source staging provenance validation failed: {exc}")
    else:
        if staging["schema"] != STAGING_PROVENANCE_SCHEMA:
            failures.append("candidate raw-source staging provenance schema is invalid")
        if staging["method"] != STAGING_PROVENANCE_METHOD:
            failures.append("candidate raw-source staging provenance method is not fixed")
        if staging["source_class"] != STAGING_PROVENANCE_SOURCE_CLASS:
            failures.append("candidate raw-source staging provenance source_class is not fixed")
        if source.get("staging_provenance_method") != staging["method"]:
            failures.append("candidate raw-source staging provenance method does not match the file")
        if source.get("staging_provenance_source_class") != staging["source_class"]:
            failures.append("candidate raw-source staging provenance source_class does not match the file")
        if source.get("method") != staging["method"]:
            failures.append("candidate raw-source method does not match staging provenance")
        if source.get("source_class") != staging["source_class"]:
            failures.append("candidate raw-source source_class does not match staging provenance")
        if source.get("staging_provenance_branch") != staging["branch"]:
            failures.append("candidate raw-source staging provenance branch does not match the file")
    if str(metadata.get("slurm_job_id") or "").strip() != expected_job_id:
        failures.append("candidate raw metadata job id does not match the audit")
    if metadata.get("status") != "completed" or metadata.get("exit_code") != 0:
        failures.append("candidate raw metadata is not completed with exit code 0")
    if metadata.get("propagation_invocations") != 1:
        failures.append("candidate raw metadata propagation_invocations is not exactly 1")
    if metadata.get("gpu_model") != EXPECTED_GPU_MODEL:
        failures.append("candidate raw metadata GPU model is not fixed")
    if metadata.get("operator_mode") != "full_isaacs_eq27_complete":
        failures.append("candidate raw metadata operator_mode is not complete")
    if metadata.get("use_raman_full_operator") is not True:
        failures.append("candidate raw metadata use_raman_full_operator is not true")
    if str(metadata.get("npz_sha256") or "") != records["npz"]["sha256"]:
        failures.append("candidate raw metadata npz_sha256 does not match the raw NPZ")
    if str(metadata.get("config_sha256") or "") != records["config"]["sha256"]:
        failures.append("candidate raw metadata config_sha256 does not match the config")
    if metadata.get("execution_lock_path") != records["execution_lock"]["path"]:
        failures.append("candidate raw metadata execution_lock_path does not match the raw chain")
    if metadata.get("execution_lock_sha256") != records["execution_lock"]["sha256"]:
        failures.append("candidate raw metadata execution_lock_sha256 does not match the raw chain")
    if metadata.get("manifest_path") != records["manifest"]["path"] or metadata.get("manifest_sha256") != records["manifest"]["sha256"]:
        failures.append("candidate raw metadata manifest binding does not match the raw chain")
    if metadata.get("staging_provenance_path") != records["staging_provenance"]["path"]:
        failures.append("candidate raw metadata staging provenance path does not match the raw chain")
    if str(metadata.get("staging_provenance_sha256") or "").strip().lower() != records["staging_provenance"]["sha256"]:
        failures.append("candidate raw metadata staging provenance SHA does not match the raw chain")
    if metadata.get("staging_provenance_method") != STAGING_PROVENANCE_METHOD:
        failures.append("candidate raw metadata staging provenance method is not fixed")
    if metadata.get("staging_provenance_source_class") != STAGING_PROVENANCE_SOURCE_CLASS:
        failures.append("candidate raw metadata staging provenance source_class is not fixed")
    if metadata.get("method") != STAGING_PROVENANCE_METHOD:
        failures.append("candidate raw metadata method is not fixed")
    if metadata.get("source_class") != STAGING_PROVENANCE_SOURCE_CLASS:
        failures.append("candidate raw metadata source_class is not fixed")
    if staging is not None and metadata.get("staging_provenance_branch") != staging["branch"]:
        failures.append("candidate raw metadata staging provenance branch does not match the file")
    try:
        receipt = _read_raw_metadata(paths["job_receipt"], label="candidate held-job receipt")
    except InsufficientEvidenceError as exc:
        return [str(exc)]
    receipt_job = str(receipt.get("job_id") or "").strip()
    metadata_job = str(metadata.get("slurm_job_id") or "").strip()
    if receipt.get("schema") != JOB_RECEIPT_SCHEMA or receipt.get("state") != "held":
        failures.append("candidate held-job receipt schema/state is invalid")
    if receipt_job != metadata_job or receipt_job != expected_job_id or source.get("job_id") != receipt_job:
        failures.append("candidate job id must match receipt, metadata, raw source, and audit")
    receipt_token = str(receipt.get("reservation_token") or "").strip()
    if not receipt_token:
        failures.append("candidate held-job receipt reservation_token is empty")
    try:
        submission_text = paths["submission_lock"].read_text(encoding="utf-8")
        global_text = paths["global_consumed_lock"].read_text(encoding="utf-8")
    except OSError as exc:
        failures.append(f"candidate submission/global records cannot be read for receipt binding: {exc}")
        submission_text = global_text = ""
    submission_token = next((line.split("=", 1)[1].strip() for line in submission_text.splitlines() if line.startswith("reservation_token=")), "")
    global_token = next((line.split("=", 1)[1].strip() for line in global_text.splitlines() if line.startswith("reservation_token=")), "")
    if any(line.startswith("job_id=") for line in (*submission_text.splitlines(), *global_text.splitlines())):
        failures.append("candidate submission/global records were edited with a post-sbatch job id")
    if submission_token != receipt_token or global_token != receipt_token:
        failures.append("candidate receipt reservation_token does not match submission/global records")
    if metadata.get("job_receipt_path") != records["job_receipt"]["path"] or metadata.get("job_receipt_sha256") != records["job_receipt"]["sha256"]:
        failures.append("candidate raw metadata job receipt binding does not match the raw chain")
    if source.get("staging_provenance") != records["staging_provenance"]:
        failures.append("candidate raw-source staging provenance record does not match the raw chain")
    for key, expected in {
        "campaign_id": "isaacs_complete_eq27_c2",
        "remote_campaign_root": "/data/run01/scvi806/user_Wangjimin/isaacs_complete_eq27_c2",
        "run_dir": str(paths["npz"].parent),
        "manifest_path": records["manifest"]["path"],
        "manifest_sha256": records["manifest"]["sha256"],
        "execution_lock_path": records["execution_lock"]["path"],
        "execution_lock_sha256": records["execution_lock"]["sha256"],
        "config_path": records["config"]["path"],
        "config_sha256": records["config"]["sha256"],
        "expected_git_sha": expected_sha,
        "staging_provenance_path": records["staging_provenance"]["path"],
        "staging_provenance_sha256": records["staging_provenance"]["sha256"],
        "staging_provenance_method": STAGING_PROVENANCE_METHOD,
        "staging_provenance_source_class": STAGING_PROVENANCE_SOURCE_CLASS,
        "staging_provenance_branch": str(metadata.get("staging_provenance_branch") or ""),
        "method": STAGING_PROVENANCE_METHOD,
        "source_class": STAGING_PROVENANCE_SOURCE_CLASS,
    }.items():
        if receipt.get(key) != expected:
            failures.append(f"candidate held-job receipt {key} is not fixed")
    try:
        locked = validate_manifest_lock(
            paths["manifest"],
            paths["execution_lock"],
            expected_manifest_sha256=records["manifest"]["sha256"],
            expected_lock_sha256=records["execution_lock"]["sha256"],
            expected_git_sha=expected_sha,
            require_clean=True,
            require_committed_manifest=True,
        )
    except Exception as exc:
        failures.append(f"shared manifest/C1 execution-lock validation failed: {exc}")
    else:
        if locked["config_path"].resolve() != paths["config"]:
            failures.append("shared manifest validator config path does not match candidate raw chain")
    try:
        lock = _read_raw_metadata(paths["execution_lock"], label="candidate execution lock")
    except InsufficientEvidenceError as exc:
        return [str(exc)]
    for key, expected in {
        "schema": "khz_filament.isaacs_complete_eq27.c2_execution_lock.v1",
        "campaign_id": "isaacs_complete_eq27_c2",
        "remote_campaign_root": "/data/run01/scvi806/user_Wangjimin/isaacs_complete_eq27_c2",
        "status": "authorized_not_consumed",
        "expected_gpu_model": EXPECTED_GPU_MODEL,
        "operator_mode": "full_isaacs_eq27_complete",
        "use_raman_full_operator": True,
        "config_path": "results/isaacs_complete_eq27/120fs_talebpour_isaacs_complete_eq27.json",
        "derived_config_path": "results/isaacs_complete_eq27/120fs_talebpour_isaacs_complete_eq27.json",
        "manifest_path": "Filament_python/results/isaacs_complete_eq27/submission_manifest.json",
        "config_sha256": records["config"]["sha256"],
        "manifest_sha256": records["manifest"]["sha256"],
    }.items():
        if lock.get(key) != expected:
            failures.append(f"candidate execution lock {key} is not fixed")
    if lock.get("expected_git_sha") != expected_sha:
        failures.append("candidate execution lock expected_git_sha does not match metadata")
    if staging is not None and staging["expected_git_sha"] != lock.get("expected_git_sha"):
        failures.append("candidate staging provenance expected_git_sha does not match execution lock")
    try:
        manifest = _read_raw_metadata(paths["manifest"], label="candidate C2 manifest")
    except InsufficientEvidenceError as exc:
        return [str(exc)]
    if manifest.get("campaign_id") != "isaacs_complete_eq27_c2" or manifest.get("derived_config_sha256") != records["config"]["sha256"]:
        failures.append("candidate C2 manifest does not bind the fixed config/campaign")
    try:
        config_payload = json.loads(paths["config"].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        failures.append(f"candidate fixed config is unreadable: {exc}")
        config_payload = {}
    if config_payload.get("raman", {}).get("operator_mode") != "full_isaacs_eq27_complete":
        failures.append("candidate fixed config operator_mode is not complete")
    if config_payload.get("propagation", {}).get("use_raman_full_operator") is not True:
        failures.append("candidate fixed config use_raman_full_operator is not true")
    # SUBMISSION_LOCK is a key=value record rather than JSON.
    submission: dict[str, str] = {}
    try:
        for line in paths["submission_lock"].read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                submission[key.strip()] = value.strip()
    except OSError as exc:
        failures.append(f"candidate submission lock is unreadable: {exc}")
    for key, expected in {
        "case_id": "complete_eq27",
        "campaign_id": "isaacs_complete_eq27_c2",
        "manifest_sha256": records["manifest"]["sha256"],
        "execution_lock_sha256": records["execution_lock"]["sha256"],
        "expected_config_sha256": records["config"]["sha256"],
        "expected_git_sha": expected_sha,
    }.items():
        if submission.get(key) != expected:
            failures.append(f"candidate submission lock {key} is not fixed")
    global_record = {}
    try:
        for line in paths["global_consumed_lock"].read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                global_record[key.strip()] = value.strip()
    except OSError as exc:
        failures.append(f"candidate global consumed record is unreadable: {exc}")
    for key, expected in {
        "campaign_id": "isaacs_complete_eq27_c2",
        "manifest_sha256": records["manifest"]["sha256"],
        "execution_lock_sha256": records["execution_lock"]["sha256"],
        "expected_git_sha": expected_sha,
        "run_dir": str(paths["npz"].parent),
    }.items():
        if global_record.get(key) != expected:
            failures.append(f"candidate global consumed record {key} is not fixed")

    if failures:
        return failures
    extractor_path = Path(__file__).with_name("postprocess_isaacs_complete_eq27.py")
    post = None
    loaded = __import__("sys").modules.get("postprocess_isaacs_complete_eq27")
    if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() == extractor_path.resolve():
        post = loaded
    if post is None:
        spec = importlib.util.spec_from_file_location("_c2_candidate_postprocess_for_compare", extractor_path)
        if spec is None or spec.loader is None:
            return ["candidate postprocess validator cannot be loaded"]
        post = importlib.util.module_from_spec(spec)
    try:
        if loaded is None or post is not loaded:
            spec.loader.exec_module(post)
        result = post.validate(
            paths["npz"], paths["config"], paths["metadata"],
            expected_execution_sha=expected_sha,
            execution_lock_path=paths["execution_lock"],
            submission_lock_path=paths["submission_lock"],
            manifest_path=paths["manifest"],
            expected_execution_lock_sha256=records["execution_lock"]["sha256"],
        )
        if not result["passed"]:
            failures.extend(f"candidate postprocess validation: {item}" for item in result["failures"])
        with tempfile.TemporaryDirectory(prefix="c2-candidate-compare-") as temporary:
            post.write_audit(result, Path(temporary), npz_path=paths["npz"], config_path=paths["config"])
            derived_axial = Path(temporary) / "isaacs_complete_eq27_axial_diagnostics.csv"
            derived_extras = Path(temporary) / "isaacs_complete_eq27_raman_extras.csv"
            for name, supplied, derived in (("axial", axial, derived_axial), ("extras", extras, derived_extras)):
                if supplied is None or not supplied.is_file():
                    failures.append(f"candidate {name} CSV is missing")
                elif _sha256(supplied) != _sha256(derived):
                    failures.append(f"candidate {name} CSV is not re-derived from the raw NPZ")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        failures.append(f"candidate raw-to-CSV validation failed: {exc}")
    return failures


def _fixed_pycap_failures(path: Path) -> list[str]:
    if path.resolve() != FIXED_PYCAP_PATH.resolve():
        return ["PyCAP path is not the fixed repository input"]
    if not path.is_file():
        return [f"fixed PyCAP input does not exist: {path}"]
    try:
        actual = _sha256(path)
    except OSError as exc:
        return [f"fixed PyCAP input cannot be hashed: {exc}"]
    return [] if actual == FIXED_PYCAP_SHA256 else [f"fixed PyCAP SHA256 mismatch: expected={FIXED_PYCAP_SHA256} actual={actual}"]


def _fixed_cap_failures() -> list[str]:
    try:
        payload = json.loads(FIXED_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"fixed C2 config cannot be read for ionization.I_cap: {exc}"]
    if payload.get("ionization", {}).get("I_cap") != EXPECTED_I_CAP:
        return [f"fixed C2 config ionization.I_cap is not {EXPECTED_I_CAP:g}"]
    return []


def _case_cap_failures(series: dict[str, dict[str, np.ndarray]]) -> list[str]:
    failures: list[str] = []
    margin = EXPECTED_I_CAP * (1.0 - 1.0e-6)
    for name in ("current_full_eq27", "raman_off", "candidate_complete_eq27"):
        values = np.asarray(series[name].get("I_max_z", []), dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            failures.append(f"{name} I_max_z is missing/nonfinite")
        elif float(np.max(values)) >= margin:
            failures.append(f"{name} I_max_z reaches fixed ionization.I_cap margin")
    return failures


def _audit_gate_failures(audit: dict[str, Any], *, label: str) -> list[str]:
    """Require an explicit passed audit gate/status, not merely run completion."""
    markers: list[tuple[str, bool]] = []
    for key, value in audit.items():
        normalised = _normalise_key(key)
        if normalised == "passed":
            markers.append((key, value is True))
        elif normalised == "status":
            # Candidate job metadata legitimately reports ``completed`` while
            # the postprocess audit reports the independent ``gate``.  A
            # failed/error run remains a hard rejection; completion is not an
            # audit pass by itself and still requires a passed gate marker.
            markers.append((key, str(value).strip().lower() in {"passed", "completed", "complete", "success"}))
        elif normalised in {
            "gate", "status", "audit_status", "validation_status", "comparison_gate",
            "raman_phase_off_gate", "current_full_eq27_gate", "candidate_gate",
        }:
            markers.append((key, str(value).strip().lower() == "passed"))
    failures: list[str] = []
    explicit_markers = [
        (key, passed) for key, passed in markers
        if _normalise_key(key) != "status"
    ]
    if not explicit_markers:
        failures.append(f"{label} audit lacks an explicit passed status/gate")
    elif not any(passed for _, passed in explicit_markers):
        failures.append(f"{label} audit status/gate is not passed")
    elif any(not passed for _, passed in explicit_markers):
        failures.append(f"{label} audit status/gate is not passed")
    admission = audit.get("numerical_admission")
    if admission is not None:
        admission_passed = (
            admission is True
            or str(admission).strip().lower() in {"passed", "pass", "true", "completed"}
            or (isinstance(admission, dict) and admission.get("passed") is True)
        )
        if not admission_passed:
            failures.append(f"{label} numerical admission is not passed")
    for key in ("failures", "errors"):
        value = audit.get(key)
        if value:
            failures.append(f"{label} audit records {key}")
    return failures


def _audit_job_id(audit: dict[str, Any], *, label: str) -> tuple[str | None, list[str]]:
    """Extract one execution job id while ignoring descriptive exclusions."""
    values: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        text = str(value).strip()
        if text:
            values.append(text)

    for key in ("job_id", "slurm_job_id", "scheduler_job_id"):
        if key in audit:
            add(audit[key])
    for container_key in ("run_metadata", "execution", "metadata", "slurm", "job"):
        container = audit.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in ("job_id", "slurm_job_id", "scheduler_job_id"):
            if key in container:
                add(container[key])
        if isinstance(container.get("job"), dict):
            add(container["job"].get("id"))
        add(container.get("id") if container_key in {"job", "slurm"} else None)
    unique = list(dict.fromkeys(values))
    failures: list[str] = []
    if not unique:
        failures.append(f"{label} audit lacks a non-empty job id")
        return None, failures
    if len(unique) != 1:
        failures.append(f"{label} audit has conflicting job ids: {unique}")
        return None, failures
    job_id = unique[0]
    if job_id in INVALID_JOB_IDS:
        failures.append(f"{label} audit uses excluded invalid job {job_id}")
    return job_id, failures


def _artifact_context_matches(context: tuple[str, ...], role: str, path_value: Any = None) -> bool:
    # Provenance blocks may contain their own JSON path/SHA records.  They are
    # not the CSV inputs consumed by this comparison, even when a surrounding
    # role name contains ``raman`` (for example ``raman_off``).
    if path_value is not None and Path(str(path_value)).suffix.lower() != ".csv":
        return False
    text = "_".join(_normalise_key(item) for item in context)
    if role == "axial":
        return "axial" in text and "extra" not in text
    # Do not classify a comparator role such as ``raman_off`` as an extras
    # artifact.  Only an explicit extras/raman_rows context is admissible.
    return "extra" in text or "raman_rows" in text or "raman_extras" in text


def _artifact_records(audit: dict[str, Any], *, role: str) -> list[dict[str, str]]:
    """Find path/SHA records in common audit input/artifact layouts."""
    path_keys = {"path", "file", "filename", "csv_path", "source_path"}
    sha_keys = {"sha", "sha256", "file_sha256", "input_sha256", "digest"}
    records: list[dict[str, str]] = []

    def append(path_value: Any, sha_value: Any) -> None:
        if path_value is None or sha_value is None:
            return
        path_text = str(path_value).strip()
        sha_text = str(sha_value).strip().lower()
        if path_text and sha_text:
            records.append({"path": path_text, "sha256": sha_text})

    def walk(value: Any, context: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            path_value = next((value[key] for key in value if _normalise_key(key) in path_keys), None)
            sha_value = next((value[key] for key in value if _normalise_key(key) in sha_keys), None)
            semantic_context = list(context)
            for semantic_key in ("label", "name", "role", "kind", "type", "case"):
                for key, child in value.items():
                    if _normalise_key(key) == semantic_key and isinstance(child, str):
                        semantic_context.append(child)
            effective_context = tuple(semantic_context)
            if path_value is not None and sha_value is not None and _artifact_context_matches(effective_context, role, path_value):
                append(path_value, sha_value)
            for key, child in value.items():
                walk(child, (*effective_context, str(key)))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, (*context, str(index)))

    walk(audit, ())

    # Also accept the compact top-level form: axial_path/axial_sha256, etc.
    aliases = ("axial", "axial_diagnostics") if role == "axial" else ("extras", "raman_extras")
    for alias in aliases:
        for path_key in (f"{alias}_path", f"{alias}_csv", f"{alias}_csv_path"):
            for sha_key in (f"{alias}_sha256", f"{alias}_sha"):
                if path_key in audit and sha_key in audit:
                    append(audit[path_key], audit[sha_key])

    # phase-6 style traceability maps path -> sha256; infer the role from the
    # immutable file name while still requiring a path and digest pair.
    traceability = audit.get("traceability")
    if isinstance(traceability, dict):
        digest_map = traceability.get("input_sha256")
        if isinstance(digest_map, dict):
            for path_value, sha_value in digest_map.items():
                if _artifact_context_matches((str(path_value),), role, path_value):
                    append(path_value, sha_value)
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (record["path"], record["sha256"])
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique


def _path_matches(audit_path: Path, recorded: str, supplied: Path) -> bool:
    recorded_path = Path(recorded)
    candidates = [recorded_path]
    if not recorded_path.is_absolute():
        candidates.append(audit_path.parent / recorded_path)
    supplied_resolved = supplied.resolve()
    return any(candidate.resolve() == supplied_resolved for candidate in candidates)


def _verify_input_artifact(
    audit_path: Path,
    audit: dict[str, Any],
    supplied: Path | None,
    *,
    role: str,
    label: str,
) -> list[str]:
    failures: list[str] = []
    records = _artifact_records(audit, role=role)
    if len(records) != 1:
        failures.append(f"{label} audit must contain exactly one {role} input path/SHA record")
        return failures
    if supplied is None:
        failures.append(f"{label} {role} CSV is required and must match the audit")
        return failures
    if not supplied.is_file():
        failures.append(f"{label} {role} CSV does not exist: {supplied}")
        return failures
    record = records[0]
    if not _path_matches(audit_path, record["path"], supplied):
        failures.append(f"{label} {role} CSV path does not match audit")
    digest = record["sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        failures.append(f"{label} {role} audit SHA256 is malformed")
    else:
        try:
            actual = _sha256(supplied)
        except OSError as exc:
            failures.append(f"{label} {role} CSV cannot be hashed: {exc}")
        else:
            if actual != digest:
                failures.append(f"{label} {role} CSV SHA256 does not match audit")
    return failures


def _audit_evidence_failures(audit: dict[str, Any], *, label: str) -> list[str]:
    """Reject explicit failed finite/crossing/overlap/numerical evidence checks."""
    failures: list[str] = []
    relevant = ("finite", "crossing", "overlap", "numerical", "evidence")

    def walk(value: Any, context: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalised = _normalise_key(key)
                if any(token in normalised for token in relevant):
                    if child is False or child is None:
                        failures.append(f"{label} audit marks {'.'.join((*context, str(key)))} unavailable")
                walk(child, (*context, str(key)))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, (*context, str(index)))

    walk(audit, ())
    return failures


def _validate_audit(
    path: Path | None,
    *,
    label: str,
    expected_job_id: str | None,
    axial: Path | None,
    extras: Path | None,
    chain_role: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    if path is None:
        raise InsufficientEvidenceError(f"{label} audit JSON is required")
    audit = _read_audit(path, label=label)
    failures = _audit_gate_failures(audit, label=label)
    job_id, job_failures = _audit_job_id(audit, label=label)
    failures.extend(job_failures)
    if expected_job_id is not None and job_id != expected_job_id:
        failures.append(f"{label} audit job id must be exactly {expected_job_id}")
    if chain_role is not None:
        failures.extend(_fallback_raw_chain(audit, label=chain_role, expected_job_id=expected_job_id or ""))
    failures.extend(_verify_input_artifact(path, audit, axial, role="axial", label=label))
    failures.extend(_verify_input_artifact(path, audit, extras, role="extras", label=label))
    if chain_role is not None:
        failures.extend(_derived_csv_failures(audit, role=chain_role, axial=axial, extras=extras))
    failures.extend(_audit_evidence_failures(audit, label=label))
    if failures:
        raise InsufficientEvidenceError(failures)
    return audit, job_id
CASE_COLORS = {
    "current_full_eq27": "#b91c1c",
    "raman_off": "#475569",
    "candidate_complete_eq27": "#0369a1",
}


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    fields: dict[str, np.ndarray] = {}
    for key in rows[0]:
        values = []
        for row in rows:
            raw = row.get(key, "")
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                values.append(float("nan"))
        fields[key] = np.asarray(values, dtype=float)
    return fields


def _coordinate(fields: dict[str, np.ndarray], *, label: str) -> np.ndarray:
    if "z_m" in fields:
        z = np.asarray(fields["z_m"], dtype=float)
        x = 100.0 * (z - FOCUS_M)
        if "x_focus_cm" in fields:
            supplied = np.asarray(fields["x_focus_cm"], dtype=float)
            if supplied.shape == x.shape and np.any(np.isfinite(supplied)):
                if not np.allclose(supplied, x, rtol=0.0, atol=2e-5, equal_nan=False):
                    raise ValueError(f"{label} x_focus_cm does not use the locked coordinate formula")
        fields["x_focus_cm"] = x
        return z
    if "x_focus_cm" in fields:
        x = np.asarray(fields["x_focus_cm"], dtype=float)
        fields["x_focus_cm"] = x
        return (x / 100.0) + FOCUS_M
    raise ValueError(f"{label} CSV lacks z_m and x_focus_cm")


def _merge(axial_path: Path, extras_path: Path | None, *, label: str) -> dict[str, np.ndarray]:
    axial = _read_csv(axial_path)
    z = _coordinate(axial, label=label)
    if extras_path is not None:
        extras = _read_csv(extras_path)
        ez = _coordinate(extras, label=f"{label} extras")
        for key, values in extras.items():
            if key in ("z_m", "x_focus_cm"):
                continue
            values = np.asarray(values, dtype=float)
            if values.size == z.size and np.allclose(ez, z, rtol=0.0, atol=2e-6, equal_nan=False):
                axial[key] = values
            elif values.size >= 2 and np.all(np.diff(ez) > 0.0):
                axial[key] = np.interp(z, ez, values, left=np.nan, right=np.nan)
            else:
                raise ValueError(f"{label} extras cannot be aligned to axial z axis")
    order = np.argsort(axial["x_focus_cm"])
    return {key: np.asarray(values, dtype=float)[order] for key, values in axial.items()}


def _first_crossing(x: np.ndarray, y: np.ndarray, threshold: float, *, descending: bool = False) -> float | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size != x.size:
        return None
    if descending:
        mask = np.isfinite(y[:-1]) & np.isfinite(y[1:]) & (y[:-1] >= threshold) & (y[1:] < threshold)
    else:
        mask = np.isfinite(y[:-1]) & np.isfinite(y[1:]) & (y[:-1] < threshold) & (y[1:] >= threshold)
    indices = np.flatnonzero(mask)
    if not indices.size:
        return None
    index = int(indices[0])
    y0, y1 = float(y[index]), float(y[index + 1])
    if y1 == y0:
        return float(x[index])
    return float(x[index] + (threshold - y0) * (x[index + 1] - x[index]) / (y1 - y0))


def _crossing_in_slice(x: np.ndarray, y: np.ndarray, threshold: float, *, descending: bool = False) -> float | None:
    return _first_crossing(x, y, threshold, descending=descending)


def _density_metrics(x: np.ndarray, rho: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    rho = np.asarray(rho, dtype=float)
    index = int(np.nanargmax(rho))
    peak = float(rho[index])
    top_mask = rho >= 0.99 * peak
    left, right = index, index
    while left > 0 and top_mask[left - 1]:
        left -= 1
    while right + 1 < rho.size and top_mask[right + 1]:
        right += 1
    half = 0.5 * peak
    ten, ninety = 0.1 * peak, 0.9 * peak
    left_half = _crossing_in_slice(x[: index + 1], rho[: index + 1], half)
    right_half = _crossing_in_slice(x[index:], rho[index:], half, descending=True)
    left_ten = _crossing_in_slice(x[: index + 1], rho[: index + 1], ten)
    left_ninety = _crossing_in_slice(x[: index + 1], rho[: index + 1], ninety)
    right_ninety = _crossing_in_slice(x[index:], rho[index:], ninety, descending=True)
    right_ten = _crossing_in_slice(x[index:], rho[index:], ten, descending=True)
    tail = float(np.trapezoid(np.maximum(rho[index:] - half, 0.0), x[index:]))
    tail5 = float(np.trapezoid(np.maximum(rho[x >= x[index] + 5.0] - half, 0.0), x[x >= x[index] + 5.0])) if np.any(x >= x[index] + 5.0) else None
    tail10 = float(np.trapezoid(np.maximum(rho[x >= x[index] + 10.0] - half, 0.0), x[x >= x[index] + 10.0])) if np.any(x >= x[index] + 10.0) else None
    crossings = {str(int(threshold)): _first_crossing(x, rho, threshold) for threshold in RHO_THRESHOLDS}
    return {
        "rho_peak_m3": peak,
        "peak_x_cm": float(x[index]),
        "peak_top_center_cm": float((x[left] + x[right]) / 2.0),
        "fwhm_cm": None if left_half is None or right_half is None else float(right_half - left_half),
        "rise_10_90_cm": None if left_ten is None or left_ninety is None else float(left_ninety - left_ten),
        "fall_90_10_cm": None if right_ninety is None or right_ten is None else float(right_ten - right_ninety),
        "post_peak_tail_area_above_half_m3_cm": tail,
        "post_peak_plus_5cm_tail_area_above_half_m3_cm": tail5,
        "post_peak_plus_10cm_tail_area_above_half_m3_cm": tail10,
        "post_peak_half_distance_cm": None if right_half is None else float(right_half - x[index]),
        "crossings": crossings,
        "intensity_peak_x_cm": None,
    }


def _rmse(x: np.ndarray, y: np.ndarray, px: np.ndarray, py: np.ndarray) -> float | None:
    lo, hi = max(float(np.min(x)), float(np.min(px))), min(float(np.max(x)), float(np.max(px)))
    if not lo < hi:
        return None
    step_candidates = [np.median(np.diff(x)), np.median(np.diff(px))]
    step = max(float(min(v for v in step_candidates if v > 0.0)), 1e-6)
    grid = np.arange(lo, hi + 0.5 * step, step)
    if grid.size < 2:
        return None
    return float(np.sqrt(np.mean((np.interp(grid, x, y) - np.interp(grid, px, py)) ** 2)))


def _finite_case(data: dict[str, np.ndarray]) -> bool:
    for key in (*REQUIRED_SERIES_FIELDS, *REQUIRED_NUMERICAL_FIELDS):
        if key not in data or data[key].size == 0 or not np.all(np.isfinite(data[key])):
            return False
    return True


def _series_evidence_failures(data: dict[str, np.ndarray], *, label: str) -> list[str]:
    """Check the data-level evidence needed before applying C2 thresholds."""
    failures: list[str] = []
    for key in REQUIRED_SERIES_FIELDS:
        if key not in data:
            failures.append(f"{label} is missing {key}")
            continue
        values = np.asarray(data[key], dtype=float)
        if values.ndim != 1 or values.size < 2:
            failures.append(f"{label} {key} has fewer than two aligned records")
        elif not np.all(np.isfinite(values)):
            failures.append(f"{label} {key} contains NaN/Inf")
    x = np.asarray(data.get("x_focus_cm", []), dtype=float)
    if x.size >= 2 and np.any(np.diff(x) <= 0.0):
        failures.append(f"{label} x_focus_cm is not strictly increasing")
    for key in REQUIRED_NUMERICAL_FIELDS:
        if key not in data:
            failures.append(f"{label} lacks numerical evidence field {key}")
            continue
        values = np.asarray(data[key], dtype=float)
        if values.size != x.size:
            failures.append(f"{label} numerical evidence field {key} is not x-aligned")
        elif not np.all(np.isfinite(values)):
            failures.append(f"{label} numerical evidence field {key} contains NaN/Inf")
    for key in ("rho_max_z", "I_max_z"):
        values = np.asarray(data.get(key, []), dtype=float)
        if values.size and np.all(np.isfinite(values)) and np.any(values < 0.0):
            failures.append(f"{label} {key} contains negative values")
    return failures


def _all_fields_finite(data: dict[str, np.ndarray], *, label: str) -> list[str]:
    failures: list[str] = []
    for key, values in data.items():
        values = np.asarray(values, dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            failures.append(f"{label} field {key} contains missing/nonfinite evidence")
    return failures


def _overlap_failures(series: dict[str, dict[str, np.ndarray]], px: np.ndarray) -> list[str]:
    axes = [np.asarray(data["x_focus_cm"], dtype=float) for data in series.values()]
    axes.append(np.asarray(px, dtype=float))
    lo = max(float(np.min(axis)) for axis in axes)
    hi = min(float(np.max(axis)) for axis in axes)
    if not lo < hi:
        return ["series and PyCAP x_focus_cm domains have no common overlap"]
    failures: list[str] = []
    for name, data in series.items():
        case_lo = max(float(np.min(data["x_focus_cm"])), float(np.min(px)))
        case_hi = min(float(np.max(data["x_focus_cm"])), float(np.max(px)))
        if not case_lo < case_hi:
            failures.append(f"{name} has no overlap with PyCAP x_focus_cm domain")
    return failures


def classify(
    *,
    shift_abs_cm: float | None = None,
    candidate_current_shift_cm: float | None = None,
    onset_improvement_cm: float | None = None,
    candidate_peak_density_rel_error: float | None = None,
    current_rmse: float | None = None,
    candidate_rmse: float | None = None,
    candidate_peak_position_error_cm: float | None = None,
    current_peak_position_error_cm: float | None = None,
) -> str:
    """Apply the explicit A/B/C C2 thresholds.

    ``shift_abs_cm`` and ``candidate_current_shift_cm`` are accepted as
    aliases to keep the helper convenient for focused tests and reports.
    """
    if shift_abs_cm is None and candidate_current_shift_cm is not None:
        shift_abs_cm = abs(float(candidate_current_shift_cm))
    values = (
        shift_abs_cm, onset_improvement_cm, candidate_peak_density_rel_error,
        current_rmse, candidate_rmse, candidate_peak_position_error_cm,
        current_peak_position_error_cm,
    )
    if any(value is None or not math.isfinite(float(value)) for value in values):
        return "insufficient_evidence"
    shift_abs_cm = float(shift_abs_cm)
    improvement = float(onset_improvement_cm)
    peak_error = float(candidate_peak_density_rel_error)
    current_rmse = float(current_rmse)
    candidate_rmse = float(candidate_rmse)
    candidate_position = float(candidate_peak_position_error_cm)
    current_position = float(current_peak_position_error_cm)
    if shift_abs_cm < ONSET_SIGNIFICANCE_CM or improvement < ONSET_SIGNIFICANCE_CM:
        return "electronic_eq27_operator_not_supported"
    if (
        improvement >= SUPPORTED_IMPROVEMENT_CM
        and peak_error <= PEAK_DENSITY_REL_ERR
        and candidate_rmse <= current_rmse * (1.0 + RMSE_WORSENING_FRACTION)
        and candidate_position <= current_position + PEAK_POSITION_WORSENING_CM
    ):
        return "electronic_eq27_operator_supported"
    if improvement >= ONSET_SIGNIFICANCE_CM:
        return "electronic_eq27_operator_partial"
    return "electronic_eq27_operator_not_supported"


classify_operator = classify


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot(out_dir: Path, series: dict[str, dict[str, np.ndarray]], px: np.ndarray, py: np.ndarray) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = tuple(name for name in ("current_full_eq27", "raman_off", "candidate_complete_eq27") if name in series)
    figures: list[str] = []
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for name in names:
        ax.semilogy(series[name]["x_focus_cm"], series[name]["rho_max_z"], label=CASE_LABELS[name], color=CASE_COLORS[name])
    ax.semilogy(px, py, "k--", label=CASE_LABELS["pycap"])
    ax.set(xlabel="x relative to focus (cm)", ylabel=r"peak electron density (m$^{-3}$)")
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(out_dir / "rho_vs_x.png"); plt.close(fig); figures.append("rho_vs_x.png")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for name in names:
        ax.semilogy(series[name]["x_focus_cm"], series[name]["rho_max_z"], label=CASE_LABELS[name], color=CASE_COLORS[name])
    ax.semilogy(px, py, "k--", label=CASE_LABELS["pycap"])
    ax.axhline(1e22, color="#111827", linewidth=0.8, linestyle=":")
    ax.set(xlabel="x relative to focus (cm)", ylabel=r"peak electron density (m$^{-3}$)")
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(out_dir / "onset_zoom_1e22.png"); plt.close(fig); figures.append("onset_zoom_1e22.png")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for name in names:
        ax.semilogy(series[name]["x_focus_cm"], np.maximum(series[name]["I_max_z"], 1e-300), label=CASE_LABELS[name], color=CASE_COLORS[name])
    ax.set(xlabel="x relative to focus (cm)", ylabel=r"I$_{max}$ (W m$^{-2}$)")
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(out_dir / "Imax_vs_x.png"); plt.close(fig); figures.append("Imax_vs_x.png")
    return figures


def compare(
    current_axial: Path,
    current_extras: Path | None,
    raman_off_axial: Path,
    raman_off_extras: Path | None,
    candidate_axial: Path,
    candidate_extras: Path | None,
    pycap_path: Path,
    out_dir: Path,
    current_audit: Path | None = None,
    raman_off_audit: Path | None = None,
    candidate_audit: Path | None = None,
) -> dict[str, Any]:
    # Audit gates and immutable input hashes are checked before any metrics or
    # reports are produced.  An incomplete evidence set must stop the
    # comparison instead of being classified as a physical non-support result.
    current_audit_payload, current_job_id = _validate_audit(
        current_audit,
        label="current full Eq.27",
        expected_job_id="180748",
        axial=current_axial,
        extras=current_extras,
        chain_role="current_full_eq27",
    )
    raman_off_audit_payload, raman_off_job_id = _validate_audit(
        raman_off_audit,
        label="Raman-OFF",
        expected_job_id="180749",
        axial=raman_off_axial,
        extras=raman_off_extras,
        chain_role="raman_off",
    )
    candidate_audit_payload, candidate_job_id = _validate_audit(
        candidate_audit,
        label="complete Eq.27 candidate",
        expected_job_id=None,
        axial=candidate_axial,
        extras=candidate_extras,
    )
    if candidate_job_id is None:
        raise InsufficientEvidenceError("complete Eq.27 candidate audit lacks a fixed raw execution chain")
    candidate_chain_failures = _candidate_raw_chain(
        candidate_audit_payload,
        audit_path=Path(candidate_audit),
        expected_job_id=candidate_job_id,
        axial=candidate_axial,
        extras=candidate_extras,
    )
    if candidate_chain_failures:
        raise InsufficientEvidenceError(candidate_chain_failures)

    if candidate_job_id in INVALID_JOB_IDS:
        # _validate_audit already records this, but retain an explicit guard
        # at the comparison boundary for future callers that bypass helpers.
        raise InsufficientEvidenceError(
            f"complete Eq.27 candidate audit uses excluded invalid job {candidate_job_id}"
        )

    try:
        series = {
            "current_full_eq27": _merge(current_axial, current_extras, label="current full Eq.27"),
            "raman_off": _merge(raman_off_axial, raman_off_extras, label="Raman-OFF"),
            "candidate_complete_eq27": _merge(candidate_axial, candidate_extras, label="complete Eq.27 candidate"),
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise InsufficientEvidenceError(f"comparison CSV evidence is unreadable: {exc}") from exc
    evidence_failures: list[str] = []
    for name, data in series.items():
        evidence_failures.extend(_all_fields_finite(data, label=name))
        evidence_failures.extend(_series_evidence_failures(data, label=name))

    pycap_path = Path(pycap_path).resolve()
    pycap_path_failures = _fixed_pycap_failures(pycap_path)
    if pycap_path_failures:
        raise InsufficientEvidenceError(pycap_path_failures)
    try:
        paper = _read_csv(pycap_path)
        _coordinate(paper, label="PyCAP")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise InsufficientEvidenceError(f"PyCAP CSV evidence is unreadable: {exc}") from exc
    evidence_failures.extend(_all_fields_finite(paper, label="PyCAP"))
    if "x_focus_cm" not in paper or paper["x_focus_cm"].size < 2:
        evidence_failures.append("PyCAP lacks at least two x_focus_cm records")
    elif np.any(np.diff(np.asarray(paper["x_focus_cm"], dtype=float)) <= 0.0):
        evidence_failures.append("PyCAP x_focus_cm is not strictly increasing")
    if "rho_1e16_cm3" in paper:
        pycap_rho = np.asarray(paper["rho_1e16_cm3"], dtype=float) * 1e22
    elif "rho_m3" in paper:
        pycap_rho = np.asarray(paper["rho_m3"], dtype=float)
    else:
        evidence_failures.append("PyCAP CSV lacks rho_1e16_cm3 or rho_m3")
        pycap_rho = np.asarray([])
    px = np.asarray(paper["x_focus_cm"], dtype=float)
    if pycap_rho.size != px.size or pycap_rho.size < 2 or not np.all(np.isfinite(pycap_rho)):
        evidence_failures.append("PyCAP density evidence is missing/nonfinite or x-misaligned")
    elif np.any(pycap_rho < 0.0):
        evidence_failures.append("PyCAP density evidence contains negative values")
    if not evidence_failures:
        evidence_failures.extend(_overlap_failures(series, px))
    if evidence_failures:
        raise InsufficientEvidenceError(evidence_failures)

    pycap = {"x_focus_cm": px, "rho_max_z": pycap_rho, "I_max_z": np.full(px.shape, np.nan)}
    metrics = {name: _density_metrics(data["x_focus_cm"], data["rho_max_z"]) for name, data in series.items()}
    metrics["pycap"] = _density_metrics(px, pycap_rho)
    for name, data in series.items():
        if "I_max_z" in data and data["I_max_z"].size:
            metrics[name]["intensity_peak_x_cm"] = float(data["x_focus_cm"][int(np.nanargmax(data["I_max_z"]))])

    threshold_rows: list[dict[str, Any]] = []
    for threshold in RHO_THRESHOLDS:
        key = str(int(threshold))
        row: dict[str, Any] = {"family": "rho_total", "threshold_m3": threshold}
        for name in (*series.keys(), "pycap"):
            row[f"x_{name}_cm"] = metrics[name]["crossings"][key]
        current_x = row["x_current_full_eq27_cm"]
        candidate_x = row["x_candidate_complete_eq27_cm"]
        pycap_x = row["x_pycap_cm"]
        row["candidate_minus_current_cm"] = None if current_x is None or candidate_x is None else candidate_x - current_x
        row["current_error_to_pycap_cm"] = None if current_x is None or pycap_x is None else abs(current_x - pycap_x)
        row["candidate_error_to_pycap_cm"] = None if candidate_x is None or pycap_x is None else abs(candidate_x - pycap_x)
        row["candidate_onset_improvement_cm"] = None if row["current_error_to_pycap_cm"] is None or row["candidate_error_to_pycap_cm"] is None else row["current_error_to_pycap_cm"] - row["candidate_error_to_pycap_cm"]
        threshold_rows.append(row)

    intensity_rows: list[dict[str, Any]] = []
    for threshold in INTENSITY_THRESHOLDS:
        for name, data in series.items():
            values = np.asarray(data.get("I_max_z", []), dtype=float)
            intensity_rows.append({
                "threshold_W_m2": threshold,
                "case": name,
                "x_crossing_cm": None if values.size == 0 else _first_crossing(data["x_focus_cm"], values, threshold),
            })

    shape_rows = [{"case": name, **values, "rmse_vs_pycap_m3": _rmse(data["x_focus_cm"], data["rho_max_z"], px, pycap_rho) if name != "pycap" else 0.0} for name, (data, values) in [(name, (series[name], metrics[name])) for name in series]]
    shape_rows.append({"case": "pycap", **metrics["pycap"], "rmse_vs_pycap_m3": 0.0})

    numerical_rows: list[dict[str, Any]] = []
    for name, data in series.items():
        finite = _finite_case(data)
        def final_or_none(key: str) -> float | None:
            values = np.asarray(data.get(key, []), dtype=float)
            return None if values.size == 0 else float(values[-1])
        def maxabs_or_none(key: str) -> float | None:
            values = np.asarray(data.get(key, []), dtype=float)
            return None if values.size == 0 else float(np.max(np.abs(values)))
        numerical_rows.append({
            "case": name,
            "finite_required_fields": finite,
            "U_rel_change_final": final_or_none("U_rel_change_z"),
            "U_rel_change_max_abs": maxabs_or_none("U_rel_change_z"),
            "E_dep_cumulative_final_J": final_or_none("E_dep_cumulative_z"),
            "E_loss_from_input_final_J": final_or_none("E_loss_from_input_z"),
            "dz_min_m": None if "dz_used_z" not in data else float(np.min(data["dz_used_z"])),
            "dz_max_m": None if "dz_used_z" not in data else float(np.max(data["dz_used_z"])),
            "adaptive_rejection_count_max": None if "adaptive_rejection_count_z" not in data else float(np.max(data["adaptive_rejection_count_z"])),
            "safety_trigger_count_max": None if "safety_mode_trigger_count_z" not in data else float(np.max(data["safety_mode_trigger_count_z"])),
            "gpu_allocated_peak_bytes": None if "gpu_allocated_step_bytes" not in data else float(np.max(data["gpu_allocated_step_bytes"])),
            "gpu_reserved_peak_bytes": None if "gpu_reserved_step_bytes" not in data else float(np.max(data["gpu_reserved_step_bytes"])),
        })

    evidence_failures = []
    evidence_failures.extend(_fixed_cap_failures())
    evidence_failures.extend(_case_cap_failures(series))
    # C2 onset comparison is defined at 1e22 m^-3.  All cases that enter the
    # physical threshold comparison must provide that crossing; otherwise a
    # missing crossing is an evidence failure, not a negative result.
    onset_key = str(int(1e22))
    for name in (*series.keys(), "pycap"):
        if metrics[name]["crossings"].get(onset_key) is None:
            evidence_failures.append(f"{name} lacks a finite 1e22 m^-3 crossing")
    rmse = {name: _rmse(data["x_focus_cm"], data["rho_max_z"], px, pycap_rho) for name, data in series.items()}
    for name, value in rmse.items():
        if value is None or not math.isfinite(float(value)):
            evidence_failures.append(f"{name} lacks finite overlap/RMSE evidence against PyCAP")
    if evidence_failures:
        raise InsufficientEvidenceError(evidence_failures)

    onset = next(row for row in threshold_rows if row["threshold_m3"] == 1e22)
    candidate_peak_rel = abs(metrics["candidate_complete_eq27"]["rho_peak_m3"] - metrics["pycap"]["rho_peak_m3"]) / max(abs(metrics["pycap"]["rho_peak_m3"]), 1e-300)
    current_peak_error = abs(metrics["current_full_eq27"]["peak_x_cm"] - metrics["pycap"]["peak_x_cm"])
    candidate_peak_error = abs(metrics["candidate_complete_eq27"]["peak_x_cm"] - metrics["pycap"]["peak_x_cm"])
    classification = classify(
        candidate_current_shift_cm=onset["candidate_minus_current_cm"],
        onset_improvement_cm=onset["candidate_onset_improvement_cm"],
        candidate_peak_density_rel_error=candidate_peak_rel,
        current_rmse=rmse["current_full_eq27"],
        candidate_rmse=rmse["candidate_complete_eq27"],
        candidate_peak_position_error_cm=candidate_peak_error,
        current_peak_position_error_cm=current_peak_error,
    )
    if classification == "insufficient_evidence":
        raise InsufficientEvidenceError("classification inputs are missing or nonfinite")

    out_dir.mkdir(parents=True, exist_ok=True)
    figures = _plot(out_dir, series, px, pycap_rho)
    summary = {
        "schema": "khz_filament.isaacs_complete_eq27.c2_comparison.v1",
        "coordinate_definition": "x_focus_cm = 100 * (z_m - 0.95)",
        "classification": classification,
        "classification_basis": {
            "candidate_minus_current_x_1e22_cm": onset["candidate_minus_current_cm"],
            "candidate_current_shift_abs_cm": None if onset["candidate_minus_current_cm"] is None else abs(onset["candidate_minus_current_cm"]),
            "candidate_onset_improvement_to_pycap_cm": onset["candidate_onset_improvement_cm"],
            "candidate_peak_density_relative_error": candidate_peak_rel,
            "candidate_rmse_vs_pycap_m3": rmse["candidate_complete_eq27"],
            "current_rmse_vs_pycap_m3": rmse["current_full_eq27"],
            "candidate_peak_position_error_cm": candidate_peak_error,
            "current_peak_position_error_cm": current_peak_error,
            "thresholds": {
                "not_supported_shift_lt_cm": ONSET_SIGNIFICANCE_CM,
                "not_supported_improvement_lt_cm": ONSET_SIGNIFICANCE_CM,
                "supported_improvement_ge_cm": SUPPORTED_IMPROVEMENT_CM,
                "supported_peak_density_relative_error_le": PEAK_DENSITY_REL_ERR,
                "supported_rmse_worsening_fraction_le": RMSE_WORSENING_FRACTION,
                "supported_peak_position_worsening_le_cm": PEAK_POSITION_WORSENING_CM,
            },
        },
        "causal_interpretation": (
            "classification applies to the complete combined Eq.27 implementation, including "
            "the electronic move from the central scalar phase/shock approximation into the "
            "combined electronic+rotational Strang half-stages"
        ),
        "causal_limit": (
            "the result does not separately identify derivative algebra, electronic stage "
            "placement, or electronic-rotational Heun coupling"
        ),
        "comparator_provenance": {
            "class": FALLBACK_PROVENANCE_CLASS,
            "qualification": (
                "Current full Eq.27 and Raman-OFF passed their supplied audit, job-id, "
                "input path, and input SHA gates, but remain non-strict fallback "
                "comparators rather than a strict same-run pair. They used mixed_precision, "
                "while the locked mother/candidate configuration retains its baseline default "
                "linear precision."
            ),
            "current_full_eq27": {
                "job_id": current_job_id,
                "role": "fallback comparator",
                "provenance_class": FALLBACK_PROVENANCE_CLASS,
                "physical_classification_allowed": True,
                "audit_json": str(current_audit),
            },
            "raman_off": {
                "job_id": raman_off_job_id,
                "role": "fallback comparator",
                "provenance_class": FALLBACK_PROVENANCE_CLASS,
                "physical_classification_allowed": True,
                "audit_json": str(raman_off_audit),
            },
            "candidate": {
                "job_id": candidate_job_id,
                "audit_json": str(candidate_audit),
                "audit_gate": "passed",
                "provenance_class": STAGING_PROVENANCE_SOURCE_CLASS,
                "staging_provenance": candidate_audit_payload.get("provenance", {}).get("staging", {}),
                "qualification": (
                    "Candidate execution is bound to an externally verified Git bundle, "
                    "but this verified_bundle_non_strict source does not prove a direct "
                    "GitHub remote push/fetch."
                ),
            },
            "excluded_invalid_jobs": ["179706", "179988"],
            "excluded_invalid_jobs_reason": "not used for physical classification",
        },
        "series": {
            "current_full_eq27": str(current_axial),
            "raman_off": str(raman_off_axial),
            "candidate_complete_eq27": str(candidate_axial),
            "pycap": str(pycap_path),
        },
        "rho_threshold_crossings": threshold_rows,
        "intensity_threshold_crossings": intensity_rows,
        "shape_metrics": shape_rows,
        "rmse_vs_pycap_m3": rmse,
        "energy_numerical_path": numerical_rows,
        "generated_figures": figures,
        "evidence_gate": "passed",
    }
    (out_dir / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(out_dir / "rho_threshold_crossings.csv", threshold_rows)
    _write_csv(out_dir / "intensity_threshold_crossings.csv", intensity_rows)
    _write_csv(out_dir / "shape_metrics.csv", shape_rows)
    _write_csv(out_dir / "energy_numerical_path.csv", numerical_rows)
    report = [
        "# Complete Isaacs Eq. (27) C2 comparison",
        "",
        f"Classification: **{classification}** (evidence gate passed).",
        "",
        f"- Candidate-current 1e22 onset shift: `{onset['candidate_minus_current_cm']}` cm.",
        f"- Candidate improvement toward PyCAP at 1e22: `{onset['candidate_onset_improvement_cm']}` cm.",
        f"- Candidate peak-density relative error to PyCAP: `{candidate_peak_rel:.6g}`.",
        f"- Full-axis RMSE current/candidate: `{rmse['current_full_eq27']}` / `{rmse['candidate_complete_eq27']}` m^-3.",
        "",
        f"Fallback qualification: current full Eq.27 job 180748 and Raman-OFF job 180749 are `{FALLBACK_PROVENANCE_CLASS}` comparators. Their supplied audits and CSV path/SHA records passed, but this remains a non-strict fallback comparison and is not evidence of a strict same-run pair; those jobs used mixed_precision while the locked mother/candidate retains its baseline default linear precision.",
        f"Candidate provenance qualification: `{STAGING_PROVENANCE_SOURCE_CLASS}` (verified Git bundle after remote GitHub transport failure); this does not establish direct GitHub remote push/fetch verification.",
        "Invalid jobs 179706 and 179988 are excluded from physical classification.",
        "",
        "Causal interpretation: this classification covers the complete combined Eq.27 implementation, including electronic stage placement and electronic-rotational Heun coupling; it does not isolate the derivative algebra alone.",
        "",
        "No coordinate shift, smoothing, renormalization, or replacement of the fixed PyCAP curve is applied.",
    ]
    (out_dir / "comparison_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-axial", "--production", dest="current_axial", required=True, type=Path)
    parser.add_argument("--current-extras", "--production-extras", dest="current_extras", type=Path, default=None)
    parser.add_argument("--raman-off-axial", "--raman-off", dest="raman_off_axial", required=True, type=Path)
    parser.add_argument("--raman-off-extras", dest="raman_off_extras", type=Path, default=None)
    parser.add_argument("--candidate-axial", "--candidate", dest="candidate_axial", required=True, type=Path)
    parser.add_argument("--candidate-extras", dest="candidate_extras", type=Path, default=None)
    parser.add_argument("--current-audit", "--production-audit", dest="current_audit", required=True, type=Path)
    parser.add_argument("--raman-off-audit", dest="raman_off_audit", required=True, type=Path)
    parser.add_argument("--candidate-audit", dest="candidate_audit", required=True, type=Path)
    parser.add_argument("--pycap", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        summary = compare(
            args.current_axial, args.current_extras,
            args.raman_off_axial, args.raman_off_extras,
            args.candidate_axial, args.candidate_extras,
            args.pycap, args.out_dir,
            args.current_audit, args.raman_off_audit, args.candidate_audit,
        )
    except InsufficientEvidenceError as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps({"classification": summary["classification"]}, indent=2))


if __name__ == "__main__":
    main()
