#!/usr/bin/env python3
"""Generate the fixed Eq.27 fallback comparator audits from raw NPZ evidence.

The two archived comparator jobs are immutable operational evidence.  This
tool hashes the fixed raw NPZ and job metadata, verifies their execution
identity, and derives the CSV inputs directly from the NPZ.  It intentionally
has no CLI options for caller-supplied CSVs, reports, or alternate raw paths.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_GPU_MODEL = "NVIDIA GeForce RTX 5090"
EXPECTED_EXECUTION_SHA = "f0a7b5d5ac103546bd693378e8f8efb4f07c6c27"
PROVENANCE_CLASS = "fallback_verified_non_strict"
RAW_REQUIRED_FIELDS = (
    "z_axis", "rho_max_z", "rho_onaxis_max_z", "I_max_z", "w_mom_z", "fwhm_time_z", "U_z",
    "alpha_ion_applied_max_z", "dphi_plasma_applied_max_abs_z", "dphi_elec_applied_max_abs_z",
    "raman_IR_max_raw", "raman_rhs_l2_norm", "raman_target_loss_step_J", "raman_actual_loss_step_J",
    "raman_closure_residual_step", "raman_cumulative_closure_residual",
    "U_rel_change_z", "E_dep_cumulative_z", "E_loss_from_input_z", "dz_used_z",
    "adaptive_rejection_count_z", "safety_mode_trigger_count_z",
)
AXIAL_FIELDS = (
    "z_m", "x_focus_cm", "rho_max_z", "rho_onaxis_max_z", "I_max_z", "U_z", "w_mom_z",
    "fwhm_time_z", "U_rel_change_z", "E_dep_cumulative_z", "E_loss_from_input_z", "dz_used_z",
    "adaptive_rejection_count_z", "safety_mode_trigger_count_z",
)
EXTRAS_FIELDS = (
    "z_m", "x_focus_cm", "raman_IR_max_raw", "raman_rhs_l2_norm", "raman_target_loss_step_J",
    "raman_actual_loss_step_J", "raman_closure_residual_step", "raman_cumulative_closure_residual",
    "alpha_ion_applied_max_z", "dphi_plasma_applied_max_abs_z", "dphi_elec_applied_max_abs_z",
)

FIXED_RAW_EVIDENCE: dict[str, dict[str, str]] = {
    "current_full_eq27": {
        "job_id": "180748",
        "case": "on",
        "operator_mode": "full_isaacs_eq27",
        "operator_state": "applied",
        "npz_path": "/data/run01/scvi806/user_Wangjimin/phase8c_b_runs/test_a_f0a7b5d5ac10_fallback/on/test_a_on.npz",
        "npz_sha256": "68d846d4815cd8387c7a4c4934b26dfe48bcef77cc9140d2f06d2fa8e929a218",
        "metadata_path": "/data/run01/scvi806/user_Wangjimin/phase8c_b_runs/test_a_f0a7b5d5ac10_fallback/on/test_a_on_job_metadata.json",
        "metadata_sha256": "0b057fed4763bb2719d7b8288e820d30cf4f458b3752632d65a026cf1eee9f21",
        "diagnostic_report_path": "/data/run01/scvi806/user_Wangjimin/phase8c_b_runs/test_a_f0a7b5d5ac10_fallback/on/test_a_on.diagnostic_report.json",
        "diagnostic_report_sha256": "5766a2454818f6ad495353f4abcdc6e0668016db2cdb84b1538c910799d8ed84",
        "config_sha256": "aafec917d06c252617e5bfdd2ce3a73dd276401c271c33380d59e0172055cf78",
    },
    "raman_off": {
        "job_id": "180749",
        "case": "off",
        "operator_mode": "full_isaacs_eq27",
        "operator_state": "disabled",
        "npz_path": "/data/run01/scvi806/user_Wangjimin/phase8c_b_runs/test_a_f0a7b5d5ac10_fallback/off/test_a_off.npz",
        "npz_sha256": "e85b8dbbc0fd20b50f6c8234d3de677119ff46f4acaf459e43b1b8ff5e5dc6f9",
        "metadata_path": "/data/run01/scvi806/user_Wangjimin/phase8c_b_runs/test_a_f0a7b5d5ac10_fallback/off/test_a_off_job_metadata.json",
        "metadata_sha256": "d2bd43c85099a03c2b3f226127829c07b99fc955c486989d443d09c08d21716a",
        "diagnostic_report_path": "/data/run01/scvi806/user_Wangjimin/phase8c_b_runs/test_a_f0a7b5d5ac10_fallback/off/test_a_off.diagnostic_report.json",
        "diagnostic_report_sha256": "17ab54bdef325e399618e12bc45141f8cfb4fe65ae2ed945b4573a3afbcd35a2",
        "config_sha256": "1c1415941d4497a6caaf6a37ee8559bbd8b8b20a9eeee6377a8dbbc7d28f41ef",
    },
}
INVALID_JOB_IDS = frozenset({"179706", "179988"})


class FallbackAuditError(ValueError):
    """Raised when fixed fallback provenance cannot be verified."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FallbackAuditError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FallbackAuditError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise FallbackAuditError(f"{label} must be a JSON object")
    return payload


def _scheduler_evidence(job_id: str) -> dict[str, str]:
    """Require a live Slurm terminal record for the fixed archived job."""
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
        raise FallbackAuditError(f"cannot query Slurm evidence for job {job_id}: {exc}") from exc
    rows = [line.strip().split("|") for line in completed.stdout.splitlines() if line.strip()]
    matches = [row for row in rows if len(row) >= 8 and row[0] == job_id]
    if len(matches) != 1:
        raise FallbackAuditError(f"Slurm evidence for job {job_id} is missing or ambiguous")
    row = matches[0]
    if row[1] != "COMPLETED" or row[2] != "0:0":
        raise FallbackAuditError(
            f"Slurm job {job_id} is not an admitted completed run: state={row[1]!r} exit={row[2]!r}"
        )
    return {
        "job_id": row[0],
        "state": row[1],
        "exit_code": row[2],
        "elapsed": row[3],
        "node_list": row[4],
        "submit_time": row[5],
        "start_time": row[6],
        "end_time": row[7],
        "source": "live_sacct",
    }


def _load_raw_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FallbackAuditError(f"raw NPZ does not exist: {path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(set(RAW_REQUIRED_FIELDS).difference(data.files))
            if missing:
                raise FallbackAuditError(f"raw NPZ lacks required diagnostics: {missing}")
            result = {key: np.asarray(data[key]) for key in RAW_REQUIRED_FIELDS}
            for key in (
                "raman_operator_feedback_enabled", "raman_operator_applied", "alpha_R_applied_max_z",
                "raman_convolution_count_step", "raman_operator_substep_count",
            ):
                if key in data.files:
                    result[key] = np.asarray(data[key])
    except FallbackAuditError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise FallbackAuditError(f"raw NPZ is unreadable: {exc}") from exc
    z = np.asarray(result["z_axis"], dtype=float)
    if z.ndim != 1 or z.size < 2 or not np.all(np.diff(z) > 0.0):
        raise FallbackAuditError(f"raw NPZ has invalid z_axis: {path}")
    for key, value in result.items():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array.astype(float))):
            raise FallbackAuditError(f"raw NPZ field {key} contains NaN/Inf")
    result["z_m"] = z
    result["x_focus_cm"] = 100.0 * (z - 0.95)
    n = z.size
    for key, value in result.items():
        if key in {"z_m", "x_focus_cm"}:
            continue
        if key == "raman_operator_feedback_enabled" and np.asarray(value).ndim == 0:
            continue
        if np.asarray(value).ndim != 1 or np.asarray(value).size != n:
            raise FallbackAuditError(f"raw NPZ field {key} is not aligned to z_axis")
    return result


def _verify_fixed_source(role: str) -> tuple[dict[str, str], dict[str, Any], dict[str, np.ndarray]]:
    if role not in FIXED_RAW_EVIDENCE:
        raise FallbackAuditError(f"unsupported fallback role: {role}")
    expected = FIXED_RAW_EVIDENCE[role]
    if expected["job_id"] in INVALID_JOB_IDS:
        raise FallbackAuditError(f"fixed fallback role uses excluded invalid job {expected['job_id']}")
    npz_path = Path(expected["npz_path"])
    metadata_path = Path(expected["metadata_path"])
    try:
        actual_npz_sha = _sha256(npz_path)
        actual_metadata_sha = _sha256(metadata_path)
    except OSError as exc:
        raise FallbackAuditError(f"fixed fallback source cannot be hashed: {exc}") from exc
    if actual_npz_sha != expected["npz_sha256"]:
        raise FallbackAuditError(
            f"{role} raw NPZ SHA256 mismatch: expected={expected['npz_sha256']} actual={actual_npz_sha}"
        )
    if actual_metadata_sha != expected["metadata_sha256"]:
        raise FallbackAuditError(
            f"{role} metadata SHA256 mismatch: expected={expected['metadata_sha256']} actual={actual_metadata_sha}"
        )
    metadata = _read_json(metadata_path, f"{role} fixed job metadata")
    exact = {
        "schema": "phase8c.full_eq27_raman.test_a.job_metadata.v1",
        "case_id": expected["case"],
        "slurm_job_id": expected["job_id"],
        "config_sha256": expected["config_sha256"],
        "gpu_model": EXPECTED_GPU_MODEL,
        "expected_sha": EXPECTED_EXECUTION_SHA,
        "actual_sha": EXPECTED_EXECUTION_SHA,
    }
    for key, value in exact.items():
        if str(metadata.get(key, "")) != value:
            raise FallbackAuditError(f"{role} metadata {key} does not match fixed evidence")
    for key in ("sha_match", "git_status_clean"):
        if metadata.get(key) is not True:
            raise FallbackAuditError(f"{role} metadata {key} is not true")
    config_path = metadata.get("config_path")
    if not isinstance(config_path, str) or not config_path.strip():
        raise FallbackAuditError(f"{role} metadata config_path is empty")
    diagnostic_path = Path(expected["diagnostic_report_path"])
    if not diagnostic_path.is_file() or _sha256(diagnostic_path) != expected["diagnostic_report_sha256"]:
        raise FallbackAuditError(f"{role} fixed diagnostic report is missing or has the wrong SHA256")
    diagnostic = _read_json(diagnostic_path, f"{role} fixed diagnostic report")
    validation = diagnostic.get("validation")
    if diagnostic.get("schema") != "khz_filament.nonlinear_diagnostics.v1":
        raise FallbackAuditError(f"{role} diagnostic report schema is invalid")
    if diagnostic.get("npz_path") != expected["npz_path"]:
        raise FallbackAuditError(f"{role} diagnostic report NPZ path is not the fixed raw source")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise FallbackAuditError(f"{role} diagnostic validation did not pass")
    raw = _load_raw_npz(npz_path)
    return expected, metadata, raw


def _write_csv(path: Path, fields: tuple[str, ...], data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        n = int(np.asarray(data[fields[0]]).size)
        for index in range(n):
            writer.writerow({key: float(np.asarray(data[key])[index]) for key in fields})


def _derived_artifacts(role: str, raw: dict[str, np.ndarray], out_dir: Path) -> tuple[Path, Path]:
    stem = "current_full_eq27" if role == "current_full_eq27" else "raman_off"
    axial = out_dir / f"{stem}_fallback_axial_diagnostics.csv"
    extras = out_dir / f"{stem}_fallback_raman_extras.csv"
    _write_csv(axial, AXIAL_FIELDS, raw)
    _write_csv(extras, EXTRAS_FIELDS, raw)
    return axial, extras


def audit_comparator(*, role: str, out_dir: Path) -> dict[str, Any]:
    """Generate one fixed fallback audit and its NPZ-derived CSV artifacts."""
    expected, metadata, raw = _verify_fixed_source(role)
    scheduler = _scheduler_evidence(expected["job_id"])
    out_dir = out_dir.resolve()
    axial, extras = _derived_artifacts(role, raw, out_dir)
    axial_record = {"path": str(axial.resolve()), "sha256": _sha256(axial)}
    extras_record = {"path": str(extras.resolve()), "sha256": _sha256(extras)}
    payload: dict[str, Any] = {
        "schema": "khz_filament.isaacs_eq27.fallback_comparator_audit.v2",
        "job_id": expected["job_id"],
        "role": role,
        "case": expected["case"],
        "status": "passed",
        "gate": "passed",
        "operator_mode": expected["operator_mode"],
        "operator_state": expected["operator_state"],
        "operator": {"mode": expected["operator_mode"], "state": expected["operator_state"]},
        "numerical_admission": "passed",
        "provenance_class": PROVENANCE_CLASS,
        "raw_source": {
            "generated_from_raw_npz": True,
            "npz": {"path": expected["npz_path"], "sha256": expected["npz_sha256"]},
            "metadata": {"path": expected["metadata_path"], "sha256": expected["metadata_sha256"]},
            "case": expected["case"],
            "job_id": expected["job_id"],
            "config_path": metadata["config_path"],
            "config_sha256": expected["config_sha256"],
            "expected_sha": EXPECTED_EXECUTION_SHA,
            "actual_sha": EXPECTED_EXECUTION_SHA,
            "gpu_model": EXPECTED_GPU_MODEL,
            "scheduler_evidence": {
                **scheduler,
                "diagnostic_report": {"path": expected["diagnostic_report_path"], "sha256": expected["diagnostic_report_sha256"]},
            },
        },
        "artifacts": {"axial": axial_record, "extras": extras_record},
        "axial": axial_record,
        "extras": extras_record,
        "evidence": {
            "finite_required_fields": True,
            "raw_npz_hash_verified": True,
            "metadata_hash_verified": True,
            "axial_rows": int(raw["z_m"].size),
        },
    }
    output = out_dir / ("current_full_eq27_fallback_audit.json" if role == "current_full_eq27" else "raman_off_fallback_audit.json")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def prepare_pair(*, out_dir: Path) -> dict[str, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return {role: audit_comparator(role=role, out_dir=out_dir) for role in ("current_full_eq27", "raman_off")}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        payload = prepare_pair(out_dir=args.out_dir)
    except FallbackAuditError as exc:
        parser.exit(2, f"fallback_audit_failed: {exc}\n")
    print(json.dumps({"status": "passed", "roles": list(payload)}, indent=2))


if __name__ == "__main__":
    main()
