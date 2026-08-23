#!/usr/bin/env python3
"""Audit one completed Hybrid Propagation 0.60 m paired run.

The raw ``reference/*.npz`` and ``hybrid/*.npz`` files are read in place and
are never copied into the repository.  This tool derives compact axial and
performance CSVs plus an audit JSON.  It deliberately refuses to report a
mechanical supported/not-supported classification when scheduler-terminal or
execution provenance evidence is incomplete; that decision belongs to the
comparison tool after the complete pair is available.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CAMPAIGN_ID = "hybrid_propagation_validation_0p60"
REMOTE_ROOT = "/data/run01/scvi806/user_Wangjimin/hybrid_propagation_validation_0p60"
EXPECTED_GPU = "NVIDIA GeForce RTX 5090"
EXPECTED_NODE = "m4gn1401"
LUT_AUDIT_SCHEMA = "khz_filament.hybrid_propagation_validation.lut_build_audit.v1"
LUT_BUILDER_DEFAULT_CAP = 1.0e16
MANIFEST_SCHEMA = "khz_filament.hybrid_propagation_validation.submission_manifest.v1"
PAIR_SCHEMA = "khz_filament.hybrid_propagation_validation.paired_metadata.v1"
CASE_SCHEMA = "khz_filament.hybrid_propagation_validation.case_metadata.v1"
AUDIT_SCHEMA = "khz_filament.hybrid_propagation_validation.postprocess_audit.v1"

REQUIRED_FIELDS = (
    "z_axis", "rho_max_z", "I_max_z", "U_z", "step_start_z_m", "step_end_z_m",
    "nonlinear_operator_applied", "nonlinear_operator_call_count_step",
    "ionization_solver_call_count_step", "linear_walltime_step_s",
    "nonlinear_walltime_step_s", "ionization_walltime_step_s",
    "raman_operator_walltime_step_s", "total_walltime_step_s",
    "raman_operator_substep_count", "raman_convolution_count_step",
    "raman_operator_applied", "rho_onaxis_max_z", "E_dep_z", "E_dep_rot_z",
    "E_dep_total_z", "alpha_R_max_z", "alpha_R_raw_max_z",
    "alpha_R_applied_max_z", "alpha_ion_raw_max_z", "alpha_ion_corr_max_z",
    "alpha_ion_applied_max_z", "alpha_ib_max_z", "alpha_total_max_z",
    "delta_n_elec_max_z", "delta_n_rot_max_z", "delta_n_plasma_min_z",
    "delta_n_elec_applied_max_z", "delta_n_rot_applied_max_z",
    "delta_n_plasma_applied_min_z", "dphi_kerr_max_abs_z",
    "dphi_elec_max_abs_z", "dphi_rot_max_abs_z", "dphi_plasma_max_abs_z",
    "dphi_elec_applied_max_abs_z", "dphi_rot_applied_max_abs_z",
    "dphi_plasma_raw_max_abs_z", "dphi_plasma_applied_max_abs_z",
    "raman_rhs_l2_norm", "raman_IR_max_raw", "raman_target_loss_step_J",
    "raman_actual_loss_step_J", "gpu_allocated_step_bytes",
    "gpu_reserved_step_bytes",
    "energy_step_start_J", "energy_after_linear_half1_J",
    "energy_after_raman_pre_J", "energy_after_nonraman_J",
    "energy_after_raman_post_J", "energy_after_linear_half2_J",
    "propagation_mode", "z_nl_start_m", "diagnostic_validation_passed",
    "operator_energy_diagnostics_enabled",
)
PERFORMANCE_FIELDS = (
    "linear_walltime_step_s", "nonlinear_walltime_step_s",
    "ionization_walltime_step_s", "raman_operator_walltime_step_s",
    "total_walltime_step_s", "nonlinear_operator_call_count_step",
    "ionization_solver_call_count_step", "raman_operator_substep_count",
    "raman_convolution_count_step", "gpu_allocated_step_bytes",
    "gpu_reserved_step_bytes",
)

LINEAR_ZERO_FIELDS = (
    "rho_max_z", "rho_onaxis_max_z", "E_dep_z", "E_dep_rot_z",
    "E_dep_total_z", "alpha_R_max_z", "alpha_R_raw_max_z",
    "alpha_R_applied_max_z", "alpha_ion_raw_max_z", "alpha_ion_corr_max_z",
    "alpha_ion_applied_max_z", "alpha_ib_max_z", "alpha_total_max_z",
    "delta_n_elec_max_z", "delta_n_rot_max_z", "delta_n_plasma_min_z",
    "delta_n_elec_applied_max_z", "delta_n_rot_applied_max_z",
    "delta_n_plasma_applied_min_z", "dphi_kerr_max_abs_z",
    "dphi_elec_max_abs_z", "dphi_rot_max_abs_z", "dphi_plasma_max_abs_z",
    "dphi_elec_applied_max_abs_z", "dphi_rot_applied_max_abs_z",
    "dphi_plasma_raw_max_abs_z", "dphi_plasma_applied_max_abs_z",
    "raman_rhs_l2_norm", "raman_IR_max_raw", "raman_target_loss_step_J",
    "raman_actual_loss_step_J", "raman_operator_substep_count",
    "raman_convolution_count_step", "nonlinear_operator_call_count_step",
    "ionization_solver_call_count_step", "nonlinear_walltime_step_s",
    "ionization_walltime_step_s", "raman_operator_walltime_step_s",
)


class InsufficientEvidenceError(RuntimeError):
    """Raised when a complete scientific classification cannot be attempted."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InsufficientEvidenceError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise InsufficientEvidenceError(f"{label} must be a JSON object")
    return value


def _npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            return {key: np.asarray(loaded[key]) for key in loaded.files}
    except Exception as exc:
        raise InsufficientEvidenceError(f"raw NPZ cannot be loaded safely: {exc}") from exc


def _scalar(data: dict[str, np.ndarray], key: str, default: Any = None) -> Any:
    if key not in data:
        return default
    value = np.asarray(data[key])
    if value.size != 1:
        return default
    item = value.reshape(-1)[0]
    return item.item() if hasattr(item, "item") else item


def _finite_numeric(value: np.ndarray, key: str) -> np.ndarray:
    try:
        numeric = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InsufficientEvidenceError(f"{key} is not numeric") from exc
    if not np.all(np.isfinite(numeric)):
        raise InsufficientEvidenceError(f"{key} contains NaN/Inf")
    return numeric


def _case_path(run_dir: Path, case: str, suffix: str) -> Path:
    return run_dir / case / f"{case}{suffix}"


def _manifest_case(manifest: dict[str, Any], case: str) -> dict[str, Any]:
    cases = manifest.get("cases")
    if not isinstance(cases, dict) or case not in cases or not isinstance(cases[case], dict):
        raise InsufficientEvidenceError(f"manifest lacks {case} case binding")
    return cases[case]


def _validate_lut_build_audit(value: Any, label: str) -> dict[str, Any]:
    def _number(raw: Any) -> float:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float("nan")

    if not isinstance(value, dict):
        raise InsufficientEvidenceError(f"{label} LUT build audit is missing")
    if value.get("schema") != LUT_AUDIT_SCHEMA:
        raise InsufficientEvidenceError(f"{label} LUT build audit schema is invalid")
    if value.get("status") != "passed" or value.get("passed") is not True or value.get("required") is not True:
        raise InsufficientEvidenceError(f"{label} LUT build audit did not pass")
    builder_cap = _number(value.get("builder_default_cap"))
    threshold = _number(value.get("cap_threshold"))
    expected_threshold = 0.999 * LUT_BUILDER_DEFAULT_CAP
    if not math.isfinite(builder_cap) or not math.isclose(builder_cap, LUT_BUILDER_DEFAULT_CAP, rel_tol=0.0, abs_tol=0.0):
        raise InsufficientEvidenceError(f"{label} LUT builder default cap is invalid")
    if not math.isfinite(threshold) or not math.isclose(threshold, expected_threshold, rel_tol=1e-12, abs_tol=0.0):
        raise InsufficientEvidenceError(f"{label} LUT cap threshold is invalid")
    for key in ("all_configured_caps_valid", "all_builder_caps_consistent", "all_finite", "all_cap_inactive"):
        if value.get(key) is not True:
            raise InsufficientEvidenceError(f"{label} LUT audit aggregate {key} is false")
    species = value.get("species")
    if not isinstance(species, list) or not species:
        raise InsufficientEvidenceError(f"{label} LUT audit has no species records")
    names: set[str] = set()
    for item in species:
        if not isinstance(item, dict):
            raise InsufficientEvidenceError(f"{label} LUT species record is invalid")
        name = str(item.get("name") or "")
        if not name or name in names:
            raise InsufficientEvidenceError(f"{label} LUT species names are invalid or duplicated")
        names.add(name)
        maximum = _number(item.get("W_grid_max"))
        configured_cap = _number(item.get("configured_W_cap"))
        item_builder_cap = _number(item.get("builder_default_cap"))
        if not math.isfinite(maximum) or not math.isfinite(configured_cap) or configured_cap <= 0.0:
            raise InsufficientEvidenceError(f"{label} LUT species {name} has nonfinite cap/max")
        if item.get("configured_cap_valid") is not True:
            raise InsufficientEvidenceError(f"{label} LUT species {name} configured cap is invalid")
        if not math.isclose(item_builder_cap, builder_cap, rel_tol=0.0, abs_tol=0.0):
            raise InsufficientEvidenceError(f"{label} LUT species {name} builder cap is inconsistent")
        if item.get("builder_cap_consistent") is not True:
            raise InsufficientEvidenceError(f"{label} LUT species {name} builder cap consistency is false")
        if item.get("finite") is not True or item.get("cap_inactive") is not True:
            raise InsufficientEvidenceError(f"{label} LUT species {name} finite/cap-inactive checks failed")
        nondecreasing = item.get("nondecreasing")
        negative_step_count = _number(item.get("negative_step_count"))
        max_relative_drop = _number(item.get("max_relative_drop"))
        if not isinstance(nondecreasing, bool) or not math.isfinite(negative_step_count) or negative_step_count < 0.0 or not negative_step_count.is_integer():
            raise InsufficientEvidenceError(f"{label} LUT species {name} nondecreasing diagnostics are invalid")
        if not math.isfinite(max_relative_drop) or max_relative_drop < 0.0:
            raise InsufficientEvidenceError(f"{label} LUT species {name} relative-drop diagnostic is invalid")
        if nondecreasing and negative_step_count != 0.0:
            raise InsufficientEvidenceError(f"{label} LUT species {name} nondecreasing diagnostics are inconsistent")
        if not maximum < threshold:
            raise InsufficientEvidenceError(f"{label} LUT species {name} reaches the builder cap")
    if not all(item.get("configured_cap_valid") is True for item in species):
        raise InsufficientEvidenceError(f"{label} LUT configured-cap aggregate is inconsistent")
    if not all(item.get("builder_cap_consistent") is True for item in species):
        raise InsufficientEvidenceError(f"{label} LUT builder-cap aggregate is inconsistent")
    if not all(item.get("finite") is True for item in species):
        raise InsufficientEvidenceError(f"{label} LUT finite aggregate is inconsistent")
    if not all(item.get("cap_inactive") is True for item in species):
        raise InsufficientEvidenceError(f"{label} LUT cap-inactive aggregate is inconsistent")
    return value


def _validate_scheduler_evidence(
    path: Path | None,
    *,
    run_dir: Path,
    pair: dict[str, Any],
) -> dict[str, Any]:
    if path is None:
        default = run_dir / "scheduler_terminal_evidence.json"
        path = default if default.is_file() else None
    if path is None or not path.is_file():
        raise InsufficientEvidenceError(
            "scheduler terminal evidence is required; provide --scheduler-terminal-evidence"
        )
    evidence = _json(path, "scheduler terminal evidence")
    state = str(evidence.get("state") or evidence.get("State") or "").upper()
    exit_code = str(evidence.get("exit_code") or evidence.get("ExitCode") or "")
    job_id = str(pair.get("slurm_job_id") or "").strip()
    supplied_job = str(evidence.get("job_id") or evidence.get("JobID") or "").strip()
    if state not in {"COMPLETED", "COMPLETE"}:
        raise InsufficientEvidenceError(f"scheduler terminal state is not COMPLETED: {state!r}")
    if exit_code not in {"0", "0:0", "COMPLETED"}:
        raise InsufficientEvidenceError(f"scheduler exit code is not zero: {exit_code!r}")
    if not supplied_job:
        raise InsufficientEvidenceError("scheduler terminal evidence must include job_id")
    if job_id and supplied_job.split(".", 1)[0] != job_id.split(".", 1)[0]:
        raise InsufficientEvidenceError("scheduler terminal evidence job id does not match pair metadata")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "state": state,
        "exit_code": exit_code,
        "job_id": supplied_job or job_id,
        "source": evidence.get("source", "scheduler_terminal_evidence"),
        "raw": evidence,
    }


def _validate_case(
    run_dir: Path,
    case: str,
    manifest: dict[str, Any],
    pair: dict[str, Any],
    *,
    expected_sha: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    case_dir = run_dir / case
    metadata_path = case_dir / f"{case}_job_metadata.json"
    npz_path = case_dir / f"{case}.npz"
    report_path = case_dir / f"{case}.diagnostic_report.json"
    if not metadata_path.is_file() or not npz_path.is_file():
        raise InsufficientEvidenceError(f"{case} raw NPZ/metadata is missing")
    metadata = _json(metadata_path, f"{case} metadata")
    if metadata.get("schema") != CASE_SCHEMA:
        raise InsufficientEvidenceError(f"{case} metadata schema is invalid")
    if metadata.get("campaign_id") != CAMPAIGN_ID or metadata.get("case_id") != case:
        raise InsufficientEvidenceError(f"{case} metadata campaign/case binding is invalid")
    if metadata.get("status") != "completed" or int(metadata.get("exit_code", 1)) != 0:
        raise InsufficientEvidenceError(f"{case} metadata does not show completed success")
    if str(metadata.get("execution_git_sha") or "") != expected_sha:
        raise InsufficientEvidenceError(f"{case} execution SHA does not match pair")
    if metadata.get("gpu_model") != EXPECTED_GPU:
        raise InsufficientEvidenceError(f"{case} GPU model is not the fixed RTX 5090")
    if metadata.get("nodelist") != EXPECTED_NODE or metadata.get("expected_node") != EXPECTED_NODE:
        raise InsufficientEvidenceError(f"{case} node binding is not the fixed {EXPECTED_NODE}")
    if metadata.get("nodelist") != pair.get("nodelist") or metadata.get("expected_node") != pair.get("expected_node"):
        raise InsufficientEvidenceError(f"{case} node binding does not match pair metadata")
    case_lut_audit = _validate_lut_build_audit(metadata.get("lut_build_audit"), f"{case} metadata")
    if case_lut_audit != pair.get("lut_build_audit"):
        raise InsufficientEvidenceError(f"{case} LUT build audit does not match pair metadata")
    if metadata.get("backend") != "cupy" or metadata.get("dtype") != "fp32":
        raise InsufficientEvidenceError(f"{case} backend/dtype is not the fixed cupy/fp32 contract")
    if int(metadata.get("cpu_threads", 0)) != 8:
        raise InsufficientEvidenceError(f"{case} CPU thread count is not 8")
    if str(metadata.get("slurm_job_id") or "") != str(pair.get("slurm_job_id") or ""):
        raise InsufficientEvidenceError(f"{case} Slurm job id does not match pair")
    if not str(metadata.get("started_at_utc") or "") or not str(metadata.get("ended_at_utc") or ""):
        raise InsufficientEvidenceError(f"{case} start/end timestamps are missing")
    binding = _manifest_case(manifest, case)
    expected_config = str(binding.get("config_sha256") or "")
    if metadata.get("config_sha256") != expected_config:
        raise InsufficientEvidenceError(f"{case} config SHA does not match manifest")
    if not isinstance(metadata.get("npz_sha256"), str) or sha256(npz_path) != metadata["npz_sha256"]:
        raise InsufficientEvidenceError(f"{case} NPZ SHA does not match metadata")
    data = _npz(npz_path)
    missing = [key for key in REQUIRED_FIELDS if key not in data]
    if missing:
        raise InsufficientEvidenceError(f"{case} NPZ missing required diagnostic fields: {missing}")
    for key, value in data.items():
        array = np.asarray(value)
        if array.dtype.kind in "biufc" and not np.all(np.isfinite(array)):
            raise InsufficientEvidenceError(f"{case}.{key} contains NaN/Inf")
    z = _finite_numeric(data["z_axis"], f"{case}.z_axis").reshape(-1)
    n = z.size
    if n < 2 or np.any(np.diff(z) < 0.0):
        raise InsufficientEvidenceError(f"{case} z_axis is empty or not non-decreasing")
    for key in REQUIRED_FIELDS:
        value = np.asarray(data[key])
        if key in {
            "propagation_mode", "z_nl_start_m", "diagnostic_validation_passed",
            "operator_energy_diagnostics_enabled",
        }:
            continue
        if value.ndim != 1 or value.size != n:
            raise InsufficientEvidenceError(f"{case}.{key} is not z-aligned")
        if value.dtype.kind in "biufc":
            _finite_numeric(value, f"{case}.{key}")
    mode = str(_scalar(data, "propagation_mode", ""))
    start = float(_scalar(data, "z_nl_start_m", float("nan")))
    expected_mode = "full_nonlinear_from_z0" if case == "reference" else "hybrid"
    expected_start = 0.0 if case == "reference" else 0.6
    if mode != expected_mode or not math.isclose(start, expected_start, rel_tol=0.0, abs_tol=1e-15):
        raise InsufficientEvidenceError(f"{case} propagation mode/start mismatch: {mode!r}, {start!r}")
    if bool(_scalar(data, "diagnostic_validation_passed", False)) is not True:
        raise InsufficientEvidenceError(f"{case} diagnostic_validation_passed is not true")
    if bool(_scalar(data, "operator_energy_diagnostics_enabled", False)) is not True:
        raise InsufficientEvidenceError(f"{case} operator-energy diagnostics are not enabled")
    applied = np.asarray(data["nonlinear_operator_applied"], dtype=bool)
    nonlinear_calls = _finite_numeric(data["nonlinear_operator_call_count_step"], f"{case}.nonlinear_calls")
    ion_calls = _finite_numeric(data["ionization_solver_call_count_step"], f"{case}.ionization_calls")
    if np.any(nonlinear_calls[~applied] != 0.0) or np.any(ion_calls[~applied] != 0.0):
        raise InsufficientEvidenceError(f"{case} linear-only step has nonlinear/ionization calls")
    if np.any(nonlinear_calls[applied] != 1.0) or np.any(ion_calls[applied] < 0.0):
        raise InsufficientEvidenceError(f"{case} active-step call counters are invalid")
    starts = _finite_numeric(data["step_start_z_m"], f"{case}.step_start_z_m")
    ends = _finite_numeric(data["step_end_z_m"], f"{case}.step_end_z_m")
    # z_axis preserves the historical run precision (float32 in production),
    # while the new step boundaries are deliberately float64.  Compare them
    # at the established diagnostic-coordinate tolerance; the exact 0.60 m
    # activation check below still uses the float64 boundary arrays.
    if np.any(ends <= starts) or not np.allclose(ends, z, rtol=0.0, atol=5e-7):
        raise InsufficientEvidenceError(f"{case} step interval/end-coordinate contract is invalid")
    if np.any(starts[1:] != ends[:-1]):
        raise InsufficientEvidenceError(f"{case} saved step intervals are not contiguous")
    if case == "reference" and not np.all(applied):
        raise InsufficientEvidenceError("reference must apply the nonlinear operator at every step")
    if case == "hybrid":
        if not np.any(starts == np.float64(0.6)):
            raise InsufficientEvidenceError("hybrid has no exact 0.60 m step start")
        boundary = np.flatnonzero(ends == np.float64(0.6))
        if boundary.size == 0 or np.any(applied[: boundary[0] + 1]):
            raise InsufficientEvidenceError("hybrid linear preamble is not inactive through 0.60 m")
        expected_applied = starts >= np.float64(0.6)
        if not np.array_equal(applied, expected_applied):
            raise InsufficientEvidenceError("hybrid active mask does not match step_start_z_m >= 0.60 m")
        inactive = ~applied
        for key in LINEAR_ZERO_FIELDS:
            if np.any(np.asarray(data[key])[inactive] != 0):
                raise InsufficientEvidenceError(f"hybrid linear preamble has nonzero {key}")
        if np.any(np.asarray(data["raman_operator_applied"], dtype=bool)[inactive]):
            raise InsufficientEvidenceError("hybrid linear preamble applies Raman")
    if report_path.is_file():
        report = _json(report_path, f"{case} diagnostic report")
        if report.get("validation", {}).get("passed") is False:
            raise InsufficientEvidenceError(f"{case} diagnostic report validation failed")
    performance = {
        "case_total_walltime_s": float(metadata.get("case_total_walltime_s", float("nan"))),
        "step_time_s": float(np.sum(_finite_numeric(data["total_walltime_step_s"], f"{case}.total_walltime_step_s"))),
        "linear_time_s": float(np.sum(_finite_numeric(data.get("linear_walltime_step_s", np.zeros(n)), f"{case}.linear_walltime_step_s"))),
        "nonlinear_time_s": float(np.sum(_finite_numeric(data["nonlinear_walltime_step_s"], f"{case}.nonlinear_walltime_step_s"))),
        "ionization_time_s": float(np.sum(_finite_numeric(data.get("ionization_walltime_step_s", np.zeros(n)), f"{case}.ionization_walltime_step_s"))),
        "raman_time_s": float(np.sum(_finite_numeric(data.get("raman_operator_walltime_step_s", np.zeros(n)), f"{case}.raman_operator_walltime_step_s"))),
        "nonlinear_call_count": int(np.sum(nonlinear_calls)),
        "ionization_call_count": int(np.sum(ion_calls)),
        "raman_substep_count": int(np.sum(_finite_numeric(data["raman_operator_substep_count"], f"{case}.raman_substep_count"))),
        "raman_convolution_count": int(np.sum(_finite_numeric(data["raman_convolution_count_step"], f"{case}.raman_convolution_count"))),
        "gpu_peak_allocated_bytes": int(np.max(_finite_numeric(data["gpu_allocated_step_bytes"], f"{case}.gpu_allocated_step_bytes"))),
        "gpu_peak_reserved_bytes": int(np.max(_finite_numeric(data["gpu_reserved_step_bytes"], f"{case}.gpu_reserved_step_bytes"))),
    }
    if not math.isfinite(performance["case_total_walltime_s"]) or performance["case_total_walltime_s"] <= 0.0:
        raise InsufficientEvidenceError(f"{case} case_total_walltime_s is missing/non-positive")
    metadata["_derived_performance"] = performance
    metadata["_npz_path"] = str(npz_path.resolve())
    metadata["_metadata_path"] = str(metadata_path.resolve())
    metadata["_npz_sha256"] = sha256(npz_path)
    return metadata, data


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _axial_rows(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    z = np.asarray(data["z_axis"], dtype=float).reshape(-1)
    preferred = (
        "rho_max_z", "rho_onaxis_max_z", "I_max_z", "I_onaxis_max_z", "U_z",
        "w_mom_z", "fwhm_plasma_z", "fwhm_fluence_z", "fwhm_time_z",
        "E_dep_z", "E_dep_rot_z", "E_dep_total_z", "E_dep_cumulative_z", "U_rel_change_z",
        "step_start_z_m", "step_end_z_m", "nonlinear_operator_applied",
        "nonlinear_operator_call_count_step", "ionization_solver_call_count_step",
        "linear_walltime_step_s", "nonlinear_walltime_step_s", "ionization_walltime_step_s",
        "raman_operator_walltime_step_s", "total_walltime_step_s",
        "raman_operator_applied", "raman_operator_substep_count", "raman_convolution_count_step",
        "alpha_R_max_z", "alpha_R_raw_max_z", "alpha_R_applied_max_z",
        "alpha_ion_raw_max_z", "alpha_ion_corr_max_z", "alpha_ion_applied_max_z",
        "alpha_ib_max_z", "alpha_total_max_z", "delta_n_elec_max_z",
        "delta_n_rot_max_z", "delta_n_plasma_min_z",
        "delta_n_elec_applied_max_z", "delta_n_rot_applied_max_z",
        "delta_n_plasma_applied_min_z", "dphi_kerr_max_abs_z",
        "dphi_elec_max_abs_z", "dphi_rot_max_abs_z", "dphi_plasma_max_abs_z",
        "dphi_elec_applied_max_abs_z", "dphi_rot_applied_max_abs_z",
        "dphi_plasma_raw_max_abs_z", "dphi_plasma_applied_max_abs_z",
        "adaptive_rejection_count_z", "safety_mode_trigger_count_z",
    )
    keys = [key for key in preferred if key in data and np.asarray(data[key]).ndim == 1 and np.asarray(data[key]).size == z.size]
    rows: list[dict[str, Any]] = []
    for i in range(z.size):
        row: dict[str, Any] = {"z_m": float(z[i])}
        for key in keys:
            value = np.asarray(data[key]).reshape(-1)[i]
            row[key] = value.item() if hasattr(value, "item") else value
        rows.append(row)
    return rows


def process_pair(
    run_dir: Path,
    out_dir: Path,
    *,
    manifest_path: Path | None = None,
    scheduler_terminal_evidence: Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if manifest_path is None:
        manifest_path = run_dir / "submission_manifest.json"
    if not manifest_path.is_file():
        # Submission keeps the manifest in the repository; a copied JSON is
        # accepted only as an explicit input to this read-only postprocessor.
        raise InsufficientEvidenceError("campaign manifest path is required")
    manifest = _json(Path(manifest_path), "campaign manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("campaign_id") != CAMPAIGN_ID:
        raise InsufficientEvidenceError("campaign manifest schema/campaign binding is invalid")
    if manifest.get("remote_campaign_root") != REMOTE_ROOT:
        raise InsufficientEvidenceError("campaign remote root is not fixed")
    if manifest.get("lut_build_cap_inactive_required") is not True:
        raise InsufficientEvidenceError("manifest LUT cap-inactive requirement is missing")
    expected_diff = [
        {
            "path": "propagation.propagation_mode",
            "reference": "full_nonlinear_from_z0",
            "hybrid": "hybrid",
        },
        {
            "path": "propagation.z_nl_start",
            "reference": 0.0,
            "hybrid": 0.6,
        },
    ]
    if manifest.get("strict_config_diff") != expected_diff:
        raise InsufficientEvidenceError("manifest strict_config_diff is not the fixed two-field A/B delta")
    pair_path = run_dir / "paired_job_metadata.json"
    pair = _json(pair_path, "paired job metadata")
    if pair.get("schema") != PAIR_SCHEMA or pair.get("campaign_id") != CAMPAIGN_ID:
        raise InsufficientEvidenceError("paired metadata schema/campaign binding is invalid")
    if pair.get("status") != "completed" or int(pair.get("exit_code", 1)) != 0:
        raise InsufficientEvidenceError("paired metadata does not show completed success")
    if pair.get("case_order") != ["reference", "hybrid"] or pair.get("allocation_count") != 1:
        raise InsufficientEvidenceError("paired metadata case/allocation contract is invalid")
    if pair.get("gpu_model") != EXPECTED_GPU:
        raise InsufficientEvidenceError("paired metadata GPU model is not the fixed RTX 5090")
    if pair.get("nodelist") != EXPECTED_NODE or pair.get("expected_node") != EXPECTED_NODE:
        raise InsufficientEvidenceError(f"paired metadata node binding is not the fixed {EXPECTED_NODE}")
    lut_build_audit = _validate_lut_build_audit(pair.get("lut_build_audit"), "paired metadata")
    execution_sha = str(pair.get("execution_git_sha") or "").strip()
    if not execution_sha:
        raise InsufficientEvidenceError("paired metadata lacks execution_git_sha")
    for key in ("execution_lock_sha256", "provenance_v2_sha256"):
        value = str(pair.get(key) or "")
        if len(value) != 64:
            raise InsufficientEvidenceError(f"paired metadata lacks valid {key}")
    if not str(pair.get("started_at_utc") or "") or not str(pair.get("ended_at_utc") or ""):
        raise InsufficientEvidenceError("paired metadata lacks start/end timestamps")
    manifest_sha = sha256(Path(manifest_path))
    if pair.get("manifest_sha256") != manifest_sha:
        raise InsufficientEvidenceError("paired metadata is not bound to the supplied manifest hash")
    terminal = _validate_scheduler_evidence(scheduler_terminal_evidence, run_dir=run_dir, pair=pair)
    cases: dict[str, Any] = {}
    data_map: dict[str, dict[str, np.ndarray]] = {}
    for case in ("reference", "hybrid"):
        metadata, data = _validate_case(run_dir, case, manifest, pair, expected_sha=execution_sha)
        cases[case] = metadata
        data_map[case] = data
        _write_csv(out_dir / f"{case}_axial.csv", _axial_rows(data), list(_axial_rows(data)[0]))
    for key in ("backend", "dtype", "linear_model", "linear_precision_strategy", "thread_environment", "nodelist", "expected_node"):
        if cases["reference"].get(key) != cases["hybrid"].get(key):
            raise InsufficientEvidenceError(f"paired cases do not share {key}")
    perf_fields = [
        "case", "case_total_walltime_s", "step_time_s", "linear_time_s",
        "nonlinear_time_s", "ionization_time_s", "raman_time_s",
        "nonlinear_call_count", "ionization_call_count", "raman_substep_count",
        "raman_convolution_count", "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes",
    ]
    perf_rows = [{"case": case, **cases[case]["_derived_performance"]} for case in ("reference", "hybrid")]
    _write_csv(out_dir / "performance.csv", perf_rows, perf_fields)
    audit = {
        "schema": AUDIT_SCHEMA,
        "status": "complete_evidence",
        "campaign_id": CAMPAIGN_ID,
        "remote_campaign_root": REMOTE_ROOT,
        "run_dir": str(run_dir),
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": manifest_sha,
        "execution_git_sha": execution_sha,
        "case_order": ["reference", "hybrid"],
        "gpu_model": cases["reference"]["gpu_model"],
        "nodelist": cases["reference"]["nodelist"],
        "expected_node": cases["reference"]["expected_node"],
        "lut_build_audit": lut_build_audit,
        "backend": cases["reference"]["backend"],
        "dtype": cases["reference"]["dtype"],
        "linear_model": cases["reference"]["linear_model"],
        "linear_precision_strategy": cases["reference"]["linear_precision_strategy"],
        "thread_environment": cases["reference"]["thread_environment"],
        "slurm_job_id": pair.get("slurm_job_id"),
        "pair_started_at_utc": pair.get("started_at_utc"),
        "pair_ended_at_utc": pair.get("ended_at_utc"),
        "execution_lock_sha256": pair.get("execution_lock_sha256"),
        "provenance_v2_path": pair.get("provenance_v2_path"),
        "provenance_v2_sha256": pair.get("provenance_v2_sha256"),
        "strict_config_diff": manifest["strict_config_diff"],
        "scheduler_terminal_evidence": terminal,
        "raw_npz_policy": "raw NPZ remains in HPC RUN_DIR and is not copied",
        "cases": {
            case: {
                "case_id": case,
                "status": cases[case]["status"],
                "config_path": cases[case].get("config_path"),
                "config_sha256": cases[case].get("config_sha256"),
                "execution_git_sha": cases[case].get("execution_git_sha"),
                "gpu_model": cases[case].get("gpu_model"),
                "nodelist": cases[case].get("nodelist"),
                "expected_node": cases[case].get("expected_node"),
                "lut_build_audit": cases[case].get("lut_build_audit"),
                "npz_path": cases[case]["_npz_path"],
                "npz_sha256": cases[case]["_npz_sha256"],
                "metadata_path": cases[case]["_metadata_path"],
                "case_total_walltime_s": cases[case]["_derived_performance"]["case_total_walltime_s"],
                "performance": cases[case]["_derived_performance"],
                "propagation_mode": str(_scalar(data_map[case], "propagation_mode")),
                "z_nl_start_m": float(_scalar(data_map[case], "z_nl_start_m")),
                "z_records": int(np.asarray(data_map[case]["z_axis"]).size),
                "finite": True,
            }
            for case in ("reference", "hybrid")
        },
        "derived_outputs": {
            "reference_axial_csv": str((out_dir / "reference_axial.csv").resolve()),
            "hybrid_axial_csv": str((out_dir / "hybrid_axial.csv").resolve()),
            "performance_csv": str((out_dir / "performance.csv").resolve()),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "hybrid_propagation_validation_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


# Short aliases make the read-only tool convenient for focused unit tests and
# downstream report wrappers without changing the CLI contract.
postprocess = process_pair


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scheduler-terminal-evidence", type=Path)
    args = parser.parse_args(argv)
    try:
        result = process_pair(
            args.run_dir, args.out_dir,
            manifest_path=args.manifest,
            scheduler_terminal_evidence=args.scheduler_terminal_evidence,
        )
    except InsufficientEvidenceError as exc:
        parser.error(str(exc))
    print(json.dumps({"status": result["status"], "audit": str((args.out_dir / "hybrid_propagation_validation_audit.json").resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
