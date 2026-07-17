#!/usr/bin/env python3
"""Validate and archive a current-observability Popruzhenko propagation run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
FILAMENT_ROOT = SCRIPT_DIR.parent
if str(FILAMENT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(FILAMENT_ROOT))

from KHz_filament.constants import N0_air  # noqa: E402


REQUIRED_Z_FIELDS = (
    "z_axis", "rho_max_z", "I_onaxis_max_z", "I_max_z", "delta_n_plasma_min_z", "E_dep_z",
    "E_loss_from_input_z", "E_dep_cumulative_z", "U_rel_change_z", "dphi_plasma_raw_max_abs_z",
    "dphi_plasma_applied_max_abs_z", "alpha_ion_raw_max_z", "alpha_ion_applied_max_z", "rho_N2_max_z",
    "rho_O2_max_z", "rho_O2_fraction_at_rho_total_max_z", "dz_used_z", "adaptive_rejection_count_z",
    "safety_mode_trigger_count_z",
)
REQUIRED_SCALARS = ("safety_mode_event_summary", "propagation_observability_schema", "diagnostic_validation_passed")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalised_fractions(config: dict[str, Any]) -> dict[str, float]:
    species = config["ionization"]["species"]
    raw = {str(item["name"]): max(float(item.get("fraction", 1.0)), 0.0) for item in species}
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError("ionization fractions must sum to a positive value")
    return {name: value / total for name, value in raw.items()}


def validate_npz(npz_path: Path, config_path: Path, run_metadata_path: Path | None = None) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fractions = _normalised_fractions(config)
    failures: list[str] = []
    with np.load(npz_path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_Z_FIELDS + REQUIRED_SCALARS if key not in data.files]
        if missing:
            failures.append(f"missing required diagnostics: {missing}")
        z = np.asarray(data["z_axis"], dtype=float) if "z_axis" in data.files else np.asarray([])
        n = int(z.size)
        field_status: dict[str, dict[str, Any]] = {}
        for key in REQUIRED_Z_FIELDS:
            if key not in data.files:
                continue
            values = np.asarray(data[key])
            field_status[key] = {"shape": list(values.shape), "finite": bool(np.all(np.isfinite(values)))}
            if values.ndim != 1 or values.size != n:
                failures.append(f"{key} is not z-aligned")
            if not np.all(np.isfinite(values)):
                failures.append(f"{key} contains NaN/Inf")
        if n == 0 or not np.all(np.isfinite(z)) or (n > 1 and np.any(np.diff(z) <= 0.0)):
            failures.append("z_axis is empty, non-finite, or non-increasing")
        z_max = float(config["propagation"]["z_max"])
        if n and float(z[-1]) < z_max - 2e-6:
            failures.append(f"z_end={z[-1]:.9g} m does not reach configured z_max={z_max:.9g} m")
        if "rho_N2_max_z" in data.files and np.any(np.asarray(data["rho_N2_max_z"], float) > N0_air * fractions.get("N2", 0.0) * (1.0 + 1e-5)):
            failures.append("rho_N2_max_z exceeds its neutral-density bound")
        if "rho_O2_max_z" in data.files and np.any(np.asarray(data["rho_O2_max_z"], float) > N0_air * fractions.get("O2", 0.0) * (1.0 + 1e-5)):
            failures.append("rho_O2_max_z exceeds its neutral-density bound")
        if "rho_O2_fraction_at_rho_total_max_z" in data.files:
            values = np.asarray(data["rho_O2_fraction_at_rho_total_max_z"], float)
            if np.any((values < -1e-8) | (values > 1.0 + 1e-8)):
                failures.append("rho_O2_fraction_at_rho_total_max_z is outside [0,1]")
        if "E_dep_cumulative_z" in data.files:
            cumulative = np.asarray(data["E_dep_cumulative_z"], float)
            if np.any(np.diff(cumulative) < -1e-10):
                failures.append("E_dep_cumulative_z is not non-decreasing")
        if "dz_used_z" in data.files and np.any(np.asarray(data["dz_used_z"], float) <= 0.0):
            failures.append("dz_used_z contains non-positive values")
        for key in ("adaptive_rejection_count_z", "safety_mode_trigger_count_z"):
            if key in data.files:
                values = np.asarray(data[key], float)
                if np.any(values < 0.0) or np.any(np.diff(values) < 0.0):
                    failures.append(f"{key} is not non-negative and non-decreasing")
        diagnostic_validation = bool(np.asarray(data["diagnostic_validation_passed"]).item()) if "diagnostic_validation_passed" in data.files else False
        if not diagnostic_validation:
            failures.append("diagnostic_validation_passed is false or unavailable")
        metadata = json.loads(run_metadata_path.read_text(encoding="utf-8")) if run_metadata_path and run_metadata_path.exists() else {}
        if not metadata.get("execution_git_sha"):
            failures.append("run metadata lacks execution_git_sha")
        if metadata.get("config_sha256") and metadata["config_sha256"] != sha256(config_path):
            failures.append("run metadata config_sha256 does not match the supplied config")
        summary = {
            "z_records": n,
            "z_start_m": float(z[0]) if n else None,
            "z_end_m": float(z[-1]) if n else None,
            "z_median_step_m": float(np.median(np.diff(z))) if n > 1 else None,
            "npz_sha256": sha256(npz_path),
            "npz_size_bytes": npz_path.stat().st_size,
            "config_sha256": sha256(config_path),
            "field_status": field_status,
            "diagnostic_validation_passed": diagnostic_validation,
            "safety_mode_event_summary": str(np.asarray(data["safety_mode_event_summary"]).item()) if "safety_mode_event_summary" in data.files else None,
        }
        rows = [{"z_m": float(z[index]), "x_focus_cm": 100.0 * (float(z[index]) - 0.95), **{key: float(np.asarray(data[key])[index]) for key in REQUIRED_Z_FIELDS if key != "z_axis" and key in data.files}} for index in range(n)]
    return {"passed": not failures, "failures": failures, "summary": summary, "metadata": metadata, "axial_rows": rows}


def write_audit(result: dict[str, Any], out_dir: Path, *, npz_path: Path, config_path: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "khz_filament.phase5.current_observability_baseline.v1",
        "formal_baseline_gate": "passed" if result["passed"] else "failed",
        "npz_path": str(npz_path),
        "config_path": str(config_path),
        "summary": result["summary"],
        "run_metadata": result["metadata"],
        "failures": result["failures"],
    }
    (out_dir / "baseline_reaudit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["z_m", "x_focus_cm", *[key for key in REQUIRED_Z_FIELDS if key != "z_axis"]]
    with (out_dir / "baseline_axial_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["axial_rows"])
    report = ["# Current-observability Popruzhenko baseline re-audit", "", f"Formal baseline gate: **{payload['formal_baseline_gate']}**.", ""]
    if result["failures"]:
        report.extend(["## Failures", "", *[f"- {item}" for item in result["failures"]]])
    else:
        report.extend(["## Result", "", "All required current-observability diagnostics are present, z-aligned, finite, and internally consistent."])
    report.append("")
    (out_dir / "baseline_reaudit_report.md").write_text("\n".join(report), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-metadata", type=Path, default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    result = validate_npz(args.npz, args.config, args.run_metadata)
    payload = write_audit(result, args.out_dir, npz_path=args.npz, config_path=args.config)
    print(f"formal_baseline_gate={payload['formal_baseline_gate']}")
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
