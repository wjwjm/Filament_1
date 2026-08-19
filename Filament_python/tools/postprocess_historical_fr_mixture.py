#!/usr/bin/env python3
"""Postprocess the completed historical_fr_mixture 120 fs propagation NPZ.

Reuses the production ``validate_npz`` base audit and additionally checks the
historical_fr_mixture phase semantics, then writes the z-aligned axial and
Raman diagnostic CSVs consumed by the comparison tool.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from validate_current_observability_baseline import sha256, validate_npz


RAMAN_Z_FIELDS = (
    "IR_max_z", "delta_n_rot_max_z", "delta_n_rot_applied_max_z",
    "dphi_rot_max_abs_z", "dphi_rot_applied_max_abs_z",
    "alpha_R_raw_max_z", "alpha_R_applied_max_z",
)


def validate(npz_path: Path, config_path: Path, metadata_path: Path | None = None,
             *, expected_execution_sha: str | None = None) -> dict[str, Any]:
    base = validate_npz(npz_path, config_path, metadata_path)
    failures = list(base["failures"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    propagation = config.get("propagation", {})
    raman = config.get("raman", {})
    if propagation.get("use_raman_phase") is not True:
        failures.append("historical_fr_mixture requires propagation.use_raman_phase=true")
    if propagation.get("use_raman_absorption") is not True or raman.get("absorption") is not True:
        failures.append("historical_fr_mixture requires Raman absorption enabled")
    if raman.get("operator_mode") != "historical_fr_mixture":
        failures.append("configuration is not raman.operator_mode=historical_fr_mixture")
    actual_execution_sha = str(base["metadata"].get("execution_git_sha", "")).strip()
    if expected_execution_sha and not actual_execution_sha.startswith(expected_execution_sha):
        failures.append("run metadata execution_git_sha does not match the expected execution SHA")

    rows: list[dict[str, float]] = []
    fields: dict[str, dict[str, Any]] = {}
    scalars: dict[str, Any] = {}
    with np.load(npz_path, allow_pickle=False) as data:
        z = np.asarray(data["z_axis"], dtype=float) if "z_axis" in data.files else np.asarray([])
        n = z.size
        for key in RAMAN_Z_FIELDS:
            if key not in data.files:
                failures.append(f"missing Raman diagnostic: {key}")
                continue
            values = np.asarray(data[key], dtype=float)
            fields[key] = {
                "shape": list(values.shape), "finite": bool(np.all(np.isfinite(values))),
                "max_abs": float(np.max(np.abs(values))) if values.size else None,
            }
            if values.ndim != 1 or values.size != n:
                failures.append(f"{key} is not z-aligned")
            if not np.all(np.isfinite(values)):
                failures.append(f"{key} contains NaN/Inf")
        for key in ("IR_max_z", "delta_n_rot_max_z", "dphi_rot_max_abs_z"):
            if key in data.files and not np.any(np.abs(np.asarray(data[key], dtype=float)) > 0.0):
                failures.append(f"raw Raman diagnostic is unexpectedly all zero: {key}")
        for key in ("delta_n_rot_applied_max_z", "dphi_rot_applied_max_abs_z"):
            if key in data.files and not np.any(np.abs(np.asarray(data[key], dtype=float)) > 0.0):
                failures.append(f"applied Raman phase diagnostic is unexpectedly all zero: {key}")
        if "alpha_R_applied_max_z" in data.files and not np.any(np.abs(np.asarray(data["alpha_R_applied_max_z"], dtype=float)) > 0.0):
            failures.append("Raman absorption applied diagnostic is unexpectedly all zero")
        for key in ("f_R_used_historical_fr_mixture", "historical_raman_omega_R_rad_s", "historical_raman_Gamma_R_1_s"):
            if key in data.files:
                scalars[key] = float(np.asarray(data[key]).item())
            else:
                failures.append(f"missing historical_fr_mixture scalar: {key}")
        if "f_R_used_historical_fr_mixture" in scalars and abs(scalars["f_R_used_historical_fr_mixture"] - 0.15) > 1e-6:
            failures.append("f_R_used_historical_fr_mixture is not 0.15")
        if "historical_raman_omega_R_rad_s" in scalars:
            expected_omega = 2.0 * np.pi / 8.4e-12
            if abs(scalars["historical_raman_omega_R_rad_s"] - expected_omega) / expected_omega > 1e-6:
                failures.append("historical_raman_omega_R_rad_s does not match 2*pi/T_R")
        if "historical_raman_Gamma_R_1_s" in scalars:
            expected_gamma = 1.0 / 80e-12
            if abs(scalars["historical_raman_Gamma_R_1_s"] - expected_gamma) / expected_gamma > 1e-6:
                failures.append("historical_raman_Gamma_R_1_s does not match 1/T2")
        available = [key for key in RAMAN_Z_FIELDS if key in data.files]
        for index in range(n):
            rows.append({
                "z_m": float(z[index]),
                "x_focus_cm": 100.0 * (float(z[index]) - 0.95),
                **{key: float(np.asarray(data[key])[index]) for key in available},
            })

    return {
        "passed": not failures,
        "failures": failures,
        "base": base,
        "raman_field_status": fields,
        "raman_rows": rows,
        "scalars": scalars,
        "config_sha256": sha256(config_path),
        "npz_sha256": sha256(npz_path),
        "expected_execution_sha": expected_execution_sha,
    }


def write_audit(result: dict[str, Any], out_dir: Path, *, npz_path: Path, config_path: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "khz_filament.historical_fr_mixture.postprocess.v1",
        "gate": "passed" if result["passed"] else "failed",
        "npz_path": str(npz_path), "config_path": str(config_path),
        "npz_sha256": result["npz_sha256"], "config_sha256": result["config_sha256"],
        "run_metadata": result["base"]["metadata"],
        "expected_execution_sha": result["expected_execution_sha"],
        "base_summary": result["base"]["summary"],
        "raman_field_status": result["raman_field_status"],
        "historical_fr_mixture_scalars": result["scalars"],
        "failures": result["failures"],
    }
    (out_dir / "historical_fr_mixture_reaudit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    base_rows = result["base"]["axial_rows"]
    if base_rows:
        with (out_dir / "historical_fr_mixture_axial_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(base_rows[0]))
            writer.writeheader(); writer.writerows(base_rows)
    if result["raman_rows"]:
        with (out_dir / "historical_fr_mixture_raman_extras.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["raman_rows"][0]))
            writer.writeheader(); writer.writerows(result["raman_rows"])
    lines = ["# historical_fr_mixture propagation postprocess", "", f"Gate: **{payload['gate']}**.", ""]
    if result["failures"]:
        lines += ["## Failures", "", *[f"- {item}" for item in result["failures"]]]
    else:
        lines += [
            "## Result",
            "",
            "The historical_fr_mixture run passed. Raw Raman response is finite and nonzero; applied Raman phase and absorption are nonzero; the phase kernel matches the historical T2/T_R parameterization.",
        ]
    (out_dir / "historical_fr_mixture_reaudit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-metadata", type=Path, default=None)
    parser.add_argument("--expected-execution-sha", default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    result = validate(args.npz, args.config, args.run_metadata,
                      expected_execution_sha=args.expected_execution_sha)
    payload = write_audit(result, args.out_dir, npz_path=args.npz, config_path=args.config)
    print(f"historical_fr_mixture_gate={payload['gate']}")
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
