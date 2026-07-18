#!/usr/bin/env python3
"""Audit the completed 120 fs Raman-phase-off propagation without changing physics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from validate_current_observability_baseline import REQUIRED_Z_FIELDS, sha256, validate_npz


RAMAN_Z_FIELDS = (
    "IR_max_z", "delta_n_rot_max_z", "delta_n_rot_applied_max_z",
    "dphi_rot_max_abs_z", "dphi_rot_applied_max_abs_z",
    "alpha_R_raw_max_z", "alpha_R_applied_max_z",
)


def validate_raman_phase_off(
    npz_path: Path, config_path: Path, metadata_path: Path | None = None, *, expected_execution_sha: str | None = None,
) -> dict[str, Any]:
    """Return a Phase-6-specific audit result for a completed Raman-phase-off NPZ."""
    base = validate_npz(npz_path, config_path, metadata_path)
    failures = list(base["failures"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    propagation = config.get("propagation", {})
    raman = config.get("raman", {})
    if propagation.get("use_raman_phase") is not False:
        failures.append("configuration does not disable propagation.use_raman_phase")
    if propagation.get("use_raman_absorption") is not True or raman.get("absorption") is not True:
        failures.append("configuration does not keep Raman absorption enabled")
    actual_execution_sha = str(base["metadata"].get("execution_git_sha", "")).strip()
    if expected_execution_sha and not actual_execution_sha.startswith(expected_execution_sha):
        failures.append("run metadata execution_git_sha does not match the expected execution SHA")

    fields: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, float]] = []
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
            if key in data.files and not np.allclose(np.asarray(data[key], dtype=float), 0.0, rtol=0.0, atol=1e-30):
                failures.append(f"Raman phase-off applied diagnostic is not zero: {key}")
        if "alpha_R_applied_max_z" in data.files and not np.any(np.abs(np.asarray(data["alpha_R_applied_max_z"], dtype=float)) > 0.0):
            failures.append("Raman absorption applied diagnostic is unexpectedly all zero")
        available = [key for key in RAMAN_Z_FIELDS if key in data.files]
        for index in range(n):
            rows.append({"z_m": float(z[index]), "x_focus_cm": 100.0 * (float(z[index]) - 0.95), **{key: float(np.asarray(data[key])[index]) for key in available}})

    result = {
        "passed": not failures,
        "failures": failures,
        "base": base,
        "raman_field_status": fields,
        "raman_rows": rows,
        "config_sha256": sha256(config_path),
        "npz_sha256": sha256(npz_path),
        "expected_execution_sha": expected_execution_sha,
    }
    return result


def write_audit(result: dict[str, Any], out_dir: Path, *, npz_path: Path, config_path: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "khz_filament.phase6.raman_phase_off_reaudit.v1",
        "raman_phase_off_gate": "passed" if result["passed"] else "failed",
        "npz_path": str(npz_path), "config_path": str(config_path),
        "npz_sha256": result["npz_sha256"], "config_sha256": result["config_sha256"],
        "run_metadata": result["base"]["metadata"], "expected_execution_sha": result["expected_execution_sha"], "base_summary": result["base"]["summary"],
        "raman_field_status": result["raman_field_status"], "failures": result["failures"],
    }
    (out_dir / "raman_phase_off_reaudit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    base_rows = result["base"]["axial_rows"]
    if base_rows:
        with (out_dir / "raman_phase_off_axial_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(base_rows[0]))
            writer.writeheader(); writer.writerows(base_rows)
    raman_rows = result["raman_rows"]
    if raman_rows:
        with (out_dir / "raman_phase_off_raman_extras.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(raman_rows[0]))
            writer.writeheader(); writer.writerows(raman_rows)
    lines = ["# Raman-phase-off propagation re-audit", "", f"Gate: **{payload['raman_phase_off_gate']}**.", ""]
    if result["failures"]:
        lines += ["## Failures", "", *[f"- {item}" for item in result["failures"]]]
    else:
        lines += ["## Result", "", "The current-observability audit passed. Raw Raman response is finite and nonzero; applied Raman phase is zero; Raman absorption remains applied."]
    (out_dir / "raman_phase_off_reaudit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-metadata", type=Path, default=None)
    parser.add_argument("--expected-execution-sha", default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    result = validate_raman_phase_off(args.npz, args.config, args.run_metadata, expected_execution_sha=args.expected_execution_sha)
    payload = write_audit(result, args.out_dir, npz_path=args.npz, config_path=args.config)
    print(f"raman_phase_off_gate={payload['raman_phase_off_gate']}")
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
