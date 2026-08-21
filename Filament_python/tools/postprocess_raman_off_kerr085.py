#!/usr/bin/env python3
"""Validate and export the Raman-phase-OFF + 0.85 electronic-Kerr result."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from validate_raman_phase_ablation import validate_raman_phase_off


EXPECTED_N2 = 6.63e-24


def validate(npz: Path, config: Path, metadata: Path, expected_sha: str) -> dict:
    result = validate_raman_phase_off(npz, config, metadata, expected_execution_sha=expected_sha)
    failures = list(result["failures"])
    cfg = json.loads(config.read_text(encoding="utf-8"))
    if float(cfg["beam"]["n2_air"]) != EXPECTED_N2:
        failures.append("beam.n2_air is not 6.63e-24")
    scalars = {}
    with np.load(npz, allow_pickle=False) as data:
        if "n2_elec_used" not in data.files:
            failures.append("missing n2_elec_used")
        else:
            scalars["n2_elec_used"] = float(np.asarray(data["n2_elec_used"]).item())
            if not np.isclose(scalars["n2_elec_used"], EXPECTED_N2, rtol=1e-7, atol=0.0):
                failures.append("n2_elec_used is not 6.63e-24")
        if "delta_n_elec_max_z" in data.files and "I_max_z" in data.files:
            dn = np.asarray(data["delta_n_elec_max_z"], float)
            imax = np.asarray(data["I_max_z"], float)
            mask = imax > 0.0
            ratio = dn[mask] / imax[mask]
            scalars["median_delta_n_elec_over_I"] = float(np.median(ratio))
            if not np.allclose(ratio, EXPECTED_N2, rtol=2e-5, atol=0.0):
                failures.append("delta_n_elec_max_z / I_max_z does not match 6.63e-24")
    result["failures"] = failures
    result["passed"] = not failures
    result["candidate_scalars"] = scalars
    return result


def write(result: dict, out_dir: Path, npz: Path, config: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "khz_filament.raman_off_kerr085.postprocess.v1",
        "gate": "passed" if result["passed"] else "failed",
        "npz_path": str(npz), "config_path": str(config),
        "npz_sha256": result["npz_sha256"], "config_sha256": result["config_sha256"],
        "run_metadata": result["base"]["metadata"],
        "base_summary": result["base"]["summary"],
        "raman_field_status": result["raman_field_status"],
        "candidate_scalars": result["candidate_scalars"],
        "failures": result["failures"],
    }
    (out_dir / "raman_off_kerr085_reaudit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    rows = result["base"]["axial_rows"]
    with (out_dir / "raman_off_kerr085_axial_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    raman_rows = result["raman_rows"]
    with (out_dir / "raman_off_kerr085_raman_extras.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raman_rows[0])); writer.writeheader(); writer.writerows(raman_rows)
    report = [
        "# Raman phase OFF + 0.85 electronic Kerr postprocess", "",
        f"Gate: **{payload['gate']}**.", "",
        "The result preserves Raman phase OFF and active Raman absorption; the executed electronic Kerr coefficient is 6.63e-24 m^2/W."
        if result["passed"] else "Failures:",
        *([f"- {item}" for item in result["failures"]] if result["failures"] else []), "",
    ]
    (out_dir / "raman_off_kerr085_reaudit_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--expected-execution-sha", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.npz, args.config, args.run_metadata, args.expected_execution_sha)
    write(result, args.out_dir, args.npz, args.config)
    print(f"raman_off_kerr085_gate={'passed' if result['passed'] else 'failed'}")
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
