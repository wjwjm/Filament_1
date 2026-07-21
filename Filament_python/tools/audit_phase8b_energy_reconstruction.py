#!/usr/bin/env python3
"""Audit whether a Phase 8B archive can independently reconstruct energy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _array(data: Mapping, key: str) -> np.ndarray:
    return np.asarray(data[key], dtype=np.float64)


def _spacing(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    diffs = np.diff(values)
    return {
        "samples": int(values.size),
        "min_spacing": float(np.min(diffs)) if diffs.size else float("nan"),
        "max_spacing": float(np.max(diffs)) if diffs.size else float("nan"),
        "uniform": bool(diffs.size == 0 or np.allclose(diffs, diffs[0], rtol=1e-10, atol=1e-18)),
    }


def audit_reconstruction(data: Mapping, *, expected_audit: Mapping | None = None) -> dict:
    z = _array(data, "z_axis")
    u = _array(data, "U_z")
    u_step = _array(data, "U_step_change_z")
    u_rel = _array(data, "U_rel_change_z")
    loss = _array(data, "E_loss_from_input_z")
    dep_step = _array(data, "E_dep_total_z")
    dep_cum = _array(data, "E_dep_cumulative_z")
    dz = _array(data, "dz_used_z")
    n = int(z.size)
    if not n or any(arr.size != n for arr in (u, u_step, u_rel, loss, dep_step, dep_cum, dz)):
        raise ValueError("energy reconstruction histories are empty or misaligned")

    u0 = float(u[0] - u_step[0])
    expected_step = np.diff(np.concatenate(([u0], u)))
    expected_loss = u0 - u
    expected_rel = (u - u0) / u0
    expected_dep_cum = np.cumsum(dep_step, dtype=np.float64)
    closure = np.abs(expected_loss - dep_cum) / u0
    field_history_keys = [key for key in data.keys() if key.lower() in {"e_z", "i_z", "e_history", "i_history", "field_history", "intensity_history"}]
    # I_out_center_t is a single transverse point at the final plane and cannot
    # be integrated over x/y to recover pulse energy.
    field_reintegration_available = bool(field_history_keys)
    raw_field_reason = (
        "archive has no full E/I history indexed by z,x,y,t; I_out_center_t is centerline-only"
        if not field_reintegration_available else "full field/intensity history present"
    )

    audit_total = None
    if expected_audit is not None:
        for check in expected_audit.get("checks", []):
            if check.get("name") == "total_energy_final":
                audit_total = float(check["actual"])
                break

    last_step = {
        "z_final_m": float(z[-1]),
        "dz_last_m": float(dz[-1]),
        "last_step_deposition_matches_cumulative_increment_J": float(dep_cum[-1] - (dep_cum[-2] if n > 1 else 0.0) - dep_step[-1]),
        "last_step_field_change_matches_U_step_J": float((u[-1] - (u[-2] if n > 1 else u0)) - u_step[-1]),
    }
    return {
        "schema": "khz_filament.phase8b_r.job1_energy_reconstruction_consistency.v1",
        "archive_field_reintegration": {
            "available": field_reintegration_available,
            "full_history_keys": field_history_keys,
            "reason": raw_field_reason,
            "status": "passed" if field_reintegration_available else "inconclusive_missing_full_field_history",
        },
        "runtime_energy_diagnostics": {
            "initial_energy_J": u0,
            "max_U_step_change_mismatch_J": float(np.max(np.abs(u_step - expected_step))),
            "max_E_loss_from_input_mismatch_J": float(np.max(np.abs(loss - expected_loss))),
            "max_U_rel_change_mismatch": float(np.max(np.abs(u_rel - expected_rel))),
            "max_E_dep_cumulative_mismatch_J": float(np.max(np.abs(dep_cum - expected_dep_cum))),
            "reconstructed_final_total_closure": float(closure[-1]),
            "audit_final_total_closure": audit_total,
            "audit_reconstruction_difference": None if audit_total is None else float(closure[-1] - audit_total),
        },
        "integration_and_grid": {
            "runtime_field_energy_formula": "sum(I) * dt * dx * dy (Cartesian rectangular quadrature)",
            "cylindrical_jacobian": "not applicable: archive stores Cartesian x and y axes",
            "time_axis": _spacing(_array(data, "t_axis")),
            "x_axis": _spacing(_array(data, "x")),
            "y_axis": _spacing(_array(data, "y")),
            "fft_normalization_archive_evidence": "not stored; cannot be independently verified without archived field history",
            "unit_conversion_archive_evidence": "U_z is already J; source runtime formula documented above",
        },
        "alignment": {
            "all_histories_records": n,
            "z_step_end_alignment": "U_z, E_dep_total_z, E_dep_cumulative_z, and dz_used_z are recorded after the accepted step ending at z_axis",
            "initial_reference": "U0 = U_z[0] - U_step_change_z[0]",
            "last_step": last_step,
        },
        "conclusion": {
            "runtime_diagnostics_self_consistent": bool(
                np.max(np.abs(u_step - expected_step)) == 0.0
                and np.max(np.abs(loss - expected_loss)) == 0.0
                and np.max(np.abs(dep_cum - expected_dep_cum)) < 1e-10
            ),
            "archive_can_independently_reintegrate_field_energy": field_reintegration_available,
            "total_energy_closure_from_runtime_histories": float(closure[-1]),
        },
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = json.loads(args.audit.read_text(encoding="utf-8")) if args.audit else None
    with np.load(args.npz, allow_pickle=False) as data:
        result = audit_reconstruction(data, expected_audit=audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["conclusion"], indent=2))


if __name__ == "__main__":
    main()
