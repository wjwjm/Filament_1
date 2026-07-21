#!/usr/bin/env python3
"""Build the machine-readable diagnostic contract for Phase 8B full jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from KHz_filament.diagnostics import Z_HISTORY_TRACE_KEYS  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def nominal_step_count(propagation: dict) -> int:
    z = 0.0
    count = 0
    z_max = float(propagation["z_max"])
    dz_base = float(propagation["dz"])
    use_focus = bool(propagation.get("focus_window_step", False))
    center = float(propagation.get("focus_center_m", 0.0))
    halfwidth = float(propagation.get("focus_halfwidth_m", 0.0))
    dz_focus = float(propagation.get("dz_focus", dz_base))
    while z < z_max - 1e-16:
        if use_focus:
            midpoint = z + 0.5 * dz_base
            dz = dz_focus if abs(midpoint - center) <= halfwidth else dz_base
        else:
            dz = dz_base
        z += min(dz, z_max - z)
        count += 1
    return count


def build_contract(on_path: Path, off_path: Path) -> dict:
    on = _load(on_path)
    off = _load(off_path)
    propagation = on["propagation"]
    nominal_records = nominal_step_count(propagation)
    if nominal_records != 15000:
        raise ValueError(f"Phase 8B nominal record count changed: {nominal_records}")
    return {
        "schema": "khz_filament.phase8b.expected_diagnostic_contract.v1",
        "phase": "Phase 8B corrected Isaacs Raman controlled propagation",
        "source_configs": {
            "full_operator_on": {"path": _repo_path(on_path), "sha256": _sha256(on_path)},
            "full_operator_feedback_off": {"path": _repo_path(off_path), "sha256": _sha256(off_path)},
        },
        "fixed_coordinates": {
            "z_final_m": 1.3,
            "z_final_absolute_tolerance_m": 1e-12,
            "vacuum_focus_m": 0.95,
            "x_focus_cm_formula": "100 * (z_m - 0.95)",
            "coordinate_zero_redefinition_allowed": False,
        },
        "record_axis": {
            "nominal_record_count": nominal_records,
            "nominal_derivation": {
                "base_dz_m": float(propagation["dz"]),
                "focus_center_m": float(propagation["focus_center_m"]),
                "focus_halfwidth_m": float(propagation["focus_halfwidth_m"]),
                "focus_dz_m": float(propagation["dz_focus"]),
                "adaptive_substep_enabled": bool(propagation["auto_substep"]),
            },
            "actual_record_count_rule": "len(z_axis) == len(dz_used_z); z_axis == cumsum(dz_used_z); sum(dz_used_z) == 1.300 m",
            "strictly_increasing": True,
            "duplicates_allowed": False,
            "adaptive_rule": "actual count may exceed 15000 only when positive accepted dz_used_z values reconstruct the strict z_axis",
        },
        "required_fields": {
            "z_axis": "m",
            "aligned_z_histories": list(Z_HISTORY_TRACE_KEYS),
            "z_leading_arrays": {"rho_onaxis_t_z": ["z_record", "time_sample"]},
            "scalar_or_text": [
                "raman_operator_mode",
                "raman_operator_feedback_enabled",
                "raman_absorption_on",
                "delta_n_rot_applied_semantics",
                "raman_closure_residual_semantics",
                "n2_elec_used",
                "n_R_used",
            ],
            "required_species_histories": ["rho_N2_max_z", "rho_O2_max_z"],
        },
        "units": {
            "z_axis": "m",
            "dz_used_z": "m",
            "U_z": "J",
            "I_max_z": "W m^-2",
            "rho_max_z": "m^-3",
            "raman_target_loss_step_J": "J",
            "raman_actual_loss_step_J": "J",
            "raman_target_loss_cumulative_J": "J",
            "raman_actual_loss_cumulative_J": "J",
            "E_dep_z": "J per accepted z step",
            "E_dep_rot_z": "J per accepted z step",
            "E_dep_cumulative_z": "J",
        },
        "common_invariants": {
            "all_required_arrays_present": True,
            "all_numeric_arrays_finite": True,
            "all_z_histories_aligned": True,
            "N2_and_O2_present": True,
            "legacy_raman_alpha_exactly_zero": True,
            "legacy_conv_deriv_executed": False,
            "raman_operator_mode": "full_isaacs_eq27",
        },
        "job1_full_operator_on": {
            "raman_operator_feedback_enabled": True,
            "raman_operator_applied_all_steps": True,
            "raman_IR_max_raw_max_gt": 0.0,
            "raman_rhs_l2_norm_max_gt": 0.0,
            "raman_target_loss_cumulative_final_gt_J": 0.0,
            "raman_actual_loss_cumulative_final_gt_J": 0.0,
            "raman_operator_substeps_per_z_step": 2,
            "raman_convolutions_per_operator_substep": 2,
            "raman_convolutions_per_strang_z_step": 4,
            "delta_n_rot_applied_semantics": "not_applicable_full_complex_operator",
        },
        "job2_full_operator_feedback_off": {
            "raman_operator_feedback_enabled": False,
            "raman_operator_applied_all_steps": False,
            "raman_IR_max_raw_max_gt": 0.0,
            "raman_target_loss_cumulative_final_gt_J": 0.0,
            "raman_rhs_l2_norm_max_abs_le": 0.0,
            "raman_actual_loss_cumulative_final_abs_le_J": 1e-15,
            "raman_operator_substeps_per_z_step": 0,
            "raw_diagnostic_convolutions_per_z_step": 1,
        },
        "raman_energy_contract": {
            "per_step_formula": "abs(E_R_actual-E_R_target) / max(E_R_target, U_before*1e-15)",
            "per_step_p99_lt": 1e-3,
            "cumulative_final_lt": 5e-3,
        },
        "total_energy_contract": {
            "formula": "abs((U0-U_z) - (E_ion+E_IB+E_R)_cumulative) / U0",
            "implementation_fields": "abs((U0-U_z)-E_dep_cumulative_z)/U0, with U0 reconstructed from U_z and U_step_change_z",
            "final_lt": 1e-2,
            "near_focus_max_lt": 2e-2,
            "near_focus_window_m": [0.85, 1.05],
        },
        "submission_policy": {
            "job1_must_pass_before_job2_submission": True,
            "full_jobs_must_be_serial": True,
            "maximum_full_jobs": 2,
            "phase8b_r_requires_separate_user_approval": True,
        },
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--on-config", type=Path, default=ROOT / "configs" / "isaacs_raman_closure" / "120fs_talebpour_isaacs_full_operator_on.json")
    parser.add_argument("--off-config", type=Path, default=ROOT / "configs" / "isaacs_raman_closure" / "120fs_talebpour_isaacs_full_operator_feedback_off.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "isaacs_raman_closure" / "phase8b_preflight" / "phase8b_expected_diagnostic_contract.json")
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_contract(args.on_config, args.off_config), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
