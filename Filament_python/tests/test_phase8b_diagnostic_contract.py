from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "results" / "isaacs_raman_closure" / "phase8b_preflight" / "phase8b_expected_diagnostic_contract.json"


def _load_tool(name):
    path = ROOT / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_contract_derives_nominal_15000_records_and_fixed_coordinates():
    build = _load_tool("build_phase8b_diagnostic_contract.py")
    on = ROOT / "configs" / "isaacs_raman_closure" / "120fs_talebpour_isaacs_full_operator_on.json"
    off = ROOT / "configs" / "isaacs_raman_closure" / "120fs_talebpour_isaacs_full_operator_feedback_off.json"
    contract = build.build_contract(on, off)
    assert contract["record_axis"]["nominal_record_count"] == 15000
    assert contract["fixed_coordinates"]["z_final_m"] == 1.3
    assert contract["fixed_coordinates"]["vacuum_focus_m"] == 0.95
    assert contract["job1_full_operator_on"]["raman_convolutions_per_strang_z_step"] == 4
    assert contract["job2_full_operator_feedback_off"]["raw_diagnostic_convolutions_per_z_step"] == 1


def test_contract_records_energy_thresholds_and_serial_submission_policy():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["raman_energy_contract"]["per_step_p99_lt"] == 1e-3
    assert contract["raman_energy_contract"]["cumulative_final_lt"] == 5e-3
    assert contract["total_energy_contract"]["final_lt"] == 1e-2
    assert contract["total_energy_contract"]["near_focus_max_lt"] == 2e-2
    assert contract["submission_policy"]["job1_must_pass_before_job2_submission"]
    assert contract["submission_policy"]["phase8b_r_requires_separate_user_approval"]


def test_auditor_rejects_duplicate_z_axis(tmp_path):
    audit_tool = _load_tool("audit_phase8b_diagnostics.py")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    n = 3
    payload = {key: np.ones(n) for key in contract["required_fields"]["aligned_z_histories"]}
    payload.update({
        "z_axis": np.array([0.4, 0.4, 1.3]),
        "dz_used_z": np.array([0.4, 0.0, 0.9]),
        "rho_onaxis_t_z": np.ones((n, 4)),
        "raman_operator_mode": np.asarray("full_isaacs_eq27"),
        "raman_operator_feedback_enabled": np.asarray(False),
        "raman_absorption_on": np.asarray(False),
        "delta_n_rot_applied_semantics": np.asarray("not_applicable_full_complex_operator"),
        "raman_closure_residual_semantics": np.asarray("not_applicable_feedback_off_or_legacy"),
        "n2_elec_used": np.asarray(7.8e-24),
        "n_R_used": np.asarray(2.3e-23),
        "raman_operator_applied": np.zeros(n, dtype=bool),
        "raman_rhs_l2_norm": np.zeros(n),
        "raman_IR_max_raw": np.ones(n),
        "raman_target_loss_cumulative_J": np.arange(1, n+1, dtype=float),
        "raman_actual_loss_cumulative_J": np.zeros(n),
        "raman_convolution_count_step": np.ones(n, dtype=int),
        "raman_operator_substep_count": np.zeros(n, dtype=int),
        "alpha_R_applied_max_z": np.zeros(n),
        "U_z": np.ones(n),
        "U_step_change_z": np.zeros(n),
        "E_dep_cumulative_z": np.zeros(n),
    })
    path = tmp_path / "duplicate.npz"
    np.savez(path, **payload)
    with np.load(path, allow_pickle=False) as data:
        result = audit_tool.audit(data, contract, "off")
    checks = {item["name"]: item["passed"] for item in result["checks"]}
    assert not checks["z_strictly_increasing"]
    assert not checks["positive_dz"]
    assert result["status"] == "failed"
