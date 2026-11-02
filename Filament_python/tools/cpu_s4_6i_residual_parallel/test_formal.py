"""Scalar formal-protocol, analysis, and admission tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.hpc_ops import provenance_v2 as pv

from . import formal_analysis, formal_entry, mechanism_admission, formal_protocol


def _formal_cases() -> dict[str, dict[str, object]]:
    cases: dict[str, dict[str, object]] = {}
    for row in formal_protocol.matrix():
        role = row["role"]
        # Keep the reference and selected clocks deliberately scalar.  The
        # analysis test exercises gates only; it never starts propagation.
        selected = role == "X"
        cases[row["case"]] = {
            **row,
            "exact_gate_pass": True,
            "later_body_s": 8.0 if selected else 10.0,
            "post_gap_median_s": 8.0 if selected else 10.0,
            "application_service_s": 7.0 if selected else 10.0,
            "propagation_service_s": 8.0 if selected else 10.0,
            "first_POST_latency_s": 1.0,
            "initialization_within_application_s": 0.1,
            "first_body_s": 1.0,
            "clock_domain_pass": True,
        }
    return cases


def test_formal_protocol_reuses_fourteen_windows_and_fixed_resources() -> None:
    matrix = formal_protocol.matrix()
    pairings = formal_protocol.pairings()
    assert len(matrix) == 14
    assert len(pairings) == 4
    assert [row["case"] for row in matrix] == [
        "QUAL_R2",
        "QUAL_X2",
        "CTRL_EARLY_R1",
        "CTRL_EARLY_R2",
        "K1_R1",
        "K1_X1",
        "K1_X2",
        "K1_R2",
        "K2_X1",
        "K2_R1",
        "K2_R2",
        "K2_X2",
        "CTRL_LATE_R1",
        "CTRL_LATE_R2",
    ]
    value = formal_protocol.plan()
    assert value["candidate"] == formal_protocol.CANDIDATE
    assert value["phase"] == "FORMAL_FOURTEEN"
    assert value["resources"]["physical_cpus"] == 96
    assert value["resources"]["memory_GiB"] == 128
    assert value["resources"]["walltime_s"] == 7200
    assert value["resources"]["work_deadline_s"] == 7080
    assert value["resources"]["gpus"] == 0
    assert value["resources"]["per_job_allocated_core_hours"] == 192
    assert value["normal_study_job_count"] == 3
    assert value["normal_study_allocated_core_hours_max"] == 480
    assert value["formal_acceptance"]["role_JIT"] == "EMPTY_AT_QUALIFICATION_THEN_REUSE_BY_ROLE"
    assert value["role_components"]["R"] == ["intensity", "eq10", "c", "L1"]
    assert value["role_components"]["X"][-1] == "power_volume"


def test_formal_analysis_adds_service_savings_without_replacing_original_gates() -> None:
    result = formal_analysis.analyze(_formal_cases())
    assert result["original_gates_unchanged"] is True
    assert result["application_speedup_supported"] is True
    assert result["full_application_net_saved_s"] > 0.0
    assert all(row["arithmetic_saved_positive"] for row in result["block_application"])
    assert all(
        "propagation_service_s" in row
        and "propagation_saved_s" in row
        and "application_saved_s" in row
        for row in result["pairs"]
    )

    cases = _formal_cases()
    cases["K1_X1"]["application_service_s"] = 20.0
    rejected = formal_analysis.analyze(cases)
    assert rejected["application_speedup_supported"] is False
    assert rejected["gates"]["application_guard"] is False


def _admission_fixture() -> dict[str, object]:
    digest = "a" * 64
    keys = [(interval, context) for interval in (1, 4, 8) for context in formal_protocol.CONTEXTS]
    return {
        "schema": mechanism_admission.SCHEMA,
        "status": mechanism_admission.STATUS,
        "candidate": formal_protocol.CANDIDATE,
        "shape": list(formal_protocol.SHAPE),
        "capture_execution_commit": "1" * 40,
        "capture_job_id": "8861678",
        "capture_remote_root": "filament_cpu_s46i_power_capture_8861678",
        "capture_result_raw_sha256": digest,
        "capture_source_manifest_raw_sha256": digest,
        "input_contract_raw_sha256": digest,
        "input_count": 12,
        "warmup_calls": 144,
        "formal_calls": 360,
        "blocks": [
            {
                "group": index // 12 + 1,
                "key": f"i{interval}_{context}",
                "p24_gain": 0.10,
                "gain_pass": True,
            }
            for index, (interval, context) in enumerate(keys * 2)
        ],
        "all_exact": True,
        "formal14_allowed": True,
        "scheduler": {
            "job_id": "8861678",
            "all_steps_completed": True,
            "steps": [{"job_id": job, "state": "COMPLETED", "exit_code": "0:0"}
                      for job in ('8861678', '8861678.batch', '8861678.0')],
        },
        "unchanged_runtime_records": [],
    }


def test_mechanism_admission_rejects_tampered_gain_after_execution_rebind(tmp_path, monkeypatch) -> None:
    report = tmp_path / "report"
    report.mkdir()
    admission_path = report / "MECHANISM_ADMISSION.json"
    execution_path = report / "EXECUTION.json"
    admission = _admission_fixture()
    admission_path.write_text(json.dumps(admission) + "\n", encoding="utf-8")
    input_contract_path = report / "INPUT_CONTRACT.json"
    input_contract_path.write_text("{}\n", encoding="utf-8")
    real_raw_sha256_file = pv.raw_sha256_file
    admission_sha256 = real_raw_sha256_file(admission_path)

    def raw_sha256_file(path):
        path = str(path)
        if path.endswith("INPUT_CONTRACT.json"):
            return admission["input_contract_raw_sha256"]
        if path.endswith("MECHANISM_ADMISSION.json"):
            return admission_sha256
        return real_raw_sha256_file(Path(path))

    monkeypatch.setattr(mechanism_admission.pv, "raw_sha256_file", raw_sha256_file)
    execution = {
        "capture_source_manifest_raw_sha256": admission["capture_source_manifest_raw_sha256"],
        "input_contract_raw_sha256": admission["input_contract_raw_sha256"],
        "mechanism_admission_raw_sha256": admission_sha256,
    }
    execution_path.write_text(json.dumps(execution) + "\n", encoding="utf-8")
    monkeypatch.setattr(mechanism_admission, "_runtime_identity", lambda *args, **kwargs: {})
    monkeypatch.setattr(mechanism_admission.pv, "_git", lambda *args: "head")
    accepted = mechanism_admission.verify(report, repo=tmp_path, local_engineering=True)
    assert accepted["formal14_allowed"] is True

    admission["blocks"][0]["p24_gain"] = 0.01
    admission_path.write_text(json.dumps(admission) + "\n", encoding="utf-8")
    execution["mechanism_admission_raw_sha256"] = admission_sha256
    execution_path.write_text(json.dumps(execution) + "\n", encoding="utf-8")
    with pytest.raises(mechanism_admission.MechanismAdmissionError, match="GAIN_GATE"):
        mechanism_admission.verify(report, repo=tmp_path, local_engineering=True)


def test_formal_entry_forces_capture_and_mechanism_off(tmp_path, monkeypatch) -> None:
    work = tmp_path / "case"
    work.mkdir()
    seen: dict[str, bool] = {}

    def fake_run(args):
        seen["capture"] = args.capture
        seen["mechanism"] = args.mechanism
        return {"status": "SCALAR_PROBE", "role": args.role, "intervals": args.intervals}

    monkeypatch.setattr(formal_entry, "_formal_run_function", lambda: fake_run)
    args = SimpleNamespace(work=work, capture=True, mechanism=True, role="X", intervals=8)
    result = formal_entry.run(args)
    assert result["capture_enabled"] is False
    assert result["mechanism_enabled"] is False
    assert seen == {"capture": False, "mechanism": False}
