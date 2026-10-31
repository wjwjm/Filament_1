"""Focused C14 protocol, admission, and scalar-analysis checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.hpc_ops import provenance_v2 as pv
from tools.cpu_s4_3g3_smallgrid import benchmark as sg

from . import c_formal14_controller as controller
from .c_formal14_analysis import analyze
from .c_formal14_evidence import validate_mechanism_evidence
from .c_formal14_protocol import MATRIX, MICRO_SHAPE, PAIRS, PLAN_SCHEMA, RESULT_SCHEMA, SHAPE, plan


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _synthetic_mechanism_export(root: Path) -> None:
    required = (
        "PLAN.json",
        "HPC_EXECUTION_PLAN.json",
        "EXECUTION.json",
        "EXECUTION_SOURCE_MANIFEST.json",
        "INPUT_CONTRACT.json",
        "REFERENCE_INPUT_RAW_RECORDS.json",
        "RESOURCE_AUTHORIZATION.json",
        "SUBMISSION_PREFLIGHT.json",
        "SUBMISSION_LOCK.json",
        "SUBMISSION_RECEIPT.json",
        "NODE_GATE.json",
        "BENCHMARK_INPUT_MANIFEST.json",
        "RESULT.json",
        "SCHEDULER_ACCOUNTING.json",
        "SCALAR_REPLAY.json",
    )
    result_gains = {key: 0.75 for key in ("G1:0", "G1:3", "G1:7", "G2:0", "G2:3", "G2:7")}
    formal_rows = []
    for group, configs in (("G1", ("REF", "P1", "P24")), ("G2", ("P24", "P1", "REF"))):
        for config in configs:
            for interval in (0, 3, 7):
                for repeat in range(1, 6):
                    seconds = 1.0 if config == "REF" else (0.5 if config == "P1" else 0.25)
                    formal_rows.append({"group": group, "config": config, "interval_index": interval, "repeat": repeat, "seconds": seconds, "exact": True})
    _write_json(root / "EXECUTION.json", {"schema": "filament.cpu_s4_6e_c.execution.v1", "task": "CPU-S4-6E-C-MECHANISM-20261101", "execution_commit": "c19258163a2d570424fd680263421ce8b6f9cd1f", "baseline_commit": "1c9f599d4f7fb0426374997981d0c52ffeaab1aa"})
    _write_json(root / "EXECUTION_SOURCE_MANIFEST.json", {"head": "c19258163a2d570424fd680263421ce8b6f9cd1f", "records": [{"path": path, "classification": "tracked_text", "hash_scope": pv.TRACKED_HASH_SCOPE, "canonical_lf_sha256": "source"} for path in ("Filament_python/tools/cpu_s4_6e_array_propagation/c_candidate.py", "Filament_python/tools/cpu_s4_6e_array_propagation/c_instrument.py", "Filament_python/tools/cpu_s4_6e_array_propagation/c_pool.py", "Filament_python/tools/cpu_s4_6e_array_propagation/c_mechanism.py", "Filament_python/tools/cpu_s4_6e_array_propagation/c_capture.py", "Filament_python/tools/cpu_s4_6e_array_propagation/measurement.py")] + [{"path": "Filament_python/KHz_filament/nonlinear.py", "classification": "tracked_text", "hash_scope": pv.TRACKED_HASH_SCOPE, "canonical_lf_sha256": "science"}]})
    _write_json(root / "RESULT.json", {"schema": "filament.cpu_s4_6e_c.result.v1", "status": "C_CAPTURE_AND_MECHANISM_EXACT_GATE_PASS_MANUAL_REVIEW_PENDING", "production_qualified": False, "mechanism": {"formal_calls": 90, "warmup_calls": 36, "mechanism_gate": True, "exact_by_config": {"REF": True, "P1": True, "P24": True}, "config_order": {"G1": ["REF", "P1", "P24"], "G2": ["P24", "P1", "REF"]}, "formal_rows": formal_rows, "p24_block_gains": result_gains}})
    _write_json(root / "SCALAR_REPLAY.json", {"status": "PASS_EXACT_SCALAR_RECOMPUTATION", "formal_calls": 90, "warmup_calls": 36, "mechanism_gate": True, "p24_block_gains": result_gains})
    _write_json(root / "SCHEDULER_ACCOUNTING.json", {"job_id": "8859283", "scheduler_terminal_state": "COMPLETED", "exit_code": "0:0", "allocated_cpus": 96, "requested_memory": "32G", "rows": [{"State": "COMPLETED", "ExitCode": "0:0"}]})
    for name in required:
        path = root / name
        if not path.exists():
            _write_json(path, {})
    records = []
    for path in sorted(root.glob("*.json")):
        if path.name == "EXPORT_MANIFEST.json":
            continue
        records.append({"path": path.name, "classification": "external", "hash_scope": "raw_bytes", "size_bytes": path.stat().st_size, "raw_sha256": pv.raw_sha256_file(path)})
    _write_json(root / "EXPORT_MANIFEST.json", {"schema": "filament.cpu_s46e.c_scalar_export.v1", "execution_commit": "c19258163a2d570424fd680263421ce8b6f9cd1f", "records": records})


def test_s46e_c14_protocol_matrix_and_counts_are_fixed():
    assert plan()["schema"] == PLAN_SCHEMA
    assert len(MATRIX) == 14
    assert len(PAIRS) == 4
    assert all("capture" not in row for row in plan()["local_cases"])
    assert plan()["counts"]["two_interval"] == {"half_steps": 4, "fresh_responses": 8, "original_math_calls": 32, "c_round_tasks": 4}
    assert plan()["counts"]["eight_interval"] == {"half_steps": 16, "fresh_responses": 32, "original_math_calls": 128, "c_round_tasks": 16}


def test_s46e_c14_mechanism_export_is_admitted_without_scientific_arrays(tmp_path):
    _synthetic_mechanism_export(tmp_path)
    admission = validate_mechanism_evidence(tmp_path)
    assert admission["status"] == "PASS_C14_MECHANISM_ADMISSION"
    assert admission["formal_calls"] == 90 and admission["warmup_calls"] == 36
    assert admission["scientific_arrays_loaded"] == 0
    assert len(admission["p24_block_gains"]) == 6
    assert len(admission["source_identities"]) == 6


@pytest.mark.skipif(not os.environ.get("S46E_C14_MECHANISM_EVIDENCE"), reason="Requires the preserved C mechanism export")
def test_s46e_c14_real_mechanism_export_revalidates_formal_rows():
    admission = validate_mechanism_evidence(os.environ["S46E_C14_MECHANISM_EVIDENCE"], repo=Path(__file__).resolve().parents[3])
    assert admission["status"] == "PASS_C14_MECHANISM_ADMISSION"
    assert admission["formal_calls"] == 90 and admission["warmup_calls"] == 36
    assert min(admission["p24_block_gains"].values()) >= 0.05
    assert len(admission["source_identities"]) == 6


def test_s46e_c14_mechanism_false_blocks_launch_and_preserves_negative_result(tmp_path, monkeypatch):
    report = tmp_path / "report"
    report.mkdir()
    (report / "MECHANISM_ADMISSION.json").write_text(json.dumps({"status": "SCIENTIFIC_INSUFFICIENT_GAIN"}), encoding="utf-8")
    monkeypatch.setattr(controller, "preflight", lambda value: controller._mechanism_gate(value))
    monkeypatch.setattr(controller, "_launch_case", lambda *args, **kwargs: pytest.fail("C14 launch must be blocked by mechanism admission"))
    with pytest.raises(RuntimeError, match="MECHANISM_GATE_MISSING"):
        controller.run(report)
    result = controller._scientific_stop(report, {"job_id": "mock"}, 0.0, {"R8": {"status": "SCIENTIFIC_NONEXACT"}}, stage="R8")
    assert result["status"] == "SCIENTIFIC_NONEXACT"
    assert json.loads((report / "RESULT.json").read_text(encoding="utf-8"))["scientific_stop_stage"] == "R8"


def test_s46e_c14_preflight_is_readonly_and_returns_execution_commit(tmp_path, monkeypatch):
    report = tmp_path / "report"
    report.mkdir()
    admission = {"status": "PASS_C14_MECHANISM_ADMISSION", "mechanism_execution_commit": plan()["mechanism_execution_commit"], "formal_calls": 90, "warmup_calls": 36, "all_configs_exact": True, "p24_block_gains": {key: 0.75 for key in ("G1:0", "G1:3", "G1:7", "G2:0", "G2:3", "G2:7")}}
    monkeypatch.setattr(controller, "_mechanism_gate", lambda value: admission)
    monkeypatch.setattr(controller, "_own_batch_gate", lambda: {"status": "passed"})
    monkeypatch.setattr(controller.c_hpc, "preflight", lambda value: {"execution_commit": "new-c14-sha", "propagation_called": False})
    monkeypatch.setattr(controller.c_hpc, "node_gate", lambda value: pytest.fail("read-only preflight must not allocate node gate"))
    value = controller.preflight(report)
    assert value["execution_commit"] == "new-c14-sha"
    assert value["propagation_called"] is False


def test_s46e_c14_run_uses_one_node_gate_without_duplicate_full_preflight(tmp_path, monkeypatch):
    report = tmp_path / "report"
    report.mkdir()
    _write_json(report / "EXECUTION.json", {"execution_commit": "new-c14-sha"})
    _write_json(report / "INPUT_CONTRACT.json", {"expected_input_fingerprints": {"config": "frozen"}})
    _write_json(report / "REFERENCE_INPUT_RAW_RECORDS.json", {"reference": {"boundary_path": "boundary", "stages": "stages"}})
    admission = {"status": "PASS_C14_MECHANISM_ADMISSION", "mechanism_execution_commit": plan()["mechanism_execution_commit"], "formal_calls": 90, "warmup_calls": 36, "all_configs_exact": True, "p24_block_gains": {key: 0.75 for key in ("G1:0", "G1:3", "G1:7", "G2:0", "G2:3", "G2:7")}}
    monkeypatch.setattr(controller, "_mechanism_gate", lambda value: admission)
    monkeypatch.setattr(controller, "_own_batch_gate", lambda: {"status": "passed"})
    monkeypatch.setattr(controller, "preflight", lambda value: pytest.fail("run must not repeat full C14 preflight"))
    node_calls = []
    monkeypatch.setattr(controller.c_hpc, "node_gate", lambda value: node_calls.append(value) or {"job_id": "mock"})
    monkeypatch.setattr(controller.sg, "input_manifest", lambda *args, **kwargs: {"fingerprints": {"config": "frozen"}})
    monkeypatch.setattr(controller, "_exact", lambda value, **kwargs: value.update(exact_gate_pass=True))

    def fake_launch(_report, *, case, role, intervals, kind, reference, deadline, node):
        base = 10.0 if role == "R" else 8.0
        return {"case_id": case, "role": role, "intervals": intervals, "execution_domain": "S46E_SMALLGRID", "shape": list(SHAPE), "later_body_s": base, "post_gap_median_s": base, "application_service_s": base, "first_POST_latency_s": 1.0, "first_body_s": 1.0, "initialization_within_application_s": 0.5, "clock_domain_pass": True, "c_pool": {"pool": {"closed": True}, "activity": []}}

    monkeypatch.setattr(controller, "_launch_case", fake_launch)
    value = controller.run(report)
    assert len(node_calls) == 1
    assert value["schema"] == RESULT_SCHEMA


@pytest.mark.skipif(not os.environ.get("S46E_C14_REAL_ENTRY_REPORT"), reason="Explicit clean-source C14 local micro entry phase only")
def test_s46e_c14_real_entry_micro_r2_c2_r8_c8():
    report = Path(os.environ["S46E_C14_REAL_ENTRY_REPORT"]).resolve(strict=True)
    sg.input_manifest(report, shape=MICRO_SHAPE)
    root = report / "local_c14"
    root.mkdir(exist_ok=False)

    def run_case(case, role, intervals, reference=None):
        folder = root / case
        folder.mkdir(exist_ok=False)
        work = folder / "work"
        argv = [sys.executable, "-s", "-B", "-m", "tools.cpu_s4_6e_array_propagation.c_entry", "--report", str(report), "--work", str(work), "--role", role, "--intervals", str(intervals), "--local-engineering", "--input-manifest", str(report / "BENCHMARK_INPUT_MANIFEST.json")]
        if reference is not None:
            argv.extend(("--reference-output", reference["output"]))
            if intervals == 2:
                argv.extend(("--reference-boundary", reference["boundary_path"], "--reference-stages", reference["stages"]))
        env = os.environ.copy()
        env["NUMBA_NUM_THREADS"] = "1"
        env["UPPE_USE_GPU"] = "0"
        completed = subprocess.run(argv, cwd=str(Path(__file__).resolve().parents[3]), env=env, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        return json.loads((work / "CASE_RECEIPT.json").read_text(encoding="utf-8"))

    r2 = run_case("R2", "R", 2)
    c2 = run_case("C2", "C", 2, reference=r2)
    r8 = run_case("R8", "R", 8)
    c8 = run_case("C8", "C", 8, reference=r8)
    assert r2["reference_integrity"]["historical_two_interval_terminal_exact"] is False
    assert c2["comparisons"]["terminal"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
    assert c2["comparisons"]["boundary"]["status"] == "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT"
    assert c2["comparisons"]["stages"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
    assert "terminal" not in r8.get("comparisons", {})
    assert c8["comparisons"]["terminal"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
    for result, count in ((r2, 0), (c2, 2), (r8, 0), (c8, 8)):
        assert result["clock_domain_pass"] is True
        assert result["c_pool"]["pool"]["closed"] is True
        assert result["c_pool"]["pool"]["operation_counts"] == {"C_PHASE": count, "C_ABSORPTION": count}


def test_s46e_c14_analysis_uses_c_pool_and_fixed_four_pairs():
    cases = {}
    for case, role, intervals, _kind in MATRIX:
        cases[case] = {
            "role": role,
            "intervals": intervals,
            "exact_gate_pass": True,
            "later_body_s": 10.0 if role == "R" else 8.0,
            "post_gap_median_s": 2.0 if role == "R" else 1.8,
            "application_service_s": 20.0 if role == "R" else 18.0,
            "first_POST_latency_s": 1.0,
            "first_body_s": 1.0,
            "initialization_within_application_s": 0.5,
            "clock_domain_pass": True,
            "c_pool": {"pool": {"closed": True}, "activity": []},
        }
    result = analyze(cases)
    assert result["schema"] == RESULT_SCHEMA
    assert len(result["pairs"]) == 4
    assert result["gates"]["body_paired_median_at_least_3percent"] is True
    assert result["classification"] == "MANUAL_REVIEW_PENDING"

    body_failure = {name: dict(row) for name, row in cases.items()}
    for name, row in body_failure.items():
        if row["role"] == "C":
            row["later_body_s"] = 12.0
    result = analyze(body_failure)
    assert result["status"] == "C14_NO_QUALIFYING_FULL_BODY_GAIN"
    assert result["classification"] == "C14_NO_QUALIFYING_FULL_BODY_GAIN"

    application_failure = {name: dict(row) for name, row in cases.items()}
    for name, row in application_failure.items():
        if row["role"] == "C":
            row["application_service_s"] = 21.0
            row["post_gap_median_s"] = 2.1
    result = analyze(application_failure)
    assert result["status"] == "C14_APPLICATION_OR_POST_GUARD_FAILURE"
    assert result["classification"] == "C14_APPLICATION_OR_POST_GUARD_FAILURE"

    variability = {name: dict(row) for name, row in cases.items()}
    variability["K1_C1"]["later_body_s"] = 10.1
    variability["K2_C2"]["later_body_s"] = 10.1
    result = analyze(variability)
    assert result["status"] == "C14_INCONCLUSIVE_VARIABILITY_OR_ORDER"
    assert result["classification"] == "C14_INCONCLUSIVE_VARIABILITY_OR_ORDER"
