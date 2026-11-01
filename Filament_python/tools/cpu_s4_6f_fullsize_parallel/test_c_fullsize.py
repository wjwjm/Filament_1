from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
from types import SimpleNamespace

import pytest

from . import role_spec
from . import c_fullsize_entry, c_fullsize_hpc
from .c_fullsize_evidence import MechanismEvidenceError, _gain_map, validate_mechanism_evidence
from .c_fullsize_protocol import CAPTURE_INTERVALS, CAPTURE_TOTAL_CALLS, FORMAL_CALLS, MICRO_SHAPE, RESULT_SCHEMA, RESOURCES, SHAPE, WARMUP_CALLS, plan
from tools.hpc_ops import provenance_v2 as pv


def test_fullsize_mechanism_contract_is_fixed_and_serial():
    value = plan()
    assert value["shape"] == list(SHAPE)
    assert value["capture"] == {
        "selected_interval_indices": list(CAPTURE_INTERVALS),
        "expected_total_calls": CAPTURE_TOTAL_CALLS,
        "source_role": "R",
        "array_storage": "NPY_ALLOW_PICKLE_FALSE",
    }
    assert value["mechanism"]["warmup_calls"] == WARMUP_CALLS
    assert value["mechanism"]["formal_calls"] == FORMAL_CALLS
    assert value["cpu_diagnostics_profile"] is True and value["fine_performance_profile"] is False
    assert value["resources"] == RESOURCES
    assert value["serial"] is True and value["fullsize"] is True and value["GPU"] is False


def test_role_spec_forwards_capture_store_only_to_c(monkeypatch):
    dispatch = SimpleNamespace(cpu_s4_4b_cache_snapshot=lambda: {"configured_shape": list(SHAPE)})
    marker = object()
    captured = {}

    class FakeModule:
        @staticmethod
        def build_role(mode, **kwargs):
            captured.update(kwargs)
            return (object(), dispatch, object(), {}, object(), None)

    monkeypatch.setattr(role_spec, "_legacy_module", lambda spec: FakeModule)
    monkeypatch.setattr(role_spec.cache_contract, "validate_shape", lambda *args, **kwargs: None)
    monkeypatch.setattr(role_spec.cache_candidate, "_replace_empty_cache", lambda *args, **kwargs: {})
    role_spec.build_role("PHASE_ABSORB_P24", "PURE_OFF_FULLSIZE", capture_root="unused", role="R", local_engineering=True, recorder_pair=object(), shape=MICRO_SHAPE, pool_override=1, capture_store=marker)
    assert captured["capture_store"] is marker


def test_mechanism_gain_validator_recomputes_all_six_blocks():
    rows = []
    gains = {}
    order = {"G1": ("REF", "P1", "P24"), "G2": ("P24", "P1", "REF")}
    for group, configs in order.items():
        for config in configs:
            for interval in CAPTURE_INTERVALS:
                value = {"REF": 1.0, "P1": 0.9, "P24": 0.2}[config]
                for repeat in range(1, 6):
                    rows.append({"group": group, "config": config, "interval_index": interval, "repeat": repeat, "seconds": value, "exact": True})
                if config == "P24":
                    gains[f"{group}:{interval}"] = 0.8
    result = {"mechanism": {"formal_calls": 90, "warmup_calls": 36, "exact_by_config": {"REF": True, "P1": True, "P24": True}, "config_order": {"G1": ["REF", "P1", "P24"], "G2": ["P24", "P1", "REF"]}, "formal_rows": rows, "p24_block_gains": gains}}
    assert _gain_map(result) == gains


def test_mechanism_gain_validator_rejects_duplicate_repeat_ids():
    rows = []
    gains = {f"{group}:{interval}": 0.8 for group in ("G1", "G2") for interval in CAPTURE_INTERVALS}
    order = {"G1": ("REF", "P1", "P24"), "G2": ("P24", "P1", "REF")}
    for group, configs in order.items():
        for config in configs:
            for interval in CAPTURE_INTERVALS:
                value = {"REF": 1.0, "P1": 0.9, "P24": 0.2}[config]
                repeats = (1, 2, 3, 4, 4)  # same row count, but no fixed 1..5 set
                rows.extend({"group": group, "config": config, "interval_index": interval, "repeat": repeat, "seconds": value, "exact": True} for repeat in repeats)
    result = {"mechanism": {"formal_calls": 90, "warmup_calls": 36, "exact_by_config": {"REF": True, "P1": True, "P24": True}, "config_order": {"G1": ["REF", "P1", "P24"], "G2": ["P24", "P1", "REF"]}, "formal_rows": rows, "p24_block_gains": gains}}
    with pytest.raises(MechanismEvidenceError, match="REPEAT_SET"):
        _gain_map(result)


def test_c_child_failure_classification_preserves_resource_priority():
    scientific = {"classification": "SCIENTIFIC_NONEXACT"}
    assert c_fullsize_hpc.full_hpc._child_failure_classification({"exit_code": 1, "stop_reason": "WORK_DEADLINE_3480_SECONDS"}, scientific) == "ENGINEERING_OR_RESOURCE_FAILURE"
    assert c_fullsize_hpc.full_hpc._child_failure_classification({"exit_code": 1, "stop_reason": None}, scientific) == "SCIENTIFIC_NONEXACT"
    assert c_fullsize_hpc.full_hpc._child_failure_classification({"exit_code": 1, "stop_reason": None}, {"classification": "ENGINEERING_OR_RESOURCE_FAILURE"}) == "ENGINEERING_OR_RESOURCE_FAILURE"


def test_c_fullsize_launch_handles_success_science_and_engineering(monkeypatch, tmp_path: Path):
    def exercise(classification=None, stop_reason=None):
        report = tmp_path / ("report_" + (classification or "success") + ("_stop" if stop_reason else ""))
        report.mkdir()

        class FakeProcess:
            pid = 123

        def fake_popen(argv, **kwargs):
            work = Path(argv[argv.index("--work") + 1])
            work.mkdir(parents=True)
            if classification:
                (work / "ENTRY_FAILURE.json").write_text(json.dumps({"classification": classification}), encoding="utf-8")
            else:
                (work / "CASE_RECEIPT.json").write_text(json.dumps({"execution_domain": "S46F_FULLSIZE", "shape": list(SHAPE)}), encoding="utf-8")
            return FakeProcess()

        monkeypatch.setattr(c_fullsize_hpc.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(c_fullsize_hpc, "_supervise_process", lambda *args, **kwargs: {"exit_code": 1 if classification else 0, "stop_reason": stop_reason})
        if classification == "SCIENTIFIC_NONEXACT" and stop_reason is None:
            value = c_fullsize_hpc._launch(report, case="R8_UNCAPTURED", capture=False, reference_output="/ref/output", deadline=float("inf"), node={}, input_manifest=report / "FULL_INPUT_MANIFEST.json")
            assert value["status"] == "SCIENTIFIC_NONEXACT"
        elif classification:
            with pytest.raises(RuntimeError, match="S46F_C_FULL_CHILD_FAILURE"):
                c_fullsize_hpc._launch(report, case="R8_UNCAPTURED", capture=False, reference_output="/ref/output", deadline=float("inf"), node={}, input_manifest=report / "FULL_INPUT_MANIFEST.json")
        else:
            value = c_fullsize_hpc._launch(report, case="R8_UNCAPTURED", capture=False, reference_output="/ref/output", deadline=float("inf"), node={}, input_manifest=report / "FULL_INPUT_MANIFEST.json")
            assert value["execution_domain"] == "S46F_FULLSIZE"

    exercise()
    exercise("SCIENTIFIC_NONEXACT")
    exercise("ENGINEERING_OR_RESOURCE_FAILURE")
    exercise("SCIENTIFIC_NONEXACT", "WORK_DEADLINE_3480_SECONDS")


def test_finalize_evidence_appends_terminal_receipts_without_overwrite(monkeypatch, tmp_path: Path):
    root = tmp_path / "evidence"
    root.mkdir()
    pending = {
        "ADMISSION.json": {"schema": "filament.cpu_s4_6f.c_fullsize_mechanism.evidence.v1", "status": "PENDING_SCHEDULER_TERMINAL", "execution_commit": "abc"},
        "SCHEDULER_ACCOUNTING.json": {"scheduler_terminal_state": "AWAIT_SACCT"},
        "RESOURCE_RECEIPT.json": {"terminal": False},
        "RESULT.json": {"mechanism": {"formal_calls": 90, "warmup_calls": 36}, "execution_commit": "abc"},
        "EXECUTION_SOURCE_MANIFEST.json": {},
    }
    for name, value in pending.items():
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    original = {name: (root / name).read_bytes() for name in pending}
    monkeypatch.setattr(c_fullsize_hpc, "_gain_map", lambda value: {"G1:0": 0.8})
    monkeypatch.setattr(c_fullsize_hpc, "validate_mechanism_evidence", lambda value: {"status": "PASS_C_FULLSIZE_MECHANISM_ADMISSION"})
    result = c_fullsize_hpc.finalize_evidence(root, {"scheduler_terminal_state": "COMPLETED", "exit_code": "0:0"})
    assert result["status"] == "PASS_C_FULLSIZE_MECHANISM_ADMISSION"
    assert (root / "ADMISSION_TERMINAL.json").is_file()
    assert (root / "SCHEDULER_ACCOUNTING_TERMINAL.json").is_file()
    assert (root / "RESOURCE_RECEIPT_TERMINAL.json").is_file()
    assert all((root / name).read_bytes() == raw for name, raw in original.items())


def test_full_input_contract_mismatch_fails_closed():
    base = dict(shape=list(SHAPE), dtype="complex128", native_layout="C")
    binding = dict(canonical_manifest_raw_sha256="canon", reference_shape=list(SHAPE), reference_intervals=8, reference_role="L1_LINEAR_FACTOR_CACHE")
    input_contract = dict(base, schema="filament.cpu_s4_6f.input_contract.v1", classification="external", hash_scope="raw_bytes", expected_input_fingerprints={"E": "a"}, canonical_manifest_raw_sha256="canon")
    full_manifest = dict(base, schema="filament.cpu_s4_6f.full_input_manifest.v1", classification="external", hash_scope="raw_bytes", fingerprints={"E": "tampered"}, canonical_manifest_raw_sha256="canon")
    reference = {"schema": "filament.cpu_s4_5a.reference_files.v1", "classification": "external", "hash_scope": "raw_bytes", "reference": {"shape": list(SHAPE), "intervals": 8, "role": "L1_LINEAR_FACTOR_CACHE", "output": "/ref/out", "boundary_path": "/ref/boundary", "stages": "/ref/stages"}}
    with pytest.raises(RuntimeError, match="FINGERPRINT_BINDING"):
        c_fullsize_entry._validate_full_input_contracts(binding, input_contract, full_manifest, reference)


def test_full_input_contract_accepts_frozen_5a_reference_without_top_classification():
    base = dict(shape=list(SHAPE), dtype="complex128", native_layout="C")
    binding = dict(canonical_manifest_raw_sha256="canon", reference_shape=list(SHAPE), reference_intervals=8, reference_role="L1_LINEAR_FACTOR_CACHE")
    fingerprints = {"E": "a"}
    input_contract = dict(base, schema="filament.cpu_s4_6f.input_contract.v1", classification="external", hash_scope="raw_bytes", expected_input_fingerprints=fingerprints, canonical_manifest_raw_sha256="canon")
    full_manifest = dict(base, schema="filament.cpu_s4_6f.full_input_manifest.v1", classification="external", hash_scope="raw_bytes", fingerprints=fingerprints, canonical_manifest_raw_sha256="canon")
    reference = {"schema": "filament.cpu_s4_5a.reference_files.v1", "hash_scope": "raw_bytes", "reference": {"shape": list(SHAPE), "intervals": 8, "role": "L1_LINEAR_FACTOR_CACHE", "output": "/ref/out", "boundary_path": "/ref/boundary", "stages": "/ref/stages"}}
    c_fullsize_entry._validate_full_input_contracts(binding, input_contract, full_manifest, reference)


def test_capture_runtime_binding_comes_from_factory_tuple_item():
    runtime = {"c_role": "R", "c_mode": "PURE_OFF_SMALLGRID", "c_candidate_schema": "c.v1", "c_source_binding": {field: "bound" for field in ("path", "relative_path", "classification", "hash_scope", "canonical_lf_sha256", "source_blob", "function", "function_first_line", "raw_sha256")}}
    factory_binding = {"runtime_wrapper": runtime}
    assert c_fullsize_entry._capture_runtime_binding(factory_binding, role="R", mode="PURE_OFF_SMALLGRID") is runtime


def _roundtrip_mechanism_result() -> dict:
    rows = []
    gains = {}
    order = {"G1": ("REF", "P1", "P24"), "G2": ("P24", "P1", "REF")}
    for group, configs in order.items():
        for config in configs:
            for interval in CAPTURE_INTERVALS:
                value = {"REF": 1.0, "P1": 0.9, "P24": 0.2}[config]
                for repeat in range(1, 6):
                    rows.append({"group": group, "config": config, "interval_index": interval, "repeat": repeat, "seconds": value, "exact": True})
                if config == "P24":
                    gains[f"{group}:{interval}"] = 0.8
    return {"schema": RESULT_SCHEMA, "status": "PASS_C_FULLSIZE_MECHANISM_EXACT_GAIN_GATE", "execution_commit": "synthetic-commit", "mechanism": {"formal_calls": 90, "warmup_calls": 36, "exact_by_config": {"REF": True, "P1": True, "P24": True}, "config_order": {key: list(value) for key, value in order.items()}, "formal_rows": rows, "p24_block_gains": gains}}


def test_write_finalize_and_real_validator_roundtrip_without_arrays(tmp_path: Path):
    report = tmp_path / "report"
    report.mkdir()
    repo = Path(__file__).resolve().parents[3]
    source_records = []
    for relative in c_fullsize_hpc.validate_mechanism_evidence.__globals__["REQUIRED_SOURCE_PATHS"]:
        path = repo / relative
        source_records.append({"path": relative, "classification": "tracked_text", "hash_scope": pv.TRACKED_HASH_SCOPE, "canonical_lf_sha256": pv.canonical_lf_sha256(path.read_bytes())})
    execution = {"schema": "synthetic", "execution_commit": "synthetic-commit", "canonical_manifest_raw_sha256": "c" * 64}
    source_manifest = {"head": "synthetic-commit", "records": source_records}
    fingerprints = {"E": "input"}
    input_contract = {"schema": "filament.cpu_s4_6f.input_contract.v1", "classification": "external", "hash_scope": "raw_bytes", "shape": list(SHAPE), "dtype": "complex128", "native_layout": "C", "expected_input_fingerprints": fingerprints, "canonical_manifest_raw_sha256": "c" * 64}
    full_manifest = {"schema": "filament.cpu_s4_6f.full_input_manifest.v1", "classification": "external", "hash_scope": "raw_bytes", "shape": list(SHAPE), "dtype": "complex128", "native_layout": "C", "fingerprints": fingerprints, "canonical_manifest_raw_sha256": "c" * 64}
    reference = {"schema": "filament.cpu_s4_5a.reference_files.v1", "hash_scope": "raw_bytes", "reference": {"shape": list(SHAPE), "intervals": 8, "role": "L1_LINEAR_FACTOR_CACHE", "output": "/ref/out", "boundary_path": "/ref/boundary", "stages": "/ref/stages"}}
    for name, value in (("EXECUTION.json", execution), ("EXECUTION_SOURCE_MANIFEST.json", source_manifest), ("INPUT_CONTRACT.json", input_contract), ("FULL_INPUT_MANIFEST.json", full_manifest), ("REFERENCE_INPUT_RAW_RECORDS.json", reference)):
        (report / name).write_text(json.dumps(value), encoding="utf-8")
    captured_work = report / "captured_work"
    captured_work.mkdir()
    package_root = report / "capture"
    package_root.mkdir()
    raw_path = package_root / "scalar.bin"
    raw_path.write_bytes(b"x")
    raw_record = {"path": str(raw_path), "classification": "external", "hash_scope": "raw_bytes", "size_bytes": 1, "raw_sha256": pv.raw_sha256_file(raw_path)}
    metadata = {"selected_interval_indices": list(CAPTURE_INTERVALS), "expected_total_calls": 8, "record_count": 3, "records": {str(interval): {"input_fingerprints": {"E": {}, "phase": {}, "alpha": {}, "dn_gas": {}}, "output_fingerprint": {}} for interval in CAPTURE_INTERVALS}}
    (package_root / "CAPTURE.json").write_text(json.dumps({"schema": "filament.cpu_s4_6e.c_capture.package.v1", "metadata": metadata, "records": {}, "files": [raw_record], "live_original_array_refs_retained": False, "captured_owned_arrays": True}), encoding="utf-8")
    (captured_work / "CAPTURE_RECEIPT.json").write_text(json.dumps({"capture_package": str(package_root), "source_binding": {"role": "R", "source_identity_digest": "source"}, "capture_metadata": metadata, "raw_files": [raw_record], "diagnostic_outputs": True, "cpu_diagnostics_profile": True, "fine_performance_profile": False}), encoding="utf-8")
    uncaptured = {"comparisons": {"terminal": {"status": "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"}}, "reference_output": "/historical/out", "output": "/uncaptured/out"}
    captured = {"work_root": str(captured_work), "comparisons": {"terminal": {"status": "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"}}, "reference_output": uncaptured["output"], "output": "/captured/out", "diagnostic_outputs": True, "cpu_diagnostics_profile": True, "fine_performance_profile": False}
    evidence = c_fullsize_hpc._write_evidence(report, node={"status": "NODE"}, uncaptured=uncaptured, captured=captured, mechanism=_roundtrip_mechanism_result())
    admission = c_fullsize_hpc.finalize_evidence(evidence, {"scheduler_terminal_state": "COMPLETED", "exit_code": "0:0", "allocated_cpus": 96, "requested_memory": "128G", "gpus": 0})
    assert admission["status"] == "PASS_C_FULLSIZE_MECHANISM_ADMISSION"
    assert (evidence / "ADMISSION.json").exists() and (evidence / "ADMISSION_TERMINAL.json").exists()


def test_local_mechanism_pool_override_is_w1_and_hpc_factory_is_unchanged(monkeypatch):
    from tools.cpu_s4_6e_array_propagation import c_mechanism

    seen = []

    def fake_run(_groups):
        pool = c_mechanism.CPool(24)
        seen.append(pool.count)
        pool.close()
        return {"status": "SCIENTIFIC_NONEXACT"}

    original_pool = c_mechanism.CPool
    monkeypatch.setattr(c_mechanism, "run_local_mechanism", fake_run)
    c_fullsize_entry._run_mechanism({}, local_engineering=True)
    assert seen == [1]
    assert c_mechanism.CPool is original_pool
    c_fullsize_entry._run_mechanism({}, local_engineering=False)
    assert seen == [1, 24]


def test_evidence_validator_rejects_missing_terminal_and_raw_source(tmp_path: Path):
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "ADMISSION.json").write_text(json.dumps({"schema": "filament.cpu_s4_6f.c_fullsize_mechanism.evidence.v1", "status": "PASS_C_FULLSIZE_MECHANISM_EXACT_GAIN_GATE"}), encoding="utf-8")
    with pytest.raises(MechanismEvidenceError, match="ADMISSION|JSON|SOURCE|EXECUTION"):
        validate_mechanism_evidence(root)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable for batch-entry audit")
def test_fullsize_mechanism_batch_entry_is_auditable():
    from tools.hpc_ops.audit_batch_entry import audit_batch_entry
    from .c_fullsize_protocol import REMOTE_PYTHON

    audit, code = audit_batch_entry(Path(__file__).with_name("c_fullsize_cpu.sbatch"), REMOTE_PYTHON, "bash")
    assert code == 0 and audit.get("status") == "passed"


@pytest.mark.skipif(not os.environ.get("S46F_C_FULL_REAL_ENTRY_REPORT"), reason="opt-in fullsize C capture/mechanism local micro entry")
def test_real_local_capture_and_mechanism_opt_in():
    report = Path(os.environ["S46F_C_FULL_REAL_ENTRY_REPORT"]).resolve(strict=True)
    root = Path(os.environ["S46F_C_FULL_REAL_ENTRY_WORK"]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    log_root = root / "logs"
    log_root.mkdir()
    manifest = Path(os.environ.get("S46F_C_FULL_REAL_ENTRY_INPUT_MANIFEST", str(report / "BENCHMARK_INPUT_MANIFEST.json"))).resolve(strict=True)
    python = sys.executable
    base = [python, "-s", "-B", "-m", "tools.cpu_s4_6f_fullsize_parallel.c_fullsize_entry", "--report", str(report), "--role", "R", "--intervals", "8", "--local-engineering", "--input-manifest", str(manifest)]

    def run_logged(argv, name):
        stdout_path = log_root / f"{name}.stdout.log"
        stderr_path = log_root / f"{name}.stderr.log"
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            completed = subprocess.run(argv, stdout=stdout, stderr=stderr, text=True)
        assert completed.returncode == 0, f"{name} failed; inspect {stdout_path} and {stderr_path}"

    uncaptured_work = root / "R8_UNCAPTURED"
    run_logged(base + ["--work", str(uncaptured_work)], "R8_UNCAPTURED")
    uncaptured_receipt = json.loads((uncaptured_work / "CASE_RECEIPT.json").read_text(encoding="utf-8"))
    captured_work = root / "R8_CAPTURED"
    run_logged(base + ["--work", str(captured_work), "--capture", "--capture-root", str(captured_work / "CAPTURE"), "--reference-output", uncaptured_receipt["output"]], "R8_CAPTURED")
    captured_receipt = json.loads((captured_work / "CASE_RECEIPT.json").read_text(encoding="utf-8"))
    mechanism_output = root / "MECHANISM" / "RESULT.json"
    mechanism_output.parent.mkdir()
    run_logged([python, "-s", "-B", "-m", "tools.cpu_s4_6f_fullsize_parallel.c_fullsize_entry", "--report", str(report), "--mechanism", "--local-engineering", "--capture-root", str(captured_work / "CAPTURE"), "--output", str(mechanism_output), "--input-manifest", str(manifest)], "MECHANISM")
    result = json.loads(mechanism_output.read_text(encoding="utf-8"))
    assert uncaptured_receipt["shape"] == list(MICRO_SHAPE)
    assert captured_receipt["comparisons"]["terminal"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
    assert result["mechanism"]["formal_calls"] == FORMAL_CALLS
    assert result["mechanism"]["warmup_calls"] == WARMUP_CALLS
