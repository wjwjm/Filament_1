"""Low-cost guards for the S4-6J formal protocol and analysis gates."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import FunctionType, SimpleNamespace

import pytest

from tools.cpu_s4_6j_alpha_factory import alpha_formal_analysis as analysis
from tools.cpu_s4_6j_alpha_factory import alpha_formal_entry
from tools.cpu_s4_6j_alpha_factory.alpha_formal_protocol import (
    CANDIDATE,
    matrix,
    pairings,
    plan,
)
from tools.cpu_s4_6j_alpha_factory.alpha_mechanism_admission import (
    RUNTIME_PATHS,
    SCHEMA as ADMISSION_SCHEMA,
    STATUS as ADMISSION_STATUS,
    MechanismAdmissionError,
    _scheduler_ok,
)


def _cases(*, body_delta: float = -0.01, app_delta: float = -0.01) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in matrix(CANDIDATE):
        body = 100.0
        app = 200.0
        if row["role"] == "X":
            body *= 1.0 + body_delta
            app *= 1.0 + app_delta
        result[row["case"]] = {
            "case": row["case"],
            "role": row["role"],
            "intervals": row["intervals"],
            "exact_gate_pass": True,
            "later_body_s": body,
            "post_gap_median_s": body,
            "application_service_s": app,
            "propagation_service_s": body + 10.0,
            "clock_domain_pass": True,
            "first_POST_latency_s": 1.0,
            "first_body_s": body,
            "initialization_within_application_s": 0.2,
            "source_factory_preparation_s": 0.1,
            "pointwise_pool": {"pool": {}, "activity": []},
        }
    # The two RR controls are intentionally stable but nonzero, so the drift
    # screen remains a real guard rather than a vacuous zero comparison.
    result["CTRL_EARLY_R2"]["later_body_s"] = 100.1
    result["CTRL_LATE_R2"]["later_body_s"] = 100.1
    return result


def test_formal_protocol_reuses_frozen_matrix_and_resources() -> None:
    assert [row["case"] for row in matrix(CANDIDATE)] == [
        "QUAL_R2", "QUAL_X2", "CTRL_EARLY_R1", "CTRL_EARLY_R2",
        "K1_R1", "K1_X1", "K1_X2", "K1_R2", "K2_X1", "K2_R1",
        "K2_R2", "K2_X2", "CTRL_LATE_R1", "CTRL_LATE_R2",
    ]
    assert len(pairings(CANDIDATE)) == 4
    assert plan()["resources"]["work_deadline_s"] == 7080
    assert plan()["resources"]["walltime_s"] == 7200


def test_one_percent_positive_body_and_application_gain_passes_new_gate() -> None:
    result = analysis.analyze(_cases(body_delta=-0.01, app_delta=-0.01))
    assert result["gates"]["body_paired_median_strictly_positive"] is True
    assert result["gates"]["application_saved_median_strictly_positive"] is True
    assert result["status"] == "FULLSIZE_ALPHA_PERFORMANCE_GATES_PASS_MANUAL_REVIEW_PENDING"


@pytest.mark.parametrize("delta", [0.0, 0.01])
def test_zero_or_negative_body_gain_does_not_pass(delta: float) -> None:
    result = analysis.analyze(_cases(body_delta=delta, app_delta=-0.01))
    assert result["gates"]["body_paired_median_strictly_positive"] is False
    assert result["status"] == "NO_POSITIVE_FULL_BODY_GAIN"


def test_application_net_negative_is_rejected_even_with_body_gain() -> None:
    result = analysis.analyze(_cases(body_delta=-0.01, app_delta=0.01))
    assert result["gates"]["body_paired_median_strictly_positive"] is True
    assert result["gates"]["application_saved_median_strictly_positive"] is False
    assert result["status"] == "FULLSIZE_ALPHA_APPLICATION_GUARD_FAILURE"


def test_case_identity_and_scientific_exact_are_hard_stops() -> None:
    wrong_role = _cases()
    wrong_role["K1_R1"]["role"] = "X"
    with pytest.raises(RuntimeError, match="FIXED_CASE_IDENTITY"):
        analysis.analyze(wrong_role)
    nonexact = _cases()
    nonexact["K1_X1"]["exact_gate_pass"] = False
    with pytest.raises(RuntimeError, match="ALL_CASES_EXACT"):
        analysis.analyze(nonexact)


def test_formal_entry_disables_capture_and_mechanism(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_verify(*args, **kwargs):
        assert seen.get("inside_application") is True
        seen["source_checks"] = int(seen.get("source_checks", 0)) + 1
        return {"execution_commit": "formal", "mechanism_admission_raw_sha256": "admission",
            "mechanism_admission": {"status": "PASS", "capture_execution_commit": "capture"}}

    def fake_run(args):
        seen["inside_application"] = True
        assert plan()["phase"] == "FORMAL_ALPHA_FOURTEEN"
        verify_source(args.report)  # resolved through the private run globals
        seen["capture"] = args.capture
        seen["mechanism"] = args.mechanism
        return {
            "status": "PASS_S46J_WINDOW_EXACT",
            "candidate": CANDIDATE,
            "role": args.role,
            "intervals": args.intervals,
            "shape": list(args.shape),
        }

    monkeypatch.setattr(alpha_formal_entry, "verify_source", fake_verify)
    monkeypatch.setattr(alpha_formal_entry.alpha_entry, "run", fake_run)
    args = SimpleNamespace(
        report=tmp_path / "report",
        work=tmp_path / "work",
        candidate=CANDIDATE,
        role="R",
        intervals=2,
        local_engineering=True,
        shape=(384, 17, 19),
        capture=True,
        mechanism=True,
        capture_root=tmp_path / "capture",
    )
    args.work.mkdir()
    result = alpha_formal_entry.run(args)
    assert seen == {"inside_application": True, "source_checks": 1, "capture": False, "mechanism": False}
    assert result["capture_enabled"] is False
    assert result["mechanism_enabled"] is False
    assert result["formal_execution_commit"] == "formal"
    assert result["capture_execution_commit"] == "capture"


def test_formal_source_clone_keeps_distinct_plan_and_node_gate() -> None:
    bound = alpha_formal_entry._bind(alpha_formal_entry.alpha_entry.verify_source)
    assert bound.__globals__["plan"] is plan
    assert bound.__globals__["FULL_GATE_STATUS"] == "PASS_S46J_ALPHA_FORMAL_SOURCE_RESOURCE_INPUT_GATES"
    assert alpha_formal_entry.alpha_entry.FULL_GATE_STATUS == "PASS_S46J_ALPHA_FULLSIZE_SOURCE_RESOURCE_INPUT_GATES"
    assert alpha_formal_entry.read is alpha_formal_entry.alpha_entry.read


def test_cold_preflight_does_not_require_node_receipt(tmp_path) -> None:
    from . import alpha_formal_hpc as hpc
    from .alpha_formal_protocol import THREAD_ENV, VERSIONS

    class SourceBoundaryReached(Exception):
        pass

    def verify(*args, **kwargs):
        assert not (tmp_path / "NODE_GATE.json").exists()
        assert kwargs["require_node"] is False
        raise SourceBoundaryReached

    interpreter = str(tmp_path / "python")
    namespace = dict(hpc.preflight.__globals__, REMOTE_PYTHON=interpreter,
        getpass=SimpleNamespace(getuser=lambda: "t0s000727"),
        sys=SimpleNamespace(executable=interpreter, version_info=(3, 12, 8)),
        importlib=SimpleNamespace(import_module=lambda name: SimpleNamespace(__version__=VERSIONS[name])),
        os=SimpleNamespace(environ=dict(THREAD_ENV, UPPE_USE_GPU="0")),
        entry=SimpleNamespace(verify_source=verify))
    cold_preflight = FunctionType(hpc.preflight.__code__, namespace,
        argdefs=hpc.preflight.__defaults__)
    with pytest.raises(SourceBoundaryReached):
        cold_preflight(tmp_path)


def test_scheduler_requires_completed_zero_exit_steps() -> None:
    admission = {"capture_job_id": "12345", "scheduler": {"job_id": "12345", "all_steps_completed": True, "steps": [{"job_id": job, "state": "COMPLETED", "exit_code": "0:0"} for job in ("12345", "12345.batch", "12345.0")]}}
    assert _scheduler_ok(admission)["all_steps_completed"] is True
    broken = copy.deepcopy(admission)
    broken["scheduler"]["steps"][0]["exit_code"] = "1:0"
    with pytest.raises(MechanismAdmissionError, match="SCHEDULER_EXIT"):
        _scheduler_ok(broken)
    incomplete = copy.deepcopy(admission)
    incomplete["scheduler"]["steps"] = incomplete["scheduler"]["steps"][:1]
    with pytest.raises(MechanismAdmissionError, match="STEPS_MISSING"):
        _scheduler_ok(incomplete)


def test_admission_surface_is_explicit_and_not_ancestor_only() -> None:
    assert ADMISSION_SCHEMA == "filament.cpu.s4_6j.alpha_mechanism_admission.v1"
    assert ADMISSION_STATUS == "PASS_S46J_ALPHA_ORIGINAL_SIZE_MECHANISM"
    assert len(RUNTIME_PATHS) >= 12
    assert "Filament_python/tools/cpu_s4_6j_alpha_factory/alpha_capture_hpc.py" in RUNTIME_PATHS
    assert all("alpha_" in path or "6f_fullsize" in path or "6i_residual" in path for path in RUNTIME_PATHS)


@pytest.mark.parametrize("classification", ["SCIENTIFIC_NONEXACT", "ENGINEERING_OR_RESOURCE_FAILURE"])
def test_formal_controller_preserves_child_failure(tmp_path, monkeypatch, classification):
    from . import alpha_formal_hpc as hpc
    from .alpha_formal_protocol import SHAPE
    values = {"FULL_INPUT_MANIFEST.json": dict(shape=list(SHAPE), fingerprints={}),
        "INPUT_CONTRACT.json": dict(expected_input_fingerprints={}),
        "REFERENCE_INPUT_RAW_RECORDS.json": dict(reference=dict(boundary_path="boundary", stages="stages"))}
    for name, value in values.items():
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(hpc, "_require_fresh_controller", lambda path: None)
    monkeypatch.setattr(hpc, "node_gate", lambda *args: dict(job_id="synthetic"))
    failed = dict(status=classification, entry_failure=dict(error="probe"))
    monkeypatch.setattr(hpc, "launch", lambda *args, **kwargs: failed)
    with pytest.raises(RuntimeError, match=classification):
        hpc.run(tmp_path)
    receipt = json.loads((tmp_path / "JOB_FAILURE.json").read_text())
    assert receipt["active_case"] == "QUAL_R2"
    assert receipt["failed_case"] == failed
    assert receipt["classification"] == classification


@pytest.mark.skipif(
    not os.environ.get("S46J_ALPHA_FORMAL_ENTRY_REPORT"),
    reason="opt-in actual W1 formal entry only",
)
def test_real_w1_formal_entry_opt_in() -> None:
    """Run R2/X2/R8/X8 through the formal wrapper when a real report is supplied.

    The report must already contain the independently produced mechanism
    admission.  This test never fabricates that admission and is skipped in
    ordinary local targeted runs.
    """

    report = Path(os.environ["S46J_ALPHA_FORMAL_ENTRY_REPORT"]).resolve(strict=True)
    work = Path(os.environ["S46J_ALPHA_FORMAL_ENTRY_WORK"]).resolve()
    work.mkdir(parents=True, exist_ok=False)
    admission = report / "MECHANISM_ADMISSION.json"
    assert admission.is_file(), "formal W1 entry requires the real admission receipt"
    input_manifest = os.environ.get("S46J_ALPHA_FORMAL_ENTRY_INPUT_MANIFEST")
    if input_manifest is None:
        from tools.cpu_s4_3g3_smallgrid import benchmark as sg
        candidate = report / "BENCHMARK_INPUT_MANIFEST.json"
        if not candidate.is_file():
            sg.input_manifest(report, shape=(384, 17, 19))
        input_manifest = str(candidate)

    cases: dict[str, dict[str, object]] = {}
    for label, role, intervals in (
        ("R2", "R", 2),
        ("X2", "X", 2),
        ("R8", "R", 8),
        ("X8", "X", 8),
    ):
        case_work = work / label
        argv = [
            sys.executable,
            "-s",
            "-B",
            "-m",
            "tools.cpu_s4_6j_alpha_factory.alpha_formal_entry",
            "--report",
            str(report),
            "--work",
            str(case_work),
            "--candidate",
            CANDIDATE,
            "--role",
            role,
            "--intervals",
            str(intervals),
            "--local-engineering",
        ]
        if input_manifest:
            argv.extend(("--input-manifest", input_manifest))
        reference = cases.get(f"R{intervals}")
        if reference is not None:
            argv.extend(("--reference-output", str(reference["output"])))
            if intervals == 2:
                argv.extend(("--reference-boundary", str(reference["boundary_path"])))
                argv.extend(("--reference-stages", str(reference["stages"])))
        with (work / (label + ".stdout.log")).open("xb") as out, (work / (label + ".stderr.log")).open("xb") as err:
            completed = subprocess.run(argv, env=os.environ.copy(), stdout=out, stderr=err, check=False)
        assert completed.returncode == 0, f"{label}: formal W1 entry failed"
        receipt_path = case_work / "FORMAL_CASE_RECEIPT.json"
        assert receipt_path.is_file()
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["formal_phase"] == "FORMAL_ALPHA_FOURTEEN"
        assert receipt["capture_enabled"] is False
        assert receipt["mechanism_enabled"] is False
        assert receipt["shape"] == [384, 17, 19]
        assert receipt["pointwise_calls"] == 38 * intervals
        assert receipt["alpha_rounds"] == (5 * intervals if role == "X" else 0)
        assert receipt["route"]["math_dispatches"] == 16 * intervals
        assert receipt["route"]["fallback"] == 0
        assert receipt["pointwise_pool"]["pool"]["closed"]
        assert not receipt["pointwise_pool"]["pool"]["active_thread_ids"]
        if reference:
            assert receipt["comparisons"]["terminal"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
            if intervals == 2:
                assert receipt["comparisons"]["boundary"]["status"] == "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT"
                assert receipt["comparisons"]["stages"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
        cases[label] = receipt
    from tools.cpu_s4_3g3_smallgrid import benchmark as sg
    sg.write(work / "LOCAL_FORMAL_ACCEPTANCE.json", dict(status="PASS_W1_FORMAL_ENGINEERING",
        shape=[384, 17, 19], cases=list(cases), scientific_qualification=False,
        original_size_admission_replaced_by_local_test=False))
