"""New joint admission and original-clock contracts; propagation is opt-in."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.cpu_s4_6h_combined_parallel import entry, hpc
from tools.cpu_s4_6h_combined_parallel.prepare import _candidate_paths
from tools.cpu_s4_6h_combined_parallel.protocol import CANDIDATE, MICRO_SHAPE, SHAPE, plan


def test_s46h_entry_domains_and_four_window_scope(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    entry.admission(candidate=CANDIDATE, local_engineering=True, shape=MICRO_SHAPE,
                    intervals=2, mode="QUALIFY_BOUNDARY")
    with pytest.raises(RuntimeError, match="LOCAL_MICRO_ONLY"):
        entry.admission(candidate=CANDIDATE, local_engineering=True, shape=SHAPE,
                        intervals=8, mode="PURE_OFF_FULLSIZE")
    with pytest.raises(RuntimeError, match="TWO_QUAL_OR_EIGHT_OFF"):
        entry.admission(candidate=CANDIDATE, local_engineering=True, shape=MICRO_SHAPE,
                        intervals=16, mode="PURE_OFF_FULLSIZE")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    entry.admission(candidate=CANDIDATE, local_engineering=False, shape=SHAPE,
                    intervals=8, mode="PURE_OFF_FULLSIZE")
    frozen = plan()
    assert [(row["role"], row["intervals"]) for row in frozen["matrix"]] == [
        ("R", 2), ("X", 2), ("R", 8), ("X", 8)]
    assert frozen["resources"]["work_deadline_s"] == 3480
    assert frozen["pointwise_groups"]["8"] == dict(INTENSITY_P24=96, EQ10_P48=128, PHASE_ABSORB_P24=16)


def test_s46h_joint_manifest_includes_actual_pool_and_worker_sources():
    repo = Path(__file__).resolve().parents[3]
    paths = _candidate_paths(repo)
    for name in ("cpu_s4_6c_array_coverage/chains.py",
                 "cpu_s4_6d_eq10_propagation/pool.py",
                 "cpu_s4_6e_array_propagation/pool.py",
                 "cpu_s4_6e_array_propagation/c_pool.py",
                 "cpu_s4_6h_combined_parallel/combined_candidate.py",
                 "cpu_s4_6h_combined_parallel/hpc.py"):
        assert "Filament_python/tools/" + name in paths


def test_s46h_complete_propagation_is_not_body_sum(tmp_path):
    rows = []
    for name, body, app, prop in (("QUAL_R8", 70., 110., 100.), ("QUAL_X8", 63., 106., 93.)):
        case = dict(case_id=name, later_body_s=body, application_service_s=app,
                    first_body_s=10., first_POST_latency_s=20.,
                    initialization_total_before_propagation_s=5.)
        work = tmp_path / "runs/S46H" / CANDIDATE / name / "work"
        work.mkdir(parents=True)
        (work / "RAW_TIMING.json").write_text(json.dumps(dict(b=dict(records=[
            dict(label="PROPAGATION", wall_inclusive_ns=int(prop*1e9))]))), encoding="utf-8")
        rows.append(case)
    result = hpc._paired_observation(tmp_path, *rows)
    assert result["propagation"]["reference_s"] == 100.
    assert result["propagation"]["saved_s"] == 7.
    assert result["body_2_to_8"]["reference_s"] == 70.
    assert result["application"]["saved_s"] == 4.


def test_s46h_resource_failure_precedes_nested_science_text():
    error = RuntimeError("S46H_STOP_CHILD_FAILURE: {'classification':'SCIENTIFIC_NONEXACT'}")
    assert hpc._classify_failure(error) == "ENGINEERING_OR_RESOURCE_FAILURE"
    assert hpc._classify_failure(RuntimeError("S46F_STOP_TERMINAL_NONEXACT")) == "SCIENTIFIC_NONEXACT"


def test_s46h_cleanup_failure_note_takes_priority_in_entry_receipt(monkeypatch, tmp_path):
    from . import combined_candidate as combined

    error = RuntimeError("STOP_FIRST_SCIENCE_NONEXACT")

    def fail(*args, **kwargs):
        raise error

    def fail_cleanup(*args, **kwargs):
        raise RuntimeError("injected join failure")

    monkeypatch.setattr(combined, "finalize_factory", fail_cleanup)
    monkeypatch.setattr(entry, "run", combined._wrap_propagator_for_failure(fail, object()))
    with pytest.raises(RuntimeError, match="STOP_FIRST_SCIENCE_NONEXACT") as caught:
        entry.main(["--report", str(tmp_path), "--work", str(tmp_path), "--role", "X",
                    "--intervals", "2", "--local-engineering"])
    assert caught.value is error
    receipt = json.loads((tmp_path / "ENTRY_FAILURE.json").read_text(encoding="utf-8"))
    assert receipt["classification"] == "ENGINEERING_OR_RESOURCE_FAILURE"
    assert receipt["notes"] == ["S46H_CLOSE_FAILURE:RuntimeError"]
    assert "injected join failure" not in receipt["error"]
    assert hpc._classify_failure(error) == "ENGINEERING_OR_RESOURCE_FAILURE"


@pytest.mark.skipif(not os.environ.get("S46H_REAL_ENTRY_REPORT"), reason="opt-in real W1 entry only")
def test_s46h_real_micro_entry_opt_in():
    report = Path(os.environ["S46H_REAL_ENTRY_REPORT"]).resolve(strict=True)
    work = Path(os.environ["S46H_REAL_ENTRY_WORK"]).resolve()
    from tools.cpu_s4_3g3_smallgrid import benchmark as sg
    input_manifest = os.environ.get("S46H_REAL_ENTRY_INPUT_MANIFEST")
    if input_manifest is None:
        manifest = report / "BENCHMARK_INPUT_MANIFEST.json"
        if not manifest.is_file():
            sg.input_manifest(report, shape=MICRO_SHAPE)
        input_manifest = str(manifest)
    work.mkdir(parents=True, exist_ok=False)
    results = {}
    for label, role, n in (("R2", "R", 2), ("X2", "X", 2), ("R8", "R", 8), ("X8", "X", 8)):
        target = work / label
        argv = [sys.executable, "-s", "-B", "-m", "tools.cpu_s4_6h_combined_parallel.entry",
                "--report", str(report), "--work", str(target), "--candidate", CANDIDATE,
                "--role", role, "--intervals", str(n), "--local-engineering",
                "--input-manifest", input_manifest]
        if role == "X":
            reference = results["R" + str(n)]
            argv += ["--reference-output", reference["output"]]
            if n == 2:
                argv += ["--reference-boundary", reference["boundary_path"], "--reference-stages", reference["stages"]]
        with (work / (label + ".stdout.log")).open("xb") as stdout, (work / (label + ".stderr.log")).open("xb") as stderr:
            process = subprocess.run(argv, env=os.environ.copy(), stdout=stdout, stderr=stderr)
        assert process.returncode == 0, f"{label} exit {process.returncode}; see {work}"
        result = json.loads((target / "CASE_RECEIPT.json").read_text(encoding="utf-8"))
        assert result["execution_domain"] == "S46H_LOCAL_MICRO"
        assert result["route"]["nH"] == 2*n and result["route"]["math_dispatches"] == 16*n
        assert result["route"]["fresh_responses"] == 4*n and result["route"]["fallback"] == 0
        assert result["pointwise_calls"] == (30*n if role == "X" else 0)
        aggregate = result["pointwise_pool"]
        entry.require(aggregate["pool"]["closed"] and not aggregate["pool"]["active_thread_ids"], "MICRO_ALL_THREE_JOIN")
        from tools.cpu_s4_6h_combined_parallel.measurement import validate_pools
        validate_pools(CANDIDATE, role, n, aggregate["pool"], aggregate["activity"], local_engineering=True)
        if role == "X":
            assert result["comparisons"]["terminal"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
            if n == 2:
                assert result["comparisons"]["boundary"]["status"] == "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT"
                assert result["comparisons"]["stages"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
        results[label] = result
