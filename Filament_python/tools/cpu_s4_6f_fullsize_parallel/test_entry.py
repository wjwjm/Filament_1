from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from .entry import _canonical_bundle_from_contract, _restore_original_cpu_diagnostic_profile, admission, input_domain
from .hpc import _child_failure_classification, _classify_failure, _require_fresh_controller, validate_physical_topology
from .protocol import MICRO_SHAPE, SHAPE


def test_admission_has_only_two_qualification_or_eight_off_windows(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    admission(candidate="EQ10_P48", local_engineering=True, shape=MICRO_SHAPE, intervals=2, mode="QUALIFY_BOUNDARY")
    admission(candidate="INTENSITY_P24", local_engineering=True, shape=MICRO_SHAPE, intervals=8, mode="PURE_OFF_FULLSIZE")
    admission(candidate="PHASE_ABSORB_P24", local_engineering=True, shape=MICRO_SHAPE, intervals=2, mode="QUALIFY_BOUNDARY")
    with pytest.raises(RuntimeError, match="TWO_QUAL_OR_EIGHT_OFF"):
        admission(candidate="EQ10_P48", local_engineering=True, shape=MICRO_SHAPE, intervals=1, mode="PURE_OFF_FULLSIZE")
    with pytest.raises(RuntimeError, match="LOCAL_MICRO_ONLY"):
        admission(candidate="EQ10_P48", local_engineering=True, shape=SHAPE, intervals=2, mode="QUALIFY_BOUNDARY")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(RuntimeError, match="PHASE2_NOT_IMPLEMENTED"):
        admission(candidate="PHASE_ABSORB_P24", local_engineering=False, shape=SHAPE, intervals=2, mode="QUALIFY_BOUNDARY")


def test_input_domain_requires_complex128_c_layout():
    good = {"shape": list(MICRO_SHAPE), "dtype": "complex128", "native_layout": "C"}
    input_domain(good, local_engineering=True)
    with pytest.raises(RuntimeError, match="DTYPE_LAYOUT"):
        input_domain({**good, "dtype": "complex64"}, local_engineering=True)


def test_external_remote_paths_are_not_rebased_on_windows(tmp_path: Path):
    from .prepare import _external_path

    assert _external_path("/publicfs01/fs1-t/home/t0s000727/reference") == "/publicfs01/fs1-t/home/t0s000727/reference"
    assert _external_path(tmp_path / "reference") == str((tmp_path / "reference").resolve())
    with pytest.raises(ValueError, match="LOST_POSIX_ROOT"):
        _external_path(Path("\\publicfs01\\fs1-t\\reference"))


def test_original_cpu_diagnostic_profile_is_restored():
    prop = SimpleNamespace(diag_bk_nee_profile=False)
    ctx = {"prop": prop}
    receipt = _restore_original_cpu_diagnostic_profile(ctx)
    assert prop.diag_bk_nee_profile is True
    assert receipt["loader_value"] is False
    assert receipt["accepted_original_cpu_value"] is True


def test_fullsize_topology_gate_requires_distinct_physical_four_numa():
    rows = "\n".join(f"{index},{index},0,{index // 24}" for index in range(96))
    receipt = validate_physical_topology("Intel(R) Xeon(R) Platinum 9242", rows, list(range(96)))
    assert receipt["physical_cores"] == 96
    assert receipt["numa_nodes"] == ["0", "1", "2", "3"]
    with pytest.raises(RuntimeError, match="VALIDATED_CPU_MODEL"):
        validate_physical_topology("other", rows, list(range(96)))


def test_controller_refuses_previous_terminal_state_without_writing(tmp_path: Path):
    (tmp_path / "RESULT.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ALREADY_EXECUTED"):
        _require_fresh_controller(tmp_path)
    assert not (tmp_path / "JOB_FAILURE.json").exists()


def test_supervisor_preserves_resource_priority_before_scientific_child_failure():
    scientific = {"classification": "SCIENTIFIC_NONEXACT", "error": "STOP_FIRST_SCIENCE_NONEXACT"}
    assert _child_failure_classification({"exit_code": 1, "stop_reason": None}, scientific) == "SCIENTIFIC_NONEXACT"
    assert _child_failure_classification({"exit_code": 1, "stop_reason": None}, {"classification": "ENGINEERING_OR_RESOURCE_FAILURE"}) == "ENGINEERING_OR_RESOURCE_FAILURE"
    assert _child_failure_classification({"exit_code": 1, "stop_reason": "ACTUAL_MEMORY_GUARD_80_PERCENT"}, scientific) == "ENGINEERING_OR_RESOURCE_FAILURE"
    assert _classify_failure(RuntimeError("S46F_STOP_CHILD_FAILURE:" + repr(scientific))) == "ENGINEERING_OR_RESOURCE_FAILURE"
    assert _classify_failure(RuntimeError("S46F_STOP_TERMINAL_NONEXACT")) == "SCIENTIFIC_NONEXACT"


def test_canonical_bundle_contract_never_guesses_science_root(tmp_path: Path):
    bundle = tmp_path / "384x351x301"
    bundle.mkdir()
    (bundle / "INPUT_MANIFEST.json").write_text("{}\n", encoding="utf-8")
    assert _canonical_bundle_from_contract({"canonical_input_manifest": str(bundle)}) == bundle.resolve()
    assert _canonical_bundle_from_contract({"canonical_root": str(bundle)}) == bundle.resolve()
    metadata = tmp_path / "CANONICAL_INPUT_MANIFEST.json"
    metadata.write_text("{\"bundle_root\": " + repr(str(bundle)).replace("'", "\"") + "}\n", encoding="utf-8")
    # The metadata path is accepted only through its explicit bundle_root.
    assert _canonical_bundle_from_contract({"canonical_input_manifest": str(metadata)}) == bundle.resolve()
    with pytest.raises(RuntimeError, match="BUNDLE_PATH_MISSING"):
        _canonical_bundle_from_contract({"canonical_input_manifest": str(tmp_path / "missing.json")})


@pytest.mark.skipif(not os.environ.get("S46F_REAL_ENTRY_REPORT"), reason="opt-in real micro entry only")
def test_real_micro_entry_opt_in():
    """Parent supplies a fresh report/work root; never runs by default.

    Each case is a separate interpreter, factory, and cache.  The four cases
    deliberately exercise both qualification and eight-interval activity for
    the selected semantic candidate without making a performance claim.
    """

    report = Path(os.environ["S46F_REAL_ENTRY_REPORT"]).resolve(strict=True)
    work = Path(os.environ["S46F_REAL_ENTRY_WORK"]).resolve()
    candidate = os.environ.get("S46F_REAL_ENTRY_CANDIDATE", "INTENSITY_P24")
    from tools.cpu_s4_3g3_smallgrid import benchmark as sg

    input_manifest = os.environ.get("S46F_REAL_ENTRY_INPUT_MANIFEST")
    if input_manifest is None:
        manifest_path = report / "BENCHMARK_INPUT_MANIFEST.json"
        if not manifest_path.is_file():
            sg.input_manifest(report, shape=MICRO_SHAPE)
        input_manifest = str(manifest_path)
    results = {}
    for label, role, intervals in (("R2", "R", 2), ("X2", "X", 2), ("R8", "R", 8), ("X8", "X", 8)):
        case_work = work / label
        env = os.environ.copy()
        env.update(NUMBA_NUM_THREADS="1", NUMBA_THREADING_LAYER="workqueue", UPPE_USE_GPU="0")
        argv = [sys.executable, "-s", "-B", "-m", "tools.cpu_s4_6f_fullsize_parallel.entry", "--report", str(report), "--work", str(case_work), "--candidate", candidate, "--role", role, "--intervals", str(intervals), "--local-engineering", "--input-manifest", input_manifest]
        if role == "X":
            reference = results[f"R{intervals}"]
            argv.extend(("--reference-output", reference["output"]))
            if intervals == 2:
                argv.extend(("--reference-boundary", reference["boundary_path"], "--reference-stages", reference["stages"]))
        subprocess.run(argv, check=True, env=env)
        receipt = json.loads((case_work / "CASE_RECEIPT.json").read_text(encoding="utf-8"))
        assert receipt["execution_domain"] == "S46F_LOCAL_MICRO"
        assert receipt["route"]["nH"] == 2 * intervals and receipt["route"]["fresh_responses"] == 4 * intervals and receipt["route"]["math_dispatches"] == 16 * intervals
        expected_requests = 1536 if intervals == 2 else 6144
        assert receipt["cache"]["stats"]["factor_requests"] == expected_requests
        assert receipt["pointwise_pool"]["pool"]["closed"] is True
        assert receipt["pointwise_pool"]["pool"]["active_thread_ids"] == []
        assert receipt["pointwise_pool"]["pool"]["pending_scalar_records"] == 0
        if role == "X":
            assert receipt["comparisons"]["terminal"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
            if intervals == 2:
                assert receipt["comparisons"]["boundary"]["status"] == "PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT"
                assert receipt["comparisons"]["stages"]["status"] == "PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS"
        results[label] = receipt
