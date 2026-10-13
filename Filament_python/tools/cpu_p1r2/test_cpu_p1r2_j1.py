from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from tools.cpu_p1r2 import cpu_p1r2_j1 as j1


def _tiny_run_kwargs(context):
    return {key: context[key] for key in ("source", "pre_fields", "grid", "beam",
        "prop", "ion", "heat", "raman", "schedule")}


def test_pre_view_preserves_nonzero_next_velocities_exactly():
    fields = {key: np.zeros((8, 3, 4), dtype=np.float64, order="C")
              for key in ("delta_n", "vx", "vy")}
    fields["vx"][:] = 1.25e-7
    fields["vy"][:] = -2.5e-7
    view = j1.MemoryPREView(fields, state_origin="NEXT", initial_pre0=False)
    for index in range(8):
        view.read_interval(index)
        view.update_interval(index, np.zeros((3, 4), dtype=np.float64))
    post = view.final_fields()
    assert view.complete
    assert post["vx"].dtype == np.float64 and post["vx"].flags.c_contiguous
    np.testing.assert_array_equal(post["vx"], fields["vx"])
    np.testing.assert_array_equal(post["vy"], fields["vy"])
    with pytest.raises(j1.J1Error, match="positive float64 zero"):
        j1.MemoryPREView(fields, state_origin="PRE0", initial_pre0=True)


def test_initial_pre0_hash_is_bound_to_actual_delta_n():
    fields = {key: np.zeros((8, 3, 4), dtype=np.float64, order="C")
              for key in ("delta_n", "vx", "vy")}
    view = j1.MemoryPREView(fields, state_origin="PRE0", initial_pre0=True)
    actual = view._field_hashes["delta_n"]
    assert j1._pre0_hash_gate(actual, actual, initial_pre0=True,
        execution_scope="J1_REAL_INPUT") == actual
    with pytest.raises(j1.J1Error, match="differs from sealed PRE0"):
        j1._pre0_hash_gate(actual, "0" * 64, initial_pre0=True,
            execution_scope="J1_REAL_INPUT")
    with pytest.raises(j1.J1Error, match="requires the sealed canonical hash"):
        j1._pre0_hash_gate(actual, None, initial_pre0=True,
            execution_scope="J1_REAL_INPUT")
    assert j1._pre0_hash_gate(actual, None, initial_pre0=True,
        execution_scope="SOFTWARE_SMOKE_ONLY") == actual


def test_effective_precision_strategy_is_reported_and_projection_refused():
    class Prop:
        linear_precision_strategy = "baseline_complex64"
    assert j1._effective_precision_strategy(Prop()) == "baseline_complex64"
    class ProjectionProp:
        linear_precision_strategy = "unitary_projection"
    with pytest.raises(j1.J1Error, match="does not admit.*unitary_projection"):
        j1._effective_precision_strategy(ProjectionProp())


def test_source_contract_rejects_dtype_layout_and_hash_mismatch():
    good = np.ones((4, 3, 2), dtype=np.complex128, order="C")
    digest = j1._validate_source_array(good)
    assert j1._validate_source_array(good, expected_hash=digest) == digest
    with pytest.raises(j1.J1Error, match="C-contiguous complex128"):
        j1._validate_source_array(good.astype(np.complex64))
    with pytest.raises(j1.J1Error, match="C-contiguous complex128"):
        j1._validate_source_array(np.asfortranarray(good))
    with pytest.raises(j1.J1Error, match="canonical hash mismatch"):
        j1._validate_source_array(good, expected_hash="0" * 64)


def test_j1_requires_exactly_eight_intervals():
    class Schedule:
        def __init__(self, count):
            self.n_intervals = count
    assert j1._require_eight_schedule(Schedule(8)).n_intervals == 8
    with pytest.raises(j1.J1Error, match="exactly eight"):
        j1._require_eight_schedule(Schedule(1))


def test_frozen_interval_source_binding_is_checked_before_observation():
    repo_root = Path(__file__).resolve().parents[3]
    binding = j1.p1r._source_binding(repo_root)
    assert binding["interval_start_line"] == 628
    assert binding["accepted_boundary_line"] == 1503
    lines = (repo_root / "Filament_python/KHz_filament/propagate.py").read_text(encoding="utf-8").splitlines()
    assert lines[1502].strip() == 'pe = int(getattr(p, "progress_every_z", 0) or 0)'
    assert any(line.strip() == "z += dz_try" for line in lines[:1502])


def test_private_profile_namespace_does_not_mutate_science_module_xp():
    import KHz_filament.linear as linear
    import KHz_filament.raman as raman
    from KHz_filament.device import xp
    before_linear, before_raman = linear.xp, raman.xp
    function, collector = j1._profiled_propagator()
    assert callable(function)
    assert linear.xp is before_linear is xp
    assert raman.xp is before_raman is xp
    assert collector.report()["global_numpy_fft_modified"] is False
    assert collector.report()["global_linear_raman_xp_modified"] is False


def _stage_paths(tmp_path, stage):
    return {key + "_path": str(tmp_path / stage / filename) for key, filename in
        (("final_E", "final_E.npy"), ("final_POST", "final_POST.npz"),
         ("diagnostics", "diagnostics.npz"))}


def test_tiny_frozen_rotational_window_observer_profile_are_exact(tmp_path):
    if sys.version_info < (3, 12) or not hasattr(sys, "monitoring"):
        pytest.skip("sys.monitoring requires Python 3.12+")
    context = j1._tiny_context()
    results = {}
    for stage, profile, observe in (("unobserved", False, False),
                                    ("observed", False, True),
                                    ("profile", True, True)):
        results[stage] = j1._run_window(**_tiny_run_kwargs(context),
            output_dir=tmp_path / stage, source_label="synthetic_tiny_PRE0",
            initial_pre0=True, profile=profile, stage_name=stage,
            execution_scope="SOFTWARE_SMOKE_ONLY", observe=observe)
        assert results[stage]["PRE_complete"] is True
        assert results[stage]["observer_enabled"] is observe
        assert len(results[stage]["interval_rows"]) == (8 if observe else 0)
        if observe:
            assert all(row["interval_index"] == index
                       for index, row in enumerate(results[stage]["interval_rows"]))
    for reference, candidate in (("unobserved", "observed"), ("observed", "profile")):
        compared = j1._compare_stage_files(_stage_paths(tmp_path, reference),
            _stage_paths(tmp_path, candidate), f"{reference}_vs_{candidate}")
        assert compared["canonical_exact_match"]
    records = results["profile"]["profile"]["records"]
    names = {record["function"] for record in records}
    assert any(name.endswith(".isaacs_raman_stage") for name in names)
    assert not any(name.endswith(".isaacs_complete_eq27_stage") for name in names)
    assert any(name.endswith(".raman_convolve_intensity") for name in names)
    assert any(name.endswith(".evolve_rho_time") for name in names)
    ledger_rows = [record for record in records
        if record["function"].endswith(".build_unified_deposition_ledger")]
    assert len(ledger_rows) == 1 and ledger_rows[0]["calls"] == 1
    thermal_rows = [record for record in records
        if record["function"].endswith(".thermalize_interval")]
    acoustic_rows = [record for record in records
        if record["function"].endswith(".map_post_acoustic_increment")]
    assert len(thermal_rows) == 1 and thermal_rows[0]["calls"] == 8
    assert len(acoustic_rows) == 1 and acoustic_rows[0]["calls"] == 8
    raman_iir = [record for record in records
        if record["function"].endswith(".raman_convolve_intensity")
        and record.get("scalar_kwargs", {}).get("method") == "iir"]
    assert raman_iir and all(row["scalar_kwargs"].get("iir_sampling") == "exact_piecewise_linear"
                             for row in raman_iir)
    assert results["profile"]["profile"]["fft_call_count"] > 0
    with np.load(tmp_path / "observed" / "final_POST.npz", allow_pickle=False) as post:
        assert set(post.files) == {"delta_n", "vx", "vy"}
        assert all(post[name].shape[0] == 8 for name in post.files)


def test_tiny_optional_complete_eq27_branch_is_exercised(tmp_path):
    if sys.version_info < (3, 12) or not hasattr(sys, "monitoring"):
        pytest.skip("sys.monitoring requires Python 3.12+")
    context = j1._tiny_context(operator_mode="full_isaacs_eq27_complete")
    result = j1._run_window(**_tiny_run_kwargs(context),
        output_dir=tmp_path / "complete", source_label="synthetic_tiny_PRE0",
        initial_pre0=True, profile=True, observe=True, stage_name="complete",
        execution_scope="SOFTWARE_SMOKE_ONLY")
    names = {record["function"] for record in result["profile"]["records"]}
    assert any(name.endswith(".isaacs_complete_eq27_stage") for name in names)
    assert len(result["interval_rows"]) == 8


def test_rss_warn_latches_at_eighty_and_abort_is_ninety():
    total = 320 * 1024**3
    assert j1._rss_action(int(total * 0.80), total)["warn_80"]
    assert not j1._rss_action(int(total * 0.80), total)["abort_90"]
    assert j1._rss_action(int(total * 0.90), total)["abort_90"]
    assert j1._stage_block_reason(True, [{"status": "DONE"}]) == "BLOCKED_RSS_80_PERCENT_LATCH"
    assert j1._stage_block_reason(False, [{"status": "DONE"}],
        baseline_repeat={"canonical_exact_match": False}) == "BLOCKED_BASELINE_REPEAT_DIFFERENCE"
    assert j1._stage_block_reason(False, [{"status": "FAIL_CHILD"}]) == "BLOCKED_AFTER_PRIOR_STAGE_FAILURE"


def test_rss_ninety_percent_terminates_current_child(tmp_path, monkeypatch):
    output = tmp_path / "rss-stage"
    output.mkdir()
    total = 320 * 1024**3
    calls = {"n": 0}
    def snapshot(_pid):
        calls["n"] += 1
        return {"rss": int(total * (0.91 if calls["n"] > 1 else 0.1)), "cpu": 0.0}
    monkeypatch.setattr(j1.common, "tree_snapshot", snapshot)
    result = j1._spawn_and_monitor([sys.executable, "-c", "import time; time.sleep(30)"],
        stage_dir=output, cwd=tmp_path, env=os.environ.copy(),
        timeout_seconds=15.0, memory_bytes=total, poll_seconds=0.02)
    assert result["stop"]["reason"] == "RSS_90_PERCENT"
    assert result["exit_code"] is not None


def test_timeout_preserves_file_backed_partial_output_and_reaps_child(tmp_path):
    output = tmp_path / "timeout-stage"
    output.mkdir()
    code = "import time; print('partial-accepted-interval', flush=True); time.sleep(30)"
    result = j1._spawn_and_monitor([sys.executable, "-c", code], stage_dir=output,
        cwd=tmp_path, env=os.environ.copy(), timeout_seconds=0.3,
        memory_bytes=None, poll_seconds=0.05)
    assert result["timed_out"] is True
    assert result["exit_code"] is not None
    assert "partial-accepted-interval" in (output / "stdout.log").read_text(encoding="utf-8")
    assert json.loads((output / "supervisor_receipt.json").read_text(encoding="utf-8"))["stop"]["reason"] == "STAGE_TIMEOUT"



def test_time_budget_does_not_claim_memory_pressure():
    assert j1._stage_block_reason(False, [{"status": "SKIPPED_INSUFFICIENT_TIME"}]) == "SKIPPED_INSUFFICIENT_TIME"


def test_probe_queries_distinguish_failure_presence_and_absence():
    from types import SimpleNamespace
    from tools.cpu_p1r2.allocation_probe import query_steps
    def runner(code, output):
        return lambda *args, **kwargs: SimpleNamespace(returncode=code, stdout=output, stderr="err" if code else "")
    assert query_steps("123", "P1R2-probe", None, runner(1, ""))["state"] == "QUERY_FAILED"
    present = query_steps("123", "P1R2-probe", None, runner(0, "123.0|P1R2-probe\n123.batch|batch\n124.0|P1R2-probe\n"))
    assert present["state"] == "STEP_PRESENT" and present["step_ids"] == ["123.0"]
    assert query_steps("123", "P1R2-probe", "123.0", runner(0, "123.batch|batch\n"))["state"] == "STEP_ABSENT"


def test_unresolved_termination_never_claims_stopped(tmp_path, monkeypatch):
    from types import SimpleNamespace
    args = SimpleNamespace(output_root=tmp_path, bench_root=tmp_path, input_root=tmp_path, vendor_dir=tmp_path)
    monkeypatch.setattr(j1, "_spawn_and_monitor", lambda *a, **kw: {
        "exit_code": None, "stop": {"status": "STOP_UNRESOLVED"}, "timed_out": True})
    result = j1._supervise_stage(args, "baseline-a", timeout_seconds=1, memory_bytes=100)
    assert result["status"] == "STOP_UNRESOLVED"
