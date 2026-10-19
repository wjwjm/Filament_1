from __future__ import annotations

import json
import pytest

from tools.cpu_o1.cpu_s4.profile import (
    NestedTimer,
    run_validated_context_pair,
)


def test_nested_timer_accounts_nested_wall_cpu_and_json_safely():
    timer = NestedTimer()
    inner = timer.wrap("inner", lambda: sum(range(20)))

    def outer_body():
        inner()
        return 7

    outer = timer.wrap("outer", outer_body)
    assert outer() == 7
    receipt = timer.report()
    assert receipt["calls"] == 2
    assert receipt["failures"] == 0
    rows = {row["label"]: row for row in receipt["records"]}
    assert rows["outer"]["inclusive_wall_seconds"] >= rows["outer"]["exclusive_wall_seconds"]
    assert rows["outer"]["inclusive_process_cpu_seconds"] >= rows["outer"]["exclusive_process_cpu_seconds"]
    assert rows["inner"]["calls"] == 1
    json.dumps(receipt, allow_nan=False)


def test_nested_timer_zero_calls_and_exception_are_recorded():
    empty = NestedTimer().report()
    assert empty["calls"] == 0 and empty["records"] == []

    timer = NestedTimer()

    def fail():
        raise LookupError("expected timer fixture failure")

    with pytest.raises(LookupError, match="expected timer fixture failure"):
        timer.wrap("failing", fail)()
    row = timer.report()["records"][0]
    assert row["calls"] == 1 and row["failures"] == 1


def test_profiled_i_factory_changes_only_private_clone(monkeypatch):
    import KHz_filament.linear as linear
    import KHz_filament.raman as raman
    import importlib
    propagate = importlib.import_module("KHz_filament.propagate")
    import numpy as np
    from tools.cpu_o1.cpu_s4.profile import make_profiled_i_factory

    original = (linear.xp, raman.xp, propagate.xp, np.fft)
    factory = make_profiled_i_factory(enabled=True)
    private, dispatch = factory(propagate.propagate_one_pulse)
    assert getattr(private, "__cpu_s4_private_profile__", False)
    assert factory.last_propagator is not private
    assert factory.last_dispatch is dispatch.dispatch
    assert (linear.xp, raman.xp, propagate.xp, np.fft) == original
    assert factory.last_propagator._cpu_o1_execution["identity"] == "I"
    assert factory.last_propagator._cpu_o1_execution["temporal_workers"] == 48
    assert factory.last_propagator._cpu_o1_execution["spatial_workers"] == 24
    assert factory.last_propagator._cpu_o1_execution["kernel_threads"] == 24


def test_tiny_off_profile_runs_frozen_i_and_exact_scientific_gate(tmp_path):
    from tools.cpu_p1r2 import cpu_p1r2_j1 as j1

    context = j1._tiny_context(operator_mode="full_isaacs_eq27")
    context["source_hash"] = j1._canonical_hash(context["source"])
    context["assets"] = {"pre0": context["pre0"],
        "pre0_canonical_sha256": j1._canonical_hash(context["pre0"])}
    source_before = j1._canonical_hash(context["source"])
    pre_before = {key: j1._canonical_hash(value)
                  for key, value in context["pre_fields"].items()}

    result = run_validated_context_pair(context, tmp_path / "s4-tiny")

    assert result["status"] == "LOCAL_PASS"
    summary = json.loads((tmp_path / "s4-tiny" / "S4_PROFILE_SUMMARY.json").read_text())
    assert summary["production_qualification"] == "NOT_PRODUCTION_QUALIFIED"
    assert summary["hpc_submission"] == "NOT_SUBMITTED"
    assert summary["comparison_gate"]["status"] == "EXACT"
    assert summary["comparison_gate"]["raw_row_count"] > 0
    assert summary["modes"]["off"]["candidate_event_count"] > 0
    assert summary["modes"]["profile"]["candidate_event_count"] > 0
    assert len(summary["modes"]["off"]["interval_rows"]) == 8
    assert len(summary["modes"]["profile"]["interval_rows"]) == 8
    assert all(r["profiled_call_count"] > 0 for r in summary["modes"]["profile"]["interval_rows"])
    profile = json.loads((tmp_path / "s4-tiny" / "profile" / "propagation" /
                          "function_fft_profile.json").read_text())
    assert profile["mode"] == "PROFILE"
    assert profile["nested_timers"]["calls"] > 0
    assert sum(r["calls"] for r in profile["nested_timers"]["records"]
               if r["label"] == "ionization.full_time_rk4_candidate") == summary["modes"]["profile"]["candidate_event_count"]
    assert profile["global_science_modules_modified"] is False
    assert profile["full_arrays_retained"] is False
    assert j1._canonical_hash(context["source"]) == source_before
    assert {key: j1._canonical_hash(value) for key, value in context["pre_fields"].items()} == pre_before


def test_nested_exact_clock_accounting_and_interrupt(monkeypatch):
    import tools.cpu_o1.cpu_s4.profile as module
    wall = iter([0., 1., 3., 5.])
    cpu = iter([0., 2., 6., 10.])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(wall))
    monkeypatch.setattr(module.time, "process_time", lambda: next(cpu))
    timer = NestedTimer()
    inner = timer.wrap("inner", lambda: None)
    timer.wrap("propagate.propagate_one_pulse", lambda: inner())()
    r = timer.report()
    assert r["reconciliation"]["sum_all_profiled_exclusive_wall_seconds"] == 5
    assert r["reconciliation"]["sum_all_profiled_exclusive_process_cpu_seconds"] == 10
    assert r["reconciliation"]["wall_reconciliation_delta_seconds"] == 0
    assert r["nested_profile_calls_observed"] == 1
    assert r["cross_thread_overlapping_profile_calls_observed"] == 0
    monkeypatch.undo()
    def stop():
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        timer.wrap("interrupt", stop)()
    assert timer.report()["failures"] == 1
    assert not timer._stack()


@pytest.mark.parametrize("fault", ["missing", "mode", "identity", "interval", "events"])
def test_profile_receipt_fail_closed(fault):
    from tools.cpu_o1.cpu_s4.profile import _validate_profile_receipt, FIXED_EXECUTION
    row = dict(mode="PROFILE", implementation="I", execution=FIXED_EXECUTION.copy(),
        nested_timers={"records": [{"interval_index": i, "calls": 1, "label": "propagate.propagate_one_pulse"} for i in range(8)]},
        ionization_candidate_events=[{"kind": "ionization"}])
    assert _validate_profile_receipt(row) == list(range(8))
    if fault == "missing": row = None
    elif fault == "mode": row["mode"] = "OFF"
    elif fault == "identity": row["execution"]["kernel_threads"] = 1
    elif fault == "interval": row["nested_timers"]["records"].pop()
    else: row["ionization_candidate_events"] = []
    with pytest.raises(RuntimeError): _validate_profile_receipt(row)


def test_reject_existing_output_and_invalid_schedule(tmp_path):
    with pytest.raises(FileExistsError): run_validated_context_pair({}, tmp_path)
    with pytest.raises(ValueError, match="eight-interval"):
        run_validated_context_pair({}, tmp_path / "new")
    assert not (tmp_path / "new").exists()
