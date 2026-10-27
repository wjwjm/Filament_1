"""Focused contract tests for the CPU-S4-5A scalar timing recorder."""

from __future__ import annotations

import threading

import pytest

from tools.cpu_s4_5a_l1_residual import timing


def _clock_pair(monkeypatch, wall, cpu):
    wall_values = iter(wall)
    cpu_values = iter(cpu)
    monkeypatch.setattr(timing, "perf_counter_ns", lambda: next(wall_values))
    monkeypatch.setattr(timing, "process_time_ns", lambda: next(cpu_values))


def test_nested_direct_child_closure_and_scalar_metadata(monkeypatch):
    _clock_pair(monkeypatch, [100, 110, 120, 150], [10, 12, 14, 20])
    recorder = timing.TimingRecorder()
    token = recorder.begin("APPLICATION", context=timing.context(3, "pre", "stage1"))
    child = recorder.begin("RAMAN_STAGE", context=timing.context(3, "pre", "stage1"))
    child_record = recorder.end(child)
    root_record = recorder.end(token)
    assert child_record["wall_inclusive_ns"] == 10
    assert root_record["wall_inclusive_ns"] == 50
    assert root_record["wall_exclusive_ns"] == 40
    assert root_record["cpu_exclusive_ns"] == 8
    snapshot = recorder.snapshot()
    assert snapshot["events_dropped"] == 0
    assert snapshot["clock_closure"]["pass"] is True
    assert snapshot["records"][0]["context"] == {"interval": 3, "phase": "pre", "stage": "stage1"}
    assert snapshot["summary_by_label"]["RAMAN_STAGE"]["call_count"] == 1


def test_exception_scope_is_failed_without_retaining_exception():
    recorder = timing.TimingRecorder()
    with pytest.raises(ValueError):
        with recorder.scope("RAMAN_IIR", interval=0, phase="post", stage="stage2"):
            raise ValueError("transient")
    record = recorder.snapshot()["records"][0]
    assert record["failed"] is True
    assert record["status"] == "failed"


def test_mismatch_and_repeated_exit_are_explicit(monkeypatch):
    _clock_pair(monkeypatch, [1, 2, 3, 4], [1, 2, 3, 4])
    recorder = timing.TimingRecorder()
    first = recorder.begin("APPLICATION")
    second = recorder.begin("INTERVAL")
    with pytest.raises(timing.TimingError, match="EXIT_MISMATCH"):
        recorder.end(first)
    recorder.end(second)
    recorder.end(first)
    with pytest.raises(timing.TimingError, match="ALREADY_ENDED"):
        recorder.end(first)


def test_capacity_refuses_without_dropping(monkeypatch):
    _clock_pair(monkeypatch, [1, 2], [1, 2])
    recorder = timing.TimingRecorder(max_scopes=1)
    token = recorder.begin("MANAGEMENT")
    with pytest.raises(timing.TimingError, match="CAPACITY_EXCEEDED"):
        recorder.begin("MANAGEMENT")
    recorder.end(token)
    assert recorder.snapshot()["events_dropped"] == 0


def test_disabled_recorder_reads_no_clocks(monkeypatch):
    def forbidden_clock():
        raise AssertionError("disabled recorder read a clock")

    monkeypatch.setattr(timing, "perf_counter_ns", forbidden_clock)
    monkeypatch.setattr(timing, "process_time_ns", forbidden_clock)
    recorder = timing.TimingRecorder(enabled=False)
    token = recorder.begin("APPLICATION")
    assert token == 0
    assert recorder.end(token) is None
    assert recorder.snapshot()["records"] == []


def test_owner_thread_and_context_reentry():
    recorder = timing.TimingRecorder()
    errors = []

    def foreign_call():
        try:
            recorder.begin("MANAGEMENT")
        except Exception as exc:  # noqa: BLE001 - contract assertion
            errors.append(exc)

    thread = threading.Thread(target=foreign_call)
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert isinstance(errors[0], timing.TimingError)

    scope = recorder.scope("MANAGEMENT")
    with scope:
        pass
    with pytest.raises(timing.TimingError, match="EXIT_WITHOUT_ENTER"):
        scope.__exit__(None, None, None)


def test_empty_control_is_marked_non_subtractive(monkeypatch):
    _clock_pair(monkeypatch, [10, 11, 12, 13], [20, 21, 23, 25])
    value = timing.empty_scope_control_value()
    assert value["wall_ns"] == 3
    assert value["cpu_ns"] == 5
    assert value["empty_scope_wall_ns"] == 1
    assert value["subtract_from_scopes"] is False
    assert timing.empty_scope_control_value(enabled=False)["wall_ns"] == 0


def test_scalar_recorder_does_not_keep_returned_values_alive():
    import gc
    import weakref
    class Payload:pass
    recorder=timing.TimingRecorder()
    value=Payload();reference=weakref.ref(value)
    with recorder.scope("RAMAN_STAGE"):
        result=value
    del result,value
    gc.collect()
    assert reference() is None
    assert recorder.snapshot()["completed_scopes"]==1


def test_clock_failure_has_bounded_incomplete_retirement(monkeypatch):
    _clock_pair(monkeypatch,[100,90],[100,110])
    recorder=timing.TimingRecorder();token=recorder.begin("APPLICATION")
    with pytest.raises(timing.TimingError,match="CLOCK_WENT_BACKWARDS"):recorder.end(token)
    recorder.retire_incomplete("CLOCK_WENT_BACKWARDS")
    snapshot=recorder.snapshot()
    assert snapshot["completed_scopes"]==0 and len(snapshot["incomplete_scopes"])==1
    assert snapshot["incomplete_scopes"][0]["status"]=="incomplete"

