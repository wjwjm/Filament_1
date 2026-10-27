"""Low-cost unit checks for the B-owned scalar recorder."""

from __future__ import annotations

import pytest

from .timing import (
    DEFAULT_MAX_RECORDS,
    NEW_A8_LABELS,
    NEW_B8_LABELS,
    OLD_SCOPE_LABELS,
    TimingError,
    TimingRecorder,
)
from .adapter import make_recorder_pair


def test_label_contract_and_common_off_shells() -> None:
    assert len(OLD_SCOPE_LABELS) == 24
    assert len(NEW_A8_LABELS) == 8
    assert len(NEW_B8_LABELS) == 8
    recorder = TimingRecorder(enabled=False)
    with recorder.scope("APPLICATION"):
        pass
    with recorder.scope("INTERVAL_BODY", interval=0, occurrence_class="PER_INTERVAL"):
        pass
    with recorder.scope("EQ10_FFT"):
        pass
    recorder.marker("ZERO_INTENSITY", interval_index=0)
    snapshot = recorder.snapshot()
    assert snapshot["profile"] is False
    assert snapshot["summary_by_label"]["APPLICATION"]["call_count"] == 1
    assert snapshot["summary_by_label"]["INTERVAL_BODY"]["call_count"] == 1
    assert "EQ10_FFT" not in snapshot["summary_by_label"]
    assert snapshot["marker_count"] == 0
    assert snapshot["events_dropped"] == 0


def test_nested_context_and_failure_are_scalar() -> None:
    recorder = TimingRecorder(enabled=True)
    try:
        with recorder.interval_scope("INTERVAL_BODY", 0):
            with recorder.interval_scope("EQ10_SUM", 0, occurrence_class="PER_INTERVAL"):
                raise ValueError("synthetic")
    except ValueError:
        pass
    row = recorder.snapshot()["records"]
    assert len(row) == 2
    failed = next(item for item in row if item["label"] == "EQ10_SUM")
    assert failed["status"] == "failed"
    assert failed["context"]["interval_index"] == 0
    assert failed["context"]["ordinal"] == 0
    assert failed["context"]["ordinal_1based"] == 1
    assert failed["context"]["attempt"] == 1
    assert recorder.snapshot()["clock_closure"]["pass"] is True


def test_capacity_fails_closed_without_drop() -> None:
    recorder = TimingRecorder(enabled=True, max_records=1)
    with recorder.scope("EQ10_SUM"):
        pass
    with pytest.raises(TimingError, match="CAPACITY"):
        with recorder.scope("EQ10_FFT"):
            pass
    snapshot = recorder.snapshot()
    assert snapshot["event_count"] == 1
    assert snapshot["events_dropped"] == 0
    assert snapshot["max_records"] <= DEFAULT_MAX_RECORDS


def test_pair_uses_one_global_scope_budget_and_scalar_bridge() -> None:
    pair = make_recorder_pair(True, max_records=8)
    assert pair.max_records == 8
    assert pair.s45a.max_scopes <= 4
    assert pair.b.max_records <= 8 - pair.s45a.max_scopes
    with pair.s45a.scope("APPLICATION", pair.s45a.context(3, "pre", "stage1")):
        with pair.b.interval_scope("INTERVAL_BODY", 0):
            for _ in range(2_000_000):
                pass
            bridged = pair.b.bridged_context(require_companion=True, source_site="TEST@1")
            assert bridged.interval == 0
            assert bridged.ordinal == 0
            assert bridged.ordinal_1based == 1
            assert bridged.phase == "pre"
            assert bridged.stage == "stage1"
            assert bridged.source_site == "TEST@1"
    snapshot = pair.snapshot()
    assert snapshot["global_max_records"] == 8
    assert snapshot["bridge"]["bound"] is True
    assert snapshot["merged"]["raw_rows_retained"] == 2
