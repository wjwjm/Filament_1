"""Focused tests for opt-in storage lock diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from KHz_filament import hr4e5_storage as storage


def _reset_trace_state() -> None:
    storage._LOCK_TRACE_LOCAL.__dict__.clear()


def test_lock_trace_is_disabled_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_trace_state()
    monkeypatch.delenv(storage._LOCK_TRACE_DIR_ENV, raising=False)
    root = tmp_path / "campaign"
    budget = storage.StorageBudget(root, campaign_id="lock-trace-disabled")

    with budget._locked(operation_name="test.disabled"):
        pass

    status = storage._lock_trace_status()
    assert status["enabled"] is False
    assert status["write_count"] == 0
    assert status["error_count"] == 0
    assert not list(tmp_path.rglob("lock-*.jsonl"))


def test_lock_trace_records_lock_and_aggregated_intent_hash_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_trace_state()
    root = tmp_path / "campaign"
    trace_dir = tmp_path / "diagnostics"
    trace_dir.mkdir()
    monkeypatch.setenv(storage._LOCK_TRACE_DIR_ENV, str(trace_dir.resolve()))
    budget = storage.StorageBudget(root, campaign_id="lock-trace-enabled")

    files = root / "samples"
    files.mkdir()
    (files / "a.bin").write_bytes(b"abc")
    (files / "b.bin").write_bytes(b"12345")
    with budget._locked(operation_name="test.intent_snapshot"):
        snapshot = budget._snapshot_intent_files(["samples"])

    assert set(snapshot) == {"samples/a.bin", "samples/b.bin"}
    trace_files = list(trace_dir.glob("lock-*.jsonl"))
    assert len(trace_files) == 1
    records = [json.loads(line) for line in trace_files[0].read_text(encoding="utf-8").splitlines()]
    lock_events = [record for record in records if record["operation_name"] == "test.intent_snapshot"]
    assert {record["event"] for record in lock_events} >= {"requested", "acquired", "released"}
    for record in lock_events:
        assert record["schema"] == "khz_filament.hr4e5.storage_lock_trace.v1"
        assert record["lock_kind"] == "storage_ledger"
        assert record["lock_path"].endswith(".storage_budget.lock")
        assert record["host_identity"] and record["hostname"]
        assert record["pid"] > 0 and record["process_start_identity"]
        assert record["thread_id"] > 0 and record["lock_span_id"]
        assert record["wall_time_ns"] > 0 and record["monotonic_ns"] > 0
    acquired = next(record for record in lock_events if record["event"] == "acquired")
    released = next(record for record in lock_events if record["event"] == "released")
    assert acquired["wait_duration_ns"] >= 0
    assert released["hold_duration_ns"] >= 0

    snapshots = [record for record in records
                 if record.get("phase_name") == "intent_snapshot_and_hash"]
    assert len(snapshots) == 1
    metrics = snapshots[0]["phase_metrics"]
    assert metrics["sha256_file_count"] == 2
    assert metrics["sha256_bytes"] == 8
    assert metrics["sha256_duration_ns"] >= 0
    assert not any(record.get("phase_name") == "file_sha256" for record in records)
    status = storage._lock_trace_status()
    assert status["write_count"] == len(records)
    assert status["write_attempt_count"] == len(records)
    assert status["write_duration_ns"] >= status["last_append_duration_ns"] >= 0
    assert status["error_count"] == 0


def test_invalid_trace_directory_does_not_change_lock_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_trace_state()
    root = tmp_path / "campaign"
    root.mkdir()
    monkeypatch.setenv(storage._LOCK_TRACE_DIR_ENV, str(root.resolve()))
    budget = storage.StorageBudget(root, campaign_id="lock-trace-invalid-dir")

    with budget._locked(operation_name="test.invalid_trace_dir"):
        pass

    assert storage._lock_trace_status()["error_count"] > 0
    assert not list(root.glob("lock-*.jsonl"))


def test_trace_append_error_does_not_change_lock_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_trace_state()
    trace_dir = tmp_path / "diagnostics"
    trace_dir.mkdir()
    monkeypatch.setenv(storage._LOCK_TRACE_DIR_ENV, str(trace_dir.resolve()))
    lock_path = tmp_path / "campaign" / ".storage_budget.lock"
    real_open = os.open

    def fail_trace_open(path, flags, mode=0o777, *, dir_fd=None):
        if Path(path).name.startswith("lock-"):
            raise OSError("injected trace append failure")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(storage.os, "open", fail_trace_open)
    with storage._FileLock(lock_path, operation_name="test.trace_io_failure"):
        assert lock_path.exists()
    assert storage._lock_trace_status()["error_count"] > 0


def test_body_exception_still_releases_lock_and_records_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_trace_state()
    trace_dir = tmp_path / "diagnostics"
    trace_dir.mkdir()
    monkeypatch.setenv(storage._LOCK_TRACE_DIR_ENV, str(trace_dir.resolve()))
    lock_path = tmp_path / "campaign" / ".storage_budget.lock"

    with pytest.raises(RuntimeError, match="body failed"):
        with storage._FileLock(lock_path, operation_name="test.body_exception"):
            raise RuntimeError("body failed")
    with storage._FileLock(lock_path, operation_name="test.reacquire"):
        pass

    records = [json.loads(line) for path in trace_dir.glob("lock-*.jsonl")
               for line in path.read_text(encoding="utf-8").splitlines()]
    body_events = [record["event"] for record in records
                   if record["operation_name"] == "test.body_exception"]
    assert "acquired" in body_events and "released" in body_events


def test_lock_timeout_default_remains_ten_seconds() -> None:
    assert storage._FileLock.__init__.__defaults__ == (10.0,)


def test_interrupt_during_acquired_trace_releases_lock(tmp_path, monkeypatch):
    _reset_trace_state()
    monkeypatch.setenv(storage._LOCK_TRACE_DIR_ENV, str(tmp_path / "trace"))
    lock_path = tmp_path / "campaign" / ".storage_budget.lock"
    emit = storage._emit_lock_trace

    def interrupt_acquired(event, context, **fields):
        if event == "acquired":
            raise KeyboardInterrupt("diagnostic interrupted")
        return emit(event, context, **fields)

    lock = storage._FileLock(lock_path)
    with monkeypatch.context() as scoped:
        scoped.setattr(storage, "_emit_lock_trace", interrupt_acquired)
        with pytest.raises(KeyboardInterrupt, match="diagnostic interrupted"):
            with lock:
                pytest.fail("interrupted acquisition entered its body")
    assert lock._handle is None
    assert getattr(storage._LOCK_TRACE_LOCAL, "lock_context", None) is None
    with storage._FileLock(lock_path, timeout_s=0.1):
        pass