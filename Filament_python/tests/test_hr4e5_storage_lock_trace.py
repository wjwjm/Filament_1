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


def test_capacity_uses_one_fresh_traversal(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    budget = storage.StorageBudget(root, campaign_id="one-walk", require_quota=False)
    for directory, size in [("C", 31), ("R", 17)]:
        (root / directory).mkdir()
        (root / directory / "a").write_bytes(b"a" * size)
    original = storage._iter_regular_files
    walks = []

    def counted(path, **kwargs):
        walks.append(Path(path))
        yield from original(path, **kwargs)

    monkeypatch.setattr(storage, "_iter_regular_files", counted)
    reservation = budget.reserve(100, purpose="C", allocation_paths=["C"])
    assert walks == [root]
    assert budget._read()["reservations"][reservation.reservation_id]["allocation_start_bytes"] == 31
    (root / "C" / "new").write_bytes(b"c" * 23)
    walks.clear()
    report = budget.check_capacity()
    assert walks == [root]
    assert report["reserved_bytes"] == 77
    assert report["actual_bytes"] == sum(p.stat().st_size for p in original(root))
    (root / "C" / "new").write_bytes(b"c" * 40)
    walks.clear()
    fresh = budget.budget_report()
    assert walks == [root]
    assert fresh["reserved_bytes"] == 60


def test_accounting_scope_prefix_and_external_bytes_remain_strict(tmp_path):
    root = tmp_path / "campaign"
    budget = storage.StorageBudget(root, require_quota=False, external_prefix_bytes=53)
    for directory, size in [("C", 10), ("C_other", 91), ("R", 20)]:
        (root / directory).mkdir()
        (root / directory / "a").write_bytes(b"x" * size)
    reservation = budget.reserve(100, purpose="C", allocation_paths=["C", "R/a"])
    assert budget._read()["reservations"][reservation.reservation_id]["allocation_start_bytes"] == 30
    report = budget.check_capacity()
    assert report["external_prefix_bytes"] == 53
    assert report["reserved_bytes"] == 100
    assert report["projected_bytes"] == report["actual_bytes"] + 53 + 100 + report["safety_margin_bytes"]


def test_overlapping_active_scope_rejects_before_scan(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    budget = storage.StorageBudget(root, require_quota=False)
    (root / "C").mkdir()
    budget.reserve(100, purpose="first", allocation_paths=["C"])

    def forbidden(*args, **kwargs):
        pytest.fail("overlap must reject before filesystem traversal")

    monkeypatch.setattr(budget, "_capacity_accounting", forbidden)
    with pytest.raises(storage.StorageBudgetError, match="overlapping allocation"):
        budget.reserve(100, purpose="second", allocation_paths=["C/child"])


@pytest.mark.parametrize("value", [True, 0, -1, 180.01, float("nan"), float("inf")])
def test_invalid_campaign_lock_wait_fails_before_root_creation(tmp_path, value):
    root = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="storage lock timeout"):
        storage.StorageBudget(root, lock_timeout_s=value)
    assert not root.exists()


def test_storage_wait_is_explicit_instance_bound_and_default_unchanged(tmp_path, monkeypatch):
    factory = storage._FileLock
    values = []

    def observed(path, timeout_s=10.0, **kwargs):
        values.append((Path(path), timeout_s))
        return factory(path, timeout_s=timeout_s, **kwargs)

    monkeypatch.setattr(storage, "_FileLock", observed)
    custom = storage.StorageBudget(tmp_path / "custom", lock_timeout_s=0.05, require_quota=False)
    custom.intents()
    custom.check_capacity()
    custom.budget_report()
    assert values and all(timeout == 0.05 for _, timeout in values)
    values.clear()
    ordinary = storage.StorageBudget(tmp_path / "ordinary", require_quota=False)
    ordinary.intents()
    assert all(timeout == 10.0 for _, timeout in values)
    assert factory(tmp_path / "campaign_state.lock").timeout_s == 10.0


def test_intent_content_snapshot_still_excludes_other_lock_owners(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    budget = storage.StorageBudget(root, require_quota=False)
    (root / "C").mkdir()
    (root / "C" / "state.bin").write_bytes(b"actual content")
    reservation = budget.reserve(100, purpose="intent", allocation_paths=["C"])
    snapshot = budget._snapshot_intent_files

    def checked(paths):
        with pytest.raises(storage.StorageBudgetError, match="timed out"):
            with storage._FileLock(budget.lock_path, timeout_s=0.03):
                pytest.fail("content snapshot ran outside ledger lock")
        return snapshot(paths)

    monkeypatch.setattr(budget, "_snapshot_intent_files", checked)
    intent = budget.create_intent(reservation_id=reservation.reservation_id, trajectory="C", pulse=0,
                                 attempt=0, role="OPTICAL", allowed_paths=["C"], expected_bytes=100)
    row = intent["initial_files"]["C/state.bin"]
    assert row["sha256"] == storage._sha256_file(root / "C" / "state.bin")
    assert row["identity"]["size"] == len(b"actual content")



def test_reconstructed_worker_view_preserves_default_and_explicit_wait(tmp_path):
    identity = {"identity_sha256": "a" * 64, "budget_identity": {}}
    root = tmp_path / "campaign"
    storage.StorageBudget(root, campaign_id="worker-policy", require_quota=False,
                          require_intents=True, admission_hash="a" * 64, admission_identity=identity)
    default = storage.StorageBudget.open_for_worker(root, campaign_id="worker-policy", admission_identity=identity)
    custom = storage.StorageBudget.open_for_worker(root, campaign_id="worker-policy", admission_identity=identity,
                                                  lock_timeout_s=0.04)
    assert default.lock_timeout_s == 10.0
    assert custom.lock_timeout_s == 0.04
    assert custom.writer_grants() == default.writer_grants() == {}
    with pytest.raises(ValueError, match="storage lock timeout"):
        storage.StorageBudget.open_for_worker(root, campaign_id="worker-policy", admission_identity=identity,
                                              lock_timeout_s=float("inf"))


def test_instance_wait_enforces_real_mutex_timeout_and_releases(tmp_path):
    root = tmp_path / "campaign"
    budget = storage.StorageBudget(root, require_quota=False, lock_timeout_s=0.04)
    with storage._FileLock(budget.lock_path):
        with pytest.raises(storage.StorageBudgetError, match="timed out"):
            budget.intents()
    assert budget.intents() == {}
