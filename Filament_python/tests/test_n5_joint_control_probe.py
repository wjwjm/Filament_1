"""Software checks for the task-specific external N5 control diagnostic."""
import importlib.util
import os
from pathlib import Path
import pytest

@pytest.fixture
def probe():
    path = os.environ.get("FILAMENT_JOINT_CONTROL_PROBE")
    if not path:
        pytest.skip("task-specific external control probe not selected")
    spec = importlib.util.spec_from_file_location("joint_control_probe", Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_joint_control_nested_initial_identity(probe, tmp_path):
    data = {"C/p0/state/current/screen_000000.npz": {
        "identity": {"size": 2535624, "st_nlink": 1}, "sha256": "a" * 64}}
    assert probe.initial_file_manifest(data, tmp_path) == {
        "C/p0/state/current/screen_000000.npz": 2535624}


def test_joint_control_holder_uses_actual_unlock_timestamp(probe, tmp_path):
    lock = tmp_path / ".storage_budget.lock"
    common = {"schema": probe.TRACE_SCHEMA, "lock_kind": "storage_ledger",
              "lock_path": str(lock), "pid": 123, "lock_span_id": "test",
              "operation_name": "StorageBudget.create_intent", "thread_name": "C/test"}
    events = [dict(common, event="acquired", monotonic_ns=100, wait_duration_ns=5),
              dict(common, event="released", monotonic_ns=220,
                   released_monotonic_ns=180, hold_duration_ns=80)]
    assert probe.holder_spans_at(events, lock, 150)
    assert not probe.holder_spans_at(events, lock, 200)
    span = probe.holder_spans(events, lock)[0]
    assert span["wait_ns"] == 5 and span["hold_ns"] == 80
