from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from .records import (
    BudgetError,
    ClockError,
    RecordError,
    TimingGateError,
    budget_check,
    load_json_strict,
    timing_gate,
    resolve_record_path,
    validate_case_order,
    validate_clock_records,
    validate_source_records,
)


def _clock_rows():
    return [
        {"scope": "window", "start_ns": 0, "end_ns": 100, "completed": True},
        {"scope": "interval1", "start_ns": 0, "end_ns": 40, "completed": True, "parent": "window"},
        {"scope": "later", "start_ns": 40, "end_ns": 100, "completed": True, "parent": "window"},
    ]


def test_strict_json_rejects_duplicate_and_nonfinite(tmp_path: Path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(RecordError):
        load_json_strict(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a": NaN}', encoding="utf-8")
    with pytest.raises(RecordError):
        load_json_strict(nonfinite)
    overflow = tmp_path / "overflow.json"
    overflow.write_text('{"a": 1e999}', encoding="utf-8")
    with pytest.raises(RecordError):
        load_json_strict(overflow)


def test_source_records_hash_and_basename_guards(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"source")
    digest = hashlib.sha256(b"source").hexdigest()
    result = validate_source_records(tmp_path, [{"path": "source.txt", "hash_scope": "raw_bytes", "raw_sha256": digest}])
    assert result["status"] == "PASS_SOURCE_RECORDS"
    with pytest.raises(RecordError):
        validate_source_records(tmp_path, [{"path": "source.txt", "hash_scope": "raw_bytes", "raw_sha256": "0" * 64}])
    with pytest.raises(RecordError):
        validate_source_records(tmp_path, [{"path": "../source.txt", "hash_scope": "raw_bytes", "raw_sha256": digest}])
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "source.txt").write_bytes(b"source")
    with pytest.raises(RecordError, match="BASENAME_AMBIGUOUS"):
        validate_source_records(tmp_path, [{"path": "source.txt", "hash_scope": "raw_bytes", "raw_sha256": digest}])
    (tmp_path / "top").mkdir()
    (tmp_path / "top" / "source.txt").write_bytes(b"source")
    qualified = validate_source_records(tmp_path, [
        {"path": "top/source.txt", "hash_scope": "raw_bytes", "raw_sha256": digest},
        {"path": "nested/source.txt", "hash_scope": "raw_bytes", "raw_sha256": digest},
    ])
    assert len(qualified["records"]) == 2
    link_dir = tmp_path / "linkdir"
    link_dir.mkdir()
    link = link_dir / "source.txt"
    try:
        link.symlink_to(source)
    except OSError:
        link = None
    if link is not None:
        with pytest.raises(RecordError, match="SYMLINK"):
            resolve_record_path(tmp_path, "linkdir/source.txt", expected_raw_sha256=digest)


def test_clock_disjoint_and_negative_guards():
    result = validate_clock_records(_clock_rows())
    assert result["durations_ns"] == {"window": 100, "interval1": 40, "later": 60}
    bad = _clock_rows()
    bad[2] = dict(bad[2], start_ns=20)
    with pytest.raises(ClockError):
        validate_clock_records(bad)
    bad = _clock_rows()
    bad[1] = dict(bad[1], end_ns=-1)
    with pytest.raises(ClockError):
        validate_clock_records(bad)


def test_case_order_and_timing_gate():
    rows = [{"case": name} for name in ("QUAL_OFF", "QUAL_LIGHT", "P1_OFF", "P1_LIGHT", "P2_LIGHT", "P2_OFF", "P3_OFF", "P3_LIGHT")]
    assert validate_case_order(rows)["status"] == "PASS_CASE_ORDER"
    good = []
    for pair in ("P1", "P2", "P3"):
        good.extend([
            {"pair": pair, "kind": "OFF", "application_s": 10, "window_s": 8, "later_s": 6},
            {"pair": pair, "kind": "LIGHT", "application_s": 10.1, "window_s": 8.1, "later_s": 6.1},
        ])
    assert timing_gate(good)["status"] == "PASS_TIMING_GATE"
    failed = [dict(row) for row in good]
    failed[1]["application_s"] = 12.0
    failure = timing_gate(failed)
    assert failure["status"] == "TIMING_GATE_FAIL"
    assert failure["metrics"]["application"]["ratios"][0] == pytest.approx(1.2)
    with pytest.raises(TimingGateError, match="REQUIRED_PAIRS"):
        timing_gate(good[:2])
    with pytest.raises(TimingGateError):
        timing_gate([{**good[0], "application_s": float("nan")}, good[1]])


def test_budget_ignores_batch_and_enforces_total_and_inflight():
    completed = [dict(job_id=str(i), category="repair", state="completed",
                      walltime_s=1200, actual_walltime_s=1) for i in range(3)]
    fourth = dict(job_id="fourth", category="large", state="pending", walltime_s=5400)
    assert budget_check(completed, "CPU", new_request=fourth)["status"] == "PASS_BUDGET"
    fifth = dict(job_id="fifth", category="repair", state="pending", walltime_s=1200)
    assert "NEW_MAX_JOB_IDS" in budget_check(
        completed + [dict(fourth, state="completed", actual_walltime_s=1)],
        "CPU", new_request=fifth)["reasons"]
    gpu_prior = [dict(job_id="290147", category="main", state="failed", walltime_s=5400,
                      actual_walltime_s=47),
                 dict(job_id="290163", category="repair", state="failed", walltime_s=1800,
                      actual_walltime_s=206)]
    gpu_third = dict(job_id="third", category="main", state="pending", walltime_s=5400)
    assert budget_check(gpu_prior, "GPU", new_request=gpu_third)["status"] == "PASS_BUDGET"
    gpu_three = gpu_prior + [dict(gpu_third, state="failed", actual_walltime_s=197)]
    gpu_fourth = dict(job_id="fourth", category="main", state="pending", walltime_s=5400)
    fourth_gate = budget_check(gpu_three, "GPU", new_request=gpu_fourth)
    assert fourth_gate["status"] == "PASS_BUDGET"
    assert fourth_gate["gpu_hours_worst_total"] == pytest.approx(1.625)
    assert fourth_gate["core_hours_worst_total"] == pytest.approx(13.0)
    assert "NEW_MAX_JOB_IDS" in budget_check(
        gpu_three + [dict(gpu_fourth, state="completed", actual_walltime_s=1)], "GPU",
        new_request=dict(job_id="fifth", category="repair", state="pending", walltime_s=1))["reasons"]
    ledger = [{"job_id": "batch", "category": "batch", "state": "completed", "walltime_s": 1000}]
    result = budget_check(ledger, "CPU", new_request={"job_id": "cpu1", "category": "small", "state": "pending", "walltime_s": 1200})
    assert result["status"] == "PASS_BUDGET"
    blocked_inflight = budget_check([
        {"job_id": "1", "category": "small", "state": "running", "walltime_s": 1200},
        {"job_id": "2", "category": "small", "state": "pending", "walltime_s": 1200},
    ], "CPU")
    assert blocked_inflight["status"] == "BUDGET_BLOCKED"
    assert "MAX_INFLIGHT" in blocked_inflight["reasons"]
    exact = budget_check(
        [{"job_id": "g0", "category": "repair", "state": "completed", "walltime_s": 1800, "gpus": 1, "cpus": 8}],
        "GPU",
        new_request={"job_id": "g1", "category": "main", "state": "pending", "walltime_s": 5400},
    )
    assert exact["status"] == "PASS_BUDGET"
    exceeded = budget_check(
        [{"job_id": "g0", "category": "repair", "state": "completed", "walltime_s": 1801, "gpus": 1, "cpus": 8}],
        "GPU",
        new_request={"job_id": "g1", "category": "main", "state": "pending", "walltime_s": 5400},
    )
    assert exceeded["status"] == "BUDGET_BLOCKED"
    assert "GPU_HOURS_BUDGET" in exceeded["reasons"]


def test_budget_rejects_unknown_state_and_resource_counts_and_allows_cancelled_zero():
    with pytest.raises(BudgetError):
        budget_check([{"job_id": "x", "category": "mystery", "state": "completed", "walltime_s": 1}], "CPU")
    with pytest.raises(BudgetError):
        budget_check([{"job_id": "x", "category": "small", "state": "mystery", "walltime_s": 1}], "CPU")
    with pytest.raises(BudgetError):
        budget_check([{"job_id": "x", "category": "small", "state": "completed", "walltime_s": 1, "cpus": float("nan")}], "CPU")
    with pytest.raises(BudgetError):
        budget_check([{"job_id": "x", "category": "small", "state": "completed", "walltime_s": 1, "gpus": 1}], "CPU")
    result = budget_check([{"job_id": "x", "category": "small", "state": "cancelled", "walltime_s": 0}], "CPU")
    assert result["status"] == "PASS_BUDGET"


def test_actual_merged_cpu_schema_requires_completed_off_shell():
    from .records import ClockError, cpu_completed_metrics
    def row(label,start,end,index=None):
        return dict(label=label,start_wall_ns=start,end_wall_ns=end,start_cpu_ns=start,end_cpu_ns=end,
                    wall_inclusive_ns=end-start,cpu_inclusive_ns=end-start,failed=False,
                    raw_context=dict(interval_index=index))
    bodies=[row('INTERVAL_BODY',10,40,0),row('INTERVAL_BODY',50,90,1)]
    raw=dict(merged=dict(records=[row('PROPAGATION',0,100),*bodies]))
    value=cpu_completed_metrics(raw,2)
    assert value['window_completed_s']==100/1e9 and value['later_body_s']==40/1e9
    with pytest.raises(ClockError,match='WINDOW_OR_INTERVALS_MISSING'):
        cpu_completed_metrics(dict(merged=dict(records=bodies)),2)
    legacy=dict(bodies[0]);legacy['context']=legacy.pop('raw_context')
    with pytest.raises(ClockError,match='MERGED_TIMING_SCHEMA'):
        cpu_completed_metrics(dict(merged=dict(records=[row('PROPAGATION',0,100),legacy,bodies[1]])),2)


def test_full_gpu_memory_admission_uses_measured_workspace_and_free_capacity():
    from .gpu_memory_probe import FULL,GIB,full_admission
    total=32*GIB
    probe=dict(status='PASS_FFT_PLAN_RESOURCE_PROBE',device_total_bytes=total,
               grids=[dict(shape=list(FULL),plan_workspace_bytes=GIB,free_after_plans_bytes=28*GIB)])
    passed=full_admission(probe,2689588736,30*GIB,total)
    assert passed['status']=='PASS_FULLSIZE_MEMORY_FORECAST'
    assert passed['required_bytes']==32*(384*351*301*16)+3*GIB
    assert passed['measured_plan_workspace_bytes']==GIB
    blocked=full_admission(probe,2689588736,20*GIB,total)
    assert blocked['status']=='FULLSIZE_RESOURCE_BLOCKED'
    probe['grids'][0]['free_after_plans_bytes']=20*GIB
    assert full_admission(probe,2689588736,30*GIB,total)['status']=='FULLSIZE_RESOURCE_BLOCKED'
    assert full_admission(dict(status='BLOCKED'),2689588736,30*GIB,total)['status']=='FULLSIZE_RESOURCE_BLOCKED'
