"""Bounded N3 admission tests; no scheduler or scientific allocations."""
import json
import pytest

from KHz_filament.hr4e5_final_gates import _validate_runtime_budget, require_final_gate_path
from tools.hpc_ops.provenance_v2 import raw_sha256_file


def runtime_fixture(tmp_path):
    source = tmp_path / "source.json"
    old = []
    scenarios = []
    for name, scale in (("base", 1), ("conservative", 2), ("stress", 4)):
        row = dict(name=name, R_nonfinal_child_s=10000*scale,
                   C_critical_each_s=12000*scale, R_optical_each_s=2000*scale,
                   exact_total_s=579456, exact_metadata_s=50,
                   successor_reclaim_s=2000, init_terminal_s=1000, margin_fraction=.15)
        old.append(row)
        components = dict(pair_critical_s=26000*scale,
                          exact_s=(579456-50)*338016/579456+30,
                          successor_reclaim_s=1000, init_terminal_s=1000)
        components["margin_s"] = sum(components.values())*.15
        total = sum(components.values())
        scenarios.append(dict(row, components_s=components, total_s=total, total_h=total/3600))
    source.write_text(json.dumps(dict(scenarios=old)), encoding="utf-8")
    # Reduce exact time so the conservative scenario fits 120 hours.
    for row in old:
        row["exact_total_s"] = 57945.6
    source.write_text(json.dumps(dict(scenarios=old)), encoding="utf-8")
    for row in scenarios:
        c = row["components_s"]
        c["exact_s"] = (57945.6-50)*338016/579456+30
        c["margin_s"] = sum(v for k,v in c.items() if k != "margin_s")*.15
        row["total_s"] = sum(c.values()); row["total_h"] = row["total_s"]/3600
    budget = dict(schema="khz_filament.hr4e5.n3_runtime_budget.v1",
                  status="PASS_WITH_DOCUMENTED_ESTIMATION", n_pulses=3, k=8048,
                  allocation_seconds=432000, science_child_timeout_s=432000,
                  exact_field_pairs=338016, metadata_keys=30, nonfinal_pairs=2,
                  terminal_pairs=1, estimate_not_completion_guarantee=True,
                  scenarios=scenarios, conservative_total_s=scenarios[1]["total_s"],
                  source=dict(path=str(source), hash_scope="raw_bytes", raw_sha256=raw_sha256_file(source)))
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(budget), encoding="utf-8")
    gate = dict(status=budget["status"], budget_path=str(path),
                budget_raw_sha256=raw_sha256_file(path),
                conservative_total_s=budget["conservative_total_s"], allocation_seconds=432000)
    return path, budget, gate


def test_n3_runtime_budget_recomputed(tmp_path):
    _, budget, gate = runtime_fixture(tmp_path)
    result = _validate_runtime_budget(gate, campaign_root=tmp_path / "run")
    assert result["conservative_total_s"] == budget["conservative_total_s"]


def test_n3_runtime_source_rejects_self_consistent_underestimate(tmp_path):
    path, budget, gate = runtime_fixture(tmp_path)
    row = budget["scenarios"][1]
    row["components_s"]["exact_s"] /= 2
    c = row["components_s"]
    c["margin_s"] = sum(v for k,v in c.items() if k != "margin_s")*.15
    row["total_s"] = sum(c.values()); row["total_h"] = row["total_s"]/3600
    budget["conservative_total_s"] = gate["conservative_total_s"] = row["total_s"]
    path.write_text(json.dumps(budget), encoding="utf-8")
    gate["budget_raw_sha256"] = raw_sha256_file(path)
    with pytest.raises(ValueError, match="source|exact"):
        _validate_runtime_budget(gate, campaign_root=tmp_path / "run")


def test_n3_final_gate_path_required():
    with pytest.raises(ValueError, match="requires"):
        require_final_gate_path(None)


def test_n3_capacity_counts_objects_and_unchanged_allowance(tmp_path):
    from KHz_filament.hr4e5_journal_capacity import build_n3_journal_capacity_bound
    from KHz_filament.hr4e5_storage import budget_object_table
    bound = build_n3_journal_capacity_bound(campaign_root=tmp_path / "run", campaign_id="N3-test")
    assert bound["transaction_counts"]["base"] == 233420
    assert bound["transaction_counts"]["backpressure_max"] == 249516
    assert sum(bound["journal_bytes_by_root"].values()) == bound["journal_bytes_total"]
    assert sum(bound["disk_objects"].values()) == bound["additional_capacity_charge_bytes"]
    args = dict(n_pulses=3, k=8048, ny=351, nx=301, nt=384)
    old = budget_object_table(**args)
    explicit_none = budget_object_table(**args, journal_capacity_objects=None)
    assert old == explicit_none
    new = budget_object_table(**args, journal_capacity_objects=bound["disk_objects"])
    assert old["metadata_bytes"] == new["metadata_bytes"] == 2*1024**3
    for stage in old["stage_bytes"]:
        assert new["stage_bytes"][stage] - old["stage_bytes"][stage] == sum(bound["disk_objects"].values())
        assert sum(row["bytes"] for row in new["stages"][stage]) == new["stage_bytes"][stage]


def test_n3_capacity_policy_boundary_refuses_one_byte(tmp_path):
    from KHz_filament.hr4e5_journal_capacity import build_n3_journal_capacity_bound
    from KHz_filament.hr4e5_storage import plan_campaign_budget, StorageBudgetError
    bound = build_n3_journal_capacity_bound(campaign_root=tmp_path / "run", campaign_id="N3-test")
    args = dict(n_pulses=3, k=8048, ny=351, nx=301, nt=384,
                journal_capacity_objects=bound["disk_objects"])
    plan = plan_campaign_budget(**args)
    assert plan_campaign_budget(**args, max_campaign_live_bytes=plan["peak_bytes"])["headroom_bytes"] == 0
    with pytest.raises(StorageBudgetError, match="exceeds cap"):
        plan_campaign_budget(**args, max_campaign_live_bytes=plan["peak_bytes"]-1)


def test_n3_guard_rejects_overbound_without_truncation(tmp_path):
    from KHz_filament.hr4e5_journal_capacity import validate_n3_journal_entry_strings, MAX_WORKER_ID_BYTES
    args = dict(campaign_root=tmp_path / "run", campaign_id="N3-test", actor="coordinator",
                event="OPTICAL_START", worker_id="worker", cuda_visible_devices="0,1,2,3,4,5",
                process_identity=dict(pid=1, host_identity="host:boot", process_start_token="123"),
                current_generation="R:p0:PRE0", next_generation="R:p0:PRE0:next")
    validate_n3_journal_entry_strings(**args)
    args["worker_id"] = "X"*(MAX_WORKER_ID_BYTES+1)
    with pytest.raises(ValueError, match="bound"):
        validate_n3_journal_entry_strings(**args)


def test_n3_legacy_optional_gate_preserves_admission(tmp_path):
    from dataclasses import replace
    from test_hr4e5_e5_1a_r import _local_production_spec
    from KHz_filament.hr4e5_production import _build_admission
    spec = _local_production_spec(tmp_path, n_pulses=3)
    before = _build_admission(spec)[0]
    after = _build_admission(replace(spec, final_gate_manifest_path=None))[0]
    assert before == after
    assert "final_gate_identity" not in before["runtime_or_compatibility"]
    with pytest.raises(ValueError, match="fixed N3 formal scope"):
        _build_admission(replace(spec, final_gate_manifest_path=tmp_path / "missing.json"))


def test_n3_manifest_binds_input_source_and_report(tmp_path):
    from KHz_filament.hr4e5_final_gates import (
        N3_FORMAL_SCOPE, FINAL_GATES_SCHEMA, PERFORMANCE_CHECKS, PERFORMANCE_HOT_PATHS,
        validate_final_gate_manifest,
    )
    from tools.hpc_ops.provenance_v2 import canonical_lf_sha256
    _, _, runtime = runtime_fixture(tmp_path)
    def write(name, value):
        p = tmp_path / name
        p.write_text(json.dumps(value), encoding="utf-8")
        return p
    hot = []
    for relative in PERFORMANCE_HOT_PATHS:
        p = tmp_path / "package" / relative
        p.parent.mkdir(exist_ok=True, parents=True); p.write_bytes(b"# fixture\n")
        hot.append(dict(path=relative, canonical_lf_sha256=canonical_lf_sha256(p.read_bytes())))
    perf = write("performance.json", dict(status="PASS", checks=dict.fromkeys(PERFORMANCE_CHECKS, True),
        ratios=dict(p95_hold_speedup=10, metadata_write_reduction=.99, matched_serial_throughput_speedup=15),
        metrics=dict(new_serial=dict(observer_fraction=.01,full_manifest_writes=0),
                     new_concurrent=dict(observer_fraction=.01,full_manifest_writes=0,wait_max_s=2))))
    bound = {"fixture": 1}
    capacity = write("capacity.json", dict(status="PASS",bound=bound))
    inputs = {"config": dict(path="fixture", raw_sha256="a"*64)}
    manifest = dict(schema=FINAL_GATES_SCHEMA,campaign_id="N3-test",scope=N3_FORMAL_SCOPE,
        implementation_identity_sha256="b"*64,inputs=inputs,runtime=runtime,
        performance=dict(status="PASS",evidence_path=str(perf),evidence_raw_sha256=raw_sha256_file(perf),
                         baseline_head="46b6a446754e44b8b126911e759954720fe9438c",unchanged_hot_paths=hot),
        storage=dict(status="PASS",bound=bound,capacity_report_path=str(capacity),
                     capacity_report_raw_sha256=raw_sha256_file(capacity)))
    path = write("gates.json",manifest)
    args = dict(campaign_root=tmp_path/"run",package_root=tmp_path/"package",campaign_id="N3-test",
                expected_inputs=inputs,implementation_identity_sha256="b"*64,journal_capacity_bound=bound)
    result = validate_final_gate_manifest(path,**args)
    assert result["raw_sha256"] == raw_sha256_file(path)
    for key, value in (("expected_inputs",{}),("implementation_identity_sha256","c"*64),
                       ("journal_capacity_bound",{"fixture":2})):
        with pytest.raises(ValueError):
            validate_final_gate_manifest(path,**dict(args,**{key:value}))
    capacity.write_text('{}',encoding="utf-8")
    with pytest.raises(ValueError,match="SHA256"):
        validate_final_gate_manifest(path,**args)


def test_n3_full_k_barrier_promotion_actual_encoder_and_replay(tmp_path, monkeypatch):
    """Exercise real metadata transitions/encoder; array provenance is fixture-only."""
    from contextlib import nullcontext
    from KHz_filament.hr4e5_streaming_journal import JournalAuthority, _ChangeSet
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle
    from KHz_filament.hr4e5_journal_capacity import build_n3_journal_capacity_bound
    root = tmp_path / "state"; root.mkdir()
    records = []
    for i in range(8048):
        artifact = dict(file_sha256="a"*64, metadata=dict(ordinal=i, screen_id=f"source_index_{i:06d}",z_m=float(i)))
        records.append(dict(ordinal=i,screen_id=f"source_index_{i:06d}",z_m=float(i),
                            state="NEXT_COMMITTED",retry_count=0,current=artifact,
                            post=artifact,next=artifact,transitions=[]))
    manifest = dict(schema="khz_filament.hr4e5s.streaming.v1",expected_screen_count=8048,
        current_generation="R:p0:PRE0",next_generation="R:p0:PRE0:next",current_content_sha256="b"*64,
        records=records,queue=[],queue_depth=16,recovery_backlog=[],recovery_backlog_cursor=0,
        barrier=None,promotion=None,telemetry_events=[])
    raw=(json.dumps(manifest,sort_keys=True,indent=2)+"\n").encode()
    (root/"streaming_manifest.json").write_bytes(raw)
    authority=JournalAuthority.initialize(root,raw); changes=_ChangeSet()
    life=StreamingLifecycle.__new__(StreamingLifecycle);life.root=root
    life.manifest,head,_=authority.load(changes=changes)
    monkeypatch.setattr(life,"_locked_manifest",lambda *args,**kwargs:nullcontext())
    monkeypatch.setattr(life,"_assert_no_staged_or_orphaned_artifacts",lambda:None)
    monkeypatch.setattr(life,"_validate_record_provenance",lambda *args,**kwargs:None)
    monkeypatch.setattr(life,"_artifact_metadata",lambda entry:entry["metadata"])
    monkeypatch.setattr(life,"_maybe_inject_s5_fault",lambda *args,**kwargs:None)
    sizes=[]
    def save():
        nonlocal head
        operation="promote_next_to_current" if any(op.get("path")==["promotion"] for op in changes.ops) else "validate_barrier"
        before=head.committed_offset
        head,_=authority.commit(changes.ops,previous=head,operation=operation,actor="fixture",process={"pid":1})
        sizes.append((operation,len(changes.ops),head.committed_offset-before));changes.ops.clear()
    monkeypatch.setattr(life,"_save",save)
    assert life.validate_barrier(actor="fixture")["status"]=="PASS"
    life.promote_next_to_current(actor="fixture")
    bound=build_n3_journal_capacity_bound(campaign_root=tmp_path/"run",campaign_id="N3-test")
    assert max(size for _,_,size in sizes)<=bound["encoded_transaction_upper_bounds"]["barrier_transition"]["bytes"]
    assert any(count==1+2*8048 for _,count,_ in sizes)
    replayed,reopened,_=JournalAuthority(root).load()
    assert reopened==head and replayed["promotion"]["authoritative_namespace"]=="NEXT"
    assert all(row["state"]=="BARRIER_VALIDATED" for row in replayed["records"])
