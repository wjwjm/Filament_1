"""Bounded N3 admission tests; no scheduler or scientific allocations."""
import json
from pathlib import Path
import subprocess
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


def performance_requalification_fixture(tmp_path, package, performance_gate):
    from KHz_filament.hr4e5_final_gates import (
        NORMAL_MAX_JOURNAL_BYTES, PERFORMANCE_REQUALIFICATION_TOOLS,
        TERMINAL_MAX_JOURNAL_BYTES,
    )
    from tools.hpc_ops.provenance_v2 import create_manifest

    repo = package.parent
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True, capture_output=True)
    package.mkdir(parents=True, exist_ok=True)
    relative_paths = (
        "Filament_python/KHz_filament/hr4e5_streaming_journal.py",
        "Filament_python/KHz_filament/hr4e5s_streaming.py",
    )
    for index, relative in enumerate(relative_paths):
        module_relative = relative.removeprefix("Filament_python/")
        module = package / module_relative
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_bytes(f"# module {index}\n".encode())
    subprocess.run(["git", "-C", str(repo), "add", "--", *relative_paths],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Codex Fixture",
                    "-c", "user.email=codex-fixture@example.invalid", "commit", "-m", "fixture"],
                   check=True, capture_output=True)
    source_manifest = create_manifest(
        repo, tmp_path / "fixture_source_provenance.json",
        tracked_paths=relative_paths, external_paths=(),
    )
    source_records = {
        record["path"]: {key: record[key] for key in (
            "git_blob_oid", "canonical_lf_sha256", "hash_scope")}
        for record in source_manifest["records"]
    }

    gate_value = json.loads(performance_gate.read_text(encoding="utf-8"))
    gate_sha = raw_sha256_file(performance_gate)
    normal_time = 0.2766064116731286
    terminal_time = 0.419
    site_root = tmp_path / "site"
    site_root.mkdir()
    tool_hashes = {}
    for name in PERFORMANCE_REQUALIFICATION_TOOLS:
        tool = site_root / name
        tool.write_bytes(("tool:" + name).encode())
        tool_hashes[name] = raw_sha256_file(tool)
    candidate_path = site_root / "source" / "Filament_python" / "KHz_filament" / "hr4e5_streaming_journal.py"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_bytes((package / "KHz_filament" / "hr4e5_streaming_journal.py").read_bytes())
    baseline_path = site_root / "baseline_journal.py"
    fixture = dict(
        schema="n3.journal_performance_requalification_fixture.v2",
        science_calls=0, formal_submissions=0,
        normal_max_journal_bytes=NORMAL_MAX_JOURNAL_BYTES,
        terminal_max_journal_bytes=TERMINAL_MAX_JOURNAL_BYTES,
        status="PASS_SITE_FIXTURE_ONLY",
        candidate_module=dict(path=str(candidate_path), raw_sha256=raw_sha256_file(candidate_path)),
        baseline_module=dict(path=str(baseline_path), raw_sha256=raw_sha256_file(baseline_path)),
        metadata_authority_cases={
            "pre_repair": {"status": "PASS_EXPECTED_REJECT"},
            "repaired": {"status": "PASS_ACCEPT_AND_REHASH"},
        },
        same_length_corruption={"status": "PASS_REJECT_SAME_LENGTH_CORRUPTION"},
        capacity_sized_valid_journal_rechecks=[
            dict(label="normal_max_C_p0_journal", journal_bytes=NORMAL_MAX_JOURNAL_BYTES,
                 status="PASS_VALID_SYNTHETIC_AUTHORITY_BOUNDARY",
                 metadata_refresh_digest_recheck_count=1,
                 metadata_refresh_elapsed_s=normal_time),
            dict(label="terminal_max_single_journal", journal_bytes=TERMINAL_MAX_JOURNAL_BYTES,
                 status="PASS_VALID_SYNTHETIC_AUTHORITY_BOUNDARY",
                 metadata_refresh_digest_recheck_count=1,
                 metadata_refresh_elapsed_s=terminal_time),
        ],
    )
    fixture_path = site_root / "SITE_FIXTURE_REQUAL.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    receipt = dict(
        schema="n3.performance_requalification.v1",
        status="PASS_PERFORMANCE_REQUALIFICATION_ONLY",
        performance_allocation_only=True, science_calls=0, formal_submissions=0,
        runtime=dict(slurm_job_id="275219", slurm_cpus_per_task=8,
                     allocated_gpu_count=1, UPPE_USE_GPU="0"),
        source_identities=dict(runtime_manifest=dict(
            head=subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True).stdout.strip(),
            journal_and_lifecycle=source_records)),
        performance_gate=dict(path=str(performance_gate), raw_sha256=gate_sha,
                              status="PASS", checks=gate_value["checks"],
                              ratios=gate_value["ratios"]),
        journal_fixture=dict(path=str(fixture_path), raw_sha256=raw_sha256_file(fixture_path),
                             status="PASS_SITE_FIXTURE_ONLY",
                             max_single_recheck_s=max(normal_time, terminal_time)),
        tool_raw_sha256=tool_hashes,
    )
    receipt_path = site_root / "N3_PERFORMANCE_REQUALIFICATION.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    performance = dict(status="PASS", evidence_path=str(performance_gate),
                       evidence_raw_sha256=gate_sha,
                       baseline_head="46b6a446754e44b8b126911e759954720fe9438c",
                       revalidation_path=str(receipt_path),
                       revalidation_raw_sha256=raw_sha256_file(receipt_path))
    return performance, receipt_path, receipt


def add_runtime_revalidation_budget(budget, gate, receipt_path, *, counts=None,
                                    normal_max=None, terminal_max=None):
    normal_max = 0.2766064116731286 if normal_max is None else normal_max
    terminal_max = 0.419 if terminal_max is None else terminal_max
    counts = {"base": 233420, "conservative": 249516, "stress": 499032} if counts is None else counts
    budget["journal_revalidation"] = dict(
        revalidation_path=str(receipt_path),
        revalidation_raw_sha256=raw_sha256_file(receipt_path),
        normal_max_recheck_s=normal_max,
        terminal_max_recheck_s=terminal_max,
        scenario_recheck_counts=counts,
        terminal_recheck_count=6,
        planning_basis={
            "base": "One maximum normal-journal recheck per base lifecycle transaction.",
            "conservative": "One maximum normal-journal recheck per backpressure transaction.",
            "stress": "Two maximum normal-journal rechecks per backpressure transaction.",
        },
    )
    for row in budget["scenarios"]:
        name = row["name"]
        components = row["components_s"]
        components["journal_revalidation_s"] = counts[name] * normal_max + 6 * terminal_max
        fraction = row["margin_fraction"]
        components["margin_s"] = sum(value for key, value in components.items() if key != "margin_s") * fraction
        row["total_s"] = sum(components.values())
        row["total_h"] = row["total_s"] / 3600
    budget["conservative_total_s"] = next(
        row["total_s"] for row in budget["scenarios"] if row["name"] == "conservative")
    budget_path = Path(gate["budget_path"])
    budget_path.write_text(json.dumps(budget), encoding="utf-8")
    gate["budget_raw_sha256"] = raw_sha256_file(budget_path)
    gate["conservative_total_s"] = budget["conservative_total_s"]


def final_gate_requalification_fixture(tmp_path):
    from KHz_filament.hr4e5_final_gates import (
        N3_FORMAL_SCOPE, FINAL_GATES_SCHEMA, PERFORMANCE_CHECKS,
    )

    tmp_path.mkdir(parents=True, exist_ok=True)
    _, _, runtime = runtime_fixture(tmp_path)
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    performance_gate = write("performance_requalified.json", dict(
        status="PASS", checks=dict.fromkeys(PERFORMANCE_CHECKS, True),
        ratios=dict(p95_hold_speedup=10, metadata_write_reduction=.99,
                    matched_serial_throughput_speedup=15),
        metrics=dict(new_serial=dict(observer_fraction=.01, full_manifest_writes=0),
                     new_concurrent=dict(observer_fraction=.01, full_manifest_writes=0,
                                         wait_max_s=2))))
    package = tmp_path / "source_repo" / "Filament_python"
    performance, receipt_path, receipt = performance_requalification_fixture(
        tmp_path, package, performance_gate)
    budget = json.loads(Path(runtime["budget_path"]).read_text(encoding="utf-8"))
    add_runtime_revalidation_budget(budget, runtime, receipt_path)
    from KHz_filament.hr4e5_journal_capacity import build_n3_journal_capacity_bound
    bound = build_n3_journal_capacity_bound(campaign_root=tmp_path / "run", campaign_id="N3-requal")
    capacity = write("capacity_requalified.json", dict(status="PASS", bound=bound))
    inputs = {"config": dict(path="fixture", raw_sha256="a" * 64)}
    manifest = dict(schema=FINAL_GATES_SCHEMA, campaign_id="N3-requal",
        scope=N3_FORMAL_SCOPE, implementation_identity_sha256="b" * 64,
        inputs=inputs, runtime=runtime, performance=performance,
        storage=dict(status="PASS", bound=bound, capacity_report_path=str(capacity),
                     capacity_report_raw_sha256=raw_sha256_file(capacity)))
    manifest_path = write("gates_requalified.json", manifest)
    args = dict(campaign_root=tmp_path / "run", package_root=package,
                campaign_id="N3-requal", expected_inputs=inputs,
                implementation_identity_sha256="b" * 64,
                journal_capacity_bound=bound)
    return manifest_path, manifest, args, receipt_path, receipt, package, performance_gate


def validate_requalified_fixture(fixture):
    from KHz_filament.hr4e5_final_gates import validate_final_gate_manifest
    manifest_path, manifest, args, *_ = fixture
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return validate_final_gate_manifest(manifest_path, **args)


def update_requalification_fixture_artifacts(fixture):
    _, manifest, _, receipt_path, receipt, *_ = fixture
    fixture_path = Path(receipt["journal_fixture"]["path"])
    receipt["journal_fixture"]["raw_sha256"] = raw_sha256_file(fixture_path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    manifest["performance"]["revalidation_raw_sha256"] = raw_sha256_file(receipt_path)
    budget_path = Path(manifest["runtime"]["budget_path"])
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    budget["journal_revalidation"]["revalidation_raw_sha256"] = raw_sha256_file(receipt_path)
    budget_path.write_text(json.dumps(budget), encoding="utf-8")
    manifest["runtime"]["budget_raw_sha256"] = raw_sha256_file(budget_path)


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


def test_n3_performance_requalification_and_runtime_overhead_are_bound(tmp_path):
    fixture = final_gate_requalification_fixture(tmp_path)
    result = validate_requalified_fixture(fixture)
    _, _, _, receipt_path, _, _, _ = fixture
    assert result["performance"]["revalidation"]["raw_sha256"] == raw_sha256_file(receipt_path)
    journal_revalidation = result["runtime"]["journal_revalidation"]
    assert journal_revalidation["terminal_recheck_count"] == 6
    assert journal_revalidation["scenario_overhead_s"]["conservative"] == pytest.approx(
        249516 * 0.2766064116731286 + 6 * 0.419)


def test_n3_performance_requalification_requires_path_and_digest_pair(tmp_path):
    fixture = final_gate_requalification_fixture(tmp_path)
    fixture[1]["performance"].pop("revalidation_raw_sha256")
    with pytest.raises(ValueError, match="supplied together"):
        validate_requalified_fixture(fixture)


def test_n3_performance_requalification_rejects_missing_source_identity(tmp_path):
    fixture = final_gate_requalification_fixture(tmp_path)
    _, manifest, _, receipt_path, receipt, *_ = fixture
    receipt["source_identities"]["runtime_manifest"]["journal_and_lifecycle"].pop(
        "Filament_python/KHz_filament/hr4e5_streaming_journal.py")
    update_requalification_fixture_artifacts(fixture)
    with pytest.raises(ValueError, match="source identities"):
        validate_requalified_fixture(fixture)


def test_n3_performance_requalification_rejects_changed_actual_source(tmp_path):
    fixture = final_gate_requalification_fixture(tmp_path)
    package = fixture[5]
    (package / "KHz_filament" / "hr4e5_streaming_journal.py").write_bytes(b"# changed source\n")
    with pytest.raises(ValueError, match="differs from the running module"):
        validate_requalified_fixture(fixture)


def test_n3_performance_requalification_binds_hashed_gate_contents(tmp_path):
    fixture = final_gate_requalification_fixture(tmp_path)
    _, manifest, _, receipt_path, receipt, *_ = fixture
    receipt["performance_gate"]["raw_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    manifest["performance"]["revalidation_raw_sha256"] = raw_sha256_file(receipt_path)
    with pytest.raises(ValueError, match="hash differs from the hashed gate"):
        validate_requalified_fixture(fixture)


def test_n3_performance_requalification_rejects_changed_tool(tmp_path):
    fixture = final_gate_requalification_fixture(tmp_path)
    receipt_path = fixture[3]
    (receipt_path.parent / "site_control_driver_n3.py").write_bytes(b"changed driver")
    with pytest.raises(ValueError, match="tool hash mismatch"):
        validate_requalified_fixture(fixture)


def test_n3_performance_requalification_checks_git_blob_with_provenance_authority(tmp_path):
    fixture = final_gate_requalification_fixture(tmp_path)
    receipt = fixture[4]
    record = receipt["source_identities"]["runtime_manifest"]["journal_and_lifecycle"][
        "Filament_python/KHz_filament/hr4e5_streaming_journal.py"]
    record["git_blob_oid"] = "0" * 40
    update_requalification_fixture_artifacts(fixture)
    with pytest.raises(ValueError, match="Git blob identity differs from the running source"):
        validate_requalified_fixture(fixture)


def test_n3_performance_requalification_binds_fixture_candidate_and_baseline_raw_hashes(tmp_path):
    fixture = final_gate_requalification_fixture(tmp_path)
    receipt = fixture[4]
    fixture_path = Path(receipt["journal_fixture"]["path"])
    actual = json.loads(fixture_path.read_text(encoding="utf-8"))
    actual["candidate_module"]["raw_sha256"] = "0" * 64
    fixture_path.write_text(json.dumps(actual), encoding="utf-8")
    update_requalification_fixture_artifacts(fixture)
    with pytest.raises(ValueError, match="candidate raw hash"):
        validate_requalified_fixture(fixture)

    fixture = final_gate_requalification_fixture(tmp_path / "baseline")
    receipt = fixture[4]
    fixture_path = Path(receipt["journal_fixture"]["path"])
    actual = json.loads(fixture_path.read_text(encoding="utf-8"))
    actual["baseline_module"]["raw_sha256"] = "0" * 64
    fixture_path.write_text(json.dumps(actual), encoding="utf-8")
    update_requalification_fixture_artifacts(fixture)
    with pytest.raises(ValueError, match="baseline module differs from its hashed baseline tool"):
        validate_requalified_fixture(fixture)


def test_n3_runtime_revalidation_must_match_performance_receipt(tmp_path):
    fixture = final_gate_requalification_fixture(tmp_path)
    manifest = fixture[1]
    budget = json.loads(Path(manifest["runtime"]["budget_path"]).read_text(encoding="utf-8"))
    budget["journal_revalidation"]["revalidation_raw_sha256"] = "0" * 64
    budget_path = Path(manifest["runtime"]["budget_path"])
    budget_path.write_text(json.dumps(budget), encoding="utf-8")
    manifest["runtime"]["budget_raw_sha256"] = raw_sha256_file(budget_path)
    with pytest.raises(ValueError, match="receipt hash differs from performance gate"):
        validate_requalified_fixture(fixture)


def test_n3_runtime_revalidation_rejects_counts_below_capacity_budget(tmp_path):
    fixture = final_gate_requalification_fixture(tmp_path)
    _, manifest, _, receipt_path, *_ = fixture
    budget = json.loads(Path(manifest["runtime"]["budget_path"]).read_text(encoding="utf-8"))
    add_runtime_revalidation_budget(
        budget, manifest["runtime"], receipt_path,
        counts={"base": 0, "conservative": 0, "stress": 0})
    manifest["runtime"]["budget_raw_sha256"] = raw_sha256_file(
        Path(manifest["runtime"]["budget_path"]))
    with pytest.raises(ValueError, match="counts understate or differ"):
        validate_requalified_fixture(fixture)


def test_n3_runtime_revalidation_rejects_underestimated_recheck_time(tmp_path):
    fixture = final_gate_requalification_fixture(tmp_path)
    _, manifest, _, receipt_path, *_ = fixture
    budget = json.loads(Path(manifest["runtime"]["budget_path"]).read_text(encoding="utf-8"))
    add_runtime_revalidation_budget(
        budget, manifest["runtime"], receipt_path, normal_max=0.2)
    manifest["runtime"]["budget_raw_sha256"] = raw_sha256_file(
        Path(manifest["runtime"]["budget_path"]))
    with pytest.raises(ValueError, match="underestimates the measured maximum"):
        validate_requalified_fixture(fixture)


def test_n3_runtime_revalidation_over_120_hours_fails_closed(tmp_path):
    fixture = final_gate_requalification_fixture(tmp_path)
    _, manifest, _, receipt_path, *_ = fixture
    budget = json.loads(Path(manifest["runtime"]["budget_path"]).read_text(encoding="utf-8"))
    add_runtime_revalidation_budget(
        budget, manifest["runtime"], receipt_path, normal_max=2.0)
    manifest["runtime"]["budget_raw_sha256"] = raw_sha256_file(
        Path(manifest["runtime"]["budget_path"]))
    with pytest.raises(ValueError, match="exceeds the 120-hour allocation"):
        validate_requalified_fixture(fixture)


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
