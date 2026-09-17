"""Low-cost E5-1A glue qualification tests.

These tests use only tiny float64 screen volumes.  The real-call test invokes
the repository's existing ``propagate_one_pulse`` on an 8-screen CPU fixture;
the remaining tests exercise durable orchestration and storage contracts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest


def _schedule(count: int = 8):
    from KHz_filament.longitudinal import build_longitudinal_schedule

    return build_longitudinal_schedule(dz=1.0e-4, z_max=count * 1.0e-4)


def _fields(count: int = 8, shape: tuple[int, int] = (4, 4), *, velocity: bool = False):
    yy, xx = np.indices(shape, dtype=np.float64)
    delta = np.stack([-1.0e-6 * (index + 1) * (1.0 + xx + yy) for index in range(count)]).astype(np.float64)
    vx = np.stack([np.full(shape, 0.01 * (index + 1), dtype=np.float64) for index in range(count)]) if velocity else np.zeros_like(delta)
    vy = np.stack([np.full(shape, -0.02 * (index + 1), dtype=np.float64) for index in range(count)]) if velocity else np.zeros_like(delta)
    return {"delta_n": delta, "vx": vx, "vy": vy}


def _records(count: int = 8):
    schedule = _schedule(count)
    return [{"ordinal": index, "screen_id": f"fixture_{index:04d}", "z_m": float((schedule.z_edges[index] + schedule.z_edges[index + 1]) / 2.0)} for index in range(count)]


def _complete_stream(root: Path, *, final: bool = False):
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle

    lifecycle = StreamingLifecycle.create(
        root=root, current=_fields(), screen_records=_records(), current_generation="fixture:pre", dx_m=1.0e-4, dy_m=1.0e-4,
    )
    for ordinal in range(8):
        lifecycle.deposition_finalized(ordinal)
        current = lifecycle.current_fields(ordinal)
        lifecycle.commit_post(ordinal, current if not final else {name: np.array(value, copy=True) for name, value in current.items()})
        if not final:
            lifecycle.enqueue_post(ordinal)
    if not final:
        for ordinal in lifecycle.claim_block():
            lifecycle.commit_next(ordinal, lifecycle._artifact_fields(lifecycle.manifest["records"][ordinal]["post"], namespace="POST"))
        lifecycle.validate_barrier()
        lifecycle.promote_next_to_current()
    return lifecycle


def test_a01_prefix_pre0_and_read_view_are_identity_bound(tmp_path):
    from KHz_filament.hr4e5_formal_entry import StreamingPulseReadView, build_prefix_schedule, create_pre0_root

    full = _schedule(16)
    prefix = build_prefix_schedule(full, 8)
    assert prefix.z_edges == full.z_edges[:9]
    assert prefix.dz_intervals == full.dz_intervals[:8]
    assert prefix.intervals == full.intervals[:8]
    with pytest.raises(ValueError, match="block"):
        build_prefix_schedule(full, 6)
    lifecycle = create_pre0_root(root=tmp_path / "pre0", delta_n=_fields()["delta_n"], schedule=prefix, dx_m=1e-4, dy_m=1e-4)
    view = StreamingPulseReadView(lifecycle)
    before = lifecycle.current_fields(0)["delta_n"].copy()
    value = view.read_interval(0)
    value[...] = 123.0
    np.testing.assert_array_equal(lifecycle.current_fields(0)["delta_n"], before)
    post = view.update_interval(0, np.full((4, 4), -1e-8))
    payload = view.post_fields(0, post)
    np.testing.assert_array_equal(payload["vx"], np.zeros((4, 4)))
    assert not view.complete
    view.close()


def test_a02_a03_successor_carries_nonzero_velocity_and_exact_binding(tmp_path):
    from KHz_filament.hr4e5_formal_entry import create_successor_root, open_successor_root, validate_successor_ready
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle

    fields = _fields(velocity=True)
    parent = StreamingLifecycle.create(root=tmp_path / "parent", current=fields, screen_records=_records(), current_generation="fixture:pre", dx_m=1e-4, dy_m=1e-4)
    for ordinal in range(8):
        parent.deposition_finalized(ordinal)
        current = parent.current_fields(ordinal)
        parent.commit_post(ordinal, current)
        parent.enqueue_post(ordinal)
    for ordinal in parent.claim_block():
        parent.commit_next(ordinal, parent._artifact_fields(parent.manifest["records"][ordinal]["post"], namespace="POST"))
    parent.validate_barrier()
    parent.promote_next_to_current()
    child, receipt = create_successor_root(parent_root=parent.root, child_root=tmp_path / "child")
    assert receipt["status"] == "READY"
    assert validate_successor_ready(child.root)["child_root"] == str(child.root.resolve())
    np.testing.assert_array_equal(child.current_fields(0)["vx"], fields["vx"][0])
    with pytest.raises(Exception, match="archived"):
        open_successor_root(parent.root)


def test_a06_final_post_is_post_only_and_reopen_is_idempotent(tmp_path):
    from KHz_filament.hr4e5_formal_entry import commit_final_post, resume_final_post, validate_final_post
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle

    lifecycle = StreamingLifecycle.create(root=tmp_path / "final", current=_fields(), screen_records=_records(), current_generation="fixture:pre", dx_m=1e-4, dy_m=1e-4)
    for ordinal in range(8):
        commit_final_post(lifecycle_root=lifecycle.root, ordinal=ordinal, state_after=lifecycle.current_fields(ordinal)["delta_n"])
    failed = validate_final_post(lifecycle_root=lifecycle.root)
    assert failed["status"] == "FAIL" and "writer_quiescence_not_attested" in failed["failures"]
    receipt = tmp_path / "POST_FINAL_READY.json"
    passed = validate_final_post(lifecycle_root=lifecycle.root, writer_quiescent=True)
    assert passed["status"] == "PASS"
    assert all(record["next"] is None for record in lifecycle.manifest["records"])
    assert not lifecycle.manifest["queue"]
    with pytest.raises(ValueError, match='replay inputs'):
        resume_final_post(lifecycle_root=lifecycle.root, receipt_path=receipt)


def test_a08_exact_requires_shape_dtype_finite_and_array_equality():
    from KHz_filament.hr4e5_evidence import compare_arrays_exact, compare_object_sets

    left = np.ones((2, 2), dtype=np.float64)
    assert compare_arrays_exact(left, left.copy())["status"] == "PASS"
    assert compare_arrays_exact(left, left.astype(np.float32))["status"] == "FAIL"
    assert compare_arrays_exact(left, left.reshape(4))["status"] == "FAIL"
    changed = left.copy(); changed[0, 1] = 2.0
    mismatch = compare_arrays_exact(left, changed)
    assert mismatch["status"] == "FAIL" and mismatch["array_equal"] is False
    nonfinite = left.copy(); nonfinite[0, 0] = np.nan
    assert compare_arrays_exact(left, nonfinite)["status"] == "FAIL"
    assert compare_object_sets({"a": left}, {"b": left})["status"] == "FAIL"


def _writer_receipt(path: Path, *, campaign_id: str = "fixture-campaign", epoch: str = "epoch-1"):
    path.write_text(json.dumps({"schema": "fixture.writer.v1", "status": "PASS", "campaign_id": campaign_id, "active_writers": [], "writer_epoch": epoch, "coordinator_process_id": "pytest"}), encoding="utf-8")
    return path


def _gc_prerequisite(budget, targets, *, label="gc"):
    """Create a tiny durable exact/READY/dependency receipt for GC tests."""
    from KHz_filament.hr4e5_storage import RECLAIM_PREREQUISITE_SCHEMA
    from KHz_filament.hr4e_timestep import sha256_file

    evidence = budget.root / f"evidence_{label}"
    child = evidence / "child"
    child.mkdir(parents=True, exist_ok=True)
    field = child / "field.bin"
    field.write_bytes(b"self-contained-child")
    binding = child / "binding.json"
    binding.write_text(json.dumps({"status": "PASS", "rows": [{"status": "PASS"}]}), encoding="utf-8")
    ready = child / "READY.json"
    ready.write_text(json.dumps({
        "schema": "fixture.ready.v1", "status": "READY", "self_contained": True,
        "parent_payload_required": False, "child_root": str(child),
        "fields": {"field": {"path": "field.bin", "sha256": sha256_file(field)}},
        "binding": {"path": str(binding), "sha256": sha256_file(binding)},
    }), encoding="utf-8")
    exact = evidence / "exact.json"
    exact.write_text(json.dumps({"status": "PASS", "mismatch_count": 0,
                                 "missing_reference": [], "missing_candidate": [],
                                 "rows": [{"status": "PASS"}]}), encoding="utf-8")
    dependency = evidence / "dependency.json"
    dependency.write_text(json.dumps({"status": "PASS", "no_future_dependency": True}), encoding="utf-8")
    artifacts = budget.artifacts()
    bindings = []
    for target in targets:
        relative = target.resolve().relative_to(budget.root).as_posix()
        item = artifacts[relative]
        bindings.append({"relative_path": relative, "identity": dict(item["identity"]), "sha256": item["sha256"]})
    prerequisite = evidence / "prerequisites.json"
    prerequisite.write_text(json.dumps({
        "schema": RECLAIM_PREREQUISITE_SCHEMA, "status": "PASS",
        "campaign_id": budget.campaign_id,
        "gates": {name: True for name in (
            "exact_complete", "successor_ready", "no_active_writers",
            "no_future_dependency", "receipts_durable",
        )},
        "target_bindings": bindings,
        "evidence": {
            "exact_complete": {"path": str(exact), "sha256": sha256_file(exact)},
            "successor_ready": {"path": str(ready), "sha256": sha256_file(ready)},
            "no_future_dependency": {"path": str(dependency), "sha256": sha256_file(dependency)},
        },
    }), encoding="utf-8")
    return prerequisite


def test_a09_a10_a11_storage_gc_is_durable_and_whitelisted(tmp_path):
    from KHz_filament.hr4e5_storage import StorageBudget, StorageBudgetError

    root = tmp_path / "storage"
    budget = StorageBudget(root, cap_bytes=1024 * 1024, final_output_budget_bytes=512 * 1024, safety_margin_bytes=128, campaign_id="fixture-campaign")
    first, second = root / "r1.bin", root / "r2.bin"
    first.write_bytes(b"one"); second.write_bytes(b"two")
    budget.register_artifact(first, role="reference_intermediate", reclaimable=True)
    budget.register_artifact(second, role="candidate_intermediate", reclaimable=True)
    writer = _writer_receipt(tmp_path / "writer.json")
    with pytest.raises(StorageBudgetError, match="prerequisites"):
        budget.plan_reclaim([first], exact_complete=False, successor_ready=True, no_active_writers=True, no_future_dependency=True, receipts_durable=True, writer_receipt=writer)
    prerequisite = _gc_prerequisite(budget, [first, second], label="fixture")
    plan = budget.plan_reclaim([first, second], exact_complete=True, successor_ready=True, no_active_writers=True, no_future_dependency=True, receipts_durable=True, writer_receipt=writer, prerequisite_receipt=prerequisite, plan_id="gc-fixture")
    interrupted = budget.apply_reclaim("gc-fixture", interrupt_after=1)
    assert interrupted["status"] == "INTERRUPTED"
    ledger = json.loads((root / "storage_ledger.json").read_text(encoding="utf-8"))
    assert ledger["gc_plans"]["gc-fixture"]["targets"][0]["status"] == "DELETED"
    resumed = budget.resume_reclaim("gc-fixture")
    assert resumed["status"] == "COMPLETED" and budget.verify_reclaim("gc-fixture")["status"] == "PASS"
    assert not first.exists() and not second.exists()


def test_a12_storage_rejects_hardlinks_and_unknown_paths(tmp_path):
    from KHz_filament.hr4e5_storage import StorageBudget, StorageIntegrityError

    root = tmp_path / "storage"
    budget = StorageBudget(root, cap_bytes=1024 * 1024, final_output_budget_bytes=512 * 1024, safety_margin_bytes=128, campaign_id="fixture-campaign")
    source = root / "source.bin"; source.write_bytes(b"x")
    linked = root / "linked.bin"
    try:
        os.link(source, linked)
    except (OSError, NotImplementedError):
        pytest.skip("hardlink semantics unavailable")
    budget.register_artifact(linked, role="intermediate", reclaimable=True)
    writer = _writer_receipt(tmp_path / "writer.json")
    with pytest.raises(StorageIntegrityError, match="hard"):
        budget.plan_reclaim([linked], exact_complete=True, successor_ready=True, no_active_writers=True, no_future_dependency=True, receipts_durable=True, writer_receipt=writer, prerequisite_receipt=_gc_prerequisite(budget, [linked], label="hardlink"))
    with pytest.raises(StorageIntegrityError, match="escapes"):
        budget.plan_reclaim([tmp_path / "outside.bin"], exact_complete=True, successor_ready=True, no_active_writers=True, no_future_dependency=True, receipts_durable=True, writer_receipt=writer)


def test_a13_a14_reservations_are_cross_call_fail_closed(tmp_path):
    from KHz_filament.hr4e5_storage import HARD_CAP_BYTES, MockQuotaProvider, StorageBudget, StorageBudgetError

    assert HARD_CAP_BYTES == 322122547200
    with pytest.raises(StorageBudgetError, match="hard limit"):
        StorageBudget(tmp_path / "too_large", cap_bytes=HARD_CAP_BYTES + 1, safety_margin_bytes=1)
    provider = MockQuotaProvider(free_bytes=100000, quota_bytes=100000)
    budget = StorageBudget(tmp_path / "quota", cap_bytes=100000, final_output_budget_bytes=50000, safety_margin_bytes=1024, provider=provider)
    budget.reserve(40000, purpose="first", reservation_id="r1")
    with pytest.raises(StorageBudgetError, match="quota|filesystem|cap"):
        budget.reserve(60000, purpose="double-spend", reservation_id="r2")
    provider.set_free(1)
    with pytest.raises(StorageBudgetError, match="filesystem"):
        budget.check_capacity(1)


def test_pair_metadata_reopen_and_event_predicate_unit_only(tmp_path):
    from KHz_filament.hr4e5_paired_campaign import PairedCampaign, validate_overlap_events

    root = tmp_path / "campaign"
    campaign = PairedCampaign.create(root, n_pulses=3, campaign_id="fixture-campaign", safety_margin_bytes=128)
    campaign.register_process_start(process_id="p0")

    def side(index):
        return {"status": "PASS", "pulse_index": index}

    def compare(index, reference, candidate):
        return {"status": "PASS", "pulse_index": index, "same_identity": reference["pulse_index"] == candidate["pulse_index"]}

    gates = dict(successor_step=lambda *a: {'status':'PASS'}, reclaim_step=lambda *a: {'status':'PASS'})
    first = campaign.run_pair(0, reference_step=side, candidate_step=side, compare_step=compare, **gates)
    assert first.status == "PASS" and campaign.next_pair_index == 1
    campaign.register_process_exit(process_id="p0")
    resumed = PairedCampaign.resume_from_disk(root, campaign_id="fixture-campaign", safety_margin_bytes=128)
    resumed.register_process_start(process_id="p1", source="new_process")
    result = resumed.run(reference_step=side, candidate_step=side, compare_step=compare, **gates)
    assert result["status"] == "COMPLETE" and result["next_pair_index"] == 3
    events = [
        {"event": "OPTICAL_START", "monotonic_s": 1.0},
        {"event": "HYDRO_BLOCK_START", "monotonic_s": 2.0},
        {"event": "OPTICAL_COMPLETE", "monotonic_s": 3.0},
    ]
    assert validate_overlap_events(events)["status"] == "PASS"
    events[1]["monotonic_s"] = 4.0
    assert validate_overlap_events(events)["status"] == "FAIL"


@pytest.mark.parametrize('partial_final', [False, True])
def test_a05_real_existing_propagate_and_hr3b_injection(tmp_path, partial_final):
    """Real CPU optical call plus existing Streaming POST hook on 8 screens."""
    from KHz_filament.config import BeamConfig, GridConfig, HeatConfig, IonizationConfig, PropagationConfig, RamanConfig, RunConfig
    from KHz_filament.hr4e5_formal_entry import run_streaming_optical_pulse
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle

    grid = GridConfig(Nx=8, Ny=8, Nt=8, Lx=8e-4, Ly=8e-4, Twin=80e-15)
    beam = BeamConfig(w0=1.5e-4, tau_fwhm=40e-15, energy_J=1e-10, focal_length=None)
    prop = PropagationConfig(z_max=8e-4, dz=1e-4, linear_model="paraxial", auto_substep=False, focus_window_step=False, limit_focus_window=False, progress_every_z=0, energy_probe_every=0, diag_extra=False, use_electronic_kerr=False, use_raman_phase=False, use_raman_absorption=False, use_plasma_phase=False, use_ionization_loss=False, use_ionization_solver=False)
    heat = HeatConfig(hr3b_enabled=True)
    components = {"grid": grid, "beam": beam, "prop": prop, "ion": IonizationConfig(species=[]), "heat": heat, "run": RunConfig(Npulses=1), "raman": RamanConfig(enabled=False, absorption=False)}
    lifecycle = StreamingLifecycle.create(root=tmp_path / "real", current=_fields(shape=(8, 8)), screen_records=_records(), current_generation="fixture:pre", dx_m=grid.Lx / grid.Nx, dy_m=grid.Ly / grid.Ny)
    from KHz_filament.hr4e5_storage import StorageBudget, MockQuotaProvider
    budget = StorageBudget(tmp_path, safety_margin_bytes=1024**2,
        provider=MockQuotaProvider(free_bytes=2**40, quota_bytes=2**40))
    kwargs=dict(lifecycle_root=lifecycle.root, schedule=_schedule(), output_dir=tmp_path / 'optical', components=components, storage_budget=budget)
    if partial_final:
        from KHz_filament.propagate import propagate_one_pulse
        from KHz_filament.hr4e5_formal_entry import resume_final_post
        from KHz_filament.hr4e_timestep import sha256_file
        def interrupted(field,**kw):
            hook=kw['post_commit_hook']
            def stop_after_post(**payload):
                hook(**payload)
                if payload['interval'].index==3: raise InterruptedError('fixture optical interrupt')
            kw['post_commit_hook']=stop_after_post
            return propagate_one_pulse(field,**kw)
        with pytest.raises(InterruptedError):
            run_streaming_optical_pulse(**kwargs,final=True,propagate_fn=interrupted)
        before={p.name:sha256_file(p) for p in (lifecycle.root/'post').glob('*.npz')}
        assert len(before)==4
        replay=dict(kwargs);replay.pop('lifecycle_root')
        result=resume_final_post(lifecycle_root=lifecycle.root,receipt_path=lifecycle.root/'POST_FINAL_READY.json',replay_kwargs=replay)
        assert result['status']=='PASS'
        assert before=={name:sha256_file(lifecycle.root/'post'/name) for name in before}
        assert resume_final_post(lifecycle_root=lifecycle.root,receipt_path=lifecycle.root/'POST_FINAL_READY.json')['status']=='PASS'
        assert not list((lifecycle.root/'next').glob('*.npz'))
        return
    result = run_streaming_optical_pulse(**kwargs)
    assert result["status"] == "PASS"
    assert result["schedule_intervals"] == 8
    reopened = StreamingLifecycle.open(lifecycle.root)
    assert all(record["post"] is not None and record["state"] == "HYDRO_QUEUED" for record in reopened.manifest["records"])
    assert (tmp_path / "optical" / "final_optical_field.npy").is_file()


def test_persistent_three_pair_fixture_with_real_hydro(tmp_path):
    """Fresh OS processes; full arrays exact before deleting parent payloads."""
    import subprocess
    import sys
    import time
    import copy
    from KHz_filament.hr4e5_paired_campaign import validate_overlap_events
    from KHz_filament.hr4e5_formal_entry import resume_final_post
    from KHz_filament.hr4e5s_streaming import _atomic_json
    script = Path(__file__).with_name('hr4e5_fixture_campaign.py')
    split=tmp_path/'split'; continuous=tmp_path/'continuous'
    logs=[]
    def invoke(root,stop):
        result=subprocess.run([sys.executable,'-s','-B',str(script),str(root),'--stop',str(stop)],
            text=True,capture_output=True,timeout=180)
        logs.append({'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        assert result.returncode==0,result.stdout+'\n'+result.stderr
        return json.loads(result.stdout.splitlines()[-1])
    first=invoke(split,1)
    assert not list((split/'C/p0/state/current').glob('*.npz'))
    assert not (split/'R/p0/pre_delta_n.npy').exists()
    second=invoke(split,3)
    assert first['pid']!=second['pid'] and first['epoch']!=second['epoch']
    invoke(continuous,3)
    summaries=[]
    for p in range(3):
        report=json.loads((split/f'exact{p}.json').read_text())
        assert report['screen_count']==(12 if p==2 else 15)*32
        assert report['ledger_count']==9 and report['optical_count']==1
        summaries.append({k:v for k,v in report.items() if k!='rows'})
        for side in ('R','C'):
            np.testing.assert_array_equal(np.load(split/side/f'p{p}/final_optical_field.npy'),
                                          np.load(continuous/side/f'p{p}/final_optical_field.npy'))
        if p<2:
            manifest=json.loads((split/f'C/p{p}/state/streaming_manifest.json').read_text())
            assert validate_overlap_events(manifest['telemetry_events'])['status']=='PASS'
    for i in range(32):
        for namespace in ('current','post'):
            name=f'C/p2/state/{namespace}/screen_{i:06d}.npz'
            with np.load(split/name) as a,np.load(continuous/name) as b:
                for f in ('delta_n','vx','vy'): np.testing.assert_array_equal(a[f],b[f])
    assert not (split/'C/p3').exists()
    assert resume_final_post(lifecycle_root=split/'C/p2/state',receipt_path=split/'C/p2/state/POST_FINAL_READY.json')['status']=='PASS'
    # A18: three representative full-manifest writes, not a K-squared campaign.
    manifest=json.loads((split/'C/p2/state/streaming_manifest.json').read_text())
    prototype=copy.deepcopy(manifest)
    prototype['records']=[dict(manifest['records'][i%32],ordinal=i) for i in range(8048)]
    prototype['expected_screen_count']=8048
    probe=tmp_path/'metadata_probe.json'; durations=[]
    for _ in range(3):
        start=time.perf_counter();_atomic_json(probe,prototype);durations.append(time.perf_counter()-start)
    result={'scope':'TEST_FIXTURE_ONLY; optical double + original CPU hydro',
        'NEW_PROCESS_PERSISTENT_RESUME':'PASS','processes':[first,second],
        'screen_array_comparisons':sum(x['screen_count'] for x in summaries),
        'ledger_comparisons':27,'optical_comparisons':3,'K':32,'N':3,
        'pairs':summaries,'subprocess_logs':logs,
        'metadata_probe':{'prototype_K':8048,'writes':3,'bytes_per_write':probe.stat().st_size,
            'duration_s':durations,'limitation':'local prototype only; no K-squared loop or HPC I/O extrapolation'},
        'final_live_bytes':sum(p.stat().st_size for p in split.rglob('*') if p.is_file())}
    evidence=os.environ.get('E5_1A_EVIDENCE_DIR')
    if evidence:
        dest=Path(evidence);dest.mkdir(parents=True,exist_ok=True)
        (dest/'integrated_fixture.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        (dest/'tiny_terminal_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


def test_a15_a17_bounded_final_output_and_integer_hydro_contract():
    from KHz_filament.hr4e5_storage import plan_final_output, StorageBudgetError
    from KHz_filament.hr4e5_formal_entry import interpulse_worker_parameters
    args=dict(n_pulses=3,k=8048,ny=351,nx=301,nt=384)
    plan=plan_final_output(**args)
    assert plan['final_output_bytes']==47504725760
    assert plan['final_output_budget_bytes']==68719476736
    with pytest.raises(StorageBudgetError):plan_final_output(**dict(args,n_pulses=100))
    with pytest.raises(StorageBudgetError):plan_final_output(**args,diagnostics=['ion'])
    assert interpulse_worker_parameters(f_rep=5e6,dt_hydro=1e-7)['n_hydro_steps']==2
    with pytest.raises(ValueError,match='remainder'):
        interpulse_worker_parameters(f_rep=4e6,dt_hydro=1e-7)


def test_a07_ready_rechecks_payload_and_completes_missing_receipt(tmp_path):
    from KHz_filament.hr4e5_formal_entry import create_successor_root, validate_successor_ready
    parent=_complete_stream(tmp_path/'parent')
    child,receipt=create_successor_root(parent_root=parent.root,child_root=tmp_path/'child')
    # Remove only this test's receipt to reproduce a complete-root/receipt gap.
    (child.root/'E5_1A_READY.json').unlink()
    create_successor_root(parent_root=parent.root,child_root=child.root)
    create_successor_root(parent_root=parent.root,child_root=child.root)
    file=child.root/'current/screen_000000.npz'
    with file.open('ab') as stream: stream.write(b'identity replacement')
    with pytest.raises(ValueError,match='hash'):
        validate_successor_ready(child.root)


def test_a14_two_process_reservations_and_unknown_quota(tmp_path):
    import subprocess, sys
    from KHz_filament.hr4e5_storage import StorageBudget, StorageBudgetError, MockQuotaProvider
    root=tmp_path/'race'
    StorageBudget(root,cap_bytes=100000,final_output_budget_bytes=50000,safety_margin_bytes=1024,require_quota=False)
    code='''import sys
from KHz_filament.hr4e5_storage import StorageBudget, StorageBudgetError
b=StorageBudget(sys.argv[1],cap_bytes=100000,final_output_budget_bytes=50000,safety_margin_bytes=1024,require_quota=False)
print("READY",flush=True)
sys.stdin.readline()
try:
 b.reserve(60000,purpose="competing-writer")
 print("ACCEPTED",flush=True)
except StorageBudgetError:
 print("REFUSED",flush=True)
'''
    processes=[subprocess.Popen([sys.executable,'-s','-B','-c',code,str(root)],stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
    try:
        assert all(p.stdout.readline().strip()=='READY' for p in processes)
        for p in processes: p.stdin.write('GO\n');p.stdin.flush()
        outputs=[p.communicate(timeout=30) for p in processes]
        assert sorted(o[0].strip() for o in outputs)==['ACCEPTED','REFUSED']
        assert all(p.returncode==0 for p in processes)
    finally:
        for p in processes:
            if p.poll() is None:p.kill();p.wait()
    unknown=StorageBudget(tmp_path/'unknown',require_quota=True,
        provider=MockQuotaProvider(free_bytes=2**40,quota_bytes=None))
    with pytest.raises(StorageBudgetError,match='quota is unknown'):unknown.reserve(1,purpose='must-refuse')


def test_a11_hard_process_exit_after_unlink_and_unknown_missing(tmp_path):
    import subprocess,sys
    from KHz_filament.hr4e5_storage import StorageBudget, StorageIntegrityError
    root=tmp_path/'gc';b=StorageBudget(root,safety_margin_bytes=1024,campaign_id='fixture-campaign')
    target=root/'owned.bin';target.write_bytes(b'owned');b.register_artifact(target,role='intermediate',reclaimable=True)
    writer=_writer_receipt(root/'writer.json')
    plan=b.plan_reclaim([target],exact_complete=True,successor_ready=True,no_active_writers=True,
        no_future_dependency=True,receipts_durable=True,writer_receipt=writer,
        prerequisite_receipt=_gc_prerequisite(b, [target], label='crash'), plan_id='crash')
    code='''import os,sys
from KHz_filament.hr4e5_storage import StorageBudget
b=StorageBudget(sys.argv[1],safety_margin_bytes=1024,campaign_id="fixture-campaign")
b.apply_reclaim("crash",fault_hook=lambda event,path:os._exit(73))
'''
    done=subprocess.run([sys.executable,'-s','-B','-c',code,str(root)],capture_output=True,timeout=30)
    assert done.returncode==73
    assert not target.exists()
    b.resume_reclaim('crash');b.resume_reclaim('crash')
    assert b.verify_reclaim('crash')['status']=='PASS'
    other=root/'unknown-disappearance.bin';other.write_bytes(b'owned')
    b.register_artifact(other,role='intermediate',reclaimable=True)
    b.plan_reclaim([other],exact_complete=True,successor_ready=True,no_active_writers=True,
        no_future_dependency=True,receipts_durable=True,writer_receipt=writer,
        prerequisite_receipt=_gc_prerequisite(b, [other], label='unknown'), plan_id='unknown')
    other.unlink()
    with pytest.raises(StorageIntegrityError):b.resume_reclaim('unknown')


def test_a12_identity_reparse_and_prefix_collision(tmp_path,monkeypatch):
    import KHz_filament.hr4e5_storage as s
    root=tmp_path/'owned';b=s.StorageBudget(root,safety_margin_bytes=1024,campaign_id='fixture-campaign')
    outside=tmp_path/'owned-sibling';outside.mkdir();sentinel=outside/'sentinel';sentinel.write_bytes(b'protected')
    with pytest.raises(s.StorageIntegrityError):b.register_artifact(sentinel,role='intermediate',reclaimable=True)
    target=root/'target';target.write_bytes(b'original');b.register_artifact(target,role='intermediate',reclaimable=True)
    writer=_writer_receipt(root/'writer.json')
    args=dict(exact_complete=True,successor_ready=True,no_active_writers=True,no_future_dependency=True,
        receipts_durable=True,writer_receipt=writer)
    plan=b.plan_reclaim([target],prerequisite_receipt=_gc_prerequisite(b, [target], label='identity'),**args)
    target.write_bytes(b'replaced')
    with pytest.raises(s.StorageIntegrityError):b.apply_reclaim(plan['plan_id'])
    original=s._is_reparse
    monkeypatch.setattr(s,'_is_reparse',lambda p:Path(p)==target or original(p))
    with pytest.raises(s.StorageIntegrityError):b.register_artifact(target,role='intermediate',reclaimable=True)
    assert sentinel.read_bytes()==b'protected' and target.read_bytes()==b'replaced'


@pytest.mark.parametrize('gate',['exact_complete','successor_ready','no_active_writers','no_future_dependency','receipts_durable'])
def test_a09_each_reclaim_gate_refuses_without_deletion(tmp_path,gate):
    from KHz_filament.hr4e5_storage import StorageBudget,StorageBudgetError
    b=StorageBudget(tmp_path/'root',safety_margin_bytes=1024,campaign_id='fixture-campaign')
    p=b.root/'owned';p.write_bytes(b'owned');b.register_artifact(p,role='intermediate',reclaimable=True)
    args={name:True for name in ['exact_complete','successor_ready','no_active_writers','no_future_dependency','receipts_durable']}
    args[gate]=False
    with pytest.raises(StorageBudgetError):b.plan_reclaim([p],writer_receipt=_writer_receipt(tmp_path/'writer.json'),**args)
    assert p.read_bytes()==b'owned'
