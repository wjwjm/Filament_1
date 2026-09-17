"""Focused refusal tests from the independent E5-1A entry review."""
import numpy as np
import pytest

from test_hr4e5_e5_1a import _fields, _records, _schedule


def test_direct_final_replay_rejects_changed_post(tmp_path):
    from KHz_filament.hr4e5_formal_entry import commit_final_post
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle, StreamingLifecycleError
    state = StreamingLifecycle.create(root=tmp_path/'state', current=_fields(),
        screen_records=_records(), current_generation='fixture:pre', dx_m=1e-4, dy_m=1e-4)
    before = state.current_fields(0)['delta_n']
    commit_final_post(lifecycle_root=state.root, ordinal=0, state_after=before)
    commit_final_post(lifecycle_root=state.root, ordinal=0, state_after=before.copy(), resume=True)
    with pytest.raises(StreamingLifecycleError, match='replay differs'):
        commit_final_post(lifecycle_root=state.root, ordinal=0, state_after=before+1, resume=True)


@pytest.mark.parametrize('invalid', ['disabled_hr3b', 'shifted_schedule', 'grid_spacing'])
def test_optical_contract_rejects_before_output_creation(tmp_path, invalid):
    from KHz_filament.config import BeamConfig, GridConfig, HeatConfig, IonizationConfig, PropagationConfig, RamanConfig, RunConfig
    from KHz_filament.hr4e5_formal_entry import run_streaming_optical_pulse
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle
    from KHz_filament.longitudinal import build_longitudinal_schedule
    state = StreamingLifecycle.create(root=tmp_path/'state', current=_fields(shape=(8,8)),
        screen_records=_records(), current_generation='fixture:pre', dx_m=1e-4, dy_m=1e-4)
    grid = GridConfig(Nx=8, Ny=8, Nt=8, Lx=8e-4, Ly=8e-4, Twin=80e-15)
    heat = HeatConfig(hr3b_enabled=invalid != 'disabled_hr3b')
    schedule = _schedule()
    if invalid == 'shifted_schedule':
        schedule = build_longitudinal_schedule(dz=2e-4, z_max=16e-4)
    if invalid == 'grid_spacing':
        grid.Lx = 16e-4
    components = dict(grid=grid, beam=BeamConfig(), prop=PropagationConfig(),
        ion=IonizationConfig(), heat=heat, run=RunConfig(), raman=RamanConfig())
    with pytest.raises(ValueError, match='HR-3B|coordinates|grid differs'):
        run_streaming_optical_pulse(lifecycle_root=state.root, schedule=schedule,
            output_dir=tmp_path/'output', components=components)
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('changed', ['exact.json', 'dependency.json', 'child/field.bin'])
def test_gc_revalidates_durable_dependencies_before_delete(tmp_path, changed):
    from test_hr4e5_e5_1a import _gc_prerequisite, _writer_receipt
    from KHz_filament.hr4e5_storage import StorageBudget, StorageBudgetError
    budget = StorageBudget(tmp_path/'campaign', campaign_id='fixture-campaign', safety_margin_bytes=128)
    target = budget.root/'owned.bin'
    target.write_bytes(b'owned')
    budget.register_artifact(target, role='intermediate', reclaimable=True)
    prerequisite = _gc_prerequisite(budget, [target])
    plan = budget.plan_reclaim([target], exact_complete=True, successor_ready=True,
        no_active_writers=True, no_future_dependency=True, receipts_durable=True,
        writer_receipt=_writer_receipt(tmp_path/'writer.json'), prerequisite_receipt=prerequisite)
    (prerequisite.parent/changed).write_bytes(b'changed')
    with pytest.raises(StorageBudgetError):
        budget.apply_reclaim(plan['plan_id'])
    assert target.read_bytes() == b'owned'


def test_reservation_overlap_and_consumption_bound(tmp_path):
    from KHz_filament.hr4e5_storage import StorageBudget, StorageBudgetError, MockQuotaProvider
    budget = StorageBudget(tmp_path, safety_margin_bytes=128,
        provider=MockQuotaProvider(free_bytes=2**30, quota_bytes=2**30))
    with pytest.raises(StorageBudgetError, match='overlapping'):
        budget.reserve(100, purpose='invalid', allocation_paths=['a', 'a/b'])
    reservation = budget.reserve(8, purpose='bounded', allocation_paths=['payload.bin'])
    (tmp_path/'payload.bin').write_bytes(b'0123456789')
    with pytest.raises(StorageBudgetError, match='exceeded'):
        budget.consume(reservation.reservation_id)
    assert budget._read()['reservations'][reservation.reservation_id]['status'] == 'ACTIVE'


def test_stale_coordinator_rereads_state_before_pair(tmp_path):
    from KHz_filament.hr4e5_paired_campaign import PairedCampaign
    a = PairedCampaign.create(tmp_path, safety_margin_bytes=128)
    b = PairedCampaign.open(tmp_path, safety_margin_bytes=128)
    steps = dict(reference_step=lambda *a: {'status':'PASS'}, candidate_step=lambda *a: {'status':'PASS'},
        compare_step=lambda *a: {'status':'PASS'}, successor_step=lambda *a: {'status':'PASS'},
        reclaim_step=lambda *a: {'status':'PASS'})
    assert a.run_pair(0, **steps).status == 'PASS'
    with pytest.raises(ValueError, match='not the next durable pair'):
        b.run_pair(0, **steps)
