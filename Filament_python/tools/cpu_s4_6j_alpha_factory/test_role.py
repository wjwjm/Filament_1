from tools.cpu_s4_5b_steady_cost.adapter import make_recorder_pair
from tools.cpu_s4_6j_alpha_factory.role_spec import build_role, verify_role, close_role
from tools.cpu_s4_6j_alpha_factory.measurement import _snapshot
from tools.cpu_s4_6j_alpha_factory.protocol import CANDIDATE, MICRO_SHAPE
import os
import json
from pathlib import Path
import pytest

def test_cost_factory_has_four_pools_and_source_bound_alpha_clock(tmp_path):
    pair = make_recorder_pair(False)
    built = None
    try:
        with pair.b.scope('L1_FACTORY', common=True):
            built = build_role(CANDIDATE, os.environ.get('S46J_FACTORY_PROBE_MODE','PURE_OFF_FULLSIZE'), capture_root=tmp_path/'unused',
                role='R', local_engineering=True, recorder_pair=pair, cost_mode='COST', shape=MICRO_SHAPE)
        assert built[1].cpu_s4_6i_selected_role=='X'
        assert len(built)==6
        verify_role(built[1])
        hook = built[1].cpu_s4_6i_power_hook
        original = hook._alpha_recorder
        hook._alpha_recorder = make_recorder_pair(False).b
        with pytest.raises(RuntimeError, match='RECORDER_IDENTITY'):
            hook._alpha_scope(0)
        hook._alpha_recorder = original
    finally:
        if built is not None:
            closed = close_role(built[1])
            assert closed['all_pools_retired_before_output']
    snapshot = _snapshot(built[1], CANDIDATE)
    assert set(snapshot['components'])=={'intensity','eq10','c','power_volume'}
    assert snapshot['closed'] and not snapshot['active_thread_ids']
    assert any(row['context']['source_site'].startswith('S46J.factory.') for row in pair.b.snapshot()['records'])
    assert not (tmp_path/'unused').exists()
    if os.environ.get('S46J_FACTORY_PROBE'):
        path = Path(os.environ['S46J_FACTORY_PROBE'])
        with path.open('x', encoding='utf-8') as stream:
            json.dump(pair.b.snapshot(),stream,sort_keys=True,indent=2)
