"""Pre-science source/AST seams and fail-closed helper slots."""
import json
from pathlib import Path
import pytest
from .adapter import BodyClock
from .candidate import build_role,verify_bindings

@pytest.mark.parametrize('role',('L1_REF_5C','C_EQ10_V1'))
def test_s45c_actual_factory_metadata(role,tmp_path):
    built=build_role('OFF',role=role,capture_root=tmp_path/'BOUNDARY',local_engineering=True,pool_override=1)
    dispatch=built[1];verify_bindings(dispatch)
    assert dispatch.cpu_s4_4a_cache_snapshot()['stats']['factor_requests']==0
    assert dispatch.cpu_s45c_executor is None if role=='L1_REF_5C' else dispatch.cpu_s45c_executor.snapshot()['dispatch_count']==0
    clock=built[3]['s45c_body_clock'];assert clock['baseline_ast_sha256']==clock['stripped_ast_sha256']
    assert built[3]['profile_enabled'] is False
    if role=='C_EQ10_V1':
        assert built[3]['s45c_stage_binding']['undo_AST_exact']
        state=dispatch.cpu_s4_3g_state;namespace=state._internal_stage_inner.__globals__
        original=namespace['__cpu_s45c_k1__'];namespace['__cpu_s45c_k1__']=lambda *unused:None
        try:
            with pytest.raises(RuntimeError,match='C_KERNEL_SLOT_CHANGED'):verify_bindings(dispatch)
        finally:namespace['__cpu_s45c_k1__']=original
        verify_bindings(dispatch)
    (tmp_path/'FACTORY_METADATA.json').write_text(json.dumps(dict(status='PASS_ACTUAL_C_FACTORY_METADATA',role=role,
        binding=built[3],science_execution='NOT_STARTED'),indent=2),encoding='utf-8')

def test_s45c_common_clock_complete_ordinals():
    clock=BodyClock()
    with clock.body(0):pass
    with clock.body(1):pass
    rows=clock.snapshot()['rows']
    assert [row['ordinal'] for row in rows]==[1,2]
    assert all(row['wall_ns']>=0 and row['cpu_ns']>=0 and not row['failed'] for row in rows)
