"""Fresh-process actual L1 factory and fail-closed instrumentation checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

PROBE=r'''
import json,sys
from pathlib import Path
from tools.cpu_s4_5a_l1_residual.candidate import build_role,verify_bindings
profile=sys.argv[1]=='PROFILE'
built=build_role('PURE_OFF_SMALLGRID',capture_root=Path(sys.argv[2]),local_engineering=True,pool_override=1,profile=profile)
state=built[1].cpu_s45a_state
gate=verify_bindings(state)
stats=built[1].cpu_s4_4a_cache_snapshot()
if stats['max_bytes']!=268435456 or stats['max_tables']!=2 or stats['stats']['factor_requests']!=0:raise RuntimeError('ORIGINAL_EMPTY_SMALLGRID_CACHE_REQUIRED')
negative=[]
if profile:
    original=state._internal_stage_inner.__code__
    def replacement():return None
    try:
        state._internal_stage_inner.__code__=replacement.__code__
        try:verify_bindings(state)
        except RuntimeError as error:
            if 'INSTRUMENT_STAGE_CODE_CHANGED' not in str(error):raise
            negative.append('INSTRUMENT_STAGE_CODE_CHANGED')
        else:raise RuntimeError('CHANGED_ACTUAL_CODE_WAS_ACCEPTED')
    finally:state._internal_stage_inner.__code__=original
    verify_bindings(state)
    namespace=state._cpu_s45a_private_namespace
    original_xp=namespace['xp']
    try:
        namespace['xp']=object()
        try:verify_bindings(state)
        except RuntimeError as error:
            if 'INSTRUMENT_GLOBAL_ALIAS_CHANGED:xp' not in str(error):raise
            negative.append('INSTRUMENT_GLOBAL_ALIAS_CHANGED:xp')
        else:raise RuntimeError('CHANGED_NUMPY_ALIAS_WAS_ACCEPTED')
    finally:namespace['xp']=original_xp
    verify_bindings(state)
    helper=namespace['_s43c_product'];cells=dict(zip(helper.__code__.co_freevars,helper.__closure__))
    original_call=cells['original'].cell_contents
    try:
        cells['original'].cell_contents=lambda *args,**kwargs:None
        try:verify_bindings(state)
        except RuntimeError as error:
            if 'INSTRUMENT_FIXED_CLOSURE_CHANGED:_s43c_product' not in str(error):raise
            negative.append('INSTRUMENT_FIXED_CLOSURE_CHANGED:_s43c_product')
        else:raise RuntimeError('FOREIGN_INNER_CALLBACK_WAS_ACCEPTED')
    finally:cells['original'].cell_contents=original_call
    verify_bindings(state)
    old=state._cpu_s45a_private_namespace['__cpu_s45a_scope__']
    # Actual registered monitoring tools remain forbidden by the inherited G gate.
    tool=5
    if sys.monitoring.get_tool(tool) is not None:raise RuntimeError('CLEAN_CHILD_MONITORING_SLOT_REQUIRED')
    try:
        sys.monitoring.use_tool_id(tool,'S45A_NEGATIVE')
        try:verify_bindings(state)
        except RuntimeError as error:
            if 'UNKNOWN_SYS_MONITORING_TOOL' not in str(error):raise
            negative.append('UNKNOWN_SYS_MONITORING_TOOL')
        else:raise RuntimeError('MONITORING_TOOL_WAS_ACCEPTED')
    finally:sys.monitoring.free_tool_id(tool)
    verify_bindings(state)
    from tools.cpu_s4_4a_linear_cache import candidate as parent
    from tools.cpu_s4_5a_l1_residual.candidate import _clone_make_factory
    original_factory=parent.make_factory
    try:
        parent.make_factory=lambda:None
        try:_clone_make_factory(profile=True,recorder=built[1].cpu_s45a_recorder)
        except RuntimeError as error:
            if 'S44A_PARENT_FACTORY_PATH_OR_GLOBALS_MISMATCH' not in str(error):raise
            negative.append('S44A_PARENT_FACTORY_PATH_OR_GLOBALS_MISMATCH')
        else:raise RuntimeError('FOREIGN_PARENT_FACTORY_WAS_ACCEPTED')
    finally:parent.make_factory=original_factory
    _clone_make_factory(profile=True,recorder=built[1].cpu_s45a_recorder)
print(json.dumps(dict(status='PASS_ACTUAL_L1_FACTORY_BINDINGS_AND_RECOVERY',profile=profile,gate=gate,negative_reasons=negative,cache=stats,science_executed=False)))
'''

@pytest.mark.parametrize('role',('OFF','PROFILE'))
def test_s45a_actual_fresh_factory_binding_and_negative_recovery(role,tmp_path):
    env=os.environ.copy();env['NUMBA_CACHE_DIR']=str(tmp_path/'numba')
    result=subprocess.run([sys.executable,'-s','-B','-c',PROBE,role,str(tmp_path/'capture')],env=env,capture_output=True,text=True,encoding='utf-8',timeout=180)
    (tmp_path/'stdout.log').write_text(result.stdout,encoding='utf-8');(tmp_path/'stderr.log').write_text(result.stderr,encoding='utf-8')
    assert result.returncode==0,result.stderr
    record=json.loads(result.stdout.strip().splitlines()[-1])
    assert record['status']=='PASS_ACTUAL_L1_FACTORY_BINDINGS_AND_RECOVERY' and record['profile']==(role=='PROFILE')
    assert record['negative_reasons']==(['INSTRUMENT_STAGE_CODE_CHANGED','INSTRUMENT_GLOBAL_ALIAS_CHANGED:xp','INSTRUMENT_FIXED_CLOSURE_CHANGED:_s43c_product','UNKNOWN_SYS_MONITORING_TOOL','S44A_PARENT_FACTORY_PATH_OR_GLOBALS_MISMATCH'] if role=='PROFILE' else [])
