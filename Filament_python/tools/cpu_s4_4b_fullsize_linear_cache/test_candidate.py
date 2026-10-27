"""Actual accepted factory seam: baseline, directed refusal, restored use."""
import numpy as np
import pytest
from . import contract
from .candidate import SourceBindingError, _replace_empty_cache, build_role
from .cache import FullsizeLinearFactorCache

def old_role(tmp_path):
    from tools.cpu_s4_4a_linear_cache.candidate import build_role as old_build
    return old_build('PURE_OFF_SMALLGRID',capture_root=tmp_path,local_engineering=True,pool_override=1)

def replace(dispatch):
    return _replace_empty_cache(dispatch,local_engineering=True,configured_shape=contract.MICRO_SHAPE_32)

def test_actual_factory_preserves_state_owner_and_local_budget(tmp_path):
    built=build_role('PURE_OFF_FULLSIZE',capture_root=tmp_path,local_engineering=True,pool_override=1,shape=contract.MICRO_SHAPE_32)
    state=built[1].cpu_s4_4a_state;cache=state._cpu_s4_4a_cache
    assert type(cache) is FullsizeLinearFactorCache and cache.owner_identity==f'state:{id(state)}'
    assert built[1].cpu_s4_4b_cache_snapshot()['max_bytes']==256*1024**2
    assert cache.snapshot()['stats']['factor_requests']==cache.snapshot()['stats']['entries']==0
    metadata=FullsizeLinearFactorCache(owner_identity='metadata-only',algorithm_source_identity='metadata-only',configured_shape=contract.FULL_SHAPE)
    assert metadata.max_bytes==1610612736 and metadata.snapshot()['stats']['cache_bytes']==0

def test_actual_factory_unknown_source_refused_then_restored(tmp_path):
    built=old_role(tmp_path);state=built[1].cpu_s4_4a_state
    original=state._cpu_s4_4a_source_version
    state._cpu_s4_4a_ensure_cached(state._g_linear_function)
    state._cpu_s4_4a_source_version='unknown-source'
    with pytest.raises(SourceBindingError,match='SOURCE_VERSION_OR_ALGORITHM'):
        replace(built[1])
    state._cpu_s4_4a_source_version=original
    assert replace(built[1])['owner_identity']==f'state:{id(state)}'

def test_actual_factory_pending_refused_then_cleanup_and_restore(tmp_path):
    built=old_role(tmp_path);state=built[1].cpu_s4_4a_state;cache=state._cpu_s4_4a_cache
    args=dict(omega=np.ones(32),kperp2=np.ones((17,19)),geometry=(32,17,19),work_dtype=np.complex128,output_dtype=np.complex128,k0=1.,omega0=2.,beta2=.1,denom_floor=1e-4,dz=.25,precision_strategy='baseline_complex64',norm_strategy='baseline_complex64',backend=np,source_version=state._cpu_s4_4a_source_version,allow_toy_backend=True)
    cache.begin_half(**args)
    assert cache.snapshot()['stats']['factor_requests']==0
    with pytest.raises(SourceBindingError,match='PENDING_HALF'):
        replace(built[1])
    cache.finish_half(success=False)
    assert replace(built[1])['snapshot']['stats']['entries']==0

def test_full_shape_forbidden_in_actual_local_factory(tmp_path):
    with pytest.raises(Exception,match='FULLSIZE_SHAPE_FORBIDDEN'):
        build_role('PURE_OFF_FULLSIZE',capture_root=tmp_path,local_engineering=True,pool_override=1,shape=contract.FULL_SHAPE)
    assert len(build_role('PURE_OFF_FULLSIZE',capture_root=tmp_path/'restored',local_engineering=True,pool_override=1,shape=contract.MICRO_SHAPE_32))==6
