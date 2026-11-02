"""Attach the existing original-grid L1 cache to the combined factory."""
import copy

from tools.cpu_s4_4b_fullsize_linear_cache import candidate as cache_candidate
from tools.cpu_s4_4b_fullsize_linear_cache import contract as cache_contract

from . import combined_candidate as joint
from .protocol import CANDIDATE, MICRO_SHAPE, SHAPE, candidate_spec


def build_role(semantic, mode, *, capture_root, role, local_engineering,
               recorder_pair, shape=None, pool_override=None):
    candidate_spec(semantic)
    expected = MICRO_SHAPE if local_engineering else SHAPE
    if shape is not None and tuple(shape) != expected:
        raise ValueError("S46H_GRID_DOMAIN")
    if role not in ("R", "X"):
        raise ValueError("S46H_ROLE_DOMAIN")
    cache_contract.validate_shape(expected, scope=(cache_contract.LOCAL_ENGINEERING_SCOPE
                                   if local_engineering else cache_contract.FULLSIZE_SCOPE))
    built = joint.build_role(
        "PURE_OFF_SMALLGRID" if mode == "PURE_OFF_FULLSIZE" else mode,
        capture_root=capture_root, role=joint.ROLE_REFERENCE if role == "R" else joint.ROLE_CANDIDATE,
        local_engineering=local_engineering, recorder_pair=recorder_pair,
        pool_override=pool_override)
    try:
        cache = cache_candidate._replace_empty_cache(built[1],
                    local_engineering=local_engineering, configured_shape=expected)
        built[1].cpu_s4_6h_shape = expected
        binding = copy.deepcopy(built[3])
        binding.update(semantic_id=CANDIDATE, shape=list(expected), fullsize_cache_binding=cache)
        return built[:3] + (binding,) + built[4:]
    except BaseException:
        joint.finalize_factory(built[1], reason="CACHE_CONSTRUCTION_FAILURE")
        raise


def verify_role(dispatch, semantic=CANDIDATE):
    candidate_spec(semantic)
    result = joint.verify_bindings(dispatch)
    cache = dispatch.cpu_s4_4b_cache_snapshot()
    if tuple(cache.get("configured_shape", ())) != tuple(dispatch.cpu_s4_6h_shape):
        raise RuntimeError("S46H_CACHE_SHAPE_BINDING_CHANGED")
    return dict(combined=result, cache=cache, semantic_id=CANDIDATE)


def close_role(dispatch, semantic=CANDIDATE):
    candidate_spec(semantic)
    return joint.finalize_factory(dispatch)
