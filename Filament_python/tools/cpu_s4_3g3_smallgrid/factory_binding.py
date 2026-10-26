"""Private construction globals for the frozen G factory; no shared mutation."""
from __future__ import annotations

import copy
import os
from types import CodeType, FunctionType

from tools.hpc_ops import provenance_v2 as pv


def _local_resource_code(code: CodeType, changes: list) -> CodeType:
    constants = []
    for value in code.co_consts:
        if isinstance(value, CodeType):
            constants.append(_local_resource_code(value, changes))
        elif type(value) is int and value in (24, 48):
            changes.append(dict(function=code.co_qualname, first_line=code.co_firstlineno, before=value, after=1))
            constants.append(1)
        else:
            constants.append(value)
    return code.replace(co_consts=tuple(constants))


def private_factory(legacy, *, owned_identity, lifecycle_builder=None, local_engineering=False, candidate_schema=None):
    """Clone construction helpers only; numerical modules and bodies stay frozen.

    The only local resource is one worker, already admitted by frozen E/R8.
    The old G module contains no numerical bodies: its 24/48 integer constants
    describe resource defaults, checks and metadata.  The exact changes are
    exposed on the returned factory and are never used inside Slurm.
    """
    if local_engineering and os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("LOCAL_ENGINEERING_FORBIDDEN_IN_SLURM")
    namespace = dict(vars(legacy))
    namespace["_MODULE_CACHE"] = {}
    if candidate_schema is not None:
        namespace["CANDIDATE_SCHEMA"] = str(candidate_schema)
    changes = []
    for name, value in vars(legacy).items():
        if isinstance(value, FunctionType) and value.__globals__ is vars(legacy):
            code = _local_resource_code(value.__code__, changes) if local_engineering else value.__code__
            clone = FunctionType(code, namespace, value.__name__, value.__defaults__, value.__closure__)
            clone.__kwdefaults__ = value.__kwdefaults__
            clone.__annotations__ = value.__annotations__
            clone.__module__ = value.__module__
            clone.__qualname__ = value.__qualname__
            namespace[name] = clone
    old_own = legacy._own_record()
    identity = copy.deepcopy(owned_identity)
    token = pv.raw_sha256((str(identity["composite_raw_sha256"]) + (":LOCAL1" if local_engineering else ":HPC24_48")).encode("ascii"))

    def own_record():
        record = copy.deepcopy(old_own)
        record["frozen_factory_dependency"] = copy.deepcopy(old_own)
        record["new_owned_sources"] = copy.deepcopy(identity)
        record["implementation_sha256"] = token
        record["construction_scope"] = "LOCAL_ENGINEERING_ONLY_FIXED_1" if local_engineering else "HPC_FIXED_24_48"
        return record

    namespace["_own_record"] = own_record
    if lifecycle_builder is not None:
        namespace["make_controlled_state_class"] = lifecycle_builder
    settings = dict(kernel_threads=1, temporal_workers=1, spatial_workers=1, raman_workers=1) if local_engineering else dict(kernel_threads=24, temporal_workers=48, spatial_workers=24, raman_workers=48)
    factory = namespace["make_factory"]("G_CONTROLLED_INPUT", block_size=32, **settings)
    factory.smallgrid_construction = dict(local_engineering=bool(local_engineering), resource_constants=changes, role_cache_identity=token, shared_module_mutations=0, settings=settings)
    return factory
