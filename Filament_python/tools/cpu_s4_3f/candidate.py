"""Frozen E handoff composed with D's exact contiguous five bodies."""
from __future__ import annotations
import copy
import importlib.util
import sys
from pathlib import Path
from tools.hpc_ops import provenance_v2 as pv

R8_RAW_SHA256='252fc1e6521812c29872d5b848e4082841f0b8ae0ad15661394f6b31347e0a32'
S43E_RAW_SHA256='a268bcfb74fbde603b0b3dbae26028e73a59a4022e3140bba9bae39a20e01e34'
S43D_RAW_SHA256='a42c02ec76e0d3f5e0b6d809d2696dd58d27ea6ce3fb95e784080b2b985f58ee'
R8_SIZE_BYTES,S43E_SIZE_BYTES,S43D_SIZE_BYTES=59808,49397,19012
CANDIDATE_SCHEMA='filament.cpu_s4_3f.internal_contiguous_handoff.v1'
TRANSFORMATION_IDENTITY='frozen_E_internal_handoff_plus_frozen_D_flat_five_bodies_v1'
TRAVERSAL_VARIANT='C_FLAT_EQWORK'
TRAVERSAL_TARGET_BODY_NAMES=('_product_body','_derivative_body','_rhs_body','_predictor_body','_corrector_body')
_MODULES={}

class SourceBindingError(RuntimeError):
    pass

def _default_s43e_path():
    sibling=Path(__file__).resolve().with_name('candidate_s43e.py')
    return sibling if sibling.is_file() else Path(__file__).resolve().parents[1]/'cpu_s4_3e/candidate.py'

def _default_s43d_path():
    sibling=Path(__file__).resolve().with_name('variants_s43d.py')
    return sibling if sibling.is_file() else Path(__file__).resolve().parents[1]/'cpu_s4_3d/variants.py'

def _default_r8_path():
    return Path(__file__).resolve().parents[1]/'cpu_s4_3c/candidate.py'

def _record(path,digest,size,label):
    path=Path(path).resolve(strict=True);actual=pv.raw_sha256_file(path)
    if path.stat().st_size!=size or actual!=digest:raise SourceBindingError(f'{label} source identity conflict: {path}; {actual}; {path.stat().st_size}')
    return dict(path=str(path),classification='external',hash_scope='raw_bytes',raw_sha256=actual,size_bytes=size)

def _own_identity():
    path=Path(__file__).resolve()
    return dict(path=str(path),classification='external',hash_scope='raw_bytes',raw_sha256=pv.raw_sha256_file(path),size_bytes=path.stat().st_size)

def _load(path,role):
    path=Path(path).resolve();key=(role,str(path))
    if key not in _MODULES:
        token=pv.raw_sha256((_own_identity()['raw_sha256']+'|'+str(path)).encode())[:20]
        name=f'_filament_cpu_s4f_{role}_{token}'
        spec=importlib.util.spec_from_file_location(name,path)
        if spec is None or spec.loader is None:raise SourceBindingError(f'import spec unavailable: {path}')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        try:spec.loader.exec_module(module)
        except BaseException:sys.modules.pop(name,None);raise
        _MODULES[key]=module
    return _MODULES[key]

def __getattr__(name):
    if name in ('CandidateUnsupported','HandoffRejected','StageBindingError'):
        path=_default_s43e_path();_record(path,S43E_RAW_SHA256,S43E_SIZE_BYTES,'E')
        return getattr(_load(path,'handoff'),name)
    raise AttributeError(name)

def _paths(frozen_r8_path=None,handoff_source_path=None,traversal_source_path=None,s43e_path=None,s43d_path=None):
    return (Path(frozen_r8_path or _default_r8_path()).resolve(),Path(handoff_source_path or s43e_path or _default_s43e_path()).resolve(),Path(traversal_source_path or s43d_path or _default_s43d_path()).resolve())

def traversal_identity(s43d_path=None):
    return dict(variant=TRAVERSAL_VARIANT,variant_source=_record(s43d_path or _default_s43d_path(),S43D_RAW_SHA256,S43D_SIZE_BYTES,'D'),block_size=32,
        body_names=list(TRAVERSAL_TARGET_BODY_NAMES),chunk_formula='Nt*32',derivative_axis_formula='multiplier[flat//pixels]',tail_rule='min(start+chunk,N)',full_field_reorder=False)

def source_identity(frozen_r8_path=None,*,handoff_source_path=None,traversal_source_path=None,s43e_path=None,s43d_path=None):
    r,e,d=_paths(frozen_r8_path,handoff_source_path,traversal_source_path,s43e_path,s43d_path)
    rr=_record(r,R8_RAW_SHA256,R8_SIZE_BYTES,'R8');er=_record(e,S43E_RAW_SHA256,S43E_SIZE_BYTES,'E');dr=_record(d,S43D_RAW_SHA256,S43D_SIZE_BYTES,'D')
    original=_load(e,'handoff').source_identity(r)
    return dict(original,schema=CANDIDATE_SCHEMA,transformation=TRANSFORMATION_IDENTITY,r8=rr,s43e_handoff_source=er,s43d_traversal_source=dr,
        composition_source=_own_identity(),traversal=traversal_identity(d),inherited_handoff_identity=original)

def _load_contiguous_r8(r8_path,s43d_path):
    r=_record(r8_path,R8_RAW_SHA256,R8_SIZE_BYTES,'R8');_record(s43d_path,S43D_RAW_SHA256,S43D_SIZE_BYTES,'D');d=_load(s43d_path,'traversal')
    token=pv.raw_sha256((_own_identity()['raw_sha256']+'|'+S43E_RAW_SHA256+'|'+S43D_RAW_SHA256+'|'+R8_RAW_SHA256).encode())
    module=d._load_r8(Path(r['path']),'S43F_CCONTIG',token,R8_RAW_SHA256)
    if not module.__dict__.get('__cpu_s4_3f_flat_installed__'):
        if module.__dict__.get('_NUMBA_BUNDLE') is not None:raise SourceBindingError('F bundle compiled before flat installation')
        d._install_flat_bodies(module)
        module.__dict__.update(__cpu_s4_3f_flat_installed__=True,__cpu_s4_3d_variant__=TRAVERSAL_VARIANT,__cpu_s4_3f_composition_token__=token)
    if not module.__name__.startswith('_filament_cpu_s4d_r8_s43f_ccontig_'):raise SourceBindingError('F cache namespace not isolated')
    for name in TRAVERSAL_TARGET_BODY_NAMES:
        body=module.__dict__.get(name)
        if not callable(body) or body.__globals__ is not module.__dict__ or token[:16] not in body.__name__:raise SourceBindingError(f'F flat body binding mismatch: {name}')
    return module

def make_factory(variant='CCONTIG',*,frozen_r8_path=None,handoff_source_path=None,traversal_source_path=None,s43e_path=None,s43d_path=None,**factory_args):
    normalized=str(variant).upper()
    if normalized not in ('CCONTIG','C_INTERNAL','C_STRICT'):raise ValueError(f'unknown F variant: {variant}')
    if int(factory_args.get('block_size',32))!=32:raise ValueError('F freezes contiguous task size at Nt*32')
    r,e,d=_paths(frozen_r8_path,handoff_source_path,traversal_source_path,s43e_path,s43d_path)
    identity=source_identity(r,handoff_source_path=e,traversal_source_path=d);module=_load_contiguous_r8(r,d);handoff=_load(e,'handoff')
    def private_r8_loader(path):
        checked=_record(path,R8_RAW_SHA256,R8_SIZE_BYTES,'R8')
        if Path(checked['path'])!=r:raise SourceBindingError('R8 requested path differs from factory binding')
        return module
    handoff.__dict__['_load_r8']=private_r8_loader
    args=dict(kernel_threads=24,temporal_workers=48,spatial_workers=24,raman_workers=48,block_size=32);args.update(factory_args)
    inherited=handoff.make_factory('C_STRICT' if normalized=='C_STRICT' else 'C_INTERNAL',frozen_r8_path=r,**args)
    def factory(frozen_propagator):
        prop,dispatch=inherited(frozen_propagator);state=dispatch.cpu_s4_3e_state
        def metrics():
            value=state.metrics()
            value.update(schema=CANDIDATE_SCHEMA,composition_variant='CCONTIG',source_identity=copy.deepcopy(identity),traversal=copy.deepcopy(identity['traversal']),private_r8_module=module.__name__,
                composition_token=module.__cpu_s4_3f_composition_token__,actual_body_names={name:module.__dict__[name].__name__ for name in TRAVERSAL_TARGET_BODY_NAMES},entry_copy_bytes=0,per_worker_scratch_bytes=0)
            return value
        dispatch.cpu_s4_3f_state=state;dispatch.cpu_s4_3f_full_heun=state.full_heun;dispatch.cpu_s4_3f_metrics=metrics;dispatch.cpu_s4_3f_source_identity=copy.deepcopy(identity)
        prop._cpu_s4_3f_state=state;prop._cpu_s4_3f_full_heun=state.full_heun
        for name,value in inherited.__dict__.items():
            if name not in ('source_identity','variant','candidate_schema','candidate_family'):setattr(factory,name,value)
        return prop,dispatch
    factory.__dict__.update(inherited.__dict__)
    factory.variant='C_STRICT' if normalized=='C_STRICT' else 'CCONTIG';factory.candidate_schema=CANDIDATE_SCHEMA;factory.candidate_family='RAMAN_TRUSTED_INTERNAL_CONTIGUOUS_V1'
    factory.source_identity=identity;factory.variant_module=module;factory.private_r8_module=module.__name__;factory.inherited_s43e_factory=inherited
    return factory

__all__=['make_factory','source_identity','traversal_identity','CANDIDATE_SCHEMA','SourceBindingError','CandidateUnsupported','HandoffRejected','StageBindingError']
