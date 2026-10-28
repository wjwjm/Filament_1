"""Source-bound L1 and the single Eq10 two-materialization candidate."""
from __future__ import annotations
import ast
import copy
import os
from pathlib import Path

from tools.hpc_ops import provenance_v2 as pv
from tools.cpu_s4_5a_l1_residual import instrument as old_ast
from tools.cpu_s4_5b_steady_cost.instrument import seal_callable,verify_callable_seal
from .adapter import BodyClock,attach,require,dump

ROLES=('L1_REF_5C','C_EQ10_V1')

def owned_identity():
    root=Path(__file__).resolve().parent;repo=root.parents[2];files={}
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix not in ('.py','.md'):continue
        relative=path.relative_to(repo).as_posix();raw=path.read_bytes()
        oid=pv._git_blob_oid(repo,relative);blob=pv._read_git_blob(repo,oid)
        require(pv.canonical_lf_bytes(raw)==pv.canonical_lf_bytes(blob),'C_OWNED_COMMITTED_CANONICAL:'+relative)
        files[path.name]=dict(path=str(path),relative_path=relative,size_bytes=len(raw),raw_sha256=pv.raw_sha256(raw),
            classification='tracked_text',hash_scope=pv.TRACKED_HASH_SCOPE,git_blob_oid=oid,canonical_lf_sha256=pv.canonical_lf_sha256(blob),
            raw_byte_observation='Checkout only; canonical identity is authoritative across platforms')
    require('kernels.py' in files and 'candidate.py' in files,'C_COMPLETE_OWNED_NAMESPACE')
    return dict(schema='filament.cpu_s4_5c.owned_sources.v1',files=files,producer_module_identity='tools.cpu_s4_5c_eq10_pointwise')

class _Kernels(ast.NodeTransformer):
    def __init__(self):self.count=0
    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and t.id=='derivative_increment_norm' for t in node.targets):
            require(isinstance(node.value,ast.BinOp) and isinstance(node.value.op,ast.Div),'C_EQ10_ORIGINAL_DIVISION')
            real=node.value.left
            require(isinstance(real,ast.Attribute) and real.attr=='real' and isinstance(real.value,ast.Call),'C_EQ10_ORIGINAL_REAL_IFFT')
            product=real.value.args[0]
            require(isinstance(product,ast.BinOp) and isinstance(product.op,ast.Mult),'C_EQ10_ORIGINAL_COMPLEX_PRODUCT')
            real.value.args[0]=ast.copy_location(ast.Call(func=ast.Name(id='__cpu_s45c_k1__',ctx=ast.Load()),args=[product.left,product.right],keywords=[]),product)
            node.value=ast.copy_location(ast.Call(func=ast.Name(id='__cpu_s45c_k2__',ctx=ast.Load()),args=[real,node.value.right],keywords=[]),node.value)
            self.count+=1
        return node

class _Undo(ast.NodeTransformer):
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id in ('__cpu_s45c_k1__','__cpu_s45c_k2__'):
            require(len(node.args)==2 and not node.keywords,'C_EXACT_KERNEL_CALL_SHAPE')
            return ast.copy_location(ast.BinOp(left=node.args[0],op=ast.Mult() if node.func.id.endswith('k1__') else ast.Div(),right=node.args[1]),node)
        return node

def install_stage(state,executor,body_clock):
    original=state._internal_stage_inner;apply=state._internal_apply_inner
    lineage=old_ast._verify_uninstrumented_r8_pair(state,original,apply)
    path,node,expected=old_ast._expected_transformed_node(name='isaacs_raman_stage',transform_name='_transform_stage',candidate_namespace=state._candidate_namespace)
    before=copy.deepcopy(node);transform=_Kernels();transform.visit(node)
    require(transform.count==1 and dump(_Undo().visit(copy.deepcopy(node)))==dump(before),'C_ONLY_TWO_REVERSIBLE_MATERIALIZATIONS')
    namespace=dict(original.__globals__);records=[]
    def k1(coeff,spectrum):
        start=len(executor._dispatches)
        result=executor.k1(coeff,spectrum)
        executor._dispatches[start]['interval_index']=body_clock.active
        return result
    def k2(real_view,scale):
        start=len(executor._dispatches)
        result=executor.k2(real_view,scale)
        executor._dispatches[start]['interval_index']=body_clock.active
        return result
    namespace.update(__cpu_s45c_k1__=k1,__cpu_s45c_k2__=k2)
    stage=old_ast._compile_function_node(node,namespace=namespace,source_path=path)
    state._internal_stage_inner=stage
    state._cpu_s45c_original_stage=original
    state._cpu_s45c_seals={name:seal_callable(fn) for name,fn in dict(stage=stage,k1=k1,k2=k2,original=original,apply=apply).items()}
    state._cpu_s45c_slots=dict(__cpu_s45c_k1__=k1,__cpu_s45c_k2__=k2)
    return dict(lineage=lineage,original_stage_ast_sha256=pv.raw_sha256(dump(before).encode()),
                candidate_stage_ast_sha256=pv.raw_sha256(dump(node).encode()),undo_AST_exact=True,
                replaced_materializations=2,FFT_IFFT_sum_expressions_unchanged=True,
                input_order='Original coefficient argument, original FFT argument, K1, original IFFT.real argument, original current scale, K2')

def verify_bindings(dispatch):
    state=dispatch.cpu_s4_3g_state
    for name,seal in getattr(state,'_cpu_s45c_seals',{}).items():verify_callable_seal(seal)
    if getattr(dispatch,'cpu_s45c_executor',None) is not None:
        require(state._internal_stage_inner is state._cpu_s45c_seals['stage']['function'],'C_STAGE_SLOT_CHANGED')
        for name,fn in state._cpu_s45c_slots.items():require(state._internal_stage_inner.__globals__.get(name) is fn,'C_KERNEL_SLOT_CHANGED')
    return dict(status='PASS_C_SOURCE_BOUND_PRIVATE_STAGE',role=dispatch.cpu_s45c_role,
        candidate_active=dispatch.cpu_s45c_executor is not None,old_owned_guards='RETAINED',new_math_scope='Only K1/K2',default='I0')

def build_role(mode,*,role,capture_root,local_engineering=False,pool_override=None,fullsize=False):
    require(role in ROLES and mode in ('QUALIFY_BOUNDARY','OFF','PURE_OFF_SMALLGRID'),'C_ROLE_MODE')
    from tools.cpu_s4_4a_linear_cache import candidate as l1
    import numba
    pool=1 if local_engineering else 48;outer=1 if local_engineering else 24
    require(numba.config.NUMBA_NUM_THREADS==pool and (not local_engineering or pool_override==1),'C_FIXED_ACTUAL_POOL')
    numba.set_num_threads(outer)
    if fullsize:
        from tools.cpu_s4_4b_fullsize_linear_cache import candidate as full
        factory=full.make_factory(local_engineering=False)
    else:
        factory=l1.make_factory(kernel_threads=1 if local_engineering else 24,temporal_workers=pool,
            spatial_workers=1 if local_engineering else 24,raman_workers=pool,local_engineering=local_engineering)
    prop,dispatch=l1._bind_factory(factory)
    if fullsize:
        dispatch.cpu_s4_4a_cache_snapshot=dispatch.cpu_s4_4b_cache_snapshot
    state=dispatch.cpu_s4_3g_state;clock=BodyClock();executor=None;stage_binding=None
    if role=='C_EQ10_V1':
        from .kernels import KernelExecutor
        executor=KernelExecutor(workers=pool,role=role)
        stage_binding=install_stage(state,executor,clock)
    dispatch.cpu_s45c_executor=executor;dispatch.cpu_s45c_role=role;dispatch.cpu_s45c_body_clock=clock
    own=owned_identity()
    wrapped,actual,legacy_clock,binding,audit,lease=attach(prop,dispatch,role=role,mode=mode,capture_root=capture_root,owned_identity=own,body_clock=clock)
    binding=dict(binding,s45c_stage_binding=stage_binding,role=role,actual_pool=pool,profile_enabled=False,
        fixed_cache_cap_bytes=1610612736 if fullsize else 268435456)
    dispatch.cpu_s45c_binding=binding
    verify_bindings(dispatch)
    return wrapped,actual,legacy_clock,binding,audit,lease
