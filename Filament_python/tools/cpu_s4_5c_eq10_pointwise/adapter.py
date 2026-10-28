"""One common body clock, inserted before the unchanged G2 guard is captured."""
from __future__ import annotations
import ast
import builtins
import contextlib
import copy
import time
from pathlib import Path
from types import SimpleNamespace

from tools.hpc_ops import provenance_v2 as pv
from tools.cpu_s4_5b_steady_cost.adapter import clone_g2_namespace

def require(ok,why):
    if not ok:raise RuntimeError(why)

def dump(node):return ast.dump(node,include_attributes=False)

class BodyClock:
    def __init__(self):self.rows=[];self.active=None
    @contextlib.contextmanager
    def body(self,index):
        require(self.active is None and len(self.rows)<8,'C_BODY_CLOCK_CAPACITY_OR_REENTRY')
        index=int(index);require(index==len(self.rows),'C_BODY_SOURCE_ORDINAL')
        self.active=index
        wall=time.perf_counter_ns();cpu=time.process_time_ns();failed=True
        try:
            yield
            failed=False
        finally:
            endcpu=time.process_time_ns();endwall=time.perf_counter_ns()
            self.rows.append(dict(interval_index=index,ordinal=index+1,attempt=1,
                start_wall_ns=wall,end_wall_ns=endwall,start_cpu_ns=cpu,end_cpu_ns=endcpu,
                wall_ns=endwall-wall,cpu_ns=endcpu-cpu,failed=failed))
            self.active=None
    def snapshot(self):
        return dict(schema='filament.cpu_s4_5c.complete_body_clock.v1',rows=copy.deepcopy(self.rows),
            first_body_s=sum(row['wall_ns'] for row in self.rows if row['ordinal']==1)*1e-9,
            later_body_s=sum(row['wall_ns'] for row in self.rows if row['ordinal']>=2)*1e-9,
            body_contract='Identical original for-body first statement through original bottom, including save/progress/control',
            fine_profile_enabled=False,empty_scope_subtracted=False)

class _Strip(ast.NodeTransformer):
    def visit_With(self,node):
        if len(node.items)==1 and isinstance(node.items[0].context_expr,ast.Call):
            call=node.items[0].context_expr
            if isinstance(call.func,ast.Name) and call.func.id=='__cpu_s45c_body__':
                return [self.visit(item) for item in node.body]
        return self.generic_visit(node)

def attach(raw_propagator,dispatch,*,role,mode,capture_root,owned_identity,body_clock):
    from tools.cpu_s4_checkpoint_support import load_g2_adapter,g2_adapter_path
    from tools.cpu_s4_3g3_smallgrid.scientific_context import frozen_tools
    repo=Path(__file__).resolve().parents[3];frozen_tools(repo)
    original=load_g2_adapter()
    require(Path(original.__file__).resolve()==g2_adapter_path().resolve(strict=True),'C_ACTUAL_FROZEN_G2_MODULE')
    namespace=clone_g2_namespace(original)
    common=SimpleNamespace(**vars(original._common))
    common.factory=lambda unused_role:(lambda unused_base:(raw_propagator,dispatch))
    namespace['_common']=common
    old_record=namespace['_frozen_tracked_record']
    owned={row['relative_path']:row for row in owned_identity['files'].values()}
    def tracked_record(relative):
        if relative not in owned:return old_record(relative)
        record=owned[relative]
        return dict(path=relative,**{key:record[key] for key in ('classification','hash_scope','git_blob_oid','canonical_lf_sha256')})
    namespace['_frozen_tracked_record']=tracked_record
    receipts=[]
    def make_interceptor(*,role,mode,bindings):
        injected=False
        def compile_interceptor(source,filename,compile_mode,*args,**kwargs):
            nonlocal injected
            if not injected and isinstance(source,ast.Module):
                baseline=copy.deepcopy(source)
                timing=namespace['_timing']._inject_timing(source,role=role)
                boundary=namespace['_insert_boundary'](source,bindings) if mode=='QUALIFY_BOUNDARY' else None
                nodes=[node for node in source.body if isinstance(node,ast.FunctionDef) and node.name=='propagate_one_pulse']
                require(len(nodes)==1,'C_PROP_FUNCTION_ANCHOR')
                loops=[node for node in ast.walk(nodes[0]) if isinstance(node,ast.For) and isinstance(node.target,ast.Tuple)
                       and [getattr(t,'id',None) for t in node.target.elts]==['interval_index','interval']]
                require(len(loops)==1 and len(loops[0].body)>10,'C_COMPLETE_ORIGINAL_BODY_ANCHOR')
                loop=loops[0];loop.body=[ast.copy_location(ast.With(items=[ast.withitem(context_expr=ast.Call(
                    func=ast.Name(id='__cpu_s45c_body__',ctx=ast.Load()),args=[ast.Name(id='interval_index',ctx=ast.Load())],keywords=[]))],body=loop.body),loop.body[0])]
                stripped=_Strip().visit(copy.deepcopy(source))
                stripped=namespace['_structure_without_timing_ast'](stripped)
                if mode=='QUALIFY_BOUNDARY':
                    stripped,count=namespace['_strip_boundary'](stripped);require(count==1,'C_ORIGINAL_BOUNDARY_STRIP')
                require(dump(stripped)==dump(baseline),'C_BODY_STRIP_RESTORES_ORIGINAL_G2_AST')
                ast.fix_missing_locations(source)
                receipts.append(dict(baseline_ast_sha256=pv.raw_sha256(dump(baseline).encode()),
                    stripped_ast_sha256=pv.raw_sha256(dump(stripped).encode()),timing=timing,boundary=boundary,
                    common_clock_types=1,old_guard_capture='AFTER_COMPILE_INSERTION',scientific_expressions_changed=0))
                compile_interceptor.static_receipt=copy.deepcopy(receipts[-1])
                injected=True
            return builtins.compile(source,filename,compile_mode,*args,**kwargs)
        compile_interceptor.static_receipt=None
        return compile_interceptor
    namespace['_make_compile_interceptor']=make_interceptor
    original_clone=namespace['_clone_g']
    helper=body_clock.body
    def clone(*args,helpers,**kwargs):
        extended=dict(helpers);extended['__cpu_s45c_body__']=helper
        return original_clone(*args,helpers=extended,**kwargs)
    namespace['_clone_g']=clone
    result=namespace['build_role']('G_CONTROLLED_INPUT','QUALIFY_BOUNDARY' if mode=='QUALIFY_BOUNDARY' else 'PURE_OFF_REAL8',capture_root=Path(capture_root).resolve())
    require(result[1] is dispatch and len(receipts)==1,'C_ACTUAL_G2_DISPATCH_AND_CLOCK_CLONE')
    result[3]['s45c_body_clock']=receipts[0]
    result[3]['s45c_reported_role']=role
    return result
