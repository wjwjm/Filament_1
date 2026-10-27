"""Source-bound timing statements around the unchanged SG application harness."""
from __future__ import annotations
import ast
import copy
import inspect
import marshal
import textwrap
from types import FunctionType
from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.hpc_ops import provenance_v2 as pv

def require(ok,why):
    if not ok:raise RuntimeError(why)

def _dump(node):return ast.dump(node,include_attributes=False)

class _Add(ast.NodeTransformer):
    def __init__(self):self.counts={}
    def visit_Assign(self,node):
        target=ast.unparse(node.targets[0]);label=None
        if target=='(field, kwargs, view)':label='INPUT_PREPARATION'
        elif target=='result' and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='prop':label='PROPAGATION'
        elif target=='bundle':label='DIAGNOSTICS'
        elif target=='ordinary':label='OUTPUT'
        if label:
            self.counts[label]=self.counts.get(label,0)+1
            return ast.copy_location(ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Attribute(value=ast.Name(id='_s45a_timer',ctx=ast.Load()),attr='scope',ctx=ast.Load()),args=[ast.Constant(label)],keywords=[]))],body=[node]),node)
        if target=='application_finished':
            return [ast.copy_location(ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='_s45a_timer',ctx=ast.Load()),attr='end',ctx=ast.Load()),args=[ast.Name(id='_s45a_application_token',ctx=ast.Load())],keywords=[])),node),node]
        return node

class _Strip(ast.NodeTransformer):
    def visit_With(self,node):
        call=node.items[0].context_expr
        if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and call.func.value.id=='_s45a_timer':return self.visit(node.body[0])
        return self.generic_visit(node)
    def visit_Expr(self,node):
        call=node.value
        if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and call.func.value.id=='_s45a_timer' and call.func.attr=='end':return None
        return self.generic_visit(node)

def run_window(args,built,*,timer,application_token,**kwargs):
    original=sg._run_window
    source=textwrap.dedent(inspect.getsource(original));tree=ast.parse(source)
    before=_dump(tree);adder=_Add();transformed=adder.visit(copy.deepcopy(tree));ast.fix_missing_locations(transformed)
    require(adder.counts==dict(INPUT_PREPARATION=1,PROPAGATION=1,DIAGNOSTICS=1,OUTPUT=1),'MEASUREMENT_HARNESS_EXACT_ANCHORS')
    require(_dump(_Strip().visit(copy.deepcopy(transformed)))==before,'REMOVING_TIMERS_RESTORES_EXACT_HARNESS_AST')
    namespace=dict(original.__globals__);namespace['_s45a_timer']=timer;namespace['_s45a_application_token']=application_token
    # These wrappers forward existing required input work exactly once.
    for name in ('context','fingerprints'):
        frozen=namespace[name]
        def forward(*values,_frozen=frozen,**named):
            with timer.scope('INPUT_PREPARATION'):return _frozen(*values,**named)
        namespace[name]=forward
    local={};exec(compile(transformed,original.__code__.co_filename,'exec'),namespace,local)
    inner=local['_run_window'];namespace['_run_window']=inner
    binding=dict(schema='filament.cpu_s4_5a.harness_binding.v1',source_file=original.__code__.co_filename,
        source_canonical_lf_sha256=pv.canonical_lf_sha256(source.encode('utf-8')),original_ast_sha256=pv.raw_sha256(before.encode('utf-8')),
        timed_ast_sha256=pv.raw_sha256(_dump(transformed).encode('utf-8')),stripped_AST_exact=True,
        actual_code_sha256=pv.raw_sha256(marshal.dumps(inner.__code__)),anchors=adder.counts,
        scientific_expression_changes=0,propagate_AST_rebuilt=False)
    wrapped=FunctionType(sg.run_window.__code__,namespace,sg.run_window.__name__,sg.run_window.__defaults__,sg.run_window.__closure__)
    wrapped.__kwdefaults__=sg.run_window.__kwdefaults__
    result=wrapped(args,built,**kwargs)
    sg.write(args.work/'MEASUREMENT_HARNESS_BINDING.json',binding)
    return result
