"""Original SG application clock with a source-bound pre-output pool join."""
from __future__ import annotations

import ast
import copy
import inspect
import marshal
import textwrap
import time
from types import FunctionType

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.cpu_s4_5a_l1_residual import measurement as parent
from tools.hpc_ops import provenance_v2 as pv


class _Add(parent._Add):
    def __init__(self):
        super().__init__()
        self.close_anchors = 0

    def visit_Assign(self, node):
        result = super().visit_Assign(node)
        if ast.unparse(node.targets[0]) == 'ordinary':
            self.close_anchors += 1
            close = ast.Expr(value=ast.Call(func=ast.Name(id='_s46d_close_before_output', ctx=ast.Load()), args=[], keywords=[]))
            return [ast.copy_location(close, node), result]
        return result


class _Strip(parent._Strip):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == '_s46d_close_before_output':
            return None
        return super().visit_Expr(node)


def run_window(args, built, *, timer, application_token, **kwargs):
    original = sg._run_window
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    before = parent._dump(tree)
    adder = _Add()
    transformed = ast.fix_missing_locations(adder.visit(copy.deepcopy(tree)))
    parent.require(adder.close_anchors == 1 and adder.counts == dict(INPUT_PREPARATION=1, PROPAGATION=1, DIAGNOSTICS=1, OUTPUT=1), 'S46D_EXACT_APPLICATION_ANCHORS')
    parent.require(parent._dump(_Strip().visit(copy.deepcopy(transformed))) == before, 'S46D_REMOVING_TIMER_AND_CLOSE_RESTORES_HARNESS_AST')
    close = built[1].close_eq10_pool
    lifecycle = {}

    def close_before_output():
        lifecycle['close_started_s'] = time.perf_counter()
        snapshot = close()
        lifecycle['join_completed_s'] = time.perf_counter()
        parent.require(snapshot['closed'] and snapshot['close_error'] is None, 'S46D_POOL_JOIN_BEFORE_NORMAL_OUTPUT')

    namespace = dict(original.__globals__, _s45a_timer=timer,
                     _s45a_application_token=application_token,
                     _s46d_close_before_output=close_before_output)
    for name in ('context', 'fingerprints'):
        frozen = namespace[name]
        def forward(*values, _frozen=frozen, **named):
            with timer.scope('INPUT_PREPARATION'):
                return _frozen(*values, **named)
        namespace[name] = forward
    local = {}
    exec(compile(transformed, original.__code__.co_filename, 'exec'), namespace, local)
    inner = local['_run_window']
    namespace['_run_window'] = inner
    wrapped = FunctionType(sg.run_window.__code__, namespace, sg.run_window.__name__, sg.run_window.__defaults__, sg.run_window.__closure__)
    wrapped.__kwdefaults__ = sg.run_window.__kwdefaults__
    try:
        result = wrapped(args, built, **kwargs)
    except BaseException as error:
        try:
            close()
        except BaseException as cleanup:
            error.add_note('S46D_CLOSE_FAILURE:' + type(cleanup).__name__)
        raise
    finally:
        # On success the call is idempotent. On failure the except path already
        # waits for all work, without replacing the original exception.
        if lifecycle.get('join_completed_s') is not None:
            close()
    end = kwargs['application_started'] + result['optical_application_window_s']
    parent.require(lifecycle['join_completed_s'] < end, 'S46D_CLOSE_INSIDE_ORIGINAL_APPLICATION_CLOCK')
    binding = dict(schema='filament.cpu_s4_6d.harness.v1', source_file=original.__code__.co_filename,
                   source_canonical_lf_sha256=pv.canonical_lf_sha256(source.encode()),
                   original_ast_sha256=pv.raw_sha256(before.encode()),
                   actual_code_sha256=pv.raw_sha256(marshal.dumps(inner.__code__)),
                   stripped_AST_exact=True, close_anchor='before original ordinary=writer(...)',
                   lifecycle=lifecycle, original_application_finished_s=end,
                   scientific_expression_changes=0, propagate_AST_rebuilt=False)
    sg.write(args.work / 'MEASUREMENT_HARNESS_BINDING.json', binding)
    return result
