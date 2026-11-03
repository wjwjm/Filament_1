"""Compose the unchanged power-volume hook with one alpha timing-only scope."""
from __future__ import annotations
import ast
import copy
import contextlib
from pathlib import Path
import numpy as np

from tools.cpu_s4_6i_residual_parallel.power_hook import PowerVolumeHook
from tools.cpu_s4_6i_residual_parallel.instrument import _iter_statement_lists
from tools.cpu_s4_6h_combined_parallel.instrument import _seal, verify_seal
from .factory_timing import source_identity

ALPHA_HELPER = '__cpu_s46j_alpha_cost_scope__'
ALPHA_SITE = 'S46J.alpha.propagate.py:830-833'
ALPHA_LABEL = 'NONLINEAR_ARRAYS'
_SOURCE = '''ion_source_prop = xp.nan_to_num(ion_source_prop, nan=0.0, posinf=0.0, neginf=0.0)
alpha_ion = ion_source_prop / (I + I_floor)
alpha_ion = xp.nan_to_num(alpha_ion, nan=0.0, posinf=0.0, neginf=0.0)
xp.maximum(alpha_ion, 0.0, out=alpha_ion)
'''

def _dump_block(body):
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)

def instrument_alpha_cost(node):
    before = copy.deepcopy(node)
    expected = _dump_block(ast.parse(_SOURCE).body)
    matches = []
    for body in _iter_statement_lists(node):
        for i, statement in enumerate(body):
            if getattr(statement, 'lineno', None) == 830 and len(body) >= i+4:
                block = body[i:i+4]
                if _dump_block(block) == expected:
                    matches.append((body, i))
    if len(matches) != 1:
        raise RuntimeError('S46J_ALPHA_FROZEN_FOUR_STATEMENTS_REQUIRED')
    body, start = matches[0]
    block = body[start:start+4]
    wrapper = ast.With(items=[ast.withitem(context_expr=ast.Call(
        func=ast.Name(id=ALPHA_HELPER, ctx=ast.Load()),
        args=[ast.Name(id='interval_index', ctx=ast.Load())], keywords=[]))],
        body=block, type_comment=None)
    ast.copy_location(wrapper, block[0])
    body[start:start+4] = [wrapper]
    if ast.dump(strip_alpha_cost(node), include_attributes=False) != ast.dump(before, include_attributes=False):
        raise RuntimeError('S46J_ALPHA_COST_INVERSE_MISMATCH')
    return dict(source_site=ALPHA_SITE, original_four_statements_preserved=True,
                mathematical_changes=0, assignments_and_lifetimes_preserved=True)

def strip_alpha_cost(node):
    class Strip(ast.NodeTransformer):
        def visit_With(self, current):
            self.generic_visit(current)
            if len(current.items) == 1:
                call = current.items[0].context_expr
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == ALPHA_HELPER:
                    if _dump_block(current.body) != _dump_block(ast.parse(_SOURCE).body):
                        raise RuntimeError('S46J_ALPHA_SCOPE_BODY_CHANGED')
                    if ast.dump(call, include_attributes=False) != ast.dump(ast.parse(
                        ALPHA_HELPER+'(interval_index)', mode='eval').body, include_attributes=False):
                        raise RuntimeError('S46J_ALPHA_SCOPE_ARGUMENTS_CHANGED')
                    return current.body
            return current
    return Strip().visit(copy.deepcopy(node))

class AlphaCostHook(PowerVolumeHook):
    """One private G2 hook; the fourth mathematical component remains frozen."""
    def __init__(self, pool, *, recorder, enabled, capture_store=None):
        self._alpha_recorder = recorder
        self._bound_alpha_recorder = recorder
        self._alpha_enabled = bool(enabled)
        self._bound_alpha_enabled = self._alpha_enabled
        self._alpha_sources = source_identity(extra_paths=(__file__,))
        self._alpha_backend = {name:getattr(np, name) for name in ('add','divide','nan_to_num','maximum')}
        self._alpha_recorder_methods = {name:getattr(recorder, name).__func__ for name in ('scope','context')}
        self._alpha_method_seals = {name:_seal(getattr(type(self), name)) for name in
            ('instrument','strip_generated_nodes','runtime_helpers','verify_binding','binding','_verify_recorder')}
        self._alpha_transform_seals = [_seal(instrument_alpha_cost), _seal(strip_alpha_cost)]
        owner = self
        def alpha_scope(interval_index):
            verify_seal(owner._alpha_scope_seal)
            owner._verify_recorder()
            if type(interval_index) is not int or not 0 <= interval_index < 8:
                raise RuntimeError('S46J_ALPHA_INTERVAL_DOMAIN')
            if not owner._alpha_enabled:
                return contextlib.nullcontext()
            return owner._alpha_recorder.scope(ALPHA_LABEL, common=True,
                context=owner._alpha_recorder.context(interval_index, ordinal=interval_index,
                    occurrence_class='PER_INTERVAL', source_site=ALPHA_SITE))
        self._alpha_scope = alpha_scope
        self._alpha_scope_seal = _seal(alpha_scope)
        super().__init__(pool, capture_store=capture_store)

    @property
    def recorder(self):
        return self._alpha_recorder

    def _verify_recorder(self):
        if (self._alpha_recorder is not self._bound_alpha_recorder or self._alpha_recorder.enabled is not False
                or self._alpha_enabled is not self._bound_alpha_enabled):
            raise RuntimeError('S46J_ALPHA_COST_RECORDER_IDENTITY_CHANGED')
        for name, function in self._alpha_recorder_methods.items():
            if getattr(self._alpha_recorder, name).__func__ is not function:
                raise RuntimeError('S46J_ALPHA_COST_RECORDER_CHANGED:'+name)

    def instrument(self, node):
        result = super().instrument(node)
        result = dict(result, s46j_alpha_cost=instrument_alpha_cost(node))
        return result

    def strip_generated_nodes(self, node):
        return super().strip_generated_nodes(strip_alpha_cost(node))

    def runtime_helpers(self):
        return dict(super().runtime_helpers(), **{ALPHA_HELPER:self._alpha_scope})

    def binding(self):
        return dict(super().binding(), alpha_timing_only=True, alpha_timing_enabled=self._alpha_enabled,
                    alpha_source_site=ALPHA_SITE, alpha_owned_sources=copy.deepcopy(self._alpha_sources))

    def verify_binding(self, active_globals=None):
        self._verify_recorder()
        result = super().verify_binding(active_globals)
        if source_identity(extra_paths=(__file__,)) != self._alpha_sources:
            raise RuntimeError('S46J_ALPHA_COST_SOURCE_CHANGED')
        for seal in (*self._alpha_method_seals.values(), *self._alpha_transform_seals, self._alpha_scope_seal):
            verify_seal(seal)
        for name, function in self._alpha_backend.items():
            if getattr(np, name) is not function:
                raise RuntimeError('S46J_ALPHA_COST_BACKEND_CHANGED:'+name)
        for name, function in self._alpha_recorder_methods.items():
            if getattr(self._alpha_recorder, name).__func__ is not function:
                raise RuntimeError('S46J_ALPHA_COST_RECORDER_CHANGED:'+name)
        if active_globals is not None and (active_globals.get(ALPHA_HELPER) is not self._alpha_scope
                                           or active_globals.get('xp') is not np):
            raise RuntimeError('S46J_ALPHA_COST_ACTIVE_HELPER_CHANGED')
        return dict(result, alpha_cost=self.binding())
