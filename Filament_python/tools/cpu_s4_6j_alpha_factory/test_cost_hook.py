import ast
import copy
from pathlib import Path
import pytest
from tools.cpu_s4_6j_alpha_factory.cost_hook import instrument_alpha_cost, strip_alpha_cost, ALPHA_HELPER
from tools.cpu_s4_6j_alpha_factory.reverse_ast_proof import prove, _normalize_source_function, validate_static_inventory, SOURCE_FILES, ProofError

def _source():
    path = Path(__file__).resolve().parents[2]/'KHz_filament/propagate.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    return next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='propagate_one_pulse')

def test_alpha_scope_restores_original_four_statements():
    original = _source()
    changed = copy.deepcopy(original)
    receipt = instrument_alpha_cost(changed)
    assert receipt['mathematical_changes']==0
    assert ast.dump(strip_alpha_cost(changed),include_attributes=False)==ast.dump(original,include_attributes=False)
    assert sum(isinstance(node,ast.Name) and node.id==ALPHA_HELPER for node in ast.walk(changed))==1
    compile(ast.Module(body=[changed], type_ignores=[]), '<alpha-cost-compile-test>', 'exec')
    with pytest.raises(RuntimeError,match='FOUR_STATEMENTS_REQUIRED'):
        broken = _source()
        for node in ast.walk(broken):
            if isinstance(node,ast.Call) and getattr(node,'lineno',None)==830:
                node.keywords[0].value=ast.Constant(1.0)
        instrument_alpha_cost(broken)

def test_factory_timing_reverse_ast_proof():
    repo = Path(__file__).resolve().parents[3]
    assert prove(repo=repo)['status']=='PASS'

def test_missing_wrapper_cannot_hide_behind_inverse_proof():
    repo = Path(__file__).resolve().parents[3]
    tree = ast.parse((repo/SOURCE_FILES[0]).read_text(encoding='utf-8'))
    target = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id=='factory_scope')
    target.func.id = 'removed_scope'
    with pytest.raises(ProofError, match='STATIC_SCOPE_INVENTORY'):
        validate_static_inventory(tree, SOURCE_FILES[0])

def test_reverse_normalization_does_not_mask_changed_return():
    source = '''def _source_function(path, name):
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError):
        raise
    return "changed"
'''
    current = ast.parse(source)
    _normalize_source_function(current, ast.parse('pass'))
    assert isinstance(current.body[0].body[-1].value, ast.Constant)
    assert current.body[0].body[-1].value.value=='changed'
