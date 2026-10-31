"""Admission and actual microgrid checks, using the repository test wrapper."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from .entry import admission, input_domain, read
from .protocol import MICRO_SHAPE, SHAPE


def test_s46d_entry_domain_refusals(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    admission(local_engineering=True, shape=MICRO_SHAPE, intervals=2, mode='QUALIFY_BOUNDARY')
    with pytest.raises(RuntimeError, match='LOCAL_NT384'):
        admission(local_engineering=True, shape=SHAPE, intervals=2, mode='QUALIFY_BOUNDARY')
    with pytest.raises(RuntimeError, match='ACTUAL_SLURM'):
        admission(local_engineering=False, shape=SHAPE, intervals=8, mode='PURE_OFF_SMALLGRID')
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    admission(local_engineering=False, shape=SHAPE, intervals=2, mode='QUALIFY_BOUNDARY')
    with pytest.raises(RuntimeError, match='TWO_QUAL'):
        admission(local_engineering=False, shape=SHAPE, intervals=2, mode='PURE_OFF_SMALLGRID')
    with pytest.raises(RuntimeError, match='LOCAL_NT384'):
        admission(local_engineering=True, shape=MICRO_SHAPE, intervals=8, mode='PURE_OFF_SMALLGRID')
    input_domain(dict(shape=list(MICRO_SHAPE)), local_engineering=True)
    with pytest.raises(RuntimeError, match='MANIFEST_SHAPE'):
        input_domain(dict(shape=list(SHAPE)), local_engineering=True)
    with pytest.raises(RuntimeError, match='MANIFEST_SHAPE'):
        input_domain(dict(shape=list(MICRO_SHAPE)), local_engineering=False)


def test_s46d_application_harness_strip_proof():
    import ast, copy, inspect, textwrap
    from .measurement import _Add, _Strip
    from tools.cpu_s4_3g3_smallgrid import benchmark as sg
    tree = ast.parse(textwrap.dedent(inspect.getsource(sg._run_window)))
    add = _Add()
    transformed = add.visit(copy.deepcopy(tree))
    assert add.close_anchors == 1
    assert ast.dump(_Strip().visit(transformed), include_attributes=False) == ast.dump(tree, include_attributes=False)


@pytest.mark.skipif(not os.environ.get('S46D_REAL_ENTRY_REPORT'), reason='Explicit clean-source micro entry phase only')
def test_s46d_real_entry_micro():
    """Fresh processes R2/B2/R8/B8; no W48 performance qualification."""
    from tools.cpu_s4_3g3_smallgrid import benchmark as sg
    report = Path(os.environ['S46D_REAL_ENTRY_REPORT']).resolve(strict=True)
    sg.input_manifest(report, shape=MICRO_SHAPE)
    results = {}
    for intervals in (2, 8):
        for role in ('R', 'B'):
            case = f'LOCAL_{role}{intervals}'
            folder = report / 'local' / case
            folder.mkdir(parents=True, exist_ok=False)
            argv = [sys.executable, '-s', '-B', '-m', 'tools.cpu_s4_6d_eq10_propagation.entry', '--report', str(report), '--work', str(folder / 'work'), '--role', role, '--intervals', str(intervals), '--local-engineering']
            if role == 'B':
                reference = results[f'LOCAL_R{intervals}']
                argv.extend(('--reference-output', reference['output']))
                if intervals == 2:
                    argv.extend(('--reference-boundary', reference['boundary_path'], '--reference-stages', reference['stages']))
            with (folder / 'stdout.log').open('xb') as stdout, (folder / 'stderr.log').open('xb') as stderr:
                completed = subprocess.run(argv, stdout=stdout, stderr=stderr, timeout=600)
            assert completed.returncode == 0, str(folder / 'stderr.log')
            result = read(folder / 'work/CASE_RECEIPT.json')
            assert result['route']['nH'] == 2*intervals
            assert result['eq10_pool']['closed'] is True
            assert result['profile_enabled'] is False
            if role == 'B':
                assert result['comparisons']['terminal']['status'] == 'PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
                if intervals == 2:
                    assert result['comparisons']['boundary']['status'] == 'PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT'
                    assert result['comparisons']['stages']['status'] == 'PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
            results[case] = result
    sg.write(report / 'LOCAL_ENTRY_ACCEPTANCE.json', dict(status='PASS_FOUR_FRESH_PROCESS_MICRO_WINDOWS', cases=list(results), shape=list(MICRO_SHAPE), workers=1, performance_qualified=False, scientific_qualification=False))
