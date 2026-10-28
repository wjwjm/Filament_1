"""Real CPU micro windows, separate from GPU manager simulation and HPC timings."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from .inputs import seal


@pytest.mark.parametrize('shape,intervals', [((32,17,19),2), ((32,17,19),8), ((384,17,19),8)])
def test_s46a_accepted_l1_to_off_to_light(tmp_path, shape, intervals):
    report = Path(os.environ['S46A_REPORT_ROOT'])
    bundle = tmp_path / 'canonical'
    seal(bundle, os.environ['CPU_S4_DATA_ROOT'], shape)
    prior = None
    cases = []
    for mode in ('REFERENCE', 'OFF', 'LIGHT'):
        work = tmp_path / mode
        argv = [sys.executable, '-s', '-B', '-m', 'tools.cpu_s4_6a.entry', '--backend', 'CPU',
                '--report', str(report), '--bundle', str(bundle), '--work', str(work), '--case', mode,
                '--mode', mode, '--local', '--qualify', '--intervals', str(intervals)]
        if prior:
            argv += ['--reference-output', prior['output'], '--reference-boundary', prior['boundary_path']]
            if prior.get('stages'):
                argv += ['--reference-stages', prior['stages']]
        env = os.environ.copy(); env['NUMBA_CACHE_DIR'] = str(tmp_path / 'jit' / mode)
        value = subprocess.run(argv, env=env, capture_output=True, text=True, encoding='utf-8', timeout=480)
        (tmp_path / (mode + '.stdout.log')).write_text(value.stdout, encoding='utf-8')
        (tmp_path / (mode + '.stderr.log')).write_text(value.stderr, encoding='utf-8')
        assert value.returncode == 0, value.stderr
        receipt = json.loads((work / 'CASE_RECEIPT.json').read_text(encoding='utf-8'))
        assert receipt['shape'] == list(shape) and receipt['intervals'] == intervals
        assert receipt['route']['nH'] == intervals * 2 and receipt['route']['fallback'] == 0
        assert receipt['boundary']['field_count'] == 112 and receipt['boundary']['retained_live_array_refs'] == 0
        if prior:
            assert receipt['comparisons']['terminal']['status'] == 'PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
            assert receipt['comparisons']['boundary']['status'] == 'PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT'
        prior = receipt
        cases.append(dict(mode=mode, shape=list(shape), intervals=intervals, comparisons=receipt['comparisons']))
    (tmp_path / 'LOCAL_TRANSPARENCY.json').write_text(json.dumps(cases, indent=2), encoding='utf-8')
