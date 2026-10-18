"""Bounded control checks; benchmark opt-in requires the normal local runner."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize('count', [8, 64])
def test_journal_control_reference_blocks_and_fresh_replay(tmp_path, count):
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle, FIELDS
    fields = {key: np.zeros((count, 3, 3), dtype=np.float64) for key in FIELDS}
    fields['delta_n'][:] = -1.e-8
    records = [dict(ordinal=i, screen_id=f'z{i:05d}', z_m=i * 1.e-4) for i in range(count)]
    root = tmp_path / 'stream'
    life = StreamingLifecycle.create(root=root, current=fields, screen_records=records,
        current_generation='control:p0', dx_m=1.e-4, dy_m=1.e-4)
    for offset in range(0, count, 8):
        for i in range(offset, offset + 8):
            life.deposition_finalized(i)
            life.commit_post(i, {key: fields[key][i] for key in FIELDS})
            life.enqueue_post(i)
        block = life.claim_block(actor='H0')
        assert block == list(range(offset, offset + 8))
        for i in block:
            life.begin_hydro_screen(i, actor='H0')
            life.commit_next(i, {key: fields[key][i] for key in FIELDS}, actor='H0')
    reopened = StreamingLifecycle.open(root)
    assert all(r['state'] == 'NEXT_COMMITTED' for r in reopened.manifest['records'])
    assert reopened.validate_barrier()['status'] == 'PASS'
    promoted = reopened.promote_next_to_current()
    fresh = StreamingLifecycle.open(root)
    assert fresh.promote_next_to_current() == promoted
    for i in range(count):
        for key in FIELDS:
            np.testing.assert_array_equal(fresh.current_fields(i)[key], fields[key][i])


@pytest.mark.skipif(not os.environ.get('N5_CONTROL_BENCHMARK_EVIDENCE'), reason='explicit performance run only')
def test_journal_control_k8048_external_benchmark():
    evidence = Path(os.environ['N5_CONTROL_BENCHMARK_EVIDENCE'])
    mode = os.environ['N5_CONTROL_BENCHMARK_MODE']
    source = Path(os.environ['N5_CONTROL_BENCHMARK_SOURCE'])
    tag = os.environ['N5_CONTROL_BENCHMARK_TAG']
    cycles = os.environ.get('N5_CONTROL_BENCHMARK_CYCLES', '32')
    old = Path(r'D:\Filament_1\review_evidence\E5_N5_JOB273606_FAILURE_REPAIR_20261017T020822897Z')
    original = old / 'FAILURE_EXPORT_V1' / 'REMOTE' / 'run'
    output = evidence / tag
    env = dict(os.environ, PYTHONPATH=str(source / 'Filament_python'), UPPE_USE_GPU='0',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    script = os.environ.get('N5_CONTROL_BENCHMARK_SCRIPT', 'control_benchmark_v2.py')
    action = os.environ.get('N5_CONTROL_BENCHMARK_ACTION', 'run')
    command = [sys.executable, '-s', '-B', str(evidence / script), action,
               '--mode', mode, '--cycles', cycles, '--output', str(output),
               '--rsource', os.environ.get('N5_CONTROL_RSOURCE', str(original / 'R' / 'p0' / 'state' / 'streaming_manifest.json')),
               '--csource', os.environ.get('N5_CONTROL_CSOURCE', str(original / 'C' / 'p0' / 'state' / 'streaming_manifest.json'))]
    if os.environ.get('N5_CONTROL_REAL_PAYLOADS') == '1':
        command += ['--real-payloads', '--prune-completed-prefix']
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=1200)
    (evidence / (tag + '.stdout')).write_text(result.stdout, encoding='utf-8')
    (evidence / (tag + '.stderr')).write_text(result.stderr, encoding='utf-8')
    receipt = json.loads((output / 'BENCHMARK.json').read_text())
    print(json.dumps(dict(mode=mode, status=receipt['status'], elapsed_s=receipt['elapsed_s'], output=str(output))))
    if mode == 'old' and receipt['status'] == 'FAIL_STOPPED':
        actor_receipts = list(output.glob('cycle_*/*/RECEIPT.json'))
        failures = [json.loads(p.read_text()).get('traceback', '') for p in actor_receipts]
        assert any('timed out acquiring lifecycle manifest lock' in t for t in failures), receipt
        assert actor_receipts, 'censored baseline must retain raw observations'
        return
    assert result.returncode == 0, receipt.get('traceback', result.stderr)
    assert receipt['status'] == 'PASS_CONTROL_ONLY'
