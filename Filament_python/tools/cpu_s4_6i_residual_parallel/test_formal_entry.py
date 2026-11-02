"""Actual W1 formal entry and retained-capture identity checks."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.hpc_ops import provenance_v2 as pv
from . import mechanism_admission as ma
from .formal_protocol import CANDIDATE
from .test_formal import _admission_fixture


def test_formal_retained_capture_identity_and_tamper(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, 'REMOTE_OWNER', tmp_path)
    root = tmp_path / 'filament_cpu_s46i_power_capture_fixture'
    report = root / 'report'
    report.mkdir(parents=True)
    admission = _admission_fixture()
    admission['capture_remote_root'] = str(root)
    result = dict(execution_commit=admission['capture_execution_commit'], candidate=CANDIDATE,
                  shape=[384, 351, 301], science_exact=True, formal14_allowed=True,
                  mechanism=dict(blocks=admission['blocks']))
    records = {
        'RESULT.json': result, 'EXECUTION_SOURCE_MANIFEST.json': {}, 'INPUT_CONTRACT.json': {},
        'EXECUTION.json': dict(execution_commit=admission['capture_execution_commit']),
        'SUBMISSION_RECEIPT.json': dict(job_id='8861678'),
    }
    for name, value in records.items():
        (report / name).write_text(json.dumps(value), encoding='utf-8')
    for filename, field in (
        ('RESULT.json', 'capture_result_raw_sha256'),
        ('EXECUTION_SOURCE_MANIFEST.json', 'capture_source_manifest_raw_sha256'),
        ('INPUT_CONTRACT.json', 'input_contract_raw_sha256'),
    ):
        admission[field] = pv.raw_sha256_file(report / filename)
    assert ma._remote_capture_identity(admission)['status'] == 'PASS_RETAINED_CAPTURE_IDENTITY'
    with (report / 'RESULT.json').open('a', encoding='utf-8') as stream:
        stream.write(' ')
    with pytest.raises(ma.MechanismAdmissionError, match='REMOTE_IDENTITY:RESULT'):
        ma._remote_capture_identity(admission)


def _controller_fixture(tmp_path, monkeypatch, result):
    from . import formal_hpc
    for filename, value in (
        ('FULL_INPUT_MANIFEST.json', dict(shape=[384, 351, 301], fingerprints={})),
        ('INPUT_CONTRACT.json', dict(expected_input_fingerprints={})),
        ('REFERENCE_INPUT_RAW_RECORDS.json', dict(reference=dict(boundary_path='unused', stages='unused'))),
    ):
        (tmp_path / filename).write_text(json.dumps(value), encoding='utf-8')
    monkeypatch.setattr(formal_hpc, 'node_gate', lambda *args: dict(preflight=dict(execution_commit='probe')))
    launched = []
    def launch(report, candidate, spec, reference, **kwargs):
        launched.append(spec['case'])
        return dict(result)
    monkeypatch.setattr(formal_hpc, 'launch', launch)
    monkeypatch.setattr(formal_hpc, '_exact', lambda *args, **kwargs: None)
    return formal_hpc, launched


def test_formal_controller_retains_nonexact_case_and_stops(tmp_path, monkeypatch):
    controller, launched = _controller_fixture(tmp_path, monkeypatch,
        dict(status='SCIENTIFIC_NONEXACT', role='R', intervals=2, entry_failure=dict(error='NONEXACT_PROBE')))
    assert controller.run(tmp_path)['status'] == 'SCIENTIFIC_NONEXACT'
    assert launched == ['QUAL_R2']
    result = sg.read(tmp_path / 'RESULT.json')
    assert result['scientific_stop_stage'] == 'QUAL_R2'
    assert result['cases']['QUAL_R2']['entry_failure']['error'] == 'NONEXACT_PROBE'


@pytest.mark.parametrize('changed', [dict(cost_mode='COST'), dict(mechanism_enabled=True)])
def test_formal_controller_rejects_instrumented_window(tmp_path, monkeypatch, changed):
    value = dict(status='CURRENT_REFERENCE', capture_enabled=False, mechanism_enabled=False,
                 cost_mode='OFF', residual_cost_records=0)
    value.update(changed)
    controller, launched = _controller_fixture(tmp_path, monkeypatch, value)
    with pytest.raises(RuntimeError, match='NO_CAPTURE_NO_RESIDUAL'):
        controller.run(tmp_path)
    assert launched == ['QUAL_R2']
    assert (tmp_path / 'JOB_FAILURE.json').is_file()


@pytest.mark.skipif(not os.environ.get('S46I_FORMAL_REAL_REPORT'), reason='opt-in prepared W1 formal entry')
def test_formal_real_micro_entry_opt_in():
    report = Path(os.environ['S46I_FORMAL_REAL_REPORT'])
    work = Path(os.environ['S46I_FORMAL_REAL_WORK'])
    work.mkdir(parents=True, exist_ok=False)
    cases = {}
    for role in ('R', 'X'):
        name = role + '2'
        target = work / name
        argv = [sys.executable, '-s', '-B', '-m', 'tools.cpu_s4_6i_residual_parallel.formal_entry',
                '--report', str(report), '--work', str(target), '--role', role,
                '--intervals', '2', '--local-engineering']
        reference = cases.get('R2')
        if reference:
            argv += ['--reference-output', reference['output'],
                     '--reference-boundary', reference['boundary_path'],
                     '--reference-stages', reference['stages']]
        with (work / (name + '.stdout.log')).open('xb') as out, (work / (name + '.stderr.log')).open('xb') as err:
            process = subprocess.run(argv, env=os.environ.copy(), stdout=out, stderr=err)
        assert process.returncode == 0, f'{name}: see {work}'
        value = sg.read(target / 'FORMAL_CASE_RECEIPT.json')
        assert value['baseline_pointwise_groups'] == 60
        assert value['selected_operation_groups'] == (16 if role == 'X' else 0)
        assert value['capture_enabled'] is False and value['residual_cost_records'] == 0
        assert value['pointwise_pool']['pool']['closed'] and value['route']['fallback'] == 0
        if reference:
            assert value['comparisons']['terminal']['status'] == 'PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
            assert value['comparisons']['stages']['status'] == 'PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
            assert value['comparisons']['boundary']['status'] == 'PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT'
        cases[name] = value
