"""S4-6I scalar gates and opt-in W1 entry validation."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.cpu_s4_6k_five_chain_cost import entry, hpc
from tools.cpu_s4_6k_five_chain_cost.cost_analysis import analyze_cost_six, validate_cost_records
from tools.cpu_s4_6k_five_chain_cost.protocol import CANDIDATE, MICRO_SHAPE, SHAPE, matrix, plan


@pytest.mark.skipif(not os.environ.get('S46K_REAL_ENTRY_REPORT'),reason='opt-in real W1 entry only')
def test_s46k_real_micro_entry_opt_in():
    report=Path(os.environ['S46K_REAL_ENTRY_REPORT']).resolve(strict=True)
    work=Path(os.environ['S46K_REAL_ENTRY_WORK']).resolve()
    from tools.cpu_s4_3g3_smallgrid import benchmark as sg
    manifest=report/'BENCHMARK_INPUT_MANIFEST.json'
    if not manifest.exists():
        sg.input_manifest(report,shape=MICRO_SHAPE)
    work.mkdir(parents=True,exist_ok=False)
    results={}
    for spec in matrix():
        label,role,n=spec['case'],spec['role'],spec['intervals']
        target=work/label
        argv=[sys.executable,'-s','-B','-m','tools.cpu_s4_6k_five_chain_cost.entry',
              '--report',str(report),'--work',str(target),'--candidate',CANDIDATE,
              '--role',role,'--cost-mode',spec['cost_mode'],'--intervals',str(n),
              '--local-engineering','--input-manifest',str(manifest)]
        reference=results.get('OFF2' if n==2 else 'OFF8_A')
        if reference:
            argv += ['--reference-output',reference['output']]
            if n==2:
                argv += ['--reference-boundary',reference['boundary_path'],'--reference-stages',reference['stages']]
        with (work/(label+'.stdout.log')).open('xb') as out, (work/(label+'.stderr.log')).open('xb') as err:
            process=subprocess.run(argv,env=os.environ.copy(),stdout=out,stderr=err)
        assert process.returncode==0, f'{label} exit {process.returncode}; see {work}'
        result=json.loads((target/'CASE_RECEIPT.json').read_text(encoding='utf-8'))
        assert result['execution_domain']=='S46K_LOCAL_MICRO' and result['pool_role']=='X'
        assert result['role']==role and result['pointwise_calls']==43*n and result['alpha_rounds']==5*n
        assert result['route']['math_dispatches']==16*n and result['route']['fallback']==0
        assert result['pointwise_pool']['pool']['closed'] and not result['pointwise_pool']['pool']['active_thread_ids']
        if spec['cost_mode']=='OFF':
            assert result['residual_cost_records']==0
        else:
            assert result['residual_cost_records']>0
        if reference:
            assert result['comparisons']['terminal']['status']=='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
            if n==2:
                assert result['comparisons']['boundary']['status']=='PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT'
                assert result['comparisons']['stages']['status']=='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
        results[label]=result
