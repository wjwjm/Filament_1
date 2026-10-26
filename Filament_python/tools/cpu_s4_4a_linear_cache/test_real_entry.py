"""Actual new entry, fresh processes, Nt384 continuous 2/8 disk roundtrips."""
import json,os,shutil,subprocess,sys
from pathlib import Path
import pytest
from tools.cpu_s4_3g3_smallgrid import benchmark as sg

@pytest.mark.parametrize('intervals',(2,8))
def test_l1_real_entry_continuous_post_and_original_disk_roundtrip(intervals,tmp_path):
    root=Path(os.environ['S43SG_REPORT_ROOT']);report=tmp_path/'report';report.mkdir()
    for name in ('EXPERIMENT_PLAN.json','EXECUTION_BINDING.json'):shutil.copyfile(root/name,report/name)
    shape=(384,17,19);sg.input_manifest(report,shape=shape)
    previous=None;records=[]
    for role in ('G3_REF','L1_LINEAR_FACTOR_CACHE'):
        work=tmp_path/role
        argv=[sys.executable,'-s','-B','-m','tools.cpu_s4_4a_linear_cache.entry','--report',str(report),'--work',str(work),'--role',role,'--mode','QUALIFY_BOUNDARY','--local-engineering','--micro-shape','384,17,19','--intervals',str(intervals),'--input-manifest',str(report/'BENCHMARK_INPUT_MANIFEST.json')]
        if previous:
            argv.extend(('--reference-output',previous['output'],'--reference-boundary',previous['boundary_path'],'--reference-stages',previous['stages']))
        result=subprocess.run(argv,capture_output=True,text=True,encoding='utf-8',timeout=900)
        (tmp_path/(role+'.stdout.log')).write_text(result.stdout,encoding='utf-8')
        (tmp_path/(role+'.stderr.log')).write_text(result.stderr,encoding='utf-8')
        assert result.returncode==0,result.stderr
        current=json.loads((work/'CASE_RECEIPT.json').read_text(encoding='utf-8'))
        assert current['shape']==list(shape) and current['intervals']==intervals
        assert current['route']['nH']==2*intervals and current['route']['fallback']==0
        assert current['route']['lifecycle'] and all(value==2*intervals for value in current['route']['lifecycle'].values())
        stream=json.loads((work/'formal/PROPAGATION/state/streaming_manifest.json').read_text())
        assert len(stream['records'])==intervals and all(item.get('post') is not None for item in stream['records'])
        disk=json.loads((work/'DISK_BOUNDARY_ROUNDTRIP.json').read_text())
        assert disk['collector_retired'] is True and disk['live_array_references']==0
        assert disk['both_comparison_inputs']=='ACTUAL_PATHS_AFTER_COLLECTOR_AND_VIEW_CLOSE'
        if previous:
            assert current['comparisons']['terminal']['status']=='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
            assert current['comparisons']['boundary']['status']=='PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT'
            assert current['comparisons']['stages']['status']=='PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'
            stats=current['cache']['stats']
            assert stats['factor_requests']==shape[0]*2*intervals
            assert stats['factor_requests']==stats['hits']+stats['plane_builds']
            assert stats['invalidations']==stats['failed_requests']==0
        else:assert current['cache'] is None
        previous=current;records.append(current)
    (tmp_path/'REAL_ENTRY_DISK_ROUNDTRIP_RECEIPT.json').write_text(json.dumps(dict(status='PASS_ACTUAL_G3_L1_ENTRY_AND_DISK_ROUNDTRIP',intervals=intervals,shape=list(shape),records=records,scope='LOCAL_MICRO_ONLY; no W48/full benchmark qualification'),indent=2),encoding='utf-8')
