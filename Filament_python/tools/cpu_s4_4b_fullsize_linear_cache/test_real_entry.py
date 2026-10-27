"""Fresh actual G3/new-capacity L1 processes, continuous micro two/eight windows."""
import json,os,shutil,subprocess,sys
from pathlib import Path
import pytest
from tools.cpu_s4_3g3_smallgrid import benchmark as sg

@pytest.mark.parametrize('intervals',(2,8))
def test_real_entry_current_g3_and_new_capacity_exact_disk_roundtrip(intervals,tmp_path):
    origin=Path(os.environ['S44B_REPORT_ROOT']);report=tmp_path/'report';report.mkdir()
    shutil.copyfile(origin/'S44B_EXECUTION_BINDING.json',report/'S44B_EXECUTION_BINDING.json')
    sg.input_manifest(report,shape=(384,17,19));previous=None;rows=[]
    for role in ('G3_REF_FULLSIZE','L1_CACHE_FULLSIZE'):
        work=tmp_path/role
        command=[sys.executable,'-s','-B','-m','tools.cpu_s4_4b_fullsize_linear_cache.entry','--report',str(report),'--work',str(work),'--role',role,'--mode','QUALIFY_BOUNDARY','--local-engineering','--micro-shape','384,17,19','--intervals',str(intervals),'--input-manifest',str(report/'BENCHMARK_INPUT_MANIFEST.json')]
        if previous:command+=['--reference-output',previous['output'],'--reference-boundary',previous['boundary_path'],'--reference-stages',previous['stages']]
        process=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=900)
        (tmp_path/(role+'.stdout.log')).write_text(process.stdout,encoding='utf-8');(tmp_path/(role+'.stderr.log')).write_text(process.stderr,encoding='utf-8')
        assert process.returncode==0,process.stderr
        current=json.loads((work/'CASE_RECEIPT.json').read_text())
        assert current['shape']==[384,17,19] and current['intervals']==intervals
        assert current['route']['nH']==2*intervals and current['route']['fallback']==current['route']['rejected']==current['cache_fallback']==0
        disk=json.loads((work/'DISK_BOUNDARY_ROUNDTRIP.json').read_text())
        assert disk['collector_retired'] is True and disk['live_array_references']==0
        stream=json.loads((work/'formal/PROPAGATION/state/streaming_manifest.json').read_text())
        assert len(stream['records'])==intervals and all(row.get('post') is not None for row in stream['records'])
        if previous:
            assert current['L1_executed'] is True and current['cache']['max_bytes']==268435456
            assert current['cache']['owned_object_audit']['status']=='PASS'
            assert current['cache']['stats']['factor_requests']==384*2*intervals
            for name,status in {'terminal':'PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS','boundary':'PASS_CURRENT_BOUNDARY_SNAPSHOT_NATIVE_EXACT','stages':'PASS_FULL_ELEMENT_NATIVE_LAYOUT_ALIAS'}.items():assert current['comparisons'][name]['status']==status
            assert current['FFT_transform_counts']==previous['FFT_transform_counts']
        else:assert current['cache'] is None and current['L1_executed'] is False
        rows.append(current);previous=current
    (tmp_path/'S44B_REAL_ENTRY_ROUNDTRIP_RECEIPT.json').write_text(json.dumps(dict(status='PASS_ACTUAL_S44B_MICRO_2_8_DISK_EXACT',intervals=intervals,shape=[384,17,19],records=rows,fullsize_science='NOT_RUN_LOCAL_MICRO_ONLY'),indent=2),encoding='utf-8')
