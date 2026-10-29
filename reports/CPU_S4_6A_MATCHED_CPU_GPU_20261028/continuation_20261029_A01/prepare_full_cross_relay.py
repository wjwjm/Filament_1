"""Prepare the existing byte-relay protocol for one completed CPU FULL QUAL_OFF."""
import json,sys
from pathlib import Path
repo=Path.cwd();out=Path(__file__).resolve().parent;control=out/'control'
sys.path.insert(0,str(repo/'Filament_python'))
from tools.hpc_ops import provenance_v2 as pv
src=(control/'preposition_source.py').read_text(encoding='utf8')
src=src.replace('completed small CPU reference once.','completed FULL QUAL_OFF reference once, after CPU terminal state.')
src=src.replace("report=root/'report';canonical=root.parent/'filament_cpu_s46a_20261028_a01/report/canonical'", "report=root/'report_full'")
src=src.replace("from tools.cpu_s4_6a.records import validate_source_records", "from tools.cpu_s4_6a.records import validate_source_records\nassert len(sys.argv)==2 and pv.raw_sha256_file(Path(__file__))==sys.argv[1],'SOURCE_SCRIPT_RAW_IDENTITY'")
src=src.replace('MATRIX_RESULT_CPU_117.json','MATRIX_RESULT_CPU_351.json')
start=src.index("for shape in ('384x117x129','384x351x301'):")
end=src.index('files.extend(',start)
src=src[:start]+src[end:]
src=src.replace('CPU_SMALLGRID_TRANSPARENCY.json','CPU_FULLGRID_TRANSPARENCY.json').replace('CPU_SMALLGRID_OUTPUT','CPU_FULLGRID_OUTPUT').replace('CPU_SMALLGRID_FIRST_BOUNDARY','CPU_FULLGRID_FIRST_BOUNDARY').replace('CROSS_FIRST_CPU_117','CROSS_FIRST_CPU_351')
src=src.replace('PREPOSITIONED_INPUTS_AND_CPU_REFERENCE.json','CPU_FULL_CROSS_REFERENCE_MANIFEST_A01.json').replace('CPU_small_timing_gate','CPU_full_timing_gate')
src=src.replace("matrix=read(report/'MATRIX_RESULT_CPU_351.json');assert len(matrix['cases'])==8", "matrix=read(report/'MATRIX_RESULT_CPU_351.json');assert len(matrix['cases'])==8 and matrix['shape']==[384,351,301]\nassert done['job_id']=='8855018' and matrix['cases'][0]['case']=='QUAL_OFF'\nassert matrix['cases'][0]['source_binding']['head']=='ed8754041625549a80959db7a8abd804d5d219ae'")
receiver=(control/'preposition_receive.py').read_text(encoding='utf8')
receiver=receiver.replace('filament_cpu_s46a_20261029_r2','filament_cpu_s46a_20261029_r6').replace("report=root/'report'", "report=root/'CROSS_FULL_NUMERIC_A01'")
receiver=receiver.replace("assert not (root/'SUBMISSION_LOCK.json').exists() and not (report/'canonical').exists()", "assert not report.exists(),'NON_OVERWRITING_FULL_NUMERIC_STAGE'")
receiver=receiver.replace('PREPOSITIONED_INPUTS_AND_CPU_REFERENCE.json','CPU_FULL_CROSS_REFERENCE_MANIFEST_A01.json')
receiver=receiver.replace('PASS_PREPOSITIONED_ORIGINAL_RAW_BYTES','PASS_CPU_FULL_QUAL_SNAPSHOTS_ORIGINAL_RAW_BYTES').replace('PREPOSITION_RECEIPT.json','CPU_FULL_CROSS_RECEIPT_A01.json')
receiver=receiver.replace("from tools.cpu_s4_6a.records import validate_source_records", "from tools.cpu_s4_6a.records import validate_source_records\nassert len(sys.argv)==2 and pv.raw_sha256_file(Path(__file__))==sys.argv[1],'RECEIVER_SCRIPT_RAW_IDENTITY'\nassert json.loads((root/'report/ALLOCATION_COMPLETED.json').read_text())['job_id']=='290195'\nreport.mkdir()")
receiver=receiver.replace("validate_source_records(report,manifest['records'])", "assert manifest['source_job']=='8855018'\nvalidate_source_records(report,manifest['records'])")
for name,value in (('preposition_full_source_A01.py',src),('preposition_full_receive_A01.py',receiver)):
 compile(value,name,'exec')
 with (control/name).open('x',encoding='utf8',newline='\n') as stream:stream.write(value)
source_sha=pv.raw_sha256_file(control/'preposition_full_source_A01.py')
receive_sha=pv.raw_sha256_file(control/'preposition_full_receive_A01.py')
relay=(out/'relay_inputs_final.py').read_text(encoding='utf8')
relay=relay.replace("r=Path(__file__).resolve().parent", "r=Path(__file__).resolve().parent.parent")
relay=relay.replace("/preposition_source.py']", "/preposition_full_source_A01.py',"+repr(source_sha)+"]")
relay=relay.replace('filament_cpu_s46a_20261029_r3/preposition_receive.py', 'filament_cpu_s46a_20261029_r6/preposition_full_receive_A01.py')
relay=relay.replace("/preposition_full_receive_A01.py']", "/preposition_full_receive_A01.py',"+repr(receive_sha)+"]")
relay=relay.replace('RELAY_SOURCE','FULL_CROSS_RELAY_SOURCE_A01').replace('RELAY_DEST','FULL_CROSS_RELAY_DEST_A01').replace('RELAY_RECEIPT','FULL_CROSS_RELAY_RECEIPT_A01')
compile(relay,'relay_full_cross_A01.py','exec')
with (control/'relay_full_cross_A01.py').open('x',encoding='utf8',newline='\n') as stream:stream.write(relay)
paths=[control/'preposition_full_source_A01.py',control/'preposition_full_receive_A01.py',control/'relay_full_cross_A01.py',out/'CROSS_FULL_NUMERIC_A01/cross_full_readonly.py']
mf=pv.create_manifest(repo,repo.parent/'S46A_FULL_CROSS_CONTROL_A01.json',[],[str(path) for path in paths])
with (out/'CROSS_FULL_NUMERIC_A01/CONTROL_PROVENANCE_A01.json').open('x',encoding='utf8',newline='\n') as stream:json.dump(mf,stream,indent=2);stream.write('\n')
print(json.dumps(dict(status='FULL_CROSS_RELAY_PREPARED_ONLY_NOT_EXECUTED',source_raw_sha256=source_sha,receiver_raw_sha256=receive_sha,adapter_raw_sha256=pv.raw_sha256_file(paths[-1]),arrays_saved_local=False)))
