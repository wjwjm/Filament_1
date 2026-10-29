#!/usr/bin/env bash
set -euo pipefail
test "$(id -un)" = scvi806
export UPPE_USE_GPU=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
/data/home/scvi806/.conda/envs/Filament_python/bin/python -s -B - "$1" <<'PY'
import collections,datetime,json,subprocess,sys,time,traceback
from pathlib import Path
root=Path('/data/run01/scvi806/user_Wangjimin/projects/filament_cpu_s46a_20261029_r6')
code=root/'code';report=root/'report';stage=root/'CROSS_FULL_NUMERIC_A01'
assert root.resolve()==root and stage.is_dir()
sys.path.insert(0,str(code/'Filament_python'))
from tools.hpc_ops import provenance_v2 as pv
from tools.cpu_s4_6a.records import load_json_strict,validate_source_records
adapter=stage/'cross_full_readonly.py'
assert pv.raw_sha256_file(adapter)=='02ef997d21f633f29a3dc5e63f7368d021123ecb0ea30eefbc16edd17c6bb884','ADAPTER_CONTROL_BINDING'
transfer=stage/'CPU_FULL_CROSS_REFERENCE_MANIFEST_A01.json'
assert pv.raw_sha256_file(transfer)==sys.argv[1],'CPU_TRANSFER_MANIFEST_RAW_BINDING'
cpu_transfer=load_json_strict(transfer)
assert cpu_transfer['source_job']=='8855018'
validate_source_records(stage,cpu_transfer['records'])
index_path=root/'METADATA_GPU_COMPLETED_A01/raw/GPU/EXPORTED_FILE_INDEX.json'
assert pv.raw_sha256_file(index_path)=='da8568ff12a4cd93922aae3d2f3db6590a8fc94c9ba9ff1aef08ccc7c17480d7','VERIFIED_GPU_EXPORT_INDEX_BINDING'
index=load_json_strict(index_path)
export=root/'METADATA_GPU_COMPLETED_A01'
def original_metadata(relative):
 original=report/relative
 row=next(row for row in index['records'] if row['original_path']==str(original))
 archived=export/row['path']
 assert archived.resolve().is_relative_to(export)
 assert pv.raw_sha256_file(archived)==row['raw_sha256'] and archived.stat().st_size==row['size_bytes']
 assert pv.raw_sha256_file(original)==row['raw_sha256'] and original.stat().st_size==row['size_bytes']
 return load_json_strict(archived),dict(path=str(original),classification='external',hash_scope='raw_bytes',raw_sha256=row['raw_sha256'],size_bytes=row['size_bytes'])
gpu_matrix,matrix_binding=original_metadata(Path('MATRIX_RESULT_GPU_351.json'))
qual=report/'runs/GPU/384x351x301/QUAL_OFF/work'
assert gpu_matrix['cases'][0]['case']=='QUAL_OFF'
assert Path(gpu_matrix['cases'][0]['output'])==qual/'OUTPUT'
assert Path(gpu_matrix['cases'][0]['boundary_path'])==qual/'BOUNDARY'
raw_index,raw_index_binding=original_metadata(qual.relative_to(report)/'OUTPUT_RAW_INDEX.json')
boundary_proof,boundary_binding=original_metadata(qual.relative_to(report)/'BOUNDARY_NATIVE_READBACK_PROVENANCE.json')
gpu_records=[]
for row in raw_index['roots']['OUTPUT']:
 path=qual/'OUTPUT'/row['path']
 assert path.resolve().is_relative_to(qual/'OUTPUT') and path.is_file() and not path.is_symlink()
 assert row['hash_scope']=='raw_bytes' and path.stat().st_size==row['size_bytes']
 assert pv.raw_sha256_file(path)==row['sha256_raw']
 gpu_records.append(dict(path=str(path),classification='external',hash_scope='raw_bytes',raw_sha256=row['sha256_raw'],size_bytes=row['size_bytes']))
for name,row in boundary_proof['candidate_snapshot'].items():
 path=qual/'BOUNDARY'/name
 assert str(path)==row['path'] and path.stat().st_size==row['size_bytes']
 assert row['hash_scope']=='raw_bytes' and pv.raw_sha256_file(path)==row['raw_sha256']
 gpu_records.append(row)
for tree in (stage/'CPU_FULLGRID_FIRST_BOUNDARY/TREE.json',qual/'BOUNDARY/TREE.json'):
 value=load_json_strict(tree)
 assert value['kind']=='dict' and len(value['items'])==112,'ORIGINAL_112_FIELDS_REQUIRED'
binding=dict(schema='filament.cpu_s46a.full_cross_entry_binding.v1',hash_scope='classified',
 CPU_transfer_manifest_raw_sha256=sys.argv[1],CPU_transfer_job='8855018',GPU_source_job='290195',
 GPU_matrix=matrix_binding,GPU_raw_index=raw_index_binding,GPU_boundary_proof=boundary_binding,
 GPU_snapshot_records=gpu_records,analysis_adapter=dict(path=str(adapter),classification='external',hash_scope='raw_bytes',raw_sha256=pv.raw_sha256_file(adapter)),
 arrays_saved_on_local_machine=False,propagations_executed=0,GPU_computations_executed=0,
 started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
with (stage/'FULL_CROSS_ENTRY_BINDING_A01.json').open('x',encoding='utf8') as stream:json.dump(binding,stream,indent=2,allow_nan=False)
argv=[sys.executable,'-s','-B',str(adapter),'--stage',str(stage),'--gpu-result',str(report/'MATRIX_RESULT_GPU_351.json'),'--code-root',str(code)]
started=time.perf_counter_ns()
try:
 child=subprocess.run(argv,capture_output=True,text=True,timeout=300)
 receipt=dict(argv=argv,exit_code=child.returncode,stdout=child.stdout,stderr=child.stderr,wall_ns=time.perf_counter_ns()-started)
 with (stage/'FULL_CROSS_PROCESS_RECEIPT_A01.json').open('x',encoding='utf8') as stream:json.dump(receipt,stream,indent=2)
 assert child.returncode==0,'FULL_CROSS_PROCESS_FAILURE'
 result=load_json_strict(stage/'CROSS_BACKEND_NUMERIC_DIFFERENCES_FULL.json')
 counts=dict(collections.Counter(row['domain'] for row in result['fields']))
 assert result['status']=='CROSS_BACKEND_NUMERICAL_EQUIVALENCE_NOT_QUALIFIED' and result['matched_input_workload']
 assert counts['first_boundary']==112,'FULL_112_NUMERIC_FIELDS_REQUIRED'
 print(json.dumps(dict(schema='filament.cpu_s46a.full_cross_execution.v1',ok=True,state='completed',
  status=result['status'],matched_input_workload=True,field_counts=counts,
  numeric_difference_raw_sha256=pv.raw_sha256_file(stage/'CROSS_BACKEND_NUMERIC_DIFFERENCES_FULL.json'),
  sidecar_raw_sha256=pv.raw_sha256_file(stage/'CROSS_FULL_NUMERIC_A01_SIDECAR.json'),
  propagations_executed=0,GPU_computations_executed=0,arrays_saved_on_local_machine=False,wall_s=receipt['wall_ns']/1e9)))
except BaseException as error:
 with (stage/'FULL_CROSS_FAILURE_A01.json').open('x',encoding='utf8') as stream:json.dump(dict(error=repr(error),traceback=traceback.format_exc()),stream,indent=2)
 print(json.dumps(dict(schema='filament.cpu_s46a.full_cross_execution.v1',ok=False,state='failed',error=repr(error))))
 raise
PY
