"""Terminal metadata-only export through stdout; no remote mutations."""
import io,json,subprocess,sys,zipfile
from pathlib import Path
root=Path('/publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_s46c_20261031_a02');report=root/'report';code=root/'code'
job=json.loads((report/'SUBMISSION_RECEIPT.json').read_text())['job_id']
p=subprocess.run(['sacct','-j',job,'-n','--parsable2','--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,TimelimitRaw'],capture_output=True,text=True,check=True)
main=[r.split('|') for r in p.stdout.splitlines() if r.split('|')[0]==job]
assert len(main)==1 and main[0][1].split()[0] in ('COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED'),'TERMINAL_MAIN_ALLOCATION_REQUIRED'
sys.path.insert(0,str(code/'Filament_python'))
from tools.hpc_ops import provenance_v2 as pv
buffer=io.BytesIO();records=[];excluded=[]
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for path in sorted(report.rglob('*')):
  if not path.is_file():continue
  relative=path.relative_to(report)
  if 'numba_cache' in relative.parts or path.suffix in ('.npy','.npz','.nbi','.nbc'):
   excluded.append(dict(path=relative.as_posix(),bytes=path.stat().st_size,reason='SCIENCE_ARRAY_OR_COMPILER_CACHE_REMAINS_REMOTE'));continue
  if path.suffix not in ('.json','.jsonl','.log','.md','.csv','.bin'):continue
  payload=path.read_bytes();name='raw/'+relative.as_posix();z.writestr(name,payload)
  records.append(dict(path=name,hash_scope='raw_bytes',raw_sha256=pv.raw_sha256(payload),bytes=len(payload)))
 for src,name in [(code/'Filament_python/tools/cpu_o1/numeric_gate.py','numeric_gate.py'),
                  (code/'Filament_python/KHz_filament/raman.py','raman.py')]:
  payload=src.read_bytes();name='execution_sources/'+name;z.writestr(name,payload)
  records.append(dict(path=name,hash_scope='raw_bytes',raw_sha256=pv.raw_sha256(payload),bytes=len(payload)))
 z.writestr('SCHEDULER_TERMINAL.json',json.dumps(dict(job_id=job,argv=p.args,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2))
 z.writestr('EXPORT_MANIFEST.json',json.dumps(dict(hash_scope='raw_bytes',records=records,excluded=excluded,science_arrays_copied=False),indent=2))
sys.stdout.buffer.write(buffer.getvalue())
