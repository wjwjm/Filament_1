"""Read only sole S46C Job1 scheduler and key worker progress."""
import json,subprocess
from pathlib import Path
root=Path('/publicfs01/fs1-t/home/t0s000727/user_wjm/filament_cpu_s46c_20261031_a02');report=root/'report'
receipt=json.loads((report/'SUBMISSION_RECEIPT.json').read_text());job=receipt['job_id']
def call(argv):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=30)
 return dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
def tail(name,n=5):
 p=report/name
 return p.read_text(errors='replace').splitlines()[-n:] if p.is_file() else []
qualification=report/'exact/QUALIFICATION.jsonl'
qualified=[]
if qualification.is_file():
 for line in qualification.read_text().splitlines():
  try:
   row=json.loads(line);qualified.append({k:row[k] for k in ('input','chain','mode','workers')})
  except json.JSONDecodeError:pass
print(json.dumps(dict(job_id=job,queue=call(['squeue','-j',job,'-h','-o','%i|%T|%M|%l|%R']),
 accounting=call(['sacct','-j',job,'-n','--parsable2','--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,TimelimitRaw']),
 progress=tail('worker.stdout.log'),qualification_count=len(qualified),qualification_last=qualified[-1:] if qualified else [],error=tail('worker.stderr.log',18),batch_error=tail('slurm-'+job+'.stderr.log',18),
 gates={n:json.loads((report/n).read_text()) for n in ['SUPERVISOR_RECEIPT.json','ALL_REAL_EXACT.json','MECHANISM_COMPLETED.json'] if (report/n).is_file()},ETA='UNKNOWN')))
