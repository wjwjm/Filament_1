"""One tiny allocation-local step with bounded cleanup and raw query receipts."""
from pathlib import Path
import argparse,datetime,json,os,subprocess,sys,time


def query_steps(job, name, step_id, runner=subprocess.run):
    cmd=['squeue','--steps','--noheader','--jobs',job,'--format=%i|%j']
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();started=time.monotonic()
    try:
        q=runner(cmd,capture_output=True,text=True,timeout=5)
        rows=[line.split('|',1) for line in q.stdout.splitlines() if line.strip()]
        ids=[r[0].strip() for r in rows if r[0].strip().startswith(job+'.')
             and r[0].strip().split('.',1)[1].isdigit()
             and (r[0].strip()==step_id or (len(r)>1 and r[1].strip()==name))]
        state='QUERY_FAILED' if q.returncode else ('STEP_PRESENT' if ids else 'STEP_ABSENT')
        return dict(command=cmd,utc=stamp,returncode=q.returncode,stdout=q.stdout,stderr=q.stderr,state=state,step_ids=ids,seconds=time.monotonic()-started)
    except Exception as exc:
        return dict(command=cmd,utc=stamp,state='QUERY_FAILED',error=repr(exc),seconds=time.monotonic()-started)


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    root=a.output;root.mkdir(parents=True,exist_ok=False)
    assert os.environ.get('SLURM_JOB_NUM_NODES')=='1'
    assert os.environ.get('SLURM_CPUS_PER_TASK')=='96'
    name='P1R2-probe';job=os.environ['SLURM_JOB_ID'];assert job.isdigit()
    code='import os,json,socket; from KHz_filament.device import USE_GPU,xp; assert not USE_GPU and xp.__name__=="numpy"; assert len(os.sched_getaffinity(0))==1; print(json.dumps(dict(hostname=socket.gethostname(),affinity=sorted(os.sched_getaffinity(0)),step_id=os.environ["SLURM_STEP_ID"],backend=xp.__name__,rank=os.environ["SLURM_PROCID"])),flush=True)'
    cmd=['srun','--exclusive','--nodes=1','--ntasks=1','--cpus-per-task=1','--cpu-bind=cores','--job-name='+name,sys.executable,'-s','-B','-c',code]
    started=time.monotonic();step_id=None;parse_error=None;cancelled=set()
    with (root/'step.stdout').open('xb') as out,(root/'step.stderr').open('xb') as err:
        proc=subprocess.Popen(cmd,stdout=out,stderr=err)
        try:rc=proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
            rc=124
    try:
        step=json.loads((root/'step.stdout').read_text().strip().splitlines()[-1])
        assert str(step['step_id']).isdigit()
        step_id=job+'.'+str(step['step_id'])
    except Exception as exc:parse_error=repr(exc)
    queries=[];absent=False
    for attempt in range(6):
        record=query_steps(job,name,step_id);queries.append(record)
        with (root/'queries.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
        if record['state']=='STEP_ABSENT':absent=True;break
        if record['state']=='STEP_PRESENT' and (rc!=0 or attempt>=2):
            for target in record['step_ids']:
                if target in cancelled:continue
                cancel_cmd=['scancel',target]
                try:
                    q=subprocess.run(cancel_cmd,capture_output=True,text=True,timeout=5)
                    cancel=dict(command=cancel_cmd,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=q.returncode,stdout=q.stdout,stderr=q.stderr)
                except Exception as exc:cancel=dict(command=cancel_cmd,error=repr(exc))
                with (root/'cleanup.jsonl').open('a') as f:f.write(json.dumps(cancel)+'\n');f.flush();os.fsync(f.fileno())
                cancelled.add(target)
        time.sleep(2)
    result=dict(command=cmd,exit_code=rc,step_id=step_id,parse_error=parse_error,teardown_verified=absent,query_count=len(queries),cancelled_own_steps=sorted(cancelled),wall_seconds=time.monotonic()-started,status='PASS' if rc==0 and absent and not parse_error and not cancelled else 'FAIL',historical_H2_root_cause='UNRESOLVED')
    with (root/'receipt.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result));return 0 if result['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())