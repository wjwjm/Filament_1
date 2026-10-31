"""S46C admission and existing process-group / cgroup guardian reuse."""
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from tools.hpc_ops import provenance_v2 as pv
from tools.cpu_s4_6a.inputs import write, require
from tools.cpu_s4_6a.matrix import command, proc_tree
from tools.cpu_o1.cpu_s4.memory_limits import snapshot_cgroup_memory
from tools.cpu_o1.cpu_s4.supervisor import terminate


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    report=args.report.resolve();repo=Path(__file__).resolve().parents[3]
    plan=json.loads((report/'PLAN.json').read_text());binding=json.loads((report/'EXECUTION.json').read_text())
    require(pv.raw_sha256_file(report/'PLAN.json')==os.environ['S46C_PLAN_SHA256'],'FROZEN_PLAN')
    source=pv.validate_manifest(repo,report/'EXECUTION_SOURCE_MANIFEST.json',require_clean=True,require_hash_scope=True)
    require(source['head']==binding['execution_commit'],'EXECUTION_COMMIT')
    require(pv._git(repo,'rev-parse','HEAD:Filament_python/KHz_filament')==plan['science_subtree'],'FROZEN_SCIENCE')
    import getpass,psutil
    require(getpass.getuser()=='t0s000727','T0_ACCOUNT')
    job=os.environ.get('SLURM_JOB_ID','');require(job.isdigit(),'COMPUTE_NODE')
    require(os.environ.get('SLURM_CPUS_PER_TASK')=='96' and os.environ.get('SLURM_JOB_NUM_NODES')=='1','ONE_NODE_CPU96')
    desc=command(['scontrol','show','job','-o',job]);require(desc['exit_code']==0,'SCONTROL')
    fields=dict(x.split('=',1) for x in desc['stdout'].split() if '=' in x)
    require(fields['NumCPUs']=='96' and fields['TimeLimit']=='00:30:00' and fields['Account']=='bscc-t'
            and fields['Partition']=='v6_384' and 'gres/gpu' not in desc['stdout'],'FIXED_ALLOCATION')
    tres=dict(item.split('=',1) for item in fields['AllocTRES'].split(','))
    require(tres.get('mem')=='64G','MEMORY_64G_NO_EXPANSION')
    affinity=sorted(os.sched_getaffinity(0));topology=command(['lscpu','-p=CPU,CORE,SOCKET,NODE'])
    require(topology['exit_code']==0,'TOPOLOGY')
    rows=[r.split(',') for r in topology['stdout'].splitlines() if r and not r.startswith('#')]
    require(len(affinity)==len({(r[2],r[1]) for r in rows if int(r[0]) in affinity})==96,'96_PHYSICAL_CORES')
    versions={n:importlib.import_module(n).__version__ for n in plan['versions']}
    require(versions==plan['versions'] and sys.version_info[:3]==(3,12,8),'FROZEN_ENVIRONMENT')
    memory=snapshot_cgroup_memory(os.getpid(),allocation_bytes=64*1024**3)
    require(memory['status']=='READ' and memory['effective_limit_bytes']>0,'ACTUAL_CGROUP')
    guard=int(memory['effective_limit_bytes']*.8)
    require(plan['memory_forecast_peak_bytes']<guard,'FORECAST_WITHIN_ACTUAL_80_PERCENT')
    perf=shutil.which('perf')
    counters=command([perf,'stat','-e','cycles,instructions,cache-misses','--','true']) if perf else dict(status='NOT_INSTALLED')
    node=dict(job_id=job,status='PASS',allocation=desc,affinity=affinity,topology=topology,
              hardware=command(['lscpu']),versions=versions,executable=sys.executable,memory=memory,
              guardian_bytes=guard,perf_permission_probe=counters,
              hardware_counters='NOT_MEASURED_FOR_TARGET' if counters.get('exit_code')==0 else 'UNAVAILABLE',
              source_commit=source['head'],thread_env={k:os.environ.get(k) for k in plan['thread_environment']})
    require(node['thread_env']==plan['thread_environment'],'THREAD_ENVIRONMENT')
    write(report/'NODE_GATE.json',node)
    deadline=time.monotonic()+1680
    argv=[sys.executable,'-s','-B','-m','tools.cpu_s4_6c_array_coverage.runner','--report',str(report)]
    start=time.monotonic()
    with (report/'worker.stdout.log').open('xb') as so,(report/'worker.stderr.log').open('xb') as se:
        process=subprocess.Popen(argv,stdout=so,stderr=se,start_new_session=True)
    created=psutil.Process(process.pid).create_time();peak=0;stop=None
    try:
        while process.poll() is None:
            rss=proc_tree(process.pid);peak=max(peak,rss)
            usage=[int((Path(r['path'])/('memory.usage_in_bytes' if memory['version']==1 else 'memory.current')).read_text())
                   for r in memory['enforced_cgroup_limits']]
            if max([rss]+usage)>guard:stop='MEMORY_GUARD_80_PERCENT';break
            if time.monotonic()>deadline:stop='DEADLINE_1680_SECONDS';break
            time.sleep(1)
    finally:
        if process.poll() is None: terminate(process.pid,created)
    code=process.wait(timeout=15)
    write(report/'SUPERVISOR_RECEIPT.json',dict(exit_code=code,stop_reason=stop,peak_tree_rss_bytes=peak,
         process_wall_s=time.monotonic()-start,job_id=job,argv=argv))
    require(code==0 and stop is None,'WORKER_FAILURE')
    require((report/'MECHANISM_COMPLETED.json').is_file(),'COMPLETE_RAW_RECORDS')
    write(report/'ALLOCATION_COMPLETED.json',dict(job_id=job,status='MECHANISM_COMPLETE',review_gate='MANUAL_REVIEW_PENDING'))


if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();sys.exit(1)
