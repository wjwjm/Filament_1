"""Use the existing Linux runtime; stage the review on its native filesystem."""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

def native(value):
    return value.replace('\\','/').replace('C:/','/mnt/c/').replace('D:/','/mnt/d/')

mode=sys.argv[1]
continuation=Path(__file__).resolve().parent
context=continuation/'LINUX_DELIVERY_CONTEXT_A02.json'
runtime=Path('/mnt/d/Filament_1/.worktrees/cpu_optimization/reports/CPU_S4_3B_FIXTURE_REPAIR_20261020_090443/linux/venv/bin/python')
if not runtime.is_file(): raise RuntimeError('EXISTING_REAL_LINUX_RUNTIME_REQUIRED')
if mode=='stage':
    source=Path(native(sys.argv[2])).resolve()
    root=Path(tempfile.mkdtemp(prefix='s46a-review-final-20261029-', dir='/var/tmp'))
    target=root/'reports'/source.parent.name/source.name
    shutil.copytree(source,target)
    helper=target/'source_roles/CPU/executed_source/Filament_python/tools/hpc_ops/provenance_v2.py'
    destination=root/'Filament_python/tools/hpc_ops/provenance_v2.py'
    destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(helper,destination)
    value=dict(native_root=str(root),package=str(target),source=str(source),runtime=str(runtime),
               scope='Only review metadata and original packaged Git blobs; no scientific arrays/environment installation')
    with context.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
    print(json.dumps(value))
elif mode=='replay':
    value=json.loads(context.read_text(encoding='utf-8'))
    target=Path(value['package'])
    output=continuation/'LINUX_DELIVERY_REPLAY_A02.json'
    env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
    argv=[str(runtime),'-s','-B',str(continuation/'validate_final_delivery_A01.py'),'--root',str(target),'--output',str(output)]
    result=subprocess.run(argv,env=env,cwd=target)
    with (continuation/'LINUX_DELIVERY_REPLAY_EXECUTION_A02.json').open('x',encoding='utf-8') as f:
        json.dump(dict(argv=argv,exit_code=result.returncode,platform='Real WSL Ubuntu Linux interpreter',
                       native_package=str(target)),f,indent=2)
    raise SystemExit(result.returncode)
elif mode=='collect':
    value=json.loads(context.read_text(encoding='utf-8'))
    target=Path(value['package'])
    source=target/'LINUX_REPLAY_FINAL_PORTABLE.xml'
    destination=continuation/'LINUX_REPLAY_FINAL_PORTABLE_A03.xml'
    if destination.exists():raise RuntimeError('COLLECT_WOULD_OVERWRITE')
    shutil.copyfile(source,destination)
    print(json.dumps(dict(xml=str(destination),bytes=destination.stat().st_size)))
elif mode=='sync_final_tests':
    value=json.loads(context.read_text(encoding='utf-8'))
    target=Path(value['package'])
    shutil.copyfile(continuation/'replay_tests.py',target/'replay_tests.py')
    print(json.dumps(dict(refreshed='replay_tests.py',scope='Portable test-fixture helper source only; raw evidence and replay core unchanged')))
else: raise RuntimeError('UNKNOWN_MODE')
