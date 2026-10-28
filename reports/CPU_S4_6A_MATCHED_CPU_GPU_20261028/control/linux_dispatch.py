"""Independent Linux interpreter, actual shared checkout, isolated Linux JIT cache."""
import json,os,subprocess,sys
from pathlib import Path

def native(value):
    return value.replace('\\','/').replace('C:/','/mnt/c/').replace('D:/','/mnt/d/')

request_path=Path(sys.argv[1]);request=json.loads(request_path.read_text(encoding='utf-8-sig'))
repo=Path(native(request['repo']))
runtime=Path('/mnt/d/Filament_1/.worktrees/cpu_optimization/reports/CPU_S4_3B_FIXTURE_REPAIR_20261020_090443/linux/venv/bin/python')
git_dir=Path(native((repo/'.git').read_text(encoding='utf-8').strip()[8:]))
if not runtime.is_file() or not git_dir.is_dir():raise RuntimeError('REAL_LINUX_RUNTIME_GIT_REQUIRED')
env=os.environ.copy()
for name,value in request['environment'].items():
    if value is not None:env[name]=native(value)
    else:env.pop(name,None)
stage=env.get('S46A_TEST_STAGE','backend')
report=Path(env['S46A_REPORT_ROOT'])
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
env.update(GIT_DIR=str(git_dir),GIT_WORK_TREE=str(repo),GIT_OPTIONAL_LOCKS='0',PYTHONPATH=str(repo/'Filament_python'),
           PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',NUMBA_CACHE_DIR=str(report/'cache/linux'/stage),PYTHONUTF8='1')
env.update(S43G2_SOURCE_ROOT=str(repo),S43F_SOURCE_ROOT=str(repo),S43C_SOURCE_ROOT=str(repo))
env.pop('PYTEST_ADDOPTS',None)
args=[native(arg) if len(arg)>2 and arg[1]==':' and arg[2] in '/\\' else arg for arg in request['argv']]
if args[:4]==['-s','-B','-m','pytest'] and not stage.startswith('sanity_'):
    targets=['test_micro.py'] if stage.startswith('micro') else ['test_records.py','test_gpu_manager.py']
    target_paths=[str(repo/'Filament_python/tools/cpu_s4_6a'/name) for name in targets]
    if stage.startswith('replay'):target_paths=[str(report/'replay_tests.py')]
    (report/'tmp').mkdir(exist_ok=True)
    args=['-s','-B','-m','pytest','-p','no:cacheprovider','-q','-x','--junitxml',str(report/('LINUX_'+stage.upper()+'.xml')),
          '--basetemp',str(report/'tmp'/('linux_'+stage)),*target_paths]
result=subprocess.run([str(runtime),*args],cwd=repo,env=env)
with request_path.with_suffix('.execution.json').open('x',encoding='utf-8') as f:
    json.dump(dict(argv=[str(runtime),*args],exit_code=result.returncode,cwd=str(repo),real_git_dir=str(git_dir),gpu_real_not_run=True),f,indent=2)
raise SystemExit(result.returncode)
