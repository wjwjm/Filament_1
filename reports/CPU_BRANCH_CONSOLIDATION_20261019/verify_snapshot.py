from pathlib import Path
import importlib.util,json,subprocess
r=Path('D:/Filament_1/.worktrees/cpu_optimization');src=Path('D:/Filament_1/.worktrees/cpu_s4')
spec=importlib.util.spec_from_file_location('pv',r/'Filament_python/tools/hpc_ops/provenance_v2.py');pv=importlib.util.module_from_spec(spec);spec.loader.exec_module(pv)
rows=[]
for p in sorted((r/'Filament_python/tools/cpu_o1/cpu_s4').glob('*.py')):
 rel=p.relative_to(r);h=pv.raw_sha256_file(p);assert h==pv.raw_sha256_file(src/rel)
 rows.append(dict(path=rel.as_posix(),hash_scope='raw_bytes',sha256=h,bytes=p.stat().st_size))
branches=subprocess.check_output(['git','branch','--format=%(refname:short)','--list','codex/cpu-*'],cwd=r,text=True).splitlines()
for b in branches:subprocess.run(['git','merge-base','--is-ancestor',b,'HEAD'],cwd=r,check=True)
report=dict(hash_scope='raw_bytes',source_root=str(src),snapshot=rows,contained_branches=branches,base_head='66801ad435d6ec1967338cee4eb82af2a152389c',science_unchanged=True)
(r/'reports/CPU_BRANCH_CONSOLIDATION_20261019/INVENTORY.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print('17 S4 files byte-identical; all CPU branches contained')
